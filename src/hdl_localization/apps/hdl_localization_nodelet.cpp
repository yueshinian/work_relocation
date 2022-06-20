#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>
//lym
#include <global_location.hpp>

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {
  }
  virtual ~HdlLocalizationNodelet() {
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    readParameters();//lym
    initialize();//lym
    initialize_params();

    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);
    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("/state_estimation", 5, false);//odom, integrated_to_init
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/registered_scan", 5, false);//aligned_points, velodyne_cloud_registered
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);
    localMapPub = nh.advertise<sensor_msgs::PointCloud2>("/localMap", 5, false);
    // imuPreSub = nh.subscribe<nav_msgs::Odometry>("/integrated_to_init2_incremental", 200, &HdlLocalizationNodelet::imuPreCb, this);//odomTopic+"_incremental"

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if(use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
      //hdl_global_client = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/relocalize");
    }

    // initialize pose estimator
    if(private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      geometry_msgs::Pose init_pose;
      init_pose.position.x = private_nh.param<double>("init_pos_x", 0.0);
      init_pose.position.y = private_nh.param<double>("init_pos_y", 0.0);
      init_pose.position.z = private_nh.param<double>("init_pos_z", 0.0);
      init_pose.orientation.w = private_nh.param<double>("init_ori_w", 1.0);
      init_pose.orientation.x = private_nh.param<double>("init_ori_x", 0.0);
      init_pose.orientation.y = private_nh.param<double>("init_ori_y", 0.0);
      init_pose.orientation.z = private_nh.param<double>("init_ori_z", 0.0);
      if(use_icp){
        std::cout<<"use icp, waiting..."<<std::endl;
        ros::Rate loop(100);
        while(globalmap==nullptr || pointScan==nullptr || pointScan->points.size()<=0 || globalmap->points.size()<=0){
          NODELET_INFO("waiting scan...");
          ros::spinOnce();
          loop.sleep();
        }
        pcl::PointCloud<PointT>::ConstPtr scan = last_scan;
        pcl::PointCloud<PointT>::ConstPtr map = globalmap;
        Eigen::Matrix4f initMatrix = g_location_ptr->pose2eigen(init_pose);
        auto icpResult = g_location_ptr->icpAligned(scan, map, initMatrix);
        Eigen::Matrix4f alignMatrix = icpResult.second;
        double score = icpResult.first;
        
        init_pose = g_location_ptr->eigen2pose(alignMatrix);
      }
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        ros::Time::now(),
        Eigen::Vector3f(init_pose.position.x, init_pose.position.y, init_pose.position.z),
        Eigen::Quaternionf(init_pose.orientation.w, init_pose.orientation.x, init_pose.orientation.y, init_pose.orientation.z),
        private_nh.param<double>("cool_time_duration", 0.5)
      ));
    }
  }

private:

  void imuPreCb(const nav_msgs::Odometry::ConstPtr &odometryMsg)
  {
     std::lock_guard<std::mutex> lock2(odomLock);
    odomQueue.push_back(*odometryMsg);
    if(odomQueue.size()>100) odomQueue.pop_front();
  }

  void initialize()
  {
    g_location_ptr.reset(new GLOBAL_LOCALION::globalLocation());
  }

  void readParameters()
  {
    use_icp = private_nh.param<bool>("use_icp", false);
    use_global = private_nh.param<bool>("use_global", false);
    once_init = private_nh.param<bool>("once_init", false);
    use_init = private_nh.param<bool>("use_init", false);
    use_icp_ndt = private_nh.param<bool>("use_icp_ndt", false);
    use_crop = private_nh.param<bool>("use_crop", false);
    use_imuPre = private_nh.param<bool>("use_imuPre", false);
  }

  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if(reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if(reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // // initialize pose estimator
    // if(private_nh.param<bool>("specify_init_pose", true)) {
    //   NODELET_INFO("initialize pose estimator with specified parameters!!");
    //   geometry_msgs::Pose init_pose;
    //   init_pose.pose.pose.position.x = private_nh.param<double>("init_pos_x", 0.0);
    //   init_pose.pose.pose.position.y = private_nh.param<double>("init_pos_y", 0.0);
    //   init_pose.pose.pose.position.z = private_nh.param<double>("init_pos_z", 0.0);
    //   init_pose.pose.pose.orientation.w = private_nh.param<double>("init_ori_w", 1.0);
    //   init_pose.pose.pose.orientation.x = private_nh.param<double>("init_ori_w", 0.0);
    //   init_pose.pose.pose.orientation.y = private_nh.param<double>("init_ori_w", 0.0);
    //   init_pose.pose.pose.orientation.z = private_nh.param<double>("init_ori_w", 0.0);
    //   if(use_icp){
    //     std::cout<<"use icp, waiting..."<<std::endl;
    //     ros::Rate loop(100);
    //     while(globalmap==nullptr || pointScan==nullptr || pointScan->points.size()<=0 || globalmap->points.size()<=0){
    //       NODELET_INFO("waiting scan...");
    //       ros::spinOnce();
    //       loop.sleep();
    //     }
    //     pcl::PointCloud<PointT>::ConstPtr scan = pointScan;
    //     pcl::PointCloud<PointT>::ConstPtr map = globalmap;
    //     Eigen::Matrix4f initMatrix = g_location_ptr->pose2eigen(init_pose)
    //     Eigen::Matrix4f alignMatrix = g_location_ptr->icpAligned(scan, map, initMatrix);
    //     init_pose = g_location_ptr->eigen2pose(alignMatrix);
    //   }
    //   pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
    //     ros::Time::now(),
    //     Eigen::Vector3f(init_pose.pose.pose.position.x, init_pose.pose.pose.position.y, init_pose.pose.pose.position.z),
    //     Eigen::Quaternionf(init_pose.pose.pose.orientation.w, init_pose.pose.pose.orientation.x, init_pose.pose.pose.orientation.y, init_pose.pose.pose.orientation.z),
    //     private_nh.param<double>("cool_time_duration", 0.5)
    //   ));
    // }
  }

private:
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    static int count = 0;
    if(count>0) return;
    if(once_init) ++count;
   
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);

    auto p = pose_msg->pose.pose.position;
    auto q = pose_msg->pose.pose.orientation;

    if(use_icp){
      std::cout<<"use icp, waiting..."<<std::endl;
      pcl::PointCloud<PointT>::ConstPtr scan = pointScan;
      pcl::PointCloud<PointT>::ConstPtr map = globalmap;
      Eigen::Matrix4f initMatrix = Eigen::Matrix4f::Identity();
      if(use_init){
        geometry_msgs::Pose initPose;
        initPose.position = pose_msg->pose.pose.position;
        initPose.orientation = pose_msg->pose.pose.orientation;
        initMatrix = g_location_ptr->pose2eigen(initPose);
      }
      auto icpResult = g_location_ptr->icpAligned(scan, map, initMatrix);
      Eigen::Matrix4f alignMatrix = icpResult.second;
      double score = icpResult.first;
      geometry_msgs::Pose pose = g_location_ptr->eigen2pose(alignMatrix);
      p = pose.position;
      q = pose.orientation;
    }

    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  void onlyIcp() 
  {
    // if(use_crop){
    //   pcl::PointCloud<PointT>::Ptr globalMap = globalmap;
    //   pcl::PointCloud<PointT>::Ptr cropMap(new pcl::PointCloud<PointT>());
    //   std::vector<double> square = g_location_ptr->getSquare(pointScan);
    //   g_location_ptr->setCropParam(square[0], square[1], square[2], square[3], square[4], square[5]);
    //   g_location_ptr->cropCloud(globalMap, cropMap);
    // }
    pcl::PointCloud<PointT>::ConstPtr scan = pointScan;
    pcl::PointCloud<PointT>::ConstPtr map = globalmap;
    auto icpResult = g_location_ptr->icpAligned(scan, map);
    Eigen::Matrix4f alignMatrix = icpResult.second;
    double score = icpResult.first;
    geometry_msgs::Pose pose = g_location_ptr->eigen2pose(alignMatrix);
    auto p = pose.position;
    auto q = pose.orientation;

    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  void icp_with_ndt() 
  {
    pcl::PointCloud<PointT>::ConstPtr scan = pointScan;
    pcl::PointCloud<PointT>::ConstPtr map = globalmap;
    auto icpResult = g_location_ptr->icpAligned(scan, map);
    Eigen::Matrix4f alignMatrix = icpResult.second;
    double score = icpResult.first;
    //alignMatrix.block<3,3>(0,0) = Eigen::Matrix3f::Identity();
    alignMatrix = g_location_ptr->ndtAligned(scan, map, alignMatrix);
    geometry_msgs::Pose pose = g_location_ptr->eigen2pose(alignMatrix);
    auto p = pose.position;
    auto q = pose.orientation;

    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }
  
  int point_count = 0;
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    if(point_count < 50){
      point_count++;
      return;
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);

    if(!globalmap || !is_globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if(pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }
    pointScan = pcl_cloud;
    // transform pointcloud into odom_child_frame_id
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    cloud = pcl_cloud;
    // if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
    //     NODELET_ERROR("point cloud cannot be transformed into target frame!!");
    //     return;
    // }

    auto filtered = downsample(cloud);
    last_scan = filtered;

    if(!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      if(private_nh.param<bool>("specify_init_pose", true)) return;
      if(use_global){
        static int locationCount = 1;
        static TicToc tictoc2;
        NODELET_INFO("use global init!!");
        ros::service::waitForService("/relocalize");
        TicToc tictoc;
        relocalize2();
        std::cout<<"fpfh cost time is: "<<tictoc.toc()<<" ms, ["<<locationCount<<"], all time is:"<<tictoc.toc()<<" ms"<<std::endl;
        locationCount++;
      }else if(use_icp_ndt){
        NODELET_INFO("use icp ndt init!!");
        icp_with_ndt();
        NODELET_INFO("use icp ndt end!!");
      }else if(use_icp && !use_init){
        NODELET_INFO("use icp init!!");
        onlyIcp();
        NODELET_INFO("use icp end!!");
      }
      return;
    }

    if(relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if(!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    ros::Time last_correction_time = pose_estimator->last_correction_time();
    if(private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;
      if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.1))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0));
      } else if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0));
      }

      if(odom_delta.header.stamp.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    // correct
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    bool imuPre = true;
    if (use_imuPre && imuPre && !odomQueue.empty()) {
      double timeScanCur = points_msg->header.stamp.toSec();
      while (!odomQueue.empty()) {
        if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
          odomQueue.pop_front();
        else
          break;
      }

      if (odomQueue.empty() || odomQueue.front().header.stamp.toSec() > timeScanCur) imuPre = false;
      if (imuPre) {
        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i) {
          startOdomMsg = odomQueue[i];
          if (startOdomMsg.header,stamp.toSec() < timeScanCur)
            continue;
          else
            break;
        }

        Eigen::Matrix4f initGuess = g_location_ptr->pose2eigen(startOdomMsg.pose.pose);
        aligned = pose_estimator->correct(stamp, filtered, initGuess);
      }
    }

    if(!imuPre || !use_imuPre){
      aligned = pose_estimator->correct(stamp, filtered);
    }
    // auto aligned = pose_estimator->correct(stamp, filtered);
    if(aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

    if(status_pub.getNumSubscribers()) {
      publish_scan_matching_status(points_msg->header, aligned);
    }

    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());
  }

  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }


  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);

    if(use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if(!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
    is_globalmap = true;
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if(last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if(!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      ros::Time::now(),
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  bool relocalize2() {
    if(last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    if(use_crop){
      NODELET_INFO("crop globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::PointCloud<PointT>::Ptr globalMap = globalmap;
      pcl::PointCloud<PointT>::Ptr cropMap(new pcl::PointCloud<PointT>());
      std::vector<double> square = g_location_ptr->getSquare(pointScan);
      g_location_ptr->setCropParam(square[0], square[1], square[2], square[3], square[4], square[5]);
      g_location_ptr->cropCloud(globalMap, cropMap);
      
      pcl::toROSMsg(*cropMap, srv.request.global_map);
      if(!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set crop map");
      } else {
        NODELET_INFO("succeed to set crop map");
      }

      sensor_msgs::PointCloud2 mapPoints;
      pcl::toROSMsg(*cropMap, mapPoints);
      mapPoints.header.frame_id = "/map";
      localMapPub.publish(mapPoints);
    }

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;
    TicToc tictoc;
    if(!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }
    std::cout<<"global location cost time is: "<<tictoc.toc()<<" ms"<<std::endl;

    auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    bool checkError = true;
    if(checkError){
      pcl::PointCloud<pcl::PointXYZI>::Ptr transCloud(new pcl::PointCloud<pcl::PointXYZI>());
      geometry_msgs::Pose pose;
      pose.position = result.position;
      pose.orientation = result.orientation;
      Eigen::Matrix4f transMatrix = g_location_ptr->pose2eigen(pose);
      pcl::transformPointCloud(*scan, *transCloud, transMatrix);
      std::vector<double> square = g_location_ptr->getSquare(transCloud);

      pcl::PointCloud<PointT>::Ptr globalMap = globalmap;
      pcl::PointCloud<PointT>::Ptr cropMap(new pcl::PointCloud<PointT>());
      g_location_ptr->setCropParam(square[0], square[1], square[2], square[3], square[4], square[5]);
      g_location_ptr->cropCloud(globalMap, cropMap);
      sensor_msgs::PointCloud2 mapPoints;
      pcl::toROSMsg(*cropMap, mapPoints);
      mapPoints.header.frame_id = "/map";
      localMapPub.publish(mapPoints);

      auto icpResult = g_location_ptr->icpAligned(transCloud, cropMap);
      if(icpResult.first>0.10){
        std::cout<<"fpfh init failed!"<<std::endl;
        return false;
      }else{
        pose = g_location_ptr->eigen2pose(icpResult.second);
        //result.position = pose.position;
        //result.orientation = pose.orientation;
      }
    }

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    //std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      ros::Time::now(),
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    if(tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = "map";

      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(0.1));
      Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = robot_odom_frame_id;

      tf_broadcaster.sendTransform(odom_trans);
    } else {
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = odom_child_frame_id;
      tf_broadcaster.sendTransform(odom_trans);
    }

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub.publish(odom);
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if(pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;
  //lym
  bool use_icp, use_global, once_init, use_init,use_icp_ndt,use_crop,use_imuPre;
  std::unique_ptr<GLOBAL_LOCALION::globalLocation> g_location_ptr;
  ros::ServiceClient hdl_global_client;
  pcl::PointCloud<PointT>::ConstPtr pointScan;
  bool is_globalmap = false;
  ros::Publisher localMapPub;
  ros::Subscriber imuPreSub;
  std::deque<nav_msgs::Odometry> odomQueue;
  std::mutex odomLock;
};
}


PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
