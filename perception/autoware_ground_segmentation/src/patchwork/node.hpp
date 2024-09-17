// Copyright 2023 TIER IV, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PATCHWORK__NODE_HPP_
#define PATCHWORK__NODE_HPP_

#include "params.hpp"

#include <autoware/universe_utils/ros/debug_publisher.hpp>
#include <autoware/universe_utils/system/stop_watch.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <math.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

#include <chrono>
#include <memory>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace autoware::ground_segmentation
{
using PointT = pcl::PointXYZI;
using Ring = std::vector<pcl::PointCloud<PointT>>;
using Zone = std::vector<Ring>;
using StopWatch = autoware::universe_utils::StopWatch<std::chrono::milliseconds>;
using DebugPublisher = autoware::universe_utils::DebugPublisher;

class PatchWorkComponent : public rclcpp::Node
{
public:
  /**
   * @brief Candidate of temporal ground revert.
   */
  struct TGRCandidate
  {
    const size_t zone_idx;
    const pcl::PointCloud<PointT> ground_cloud;
    const double flatness;

    explicit TGRCandidate(
      const size_t _zone_idx, const pcl::PointCloud<PointT> & _ground_cloud, const double _flatness)
    : zone_idx(_zone_idx), ground_cloud(_ground_cloud), flatness(_flatness)
    {
    }
  };

  explicit PatchWorkComponent(const rclcpp::NodeOptions & options);

private:
  CommonParams common_params_;
  RNRParams rnr_params_;
  CZMParams czm_params_;
  RPFParams rpf_params_;
  GLEParams gle_params_;
  TGRParams tgr_params_;

  size_t max_queue_size_;
  bool debug_;

  pcl::PointCloud<PointT>::Ptr in_cloud_, ground_cloud_, non_ground_cloud_,
    sector_non_ground_cloud_, sector_ground_cloud_;
  std::vector<Zone> czm_;

  Eigen::Matrix3d covariance_matrix_;
  Eigen::Vector4d centroid_;
  Eigen::Vector3d v_normal_;
  Eigen::Vector3d v_eigenvalues_;
  const Eigen::Vector3d u_normal_{Eigen::Vector3d(0, 0, 1)};
  double d_k_;

  /* Buffer of elevation and flatness for each n-th sector*/
  std::vector<std::vector<double>> elevation_buffer_;
  std::vector<std::vector<double>> flatness_buffer_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_non_ground_cloud_,
    pub_ground_cloud_{nullptr};
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;

  std::unique_ptr<DebugPublisher> debug_publisher_ptr_{nullptr};
  std::unique_ptr<StopWatch> stop_watch_ptr_{nullptr};

  /**
   * @brief Initialize debugger to publish ground cloud, if `debug=true`.
   *
   */
  void initialize_debugger()
  {
    pub_ground_cloud_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/debug/ground/pointcloud", 1);

    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, "patchwork_pp");
    stop_watch_ptr_ = std::make_unique<StopWatch>();
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  Zone initialize_zone(const size_t zone_idx)
  {
    Zone zone;
    Ring ring;
    pcl::PointCloud<PointT> cloud;
    for (int sector_idx = 0; sector_idx < czm_params_.num_sectors(zone_idx); ++sector_idx) {
      ring.emplace_back(cloud);
    }
    for (int ring_idx = 0; ring_idx < czm_params_.num_rings(zone_idx); ++ring_idx) {
      zone.emplace_back(ring);
    }
    return zone;
  }

  void refresh_czm()
  {
    for (auto & zone : czm_) {
      for (auto & ring : zone) {
        for (auto & sector : ring) {
          sector.points.clear();
        }
      }
    }
  }

  /**
   * @brief Publish non-ground cloud, and also publish ground cloud, if `debug=true`.
   *
   * @param header
   */
  void publish(const std_msgs::msg::Header & header) const
  {
    non_ground_cloud_->width = non_ground_cloud_->points.size();
    non_ground_cloud_->height = 1;

    sensor_msgs::msg::PointCloud2 non_ground_cloud_msg;
    pcl::toROSMsg(*non_ground_cloud_, non_ground_cloud_msg);
    non_ground_cloud_msg.header = header;
    pub_non_ground_cloud_->publish(non_ground_cloud_msg);

    if (debug_) {
      ground_cloud_->width = ground_cloud_->points.size();
      ground_cloud_->height = 1;

      sensor_msgs::msg::PointCloud2 ground_cloud_msg;
      pcl::toROSMsg(*ground_cloud_, ground_cloud_msg);
      ground_cloud_msg.header = header;
      pub_ground_cloud_->publish(ground_cloud_msg);

      const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
      const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
      debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
        "debug/cyclic_time_ms", cyclic_time_ms);
      debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
        "debug/processing_time_ms", processing_time_ms);
    }
  }

  // TODO(ktro2828): use a util function from other package
  /**
   * @brief Returns radius [m].
   *
   * @param point
   * @return double
   */
  static double calculate_radius(const PointT & point) { return std::hypot(point.x, point.y); }

  /**
   * @brief Returns yaw angle [deg].
   *
   * @param point
   * @return double
   */
  static double calculate_yaw(const PointT & point)
  {
    double yaw = std::atan2(point.y, point.x);
    return 0 < yaw ? yaw : yaw + 2 * M_PI;
  }

  /**
   * @brief Calculate mean and standard deviation values.
   *
   * @param values
   * @return std::pair<double, double>
   */
  static std::pair<double, double> calculate_mean_stddev(const std::vector<double> & values)
  {
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    double std_dev =
      std::accumulate(
        values.begin(), values.end(), 0.0,
        [&](double acc, const double & v) { return acc + std::pow(v - mean, 2.0); }) /
      values.size();

    std_dev = std::sqrt(std_dev);

    return {mean, std_dev};
  }

  /**
   * @brief Insert pointcloud into other pointcloud.
   *
   * @param in_cloud
   * @param out_cloud
   */
  void insert_cloud(const pcl::PointCloud<PointT> & in_cloud, pcl::PointCloud<PointT> & out_cloud)
  {
    out_cloud.insert(out_cloud.end(), in_cloud.cbegin(), in_cloud.cend());
  }

  /**
   * @brief Execute Reflected Noise Removal a.k.a RNR.
   *
   * @param in_cloud
   * @return std::queue<size_t>
   */
  std::queue<size_t> remove_reflected_noise(const pcl::PointCloud<PointT> & in_cloud) const;

  /**
   * @brief Sample initial seed points in each zone.
   * For the nearest zone from sensor (zone=0), skip points that have too low value that will be
   * used as threshold.
   *
   * @param zone_idx
   * @param in_cloud
   * @param seed_cloud
   * @param seed_margin
   */
  void sample_initial_seed(
    const size_t zone_idx, const pcl::PointCloud<PointT> & in_cloud,
    pcl::PointCloud<PointT> & seed_cloud, const double seed_margin) const;

  /**
   * @brief Execute Region-wise Plane Fitting.
   * 1. R-VPF (Region-wise Vertical Plane Fitting).
   * 2. R-GPF (Region-wise Ground Plane Fitting).
   *
   * @param zone_idx
   * @param zone_cloud
   * @param ground_cloud
   * @param non_ground_cloud
   */
  void region_wise_plane_fitting(
    const size_t zone_idx, pcl::PointCloud<PointT> & zone_cloud,
    pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief PCA-based vertical plane estimation a.k.a R-VPF.
   *
   * @param in_cloud
   * @param non_vertical_cloud
   * @param ground_cloud
   * @param non_ground_cloud
   */
  void estimate_vertical_plane(
    const size_t zone_idx, pcl::PointCloud<PointT> & in_cloud,
    pcl::PointCloud<PointT> & non_vertical_cloud, pcl::PointCloud<PointT> & ground_cloud,
    pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief PCA-based ground plane estimation a.k.a R-GPF.
   *
   * @param in_cloud
   * @param ground_cloud
   * @param non_ground_cloud
   */
  void estimate_ground_plane(
    const size_t zone_idx, pcl::PointCloud<PointT> & in_cloud,
    pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief Estimate ground plane with calculating eigenvector and eigenvalues.
   *
   * @param ground_cloud
   */
  void estimate_plane(const pcl::PointCloud<PointT> & ground_cloud);

  /**
   * @brief
   *
   * @param candidates
   * @param ring_flatness
   */
  void temporal_ground_revert(
    const std::vector<TGRCandidate> & candidates, const std::vector<double> & ring_flatness);

  /**
   * @brief Update elevation thresholds. In paper p5, e <- mean(E)  + a * std_dev(E).
   *
   */
  void update_elevation_thresholds();

  /**
   * @brief Update flatness thresholds. In paper p5, f <- mean(F) + b * std_dev(F).
   *
   */
  void update_flatness_thresholds();

  /**
   * @brief Update noise removal height threshold. In paper p5, h <- mean(E1) + margin.
   *
   */
  void update_height_threshold();

  /**
   * @brief Set input points to Concentric Zone Model a.k.a CZM.
   *
   * @param in_cloud
   * @param noise_indices
   */
  void cloud_to_czm(const pcl::PointCloud<PointT> & in_cloud, std::queue<size_t> & noise_indices);

  /**
   * @brief Callback. Estimate ground and non-ground points.
   *
   * @param cloud_msg
   */
  void cloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg);
};  // class PatchWorkPP

}  // namespace autoware::ground_segmentation

#endif  // PATCHWORK__NODE_HPP_
