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

#ifndef PATCHWORKPP__PATCHWORKPP_HPP_
#define PATCHWORKPP__PATCHWORKPP_HPP_

#include "patchworkpp/params.hpp"

#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <math.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

#include <numeric>
#include <utility>
#include <vector>

namespace patchwork_pp
{
using PointT = pcl::PointXYZI;
using Ring = std::vector<pcl::PointCloud<PointT>>;
using Zone = std::vector<Ring>;

class PatchWorkPP : public rclcpp::Node
{
public:
  /**
   * @brief Candidate of temporal ground revert.
   */
  struct TGRCandidate
  {
    const int zone_idx;
    const double flatness;
    const pcl::PointCloud<PointT> ground_cloud;

    explicit TGRCandidate(
      const int _zone_idx, const double _flatness, const pcl::PointCloud<PointT> & _ground_cloud)
    : zone_idx(_zone_idx), flatness(_flatness), ground_cloud(_ground_cloud)
    {
    }
  };

  explicit PatchWorkPP(const rclcpp::NodeOptions & options);

private:
  CommonParams common_params_;
  RNRParams rnr_params_;
  CZMParams czm_params_;
  RPFParams rpf_params_;
  GLEParams gle_params_;
  TGRParams tgr_params_;

  bool debug_;

  pcl::PointCloud<PointT>::Ptr in_cloud_, ground_cloud_, non_ground_cloud_;
  std::vector<Zone> czm_;

  Eigen::Matrix3d covariance_matrix_;
  Eigen::Vector4d centroid_;
  Eigen::Vector3d v_normal_;
  Eigen::Vector3d v_eigenvalues_;
  const Eigen::Vector3d u_normal_{Eigen::Vector3d(0, 0, 1)};

  /* Buffer of elevation and flatness for each n-th sector*/
  std::vector<std::vector<double>> elevation_buffer_;
  std::vector<std::vector<double>> flatness_buffer_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_non_ground_cloud_,
    pub_ground_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;

  /**
   * @brief Initialize debugger to publish ground cloud, if `debug=true`.
   *
   */
  void initializeDebugger()
  {
    pub_ground_cloud_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/debug/ground/pointcloud", 1);
  }

  Zone initializeZone(const int zone_idx)
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

  void refreshCZM()
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
    sensor_msgs::msg::PointCloud2 non_ground_cloud_msg;
    pcl::toROSMsg(*non_ground_cloud_, non_ground_cloud_msg);
    non_ground_cloud_msg.header = header;
    pub_ground_cloud_->publish(non_ground_cloud_msg);

    if (debug_) {
      sensor_msgs::msg::PointCloud2 ground_cloud_msg;
      pcl::toROSMsg(*ground_cloud_, ground_cloud_msg);
      ground_cloud_msg.header = header;
      pub_ground_cloud_->publish(ground_cloud_msg);
    }
  }

  // TODO(ktro2828): use a util function from other package
  /**
   * @brief Returns radius [m].
   *
   * @param point
   * @return double
   */
  static double calculateRadius(const PointT & point) { return std::hypot(point.x, point.y); }

  /**
   * @brief Returns yaw angle [deg].
   *
   * @param point
   * @return double
   */
  static double calculateYaw(const PointT & point)
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
  static std::pair<double, double> calculateMeanStd(const std::vector<double> & values)
  {
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    double std_dev = std::accumulate(
      values.begin(), values.end(), 0.0,
      [&](double acc, const double & v) { return acc + std::pow(v - mean, 2.0); });

    return {mean, std_dev};
  }

  /**
   * @brief Execute Reflected Noise Removal a.k.a RNR.
   *
   * @param in_cloud
   */
  void executeRNR(const pcl::PointCloud<PointT> & in_cloud) const;

  /**
   * @brief Sample initial seed points in each zone.
   * For the nearest zone from sensor (zone=0), skip points that have too low value that will be
   * used as threshold.
   *
   * @param zone_idx
   * @param in_cloud
   * @return pcl::PointCloud<Point>
   */
  pcl::PointCloud<PointT> sampleInitialSeed(
    const int zone_idx, pcl::PointCloud<PointT> & in_cloud) const;

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
  void executeRPF(
    const int zone_idx, pcl::PointCloud<PointT> & zone_cloud,
    pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief PCA-based vertical plane estimation a.k.a R-VPF.
   *
   * @param seed_cloud
   * @param non_vertical_cloud
   * @param non_ground_cloud
   */
  void estimateVerticalPlane(
    pcl::PointCloud<PointT> & seed_cloud, pcl::PointCloud<PointT> & non_vertical_cloud,
    pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief PCA-based ground plane estimation a.k.a R-GPF.
   *
   * @param seed_cloud
   * @param ground_cloud
   * @param non_ground_cloud
   */
  void estimateGroundPlane(
    pcl::PointCloud<PointT> & seed_cloud, pcl::PointCloud<PointT> & ground_cloud,
    pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief
   *
   * @param candidates
   * @param zone_flatness
   */
  void executeTGR(std::vector<TGRCandidate> & candidates, std::vector<double> & zone_flatness);

  /**
   * @brief Update elevation thresholds. In paper p5, e <- mean(E)  + a * std_dev(E).
   *
   */
  void updateElevationThresholds();

  /**
   * @brief Update flatness thresholds. In paper p5, f <- mean(F) + b * std_dev(F).
   *
   */
  void updateFlatnessThresholds();

  /**
   * @brief Update noise removal height threshold. In paper p5, h <- mean(E1) + margin.
   *
   */
  void updateHeightThreshold();

  /**
   * @brief Set input points to Concentric Zone Model a.k.a CZM.
   *
   * @param cloud
   */
  void pointCloud2CZM(
    const pcl::PointCloud<PointT> & cloud, pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief Callback. Estimate ground and non-ground points.
   *
   * @param cloud_msg
   */
  void cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg);
};  // class PatchWorkPP
}  // namespace patchwork_pp
#endif  // PATCHWORKPP__PATCHWORKPP_HPP_
