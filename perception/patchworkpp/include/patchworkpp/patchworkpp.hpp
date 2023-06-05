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

#include <math.h>
#include <pcl/common/common.h>

#include <numeric>
#include <utility>
#include <vector>

namespace patchwork_pp
{
class PatchWorkPP : public rclcpp::Node
{
public:
  using PointT = pcl::PointXYZI;
  using Ring = std::vector<pcl::PointCloud<PointT>>;
  using Zone = std::vector<Ring>;

  explicit PatchWorkPP(const rclcpp::NodeOptions & options);

private:
  CommonParams common_params_;
  RNRParams rnr_params_;
  CZMParams czm_params_;
  RPFParams rpf_params_;
  GLEParams gle_params_;
  TGRParams tgr_params_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;

  struct TGRCandidate
  {
    const double flatness;
    const pcl::PointCloud<PointT> ground_cloud;

    TGRCandidate(const double _flatness, const pcl::PointCloud<PointT> & _ground_cloud)
    : flatness(_flatness), ground_cloud(_ground_cloud)
    {
    }
  };

  // TODO(ktro2828): use a util function from other package
  /**
   * @brief Returns radius [m].
   *
   * @param point
   * @return double
   */
  static double calculateRadius(const pcl::PointCloud<PointT> & point)
  {
    return std::hypot(point.x, point.y);
  }

  /**
   * @brief Returns yaw angle [deg].
   *
   * @param point
   * @return double
   */
  static double calculateYaw(const pcl::PointCloud<PointT> & point)
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
    // if (values.size() <= 1) {
    //   return {0.0, 0.0};
    // }

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
   * @return pcl::PointCloud<PointT>
   */
  pcl::PointCloud<PointT> executeRNR(const pcl::PointCloud<PointT> & in_cloud) const;

  /**
   * @brief Execute Region-wise Plane Fitting.
   * 1. R-VPF (Region-wise Vertical Plane Fitting).
   * 2. R-GPF (Region-wise Ground Plane Fitting).
   *
   * @param zone_idx
   * @param zone_cloud
   * @param non_ground_cloud
   */
  void executeRPF(
    const int zone_idx, const pcl::PointCloud<PointT> & zone_cloud,
    pcl::PointCloud<PointT> & non_ground_cloud) const;

  /**
   * @brief PCA-based vertical plane estimation a.k.a R-VPF.
   *
   * @param ground_cloud
   * @param non_ground_cloud
   */
  void estimateVerticalPlane(
    const pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief PCA-based ground plane estimation a.k.a R-GPF.
   *
   * @param ground_cloud
   * @param non_ground_cloud
   */
  void estimateGroundPlane(
    const pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud);

  /**
   * @brief Sample seed points.
   *
   * @param in_cloud
   * @return pcl::PointCloud<Point>
   */
  pcl::PointCloud<Point> sampleSeeds(const pcl::PointCloud<PointT> & in_cloud) const;

  void executeTGR();

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
   *
   * @return std::vector<Zone>
   */
  std::vector<Zone> pointCloud2CZM(
    const pcl::PointCloud<PointT> & cloud, pcl::PointCloud<PointT> & non_ground_cloud) const;

  /**
   * @brief Callback. Estimate ground and non-ground points.
   *
   * @param cloud_msg
   */
  void cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg);
};  // class PatchWorkPP
}  // namespace patchwork_pp
#endif  // PATCHWORKPP__PATCHWORKPP_HPP_
