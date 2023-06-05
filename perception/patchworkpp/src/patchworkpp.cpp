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

#include "patchworkpp/patchworkpp.hpp"

#include <Eigen/Dense>

#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <numeric>

namespace patchwork_pp
{
PatchWorkPP::PatchWorkPP(const rclcpp::NodeOptions & options)
: rclcpp::Node("patchwork_pp", options),
  common_params_(*this),
  rnr_params_(*this),
  czm_params_(*this),
  rpf_params_(*this),
  gle_params_(*this),
  tgr_params_(*this)
{
  std::string input_topic = declare_parameter<std::string>("input");
  std::string output_topic = declare_parameter<std::string>("output");

  pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 1);
  sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic, 1, std::bind(&PatchWorkPP::cloudCallback, this, std::placeholders::_1));
}

void PatchWorkPP::cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg)
{
  pcl::PointCloud<PointT> in_cloud;
  pcl::fromROSMsg(*cloud_msg, in_cloud);

  // 1. RNR
  pcl::PointCloud<PointT> non_ground_cloud = executeRNR(in_cloud);
  // 2. CZM
  std::vector<Zone> czm = pointCloud2CZM(in_cloud, non_ground_cloud);

  for (int zone_idx = 0; zone_idx < czm_params_.num_zone(); ++i) {
    Zone zone = czm.at(zone_idx);
    for (int ring_idx = 0; ring_idx < czm_params_.num_rings(zone_idx); ++ring_idx) {
      for (int sector_idx = 0; sector_idx < czm_params_.num_sectors(zone_idx); ++sector_idx) {
        pcl::PointCloud<PointT> & zone_cloud = zone.at(ring_idx).at(sector_idx);
        if (zone_cloud.points.size() < czm_params_.min_num_points()) {
          non_ground_cloud.emplace_back(zone_cloud);
          continue;
        }

        // sort by z-value
        std::sort(
          zone_cloud.points.begin(), zone_cloud.points.end(),
          [](const PointT & pt1, const PointT & pt2) { return pt1.z < pt2.z; })
      }
    }
  }
}

pcl::PointCloud<PointT> PatchWorkPP::sampleSeeds(const pcl::PointCloud<PointT> & in_cloud) const
{
  pcl::PointCloud<PointT> seed_cloud;
  // sort by z-value
  std::sort(
    in_cloud.points.begin(), in_cloud.points.end(),
    [](const PointT & pt1, const PointT & pt2) { return pt1.z < pt2.z; });

  double sum_z = std::accumulate(
    in_cloud.points.begin(), in_cloud.points.end(), 0.0,
    [](double acc, const auto & pt) { return acc + pt.z; });

  double lowest_z = in_cloud.points.size() != 0 ? sum_z / in_cloud.points.size() : 0.0;

  for (const auto & point : in_cloud.points) {
    if (point.z < lowest_z) {
      seed_cloud.points.emplace_back(point);
    }
  }
  return seed_cloud;
}

pcl::PointCloud<PointT> PatchWorkPP::executeRNR(const pcl::PointCloud<PointT> & in_cloud) const
{
  pcl::PointCloud<PointT> non_ground_cloud;
  for (const auto & point : in_cloud) {
    double radius = calculateRadius(point);                            // [m]
    double incident_angle = std::atan2(point.z, radius) * 180 / M_PI;  // [deg]
    if (
      point.z < rnr_params_.min_height_threshold() &&
      incident_angle < rnr_params_.min_vertical_angle_threshold() &&
      point.intensity < rnr_params_.min_intensity_threshold()) {
      non_ground_cloud.emplace_back(point);
    }
  }
  return non_ground_cloud;
}

void PatchWorkPP::executeRPF(
  const int zone_idx, const pcl::PointCloud<PointT> & zone_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud) const
{
  // 1. R-VPF
  for (int n = 0; n < rpf_params_.num_iterator(); ++n) {
  }
  // 2. R-GPF
  for (int n = 0; n < rpf_params_.num_iterator(); ++n) {
  }
}

void PatchWorkPP::estimateVerticalPlane(
  const pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud)
{
  // Compute mean(m^k_n) and covariance(C)
  // TODO(ktro2828): Following variables can initialize in constructor
  Eigen::Matrix3f covariance_matrix;
  Eigen::Vector4f centroid;
  const Eigen::Vector3f u_z(0, 0, 1);

  pcl::computeMeanAndCovarianceMatrix(ground_cloud, covariance_matrix, centroid);
  // Compute eigenvalue(lambda1~3) and eigenvector(v1~3) for C(3x3)
  Eigen::EigenSolver<Eigen::Matrix3f> solver(covariance_matrix);
  // NOTE: The lowest eigenvector is v^k_3,n
  // TODO(ktro2828): sort by eigenvalues
  Eigen::Vector3cf v = solver.eigenvectors().col(2).normalized();
  for (const auto & point : ground_cloud) {
    // [Eq.2] d = |(p - m^k_n) * v^k_3,n|
    // [Eq.3] PI / 2 - cos^(-1) (v^k_3,n * u_z)
    Eigen::Vector3f p(point.x, point.y, point.z);
    auto distance = v.dot((p - centroid.head<3>()).cwiseAbs());
    auto angle = 0.5 * M_PI - std::acos(v.dot(u_z));
    // TODO(ktro2828): Define max_distance_threshold and max_vertical_distance_threshold explicitly
    if (
      (distance < rpf_params_.max_distance_threshold()) &&
      angle < rpf_params_.max_angle_threshold()) {
      non_ground_cloud.emplace_back(point);
    }
  }
}

void PatchWorkPP::estimateGroundPlane(
  const pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud)
{
  // Compute mean(m^k_n) and covariance(C)
  // TODO(ktro2828): Following variables can initialize in constructor
  Eigen::Matrix3f covariance_matrix;
  Eigen::Vector4f centroid;

  pcl::computeMeanAndCovarianceMatrix(ground_cloud, covariance_matrix, centroid);
  // Compute eigenvalue(lambda1~3) and eigenvector(v1~3) for C(3x3)
  Eigen::EigenSolver<Eigen::Matrix3f> solver(covariance_matrix);
  // NOTE: The lowest eigenvector is v^k_3,n
  // TODO(ktro2828): sort by eigenvalues
  Eigen::Vector3cf v = solver.eigenvectors().col(2).normalized();
  for (const auto & point : ground_cloud) {
    Eigen::Vector3f p(point.x, point.y, point.z);
    auto distance = v.dot((p - centroid.head<3>()).cwiseAbs());
    // TODO(ktro2828): Define max_distance_threshold and max_vertical_distance_threshold explicitly
    if (distance < rpf_params_.max_distance_threshold()) {
      non_ground_cloud.emplace_back(point);
    }
  }
}

void PatchWorkPP::updateElevationThresholds()
{
  for (int i = 0; i < czm_params_.num_zone(); ++i) {
    // TODO(ktro2828): set a container of elevation thresholds to stock thresholds while some length
    // time step
    // Use .at(i)
    if (TODO[i].empty()) {
      return;
    }
    const auto & [mean, std_dev] = calculateMeanStd(TODO[i]);
    double new_threshold = mean + gle_params_.elevation_std_weights(i) * std_dev;
    czm_params_.updateElevationThreshold(i, new_threshold);
  }
}

void PatchWorkPP::updateFlatnessThresholds()
{
  for (int i = 0; i < czm_params_.num_zone(); ++i) {
    // TODO(ktro2828): set a container of flatness thresholds to stock thresholds while some length
    // time step.
    // Use .at(i)
    if (TODO[i].empty()) {
      return;
    }
    const auto & [mean, std_dev] = calculateMeanStd(TODO[i]);
    double new_threshold = mean + gle_params_.flatness_std_weights(i) * std_dev;
    czm_params_.updateFlatnessThreshold(i, new_threshold);
  }
}

void PatchWorkPP::updateHeightThreshold()
{
  // TODO(ktro2828): set a container of elevation thresholds to stock thresholds while some length
  // time step
  // Use .at(0)
  if (TODO[0].empty()) {
    return;
  }
  const auto & [mean, std_dev] = calculateMeanStd(TODO[0]);
  double new_threshold = mean + gle_params_.height_noise_margin();
  rnr_params_.updateHeightThreshold(new_threshold);
}

std::vector<Zone> PatchWorkPP::pointCloud2CZM(
  const pcl::PointCloud<PointT> & cloud, pcl::PointCloud<PointT> & non_ground_cloud) const
{
  // NOTE: It might be better to initialize CZM and define it as member variable.
  std::vector<Zone> czm;
  for (const auto & point : cloud) {
    // TODO(ktro2828): use a util function from other package
    double radius = calculateRadius(point);

    if ((radius < common_params_.min_range()) || (common_params_.max_range() < radius)) {
      non_ground_cloud.emplace_back(point);
      continue;
    }

    // TODO(ktro2828): use a util function from other package
    double yaw = calculateYaw(point);

    for (size_t i = 1; i < czm_params_.num_zone(); ++i) {
      if (czm_params_.min_zone_ranges(i) <= radius) {
        continue;
      }
      double ring_size = czm_params_.ring_sizes(i);
      double sector_size = czm_params_.sector_sizes(i);

      int64_t ring_idx = std::min(
        static_cast<int64_t>((radius - czm_params_.min_zone_ranges(i - 1)) / ring_size),
        czm_params_.num_rings(i - 1) - 1);
      int64_t sector_idx =
        std::min(static_cast<int64_t>(yaw / sector_size), czm_params_.num_sectors(i - 1) - 1);

      czm[i - 1][ring_idx][sector_idx].points.emplace_back(point);
    }
  }
  return czm;
}
}  // namespace patchwork_pp
