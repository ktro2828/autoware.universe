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
#include <pcl/common/io.h>
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
  debug_ = declare_parameter<bool>("debug", false);

  elevation_list_.resize(czm_params_.num_zone());
  flatness_list_.resize(czm_params_.num_zone());

  pub_non_ground_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 1);
  sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic, 1, std::bind(&PatchWorkPP::cloudCallback, this, std::placeholders::_1));

  if (debug_) {
    initializeDebugger();
  }
}

void PatchWorkPP::cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg)
{
  in_cloud_->clear();
  ground_cloud_->clear();
  non_ground_cloud_->clear();
  czm_.clear();

  pcl::fromROSMsg(*cloud_msg, *in_cloud_);

  // 1. RNR
  executeRNR(*in_cloud_);
  // 2. CZM
  pointCloud2CZM(*in_cloud_, *non_ground_cloud_);

  for (int zone_idx = 0; zone_idx < czm_params_.num_zone(); ++zone_idx) {
    Zone zone = czm_.at(zone_idx);
    for (int ring_idx = 0; ring_idx < czm_params_.num_rings(zone_idx); ++ring_idx) {
      for (int sector_idx = 0; sector_idx < czm_params_.num_sectors(zone_idx); ++sector_idx) {
        pcl::PointCloud<PointT> & zone_cloud = zone.at(ring_idx).at(sector_idx);
        if (static_cast<int>(zone_cloud.points.size()) < czm_params_.min_num_points()) {
          std::for_each(
            zone_cloud.points.begin(), zone_cloud.points.end(),
            [&](const auto & point) { non_ground_cloud_->emplace_back(point); });
          continue;
        }

        // sort by z-value
        std::sort(
          zone_cloud.points.begin(), zone_cloud.points.end(),
          [](const PointT & pt1, const PointT & pt2) { return pt1.z < pt2.z; });

        // 3. RPF
        executeRPF(zone_idx, zone_cloud, *ground_cloud_, *non_ground_cloud_);

        // 4. A-GLE
        // TODO(ktro2828): update A-GLE and TGR
        const double uprightness = v_normal_(2, 0);
        const double elevation = centroid_(2, 0);
        const double flatness = v_eigenvalues_(0);

        const int concentric_idx = zone_idx + ring_idx + sector_idx;
        const bool is_upright = gle_params_.uprightness_threshold() < uprightness;
        const bool is_elevated = elevation < czm_params_.elevation_thresholds(concentric_idx);
        const bool is_flat = flatness < czm_params_.flatness_thresholds(concentric_idx);

        if (!is_upright) {
          non_ground_cloud_->insert(
            non_ground_cloud_->end(), ground_cloud_->begin(), ground_cloud_->end());
          continue;
        }
      }
    }
  }
}

void PatchWorkPP::executeRNR(const pcl::PointCloud<PointT> & in_cloud) const
{
  for (const auto & point : in_cloud) {
    double radius = calculateRadius(point);                            // [m]
    double incident_angle = std::atan2(point.z, radius) * 180 / M_PI;  // [deg]
    if (
      point.z < rnr_params_.min_height_threshold() &&
      incident_angle < rnr_params_.min_vertical_angle_threshold() &&
      point.intensity < rnr_params_.min_intensity_threshold()) {
      non_ground_cloud_->emplace_back(point);
    }
  }
}

pcl::PointCloud<PointT> PatchWorkPP::sampleInitialSeed(
  const int zone_idx, pcl::PointCloud<PointT> & in_cloud) const
{
  pcl::PointCloud<PointT> seed_cloud;
  // sort by z-value
  std::sort(
    in_cloud.points.begin(), in_cloud.points.end(),
    [](const PointT & pt1, const PointT & pt2) { return pt1.z < pt2.z; });

  int init_idx = 0;
  if (zone_idx == 0) {
    double lowest_z_in_close_zone =
      common_params_.sensor_height() == 0 ? -0.1 : common_params_.lowest_z_in_close_zone();
    for (const auto & point : in_cloud.points) {
      if (point.z < lowest_z_in_close_zone) {
        ++init_idx;
      }
    }
  }

  size_t num_sample = std::max(
    std::min(static_cast<int>(in_cloud.points.size()), rpf_params_.num_sample()), init_idx);

  double sum_z = std::accumulate(
    in_cloud.points.begin() + init_idx, in_cloud.points.begin() + num_sample, 0.0,
    [](double acc, const auto & pt) { return acc + pt.z; });

  double lowest_z = num_sample != 0 ? sum_z / in_cloud.points.size() : 0.0;

  for (const auto & point : in_cloud.points) {
    if (point.z < lowest_z) {
      seed_cloud.points.emplace_back(point);
    }
  }
  return seed_cloud;
}

void PatchWorkPP::executeRPF(
  const int zone_idx, pcl::PointCloud<PointT> & zone_cloud, pcl::PointCloud<PointT> & ground_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud)
{
  // 1. R-VPF ... Extract vertical cloud as non-ground cloud from seed cloud
  auto seed_cloud1 = sampleInitialSeed(zone_idx, zone_cloud);
  pcl::PointCloud<PointT> non_vertical_cloud;
  estimateVerticalPlane(seed_cloud1, non_vertical_cloud, non_ground_cloud);

  // 2. R-GPF ... Extract non-ground cloud from non-vertical cloud
  auto seed_cloud2 = sampleInitialSeed(zone_idx, non_vertical_cloud);
  estimateGroundPlane(seed_cloud2, ground_cloud, non_ground_cloud);
}

void PatchWorkPP::estimateVerticalPlane(
  pcl::PointCloud<PointT> & seed_cloud, pcl::PointCloud<PointT> & non_vertical_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud)
{
  // Compute mean(m^k_n) and covariance(C)
  // TODO(ktro2828): Following variables can initialize in constructor
  const Eigen::Vector3d u_z(0, 0, 1);

  for (int n = 0; n < rpf_params_.num_iterator(); ++n) {
    non_vertical_cloud.clear();
    pcl::computeMeanAndCovarianceMatrix(seed_cloud, covariance_matrix_, centroid_);
    // Compute eigenvalue(lambda1~3) and eigenvector(v1~3) for C(3x3)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance_matrix_);
    // NOTE: The lowest eigenvector is v^k_3,n
    v_normal_ = solver.eigenvectors().col(2).normalized();
    v_eigenvalues_ = solver.eigenvalues();
    for (const auto & point : seed_cloud) {
      // [Eq.2] d = |(p - m^k_n) * v^k_3,n|
      // [Eq.3] PI / 2 - cos^(-1) (v^k_3,n * u_z)
      Eigen::Vector3d p(point.x, point.y, point.z);
      auto distance = v_normal_.dot((p - centroid_.head<3>()).cwiseAbs());
      auto angle = 0.5 * M_PI - std::acos(v_normal_.dot(u_z));
      if (
        (distance < rpf_params_.max_vertical_distance_threshold()) &&
        angle < rpf_params_.max_angle_threshold()) {
        non_ground_cloud.emplace_back(point);
      } else {
        non_vertical_cloud.emplace_back(point);
      }
    }
    pcl::copyPointCloud(non_vertical_cloud, seed_cloud);
  }
}

void PatchWorkPP::estimateGroundPlane(
  pcl::PointCloud<PointT> & seed_cloud, pcl::PointCloud<PointT> & ground_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud)
{
  // Compute mean(m^k_n) and covariance(C)
  for (int n = 0; n < rpf_params_.num_iterator(); ++n) {
    ground_cloud.clear();
    pcl::computeMeanAndCovarianceMatrix(seed_cloud, covariance_matrix_, centroid_);
    // Compute eigenvalue(lambda1~3) and eigenvector(v1~3) for C(3x3)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance_matrix_);
    // The lowest eigenvector is v^k_3,n
    v_normal_ = solver.eigenvectors().col(0).normalized();
    v_eigenvalues_ = solver.eigenvalues();
    for (const auto & point : seed_cloud) {
      Eigen::Vector3d p(point.x, point.y, point.z);
      auto distance = v_normal_.dot((p - centroid_.head<3>()).cwiseAbs());
      if (distance < rpf_params_.max_distance_threshold()) {
        non_ground_cloud.emplace_back(point);
      } else {
        ground_cloud.emplace_back(point);
      }
    }
    pcl::copyPointCloud(ground_cloud, seed_cloud);
  }
}

void PatchWorkPP::executeTGR(
  std::vector<double> & ring_flatness, std::vector<TGRCandidate> & candidates)
{
  // TODO(ktro2828): update TGR
  const auto & [mean_flatness, std_flatness] = calculateMeanStd();
  const double flatness_t = mean_flatness + tgr_params_.std_weight() * std_flatness;
  for (auto & candidate : candidates) {
    // PatchWork p.5
    // P_e = r_n < L_t ? 1 / (1 + exp^(e_t - z_n) : 1 (10)
    // P_f = P_e < 0.5 ? zeta * exp^(f_t - f_n) : 1   (11)
    const double prob = std::exp(flatness_t - candidate.flatness);
  }
}

void PatchWorkPP::updateElevationThresholds()
{
  for (int i = 0; i < czm_params_.num_zone(); ++i) {
    if (elevation_list_.at(i).empty()) {
      return;
    }
    const auto & [mean, std_dev] = calculateMeanStd(elevation_list_.at(i));
    double new_threshold = mean + gle_params_.elevation_std_weights(i) * std_dev;
    czm_params_.updateElevationThreshold(i, new_threshold);
  }
}

void PatchWorkPP::updateFlatnessThresholds()
{
  for (int i = 0; i < czm_params_.num_zone(); ++i) {
    if (flatness_list_.at(i).empty()) {
      return;
    }
    const auto & [mean, std_dev] = calculateMeanStd(flatness_list_.at(i));
    double new_threshold = mean + gle_params_.flatness_std_weights(i) * std_dev;
    czm_params_.updateFlatnessThreshold(i, new_threshold);
  }
}

void PatchWorkPP::updateHeightThreshold()
{
  if (elevation_list_.at(0).empty()) {
    return;
  }
  const auto & [mean, std_dev] = calculateMeanStd(elevation_list_.at(0));
  double new_threshold = mean + gle_params_.height_noise_margin();
  rnr_params_.updateHeightThreshold(new_threshold);
}

void PatchWorkPP::pointCloud2CZM(
  const pcl::PointCloud<PointT> & cloud, pcl::PointCloud<PointT> & non_ground_cloud)
{
  for (const auto & point : cloud) {
    // TODO(ktro2828): use a util function from other package
    double radius = calculateRadius(point);

    if ((radius < common_params_.min_range()) || (common_params_.max_range() < radius)) {
      non_ground_cloud.emplace_back(point);
      continue;
    }

    // TODO(ktro2828): use a util function from other package
    double yaw = calculateYaw(point);

    for (int i = 1; i < czm_params_.num_zone(); ++i) {
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

      czm_[i - 1][ring_idx][sector_idx].points.emplace_back(point);
    }
  }
}
}  // namespace patchwork_pp
