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

// for debug
#include <chrono>
#include <ostream>

namespace
{
template <typename T, typename U>
std::ostream & operator<<(std::basic_ostream<U> & os, const std::vector<T> & values)
{
  os << "(";
  for (const T & v : values) {
    os << v << ", ";
  }
  os << ")";
  return os;
}
}  // namespace

namespace patchwork_pp
{
PatchWorkPP::PatchWorkPP(const rclcpp::NodeOptions & options)
: rclcpp::Node("patchwork_pp", options),
  common_params_(this),
  rnr_params_(this),
  czm_params_(this),
  rpf_params_(this),
  gle_params_(this),
  tgr_params_(this)
{
  debug_ = declare_parameter<bool>("debug");

  in_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  non_ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  sector_non_ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  sector_ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();

  for (int zone_idx = 0; zone_idx < czm_params_.num_zone(); ++zone_idx) {
    Zone zone = initialize_zone(zone_idx);
    czm_.emplace_back(zone);
  }

  elevation_buffer_.resize(czm_params_.num_near_ring());
  flatness_buffer_.resize(czm_params_.num_near_ring());

  sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "input/pointcloud", 10, std::bind(&PatchWorkPP::cloud_callback, this, std::placeholders::_1));
  pub_non_ground_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("output/pointcloud", 1);

  if (debug_) {
    initialize_debugger();
  }
}

void PatchWorkPP::cloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg)
{
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  in_cloud_->clear();
  ground_cloud_->clear();
  non_ground_cloud_->clear();
  refresh_czm();

  pcl::fromROSMsg(*cloud_msg, *in_cloud_);

  // 1. RNR
  remove_reflected_noise(*in_cloud_);
  // 2. CZM
  cloud_to_czm(*in_cloud_, *non_ground_cloud_);

  std::vector<TGRCandidate> candidates;
  std::vector<double> zone_flatness;
  int concentric_idx = 0;
  for (int zone_idx = 0; zone_idx < czm_params_.num_zone(); ++zone_idx) {
    Zone & zone = czm_.at(zone_idx);
    for (int ring_idx = 0; ring_idx < czm_params_.num_rings(zone_idx); ++ring_idx) {
      for (int sector_idx = 0; sector_idx < czm_params_.num_sectors(zone_idx); ++sector_idx) {
        pcl::PointCloud<PointT> & zone_cloud = zone.at(ring_idx).at(sector_idx);

        if (zone_cloud.points.empty()) {
          continue;
        }

        if (static_cast<int>(zone_cloud.points.size()) < czm_params_.min_num_point()) {
          for (const auto & point : zone_cloud) {
            non_ground_cloud_->points.emplace_back(point);
          }
          continue;
        }

        // sort by z-value
        std::sort(
          zone_cloud.points.begin(), zone_cloud.points.end(),
          [](const PointT & pt1, const PointT & pt2) { return pt1.z < pt2.z; });

        sector_non_ground_cloud_->clear();
        sector_ground_cloud_->clear();

        // 3. RPF
        region_wise_plane_fitting(
          zone_idx, zone_cloud, *sector_ground_cloud_, *sector_non_ground_cloud_);

        // 4. A-GLE
        const double uprightness = v_normal_(2, 0);
        const double elevation = centroid_(2, 0);
        const double flatness = v_eigenvalues_(0);

        const bool is_near_ring = concentric_idx < czm_params_.num_near_ring();
        const bool is_upright = gle_params_.uprightness_threshold() < uprightness;
        const bool is_not_elevated =
          is_near_ring ? elevation < czm_params_.elevation_thresholds(zone_idx) : false;
        const bool is_flat =
          is_near_ring ? flatness < czm_params_.flatness_thresholds(zone_idx) : false;

        if (is_upright && is_not_elevated && is_near_ring) {
          elevation_buffer_.at(concentric_idx).emplace_back(elevation);
          flatness_buffer_.at(concentric_idx).emplace_back(flatness);
          zone_flatness.emplace_back(flatness);
        }

        if (!is_upright) {
          insert_cloud(*sector_ground_cloud_, *non_ground_cloud_);
        } else if (!is_near_ring || is_not_elevated || is_flat) {
          insert_cloud(*sector_ground_cloud_, *ground_cloud_);
        } else {
          TGRCandidate candidate(zone_idx, flatness, *sector_ground_cloud_);
          candidates.emplace_back(candidate);
        }
        insert_cloud(*sector_non_ground_cloud_, *non_ground_cloud_);
      }

      if (!candidates.empty()) {
        temporal_ground_revert(candidates, zone_flatness);
        candidates.clear();
        zone_flatness.clear();
      }
      ++concentric_idx;
    }
  }

  update_elevation_thresholds();
  update_flatness_thresholds();
  update_height_threshold();

  RCLCPP_INFO_STREAM(get_logger(), "elevation thresholds: " << czm_params_.elevation_thresholds());
  RCLCPP_INFO_STREAM(get_logger(), "flatness thresholds: " << czm_params_.flatness_thresholds());

  end = std::chrono::system_clock::now();
  double elapsed =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3;
  RCLCPP_INFO_STREAM(get_logger(), "Processing time: " << elapsed);

  publish(cloud_msg->header);
}

void PatchWorkPP::remove_reflected_noise(const pcl::PointCloud<PointT> & in_cloud) const
{
  for (const auto & point : in_cloud) {
    const double radius = calculate_radius(point);                           // [m]
    const double incident_angle = std::atan2(point.z, radius) * 180 / M_PI;  // [deg]
    if (
      point.z < rnr_params_.min_height_threshold() &&
      incident_angle < rnr_params_.min_vertical_angle_threshold() &&
      point.intensity < rnr_params_.min_intensity_threshold()) {
      non_ground_cloud_->emplace_back(point);
    }
  }
}

void PatchWorkPP::sample_initial_seed(
  const int zone_idx, const pcl::PointCloud<PointT> & in_cloud,
  pcl::PointCloud<PointT> & seed_cloud) const
{
  seed_cloud.points.clear();

  int init_idx = 0;
  if (zone_idx == 0) {
    double lowest_z_in_close_zone =
      common_params_.sensor_height() == 0 ? -0.1 : common_params_.lowest_z_in_close_zone();
    for (const auto & point : in_cloud.points) {
      if (lowest_z_in_close_zone < point.z) {
        break;
      }
      ++init_idx;
    }
  }

  const size_t num_sample = std::max(
    std::min(static_cast<int>(in_cloud.points.size()), rpf_params_.num_sample()), init_idx);

  const double sum_z = std::accumulate(
    in_cloud.points.cbegin() + init_idx, in_cloud.points.cbegin() + init_idx + num_sample, 0.0,
    [](double acc, const auto & pt) { return acc + pt.z; });

  const double lowest_z = num_sample != 0 ? sum_z / num_sample : 0.0;

  for (const auto & point : in_cloud.points) {
    if (point.z < lowest_z) {
      seed_cloud.points.emplace_back(point);
    }
  }
}

void PatchWorkPP::region_wise_plane_fitting(
  const int zone_idx, pcl::PointCloud<PointT> & zone_cloud, pcl::PointCloud<PointT> & ground_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud)
{
  // 1. R-VPF ... Extract vertical cloud as non-ground cloud from seed cloud
  pcl::PointCloud<PointT> non_vertical_cloud;
  estimate_vertical_plane(zone_idx, zone_cloud, non_vertical_cloud, ground_cloud, non_ground_cloud);

  // 2. R-GPF ... Extract non-ground cloud from non-vertical cloud
  estimate_ground_plane(zone_idx, non_vertical_cloud, ground_cloud, non_ground_cloud);
}

void PatchWorkPP::estimate_vertical_plane(
  const int zone_idx, pcl::PointCloud<PointT> & in_cloud,
  pcl::PointCloud<PointT> & non_vertical_cloud, pcl::PointCloud<PointT> & ground_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud)
{
  if (!ground_cloud.empty()) {
    ground_cloud.clear();
  }
  if (!non_ground_cloud.empty()) {
    non_ground_cloud.empty();
  }
  // pcl::PointCloud<PointT> src(in_cloud);
  non_vertical_cloud = in_cloud;
  for (int n = 0; n < rpf_params_.num_iterator(); ++n) {
    sample_initial_seed(zone_idx, non_vertical_cloud, ground_cloud);
    estimate_plane(ground_cloud);

    if (zone_idx != 0 || gle_params_.uprightness_threshold() <= v_normal_(0)) {
      break;
    }

    auto tmp_cloud = non_vertical_cloud;
    non_vertical_cloud.clear();
    for (const auto & point : tmp_cloud.points) {
      Eigen::Vector3d p(point.x, point.y, point.z);
      const double distance = v_normal_.dot((p - centroid_.head<3>()).cwiseAbs());  // eq(2)
      const double angle = 0.5 * M_PI - std::acos(v_normal_.dot(u_normal_));        // eq(3)
      if (
        (distance < rpf_params_.max_vertical_distance_threshold()) &&
        angle < rpf_params_.max_angle_threshold()) {
        non_ground_cloud.points.emplace_back(point);
      } else {
        non_vertical_cloud.points.emplace_back(point);
      }
    }
  }
}

void PatchWorkPP::estimate_ground_plane(
  const int zone_idx, pcl::PointCloud<PointT> & non_vertical_cloud,
  pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud)
{
  sample_initial_seed(zone_idx, non_vertical_cloud, ground_cloud);

  for (int n = 0; n < rpf_params_.num_iterator(); ++n) {
    ground_cloud.clear();
    for (const auto & point : non_vertical_cloud.points) {
      Eigen::Vector3d p(point.x, point.y, point.z);
      const double distance = v_normal_.dot((p - centroid_.head<3>()).cwiseAbs());
      if (distance < rpf_params_.max_distance_threshold()) {
        non_ground_cloud.points.emplace_back(point);
      } else {
        ground_cloud.points.emplace_back(point);
      }
    }
    if (n < rpf_params_.num_iterator() - 1) {
      estimate_plane(ground_cloud);
    } else {
      estimate_plane(non_ground_cloud);
    }
  }
}

void PatchWorkPP::estimate_plane(const pcl::PointCloud<PointT> & ground_cloud)
{
  pcl::computeMeanAndCovarianceMatrix(ground_cloud, covariance_matrix_, centroid_);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance_matrix_);
  v_normal_ = solver.eigenvectors().col(0).normalized();
  v_eigenvalues_ = solver.eigenvalues();
}

void PatchWorkPP::temporal_ground_revert(
  std::vector<TGRCandidate> & candidates, std::vector<double> & zone_flatness)
{
  const auto & [mean_flatness, std_flatness] = calculate_mean_stddev(zone_flatness);
  const double flatness_t = mean_flatness + tgr_params_.std_weight() * std_flatness;  // eq(8)
  for (auto & candidate : candidates) {
    if (candidate.flatness < flatness_t) {
      insert_cloud(candidate.ground_cloud, *ground_cloud_);
    } else {
      insert_cloud(candidate.ground_cloud, *non_ground_cloud_);
    }
  }
}

void PatchWorkPP::update_elevation_thresholds()
{
  for (int m = 0; m < czm_params_.num_near_ring(); ++m) {
    if (elevation_buffer_.at(m).empty()) {
      continue;
    }
    const auto & [mean, stddev] = calculate_mean_stddev(elevation_buffer_.at(m));
    const double new_threshold = mean + gle_params_.elevation_std_weights(m) * stddev;  // eq(5)
    czm_params_.update_elevation_threshold(m, new_threshold);

    const int num_exceed = elevation_buffer_.at(m).size() - gle_params_.buffer_storage();
    if (0 < num_exceed) {
      elevation_buffer_.at(m).erase(
        elevation_buffer_.at(m).begin(), elevation_buffer_.at(m).begin() + num_exceed);
    }
  }
}

void PatchWorkPP::update_flatness_thresholds()
{
  for (int m = 0; m < czm_params_.num_near_ring(); ++m) {
    if (flatness_buffer_.at(m).empty()) {
      continue;
    }
    const auto & [mean, stddev] = calculate_mean_stddev(flatness_buffer_.at(m));
    const double new_threshold = mean + gle_params_.flatness_std_weights(m) * stddev;  // eq(6)
    czm_params_.update_flatness_threshold(m, new_threshold);

    const int num_exceed = flatness_buffer_.at(m).size() - gle_params_.buffer_storage();
    if (0 < num_exceed) {
      flatness_buffer_.at(m).erase(
        flatness_buffer_.at(m).begin(), flatness_buffer_.at(m).begin() + num_exceed);
    }
  }
}

void PatchWorkPP::update_height_threshold()
{
  if (elevation_buffer_.at(0).empty()) {
    return;
  }
  const auto & [mean, _] = calculate_mean_stddev(elevation_buffer_.at(0));
  const double new_threshold = mean + gle_params_.height_noise_margin();  // eq(7)
  rnr_params_.updateHeightThreshold(new_threshold);
}

void PatchWorkPP::cloud_to_czm(
  const pcl::PointCloud<PointT> & cloud, pcl::PointCloud<PointT> & non_ground_cloud)
{
  for (const auto & point : cloud) {
    const double radius = calculate_radius(point);

    if ((radius < common_params_.min_range()) || (common_params_.max_range() < radius)) {
      non_ground_cloud.points.emplace_back(point);
      continue;
    }

    const double yaw = calculate_yaw(point);

    for (int i = 1; i < czm_params_.num_zone() - 1; ++i) {
      if (czm_params_.min_zone_ranges(i) <= radius) {
        continue;
      }
      const double ring_size = czm_params_.ring_sizes(i);
      const double sector_size = czm_params_.sector_sizes(i);

      const int64_t ring_idx = std::min(
        static_cast<int64_t>((radius - czm_params_.min_zone_ranges(i - 1)) / ring_size),
        czm_params_.num_rings(i - 1) - 1);
      const int64_t sector_idx =
        std::min(static_cast<int64_t>(yaw / sector_size), czm_params_.num_sectors(i - 1) - 1);

      czm_[i - 1][ring_idx][sector_idx].points.emplace_back(point);
    }
  }
}

}  // namespace patchwork_pp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(patchwork_pp::PatchWorkPP);
