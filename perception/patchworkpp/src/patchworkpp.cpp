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

template <typename U>
std::ostream & operator<<(
  std::basic_ostream<U> & os, const std::vector<std::pair<double, double>> & values)
{
  os << "(";
  for (const auto & [v1, v2] : values) {
    os << "(" << v1 << ", " << v2 << "), ";
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
  gle_params_(this)
{
  debug_ = declare_parameter<bool>("debug");

  in_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  non_ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  sector_non_ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();
  sector_ground_cloud_ = std::make_shared<pcl::PointCloud<PointT>>();

  for (size_t zone_idx = 0; zone_idx < czm_params_.num_zone(); ++zone_idx) {
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
  pcl::fromROSMsg(*cloud_msg, *in_cloud_);

  // 1. RNR
  std::queue<size_t> noise_indices = remove_reflected_noise(*in_cloud_);

  // 2. CZM
  refresh_czm();
  cloud_to_czm(*in_cloud_, *non_ground_cloud_, noise_indices);

  std::vector<TGRCandidate> candidates;
  std::vector<double> ring_flatness;
  size_t concentric_idx = 0;
  for (size_t zone_idx = 0; zone_idx < czm_params_.num_zone(); ++zone_idx) {
    Zone & zone = czm_.at(zone_idx);
    for (int ring_idx = 0; ring_idx < czm_params_.num_rings(zone_idx); ++ring_idx) {
      for (int sector_idx = 0; sector_idx < czm_params_.num_sectors(zone_idx); ++sector_idx) {
        pcl::PointCloud<PointT> & zone_cloud = zone.at(ring_idx).at(sector_idx);

        if (zone_cloud.points.empty()) {
          continue;
        }

        if (zone_cloud.points.size() < czm_params_.min_num_point()) {
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
        // v_eigenvalues_ = (e3, e2, e1) ... (e3 <= e2 <= e1)
        // v_normal_ = (vx, vy, vz) ... corresponding to e3
        // centroid_ = (cx, cy, cz)
        const double uprightness = v_normal_(2, 0);
        const double elevation = centroid_(2, 0);
        const double flatness = v_eigenvalues_.minCoeff();

        const bool is_near_ring = concentric_idx < czm_params_.num_near_ring();
        const bool is_upright = gle_params_.uprightness_threshold() < uprightness;
        const bool is_not_elevated =
          is_near_ring ? elevation < czm_params_.elevation_thresholds(zone_idx) : false;
        const bool is_flat =
          is_near_ring ? flatness < czm_params_.flatness_thresholds(zone_idx) : false;

        if (is_upright && is_not_elevated && is_near_ring) {
          elevation_buffer_.at(concentric_idx).emplace_back(elevation);
          flatness_buffer_.at(concentric_idx).emplace_back(flatness);
          ring_flatness.emplace_back(flatness);
        }

        if (!is_upright) {
          insert_cloud(*sector_ground_cloud_, *non_ground_cloud_);
        } else if (!is_near_ring || is_not_elevated || is_flat) {
          insert_cloud(*sector_ground_cloud_, *ground_cloud_);
        } else {
          TGRCandidate candidate(zone_idx, *sector_ground_cloud_, flatness);
          candidates.emplace_back(candidate);
        }
        insert_cloud(*sector_non_ground_cloud_, *non_ground_cloud_);
      }

      if (!candidates.empty()) {
        temporal_ground_revert(candidates, ring_flatness);
        candidates.clear();
        ring_flatness.clear();
      }
      ++concentric_idx;
    }
  }

  update_elevation_thresholds();
  update_flatness_thresholds();
  update_height_threshold();

  // ===== DEBUG =====
  RCLCPP_INFO_STREAM(get_logger(), "min zone ranges: " << czm_params_.minmax_zone_ranges());
  RCLCPP_INFO_STREAM(get_logger(), "elevation thresholds: " << czm_params_.elevation_thresholds());
  RCLCPP_INFO_STREAM(get_logger(), "flatness thresholds: " << czm_params_.flatness_thresholds());

  end = std::chrono::system_clock::now();
  double elapsed =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3;
  RCLCPP_INFO_STREAM(get_logger(), "Processing time: " << elapsed);

  pcl::PointCloud<PointT> in_cloud_filtered;
  for (const auto & point : in_cloud_->points) {
    const double radius = calculate_radius(point);
    if (czm_params_.min_range() <= radius && radius <= czm_params_.max_range()) {
      in_cloud_filtered.push_back(point);
    }
  }

  if (
    non_ground_cloud_->points.size() + ground_cloud_->points.size() !=
    in_cloud_filtered.points.size()) {
    RCLCPP_WARN_STREAM(
      get_logger(), "Size of points between input and output is different!! [input]: "
                      << in_cloud_filtered.points.size() << ", [output]: (total)"
                      << non_ground_cloud_->points.size() + ground_cloud_->points.size()
                      << ", (non_ground): " << non_ground_cloud_->points.size()
                      << ", (ground): " << ground_cloud_->points.size());
  } else {
    RCLCPP_INFO(get_logger(), "Size of points between input and output is same");
  }
  // ==================

  publish(cloud_msg->header);
}

std::queue<size_t> PatchWorkPP::remove_reflected_noise(
  const pcl::PointCloud<PointT> & in_cloud) const
{
  std::queue<size_t> noise_indices;
  for (size_t i = 0; i < in_cloud.points.size(); ++i) {
    const auto & point = in_cloud.points.at(i);
    const double radius = calculate_radius(point);                           // [m]
    const double incident_angle = std::atan2(point.z, radius) * 180 / M_PI;  // [deg]
    if (
      point.z < rnr_params_.min_height_threshold() &&
      incident_angle < rnr_params_.min_vertical_angle_threshold() &&
      point.intensity < rnr_params_.min_intensity_threshold()) {
      non_ground_cloud_->emplace_back(point);
      noise_indices.push(i);
    }
  }
  return noise_indices;
}

void PatchWorkPP::sample_initial_seed(
  const size_t zone_idx, const pcl::PointCloud<PointT> & in_cloud,
  pcl::PointCloud<PointT> & seed_cloud, const double seed_margin) const
{
  seed_cloud.points.clear();

  size_t init_idx = 0;
  if (zone_idx == 0) {
    double lowest_z_in_close_zone = common_params_.lowest_z_in_close_zone();
    for (const auto & point : in_cloud.points) {
      if (lowest_z_in_close_zone < point.z) {
        break;
      }
      ++init_idx;
    }
  }

  const size_t num_sample =
    std::max(std::min(in_cloud.points.size(), rpf_params_.num_sample()), init_idx);

  const double sum_z = std::accumulate(
    in_cloud.points.cbegin() + init_idx, in_cloud.points.cbegin() + init_idx + num_sample, 0.0,
    [](double acc, const auto & pt) { return acc + pt.z; });

  const double lowest_z = num_sample != 0 ? sum_z / num_sample : 0.0;

  for (const auto & point : in_cloud.points) {
    if (point.z < lowest_z + seed_margin) {
      seed_cloud.points.emplace_back(point);
    }
  }
}

void PatchWorkPP::region_wise_plane_fitting(
  const size_t zone_idx, pcl::PointCloud<PointT> & zone_cloud,
  pcl::PointCloud<PointT> & ground_cloud, pcl::PointCloud<PointT> & non_ground_cloud)
{
  // 1. R-VPF ... Extract vertical cloud as non-ground cloud from seed cloud
  pcl::PointCloud<PointT> non_vertical_cloud;
  estimate_vertical_plane(zone_idx, zone_cloud, non_vertical_cloud, ground_cloud, non_ground_cloud);

  // 2. R-GPF ... Extract non-ground cloud from non-vertical cloud
  estimate_ground_plane(zone_idx, non_vertical_cloud, ground_cloud, non_ground_cloud);
}

void PatchWorkPP::estimate_vertical_plane(
  const size_t zone_idx, pcl::PointCloud<PointT> & in_cloud,
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
  for (size_t n = 0; n < rpf_params_.num_iterator(); ++n) {
    sample_initial_seed(zone_idx, non_vertical_cloud, ground_cloud, rpf_params_.vpf_seed_margin());
    estimate_plane(ground_cloud);

    if (zone_idx != 0 || gle_params_.uprightness_threshold() <= v_normal_(2, 0)) {
      break;
    }

    auto tmp_cloud = non_vertical_cloud;
    non_vertical_cloud.clear();
    for (const auto & point : tmp_cloud.points) {
      Eigen::Vector3d p(
        point.x - centroid_(0, 0), point.y - centroid_(1, 0), point.z - centroid_(2, 0));
      const double distance = std::abs(v_normal_.dot(p));                               // eq(2)
      const double angle = std::abs(0.5 * M_PI - std::acos(v_normal_.dot(u_normal_)));  // eq(3)
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
  const size_t zone_idx, pcl::PointCloud<PointT> & in_cloud, pcl::PointCloud<PointT> & ground_cloud,
  pcl::PointCloud<PointT> & non_ground_cloud)
{
  auto tmp_ground_cloud = ground_cloud;
  ground_cloud.clear();

  sample_initial_seed(zone_idx, in_cloud, tmp_ground_cloud, rpf_params_.gpf_seed_margin());
  estimate_plane(tmp_ground_cloud);

  for (size_t n = 0; n < rpf_params_.num_iterator(); ++n) {
    tmp_ground_cloud.clear();
    for (const auto & point : in_cloud.points) {
      Eigen::Vector3d p(
        point.x - centroid_(0, 0), point.y - centroid_(1, 0), point.z - centroid_(2, 0));
      const double distance = v_normal_.dot(p);
      if (n < rpf_params_.num_iterator() - 1) {
        if (distance < rpf_params_.max_distance_threshold()) {
          tmp_ground_cloud.points.emplace_back(point);
        }
      } else {
        if (distance < rpf_params_.max_distance_threshold()) {
          ground_cloud.points.emplace_back(point);
        } else {
          non_ground_cloud.points.emplace_back(point);
        }
      }
    }

    if (n < rpf_params_.num_iterator() - 1) {
      estimate_plane(tmp_ground_cloud);
    } else {
      estimate_plane(ground_cloud);
    }
  }
}

void PatchWorkPP::estimate_plane(const pcl::PointCloud<PointT> & ground_cloud)
{
  pcl::computeMeanAndCovarianceMatrix(ground_cloud, covariance_matrix_, centroid_);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance_matrix_);
  v_normal_ = solver.eigenvectors().col(0).normalized();
  v_eigenvalues_ = solver.eigenvalues();

  if (v_normal_(2, 0) < 0.0) {
    for (auto & v : v_normal_) {
      v *= -1;
    }
  }
}

void PatchWorkPP::temporal_ground_revert(
  const std::vector<TGRCandidate> & candidates, const std::vector<double> & ring_flatness)
{
  for (const auto & candidate : candidates) {
    const auto & [mean_flatness, std_flatness] = calculate_mean_stddev(ring_flatness);

    const double ring_flatness_t =
      mean_flatness + gle_params_.flatness_std_weights(candidate.zone_idx) * std_flatness;  // eq(8)

    if (candidate.flatness < ring_flatness_t) {
      insert_cloud(candidate.ground_cloud, *ground_cloud_);
    } else {
      insert_cloud(candidate.ground_cloud, *non_ground_cloud_);
    }
  }
}

void PatchWorkPP::update_elevation_thresholds()
{
  for (size_t m = 0; m < czm_params_.num_near_ring(); ++m) {
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
  for (size_t m = 0; m < czm_params_.num_near_ring(); ++m) {
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
  const pcl::PointCloud<PointT> & in_cloud, pcl::PointCloud<PointT> & non_ground_cloud,
  std::queue<size_t> & noise_indices)
{
  for (size_t pt_idx = 0; pt_idx < in_cloud.size(); ++pt_idx) {
    if ((!noise_indices.empty() && pt_idx == noise_indices.front())) {
      noise_indices.pop();
      continue;
    }

    const auto & point = in_cloud.points.at(pt_idx);
    const double radius = calculate_radius(point);

    if ((radius < czm_params_.min_range()) || (czm_params_.max_range() < radius)) {
      non_ground_cloud.points.emplace_back(point);
      continue;
    }

    const double yaw = calculate_yaw(point);

    for (size_t i = 0; i < czm_params_.num_zone(); ++i) {
      const auto & [min_zone_range, max_zone_range] = czm_params_.minmax_zone_ranges(i);
      if (radius < min_zone_range || max_zone_range <= radius) {
        continue;
      }
      const double ring_size = czm_params_.ring_sizes(i);
      const double sector_size = czm_params_.sector_sizes(i);

      const int64_t ring_idx = std::min(
        static_cast<int64_t>((radius - min_zone_range) / ring_size), czm_params_.num_rings(i) - 1);
      const int64_t sector_idx =
        std::min(static_cast<int64_t>(yaw / sector_size), czm_params_.num_sectors(i) - 1);

      czm_[i][ring_idx][sector_idx].points.emplace_back(point);
    }
  }
}

}  // namespace patchwork_pp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(patchwork_pp::PatchWorkPP);
