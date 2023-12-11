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

#include "ground_segmentation/patchwork/params.hpp"

namespace ground_segmentation
{
CommonParams::CommonParams(rclcpp::Node * node)
{
  sensor_height_ = node->declare_parameter<double>("common.sensor_height");
  seed_selection_weight_ = node->declare_parameter<double>("common.seed_selection_weight");
}

RNRParams::RNRParams(rclcpp::Node * node)
{
  min_height_threshold_ = node->declare_parameter<double>("rnr.min_height_threshold");
  min_vertical_angle_threshold_ =
    node->declare_parameter<double>("rnr.min_vertical_angle_threshold");
  min_intensity_threshold_ = node->declare_parameter<double>("rnr.min_intensity_threshold");
}

CZMParams::CZMParams(rclcpp::Node * node)
{
  min_range_ = node->declare_parameter<double>("czm.min_range");
  max_range_ = node->declare_parameter<double>("czm.max_range");
  num_zone_ = static_cast<size_t>(node->declare_parameter<int>("czm.num_zone"));
  num_near_ring_ = static_cast<size_t>(node->declare_parameter<int>("czm.num_near_ring"));
  min_num_point_ = static_cast<size_t>(node->declare_parameter<int>("czm.min_num_point"));
  num_sectors_ = node->declare_parameter<std::vector<int64_t>>("czm.num_sectors");
  num_rings_ = node->declare_parameter<std::vector<int64_t>>("czm.num_rings");
  elevation_thresholds_ = node->declare_parameter<std::vector<double>>("czm.elevation_thresholds");
  flatness_thresholds_ = node->declare_parameter<std::vector<double>>("czm.flatness_thresholds");

  if (num_zone_ != num_sectors_.size()) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `num_sectors` has the same number of elements as `num_zone`, but got "
        << num_sectors_.size() << " and " << num_zone_);
  }
  if (num_zone_ != num_rings_.size()) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `num_rings` has the same number of elements as `num_zone`, but got "
        << num_rings_.size() << " and " << num_zone_);
  }
  if (num_zone_ != elevation_thresholds_.size()) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `elevation_thresholds` has the same number of elements as `num_zone`, but got "
        << elevation_thresholds_.size() << " and " << num_zone_);
  }
  if (num_zone_ != flatness_thresholds_.size()) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `flatness_thresholds` has the same number of elements as `num_zone`, but got "
        << flatness_thresholds_.size() << " and " << num_zone_);
  }

  if (num_zone_ < num_near_ring_) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `num_near_ring` has less or equal number with `num_zone`, but got "
        << num_near_ring_ << " and " << num_zone_);
  }

  for (const auto & num : num_sectors_) {
    sector_sizes_.push_back(2 * M_PI / num);
  }

  {
    // eq(3) in Patchwork
    minmax_zone_ranges_.resize(num_zone_);
    minmax_zone_ranges_.at(0) = std::make_pair(
      min_range_,
      ((std::pow(2.0, num_zone_) - 1.0) * min_range_ + max_range_) / std::pow(2.0, num_zone_));
    for (size_t i = 1; i < num_zone_ - 1; ++i) {
      const double min_zone_range = minmax_zone_ranges(i - 1).second;
      const double max_scale = std::pow(2.0, num_zone_ - (i + 1));
      const double max_zone_range = ((max_scale - 1.0) * min_range_ + max_range_) / max_scale;
      minmax_zone_ranges_.at(i) = std::make_pair(min_zone_range, max_zone_range);
    }
    minmax_zone_ranges_.at(num_zone_ - 1) =
      std::make_pair((min_range_ + max_range_) / 2.0, max_range_);
  }

  {
    ring_sizes_.resize(num_zone_);
    for (size_t i = 0; i < num_zone_; ++i) {
      const auto & [min_zone_range, max_zone_range] = minmax_zone_ranges_.at(i);
      ring_sizes_.at(i) = (max_zone_range - min_zone_range) / num_rings_.at(i);
    }
  }
}

RPFParams::RPFParams(rclcpp::Node * node)
{
  max_vertical_distance_threshold_ =
    node->declare_parameter<double>("rpf.max_vertical_distance_threshold");
  max_angle_threshold_ = node->declare_parameter<double>("rpf.max_angle_threshold");
  max_distance_threshold_ = node->declare_parameter<double>("rpf.max_distance_threshold");
  gpf_seed_margin_ = node->declare_parameter<double>("rpf.gpf_seed_margin");
  vpf_seed_margin_ = node->declare_parameter<double>("rpf.vpf_seed_margin");
  num_iterator_ = static_cast<size_t>(node->declare_parameter<int>("rpf.num_iterator"));
  num_sample_ = static_cast<size_t>(node->declare_parameter<int>("rpf.num_sample"));
}

GLEParams::GLEParams(rclcpp::Node * node)
{
  uprightness_threshold_ = node->declare_parameter<double>("gle.uprightness_threshold");
  elevation_std_weights_ =
    node->declare_parameter<std::vector<double>>("gle.elevation_std_weights");
  flatness_std_weights_ = node->declare_parameter<std::vector<double>>("gle.flatness_std_weights");
  height_noise_margin_ = node->declare_parameter<double>("gle.height_noise_margin");
  buffer_storage_ = node->declare_parameter<int>("gle.buffer_storage");

  if (buffer_storage_ <= 0) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(), "Expected `buffer_storage` > 0, but got " << buffer_storage_);
  }
}

TGRParams::TGRParams(rclcpp::Node * node)
{
  flatness_std_weights_ = node->declare_parameter<std::vector<double>>("tgr.flatness_std_weights");
}
}  // namespace ground_segmentation
