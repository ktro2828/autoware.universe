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

#include "patchworkpp/params.hpp"

#include <rcpputils/asserts.hpp>

#include <math.h>

namespace patchwork_pp
{
CommonParams::CommonParams(rclcpp::Node * node)
{
  sensor_height_ = node->declare_parameter<double>("common.sensor_height");
  min_range_ = node->declare_parameter<double>("common.min_range");
  max_range_ = node->declare_parameter<double>("common.max_range");
  seed_selection_weight_ = node->declare_parameter<double>("common.seed_selection_weight");
}

RNRParams::RNRParams(rclcpp::Node * node)
{
  num_sample_ = node->declare_parameter<int>("rnr.num_sample");
  min_height_threshold_ = node->declare_parameter<double>("rnr.min_height_threshold");
  min_vertical_angle_threshold_ =
    node->declare_parameter<double>("rnr.min_vertical_angle_threshold");
  min_intensity_threshold_ = node->declare_parameter<double>("rnr.min_intensity_threshold");
}

CZMParams::CZMParams(rclcpp::Node * node)
{
  num_zone_ = node->declare_parameter<int>("czm.num_zone");
  num_near_ring_ = node->declare_parameter<int>("czm.num_near_ring");
  min_num_point_ = node->declare_parameter<int>("czm.min_num_point");
  min_zone_ranges_ = node->declare_parameter<std::vector<double>>("czm.min_zone_ranges");
  num_sectors_ = node->declare_parameter<std::vector<int64_t>>("czm.num_sectors");
  num_rings_ = node->declare_parameter<std::vector<int64_t>>("czm.num_rings");
  elevation_thresholds_ = node->declare_parameter<std::vector<double>>("czm.elevation_thresholds");
  flatness_thresholds_ = node->declare_parameter<std::vector<double>>("czm.flatness_thresholds");

  if (num_zone_ != static_cast<int>(min_zone_ranges_.size())) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `min_zone_ranges` has the same number of elements as `num_zone`, but got "
        << min_zone_ranges_.size() << " and " << num_zone_);
  }
  if (num_zone_ != static_cast<int>(num_sectors_.size())) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `num_sectors` has the same number of elements as `num_zone`, but got "
        << num_sectors_.size() << " and " << num_zone_);
  }
  if (num_zone_ != static_cast<int>(num_rings_.size())) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `num_rings` has the same number of elements as `num_zone`, but got "
        << num_rings_.size() << " and " << num_zone_);
  }
  if (num_near_ring_ != static_cast<int>(elevation_thresholds_.size())) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `elevation_thresholds` has the same number of elements as `num_near_ring`, but got "
        << elevation_thresholds_.size() << " and " << num_zone_);
  }
  if (num_near_ring_ != static_cast<int>(flatness_thresholds_.size())) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Expected `flatness_thresholds` has the same number of elements as `num_near_ring`, but got "
        << flatness_thresholds_.size() << " and " << num_zone_);
  }

  for (const auto & num : num_sectors_) {
    sector_sizes_.push_back(2 * M_PI / num);
  }

  for (int i = 1; i < num_zone_; ++i) {
    ring_sizes_.push_back(
      min_zone_ranges_.at(i) - min_zone_ranges_.at(i - 1) / num_rings_.at(i - 1));
  }
}

RPFParams::RPFParams(rclcpp::Node * node)
{
  max_vertical_distance_threshold_ =
    node->declare_parameter<double>("rpf.max_vertical_distance_threshold");
  max_angle_threshold_ = node->declare_parameter<double>("rpf.max_angle_threshold");
  max_distance_threshold_ = node->declare_parameter<double>("rpf.max_distance_threshold");
  num_iterator_ = node->declare_parameter<int>("rpf.num_iterator");
  num_sample_ = node->declare_parameter<int>("rpf.num_sample");
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
  std_weight_ = node->declare_parameter<double>("tgr.std_weight");
}

}  // namespace patchwork_pp