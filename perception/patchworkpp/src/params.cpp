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
  min_num_points_ = node->declare_parameter<int>("czm.min_num_points");
  min_zone_ranges_ = node->declare_parameter<std::vector<int64_t>>("czm.min_zone_ranges");
  num_sectors_ = node->declare_parameter<std::vector<int64_t>>("czm.num_sectors");
  num_rings_ = node->declare_parameter<std::vector<int64_t>>("czm.num_rings");
  elevation_thresholds_ = node->declare_parameter<std::vector<double>>("czm.elevation_thresholds");
  flatness_thresholds_ = node->declare_parameter<std::vector<double>>("czm.flatness_thresholds");

  for (const auto & num : num_sectors_) {
    sector_sizes_.push_back(2 * M_PI / num);
  }

  for (int i = 1; i < num_zone_; ++i) {
    ring_sizes_.push_back(
      min_zone_ranges_.at(i) - min_zone_ranges_.at(i - 1) / num_rings_.at(i - 1));
  }

  // TODO(ktro2828): validate num_zone and size of each vector is same.
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
  elevation_std_weights_ = node->declare_parameter<double>("gle.elevation_std_weights");
  flatness_std_weights_ = node->declare_parameter<std::vector<double>>("gle.flatness_std_weights");
  height_noise_margin_ = node->declare_parameter<double>("gle.height_noise_margin");
}

TGRParams::TGRParams(rclcpp::Node * node)
{
  std_weight_ = node->declare_parameter<double>("tgr.std_weight");
}

}  // namespace patchwork_pp
