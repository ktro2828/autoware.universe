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

#ifndef PATCHWORKPP__PARAMS_HPP_
#define PATCHWORKPP__PARAMS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <utility>
#include <vector>

namespace patchwork_pp
{
class CommonParams
{
private:
  double sensor_height_;
  double seed_selection_weight_;

public:
  CommonParams() = delete;

  explicit CommonParams(rclcpp::Node * node);

  double sensor_height() const { return sensor_height_; }

  double lowest_z_in_close_zone() const { return sensor_height_ * seed_selection_weight_; }
};

class RNRParams
{
private:
  double min_height_threshold_;
  double min_vertical_angle_threshold_;
  double min_intensity_threshold_;

public:
  RNRParams() = delete;

  explicit RNRParams(rclcpp::Node * node);

  double min_height_threshold() const { return min_height_threshold_; }

  double min_vertical_angle_threshold() const { return min_vertical_angle_threshold_; }

  double min_intensity_threshold() const { return min_intensity_threshold_; }

  void updateHeightThreshold(const double threshold) { min_height_threshold_ = threshold; }
};  // class RNRParams

class CZMParams
{
private:
  size_t num_zone_;
  double min_range_;
  double max_range_;
  size_t num_near_ring_;
  size_t min_num_point_;
  std::vector<std::pair<double, double>> minmax_zone_ranges_;
  std::vector<int64_t> num_sectors_;
  std::vector<int64_t> num_rings_;
  std::vector<double> elevation_thresholds_;
  std::vector<double> flatness_thresholds_;
  std::vector<double> sector_sizes_;
  std::vector<double> ring_sizes_;

public:
  CZMParams() = delete;

  explicit CZMParams(rclcpp::Node * node);

  double min_range() const { return min_range_; }

  double max_range() const { return max_range_; }

  size_t num_zone() const { return num_zone_; }

  size_t num_near_ring() const { return num_near_ring_; }

  size_t min_num_point() const { return min_num_point_; }

  const std::vector<std::pair<double, double>> & minmax_zone_ranges() const
  {
    return minmax_zone_ranges_;
  }
  const std::pair<double, double> & minmax_zone_ranges(const size_t i) const
  {
    return minmax_zone_ranges_.at(i);
  }

  const std::vector<int64_t> & num_sectors() const { return num_sectors_; }
  int64_t num_sectors(const size_t i) const { return num_sectors_.at(i); }

  const std::vector<int64_t> & num_rings() const { return num_rings_; }
  int64_t num_rings(const size_t i) const { return num_rings_.at(i); }

  const std::vector<double> & elevation_thresholds() const { return elevation_thresholds_; }
  double elevation_thresholds(const size_t i) const { return elevation_thresholds_.at(i); }

  const std::vector<double> & flatness_thresholds() const { return flatness_thresholds_; }
  double flatness_thresholds(const size_t i) const { return flatness_thresholds_.at(i); }

  const std::vector<double> & sector_sizes() const { return sector_sizes_; }
  double sector_sizes(const size_t i) const { return sector_sizes_.at(i); }

  const std::vector<double> & ring_sizes() const { return ring_sizes_; }
  double ring_sizes(const size_t i) const { return ring_sizes_.at(i); }

  void update_elevation_threshold(const size_t i, const double threshold)
  {
    elevation_thresholds_[i] = threshold;
  }

  void update_flatness_threshold(const size_t i, const double threshold)
  {
    flatness_thresholds_[i] = threshold;
  }
};  // class CZMParams

class RPFParams
{
private:
  double max_vertical_distance_threshold_;
  double max_angle_threshold_;
  double max_distance_threshold_;
  double gpf_seed_margin_;
  double vpf_seed_margin_;
  size_t num_iterator_;
  size_t num_sample_;

public:
  RPFParams() = delete;

  explicit RPFParams(rclcpp::Node * node);

  double max_vertical_distance_threshold() const { return max_vertical_distance_threshold_; }

  double max_distance_threshold() const { return max_distance_threshold_; }

  double max_angle_threshold() const { return max_angle_threshold_; }

  double gpf_seed_margin() const { return gpf_seed_margin_; }

  double vpf_seed_margin() const { return vpf_seed_margin_; }

  size_t num_iterator() const { return num_iterator_; }

  size_t num_sample() const { return num_sample_; }
};  // class RPFParams

class GLEParams
{
private:
  double uprightness_threshold_;
  std::vector<double> elevation_std_weights_;
  std::vector<double> flatness_std_weights_;
  double height_noise_margin_;
  int buffer_storage_;

public:
  GLEParams() = delete;

  explicit GLEParams(rclcpp::Node * node);

  double uprightness_threshold() const { return uprightness_threshold_; }

  const std::vector<double> elevation_std_weights() const { return elevation_std_weights_; }
  double elevation_std_weights(const size_t i) const { return elevation_std_weights_.at(i); }

  const std::vector<double> & flatness_std_weights() const { return flatness_std_weights_; }
  double flatness_std_weights(const size_t i) const { return flatness_std_weights_.at(i); }

  double height_noise_margin() const { return height_noise_margin_; }

  int buffer_storage() const { return buffer_storage_; }
};  // class GLEParams

}  // namespace patchwork_pp
#endif  // PATCHWORKPP__PARAMS_HPP_
