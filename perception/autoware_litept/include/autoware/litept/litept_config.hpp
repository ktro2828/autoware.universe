// Copyright 2026 TIER IV, Inc.
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

#ifndef AUTOWARE__LITEPT__LITEPT_CONFIG_HPP_
#define AUTOWARE__LITEPT__LITEPT_CONFIG_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::litept
{
class LitePTConfig
{
private:
  template <typename T>
  struct Range
  {
    T min;
    T max;
  };

  template <typename T>
  struct XYZ
  {
    T x;
    T y;
    T z;
  };

  template <typename T>
  struct XYZAndRange : public XYZ<T>, public Range<T>
  {
  };

public:
  LitePTConfig(
    const std::string & plugins_path, std::int64_t cloud_capacity,
    const std::vector<std::int64_t> & num_voxels, const std::vector<float> & point_cloud_ranges,
    const std::vector<float> & voxel_sizes, const std::vector<std::string> & class_names)
  : plugins_path(plugins_path), cloud_capacity(cloud_capacity), class_names(class_names)
  {
    // Set voxel
    if (num_voxels.size() == 3) {
      num_voxel.x = num_voxels[0];
      num_voxel.y = num_voxels[1];
      num_voxel.z = num_voxels[2];
      num_voxel.min = num_voxels[0];
      num_voxel.max = num_voxels[2];
    } else {
      throw std::invalid_argument("num_voxels must have 3 elements");
    }

    // Set xyz ranges
    if (point_cloud_ranges.size() == 6) {
      xyz_range.x.min = point_cloud_ranges[0];
      xyz_range.y.min = point_cloud_ranges[1];
      xyz_range.z.min = point_cloud_ranges[2];
      xyz_range.x.max = point_cloud_ranges[3];
      xyz_range.y.max = point_cloud_ranges[4];
      xyz_range.z.max = point_cloud_ranges[5];
    } else {
      throw std::invalid_argument("point_cloud_ranges must have 6 elements");
    }

    // Voxel size calculation
    if (voxel_sizes.size() == 3) {
      voxel_size.x = voxel_sizes[0];
      voxel_size.y = voxel_sizes[1];
      voxel_size.z = voxel_sizes[2];
    } else {
      throw std::invalid_argument("voxel_sizes must have 3 elements");
    }

    // Grid size calculation
    {
      auto compute_grid_size = [](const Range<float> & range, const float & voxel_size) -> float {
        return static_cast<std::int64_t>((range.max - range.min) / voxel_size);
      };

      grid_size.x = compute_grid_size(xyz_range.x, voxel_size.x);
      grid_size.y = compute_grid_size(xyz_range.y, voxel_size.y);
      grid_size.z = compute_grid_size(xyz_range.z, voxel_size.z);
    }

    // Check grid depth and size
    {
      auto max_grid_size = std::max({grid_size.x, grid_size.y, grid_size.z});
      auto serialization_depth =
        static_cast<std::int32_t>(std::ceil(std::log2(static_cast<float>(max_grid_size))));

      auto max_voxel_depth =
        static_cast<std::int32_t>(std::ceil(std::log2(static_cast<float>(num_voxel.max))));
      if (serialization_depth * 3 + max_voxel_depth >= 64) {
        throw std::invalid_argument("Serialization depth exceeds maximum allowed value");
      }
    }

    // Use 64-bit hash if grid size exceeds 32-bit limit in postprocessing
    use_64bit_hash =
      grid_size.x * grid_size.y * grid_size.z > std::numeric_limits<std::uint32_t>::max();
  }

  static const std::uint32_t threads_per_block{256};  //!< Number of threads per block
  static const std::int64_t num_point_feature{4};     //!< [x, y, z, intensity]

  const std::string plugins_path;              //!< Path to TensorRT plugins shared library
  const std::int64_t cloud_capacity;           //!< Maximum number of points in point cloud
  const std::vector<std::string> class_names;  //!< List of class names
  const std::int64_t num_classes =
    static_cast<std::int64_t>(class_names.size());  //!< Number of classes
  XYZAndRange<std::int64_t> num_voxel;              //!< Number of voxels in each dimension
  XYZ<Range<float>> xyz_range;                      //!< Point cloud range in meters
  XYZ<float> voxel_size;                            //!< Voxel size in meters
  XYZ<std::int64_t> grid_size;                      //!< Grid size in voxels
  bool use_64bit_hash;                              //!< Use 64-bit hash for voxel indices
};
}  // namespace autoware::litept
#endif  // AUTOWARE__LITEPT__LITEPT_CONFIG_HPP_
