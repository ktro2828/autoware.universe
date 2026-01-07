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

#ifndef AUTOWARE__LITEPT__PREPROCESS__PREPROCESSOR_HPP_
#define AUTOWARE__LITEPT__PREPROCESS__PREPROCESSOR_HPP_

#include "autoware/litept/litept_config.hpp"
#include "autoware/litept/preprocess/point_type.hpp"

#include <autoware/cuda_utils/cuda_unique_ptr.hpp>

#include <cuda_runtime.h>

namespace autoware::litept
{
class Preprocessor
{
public:
  Preprocessor(const LitePTConfig & config, cudaStream_t stream);

  /**
   * @brief Performs preprocessing.
   *
   * @param input_data Input pointcloud.
   * @param num_points The number of source points.
   * @param voxel_coords Output pointer to the 3D voxel coordinates.
   * @param voxel_features Output pointer to the voxel features.
   * @return std::int64_t The number of voxels.
   */
  std::int64_t process(
    const InputPointType * input_data, uint num_points, std::int64_t * voxel_coords,
    float * voxel_features);

private:
  LitePTConfig config_;
  cudaStream_t stream_;

  cuda_utils::CudaUniquePtr<float[]> points_d_{nullptr};
  cuda_utils::CudaUniquePtr<float[]> cropped_points_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> crop_mask_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> crop_indices_d_{nullptr};

  //!< allocate only if config.use_64bit_hash=true when the gird size exceeds 32-bit limit
  cuda_utils::CudaUniquePtr<std::uint64_t[]> hashes64_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint64_t[]> sorted_hashes64_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint64_t[]> hash_indexes64_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint64_t[]> sorted_hash_indexes64_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint64_t[]> unique_mask64_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint64_t[]> unique_indices64_d_{nullptr};

  //!< buffers used by default
  cuda_utils::CudaUniquePtr<std::uint32_t[]> hashes32_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> sorted_hashes32_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> hash_indexes32_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> sorted_hash_indexes32_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> unique_mask32_d_{nullptr};
  cuda_utils::CudaUniquePtr<std::uint32_t[]> unique_indices32_d_{nullptr};

  cuda_utils::CudaUniquePtr<std::uint8_t[]> sort_workspace_d_{nullptr};
  std::size_t sort_workspace_size_{0};
};
}  // namespace autoware::litept
#endif  // AUTOWARE__LITEPT__PREPROCESS__PREPROCESSOR_HPP_
