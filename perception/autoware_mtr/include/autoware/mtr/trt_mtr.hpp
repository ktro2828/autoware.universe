// Copyright 2025 TIER IV, Inc.
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

#ifndef AUTOWARE__MTR__TRT_MTR_HPP_
#define AUTOWARE__MTR__TRT_MTR_HPP_

#include "autoware/mtr/archetype/result.hpp"
#include "autoware/mtr/archetype/tensor.hpp"

#include <autoware/cuda_utils/cuda_unique_ptr.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <autoware/tensorrt_common/utils.hpp>

#include <memory>
#include <tuple>
#include <vector>

namespace autoware::mtr
{
/**
 * @brief A class of mtr for inference with TensorRT.
 */
class TrtMTR
{
public:
  using score_type = std::vector<float>;
  using trajectory_type = std::vector<float>;
  using output_type = std::tuple<score_type, trajectory_type>;

  /**
   * @brief Construct a new TrtMTR object.
   *
   * @param config Configuration of TensorRT engine.
   */
  explicit TrtMTR(const tensorrt_common::TrtCommonConfig & config);

  /**
   * @brief Execute inference.
   *
   * @param agent_tensor Agent tensor.
   * @param map_tensor Map tensor.
   * @return archetype::Result<output_type>
   */
  archetype::Result<output_type> do_inference(
    const archetype::AgentTensor & agent_tensor, const archetype::MapTensor & map_tensor) noexcept;

private:
  /**
   * @brief Initialize and setup cuda pointers.
   */
  void init_cuda_ptr(
    const archetype::AgentTensor & agent_tensor, const archetype::MapTensor & map_tensor);

  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;  //!< TensorRT common.
  cudaStream_t stream_;                                     //!< CUDA stream.

  cuda_utils::CudaUniquePtr<float[]> in_agent_d_;         //!< Input agent tensor on device.
  cuda_utils::CudaUniquePtr<uint8_t[]> in_agent_mask_d_;  //!< Input agent mask tensor on device.
  cuda_utils::CudaUniquePtr<float[]> in_agent_center_d_;  //!< Input agent center tensor on device.
  cuda_utils::CudaUniquePtr<float[]> in_map_d_;           //!< Input map tensor on device.
  cuda_utils::CudaUniquePtr<uint8_t[]> in_map_mask_d_;    //!< Input map mask tensor on device.
  cuda_utils::CudaUniquePtr<float[]> in_map_center_d_;    //!< Input map center tensor on device.
  cuda_utils::CudaUniquePtr<int[]> in_target_index_d_;  //!< Input target indices tensor on device.
  cuda_utils::CudaUniquePtr<int[]> in_target_type_d_;   //!< Input target type tensor on device.

  cuda_utils::CudaUniquePtr<float[]> out_score_d_;       //!< Output score tensor on device.
  cuda_utils::CudaUniquePtr<float[]> out_trajectory_d_;  //!< Output trajectory tensor on device.
};
}  // namespace autoware::mtr
#endif  // AUTOWARE__MTR__TRT_MTR_HPP_
