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

#ifndef AUTOWARE__LITEPT__POSTPROCESS__POSTPROCESSOR_HPP_
#define AUTOWARE__LITEPT__POSTPROCESS__POSTPROCESSOR_HPP_

#include "autoware/litept/litept_config.hpp"

#include <autoware/cuda_utils/cuda_unique_ptr.hpp>

#include <cuda_runtime.h>

namespace autoware::litept
{
class Postprocessor
{
public:
  Postprocessor(const LitePTConfig & config, cudaStream_t stream);

  /**
   * @brief Performs postprocessing.
   *
   * @param input_features Read-only pointer to input features.
   * @param pred_labels Read-only pointer to predicted labels.
   * @param output_points Output pointer to postprocessed points.
   * @param num_points The number of points.
   */
  void process(
    const float * input_features, const std::int64_t * pred_labels, float * output_points,
    std::size_t num_points);

private:
  LitePTConfig config_;  //!< Configuration parameters for the LitePT model.
  cudaStream_t stream_;  //!< CUDA stream for asynchronous operations.
  cuda_utils::CudaUniquePtr<float[]> color_map_d_;  //!< Device-side color map.
};
}  // namespace autoware::litept
#endif  // AUTOWARE__LITEPT__POSTPROCESS__POSTPROCESSOR_HPP_
