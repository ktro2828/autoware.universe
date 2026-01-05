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

#ifndef AUTOWARE__POINT_ROPE_OPS__POINT_ROPE_KERNEL_HPP_
#define AUTOWARE__POINT_ROPE_OPS__POINT_ROPE_KERNEL_HPP_

#include <cuda_runtime.h>

#include <cstdint>

/**
 * @brief Launches the point RoPE (Rotary Positional Embedding) kernel.
 *
 * @param tokens Pointer to the tokens array.
 * @param B Batch size.
 * @param N Number of points.
 * @param H Height.
 * @param D Depth.
 * @param pos Pointer to the position array.
 * @param base Base value.
 * @param f0 Frequency value.
 * @param stream CUDA stream.
 */
template <typename scalar_t>
void point_rope_launch(
  scalar_t * tokens, const std::int64_t B, const std::int64_t N, const std::int64_t H,
  const std::int64_t D, const std::int64_t * pos, const float base, const float f0,
  cudaStream_t stream);

#endif  // AUTOWARE__POINT_ROPE_OPS__POINT_ROPE_KERNEL_HPP_
