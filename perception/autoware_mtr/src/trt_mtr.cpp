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

#include "autoware/mtr/trt_mtr.hpp"

#include <NvInferRuntimeBase.h>

#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace autoware::mtr
{
TrtMTR::TrtMTR(const tensorrt_common::TrtCommonConfig & config)
{
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(config);

  // TODO(ktro2828): add support of dynamic shape inference
  auto profile_dims = std::make_unique<std::vector<tensorrt_common::ProfileDims>>();
  auto network_io = std::make_unique<std::vector<tensorrt_common::NetworkIO>>();
  {
    constexpr size_t num_input = 8;   // (agent, agent_mask, agent_center,
                                      // map, map_mask, map_center, target_index, target_type)
    constexpr size_t num_output = 2;  // (score, trajectory)
    // input profiles
    for (size_t i = 0; i < num_input; ++i) {
      const auto dims = trt_common_->getInputDims(i);
      profile_dims->emplace_back(i, dims, dims, dims);

      const auto name = trt_common_->getIOTensorName(i);
      network_io->emplace_back(name, dims);
    }

    // output profiles
    for (size_t i = 0; i < num_output; ++i) {
      const auto dims = trt_common_->getOutputDims(i);
      const auto name = trt_common_->getIOTensorName(i + num_input);
      network_io->emplace_back(name, dims);
    }
  }

  if (!trt_common_->setup(std::move(profile_dims), std::move(network_io))) {
    throw archetype::MTRException(
      archetype::MTRError_t::TensorRT, "Failed to setup TensorRT engine.");
  }

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

archetype::Result<TrtMTR::output_type> TrtMTR::do_inference(
  const archetype::AgentTensor & agent_tensor, const archetype::MapTensor & map_tensor) noexcept
{
  // Copy inputs from host to device
  try {
    init_cuda_ptr(agent_tensor, map_tensor);
  } catch (const std::runtime_error & e) {
    return archetype::Err<output_type>(archetype::MTRError_t::Cuda, e.what());
  }

  // function to compute the size of elements
  auto compute_dim_size = [](const nvinfer1::Dims & dims) -> size_t {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
  };

  const auto score_size = compute_dim_size(trt_common_->getOutputDims(0));
  const auto trajectory_size = compute_dim_size(trt_common_->getOutputDims(1));
  out_score_d_ = cuda_utils::make_unique<float[]>(score_size);
  out_trajectory_d_ = cuda_utils::make_unique<float[]>(trajectory_size);

  // TODO(ktro2828): set tensors addresses
  std::vector<void *> tensors{
    in_agent_d_.get(),    in_agent_mask_d_.get(), in_agent_center_d_.get(), in_map_d_.get(),
    in_map_mask_d_.get(), in_map_center_d_.get(), in_target_index_d_.get(), in_target_type_d_.get(),
    out_score_d_.get(),   out_trajectory_d_.get()};
  if (!trt_common_->setTensorsAddresses(tensors)) {
    return archetype::Err<output_type>(
      archetype::MTRError_t::TensorRT, "Failed to set tensor addresses");
  }

  // Execute inference
  if (!trt_common_->enqueueV3(stream_)) {
    return archetype::Err<output_type>(archetype::MTRError_t::TensorRT, "Failed to enqueue.");
  }

  // Copy outputs from device to host
  score_type score_h(score_size);
  trajectory_type trajectory_h(trajectory_size);

  try {
    CHECK_CUDA_ERROR(cudaMemcpy(
      score_h.data(), out_score_d_.get(), sizeof(float) * score_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(
      trajectory_h.data(), out_trajectory_d_.get(), sizeof(float) * trajectory_size,
      cudaMemcpyDeviceToHost));
  } catch (const std::runtime_error & e) {
    return archetype::Err<output_type>(archetype::MTRError_t::Cuda, e.what());
  }

  return archetype::Ok<output_type>({score_h, trajectory_h});
}

void TrtMTR::init_cuda_ptr(
  const archetype::AgentTensor & agent_tensor, const archetype::MapTensor & map_tensor)
{
  in_agent_d_ = cuda_utils::make_unique<float[]>(agent_tensor.in_agent.size());
  in_agent_mask_d_ = cuda_utils::make_unique<uint8_t[]>(agent_tensor.in_agent_mask.size());
  in_agent_center_d_ = cuda_utils::make_unique<float[]>(agent_tensor.in_agent_center.size());
  in_map_d_ = cuda_utils::make_unique<float[]>(map_tensor.in_map.size());
  in_map_mask_d_ = cuda_utils::make_unique<uint8_t[]>(map_tensor.in_map_mask.size());
  in_map_center_d_ = cuda_utils::make_unique<float[]>(map_tensor.in_map_center.size());
  in_target_index_d_ = cuda_utils::make_unique<int[]>(agent_tensor.target_indices.size());
  in_target_type_d_ = cuda_utils::make_unique<int[]>(agent_tensor.target_labels.size());

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_agent_d_.get(), agent_tensor.in_agent.data(), sizeof(float) * agent_tensor.in_agent.size(),
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_agent_mask_d_.get(), agent_tensor.in_agent_mask.data(),
    sizeof(uint8_t) * agent_tensor.in_agent_mask.size(), cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_agent_center_d_.get(), agent_tensor.in_agent_center.data(),
    sizeof(float) * agent_tensor.in_agent_center.size(), cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_map_d_.get(), map_tensor.in_map.data(), sizeof(float) * map_tensor.in_map.size(),
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_map_mask_d_.get(), map_tensor.in_map_mask.data(),
    sizeof(uint8_t) * map_tensor.in_map_mask.size(), cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_map_center_d_.get(), map_tensor.in_map_center.data(),
    sizeof(float) * map_tensor.in_map_center.size(), cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_target_index_d_.get(), agent_tensor.target_indices.data(),
    sizeof(int) * agent_tensor.target_indices.size(), cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    in_target_type_d_.get(), agent_tensor.target_labels.data(),
    sizeof(int) * agent_tensor.target_labels.size(), cudaMemcpyHostToDevice, stream_));
}
}  // namespace autoware::mtr
