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

#include "autoware/tensorrt_plugins/point_rope_plugin.hpp"

#include "autoware/point_rope_ops/point_rope_kernel.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <cuda_fp16.h>

#include <exception>
#include <string>

namespace nvinfer1::plugin
{
PointRoPEPlugin::PointRoPEPlugin(
  const std::string & name, const PointRoPEParameters & params) noexcept
: layer_name_(name), params_(params)
{
  initFieldsToSerialize();
}

void PointRoPEPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back("base", &params_.base, PluginFieldType::kFLOAT32, 1);
  data_to_serialize_.emplace_back("f0", &params_.f0, PluginFieldType::kFLOAT32, 1);

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * PointRoPEPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild *>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime *>(this);
    }
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3 * PointRoPEPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new PointRoPEPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * PointRoPEPlugin::getPluginName() const noexcept
{
  return kPOINT_ROPE_PLUGIN_NAME;
}

char const * PointRoPEPlugin::getPluginVersion() const noexcept
{
  return kPOINT_ROPE_PLUGIN_VERSION;
}

char const * PointRoPEPlugin::getPluginNamespace() const noexcept
{
  return kPOINT_ROPE_PLUGIN_NAMESPACE;
}

std::int32_t PointRoPEPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t PointRoPEPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 2);   // tokens, positions
  PLUGIN_ASSERT(num_outputs == 1);  // tokens

  // tokens
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);  // [B, N, H, D]
  PLUGIN_ASSERT(in[0].desc.type == DataType::kFLOAT || in[0].desc.type == DataType::kHALF);

  // positions
  PLUGIN_ASSERT(in[1].desc.dims.nbDims == 3);  // [B, N, 3]
  PLUGIN_ASSERT(in[1].desc.type == DataType::kINT64);
  PLUGIN_ASSERT(in[1].desc.dims.d[0] == in[0].desc.dims.d[0]);
  PLUGIN_ASSERT(in[1].desc.dims.d[1] == in[0].desc.dims.d[1]);
  PLUGIN_ASSERT(in[1].desc.dims.d[2] == 3);

  // output
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 4);  // [B, N, H, D]
  PLUGIN_ASSERT(out[0].desc.type == in[0].desc.type);

  return 0;
}

bool PointRoPEPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(0 <= pos && pos < 3);
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);

  constexpr std::int32_t IN_TOKENS_INDEX = 0;
  constexpr std::int32_t IN_POSITIONS_INDEX = 1;
  constexpr std::int32_t OUT_TOKENS_INDEX = 2;

  bool type_ok = false;
  switch (pos) {
    case IN_TOKENS_INDEX:
    case OUT_TOKENS_INDEX:
      type_ok =
        (in_out[pos].desc.type == DataType::kFLOAT || in_out[pos].desc.type == DataType::kHALF);
      break;
    case IN_POSITIONS_INDEX:
      type_ok = in_out[pos].desc.type == DataType::kINT64;
      break;
  }
  return in_out[pos].desc.format == TensorFormat::kLINEAR && type_ok;
}

std::int32_t PointRoPEPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);
  output_types[0] = input_types[0];
  return 0;
}

std::int32_t PointRoPEPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs[0].nbDims == 4);  // [B, N, H, D]
  PLUGIN_ASSERT(inputs[1].nbDims == 3);  // [B, N, 3]

  outputs[0] = inputs[0];
  return 0;
}

std::int32_t PointRoPEPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  const auto B = input_desc[0].dims.d[0];
  const auto N = input_desc[0].dims.d[1];
  const auto H = input_desc[0].dims.d[2];
  const auto D = input_desc[0].dims.d[3];

  const auto pos = static_cast<const std::int64_t *>(inputs[1]);

  if (input_desc[0].type == DataType::kFLOAT) {
    cudaMemcpyAsync(
      outputs[0], inputs[0], B * N * H * D * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    point_rope_launch<float>(
      reinterpret_cast<float *>(outputs[0]), B, N, H, D, pos, params_.base, params_.f0, stream);
  } else if (input_desc[0].type == DataType::kHALF) {
    cudaMemcpyAsync(
      outputs[0], inputs[0], B * N * H * D * sizeof(__half), cudaMemcpyDeviceToDevice, stream);

    point_rope_launch<__half>(
      reinterpret_cast<__half *>(outputs[0]), B, N, H, D, pos, params_.base, params_.f0, stream);
  }

  return 0;
}

std::int32_t PointRoPEPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * PointRoPEPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * PointRoPEPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t PointRoPEPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}
}  // namespace nvinfer1::plugin
