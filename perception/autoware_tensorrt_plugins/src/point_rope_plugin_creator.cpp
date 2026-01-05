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

#include "autoware/tensorrt_plugins/point_rope_plugin_creator.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"
#include "autoware/tensorrt_plugins/point_rope_plugin.hpp"

#include <NvInferRuntime.h>

#include <cstring>
#include <string>

namespace nvinfer1::plugin
{
REGISTER_TENSORRT_PLUGIN(PointRoPEPluginCreator);

PointRoPEPluginCreator::PointRoPEPluginCreator()
{
  plugin_attributes_.clear();
  plugin_attributes_.emplace_back("base", nullptr, PluginFieldType::kFLOAT32, 1);
  plugin_attributes_.emplace_back("f0", nullptr, PluginFieldType::kFLOAT32, 1);

  fc_.nbFields = plugin_attributes_.size();
  fc_.fields = plugin_attributes_.data();
}

PluginFieldCollection const * PointRoPEPluginCreator::getFieldNames() noexcept
{
  return &fc_;
}

IPluginV3 * PointRoPEPluginCreator::createPlugin(
  char const * name, PluginFieldCollection const * fc, TensorRTPhase phase) noexcept
{
  if (phase == TensorRTPhase::kBUILD || phase == TensorRTPhase::kRUNTIME) {
    const PluginField * fields = fc->fields;
    const std::int32_t num_fields = fc->nbFields;

    PLUGIN_VALIDATE(num_fields == 2);

    PointRoPEParameters parameters;
    for (std::int32_t i = 0; i < num_fields; ++i) {
      const std::string & name = fields[i].name;
      PluginFieldType type = fields[i].type;

      PLUGIN_VALIDATE(name == "base" || name == "f0");
      PLUGIN_VALIDATE(type == PluginFieldType::kFLOAT32);
      PLUGIN_VALIDATE(fields[i].length == 1);

      if (name == "base") {
        parameters.base = *static_cast<float const *>(fields[i].data);
      } else if (name == "f0") {
        parameters.f0 = *static_cast<float const *>(fields[i].data);
      }
    }
    return new (std::nothrow) PointRoPEPlugin(name, parameters);
  } else {
    return nullptr;
  }
}
}  // namespace nvinfer1::plugin
