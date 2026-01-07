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

#include "autoware/litept/postprocess/postprocessor.hpp"
#include "autoware/litept/utility.hpp"

#include <autoware/cuda_utils/cuda_check_error.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <array>
#include <cstring>
#include <vector>

namespace autoware::litept
{
namespace
{
// Predefined palette (~47 colors). If num_classes exceeds the palette size, cycle through it.
// Colors are packed as 0xRRGGBB. Names are best-effort, for readability.
static constexpr std::array<std::array<std::uint8_t, 3>, 50> kPaletteRgb = {{
  {{230, 25, 75}},   // 0xE6194B red
  {{60, 180, 75}},   // 0x3CB44B green
  {{255, 225, 25}},  // 0xFFE119 yellow
  {{67, 99, 216}},   // 0x4363D8 blue
  {{245, 130, 49}},  // 0xF58231 orange
  {{145, 30, 180}},  // 0x911EB4 purple
  {{70, 240, 240}},  // 0x46F0F0 cyan
  {{240, 50, 230}},  // 0xF032E6 magenta
  {{188, 246, 12}},  // 0xBCF60C lime

  {{250, 190, 190}},  // 0xFABEBE pink
  {{0, 128, 128}},    // 0x008080 teal
  {{230, 190, 255}},  // 0xE6BEFF lavender
  {{154, 99, 36}},    // 0x9A6324 brown
  {{255, 250, 200}},  // 0xFFFAC8 beige
  {{128, 0, 0}},      // 0x800000 maroon
  {{170, 255, 195}},  // 0xAAFFC3 mint
  {{128, 128, 0}},    // 0x808000 olive
  {{255, 216, 177}},  // 0xFFD8B1 apricot

  {{0, 0, 117}},      // 0x000075 navy
  {{255, 127, 80}},   // 0xFF7F50 coral
  {{30, 144, 255}},   // 0x1E90FF dodger blue
  {{46, 139, 87}},    // 0x2E8B57 sea green
  {{218, 112, 214}},  // 0xDA70D6 orchid
  {{50, 205, 50}},    // 0x32CD32 lime green

  {{255, 20, 147}},  // 0xFF1493 deep pink
  {{138, 43, 226}},  // 0x8A2BE2 blue violet
  {{0, 206, 209}},   // 0x00CED1 dark turquoise
  {{255, 140, 0}},   // 0xFF8C00 dark orange
  {{127, 255, 0}},   // 0x7FFF00 chartreuse
  {{0, 255, 127}},   // 0x00FF7F spring green
  {{220, 20, 60}},   // 0xDC143C crimson
  {{65, 105, 225}},  // 0x4169E1 royal blue
  {{32, 178, 170}},  // 0x20B2AA light sea green

  {{178, 34, 34}},    // 0xB22222 firebrick
  {{255, 182, 193}},  // 0xFFB6C1 light pink
  {{0, 191, 255}},    // 0x00BFFF deep sky blue
  {{34, 139, 34}},    // 0x228B22 forest green
  {{255, 0, 255}},    // 0xFF00FF fuchsia
  {{0, 255, 0}},      // 0x00FF00 green (pure)
  {{255, 215, 0}},    // 0xFFD700 gold
  {{173, 255, 47}},   // 0xADFF2F green yellow
  {{135, 206, 235}},  // 0x87CEEB sky blue

  {{255, 69, 0}},    // 0xFF4500 orange red
  {{106, 90, 205}},  // 0x6A5ACD slate blue
  {{64, 224, 208}},  // 0x40E0D0 turquoise
  {{199, 21, 133}},  // 0xC71585 medium violet red
  {{139, 69, 19}}    // 0x8B4513 saddle brown
}};

/**
 * @brief Generate an RGB color map for the given number of classes.
 * @param config The configuration for the color map generation.
 * @return A vector containing the RGB color map.
 */
std::vector<float> generate_color_map(const LitePTConfig & config)
{
  std::vector<float> color_map(config.num_classes);
  for (std::size_t i = 0; i < config.num_classes; ++i) {
    const auto rgb = kPaletteRgb[i % kPaletteRgb.size()];
    const std::uint32_t packed = (static_cast<std::uint32_t>(rgb[0]) << 16) |
                                 (static_cast<std::uint32_t>(rgb[1]) << 8) |
                                 (static_cast<std::uint32_t>(rgb[2]) << 0);
    std::memcpy(&color_map[i], &packed, sizeof(float));
  }
  return color_map;
}
}  // namespace

__global__ void paintPointcloudKernel(
  const float4 * input_features, const float * colors, const std::int64_t * labels,
  float4 * output_points, std::size_t num_points)
{
  const auto idx = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_points) {
    return;
  }

  const auto label = labels[idx];
  const auto color = colors[label];

  output_points[idx] =
    make_float4(input_features[idx].x, input_features[idx].y, input_features[idx].z, color);
}

Postprocessor::Postprocessor(const LitePTConfig & config, cudaStream_t stream)
: config_(config), stream_(stream)
{
  color_map_d_ = cuda_utils::make_unique<float[]>(config_.num_classes);
  const auto color_map_h = generate_color_map(config_);
  CHECK_CUDA_ERROR(cudaMemcpy(
    color_map_d_.get(), color_map_h.data(), color_map_h.size() * sizeof(float),
    cudaMemcpyHostToDevice));
}

void Postprocessor::process(
  const float * input_features, const std::int64_t * pred_labels, float * output_points,
  std::size_t num_points)
{
  auto num_blocks = divup(num_points, config_.threads_per_block);

  paintPointcloudKernel<<<num_blocks, config_.threads_per_block, 0, stream_>>>(
    reinterpret_cast<const float4 *>(input_features), color_map_d_.get(), pred_labels,
    reinterpret_cast<float4 *>(output_points), num_points);

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
}
}  // namespace autoware::litept
