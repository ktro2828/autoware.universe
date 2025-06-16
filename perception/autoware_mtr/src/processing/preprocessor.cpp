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

// cspell: ignore onehot

#include "autoware/mtr/processing/preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace autoware::mtr::processing
{
namespace
{
/////// polyline ///////
/**
 * @brief Break polylines.
 *
 * @param polylines Vector of source polylines.
 * @param max_num_point Maximum number of points contained in a single polyline.
 * @param break_distance Distance threshold to break two polylines.
 */
std::vector<archetype::Polyline> break_polylines(
  const std::vector<archetype::Polyline> & polylines, size_t max_num_point, double break_distance)
{
  std::vector<archetype::Polyline> output;
  if (polylines.empty()) {
    return polylines;
  }

  std::vector<archetype::MapPoint> flattened;
  for (const auto & polyline : polylines) {
    for (const auto & point : polyline) {
      flattened.emplace_back(point);
    }
  }

  std::vector<archetype::MapPoint> buffer;
  buffer.emplace_back(flattened.front());

  for (size_t i = 1; i < flattened.size(); ++i) {
    const auto & previous = flattened[i - 1];
    const auto & current = flattened[i];

    const bool break_polyline =
      buffer.size() >= max_num_point || previous.distance_from(current) > break_distance;

    if (break_polyline) {
      output.emplace_back(buffer);
      buffer.clear();
    }

    buffer.emplace_back(current);
  }

  if (!buffer.empty()) {
    output.emplace_back(buffer);
  }

  return output;
}
}  // namespace

PreProcessor::PreProcessor(
  const std::vector<size_t> & label_ids, size_t max_num_target, size_t max_num_agent,
  size_t num_past, size_t max_num_polyline, size_t max_num_point, double polyline_range_distance,
  double polyline_break_distance)
: label_ids_(label_ids),
  max_num_target_(max_num_target),
  max_num_agent_(max_num_agent),
  num_past_(num_past),
  max_num_polyline_(max_num_polyline),
  max_num_point_(max_num_point),
  polyline_range_distance_(polyline_range_distance),
  polyline_break_distance_(polyline_break_distance)
{
}

PreProcessor::output_type PreProcessor::process(
  const std::vector<double> & timestamps, const std::vector<archetype::AgentHistory> & histories,
  const std::vector<archetype::Polyline> & polylines, size_t ego_index) const
{
  const auto agent_tensor = this->process_agent(timestamps, histories, ego_index);

  const auto map_tensor =
    this->process_map(polylines, histories, agent_tensor.target_indices, ego_index);

  return {agent_tensor, map_tensor};
}

archetype::AgentTensor PreProcessor::process_agent(
  const std::vector<double> & timestamps, const std::vector<archetype::AgentHistory> & histories,
  size_t ego_index) const
{
  // trim top-k nearest neighbor histories
  const auto target_indices =
    archetype::trim_neighbor_indices(histories, ego_index, max_num_agent_);

  const size_t num_label = label_ids_.size();
  const size_t num_attribute = num_past_ + num_label + 16;  // T + L + 16

  // index offsets
  const size_t offset_type = 6;
  const size_t offset_time = offset_type + num_label + 2;
  const size_t offset_yaw = offset_time + num_past_ + 1;
  const size_t offset_vel = offset_yaw + 2;
  const size_t offset_accel = offset_vel + 2;

  // TODO(ktro2828): max_num_agent should be renamed to max_num_target
  std::vector<float> in_agent(
    max_num_target_ * max_num_agent_ * num_past_ * num_attribute);  // (B * N * T * A)
  std::vector<uint8_t> in_agent_mask(max_num_target_ * max_num_agent_ * num_past_);  // (B * N * T)
  std::vector<float> in_agent_center(max_num_target_ * max_num_agent_ * 3);          // (B * N * 3)
  std::vector<int> target_labels(max_num_target_);                                   // (B)

  std::vector<std::string> target_ids;  // NOTE: used in postprocessing
  for (size_t b = 0; b < target_indices.size() && b < max_num_target_; ++b) {
    const auto & agent_index = target_indices.at(b);
    const auto & target_history = histories.at(agent_index);
    const auto & target_current = target_history.current();

    target_ids.emplace_back(target_history.agent_id);

    const auto label_id = static_cast<size_t>(target_current.label);
    target_labels[b] = label_id;
    for (size_t n = 0; n < histories.size() && n < max_num_agent_; ++n) {
      const auto & history_n = histories.at(n);

      // transform from map coordinate to current state relative coordinate
      const auto transformed = history_n.transform(target_current);

      // agent center
      const size_t center_idx = (b * max_num_agent_ + n) * 3;
      const auto & center = transformed.current();
      in_agent_center[center_idx] = static_cast<float>(center.x);
      in_agent_center[center_idx + 1] = static_cast<float>(center.y);
      in_agent_center[center_idx + 2] = static_cast<float>(center.z);

      for (size_t t = 0; t < transformed.size(); ++t) {
        const auto & state_t = transformed.at(t);

        // agent mask
        const size_t mask_idx = (b * max_num_agent_ + n) * num_past_ + t;
        in_agent_mask[mask_idx] = state_t.is_valid ? 1 : 0;

        // agent attributes
        const size_t idx = mask_idx * num_attribute;
        // 1.xyz
        in_agent[idx] = static_cast<float>(state_t.x);
        in_agent[idx + 1] = static_cast<float>(state_t.y);
        in_agent[idx + 2] = static_cast<float>(state_t.z);
        // 2.size
        in_agent[idx + 3] = static_cast<float>(state_t.length);
        in_agent[idx + 4] = static_cast<float>(state_t.width);
        in_agent[idx + 5] = static_cast<float>(state_t.height);
        // 3.type onehot
        in_agent[idx + offset_type + label_id] = 1.0f;
        if (b == n) {
          in_agent[idx + offset_type + num_label] = 1.0f;
        }
        if (n == ego_index) {
          in_agent[idx + offset_type + num_label + 1] = 1.0f;
        }
        // 4.time embedding
        in_agent[idx + offset_time + t] = 1.0f;
        in_agent[idx + offset_time + num_past_] = static_cast<float>(timestamps.at(t));
        // 5.yaw embedding
        in_agent[idx + offset_yaw] = std::sin(state_t.yaw);
        in_agent[idx + offset_yaw + 1] = std::cos(state_t.yaw);
        // 6. vxy
        in_agent[idx + offset_vel] = state_t.vx;
        in_agent[idx + offset_vel + 1] = state_t.vy;
        // 7. accel
        if (t > 0) {
          const auto & prev = transformed.at(t - 1);
          in_agent[idx + offset_accel] = static_cast<float>(state_t.vx - prev.vx) / 0.1f;
          in_agent[idx + offset_accel + 1] = static_cast<float>(state_t.vy - prev.vy) / 0.1f;
        } else if (t == 0 && transformed.size() > 1) {
          const auto & state_t1 = transformed.at(1);
          in_agent[idx + offset_accel] = static_cast<float>(state_t1.x - state_t.vx) / 0.1f;
          in_agent[idx + offset_accel + 1] = static_cast<float>(state_t1.vy - state_t.vy) / 0.1f;
        }
      }
    }
  }

  return archetype::AgentTensor(
    in_agent, in_agent_mask, in_agent_center, target_indices, target_labels, target_ids,
    max_num_target_, max_num_agent_, num_past_, num_attribute);
}

archetype::MapTensor PreProcessor::process_map(
  const std::vector<archetype::Polyline> & polylines,
  const std::vector<archetype::AgentHistory> & histories, const std::vector<int> & target_indices,
  size_t ego_index) const
{
  // trim neighbor polylines
  const auto & current_ego = histories.at(ego_index).current();
  auto result = archetype::trim_neighbors(polylines, current_ego, polyline_range_distance_);

  // w.r.t map coordinate frame
  result = break_polylines(result, max_num_point_, polyline_break_distance_);

  std::sort(
    result.begin(), result.end(),
    [&current_ego](const archetype::Polyline & p1, const archetype::Polyline & p2) {
      const auto d1 = p1.distance_from(current_ego);
      const auto d2 = p2.distance_from(current_ego);
      return d1 < d2;
    });

  // create tensor, node centers and vectors
  constexpr size_t num_attribute = 8;  // (x, y, z, dx, dy, dz, pre_x, pre_y)
  std::vector<float> in_map(
    max_num_target_ * max_num_polyline_ * max_num_point_ * num_attribute);  // (B * K * P * A)
  std::vector<uint8_t> in_map_mask(
    max_num_target_ * max_num_polyline_ * max_num_point_);                    // (B * K * P)
  std::vector<float> in_map_center(max_num_target_ * max_num_polyline_ * 3);  // (B * K * 3)
  for (size_t b = 0; b < target_indices.size() && b < max_num_target_; ++b) {
    const auto & agent_index = target_indices.at(b);
    const auto & target_current = histories.at(agent_index).current();
    for (size_t k = 0; k < result.size() && k < max_num_polyline_; ++k) {
      // transform map to ego
      const auto & polyline = result.at(k).transform(target_current);

      // center
      const auto center_idx = (b * max_num_polyline_ + k) * 3;
      const auto & center = polyline.center();
      in_map_center[center_idx] = static_cast<float>(center.x);
      in_map_center[center_idx + 1] = static_cast<float>(center.y);
      in_map_center[center_idx + 2] = static_cast<float>(center.z);
      for (size_t p = 0; p < polyline.size() && p < max_num_point_; ++p) {
        const auto & point = polyline.at(p);
        // map mask
        const size_t mask_idx = (b * max_num_polyline_ + k) * max_num_point_ + p;
        in_map_mask[mask_idx] = 1;

        // map attributes
        const size_t idx = mask_idx * num_attribute;
        // 1.xyz
        in_map[idx] = static_cast<float>(point.x);
        in_map[idx + 1] = static_cast<float>(point.y);
        in_map[idx + 2] = static_cast<float>(point.z);
        // 2.dx, dy, dz
        if (p > 0) {
          const auto & previous = polyline.at(p - 1);
          const auto [dx, dy, dz] = point.diff(previous);
          in_map[idx + 3] = static_cast<float>(dx);
          in_map[idx + 4] = static_cast<float>(dy);
          in_map[idx + 5] = static_cast<float>(dz);
        } else {
          in_map[idx + 3] = 0.0f;
          in_map[idx + 4] = 0.0f;
          in_map[idx + 5] = 0.0f;
        }
        // 3.pre_x, pre_y
        if (p > 0) {
          const auto & previous = polyline.at(p - 1);
          in_map[idx + 6] = static_cast<float>(previous.x);
          in_map[idx + 7] = static_cast<float>(previous.y);
        } else {
          in_map[idx + 6] = static_cast<float>(point.x);
          in_map[idx + 7] = static_cast<float>(point.y);
        }
      }
    }
  }

  return archetype::MapTensor(
    in_map, in_map_mask, in_map_center, max_num_agent_, max_num_polyline_, max_num_point_,
    num_attribute);
}
}  // namespace autoware::mtr::processing
