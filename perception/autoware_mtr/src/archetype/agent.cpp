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

#include "autoware/mtr/archetype/agent.hpp"

#include "autoware/mtr/archetype/exception.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace autoware::mtr::archetype
{
std::vector<size_t> to_label_ids(const std::vector<std::string> & label_names)
{
  std::vector<size_t> output;
  for (const auto & name : label_names) {
    if (name == "VEHICLE") {
      output.push_back(static_cast<size_t>(AgentLabel::VEHICLE));
    } else if (name == "PEDESTRIAN") {
      output.push_back(static_cast<size_t>(AgentLabel::PEDESTRIAN));
    } else if (name == "MOTORCYCLIST") {
      output.push_back(static_cast<size_t>(AgentLabel::MOTORCYCLIST));
    } else if (name == "CYCLIST") {
      output.push_back(static_cast<size_t>(AgentLabel::CYCLIST));
    } else if (name == "LARGE_VEHICLE") {
      output.push_back(static_cast<size_t>(AgentLabel::LARGE_VEHICLE));
    } else if (name == "UNKNOWN") {
      output.push_back(static_cast<size_t>(AgentLabel::UNKNOWN));
    } else {
      std::ostringstream msg;
      msg << "Unexpected agent label name: " << name;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }
  }
  return output;
}

AgentState AgentState::transform(const AgentState & to_state) const
{
  const auto to_cos = std::cos(to_state.yaw);
  const auto to_sin = std::sin(to_state.yaw);

  const auto tx = (x - to_state.x) * to_cos + (y - to_state.y) * to_sin;
  const auto ty = -(x - to_state.x) * to_sin + (y - to_state.y) * to_cos;
  const auto t_yaw = yaw - to_state.yaw;
  const auto t_vx = vx * to_cos + vy * to_sin;
  const auto t_vy = -vx * to_sin + vy * to_cos;

  return {tx, ty, z, length, width, height, t_yaw, t_vx, t_vy, label, is_valid};
}

AgentHistory AgentHistory::transform(const AgentState & state) const
{
  AgentHistory output(agent_id, queue_.size());
  for (const auto & state_t : *this) {
    if (state_t.is_valid) {
      output.update(state_t.transform(state));
    } else {
      output.update({});
    }
  }
  return output;
}

std::vector<size_t> trim_neighbor_indices(
  const std::vector<AgentHistory> & histories, size_t ego_index, size_t top_k)
{
  std::vector<size_t> output;
  for (size_t i = 0; i < histories.size(); ++i) {
    if (i != ego_index) {
      output.push_back(i);
    }
  }

  const auto & current_ego = histories.at(ego_index).current();

  std::sort(output.begin(), output.end(), [&histories, &current_ego](size_t i1, size_t i2) {
    return histories[i1].distance_from(current_ego) < histories[i2].distance_from(current_ego);
  });

  // keep only the top_k closest agents
  if (output.size() > top_k) {
    output.erase(output.begin() + top_k, output.end());
  }

  return output;
}
}  // namespace autoware::mtr::archetype
