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

#ifndef AUTOWARE__MTR__ARCHETYPE__TENSOR_HPP_
#define AUTOWARE__MTR__ARCHETYPE__TENSOR_HPP_

#include "autoware/mtr/archetype/exception.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace autoware::mtr::archetype
{
/**
 * @brief A class to represent agent tensor data.
 */
class AgentTensor
{
public:
  /**
   * @brief Construct a new AgentTensor object.
   *
   * @param in_agent 1D agent tensor data in the shape of (B*N*T*Da).
   * @param in_agent_mask 1D mask for agent tensor data in the shape of (B*N).
   * @param in_agent_center 1D center of agent tensor data in the shape of (B*N*3).
   * @param num_target Number of targets (B).
   * @param num_agent Number of agents (N).
   * @param num_past Number of past timestamps (T).
   * @param num_attribute Number of attributes (Da).
   */
  AgentTensor(
    const std::vector<float> & in_agent, const std::vector<uint8_t> & in_agent_mask,
    const std::vector<float> & in_agent_center, const std::vector<int> & target_indices,
    const std::vector<int> & target_labels, const std::vector<std::string> & target_ids,
    size_t num_target, size_t num_agent, size_t num_past, size_t num_attribute)
  : num_target(num_target),
    num_agent(num_agent),
    num_past(num_past),
    num_attribute(num_attribute),
    in_agent(in_agent),
    in_agent_mask(in_agent_mask),
    in_agent_center(in_agent_center),
    target_indices(target_indices),
    target_labels(target_labels),
    target_ids(target_ids)
  {
    // check the size of in_agent.
    if (in_agent.size() != num_target * num_agent * num_past * num_attribute) {
      std::ostringstream msg;
      msg << "Invalid size of agent tensor: " << in_agent.size()
          << " != " << num_target * num_agent * num_past * num_attribute;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }

    // check the size of in_agent_mask.
    if (in_agent_mask.size() != num_target * num_agent * num_past) {
      std::ostringstream msg;
      msg << "Invalid size of agent mask: " << in_agent_mask.size()
          << " != " << num_target * num_agent * num_past;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }

    // check the size of in_agent_center.
    if (in_agent_center.size() != num_target * num_agent * 3) {
      std::ostringstream msg;
      msg << "Invalid size of agent center: " << in_agent_center.size()
          << " != " << num_target * num_agent * 3;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }
  }

  const size_t num_target;     //!< Number of targets (B).
  const size_t num_agent;      //!< Number of agents (N).
  const size_t num_past;       //!< Number of past timestamps (T).
  const size_t num_attribute;  //!< Number of attributes (Da).

  const std::vector<float> in_agent;          //!< Agent tensor data.
  const std::vector<uint8_t> in_agent_mask;   //!< Mask for agent tensor data.
  const std::vector<float> in_agent_center;   //!< Center of agent tensor data.
  const std::vector<int> target_indices;      //!< Indices of agents in the tensor data.
  const std::vector<int> target_labels;       //!< Labels for agent tensor data.
  const std::vector<std::string> target_ids;  //!< IDs for agent tensor data.
};

/**
 * @brief A class to represent map tensor data.
 */
class MapTensor
{
public:
  using size_type = std::vector<float>::size_type;

  /**
   * @brief Construct a new MapTensor object.
   *
   * @param in_map 1D map tensor data in the shape of (B*K*P*Dm).
   * @param in_map_mask 1D mask for map tensor data in the shape of (B*K).
   * @param in_map_center 1D center of map tensor data in the shape of (B*K*3).
   * @param num_target Number of targets (B).
   * @param num_polyline Number of polylines (K).
   * @param num_point Number of points contained in a single polyline (P).
   * @param num_attribute Number of attributes (Dm).
   */
  MapTensor(
    const std::vector<float> & in_map, const std::vector<uint8_t> & in_map_mask,
    const std::vector<float> & in_map_center, size_t num_target, size_t num_polyline,
    size_t num_point, size_t num_attribute)
  : num_target(num_target),
    num_polyline(num_polyline),
    num_point(num_point),
    num_attribute(num_attribute),
    in_map(in_map),
    in_map_mask(in_map_mask),
    in_map_center(in_map_center)
  {
    // check the size of in_map.
    if (in_map.size() != num_target * num_polyline * num_point * num_attribute) {
      std::ostringstream msg;
      msg << "Invalid size of map tensor: " << in_map.size()
          << " != " << num_target * num_polyline * num_point * num_attribute;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }

    // check the size of in_map_mask.
    if (in_map_mask.size() != num_target * num_polyline * num_point) {
      std::ostringstream msg;
      msg << "Invalid size of map mask: " << in_map_mask.size()
          << " != " << num_target * num_polyline * num_point;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }

    // check the size of in_map_center.
    if (in_map_center.size() != num_target * num_polyline * 3) {
      std::ostringstream msg;
      msg << "Invalid size of map center: " << in_map_center.size()
          << " != " << num_target * num_polyline * 3;
      throw MTRException(MTRError_t::InvalidValue, msg.str());
    }
  }

  const size_t num_target;     //!< Number of targets (B).
  const size_t num_polyline;   //!< Number of polylines (K).
  const size_t num_point;      //!< Number of points contained in a single polyline (P).
  const size_t num_attribute;  //!< Number of attributes (Dm).

  const std::vector<float> in_map;         //!< Map tensor data.
  const std::vector<uint8_t> in_map_mask;  //!< Mask for map tensor data.
  const std::vector<float> in_map_center;  //!< Center of map tensor data.
};
}  // namespace autoware::mtr::archetype
#endif  // AUTOWARE__MTR__ARCHETYPE__TENSOR_HPP_
