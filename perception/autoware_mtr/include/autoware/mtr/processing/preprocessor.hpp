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

#ifndef AUTOWARE__MTR__PROCESSING__PREPROCESSOR_HPP_
#define AUTOWARE__MTR__PROCESSING__PREPROCESSOR_HPP_

#include "autoware/mtr/archetype/agent.hpp"
#include "autoware/mtr/archetype/polyline.hpp"
#include "autoware/mtr/archetype/tensor.hpp"

#include <tuple>
#include <utility>
#include <vector>

namespace autoware::mtr::processing
{
/**
 * @brief A class to execute preprocessing.
 */
class PreProcessor
{
public:
  using output_type = std::tuple<archetype::AgentTensor, archetype::MapTensor>;

  /**
   * @brief Construct a new Preprocessor object.
   *
   * @param label_ids Vector of predictable label ids.
   * @param max_num_target Maximum number of predictable agents (B).
   * @param max_num_agent Maximum number of other agents (N).
   * @param num_past Number of past timestamps (T).
   * @param max_num_polyline Maximum number of polylines (K).
   * @param max_num_point Maximum number of points in a single polyline (P).
   * @param polyline_range_distance Distance threshold from ego to trim polylines [m].
   * @param polyline_break_distance Distance threshold to break two polylines [m].
   */
  PreProcessor(
    const std::vector<size_t> & label_ids, size_t max_num_target, size_t max_num_agent,
    size_t num_past, size_t max_num_polyline, size_t max_num_point, double polyline_range_distance,
    double polyline_break_distance);

  /**
   * @brief Execute preprocessing.
   *
   * @param timestamps Vector of relative timestamps [sec].
   * @param histories Vector of histories for each agent.
   * @param polylines Vector of polylines.
   * @param ego_index Ego index in the histories.
   * @return Returns `archetype::AgentTensor`, `archetype::MapTensor`.
   */
  output_type process(
    const std::vector<double> & timestamps, const std::vector<archetype::AgentHistory> & histories,
    const std::vector<archetype::Polyline> & polylines, size_t ego_index) const;

private:
  /**
   * @brief Return the agent indices to be considered by filtering histories with its label and
   * distance.
   *
   * @param histories Vector of agent histories.
   * @param ego_index Index of ego in the histories.
   * @return target indices and neighbor indices.
   */
  std::pair<std::vector<size_t>, std::vector<size_t>> filter_agent(
    const std::vector<archetype::AgentHistory> & histories, size_t ego_index) const;

  /**
   * @brief Execute preprocessing for agent tensor.
   *
   * @param timestamps Vector of timestamps [sec].
   * @param histories Vector of histories for each agent.
   * @param ego_index Ego index in the histories.
   */
  archetype::AgentTensor process_agent(
    const std::vector<double> & timestamps, const std::vector<archetype::AgentHistory> & histories,
    size_t ego_index) const;

  /**
   * @brief Execute preprocessing for map tensor.
   *
   * @param polylines Vector of polylines.
   * @param histories Vector of histories for each agent.
   * @param target_indices Vector of target indices in the histories.
   * @param ego_index Ego index in the histories.
   */
  archetype::MapTensor process_map(
    const std::vector<archetype::Polyline> & polylines,
    const std::vector<archetype::AgentHistory> & histories, const std::vector<int> & target_indices,
    size_t ego_index) const;

  const std::vector<size_t> label_ids_;   //!< Vector of predictable label ids.
  const size_t max_num_target_;           //!< Maximum number of predictable agents (B).
  const size_t max_num_agent_;            //!< Maximum number of predictable agents (N).
  const size_t num_past_;                 //!< Number of past timestamps (Tp).
  const size_t max_num_polyline_;         //!< Maximum number of polylines (K).
  const size_t max_num_point_;            //!< Maximum number of points in a single polyline (P).
  const double polyline_range_distance_;  //!< Distance threshold from ego to trim polylines [m].
  const double polyline_break_distance_;  //!< Distance threshold to break two polylines [m].
};
}  // namespace autoware::mtr::processing
#endif  // AUTOWARE__MTR__PROCESSING__PREPROCESSOR_HPP_
