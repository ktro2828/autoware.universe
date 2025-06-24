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

#ifndef AUTOWARE__MTR__PROCESSING__PROCESSOR_HPP_
#define AUTOWARE__MTR__PROCESSING__PROCESSOR_HPP_

#include "autoware/mtr/archetype/agent.hpp"
#include "autoware/mtr/archetype/polyline.hpp"
#include "autoware/mtr/archetype/tensor.hpp"

#include <autoware_perception_msgs/msg/predicted_object.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_perception_msgs/msg/predicted_path.hpp>
#include <autoware_perception_msgs/msg/tracked_object.hpp>
#include <std_msgs/msg/header.hpp>

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace autoware::mtr::processing
{
/**
 * @brief Interface of preprocessor.
 */
class IPreProcessor
{
public:
  using output_type = std::tuple<archetype::AgentTensor, archetype::MapTensor>;

  /**
   * @brief Construct a new IPreProcessor object.
   *
   * @param max_num_target Maximum number of predictable agents (B).
   * @param max_num_agent Maximum number of other agents (N).
   * @param num_past Number of past timestamps (T).
   * @param max_num_polyline Maximum number of polylines (K).
   * @param max_num_point Maximum number of points in a single polyline (P).
   * @param polyline_range_distance Distance threshold from ego to trim polylines [m].
   * @param polyline_break_distance Distance threshold to break two polylines [m].
   */
  IPreProcessor(
    size_t max_num_target, size_t max_num_agent, size_t num_past, size_t max_num_polyline,
    size_t max_num_point, double polyline_range_distance, double polyline_break_distance);

  /**
   * @brief Execute preprocessing.
   *
   * @param timestamps Vector of relative timestamps [sec].
   * @param histories Vector of histories for each agent.
   * @param polylines Vector of polylines.
   * @param ego_index Ego index in the histories.
   * @return Returns `archetype::AgentTensor`, `archetype::MapTensor`.
   */
  virtual output_type process(
    const std::vector<double> & timestamps, const std::vector<archetype::AgentHistory> & histories,
    const std::vector<archetype::Polyline> & polylines, size_t ego_index) const = 0;

protected:
  size_t max_num_target_;           //!< Maximum number of predictable agents (B).
  size_t max_num_agent_;            //!< Maximum number of predictable agents (N).
  size_t num_past_;                 //!< Number of past timestamps (Tp).
  size_t max_num_polyline_;         //!< Maximum number of polylines (K).
  size_t max_num_point_;            //!< Maximum number of points in a single polyline (P).
  double polyline_range_distance_;  //!< Distance threshold from ego to trim polylines [m].
  double polyline_break_distance_;  //!< Distance threshold to break two polylines [m].
};

/**
 * @brief Interface of postprocessor.
 */
class IPostProcessor
{
public:
  using Header = std_msgs::msg::Header;
  using PredictedObjects = autoware_perception_msgs::msg::PredictedObjects;
  using PredictedObject = autoware_perception_msgs::msg::PredictedObject;
  using PredictedPath = autoware_perception_msgs::msg::PredictedPath;
  using TrackedObject = autoware_perception_msgs::msg::TrackedObject;
  using output_type = PredictedObjects;

  /**
   * @brief Construct a new PostProcessor object.
   *
   * @param num_mode Number of modes (M).
   * @param num_future Number of predicted future timestamps (Tf).
   * @param score_threshold Score threshold [0, 1].
   */
  IPostProcessor(size_t num_mode, size_t num_future, double score_threshold);

  /**
   * @brief Execute postprocessing.
   *
   * @param scores Vector of scores [BxM].
   * @param trajectories Vector of predicted trajectory attributes [BxMxTfx7].
   * @param agent_ids Agent IDs [B].
   * @param header ROS message header.
   * @param tracked_object_map Hasmap of agent id and tracked object message.
   * @return Return the predicted objects.
   */
  virtual output_type process(
    const std::vector<float> & scores, const std::vector<float> & trajectories,
    const std::vector<std::string> & agent_ids, const Header & header,
    const std::unordered_map<std::string, TrackedObject> & tracked_object_map) const = 0;

protected:
  size_t num_mode_;         //!< Number of modes (M).
  size_t num_future_;       //!< Number of predicted future timestamps (Tf).
  double score_threshold_;  //!< Score threshold [0, 1].
};
}  // namespace autoware::mtr::processing
#endif  // AUTOWARE__MTR__PROCESSING__PROCESSOR_HPP_
