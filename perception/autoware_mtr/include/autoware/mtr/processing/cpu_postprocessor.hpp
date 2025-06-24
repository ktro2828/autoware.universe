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

#ifndef AUTOWARE__MTR__PROCESSING__CPU_POSTPROCESSOR_HPP_
#define AUTOWARE__MTR__PROCESSING__CPU_POSTPROCESSOR_HPP_

#include "autoware/mtr/processing/processor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::mtr::processing
{
/**
 * @brief A class for postprocessing on CPU.
 */
class CpuPostProcessor : public IPostProcessor
{
public:
  /**
   * @brief Construct a new PostProcessor object.
   *
   * @param num_mode Number of modes (M).
   * @param num_future Number of predicted future timestamps (Tf).
   * @param score_threshold Score threshold [0, 1].
   */
  CpuPostProcessor(size_t num_mode, size_t num_future, double score_threshold);

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
  output_type process(
    const std::vector<float> & scores, const std::vector<float> & trajectories,
    const std::vector<std::string> & agent_ids, const Header & header,
    const std::unordered_map<std::string, TrackedObject> & tracked_object_map) const;
};
}  // namespace autoware::mtr::processing
#endif  // AUTOWARE__MTR__PROCESSING__CPU_POSTPROCESSOR_HPP_
