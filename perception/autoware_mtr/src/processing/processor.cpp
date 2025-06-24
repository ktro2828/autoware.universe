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

#include "autoware/mtr/processing/processor.hpp"

namespace autoware::mtr::processing
{
IPreProcessor::IPreProcessor(
  size_t max_num_target, size_t max_num_agent, size_t num_past, size_t max_num_polyline,
  size_t max_num_point, double polyline_range_distance, double polyline_break_distance)
: max_num_target_(max_num_target),
  max_num_agent_(max_num_agent),
  num_past_(num_past),
  max_num_polyline_(max_num_polyline),
  max_num_point_(max_num_point),
  polyline_range_distance_(polyline_range_distance),
  polyline_break_distance_(polyline_break_distance)
{
}

IPostProcessor::IPostProcessor(size_t num_mode, size_t num_future, double score_threshold)
: num_mode_(num_mode), num_future_(num_future), score_threshold_(score_threshold)
{
}
}  // namespace autoware::mtr::processing
