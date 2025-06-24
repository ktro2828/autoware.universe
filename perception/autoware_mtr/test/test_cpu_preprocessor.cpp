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

#include "autoware/mtr/processing/cpu_preprocessor.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace autoware::mtr::test
{
class CpuPreProcessorTest : public ::testing::Test
{
public:
protected:
  void SetUp() override
  {
    setup_agent();
    setup_map();
  }

  static constexpr size_t num_target = 1;
  static constexpr size_t num_agent = 3;
  static constexpr size_t num_past = 3;
  static constexpr size_t num_polyline = 1;
  static constexpr size_t num_point = 12;
  static constexpr double polyline_range_distance = 100.0;
  static constexpr double polyline_break_distance = 1.0;

  std::vector<size_t> label_ids;
  std::vector<double> timestamps;
  std::vector<archetype::AgentHistory> histories;
  std::vector<archetype::Polyline> polylines;
  size_t ego_index;

private:
  void setup_agent()
  {
    label_ids = archetype::to_label_ids({"VEHICLE", "PEDESTRIAN", "CYCLIST"});
    std::vector<std::pair<std::string, archetype::AgentLabel>> agent_id_and_label{
      {"a", archetype::AgentLabel::VEHICLE},
      {"ego", archetype::AgentLabel::VEHICLE},
      {"b", archetype::AgentLabel::PEDESTRIAN},
    };

    for (const auto & [agent_id, label] : agent_id_and_label) {
      archetype::AgentHistory history(agent_id, num_past);
      for (size_t t = 0; t < num_past; ++t) {
        history.update(archetype::AgentState(1, 1, 1, 1, 1, 1, 1, 1, 1, label, true));
      }
      histories.emplace_back(history);
    }
    for (size_t t = 0; t < num_past; ++t) {
      timestamps.push_back(t);
    }
    ego_index = 1;
  }

  void setup_map()
  {
    // lane
    archetype::Polyline lane{{
      {1.0, 1.0, 1.0, archetype::MapLabel::ROADWAY},
      {1.0, 1.0, 1.0, archetype::MapLabel::ROADWAY},
      {1.0, 1.0, 1.0, archetype::MapLabel::ROADWAY},
    }};
    polylines.emplace_back(lane);

    // left boundary

    archetype::Polyline left_boundary{{
      {1.0, 1.0, 1.0, archetype::MapLabel::SOLID},
      {1.0, 1.0, 1.0, archetype::MapLabel::SOLID},
    }};
    polylines.emplace_back(left_boundary);

    // right boundary
    archetype::Polyline right_boundary{{{1.0, 1.0, 1.0, archetype::MapLabel::DASHED}}};
    polylines.emplace_back(right_boundary);

    // crosswalk
    archetype::Polyline crosswalk{{
      {1.0, 1.0, 1.0, archetype::MapLabel::CROSSWALK},
      {1.0, 1.0, 1.0, archetype::MapLabel::CROSSWALK},
      {1.0, 1.0, 1.0, archetype::MapLabel::CROSSWALK},
      {1.0, 1.0, 1.0, archetype::MapLabel::CROSSWALK},
      {1.0, 1.0, 1.0, archetype::MapLabel::CROSSWALK},
    }};
    polylines.emplace_back(crosswalk);
  }
};

TEST_F(CpuPreProcessorTest, CpuPreProcess)
{
  processing::CpuPreProcessor preprocessor(
    label_ids, num_target, num_agent, num_past, num_polyline, num_point, polyline_range_distance,
    polyline_break_distance);

  const auto [agent_tensor, map_tensor] =
    preprocessor.process(timestamps, histories, polylines, ego_index);

  ///// check agent tensor /////
  // clang-format off
  std::vector<float> expected_in_agent{
    // B=0, N=0, T=0, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=0, T=1, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=0, T=2, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=1, T=0, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=1, T=1, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=1, T=2, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=2, T=0, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=2, T=1, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
    // B=0, N=2, T=2, D=0..20
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.3817732, -0.30116862, 0.0, 0.0,  // NOLINT
  };
  std::vector<uint8_t> expected_in_agent_mask{
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
  };
  std::vector<float> expected_in_agent_center{
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
  };
  // clang-format on

  EXPECT_EQ(agent_tensor.in_agent.size(), expected_in_agent.size());
  EXPECT_EQ(agent_tensor.in_agent_mask.size(), expected_in_agent_mask.size());
  EXPECT_EQ(agent_tensor.in_agent_center.size(), expected_in_agent_center.size());

  for (size_t i = 0; i < agent_tensor.in_agent.size(); ++i) {
    auto & answer = agent_tensor.in_agent[i];
    auto & expect = expected_in_agent[i];
    EXPECT_NEAR(answer, expect, 1e-5);
  }

  for (size_t i = 0; i < agent_tensor.in_agent_mask.size(); ++i) {
    auto & answer = agent_tensor.in_agent_mask[i];
    auto & expect = expected_in_agent_mask[i];
    EXPECT_NEAR(answer, expect, 1e-5);
  }

  for (size_t i = 0; i < agent_tensor.in_agent_center.size(); ++i) {
    auto & answer = agent_tensor.in_agent_center[i];
    auto & expect = expected_in_agent_center[i];
    EXPECT_NEAR(answer, expect, 1e-5);
  }

  ///// check map tensor /////
  // clang-format off
  std::vector<float> expected_in_map{
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };
  std::vector<uint8_t> expected_in_map_mask{
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
  };
  std::vector<float> expected_in_map_center{
    0.0, 0.0, 0.0,
  };
  // clang-format on

  EXPECT_EQ(map_tensor.in_map.size(), expected_in_map.size());
  EXPECT_EQ(map_tensor.in_map_mask.size(), expected_in_map_mask.size());
  EXPECT_EQ(map_tensor.in_map_center.size(), expected_in_map_center.size());

  for (size_t i = 0; i < map_tensor.in_map.size(); ++i) {
    auto & answer = map_tensor.in_map[i];
    auto & expect = expected_in_map[i];
    EXPECT_NEAR(answer, expect, 1e-5);
  }

  for (size_t i = 0; i < map_tensor.in_map_mask.size(); ++i) {
    auto & answer = map_tensor.in_map_mask[i];
    auto & expect = expected_in_map_mask[i];
    EXPECT_NEAR(answer, expect, 1e-5);
  }

  for (size_t i = 0; i < map_tensor.in_map_center.size(); ++i) {
    auto & answer = map_tensor.in_map_center[i];
    auto & expect = expected_in_map_center[i];
    EXPECT_NEAR(answer, expect, 1e-5);
  }
}
}  // namespace autoware::mtr::test
