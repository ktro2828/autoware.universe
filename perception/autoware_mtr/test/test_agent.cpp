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

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace autoware::mtr::test
{

TEST(AgentStateTest, DistanceFrom)
{
  archetype::AgentState state1;
  state1.x = 1.0;
  state1.y = 2.0;

  archetype::AgentState state2;
  state2.x = 4.0;
  state2.y = 6.0;

  double distance = state1.distance_from(state2);
  EXPECT_NEAR(distance, 5.0, 1e-5);  // sqrt(3^2 + 4^2)
}

TEST(AgentStateTest, TransformCase1)
{
  // Transform state1 to the coordinate frame of to_state
  archetype::AgentState state1;
  state1.x = 1.0;
  state1.y = 2.0;
  state1.z = 0.0;
  state1.yaw = M_PI / 4;  // 45 degrees
  state1.vx = 1.0;
  state1.vy = 0.0;

  archetype::AgentState to_state;
  to_state.x = 0.0;
  to_state.y = 0.0;
  to_state.z = 0.0;
  to_state.yaw = M_PI / 4;  // same orientation

  auto result = state1.transform(to_state);

  // x': (1-0)*cos(45) + (2-0)*sin(45) = sqrt(2)/2 + 2*sqrt(2)/2 = 1.5*sqrt(2) ≈ 2.1213
  // y': -(1-0)*sin(45) + (2-0)*cos(45) = -sqrt(2)/2 + 2*sqrt(2)/2 = sqrt(2)/2 ≈ 0.7071
  double root2 = std::sqrt(2);
  EXPECT_NEAR(result.x, 1.5 * root2, 1e-5);
  EXPECT_NEAR(result.y, 0.5 * root2, 1e-5);
  EXPECT_NEAR(result.z, 0.0, 1e-5);
  EXPECT_NEAR(result.yaw, 0.0, 1e-5);
  EXPECT_NEAR(result.vx, root2 / 2, 1e-5);  // vx=1.0, vy=0.0, yaw=45°
  EXPECT_NEAR(result.vy, -root2 / 2, 1e-5);
}

TEST(AgentStateTest, TransformCase2)
{
  archetype::AgentState state1;
  state1.x = 2.0;
  state1.y = 3.0;
  state1.z = 0.0;
  state1.yaw = M_PI / 2;  // 90 degrees
  state1.vx = 0.0;
  state1.vy = 1.0;

  archetype::AgentState to_state;
  to_state.x = 1.0;
  to_state.y = 1.0;
  to_state.z = 0.0;
  to_state.yaw = M_PI / 2;  // same orientation

  auto result = state1.transform(to_state);

  // x': (2-1)*cos(90) + (3-1)*sin(90) = 0 + 2*1 = 2
  // y': -(2-1)*sin(90) + (3-1)*cos(90) = -1*1 + 2*0 = -1
  EXPECT_NEAR(result.x, 2.0, 1e-5);
  EXPECT_NEAR(result.y, -1.0, 1e-5);
  EXPECT_NEAR(result.z, 0.0, 1e-5);
  EXPECT_NEAR(result.yaw, 0.0, 1e-5);
  // vx=0, vy=1 → vx'=-vy*sin+vx*cos = -1*1+0*0= -1 (vx), vy'=vy*cos+vx*sin=1*0+0*1=0
  EXPECT_NEAR(result.vx, 1.0, 1e-5);
  EXPECT_NEAR(result.vy, 0.0, 1e-5);
}

TEST(AgentHistoryTest, Transform)
{
  // Transform history from one coordinate frame to another
  archetype::AgentState s1(
    1.0, 2.0, 0.0, 4.0, 2.0, 1.5, M_PI / 4, 1.0, 0.5, archetype::AgentLabel::VEHICLE, true);
  archetype::AgentState s2(
    3.0, 4.0, 0.0, 4.0, 2.0, 1.5, M_PI / 4, 2.0, 1.5, archetype::AgentLabel::VEHICLE, true);

  archetype::AgentHistory history("test_agent", 2);
  history.update(s1);
  history.update(s2);

  archetype::AgentState to_state(
    1.0, 2.0, 0.0, 4.0, 2.0, 1.5, M_PI / 4, 1.0, 0.5, archetype::AgentLabel::VEHICLE, true);

  auto result = history.transform(to_state);

  EXPECT_EQ(result.agent_id, "test_agent");
  EXPECT_EQ(result.size(), 2u);

  EXPECT_NEAR(result.at(0).x, 0.0, 1e-5);
  EXPECT_NEAR(result.at(0).y, 0.0, 1e-5);

  double dx = 3.0 - 1.0;
  double dy = 4.0 - 2.0;
  double theta = M_PI / 4;
  double x2 = dx * std::cos(theta) + dy * std::sin(theta);
  double y2 = -dx * std::sin(theta) + dy * std::cos(theta);
  EXPECT_NEAR(result.at(1).x, x2, 1e-5);
  EXPECT_NEAR(result.at(1).y, y2, 1e-5);
}

TEST(LabelTest, ToLabelIds)
{
  std::vector<std::string> label_names = {"VEHICLE", "PEDESTRIAN",    "MOTORCYCLIST",
                                          "CYCLIST", "LARGE_VEHICLE", "UNKNOWN"};
  auto label_ids = archetype::to_label_ids(label_names);

  EXPECT_EQ(label_ids.size(), 6u);
  EXPECT_EQ(label_ids[0], static_cast<size_t>(archetype::AgentLabel::VEHICLE));
  EXPECT_EQ(label_ids[1], static_cast<size_t>(archetype::AgentLabel::PEDESTRIAN));
  EXPECT_EQ(label_ids[2], static_cast<size_t>(archetype::AgentLabel::MOTORCYCLIST));
  EXPECT_EQ(label_ids[3], static_cast<size_t>(archetype::AgentLabel::CYCLIST));
  EXPECT_EQ(label_ids[4], static_cast<size_t>(archetype::AgentLabel::LARGE_VEHICLE));
  EXPECT_EQ(label_ids[5], static_cast<size_t>(archetype::AgentLabel::UNKNOWN));
}

TEST(LabelTest, ToLabelIdsInvalid)
{
  std::vector<std::string> invalid_label_names = {"VEHICLE", "INVALID_LABEL"};
  EXPECT_THROW(archetype::to_label_ids(invalid_label_names), archetype::MTRException);
}

TEST(NeighborTest, TrimNeighborIndices)
{
  std::vector<archetype::AgentHistory> histories;
  // ego at (0,0), two neighbors (0,4) and (3,0)
  histories.emplace_back(
    "ego", 1,
    archetype::AgentState(0, 0, 0, 4, 2, 1, 0, 0, 0, archetype::AgentLabel::VEHICLE, true));
  histories.emplace_back(
    "a", 1, archetype::AgentState(0, 4, 0, 4, 2, 1, 0, 0, 0, archetype::AgentLabel::CYCLIST, true));
  histories.emplace_back(
    "b", 1,
    archetype::AgentState(3, 0, 0, 4, 2, 1, 0, 0, 0, archetype::AgentLabel::PEDESTRIAN, true));

  auto indices_k1 = archetype::trim_neighbor_indices(histories, 0, 1);
  EXPECT_EQ(indices_k1.size(), 1u);
  EXPECT_EQ(indices_k1[0], 2);

  // if top_k > num_hist -> num_idx == num_hist
  auto indices_k3 = archetype::trim_neighbor_indices(histories, 0, 3);
  EXPECT_EQ(indices_k3.size(), 2u);
  EXPECT_EQ(indices_k3[0], 2);
  EXPECT_EQ(indices_k3[1], 1);
}

}  // namespace autoware::mtr::test
