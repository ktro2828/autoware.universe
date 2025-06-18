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

#include "autoware/mtr/archetype/map.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace autoware::mtr::test
{

TEST(MapPointTest, ConstructionAndAttributes)
{
  archetype::MapPoint p1;
  EXPECT_DOUBLE_EQ(p1.x, 0.0);
  EXPECT_DOUBLE_EQ(p1.y, 0.0);
  EXPECT_DOUBLE_EQ(p1.z, 0.0);
  EXPECT_EQ(p1.label, archetype::MapLabel::UNKNOWN);

  archetype::MapPoint p2(1.2, 3.4, -5.6, archetype::MapLabel::BUS_LANE);
  EXPECT_DOUBLE_EQ(p2.x, 1.2);
  EXPECT_DOUBLE_EQ(p2.y, 3.4);
  EXPECT_DOUBLE_EQ(p2.z, -5.6);
  EXPECT_EQ(p2.label, archetype::MapLabel::BUS_LANE);
}

TEST(MapPointTest, Distance)
{
  archetype::MapPoint p(3.0, 4.0, 0.0, archetype::MapLabel::ROADWAY);
  EXPECT_DOUBLE_EQ(p.distance(), 5.0);  // sqrt(3^2 + 4^2)
}

TEST(MapPointTest, DistanceFromPoint)
{
  archetype::MapPoint p1(1.0, 2.0, 0.0, archetype::MapLabel::ROADWAY);
  archetype::MapPoint p2(4.0, 6.0, 0.0, archetype::MapLabel::ROADWAY);
  EXPECT_DOUBLE_EQ(p1.distance_from(p2), 5.0);  // sqrt((4-1)^2 + (6-2)^2)
}

TEST(MapPointTest, DistanceFromAgentState)
{
  archetype::MapPoint p(1.0, 2.0, 0.0, archetype::MapLabel::ROADWAY);
  archetype::AgentState state;
  state.x = 4.0;
  state.y = 6.0;
  EXPECT_DOUBLE_EQ(p.distance_from(state), 5.0);
}

TEST(MapPointTest, DiffNormalized)
{
  archetype::MapPoint p1(3.0, 0.0, 0.0, archetype::MapLabel::ROADWAY);
  archetype::MapPoint p2(0.0, 4.0, 0.0, archetype::MapLabel::ROADWAY);
  auto [dx, dy, dz] = p1.diff(p2, true);
  // (3, -4, 0) / 5
  EXPECT_NEAR(dx, 0.6, 1e-6);
  EXPECT_NEAR(dy, -0.8, 1e-6);
  EXPECT_NEAR(dz, 0.0, 1e-6);
}

TEST(MapPointTest, DiffNotNormalized)
{
  archetype::MapPoint p1(3.0, 0.0, 1.0, archetype::MapLabel::ROADWAY);
  archetype::MapPoint p2(0.0, 4.0, 1.0, archetype::MapLabel::ROADWAY);
  auto [dx, dy, dz] = p1.diff(p2, false);
  // (3-0, 0-4, 1-1)
  EXPECT_DOUBLE_EQ(dx, 3.0);
  EXPECT_DOUBLE_EQ(dy, -4.0);
  EXPECT_DOUBLE_EQ(dz, 0.0);
}

TEST(MapPointTest, Lerp)
{
  archetype::MapPoint p1(0.0, 0.0, 0.0, archetype::MapLabel::DASHED);
  archetype::MapPoint p2(10.0, 10.0, 10.0, archetype::MapLabel::DASHED);

  auto mid = p1.lerp(p2, 0.5);
  EXPECT_DOUBLE_EQ(mid.x, 5.0);
  EXPECT_DOUBLE_EQ(mid.y, 5.0);
  EXPECT_DOUBLE_EQ(mid.z, 5.0);
  EXPECT_EQ(mid.label, archetype::MapLabel::DASHED);

  auto end = p1.lerp(p2, 1.0);
  EXPECT_DOUBLE_EQ(end.x, 10.0);
  EXPECT_DOUBLE_EQ(end.y, 10.0);
  EXPECT_DOUBLE_EQ(end.z, 10.0);
  EXPECT_EQ(end.label, archetype::MapLabel::DASHED);
}

}  // namespace autoware::mtr::test
