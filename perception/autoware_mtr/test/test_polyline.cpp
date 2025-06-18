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

#include "autoware/mtr/archetype/polyline.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace autoware::mtr::test
{
// Test the construction, access and properties of Polyline.
TEST(PolylineTest, ConstructionAndAccess)
{
  std::vector<archetype::MapPoint> points;
  points.emplace_back(0.0, 0.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(2.0, 0.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(2.0, 2.0, 0.0, archetype::MapLabel::ROADWAY);

  archetype::Polyline poly(points);

  EXPECT_EQ(poly.size(), 3u);
  EXPECT_DOUBLE_EQ(poly[0].x, 0.0);
  EXPECT_DOUBLE_EQ(poly[1].x, 2.0);
  EXPECT_DOUBLE_EQ(poly[2].y, 2.0);
  EXPECT_FALSE(poly.empty());
}

// Test the center calculation using linear interpolation along the curve length
TEST(PolylineTest, CenterCalculation)
{
  // (0,0)-(2,0)-(2,2) total length=4, midpoint=2
  std::vector<archetype::MapPoint> points;
  points.emplace_back(0.0, 0.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(2.0, 0.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(2.0, 2.0, 0.0, archetype::MapLabel::ROADWAY);

  archetype::Polyline poly(points);
  const archetype::MapPoint & center = poly.center();

  // Midpoint falls at (2.0,0.0)-(2.0,2.0), at t=(2-2)/2=0; so (2.0,0.0)
  EXPECT_NEAR(center.x, 2.0, 1e-6);
  EXPECT_NEAR(center.y, 0.0, 1e-6);
}

// Test distance from polyline center to agent state
TEST(PolylineTest, DistanceFromAgentState)
{
  std::vector<archetype::MapPoint> points;
  points.emplace_back(1.0, 1.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(3.0, 1.0, 0.0, archetype::MapLabel::ROADWAY);
  archetype::Polyline poly(points);

  archetype::AgentState state(
    3.0, 1.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, archetype::AgentLabel::VEHICLE, true);

  // center is midpoint: (1.0,1.0)-(3.0,1.0)→mid=(2.0,1.0)
  EXPECT_DOUBLE_EQ(poly.distance_from(state), 1.0);
}

// Test Polyline transformation (XY translation + yaw rotation)
TEST(PolylineTest, TransformWithXYAndYaw)
{
  std::vector<archetype::MapPoint> points;
  points.emplace_back(1.0, 0.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(0.0, 1.0, 0.0, archetype::MapLabel::ROADWAY);
  archetype::Polyline poly(points);

  // 90 degree rotation (pi/2), origin at (0,0)
  archetype::Polyline tr_poly = poly.transform(0.0, 0.0, 0.0, M_PI_2);

  EXPECT_NEAR(tr_poly[0].x, 0.0, 1e-6);
  EXPECT_NEAR(tr_poly[0].y, -1.0, 1e-6);
  EXPECT_NEAR(tr_poly[1].x, 1.0, 1e-6);
  EXPECT_NEAR(tr_poly[1].y, 0.0, 1e-6);

  // Pure translation: shift x by -1, y by -2 (yaw=0)
  archetype::Polyline shifted = poly.transform(-1.0, -2.0, 0.0, 0.0);
  EXPECT_NEAR(shifted[0].x, 2.0, 1e-6);
  EXPECT_NEAR(shifted[0].y, 2.0, 1e-6);
  EXPECT_NEAR(shifted[1].x, 1.0, 1e-6);
  EXPECT_NEAR(shifted[1].y, 3.0, 1e-6);
}

// Test transform to agent state coordinates
TEST(PolylineTest, TransformToAgentState)
{
  std::vector<archetype::MapPoint> points;
  points.emplace_back(3.0, 1.0, 0.0, archetype::MapLabel::ROADWAY);
  points.emplace_back(4.0, 2.0, 0.0, archetype::MapLabel::ROADWAY);
  archetype::Polyline poly(points);

  archetype::AgentState state(
    1.0, 1.0, 0.0, 4.0, 2.0, 1.5, M_PI_2, 0.0, 0.0, archetype::AgentLabel::VEHICLE, true);

  archetype::Polyline tr_poly = poly.transform(state);

  // Should match the above calculation
  EXPECT_NEAR(tr_poly[0].x, 0.0, 1e-6);
  EXPECT_NEAR(tr_poly[0].y, -2.0, 1e-6);
  EXPECT_NEAR(tr_poly[1].x, 1.0, 1e-6);
  EXPECT_NEAR(tr_poly[1].y, -3.0, 1e-6);
}

// Test empty polyline edge cases
TEST(PolylineTest, EmptyPolyline)
{
  archetype::Polyline poly;
  EXPECT_TRUE(poly.empty());
  EXPECT_EQ(poly.size(), 0u);

  // Center should be (0,0,0)
  const archetype::MapPoint & c = poly.center();
  EXPECT_DOUBLE_EQ(c.x, 0.0);
  EXPECT_DOUBLE_EQ(c.y, 0.0);
  EXPECT_DOUBLE_EQ(c.z, 0.0);
}

// Test trim_neighbors function for different tolerance
TEST(PolylineTest, TrimNeighbors)
{
  std::vector<archetype::Polyline> polylines;
  // (0,0)-(0,1): center at (0,0.5), (10,0)-(10,1): center at (10,0.5), (0,20)-(1,20): center at
  // (0.5,20)
  polylines.emplace_back(std::vector<archetype::MapPoint>{
    archetype::MapPoint(0.0, 0.0, 0.0, archetype::MapLabel::ROADWAY),
    archetype::MapPoint(0.0, 1.0, 0.0, archetype::MapLabel::ROADWAY)});
  polylines.emplace_back(std::vector<archetype::MapPoint>{
    archetype::MapPoint(10.0, 0.0, 0.0, archetype::MapLabel::ROADWAY),
    archetype::MapPoint(10.0, 1.0, 0.0, archetype::MapLabel::ROADWAY)});
  polylines.emplace_back(std::vector<archetype::MapPoint>{
    archetype::MapPoint(0.0, 20.0, 0.0, archetype::MapLabel::ROADWAY),
    archetype::MapPoint(1.0, 20.0, 0.0, archetype::MapLabel::ROADWAY)});

  archetype::AgentState state(
    0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, archetype::AgentLabel::VEHICLE, true);

  // Only the first polyline center is within 5 units from (0,0)
  std::vector<archetype::Polyline> trimmed = archetype::trim_neighbors(polylines, state, 5.0);
  ASSERT_EQ(trimmed.size(), 1u);
  EXPECT_NEAR(trimmed[0].center().x, 0.0, 1e-6);

  // With tolerance 15, both first and second polyline are within range
  trimmed = archetype::trim_neighbors(polylines, state, 15.0);
  ASSERT_EQ(trimmed.size(), 2u);
  EXPECT_NEAR(trimmed[1].center().x, 10.0, 1e-6);
}
}  // namespace autoware::mtr::test
