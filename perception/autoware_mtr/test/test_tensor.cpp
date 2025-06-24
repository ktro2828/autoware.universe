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

#include "autoware/mtr/archetype/tensor.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace autoware::mtr::test
{
TEST(TensorTest, AgentTensorValid)
{
  // Create AgentTensor with valid sizes
  size_t num_target = 2;
  size_t num_agent = 3;
  size_t num_past = 4;
  size_t num_attribute = 5;
  std::vector<float> in_agent(num_target * num_agent * num_past * num_attribute, 1.0f);
  std::vector<uint8_t> in_agent_mask(num_target * num_agent * num_past, 1);
  std::vector<float> in_agent_center(num_target * num_agent * 3, 0.0f);
  std::vector<int> target_indices{0, 1};
  std::vector<int> target_labels{0, 1};
  std::vector<std::string> target_ids{"id0", "id1"};

  archetype::AgentTensor tensor(
    in_agent, in_agent_mask, in_agent_center, target_indices, target_labels, target_ids, num_target,
    num_agent, num_past, num_attribute);

  EXPECT_EQ(tensor.in_agent.size(), num_target * num_agent * num_past * num_attribute);
  EXPECT_EQ(tensor.in_agent_mask.size(), num_target * num_agent * num_past);
  EXPECT_EQ(tensor.in_agent_center.size(), num_target * num_agent * 3);
  EXPECT_EQ(tensor.target_indices.size(), target_indices.size());
  EXPECT_EQ(tensor.target_labels.size(), target_labels.size());
  EXPECT_EQ(tensor.target_ids.size(), target_ids.size());
}

TEST(TensorTest, AgentTensorInvalidInAgent)
{
  // Invalid in_agent size
  size_t num_target = 1, num_agent = 2, num_past = 3, num_attribute = 4;
  std::vector<float> in_agent(5, 1.0f);  // Wrong size!
  std::vector<uint8_t> in_agent_mask(num_target * num_agent, 1);
  std::vector<float> in_agent_center(num_target * num_agent * 3, 0.0f);
  std::vector<int> target_indices{0};
  std::vector<int> target_labels{0};
  std::vector<std::string> target_ids{"id0"};
  EXPECT_THROW(
    archetype::AgentTensor(
      in_agent, in_agent_mask, in_agent_center, target_indices, target_labels, target_ids,
      num_target, num_agent, num_past, num_attribute),
    archetype::MTRException);
}

TEST(TensorTest, AgentTensorInvalidMask)
{
  // Invalid in_agent_mask size
  size_t num_target = 2, num_agent = 2, num_past = 2, num_attribute = 2;
  std::vector<float> in_agent(num_target * num_agent * num_past * num_attribute, 1.0f);
  std::vector<uint8_t> in_agent_mask(1, 1);  // Wrong size!
  std::vector<float> in_agent_center(num_target * num_agent * 3, 0.0f);
  std::vector<int> target_indices{0, 1};
  std::vector<int> target_labels{0, 1};
  std::vector<std::string> target_ids{"id0", "id1"};
  EXPECT_THROW(
    archetype::AgentTensor(
      in_agent, in_agent_mask, in_agent_center, target_indices, target_labels, target_ids,
      num_target, num_agent, num_past, num_attribute),
    archetype::MTRException);
}

TEST(TensorTest, AgentTensorInvalidCenter)
{
  // Invalid in_agent_center size
  size_t num_target = 1, num_agent = 1, num_past = 1, num_attribute = 1;
  std::vector<float> in_agent(1, 1.0f);
  std::vector<uint8_t> in_agent_mask(1, 1);
  std::vector<float> in_agent_center(2, 0.0f);  // Wrong size!
  std::vector<int> target_indices{0};
  std::vector<int> target_labels{0};
  std::vector<std::string> target_ids{"id0"};
  EXPECT_THROW(
    archetype::AgentTensor(
      in_agent, in_agent_mask, in_agent_center, target_indices, target_labels, target_ids,
      num_target, num_agent, num_past, num_attribute),
    archetype::MTRException);
}

TEST(TensorTest, MapTensorValid)
{
  // Create MapTensor with valid sizes
  size_t num_target = 2, num_polyline = 3, num_point = 4, num_attribute = 5;
  std::vector<float> in_map(num_target * num_polyline * num_point * num_attribute, 1.0f);
  std::vector<uint8_t> in_map_mask(num_target * num_polyline * num_point, 1);
  std::vector<float> in_map_center(num_target * num_polyline * 3, 0.0f);

  archetype::MapTensor tensor(
    in_map, in_map_mask, in_map_center, num_target, num_polyline, num_point, num_attribute);

  EXPECT_EQ(tensor.in_map.size(), num_target * num_polyline * num_point * num_attribute);
  EXPECT_EQ(tensor.in_map_mask.size(), num_target * num_polyline * num_point);
  EXPECT_EQ(tensor.in_map_center.size(), num_target * num_polyline * 3);
}

TEST(TensorTest, MapTensorInvalidInMap)
{
  // Invalid in_map size
  size_t num_target = 2, num_polyline = 3, num_point = 4, num_attribute = 5;
  std::vector<float> in_map(1, 1.0f);  // Wrong size!
  std::vector<uint8_t> in_map_mask(num_target * num_polyline, 1);
  std::vector<float> in_map_center(num_target * num_polyline * 3, 0.0f);
  EXPECT_THROW(
    archetype::MapTensor(
      in_map, in_map_mask, in_map_center, num_target, num_polyline, num_point, num_attribute),
    archetype::MTRException);
}

TEST(TensorTest, MapTensorInvalidMask)
{
  // Invalid in_map_mask size
  size_t num_target = 2, num_polyline = 3, num_point = 4, num_attribute = 5;
  std::vector<float> in_map(num_target * num_polyline * num_point * num_attribute, 1.0f);
  std::vector<uint8_t> in_map_mask(1, 1);  // Wrong size!
  std::vector<float> in_map_center(num_target * num_polyline * 3, 0.0f);
  EXPECT_THROW(
    archetype::MapTensor(
      in_map, in_map_mask, in_map_center, num_target, num_polyline, num_point, num_attribute),
    archetype::MTRException);
}

TEST(TensorTest, MapTensorInvalidCenter)
{
  // Invalid in_map_center size
  size_t num_target = 2, num_polyline = 3, num_point = 4, num_attribute = 5;
  std::vector<float> in_map(num_target * num_polyline * num_point * num_attribute, 1.0f);
  std::vector<uint8_t> in_map_mask(num_target * num_polyline, 1);
  std::vector<float> in_map_center(1, 0.0f);  // Wrong size!
  EXPECT_THROW(
    archetype::MapTensor(
      in_map, in_map_mask, in_map_center, num_target, num_polyline, num_point, num_attribute),
    archetype::MTRException);
}
}  // namespace autoware::mtr::test
