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

#include "autoware/mtr/archetype/fixed_queue.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace autoware::mtr::test
{

TEST(FixedQueueTest, ConstructionAndSize)
{
  archetype::FixedQueue<int> q(3);
  EXPECT_EQ(q.size(), 3u);
}

TEST(FixedQueueTest, PushBackAndOrder)
{
  archetype::FixedQueue<int> q(3);
  // Fill with default 0's (std::deque default-constructs elements)
  q.push_back(1);
  q.push_back(2);
  q.push_back(3);
  // After three pushes, queue should be [1,2,3]
  EXPECT_EQ(q.front(), 1);
  EXPECT_EQ(q.back(), 3);

  // Pushing again removes the first and adds to end: [2,3,4]
  q.push_back(4);
  EXPECT_EQ(q.front(), 2);
  EXPECT_EQ(q.back(), 4);

  q.push_back(5);  // [3,4,5]
  EXPECT_EQ(q.front(), 3);
  EXPECT_EQ(q.back(), 5);
}

TEST(FixedQueueTest, AtAndAccess)
{
  archetype::FixedQueue<std::string> q(2);
  q.push_back("hello");
  q.push_back("world");
  EXPECT_EQ(q.at(0), "hello");
  EXPECT_EQ(q.at(1), "world");
  // Overwrite front
  q.push_back("autoware");
  EXPECT_EQ(q.at(0), "world");
  EXPECT_EQ(q.at(1), "autoware");
}

TEST(FixedQueueTest, Iterator)
{
  archetype::FixedQueue<int> q(3);
  q.push_back(1);
  q.push_back(2);
  q.push_back(3);
  int sum = 0;
  for (auto it = q.begin(); it != q.end(); ++it) {
    sum += *it;
  }
  EXPECT_EQ(sum, 6);
}

}  // namespace autoware::mtr::test
