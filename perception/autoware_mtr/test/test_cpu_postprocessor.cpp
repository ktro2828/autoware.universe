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

#include "autoware/mtr/processing/cpu_postprocessor.hpp"
#include "autoware/mtr/processing/processor.hpp"

#include <autoware_utils_geometry/geometry.hpp>
#include <autoware_utils_uuid/uuid_helper.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::mtr::test
{
class CpuPostProcessorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // === scores ===
    scores = {
      0.05f,  // m=0 -> to Be Filtered
      0.15f,  // m=1
      0.80f,  // m=2
    };  // to be sorted in descending order

    // === trajectories ===
    trajectories = {
      1.0f, 0.0f, 0, 0, 0, 0, 0,  // m=0, t=0
      0.0f, 2.0f, 0, 0, 0, 0, 0,  // m=0, t=1
      1.0f, 0.0f, 0, 0, 0, 0, 0,  // m=1, t=0
      0.0f, 2.0f, 0, 0, 0, 0, 0,  // m=1, t=1
      1.0f, 0.0f, 0, 0, 0, 0, 0,  // m=2, t=0
      0.0f, 2.0f, 0, 0, 0, 0, 0,  // m=2, t=1
    };

    std::string agent_id{"agent1"};
    // === agent_ids ===
    agent_ids = {agent_id};

    // === header ===
    header.frame_id = "map";
    header.stamp.sec = 1;

    // === tracked_object_map ===
    processing::IPostProcessor::TrackedObject tracked_object;
    tracked_object.object_id = autoware_utils_uuid::generate_uuid();
    tracked_object.kinematics.pose_with_covariance.pose.position.x = 100.0;
    tracked_object.kinematics.pose_with_covariance.pose.position.y = 10.0;
    tracked_object.kinematics.pose_with_covariance.pose.position.z = 0.0;
    tracked_object.kinematics.pose_with_covariance.pose.orientation =
      autoware_utils_geometry::create_quaternion_from_yaw(M_PI / 2.0);
    tracked_object_map[agent_id] = tracked_object;
  }

  static constexpr size_t num_mode = 3;
  static constexpr size_t num_future = 2;
  static constexpr double score_threshold = 0.1;

  std::vector<float> scores;
  std::vector<float> trajectories;
  std::vector<std::string> agent_ids;
  processing::IPostProcessor::Header header;
  std::unordered_map<std::string, processing::IPostProcessor::TrackedObject> tracked_object_map;
};

TEST_F(CpuPostProcessorTest, CpuPostProcess)
{
  processing::CpuPostProcessor postprocessor(num_mode, num_future, score_threshold);

  auto output = postprocessor.process(scores, trajectories, agent_ids, header, tracked_object_map);

  ASSERT_EQ(output.objects.size(), 1);

  const auto & prediction = output.objects.front();
  ASSERT_EQ(prediction.kinematics.predicted_paths.size(), 2);

  const auto & mode1 = prediction.kinematics.predicted_paths[0];
  const auto & mode2 = prediction.kinematics.predicted_paths[1];
  EXPECT_NEAR(mode1.confidence, 0.8, 1e-2);
  EXPECT_NEAR(mode2.confidence, 0.15, 1e-2);

  EXPECT_NEAR(mode1.path[0].position.x, 100.0, 1e-6);
  EXPECT_NEAR(mode1.path[0].position.y, 10.0, 1e-6);
  EXPECT_NEAR(mode1.path[1].position.x, 100.0, 1e-6);
  EXPECT_NEAR(mode1.path[1].position.y, 11.0, 1e-6);
  EXPECT_NEAR(mode1.path[2].position.x, 98.0, 1e-6);
  EXPECT_NEAR(mode1.path[2].position.y, 10.0, 1e-6);
}
}  // namespace autoware::mtr::test
