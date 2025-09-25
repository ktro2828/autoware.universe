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

#ifndef AUTOWARE__PREDICTED_PATH_POSTPROCESSOR__NODE_HPP_
#define AUTOWARE__PREDICTED_PATH_POSTPROCESSOR__NODE_HPP_

#include "autoware/predicted_path_postprocessor/debug/intermediate_publisher.hpp"
#include "autoware/predicted_path_postprocessor/processor/composable.hpp"
#include "autoware/predicted_path_postprocessor/processor/interface.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/predicted_objects.hpp>

#include <memory>
#include <string>
#include <vector>

namespace autoware::predicted_path_postprocessor
{
class PredictedPathPostprocessorNode : public rclcpp::Node
{
public:
  explicit PredictedPathPostprocessorNode(const rclcpp::NodeOptions & options);

private:
  /**
   * @brief Main callback function for predicted objects message
   * @param msg Predicted objects message
   */
  void callback(const autoware_perception_msgs::msg::PredictedObjects::ConstSharedPtr & msg);

  //////////////////////// member variables ////////////////////////
  //!< @brief Subscriber for predicted objects message
  rclcpp::Subscription<autoware_perception_msgs::msg::PredictedObjects>::SharedPtr
    object_subscription_;

  //!< @brief Publisher for processed predicted objects message
  rclcpp::Publisher<autoware_perception_msgs::msg::PredictedObjects>::SharedPtr object_publisher_;

  //!< @brief Timer for connecting parameters client
  rclcpp::TimerBase::SharedPtr timer_;

  //!< @brief Processing context
  std::unique_ptr<processor::Context> context_;

  //!< @brief Processor for predicted objects message
  std::unique_ptr<processor::ComposableProcessor> processor_;

  //!< @brief Publisher for intermediate predicted objects message
  std::unique_ptr<debug::IntermediatePublisher> intermediate_publisher_;
};
}  // namespace autoware::predicted_path_postprocessor

#endif  // AUTOWARE__PREDICTED_PATH_POSTPROCESSOR__NODE_HPP_
