// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NODE_HPP_
#define NODE_HPP_

#include "camera_buffer.hpp"

#include <autoware_utils_rclcpp/polling_subscriber.hpp>
#include <image_transport/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <sensor_msgs/msg/camera_info.hpp>

#include <memory>
#include <vector>

namespace autoware::bevformer
{
class BEVFormerNode : public rclcpp::Node
{
public:
  explicit BEVFormerNode(const rclcpp::NodeOptions & node_options);

private:
  void callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg, size_t camera_id, bool is_anchor);

  std::vector<image_transport::Subscriber> image_subscriptions_;
  std::vector<
    autoware_utils_rclcpp::InterProcessPollingSubscriber<sensor_msgs::msg::CameraInfo>::SharedPtr>
    camera_info_subscriptions_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::unique_ptr<MultiCameraBuffer> camera_buffer_;
};
}  // namespace autoware::bevformer
#endif  // NODE_HPP_
