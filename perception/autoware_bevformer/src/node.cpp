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

#include "node.hpp"

#include "camera_buffer.hpp"

#include <autoware_utils_rclcpp/polling_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/qos.hpp>
#include <tf2/exceptions.hpp>
#include <tf2/time.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <rmw/qos_profiles.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::bevformer
{
BEVFormerNode::BEVFormerNode(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("bevformer", node_options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  const auto camera_namespaces = declare_parameter<std::vector<std::string>>("camera_namespaces");
  const auto image_suffix = declare_parameter<std::string>("image_suffix");
  const auto camera_info_suffix = declare_parameter<std::string>("camera_info_suffix");
  const auto anchor_namespace = declare_parameter<std::string>("anchor_namespace");
  const auto use_raw = declare_parameter<bool>("use_raw");

  const auto num_cameras = camera_namespaces.size();
  image_subscriptions_.resize(num_cameras);
  camera_info_subscriptions_.resize(num_cameras);
  std::optional<size_t> anchor_id = std::nullopt;
  for (size_t i = 0; i < num_cameras; ++i) {
    const auto & ns = camera_namespaces[i];

    const auto image_topic = ns + image_suffix;
    const auto camera_info_topic = ns + camera_info_suffix;

    bool is_anchor = ns == anchor_namespace;
    if (is_anchor) {
      anchor_id = i;
    }

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    image_subscriptions_[i] = image_transport::create_subscription(
      this, image_topic,
      [this, i, is_anchor](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        this->callback(msg, i, is_anchor);
      },
      use_raw ? "raw" : "compressed", rmw_qos_profile_sensor_data, subscription_options);

    camera_info_subscriptions_[i] =
      autoware_utils_rclcpp::InterProcessPollingSubscriber<sensor_msgs::msg::CameraInfo>::
        create_subscription(this, camera_info_topic, rclcpp::SensorDataQoS{}.keep_last(1));
  }

  if (!anchor_id) {
    throw std::runtime_error("Anchor camera not found");
  }

  auto timestamp_tolerance = declare_parameter<double>("timestamp_tolerance");
  camera_buffer_ =
    std::make_unique<MultiCameraBuffer>(anchor_id.value(), num_cameras, timestamp_tolerance);
}

void BEVFormerNode::callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image, size_t camera_id, bool is_anchor)
{
  auto camera_info = camera_info_subscriptions_[camera_id]->take_data();
  if (!camera_info) {
    RCLCPP_ERROR(get_logger(), "No camera info received for camera %zu", camera_id);
    return;
  }

  geometry_msgs::msg::TransformStamped camera2ego;
  try {
    camera2ego =
      tf_buffer_.lookupTransform(camera_info->header.frame_id, "base_link", tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(get_logger(), "Failed to lookup transform: %s", ex.what());
    return;
  }

  camera_buffer_->emplace(camera_id, image, camera_info, camera2ego);

  if (!camera_buffer_->is_ready() || !is_anchor) {
    return;
  }
}
}  // namespace autoware::bevformer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::bevformer::BEVFormerNode);
