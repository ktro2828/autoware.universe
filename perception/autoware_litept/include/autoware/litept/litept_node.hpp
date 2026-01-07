// Copyright 2026 TIER IV, Inc.
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

#ifndef AUTOWARE__LITEPT__LITEPT_NODE_HPP_
#define AUTOWARE__LITEPT__LITEPT_NODE_HPP_

#include "autoware/litept/litept_trt.hpp"

#include <autoware_utils_debug/debug_publisher.hpp>
#include <autoware_utils_debug/published_time_publisher.hpp>
#include <autoware_utils_system/stop_watch.hpp>
#include <cuda_blackboard/cuda_blackboard_publisher.hpp>
#include <cuda_blackboard/cuda_blackboard_subscriber.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <chrono>
#include <memory>

namespace autoware::litept
{
class LitePTNode : public rclcpp::Node
{
public:
  explicit LitePTNode(const rclcpp::NodeOptions & options);

private:
  void callback(const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg);

  // subscriber and publisher
  std::unique_ptr<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>
    subscriber_;
  std::unique_ptr<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>
    publisher_;

  std::unique_ptr<LitePTTRT> model_{nullptr};  //!< LitePT model

  //!< debugger
  std::unique_ptr<autoware_utils_system::StopWatch<std::chrono::milliseconds>> stop_watch_{nullptr};
  std::unique_ptr<autoware_utils_debug::DebugPublisher> debug_publisher_{nullptr};
  std::unique_ptr<autoware_utils_debug::PublishedTimePublisher> published_time_publisher_{nullptr};
};
}  // namespace autoware::litept
#endif  // AUTOWARE__LITEPT__LITEPT_NODE_HPP_
