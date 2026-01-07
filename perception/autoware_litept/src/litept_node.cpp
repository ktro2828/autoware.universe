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

#include "autoware/litept/litept_node.hpp"

#include "autoware/litept/litept_config.hpp"

#include <autoware/tensorrt_common/utils.hpp>

#include <autoware_internal_debug_msgs/msg/float64_stamped.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::litept
{
namespace
{
static constexpr char const * kCYCLIC_TIME_KEY{"cyclic"};
static constexpr char const * kPROCESSING_TIME_KEY{"processing/total"};

/**
 * @brief Convert double vector to float vector.
 */
std::vector<float> to_float_vector(const std::vector<double> & v)
{
  return std::vector<float>(v.begin(), v.end());
}
}  // namespace

LitePTNode::LitePTNode(const rclcpp::NodeOptions & options) : Node("litept_node", options)
{
  // TensorRT parameters
  auto plugins_path = this->declare_parameter<std::string>("plugins_path");
  auto onnx_path = this->declare_parameter<std::string>("onnx_path");
  auto engine_path = this->declare_parameter<std::string>("engine_path");
  auto precision = this->declare_parameter<std::string>("precision");

  auto trt_config = tensorrt_common::TrtCommonConfig(
    std::move(onnx_path), std::move(precision), std::move(engine_path), 1ULL << 33U);

  // preprocess
  auto cloud_capacity = this->declare_parameter<int>("cloud_capacity");
  auto num_voxels = this->declare_parameter<std::vector<int>>("num_voxels");
  auto point_cloud_ranges =
    to_float_vector(this->declare_parameter<std::vector<double>>("point_cloud_ranges"));
  auto voxel_sizes = to_float_vector(this->declare_parameter<std::vector<double>>("voxel_sizes"));
  auto class_names = this->declare_parameter<std::vector<std::string>>("class_names");

  auto litpet_config = LitePTConfig(
    std::move(plugins_path), std::move(cloud_capacity), std::move(num_voxels),
    std::move(point_cloud_ranges), std::move(voxel_sizes), std::move(class_names));

  model_ = std::make_unique<LitePTTRT>(std::move(trt_config), std::move(litpet_config));

  // subscriber and publisher
  {
    subscriber_ =
      std::make_unique<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
        *this, "~/input/pointcloud",
        [this](const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg) {
          if (stop_watch_) {
            stop_watch_->toc(kPROCESSING_TIME_KEY, true);
          }
          // run main callback
          callback(msg);

          if (stop_watch_ && debug_publisher_) {
            using autoware_internal_debug_msgs::msg::Float64Stamped;

            const double cyclic_time_ms = stop_watch_->toc(kCYCLIC_TIME_KEY, true);
            const double processing_time_ms = stop_watch_->toc(kPROCESSING_TIME_KEY, true);
            const double latency_ms =
              std::chrono::duration<double, std::milli>(
                std::chrono::nanoseconds(
                  (this->get_clock()->now() - msg->header.stamp).nanoseconds()))
                .count();

            debug_publisher_->publish<Float64Stamped>("debug/cyclic_time_ms", cyclic_time_ms);
            debug_publisher_->publish<Float64Stamped>(
              "debug/processing_time_ms", processing_time_ms);
            debug_publisher_->publish<Float64Stamped>("debug/latency_ms", latency_ms);
          }
        });

    publisher_ =
      std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
        *this, "~/output/pointcloud");
  }

  // debugger
  {
    using autoware_utils_debug::DebugPublisher;
    using autoware_utils_debug::PublishedTimePublisher;
    using autoware_utils_system::StopWatch;

    stop_watch_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, this->get_name());
    published_time_publisher_ = std::make_unique<PublishedTimePublisher>(this);

    stop_watch_->tic(kCYCLIC_TIME_KEY);
    stop_watch_->tic(kPROCESSING_TIME_KEY);
  }

  // terminate node if build_only=true
  if (this->declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }
}

void LitePTNode::callback(const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg)
{
  if (auto result = model_->segment(msg); !result) {
    RCLCPP_ERROR_STREAM(get_logger(), result.error());
  } else {
    publisher_->publish(std::move(result.value()));
  }
}
}  // namespace autoware::litept

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::litept::LitePTNode)
