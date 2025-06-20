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

#include "autoware/mtr/mtr_node.hpp"

#include "autoware/mtr/archetype/agent.hpp"
#include "autoware/mtr/conversion/tracked_object.hpp"
#include "autoware/mtr/processing/preprocessor.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>

#include <glog/logging.h>
#include <lanelet2_core/LaneletMap.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace autoware::mtr
{
MTRNode::MTRNode(const rclcpp::NodeOptions & options) : rclcpp::Node("mtr", options)
{
  google::InitGoogleLogging(get_name());
  google::InstallFailureSignalHandler();

  {
    // Subscriptions and publisher
    using std::placeholders::_1;

    objects_subscription_ = create_subscription<TrackedObjects>(
      "~/input/objects", rclcpp::QoS{1}, std::bind(&MTRNode::callback, this, _1));

    lanelet_subscription_ = create_subscription<LaneletMapBin>(
      "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
      std::bind(&MTRNode::on_map, this, _1));

    objects_publisher_ = create_publisher<PredictedObjects>("~/output/objects", rclcpp::QoS{1});
  }

  {
    // Lanelet converter
    lanelet_converter_ptr_ = std::make_unique<conversion::LaneletConverter>();
  }

  {
    // Pre-processor
    const auto label_names = declare_parameter<std::vector<std::string>>("preprocess.labels");
    const auto label_ids = archetype::to_label_ids(label_names);
    const auto max_num_target = declare_parameter<int>("preprocess.max_num_target");
    const auto max_num_agent = declare_parameter<int>("preprocess.max_num_agent");
    num_past_ = declare_parameter<int>("preprocess.num_past");
    const auto max_num_polyline = declare_parameter<int>("preprocess.max_num_polyline");
    const auto max_num_point = declare_parameter<int>("preprocess.max_num_point");
    const auto polyline_range_distance =
      declare_parameter<double>("preprocess.polyline_range_distance");
    const auto polyline_break_distance =
      declare_parameter<double>("preprocess.polyline_break_distance");

    preprocessor_ = std::make_unique<processing::PreProcessor>(
      label_ids, max_num_target, max_num_agent, num_past_, max_num_polyline, max_num_point,
      polyline_range_distance, polyline_break_distance);

    timestamp_buffer_ = std::make_unique<archetype::FixedQueue<double>>(num_past_);
  }

  {
    // Post-processor
    const auto num_mode = declare_parameter<int>("postprocess.num_mode");
    const auto num_future = declare_parameter<int>("postprocess.num_future");
    const auto score_threshold = declare_parameter<double>("postprocess.score_threshold");
    postprocessor_ =
      std::make_unique<processing::PostProcessor>(num_mode, num_future, score_threshold);
  }

  {
    // Detector
    const auto onnx_path = declare_parameter<std::string>("detector.onnx_path");
    const auto precision = declare_parameter<std::string>("detector.precision");
    const auto engine_path = declare_parameter<std::string>("detector.engine_path");
    tensorrt_common::TrtCommonConfig config(onnx_path, precision, engine_path, 1UL << 60U);
    detector_ = std::make_unique<TrtMTR>(config);
  }

  if (declare_parameter<bool>("build_only")) {
    RCLCPP_INFO(get_logger(), "TensorRT engine file is built and exit.");
    rclcpp::shutdown();
  }

  {
    // Debug processing time
    stopwatch_ptr_ =
      std::make_unique<autoware_utils_system::StopWatch<std::chrono::milliseconds>>();
    stopwatch_ptr_->tic("cyclic_time");
    stopwatch_ptr_->tic("processing_time");
    processing_time_publisher_ =
      std::make_unique<autoware_utils_debug::DebugPublisher>(this, get_name());
  }
}

void MTRNode::callback(const TrackedObjects::ConstSharedPtr objects_msg)
{
  stopwatch_ptr_->toc("processing_time", true);

  // push back the current timestamp and convert timestamps to relative time
  timestamp_buffer_->push_back(rclcpp::Time(objects_msg->header.stamp).seconds());
  std::vector<double> timestamps(timestamp_buffer_->size());
  for (size_t i = 0; i < timestamp_buffer_->size(); ++i) {
    timestamps[i] = timestamp_buffer_->at(i) - timestamp_buffer_->front();
  }

  const auto polylines_opt = lanelet_converter_ptr_->polylines();
  if (!polylines_opt) {
    RCLCPP_WARN(get_logger(), "No map points.");
    return;
  }
  const auto & polylines = polylines_opt.value();

  auto result = update_history(objects_msg);
  if (!result) {
    RCLCPP_WARN(get_logger(), "Failed to update history.");
    return;
  }
  const auto [ego_index, histories] = result.value();

  const auto [agent_tensor, map_tensor] =
    preprocessor_->process(timestamps, histories, polylines, ego_index);

  try {
    const auto [scores, trajectories] = detector_->do_inference(agent_tensor, map_tensor).unwrap();

    const auto predicted_objects = postprocessor_->process(
      scores, trajectories, agent_tensor.target_ids, objects_msg->header, tracked_object_map_);

    objects_publisher_->publish(predicted_objects);
  } catch (const archetype::MTRException & e) {
    RCLCPP_ERROR_STREAM(get_logger(), "Inference failed: " << e.what());
  }

  ////////////// DEBUG //////////////
  {
    // processing time
    const auto cyclic_time_ms = stopwatch_ptr_->toc("cyclic_time", true);
    const auto processing_time_ms = stopwatch_ptr_->toc("processing_time", true);
    processing_time_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    processing_time_publisher_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
}

void MTRNode::on_map(const LaneletMapBin::ConstSharedPtr map_msg)
{
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*map_msg, lanelet_map_ptr_);

  lanelet_converter_ptr_->convert(lanelet_map_ptr_);
}

std::optional<archetype::AgentState> MTRNode::subscribe_ego()
{
  const auto odometry_msg = odometry_subscription_.take_data();
  if (!odometry_msg) {
    return std::nullopt;
  }
  return conversion::to_agent_state(*odometry_msg);
}

std::optional<std::pair<size_t, std::vector<archetype::AgentHistory>>> MTRNode::update_history(
  const TrackedObjects::ConstSharedPtr objects_msg)
{
  std::vector<archetype::AgentHistory> histories;
  if (!objects_msg) {
    return std::nullopt;
  }

  // get current ego state
  const auto current_ego_opt = subscribe_ego();
  if (!current_ego_opt) {
    return std::nullopt;
  }
  const auto & current_ego = current_ego_opt.value();

  std::unordered_set<std::string> observed_ids;
  // update agent histories
  for (const auto & object : objects_msg->objects) {
    const auto agent_id = autoware_utils::to_hex_string(object.object_id);
    observed_ids.insert(agent_id);

    // update history with the current state
    const auto state = conversion::to_agent_state(object);
    auto [it, init] = history_map_.try_emplace(agent_id, agent_id, num_past_, state);
    if (!init) {
      it->second.update(state);
    }
    histories.emplace_back(it->second);

    // update tracked object map
    tracked_object_map_.insert_or_assign(agent_id, object);
  }

  // update ego history
  observed_ids.insert(ego_id);
  auto [ego_it, ego_init] = history_map_.try_emplace(ego_id, ego_id, num_past_, current_ego);
  if (!ego_init) {
    ego_it->second.update(current_ego);
  }
  // add ego history to histories vector
  histories.emplace_back(ego_it->second);

  // remove histories that are not observed at the current
  for (auto itr = history_map_.begin(); itr != history_map_.end();) {
    const auto & agent_id = itr->first;
    // update unobserved history with empty
    if (observed_ids.count(agent_id) == 0) {
      tracked_object_map_.erase(agent_id);
      itr = history_map_.erase(itr);
    } else {
      ++itr;
    }
  }
  return {std::make_pair(histories.size() - 1, histories)};
}
}  // namespace autoware::mtr

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::mtr::MTRNode);
