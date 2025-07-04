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

#include "autoware/simpl/simpl_node.hpp"

#include "autoware/simpl/archetype/agent.hpp"
#include "autoware/simpl/archetype/map.hpp"
#include "autoware/simpl/archetype/polyline.hpp"
#include "autoware/simpl/conversion/tracked_object.hpp"
#include "autoware/simpl/debug/marker.hpp"
#include "autoware/simpl/processing/preprocessor.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>

#include <glog/logging.h>
#include <lanelet2_core/LaneletMap.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace autoware::simpl
{
SimplNode::SimplNode(const rclcpp::NodeOptions & options) : rclcpp::Node("simpl", options)
{
  google::InitGoogleLogging(get_name());
  google::InstallFailureSignalHandler();

  {
    // Subscriptions and publisher
    using std::placeholders::_1;

    objects_subscription_ = create_subscription<TrackedObjects>(
      "~/input/objects", rclcpp::QoS{1}, std::bind(&SimplNode::callback, this, _1));

    lanelet_subscription_ = create_subscription<LaneletMapBin>(
      "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
      std::bind(&SimplNode::on_map, this, _1));

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
    const auto max_num_agent = declare_parameter<int>("preprocess.max_num_agent");
    num_past_ = declare_parameter<int>("preprocess.num_past");
    const auto max_num_polyline = declare_parameter<int>("preprocess.max_num_polyline");
    const auto max_num_point = declare_parameter<int>("preprocess.max_num_point");
    const auto polyline_range_distance =
      declare_parameter<double>("preprocess.polyline_range_distance");
    const auto polyline_break_distance =
      declare_parameter<double>("preprocess.polyline_break_distance");

    preprocessor_ = std::make_unique<processing::PreProcessor>(
      label_ids, max_num_agent, num_past_, max_num_polyline, max_num_point, polyline_range_distance,
      polyline_break_distance);
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
    detector_ = std::make_unique<TrtSimpl>(config);
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

  {
    // Debug marker publisher
    history_marker_publisher_ =
      this->create_publisher<MarkerArray>("~/debug/histories", rclcpp::QoS{1});
    polyline_marker_publisher_ =
      this->create_publisher<MarkerArray>("~/debug/map_points", rclcpp::QoS{1});
    processed_map_marker_publisher_ =
      this->create_publisher<MarkerArray>("~/debug/processed_map", rclcpp::QoS{1});
  }
}

void SimplNode::callback(const TrackedObjects::ConstSharedPtr objects_msg)
{
  stopwatch_ptr_->toc("processing_time", true);

  const auto current_ego_opt = subscribe_ego();
  if (!current_ego_opt) {
    RCLCPP_WARN(get_logger(), "Failed to subscribe ego vehicle state.");
    return;
  }
  const auto & current_ego = current_ego_opt.value();

  const auto polylines_opt = lanelet_converter_ptr_->polylines();
  if (!polylines_opt) {
    RCLCPP_WARN(get_logger(), "No map points.");
    return;
  }
  const auto & polylines = polylines_opt.value();

  const auto histories = update_history(objects_msg);

  const auto [agent_metadata, map_metadata, rpe_tensor] =
    preprocessor_->process(histories, polylines, current_ego);

  try {
    const auto [scores, trajectories] =
      detector_->do_inference(agent_metadata.tensor, map_metadata.tensor, rpe_tensor).unwrap();

    const auto predicted_objects = postprocessor_->process(
      scores, trajectories, agent_metadata.agent_ids, objects_msg->header, tracked_object_map_);

    objects_publisher_->publish(predicted_objects);
  } catch (const archetype::SimplException & e) {
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

  {
    // debug marker
    const auto history_marker_array =
      debug::create_history_marker_array(history_map_, objects_msg->header);
    const auto polyline_marker_array =
      debug::create_polyline_marker_array(polylines, objects_msg->header);
    const auto processed_map_marker_array = debug::create_processed_map_marker_array(
      map_metadata.tensor.polylines, map_metadata.centers, map_metadata.vectors,
      objects_msg->header);
    history_marker_publisher_->publish(history_marker_array);
    polyline_marker_publisher_->publish(polyline_marker_array);
    processed_map_marker_publisher_->publish(processed_map_marker_array);
  }

  {
    /**
     * @brief save source data and preprocessed tensors.
     *
     * OUTPUT_ROOT/
     *  |----timestamp.txt
     *  |--- agent_id/
     *  |     |--- <TIMESTAMP1>.txt
     *  |     |--- <TIMESTAMP2>.txt
     *  |     ...
     *  |---- history/
     *  |     |--- <TIMESTAMP1>.csv
     *  |     |--- <TIMESTAMP2>.csv
     *  |     ...
     *  |---- polyline/
     *  |     |--- <TIMESTAMP1>.bin
     *  |     |--- <TIMESTAMP2>.bin
     *  |     ...
     *  |--- actor/
     *  |     |--- <TIMESTAMP1>.bin
     *  |     |--- <TIMESTAMP2>.bin
     *  |     ...
     *  |--- lane/
     *  |     |--- <TIMESTAMP1>.bin
     *  |     |--- <TIMESTAMP2>.bin
     *  |     ...
     *  |--- rpe/
     *  |     |--- <TIMESTAMP1>.bin
     *  |     |--- <TIMESTAMP2>.bin
     *  |     ...
     */

    const auto histories_with_ego = update_history_with_ego(objects_msg, current_ego);

    const auto & stamp = objects_msg->header.stamp;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
    std::string timestamp_str = oss.str();

    auto save_tensor = [](const float * tensor, size_t size, const std::string & filename) -> void {
      std::ofstream ofs(filename, std::ios::binary);
      ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
      ofs.write(reinterpret_cast<const char *>(tensor), size * sizeof(float));
    };

    auto save_agent_ids =
      [](const std::vector<std::string> & agent_ids, const std::string & filename) -> void {
      std::ofstream ofs(filename);
      for (const auto & id : agent_ids) {
        ofs << id << "\n";
      }
    };

    // save history as
    auto save_history = [](
                          const std::vector<archetype::AgentHistory> & histories,
                          const std::string & filename) -> void {
      auto label2string = [](archetype::AgentLabel label) -> std::string {
        switch (label) {
          case archetype::AgentLabel::VEHICLE:
            return "VEHICLE";
          case archetype::AgentLabel::PEDESTRIAN:
            return "PEDESTRIAN";
          case archetype::AgentLabel::MOTORCYCLIST:
            return "MOTORCYCLIST";
          case archetype::AgentLabel::CYCLIST:
            return "CYCLIST";
          case archetype::AgentLabel::LARGE_VEHICLE:
            return "LARGE_VEHICLE";
          case archetype::AgentLabel::UNKNOWN:
            return "UNKNOWN";
          default:
            return "UNKNOWN";
        }
      };

      std::ofstream ofs(filename);
      ofs << "agent_id,t,x,y,z,yaw,vx,vy,label,is_valid\n";
      for (const auto & h : histories) {
        size_t t = 0;
        for (const auto & s : h) {
          ofs << h.agent_id << ',' << t << ',' << s.x << ',' << s.y << ',' << s.z << ',' << s.yaw
              << ',' << s.vx << ',' << s.vy << ',' << label2string(s.label) << ','
              << (s.is_valid ? "true" : "false") << '\n';
          ++t;
        }
      }
    };

    auto save_polyline =
      [](const std::vector<archetype::Polyline> & polylines, const std::string & filename) -> void {
      auto label2string = [](archetype::MapLabel label) -> std::string {
        switch (label) {
          case archetype::MapLabel::ROADWAY:
            return "ROADWAY";
          case archetype::MapLabel::BUS_LANE:
            return "BUS_LANE";
          case archetype::MapLabel::BIKE_LANE:
            return "BIKE_LANE";
          case archetype::MapLabel::DASH_SOLID:
            return "DASH_SOLID";
          case archetype::MapLabel::DASHED:
            return "DASHED";
          case archetype::MapLabel::DOUBLE_DASH:
            return "DOUBLE_DASH";
          case archetype::MapLabel::SOLID:
            return "SOLID";
          case archetype::MapLabel::DOUBLE_SOLID:
            return "DOUBLE_SOLID";
          case archetype::MapLabel::SOLID_DASH:
            return "SOLID_DASH";
          case archetype::MapLabel::CROSSWALK:
            return "CROSSWALK";
          case archetype::MapLabel::UNKNOWN:
            return "UNKNOWN";
          default:
            return "UNKNOWN";
        }
      };

      std::ofstream ofs(filename);
      ofs << "polyline_id,x,y,z,label\n";
      for (const auto & polyline : polylines) {
        for (const auto & pt : polyline) {
          ofs << polyline.id() << ','                       // polyline_id
              << pt.x << ',' << pt.y << ',' << pt.z << ','  // xyz
              << label2string(pt.label)                     // label
              << '\n';
        }
      }
    };

    auto append_timestamp =
      [](const std::string & timestamp_str, const std::string & filename) -> void {
      std::ofstream ofs(filename, std::ios::app);
      ofs << timestamp_str << "\n";
    };

    std::string output_root = "/tmp/simpl_debug";
    // source data
    std::filesystem::create_directories(output_root + "/agent_id");
    std::filesystem::create_directories(output_root + "/history");
    std::filesystem::create_directories(output_root + "/polyline");
    // preprocessed data
    std::filesystem::create_directories(output_root + "/actor");
    std::filesystem::create_directories(output_root + "/lane");
    std::filesystem::create_directories(output_root + "/rpe");

    save_tensor(
      agent_metadata.tensor.data(), agent_metadata.tensor.size(),
      output_root + "/actor/" + timestamp_str + ".bin");

    save_tensor(
      map_metadata.tensor.data(), map_metadata.tensor.size(),
      output_root + "/lane/" + timestamp_str + ".bin");

    save_tensor(
      rpe_tensor.data(), rpe_tensor.size(), output_root + "/rpe/" + timestamp_str + ".bin");

    save_agent_ids(agent_metadata.agent_ids, output_root + "/agent_id/" + timestamp_str + ".txt");

    save_history(histories_with_ego, output_root + "/history/" + timestamp_str + ".csv");

    save_polyline(polylines, output_root + "/polyline/" + timestamp_str + ".csv");

    append_timestamp(timestamp_str, output_root + "/timestamp.txt");
  }
}

void SimplNode::on_map(const LaneletMapBin::ConstSharedPtr map_msg)
{
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*map_msg, lanelet_map_ptr_);

  lanelet_converter_ptr_->convert(lanelet_map_ptr_);
}

std::optional<archetype::AgentState> SimplNode::subscribe_ego()
{
  const auto odometry_msg = odometry_subscription_.take_data();
  if (!odometry_msg) {
    return std::nullopt;
  }
  return conversion::to_agent_state(*odometry_msg);
}

std::vector<archetype::AgentHistory> SimplNode::update_history(
  const TrackedObjects::ConstSharedPtr objects_msg)
{
  std::vector<archetype::AgentHistory> histories;
  if (!objects_msg) {
    return histories;
  }

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

  // remove histories that are not observed at the current
  for (auto itr = history_map_.begin(); itr != history_map_.end();) {
    const auto & agent_id = itr->first;
    // update unobserved history with empty
    if (std::find(observed_ids.begin(), observed_ids.end(), agent_id) == observed_ids.end()) {
      tracked_object_map_.erase(agent_id);
      itr = history_map_.erase(itr);
    } else {
      ++itr;
    }
  }
  return histories;
}

std::vector<archetype::AgentHistory> SimplNode::update_history_with_ego(
  const TrackedObjects::ConstSharedPtr objects_msg, const archetype::AgentState & current_ego)
{
  std::vector<archetype::AgentHistory> histories;
  if (!objects_msg) {
    return histories;
  }

  std::unordered_set<std::string> observed_ids;

  // update agent histories
  for (const auto & object : objects_msg->objects) {
    const auto agent_id = autoware_utils::to_hex_string(object.object_id);
    observed_ids.insert(agent_id);

    // update history with the current state
    const auto state = conversion::to_agent_state(object);
    auto [it, init] = history_map_with_ego_.try_emplace(agent_id, agent_id, num_past_, state);
    if (!init) {
      it->second.update(state);
    }
    histories.emplace_back(it->second);
  }

  static const std::string ego_id = "EGO";
  auto [it, init] = history_map_with_ego_.try_emplace(ego_id, ego_id, num_past_, current_ego);
  if (!init) {
    it->second.update(current_ego);
  }
  histories.emplace_back(it->second);
  observed_ids.insert(ego_id);

  // remove histories that are not observed at the current
  for (auto itr = history_map_with_ego_.begin(); itr != history_map_with_ego_.end();) {
    const auto & agent_id = itr->first;
    // update unobserved history with empty
    if (std::find(observed_ids.begin(), observed_ids.end(), agent_id) == observed_ids.end()) {
      itr = history_map_with_ego_.erase(itr);
    } else {
      ++itr;
    }
  }
  return histories;
}
}  // namespace autoware::simpl

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::simpl::SimplNode);
