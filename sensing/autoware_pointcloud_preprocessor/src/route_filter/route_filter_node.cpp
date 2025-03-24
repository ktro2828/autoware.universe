// Copyright 2025 Tier IV, Inc.
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

#include "autoware/pointcloud_preprocessor/route_filter/route_filter_node.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <rclcpp/qos.hpp>

#include <Eigen/src/Core/Matrix.h>
#include <lanelet2_core/Exceptions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>

namespace autoware::pointcloud_preprocessor
{
RouteFilterComponent::RouteFilterComponent(const rclcpp::NodeOptions & options)
: Filter("RouteFilter", options)
{
  {
    // subscriptions
    using std::placeholders::_1;

    map_subscription_ = create_subscription<LaneletMapBin>(
      "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
      std::bind(&RouteFilterComponent::on_map, this, _1));

    route_subscription_ = create_subscription<LaneletRoute>(
      "~/input/route", rclcpp::QoS{1}.transient_local(),
      std::bind(&RouteFilterComponent::on_route, this, _1));
  }

  distance_ = declare_parameter<double>("distance");
}

void RouteFilterComponent::filter(
  const PointCloud2ConstPtr & input, const IndicesPtr &, PointCloud2 & output)
{
  const int x_offset = input->fields[pcl::getFieldIndex(*input, "x")].offset;
  const int y_offset = input->fields[pcl::getFieldIndex(*input, "y")].offset;
  const int z_offset = input->fields[pcl::getFieldIndex(*input, "z")].offset;

  output.data.resize(input->data.size());
  size_t output_size = 0;
  for (size_t global_offset = 0; global_offset + input->point_step <= input->data.size();
       global_offset += input->point_step) {
    Eigen::Vector4f point;
    std::memcpy(&point[0], &input->data[global_offset + x_offset], sizeof(float));
    std::memcpy(&point[1], &input->data[global_offset + y_offset], sizeof(float));
    std::memcpy(&point[2], &input->data[global_offset + z_offset], sizeof(float));
    point[3] = 1;

    bool is_point_inside = point[0] > param_.min_x && point[0] < param_.max_x &&
                           point[1] > param_.min_y && point[1] < param_.max_y;
    if (is_point_inside) {
      memcpy(&output.data[output_size], &input->data[global_offset], input->point_step);
      output_size += input->point_step;
    }
  }
  output.data.resize(output_size);
  output.height = 1;
  output.fields = input->fields;
  output.is_bigendian = input->is_bigendian;
  output.point_step = input->point_step;
  output.is_dense = input->is_dense;
  output.width = static_cast<uint32_t>(output.data.size() / output.height / output.point_step);
  output.row_step = static_cast<uint32_t>(output.data.size() / output.height);
}

void RouteFilterComponent::on_map(const LaneletMapBin::ConstSharedPtr map_bin)
{
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*map_bin, lanelet_map_ptr_);
}

void RouteFilterComponent::on_route(const LaneletRoute::ConstSharedPtr route)
{
  if (!lanelet_map_ptr_) {
    RCLCPP_WARN(get_logger(), "Lanelet map has not been received yet.");
    return;
  }

  route_lanelets_.clear();
  for (const auto & segment : route->segments) {
    for (const auto & primitive : segment.primitives) {
      try {
        route_lanelets_.emplace_back(lanelet_map_ptr_->laneletLayer.get(primitive.id));
      } catch (const lanelet::NoSuchPrimitiveError & e) {
        RCLCPP_ERROR(get_logger(), "%s", e.what());
        return;
      }
    }
  }
}
}  // namespace autoware::pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::pointcloud_preprocessor::RouteFilterComponent)
