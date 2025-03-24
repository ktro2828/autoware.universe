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

#ifndef AUTOWARE__POINTCLOUD_PREPROCESSOR__ROUTE_FILTER__ROUTE_FILTER_NODE_HPP_
#define AUTOWARE__POINTCLOUD_PREPROCESSOR__ROUTE_FILTER__ROUTE_FILTER_NODE_HPP_

#include "autoware/pointcloud_preprocessor/filter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>

#include <lanelet2_core/Forward.h>
#include <lanelet2_core/LaneletMap.h>
#include <pcl/memory.h>

namespace autoware::pointcloud_preprocessor
{
class RouteFilterComponent : public autoware::pointcloud_preprocessor::Filter
{
  using LaneletMapBin = autoware_map_msgs::msg::LaneletMapBin;
  using LaneletRoute = autoware_planning_msgs::msg::LaneletRoute;

protected:
  void filter(
    const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output) override;

private:
  /**
   * @brief Convert binary lanelet map message to pointer to lanelet map.
   *
   * @param map_bin Binary lanelet map message.
   */
  void on_map(const LaneletMapBin::ConstSharedPtr map_bin);

  /**
   * @brief Extract route lanelet map.
   *
   * @param route Route message.
   */
  void on_route(const LaneletRoute::ConstSharedPtr route);

  struct CropRegionParam
  {
    float min_x;
    float max_x;
    float min_y;
    float max_y;
  } param_;

  rclcpp::Subscription<LaneletMapBin>::SharedPtr map_subscription_;
  rclcpp::Subscription<LaneletRoute>::SharedPtr route_subscription_;

  lanelet::LaneletMapPtr lanelet_map_ptr_;
  lanelet::ConstLanelets route_lanelets_;
  double distance_;
  double y_offset_;

public:
  PCL_MAKE_ALIGNED_OPERATOR_NEW
  explicit RouteFilterComponent(const rclcpp::NodeOptions & options);
};
}  // namespace autoware::pointcloud_preprocessor
#endif  // AUTOWARE__POINTCLOUD_PREPROCESSOR__ROUTE_FILTER__ROUTE_FILTER_NODE_HPP_
