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

#include "camera_buffer.hpp"

#include <Eigen/Core>
#include <rclcpp/time.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <cv_bridge/cv_bridge.h>

#include <algorithm>
#include <mutex>
#include <utility>

namespace autoware::bevformer
{
CameraRecord::CameraRecord(
  double timestamp, cv_bridge::CvImagePtr image, const Eigen::Matrix3d & intrinsic,
  const Eigen::Matrix<double, 3, 4> & extrinsic)
: timestamp_(timestamp),
  image_(std::move(image)),
  intrinsic_(intrinsic),
  extrinsic_(extrinsic),
  projection_(intrinsic * extrinsic)
{
}

MultiCameraBuffer::MultiCameraBuffer(
  size_t anchor_id, size_t num_cameras, double timestamp_tolerance)
: anchor_id_(anchor_id), num_cameras_(num_cameras), timestamp_tolerance_(timestamp_tolerance)
{
}

void MultiCameraBuffer::emplace(
  key_type camera_id, const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
  const geometry_msgs::msg::TransformStamped & camera2ego)
{
  const auto timestamp = rclcpp::Time(image->header.stamp).seconds();
  std::lock_guard<std::mutex> lock(mutex_);
  if (const auto itr = buffer_.find(camera_id);
      itr == buffer_.end() || timestamp >= itr->second.timestamp()) {
    auto cv_image = cv_bridge::toCvCopy(image, image->encoding);

    Eigen::Matrix3d intrinsic = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(camera_info->k.data());

    Eigen::Matrix4d extrinsic_4x4 = tf2::transformToEigen(camera2ego).matrix();
    Eigen::Matrix<double, 3, 4> extrinsic = extrinsic_4x4.block<3, 4>(0, 0);

    buffer_[camera_id] =
      value_type{timestamp, cv_image, std::move(intrinsic), std::move(extrinsic)};
  }
}

bool MultiCameraBuffer::is_ready() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffer_.size() < num_cameras_ || buffer_.count(anchor_id_) == 0) {
    return false;
  }

  const auto t_anchor = buffer_.at(anchor_id_).timestamp();
  return std::all_of(buffer_.begin(), buffer_.end(), [this, t_anchor](const auto & itr) {
    return itr.first == anchor_id_ ||
           std::abs(itr.second.timestamp() - t_anchor) <= timestamp_tolerance_;
  });
}
}  // namespace autoware::bevformer
