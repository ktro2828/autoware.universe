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

#ifndef CAMERA_BUFFER_HPP_
#define CAMERA_BUFFER_HPP_

#include <Eigen/Core>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace autoware::bevformer
{
class CameraRecord
{
public:
  CameraRecord() = default;
  CameraRecord(
    double timestamp, cv_bridge::CvImagePtr image, const Eigen::Matrix3d & intrinsic,
    const Eigen::Matrix<double, 3, 4> & extrinsic);

  double timestamp() const { return timestamp_; }
  const cv_bridge::CvImagePtr & image() const { return image_; }
  const Eigen::Matrix3d & intrinsic() const { return intrinsic_; }
  const Eigen::Matrix<double, 3, 4> & extrinsic() const { return extrinsic_; }
  const Eigen::Matrix<double, 3, 4> & projection() const { return projection_; }

private:
  double timestamp_;                        //!< Timestamp of the camera record
  cv_bridge::CvImagePtr image_;             //!< Camera image
  Eigen::Matrix3d intrinsic_;               //!< 3x3 camera intrinsic matrix
  Eigen::Matrix<double, 3, 4> extrinsic_;   //!< 3x4 camera extrinsic matrix
  Eigen::Matrix<double, 3, 4> projection_;  //!< 4x4 camera projection matrix P = K | [R | t]
};

class MultiCameraBuffer
{
public:
  using key_type = size_t;
  using value_type = CameraRecord;

  MultiCameraBuffer(size_t anchor_id, size_t num_cameras, double timestamp_tolerance);

  void emplace(
    key_type camera_id, const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
    const geometry_msgs::msg::TransformStamped & camera2ego);

  bool is_ready() const;

private:
  const size_t anchor_id_;            //!< Anchor camera ID
  const size_t num_cameras_;          //!< Number of cameras
  const double timestamp_tolerance_;  //!< Timestamp tolerance

  mutable std::mutex mutex_;
  std::unordered_map<key_type, value_type> buffer_;
};
}  // namespace autoware::bevformer
#endif  // CAMERA_BUFFER_HPP_
