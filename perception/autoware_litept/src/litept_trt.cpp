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

#include "autoware/litept/litept_trt.hpp"

#include "autoware/litept/preprocess/point_type.hpp"

#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware/tensorrt_common/profiler.hpp>
#include <autoware/tensorrt_common/utils.hpp>

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::litept
{
namespace
{
sensor_msgs::msg::PointField make_point_field(
  const std::string & name, int offset, int datatype, int count)
{
  return sensor_msgs::build<sensor_msgs::msg::PointField>()
    .name(name)
    .offset(offset)
    .datatype(datatype)
    .count(count);
}
}  // namespace

LitePTTRT::LitePTTRT(
  const tensorrt_common::TrtCommonConfig & trt_config, const LitePTConfig & config)
: config_(config),
  point_fields_(
    {make_point_field("x", 0, sensor_msgs::msg::PointField::FLOAT32, 1),
     make_point_field("y", 4, sensor_msgs::msg::PointField::FLOAT32, 1),
     make_point_field("z", 8, sensor_msgs::msg::PointField::FLOAT32, 1),
     make_point_field("rgb", 12, sensor_msgs::msg::PointField::FLOAT32, 1)})
{
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
  this->initialize_resources();
  this->initialize_network(trt_config);

  preprocessor_ = std::make_unique<Preprocessor>(config_, stream_);
  postprocessor_ = std::make_unique<Postprocessor>(config_, stream_);
}

LitePTTRT::~LitePTTRT()
{
  if (stream_) {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }
}

void LitePTTRT::initialize_resources()
{
  grid_coord_d_ = cuda_utils::make_unique<std::int64_t[]>(config_.num_voxel.max * 3);
  feat_d_ = cuda_utils::make_unique<float[]>(config_.num_voxel.max * 4);
  // serialized_code_d_ = cuda_utils::make_unique<std::int64_t[]>(config_.num_voxel.max * 2);
  pred_label_d_ = cuda_utils::make_unique<std::int64_t[]>(config_.num_voxel.max);
  pred_score_d_ = cuda_utils::make_unique<float[]>(config_.num_voxel.max * config_.num_classes);

  this->allocate_messages();
}

void LitePTTRT::allocate_messages()
{
  if (!pointcloud_) {
    pointcloud_ = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    pointcloud_->height = 1;
    pointcloud_->width = config_.num_voxel.max;
    pointcloud_->fields = point_fields_;
    pointcloud_->is_bigendian = false;
    pointcloud_->is_dense = false;
    pointcloud_->point_step = static_cast<std::uint32_t>(point_fields_.size() * sizeof(float));
    pointcloud_->data =
      cuda_blackboard::make_unique<std::uint8_t[]>(config_.num_voxel.max * pointcloud_->point_step);
  }
}

void LitePTTRT::initialize_network(const tensorrt_common::TrtCommonConfig & trt_config)
{
  auto network_io = std::make_unique<std::vector<tensorrt_common::NetworkIO>>();

  // Inputs
  network_io->emplace_back("grid_coord", nvinfer1::Dims2{-1, 3});
  network_io->emplace_back("feat", nvinfer1::Dims2{-1, 4});

  // Outputs
  network_io->emplace_back("pred_label", nvinfer1::Dims{1, {-1}});
  network_io->emplace_back("pred_score", nvinfer1::Dims2{-1, config_.num_classes});

  auto profile_dims = std::make_unique<std::vector<tensorrt_common::ProfileDims>>();

  profile_dims->emplace_back(
    "grid_coord", nvinfer1::Dims2{config_.num_voxel.x, 3},  // min
    nvinfer1::Dims2{config_.num_voxel.y, 3},                // opt
    nvinfer1::Dims2{config_.num_voxel.z, 3});               // max

  profile_dims->emplace_back(
    "feat", nvinfer1::Dims2{config_.num_voxel.x, 4},  // min
    nvinfer1::Dims2{config_.num_voxel.y, 4},          // opt
    nvinfer1::Dims2{config_.num_voxel.z, 4});         // max

  network_ = std::make_unique<tensorrt_common::TrtCommon>(
    trt_config, std::make_shared<tensorrt_common::Profiler>(),
    std::vector<std::string>{config_.plugins_path});

  if (!network_->setup(std::move(profile_dims), std::move(network_io))) {
    throw std::runtime_error("Failed to setup TensorRT engine: " + trt_config.onnx_path.string());
  }

  network_->setTensorAddress("grid_coord", grid_coord_d_.get());
  network_->setTensorAddress("feat", feat_d_.get());
  network_->setTensorAddress("pred_label", pred_label_d_.get());
  network_->setTensorAddress("pred_score", pred_score_d_.get());
}

LitePTTRT::segment_t LitePTTRT::segment(const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg)
{
  if (auto result = preprocess(msg); !result) {
    return tl::make_unexpected("Preprocessing failed: " + result.error());
  }

  if (auto result = inference(); !result) {
    return tl::make_unexpected("Inference failed: " + result.error());
  }

  if (auto result = postprocess(msg->header); !result) {
    return tl::make_unexpected("Postprocessing failed: " + result.error());
  }

  auto output = std::move(pointcloud_);
  allocate_messages();
  return output;
}

LitePTTRT::process_t LitePTTRT::preprocess(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg)
{
  const auto num_points = msg->height * msg->width;
  if (num_points == 0) {
    // TODO(ktro2828): Should publish empty points?
    return tl::make_unexpected("Empty points");
  }

  num_voxels_ = preprocessor_->process(
    reinterpret_cast<InputPointType *>(msg->data.get()), num_points, grid_coord_d_.get(),
    feat_d_.get());

  if (num_voxels_ < config_.num_voxel.min || num_voxels_ > config_.num_voxel.max) {
    return tl::make_unexpected(
      "Too few or too many voxels: " + std::to_string(num_voxels_) +
      " (min: " + std::to_string(config_.num_voxel.min) +
      ", max: " + std::to_string(config_.num_voxel.max) + ")");
  }

  network_->setInputShape("grid_coord", nvinfer1::Dims2{num_voxels_, 3});
  network_->setInputShape("feat", nvinfer1::Dims2{num_voxels_, 4});

  return std::monostate();
}

LitePTTRT::process_t LitePTTRT::inference()
{
  auto status = network_->enqueueV3(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  if (!status) {
    return tl::make_unexpected("Failure to run enqueueV3()");
  } else {
    return std::monostate();
  }
}

LitePTTRT::process_t LitePTTRT::postprocess(const std_msgs::msg::Header & header)
{
  postprocessor_->process(
    feat_d_.get(), pred_label_d_.get(), reinterpret_cast<float *>(pointcloud_->data.get()),
    num_voxels_);

  pointcloud_->header = header;
  pointcloud_->width = num_voxels_;
  return std::monostate();
}
}  // namespace autoware::litept
