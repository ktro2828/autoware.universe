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

#ifndef AUTOWARE__LITEPT__LITEPT_TRT_HPP_
#define AUTOWARE__LITEPT__LITEPT_TRT_HPP_

#include "autoware/litept/litept_config.hpp"
#include "autoware/litept/postprocess/postprocessor.hpp"
#include "autoware/litept/preprocess/preprocessor.hpp"

#include <autoware/cuda_utils/cuda_unique_ptr.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <tl_expected/expected.hpp>

#include <sensor_msgs/msg/point_field.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace autoware::litept
{
class LitePTTRT
{
public:
  using segment_t = tl::expected<cuda_blackboard::CudaPointCloud2::UniquePtr, std::string>;
  using process_t = tl::expected<std::monostate, std::string>;

  LitePTTRT(const tensorrt_common::TrtCommonConfig & trt_config, const LitePTConfig & config);
  ~LitePTTRT();

  segment_t segment(const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg);

private:
  /**
   * @brief Initialize memory resources.
   */
  void initialize_resources();

  /**
   * @brief Allocate ROS message resources.
   */
  void allocate_messages();

  /**
   * @brief Initialize TensorRT network.
   * @param trt_config TensorRT common config.
   */
  void initialize_network(const tensorrt_common::TrtCommonConfig & trt_config);

  /**
   * @brief Execute preprocessing.
   * @param msg Input message.
   * @return bool Return true if processing succeeded.
   */
  process_t preprocess(const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg);

  /**
   * @brief Execute inference.
   * @return bool Return true if inference succeeded.
   */
  process_t inference();

  /**
   * @brief Execute postprocessing.
   * @param header Input message header.
   * @return bool Return true if processing succeeded.
   */
  process_t postprocess(const std_msgs::msg::Header & header);

  const LitePTConfig config_;                                        //!< Model config
  const std::vector<sensor_msgs::msg::PointField> point_fields_;     //!< Point fields
  cuda_blackboard::CudaPointCloud2::UniquePtr pointcloud_{nullptr};  //!< Output message

  // CUDA resources
  std::int64_t num_voxels_{0};                                       //!< The number of voxels N
  cuda_utils::CudaUniquePtr<std::int64_t[]> grid_coord_d_{nullptr};  //!< Input coordinates (N, 3)
  cuda_utils::CudaUniquePtr<float[]> feat_d_{nullptr};               //!< Input features (N, 4)
  cuda_utils::CudaUniquePtr<std::int64_t[]> pred_label_d_{nullptr};  //!< Predicted labels (N,)
  cuda_utils::CudaUniquePtr<float[]> pred_score_d_{nullptr};         //!< Predicted scores (N, C)
  cudaStream_t stream_{nullptr};                                     //!< CUDA stream

  // TensorRT and processors
  std::unique_ptr<tensorrt_common::TrtCommon> network_{nullptr};  //!< TensorRT network
  std::unique_ptr<Preprocessor> preprocessor_{nullptr};           //!< Preprocessor
  std::unique_ptr<Postprocessor> postprocessor_{nullptr};         //!< Postprocessor
};
}  // namespace autoware::litept
#endif  // AUTOWARE__LITEPT__LITEPT_TRT_HPP_
