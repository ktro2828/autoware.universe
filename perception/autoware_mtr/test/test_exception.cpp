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

#include "autoware/mtr/archetype/exception.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace autoware::mtr::test
{

TEST(MTRExceptionTest, TensorRTError)
{
  archetype::MTRException e(archetype::MTRError_t::TensorRT, "TensorRT failure");
  EXPECT_STREQ(e.what(), "[TensorRT]: TensorRT failure");
}

TEST(MTRExceptionTest, CudaError)
{
  archetype::MTRException e(archetype::MTRError_t::Cuda, "CUDA memory error");
  EXPECT_STREQ(e.what(), "[CUDA]: CUDA memory error");
}

TEST(MTRExceptionTest, InvalidValueError)
{
  archetype::MTRException e(archetype::MTRError_t::InvalidValue, "Invalid input value");
  EXPECT_STREQ(e.what(), "[InvalidValue]: Invalid input value");
}

TEST(MTRExceptionTest, UnknownError)
{
  archetype::MTRException e(archetype::MTRError_t::Unknown, "Unknown error occurred");
  EXPECT_STREQ(e.what(), "[UNKNOWN]: Unknown error occurred");
}

TEST(MTRExceptionTest, ThrowAndCatch)
{
  try {
    throw archetype::MTRException(archetype::MTRError_t::Cuda, "Test throw/catch");
    FAIL() << "Exception not thrown!";
  } catch (const archetype::MTRException & e) {
    EXPECT_STREQ(e.what(), "[CUDA]: Test throw/catch");
  } catch (...) {
    FAIL() << "Caught wrong exception type!";
  }
}

}  // namespace autoware::mtr::test
