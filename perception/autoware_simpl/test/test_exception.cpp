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

#include "autoware/simpl/archetype/exception.hpp"

#include <gtest/gtest.h>

namespace autoware::simpl::test
{
using autoware::simpl::archetype::SimplError;
using autoware::simpl::archetype::SimplError_t;
using autoware::simpl::archetype::SimplException;

TEST(TestSimplException, MessageFromErrorObject)
{
  SimplError error(SimplError_t::TensorRT, "Failed to initialize engine");
  try {
    throw SimplException(error);
  } catch (const SimplException & e) {
    EXPECT_STREQ(e.what(), "[TensorRT]: Failed to initialize engine");
  }
}

TEST(TestSimplException, MessageFromKindAndText)
{
  try {
    throw SimplException(SimplError_t::Cuda, "CUDA device unavailable");
  } catch (const SimplException & e) {
    EXPECT_STREQ(e.what(), "[CUDA]: CUDA device unavailable");
  }
}

TEST(TestSimplException, InvalidValueMessage)
{
  try {
    throw SimplException(SimplError_t::InvalidValue, "Unsupported type");
  } catch (const SimplException & e) {
    EXPECT_STREQ(e.what(), "[InvalidValue]: Unsupported type");
  }
}

TEST(TestSimplException, UnknownErrorKind)
{
  try {
    throw SimplException(SimplError_t::Unknown, "Something went wrong");
  } catch (const SimplException & e) {
    EXPECT_STREQ(e.what(), "[UNKNOWN]: Something went wrong");
  }
}

TEST(SimplErrorTest, DefaultConstructor)
{
  SimplError error(SimplError_t::InvalidValue);
  EXPECT_EQ(error.kind, SimplError_t::InvalidValue);
  EXPECT_EQ(error.msg, "");
}
}  // namespace autoware::simpl::test
