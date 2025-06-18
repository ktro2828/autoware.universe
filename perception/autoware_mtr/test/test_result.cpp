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

#include "autoware/mtr/archetype/result.hpp"

#include <gtest/gtest.h>

#include <string>

namespace autoware::mtr::archetype
{
TEST(ResultTest, OkCaseInt)
{
  // Construct an Ok Result with int
  archetype::Result<int> res = archetype::Ok<int>(42);
  EXPECT_TRUE(res.is_ok());
  EXPECT_EQ(res.unwrap(), 42);
}

TEST(ResultTest, OkCaseString)
{
  // Construct an Ok Result with string
  archetype::Result<std::string> res = archetype::Ok<std::string>("test");
  EXPECT_TRUE(res.is_ok());
  EXPECT_EQ(res.unwrap(), "test");
}

TEST(ResultTest, ErrCase)
{
  // Construct an Err Result and verify is_ok returns false and unwrap throws
  archetype::Result<double> res =
    archetype::Err<double>(archetype::MTRError_t::InvalidValue, "NG value");
  EXPECT_FALSE(res.is_ok());
  try {
    (void)res.unwrap();
    FAIL() << "Expected MTRException";
  } catch (const archetype::MTRException & ex) {
    std::string what = ex.what();
    EXPECT_NE(what.find("InvalidValue"), std::string::npos);
    EXPECT_NE(what.find("NG value"), std::string::npos);
  } catch (...) {
    FAIL() << "Expected MTRException";
  }
}

TEST(ResultTest, ErrShortcutCase)
{
  // Using Err with just kind
  archetype::Result<float> res = archetype::Err<float>(archetype::MTRError_t::Cuda);
  EXPECT_FALSE(res.is_ok());
  EXPECT_THROW(res.unwrap(), archetype::MTRException);
}
}  // namespace autoware::mtr::archetype
