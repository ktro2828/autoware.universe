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

#ifndef AUTOWARE__LITEPT__UTILITY_HPP_
#define AUTOWARE__LITEPT__UTILITY_HPP_

#include <stdexcept>

namespace autoware::litept
{
// cspell: ignore divup
template <typename T1, typename T2>
unsigned int divup(const T1 a, const T2 b)
{
  if (a == 0) {
    throw std::runtime_error("A dividend of divup isn't positive.");
  }
  if (b == 0) {
    throw std::runtime_error("A divisor of divup isn't positive.");
  }

  return (a + b - 1) / b;
}
}  // namespace autoware::litept
#endif  // AUTOWARE__LITEPT__UTILITY_HPP_
