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

#include "autoware/simpl/conversion/lanelet.hpp"

#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>

#include <gtest/gtest.h>
#include <lanelet2_io/Io.h>

namespace autoware::simpl::test
{
TEST(TestWithData, RunTestWithData)
{
  std::string filepath =
    "/home/kotarouetake/.webauto/simulation/data/map/prd_jt/1423/1423-20250602074501626038/"
    "lanelet2_map.osm";
  lanelet::projection::MGRSProjector projector;
  auto lanelet_map = lanelet::load(filepath, projector);

  // ==== Load polylines ====
  conversion::LaneletConverter converter;
  converter.convert(std::move(lanelet_map));

  const auto polylines_opt = converter.polylines();
  const auto & polylines = polylines_opt.value();
  size_t num_point = 0;
  for (const auto & polyline : polylines) {
    for (const auto & point : polyline) {
      std::cout << std::setprecision(12) << "(" << point.x << ", " << point.y << ", " << point.z
                << ")\n";
      ++num_point;
    }
  }
  std::cout << "Num point: " << num_point;
}
}  // namespace autoware::simpl::test
