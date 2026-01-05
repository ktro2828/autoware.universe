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

#include "autoware/point_rope_ops/point_rope_kernel.hpp"

#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v)
{
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __half2float(v);
  } else {
    return static_cast<float>(v);
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v)
{
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half_rn(v);
  } else {
    return static_cast<scalar_t>(v);
  }
}

template <typename scalar_t>
__global__ void point_rope_kernel(
  scalar_t * tokens, const std::int64_t N, const std::int64_t H, const std::int64_t D,
  const std::int64_t * pos, const float base, const float f0)
{
  // each block update a single token, for all heads each thread takes care of a single output
  extern __shared__ float shared[];
  float * shared_inv_freq = shared + D;

  const uint b = blockIdx.x / N;
  const uint n = blockIdx.x % N;

  const uint Q = D / 6;  // D = 18, Q = 3
  // one token = [0..Q : Q..2Q : 2Q..3Q : 3Q..4Q : 4Q..5Q : 5Q..D]
  //              u_X     v_X     u_Y      v_Y       u_Z      v_Z

  // shared memory: first, compute inv_freq
  if (threadIdx.x < Q) {
    shared_inv_freq[threadIdx.x] = f0 / powf(base, threadIdx.x / float(Q));
  }
  __syncthreads();

  // range of threadIdx.x is [0, 1, ..., 17]

  // start of X or Y or Z part
  const uint X = threadIdx.x * 3 / D;
  const uint m = (X * D / 3) + (threadIdx.x % Q);  // index of u_Y or u_X

  // grab the cos, sin
  const float freq = pos[blockIdx.x * 3 + X] * shared_inv_freq[threadIdx.x % Q];
  const float cos = cosf(freq);
  const float sin = sinf(freq);

  for (int h = 0; h < H; h++) {
    const uint idx = (((b * N + n) * H + h) * D) + threadIdx.x;
    // then, load all the token for this head in shared memory
    shared[threadIdx.x] = to_float<scalar_t>(tokens[idx]);
    __syncthreads();

    const float u = shared[m];
    const float v = shared[m + Q];

    // write output
    if ((threadIdx.x % (D / 3)) < Q) {
      tokens[idx] = from_float<scalar_t>(u * cos - v * sin);
    } else {
      tokens[idx] = from_float<scalar_t>(v * cos + u * sin);
    }
  }
}

template <typename scalar_t>
void point_rope_launch(
  scalar_t * tokens, const std::int64_t B, const std::int64_t N, const std::int64_t H,
  const std::int64_t D, const std::int64_t * pos, const float base, const float f0,
  cudaStream_t stream)
{
  // one block for each layer, one thread per local-max
  const int threads_per_block = D;
  const int n_blocks = B * N;  // each block takes care of H*D values
  const int shared_mem_size = sizeof(float) * (D + D / 6);

  point_rope_kernel<scalar_t>
    <<<n_blocks, threads_per_block, shared_mem_size, stream>>>(tokens, N, H, D, pos, base, f0);
}

template void point_rope_launch<float>(
  float * tokens, const std::int64_t B, const std::int64_t N, const std::int64_t H,
  const std::int64_t D, const std::int64_t * pos, const float base, const float f0,
  cudaStream_t stream);

template void point_rope_launch<__half>(
  __half * tokens, const std::int64_t B, const std::int64_t N, const std::int64_t H,
  const std::int64_t D, const std::int64_t * pos, const float base, const float f0,
  cudaStream_t stream);
