#pragma once

#include "config.cuh"
#include <cmath>
#include <vector>

__device__ __forceinline__ uint32_t load_warp_flag(const uint8_t *flag_pack,
                                                   size_t warp_id) {
  return reinterpret_cast<const uint32_t *>(flag_pack)[warp_id];
}

__global__ void compute_warp_counts(const uint8_t *__restrict__ flag_pack,
                                    uint32_t *__restrict__ warp_counts,
                                    size_t num_warps);

void compute_global_warp_offsets(const uint8_t *d_flag_pack,
                                 uint32_t *d_warp_offsets, // size num_warps
                                 size_t N);

template <typename T>
__global__ void unpack_and_scatter(const T *__restrict__ compact,
                                   const uint8_t *__restrict__ flag_pack,
                                   const uint32_t *__restrict__ warp_offsets,
                                   T *__restrict__ output, size_t N);

template <typename T>
__global__ void dequantize(const UInt *quant, T *arr, T min_val, T max_val,
                           size_t size, T quant_0);

template <typename T>
void reconstructEdits(const UInt *d_compact_quantized,
                      const uint8_t *d_flag_pack, T *d_output, T min_val,
                      T max_val, size_t M, size_t N);
