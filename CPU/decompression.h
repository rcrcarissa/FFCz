#pragma once

#include "config.h"
#include <cmath>
#include <cstdint>
#include <vector>

// Load packed flag word (32 boolean flags packed into one uint32_t)
inline uint32_t load_packed_flags(const uint8_t *flag_pack, size_t group_id) {
  return reinterpret_cast<const uint32_t *>(flag_pack)[group_id];
}

// Count bits set in a 32-bit integer (population count)
inline int popcount(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(x);
#else
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
#endif
}

// Compute number of set flags per group (number of non-zero elements per
// 32-element chunk)
void compute_group_counts(const uint8_t *flag_pack, uint32_t *group_counts,
                          size_t num_groups);

// Compute prefix sums over group counts to get global offsets into compact
// array
void compute_prefix_sums(const uint8_t *flag_pack, uint32_t *group_offsets,
                         size_t N);

// Unpack compact data and scatter to full-size output using packed flags
template <typename T>
void unpack_and_scatter(const T *compact, const uint8_t *flag_pack,
                        const uint32_t *group_offsets, T *output, size_t N);

// Dequantize quantized values back to floating point
template <typename T>
void dequantize(const UInt *quant, T *arr, T min_val, T max_val, size_t size,
                T quant_0);

// Main reconstruction function
template <typename T>
void reconstructEdits(const UInt *compact_quantized, const uint8_t *flag_pack,
                      T *output, T min_val, T max_val, size_t M, size_t N);
