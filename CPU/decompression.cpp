#include "decompression.h"
#include <cmath>
#include <numeric>

void compute_group_counts(const uint8_t *flag_pack, uint32_t *group_counts,
                          size_t num_groups) {
  for (size_t group_id = 0; group_id < num_groups; ++group_id) {
    uint32_t flags = reinterpret_cast<const uint32_t *>(flag_pack)[group_id];
    group_counts[group_id] = popcount(flags);
  }
}

void compute_prefix_sums(const uint8_t *flag_pack, uint32_t *group_offsets,
                         size_t N) {
  size_t num_groups = (N + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD;

  std::vector<uint32_t> group_counts(num_groups);
  compute_group_counts(flag_pack, group_counts.data(), num_groups);

  // Exclusive prefix sum using C++17 standard library
  std::exclusive_scan(group_counts.begin(), group_counts.end(), group_offsets,
                      uint32_t{0});
}

template <typename T>
void unpack_and_scatter(const T *compact, const uint8_t *flag_pack,
                        const uint32_t *group_offsets, T *output, size_t N) {
  for (size_t idx = 0; idx < N; ++idx) {
    size_t group_id = idx / BITS_PER_FLAG_WORD;
    int bit_position = idx % BITS_PER_FLAG_WORD;

    uint32_t flags = load_packed_flags(flag_pack, group_id);
    bool is_nonzero = (flags >> bit_position) & 1;

    if (is_nonzero) {
      // Count set bits before this position to find rank in compact array
      uint32_t mask = (1u << bit_position) - 1;
      uint32_t rank = popcount(flags & mask);
      output[idx] = compact[group_offsets[group_id] + rank];
    } else {
      output[idx] = T(0);
    }
  }
}

template <typename T>
void dequantize(const UInt *quant, T *arr, T min_val, T max_val, size_t size,
                T quant_0) {
  for (size_t idx = 0; idx < size; ++idx) {
    T q = static_cast<T>(quant[idx]);
    if (q == quant_0 || q == quant_0 + 1) {
      arr[idx] = T(0);
    } else {
      arr[idx] =
          q / static_cast<T>((1u << m) - 1) * (max_val - min_val) + min_val;
    }
  }
}

template <typename T>
void reconstructEdits(const UInt *compact_quantized, const uint8_t *flag_pack,
                      T *output, T min_val, T max_val, size_t M, size_t N) {
  // Dequantize compact values
  std::vector<T> compact(M);
  T normalized_0 = -min_val / (max_val - min_val);
  T quant_0;
  if (normalized_0 > T(0.5))
    quant_0 = std::ceil(normalized_0 * ((1u << m) - 1));
  else
    quant_0 = std::floor(normalized_0 * ((1u << m) - 1));

  dequantize(compact_quantized, compact.data(), min_val, max_val, M, quant_0);

  // Compute prefix sums for scatter offsets
  size_t num_groups = (N + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD;
  std::vector<uint32_t> group_counts(num_groups);
  std::vector<uint32_t> group_offsets(num_groups);

  compute_group_counts(flag_pack, group_counts.data(), num_groups);
  std::exclusive_scan(group_counts.begin(), group_counts.end(),
                      group_offsets.begin(), uint32_t{0});

  // Scatter compact values to full output
  unpack_and_scatter(compact.data(), flag_pack, group_offsets.data(), output,
                     N);
}

// Explicit template instantiations
template void unpack_and_scatter<float>(const float *, const uint8_t *,
                                        const uint32_t *, float *, size_t);
template void unpack_and_scatter<double>(const double *, const uint8_t *,
                                         const uint32_t *, double *, size_t);

template void dequantize<float>(const UInt *, float *, float, float, size_t,
                                float);
template void dequantize<double>(const UInt *, double *, double, double, size_t,
                                 double);

template void reconstructEdits<float>(const UInt *, const uint8_t *, float *,
                                      float, float, size_t, size_t);
template void reconstructEdits<double>(const UInt *, const uint8_t *, double *,
                                       double, double, size_t, size_t);
