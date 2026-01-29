#include "decompression.cuh"

__global__ void compute_warp_counts(const uint8_t *__restrict__ flag_pack,
                                    uint32_t *__restrict__ warp_counts,
                                    size_t num_warps) {
  size_t warp_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (warp_id < num_warps) {
    uint32_t flag = reinterpret_cast<const uint32_t *>(flag_pack)[warp_id];
    warp_counts[warp_id] = __popc(flag);
  }
}

void compute_global_warp_offsets(const uint8_t *d_flag_pack,
                                 uint32_t *d_warp_offsets, // size num_warps
                                 size_t N) {
  size_t num_warps = (N + 31) / 32;

  uint32_t *d_warp_counts;
  CHECK_CUDA(cudaMalloc(&d_warp_counts, num_warps * sizeof(uint32_t)));

  dim3 block(BLOCK_SIZE);
  dim3 grid((num_warps + block.x - 1) / block.x);
  compute_warp_counts<<<grid, block>>>(d_flag_pack, d_warp_counts, num_warps);
  CHECK_CUDA(cudaGetLastError());

  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(nullptr, tmp_storage_bytes, d_warp_counts,
                                d_warp_offsets, num_warps);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes, d_warp_counts,
                                d_warp_offsets, num_warps);

  CHECK_CUDA(cudaFree(d_tmp_storage));
  CHECK_CUDA(cudaFree(d_warp_counts));
}

// Unpack the non-zero-flag and scatter the full size of edits
template <typename T>
__global__ void unpack_and_scatter(const T *__restrict__ compact,
                                   const uint8_t *__restrict__ flag_pack,
                                   const uint32_t *__restrict__ warp_offsets,
                                   T *__restrict__ output, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  size_t warp_id = idx >> 5;
  int lane = idx & 31;

  uint32_t flag = load_warp_flag(flag_pack, warp_id);
  bool nz = (flag >> lane) & 1;

  if (nz) {
    uint32_t rank = __popc(flag & ((1u << lane) - 1));
    output[idx] = compact[warp_offsets[warp_id] + rank];
  } else {
    output[idx] = T(0);
  }
}

template <typename T>
__global__ void dequantize(const UInt *quant, T *arr, T min_val, T max_val,
                           size_t size, T quant_0) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  T q = static_cast<T>(quant[idx]);
  if (q == quant_0 || q == quant_0 + 1)
    arr[idx] = T(0);
  else
    arr[idx] =
        q / static_cast<T>((1u << m) - 1) * (max_val - min_val) + min_val;
}

template <typename T>
void reconstructEdits(const UInt *d_compact_quantized,
                      const uint8_t *d_flag_pack, T *d_output, T min_val,
                      T max_val, size_t M, size_t N) {
  // Dequantize
  T *d_compact;
  CHECK_CUDA(cudaMalloc(&d_compact, M * sizeof(T)));
  dim3 block(BLOCK_SIZE);
  dim3 dequant_grid((M + block.x - 1) / block.x);
  T normalized_0 = -min_val / (max_val - min_val);
  T quant_0;
  if (normalized_0 > T(0.5))
    quant_0 = ceil(normalized_0 * ((1u << m) - 1));
  else
    quant_0 = floor(normalized_0 * ((1u << m) - 1));
  dequantize<<<dequant_grid, block>>>(d_compact_quantized, d_compact, min_val,
                                      max_val, M, quant_0);

  // Compute global warp offsets
  size_t num_warps = (N + 31) / 32;
  uint32_t *d_warp_counts, *d_warp_offsets;
  CHECK_CUDA(cudaMalloc(&d_warp_counts, num_warps * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_warp_offsets, num_warps * sizeof(uint32_t)));

  dim3 warp_grid((num_warps + block.x - 1) / block.x);
  compute_warp_counts<<<warp_grid, block>>>(d_flag_pack, d_warp_counts,
                                            num_warps);
  CHECK_CUDA(cudaGetLastError());

  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(nullptr, tmp_storage_bytes, d_warp_counts,
                                d_warp_offsets, num_warps);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes, d_warp_counts,
                                d_warp_offsets, num_warps);

  // Reconstruct
  dim3 scatter_grid((N + block.x - 1) / block.x);
  unpack_and_scatter<<<scatter_grid, block>>>(d_compact, d_flag_pack,
                                              d_warp_offsets, d_output, N);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaFree(d_warp_counts));
  CHECK_CUDA(cudaFree(d_warp_offsets));
  CHECK_CUDA(cudaFree(d_compact));
  CHECK_CUDA(cudaFree(d_tmp_storage));
}

// Kernel instantiations
template __global__ void unpack_and_scatter<float>(const float *__restrict__,
                                                   const uint8_t *__restrict__,
                                                   const uint32_t *__restrict__,
                                                   float *__restrict__, size_t);
template __global__ void unpack_and_scatter<double>(
    const double *__restrict__, const uint8_t *__restrict__,
    const uint32_t *__restrict__, double *__restrict__, size_t);

template __global__ void dequantize<float>(const UInt *, float *, float, float,
                                           size_t, float);
template __global__ void dequantize<double>(const UInt *, double *, double,
                                            double, size_t, double);

template void reconstructEdits<float>(const UInt *, const uint8_t *, float *,
                                      float, float, size_t, size_t);
template void reconstructEdits<double>(const UInt *, const uint8_t *, double *,
                                       double, double, size_t, size_t);