#include "projection_algorithm.cuh"

template <typename T>
__global__ void normalize_kernel(T *data, T factor, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] *= factor;
}

template <typename T>
__global__ void difference_kernel(const T *org_data, T *error, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    error[idx] -= org_data[idx];
}

template <typename T>
__global__ void normalize_add_kernel(const T *arr1, T *arr2, T factor,
                                     size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    arr2[idx] *= factor;
    arr2[idx] += arr1[idx];
  }
}

template <typename T>
__global__ void
normalize_add_two_difference_kernel(const T *arr1, const T *arr2, const T *arr3,
                                    T *arr4, T factor, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    arr4[idx] *= factor;
    arr4[idx] += arr1[idx] + arr2[idx] - arr3[idx];
  }
}

template <typename T>
__global__ void
real_imag_to_complex(const T *real_imag,
                     typename CufftTraits<T>::ComplexType *complex, size_t F) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= F)
    return;

  complex[idx].x = real_imag[idx];
  complex[idx].y = real_imag[idx + F];
}

template <typename T>
__global__ void mse_partial_sums_kernel(const T *err, T *partial_sums,
                                        size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for block-level reduction (align to 8 for double support)
  extern __shared__ __align__(8) unsigned char smem[];
  T *s_data = reinterpret_cast<T *>(smem);

  // Each thread computes one squared error
  T sum = 0.0f;
  if (idx < N) {
    sum = err[idx] * err[idx];
  }
  s_data[tid] = sum;
  __syncthreads();

  // Parallel reduction within block
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  // First thread writes block result
  if (tid == 0) {
    partial_sums[blockIdx.x] = s_data[0];
  }
}

template <typename T>
__global__ void check_convergence_abs_early_exit(
    const typename CufftTraits<T>::ComplexType *freq_errs, T delta,
    T convergence_tol, int *if_converge, size_t F) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= F)
    return;

  if (!*if_converge)
    return;

  if (abs_val(freq_errs[idx].x) > delta + convergence_tol) {
    atomicAnd(if_converge, 0); // Mark violation found
    return;
  }
  if (abs_val(freq_errs[idx].y) > delta + convergence_tol) {
    atomicAnd(if_converge, 0); // Mark violation found
  }
}

template <typename T>
__global__ void check_convergence_ptw_early_exit(
    const typename CufftTraits<T>::ComplexType *orig_freq,
    const typename CufftTraits<T>::ComplexType *freq_errs, T delta_ptw,
    T convergence_tol, int *if_converge, size_t F) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= F)
    return;

  if (!*if_converge)
    return;

  T orig_mag = complex_amplitude_squared<T>(orig_freq[idx]);
  typename CufftTraits<T>::ComplexType curr_freq;
  curr_freq.x = orig_freq[idx].x + freq_errs[idx].x;
  curr_freq.y = orig_freq[idx].y + freq_errs[idx].y;
  T curr_mag = complex_amplitude_squared<T>(curr_freq);
  T curr_ratio = curr_mag / orig_mag;
  if (curr_ratio <= sqrt_val(1 + delta_ptw) &&
      curr_ratio >= sqrt_val(1 - delta_ptw))
    return;

  T delta = fmin(1 - sqrt_val(1 - delta_ptw), sqrt_val(1 + delta_ptw) - 1) /
            sqrt_val(2.0f);

  if (abs_val(freq_errs[idx].x) > delta + convergence_tol) {
    atomicAnd(if_converge, 0); // Mark violation found
    return;
  }
  if (abs_val(freq_errs[idx].y) > delta + convergence_tol) {
    atomicAnd(if_converge, 0); // Mark violation found
  }
}

// Frequency projection kernel (ABSOLUTE bound)
// One thread per idx; no atomicAdd needed
template <typename T>
__global__ void project_frequency_constraints_abs(
    typename CufftTraits<T>::ComplexType *curr_freq_err, const T delta,
    T *freq_edits, size_t F) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= F)
    return;

  // Process real part
  T curr_err = curr_freq_err[idx].x;
  if (curr_err > delta) {
    freq_edits[idx] += (delta - curr_err); // Accumulate frequency edit
    curr_freq_err[idx].x = delta;
  } else if (curr_err < -delta) {
    freq_edits[idx] += (-delta - curr_err); // Accumulate frequency edit
    curr_freq_err[idx].x = -delta;
  }

  // Process imaginary part (no need to skip DC and Nyquist components)
  curr_err = curr_freq_err[idx].y;
  if (curr_err > delta) {
    freq_edits[idx + F] += (delta - curr_err);
    curr_freq_err[idx].y = delta;
  } else if (curr_err < -delta) {
    freq_edits[idx + F] += (-delta - curr_err);
    curr_freq_err[idx].y = -delta;
  }
}

// Frequency projection kernel (POINTWISE relative bound)
template <typename T>
__global__ void project_frequency_constraints_ptw(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const typename CufftTraits<T>::ComplexType *orig_freq, const T delta_ptw,
    T *freq_edits, size_t F) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= F)
    return;

  // separated condition for DC component
  if (idx == 0) {
    T lower_bound = orig_freq[0].x * (1 / sqrt_val(1 + delta_ptw) - 1);
    T upper_bound = orig_freq[0].x * (1 / sqrt_val(1 - delta_ptw) - 1);
    T curr_err = curr_freq_err[0].x;
    if (curr_err > upper_bound || curr_err < lower_bound) {
      T edit_dist = -curr_err;
      freq_edits[0] -= curr_err; // edit DC component error close to 0
      curr_freq_err[0].x = 0;
    }
    return;
  }

  // conditions for non-DC components
  T orig_mag = complex_amplitude_squared<T>(orig_freq[idx]);
  typename CufftTraits<T>::ComplexType curr_freq;
  curr_freq.x = orig_freq[idx].x + curr_freq_err[idx].x;
  curr_freq.y = orig_freq[idx].y + curr_freq_err[idx].y;
  T curr_mag = complex_amplitude_squared<T>(curr_freq);
  T curr_ratio = curr_mag / orig_mag;
  if (curr_ratio <= sqrt_val(1 + delta_ptw) &&
      curr_ratio >= sqrt_val(1 - delta_ptw))
    return;

  T delta = fmin(1 - sqrt_val(1 - delta_ptw), sqrt_val(1 + delta_ptw) - 1) /
            sqrt_val(2.0f);

  // Process real part
  T curr_err = curr_freq_err[idx].x;
  if (curr_err > delta) {
    freq_edits[idx] += (delta - curr_err);
    curr_freq_err[idx].x = delta;
  } else if (curr_err < -delta) {
    freq_edits[idx] += (-delta - curr_err);
    curr_freq_err[idx].x = -delta;
  }

  // Process imaginary part (no need to skip DC and Nyquist components)
  curr_err = curr_freq_err[idx].y;
  if (curr_err > delta) {
    freq_edits[idx + F] += (delta - curr_err);
    curr_freq_err[idx].y = delta;
  } else if (curr_err < -delta) {
    freq_edits[idx + F] += (-delta - curr_err);
    curr_freq_err[idx].y = -delta;
  }
}

// Spatial projection kernel
template <typename T>
__global__ void project_spatial_constraints(T *spat_error, T epsilon,
                                            T *spat_edits, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  T curr_err = spat_error[idx];

  T edit_distance = 0;

  if (curr_err > epsilon) {
    spat_edits[idx] += epsilon - curr_err;
    spat_error[idx] = epsilon;
  } else if (curr_err < -epsilon) {
    spat_edits[idx] += -epsilon - curr_err;
    spat_error[idx] = -epsilon;
  }
}

// Utility kernels
// Create and pack the flag indicating the position of non-zero edits
template <typename T>
__global__ void create_and_pack_bool_flags(const T *__restrict__ input,
                                           uint8_t *__restrict__ flag_pack,
                                           T tol, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  size_t warp_id = idx >> 5;

  int pred = (idx < N) && (abs_val(input[idx]) > tol);
  uint32_t flag = __ballot_sync(0xffffffff, pred);

  if (lane == 0 && warp_id * 32 < N) {
    reinterpret_cast<uchar4 *>(flag_pack)[warp_id] =
        *reinterpret_cast<uchar4 *>(&flag);
  }
}

template <typename T>
__global__ void quantize(const T *arr, UInt *quant, T min_val, T max_val,
                         size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  T normalized = (arr[idx] - min_val) / (max_val - min_val);
  if (normalized > 0.5)
    quant[idx] = ceil(normalized * ((1u << m) - 1));
  else
    quant[idx] = floor(normalized * ((1u << m) - 1));
}

template <typename T>
__global__ void quantize_and_dequantize(const T *org_arr, T *dequant_arr,
                                        T min_val, T max_val, size_t size,
                                        T thres_lower, T thres_upper) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  T org_val = org_arr[idx];
  if (org_val > thres_lower && org_val < thres_upper) {
    dequant_arr[idx] = 0;
  } else {
    T normalized = (org_val - min_val) / (max_val - min_val);
    T quant;
    if (normalized > 0.5)
      quant = ceil(normalized * ((1u << m) - 1));
    else
      quant = floor(normalized * ((1u << m) - 1));
    dequant_arr[idx] = quant / ((1u << m) - 1) * (max_val - min_val) + min_val;
  }
}

template <typename T>
void compactNonZero(const T *d_input, T *d_compact, size_t &num_nonzeros, T tol,
                    size_t N) {
  int *d_num_nonzeros;
  CHECK_CUDA(cudaMalloc(&d_num_nonzeros, sizeof(int)));

  IsNonZeroOp<T> pred{tol};
  auto flag_itr = cub::TransformInputIterator<bool, IsNonZeroOp<T>, const T *>(
      d_input, pred);

  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, tmp_storage_bytes, d_input, flag_itr,
                             d_compact, d_num_nonzeros, N);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceSelect::Flagged(d_tmp_storage, tmp_storage_bytes, d_input,
                             flag_itr, d_compact, d_num_nonzeros, N);

  int h_num_nonzeros = 0;
  CHECK_CUDA(cudaMemcpy(&h_num_nonzeros, d_num_nonzeros, sizeof(int),
                        cudaMemcpyDeviceToHost));
  num_nonzeros = static_cast<size_t>(h_num_nonzeros);
  CHECK_CUDA(cudaFree(d_tmp_storage));
  CHECK_CUDA(cudaFree(d_num_nonzeros));
}

// Kernel instantiations
template __global__ void normalize_kernel<float>(float *, float, size_t);
template __global__ void normalize_kernel<double>(double *, double, size_t);

template __global__ void difference_kernel<float>(const float *, float *,
                                                  size_t);
template __global__ void difference_kernel<double>(const double *, double *,
                                                   size_t);

template __global__ void normalize_add_kernel<float>(const float *, float *,
                                                     float, size_t);
template __global__ void normalize_add_kernel<double>(const double *, double *,
                                                      double, size_t);

template __global__ void normalize_add_two_difference_kernel<float>(
    const float *, const float *, const float *, float *, float, size_t);
template __global__ void normalize_add_two_difference_kernel<double>(
    const double *, const double *, const double *, double *, double, size_t);

template __global__ void real_imag_to_complex<float>(const float *,
                                                     cufftComplex *, size_t);
template __global__ void
real_imag_to_complex<double>(const double *, cufftDoubleComplex *, size_t);

template __global__ void mse_partial_sums_kernel<float>(const float *, float *,
                                                        size_t);
template __global__ void mse_partial_sums_kernel<double>(const double *,
                                                         double *, size_t);

template __global__ void
check_convergence_abs_early_exit<float>(const cufftComplex *, float, float,
                                        int *, size_t);
template __global__ void
check_convergence_abs_early_exit<double>(const cufftDoubleComplex *, double,
                                         double, int *, size_t);

template __global__ void check_convergence_ptw_early_exit<float>(
    const cufftComplex *, const cufftComplex *, float, float, int *, size_t);
template __global__ void
check_convergence_ptw_early_exit<double>(const cufftDoubleComplex *,
                                         const cufftDoubleComplex *, double,
                                         double, int *, size_t);

template __global__ void
project_frequency_constraints_abs<float>(cufftComplex *, const float, float *,
                                         size_t);

template __global__ void
project_frequency_constraints_abs<double>(cufftDoubleComplex *, const double,
                                          double *, size_t);

template __global__ void
project_frequency_constraints_ptw<float>(cufftComplex *, const cufftComplex *,
                                         const float, float *, size_t);

template __global__ void
project_frequency_constraints_ptw<double>(cufftDoubleComplex *,
                                          const cufftDoubleComplex *,
                                          const double, double *, size_t);

template __global__ void project_spatial_constraints<float>(float *, float,
                                                            float *, size_t);

template __global__ void project_spatial_constraints<double>(double *, double,
                                                             double *, size_t);

template __global__ void
create_and_pack_bool_flags<float>(const float *__restrict__,
                                  uint8_t *__restrict__, float, size_t);
template __global__ void
create_and_pack_bool_flags<double>(const double *__restrict__,
                                   uint8_t *__restrict__, double, size_t);

template __global__ void quantize<float>(const float *, UInt *, float, float,
                                         size_t);
template __global__ void quantize<double>(const double *, UInt *, double,
                                          double, size_t);

template __global__ void quantize_and_dequantize<float>(const float *, float *,
                                                        float, float, size_t,
                                                        float, float);
template __global__ void quantize_and_dequantize<double>(const double *,
                                                         double *, double,
                                                         double, size_t, double,
                                                         double);

template void compactNonZero<float>(const float *, float *, size_t &, float,
                                    size_t);
template void compactNonZero<double>(const double *, double *, size_t &, double,
                                     size_t);

template <typename T>
__global__ void ssnr_partial_sums_kernel(
    const typename CufftTraits<T>::ComplexType *d_orig_freq,
    const typename CufftTraits<T>::ComplexType *d_freq_err, T *d_partial_orig,
    T *d_partial_err, size_t Nx, size_t Ny, size_t Nz_stored, size_t Nz,
    size_t F) {

  // Shared memory for block-level reduction (align to 8 for double support)
  extern __shared__ __align__(8) unsigned char smem[];
  T *s_orig = reinterpret_cast<T *>(smem);
  T *s_err = s_orig + blockDim.x;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  T orig_sum = T(0);
  T err_sum = T(0);

  if (idx < F) {
    size_t iz = idx % Nz_stored;

    int weight = (iz > 0 && iz < Nz_stored - 1) ? 2 : 1;
    if (Nz % 2 == 1 && iz == Nz_stored - 1) {
      weight = 2;
    }

    T orig_real = d_orig_freq[idx].x;
    T orig_imag = d_orig_freq[idx].y;
    T err_real = d_freq_err[idx].x;
    T err_imag = d_freq_err[idx].y;

    orig_sum = weight * (orig_real * orig_real + orig_imag * orig_imag);
    err_sum = weight * (err_real * err_real + err_imag * err_imag);
  }

  s_orig[tid] = orig_sum;
  s_err[tid] = err_sum;
  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_orig[tid] += s_orig[tid + s];
      s_err[tid] += s_err[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_partial_orig[blockIdx.x] = s_orig[0];
    d_partial_err[blockIdx.x] = s_err[0];
  }
}

template <typename T>
T computeSSNR(const typename CufftTraits<T>::ComplexType *d_orig_freq,
              const typename CufftTraits<T>::ComplexType *d_freq_err, size_t Nx,
              size_t Ny, size_t Nz) {
  size_t Nz_stored = Nz / 2 + 1;
  size_t F = Nx * Ny * Nz_stored;

  const int block_size = 256;
  int num_blocks = (F + block_size - 1) / block_size;

  T *d_partial_orig, *d_partial_err;
  CHECK_CUDA(cudaMalloc(&d_partial_orig, num_blocks * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_partial_err, num_blocks * sizeof(T)));

  size_t shared_mem_size = 2 * block_size * sizeof(T);
  ssnr_partial_sums_kernel<T><<<num_blocks, block_size, shared_mem_size>>>(
      d_orig_freq, d_freq_err, d_partial_orig, d_partial_err, Nx, Ny, Nz_stored,
      Nz, F);
  CHECK_CUDA(cudaGetLastError());

  T *d_total_orig, *d_total_err;
  CHECK_CUDA(cudaMalloc(&d_total_orig, sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_total_err, sizeof(T)));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_partial_orig,
                         d_total_orig, num_blocks);
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_partial_orig,
                         d_total_orig, num_blocks);

  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_partial_err,
                         d_total_err, num_blocks);

  T total_orig, total_err;
  CHECK_CUDA(
      cudaMemcpy(&total_orig, d_total_orig, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(&total_err, d_total_err, sizeof(T), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_partial_orig));
  CHECK_CUDA(cudaFree(d_partial_err));
  CHECK_CUDA(cudaFree(d_total_orig));
  CHECK_CUDA(cudaFree(d_total_err));
  CHECK_CUDA(cudaFree(d_temp_storage));

  if (total_err == T(0)) {
    return std::numeric_limits<T>::infinity();
  }
  return T(10) * std::log10(total_orig / total_err);
}

// Explicit template instantiations
template __global__ void ssnr_partial_sums_kernel<float>(const cufftComplex *,
                                                         const cufftComplex *,
                                                         float *, float *,
                                                         size_t, size_t, size_t,
                                                         size_t, size_t);
template __global__ void
ssnr_partial_sums_kernel<double>(const cufftDoubleComplex *,
                                 const cufftDoubleComplex *, double *, double *,
                                 size_t, size_t, size_t, size_t, size_t);

template float computeSSNR<float>(const cufftComplex *, const cufftComplex *,
                                  size_t, size_t, size_t);
template double computeSSNR<double>(const cufftDoubleComplex *,
                                    const cufftDoubleComplex *, size_t, size_t,
                                    size_t);