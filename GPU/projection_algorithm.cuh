#pragma once

#include "HuffmanZSTDCoder.cuh"
#include "config.cuh"
#include <cmath>
#include <vector>

// Structure to hold projection parameters
template <typename T> struct ProjectionParams {
  T *d_orig_data;  // Original data (Nx*Ny*Nz)
  T *d_spat_error; // Current spatial error vector (Nx*Ny*Nz)
  typename CufftTraits<T>::ComplexType *d_orig_freq; // Original frequency
  typename CufftTraits<T>::ComplexType
      *d_freq_error; // Current frequency error vector (Nx*Ny*(Nz/2+1))
  T delta;           // Absolute tolerance for all frequency components
  T delta_ptw;
  T epsilon;                    // Spatial constraint tolerance
  size_t Nx, Ny, Nz;            // Grid dimensions
  size_t N, F;                  // Number of spatial values and frequency modes
  cufftHandle fft_plan_forward; // FFT plan (spatial to frequency)
  cufftHandle fft_plan_inverse; // IFFT plan (frequency to spatial)
  T spat_min, spat_max, freq_amp_max;
};

// Result structure
template <typename T> struct IterationResults {
  // Accumulator arrays (the core storage)
  T *d_freq_edits; // Size: Nx*Ny*(Nz/2+1)*2 - accumulated edit distances from
                   // orig_freq
  T *d_spat_edits; // Size: Nx*Ny*Nz - accumulated edit distances from
                   // orig_spatial (zero)

  // Compact representation (extracted when needed)
  uint8_t *d_freq_flag_pack; // Size: Nx*Ny*(Nz/2+1)*2/8
  uint8_t *d_spat_flag_pack; // Size: Nx*Ny*Nz/8
  T *d_freq_edit_compact;    // Variable size - only non-zero edit distances
  T *d_spat_edit_compact;    // Variable size - only non-zero edit distances

  // Compact and quantized representation
  UInt *d_freq_edit_compact_quantized; // Variable size - only non-zero
                                       // quantized edit distances
  UInt *d_spat_edit_compact_quantized; // Variable size - only non-zero
                                       // quantized edit distances

  // Final compressed result
  typename HuffmanZstdCompressor<UInt>::CompressionResult
      h_freq_edit_compressed;
  typename HuffmanZstdCompressor<uint8_t>::CompressionResult
      h_freq_flag_compressed;
  typename HuffmanZstdCompressor<UInt>::CompressionResult
      h_spat_edit_compressed;
  typename HuffmanZstdCompressor<uint8_t>::CompressionResult
      h_spat_flag_compressed;

  T h_freq_min_edit;
  T h_freq_max_edit;
  T h_spat_min_edit;
  T h_spat_max_edit;

  // Counters
  size_t h_num_active_freq; // Host copy of active frequency count
  size_t h_num_active_spat; // Host copy of active spatial count
  size_t max_iterations;
  size_t iteration_count; // Iteration number
  T convergence_tol;
};

// Device helper functions
template <typename T>
__device__ __forceinline__ T
complex_amplitude_squared(const typename CufftTraits<T>::ComplexType &c);

template <>
__device__ __forceinline__ float
complex_amplitude_squared<float>(const cufftComplex &c) {
  return c.x * c.x + c.y * c.y;
}

template <>
__device__ __forceinline__ double
complex_amplitude_squared<double>(const cufftDoubleComplex &c) {
  return c.x * c.x + c.y * c.y;
}

template <typename T> __device__ __forceinline__ T abs_val(T x);

template <> __device__ __forceinline__ float abs_val(float x) {
  return fabsf(x);
}

template <> __device__ __forceinline__ double abs_val(double x) {
  return fabs(x);
}

template <typename T> __host__ __device__ __forceinline__ T sqrt_val(T x);

template <> __host__ __device__ __forceinline__ float sqrt_val(float x) {
  return sqrtf(x);
}

template <> __host__ __device__ __forceinline__ double sqrt_val(double x) {
  return sqrt(x);
}

// Max-min struct for a single CUB reduction pass
template <typename T> struct MinMaxPair {
  T min;
  T max;
};

template <typename T> struct AbsOp {
  __device__ __forceinline__ T operator()(const T &x) const {
    return abs_val(x);
  }
};

template <typename T> struct MinMaxReduceOp {
  __device__ __forceinline__ MinMaxPair<T>
  operator()(const MinMaxPair<T> &p1, const MinMaxPair<T> &p2) const {
    return {min(p1.min, p2.min), max(p1.max, p2.max)};
  }
};

template <typename T> struct MinMaxPairOp {
  __device__ __forceinline__ MinMaxPair<T> operator()(const T &x) const {
    return {x, x};
  }
};

template <typename T> struct ComplexAbsSquaredOp {
  __device__ __forceinline__ T
  operator()(const typename CufftTraits<T>::ComplexType &c) const {
    return complex_amplitude_squared<T>(c);
  }
};

template <typename T> struct ComplexMaxAbsOp {
  __device__ __forceinline__ T
  operator()(const typename CufftTraits<T>::ComplexType &c) const {
    return max(abs_val(c.x), abs_val(c.y));
  }
};

template <typename T> struct ConvergenceOp {
  T epsilon;
  __device__ __forceinline__ T operator()(const T &x) const {
    return (abs_val(x) > epsilon) ? abs_val(x) - epsilon : T(0);
  }
};

template <typename T> T findMaxAbs(const T *d_arr, size_t N) {
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;

  T *d_result;
  CHECK_CUDA(cudaMalloc(&d_result, sizeof(T)));

  AbsOp<T> abs_op;
  auto abs_itr =
      cub::TransformInputIterator<T, AbsOp<T>, const T *>(d_arr, abs_op);

  cub::DeviceReduce::Max(nullptr, tmp_storage_bytes, abs_itr, d_result, N);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceReduce::Max(d_tmp_storage, tmp_storage_bytes, abs_itr, d_result,
                         N);

  T result;
  CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_tmp_storage));
  CHECK_CUDA(cudaFree(d_result));
  return result;
}

template <typename T> MinMaxPair<T> findMinMax(const T *d_arr, size_t N) {
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;

  MinMaxPair<T> *d_result;
  CHECK_CUDA(cudaMalloc(&d_result, sizeof(MinMaxPair<T>)));

  MinMaxPairOp<T> transform_op;
  cub::TransformInputIterator<MinMaxPair<T>, MinMaxPairOp<T>, const T *> itr(
      d_arr, transform_op);

  MinMaxPair<T> init;
  init.min = std::numeric_limits<T>::max();
  init.max = std::numeric_limits<T>::lowest();

  MinMaxReduceOp<T> reduce_op;
  cub::DeviceReduce::Reduce(nullptr, tmp_storage_bytes, itr, d_result, N,
                            reduce_op, init);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceReduce::Reduce(d_tmp_storage, tmp_storage_bytes, itr, d_result, N,
                            reduce_op, init);

  MinMaxPair<T> result;
  CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(MinMaxPair<T>),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_tmp_storage));
  CHECK_CUDA(cudaFree(d_result));
  return result;
}

template <typename T>
T findMaxAbsComplex(const typename CufftTraits<T>::ComplexType *d_arr,
                    size_t F) {
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;
  T *d_result;
  CHECK_CUDA(cudaMalloc(&d_result, sizeof(T)));

  ComplexMaxAbsOp<T> max_abs_op;
  auto itr =
      cub::TransformInputIterator<T, ComplexMaxAbsOp<T>,
                                  const typename CufftTraits<T>::ComplexType *>(
          d_arr, max_abs_op);

  cub::DeviceReduce::Max(nullptr, tmp_storage_bytes, itr, d_result, F);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceReduce::Max(d_tmp_storage, tmp_storage_bytes, itr, d_result, F);

  T result;
  CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_tmp_storage));
  CHECK_CUDA(cudaFree(d_result));

  return result;
}

template <typename T>
T findMaxComplexAmplitude(const typename CufftTraits<T>::ComplexType *d_arr,
                          size_t F) {
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;
  T *d_result;
  CHECK_CUDA(cudaMalloc(&d_result, sizeof(T)));

  ComplexAbsSquaredOp<T> amp_op;
  auto itr =
      cub::TransformInputIterator<T, ComplexAbsSquaredOp<T>,
                                  const typename CufftTraits<T>::ComplexType *>(
          d_arr, amp_op);

  cub::DeviceReduce::Max(nullptr, tmp_storage_bytes, itr, d_result, F);
  CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceReduce::Max(d_tmp_storage, tmp_storage_bytes, itr, d_result, F);

  T result;
  CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_tmp_storage));
  CHECK_CUDA(cudaFree(d_result));

  return sqrt_val(result);
}

template <typename T> struct IsNonZeroOp {
  T tol;
  __device__ __forceinline__ bool operator()(T x) const {
    return abs_val(x) > tol;
  }
};

template <typename T>
void compactNonZero(const T *d_input, T *d_compact, size_t &num_nonzeros, T tol,
                    size_t N);

// Kernel declarations
template <typename T>
__global__ void normalize_kernel(T *data, T factor, size_t size);

template <typename T>
__global__ void difference_kernel(const T *org_data, T *error, size_t size);

template <typename T>
__global__ void normalize_add_kernel(const T *arr1, T *arr2, T factor,
                                     size_t size);

template <typename T>
__global__ void
normalize_add_two_difference_kernel(const T *arr1, const T *arr2, const T *arr3,
                                    T *arr4, T factor, size_t size);

template <typename T>
__global__ void
real_imag_to_complex(const T *real_imag,
                     typename CufftTraits<T>::ComplexType *complex, size_t F);

template <typename T>
__global__ void mse_partial_sums_kernel(const T *err, T *partial_sums,
                                        size_t N);

// uniform ABSolute error bound
template <typename T>
__global__ void project_frequency_constraints_abs(
    typename CufftTraits<T>::ComplexType *curr_freq_err, const T delta,
    T *freq_edits, size_t F);

// for power spectrum: POINTWISE relative error bound
template <typename T>
__global__ void project_frequency_constraints_ptw(
    typename CufftTraits<T>::ComplexType *curr_freq_err,
    const typename CufftTraits<T>::ComplexType *orig_freq, const T delta,
    T *freq_edits, size_t F);

template <typename T>
__global__ void project_spatial_constraints(T *spat_error, T epsilon,
                                            T *spat_edits, size_t N);

template <typename T>
__global__ void create_and_pack_bool_flags(const T *__restrict__ input,
                                           uint8_t *__restrict__ flag_pack,
                                           T tol, size_t N);

template <typename T>
__global__ void check_convergence_abs_early_exit(
    const typename CufftTraits<T>::ComplexType *freq_errs, T delta,
    T convergence_tol, int *if_converge, size_t F);

template <typename T>
__global__ void check_convergence_ptw_early_exit(
    const typename CufftTraits<T>::ComplexType *orig_freq,
    const typename CufftTraits<T>::ComplexType *freq_errs, T delta_ptw,
    T convergence_tol, int *if_converge, size_t F);

template <typename T>
__global__ void quantize(const T *arr, UInt *quant, T min_val, T max_val,
                         size_t size);

template <typename T>
__global__ void quantize_and_dequantize(const T *org_arr, T *dequant_arr,
                                        T min_val, T max_val, size_t size,
                                        T thres_lower, T thres_upper);

// SSNR (Spectral Signal-to-Noise Ratio) kernel and function
// SSNR = 10 * log10(sum|X_k|^2 / sum|dX_k|^2)
template <typename T>
__global__ void ssnr_partial_sums_kernel(
    const typename CufftTraits<T>::ComplexType *d_orig_freq,
    const typename CufftTraits<T>::ComplexType *d_freq_err, T *d_partial_orig,
    T *d_partial_err, size_t Nx, size_t Ny, size_t Nz_stored, size_t Nz,
    size_t F);

template <typename T>
T computeSSNR(const typename CufftTraits<T>::ComplexType *d_orig_freq,
              const typename CufftTraits<T>::ComplexType *d_freq_err, size_t Nx,
              size_t Ny, size_t Nz);

// Host function declarations
template <typename T> class ProjectionSolver {
public:
  ProjectionSolver(size_t Nx, size_t Ny, size_t Nz);
  ~ProjectionSolver();

  void initialize(const T *h_orig_data, const T *h_base_data, T epsilon,
                  T delta, size_t max_iterations, T convergence_tol,
                  bool isSpatialABS, bool isFreqABS, bool isFreqPTW);

  void solve_abs();

  void solve_ptw();

  // Methods to get results
  typename HuffmanZstdCompressor<UInt>::CompressionResult
  get_freq_edits_compressed();
  typename HuffmanZstdCompressor<uint8_t>::CompressionResult
  get_freq_flags_compressed();
  typename HuffmanZstdCompressor<UInt>::CompressionResult
  get_spat_edits_compressed();
  typename HuffmanZstdCompressor<uint8_t>::CompressionResult
  get_spat_flags_compressed();
  T get_min_freq_edit();
  T get_max_freq_edit();
  T get_min_spat_edit();
  T get_max_spat_edit();
  size_t get_num_active_freq();
  size_t get_num_active_spat();
  size_t get_iteration_count();
  void calculate_statistics(T *h_base_data);

private:
  ProjectionParams<T> params_;
  IterationResults<T> results_;
  bool initialized_;

  void setup_fft_plans();
  void cleanup_fft_plans();
  void allocate_device_memory();
  void deallocate_device_memory();

  void project_onto_frequency_cube_abs();
  void project_onto_frequency_cube_ptw();
  void project_onto_spatial_cube();
  int check_convergence_abs();
  int check_convergence_ptw();

  void extract_compact_representation();
  void lossless_compression();
};

// Include implementation
#include "projection_algorithm_impl.cuh"