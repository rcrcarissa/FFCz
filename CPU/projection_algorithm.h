#pragma once

#include "HuffmanZSTDCoder.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// Structure to hold projection parameters
template <typename T> struct ProjectionParams {
  T *orig_data;  // Original data (Nx*Ny*Nz)
  T *spat_error; // Current spatial error vector (Nx*Ny*Nz)
  typename FftwTraits<T>::ComplexType *orig_freq;  // Original frequency
  typename FftwTraits<T>::ComplexType *freq_error; // Current frequency error
  T delta; // Absolute tolerance for all frequency components
  T delta_ptw;
  T epsilon;         // Spatial constraint tolerance
  size_t Nx, Ny, Nz; // Grid dimensions
  size_t N, F;       // Number of spatial values and frequency modes
  typename FftwTraits<T>::PlanType fft_plan_forward;
  typename FftwTraits<T>::PlanType fft_plan_inverse;
  T spat_min, spat_max, freq_amp_max;
};

// Result structure
template <typename T> struct IterationResults {
  T *freq_edits; // Size: Nx*Ny*(Nz/2+1)*2
  T *spat_edits; // Size: Nx*Ny*Nz

  uint8_t *freq_flag_pack;
  uint8_t *spat_flag_pack;
  T *freq_edit_compact;
  T *spat_edit_compact;

  UInt *freq_edit_compact_quantized;
  UInt *spat_edit_compact_quantized;

  typename HuffmanZstdCompressor<UInt>::CompressionResult freq_edit_compressed;
  typename HuffmanZstdCompressor<uint8_t>::CompressionResult
      freq_flag_compressed;
  typename HuffmanZstdCompressor<UInt>::CompressionResult spat_edit_compressed;
  typename HuffmanZstdCompressor<uint8_t>::CompressionResult
      spat_flag_compressed;

  T freq_min_edit;
  T freq_max_edit;
  T spat_min_edit;
  T spat_max_edit;

  size_t num_active_freq;
  size_t num_active_spat;
  size_t max_iterations;
  size_t iteration_count;
  T convergence_tol;
};

// Helper functions
template <typename T>
inline T
complex_amplitude_squared(const typename FftwTraits<T>::ComplexType &c) {
  return c[0] * c[0] + c[1] * c[1];
}

template <typename T> inline T abs_val(T x) { return std::abs(x); }

template <typename T> inline T sqrt_val(T x) { return std::sqrt(x); }

// Min-max pair
template <typename T> struct MinMaxPair {
  T min;
  T max;
};

template <typename T> T findMaxAbs(const T *arr, size_t N) {
  T max_abs = T(0);
  for (size_t i = 0; i < N; ++i) {
    T val = abs_val(arr[i]);
    if (val > max_abs)
      max_abs = val;
  }
  return max_abs;
}

template <typename T> MinMaxPair<T> findMinMax(const T *arr, size_t N) {
  MinMaxPair<T> result;
  result.min = std::numeric_limits<T>::max();
  result.max = std::numeric_limits<T>::lowest();

  for (size_t i = 0; i < N; ++i) {
    if (arr[i] < result.min)
      result.min = arr[i];
    if (arr[i] > result.max)
      result.max = arr[i];
  }
  return result;
}

template <typename T>
T findMaxAbsComplex(const typename FftwTraits<T>::ComplexType *arr, size_t F) {
  T max_abs = T(0);
  for (size_t i = 0; i < F; ++i) {
    T val = std::max(abs_val(arr[i][0]), abs_val(arr[i][1]));
    if (val > max_abs)
      max_abs = val;
  }
  return max_abs;
}

template <typename T>
T findMaxComplexAmplitude(const typename FftwTraits<T>::ComplexType *arr,
                          size_t F) {
  T max_amp_sq = T(0);
  for (size_t i = 0; i < F; ++i) {
    T amp_sq = complex_amplitude_squared<T>(arr[i]);
    if (amp_sq > max_amp_sq)
      max_amp_sq = amp_sq;
  }
  return sqrt_val(max_amp_sq);
}

template <typename T>
void compactNonZero(const T *input, T *compact, size_t &num_nonzeros, T tol,
                    size_t N) {
  num_nonzeros = 0;
  for (size_t i = 0; i < N; ++i) {
    if (abs_val(input[i]) > tol) {
      compact[num_nonzeros++] = input[i];
    }
  }
}

template <typename T>
void real_imag_to_complex(const T *real_imag,
                          typename FftwTraits<T>::ComplexType *complex,
                          size_t F);

template <typename T> T compute_mse(const T *err, size_t N);

template <typename T>
void project_frequency_constraints_abs(
    typename FftwTraits<T>::ComplexType *curr_freq_err, T delta, T *freq_edits,
    size_t F);

template <typename T>
void project_frequency_constraints_ptw(
    typename FftwTraits<T>::ComplexType *curr_freq_err,
    const typename FftwTraits<T>::ComplexType *orig_freq, T delta,
    T *freq_edits, size_t F);

template <typename T>
void project_spatial_constraints(T *spat_error, T epsilon, T *spat_edits,
                                 size_t N);

template <typename T>
void create_and_pack_bool_flags(const T *input, uint8_t *flag_pack, T tol,
                                size_t N);

template <typename T>
int check_convergence_abs(const typename FftwTraits<T>::ComplexType *freq_errs,
                          T delta, T convergence_tol, size_t F);

template <typename T>
int check_convergence_ptw(const typename FftwTraits<T>::ComplexType *orig_freq,
                          const typename FftwTraits<T>::ComplexType *freq_errs,
                          T delta_ptw, T convergence_tol, size_t F);

template <typename T>
void quantize(const T *arr, UInt *quant, T min_val, T max_val, size_t size);

template <typename T>
void quantize_and_dequantize(const T *org_arr, T *dequant_arr, T min_val,
                             T max_val, size_t size, T thres_lower,
                             T thres_upper);

template <typename T>
T computeSSNR(const typename FftwTraits<T>::ComplexType *orig_freq,
              const typename FftwTraits<T>::ComplexType *freq_err, size_t Nx,
              size_t Ny, size_t Nz);

// ProjectionSolver class
template <typename T> class ProjectionSolver {
public:
  ProjectionSolver(size_t Nx, size_t Ny, size_t Nz);
  ~ProjectionSolver();

  void initialize(const T *h_orig_data, const T *h_base_data, T epsilon,
                  T delta, size_t max_iterations, T convergence_tol,
                  bool isSpatialABS, bool isFreqABS, bool isFreqPTW);

  void solve_abs();
  void solve_ptw();

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
  void allocate_memory();
  void deallocate_memory();

  void project_onto_frequency_cube_abs();
  void project_onto_frequency_cube_ptw();
  void project_onto_spatial_cube();
  int check_convergence_abs_impl();
  int check_convergence_ptw_impl();

  void extract_compact_representation();
  void lossless_compression();
};

#include "projection_algorithm_impl.h"
