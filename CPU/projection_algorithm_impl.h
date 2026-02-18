#pragma once

#include <cstring>
#include <iostream>

// ProjectionSolver implementation
template <typename T>
ProjectionSolver<T>::ProjectionSolver(size_t Nx, size_t Ny, size_t Nz)
    : initialized_(false) {
  params_.Nx = Nx;
  params_.Ny = Ny;
  params_.Nz = Nz;
  params_.N = Nx * Ny * Nz;
  params_.F = Nx * Ny * (Nz / 2 + 1);

  allocate_memory();
  setup_fft_plans();
}

template <typename T> ProjectionSolver<T>::~ProjectionSolver() {
  cleanup_fft_plans();
  deallocate_memory();
}

template <typename T> void ProjectionSolver<T>::allocate_memory() {
  size_t N = params_.N;
  size_t F = params_.F;

  params_.orig_data = new T[N];
  params_.spat_error = new T[N];
  params_.orig_freq = reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
      fftw_malloc(F * sizeof(typename FftwTraits<T>::ComplexType)));
  params_.freq_error = reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
      fftw_malloc(F * sizeof(typename FftwTraits<T>::ComplexType)));

  results_.freq_edits = new T[2 * F]();
  results_.spat_edits = new T[N]();

  size_t freq_flag_bytes =
      ((2 * F + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD) *
      sizeof(uint32_t);
  size_t spat_flag_bytes =
      ((N + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD) * sizeof(uint32_t);
  results_.freq_flag_pack = new uint8_t[freq_flag_bytes]();
  results_.spat_flag_pack = new uint8_t[spat_flag_bytes]();

  results_.freq_edit_compact = new T[2 * F];
  results_.spat_edit_compact = new T[N];
  results_.freq_edit_compact_quantized = new UInt[2 * F];
  results_.spat_edit_compact_quantized = new UInt[N];
}

template <typename T> void ProjectionSolver<T>::deallocate_memory() {
  delete[] params_.orig_data;
  delete[] params_.spat_error;
  if (params_.orig_freq)
    fftw_free(params_.orig_freq);
  if (params_.freq_error)
    fftw_free(params_.freq_error);

  delete[] results_.freq_edits;
  delete[] results_.spat_edits;
  delete[] results_.freq_flag_pack;
  delete[] results_.spat_flag_pack;
  delete[] results_.freq_edit_compact;
  delete[] results_.spat_edit_compact;
  delete[] results_.freq_edit_compact_quantized;
  delete[] results_.spat_edit_compact_quantized;

  if (results_.freq_edit_compressed.compressed_data) {
    results_.freq_edit_compressed.freeSymbolTables();
    results_.freq_edit_compressed.freeCompressedData();
  }
  if (results_.freq_flag_compressed.compressed_data) {
    results_.freq_flag_compressed.freeSymbolTables();
    results_.freq_flag_compressed.freeCompressedData();
  }
  if (results_.spat_edit_compressed.compressed_data) {
    results_.spat_edit_compressed.freeSymbolTables();
    results_.spat_edit_compressed.freeCompressedData();
  }
  if (results_.spat_flag_compressed.compressed_data) {
    results_.spat_flag_compressed.freeSymbolTables();
    results_.spat_flag_compressed.freeCompressedData();
  }
}

template <typename T> void ProjectionSolver<T>::setup_fft_plans() {
  // Allocate temporary arrays for planning
  T *tmp_real = new T[params_.N];
  auto *tmp_complex = reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
      fftw_malloc(params_.F * sizeof(typename FftwTraits<T>::ComplexType)));

  params_.fft_plan_forward = FftwTraits<T>::plan_dft_r2c_3d(
      params_.Nx, params_.Ny, params_.Nz, tmp_real, tmp_complex, FFTW_ESTIMATE);

  params_.fft_plan_inverse = FftwTraits<T>::plan_dft_c2r_3d(
      params_.Nx, params_.Ny, params_.Nz, tmp_complex, tmp_real, FFTW_ESTIMATE);

  delete[] tmp_real;
  fftw_free(tmp_complex);
}

template <typename T> void ProjectionSolver<T>::cleanup_fft_plans() {
  if (params_.fft_plan_forward)
    FftwTraits<T>::destroy_plan(params_.fft_plan_forward);
  if (params_.fft_plan_inverse)
    FftwTraits<T>::destroy_plan(params_.fft_plan_inverse);
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_cube_abs() {
  size_t N = params_.N;
  size_t F = params_.F;

  project_frequency_constraints_abs<T>(params_.freq_error, params_.delta,
                                       results_.freq_edits, F);

  // Transform back to spatial domain
  FftwTraits<T>::execute_dft_c2r(params_.fft_plan_inverse, params_.freq_error,
                                 params_.spat_error);

  // Normalize after inverse FFT
  T norm_factor = T(1) / static_cast<T>(N);
  for (size_t i = 0; i < N; ++i)
    params_.spat_error[i] *= norm_factor;
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_cube_ptw() {
  size_t N = params_.N;
  size_t F = params_.F;

  project_frequency_constraints_ptw<T>(params_.freq_error, params_.orig_freq,
                                       params_.delta_ptw, results_.freq_edits,
                                       F);

  // Transform back to spatial domain
  FftwTraits<T>::execute_dft_c2r(params_.fft_plan_inverse, params_.freq_error,
                                 params_.spat_error);

  // Normalize after inverse FFT
  T norm_factor = T(1) / static_cast<T>(N);
  for (size_t i = 0; i < N; ++i)
    params_.spat_error[i] *= norm_factor;
}

template <typename T> void ProjectionSolver<T>::project_onto_spatial_cube() {
  size_t N = params_.N;
  project_spatial_constraints<T>(params_.spat_error, params_.epsilon,
                                 results_.spat_edits, N);
}

template <typename T>
void ProjectionSolver<T>::extract_compact_representation() {
  size_t N = params_.N;
  size_t F = params_.F;

  results_.freq_min_edit = T(0);
  results_.freq_max_edit = T(0);
  results_.spat_min_edit = T(0);
  results_.spat_max_edit = T(0);

  compactNonZero(results_.freq_edits, results_.freq_edit_compact,
                 results_.num_active_freq, results_.convergence_tol, 2 * F);
  compactNonZero(results_.spat_edits, results_.spat_edit_compact,
                 results_.num_active_spat, results_.convergence_tol, N);

  // If too many edits, store as spatial edits
  if (results_.num_active_freq + results_.num_active_spat > N) {
    results_.num_active_freq = 0;
    results_.num_active_spat = N;

    std::vector<T> freq_edit_to_spat(N);

    // Transform back to spatial domain
    real_imag_to_complex<T>(results_.freq_edits, params_.freq_error, F);
    FftwTraits<T>::execute_dft_c2r(params_.fft_plan_inverse, params_.freq_error,
                                   freq_edit_to_spat.data());

    // Normalize after inverse FFT and add to spatial edits
    T norm_factor = T(1) / static_cast<T>(N);
    for (size_t i = 0; i < N; ++i) {
      results_.spat_edits[i] *= norm_factor;
      results_.spat_edits[i] += freq_edit_to_spat[i];
    }

  } else {
    create_and_pack_bool_flags(results_.freq_edits, results_.freq_flag_pack,
                               results_.convergence_tol, 2 * F);

    if (results_.num_active_freq > 0) {
      auto freq_edit_extreme =
          findMinMax<T>(results_.freq_edit_compact, results_.num_active_freq);

      results_.freq_min_edit = freq_edit_extreme.min;
      results_.freq_max_edit = freq_edit_extreme.max;

      quantize(results_.freq_edit_compact, results_.freq_edit_compact_quantized,
               freq_edit_extreme.min, freq_edit_extreme.max,
               results_.num_active_freq);
    }
  }

  create_and_pack_bool_flags(results_.spat_edits, results_.spat_flag_pack,
                             results_.convergence_tol, N);

  if (results_.num_active_spat > 0) {
    auto spat_edit_extreme =
        findMinMax<T>(results_.spat_edit_compact, results_.num_active_spat);
    results_.spat_min_edit = spat_edit_extreme.min;
    results_.spat_max_edit = spat_edit_extreme.max;

    quantize(results_.spat_edit_compact, results_.spat_edit_compact_quantized,
             spat_edit_extreme.min, spat_edit_extreme.max,
             results_.num_active_spat);
  }
}

template <typename T> void ProjectionSolver<T>::lossless_compression() {
  size_t freq_flag_bytes =
      ((2 * params_.F + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD) *
      sizeof(uint32_t);
  size_t spat_flag_bytes =
      ((params_.N + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD) *
      sizeof(uint32_t);

  if (results_.num_active_freq > 0) {
    results_.freq_edit_compressed = compressHuffmanZstd<UInt>(
        results_.freq_edit_compact_quantized, results_.num_active_freq);
    results_.freq_flag_compressed =
        compressHuffmanZstd<uint8_t>(results_.freq_flag_pack, freq_flag_bytes);
  }

  if (results_.num_active_spat > 0) {
    results_.spat_edit_compressed = compressHuffmanZstd<UInt>(
        results_.spat_edit_compact_quantized, results_.num_active_spat);
    results_.spat_flag_compressed =
        compressHuffmanZstd<uint8_t>(results_.spat_flag_pack, spat_flag_bytes);
  }
}

template <typename T>
void ProjectionSolver<T>::initialize(const T *h_orig_data, const T *h_base_data,
                                     T epsilon, T delta, size_t max_iterations,
                                     T convergence_tol, bool isSpatialABS,
                                     bool isFreqABS, bool isFreqPTW) {
  params_.epsilon = epsilon;
  params_.delta = delta;
  results_.max_iterations = max_iterations;
  results_.convergence_tol = convergence_tol;

  size_t N = params_.N;
  size_t F = params_.F;

  // Copy data
  std::copy(h_orig_data, h_orig_data + N, params_.orig_data);
  std::copy(h_base_data, h_base_data + N, params_.spat_error);

  // Compute spatial errors: spat_error = base_data - orig_data
  for (size_t i = 0; i < N; ++i)
    params_.spat_error[i] -= params_.orig_data[i];

  // Initialize edit accumulators to zero
  std::memset(results_.freq_edits, 0, 2 * F * sizeof(T));
  std::memset(results_.spat_edits, 0, N * sizeof(T));

  size_t freq_flag_bytes =
      ((2 * F + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD) *
      sizeof(uint32_t);
  size_t spat_flag_bytes =
      ((N + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD) * sizeof(uint32_t);
  std::memset(results_.freq_flag_pack, 0, freq_flag_bytes);
  std::memset(results_.spat_flag_pack, 0, spat_flag_bytes);

  // Find min/max of original data
  auto spat_extreme = findMinMax<T>(params_.orig_data, N);
  params_.spat_min = spat_extreme.min;
  params_.spat_max = spat_extreme.max;

  // Compute FFT of original
  FftwTraits<T>::execute_dft_r2c(params_.fft_plan_forward, params_.orig_data,
                                 params_.orig_freq);
  params_.freq_amp_max = findMaxComplexAmplitude<T>(params_.orig_freq, F);

  if (!isSpatialABS) {
    params_.epsilon *= (spat_extreme.max - spat_extreme.min);
  }
  if (!isFreqABS && !isFreqPTW) {
    params_.delta *= params_.freq_amp_max;
  }

  initialized_ = true;
}

template <typename T> int ProjectionSolver<T>::check_convergence_abs_impl() {
  // Transform current spatial error to frequency domain
  FftwTraits<T>::execute_dft_r2c(params_.fft_plan_forward, params_.spat_error,
                                 params_.freq_error);

  return check_convergence_abs<T>(params_.freq_error, params_.delta,
                                  results_.convergence_tol, params_.F);
}

template <typename T> int ProjectionSolver<T>::check_convergence_ptw_impl() {
  // Transform current spatial error to frequency domain
  FftwTraits<T>::execute_dft_r2c(params_.fft_plan_forward, params_.spat_error,
                                 params_.freq_error);

  return check_convergence_ptw<T>(params_.orig_freq, params_.freq_error,
                                  params_.delta, results_.convergence_tol,
                                  params_.F);
}

template <typename T> void ProjectionSolver<T>::solve_abs() {
  if (!initialized_) {
    throw std::runtime_error("Solver not initialized");
  }

  results_.iteration_count = results_.max_iterations;
  for (size_t iter = 0; iter < results_.max_iterations; ++iter) {
    int if_converge = check_convergence_abs_impl();
    if (if_converge) {
      results_.iteration_count = iter;
      break;
    }

    project_onto_frequency_cube_abs();
    project_onto_spatial_cube();
  }

  extract_compact_representation();
  lossless_compression();
}

template <typename T> void ProjectionSolver<T>::solve_ptw() {
  if (!initialized_) {
    throw std::runtime_error("Solver not initialized");
  }

  results_.iteration_count = results_.max_iterations;
  for (size_t iter = 0; iter < results_.max_iterations; ++iter) {
    int if_converge = check_convergence_ptw_impl();
    if (if_converge) {
      results_.iteration_count = iter;
      break;
    }

    project_onto_frequency_cube_ptw();
    project_onto_spatial_cube();
  }

  extract_compact_representation();
  lossless_compression();
}

template <typename T>
typename HuffmanZstdCompressor<UInt>::CompressionResult
ProjectionSolver<T>::get_freq_edits_compressed() {
  return results_.freq_edit_compressed;
}

template <typename T>
typename HuffmanZstdCompressor<uint8_t>::CompressionResult
ProjectionSolver<T>::get_freq_flags_compressed() {
  return results_.freq_flag_compressed;
}

template <typename T>
typename HuffmanZstdCompressor<UInt>::CompressionResult
ProjectionSolver<T>::get_spat_edits_compressed() {
  return results_.spat_edit_compressed;
}

template <typename T>
typename HuffmanZstdCompressor<uint8_t>::CompressionResult
ProjectionSolver<T>::get_spat_flags_compressed() {
  return results_.spat_flag_compressed;
}

template <typename T> T ProjectionSolver<T>::get_min_freq_edit() {
  return results_.freq_min_edit;
}

template <typename T> T ProjectionSolver<T>::get_max_freq_edit() {
  return results_.freq_max_edit;
}

template <typename T> T ProjectionSolver<T>::get_min_spat_edit() {
  return results_.spat_min_edit;
}

template <typename T> T ProjectionSolver<T>::get_max_spat_edit() {
  return results_.spat_max_edit;
}

template <typename T> size_t ProjectionSolver<T>::get_num_active_freq() {
  return results_.num_active_freq;
}

template <typename T> size_t ProjectionSolver<T>::get_num_active_spat() {
  return results_.num_active_spat;
}

template <typename T> size_t ProjectionSolver<T>::get_iteration_count() {
  return results_.iteration_count;
}

template <typename T>
void ProjectionSolver<T>::calculate_statistics(T *h_base_data) {
  T mae = 0, mse = 0, rmse, nrmse, psnr, mrfe, ssnr;
  size_t N = params_.N;
  size_t F = params_.F;

  std::vector<T> freq_edit_decomp(2 * F);
  std::vector<T> spat_edit_decomp(N);
  std::vector<T> decomp_err(N);
  auto *freq_edit_decomp_complex =
      reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
          fftw_malloc(F * sizeof(typename FftwTraits<T>::ComplexType)));
  auto *decomp_err_freq =
      reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
          fftw_malloc(F * sizeof(typename FftwTraits<T>::ComplexType)));

  // Frequency edits: quantize-dequantize, or copy directly if min == max
  if (results_.freq_min_edit != results_.freq_max_edit) {
    T normalized_0 = -results_.freq_min_edit /
                     (results_.freq_max_edit - results_.freq_min_edit);
    T quant_0;
    if (normalized_0 > T(0.5))
      quant_0 = std::ceil(normalized_0 * ((1u << m) - 1));
    else
      quant_0 = std::floor(normalized_0 * ((1u << m) - 1));
    T thres_lower = quant_0 / ((1u << m) - 1) *
                        (results_.freq_max_edit - results_.freq_min_edit) +
                    results_.freq_min_edit;
    T thres_upper = (quant_0 + 1) / ((1u << m) - 1) *
                        (results_.freq_max_edit - results_.freq_min_edit) +
                    results_.freq_min_edit;
    quantize_and_dequantize(results_.freq_edits, freq_edit_decomp.data(),
                            results_.freq_min_edit, results_.freq_max_edit,
                            2 * F, thres_lower, thres_upper);
  } else {
    std::copy(results_.freq_edits, results_.freq_edits + 2 * F,
              freq_edit_decomp.data());
  }

  // Spatial edits: quantize-dequantize, or copy directly if min == max
  if (results_.spat_min_edit != results_.spat_max_edit) {
    T normalized_0 = -results_.spat_min_edit /
                     (results_.spat_max_edit - results_.spat_min_edit);
    T quant_0;
    if (normalized_0 > T(0.5))
      quant_0 = std::ceil(normalized_0 * ((1u << m) - 1));
    else
      quant_0 = std::floor(normalized_0 * ((1u << m) - 1));
    T thres_lower = quant_0 / ((1u << m) - 1) *
                        (results_.spat_max_edit - results_.spat_min_edit) +
                    results_.spat_min_edit;
    T thres_upper = (quant_0 + 1) / ((1u << m) - 1) *
                        (results_.spat_max_edit - results_.spat_min_edit) +
                    results_.spat_min_edit;
    quantize_and_dequantize(results_.spat_edits, spat_edit_decomp.data(),
                            results_.spat_min_edit, results_.spat_max_edit, N,
                            thres_lower, thres_upper);
  } else {
    std::copy(results_.spat_edits, results_.spat_edits + N,
              spat_edit_decomp.data());
  }

  real_imag_to_complex<T>(freq_edit_decomp.data(), freq_edit_decomp_complex, F);

  FftwTraits<T>::execute_dft_c2r(params_.fft_plan_inverse,
                                 freq_edit_decomp_complex, decomp_err.data());

  T norm_factor = T(1) / static_cast<T>(N);
  for (size_t i = 0; i < N; ++i) {
    decomp_err[i] *= norm_factor;
    decomp_err[i] +=
        h_base_data[i] + spat_edit_decomp[i] - params_.orig_data[i];
  }

  mae = findMaxAbs(decomp_err.data(), N);
  mse = compute_mse(decomp_err.data(), N) / N;
  rmse = sqrt_val(mse);
  nrmse = rmse / (params_.spat_max - params_.spat_min);
  psnr = -20 * std::log10(nrmse);

  FftwTraits<T>::execute_dft_r2c(params_.fft_plan_forward, decomp_err.data(),
                                 decomp_err_freq);

  mrfe = findMaxAbsComplex<T>(decomp_err_freq, F);
  mrfe /= params_.freq_amp_max;
  ssnr = computeSSNR<T>(params_.orig_freq, decomp_err_freq, params_.Nx,
                        params_.Ny, params_.Nz);

  std::cout << "MAE: " << mae << std::endl;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "RMSE: " << rmse << std::endl;
  std::cout << "NRMSE: " << nrmse << std::endl;
  std::cout << "PNSR: " << psnr << " dB" << std::endl;
  mrfe = findMaxAbsComplex<T>(decomp_err_freq, F);
  std::cout << "max absolute frequency error: " << mrfe << std::endl;
  mrfe /= params_.freq_amp_max;
  std::cout << "max relative frequency error: " << mrfe << std::endl;
  std::cout << "SSNR: " << ssnr << " dB" << std::endl;

  fftw_free(freq_edit_decomp_complex);
  fftw_free(decomp_err_freq);
}
