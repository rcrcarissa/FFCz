#pragma once

// ProjectionSolver host function implementation
template <typename T>
ProjectionSolver<T>::ProjectionSolver(size_t Nx, size_t Ny, size_t Nz)
    : initialized_(false) {
  params_.Nx = Nx;
  params_.Ny = Ny;
  params_.Nz = Nz;
  params_.N = Nx * Ny * Nz;
  params_.F = Nx * Ny * (Nz / 2 + 1);

  allocate_device_memory();
  setup_fft_plans();
}

template <typename T> ProjectionSolver<T>::~ProjectionSolver() {
  cleanup_fft_plans();
  deallocate_device_memory();
}

template <typename T> void ProjectionSolver<T>::allocate_device_memory() {
  size_t N = params_.N;
  size_t F = params_.F;

  // Core data arrays
  CHECK_CUDA(cudaMalloc(&params_.d_orig_data, N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&params_.d_spat_error, N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&params_.d_orig_freq,
                        F * sizeof(typename CufftTraits<T>::ComplexType)));
  CHECK_CUDA(cudaMalloc(&params_.d_freq_error,
                        F * sizeof(typename CufftTraits<T>::ComplexType)));

  // Full-size accumulator arrays
  CHECK_CUDA(cudaMalloc(&results_.d_freq_edits, 2 * F * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&results_.d_spat_edits, N * sizeof(T)));

  // Flag arrays
  size_t freq_flag_bytes = (2 * F + 31) / 32 * 4;
  size_t spat_flag_bytes = (N + 31) / 32 * 4;
  CHECK_CUDA(cudaMalloc(&results_.d_freq_flag_pack,
                        freq_flag_bytes * sizeof(uint8_t)));
  CHECK_CUDA(cudaMalloc(&results_.d_spat_flag_pack,
                        spat_flag_bytes * sizeof(uint8_t)));

  // Compact edit arrays (pre-allocate maximum possible size)
  CHECK_CUDA(cudaMalloc(&results_.d_freq_edit_compact, 2 * F * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&results_.d_spat_edit_compact, N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&results_.d_freq_edit_compact_quantized,
                        2 * F * sizeof(UInt)));
  CHECK_CUDA(
      cudaMalloc(&results_.d_spat_edit_compact_quantized, N * sizeof(UInt)));
}

template <typename T> void ProjectionSolver<T>::deallocate_device_memory() {
  // Core arrays
  if (params_.d_orig_data)
    CHECK_CUDA(cudaFree(params_.d_orig_data));
  if (params_.d_spat_error)
    CHECK_CUDA(cudaFree(params_.d_spat_error));
  if (params_.d_orig_freq)
    CHECK_CUDA(cudaFree(params_.d_orig_freq));
  if (params_.d_freq_error)
    CHECK_CUDA(cudaFree(params_.d_freq_error));

  // Accumulator arrays
  if (results_.d_freq_edits)
    CHECK_CUDA(cudaFree(results_.d_freq_edits));
  if (results_.d_spat_edits)
    CHECK_CUDA(cudaFree(results_.d_spat_edits));

  // Flag and compact arrays
  if (results_.d_freq_flag_pack)
    CHECK_CUDA(cudaFree(results_.d_freq_flag_pack));
  if (results_.d_spat_flag_pack)
    CHECK_CUDA(cudaFree(results_.d_spat_flag_pack));
  if (results_.d_freq_edit_compact)
    CHECK_CUDA(cudaFree(results_.d_freq_edit_compact));
  if (results_.d_spat_edit_compact)
    CHECK_CUDA(cudaFree(results_.d_spat_edit_compact));
  if (results_.d_freq_edit_compact_quantized)
    CHECK_CUDA(cudaFree(results_.d_freq_edit_compact_quantized));
  if (results_.d_spat_edit_compact_quantized)
    CHECK_CUDA(cudaFree(results_.d_spat_edit_compact_quantized));
  if (results_.h_freq_edit_compressed.compressed_data) {
    results_.h_freq_edit_compressed.freeSymbolTables();
    results_.h_freq_edit_compressed.freeCompressedData();
  }
  if (results_.h_freq_flag_compressed.compressed_data) {
    results_.h_freq_flag_compressed.freeSymbolTables();
    results_.h_freq_flag_compressed.freeCompressedData();
  }
  if (results_.h_spat_edit_compressed.compressed_data) {
    results_.h_spat_edit_compressed.freeSymbolTables();
    results_.h_spat_edit_compressed.freeCompressedData();
  }
  if (results_.h_spat_flag_compressed.compressed_data) {
    results_.h_spat_flag_compressed.freeSymbolTables();
    results_.h_spat_flag_compressed.freeCompressedData();
  }
}

template <typename T> void ProjectionSolver<T>::setup_fft_plans() {
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_forward, params_.Nx, params_.Ny,
                            params_.Nz, CUFFT_R2C));
    CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_inverse, params_.Nx, params_.Ny,
                            params_.Nz, CUFFT_C2R));
  } else {
    CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_forward, params_.Nx, params_.Ny,
                            params_.Nz, CUFFT_D2Z));
    CHECK_CUFFT(cufftPlan3d(&params_.fft_plan_inverse, params_.Nx, params_.Ny,
                            params_.Nz, CUFFT_Z2D));
  }
}

template <typename T> void ProjectionSolver<T>::cleanup_fft_plans() {
  if (params_.fft_plan_forward)
    cufftDestroy(params_.fft_plan_forward);
  if (params_.fft_plan_inverse)
    cufftDestroy(params_.fft_plan_inverse);
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_cube_abs() {
  size_t N = params_.N;
  size_t F = params_.F;

  // Project frequency components and update edits
  dim3 block(BLOCK_SIZE);
  dim3 freq_grid((F + block.x - 1) / block.x);

  project_frequency_constraints_abs<T><<<freq_grid, block>>>(
      params_.d_freq_error, params_.delta, results_.d_freq_edits, F);
  CHECK_CUDA(cudaGetLastError());

  // Transform back to spatial domain
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse,
                             (cufftComplex *)params_.d_freq_error,
                             params_.d_spat_error));
  } else {
    CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse,
                             (cufftDoubleComplex *)params_.d_freq_error,
                             params_.d_spat_error));
  }

  // Normalize after inverse FFT
  T norm_factor = T(1) / static_cast<T>(N);
  dim3 spat_grid((N + block.x - 1) / block.x);
  normalize_kernel<T>
      <<<spat_grid, block>>>(params_.d_spat_error, norm_factor, N);
  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::project_onto_frequency_cube_ptw() {
  size_t N = params_.N;
  size_t F = params_.F;

  // Project frequency components and update edits
  dim3 block(BLOCK_SIZE);
  dim3 freq_grid((F + block.x - 1) / block.x);

  project_frequency_constraints_ptw<T>
      <<<freq_grid, block>>>(params_.d_freq_error, params_.d_orig_freq,
                             params_.delta_ptw, results_.d_freq_edits, F);

  CHECK_CUDA(cudaGetLastError());

  // Transform back to spatial domain
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse,
                             (cufftComplex *)params_.d_freq_error,
                             params_.d_spat_error));
  } else {
    CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse,
                             (cufftDoubleComplex *)params_.d_freq_error,
                             params_.d_spat_error));
  }

  // Normalize after inverse FFT
  T norm_factor = T(1) / static_cast<T>(N);
  dim3 spat_grid((N + block.x - 1) / block.x);
  normalize_kernel<T>
      <<<spat_grid, block>>>(params_.d_spat_error, norm_factor, N);
  CHECK_CUDA(cudaGetLastError());
}

template <typename T> void ProjectionSolver<T>::project_onto_spatial_cube() {
  size_t N = params_.N;
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + block.x - 1) / block.x);

  project_spatial_constraints<T><<<grid, block>>>(
      params_.d_spat_error, params_.epsilon, results_.d_spat_edits, N);

  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void ProjectionSolver<T>::extract_compact_representation() {
  size_t N = params_.N;
  size_t F = params_.F;
  dim3 block(BLOCK_SIZE);
  dim3 spat_grid((N + block.x - 1) / block.x);

  compactNonZero(results_.d_freq_edits, results_.d_freq_edit_compact,
                 results_.h_num_active_freq, results_.convergence_tol, 2 * F);
  compactNonZero(results_.d_spat_edits, results_.d_spat_edit_compact,
                 results_.h_num_active_spat, results_.convergence_tol, N);

  // If the number of edits are more than N, just store as spatial edits
  if (results_.h_num_active_freq + results_.h_num_active_spat > N) {
    results_.h_num_active_freq = 0;
    results_.h_num_active_spat = N;

    T *d_freq_edit_to_spat;
    CHECK_CUDA(cudaMalloc(&d_freq_edit_to_spat, N * sizeof(T)));

    // Transform back to spatial domain
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse,
                               (cufftComplex *)results_.d_freq_edits,
                               d_freq_edit_to_spat));
    } else {
      CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse,
                               (cufftDoubleComplex *)results_.d_freq_edits,
                               d_freq_edit_to_spat));
    }

    // Normalize after inverse FFT
    T norm_factor = T(1) / static_cast<T>(N);
    normalize_add_kernel<T><<<spat_grid, block>>>(
        d_freq_edit_to_spat, results_.d_spat_edits, norm_factor, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_freq_edit_to_spat));

  } else {
    dim3 freq_grid((2 * F + block.x - 1) / block.x);
    create_and_pack_bool_flags<<<freq_grid, block>>>(
        results_.d_freq_edits, results_.d_freq_flag_pack,
        results_.convergence_tol, 2 * F);
    CHECK_CUDA(cudaGetLastError());

    if (results_.h_num_active_freq > 0) {
      auto freq_edit_extreme = findMinMax<T>(results_.d_freq_edit_compact,
                                             results_.h_num_active_freq);

      results_.h_freq_min_edit = freq_edit_extreme.min;
      results_.h_freq_max_edit = freq_edit_extreme.max;

      dim3 freq_compact_grid((results_.h_num_active_freq + block.x - 1) /
                             block.x);
      quantize<<<freq_compact_grid, block>>>(
          results_.d_freq_edit_compact, results_.d_freq_edit_compact_quantized,
          freq_edit_extreme.min, freq_edit_extreme.max,
          results_.h_num_active_freq);
      CHECK_CUDA(cudaGetLastError());
    }
  }

  create_and_pack_bool_flags<<<spat_grid, block>>>(results_.d_spat_edits,
                                                   results_.d_spat_flag_pack,
                                                   results_.convergence_tol, N);
  CHECK_CUDA(cudaGetLastError());

  if (results_.h_num_active_spat > 0) {
    auto spat_edit_extreme =
        findMinMax<T>(results_.d_spat_edit_compact, results_.h_num_active_spat);
    results_.h_spat_min_edit = spat_edit_extreme.min;
    results_.h_spat_max_edit = spat_edit_extreme.max;

    dim3 spat_compact_grid((results_.h_num_active_spat + block.x - 1) /
                           block.x);
    quantize<<<spat_compact_grid, block>>>(
        results_.d_spat_edit_compact, results_.d_spat_edit_compact_quantized,
        spat_edit_extreme.min, spat_edit_extreme.max,
        results_.h_num_active_spat);
    CHECK_CUDA(cudaGetLastError());
  }
}

template <typename T> void ProjectionSolver<T>::lossless_compression() {
  size_t freq_flag_bytes = (2 * params_.F + 31) / 32 * 4;
  size_t spat_flag_bytes = (params_.N + 31) / 32 * 4;

  if (results_.h_num_active_freq > 0) {
    results_.h_freq_edit_compressed = compressHuffmanZstd<UInt>(
        results_.d_freq_edit_compact_quantized, results_.h_num_active_freq);
    results_.h_freq_flag_compressed = compressHuffmanZstd<uint8_t>(
        results_.d_freq_flag_pack, freq_flag_bytes);
  }

  if (results_.h_num_active_spat > 0) {
    results_.h_spat_edit_compressed = compressHuffmanZstd<UInt>(
        results_.d_spat_edit_compact_quantized, results_.h_num_active_spat);
    results_.h_spat_flag_compressed = compressHuffmanZstd<uint8_t>(
        results_.d_spat_flag_pack, spat_flag_bytes);
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

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(params_.d_orig_data, h_orig_data, N * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(params_.d_spat_error, h_base_data, N * sizeof(T),
                        cudaMemcpyHostToDevice));

  // Compute spatial errors
  size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  difference_kernel<T><<<num_blocks, BLOCK_SIZE>>>(params_.d_orig_data,
                                                   params_.d_spat_error, N);

  // Initialize edit accumulators & edits to zero
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  CHECK_CUDA(
      cudaMemsetAsync(results_.d_freq_edits, 0, 2 * F * sizeof(T), stream));
  CHECK_CUDA(cudaMemsetAsync(results_.d_spat_edits, 0, N * sizeof(T), stream));

  size_t freq_flag_bytes = (2 * F + 31) / 32 * 4;
  size_t spat_flag_bytes = (N + 31) / 32 * 4;
  CHECK_CUDA(cudaMemset(results_.d_freq_flag_pack, 0, freq_flag_bytes));
  CHECK_CUDA(cudaMemset(results_.d_spat_flag_pack, 0, spat_flag_bytes));

  // Derive absolute error bound for different modes
  auto spat_extreme = findMinMax<T>(params_.d_orig_data, N);
  params_.spat_min = spat_extreme.min;
  params_.spat_max = spat_extreme.max;
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_orig_data,
                             (cufftComplex *)params_.d_orig_freq));
  } else {
    CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_orig_data,
                             (cufftDoubleComplex *)params_.d_orig_freq));
  }
  params_.freq_amp_max = findMaxComplexAmplitude<T>(params_.d_orig_freq, F);

  if (!isSpatialABS) {
    params_.epsilon *= (spat_extreme.max - spat_extreme.min);
  }
  if (!isFreqABS && !isFreqPTW) {
    params_.delta *= params_.freq_amp_max;
  }

  initialized_ = true;
}

// Method for checking only frequency domain convergence
template <typename T> int ProjectionSolver<T>::check_convergence_abs() {
  // Transform current spatial error to frequency domain
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_spat_error,
                             (cufftComplex *)params_.d_freq_error));
  } else {
    CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_spat_error,
                             (cufftDoubleComplex *)params_.d_freq_error));
  }

  int *d_if_converge;
  CHECK_CUDA(cudaMalloc(&d_if_converge, sizeof(int)));
  int init_val = 1;
  CHECK_CUDA(cudaMemcpy(d_if_converge, &init_val, sizeof(int),
                        cudaMemcpyHostToDevice));

  size_t F = params_.F;
  size_t num_blocks = (F + BLOCK_SIZE - 1) / BLOCK_SIZE;
  check_convergence_abs_early_exit<T>
      <<<num_blocks, BLOCK_SIZE>>>(params_.d_freq_error, params_.delta,
                                   results_.convergence_tol, d_if_converge, F);

  int if_converge;
  CHECK_CUDA(cudaMemcpy(&if_converge, d_if_converge, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_if_converge));
  return if_converge;
}

template <typename T> int ProjectionSolver<T>::check_convergence_ptw() {
  // Transform current spatial error to frequency domain
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, params_.d_spat_error,
                             (cufftComplex *)params_.d_freq_error));
  } else {
    CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, params_.d_spat_error,
                             (cufftDoubleComplex *)params_.d_freq_error));
  }

  int *d_if_converge;
  CHECK_CUDA(cudaMalloc(&d_if_converge, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_if_converge, 1, sizeof(int)));

  size_t F = params_.F;
  size_t num_blocks = (F + BLOCK_SIZE - 1) / BLOCK_SIZE;
  check_convergence_ptw_early_exit<T><<<num_blocks, BLOCK_SIZE>>>(
      params_.d_orig_freq, params_.d_freq_error, params_.delta,
      results_.convergence_tol, d_if_converge, F);

  int if_converge;
  CHECK_CUDA(cudaMemcpy(&if_converge, d_if_converge, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_if_converge));
  return if_converge;
}

template <typename T> void ProjectionSolver<T>::solve_abs() {
  if (!initialized_) {
    throw std::runtime_error("Solver not initialized");
  }

  results_.iteration_count = results_.max_iterations;
  for (size_t iter = 0; iter < results_.max_iterations; ++iter) {
    // Check spatial convergence
    int if_converge = check_convergence_abs();
    if (if_converge) {
      results_.iteration_count = iter;
      break;
    }

    // Project onto frequency polytope (accumulates in freq_edits)
    project_onto_frequency_cube_abs();

    // Project onto spatial box (accumulates in spat_edits)
    project_onto_spatial_cube();
  }

  // Extract compact representation from edits
  extract_compact_representation();

  // Lossless compression
  lossless_compression();
}

template <typename T> void ProjectionSolver<T>::solve_ptw() {
  if (!initialized_) {
    throw std::runtime_error("Solver not initialized");
  }

  results_.iteration_count = results_.max_iterations;
  for (size_t iter = 0; iter < results_.max_iterations; ++iter) {
    // Check spatial convergence
    int if_converge = check_convergence_ptw();
    if (if_converge) {
      results_.iteration_count = iter;
      break;
    }

    // Project onto frequency polytope (accumulates in freq_edits)
    project_onto_frequency_cube_ptw();

    // Project onto spatial box (accumulates in spat_edits)
    project_onto_spatial_cube();
  }

  // Extract compact representation from edits
  extract_compact_representation();

  // Lossless compression
  lossless_compression();
}

template <typename T>
typename HuffmanZstdCompressor<UInt>::CompressionResult
ProjectionSolver<T>::get_freq_edits_compressed() {
  return results_.h_freq_edit_compressed;
  ;
}

template <typename T>
typename HuffmanZstdCompressor<uint8_t>::CompressionResult
ProjectionSolver<T>::get_freq_flags_compressed() {
  return results_.h_freq_flag_compressed;
}

template <typename T>
typename HuffmanZstdCompressor<UInt>::CompressionResult
ProjectionSolver<T>::get_spat_edits_compressed() {
  return results_.h_spat_edit_compressed;
}

template <typename T>
typename HuffmanZstdCompressor<uint8_t>::CompressionResult
ProjectionSolver<T>::get_spat_flags_compressed() {
  return results_.h_spat_flag_compressed;
}

template <typename T> T ProjectionSolver<T>::get_min_freq_edit() {
  return results_.h_freq_min_edit;
}

template <typename T> T ProjectionSolver<T>::get_max_freq_edit() {
  return results_.h_freq_max_edit;
}

template <typename T> T ProjectionSolver<T>::get_min_spat_edit() {
  return results_.h_spat_min_edit;
}

template <typename T> T ProjectionSolver<T>::get_max_spat_edit() {
  return results_.h_spat_max_edit;
}

template <typename T> size_t ProjectionSolver<T>::get_num_active_freq() {
  return results_.h_num_active_freq;
}

template <typename T> size_t ProjectionSolver<T>::get_num_active_spat() {
  return results_.h_num_active_spat;
}

template <typename T> size_t ProjectionSolver<T>::get_iteration_count() {
  return results_.iteration_count;
}

template <typename T>
void ProjectionSolver<T>::calculate_statistics(T *h_base_data) {
  T mae = 0, mse = 0, rmse, nrmse, psnr, mrfe, ssnr;
  size_t N = params_.N;
  size_t F = params_.F;

  T *d_freq_edit_decomp, *d_spat_edit_decomp, *d_base_data, *d_decomp_err;
  typename CufftTraits<T>::ComplexType *d_freq_edit_decomp_complex,
      *d_decomp_err_freq;
  CHECK_CUDA(cudaMalloc(&d_freq_edit_decomp, 2 * F * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_spat_edit_decomp, N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_base_data, N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_decomp_err, N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_freq_edit_decomp_complex,
                        F * sizeof(typename CufftTraits<T>::ComplexType)));
  CHECK_CUDA(cudaMalloc(&d_decomp_err_freq,
                        F * sizeof(typename CufftTraits<T>::ComplexType)));
  CHECK_CUDA(cudaMemcpy(d_base_data, h_base_data, N * sizeof(T),
                        cudaMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE);
  dim3 freq_grid((2 * F + block.x - 1) / block.x);
  // Values quantized as 0 are dequantized to be 0
  T normalized_0 = -results_.h_freq_min_edit /
                   (results_.h_freq_max_edit - results_.h_freq_min_edit);
  T quant_0;
  if (normalized_0 > 0.5)
    quant_0 = ceil(normalized_0 * ((1u << m) - 1));
  else
    quant_0 = floor(normalized_0 * ((1u << m) - 1));
  T thres_lower = quant_0 / ((1u << m) - 1) *
                      (results_.h_freq_max_edit - results_.h_freq_min_edit) +
                  results_.h_freq_min_edit;
  T thres_upper = (quant_0 + 1) / ((1u << m) - 1) *
                      (results_.h_freq_max_edit - results_.h_freq_min_edit) +
                  results_.h_freq_min_edit;
  quantize_and_dequantize<<<freq_grid, block>>>(
      results_.d_freq_edits, d_freq_edit_decomp, results_.h_freq_min_edit,
      results_.h_freq_max_edit, 2 * F, thres_lower, thres_upper);

  dim3 spat_grid((N + block.x - 1) / block.x);
  normalized_0 = -results_.h_spat_min_edit /
                 (results_.h_spat_max_edit - results_.h_spat_min_edit);
  if (normalized_0 > 0.5)
    quant_0 = ceil(normalized_0 * ((1u << m) - 1));
  else
    quant_0 = floor(normalized_0 * ((1u << m) - 1));
  thres_lower = quant_0 / ((1u << m) - 1) *
                    (results_.h_spat_max_edit - results_.h_spat_min_edit) +
                results_.h_spat_min_edit;
  thres_upper = (quant_0 + 1) / ((1u << m) - 1) *
                    (results_.h_spat_max_edit - results_.h_spat_min_edit) +
                results_.h_spat_min_edit;
  quantize_and_dequantize<<<spat_grid, block>>>(
      results_.d_spat_edits, d_spat_edit_decomp, results_.h_spat_min_edit,
      results_.h_spat_max_edit, N, thres_lower, thres_upper);

  dim3 half_freq_grid((F + block.x - 1) / block.x);
  real_imag_to_complex<T><<<half_freq_grid, block>>>(
      d_freq_edit_decomp, d_freq_edit_decomp_complex, F);

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecC2R(params_.fft_plan_inverse,
                             (cufftComplex *)d_freq_edit_decomp_complex,
                             d_decomp_err));
  } else {
    CHECK_CUFFT(cufftExecZ2D(params_.fft_plan_inverse,
                             (cufftDoubleComplex *)d_freq_edit_decomp_complex,
                             d_decomp_err));
  }

  T norm_factor = T(1) / static_cast<T>(N);
  normalize_add_two_difference_kernel<T>
      <<<spat_grid, block>>>(d_base_data, d_spat_edit_decomp,
                             params_.d_orig_data, d_decomp_err, norm_factor, N);
  CHECK_CUDA(cudaGetLastError());

  mae = findMaxAbs(d_decomp_err, N);

  T *d_partial_sums;
  CHECK_CUDA(cudaMalloc(&d_partial_sums, spat_grid.x * sizeof(T)));

  size_t shared_mem_size = block.x * sizeof(T);
  mse_partial_sums_kernel<T>
      <<<spat_grid, block, shared_mem_size>>>(d_decomp_err, d_partial_sums, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  T *d_total_sum;
  CHECK_CUDA(cudaMalloc(&d_total_sum, sizeof(T)));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_partial_sums,
                         d_total_sum, spat_grid.x);
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_partial_sums,
                         d_total_sum, spat_grid.x);

  CHECK_CUDA(cudaMemcpy(&mse, d_total_sum, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_partial_sums));
  CHECK_CUDA(cudaFree(d_total_sum));
  CHECK_CUDA(cudaFree(d_temp_storage));
  mse /= N;
  rmse = sqrt_val(mse);
  nrmse = rmse / (params_.spat_max - params_.spat_min);
  psnr = -20 * log10(nrmse);

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUFFT(cufftExecR2C(params_.fft_plan_forward, d_decomp_err,
                             (cufftComplex *)d_decomp_err_freq));
  } else {
    CHECK_CUFFT(cufftExecD2Z(params_.fft_plan_forward, d_decomp_err,
                             (cufftDoubleComplex *)d_decomp_err_freq));
  }

  std::cout << "MAE: " << mae << std::endl;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "RMSE: " << rmse << std::endl;
  std::cout << "NRMSE: " << nrmse << std::endl;
  std::cout << "PNSR: " << psnr << " dB" << std::endl;
  mrfe = findMaxAbsComplex<T>(d_decomp_err_freq, F);
  std::cout << "max absolute frequency error: " << mrfe << std::endl;
  mrfe /= params_.freq_amp_max;
  std::cout << "max relative frequency error: " << mrfe << std::endl;
  ssnr = computeSSNR<T>(params_.d_orig_freq, d_decomp_err_freq, params_.Nx,
                        params_.Ny, params_.Nz);
  std::cout << "SSNR: " << ssnr << " dB" << std::endl;

  CHECK_CUDA(cudaFree(d_freq_edit_decomp));
  CHECK_CUDA(cudaFree(d_spat_edit_decomp));
  CHECK_CUDA(cudaFree(d_base_data));
  CHECK_CUDA(cudaFree(d_decomp_err));
  CHECK_CUDA(cudaFree(d_freq_edit_decomp_complex));
  CHECK_CUDA(cudaFree(d_decomp_err_freq));
}
