#include "projection_algorithm.h"
#include <algorithm>
#include <cmath>

template <typename T>
void real_imag_to_complex(const T *real_imag,
                          typename FftwTraits<T>::ComplexType *complex,
                          size_t F) {
  for (size_t idx = 0; idx < F; ++idx) {
    complex[idx][0] = real_imag[idx];
    complex[idx][1] = real_imag[idx + F];
  }
}

template <typename T> T compute_mse(const T *err, size_t N) {
  T sum = T(0);
  for (size_t idx = 0; idx < N; ++idx) {
    sum += err[idx] * err[idx];
  }
  return sum;
}

template <typename T>
int check_convergence_abs(const typename FftwTraits<T>::ComplexType *freq_errs,
                          T delta, T convergence_tol, size_t F) {
  for (size_t idx = 0; idx < F; ++idx) {
    if (abs_val(freq_errs[idx][0]) > delta + convergence_tol) {
      return 0;
    }
    if (abs_val(freq_errs[idx][1]) > delta + convergence_tol) {
      return 0;
    }
  }
  return 1;
}

template <typename T>
int check_convergence_ptw(const typename FftwTraits<T>::ComplexType *orig_freq,
                          const typename FftwTraits<T>::ComplexType *freq_errs,
                          T delta_ptw, T convergence_tol, size_t F) {
  for (size_t idx = 0; idx < F; ++idx) {
    T orig_mag = complex_amplitude_squared<T>(orig_freq[idx]);
    typename FftwTraits<T>::ComplexType curr_freq;
    curr_freq[0] = orig_freq[idx][0] + freq_errs[idx][0];
    curr_freq[1] = orig_freq[idx][1] + freq_errs[idx][1];
    T curr_mag = complex_amplitude_squared<T>(curr_freq);
    T curr_ratio = curr_mag / orig_mag;

    if (curr_ratio <= sqrt_val(T(1) + delta_ptw) &&
        curr_ratio >= sqrt_val(T(1) - delta_ptw)) {
      continue;
    }

    T delta = std::min(T(1) - sqrt_val(T(1) - delta_ptw),
                       sqrt_val(T(1) + delta_ptw) - T(1)) /
              sqrt_val(T(2));

    if (abs_val(freq_errs[idx][0]) > delta + convergence_tol) {
      return 0;
    }
    if (abs_val(freq_errs[idx][1]) > delta + convergence_tol) {
      return 0;
    }
  }
  return 1;
}

template <typename T>
void project_frequency_constraints_abs(
    typename FftwTraits<T>::ComplexType *curr_freq_err, T delta, T *freq_edits,
    size_t F) {
  for (size_t idx = 0; idx < F; ++idx) {
    // Process real part
    T curr_err = curr_freq_err[idx][0];
    if (curr_err > delta) {
      freq_edits[idx] += (delta - curr_err);
      curr_freq_err[idx][0] = delta;
    } else if (curr_err < -delta) {
      freq_edits[idx] += (-delta - curr_err);
      curr_freq_err[idx][0] = -delta;
    }

    // Process imaginary part
    curr_err = curr_freq_err[idx][1];
    if (curr_err > delta) {
      freq_edits[idx + F] += (delta - curr_err);
      curr_freq_err[idx][1] = delta;
    } else if (curr_err < -delta) {
      freq_edits[idx + F] += (-delta - curr_err);
      curr_freq_err[idx][1] = -delta;
    }
  }
}

template <typename T>
void project_frequency_constraints_ptw(
    typename FftwTraits<T>::ComplexType *curr_freq_err,
    const typename FftwTraits<T>::ComplexType *orig_freq, T delta_ptw,
    T *freq_edits, size_t F) {
  for (size_t idx = 0; idx < F; ++idx) {
    // DC component
    if (idx == 0) {
      T lower_bound =
          orig_freq[0][0] * (T(1) / sqrt_val(T(1) + delta_ptw) - T(1));
      T upper_bound =
          orig_freq[0][0] * (T(1) / sqrt_val(T(1) - delta_ptw) - T(1));
      T curr_err = curr_freq_err[0][0];
      if (curr_err > upper_bound || curr_err < lower_bound) {
        freq_edits[0] -= curr_err;
        curr_freq_err[0][0] = 0;
      }
      continue;
    }

    // Non-DC components
    T orig_mag = complex_amplitude_squared<T>(orig_freq[idx]);
    typename FftwTraits<T>::ComplexType curr_freq;
    curr_freq[0] = orig_freq[idx][0] + curr_freq_err[idx][0];
    curr_freq[1] = orig_freq[idx][1] + curr_freq_err[idx][1];
    T curr_mag = complex_amplitude_squared<T>(curr_freq);
    T curr_ratio = curr_mag / orig_mag;

    if (curr_ratio <= sqrt_val(T(1) + delta_ptw) &&
        curr_ratio >= sqrt_val(T(1) - delta_ptw)) {
      continue;
    }

    T delta = std::min(T(1) - sqrt_val(T(1) - delta_ptw),
                       sqrt_val(T(1) + delta_ptw) - T(1)) /
              sqrt_val(T(2));

    // Process real part
    T curr_err = curr_freq_err[idx][0];
    if (curr_err > delta) {
      freq_edits[idx] += (delta - curr_err);
      curr_freq_err[idx][0] = delta;
    } else if (curr_err < -delta) {
      freq_edits[idx] += (-delta - curr_err);
      curr_freq_err[idx][0] = -delta;
    }

    // Process imaginary part
    curr_err = curr_freq_err[idx][1];
    if (curr_err > delta) {
      freq_edits[idx + F] += (delta - curr_err);
      curr_freq_err[idx][1] = delta;
    } else if (curr_err < -delta) {
      freq_edits[idx + F] += (-delta - curr_err);
      curr_freq_err[idx][1] = -delta;
    }
  }
}

template <typename T>
void project_spatial_constraints(T *spat_error, T epsilon, T *spat_edits,
                                 size_t N) {
  for (size_t idx = 0; idx < N; ++idx) {
    T curr_err = spat_error[idx];

    if (curr_err > epsilon) {
      spat_edits[idx] += epsilon - curr_err;
      spat_error[idx] = epsilon;
    } else if (curr_err < -epsilon) {
      spat_edits[idx] += -epsilon - curr_err;
      spat_error[idx] = -epsilon;
    }
  }
}

template <typename T>
void create_and_pack_bool_flags(const T *input, uint8_t *flag_pack, T tol,
                                size_t N) {
  size_t num_groups = (N + BITS_PER_FLAG_WORD - 1) / BITS_PER_FLAG_WORD;

  for (size_t group_id = 0; group_id < num_groups; ++group_id) {
    uint32_t flags = 0;
    for (size_t bit_position = 0; bit_position < BITS_PER_FLAG_WORD;
         ++bit_position) {
      size_t idx = group_id * BITS_PER_FLAG_WORD + bit_position;
      if (idx < N && abs_val(input[idx]) > tol) {
        flags |= (1u << bit_position);
      }
    }
    reinterpret_cast<uint32_t *>(flag_pack)[group_id] = flags;
  }
}

template <typename T>
void quantize(const T *arr, UInt *quant, T min_val, T max_val, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    T normalized = (arr[idx] - min_val) / (max_val - min_val);
    if (normalized > T(0.5))
      quant[idx] = static_cast<UInt>(std::ceil(normalized * ((1u << m) - 1)));
    else
      quant[idx] = static_cast<UInt>(std::floor(normalized * ((1u << m) - 1)));
  }
}

template <typename T>
void quantize_and_dequantize(const T *org_arr, T *dequant_arr, T min_val,
                             T max_val, size_t size, T thres_lower,
                             T thres_upper) {
  for (size_t idx = 0; idx < size; ++idx) {
    T org_val = org_arr[idx];
    if (org_val > thres_lower && org_val < thres_upper) {
      dequant_arr[idx] = 0;
    } else {
      T normalized = (org_val - min_val) / (max_val - min_val);
      T quant;
      if (normalized > T(0.5))
        quant = std::ceil(normalized * ((1u << m) - 1));
      else
        quant = std::floor(normalized * ((1u << m) - 1));
      dequant_arr[idx] =
          quant / ((1u << m) - 1) * (max_val - min_val) + min_val;
    }
  }
}

template <typename T>
T computeSSNR(const typename FftwTraits<T>::ComplexType *orig_freq,
              const typename FftwTraits<T>::ComplexType *freq_err, size_t Nx,
              size_t Ny, size_t Nz) {
  size_t Nz_stored = Nz / 2 + 1;
  size_t F = Nx * Ny * Nz_stored;

  T total_orig = T(0);
  T total_err = T(0);

  for (size_t idx = 0; idx < F; ++idx) {
    size_t iz = idx % Nz_stored;

    int weight = (iz > 0 && iz < Nz_stored - 1) ? 2 : 1;
    if (Nz % 2 == 1 && iz == Nz_stored - 1) {
      weight = 2;
    }

    T orig_real = orig_freq[idx][0];
    T orig_imag = orig_freq[idx][1];
    T err_real = freq_err[idx][0];
    T err_imag = freq_err[idx][1];

    total_orig += weight * (orig_real * orig_real + orig_imag * orig_imag);
    total_err += weight * (err_real * err_real + err_imag * err_imag);
  }

  if (total_err == T(0)) {
    return std::numeric_limits<T>::infinity();
  }
  return T(10) * std::log10(total_orig / total_err);
}

// Explicit template instantiations
template void real_imag_to_complex<float>(const float *, fftwf_complex *,
                                          size_t);
template void real_imag_to_complex<double>(const double *, fftw_complex *,
                                           size_t);

template float compute_mse<float>(const float *, size_t);
template double compute_mse<double>(const double *, size_t);

template int check_convergence_abs<float>(const fftwf_complex *, float, float,
                                          size_t);
template int check_convergence_abs<double>(const fftw_complex *, double, double,
                                           size_t);

template int check_convergence_ptw<float>(const fftwf_complex *,
                                          const fftwf_complex *, float, float,
                                          size_t);
template int check_convergence_ptw<double>(const fftw_complex *,
                                           const fftw_complex *, double, double,
                                           size_t);

template void project_frequency_constraints_abs<float>(fftwf_complex *, float,
                                                       float *, size_t);
template void project_frequency_constraints_abs<double>(fftw_complex *, double,
                                                        double *, size_t);

template void project_frequency_constraints_ptw<float>(fftwf_complex *,
                                                       const fftwf_complex *,
                                                       float, float *, size_t);
template void project_frequency_constraints_ptw<double>(fftw_complex *,
                                                        const fftw_complex *,
                                                        double, double *,
                                                        size_t);

template void project_spatial_constraints<float>(float *, float, float *,
                                                 size_t);
template void project_spatial_constraints<double>(double *, double, double *,
                                                  size_t);

template void create_and_pack_bool_flags<float>(const float *, uint8_t *, float,
                                                size_t);
template void create_and_pack_bool_flags<double>(const double *, uint8_t *,
                                                 double, size_t);

template void quantize<float>(const float *, UInt *, float, float, size_t);
template void quantize<double>(const double *, UInt *, double, double, size_t);

template void quantize_and_dequantize<float>(const float *, float *, float,
                                             float, size_t, float, float);
template void quantize_and_dequantize<double>(const double *, double *, double,
                                              double, size_t, double, double);

template float computeSSNR<float>(const fftwf_complex *, const fftwf_complex *,
                                  size_t, size_t, size_t);
template double computeSSNR<double>(const fftw_complex *, const fftw_complex *,
                                    size_t, size_t, size_t);
