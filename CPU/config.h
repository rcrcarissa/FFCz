#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <complex>
#include <fftw3.h>

// Constants
inline constexpr size_t CHUNK_SIZE = 256;  // Processing batch size
inline constexpr size_t BITS_PER_FLAG_WORD = 32;  // Bits in uint32_t for packed flags
inline constexpr size_t m = 16;  // Quantization bits

// Type for quantization codes
template <size_t M>
using UIntForSize = std::conditional_t<
    M == 8, uint8_t,
    std::conditional_t<M == 16, uint16_t,
                       std::conditional_t<M == 32, uint32_t, void>>>;

using UInt = UIntForSize<m>;

// FFTW traits for complex types
template <typename T> struct FftwTraits;

template <> struct FftwTraits<float> {
  using ComplexType = fftwf_complex;
  using PlanType = fftwf_plan;

  static PlanType plan_dft_r2c_3d(int n0, int n1, int n2, float* in, ComplexType* out, unsigned flags) {
    return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
  }

  static PlanType plan_dft_c2r_3d(int n0, int n1, int n2, ComplexType* in, float* out, unsigned flags) {
    return fftwf_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
  }

  static void execute(PlanType plan) { fftwf_execute(plan); }
  static void destroy_plan(PlanType plan) { fftwf_destroy_plan(plan); }
  static void execute_dft_r2c(PlanType plan, float* in, ComplexType* out) {
    fftwf_execute_dft_r2c(plan, in, out);
  }
  static void execute_dft_c2r(PlanType plan, ComplexType* in, float* out) {
    fftwf_execute_dft_c2r(plan, in, out);
  }
};

template <> struct FftwTraits<double> {
  using ComplexType = fftw_complex;
  using PlanType = fftw_plan;

  static PlanType plan_dft_r2c_3d(int n0, int n1, int n2, double* in, ComplexType* out, unsigned flags) {
    return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
  }

  static PlanType plan_dft_c2r_3d(int n0, int n1, int n2, ComplexType* in, double* out, unsigned flags) {
    return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
  }

  static void execute(PlanType plan) { fftw_execute(plan); }
  static void destroy_plan(PlanType plan) { fftw_destroy_plan(plan); }
  static void execute_dft_r2c(PlanType plan, double* in, ComplexType* out) {
    fftw_execute_dft_r2c(plan, in, out);
  }
  static void execute_dft_c2r(PlanType plan, ComplexType* in, double* out) {
    fftw_execute_dft_c2r(plan, in, out);
  }
};

// Helper to get real and imaginary parts from FFTW complex
template <typename T>
inline T& complex_real(typename FftwTraits<T>::ComplexType& c) { return c[0]; }

template <typename T>
inline T& complex_imag(typename FftwTraits<T>::ComplexType& c) { return c[1]; }

template <typename T>
inline const T& complex_real(const typename FftwTraits<T>::ComplexType& c) { return c[0]; }

template <typename T>
inline const T& complex_imag(const typename FftwTraits<T>::ComplexType& c) { return c[1]; }
