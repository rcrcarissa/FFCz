#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cufft.h>

// Constants
inline constexpr size_t BLOCK_SIZE = 256;
inline constexpr size_t m = 16;

// Type for quantization codes
template <size_t M>
using UIntForSize = std::conditional_t<
    M == 8, uint8_t,
    std::conditional_t<M == 16, uint16_t,
                       std::conditional_t<M == 32, uint32_t, void>>>;

using UInt = UIntForSize<m>;

// Error checking macros
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
#endif

#ifndef CHECK_CUFFT
#define CHECK_CUFFT(call)                                                      \
  do {                                                                         \
    cufftResult error = (call);                                                \
    if (error != CUFFT_SUCCESS) {                                              \
      std::cerr << "CUFFT error at " << __FILE__ << ":" << __LINE__ << " - "   \
                << static_cast<int>(error) << std::endl;                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
#endif

// CUFFT traits for cuFFT types
template <typename T> struct CufftTraits;

template <> struct CufftTraits<float> {
  using ComplexType = cufftComplex;
  static constexpr cufftType R2C = CUFFT_R2C;
  static constexpr cufftType C2R = CUFFT_C2R;
};

template <> struct CufftTraits<double> {
  using ComplexType = cufftDoubleComplex;
  static constexpr cufftType R2C = CUFFT_D2Z;
  static constexpr cufftType C2R = CUFFT_Z2D;
};

// Create a cuFFT plan that adapts to the actual dimensionality (1D/2D/3D).
// Nx, Ny, Nz follow the convention: 1D uses Nz, 2D uses Ny*Nz, 3D uses all.
inline void createCufftPlan(cufftHandle *plan, size_t Nx, size_t Ny, size_t Nz,
                            cufftType type) {
  int rank;
  int dims[3];
  if (Nx > 1) {
    rank = 3;
    dims[0] = static_cast<int>(Nx);
    dims[1] = static_cast<int>(Ny);
    dims[2] = static_cast<int>(Nz);
  } else if (Ny > 1) {
    rank = 2;
    dims[0] = static_cast<int>(Ny);
    dims[1] = static_cast<int>(Nz);
  } else {
    rank = 1;
    dims[0] = static_cast<int>(Nz);
  }
  CHECK_CUFFT(cufftPlanMany(plan, rank, dims, NULL, 1, 0, NULL, 1, 0, type, 1));
}
