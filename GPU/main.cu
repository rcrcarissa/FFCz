#include "decompression.cuh"
#include "fileIO.h"
#include "projection_algorithm.cuh"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cub/cub.cuh>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <vector>

std::string originalFile = "";
std::string baseDecompFile;
std::string compressedFile;
std::string decompressedFile;
size_t Nx = 1;
size_t Ny = 1;
size_t Nz = 1;
size_t N = 1;
bool isDouble = false;
bool isSpatialABS = false;
bool isFreqABS = false;
bool isFreqPTW = false;
float spat_epsilon; // spatial error bound
float freq_delta;   // frequency error bound

void parseError(const char error[]) {
  std::cout << error << std::endl;
  std::cout << "Usage:\n";
  std::cout << "  -i <file_path>  : Specify the file of original data\n";
  std::cout << "  -e <file_path>  : Specify the reconstructed file of base "
               "compressor\n";
  std::cout << "  -z <file_path>  : Specify the compressed file\n";
  std::cout << "  -o <file_path>  : Specify the decompressed file (optional)\n";
  std::cout << "  -1 <nx>                : 1D data with <nx> values\n";
  std::cout << "  -2 <nx> <ny>           : 2D data with <nx> * <ny> values\n";
  std::cout
      << "  -3 <nx> <ny> <nz>      : 3D data with <nx> * <ny> * <nz> values\n";
  std::cout << "  -f              : Use float data type\n";
  std::cout << "  -d              : Use double data type\n";
  std::cout << "  -M ABS <xi>     : Specify the absolute error bound in "
               "spatial domain\n";
  std::cout << "  -M REL <xi>     : Specify the relative error bound in "
               "spatial domain\n";
  std::cout << "  -F ABS <xi>     : Specify the absolute error bound in "
               "frequency domain\n";
  std::cout << "  -F REL <xi>     : Specify the relative error bound in "
               "frequency domain\n";
  std::cout
      << "  -F PTW <xi>     : Specify the pointwise relative error bound in "
         "frequency domain\n";
  exit(EXIT_FAILURE);
}

void Parsing(int argc, char *argv[]) {
  bool originalFileSpecified = false;
  bool baseDecompFileSpecified = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-i") {
      if (i + 1 >= argc)
        parseError("Missing input file path");
      originalFile = argv[++i];
      originalFileSpecified = true;
    } else if (arg == "-e") {
      if (i + 1 >= argc)
        parseError("Missing base reconstructed file path");
      baseDecompFile = argv[++i];
      baseDecompFileSpecified = true;
    } else if (arg == "-z") {
      if (i + 1 >= argc)
        parseError("Missing compressed file path");
      compressedFile = argv[++i];
    } else if (arg == "-o") {
      if (i + 1 >= argc)
        parseError("Missing decompressed file path");
      decompressedFile = argv[++i];
    } else if (arg == "-1" || arg == "-2" || arg == "-3") {
      if (arg == "-1") {
        Nz = std::stoi(argv[++i]);
        N = Nz;
      } else if (arg == "-2") {
        Ny = std::stoi(argv[++i]);
        Nz = std::stoi(argv[++i]);
        N = Ny * Nz;
      } else if (arg == "-3") {
        Nx = std::stoi(argv[++i]);
        Ny = std::stoi(argv[++i]);
        Nz = std::stoi(argv[++i]);
        N = Nx * Ny * Nz;
      }

      if (i + 1 >= argc) {
        parseError("Missing size for 1D data");
      }
    } else if (arg == "-f") {
      isDouble = false; // Use float type
    } else if (arg == "-d") {
      isDouble = true; // Use double type
    } else if (arg == "-M") {
      isSpatialABS = std::strcmp(argv[++i], "ABS") == 0;
      if (i + 1 >= argc)
        parseError("Missing spatial error bound");
      spat_epsilon = std::stof(argv[++i]);
    } else if (arg == "-F") {
      char *bound_type = argv[++i];
      isFreqABS = std::strcmp(bound_type, "ABS") == 0;
      isFreqPTW = std::strcmp(bound_type, "PTW") == 0;
      if (i + 1 >= argc)
        parseError("Missing frequency error bound");
      freq_delta = std::stof(argv[++i]);
    } else {
      parseError("Unknown argument");
    }
  }

  if (!originalFileSpecified && !baseDecompFileSpecified) {
    parseError("Input files of original data (-i) and base decompressed data "
               "(-e) are mandatory");
  }
}

template <typename T>
__global__ void normalize_add_two_kernel(const T *arr1, const T *arr2, T *arr3,
                                         T factor, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    arr3[idx] *= factor;
    arr3[idx] += arr1[idx] + arr2[idx];
  }
}

template <typename T> void Run() {
  size_t F = Nx * Ny * (Nz / 2 + 1);
  T *h_base_data = new T[N];
  if (isDouble) {
    readRawArrayBinary(baseDecompFile, h_base_data, N, DataType::DOUBLE);
  } else {
    readRawArrayBinary(baseDecompFile, h_base_data, N, DataType::FLOAT);
  }

  if (originalFile == "") {
    // Decompression mode
    dim3 block(BLOCK_SIZE);
    dim3 spat_grid((N + block.x - 1) / block.x);
    dim3 freq_grid((F + block.x - 1) / block.x);

    auto freq_edits_compressed =
        readCompressedData<UInt>(compressedFile + ".fedits");
    auto freq_flags_compressed =
        readCompressedData<uint8_t>(compressedFile + ".fflags");
    auto spat_edits_compressed =
        readCompressedData<UInt>(compressedFile + ".sedits");
    auto spat_flags_compressed =
        readCompressedData<uint8_t>(compressedFile + ".sflags");
    T extreme[4];
    cufftHandle fft_plan_forward;
    cufftHandle fft_plan_inverse;
    if constexpr (std::is_same_v<T, float>) {
      readRawArrayBinary(compressedFile + ".extreme", extreme, 4,
                         DataType::FLOAT);
      CHECK_CUFFT(cufftPlan3d(&fft_plan_forward, Nx, Ny, Nz, CUFFT_R2C));
      CHECK_CUFFT(cufftPlan3d(&fft_plan_inverse, Nx, Ny, Nz, CUFFT_C2R));
    } else {
      readRawArrayBinary(compressedFile + ".extreme", extreme, 4,
                         DataType::DOUBLE);
      CHECK_CUFFT(cufftPlan3d(&fft_plan_forward, Nx, Ny, Nz, CUFFT_D2Z));
      CHECK_CUFFT(cufftPlan3d(&fft_plan_inverse, Nx, Ny, Nz, CUFFT_Z2D));
    }

    T *d_base_data, *d_spat_edit, *d_decompressed_data;
    CHECK_CUDA(cudaMalloc(&d_base_data, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_spat_edit, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_decompressed_data, N * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_base_data, h_base_data, N * sizeof(T),
                          cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();

    if (freq_edits_compressed.num_elements > 0) {
      UInt *d_freq_edit_compact_quantized;
      uint8_t *d_freq_flag_pack;
      T *d_freq_edit;
      typename CufftTraits<T>::ComplexType *d_freq_edit_complex;
      CHECK_CUDA(cudaMalloc(&d_freq_edit_compact_quantized,
                            freq_edits_compressed.num_elements * sizeof(UInt)));
      CHECK_CUDA(
          cudaMalloc(&d_freq_flag_pack, freq_flags_compressed.num_elements));
      CHECK_CUDA(cudaMalloc(&d_freq_edit, 2 * F * sizeof(T)));
      CHECK_CUDA(cudaMalloc(&d_freq_edit_complex,
                            F * sizeof(typename CufftTraits<T>::ComplexType)));

      decompressHuffmanZstd<UInt>(freq_edits_compressed,
                                  d_freq_edit_compact_quantized);
      decompressHuffmanZstd<uint8_t>(freq_flags_compressed, d_freq_flag_pack);

      reconstructEdits<T>(d_freq_edit_compact_quantized, d_freq_flag_pack,
                          d_freq_edit, extreme[0], extreme[1],
                          freq_edits_compressed.num_elements, 2 * F);

      real_imag_to_complex<T>
          <<<freq_grid, block>>>(d_freq_edit, d_freq_edit_complex, F);

      if constexpr (std::is_same_v<T, float>) {
        CHECK_CUFFT(cufftExecC2R(fft_plan_inverse,
                                 (cufftComplex *)d_freq_edit_complex,
                                 d_decompressed_data));
      } else {
        CHECK_CUFFT(cufftExecZ2D(fft_plan_inverse,
                                 (cufftDoubleComplex *)d_freq_edit_complex,
                                 d_decompressed_data));
      }

      CHECK_CUDA(cudaFree(d_freq_edit_compact_quantized));
      CHECK_CUDA(cudaFree(d_freq_flag_pack));
      CHECK_CUDA(cudaFree(d_freq_edit));
      CHECK_CUDA(cudaFree(d_freq_edit_complex));
    } else {
      CHECK_CUDA(cudaMemset(d_decompressed_data, 0, N * sizeof(T)));
    }
    if (spat_edits_compressed.num_elements > 0) {
      UInt *d_spat_edit_compact_quantized;
      uint8_t *d_spat_flag_pack;
      CHECK_CUDA(cudaMalloc(&d_spat_edit_compact_quantized,
                            spat_edits_compressed.num_elements * sizeof(UInt)));
      CHECK_CUDA(
          cudaMalloc(&d_spat_flag_pack, spat_flags_compressed.num_elements));

      decompressHuffmanZstd<UInt>(spat_edits_compressed,
                                  d_spat_edit_compact_quantized);
      decompressHuffmanZstd<uint8_t>(spat_flags_compressed, d_spat_flag_pack);

      reconstructEdits<T>(d_spat_edit_compact_quantized, d_spat_flag_pack,
                          d_spat_edit, extreme[2], extreme[3],
                          spat_edits_compressed.num_elements, N);

      CHECK_CUDA(cudaFree(d_spat_edit_compact_quantized));
      CHECK_CUDA(cudaFree(d_spat_flag_pack));
    } else {
      CHECK_CUDA(cudaMemset(d_spat_edit, 0, N * sizeof(T)));
    }

    T norm_factor = T(1) / static_cast<T>(N);
    normalize_add_two_kernel<T><<<spat_grid, block>>>(
        d_base_data, d_spat_edit, d_decompressed_data, norm_factor, N);
    CHECK_CUDA(cudaGetLastError());

    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Decompression time: " << time.count() << " ms" << std::endl;

    T *h_decompressed_data = new T[N];
    CHECK_CUDA(cudaMemcpy(h_decompressed_data, d_decompressed_data,
                          N * sizeof(T), cudaMemcpyDeviceToHost));
    writeRawArrayBinary(h_decompressed_data, N, decompressedFile);

    CHECK_CUDA(cudaFree(d_base_data));
    CHECK_CUDA(cudaFree(d_spat_edit));
    CHECK_CUDA(cudaFree(d_decompressed_data));
  } else {
    // Compression mode
    T *h_orig_data = new T[N];
    if (isDouble) {
      readRawArrayBinary(originalFile, h_orig_data, N, DataType::DOUBLE);
    } else {
      readRawArrayBinary(originalFile, h_orig_data, N, DataType::FLOAT);
    }
    const size_t max_iterations = 40;
    const T tolerance = 1e-6f;

    auto start = std::chrono::high_resolution_clock::now();

    ProjectionSolver<T> solver(Nx, Ny, Nz);
    solver.initialize(h_orig_data, h_base_data, spat_epsilon, freq_delta,
                      max_iterations, tolerance, isSpatialABS, isFreqABS,
                      isFreqPTW);
    if (isFreqPTW) {
      solver.solve_ptw();
    } else {
      solver.solve_abs();
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Compression time: " << time.count() << " ms" << std::endl;
    std::cout << "Number of iterations: " << solver.get_iteration_count()
              << std::endl;
    auto freq_edits_compressed = solver.get_freq_edits_compressed();
    auto freq_flags_compressed = solver.get_freq_flags_compressed();
    auto spat_edits_compressed = solver.get_spat_edits_compressed();
    auto spat_flags_compressed = solver.get_spat_flags_compressed();
    size_t extra_storage = freq_edits_compressed.compressed_size +
                           freq_flags_compressed.compressed_size +
                           spat_edits_compressed.compressed_size +
                           spat_flags_compressed.compressed_size;
    size_t num_active_freq = solver.get_num_active_freq();
    size_t num_active_spat = solver.get_num_active_spat();
    if (num_active_freq)
      extra_storage += 2 * sizeof(T);
    if (num_active_spat)
      extra_storage += 2 * sizeof(T);
    std::cout << "Number of active frequency edits: " << num_active_freq
              << std::endl;
    std::cout << "Number of active spatial edits: " << num_active_spat
              << std::endl;
    std::cout << "Additional storage: " << extra_storage << " bytes"
              << std::endl;
    writeCompressedData<UInt>(freq_edits_compressed,
                              compressedFile + ".fedits");
    writeCompressedData<uint8_t>(freq_flags_compressed,
                                 compressedFile + ".fflags");
    writeCompressedData<UInt>(spat_edits_compressed,
                              compressedFile + ".sedits");
    writeCompressedData<uint8_t>(spat_flags_compressed,
                                 compressedFile + ".sflags");
    T edit_extreme[4] = {solver.get_min_freq_edit(), solver.get_max_freq_edit(),
                         solver.get_min_spat_edit(),
                         solver.get_max_spat_edit()};
    writeRawArrayBinary(edit_extreme, 4, compressedFile + ".extreme");

    // Compute statistics
    solver.calculate_statistics(h_base_data);
    // T spat_max = h_orig_data[0], spat_min = h_orig_data[0];
    // for (size_t i = 0; i < N; ++i) {
    //   h_spat_err[i] += h_base_data[i] - h_orig_data[i];
    //   T abs_err = abs(h_spat_err[i]);
    //   if (abs_err > mae)
    //     mae = abs_err;
    //   mse += abs_err * abs_err;

    //   T val = h_orig_data[i];
    //   if (val > spat_max)
    //     spat_max = val;
    //   else if (val < spat_min)
    //     spat_min = val;
    // }
  }
}

int main(int argc, char *argv[]) {
  Parsing(argc, argv);

  if (isDouble) {
    Run<double>();
  } else {
    Run<float>();
  }
}
