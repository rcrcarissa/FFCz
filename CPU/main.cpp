#include "decompression.h"
#include "fileIO.h"
#include "projection_algorithm.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
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
float spat_epsilon;
float freq_delta;

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
      isDouble = false;
    } else if (arg == "-d") {
      isDouble = true;
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
    auto freq_edits_compressed =
        readCompressedData<UInt>(compressedFile + ".fedits");
    auto freq_flags_compressed =
        readCompressedData<uint8_t>(compressedFile + ".fflags");
    auto spat_edits_compressed =
        readCompressedData<UInt>(compressedFile + ".sedits");
    auto spat_flags_compressed =
        readCompressedData<uint8_t>(compressedFile + ".sflags");
    T extreme[4];
    if constexpr (std::is_same_v<T, float>) {
      readRawArrayBinary(compressedFile + ".extreme", extreme, 4,
                         DataType::FLOAT);
    } else {
      readRawArrayBinary(compressedFile + ".extreme", extreme, 4,
                         DataType::DOUBLE);
    }

    T *tmp_real = new T[N];
    auto *tmp_complex = reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
        fftw_malloc(F * sizeof(typename FftwTraits<T>::ComplexType)));

    auto fft_plan_inverse = FftwTraits<T>::plan_dft_c2r_3d(
        Nx, Ny, Nz, tmp_complex, tmp_real, FFTW_ESTIMATE);

    delete[] tmp_real;
    fftw_free(tmp_complex);

    std::vector<T> spat_edit(N, T(0));
    std::vector<T> decompressed_data(N, T(0));

    auto start = std::chrono::high_resolution_clock::now();

    if (freq_edits_compressed.num_elements > 0) {
      std::vector<UInt> freq_edit_compact_quantized(
          freq_edits_compressed.num_elements);
      std::vector<uint8_t> freq_flag_pack(freq_flags_compressed.num_elements);
      std::vector<T> freq_edit(2 * F);
      auto *freq_edit_complex =
          reinterpret_cast<typename FftwTraits<T>::ComplexType *>(
              fftw_malloc(F * sizeof(typename FftwTraits<T>::ComplexType)));

      decompressHuffmanZstd<UInt>(freq_edits_compressed,
                                  freq_edit_compact_quantized.data());
      decompressHuffmanZstd<uint8_t>(freq_flags_compressed,
                                     freq_flag_pack.data());

      reconstructEdits<T>(freq_edit_compact_quantized.data(),
                          freq_flag_pack.data(), freq_edit.data(), extreme[0],
                          extreme[1], freq_edits_compressed.num_elements,
                          2 * F);

      real_imag_to_complex<T>(freq_edit.data(), freq_edit_complex, F);

      FftwTraits<T>::execute_dft_c2r(fft_plan_inverse, freq_edit_complex,
                                     decompressed_data.data());

      fftw_free(freq_edit_complex);
    }

    if (spat_edits_compressed.num_elements > 0) {
      std::vector<UInt> spat_edit_compact_quantized(
          spat_edits_compressed.num_elements);
      std::vector<uint8_t> spat_flag_pack(spat_flags_compressed.num_elements);

      decompressHuffmanZstd<UInt>(spat_edits_compressed,
                                  spat_edit_compact_quantized.data());
      decompressHuffmanZstd<uint8_t>(spat_flags_compressed,
                                     spat_flag_pack.data());

      reconstructEdits<T>(spat_edit_compact_quantized.data(),
                          spat_flag_pack.data(), spat_edit.data(), extreme[2],
                          extreme[3], spat_edits_compressed.num_elements, N);
    }

    T norm_factor = T(1) / static_cast<T>(N);
    for (size_t i = 0; i < N; ++i) {
      decompressed_data[i] *= norm_factor;
      decompressed_data[i] += h_base_data[i] + spat_edit[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Decompression time: " << time.count() << " ms" << std::endl;

    writeRawArrayBinary(decompressed_data.data(), N, decompressedFile);

    FftwTraits<T>::destroy_plan(fft_plan_inverse);

  } else {
    // Compression mode
    T *h_orig_data = new T[N];
    if (isDouble) {
      readRawArrayBinary(originalFile, h_orig_data, N, DataType::DOUBLE);
    } else {
      readRawArrayBinary(originalFile, h_orig_data, N, DataType::FLOAT);
    }
    const size_t max_iterations = 100;
    const T tolerance = 1e-9f;

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

    delete[] h_orig_data;
  }

  delete[] h_base_data;
}

int main(int argc, char *argv[]) {
  Parsing(argc, argv);

  if (isDouble) {
    Run<double>();
  } else {
    Run<float>();
  }

  return 0;
}
