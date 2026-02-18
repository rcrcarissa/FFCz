cmake_minimum_required(VERSION 3.18)
project(FFCz_CPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Find required packages
find_package(PkgConfig REQUIRED)

# Find FFTW3 & ZSTD
pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(ZSTD REQUIRED libzstd)

# If pkg-config doesn't find them, try find_library
if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIBRARIES NAMES fftw3 REQUIRED)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h REQUIRED)
endif()

if(NOT FFTW3F_FOUND)
    find_library(FFTW3F_LIBRARIES NAMES fftw3f REQUIRED)
endif()

if(NOT ZSTD_FOUND)
    find_library(ZSTD_LIBRARIES NAMES zstd REQUIRED)
    find_path(ZSTD_INCLUDE_DIRS NAMES zstd.h REQUIRED)
endif()

# Utility library
add_library(util STATIC
    fileIO.cpp
)

target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(ffcz_cpu
    main.cpp
    decompression.cpp
    projection_algorithm.cpp
)

target_include_directories(ffcz_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FFTW3_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIRS}
)

target_link_directories(ffcz_cpu PRIVATE
    ${FFTW3_LIBRARY_DIRS}
    ${FFTW3F_LIBRARY_DIRS}
    ${ZSTD_LIBRARY_DIRS}
)

target_link_libraries(ffcz_cpu PRIVATE
    util
    ${FFTW3_LIBRARIES}
    ${FFTW3F_LIBRARIES}
    ${ZSTD_LIBRARIES}
)

# Install target
install(TARGETS ffcz_cpu DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "FFCz CPU Configuration Summary")
message(STATUS "==============================")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "FFTW3 Libraries: ${FFTW3_LIBRARIES} ${FFTW3F_LIBRARIES}")
message(STATUS "ZSTD Libraries: ${ZSTD_LIBRARIES}")
message(STATUS "")
