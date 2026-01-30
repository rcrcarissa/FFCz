#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments
BUILD_SZ3=false
BUILD_ZFP=false
BUILD_SPERR=false

for arg in "$@"; do
    case "$arg" in
        sz3|SZ3)     BUILD_SZ3=true ;;
        zfp|ZFP)     BUILD_ZFP=true ;;
        sperr|SPERR) BUILD_SPERR=true ;;
        all|ALL)     BUILD_SZ3=true; BUILD_ZFP=true; BUILD_SPERR=true ;;
        *)           echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Always build FFCz GPU
echo "Building FFCz GPU..."
cd "$SCRIPT_DIR/FFCz/GPU"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
echo "FFCz GPU built successfully!"

# Always build FFCz CPU
echo "Building FFCz CPU..."
cd "$SCRIPT_DIR/FFCz/CPU"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
echo "FFCz CPU built successfully!"

# Conditionally build SZ3
if [ "$BUILD_SZ3" = true ]; then
    echo "Downloading & building SZ3..."
    cd "$SCRIPT_DIR"
    if [ ! -d "SZ3" ]; then
        git clone https://github.com/szcompressor/SZ3.git
    fi
    cd SZ3
    mkdir -p build && cd build
    cmake ..
    make -j16
    echo "SZ3 built successfully!"
fi

# Conditionally build ZFP
if [ "$BUILD_ZFP" = true ]; then
    echo "Downloading & building ZFP..."
    cd "$SCRIPT_DIR"
    if [ ! -d "zfp" ]; then
        git clone https://github.com/LLNL/zfp.git
    fi
    cd zfp
    mkdir -p build && cd build
    cmake ..
    make -j16
    echo "ZFP built successfully!"
fi

# Conditionally build SPERR
if [ "$BUILD_SPERR" = true ]; then
    echo "Downloading & building SPERR..."
    cd "$SCRIPT_DIR"
    if [ ! -d "SPERR" ]; then
        git clone https://github.com/NCAR/SPERR.git
    fi
    cd SPERR
    mkdir -p build && cd build
    cmake ..
    make -j16
    echo "SPERR built successfully!"
fi

echo ""
echo "Build complete!"
echo "  FFCz CPU: $SCRIPT_DIR/CPU/build/ffcz_cpu"
echo "  FFCz GPU: $SCRIPT_DIR/GPU/build/ffcz"
[ "$BUILD_SZ3" = true ] && echo "  SZ3: $SCRIPT_DIR/SZ3/build/tools/sz3/sz3"
[ "$BUILD_ZFP" = true ] && echo "  ZFP: $SCRIPT_DIR/zfp/build/bin/zfp"
[ "$BUILD_SPERR" = true ] && echo "  SPERR: $SCRIPT_DIR/SPERR/build/bin/sperr3d"

# Build PATH export string
PATH_EXPORT="export PATH=\"$SCRIPT_DIR/CPU/build:$SCRIPT_DIR/GPU/build"
[ "$BUILD_SZ3" = true ] && PATH_EXPORT="$PATH_EXPORT:$SCRIPT_DIR/SZ3/build/tools/sz3"
[ "$BUILD_ZFP" = true ] && PATH_EXPORT="$PATH_EXPORT:$SCRIPT_DIR/zfp/build/bin"
[ "$BUILD_SPERR" = true ] && PATH_EXPORT="$PATH_EXPORT:$SCRIPT_DIR/SPERR/build/bin"
PATH_EXPORT="$PATH_EXPORT:\$PATH\""
echo "Executables added to PATH!"