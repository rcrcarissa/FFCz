#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building FFCz GPU..."
cd "$SCRIPT_DIR/FFCz/GPU"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
echo "FFCz GPU built successfully!"

echo "Building FFCz CPU..."
cd "$SCRIPT_DIR/FFCz/CPU"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
echo "FFCz CPU built successfully!"

echo ""
echo "Build complete!"
echo "  FFCz GPU: $SCRIPT_DIR/GPU/build/ffcz"
echo "  FFCz CPU: $SCRIPT_DIR/CPU/build

PATH_EXPORT="export PATH=\"$SCRIPT_DIR/GPU/build:$SCRIPT_DIR/CPU/build"
PATH_EXPORT="$PATH_EXPORT:\$PATH\""
echo "Executables added to PATH!"
