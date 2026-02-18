# FFCz

FFCz is a correction-based method for lossy compression of regular-grid data that preserves accuracy in both **spatial** and **frequency** domains. It augments decompressed results from existing compressors by applying a GPU/CPU-parallel correction stage to enforce **dual-domain error guarantees**.

## Installation

### Requirements
- CMake (tested with 3.18+)
- CUDA (for GPU build and runs; tested with 12.4+)
- FFTW (for CPU build and runs; tested with 3.3.10+)

### Compile

```bash
git clone https://github.com/rcrcarissa/FFCz.git
cd FFCz

# Build FFCz GPU only
cd GPU
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
make -j8
cd ../../

# Build FFCz CPU only
cd CPU
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
make -j8
cd ../../

# Build both GPU and CPU versions of FFCz
sh build.sh
```

## Usage
### Correction (editing) stage
```bash
ffcz (-f | -d) -i <original_path> -e <base_path> -z <compressed_path> \
     -M REL <epsilon> -F REL <delta> (-1 N | -2 Nx Ny | -3 Nx Ny Nz)
```
### Decompression stage
```bash
ffcz (-f | -d) -e <base_path> -z <compressed_path> -o <decompressed_path> \
     -M REL <epsilon> -F REL <delta> (-1 N | -2 Nx Ny | -3 Nx Ny Nz)
```
### Example
To edit decompressed data `decomp.dat` in 3D with
512×512×512 single-precision floating-point values, using the original data
`original.dat`, and write the corrected compressed output to `comp.dat`:
```bash
ffcz -f -i original.dat -e decomp.dat -z comp.dat -M REL 1e-3 -F REL 1e-3 -3 512 512 512
```
### Options

**Input / output**
- `-i <path>`: path to original data
- `-e <path>`: path to base decompressed data
- `-z <path>`: path to output compressed file
- `-o <path>`: path to output decompressed file

**Spatial error bounds**
- `-M ABS <epsilon>`: spatial absolute error bound
- `-M REL <epsilon>`: spatial relative error bound

**Frequency error bounds**
- `-F ABS <delta>`: frequency absolute error bound
- `-F REL <delta>`: frequency relative error bound
- `-F PTW <delta>`: frequency pointwise error bound

**Data dimensions**
- `-1 Nx`: 1D data dimensions
- `-2 Nx Ny`: 2D data dimensions
- `-3 Nx Ny Nz`: 3D data dimensions

### Notes
- To run the CPU version of FFCz, replace `ffcz` with `ffcz_cpu`.

- For a full list of options, run `ffcz` without any arguments.

## Citation
If you use FFCz, please cite:
```
@article{ren2026ffcz,
  title={FFCz: Fast Fourier Correction for Spectrum-Preserving Lossy Compression of Scientific Data},
  author={Ren, Congrong and Underwood, Robert and Di, Sheng and Kutay, Emrecan and Lukic, Zarija and Yener, Aylin and Cappello, Franck and Guo, Hanqi},
  journal={arXiv preprint arXiv:2601.01596},
  year={2026}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
Reach out to <ren.452@osu.edu> if you have any questions.
