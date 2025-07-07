# SGEMM on Apple Silicon

This project is largely educational -- I want to learn how to write fast matrix multiplication operations 
for the Apple Silicon CPU and GPU.

To begin, I will follow Aman Salykov's excellent tutorials on accelerating matrix multiplication routines 
on [x86 CPUs](https://salykova.github.io/matmul-cpu) and [NVIDIA GPUs](https://salykova.github.io/sgemm-gpu), 
largely just translating to the SIMD and GPGPU interfaces available on Apple Silicon. 

## Hardware requirements

These programs run on Apple Silicon devices (M1 / M2 / M3 macbook pro / macbook air / mac studio / mac pro).

## Software requirements

### CMake

The easiest way to install CMake on MacOS is via [homebrew](https://formulae.brew.sh/formula/cmake). 

### OpenMP

Apple does not ship the dynamic library needed to support OpenMP at runtime, however, the clang compiler 
that ships with xcode supports OpenMP (see [here](https://mac.r-project.org/openmp/) for more detail). 
OpenMP for MacOS can also be installed via [homebrew](https://formulae.brew.sh/formula/libomp)

To detect whether openmp is available at build time, we use [a modified version of the LightGBM `CMakeLists.txt`](https://github.com/microsoft/LightGBM/blob/195c26fc7b00eb0fec252dfe841e2e66d6833954/CMakeLists.txt#L163).

## Building the projects

### CPU

With cmake (and ideally openMP) installed, you can build the CPU-based program matrix multiplication program by running

```
rm -rf build
mkdir build
cmake -S . -B build
cmake --build build
```

from the project directory at the command line. Then you can run the program with `./build/sgemm_silicon_cpu`.
