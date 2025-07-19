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

The easiest way to install CMake on MacOS is via [homebrew](https://formulae.brew.sh/formula/cmake)

```bash
brew install cmake
```

### OpenMP

Apple does not ship the dynamic library needed to support OpenMP at runtime, however, the clang compiler 
that ships with xcode supports OpenMP (see [here](https://mac.r-project.org/openmp/) for more detail). 
OpenMP for MacOS can be installed via [homebrew](https://formulae.brew.sh/formula/libomp)

```bash
brew install libomp
```

To detect whether openmp is available at build time, we use [a modified version of the LightGBM `CMakeLists.txt`](https://github.com/microsoft/LightGBM/blob/195c26fc7b00eb0fec252dfe841e2e66d6833954/CMakeLists.txt#L163).

### Eigen

Eigen for MacOS can be installed via [homebrew](https://formulae.brew.sh/formula/eigen). 

```bash
brew install eigen
```

We use a similar trick as we used for openmp to detect eigen headers at build time.

## Building the projects

### CPU

With cmake (and ideally openMP) installed, you can build the CPU-based program matrix multiplication program by running

```
rm -rf build             
mkdir build
cmake -S . -B build -DOPENMP_FLAG=ON -DDEBUG_FLAG=OFF -DUNITTEST_FLAG=OFF -DASAN_FLAG=OFF -DEIGEN_FLAG=ON -DOMP_NTHREADS=k
cmake --build build
```

from the project directory at the command line (replace `k` with the number of threads you intend to use). 
Then you can run the program with `./build/sgemm_silicon_cpu`.

### XCode

To generate an Xcode project based on the build targets and specifications defined in a `CMakeLists.txt`, navigate to the main project folder (i.e. `cd /path/to/sgemm_silicon`) and run the following commands (replacing `k` with the desired number of threads you'd like openMP to use):

```{bash}
rm -rf xcode/            
mkdir xcode
cd xcode                                                                               
cmake -G Xcode .. -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=c++ -DOPENMP_FLAG=ON -DDEBUG_FLAG=OFF -DEIGEN_FLAG=ON -DOMP_NTHREADS=k
cd ..
```

Now, if you navigate to the xcode subfolder (in Finder), you should be able to click on a `.xcodeproj` file and the project will open in Xcode.
