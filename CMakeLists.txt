# Project definition
cmake_minimum_required(VERSION 3.10)
project(sgemm_silicon LANGUAGES C)

# C flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# OpenMP
if(APPLE)
    # libomp 15.0+ from brew is keg-only, so have to search in other locations.
    # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include -L${HOMEBREW_LIBOMP_PREFIX}/lib")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include -L${HOMEBREW_LIBOMP_PREFIX}/lib")
    set(OpenMP_C_LIB_NAMES omp)
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
    find_package(OpenMP REQUIRED)
else()
    find_package(OpenMP REQUIRED)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Library defined in src/ folder
include_directories("${PROJECT_SOURCE_DIR}/src")
add_library(matmul src/matmul.c src/utils.c src/kernel.c)
if((NOT DEFINED NTHREADS) OR (NTHREADS STREQUAL ""))
  message(FATAL_ERROR "Error, please specify the number of threads!")
endif()
if((NOT DEFINED OMP_SCHEDULE) OR (OMP_SCHEDULE STREQUAL ""))
  set(OMP_SCHEDULE "auto")
  message(WARNING "OMP_SCHEDULE was not specified, set to \"auto\"")
endif()
target_compile_definitions(matmul PRIVATE OMP_SCHEDULE=${OMP_SCHEDULE} NTHREADS=${NTHREADS})

# Main executable
add_executable(sgemm_silicon_cpu test.c)
list(APPEND MATMUL_LIBS -lm -lpthread ${OpenMP_C_LIB_NAMES})
target_link_libraries(sgemm_silicon_cpu PUBLIC matmul ${MATMUL_LIBS})