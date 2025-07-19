# Project definition
cmake_minimum_required(VERSION 3.20)
project(sgemm_silicon LANGUAGES C CXX)

# Build options
option(OPENMP_FLAG "Use openMP" OFF)
option(DEBUG_FLAG "Compile with debug symbols and no optimization" OFF)
option(ASAN_FLAG "Address sanitizer" OFF)
option(UNITTEST_FLAG "C++ unit tests" OFF)
option(EIGEN_FLAG "C++ unit tests" OFF)

# Optional address sanitizer compiler flags
if(ASAN_FLAG)
    add_definitions(-DASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

# Debug symbols and optimization level in compiler flags
if(DEBUG_FLAG)
    add_definitions(-DDEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -march=native -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -march=native -Wall")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")
endif()

# Compile commands for IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Project source headers
set(SOURCE_HEADER_DIR ${PROJECT_SOURCE_DIR}/src)

# Eigen headers
if(EIGEN_FLAG)
  add_definitions(-DEIGEN_ON)
  # add_definitions(-DEIGEN_USE_BLAS)
  execute_process(COMMAND brew --prefix eigen
                  OUTPUT_VARIABLE HOMEBREW_EIGEN_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(EIGEN_HEADER_DIR ${HOMEBREW_EIGEN_PREFIX}/include/eigen3)
endif()

# OpenMP
if(OPENMP_FLAG)
  add_definitions(-DOMP_ON)
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include -L${HOMEBREW_LIBOMP_PREFIX}/lib")
  set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include -L${HOMEBREW_LIBOMP_PREFIX}/lib")
  set(OpenMP_LINK_FLAG "-L${HOMEBREW_LIBOMP_PREFIX}/lib")
  set(OpenMP_C_LIB_NAMES omp)
  set(OpenMP_CXX_LIB_NAMES omp)
  set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
  
  # Update flags with openmp 
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  list(APPEND MATMUL_LIBS -lm -lpthread ${OpenMP_C_LIB_NAMES})
  list(APPEND TEST_LIBS -lm -lpthread ${OpenMP_C_LIB_NAMES})
else()
  list(APPEND MATMUL_LIBS -lm)
  list(APPEND TEST_LIBS -lm)
endif()

# Accelerate framework
find_library( FRAMEWORK_ACCELERATE
  NAMES ACCELERATE
  PATHS ${CMAKE_OSX_SYSROOT}/System/Library
  PATH_SUFFIXES Frameworks 
  CMAKE_FIND_FRAMEWORK only
  NO_DEFAULT_PATH
)

# C library defined in src/ folder
include_directories("${SOURCE_HEADER_DIR}")
add_library(matmul src/matmul.c src/utils.c src/kernel.c)
if(OPENMP_FLAG)
  if((NOT DEFINED OMP_NTHREADS) OR (OMP_NTHREADS STREQUAL ""))
    set(OMP_NTHREADS "1")
    message(WARNING "OMP_NTHREADS was not specified, set to \"1\"")
  endif()
  if((NOT DEFINED OMP_SCHEDULE) OR (OMP_SCHEDULE STREQUAL ""))
    set(OMP_SCHEDULE "auto")
    message(WARNING "OMP_SCHEDULE was not specified, set to \"auto\"")
  endif()
  target_compile_definitions(matmul PRIVATE OMP_SCHEDULE=${OMP_SCHEDULE} OMP_NTHREADS=${OMP_NTHREADS})
endif()
if(EIGEN_FLAG)
  target_include_directories(matmul PRIVATE ${EIGEN_HEADER_DIR})
endif()
set_target_properties(matmul PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(matmul PUBLIC ${FRAMEWORK_ACCELERATE})
if(OPENMP_FLAG)
  target_link_options(matmul PRIVATE ${OpenMP_LINK_FLAG})
endif()

# C++ library that uses Eigen
if(EIGEN_FLAG)
  add_library(matmul_eigen STATIC src/eigen_extension.cpp)
  target_include_directories(matmul_eigen PRIVATE ${EIGEN_HEADER_DIR} ${SOURCE_HEADER_DIR})
  set_target_properties(matmul_eigen PROPERTIES LINKER_LANGUAGE CXX)
endif()

# Main executable
add_executable(sgemm_silicon_cpu test.c)
if(EIGEN_FLAG)
  target_link_libraries(sgemm_silicon_cpu PUBLIC matmul matmul_eigen ${MATMUL_LIBS} ${FRAMEWORK_ACCELERATE})
else()
  target_link_libraries(sgemm_silicon_cpu PUBLIC matmul ${MATMUL_LIBS} ${FRAMEWORK_ACCELERATE})
endif()
set_target_properties(sgemm_silicon_cpu PROPERTIES LINKER_LANGUAGE C)
if(OPENMP_FLAG)
  target_link_options(sgemm_silicon_cpu PRIVATE ${OpenMP_LINK_FLAG})
endif()

# Optional C++ unit test
if(UNITTEST_FLAG)
  # Check if user specified a local clone of the GoogleTest repo, use Github repo if not
  if (NOT DEFINED GOOGLETEST_GIT_REPO)
    set(GOOGLETEST_GIT_REPO https://github.com/google/googletest.git)
  endif()
  
  # Fetch and install GoogleTest dependency
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY ${GOOGLETEST_GIT_REPO}
    GIT_TAG        6910c9d9165801d8827d628cb72eb7ea9dd538c5 # release-1.16.0
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Build test suite
  enable_testing()
  include(GoogleTest)
  file(GLOB CPP_TEST_SOURCES test/*.cpp)
  list(APPEND TEST_LIBS -lm -lpthread ${OpenMP_CXX_LIB_NAMES})
  add_executable(testprogram ${CPP_TEST_SOURCES})
  target_include_directories(testprogram PRIVATE ${SOURCE_HEADER_DIR})
  target_link_libraries(testprogram PRIVATE matmul GTest::gtest_main ${TEST_LIBS} ${FRAMEWORK_ACCELERATE})
  if(OPENMP_FLAG)
    target_link_options(testprogram PRIVATE ${OpenMP_LINK_FLAG})
  endif()
  set_target_properties(testprogram PROPERTIES LINKER_LANGUAGE CXX)
  gtest_discover_tests(testprogram)
endif()