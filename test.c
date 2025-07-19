#include "src/utils.h"
#include "src/matmul.h"
#ifdef EIGEN_ON
#include "src/eigen_extension.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mm_malloc.h>

#define MEM_ALIGN 64

#ifndef MDIM
    #define MDIM 1000
#endif

#ifndef NDIM
    #define NDIM 1000
#endif

#ifndef KDIM
    #define KDIM 2000
#endif

#ifndef NITER
    #define NITER 10
#endif

int main() {
    const int M = MDIM;
    const int N = NDIM;
    const int K = KDIM;
    float* A = (float*)_mm_malloc(M * K * sizeof(float), MEM_ALIGN);
    float* B = (float*)_mm_malloc(K * N * sizeof(float), MEM_ALIGN);
    float* C = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_accel = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_eigen = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_ref = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    init_rand(A, M * K);
    init_rand(B, K * N);
  
//    for (int i = 0; i < MDIM; i++) {
//      for (int j = 0; j < KDIM; j++) {
//        printf("A[%d,%d] = %.3f\n", i, j, *(A + j * M + i));
//      }
//    }
//  
//    for (int i = 0; i < KDIM; i++) {
//      for (int j = 0; j < NDIM; j++) {
//        printf("B[%d,%d] = %.3f\n", i, j, *(B + j * K + i));
//      }
//    }

    double FLOP = 2 * (double)M * N * K;

    uint64_t start = timer();
    matmul_naive(A, B, C_ref, M, N, K);
    uint64_t end = timer();

    double exec_time = (end - start) * 1e-9;
    double FLOPS = FLOP / exec_time;

    printf("Exec. time = %.3fms\n", exec_time * 1000);
    printf("GFLOPS = %.3f\n", FLOPS / 1e9);
    
    for (int i = 0; i < NITER; i++) {
        init_const(C, 0.0, M*N);
        start = timer();
        matmul_blocked_kernel(A, B, C, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        FLOPS = FLOP / exec_time;

        printf("Exec. time (custom) = %.3fms\n", exec_time * 1000);
        printf("GFLOPS (custom) = %.3f\n", FLOPS / 1e9);

        struct val_stat_t val_results = validate_mat(C, C_ref, M * N, 1e-4);
        printf("Number of mismatches (custom) = %d\n", val_results.n_error);

        #ifdef EIGEN_ON
        
        start = timer();
        matmul_eigen(A, B, C_eigen, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        FLOPS = FLOP / exec_time;

        printf("Exec. time (eigen) = %.3fms\n", exec_time * 1000);
        printf("GFLOPS (eigen) = %.3f\n", FLOPS / 1e9);

        val_results = validate_mat(C_eigen, C_ref, M * N, 1e-4);
        printf("Number of mismatches (eigen) = %d\n", val_results.n_error);

        #endif

        start = timer();
        matmul_accelerate(A, B, C_accel, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        FLOPS = FLOP / exec_time;

        printf("Exec. time (accelerate) = %.3fms\n", exec_time * 1000);
        printf("GFLOPS (accelerate) = %.3f\n", FLOPS / 1e9);

        val_results = validate_mat(C_accel, C_ref, M * N, 1e-4);
        printf("Number of mismatches (accelerate) = %d\n", val_results.n_error);

//        if (i == NITER - 1) {
//            for (int j = 0; j < 5; j++) {
//                for (int k = 0; k < 5; k++) {
//                    printf("C[%d,%d] = %.3f\n", j, k, *(C + k * M + j));
//                    printf("C_ref[%d,%d] = %.3f\n", j, k, *(C_ref + k * M + j));
//                }
//            }
//        }
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0;
}
