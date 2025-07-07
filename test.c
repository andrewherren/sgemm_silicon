#include "src/utils.h"
#include "src/matmul.h"
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
    #define KDIM 1000
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
    float* C_ref = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    init_rand(A, M * K);
    init_rand(B, K * N);

    double FLOP = 2 * (double)M * N * K;

    uint64_t start = timer();
    matmul_naive(A, B, C_ref, M, N, K);
    uint64_t end = timer();

    double exec_time = (end - start) * 1e-9;
    double FLOPS = FLOP / exec_time;

    printf("Exec. time = %.3fms\n", exec_time * 1000);
    printf("GFLOPS = %.3f\n", FLOPS / 1e9);
    
    for (int i = 0; i < NITER; i++) {
        start = timer();
        matmul(A, B, C, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        FLOPS = FLOP / exec_time;

        printf("Exec. time = %.3fms\n", exec_time * 1000);
        printf("GFLOPS = %.3f\n", FLOPS / 1e9);

        struct val_stat_t val_results = validate_mat(C, C_ref, M * N, 1e-4);

        printf("Number of mismatches = %d\n", val_results.n_error);
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0;
}