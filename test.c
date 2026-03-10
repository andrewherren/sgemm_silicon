#include "src/utils.h"
#include "src/matmul.h"
#ifdef EIGEN_ON
#include "src/eigen_extension.h"
#endif
#ifdef MLX_ON
#include "src/mlx_extension.h"
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
    float* C_unrolled = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_simd = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_blis = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_accel = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_eigen = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_mlx = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
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

    double exec_time_naive = (end - start) * 1e-9;
    double FLOPS_naive = FLOP / exec_time_naive;

    printf("Exec. time = %.3fms\n", exec_time_naive * 1000);
    printf("GFLOPS = %.3f\n", FLOPS_naive / 1e9);

    double exec_time;

    double avg_exec_time_block = 0.0;
    double avg_FLOPS_block = 0.0;
    double avg_num_mismatches_block = 0.0;
    double total_exec_time_block = 0.0;
    double total_FLOPS_block = 0.0;
    double total_num_mismatches_block = 0.0;

    double avg_exec_time_block_unroll = 0.0;
    double avg_FLOPS_block_unroll = 0.0;
    double avg_num_mismatches_block_unroll = 0.0;
    double total_exec_time_block_unroll = 0.0;
    double total_FLOPS_block_unroll = 0.0;
    double total_num_mismatches_block_unroll = 0.0;

    double avg_exec_time_block_unroll_simd = 0.0;
    double avg_FLOPS_block_unroll_simd = 0.0;
    double avg_num_mismatches_block_unroll_simd = 0.0;
    double total_exec_time_block_unroll_simd = 0.0;
    double total_FLOPS_block_unroll_simd = 0.0;
    double total_num_mismatches_block_unroll_simd = 0.0;

    double avg_exec_time_eigen = 0.0;
    double avg_FLOPS_eigen = 0.0;
    double avg_num_mismatches_eigen = 0.0;
    double total_exec_time_eigen = 0.0;
    double total_FLOPS_eigen = 0.0;
    double total_num_mismatches_eigen = 0.0;

    double avg_exec_time_mlx = 0.0;
    double avg_FLOPS_mlx = 0.0;
    double avg_num_mismatches_mlx = 0.0;
    double total_exec_time_mlx = 0.0;
    double total_FLOPS_mlx = 0.0;
    double total_num_mismatches_mlx = 0.0;

    double avg_exec_time_accel = 0.0;
    double avg_FLOPS_accel = 0.0;
    double avg_num_mismatches_accel = 0.0;
    double total_exec_time_accel = 0.0;
    double total_FLOPS_accel = 0.0;
    double total_num_mismatches_accel = 0.0;

    double avg_exec_time_blis = 0.0;
    double avg_FLOPS_blis = 0.0;
    double avg_num_mismatches_blis = 0.0;
    double total_exec_time_blis = 0.0;
    double total_FLOPS_blis = 0.0;
    double total_num_mismatches_blis = 0.0;
    
    for (int i = 0; i < NITER; i++) {
        init_const(C, 0.0, M*N);
        start = timer();
        matmul_blocked_kernel(A, B, C, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_block += FLOP / exec_time;
        total_exec_time_block += exec_time;

        struct val_stat_t val_results = validate_mat(C, C_ref, M * N, 1e-4);
        total_num_mismatches_block += val_results.n_error;
        
        init_const(C_unrolled, 0.0, M*N);
        start = timer();
        matmul_blocked_kernel_unrolled(A, B, C_unrolled, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_block_unroll += FLOP / exec_time;
        total_exec_time_block_unroll += exec_time;

        val_results = validate_mat(C_unrolled, C_ref, M * N, 1e-4);
        total_num_mismatches_block_unroll += val_results.n_error;
        
        init_const(C_blis, 0.0, M*N);
        start = timer();
        matmul_blocked_kernel_unrolled(A, B, C_blis, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_blis += FLOP / exec_time;
        total_exec_time_blis += exec_time;

        val_results = validate_mat(C_blis, C_ref, M * N, 1e-4);
        total_num_mismatches_blis += val_results.n_error;
        
        init_const(C_simd, 0.0, M*N);
        start = timer();
        matmul_blocked_kernel_simd(A, B, C_simd, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_block_unroll_simd += FLOP / exec_time;
        total_exec_time_block_unroll_simd += exec_time;

        val_results = validate_mat(C_simd, C_ref, M * N, 1e-4);
        total_num_mismatches_block_unroll_simd += val_results.n_error;

        #ifdef EIGEN_ON
        
        start = timer();
        matmul_eigen(A, B, C_eigen, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_eigen += FLOP / exec_time;
        total_exec_time_eigen += exec_time;

        val_results = validate_mat(C_eigen, C_ref, M * N, 1e-4);
        total_num_mismatches_eigen += val_results.n_error;

        #endif

        #ifdef MLX_ON
        
        start = timer();
        matmul_mlx(A, B, C_mlx, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_mlx += FLOP / exec_time;
        total_exec_time_mlx += exec_time;

        val_results = validate_mat(C_mlx, C_ref, M * N, 1e-4);
        total_num_mismatches_mlx += val_results.n_error;

        #endif
        
        start = timer();
        matmul_accelerate(A, B, C_accel, M, N, K);
        end = timer();

        exec_time = (end - start) * 1e-9;
        total_FLOPS_accel += FLOP / exec_time;
        total_exec_time_accel += exec_time;

        val_results = validate_mat(C_accel, C_ref, M * N, 1e-4);
        total_num_mismatches_accel += val_results.n_error;

        // if (i == NITER - 1) {
        //    for (int j = 0; j < 5; j++) {
        //        for (int k = 0; k < 5; k++) {
        //            printf("C_simd[%d,%d] = %.3f\n", j, k, *(C_simd + k * M + j));
        //            printf("C_ref[%d,%d] = %.3f\n", j, k, *(C_ref + k * M + j));
        //        }
        //    }
        // }
    }

    // Compute averages
    avg_FLOPS_block = total_FLOPS_block / NITER;
    avg_exec_time_block = total_exec_time_block / NITER;
    printf("Exec. time (blocked, not unrolled) = %.3fms\n", avg_exec_time_block * 1000);
    printf("GFLOPS (blocked, not unrolled) = %.3f\n", avg_FLOPS_block / 1e9);
    avg_num_mismatches_block = total_num_mismatches_block / NITER;
    printf("Number of mismatches (blocked, not unrolled) = %.2f\n", avg_num_mismatches_block);

    avg_FLOPS_block_unroll = total_FLOPS_block_unroll / NITER;
    avg_exec_time_block_unroll = total_exec_time_block_unroll / NITER;
    printf("Exec. time (blocked, unrolled) = %.3fms\n", avg_exec_time_block_unroll * 1000);
    printf("GFLOPS (blocked, unrolled) = %.3f\n", avg_FLOPS_block_unroll / 1e9);
    avg_num_mismatches_block_unroll = total_num_mismatches_block_unroll / NITER;
    printf("Number of mismatches (blocked, unrolled) = %.2f\n", avg_num_mismatches_block_unroll);

    avg_FLOPS_block_unroll_simd = total_FLOPS_block_unroll_simd / NITER;
    avg_exec_time_block_unroll_simd = total_exec_time_block_unroll_simd / NITER;
    printf("Exec. time (blocked, unrolled, SIMD) = %.3fms\n", avg_exec_time_block_unroll_simd * 1000);
    printf("GFLOPS (blocked, unrolled, SIMD) = %.3f\n", avg_FLOPS_block_unroll_simd / 1e9);
    avg_num_mismatches_block_unroll_simd = total_num_mismatches_block_unroll_simd / NITER;
    printf("Number of mismatches (blocked, unrolled, SIMD) = %.2f\n", avg_num_mismatches_block_unroll_simd);
    
    avg_FLOPS_blis = total_FLOPS_blis / NITER;
    avg_exec_time_blis = total_exec_time_blis / NITER;
    printf("Exec. time (BLIS) = %.3fms\n", avg_exec_time_blis * 1000);
    printf("GFLOPS (BLIS) = %.3f\n", avg_FLOPS_blis / 1e9);
    avg_num_mismatches_blis = total_num_mismatches_blis / NITER;
    printf("Number of mismatches (BLIS) = %.2f\n", avg_num_mismatches_blis);

    avg_FLOPS_accel = total_FLOPS_accel / NITER;
    avg_exec_time_accel = total_exec_time_accel / NITER;
    printf("Exec. time (accelerate) = %.3fms\n", avg_exec_time_accel * 1000);
    printf("GFLOPS (accelerate) = %.3f\n", avg_FLOPS_accel / 1e9);
    avg_num_mismatches_accel = total_num_mismatches_accel / NITER;
    printf("Number of mismatches (accelerate) = %.2f\n", avg_num_mismatches_accel);
    
    #ifdef EIGEN_ON
    
    avg_FLOPS_eigen = total_FLOPS_eigen / NITER;
    avg_exec_time_eigen = total_exec_time_eigen / NITER;
    printf("Exec. time (eigen) = %.3fms\n", avg_exec_time_eigen * 1000);
    printf("GFLOPS (eigen) = %.3f\n", avg_FLOPS_eigen / 1e9);
    avg_num_mismatches_eigen = total_num_mismatches_eigen / NITER;
    printf("Number of mismatches (eigen) = %.2f\n", avg_num_mismatches_eigen);

    #endif

    #ifdef MLX_ON
    
    avg_FLOPS_mlx = total_FLOPS_mlx / NITER;
    avg_exec_time_mlx = total_exec_time_mlx / NITER;
    printf("Exec. time (mlx) = %.3fms\n", avg_exec_time_mlx * 1000);
    printf("GFLOPS (mlx) = %.3f\n", avg_FLOPS_mlx / 1e9);
    avg_num_mismatches_mlx = total_num_mismatches_mlx / NITER;
    printf("Number of mismatches (mlx) = %.2f\n", avg_num_mismatches_mlx);

    #endif

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);
    _mm_free(C_unrolled);
    _mm_free(C_simd);
    _mm_free(C_blis);
    _mm_free(C_accel);
    _mm_free(C_eigen);
    _mm_free(C_mlx);

    return 0;
}
