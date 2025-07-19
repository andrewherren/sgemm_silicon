#include "kernel.h"
#ifdef OMP_ON
#include <omp.h>
#endif
#include <stdio.h>
#include <Accelerate/Accelerate.h>
#include <vecLib/cblas.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

static float panelA_packed[MC * KC] __attribute__((aligned(64)));
static float panelB_packed[NC * KC] __attribute__((aligned(64)));

void pack_panelA(float* A_matrix, float* A_panel, int M, int K, int mc, int kc, int ic, int pc, int mr) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < mc; i += mr) {
        int mr_actual = min(mr, mc - i);
        for (int p = 0; p < kc; p++) {
            for (int k = 0; k < mr_actual; k++) {
                *(A_panel + i * kc + p * mr + k) = *(A_matrix + pc * M + ic + p * M + (i + k));
            }
            for (int k = mr_actual; k < mr; k++) {
                *(A_panel + i * kc + p * mr + k) = 0.0;
            }
        }
    }    
}

void pack_panelB(float* B_matrix, float* B_panel, int K, int N, int kc, int nc, int pc, int jc, int nr) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int j = 0; j < nc; j += nr) {
        int nr_actual = min(nr, nc - j);
        for (int p = 0; p < kc; p++) {
            for (int k = 0; k < nr_actual; k++) {
                *(B_panel + j * kc + p * nr + k) = *(B_matrix + jc * K + pc + (j + k) * K + p);
            }
            for (int k = nr_actual; k < nr; k++) {
                *(B_panel + j * kc + p * nr + k) = 0;
            }
        }
    }    
}

void matmul_blocked_kernel(float* A, float* B, float* C, int M, int N, int K) {
    for (int jc = 0; jc < N; jc += NC) {
        int nc = min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            int kc = min(KC, K - pc);
            pack_panelB(B, &panelB_packed[0], K, N, kc, nc, pc, jc, NR);
            for (int ic = 0; ic < M; ic += MC) {
                int mc = min(MC, M - ic);
                pack_panelA(A, &panelA_packed[0], M, K, mc, kc, ic, pc, MR);
                PRAGMA_OMP_PARALLEL_FOR
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = min(MR, mc - ir);
                        // printf("i = %d, j = %d, ir = %d, jr = %d, mc = %d, nc = %d, mr = %d, nr = %d\n", i, j, ir, jr, mc, nc, mr, nr);
                        kernel_blocked(&panelA_packed[0], &panelB_packed[0], C, M, N, K, mc, nc, kc, mr, nr, ic, jc, pc, ir, jr);
                    }
                }
            }
        }
    }
}

void matmul_basic_kernel(float* A, float* B, float* C, int M, int N, int K) {
PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            int kernel_dim_1 = min(MR, M - i);
            int kernel_dim_2 = min(NR, N - j);
            kernel_no_blocking(&A[i], &B[j * K], &C[j * M + i], M, N, K, kernel_dim_1, kernel_dim_2);
        }
    }
}

void matmul_naive(float* A, float* B, float* C, int m, int n, int k) {
PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float accumulator = 0;
            for (int p = 0; p < k; p++) {
                accumulator += A[p * m + i] * B[j * k + p];
            }
            C[j * m + i] = accumulator;
        }
    }
}

void matmul_accelerate(float* A, float* B, float* C, int m, int n, int k) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1., A, m, B, k, 0., C, m);
}
