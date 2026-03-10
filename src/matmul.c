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
    for (int i = 0; i < mc; i += MR) {
        int mr_actual = min(MR, mc - i);
        for (int p = 0; p < kc; p++) {
            for (int k = 0; k < mr_actual; k++) {
                *(A_panel + i * kc + p * MR + k) = *(A_matrix + pc * M + ic + p * M + (i + k));
            }
            for (int k = mr_actual; k < MR; k++) {
                *(A_panel + i * kc + p * MR + k) = 0.0;
            }
        }
    }    
}

void pack_panelB(float* B_matrix, float* B_panel, int K, int N, int kc, int nc, int pc, int jc, int nr) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int j = 0; j < nc; j += NR) {
        int nr_actual = min(NR, nc - j);
        for (int p = 0; p < kc; p++) {
            for (int k = 0; k < nr_actual; k++) {
                *(B_panel + j * kc + p * NR + k) = *(B_matrix + jc * K + pc + (j + k) * K + p);
            }
            for (int k = nr_actual; k < NR; k++) {
                *(B_panel + j * kc + p * NR + k) = 0;
            }
        }
    }    
}

void matmul_blocked_kernel_simd(float* A, float* B, float* C, int M, int N, int K) {
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
                        kernel_blocked_simd(&panelA_packed[0], &panelB_packed[0], C, M, N, K, mc, nc, kc, mr, nr, ic, jc, pc, ir, jr);
                    }
                }
            }
        }
    }
}

void matmul_blocked_kernel_unrolled(float* A, float* B, float* C, int M, int N, int K) {
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
                        kernel_blocked_unrolled(&panelA_packed[0], &panelB_packed[0], C, M, N, K, mc, nc, kc, mr, nr, ic, jc, pc, ir, jr);
                    }
                }
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

void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float accumulator = 0;
            for (int p = 0; p < K; p++) {
                accumulator += A[p * M + i] * B[j * K + p];
            }
            C[j * M + i] = accumulator;
        }
    }
}

void matmul_accelerate(float* A, float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., A, M, B, K, 0., C, M);
}

void matmul_blis(float* A, float* B, float* C, const int M, const int N, const int K) {
    // Execute the 5-level loop outlined in the BLIS paper
    #pragma omp parallel for num_threads(OMP_NTHREADS)
    for (int jc = 0; jc < N; jc += NC_BLIS) {
        // Pre-allocated panelA and panelB
        float panelA[MC_BLIS * KC_BLIS] __attribute__((aligned(64))) = {};
        float panelB[NC_BLIS * KC_BLIS] __attribute__((aligned(64))) = {};
        // Pre-allocate accumulated a and b
        float a, b;
        int nc = min(NC_BLIS, N - jc);
        for (int pc = 0; pc < K; pc += KC_BLIS) {
            int kc = min(KC_BLIS, K - pc);
            // Pack a column-major "panel" of a kc x nc submatrix of B
            for (int j = 0; j < nc; j += NR_BLIS) {
                int nr_actual = min(NR_BLIS, nc - j);
                for (int p = 0; p < kc; p++) {
                    for (int k = 0; k < nr_actual; k++) {
                        *(panelB + j * kc + p * NR_BLIS + k) = *(B + jc * K + pc + (j + k) * K + p);
                    }
                    for (int k = nr_actual; k < NR_BLIS; k++) {
                        *(panelB + j * kc + p * NR_BLIS + k) = 0;
                    }
                }
            }
            for (int ic = 0; ic < M; ic += MC_BLIS) {
                int mc = min(MC_BLIS, M - ic);
                // Pack a row-major "panel" of a mc by kc submatrix of A
                for (int i = 0; i < mc; i += MR_BLIS) {
                    int mr_actual = min(MR_BLIS, mc - i);
                    for (int p = 0; p < kc; p++) {
                        for (int k = 0; k < mr_actual; k++) {
                            *(panelA + i * kc + p * MR_BLIS + k) = *(A + pc * M + ic + p * M + (i + k));
                        }
                        for (int k = mr_actual; k < MR_BLIS; k++) {
                            *(panelA + i * kc + p * MR_BLIS + k) = 0.0;
                        }
                    }
                }
                for (int jr = 0; jr < nc; jr += NR_BLIS) {
                    int nr = min(NR_BLIS, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR_BLIS) {
                        int mr = min(MR_BLIS, mc - ir);
                        // printf("i = %d, j = %d, ir = %d, jr = %d, mc = %d, nc = %d, mr = %d, nr = %d\n", i, j, ir, jr, mc, nc, mr, nr);
                        // Load existing output into auxiliary accumulator
                        float C_aux[NR_BLIS * MR_BLIS] = {};
                        for (int j = 0; j < nr; j++) {
                            for (int i = 0; i < mr; i++) {
                                C_aux[j*MR_BLIS + i] = C[(jc + jr + j) * M + ic + ir + i];
                            }
                        }
                        
                        // Accumulate result
                        for (int p = 0; p < kc; p++) {
                            for (int i = 0; i < nr; i++) {
                                b = *(panelB + jr * kc + p * NR_BLIS + i);
                                for (int j = 0; j < mr; j++) {
                                    a = *(panelA + ir * kc + p * MR_BLIS + j);
                                    C_aux[i*MR_BLIS + j] += a * b;
                                }
                            }
                        }

                        // Save accumulated data back to output matrix
                        for (int j = 0; j < nr; j++) {
                            for (int i = 0; i < mr; i++) {
                                C[(jc + jr + j) * M + ic + ir + i] = C_aux[j*MR_BLIS + i];
                            }
                        }
                    }
                }
            }
        }
    }
}
