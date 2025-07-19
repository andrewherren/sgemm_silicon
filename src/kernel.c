#include "kernel.h"
#include <arm_neon.h>
#include <stdio.h>


void kernel_no_blocking(float* A_tilde,
                        float* B_tilde,
                        float* C_tilde,
                        int M,
                        int N,
                        int K,
                        int kernel_dim_1,
                        int kernel_dim_2) {
    // Accumulate C_tilde
    float C_accum[NR][MR] = {};
    for (int p = 0; p < K; p++) {
        for (int j = 0; j < kernel_dim_2; j++) {
            float b = B_tilde[j * K + p];
            for (int i = 0; i < kernel_dim_1; i++) {
                C_accum[j][i] += A_tilde[p * M + i] * b;
            }
        }
    }

    // Write back to C
    for (int i = 0; i < kernel_dim_1; i++) {
        for (int j = 0; j < kernel_dim_2; j++) {
            C_tilde[j * M + i] = C_accum[j][i];
        }
    }
}

void kernel_blocked(float* panelA_packed,
                    float* panelB_packed,
                    float* C,
                    int M,
                    int N,
                    int K,
                    int mc,
                    int nc,
                    int kc,
                    int mr,
                    int nr,
                    int ic,
                    int jc,
                    int pc,
                    int ir,
                    int jr) {
    // Load existing output into auxiliary accumulator
    float C_aux[NR * MR] = {};
    for (int j = 0; j < nr; j++) {
        for (int i = 0; i < mr; i++) {
            C_aux[j*MR + i] = C[(jc + jr + j) * M + ic + ir + i];
        }
    }

    // Accumulate kc outer products of mr rows from A with nr columns of B
    float a, b;
    for (int p = 0; p < kc; p++) {
        b = *(panelB_packed + jr * KC + p * NR + 0);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[0*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[0*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[0*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[0*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[0*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[0*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[0*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[0*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 1);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[1*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[1*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[1*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[1*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[1*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[1*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[1*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[1*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 2);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[2*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[2*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[2*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[2*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[2*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[2*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[2*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[2*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 3);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[3*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[3*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[3*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[3*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[3*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[3*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[3*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[3*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 4);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[4*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[4*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[4*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[4*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[4*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[4*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[4*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[4*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 5);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[5*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[5*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[5*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[5*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[5*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[5*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[5*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[5*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 6);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[6*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[6*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[6*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[6*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[6*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[6*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[6*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[6*MR + 7] += a * b;

        b = *(panelB_packed + jr * KC + p * NR + 7);
        a = *(panelA_packed + ir * KC + p * MR + 0);
        C_aux[7*MR + 0] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 1);
        C_aux[7*MR + 1] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 2);
        C_aux[7*MR + 2] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 3);
        C_aux[7*MR + 3] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 4);
        C_aux[7*MR + 4] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 5);
        C_aux[7*MR + 5] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 6);
        C_aux[7*MR + 6] += a * b;
        a = *(panelA_packed + ir * KC + p * MR + 7);
        C_aux[7*MR + 7] += a * b;
    }
    
    // Save accumulated data back to output matrix
    for (int j = 0; j < nr; j++) {
        for (int i = 0; i < mr; i++) {
            C[(jc + jr + j) * M + ic + ir + i] = C_aux[j*MR + i];
        }
    }
}
