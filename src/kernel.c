#include "kernel.h"
#include <arm_neon.h>

inline void fma_loop(float* blockA_packed,
                     float* blockB_packed,
                     float* C,
                     int M,
                     int mr,
                     int nr, 
                     int kc) {

    float C_accum[KERNEL_DIM_2][KERNEL_DIM_1] = {};
    
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            C_accum[0][i] += *(blockA_packed+i) * *(blockB_packed);
            C_accum[1][i] += *(blockA_packed+i) * *(blockB_packed+1);
            C_accum[2][i] += *(blockA_packed+i) * *(blockB_packed+2);
        }
        blockA_packed += KERNEL_DIM_1;
        blockB_packed += KERNEL_DIM_2;
    }

    for (int j = 0; j < nr; j++) {
        for (int i = 0; i < mr; i++) {
            *(C + j * M + i) = C_accum[j][i];
        }        
    }
}

void kernel_12x3(float* blockA_packed,
                 float* blockB_packed,
                 float* C,
                 int M,
                 int mr,
                 int nr,
                 int kc) {
    fma_loop(blockA_packed, 
             blockB_packed, 
             C, M, mr, nr, kc);
}

void kernel_12x3_no_blocking(float* A_tilde,
                             float* B_tilde,
                             float* C_tilde,
                             int M,
                             int N,
                             int K, 
                             int kernel_dim_1, 
                             int kernel_dim_2) {
    // Accumulate C_tilde
    float C_accum[KERNEL_DIM_2][KERNEL_DIM_1] = {};
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
