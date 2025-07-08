#ifndef KERNEL_H_
#define KERNEL_H_
#include <arm_neon.h>

#define KERNEL_DIM_1 12
#define KERNEL_DIM_2 3
#define MASK_DIM_1 KERNEL_DIM_1 / 2

void kernel_12x3(float* blockA_packed,
                 float* blockB_packed,
                 float* C,
                 int M,
                 int mr,
                 int nr,
                 int kc);

void kernel_12x3_no_blocking(float* A_tilde,
                             float* B_tilde,
                             float* C_tilde,
                             int M,
                             int N,
                             int K, 
                             int kernel_dim_1, 
                             int kernel_dim_2);

#endif // KERNEL_H_