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

#endif // KERNEL_H_