#ifndef MATMUL_H_
#define MATMUL_H_

void matmul_basic_kernel(float* A, float* B, float* C, const int m, const int n, const int k);
void matmul_naive(float* A, float* B, float* C, const int m, const int n, const int k);

#endif // MATMUL_H_