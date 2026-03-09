#ifndef MATMUL_H_
#define MATMUL_H_

void pack_panelA(float* A_matrix, float* A_panel, int M, int K, int mc, int kc, int ic, int pc, int mr);
void pack_panelB(float* B_matrix, float* B_panel, int K, int N, int kc, int nc, int pc, int jc, int nr);
void matmul_blocked_kernel_simd(float* A, float* B, float* C, const int M, const int N, const int K);
void matmul_blocked_kernel_unrolled(float* A, float* B, float* C, const int M, const int N, const int K);
void matmul_blocked_kernel(float* A, float* B, float* C, const int M, const int N, const int K);
void matmul_basic_kernel(float* A, float* B, float* C, const int M, const int N, const int K);
void matmul_naive(float* A, float* B, float* C, const int M, const int N, const int K);
void matmul_accelerate(float* A, float* B, float* C, const int M, const int N, const int K);
void matmul_blis(float* A, float* B, float* C, const int M, const int N, const int K);

#endif // MATMUL_H_