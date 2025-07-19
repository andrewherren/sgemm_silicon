#ifndef MATMUL_H_
#define MATMUL_H_

void pack_panelA(float* A_matrix, float* A_panel, int M, int K, int mc, int kc, int ic, int pc, int mr);
void pack_panelB(float* B_matrix, float* B_panel, int K, int N, int kc, int nc, int pc, int jc, int nr);
void matmul_blocked_kernel(float* A, float* B, float* C, const int m, const int n, const int k);
void matmul_basic_kernel(float* A, float* B, float* C, const int m, const int n, const int k);
void matmul_naive(float* A, float* B, float* C, const int m, const int n, const int k);
void matmul_accelerate(float* A, float* B, float* C, const int m, const int n, const int k);

#endif // MATMUL_H_