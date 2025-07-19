#ifdef __cplusplus
extern "C" {
#endif
#ifndef MATMUL_EIGEN_H_
#define MATMUL_EIGEN_H_

#ifdef EIGEN_ON
void matmul_eigen(float* A, float* B, float* C, const int m, const int n, const int k);
#endif

#endif // MATMUL_EIGEN_H_
#ifdef __cplusplus
}
#endif