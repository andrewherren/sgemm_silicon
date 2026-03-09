/*! 
 * Placeholder comparison to MLX using its simplest CPU sgemm implementation
 * 
 * To be replaced by directly calling `mlx::matmul` (which can dispatch other kernels)
 */

#ifdef __cplusplus
extern "C" {
#endif
#ifndef MATMUL_MLX_H_
#define MATMUL_MLX_H_

#ifdef MLX_ON
void matmul_mlx(float* A, float* B, float* C, const int m, const int n, const int k);
#endif // 

#endif // MATMUL_MLX_H_
#ifdef __cplusplus
}
#endif