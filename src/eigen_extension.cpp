#include "eigen_extension.h"
#ifdef EIGEN_ON
#include <Eigen/Core>

void matmul_eigen(float* A, float* B, float* C, int m, int n, int k) {
    Eigen::Map<Eigen::MatrixXf> mA(A,m,k);
    Eigen::Map<Eigen::MatrixXf> mB(B,k,n);
    Eigen::Map<Eigen::MatrixXf> mC(C,m,n);
    mC = mA * mB;
}

#endif
