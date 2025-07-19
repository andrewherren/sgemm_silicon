#ifndef KERNEL_H_
#define KERNEL_H_
#include <arm_neon.h>

#ifdef OMP_ON
#ifndef OMP_NTHREADS
    #define OMP_NTHREADS 8
#endif
#ifndef OMP_SCHEDULE
    #define OMP_SCHEDULE auto
#endif
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(OMP_NTHREADS)")
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE _Pragma("omp parallel for collapse(2) schedule(OMP_SCHEDULE) num_threads(OMP_NTHREADS)")
#else
#define OMP_NTHREADS 8
#define OMP_SCHEDULE auto
#define PRAGMA_OMP_PARALLEL_FOR
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE
#endif

#define MR 8
#define NR 8
#define MC 400
#define NC 400
#define KC 1000

void kernel_no_blocking(float* A_tilde,
                        float* B_tilde,
                        float* C_tilde,
                        int M,
                        int N,
                        int K, 
                        int kernel_dim_1, 
                        int kernel_dim_2);

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
                    int jr);

#endif // KERNEL_H_
