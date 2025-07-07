#ifndef UTILS_H_
#define UTILS_H_
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_rand(float* mat, size_t n_elem);

void init_const(float* mat, float value, size_t n_elem);

struct val_stat_t {
    int n_error;
    int n_nans;
    int n_inf;
};

struct val_stat_t validate_mat(float* mat, float* mat_ref, size_t n_elem, float eps);

uint64_t timer();

void printfn(const char* str, int n);

#endif // UTILS_H_