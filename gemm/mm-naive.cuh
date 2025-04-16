#pragma once

#include <cuda_runtime.h>

#include <utils/type.cuh>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>

// #include "compare.h"
// #include "cxtimers.h"

#define NAIVE_GEMM 1

template <typename T>
__global__ void mm_naive(ccr_ptr<T> A, ccr_ptr<T> B, r_ptr<T> C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// using mm_naive_t = void (*)(float *, float *, float *, int, int, int);

