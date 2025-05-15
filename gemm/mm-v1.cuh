#pragma once

#include <cuda_runtime.h>

#include <utils/type.cuh>

// 写入时合并访存
template <typename T>
__global__ void mm_v1(ccr_ptr<T> A, ccr_ptr<T> B, r_ptr<T> C, int M, int N, int K) {
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