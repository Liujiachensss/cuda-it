#pragma once

#include <cublas_v2.h>

#include <utils/type.cuh>

template <typename T>
void mm_cublas(ccr_ptr<T> A, ccr_ptr<T> B, r_ptr<T> C, int M, int N, int K) {
  auto alpha = 1.0f;
  auto beta = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K,
              &beta, C, M);
}
