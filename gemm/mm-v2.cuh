#pragma once

#include <utils/type.cuh>

const inline uint TileSize = 32;

template <class T>
__global__ void mm_v2(ccr_ptr<T> A, ccr_ptr<T> B, r_ptr<T> C, int M, int N,
                      int K) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ T As[TileSize][TileSize];
  __shared__ T Bs[TileSize][TileSize];

  const uint threadCol = threadIdx.x % TileSize;
  const uint threadRow = threadIdx.x / TileSize;
  
  rd_cr_ptr<T> A_ = A + cRow * TileSize * K;
  rd_cr_ptr<T> B_ = B + cRow * TileSize * N;
//   A += cRow * TileSize * K;
//   B += cCol * TileSize * N;

  T sum = 0.0f;
  for (uint i = 0; i < ceil(K / TileSize); i++) {
    As[threadRow][threadCol] = A[threadRow * K + i * TileSize + threadCol];
    Bs[threadRow][threadCol] = B[i * TileSize + threadRow * N + threadCol];
    __syncthreads();

    A_ += TileSize;
    B_ += TileSize;
  }
}

void launchSgemm(rd_cr_ptr<float> A, rd_cr_ptr<float> B, r_ptr<float> C, int M,
                 int N, int K) {
  dim3 gridDim(ceil(M / TileSize), ceil(N / TileSize), 1);
  dim3 blockDim(TileSize, TileSize, 1);

  mm_v2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}