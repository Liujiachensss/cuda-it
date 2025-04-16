#include "kernels.cuh"

// #include "utils/verify.cuh"
// #include "init.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  int M = 4092;
  int N = 4092;
  int K = 4092;
  int size_A = M * K;
  int size_B = K * N;
  int size_C = M * N;

  thrust::host_vector<float> h_A(size_A);
  thrust::host_vector<float> h_B(size_B);
  thrust::host_vector<float> h_C(size_C);
  thrust::device_vector<float> d_A(size_A);
  thrust::device_vector<float> d_B(size_B);
  thrust::device_vector<float> d_C(size_C);

  utils::init(h_A);
  utils::init(h_B);
  utils::init(h_C, 0.0f);

  d_A = h_A;
  d_B = h_B;
  d_C = h_C;

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  auto ops = Kernels<float>(d_A, d_B, d_C);
  ops.verify(M, N, K);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (auto mp : ops) {
    run_kernel(mp.first, start, stop, mp.second, grid, block, d_A.data().get(),
               d_B.data().get(), d_C.data().get(), M, N, K);
  }

  return 0;
}