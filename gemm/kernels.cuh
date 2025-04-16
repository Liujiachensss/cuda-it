#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <string>
#include <unordered_map>
#include <utils/init.cuh>
#include <utils/verify.cuh>

#include "kernels.h"
#include "mm-cublas.cuh"
#include "mm-naive.cuh"
#include "mm-v1.cuh"

#define TH_PTR(obj) obj.data().get()

template <class T>
using kernel_info = std::unordered_map<std::string, kernel_func<T>>;

using thrust_hv = thrust::host_vector<float>;
using thrust_dv = thrust::device_vector<float>;

template <class T>
class Kernels {
 public:
  Kernels(thrust_dv &d_A, thrust_dv &d_B, thrust_dv &d_C)
      : d_A(&d_A),
        d_B(&d_B),
        d_C(&d_C),
        d_a(d_A.data().get()),
        d_b(d_B.data().get()),
        d_c(d_C.data().get()) {
    registor();
  }

  //
  void registor() { KERNEL_LIST() }

  // void registor() {
  //   // void (*f_cublas)(cc_ptr<float>, cc_ptr<float>, _ptr<float>, int, int,
  //   int) =
  //   //     mm_cublas<float>;
  //   // kernel_func_t<float> f_cublas = mm_cublas<float>;
  //   auto f_cublas = mm_cublas<float>;
  //   register_kernel("mm-cublas", f_cublas);
  // }

  void verify(int M, int N, int K, std::string const &base_name = {}) {
    kernel_func<T> base_func =
        base_name.empty() ? kernels.begin()->second : kernels[base_name];
    printf("verify base is : %s\n", base_name.c_str());

    thrust::host_vector<float> h_base(*d_C);
    utils::init(h_base, 0.0f);
    thrust::host_vector<float> h_out(*d_C);
    utils::init(h_out, 0.0f);

    run(base_func, d_a, d_b, d_c, M, N, K);
    h_base = *d_C;

    for (auto mp : kernels) {
      printf("verify %s: \n", mp.first.c_str());
      run(mp.second, d_a, d_b, d_c, M, N, K);
      h_out = *d_C;
      utils::compare(h_base, h_out);
      utils::init(h_out, 0.0f);
    }
  }

  kernel_info<float> &getInfo() { return kernels; }

  auto begin() { return kernels.begin(); }
  auto end() { return kernels.end(); }

  // void verify_kernel(std::string const &base_name, kernel_func<float> kernel,
  //                    int M, int N, int K) {
  //   run(kernel, d_a, d_b, d_c, M, N, K);

  //   for (auto mp : kernels) {
  //     run(mp.first, kernel, M, N, K);
  //   }
  // }

  static void print_info() {
    for (auto mp : kernels) {
      printf("%s\n", mp.first.c_str());
    }
  }

 private:
  kernel_info<T> register_kernel(std::string name, kernel_func<T> kernel) {
    kernels[name] = kernel;
    return kernels;
  }

  void run(kernel_func<T> kernel, ccr_ptr<T> A, ccr_ptr<T> B, r_ptr<T> C, int M,
           int N, int K) {
    dim3 grids(1, 16, 16);
    dim3 blocks(1, 16, 16);
    kernel<<<grids, blocks>>>(A, B, C, M, N, K);
  }

 private:
  static inline kernel_info<T> kernels{};

  thrust_dv *d_A;
  thrust_dv *d_B;
  thrust_dv *d_C;

  ccr_ptr<T> d_a;
  ccr_ptr<T> d_b;
  r_ptr<T> d_c;

 public:
#ifdef REPEAT
  static const inline int repeat = REPEAT;
#else
  static const inline int repeat = 1;
#endif
};

template <class T>
void run_kernel(std::string const &name, cudaEvent_t &start, cudaEvent_t &stop,
                kernel_func<T> kernel, dim3 const &grids, dim3 const &blocks,
                ccr_ptr<T> A, ccr_ptr<T> B, r_ptr<T> C, int M, int N, int K) {
  cudaEventRecord(start, 0);

#pragma unroll
  for (int i = 0; i < Kernels<T>::repeat; ++i) {
    kernel<<<grids, blocks>>>(A, B, C, M, N, K);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  printf("%s: %f ms\n", name.c_str(), time);
}
