#pragma once

// void registor() {
//   cudaFuncSetAttribute(mm_v1, cudaFuncAttributeMaxDynamicSharedMemorySize,
//   16384);
// }

#include <utils/type.cuh>

#include "cx.h"

template <class T>
using kernel_func = void (*)(cc_ptr<T>, cc_ptr<T>, _ptr<T>, int, int, int);

// #define KERNEL_LIST()                                                          \
//   register_kernel("naive", mm_naive<float>);                                   \
//   register_kernel("v1", mm_v1<float>);                                         \
//   register_kernel("cublas", mm_cublas<float>);

#define KERNEL_LIST()                     \
  auto f_cublas = mm_cublas<float>;       \
  register_kernel("mm_cublas", f_cublas); \
  auto f_naive = mm_naive<float>;         \
  register_kernel("mm_naive", f_naive);   \
  auto f_v1 = mm_v1<float>;               \
  register_kernel("mm_v1", f_v1);
