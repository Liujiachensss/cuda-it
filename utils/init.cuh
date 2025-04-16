#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace utils {

template <typename T> void init(thrust::device_vector<T> &v, T val) {
  thrust::fill(v.begin(), v.end(), val);
}

template <typename T> void init(thrust::host_vector<T> &v, T val) {
  thrust::fill(v.begin(), v.end(), val);
}

// random initialization
template <typename T> void init(thrust::device_vector<T> &v) {
  thrust::generate(v.begin(), v.end(), []() { return rand() % 100; });
}

template <typename T> void init(thrust::host_vector<T> &v) {
  thrust::generate(v.begin(), v.end(), []() { return rand() % 100; });
}

} // namespace utils
