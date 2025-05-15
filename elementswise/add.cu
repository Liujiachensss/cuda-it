#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <utils/init.cuh>
#include <utils/type.cuh>

template <typename T>
__global__ void add_v0(T *a, T *b, T *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
__global__ void add_v1(const T *a, const T *b, T *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
__global__ void add_v2(ccr_ptr<T> a, ccr_ptr<T> b, r_ptr<T> c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() { 
    uint len = 2<<20;
    
    thrust::device_vector<float> a(len), b(len), c(len);
    init(a);
    init(b);
    init(c);
    

    add_v2<<<(len + 255) / 256, 256>>>(a, b, c, len);

    // add_v0<<<(len + 255) / 256, 256>>>(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(c.data()), len);

 }