#include <cuda.h>

#include <iostream>

__constant__ int val = 1;
__global__ void kernel_constant_sc() {
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  if (tid != 0) {
    printf("Thread %d, val %d\n", tid, val);  // load val to Const$
  } else {
    // remove constant
    int *mut_val = const_cast<int *>(&val);
    asm volatile("" : "+l"(mut_val));
    // store new value
    *mut_val = 42;
  }
}

int main(int argc, const char *argv[]) {
  int n = 2;
  if (argc == 2) {
    n = strtol(argv[1], NULL, 10);
  }
  kernel_constant_sc<<<1, n>>>();
  cudaDeviceSynchronize();
  return 0;
}
