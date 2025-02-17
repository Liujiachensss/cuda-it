#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r,
                                            int width, int height) {
  int outCol = blockIdx.x * blockDim.y + threadIdx.y;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  float Pvalue = 0.0f;
  for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue += F[fRow + fCol * width] * N[inRow * width + inCol];
      }
    }
  }
  P[outRow + outCol * width] = Pvalue;
}