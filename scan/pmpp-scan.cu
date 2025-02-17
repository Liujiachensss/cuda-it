#define SECTION_SIZE 1024
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
  __shared__ float XY[SECTION_SIZE];
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
      XY[threadIdx.x] = X[i];
  }
  else {
      XY[threadIdx.x] = 0.0f;
  }
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
      __syncthreads();
      float temp{};
      if (threadIdx.x >= stride)
          temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
      __syncthreads();
      if (threadIdx.x >= stride)
          XY[threadIdx.x] = temp;
  }
  if (i < N) {
      Y[i] = XY[threadIdx.x];
  }
}
