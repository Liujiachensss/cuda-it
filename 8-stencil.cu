#include <cuda_runtime.h>
#include <thrust/device_vector.h>

template <class T>
__global__ void stencil_naive(T* in, T* c, T* out, uint n) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint j = blockIdx.y * blockDim.y + threadIdx.y;
  uint k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i > 0 && i < n - 1 && j > 0 && j < n - 1 && k > 0 && k < n - 1) {
    // out[i] = (in[i] + in[i+1] + in[i+2] + in[i+3] + in[i+4]) / 5;
    out[i * n * n + j * n + k] = (c[0] * in[(i - 1) * n * n + j * n + k] +
                                  c[1] * in[(i + 1) * n * n + j * n + k] +
                                  c[2] * in[i * n * n + (j - 1) * n + k] +
                                  c[3] * in[i * n * n + (j + 1) * n + k] +
                                  c[4] * in[i * n * n + j * n + k - 1] +
                                  c[5] * in[i * n * n + j * n + k + 1]);
  }
}

// template <class T>
__global__ void stencil_kernel(float* in, float* out, unsigned int N,
                               float* c) {
  constexpr const uint IN_TILE_DIM = 8;
  constexpr const uint OUT_TILE_DIM = 8;

  uint i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  uint j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  uint k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  
  if (i < N && j < N && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  }
  __syncthreads();
  
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
        threadIdx.x < IN_TILE_DIM - 1) {
      out[i * N * N + j * N + k] =
          c[0] * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
          c[1] * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
          c[2] * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
          c[3] * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
          c[4] * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
          c[5] * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
          c[6] * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
    }
  }
}

int main(int argc, char** argv) {
  // ...
  uint n = 1024;
  dim3 block(8, 8, 8);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y,
            (n + block.z - 1) / block.z);

  thrust::device_vector<float> d_in(n * n * n, 10.0f);
  thrust::device_vector<float> d_out(n * n * n);
  thrust::device_vector<float> d_c(6);

  if (argc > 1) {
    int arg = atoi(argv[1]);
    switch (arg) {
      case 0:
        stencil_naive<float><<<grid, block>>>(
            d_in.data().get(), d_c.data().get(), d_out.data().get(), n);
        break;

      case 1:
        stencil_kernel<<<grid, block>>>(d_in.data().get(), d_out.data().get(),
                                        n, d_c.data().get());
        break;
      default:
        break;
    }
  }
  // ...
}