#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "compare.h"
#include "cxtimers.h"

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
__global__ void stencil_v1(float* in, float* out, unsigned int N, float* c) {
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

__global__ void stencil_v2(float* in, float* out, unsigned int N, float* c) {
  constexpr const uint IN_TILE_DIM = 8;
  constexpr const uint OUT_TILE_DIM = 8;

  int iStart = blockIdx.z * OUT_TILE_DIM;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.x - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.y - 1;
  __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
  __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
  __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

  if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
    inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
  }

  if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
  }

  for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
    if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
      inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
          threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
        out[i * N * N + j * N + k] =
            c[0] * inCurr_s[threadIdx.y][threadIdx.x] +
            c[1] * inCurr_s[threadIdx.y][threadIdx.x + 1] +
            c[2] * inCurr_s[threadIdx.y + 1][threadIdx.x] +
            c[3] * inCurr_s[threadIdx.y + 1][threadIdx.x + 1] +
            c[4] * inPrev_s[threadIdx.y + 1][threadIdx.x] +
            c[5] * inPrev_s[threadIdx.y + 1][threadIdx.x + 1] +
            c[6] * inNext_s[threadIdx.y + 1][threadIdx.x];
      }
    }
    __syncthreads();
    inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
    inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
  }
}

__global__ void stencil_v3(float* in, float* out, unsigned int N, float* c) {
  constexpr const uint IN_TILE_DIM = 8;
  constexpr const uint OUT_TILE_DIM = 8;

  int iStart = blockIdx.z * OUT_TILE_DIM;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
  float inPrev;
  __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
  float inCurr;
  float inNext;
  if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
    inPrev = in[(iStart - 1) * N * N + j * N + k];
  }
  if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr = in[iStart * N * N + j * N + k];
    inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
  }
  for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
    if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
      inNext = in[(i + 1) * N * N + j * N + k];
    }

    __syncthreads();
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
          threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
        out[i * N * N + j * N + k] =
            c[0] * inCurr_s[threadIdx.y][threadIdx.x - 1] +
            c[1] * inCurr_s[threadIdx.y][threadIdx.x + 1] +
            c[2] * inCurr_s[threadIdx.y + 1][threadIdx.x] +
            c[3] * inCurr_s[threadIdx.y + 1][threadIdx.x + 1] +
            c[4] * inCurr_s[threadIdx.y - 1][threadIdx.x] + c[5] * inPrev +
            c[6] * inNext;
      }
    }
    __syncthreads();
    inPrev = inCurr;
    inCurr = inNext;
    inCurr_s[threadIdx.y][threadIdx.x] = inNext;
  }
}

int main(int argc, char** argv) {
  int nx = (argc > 1) ? atoi(argv[1]) : 1024;
  int ny = (argc > 2) ? atoi(argv[2]) : 1024;
  int nz = (argc > 3) ? atoi(argv[3]) : 2;
  uint threadx = (argc > 4) ? atoi(argv[4]) : 16;  // 8, 16, 32 or 64
  uint thready = (argc > 5) ? atoi(argv[5]) : 16;  // product <= 1024
  uint threadz = (argc > 6) ? atoi(argv[6]) : 2;  // product <= 1024

  int size = nx * ny;
  int n = size;

  thrust::host_vector<float> h_a(size);
  thrust::host_vector<float> h_b(size);

  thrust::device_vector<float> d_in(size);
  thrust::device_vector<float> d_out(size);
  thrust::device_vector<float> d_c{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

  dim3 block = {threadx, thready, threadz};
  dim3 grid = {(nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
               (nz + block.z - 1) / block.z};

  auto idx = [&nx](int y, int x) { return y * nx + x; };
  for (int y = 0; y < ny; y++) {
    h_a[idx(y, 0)] = h_a[idx(y, nx - 1)] = 1.0f;
  }

  h_a[idx(0, 0)] = h_a[idx(0, nx - 1)] = h_a[idx(ny - 1, 0)] =
      h_a[idx(ny - 1, nx - 1)] = 0.5f;

  d_in = h_a;
  d_out = h_a;
  h_b = h_a;

  std::vector<double> times;
  cx::timer tm;
  stencil_naive<float><<<grid, block>>>(d_in.data().get(), d_c.data().get(),
                                        d_out.data().get(), n);
  cudaDeviceSynchronize();
  double duration = tm.lap_ms();
  times.push_back(duration);
  thrust::host_vector<float> h_out = d_out;

  {
    tm.reset();
    stencil_v1<<<grid, block>>>(d_in.data().get(), d_out.data().get(), n,
                                d_c.data().get());
    cudaDeviceSynchronize();
    duration = tm.lap_ms();
    times.push_back(duration);
    thrust::host_vector<float> h_out_v1 = d_out;
    compare_host_vectors(h_out, h_out_v1);
  }

  {
    tm.reset();
    stencil_v2<<<grid, block>>>(d_in.data().get(), d_out.data().get(), n,
                                d_c.data().get());
    cudaDeviceSynchronize();
    duration = tm.lap_ms();
    times.push_back(duration);
    thrust::host_vector<float> h_out_v2 = d_out;
    compare_host_vectors(h_out, h_out_v2);
  }

  {
    tm.reset();
    stencil_v3<<<grid, block>>>(d_in.data().get(), d_out.data().get(), n,
                                d_c.data().get());
    cudaDeviceSynchronize();
    duration = tm.lap_ms();
    times.push_back(duration);
    thrust::host_vector<float> h_out_v3 = d_out;
    compare_host_vectors(h_out, h_out_v3);
  }
  // double gflops_gpu =
  //     (double)(iter_gpu * 4) * (double)size / (duration * 1000000);

  // printf("stencil2d size %d x %d gflops_gpu %.3f\n", nx, ny, gflops_gpu);

  auto trans2gflops = [](double time, int size) {
    return (double)(size) / (time * 1000'000);
  };

  std::vector<double> gflops(times.size());
  std::transform(times.begin(), times.end(), gflops.begin(),
                 [&](double time) { return trans2gflops(time, size * 4); });

  std::cout << "Naive: " << gflops[0] << " GFLOPS" << std::endl;
  std::cout << "V1   : " << gflops[1] << " GFLOPS" << std::endl;
  std::cout << "V2   : " << gflops[2] << " GFLOPS" << std::endl;
  std::cout << "V3   : " << gflops[3] << " GFLOPS" << std::endl;

  // std::cout << "Naive: " << times[0] << " ms" << std::endl;
  // std::cout << "V1   : " << times[1] << " ms" << std::endl;
  // std::cout << "V2   : " << times[2] << " ms" << std::endl;
  // std::cout << "V3   : " << times[3] << " ms" << std::endl;
}