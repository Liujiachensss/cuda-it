#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cx.h"

__global__ void convolution_2D_basic_kernel(float *N, float **F, float **P, int r,
                                            int width, int height) {
  int outCol = blockIdx.x * blockDim.y + threadIdx.y;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  float Pvalue = 0.0f;
  for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
      }
    }
  }
  P[outRow][outCol] = Pvalue;
}

__global__ void convolution_2D_const_mem_kernel(float *N, float **F, float *P, int r,
                                                int width, int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  float Pvalue = 0.0f;
  for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
      }
    }
  }
  P[outRow * width + outCol] = Pvalue;
}

#define IN_TILE_DIM 32
#define FILTER_RADIUS 5 //
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_2D_const_mem_kernel(float *N, float *P,
                                                int width, int height)  {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    //loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >=0 && row < height) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && tileRow>=0 
            && tileRow<OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow][fCol]*N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }
}

#define TILE_DIM 32
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N,
                                                             float *P,
                                                             int width, int height)  {
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;
    //loading input tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if(row<height && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    // turning off the threads at the edges of the block
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                if (threadIdx.x-FILTER_RADIUS+fCol < TILE_DIM && 
                    threadIdx.y-FILTER_RADIUS+fRow < TILE_DIM) {
                    Pvalue += F_c[fRow][fCol]*N_s[threadIdx.y-FILTER_RADIUS+fRow][threadIdx.x-FILTER_RADIUS+fCol];
                } else {
                    if (row-FILTER_RADIUS+fRow < height && 
                        row-FILTER_RADIUS+fRow >= 0 && 
                        col-FILTER_RADIUS+fCol >= 0 && 
                        col-FILTER_RADIUS+fCol < width) {
                        Pvalue += F_c[fRow][fCol]*N[(row-FILTER_RADIUS+fRow)*width+col-FILTER_RADIUS+fCol];
                    }
                }
            }
        }
        P[row*width+col] = Pvalue;
    }
}


int main(int argc,char *argv[])
{
	int nx =        (argc>1) ? atoi(argv[1]) : 1024;
	int ny =        (argc>2) ? atoi(argv[2]) : 1024;
	int iter_host = (argc>3) ? atoi(argv[3]) : 1000;
	int iter_gpu =  (argc>4) ? atoi(argv[4]) : 10000;
	uint threadx  = (argc>5) ? atoi(argv[5]) : 16;  // 8, 16, 32 or 64
	uint thready  = (argc>6) ? atoi(argv[6]) : 16;  // product <= 1024

	int size = nx*ny;

	thrust::host_vector<float>     a(size);
	thrust::host_vector<float>     b(size);
	thrust::device_vector<float> dev_a(size);
	thrust::device_vector<float> dev_b(size);

  auto idx = [&nx](int y,int x){ return y*nx+x; };
	for(int y=0;y<ny;y++) a[idx(y,0)] = a[idx(y,nx-1)] = 1.0f;
	// corner adjustment
	a[idx(0,0)] = a[idx(0,nx-1)] = a[idx(ny-1,0)] = a[idx(ny-1,nx-1)] = 0.5f;
	
	dev_a = a;  // copy to both
	dev_b = a;  // dev_a and dev_b 
	b = a;      // and for host_gpu

	cx::timer tim; // apply stencil iter_host times
	for(int k=0;k<iter_host/2;k++){ // ping pong buffers a and b
		stencil2D_host(a.data(),b.data(),nx,ny); // a=>b
		stencil2D_host(b.data(),a.data(),nx,ny); // b=>a
	}
	double t1 = tim.lap_ms();
	double gflops_host  = (double)(iter_host*4)*(double)size/(t1*1000000);
	cx::write_raw("stencil2D_host.raw",a.data(),size);

	dim3 threads ={threadx,thready,1};
	dim3 blocks ={(nx+threads.x-1)/threads.x,(ny+threads.y-1)/threads.y,1};

	tim.reset();  // apply stencil iter_gpu times
	for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
		stencil2D<<<blocks,threads>>>(dev_a.data().get(), dev_b.data().get(), nx, ny); // a=>b
		stencil2D<<<blocks,threads>>>(dev_b.data().get(), dev_a.data().get(), nx, ny); // b=>a
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();

	a = dev_a; 
	//
	// do something with result
	cx::write_raw("stencil2D_gpu.raw",a.data(),size);
	//
	double gflops_gpu = (double)(iter_gpu*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_gpu, t2,gflops_gpu);

	// for logging
	FILE* flog = fopen("stencil4PT_gpu.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_gpu, gflops_gpu);
	fclose(flog);
	flog = fopen("stencil4PT_host.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_host, gflops_host);
	fclose(flog);
	return 0;

}