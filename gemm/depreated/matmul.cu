#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "mm-cublas.cuh"
#include "mm-naive.cuh"

int main(int argc, char** argv) {
    int M = 4092;
    int N = 4092;
    int K = 4092;
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    thrust::host_vector<float> h_A(size_A);
    thrust::host_vector<float> h_B(size_B);
    thrust::host_vector<float> h_C(size_C);
    thrust::device_vector<float> d_A(size_A);
    thrust::device_vector<float> d_B(size_B);
    thrust::device_vector<float> d_C(size_C);
    

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    #ifdef NAIVE_GEMM
    mm_naive<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V1_GEMM)
    mm_v1<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V2_GEMM)
    mm_v2<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V3_GEMM)
    mm_v3<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V4_GEMM)
    mm_v4<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V5_GEMM)
    mm_v5<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V6_GEMM)
    mm_v6<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #elif defined(V7_GEMM)
    mm_v7<<<grid, block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    #else
    mm_cublas(handle, M, N, K, 1.0f, d_A.data().get(), d_B.data().get(), 0.0f, d_C.data().get());
    #endif

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    std::string op_name{};
    #ifdef NAIVE_GEMM
    op_name = "naive";
    #elif defined(V1_GEMM)
    op_name = "v1";
    #elif defined(V2_GEMM)
    op_name = "v2";
    #elif defined(V3_GEMM)      
    op_name = "v3";
    #elif defined(V4_GEMM)
    op_name = "v4";
    #elif defined(V5_GEMM)
    op_name = "v5";
    #elif defined(V6_GEMM)
    op_name = "v6";
    #elif defined(V7_GEMM)
    op_name = "v7";
    #else
    op_name = "cublas";    
    #endif
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time: %f ms with %s\n", time, op_name.c_str());
}