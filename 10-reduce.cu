#include <cuda_runtime.h>

// 10.6
__global__ void SimpleSumReductionKernel(float* input, float* output) {
    uint i = 2 * threadIdx.x;
    for (uint stride = 1; stride <= blockDim.x; stride *= 2){
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

// 10.9
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    uint i = 2 * threadIdx.x;
    for (uint stride = blockDim.x; stride >= 1; stride /= 2){
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

// 10.11
__global__ void SharedMemorySumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    uint t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

// 10.13
__global__ void SegmentedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    uint segment = 2 * blockDim.x * blockIdx.x;
    uint i = segment + threadIdx.x;
    uint t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride) {
            input_s[i] += input_s[i + stride];
        }
    }
    if(threadIdx.x == 0){
        atomicAdd(output, intput_s[0]);
    }
}

// 10.15
__global__ void ReduceSum(float* input, float* output, uint n) {
   __shared__ float input_s[BLOCK_DIM];
    uint segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    uint i = segment + threadIdx.x;
    uint t = threadIdx.x;
    float sum = input[i];
    for (uint tile = 1; tile < 2 * COARSE_FACTOR; tile++) {
        sum += input[i + tile * BLOCK_DIM];
    }
    input_s[t] = sum;
    for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride) {
            input_s[i] += input_s[i + stride];
        }
    }
    if(threadIdx.x == 0){
        atomicAdd(output, input_s[0]);
    }
}