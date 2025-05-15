#include <assert.h>
#include <stdio.h>
#define N 1024
__global__ void blockReduceArray(int* array, int* sum) {
    int threadGlobalID = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int blockSum;
    if(threadIdx.x  == 0 ) {
        sum[blockIdx.x] = 0; 
        blockSum = 0;        
    }
    __syncthreads();
    blockSum += array[threadGlobalID];
    __syncthreads();
    if(threadIdx.x  == 0 ) sum[blockIdx.x] = blockSum; 
    return;
}
int main() {
    int globalSum;
    int* sum;
    int* array;
    int numBlocks = 4;
    cudaMallocManaged(&array, N*sizeof(int));
    cudaMallocManaged(&sum, numBlocks*sizeof(int));
    for(int i=0; i<N; i++) array[i] = 1; 
    blockReduceArray<<<numBlocks, N/numBlocks>>>(array, sum);
    cudaDeviceSynchronize();
    globalSum = 0;
    for(int i=0; i<numBlocks; i++) globalSum += sum[i];
    printf("After kernel - global sum = %d\n", globalSum);
    cudaFree(sum);
    cudaFree(array);
    exit(0);
} 