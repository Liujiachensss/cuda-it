# 使用compute-sanitizer
注意，下面的例子都加了编译选项--lineinfo,以显示行号。

## 例子1：memcheck检查内存写入越界
### 直接写
```sh
$ compute-sanitizer /home/liujch/learning/CUDA/cuda-it/build/test/cs-writeValidMem 
========= COMPUTE-SANITIZER
Before: Array 0, 1 .. N-1: 1.000000 1.000000 1.000000
========= Invalid __global__ read of size 4 bytes
=========     at scaleArray(float *, float)+0x70 in writeValidMem.cu:6
=========     by thread (255,0,0) in block (3,0,0)
=========     Address 0x74b834000ffc is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x74b834000000 of size 4,092 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x8ae3] in cs-writeValidMem
========= 
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: main [0x8a94] in cs-writeValidMem
========= 
After : Array 0, 1 .. N-1: 3.000000 3.000000 1.000000
========= ERROR SUMMARY: 2 errors
```

### 用memcheck
```sh
$ compute-sanitizer --tool memcheck --leak-check=full /home/liujch/learning/CUDA/cuda-it/build/test/cs-writeValidMem
========= COMPUTE-SANITIZER
Before: Array 0, 1 .. N-1: 1.000000 1.000000 1.000000
========= Invalid __global__ read of size 4 bytes
=========     at scaleArray(float *, float)+0x70 in writeValidMem.cu:6
=========     by thread (255,0,0) in block (3,0,0)
=========     Address 0x7b95b4000ffc is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x7b95b4000000 of size 4,092 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x8ae3] in cs-writeValidMem
========= 
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: main [0x8a94] in cs-writeValidMem
========= 
After : Array 0, 1 .. N-1: 1.000000 1.000000 1.000000
========= Leaked 4,092 bytes at 0x7b95b4000000
=========     Saved host backtrace up to driver entry point at allocation time
=========         Host Frame: cudaMallocManaged [0x527bb] in cs-writeValidMem
=========         Host Frame: main [0x89bb] in cs-writeValidMem
========= 
========= LEAK SUMMARY: 4092 bytes leaked in 1 allocations
========= ERROR SUMMARY: 3 errors

```

## 例子2：racecheck检查数据竞争
### 直接写——没用
```sh
$ compute-sanitizer /home/liujch/learning/CUDA/cuda-it/build/test/cs-racecheck
========= COMPUTE-SANITIZER
After kernel - global sum = 4
========= ERROR SUMMARY: 0 errors
```

### 用racecheck
```sh
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at blockReduceArray(int *, int *)+0xf0 in racecheck.cu:12
=========     and Read access at blockReduceArray(int *, int *)+0xd0 in racecheck.cu:12 [32 hazards]
=========     and Write access at blockReduceArray(int *, int *)+0xf0 in racecheck.cu:12 [4 hazards]
========= 
After kernel - global sum = 8
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
'''