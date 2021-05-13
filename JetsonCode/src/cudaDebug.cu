#include "cudaDebug.h"
#include "stdio.h"
#include "string.h"
#include "cuda.h"
#include <cuda_runtime_api.h>

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void cudaMemoryTest(const char *file, int line)
{
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    gpuAssert(cudaMalloc(&d_a, bytes), file, line);

    memset(h_a, 0, bytes);
    gpuAssert(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), file, line);
    gpuAssert(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost), file, line);
}