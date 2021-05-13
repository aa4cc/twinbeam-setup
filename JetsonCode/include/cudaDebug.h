/**
 * @author  Viktor-Adam Koropecky
 * 
 * Functions provided in this header are useful for debugging CUDA code 
 * either by parsing cannonical errors from cuda functions (gpuAssert)
 * or by testing if it is possible to allocate to device memory
 * 
 */

#ifndef CUDA_DEBUG_H
#define CUDA_DEBUG_H

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cuMemTest() {cudaMemoryTest(__FILE__, __LINE__);}

// This function (and its corresponding pre-compiled definition) was
// taken from https://stackoverflow.com/a/14038590
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

// This function checks for otherwise hard to detect kernel errors
// by trying to allocate device memory
void cudaMemoryTest(const char *file, int line);

#endif