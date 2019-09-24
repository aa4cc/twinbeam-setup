#include <cstdint>
#include "Kernel.h"
#include "Settings.h"
#include "Kernels.h"
#include "cufft.h"
#include "cuda.h"
#include "Definitions.h"

int numBlocksK = (Settings::get_area()/2 +BLOCKSIZE -1)/BLOCKSIZE; 

void Kernel::allocate(){
	cudaMalloc(&kernel, Settings::get_area()*sizeof(cufftComplex));
}

void Kernel::deallocate(){
	cudaFree(kernel);
}

void Kernel::set(cufftComplex* new_kernel){
	kernel = new_kernel;
}

void Kernel::setInPhase(int dim, float* new_kernel){
	kernelToImage<<<numBlocksK, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, dim, new_kernel, kernel);
	cufftHandle plan;
    cufftPlan2d(&plan, STG_HEIGHT, STG_WIDTH, CUFFT_C2C);
    cufftExecC2C(plan, kernel, kernel, CUFFT_FORWARD);
    cufftDestroy(plan);
}

void Kernel::update(cufftComplex* new_kernel){
	multiplyInPlace<<<numBlocksK, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, new_kernel, kernel);
}

void Kernel::updateInPhase(int dim, float* new_kernel){
	cufftComplex* temporaryKernel;
	cudaMalloc(&temporaryKernel, Settings::get_area()*sizeof(cufftComplex));
	kernelToImage<<<numBlocksK, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, dim, new_kernel, temporaryKernel);
	cufftHandle plan;
    cufftPlan2d(&plan, STG_HEIGHT, STG_WIDTH, CUFFT_C2C);
    cufftExecC2C(plan, temporaryKernel, temporaryKernel, CUFFT_FORWARD);
    cufftDestroy(plan);
    multiplyInPlace<<<numBlocksK, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, temporaryKernel, kernel);
    cudaFree(temporaryKernel);
}