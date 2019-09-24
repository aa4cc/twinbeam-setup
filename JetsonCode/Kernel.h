#ifndef KERNEL_H
#define KERNEL_H

#include "cufft.h"
#include "cuda.h"

class Kernel{
private:
public:
	cufftComplex* kernel;
	void allocate();
	void update(cufftComplex* new_kernel);
	void updateInPhase(int dim, float* new_kernel);
	void deallocate();
	void set(cufftComplex* new_kernel);
	void setInPhase(int dim, float* new_kernel);
};

#endif