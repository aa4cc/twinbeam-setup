#ifndef PHASE_KERNELS_H
#define PHASE_KERNELS_H

#include "cuda.h"
#include "cufft.h"
#include "Definitions.h"

#define SQUARE(x) x*x

__global__ void propagator(int N, int M, double z, double dx, double n, double lambda, cufftComplex* Hq);
__global__ void multiply(int count, cufftComplex* in, cufftComplex* out);
__global__ void multiplyfc(int count, double* in, cufftDoubleComplex* out);
__global__ void multiplyf(int count, double* in1, double* in2, double* out);
__global__ void absolute(int count, cufftDoubleComplex* in, double* out);
__global__ void real(int count, cufftDoubleComplex* in, double* out);
__global__ void imag(int count, cufftDoubleComplex* in, double* out);
__global__ void angle(int count, cufftDoubleComplex* in, double* out);
__global__ void modelFunc(int count, double rOffset, double iOffset, cufftDoubleComplex* in, cufftDoubleComplex* model, double* Imodel);
__global__ void conjugate(int count, cufftComplex *in, cufftComplex* out);
__global__ void simpleDivision(double* num, double* div, double* res);
__global__ void linear(int count, double* coef, double* constant, double* in, double* out, bool sign);
__global__ void square(int count, double* in, double* out);
__global__ void simpleSum(double* in1, double* in2, double* out);
__global__ void add(int count, cufftDoubleComplex* in1, cufftDoubleComplex* in2, cufftDoubleComplex* out, bool sign);
__global__ void strictBounds(int count, cufftDoubleComplex* arr, double r_min, double r_max, double i_min, double i_max);
__global__ void positivityBounds(int count, cufftDoubleComplex* arr)
__global__ void strictBoundsf(int count, cufftDoubleComplex* arr, double r_min, double r_max);
__global__ void softBounds(int count, cufftDoubleComplex* arr, double mu, double t);
__global__ void rowConvolution(int N, int M, double diameter, double* kernel, double* image, double* output, bool horizontal);
__global__ void offset(int count, double roff, double ioff, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void offsetf(int count, double roff, double* in, double* out, bool sign);
__global__ void extend(int count, int multiple, cufftDoubleComplex* in, cufftDoubleComplex* out);
//TODO: Not very robust - expects 4 MP to 1 MP exactly
__global__ void shrinkTo1MP(int N, int M, double* in, double* out);

// Type conversion kernels
__global__ void C2Z(int count, cufftComplex* in, cufftDoubleComplex* out);
__global__ void Z2C(int count, cufftDoubleComplex* in, cufftComplex* out);
__global__ void D2u8(int count, double* in, uint8_t* out);
__global__ void U82D(int count, uint8_t* in, double* out);
__global__ void F2C(int count, double* in, cufftDoubleComplex* out);

// kernels for division by constant
//
__global__ void contractf_p(int count, double *constant, double* in, double* out);
__global__ void contractf(int count, double constant, double* in, double* out);

// kernels for multiplication by constant
//
__global__ void scalef(int count, double constant, double* in, double* out);
__global__ void scale(int count, cufftDoubleComplex constant, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void scale_p(int count, cufftDoubleComplex* constant, cufftDoubleComplex* in, cufftDoubleComplex* out);

// kernels with parallel reduction
__global__ void maximum(int count, double* in, double* result);
__global__ void minimum(int count, double* in, double* result);
__global__ void sum(int count, double* in, double* result);
__global__ void sumOfProducts(int count, double* in1, double* in2, double* result);

// wrappers for parallel reduction kernels
void h_maximum(int count, double* d_in, double* d_result, cudaStream_t stream);
void h_minimum(int count, double* d_in, double* d_result, cudaStream_t stream);
void h_sum(int count, double* d_in, double* d_result, cudaStream_t stream);
void h_sumOfProducts(int count, double* d_in1, double* d_in2, double* result, cudaStream_t stream);
void h_average(int count, double* d_in, double* d_result, cudaStream_t stream);

#endif