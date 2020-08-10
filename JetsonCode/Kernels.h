/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "stdint.h"
#include "cuda.h"
#include "cufft.h"

__global__ void calculateBackPropMatrix(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq);

__global__ void multiplyInPlace(int N, int M, cufftComplex*  input, cufftComplex*  output);
__global__ void multiply(int N, int M, cufftComplex*  input, cufftComplex*  kernel, cufftComplex* output);

__global__ void real(int M, int N, cufftComplex* input, float* output);
__global__ void imaginary(int M, int N, cufftComplex* input, float* output);
__global__ void absoluteValue(int N, int M, cufftComplex* storageArray, float* outputArray);

__global__ void convertToComplex(int count , float* real, cufftComplex* complex);
__global__ void u8ToFloat(int N, int M, uint8_t* input, float* result);
__global__ void u16ToFloat(int N, int M, uint16_t* input, float* result);
__global__ void floatToUInt8(int N, int M, float* input, uint8_t* result);
__global__ void floatToUInt16(int N, int M, float* input, uint16_t* result);

__global__ void getLocalMinima(int M, int N, float* input, uint16_t* points, uint32_t pointsMaxSize, uint32_t* pointsCounter, float thrs);
__global__ void imDivide(int M, int N, uint8_t* in1, uint8_t* in2, float* output);

#endif
