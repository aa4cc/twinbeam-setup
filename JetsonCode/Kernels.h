/**
 * @author  Viktor-Adam Koropecky
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "cufft.h"

__global__ void calculate(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq);
__global__ void multiplyInPlace(int N, int M, cufftComplex*  input, cufftComplex*  output);
__global__ void multiply(int N, int M, cufftComplex*  input, cufftComplex*  kernel, cufftComplex* output);
__global__ void absoluteValue(int N, int M, cufftComplex* storageArray, float* outputArray);
__global__ void cutAndConvert(int N, int M, cufftComplex* input, float* output);
__global__ void convertToFloat(int count , float* output, cufftComplex* input);
__global__ void transpose(int N, int M, float* transposee, float* result);
__global__ void u16ToDouble(int N, int M, uint16_t* transposee, float* result);
__global__ void convertToComplex(int count , float* real, cufftComplex* complex);
__global__ void desample(int M, int N, float* input, float* output);
__global__ void generateConvoMaskRed(int m, int n, float* convoMask);
__global__ void generateConvoMaskGreen(int m, int n, float* convoMask);
__global__ void sobelDerivation(int M, int N, float* input, float* output);
__global__ void findExtremes(int M, int N, float* input, float* extremes);
__global__ void findMaxima(int M, int N, float* input, float* maxima);
__global__ void normalize(int M, int N, float* input, float* extremes); // input is also output
__global__ void getLocalMaxima(int M, int N, float* input, float* output);
__global__ void kernelToImage(int M, int N, int kernelDim, float* kernel, cufftComplex* outputKernel);
__global__ void findPoints(int M, int N, float* input, int* output);
__global__ void generateBlurFilter(int M, int N, int margin, cufftComplex* filter);
__global__ void blurFilter(int M, int N, int margin, cufftComplex* input);
__global__ void real(int M, int N, cufftComplex* input, float* output);
__global__ void imaginary(int M, int N, cufftComplex* input, float* output);

#endif
