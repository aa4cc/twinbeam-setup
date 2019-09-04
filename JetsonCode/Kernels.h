/**
 * @author  Viktor-Adam Koropecky
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "cufft.h"

#define E 2.71828182
#define PI 3.14159265358979323846
#define SQUARE(x) x*x
#define eighth_power(X) X*X*X*X*X*X*X*X 
#define CONVO_DIM_RED 60
#define CONVO_DIM_GREEN 160


__global__ void calculate(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq);
__global__ void elMultiplication(int N, int M, cufftComplex*  Hq, cufftComplex*  Bq);
__global__ void elMultiplication2(int N, int M, cufftComplex*  input, cufftComplex*  kernel, cufftComplex* output);
__global__ void absoluteValue(int N, int M, cufftComplex* storageArray, float* outputArray);
__global__ void cutAndConvert(int N, int M, cufftComplex* input, float* output);
__global__ void convertToFloat(int count , float* output, cufftComplex* input);
__global__ void transpose(int N, int M, float* transposee, float* result);
__global__ void u16ToDouble(int N, int M, uint16_t* transposee, float* result);
__global__ void convertToComplex(int count , float* real, cufftComplex* complex);
__global__ void bayerize(int M, int N, uint8_t* input, uint16_t* output);
__global__ void demosaic(int M, int N, uint16_t* input, uint16_t* R, uint16_t* G, uint16_t* B);
__global__ void desample(int M, int N, float* input, float* output);
__global__ void generateConvoMaskRed(int m, int n, float* convoMask);
__global__ void generateConvoMaskGreen(int m, int n, float* convoMask);
__global__ void sobelDerivation(int M, int N, float* input, float* output);
__global__ void findExtremes(int M, int N, float* input, float* extremes);
__global__ void normalize(int M, int N, float* input, float* extremes); // input is also output
__global__ void getLocalMaxima(int M, int N, float* input, float* output);
__global__ void kernelToImage(int M, int N, int kernelDim, float* kernel, cufftComplex* outputKernel);
__global__ void findPoints(int M, int N, float* input, int* output);
__global__ void generateBlurFilter(int M, int N, cufftComplex* filter);
__global__ void blurFilter(int M, int N, int margin, cufftComplex* input);

#endif
