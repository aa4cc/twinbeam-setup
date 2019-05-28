#include "mex.h"
#include "stdio.h"
#include <math.h>
#include <string.h>
#include <cufft.h>
#include <cuda.h>
#include <chrono>
#include "Kernels.h"

#define E 2.71828182845904523536
#define PI 3.14159265358979323846
#define SQUARE(x) x*x

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	#define out plhs[0]
	#define img prhs[0]
	#define z mxGetPr(prhs[1])[0]				//2500e-6
	#define dx mxGetPr(prhs[2])[0]			//1.85e-6
	#define n mxGetPr(prhs[3])[0]				
	#define lambda mxGetPr(prhs[4])[0]
	#define M mxGetM(img)
	#define N mxGetN(img)
    
    cufftDoubleComplex *Hq;              //This array will be used as the main storage array in CUDA kernels
    cufftHandle plan;
    cufftDoubleComplex* image;
    double* outputArray;
    double * temp; 
    out = mxCreateDoubleMatrix(M,N, mxREAL);

    //first block for CUDA
    int blockSize = 1024;
    int numBlocks = (N*M +blockSize -1)/blockSize;

    cudaMalloc(&Hq, N*M*sizeof(cufftDoubleComplex));
    cudaMalloc(&temp, sizeof(double)*N*M);
    cudaMalloc(&image, sizeof(cufftDoubleComplex)*N*M);
    cudaMalloc(&outputArray, N*M*sizeof(double));

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    cudaMemcpy(outputArray, mxGetPr(img), N*M*sizeof(double), cudaMemcpyHostToDevice);
    std::chrono::steady_clock::time_point memcpy1 = std::chrono::steady_clock::now();

    //First step - Calculating Hq matrix - stored in Hq
	calculate<<<numBlocks, blockSize>>>(M,N, z, dx, n, lambda, Hq);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point calculation = std::chrono::steady_clock::now();

    //Transposing raw data image
    transpose<<<numBlocks, blockSize>>>(N, M, outputArray, temp);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point transposition = std::chrono::steady_clock::now();

    //Converting image data to a CUFFT-friendly complex array
    convertToComplex<<<numBlocks, blockSize>>>(N*M, temp, image);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point conversion1 = std::chrono::steady_clock::now();

    //FFT from image data
    cufftPlan2d(&plan, M,N , CUFFT_Z2Z );
    cufftExecZ2Z(plan, image, image, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point fft1 = std::chrono::steady_clock::now();

    //Multiplying the meshgrid with fft
    transposedMultiplication<<<numBlocks, blockSize>>>(N, M, Hq, image);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point multiplication = std::chrono::steady_clock::now();

    //Making an inverse fft from the product
    cufftExecZ2Z(plan, image, image, CUFFT_INVERSE);
    cufftDestroy(plan);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point ifft = std::chrono::steady_clock::now();

    //finilizing by calculating the absolute value of the inverse fft
    absoluteValue<<<numBlocks, blockSize>>>(M,N, image, outputArray);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point absoluteValue = std::chrono::steady_clock::now();

    //Copying the result to the output matrix
    cudaMemcpy(mxGetPr(out), outputArray, N*M*sizeof(double),cudaMemcpyDeviceToHost);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    mexPrintf("First memcpy took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(memcpy1 - start).count());
    mexPrintf("Calculation of Hq took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(calculation - memcpy1).count());
    mexPrintf("Transposition of raw data took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(transposition - calculation).count());
    mexPrintf("Conversion to complex data took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(conversion1 - transposition).count());
    mexPrintf("FFT took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(fft1 - conversion1).count());
    mexPrintf("Multiplication took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(multiplication - fft1).count());
    mexPrintf("Inverse FFT took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(ifft - multiplication).count());
    mexPrintf("Absolute Value %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(absoluteValue - ifft).count());
    mexPrintf("Final memcpy took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - absoluteValue).count());
    mexPrintf("All together it took %d us\n\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    cudaFree(outputArray);   
    cudaFree(image);
    cudaFree(Hq);
    cudaFree(temp);
	return;
}