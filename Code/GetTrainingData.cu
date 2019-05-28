/**
 * @author  Viktor-Adam Koropecky
 */

#include "mex.h"
#include "stdio.h"
#include "stdint.h"
#include "cuda.h"
#include "cufft.h"
#include <chrono>
#include "Kernels.h"

// Cutting the input image on CPU by copying partial columns 
void h_cut( int M, int N, int offsets_y, int offsets_x, int dimensions_y, int dimensions_x, uint8_t* input, uint8_t* resize ){
    int count = dimensions_x;
    for(int i = 0; i < count; i+=1){
        memcpy(&resize[i * dimensions_y], &input[offsets_y + M * offsets_x + M * i], sizeof(uint8_t)*dimensions_y);
        memcpy(&resize[i * dimensions_y + dimensions_y*dimensions_x], &input[offsets_y + M * offsets_x + M * i + M*N], sizeof(uint8_t)*dimensions_y);
        memcpy(&resize[i * dimensions_y + dimensions_y*dimensions_x*2], &input[offsets_y + M * offsets_x + M * i + 2*M*N], sizeof(uint8_t)*dimensions_y);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    #define Erz plhs[0]

    #define raw_data prhs[0]
    #define lambda mxGetPr(prhs[1])[0]
    #define n mxGetPr(prhs[2])[0]
    #define dx mxGetPr(prhs[3])[0]
    #define z mxGetPr(prhs[4])[0]
    #define offsts prhs[5]
    #define dims prhs[6]

    int M = (int)mxGetDimensions(raw_data)[0];
    int N = (int)mxGetDimensions(raw_data)[1];
    int Z = (int)mxGetDimensions(raw_data)[2];

    int blockSize = 1024;
    int numBlocks;
    
    uint8_t* input;
    uint8_t* h_input;

    uint16_t* RGB;
    uint16_t* R;
    uint16_t* G;
    uint16_t* B;
    uint16_t* temporary;

    cufftHandle plan;

    cufftComplex* doubleComplexArray;
    cufftComplex* Hq;
    cufftComplex* image;

    float* doubleArray;
    float* outputArray;
    float* smallerOutputArray;
    float* doubleTemporary;

    /* 
        Input data setup. Program checks whether it is necessary to cut the input data and change parameters.
    */
    if(nrhs == 7){
        int* offsets = (int*)mxGetPr(offsts);
        int* dimensions = (int*)mxGetPr(dims);

        offsets[1] = ((offsets[1] % 2) == 1) ? offsets[1]-1 : offsets[1];
        dimensions[1] = ((dimensions[1] % 2) == 1) ? dimensions[1]-1 : dimensions[1];
        
        h_input = (uint8_t*)malloc(dimensions[1]*dimensions[0]*Z*sizeof(uint8_t));

        h_cut(M, N,offsets[0],offsets[1], dimensions[0],dimensions[1], (uint8_t*)mxGetData(raw_data), h_input);

        M = dimensions[0];
        N = dimensions[1];

        cudaMalloc(&input, N*M*Z*sizeof(uint8_t));
        cudaMemcpy(input, h_input, Z*N*M*sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    else{
        cudaMalloc(&input, N*M*Z*sizeof(uint8_t));
        cudaMemcpy(input, (uint8_t*)mxGetData(raw_data), Z*N*M*sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    std::chrono::steady_clock::time_point setup = std::chrono::steady_clock::now();

    /*
        Allocation of memory. We allocate larger arrays so we can call cudaMalloc less.
    */
    cudaMalloc(&RGB, 4*N*M*sizeof(uint16_t));
    R = &RGB[0];
    G = &RGB[N*M];
    B = &RGB[2*N*M];
    temporary = &RGB[3*N*M];

    cudaMalloc(&doubleComplexArray, 2*N*M*sizeof(cufftComplex));
    Hq = &doubleComplexArray[0];
    image = &doubleComplexArray[N*M];

    cudaMalloc(&doubleArray, 2*N*M*sizeof(float));
    doubleTemporary = &doubleArray[0];
    outputArray = &doubleArray[N*M];

    cudaMalloc(&smallerOutputArray, N*M*sizeof(float)/4);

    // Declaration of output matrix.
    Erz = mxCreateNumericMatrix(M,N,mxSINGLE_CLASS,mxREAL); 
    std::chrono::steady_clock::time_point mallocs = std::chrono::steady_clock::now();

    // Declaring appropriate number of cuda Blocks.
    numBlocks = (N*M/2 +blockSize -1)/blockSize;
    // Transforming camera data to a Bayer image.
    bayerize<<<numBlocks,blockSize>>>(M, N, input, temporary);
    std::chrono::steady_clock::time_point bayerization = std::chrono::steady_clock::now();

    // Debayerization of input data
    demosaic<<<numBlocks, blockSize>>>(M,N, temporary, R,G,B);
    std::chrono::steady_clock::time_point demos = std::chrono::steady_clock::now();

    // Transposing the green channel of the RGB picture and converting to a double array.
    // Transposition is necessary for the FFT to work properly
    transposeU16ToDouble<<<numBlocks, blockSize>>>(N, M, G, doubleTemporary);
    std::chrono::steady_clock::time_point transposition = std::chrono::steady_clock::now();

    // Converting the double array to a complex array of cufftDoubleComplex type
    convertToComplex<<<numBlocks, blockSize>>>(N*M, doubleTemporary, image);
    std::chrono::steady_clock::time_point complexization = std::chrono::steady_clock::now();

    // Declaring the FFT plan
    cufftPlan2d(&plan, M,N, CUFFT_C2C);
    // Execute forward FFT on the green channel
    cufftExecC2C(plan, image, image, CUFFT_FORWARD);
    std::chrono::steady_clock::time_point fft = std::chrono::steady_clock::now();

    // Calculating the Hq matrix according to the equations in the original .m file.
    calculate<<<numBlocks, blockSize>>>(M,N, z, dx, n, lambda, Hq);
    std::chrono::steady_clock::time_point calculation = std::chrono::steady_clock::now();

    // Element-wise multiplication of Hq matrix and the image
    transposedMultiplication<<<numBlocks, blockSize>>>(N, M, Hq, image);
    std::chrono::steady_clock::time_point transMult = std::chrono::steady_clock::now();

    // Executing inverse FFT
    cufftExecC2C(plan, image, image, CUFFT_INVERSE);
    // Freeing the memory of FFT plan
    cufftDestroy(plan);
    std::chrono::steady_clock::time_point ifft = std::chrono::steady_clock::now();

    // Conversion of result matrix to a real double matrix
    absoluteValue<<<numBlocks, blockSize>>>(M,N, image, outputArray);
    std::chrono::steady_clock::time_point absolution = std::chrono::steady_clock::now();

    // Copying the memory from the outputArray to the mex Output
    // desample<<<numBlocks, blockSize>>>(M,N, outputArray, smallerOutputArray);
    cudaMemcpy(mxGetPr(Erz), outputArray, N*M*sizeof(float),cudaMemcpyDeviceToHost);
    std::chrono::steady_clock::time_point finalMemcpy = std::chrono::steady_clock::now();
    // Waiting for all the CUDA operations to finish (probably not necessary)
    cudaDeviceSynchronize();

    // Freeing memory
    cudaFree(doubleComplexArray);
    cudaFree(doubleArray);
    cudaFree(input);
    cudaFree(RGB);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


    /*
    * Printing the time measurements
    */
    mexPrintf("Setup took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(setup - start).count());
    mexPrintf("Allocating memory took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(mallocs - setup).count());
    mexPrintf("Bayerizing took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(bayerization - mallocs).count());
    mexPrintf("Demosaic took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(demos - bayerization).count());
    mexPrintf("Transposition of raw data took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(transposition - demos).count());
    mexPrintf("Conversion to complex data took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(complexization - transposition).count());
    mexPrintf("FFT took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(fft - complexization).count());
    mexPrintf("Calculation of Hq took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(calculation - fft).count());
    mexPrintf("Multiplication took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(transMult - calculation).count());
    mexPrintf("Inverse FFT took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(ifft - transMult).count());
    mexPrintf("Absolute Value %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(absolution - ifft).count());
    mexPrintf("Final memcpy took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(finalMemcpy - absolution).count());
    mexPrintf("Final memory emptying took %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - finalMemcpy).count());
    mexPrintf("All together it took %d us\n\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}