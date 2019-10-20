#ifndef BACKPROPAGATOR_H
#define BACKPROPAGATOR_H

#include "Definitions.h"
#include "cuda.h"
#include "cufft.h"

class BackPropagator
{
private:
    int numBlocks = 1024;
    int M, N;
    cufftComplex* Hq, image;
    cufftHandle fft_plan;

    __global__ void calculate_hq(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq);

public:
    BackPropagator( int m, int n, float lambda, float backprop_dist ) :M{m}, N{n}
    {
        // Allocate memory for HQ and the image
        cudaMalloc(&Hq, N*M*sizeof(cufftComplex));
        cudaMalloc(&image, N*M*sizeof(cufftComplex));

        // Declaring the FFT plan
        cufftPlan2d(&fft_plan, N, M, CUFFT_C2C);

        // Calculating the Hq matrix according to the equations in the original .m file.
        calculate<<<numBlocks, BLOCKSIZE>>>(N, M, backprop_dist, PIXEL_DX, REFRACTION_INDEX, lambda, Hq);
    };

    void backprop(float* input, float* output );

    ~BackPropagator() {
        cudaFree(Hq);
        cudaFree(image);
        cufftDestroy(plan);
    };
};

BackPropagator::backprop(float* input, float* output )
{
    // Convert the real input image to complex image
    convertToComplex<<<numBlocks, BLOCKSIZE>>>(N*M, input, image);
    
    // Execute forward FFT on the green channel
    cufftExecC2C(plan, image, image, CUFFT_FORWARD);

    // Element-wise multiplication of Hq matrix and the image
	multiplyInPlace<<<numBlocks, BLOCKSIZE>>>(M, N, Hq, image);
    
	// Executing inverse FFT
	cufftExecC2C(plan, image, image, CUFFT_INVERSE);
	// Conversion of result matrix to a real double matrix
	imaginary<<<numBlocks, BLOCKSIZE>>>(M,N, image, output);
}

#endif