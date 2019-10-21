#include "BackPropagator.h"

BackPropagator::BackPropagator( int m, int n, float lambda, float backprop_dist ) :M{m}, N{n}
    {
        // Allocate memory for HQ and the image
        cudaMalloc(&Hq, N*M*sizeof(cufftComplex));
        cudaMalloc(&image, N*M*sizeof(cufftComplex));
        cudaMalloc(&image_float, N*M*sizeof(float));

        // Declaring the FFT plan
        cufftPlan2d(&fft_plan, N, M, CUFFT_C2C);

        // Calculating the Hq matrix according to the equations in the original .m file.
        calculate<<<numBlocks, BLOCKSIZE>>>(N, M, backprop_dist, PIXEL_DX, REFRACTION_INDEX, lambda, Hq);
    };

void BackPropagator::backprop(uint16_t* input, uint16_t* output )
{
    // Convert the uint16 image to float image 
    u16ToFloat<<<numBlocks, BLOCKSIZE>>>(M, N, input, image_float);

    // Convert the real input image to complex image
    convertToComplex<<<numBlocks, BLOCKSIZE>>>(N*M, image_float, image);
    
    // Execute forward FFT on the green channel
    cufftExecC2C(fft_plan, image, image, CUFFT_FORWARD);

    // Element-wise multiplication of Hq matrix and the image
	multiplyInPlace<<<numBlocks, BLOCKSIZE>>>(M, N, Hq, image);
    
	// Executing inverse FFT
	cufftExecC2C(fft_plan, image, image, CUFFT_INVERSE);
	// Conversion of result matrix to a real float matrix
	imaginary<<<numBlocks, BLOCKSIZE>>>(M,N, image, image_float);
	// Conversion of result matrix to a real float matrix
	floatToUInt16<<<numBlocks, BLOCKSIZE>>>(M,N, image_float, output);
}

BackPropagator::~BackPropagator() {
    cudaFree(Hq);
    cudaFree(image);
    cudaFree(image_float);
    cufftDestroy(fft_plan);
};