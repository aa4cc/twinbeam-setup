/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */
 
#include "BackPropagator.h"

BackPropagator::BackPropagator( int m, int n, float lambda, float backprop_dist ) :M{m}, N{n}
    {
        // Allocate memory for HQ and the image
        cudaMalloc(&Hq, N*M*sizeof(cufftComplex));
        cudaMalloc(&image, N*M*sizeof(cufftComplex));
        cudaMalloc(&image_float, N*M*sizeof(float));

        // Declaring the FFT plan
        cufftPlan2d(&fft_plan, N, M, CUFFT_C2C);

        numBlocks = (m*n/2 + BLOCKSIZE -1)/BLOCKSIZE;

        // Calculating the Hq matrix according to the equations in the original .m file.
        calculateBackPropMatrix<<<numBlocks, BLOCKSIZE>>>(N, M, backprop_dist, PIXEL_DX, REFRACTION_INDEX, lambda, Hq);
    };

void BackPropagator::backprop(ImageData<uint8_t>& input, ImageData<uint8_t>& output)
{
    // Convert the uint8 image to float image 
    input.mtx.lock();
    u8ToFloat<<<numBlocks, BLOCKSIZE>>>(M, N, input.devicePtr(), image_float);
    input.mtx.unlock();

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
    output.mtx.lock();
	floatToUInt8<<<numBlocks, BLOCKSIZE>>>(M,N, image_float, output.devicePtr());
    output.mtx.unlock();
}

BackPropagator::~BackPropagator() {
    cudaFree(Hq);
    cudaFree(image);
    cudaFree(image_float);
    cufftDestroy(fft_plan);
};