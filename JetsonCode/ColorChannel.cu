#include "ColorChannel.h"
#include "Settings.h"
#include "Definitions.h"
#include "cuda.h"
#include "cufft.h"
#include "Kernels.h"

int numBlocks = (Settings::get_area()/2 +BLOCKSIZE -1)/BLOCKSIZE; 

void ColorChannel::initialize(bool d, int z, float l){
	display = d;
	z = zi;
	lambda = l;
}

void ColorChannel::allocate(){
	cudaMalloc(&original, Settings::get_area()*sizeof(uint16_t));
	cudaMalloc(&doubleOriginal, Settings::get_area()*sizeof(float));
	cudaMalloc(&maxima, Settings::get_area()*sizeof(float));
	cudaMalloc(&hq, Settings::get_area()*sizeof(cufftComplex));
	cudaMalloc(&convoluted, Settings::get_area()*sizeof(cufftComplex));
	calculateHq();

	if(display){
		cudaMalloc(&backpropagated, Settings::get_area()*sizeof(float));
	}
}

void ColorChannel::deallocate(){
	cudaFree(original);
	cudaFree(doubleOriginal);
	cudaFree(maxima);
	cudaFree(hq);
	cudaFree(convoluted);

	if(display){
		cudaFree(backpropagated);
	}
}

void ColorChannel::calculateHq(){
	calculate<<<numBlocks, BLOCKSIZE>>>(STG_HEIGHT, STG_WIDTH, z, PIXEL_DX, REFRACTION_INDEX, lambda, hq);
}

void ColorChannel::typeCast(){
	u16ToDouble<<<numBlocks, BLOCKSIZE>>>(STG_HEIGHT, STG_WIDTH, original, doubleOriginal);
}

void ColorChannel::backpropagate(cufftComplex* kernel){
	cufftComplex* image;
    cufftComplex* convolutedImage;
    float* filterOutput;
    float* extremes;
    cufftHandle t_plan;

    cudaMalloc(&filterOutput, Settings::get_area()*sizeof(float));
    cudaMalloc(&extremes, Settings::get_area()*sizeof(float));
    cudaMalloc(&image, Settings::get_area()*sizeof(cufftComplex));
    cudaMalloc(&convolutedImage, 2*sizeof(cufftComplex));

    convertToComplex<<<numBlocks, BLOCKSIZE>>>(Settings::get_area(), doubleOriginal, image);
    cufftPlan2d(&plan, STG_HEIGHT, STG_WIDTH, CUFFT_C2C);
    cufftExecC2C(plan, image, image, CUFFT_FORWARD);
	multiplyInPlace<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, hq, image);
	multiply<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, image, kernel, convolutedImage);
	
	if(display){
		cufftExecC2C(plan, image, image, CUFFT_INVERSE);
		absoluteValue<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, image, backpropagated);
		findExtremes<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, backpropagated, extremes);
		normalize<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, backpropagated, extremes);
	}

	//Current version of object position detection, needs to be updated

	cufftExecC2C(plan, convolutedImage, convolutedImage, CUFFT_INVERSE);
	cutAndConvert<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, convolutedImage, maxima);

    cudaFree(extremes);
    cudaMalloc(&extremes, sizeof(float)*2);

	findExtremes<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, filterOutput, extremes);
	normalize<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, filterOutput, extremes);
	getLocalMaxima<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, filterOutput, maxima);

	cufftDestroy(plan);
    cudaFree(extremes);
    cudaFree(convolutedImage);
    cudaFree(image);
    cudaFree(filterOutput);


}
