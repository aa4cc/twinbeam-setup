#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

#define CONVO_DIM_GREEN 60
#define SQUARE(x) x*x

__global__ void testerFunction(int* input){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    	int stride = blockDim.x * gridDim.x;
	int count = 20000;
	for(int i = index; i < count; i += stride){
		input[i] = i;	
	}
}

__global__ void generateConvoMaskGreen(int m, int n, float* convoMask){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = m*n;
            double temp;
            for(int i = index; i < count; i += stride){
                temp = sqrt((double)(SQUARE((double)((i%m) - (double)(m/2))) + SQUARE((double)((i/m) - (double)(m/2)))));
                convoMask[i] = 0;
                if( temp <= 20 ){
                    convoMask[i] = 1;
                }
                /*else if(temp > 36 && temp <= 38){
                    convoMask[i] = 8;
                }*/
            }
}

__global__ void kernelToImage(int M, int N, int kernelDim, float* kernel, float* outputKernel){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = kernelDim*kernelDim;
            int center = (kernelDim-1)/2;
            for(int i = index; i < count; i += stride){
                int lineNum = i/kernelDim;
                int colNum = i%kernelDim;
                if(colNum >= center && lineNum >= center)
                    outputKernel[colNum-center + (lineNum-center)*M] = kernel[i];
                else if(colNum >= center)
                    outputKernel[colNum-center + (N - center + lineNum)*M] = kernel[i];
                else if(lineNum >= center)
                    outputKernel[colNum + M - center + (lineNum-center)*M] = kernel[i];
                else
                    outputKernel[colNum + M - center + (N - center + lineNum)*M] = kernel[i];
            }
    }

int main(){
	cv::namedWindow("Basic Visualization");
	float* input;
	float* output;
    float* h_output= (float*)malloc(sizeof(float)*100*100);
<<<<<<< HEAD
    cudaMalloc(&input, (int)(60*60*sizeof(float)));
    cudaMalloc(&output, (int)(100*100*sizeof(float)));
=======
    cudaMalloc(&input, (int)60*60*sizeof(float));
    cudaMalloc(&output, (int)100*100*sizeof(float));
>>>>>>> 006307a94f25caaf10f5bc2ba533855d06155eb5
    generateConvoMaskGreen<<<(100 + 1024 - 1)/1024, 1024>>>(60,60,input);
    kernelToImage<<<(100 + 1024 - 1)/1024, 1024>>>(100,100,60,input, output);
    cudaMemcpy(h_output, output, sizeof(float)*100*100, cudaMemcpyDeviceToHost);

	const cv::Mat img(cv::Size(100, 100), CV_32F, h_output);
	cv::imshow("Basic Visualization", img);
	cv::waitKey(0);
	free(h_output);
	cudaFree(input);
    cudaFree(output);
}

