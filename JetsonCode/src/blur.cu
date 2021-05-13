#include "blur.h"
#include "cuda.h"
#include <cuda_runtime_api.h>
#include "math.h"
#include "kernels.h"
#include "stdio.h"

#define SQUARE(x) x*x

Blur::Blur():initialized(false){}

Blur::Blur(unsigned int diameter, double sigma) : diameter(diameter), initialized(true){
    h_rowKernel = (double*)malloc(diameter*sizeof(double));
    rowGaussianFilter(diameter, sigma, h_rowKernel);
    cudaMalloc(&d_rowKernel, diameter*sizeof(double));
    cudaMemcpy(d_rowKernel, h_rowKernel, diameter*sizeof(double), cudaMemcpyHostToDevice);
}

Blur::Blur(unsigned int diameter) : diameter(diameter), initialized(true){
    h_rowKernel = (double*)malloc(diameter*sizeof(double));
    for(int i  = 0; i < diameter; i++){
        h_rowKernel[i] = 1.0f/(double)diameter;
    }
    cudaMalloc(&d_rowKernel, diameter*sizeof(double));
    cudaMemcpy(d_rowKernel, h_rowKernel, diameter*sizeof(double), cudaMemcpyHostToDevice);
}

void Blur::rowGaussianFilter(unsigned int diameter, double sigma, double *ret){
    int r = diameter/2;
    double sum = 0;
    for(int i = 0 ; i < diameter; i++){
        ret[i] = (1/sqrt(2*M_PI*SQUARE(sigma)))*exp(-SQUARE((double)(i-r))/(2*SQUARE(sigma)));
        sum += ret[i];
    }
    for(int i = 0 ; i < diameter; i++){
        ret[i] /= sum;
    }
}

void Blur::gaussianBlur(int N, int M, unsigned int diameter, double sigma, double* in, double* temp, double* out){
    h_rowKernel = (double*)malloc(diameter*sizeof(double));
    rowGaussianFilter(diameter, sigma, h_rowKernel);
    cudaMalloc(&d_rowKernel, diameter*sizeof(double));
    cudaMemcpy(d_rowKernel, h_rowKernel, diameter*sizeof(double), cudaMemcpyHostToDevice);
    rowConvolution<<<N_BLOCKS, N_THREADS>>>(N, M, diameter, d_rowKernel, in, temp, true);
    rowConvolution<<<N_BLOCKS, N_THREADS>>>(N, M, diameter, d_rowKernel, temp, out, false);
    free(h_rowKernel);
    cudaFree(d_rowKernel);
}

void Blur::printKernel(){
    if(!initialized)
        printf("The filter was not set at declaration.\n");
    else{
        for (int i = 0; i < diameter; i++)
            printf("%f, ", h_rowKernel[i]);
        printf("\n");
    }
}

void Blur::boxBlur(int N, int M, unsigned int diameter, double* in, double* temp, double* out){
    h_rowKernel = (double*)malloc(diameter*sizeof(double));
    for(int i  = 0; i < diameter; i++){
        h_rowKernel[i] = 1.0f/(double)diameter;
    }
    cudaMalloc(&d_rowKernel, diameter*sizeof(double));
    cudaMemcpy(d_rowKernel, h_rowKernel, diameter*sizeof(double), cudaMemcpyHostToDevice);
    rowConvolution<<<N_BLOCKS, N_THREADS>>>(N, M, diameter, d_rowKernel, in, temp, true);
    rowConvolution<<<N_BLOCKS, N_THREADS>>>(N, M, diameter, d_rowKernel, temp, out, false);
    free(h_rowKernel);
    cudaFree(d_rowKernel);
}

void Blur::blur(int N, int M, double* in, double* temp, double* out){
    rowConvolution<<<N_BLOCKS, N_THREADS>>>(N, M, diameter, d_rowKernel, in, temp, true);
    rowConvolution<<<N_BLOCKS, N_THREADS>>>(N, M, diameter, d_rowKernel, temp, out, false);
}

Blur::~Blur(){
    if(initialized){
        free(h_rowKernel);
        cudaFree(d_rowKernel);
    }
}