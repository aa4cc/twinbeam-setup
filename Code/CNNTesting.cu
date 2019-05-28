/**
 * @author  Viktor-Adam Koropecky
 */

#include "stdint.h"
#include "cuda.h"
#include "cufft.h"
#include "stdio.h"
#include "Kernels.h"

/*
    Calculation of the Hq matrix according to the equations in original .m file
*/
__global__ void calculate(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            float FX, FY, temp, res;
            float pre = n/lambda;
            float calc = 1/dx;
            int newIndex;
            int count = N*M;
            for (int i = index; i < count; i += stride)
            {
                newIndex = (i + count/2-1) % (count);
                FX = ((float)(1+(i/M)) * calc/(float)(N)) - calc/2.0f;
                FY = ((float)(1+(i%M)) * calc/(float)(M)) - calc/2.0f;
                res = 2 * PI*z*pre * sqrt(1 - SQUARE(FX/pre) - SQUARE(FY/pre));
                temp = (sqrt(SQUARE(FX) + SQUARE(FY)) < (pre));
	            Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuComplex(cos(res) * temp, sin(res) * temp);
            }
        }

/*
    Element-wise multiplication of two (already transposed) matrices.
*/
__global__ void transposedMultiplication(int N, int M, cufftComplex*  Hq, cufftComplex*  Bq){
            cufftComplex temp;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                temp = make_cuFloatComplex(Bq[i].x/(float)(N*M), Bq[i].y/(float)(N*M));
                Bq[i] = cuCmulf(Hq[i], temp);
            }
        }

/*
    Converting a complex cufftComplex array to a float array of its absolute values
*/
__global__ void absoluteValue(int N, int M, cufftComplex* storageArray, float* outputArray){
            cufftComplex temp;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                temp = make_cuFloatComplex(storageArray[(i%N)*M + i/N].x, storageArray[i%N*M + i/N].y);
                outputArray[i] = cuCabsf(temp);
            }
        }

/*
    Transposition of a float matrix
*/
__global__ void transpose(int N, int M, float* transposee, float* result){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                result[i] = transposee[(i%N)*M + i/N];
            }
        }

/*
    Transposition of a matrix combined with conversion from uint8_t to float
*/
__global__ void transposeU16ToDouble(int N, int M, uint16_t* transposee, float* result){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                result[i] = (float)transposee[(i%N)*M + i/N];
            }
        }

/*
    Converting a real mxArray to complex cufftComplex array
*/
__global__ void convertToComplex(int count , float* real, cufftComplex* complex){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < count; i += stride){
                complex[i] = make_cuComplex(real[i], 0);
            }
        }

/*
    Bayerization function to transform camera data to a Bayer picture
*/
__global__ void bayerize(int M, int N, uint8_t* input, uint16_t* output){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = N*M;
            int temp;
            for(int i = index; i < count; i += stride){
                temp = ((i/M) % 2) == 1 ? count*2 : count;
                output[i] = (uint16_t)(input[i + temp])*eighth_power(2) + (uint16_t)(input[i]);
            }
        }

/*
    Debayerization function to transform input data into an RGB image.

    Bayer pattern is bggr or 

    B | G | B | G  
    -   -   -   - 
    G | R | G | R
    -   -   -   -
    B | G | B | G 
    -   -   -   - 
    G | R | G | R

    from which we want to demosaic an rbg bitmap picture

    R G B | R G B | R G B | R G B  
    -----   -----   -----   -----
    R G B | R G B | R G B | R G B  
    -----   -----   -----   -----
    R G B | R G B | R G B | R G B  
    -----   -----   -----   ----- 
    R G B | R G B | R G B | R G B  
*/
__global__ void demosaic(int M, int N, uint16_t* input, uint16_t* R, uint16_t* G, uint16_t* B){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = N*M;
            bool evenColumn;
            bool evenRow;
            bool firstColumn, lastColumn;
            bool firstRow, lastRow;
            for(int i = index; i < count; i += stride){
                evenColumn = (((i/M) % 2) == 1) ? true : false;
                evenRow = (((i%M) % 2) == 1) ? true : false;
                firstColumn = ((i/M) == 0) ? true : false;
                lastColumn = ((i/M) == (N-1)) ? true : false;
                firstRow = ((i&M) == 0) ? true : false;
                lastRow = ((i&M) == (M-1)) ? true : false;
                if(evenColumn){
                    if(evenRow){
                        R[i] = input[i];
                        if(lastColumn){
                            if(lastRow){
                                B[i] = input[i-M-1];
                                G[i] = (input[i-1] + input[i-M])/2;
                            }
                            else{
                                B[i] = (input[i-M-1] + input[i-M+1])/2;
                                G[i] = (input[i-1] + input[i+1])/2;
                            }
                        }
                        else{
                            if(lastRow){
                                B[i] = (input[i-M-1] + input[i+M-1])/2;
                                G[i] = (input[i-M] + input[i+M])/2;
                            }
                            else{
                                B[i] = (input[i-M-1] + input[i+M+1] + input[i-M+1] + input[i+M-1])/4;
                                G[i] = (input[i-1] + input[i+1] + input[i-M] + input[i+M])/4;
                            }  
                        }
                    }
                    else{
                        G[i] = input[i];
                        if(firstRow){
                            R[i] = input[i+1];
                        }
                        else{
                            R[i] = (input[i+1] + input[i-1])/2;
                        }
                        if(lastColumn){
                            B[i] = input[i-M];
                        }
                        else{
                            B[i] = (input[i+M] + input[i-M])/2;
                        }
                    }
                }
                else{
                    if(evenRow){
                        G[i] = input[i];
                        if(lastRow){
                            B[i] = input[i-1];
                        }
                        else{
                            B[i] = (input[i+1] + input[i-1])/2;
                        }
                        if(firstColumn){
                            R[i] = input[i+M];
                        }
                        else{
                            R[i] = (input[i+M] + input[i-M])/2;
                        }
                    }
                    else{
                        B[i] = input[i];
                        if(firstColumn){
                            if(firstRow){
                                R[i] = input[i+M+1];
                                G[i] = (input[i+1] + input[i+M])/2;
                            }
                            else{
                                R[i] = (input[i+M+1] + input[i+M-1])/2;
                                G[i] = (input[i+1] + input[i-1])/2;
                            }
                        }
                        else{
                            if(firstRow){
                                R[i] = (input[i+M+1] + input[i-M+1])/2;
                                G[i] = (input[i-M] + input[i+M])/2;
                            }
                            else{
                                R[i] = (input[i-M-1] + input[i+M+1] + input[i-M+1] + input[i+M-1])/4;
                                G[i] = (input[i-1] + input[i+1] + input[i-M] + input[i+M])/4;
                            }  
                        }
                    }
                }
            }
        }

__global__ void desample(int M, int N, float* input, float* output){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = N*M/4;
            for(int i = index; i < count; i += stride){
                output[i] = input[i*2 + M*((i*2)/M) ];
            }
}