/**
 * @author  Viktor-Adam Koropecky
 */

#include "stdint.h"
#include "cuda.h"
#include "cufft.h"
#include "stdio.h"
#include "Kernels.h"
#include "math.h"
#include <cmath>

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
                //temp = (sqrt(SQUARE(FX) + SQUARE(FY)) < (pre));
                if(temp == 0.0){
					Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuComplex(0,0);
				}
				else{
					Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuComplex(std::cos(res),std::sin(res));
				}
            }
        }

/*
    Element-wise multiplication of two (already transposed) matrices.
*/
__global__ void elMultiplication(int N, int M, cufftComplex*  Hq, cufftComplex*  Bq){
            cufftComplex temp;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                temp = make_cuFloatComplex(Bq[i].x/(float)(N*M), Bq[i].y/(float)(N*M));
                Bq[i] = cuCmulf(Hq[i], temp);
            }
        }
        
__global__ void elMultiplication2(int N, int M, cufftComplex*  input, cufftComplex*  kernel, cufftComplex* output){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                output[i] = cuCmulf(kernel[i], input[i]);
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
                //temp = make_cuFloatComplex(storageArray[(i%N)*M + i/N].x, storageArray[i%N*M + i/N].y);
                temp = make_cuFloatComplex(storageArray[i].x, storageArray[i].y);
                outputArray[i] = cuCabsf(temp);
            }
        }
        
__global__ void cutAndConvert(int N, int M, cufftComplex* input, float* output){
            cufftComplex temp;
            float floatTemp;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                floatTemp = cuCabsf(input[i]);
                if(floatTemp > 0 && input[i].x+input[i].y > 0)
					output[i] = floatTemp;
				else
					output[i] = 0;
            }
        }

/*
    Transposition of a float matrix
*/
__global__ void transpose(int N, int M, float* input, float* output){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < N*M; i += stride){
                output[i] = input[(i%N)*M + i/N];
            }
        }

__global__ void u16ToDouble(int N, int M, uint16_t* transposee, float* result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        result[i] = (float)transposee[i];
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

__global__ void convertToFloat(int count , float* output, cufftComplex* input){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = index; i < count; i += stride){
                output[i] = cuCabsf(input[i]);
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

//Function that returns desampled 
__global__ void desample(int M, int N, float* input, float* output){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = N*M/4;
            for(int i = index; i < count; i += stride){
                output[i] = input[i*2 + M*((i*2)/M) ];
                //lol
            }
}


__global__ void generateConvoMaskRed(int m, int n, float* convoMask){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = m*n;
            double temp;
            for(int i = index; i < count; i += stride){
                temp = sqrt((double)(SQUARE((double)((i%m) - (double)(m/2))) + SQUARE((double)((i/m) - (double)(m/2)))));
                convoMask[i] = 0;
                if( temp <= 24 ){
                    convoMask[i] = -0.5;
                }
                else if(temp > 27 && temp <= 30){
                    convoMask[i] = 1.3;
                }
            }
}

//This convolution core returns so far the best results for the green chanel. 
__global__ void generateConvoMaskGreen(int m, int n, float* convoMask){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = m*n;
            double temp;
            for(int i = index; i < count; i += stride){
                temp = sqrt((double)(SQUARE((double)((i%m) - (double)(m/2))) + SQUARE((double)((i/m) - (double)(m/2)))));
                convoMask[i] = 0;
                if( temp <= 23 ){
                    convoMask[i] = -0.78;
                }
                else if(temp > 23 && temp <= 28){
                    convoMask[i] = 1;
                }
            }
}

__global__ void sobelDerivation(int M, int N, float* input, float* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = M*N;
    int xFilter[9] = {1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1};
    int yFilter[9] = {1 , 0 , -1 , 2 , 0 , -2 , 1 , 0 , -1};
    short filterSize = 3;
    float xTemp;
    float yTemp;
    for(int i = index; i < count; i += stride){
        xTemp = 0;
        yTemp = 0;
        output[i] = 0;
        if( (M - i%M) > filterSize && (N - i/M) > filterSize ){
            for(int j = 0; j < SQUARE(filterSize); j++){
                xTemp += xFilter[j] * input[i + j%filterSize + M*(j/filterSize)];
                yTemp += yFilter[j] * input[i + j%filterSize + M*(j/filterSize)];
            }
            output[i] = sqrt(SQUARE(xTemp) + SQUARE(yTemp));

        }
    }
}

__global__ void findExtremes(int M, int N, float* input, float* extremes){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = M*N;
    for(int i = index; i < count; i += stride){
        if(input[i] > extremes[0])
            extremes[0] = input[i];
    } 
}

__global__ void normalize(int M, int N, float* input, float* extremes){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = M*N;
    for(int i = index; i < count; i += stride){
        input[i] = input[i] / extremes[0];
    } 
}

__global__ void getLocalMaxima(int M, int N, float* input, float* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = M*N;
    bool passable;
    for(int i = index; i < count; i += stride){
        passable = true;
        output[i] = 0;
        if( i % M != 0 && input[i-1] > input[i] )
            passable = false;
        if( i / M != 0 && input[i-M] > input[i] )
            passable = false;
        if( i % M != M-1 && input[i+1] > input[i] )
            passable = false;
        if( i / M != M-1 && input[i+M] > input[i] )
            passable = false;
        if( passable == true )
            output[i] = input[i];
    }
}

__global__ void kernelToImage(int M, int N, int kernelDim, float* kernel, cufftComplex* outputKernel){
			int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = kernelDim*kernelDim;
            int center = (kernelDim)/2;
            for(int i = index; i < count; i += stride){
				int lineNum = i/kernelDim;
                int colNum = i%kernelDim;
                if(colNum >= center && lineNum >= center)
                    outputKernel[colNum-center + (lineNum-center)*M] = make_cuComplex(kernel[i], 0);
                else if(colNum >= center)
                    outputKernel[colNum-center + (N - center + lineNum)*M] = make_cuComplex(kernel[i], 0);
                else if(lineNum >= center)
                    outputKernel[colNum + M - center + (lineNum-center)*M] = make_cuComplex(kernel[i], 0);
                else
                    outputKernel[colNum + M - center + (N - center + lineNum)*M] = make_cuComplex(kernel[i], 0);
            }
	}
	
__global__ void findPoints(int M, int N, float* input, int* output){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int count = N*M;
		for(int i = index; i < count; i += stride){
			if(input[i] > 0){
				output[i] = i;
			}
		}
	
	}

/*
__global__ void stupidSort(int M, int N, int* input, int* output, int *currentIndex){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int count = N*M;
		for(int i = index; i < count; i += stride){
			if(input[i] > 0){
				atomicAdd(currentIndex[0], 1); 
				output[currentIndex[0]] = input[i];
			}
		}
	
	}

*/