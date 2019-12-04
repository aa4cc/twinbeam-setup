/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */

#include "Kernels.h"

/*
    Calculation of the Hq matrix according to the equations in original .m file
*/
__global__ void calculateBackPropMatrix(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq)
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
    Element-wise multiplication of two  matrices.
*/
__global__ void multiplyInPlace(int N, int M, cufftComplex*  input, cufftComplex*  output)
{
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        temp = make_cuFloatComplex(output[i].x/(float)(N*M), output[i].y/(float)(N*M));
        output[i] = cuCmulf(input[i], temp);
    }
}
        
__global__ void multiply(int N, int M, cufftComplex*  input, cufftComplex*  kernel, cufftComplex* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        output[i] = cuCmulf(kernel[i], input[i]);
    }
}

__global__ void real(int M, int N, cufftComplex* input, float* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = N*M;
    for(int i = index; i < count; i += stride){
        output[i] = input[i].x;
    }
}
        
__global__ void imaginary(int M, int N, cufftComplex* input, float* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = N*M;
    for(int i = index; i < count; i += stride){
        output[i] = input[i].y;
    }
}

/*
    Converting a complex cufftComplex array to a float array of its absolute values
*/
__global__ void absoluteValue(int N, int M, cufftComplex* storageArray, float* outputArray)
{
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        //temp = make_cuFloatComplex(storageArray[(i%N)*M + i/N].x, storageArray[i%N*M + i/N].y);
        temp = make_cuFloatComplex(storageArray[i].x, storageArray[i].y);
        outputArray[i] = cuCabsf(temp);
    }
}
        
/*
    Converting a real array to complex cufftComplex array
*/
__global__ void convertToComplex(int count , float* real, cufftComplex* complex)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        complex[i] = make_cuComplex(real[i], 0);
    }
}

__global__ void u8ToFloat(int N, int M, uint8_t* input, float* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        result[i] = (float)input[i];
    }
}

__global__ void u16ToFloat(int N, int M, uint16_t* input, float* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        result[i] = (float)input[i];
    }
}

__global__ void floatToUInt8(int N, int M, float* input, uint8_t* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        result[i] = (uint8_t)input[i];
    }
}

__global__ void floatToUInt16(int N, int M, float* input, uint16_t* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        result[i] = (uint16_t)input[i];
    }
}

__global__ void getLocalMinima(int M, int N, float* input, uint16_t* points, uint32_t pointsMaxSize, uint32_t* pointsCounter, float thrs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int stride = blockDim.x * gridDim.x;
    int count = M*N;

    uint16_t idx, idy;
    for(int i = index; i < count; i += stride){
        // Continue if the value is larger than a certain value
        if (input[i] > thrs)
            continue;

        // Check whether we are on the edge of the image. If so, continue.
        idx = i % M;
        if (idx == 0 || idx == M-1)        
            continue;
        idy = i / M;
        if (idy == 0 || idy == N-1)
            continue;

        if( input[i-1] > input[i] && input[i+1] > input[i] && input[i-M] > input[i] && input[i+M] > input[i]) {
            uint32_t ind = atomicInc(pointsCounter, pointsMaxSize);
            points[2*ind] = idx;
            points[2*ind+1] = idy;
        }
    }    
}

template<typename T>
__global__ void copyKernel(int M, int N, T* input, T* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = N*M;
    for(int i = index; i < count; i += stride){
        output[i] = input[i];
    }
}

template __global__ void copyKernel<uint8_t>(int, int, uint8_t*, uint8_t*);
template __global__ void copyKernel<uint16_t>(int, int, uint16_t*, uint16_t*);
template __global__ void copyKernel<float>(int, int, float*, float*);