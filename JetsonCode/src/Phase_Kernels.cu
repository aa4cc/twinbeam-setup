#include "Phase_Kernels.h"
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cufft.h"
#include "stdio.h"

//This include is completely unnecessary and can be omitted - only used to prevent Intellisense from thinking CUDA variables are undefined
#include <device_launch_parameters.h>

__global__ void propagator(int N, int M, double z, double dx, double n, double lambda, cufftComplex* Hq){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float FX, FY, res;
    float pre = (float)(n/lambda);
    float calc = (float)(1/dx);
    int newIndex;
    int count = N*M;
    for (int i = index; i < count; i += stride)
    {
        newIndex = (i + count/2-1) % (count);
        FX = ((float)(1+(i/M)) * calc/(float)(N)) - calc/2.0f;
        FY = ((float)(1+(i%M)) * calc/(float)(M)) - calc/2.0f;
        res = 2 * (float)(M_PI*z*pre) * sqrtf(1 - SQUARE(FX/pre) - SQUARE(FY/pre));
        if (sqrtf(SQUARE(FX)+SQUARE(FX)) < pre){
            Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuFloatComplex(cosf(res),sinf(res));
        }else{
            Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuFloatComplex(0,0);
        }
    }
}

__global__ void multiply(int count, cufftComplex*  in, cufftComplex* out){
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        temp = make_cuFloatComplex(out[i].x/(float)(count), out[i].y/(float)(count));
        out[i] = cuCmulf(in[i], temp);
    }
}

__global__ void multiplyf(int count, double*  in1, double*  in2, double*  out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in1[i]*in2[i];
    }
}

__global__ void multiplyfc(int count, double* in, cufftDoubleComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = make_cuDoubleComplex(out[i].x*in[i],out[i].y*in[i]);
    }
}

__global__ void absolute(int count, cufftDoubleComplex* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCabs(in[i]);
    }
}

__global__ void real(int count, cufftDoubleComplex* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in[i].x;
    }
}

__global__ void imag(int count, cufftDoubleComplex* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in[i].y;
    }
}

__global__ void angle(int count, cufftDoubleComplex* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = atan2(in[i].y,in[i].x);
    }
}

__global__ void modelFunc(int count, double rOffset, double iOffset, cufftDoubleComplex* in, cufftDoubleComplex* model, double* Imodel){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        model[i] = cuCadd(make_cuDoubleComplex(rOffset, iOffset),in[i]);
        Imodel[i] = SQUARE(cuCabs(model[i]));
    }
}

__global__ void conjugate(int count, cufftComplex *in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuConjf(in[i]);
    }
}

__global__ void simpleDivision(double* num, double* div, double* res){
        if(div[0] == 0.0f)
            div[0] = div[0] + 0.00001f;
        res[0] = num[0] / div[0];
}

__global__ void linear(int count, double* coef, double* constant, double* in, double* out, bool sign){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if(sign)
            out[i] = fma(coef[0], in[i], constant[i]);
        else
            out[i] = fma(coef[0], in[i], -constant[i]);
    }
}

__global__ void square(int count, double* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = SQUARE(in[i]);
    }
}

__global__ void simpleSum(double* in1, double* in2, double* out){
    out[0] = in1[0] + in2[0];
}



__global__ void add(int count, cufftDoubleComplex* in1, cufftDoubleComplex* in2, cufftDoubleComplex* out, bool sign){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if (sign)
            out[i] = cuCadd(in1[i], in2[i]);
        else
            out[i] = cuCsub(in1[i], in2[i]); 
    }
}

__global__ void strictBounds(int count, cufftDoubleComplex* arr, double r_min, double r_max, double i_min, double i_max){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        arr[i].x = fmax(fmin(r_max, arr[i].x), r_min);
        arr[i].y = fmax(fmin(i_max, arr[i].y), i_min);
    }
}

__global__ void positivityBounds(int count, cufftDoubleComplex* arr){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if(arr[i].x > 0)
            arr[i].x = 0;
    }
}


__global__ void strictBoundsf(int count, double* in, double min, double max){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        in[i] = fmax(fmin(max, in[i]), min);
    }
}

__global__ void softBounds(int count, cufftDoubleComplex* arr, double mu, double t){
    double tmp = mu*t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        double real = fabs(arr[i].x) - tmp;
        double imag = fabs(arr[i].y) - tmp;
        if(real < 0)
            real = 0;
        if(imag < 0)
            imag = 0;
        arr[i] = make_cuDoubleComplex(copysign(real, arr[i].x), copysign(imag, arr[i].y));
    }
}

// Most naive implementation of gaussian bluring - only effective on very small kernel sizes
// Future implementation could use shared memory for larger bandwidth
__global__ void rowConvolution(int N, int M, double diameter, double* kernel, double* image, double* output, bool horizontal){
    int offset;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    int count = N*M;
    for(int i = index; i < count; i+=gridSize){
        output[i] = 0;
        for(int j = 0; j < diameter; j++){
            offset = j - diameter/2;
            if(horizontal){
                if((i%N)+offset >= 0 && (i%N)+offset < N){
                    output[i] += kernel[j]*image[i+offset];
                }
                else
                    output[i] += kernel[j];
            } else {
                if((i/M)+offset >= 0 && (i/M)+offset < M){
                    output[i] += kernel[j]*image[i+offset*M];
                }
                else
                    output[i] += kernel[j];
            }
        }
    }



}

__global__ void offset(int count, double roff, double ioff, cufftDoubleComplex* in, cufftDoubleComplex* out){
    cufftDoubleComplex temp = make_cuDoubleComplex(roff, ioff);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCadd(temp, in[i]);
    }
}

__global__ void offsetf(int count, double roff, double* in, double* out, bool sign){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if(sign)
            out[i] = roff + in[i];
        else
            out[i] = roff - in[i];
    }
}

__global__ void extend(int count, int multiple, cufftDoubleComplex* in, cufftDoubleComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        for(int e = 0; e < multiple; e++){
            out[i + e*count] = in[i];
        }
    }
}

//TODO: Not very robust - expects 4 MP to 1 MP exactly
__global__ void shrinkTo1MP(int N, int M, double* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        out[i] = (in[i*2 + 2048*(i/M)] + in[i*2 + 2048*(i/M + 1)] + in[i*2 + 2048*(i/M+1) + 1] + in[i*2 + 2048*(i/M+1)])/4.0f;
    }
}

//
// CONVERSION KERNELS
//

__global__ void F2C(int count, double*  in, cufftDoubleComplex*  out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = make_cuDoubleComplex(in[i], 0);
    }
}

__global__ void D2u8(int count, double* in, uint8_t* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = (uint8_t)(in[i]*255.0f);
    }
}

__global__ void U82D(int count, uint8_t* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = (double)(in[i])/255.0f;
    }
}

__global__ void C2Z(int count, cufftComplex* in, cufftDoubleComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = make_cuDoubleComplex((double)in[i].x, (double)in[i].y);
    }
}

__global__ void Z2C(int count, cufftDoubleComplex* in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = make_cuFloatComplex((float)in[i].x, (float)in[i].y);
    }
}

//
// KERNELS FOR DIVISION BY ARRAYS
//

__global__ void contractf(int count, double constant, double* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in[i] / (constant + 0.00001f);
    }
}

__global__ void contractf_p(int count, double *constant, double* in, double* out){
    if(constant[0] < 0.00001f){
        constant[0] += 0.00001f;
    }
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in[i] / constant[0];
    }
}

//
// KERNELS FOR SCALING ARRAYS BY CONSTANTS
//

__global__ void scalef(int count, double constant, double* in, double* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = constant*in[i];
    }
}

__global__ void scale(int count, cufftDoubleComplex constant, cufftDoubleComplex* in, cufftDoubleComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCmul(constant,in[i]);
    }
}

__global__ void scale_p(int count, cufftDoubleComplex* constant, cufftDoubleComplex* in, cufftDoubleComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCmul(constant[0],in[i]);
    }
}

//
// PARALLEL REDUCTION KERNELS
//

//Fast parallel sum
/* 
*   The following function of sum is taken from the publicly accessible NVidia 
*   webinar found at https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
__global__ void sum(int count, double* in, double* result){
    extern __shared__ double sharedIn[];
    int thIdx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + thIdx;
    int stride = blockDim.x*gridDim.x;
    sharedIn[thIdx] = 0;
    
    for(unsigned int i = index; i < count; i+=stride){
        sharedIn[thIdx] += in[i];
    }
    __syncthreads();
    for(unsigned int i = blockDim.x/2 ; i>0 ; i>>=1){
        if(thIdx < i){
            sharedIn[thIdx] += sharedIn[thIdx+i];
        }
        __syncthreads();
    }
    if(thIdx == 0) result[blockIdx.x] = sharedIn[0];
}

__global__ void sumOfProducts(int count, double* in1, double* in2, double* result){
    extern __shared__ double sharedIn[];
    int thIdx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + thIdx;
    int stride = blockDim.x*gridDim.x;
    sharedIn[thIdx] = 0;
    
    for(unsigned int i = index; i < count; i+=stride){
        sharedIn[thIdx] += in1[i]*in2[i];
    }
    __syncthreads();
    for(unsigned int i = blockDim.x/2 ; i>0 ; i>>=1){
        if(thIdx < i){
            sharedIn[thIdx] += sharedIn[thIdx+i];
        }
        __syncthreads();
    }
    if(thIdx == 0) result[blockIdx.x] = sharedIn[0];
}

__global__ void maximum(int count, double* in, double* result){
    extern __shared__ double sharedIn[];
    int thIdx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + thIdx;
    int stride = blockDim.x*gridDim.x;
    sharedIn[thIdx] = in[index];
    
    for(int i = index+stride; i < count; i += stride){
        sharedIn[thIdx] = fmax(sharedIn[thIdx], in[index]);
    }
    __syncthreads();
    for(unsigned int i = blockDim.x/2 ; i>0 ; i>>=1){
        if(thIdx < i){
            sharedIn[thIdx] =  fmax(sharedIn[thIdx], sharedIn[thIdx+i]);
        }
        __syncthreads();
    }
    if (thIdx == 0) result[blockIdx.x] = sharedIn[thIdx];
}

__global__ void minimum(int count, double* in, double* result){
    extern __shared__ double sharedIn[];
    int thIdx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + thIdx;
    int stride = blockDim.x*gridDim.x;
    sharedIn[thIdx] = in[index];
    
    for(int i = index+stride; i < count; i += stride){
        sharedIn[thIdx] = fmin(sharedIn[thIdx], in[index]);
    }
    __syncthreads();
    for(unsigned int i = blockDim.x/2 ; i>0 ; i>>=1){
        if(thIdx < i){
            sharedIn[thIdx] =  fmin(sharedIn[thIdx], sharedIn[thIdx+i]);
        }
        __syncthreads();
    }
    if (thIdx == 0) result[blockIdx.x] = sharedIn[thIdx];
}

//
// WRAPPERS FOR PARALLEL REDUCTION KERNELS
//

void h_maximum(int count, double* d_in, double* d_result, cudaStream_t stream){
    maximum<<<N_BLOCKS,N_THREADS,N_THREADS*sizeof(double), stream>>>(count, d_in, d_result);
    maximum<<<1, N_BLOCKS, N_BLOCKS*sizeof(double), stream>>>(N_BLOCKS, d_result, d_result);
}

void h_minimum(int count, double* d_in, double* d_result, cudaStream_t stream){
    minimum<<<N_BLOCKS,N_THREADS,N_THREADS*sizeof(double), stream>>>(count, d_in, d_result);
    minimum<<<1, N_BLOCKS, N_BLOCKS*sizeof(double), stream>>>(N_BLOCKS, d_result, d_result);
}

void h_sum(int count, double* d_in, double* d_result, cudaStream_t stream){
    sum<<<N_BLOCKS,N_THREADS,N_THREADS*sizeof(double), stream>>>(count, d_in, d_result);
    sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(double), stream>>>(N_BLOCKS, d_result, d_result);
}

void h_sumOfProducts(int count, double* d_in1, double* d_in2, double* d_result, cudaStream_t stream){
    sumOfProducts<<<N_BLOCKS,N_THREADS,N_THREADS*sizeof(double), stream>>>(count, d_in1, d_in2, d_result);
    sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(double), stream>>>(N_BLOCKS, d_result, d_result);
}

void h_average(int count, double* d_in, double* d_result, cudaStream_t stream){
    sum<<<N_BLOCKS,N_THREADS,N_THREADS*sizeof(double), stream>>>(count, d_in, d_result);
    sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(double), stream>>>(N_BLOCKS, d_result, d_result);
    
    contractf<<<1,1, 0, stream>>>(1, (double)count, d_result, d_result);
}