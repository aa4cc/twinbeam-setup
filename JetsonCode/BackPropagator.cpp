#include "BackPropagator.hpp"

#include "stdint.h"
#include "stdio.h"
#include <cmath>

/*
    Calculation of the Hq matrix according to the equations in original .m file
*/
__global__ void BackPropagator::calculate_Hq(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq)
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