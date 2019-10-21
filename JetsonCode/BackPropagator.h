#ifndef BACKPROPAGATOR_H
#define BACKPROPAGATOR_H

#include "cuda.h"
#include "cufft.h"

#include "Definitions.h"
#include "Kernels.h"


class BackPropagator
{
private:
    int numBlocks = 1024;
    int M, N;
    cufftComplex *Hq, *image;
    float* image_float;
    cufftHandle fft_plan;

    void calculate_Hq(float z, float dx, float n, float lambda, cufftComplex* Hq);

public:
    BackPropagator( int m, int n, float lambda, float backprop_dist );

    void backprop(uint16_t* input, uint16_t* output );

    ~BackPropagator();
};


#endif