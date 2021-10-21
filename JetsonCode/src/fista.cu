#include "fista.h"
#include "Phase_Kernels.h"
#include "Definitions.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include "stdio.h"
#include "cudaDebug.h"

Fista::Fista(
            double z,
            double mu,
            int width,
            int height,
            bool b_cost,
            double dx,
            double lambda,
            double n,
            cudaStream_t stream
): width(width), height(height), b_cost(b_cost), mu(mu), z(z), stream(stream)
{

    count = width*height;

    allocate();
    propagator<<<N_BLOCKS, N_THREADS>>>(width, height, -z, dx, n, lambda, Hq);
    conjugate<<<N_BLOCKS,N_THREADS>>>(count, Hq, Hn);
}

void Fista::allocate(){
    cudaMalloc(&model, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&Hq, count*sizeof(cufftComplex));
    cudaMalloc(&Hn, count*sizeof(cufftComplex));
    cudaMalloc(&propagation, count*sizeof(cufftComplex));
    cudaMalloc(&guess, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&newGuess, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&u, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&temporary, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&sumArr, 2*N_BLOCKS*sizeof(double));
    cudaMalloc(&c, sizeof(double));
    cudaMalloc(&image, count*sizeof(double));
    cudaMalloc(&Imodel, count*sizeof(double));
    cudaMalloc(&temporaryf, 2*count*sizeof(double));
    cufftPlan2d(&fftPlan, height, width, CUFFT_C2C);
}

void Fista::propagate(cufftComplex* kernel, cufftDoubleComplex* input, cufftDoubleComplex* out){
    Z2C<<<N_BLOCKS,N_THREADS>>>(count, input, propagation);
    cufftExecC2C(fftPlan, propagation, propagation, CUFFT_FORWARD);
    multiply<<<N_BLOCKS, N_THREADS>>>(count, kernel, propagation);
    cufftExecC2C(fftPlan, propagation, propagation, CUFFT_INVERSE);
    C2Z<<<N_BLOCKS,N_THREADS>>>(count, propagation, out);
}

void Fista::calculateCost(double mu, double* model, cufftDoubleComplex* guess, double* temp, double* out){
    absolute<<<N_BLOCKS,N_THREADS>>>(count, guess, &temp[count]);
    square<<<N_BLOCKS,N_THREADS>>>(count, model, &temp[count]);

    h_sum(count, &temp[count], sumArr);
    h_sum(count, &temp[count], &sumArr[N_BLOCKS]);
    
    scalef<<<1,1>>>(1,mu,sumArr,sumArr);
    simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&out[0]);
}

void Fista::normalize(int c, double* arr){
    h_minimum(c, arr, sumArr);

    double temp;
    cudaMemcpy(&temp, sumArr, sizeof(double), cudaMemcpyDeviceToHost);
    offsetf<<<N_BLOCKS,N_THREADS>>>(c, -temp, arr, arr, true);

    h_maximum(c, arr, sumArr);
    contractf_p<<<N_BLOCKS,N_THREADS>>>(c, sumArr, arr, arr);
}

void Fista::iterate(double *input, int iters, bool warm){
    // Initialization of variables
    s = 1;
    if(iters > 100)
        iters = 100;
    if(b_cost){
        cudaMalloc(&cost, (1+iters)*sizeof(double));
        h_cost = (double*)malloc((iters+1)*sizeof(double));
    }

    //Copying the input image from host to device memory - computationally complex
    cudaMemcpy(image, input, count*sizeof(double), cudaMemcpyHostToDevice);
    h_average(count, image, sumArr);
    contractf_p<<<N_BLOCKS,N_THREADS>>>(count, sumArr, image, image);
    cudaMemcpy(m, sumArr, sizeof(double), cudaMemcpyDeviceToHost);

    //Copying the device memory image to device memory guesses

    if (!warm){
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, u);
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, guess);
    }
    
    for(int iter = 0; iter < iters; iter++){
        //Calculating the current iteration model 
        propagate(Hq, u, temporary);

        //Calculation of Imodel and model arrays
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, 1.0f, 0, temporary, model, Imodel);

        //Calculation of the optimal scaling parameter c
        h_sumOfProducts(count, image, Imodel, sumArr);
        h_sumOfProducts(count, Imodel, Imodel, &sumArr[N_BLOCKS]);
        contractf_p<<<1,1>>>(1, &sumArr[N_BLOCKS], sumArr, c);

        //Cost calculation with sparsity constraint
        linear<<<N_BLOCKS,N_THREADS>>>(count, c, image, Imodel, temporaryf, false);

        if(b_cost){
            calculateCost(mu, temporaryf, guess, temporaryf, &cost[iter]);
            double t_cost[1];
            cudaMemcpy(t_cost, &cost[iter], sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << "[DEBUG] Cost at iteration " << iter << " is " << t_cost[0] << std::endl;
        }

        //Calculating residues
        multiplyfc<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, model);
        propagate(Hn, model, temporary);

        double t = 0.2;
        scalef<<<1,1>>>(1, 2*t, c, c);
        F2C<<<1,1>>>(1,c,newGuess);
        scale_p<<<N_BLOCKS,N_THREADS>>>(count, newGuess, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(count, u, temporary, newGuess, false);

        //Applying soft thresholding bounds
        softBounds<<<N_BLOCKS,N_THREADS>>>(count, newGuess, mu, t);

        //Applying strict bounds
        positivityBounds<<<N_BLOCKS,N_THREADS>>>(count, newGuess);

        double s_new = 0.5*(1+std::sqrt(1+4*s*s));
        cufftDoubleComplex temp = make_cuDoubleComplex((s-1)/s_new,0);
        add<<<N_BLOCKS,N_THREADS>>>(count, newGuess, guess, temporary, false);
        scale<<<N_BLOCKS,N_THREADS>>>(count, temp, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(count, newGuess, temporary, u, true);

        s = s_new;
        cudaMemcpy(guess, newGuess, count*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
    }
    
    // Final cost calculation
    if(b_cost){
        propagate(Hq, u, newGuess);

        //Calculation of Imodel and model arrays
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, 1.0f, 0, newGuess, model, Imodel);

        //Calculation of the optimal scaling parameter c
        h_sumOfProducts(count, image, Imodel, sumArr);
        h_sumOfProducts(count, Imodel, Imodel, &sumArr[N_BLOCKS]);
        contractf_p<<<1,1>>>(1, &sumArr[N_BLOCKS], sumArr, c);

        //Cost calculation with sparsity constraint
        linear<<<N_BLOCKS,N_THREADS>>>(count, c, image, Imodel, temporaryf, false);

        calculateCost(mu, temporaryf, guess, temporaryf, &cost[iters]);
        double t_cost[1];
        cudaMemcpy(t_cost, &cost[iters], sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Current cost at iteration " << iters << " is " << t_cost[0] << std::endl;

        cudaMemcpy(h_cost, cost, (iters+1)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(cost);
    }

    // Moving results to host memory
    // Adding one to get the light wavefront (otherwise we only have the disturbance by the particles and electrodes)
    C2Z<<<1,1>>>(1, Hn, newGuess);
    coffset<<<N_BLOCKS,N_THREADS>>>(count, newGuess, guess, temporary);

    // Check if any error occured - important to note that untested kernels can lead to exceptions at cudaMemcpy calls
    gpuErrchk(cudaPeekAtLastError());
}

void Fista::update(uint8_t* modulus, uint8_t* phase){
    // temporary contains the latest results in complex form
    
    // Processing the modulus
    absolute<<<N_BLOCKS,N_THREADS>>>(count,temporary,temporaryf);
    scalef<<<N_BLOCKS,N_THREADS>>>(count, m[0], temporaryf, temporaryf);
    D2u8<<<N_BLOCKS,N_THREADS>>>(count,temporaryf,modulus);

    // Processing the phase
    angle<<<N_BLOCKS,N_THREADS>>>(count,temporary,temporaryf);
    strictBoundsf<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, 0,2);
    normalize(count, temporaryf);
    D2u8<<<N_BLOCKS,N_THREADS>>>(count,temporaryf,phase);
}

Fista::~Fista(){
    cudaFree(Hq);
    cudaFree(Hn);
    cudaFree(temporary);
    cudaFree(image);
    cudaFree(model);
    cudaFree(guess);
    cudaFree(newGuess);
    cudaFree(u);
    cudaFree(temporaryf);
    cudaFree(c);
    cudaFree(propagation);
    cufftDestroy(fftPlan);
}