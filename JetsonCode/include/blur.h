#ifndef BLUR_H
#define BLUR_H

#include "cuda.h"

class Blur
{
    private:
        double* h_rowKernel;
        double* d_rowKernel;
        unsigned int diameter;
        void rowGaussianFilter(unsigned int diameter, double sigma, double* ret);
        bool initialized = false;

    public:
        Blur();
        Blur(unsigned int diameter, double sigma);
        Blur(unsigned int diameter);
        void gaussianBlur(int N, int M, unsigned int diameter, double sigma, double* in, double* temp, double* out);
        void printKernel();
        void boxBlur(int N, int M, unsigned int diameter, double* in, double* temp, double* out);
        void blur(int N, int M, double* in, double* temp, double* out);
        ~Blur();
};

#endif