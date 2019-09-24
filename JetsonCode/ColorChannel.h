#ifndef COLOR_CHANNEL_H
#define COLOR_CHANNEL_H

#include "cufft.h"
#include <cstdint>

class ColorChannel {

private:
	bool display;
	int z;
	float lambda;
	cufftComplex* hq;
	cufftComplex* convoluted;
	void calculateHq();
	void convolve();

public:
	void allocate();
	void deallocate();
	void backpropagate(cufftComplex* kernel);
	void typeCast();
	void initialize(bool d, int zi, float l);

	uint16_t *original;
	float *doubleOriginal;
	float *backpropagated;
	float *maxima;
};

#endif