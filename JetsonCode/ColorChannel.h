#ifndef COLOR_CHANNEL_H
#define COLOR_CHANNEL_H

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
	ColorChannel(bool d, int zi, float l);
	void allocate();
	void deallocate();
	void backpropagate(cufftComplex* kernel);
	void typeCast();

	uint16_t *original;
	float *doubleOriginal;
	float *backpropagated;
	float *maxima;
};

#endif