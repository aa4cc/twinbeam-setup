#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <stdint.h>
#include "Kernels.h"
#include "Definitions.h"

template<typename T>
class ImageData
{
private:
    T *h_data = nullptr;
    T *d_data = nullptr;         // pointers to data in host and device memory
    int width, height;           // dimensions of the image

public:
    ImageData() {};
    ImageData(uint16_t m, uint16_t n) : width{m}, height{n} {create(m, n);};

    void create(uint16_t m, uint16_t n) {
        if (h_data) release();
        width = m; height = n;
        // Allocate memory on the host side
        cudaHostAlloc((void **)&h_data,  sizeof(T)*width*height,  cudaHostAllocMapped);
        // Get the pointer 
        cudaHostGetDevicePointer((void **)&d_data,  (void *) h_data , 0);
    };
    
    void release() { if (h_data) cudaFreeHost(h_data); };
    bool isEmpty() {return h_data==nullptr;};

    T* hostPtr() {return h_data;};
    T* devicePtr() {return d_data;};
    void copyTo(const ImageData<T>& dst) { copyKernel<<<(width*height/2 + BLOCKSIZE -1)/BLOCKSIZE, BLOCKSIZE>>>(width, height, d_data, dst.d_data); };
    // void copyTo(const ImageData<T>& dst) { cudaMemcpy(dst.d_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToDevice); };

    ~ImageData() { release(); };
};


#endif