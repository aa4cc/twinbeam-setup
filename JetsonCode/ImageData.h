/**
 * @author  Martin Gurtner
 */

#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <stdint.h>
#include <mutex>
#include "cuda.h"
#include "Kernels.h"
#include "Definitions.h"

template<typename T>
class ImageData
{
private:
    T *h_data = nullptr;
    T *d_data = nullptr;         // pointers to data in host and device memory
    cudaStream_t stream = nullptr;
    int width, height;           // dimensions of the image

public:
    mutable std::mutex mtx;
    ImageData() { };
    ImageData(uint16_t m, uint16_t n) : width{m}, height{n} {create(m, n);};

    bool create(uint16_t m, uint16_t n) {
        release();
        width = m; height = n;
        // Allocate memory on the host side
        if(cudaHostAlloc((void **)&h_data,  sizeof(T)*width*height,  cudaHostAllocMapped) != cudaSuccess) return false;
        // Allocate memory on the device side
        if(cudaMalloc((void **)&d_data,  sizeof(T)*width*height) != cudaSuccess) return false;
        // 
        if(cudaStreamCreate(&stream) != cudaSuccess) return false;

        return true;
    };
    
    void release() {
        if (h_data) cudaFreeHost(h_data);
        if (d_data) cudaFree(d_data);
        if (stream) cudaStreamDestroy(stream);
    };

    bool isEmpty() {return h_data==nullptr;};

    T* hostPtr(bool sync=false) {        if(sync) {
            std::lock_guard<std::mutex> l_src(mtx);

            cudaMemcpyAsync(h_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        return h_data;
    };
    
    T* devicePtr(bool sync=false) {
        if(sync) {
            std::lock_guard<std::mutex> l_src(mtx);

            cudaMemcpyAsync(d_data, h_data, sizeof(T)*width*height, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        return d_data;
    };

    void copyTo(const ImageData<T>& dst) {
        std::lock_guard<std::mutex> l_src(mtx);
        std::lock_guard<std::mutex> l_dst(dst.mtx);
        // copyKernel<<<(width*height/2 + NBLOCKS -1)/NBLOCKS, NBLOCKS, 0, stream>>>(width, height, d_data, dst.d_data);
        cudaMemcpyAsync(dst.d_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    };

    ~ImageData() { release(); };
};


#endif