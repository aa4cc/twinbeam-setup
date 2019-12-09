/**
 * @author  Martin Gurtner
 */

#include "ImageData.h"

template<typename T>
bool ImageData<T>::create(uint16_t m, uint16_t n) {
    release();
    width = m; height = n;
    // Allocate memory on the host side
    if(cudaHostAlloc((void **)&h_data,  sizeof(T)*width*height,  cudaHostAllocMapped) != cudaSuccess) return false;
    // Allocate memory on the device side
    if(cudaMalloc((void **)&d_data,  sizeof(T)*width*height) != cudaSuccess) return false;

    return true;
};

template<typename T>
void ImageData<T>::release() {
    if (h_data) cudaFreeHost(h_data);
    if (d_data) cudaFree(d_data);
};

template<typename T>
T* ImageData<T>::hostPtr(bool sync) {
    if(sync) {
        std::lock_guard<std::mutex> l_src(mtx);

        cudaMemcpy(h_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToHost);
    }
    return h_data;
};

template<typename T>
T* ImageData<T>::devicePtr(bool sync) {
    if(sync) {
        std::lock_guard<std::mutex> l_src(mtx);

        cudaMemcpy(d_data, h_data, sizeof(T)*width*height, cudaMemcpyHostToDevice);
    }
    return d_data;
};

template<typename T>
void ImageData<T>::copyTo(const ImageData<T>& dst) {
    std::lock_guard<std::mutex> l_src(mtx);
    std::lock_guard<std::mutex> l_dst(dst.mtx);
    cudaMemcpy(dst.d_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToDevice);
};