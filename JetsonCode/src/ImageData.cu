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
        std::shared_lock<std::shared_timed_mutex> lck(mtx);

        cudaMemcpy(h_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToHost);
    }
    return h_data;
};

template<typename T>
T* ImageData<T>::hostPtrAsync(cudaStream_t stream, bool sync) {
    if(sync) {
        std::shared_lock<std::shared_timed_mutex> lck(mtx);

        cudaMemcpyAsync(h_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToHost, stream);
    }
    return h_data;
};

template<typename T>
T* ImageData<T>::devicePtr() {
    return d_data;
};

template<typename T>
void ImageData<T>::copyTo(const ImageData<T>& dst) {
    std::shared_lock<std::shared_timed_mutex> l_src(mtx);
    std::unique_lock<std::shared_timed_mutex> l_dst(dst.mtx);
    cudaMemcpy(dst.d_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToDevice);
};

template<typename T>
void ImageData<T>::copyToAsync(const ImageData<T>& dst, cudaStream_t stream) {
    std::shared_lock<std::shared_timed_mutex> l_src(mtx);
    std::unique_lock<std::shared_timed_mutex> l_dst(dst.mtx);
    cudaMemcpyAsync(dst.d_data, d_data, sizeof(T)*width*height, cudaMemcpyDeviceToDevice, stream);
};