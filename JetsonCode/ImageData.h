/**
 * @author  Martin Gurtner
 *
 * IMPORTANT - It is assumed that only the data stored in the device memory are modified.
 */

#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <stdint.h>
#include <mutex>
#include <shared_mutex>
#include "cuda.h"   

template<typename T>
class ImageData
{
private:
    T *h_data = nullptr;
    T *d_data = nullptr;         // pointers to data in host and device memory
    int width, height;           // dimensions of the image

public:
    std::mutex mtx;
    ImageData() { };
    // ImageData(const ImageData<T>&) = default;
    ImageData<T>& operator=(const ImageData<T>& id) {
        width = id.width;
        height = id.height;
        h_data = nullptr;
        d_data = nullptr;
        return *this;
    };

    ImageData(uint16_t m, uint16_t n) : width{m}, height{n} {create(m, n);};

    bool create(uint16_t m, uint16_t n);
    void release();
    bool isEmpty() {return h_data==nullptr;};

    T* hostPtr(bool sync=false);
    T* devicePtr();

    void copyTo(ImageData<T>& dst);

    ~ImageData() { release(); };
};

// explicit instatiations
template class ImageData<uint8_t>;
template class ImageData<float>;


#endif