/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */
#ifndef CAMERA_THREAD_H
#define CAMERA_THREAD_H

#include <stdint.h>
#include <mutex> 
#include <atomic>
#include "ImageData.h"


class CameraImgI{
    public:
    std::atomic<unsigned int> img_produced;
    std::atomic<unsigned int> img_processed;
    ImageData<uint8_t> G;
    ImageData<uint8_t> R;

    CameraImgI() : img_produced{0}, img_processed{0} { };
};

void camera_thread(CameraImgI& CamI);

#endif