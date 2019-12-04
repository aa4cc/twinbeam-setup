/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */
#ifndef CAMERA_THREAD_H
#define CAMERA_THREAD_H

#include <stdint.h>
#include <mutex> 
#include "ImageData.h"


class Camera{
    public:
    static ImageData<uint8_t> G;
    static ImageData<uint8_t> R;
    static uint32_t img_produced;
    static uint32_t img_processed;

    static void camera_thread();
};

#endif