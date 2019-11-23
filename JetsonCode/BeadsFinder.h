#ifndef BEADSFINDER_H
#define BEADSFINDER_H

#include <stdint.h> 
#include "cuda.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "Definitions.h"
#include "ImageData.h"

class BeadsFinder
{
private:
    uint16_t im_width, im_height;
    ImageData<uint8_t> img_data;
    cv::Ptr<cv::cuda::Filter> gaussianFilter;
    uint8_t *minima, *ret_image;
    float* img_filt_data;
    int numBlocks;

public:
    BeadsFinder(uint16_t m, uint16_t n);
    void updateImage(ImageData<uint8_t>& inputImg) { inputImg.copyTo(img_data); };
    void findBeads();
    ~BeadsFinder();
};


#endif