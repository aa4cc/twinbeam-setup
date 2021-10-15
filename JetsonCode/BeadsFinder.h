/**
 * @author  Martin Gurtner
 */

#ifndef BEADSFINDER_H
#define BEADSFINDER_H

#include <stdint.h> 
#include <mutex>
#include <vector>
#include "cuda.h"
#include <opencv2/cudafilters.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "ImageData.h"
#include "Definitions.h"

class BeadsFinder
{
private:
    bool debug;
    uint8_t img_threshold;
    mutable std::mutex _mtx;
    uint16_t im_width, im_height;
    cv::Ptr<cv::cuda::Filter> gaussianFilter;
    float* img_filt_data;
    uint32_t pointsCounter;
    uint16_t positions[2*MAX_NUMBER_BEADS];
    uint32_t *d_pointsCounterPtr;
    uint16_t *d_positions;
    int numBlocks;

public:
    BeadsFinder(uint16_t m, uint16_t n, uint8_t img_thrs, float gaussFiltSigma, bool dbg=false); 
    void findBeads(ImageData<uint8_t>& inputImg);
    void copyPositionsTo(std::vector<Position>& bead_pos);
    ~BeadsFinder();
};


#endif
