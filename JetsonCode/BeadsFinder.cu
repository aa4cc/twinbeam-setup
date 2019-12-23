/**
 * @author  Martin Gurtner
 */

#include "BeadsFinder.h"
#include <cstddef>
#include <cstdlib>
#include "stdio.h"
#include "Kernels.h"

BeadsFinder::BeadsFinder(uint16_t m, uint16_t n, uint8_t img_thrs, bool dbg): im_width{m}, im_height{n}, img_threshold{img_thrs}, debug{dbg}
{
    // Initialize the Gaussian filter
    gaussianFilter = cv::cuda::createGaussianFilter(CV_8U, CV_32F, cv::Size(29, 29), 10);

    numBlocks = (im_width*im_height/2 + NBLOCKS -1)/NBLOCKS;

    // Allocate memory for the array storing positions of the beads
    cudaMalloc(&d_positions, 2*MAX_NUMBER_BEADS*sizeof(uint16_t));
    // Allocate the memory for the counter of the found beads
    cudaMalloc((void **)&d_pointsCounterPtr, sizeof(uint32_t));
    
    // Allocates memory for the filtered image. Even thoug the filtered image is created by OpenCV and GpuMat, the memory has to be allocated manually so that the data in GpuMat are stored in a continous manner (withtou gaps after each column).
    cudaMalloc(&img_filt_data, im_width*im_height*sizeof(float));

};

uint32_t cnt = 0;
void BeadsFinder::findBeads(ImageData<uint8_t>& inputImg)
{ 
    const cv::cuda::GpuMat img_in(cv::Size(im_width, im_height), CV_8U, inputImg.devicePtr());
    const cv::cuda::GpuMat img_filt(cv::Size(im_width, im_height), CV_32F, img_filt_data);
    const cv::Mat img_write(cv::Size(im_width, im_height), CV_8U);

    {// Limit the scope of the mutex
        std::shared_lock<std::shared_timed_mutex> l(inputImg.mtx);
        // Blur the image by the gaussian filter
        gaussianFilter->apply(img_in, img_filt);
    }

    // Set the counter to zero
    cudaMemset(d_pointsCounterPtr, 0, sizeof(uint32_t));

    // Find the local minimums smaller than a given threshold and store their positions to d_positions[]
    getLocalMinima<<<numBlocks, NBLOCKS>>>(im_width, im_height, (float*)img_filt.data, d_positions, MAX_NUMBER_BEADS, d_pointsCounterPtr, img_threshold);

    // Copy back the value of the counter and the array to the CPU memory
    {
        std::lock_guard<std::mutex> l(_mtx);
        cudaMemcpy(&pointsCounter, d_pointsCounterPtr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&positions, d_positions, 2*pointsCounter*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    if(debug) {
        std::cout << "Points found: " << pointsCounter << "\n";
        for(int i = 0; i < pointsCounter; i++) 
            std::cout << "(" << positions[2*i] << "," << positions[2*i+1] << ")" << std::endl;

        
        img_filt.convertTo(img_in, CV_8U);
        img_in.download(img_write);

        for(int i = 0; i < pointsCounter; i++) 
            cv::circle(img_write, cv::Point(positions[2*i], positions[2*i+1]), 20, 255);
        
        char filename[40];
        sprintf(filename, "imgs/beadfinder_filt_%04d.png", cnt++);
        cv::imwrite( filename, img_write );        
    }
}

uint32_t BeadsFinder::copyPositionsTo(uint16_t* data) {
    std::lock_guard<std::mutex> l(_mtx);
    memcpy(data, positions, 2*pointsCounter*sizeof(uint16_t));
    return pointsCounter;
}

BeadsFinder::~BeadsFinder() {
    cudaFree(d_pointsCounterPtr);
    cudaFree(d_positions);
    cudaFree(img_filt_data);
}