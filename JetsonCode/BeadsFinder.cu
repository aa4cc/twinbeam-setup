#include "BeadsFinder.h"
#include <cstddef>
#include <cstdlib>
#include "stdio.h"
#include "Kernels.h"


BeadsFinder::BeadsFinder(uint16_t m, uint16_t n): im_width{m}, im_height{n} {
    // Initialize the Gaussian filter
    gaussianFilter = cv::cuda::createGaussianFilter(CV_8U, CV_32F, cv::Size(29, 29), 10);

    numBlocks = (im_width*im_height/2 + BLOCKSIZE -1)/BLOCKSIZE;

    // Allocate the memory for the local copy of the image where the beads are to be searched for
    img_data.create(m, n);

    // Allocates memory for the image storing the position of the local minima
    cudaMalloc(&minima, im_width*im_height*sizeof(uint8_t));
    // Allocates memory for the filtered image. Even thoug the filtered image is created by OpenCV and GpuMat, the memory has to be allocated manually so that the data in GpuMat are stored in a continous manner (withtou gaps after each column).
    cudaMalloc(&img_filt_data, im_width*im_height*sizeof(float));
};

void BeadsFinder::findBeads()
{ 
    const cv::cuda::GpuMat img_in(cv::Size(im_width, im_height), CV_8U, img_data.devicePtr());
    const cv::cuda::GpuMat img_filt(cv::Size(im_width, im_height), CV_32F, img_filt_data);
    const cv::Mat img_write(cv::Size(im_width, im_height), CV_8U);

    // img_in.download(img_write);
    // cv::imwrite( "imgs/beadfinder_in.png", img_write );

    // Blur the image by the gaussian filter
    gaussianFilter->apply(img_in, img_filt);
    
    img_filt.convertTo(img_in, CV_8U);

    img_in.download(img_write);
    cv::imwrite( "imgs/beadfinder_filt.png", img_write );

    // Find local minima in the filtered image
    getLocalMinima<<<numBlocks, BLOCKSIZE>>>(im_width, im_height, img_filt_data, minima, 90);

    // const cv::cuda::GpuMat img_min(cv::Size(im_width, im_height), CV_8U, minima);
    // img_min.download(img_write);
    // cv::imwrite( "imgs/beadfinder_minima.png", img_write );
        
}

BeadsFinder::~BeadsFinder() {
    cudaFree(minima);
    cudaFree(img_filt_data);
    img_data.release();
}