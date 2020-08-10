/**
 * @author  Martin Gurtner
 */

#include "BeadsFinder.h"
#include <cstddef>
#include <cstdlib>
#include "stdio.h"
#include "Kernels.h"

using namespace std;

BeadsFinder::BeadsFinder(uint16_t m, uint16_t n, uint8_t img_thrs, bool dbg): im_width{m}, im_height{n}, img_threshold{img_thrs}, debug{dbg}
{
    // Initialize the Gaussian filter
    gaussianFilter = cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(19, 19), 5);

    numBlocks = (im_width*im_height/2 + NBLOCKS -1)/NBLOCKS;

    // Allocate memory for the array storing positions of the beads
    cudaMalloc(&d_positions, 2*MAX_NUMBER_BEADS*sizeof(uint16_t));
    // Allocate the memory for the counter of the found beads
    cudaMalloc((void **)&d_pointsCounterPtr, sizeof(uint32_t));
    
    // Allocates memory for the filtered image. Even thoug the filtered image is created by OpenCV and GpuMat, the memory has to be allocated manually so that the data in GpuMat are stored in a continous manner (withtou gaps after each column).
    cudaMalloc(&img_filt_data, im_width*im_height*sizeof(float));

    cudaMalloc(&img_bg_data, im_width*im_height*sizeof(uint8_t));
    cudaMalloc(&img_afterdiv_data, im_width*im_height*sizeof(float));
    cudaMalloc(&img_aftermorph_data, im_width*im_height*sizeof(float));

    // Initialize the Morphology filter
    morphFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_32FC1, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

    // read the background image
    cv::Mat img_bg = cv::imread("background.png", cv::IMREAD_GRAYSCALE);

    if (img_bg.cols != im_width || img_bg.rows != im_height) {
        cerr << "WARNING: the dimension of the background image does not match the dimension of the captured image and thus the background image will be ignored." << endl;
        // img_bg.forEach<Pixel>([](cv::Pixel &p, const int * position) -> void {
        //     p.x = 120;
        // });
    }

    cout << "background image loaded" << endl;

    cv::cuda::GpuMat img_bg_gpu(cv::Size(im_width, im_height), CV_8U, img_bg_data);
    img_bg_gpu.upload(img_bg);
};

uint32_t cnt = 0;
void BeadsFinder::findBeads(ImageData<uint8_t>& inputImg)
{ 
    const cv::cuda::GpuMat img_in(cv::Size(im_width, im_height), CV_8U, inputImg.devicePtr());
    const cv::cuda::GpuMat img_afterdiv(cv::Size(im_width, im_height), CV_32F, img_afterdiv_data);
    const cv::cuda::GpuMat img_aftermorph(cv::Size(im_width, im_height), CV_32F, img_aftermorph_data);
    const cv::cuda::GpuMat img_filt(cv::Size(im_width, im_height), CV_32F, img_filt_data);
    const cv::Mat img_write(cv::Size(im_width, im_height), CV_8U);

    {// Limit the scope of the mutex
        shared_lock<shared_timed_mutex> l(inputImg.mtx);

        // Divide the source image by the background
        imDivide<<<numBlocks, NBLOCKS>>>(im_width, im_height, (uint8_t*)img_in.data, img_bg_data, img_afterdiv_data);
    }

    // Blur the image by the gaussian filter
    morphFilter->apply(img_afterdiv, img_aftermorph);

    // Blur the image by the gaussian filter
    gaussianFilter->apply(img_aftermorph, img_filt);

    // Set the counter to zero
    cudaMemset(d_pointsCounterPtr, 0, sizeof(uint32_t));

    // Find the local minimums smaller than a given threshold and store their positions to d_positions[]
    getLocalMinima<<<numBlocks, NBLOCKS>>>(im_width, im_height, (float*)img_filt.data, d_positions, MAX_NUMBER_BEADS, d_pointsCounterPtr, img_threshold);

    // Copy back the value of the counter and the array to the CPU memory
    {
        lock_guard<mutex> l(_mtx);
        cudaMemcpy(&pointsCounter, d_pointsCounterPtr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&positions, d_positions, 2*pointsCounter*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    if(debug) {
        cout << "Points found: " << pointsCounter << "\n";
        for(int i = 0; i < pointsCounter; i++) 
            cout << "(" << positions[2*i] << "," << positions[2*i+1] << ")" << endl;

        
        img_filt.convertTo(img_in, CV_8U);
        img_in.download(img_write);

        for(int i = 0; i < pointsCounter; i++) 
            cv::circle(img_write, cv::Point(positions[2*i], positions[2*i+1]), 20, 255);
        
        char filename[40];
        sprintf(filename, "imgs/beadfinder_filt_%04d.png", cnt++);
        cv::imwrite( filename, img_write );        
    }
}

void BeadsFinder::copyPositionsTo(vector<Position>& bead_pos) {
    lock_guard<mutex> l(_mtx);
    bead_pos.clear();
    for (size_t i=0; i<pointsCounter; i++) {
        bead_pos.push_back({positions[2*i], positions[2*i+1]});
    }
}

BeadsFinder::~BeadsFinder() {
    cudaFree(d_pointsCounterPtr);
    cudaFree(d_positions);
    cudaFree(img_filt_data);
}