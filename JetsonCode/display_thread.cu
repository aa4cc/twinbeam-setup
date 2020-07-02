/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <ctime>
#include <string> 
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <pthread.h>
#include "display_thread.h"
#include "argpars.h"

using namespace std;
using namespace chrono;

void display_thread(AppData& appData){
	if(Options::debug) printf("INFO: display_thread: started\n");
    
    if(Options::rtprio) {
		struct sched_param schparam;
        schparam.sched_priority = 40;
        
		if(Options::debug) printf("INFO: display_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) fprintf(stderr, "WARNING: setting the priority of display thread failed.\n");
    }
    
	char ret_key;
	int img_count = 0;

	while(!appData.appStateIs(AppData::AppState::EXITING)){
		if(Options::debug) printf("INFO: display_thread: waiting for entering the INITIALIZING state\n");
		// Wait till the app enters the INITIALIZING state. If this fails (which could happen only in case of entering the EXITING state), break the loop.
		if(!appData.waitTillState(AppData::AppState::INITIALIZING)) break;

		// Allocate the memory
		ImageData<uint8_t> img_copy(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);

        // Set the flag indicatinf whether the window is opened or not to false
        bool windowOpened = false;
		
		const cv::cuda::GpuMat c_img_resized(cv::Size(DISP_WIDTH, DISP_HEIGHT), CV_8U);
        const cv::Mat img_disp(cv::Size(DISP_WIDTH, DISP_HEIGHT), CV_8U);        
        cv::VideoWriter video_writer;
        if (Options::savevideo) {
            // Null the frame_id
            appData.frame_id = 0;

            // Define the codec and create VideoWriter object.The output is stored in '%H%M%S_%d%m%Y.avi.avi' file. 
            time_t t = time(nullptr);
            tm tm = *localtime(&t);
            stringstream filename;
            filename << put_time(&tm, "./experiments_data/%H%M%S_%d%m%Y.avi");
            video_writer.open(filename.str(), cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(DISP_WIDTH, DISP_HEIGHT), false);

            if (!video_writer.isOpened()) {
                fprintf(stderr, "ERROR: failed to open the video file.\n");
                appData.exitTheApp();
            }
        }
		
		appData.display_is_initialized = true;
		
		if(Options::debug) printf("INFO: display_thread: waiting till other App components are initialized\n");
		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;
		
		// At this point, the app is in the AppData::AppState::RUNNING state.
		if(Options::debug) printf("INFO: display_thread: entering the running stage\n");

        uint32_t img_since_last_time = 0;

		while(appData.appStateIs(AppData::AppState::RUNNING)) {
            // Wait for a new image
            unique_lock<mutex> lck(appData.cam_mtx);
            appData.cam_cv.wait(lck);
            // unlock the mutex so that the camera thread can proceed to capture a new image
            lck.unlock();

            if (++img_since_last_time >= 3) {
                img_since_last_time = 0;
            } else {
                continue;
            } 

            auto start = steady_clock::now();
            
            if (Options::show) {
                // if the window has not been opened yet, open it
                if (!windowOpened) {
                    cv::namedWindow("Basic Visualization", cv::WINDOW_NORMAL);
                    if (Options::show_fullscreen) {
                        cv::setWindowProperty("Basic Visualization", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                    } else {
                        cv::resizeWindow("Basic Visualization", 1024,1024);
                    }                    
                    
                    windowOpened = true;
                }

                // Copy the image to the local copy
                appData.img[Options::displayImageType].copyTo(img_copy);

                const cv::cuda::GpuMat c_img(cv::Size(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]), CV_8U, img_copy.devicePtr());

                // Resize the image so that it fits the display
                cv::cuda::resize(c_img, c_img_resized, cv::Size(DISP_WIDTH, DISP_HEIGHT));	
                
                c_img_resized.download(img_disp);

                if (Options::savevideo) {
                    video_writer.write(img_disp);
                    appData.frame_id++;
                }

                // Draw bead positions (if beadsearch enabled and show_markers flag active)
                if((Options::beadsearch_G || Options::beadsearch_R) && (Options::show_markers || Options::show_labels)) {

                    if(Options::beadsearch_G && (Options::displayImageType==ImageType::RAW_G || Options::displayImageType==ImageType::BACKPROP_G)) {
                        lock_guard<mutex> mtx_bp(appData.mtx_bp_G);
                        for(auto &b : appData.bead_positions_G) {
                            auto x = (b.x*DISP_WIDTH)/appData.values[STG_WIDTH];
                            auto y = (b.y*DISP_HEIGHT)/appData.values[STG_HEIGHT];
                            cv::circle(img_disp, cv::Point(x, y), 20, 255);
                        }
                        
                        int i=0;
                        for(auto &b : appData.beadTracker_G.getBeadPositions()) {
                            auto x = (b.x*DISP_WIDTH)/appData.values[STG_WIDTH];
                            auto y = (b.y*DISP_HEIGHT)/appData.values[STG_HEIGHT];

                            if(Options::show_labels) {
                                cv::putText(img_disp, std::to_string(i++), cv::Point(x-11, y+11), cv::FONT_HERSHEY_SIMPLEX, 1.0, 255);
                            } else {
                                cv::drawMarker(img_disp, cv::Point(x, y), 255);
                            }                         
                        }
                    }

                    if(Options::beadsearch_R && (Options::displayImageType==ImageType::RAW_R || Options::displayImageType==ImageType::BACKPROP_R)) {
                        lock_guard<mutex> mtx_bp(appData.mtx_bp_R);
                        for(auto &b : appData.bead_positions_R) {
                            auto x = (b.x*DISP_WIDTH)/appData.values[STG_WIDTH];
                            auto y = (b.y*DISP_HEIGHT)/appData.values[STG_HEIGHT];
                            cv::circle(img_disp, cv::Point(x, y), 20, 255);
                        }
                        
                        int i=0;
                        for(auto &b : appData.beadTracker_R.getBeadPositions()) {
                            auto x = (b.x*DISP_WIDTH)/appData.values[STG_WIDTH];
                            auto y = (b.y*DISP_HEIGHT)/appData.values[STG_HEIGHT];

                            if(Options::show_labels) {
                                cv::putText(img_disp, std::to_string(i++), cv::Point(x-11, y+11), cv::FONT_HERSHEY_SIMPLEX, 1.0, 255);
                            } else {
                                cv::drawMarker(img_disp, cv::Point(x, y), 255);
                            }                         
                        }
                    }
                }
                
                cv::imshow("Basic Visualization", img_disp);                

                ret_key = (char) cv::waitKey(1);
                if (ret_key == 27 || ret_key == 'x') appData.exitTheApp();  // exit the app if `esc' or 'x' key was pressed.					
            }

            auto end = steady_clock::now();
            if(Options::verbose) {
                cout << "TRACE: Displaying/recording the image took: " << duration_cast<microseconds>(end-start).count()/1000.0 << " ms\n";
            }

            if(!Options::show && windowOpened) {
                // just in case the Options::show flag was disabled during the run-time, close the window
                cv::destroyWindow("Basic Visualization");
                windowOpened = false;
            }

            img_count++;
        }
        
        if (windowOpened) {
            // Close the windows
            cv::destroyWindow("Basic Visualization");
        }

		appData.display_is_initialized = false;
	}

	if(Options::debug) printf("INFO: display_thread: ended\n");
}