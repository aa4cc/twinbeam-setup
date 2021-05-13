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

using namespace std;
using namespace chrono;

void display_thread(AppData& appData){
	if(appData.params.debug) printf("INFO: display_thread: started\n");
    
    if(appData.params.rtprio) {
		struct sched_param schparam;
        schparam.sched_priority = 40;
        
		if(appData.params.debug) printf("INFO: display_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) fprintf(stderr, "WARNING: setting the priority of display thread failed.\n");
    }
    
	char ret_key;
	int img_count = 0;

	while(!appData.appStateIs(AppData::AppState::EXITING)){
		if(appData.params.debug) printf("INFO: display_thread: waiting for entering the INITIALIZING state\n");
		// Wait till the app enters the INITIALIZING state. If this fails (which could happen only in case of entering the EXITING state), break the loop.
		if(!appData.waitTillState(AppData::AppState::INITIALIZING)) break;

		// Allocate the memory
		ImageData<uint8_t> img_copy(appData.params.img_width, appData.params.img_height);

        // Set the flag indicatinf whether the window is opened or not to false
        bool windowOpened = false;
		
		const cv::cuda::GpuMat c_img_resized(cv::Size(DISP_WIDTH, DISP_HEIGHT), CV_8U);
        const cv::Mat img_disp(cv::Size(DISP_WIDTH, DISP_HEIGHT), CV_8U);        
        cv::VideoWriter video_writer;
        if (appData.params.savevideo) {
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
		
		if(appData.params.debug) printf("INFO: display_thread: waiting till other App components are initialized\n");
		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;
		
		// At this point, the app is in the AppData::AppState::RUNNING state.
		if(appData.params.debug) printf("INFO: display_thread: entering the running stage\n");

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
            
            if (appData.params.show) {
                // if the window has not been opened yet, open it
                if (!windowOpened) {
                    cv::namedWindow("Basic Visualization", cv::WINDOW_NORMAL);
                    if (appData.params.show_fullscreen) {
                        cv::setWindowProperty("Basic Visualization", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                    } else {
                        cv::resizeWindow("Basic Visualization", 1024,1024);
                    }                    
                    
                    windowOpened = true;
                }

                // Copy the image to the local copy
                appData.img[appData.params.displayImageType].copyTo(img_copy);

                const cv::cuda::GpuMat c_img(cv::Size(appData.params.img_width, appData.params.img_height), CV_8U, img_copy.devicePtr());

                // Resize the image so that it fits the display
                cv::cuda::resize(c_img, c_img_resized, cv::Size(DISP_WIDTH, DISP_HEIGHT));	
                
                c_img_resized.download(img_disp);

                if (appData.params.savevideo) {
                    video_writer.write(img_disp);
                    appData.frame_id++;
                }

                // If the video is being recorder, draw a text rendering this fact
                if (appData.params.savevideo) {
                    cv::putText(img_disp, "RECORDING", cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, 255);
                }

                // Draw bead positions (if beadsearch enabled and show_markers flag active)
                if((appData.params.beadsearch_G || appData.params.beadsearch_R) && (appData.params.show_markers || appData.params.show_labels)) {

                    if(appData.params.beadsearch_G && (appData.params.displayImageType==ImageType::RAW_PHASE || appData.params.displayImageType==ImageType::MODULUS)) {
                        lock_guard<mutex> mtx_bp(appData.mtx_bp_G);
                        for(auto &b : appData.bead_positions_G) {
                            auto x = (b.x*DISP_WIDTH)/appData.params.img_width;
                            auto y = (b.y*DISP_HEIGHT)/appData.params.img_height;
                            cv::circle(img_disp, cv::Point(x, y), 20, 255);
                        }
                        
                        int i=0;
                        for(auto &b : appData.beadTracker_G.getBeadPositions()) {
                            auto x = (b.x*DISP_WIDTH)/appData.params.img_width;
                            auto y = (b.y*DISP_HEIGHT)/appData.params.img_height;

                            if(appData.params.show_labels) {
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
            if(appData.params.verbose) {
                cout << "TRACE: Displaying/recording the image took: " << duration_cast<microseconds>(end-start).count()/1000.0 << " ms\n";
            }

            if(!appData.params.show && windowOpened) {
                // just in case the appData.params.show flag was disabled during the run-time, close the window
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

	if(appData.params.debug) printf("INFO: display_thread: ended\n");
}