/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */

#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "display_thread.h"
#include "argpars.h"

void display_thread(AppData& appData){
	printf("INFO: display_thread: started\n");
	
	char ret_key;
	char filename [50];
	int img_count = 0;

	while(!appData.appStateIs(AppData::AppState::EXITING)){
		if(Options::debug) printf("INFO: display_thread: waiting for entering the INITIALIZING state\n");
		// Wait till the app enters the INITIALIZING state. If this fails (which could happen only in case of entering the EXITING state), break the loop.
		if(!appData.waitTillState(AppData::AppState::INITIALIZING)) break;

		// Allocate the memory
		ImageData<uint8_t> G_backprop_copy(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
		ImageData<uint8_t> G_copy(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
		ImageData<uint8_t> R_copy(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);

		if (Options::show) {
			cv::namedWindow("Basic Visualization", cv::WINDOW_NORMAL);
			cv::setWindowProperty("Basic Visualization", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
		}
		
		const cv::cuda::GpuMat c_img_resized(cv::Size(800, 800), CV_8U);
		const cv::Mat img_disp(cv::Size(800, 800), CV_8U);
		
		appData.display_is_initialized = true;
		
		if(Options::debug) printf("INFO: display_thread: waiting till other App components are initialized\n");

		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;
		
		// At this point, the app is in the AppData::AppState::RUNNING state.
		if(Options::debug) printf("INFO: display_thread: entering the running stage\n");

		uint32_t img_since_last_time = 0;

		while(appData.appStateIs(AppData::AppState::RUNNING)) {
            // Wait for a new image
            std::unique_lock<std::mutex> lck(appData.cam_mtx);
            appData.cam_cv.wait(lck);
            lck.unlock();

            if (++img_since_last_time >= 6) {
                img_since_last_time = 0;
            } else {
                continue;
            }            
			
            auto start = std::chrono::system_clock::now();
            if (Options::show && !Options::saveimgs)
                appData.G_backprop.copyTo(G_backprop_copy);
            if (Options::saveimgs) {
                appData.G_backprop.copyTo(G_backprop_copy);
                appData.G.copyTo(G_copy);
                appData.R.copyTo(R_copy);
            }
            
            if (Options::saveimgs) {					
                const cv::Mat G_img(cv::Size(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]), CV_8U, G_copy.hostPtr(true));
                const cv::Mat R_img(cv::Size(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]), CV_8U, R_copy.hostPtr(true));
                const cv::Mat G_backprop_img(cv::Size(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]), CV_8U, G_backprop_copy.hostPtr(true));
                
                sprintf (filename, "./imgs/G_%05d.png", img_count);
                cv::imwrite( filename, G_img );
                
                sprintf (filename, "./imgs/R_%05d.png", img_count);
                cv::imwrite( filename, R_img );
                
                sprintf (filename, "./imgs/G_bp_%05d.png", img_count);
                cv::imwrite( filename, G_backprop_img );
            }				
            
            if (Options::show) {
                const cv::cuda::GpuMat c_img(cv::Size(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]), CV_8U, G_backprop_copy.devicePtr());

                // Resize the image so that it fits the display
                cv::cuda::resize(c_img, c_img_resized, cv::Size(800, 800));	
                
                c_img_resized.download(img_disp);

                // Draw bead positions
                { // Limit the scope of the mutex
                    std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
                    for(int i = 0; i < appData.bead_count; i++) {
                        uint32_t x = (appData.bead_positions[2*i]*800)/appData.values[STG_WIDTH];
                        uint32_t y = (appData.bead_positions[2*i+1]*800)/appData.values[STG_HEIGHT];
                        cv::circle(img_disp, cv::Point(x, y), 20, 255);
                    }
                }
                
                cv::imshow("Basic Visualization", img_disp);
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;
                if(Options::verbose) {
                    std::cout << "TRACE: Stroring the image took: " << elapsed_seconds.count() << "s\n";
                }

                ret_key = (char) cv::waitKey(1);
                if (ret_key == 27 || ret_key == 'x') appData.exitTheApp();  // exit the app if `esc' or 'x' key was pressed.					
            }

            img_count++;
        }
        
        if (Options::show) {
            // Close the windows
            cv::destroyWindow("Basic Visualization");
        }
		appData.display_is_initialized = false;
	}

	printf("INFO: display_thread: ended\n");
}