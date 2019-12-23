/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */

#include <iostream>
#include <unistd.h>
#include <cmath>
#include "imgproc_thread.h"
#include "argpars.h"
#include "BeadsFinder.h"
#include "BackPropagator.h"

using namespace std;
using namespace std::chrono;

void imgproc_thread(AppData& appData){
	if(Options::debug) printf("INFO: imgproc_thread: started\n");
	
	while(!appData.appStateIs(AppData::AppState::EXITING)) {
		if(Options::debug) printf("INFO: imgproc_thread: waiting for entering the INITIALIZING state\n");
		// Wait till the app enters the INITIALIZING state. If this fails (which could happen only in case of entering the EXITING state), break the loop.
		if(!appData.waitTillState(AppData::AppState::INITIALIZING)) break;

		// At this point, the app is in the AppData::AppState::INITIALIZING state, thus we initialize all needed stuff

		// Initialize the BackPropagator for the green image
		BackPropagator backprop_G(appData.values[STG_WIDTH], appData.values[STG_HEIGHT], LAMBDA_GREEN, (float)appData.values[STG_Z_GREEN]/1000000.0f);

		// Initialize the BeadFinder
		BeadsFinder beadsFinder(appData.values[STG_WIDTH], appData.values[STG_HEIGHT], (uint8_t)appData.values[STG_IMGTHRS], Options::saveimgs_bp);

		// Allocate the memory for the images
		appData.G.create(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
		appData.R.create(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
		appData.G_backprop.create(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);

		// Initialize the counters for measuring the cycle duration and jitter
		double iteration_count 			= 0;
		double avg_cycle_duration_us	= 0;
		double avg_jitter_us 			= 0;
		double cycle_period_us 	    	= 1e6/((double)appData.values[STG_FPS]);

		// Set the flag indicating that the camera was initialized
		appData.imgproc_is_initialized = true;
		
		if(Options::debug) printf("INFO: imgproc_thread: waiting till other App components are initialized\n");

		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;

		// At this point, the app is in the AppData::AppState::RUNNING state as the App enters RUNNING state automatically when all components are initialized.
		if(Options::debug) printf("INFO: imgproc_thread: entering the running stage\n");

		while(appData.appStateIs(AppData::AppState::RUNNING)) {
			auto t_cycle_start = steady_clock::now();

            // wait till a new image is ready
            std::unique_lock<std::mutex> lk(appData.cam_mtx);
			appData.cam_cv.wait(lk);
			// unlock the mutex so that the camera thread can proceed to capture a new image
			lk.unlock();
			
			// If the app entered the EXITING state, break the loop and finish the thread
			if(appData.appStateIs(AppData::AppState::EXITING)) break;

			// Make copies of red and green channel
			auto t_cp_start = steady_clock::now();
			appData.camIG.copyTo(appData.G);
			appData.camIR.copyTo(appData.R);
			auto t_cp_end = steady_clock::now();

			// process the image
			// backprop
			auto t_backprop_start = steady_clock::now();
			backprop_G.backprop(appData.G, appData.G_backprop);
			auto t_backprop_end = steady_clock::now();

			// find the beads
			auto t_beadsfinder_start = steady_clock::now();
			beadsFinder.findBeads(appData.G_backprop);
			{ // Limit the scope of the mutex
				std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
				appData.bead_count = beadsFinder.copyPositionsTo(appData.bead_positions);
			}
			auto t_beadsfinder_end = steady_clock::now();
			
			auto t_cycle_end = steady_clock::now();

			auto cycle_elapsed_seconds = t_cycle_end - t_cycle_start;
			avg_cycle_duration_us 	+= 1/(iteration_count+1)*(duration_cast<microseconds>(cycle_elapsed_seconds).count() - avg_cycle_duration_us);
			avg_jitter_us 			+= 1/(iteration_count+1)*( abs(cycle_period_us - duration_cast<microseconds>(cycle_elapsed_seconds).count()) - avg_jitter_us);
			iteration_count++;

			if(Options::verbose) {
				printf("TRACE: Backprop: %6.3f ms", 	duration_cast<microseconds>(t_backprop_end - t_backprop_start).count()/1000.0);
				printf("| BF.findBeads: %6.3f ms", 		duration_cast<microseconds>(t_beadsfinder_end - t_beadsfinder_start).count()/1000.0);
				printf("| cp: %6.3f ms", 				duration_cast<microseconds>(t_cp_end - t_cp_start).count()/1000.0);
				printf("| whole cycle: %6.3f ms", 		duration_cast<microseconds>(cycle_elapsed_seconds).count()/1000.0);
				printf("| #points: %d", 				appData.bead_count);
				printf("\n");
			}			
		}

		printf("Average cycle duration: %6.3f ms| Average jitter: %6.3f ms\n",	avg_cycle_duration_us/1000.0, avg_jitter_us/1000.0);
		
		appData.imgproc_is_initialized = false;
	}
	
	if(Options::debug) printf("INFO: imgproc_thread: ended\n");
}
