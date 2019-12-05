#include <iostream>
#include <unistd.h>
#include "imgproc_thread.h"
#include "argpars.h"
#include "BeadsFinder.h"
#include "BackPropagator.h"

using namespace std;

void imgproc_thread(AppData& appData){
	printf("INFO: imgproc_thread: started\n");
	
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

		// Set the flag indicating that the camera was initialized
		appData.imgproc_is_initialized = true;

		
		if(Options::debug) printf("INFO: imgproc_thread: waiting till other App components are initialized\n");

		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;

		// At this point, the app is in the AppData::AppState::RUNNING state as the App enters RUNNING state automatically when all components are initialized.
		if(Options::debug) printf("INFO: imgproc_thread: entering the running stage\n");

		while(appData.appStateIs(AppData::AppState::RUNNING)) {
			auto t_cycle_start = std::chrono::system_clock::now();

			// wait till a new image is ready
			while(appData.camI.img_produced == appData.camI.img_processed && appData.appStateIs(AppData::AppState::RUNNING)) usleep(200);
			
			// If the app entered the EXITING state, break the loop and finish the thread
			if(appData.appStateIs(AppData::AppState::EXITING)) break;

			// Make copies of red and green channel
			auto t_cp_start = std::chrono::system_clock::now();
			appData.camI.G.copyTo(appData.G);
			appData.camI.R.copyTo(appData.R);
			auto t_cp_end = std::chrono::system_clock::now();

			// increase the number of processed images so that the camera starts capturing a new image
			++appData.camI.img_processed;

			// process the image
			// backprop
			auto t_backprop_start = std::chrono::system_clock::now();
			backprop_G.backprop(appData.G, appData.G_backprop);
			auto t_backprop_end = std::chrono::system_clock::now();

			// Update the image in beadsFinder where the beads are to be searched for
			auto t_beadsfinder_cp_start = std::chrono::system_clock::now();
			beadsFinder.updateImage(appData.G_backprop);
			auto t_beadsfinder_cp_end = std::chrono::system_clock::now();

			// find the beads
			auto t_beadsfinder_start = std::chrono::system_clock::now();
			beadsFinder.findBeads();
			{ // Limit the scope of the mutex
				std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
				appData.bead_count = beadsFinder.copyPositionsTo(appData.bead_positions);
			}
			auto t_beadsfinder_end = std::chrono::system_clock::now();

			// Set the sent_coords flag to false to indicate that new bead positions were found and can be sent to the host computer
			appData.set_sent_coords(false);

			auto t_cycle_end = std::chrono::system_clock::now();
			if(Options::verbose) {
				chrono::duration<double> cycle_elapsed_seconds = t_cycle_end - t_cycle_start;
				chrono::duration<double> cp_elapsed_seconds = t_cp_end - t_cp_start;
				chrono::duration<double> bp_elapsed_seconds = t_backprop_end - t_backprop_start;
				chrono::duration<double> bf_cp_elapsed_seconds = t_beadsfinder_cp_end - t_beadsfinder_cp_start;
				chrono::duration<double> bf_elapsed_seconds = t_beadsfinder_end - t_beadsfinder_start;

				std::cout << "TRACE: Backprop: " << bp_elapsed_seconds.count();
				std::cout << "| BF.cp: " << bf_cp_elapsed_seconds.count();
				std::cout << "| BF.findBeads: " << bf_elapsed_seconds.count();
				std::cout << "| cp: " << cp_elapsed_seconds.count();
				std::cout << "| whole cycle: " << cycle_elapsed_seconds.count();
				std::cout << "| #points: " << appData.bead_count << std::endl;
			}
		}

		appData.imgproc_is_initialized = false;
	}
	
	printf("INFO: imgproc_thread: ended\n");
}
