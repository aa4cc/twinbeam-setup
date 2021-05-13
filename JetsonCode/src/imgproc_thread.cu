/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */

#include <iostream>
#include <unistd.h>
#include <cmath>
#include <pthread.h>
#include "imgproc_thread.h"
#include "sockpp/udp_socket.h"
#include "BeadsFinder.h"
#include "BackPropagator.h"
#include "fista.h"
#include "Phase_Kernels.h"

using namespace std;
using namespace std::chrono;

void imgproc_thread(AppData& appData){
	if(appData.params.debug) printf("INFO: imgproc_thread: started\n");

	if(appData.params.rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 50;
		
		if(appData.params.debug) printf("INFO: imgproc_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) fprintf(stderr, "WARNING: setting the priority of image processing thread failed.\n");
	}

	sockpp::udp_socket udp_sock;
	if (!udp_sock) {
		cerr << "ERROR: creating the UDP v4 socket: " << udp_sock.last_error_str() << endl;
	}
	int sendbuff = 2*sizeof(uint8_t)*appData.get_area();
	socklen_t optlen = sizeof(sendbuff);
	if(!udp_sock.set_option(SOL_SOCKET, SO_SNDBUF, &sendbuff, optlen)) {
		cerr << "ERROR: failed to increase the send buffer size for the UDP communication " << udp_sock.last_error_str() << endl;
	}
	char coords_buffer[2*sizeof(uint16_t)*MAX_NUMBER_BEADS + sizeof(uint32_t)];
	
	while(!appData.appStateIs(AppData::AppState::EXITING)) {
		if(appData.params.debug) printf("INFO: imgproc_thread: waiting for entering the INITIALIZING state\n");
		// Wait till the app enters the INITIALIZING state. If this fails (which could happen only in case of entering the EXITING state), break the loop.
		if(!appData.waitTillState(AppData::AppState::INITIALIZING)) break;

		// At this point, the app is in the AppData::AppState::INITIALIZING state, thus we initialize all needed stuff

		// Initialize the BackPropagator for the green image
		BackPropagator backprop_G(appData.params.img_width, appData.params.img_height, LAMBDA_GREEN, (float)appData.params.backprop_z_G/1000000.0f, appData.streamBack);
		// Initialize the BackPropagator for the red image
		BackPropagator backprop_R(appData.params.img_width, appData.params.img_height, LAMBDA_RED, (float)appData.params.backprop_z_R/1000000.0f, appData.streamBack);
		Fista fista((double)appData.params.backprop_z_G/1000000.0f,
			appData.params.rconstr,
			appData.params.iconstr,
			appData.params.mu,
			appData.params.img_width,
			appData.params.img_height,
			appData.params.cost,
			PIXEL_DX,
			LAMBDA_GREEN,
			REFRACTION_INDEX,
			appData.streamPhase );

		// Initialize the BeadFinders
		BeadsFinder beadsFinder_G(appData.params.img_width, appData.params.img_height, (uint8_t)appData.params.improc_thrs_G, (float)appData.params.improc_gaussFiltSigma_G, appData.streamBack);
		BeadsFinder beadsFinder_R(appData.params.img_width, appData.params.img_height, (uint8_t)appData.params.improc_thrs_R, (float)appData.params.improc_gaussFiltSigma_R, appData.streamBack);

		// Allocate the memory for the images
		appData.img[ImageType::RAW_G].create(appData.params.img_width, appData.params.img_height);
		appData.img[ImageType::RAW_R].create(appData.params.img_width, appData.params.img_height);
		appData.img[ImageType::BACKPROP_G].create(appData.params.img_width, appData.params.img_height);
		appData.img[ImageType::BACKPROP_R].create(appData.params.img_width, appData.params.img_height);
		appData.img[ImageType::MODULUS].create(appData.params.img_width, appData.params.img_height);
		appData.img[ImageType::PHASE].create(appData.params.img_width, appData.params.img_height);
		appData.img[ImageType::RAW_PHASE].create(appData.params.img_width, appData.params.img_height);
		appData.phaseImg.create(appData.params.img_width, appData.params.img_height);

		// Initialize the counters for measuring the cycle duration and jitter
		double iteration_count 			= 0;
		double avg_cycle_duration_us	= 0;
		double avg_jitter_us 			= 0;
		double cycle_period_us 	    	= 1e6/((double)appData.params.cam_FPS);

		// Set the flag indicating that the camera was initialized
		appData.imgproc_is_initialized = true;
		
		if(appData.params.debug) printf("INFO: imgproc_thread: waiting till other App components are initialized\n");

		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;

		// At this point, the app is in the AppData::AppState::RUNNING state as the App enters RUNNING state automatically when all components are initialized.
		if(appData.params.debug) printf("INFO: imgproc_thread: entering the running stage\n");

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
			appData.camIG.copyToAsync(appData.img[ImageType::RAW_G], 0);
			appData.camIR.copyToAsync(appData.img[ImageType::RAW_R], 0);
			appData.camIG.copyToAsync(appData.img[ImageType::RAW_PHASE], 0);
			U82D<<<N_BLOCKS,N_THREADS>>>(appData.params.img_width*appData.params.img_height, appData.img[ImageType::RAW_PHASE].devicePtr(), appData.phaseImg.devicePtr());
			auto t_cp_end = steady_clock::now();

			// process the image
			// backprop
			auto t_backprop_start = steady_clock::now();
			backprop_G.backprop(appData.img[ImageType::RAW_G], appData.img[ImageType::BACKPROP_G]);
			backprop_R.backprop(appData.img[ImageType::RAW_R], appData.img[ImageType::BACKPROP_R]);
			auto t_backprop_end = steady_clock::now();
			auto t_phase_start = steady_clock::now();
			if(iteration_count == 0)
				fista.iterate(appData.phaseImg.devicePtr(), appData.params.iters0, false);
			else
				fista.iterate(appData.phaseImg.devicePtr(), appData.params.iters, true);
			fista.update(appData.img[ImageType::MODULUS].devicePtr(), appData.img[ImageType::PHASE].devicePtr());
			auto t_phase_end = steady_clock::now();

			// find the beads (if enabled)
			auto t_beadsfinder_start = steady_clock::now();
			if(appData.params.beadsearch_G) {
				beadsFinder_G.findBeads(appData.img[ImageType::BACKPROP_G]);
				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp_G);
					beadsFinder_G.copyPositionsTo(appData.bead_positions_G);

					appData.beadTracker_G.update(appData.bead_positions_G);
				}
			}
			if(appData.params.beadsearch_R) {
				beadsFinder_R.findBeads(appData.img[ImageType::BACKPROP_R]);
				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp_R);
					beadsFinder_R.copyPositionsTo(appData.bead_positions_R);

					appData.beadTracker_R.update(appData.bead_positions_R);
				}
			}
			auto t_beadsfinder_end = steady_clock::now();

			// Send the images to the subscribers
			for(auto const& subs: appData.img_subs) {
				ImageType imgType = subs.first;
				bool img_sync = false;
				for(auto const& sub_addr: subs.second) {
					for (size_t i=0; i<128; ++i) {
						// synchronize the image (copy it from the device memory to the host memory) only if it hasn't been already synchronized
						ssize_t sent_bytes = udp_sock.send_to(appData.img[imgType].hostPtrAsync(0, !img_sync) + i*1024*8, sizeof(uint8_t)*appData.get_area()/128, sub_addr);
						img_sync = true;
					}

					if(appData.params.debug) cout << "INFO: sending image via UDP to " << sub_addr << endl;
				}
			}

			// Send the coordinates to the subscribers
			if (!appData.coords_subs.empty()) {
				uint32_t *beadCountP = (uint32_t*)coords_buffer;
				const vector<Position>& bp = appData.beadTracker_G.getBeadPositions();
				// Store the number of tracked objects
				*beadCountP = (uint32_t)bp.size();
				// Copy the tracked positions to the coords_buffer
				memcpy(coords_buffer+sizeof(uint32_t), bp.data(), 2*(*beadCountP)*sizeof(uint16_t));

				for(auto const& sub_addr: appData.coords_subs) {
					ssize_t sent_bytes = udp_sock.send_to(coords_buffer, sizeof(uint32_t) + 2*(*beadCountP)*sizeof(uint16_t), sub_addr);

					if(appData.params.debug) cout << "INFO: sending coordinates via UDP to " << sub_addr << " - " << sent_bytes << " bytes sent" << endl;
				}
			}
			
			auto t_cycle_end = steady_clock::now();

			auto cycle_elapsed_seconds = t_cycle_end - t_cycle_start;
			avg_cycle_duration_us 	+= 1/(iteration_count+1)*(duration_cast<microseconds>(cycle_elapsed_seconds).count() - avg_cycle_duration_us);
			avg_jitter_us 			+= 1/(iteration_count+1)*( abs(cycle_period_us - duration_cast<microseconds>(cycle_elapsed_seconds).count()) - avg_jitter_us);
			iteration_count++;

			if(appData.params.verbose) {
				printf("TRACE: Backprop: %6.3f ms", 	duration_cast<microseconds>(t_backprop_end - t_backprop_start).count()/1000.0);
				printf("| Phase: %6.3f ms", 		    duration_cast<microseconds>(t_phase_end - t_phase_start).count()/1000.0);
				printf("| BF.findBeads: %6.3f ms", 		duration_cast<microseconds>(t_beadsfinder_end - t_beadsfinder_start).count()/1000.0);
				printf("| cp: %6.3f ms", 				duration_cast<microseconds>(t_cp_end - t_cp_start).count()/1000.0);
				printf("| whole cycle: %6.3f ms", 		duration_cast<microseconds>(cycle_elapsed_seconds).count()/1000.0);
				printf("| #points: (%d, %d)", 			(int)appData.bead_positions_G.size(), (int)appData.bead_positions_R.size());
				printf("\n");
			}			
		}

		printf("Average cycle duration: %6.3f ms| Average jitter: %6.3f ms\n",	avg_cycle_duration_us/1000.0, avg_jitter_us/1000.0);
		
		appData.imgproc_is_initialized = false;
	}
	
	if(appData.params.debug) printf("INFO: imgproc_thread: ended\n");
}
