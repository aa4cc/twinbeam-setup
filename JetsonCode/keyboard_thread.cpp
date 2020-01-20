/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 
#include <pthread.h>
#include "keyboard_thread.h"
#include "argpars.h"

void keyboard_thread(AppData& appData){
	if(Options::debug) printf("INFO: keyboard_thread: started\n");

	if(Options::rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 20;

		if(Options::debug) printf("INFO: keyboard_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) fprintf(stderr, "WARNING: setting the priority of keyboard thread failed.\n");
	}	

	int input;
	while(!appData.appStateIs(AppData::AppState::EXITING)){
		input = getchar();
		if(input == 's'){
			if(Options::debug) printf("INFO: Stop capturing the images from keyboard.\n");
			appData.stopTheApp();
		}
		else if(input == 'w'){
			if(Options::debug) printf("INFO: Starting capturing the images rom keyboard.\n");
			appData.startTheApp();
		}
		else if(input == 'v'){
			Options::verbose = !Options::verbose;
			if(Options::verbose) {
				if(Options::debug) printf("INFO: Switching ON the VERBOSE mode.\n");
			} else {
				if(Options::debug) printf("INFO: Switching OFF the VERBOSE mode.\n");
			}
		}
		else if(input == 'd'){
			Options::debug = !Options::debug;
			if(Options::debug) {
				if(Options::debug) printf("INFO: Switching ON the DEBUG mode.\n");
			} else {
				if(Options::debug) printf("INFO: Switching OFF the DEBUG mode.\n");
			}
		}
		else if(input == 'o'){
			Options::show = !Options::show;
			if(Options::show) {
				if(Options::debug) printf("INFO: Switching ON the DISPLAY mode.\n");
			} else {
				if(Options::debug) printf("INFO: Switching OFF the DISPLAY mode.\n");
			}
		}
		else if(input == 'l'){
			Options::savevideo = !Options::savevideo;
			if(Options::savevideo) {
				if(Options::debug) printf("INFO: Switching ON the SAVEVIDEO mode.\n");
			} else {
				if(Options::debug) printf("INFO: Switching OFF the SAVEVIDEO mode.\n");
			}
		}
		else if(input == 'e' || input == -1){
			if(Options::debug) printf("INFO: Exiting the program from keyboard.\n");
			appData.exitTheApp();
		}
	}

	if(Options::debug) printf("INFO: keyboard_thread: ended\n");
}