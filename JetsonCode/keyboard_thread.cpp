/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 
#include <pthread.h>
#include "keyboard_thread.h"

void keyboard_thread(AppData& appData){
	if(appData.params.debug) printf("INFO: keyboard_thread: started\n");

	if(appData.params.rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 20;

		if(appData.params.debug) printf("INFO: keyboard_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) fprintf(stderr, "WARNING: setting the priority of keyboard thread failed.\n");
	}	

	int input;
	while(!appData.appStateIs(AppData::AppState::EXITING)){
		input = getchar();
		if(input == 's'){
			if(appData.params.debug) printf("INFO: Stop capturing the images from keyboard.\n");
			appData.stopTheApp();
		}
		else if(input == 'w'){
			if(appData.params.debug) printf("INFO: Starting capturing the images rom keyboard.\n");
			appData.startTheApp();
		}
		else if(input == 'v'){
			appData.params.verbose = !appData.params.verbose;
			if(appData.params.verbose) {
				if(appData.params.debug) printf("INFO: Switching ON the VERBOSE mode.\n");
			} else {
				if(appData.params.debug) printf("INFO: Switching OFF the VERBOSE mode.\n");
			}
		}
		else if(input == 'd'){
			appData.params.debug = !appData.params.debug;
			if(appData.params.debug) {
				if(appData.params.debug) printf("INFO: Switching ON the DEBUG mode.\n");
			} else {
				if(appData.params.debug) printf("INFO: Switching OFF the DEBUG mode.\n");
			}
		}
		else if(input == 'o'){
			if(!appData.params.show) {
				// If the display was disabled, enable it and set the image type to the backpropagated green channel
				appData.params.show = true;
				appData.params.displayImageType = ImageType::BACKPROP_G;
				if(appData.params.debug) printf("INFO: Switching ON the DISPLAY mode.\n");
			} else {
                switch(appData.params.displayImageType) {
                    case ImageType::RAW_G:
						appData.params.displayImageType = ImageType::RAW_R;
						if(appData.params.debug) printf("INFO: Setting the type of the displayed image to RAW_R\n");
                        break;
                    case ImageType::RAW_R:
						appData.params.show = false;
						if(appData.params.debug) printf("INFO: Switching OFF the DISPLAY mode.\n");
                        break;
                    case ImageType::BACKPROP_G:
						appData.params.displayImageType = ImageType::BACKPROP_R;
						if(appData.params.debug) printf("INFO: Setting the type of the displayed image to BACKPROP_R\n");
                        break;
                    case ImageType::BACKPROP_R:
						appData.params.displayImageType = ImageType::RAW_G;
						if(appData.params.debug) printf("INFO: Setting the type of the displayed image to RAW_G\n");
                        break;
                }
			}
		}
		else if(input == 'l'){
			appData.params.savevideo = !appData.params.savevideo;
			if(appData.params.savevideo) {
				if(appData.params.debug) printf("INFO: Switching ON the SAVEVIDEO mode.\n");
			} else {
				if(appData.params.debug) printf("INFO: Switching OFF the SAVEVIDEO mode.\n");
			}
		}
		else if(input == 'f'){
			appData.params.show_fullscreen = !appData.params.show_fullscreen;
			if(appData.params.show_fullscreen) {
				if(appData.params.debug) printf("INFO: Switching ON the FULLSCREEN mode.\n");
			} else {
				if(appData.params.debug) printf("INFO: Switching OFF the FULLSCREEN mode.\n");
			}
		}
		else if(input == 'e' || input == -1){
			if(appData.params.debug) printf("INFO: Exiting the program from keyboard.\n");
			appData.exitTheApp();
		}
		else if(input == 'h' || input == -1){
			printf("Keyboard commands:\n"
					"\t s - sleep \n"
					"\t w - wakeup \n"
					"\t v - verbose ON/OFF \n"
					"\t d - debug ON/OFF \n"
					"\t o - cycle display modes \n"
					"\t l - record video ON/OFF  (you must sleep and wakeup the process to swith on/off the video recording mode) \n"
					"\t f - fullscreen mode (you must sleep and wakeup the process to swith on/off the fullscreen mode) \n"
					"\t h - keyboard commands \n"
					"\t e - exit \n");
		}
	}

	if(appData.params.debug) printf("INFO: keyboard_thread: ended\n");
}