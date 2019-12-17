/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 
#include "keyboard_thread.h"
#include "argpars.h"

void keyboard_thread(AppData& appData){
	printf("INFO: keyboard_thread: started\n");

	int input;
	while(!appData.appStateIs(AppData::AppState::EXITING)){
		input = getchar();
		if(input == 's'){
			printf("INFO: Stop capturing the images from keyboard.\n");
			appData.stopTheApp();
		}
		else if(input == 'w'){
			printf("INFO: Starting capturing the images rom keyboard.\n");
			appData.startTheApp();
		}
		else if(input == 'v'){
			Options::verbose = !Options::verbose;
			if(Options::verbose) {
				printf("INFO: Switching ON the VERBOSE mode.\n");
			} else {
				printf("INFO: Switching OFF the VERBOSE mode.\n");
			}
		}
		else if(input == 'd'){
			Options::debug = !Options::debug;
			if(Options::debug) {
				printf("INFO: Switching ON the DEBUG mode.\n");
			} else {
				printf("INFO: Switching OFF the DEBUG mode.\n");
			}
		}
		else if(input == 'e' || input == -1){
			printf("INFO: Exiting the program from keyboard.\n");
			appData.exitTheApp();
		}
	}

	printf("INFO: keyboard_thread: ended\n");
}