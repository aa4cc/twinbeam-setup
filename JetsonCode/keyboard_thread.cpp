#include "keyboard_thread.h"

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
		else if(input == 'e' || input == -1){
			printf("INFO: Exiting the program from keyboard.\n");
			appData.exitTheApp();
		}
	}

	printf("INFO: keyboard_thread: ended\n");
}