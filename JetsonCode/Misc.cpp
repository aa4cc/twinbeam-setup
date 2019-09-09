#include "Definitions.h"
#include "Misc.h"
#include <iostream>

MESSAGE_TYPE parseMessage(char* buf){
		switch (buf[0]){
			case 's':
				return MSG_WAKEUP;
			case 'q':
				return MSG_SLEEP;
			case 'o':
				return MSG_SETTINGS;
			case 'a':
				return MSG_HELLO;
			case 'd':
				return MSG_DISCONNECT;
			case 'r':
				return MSG_REQUEST;
			case 'x':
				return MSG_REQUEST_RAW_G;
			case 'y':
				return MSG_REQUEST_RAW_R;
			case 'g':
				return MSG_COORDS;
			default:
				return MSG_UNKNOWN_TYPE;
		}
}

void printArray(float* array, int width, int height){
	for(int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			printf("%f ", array[j + height*i]);
			}
			printf("\n");
		}
}
