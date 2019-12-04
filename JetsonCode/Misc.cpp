#include "Definitions.h"
#include "Misc.h"
#include <iostream>

MessageType parseMessage(char* buf){
		switch (buf[0]){
			case 's':
				return MessageType::START;
			case 'q':
				return MessageType::STOP;
			case 'o':
				return MessageType::SETTINGS;
			case 'a':
				return MessageType::HELLO;
			case 'd':
				return MessageType::DISCONNECT;
			case 'r':
				return MessageType::REQUEST;
			case 'x':
				return MessageType::REQUEST_RAW_G;
			case 'y':
				return MessageType::REQUEST_RAW_R;
			case 'g':
				return MessageType::COORDS;
			default:
				return MessageType::UNKNOWN_TYPE;
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
