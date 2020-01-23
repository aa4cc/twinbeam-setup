/**
 * @author  Viktor-Adam Koropecky
 */

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
				return MessageType::IMG_REQUEST;
			case 't':
				return MessageType::TRACKER;
			case 'g':
				return MessageType::COORDS;
			case 'h':
				return MessageType::COORDS_CLOSEST;
			case 'b':
				return MessageType::IMG_SUBSCRIBE;
			case 'c':
				return MessageType::COORDS_SUBSCRIBE;
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
