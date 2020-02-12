/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <stdint.h>
#define LAMBDA_GREEN 520e-9f
#define LAMBDA_RED 625e-9f
#define NBLOCKS 1024
#define REFRACTION_INDEX 1.45f
#define PIXEL_DX 1.55e-6f
#define BUFSIZE 1000
#define PI 3.14159265358979323846
#define CONVO_DIM_RED 60
#define CONVO_DIM_GREEN 160
#define MAX_NUMBER_BEADS 100

struct Position {
	uint16_t x;
	uint16_t y;
};

enum class ImageType {
	RAW_G,
	RAW_R,
	BACKPROP_G,
	BACKPROP_R,
};

enum class MessageType{
	HELLO,
	START,
	STOP,
	SETTINGS,
	DISCONNECT,
	IMG_REQUEST, // request on sending the backpropagated image
	TRACKER,
	COORDS_G,
	COORDS_R,
	IMG_SUBSCRIBE,
	COORDS_SUBSCRIBE,
	UNKNOWN_TYPE
};

enum SETTINGS_TYPE{
	STG_WIDTH,
	STG_HEIGHT,
	STG_OFFSET_X,
	STG_OFFSET_Y,
	STG_OFFSET_R2G_X,
	STG_OFFSET_R2G_Y,
	STG_EXPOSURE,
	STG_ANALOGGAIN,
	STG_DIGGAIN,
	STG_Z_RED,
	STG_Z_GREEN,
	STG_FPS,
	STG_IMGTHRS_G,
	STG_IMGTHRS_R,
	STG_NUMBER_OF_SETTINGS
};

#endif
