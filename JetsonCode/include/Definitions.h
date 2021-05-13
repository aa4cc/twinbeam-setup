/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <stdint.h>
#define LAMBDA_GREEN 515e-9f
#define LAMBDA_RED 625e-9f
#define REFRACTION_INDEX 1.45f
#define PIXEL_DX 1.55e-6f
#define BUFSIZE 5000
#define PI 3.14159265358979323846
#define CONVO_DIM_RED 60
#define CONVO_DIM_GREEN 160
#define MAX_NUMBER_BEADS 200
#define DISP_WIDTH 1024
#define DISP_HEIGHT 1024
#define DEFAULT_TCP_PORT 30000
#define N_BLOCKS 512
#define N_THREADS 1024

struct Position {
	uint16_t x;
	uint16_t y;
};

enum class ImageType {
	RAW_G,
	RAW_R,
	BACKPROP_G,
	BACKPROP_R,
	MODULUS,
	PHASE,
	RAW_PHASE
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

#endif
