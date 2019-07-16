#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define WIDTH 4056
#define HEIGHT 3040
#define LAMBDA_GREEN 520e-9f
#define LAMBDA_RED 625e-9f
#define BLOCKSIZE 1024
#define REFRACTION_INDEX 1.45f
#define PIXEL_DX 1.55e-6f
#define BUFSIZE 1000
#define PORT 30000

enum MESSAGE_TYPE{
	MSG_HELLO,
	MSG_WAKEUP,
	MSG_SLEEP,
	MSG_SETTINGS,
	MSG_DISCONNECT,
	MSG_REQUEST, // request on sending the backpropagated image
	MSG_REQUEST_RAW_G, // request on sending the unprocessed green channel
	MSG_REQUEST_RAW_R, // request on sending the unprocessed red channel
	MSG_UNKNOWN_TYPE
};


enum REQUEST_TYPE{
	BACKPROPAGATED,
	RAW_G,
	RAW_R
};

enum SETTINGS_TYPE{
	WIDTH,
	HEIGHT,
	OFFSET_X,
	OFFSET_Y,
	EXPOSURE,
	Z_RED,
	Z_GREEN,
	NUMBER_OF_SETTINGS
}

#endif
