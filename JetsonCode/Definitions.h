#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define WIDTH 4056
#define HEIGHT 3040
#define lambda_green 0.00000052f
#define lambda_red 625e-9
#define blockSize 1024
#define n 1.45f
#define zGreen 0.0025f
#define zRed 0.00125f
#define dx 0.00000185f

enum MESSAGE_TYPE{
	MSG_HELLO,
	MSG_WAKEUP,
	MSG_SLEEP,
	MSG_SETTINGS,
	MSG_DISCONNECT,
	MSG_REQUEST,
	MSG_UNKNOWN_TYPE
};

#endif
