#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define WIDTH 4056
#define HEIGHT 3040
#define lambda_green 520e-9f
#define lambda_red 625e-9f
#define blockSize 1024
#define n 1.45f
#define dx 1.55e-6f

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
