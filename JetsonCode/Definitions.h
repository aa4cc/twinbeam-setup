#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// #define WIDTH 4056
// #define HEIGHT 3040
#define WIDTH 3840
#define HEIGHT 2160
#define LAMBDA_GREEN 520e-9f
#define LAMBDA_RED 625e-9f
#define BLOCKSIZE 1024
#define REFRACTION_INDEX 1.45f
#define PIXEL_DX 1.55e-6f
#define BUFSIZE 1000
#define PORT 30000
#define PI 3.14159265358979323846
#define SQUARE(x) x*x
#define CONVO_DIM_RED 60
#define CONVO_DIM_GREEN 160

enum class MessageType{
	HELLO,
	START,
	STOP,
	SETTINGS,
	DISCONNECT,
	REQUEST, // request on sending the backpropagated image
	REQUEST_RAW_G, // request on sending the unprocessed green channel
	REQUEST_RAW_R, // request on sending the unprocessed red channel
	COORDS,
	UNKNOWN_TYPE
};


enum class RequestType{
	BACKPROPAGATED,
	RAW_G,
	RAW_R
};

enum SETTINGS_TYPE{
	STG_WIDTH,
	STG_HEIGHT,
	STG_OFFSET_X,
	STG_OFFSET_Y,
	STG_EXPOSURE,
	STG_ANALOGGAIN,
	STG_DIGGAIN,
	STG_Z_RED,
	STG_Z_GREEN,
	STG_FPS,
	STG_IMGTHRS,
	STG_NUMBER_OF_SETTINGS
};

enum class AppState{
	IDLING,
	INITIALIZING,
	RUNNING,
	EXITING
};

/* Application state description
 * IDLING 			- The application waits till a start() command is called.
 * INITIALIZING 	- The camera and possibly other part are being initialized. When all parts are initialized, the app continutes to the RUNNING state.
 * RUNNING			- The images are captured and processed.
 * EXITING			- All the parts are deinitialized and then the app exits.
 */

#endif
