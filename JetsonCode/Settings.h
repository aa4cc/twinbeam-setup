#ifndef SETTINGS_H
#define SETTINGS_H

#include "Definitions.h"
#include <unistd.h>

class Settings {
public:
	static int values[STG_NUMBER_OF_SETTINGS];

	static AppState appState;
	
	static void startTheApp();
	static void stopTheApp();
	static bool waitTillState(AppState appState, useconds_t useconds = 100);
	static bool waitTillAppIsInitialized(useconds_t useconds = 100);
	static bool appStateIs(AppState appState);
	static void appStateSet(AppState appState);
	static void exitTheApp();
	
	static bool camera_is_initialized;
	static bool imgproc_is_initialized;
	static bool display_is_initialized;


	static bool connected;
	static bool sent_coords;
	static RequestType requested_type;
	static bool requested_image;
	static bool requested_coords;

	static void print();
	static void set_connected(const bool value);
	static void set_sent_coords(const bool value);
	static void set_requested_type(const RequestType value);
	static void set_requested_image(const bool value);
	static void set_requested_coords(const bool value);

	static int get_area();

};

#endif
