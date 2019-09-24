#ifndef SETTINGS_H
#define SETTINGS_H

#include "Definitions.h"

#define STG_WIDTH Settings::values[STG_WIDTH]
#define STG_HEIGHT Settings::values[STG_HEIGHT]

class Settings {
public:
	static int values[STG_NUMBER_OF_SETTINGS];

	static bool connected;
	static bool sleeping;
	static bool initialized;
	static bool force_exit;
	static bool sent_coords;
	static REQUEST_TYPE requested_type;
	static bool requested_image;
	static bool requested_coords;

	static bool touch_kill;

	static void set_setting(int index, const int new_setting);

	static void set_connected(const bool value);
	static void set_sleeping(const bool value);
	static void set_initialized(const bool value);
	static void set_force_exit(const bool value);
	static void set_sent_coords(const bool value);
	static void set_touch_kill(const bool value);
	static void set_requested_type(const REQUEST_TYPE value);
	static void set_requested_image(const bool value);
	static void set_requested_coords(const bool value);

	static int get_area();

};

#endif
