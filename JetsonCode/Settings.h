#ifndef SETTINGS_H
#define SETTINGS_H

#include "Definitions.h"

class Settings {
public:
	static int values[STG_NUMBER_OF_SETTINGS];

	static bool connected;
	static bool sleeping;
	static bool initialized;
	static bool force_exit;
	static bool send_points;

	static bool touch_kill;

	static void set_setting(int index, const int new_setting);

	static void set_connected(const bool value);
	static void set_sleeping(const bool value);
	static void set_initialized(const bool value);
	static void set_force_exit(const bool value);
	static void set_send_points(const bool value);
	static void set_touch_kill(const bool value);

};

#endif
