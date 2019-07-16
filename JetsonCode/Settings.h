#ifndef SETTINGS_H
#define SETTINGS_H

#include "Definitions.h"

class Settings {
public:
	static int values[STG_NUMBER_OF_SETTINGS];

	static void set_setting(int index, int new_setting);
};

#endif
