#ifndef SETTINGS_H
#define SETTINGS_H

#include "Definitions.h"

#define SIZE NUMBER_OF_SETTINGS

class Settings {
public:
	static int values[SIZE];

	static void set_setting(int index, int new_setting);
};

#endif
