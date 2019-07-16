#ifndef MISC_H
#define MISC_H

#include "Definitions.h"

class Settings {
public:
	static int values[NUMBER_OF_SETTINGS];

	static void set_setting(int index, int new_setting);
};

#endif
