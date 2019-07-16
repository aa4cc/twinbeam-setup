#include "Settings.h"
#include <iostream>

using namespace std;

int Settings::values[WIDTH] = 1024;
int Settings::values[HEIGHT] = 1024;
int Settings::values[OFFSET_X] = 1195;
int Settings::values[OFFSET_Y] = 500;
int Settings::values[EXPOSURE] = 5000000;
int Settings::values[Z_RED] = 3100;
int Settings::values[Z_GREEN] = 2750;

void Settings::set_setting(int index, int new_setting)
	Settings::values[index] = new_setting;