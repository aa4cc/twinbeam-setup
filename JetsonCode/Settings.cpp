#include "Settings.h"

using namespace std;

int Settings::values[STG_WIDTH] = 1024;
int Settings::values[STG_HEIGHT] = 1024;
int Settings::values[STG_OFFSET_X] = 1195;
int Settings::values[STG_OFFSET_Y] = 500;
int Settings::values[STG_EXPOSURE] = 5000000;
int Settings::values[STG_Z_RED] = 3100;
int Settings::values[STG_Z_GREEN] = 2750;

void Settings::set_setting(int index, int new_setting)
	Settings::values[index] = new_setting;