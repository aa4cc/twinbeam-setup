#include "Settings.h"

using namespace std;

int Settings::values[] = {1024, 1024, 1195, 500, 5000000, 3100, 2750};

bool Settings::connected = false;
bool Settings::sleeping = true;
bool Settings::send_points = false;
bool Settings::initialized = false;
bool Settings::force_exit = false;
bool Settings::touch_kill = false;

void Settings::set_setting(int index, const int new_setting){
	Settings::values[index] = new_setting;
}

void Settings::set_connected(const bool value){
	Settings::connected = value;
}

void Settings::set_sleeping(const bool value){
	Settings::sleeping = value;
}

void Settings::set_initialized(const bool value){
	Settings::initialized = value;
}

void Settings::set_force_exit(const bool value){
	Settings::force_exit = value;
}

void Settings::set_send_points(const bool value){
	Settings::send_points = value;
}