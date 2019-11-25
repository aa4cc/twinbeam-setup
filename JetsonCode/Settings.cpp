#include "Settings.h"
#include "Definitions.h"
#include "stdio.h"

using namespace std;
//width, height
int Settings::values[] = {1200, 1200, 1352, 596, 5000000, 100, 1, 3100, 2750, 30, 80};

bool Settings::connected = false;
bool Settings::sleeping = true;
bool Settings::initialized = false;
bool Settings::force_exit = false;
bool Settings::sent_coords = false;
bool Settings::requested_coords = false;
REQUEST_TYPE Settings::requested_type = BACKPROPAGATED;
bool Settings::requested_image = false;

void Settings::print(){
	for(int i = 0 ; i < STG_NUMBER_OF_SETTINGS; i++){
		printf("%d\n", Settings::values[i]);
	}
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

void Settings::set_sent_coords(const bool value){
	Settings::sent_coords = value;
}

void Settings::set_requested_type(const REQUEST_TYPE value){
	Settings::requested_type = value;
}

void Settings::set_requested_image(const bool value){
	Settings::requested_image = value;
}

void Settings::set_requested_coords(const bool value){
	Settings::requested_coords = value;
}

int Settings::get_area(){
	return Settings::values[STG_WIDTH]*Settings::values[STG_HEIGHT];
}