#include "Settings.h"
#include "Definitions.h"
#include "stdio.h"

using namespace std;
//width, height
int Settings::values[] = {1200, 1200, 1352, 596, 5000000, 50, 1, 3100, 2400, 30, 80};

AppState Settings::appState = AppState::IDLING;

bool Settings::camera_is_initialized 	= false;
bool Settings::imgproc_is_initialized	= false;
bool Settings::display_is_initialized 	= false;

void Settings::startTheApp() {
	// If the App has been alreadu put the INTIIALIZING or RUNNING state, do nothing
	if (Settings::appStateIs(AppState::INITIALIZING) || Settings::appStateIs(AppState::RUNNING)) 
		return;

	// If the app is in the IDLING state, put it to the INITIALIZING state
	if (Settings::appStateIs(AppState::IDLING))
		Settings::appStateSet(AppState::INITIALIZING);
}

void Settings::stopTheApp() {
	// Put the App to the IDLING state, if it is in any other state except EXITING state
	if (!Settings::appStateIs(AppState::EXITING) && !Settings::appStateIs(AppState::IDLING)) 
		Settings::appStateSet(AppState::IDLING);
}

bool Settings::waitTillState(AppState appState, useconds_t useconds) {
	// Wait till the app enters the appState or AppState::EXITING
	while(Settings::appState != appState && Settings::appState != AppState::EXITING) { usleep(useconds); }

	// return true whether the appState was entered or wherther the App should end
	return Settings::appState == appState;
}

bool Settings::waitTillAppIsInitialized(useconds_t useconds) {
	// Wait till the app enters the appState or AppState::EXITING
	while((!Settings::camera_is_initialized || !Settings::display_is_initialized || !Settings::imgproc_is_initialized) && Settings::appState != AppState::EXITING) { usleep(useconds); }

	// if the initialization was succesful enter the RUNNING state and return true. Otherwise, return false.
	if(Settings::appState != AppState::EXITING) {
		Settings::appStateSet(AppState::RUNNING);
		return true;
	} else {
		return false;
	}
}

bool Settings::appStateIs(AppState appState) {
	return Settings::appState == appState;
}

void Settings::appStateSet(AppState appState) {
	Settings::appState = appState;
}

void Settings::exitTheApp() {
	Settings::appStateSet(AppState::EXITING);
}

bool Settings::connected = false;
bool Settings::sent_coords = false;
bool Settings::requested_coords = false;
RequestType Settings::requested_type = RequestType::BACKPROPAGATED;
bool Settings::requested_image = false;

void Settings::print(){
	for(int i = 0 ; i < STG_NUMBER_OF_SETTINGS; i++){
		printf("%d\n", Settings::values[i]);
	}
}

void Settings::set_connected(const bool value){
	Settings::connected = value;
}

void Settings::set_sent_coords(const bool value){
	Settings::sent_coords = value;
}

void Settings::set_requested_type(const RequestType value){
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