/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */
 
#include <cstdio>
#include "AppData.h"
#include <unistd.h>
#include <cstring>

AppData::AppData() {
    camera_is_initialized 	= false;
    imgproc_is_initialized	= false;
    display_is_initialized 	= false;
    connected = false;
    sent_coords = false;
    requested_coords = false;
    requested_type = RequestType::BACKPROPAGATED;
    requested_image = false;
}

void AppData::startTheApp() {
	// If the App has been alreadu put the INTIIALIZING or AppState::RUNNING state, do nothing
	if (appStateIs(AppState::INITIALIZING) || appStateIs(AppState::RUNNING)) 
		return;

	// If the app is in the AppState::IDLING state, put it to the AppState::INITIALIZING state
	if (appStateIs(AppState::IDLING))
		appStateSet(AppState::INITIALIZING);
}

void AppData::stopTheApp() {
	// Put the App to the AppState::IDLING state, if it is in any other state except AppState::EXITING state
	if (!appStateIs(AppState::EXITING) && !appStateIs(AppState::IDLING)) 
		appStateSet(AppState::IDLING);
}

bool AppData::waitTillState(AppState aStt, useconds_t useconds) {
	// Wait till the app enters the appState or AppState::EXITING
	while(appState != aStt && appState != AppState::EXITING) { usleep(useconds); }

	// return true whether the appState was entered or wherther the App should end
	return appState == aStt;
}

bool AppData::waitTillAppIsInitialized(useconds_t useconds) {
	// Wait till the app enters the appState or AppState::EXITING
	while((!camera_is_initialized || !display_is_initialized || !imgproc_is_initialized) && appState != AppState::EXITING) { usleep(useconds); }

	// if the initialization was succesful enter the AppState::RUNNING state and return true. Otherwise, return false.
	if(appState != AppState::EXITING) {
		appStateSet(AppState::RUNNING);
		return true;
	} else {
		return false;
	}
}

bool AppData::appStateIs(AppState aStt) {
	return appState == aStt;
}

void AppData::appStateSet(AppState aStt) {
	appState = aStt;
}

void AppData::exitTheApp() {
	appStateSet(AppState::EXITING);
}

void AppData::print(){
	for(int i = 0 ; i < STG_NUMBER_OF_SETTINGS; i++){
		printf("%d\n", values[i]);
	}
}

void AppData::saveReceivedBeadPos(uint32_t b_count, uint16_t* b_pos) {
	bead_count_received = b_count;
	memcpy(bead_positions_received, b_pos, 2*b_count*sizeof(uint16_t));

	// for(uint32_t i = 0 ; i < b_count; i++){
	// 	printf("(%d, %d)\n", b_pos[2*i], b_pos[2*i + 1]);
	// }
}

void AppData::set_connected(const bool value){
	connected = value;
}

void AppData::set_sent_coords(const bool value){
	sent_coords = value;
}

void AppData::set_requested_type(const RequestType value){
	requested_type = value;
}

void AppData::set_requested_image(const bool value){
	requested_image = value;
}

void AppData::set_requested_coords(const bool value){
	requested_coords = value;
}

void AppData::set_requested_coords_closest(const bool value){
	requested_coords_closest = value;
}

int AppData::get_area(){
	return values[STG_WIDTH]*values[STG_HEIGHT];
}