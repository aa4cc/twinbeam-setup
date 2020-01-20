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

	// Allocate the memory for the bead_position array so that the dynamic memory allocation is avoided
	bead_positions.reserve(MAX_NUMBER_BEADS);
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

int AppData::get_area(){
	return values[STG_WIDTH]*values[STG_HEIGHT];
}