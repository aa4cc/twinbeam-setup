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

	ImageData<uint8_t> G, R, G_bckp, R_bckp;
	img[ImageType::RAW_G] = G;
	img[ImageType::RAW_R] = R;
	img[ImageType::BACKPROP_G] = G_bckp;
	img[ImageType::BACKPROP_R] = R_bckp;
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

/* Subscribe utility functions */
void AppData::addSubs(std::vector<sockpp::inet_address>& listOfSubs, sockpp::inet_address inaddr) {
	// Remove the subscriber, if it has already been in the list but possibly with a different port number
	removeSubs(listOfSubs, inaddr);
	// Add the subscriber to the list
	listOfSubs.push_back(inaddr);
}

void AppData::removeSubs(std::vector<sockpp::inet_address>& listOfSubs, sockpp::inet_address inaddr) {
	// Remove the subscriber also in the case when the port number does not match
	for(auto addr = listOfSubs.begin(); addr != listOfSubs.end();) {
		if(addr->address() == inaddr.address()) {
			listOfSubs.erase(addr);
			break;
		}
	}
}

/* Image subscribe utility functions */
void AppData::removeImageSubs(sockpp::inet_address inaddr) {
	// Remove the subsriber from subscription of all the image types
	removeImageSubs(inaddr, ImageType::RAW_G);
	removeImageSubs(inaddr, ImageType::RAW_R);
	removeImageSubs(inaddr, ImageType::BACKPROP_G);
	removeImageSubs(inaddr, ImageType::BACKPROP_R);
}

void AppData::removeImageSubs(sockpp::inet_address inaddr, ImageType imgType) {
	removeSubs(img_subs[imgType], inaddr);
}

void AppData::addImageSubs(sockpp::inet_address inaddr, ImageType imgType) {
	addSubs(img_subs[imgType], inaddr);
}

/* Coordinates subscribe utility functions */
void AppData::removeCoordsSubs(sockpp::inet_address inaddr) {
	removeSubs(coords_subs, inaddr);
}

void AppData::addCoordsSubs(sockpp::inet_address inaddr) {
	addSubs(coords_subs, inaddr);
}

int AppData::get_area(){
	return values[STG_WIDTH]*values[STG_HEIGHT];
}