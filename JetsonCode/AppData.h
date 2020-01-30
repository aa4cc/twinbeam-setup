/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */

#ifndef APPDATA_H
#define APPDATA_H

#include <mutex> 
#include <condition_variable>
#include <vector>
#include <map>
#include "sockpp/inet_address.h"
#include "Definitions.h"
#include "ImageData.h"
#include "BeadTracker.h"

class AppData {
public:
    // Enum types
    enum class AppState{
        IDLING,             // IDLING 			- The application waits till a start() command is called.
        INITIALIZING,       // INITIALIZING 	- The camera and possibly other part are being initialized. When all parts are initialized, the app continutes to the RUNNING state.
        RUNNING,            // RUNNING			- The images are captured and processed.
        EXITING             // EXITING			- All the parts are deinitialized and then the app exits.
    };

    // Member variables

	int values[STG_NUMBER_OF_SETTINGS] = {1024, 1024, 1440, 592, 550, 20, 5000000, 2, 1, 3100, 2400, 30, 90, 140};
	AppState appState = AppState::IDLING;
	
	bool camera_is_initialized;
	bool imgproc_is_initialized;	bool display_is_initialized;

	std::condition_variable cam_cv;
	std::mutex cam_mtx;
	ImageData<uint8_t> camIG, camIR;

	std::map<ImageType, ImageData<uint8_t>> img;

	std::vector<Position> bead_positions;
	std::mutex mtx_bp;

	// Subscribers of the images
	std::map<ImageType, std::vector<sockpp::inet_address>> img_subs {
			{ImageType::RAW_G, {}},
			{ImageType::RAW_R, {}},
			{ImageType::BACKPROP_G, {}},
			{ImageType::BACKPROP_R, {}}
		};

	// Subscribers of the coordinates of the tracked objects
	std::vector<sockpp::inet_address> coords_subs;		

	BeadTracker beadTracker;
	
    // Constructor
    AppData();

    // Member methods
	void startTheApp();
	void stopTheApp();
	bool waitTillState(AppState appState, useconds_t useconds = 100);
	bool waitTillAppIsInitialized(useconds_t useconds = 100);
	bool appStateIs(AppState appState);
	void appStateSet(AppState appState);
	void exitTheApp();

	// Image subscribe utility functions
	void removeImageSubs(sockpp::inet_address);
	void removeImageSubs(sockpp::inet_address, ImageType);
	void addImageSubs(sockpp::inet_address, ImageType);

	// Coordinates subscribe utility functions
	void removeCoordsSubs(sockpp::inet_address);
	void addCoordsSubs(sockpp::inet_address);

	void print();

	int get_area();

private:
	void addSubs(std::vector<sockpp::inet_address>&, sockpp::inet_address);
	void removeSubs(std::vector<sockpp::inet_address>&, sockpp::inet_address);
};

#endif