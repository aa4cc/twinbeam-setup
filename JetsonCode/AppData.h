#ifndef APPDATA_H
#define APPDATA_H

#include <mutex> 
#include <atomic>
#include "Definitions.h"
#include "ImageData.h"

class CameraImgI{
    public:
    std::atomic<unsigned int> img_produced;
    std::atomic<unsigned int> img_processed;
    ImageData<uint8_t> G;
    ImageData<uint8_t> R;

    CameraImgI() : img_produced{0}, img_processed{0} { };
};

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

	int values[STG_NUMBER_OF_SETTINGS] = {1200, 1200, 1352, 596, 5000000, 50, 1, 3100, 2400, 30, 80};
	AppState appState = AppState::IDLING;
	
	bool camera_is_initialized;
	bool imgproc_is_initialized;
	bool display_is_initialized;

	bool connected;
	bool sent_coords;
	RequestType requested_type;
	bool requested_image;
	bool requested_coords;

	CameraImgI camI;
	ImageData<uint8_t> G, R, G_backprop;
	uint16_t bead_positions[2*MAX_NUMBER_BEADS];
	uint32_t bead_count;
	std::mutex mtx_bp;
	
    // Construction
    AppData();

    // Member methods
	void startTheApp();
	void stopTheApp();
	bool waitTillState(AppState appState, useconds_t useconds = 100);
	bool waitTillAppIsInitialized(useconds_t useconds = 100);
	bool appStateIs(AppState appState);
	void appStateSet(AppState appState);
	void exitTheApp();


	void print();
	void set_connected(const bool value);
	void set_sent_coords(const bool value);
	void set_requested_type(const RequestType value);
	void set_requested_image(const bool value);
	void set_requested_coords(const bool value);

	int get_area();
};

#endif