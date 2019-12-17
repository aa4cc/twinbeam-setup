/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */

#ifndef APPDATA_H
#define APPDATA_H

#include <mutex> 
#include <condition_variable>
#include "Definitions.h"
#include "ImageData.h"

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

	int values[STG_NUMBER_OF_SETTINGS] = {1200, 1200, 1352, 504, 5000000, 50, 1, 3100, 2400, 30, 80};
	AppState appState = AppState::IDLING;
	
	bool camera_is_initialized;
	bool imgproc_is_initialized;
	bool display_is_initialized;

	bool connected;

	std::condition_variable cam_cv;
	std::mutex cam_mtx;
	ImageData<uint8_t> camIG, camIR;

	ImageData<uint8_t> G, R, G_backprop;
	uint16_t bead_positions[2*MAX_NUMBER_BEADS];
	uint32_t bead_count;
	uint16_t bead_positions_received[2*MAX_NUMBER_BEADS];
	uint32_t bead_count_received;
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
	void saveReceivedBeadPos(uint32_t bead_count, uint16_t* bead_positions);
	void set_connected(const bool value);

	int get_area();
};

#endif