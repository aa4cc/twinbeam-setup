#include <signal.h>
#include <thread>
#include "AppData.h"
#include "network.h"
#include "keyboard_thread.h"
#include "display_thread.h"
#include "imgproc_thread.h"
#include "camera_thread.h"

using namespace std;

int main(int argc, char* argv[]){
	static AppData appData;

	appData.params.parseJSONConfigFile("config.json");
	appData.params.parseCmdlineArgs(argc, argv);

	if ( appData.params.debug ) {
		appData.params.print();
	}
	
	// register signal SIGINT and SIGTERM signal handler  
	struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = [](int value) { appData.exitTheApp(); };
    sa.sa_flags = 0;// not SA_RESTART!;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
	
	if (appData.params.show) {
		appData.appStateSet(AppData::AppState::INITIALIZING);
	}

	thread camera_thr (camera_thread, std::ref(appData));
	thread imgproc_thr (imgproc_thread, std::ref(appData));
	thread display_thr (display_thread, std::ref(appData));
	thread network_thr (network_thread, std::ref(appData));
	if (appData.params.keyboard) {
		thread keyboard_thr (keyboard_thread, std::ref(appData));
		keyboard_thr.join();
	}
	
	camera_thr.join();
	imgproc_thr.join();
	display_thr.join();
	network_thr.join();

	return 0;
}
