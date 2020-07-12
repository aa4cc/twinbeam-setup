#include <signal.h>
#include <thread>
#include "AppData.h"
#include "argpars.h"
#include "network.h"
#include "keyboard_thread.h"
#include "display_thread.h"
#include "imgproc_thread.h"
#include "camera_thread.h"

using namespace std;

int main(int argc, char* argv[]){
	static AppData appData;

	Options::parse(appData, argc, argv);

	// register signal SIGINT signal handler  
	struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = [](int value) { appData.exitTheApp(); };
    sa.sa_flags = 0;// not SA_RESTART!;
    sigaction(SIGINT, &sa, NULL);
	
	if(Options::debug){
		printf("DEBUG: Initial settings:");
		appData.print();
	}
	if (Options::show) {
		appData.appStateSet(AppData::AppState::INITIALIZING);
	}

	thread camera_thr (camera_thread, std::ref(appData));
	thread imgproc_thr (imgproc_thread, std::ref(appData));
	thread display_thr (display_thread, std::ref(appData));
	thread network_thr (network_thread, std::ref(appData));
	if (Options::keyboard) {
		thread keyboard_thr (keyboard_thread, std::ref(appData));
		keyboard_thr.join();
	}
	
	camera_thr.join();
	imgproc_thr.join();
	display_thr.join();
	network_thr.join();

	return 0;
}
