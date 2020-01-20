/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
#include <climits>
#include <unistd.h>
#include <thread>
#include <pthread.h>
#include "network.h"
#include "Misc.h"
#include "argpars.h"

#define SQUARE(x) ((x)*(x))

using namespace std;

void client_thread(AppData& appData, int client, sockaddr_in clientInfo) {
	char buf[BUFSIZE];
	uint8_t coords_buffer[sizeof(uint32_t) + 2*MAX_NUMBER_BEADS*sizeof(uint16_t)];
	uint32_t* beadCountP;
	MessageType response;	

	if(Options::rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 30;
		
		if(Options::debug) printf("INFO: network_client_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) {
			fprintf(stderr, "WARNING: setting the priority of network_client_thread failed.\n");
		}
	}		

	if(Options::debug) cout << "INFO: Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
	while(!appData.appStateIs(AppData::AppState::EXITING)){
		int msg_len = recv(client, buf, BUFSIZE - 1, 0);

		// If no message was received within one second, continue
		if (msg_len == -1)	continue;

		// If the connection was closed, break the loop
		if (msg_len == 0)	break;

		if(Options::debug) printf("DEBUG: Received %d bytes. MessageType: %c \n", msg_len, buf[0]);

		response = parseMessage(buf);
		switch(response){
			case MessageType::START:
				appData.startTheApp();
				break;
			case MessageType::STOP:
				appData.stopTheApp();
				break;
			case MessageType::SETTINGS:
				if(appData.appStateIs(AppData::AppState::RUNNING) || appData.appStateIs(AppData::AppState::INITIALIZING))
					fprintf(stderr, "WARN: Can't change settings while the loop is running\n");
				else{
					memcpy(appData.values, buf+1, sizeof(uint32_t)*STG_NUMBER_OF_SETTINGS);
					appData.print();
					if(Options::debug) printf("INFO: Changed settings\n");
				}
				break;
			case MessageType::REQUEST:
				if(!appData.appStateIs(AppData::AppState::RUNNING)) {
					fprintf(stderr, "WARN: Image cannot be sent since the application is not running.\n");
				} else {
					ImageData<uint8_t> temp_img(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
					appData.G_backprop.copyTo(temp_img);

					send(client, temp_img.hostPtr(true), sizeof(uint8_t)*appData.get_area(), 0);
					if(Options::debug) printf("INFO: Image sent.\n");
				}
				break;
			case MessageType::REQUEST_RAW_G:
				if(!appData.appStateIs(AppData::AppState::RUNNING)) {
					fprintf(stderr, "WARN: Image cannot be sent since the application is not running.\n");
				} else {
					ImageData<uint8_t> temp_img(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
					appData.G.copyTo(temp_img);

					send(client, temp_img.hostPtr(true), sizeof(uint8_t)*appData.get_area(), 0);
					if(Options::debug) printf("INFO: Image sent.\n");
				}
				break;
			case MessageType::REQUEST_RAW_R:
				if(!appData.appStateIs(AppData::AppState::RUNNING)) {
					fprintf(stderr, "WARN: The positions cannot be sent since the application is not running.\n");
				} else {
					ImageData<uint8_t> temp_img(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
					appData.R.copyTo(temp_img);

					send(client, temp_img.hostPtr(true), sizeof(uint8_t)*appData.get_area(), 0);
					if(Options::debug) printf("INFO: Image sent.\n");
				}
				break;
			case MessageType::COORDS:
				beadCountP = (uint32_t*)coords_buffer;

				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
					*beadCountP = (uint32_t)appData.bead_positions.size();
					memcpy(coords_buffer+sizeof(uint32_t), appData.bead_positions.data(), 2*(*beadCountP)*sizeof(uint16_t));
				}

				send(client, coords_buffer, sizeof(uint32_t) + 2*(*beadCountP)*sizeof(uint16_t), 0);

				break;
			case MessageType::COORDS_CLOSEST:
				if(!appData.appStateIs(AppData::AppState::RUNNING)) {
					fprintf(stderr, "WARN: The positions cannot be sent since the application is not running.\n");
				} else {
					uint32_t bead_count_received = ((uint32_t*)(buf+sizeof(uint8_t)))[0];
					uint16_t *bead_positions_received = (uint16_t*)(buf+sizeof(uint8_t)+sizeof(uint32_t));
				
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
					// Store the number of beads to be sent to the buffer		
					((uint32_t*)coords_buffer)[0] = bead_count_received;
					uint16_t* coords_buffer_pos = (uint16_t*)(coords_buffer+sizeof(uint32_t)); 

					// Iterate over all the received positions and find the closest measured position in appData.bead_positions
					for(int i=0; i<bead_count_received; i++) {
						// printf("\t %d -> ", i);
						int min_j = -1;
						int min_dist = INT_MAX;
						for(auto &b : appData.bead_positions) {
							int dist = SQUARE((int)b.x-(int)bead_positions_received[2*i]) + SQUARE((int)b.y-(int)bead_positions_received[2*i+1]);
							if (dist < min_dist) {
								min_dist = dist;
								// Store the closest bead position to the buffer
								coords_buffer_pos[2*i] = b.x;
								coords_buffer_pos[2*i+1] = b.y;		
							}
						}
					}
					send(client, coords_buffer, sizeof(uint32_t) + 2*bead_count_received*sizeof(uint16_t), 0);

					if(Options::debug) printf("INFO: Closest bead positions sent\n");
				}
				break;
			case MessageType::TRACKER:
				switch(buf[1]){
					case 'i': 
						{
							// Initialize the tracker
							// the next four bytes specify the number of objects to be tracked
							// and rest of the message specify the current position of the objects
							// to be tracked. The positions are stored in two bytes per coordinate (uint16)
							// and the x position of the first object is followed by the y position of
							// the first object, then x position of the second object follows and so on and
							// so on (i.e. x0, y0, x1, y1, x2, ....)

							// Clear the beadTracker - remove all the currently tracked objects from the tracker
							appData.beadTracker.clear();

							uint32_t N_objects = *(uint32_t*)(buf+2);
							uint16_t* positions = (uint16_t*)(buf+2+sizeof(uint32_t));
							for(size_t i=0; i<N_objects; ++i) {
								appData.beadTracker.addBead({positions[2*i], positions[2*i+1]});
								if(Options::debug) printf("INFO: BeadTracker: adding object at (%d, %d)\n", positions[2*i], positions[2*i+1]);
							}
							break;
						}
					case 'r':
						{
							// Send the positions of the tracked objects
							// the first four bytes specify the number of tracked objects
							// and rest of the message specify the current position of the tracked objects
							// The positions are stored in two bytes per coordinate (uint16)
							// and the x position of the first object is followed by the y position of
							// the first object, then x position of the second object follows and so on and
							// so on (i.e. x0, y0, x1, y1, x2, ....)

							beadCountP = (uint32_t*)coords_buffer;
							const vector<Position>& bp = appData.beadTracker.getBeadPositions();

							// Store the number of tracked objects
							*beadCountP = (uint32_t)bp.size();

							// Copy the tracked positions to the coords_buffer
							memcpy(coords_buffer+sizeof(uint32_t), bp.data(), 2*(*beadCountP)*sizeof(uint16_t));

							// Send coords_buffer to the client
							send(client, coords_buffer, sizeof(uint32_t) + 2*(*beadCountP)*sizeof(uint16_t), 0);
							break;
						}
				}
				break;
			case MessageType::HELLO:
				send(client, "Hello!",7,0);
				break;
		} 
	}
	if(Options::debug) cout << "INFO: Closing the connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
	close(client);
}

void network_thread(AppData& appData){
	if(Options::debug) printf("INFO: network_thread: started\n");

	sockaddr_in sockName;
	int mainSocket;	
	
	mainSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(mainSocket == -1)
		fprintf(stderr, "ERROR: Couldn't create socket!\n");
	sockName.sin_family = AF_INET;
	sockName.sin_port =	htons(PORT);
	// sockName.sin_addr.s_addr = INADDR_ANY;
	sockName.sin_addr.s_addr = inet_addr("147.32.86.177");

	// Allow reusing the port
	int yes = 1;
	if ( setsockopt(mainSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1 )
	{
		fprintf(stderr, "Setsocket failed!\n");
		appData.exitTheApp();
		return;
	}
	
	// Set the time for the rev() to 1 second
	struct timeval tv;
	tv.tv_sec = 1;
	tv.tv_usec = 0;
	if ( setsockopt(mainSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv) == -1 )
	{
		fprintf(stderr, "Setsocket failed!\n");
		appData.exitTheApp();
		return;
	}	

	bind(mainSocket, (sockaddr*)&sockName, sizeof(sockName));

	listen(mainSocket, 10000000);
	sockaddr_in clientInfo; 
	socklen_t addrlen = sizeof(clientInfo);;
	int client;
	while(!appData.appStateIs(AppData::AppState::EXITING)){		
		
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		if (client < 1) continue;

		thread client_thr (client_thread, std::ref(appData), client, clientInfo);
		client_thr.detach();
	}
	close(mainSocket);

	if(Options::debug) printf("INFO: network_thread: ended\n");
}
