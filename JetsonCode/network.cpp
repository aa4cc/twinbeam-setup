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
#include "network.h"
#include "Misc.h"
#include "argpars.h"

#define SQUARE(x) ((x)*(x))

using namespace std;

int client;
void network_thread(AppData& appData){
	if(Options::debug) printf("INFO: network_thread: started\n");

	uint8_t coords_buffer[sizeof(uint32_t) + 2*MAX_NUMBER_BEADS*sizeof(uint16_t)];
	uint32_t* beadCount;

	sockaddr_in sockName;
	sockaddr_in clientInfo; 
	int mainSocket;
	char buf[BUFSIZE];
	socklen_t addrlen;
	MessageType response;
	
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
	while(!appData.appStateIs(AppData::AppState::EXITING)){
		
		addrlen = sizeof(clientInfo);
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		if (client < 1) continue;

		appData.set_connected(true);
		if(Options::debug) cout << "INFO: Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;

		while(appData.connected && !appData.appStateIs(AppData::AppState::EXITING)){
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
				case MessageType::DISCONNECT:
					appData.set_connected(false);
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
					beadCount = (uint32_t*)coords_buffer;

					{ // Limit the scope of the mutex
						std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
						*beadCount = appData.bead_count;
						memcpy(coords_buffer+sizeof(uint32_t), appData.bead_positions, 2*(*beadCount)*sizeof(uint16_t));
					}

					send(client, coords_buffer, sizeof(uint32_t) + 2*(*beadCount)*sizeof(uint16_t), 0);

					break;
				case MessageType::COORDS_CLOSEST:
					if(!appData.appStateIs(AppData::AppState::RUNNING)) {
						fprintf(stderr, "WARN: The positions cannot be sent since the application is not running.\n");
					} else {
						appData.saveReceivedBeadPos( ((uint32_t*)(buf+sizeof(uint8_t)))[0],  (uint16_t*)(buf+sizeof(uint8_t)+sizeof(uint32_t)) );
					
						std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);			
						// Store the number of beads to be sent to the buffer		
						((uint32_t*)coords_buffer)[0] = appData.bead_count_received;
						uint16_t* coords_buffer_pos = (uint16_t*)(coords_buffer+sizeof(uint32_t)); 

						// Iterate over all the received positions and find the closest measured position in appData.bead_positions
						for(int i=0; i<appData.bead_count_received; i++) {
							// printf("\t %d -> ", i);
							int min_j = -1;
							int min_dist = INT_MAX;
							for(int j=0; j<appData.bead_count; j++) {
								int dist = SQUARE((int)appData.bead_positions[2*j]-(int)appData.bead_positions_received[2*i]) + SQUARE((int)appData.bead_positions[2*j+1]-(int)appData.bead_positions_received[2*i+1]);
								if (dist < min_dist) {
									min_dist = dist;
									min_j = j;
								}
								// printf("\t(%d, %d): %d\n", appData.bead_positions[2*j], appData.bead_positions[2*j+1], dist);
							}
							// printf("\t %d <- %d | dist: %d\n", i, min_j, min_dist);

							// Store the closest bead position to the buffer
							coords_buffer_pos[2*i] = appData.bead_positions[2*min_j];
							coords_buffer_pos[2*i+1] = appData.bead_positions[2*min_j+1];
						}
						send(client, coords_buffer, sizeof(uint32_t) + 2*appData.bead_count_received*sizeof(uint16_t), 0);

						if(Options::debug) printf("INFO: Closest bead positions sent\n");
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
	close(mainSocket);

	if(Options::debug) printf("INFO: network_thread: ended\n");
}
