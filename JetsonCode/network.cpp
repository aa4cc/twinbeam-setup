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
	printf("INFO: network_thread: started\n");

	std::string text;
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
		if (client == -1) continue;

		appData.set_connected(true);
		cout << "INFO: Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;

		while(appData.connected && !appData.appStateIs(AppData::AppState::EXITING)){
			int msg_len = recv(client, buf, BUFSIZE - 1, 0);

			if (msg_len == -1)	continue;

			if(Options::debug)
				printf("DEBUG: Received %d bytes.\n", msg_len);

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
						printf("WARN: Can't change settings while the loop is running\n");
					else{
						memcpy(appData.values, buf+1, sizeof(uint32_t)*STG_NUMBER_OF_SETTINGS);
						appData.print();
						printf("INFO: Changed settings\n");
					}
					break;
				case MessageType::DISCONNECT:
					appData.set_connected(false);
					break;
				case MessageType::REQUEST:
					appData.set_requested_image(true);
					appData.set_requested_type(RequestType::BACKPROPAGATED);
					break;
				case MessageType::REQUEST_RAW_G:
					appData.set_requested_image(true);
					appData.set_requested_type(RequestType::RAW_G);
					break;
				case MessageType::REQUEST_RAW_R:
					appData.set_requested_image(true);
					appData.set_requested_type(RequestType::RAW_R);
					break;
				case MessageType::COORDS:
					appData.set_requested_coords(true);
					break;
				case MessageType::COORDS_CLOSEST:
					appData.saveReceivedBeadPos( ((uint32_t*)(buf+sizeof(uint8_t)))[0],  (uint16_t*)(buf+sizeof(uint8_t)+sizeof(uint32_t)) );
					appData.set_requested_coords_closest(true);
					break;
				case MessageType::HELLO:
					send(client, "Hello!",7,0);
					break;
			} 
		}
		close(client);
	}
	close(mainSocket);

	printf("INFO: network_thread: ended\n");
}

void datasend_thread(AppData& appData){
	printf("INFO: datasend_thread: started\n");
	uint8_t coords_buffer[sizeof(uint32_t) + 2*MAX_NUMBER_BEADS*sizeof(uint16_t)];

	while(!appData.appStateIs(AppData::AppState::EXITING)){
		while(appData.appStateIs(AppData::AppState::RUNNING) && appData.connected) {
			if(appData.requested_image){
				ImageData<uint8_t> temp_img(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);

				switch (appData.requested_type){
					case RequestType::BACKPROPAGATED:
						appData.G_backprop.copyTo(temp_img);
						break;
					case RequestType::RAW_G:
						appData.G.copyTo(temp_img);
						break;
					case RequestType::RAW_R:
						appData.R.copyTo(temp_img);
						break;
				}	

				send(client, temp_img.hostPtr(true), sizeof(uint8_t)*appData.get_area(), 0);
				printf("INFO: Image sent.\n");
				appData.set_requested_image(false);
			}

			if(appData.requested_coords && !appData.sent_coords) {
				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);
					((uint32_t*)coords_buffer)[0] = appData.bead_count;
					memcpy(coords_buffer+sizeof(uint32_t), appData.bead_positions, 2*appData.bead_count*sizeof(uint16_t));
					send(client, coords_buffer, sizeof(uint32_t) + 2*appData.bead_count*sizeof(uint16_t), 0);
				}

				printf("INFO: Sent the found locations\n");

				appData.set_requested_coords(false);
				appData.set_sent_coords(true);
			}

			if(appData.requested_coords_closest && !appData.sent_coords) {
				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp);			
					// Store the number of beads to be sent to the buffer		
					((uint32_t*)coords_buffer)[0] = appData.bead_count_received;
					uint16_t* coords_buffer_pos = (uint16_t*)(coords_buffer+sizeof(uint32_t)); 

					// Iterate over all the received positions and find the closest measured position in appData.bead_positions
					for(int i=0; i<appData.bead_count_received; i++) {
						int min_j = -1;
						int min_dist = INT_MAX;
						for(int j=0; j<appData.bead_count; j++) {
							int dist = SQUARE((int)appData.bead_positions[2*j]-(int)appData.bead_positions_received[2*i]) + SQUARE((int)appData.bead_positions[2*j+1]-(int)appData.bead_positions_received[2*i+1]);
							if (dist < min_dist) {
								min_dist = dist;
								min_j = j;
							}
							// printf("INFO: (%d, %d) dist: %d\n", i, j, dist);
						}
						// printf("INFO: %d <- %d | dist: %d\n", i, min_j, min_dist);

						// Store the closest bead position to the buffer
						coords_buffer_pos[2*i] = appData.bead_positions[2*min_j];
						coords_buffer_pos[2*i+1] = appData.bead_positions[2*min_j+1];
					}
					send(client, coords_buffer, sizeof(uint32_t) + 2*appData.bead_count_received*sizeof(uint16_t), 0);

					appData.set_requested_coords_closest(false);
					appData.set_sent_coords(true);
				}

				printf("INFO: Sent the found locations\n");

				appData.set_requested_coords(false);
				appData.set_sent_coords(true);
			}
		}
		usleep(100);
	}

	printf("INFO: datasend_thread: ended\n");
}