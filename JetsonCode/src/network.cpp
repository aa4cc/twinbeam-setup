/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 

#include <iostream>

#include <string>
#include <sstream>

#include <climits>
#include <thread>
#include <chrono>
#include <pthread.h>
#include "sockpp/tcp_acceptor.h"
#include "sockpp/version.h"
#include "network.h"
#include "Misc.h"

#define SQUARE(x) ((x)*(x))

using namespace std;

void client_thread(AppData& appData, sockpp::tcp_socket sock) {
	ssize_t msg_len;
	char buf[BUFSIZE];
	istringstream json_config_i;
	string json_config;
	uint8_t coords_buffer[sizeof(uint32_t) + 4*MAX_NUMBER_BEADS*sizeof(uint16_t)]; // Coords buffer can store coordinates of up to MAX_NUMBER_BEADS beads for both color channels
	uint32_t* beadCountP;
	uint16_t *beadCountP_G, *beadCountP_R;
	MessageType response;	

	if(appData.params.rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 45;
		
		if(appData.params.debug) printf("INFO: network_client_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) {
			cerr << "WARNING: setting the priority of network_client_thread failed." << endl;
		}
	}		

	if(appData.params.debug) cout << "INFO: Got a connection from " << sock.peer_address() << endl;

	// Set a timeout for the socket so that the app can be killed from the outside
	if(!sock.read_timeout(chrono::seconds(1))) cerr << "ERROR: setting timeout on TCP stream failed: " << sock.last_error_str() << endl;

	while(!appData.appStateIs(AppData::AppState::EXITING)){
		msg_len = sock.read(buf, sizeof(buf));

		// If no message was received within one second, continue
		if (msg_len == -1)	continue;

		// If the connection was closed, break the loop
		if (msg_len == 0)	break;

		if(appData.params.debug) printf("DEBUG: Received %d bytes. MessageType: %c \n", (int)msg_len, buf[0]);

		response = parseMessage(buf);
		switch(response){
			case MessageType::START:
				appData.startTheApp();
				break;
			case MessageType::STOP:
				appData.stopTheApp();
				break;
			case MessageType::SETTINGS:
				if (msg_len == 1) {
					// Send back the current JSON config
					json_config = appData.params.getJSONConfigString();
					memcpy(buf, json_config.c_str(), json_config.length());
					sock.write_n(buf, json_config.length());

					if(appData.params.debug) printf("INFO: Sending the config params.\n");
				} else {
					if(appData.params.debug) printf("INFO: Received new config params.\n");
					// Receive new config params 	
					// End the string
					buf[msg_len] = '\0';
					// Update the istream
					json_config_i.str(buf+1);

					if (appData.params.debug) {
						cout << "New config parameters:" << endl;
						cout << json_config_i.str() << endl;
					}

					// parse the istream
					appData.params.parseJSONIStream(json_config_i);
				}
				break;
			case MessageType::IMG_REQUEST:
				if(!appData.appStateIs(AppData::AppState::RUNNING)) {
					cerr << "WARN: Image cannot be sent since the application is not running." << endl;
				} else {
					ImageData<uint8_t> temp_img(appData.params.img_width, appData.params.img_height);
					ImageData<uint8_t> temp_img2(appData.params.img_width, appData.params.img_height);
					switch(buf[1]) {
						case 0:
							appData.img[ImageType::BACKPROP_G].copyToAsync(temp_img, 0);
							break;
						case 1:
							appData.img[ImageType::BACKPROP_R].copyToAsync(temp_img, 0);
							break;
						case 2:
							appData.img[ImageType::RAW_G].copyToAsync(temp_img, 0);
							break;
						case 3:
							appData.img[ImageType::RAW_R].copyToAsync(temp_img, 0);
							break;
						case 4:
							// Send both, the image from G and R channel
							appData.img[ImageType::BACKPROP_G].copyToAsync(temp_img, 0);
							appData.img[ImageType::BACKPROP_R].copyToAsync(temp_img2, 0);
							break;
						case 5:
							// Send both, the image from G and R channel
							appData.img[ImageType::RAW_G].copyToAsync(temp_img, 0);
							appData.img[ImageType::RAW_R].copyToAsync(temp_img2, 0);
							break;
						default:
							cerr << "ERROR: Unknown image type recieved with image request" << endl;
							break;
					}
					
					sock.write_n(temp_img.hostPtrAsync(0,true), sizeof(uint8_t)*appData.get_area());
					if(buf[1] == 4) {
						// Both RAW_G and RAW_R images are to be sent
						sock.write_n(temp_img2.hostPtrAsync(0,true), sizeof(uint8_t)*appData.get_area());
					}

					if(appData.params.debug) printf("INFO: Image sent.\n");
				}
				break;
			case MessageType::COORDS_G:
				beadCountP = (uint32_t*)coords_buffer;

				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp_G);
					*beadCountP = (uint32_t)appData.bead_positions_G.size();
					memcpy(coords_buffer+sizeof(uint32_t), appData.bead_positions_G.data(), 2*(*beadCountP)*sizeof(uint16_t));
				}

				sock.write_n(coords_buffer, sizeof(uint32_t) + 2*(*beadCountP)*sizeof(uint16_t));

				break;
			case MessageType::COORDS_R:
				beadCountP = (uint32_t*)coords_buffer;

				{ // Limit the scope of the mutex
					std::lock_guard<std::mutex> mtx_bp(appData.mtx_bp_R);
					*beadCountP = (uint32_t)appData.bead_positions_R.size();
					memcpy(coords_buffer+sizeof(uint32_t), appData.bead_positions_R.data(), 2*(*beadCountP)*sizeof(uint16_t));
				}

				sock.write_n(coords_buffer, sizeof(uint32_t) + 2*(*beadCountP)*sizeof(uint16_t));

				break;
			case MessageType::TRACKER:
				switch(buf[1]){
					case 'i': 
						{
							// Initialize the tracker
							// the next two bytes specify the number of objects to be tracked in the RAW_G
							// image, the following two bytes specify the number of objects to be tracked in the RAW_R
							// image, and rest of the message specifies the current position of the objects
							// to be tracked. The positions are stored in two bytes per coordinate (uint16)
							// and the x position of the first object is followed by the y position of
							// the first object, then x position of the second object follows and so on and
							// so on (i.e. x0, y0, x1, y1, x2, ....)

							if(!appData.appStateIs(AppData::AppState::RUNNING)) {
								cerr << "WARN: Bead trackers cannot be initialized since the application is not running." << endl;
							}

							// If none of the trackers is used, break this case
							if(!appData.params.beadsearch_G && !appData.params.beadsearch_R) {
								cerr << "ERROR: None of the bead trackers was turned on when the App was started." << endl;
								break;
							}

							uint16_t N_objects_G = *(uint16_t*)(buf+2); // Number of objects to be tracked in RAW_G
							uint16_t N_objects_R = *(uint16_t*)(buf+4); // Number of objects to be tracked in RAW_R

							uint16_t* positions_G = (uint16_t*)(buf+2+4);
							uint16_t* positions_R = (uint16_t*)(buf+2+4+2*N_objects_G*sizeof(uint16_t));

							// Clear the beadTracker - remove all the currently tracked objects from the tracker
							appData.beadTracker_G.clear();
							appData.beadTracker_R.clear();

							// Initialize the RAW_G tracker
							for(size_t i=0; i<N_objects_G; ++i) {
								appData.beadTracker_G.addBead({positions_G[2*i], positions_G[2*i+1]});
								if(appData.params.debug) printf("INFO: BeadTracker RAW_G: adding object at (%d, %d)\n", positions_G[2*i], positions_G[2*i+1]);
							}

							// Initialize the RAW_R tracker
							for(size_t i=0; i<N_objects_R; ++i) {
								appData.beadTracker_R.addBead({positions_R[2*i], positions_R[2*i+1]});
								if(appData.params.debug) printf("INFO: BeadTracker RAW_R: adding object at (%d, %d)\n", positions_R[2*i], positions_R[2*i+1]);
							}	

							// Just in case the the trackers were not initialized before, switch on the corresponding flag
							if(N_objects_G > 0) appData.params.beadsearch_G = true;
							if(N_objects_R > 0) appData.params.beadsearch_R = true;
							break;
						}
					case 'r':
						{
							// Send the positions of the tracked objects
							// The first four bytes are reserved for frame_id.
							// The following four bytes specify the number of tracked objects
							// and rest of the message specify the current position of the tracked objects
							// The positions are stored in two bytes per coordinate (uint16)
							// and the x position of the first object is followed by the y position of
							// the first object, then x position of the second object follows and so on and
							// so on (i.e. x0, y0, x1, y1, x2, ....)

							uint32_t* frame_id = (uint32_t*)coords_buffer;
							beadCountP_G = (uint16_t*)(coords_buffer + sizeof(uint32_t));
							beadCountP_R = (uint16_t*)(coords_buffer + sizeof(uint32_t) + sizeof(uint16_t));

							// Store the frame id to the message
							*frame_id = appData.frame_id;

							// RAW_G tracker
							const vector<Position>& bp_G = appData.beadTracker_G.getBeadPositions();
							if(appData.params.beadsearch_G) {
								// Store the number of tracked objects
								*beadCountP_G = (uint16_t)bp_G.size();
								// Copy the tracked positions in RAW_G to the coords_buffer
								memcpy(coords_buffer+sizeof(uint32_t)+2*sizeof(uint16_t), bp_G.data(), 2*(*beadCountP_G)*sizeof(uint16_t));
							} else {
								*beadCountP_G = 0;
							}

							// RAW_R tracker
							const vector<Position>& bp_R = appData.beadTracker_R.getBeadPositions();
							if(appData.params.beadsearch_R) {
								// Store the number of tracked objects
								*beadCountP_R = (uint16_t)bp_R.size();
								// Copy the tracked positions in RAW_R to the coords_buffer
								memcpy(coords_buffer + sizeof(uint32_t) + 2*sizeof(uint16_t) + 2*(*beadCountP_G)*sizeof(uint16_t), bp_R.data(), 2*(*beadCountP_R)*sizeof(uint16_t));
							} else {
								*beadCountP_R = 0;
							}

							// Send coords_buffer to the client
							sock.write_n(coords_buffer,  sizeof(uint32_t) + 2*sizeof(uint16_t) + 2*(*beadCountP_G + *beadCountP_R)*sizeof(uint16_t));
							break;
						}
				}
				break;
			case MessageType::IMG_SUBSCRIBE:
				if(msg_len == 1) {
					// Empty packet means: unsubscribe me from the the list
					appData.removeImageSubs(sock.peer_address());

					if(appData.params.debug) cout << "INFO: unsubscribing from the image (ignore the port number) " << sock.peer_address() << endl;
				} else if(msg_len == 3) {
					// the packet two more bytes which specify the port to which the images are supposed to be sent
					uint16_t port = ((uint16_t*)(buf+1))[0];

					sockpp::inet_address inaddr(sock.peer_address().address(), port);
					// If the list already contains the IP address of this subscriber, delete it from the list
					appData.addImageSubs(inaddr, ImageType::BACKPROP_G);

					if(appData.params.debug) cout << "INFO: subscribing to the BACKPROP_G image publisher from " << inaddr << endl;
				} else {
					cerr << "ERROR: The subscription was not succseful as the received packet was not of the correct length." << endl;
				}
				break;
			case MessageType::COORDS_SUBSCRIBE:
				if(msg_len == 1) {
					// Empty packet means: unsubscribe me from the the list
					appData.removeCoordsSubs(sock.peer_address());

					if(appData.params.debug) cout << "INFO: unsubscribing from the coordinate publisher (ignore the port number) " << sock.peer_address() << endl;
				} else if(msg_len == 3) {
					// the packet two more bytes which specify the port to which the images are supposed to be sent
					uint16_t port = ((uint16_t*)(buf+1))[0];

					sockpp::inet_address inaddr(sock.peer_address().address(), port);
					// If the list already contains the IP address of this subscriber, delete it from the list
					appData.addCoordsSubs(inaddr);

					if(appData.params.debug) cout << "INFO: subscribing to the coordinate UDP publisher from " << inaddr << endl;
				} else {
					cerr << "ERROR: The subscription was not succseful as the received packet was not of the correct length." << endl;
				}
				break;
			case MessageType::HELLO:
				sock.write_n("Hello!",7);
				break;
		} 
	}

	// Just in case this host has been subscribed to image or coordinate publisher, remove it from the list
	appData.removeImageSubs(sock.peer_address());
	appData.removeCoordsSubs(sock.peer_address());

	if(appData.params.debug) cout << "INFO: Closing the connection from " << sock.peer_address() << endl;
}

void network_thread(AppData& appData){
	if(appData.params.debug) cout << "INFO: network_thread: started" << endl;

	sockpp::tcp_acceptor acc(appData.params.tcp_port);
	if (!acc) {
		cerr << "ERROR: Couldn't open the socket." << endl;
		return;
	}

	// Set the acceptor to te non-blocking mode
	acc.set_non_blocking();

	while(!appData.appStateIs(AppData::AppState::EXITING)){		
		sockpp::inet_address peer;

		// Accept a new client connection - since the non-blocking mode is used,
		// this function return immidiately a socket which is valid only if there
		// has been already a client waiting for connection. If not, the socket
		// is invalid, we wait 100 ms and check again for the awaiting clients. 
		sockpp::tcp_socket sock = acc.accept(&peer);

		if (sock) {
			// Create a thread and transfer the new stream to it.
			thread client_thr (client_thread, std::ref(appData), std::move(sock));
			client_thr.detach();
		}

		this_thread::sleep_for(chrono::milliseconds(100));
	}

	if(appData.params.debug) cout << "INFO: network_thread: ended" << endl;
}
