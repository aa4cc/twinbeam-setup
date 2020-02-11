/**
 * @author  Viktor-Adam Koropecky
 * @author  Martin Gurtner
 */
 

#include <iostream>
#include <climits>
#include <thread>
#include <chrono>
#include <pthread.h>
#include "sockpp/tcp_acceptor.h"
#include "sockpp/version.h"
#include "network.h"
#include "Misc.h"
#include "argpars.h"

#define SQUARE(x) ((x)*(x))

using namespace std;

void client_thread(AppData& appData, sockpp::tcp_socket sock) {
	ssize_t msg_len;
	char buf[BUFSIZE];
	uint8_t coords_buffer[sizeof(uint32_t) + 2*MAX_NUMBER_BEADS*sizeof(uint16_t)];
	uint32_t* beadCountP;
	MessageType response;	

	if(Options::rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 45;
		
		if(Options::debug) printf("INFO: network_client_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) {
			cerr << "WARNING: setting the priority of network_client_thread failed." << endl;
		}
	}		

	if(Options::debug) cout << "INFO: Got a connection from " << sock.peer_address() << endl;

	// Set a timeout for the socket so that the app can be killed from the outside
	if(!sock.read_timeout(chrono::seconds(1))) cerr << "ERROR: setting timeout on TCP stream failed: " << sock.last_error_str() << endl;

	while(!appData.appStateIs(AppData::AppState::EXITING)){
		msg_len = sock.read(buf, sizeof(buf));

		// If no message was received within one second, continue
		if (msg_len == -1)	continue;

		// If the connection was closed, break the loop
		if (msg_len == 0)	break;

		if(Options::debug) printf("DEBUG: Received %d bytes. MessageType: %c \n", (int)msg_len, buf[0]);

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
					cerr << "WARN: Can't change settings while the loop is running" << endl;
				else{
					memcpy(appData.values, buf+1, sizeof(uint32_t)*STG_NUMBER_OF_SETTINGS);
					appData.print();
					if(Options::debug) printf("INFO: Changed settings\n");
				}
				break;
			case MessageType::IMG_REQUEST:
				if(!appData.appStateIs(AppData::AppState::RUNNING)) {
					cerr << "WARN: Image cannot be sent since the application is not running." << endl;
				} else {
					ImageData<uint8_t> temp_img(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
					ImageData<uint8_t> temp_img2(appData.values[STG_WIDTH], appData.values[STG_HEIGHT]);
					switch(buf[1]) {
						case 0:
							appData.img[ImageType::BACKPROP_G].copyTo(temp_img);
							break;
						case 1:
							appData.img[ImageType::BACKPROP_R].copyTo(temp_img);
							break;
						case 2:
							appData.img[ImageType::RAW_G].copyTo(temp_img);
							break;
						case 3:
							appData.img[ImageType::RAW_R].copyTo(temp_img);
							break;
						case 4:
							// Send both, the image from G and R channel
							appData.img[ImageType::RAW_G].copyTo(temp_img);
							appData.img[ImageType::RAW_R].copyTo(temp_img2);
							break;
						default:
							cerr << "ERROR: Unknown image type recieved with image request" << endl;
							break;
					}
					
					sock.write_n(temp_img.hostPtr(true), sizeof(uint8_t)*appData.get_area());
					if(buf[1] == 4) {
						// Both RAW_G and RAW_R images are to be sent
						sock.write_n(temp_img2.hostPtr(true), sizeof(uint8_t)*appData.get_area());
					}

					if(Options::debug) printf("INFO: Image sent.\n");
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
							// the next four bytes specify the number of objects to be tracked
							// and rest of the message specify the current position of the objects
							// to be tracked. The positions are stored in two bytes per coordinate (uint16)
							// and the x position of the first object is followed by the y position of
							// the first object, then x position of the second object follows and so on and
							// so on (i.e. x0, y0, x1, y1, x2, ....)

							// @TODO currently, both trackers are initialized by the same bead positions

							// Clear the beadTracker - remove all the currently tracked objects from the tracker
							appData.beadTracker_G.clear();
							appData.beadTracker_R.clear();

							uint32_t N_objects = *(uint32_t*)(buf+2);
							uint16_t* positions = (uint16_t*)(buf+2+sizeof(uint32_t));
							for(size_t i=0; i<N_objects; ++i) {
								appData.beadTracker_G.addBead({positions[2*i], positions[2*i+1]});
								appData.beadTracker_R.addBead({positions[2*i], positions[2*i+1]});
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
							const vector<Position>& bp = appData.beadTracker_G.getBeadPositions();

							// Store the number of tracked objects
							*beadCountP = (uint32_t)bp.size();

							// Copy the tracked positions to the coords_buffer
							memcpy(coords_buffer+sizeof(uint32_t), bp.data(), 2*(*beadCountP)*sizeof(uint16_t));

							// Send coords_buffer to the client
							sock.write_n(coords_buffer, sizeof(uint32_t) + 2*(*beadCountP)*sizeof(uint16_t));
							break;
						}
				}
				break;
			case MessageType::IMG_SUBSCRIBE:
				if(msg_len == 1) {
					// Empty packet means: unsubscribe me from the the list
					appData.removeImageSubs(sock.peer_address());

					if(Options::debug) cout << "INFO: unsubscribing from the image (ignore the port number) " << sock.peer_address() << endl;
				} else if(msg_len == 3) {
					// the packet two more bytes which specify the port to which the images are supposed to be sent
					uint16_t port = ((uint16_t*)(buf+1))[0];

					sockpp::inet_address inaddr(sock.peer_address().address(), port);
					// If the list already contains the IP address of this subscriber, delete it from the list
					appData.addImageSubs(inaddr, ImageType::BACKPROP_G);

					if(Options::debug) cout << "INFO: subscribing to the BACKPROP_G image publisher from " << inaddr << endl;
				} else {
					cerr << "ERROR: The subscription was not succseful as the received packet was not of the correct length." << endl;
				}
				break;
			case MessageType::COORDS_SUBSCRIBE:
				if(msg_len == 1) {
					// Empty packet means: unsubscribe me from the the list
					appData.removeCoordsSubs(sock.peer_address());

					if(Options::debug) cout << "INFO: unsubscribing from the coordinate publisher (ignore the port number) " << sock.peer_address() << endl;
				} else if(msg_len == 3) {
					// the packet two more bytes which specify the port to which the images are supposed to be sent
					uint16_t port = ((uint16_t*)(buf+1))[0];

					sockpp::inet_address inaddr(sock.peer_address().address(), port);
					// If the list already contains the IP address of this subscriber, delete it from the list
					appData.addCoordsSubs(inaddr);

					if(Options::debug) cout << "INFO: subscribing to the coordinate UDP publisher from " << inaddr << endl;
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

	if(Options::debug) cout << "INFO: Closing the connection from " << sock.peer_address() << endl;
}

void network_thread(AppData& appData){
	if(Options::debug) cout << "INFO: network_thread: started" << endl;

	sockpp::tcp_acceptor acc(Options::tcp_port);
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

	if(Options::debug) cout << "INFO: network_thread: ended" << endl;
}
