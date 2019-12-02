#include "cuda.h"
#include "cufft.h"
#include "stdio.h"
#include "stdlib.h"
#include <iterator>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "Kernels.h"
#include <cstdlib>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <mutex>
#include "Definitions.h"
#include "Misc.h"
#include "Settings.h"
#include "argpars.h"
#include "camera_thread.h"
#include "BackPropagator.h"
#include "BeadsFinder.h"
#include "ImageData.h"

using namespace std;

CameraImgI camI;
ImageData<uint8_t> G, R, G_backprop;
uint16_t bead_positions[2*BeadsFinder::MAX_NUMBER_BEADS];
uint32_t bead_count;
std::mutex mtx_bp;


int client;

int img_count = 0;
std::chrono::duration<double> elapsed_seconds_average;


void keyboard_thread(){
	printf("INFO: keyboard_thread: started\n");

	char input;
	while(!Settings::force_exit){
		input = getchar();
		if(input == 's'){
			printf("INFO: Putting the process to sleep.\n");
			Settings::set_sleeping(true);
			Settings::set_initialized(false);
		}
		else if(input == 'c'){
			printf("INFO: Simulating connection to main computation unit.\n");
			Settings::set_connected(true);
		}
		else if(input == 'w'){
			printf("INFO: Starting the program from keyboard.\n");
			Settings::set_initialized(true);
			Settings::set_sleeping(false);
		}
		else if(input == 'd'){
			Settings::set_connected(false);
			Settings::set_sleeping(true);
			Settings::set_initialized(false);
		}
		else if(input == 'e'){
			Settings::set_force_exit(true);
		}		
	}

	printf("INFO: keyboard_thread: ended\n");
}

void network_thread(){
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
		printf("ERROR: Couldn't create socket!\n");
	sockName.sin_family = AF_INET;
	sockName.sin_port =	htons(PORT);
	sockName.sin_addr.s_addr = INADDR_ANY;

	int yes = 1;
	if ( setsockopt(mainSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1 )
	{
	    perror("setsockopt");
	}

	bind(mainSocket, (sockaddr*)&sockName, sizeof(sockName));
	listen(mainSocket, 10000000);
	while(!Settings::force_exit){
		
		addrlen = sizeof(clientInfo);
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		cout << "INFO: Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
		if (client != -1)
		 {
			 Settings::set_connected(true);
		 }

		while(Settings::connected && !Settings::force_exit){
			int msg_len = recv(client, buf, BUFSIZE - 1, 0);

			if (msg_len == -1)
			{
				printf("ERROR: Did not properly receive data.\n");
			}

			if(Options::debug)
				printf("DEBUG: Received %d bytes.\n", msg_len);

			response = parseMessage(buf);
			switch(response){
				case MessageType::WAKEUP:
				{
					Settings::set_sleeping(false);
					Settings::set_initialized(true);
					break;
				}
				case MessageType::SLEEP:
				{
					Settings::set_sleeping(true);
					Settings::set_initialized(false);
					break;
				}
				case MessageType::SETTINGS:
				{
					if(!Settings::sleeping)
						printf("WARN: Can't change settings while the loop is running\n");
					else{
						memcpy(Settings::values, buf+1, sizeof(uint32_t)*STG_NUMBER_OF_SETTINGS);
						Settings::print();
						printf("INFO: Changed settings\n");
					}
					break;
				}
				case MessageType::DISCONNECT:
				{
					Settings::set_connected(false);
					Settings::set_sleeping(true);
					Settings::set_initialized(false);
					break;
				}
				case MessageType::REQUEST:
					Settings::set_requested_image(true);
					Settings::set_requested_type(RequestType::BACKPROPAGATED);
					break;
				case MessageType::REQUEST_RAW_G:
					Settings::set_requested_image(true);
					Settings::set_requested_type(RequestType::RAW_G);
					break;
				case MessageType::REQUEST_RAW_R:
					Settings::set_requested_image(true);
					Settings::set_requested_type(RequestType::RAW_R);
					break;
				case MessageType::COORDS:
					Settings::set_requested_coords(true);
					break;
				case MessageType::HELLO:
				{
					send(client, "Hello!",7,0);
					break;
				}	
			} 
		}
		close(client);
	}
	close(mainSocket);

	printf("INFO: network_thread: ended\n");
}

void mouseEventCallback(int event, int x, int y, int flags, void* userdata)
{
	// https://www.opencv-srf.com/2011/11/mouse-events.html
	if ( event == cv::EVENT_MOUSEMOVE )
     {
     	if(Options::debug)
        	cout << "DEBUG: Mouse move over the window - position (" << x << ", " << y << ")" << endl;
        Settings::set_force_exit(true);
     }
}

void datasend_thread(){
	printf("INFO: datasend_thread: started\n");
	uint8_t coords_buffer[BeadsFinder::MAX_NUMBER_BEADS+sizeof(uint32_t)];

	while(!Settings::force_exit){
		while(Settings::sleeping && !Settings::force_exit){}
		if (Settings::force_exit) break;

		while(Settings::connected && !Settings::sleeping && !Settings::force_exit){
			if(Settings::requested_image){
				ImageData<uint8_t> temp_img(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);

				switch (Settings::requested_type){
					case RequestType::BACKPROPAGATED:
						G_backprop.copyTo(temp_img);
						break;
					case RequestType::RAW_G:
						G.copyTo(temp_img);
						break;
					case RequestType::RAW_R:
						R.copyTo(temp_img);
						break;
				}	

				send(client, temp_img.hostPtr(), sizeof(uint8_t)*Settings::get_area(), 0);
				printf("INFO: Image sent.\n");
				Settings::set_requested_image(false);
			}

			if(Settings::requested_coords && !Settings::sent_coords) {
				mtx_bp.lock();
				((uint32_t*)coords_buffer)[0] = bead_count;
				memcpy(coords_buffer+sizeof(uint32_t), bead_positions, 2*bead_count*sizeof(uint16_t));
				send(client, coords_buffer, sizeof(uint32_t) + 2*bead_count*sizeof(uint16_t), 0);
				mtx_bp.unlock();

				printf("INFO: Sent the found locations\n");

				Settings::set_requested_coords(false);
				Settings::set_sent_coords(true);
			}
		}
	}

	printf("INFO: datasend_thread: ended\n");
}

void imgproc_thread(){
	printf("INFO: imgproc_thread: started\n");
	
	while(!Settings::force_exit) {
		// Wait till the host computer connects and starts processing the images
		while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;

		// Initialize the BackPropagator for the green image
		BackPropagator backprop_G(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT], LAMBDA_GREEN, (float)Settings::values[STG_Z_GREEN]/1000000.0f);

		// Initialize the BeadFinder
		BeadsFinder beadsFinder(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT], (uint8_t)Settings::values[STG_IMGTHRS], Options::saveimgs_bp);

		// Allocate the memory for the images
		G.create(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);
		R.create(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);
		G_backprop.create(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);

		while(Settings::connected && !Settings::sleeping && !Settings::force_exit){
			auto t_cycle_start = std::chrono::system_clock::now();

			// wait till new image is ready
			while(camI.img_produced == camI.img_processed && !Settings::force_exit) {
				usleep(200);
			}
			if(Settings::force_exit) break;

			// Make copies of red and green channel
			auto t_cp_start = std::chrono::system_clock::now();
			camI.G.copyTo(G);
			camI.R.copyTo(R);
			auto t_cp_end = std::chrono::system_clock::now();

			// increase the number of processed images so that the camera starts capturing a new image
			++camI.img_processed;

			// process the image
			// backprop
			auto t_backprop_start = std::chrono::system_clock::now();
			backprop_G.backprop(G, G_backprop);
			auto t_backprop_end = std::chrono::system_clock::now();

			// Update the image in beadsFinder where the beads are to be searched for
			auto t_beadsfinder_cp_start = std::chrono::system_clock::now();
			beadsFinder.updateImage(G_backprop);
			auto t_beadsfinder_cp_end = std::chrono::system_clock::now();

			// find the beads
			auto t_beadsfinder_start = std::chrono::system_clock::now();
			beadsFinder.findBeads();
			mtx_bp.lock();
			bead_count = beadsFinder.copyPositionsTo(bead_positions);
			mtx_bp.unlock();
			auto t_beadsfinder_end = std::chrono::system_clock::now();

			// Set the sent_coords flag to false to indicate that new bead positions were found and can be sent to the host computer
			Settings::set_sent_coords(false);

			auto t_cycle_end = std::chrono::system_clock::now();
			if(Options::verbose) {
				chrono::duration<double> cycle_elapsed_seconds = t_cycle_end - t_cycle_start;
				chrono::duration<double> cp_elapsed_seconds = t_cp_end - t_cp_start;
				chrono::duration<double> bp_elapsed_seconds = t_backprop_end - t_backprop_start;
				chrono::duration<double> bf_cp_elapsed_seconds = t_beadsfinder_cp_end - t_beadsfinder_cp_start;
				chrono::duration<double> bf_elapsed_seconds = t_beadsfinder_end - t_beadsfinder_start;

				std::cout << "TRACE: Backprop: " << bp_elapsed_seconds.count();
				std::cout << "| BF.cp: " << bf_cp_elapsed_seconds.count();
				std::cout << "| BF.findBeads: " << bf_elapsed_seconds.count();
				std::cout << "| cp: " << cp_elapsed_seconds.count();
				std::cout << "| whole cycle: " << cycle_elapsed_seconds.count();
				std::cout << "| #points: " << bead_count << std::endl;
			}
		}
	}
	
	printf("INFO: imgproc_thread: ended\n");
}

void display_thread(){
	printf("INFO: display_thread: started\n");
	
	char ret_key;
	char filename [50];

	while(!Settings::force_exit){
		while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;

		// Allocate the memory
		ImageData<uint8_t> G_backprop_copy(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);
		ImageData<uint8_t> G_copy(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);
		ImageData<uint8_t> R_copy(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);

		if (Options::show) {
			cv::namedWindow("Basic Visualization", cv::WINDOW_NORMAL);
			cv::setWindowProperty("Basic Visualization", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
			//set the callback function for any mouse event
			if (Options::mousekill) {
				cv::setMouseCallback("Basic Visualization", mouseEventCallback, NULL);
			}
		}

		while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break; 

		const cv::cuda::GpuMat c_img_resized(cv::Size(800, 800), CV_8U);
		const cv::cuda::GpuMat c_img_flip(cv::Size(800, 800), CV_8U);
		const cv::Mat img_disp(cv::Size(800, 800), CV_8U);

		uint32_t last_img_processed = camI.img_processed;

		while(!Settings::sleeping && Settings::connected && !Settings::force_exit){
			if(camI.img_processed - last_img_processed > 3){
				auto start = std::chrono::system_clock::now();
				if (Options::show && !Options::saveimgs)
					G_backprop.copyTo(G_backprop_copy);
				if (Options::saveimgs) {
					G_backprop.copyTo(G_backprop_copy);
					G.copyTo(G_copy);
					R.copyTo(R_copy);
				}
				
				if (Options::saveimgs) {					
					const cv::Mat G_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, G_copy.hostPtr());
					const cv::Mat R_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, R_copy.hostPtr());
					const cv::Mat G_backprop_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, G_backprop_copy.hostPtr());
					
					sprintf (filename, "./imgs/G_%05d.png", img_count);
					cv::imwrite( filename, G_img );
					
					sprintf (filename, "./imgs/R_%05d.png", img_count);
					cv::imwrite( filename, R_img );
					
					sprintf (filename, "./imgs/G_bp_%05d.png", img_count);
					cv::imwrite( filename, G_backprop_img );
				}				
				
				if (Options::show) {
					const cv::cuda::GpuMat c_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, G_backprop_copy.devicePtr());

					// Resize the image so that it fits the display
					cv::cuda::resize(c_img, c_img_resized, cv::Size(800, 800));	
					
					// Flip the axis so that the displayed image corresponds to the actual top view on the image sensor
					cv::cuda::flip(c_img_resized, c_img_flip, 0);

					c_img_flip.download(img_disp);

					// Draw bead positions
					mtx_bp.lock();
					for(int i = 0; i < bead_count; i++) {
						uint32_t x = (bead_positions[2*i]*800)/Settings::values[STG_WIDTH];
						uint32_t y = 800 - (bead_positions[2*i+1]*800)/Settings::values[STG_HEIGHT] - 1;
						cv::circle(img_disp, cv::Point(x, y), 20, 255);
					}
					mtx_bp.unlock();
					
					cv::imshow("Basic Visualization", img_disp);
					auto end = std::chrono::system_clock::now();
					std::chrono::duration<double> elapsed_seconds = end-start;
					if(Options::verbose) {
						std::cout << "TRACE: Stroring the image took: " << elapsed_seconds.count() << "s\n";
					}
	
					ret_key = (char) cv::waitKey(1);
					if (ret_key == 27 || ret_key == 'x') Settings::set_force_exit(true);  // exit the app if `esc' or 'x' key was pressed.					
				}				

				img_count++;
				last_img_processed = camI.img_processed;
			}
			else{
				usleep(1000);
			}
		}
	}

	printf("INFO: display_thread: ended\n");
}


int main(int argc, char* argv[]){
	Options::parse(argc, argv);
	
	if(Options::debug){
		printf("DEBUG: Initial settings:");
		Settings::print();
	}

	if (Options::show) {
		Settings::set_initialized(true);
		Settings::set_connected(true);
		Settings::set_sleeping(false);
	}

	thread camera_thr (camera_thread, std::ref(camI));
	thread imgproc_thr (imgproc_thread);
	thread display_thr (display_thread);
	thread network_thr (network_thread);
	thread datasend_thr (datasend_thread);
	thread keyboard_thr (keyboard_thread);
	
	camera_thr.join();
	imgproc_thr.join();
	display_thr.join();
	datasend_thr.join();
	keyboard_thr.join();

	return 0;
}
