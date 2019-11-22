#include "cuda.h"
#include "cufft.h"
#include "thrust/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/device_ptr.h"
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
#include "Definitions.h"
#include "Misc.h"
#include "Settings.h"
#include "argpars.h"
#include "camera_thread.h"
#include "BackPropagator.h"
// #include "BeadsFinder.h"

using namespace std;


uint8_t *G, *R, *G_backprop;

mutex mtx;

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

void changeSettings(char* buf){
	int tmpSettings; 
	int count = 0;
	int current_index = 2;
	while(count < 7){
		string str = "";
		while(isdigit(buf[current_index])){
			str.append(1u,buf[current_index]);
			current_index++;
		}
		try{
			tmpSettings = atol(str.c_str());
		}
		catch(int e ){
			printf("Number is too large\n");
			tmpSettings = 0;
		}
		if(tmpSettings != 0){
			Settings::set_setting(count, tmpSettings);
			printf("%d\n", Settings::values[count]);
		}
		count++;
		current_index++;
	}
}

void network_thread(){
	printf("INFO: network_thread: started\n");

	std::string text;
	sockaddr_in sockName;
	sockaddr_in clientInfo; 
	int mainSocket;
	char buf[BUFSIZE];
	socklen_t addrlen;
	MESSAGE_TYPE response;
	
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
				case MSG_WAKEUP:
				{
					Settings::set_sleeping(false);
					Settings::set_initialized(true);
					break;
				}
				case MSG_SLEEP:
				{
					Settings::set_sleeping(true);
					Settings::set_initialized(false);
					break;
				}
				case MSG_SETTINGS:
				{
					if(!Settings::sleeping)
						printf("WARN: Can't change settings while the loop is running\n");
					else{
						changeSettings(buf);
						printf("INFO: Changed settings\n");
					}
					break;
				}
				case MSG_DISCONNECT:
				{
					Settings::set_connected(false);
					Settings::set_sleeping(true);
					Settings::set_initialized(false);
					break;
				}
				case MSG_REQUEST:
					Settings::set_requested_image(true);
					Settings::set_requested_type(BACKPROPAGATED);
					break;
				case MSG_REQUEST_RAW_G:
					Settings::set_requested_image(true);
					Settings::set_requested_type(RAW_G);
					break;
				case MSG_REQUEST_RAW_R:
					Settings::set_requested_image(true);
					Settings::set_requested_type(RAW_R);
					break;
				case MSG_COORDS:
					Settings::set_requested_coords(true);
					break;
				case MSG_HELLO:
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

	float *temporary;
	char* buffer;
	while(true){
		while(Settings::sleeping && !Settings::force_exit){}
		if (Settings::force_exit) break;

		cudaMalloc(&temporary, Settings::get_area()*sizeof(float));
		
		while(Settings::connected && !Settings::sleeping){
			if(Settings::requested_image){
				mtx.lock();
				switch (Settings::requested_type){
					case BACKPROPAGATED:
						cudaMemcpy(temporary, G_backprop, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
						break;
					case RAW_G:
						cudaMemcpy(temporary, G, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
						break;
					case RAW_R:
						cudaMemcpy(temporary, R, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
						break;
				}	

				mtx.unlock();

				buffer = (char*)malloc(Settings::get_area()*sizeof(float));
				cudaMemcpy(buffer, temporary, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToHost);
				send(client, buffer, sizeof(float)*Settings::get_area(), 0);
				free(buffer);
				printf("INFO: Image sent.\n");
				Settings::set_requested_image(false);
			}			

			if (Settings::force_exit) break;
		}
		
		cudaFree(temporary);
	}

	printf("INFO: datasend_thread: ended\n");
}

void imgproc_thread(){
	printf("INFO: imgproc_thread: started\n");
	
	// Initialize the BackPropagator for the green image
	BackPropagator backprop_G(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT], LAMBDA_GREEN, (float)Settings::values[STG_Z_GREEN]/(float)1000000);

	// Initialize the BeadFinder
	// BeadsFinder beadsFinder(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);

	// Allocate the memory for the images
	cudaMalloc(&G, Settings::get_area()*sizeof(uint8_t));
	cudaMalloc(&R, Settings::get_area()*sizeof(uint8_t));
	cudaMalloc(&G_backprop, Settings::get_area()*sizeof(uint8_t));

	// Allocate the memory for the backprop image on the host
	// uint8_t* hG_backprop = (uint8_t*)malloc(sizeof(uint8_t)*Settings::get_area());

	while(!Settings::force_exit) {
		auto t_start_cycle = std::chrono::system_clock::now();

		// wait tiil new image is ready
		while(Camera::img_produced == Camera::img_processed && !Settings::force_exit) {
			usleep(500);
		}

		mtx.lock();
		// Make copies of red and green channel
		cudaMemcpy(G, Camera::G, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);	
		cudaMemcpy(R, Camera::R, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);	

		// increase the number of processed images so that the camera starts capturing a new image
		++Camera::img_processed;

		// process the image
		auto t_backprop = std::chrono::system_clock::now();
		backprop_G.backprop(G, G_backprop);
		// if(Options::debug){
			// printf("Searching for the beads \n");
		// }
		// cudaMemcpy(hG_backprop, G_backprop, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToHost);
		// beadsFinder.update(hG_backprop);

		mtx.unlock();

		auto t_end_cycle = std::chrono::system_clock::now();
		if(Options::verbose) {
			chrono::duration<double> cycle_elapsed_seconds = t_end_cycle - t_start_cycle;
			chrono::duration<double> bp_elapsed_seconds = t_end_cycle - t_backprop;

			std::cout << "TRACE: Backpropagation took: " << bp_elapsed_seconds.count();
			std::cout << "| whole cycle took: " << cycle_elapsed_seconds.count() << " s\n";
		}
	}

	cudaFree(G);
	cudaFree(R);
	cudaFree(G_backprop);

	printf("INFO: imgproc_thread: ended\n");
}

void display_thread(){
	printf("INFO: display_thread: started\n");

	uint8_t *G_copy, *R_copy, *G_backprop_copy; 
	uint8_t *cG_copy, *cR_copy, *cG_backprop_copy; 

	char ret_key;
	char filename [50];

	while(true){
		while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;
			
		
		// Allocate memoty on the host
		// displayed img
		if (Options::show && !Options::saveimgs)
			cudaHostAlloc((void **)&G_backprop_copy,  sizeof(uint8_t)*Settings::get_area(),  cudaHostAllocMapped);
		// saved imgs
		if (Options::saveimgs) {
			cudaHostAlloc((void **)&G_copy,  sizeof(uint8_t)*Settings::get_area(),  cudaHostAllocMapped);
			cudaHostAlloc((void **)&R_copy,  sizeof(uint8_t)*Settings::get_area(),  cudaHostAllocMapped);
			cudaHostAlloc((void **)&G_backprop_copy,  sizeof(uint8_t)*Settings::get_area(),  cudaHostAllocMapped);
		}

		// Allocate memory on GPU for copies of the images
		// displayed img
		if (Options::show && !Options::saveimgs)
			cudaHostGetDevicePointer((void **)&cG_backprop_copy,  (void *) G_backprop_copy , 0);
		// saved imgs
		if (Options::saveimgs) {
			cudaHostGetDevicePointer((void **)&cG_copy,  (void *) G_copy , 0);
			cudaHostGetDevicePointer((void **)&cR_copy,  (void *) R_copy , 0);
			cudaHostGetDevicePointer((void **)&cG_backprop_copy,  (void *) G_backprop_copy , 0);
		}

		if (Options::show) {
			cv::namedWindow("Basic Visualization", cv::WINDOW_NORMAL);
			cv::setWindowProperty("Basic Visualization", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
			//set the callback function for any mouse event
			if (Options::mousekill) {
				cv::setMouseCallback("Basic Visualization", mouseEventCallback, NULL);
			}
		}

		while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit){
			if (Options::show && !Options::saveimgs) {
				cudaFreeHost(G_backprop_copy);
			}
			if (Options::saveimgs) {
				cudaFreeHost(G_copy);
				cudaFreeHost(R_copy);
				cudaFreeHost(G_backprop_copy);
			}
			break;
		} 

		const cv::cuda::GpuMat c_img_resized(cv::Size(800, 800), CV_8U);
		const cv::cuda::GpuMat c_img_flip(cv::Size(800, 800), CV_8U);
		const cv::Mat img_disp(cv::Size(800, 800), CV_8U);

		uint32_t last_img_processed = Camera::img_processed;

		while(!Settings::sleeping && Settings::connected){
			if(Camera::img_processed - last_img_processed > 3){
				auto start = std::chrono::system_clock::now();
				mtx.lock();
				if (Options::show && !Options::saveimgs)
					cudaMemcpy(cG_backprop_copy, G_backprop, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
				if (Options::saveimgs) {
					cudaMemcpy(cG_copy, G, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
					cudaMemcpy(cR_copy, R, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
					cudaMemcpy(cG_backprop_copy, G_backprop, sizeof(uint8_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
				}
				mtx.unlock();

				
				if (Options::saveimgs) {					
					const cv::Mat G_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, G_copy);
					const cv::Mat R_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, R_copy);
					const cv::Mat G_backprop_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, G_backprop_copy);
					
					sprintf (filename, "./imgs/G_%05d.png", img_count);
					cv::imwrite( filename, G_img );
					
					sprintf (filename, "./imgs/R_%05d.png", img_count);
					cv::imwrite( filename, R_img );
					
					sprintf (filename, "./imgs/G_bp_%05d.png", img_count);
					cv::imwrite( filename, G_backprop_img );
				}				
				
				if (Options::show) {
					const cv::cuda::GpuMat c_img(cv::Size(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]), CV_8U, cG_backprop_copy);

					// Resize the image so that it fits the display
					cv::cuda::resize(c_img, c_img_resized, cv::Size(800, 800));	
					// Flip the axis so that the displayed image corresponds to the actual top view on the image sensor
					cv::cuda::flip(c_img_resized, c_img_flip, 0);

					c_img_flip.download(img_disp);
					
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
				last_img_processed = Camera::img_processed;
			}
			else{
				usleep(1000);
			}

			if (Settings::force_exit) break;
		}

		if (Options::show && !Options::saveimgs) {
			cudaFreeHost(G_backprop_copy);
		}
		if (Options::saveimgs) {
			cudaFreeHost(G_copy);
			cudaFreeHost(R_copy);
			cudaFreeHost(G_backprop_copy);
		}
	}

	printf("INFO: display_thread: ended\n");
}


int main(int argc, char* argv[]){
	Options::parse(argc, argv);
	
	if(Options::debug){
		printf("DEBUG: Initial settings:");
		for(int i = 0 ; i < STG_NUMBER_OF_SETTINGS; i++){
			printf("%d\n", Settings::values[i]);
		}
	}

	if (Options::show) {
		Settings::set_initialized(true);
		Settings::set_connected(true);
		Settings::set_sleeping(false);
	}

	thread camera_thr (Camera::camera_thread);
	thread imgproc_thr (imgproc_thread);
	thread display_thr (display_thread);
	thread network_thr (network_thread);
	thread datasend_thr (datasend_thread);
	thread keyboard_thr (keyboard_thread);
	
	camera_thr.join();
	imgproc_thr.join();
	display_thr.join();
	datasend_thr.join();

	return 0;
}
