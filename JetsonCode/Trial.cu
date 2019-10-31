#include "cuda.h"
#include "cufft.h"
#include "cudaEGL.h"
#include "cuda_egl_interop.h"
#include "thrust/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/device_ptr.h"
#include "Argus/Argus.h"
#include "EGLStream/EGLStream.h"
#include "stdio.h"
#include "stdlib.h"
#include "EGL/egl.h"
#include <iterator>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Kernels.h"
#include <cstdlib>
#include <thread>
#include <mutex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include "cxxopts.hpp"
#include "Definitions.h"
#include "Misc.h"
#include "Settings.h"
#include "BackPropagator.h"

#define dSTG_WIDTH Settings::values[STG_WIDTH]
#define dSTG_HEIGHT Settings::values[STG_HEIGHT]

static const int    DEFAULT_FPS        = 30;

using namespace std;
using namespace Argus;
using namespace EGLStream;

cudaError_t res;

int client;


uint16_t *R;
uint16_t *G;
uint16_t *G_backprop;
float *G_float;

mutex mtx;

int numBlocks;
short cycles;
int final_count, img_count = 0;
std::chrono::duration<double> elapsed_seconds_average;

EGLStreamKHR eglStream;
const textureReference* uvTex;
const textureReference* yTex;

texture<unsigned char, 2, cudaReadModeElementType> yTexRef;
texture<uchar2, 2, cudaReadModeElementType> uvTexRef;

struct is_not_zero
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x != 0;
	}
};

// Options
bool opt_verbose	= false;
bool opt_debug		= false;
bool opt_show		= false;
bool opt_saveimgs	= false;
bool opt_mousekill 	= false;

// cxxopts.hpp related definitions
cxxopts::ParseResult
parse(int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], " - Twin-beam setup - image processing");
    options
      .positional_help("[optional args]")
      .show_positional_help();

    options
      .add_options()
      ("s,show", 		"Display the processed image on the display",				cxxopts::value<bool>(opt_show))
      ("saveimgs", 		"Save images", 												cxxopts::value<bool>(opt_saveimgs))
      ("d,debug", 		"Prints debug information",									cxxopts::value<bool>(opt_debug))
      ("k,mousekill", 	"Moving the mouse or toching the screen kills the app",		cxxopts::value<bool>(opt_mousekill))
      ("v,verbose", 	"Prints some additional information",						cxxopts::value<bool>(opt_verbose))
	  ("digitalgain", 	"Digital gain 0-100", 										cxxopts::value<uint32_t>())
      ("e,exp",			"Exposure time (us)",										cxxopts::value<uint32_t>())
      ("r,resolution", 	"Resolution (example -r 1024,1024)",						cxxopts::value<std::vector<uint32_t>>())
	  ("o,offset", 		"Offset of the image (example -o 123,523)", 				cxxopts::value<std::vector<uint32_t>>())
      ("help", 			"Prints help")
    ;
	
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }


    if (opt_debug) {
	    if (opt_show)
	    {
	      std::cout << "Saw option ‘s’" << std::endl;
	    }

	    if (opt_debug)
	    {
	      std::cout << "Saw option ‘d’" << std::endl;
	    }

	    if (opt_verbose)
	    {
	      std::cout << "Saw option ‘v’" << std::endl;
	    }
	}


    return result;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}


//#region

__global__ void yuv2bgr(int width, int height, int offset_x, int offset_y,
						uint16_t* G, uint16_t* R)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = width*height;
            int tx, ty, ty2;
            float y1, y2;
            float u1, v2, v1;
            for (int i = index; i < count; i += stride)
            {
            	ty = i/width + offset_y;
            	ty2 = i/width + offset_y - (512);
            	tx = i%width + offset_x;
            	y1 = (float)((tex2D<unsigned char>(yTexRef, (float)tx+0.5f, (float)ty+0.5f) - (float)16) * 1.164383f);
            	y2 = (float)((tex2D<unsigned char>(yTexRef, (float)tx+0.5f, (float)ty2+0.5f) - (float)16) * 1.164383f);
            	u1 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
            	 	 (float)(ty/2)+(float)(ty%2)+0.5f).x - 128) * 0.391762f;
            	v2 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
            	     (float)(ty2/2)+(float)(ty2%2)+0.5f).y - 128) * 1.596027f;
            	v1 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
            	     (float)(ty/2)+(float)(ty%2)+0.5f).y - 128) * 0.812968f;
				G[i] = (uint16_t)(y1-u1-v1);
				R[i] = (uint16_t)(y2+v2+u1/10);
            }
        }

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

			if(opt_debug)
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

void camera_thread(){
	printf("INFO: camera_thread: started\n");
	//Initializing LibArgus according to the tutorial for a sample project.
	// First we create a CameraProvider, necessary for each project.
	UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());
	ICameraProvider* iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
	if(!iCameraProvider){
		printf("ERROR: Failed to establish libargus connection\n");
	}
	
	// Second we select a device from which to receive pictures (camera)
	std::vector<CameraDevice*> cameraDevices;
	iCameraProvider->getCameraDevices(&cameraDevices);
	if (cameraDevices.size() == 0){
		printf("ERROR: No camera devices available\n");
	}
	CameraDevice *selectedDevice = cameraDevices[0];

	// We create a capture session 
	UniqueObj<CaptureSession> captureSession(iCameraProvider->createCaptureSession(selectedDevice));
	ICaptureSession *iCaptureSession = interface_cast<ICaptureSession>(captureSession);
	if (!iCaptureSession){
 		printf("ERROR: Failed to create CaptureSession\n");
	}
	
	//CUDA variable declarations
	cudaEglStreamConnection conn;
	cudaGraphicsResource_t resource;
	cudaEglFrame eglFrame;		
	cudaArray_t yArray;
	cudaArray_t uvArray;
	cudaChannelFormatDesc yChannelDesc;
	cudaChannelFormatDesc uvChannelDesc;

	while(!Settings::force_exit){
		while(Settings::connected && !Settings::force_exit){
			while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
			if (Settings::force_exit) break;
			// Managing the settings for the capture session.
			UniqueObj<OutputStreamSettings> streamSettings(iCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
			IEGLOutputStreamSettings *iStreamSettings = interface_cast<IEGLOutputStreamSettings>(streamSettings);
			iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
			iStreamSettings->setResolution(Size2D<uint32_t>(WIDTH,HEIGHT));
			
			// Creating an Output stream. This should already create a producer.
			UniqueObj<OutputStream> outputStream(iCaptureSession->createOutputStream(streamSettings.get()));
			IEGLOutputStream *iEGLOutputStream = interface_cast<IEGLOutputStream>(outputStream);
            if (!iEGLOutputStream)
	            printf("Failed to create EGLOutputStream");

			eglStream = iEGLOutputStream->getEGLStream();
			cudaEGLStreamConsumerConnect(&conn, eglStream);
			
			// Managing requests.
			UniqueObj<Request> request(iCaptureSession->createRequest());
			IRequest *iRequest = interface_cast<IRequest>(request);
			iRequest->enableOutputStream(outputStream.get());
			
			ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
			iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));
			iSourceSettings->setExposureTimeRange(Range<uint64_t>(Settings::values[STG_EXPOSURE],Settings::values[STG_EXPOSURE]));
			iSourceSettings->setGainRange(Range<float>(100.0,100.0));

			IAutoControlSettings *iAutoSettings = interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
			iAutoSettings->setExposureCompensation(0);
			iAutoSettings->setIspDigitalGainRange(Range<float>(Settings::values[STG_DIGGAIN],Settings::values[STG_DIGGAIN]));
			iAutoSettings->setWbGains(100);
			iAutoSettings->setColorSaturation(1.0);
			iAutoSettings->setColorSaturationBias(1.0);
			iAutoSettings->setColorSaturationEnable(true);
			iAutoSettings->setAwbLock(true);
			iAutoSettings->setAeAntibandingMode(AE_ANTIBANDING_MODE_OFF);

			IDenoiseSettings *iDenoiseSettings = interface_cast<IDenoiseSettings>(request);	
			iDenoiseSettings->setDenoiseMode(DENOISE_MODE_FAST);
			iDenoiseSettings->setDenoiseStrength(1.0);

			cudaMalloc(&G, Settings::get_area()*sizeof(uint16_t));
			cudaMalloc(&G_float, Settings::get_area()*sizeof(float));
			cudaMalloc(&R, Settings::get_area()*sizeof(uint16_t));
			cudaMalloc(&G_backprop, Settings::get_area()*sizeof(uint16_t));
			
			numBlocks = 1024;
			
			yTexRef.normalized = 0;
			yTexRef.filterMode = cudaFilterModePoint;
			yTexRef.addressMode[0] = cudaAddressModeClamp;
			yTexRef.addressMode[1] = cudaAddressModeClamp;
			cudaGetTextureReference(&yTex, &yTexRef);
			
			uvTexRef.normalized = 0;
			uvTexRef.filterMode = cudaFilterModePoint;
			uvTexRef.addressMode[0] = cudaAddressModeClamp;
			uvTexRef.addressMode[1] = cudaAddressModeClamp;
			cudaGetTextureReference(&uvTex, &uvTexRef);

			// Initialize the BackPropagator for the green image
			BackPropagator backprop_G(dSTG_WIDTH, dSTG_HEIGHT, LAMBDA_GREEN, (float)Settings::values[STG_Z_GREEN]/(float)1000000);
			
			//CUDA initialization
			//Main loop
			auto initializer = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds_average = initializer-initializer;

			final_count = 0;
			while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
			if (Settings::force_exit) break;

			while(!Settings::sleeping && Settings::connected && ! Settings::force_exit){
				auto start = std::chrono::system_clock::now();
				
				
				iCaptureSession->capture(request.get());
				res = cudaEGLStreamConsumerAcquireFrame(&conn, &resource, 0, 5000);
				if(res != cudaSuccess){
					continue;
				}
				cudaGraphicsResourceGetMappedEglFrame(&eglFrame, resource, 0, 0);
				yArray = eglFrame.frame.pArray[0];
				uvArray = eglFrame.frame.pArray[1];
				
				cudaGetChannelDesc(&yChannelDesc, (cudaArray_const_t)(yArray));
				cudaBindTextureToArray(yTex, (cudaArray_const_t)(yArray), &yChannelDesc);
				cudaGetChannelDesc(&uvChannelDesc, (cudaArray_const_t)(uvArray));
				cudaBindTextureToArray(uvTex, (cudaArray_const_t)(uvArray), &uvChannelDesc);
				auto initialization = std::chrono::system_clock::now();

				numBlocks = (Settings::get_area()/2 +BLOCKSIZE -1)/BLOCKSIZE;
				
				mtx.lock();
				yuv2bgr<<<numBlocks, BLOCKSIZE>>>(dSTG_WIDTH, dSTG_HEIGHT,
												Settings::values[STG_OFFSET_X], Settings::values[STG_OFFSET_Y], G, R);
				backprop_G.backprop(G, G_backprop);

				u16ToFloat<<<numBlocks, BLOCKSIZE>>>(dSTG_WIDTH, dSTG_HEIGHT, G_backprop, G_float);

				mtx.unlock();
				
				auto test2 = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = test2-initialization;
				
				if(opt_verbose) {
					std::cout << "TRACE: Converting the image format + backprop took: " << elapsed_seconds.count() << "s\n";
				}

				cudaUnbindTexture(yTex);
				cudaUnbindTexture(uvTex);
				
				cudaEGLStreamConsumerReleaseFrame(&conn, resource, 0);
				
				auto end = std::chrono::system_clock::now();
				elapsed_seconds = end-start;
				elapsed_seconds_average +=elapsed_seconds;
				final_count++;
				Settings::sent_coords = false;
				
				if(opt_verbose) {
					std::cout << "TRACE: This cycle took: " << elapsed_seconds.count() << "s\n";
				}

				cycles++;				
			}
			std::cout << "INFO: Average time to complete a cycle: " << elapsed_seconds_average.count()/final_count << "s\n";
			iCaptureSession->waitForIdle();
			
			cudaFree(G);
			cudaFree(R);
			cudaFree(G_backprop);
			
			cudaEGLStreamConsumerDisconnect(&conn);
			iEGLOutputStream->disconnect();
			outputStream.reset();
		}
	}

	printf("INFO: camera_thread: ended\n");
}

void mouseEventCallback(int event, int x, int y, int flags, void* userdata)
{
	// https://www.opencv-srf.com/2011/11/mouse-events.html
	if ( event == CV_EVENT_MOUSEMOVE )
     {
     	if(opt_debug)
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

void display_thread(){
	printf("INFO: display_thread: started\n");

	float *cimageToDisplay, *imageToDisplay;

	uint16_t *G_copy, *R_copy, *G_backprop_copy; 
	uint16_t *cG_copy, *cR_copy, *cG_backprop_copy; 

	char ret_key;
	char filename [50];

	while(true){
		while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;
			
		// Allocate memory on GPU for copies of the images
		// displayed img
		if (opt_show)
			cudaMalloc(&cimageToDisplay, sizeof(float)*Settings::get_area());
		// saved imgs
		if (opt_saveimgs) {
			cudaMalloc(&cG_copy, sizeof(uint16_t)*Settings::get_area());
			cudaMalloc(&cR_copy, sizeof(uint16_t)*Settings::get_area());
			cudaMalloc(&cG_backprop_copy, sizeof(uint16_t)*Settings::get_area());
		}

		// Allocate memoty on the host
		// displayed img
		if (opt_show)
			imageToDisplay = (float*)malloc(sizeof(float)*Settings::get_area());
		// saved imgs
		if (opt_saveimgs) {
			G_copy = (uint16_t*)malloc(sizeof(uint16_t)*Settings::get_area());
			R_copy = (uint16_t*)malloc(sizeof(uint16_t)*Settings::get_area());
			G_backprop_copy = (uint16_t*)malloc(sizeof(uint16_t)*Settings::get_area());
		}

		if (opt_show) {
			cv::namedWindow("Basic Visualization", CV_WINDOW_NORMAL);
			cv::setWindowProperty("Basic Visualization", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			//set the callback function for any mouse event
			if (opt_mousekill) {
				cv::setMouseCallback("Basic Visualization", mouseEventCallback, NULL);
			}
		}

		while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit){
			if (opt_show) {
				cudaFree(cimageToDisplay);
				free(imageToDisplay);
			}
			if (opt_saveimgs) {
				cudaFree(cG_copy);
				cudaFree(cR_copy);
				cudaFree(cG_backprop_copy);
				free(G_copy);
				free(R_copy);
				free(G_backprop_copy);
			}
			break;
		} 

		const cv::Mat img_trans(cv::Size(dSTG_WIDTH, dSTG_HEIGHT), CV_32F);
		const cv::Mat img_u8(cv::Size(dSTG_WIDTH, dSTG_HEIGHT), CV_8U);

		while(!Settings::sleeping && Settings::connected){
			if(cycles >= 3){
				auto start = std::chrono::system_clock::now();
				cycles = 0;
				mtx.lock();
				if (opt_show)
					cudaMemcpy(cimageToDisplay, G_float, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
				if (opt_saveimgs) {
					cudaMemcpy(cG_copy, G, sizeof(uint16_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
					cudaMemcpy(cR_copy, R, sizeof(uint16_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
					cudaMemcpy(cG_backprop_copy, G_backprop, sizeof(uint16_t)*Settings::get_area(), cudaMemcpyDeviceToDevice);					
				}
				mtx.unlock();

				
				if (opt_saveimgs) {
					cudaMemcpy(G_copy, cG_copy, sizeof(uint16_t)*Settings::get_area(), cudaMemcpyDeviceToHost);					
					cudaMemcpy(R_copy, cR_copy, sizeof(uint16_t)*Settings::get_area(), cudaMemcpyDeviceToHost);					
					cudaMemcpy(G_backprop_copy, cG_backprop_copy, sizeof(uint16_t)*Settings::get_area(), cudaMemcpyDeviceToHost);					
					
					const cv::Mat G_img(cv::Size(dSTG_WIDTH, dSTG_HEIGHT), CV_16U, G_copy);
					const cv::Mat R_img(cv::Size(dSTG_WIDTH, dSTG_HEIGHT), CV_16U, R_copy);
					const cv::Mat G_backprop_img(cv::Size(dSTG_WIDTH, dSTG_HEIGHT), CV_16U, G_backprop_copy);
					
					sprintf (filename, "./imgs/G_%05d.png", img_count);
					cv::imwrite( filename, G_img );
					
					sprintf (filename, "./imgs/R_%05d.png", img_count);
					cv::imwrite( filename, R_img );
					
					sprintf (filename, "./imgs/G_bp_%05d.png", img_count);
					cv::imwrite( filename, G_backprop_img );
				}				
				
				if (opt_show) {
					cudaMemcpy(imageToDisplay, cimageToDisplay, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToHost);
					const cv::Mat img(cv::Size(dSTG_WIDTH, dSTG_HEIGHT), CV_32F, imageToDisplay);

					cv::flip(img, img_trans, -1);
					cv::transpose(img_trans, img);
					
					img.convertTo(img_u8,CV_8U);

					cv::imshow("Basic Visualization", img_u8);
					auto end = std::chrono::system_clock::now();
					std::chrono::duration<double> elapsed_seconds = end-start;
					if(opt_verbose) {
						std::cout << "TRACE: Stroring the image took: " << elapsed_seconds.count() << "s\n";
					}
	
					ret_key = (char) cv::waitKey(1);
					if (ret_key == 27 || ret_key == 'x') Settings::set_force_exit(true);  // exit the app if `esc' or 'x' key was pressed.					
				}				

				img_count++;
			}
			else{
				usleep(5000);
			}

			if (Settings::force_exit) break;
		}

		if (opt_show) {
			cudaFree(cimageToDisplay);
			free(imageToDisplay);
		}
		if (opt_saveimgs) {
			cudaFree(cG_copy);
			cudaFree(cR_copy);
			cudaFree(cG_backprop_copy);
			free(G_copy);
			free(R_copy);
			free(G_backprop_copy);
		}
	}

	printf("INFO: display_thread: ended\n");
}


int main(int argc, char* argv[]){
	auto result = parse(argc, argv);

	if (result.count("exp") > 0)
		Settings::values[STG_EXPOSURE] 	= result["exp"].as<uint32_t>();
	if (result.count("digitalgain") > 0)
		Settings::values[STG_DIGGAIN] 	= result["digitalgain"].as<uint32_t>();
	if (result.count("resolution") > 0) {
		const auto values = result["resolution"].as<std::vector<uint32_t>>();
		Settings::values[STG_WIDTH] 	= values[0];
		Settings::values[STG_HEIGHT] 	= values[1];
	}
	if (result.count("offset") > 0) {
		const auto values = result["offset"].as<std::vector<uint32_t>>();
		Settings::values[STG_OFFSET_X] 	= values[0];
		Settings::values[STG_OFFSET_Y] 	= values[1];
	}	
	

	if(opt_debug){
		printf("DEBUG: Initial settings:");
		for(int i = 0 ; i < STG_NUMBER_OF_SETTINGS; i++){
			printf("%d\n", Settings::values[i]);
		}
	}
	
	cycles = 0;

	if (opt_show) {
		Settings::set_initialized(true);
		Settings::set_connected(true);
		Settings::set_sleeping(false);
	}

	thread camera_thr (camera_thread);
	thread display_thr (display_thread);
	thread network_thr (network_thread);
	thread datasend_thr (datasend_thread);
	thread keyboard_thr (keyboard_thread);
	
	camera_thr.join();
	display_thr.join();
	datasend_thr.join();

	return 0;
}
