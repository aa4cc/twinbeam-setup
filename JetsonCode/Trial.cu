#include "cuda.h"
#include "cufft.h"
#include "cudaEGL.h"
#include "cuda_egl_interop.h"
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
#include "Definitions.h"
#include "cxxopts.hpp"

#define BUFSIZE 1000
#define PORT 30000

static const int    DEFAULT_FPS        = 30;

using namespace std;
using namespace Argus;
using namespace EGLStream;

cudaError_t res;

float* doubleArray;
float* outputArray;
float* convoOutputArray;
float* convoOutputArrayRed;
cufftComplex* kernelGreen;
cufftComplex* kernelRed;

float* redConverted;

float* convolutionMaskGreen;
float* convolutionMaskRed;
float* convoOutputArrayGreen;

float* maximaRed;
float* maximaGreen;
float* doubleTemporary;

int* greenPoints;
int* redPoints;
int* positionsGreen;
int* positionsRed;

int* redPointsLast;
int* greenPointsLast;
int* current_index;

bool playSequence;
bool sleeping;
bool connected;
bool initialized;
bool requested_image;
REQUEST_TYPE requested_type;
bool send_points;
bool force_exit;
bool touch_kill;
int settings[7];
int client;

// Options
bool opt_verbose	= false;
bool opt_debug		= false;
bool opt_show		= false;

uint16_t *R;
uint16_t *G;

mutex mtx;
mutex outputMtx;

int numBlocks;

bool quitSequence;
short cycles;

std::chrono::duration<double> elapsed_seconds_average;
std::chrono::duration<double> initialization_seconds_average;
std::chrono::duration<double> conversion_seconds_average;
std::chrono::duration<double> back_propagation_seconds_average;
std::chrono::duration<double> convolution_seconds_average;
std::chrono::duration<double> localmaxima_seconds_average;
std::chrono::duration<double> sorting_seconds_average;


EGLStreamKHR eglStream;

const textureReference* uvTex;
const textureReference* yTex;

texture<unsigned char, 2, cudaReadModeElementType> yTexRef;
texture<uchar2, 2, cudaReadModeElementType> uvTexRef;

struct is_zero
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return (x == 0);
  }
};


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
      ("s,show", "Display the processed image on the display", 	cxxopts::value<bool>(opt_show))
      ("d,debug", "Prints debug information",					cxxopts::value<bool>(opt_debug))
      ("v,verbose", "Prints some additional information",		cxxopts::value<bool>(opt_verbose))
      ("help", "Prints help")
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


int processPoints(float* inputPoints, int* outputArray){
	int* positions;
	int* positionsSorted;
	int* counter;
	float* points;
	int* counting;
	int* temp;
	
	cudaMalloc(&points, settings[0]*settings[1]*sizeof(float));
	cudaMalloc(&positions, settings[0]*settings[1]*sizeof(int));
	cudaMalloc(&positionsSorted, settings[0]*settings[1]*sizeof(int));
	cudaMalloc(&counter, sizeof(int));
	cudaMallocHost(&counting,sizeof(int));
    memset(counting, 0, sizeof(int));
	
	cudaMemcpy(points, inputPoints, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToDevice);
	findPoints<<<numBlocks, BLOCKSIZE>>>(settings[0], settings[1], points, positions, counter);
	stupidSort<<<numBlocks, BLOCKSIZE>>>(settings[0], settings[1],positions, positionsSorted, counting);
	cudaMemcpy(counting, counter, sizeof(int), cudaMemcpyDeviceToHost);
	temp = (int*)malloc(sizeof(int)*counting[0]);
	cudaMemcpy(temp, positionsSorted, sizeof(int)*counting[0], cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(points);
	cudaFree(positions);
	cudaFree(counter);
	cudaFree(positionsSorted);
	outputArray = temp;
	return counting[0];
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
            	ty2 = i/width + offset_y - (512-50);
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

void transformKernel(int M, int N, int kernelDim, float* kernel, cufftComplex* outputKernel){
	
	kernelToImage<<<numBlocks, BLOCKSIZE>>>(M, N, kernelDim, kernel, outputKernel);
   
    cufftHandle plan;
    
    cufftPlan2d(&plan, N,M, CUFFT_C2C);
    
    cufftExecC2C(plan, outputKernel, outputKernel, CUFFT_FORWARD);
    
    cufftDestroy(plan);
}

void printArray(float* array, int width, int height){
	for(int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			printf("%f ", array[j + height*i]);
			}
			printf("\n");
		}
}


void h_backPropagate(int M, int N, float lambda, float z, float* input,
		cufftComplex* kernel, float* output, float* output2, bool display)
{
    cufftComplex* doubleComplexArray;
    cufftComplex* Hq;
    cufftComplex* image;
    cufftComplex* kernelizedImage;
    float* temporary;
    float* extremes;
    cufftHandle plan;

    cudaMalloc(&doubleComplexArray, 3*N*M*sizeof(cufftComplex));
    Hq = &doubleComplexArray[0];
    image = &doubleComplexArray[N*M];
    kernelizedImage = &doubleComplexArray[2*N*M];
    cudaMalloc(&temporary, N*M*sizeof(float));
    cudaMalloc(&extremes, sizeof(float));

    convertToComplex<<<numBlocks, BLOCKSIZE>>>(N*M, input, image);
    // Declaring the FFT plan
    cufftPlan2d(&plan, N,M, CUFFT_C2C);
    // Execute forward FFT on the green channel
    cufftExecC2C(plan, image, image, CUFFT_FORWARD);
    // Calculating the Hq matrix according to the equations in the original .m file.
    calculate<<<numBlocks, BLOCKSIZE>>>(N,M, z, PIXEL_DX, REFRACTION_INDEX, lambda, Hq);
    // Element-wise multiplication of Hq matrix and the image
	elMultiplication<<<numBlocks, BLOCKSIZE>>>(M, N, Hq, image);
	elMultiplication2<<<numBlocks, BLOCKSIZE>>>(M, N, image, kernel, kernelizedImage);
    if(display){
		// Executing inverse FFT
		cufftExecC2C(plan, image, image, CUFFT_INVERSE);
		// Conversion of result matrix to a real double matrix
		absoluteValue<<<numBlocks, BLOCKSIZE>>>(M,N, image, output);

		findExtremes<<<numBlocks, BLOCKSIZE>>>(M,N, output, extremes);
		normalize<<<numBlocks, BLOCKSIZE>>>(M,N, output, extremes);
	}
	cufftExecC2C(plan, kernelizedImage, kernelizedImage, CUFFT_INVERSE);
	cutAndConvert<<<numBlocks, BLOCKSIZE>>>(M,N,kernelizedImage, convoOutputArrayGreen);
    cudaFree(extremes);
    cudaMalloc(&extremes, sizeof(float));
	findExtremes<<<numBlocks, BLOCKSIZE>>>(M,N, convoOutputArrayGreen, extremes);
	normalize<<<numBlocks, BLOCKSIZE>>>(M,N, convoOutputArrayGreen, extremes);
	getLocalMaxima<<<numBlocks, BLOCKSIZE>>>(M,N,convoOutputArrayGreen,output2);
	// Freeing the memory of FFT plan
	cufftDestroy(plan);

    cudaFree(extremes);
    cudaFree(doubleComplexArray);
    cudaFree(temporary);
}

void printErrorRuntime(cudaError_t result){
	const char* pstr = cudaGetErrorName(result);
	//printf("%s\n", pstr);
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
			settings[count] = tmpSettings;
			printf("%d\n", settings[count]);
		}
		count++;
		current_index++;
	}
}

MESSAGE_TYPE parseMessage(char* buf){
		switch (buf[0]){
			case 's':
				return MSG_WAKEUP;
			case 'q':
				return MSG_SLEEP;
			case 'o':
				return MSG_SETTINGS;
			case 'a':
				return MSG_HELLO;
			case 'd':
				return MSG_DISCONNECT;
			case 'r':
				requested_type = BACKPROPAGATED;
				return MSG_REQUEST;
			case 'x':
				return MSG_REQUEST_RAW_G;
			case 'y':
				return MSG_REQUEST_RAW_R;
			default:
				return MSG_UNKNOWN_TYPE;
		}
}



void keyboard_thread(){
	printf("keyboard_thread: started\n");

	char input;
	while(!force_exit){
		input = getchar();
		if(input == 's'){
			printf("Putting the process to sleep\n");
			sleeping = true;
			initialized = false;
		}
		else if(input == 'c'){
			printf("Connected the main manipulation computer\n");
			connected = true;
		}
		else if(input == 'w'){
			printf("Starting the program from keyboard\n");
			initialized = true;
			sleeping = false;
		}
		else if(input == 'd'){
			connected = false;
			sleeping = true;
			initialized = false;
		}
		else if(input == 'e'){
			force_exit = true;
		}		
	}

	printf("keyboard_thread: ended\n");
}

void input_thread(){
	printf("input_thread: started\n");

	std::string text;
	sockaddr_in sockName;
	sockaddr_in clientInfo; 
	int mainSocket;
	char buf[BUFSIZE];
	socklen_t addrlen;
	MESSAGE_TYPE response;
	int size;
	
	mainSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(mainSocket == -1)
		printf("Couldn't create socket!\n");
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
	while(!force_exit){
		
		addrlen = sizeof(clientInfo);
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		cout << "Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
		if (client != -1)
		 {
			 connected = true;
		 }

		while(connected && !force_exit){
			int msg_len = recv(client, buf, BUFSIZE - 1, 0);

			if (msg_len == -1)
			{
				printf("Error while receiving data\n");
			}

			printf("Received bytes: %d\n", msg_len);

			response = parseMessage(buf);
			switch(response){
				case MSG_WAKEUP:
				{
					sleeping = false;
					initialized = true;
					break;
				}
				case MSG_SLEEP:
				{
					sleeping = true;
					initialized = false;
					break;
				}
				case MSG_SETTINGS:
				{
					if(sleeping == false)
						printf("Can't change settings while the loop is running\n");
					else{
						changeSettings(buf);
						printf("Changed settings\n");
					}
					break;
				}
				case MSG_DISCONNECT:
				{
					connected = false;
					sleeping = true;
					initialized = false;
					break;
				}
				case MSG_REQUEST:
					requested_image = true;
					break;
				case MSG_REQUEST_RAW_G:
					requested_image = true;
					requested_type = RAW_G;
					break;
				case MSG_REQUEST_RAW_R:
					requested_image = true;
					requested_type = RAW_R;
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

	printf("input_thread: ended\n");
}

//#end_region
void consumer_thread(){
	printf("consumer_thread: started\n");
	//Initializing LibArgus according to the tutorial for a sample project.
	// First we create a CameraProvider, necessary for each project.
	UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());
	ICameraProvider* iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
	if(!iCameraProvider){
		printf("Failed to establish libargus connection\n");
	}
	
	// Second we select a device from which to receive pictures (camera)
	std::vector<CameraDevice*> cameraDevices;
	iCameraProvider->getCameraDevices(&cameraDevices);
	if (cameraDevices.size() == 0){
		printf("No camera devices available\n");
	}
	CameraDevice *selectedDevice = cameraDevices[0];

	// We create a capture session 
	UniqueObj<CaptureSession> captureSession(iCameraProvider->createCaptureSession(selectedDevice));
	ICaptureSession *iCaptureSession = interface_cast<ICaptureSession>(captureSession);
	if (!iCaptureSession){
 		printf("Failed to create CaptureSession\n");
	}
	
	//CUDA variable declarations
	cudaEglStreamConnection conn;
	cudaGraphicsResource_t resource;
	cudaEglFrame eglFrame;		
	cudaArray_t yArray;
	cudaArray_t uvArray;
	float* extremesMapGreen;
	
	
	
	cudaChannelFormatDesc yChannelDesc;
	cudaChannelFormatDesc uvChannelDesc;
	while(!force_exit){
		while(connected && !force_exit){
			while(sleeping && connected && !force_exit){}
			if (force_exit) break;
			// Managing the settings for the capture session.
			UniqueObj<OutputStreamSettings> streamSettings(iCaptureSession->createOutputStreamSettings());
			IOutputStreamSettings *iStreamSettings = interface_cast<IOutputStreamSettings>(streamSettings);
			iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
			iStreamSettings->setResolution(Size2D<uint32_t>(WIDTH,HEIGHT));
			
			// Creating an Output stream. This should already create a producer.
			UniqueObj<OutputStream> outputStream(iCaptureSession->createOutputStream(streamSettings.get()));
			IStream* iStream = interface_cast<IStream>(outputStream);
			if (!iStream){
				printf("Failed to create OutputStream\n");
			}
			eglStream = iStream->getEGLStream();
			cudaEGLStreamConsumerConnect(&conn, eglStream);
			
			// Managing requests.
			UniqueObj<Request> request(iCaptureSession->createRequest());
			IRequest *iRequest = interface_cast<IRequest>(request);
			iRequest->enableOutputStream(outputStream.get());
			
			ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
			iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));
			iSourceSettings->setExposureTimeRange(Range<uint64_t>(settings[4]));

			cudaMalloc(&G, settings[0]*settings[1]*sizeof(uint16_t));
			cudaMalloc(&R, settings[0]*settings[1]*sizeof(uint16_t));
			cudaMalloc(&positionsGreen, settings[0]*settings[1]*sizeof(float));
			cudaMalloc(&positionsRed, settings[0]*settings[1]*sizeof(float));
			
			
			cudaMalloc(&convolutionMaskGreen, CONVO_DIM_GREEN*CONVO_DIM_GREEN*sizeof(float));
			cudaMalloc(&convolutionMaskRed, CONVO_DIM_RED*CONVO_DIM_RED*sizeof(float));
			numBlocks = 1024;
			generateConvoMaskGreen<<<numBlocks, BLOCKSIZE>>>(CONVO_DIM_GREEN, CONVO_DIM_GREEN, convolutionMaskGreen);
			generateConvoMaskRed<<<numBlocks, BLOCKSIZE>>>(CONVO_DIM_RED, CONVO_DIM_RED, convolutionMaskRed);
			
			cudaMalloc(&kernelGreen, settings[0]*settings[1]*sizeof(cufftComplex));
			cudaMalloc(&kernelRed, settings[0]*settings[1]*sizeof(cufftComplex));
			
			transformKernel(settings[0], settings[1], CONVO_DIM_GREEN, convolutionMaskGreen, kernelGreen);
			transformKernel(settings[0], settings[1], CONVO_DIM_RED, convolutionMaskRed, kernelRed);
			
			cudaMalloc(&convoOutputArrayGreen, settings[0]*settings[1]*sizeof(float));
			cudaMalloc(&convoOutputArrayRed, settings[0]*settings[1]*sizeof(float));
			cudaMallocManaged(&current_index, sizeof(int));
			mtx.lock();
			cudaMalloc(&maximaGreen, settings[0]*settings[1]*sizeof(float));
			cudaMalloc(&maximaRed, settings[0]*settings[1]*sizeof(float));
			cudaMalloc(&doubleArray, 2*settings[1]*settings[0]*sizeof(float));
			doubleTemporary = &doubleArray[0];
			outputArray = &doubleArray[settings[1]*settings[0]];
			cudaMalloc(&convoOutputArray, settings[0]*settings[1]*sizeof(float));
			cudaMalloc(&redConverted, settings[0]*settings[1]*sizeof(float));
			mtx.unlock();

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
			
			//CUDA initialization
			//Main loop
			auto initializer = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds_average = initializer-initializer;
			std::chrono::duration<double> initialization_seconds_average = initializer-initializer;
			std::chrono::duration<double> conversion_seconds_average = initializer-initializer;
			std::chrono::duration<double> back_propagation_seconds_average = initializer-initializer;
			std::chrono::duration<double> convolution_seconds_average = initializer-initializer;
			std::chrono::duration<double> localmaxima_seconds_average = initializer-initializer;
			std::chrono::duration<double> sorting_seconds_average = initializer-initializer;

			int final_count = 0;
			while(!initialized && connected && !force_exit){}
			if (force_exit) break;

			while(!sleeping && connected && ! force_exit){
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

				
				numBlocks = (settings[0]*settings[1]/2 +BLOCKSIZE -1)/BLOCKSIZE;
				yuv2bgr<<<numBlocks, BLOCKSIZE>>>(settings[0], settings[1], settings[2], settings[3], G, R);
				auto test = std::chrono::system_clock::now();
				conversion_seconds_average += test - initialization;
				initialization_seconds_average += initialization-start;
				u16ToDouble<<<numBlocks, BLOCKSIZE>>>(settings[0], settings[1], G, doubleTemporary);
				u16ToDouble<<<numBlocks, BLOCKSIZE>>>(settings[0], settings[1], R, redConverted);
				mtx.lock();
				h_backPropagate(settings[0], settings[1], LAMBDA_GREEN, (float)settings[6]/(float)1000000,
						doubleTemporary, kernelGreen, outputArray, maximaGreen, true);		
				h_backPropagate(settings[0],settings[1], LAMBDA_RED, (float)settings[5]/(float)1000000,
						redConverted, kernelRed, convoOutputArray, maximaRed, false);
				mtx.unlock();
				
				
				auto test2 = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = test2-test;
				
				if(opt_verbose) {
					std::cout << "Converting the image format took: " << elapsed_seconds.count() << "s\n";
				}

				cudaUnbindTexture(yTex);
				cudaUnbindTexture(uvTex);
				
				cudaEGLStreamConsumerReleaseFrame(&conn, resource, 0);
				
				auto end = std::chrono::system_clock::now();
				elapsed_seconds = end-start;
				elapsed_seconds_average +=elapsed_seconds;
				final_count++;
				send_points = true;
				
				if(opt_verbose) {
					std::cout << "This cycle took: " << elapsed_seconds.count() << "s\n";
					printf("%d\n", quitSequence);
				}

				cycles++;				
			}
			std::cout << "average complete: " << elapsed_seconds_average.count()/final_count << "s\n";
			iCaptureSession->waitForIdle();
			
			cudaFree(G);
			cudaFree(R);
			
			cudaFree(doubleArray);
			cudaFree(convoOutputArray);
			cudaFree(redConverted);
			cudaFree(maximaGreen);
			cudaFree(maximaRed);
			cudaFree(convolutionMaskGreen);
			cudaFree(convolutionMaskRed);
			cudaFree(convoOutputArrayRed);
			cudaFree(kernelGreen);
			cudaFree(kernelRed);
			
			cudaEGLStreamConsumerDisconnect(&conn);
			iStream->disconnect();
			outputStream.reset();
		}
	}

	printf("consumer_thread: ended\n");
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	// https://www.opencv-srf.com/2011/11/mouse-events.html
	if ( event == CV_EVENT_MOUSEMOVE )
     {
        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
        if (touch_kill) {
        	force_exit = true;
      	}
     }
}

void print_thread(){
	printf("print_thread: started\n");

	float* tempArray;
	float* tempArray2;
	while(true){
		while(sleeping && connected && !force_exit){}
		if (force_exit) break;
			
		cudaMalloc(&tempArray, sizeof(float)*settings[0]*settings[1]);
		cudaMalloc(&tempArray2, sizeof(float)*settings[0]*settings[1]);
		float* output = (float*)malloc(sizeof(float)*settings[0]*settings[1]);
		float* output2 = (float*)malloc(sizeof(float)*settings[0]*settings[1]);
		cv::namedWindow("Basic Visualization", CV_WINDOW_NORMAL);
		cv::setWindowProperty("Basic Visualization", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		//set the callback function for any mouse event
     	cv::setMouseCallback("Basic Visualization", CallBackFunc, NULL);

		while(!initialized && connected && !force_exit){}
		if (force_exit) break;

		while(!sleeping && connected){
			if(cycles >= 3){
				cycles = 0;
				mtx.lock();
				cudaMemcpy(tempArray, maximaGreen, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToDevice);
				cudaMemcpy(tempArray2, outputArray, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToDevice);
				mtx.unlock();
				cudaMemcpy(output, tempArray, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToHost);
				cudaMemcpy(output2, tempArray2, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToHost);
				const cv::Mat img(cv::Size(settings[0], settings[1]), CV_32F, output);
				const cv::Mat img2(cv::Size(settings[0], settings[1]), CV_32F, output2);

				const cv::Mat img2_trans(cv::Size(settings[0], settings[1]), CV_32F);

				cv::flip(img2, img2_trans, -1);
				cv::transpose(img2_trans, img2);

				const cv::Mat result = img2;
				cv::imshow("Basic Visualization", result);
				cv::waitKey(1);
			}
			else{
				usleep(5000);
			}

			if (force_exit) break;
		}
		cudaFree(tempArray);
		cudaFree(tempArray2);
		free(output);
		free(output2);
	}

	printf("print_thread: ended\n");
}


void output_thread(){
	printf("output_thread: started\n");

	float *temporary;
	float *temporary_red_positions;
	float *temporary_green_positions;
	int *sorted_red_positions;
	int *sorted_green_positions;
	while(true){
		while(sleeping && !force_exit){}
		if (force_exit) break;

		cudaMalloc(&temporary, settings[0]*settings[1]*sizeof(float));
		cudaMalloc(&temporary_red_positions, settings[0]*settings[1]*sizeof(float));
		cudaMalloc(&temporary_green_positions, settings[0]*settings[1]*sizeof(float));
		
		char* buffer = (char*)malloc(settings[0]*settings[1]*sizeof(float));
		
		while(connected && !sleeping){
			if(requested_image){
				mtx.lock();
				switch (requested_type){
					case BACKPROPAGATED:
						cudaMemcpy(temporary, outputArray, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToDevice);
						break;
					case RAW_G:
						cudaMemcpy(temporary, doubleTemporary, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToDevice);
						break;
					case RAW_R:
						cudaMemcpy(temporary, redConverted, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToDevice);
						break;
				}	

				mtx.unlock();
				cudaMemcpy(buffer, temporary, sizeof(float)*settings[0]*settings[1], cudaMemcpyDeviceToHost);
				send(client, buffer, sizeof(float)*settings[0]*settings[1], 0);
				printf("Image sent!\n");
				requested_image = false;
			}
			if(send_points){
				auto test3 = std::chrono::system_clock::now();
				outputMtx.lock();
				cudaMemcpy(temporary_green_positions, maximaGreen, sizeof(int)*settings[0]*settings[1], cudaMemcpyDeviceToHost);
				cudaMemcpy(temporary_red_positions, maximaGreen, sizeof(int)*settings[0]*settings[1], cudaMemcpyDeviceToHost);
				outputMtx.unlock();
				int greenCount = processPoints(temporary_green_positions, sorted_green_positions);
				int redCount = processPoints(temporary_red_positions, sorted_red_positions);
				//send(client, buffer, sizeof(float)*settings[0]*settings[1], 0);
				//send(client, buffer, sizeof(float)*settings[0]*settings[1], 0);
				// printf("%d; %d\n", redCount, greenCount);
				send_points = false;
				auto test4 = std::chrono::system_clock::now();
				sorting_seconds_average += test4-test3;
			}

			usleep(1000);

			if (force_exit) break;
		}
		cudaFree(temporary_red_positions);
		cudaFree(temporary_green_positions);
		
		cudaFree(temporary);
		free(buffer);		
	}

	printf("output_thread: ended\n");
}

int main(int argc, char* argv[]){
	// cxxopt.h initialization
  	auto result = parse(argc, argv);

	force_exit = false;
	
	cycles = 0;
	settings[0] = 1024;
	settings[1] = 1024;
	settings[2] = 1195;
	settings[3] = 500;
	settings[4] = 5000000;
	settings[5] = 3100;
	settings[6] = 2400;
	
	quitSequence = false;
	playSequence = false;
	send_points = false;

	if (opt_show) {
		initialized = true;
		connected = true;
		sleeping = false;
		touch_kill = true;
	} else {
		connected = false;
		sleeping = true;
		touch_kill = false;
	}

	thread consumr_thr (consumer_thread);
	thread print_thr (print_thread);
	thread input_thr (input_thread);
	thread output_thr (output_thread);
	thread keyboard_thr (keyboard_thread);
	
	consumr_thr.join();
	print_thr.join();
	// input_thr.join();
	output_thr.join();
	// keyboard_thr.join();


	return 0;
}
