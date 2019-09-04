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

#define STG_WIDTH Settings::values[STG_WIDTH]
#define STG_HEIGHT Settings::values[STG_HEIGHT]

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

cufftComplex* convolutionFilterBlur;

int client;

// Options
bool opt_verbose	= false;
bool opt_debug		= false;
bool opt_show		= false;

uint16_t *R;
uint16_t *G;

mutex mtx;

int numBlocks;
short cycles;
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

void processPoints(float* greenInputPoints, float* redInputPoints, int* outputGreenCoords, int* outputRedCoords, int* h_count){
	float* points;
	int* greenCoords;
	int* sortedGreenCoords;
	int* redCoords;
	int* sortedRedCoords;
	
	cudaMalloc(&points, 2*Settings::get_area()*sizeof(float));
	cudaMalloc(&greenCoords, Settings::get_area()*sizeof(int));
	cudaMalloc(&redCoords, Settings::get_area()*sizeof(int));
	cudaMalloc(&sortedGreenCoords, Settings::get_area()*sizeof(int));
	cudaMalloc(&sortedRedCoords, Settings::get_area()*sizeof(int));

	cudaMemcpy(points, greenInputPoints, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
	cudaMemcpy(&points[Settings::get_area()], redInputPoints, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
	findPoints<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, points, greenCoords);
	findPoints<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, &points[Settings::get_area()], redCoords);
	thrust::device_ptr<int> greenCoordsPtr(greenCoords);
	thrust::device_ptr<int> redCoordsPtr(redCoords);

	thrust::device_ptr<int> sortedGreenCoordsPtr(sortedGreenCoords);
	thrust::device_ptr<int> sortedRedCoordsPtr(sortedRedCoords);

	auto endGreenPointer = thrust::copy_if(thrust::device, greenCoordsPtr, greenCoordsPtr+Settings::get_area(), sortedGreenCoordsPtr, is_not_zero());
	auto endRedPointer = thrust::copy_if(thrust::device, redCoordsPtr, redCoordsPtr+Settings::get_area(), sortedRedCoordsPtr, is_not_zero());

	h_count[0] = (int)(endGreenPointer - sortedGreenCoordsPtr);
	h_count[1] = (int)(endRedPointer - sortedRedCoordsPtr);

	cudaMemcpy(outputGreenCoords, sortedGreenCoords, sizeof(int)*h_count[0], cudaMemcpyDeviceToHost);
	cudaMemcpy(outputRedCoords, sortedRedCoords, sizeof(int)*h_count[1], cudaMemcpyDeviceToHost);

	cudaFree(points);
	cudaFree(greenCoords);
	cudaFree(redCoords);
	cudaFree(sortedRedCoords);
	cudaFree(sortedGreenCoords);
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
	blurFilter<<<numBlocks, BLOCKSIZE>>>(M, N, 3, image);
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

void keyboard_thread(){
	printf("keyboard_thread: started\n");

	char input;
	while(!Settings::force_exit){
		input = getchar();
		if(input == 's'){
			printf("Putting the process to sleep\n");
			Settings::set_sleeping(true);
			Settings::set_initialized(false);
		}
		else if(input == 'c'){
			printf("Connected the main manipulation computer\n");
			Settings::set_connected(true);
		}
		else if(input == 'w'){
			printf("Starting the program from keyboard\n");
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

	printf("keyboard_thread: ended\n");
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

void input_thread(){
	printf("input_thread: started\n");

	std::string text;
	sockaddr_in sockName;
	sockaddr_in clientInfo; 
	int mainSocket;
	char buf[BUFSIZE];
	socklen_t addrlen;
	MESSAGE_TYPE response;
	
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
	while(!Settings::force_exit){
		
		addrlen = sizeof(clientInfo);
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		cout << "Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
		if (client != -1)
		 {
			 Settings::set_connected(true);
		 }

		while(Settings::connected && !Settings::force_exit){
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
						printf("Can't change settings while the loop is running\n");
					else{
						changeSettings(buf);
						printf("Changed settings\n");
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

	printf("input_thread: ended\n");
}

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
	cudaChannelFormatDesc yChannelDesc;
	cudaChannelFormatDesc uvChannelDesc;

	while(!Settings::force_exit){
		while(Settings::connected && !Settings::force_exit){
			while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
			if (Settings::force_exit) break;
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
			iSourceSettings->setExposureTimeRange(Range<uint64_t>(Settings::values[STG_EXPOSURE],Settings::values[STG_EXPOSURE]));
			iSourceSettings->setGainRange(Range<float>(0.5,1.5));

			IAutoControlSettings *iAutoSettings = interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
			iAutoSettings->setExposureCompensation(0);
			iAutoSettings->setIspDigitalGainRange(Range<float>(0,0));
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
			cudaMalloc(&R, Settings::get_area()*sizeof(uint16_t));
			cudaMalloc(&positionsGreen, Settings::get_area()*sizeof(float));
			cudaMalloc(&positionsRed, Settings::get_area()*sizeof(float));		
			
			cudaMalloc(&convolutionMaskGreen, CONVO_DIM_GREEN*CONVO_DIM_GREEN*sizeof(float));
			cudaMalloc(&convolutionMaskRed, CONVO_DIM_RED*CONVO_DIM_RED*sizeof(float));
			numBlocks = 1024;
			generateConvoMaskGreen<<<numBlocks, BLOCKSIZE>>>(CONVO_DIM_GREEN, CONVO_DIM_GREEN, convolutionMaskGreen);
			generateConvoMaskRed<<<numBlocks, BLOCKSIZE>>>(CONVO_DIM_RED, CONVO_DIM_RED, convolutionMaskRed);
			
			cudaMalloc(&kernelGreen, Settings::get_area()*sizeof(cufftComplex));
			cudaMalloc(&kernelRed, Settings::get_area()*sizeof(cufftComplex));

			cudaMalloc(&convolutionFilterBlur, Settings::get_area()*sizeof(cufftComplex));
			generateBlurFilter<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, convolutionFilterBlur);
			
			transformKernel(STG_WIDTH, STG_HEIGHT, CONVO_DIM_GREEN, convolutionMaskGreen, kernelGreen);
			transformKernel(STG_WIDTH, STG_HEIGHT, CONVO_DIM_RED, convolutionMaskRed, kernelRed);
			
			cudaMalloc(&convoOutputArrayGreen, Settings::get_area()*sizeof(float));
			cudaMalloc(&convoOutputArrayRed, Settings::get_area()*sizeof(float));
			cudaMallocManaged(&current_index, sizeof(int));
			mtx.lock();
			cudaMalloc(&maximaGreen, Settings::get_area()*sizeof(float));
			cudaMalloc(&maximaRed, Settings::get_area()*sizeof(float));
			cudaMalloc(&doubleArray, 2*Settings::get_area()*sizeof(float));
			doubleTemporary = &doubleArray[0];
			outputArray = &doubleArray[Settings::get_area()];
			cudaMalloc(&convoOutputArray, Settings::get_area()*sizeof(float));
			cudaMalloc(&redConverted, Settings::get_area()*sizeof(float));
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

			int final_count = 0;
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
				yuv2bgr<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT,
												Settings::values[STG_OFFSET_X], Settings::values[STG_OFFSET_Y], G, R);
				auto test = std::chrono::system_clock::now();
				u16ToDouble<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, G, doubleTemporary);
				u16ToDouble<<<numBlocks, BLOCKSIZE>>>(STG_WIDTH, STG_HEIGHT, R, redConverted);
				mtx.lock();
				h_backPropagate(STG_WIDTH, STG_HEIGHT, LAMBDA_GREEN, (float)Settings::values[STG_Z_GREEN]/(float)1000000,
						doubleTemporary, kernelGreen, outputArray, maximaGreen, true);		
				h_backPropagate(STG_WIDTH,STG_HEIGHT, LAMBDA_RED, (float)Settings::values[STG_Z_RED]/(float)1000000,
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
				Settings::sent_coords = false;
				
				if(opt_verbose) {
					std::cout << "This cycle took: " << elapsed_seconds.count() << "s\n";
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
        if (Settings::touch_kill) {
        	Settings::set_force_exit(true);
      	}
     }
}

void output_thread(){
	printf("output_thread: started\n");

	float *temporary;
	float *temporary_red_positions;
	float *temporary_green_positions;
	char* buffer;
	while(true){
		while(Settings::sleeping && !Settings::force_exit){}
		if (Settings::force_exit) break;

		cudaMalloc(&temporary, Settings::get_area()*sizeof(float));
		cudaMalloc(&temporary_red_positions, Settings::get_area()*sizeof(float));
		cudaMalloc(&temporary_green_positions, Settings::get_area()*sizeof(float));
		
		while(Settings::connected && !Settings::sleeping){
			if(Settings::requested_image){
				mtx.lock();
				switch (Settings::requested_type){
					case BACKPROPAGATED:
						cudaMemcpy(temporary, outputArray, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
						break;
					case RAW_G:
						cudaMemcpy(temporary, doubleTemporary, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
						break;
					case RAW_R:
						cudaMemcpy(temporary, redConverted, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
						break;
				}	

				mtx.unlock();

				buffer = (char*)malloc(Settings::get_area()*sizeof(float));
				cudaMemcpy(buffer, temporary, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToHost);
				send(client, buffer, sizeof(float)*Settings::get_area(), 0);
				free(buffer);
				printf("Image sent!\n");
				Settings::set_requested_image(false);
			}
			if(!Settings::sent_coords && Settings::requested_coords){
				int* sorted_green_positions = (int*)malloc(sizeof(int)*Settings::get_area());
				int* sorted_red_positions = (int*)malloc(sizeof(int)*Settings::get_area());
				mtx.lock();
				cudaMemcpy(temporary_green_positions, maximaGreen, sizeof(int)*Settings::get_area(), cudaMemcpyDeviceToDevice);
				cudaMemcpy(temporary_red_positions, maximaRed, sizeof(int)*Settings::get_area(), cudaMemcpyDeviceToDevice);
				mtx.unlock();

				int* count = (int*)malloc(sizeof(int)*2);

				processPoints(temporary_green_positions, temporary_red_positions, sorted_green_positions, sorted_red_positions, count);

				buffer = (char*)malloc(sizeof(int)*(2+count[0]+count[1]));

				memcpy(&buffer[0], &count[0], sizeof(int));
				memcpy(&buffer[4], sorted_green_positions, count[0]*sizeof(int));
				memcpy(&buffer[4*(1+count[0])], &count[1], sizeof(int));
				memcpy(&buffer[4*(2+count[0])], sorted_red_positions, count[1]*sizeof(int));

				send(client, buffer, sizeof(int)*(2+count[0]+count[1]), 0);

				free(buffer);
				free(count);
				free(sorted_green_positions);
				free(sorted_red_positions);

				Settings::set_sent_coords(true);
				Settings::set_requested_coords(false);
			}

			if (Settings::force_exit) break;
		}
		cudaFree(temporary_red_positions);
		cudaFree(temporary_green_positions);
		
		cudaFree(temporary);
	}

	printf("output_thread: ended\n");
}

void print_thread(){
	printf("print_thread: started\n");

	float* tempArray;
	float* tempArray2;
	while(true){
		while(Settings::sleeping && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit) break;
			
		cudaMalloc(&tempArray, sizeof(float)*Settings::get_area());
		cudaMalloc(&tempArray2, sizeof(float)*Settings::get_area());
		float* output = (float*)malloc(sizeof(float)*Settings::get_area());
		float* output2 = (float*)malloc(sizeof(float)*Settings::get_area());
		cv::namedWindow("Basic Visualization", CV_WINDOW_NORMAL);
		cv::setWindowProperty("Basic Visualization", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		//set the callback function for any mouse event
     	cv::setMouseCallback("Basic Visualization", CallBackFunc, NULL);

		while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
		if (Settings::force_exit){
			cudaFree(tempArray);
			cudaFree(tempArray2);
			free(output);
			free(output2);
			break;
		} 

		while(!Settings::sleeping && Settings::connected){
			if(cycles >= 3){
				cycles = 0;
				mtx.lock();
				cudaMemcpy(tempArray, maximaGreen, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
				cudaMemcpy(tempArray2, outputArray, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToDevice);
				mtx.unlock();

				cudaMemcpy(output, tempArray, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToHost);
				cudaMemcpy(output2, tempArray2, sizeof(float)*Settings::get_area(), cudaMemcpyDeviceToHost);
				printf("%g \n", output2[STG_HEIGHT*STG_HEIGHT/2 + STG_WIDTH]);
				const cv::Mat img(cv::Size(STG_WIDTH, STG_HEIGHT), CV_32F, output);
				const cv::Mat img2(cv::Size(STG_WIDTH, STG_HEIGHT), CV_32F, output2);

				const cv::Mat img2_trans(cv::Size(STG_WIDTH, STG_HEIGHT), CV_32F);
				const cv::Mat img_trans(cv::Size(STG_WIDTH, STG_HEIGHT), CV_32F);

				cv::flip(img2, img2_trans, -1);
				cv::transpose(img2_trans, img2);
				cv::flip(img, img_trans, -1);
				cv::transpose(img_trans, img);

				const cv::Mat result = img2;
				cv::imshow("Basic Visualization", result);
				cv::waitKey(1);
			}
			else{
				usleep(5000);
			}

			if (Settings::force_exit) break;
		}
		cudaFree(tempArray);
		cudaFree(tempArray2);
		free(output);
		free(output2);
	}

	printf("print_thread: ended\n");
}


int main(int argc, char* argv[]){
	for(int i = 0 ; i < STG_NUMBER_OF_SETTINGS; i++){
		printf("%d\n", Settings::values[i]);
	}
  	auto result = parse(argc, argv);
	
	cycles = 0;

	if (opt_show) {
		Settings::set_initialized(true);
		Settings::set_connected(true);
		Settings::set_sleeping(false);
		Settings::set_touch_kill(true);
	}

	thread consumr_thr (consumer_thread);
	thread print_thr (print_thread);
	thread input_thr (input_thread);
	thread output_thr (output_thread);
	thread keyboard_thr (keyboard_thread);
	
	consumr_thr.join();
	print_thr.join();
	output_thr.join();

	return 0;
}
