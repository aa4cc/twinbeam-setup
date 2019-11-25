#include "camera_thread.h"
#include "Definitions.h"
#include "Kernels.h"
#include "cuda.h"
#include "cudaEGL.h"
#include "Argus/Argus.h"
#include "EGLStream/EGLStream.h"
#include "EGL/egl.h"
#include "cuda_egl_interop.h"
#include "Settings.h"
#include "argpars.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define dSTG_WIDTH Settings::values[STG_WIDTH]
#define dSTG_HEIGHT Settings::values[STG_HEIGHT]

using namespace std;
using namespace Argus;
using namespace EGLStream;

ImageData<uint8_t> Camera::G;
ImageData<uint8_t> Camera::R;
uint32_t Camera::img_produced 		= 0;
uint32_t Camera::img_processed 		= 0;


cudaError_t res;

EGLStreamKHR eglStream;

cudaArray_t yArray;
cudaArray_t uvArray;
texture<unsigned char, 2, cudaReadModeElementType> yTexRef;
texture<uchar2, 2, cudaReadModeElementType> uvTexRef;
cudaChannelFormatDesc yChannelDesc;
cudaChannelFormatDesc uvChannelDesc;
const textureReference* uvTex;
const textureReference* yTex;

int numBlocks;

#define  CLAMP_F2UINT8(in) ((in) > 255 ? 255: (in))

__global__ void yuv2bgr(int width, int height, int offset_x, int offset_y,
						uint8_t* G, uint8_t* R)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int count = width*height;
            int tx, tx2, ty, ty2;
            float y1, y2;
            float u1, v1, v2;
            for (int i = index; i < count; i += stride)
            {
            	ty 	= i/width + offset_y;
            	ty2 = i/width + offset_y;
				tx 	= i%width + offset_x;
				tx2 = i%width + offset_x + (512);
            	y1 = (float)((tex2D<unsigned char>(yTexRef, (float)tx+0.5f, (float)ty+0.5f) - (float)16) * 1.164383f);
            	y2 = (float)((tex2D<unsigned char>(yTexRef, (float)tx2+0.5f, (float)ty2+0.5f) - (float)16) * 1.164383f);
            	u1 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
					  (float)(ty/2)+(float)(ty%2)+0.5f).x - 128) * 0.391762f;
            	v2 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx2/2)+(float)(tx2%2)+0.5f,
            	     (float)(ty2/2)+(float)(ty2%2)+0.5f).y - 128) * 1.596027f;
            	v1 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
            	     (float)(ty/2)+(float)(ty%2)+0.5f).y - 128) * 0.812968f;
				G[i] = CLAMP_F2UINT8(y1-u1-v1);
				R[i] = CLAMP_F2UINT8(y2+v2);
            }
		}
		

// 		__global__ void yuv2bgr(int width, int height, int offset_x, int offset_y,
// 			uint8_t* G, uint8_t* R)
// {
// int index = blockIdx.x * blockDim.x + threadIdx.x;
// int stride = blockDim.x * gridDim.x;
// int count = width*height;
// int tx, ty, ty2;
// for (int i = index; i < count; i += stride)
// {
// 	ty = i/width + offset_y;
// 	ty2 = i/width + offset_y - (512);
// 	tx = i%width + offset_x;
	
// 	unsigned char Y_1  	= tex2D<unsigned char>(yTexRef, (float)tx+0.5f, (float)ty+0.5f);
// 	unsigned char Y_2  	= tex2D<unsigned char>(yTexRef, (float)tx+0.5f, (float)ty2+0.5f);
// 	uchar2 UV_1 		= tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f, (float)(ty/2)+(float)(ty%2)+0.5f);
// 	uchar2 UV_2  		= tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,  (float)(ty2/2)+(float)(ty2%2)+0.5f);

// 	uint16_t C_1 =  (uint16_t)Y_1 		- 16;
// 	uint16_t D_1 =  (uint16_t)UV_1.x 	- 128;
// 	uint16_t E_1 =  (uint16_t)UV_1.y 	- 128;

// 	uint16_t C_2 =  (uint16_t)Y_2 		- 16;
// 	uint16_t E_2 =  (uint16_t)UV_2.y 	- 128;

// 	uint16_t G_u16 = (298*C_1 - 100*D_1 - 208*E_1 + 128) >> 8;
// 	uint16_t R_u16 = (298*C_2 + 409*E_2 + 128) >> 8;
// 	G[i] = CLAMP_U16_2_U8(G_u16);
// 	R[i] = CLAMP_U16_2_U8(R_u16);
// }
// }

void Camera::camera_thread(){
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

			ICameraProperties *iCameraProperties = interface_cast<ICameraProperties>(selectedDevice);
			if (!iCameraProperties)
				printf("Failed to get ICameraProperties interface");			

			ISensorMode *iSensorMode;
			std::vector<SensorMode*> sensorModes;
			iCameraProperties->getBasicSensorModes(&sensorModes);
			if (sensorModes.size() == 0)
				printf("Failed to get sensor modes");
		
			if(Options::debug) {
				printf("Available Sensor modes :\n");
				for (uint32_t i = 0; i < sensorModes.size(); i++) {
					iSensorMode = interface_cast<ISensorMode>(sensorModes[i]);
					Size2D<uint32_t> resolution = iSensorMode->getResolution();
					printf("[%u] W=%u H=%u\n", i, resolution.width(), resolution.height());
				}
			}
						
			uint32_t SENSOR_MODE = 1;
			// Check sensor mode index
			if (SENSOR_MODE >= sensorModes.size())
				printf("Sensor mode index is out of range");
			SensorMode *sensorMode = sensorModes[SENSOR_MODE];
			
			ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
			iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/Settings::values[STG_FPS]));
			iSourceSettings->setExposureTimeRange(Range<uint64_t>(Settings::values[STG_EXPOSURE],Settings::values[STG_EXPOSURE]));
			iSourceSettings->setGainRange(Range<float>(50.0,50.0));
			iSourceSettings->setSensorMode(sensorMode);	

			IAutoControlSettings *iAutoSettings = interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
			iAutoSettings->setExposureCompensation(0);
			iAutoSettings->setIspDigitalGainRange(Range<float>(Settings::values[STG_DIGGAIN],Settings::values[STG_DIGGAIN]));
			iAutoSettings->setWbGains(1.0f);
			iAutoSettings->setColorSaturation(1.0);
			iAutoSettings->setColorSaturationBias(1.0);
			iAutoSettings->setColorSaturationEnable(true);
			iAutoSettings->setAwbLock(true);
			iAutoSettings->setAeAntibandingMode(AE_ANTIBANDING_MODE_OFF);
			 
			IDenoiseSettings *iDenoiseSettings = interface_cast<IDenoiseSettings>(request);	
			iDenoiseSettings->setDenoiseMode(DENOISE_MODE_FAST);
			iDenoiseSettings->setDenoiseStrength(1.0);

			Camera::G.create(dSTG_WIDTH, dSTG_HEIGHT);
			Camera::R.create(dSTG_WIDTH, dSTG_HEIGHT);
			
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
			
			//CUDA initialization
			//Main loop

			Camera::img_produced = 0;
			Camera::img_processed = 0;
			while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
			if (Settings::force_exit) break;

			while(!Settings::sleeping && Settings::connected && ! Settings::force_exit){
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

				numBlocks = (Settings::get_area()/2 +BLOCKSIZE -1)/BLOCKSIZE;
				
				Camera::G.mtx.lock();
				Camera::R.mtx.lock();
				yuv2bgr<<<numBlocks, BLOCKSIZE>>>(dSTG_WIDTH, dSTG_HEIGHT,
												Settings::values[STG_OFFSET_X], Settings::values[STG_OFFSET_Y], Camera::G.devicePtr(), Camera::R.devicePtr());
				Camera::G.mtx.unlock();
				Camera::R.mtx.unlock();

				++Camera::img_produced;
				// printf("Produced: %d\t, processed: %d\t\n", Camera::img_produced, Camera::img_processed);
				// Wait until the image is processed
				while(Camera::img_produced != Camera::img_processed && !Settings::force_exit) {
					usleep(500);
				}

				cudaUnbindTexture(yTex);
				cudaUnbindTexture(uvTex);
				
				cudaEGLStreamConsumerReleaseFrame(&conn, resource, 0);
			}
			iCaptureSession->waitForIdle();
			
			Camera::G.release();
			Camera::R.release();
			
			cudaEGLStreamConsumerDisconnect(&conn);
			iEGLOutputStream->disconnect();
			outputStream.reset();
		}
	}

	printf("INFO: camera_thread: ended\n");
}