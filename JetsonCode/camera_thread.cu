/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */
 
 #include <unistd.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <thread>
 #include <cmath>
 #include <pthread.h>
 #include "camera_thread.h"
#include "Definitions.h"
#include "Kernels.h"
#include "cuda.h"
#include "cudaEGL.h"
#include "Argus/Argus.h"
#include "EGLStream/EGLStream.h"
#include "EGL/egl.h"
#include "cuda_egl_interop.h"
#include "CameraController.h"

using namespace std;
using namespace Argus;
using namespace EGLStream;

cudaError_t res;
cudaArray_t yArray;
cudaArray_t uvArray;
texture<unsigned char, 2, cudaReadModeElementType> yTexRef;
texture<uchar2, 2, cudaReadModeElementType> uvTexRef;
const textureReference* uvTex;
const textureReference* yTex;
cudaChannelFormatDesc yChannelDesc;
cudaChannelFormatDesc uvChannelDesc;

int numBlocks;

__device__ uint8_t clampfloat2uint8(float in) {
	return (uint8_t)fmin(fmax(in, 0.0f), 255.0f);
}

// Converts the captured image in YUV format stored in yTexRef and uvTexRef to red and green channel stored in G and R arrays
// !Important: the y-axis is flipped and red channel is shifted with respect to the green channel by an offset.
__global__ void yuv2bgr(int width, int height, int offset_x, int offset_y, int offset_R2G_x, int offset_R2G_y,
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
            	ty 	= height - i/width - 1 + offset_y;
            	ty2 = height - i/width - 1 + offset_y + offset_R2G_y;
				tx 	= i%width + offset_x;
				tx2 = i%width + offset_x + offset_R2G_x;
            	y1 = (float)((tex2D<unsigned char>(yTexRef, (float)tx+0.5f, (float)ty+0.5f) - (float)16) * 1.164383f);
            	y2 = (float)((tex2D<unsigned char>(yTexRef, (float)tx2+0.5f, (float)ty2+0.5f) - (float)16) * 1.164383f);
            	u1 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
					  (float)(ty/2)+(float)(ty%2)+0.5f).x - 128) * 0.391762f;
            	v2 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx2/2)+(float)(tx2%2)+0.5f,
            	     (float)(ty2/2)+(float)(ty2%2)+0.5f).y - 128) * 1.596027f;
            	v1 = (float)(tex2D<uchar2>(uvTexRef, (float)(tx/2)+(float)(tx%2)+0.5f,
            	     (float)(ty/2)+(float)(ty%2)+0.5f).y - 128) * 0.812968f;
				G[i] = clampfloat2uint8(y1-u1-v1);
				R[i] = clampfloat2uint8(y2+v2);
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

void camera_thread(AppData& appData){
	if(appData.params.debug) printf("INFO: camera_thread: started\n");

	if(appData.params.rtprio) {
		struct sched_param schparam;
		schparam.sched_priority = 50;
		
		if(appData.params.debug) printf("INFO: camera_thread: setting rt priority to %d\n", schparam.sched_priority);

		int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
		if (s != 0) fprintf(stderr, "WARNING: setting the priority of camera thread failed.\n");
	}
	
	CameraController camController(0, 1, appData.params.verbose, appData.params.debug);
	if(!camController.Initialize()) {
		fprintf(stderr, "ERROR: Unable to initialize the camer!\n");
		appData.exitTheApp();
	}
	
	//CUDA variable declarations

	cudaEglStreamConnection conn;
	cudaGraphicsResource_t resource;
	cudaEglFrame eglFrame;		
	cudaArray_t yArray;
	cudaArray_t uvArray;
	cudaChannelFormatDesc yChannelDesc;
	cudaChannelFormatDesc uvChannelDesc;
	
	while(!appData.appStateIs(AppData::AppState::EXITING)){
		if(appData.params.debug) printf("INFO: camera_thread: waiting for entering the INITIALIZING state\n");

		// Wait till the app enters the INITIALIZING state. If this fails (which could happen only in case of entering the EXITING state), break the loop.
		if(!appData.waitTillState(AppData::AppState::INITIALIZING)) break;

		// The app is in the INITIALIZING state
		// Initialize the camera
		if(!camController.Start(appData.params.img_width, appData.params.img_height, appData.params.cam_exposure, appData.params.cam_analoggain, appData.params.cam_digitalgain)) {
			fprintf(stderr, "ERROR: Unable to start capturing the images from the camera\n");
			appData.exitTheApp();
			break;				
		}

		res = cudaEGLStreamConsumerConnect(&conn, camController.GetEGLStream());
		if (res != cudaSuccess) {
			fprintf(stderr, "ERROR: Unable to connect CUDA to EGLStream as a consumer\n");
			appData.exitTheApp();
			break;
		}

		appData.camIG.create(appData.params.img_width, appData.params.img_height);
		appData.camIR.create(appData.params.img_width, appData.params.img_height);
		
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
		
		// Set the flag indicating that the camera was initialized
		appData.camera_is_initialized = true;

		if(appData.params.debug) printf("INFO: camera_thread: waiting till other App components are initialized\n");

		// Wait till all the components of the App are initialized. If this fails, break the loop.
		if(!appData.waitTillAppIsInitialized()) break;

		// At this point, the app is in the AppData::AppState::RUNNING state.
		if(appData.params.debug) printf("INFO: camera_thread: entering the running stage\n");

		chrono::nanoseconds period_us((int64_t)1e9/appData.params.cam_FPS);
		// Capture the images for as long as the App remains in the RUNNING state
		while(appData.appStateIs(AppData::AppState::RUNNING)){
			chrono::steady_clock::time_point next_time = chrono::steady_clock::now() + period_us;

			camController.NewFrameRequest();

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

			numBlocks = (appData.get_area()/2 +NBLOCKS -1)/NBLOCKS;
			
			{
				// lock all mutexes without a deadlock
				lock(appData.cam_mtx, appData.camIG.mtx, appData.camIR.mtx);
				// make sure all mutexes are unlocked when the scope is left
				lock_guard<mutex> lk(appData.cam_mtx, adopt_lock);
				unique_lock<shared_timed_mutex> G_lk(appData.camIG.mtx, adopt_lock);
				unique_lock<shared_timed_mutex> R_lk(appData.camIR.mtx, adopt_lock);

				yuv2bgr<<<numBlocks, NBLOCKS>>>(appData.params.img_width, appData.params.img_height,
												appData.params.img_offset_X, appData.params.img_offset_Y,
												appData.params.img_offset_R2G_X, appData.params.img_offset_R2G_Y,
												appData.camIG.devicePtr(), appData.camIR.devicePtr());
			}
			appData.cam_cv.notify_all();

			cudaUnbindTexture(yTex);
			cudaUnbindTexture(uvTex);
			
			cudaEGLStreamConsumerReleaseFrame(&conn, resource, 0);

			this_thread::sleep_until(next_time);
		}

		// Notify all threads - just in case some threads are still waiting for a new image
		appData.cam_cv.notify_all();

		// Deinitialize the camera
		if(!camController.Stop()) {
			fprintf(stderr, "ERROR: Unable to stop capturing the images by the camera!\n");
			appData.exitTheApp();
		}					
		
		cudaEGLStreamConsumerDisconnect(&conn);

		// Set the flag indicating that the camera was initialized
		appData.camera_is_initialized = false;		
	}

	if(appData.params.debug) printf("INFO: camera_thread: ended\n");
}