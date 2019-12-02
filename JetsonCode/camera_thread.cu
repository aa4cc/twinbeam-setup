/**
 * @author  Martin Gurtner
 * @author  Viktor-Adam Koropecky
 */
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
#include "CameraController.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

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

#define  CLAMP_F2UINT8(in) ((in) > 255 ? 255: (in))

// Converts the captured image in YUV format stored in yTexRef and uvTexRef to red and green channel stored in G and R arrays
// !Important: the y-axis is flipped and red channel is shifted with respect to the green channel by an offset.
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
            	ty 	= height - i/width - 1 + offset_y;
            	ty2 = height - i/width - 1 + offset_y;
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

void camera_thread(CameraImgI& CamI){
	printf("INFO: camera_thread: started\n");
	
	CameraController camController(0, 1, Options::verbose, Options::debug);
	camController.Initialize();
	
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
			
			camController.Start(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT],Settings::values[STG_FPS], Settings::values[STG_EXPOSURE], Settings::values[STG_ANALOGGAIN], Settings::values[STG_DIGGAIN]);

			res = cudaEGLStreamConsumerConnect(&conn, camController.GetEGLStream());
			if (res != cudaSuccess) {
				fprintf(stderr, "ERROR: Unable to connect CUDA to EGLStream as a consumer\n");
				// return false;
			}

			CamI.G.create(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);
			CamI.R.create(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT]);
			
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

			CamI.img_produced = 0;
			CamI.img_processed = 0;
			while(!Settings::initialized && Settings::connected && !Settings::force_exit){}
			if (Settings::force_exit) break;

			while(!Settings::sleeping && Settings::connected && ! Settings::force_exit){
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

				numBlocks = (Settings::get_area()/2 +BLOCKSIZE -1)/BLOCKSIZE;
				
				CamI.G.mtx.lock();
				CamI.R.mtx.lock();
				yuv2bgr<<<numBlocks, BLOCKSIZE>>>(Settings::values[STG_WIDTH], Settings::values[STG_HEIGHT],
												Settings::values[STG_OFFSET_X], Settings::values[STG_OFFSET_Y], CamI.G.devicePtr(), CamI.R.devicePtr());
				CamI.G.mtx.unlock();
				CamI.R.mtx.unlock();

				++CamI.img_produced;
				// printf("Produced: %d\t, processed: %d\t\n", CamI.img_produced, CamI.img_processed);
				// Wait until the image is processed
				while(CamI.img_produced != CamI.img_processed && !Settings::force_exit) {
					usleep(500);
				}

				cudaUnbindTexture(yTex);
				cudaUnbindTexture(uvTex);
				
				cudaEGLStreamConsumerReleaseFrame(&conn, resource, 0);
			}

			camController.Stop();
			
			CamI.G.release();
			CamI.R.release();
			
			cudaEGLStreamConsumerDisconnect(&conn);
		}
	}

	printf("INFO: camera_thread: ended\n");
}