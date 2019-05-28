#include "ArgusInit.h"
#include "Argus/Argus.h"
#include "EGLStream/EGLStream.h"
#include "EGL/egl.h"
#include "stdio.h"
#include "stdlib.h"

using namespace Argus;
using namespace EGLStream;
using namespace std;

ArgusProducer::ArgusProducer(){
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
	
	// Managing the settings for the capture session.
	UniqueObj<OutputStreamSettings> streamSettings(iCaptureSession->createOutputStreamSettings());
	IOutputStreamSettings *iStreamSettings = interface_cast<IOutputStreamSettings>(streamSettings);
	iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
	iStreamSettings->setResolution(Size2D<uint32_t>(WIDTH,HEIGHT));
	
	// Creating an Output stream. This should already create a producer.
	UniqueObj<OutputStream> outputStream(iCaptureSession->createOutputStream(streamSettings.get()));
	iStream = interface_cast<IStream>(outputStream);
	if (!iStream){
 		printf("Failed to create OutputStream\n");
	}
	
	// Managing requests.
	ArgusProducer::request = new UniqueObj<Request>(iCaptureSession->createRequest());
	IRequest *iRequest = interface_cast<IRequest>(request[0]);
	iRequest->enableOutputStream(outputStream.get());
	
	ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
	iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));
	printf("Successful so far\n");
}

int ArgusProducer::captureFrame(){
	iCaptureSession->capture(ArgusProducer::request->get());
	return 0;
}

int ArgusProducer::destroyProducer(){
	return 1;

}

EGLStreamKHR ArgusProducer::getStream(){
		EGLStreamKHR temp = iStream->getEGLStream();
		printf("this is fubar\n");
		return temp;
}
