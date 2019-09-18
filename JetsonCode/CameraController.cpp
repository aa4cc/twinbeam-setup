#include "CameraController.h"
#include "Argus/Argus.h"
#include "Definitions.h"
#include "Settings.h"

using namespace Argus;

void CameraController::Initialize(){
	cameraProvider = UniqueObj<CameraProvider>(CameraProvider::create());
	iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
	if(!iCameraProvider){
		printf("ERROR: Failed to establish libargus connection\n");
	}

	std::vector<CameraDevice*> cameraDevices;
	iCameraProvider->getCameraDevices(&cameraDevices);
	if (cameraDevices.size() == 0){
		printf("ERROR: No camera devices available\n");
	}
	CameraDevice *selectedDevice = cameraDevices[0];

	captureSession = UniqueObj<CaptureSession>(iCameraProvider->createCaptureSession(selectedDevice));
	iCaptureSession = interface_cast<ICaptureSession>(captureSession);
	if (!iCaptureSession){
 		printf("ERROR: Failed to create CaptureSession\n");
	}
}

bool CameraController::Start(){
	streamSettings = UniqueObj<OutputStreamSettings>(iCaptureSession->createOutputStreamSettings());
	iStreamSettings = interface_cast<IOutputStreamSettings>(streamSettings);
	iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
	iStreamSettings->setResolution(Size2D<uint32_t>(WIDTH,HEIGHT));

	outputStream = UniqueObj<OutputStream>(iCaptureSession->createOutputStream(streamSettings.get()));
	iStream = interface_cast<IStream>(outputStream);
	if (!iStream){
		printf("ERROR: Failed to create OutputStream\n");
		return false;
	}

	request = UniqueObj<Request>(iCaptureSession->createRequest());
	iRequest = interface_cast<IRequest>(request);
	iRequest->enableOutputStream(outputStream.get());
	
	iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
	iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));
	iSourceSettings->setExposureTimeRange(Range<uint64_t>(Settings::values[STG_EXPOSURE],Settings::values[STG_EXPOSURE]));
	iSourceSettings->setGainRange(Range<float>(0.5,1.5));

	iAutoSettings = interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
	iAutoSettings->setExposureCompensation(0);
	iAutoSettings->setIspDigitalGainRange(Range<float>(0,0));
	iAutoSettings->setWbGains(100);
	iAutoSettings->setColorSaturation(1.0);
	iAutoSettings->setColorSaturationBias(1.0);
	iAutoSettings->setColorSaturationEnable(true);
	iAutoSettings->setAwbLock(true);
	iAutoSettings->setAeAntibandingMode(AE_ANTIBANDING_MODE_OFF);

	iDenoiseSettings = interface_cast<IDenoiseSettings>(request);	
	iDenoiseSettings->setDenoiseMode(DENOISE_MODE_FAST);
	iDenoiseSettings->setDenoiseStrength(1.0);

	return true;
}

EGLStreamKHR CameraController::GetEGLStream(){
	return iStream->getEGLStream();
}

void CameraController::Update(){
	iCaptureSession->capture(request.get());
}

bool CameraController::Stop(){
	iCaptureSession->waitForIdle();
	iStream->disconnect();
	outputStream.reset();

	return true;
}