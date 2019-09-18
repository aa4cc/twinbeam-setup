#include "CameraController.h"
#include "Argus/Argus.h"
#include "Definitions.h"
#include "Settings.h"

using namespace Argus;

CameraController::CameraController(){
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
	CameraController::streamSettings = UniqueObj<OutputStreamSettings>(CameraController::iCaptureSession->createOutputStreamSettings());
	CameraController::iStreamSettings = interface_cast<IOutputStreamSettings>(CameraController::streamSettings);
	CameraController::iStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
	CameraController::iStreamSettings->setResolution(Size2D<uint32_t>(WIDTH,HEIGHT));

	CameraController::outputStream = UniqueObj<OutputStream>(CameraController::iCaptureSession->createOutputStream(CameraController::streamSettings.get()));
	CameraController::iStream = interface_cast<IStream>(CameraController::outputStream);
	if (!CameraController::iStream){
		printf("ERROR: Failed to create OutputStream\n");
		return false;
	}

	CameraController::request = UniqueObj<Request>(CameraController::iCaptureSession->createRequest());
	CameraController::iRequest = interface_cast<IRequest>(CameraController::request);
	CameraController::iRequest->enableOutputStream(CameraController::outputStream.get());
	
	CameraController::iSourceSettings = interface_cast<ISourceSettings>(CameraController::iRequest->getSourceSettings());
	CameraController::iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));
	CameraController::iSourceSettings->setExposureTimeRange(Range<uint64_t>(Settings::values[STG_EXPOSURE],Settings::values[STG_EXPOSURE]));
	CameraController::iSourceSettings->setGainRange(Range<float>(0.5,1.5));

	CameraController::iAutoSettings = interface_cast<IAutoControlSettings>(CameraController::iRequest->getAutoControlSettings());
	CameraController::iAutoSettings->setExposureCompensation(0);
	CameraController::iAutoSettings->setIspDigitalGainRange(Range<float>(0,0));
	CameraController::iAutoSettings->setWbGains(100);
	CameraController::iAutoSettings->setColorSaturation(1.0);
	CameraController::iAutoSettings->setColorSaturationBias(1.0);
	CameraController::iAutoSettings->setColorSaturationEnable(true);
	CameraController::iAutoSettings->setAwbLock(true);
	CameraController::iAutoSettings->setAeAntibandingMode(AE_ANTIBANDING_MODE_OFF);

	CameraController::iDenoiseSettings = interface_cast<IDenoiseSettings>(CameraController::request);	
	CameraController::iDenoiseSettings->setDenoiseMode(DENOISE_MODE_FAST);
	CameraController::iDenoiseSettings->setDenoiseStrength(1.0);

	return true;
}

EGLStreamKHR CameraController::GetEGLStream(){
	return CameraController::iStream->getEGLStream();
}

void CameraController::Update(){
	CameraController::iCaptureSession->capture(CameraController::request.get());
}

bool CameraController::Stop(){
	CameraController::iCaptureSession->waitForIdle();
	CameraController::iStream->disconnect();
	CameraController::outputStream.reset();

	return true;
}