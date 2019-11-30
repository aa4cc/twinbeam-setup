#include <stdio.h>
#include "CameraController.h"
#include "Argus/Argus.h"
#include "cuda.h"

using namespace Argus;

bool CameraController::Initialize(){
	// First we create a CameraProvider, necessary for each project.
	cameraProvider = UniqueObj<CameraProvider>(CameraProvider::create());
	iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
	if(!iCameraProvider){
		fprintf(stderr, "ERROR: Failed to establish libargus connection\n");
		return false;
	}

	// Second we select a device from which to receive pictures (camera)
	std::vector<CameraDevice*> cameraDevices;
	Argus::Status status = iCameraProvider->getCameraDevices(&cameraDevices);
	if (status != Argus::STATUS_OK)
    {
        fprintf(stderr, "ERROR: Failed to get camera devices from provider.\n");
        return false;
    }
	if (cameraDevices.size() == 0){
		fprintf(stderr, "ERROR: No camera devices available\n");
		return false;
	}
	CameraDevice *selectedDevice = cameraDevices[0];

	// Set the sensor mode
	ICameraProperties *iCameraProperties = interface_cast<ICameraProperties>(selectedDevice);
	if (!iCameraProperties) {
		fprintf(stderr, "Failed to get ICameraProperties interface\n");
		return false;	
	}

	std::vector<SensorMode*> sensorModes;
	iCameraProperties->getBasicSensorModes(&sensorModes);
	if (sensorModes.size() == 0) {
		fprintf(stderr, "Failed to get sensor modes\n");
		return false;
	}

	if(f_debug) {
		ISensorMode *iSensorMode;
		printf("Available Sensor modes :\n");
		for (uint32_t i = 0; i < sensorModes.size(); i++) {
			iSensorMode = interface_cast<ISensorMode>(sensorModes[i]);
			Size2D<uint32_t> resolution = iSensorMode->getResolution();
			printf("[%u] W=%u H=%u\n", i, resolution.width(), resolution.height());
		}
	}

	// Check sensor mode index
	if (sensore_mode >= sensorModes.size()) {
		fprintf(stderr, "Sensor mode index is out of range\n");
		return false;
	}
	sensorMode = sensorModes[sensore_mode];
	
	ISensorMode *iSensorMode = interface_cast<ISensorMode>(sensorModes[sensore_mode]);
	resolution = iSensorMode->getResolution();
	if(f_verbose) {
		printf("Chosen sensore mode's image resolution: W=%u H=%u\n", resolution.width(), resolution.height());
	}
	
	// We create a capture session 
	captureSession = UniqueObj<CaptureSession>(iCameraProvider->createCaptureSession(selectedDevice));
	iCaptureSession = interface_cast<ICaptureSession>(captureSession);
	if (!iCaptureSession){
		 fprintf(stderr, "ERROR: Failed to create CaptureSession\n");
		 return false;
	}

	return true;
}

bool CameraController::Start(uint32_t imgWidth, uint32_t imgHeight, uint32_t fps, uint32_t exposure, float analogGain, float digitalGain){
	// Managing the settings for the capture session
	streamSettings = UniqueObj<OutputStreamSettings>(iCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
	iEGLStreamSettings = interface_cast<IEGLOutputStreamSettings>(streamSettings);
	if (!iEGLStreamSettings) {
		fprintf(stderr, "ERROR: Failed to create OutputStreamSettings\n");
		return false;
	}
	iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
	iEGLStreamSettings->setResolution(resolution);

	// Creating an Output stream. This should already create a producer.
	outputStream = UniqueObj<OutputStream>(iCaptureSession->createOutputStream(streamSettings.get()));
	iEGLOutputStream = interface_cast<IEGLOutputStream>(outputStream);
	if (!iEGLOutputStream){
		fprintf(stderr, "ERROR: Failed to create EGLOutputStream\n");
		return false;
	}

	// Managing requests.
	request = UniqueObj<Request>(iCaptureSession->createRequest());
	iRequest = interface_cast<IRequest>(request);
	if (!iRequest) {
		fprintf(stderr, "ERROR: Failed to create Request\n");
		return false;
	}
	iRequest->enableOutputStream(outputStream.get());
	
	iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
	if (!iSourceSettings) {
		fprintf(stderr, "ERROR: Failed to get source settings request interface\n");
		return false;
	}
	iSourceSettings->setSensorMode(sensorMode);
	iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/fps));
	iSourceSettings->setExposureTimeRange(Range<uint64_t>(exposure,exposure));
	iSourceSettings->setGainRange(Range<float>(analogGain,analogGain));

	iAutoSettings = interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
	iAutoSettings->setExposureCompensation(0);
	iAutoSettings->setIspDigitalGainRange(Range<float>(digitalGain,digitalGain));
	iAutoSettings->setWbGains(1.0f);
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
	return iEGLOutputStream->getEGLStream();
}

void CameraController::NewFrameRequest(){
	iCaptureSession->capture(request.get());
}

bool CameraController::Stop(){
	iCaptureSession->waitForIdle();
	iEGLOutputStream->disconnect();
	outputStream.reset();

	return true;
}