#include "CameraController.h"
#include "Argus/Argus.h"

using namespace Argus;

CameraController::CameraController(){
	CameraController::cameraProvider = UniqueObj<CameraProvider>(CameraProvider::create());
	CameraController::iCameraProvider = interface_cast<ICameraProvider>(CameraController::cameraProvider);
	if(!CameraController::iCameraProvider){
		printf("ERROR: Failed to establish libargus connection\n");
	}

	std::vector<CameraDevice*> cameraDevices;
	CameraController::iCameraProvider->getCameraDevices(&cameraDevices);
	if (cameraDevices.size() == 0){
		printf("ERROR: No camera devices available\n");
	}
	CameraDevice *selectedDevice = cameraDevices[0];

	CameraController::captureSession = UniqueObj<CaptureSession>(CameraController::iCameraProvider->createCaptureSession(selectedDevice));
	CameraController::iCaptureSession = interface_cast<ICaptureSession>(captureSession);
}