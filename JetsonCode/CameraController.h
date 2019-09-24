#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "Argus/Argus.h"

using namespace Argus;

class CameraController {
private:
	UniqueObj<CameraProvider> cameraProvider;
	ICameraProvider* iCameraProvider;

	UniqueObj<CaptureSession> captureSession;
	ICaptureSession *iCaptureSession;

	UniqueObj<OutputStreamSettings> streamSettings;
	IOutputStreamSettings *iStreamSettings;

	UniqueObj<OutputStream> outputStream;
	IStream* iStream;

	UniqueObj<Request> request;
	IRequest *iRequest;

	ISourceSettings *iSourceSettings;
	IAutoControlSettings *iAutoSettings;
	IDenoiseSettings *iDenoiseSettings;

public:
	void Initialize();
	bool Start();
	EGLStreamKHR GetEGLStream();
	void Update();
	bool Stop();
};

#endif