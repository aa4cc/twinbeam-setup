#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "Argus/Argus.h"
#include "cuda.h"
#include "cudaEGL.h"

using namespace Argus;

class CameraController {
private:
	UniqueObj<CameraProvider> cameraProvider;
	ICameraProvider* iCameraProvider;

	SensorMode *sensorMode;

	UniqueObj<CaptureSession> captureSession;
	ICaptureSession *iCaptureSession;

	UniqueObj<OutputStreamSettings> streamSettings;
	IEGLOutputStreamSettings *iEGLStreamSettings;

	UniqueObj<OutputStream> outputStream;
	IEGLOutputStream* iEGLOutputStream;

	UniqueObj<Request> request;
	IRequest *iRequest;

	ISourceSettings *iSourceSettings;
	IAutoControlSettings *iAutoSettings;
	IDenoiseSettings *iDenoiseSettings;

	bool f_debug, f_verbose;
	uint32_t camera_id, sensore_mode;
	Size2D<uint32_t> resolution;
public:

	CameraController(uint32_t cam_id, uint32_t sens_mode, bool verb=false, bool dbg=false) : camera_id{cam_id}, sensore_mode{sens_mode}, f_verbose{verb}, f_debug{dbg} {};
	bool Initialize();
	bool Start(uint32_t imgWidth, uint32_t imgHeight, uint32_t fps, uint32_t exposure, float analogGain, float digitalGain);
	EGLStreamKHR GetEGLStream();
	void NewFrameRequest();
	bool Stop();
};

#endif