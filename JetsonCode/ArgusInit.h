#ifndef ARGUS_INIT_H
#define ARGUS_INIT_H

#include "Argus/Argus.h"
#include "EGLStream/EGLStream.h"
#include "EGL/egl.h"

static const int    DEFAULT_FPS        = 30;
#define WIDTH 4056
#define HEIGHT 3040

using namespace Argus;
using namespace EGLStream;

class ArgusProducer
{
	private:
	
	UniqueObj<Request> *request;
	ICaptureSession *iCaptureSession;
	EGLStreamKHR eglStream;
	
	
	public:
	
	ArgusProducer();
	IStream* iStream;
	int captureFrame();
	int destroyProducer();
	EGLStreamKHR getStream();
	
};


#endif
