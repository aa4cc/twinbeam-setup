cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(3D_scanning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include(CTest)
FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS #TX2
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_62,code=sm_62
		-rdc=true -std=c++11
        )

FIND_LIBRARY(ARGUS NAMES libargus.so PATHS /usr/lib/aarch64-linux-gnu/tegra)
FIND_LIBRARY(EGLSTREAM NAMES libnvidia-egl-wayland.so PATHS /usr/lib/aarch64-linux-gnu/tegra)
FIND_LIBRARY(EGL NAMES libEGL.so libEGL.so.1 libEGL.so.1.0.0 PATHS /usr/lib/aarch64-linux-gnu/mesa-egl)
FIND_LIBRARY(CUDA NAMES libcuda.so libcuda.so.1 libcuda.so.1.1 PATHS /usr/lib/aarch64-linux-gnu/tegra)
FIND_LIBRARY(CUFFT NAMES libcufft.so libcufft.so.9.0 libcufft.so.9.0.252 PATHS /usr/local/cuda/lib64)

SET(OpenCV_DIR "usr/lib")
SET(HEADERS_FOLDER "{CMAKE_CURRENT_SOURCE_DIR}")
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core highgui)

include_directories($ENV{HOME}/tegra_multimedia_api/include ${OpenCV_INCLUDE_DIRS} ${HEADERS_FOLDER})
cuda_add_executable(trial Trial.cu Kernels.cu Misc.cpp Settings.cpp BackPropagator.cu)
target_include_directories(trial PRIVATE ${HEADERS_FOLDER})
set_property(TARGET trial
	PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(trial PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(trial ${ARGUS} ${EGLSTREAM} ${CUDA} ${OpenCV_LIBS} ${CUFFT})
