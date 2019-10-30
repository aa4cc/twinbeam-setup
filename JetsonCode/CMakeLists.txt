cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(3D_scanning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_SOURCE_DIR "/usr/src/tegra_multimedia_api/argus")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_MODULE_PATH}")

find_package(CUDA REQUIRED)
find_package(Argus REQUIRED)
find_package(EGL REQUIRED)


set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS #TX2
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_62,code=sm_62
		-rdc=true -std=c++11
        )

SET(OpenCV_DIR "usr/lib")
SET(HEADERS_FOLDER "{CMAKE_CURRENT_SOURCE_DIR}")
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core highgui)

set(SOURCES
        Trial.cu
		Kernels.cu
		Misc.cpp
		Settings.cpp
		BackPropagator.cu
        )

include_directories(
	 ${ARGUS_INCLUDE_DIR}
	 ${EGL_INCLUDE_DIRS}
	 ${OpenCV_INCLUDE_DIRS}
	 ${HEADERS_FOLDER}
	 )

cuda_add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
        ${ARGUS_LIBRARIES}
	 	${EGL_LIBRARIES}
        cuda
        ${CUDA_CUFFT_LIBRARIES}
		${OpenCV_LIBS}
        )


target_include_directories(${PROJECT_NAME} PRIVATE ${HEADERS_FOLDER})
set_property(TARGET ${PROJECT_NAME}
	PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
#target_link_libraries(${PROJECT_NAME} ${ARGUS} ${EGLSTREAM} ${CUDA} ${OpenCV_LIBS} ${CUFFT})



#FIND_LIBRARY(ARGUS NAMES libnvargus.so PATHS /usr/lib/aarch64-linux-gnu/tegra)
#FIND_LIBRARY(EGLSTREAM NAMES libnvidia-egl-wayland.so PATHS /usr/lib/aarch64-linux-gnu/tegra)
#FIND_LIBRARY(EGL NAMES libEGL.so libEGL.so.1 libEGL.so.1.0.0 PATHS /usr/lib/aarch64-linux-gnu/mesa-egl)
#FIND_LIBRARY(CUDA NAMES libcuda.so libcuda.so.1 libcuda.so.1.1 PATHS /usr/lib/aarch64-linux-gnu/tegra)
#FIND_LIBRARY(CUFFT NAMES libcufft.so libcufft.so.10.0 libcufft.so.10.0.326 PATHS /usr/local/cuda/lib64)