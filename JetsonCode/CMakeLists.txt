cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(3D_scanning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CMAKE_SOURCE_DIR "/usr/src/tegra_multimedia_api/argus")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_MODULE_PATH}")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(Argus REQUIRED)
find_package(EGL REQUIRED)


set(CUDA_NVCC_FLAGS #TX2
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_62,code=sm_62
		-rdc=true -std=c++14 -O3
        )

SET(OpenCV_DIR "usr/lib")
SET(HEADERS_FOLDER "{CMAKE_CURRENT_SOURCE_DIR}")
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core highgui cudaarithm cudawarping cudaimgproc cudafilters)

set(SOURCES
        Trial.cpp
		Kernels.cu
		Misc.cpp
		BackPropagator.cu
		BeadsFinder.cu
		camera_thread.cu
		keyboard_thread.cpp
		display_thread.cu
		imgproc_thread.cu
		CameraController.cu
		argpars.cpp
		AppData.cpp
		network.cpp
		ImageData.cu
		# Camera.cu
        )

include_directories(
	 ${ARGUS_INCLUDE_DIR}
	 ${EGL_INCLUDE_DIRS}
	 ${OpenCV_INCLUDE_DIRS}
	 ${HEADERS_FOLDER}
	 )

cuda_add_executable(${PROJECT_NAME} ${SOURCES})

message( ${CUDA_LIBRARIES} )

target_link_libraries(${PROJECT_NAME}
        ${ARGUS_LIBRARIES}
	 	${EGL_LIBRARIES}
		${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
		${OpenCV_LIBS}
        )


target_include_directories(${PROJECT_NAME} PRIVATE ${HEADERS_FOLDER})
set_property(TARGET ${PROJECT_NAME}
	PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)