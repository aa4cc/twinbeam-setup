cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(3D_scanning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CMAKE_SOURCE_DIR "/home/nvidia/tegra_multimedia_api/argus")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_MODULE_PATH}")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(Argus REQUIRED)
find_package(EGL REQUIRED)


set(CUDA_NVCC_FLAGS #Xavier
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_72,code=sm_72
		-rdc=true -std=c++14 -O3
        )

SET(OpenCV_DIR "usr/lib")
SET(HEADERS_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/nlohmann")
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core highgui cudaarithm cudawarping cudaimgproc cudafilters)

set(SOURCES
        Trial.cpp
		${SRC_DIR}/Kernels.cu
		${SRC_DIR}/Phase_Kernels.cu
		${SRC_DIR}/cudaDebug.cu
		${SRC_DIR}/fista.cu
		${SRC_DIR}/Misc.cpp
		${SRC_DIR}/BackPropagator.cu
		${SRC_DIR}/BeadsFinder.cu
		${SRC_DIR}/BeadTracker.cpp
		${SRC_DIR}/camera_thread.cu
		${SRC_DIR}/keyboard_thread.cpp
		${SRC_DIR}/display_thread.cu
		${SRC_DIR}/imgproc_thread.cu
		${SRC_DIR}/CameraController.cu
		${SRC_DIR}/Params.cpp
		${SRC_DIR}/AppData.cpp
		${SRC_DIR}/network.cpp
		${SRC_DIR}/ImageData.cu
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
		sockpp
        )


target_include_directories(${PROJECT_NAME} PRIVATE ${HEADERS_FOLDER})
set_property(TARGET ${PROJECT_NAME}
	PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
