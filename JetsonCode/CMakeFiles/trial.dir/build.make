# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Documents/koropvik/Code/JetsonCode

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Documents/koropvik/Code/JetsonCode

# Include any dependencies generated for this target.
include CMakeFiles/trial.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/trial.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/trial.dir/flags.make

CMakeFiles/trial.dir/trial_generated_Trial.cu.o: CMakeFiles/trial.dir/trial_generated_Trial.cu.o.depend
CMakeFiles/trial.dir/trial_generated_Trial.cu.o: CMakeFiles/trial.dir/trial_generated_Trial.cu.o.cmake
CMakeFiles/trial.dir/trial_generated_Trial.cu.o: Trial.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/trial.dir/trial_generated_Trial.cu.o"
	cd /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir && /usr/local/bin/cmake -E make_directory /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir//.
	cd /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir//./trial_generated_Trial.cu.o -D generated_cubin_file:STRING=/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir//./trial_generated_Trial.cu.o.cubin.txt -P /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir//trial_generated_Trial.cu.o.cmake

CMakeFiles/trial.dir/trial_intermediate_link.o: CMakeFiles/trial.dir/trial_generated_Trial.cu.o
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC intermediate link file CMakeFiles/trial.dir/trial_intermediate_link.o"
	/usr/local/cuda/bin/nvcc -gencode arch=compute_62,code=sm_62 -rdc=true -std=c++11 -m64 -ccbin /usr/bin/cc -dlink /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir//./trial_generated_Trial.cu.o -o /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir/./trial_intermediate_link.o

# Object files for target trial
trial_OBJECTS =

# External object files for target trial
trial_EXTERNAL_OBJECTS = \
"/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir/trial_generated_Trial.cu.o" \
"/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir/trial_intermediate_link.o"

trial: CMakeFiles/trial.dir/trial_generated_Trial.cu.o
trial: CMakeFiles/trial.dir/trial_intermediate_link.o
trial: CMakeFiles/trial.dir/build.make
trial: /usr/local/cuda/lib64/libcudart_static.a
trial: /usr/lib/aarch64-linux-gnu/librt.so
trial: /usr/lib/aarch64-linux-gnu/tegra/libargus.so
trial: /usr/lib/aarch64-linux-gnu/tegra/libnvidia-egl-wayland.so
trial: /usr/lib/aarch64-linux-gnu/libcuda.so
trial: /usr/lib/libopencv_highgui.so.3.3.1
trial: /usr/lib/libopencv_videoio.so.3.3.1
trial: /usr/lib/libopencv_imgcodecs.so.3.3.1
trial: /usr/lib/libopencv_imgproc.so.3.3.1
trial: /usr/lib/libopencv_core.so.3.3.1
trial: CMakeFiles/trial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable trial"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/trial.dir/build: trial

.PHONY : CMakeFiles/trial.dir/build

CMakeFiles/trial.dir/requires:

.PHONY : CMakeFiles/trial.dir/requires

CMakeFiles/trial.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/trial.dir/cmake_clean.cmake
.PHONY : CMakeFiles/trial.dir/clean

CMakeFiles/trial.dir/depend: CMakeFiles/trial.dir/trial_generated_Trial.cu.o
CMakeFiles/trial.dir/depend: CMakeFiles/trial.dir/trial_intermediate_link.o
	cd /home/nvidia/Documents/koropvik/Code/JetsonCode && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Documents/koropvik/Code/JetsonCode /home/nvidia/Documents/koropvik/Code/JetsonCode /home/nvidia/Documents/koropvik/Code/JetsonCode /home/nvidia/Documents/koropvik/Code/JetsonCode /home/nvidia/Documents/koropvik/Code/JetsonCode/CMakeFiles/trial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/trial.dir/depend

