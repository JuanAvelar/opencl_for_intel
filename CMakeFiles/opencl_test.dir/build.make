# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/egraf/Documents/eGraf/OPENCL_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/egraf/Documents/eGraf/OPENCL_test

# Include any dependencies generated for this target.
include CMakeFiles/opencl_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/opencl_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencl_test.dir/flags.make

CMakeFiles/opencl_test.dir/test.cpp.o: CMakeFiles/opencl_test.dir/flags.make
CMakeFiles/opencl_test.dir/test.cpp.o: test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/egraf/Documents/eGraf/OPENCL_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencl_test.dir/test.cpp.o"
	/usr/bin/cmake -E time /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencl_test.dir/test.cpp.o -c /home/egraf/Documents/eGraf/OPENCL_test/test.cpp

CMakeFiles/opencl_test.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencl_test.dir/test.cpp.i"
	/usr/bin/cmake -E time /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/egraf/Documents/eGraf/OPENCL_test/test.cpp > CMakeFiles/opencl_test.dir/test.cpp.i

CMakeFiles/opencl_test.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencl_test.dir/test.cpp.s"
	/usr/bin/cmake -E time /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/egraf/Documents/eGraf/OPENCL_test/test.cpp -o CMakeFiles/opencl_test.dir/test.cpp.s

CMakeFiles/opencl_test.dir/test.cpp.o.requires:

.PHONY : CMakeFiles/opencl_test.dir/test.cpp.o.requires

CMakeFiles/opencl_test.dir/test.cpp.o.provides: CMakeFiles/opencl_test.dir/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/opencl_test.dir/build.make CMakeFiles/opencl_test.dir/test.cpp.o.provides.build
.PHONY : CMakeFiles/opencl_test.dir/test.cpp.o.provides

CMakeFiles/opencl_test.dir/test.cpp.o.provides.build: CMakeFiles/opencl_test.dir/test.cpp.o


# Object files for target opencl_test
opencl_test_OBJECTS = \
"CMakeFiles/opencl_test.dir/test.cpp.o"

# External object files for target opencl_test
opencl_test_EXTERNAL_OBJECTS =

opencl_test: CMakeFiles/opencl_test.dir/test.cpp.o
opencl_test: CMakeFiles/opencl_test.dir/build.make
opencl_test: /usr/local/lib/libopencv_stitching.so.3.3.1
opencl_test: /usr/local/lib/libopencv_superres.so.3.3.1
opencl_test: /usr/local/lib/libopencv_videostab.so.3.3.1
opencl_test: /usr/local/lib/libopencv_aruco.so.3.3.1
opencl_test: /usr/local/lib/libopencv_bgsegm.so.3.3.1
opencl_test: /usr/local/lib/libopencv_bioinspired.so.3.3.1
opencl_test: /usr/local/lib/libopencv_ccalib.so.3.3.1
opencl_test: /usr/local/lib/libopencv_cvv.so.3.3.1
opencl_test: /usr/local/lib/libopencv_dpm.so.3.3.1
opencl_test: /usr/local/lib/libopencv_face.so.3.3.1
opencl_test: /usr/local/lib/libopencv_freetype.so.3.3.1
opencl_test: /usr/local/lib/libopencv_fuzzy.so.3.3.1
opencl_test: /usr/local/lib/libopencv_hdf.so.3.3.1
opencl_test: /usr/local/lib/libopencv_img_hash.so.3.3.1
opencl_test: /usr/local/lib/libopencv_line_descriptor.so.3.3.1
opencl_test: /usr/local/lib/libopencv_optflow.so.3.3.1
opencl_test: /usr/local/lib/libopencv_reg.so.3.3.1
opencl_test: /usr/local/lib/libopencv_rgbd.so.3.3.1
opencl_test: /usr/local/lib/libopencv_saliency.so.3.3.1
opencl_test: /usr/local/lib/libopencv_sfm.so.3.3.1
opencl_test: /usr/local/lib/libopencv_stereo.so.3.3.1
opencl_test: /usr/local/lib/libopencv_structured_light.so.3.3.1
opencl_test: /usr/local/lib/libopencv_surface_matching.so.3.3.1
opencl_test: /usr/local/lib/libopencv_tracking.so.3.3.1
opencl_test: /usr/local/lib/libopencv_xfeatures2d.so.3.3.1
opencl_test: /usr/local/lib/libopencv_ximgproc.so.3.3.1
opencl_test: /usr/local/lib/libopencv_xobjdetect.so.3.3.1
opencl_test: /usr/local/lib/libopencv_xphoto.so.3.3.1
opencl_test: /usr/local/lib/libopencv_photo.so.3.3.1
opencl_test: /usr/local/lib/libopencv_shape.so.3.3.1
opencl_test: /usr/local/lib/libopencv_calib3d.so.3.3.1
opencl_test: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.1
opencl_test: /usr/local/lib/libopencv_video.so.3.3.1
opencl_test: /usr/local/lib/libopencv_datasets.so.3.3.1
opencl_test: /usr/local/lib/libopencv_plot.so.3.3.1
opencl_test: /usr/local/lib/libopencv_text.so.3.3.1
opencl_test: /usr/local/lib/libopencv_dnn.so.3.3.1
opencl_test: /usr/local/lib/libopencv_features2d.so.3.3.1
opencl_test: /usr/local/lib/libopencv_flann.so.3.3.1
opencl_test: /usr/local/lib/libopencv_highgui.so.3.3.1
opencl_test: /usr/local/lib/libopencv_ml.so.3.3.1
opencl_test: /usr/local/lib/libopencv_videoio.so.3.3.1
opencl_test: /usr/local/lib/libopencv_imgcodecs.so.3.3.1
opencl_test: /usr/local/lib/libopencv_objdetect.so.3.3.1
opencl_test: /usr/local/lib/libopencv_imgproc.so.3.3.1
opencl_test: /usr/local/lib/libopencv_core.so.3.3.1
opencl_test: CMakeFiles/opencl_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/egraf/Documents/eGraf/OPENCL_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencl_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencl_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencl_test.dir/build: opencl_test

.PHONY : CMakeFiles/opencl_test.dir/build

CMakeFiles/opencl_test.dir/requires: CMakeFiles/opencl_test.dir/test.cpp.o.requires

.PHONY : CMakeFiles/opencl_test.dir/requires

CMakeFiles/opencl_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencl_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencl_test.dir/clean

CMakeFiles/opencl_test.dir/depend:
	cd /home/egraf/Documents/eGraf/OPENCL_test && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/egraf/Documents/eGraf/OPENCL_test /home/egraf/Documents/eGraf/OPENCL_test /home/egraf/Documents/eGraf/OPENCL_test /home/egraf/Documents/eGraf/OPENCL_test /home/egraf/Documents/eGraf/OPENCL_test/CMakeFiles/opencl_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencl_test.dir/depend
