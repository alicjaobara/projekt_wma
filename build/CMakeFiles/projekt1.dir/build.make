# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/alicja/gnu/projekt_wma

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alicja/gnu/projekt_wma/build

# Include any dependencies generated for this target.
include CMakeFiles/projekt1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/projekt1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/projekt1.dir/flags.make

CMakeFiles/projekt1.dir/src/projekt1.cpp.o: CMakeFiles/projekt1.dir/flags.make
CMakeFiles/projekt1.dir/src/projekt1.cpp.o: ../src/projekt1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alicja/gnu/projekt_wma/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/projekt1.dir/src/projekt1.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/projekt1.dir/src/projekt1.cpp.o -c /home/alicja/gnu/projekt_wma/src/projekt1.cpp

CMakeFiles/projekt1.dir/src/projekt1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/projekt1.dir/src/projekt1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alicja/gnu/projekt_wma/src/projekt1.cpp > CMakeFiles/projekt1.dir/src/projekt1.cpp.i

CMakeFiles/projekt1.dir/src/projekt1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/projekt1.dir/src/projekt1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alicja/gnu/projekt_wma/src/projekt1.cpp -o CMakeFiles/projekt1.dir/src/projekt1.cpp.s

CMakeFiles/projekt1.dir/src/projekt1.cpp.o.requires:

.PHONY : CMakeFiles/projekt1.dir/src/projekt1.cpp.o.requires

CMakeFiles/projekt1.dir/src/projekt1.cpp.o.provides: CMakeFiles/projekt1.dir/src/projekt1.cpp.o.requires
	$(MAKE) -f CMakeFiles/projekt1.dir/build.make CMakeFiles/projekt1.dir/src/projekt1.cpp.o.provides.build
.PHONY : CMakeFiles/projekt1.dir/src/projekt1.cpp.o.provides

CMakeFiles/projekt1.dir/src/projekt1.cpp.o.provides.build: CMakeFiles/projekt1.dir/src/projekt1.cpp.o


# Object files for target projekt1
projekt1_OBJECTS = \
"CMakeFiles/projekt1.dir/src/projekt1.cpp.o"

# External object files for target projekt1
projekt1_EXTERNAL_OBJECTS =

projekt1: CMakeFiles/projekt1.dir/src/projekt1.cpp.o
projekt1: CMakeFiles/projekt1.dir/build.make
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
projekt1: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
projekt1: CMakeFiles/projekt1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alicja/gnu/projekt_wma/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable projekt1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/projekt1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/projekt1.dir/build: projekt1

.PHONY : CMakeFiles/projekt1.dir/build

CMakeFiles/projekt1.dir/requires: CMakeFiles/projekt1.dir/src/projekt1.cpp.o.requires

.PHONY : CMakeFiles/projekt1.dir/requires

CMakeFiles/projekt1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/projekt1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/projekt1.dir/clean

CMakeFiles/projekt1.dir/depend:
	cd /home/alicja/gnu/projekt_wma/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alicja/gnu/projekt_wma /home/alicja/gnu/projekt_wma /home/alicja/gnu/projekt_wma/build /home/alicja/gnu/projekt_wma/build /home/alicja/gnu/projekt_wma/build/CMakeFiles/projekt1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/projekt1.dir/depend

