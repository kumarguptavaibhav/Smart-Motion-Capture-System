# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp

# Include any dependencies generated for this target.
include CMakeFiles/pose_extractor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pose_extractor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pose_extractor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose_extractor.dir/flags.make

CMakeFiles/pose_extractor.dir/wrapper.cpp.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/wrapper.cpp.o: ../../wrapper.cpp
CMakeFiles/pose_extractor.dir/wrapper.cpp.o: CMakeFiles/pose_extractor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pose_extractor.dir/wrapper.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pose_extractor.dir/wrapper.cpp.o -MF CMakeFiles/pose_extractor.dir/wrapper.cpp.o.d -o CMakeFiles/pose_extractor.dir/wrapper.cpp.o -c /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/wrapper.cpp

CMakeFiles/pose_extractor.dir/wrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/wrapper.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/wrapper.cpp > CMakeFiles/pose_extractor.dir/wrapper.cpp.i

CMakeFiles/pose_extractor.dir/wrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/wrapper.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/wrapper.cpp -o CMakeFiles/pose_extractor.dir/wrapper.cpp.s

CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o: ../../src/extract_poses.cpp
CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o: CMakeFiles/pose_extractor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o -MF CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o.d -o CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o -c /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/extract_poses.cpp

CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/extract_poses.cpp > CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.i

CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/extract_poses.cpp -o CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.s

CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o: ../../src/human_pose.cpp
CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o: CMakeFiles/pose_extractor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o -MF CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o.d -o CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o -c /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/human_pose.cpp

CMakeFiles/pose_extractor.dir/src/human_pose.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/src/human_pose.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/human_pose.cpp > CMakeFiles/pose_extractor.dir/src/human_pose.cpp.i

CMakeFiles/pose_extractor.dir/src/human_pose.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/src/human_pose.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/human_pose.cpp -o CMakeFiles/pose_extractor.dir/src/human_pose.cpp.s

CMakeFiles/pose_extractor.dir/src/peak.cpp.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/src/peak.cpp.o: ../../src/peak.cpp
CMakeFiles/pose_extractor.dir/src/peak.cpp.o: CMakeFiles/pose_extractor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/pose_extractor.dir/src/peak.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pose_extractor.dir/src/peak.cpp.o -MF CMakeFiles/pose_extractor.dir/src/peak.cpp.o.d -o CMakeFiles/pose_extractor.dir/src/peak.cpp.o -c /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/peak.cpp

CMakeFiles/pose_extractor.dir/src/peak.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/src/peak.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/peak.cpp > CMakeFiles/pose_extractor.dir/src/peak.cpp.i

CMakeFiles/pose_extractor.dir/src/peak.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/src/peak.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/src/peak.cpp -o CMakeFiles/pose_extractor.dir/src/peak.cpp.s

# Object files for target pose_extractor
pose_extractor_OBJECTS = \
"CMakeFiles/pose_extractor.dir/wrapper.cpp.o" \
"CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o" \
"CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o" \
"CMakeFiles/pose_extractor.dir/src/peak.cpp.o"

# External object files for target pose_extractor
pose_extractor_EXTERNAL_OBJECTS =

../pose_extractor.so: CMakeFiles/pose_extractor.dir/wrapper.cpp.o
../pose_extractor.so: CMakeFiles/pose_extractor.dir/src/extract_poses.cpp.o
../pose_extractor.so: CMakeFiles/pose_extractor.dir/src/human_pose.cpp.o
../pose_extractor.so: CMakeFiles/pose_extractor.dir/src/peak.cpp.o
../pose_extractor.so: CMakeFiles/pose_extractor.dir/build.make
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libpython3.11.so
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_gapi.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_stitching.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_alphamat.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_aruco.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_barcode.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_bgsegm.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_bioinspired.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_ccalib.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_cvv.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_dnn_objdetect.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_dnn_superres.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_dpm.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_face.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_freetype.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_fuzzy.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_hdf.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_hfs.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_img_hash.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_intensity_transform.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_line_descriptor.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_mcc.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_quality.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_rapid.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_reg.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_rgbd.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_saliency.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_stereo.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_structured_light.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_superres.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_surface_matching.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_tracking.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_videostab.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_wechat_qrcode.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_xfeatures2d.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_xobjdetect.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_xphoto.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_shape.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_highgui.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_datasets.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_plot.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_text.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_ml.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_phase_unwrapping.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_optflow.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_ximgproc.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_video.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_videoio.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_imgcodecs.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_objdetect.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_calib3d.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_dnn.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_features2d.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_flann.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_photo.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_imgproc.so.4.6.0
../pose_extractor.so: /home/hooman/.local/bin/miniconda3/envs/aiml/lib/libopencv_core.so.4.6.0
../pose_extractor.so: CMakeFiles/pose_extractor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library ../pose_extractor.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose_extractor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose_extractor.dir/build: ../pose_extractor.so
.PHONY : CMakeFiles/pose_extractor.dir/build

CMakeFiles/pose_extractor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose_extractor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose_extractor.dir/clean

CMakeFiles/pose_extractor.dir/depend:
	cd /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp /home/hooman/gitjargan/Posture/lightweight-human-pose-estimation-3d-demo.pytorch/pose_extractor/build/tmp/CMakeFiles/pose_extractor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pose_extractor.dir/depend

