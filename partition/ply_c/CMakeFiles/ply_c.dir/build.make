# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c

# Include any dependencies generated for this target.
include CMakeFiles/ply_c.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ply_c.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ply_c.dir/flags.make

CMakeFiles/ply_c.dir/ply_c.cpp.o: CMakeFiles/ply_c.dir/flags.make
CMakeFiles/ply_c.dir/ply_c.cpp.o: ply_c.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ply_c.dir/ply_c.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ply_c.dir/ply_c.cpp.o -c /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c/ply_c.cpp

CMakeFiles/ply_c.dir/ply_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ply_c.dir/ply_c.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c/ply_c.cpp > CMakeFiles/ply_c.dir/ply_c.cpp.i

CMakeFiles/ply_c.dir/ply_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ply_c.dir/ply_c.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c/ply_c.cpp -o CMakeFiles/ply_c.dir/ply_c.cpp.s

# Object files for target ply_c
ply_c_OBJECTS = \
"CMakeFiles/ply_c.dir/ply_c.cpp.o"

# External object files for target ply_c
ply_c_EXTERNAL_OBJECTS =

libply_c.so: CMakeFiles/ply_c.dir/ply_c.cpp.o
libply_c.so: CMakeFiles/ply_c.dir/build.make
libply_c.so: /home/thomas/anaconda3/envs/SPP/lib/libboost_numpy38.so.1.74.0
libply_c.so: /home/thomas/anaconda3/envs/SPP/lib/libpython3.8.so
libply_c.so: /home/thomas/anaconda3/envs/SPP/lib/libboost_python38.so.1.74.0
libply_c.so: CMakeFiles/ply_c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libply_c.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ply_c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ply_c.dir/build: libply_c.so

.PHONY : CMakeFiles/ply_c.dir/build

CMakeFiles/ply_c.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ply_c.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ply_c.dir/clean

CMakeFiles/ply_c.dir/depend:
	cd /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c /home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/partition/ply_c/CMakeFiles/ply_c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ply_c.dir/depend

