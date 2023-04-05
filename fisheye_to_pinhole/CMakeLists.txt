CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(Fisheye2Pinhole)

add_compile_options(-std=c++11)

OPTION(DEBUG_ON "Turn on DEBUG Mode" OFF)
IF(DEBUG_ON)
	SET(EXE_NAME Fisheye2Pinhole_d)
ELSE()
	SET(EXE_NAME Fisheye2Pinhole)
ENDIF()

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
"/usr/include/eigen3"
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/fisheye_to_pinhole.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(fisheye_to_pinhole src/fisheye_to_pinhole.cpp)
target_link_libraries(fisheye_to_pinhole 
${OpenCV_LIBS}
${catkin_LIBRARIES}
)

add_executable(save_img src/save_img.cpp)
target_link_libraries(save_img 
${OpenCV_LIBS}
${catkin_LIBRARIES}
)