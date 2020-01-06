cmake_minimum_required(VERSION 2.8.3)

project(opencl_test)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 ${CMAKE_CXX_FLAGS} -pthread -D WITH_OPENCL=ON -D WITH_TBB=ON") 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)



find_package(PkgConfig REQUIRED)
pkg_check_modules(udev REQUIRED libudev)


INCLUDE_DIRECTORIES(
include
${OpenCV_INCLUDE_DIR} 
)

add_executable(
opencl_test
test.cpp
)


target_link_libraries(
opencl_test
${OpenCV_LIBS} 
${CMAKE_THREADS_LIBS_INIT} 
)
#linkear librerias
