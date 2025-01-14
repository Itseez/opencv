cmake_minimum_required(VERSION 3.13)

set(target JNIpart)
project(${target} CXX)

if (OPENCV_FROM_SDK)
  message(STATUS "Using OpenCV from local SDK")
  set(ANDROID_OPENCV_COMPONENTS "opencv_java" CACHE STRING "")
else()
  message(STATUS "Using OpenCV from AAR (Maven repo)")
  set(ANDROID_OPENCV_COMPONENTS "OpenCV::opencv_java${OPENCV_VERSION_MAJOR}" CACHE STRING "")
endif()

message(STATUS "ANDROID_ABI=${ANDROID_ABI}")
find_package(OpenCV REQUIRED COMPONENTS ${ANDROID_OPENCV_COMPONENTS})
find_package(OpenCL QUIET)

file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

# For 16k pages support with NDK prior 27
# Details: https://developer.android.com/guide/practices/page-sizes?hl=en
if(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES)
  if(ANDROID_ABI STREQUAL arm64-v8a OR ANDROID_ABI STREQUAL x86_64)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
  endif()
endif()

include_directories("${CMAKE_CURRENT_LIST_DIR}")
add_library(${target} SHARED ${srcs} ${hdrs})

target_link_libraries(${target} ${ANDROID_OPENCV_COMPONENTS} -lGLESv2 -lEGL -llog)

if(OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIRS})
  target_link_libraries(${target} ${OpenCL_LIBRARIES})
  add_definitions("-DOPENCL_FOUND")
elseif(NOT ("${ANDROID_OPENCL_SDK}" STREQUAL ""))
  include_directories(${ANDROID_OPENCL_SDK}/include)
  link_directories(${ANDROID_OPENCL_SDK}/lib)
  target_link_directories(${target} PRIVATE ${ANDROID_OPENCL_SDK}/lib)

  set_target_properties(${target} PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")
  target_link_libraries(${target} -lOpenCL)

  add_definitions("-DOPENCL_FOUND")
  add_definitions("-DCL_HPP_MINIMUM_OPENCL_VERSION=120")
  add_definitions("-DCL_HPP_TARGET_OPENCL_VERSION=120")
  add_definitions("-DCL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY")
endif()
