# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------
if(DEFINED OPENCV_INITIAL_PASS)  # OpenCV build

  if(ANDROID OR APPLE_FRAMEWORK OR WINRT)
    ocv_module_disable_(python3)
    return()
  elseif(BUILD_opencv_world OR (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug"))
    if(NOT DEFINED BUILD_opencv_python3)
      set(__disable_python3 ON)
    endif()
  endif()

  add_subdirectory(bindings)

  add_subdirectory(test)

  if(__disable_python3)
    ocv_module_disable_(python3)
    return()
  endif()

  add_subdirectory(python3)

else()  # standalone build

  cmake_minimum_required(VERSION 3.13)
  project(OpenCVPython CXX C)
  include("./standalone.cmake")

endif()
