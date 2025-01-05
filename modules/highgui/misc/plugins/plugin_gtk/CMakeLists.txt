cmake_minimum_required(VERSION 3.5)
project(opencv_highgui_gtk)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

# scan dependencies
set(WITH_GTK ON)
include("${OpenCV_SOURCE_DIR}/modules/highgui/cmake/init.cmake")

if(NOT HAVE_GTK)
  message(FATAL_ERROR "GTK: NO")
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

set(OPENCV_PLUGIN_DEPS core imgproc imgcodecs)
if(TARGET ocv.3rdparty.gtk4)
  set(__deps ocv.3rdparty.gtk4)
elseif(TARGET ocv.3rdparty.gtk3)
  set(__deps ocv.3rdparty.gtk3)
elseif(TARGET ocv.3rdparty.gtk2)
  set(__deps ocv.3rdparty.gtk2)
elseif(TARGET ocv.3rdparty.gtk)
  set(__deps ocv.3rdparty.gtk)
else()
  message(FATAL_ERROR "Missing dependency target for GTK libraries")
endif()
ocv_create_plugin(highgui "opencv_highgui_gtk" "${__deps}" "GTK" "src/window_gtk.cpp")
if(WITH_OPENGL)
  if(HAVE_GTK2
      AND TARGET ocv.3rdparty.gtkglext
      AND TARGET ocv.3rdparty.gtk_opengl
      AND NOT OPENCV_GTK_DISABLE_GTKGLEXT
      AND NOT OPENCV_GTK_DISABLE_OPENGL
  )
    message(STATUS "OpenGL: YES")
    target_link_libraries(${OPENCV_PLUGIN_NAME} PRIVATE
        ocv.3rdparty.gtkglext ocv.3rdparty.gtk_opengl
    )
  else()
    message(WARNING "OpenGL dependencies are not available!")
  endif()
endif()

if(HAVE_GTK4)
  message(STATUS "GTK4+: ver ${GTK4_VERSION}")
elseif(HAVE_GTK3)
  message(STATUS "GTK3+: ver ${GTK3_VERSION}")
elseif(HAVE_GTK2)
  message(STATUS "GTK2+: ver ${GTK2_VERSION}")
elseif(DEFINED GTK_VERSION)
  message(STATUS "GTK+: ver ${GTK_VERSION}")
else()
  message(STATUS "GTK+: YES")
endif()
