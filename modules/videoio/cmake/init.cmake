macro(add_backend backend_id cond_var)
  if(${cond_var})
    include("${CMAKE_CURRENT_LIST_DIR}/detect_${backend_id}.cmake")
  endif()
endmacro()

macro(ocv_add_external_target name inc link def)
  add_library(ocv.3rdparty.${name} INTERFACE)
  set_target_properties(ocv.3rdparty.${name} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${inc}"
    INTERFACE_LINK_LIBRARIES "${link}"
    INTERFACE_COMPILE_DEFINITIONS "${def}")
  if(NOT BUILD_SHARED_LIBS)
    export(TARGETS ocv.3rdparty.${name} APPEND FILE "${CMAKE_BINARY_DIR}/OpenCVModules.cmake")
    install(TARGETS ocv.3rdparty.${name} EXPORT OpenCVModules)
  endif()
endmacro()

add_backend("ffmpeg" WITH_FFMPEG)
add_backend("gstreamer" WITH_GSTREAMER)
add_backend("v4l" WITH_V4L)

add_backend("aravis" WITH_ARAVIS)
add_backend("dc1394" WITH_1394)
add_backend("gphoto" WITH_GPHOTO2)
add_backend("msdk" WITH_MFX)
add_backend("openni2" WITH_OPENNI2)
add_backend("pvapi" WITH_PVAPI)
add_backend("realsense" WITH_LIBREALSENSE)
add_backend("ximea" WITH_XIMEA)
add_backend("xine" WITH_XINE)

add_backend("avfoundation" WITH_AVFOUNDATION)
add_backend("ios" WITH_CAP_IOS)

add_backend("dshow" WITH_DSHOW)
add_backend("msmf" WITH_MSMF)
