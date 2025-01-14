set(the_description "Image I/O")
ocv_add_module(imgcodecs opencv_imgproc WRAP java objc python)

# ----------------------------------------------------------------------------
#  CMake file for imgcodecs. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

ocv_clear_vars(GRFMT_LIBS)

if(HAVE_WINRT_CX AND NOT WINRT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if (HAVE_AVIF)
  ocv_include_directories(${AVIF_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${AVIF_LIBRARY})
endif()

if(HAVE_JPEG)
  ocv_include_directories(${JPEG_INCLUDE_DIR} ${${JPEG_LIBRARY}_BINARY_DIR})
  list(APPEND GRFMT_LIBS ${JPEG_LIBRARIES})
endif()

if(HAVE_JPEGXL)
  ocv_include_directories(${OPENJPEG_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${OPENJPEG_LIBRARIES})
endif()

if(HAVE_WEBP)
  add_definitions(-DHAVE_WEBP)
  ocv_include_directories(${WEBP_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${WEBP_LIBRARIES})
endif()

if(HAVE_SPNG)
  add_definitions(${SPNG_DEFINITIONS})
  ocv_include_directories(${SPNG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${SPNG_LIBRARY})
endif()

if(HAVE_PNG)
  add_definitions(${PNG_DEFINITIONS})
  ocv_include_directories(${PNG_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${PNG_LIBRARIES})
endif()

if(HAVE_GDCM)
  ocv_include_directories(${GDCM_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${GDCM_LIBRARIES})
endif()

if(HAVE_TIFF)
  ocv_include_directories(${TIFF_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${TIFF_LIBRARIES})
endif()

if(HAVE_JPEGXL)
  ocv_include_directories(${JPEGXL_INCLUDE_DIRS})
  message(STATUS "JPEGXL_INCLUDE_DIRS: ${JPEGXL_INCLUDE_DIRS}")
  list(APPEND GRFMT_LIBS ${JPEGXL_LIBRARIES})
endif()

if(HAVE_OPENJPEG)
  ocv_include_directories(${OPENJPEG_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${OPENJPEG_LIBRARIES})
endif()

if(HAVE_JASPER)
  ocv_include_directories(${JASPER_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${JASPER_LIBRARIES})
  if(OPENCV_IO_FORCE_JASPER)
    add_definitions(-DOPENCV_IMGCODECS_FORCE_JASPER=1)
  else()
    message(STATUS "imgcodecs: Jasper codec is disabled in runtime. Details: https://github.com/opencv/opencv/issues/14058")
  endif()
endif()

if(HAVE_OPENEXR)
  include_directories(SYSTEM ${OPENEXR_INCLUDE_PATHS})
  list(APPEND GRFMT_LIBS ${OPENEXR_LIBRARIES})
  if(OPENCV_IO_FORCE_OPENEXR
      OR NOT BUILD_OPENEXR  # external OpenEXR versions are not disabled
  )
    add_definitions(-DOPENCV_IMGCODECS_USE_OPENEXR=1)
  else()
    message(STATUS "imgcodecs: OpenEXR codec is disabled in runtime. Details: https://github.com/opencv/opencv/issues/21326")
  endif()
endif()

if(HAVE_PNG OR HAVE_TIFF OR HAVE_OPENEXR OR HAVE_SPNG OR HAVE_JPEGXL)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND GRFMT_LIBS ${ZLIB_LIBRARIES})
endif()

if(HAVE_GDAL)
  include_directories(SYSTEM ${GDAL_INCLUDE_DIR})
  list(APPEND GRFMT_LIBS ${GDAL_LIBRARY})
endif()

if(HAVE_IMGCODEC_GIF)
  add_definitions(-DHAVE_IMGCODEC_GIF)
endif()

if(HAVE_IMGCODEC_HDR)
  add_definitions(-DHAVE_IMGCODEC_HDR)
endif()

if(HAVE_IMGCODEC_SUNRASTER)
  add_definitions(-DHAVE_IMGCODEC_SUNRASTER)
endif()

if(HAVE_IMGCODEC_PXM)
  add_definitions(-DHAVE_IMGCODEC_PXM)
endif()

if (HAVE_IMGCODEC_PFM)
  add_definitions(-DHAVE_IMGCODEC_PFM)
endif()

file(GLOB grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/grfmt*.hpp)
file(GLOB grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/grfmt*.cpp)

list(APPEND grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/bitstrm.hpp)
list(APPEND grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/bitstrm.cpp)
list(APPEND grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/rgbe.hpp)
list(APPEND grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/rgbe.cpp)
list(APPEND grfmt_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/exif.hpp)
list(APPEND grfmt_srcs ${CMAKE_CURRENT_LIST_DIR}/src/exif.cpp)

source_group("Src\\grfmts" FILES ${grfmt_hdrs} ${grfmt_srcs})

set(imgcodecs_hdrs
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.hpp
    )

set(imgcodecs_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/loadsave.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
    )

file(GLOB imgcodecs_ext_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/legacy/*.h"
     )

if(APPLE OR APPLE_FRAMEWORK)
  list(APPEND imgcodecs_srcs ${CMAKE_CURRENT_LIST_DIR}/src/apple_conversions.h)
  list(APPEND imgcodecs_srcs ${CMAKE_CURRENT_LIST_DIR}/src/apple_conversions.mm)
endif()
if(IOS OR XROS)
  list(APPEND imgcodecs_srcs ${CMAKE_CURRENT_LIST_DIR}/src/ios_conversions.mm)
  list(APPEND IMGCODECS_LIBRARIES "-framework UIKit")
endif()
if(APPLE AND (NOT IOS) AND (NOT XROS))
  list(APPEND imgcodecs_srcs ${CMAKE_CURRENT_LIST_DIR}/src/macosx_conversions.mm)
  list(APPEND IMGCODECS_LIBRARIES "-framework AppKit")
endif()
if(APPLE_FRAMEWORK)
  list(APPEND IMGCODECS_LIBRARIES "-framework Accelerate" "-framework CoreGraphics" "-framework QuartzCore")
endif()

if(TRUE)
  # these variables are set by 'ocv_append_build_options(IMGCODECS ...)'
  foreach(P ${IMGCODECS_INCLUDE_DIRS})
    ocv_include_directories(${P})
  endforeach()

  foreach(P ${IMGCODECS_LIBRARY_DIRS})
    link_directories(${P})
  endforeach()
endif()

source_group("Src" FILES ${imgcodecs_srcs} ${imgcodecs_hdrs})
source_group("Include" FILES ${imgcodecs_ext_hdrs})
ocv_set_module_sources(HEADERS ${imgcodecs_ext_hdrs} SOURCES ${imgcodecs_srcs} ${imgcodecs_hdrs} ${grfmt_srcs} ${grfmt_hdrs})
ocv_module_include_directories()

ocv_create_module(${GRFMT_LIBS} ${IMGCODECS_LIBRARIES})

macro(ocv_imgcodecs_configure_target)
if(APPLE)
  add_apple_compiler_options(${the_module})
endif()

if(MSVC AND NOT BUILD_SHARED_LIBS AND BUILD_WITH_STATIC_CRT)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)
endmacro()

if(NOT BUILD_opencv_world)
  ocv_imgcodecs_configure_target()
endif()

ocv_add_accuracy_tests()
if(TARGET opencv_test_imgcodecs AND HAVE_JASPER AND "$ENV{OPENCV_IO_ENABLE_JASPER}")
  ocv_target_compile_definitions(opencv_test_imgcodecs PRIVATE OPENCV_IMGCODECS_ENABLE_JASPER_TESTS=1)
endif()
if(TARGET opencv_test_imgcodecs AND HAVE_OPENEXR AND "$ENV{OPENCV_IO_ENABLE_OPENEXR}")
  ocv_target_compile_definitions(opencv_test_imgcodecs PRIVATE OPENCV_IMGCODECS_ENABLE_OPENEXR_TESTS=1)
endif()
if(TARGET opencv_test_imgcodecs AND ((HAVE_PNG AND NOT (PNG_VERSION_STRING VERSION_LESS "1.6.31")) OR HAVE_SPNG))
  # details: https://github.com/glennrp/libpng/commit/68cb0aaee3de6371b81a4613476d9b33e43e95b1
  ocv_target_compile_definitions(opencv_test_imgcodecs PRIVATE OPENCV_IMGCODECS_PNG_WITH_EXIF=1)
endif()
ocv_add_perf_tests()
