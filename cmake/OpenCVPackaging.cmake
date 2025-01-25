ocv_cmake_hook(INIT_CPACK)
if(NOT EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
  message(STATUS "CPack is not found. SKIP")
  return()
endif()

set(CPACK_set_DESTDIR "on")

if(NOT OPENCV_CUSTOM_PACKAGE_INFO)
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open Computer Vision Library")
  set(CPACK_PACKAGE_DESCRIPTION
"OpenCV (Open Source Computer Vision Library) is a free software library for computer vision and machine learning. It was created to provide tools for computer vision applications and help machines better understand the world.
 As computer vision rapidly advances, Area Chairs support OpenCV by helping to develop and implement the latest algorithms in specific areas. 
 Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.")
  set(CPACK_PACKAGE_VENDOR "OpenCV Foundation")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_PACKAGE_CONTACT "admin@opencv.org")
  set(CPACK_PACKAGE_VERSION_MAJOR "${OPENCV_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${OPENCV_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${OPENCV_VERSION_PATCH}")
  set(CPACK_PACKAGE_VERSION "${OPENCV_VCSVERSION}")
endif(NOT OPENCV_CUSTOM_PACKAGE_INFO)

#arch
if(X86)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "i686")
elseif(X86_64)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
elseif(ARM)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "armhf")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "armhf")
elseif(AARCH64)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
elseif(PPC64LE)
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "ppc64el")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "ppc64le")
else()
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
  set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif()

if(CPACK_GENERATOR STREQUAL "DEB")
  set(OPENCV_PACKAGE_ARCH_SUFFIX ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
elseif(CPACK_GENERATOR STREQUAL "RPM")
  set(OPENCV_PACKAGE_ARCH_SUFFIX ${CPACK_RPM_PACKAGE_ARCHITECTURE})
else()
  set(OPENCV_PACKAGE_ARCH_SUFFIX ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${OPENCV_PACKAGE_ARCH_SUFFIX}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${OPENCV_PACKAGE_ARCH_SUFFIX}")

#rpm options
set(CPACK_RPM_COMPONENT_INSTALL TRUE)
set(CPACK_RPM_PACKAGE_SUMMARY ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
set(CPACK_RPM_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_DESCRIPTION})
set(CPACK_RPM_PACKAGE_URL "http://opencv.org")
set(CPACK_RPM_PACKAGE_LICENSE "Apache 2.0")

#deb options
set(CPACK_DEB_COMPONENT_INSTALL TRUE)
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "http://opencv.org")

#display names
set(CPACK_COMPONENT_DEV_DISPLAY_NAME     "Development files")
set(CPACK_COMPONENT_DOCS_DISPLAY_NAME    "Documentation")
set(CPACK_COMPONENT_JAVA_DISPLAY_NAME    "Java bindings")
set(CPACK_COMPONENT_LIBS_DISPLAY_NAME    "Libraries and data")
set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME  "Python bindings")
set(CPACK_COMPONENT_SAMPLES_DISPLAY_NAME "Samples")
set(CPACK_COMPONENT_TESTS_DISPLAY_NAME   "Tests")

#depencencies
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
set(CPACK_COMPONENT_LIBS_REQUIRED TRUE)
set(CPACK_COMPONENT_SAMPLES_DEPENDS libs)
set(CPACK_COMPONENT_DEV_DEPENDS libs)
set(CPACK_COMPONENT_DOCS_DEPENDS libs)
set(CPACK_COMPONENT_JAVA_DEPENDS libs)
set(CPACK_COMPONENT_PYTHON_DEPENDS libs)
set(CPACK_COMPONENT_TESTS_DEPENDS libs)

if(HAVE_CUDA)
  string(REPLACE "." "-" cuda_version_suffix ${CUDA_VERSION})
  if(CUDA_VERSION VERSION_LESS "6.5")
    set(CPACK_DEB_libs_PACKAGE_DEPENDS "cuda-core-libs-${cuda_version_suffix}, cuda-extra-libs-${cuda_version_suffix}")
    set(CPACK_DEB_dev_PACKAGE_DEPENDS "cuda-headers-${cuda_version_suffix}")
  else()
    set(CPACK_DEB_libs_PACKAGE_DEPENDS "cuda-cudart-${cuda_version_suffix}, cuda-npp-${cuda_version_suffix}")
    set(CPACK_DEB_dev_PACKAGE_DEPENDS "cuda-cudart-dev-${cuda_version_suffix}, cuda-npp-dev-${cuda_version_suffix}")
    if(HAVE_CUFFT)
      set(CPACK_DEB_libs_PACKAGE_DEPENDS "${CPACK_DEB_libs_PACKAGE_DEPENDS}, cuda-cufft-${cuda_version_suffix}")
      set(CPACK_DEB_dev_PACKAGE_DEPENDS "${CPACK_DEB_dev_PACKAGE_DEPENDS}, cuda-cufft-dev-${cuda_version_suffix}")
    endif()
    if(HAVE_CUBLAS)
      set(CPACK_DEB_libs_PACKAGE_DEPENDS "${CPACK_DEB_libs_PACKAGE_DEPENDS}, cuda-cublas-${cuda_version_suffix}")
      set(CPACK_DEB_dev_PACKAGE_DEPENDS "${CPACK_DEB_dev_PACKAGE_DEPENDS}, cuda-cublas-dev-${cuda_version_suffix}")
    endif()
    if(HAVE_CUDNN)
      # TODO
      #ex: libcudnn7_7.5.1.10-1+cuda10.1_amd64
      #ex: libcudnn7-dev_7.5.1.10-1+cuda10.1_amd64
    endif()
  endif()
  set(CPACK_COMPONENT_dev_DEPENDS libs)
endif()

if(HAVE_TBB AND NOT BUILD_TBB)
  if(CPACK_DEB_DEV_PACKAGE_DEPENDS)
    set(CPACK_DEB_DEV_PACKAGE_DEPENDS "${CPACK_DEB_DEV_PACKAGE_DEPENDS}, libtbb-dev")
  else()
    set(CPACK_DEB_DEV_PACKAGE_DEPENDS "libtbb-dev")
  endif()
endif()

set(STD_OPENCV_LIBS opencv-data)
set(STD_OPENCV_DEV libopencv-dev)

foreach(module calib3d core dnn features2d flann gapi highgui
               imgcodecs imgproc ml objdetect
               photo stitching ts video videoio)
  if(HAVE_opencv_${module})
    list(APPEND STD_OPENCV_LIBS "libopencv-${module}4.0")
    list(APPEND STD_OPENCV_DEV "libopencv-${module}-dev")
  endif()
endforeach()

string(REPLACE ";" ", " CPACK_COMPONENT_LIBS_CONFLICTS "${STD_OPENCV_LIBS}")
string(REPLACE ";" ", " CPACK_COMPONENT_LIBS_PROVIDES "${STD_OPENCV_LIBS}")
string(REPLACE ";" ", " CPACK_COMPONENT_LIBS_REPLACES "${STD_OPENCV_LIBS}")

string(REPLACE ";" ", " CPACK_COMPONENT_DEV_CONFLICTS "${STD_OPENCV_DEV}")
string(REPLACE ";" ", " CPACK_COMPONENT_DEV_PROVIDES "${STD_OPENCV_DEV}")
string(REPLACE ";" ", " CPACK_COMPONENT_DEV_REPLACES "${STD_OPENCV_DEV}")

set(CPACK_COMPONENT_PYTHON_CONFLICTS python-opencv)
set(CPACK_COMPONENT_PYTHON_PROVIDES python-opencv)
set(CPACK_COMPONENT_PYTHON_REPLACES python-opencv)

set(CPACK_COMPONENT_JAVA_CONFLICTS "libopencv4.0-java, libopencv4.0-jni")
set(CPACK_COMPONENT_JAVA_PROVIDES "libopencv4.0-java, libopencv4.0-jni")
set(CPACK_COMPONENT_JAVA_REPLACES "libopencv4.0-java, libopencv4.0-jni")

set(CPACK_COMPONENT_DOCS_CONFLICTS opencv-doc)
set(CPACK_COMPONENT_SAMPLES_CONFLICTS opencv-doc)

if(NOT OPENCV_CUSTOM_PACKAGE_INFO)
  set(CPACK_COMPONENT_LIBS_DESCRIPTION "Open Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_LIBS_NAME "lib${CMAKE_PROJECT_NAME}")

  set(CPACK_COMPONENT_PYTHON_DESCRIPTION "Python bindings for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_PYTHON_NAME "lib${CMAKE_PROJECT_NAME}-python")

  set(CPACK_COMPONENT_JAVA_DESCRIPTION "Java bindings for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_JAVA_NAME "lib${CMAKE_PROJECT_NAME}-java")

  set(CPACK_COMPONENT_DEV_DESCRIPTION "Development files for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_DEV_NAME "lib${CMAKE_PROJECT_NAME}-dev")

  set(CPACK_COMPONENT_DOCS_DESCRIPTION "Documentation for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_DOCS_NAME "lib${CMAKE_PROJECT_NAME}-docs")

  set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "Samples for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_SAMPLES_NAME "lib${CMAKE_PROJECT_NAME}-samples")

  set(CPACK_COMPONENT_TESTS_DESCRIPTION "Accuracy and performance tests for Open Source Computer Vision Library")
  set(CPACK_DEBIAN_COMPONENT_TESTS_NAME "lib${CMAKE_PROJECT_NAME}-tests")
endif(NOT OPENCV_CUSTOM_PACKAGE_INFO)

ocv_cmake_hook(PRE_CPACK)
include(CPack)
ocv_cmake_hook(POST_CPACK)
