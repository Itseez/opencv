if(APPLE)
  return()
endif()

if(UNIX)
  find_package(X11 QUIET)
endif()

find_package(PkgConfig QUIET)

SET(OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui)
ocv_check_dependencies(${OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(opengl_samples)
  ocv_include_modules_recurse(${OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS})
  file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
  if(NOT X11_FOUND)
    ocv_list_filterout(all_samples "opengl_interop")
  endif()
  foreach(sample_filename ${all_samples})
    ocv_define_sample(tgt ${sample_filename} opengl)
    ocv_target_link_libraries(${tgt} PRIVATE "${OPENGL_LIBRARIES}" "${OPENCV_OPENGL_SAMPLES_REQUIRED_DEPS}")
    if(sample_filename STREQUAL "opengl_interop.cpp")
      ocv_target_link_libraries(${tgt} PRIVATE ${X11_LIBRARIES})
      ocv_target_include_directories(${tgt} ${X11_INCLUDE_DIR})
    endif()
  endforeach()
endif()

ocv_install_example_src(opengl *.cpp *.hpp CMakeLists.txt)
