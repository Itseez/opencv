if(HAVE_FASTCV)
  set(FASTCV_HAL_VERSION 0.0.1 CACHE INTERNAL "")
  set(FASTCV_HAL_LIBRARIES "fastcv_hal" CACHE INTERNAL "")
  set(FASTCV_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
  set(FASTCV_HAL_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/fastcv_hal_core.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/fastcv_hal_imgproc.hpp"
    CACHE INTERNAL "")

  file(GLOB FASTCV_HAL_FILES    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

  add_library(fastcv_hal STATIC ${FASTCV_HAL_FILES})

  target_include_directories(fastcv_hal PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/core/include
    ${CMAKE_SOURCE_DIR}/modules/imgproc/include
    ${FASTCV_HAL_INCLUDE_DIRS} ${FastCV_INCLUDE_PATH})

  target_link_libraries(fastcv_hal PUBLIC ${FASTCV_LIBRARY})

  set_target_properties(fastcv_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

  if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(fastcv_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
  endif()

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(fastcv_hal PROPERTIES FOLDER "3rdparty")
  endif()
else()
  message(STATUS "FastCV is not available, disabling related HAL")
endif(HAVE_FASTCV)
