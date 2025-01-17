if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(_apple_device_min_target_os_version "13.3")
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(_apple_device_min_target_os_version "16.4")
elseif(CMAKE_SYSTEM_NAME STREQUAL "watchOS")
  set(_apple_device_min_target_os_version "9.4")
elseif(CMAKE_SYSTEM_NAME STREQUAL "tvOS")
  set(_apple_device_min_target_os_version "16.4")
elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
  set(_apple_device_min_target_os_version "1.0")
endif()

if(DEFINED _apple_device_min_target_os_version AND
   ("${CMAKE_OSX_DEPLOYMENT_TARGET}" VERSION_GREATER "${_apple_device_min_target_os_version}" OR
    "${CMAKE_OSX_DEPLOYMENT_TARGET}" VERSION_EQUAL "${_apple_device_min_target_os_version}"))
  set(_apple_device_has_required_min_os_version ON)
else()
  set(_apple_device_has_required_min_os_version OFF)
endif()

OCV_OPTION(OPENCV_OSX_USE_ACCELERATE_NEW_LAPACK "Use new BLAS/LAPACK interfaces from Accelerate framework on Apple platform" _apple_device_has_required_min_os_version
  VISIBLE_IF APPLE)

macro(_find_header_file_in_dirs VAR NAME)
  unset(${VAR})
  unset(${VAR} CACHE)
  if(" ${ARGN}" STREQUAL " ")
    check_include_file("${NAME}" HAVE_${VAR})
    if(HAVE_${VAR})
      set(${VAR} "${NAME}") # fallback
    else()
      set(${VAR} "")
    endif()
  else()
    find_path(${VAR} "${NAME}" ${ARGN} NO_DEFAULT_PATH)
    if(${VAR})
      set(${VAR} "${${VAR}}/${NAME}")
      unset(${VAR} CACHE)
    else()
      unset(${VAR} CACHE)
      set(${VAR} "")
    endif()
  endif()
endmacro()

macro(ocv_lapack_check)
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" _lapack_impl "${LAPACK_IMPL}")
  message(STATUS "LAPACK(${LAPACK_IMPL}): LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
  _find_header_file_in_dirs(OPENCV_CBLAS_H_PATH_${_lapack_impl} "${LAPACK_CBLAS_H}" "${LAPACK_INCLUDE_DIR}")
  _find_header_file_in_dirs(OPENCV_LAPACKE_H_PATH_${_lapack_impl} "${LAPACK_LAPACKE_H}" "${LAPACK_INCLUDE_DIR}")
  if(NOT OPENCV_CBLAS_H_PATH_${_lapack_impl} OR NOT OPENCV_LAPACKE_H_PATH_${_lapack_impl})
    message(WARNING "LAPACK(${LAPACK_IMPL}): CBLAS/LAPACK headers are not found in '${LAPACK_INCLUDE_DIR}'")
    unset(LAPACK_LIBRARIES)
  else()
    # adding proxy opencv_lapack.h header
    set(CBLAS_H_PROXY_PATH ${CMAKE_BINARY_DIR}/opencv_lapack.h)

    set(_lapack_add_extern_c NOT (APPLE OR OPENCV_SKIP_LAPACK_EXTERN_C) OR OPENCV_FORCE_LAPACK_EXTERN_C)

    set(_lapack_content "// This file is auto-generated\n")
    if(${_lapack_add_extern_c})
      list(APPEND _lapack_content "extern \"C\" {")
    endif()
    if(NOT OPENCV_SKIP_LAPACK_MSVC_FIX)
      list(APPEND _lapack_content "
#ifdef _MSC_VER
#include <complex.h>
#define lapack_complex_float _Fcomplex
#define lapack_complex_double _Dcomplex
#endif
")
    endif()
    list(APPEND _lapack_content "#include \"${OPENCV_CBLAS_H_PATH_${_lapack_impl}}\"")
    if(NOT "${OPENCV_CBLAS_H_PATH_${_lapack_impl}}" STREQUAL "${OPENCV_LAPACKE_H_PATH_${_lapack_impl}}")
      list(APPEND _lapack_content "#include \"${OPENCV_LAPACKE_H_PATH_${_lapack_impl}}\"")
    endif()
    list(APPEND _lapack_content "
#if defined(LAPACK_GLOBAL) || defined(LAPACK_NAME)
/*
 * Using netlib's reference LAPACK implementation version >= 3.4.0 (first with C interface).
 * Use LAPACK_xxxx to transparently (via predefined lapack macros) deal with pre and post 3.9.1 versions.
 * LAPACK 3.9.1 introduces LAPACK_FORTRAN_STRLEN_END and modifies (through preprocessing) the declarations of the following functions used in opencv
 *        sposv_, dposv_, spotrf_, dpotrf_, sgesdd_, dgesdd_, sgels_, dgels_
 * which end up with an extra parameter.
 * So we also need to preprocess the function calls in opencv coding by prefixing them with LAPACK_.
 * The good news is the preprocessing works fine whatever netlib's LAPACK version.
 */
#define OCV_LAPACK_FUNC(f) LAPACK_##f
#else
/* Using other LAPACK implementations so fall back to opencv's assumption until now */
#define OCV_LAPACK_FUNC(f) f##_
#endif
")
    if(${_lapack_add_extern_c})
      list(APPEND _lapack_content "}")
    endif()

    string(REPLACE ";" "\n" _lapack_content "${_lapack_content}")
    ocv_update_file("${CBLAS_H_PROXY_PATH}" "${_lapack_content}")

    if(CMAKE_GENERATOR MATCHES "Visual Studio"  # MSBuild
        AND LAPACK_IMPL STREQUAL "MKL"
        AND ";${LAPACK_LIBRARIES};" MATCHES ";tbb;" AND TARGET tbb
        AND DEFINED TBB_INTERFACE_VERSION AND NOT (TBB_INTERFACE_VERSION LESS 12000)  # oneTBB/oneAPI workaround
    )
      # workaround DEFAULTLIB:tbb12.lib issue
      get_target_property(_tbb_lib tbb IMPORTED_LOCATION)
      if(NOT _tbb_lib)
        get_target_property(_tbb_lib tbb IMPORTED_LOCATION_RELEASE)
      endif()
      if(_tbb_lib AND NOT OPENCV_SKIP_WORKAROUND_MKL_LINK_DIRECTORIES_TBB)
        # MSBuild drops content of 'LIB' environment variable,
        # so pass TBB library directory through `link_directories()`
        get_filename_component(_tbb_lib_dir "${_tbb_lib}" DIRECTORY)
        message(STATUS "MKL: adding '${_tbb_lib_dir}' to link directories (workaround DEFAULTLIB issue)")
        link_directories("${_tbb_lib_dir}")
      elseif(NOT OPENCV_SKIP_WORKAROUND_MKL_DEFAULTLIB)
        # We may have tbb.lib for 'tbb' target, but not 'tbb12.lib'
        ocv_update(OPENCV_MKL_IGNORE_DEFAULTLIB_TBB "tbb12.lib")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:${OPENCV_MKL_IGNORE_DEFAULTLIB_TBB}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:${OPENCV_MKL_IGNORE_DEFAULTLIB_TBB}")
      endif()
    endif()

    # TODO add cache for try_compile() inputs/results

    get_property(__link_directories DIRECTORY PROPERTY LINK_DIRECTORIES)
    if(LAPACK_LINK_LIBRARIES)
      list(APPEND __link_directories ${LAPACK_LINK_LIBRARIES})
    endif()

    set(LAPACK_TRY_COMPILE_DEF "")
    if(LAPACK_IMPL STREQUAL "LAPACK/Apple" AND OPENCV_OSX_USE_ACCELERATE_NEW_LAPACK)
      message(STATUS "LAPACK(${LAPACK_IMPL}): Accelerate New LAPACK is enabled.")
      set(LAPACK_TRY_COMPILE_DEF "-DACCELERATE_NEW_LAPACK")
      add_compile_definitions(ACCELERATE_NEW_LAPACK)
      add_compile_definitions(ACCELERATE_LAPACK_ILP64)
    endif()

    try_compile(__VALID_LAPACK
        "${OpenCV_BINARY_DIR}"
        "${OpenCV_SOURCE_DIR}/cmake/checks/lapack_check.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LAPACK_INCLUDE_DIR}\;${CMAKE_BINARY_DIR}"
                    "-DLINK_DIRECTORIES:STRING=${__link_directories}"
        COMPILE_DEFINITIONS ${LAPACK_TRY_COMPILE_DEF}
        LINK_LIBRARIES ${LAPACK_LIBRARIES}
        OUTPUT_VARIABLE TRY_OUT
    )
    if(NOT __VALID_LAPACK)
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
          "\nLAPACK(${LAPACK_IMPL}) check FAILED:\n"
          "    LAPACK_INCLUDE_DIR: '${LAPACK_INCLUDE_DIR}'\n"
          "    LAPACK_LIBRARIES: '${LAPACK_LIBRARIES}'\n"
          "    LAPACK_LINK_LIBRARIES: '${__link_directories}'\n"
          "    Output:\n${TRY_OUT}\n\n")
      message(STATUS "LAPACK(${LAPACK_IMPL}): Can't build LAPACK check code. This LAPACK version is not supported.")
      unset(LAPACK_LIBRARIES)
    else()
      message(STATUS "LAPACK(${LAPACK_IMPL}): Support is enabled.")
      ocv_include_directories(${LAPACK_INCLUDE_DIR})
      set(HAVE_LAPACK 1)
    endif()
  endif()
endmacro()

if(WITH_LAPACK)
  ocv_update(LAPACK_IMPL "Unknown")
  if(NOT OPENCV_LAPACK_FIND_PACKAGE_ONLY)
    if(NOT LAPACK_LIBRARIES AND NOT OPENCV_LAPACK_DISABLE_MKL)
      include(cmake/OpenCVFindMKL.cmake)
      if(HAVE_MKL)
        set(LAPACK_INCLUDE_DIR  ${MKL_INCLUDE_DIRS})
        set(LAPACK_LIBRARIES    ${MKL_LIBRARIES})
        set(LAPACK_CBLAS_H      "mkl_cblas.h")
        set(LAPACK_LAPACKE_H    "mkl_lapack.h")
        set(LAPACK_IMPL         "MKL")
        ocv_lapack_check()
      endif()
    endif()
    if(NOT LAPACK_LIBRARIES)
      include(cmake/OpenCVFindOpenBLAS.cmake)
      if(OpenBLAS_FOUND)
        set(LAPACK_INCLUDE_DIR  "${OpenBLAS_INCLUDE_DIR};/usr/include")
        set(LAPACK_LIBRARIES    ${OpenBLAS_LIB})
        set(LAPACK_CBLAS_H      "cblas.h")
        set(LAPACK_LAPACKE_H    "lapacke.h")
        set(LAPACK_IMPL         "OpenBLAS")
        ocv_lapack_check()
      endif()
    endif()
    if(NOT LAPACK_LIBRARIES AND UNIX)
      include(cmake/OpenCVFindAtlas.cmake)
      if(ATLAS_FOUND)
        set(LAPACK_INCLUDE_DIR  ${Atlas_INCLUDE_DIR})
        set(LAPACK_LIBRARIES    ${Atlas_LIBRARIES})
        set(LAPACK_CBLAS_H      "cblas.h")
        set(LAPACK_LAPACKE_H    "lapacke.h")
        set(LAPACK_IMPL         "Atlas")
        ocv_lapack_check()
      endif()
    endif()
  endif()

  if(NOT LAPACK_LIBRARIES)
    if(WIN32 AND NOT OPENCV_LAPACK_SHARED_LIBS)
      set(BLA_STATIC 1)
    endif()
    find_package(LAPACK)
    if(LAPACK_FOUND)
      if(NOT DEFINED LAPACKE_INCLUDE_DIR)
        find_path(LAPACKE_INCLUDE_DIR "lapacke.h")
      endif()
      if(NOT DEFINED MKL_LAPACKE_INCLUDE_DIR)
        find_path(MKL_LAPACKE_INCLUDE_DIR "mkl_lapack.h")
      endif()
      if(MKL_LAPACKE_INCLUDE_DIR AND NOT OPENCV_LAPACK_DISABLE_MKL)
        set(LAPACK_INCLUDE_DIR  ${MKL_LAPACKE_INCLUDE_DIR})
        set(LAPACK_CBLAS_H      "mkl_cblas.h")
        set(LAPACK_LAPACKE_H    "mkl_lapack.h")
        set(LAPACK_IMPL         "LAPACK/MKL")
        ocv_lapack_check()
      endif()
      if(NOT HAVE_LAPACK)
        if(LAPACKE_INCLUDE_DIR)
          set(LAPACK_INCLUDE_DIR  ${LAPACKE_INCLUDE_DIR})
          set(LAPACK_CBLAS_H      "cblas.h")
          set(LAPACK_LAPACKE_H    "lapacke.h")
          set(LAPACK_IMPL         "LAPACK/Generic")
          ocv_lapack_check()
        elseif(APPLE)
          set(LAPACK_CBLAS_H      "Accelerate/Accelerate.h")
          set(LAPACK_LAPACKE_H    "Accelerate/Accelerate.h")
          set(LAPACK_IMPL         "LAPACK/Apple")
          ocv_lapack_check()
        endif()
      endif()
    endif()
    if(NOT HAVE_LAPACK)
      unset(LAPACK_LIBRARIES)
      unset(LAPACK_LIBRARIES CACHE)
    endif()
  endif()

  if(NOT LAPACK_LIBRARIES AND APPLE AND NOT OPENCV_LAPACK_FIND_PACKAGE_ONLY)
    set(LAPACK_INCLUDE_DIR  "")
    set(LAPACK_LIBRARIES    "-framework Accelerate")
    set(LAPACK_CBLAS_H      "Accelerate/Accelerate.h")
    set(LAPACK_LAPACKE_H    "Accelerate/Accelerate.h")
    set(LAPACK_IMPL         "Apple")
    ocv_lapack_check()
  endif()

  if(NOT HAVE_LAPACK AND LAPACK_LIBRARIES AND LAPACK_CBLAS_H AND LAPACK_LAPACKE_H)
    ocv_lapack_check()
  endif()

  set(LAPACK_INCLUDE_DIR ${LAPACK_INCLUDE_DIR} CACHE PATH   "Path to BLAS include dir" FORCE)
  set(LAPACK_CBLAS_H     ${LAPACK_CBLAS_H}     CACHE STRING "Alternative name of cblas.h" FORCE)
  set(LAPACK_LAPACKE_H   ${LAPACK_LAPACKE_H}   CACHE STRING "Alternative name of lapacke.h" FORCE)
  set(LAPACK_LIBRARIES   ${LAPACK_LIBRARIES}   CACHE STRING "Names of BLAS & LAPACK binaries (.so, .dll, .a, .lib)" FORCE)
  set(LAPACK_IMPL        ${LAPACK_IMPL}        CACHE STRING "Lapack implementation id" FORCE)
endif()
