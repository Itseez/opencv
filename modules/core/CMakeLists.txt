set(the_description "The Core Functionality")

ocv_add_dispatched_file(mathfuncs_core SSE2 AVX AVX2 LASX)
ocv_add_dispatched_file(stat SSE4_2 AVX2 LASX)
ocv_add_dispatched_file(arithm SSE2 SSE4_1 AVX2 VSX3 LASX)
ocv_add_dispatched_file(convert SSE2 AVX2 VSX3 LASX)
ocv_add_dispatched_file(convert_scale SSE2 AVX2 LASX)
ocv_add_dispatched_file(count_non_zero SSE2 AVX2 LASX)
ocv_add_dispatched_file(has_non_zero SSE2 AVX2 LASX )
ocv_add_dispatched_file(matmul SSE2 SSE4_1 AVX2 AVX512_SKX NEON_DOTPROD LASX)
ocv_add_dispatched_file(mean SSE2 AVX2 LASX)
ocv_add_dispatched_file(merge SSE2 AVX2 LASX)
ocv_add_dispatched_file(split SSE2 AVX2 LASX)
ocv_add_dispatched_file(sum SSE2 AVX2 LASX)

# dispatching for accuracy tests
ocv_add_dispatched_file_force_all(test_intrin128 TEST SSE2 SSE3 SSSE3 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX)
ocv_add_dispatched_file_force_all(test_intrin256 TEST AVX2 AVX512_SKX LASX)
ocv_add_dispatched_file_force_all(test_intrin512 TEST AVX512_SKX)


set(PARALLEL_ENABLE_PLUGINS_DEFAULT ON)
if(EMSCRIPTEN OR IOS OR XROS OR WINRT)
  set(PARALLEL_ENABLE_PLUGINS_DEFAULT OFF)
endif()
# parallel backends configuration
set(PARALLEL_ENABLE_PLUGINS "${PARALLEL_ENABLE_PLUGINS_DEFAULT}" CACHE BOOL "Allow building parallel plugin support")
# TODO building plugins with OpenCV is not supported yet
#set(PARALLEL_PLUGIN_LIST "" CACHE STRING "List of parallel backends to be compiled as plugins (tbb, openmp or special value 'all')")
#string(REPLACE "," ";" PARALLEL_PLUGIN_LIST "${PARALLEL_PLUGIN_LIST}")  # support comma-separated list (,) too
#string(TOLOWER "${PARALLEL_PLUGIN_LIST}" PARALLEL_PLUGIN_LIST)


ocv_add_module(core
               OPTIONAL opencv_cudev
               WRAP java objc python js)

set(extra_libs "")

if(WINRT AND WINDOWS_STORE AND CMAKE_SYSTEM_VERSION MATCHES "8.0")
  list(APPEND extra_libs ole32.lib)
endif()

if(HAVE_TBB)
  list(APPEND extra_libs tbb)
endif()

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(CV_TRACE AND HAVE_ITT)
  add_definitions(-DOPENCV_WITH_ITT=1)
endif()

# https://github.com/opencv/opencv/issues/24145
if(HAVE_IPP)
  OCV_OPTION(OPENCV_IPP_ENABLE_ALL "Enable all OPENCV_IPP_ options at once" OFF)
  OCV_OPTION(OPENCV_IPP_MEAN   "Enable IPP optimizations for mean (+200Kb in binary size)"                OPENCV_IPP_ENABLE_ALL)
  OCV_OPTION(OPENCV_IPP_MINMAX "Enable IPP optimizations for minMaxLoc/minMaxIdx (+200Kb in binary size)" OPENCV_IPP_ENABLE_ALL)
  OCV_OPTION(OPENCV_IPP_SUM    "Enable IPP optimizations for sum (+100Kb in binary size)"                 OPENCV_IPP_ENABLE_ALL)

  if(OPENCV_IPP_MEAN)
    ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/mean.dispatch.cpp "OPENCV_IPP_MEAN=1")
  endif()

  if(OPENCV_IPP_MINMAX)
    ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/minmax.cpp "OPENCV_IPP_MINMAX=1")
  endif()

  if(OPENCV_IPP_SUM)
    ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/sum.dispatch.cpp "OPENCV_IPP_SUM=1")
  endif()
endif()

file(GLOB lib_cuda_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/*.h")
file(GLOB lib_cuda_hdrs_detail
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/detail/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/cuda/detail/*.h")
file(GLOB_RECURSE module_opencl_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/opencl/*")

source_group("Include\\Cuda Headers"         FILES ${lib_cuda_hdrs})
source_group("Include\\Cuda Headers\\Detail" FILES ${lib_cuda_hdrs_detail})

file(GLOB_RECURSE core_parallel_hdrs
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/parallel/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/parallel/*.h")
ocv_source_group("Include" DIRBASE "${CMAKE_CURRENT_LIST_DIR}/include" FILES ${core_parallel_hdrs})

source_group("Src" FILES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc")

ocv_glob_module_sources(SOURCES "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc"
                        HEADERS ${core_parallel_hdrs} ${module_opencl_hdrs} ${lib_cuda_hdrs} ${lib_cuda_hdrs_detail})

ocv_module_include_directories(${the_module} ${ZLIB_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS})
if(ANDROID AND HAVE_CPUFEATURES)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/system.cpp "HAVE_CPUFEATURES=1")
  ocv_module_include_directories(${CPUFEATURES_INCLUDE_DIRS})
endif()
if(ITT_INCLUDE_DIRS)
  ocv_module_include_directories(${ITT_INCLUDE_DIRS})
endif()
if(HAVE_POSIX_MEMALIGN)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_POSIX_MEMALIGN=1")
endif()
if(HAVE_MALLOC_H)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_MALLOC_H=1")
endif()
if(HAVE_MEMALIGN)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_MEMALIGN=1")
endif()
if(HAVE_WIN32_ALIGNED_MALLOC)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp "HAVE_WIN32_ALIGNED_MALLOC=1")
endif()
if(HAVE_GETAUXVAL)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/system.cpp "HAVE_GETAUXVAL=1")
endif()
if(HAVE_ELF_AUX_INFO)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/system.cpp "HAVE_ELF_AUX_INFO=1")
endif()
if(HAVE_VA_INTEL_OLD_HEADER)
  ocv_append_source_file_compile_definitions("${CMAKE_CURRENT_LIST_DIR}/src/va_intel.cpp" "HAVE_VA_INTEL_OLD_HEADER")
endif()
if(OPENCV_LIBVA_LINK)
  ocv_append_source_file_compile_definitions("${CMAKE_CURRENT_LIST_DIR}/src/va_intel.cpp" "OPENCV_LIBVA_LINK=1")
endif()
if(OPENCV_ENABLE_EGL_INTEROP)
  ocv_append_source_file_compile_definitions("${CMAKE_CURRENT_LIST_DIR}/src/opengl.cpp" "HAVE_EGL_INTEROP")
endif()

option(OPENCV_ENABLE_ALLOCATOR_STATS "Enable Allocator metrics" ON)

if(NOT OPENCV_ENABLE_ALLOCATOR_STATS)
  add_definitions(-DOPENCV_DISABLE_ALLOCATOR_STATS=1)
elseif(HAVE_CXX11 OR DEFINED OPENCV_ALLOCATOR_STATS_COUNTER_TYPE)
  if(NOT DEFINED OPENCV_ALLOCATOR_STATS_COUNTER_TYPE)
    if(HAVE_ATOMIC_LONG_LONG AND OPENCV_ENABLE_ATOMIC_LONG_LONG)
      if(MINGW)
        # command-line generation issue due to space in value, int/int64_t should be used instead
        # https://github.com/opencv/opencv/issues/16990
        message(STATUS "Consider adding OPENCV_ALLOCATOR_STATS_COUNTER_TYPE=int/int64_t according to your build configuration")
      else()
        set(OPENCV_ALLOCATOR_STATS_COUNTER_TYPE "long long")
      endif()
    else()
      set(OPENCV_ALLOCATOR_STATS_COUNTER_TYPE "int")
    endif()
  endif()
  if(DEFINED OPENCV_ALLOCATOR_STATS_COUNTER_TYPE)
    message(STATUS "Allocator metrics storage type: '${OPENCV_ALLOCATOR_STATS_COUNTER_TYPE}'")
    add_definitions("-DOPENCV_ALLOCATOR_STATS_COUNTER_TYPE=${OPENCV_ALLOCATOR_STATS_COUNTER_TYPE}")
  endif()
endif()

if(PARALLEL_ENABLE_PLUGINS)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/parallel/parallel.cpp "PARALLEL_ENABLE_PLUGINS=1")
  if(OPENCV_DEBUG_POSTFIX)
    ocv_append_source_file_compile_definitions("${CMAKE_CURRENT_LIST_DIR}/src/parallel/parallel.cpp" "DEBUG_POSTFIX=${OPENCV_DEBUG_POSTFIX}")
  endif()
endif()

if(HAVE_CUDA)
  if(NOT HAVE_opencv_cudev)
    message(FATAL_ERROR "CUDA: OpenCV requires enabled 'cudev' module from 'opencv_contrib' repository: https://github.com/opencv/opencv_contrib")
  endif()
  if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
   ocv_module_include_directories(${CUDAToolkit_INCLUDE_DIRS})
  endif()
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wenum-compare -Wunused-function -Wshadow)
endif()

ocv_create_module(${extra_libs})

ocv_target_link_libraries(${the_module} PRIVATE
    "${ZLIB_LIBRARIES}" "${OPENCL_LIBRARIES}" "${VA_LIBRARIES}"
    "${OPENGL_LIBRARIES}" "${OPENGL_egl_LIBRARY}"
    "${LAPACK_LIBRARIES}" "${CPUFEATURES_LIBRARIES}" "${HALIDE_LIBRARIES}"
    "${ITT_LIBRARIES}"
    "${OPENCV_HAL_LINKER_LIBS}"
)

if(OPENCV_CORE_EXCLUDE_C_API)
  ocv_target_compile_definitions(${the_module} PRIVATE "OPENCV_EXCLUDE_C_API=1")
endif()

if(OPENCV_DISABLE_THREAD_SUPPORT)
  ocv_target_compile_definitions(${the_module} PUBLIC "OPENCV_DISABLE_THREAD_SUPPORT=1")
endif()

if(OPENCV_DISABLE_ENV_SUPPORT)
  ocv_append_source_file_compile_definitions(${CMAKE_CURRENT_SOURCE_DIR}/src/system.cpp "NO_GETENV")
endif()

if(OPENCV_SEMIHOSTING)
  ocv_target_compile_definitions(${the_module} PRIVATE "-DOPENCV_SEMIHOSTING")
endif(OPENCV_SEMIHOSTING)

if(DEFINED OPENCV_ALGO_HINT_DEFAULT)
  ocv_target_compile_definitions(${the_module} PRIVATE "-DOPENCV_ALGO_HINT_DEFAULT=${OPENCV_ALGO_HINT_DEFAULT}")
endif(DEFINED OPENCV_ALGO_HINT_DEFAULT)

if(HAVE_HPX)
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${HPX_LIBRARIES}")
endif()

if(HAVE_OPENMP AND DEFINED OpenMP_CXX_LIBRARIES AND OpenMP_CXX_LIBRARIES)
  ocv_target_link_libraries(${the_module} LINK_PRIVATE "${OpenMP_CXX_LIBRARIES}")
endif()

ocv_add_accuracy_tests()
ocv_add_perf_tests()

ocv_install_3rdparty_licenses(SoftFloat "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SoftFloat/COPYING.txt")


# generate data (samples data) config file
set(OPENCV_DATA_CONFIG_FILE "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv_data_config.hpp")
set(OPENCV_DATA_CONFIG_STR "")

if(CMAKE_INSTALL_PREFIX)
  set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_INSTALL_PREFIX \"${CMAKE_INSTALL_PREFIX}\"
")
endif()
if(OPENCV_OTHER_INSTALL_PATH)
  set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_DATA_INSTALL_PATH \"${OPENCV_OTHER_INSTALL_PATH}\"
")
endif()

set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_BUILD_DIR \"${CMAKE_BINARY_DIR}\"
")

file(RELATIVE_PATH SOURCE_DIR_RELATIVE ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_DATA_BUILD_DIR_SEARCH_PATHS \\
    \"${SOURCE_DIR_RELATIVE}/\"
")

if(WIN32)
  file(RELATIVE_PATH INSTALL_DATA_DIR_RELATIVE "${CMAKE_INSTALL_PREFIX}/${OPENCV_BIN_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${OPENCV_OTHER_INSTALL_PATH}")
else()
  file(RELATIVE_PATH INSTALL_DATA_DIR_RELATIVE "${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${OPENCV_OTHER_INSTALL_PATH}")
endif()
list(APPEND OPENCV_INSTALL_DATA_DIR_RELATIVE "${INSTALL_DATA_DIR_RELATIVE}")
string(REPLACE ";" "\",\\\n    \"" OPENCV_INSTALL_DATA_DIR_RELATIVE_STR "\"${OPENCV_INSTALL_DATA_DIR_RELATIVE}\"")
set(OPENCV_DATA_CONFIG_STR "${OPENCV_DATA_CONFIG_STR}
#define OPENCV_INSTALL_DATA_DIR_RELATIVE ${OPENCV_INSTALL_DATA_DIR_RELATIVE_STR}
")

if(EXISTS "${OPENCV_DATA_CONFIG_FILE}")
  file(READ "${OPENCV_DATA_CONFIG_FILE}" __content)
endif()
if(NOT OPENCV_DATA_CONFIG_STR STREQUAL "${__content}")
  file(WRITE "${OPENCV_DATA_CONFIG_FILE}" "${OPENCV_DATA_CONFIG_STR}")
endif()
