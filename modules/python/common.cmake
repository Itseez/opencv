# This file is included from a subdirectory
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

ocv_add_module(${MODULE_NAME} BINDINGS PRIVATE_REQUIRED opencv_python_bindings_generator)

include_directories(SYSTEM
    "${${PYTHON}_INCLUDE_DIRS}"
    ${${PYTHON}_NUMPY_INCLUDE_DIRS}
)
ocv_module_include_directories(
    "${PYTHON_SOURCE_DIR}/src2"
    "${OPENCV_PYTHON_BINDINGS_DIR}"
)

# try to use dynamic symbols linking with libpython.so
set(OPENCV_FORCE_PYTHON_LIBS OFF CACHE BOOL "")
string(REPLACE "-Wl,--no-undefined" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
if(NOT WIN32 AND NOT APPLE AND NOT OPENCV_PYTHON_SKIP_LINKER_EXCLUDE_LIBS)
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--exclude-libs=ALL")
endif()

ocv_add_library(${the_module} MODULE
  ${PYTHON_SOURCE_DIR}/src2/cv2.cpp
  ${PYTHON_SOURCE_DIR}/src2/cv2_util.cpp
  ${PYTHON_SOURCE_DIR}/src2/cv2_numpy.cpp
  ${PYTHON_SOURCE_DIR}/src2/cv2_convert.cpp
  ${PYTHON_SOURCE_DIR}/src2/cv2_highgui.cpp
  ${cv2_generated_hdrs}
  ${opencv_userdef_hdrs}
  ${cv2_custom_hdr}
)

if(TARGET gen_opencv_python_source)
  add_dependencies(${the_module} gen_opencv_python_source)
endif()

if(TARGET copy_opencv_typing_stubs)
  # Python 3.6+
  add_dependencies(${the_module} copy_opencv_typing_stubs)
endif()

ocv_assert(${PYTHON}_VERSION_MAJOR)
ocv_assert(${PYTHON}_VERSION_MINOR)

if(${PYTHON}_LIMITED_API)
  # support only python3.3+
  ocv_assert(${PYTHON}_VERSION_MAJOR EQUAL 3 AND ${PYTHON}_VERSION_MINOR GREATER 2)
  target_compile_definitions(${the_module} PRIVATE CVPY_DYNAMIC_INIT)
  target_compile_definitions(${the_module} PRIVATE PYTHON3_LIMITED_API_VERSION=${PYTHON3_LIMITED_API_VERSION})
  if(WIN32)
    string(REPLACE
      "python${${PYTHON}_VERSION_MAJOR}${${PYTHON}_VERSION_MINOR}.lib"
      "python${${PYTHON}_VERSION_MAJOR}.lib"
      ${PYTHON}_LIBRARIES
      "${${PYTHON}_LIBRARIES}")
  endif()
endif()

if(APPLE)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
elseif(WIN32 OR OPENCV_FORCE_PYTHON_LIBS)
  if(${PYTHON}_DEBUG_LIBRARIES AND NOT ${PYTHON}_LIBRARIES MATCHES "optimized.*debug")
    ocv_target_link_libraries(${the_module} PRIVATE debug ${${PYTHON}_DEBUG_LIBRARIES} optimized ${${PYTHON}_LIBRARIES})
  else()
    ocv_target_link_libraries(${the_module} PRIVATE ${${PYTHON}_LIBRARIES})
  endif()
endif()

if(TARGET gen_opencv_python_source)
  set(deps ${OPENCV_MODULE_${the_module}_DEPS})
  list(REMOVE_ITEM deps opencv_python_bindings_generator) # don't add dummy module to target_link_libraries list
endif()
ocv_target_link_libraries(${the_module} PRIVATE ${deps})

if(DEFINED ${PYTHON}_CVPY_SUFFIX)
  set(CVPY_SUFFIX "${${PYTHON}_CVPY_SUFFIX}")
else()
  set(__python_ext_suffix_var "EXT_SUFFIX")
  if("${${PYTHON}_VERSION_MAJOR}" STREQUAL "2")
    set(__python_ext_suffix_var "SO")
  endif()
  execute_process(COMMAND ${${PYTHON}_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('${__python_ext_suffix_var}'))"
                  RESULT_VARIABLE PYTHON_CVPY_PROCESS
                  OUTPUT_VARIABLE CVPY_SUFFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT PYTHON_CVPY_PROCESS EQUAL 0)
    set(CVPY_SUFFIX ".so")
  endif()
  if(${PYTHON}_LIMITED_API)
    if(WIN32)
      string(REGEX REPLACE "\\.[^\\.]*\\." "." CVPY_SUFFIX "${CVPY_SUFFIX}")
    else()
      string(REGEX REPLACE "\\.[^\\.]*\\." ".abi${${PYTHON}_VERSION_MAJOR}." CVPY_SUFFIX "${CVPY_SUFFIX}")
    endif()
  endif()
endif()

ocv_update(OPENCV_PYTHON_EXTENSION_BUILD_PATH "${LIBRARY_OUTPUT_PATH}/${MODULE_INSTALL_SUBDIR}")

set_target_properties(${the_module} PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY  "${OPENCV_PYTHON_EXTENSION_BUILD_PATH}"
                      ARCHIVE_OUTPUT_NAME ${the_module}  # prevent name conflict for python2/3 outputs
                      PREFIX ""
                      OUTPUT_NAME cv2
                      SUFFIX "${CVPY_SUFFIX}")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_module} PROPERTIES FOLDER "bindings")
endif()

if(MSVC)
  add_definitions(-DCVAPI_EXPORTS)
endif()

if((CV_GCC OR CV_CLANG) AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif()

if(MSVC AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100") #unreferenced formal parameter
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4127") #conditional expression is constant
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4505") #unreferenced local function has been removed
  string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


if(MSVC)
  ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4996)
else()
  ocv_warnings_disable(CMAKE_CXX_FLAGS
      -Wdeprecated-declarations
      -Woverloaded-virtual -Wunused-private-field
      -Wundef # accurate guard via #pragma doesn't work (C++ preprocessor doesn't handle #pragma)
  )
endif()

if(MSVC AND NOT BUILD_SHARED_LIBS)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

if(MSVC AND NOT ${PYTHON}_DEBUG_LIBRARIES)
  set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else()
  set(PYTHON_INSTALL_CONFIGURATIONS "")
endif()

if(WIN32)
  set(PYTHON_INSTALL_ARCHIVE "")
else()
  set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${${PYTHON}_PACKAGES_PATH} COMPONENT python)
endif()

set(__python_loader_subdir "")
if(NOT OPENCV_SKIP_PYTHON_LOADER)
  set(__python_loader_subdir "cv2/")
endif()

if(NOT " ${PYTHON}" STREQUAL " PYTHON"
    AND NOT DEFINED OPENCV_PYTHON_INSTALL_PATH
)
  if(DEFINED OPENCV_${PYTHON}_INSTALL_PATH)
    set(OPENCV_PYTHON_INSTALL_PATH "${OPENCV_${PYTHON}_INSTALL_PATH}")
  elseif(NOT OPENCV_SKIP_PYTHON_LOADER)
    set(OPENCV_PYTHON_INSTALL_PATH "${${PYTHON}_PACKAGES_PATH}")
  endif()
endif()

if(NOT OPENCV_SKIP_PYTHON_LOADER AND DEFINED OPENCV_PYTHON_INSTALL_PATH)
  include("${CMAKE_CURRENT_LIST_DIR}/python_loader.cmake")
  set(OPENCV_PYTHON_INSTALL_PATH_SETUPVARS "${OPENCV_PYTHON_INSTALL_PATH}" CACHE INTERNAL "")
endif()

if(OPENCV_SKIP_PYTHON_LOADER)
  if(DEFINED OPENCV_${PYTHON}_INSTALL_PATH)
    set(__python_binary_install_path "${OPENCV_${PYTHON}_INSTALL_PATH}")
  elseif(DEFINED ${PYTHON}_PACKAGES_PATH)
    set(__python_binary_install_path "${${PYTHON}_PACKAGES_PATH}")
  else()
    message(FATAL_ERROR "Specify 'OPENCV_${PYTHON}_INSTALL_PATH' variable")
  endif()
else()
  ocv_assert(DEFINED OPENCV_PYTHON_INSTALL_PATH)
  if(${PYTHON}_LIMITED_API)
    set(__python_binary_subdir "python-${${PYTHON}_VERSION_MAJOR}")
  else()
    set(__python_binary_subdir "python-${${PYTHON}_VERSION_MAJOR}.${${PYTHON}_VERSION_MINOR}")
  endif()
  set(__python_binary_install_path "${OPENCV_PYTHON_INSTALL_PATH}/${__python_loader_subdir}${__python_binary_subdir}")
endif()

install(TARGETS ${the_module}
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION "${__python_binary_install_path}" COMPONENT python
        LIBRARY DESTINATION "${__python_binary_install_path}" COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )

set(__INSTALL_PATH_${PYTHON} "${__python_binary_install_path}" CACHE INTERNAL "")  # CMake status

if(NOT OPENCV_SKIP_PYTHON_LOADER)
  ocv_assert(DEFINED OPENCV_PYTHON_INSTALL_PATH)
  if(OpenCV_FOUND)
    set(__loader_path "${OpenCV_BINARY_DIR}/python_loader")
  else()
    set(__loader_path "${CMAKE_BINARY_DIR}/python_loader")
  endif()

  set(__python_loader_install_tmp_path "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/python_loader/")
  set(OpenCV_PYTHON_LOADER_FULL_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${OPENCV_PYTHON_INSTALL_PATH}/cv2")
  if(IS_ABSOLUTE "${OPENCV_PYTHON_INSTALL_PATH}")
    set(CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE "'${OPENCV_PYTHON_INSTALL_PATH}/cv2'")
  else()
    set(CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE "LOADER_DIR")
  endif()

  if(DEFINED ${PYTHON}_VERSION_MINOR AND NOT ${PYTHON}_LIMITED_API)
    set(__target_config "config-${${PYTHON}_VERSION_MAJOR}.${${PYTHON}_VERSION_MINOR}.py")
  else()
    set(__target_config "config-${${PYTHON}_VERSION_MAJOR}.py")
  endif()

  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_PYTHON_EXTENSION_PATH "'${OPENCV_PYTHON_EXTENSION_BUILD_PATH}/Release'")  # TODO: CMAKE_BUILD_TYPE is not defined
  else()
    set(CMAKE_PYTHON_EXTENSION_PATH "'${OPENCV_PYTHON_EXTENSION_BUILD_PATH}'")
  endif()
  configure_file("${PYTHON_SOURCE_DIR}/package/template/config-x.y.py.in" "${__loader_path}/cv2/${__target_config}" @ONLY)

  if(IS_ABSOLUTE __python_binary_install_path)
    set(CMAKE_PYTHON_EXTENSION_PATH "'${__python_binary_install_path}'")
  else()
    file(RELATIVE_PATH OpenCV_PYTHON_BINARY_RELATIVE_INSTALL_PATH "${OpenCV_PYTHON_LOADER_FULL_INSTALL_PATH}" "${CMAKE_INSTALL_PREFIX}/${__python_binary_install_path}")
    set(CMAKE_PYTHON_EXTENSION_PATH "os.path.join(${CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE}, '${OpenCV_PYTHON_BINARY_RELATIVE_INSTALL_PATH}')")
  endif()
  configure_file("${PYTHON_SOURCE_DIR}/package/template/config-x.y.py.in" "${__python_loader_install_tmp_path}/cv2/${__target_config}" @ONLY)
  install(FILES "${__python_loader_install_tmp_path}/cv2/${__target_config}" DESTINATION "${OPENCV_PYTHON_INSTALL_PATH}/cv2/" COMPONENT python)
endif()  # NOT OPENCV_SKIP_PYTHON_LOADER

unset(PYTHON_SRC_DIR)
unset(PYTHON_CVPY_PROCESS)
unset(CVPY_SUFFIX)
unset(PYTHON_INSTALL_CONFIGURATIONS)
unset(PYTHON_INSTALL_ARCHIVE)
