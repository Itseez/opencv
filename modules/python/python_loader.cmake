ocv_assert(NOT OPENCV_SKIP_PYTHON_LOADER)

set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(OpenCV_FOUND)
  set(__loader_path "${OpenCV_BINARY_DIR}/python_loader")
  message(STATUS "OpenCV Python: during development append to PYTHONPATH: ${__loader_path}")
else()
  set(__loader_path "${CMAKE_BINARY_DIR}/python_loader")
endif()

set(__python_loader_install_tmp_path "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/python_loader/")
if(DEFINED OPENCV_PYTHON_INSTALL_PATH)
  if(IS_ABSOLUTE "${OPENCV_PYTHON_INSTALL_PATH}")
    set(OpenCV_PYTHON_INSTALL_PATH_RELATIVE_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}/")
    set(CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE "'${CMAKE_INSTALL_PREFIX}'")
  else()
    file(RELATIVE_PATH OpenCV_PYTHON_INSTALL_PATH_RELATIVE_CONFIGCMAKE "${CMAKE_INSTALL_PREFIX}/${OPENCV_PYTHON_INSTALL_PATH}/cv2" ${CMAKE_INSTALL_PREFIX})
    set(CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE "os.path.join(LOADER_DIR, '${OpenCV_PYTHON_INSTALL_PATH_RELATIVE_CONFIGCMAKE}')")
  endif()
else()
  set(CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE "os.path.join(LOADER_DIR, 'not_installed')")
endif()

if(OpenCV_FOUND)
  return()  # Ignore "standalone" builds of Python bindings
endif()



set(PYTHON_LOADER_FILES
    "setup.py" "cv2/__init__.py"
    "cv2/load_config_py2.py" "cv2/load_config_py3.py"
)
foreach(fname ${PYTHON_LOADER_FILES})
  get_filename_component(__dir "${fname}" DIRECTORY)
  # avoid using of file(COPY) to rerun CMake on changes
  configure_file("${PYTHON_SOURCE_DIR}/package/${fname}" "${__loader_path}/${fname}" COPYONLY)
  if(fname STREQUAL "setup.py")
    if(OPENCV_PYTHON_SETUP_PY_INSTALL_PATH)
      install(FILES "${PYTHON_SOURCE_DIR}/package/${fname}" DESTINATION "${OPENCV_PYTHON_SETUP_PY_INSTALL_PATH}" COMPONENT python)
    endif()
  elseif(DEFINED OPENCV_PYTHON_INSTALL_PATH)
    install(FILES "${PYTHON_SOURCE_DIR}/package/${fname}" DESTINATION "${OPENCV_PYTHON_INSTALL_PATH}/${__dir}" COMPONENT python)
  endif()
endforeach()

if(WIN32)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    list(APPEND CMAKE_PYTHON_BINARIES_PATH "'${EXECUTABLE_OUTPUT_PATH}/Release'")  # TODO: CMAKE_BUILD_TYPE is not defined
  else()
    list(APPEND CMAKE_PYTHON_BINARIES_PATH "'${EXECUTABLE_OUTPUT_PATH}'")
  endif()
else()
  list(APPEND CMAKE_PYTHON_BINARIES_PATH "'${LIBRARY_OUTPUT_PATH}'")
endif()
string(REPLACE ";" ",\n    " CMAKE_PYTHON_BINARIES_PATH "${CMAKE_PYTHON_BINARIES_PATH}")
configure_file("${PYTHON_SOURCE_DIR}/package/template/config.py.in" "${__loader_path}/cv2/config.py" @ONLY)



# install
if(DEFINED OPENCV_PYTHON_INSTALL_PATH)
  if(WIN32)
    list(APPEND CMAKE_PYTHON_BINARIES_INSTALL_PATH "os.path.join(${CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE}, '${OPENCV_BIN_INSTALL_PATH}')")
  else()
    list(APPEND CMAKE_PYTHON_BINARIES_INSTALL_PATH "os.path.join(${CMAKE_PYTHON_EXTENSION_INSTALL_PATH_BASE}, '${OPENCV_LIB_INSTALL_PATH}')")
  endif()
  set(CMAKE_PYTHON_BINARIES_PATH "${CMAKE_PYTHON_BINARIES_INSTALL_PATH}")
  if (WIN32 AND HAVE_CUDA)
    if (ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
      if (DEFINED CUDAToolkit_LIBRARY_ROOT)
        list(APPEND CMAKE_PYTHON_BINARIES_PATH "os.path.join(os.getenv('CUDA_PATH', '${CUDAToolkit_LIBRARY_ROOT}'), 'bin')")
      endif()
    else()
      if (DEFINED CUDA_TOOLKIT_ROOT_DIR)
        list(APPEND CMAKE_PYTHON_BINARIES_PATH "os.path.join(os.getenv('CUDA_PATH', '${CUDA_TOOLKIT_ROOT_DIR}'), 'bin')")
      endif()
    endif()
  endif()
  string(REPLACE ";" ",\n    " CMAKE_PYTHON_BINARIES_PATH "${CMAKE_PYTHON_BINARIES_PATH}")
  configure_file("${PYTHON_SOURCE_DIR}/package/template/config.py.in" "${__python_loader_install_tmp_path}/cv2/config.py" @ONLY)
  install(FILES "${__python_loader_install_tmp_path}/cv2/config.py" DESTINATION "${OPENCV_PYTHON_INSTALL_PATH}/cv2/" COMPONENT python)
endif()



#
# Handle Python extra code (submodules)
#
function(ocv_add_python_files_from_path search_path)
  file(GLOB_RECURSE extra_py_files
       RELATIVE "${search_path}"
       # Plain Python code
       "${search_path}/*.py"
       # Type annotations
       "${search_path}/*.pyi"
  )
  ocv_debug_message("Extra Py files for ${search_path}: ${extra_py_files}")
  if(extra_py_files)
    list(SORT extra_py_files)
    foreach(filename ${extra_py_files})
      get_filename_component(module "${filename}" DIRECTORY)
      if(NOT ${module} IN_LIST extra_modules)
        list(APPEND extra_modules ${module})
      endif()
      configure_file("${search_path}/${filename}" "${__loader_path}/cv2/${filename}" COPYONLY)
      if(DEFINED OPENCV_PYTHON_INSTALL_PATH)
        install(FILES "${search_path}/${filename}" DESTINATION "${OPENCV_PYTHON_INSTALL_PATH}/cv2/${module}/" COMPONENT python)
      endif()
    endforeach()
    message(STATUS "Found '${extra_modules}' Python modules from ${search_path}")
  else()
    message(WARNING "Can't add Python files and modules from '${module_path}'. There is no .py or .pyi files")
  endif()
endfunction()

ocv_add_python_files_from_path("${PYTHON_SOURCE_DIR}/package/extra_modules")

foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";python;" AND HAVE_${m}
      AND EXISTS "${OPENCV_MODULE_${m}_LOCATION}/misc/python/package"
  )
    ocv_add_python_files_from_path("${OPENCV_MODULE_${m}_LOCATION}/misc/python/package")
  endif()
endforeach(m)

if(NOT "${OPENCV_PYTHON_EXTRA_MODULES_PATH}" STREQUAL "")
  foreach(extra_ocv_py_modules_path ${OPENCV_PYTHON_EXTRA_MODULES_PATH})
    ocv_add_python_files_from_path(${extra_ocv_py_modules_path})
  endforeach()
endif()

if(${PYTHON}_VERSION_STRING VERSION_GREATER "3.6" AND PYTHON_DEFAULT_VERSION VERSION_GREATER "3.6")
  add_custom_target(copy_opencv_typing_stubs)
  # Copy all generated stub files to python_loader directory only if
  # generation succeeds, this behvoir can't be achieved with default
  # CMake constructions, because failed generation produces a warning instead of
  # halts on hard error.
  add_custom_command(
    TARGET copy_opencv_typing_stubs
    POST_BUILD
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} ${PYTHON_SOURCE_DIR}/src2/copy_typings_stubs_on_success.py
            --stubs_dir ${OPENCV_PYTHON_BINDINGS_DIR}/cv2
            --output_dir ${__loader_path}/cv2
  )
  if(DEFINED OPENCV_PYTHON_INSTALL_PATH)
    install(DIRECTORY "${OPENCV_PYTHON_BINDINGS_DIR}/cv2" DESTINATION "${OPENCV_PYTHON_INSTALL_PATH}" COMPONENT python)
  endif()
endif()
