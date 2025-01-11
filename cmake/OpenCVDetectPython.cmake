# Find specified Python version
# Arguments:
#   min_version (value): Minimum supported version
#   found (variable): Set if interpreter found
#   executable (variable): Output of executable found
#   version_string (variable): Output of found version
#   version_major (variable): Output of found major version
#   version_minor (variable): Output of found minor version
#   libs_found (variable): Set if libs found
#   libs_version_string (variable): Output of found libs version
#   libraries (variable): Output of found Python libraries
#   include_dirs (variable): Output of found Python include dir
#   packages_path (variable): Output of found Python packages path
#   numpy_include_dirs (variable): Output of found Python Numpy include dirs
#   numpy_version (variable): Output of found Python Numpy version
function(find_python min_version
    found executable version_string version_major version_minor
    libs_found libs_version_string libraries include_dirs packages_path
    numpy_include_dirs numpy_version)
  # Detect Python in PATH first. Available in CMake 3.16+
  if(NOT DEFINED Python3_EXECUTABLE)
    find_program(Python3_PATH "python3")
    if(NOT Python3_PATH)
      find_program(Python3_PATH "python")
    endif()
    if(Python3_PATH)
      set(Python3_EXECUTABLE "${Python3_PATH}")
    endif()
  endif()

  # This line can be remove after raising cmake_minimum_required to 3.15
  set(Python3_FIND_STRATEGY "LOCATION")

  if(min_version)
    find_host_package(Python3 "${min_version}" COMPONENTS Interpreter)
  else()
    find_host_package(Python3 COMPONENTS Interpreter)
  endif()

  if(Python3_Interpreter_FOUND)
    set(_found ${Python3_Interpreter_FOUND})
    set(_executable ${Python3_EXECUTABLE})
    set(_version_string ${Python3_VERSION})
    set(_version_major ${Python3_VERSION_MAJOR})
    set(_version_minor ${Python3_VERSION_MINOR})
    set(_version_patch ${Python3_VERSION_PATCH})
  endif()

  if(_found)
    set(_version_major_minor "${_version_major}.${_version_minor}")

    if(NOT ANDROID AND NOT APPLE_FRAMEWORK)
      if(APPLE AND NOT CMAKE_CROSSCOMPILING)
        if(NOT PYTHON_LIBRARY AND NOT PYTHON_INCLUDE_DIR)
          execute_process(COMMAND ${_executable} -c "from sysconfig import *; print(get_config_var('INCLUDEPY'))"
              RESULT_VARIABLE _cvpy_process
              OUTPUT_VARIABLE __include_dirs
              OUTPUT_STRIP_TRAILING_WHITESPACE)
          execute_process(COMMAND ${_executable} -c "from sysconfig import *; print('%s/%s' % (get_config_var('LIBDIR'), get_config_var('LIBRARY').replace('.a', '.dylib' if get_platform().startswith('macos') else '.so')))"
              RESULT_VARIABLE _cvpy_process
              OUTPUT_VARIABLE __libraries
              OUTPUT_STRIP_TRAILING_WHITESPACE)
          if(__include_dirs AND __libraries AND EXISTS "${__include_dirs}/Python.h" AND EXISTS "${__libraries}")
            set(${_include_dirs} "${__include_dirs}")
            set(${_libraries} "${__libraries}")
          endif()
        endif()
      endif()

      # not using _version_string here, because it might not conform to the CMake version format
      if(CMAKE_CROSSCOMPILING)
        # builder version can differ from target, matching base version (e.g. 2.7)
        find_package(Python3 COMPONENTS Development "${_version_major_minor}")
      else()
        find_package(Python3 "${_version_major_minor}.${_version_patch}" EXACT COMPONENTS Development)
      endif()

      if(Python3_Development_FOUND)
        # Copy outputs
        set(_libs_found ${Python3_Development_FOUND})
        set(_libraries ${Python3_LIBRARIES})
        set(_include_dirs ${Python3_INCLUDE_DIRS})
        set(_libs_version_string ${Python3_VERSION})
      endif()
    endif()

    if(NOT ANDROID AND NOT IOS AND NOT XROS)
      if(CMAKE_HOST_UNIX)
        execute_process(COMMAND ${_executable} -c "from sysconfig import *; print(get_path('purelib'))"
            RESULT_VARIABLE _cvpy_process
            OUTPUT_VARIABLE _std_packages_path
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if("${_std_packages_path}" MATCHES "site-packages")
          set(_packages_path "python${_version_major_minor}/site-packages")
        else() #debian based assumed, install to the dist-packages.
          set(_packages_path "python${_version_major_minor}/dist-packages")
        endif()
        set(_packages_path "lib/${_packages_path}")
      elseif(CMAKE_HOST_WIN32)
        get_filename_component(_path "${_executable}" PATH)
        file(TO_CMAKE_PATH "${_path}" _path)
        if(NOT EXISTS "${_path}/Lib/site-packages")
          unset(_path)
          get_filename_component(_path "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_version_major_minor}\\InstallPath]" ABSOLUTE)
          if(NOT _path)
            get_filename_component(_path "[HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_version_major_minor}\\InstallPath]" ABSOLUTE)
          endif()
          file(TO_CMAKE_PATH "${_path}" _path)
        endif()
        set(_packages_path "${_path}/Lib/site-packages")
        unset(_path)
      endif()

      set(_numpy_include_dirs "${${numpy_include_dirs}}")

      if(NOT _numpy_include_dirs)
        if(CMAKE_CROSSCOMPILING)
          message(STATUS "Cannot probe for Python/Numpy support (because we are cross-compiling OpenCV)")
          message(STATUS "If you want to enable Python/Numpy support, set the following variables:")
          message(STATUS "  PYTHON3_INCLUDE_DIRS")
          message(STATUS "  PYTHON3_LIBRARIES (optional on Unix-like systems)")
          message(STATUS "  PYTHON3_NUMPY_INCLUDE_DIRS")
        else()
          # Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
          execute_process(COMMAND "${_executable}" -c "import numpy; print(numpy.get_include())"
              RESULT_VARIABLE _numpy_process
              OUTPUT_VARIABLE _numpy_include_dirs
              OUTPUT_STRIP_TRAILING_WHITESPACE)

          if(NOT _numpy_process EQUAL 0)
            unset(_numpy_include_dirs)
          endif()
        endif()
      endif()

      if(_numpy_include_dirs)
        file(TO_CMAKE_PATH "${_numpy_include_dirs}" _numpy_include_dirs)
        if(CMAKE_CROSSCOMPILING)
          if(NOT _numpy_version)
            set(_numpy_version "undefined - cannot be probed because of the cross-compilation")
          endif()
        else()
          execute_process(COMMAND "${_executable}" -c "import numpy; print(numpy.version.version)"
              RESULT_VARIABLE _numpy_process
              OUTPUT_VARIABLE _numpy_version
              OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
      endif()
    endif(NOT ANDROID AND NOT IOS AND NOT XROS)
  endif()

  # Export return values
  set(${found} "${_found}" PARENT_SCOPE)
  set(${executable} "${_executable}" PARENT_SCOPE)
  set(${version_string} "${_version_string}" PARENT_SCOPE)
  set(${version_major} "${_version_major}" PARENT_SCOPE)
  set(${version_minor} "${_version_minor}" PARENT_SCOPE)
  set(${libs_found} "${_libs_found}" PARENT_SCOPE)
  set(${libs_version_string} "${_libs_version_string}" PARENT_SCOPE)
  set(${libraries} "${_libraries}" PARENT_SCOPE)
  set(${include_dirs} "${_include_dirs}" PARENT_SCOPE)
  set(${packages_path} "${_packages_path}" PARENT_SCOPE)
  set(${numpy_include_dirs} ${_numpy_include_dirs} PARENT_SCOPE)
  set(${numpy_version} "${_numpy_version}" PARENT_SCOPE)
endfunction(find_python)

if(OPENCV_PYTHON_SKIP_DETECTION)
  return()
endif()

find_python("${MIN_VER_PYTHON3}"
    PYTHON3INTERP_FOUND PYTHON3_EXECUTABLE PYTHON3_VERSION_STRING
    PYTHON3_VERSION_MAJOR PYTHON3_VERSION_MINOR PYTHON3LIBS_FOUND
    PYTHON3LIBS_VERSION_STRING PYTHON3_LIBRARIES
    PYTHON3_INCLUDE_DIRS PYTHON3_PACKAGES_PATH
    PYTHON3_NUMPY_INCLUDE_DIRS PYTHON3_NUMPY_VERSION)

# Problem in numpy >=1.15 <1.17
OCV_OPTION(PYTHON3_LIMITED_API "Build with Python Limited API (not available with numpy >=1.15 <1.17)" NO
    VISIBLE_IF PYTHON3_NUMPY_VERSION VERSION_LESS "1.15" OR NOT PYTHON3_NUMPY_VERSION VERSION_LESS "1.17")
if(PYTHON3_LIMITED_API)
  set(_default_ver "0x03060000")
  if(PYTHON3_VERSION_STRING VERSION_LESS "3.6")
    # fix for older pythons
    set(_default_ver "0x030${PYTHON3_VERSION_MINOR}0000")
  endif()
  set(PYTHON3_LIMITED_API_VERSION ${_default_ver})
endif()
