if(OPENCV_INITIAL_PASS)
  # generator for JNI/JAR source code and documentation signatures
  add_subdirectory(generator)
endif()

if(APPLE_FRAMEWORK OR WINRT
    OR NOT PYTHON3INTERP_FOUND
    OR NOT (ANT_EXECUTABLE OR Java_FOUND OR ANDROID_PROJECTS_BUILD_TYPE STREQUAL "GRADLE")
    OR NOT (JNI_FOUND OR (ANDROID AND (NOT DEFINED ANDROID_NATIVE_API_LEVEL OR ANDROID_NATIVE_API_LEVEL GREATER 7)))
    OR BUILD_opencv_world
    )
  ocv_module_disable(java)
endif()

set(the_description "The java bindings")
ocv_add_module(java BINDINGS opencv_core opencv_imgproc PRIVATE_REQUIRED opencv_java_bindings_generator)

include(${CMAKE_CURRENT_SOURCE_DIR}/common.cmake)

# UTILITY: glob specific sources and append them to list (type is in H, CPP, JAVA)
macro(glob_more_specific_sources _type _root _output)
  unset(_masks)
  if(${_type} STREQUAL "H")
    set(_masks "${_root}/cpp/*.h" "${_root}/cpp/*.hpp")
  elseif(${_type} STREQUAL "CPP")
    set(_masks "${_root}/cpp/*.cpp")
  elseif(${_type} STREQUAL "JAVA")
    set(_masks "${_root}/java/*.java" "${_root}/java/*.java.in")
  endif()
  if (_masks)
    file(GLOB _result ${_masks})
    list(APPEND ${_output} ${_result})
  else()
    message(WARNING "Bad argument passed to macro: skipped")
  endif()
endmacro()

# UTILITY: copy common java test files and add them to _deps
# copy_common_tests(<source-folder> <destination-folder> <variable-to-store-deps>)
macro(copy_common_tests _src_location _dst_location _deps)
  set(_src ${_src_location})
  set(_dst ${_dst_location})
  file(GLOB_RECURSE _files RELATIVE "${_src}" "${_src}/res/*" "${_src}/src/*")
  foreach(f ${_files})
    add_custom_command(
        OUTPUT "${_dst}/${f}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}/${f}" "${_dst}/${f}"
        MAIN_DEPENDENCY "${_src}/${f}"
        COMMENT "Copying ${f}")
    list(APPEND ${_deps} "${_src}/${f}" "${_dst}/${f}")
  endforeach()
  unset(_files)
  unset(_src)
  unset(_dst)
endmacro()


add_subdirectory(jni)  # generates ${the_module} target (${the_module}_jni doesn't work properly with Android non-gradle samples)
if(ANDROID)
  add_subdirectory(android_sdk)  # generates ${the_module}_android target
else()
  add_subdirectory(jar)  # generates ${the_module}_jar target
endif()

if(BUILD_TESTS)
  if(ANDROID)
    add_subdirectory(test/android_test)
  else()
    add_subdirectory(test/pure_test)
  endif()
endif()
