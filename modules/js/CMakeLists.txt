# ----------------------------------------------------------------------------
#  CMake file for js support
# ----------------------------------------------------------------------------
if(OPENCV_INITIAL_PASS)
  # generator for Objective-C source code and documentation signatures
  add_subdirectory(generator)
endif()

if(NOT BUILD_opencv_js)  # should be enabled explicitly (by build_js.py script)
  return()
endif()

set(the_description "The JavaScript(JS) bindings")

set(OPENCV_JS "opencv.js")
set(JS_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/src/helpers.js")

find_path(EMSCRIPTEN_INCLUDE_DIR
          emscripten/bind.h
          PATHS
            ENV EMSCRIPTEN_ROOT
          PATH_SUFFIXES system/include include
          DOC "Location of Emscripten SDK")

if(NOT EMSCRIPTEN_INCLUDE_DIR OR NOT PYTHON3INTERP_FOUND)
  set(DISABLE_MSG "Module 'js' disabled because the following dependencies are not found:")
  if(NOT EMSCRIPTEN_INCLUDE_DIR)
    set(DISABLE_MSG "${DISABLE_MSG} Emscripten")
  endif()
  if(NOT PYTHON3INTERP_FOUND)
    set(DISABLE_MSG "${DISABLE_MSG} Python")
  endif()
  message(STATUS ${DISABLE_MSG})
  ocv_module_disable(js)
endif()

ocv_add_module(js BINDINGS PRIVATE_REQUIRED opencv_js_bindings_generator)

ocv_module_include_directories(${EMSCRIPTEN_INCLUDE_DIR})

add_definitions("-std=c++11")

set(deps ${OPENCV_MODULE_${the_module}_DEPS})
list(REMOVE_ITEM deps opencv_js_bindings_generator)  # don't add dummy module
link_libraries(${deps})

set(bindings_cpp "${OPENCV_JS_BINDINGS_DIR}/gen/bindings.cpp")
set_source_files_properties(${bindings_cpp} PROPERTIES GENERATED TRUE)

OCV_OPTION(BUILD_WASM_INTRIN_TESTS "Build WASM intrin tests" OFF )
if(BUILD_WASM_INTRIN_TESTS)
  add_definitions(-DTEST_WASM_INTRIN)
  ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../ts/include")
  ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../imgcodecs/include")
  ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../videoio/include")
  ocv_module_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../highgui/include")
  ocv_add_executable(${the_module} ${bindings_cpp} "${CMAKE_CURRENT_SOURCE_DIR}/../ts/src/ts_gtest.cpp")
else()
  ocv_add_executable(${the_module} ${bindings_cpp})
endif()

add_dependencies(${the_module} gen_opencv_js_source)

set(COMPILE_FLAGS "")
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -Wno-missing-prototypes")
endif()
if(COMPILE_FLAGS)
    set_target_properties(${the_module} PROPERTIES COMPILE_FLAGS ${COMPILE_FLAGS})
endif()

set(EMSCRIPTEN_LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS} -s TOTAL_MEMORY=128MB -s WASM_MEM_MAX=1GB -s ALLOW_MEMORY_GROWTH=1")
set(EMSCRIPTEN_LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS} -s MODULARIZE=1")
set(EMSCRIPTEN_LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS} -s EXPORT_NAME=\"'cv'\" -s DEMANGLE_SUPPORT=1")
set(EMSCRIPTEN_LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS} -s FORCE_FILESYSTEM=1 --use-preload-plugins --bind --post-js ${JS_HELPER} ${COMPILE_FLAGS}")
set_target_properties(${the_module} PROPERTIES LINK_FLAGS "${EMSCRIPTEN_LINK_FLAGS}")

# add UMD wrapper
set(MODULE_JS_PATH "${OpenCV_BINARY_DIR}/bin/${the_module}.js")
set(OCV_JS_PATH "${OpenCV_BINARY_DIR}/bin/${OPENCV_JS}")

add_custom_command(
   OUTPUT ${OCV_JS_PATH}
   COMMAND ${PYTHON3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/make_umd.py" ${MODULE_JS_PATH} "${OCV_JS_PATH}"
   DEPENDS ${the_module}
   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/make_umd.py")

add_custom_target(${OPENCV_JS} ALL
                  DEPENDS ${OCV_JS_PATH}
                  DEPENDS ${the_module})

# test
set(opencv_test_js_bin_dir "${EXECUTABLE_OUTPUT_PATH}")
set(test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test)

set(opencv_test_js_file_deps "")

# message(STATUS "${opencv_test_js_bin_dir}")

# make sure the build directory exists
file(MAKE_DIRECTORY "${opencv_test_js_bin_dir}")

# gather and copy specific files for js test
file(GLOB_RECURSE test_files RELATIVE "${test_dir}" "${test_dir}/*")
foreach(f ${test_files})
  # message(STATUS "copy ${test_dir}/${f} ${opencv_test_js_bin_dir}/${f}")
  add_custom_command(OUTPUT "${opencv_test_js_bin_dir}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${opencv_test_js_bin_dir}/${f}"
                     DEPENDS "${test_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND opencv_test_js_file_deps "${test_dir}/${f}" "${opencv_test_js_bin_dir}/${f}")
endforeach()

add_custom_target(${PROJECT_NAME}_test
                  DEPENDS ${OCV_JS_PATH} ${opencv_test_js_file_deps})

# perf
set(opencv_perf_js_bin_dir "${EXECUTABLE_OUTPUT_PATH}/perf")
set(perf_dir ${CMAKE_CURRENT_SOURCE_DIR}/perf)

set(opencv_perf_js_file_deps "")

# make sure the build directory exists
file(MAKE_DIRECTORY "${opencv_perf_js_bin_dir}")

# gather and copy specific files for js perf
file(GLOB_RECURSE perf_files RELATIVE "${perf_dir}" "${perf_dir}/*")
foreach(f ${perf_files})
  add_custom_command(OUTPUT "${opencv_perf_js_bin_dir}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${perf_dir}/${f}" "${opencv_perf_js_bin_dir}/${f}"
                     DEPENDS "${perf_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND opencv_perf_js_file_deps "${perf_dir}/${f}" "${opencv_perf_js_bin_dir}/${f}")
endforeach()

add_custom_target(${PROJECT_NAME}_perf
                  DEPENDS ${OCV_JS_PATH} ${opencv_perf_js_file_deps})

#loader
set(opencv_loader_js_bin_dir "${EXECUTABLE_OUTPUT_PATH}")
set(loader_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(opencv_loader_js_file_deps "")

# make sure the build directory exists
file(MAKE_DIRECTORY "${opencv_loader_js_bin_dir}")

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${loader_dir}/loader.js
                ${opencv_loader_js_bin_dir}/loader.js)
list(APPEND opencv_loader_js_file_deps "${loader_dir}/loader.js" "${opencv_loader_js_bin_dir}/loader.js")

add_custom_target(${PROJECT_NAME}_loader ALL
                  DEPENDS ${OCV_JS_PATH} ${opencv_loader_js_file_deps})

add_custom_target(opencv_test_js ALL DEPENDS opencv_js_test opencv_js_perf opencv_js_loader)
