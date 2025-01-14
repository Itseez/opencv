set(the_description "High-level GUI")

if(ANDROID)
  ocv_add_module(highgui opencv_imgproc OPTIONAL opencv_imgcodecs opencv_videoio WRAP python)
else()
  ocv_add_module(highgui opencv_imgproc OPTIONAL opencv_imgcodecs opencv_videoio WRAP python java)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/plugin.cmake)

set(tgts "PRIVATE")

set(highgui_hdrs
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    )

set(highgui_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/backend.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/window.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/roiSelector.cpp
    )

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

if(DEFINED WINRT AND NOT DEFINED ENABLE_WINRT_MODE_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /ZW")
endif()

if(APPLE)
  ocv_include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND HIGHGUI_LIBRARIES ${ZLIB_LIBRARIES})
endif()

if(HAVE_WEBP)
  add_definitions(-DHAVE_WEBP)
endif()

file(GLOB highgui_ext_hdrs
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
     "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h")

# Removing WinRT API headers by default
list(REMOVE_ITEM highgui_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/highgui_winrt.hpp")

set(OPENCV_HIGHGUI_BUILTIN_BACKEND "")

if(WITH_FRAMEBUFFER AND HAVE_FRAMEBUFFER)
  set(OPENCV_HIGHGUI_BUILTIN_BACKEND "FB")
  add_definitions(-DHAVE_FRAMEBUFFER)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_framebuffer.cpp)
  list(APPEND highgui_hdrs ${CMAKE_CURRENT_LIST_DIR}/src/window_framebuffer.hpp)
  if(HAVE_FRAMEBUFFER_XVFB)
    add_definitions(-DHAVE_FRAMEBUFFER_XVFB)
  endif()
endif()

if(WITH_WAYLAND AND HAVE_WAYLAND)
  set(OPENCV_HIGHGUI_BUILTIN_BACKEND "Wayland")
  add_definitions(-DHAVE_WAYLAND)

  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  if (HAVE_WAYLAND_PROTOCOLS)
      ocv_wayland_generate(
            ${WAYLAND_PROTOCOLS_BASE}/stable/xdg-shell/xdg-shell.xml
            xdg-shell-client-protocol)
  endif()

  list(APPEND highgui_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/window_wayland.cpp
    ${WAYLAND_PROTOCOL_SOURCES}
  )
  list(APPEND HIGHGUI_LIBRARIES "${WAYLAND_CLIENT_LINK_LIBRARIES};${WAYLAND_CURSOR_LINK_LIBRARIES};${XKBCOMMON_LINK_LIBRARIES}")

  if(HAVE_WAYLAND_EGL)
    if(WAYLAND_EGL_LIBRARIES)
      list(APPEND HIGHGUI_LIBRARIES "${WAYLAND_EGL_LIBRARIES}")
    endif()
  endif()

  ocv_module_include_directories(${WAYLAND_CLIENT_INCLUDE_DIRS} ${XKBCOMMON_INCLUDE_DIRS})
elseif(HAVE_QT)
  set(OPENCV_HIGHGUI_BUILTIN_BACKEND "QT${QT_VERSION_MAJOR}")
  add_definitions(-DHAVE_QT)

  if(QT_VERSION_MAJOR GREATER 4)
    # "Automoc" doesn't work properly with opencv_world build, use QT<ver>_WRAP_CPP() directly
    #set(CMAKE_AUTOMOC ON)

    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    if(QT_VERSION_MAJOR EQUAL 6)
      add_definitions(-DHAVE_QT6) # QGLWidget deprecated for QT6, use this preprocessor to adjust window_QT.[h,cpp]
      QT6_ADD_RESOURCES(_RCC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.qrc)
      QT6_WRAP_CPP(_MOC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h)
    elseif(QT_VERSION_MAJOR EQUAL 5)
      QT5_ADD_RESOURCES(_RCC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.qrc)
      QT5_WRAP_CPP(_MOC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h)
    else()
      message(FATAL_ERROR "Unsupported QT version: ${QT_VERSION_MAJOR}")
    endif()

    list(APPEND highgui_srcs
         ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h
         ${_MOC_OUTFILES}
         ${_RCC_OUTFILES})

    set(qt_deps Core Gui Widgets Test Concurrent)
    if(HAVE_QT_OPENGL)
      add_definitions(-DHAVE_QT_OPENGL)
      # QOpenGLWidget requires Qt6 package component OpenGLWidgets
      if(QT_VERSION_MAJOR GREATER 5)
        list(APPEND qt_deps OpenGLWidgets)
      endif()
      list(APPEND qt_deps OpenGL)
      if(OPENGL_LIBRARIES)
        list(APPEND HIGHGUI_LIBRARIES "${OPENGL_LIBRARIES}")
      endif()
    endif()

    foreach(dt_dep ${qt_deps})
      add_definitions(${Qt${QT_VERSION_MAJOR}${dt_dep}_DEFINITIONS})
      include_directories(${Qt${QT_VERSION_MAJOR}${dt_dep}_INCLUDE_DIRS})
      list(APPEND HIGHGUI_LIBRARIES ${Qt${QT_VERSION_MAJOR}${dt_dep}_LIBRARIES})
    endforeach()
  else()
    ocv_assert(QT_VERSION_MAJOR EQUAL 4)
    if(HAVE_QT_OPENGL)
      set(QT_USE_QTOPENGL TRUE)
      if(OPENGL_LIBRARIES)
        list(APPEND HIGHGUI_LIBRARIES "${OPENGL_LIBRARIES}")
      endif()
    endif()
    include(${QT_USE_FILE})

    QT4_ADD_RESOURCES(_RCC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.qrc)
    QT4_WRAP_CPP(_MOC_OUTFILES ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.h)

    list(APPEND HIGHGUI_LIBRARIES ${QT_LIBRARIES})
    list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_QT.cpp ${_MOC_OUTFILES} ${_RCC_OUTFILES})
    ocv_check_flag_support(CXX -Wno-missing-declarations _have_flag "")
    if(${_have_flag})
      set_source_files_properties(${_RCC_OUTFILES} PROPERTIES COMPILE_FLAGS -Wno-missing-declarations)
    endif()
  endif()
elseif(WINRT)
  set(OPENCV_HIGHGUI_BUILTIN_BACKEND "WINRT")
  if(NOT WINRT_8_0)
    # Dependencies used by the implementation referenced
    # below are not available on WinRT 8.0.
    # Enabling it for WiRT 8.1+ only.

    # WinRT 8.1+ detected. Adding WinRT API header.
    message(STATUS "  ${name}: WinRT detected. Adding WinRT API header")
    list(APPEND highgui_ext_hdrs "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/highgui_winrt.hpp")


    list(APPEND highgui_srcs
      ${CMAKE_CURRENT_LIST_DIR}/src/window_winrt.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/window_winrt_bridge.cpp)
    list(APPEND highgui_hdrs
      ${CMAKE_CURRENT_LIST_DIR}/src/window_winrt_bridge.hpp)
  endif()

  # libraries below are neither available nor required
  # on ARM devices and/or Windows Phone
  if(WINRT_PHONE OR (OpenCV_ARCH STREQUAL "ARM"))
    list(REMOVE_ITEM HIGHGUI_LIBRARIES "comctl32" "gdi32" "ole32" "setupapi")
    if(WINRT_PHONE)
      message(STATUS "  ${name}: Windows Phone detected")
    elseif(OpenCV_ARCH STREQUAL "ARM")
      message(STATUS "  ${name}: ARM detected")
      if(WINRT_STORE)
        list(REMOVE_ITEM HIGHGUI_LIBRARIES "ws2_32")
        message(STATUS "  ${name}:   Removing 'ws2_32.lib'")
      endif()
    endif()
    message(STATUS "  ${name}:   Removing 'comctl32.lib, gdi32.lib, ole32.lib, setupapi.lib'")
    message(STATUS "  ${name}:   Leaving '${HIGHGUI_LIBRARIES}'")
  endif()
elseif(HAVE_COCOA)
  set(OPENCV_HIGHGUI_BUILTIN_BACKEND "COCOA")
  add_definitions(-DHAVE_COCOA)
  list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_cocoa.mm)
  list(APPEND HIGHGUI_LIBRARIES "-framework Cocoa")
endif()

if(TARGET ocv.3rdparty.win32ui)
  if("win32ui" IN_LIST HIGHGUI_PLUGIN_LIST OR HIGHGUI_PLUGIN_LIST STREQUAL "all")
    ocv_create_builtin_highgui_plugin(opencv_highgui_win32 ocv.3rdparty.win32ui "window_w32.cpp")
  elseif(NOT OPENCV_HIGHGUI_BUILTIN_BACKEND)
    set(OPENCV_HIGHGUI_BUILTIN_BACKEND "WIN32UI")
    list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_w32.cpp)
    list(APPEND tgts ocv.3rdparty.win32ui)
    if(HAVE_OPENGL AND OPENGL_LIBRARIES)
      list(APPEND tgts "${OPENGL_LIBRARIES}")
    endif()
  endif()
endif()

if(TARGET ocv.3rdparty.gtk3 OR TARGET ocv.3rdparty.gtk2)
  if(TARGET ocv.3rdparty.gtk3 AND NOT WITH_GTK_2_X)
    set(__gtk_dependency "ocv.3rdparty.gtk3")
  else()
    set(__gtk_dependency "ocv.3rdparty.gtk2")
  endif()
  if(
    NOT HIGHGUI_PLUGIN_LIST STREQUAL "all"
    AND NOT "gtk" IN_LIST HIGHGUI_PLUGIN_LIST
    AND NOT "gtk2" IN_LIST HIGHGUI_PLUGIN_LIST
    AND NOT "gtk3" IN_LIST HIGHGUI_PLUGIN_LIST
    AND NOT OPENCV_HIGHGUI_BUILTIN_BACKEND
  )
    if(__gtk_dependency STREQUAL "ocv.3rdparty.gtk3")
      set(OPENCV_HIGHGUI_BUILTIN_BACKEND "GTK3")
      if(OPENGL_LIBRARIES)
        list(APPEND HIGHGUI_LIBRARIES "${OPENGL_LIBRARIES}")
      endif()
    elseif(__gtk_dependency STREQUAL "ocv.3rdparty.gtk2")
      set(OPENCV_HIGHGUI_BUILTIN_BACKEND "GTK2")
    else()
      set(OPENCV_HIGHGUI_BUILTIN_BACKEND "GTK")
    endif()
    list(APPEND highgui_srcs ${CMAKE_CURRENT_LIST_DIR}/src/window_gtk.cpp)
    list(APPEND tgts ${__gtk_dependency})
    if(TARGET ocv.3rdparty.gtkglext
        AND __gtk_dependency STREQUAL "ocv.3rdparty.gtk2"
        AND NOT OPENCV_GTK_DISABLE_GTKGLEXT
    )
      list(APPEND tgts ocv.3rdparty.gtkglext)
      if(TARGET ocv.3rdparty.gtk_opengl
          AND __gtk_dependency STREQUAL "ocv.3rdparty.gtk2"
          AND NOT OPENCV_GTK_DISABLE_OPENGL
      )
        list(APPEND tgts ocv.3rdparty.gtk_opengl)
      endif()
    endif()
  elseif("gtk" IN_LIST HIGHGUI_PLUGIN_LIST)
    ocv_create_builtin_highgui_plugin(opencv_highgui_gtk ${__gtk_dependency} "window_gtk.cpp")
    if(TARGET ocv.3rdparty.gtkglext)
      ocv_target_link_libraries(opencv_highgui_gtk ocv.3rdparty.gtkglext)
    endif()
  else()
    if(TARGET ocv.3rdparty.gtk3 AND ("gtk3" IN_LIST HIGHGUI_PLUGIN_LIST OR HIGHGUI_PLUGIN_LIST STREQUAL "all"))
      ocv_create_builtin_highgui_plugin(opencv_highgui_gtk3 ocv.3rdparty.gtk3 "window_gtk.cpp")
      if(TARGET ocv.3rdparty.gtkglext)
        ocv_target_link_libraries(opencv_highgui_gtk3 ocv.3rdparty.gtkglext)
      endif()
    endif()
    if(TARGET ocv.3rdparty.gtk2 AND ("gtk2" IN_LIST HIGHGUI_PLUGIN_LIST OR HIGHGUI_PLUGIN_LIST STREQUAL "all"))
      ocv_create_builtin_highgui_plugin(opencv_highgui_gtk2 ocv.3rdparty.gtk2 "window_gtk.cpp")
      if(TARGET ocv.3rdparty.gtkglext)
        ocv_target_link_libraries(opencv_highgui_gtk2 ocv.3rdparty.gtkglext)
      endif()
    endif()
  endif()
endif()

if(NOT OPENCV_HIGHGUI_BUILTIN_BACKEND)
  set(OPENCV_HIGHGUI_BUILTIN_BACKEND "NONE")
endif()
message(STATUS "highgui: using builtin backend: ${OPENCV_HIGHGUI_BUILTIN_BACKEND}")
set(OPENCV_HIGHGUI_BUILTIN_BACKEND "${OPENCV_HIGHGUI_BUILTIN_BACKEND}" PARENT_SCOPE)  # informational

if(TRUE)
  # these variables are set by 'ocv_append_build_options(HIGHGUI ...)'
  foreach(P ${HIGHGUI_INCLUDE_DIRS})
    ocv_include_directories(${P})
  endforeach()

  foreach(P ${HIGHGUI_LIBRARY_DIRS})
    link_directories(${P})
  endforeach()
endif()

if(tgts STREQUAL "PRIVATE")
  set(tgts "")
endif()

ocv_install_used_external_targets(${tgts})

source_group("Src" FILES ${highgui_srcs} ${highgui_hdrs})
source_group("Include" FILES ${highgui_ext_hdrs})
ocv_set_module_sources(HEADERS ${highgui_ext_hdrs} SOURCES ${highgui_srcs} ${highgui_hdrs})
ocv_module_include_directories()

ocv_create_module(${HIGHGUI_LIBRARIES})

macro(ocv_highgui_configure_target)
if(APPLE)
  add_apple_compiler_options(${the_module})
endif()

if(MSVC AND NOT BUILD_SHARED_LIBS AND BUILD_WITH_STATIC_CRT)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)
endmacro()

if(NOT BUILD_opencv_world)
  ocv_highgui_configure_target()
endif()

ocv_add_accuracy_tests(${tgts})
#ocv_add_perf_tests(${tgts})

if(HIGHGUI_ENABLE_PLUGINS)
  ocv_target_compile_definitions(${the_module} PRIVATE ENABLE_PLUGINS)
  if(TARGET opencv_test_highgui)
    ocv_target_compile_definitions(opencv_test_highgui PRIVATE ENABLE_PLUGINS)
  endif()
endif()

ocv_target_link_libraries(${the_module} LINK_PRIVATE ${tgts})

# generate module configuration
set(CONFIG_STR "// Auto-generated file
#define OPENCV_HIGHGUI_BUILTIN_BACKEND_STR \"${OPENCV_HIGHGUI_BUILTIN_BACKEND}\"
")
if(OPENCV_HIGHGUI_BUILTIN_BACKEND STREQUAL "NONE")
set(CONFIG_STR "${CONFIG_STR}
#define OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND 1
")
endif()

ocv_update_file("${CMAKE_CURRENT_BINARY_DIR}/opencv_highgui_config.hpp" "${CONFIG_STR}")
