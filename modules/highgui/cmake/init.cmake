if(PROJECT_NAME STREQUAL "OpenCV")
  set(ENABLE_PLUGINS_DEFAULT ON)
  if(EMSCRIPTEN OR IOS OR WINRT)
    set(ENABLE_PLUGINS_DEFAULT OFF)
  endif()
  set(HIGHGUI_PLUGIN_LIST "" CACHE STRING "List of GUI backends to be compiled as plugins (gtk, gtk2/gtk3/gtk4, qt, win32 or special value 'all')")
  set(HIGHGUI_ENABLE_PLUGINS "${ENABLE_PLUGINS_DEFAULT}" CACHE BOOL "Allow building and using of GUI plugins")
  mark_as_advanced(HIGHGUI_PLUGIN_LIST HIGHGUI_ENABLE_PLUGINS)

  string(REPLACE "," ";" HIGHGUI_PLUGIN_LIST "${HIGHGUI_PLUGIN_LIST}")  # support comma-separated list (,) too
  string(TOLOWER "${HIGHGUI_PLUGIN_LIST}" HIGHGUI_PLUGIN_LIST)
  if(NOT HIGHGUI_ENABLE_PLUGINS)
    if(HIGHGUI_PLUGIN_LIST)
      message(WARNING "HighGUI: plugins are disabled through HIGHGUI_ENABLE_PLUGINS, so HIGHGUI_PLUGIN_LIST='${HIGHGUI_PLUGIN_LIST}' is ignored")
      set(HIGHGUI_PLUGIN_LIST "")
    endif()
  else()
    # Make virtual plugins target
    if(NOT TARGET opencv_highgui_plugins)
      add_custom_target(opencv_highgui_plugins ALL)
    endif()
  endif()
endif()

#
# Detect available dependencies
#

if(NOT PROJECT_NAME STREQUAL "OpenCV")
  include(FindPkgConfig)
endif()

macro(add_backend backend_id cond_var)
  if(${cond_var})
    include("${CMAKE_CURRENT_LIST_DIR}/detect_${backend_id}.cmake")
  endif()
endmacro()

add_backend("gtk" WITH_GTK)
add_backend("win32ui" WITH_WIN32UI)
add_backend("wayland" WITH_WAYLAND)
add_backend("framebuffer" WITH_FRAMEBUFFER)

# TODO cocoa
# TODO qt
# TODO opengl

# FIXIT: move content of cmake/OpenCVFindLibsGUI.cmake here (need to resolve CMake scope issues)
