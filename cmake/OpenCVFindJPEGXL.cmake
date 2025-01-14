# The script is taken from https://webkit.googlesource.com/WebKit/+/master/Source/cmake/FindJPEGXL.cmake

# Copyright (C) 2021 Sony Interactive Entertainment Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY APPLE INC. AND ITS CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL APPLE INC. OR ITS CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#[=======================================================================[.rst:
FindJPEGXL
---------
Find JPEGXL headers and libraries.
Imported Targets
^^^^^^^^^^^^^^^^
``JPEGXL::jxl``
  The JPEGXL library, if found.
Result Variables
^^^^^^^^^^^^^^^^
This will define the following variables in your project:
``JPEGXL_FOUND``
  true if (the requested version of) JPEGXL is available.
``JPEGXL_VERSION``
  the version of JPEGXL.
``JPEGXL_LIBRARIES``
  the libraries to link against to use JPEGXL.
``JPEGXL_INCLUDE_DIRS``
  where to find the JPEGXL headers.
``JPEGXL_COMPILE_OPTIONS``
  this should be passed to target_compile_options(), if the
  target is not used for linking
#]=======================================================================]

if(NOT OPENCV_SKIP_JPEGXL_FIND_PACKAGE)
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(PC_JPEGXL QUIET jxl)
    set(JPEGXL_COMPILE_OPTIONS ${PC_JPEGXL_CFLAGS_OTHER})
    set(JPEGXL_VERSION ${PC_JPEGXL_VERSION})
endif ()
find_path(JPEGXL_INCLUDE_DIR
    NAMES jxl/decode.h
    HINTS ${PC_JPEGXL_INCLUDEDIR} ${PC_JPEGXL_INCLUDE_DIRS} ${JPEGXL_INCLUDE_DIR}
)
find_library(JPEGXL_LIBRARY
    NAMES ${JPEGXL_NAMES} jxl
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
find_library(JPEGXL_CMS_LIBRARY
    NAMES ${JPEGXL_NAMES} jxl_cms
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
if (JPEGXL_LIBRARY AND JPEGXL_CMS_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_CMS_LIBRARY})
endif ()
find_library(JPEGXL_CMS2_LIBRARY
    NAMES ${JPEGXL_NAMES} lcms2
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
if (JPEGXL_LIBRARY AND JPEGXL_CMS2_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_CMS2_LIBRARY})
endif ()
find_library(JPEGXL_THREADS_LIBRARY
    NAMES ${JPEGXL_NAMES} jxl_threads
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
if (JPEGXL_LIBRARY AND JPEGXL_THREADS_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_THREADS_LIBRARY})
endif ()
find_library(JPEGXL_BROTLICOMMON_LIBRARY
    NAMES ${JPEGXL_NAMES} brotlicommon
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
find_library(JPEGXL_BROTLIDEC_LIBRARY
    NAMES ${JPEGXL_NAMES} brotlidec
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
if (JPEGXL_LIBRARY AND JPEGXL_BROTLIDEC_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_BROTLIDEC_LIBRARY})
endif ()
find_library(JPEGXL_BROTLIENC_LIBRARY
    NAMES ${JPEGXL_NAMES} brotlienc
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
if (JPEGXL_LIBRARY AND JPEGXL_BROTLIENC_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_BROTLIENC_LIBRARY})
endif ()
if (JPEGXL_LIBRARY AND JPEGXL_BROTLICOMMON_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_BROTLICOMMON_LIBRARY})
endif ()
find_library(JPEGXL_HWY_LIBRARY
    NAMES ${JPEGXL_NAMES} hwy
    HINTS ${PC_JPEGXL_LIBDIR} ${PC_JPEGXL_LIBRARY_DIRS}
)
if (JPEGXL_LIBRARY AND JPEGXL_HWY_LIBRARY)
    set(JPEGXL_LIBRARY ${JPEGXL_LIBRARY} ${JPEGXL_HWY_LIBRARY})
endif ()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JPEGXL
    FOUND_VAR JPEGXL_FOUND
    REQUIRED_VARS JPEGXL_LIBRARY JPEGXL_INCLUDE_DIR
    VERSION_VAR JPEGXL_VERSION
)
if (NOT EXISTS "${JPEGXL_INCLUDE_DIR}/jxl/version.h")
  # the library version older 0.6 is not supported (no version.h file there)
  set(JPEGXL_FOUND FALSE CACHE BOOL "libjxl found" FORCE)
  message(STATUS "Ignored incompatible version of libjxl")
endif ()

if (JPEGXL_LIBRARY AND NOT TARGET JPEGXL::jxl)
    add_library(JPEGXL::jxl UNKNOWN IMPORTED GLOBAL)
    set_target_properties(JPEGXL::jxl PROPERTIES
        IMPORTED_LOCATION "${JPEGXL_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${JPEGXL_COMPILE_OPTIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${JPEGXL_INCLUDE_DIR}"
    )
endif ()
mark_as_advanced(JPEGXL_INCLUDE_DIR JPEGXL_LIBRARY)
if (JPEGXL_FOUND)
    set(JPEGXL_LIBRARIES ${JPEGXL_LIBRARY})
    set(JPEGXL_INCLUDE_DIRS ${JPEGXL_INCLUDE_DIR})
    if (NOT JPEGXL_VERSION)
        file(READ "${JPEGXL_INCLUDE_DIR}/jxl/version.h" VERSION_HEADER_CONTENTS)
        string(REGEX MATCH "#define JPEGXL_MAJOR_VERSION ([0-9]+)" _ ${VERSION_HEADER_CONTENTS})
        set(JXL_VERSION_MAJOR ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define JPEGXL_MINOR_VERSION ([0-9]+)" _ ${VERSION_HEADER_CONTENTS})
        set(JXL_VERSION_MINOR ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define JPEGXL_PATCH_VERSION ([0-9]+)" _ ${VERSION_HEADER_CONTENTS})
        set(JXL_VERSION_PATCH ${CMAKE_MATCH_1})
        set(JPEGXL_VERSION "${JXL_VERSION_MAJOR}.${JXL_VERSION_MINOR}.${JXL_VERSION_PATCH}")
    endif()
endif ()
endif()
