#=============================================================================
# Find AVIF library
#=============================================================================
# Find the native AVIF headers and libraries.
#
#  AVIF_INCLUDE_DIRS - where to find avif/avif.h, etc.
#  AVIF_LIBRARIES    - List of libraries when using AVIF.
#  AVIF_FOUND        - True if AVIF is found.
#=============================================================================

# Look for the header file.

unset(AVIF_FOUND)

find_package(libavif QUIET)

if(TARGET avif)
    MARK_AS_ADVANCED(AVIF_INCLUDE_DIR)
    MARK_AS_ADVANCED(AVIF_LIBRARY)

    SET(AVIF_FOUND TRUE)
    SET(AVIF_LIBRARY avif)
    GET_TARGET_PROPERTY(AVIF_INCLUDE_DIR1 avif INCLUDE_DIRECTORIES)
    GET_TARGET_PROPERTY(AVIF_INCLUDE_DIR2 avif INTERFACE_INCLUDE_DIRECTORIES)
    GET_TARGET_PROPERTY(AVIF_DEPENDENCIES avif INTERFACE_LINK_LIBRARIES)
    if(AVIF_DEPENDENCIES)
        LIST(INSERT AVIF_LIBRARY 0 ${AVIF_DEPENDENCIES})
    endif()
    set(AVIF_INCLUDE_DIR)
    if(AVIF_INCLUDE_DIR1)
        LIST(APPEND AVIF_INCLUDE_DIR ${AVIF_INCLUDE_DIR1})
    endif()
    if(AVIF_INCLUDE_DIR2)
        LIST(APPEND AVIF_INCLUDE_DIR ${AVIF_INCLUDE_DIR2})
    endif()
else()
    FIND_PATH(AVIF_INCLUDE_DIR NAMES avif/avif.h)

     # Look for the library.
    FIND_LIBRARY(AVIF_LIBRARY NAMES avif)
    MARK_AS_ADVANCED(AVIF_LIBRARY)

    # handle the QUIETLY and REQUIRED arguments and set AVIF_FOUND to TRUE if
    # all listed variables are TRUE
    INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(AVIF DEFAULT_MSG AVIF_LIBRARY AVIF_INCLUDE_DIR)

    SET(AVIF_LIBRARIES ${AVIF_LIBRARY})
    SET(AVIF_INCLUDE_DIRS ${AVIF_INCLUDE_DIR})
endif()
