#COPYRIGHT
#
#All contributions by the University of California:
#Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
#All rights reserved.
#
#All other contributions:
#Copyright (c) 2014, 2015, the respective contributors
#All rights reserved.
#
#Caffe uses a shared copyright model: each contributor holds copyright over
#their contributions to Caffe. The project versioning records all such
#contribution and copyright details. If a contributor wants to further mark
#their specific copyright on a particular contribution, they should indicate
#their copyright solely in the commit message of the change when it is
#committed.
#
#LICENSE
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#CONTRIBUTION AGREEMENT
#
#By contributing to the BVLC/caffe repository through pull-request, comment,
#or otherwise, the contributor releases their content to the
#license and copyright terms herein.

SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  $ENV{OpenBLAS_HOME}/include/openblas
  /opt/OpenBLAS/include
  /usr/local/include/openblas
  /usr/include/openblas
  /usr/local/include/openblas-base
  /usr/include/openblas-base
  /usr/local/include
  /usr/include
  /usr/include/x86_64-linux-gnu
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        $ENV{OpenBLAS}
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
        /opt/OpenBLAS/lib
        /usr/local/lib64
        /usr/local/lib
        /lib/openblas-base
        /lib64/
        /lib/
        /usr/lib/openblas-base
        /usr/lib64
        /usr/lib
	/usr/lib/x86_64-linux-gnu
)

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS} NO_DEFAULT_PATH)
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas libopenblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS}  NO_DEFAULT_PATH)

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)
