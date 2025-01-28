// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2015, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_VA_INTEL_HPP
#define OPENCV_CORE_VA_INTEL_HPP

#ifndef __cplusplus
#  error va_intel.hpp header must be compiled as C++
#endif

#include "ocl.hpp"

#if defined(HAVE_VA)
# include "va/va.h"
#else  // HAVE_VA
# if !defined(_VA_H_)
    typedef void* VADisplay;
    typedef unsigned int VASurfaceID;
# endif // !_VA_H_
#endif // HAVE_VA

namespace cv { namespace va_intel {

/** @addtogroup core_va_intel
This section describes Intel VA-API/OpenCL (CL-VA) interoperability.

To enable basic VA interoperability build OpenCV with libva library integration enabled: `-DWITH_VA=ON` (corresponding dev package should be installed).

To enable advanced CL-VA interoperability support on Intel HW, enable option: `-DWITH_VA_INTEL=ON` (OpenCL integration should be enabled which is the default setting). Special runtime environment should be set up in order to use this feature: correct combination of [libva](https://github.com/intel/libva), [OpenCL runtime](https://github.com/intel/compute-runtime) and [media driver](https://github.com/intel/media-driver) should be installed.

Check usage example for details: samples/va_intel/va_intel_interop.cpp
*/
//! @{

/////////////////// CL-VA Interoperability Functions ///////////////////

namespace ocl {
using namespace cv::ocl;

// TODO static functions in the Context class
/** @brief Creates OpenCL context from VA.
@param display    - VADisplay for which CL interop should be established.
@return Returns reference to OpenCL Context
 */
CV_EXPORTS Context& initializeContextFromVA(VADisplay display);

} // namespace cv::va_intel::ocl

/** @brief Converts InputArray to VASurfaceID object.
@param display - VADisplay object.
@param src     - source InputArray.
@param surface - destination VASurfaceID object.
@param size    - size of image represented by VASurfaceID object.
 */
CV_EXPORTS void convertToVASurface(VADisplay display, InputArray src, VASurfaceID surface, Size size);

/** @brief Converts VASurfaceID object to OutputArray.
@param display - VADisplay object.
@param surface - source VASurfaceID object.
@param size    - size of image represented by VASurfaceID object.
@param dst     - destination OutputArray.
 */
CV_EXPORTS void convertFromVASurface(VADisplay display, VASurfaceID surface, Size size, OutputArray dst);

//! @}

}} // namespace cv::va_intel

#endif /* OPENCV_CORE_VA_INTEL_HPP */
