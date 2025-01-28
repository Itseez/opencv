// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_VA_INTEL_INTEROP_HPP
#define OPENCV_CORE_VA_INTEL_INTEROP_HPP

#ifndef __cplusplus
#  error va_intel_interop.hpp header must be compiled as C++
#endif

#if defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL)
#  include <CL/cl.h>
#  ifdef HAVE_VA_INTEL_OLD_HEADER
#    include <CL/va_ext.h>
#  else
#    include <CL/cl_va_api_media_sharing_intel.h>
#  endif
#  include "opencv2/core.hpp"
#  include "opencv2/core/ocl.hpp"

namespace cv { namespace va_intel {

class VAAPIInterop : public cv::ocl::Context::UserContext
{
public:
    VAAPIInterop(cl_platform_id platform);
    virtual ~VAAPIInterop() {};
    clCreateFromVA_APIMediaSurfaceINTEL_fn       clCreateFromVA_APIMediaSurfaceINTEL;
    clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn  clEnqueueAcquireVA_APIMediaSurfacesINTEL;
    clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn  clEnqueueReleaseVA_APIMediaSurfacesINTEL;
};

}} // namespace cv::va_intel
#endif /* defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL) */
#endif /* OPENCV_CORE_VA_INTEL_INTEROP_HPP */
