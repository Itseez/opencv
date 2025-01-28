// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#if defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL)
#  include "opencv2/core/opencl/runtime/opencl_core.hpp"
#  include "opencv2/core/detail/va_intel_interop.hpp"

namespace cv { namespace va_intel {

VAAPIInterop::VAAPIInterop(cl_platform_id platform) {
    clCreateFromVA_APIMediaSurfaceINTEL       = (clCreateFromVA_APIMediaSurfaceINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromVA_APIMediaSurfaceINTEL");
    clEnqueueAcquireVA_APIMediaSurfacesINTEL  = (clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueAcquireVA_APIMediaSurfacesINTEL");
    clEnqueueReleaseVA_APIMediaSurfacesINTEL  = (clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueReleaseVA_APIMediaSurfacesINTEL");
    if (!clCreateFromVA_APIMediaSurfaceINTEL ||
        !clEnqueueAcquireVA_APIMediaSurfacesINTEL ||
        !clEnqueueReleaseVA_APIMediaSurfacesINTEL) {
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get extension function for VA-API interop");
    }
}
}} // namespace cv::va_intel
#endif /* defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL) */
