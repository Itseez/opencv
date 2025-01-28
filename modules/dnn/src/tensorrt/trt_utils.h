// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "NvInferRuntimeCommon.h"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
namespace dnn_trt {

/*
 * Get compute capability
 *
 */
std::string getComputeCapacity(const cudaDeviceProp& prop) {
  const std::string compute_capability = std::to_string(prop.major * 10 + prop.minor);
  return compute_capability;
}

/*
 * Get Timing by compute capability
 *
 */
std::string getTimingCachePath(const std::string& root, std::string& compute_cap) {
  // append compute capability of the GPU as this invalidates the cache and TRT will throw when loading the cache
  const std::string timing_cache_name = root + "OpenCV_DNN_TensorRT_cache_sm" +
                                        compute_cap + ".timing";
  return timing_cache_name;
}

}
CV__DNN_INLINE_NS_END
}}