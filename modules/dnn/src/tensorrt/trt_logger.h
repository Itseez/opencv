// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "NvInferRuntimeCommon.h"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
namespace dnn_trt {

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      struct tm stm;
#ifdef _MSC_VER
      gmtime_s(&stm, &rawtime);
#else
      gmtime_r(&rawtime, &stm);
#endif
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               &stm);
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR"
                                                                            : severity == Severity::kWARNING ? "WARNING"
                                                                            : severity == Severity::kINFO    ? "   INFO"
                                                                                                             : "UNKNOWN");
      if (severity <= Severity::kERROR) {
        CV_LOG_ERROR(NULL, "[" << buf << " " << sevstr << "] " << msg);
      } else {
        CV_LOG_WARNING(NULL, "[" << buf << " " << sevstr << "] " << msg);
      }
    }
  }
  void set_level(Severity verbosity) {
    verbosity_ = verbosity;
  }
  Severity get_level() const {
    return verbosity_;
  }
};

}
CV__DNN_INLINE_NS_END
}}  //