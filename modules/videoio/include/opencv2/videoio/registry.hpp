// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_REGISTRY_HPP
#define OPENCV_VIDEOIO_REGISTRY_HPP

#include <opencv2/videoio.hpp>

namespace cv { namespace videoio_registry {
/** @addtogroup videoio_registry
This section contains API description how to query/configure available Video I/O backends.

Runtime configuration options:
- enable debug mode: `OPENCV_VIDEOIO_DEBUG=1`
- change backend priority: `OPENCV_VIDEOIO_PRIORITY_<backend>=9999`
- disable backend: `OPENCV_VIDEOIO_PRIORITY_<backend>=0`
- specify list of backends with high priority (>100000): `OPENCV_VIDEOIO_PRIORITY_LIST=FFMPEG,GSTREAMER`

@{
 */


/** @brief Returns backend API name or "UnknownVideoAPI(xxx)"
@param api backend ID (#VideoCaptureAPIs)
*/
CV_EXPORTS_W cv::String getBackendName(VideoCaptureAPIs api);

/** @brief Returns list of all available backends */
CV_EXPORTS_W std::vector<VideoCaptureAPIs> getBackends();

/** @brief Returns list of available backends which works via `cv::VideoCapture(int index)` */
CV_EXPORTS_W std::vector<VideoCaptureAPIs> getCameraBackends();

/** @brief Returns list of available backends which works via `cv::VideoCapture(filename)` */
CV_EXPORTS_W std::vector<VideoCaptureAPIs> getStreamBackends();

/** @brief Returns list of available backends which works via `cv::VideoCapture(buffer)` */
CV_EXPORTS_W std::vector<VideoCaptureAPIs> getStreamBufferedBackends();

/** @brief Returns list of available backends which works via `cv::VideoWriter()` */
CV_EXPORTS_W std::vector<VideoCaptureAPIs> getWriterBackends();

/** @brief Returns true if backend is available */
CV_EXPORTS_W bool hasBackend(VideoCaptureAPIs api);

/** @brief Returns true if backend is built in (false if backend is used as plugin) */
CV_EXPORTS_W bool isBackendBuiltIn(VideoCaptureAPIs api);

/** @brief Returns description and ABI/API version of videoio plugin's camera interface */
CV_EXPORTS_W std::string getCameraBackendPluginVersion(
    VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
);

/** @brief Returns description and ABI/API version of videoio plugin's stream capture interface */
CV_EXPORTS_W std::string getStreamBackendPluginVersion(
    VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
);

/** @brief Returns description and ABI/API version of videoio plugin's buffer capture interface */
CV_EXPORTS_W std::string getStreamBufferedBackendPluginVersion(
    VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
);

/** @brief Returns description and ABI/API version of videoio plugin's writer interface */
CV_EXPORTS_W std::string getWriterBackendPluginVersion(
    VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
);


//! @}
}} // namespace

#endif // OPENCV_VIDEOIO_REGISTRY_HPP
