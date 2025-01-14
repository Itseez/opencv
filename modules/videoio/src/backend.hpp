// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef BACKEND_HPP_DEFINED
#define BACKEND_HPP_DEFINED

#include "cap_interface.hpp"
#include "opencv2/videoio/registry.hpp"

namespace cv {

// TODO: move to public interface
// TODO: allow runtime backend registration
class IBackend
{
public:
    virtual ~IBackend() {}
    virtual Ptr<IVideoCapture> createCapture(int camera, const VideoCaptureParameters& params) const = 0;
    virtual Ptr<IVideoCapture> createCapture(const std::string &filename, const VideoCaptureParameters& params) const = 0;
    virtual Ptr<IVideoCapture> createCapture(const Ptr<IStreamReader>&stream, const VideoCaptureParameters& params) const = 0;
    virtual Ptr<IVideoWriter> createWriter(const std::string& filename, int fourcc, double fps, const cv::Size& sz,
                                           const VideoWriterParameters& params) const = 0;
};

class IBackendFactory
{
public:
    virtual ~IBackendFactory() {}
    virtual Ptr<IBackend> getBackend() const = 0;
    virtual bool isBuiltIn() const = 0;
};

//=============================================================================

typedef Ptr<IVideoCapture> (*FN_createCaptureFile)(const std::string & filename);
typedef Ptr<IVideoCapture> (*FN_createCaptureCamera)(int camera);
typedef Ptr<IVideoCapture> (*FN_createCaptureStream)(const Ptr<IStreamReader>& stream);
typedef Ptr<IVideoCapture> (*FN_createCaptureFileWithParams)(const std::string & filename, const VideoCaptureParameters& params);
typedef Ptr<IVideoCapture> (*FN_createCaptureCameraWithParams)(int camera, const VideoCaptureParameters& params);
typedef Ptr<IVideoCapture> (*FN_createCaptureStreamWithParams)(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params);
typedef Ptr<IVideoWriter>  (*FN_createWriter)(const std::string& filename, int fourcc, double fps, const Size& sz,
                                              const VideoWriterParameters& params);
Ptr<IBackendFactory> createBackendFactory(FN_createCaptureFile createCaptureFile,
                                          FN_createCaptureCamera createCaptureCamera,
                                          FN_createCaptureStream createCaptureStream,
                                          FN_createWriter createWriter);
Ptr<IBackendFactory> createBackendFactory(FN_createCaptureFileWithParams createCaptureFile,
                                          FN_createCaptureCameraWithParams createCaptureCamera,
                                          FN_createCaptureStreamWithParams createCaptureStream,
                                          FN_createWriter createWriter);

Ptr<IBackendFactory> createPluginBackendFactory(VideoCaptureAPIs id, const char* baseName);

void applyParametersFallback(const Ptr<IVideoCapture>& cap, const VideoCaptureParameters& params);

std::string getCapturePluginVersion(
    const Ptr<IBackendFactory>& backend_factory,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
);
std::string getWriterPluginVersion(
    const Ptr<IBackendFactory>& backend_factory,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
);

} // namespace cv::

#endif // BACKEND_HPP_DEFINED
