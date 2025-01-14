// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "backend.hpp"

namespace cv {


void applyParametersFallback(const Ptr<IVideoCapture>& cap, const VideoCaptureParameters& params)
{
    std::vector<int> props = params.getUnused();
    CV_LOG_INFO(NULL, "VIDEOIO: Backend '" << videoio_registry::getBackendName((VideoCaptureAPIs)cap->getCaptureDomain()) <<
                      "' implementation doesn't support parameters in .open(). Applying " <<
                      props.size() << " properties through .setProperty()");
    for (int prop : props)
    {
        double value = params.get<double>(prop, -1);
        CV_LOG_INFO(NULL, "VIDEOIO: apply parameter: [" << prop << "]=" <<
                          cv::format("%g / %lld / 0x%016llx", value, (long long)value, (long long)value));
        if (!cap->setProperty(prop, value))
        {
            if (prop != CAP_PROP_HW_ACCELERATION && prop != CAP_PROP_HW_DEVICE) { // optional parameters
                CV_Error_(cv::Error::StsNotImplemented, ("VIDEOIO: Failed to apply invalid or unsupported parameter: [%d]=%g / %lld / 0x%08llx", prop, value, (long long)value, (long long)value));
            }
        }
    }
    // NB: there is no dedicated "commit" parameters event, implementations should commit after each property automatically
}

// Legacy API. Modern API with parameters is below
class StaticBackend: public IBackend
{
public:
    FN_createCaptureFile fn_createCaptureFile_;
    FN_createCaptureCamera fn_createCaptureCamera_;
    FN_createCaptureStream fn_createCaptureStream_;
    FN_createWriter fn_createWriter_;

    StaticBackend(FN_createCaptureFile fn_createCaptureFile, FN_createCaptureCamera fn_createCaptureCamera, FN_createCaptureStream fn_createCaptureStream, FN_createWriter fn_createWriter)
        : fn_createCaptureFile_(fn_createCaptureFile), fn_createCaptureCamera_(fn_createCaptureCamera), fn_createCaptureStream_(fn_createCaptureStream), fn_createWriter_(fn_createWriter)
    {
        // nothing
    }

    ~StaticBackend() CV_OVERRIDE {}

    Ptr<IVideoCapture> createCapture(int camera, const VideoCaptureParameters& params) const CV_OVERRIDE
    {
        if (fn_createCaptureCamera_)
        {
            Ptr<IVideoCapture> cap = fn_createCaptureCamera_(camera);
            if (cap && !params.empty())
            {
                applyParametersFallback(cap, params);
            }
            return cap;
        }
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoCapture> createCapture(const std::string &filename, const VideoCaptureParameters& params) const CV_OVERRIDE
    {
        if (fn_createCaptureFile_)
        {
            Ptr<IVideoCapture> cap = fn_createCaptureFile_(filename);
            if (cap && !params.empty())
            {
                applyParametersFallback(cap, params);
            }
            return cap;
        }
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoCapture> createCapture(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params) const CV_OVERRIDE
    {
        if (fn_createCaptureStream_)
        {
            Ptr<IVideoCapture> cap = fn_createCaptureStream_(stream);
            if (cap && !params.empty())
            {
                applyParametersFallback(cap, params);
            }
            return cap;
        }
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoWriter> createWriter(const std::string& filename, int fourcc, double fps,
                                   const cv::Size& sz, const VideoWriterParameters& params) const CV_OVERRIDE
    {
        if (fn_createWriter_)
            return fn_createWriter_(filename, fourcc, fps, sz, params);
        return Ptr<IVideoWriter>();
    }
}; // StaticBackend

class StaticBackendFactory : public IBackendFactory
{
protected:
    Ptr<StaticBackend> backend;

public:
    StaticBackendFactory(FN_createCaptureFile createCaptureFile, FN_createCaptureCamera createCaptureCamera, FN_createCaptureStream createCaptureStream, FN_createWriter createWriter)
        : backend(makePtr<StaticBackend>(createCaptureFile, createCaptureCamera, createCaptureStream, createWriter))
    {
        // nothing
    }

    ~StaticBackendFactory() CV_OVERRIDE {}

    Ptr<IBackend> getBackend() const CV_OVERRIDE
    {
        return backend.staticCast<IBackend>();
    }

    bool isBuiltIn() const CV_OVERRIDE { return true; }
};


Ptr<IBackendFactory> createBackendFactory(FN_createCaptureFile createCaptureFile,
                                          FN_createCaptureCamera createCaptureCamera,
                                          FN_createCaptureStream createCaptureStream,
                                          FN_createWriter createWriter)
{
    return makePtr<StaticBackendFactory>(createCaptureFile, createCaptureCamera, createCaptureStream, createWriter).staticCast<IBackendFactory>();
}



class StaticBackendWithParams: public IBackend
{
public:
    FN_createCaptureFileWithParams fn_createCaptureFile_;
    FN_createCaptureCameraWithParams fn_createCaptureCamera_;
    FN_createCaptureStreamWithParams fn_createCaptureStream_;
    FN_createWriter fn_createWriter_;

    StaticBackendWithParams(FN_createCaptureFileWithParams fn_createCaptureFile, FN_createCaptureCameraWithParams fn_createCaptureCamera, FN_createCaptureStreamWithParams fn_createCaptureStream, FN_createWriter fn_createWriter)
        : fn_createCaptureFile_(fn_createCaptureFile), fn_createCaptureCamera_(fn_createCaptureCamera), fn_createCaptureStream_(fn_createCaptureStream), fn_createWriter_(fn_createWriter)
    {
        // nothing
    }

    ~StaticBackendWithParams() CV_OVERRIDE {}

    Ptr<IVideoCapture> createCapture(int camera, const VideoCaptureParameters& params) const CV_OVERRIDE
    {
        if (fn_createCaptureCamera_)
            return fn_createCaptureCamera_(camera, params);
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoCapture> createCapture(const std::string &filename, const VideoCaptureParameters& params) const CV_OVERRIDE
    {
        if (fn_createCaptureFile_)
            return fn_createCaptureFile_(filename, params);
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoCapture> createCapture(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params) const CV_OVERRIDE
    {
        if (fn_createCaptureStream_)
            return fn_createCaptureStream_(stream, params);
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoWriter> createWriter(const std::string& filename, int fourcc, double fps,
                                   const cv::Size& sz, const VideoWriterParameters& params) const CV_OVERRIDE
    {
        if (fn_createWriter_)
            return fn_createWriter_(filename, fourcc, fps, sz, params);
        return Ptr<IVideoWriter>();
    }
}; // StaticBackendWithParams

class StaticBackendWithParamsFactory : public IBackendFactory
{
protected:
    Ptr<StaticBackendWithParams> backend;

public:
    StaticBackendWithParamsFactory(FN_createCaptureFileWithParams createCaptureFile, FN_createCaptureCameraWithParams createCaptureCamera, FN_createCaptureStreamWithParams createCaptureStream, FN_createWriter createWriter)
        : backend(makePtr<StaticBackendWithParams>(createCaptureFile, createCaptureCamera, createCaptureStream, createWriter))
    {
        // nothing
    }

    ~StaticBackendWithParamsFactory() CV_OVERRIDE {}

    Ptr<IBackend> getBackend() const CV_OVERRIDE
    {
        return backend.staticCast<IBackend>();
    }

    bool isBuiltIn() const CV_OVERRIDE { return true; }
};


Ptr<IBackendFactory> createBackendFactory(FN_createCaptureFileWithParams createCaptureFile,
                                          FN_createCaptureCameraWithParams createCaptureCamera,
                                          FN_createCaptureStreamWithParams createCaptureStream,
                                          FN_createWriter createWriter)
{
    return makePtr<StaticBackendWithParamsFactory>(createCaptureFile, createCaptureCamera, createCaptureStream, createWriter).staticCast<IBackendFactory>();
}


} // namespace
