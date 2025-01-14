// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "videoio_registry.hpp"

#include "opencv2/videoio/registry.hpp"

#include "opencv2/core/utils/filesystem.private.hpp" // OPENCV_HAVE_FILESYSTEM_SUPPORT

#include "cap_librealsense.hpp"
#include "cap_dshow.hpp"

#ifdef HAVE_MFX
#include "cap_mfx_reader.hpp"
#include "cap_mfx_writer.hpp"
#endif

// All WinRT versions older than 8.0 should provide classes used for video support
#if defined(WINRT) && !defined(WINRT_8_0) && defined(__cplusplus_winrt)
#   include "cap_winrt_capture.hpp"
#   include "cap_winrt_bridge.hpp"
#   define WINRT_VIDEO
#endif

#if defined _M_X64 && defined _MSC_VER && !defined CV_ICC
#pragma optimize("",off)
#pragma warning(disable: 4748)
#endif

using namespace cv;

namespace cv {

namespace {

#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
#define DECLARE_DYNAMIC_BACKEND(cap, name, mode) \
{ \
    cap, (BackendMode)(mode), 1000, name, createPluginBackendFactory(cap, name) \
},
#else
#define DECLARE_DYNAMIC_BACKEND(cap, name, mode)  /* nothing */
#endif

#define DECLARE_STATIC_BACKEND(cap, name, mode, createCaptureFile, createCaptureCamera, createWriter) \
{ \
    cap, (BackendMode)(mode), 1000, name, createBackendFactory(createCaptureFile, createCaptureCamera, 0, createWriter) \
},

#define DECLARE_STATIC_BACKEND_WITH_STREAM_SUPPORT(cap, name, mode, createCaptureStream) \
{ \
    cap, (BackendMode)(mode), 1000, name, createBackendFactory(0, 0, createCaptureStream, 0) \
},

/** Ordering guidelines:
- modern optimized, multi-platform libraries: ffmpeg, gstreamer, Media SDK
- platform specific universal SDK: WINRT, AVFOUNDATION, MSMF/DSHOW, V4L/V4L2
- RGB-D: OpenNI/OpenNI2, REALSENSE, OBSENSOR
- special OpenCV (file-based): "images", "mjpeg"
- special camera SDKs, including stereo: other special SDKs: FIREWIRE/1394, XIMEA/ARAVIS/GIGANETIX/PVAPI(GigE)/uEye
- other: XINE, gphoto2, etc
*/
static const struct VideoBackendInfo builtin_backends[] =
{
#ifdef HAVE_FFMPEG
    DECLARE_STATIC_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, cvCreateFileCapture_FFMPEG_proxy, 0, cvCreateVideoWriter_FFMPEG_proxy)
    DECLARE_STATIC_BACKEND_WITH_STREAM_SUPPORT(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_STREAM, cvCreateStreamCapture_FFMPEG_proxy)
#elif defined(ENABLE_PLUGINS) || defined(HAVE_FFMPEG_WRAPPER)
    DECLARE_DYNAMIC_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_CAPTURE_BY_STREAM | MODE_WRITER)
#endif

#ifdef HAVE_GSTREAMER
    DECLARE_STATIC_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER, createGStreamerCapture_file, createGStreamerCapture_cam, create_GStreamer_writer)
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER)
#endif

#ifdef HAVE_MFX // Media SDK
    DECLARE_STATIC_BACKEND(CAP_INTEL_MFX, "INTEL_MFX", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, create_MFX_capture, 0, create_MFX_writer)
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_INTEL_MFX, "INTEL_MFX", MODE_CAPTURE_BY_FILENAME | MODE_WRITER)
#endif

    // Apple platform
#ifdef HAVE_AVFOUNDATION
    DECLARE_STATIC_BACKEND(CAP_AVFOUNDATION, "AVFOUNDATION", MODE_CAPTURE_ALL | MODE_WRITER, create_AVFoundation_capture_file, create_AVFoundation_capture_cam, create_AVFoundation_writer)
#endif

    // Windows
#ifdef WINRT_VIDEO
    DECLARE_STATIC_BACKEND(CAP_WINRT, "WINRT", MODE_CAPTURE_BY_INDEX, 0, create_WRT_capture, 0)
#endif

#ifdef HAVE_MSMF
    DECLARE_STATIC_BACKEND(CAP_MSMF, "MSMF", MODE_CAPTURE_ALL | MODE_WRITER, cvCreateCapture_MSMF, cvCreateCapture_MSMF, cvCreateVideoWriter_MSMF)
    DECLARE_STATIC_BACKEND_WITH_STREAM_SUPPORT(CAP_MSMF, "MSMF", MODE_CAPTURE_BY_STREAM, cvCreateCapture_MSMF)
#elif defined(ENABLE_PLUGINS) && defined(_WIN32)
    DECLARE_DYNAMIC_BACKEND(CAP_MSMF, "MSMF", MODE_CAPTURE_ALL | MODE_CAPTURE_BY_STREAM | MODE_WRITER)
#endif

#ifdef HAVE_DSHOW
    DECLARE_STATIC_BACKEND(CAP_DSHOW, "DSHOW", MODE_CAPTURE_BY_INDEX, 0, create_DShow_capture, 0)
#endif

    // Linux, some Unix
#if defined HAVE_CAMV4L2
    DECLARE_STATIC_BACKEND(CAP_V4L2, "V4L2", MODE_CAPTURE_ALL, create_V4L_capture_file, create_V4L_capture_cam, 0)
#elif defined HAVE_VIDEOIO
    DECLARE_STATIC_BACKEND(CAP_V4L, "V4L_BSD", MODE_CAPTURE_ALL, create_V4L_capture_file, create_V4L_capture_cam, 0)
#endif


    // RGB-D universal
#ifdef HAVE_OPENNI2
    DECLARE_STATIC_BACKEND(CAP_OPENNI2, "OPENNI2", MODE_CAPTURE_ALL, create_OpenNI2_capture_file, create_OpenNI2_capture_cam, 0)
#endif

#ifdef HAVE_LIBREALSENSE
    DECLARE_STATIC_BACKEND(CAP_REALSENSE, "INTEL_REALSENSE", MODE_CAPTURE_BY_INDEX, 0, create_RealSense_capture, 0)
#endif

    // OpenCV file-based only
    DECLARE_STATIC_BACKEND(CAP_IMAGES, "CV_IMAGES", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, create_Images_capture, 0, create_Images_writer)
    DECLARE_STATIC_BACKEND(CAP_OPENCV_MJPEG, "CV_MJPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, createMotionJpegCapture, 0, createMotionJpegWriter)

    // special interfaces / stereo cameras / other SDKs
#if defined(HAVE_DC1394_2)
    DECLARE_STATIC_BACKEND(CAP_FIREWIRE, "FIREWIRE", MODE_CAPTURE_BY_INDEX, 0, create_DC1394_capture, 0)
#endif
    // GigE
#ifdef HAVE_PVAPI
    DECLARE_STATIC_BACKEND(CAP_PVAPI, "PVAPI", MODE_CAPTURE_BY_INDEX, 0, create_PvAPI_capture, 0)
#endif
#ifdef HAVE_XIMEA
    DECLARE_STATIC_BACKEND(CAP_XIAPI, "XIMEA", MODE_CAPTURE_ALL, create_XIMEA_capture_file, create_XIMEA_capture_cam, 0)
#endif
#ifdef HAVE_ARAVIS_API
    DECLARE_STATIC_BACKEND(CAP_ARAVIS, "ARAVIS", MODE_CAPTURE_BY_INDEX, 0, create_Aravis_capture, 0)
#endif

#ifdef HAVE_UEYE // uEye
    DECLARE_STATIC_BACKEND(CAP_UEYE, "UEYE", MODE_CAPTURE_BY_INDEX, 0, create_ueye_camera, 0)
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_UEYE, "UEYE", MODE_CAPTURE_BY_INDEX)
#endif

#ifdef HAVE_GPHOTO2
    DECLARE_STATIC_BACKEND(CAP_GPHOTO2, "GPHOTO2", MODE_CAPTURE_ALL, createGPhoto2Capture, createGPhoto2Capture, 0)
#endif
#ifdef HAVE_XINE
    DECLARE_STATIC_BACKEND(CAP_XINE, "XINE", MODE_CAPTURE_BY_FILENAME, createXINECapture, 0, 0)
#endif
#if defined(HAVE_ANDROID_MEDIANDK) || defined(HAVE_ANDROID_NATIVE_CAMERA)
    DECLARE_STATIC_BACKEND(CAP_ANDROID, "ANDROID_NATIVE",
#ifdef HAVE_ANDROID_MEDIANDK
                           MODE_CAPTURE_BY_FILENAME | MODE_WRITER
#else
                           0
#endif
                           |
#ifdef HAVE_ANDROID_NATIVE_CAMERA
                           MODE_CAPTURE_BY_INDEX,
#else
                           0,
#endif
#ifdef HAVE_ANDROID_MEDIANDK
                           createAndroidCapture_file,
#else
                           0,
#endif
#ifdef HAVE_ANDROID_NATIVE_CAMERA
                           createAndroidCapture_cam,
#else
                           0,
#endif
#ifdef HAVE_ANDROID_MEDIANDK
                           createAndroidVideoWriter)
#else
                           0)
#endif
#endif

#ifdef HAVE_OBSENSOR
    DECLARE_STATIC_BACKEND(CAP_OBSENSOR, "OBSENSOR", MODE_CAPTURE_BY_INDEX, 0, create_obsensor_capture, 0)
#endif

    // dropped backends: MIL, TYZX
};

#if 0 // deprecated_backends is empty in 5.x for now
static const struct VideoDeprecatedBackendInfo deprecated_backends[] =
{
};
#endif

bool sortByPriority(const VideoBackendInfo &lhs, const VideoBackendInfo &rhs)
{
    return lhs.priority > rhs.priority;
}

/** @brief Manages list of enabled backends
 */
class VideoBackendRegistry
{
protected:
    std::vector<VideoBackendInfo> enabledBackends;
    VideoBackendRegistry()
    {
        const int N = sizeof(builtin_backends)/sizeof(builtin_backends[0]);
        enabledBackends.assign(builtin_backends, builtin_backends + N);
        for (int i = 0; i < N; i++)
        {
            VideoBackendInfo& info = enabledBackends[i];
            info.priority = 1000 - i * 10;
        }
        CV_LOG_DEBUG(NULL, "VIDEOIO: Builtin backends(" << N << "): " << dumpBackends());
        if (readPrioritySettings())
        {
            CV_LOG_INFO(NULL, "VIDEOIO: Updated backends priorities: " << dumpBackends());
        }
        int enabled = 0;
        for (int i = 0; i < N; i++)
        {
            VideoBackendInfo& info = enabledBackends[enabled];
            if (enabled != i)
                info = enabledBackends[i];
            size_t param_priority = utils::getConfigurationParameterSizeT(cv::format("OPENCV_VIDEOIO_PRIORITY_%s", info.name).c_str(), (size_t)info.priority);
            CV_Assert(param_priority == (size_t)(int)param_priority); // overflow check
            if (param_priority > 0)
            {
                info.priority = (int)param_priority;
                enabled++;
            }
            else
            {
                CV_LOG_INFO(NULL, "VIDEOIO: Disable backend: " << info.name);
            }
        }
        enabledBackends.resize(enabled);
        CV_LOG_DEBUG(NULL, "VIDEOIO: Available backends(" << enabled << "): " << dumpBackends());
        std::sort(enabledBackends.begin(), enabledBackends.end(), sortByPriority);
        CV_LOG_INFO(NULL, "VIDEOIO: Enabled backends(" << enabled << ", sorted by priority): " << dumpBackends());
    }

    static std::vector<std::string> tokenize_string(const std::string& input, char token)
    {
        std::vector<std::string> result;
        std::string::size_type prev_pos = 0, pos = 0;
        while((pos = input.find(token, pos)) != std::string::npos)
        {
            result.push_back(input.substr(prev_pos, pos-prev_pos));
            prev_pos = ++pos;
        }
        result.push_back(input.substr(prev_pos));
        return result;
    }
    bool readPrioritySettings()
    {
        bool hasChanges = false;
        cv::String prioritized_backends = utils::getConfigurationParameterString("OPENCV_VIDEOIO_PRIORITY_LIST");
        if (prioritized_backends.empty())
            return hasChanges;
        CV_LOG_INFO(NULL, "VIDEOIO: Configured priority list (OPENCV_VIDEOIO_PRIORITY_LIST): " << prioritized_backends);
        const std::vector<std::string> names = tokenize_string(prioritized_backends, ',');
        for (size_t i = 0; i < names.size(); i++)
        {
            const std::string& name = names[i];
            bool found = false;
            for (size_t k = 0; k < enabledBackends.size(); k++)
            {
                VideoBackendInfo& info = enabledBackends[k];
                if (name == info.name)
                {
                    info.priority = (int)(100000 + (names.size() - i) * 1000);
                    CV_LOG_DEBUG(NULL, "VIDEOIO: New backend priority: '" << name << "' => " << info.priority);
                    found = true;
                    hasChanges = true;
                    break;
                }
            }
            if (!found)
            {
                CV_LOG_WARNING(NULL, "VIDEOIO: Can't prioritize unknown/unavailable backend: '" << name << "'");
            }
        }
        return hasChanges;
    }
public:
    std::string dumpBackends() const
    {
        std::ostringstream os;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            if (i > 0) os << "; ";
            const VideoBackendInfo& info = enabledBackends[i];
            os << info.name << '(' << info.priority << ')';
        }
        return os.str();
    }

    static VideoBackendRegistry& getInstance()
    {
        static VideoBackendRegistry g_instance;
        return g_instance;
    }

    inline std::vector<VideoBackendInfo> getEnabledBackends() const { return enabledBackends; }

    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_CAPTURE_BY_INDEX)
                result.push_back(info);
        }
        return result;
    }
    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_CAPTURE_BY_FILENAME)
                result.push_back(info);
        }
        return result;
    }
    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByStream() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_CAPTURE_BY_STREAM)
                result.push_back(info);
        }
        return result;
    }
    inline std::vector<VideoBackendInfo> getAvailableBackends_Writer() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_WRITER)
                result.push_back(info);
        }
        return result;
    }
};

} // namespace

namespace videoio_registry {

std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByIndex();
    return result;
}
std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByFilename();
    return result;
}
std::vector<VideoBackendInfo> getAvailableBackends_CaptureByStream()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByStream();
    return result;
}
std::vector<VideoBackendInfo> getAvailableBackends_Writer()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_Writer();
    return result;
}

bool checkDeprecatedBackend(int api) {
    CV_UNUSED(api);
#if 0 // deprecated_backends is empty in 5.x for now
    const int M = sizeof(deprecated_backends) / sizeof(deprecated_backends[0]);
    for (size_t i = 0; i < M; i++)
    {
        if (deprecated_backends[i].id == api)
            return true;
    }
#endif
    return false;
}

cv::String getBackendName(VideoCaptureAPIs api)
{
    if (api == CAP_ANY)
        return "CAP_ANY";  // special case, not a part of backends list
    const int N = sizeof(builtin_backends)/sizeof(builtin_backends[0]);
    for (size_t i = 0; i < N; i++)
    {
        const VideoBackendInfo& backend = builtin_backends[i];
        if (backend.id == api)
            return backend.name;
    }
#if 0 // deprecated_backends is empty in 5.x for now
    const int M = sizeof(deprecated_backends) / sizeof(deprecated_backends[0]);
    for (size_t i = 0; i < M; i++)
    {
        if (deprecated_backends[i].id == api)
            return deprecated_backends[i].name;
    }
#endif
    return cv::format("UnknownVideoAPI(%d)", (int)api);
}

std::vector<VideoCaptureAPIs> getBackends()
{
    std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getEnabledBackends();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;
}

std::vector<VideoCaptureAPIs> getCameraBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByIndex();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;

}

std::vector<VideoCaptureAPIs> getStreamBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByFilename();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;

}

std::vector<VideoCaptureAPIs> getStreamBufferedBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByStream();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;
}

std::vector<VideoCaptureAPIs> getWriterBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_Writer();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;
}

bool hasBackend(VideoCaptureAPIs api)
{
    std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getEnabledBackends();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (api == info.id)
        {
            CV_Assert(!info.backendFactory.empty());
            return !info.backendFactory->getBackend().empty();
        }
    }
    return false;
}

bool isBackendBuiltIn(VideoCaptureAPIs api)
{
    std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getEnabledBackends();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (api == info.id)
        {
            CV_Assert(!info.backendFactory.empty());
            return info.backendFactory->isBuiltIn();
        }
    }
    return false;
}

std::string getCameraBackendPluginVersion(VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
)
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByIndex();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (api == info.id)
        {
            CV_Assert(!info.backendFactory.empty());
            CV_Assert(!info.backendFactory->isBuiltIn());
            return getCapturePluginVersion(info.backendFactory, version_ABI, version_API);
        }
    }
    CV_Error(Error::StsError, "Unknown or wrong backend ID");
}

std::string getStreamBackendPluginVersion(VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
)
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByFilename();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (api == info.id)
        {
            CV_Assert(!info.backendFactory.empty());
            CV_Assert(!info.backendFactory->isBuiltIn());
            return getCapturePluginVersion(info.backendFactory, version_ABI, version_API);
        }
    }
    CV_Error(Error::StsError, "Unknown or wrong backend ID");
}

std::string getStreamBufferedBackendPluginVersion(VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
)
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByStream();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (api == info.id)
        {
            CV_Assert(!info.backendFactory.empty());
            CV_Assert(!info.backendFactory->isBuiltIn());
            return getCapturePluginVersion(info.backendFactory, version_ABI, version_API);
        }
    }
    CV_Error(Error::StsError, "Unknown or wrong backend ID");
}

/** @brief Returns description and ABI/API version of videoio plugin's writer interface */
std::string getWriterBackendPluginVersion(VideoCaptureAPIs api,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
)
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_Writer();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (api == info.id)
        {
            CV_Assert(!info.backendFactory.empty());
            CV_Assert(!info.backendFactory->isBuiltIn());
            return getWriterPluginVersion(info.backendFactory, version_ABI, version_API);
        }
    }
    CV_Error(Error::StsError, "Unknown or wrong backend ID");
}


} // namespace registry

} // namespace
