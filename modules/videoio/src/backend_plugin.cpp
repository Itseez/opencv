// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "backend.hpp"
#include "plugin_api.hpp"
#include "plugin_capture_api.hpp"
#include "plugin_writer_api.hpp"

#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "opencv2/core/private.hpp"
#include "videoio_registry.hpp"

//==================================================================================================
// Dynamic backend implementation

#include "opencv2/core/utils/plugin_loader.private.hpp"


#include "backend_plugin_legacy.impl.hpp"


namespace cv { namespace impl {

#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)

using namespace cv::plugin::impl;  // plugin_loader.hpp

static Mutex& getInitializationMutex()
{
    static Mutex initializationMutex;
    return initializationMutex;
}


class PluginBackend: public IBackend
{
protected:

    void initCaptureAPI()
    {
        const char* init_name = "opencv_videoio_capture_plugin_init_v1";
        FN_opencv_videoio_capture_plugin_init_t fn_init = reinterpret_cast<FN_opencv_videoio_capture_plugin_init_t>(lib_->getSymbol(init_name));
        if (fn_init)
        {
            CV_LOG_INFO(NULL, "Found entry: '" << init_name << "'");
            for (int supported_api_version = CAPTURE_API_VERSION; supported_api_version >= 0; supported_api_version--)
            {
                capture_api_ = fn_init(CAPTURE_ABI_VERSION, supported_api_version, NULL);
                if (capture_api_)
                    break;
            }
            if (!capture_api_)
            {
                CV_LOG_INFO(NULL, "Video I/O: plugin is incompatible (can't be initialized): " << lib_->getName());
                return;
            }
            if (!checkCompatibility(
                    capture_api_->api_header, CAPTURE_ABI_VERSION, CAPTURE_API_VERSION,
                    capture_api_->v0.id != CAP_FFMPEG))
            {
                capture_api_ = NULL;
                return;
            }
            CV_LOG_INFO(NULL, "Video I/O: plugin is ready to use '" << capture_api_->api_header.api_description << "'");
        }
        else
        {
            CV_LOG_INFO(NULL, "Video I/O: missing plugin init function: '" << init_name << "', file: " << lib_->getName());
        }
    }


    void initWriterAPI()
    {
        const char* init_name = "opencv_videoio_writer_plugin_init_v1";
        FN_opencv_videoio_writer_plugin_init_t fn_init = reinterpret_cast<FN_opencv_videoio_writer_plugin_init_t>(lib_->getSymbol(init_name));
        if (fn_init)
        {
            CV_LOG_INFO(NULL, "Found entry: '" << init_name << "'");
            for (int supported_api_version = WRITER_API_VERSION; supported_api_version >= 0; supported_api_version--)
            {
                writer_api_ = fn_init(WRITER_ABI_VERSION, supported_api_version, NULL);
                if (writer_api_)
                    break;
            }
            if (!writer_api_)
            {
                CV_LOG_INFO(NULL, "Video I/O: plugin is incompatible (can't be initialized): " << lib_->getName());
                return;
            }
            if (!checkCompatibility(
                    writer_api_->api_header, WRITER_ABI_VERSION, WRITER_API_VERSION,
                    writer_api_->v0.id != CAP_FFMPEG))
            {
                writer_api_ = NULL;
                return;
            }
            CV_LOG_INFO(NULL, "Video I/O: plugin is ready to use '" << writer_api_->api_header.api_description << "'");
        }
        else
        {
            CV_LOG_INFO(NULL, "Video I/O: missing plugin init function: '" << init_name << "', file: " << lib_->getName());
        }
    }


    void initPluginLegacyAPI()
    {
        const char* init_name = "opencv_videoio_plugin_init_v0";
        FN_opencv_videoio_plugin_init_t fn_init = reinterpret_cast<FN_opencv_videoio_plugin_init_t>(lib_->getSymbol(init_name));
        if (fn_init)
        {
            CV_LOG_INFO(NULL, "Found entry: '" << init_name << "'");
            for (int supported_api_version = API_VERSION; supported_api_version >= 0; supported_api_version--)
            {
                plugin_api_ = fn_init(ABI_VERSION, supported_api_version, NULL);
                if (plugin_api_)
                    break;
            }
            if (!plugin_api_)
            {
                CV_LOG_INFO(NULL, "Video I/O: plugin is incompatible (can't be initialized): " << lib_->getName());
                return;
            }
            if (!checkCompatibility(
                    plugin_api_->api_header, ABI_VERSION, API_VERSION,
                    plugin_api_->v0.captureAPI != CAP_FFMPEG))
            {
                plugin_api_ = NULL;
                return;
            }
            CV_LOG_INFO(NULL, "Video I/O: plugin is ready to use '" << plugin_api_->api_header.api_description << "'");
        }
        else
        {
            CV_LOG_INFO(NULL, "Video I/O: plugin is incompatible, missing init function: '" << init_name << "', file: " << lib_->getName());
        }
    }


    bool checkCompatibility(const OpenCV_API_Header& api_header, unsigned int abi_version, unsigned int api_version, bool checkMinorOpenCVVersion)
    {
        if (api_header.opencv_version_major != CV_VERSION_MAJOR)
        {
            CV_LOG_ERROR(NULL, "Video I/O: wrong OpenCV major version used by plugin '" << api_header.api_description << "': " <<
                cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", api_header.opencv_version_major, api_header.opencv_version_minor))
            return false;
        }
        if (!checkMinorOpenCVVersion)
        {
            // no checks for OpenCV minor version
        }
        else if (api_header.opencv_version_minor != CV_VERSION_MINOR)
        {
            CV_LOG_ERROR(NULL, "Video I/O: wrong OpenCV minor version used by plugin '" << api_header.api_description << "': " <<
                cv::format("%d.%d, OpenCV version is '" CV_VERSION "'", api_header.opencv_version_major, api_header.opencv_version_minor))
            return false;
        }
        CV_LOG_INFO(NULL, "Video I/O: initialized '" << api_header.api_description << "': built with "
            << cv::format("OpenCV %d.%d (ABI/API = %d/%d)",
                 api_header.opencv_version_major, api_header.opencv_version_minor,
                 api_header.min_api_version, api_header.api_version)
            << ", current OpenCV version is '" CV_VERSION "' (ABI/API = " << abi_version << "/" << api_version << ")"
        );
        if (api_header.min_api_version != abi_version)  // future: range can be here
        {
            // actually this should never happen due to checks in plugin's init() function
            CV_LOG_ERROR(NULL, "Video I/O: plugin is not supported due to incompatible ABI = " << api_header.min_api_version);
            return false;
        }
        if (api_header.api_version != api_version)
        {
            CV_LOG_INFO(NULL, "Video I/O: NOTE: plugin is supported, but there is API version mismath: "
                << cv::format("plugin API level (%d) != OpenCV API level (%d)", api_header.api_version, api_version));
            if (api_header.api_version < api_version)
            {
                CV_LOG_INFO(NULL, "Video I/O: NOTE: some functionality may be unavailable due to lack of support by plugin implementation");
            }
        }
        return true;
    }

public:
    Ptr<cv::plugin::impl::DynamicLib> lib_;
    const OpenCV_VideoIO_Capture_Plugin_API* capture_api_;
    const OpenCV_VideoIO_Writer_Plugin_API* writer_api_;
    const OpenCV_VideoIO_Plugin_API_preview* plugin_api_;  //!< deprecated

    PluginBackend(const Ptr<cv::plugin::impl::DynamicLib>& lib)
        : lib_(lib)
        , capture_api_(NULL), writer_api_(NULL)
        , plugin_api_(NULL)
    {
        initCaptureAPI();
        initWriterAPI();
        if (capture_api_ == NULL && writer_api_ == NULL)
        {
            initPluginLegacyAPI();
        }
    }

    Ptr<IVideoCapture> createCapture(int camera) const;
    Ptr<IVideoCapture> createCapture(int camera, const VideoCaptureParameters& params) const CV_OVERRIDE;
    Ptr<IVideoCapture> createCapture(const std::string &filename) const;
    Ptr<IVideoCapture> createCapture(const std::string &filename, const VideoCaptureParameters& params) const CV_OVERRIDE;
    Ptr<IVideoCapture> createCapture(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params) const CV_OVERRIDE;
    Ptr<IVideoWriter> createWriter(const std::string& filename, int fourcc, double fps,
                                   const cv::Size& sz, const VideoWriterParameters& params) const CV_OVERRIDE;

    std::string getCapturePluginVersion(CV_OUT int& version_ABI, CV_OUT int& version_API)
    {
        CV_Assert(capture_api_ || plugin_api_);
        const OpenCV_API_Header& api_header = capture_api_ ? capture_api_->api_header : plugin_api_->api_header;
        version_ABI = api_header.min_api_version;
        version_API = api_header.api_version;
        return api_header.api_description;
    }

    std::string getWriterPluginVersion(CV_OUT int& version_ABI, CV_OUT int& version_API)
    {
        CV_Assert(writer_api_ || plugin_api_);
        const OpenCV_API_Header& api_header = writer_api_ ? writer_api_->api_header : plugin_api_->api_header;
        version_ABI = api_header.min_api_version;
        version_API = api_header.api_version;
        return api_header.api_description;
    }
};

class PluginBackendFactory : public IBackendFactory
{
public:
    VideoCaptureAPIs id_;
    const char* baseName_;
    Ptr<PluginBackend> backend;
    bool initialized;
public:
    PluginBackendFactory(VideoCaptureAPIs id, const char* baseName) :
        id_(id), baseName_(baseName),
        initialized(false)
    {
        // nothing, plugins are loaded on demand
    }

    Ptr<IBackend> getBackend() const CV_OVERRIDE
    {
        initBackend();
        return backend.staticCast<IBackend>();
    }

    bool isBuiltIn() const CV_OVERRIDE { return false; }

    std::string getCapturePluginVersion(
            CV_OUT int& version_ABI,
            CV_OUT int& version_API) const
    {
        initBackend();
        if (!backend)
            CV_Error_(Error::StsNotImplemented, ("Backend '%s' is not available", baseName_));
        return backend->getCapturePluginVersion(version_ABI, version_API);
    }

    std::string getWriterPluginVersion(
            CV_OUT int& version_ABI,
            CV_OUT int& version_API) const
    {
        initBackend();
        if (!backend)
            CV_Error_(Error::StsNotImplemented, ("Backend '%s' is not available", baseName_));
        return backend->getWriterPluginVersion(version_ABI, version_API);
    }

protected:
    inline void initBackend() const
    {
        if (!initialized)
        {
            const_cast<PluginBackendFactory*>(this)->initBackend_();
        }
    }
    void initBackend_()
    {
        AutoLock lock(getInitializationMutex());
        try {
            if (!initialized)
                loadPlugin();
        }
        catch (...)
        {
            CV_LOG_INFO(NULL, "Video I/O: exception during plugin loading: " << baseName_ << ". SKIP");
        }
        initialized = true;
    }
    void loadPlugin();
};

static
std::vector<FileSystemPath_t> getPluginCandidates(const std::string& baseName)
{
    using namespace cv::utils;
    using namespace cv::utils::fs;
    const std::string baseName_l = toLowerCase(baseName);
    const std::string baseName_u = toUpperCase(baseName);
    const FileSystemPath_t baseName_l_fs = toFileSystemPath(baseName_l);
    std::vector<FileSystemPath_t> paths;
    const std::vector<std::string> paths_ = getConfigurationParameterPaths("OPENCV_VIDEOIO_PLUGIN_PATH", std::vector<std::string>());
    if (paths_.size() != 0)
    {
        for (size_t i = 0; i < paths_.size(); i++)
        {
            paths.push_back(toFileSystemPath(paths_[i]));
        }
    }
    else
    {
        FileSystemPath_t binaryLocation;
        if (getBinLocation(binaryLocation))
        {
            binaryLocation = getParent(binaryLocation);
#ifndef CV_VIDEOIO_PLUGIN_SUBDIRECTORY
            paths.push_back(binaryLocation);
#else
            paths.push_back(binaryLocation + toFileSystemPath("/") + toFileSystemPath(CV_VIDEOIO_PLUGIN_SUBDIRECTORY_STR));
#endif
        }
    }
    const std::string default_expr = libraryPrefix() + "opencv_videoio_" + baseName_l + "*" + librarySuffix();
    const std::string plugin_expr = getConfigurationParameterString((std::string("OPENCV_VIDEOIO_PLUGIN_") + baseName_u).c_str(), default_expr.c_str());
    std::vector<FileSystemPath_t> results;
#ifdef _WIN32
    FileSystemPath_t moduleName = toFileSystemPath(libraryPrefix() + "opencv_videoio_" + baseName_l + librarySuffix());
    if (baseName_u == "FFMPEG")  // backward compatibility
    {
        const std::string ffmpeg_env_path = cv::utils::getConfigurationParameterString("OPENCV_FFMPEG_DLL_DIR");
        if (!ffmpeg_env_path.empty())
        {
            results.push_back(toFileSystemPath(ffmpeg_env_path + "\\" + toPrintablePath(moduleName)));
        }
    }
    if (plugin_expr != default_expr)
    {
        moduleName = toFileSystemPath(plugin_expr);
        results.push_back(moduleName);
    }
    for (const FileSystemPath_t& path : paths)
    {
        results.push_back(path + L"\\" + moduleName);
    }
    results.push_back(moduleName);
#if defined(_DEBUG) && defined(DEBUG_POSTFIX)
    if (baseName_u == "FFMPEG")  // backward compatibility
    {
        const FileSystemPath_t templ = toFileSystemPath(CVAUX_STR(DEBUG_POSTFIX) ".dll");
        FileSystemPath_t nonDebugName(moduleName);
        size_t suf = nonDebugName.rfind(templ);
        if (suf != FileSystemPath_t::npos)
        {
            nonDebugName.replace(suf, suf + templ.size(), L".dll");
            results.push_back(nonDebugName);
        }
    }
#endif // _DEBUG && DEBUG_POSTFIX
#else
    CV_LOG_INFO(NULL, "VideoIO plugin (" << baseName << "): glob is '" << plugin_expr << "', " << paths.size() << " location(s)");
    for (const std::string& path : paths)
    {
        if (path.empty())
            continue;
        std::vector<std::string> candidates;
        cv::glob(utils::fs::join(path, plugin_expr), candidates);
        // Prefer candisates with higher versions
        // TODO: implemented accurate versions-based comparator
        std::sort(candidates.begin(), candidates.end(), std::greater<std::string>());
        CV_LOG_INFO(NULL, "    - " << path << ": " << candidates.size());
        copy(candidates.begin(), candidates.end(), back_inserter(results));
    }
#endif
    CV_LOG_INFO(NULL, "Found " << results.size() << " plugin(s) for " << baseName);
    return results;
}

void PluginBackendFactory::loadPlugin()
{
    for (const FileSystemPath_t& plugin : getPluginCandidates(baseName_))
    {
        auto lib = makePtr<cv::plugin::impl::DynamicLib>(plugin);
        if (!lib->isLoaded())
            continue;
        try
        {
            Ptr<PluginBackend> pluginBackend = makePtr<PluginBackend>(lib);
            if (!pluginBackend)
                return;
            if (pluginBackend->capture_api_)
            {
                if (pluginBackend->capture_api_->v0.id != id_)
                {
                    CV_LOG_ERROR(NULL, "Video I/O: plugin '" << pluginBackend->capture_api_->api_header.api_description <<
                                       "': unexpected backend ID: " <<
                                       pluginBackend->capture_api_->v0.id << " vs " << (int)id_ << " (expected)");
                    return;
                }
            }
            if (pluginBackend->writer_api_)
            {
                if (pluginBackend->writer_api_->v0.id != id_)
                {
                    CV_LOG_ERROR(NULL, "Video I/O: plugin '" << pluginBackend->writer_api_->api_header.api_description <<
                                       "': unexpected backend ID: " <<
                                       pluginBackend->writer_api_->v0.id << " vs " << (int)id_ << " (expected)");
                    return;
                }
            }
            if (pluginBackend->plugin_api_)
            {
                if (pluginBackend->plugin_api_->v0.captureAPI != id_)
                {
                    CV_LOG_ERROR(NULL, "Video I/O: plugin '" << pluginBackend->plugin_api_->api_header.api_description <<
                                       "': unexpected backend ID: " <<
                                       pluginBackend->plugin_api_->v0.captureAPI << " vs " << (int)id_ << " (expected)");
                    return;
                }
            }
            if (pluginBackend->capture_api_ == NULL && pluginBackend->writer_api_ == NULL
                && pluginBackend->plugin_api_ == NULL)
            {
                CV_LOG_ERROR(NULL, "Video I/O: no compatible plugin API for backend ID: " << (int)id_);
                return;
            }
            backend = pluginBackend;
            return;
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "Video I/O: exception during plugin initialization: " << toPrintablePath(plugin) << ". SKIP");
        }
    }
}


//==================================================================================================

class PluginCapture : public cv::IVideoCapture
{
    const OpenCV_VideoIO_Capture_Plugin_API* plugin_api_;
    CvPluginCapture capture_;
    Ptr<IStreamReader> readStream_;

public:
    static
    Ptr<PluginCapture> create(const OpenCV_VideoIO_Capture_Plugin_API* plugin_api,
            const std::string &filename, const Ptr<IStreamReader>& stream, int camera, const VideoCaptureParameters& params)
    {
        CV_Assert(plugin_api);
        CV_Assert(plugin_api->v0.Capture_release);

        CvPluginCapture capture = NULL;
        if (stream && plugin_api->api_header.api_version >= 2 && plugin_api->v2.Capture_open_stream)
        {
            std::vector<int> vint_params = params.getIntVector();
            int* c_params = vint_params.data();
            unsigned n_params = (unsigned)(vint_params.size() / 2);

            if (CV_ERROR_OK == plugin_api->v2.Capture_open_stream(
                stream.get(),
                [](void* opaque, char* buffer, long long size) -> long long {
                    CV_LOG_VERBOSE(NULL, 0, "IStreamReader::read(" << size << ")...");
                    auto is = reinterpret_cast<IStreamReader*>(opaque);
                    try {
                        return is->read(buffer, size);
                    } catch (...) {
                        CV_LOG_WARNING(NULL, "IStreamReader::read(" << size << ") failed");
                        return 0;
                    }
                },
                [](void* opaque, long long offset, int way) -> long long {
                    CV_LOG_VERBOSE(NULL, 0, "IStreamReader::seek(" << offset << ", way=" << way << ")...");
                    auto is = reinterpret_cast<IStreamReader*>(opaque);
                    try {
                        return is->seek(offset, way);
                    } catch (...) {
                        CV_LOG_WARNING(NULL, "IStreamReader::seek(" << offset << ", way=" << way << ") failed");
                        return -1;
                    }
                }, c_params, n_params, &capture))
            {
                CV_Assert(capture);
                return makePtr<PluginCapture>(plugin_api, capture, stream);
            }
        }
        else if (stream)
            return Ptr<PluginCapture>();

        if (plugin_api->api_header.api_version >= 1 && plugin_api->v1.Capture_open_with_params)
        {
            std::vector<int> vint_params = params.getIntVector();
            int* c_params = vint_params.data();
            unsigned n_params = (unsigned)(vint_params.size() / 2);

            if (CV_ERROR_OK == plugin_api->v1.Capture_open_with_params(
                    filename.empty() ? 0 : filename.c_str(), camera, c_params, n_params, &capture))
            {
                CV_Assert(capture);
                return makePtr<PluginCapture>(plugin_api, capture);
            }
        }
        else if (plugin_api->v0.Capture_open)
        {
            if (CV_ERROR_OK == plugin_api->v0.Capture_open(filename.empty() ? 0 : filename.c_str(), camera, &capture))
            {
                CV_Assert(capture);
                Ptr<PluginCapture> cap = makePtr<PluginCapture>(plugin_api, capture);
                if (cap && !params.empty())
                {
                    applyParametersFallback(cap, params);
                }
                return cap;
            }
        }

        return Ptr<PluginCapture>();
    }

    PluginCapture(const OpenCV_VideoIO_Capture_Plugin_API* plugin_api, CvPluginCapture capture, const Ptr<IStreamReader>& readStream = Ptr<IStreamReader>())
        : plugin_api_(plugin_api), capture_(capture), readStream_(readStream)
    {
        CV_Assert(plugin_api_); CV_Assert(capture_);
    }

    ~PluginCapture()
    {
        CV_DbgAssert(plugin_api_->v0.Capture_release);
        if (CV_ERROR_OK != plugin_api_->v0.Capture_release(capture_))
            CV_LOG_ERROR(NULL, "Video I/O: Can't release capture by plugin '" << plugin_api_->api_header.api_description << "'");
        capture_ = NULL;
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        double val = -1;
        if (plugin_api_->v0.Capture_getProperty)
            if (CV_ERROR_OK != plugin_api_->v0.Capture_getProperty(capture_, prop, &val))
                val = -1;
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        if (plugin_api_->v0.Capture_setProperty)
            if (CV_ERROR_OK == plugin_api_->v0.Capture_setProperty(capture_, prop, val))
                return true;
        return false;
    }
    bool grabFrame() CV_OVERRIDE
    {
        if (plugin_api_->v0.Capture_grab)
            if (CV_ERROR_OK == plugin_api_->v0.Capture_grab(capture_))
                return true;
        return false;
    }
    static CvResult CV_API_CALL retrieve_callback(int stream_idx, const unsigned char* data, int step, int width, int height, int type, void* userdata)
    {
        CV_UNUSED(stream_idx);
        cv::_OutputArray* dst = static_cast<cv::_OutputArray*>(userdata);
        if (!dst)
            return CV_ERROR_FAIL;
        int depth = CV_MAT_DEPTH(type);
        // [TODO] Remove this condition after rebuilding plugins or add a new
        // version of plugins. Convert type from the old one to the new one (5 bits)
        if (depth > 7) {
            type = CV_MAKETYPE((type & 7), (type >> 3) + 1);
        }
        cv::Mat(cv::Size(width, height), type, (void*)data, step).copyTo(*dst);
        return CV_ERROR_OK;
    }
    bool retrieveFrame(int idx, cv::OutputArray img) CV_OVERRIDE
    {
        bool res = false;
        if (plugin_api_->v0.Capture_retreive)
            if (CV_ERROR_OK == plugin_api_->v0.Capture_retreive(capture_, idx, retrieve_callback, (cv::_OutputArray*)&img))
                res = true;
        return res;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return capture_ != NULL;  // TODO always true
    }
    int getCaptureDomain() CV_OVERRIDE
    {
        return plugin_api_->v0.id;
    }
};


//==================================================================================================

class PluginWriter : public cv::IVideoWriter
{
    const OpenCV_VideoIO_Writer_Plugin_API* plugin_api_;
    CvPluginWriter writer_;

public:
    static
    Ptr<PluginWriter> create(const OpenCV_VideoIO_Writer_Plugin_API* plugin_api,
            const std::string& filename, int fourcc, double fps, const cv::Size& sz,
            const VideoWriterParameters& params)
    {
        CV_Assert(plugin_api);
        CV_Assert(plugin_api->v0.Writer_release);
        CV_Assert(!filename.empty());

        CvPluginWriter writer = NULL;

        if (plugin_api->api_header.api_version >= 1 && plugin_api->v1.Writer_open_with_params)
        {
            std::vector<int> vint_params = params.getIntVector();
            int* c_params = &vint_params[0];
            unsigned n_params = (unsigned)(vint_params.size() / 2);

            if (CV_ERROR_OK == plugin_api->v1.Writer_open_with_params(filename.c_str(), fourcc, fps, sz.width, sz.height, c_params, n_params, &writer))
            {
                CV_Assert(writer);
                return makePtr<PluginWriter>(plugin_api, writer);
            }
        }
        else if (plugin_api->v0.Writer_open)
        {
            const bool isColor = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
            const int depth = params.get(VIDEOWRITER_PROP_DEPTH, CV_8U);
            if (depth != CV_8U)
            {
                CV_LOG_WARNING(NULL, "Video I/O plugin doesn't support (due to lower API level) creation of VideoWriter with depth != CV_8U");
                return Ptr<PluginWriter>();
            }
            if (params.warnUnusedParameters())
            {
                CV_LOG_ERROR(NULL, "VIDEOIO: unsupported parameters in VideoWriter, see logger INFO channel for details");
                return Ptr<PluginWriter>();
            }
            if (CV_ERROR_OK == plugin_api->v0.Writer_open(filename.c_str(), fourcc, fps, sz.width, sz.height, isColor, &writer))
            {
                CV_Assert(writer);
                return makePtr<PluginWriter>(plugin_api, writer);
            }
        }

        return Ptr<PluginWriter>();
    }

    PluginWriter(const OpenCV_VideoIO_Writer_Plugin_API* plugin_api, CvPluginWriter writer)
        : plugin_api_(plugin_api), writer_(writer)
    {
        CV_Assert(plugin_api_); CV_Assert(writer_);
    }

    ~PluginWriter()
    {
        CV_DbgAssert(plugin_api_->v0.Writer_release);
        if (CV_ERROR_OK != plugin_api_->v0.Writer_release(writer_))
            CV_LOG_ERROR(NULL, "Video I/O: Can't release writer by plugin '" << plugin_api_->api_header.api_description << "'");
        writer_ = NULL;
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        double val = -1;
        if (plugin_api_->v0.Writer_getProperty)
            if (CV_ERROR_OK != plugin_api_->v0.Writer_getProperty(writer_, prop, &val))
                val = -1;
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        if (plugin_api_->v0.Writer_setProperty)
            if (CV_ERROR_OK == plugin_api_->v0.Writer_setProperty(writer_, prop, val))
                return true;
        return false;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return writer_ != NULL;  // TODO always true
    }
    void write(cv::InputArray arr) CV_OVERRIDE
    {
        cv::Mat img = arr.getMat();
        CV_DbgAssert(writer_);
        CV_Assert(plugin_api_->v0.Writer_write);
        if (CV_ERROR_OK != plugin_api_->v0.Writer_write(writer_, img.data, (int)img.step[0], img.cols, img.rows, img.channels()))
        {
            CV_LOG_DEBUG(NULL, "Video I/O: Can't write frame by plugin '" << plugin_api_->api_header.api_description << "'");
        }
        // TODO return bool result?
    }
    int getCaptureDomain() const CV_OVERRIDE
    {
        return plugin_api_->v0.id;
    }
};


Ptr<IVideoCapture> PluginBackend::createCapture(int camera, const VideoCaptureParameters& params) const
{
    try
    {
        if (capture_api_)
            return PluginCapture::create(capture_api_, std::string(), nullptr, camera, params); //.staticCast<IVideoCapture>();
        if (plugin_api_)
        {
            Ptr<IVideoCapture> cap = legacy::PluginCapture::create(plugin_api_, std::string(), camera); //.staticCast<IVideoCapture>();
            if (cap && !params.empty())
            {
                applyParametersFallback(cap, params);
            }
            return cap;
        }
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't create camera capture: " << camera);
        throw;
    }
    return Ptr<IVideoCapture>();
}

Ptr<IVideoCapture> PluginBackend::createCapture(const std::string &filename, const VideoCaptureParameters& params) const
{
    try
    {
        if (capture_api_)
            return PluginCapture::create(capture_api_, filename, nullptr, 0, params); //.staticCast<IVideoCapture>();
        if (plugin_api_)
        {
            Ptr<IVideoCapture> cap = legacy::PluginCapture::create(plugin_api_, filename, 0); //.staticCast<IVideoCapture>();
            if (cap && !params.empty())
            {
                applyParametersFallback(cap, params);
            }
            return cap;
        }
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't open file capture: " << filename);
        throw;
    }
    return Ptr<IVideoCapture>();
}

Ptr<IVideoCapture> PluginBackend::createCapture(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params) const
{
    try
    {
        if (capture_api_)
            return PluginCapture::create(capture_api_, std::string(), stream, 0, params); //.staticCast<IVideoCapture>();
        if (plugin_api_)
        {
            CV_Error(Error::StsNotImplemented, "Legacy plugin API for stream capture");
        }
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't open stream capture");
        throw;
    }
    return Ptr<IVideoCapture>();
}

Ptr<IVideoWriter> PluginBackend::createWriter(const std::string& filename, int fourcc, double fps,
                                              const cv::Size& sz, const VideoWriterParameters& params) const
{
    try
    {
        if (writer_api_)
            return PluginWriter::create(writer_api_, filename, fourcc, fps, sz, params); //.staticCast<IVideoWriter>();
        if (plugin_api_)
            return legacy::PluginWriter::create(plugin_api_, filename, fourcc, fps, sz, params); //.staticCast<IVideoWriter>();
    }
    catch (...)
    {
        CV_LOG_DEBUG(NULL, "Video I/O: can't open writer: " << filename);
    }
    return Ptr<IVideoWriter>();
}

#endif  // OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)

}  // namespace

Ptr<IBackendFactory> createPluginBackendFactory(VideoCaptureAPIs id, const char* baseName)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
    return makePtr<impl::PluginBackendFactory>(id, baseName); //.staticCast<IBackendFactory>();
#else
    CV_UNUSED(id);
    CV_UNUSED(baseName);
    return Ptr<IBackendFactory>();
#endif
}


std::string getCapturePluginVersion(
    const Ptr<IBackendFactory>& backend_factory,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
    using namespace impl;
    CV_Assert(backend_factory);
    PluginBackendFactory* plugin_backend_factory = dynamic_cast<PluginBackendFactory*>(backend_factory.get());
    CV_Assert(plugin_backend_factory);
    return plugin_backend_factory->getCapturePluginVersion(version_ABI, version_API);
#else
    CV_UNUSED(backend_factory);
    CV_UNUSED(version_ABI);
    CV_UNUSED(version_API);
    CV_Error(Error::StsBadFunc, "Plugins are not available in this build");
#endif
}

std::string getWriterPluginVersion(
    const Ptr<IBackendFactory>& backend_factory,
    CV_OUT int& version_ABI,
    CV_OUT int& version_API
)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT && defined(ENABLE_PLUGINS)
    using namespace impl;
    CV_Assert(backend_factory);
    PluginBackendFactory* plugin_backend_factory = dynamic_cast<PluginBackendFactory*>(backend_factory.get());
    CV_Assert(plugin_backend_factory);
    return plugin_backend_factory->getWriterPluginVersion(version_ABI, version_API);
#else
    CV_UNUSED(backend_factory);
    CV_UNUSED(version_ABI);
    CV_UNUSED(version_API);
    CV_Error(Error::StsBadFunc, "Plugins are not available in this build");
#endif
}

}  // namespace
