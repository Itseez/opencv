// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CAP_INTERFACE_HPP
#define CAP_INTERFACE_HPP

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"

//===================================================

// Modern classes

namespace cv
{
namespace
{
template <class T>
inline T castParameterTo(int paramValue)
{
    return static_cast<T>(paramValue);
}

template <>
inline bool castParameterTo(int paramValue)
{
    return paramValue != 0;
}
}

class VideoParameters
{
public:
    struct VideoParameter {
        VideoParameter() = default;

        VideoParameter(int key_, int value_) : key(key_), value(value_) {}

        int key{-1};
        int value{-1};
        mutable bool isConsumed{false};
    };

    VideoParameters() = default;

    explicit VideoParameters(const std::vector<int>& params)
    {
        const auto count = params.size();
        if (count % 2 != 0)
        {
            CV_Error_(Error::StsVecLengthErr,
                      ("Vector of VideoWriter parameters should have even length"));
        }
        params_.reserve(count / 2);
        for (std::size_t i = 0; i < count; i += 2)
        {
            add(params[i], params[i + 1]);
        }
    }

    VideoParameters(int* params, unsigned n_params)
    {
        params_.reserve(n_params);
        for (unsigned i = 0; i < n_params; ++i)
        {
            add(params[2*i], params[2*i + 1]);
        }
    }

    void add(int key, int value)
    {
        params_.emplace_back(key, value);
    }

    bool has(int key) const
    {
        auto it = std::find_if(params_.begin(), params_.end(),
            [key](const VideoParameter &param)
            {
                return param.key == key;
            }
        );
        return it != params_.end();
    }

    template <class ValueType>
    ValueType get(int key) const
    {
        auto it = std::find_if(params_.begin(), params_.end(),
            [key](const VideoParameter &param)
            {
                return param.key == key;
            }
        );
        if (it != params_.end())
        {
            it->isConsumed = true;
            return castParameterTo<ValueType>(it->value);
        }
        else
        {
            CV_Error_(Error::StsBadArg, ("Missing value for parameter: [%d]", key));
        }
    }

    template <class ValueType>
    ValueType get(int key, ValueType defaultValue) const
    {
        auto it = std::find_if(params_.begin(), params_.end(),
            [key](const VideoParameter &param)
            {
                return param.key == key;
            }
        );
        if (it != params_.end())
        {
            it->isConsumed = true;
            return castParameterTo<ValueType>(it->value);
        }
        else
        {
            return defaultValue;
        }
    }

    std::vector<int> getUnused() const
    {
        std::vector<int> unusedParams;
        for (const auto &param : params_)
        {
            if (!param.isConsumed)
            {
                unusedParams.push_back(param.key);
            }
        }
        return unusedParams;
    }

    std::vector<int> getIntVector() const
    {
        std::vector<int> vint_params;
        for (const auto& param : params_)
        {
            vint_params.push_back(param.key);
            vint_params.push_back(param.value);
        }
        return vint_params;
    }

    bool empty() const
    {
        return params_.empty();
    }

    bool warnUnusedParameters() const
    {
        bool found = false;
        for (const auto &param : params_)
        {
            if (!param.isConsumed)
            {
                found = true;
                CV_LOG_INFO(NULL, "VIDEOIO: unused parameter: [" << param.key << "]=" <<
                    cv::format("%lld / 0x%016llx", (long long)param.value, (long long)param.value));
            }
        }
        return found;
    }


private:
    std::vector<VideoParameter> params_;
};

class VideoWriterParameters : public VideoParameters
{
public:
    using VideoParameters::VideoParameters;  // reuse constructors
};

class VideoCaptureParameters : public VideoParameters
{
public:
    using VideoParameters::VideoParameters;  // reuse constructors
};

class IVideoCapture
{
public:
    virtual ~IVideoCapture() {}
    virtual double getProperty(int) const { return 0; }
    virtual bool setProperty(int, double) { return false; }
    virtual bool grabFrame() = 0;
    virtual bool retrieveFrame(int, OutputArray) = 0;
    virtual bool isOpened() const = 0;
    virtual int getCaptureDomain() { return CAP_ANY; } // Return the type of the capture object: CAP_DSHOW, etc...
};

class IVideoWriter
{
public:
    virtual ~IVideoWriter() {}
    virtual double getProperty(int) const { return 0; }
    virtual bool setProperty(int, double) { return false; }
    virtual bool isOpened() const = 0;
    virtual void write(InputArray) = 0;
    virtual int getCaptureDomain() const { return cv::CAP_ANY; } // Return the type of the capture object: CAP_FFMPEG, etc...
};

namespace internal {
class VideoCapturePrivateAccessor
{
public:
    static
    IVideoCapture* getIVideoCapture(const VideoCapture& cap) { return cap.icap.get(); }
};
} // namespace


// Advanced base class for VideoCapture backends providing some extra functionality
class VideoCaptureBase : public IVideoCapture
{
public:
    VideoCaptureBase() : autorotate(false) {}
    double getProperty(int propId) const CV_OVERRIDE
    {
        switch(propId)
        {
            case cv::CAP_PROP_ORIENTATION_AUTO:
                return static_cast<double>(autorotate);

            case cv::CAP_PROP_FRAME_WIDTH:
                return shouldSwapWidthHeight() ? getProperty_(cv::CAP_PROP_FRAME_HEIGHT) : getProperty_(cv::CAP_PROP_FRAME_WIDTH);

            case cv::CAP_PROP_FRAME_HEIGHT:
                return shouldSwapWidthHeight() ? getProperty_(cv::CAP_PROP_FRAME_WIDTH) : getProperty_(cv::CAP_PROP_FRAME_HEIGHT);

            default:
                return getProperty_(propId);
        }
    }
    bool setProperty(int propId, double value) CV_OVERRIDE
    {
        switch(propId)
        {
            case cv::CAP_PROP_ORIENTATION_AUTO:
                autorotate = (value != 0);
                return true;

            default:
                return setProperty_(propId, value);
        }
    }
    bool retrieveFrame(int channel, OutputArray image) CV_OVERRIDE
    {
        const bool res = retrieveFrame_(channel, image);
        if (res)
            applyMetadataRotation(image);
        return res;
    }

protected:
    virtual double getProperty_(int) const = 0;
    virtual bool setProperty_(int, double) = 0;
    virtual bool retrieveFrame_(int, OutputArray) = 0;

protected:
    bool shouldSwapWidthHeight() const
    {
        if (!autorotate)
            return false;
        int rotation = static_cast<int>(getProperty(cv::CAP_PROP_ORIENTATION_META));
        return std::abs(rotation % 180) == 90;
    }
    void applyMetadataRotation(OutputArray mat) const
    {
        bool rotation_auto = 0 != getProperty(CAP_PROP_ORIENTATION_AUTO);
        int rotation_angle = static_cast<int>(getProperty(CAP_PROP_ORIENTATION_META));
        if(!rotation_auto || rotation_angle%360 == 0)
        {
            return;
        }
        cv::RotateFlags flag;
        if(rotation_angle == 90 || rotation_angle == -270) { // Rotate clockwise 90 degrees
            flag = cv::ROTATE_90_CLOCKWISE;
        } else if(rotation_angle == 270 || rotation_angle == -90) { // Rotate clockwise 270 degrees
            flag = cv::ROTATE_90_COUNTERCLOCKWISE;
        } else if(rotation_angle == 180 || rotation_angle == -180) { // Rotate clockwise 180 degrees
            flag = cv::ROTATE_180;
        } else { // Unsupported rotation
            return;
        }
        cv::rotate(mat, mat, flag);
    }

protected:
    bool autorotate;
};


//==================================================================================================

Ptr<IVideoCapture> cvCreateFileCapture_FFMPEG_proxy(const std::string &filename, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateStreamCapture_FFMPEG_proxy(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params);
Ptr<IVideoWriter> cvCreateVideoWriter_FFMPEG_proxy(const std::string& filename, int fourcc,
                                                   double fps, const Size& frameSize,
                                                   const VideoWriterParameters& params);

Ptr<IVideoCapture> createGStreamerCapture_file(const std::string& filename, const cv::VideoCaptureParameters& params);
Ptr<IVideoCapture> createGStreamerCapture_cam(int index, const cv::VideoCaptureParameters& params);
Ptr<IVideoWriter> create_GStreamer_writer(const std::string& filename, int fourcc,
                                          double fps, const Size& frameSize,
                                          const VideoWriterParameters& params);

Ptr<IVideoCapture> create_MFX_capture(const std::string &filename);
Ptr<IVideoWriter> create_MFX_writer(const std::string& filename, int _fourcc,
                                    double fps, const Size& frameSize,
                                    const VideoWriterParameters& params);

Ptr<IVideoCapture> create_AVFoundation_capture_file(const std::string &filename);
Ptr<IVideoCapture> create_AVFoundation_capture_cam(int index);
Ptr<IVideoWriter> create_AVFoundation_writer(const std::string& filename, int fourcc,
                                             double fps, const Size& frameSize,
                                             const VideoWriterParameters& params);

Ptr<IVideoCapture> create_WRT_capture(int device);

Ptr<IVideoCapture> cvCreateCapture_MSMF(int index, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateCapture_MSMF(const std::string& filename, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateCapture_MSMF(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params);
Ptr<IVideoWriter> cvCreateVideoWriter_MSMF(const std::string& filename, int fourcc,
                                           double fps, const Size& frameSize,
                                           const VideoWriterParameters& params);

Ptr<IVideoCapture> create_DShow_capture(int index);

Ptr<IVideoCapture> create_V4L_capture_cam(int index);
Ptr<IVideoCapture> create_V4L_capture_file(const std::string &filename);

Ptr<IVideoCapture> create_OpenNI2_capture_cam( int index );
Ptr<IVideoCapture> create_OpenNI2_capture_file( const std::string &filename );

Ptr<IVideoCapture> create_Images_capture(const std::string &filename);
Ptr<IVideoWriter> create_Images_writer(const std::string& filename, int fourcc,
                                       double fps, const Size& frameSize,
                                       const VideoWriterParameters& params);

Ptr<IVideoCapture> create_DC1394_capture(int index);

Ptr<IVideoCapture> create_RealSense_capture(int index);

Ptr<IVideoCapture> create_PvAPI_capture( int index );

Ptr<IVideoCapture> create_XIMEA_capture_cam( int index );
Ptr<IVideoCapture> create_XIMEA_capture_file( const std::string &serialNumber );

Ptr<IVideoCapture> create_ueye_camera(int camera);

Ptr<IVideoCapture> create_Aravis_capture( int index );

Ptr<IVideoCapture> createMotionJpegCapture(const std::string& filename);
Ptr<IVideoWriter> createMotionJpegWriter(const std::string& filename, int fourcc,
                                         double fps, const Size& frameSize,
                                         const VideoWriterParameters& params);

Ptr<IVideoCapture> createGPhoto2Capture(int index);
Ptr<IVideoCapture> createGPhoto2Capture(const std::string& deviceName);

Ptr<IVideoCapture> createXINECapture(const std::string &filename);

Ptr<IVideoCapture> createAndroidCapture_cam(int index, const VideoCaptureParameters& params);
Ptr<IVideoCapture> createAndroidCapture_file(const std::string &filename, const VideoCaptureParameters& params);
Ptr<IVideoWriter> createAndroidVideoWriter(const std::string& filename, int fourcc,
                                           double fps, const Size& frameSize,
                                           const VideoWriterParameters& params);

Ptr<IVideoCapture> create_obsensor_capture(int index);

bool VideoCapture_V4L_waitAny(
        const std::vector<VideoCapture>& streams,
        CV_OUT std::vector<int>& ready,
        int64 timeoutNs);

static inline
std::ostream& operator<<(std::ostream& out, const VideoAccelerationType& va_type)
{
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: out << "NONE"; return out;
    case VIDEO_ACCELERATION_ANY: out << "ANY"; return out;
    case VIDEO_ACCELERATION_D3D11: out << "D3D11"; return out;
    case VIDEO_ACCELERATION_VAAPI: out << "VAAPI"; return out;
    case VIDEO_ACCELERATION_MFX: out << "MFX"; return out;
    }
    out << cv::format("UNKNOWN(0x%ux)", static_cast<unsigned int>(va_type));
    return out;
}

} // cv::

#endif // CAP_INTERFACE_HPP
