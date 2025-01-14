// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/utils/filesystem.hpp"

namespace opencv_test
{

class Videoio_Test_Base
{
protected:
    string ext;
    string video_file;
    VideoCaptureAPIs apiPref;
protected:
    Videoio_Test_Base() {}
    virtual ~Videoio_Test_Base() {}
    virtual void writeVideo() {}
    virtual void checkFrameContent(Mat &, int) {}
    virtual void checkFrameCount(int &) {}
    void checkFrameRead(int idx, VideoCapture & cap)
    {
        //int frameID = (int)cap.get(CAP_PROP_POS_FRAMES);
        Mat img;
        ASSERT_NO_THROW(cap >> img);
        //std::cout << "idx=" << idx << " img=" << img.size() << " frameID=" << frameID << std::endl;
        ASSERT_FALSE(img.empty()) << "idx=" << idx;
        checkFrameContent(img, idx);
    }
    void checkFrameSeek(int idx, VideoCapture & cap)
    {
        bool canSeek = false;
        ASSERT_NO_THROW(canSeek = cap.set(CAP_PROP_POS_FRAMES, idx));
        if (!canSeek)
        {
            std::cout << "Seek to frame '" << idx << "' is not supported. SKIP." << std::endl;
            return;
        }
        EXPECT_EQ(idx, (int)cap.get(CAP_PROP_POS_FRAMES));
        checkFrameRead(idx, cap);
    }
public:
    void doTest()
    {
        if (!videoio_registry::hasBackend(apiPref))
            throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));
        if (cvtest::skipUnstableTests && apiPref == CAP_MSMF && (ext == "h264" || ext == "h265" || ext == "mpg"))
            throw SkipTestException("Unstable MSMF test");
#ifdef __linux__
        if (cvtest::skipUnstableTests && apiPref == CAP_GSTREAMER &&
            (ext == "avi" || ext == "mkv") &&
            (video_file.find("MPEG") != std::string::npos))
        {
            throw SkipTestException("Unstable GSTREAMER test");
        }
#endif
        writeVideo();
        VideoCapture cap;
        ASSERT_NO_THROW(cap.open(video_file, apiPref));
        if (!cap.isOpened())
        {
            std::cout << "SKIP test: backend " << apiPref << " can't open the video: " << video_file << std::endl;
            return;
        }
        int n_frames = -1;
        EXPECT_NO_THROW(n_frames = (int)cap.get(CAP_PROP_FRAME_COUNT));
        if (n_frames > 0)
        {
            ASSERT_GT(n_frames, 0);
            checkFrameCount(n_frames);
        }
        else
        {
            std::cout << "CAP_PROP_FRAME_COUNT is not supported by backend. Assume 50 frames." << std::endl;
            n_frames = 50;
        }
        // GStreamer can't read frame count of big_buck_bunny.wmv
        if (apiPref == CAP_GSTREAMER && ext == "wmv")
        {
            n_frames = 125;
        }

        {
            SCOPED_TRACE("consecutive read");
            if (apiPref == CAP_GSTREAMER)
            {
                // This workaround is for GStreamer 1.3.1.1 and older.
                // Old Gstreamer has a bug which handles the total duration 1 frame shorter
                // Old Gstreamer are used in Ubuntu 14.04, so the following code could be removed after it's EOL
                n_frames--;
            }
            for (int k = 0; k < n_frames; ++k)
            {
                checkFrameRead(k, cap);
                if (::testing::Test::HasFailure() && k % 10 == 0)
                    break;
            }
        }
        bool canSeek = false;
        EXPECT_NO_THROW(canSeek = cap.set(CAP_PROP_POS_FRAMES, 0));
        if (!canSeek)
        {
            std::cout << "Seek to frame '0' is not supported. SKIP all 'seek' tests." << std::endl;
            return;
        }

        if (ext != "wmv" && ext != "h264" && ext != "h265")
        {
            SCOPED_TRACE("progressive seek");
            bool res = false;
            EXPECT_NO_THROW(res = cap.set(CAP_PROP_POS_FRAMES, 0));
            ASSERT_TRUE(res);
            for (int k = 0; k < n_frames; k += 20)
            {
                checkFrameSeek(k, cap);
                if (::testing::Test::HasFailure() && k % 10 == 0)
                    break;
            }
        }

        if (ext != "mpg" && ext != "wmv" && ext != "h264" && ext != "h265")
        {
            SCOPED_TRACE("random seek");
            bool res = false;
            EXPECT_NO_THROW(res = cap.set(CAP_PROP_POS_FRAMES, 0));
            ASSERT_TRUE(res);
            for (int k = 0; k < 10; ++k)
            {
                checkFrameSeek(cvtest::TS::ptr()->get_rng().uniform(0, n_frames), cap);
                if (::testing::Test::HasFailure() && k % 10 == 0)
                    break;
            }
        }
    }
};

//==================================================================================================
typedef tuple<string, VideoCaptureAPIs> Backend_Type_Params;

class videoio_bunny : public Videoio_Test_Base, public testing::TestWithParam<Backend_Type_Params>
{
    BunnyParameters bunny_param;
public:
    videoio_bunny()
    {
        ext = get<0>(GetParam());
        apiPref = get<1>(GetParam());
        video_file = BunnyParameters::getFilename(String(".") + ext);
    }
    void doFrameCountTest()
    {
        if (!videoio_registry::hasBackend(apiPref))
            throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));
        if (cvtest::skipUnstableTests && apiPref == CAP_MSMF && (ext == "h264" || ext == "h265" || ext == "mpg"))
            throw SkipTestException("Unstable MSMF test");
        VideoCapture cap;
        EXPECT_NO_THROW(cap.open(video_file, apiPref));
        if (!cap.isOpened())
        {
            std::cout << "SKIP test: backend " << apiPref << " can't open the video: " << video_file << std::endl;
            return;
        }

        Size actual;
        EXPECT_NO_THROW(actual = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                                      (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
        EXPECT_EQ(bunny_param.getWidth(), actual.width);
        EXPECT_EQ(bunny_param.getHeight(), actual.height);

        double fps_prop = 0;
        EXPECT_NO_THROW(fps_prop = cap.get(CAP_PROP_FPS));
        if (fps_prop > 0)
            EXPECT_NEAR(fps_prop, bunny_param.getFps(), 1);
        else
            std::cout << "FPS is not available. SKIP check." << std::endl;

        int count_prop = 0;
        EXPECT_NO_THROW(count_prop = (int)cap.get(CAP_PROP_FRAME_COUNT));
        // mpg file reports 5.08 sec * 24 fps => property returns 122 frames
        // but actual number of frames returned is 125
        if (ext != "mpg" && !(apiPref == CAP_GSTREAMER && ext == "wmv"))
        {
            if (count_prop > 0)
            {
                EXPECT_EQ(bunny_param.getCount(), count_prop);
            }
        }

        int count_actual = 0;
        while (cap.isOpened())
        {
            Mat frame;
            EXPECT_NO_THROW(cap >> frame);
            if (frame.empty())
                break;
            EXPECT_EQ(bunny_param.getWidth(), frame.cols);
            EXPECT_EQ(bunny_param.getHeight(), frame.rows);
            count_actual += 1;
            if (::testing::Test::HasFailure() && count_actual % 10 == 0)
                break;
        }
        if (count_prop > 0)
        {
            EXPECT_NEAR(bunny_param.getCount(), count_actual, 1);
        }
        else
            std::cout << "Frames counter is not available. Actual frames: " << count_actual << ". SKIP check." << std::endl;
    }

    void doTimestampTest()
    {
        if (!isBackendAvailable(apiPref, cv::videoio_registry::getStreamBackends()))
            throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));

        if (((apiPref == CAP_GSTREAMER) && (ext == "avi")))
            throw SkipTestException(cv::String("Backend ") +  cv::videoio_registry::getBackendName(apiPref) +
                    cv::String(" does not support CAP_PROP_POS_MSEC option"));

        if (((apiPref == CAP_FFMPEG || apiPref == CAP_GSTREAMER) && ((ext == "h264") || (ext == "h265"))))
            throw SkipTestException(cv::String("Backend ") +  cv::videoio_registry::getBackendName(apiPref) +
                    cv::String(" does not support CAP_PROP_POS_MSEC option"));

        VideoCapture cap;
        EXPECT_NO_THROW(cap.open(video_file, apiPref));
        if (!cap.isOpened())
            throw SkipTestException(cv::String("Backend ") +  cv::videoio_registry::getBackendName(apiPref) +
                    cv::String(" can't open the video: ")  + video_file);

        int frame_count = (int)cap.get(CAP_PROP_FRAME_COUNT);

        // HACK: Video consists of 125 frames, but cv::VideoCapture with FFmpeg reports only 122 frames for mpg video.
        // mpg file reports 5.08 sec * 24 fps => property returns 122 frames,but actual number of frames returned is 125
        // HACK: CAP_PROP_FRAME_COUNT is not supported for vmw + MSMF. Just force check for all 125 frames
        if (ext == "mpg")
            EXPECT_GE(frame_count, 114);
        else if ((ext == "wmv") && (apiPref == CAP_MSMF || apiPref == CAP_GSTREAMER))
            frame_count = 125;
        else
            EXPECT_EQ(frame_count, 125);
        Mat img;

        // HACK: FFmpeg reports picture_pts = AV_NOPTS_VALUE_ for the last frame for AVI container by some reason
        if ((ext == "avi") && (apiPref == CAP_FFMPEG))
            frame_count--;

        for (int i = 0; i < frame_count; i++)
        {
            double timestamp = 0;
            ASSERT_NO_THROW(cap >> img);
            EXPECT_NO_THROW(timestamp = cap.get(CAP_PROP_POS_MSEC));
            if (cvtest::debugLevel > 0)
                std::cout << "i = " << i << ": timestamp = " << timestamp << std::endl;
            const double frame_period = 1000.f/bunny_param.getFps();
            // big_buck_bunny.mpg starts at 0.500 msec
            if ((ext == "mpg") && (apiPref == CAP_GSTREAMER))
                timestamp -= 500.0;
            // NOTE: eps == frame_period, because videoCapture returns frame beginning timestamp or frame end
            // timestamp depending on codec and back-end. So the first frame has timestamp 0 or frame_period.
            EXPECT_NEAR(timestamp, i*frame_period, frame_period) << "i=" << i;
        }
    }
};

//==================================================================================================

struct Ext_Fourcc_PSNR
{
    const char* ext;
    const char* fourcc;
    float PSNR;
    VideoCaptureAPIs api;
};
typedef tuple<Size, Ext_Fourcc_PSNR> Size_Ext_Fourcc_PSNR;

class videoio_synthetic : public Videoio_Test_Base, public testing::TestWithParam<Size_Ext_Fourcc_PSNR>
{
    Size frame_size;
    int fourcc;
    float PSNR_GT;
    int frame_count;
    double fps;
public:
    videoio_synthetic()
    {
        frame_size = get<0>(GetParam());
        const Ext_Fourcc_PSNR p = get<1>(GetParam());
        ext = p.ext;
        fourcc = fourccFromString(p.fourcc);
        PSNR_GT = p.PSNR;
        video_file = cv::tempfile((fourccToString(fourcc) + "." + ext).c_str());
        frame_count = 100;
        fps = 25.;
        apiPref = p.api;
    }
    void TearDown()
    {
        remove(video_file.c_str());
    }
    virtual void writeVideo()
    {
        Mat img(frame_size, CV_8UC3);
        VideoWriter writer;
        EXPECT_NO_THROW(writer.open(video_file, apiPref, fourcc, fps, frame_size, true));
        ASSERT_TRUE(writer.isOpened());
        for(int i = 0; i < frame_count; ++i )
        {
            generateFrame(i, frame_count, img);
            EXPECT_NO_THROW(writer << img);
            if (::testing::Test::HasFailure() && i % 10 == 0)
                break;
        }
        EXPECT_NO_THROW(writer.release());
    }
    virtual void checkFrameContent(Mat & img, int idx)
    {
        Mat imgGT(frame_size, CV_8UC3);
        generateFrame(idx, frame_count, imgGT);
        double psnr = cvtest::PSNR(img, imgGT);
        ASSERT_GT(psnr, PSNR_GT) << "frame " << idx;
    }
    virtual void checkFrameCount(int &actual)
    {
        Range expected_frame_count = Range(frame_count, frame_count);

        // Hack! Newer FFmpeg versions in this combination produce a file
        // whose reported duration is one frame longer than needed, and so
        // the calculated frame count is also off by one. Ideally, we'd want
        // to fix both writing (to produce the correct duration) and reading
        // (to correctly report frame count for such files), but I don't know
        // how to do either, so this is a workaround for now.
        if (fourcc == VideoWriter::fourcc('M', 'P', 'E', 'G') && ext == "mkv")
            expected_frame_count.end += 1;

        // Workaround for some gstreamer pipelines
        if (apiPref == CAP_GSTREAMER)
            expected_frame_count.start -= 1;

        ASSERT_LE(expected_frame_count.start, actual);
        ASSERT_GE(expected_frame_count.end, actual);

        actual = expected_frame_count.start; // adjust actual frame boundary to possible minimum
    }
};

//==================================================================================================

static const VideoCaptureAPIs backend_params[] = {
#ifdef HAVE_AVFOUNDATION
   CAP_AVFOUNDATION,
#endif

#ifdef _WIN32
    CAP_MSMF,
#endif

    CAP_GSTREAMER,
    CAP_FFMPEG,

#ifdef HAVE_XINE
    CAP_XINE,
#endif

    CAP_OPENCV_MJPEG
    // CAP_INTEL_MFX
};

static const string bunny_params[] = {
    string("wmv"),
    string("mov"),
    string("mp4"),
    string("mpg"),
    string("avi"),
    string("h264"),
    string("h265"),
    string("mjpg.avi")
};

TEST_P(videoio_bunny, read_position) { doTest(); }

TEST_P(videoio_bunny, frame_count) { doFrameCountTest(); }

TEST_P(videoio_bunny, frame_timestamp) { doTimestampTest(); }

INSTANTIATE_TEST_CASE_P(videoio, videoio_bunny,
                          testing::Combine(
                              testing::ValuesIn(bunny_params),
                              testing::ValuesIn(backend_params)));


inline static std::ostream &operator<<(std::ostream &out, const Ext_Fourcc_PSNR &p)
{
    out << "FOURCC(" << p.fourcc << "), ." << p.ext << ", " << p.api << ", " << p.PSNR << "dB"; return out;
}

static Ext_Fourcc_PSNR synthetic_params[] = {

#ifdef _WIN32
#if !defined(_M_ARM)
    {"wmv", "WMV1", 30.f, CAP_MSMF},
    {"wmv", "WMV2", 30.f, CAP_MSMF},
#endif
    {"wmv", "WMV3", 30.f, CAP_MSMF},
    {"wmv", "WVC1", 30.f, CAP_MSMF},
    {"mov", "H264", 30.f, CAP_MSMF},
 // {"mov", "HEVC", 30.f, CAP_MSMF},  // excluded due to CI issue: https://github.com/opencv/opencv/pull/23172
#endif

#ifdef HAVE_AVFOUNDATION
   {"mov", "H264", 30.f, CAP_AVFOUNDATION},
   {"mov", "MJPG", 30.f, CAP_AVFOUNDATION},
   {"mp4", "H264", 30.f, CAP_AVFOUNDATION},
   {"mp4", "MJPG", 30.f, CAP_AVFOUNDATION},
   {"m4v", "H264", 30.f, CAP_AVFOUNDATION},
   {"m4v", "MJPG", 30.f, CAP_AVFOUNDATION},
#endif

    {"avi", "XVID", 30.f, CAP_FFMPEG},
    {"avi", "MPEG", 30.f, CAP_FFMPEG},
    {"avi", "IYUV", 30.f, CAP_FFMPEG},
    {"avi", "MJPG", 30.f, CAP_FFMPEG},

    {"mkv", "XVID", 30.f, CAP_FFMPEG},
    {"mkv", "MPEG", 30.f, CAP_FFMPEG},
    {"mkv", "MJPG", 30.f, CAP_FFMPEG},
    {"avi", "FFV1", 30.f, CAP_FFMPEG},
    {"mkv", "FFV1", 30.f, CAP_FFMPEG},

    {"avi", "MPEG", 28.f, CAP_GSTREAMER},
    {"avi", "MJPG", 30.f, CAP_GSTREAMER},
    {"avi", "H264", 30.f, CAP_GSTREAMER},

    {"mkv", "MPEG", 28.f, CAP_GSTREAMER},
    {"mkv", "MJPG", 30.f, CAP_GSTREAMER},
    {"mkv", "H264", 30.f, CAP_GSTREAMER},

    {"avi", "MJPG", 30.f, CAP_OPENCV_MJPEG},
};


Size all_sizes[] = {
    Size(640, 480),
    Size(976, 768)
};

TEST_P(videoio_synthetic, write_read_position) { doTest(); }

INSTANTIATE_TEST_CASE_P(videoio, videoio_synthetic,
                        testing::Combine(
                            testing::ValuesIn(all_sizes),
                            testing::ValuesIn(synthetic_params)));

struct Ext_Fourcc_API
{
    const char* ext;
    const char* fourcc;
    VideoCaptureAPIs api;
};

inline static std::ostream &operator<<(std::ostream &out, const Ext_Fourcc_API &p)
{
    out << "(FOURCC(" << p.fourcc << "), \"" << p.ext << "\", " << p.api << ")"; return out;
}


class Videoio_Writer : public Videoio_Test_Base, public testing::TestWithParam<Ext_Fourcc_API>
{
protected:
    Size frame_size;
    int fourcc;
    double fps;
public:
    Videoio_Writer()
    {
        frame_size = Size(640, 480);
        const Ext_Fourcc_API p = GetParam();
        ext = p.ext;
        fourcc = fourccFromString(p.fourcc);
        if (ext.size() == 3)
            video_file = cv::tempfile((fourccToString(fourcc) + "." + ext).c_str());
        else
            video_file = ext;
        fps = 25.;
        apiPref = p.api;
    }
    void SetUp()
    {
    }
    void TearDown()
    {
        if (ext.size() == 3)
            (void)remove(video_file.c_str());
    }
};

TEST_P(Videoio_Writer, write_nothing)
{
    if (!cv::videoio_registry::hasBackend(apiPref))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));

    VideoWriter writer;
    EXPECT_NO_THROW(writer.open(video_file, apiPref, fourcc, fps, frame_size, true));
    ASSERT_TRUE(writer.isOpened());
#if 0  // no frames
    cv::Mat m(frame_size, CV_8UC3, Scalar::all(127));
    writer << m;
#endif
    EXPECT_NO_THROW(writer.release());
}

static vector<Ext_Fourcc_API> generate_Ext_Fourcc_API()
{
    const size_t N = sizeof(synthetic_params)/sizeof(synthetic_params[0]);
    vector<Ext_Fourcc_API> result; result.reserve(N);
    for (size_t i = 0; i < N; i++)
    {
        const Ext_Fourcc_PSNR& src = synthetic_params[i];
        Ext_Fourcc_API e = { src.ext, src.fourcc, src.api };
        result.push_back(e);
    }

    {
        Ext_Fourcc_API e = { "appsrc ! videoconvert ! video/x-raw, format=(string)NV12 ! filesink location=test.nv12", "\0\0\0\0", CAP_GSTREAMER };
        result.push_back(e);
    }
    {
        Ext_Fourcc_API e = { "appsrc ! videoconvert ! video/x-raw, format=(string)I420 ! matroskamux ! filesink location=test.mkv", "\0\0\0\0", CAP_GSTREAMER };
        result.push_back(e);
    }
    return result;
}

INSTANTIATE_TEST_CASE_P(videoio, Videoio_Writer, testing::ValuesIn(generate_Ext_Fourcc_API()));


TEST(Videoio, exceptions)
{
    VideoCapture cap;

    Mat mat;

    EXPECT_FALSE(cap.grab());
    EXPECT_FALSE(cap.retrieve(mat));
    EXPECT_FALSE(cap.set(CAP_PROP_POS_FRAMES, 1));
    EXPECT_FALSE(cap.open("this_does_not_exist.avi", CAP_OPENCV_MJPEG));

    cap.setExceptionMode(true);

    EXPECT_THROW(cap.grab(), Exception);
    EXPECT_THROW(cap.retrieve(mat), Exception);
    EXPECT_THROW(cap.set(CAP_PROP_POS_FRAMES, 1), Exception);
    EXPECT_THROW(cap.open("this_does_not_exist.avi", CAP_OPENCV_MJPEG), Exception);
}


typedef Videoio_Writer Videoio_Writer_bad_fourcc;

TEST_P(Videoio_Writer_bad_fourcc, nocrash)
{
    if (!isBackendAvailable(apiPref, cv::videoio_registry::getStreamBackends()))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));

    VideoWriter writer;
    EXPECT_NO_THROW(writer.open(video_file, apiPref, fourcc, fps, frame_size, true));
    ASSERT_FALSE(writer.isOpened());
    EXPECT_NO_THROW(writer.release());
}

static vector<Ext_Fourcc_API> generate_Ext_Fourcc_API_nocrash()
{
    static const Ext_Fourcc_API params[] = {
#ifdef HAVE_MSMF_DISABLED  // MSMF opens writer stream
    {"wmv", "aaaa", CAP_MSMF},
    {"mov", "aaaa", CAP_MSMF},
#endif

#ifdef HAVE_AVFOUNDATION
   {"mov", "aaaa", CAP_AVFOUNDATION},
   {"mp4", "aaaa", CAP_AVFOUNDATION},
   {"m4v", "aaaa", CAP_AVFOUNDATION},
#endif

#ifdef HAVE_FFMPEG
    {"avi", "aaaa", CAP_FFMPEG},
    {"mkv", "aaaa", CAP_FFMPEG},
#endif

#ifdef HAVE_GSTREAMER
    {"avi", "aaaa", CAP_GSTREAMER},
    {"mkv", "aaaa", CAP_GSTREAMER},
#endif
    {"avi", "aaaa", CAP_OPENCV_MJPEG},
};

    const size_t N = sizeof(params)/sizeof(params[0]);
    vector<Ext_Fourcc_API> result; result.reserve(N);
    for (size_t i = 0; i < N; i++)
    {
        const Ext_Fourcc_API& src = params[i];
        Ext_Fourcc_API e = { src.ext, src.fourcc, src.api };
        result.push_back(e);
    }
    return result;
}

INSTANTIATE_TEST_CASE_P(videoio, Videoio_Writer_bad_fourcc, testing::ValuesIn(generate_Ext_Fourcc_API_nocrash()));

typedef testing::TestWithParam<VideoCaptureAPIs> safe_capture;

TEST_P(safe_capture, frames_independency)
{
    VideoCaptureAPIs apiPref = GetParam();
    if (!videoio_registry::hasBackend(apiPref))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));

    VideoCapture cap;
    String video_file = BunnyParameters::getFilename(String(".avi"));
    EXPECT_NO_THROW(cap.open(video_file, apiPref));
    if (!cap.isOpened())
    {
        std::cout << "SKIP test: backend " << apiPref << " can't open the video: " << video_file << std::endl;
        return;
    }

    Mat frames[10];
    Mat hardCopies[10];
    for(int i = 0; i < 10; i++)
    {
        ASSERT_NO_THROW(cap >> frames[i]);
        EXPECT_FALSE(frames[i].empty());
        hardCopies[i] = frames[i].clone();
    }

    for(int i = 0; i < 10; i++)
        EXPECT_EQ(0, cv::norm(frames[i], hardCopies[i], NORM_INF)) << i;
}

static VideoCaptureAPIs safe_apis[] = {CAP_FFMPEG, CAP_GSTREAMER, CAP_MSMF,CAP_AVFOUNDATION};
INSTANTIATE_TEST_CASE_P(videoio, safe_capture, testing::ValuesIn(safe_apis));

//==================================================================================================
// TEST_P(videocapture_acceleration, ...)

struct VideoCaptureAccelerationInput
{
    const char* filename;
    double psnr_threshold;
};

static inline
std::ostream& operator<<(std::ostream& out, const VideoCaptureAccelerationInput& p)
{
    out << p.filename;
    return out;
}

typedef testing::TestWithParam<tuple<VideoCaptureAccelerationInput, VideoCaptureAPIs, VideoAccelerationType, bool>> videocapture_acceleration;

TEST_P(videocapture_acceleration, read)
{
    auto param = GetParam();
    std::string filename = get<0>(param).filename;
#if 0  // FIXIT OpenCV 5.0 putText result
    double psnr_threshold = get<0>(param).psnr_threshold;
#else
    double psnr_threshold = 10;
#endif
    VideoCaptureAPIs backend = get<1>(param);
    VideoAccelerationType va_type = get<2>(param);
    bool use_umat = get<3>(param);
    const int frameNum = 15;

    std::string filepath = cvtest::findDataFile("video/" + filename);

    if (backend == CAP_MSMF && (
        filename == "sample_322x242_15frames.yuv420p.mjpeg.mp4" ||
        filename == "sample_322x242_15frames.yuv420p.libx265.mp4" ||
        filename == "sample_322x242_15frames.yuv420p.libaom-av1.mp4" ||
        filename == "sample_322x242_15frames.yuv420p.mpeg2video.mp4"
    ))
        throw SkipTestException("Format/codec is not supported");


    std::string backend_name = cv::videoio_registry::getBackendName(backend);
    if (!videoio_registry::hasBackend(backend))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + backend_name);


    // HW reader
    std::vector<int> params = { CAP_PROP_HW_ACCELERATION, static_cast<int>(va_type) };
    if (use_umat)
    {
        if (backend != CAP_FFMPEG)
            throw SkipTestException(cv::String("UMat/OpenCL mapping is not supported by current backend: ") + backend_name);
        if (!cv::videoio_registry::isBackendBuiltIn(backend))
            throw SkipTestException(cv::String("UMat/OpenCL mapping is not supported through plugins yet: ") + backend_name);
        params.push_back(CAP_PROP_HW_ACCELERATION_USE_OPENCL);
        params.push_back(1);
    }
    VideoCapture hw_reader(filepath, backend, params);
    if (!hw_reader.isOpened())
    {
        if (use_umat)
        {
            throw SkipTestException(backend_name + " VideoCapture on " + filename + " not supported with HW acceleration + OpenCL/Umat mapping, skipping");
        }
        else if (va_type == VIDEO_ACCELERATION_ANY || va_type == VIDEO_ACCELERATION_NONE)
        {
            // ANY HW acceleration should have fallback to SW codecs
            VideoCapture sw_reader(filepath, backend, {
                    CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE
            });
            if (!sw_reader.isOpened())
                throw SkipTestException(backend_name + " VideoCapture on " + filename + " not supported, skipping");

            ASSERT_TRUE(hw_reader.isOpened()) << "ANY HW acceleration should have fallback to SW codecs";
        }
        else
        {
            throw SkipTestException(backend_name + " VideoCapture on " + filename + " not supported with HW acceleration, skipping");
        }
    }

    VideoAccelerationType actual_va = static_cast<VideoAccelerationType>(static_cast<int>(hw_reader.get(CAP_PROP_HW_ACCELERATION)));
    if (va_type != VIDEO_ACCELERATION_ANY && va_type != VIDEO_ACCELERATION_NONE)
    {
        ASSERT_EQ((int)actual_va, (int)va_type) << "actual_va=" << actual_va << ", va_type=" << va_type;
    }
    std::cout << "VideoCapture " << backend_name << ":" << actual_va << std::endl << std::flush;

    double min_psnr_original = 1000;
    for (int i = 0; i < frameNum; i++)
    {
        SCOPED_TRACE(cv::format("frame=%d", i));
        Mat frame;
        if (use_umat)
        {
            UMat umat;
            bool read_umat_result = hw_reader.read(umat);
            if (!read_umat_result && i == 0)
            {
                if (filename == "sample_322x242_15frames.yuv420p.libvpx-vp9.mp4")
                    throw SkipTestException("Unable to read the first frame with VP9 codec (media stack misconfiguration / bug)");
                // FFMPEG: [av1 @ 0000027ac07d1340] Your platform doesn't suppport hardware accelerated AV1 decoding.
                if (filename == "sample_322x242_15frames.yuv420p.libaom-av1.mp4")
                    throw SkipTestException("Unable to read the first frame with AV1 codec (missing support)");
            }
#ifdef _WIN32
            if (!read_umat_result && i == 1)
            {
                if (filename == "sample_322x242_15frames.yuv420p.libvpx-vp9.mp4")
                    throw SkipTestException("Unable to read the second frame with VP9 codec (media stack misconfiguration / outdated MSMF version)");
            }
#endif
            EXPECT_TRUE(read_umat_result);
            ASSERT_FALSE(umat.empty());
            umat.copyTo(frame);
        }
        else
        {
            bool read_result = hw_reader.read(frame);
            if (!read_result && i == 0)
            {
                if (filename == "sample_322x242_15frames.yuv420p.libvpx-vp9.mp4")
                    throw SkipTestException("Unable to read the first frame with VP9 codec (media stack misconfiguration / bug)");
                // FFMPEG: [av1 @ 0000027ac07d1340] Your platform doesn't suppport hardware accelerated AV1 decoding.
                if (filename == "sample_322x242_15frames.yuv420p.libaom-av1.mp4")
                    throw SkipTestException("Unable to read the first frame with AV1 codec (missing support)");
            }
#ifdef _WIN32
            if (!read_result && i == 1)
            {
                if (filename == "sample_322x242_15frames.yuv420p.libvpx-vp9.mp4")
                    throw SkipTestException("Unable to read the second frame with VP9 codec (media stack misconfiguration / outdated MSMF version)");
            }
#endif
            EXPECT_TRUE(read_result);
        }
        ASSERT_FALSE(frame.empty());

        if (cvtest::debugLevel > 0)
        {
            imwrite(cv::format("test_frame%03d.png", i), frame);
        }

        Mat original(frame.size(), CV_8UC3, Scalar::all(0));
        generateFrame(i, frameNum, original);
        double psnr = cvtest::PSNR(frame, original);
        if (psnr < min_psnr_original)
            min_psnr_original = psnr;
    }

    std::ostringstream ss; ss << actual_va;
    std::string actual_va_str = ss.str();
    std::cout << "VideoCapture with acceleration = " << cv::format("%-6s @ %-10s", actual_va_str.c_str(), backend_name.c_str())
            << " on " << filename
            << " with PSNR-original = " << min_psnr_original
            << std::endl << std::flush;
    EXPECT_GE(min_psnr_original, psnr_threshold);
}

static const VideoCaptureAccelerationInput hw_filename[] = {
        { "sample_322x242_15frames.yuv420p.libxvid.mp4", 28.0 },
        { "sample_322x242_15frames.yuv420p.mjpeg.mp4", 20.0 },
        { "sample_322x242_15frames.yuv420p.mpeg2video.mp4", 24.0 },  // GSTREAMER on Ubuntu 18.04
        { "sample_322x242_15frames.yuv420p.libx264.mp4", 20.0 },  // 20 - D3D11 (i7-11800H), 23 - D3D11 on GHA/Windows, GSTREAMER on Ubuntu 18.04
        { "sample_322x242_15frames.yuv420p.libx265.mp4", 20.0 },  // 20 - D3D11 (i7-11800H), 23 - D3D11 on GHA/Windows
        { "sample_322x242_15frames.yuv420p.libvpx-vp9.mp4", 30.0 },
        { "sample_322x242_15frames.yuv420p.libaom-av1.mp4", 30.0 }
};

static const VideoCaptureAPIs hw_backends[] = {
        CAP_FFMPEG,
        CAP_GSTREAMER,
#ifdef _WIN32
        CAP_MSMF,
#endif
};

static const VideoAccelerationType hw_types[] = {
        VIDEO_ACCELERATION_NONE,
        VIDEO_ACCELERATION_ANY,
        VIDEO_ACCELERATION_MFX,
#ifdef _WIN32
        VIDEO_ACCELERATION_D3D11,
#else
        VIDEO_ACCELERATION_VAAPI,
#endif
};

static bool hw_use_umat[] = {
        false,
        true
};

INSTANTIATE_TEST_CASE_P(videoio, videocapture_acceleration, testing::Combine(
    testing::ValuesIn(hw_filename),
    testing::ValuesIn(hw_backends),
    testing::ValuesIn(hw_types),
    testing::ValuesIn(hw_use_umat)
));

////////////////////////////////////////// TEST_P(video_acceleration, write_read)

typedef tuple<Ext_Fourcc_PSNR, VideoAccelerationType, bool> VATestParams;

typedef testing::TestWithParam<VATestParams> videowriter_acceleration;

TEST_P(videowriter_acceleration, write)
{
    auto param = GetParam();
    VideoCaptureAPIs backend = get<0>(param).api;
    std::string codecid = get<0>(param).fourcc;
    std::string extension = get<0>(param).ext;
    double psnr_threshold = get<0>(param).PSNR;
    VideoAccelerationType va_type = get<1>(param);
    bool use_umat = get<2>(param);
    std::string backend_name = cv::videoio_registry::getBackendName(backend);
    if (!videoio_registry::hasBackend(backend))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + backend_name);
#ifdef __linux__
    if (cvtest::skipUnstableTests && backend == CAP_GSTREAMER &&
        (extension == "mkv") && (codecid == "MPEG"))
    {
        throw SkipTestException("Unstable GSTREAMER test");
    }
#endif

    const Size sz(640, 480);
    const int frameNum = 15;
    const double fps = 25;

    std::string filename = tempfile("videowriter_acceleration.") + extension;

    // Write video
    VideoAccelerationType actual_va;
    {
        std::vector<int> params = { VIDEOWRITER_PROP_HW_ACCELERATION, static_cast<int>(va_type) };
        if (use_umat) {
            if (backend != CAP_FFMPEG)
                throw SkipTestException(cv::String("UMat/OpenCL mapping is not supported by current backend: ") + backend_name);
            if (!cv::videoio_registry::isBackendBuiltIn(backend))
                throw SkipTestException(cv::String("UMat/OpenCL mapping is not supported through plugins yet: ") + backend_name);
            params.push_back(VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL);
            params.push_back(1);
        }
        VideoWriter hw_writer(
            filename,
            backend,
            VideoWriter::fourcc(codecid[0], codecid[1], codecid[2], codecid[3]),
            fps,
            sz,
            params
        );

        if (!hw_writer.isOpened())
        {
            if (use_umat)
            {
                throw SkipTestException(backend_name + " VideoWriter on " + filename + " not supported with HW acceleration + OpenCL/Umat mapping, skipping");
            }
            else if (va_type == VIDEO_ACCELERATION_ANY || va_type == VIDEO_ACCELERATION_NONE)
            {
                // ANY HW acceleration should have fallback to SW codecs
                {
                    VideoWriter sw_writer(
                        filename,
                        backend,
                        VideoWriter::fourcc(codecid[0], codecid[1], codecid[2], codecid[3]),
                        fps,
                        sz,
                        {
                            VIDEOWRITER_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE,
                        }
                    );
                    if (!sw_writer.isOpened()) {
                        remove(filename.c_str());
                        throw SkipTestException(backend_name + " VideoWriter on codec " + codecid + " not supported, skipping");
                    }
                }
                remove(filename.c_str());
                ASSERT_TRUE(hw_writer.isOpened()) << "ANY HW acceleration should have fallback to SW codecs";
            } else {
                throw SkipTestException(backend_name + " VideoWriter on " + filename + " not supported with HW acceleration, skipping");
            }
        }

        actual_va = static_cast<VideoAccelerationType>(static_cast<int>(hw_writer.get(VIDEOWRITER_PROP_HW_ACCELERATION)));
        if (va_type != VIDEO_ACCELERATION_ANY && va_type != VIDEO_ACCELERATION_NONE)
        {
            ASSERT_EQ((int)actual_va, (int)va_type) << "actual_va=" << actual_va << ", va_type=" << va_type;
        }
        std::cout << "VideoWriter " << backend_name << ":" << actual_va << std::endl << std::flush;

        Mat frame(sz, CV_8UC3);
        for (int i = 0; i < frameNum; ++i) {
            generateFrame(i, frameNum, frame);
            if (use_umat) {
                UMat umat;
                frame.copyTo(umat);
                hw_writer.write(umat);
            }
            else {
                hw_writer.write(frame);
            }
        }
    }

    std::ifstream ofile(filename, std::ios::binary);
    ofile.seekg(0, std::ios::end);
    int64 fileSize = (int64)ofile.tellg();
    ASSERT_GT(fileSize, 0);
    std::cout << "File size: " << fileSize << std::endl;

    // Read video and check PSNR on every frame
    {
        VideoCapture reader(
            filename,
            CAP_ANY /*backend*/,
            { CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE }
        );
        ASSERT_TRUE(reader.isOpened());
        double min_psnr = 1000;
        Mat reference(sz, CV_8UC3);
        for (int i = 0; i < frameNum; ++i) {
            Mat actual;
            if (use_umat) {
                UMat umat;
                EXPECT_TRUE(reader.read(umat));
                umat.copyTo(actual);
            }
            else {
                EXPECT_TRUE(reader.read(actual));
            }
            EXPECT_FALSE(actual.empty());
            generateFrame(i, frameNum, reference);
            EXPECT_EQ(reference.size(), actual.size());
            EXPECT_EQ(reference.depth(), actual.depth());
            EXPECT_EQ(reference.channels(), actual.channels());
            double psnr = cvtest::PSNR(actual, reference);
            EXPECT_GE(psnr, psnr_threshold) << " frame " << i;
            if (psnr < min_psnr)
                min_psnr = psnr;
        }
        Mat actual;
        EXPECT_FALSE(reader.read(actual));
        {
            std::ostringstream ss; ss << actual_va;
            std::string actual_va_str = ss.str();
            std::cout << "VideoWriter with acceleration = " << cv::format("%-6s @ %-10s", actual_va_str.c_str(), backend_name.c_str())
                    << " on codec=" << codecid << " (." << extension << ")"
                    << ", bitrate = " << fileSize / (frameNum / fps)
                    << ", with PSNR-original = " << min_psnr
                    << std::endl << std::flush;
        }
        remove(filename.c_str());
    }
}

static Ext_Fourcc_PSNR hw_codecs[] = {
        {"mp4", "MPEG", 29.f, CAP_FFMPEG},
        {"mp4", "H264", 29.f, CAP_FFMPEG},
        {"mp4", "HEVC", 29.f, CAP_FFMPEG},
        {"avi", "MJPG", 29.f, CAP_FFMPEG},
        {"avi", "XVID", 29.f, CAP_FFMPEG},
        //{"webm", "VP8", 29.f, CAP_FFMPEG},
        //{"webm", "VP9", 29.f, CAP_FFMPEG},

        {"mkv", "MPEG", 29.f, CAP_GSTREAMER},
        {"mkv", "H264", 29.f, CAP_GSTREAMER},

#ifdef _WIN32
        {"mp4", "MPEG", 29.f, CAP_MSMF},
        {"mp4", "H264", 29.f, CAP_MSMF},
        {"mp4", "HEVC", 29.f, CAP_MSMF},
#endif
};

INSTANTIATE_TEST_CASE_P(videoio, videowriter_acceleration, testing::Combine(
        testing::ValuesIn(hw_codecs),
        testing::ValuesIn(hw_types),
        testing::ValuesIn(hw_use_umat)
));

class BufferStream : public cv::IStreamReader
{
public:
    BufferStream(const std::string& filename)
    {
        Ptr<std::filebuf> file = makePtr<std::filebuf>();
        file->open(filename.c_str(), std::ios::in | std::ios::binary);
        stream = file;
    }

    BufferStream(const Ptr<std::stringbuf>& _stream) : stream(_stream) {}

    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        auto result = stream->sgetn(buffer, size);
        return result;
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        auto result = stream->pubseekoff(offset, way == SEEK_SET ? std::ios_base::beg : (way == SEEK_END ? std::ios_base::end : std::ios_base::cur));
        return result;
    }

private:
    Ptr<std::streambuf> stream;
};

typedef testing::TestWithParam<tuple<std::string, VideoCaptureAPIs>> stream_capture;
TEST_P(stream_capture, read)
{
    std::string ext = get<0>(GetParam());
    VideoCaptureAPIs apiPref = get<1>(GetParam());
    std::vector<VideoCaptureAPIs> supportedAPIs = videoio_registry::getStreamBufferedBackends();
    if (!videoio_registry::hasBackend(apiPref))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));
    if (std::find(supportedAPIs.begin(), supportedAPIs.end(), apiPref) == supportedAPIs.end())
        throw SkipTestException(cv::String("Backend is not supported: ") + cv::videoio_registry::getBackendName(apiPref));
    if (cvtest::skipUnstableTests && apiPref == CAP_MSMF && (ext == "h264" || ext == "h265" || ext == "mpg"))
        throw SkipTestException("Unstable MSMF test");

    if (!videoio_registry::isBackendBuiltIn(apiPref))
    {
        int pluginABI, pluginAPI;
        videoio_registry::getStreamBufferedBackendPluginVersion(apiPref, pluginABI, pluginAPI);
        if (pluginABI < 1 || (pluginABI == 1 && pluginAPI < 2))
            throw SkipTestException(format("Buffer capture supported since ABI/API = 1/2. %s plugin is %d/%d",
                                           cv::videoio_registry::getBackendName(apiPref).c_str(), pluginABI, pluginAPI));
    }

    VideoCapture cap;
    String video_file = BunnyParameters::getFilename(String(".") + ext);
    ASSERT_TRUE(utils::fs::exists(video_file));
    EXPECT_NO_THROW(cap.open(makePtr<BufferStream>(video_file), apiPref, {}));
    ASSERT_TRUE(cap.isOpened());

    const int numFrames = 10;
    Mat frames[numFrames];
    Mat hardCopies[numFrames];
    for(int i = 0; i < numFrames; i++)
    {
        ASSERT_NO_THROW(cap >> frames[i]);
        EXPECT_FALSE(frames[i].empty());
        hardCopies[i] = frames[i].clone();
    }

    for(int i = 0; i < numFrames; i++)
        EXPECT_EQ(0, cv::norm(frames[i], hardCopies[i], NORM_INF)) << i;
}
INSTANTIATE_TEST_CASE_P(videoio, stream_capture,
                          testing::Combine(
                              testing::ValuesIn(bunny_params),
                              testing::ValuesIn(backend_params)));

// This test for stream input for container format (See test_ffmpeg/videoio_container.read test)
typedef testing::TestWithParam<std::string> stream_capture_ffmpeg;
TEST_P(stream_capture_ffmpeg, raw)
{
    std::string ext = GetParam();
    VideoCaptureAPIs apiPref = CAP_FFMPEG;
    std::vector<VideoCaptureAPIs> supportedAPIs = videoio_registry::getStreamBufferedBackends();
    if (!videoio_registry::hasBackend(apiPref))
        throw SkipTestException(cv::String("Backend is not available/disabled: ") + cv::videoio_registry::getBackendName(apiPref));
    if (std::find(supportedAPIs.begin(), supportedAPIs.end(), apiPref) == supportedAPIs.end())
        throw SkipTestException(cv::String("Backend is not supported: ") + cv::videoio_registry::getBackendName(apiPref));

    if (!videoio_registry::isBackendBuiltIn(apiPref))
    {
        int pluginABI, pluginAPI;
        videoio_registry::getStreamBufferedBackendPluginVersion(apiPref, pluginABI, pluginAPI);
        if (pluginABI < 1 || (pluginABI == 1 && pluginAPI < 2))
            throw SkipTestException(format("Buffer capture supported since ABI/API = 1/2. %s plugin is %d/%d",
                                           cv::videoio_registry::getBackendName(apiPref).c_str(), pluginABI, pluginAPI));
    }

    VideoCapture container;
    String video_file = BunnyParameters::getFilename(String(".") + ext);
    ASSERT_TRUE(utils::fs::exists(video_file));
    EXPECT_NO_THROW(container.open(video_file, apiPref, {CAP_PROP_FORMAT, -1}));
    ASSERT_TRUE(container.isOpened());
    ASSERT_EQ(-1.f, container.get(CAP_PROP_FORMAT));

    auto stream = std::make_shared<std::stringbuf>();
    Mat keyFrame;
    while (true)
    {
        container >> keyFrame;
        if (keyFrame.empty())
            break;
        stream->sputn(keyFrame.ptr<char>(), keyFrame.total());
    }

    VideoCapture capRef(video_file);
    VideoCapture capStream;
    EXPECT_NO_THROW(capStream.open(makePtr<BufferStream>(stream), apiPref, {}));
    ASSERT_TRUE(capStream.isOpened());

    const int numFrames = 10;
    Mat frameRef, frame;
    for (int i = 0; i < numFrames; ++i)
    {
        capRef >> frameRef;
        ASSERT_NO_THROW(capStream >> frame);
        EXPECT_FALSE(frame.empty());
        EXPECT_EQ(0, cv::norm(frame, frameRef, NORM_INF)) << i;
    }
}
INSTANTIATE_TEST_CASE_P(videoio, stream_capture_ffmpeg, testing::Values("h264", "h265", "mjpg.avi"));

} // namespace
