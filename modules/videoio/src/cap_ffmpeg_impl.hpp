/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <opencv2/core/utils/configuration.private.hpp>
#include "cap_ffmpeg_legacy_api.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "cap_interface.hpp"

using namespace cv;

#if !(defined(_WIN32) || defined(WINCE))
# include <pthread.h>
#endif
#include <algorithm>
#include <limits>
#include <string.h>

#ifndef __OPENCV_BUILD
#define CV_FOURCC(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))
#endif

#define CALC_FFMPEG_VERSION(a,b,c) ( a<<16 | b<<8 | c )

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4244 4510 4610 )
#endif

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(disable: 4996)  // was declared deprecated
#endif

#ifndef CV_UNUSED  // Required for standalone compilation mode (OpenCV defines this in base.hpp)
#define CV_UNUSED(name) (void)name
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "ffmpeg_codecs.hpp"

#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
// https://github.com/FFmpeg/FFmpeg/blame/d79c240196f43b93bd204363f1facc270029f113/doc/APIchanges#L1689-L1695
#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(52, 85, 100) : CALC_FFMPEG_VERSION(53, 15, 0))
#include <libavutil/display.h>
#endif

#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 63, 100) : CALC_FFMPEG_VERSION(54, 6, 0))
#include <libavutil/imgutils.h>
#endif

#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#ifdef HAVE_FFMPEG_LIBAVDEVICE
#include <libavdevice/avdevice.h>
#endif

// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L602-L605
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(58, 9, 100)
#  define CV_FFMPEG_REGISTER
#endif

// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L654-L657
#if LIBAVCODEC_BUILD < CALC_FFMPEG_VERSION(58, 9, 100)
#  define CV_FFMPEG_LOCKMGR
#endif

// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L390-L392
#if LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(58, 87, 100)
#include <libavcodec/bsf.h>
#endif

#include <libavutil/pixdesc.h>

// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L208-L210
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(59, 0, 100)
#  define CV_FFMPEG_FMT_CONST const
#else
#  define CV_FFMPEG_FMT_CONST
#endif

// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L623-L624
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(58, 7, 100)
#  define CV_FFMPEG_URL
#endif

// AVStream.codec deprecated in favor of AVStream.codecpar
// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L1039-L1040
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(59, 16, 100)
//#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(57, 33, 100)
#  define CV_FFMPEG_CODECPAR
#  define CV_FFMPEG_CODEC_FIELD codecpar
#else
#  define CV_FFMPEG_CODEC_FIELD codec
#endif

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(59, 16, 100)
#  define CV_FFMPEG_PTS_FIELD pts
#else
#  define CV_FFMPEG_PTS_FIELD pkt_pts
#endif

// https://github.com/FFmpeg/FFmpeg/blob/b6af56c034759b81985f8ea094e41cbd5f7fecfb/doc/APIchanges#L1757-L1758
#if LIBAVUTIL_BUILD < CALC_FFMPEG_VERSION(52, 63, 100)
inline static AVRational av_make_q(int num, int den)
{
    AVRational res;
    res.num = num;
    res.den = den;
    return res;
}
#endif



#ifdef __cplusplus
}
#endif

// GCC 4.x compilation bug. Details: https://github.com/opencv/opencv/issues/20292
#if (defined(__GNUC__) && __GNUC__ < 5) && !defined(__clang__)
#undef USE_AV_HW_CODECS
#define USE_AV_HW_CODECS 0
#endif

//#define USE_AV_HW_CODECS 0
#ifndef USE_AV_HW_CODECS
#if LIBAVUTIL_VERSION_MAJOR >= 56 // FFMPEG 4.0+
#define USE_AV_HW_CODECS 1
#include "cap_ffmpeg_hw.hpp"
#else
#define USE_AV_HW_CODECS 0
#endif
#endif

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( default: 4244 4510 4610 )
#endif

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

#if defined _WIN32
    #include <windows.h>
    #if defined _MSC_VER && _MSC_VER < 1900
    struct timespec
    {
        time_t tv_sec;
        long   tv_nsec;
    };
  #endif
#elif defined __linux__ || defined __APPLE__ || defined __HAIKU__
    #include <unistd.h>
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/time.h>
#if defined __APPLE__
    #include <sys/sysctl.h>
    #include <mach/clock.h>
    #include <mach/mach.h>
#endif
#endif


#if defined(__APPLE__)
#define AV_NOPTS_VALUE_ ((int64_t)0x8000000000000000LL)
#else
#define AV_NOPTS_VALUE_ ((int64_t)AV_NOPTS_VALUE)
#endif

#ifndef AVERROR_EOF
#define AVERROR_EOF (-MKTAG( 'E','O','F',' '))
#endif

#if LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(54,25,0)
#  define CV_CODEC_ID AVCodecID
#  define CV_CODEC(name) AV_##name
#else
#  define CV_CODEC_ID CodecID
#  define CV_CODEC(name) name
#endif

#ifndef PKT_FLAG_KEY
#define PKT_FLAG_KEY AV_PKT_FLAG_KEY
#endif

#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(52, 38, 100) : CALC_FFMPEG_VERSION(52, 13, 0))
#define USE_AV_FRAME_GET_BUFFER 1
#else
#define USE_AV_FRAME_GET_BUFFER 0
#ifndef AV_NUM_DATA_POINTERS // required for 0.7.x/0.8.x ffmpeg releases
#define AV_NUM_DATA_POINTERS 4
#endif
#endif


#ifndef USE_AV_INTERRUPT_CALLBACK
#define USE_AV_INTERRUPT_CALLBACK 1
#endif

#ifndef USE_AV_SEND_FRAME_API
// https://github.com/FFmpeg/FFmpeg/commit/7fc329e2dd6226dfecaa4a1d7adf353bf2773726
#if LIBAVCODEC_VERSION_MICRO >= 100 \
    && LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(57, 37, 100)
#define USE_AV_SEND_FRAME_API 1
#else
#define USE_AV_SEND_FRAME_API 0
#endif
#endif

#if USE_AV_INTERRUPT_CALLBACK
#define LIBAVFORMAT_INTERRUPT_OPEN_DEFAULT_TIMEOUT_MS 30000
#define LIBAVFORMAT_INTERRUPT_READ_DEFAULT_TIMEOUT_MS 30000

#ifdef _WIN32
// http://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows

static
inline LARGE_INTEGER get_filetime_offset()
{
    SYSTEMTIME s;
    FILETIME f;
    LARGE_INTEGER t;

    s.wYear = 1970;
    s.wMonth = 1;
    s.wDay = 1;
    s.wHour = 0;
    s.wMinute = 0;
    s.wSecond = 0;
    s.wMilliseconds = 0;
    SystemTimeToFileTime(&s, &f);
    t.QuadPart = f.dwHighDateTime;
    t.QuadPart <<= 32;
    t.QuadPart |= f.dwLowDateTime;
    return t;
}

static
inline void get_monotonic_time(timespec *tv)
{
    LARGE_INTEGER           t;
    FILETIME				f;
    double                  microseconds;
    static LARGE_INTEGER    offset;
    static double           frequencyToMicroseconds;
    static int              initialized = 0;
    static BOOL             usePerformanceCounter = 0;

    if (!initialized)
    {
        LARGE_INTEGER performanceFrequency;
        initialized = 1;
        usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
        if (usePerformanceCounter)
        {
            QueryPerformanceCounter(&offset);
            frequencyToMicroseconds = (double)performanceFrequency.QuadPart / 1000000.;
        }
        else
        {
            offset = get_filetime_offset();
            frequencyToMicroseconds = 10.;
        }
    }

    if (usePerformanceCounter)
    {
        QueryPerformanceCounter(&t);
    } else {
        GetSystemTimeAsFileTime(&f);
        t.QuadPart = f.dwHighDateTime;
        t.QuadPart <<= 32;
        t.QuadPart |= f.dwLowDateTime;
    }

    t.QuadPart -= offset.QuadPart;
    microseconds = (double)t.QuadPart / frequencyToMicroseconds;
    t.QuadPart = (LONGLONG)microseconds;
    tv->tv_sec = t.QuadPart / 1000000;
    tv->tv_nsec = (t.QuadPart % 1000000) * 1000;
}
#else
static
inline void get_monotonic_time(timespec *time)
{
#if defined(__APPLE__) && defined(__MACH__)
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    time->tv_sec = mts.tv_sec;
    time->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_MONOTONIC, time);
#endif
}
#endif

static
inline timespec get_monotonic_time_diff(timespec start, timespec end)
{
    timespec temp;
    if (end.tv_nsec - start.tv_nsec < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

static
inline double get_monotonic_time_diff_ms(timespec time1, timespec time2)
{
    timespec delta = get_monotonic_time_diff(time1, time2);
    double milliseconds = delta.tv_sec * 1000 + (double)delta.tv_nsec / 1000000.0;

    return milliseconds;
}
#endif // USE_AV_INTERRUPT_CALLBACK


struct Image_FFMPEG
{
    unsigned char* data;
    int step;
    int width;
    int height;
};


#if USE_AV_INTERRUPT_CALLBACK
struct AVInterruptCallbackMetadata
{
    timespec value;
    unsigned int timeout_after_ms;
    int timeout;
};

// https://github.com/opencv/opencv/pull/12693#issuecomment-426236731
static
inline const char* _opencv_avcodec_get_name(CV_CODEC_ID id)
{
#if LIBAVCODEC_VERSION_MICRO >= 100 \
    && LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(53, 47, 100)
    return avcodec_get_name(id);
#else
    const AVCodecDescriptor *cd;
    AVCodec *codec;

    if (id == AV_CODEC_ID_NONE)
    {
        return "none";
    }
    cd = avcodec_descriptor_get(id);
    if (cd)
    {
        return cd->name;
    }
    codec = avcodec_find_decoder(id);
    if (codec)
    {
        return codec->name;
    }
    codec = avcodec_find_encoder(id);
    if (codec)
    {
        return codec->name;
    }

    return "unknown_codec";
#endif
}


static int _opencv_ffmpeg_interrupt_callback(void *ptr)
{
    AVInterruptCallbackMetadata* metadata = (AVInterruptCallbackMetadata*)ptr;

    if(!metadata)
    {
        CV_LOG_WARNING(NULL, "Stream timeout without metadata passed");
        return 0;
    }

    if (metadata->timeout_after_ms == 0)
    {
        return 0; // timeout is disabled
    }

    timespec now;
    get_monotonic_time(&now);

    double timeout = get_monotonic_time_diff_ms(metadata->value, now);
    metadata->timeout = timeout > metadata->timeout_after_ms;
    if (metadata->timeout)
    {
        CV_LOG_WARNING(NULL, cv::format("Stream timeout triggered after %lf ms", timeout));
        return -1;
    }

    return 0;
}
#endif

static
inline void _opencv_ffmpeg_av_packet_unref(AVPacket *pkt)
{
#if LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(55, 25, 100) : CALC_FFMPEG_VERSION(55, 16, 0))
    av_packet_unref(pkt);
#else
    av_free_packet(pkt);
#endif
};

static
inline void _opencv_ffmpeg_av_image_fill_arrays(void *frame, uint8_t *ptr, enum AVPixelFormat pix_fmt, int width, int height)
{
#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 63, 100) : CALC_FFMPEG_VERSION(54, 6, 0))
    av_image_fill_arrays(((AVFrame*)frame)->data, ((AVFrame*)frame)->linesize, ptr, pix_fmt, width, height, 1);
#else
    avpicture_fill((AVPicture*)frame, ptr, pix_fmt, width, height);
#endif
};

static
inline int _opencv_ffmpeg_av_image_get_buffer_size(enum AVPixelFormat pix_fmt, int width, int height)
{
#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 63, 100) : CALC_FFMPEG_VERSION(54, 6, 0))
    return av_image_get_buffer_size(pix_fmt, width, height, 1);
#else
    return avpicture_get_size(pix_fmt, width, height);
#endif
};

static AVRational _opencv_ffmpeg_get_sample_aspect_ratio(AVStream *stream)
{
#if LIBAVUTIL_VERSION_MICRO >= 100 && LIBAVUTIL_BUILD >= CALC_FFMPEG_VERSION(54, 5, 100)
    return av_guess_sample_aspect_ratio(NULL, stream, NULL);
#else
    AVRational undef = {0, 1};

    // stream
    AVRational ratio = stream ? stream->sample_aspect_ratio : undef;
    av_reduce(&ratio.num, &ratio.den, ratio.num, ratio.den, INT_MAX);
    if (ratio.num > 0 && ratio.den > 0)
        return ratio;

    // codec
    ratio  = stream && stream->codec ? stream->codec->sample_aspect_ratio : undef;
    av_reduce(&ratio.num, &ratio.den, ratio.num, ratio.den, INT_MAX);
    if (ratio.num > 0 && ratio.den > 0)
        return ratio;

    return undef;
#endif
}

inline static std::string _opencv_ffmpeg_get_error_string(int error_code)
{
    char buf[255] = {0};
    const int err = av_strerror(error_code, buf, 254);
    if (err == 0)
        return std::string(buf);
    else
        return std::string("Unknown error");
}

struct CvCapture_FFMPEG
{
    bool open(const char* filename, const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params);
    void close();

    double getProperty(int) const;
    bool setProperty(int, double);
    bool grabFrame();
    bool retrieveFrame(int flag, unsigned char** data, int* step, int* width, int* height, int* cn, int* depth);
    bool retrieveHWFrame(cv::OutputArray output);
    void rotateFrame(cv::Mat &mat) const;

    void init();

    void    seek(int64_t frame_number);
    void    seek(double sec);
    bool    slowSeek( int framenumber );

    int64_t get_total_frames() const;
    double  get_duration_sec() const;
    double  get_fps() const;
    int64_t get_bitrate() const;

    double  r2d(AVRational r) const;
    int64_t dts_to_frame_number(int64_t dts);
    double  dts_to_sec(int64_t dts) const;
    void    get_rotation_angle();

    AVFormatContext * ic;
    AVCodec         * avcodec;
    AVCodecContext  * context;
    int               video_stream;
    AVStream        * video_st;
    AVFrame         * picture;
    AVFrame           rgb_picture;
    int64_t           picture_pts;
    int64_t           pts_in_fps_time_base;
    int64_t           dts_delay_in_fps_time_base;

    AVIOContext     * avio_context;

    AVPacket          packet;
    Image_FFMPEG      frame;
    struct SwsContext *img_convert_ctx;

    int64_t frame_number, first_frame_number;

    int    rotation_angle; // valid 0, 90, 180, 270
    double eps_zero;
/*
   'filename' contains the filename of the videosource,
   'filename==NULL' indicates that ffmpeg's seek support works
   for the particular file.
   'filename!=NULL' indicates that the slow fallback function is used for seeking,
   and so the filename is needed to reopen the file on backward seeking.
*/
    char              * filename;

    Ptr<IStreamReader> readStream;

    AVDictionary *dict;
#if USE_AV_INTERRUPT_CALLBACK
    int open_timeout;
    int read_timeout;
    AVInterruptCallbackMetadata interrupt_metadata;
#endif

    bool setRaw();
    bool processRawPacket();
    bool rawMode;
    bool rawModeInitialized;
    bool rawSeek;
    bool convertRGB;
    AVPacket packet_filtered;
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(58, 20, 100)
    AVBSFContext* bsfc;
 #else
    AVBitStreamFilterContext* bsfc;
#endif
    VideoAccelerationType va_type;
    int hw_device;
    int use_opencl;
    int extraDataIdx;
};

void CvCapture_FFMPEG::init()
{
#ifdef HAVE_FFMPEG_LIBAVDEVICE
    //libavdevice is available, so let's register all input and output devices (e.g v4l2)
    avdevice_register_all();
#endif
    ic = 0;
    video_stream = -1;
    video_st = 0;
    picture = 0;
    picture_pts = AV_NOPTS_VALUE_;
    pts_in_fps_time_base = 0;
    dts_delay_in_fps_time_base = 0;
    first_frame_number = -1;
    memset( &rgb_picture, 0, sizeof(rgb_picture) );
    memset( &frame, 0, sizeof(frame) );
    filename = 0;
    memset(&packet, 0, sizeof(packet));
    av_init_packet(&packet);
    img_convert_ctx = 0;

    avcodec = 0;
    context = 0;
    avio_context = 0;
    frame_number = 0;
    eps_zero = 0.000025;

    rotation_angle = 0;

    readStream.reset();

    dict = NULL;

#if USE_AV_INTERRUPT_CALLBACK
    open_timeout = LIBAVFORMAT_INTERRUPT_OPEN_DEFAULT_TIMEOUT_MS;
    read_timeout = LIBAVFORMAT_INTERRUPT_READ_DEFAULT_TIMEOUT_MS;
#endif

    rawMode = false;
    rawModeInitialized = false;
    rawSeek = false;
    convertRGB = true;
    memset(&packet_filtered, 0, sizeof(packet_filtered));
    av_init_packet(&packet_filtered);
    bsfc = NULL;
    va_type = cv::VIDEO_ACCELERATION_NONE;  // TODO OpenCV 5.0: change to _ANY?
    hw_device = -1;
    use_opencl = 0;
    extraDataIdx = 1;
}


void CvCapture_FFMPEG::close()
{
    if( img_convert_ctx )
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = 0;
    }

    if( picture )
    {
#if LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(55, 45, 101) : CALC_FFMPEG_VERSION(55, 28, 1))
        av_frame_free(&picture);
#elif LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(54, 59, 100) : CALC_FFMPEG_VERSION(54, 28, 0))
        avcodec_free_frame(&picture);
#else
        av_free(picture);
#endif
    }

    if( video_st )
    {
#ifdef CV_FFMPEG_CODECPAR
        avcodec_close( context );
#endif
        video_st = NULL;
    }

    if (context)
    {
#ifdef CV_FFMPEG_CODECPAR
        avcodec_free_context(&context);
#endif
    }

    if( ic )
    {
        avformat_close_input(&ic);
        ic = NULL;
    }

#if USE_AV_FRAME_GET_BUFFER
    av_frame_unref(&rgb_picture);
#else
    if( rgb_picture.data[0] )
    {
        free( rgb_picture.data[0] );
        rgb_picture.data[0] = 0;
    }
#endif

    // free last packet if exist
    if (packet.data) {
        _opencv_ffmpeg_av_packet_unref (&packet);
        packet.data = NULL;
    }

    if (dict != NULL)
       av_dict_free(&dict);

    if (packet_filtered.data)
    {
        _opencv_ffmpeg_av_packet_unref(&packet_filtered);
        packet_filtered.data = NULL;
    }

    if (bsfc)
    {
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(58, 20, 100)
        av_bsf_free(&bsfc);
#else
        av_bitstream_filter_close(bsfc);
#endif
    }

    if (avio_context)
    {
        av_free(avio_context->buffer);
        av_freep(&avio_context);
    }
    readStream.reset();

    init();
}


#ifndef AVSEEK_FLAG_FRAME
#define AVSEEK_FLAG_FRAME 0
#endif
#ifndef AVSEEK_FLAG_ANY
#define AVSEEK_FLAG_ANY 1
#endif

#if defined(__OPENCV_BUILD) || defined(BUILD_PLUGIN)
typedef cv::Mutex ImplMutex;
#else
class ImplMutex
{
public:
    ImplMutex() { init(); }
    ~ImplMutex() { destroy(); }

    void init();
    void destroy();

    void lock();
    bool trylock();
    void unlock();

    struct Impl;
protected:
    Impl* impl;

private:
    ImplMutex(const ImplMutex&);
    ImplMutex& operator = (const ImplMutex& m);
};

#if defined _WIN32 || defined WINCE

struct ImplMutex::Impl
{
    void init()
    {
#if (_WIN32_WINNT >= 0x0600)
        ::InitializeCriticalSectionEx(&cs, 1000, 0);
#else
        ::InitializeCriticalSection(&cs);
#endif
        refcount = 1;
    }
    void destroy() { DeleteCriticalSection(&cs); }

    void lock() { EnterCriticalSection(&cs); }
    bool trylock() { return TryEnterCriticalSection(&cs) != 0; }
    void unlock() { LeaveCriticalSection(&cs); }

    CRITICAL_SECTION cs;
    int refcount;
};


#elif defined __APPLE__

#include <libkern/OSAtomic.h>

struct ImplMutex::Impl
{
    void init() { sl = OS_SPINLOCK_INIT; refcount = 1; }
    void destroy() { }

    void lock() { OSSpinLockLock(&sl); }
    bool trylock() { return OSSpinLockTry(&sl); }
    void unlock() { OSSpinLockUnlock(&sl); }

    OSSpinLock sl;
    int refcount;
};

#elif defined __linux__ && !defined __ANDROID__

struct ImplMutex::Impl
{
    void init() { pthread_spin_init(&sl, 0); refcount = 1; }
    void destroy() { pthread_spin_destroy(&sl); }

    void lock() { pthread_spin_lock(&sl); }
    bool trylock() { return pthread_spin_trylock(&sl) == 0; }
    void unlock() { pthread_spin_unlock(&sl); }

    pthread_spinlock_t sl;
    int refcount;
};

#else

struct ImplMutex::Impl
{
    void init() { pthread_mutex_init(&sl, 0); refcount = 1; }
    void destroy() { pthread_mutex_destroy(&sl); }

    void lock() { pthread_mutex_lock(&sl); }
    bool trylock() { return pthread_mutex_trylock(&sl) == 0; }
    void unlock() { pthread_mutex_unlock(&sl); }

    pthread_mutex_t sl;
    int refcount;
};

#endif

void ImplMutex::init()
{
    impl = new Impl();
    impl->init();
}
void ImplMutex::destroy()
{
    impl->destroy();
    delete(impl);
    impl = NULL;
}
void ImplMutex::lock() { impl->lock(); }
void ImplMutex::unlock() { impl->unlock(); }
bool ImplMutex::trylock() { return impl->trylock(); }

class AutoLock
{
public:
    AutoLock(ImplMutex& m) : mutex(&m) { mutex->lock(); }
    ~AutoLock() { mutex->unlock(); }
protected:
    ImplMutex* mutex;
private:
    AutoLock(const AutoLock&); // disabled
    AutoLock& operator = (const AutoLock&); // disabled
};
#endif


static ImplMutex _mutex;

#ifdef CV_FFMPEG_LOCKMGR
static int LockCallBack(void **mutex, AVLockOp op)
{
    ImplMutex* localMutex = reinterpret_cast<ImplMutex*>(*mutex);
    switch (op)
    {
        case AV_LOCK_CREATE:
            localMutex = new ImplMutex();
            if (!localMutex)
                return 1;
            *mutex = localMutex;
            if (!*mutex)
                return 1;
        break;

        case AV_LOCK_OBTAIN:
            localMutex->lock();
        break;

        case AV_LOCK_RELEASE:
            localMutex->unlock();
        break;

        case AV_LOCK_DESTROY:
            delete localMutex;
            localMutex = NULL;
            *mutex = NULL;
        break;
    }
    return 0;
}
#endif

static void ffmpeg_log_callback(void *ptr, int level, const char *fmt, va_list vargs)
{
    static bool skip_header = false;
    static int prev_level = -1;
    CV_UNUSED(ptr);
    if (level>av_log_get_level()) return;
    if (!skip_header || level != prev_level) printf("[OPENCV:FFMPEG:%02d] ", level);
    vprintf(fmt, vargs);
    size_t fmt_len = strlen(fmt);
    skip_header = fmt_len > 0 && fmt[fmt_len - 1] != '\n';
    prev_level = level;
}

class InternalFFMpegRegister
{
public:
    static void init(const bool threadSafe)
    {
        std::unique_lock<cv::Mutex> lock(_mutex, std::defer_lock);
        if(!threadSafe)
            lock.lock();
        static InternalFFMpegRegister instance;
    }
    static void initLogger_()
    {
        const bool debug_option = utils::getConfigurationParameterBool("OPENCV_FFMPEG_DEBUG");
        std::string level_option = utils::getConfigurationParameterString("OPENCV_FFMPEG_LOGLEVEL");
        int level = AV_LOG_VERBOSE;
        if (!level_option.empty())
        {
            level = atoi(level_option.c_str());
        }
        if ( debug_option || (!level_option.empty()) )
        {
            av_log_set_level(level);
            av_log_set_callback(ffmpeg_log_callback);
        }
        else
        {
            av_log_set_level(AV_LOG_ERROR);
        }
    }

public:
    InternalFFMpegRegister()
    {
        avformat_network_init();

#ifdef CV_FFMPEG_REGISTER
        /* register all codecs, demux and protocols */
        av_register_all();
#endif

#ifdef CV_FFMPEG_LOCKMGR
        /* register a callback function for synchronization */
        av_lockmgr_register(&LockCallBack);
#endif
        initLogger_();
    }
    ~InternalFFMpegRegister()
    {
#ifdef CV_FFMPEG_LOCKMGR
        av_lockmgr_register(NULL);
#endif
        av_log_set_callback(NULL);
    }
};

inline void fill_codec_context(AVCodecContext * enc, AVDictionary * dict)
{
    if (!enc->thread_count)
    {
        int nCpus = cv::getNumberOfCPUs();
        int requestedThreads = std::min(nCpus, 16);  // [OPENCV:FFMPEG:24] Application has requested XX threads. Using a thread count greater than 16 is not recommended.
        std::string threads_option = utils::getConfigurationParameterString("OPENCV_FFMPEG_THREADS");
        if (!threads_option.empty())
        {
            requestedThreads = atoi(threads_option.c_str());
        }
        enc->thread_count = requestedThreads;
    }

    AVDictionaryEntry* avdiscard_entry = av_dict_get(dict, "avdiscard", NULL, 0);

    if (avdiscard_entry)
    {
        if(strcmp(avdiscard_entry->value, "all") == 0)
            enc->skip_frame = AVDISCARD_ALL;
        else if (strcmp(avdiscard_entry->value, "bidir") == 0)
            enc->skip_frame = AVDISCARD_BIDIR;
        else if (strcmp(avdiscard_entry->value, "default") == 0)
            enc->skip_frame = AVDISCARD_DEFAULT;
        else if (strcmp(avdiscard_entry->value, "none") == 0)
            enc->skip_frame = AVDISCARD_NONE;
        // NONINTRA flag was introduced with version bump at revision:
        // https://github.com/FFmpeg/FFmpeg/commit/b152152df3b778d0a86dcda5d4f5d065b4175a7b
        // This key is supported only for FFMPEG version
#if LIBAVCODEC_VERSION_MICRO >= 100 && LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(55, 67, 100)
        else if (strcmp(avdiscard_entry->value, "nonintra") == 0)
            enc->skip_frame = AVDISCARD_NONINTRA;
#endif
        else if (strcmp(avdiscard_entry->value, "nonkey") == 0)
            enc->skip_frame = AVDISCARD_NONKEY;
        else if (strcmp(avdiscard_entry->value, "nonref") == 0)
            enc->skip_frame = AVDISCARD_NONREF;
    }
}

static bool isThreadSafe() {
    const bool threadSafe = utils::getConfigurationParameterBool("OPENCV_FFMPEG_IS_THREAD_SAFE", false);
    if (threadSafe) {
        CV_LOG_WARNING(NULL, "VIDEOIO/FFMPEG: OPENCV_FFMPEG_IS_THREAD_SAFE == 1, all OpenCV locks removed, relying on FFmpeg to provide thread safety.  If FFmpeg is not thread safe isOpened() may return false when multiple threads try to call open() at the same time.");
    }
    return threadSafe;
}

bool CvCapture_FFMPEG::open(const char* _filename, const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params)
{
    const bool threadSafe = isThreadSafe();
    InternalFFMpegRegister::init(threadSafe);

    std::unique_lock<cv::Mutex> lock(_mutex, std::defer_lock);
    if(!threadSafe)
        lock.lock();

    unsigned i;
    bool valid = false;
    int nThreads = 0;

    close();

    readStream = stream;

    if (!params.empty())
    {
        convertRGB = params.get<bool>(CAP_PROP_CONVERT_RGB, true);
        if (!convertRGB)
        {
            CV_LOG_WARNING(NULL, "VIDEOIO/FFMPEG: BGR conversion turned OFF, decoded frame will be "
                                 "returned in its original format. "
                                 "Multiplanar formats are not supported by the backend. "
                                 "Only GRAY8/GRAY16LE pixel formats have been tested. "
                                 "Use at your own risk.");
        }
        if (params.has(CAP_PROP_FORMAT))
        {
            int value = params.get<int>(CAP_PROP_FORMAT);
            if (value == -1)
            {
                CV_LOG_INFO(NULL, "VIDEOIO/FFMPEG: enabled demuxer only mode: '" << (_filename ? _filename : "<NULL>") << "'");
                rawMode = true;
            }
            else
            {
                CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: CAP_PROP_FORMAT parameter value is invalid/unsupported: " << value);
                return false;
            }
        }
        if(!rawMode) {
            if (params.has(CAP_PROP_HW_ACCELERATION))
            {
                va_type = params.get<VideoAccelerationType>(CAP_PROP_HW_ACCELERATION);
#if !USE_AV_HW_CODECS
                if (va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
                {
                    CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: FFmpeg backend is build without acceleration support. Can't handle CAP_PROP_HW_ACCELERATION parameter. Bailout");
                    return false;
                }
#endif
            }
            if (params.has(CAP_PROP_HW_DEVICE))
            {
                hw_device = params.get<int>(CAP_PROP_HW_DEVICE);
                if (va_type == VIDEO_ACCELERATION_NONE && hw_device != -1)
                {
                    CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: Invalid usage of CAP_PROP_HW_DEVICE without requested H/W acceleration. Bailout");
                    return false;
                }
                if (va_type == VIDEO_ACCELERATION_ANY && hw_device != -1)
                {
                    CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: Invalid usage of CAP_PROP_HW_DEVICE with 'ANY' H/W acceleration. Bailout");
                    return false;
                }
            }
            if (params.has(CAP_PROP_HW_ACCELERATION_USE_OPENCL)) {
                use_opencl = params.get<int>(CAP_PROP_HW_ACCELERATION_USE_OPENCL);
            }
        }
#if USE_AV_INTERRUPT_CALLBACK
        if (params.has(CAP_PROP_OPEN_TIMEOUT_MSEC))
        {
            open_timeout = params.get<int>(CAP_PROP_OPEN_TIMEOUT_MSEC);
        }
        if (params.has(CAP_PROP_READ_TIMEOUT_MSEC))
        {
            read_timeout = params.get<int>(CAP_PROP_READ_TIMEOUT_MSEC);
        }
#endif
        if (params.has(CAP_PROP_N_THREADS))
        {
            nThreads = params.get<int>(CAP_PROP_N_THREADS);
        }
        if (params.warnUnusedParameters())
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: unsupported parameters in .open(), see logger INFO channel for details. Bailout");
            return false;
        }
    }

#if USE_AV_INTERRUPT_CALLBACK
    /* interrupt callback */
    interrupt_metadata.timeout_after_ms = open_timeout;
    get_monotonic_time(&interrupt_metadata.value);

    ic = avformat_alloc_context();
    ic->interrupt_callback.callback = _opencv_ffmpeg_interrupt_callback;
    ic->interrupt_callback.opaque = &interrupt_metadata;
#endif

    std::string options = utils::getConfigurationParameterString("OPENCV_FFMPEG_CAPTURE_OPTIONS");
    if(!options.empty())
    {
#if LIBAVFORMAT_VERSION_MICRO >= 100  && LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(55, 48, 100)
        av_dict_set(&dict, "rtsp_flags", "prefer_tcp", 0);
#else
        av_dict_set(&dict, "rtsp_transport", "tcp", 0);
#endif
    }
    else
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO/FFMPEG: using capture options from environment: " << options);
#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 ? CALC_FFMPEG_VERSION(52, 17, 100) : CALC_FFMPEG_VERSION(52, 7, 0))
        av_dict_parse_string(&dict, options.c_str(), ";", "|", 0);
#else
        av_dict_set(&dict, "rtsp_transport", "tcp", 0);
#endif
    }
    CV_FFMPEG_FMT_CONST AVInputFormat* input_format = NULL;
    AVDictionaryEntry* entry = av_dict_get(dict, "input_format", NULL, 0);
    if (entry != 0)
    {
      input_format = av_find_input_format(entry->value);
    }

    if (!_filename)
    {
        size_t avio_ctx_buffer_size = 4096;
        uint8_t* avio_ctx_buffer = (uint8_t*)av_malloc(avio_ctx_buffer_size);
        CV_Assert(avio_ctx_buffer);
        avio_context = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0, this,
            [](void *opaque, uint8_t *buf, int buf_size) -> int {
                try {
                    auto capture = reinterpret_cast<CvCapture_FFMPEG*>(opaque);
                    auto is = capture->readStream;
                    int result = (int)is->read(reinterpret_cast<char*>(buf), buf_size);

                    // https://github.com/FFmpeg/FFmpeg/commit/858db4b01fa2b55ee55056c033054ca54ac9b0fd#diff-863c87afc9bb02fe42d071015fc8218972c80b146d603239f20b483ad0988ae9R394
                    // https://github.com/FFmpeg/FFmpeg/commit/a606f27f4c610708fa96e35eed7b7537d3d8f712
                    // https://github.com/FFmpeg/FFmpeg/blob/n4.0/libavformat/version.h#L83C41-L83C73
#if (LIBAVFORMAT_VERSION_MAJOR >= 58) && (LIBAVFORMAT_VERSION_MICRO >= 100)  // FFmpeg n4.0+
                    if (result == 0 && buf_size > 0)
                    {
                        result = AVERROR_EOF;
                    }
#endif

                    CV_LOG_VERBOSE(NULL, 0, "FFMPEG: IStreamReader::read(" << buf_size << ") = " << result);
                    return result;
                } catch (...) {
                    CV_LOG_WARNING(NULL, "FFMPEG: IStreamReader::read(" << buf_size << ") failed");
                    return 0;
                }
            },
            NULL,
            [](void *opaque, int64_t offset, int whence) -> int64_t {
                try {
                    int64_t result = -1;
                    auto capture = reinterpret_cast<CvCapture_FFMPEG*>(opaque);
                    auto is = capture->readStream;
                    int origin = whence & (~AVSEEK_FORCE);
                    if (origin == SEEK_SET || origin == SEEK_CUR || origin == SEEK_END)
                    {
                        result = is->seek(offset, origin);
                    }
                    CV_LOG_VERBOSE(NULL, 0, "FFMPEG: IStreamReader::seek(" << offset << ", whence=" << whence << ") = " << result);
                    return result;
                } catch (...) {
                    CV_LOG_WARNING(NULL, "FFMPEG: IStreamReader::seek(" << offset << ", whence=" << whence << ") failed");
                    return -1;
                }
            });
        CV_Assert(avio_context);
        ic->pb = avio_context;
    }
    int err = avformat_open_input(&ic, _filename, input_format, &dict);

    if (err < 0)
    {
        CV_WARN("Error opening file");
        CV_WARN(_filename);
        goto exit_func;
    }
    err = avformat_find_stream_info(ic, NULL);
    if (err < 0)
    {
        CV_LOG_WARNING(NULL, "Unable to read codec parameters from stream (" << _opencv_ffmpeg_get_error_string(err) << ")");
        goto exit_func;
    }

    if (rawMode) {
        video_stream = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (video_stream < 0) {
            close();
            return false;
        }
        video_st = ic->streams[video_stream];
#ifndef CV_FFMPEG_CODECPAR
        frame.height = video_st->codec->height;
        frame.width = video_st->codec->width;
#else
        frame.height = video_st->codecpar->height;
        frame.width = video_st->codecpar->width;
#endif
        return true;
    }
    for(i = 0; i < ic->nb_streams; i++)
    {
#ifndef CV_FFMPEG_CODECPAR
        context = ic->streams[i]->codec;
        AVCodecID codec_id = context->codec_id;
        AVMediaType codec_type = context->codec_type;
#else
        AVCodecParameters* par = ic->streams[i]->codecpar;
        AVCodecID codec_id = par->codec_id;
        AVMediaType codec_type = par->codec_type;
#endif

        if( AVMEDIA_TYPE_VIDEO == codec_type && video_stream < 0)
        {
            // backup encoder' width/height
#ifndef CV_FFMPEG_CODECPAR
            int enc_width = context->width;
            int enc_height = context->height;
#else
            int enc_width = par->width;
            int enc_height = par->height;
#endif

            CV_LOG_DEBUG(NULL, "FFMPEG: stream[" << i << "] is video stream with codecID=" << (int)codec_id
                    << " width=" << enc_width
                    << " height=" << enc_height
            );


#if !USE_AV_HW_CODECS
            va_type = VIDEO_ACCELERATION_NONE;
#endif

            // find and open decoder, try HW acceleration types specified in 'hw_acceleration' list (in order)
            const AVCodec *codec = NULL;
            err = -1;
#if USE_AV_HW_CODECS
            HWAccelIterator accel_iter(va_type, false/*isEncoder*/, dict);
            while (accel_iter.good())
            {
#else
            do {
#endif
#if USE_AV_HW_CODECS
                accel_iter.parse_next();
                AVHWDeviceType hw_type = accel_iter.hw_type();
                if (hw_type != AV_HWDEVICE_TYPE_NONE)
                {
                    CV_LOG_DEBUG(NULL, "FFMPEG: trying to configure H/W acceleration: '" << accel_iter.hw_type_device_string() << "'");
                    AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
                    codec = hw_find_codec(codec_id, hw_type, av_codec_is_decoder, accel_iter.disabled_codecs().c_str(), &hw_pix_fmt);
                    if (codec)
                    {
#ifdef CV_FFMPEG_CODECPAR
                        context = avcodec_alloc_context3(codec);
#endif
                        CV_Assert(context);
                        context->get_format = avcodec_default_get_format;
                        if (context->hw_device_ctx) {
                            av_buffer_unref(&context->hw_device_ctx);
                        }
                        if (hw_pix_fmt != AV_PIX_FMT_NONE)
                            context->get_format = hw_get_format_callback; // set callback to select HW pixel format, not SW format
                        context->hw_device_ctx = hw_create_device(hw_type, hw_device, accel_iter.device_subname(), use_opencl != 0);
                        if (!context->hw_device_ctx)
                        {
                            context->get_format = avcodec_default_get_format;
                            CV_LOG_DEBUG(NULL, "FFMPEG: ... can't create H/W device: '" << accel_iter.hw_type_device_string() << "'");
                            codec = NULL;
                        }
                    }
                }
                else if (hw_type == AV_HWDEVICE_TYPE_NONE)
#endif // USE_AV_HW_CODECS
                {
                    AVDictionaryEntry* video_codec_param = av_dict_get(dict, "video_codec", NULL, 0);
                    if (video_codec_param == NULL)
                    {
                        codec = avcodec_find_decoder(codec_id);
                        if (!codec)
                        {
                            CV_LOG_ERROR(NULL, "Could not find decoder for codec_id=" << (int)codec_id);
                        }
                    }
                    else
                    {
                        CV_LOG_DEBUG(NULL, "FFMPEG: Using video_codec='" << video_codec_param->value << "'");
                        codec = avcodec_find_decoder_by_name(video_codec_param->value);
                        if (!codec)
                        {
                            CV_LOG_ERROR(NULL, "Could not find decoder '" << video_codec_param->value << "'");
                        }
                    }
                    if (codec)
                    {
#ifdef CV_FFMPEG_CODECPAR
                        context = avcodec_alloc_context3(codec);
#endif
                        CV_Assert(context);
                    }
                }
                if (!codec)
                {
#ifdef CV_FFMPEG_CODECPAR
                    avcodec_free_context(&context);
#endif
                    continue;
                }
                context->thread_count = nThreads;
                fill_codec_context(context, dict);
#ifdef CV_FFMPEG_CODECPAR
                avcodec_parameters_to_context(context, par);
#endif
                err = avcodec_open2(context, codec, NULL);
                if (err >= 0) {
#if USE_AV_HW_CODECS
                    va_type = hw_type_to_va_type(hw_type);
                    if (hw_type != AV_HWDEVICE_TYPE_NONE && hw_device < 0)
                        hw_device = 0;
#endif
                    break;
                } else {
                    CV_LOG_ERROR(NULL, "Could not open codec " << codec->name << ", error: " << err);
                }
#if USE_AV_HW_CODECS
            }  // while (accel_iter.good())
#else
            } while (0);
#endif
            if (err < 0) {
                CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: Failed to initialize VideoCapture");
                goto exit_func;
            }

            // checking width/height (since decoder can sometimes alter it, eg. vp6f)
            if (enc_width && (context->width != enc_width))
                context->width = enc_width;
            if (enc_height && (context->height != enc_height))
                context->height = enc_height;

            video_stream = i;
            video_st = ic->streams[i];
#if LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(55, 45, 101) : CALC_FFMPEG_VERSION(55, 28, 1))
            picture = av_frame_alloc();
#else
            picture = avcodec_alloc_frame();
#endif

            frame.width = context->width;
            frame.height = context->height;
            frame.step = 0;
            frame.data = NULL;
            get_rotation_angle();
            break;
        }
    }

    if (video_stream >= 0)
        valid = true;

exit_func:

#if USE_AV_INTERRUPT_CALLBACK
    // deactivate interrupt callback
    interrupt_metadata.timeout_after_ms = 0;
#endif

    if( !valid )
        close();

    return valid;
}

bool CvCapture_FFMPEG::setRaw()
{
    if (!rawMode)
    {
        if (frame_number != 0)
        {
            CV_WARN("Incorrect usage: do not grab frames before .set(CAP_PROP_FORMAT, -1)");
        }
        // binary stream filter creation is moved into processRawPacket()
        rawMode = true;
    }
    return true;
}

static inline bool h26xContainer(const char* formatLongName) {
    return !strcmp(formatLongName, "QuickTime / MOV") || !strcmp(formatLongName, "FLV (Flash Video)") || !strcmp(formatLongName, "Matroska / WebM");
}

bool CvCapture_FFMPEG::processRawPacket()
{
    if (packet.data == NULL)  // EOF
        return false;
    if (!rawModeInitialized)
    {
        rawModeInitialized = true;
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(58, 20, 100)
        CV_CODEC_ID eVideoCodec = ic->streams[video_stream]->codecpar->codec_id;
#else
        CV_CODEC_ID eVideoCodec = video_st->codec->codec_id;
#endif
        const char* filterName = NULL;
        if (eVideoCodec == CV_CODEC(CODEC_ID_H264)
#if LIBAVCODEC_VERSION_MICRO >= 100 \
    && LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(57, 24, 102)  // FFmpeg 3.0
            || eVideoCodec == CV_CODEC(CODEC_ID_H265)
#elif LIBAVCODEC_VERSION_MICRO < 100 \
    && LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(55, 34, 1)  // libav v10+
            || eVideoCodec == CV_CODEC(CODEC_ID_HEVC)
#endif
        )
        {
            if(h26xContainer(ic->iformat->long_name))
                filterName = eVideoCodec == CV_CODEC(CODEC_ID_H264) ? "h264_mp4toannexb" : "hevc_mp4toannexb";
        }
        if (filterName)
        {
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(58, 20, 100)
            const AVBitStreamFilter * bsf = av_bsf_get_by_name(filterName);
            if (!bsf)
            {
                CV_WARN(cv::format("Bitstream filter is not available: %s", filterName).c_str());
                return false;
            }
            int err = av_bsf_alloc(bsf, &bsfc);
            if (err < 0)
            {
                CV_WARN("Error allocating context for bitstream buffer");
                return false;
            }
            avcodec_parameters_copy(bsfc->par_in, ic->streams[video_stream]->codecpar);
            err = av_bsf_init(bsfc);
            if (err < 0)
            {
                CV_WARN("Error initializing bitstream buffer");
                return false;
            }
#else
            bsfc = av_bitstream_filter_init(filterName);
            if (!bsfc)
            {
                CV_WARN(cv::format("Bitstream filter is not available: %s", filterName).c_str());
                return false;
            }
#endif
        }
    }
    if (bsfc)
    {
        if (packet_filtered.data)
        {
            _opencv_ffmpeg_av_packet_unref(&packet_filtered);
        }

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(58, 20, 100)
        int err = av_bsf_send_packet(bsfc, &packet);
        if (err < 0)
        {
            CV_WARN("Packet submission for filtering failed");
            return false;
        }
        err = av_bsf_receive_packet(bsfc, &packet_filtered);
        if (err < 0)
        {
            CV_WARN("Filtered packet retrieve failed");
            return false;
        }
#else
        AVCodecContext* ctx = ic->streams[video_stream]->codec;
        int err = av_bitstream_filter_filter(bsfc, ctx, NULL, &packet_filtered.data,
            &packet_filtered.size, packet.data, packet.size, packet.flags & AV_PKT_FLAG_KEY);
        if (packet.flags & AV_PKT_FLAG_KEY) packet_filtered.flags |= AV_PKT_FLAG_KEY;
        if (err < 0)
        {
            CV_WARN("Packet filtering failed");
            return false;
        }
#endif
        return packet_filtered.data != NULL;
    }
    return packet.data != NULL;
}

bool CvCapture_FFMPEG::grabFrame()
{
    if (rawSeek) {
        rawSeek = false;
        return true;
    }
    bool valid = false;

    static const size_t max_read_attempts = cv::utils::getConfigurationParameterSizeT("OPENCV_FFMPEG_READ_ATTEMPTS", 4096);
    static const size_t max_decode_attempts = cv::utils::getConfigurationParameterSizeT("OPENCV_FFMPEG_DECODE_ATTEMPTS", 64);
    size_t cur_read_attempts = 0;
    size_t cur_decode_attempts = 0;

    if( !ic || !video_st || (!rawMode && !context) )  return false;

    picture_pts = AV_NOPTS_VALUE_;

#if USE_AV_INTERRUPT_CALLBACK
    // activate interrupt callback
    interrupt_metadata.timeout = 0;
    get_monotonic_time(&interrupt_metadata.value);
    interrupt_metadata.timeout_after_ms = read_timeout;
#endif

#if USE_AV_SEND_FRAME_API
    // check if we can receive frame from previously decoded packet
    valid = rawMode ? false : avcodec_receive_frame(context, picture) >= 0;
#endif

    // get the next frame
    while (!valid)
    {

        _opencv_ffmpeg_av_packet_unref (&packet);

#if USE_AV_INTERRUPT_CALLBACK
        if (interrupt_metadata.timeout)
        {
            valid = false;
            break;
        }
#endif

        int ret = av_read_frame(ic, &packet);

        if (ret == AVERROR(EAGAIN))
            continue;

        if (ret == AVERROR_EOF)
        {
            if (rawMode)
                break;

            // flush cached frames from video decoder
            packet.data = NULL;
            packet.size = 0;
            packet.stream_index = video_stream;
        }

        if( packet.stream_index != video_stream )
        {
            _opencv_ffmpeg_av_packet_unref (&packet);
            if (++cur_read_attempts > max_read_attempts)
            {
                CV_LOG_WARNING(NULL,
                    "packet read max attempts exceeded, if your video have "
                    "multiple streams (video, audio) try to increase attempt "
                    "limit by setting environment variable OPENCV_FFMPEG_READ_ATTEMPTS "
                    "(current value is " << max_read_attempts << ")");
                break;
            }
            continue;
        }

        if (rawMode)
        {
            valid = processRawPacket();
            break;
        }

        // Decode video frame
#if USE_AV_SEND_FRAME_API
        if (avcodec_send_packet(context, &packet) < 0) {
            break;
        }
        ret = avcodec_receive_frame(context, picture);
#else
        int got_picture = 0;
        avcodec_decode_video2(context, picture, &got_picture, &packet);
        ret = got_picture ? 0 : -1;
#endif
        if (ret >= 0) {
            valid = true;
        } else if (ret == AVERROR(EAGAIN)) {
            continue;
        }
        else
        {
            if (++cur_decode_attempts > max_decode_attempts)
            {
                CV_LOG_WARNING(NULL,
                    "frame decode max attempts exceeded, try to increase attempt "
                    "limit by setting environment variable OPENCV_FFMPEG_DECODE_ATTEMPTS "
                    "(current value is " << max_decode_attempts << ")");
                break;
            }
        }
    }

    if (valid) {
        if (picture_pts == AV_NOPTS_VALUE_) {
            int64_t dts = 0;
            if (!rawMode) {
                picture_pts = picture->CV_FFMPEG_PTS_FIELD != AV_NOPTS_VALUE_ && picture->CV_FFMPEG_PTS_FIELD != 0 ? picture->CV_FFMPEG_PTS_FIELD : picture->pkt_dts;
                if(frame_number == 0) dts = picture->pkt_dts;
            }
            else {
                const AVPacket& packet_raw = packet.data != 0 ? packet : packet_filtered;
                picture_pts = packet_raw.pts != AV_NOPTS_VALUE_ && packet_raw.pts != 0 ? packet_raw.pts : packet_raw.dts;
                if (frame_number == 0) dts = packet_raw.dts;
                if (picture_pts < 0) picture_pts = 0;
            }
#if LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(54, 1, 0) || LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(52, 111, 0)
            AVRational frame_rate = video_st->avg_frame_rate;
#else
            AVRational frame_rate = video_st->r_frame_rate;
#endif
            if (picture_pts != AV_NOPTS_VALUE_)
                pts_in_fps_time_base = av_rescale_q(picture_pts, video_st->time_base, AVRational{ frame_rate.den, frame_rate.num });
            if (frame_number == 0 && dts != AV_NOPTS_VALUE_)
                dts_delay_in_fps_time_base = -av_rescale_q(dts, video_st->time_base, AVRational{ frame_rate.den, frame_rate.num });
            frame_number++;
        }
    }

    if (valid && first_frame_number < 0)
        first_frame_number = dts_to_frame_number(picture_pts);

#if USE_AV_INTERRUPT_CALLBACK
    // deactivate interrupt callback
    interrupt_metadata.timeout_after_ms = 0;
#endif

    // return if we have a new frame or not
    return valid;
}

bool CvCapture_FFMPEG::retrieveFrame(int flag, unsigned char** data, int* step, int* width, int* height, int* cn, int* depth)
{
    if (!video_st || (!rawMode && !context))
        return false;

    if (rawMode || flag == extraDataIdx)
    {
        bool ret = true;
        if (flag == 0) {
            AVPacket& p = bsfc ? packet_filtered : packet;
            *data = p.data;
            *step = p.size;
            ret = p.data != NULL;
        }
        else if (flag == extraDataIdx) {
            *data = ic->streams[video_stream]->CV_FFMPEG_CODEC_FIELD->extradata;
            *step = ic->streams[video_stream]->CV_FFMPEG_CODEC_FIELD->extradata_size;
        }
        *width = *step;
        *height = 1;
        *cn = 1;
        *depth = CV_8U;
        return  ret;
    }

    AVFrame* sw_picture = picture;
#if USE_AV_HW_CODECS
    // if hardware frame, copy it to system memory
    if (picture && picture->hw_frames_ctx) {
        sw_picture = av_frame_alloc();
        //if (av_hwframe_map(sw_picture, picture, AV_HWFRAME_MAP_READ) < 0) {
        if (av_hwframe_transfer_data(sw_picture, picture, 0) < 0) {
            CV_LOG_ERROR(NULL, "Error copying data from GPU to CPU (av_hwframe_transfer_data)");
            return false;
        }
    }
#endif

    if (!sw_picture || !sw_picture->data[0])
        return false;

    CV_LOG_DEBUG(NULL, "Input picture format: " << av_get_pix_fmt_name((AVPixelFormat)sw_picture->format));
    const AVPixelFormat result_format = convertRGB ? AV_PIX_FMT_BGR24 : (AVPixelFormat)sw_picture->format;
    switch (result_format)
    {
    case AV_PIX_FMT_BGR24: *depth = CV_8U; *cn = 3; break;
    case AV_PIX_FMT_GRAY8: *depth = CV_8U; *cn = 1; break;
    case AV_PIX_FMT_GRAY16LE: *depth = CV_16U; *cn = 1; break;
    default:
        CV_LOG_WARNING(NULL, "Unknown/unsupported picture format: " << av_get_pix_fmt_name(result_format)
                       << ", will be treated as 8UC1.");
        *depth = CV_8U;
        *cn = 1;
        break; // TODO: return false?
    }

    if( img_convert_ctx == NULL ||
        frame.width != video_st->CV_FFMPEG_CODEC_FIELD->width ||
        frame.height != video_st->CV_FFMPEG_CODEC_FIELD->height ||
        frame.data == NULL )
    {
        // Some sws_scale optimizations have some assumptions about alignment of data/step/width/height
        // Also we use coded_width/height to workaround problem with legacy ffmpeg versions (like n0.8)
        int buffer_width = context->coded_width, buffer_height = context->coded_height;

        img_convert_ctx = sws_getCachedContext(
                img_convert_ctx,
                buffer_width, buffer_height,
                (AVPixelFormat)sw_picture->format,
                buffer_width, buffer_height,
                result_format,
                SWS_BICUBIC,
                NULL, NULL, NULL
                );

        if (img_convert_ctx == NULL)
            return false;//CV_Error(0, "Cannot initialize the conversion context!");

#if USE_AV_FRAME_GET_BUFFER
        av_frame_unref(&rgb_picture);
        rgb_picture.format = result_format;
        rgb_picture.width = buffer_width;
        rgb_picture.height = buffer_height;
        if (0 != av_frame_get_buffer(&rgb_picture, 32))
        {
            CV_WARN("OutOfMemory");
            return false;
        }
#else
        int aligns[AV_NUM_DATA_POINTERS];
        avcodec_align_dimensions2(video_st->codec, &buffer_width, &buffer_height, aligns);
        rgb_picture.data[0] = (uint8_t*)realloc(rgb_picture.data[0],
                _opencv_ffmpeg_av_image_get_buffer_size( result_format,
                                    buffer_width, buffer_height ));
        _opencv_ffmpeg_av_image_fill_arrays(&rgb_picture, rgb_picture.data[0],
                        result_format, buffer_width, buffer_height );
#endif
        frame.width = video_st->CV_FFMPEG_CODEC_FIELD->width;
        frame.height = video_st->CV_FFMPEG_CODEC_FIELD->height;
        frame.data = rgb_picture.data[0];
        frame.step = rgb_picture.linesize[0];
    }

    sws_scale(
            img_convert_ctx,
            sw_picture->data,
            sw_picture->linesize,
            0, sw_picture->height,
            rgb_picture.data,
            rgb_picture.linesize
            );

    *data = frame.data;
    *step = frame.step;
    *width = frame.width;
    *height = frame.height;

#if USE_AV_HW_CODECS
    if (sw_picture != picture)
    {
        av_frame_free(&sw_picture);
    }
#endif
    return true;
}

bool CvCapture_FFMPEG::retrieveHWFrame(cv::OutputArray output)
{
#if USE_AV_HW_CODECS
    // check that we have HW frame in GPU memory
    if (!picture || !picture->hw_frames_ctx || !context) {
        return false;
    }

    // GPU color conversion NV12->BGRA, from GPU media buffer to GPU OpenCL buffer
    return hw_copy_frame_to_umat(context->hw_device_ctx, picture, output);
#else
    CV_UNUSED(output);
    return false;
#endif
}

static inline double getCodecTag(const AVCodecID codec_id) {
    const struct AVCodecTag* fallback_tags[] = {
        // APIchanges:
        // 2012-01-31 - dd6d3b0 - lavf 54.01.0
        //   Add avformat_get_riff_video_tags() and avformat_get_riff_audio_tags().
        avformat_get_riff_video_tags(),
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(55, 25, 100) && defined LIBAVFORMAT_VERSION_MICRO && LIBAVFORMAT_VERSION_MICRO >= 100
        // APIchanges: ffmpeg only
        // 2014-01-19 - 1a193c4 - lavf 55.25.100 - avformat.h
        //   Add avformat_get_mov_video_tags() and avformat_get_mov_audio_tags().
        avformat_get_mov_video_tags(),
#endif
        codec_bmp_tags, // fallback for avformat < 54.1
        NULL };
    return av_codec_get_tag(fallback_tags, codec_id);
}

static inline double getCodecIdFourcc(const AVCodecID codec_id)
{
    if (codec_id == AV_CODEC_ID_NONE) return -1;
    const char* codec_fourcc = _opencv_avcodec_get_name(codec_id);
    if (!codec_fourcc || strcmp(codec_fourcc, "unknown_codec") == 0 || strlen(codec_fourcc) != 4)
        return getCodecTag(codec_id);
    return (double)CV_FOURCC(codec_fourcc[0], codec_fourcc[1], codec_fourcc[2], codec_fourcc[3]);
}

double CvCapture_FFMPEG::getProperty( int property_id ) const
{
    if( !video_st || (!rawMode && !context) ) return 0;

    switch( property_id )
    {
    case CAP_PROP_POS_MSEC:
        if (picture_pts == AV_NOPTS_VALUE_)
        {
            return 0;
        }
        return (dts_to_sec(picture_pts) * 1000);
    case CAP_PROP_POS_FRAMES:
        return (double)frame_number;
    case CAP_PROP_POS_AVI_RATIO:
        return r2d(ic->streams[video_stream]->time_base);
    case CAP_PROP_FRAME_COUNT:
        return (double)get_total_frames();
    case CAP_PROP_FRAME_WIDTH:
        return frame.width;
    case CAP_PROP_FRAME_HEIGHT:
        return frame.height;
    case CAP_PROP_FRAME_TYPE:
        return (double)av_get_picture_type_char(picture->pict_type);
    case CAP_PROP_FPS:
        return get_fps();
    case CAP_PROP_FOURCC: {
        const double fourcc = getCodecIdFourcc(video_st->CV_FFMPEG_CODEC_FIELD->codec_id);
        if (fourcc != -1) return fourcc;
        const double codec_tag = (double)video_st->CV_FFMPEG_CODEC_FIELD->codec_tag;
        if (codec_tag) return codec_tag;
        else return -1;
    }
    case CAP_PROP_SAR_NUM:
        return _opencv_ffmpeg_get_sample_aspect_ratio(ic->streams[video_stream]).num;
    case CAP_PROP_SAR_DEN:
        return _opencv_ffmpeg_get_sample_aspect_ratio(ic->streams[video_stream]).den;
    case CAP_PROP_CODEC_PIXEL_FORMAT:
    {
#ifdef CV_FFMPEG_CODECPAR
        AVPixelFormat pix_fmt = (AVPixelFormat)video_st->codecpar->format;
#else
        AVPixelFormat pix_fmt = video_st->codec->pix_fmt;
#endif
        unsigned int fourcc_tag = avcodec_pix_fmt_to_codec_tag(pix_fmt);
        return (fourcc_tag == 0) ? (double)-1 : (double)fourcc_tag;
    }
    case CAP_PROP_FORMAT:
        if (rawMode)
            return -1;
        break;
    case CAP_PROP_CONVERT_RGB:
        return convertRGB;
    case CAP_PROP_LRF_HAS_KEY_FRAME: {
        const AVPacket& p = bsfc ? packet_filtered : packet;
        return ((p.flags & AV_PKT_FLAG_KEY) != 0) ? 1 : 0;
    }
    case CAP_PROP_CODEC_EXTRADATA_INDEX:
            return extraDataIdx;
    case CAP_PROP_BITRATE:
        return static_cast<double>(get_bitrate());
    case CAP_PROP_ORIENTATION_META:
        return static_cast<double>(rotation_angle);
#if USE_AV_HW_CODECS
    case CAP_PROP_HW_ACCELERATION:
        return static_cast<double>(va_type);
    case CAP_PROP_HW_DEVICE:
        return static_cast<double>(hw_device);
    case CAP_PROP_HW_ACCELERATION_USE_OPENCL:
        return static_cast<double>(use_opencl);
#endif  // USE_AV_HW_CODECS
    case CAP_PROP_STREAM_OPEN_TIME_USEC:
        //ic->start_time_realtime is in microseconds
        return ((double)ic->start_time_realtime);
    case CAP_PROP_N_THREADS:
        if (!rawMode)
            return static_cast<double>(context->thread_count);
        break;
    case CAP_PROP_PTS:
        return static_cast<double>(pts_in_fps_time_base);
    case CAP_PROP_DTS_DELAY:
        return static_cast<double>(dts_delay_in_fps_time_base);
    default:
        break;
    }

    return 0;
}

double CvCapture_FFMPEG::r2d(AVRational r) const
{
    return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}

double CvCapture_FFMPEG::get_duration_sec() const
{
    double sec = (double)ic->duration / (double)AV_TIME_BASE;

    if (sec < eps_zero)
    {
        sec = (double)ic->streams[video_stream]->duration * r2d(ic->streams[video_stream]->time_base);
    }

    return sec;
}

int64_t CvCapture_FFMPEG::get_bitrate() const
{
    return ic->bit_rate / 1000;
}

double CvCapture_FFMPEG::get_fps() const
{
#if LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(54, 1, 0) || LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(52, 111, 0)
    double fps = r2d(ic->streams[video_stream]->avg_frame_rate);
#else
    double fps = r2d(ic->streams[video_stream]->r_frame_rate);
#endif

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(55, 1, 100) && LIBAVFORMAT_VERSION_MICRO >= 100
    if (fps < eps_zero)
    {
        fps = r2d(av_guess_frame_rate(ic, ic->streams[video_stream], NULL));
    }
#endif

    if (fps < eps_zero)
    {
        fps = 1.0 / r2d(ic->streams[video_stream]->time_base);
    }

    return fps;
}

int64_t CvCapture_FFMPEG::get_total_frames() const
{
    int64_t nbf = ic->streams[video_stream]->nb_frames;

    if (nbf == 0)
    {
        nbf = (int64_t)floor(get_duration_sec() * get_fps() + 0.5);
    }
    return nbf;
}

int64_t CvCapture_FFMPEG::dts_to_frame_number(int64_t dts)
{
    double sec = dts_to_sec(dts);
    return (int64_t)(get_fps() * sec + 0.5);
}

double CvCapture_FFMPEG::dts_to_sec(int64_t dts) const
{
    return (double)(dts - ic->streams[video_stream]->start_time) *
        r2d(ic->streams[video_stream]->time_base);
}

void CvCapture_FFMPEG::get_rotation_angle()
{
    rotation_angle = 0;
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(57, 68, 100)
    const uint8_t *data = 0;
    data = av_stream_get_side_data(video_st, AV_PKT_DATA_DISPLAYMATRIX, NULL);
    if (data)
    {
        rotation_angle = -cvRound(av_display_rotation_get((const int32_t*)data));
        if (rotation_angle < 0)
            rotation_angle += 360;
    }
#elif LIBAVUTIL_BUILD >= CALC_FFMPEG_VERSION(52, 94, 100)
    AVDictionaryEntry *rotate_tag = av_dict_get(video_st->metadata, "rotate", NULL, 0);
    if (rotate_tag != NULL)
        rotation_angle = atoi(rotate_tag->value);
#endif
}

void CvCapture_FFMPEG::seek(int64_t _frame_number)
{
    if (!rawMode) {
        CV_Assert(context);
    }
    _frame_number = std::min(_frame_number, get_total_frames());
    int delta = !rawMode ? 16 : 0;

    // if we have not grabbed a single frame before first seek, let's read the first frame
    // and get some valuable information during the process
    if( first_frame_number < 0 && get_total_frames() > 1 )
        grabFrame();

    for(;;)
    {
        int64_t _frame_number_temp = std::max(_frame_number-delta, (int64_t)0);
        double sec = (double)_frame_number_temp / get_fps();
        int64_t time_stamp = ic->streams[video_stream]->start_time;
        double  time_base  = r2d(ic->streams[video_stream]->time_base);
        time_stamp += (int64_t)(sec / time_base + 0.5);
        if (get_total_frames() > 1) av_seek_frame(ic, video_stream, time_stamp, AVSEEK_FLAG_BACKWARD);
        if(!rawMode)
            avcodec_flush_buffers(context);
        if( _frame_number > 0 )
        {
            grabFrame();

            if( _frame_number > 1 )
            {
                frame_number = dts_to_frame_number(picture_pts) - first_frame_number;
                if (rawMode) {
                    rawSeek = true;
                    break;
                }
                //printf("_frame_number = %d, frame_number = %d, delta = %d\n",
                //       (int)_frame_number, (int)frame_number, delta);

                if( frame_number < 0 || frame_number > _frame_number-1 )
                {
                    if( _frame_number_temp == 0 || delta >= INT_MAX/4 )
                        break;
                    delta = delta < 16 ? delta*2 : delta*3/2;
                    continue;
                }
                while( frame_number < _frame_number-1 )
                {
                    if(!grabFrame())
                        break;
                }
                frame_number++;
                break;
            }
            else
            {
                frame_number = 1;
                break;
            }
        }
        else
        {
            frame_number = 0;
            break;
        }
    }
}

void CvCapture_FFMPEG::seek(double sec)
{
    seek((int64_t)(sec * get_fps() + 0.5));
}

bool CvCapture_FFMPEG::setProperty( int property_id, double value )
{
    if( !video_st ) return false;

    switch( property_id )
    {
    case CAP_PROP_POS_MSEC:
    case CAP_PROP_POS_FRAMES:
    case CAP_PROP_POS_AVI_RATIO:
        {
            switch( property_id )
            {
            case CAP_PROP_POS_FRAMES:
                seek((int64_t)value);
                break;

            case CAP_PROP_POS_MSEC:
                seek(value/1000.0);
                break;

            case CAP_PROP_POS_AVI_RATIO:
                seek((int64_t)(value*ic->duration));
                break;
            }

            picture_pts=(int64_t)value;
        }
        break;
    case CAP_PROP_FORMAT:
        if (value == -1)
            return setRaw();
        return false;
    case CAP_PROP_CONVERT_RGB:
        convertRGB = (value != 0);
        return true;
    default:
        return false;
    }

    return true;
}


///////////////// FFMPEG CvVideoWriter implementation //////////////////////////
struct CvVideoWriter_FFMPEG
{
    bool open( const char* filename, int fourcc,
               double fps, int width, int height, const VideoWriterParameters& params );
    void close();
    bool writeFrame( const unsigned char* data, int step, int width, int height, int cn, int origin );
    bool writeHWFrame(cv::InputArray input);
    double getProperty(int propId) const;
    bool setProperty(int, double);

    void init();

    CV_FFMPEG_FMT_CONST AVOutputFormat  * fmt;
    AVFormatContext * oc;
    uint8_t         * outbuf;
    uint32_t          outbuf_size;
    FILE            * outfile;
    AVFrame         * picture;
    AVFrame         * input_picture;
    uint8_t         * picbuf;
    AVStream        * video_st;
    AVCodecContext  * context;
    AVPixelFormat     input_pix_fmt;
    unsigned char   * aligned_input;
    size_t            aligned_input_size;
    int               frame_width, frame_height;
    int               frame_idx;
    bool              ok;
    struct SwsContext *img_convert_ctx;
    VideoAccelerationType va_type;
    int               hw_device;
    int               use_opencl;
    bool              encode_video;
    int               idr_period;
    bool              key_frame;
    int               pts_index;
    int               b_frame_dts_delay;
};

static const char * icvFFMPEGErrStr(int err)
{
    switch(err) {
    case AVERROR_BSF_NOT_FOUND:
        return "Bitstream filter not found";
    case AVERROR_DECODER_NOT_FOUND:
        return "Decoder not found";
    case AVERROR_DEMUXER_NOT_FOUND:
        return "Demuxer not found";
    case AVERROR_ENCODER_NOT_FOUND:
        return "Encoder not found";
    case AVERROR_EOF:
        return "End of file";
    case AVERROR_EXIT:
        return "Immediate exit was requested; the called function should not be restarted";
    case AVERROR_FILTER_NOT_FOUND:
        return "Filter not found";
    case AVERROR_INVALIDDATA:
        return "Invalid data found when processing input";
    case AVERROR_MUXER_NOT_FOUND:
        return "Muxer not found";
    case AVERROR_OPTION_NOT_FOUND:
        return "Option not found";
    case AVERROR_PATCHWELCOME:
        return "Not yet implemented in FFmpeg, patches welcome";
    case AVERROR_PROTOCOL_NOT_FOUND:
        return "Protocol not found";
    case AVERROR_STREAM_NOT_FOUND:
        return "Stream not found";
    default:
        break;
    }

    return "Unspecified error";
}

/* function internal to FFMPEG (libavformat/riff.c) to lookup codec id by fourcc tag*/
extern "C" {
    enum CV_CODEC_ID codec_get_bmp_id(unsigned int tag);
}

void CvVideoWriter_FFMPEG::init()
{
    fmt = 0;
    oc = 0;
    outbuf = 0;
    outbuf_size = 0;
    outfile = 0;
    picture = 0;
    input_picture = 0;
    picbuf = 0;
    video_st = 0;
    context = 0;
    input_pix_fmt = AV_PIX_FMT_NONE;
    aligned_input = NULL;
    aligned_input_size = 0;
    img_convert_ctx = 0;
    frame_width = frame_height = 0;
    frame_idx = 0;
    va_type = VIDEO_ACCELERATION_NONE;
    hw_device = -1;
    use_opencl = 0;
    ok = false;
    encode_video = true;
    idr_period = 0;
    key_frame = false;
    pts_index = -1;
    b_frame_dts_delay = 0;
}

/**
 * the following function is a modified version of code
 * found in ffmpeg-0.4.9-pre1/output_example.c
 */
static AVFrame * icv_alloc_picture_FFMPEG(int pix_fmt, int width, int height, bool alloc)
{
    AVFrame * picture;
    uint8_t * picture_buf = 0;
    int size;

#if LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(55, 45, 101) : CALC_FFMPEG_VERSION(55, 28, 1))
    picture = av_frame_alloc();
#else
    picture = avcodec_alloc_frame();
#endif
    if (!picture)
        return NULL;

    picture->format = pix_fmt;
    picture->width = width;
    picture->height = height;

    size = _opencv_ffmpeg_av_image_get_buffer_size( (AVPixelFormat) pix_fmt, width, height);
    if(alloc){
        picture_buf = (uint8_t *) malloc(size);
        if (!picture_buf)
        {
            av_free(picture);
            return NULL;
        }
        _opencv_ffmpeg_av_image_fill_arrays(picture, picture_buf,
                       (AVPixelFormat) pix_fmt, width, height);
    }

    return picture;
}

/* configure video stream */
static AVCodecContext * icv_configure_video_stream_FFMPEG(AVFormatContext *oc,
                                                   AVStream *st,
                                                   const AVCodec* codec,
                                                   int w, int h, int bitrate,
                                                   double fps, AVPixelFormat pixel_format, int fourcc, AVCodecID codec_id)
{
#ifdef CV_FFMPEG_CODECPAR
    AVCodecContext *c = avcodec_alloc_context3(codec);
#else
    AVCodecContext *c = st->codec;
#endif
    CV_Assert(c);

    int frame_rate, frame_rate_base;

    c->codec_id = codec ? codec->id : codec_id;

#ifndef CV_FFMPEG_CODECPAR
    // Set per-codec defaults
    CV_CODEC_ID c_id = c->codec_id;
    avcodec_get_context_defaults3(c, codec);
    // avcodec_get_context_defaults3 erases codec_id for some reason
    c->codec_id = c_id;
#endif

    c->codec_type = AVMEDIA_TYPE_VIDEO;
    c->codec_tag = fourcc;

    /* put sample parameters */
    int64_t lbit_rate = (int64_t)bitrate;
    lbit_rate += (bitrate / 2);
    lbit_rate = std::min(lbit_rate, (int64_t)INT_MAX);
    c->bit_rate = lbit_rate;

    // took advice from
    // http://ffmpeg-users.933282.n4.nabble.com/warning-clipping-1-dct-coefficients-to-127-127-td934297.html
    c->qmin = 3;

    /* resolution must be a multiple of two */
    c->width = w;
    c->height = h;

    /* time base: this is the fundamental unit of time (in seconds) in terms
       of which frame timestamps are represented. for fixed-fps content,
       timebase should be 1/framerate and timestamp increments should be
       identically 1. */
    frame_rate=(int)(fps+0.5);
    frame_rate_base=1;
    while (fabs(((double)frame_rate/frame_rate_base) - fps) > 0.001){
        frame_rate_base*=10;
        frame_rate=(int)(fps*frame_rate_base + 0.5);
    }
    c->time_base.den = frame_rate;
    c->time_base.num = frame_rate_base;
    /* adjust time base for supported framerates */
    if(codec && codec->supported_framerates){
        const AVRational *p= codec->supported_framerates;
        AVRational req = {frame_rate, frame_rate_base};
        const AVRational *best=NULL;
        AVRational best_error= {INT_MAX, 1};
        for(; p->den!=0; p++){
            AVRational error= av_sub_q(req, *p);
            if(error.num <0) error.num *= -1;
            if(av_cmp_q(error, best_error) < 0){
                best_error= error;
                best= p;
            }
        }
        if (best == NULL)
        {
#ifdef CV_FFMPEG_CODECPAR
            avcodec_free_context(&c);
#endif
            return NULL;
        }
        c->time_base.den= best->num;
        c->time_base.num= best->den;
    }

    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = pixel_format;
    if (c->codec_id == CV_CODEC(CODEC_ID_MPEG2VIDEO)) {
        c->max_b_frames = 2;
    }
    if (c->codec_id == CV_CODEC(CODEC_ID_MPEG1VIDEO) || c->codec_id == CV_CODEC(CODEC_ID_MSMPEG4V3)){
        /* needed to avoid using macroblocks in which some coeffs overflow
           this doesn't happen with normal video, it just happens here as the
           motion of the chroma plane doesn't match the luma plane */
        /* avoid FFMPEG warning 'clipping 1 dct coefficients...' */
        c->mb_decision=2;
    }

    /* Some settings for libx264 encoding, restore dummy values for gop_size
     and qmin since they will be set to reasonable defaults by the libx264
     preset system. Also, use a crf encode with the default quality rating,
     this seems easier than finding an appropriate default bitrate. */
    if (c->codec_id == AV_CODEC_ID_H264) {
      c->gop_size = -1;
      c->qmin = -1;
      c->bit_rate = 0;
      if (c->priv_data)
          av_opt_set(c->priv_data,"crf","23", 0);
    }

    // some formats want stream headers to be separate
    if(oc->oformat->flags & AVFMT_GLOBALHEADER)
    {
        // flags were renamed: https://github.com/libav/libav/commit/7c6eb0a1b7bf1aac7f033a7ec6d8cacc3b5c2615
#if LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
     ? CALC_FFMPEG_VERSION(56, 60, 100) : CALC_FFMPEG_VERSION(56, 35, 0))
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#else
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;
#endif
    }

    st->avg_frame_rate = av_make_q(frame_rate, frame_rate_base);
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(55, 20, 0)
    st->time_base = c->time_base;
#endif

    return c;
}

static const int OPENCV_NO_FRAMES_WRITTEN_CODE = 1000;

static int icv_av_encapsulate_video_FFMPEG(AVFormatContext* oc, AVStream* video_st, AVCodecContext* c,
    uint8_t* data, int sz, const int frame_idx, const int pts_index, const int b_frame_dts_delay, const bool key_frame)
{
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(57, 0, 0)
    AVPacket pkt_;
    av_init_packet(&pkt_);
    AVPacket* pkt = &pkt_;
#else
    AVPacket* pkt = av_packet_alloc();
#endif
    if(key_frame)
        pkt->flags |= PKT_FLAG_KEY;
    pkt->pts = pts_index == -1 ? frame_idx : pts_index;
    pkt->dts = frame_idx - b_frame_dts_delay;
    pkt->duration = 1;
    pkt->size = sz;
    pkt->data = data;
    av_packet_rescale_ts(pkt, c->time_base, video_st->time_base);
    int ret = av_write_frame(oc, pkt);
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(57, 0, 0)
    av_packet_free(&pkt);
#endif
    return ret;
}

static int icv_av_write_frame_FFMPEG( AVFormatContext * oc, AVStream * video_st, AVCodecContext * c,
                                      uint8_t *, uint32_t,
                                      AVFrame * picture, int frame_idx)
{
    int ret = OPENCV_NO_FRAMES_WRITTEN_CODE;

#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(57, 0, 0)
    if (oc->oformat->flags & AVFMT_RAWPICTURE)
    {
        /* raw video case. The API will change slightly in the near
           futur for that */
        AVPacket pkt;
        av_init_packet(&pkt);

        pkt.flags |= PKT_FLAG_KEY;
        pkt.stream_index= video_st->index;
        pkt.data= (uint8_t *)picture;
        pkt.size= sizeof(AVPicture);

        ret = av_write_frame(oc, &pkt);
    }
    else
#endif
    {
        /* encode the image */
#if USE_AV_SEND_FRAME_API
        if (picture == NULL && frame_idx == 0) {
            ret = 0;
        } else {
            ret = avcodec_send_frame(c, picture);
            if (ret < 0)
                CV_LOG_ERROR(NULL, "Error sending frame to encoder (avcodec_send_frame)");
        }
        while (ret >= 0)
        {
            AVPacket* pkt = av_packet_alloc();
            pkt->stream_index = video_st->index;
            ret = avcodec_receive_packet(c, pkt);

            if(!ret)
            {
                av_packet_rescale_ts(pkt, c->time_base, video_st->time_base);
                ret = av_write_frame(oc, pkt);
                av_packet_free(&pkt);
                continue;
            }

            av_packet_free(&pkt);
            break;
        }
#else
        CV_UNUSED(frame_idx);
        AVPacket pkt;
        av_init_packet(&pkt);
        int got_output = 0;
        pkt.data = NULL;
        pkt.size = 0;
        ret = avcodec_encode_video2(c, &pkt, picture, &got_output);
        if (ret < 0)
            ;
        else if (got_output) {
            if (pkt.pts != (int64_t)AV_NOPTS_VALUE)
                pkt.pts = av_rescale_q(pkt.pts, c->time_base, video_st->time_base);
            if (pkt.dts != (int64_t)AV_NOPTS_VALUE)
                pkt.dts = av_rescale_q(pkt.dts, c->time_base, video_st->time_base);
            if (pkt.duration)
                pkt.duration = av_rescale_q(pkt.duration, c->time_base, video_st->time_base);
            pkt.stream_index= video_st->index;
            ret = av_write_frame(oc, &pkt);
            _opencv_ffmpeg_av_packet_unref(&pkt);
        }
        else
            ret = OPENCV_NO_FRAMES_WRITTEN_CODE;
#endif
    }
    return ret;
}

/// write a frame with FFMPEG
bool CvVideoWriter_FFMPEG::writeFrame( const unsigned char* data, int step, int width, int height, int cn, int origin )
{
    if (!encode_video) {
        CV_Assert(cn == 1 && ((width > 0 && height == 1) || (width == 1 && height > 0 && step == 1)));
        const bool set_key_frame = key_frame ? key_frame : idr_period ? frame_idx % idr_period == 0 : 1;
        bool ret = icv_av_encapsulate_video_FFMPEG(oc, video_st, context, (uint8_t*)data, width, frame_idx, pts_index, b_frame_dts_delay, set_key_frame);
        frame_idx++;
        return ret;
    }

    // check parameters
    if (input_pix_fmt == AV_PIX_FMT_BGR24) {
        if (cn != 3) {
            return false;
        }
    }
    else if (input_pix_fmt == AV_PIX_FMT_GRAY8 || input_pix_fmt == AV_PIX_FMT_GRAY16LE) {
        if (cn != 1) {
            return false;
        }
    }
    else {
        CV_LOG_WARNING(NULL, "Input data does not match selected pixel format: "
                       << av_get_pix_fmt_name(input_pix_fmt)
                       << ", number of channels: " << cn);
        CV_Assert(false);
    }

    if( (width & -2) != frame_width || (height & -2) != frame_height || !data )
        return false;
    width = frame_width;
    height = frame_height;

    // FFmpeg contains SIMD optimizations which can sometimes read data past
    // the supplied input buffer.
    // Related info: https://trac.ffmpeg.org/ticket/6763
    // 1. To ensure that doesn't happen, we pad the step to a multiple of 32
    // (that's the minimal alignment for which Valgrind doesn't raise any warnings).
    // 2. (dataend - SIMD_SIZE) and (dataend + SIMD_SIZE) is from the same 4k page
    const int CV_STEP_ALIGNMENT = 32;
    const size_t CV_SIMD_SIZE = 32;
    const size_t CV_PAGE_MASK = ~(size_t)(4096 - 1);
    const unsigned char* dataend = data + ((size_t)height * step);
    if (step % CV_STEP_ALIGNMENT != 0 ||
        (((size_t)dataend - CV_SIMD_SIZE) & CV_PAGE_MASK) != (((size_t)dataend + CV_SIMD_SIZE) & CV_PAGE_MASK))
    {
        int aligned_step = (step + CV_STEP_ALIGNMENT - 1) & ~(CV_STEP_ALIGNMENT - 1);

        size_t new_size = (aligned_step * height + CV_SIMD_SIZE);

        if (!aligned_input || aligned_input_size < new_size)
        {
            if (aligned_input)
                av_freep(&aligned_input);
            aligned_input_size = new_size;
            aligned_input = (unsigned char*)av_mallocz(aligned_input_size);
        }

        if (origin == 1)
            for( int y = 0; y < height; y++ )
                memcpy(aligned_input + y*aligned_step, data + (height-1-y)*step, step);
        else
            for( int y = 0; y < height; y++ )
                memcpy(aligned_input + y*aligned_step, data + y*step, step);

        data = aligned_input;
        step = aligned_step;
    }

    AVPixelFormat sw_pix_fmt = context->pix_fmt;
#if USE_AV_HW_CODECS
    if (context->hw_frames_ctx)
        sw_pix_fmt = ((AVHWFramesContext*)context->hw_frames_ctx->data)->sw_format;
#endif
    if ( sw_pix_fmt != input_pix_fmt ) {
        CV_Assert( input_picture );
        // let input_picture point to the raw data buffer of 'image'
        _opencv_ffmpeg_av_image_fill_arrays(input_picture, (uint8_t *) data,
                       (AVPixelFormat)input_pix_fmt, width, height);
        input_picture->linesize[0] = step;

        if( !img_convert_ctx )
        {
            img_convert_ctx = sws_getContext(width,
                                             height,
                                             (AVPixelFormat)input_pix_fmt,
                                             context->width,
                                             context->height,
                                             sw_pix_fmt,
                                             SWS_BICUBIC,
                                             NULL, NULL, NULL);
            if( !img_convert_ctx )
                return false;
        }

        if ( sws_scale(img_convert_ctx, input_picture->data,
                       input_picture->linesize, 0,
                       height,
                       picture->data, picture->linesize) < 0 )
            return false;
    }
    else{
        _opencv_ffmpeg_av_image_fill_arrays(picture, (uint8_t *) data,
                       (AVPixelFormat)input_pix_fmt, width, height);
        picture->linesize[0] = step;
    }

    bool ret;
#if USE_AV_HW_CODECS
    if (context->hw_device_ctx) {
        // copy data to HW frame
        AVFrame* hw_frame = av_frame_alloc();
        if (!hw_frame) {
            CV_LOG_ERROR(NULL, "Error allocating AVFrame (av_frame_alloc)");
            return false;
        }
        if (av_hwframe_get_buffer(context->hw_frames_ctx, hw_frame, 0) < 0) {
            CV_LOG_ERROR(NULL, "Error obtaining HW frame (av_hwframe_get_buffer)");
            av_frame_free(&hw_frame);
            return false;
        }
        if (av_hwframe_transfer_data(hw_frame, picture, 0) < 0) {
            CV_LOG_ERROR(NULL, "Error copying data from CPU to GPU (av_hwframe_transfer_data)");
            av_frame_free(&hw_frame);
            return false;
        }
        hw_frame->pts = frame_idx;
        int ret_write = icv_av_write_frame_FFMPEG(oc, video_st, context, outbuf, outbuf_size, hw_frame, frame_idx);
        ret = ret_write >= 0 ? true : false;
        av_frame_free(&hw_frame);
    } else
#endif
    {
        picture->pts = frame_idx;
        int ret_write = icv_av_write_frame_FFMPEG(oc, video_st, context, outbuf, outbuf_size, picture, frame_idx);
        ret = ret_write >= 0 ? true : false;
    }

    frame_idx++;

    return ret;
}

bool CvVideoWriter_FFMPEG::writeHWFrame(cv::InputArray input) {
#if USE_AV_HW_CODECS
    if (!video_st || !context || !context->hw_frames_ctx || !context->hw_device_ctx)
        return false;

    // Get hardware frame from frame pool
    AVFrame* hw_frame = av_frame_alloc();
    if (!hw_frame) {
        return false;
    }
    if (av_hwframe_get_buffer(context->hw_frames_ctx, hw_frame, 0) < 0) {
        av_frame_free(&hw_frame);
        return false;
    }

    // GPU to GPU copy
    if (!hw_copy_umat_to_frame(context->hw_device_ctx, input, hw_frame)) {
        av_frame_free(&hw_frame);
        return false;
    }

    // encode
    hw_frame->pts = frame_idx;
    icv_av_write_frame_FFMPEG( oc, video_st, context, outbuf, outbuf_size, hw_frame, frame_idx);
    frame_idx++;

    av_frame_free(&hw_frame);

    return true;
#else
    CV_UNUSED(input);
    return false;
#endif
}

double CvVideoWriter_FFMPEG::getProperty(int propId) const
{
    CV_UNUSED(propId);
#if USE_AV_HW_CODECS
    if (propId == VIDEOWRITER_PROP_HW_ACCELERATION)
    {
        return static_cast<double>(va_type);
    }
    else if (propId == VIDEOWRITER_PROP_HW_DEVICE)
    {
        return static_cast<double>(hw_device);
    }
    else if (propId == VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL)
    {
        return static_cast<double>(use_opencl);
    }
#endif
    return 0;
}

bool CvVideoWriter_FFMPEG::setProperty(int property_id, double value)
{
    if (!video_st) return false;

    switch (property_id)
    {
    case VIDEOWRITER_PROP_KEY_FLAG:
        key_frame = static_cast<bool>(value);
        break;
    case VIDEOWRITER_PROP_PTS:
        pts_index = static_cast<int>(value);
        break;
    case VIDEOWRITER_PROP_DTS_DELAY:
        b_frame_dts_delay = static_cast<int>(value);
        break;
    default:
        return false;
    }
    return true;
}

/// close video output stream and free associated memory
void CvVideoWriter_FFMPEG::close()
{
    /* no more frame to compress. The codec has a latency of a few
       frames if using B frames, so we get the last frames by
       passing the same picture again */
    // TODO -- do we need to account for latency here?

    /* write the trailer, if any */
    if ((!encode_video || picture) && ok && oc)
    {
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(57, 0, 0)
        if (!(oc->oformat->flags & AVFMT_RAWPICTURE))
#endif
        {
            if (encode_video) {
                for (;;)
                {
                    int ret = icv_av_write_frame_FFMPEG(oc, video_st, context, outbuf, outbuf_size, NULL, frame_idx);
                    if (ret == OPENCV_NO_FRAMES_WRITTEN_CODE || ret < 0)
                        break;
                }
            }
        }
        av_write_trailer(oc);
    }

    if( img_convert_ctx )
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = 0;
    }

    // free pictures
    if (picture && context && context->pix_fmt != input_pix_fmt)
    {
        if(picture->data[0])
            free(picture->data[0]);
        picture->data[0] = 0;
    }
    av_free(picture);

    if (input_picture)
        av_free(input_picture);

#ifdef CV_FFMPEG_CODECPAR
    avcodec_free_context(&context);
#else
    /* close codec */
    if (context)  // fixed after https://github.com/FFmpeg/FFmpeg/commit/3e1f507f3e8f16b716aa115552d243b48ae809bd
        avcodec_close(context);
    context = NULL;
#endif

    av_free(outbuf);

    if (oc)
    {
        if (!(fmt->flags & AVFMT_NOFILE))
        {
            /* close the output file */
            avio_close(oc->pb);
        }

        /* free the stream */
        avformat_free_context(oc);
    }

    av_freep(&aligned_input);

    init();
}

#define CV_PRINTABLE_CHAR(ch) ((ch) < 32 ? '?' : (ch))
#define CV_TAG_TO_PRINTABLE_CHAR4(tag) CV_PRINTABLE_CHAR((tag) & 255), CV_PRINTABLE_CHAR(((tag) >> 8) & 255), CV_PRINTABLE_CHAR(((tag) >> 16) & 255), CV_PRINTABLE_CHAR(((tag) >> 24) & 255)

static inline bool cv_ff_codec_tag_match(const AVCodecTag *tags, CV_CODEC_ID id, unsigned int tag)
{
    while (tags->id != AV_CODEC_ID_NONE)
    {
        if (tags->id == id && tags->tag == tag)
            return true;
        tags++;
    }
    return false;
}

static inline bool cv_ff_codec_tag_list_match(const AVCodecTag *const *tags, CV_CODEC_ID id, unsigned int tag)
{
    int i;
    for (i = 0; tags && tags[i]; i++) {
        bool res = cv_ff_codec_tag_match(tags[i], id, tag);
        if (res)
            return res;
    }
    return false;
}


static inline void cv_ff_codec_tag_dump(const AVCodecTag *const *tags)
{
    int i;
    for (i = 0; tags && tags[i]; i++) {
        const AVCodecTag * ptags = tags[i];
        while (ptags->id != AV_CODEC_ID_NONE)
        {
            unsigned int tag = ptags->tag;
            printf("fourcc tag 0x%08x/'%c%c%c%c' codec_id %04X\n", tag, CV_TAG_TO_PRINTABLE_CHAR4(tag), ptags->id);
            ptags++;
        }
    }
}

/// Create a video writer object that uses FFMPEG
bool CvVideoWriter_FFMPEG::open( const char * filename, int fourcc,
                                 double fps, int width, int height, const VideoWriterParameters& params)
{
    const bool threadSafe = isThreadSafe();
    InternalFFMpegRegister::init(threadSafe);

    std::unique_lock<cv::Mutex> lock(_mutex, std::defer_lock);
    if (!threadSafe)
        lock.lock();

    CV_CODEC_ID codec_id = CV_CODEC(CODEC_ID_NONE);
    AVPixelFormat codec_pix_fmt;
    double bitrate_scale = 1;

    close();

    encode_video = !params.get(VIDEOWRITER_PROP_RAW_VIDEO, false);
    idr_period = params.get(VIDEOWRITER_PROP_KEY_INTERVAL, 0);
    const bool is_color = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
    const int depth = params.get(VIDEOWRITER_PROP_DEPTH, CV_8U);
    const bool is_supported = depth == CV_8U || (depth == CV_16U && !is_color);
    if (!is_supported)
    {
        CV_LOG_WARNING(NULL, "Unsupported depth/isColor combination is selected, "
                             "only CV_8UC1/CV_8UC3/CV_16UC1 are supported.");
        return false;
    }
    if (params.has(VIDEOWRITER_PROP_HW_ACCELERATION))
    {
        va_type = params.get<VideoAccelerationType>(VIDEOWRITER_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE);
#if !USE_AV_HW_CODECS
        if (va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: FFmpeg backend is build without acceleration support. Can't handle VIDEOWRITER_PROP_HW_ACCELERATION parameter. Bailout");
            return false;
        }
#endif
    }
    if (params.has(VIDEOWRITER_PROP_HW_DEVICE))
    {
        hw_device = params.get<int>(VIDEOWRITER_PROP_HW_DEVICE, -1);
        if (va_type == VIDEO_ACCELERATION_NONE && hw_device != -1)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: Invalid usage of VIDEOWRITER_PROP_HW_DEVICE without requested H/W acceleration. Bailout");
            return false;
        }
        if (va_type == VIDEO_ACCELERATION_ANY && hw_device != -1)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: Invalid usage of VIDEOWRITER_PROP_HW_DEVICE with 'ANY' H/W acceleration. Bailout");
            return false;
        }
    }
    if (params.has(VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL)) {
        use_opencl = params.get<int>(VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL);
    }

    if (params.warnUnusedParameters())
    {
        CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: unsupported parameters in VideoWriter, see logger INFO channel for details");
        return false;
    }

    // check arguments
    if( !filename )
        return false;
    if(fps <= 0)
        return false;

    if (encode_video) {
        // we allow frames of odd width or height, but in this case we truncate
        // the rightmost column/the bottom row. Probably, this should be handled more elegantly,
        // but some internal functions inside FFMPEG swscale require even width/height.
        width &= -2;
        height &= -2;
        if (width <= 0 || height <= 0)
            return false;
    }

    /* auto detect the output format from the name and fourcc code. */

    fmt = av_guess_format(NULL, filename, NULL);

    if (!fmt)
        return false;

    /* determine optimal pixel format */
    if (is_color)
    {
        switch (depth)
        {
        case CV_8U: input_pix_fmt = AV_PIX_FMT_BGR24; break;
        default:
            CV_LOG_WARNING(NULL, "Unsupported input depth for color image: " << depth);
            return false;
        }
    }
    else
    {
        switch (depth)
        {
        case CV_8U: input_pix_fmt = AV_PIX_FMT_GRAY8; break;
        case CV_16U: input_pix_fmt = AV_PIX_FMT_GRAY16LE; break;
        default:
            CV_LOG_WARNING(NULL, "Unsupported input depth for grayscale image: " << depth);
            return false;
        }
    }
    CV_LOG_DEBUG(NULL, "Selected pixel format: " << av_get_pix_fmt_name(input_pix_fmt));

    if (fourcc == -1)
    {
        fprintf(stderr,"OpenCV: FFMPEG: format %s / %s\n", fmt->name, fmt->long_name);
        cv_ff_codec_tag_dump(fmt->codec_tag);
        return false;
    }

    /* Lookup codec_id for given fourcc */
    if( (codec_id = av_codec_get_id(fmt->codec_tag, fourcc)) == CV_CODEC(CODEC_ID_NONE) )
    {
        const struct AVCodecTag * fallback_tags[] = {
// APIchanges:
// 2012-01-31 - dd6d3b0 - lavf 54.01.0
//   Add avformat_get_riff_video_tags() and avformat_get_riff_audio_tags().
                avformat_get_riff_video_tags(),
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(55, 25, 100) && defined LIBAVFORMAT_VERSION_MICRO && LIBAVFORMAT_VERSION_MICRO >= 100
// APIchanges: ffmpeg only
// 2014-01-19 - 1a193c4 - lavf 55.25.100 - avformat.h
//   Add avformat_get_mov_video_tags() and avformat_get_mov_audio_tags().
                avformat_get_mov_video_tags(),
#endif
                codec_bmp_tags, // fallback for avformat < 54.1
                NULL };
        if (codec_id == CV_CODEC(CODEC_ID_NONE)) {
            codec_id = av_codec_get_id(fallback_tags, fourcc);
        }
        if (codec_id == CV_CODEC(CODEC_ID_NONE)) {
            char *p = (char *) &fourcc;
            char name[] = {(char)tolower(p[0]), (char)tolower(p[1]), (char)tolower(p[2]), (char)tolower(p[3]), 0};
            const AVCodecDescriptor *desc = avcodec_descriptor_get_by_name(name);
            if (desc)
                codec_id = desc->id;
        }

        if (codec_id == CV_CODEC(CODEC_ID_NONE))
        {
            fflush(stdout);
            fprintf(stderr, "OpenCV: FFMPEG: tag 0x%08x/'%c%c%c%c' is not found (format '%s / %s')'\n",
                    fourcc, CV_TAG_TO_PRINTABLE_CHAR4(fourcc),
                    fmt->name, fmt->long_name);
            return false;
        }
    }


    // validate tag
    if (cv_ff_codec_tag_list_match(fmt->codec_tag, codec_id, fourcc) == false)
    {
        fflush(stdout);
        fprintf(stderr, "OpenCV: FFMPEG: tag 0x%08x/'%c%c%c%c' is not supported with codec id %d and format '%s / %s'\n",
                fourcc, CV_TAG_TO_PRINTABLE_CHAR4(fourcc),
                codec_id, fmt->name, fmt->long_name);
        int supported_tag;
        if( (supported_tag = av_codec_get_tag(fmt->codec_tag, codec_id)) != 0 )
        {
            fprintf(stderr, "OpenCV: FFMPEG: fallback to use tag 0x%08x/'%c%c%c%c'\n",
                    supported_tag, CV_TAG_TO_PRINTABLE_CHAR4(supported_tag));
            fourcc = supported_tag;
        }
    }

    // alloc memory for context
    oc = avformat_alloc_context();
    CV_Assert(oc);

    /* set file name */
    oc->oformat = fmt;
#ifndef CV_FFMPEG_URL
    snprintf(oc->filename, sizeof(oc->filename), "%s", filename);
#else
    size_t name_len = strlen(filename);
    oc->url = (char*)av_malloc(name_len + 1);
    CV_Assert(oc->url);
    memcpy((void*)oc->url, filename, name_len + 1);
    oc->url[name_len] = '\0';
#endif
    /* set some options */
    oc->max_delay = (int)(0.7*AV_TIME_BASE);  /* This reduces buffer underrun warnings with MPEG */

    // set a few optimal pixel formats for lossless codecs of interest..
    switch (codec_id) {
    case CV_CODEC(CODEC_ID_JPEGLS):
        // BGR24 or GRAY8 depending on is_color...
        // supported: bgr24 rgb24 gray gray16le
        // as of version 3.4.1
        codec_pix_fmt = input_pix_fmt;
        break;
    case CV_CODEC(CODEC_ID_HUFFYUV):
        // supported: yuv422p rgb24 bgra
        // as of version 3.4.1
        switch(input_pix_fmt)
        {
            case AV_PIX_FMT_RGB24:
            case AV_PIX_FMT_BGRA:
                codec_pix_fmt = input_pix_fmt;
                break;
            case AV_PIX_FMT_BGR24:
                codec_pix_fmt = AV_PIX_FMT_RGB24;
                break;
            default:
                codec_pix_fmt = AV_PIX_FMT_YUV422P;
                break;
        }
        break;
    case CV_CODEC(CODEC_ID_PNG):
        // supported: rgb24 rgba rgb48be rgba64be pal8 gray ya8 gray16be ya16be monob
        // as of version 3.4.1
        switch(input_pix_fmt)
        {
            case AV_PIX_FMT_GRAY8:
            case AV_PIX_FMT_GRAY16BE:
            case AV_PIX_FMT_RGB24:
            case AV_PIX_FMT_BGRA:
                codec_pix_fmt = input_pix_fmt;
                break;
            case AV_PIX_FMT_GRAY16LE:
                codec_pix_fmt = AV_PIX_FMT_GRAY16BE;
                break;
            case AV_PIX_FMT_BGR24:
                codec_pix_fmt = AV_PIX_FMT_RGB24;
                break;
            default:
                codec_pix_fmt = AV_PIX_FMT_YUV422P;
                break;
        }
        break;
    case CV_CODEC(CODEC_ID_FFV1):
        // supported: MANY
        // as of version 3.4.1
        switch(input_pix_fmt)
        {
            case AV_PIX_FMT_GRAY8:
            case AV_PIX_FMT_GRAY16LE:
#ifdef AV_PIX_FMT_BGR0
            case AV_PIX_FMT_BGR0:
#endif
            case AV_PIX_FMT_BGRA:
                codec_pix_fmt = input_pix_fmt;
                break;
            case AV_PIX_FMT_GRAY16BE:
                codec_pix_fmt = AV_PIX_FMT_GRAY16LE;
                break;
            case AV_PIX_FMT_BGR24:
            case AV_PIX_FMT_RGB24:
#ifdef AV_PIX_FMT_BGR0
                codec_pix_fmt = AV_PIX_FMT_BGR0;
#else
                codec_pix_fmt = AV_PIX_FMT_BGRA;
#endif
                break;
            default:
                codec_pix_fmt = AV_PIX_FMT_YUV422P;
                break;
        }
        break;
    case CV_CODEC(CODEC_ID_MJPEG):
    case CV_CODEC(CODEC_ID_LJPEG):
        codec_pix_fmt = AV_PIX_FMT_YUVJ420P;
        bitrate_scale = 3;
        break;
    case CV_CODEC(CODEC_ID_RAWVIDEO):
        // RGBA is the only RGB fourcc supported by AVI and MKV format
        if(fourcc == CV_FOURCC('R','G','B','A'))
        {
            codec_pix_fmt = AV_PIX_FMT_RGBA;
        }
        else
        {
            switch(input_pix_fmt)
            {
                case AV_PIX_FMT_GRAY8:
                case AV_PIX_FMT_GRAY16LE:
                case AV_PIX_FMT_GRAY16BE:
                    codec_pix_fmt = input_pix_fmt;
                    break;
                default:
                    codec_pix_fmt = AV_PIX_FMT_YUV420P;
                    break;
            }
        }
        break;
    default:
        // good for lossy formats, MPEG, etc.
        codec_pix_fmt = AV_PIX_FMT_YUV420P;
        break;
    }

    double bitrate = std::min(bitrate_scale*fps*width*height, (double)INT_MAX/2);

    if (codec_id == AV_CODEC_ID_NONE) {
        codec_id = av_guess_codec(oc->oformat, NULL, filename, NULL, AVMEDIA_TYPE_VIDEO);
    }

    // Add video stream to output file
    video_st = avformat_new_stream(oc, 0);
    if (!video_st) {
        CV_WARN("Could not allocate stream");
        return false;
    }

    AVDictionary *dict = NULL;
#if (LIBAVUTIL_VERSION_MAJOR >= 53)
    std::string options = utils::getConfigurationParameterString("OPENCV_FFMPEG_WRITER_OPTIONS");
    if (!options.empty())
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO/FFMPEG: using writer options from environment: " << options);
        av_dict_parse_string(&dict, options.c_str(), ";", "|", 0);
    }
#endif

    // find and open encoder, try HW acceleration types specified in 'hw_acceleration' list (in order)
    int err = -1;
    const AVCodec* codec = NULL;
#if USE_AV_HW_CODECS
    AVBufferRef* hw_device_ctx = NULL;
    HWAccelIterator accel_iter(va_type, true/*isEncoder*/, dict);
    while (accel_iter.good())
    {
        AVPixelFormat hw_format = AV_PIX_FMT_NONE;
        AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
#else
    do {
#endif
        if (encode_video) {
#if USE_AV_HW_CODECS
            accel_iter.parse_next();
            hw_type = accel_iter.hw_type();
            codec = NULL;
            hw_format = AV_PIX_FMT_NONE;
            if (hw_device_ctx)
                av_buffer_unref(&hw_device_ctx);
            if (hw_type != AV_HWDEVICE_TYPE_NONE)
            {
                codec = hw_find_codec(codec_id, hw_type, av_codec_is_encoder, accel_iter.disabled_codecs().c_str(), &hw_format);
                if (!codec)
                    continue;

                hw_device_ctx = hw_create_device(hw_type, hw_device, accel_iter.device_subname(), use_opencl != 0);
                if (!hw_device_ctx)
                    continue;
            }
            else if (hw_type == AV_HWDEVICE_TYPE_NONE)
#endif
            {
                codec = avcodec_find_encoder(codec_id);
                if (!codec) {
                    CV_LOG_ERROR(NULL, "Could not find encoder for codec_id=" << (int)codec_id << ", error: "
                        << icvFFMPEGErrStr(AVERROR_ENCODER_NOT_FOUND));
                }
            }
            if (!codec)
                continue;
        }

#if USE_AV_HW_CODECS
            AVPixelFormat format = (hw_format != AV_PIX_FMT_NONE) ? hw_format : codec_pix_fmt;
#else
            AVPixelFormat format = codec_pix_fmt;
#endif

#ifdef CV_FFMPEG_CODECPAR
        avcodec_free_context(&context);
#endif
        context = icv_configure_video_stream_FFMPEG(oc, video_st, codec,
                                              width, height, (int) (bitrate + 0.5),
                                              fps, format, fourcc, codec_id);
        if (!context)
        {
            continue;
        }

#if 0
#if FF_API_DUMP_FORMAT
        dump_format(oc, 0, filename, 1);
#else
        av_dump_format(oc, 0, filename, 1);
#endif
#endif
        if (encode_video) {
#if USE_AV_HW_CODECS
            if (hw_device_ctx) {
                context->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                if (hw_format != AV_PIX_FMT_NONE) {
                    context->hw_frames_ctx = hw_create_frames(NULL, hw_device_ctx, width, height, hw_format);
                    if (!context->hw_frames_ctx)
                        continue;
                }
            }
#endif
        }

        int64_t lbit_rate = (int64_t) context->bit_rate;
        lbit_rate += (int64_t)(bitrate / 2);
        lbit_rate = std::min(lbit_rate, (int64_t) INT_MAX);
        context->bit_rate_tolerance = (int) lbit_rate;
        context->bit_rate = (int) lbit_rate;

        /* open the codec */
        err = !encode_video ? 0 : avcodec_open2(context, codec, NULL);
        if (err >= 0) {
#if USE_AV_HW_CODECS
            va_type = hw_type_to_va_type(hw_type);
            if (hw_type != AV_HWDEVICE_TYPE_NONE && hw_device < 0)
                hw_device = 0;
#endif
            break;
        } else {
            CV_LOG_ERROR(NULL, "Could not open codec " << codec->name << ", error: " << icvFFMPEGErrStr(err) << " (" << err << ")");
        }
#if USE_AV_HW_CODECS
    }  // while (accel_iter.good())
#else
    } while (0);
#endif

#if USE_AV_HW_CODECS
    if (hw_device_ctx)
        av_buffer_unref(&hw_device_ctx);
#endif

    if (dict != NULL)
        av_dict_free(&dict);

    if (err < 0) {
        CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: Failed to initialize VideoWriter");
        return false;
    }

#ifdef CV_FFMPEG_CODECPAR
    // Copy all to codecpar...
    // !!! https://stackoverflow.com/questions/15897849/c-ffmpeg-not-writing-avcc-box-information
    avcodec_parameters_from_context(video_st->codecpar, context);
#endif

    if (encode_video) {
        outbuf = NULL;
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(57, 0, 0)
        if (!(oc->oformat->flags & AVFMT_RAWPICTURE))
#endif
        {
            /* allocate output buffer */
            /* assume we will never get codec output with more than 4 bytes per pixel... */
            outbuf_size = width * height * 4;
            outbuf = (uint8_t*)av_malloc(outbuf_size);
        }

        bool need_color_convert;
        AVPixelFormat sw_pix_fmt = context->pix_fmt;
#if USE_AV_HW_CODECS
        if (context->hw_frames_ctx)
            sw_pix_fmt = ((AVHWFramesContext*)context->hw_frames_ctx->data)->sw_format;
#endif

        need_color_convert = (sw_pix_fmt != input_pix_fmt);

        /* allocate the encoded raw picture */
        picture = icv_alloc_picture_FFMPEG(sw_pix_fmt, context->width, context->height, need_color_convert);
        if (!picture) {
            return false;
        }

        /* if the output format is not our input format, then a temporary
       picture of the input format is needed too. It is then converted
       to the required output format */
        input_picture = NULL;
        if (need_color_convert) {
            input_picture = icv_alloc_picture_FFMPEG(input_pix_fmt, context->width, context->height, false);
            if (!input_picture) {
                return false;
            }
        }
    }

    /* open the output file, if needed */
    if (!(fmt->flags & AVFMT_NOFILE))
    {
        if (avio_open(&oc->pb, filename, AVIO_FLAG_WRITE) < 0)
        {
            return false;
        }
    }

    /* write the stream header, if any */
    err=avformat_write_header(oc, NULL);

    if(err < 0)
    {
        close();
        remove(filename);
        return false;
    }
    frame_width = width;
    frame_height = height;
    frame_idx = 0;
    ok = true;

    return true;
}



static
CvCapture_FFMPEG* cvCreateFileCaptureWithParams_FFMPEG(const char* filename, const VideoCaptureParameters& params)
{
    CvCapture_FFMPEG* capture = new CvCapture_FFMPEG();
    if (!capture)
        return 0;
    capture->init();
    if (capture->open(filename, nullptr, params))
        return capture;

    capture->close();
    delete capture;
    return 0;
}

static
CvCapture_FFMPEG* cvCreateStreamCaptureWithParams_FFMPEG(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params)
{
    CvCapture_FFMPEG* capture = new CvCapture_FFMPEG();
    if (!capture)
        return 0;
    capture->init();
    if (capture->open(nullptr, stream, params))
        return capture;

    capture->close();
    delete capture;
    return 0;
}

void cvReleaseCapture_FFMPEG(CvCapture_FFMPEG** capture)
{
    if( capture && *capture )
    {
        (*capture)->close();
        delete *capture;
        *capture = 0;
    }
}

int cvSetCaptureProperty_FFMPEG(CvCapture_FFMPEG* capture, int prop_id, double value)
{
    return capture->setProperty(prop_id, value);
}

double cvGetCaptureProperty_FFMPEG(CvCapture_FFMPEG* capture, int prop_id)
{
    return capture->getProperty(prop_id);
}

int cvGrabFrame_FFMPEG(CvCapture_FFMPEG* capture)
{
    return capture->grabFrame();
}

int cvRetrieveFrame_FFMPEG(CvCapture_FFMPEG* capture, unsigned char** data, int* step, int* width, int* height, int* cn)
{
    int depth = CV_8U;
    return cvRetrieveFrame2_FFMPEG(capture, data, step, width, height, cn, &depth);
}

int cvRetrieveFrame2_FFMPEG(CvCapture_FFMPEG* capture, unsigned char** data, int* step, int* width, int* height, int* cn, int* depth)
{
    return capture->retrieveFrame(0, data, step, width, height, cn, depth);
}

static CvVideoWriter_FFMPEG* cvCreateVideoWriterWithParams_FFMPEG( const char* filename, int fourcc, double fps,
                                                  int width, int height, const VideoWriterParameters& params )
{
    CvVideoWriter_FFMPEG* writer = new CvVideoWriter_FFMPEG();
    if (!writer)
        return 0;
    writer->init();
    if( writer->open( filename, fourcc, fps, width, height, params ))
        return writer;
    writer->close();
    delete writer;
    return 0;
}

CvVideoWriter_FFMPEG* cvCreateVideoWriter_FFMPEG( const char* filename, int fourcc, double fps,
                                                  int width, int height, int isColor )
{
    VideoWriterParameters params;
    params.add(VIDEOWRITER_PROP_IS_COLOR, isColor);
    return cvCreateVideoWriterWithParams_FFMPEG(filename, fourcc, fps, width, height, params);
}

void cvReleaseVideoWriter_FFMPEG( CvVideoWriter_FFMPEG** writer )
{
    if( writer && *writer )
    {
        (*writer)->close();
        delete *writer;
        *writer = 0;
    }
}


int cvWriteFrame_FFMPEG( CvVideoWriter_FFMPEG* writer,
                         const unsigned char* data, int step,
                         int width, int height, int cn, int origin)
{
    return writer->writeFrame(data, step, width, height, cn, origin);
}
