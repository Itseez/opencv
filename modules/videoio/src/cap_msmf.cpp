// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
/*
   Media Foundation-based Video Capturing module is based on
   videoInput library by Evgeny Pereguda:
   http://www.codeproject.com/Articles/559437/Capturing-of-video-from-web-camera-on-Windows-7-an
   Originally licensed under The Code Project Open License (CPOL) 1.02:
   http://www.codeproject.com/info/cpol10.aspx
*/
//require Windows 8 for some of the formats defined otherwise could baseline on lower version
#if WINVER < _WIN32_WINNT_WIN8
#undef WINVER
#define WINVER _WIN32_WINNT_WIN8
#endif

#include <windows.h>
#include <guiddef.h>
#include <initguid.h>
#include <mfidl.h>
#include <mfapi.h>
#include <mfplay.h>
#include <mfobjects.h>
#include <tchar.h>
#include <strsafe.h>
#include <codecvt>
#include <mfreadwrite.h>
#ifdef HAVE_MSMF_DXVA
#include <d3d11.h>
#include <d3d11_4.h>
#include <locale>
#endif
#include <new>
#include <map>
#include <queue>
#include <vector>
#include <string>
#include <algorithm>
#include <deque>
#include <iterator>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#ifdef _MSC_VER
#pragma warning(disable:4503)
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mf")
#pragma comment(lib, "mfuuid")
#pragma comment(lib, "Strmiids")
#pragma comment(lib, "Mfreadwrite")
#pragma comment(lib, "dxgi")
#ifdef HAVE_MSMF_DXVA
#pragma comment(lib, "d3d11")
// MFCreateDXGIDeviceManager() is available since Win8 only.
// To avoid OpenCV loading failure on Win7 use dynamic detection of this symbol.
// Details: https://github.com/opencv/opencv/issues/11858
typedef HRESULT (WINAPI *FN_MFCreateDXGIDeviceManager)(UINT *resetToken, IMFDXGIDeviceManager **ppDeviceManager);
static bool pMFCreateDXGIDeviceManager_initialized = false;
static FN_MFCreateDXGIDeviceManager pMFCreateDXGIDeviceManager = NULL;
static void init_MFCreateDXGIDeviceManager()
{
    HMODULE h = LoadLibraryExA("mfplat.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (h)
    {
        pMFCreateDXGIDeviceManager = (FN_MFCreateDXGIDeviceManager)GetProcAddress(h, "MFCreateDXGIDeviceManager");
    }
    pMFCreateDXGIDeviceManager_initialized = true;
}
#endif
#pragma comment(lib, "Shlwapi.lib")
#endif

#include <mferror.h>
#include <comdef.h>

#include <shlwapi.h>  // QISearch

struct IMFMediaType;
struct IMFActivate;
struct IMFMediaSource;
struct IMFAttributes;

#define CV_CAP_MODE_BGR CV_FOURCC_MACRO('B','G','R','3')
#define CV_CAP_MODE_RGB CV_FOURCC_MACRO('R','G','B','3')
#define CV_CAP_MODE_GRAY CV_FOURCC_MACRO('G','R','E','Y')
#define CV_CAP_MODE_YUYV CV_FOURCC_MACRO('Y', 'U', 'Y', 'V')

using namespace cv;

namespace
{

template <class T>
class ComPtr
{
public:
    ComPtr()
    {
    }
    ComPtr(T* lp)
    {
        p = lp;
    }
    ComPtr(_In_ const ComPtr<T>& lp)
    {
        p = lp.p;
    }
    virtual ~ComPtr()
    {
    }

    void swap(_In_ ComPtr<T>& lp)
    {
        ComPtr<T> tmp(p);
        p = lp.p;
        lp.p = tmp.p;
        tmp = NULL;
    }
    T** operator&()
    {
        CV_Assert(p == NULL);
        return p.operator&();
    }
    T* operator->() const
    {
        CV_Assert(p != NULL);
        return p.operator->();
    }
    operator bool()
    {
        return p.operator!=(NULL);
    }

    T* Get() const
    {
        return p;
    }

    void Release()
    {
        if (p)
            p.Release();
    }

    // query for U interface
    template<typename U>
    HRESULT As(_Out_ ComPtr<U>& lp) const
    {
        lp.Release();
        return p->QueryInterface(__uuidof(U), reinterpret_cast<void**>((T**)&lp));
    }
private:
    _COM_SMARTPTR_TYPEDEF(T, __uuidof(T));
    TPtr p;
};

#define _ComPtr ComPtr

template <typename T> inline T absDiff(T a, T b) { return a >= b ? a - b : b - a; }

// synonym for system MFVideoFormat_D16. D3DFMT_D16 = 80
// added to fix builds with old MSVS and platform SDK
// see https://learn.microsoft.com/en-us/windows/win32/medfound/video-subtype-guids#luminance-and-depth-formats
DEFINE_MEDIATYPE_GUID( OCV_MFVideoFormat_D16, 80 );

//==================================================================================================

// Structure for collecting info about types of video which are supported by current video device
struct MediaType
{
    //video param
    UINT32 width;
    UINT32 height;
    INT32 stride; // stride is negative if image is bottom-up
    UINT32 isFixedSize;
    UINT32 frameRateNum;
    UINT32 frameRateDenom;
    UINT32 aspectRatioNum;
    UINT32 aspectRatioDenom;
    UINT32 sampleSize;
    UINT32 interlaceMode;
    //audio param
    UINT32 bit_per_sample;
    UINT32 nChannels;
    UINT32 nAvgBytesPerSec;
    UINT32 nSamplesPerSec;

    GUID majorType; // video or audio
    GUID subType; // fourCC
    _ComPtr<IMFMediaType> Type;
    MediaType(IMFMediaType *pType = 0) :
        Type(pType),
        width(0), height(0),
        stride(0),
        isFixedSize(true),
        frameRateNum(1), frameRateDenom(1),
        aspectRatioNum(1), aspectRatioDenom(1),
        sampleSize(0),
        interlaceMode(0),
        bit_per_sample(0),
        nChannels(0),
        nAvgBytesPerSec(0),
        nSamplesPerSec(0),
        majorType({ 0 }),//MFMediaType_Video
        subType({ 0 })
    {
        if (pType)
        {
            pType->GetGUID(MF_MT_MAJOR_TYPE, &majorType);
            pType->GetGUID(MF_MT_SUBTYPE, &subType);
            if (majorType == MFMediaType_Audio)
            {
                pType->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &bit_per_sample);
                pType->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &nChannels);
                pType->GetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, &nAvgBytesPerSec);
                pType->GetUINT32(MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND, &nSamplesPerSec);
            }
            else if (majorType == MFMediaType_Video)
            {
                MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height);
                pType->GetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32*)&stride); // value is stored as UINT32 but should be casted to INT3)
                pType->GetUINT32(MF_MT_FIXED_SIZE_SAMPLES, &isFixedSize);
                MFGetAttributeRatio(pType, MF_MT_FRAME_RATE, &frameRateNum, &frameRateDenom);
                MFGetAttributeRatio(pType, MF_MT_PIXEL_ASPECT_RATIO, &aspectRatioNum, &aspectRatioDenom);
                pType->GetUINT32(MF_MT_SAMPLE_SIZE, &sampleSize);
                pType->GetUINT32(MF_MT_INTERLACE_MODE, &interlaceMode);
                pType->GetUINT32(MF_MT_INTERLACE_MODE, &interlaceMode);
            }
        }
    }
    static MediaType createDefault_Video()
    {
        MediaType res;
        res.width = 640;
        res.height = 480;
        res.setFramerate(30.0);
        return res;
    }
    static MediaType createDefault_Audio()
    {
        MediaType res;
        res.majorType = MFMediaType_Audio;
        res.subType = MFAudioFormat_PCM;
        res.bit_per_sample = 16;
        res.nChannels = 1;
        res.nSamplesPerSec = 44100;
        return res;
    }
    inline bool isEmpty(bool flag = false) const
    {
        if (!flag)
            return width == 0 && height == 0;
        else
            return nChannels == 0;
    }
    _ComPtr<IMFMediaType> createMediaType_Video() const
    {
        _ComPtr<IMFMediaType> res;
        MFCreateMediaType(&res);
        if (width != 0 || height != 0)
            MFSetAttributeSize(res.Get(), MF_MT_FRAME_SIZE, width, height);
        if (stride != 0)
            res->SetUINT32(MF_MT_DEFAULT_STRIDE, stride);
        res->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, isFixedSize);
        if (frameRateNum != 0 || frameRateDenom != 0)
            MFSetAttributeRatio(res.Get(), MF_MT_FRAME_RATE, frameRateNum, frameRateDenom);
        if (aspectRatioNum != 0 || aspectRatioDenom != 0)
            MFSetAttributeRatio(res.Get(), MF_MT_PIXEL_ASPECT_RATIO, aspectRatioNum, aspectRatioDenom);
        if (sampleSize > 0)
            res->SetUINT32(MF_MT_SAMPLE_SIZE, sampleSize);
        res->SetUINT32(MF_MT_INTERLACE_MODE, interlaceMode);
        if (majorType != GUID())
            res->SetGUID(MF_MT_MAJOR_TYPE, majorType);
        if (subType != GUID())
            res->SetGUID(MF_MT_SUBTYPE, subType);
        return res;
    }
    _ComPtr<IMFMediaType> createMediaType_Audio() const
    {
        _ComPtr<IMFMediaType> res;
        MFCreateMediaType(&res);
        if (majorType != GUID())
            res->SetGUID(MF_MT_MAJOR_TYPE, majorType);
        if (subType != GUID())
            res->SetGUID(MF_MT_SUBTYPE, subType);
        if (bit_per_sample != 0)
            res->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, bit_per_sample);
        if (nChannels != 0)
            res->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, nChannels);
        if (nSamplesPerSec != 0)
            res->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, nSamplesPerSec);
        return res;
    }
    void setFramerate(double fps)
    {
        frameRateNum = (UINT32)cvRound(fps * 1000.0);
        frameRateDenom = 1000;
    }
    double getFramerate() const
    {
        return frameRateDenom != 0 ? ((double)frameRateNum) / ((double)frameRateDenom) : 0;
    }
    LONGLONG getFrameStep() const
    {
        const double fps = getFramerate();
        return (LONGLONG)(fps > 0 ? 1e7 / fps : 0);
    }
    inline unsigned long resolutionDiff(const MediaType& other) const
    {
        const unsigned long wdiff = absDiff(width, other.width);
        const unsigned long hdiff = absDiff(height, other.height);
        return wdiff + hdiff;
    }
    // check if 'this' is better than 'other' comparing to reference
    bool VideoIsBetterThan(const MediaType& other, const MediaType& ref) const
    {
        const unsigned long thisDiff = resolutionDiff(ref);
        const unsigned long otherDiff = other.resolutionDiff(ref);
        if (thisDiff < otherDiff)
            return true;
        if (thisDiff == otherDiff)
        {
            if (width > other.width)
                return true;
            if (width == other.width && height > other.height)
                return true;
            if (width == other.width && height == other.height)
            {
                const double thisRateDiff = absDiff(getFramerate(), ref.getFramerate());
                const double otherRateDiff = absDiff(other.getFramerate(), ref.getFramerate());
                if (thisRateDiff < otherRateDiff)
                    return true;
            }
        }
        return false;
    }
    bool AudioIsBetterThan(const MediaType& other, const MediaType& ref) const
    {
        double thisDiff = absDiff(nChannels, ref.nChannels);
        double otherDiff = absDiff(other.nChannels, ref.nChannels);
        if (otherDiff < thisDiff)
        {
            thisDiff = absDiff(bit_per_sample, ref.bit_per_sample);
            otherDiff = absDiff(bit_per_sample, ref.bit_per_sample);
            if (otherDiff < thisDiff)
            {
                thisDiff = absDiff(nSamplesPerSec, ref.nSamplesPerSec);
                otherDiff = absDiff(nSamplesPerSec, ref.nSamplesPerSec);
                if (otherDiff < thisDiff)
                    return true;
            }
        }
        return false;
    }
    bool VideoIsAvailable() const
    {
        return (subType != OCV_MFVideoFormat_D16);
    }
};

void printFormat(std::ostream& out, const GUID& fmt)
{
#define PRNT(FMT) else if (fmt == FMT) out << #FMT;
    if (fmt == MFVideoFormat_Base) out << "Base";
    PRNT(MFVideoFormat_RGB32)
    PRNT(MFVideoFormat_ARGB32)
    PRNT(MFVideoFormat_RGB24)
    PRNT(MFVideoFormat_RGB555)
    PRNT(MFVideoFormat_RGB565)
    PRNT(MFVideoFormat_RGB8)
    else
    {
        char fourcc[5] = { 0 };
        memcpy(fourcc, &fmt.Data1, 4);
        out << fourcc;
    }
#undef PRNT
}

std::ostream& operator<<(std::ostream& out, const MediaType& mt)
{
    out << "(" << mt.width << "x" << mt.height << " @ " << mt.getFramerate() << ") ";
    printFormat(out, mt.subType);
    return out;
}

//==================================================================================================

// Class for creating of Media Foundation context
class Media_Foundation
{
public:
    ~Media_Foundation(void) { /*CV_Assert(SUCCEEDED(MFShutdown()));*/ CoUninitialize(); }
    static Media_Foundation& getInstance()
    {
        static Media_Foundation instance;
        return instance;
    }
private:
    Media_Foundation(void) { CoInitialize(0); CV_Assert(SUCCEEDED(MFStartup(MF_VERSION))); }
};

//==================================================================================================

class SourceReaderCB : public IMFSourceReaderCallback
{
public:
    static const size_t MSMF_READER_MAX_QUEUE_SIZE = 3;

    SourceReaderCB() :
        m_nRefCount(0), m_hEvent(CreateEvent(NULL, FALSE, FALSE, NULL)), m_bEOS(FALSE), m_hrStatus(S_OK), m_reader(NULL), m_dwStreamIndex(0)
    {
    }

    // IUnknown methods
    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) CV_OVERRIDE
    {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4838)
#endif
        static const QITAB qit[] =
        {
            QITABENT(SourceReaderCB, IMFSourceReaderCallback),
            { 0 },
        };
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        return QISearch(this, qit, iid, ppv);
    }
    STDMETHODIMP_(ULONG) AddRef() CV_OVERRIDE
    {
        return InterlockedIncrement(&m_nRefCount);
    }
    STDMETHODIMP_(ULONG) Release() CV_OVERRIDE
    {
        ULONG uCount = InterlockedDecrement(&m_nRefCount);
        if (uCount == 0)
        {
            delete this;
        }
        return uCount;
    }

    STDMETHODIMP OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample *pSample) CV_OVERRIDE
    {
        HRESULT hr = 0;
        cv::AutoLock lock(m_mutex);

        if (SUCCEEDED(hrStatus))
        {
            if (pSample)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): got frame at " << llTimestamp);
                if (m_capturedFrames.size() >= MSMF_READER_MAX_QUEUE_SIZE)
                {
#if 0
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): drop frame (not processed). Timestamp=" << m_capturedFrames.front().timestamp);
                    m_capturedFrames.pop();
#else
                    // this branch reduces latency if we drop frames due to slow processing.
                    // avoid fetching of already outdated frames from the queue's front.
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): drop previous frames (not processed): " << m_capturedFrames.size());
                    std::queue<CapturedFrameInfo>().swap(m_capturedFrames);  // similar to missing m_capturedFrames.clean();
#endif
                }
                m_capturedFrames.emplace(CapturedFrameInfo{ llTimestamp, _ComPtr<IMFSample>(pSample), hrStatus });
            }
        }
        else
        {
            CV_LOG_WARNING(NULL, "videoio(MSMF): OnReadSample() is called with error status: " << hrStatus);
        }

        if (MF_SOURCE_READERF_ENDOFSTREAM & dwStreamFlags)
        {
            // Reached the end of the stream.
            m_bEOS = true;
        }
        m_hrStatus = hrStatus;

        if (FAILED(hr = m_reader->ReadSample(dwStreamIndex, 0, NULL, NULL, NULL, NULL)))
        {
            CV_LOG_WARNING(NULL, "videoio(MSMF): async ReadSample() call is failed with error status: " << hr);
            m_bEOS = true;
        }

        if (pSample || m_bEOS)
        {
            SetEvent(m_hEvent);
        }
        return S_OK;
    }

    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent *) CV_OVERRIDE
    {
        return S_OK;
    }
    STDMETHODIMP OnFlush(DWORD) CV_OVERRIDE
    {
        return S_OK;
    }

    HRESULT Wait(DWORD dwMilliseconds, _ComPtr<IMFSample>& mediaSample, LONGLONG& sampleTimestamp, BOOL& pbEOS)
    {
        pbEOS = FALSE;

        for (;;)
        {
            {
                cv::AutoLock lock(m_mutex);

                pbEOS = m_bEOS && m_capturedFrames.empty();
                if (pbEOS)
                    return m_hrStatus;

                if (!m_capturedFrames.empty())
                {
                    CV_Assert(!m_capturedFrames.empty());
                    CapturedFrameInfo frameInfo = m_capturedFrames.front(); m_capturedFrames.pop();
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): handle frame at " << frameInfo.timestamp);
                    mediaSample = frameInfo.sample;
                    CV_Assert(mediaSample);
                    sampleTimestamp = frameInfo.timestamp;
                    ResetEvent(m_hEvent);  // event is auto-reset, but we need this forced reset due time gap between wait() and mutex hold.
                    return frameInfo.hrStatus;
                }
            }

            CV_LOG_DEBUG(NULL, "videoio(MSMF): waiting for frame... ");
            DWORD dwResult = WaitForSingleObject(m_hEvent, dwMilliseconds);
            if (dwResult == WAIT_TIMEOUT)
            {
                return E_PENDING;
            }
            else if (dwResult != WAIT_OBJECT_0)
            {
                return HRESULT_FROM_WIN32(GetLastError());
            }
        }
    }

private:
    // Destructor is private. Caller should call Release.
    virtual ~SourceReaderCB()
    {
        CV_LOG_INFO(NULL, "terminating async callback");
    }

public:
    long                m_nRefCount;        // Reference count.
    cv::Mutex           m_mutex;
    HANDLE              m_hEvent;
    BOOL                m_bEOS;
    HRESULT             m_hrStatus;

    IMFSourceReader *m_reader;
    DWORD m_dwStreamIndex;

    struct CapturedFrameInfo {
        LONGLONG timestamp;
        _ComPtr<IMFSample> sample;
        HRESULT hrStatus;
    };

    std::queue<CapturedFrameInfo> m_capturedFrames;
};

//==================================================================================================

// Enumerate and store supported formats and finds format which is most similar to the one requested
class FormatStorage
{
public:
    struct MediaID
    {
        DWORD stream;
        DWORD media;
        MediaID() : stream(0), media(0) {}
        void nextStream()
        {
            stream++;
            media = 0;
        }
        void nextMedia()
        {
            media++;
        }
        bool operator<(const MediaID& other) const
        {
            return (stream < other.stream) || (stream == other.stream && media < other.media);
        }
    };
    void read(IMFSourceReader* source)
    {
        HRESULT hr = S_OK;
        MediaID cur;
        while (SUCCEEDED(hr))
        {
            _ComPtr<IMFMediaType> raw_type;
            hr = source->GetNativeMediaType(cur.stream, cur.media, &raw_type);
            if (hr == MF_E_NO_MORE_TYPES)
            {
                hr = S_OK;
                cur.nextStream();
            }
            else if (SUCCEEDED(hr))
            {
                formats[cur] = MediaType(raw_type.Get());
                cur.nextMedia();
            }
        }
    }
    void countNumberOfAudioStreams(DWORD &numberOfAudioStreams)
    {
        std::pair<MediaID, MediaType> best;
        std::map<MediaID, MediaType>::const_iterator i = formats.begin();
        for (; i != formats.end(); ++i)
        {
            if(i->second.majorType == MFMediaType_Audio)
            {
                if(best.second.isEmpty() || i->first.stream != best.first.stream)
                {
                    numberOfAudioStreams++;
                    best = *i;
                }
            }
        }
    }
    std::pair<MediaID, MediaType> findBestVideoFormat(const MediaType& newType)
    {
        std::pair<MediaID, MediaType> best;
        std::map<MediaID, MediaType>::const_iterator i = formats.begin();
        for (; i != formats.end(); ++i)
        {
            if (i->second.majorType == MFMediaType_Video)
            {
                if (best.second.isEmpty() || (i->second.VideoIsBetterThan(best.second, newType) && i->second.VideoIsAvailable()))
                {
                    best = *i;
                }
            }
        }
        return best;
    }
    std::pair<MediaID, MediaType> findBestAudioFormat(const MediaType& newType)
    {
        std::pair<MediaID, MediaType> best;
        std::map<MediaID, MediaType>::const_iterator i = formats.begin();
        best = *i;
        for (; i != formats.end(); ++i)
        {
            if (i->second.majorType == MFMediaType_Audio)
            {
                if ( i->second.AudioIsBetterThan(best.second, newType))
                {
                    best = *i;
                }
            }
        }
        return best;
    }
    std::pair<MediaID, MediaType> findAudioFormatByStream(const DWORD StreamIndex)
    {
        std::pair<MediaID, MediaType> best;
        std::map<MediaID, MediaType>::const_iterator i = formats.begin();
        for (; i != formats.end(); ++i)
        {
            if (i->second.majorType == MFMediaType_Audio)
            {
                if ((*i).first.stream == StreamIndex)
                {
                    best = *i;
                }
            }
        }
        return best;
    }
private:
    std::map<MediaID, MediaType> formats;
};

//==================================================================================================

// Enumerates devices and activates one of them
class DeviceList
{
public:
    DeviceList() : devices(NULL), count(0) {}
    ~DeviceList()
    {
        if (devices)
        {
            for (UINT32 i = 0; i < count; ++i)
                if (devices[i])
                    devices[i]->Release();
            CoTaskMemFree(devices);
        }
    }
    UINT32 read(IID sourceType = MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
    {
        _ComPtr<IMFAttributes> attr;
        if (FAILED(MFCreateAttributes(&attr, 1)) ||
            FAILED(attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, sourceType)))
        {
            CV_Error(cv::Error::StsError, "Failed to create attributes");
        }
        if (FAILED(MFEnumDeviceSources(attr.Get(), &devices, &count)))
        {
            CV_LOG_DEBUG(NULL, "Failed to enumerate MSMF devices");
            return 0;
        }
        return count;
    }
    _ComPtr<IMFMediaSource> activateSource(UINT32 index)
    {
        _ComPtr<IMFMediaSource> result;
        if (count == 0 || index >= count || FAILED(devices[index]->ActivateObject(__uuidof(IMFMediaSource), (void**)&result)))
        {
            CV_LOG_DEBUG(NULL, "Failed to activate media source (device " << index << ")");
        }
        return result;
    }
private:
    IMFActivate** devices;
    UINT32 count;
};

} // namespace::

//==================================================================================================

/******* Capturing video from camera or file via Microsoft Media Foundation **********/
class CvCapture_MSMF : public cv::IVideoCapture
{
public:
    typedef enum {
        MODE_SW = 0,
        MODE_HW = 1
    } MSMFCapture_Mode;
    CvCapture_MSMF();
    virtual ~CvCapture_MSMF();
    bool configureHW(const cv::VideoCaptureParameters& params);
    virtual bool open(int, const cv::VideoCaptureParameters* params);
    virtual bool open(const cv::String&, const Ptr<IStreamReader>&, const cv::VideoCaptureParameters* params);
    virtual void close();
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    bool configureAudioFrame();
    bool grabAudioFrame();
    bool grabVideoFrame();
    virtual bool grabFrame() CV_OVERRIDE;
    bool retrieveAudioFrame(int, OutputArray);
    bool retrieveVideoFrame(OutputArray);
    virtual bool retrieveFrame(int, cv::OutputArray) CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE { return isOpen; }
    virtual int getCaptureDomain() CV_OVERRIDE { return CAP_MSMF; }
protected:
    bool configureOutput();
    bool configureAudioOutput(MediaType newType);
    bool configureVideoOutput(MediaType newType, cv::uint32_t outFormat);
    bool setTime(double time, bool rough);
    bool setTime(int numberFrame);
    bool configureHW(bool enable);
    bool configureStreams(const cv::VideoCaptureParameters&);
    bool setAudioProperties(const cv::VideoCaptureParameters&);
    bool checkAudioProperties();

    template <typename CtrlT>
    bool readComplexPropery(long prop, long& val) const;
    template <typename CtrlT>
    bool writeComplexProperty(long prop, double val, long flags);
    _ComPtr<IMFAttributes> getDefaultSourceConfig(UINT32 num = 10);
    bool initStream(DWORD streamID, const MediaType mt);

    bool openFinalize_(const VideoCaptureParameters* params);

    Media_Foundation& MF;
    cv::String filename;
    int camid;
    MSMFCapture_Mode captureMode;
    VideoAccelerationType va_type;
    int hwDeviceIndex;
#ifdef HAVE_MSMF_DXVA
    _ComPtr<ID3D11Device> D3DDev;
    _ComPtr<IMFDXGIDeviceManager> D3DMgr;
#endif
    _ComPtr<IMFByteStream> byteStream;
    _ComPtr<IMFSourceReader> videoFileSource;
    _ComPtr<IMFSourceReaderCallback> readCallback;  // non-NULL for "live" streams (camera capture)
    std::vector<DWORD> dwStreamIndices;
    std::vector<_ComPtr<IMFSample>> audioSamples;
    _ComPtr<IMFSample> impendingVideoSample;
    _ComPtr<IMFSample> usedVideoSample;
    DWORD dwVideoStreamIndex;
    DWORD dwAudioStreamIndex;
    MediaType nativeFormat;
    MediaType captureVideoFormat;
    MediaType captureAudioFormat;
    bool device_status; //on or off
    int videoStream; // look at CAP_PROP_VIDEO_STREAM
    int audioStream; // look at CAP_PROP_AUDIO_STREAM
    bool vEOS;
    bool aEOS;
    unsigned int audioBaseIndex;
    int outputVideoFormat;
    int outputAudioFormat;
    UINT32 audioSamplesPerSecond;
    bool convertFormat;
    MFTIME duration;
    LONGLONG frameStep;
    LONGLONG nFrame;
    LONGLONG impendingVideoSampleTime;
    LONGLONG usedVideoSampleTime;
    LONGLONG videoStartOffset;
    LONGLONG videoSampleDuration;
    LONGLONG requiredAudioTime;
    LONGLONG audioSampleTime;
    LONGLONG audioStartOffset;
    LONGLONG audioSampleDuration;
    LONGLONG audioTime;
    LONGLONG chunkLengthOfBytes;
    LONGLONG givenAudioTime;
    LONGLONG numberOfAdditionalAudioBytes; // the number of additional bytes required to align the audio chunk
    double bufferedAudioDuration;
    LONGLONG audioSamplePos;
    DWORD numberOfAudioStreams;
    Mat audioFrame;
    std::deque<BYTE> bufferAudioData;
    bool isOpen;
    bool grabIsDone;
    bool syncLastFrame;
    bool lastFrame;
};

CvCapture_MSMF::CvCapture_MSMF():
    MF(Media_Foundation::getInstance()),
    filename(""),
    camid(-1),
    captureMode(MODE_SW),
    va_type(VIDEO_ACCELERATION_NONE),
    hwDeviceIndex(-1),
#ifdef HAVE_MSMF_DXVA
    D3DDev(NULL),
    D3DMgr(NULL),
#endif
    videoFileSource(NULL),
    readCallback(NULL),
    impendingVideoSample(NULL),
    usedVideoSample(NULL),
    dwVideoStreamIndex(0),
    dwAudioStreamIndex(0),
    device_status(false),
    videoStream(0),
    audioStream(-1),
    vEOS(false),
    aEOS(false),
    audioBaseIndex(1),
    outputVideoFormat(CV_CAP_MODE_BGR),
    outputAudioFormat(CV_16S),
    audioSamplesPerSecond(0),
    convertFormat(true),
    duration(0),
    frameStep(0),
    nFrame(0),
    impendingVideoSampleTime(0),
    usedVideoSampleTime(0),
    videoStartOffset(-1),
    videoSampleDuration(0),
    requiredAudioTime(0),
    audioSampleTime(0),
    audioStartOffset(-1),
    audioSampleDuration(0),
    audioTime(0),
    chunkLengthOfBytes(0),
    givenAudioTime(0),
    numberOfAdditionalAudioBytes(0),
    bufferedAudioDuration(0),
    audioSamplePos(0),
    numberOfAudioStreams(0),
    isOpen(false),
    grabIsDone(false),
    syncLastFrame(true),
    lastFrame(false)
{
}

CvCapture_MSMF::~CvCapture_MSMF()
{
    close();
    configureHW(false);
}

void CvCapture_MSMF::close()
{
    if (isOpen)
    {
        isOpen = false;
        usedVideoSample.Release();
        for (auto item : audioSamples)
            item.Release();
        videoFileSource.Release();
        device_status = false;
        camid = -1;
        filename.clear();
    }
    readCallback.Release();
}

bool CvCapture_MSMF::initStream(DWORD streamID, const MediaType mt)
{
    CV_LOG_DEBUG(NULL, "Init stream " << streamID << " with MediaType " << mt);
    _ComPtr<IMFMediaType> mediaTypesOut;
    if (mt.majorType == MFMediaType_Audio)
    {
        captureAudioFormat = mt;
        mediaTypesOut = mt.createMediaType_Audio();
    }
    if (mt.majorType == MFMediaType_Video)
    {
        captureVideoFormat = mt;
        mediaTypesOut = mt.createMediaType_Video();
    }
    if (FAILED(videoFileSource->SetStreamSelection(streamID, true)))
    {
        CV_LOG_WARNING(NULL, "Failed to select stream " << streamID);
        return false;
    }
    HRESULT hr = videoFileSource->SetCurrentMediaType(streamID, NULL, mediaTypesOut.Get());
    if (hr == MF_E_TOPO_CODEC_NOT_FOUND)
    {
        CV_LOG_WARNING(NULL, "Failed to set mediaType (stream " << streamID << ", " << mt << "(codec not found)");
        return false;
    }
    else if (hr == MF_E_INVALIDMEDIATYPE)
    {
        CV_LOG_WARNING(NULL, "Failed to set mediaType (stream " << streamID << ", " << mt << "(unsupported media type)");
        return false;
    }
    else if (FAILED(hr))
    {
        CV_LOG_WARNING(NULL, "Failed to set mediaType (stream " << streamID << ", " << mt << "(HRESULT " << hr << ")");
        return false;
    }

    return true;
}

_ComPtr<IMFAttributes> CvCapture_MSMF::getDefaultSourceConfig(UINT32 num)
{
    CV_Assert(num > 0);
    const bool OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS = utils::getConfigurationParameterBool("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", true);
    _ComPtr<IMFAttributes> res;
    if (FAILED(MFCreateAttributes(&res, num)) ||
        FAILED(res->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS)) ||
        FAILED(res->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, false)) ||
        FAILED(res->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, false)) ||
        FAILED(res->SetUINT32(MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, true))
        )
    {
        CV_Error(cv::Error::StsError, "Failed to create attributes");
    }
#ifdef HAVE_MSMF_DXVA
    if (D3DMgr)
    {
        if (FAILED(res->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, D3DMgr.Get())))
        {
            CV_Error(cv::Error::StsError, "Failed to create attributes");
        }
    }
#endif
    return res;
}

bool CvCapture_MSMF::configureHW(bool enable)
{
#ifdef HAVE_MSMF_DXVA
    if ((enable && D3DMgr && D3DDev) || (!enable && !D3DMgr && !D3DDev))
        return true;
    if (!pMFCreateDXGIDeviceManager_initialized)
        init_MFCreateDXGIDeviceManager();
    if (enable && !pMFCreateDXGIDeviceManager)
        return false;

    bool reopen = isOpen;
    int prevcam = camid;
    cv::String prevfile = filename;
    close();
    if (enable)
    {
        _ComPtr<IDXGIAdapter> pAdapter;
        if (hwDeviceIndex >= 0) {
            _ComPtr<IDXGIFactory2> pDXGIFactory;
            if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory2), (void**)& pDXGIFactory)) ||
                FAILED(pDXGIFactory->EnumAdapters(hwDeviceIndex, &pAdapter))) {
                return false;
            }
        }
        D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3,  D3D_FEATURE_LEVEL_9_2, D3D_FEATURE_LEVEL_9_1 };
        D3D_DRIVER_TYPE driverType = pAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
        if (SUCCEEDED(D3D11CreateDevice(pAdapter.Get(), driverType, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            levels, sizeof(levels) / sizeof(*levels), D3D11_SDK_VERSION, &D3DDev, NULL, NULL)))
        {
            // NOTE: Getting ready for multi-threaded operation
            _ComPtr<ID3D11Multithread> D3DDevMT;
            UINT mgrRToken;
            if (SUCCEEDED(D3DDev->QueryInterface(IID_PPV_ARGS(&D3DDevMT))))
            {
                D3DDevMT->SetMultithreadProtected(TRUE);
                D3DDevMT.Release();
                if (SUCCEEDED(pMFCreateDXGIDeviceManager(&mgrRToken, &D3DMgr)))
                {
                    if (SUCCEEDED(D3DMgr->ResetDevice(D3DDev.Get(), mgrRToken)))
                    {
                        captureMode = MODE_HW;
                        if (hwDeviceIndex < 0)
                            hwDeviceIndex = 0;
                        // Log adapter description
                        _ComPtr<IDXGIDevice> dxgiDevice;
                        if (SUCCEEDED(D3DDev->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice)))) {
                            _ComPtr<IDXGIAdapter> adapter;
                            if (SUCCEEDED(dxgiDevice->GetAdapter(&adapter))) {
                                DXGI_ADAPTER_DESC desc;
                                if (SUCCEEDED(adapter->GetDesc(&desc))) {
                                    std::wstring name(desc.Description);
                                    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
                                    CV_LOG_INFO(NULL, "MSMF: Using D3D11 video acceleration on GPU device: " << conv.to_bytes(name));
                                }
                            }
                        }
                        // Reopen if needed
                        return reopen ? (prevcam >= 0 ? open(prevcam, NULL) : open(prevfile.c_str(), nullptr, NULL)) : true;
                    }
                    D3DMgr.Release();
                }
            }
            D3DDev.Release();
        }
        return false;
    }
    else
    {
        if (D3DMgr)
            D3DMgr.Release();
        if (D3DDev)
            D3DDev.Release();
        captureMode = MODE_SW;
        return reopen ? (prevcam >= 0 ? open(prevcam, NULL) : open(prevfile.c_str(), nullptr, NULL)) : true;
    }
#else
    return !enable;
#endif
}

bool CvCapture_MSMF::configureHW(const VideoCaptureParameters& params)
{
    va_type = params.get<VideoAccelerationType>(CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY);
    hwDeviceIndex = params.get<int>(CAP_PROP_HW_DEVICE, -1);
#ifndef HAVE_MSMF_DXVA
    if (va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
    {
        CV_LOG_INFO(NULL, "VIDEOIO/MSMF: MSMF backend is build without DXVA acceleration support. Can't handle CAP_PROP_HW_ACCELERATION parameter: " << va_type);
    }
#endif
    return configureHW(va_type == VIDEO_ACCELERATION_D3D11 || va_type == VIDEO_ACCELERATION_ANY);
}

bool CvCapture_MSMF::configureAudioOutput(MediaType newType)
{
    FormatStorage formats;
    formats.read(videoFileSource.Get());
    std::pair<FormatStorage::MediaID, MediaType> bestMatch;
    formats.countNumberOfAudioStreams(numberOfAudioStreams);
    if (device_status)
        bestMatch = formats.findBestAudioFormat(newType);
    else
        bestMatch = formats.findAudioFormatByStream(audioStream);
    if (bestMatch.second.isEmpty(true))
    {
        CV_LOG_DEBUG(NULL, "Can not find audio stream with requested parameters");
        isOpen = false;
        return false;
    }
    dwAudioStreamIndex = bestMatch.first.stream;
    dwStreamIndices.push_back(dwAudioStreamIndex);
    MediaType newFormat = bestMatch.second;

    newFormat.majorType = MFMediaType_Audio;
    newFormat.nSamplesPerSec = (audioSamplesPerSecond == 0) ? 44100 : audioSamplesPerSecond;
    switch (outputAudioFormat)
    {
    case CV_8S:
        newFormat.subType = MFAudioFormat_PCM;
        newFormat.bit_per_sample = 8;
        break;
    case CV_16S:
        newFormat.subType = MFAudioFormat_PCM;
        newFormat.bit_per_sample = 16;
        break;
    case CV_32S:
        newFormat.subType = MFAudioFormat_PCM;
        newFormat.bit_per_sample = 32;
    case CV_32F:
        newFormat.subType = MFAudioFormat_Float;
        newFormat.bit_per_sample = 32;
        break;
    default:
        break;
    }

    return initStream(dwAudioStreamIndex, newFormat);
}

bool CvCapture_MSMF::configureVideoOutput(MediaType newType, cv::uint32_t outFormat)
{
    FormatStorage formats;
    formats.read(videoFileSource.Get());
    std::pair<FormatStorage::MediaID, MediaType> bestMatch = formats.findBestVideoFormat(newType);
    if (bestMatch.second.isEmpty())
    {
        CV_LOG_DEBUG(NULL, "Can not find video stream with requested parameters");
        return false;
    }
    dwVideoStreamIndex = bestMatch.first.stream;
    dwStreamIndices.push_back(dwVideoStreamIndex);
    nativeFormat = bestMatch.second;
    MediaType newFormat = nativeFormat;

    if (convertFormat)
    {
        switch (outFormat)
        {
        case CV_CAP_MODE_BGR:
        case CV_CAP_MODE_RGB:
            newFormat.subType = captureMode == MODE_HW ? MFVideoFormat_RGB32 : MFVideoFormat_RGB24;
            newFormat.stride = (captureMode == MODE_HW ? 4 : 3) * newFormat.width;
            newFormat.sampleSize = newFormat.stride * newFormat.height;
            break;
        case CV_CAP_MODE_GRAY:
            newFormat.subType = MFVideoFormat_YUY2;
            newFormat.stride = newFormat.width;
            newFormat.sampleSize = newFormat.stride * newFormat.height * 3 / 2;
            break;
        case CV_CAP_MODE_YUYV:
            newFormat.subType = MFVideoFormat_YUY2;
            newFormat.stride = 2 * newFormat.width;
            newFormat.sampleSize = newFormat.stride * newFormat.height;
            break;
        default:
            return false;
        }
        newFormat.interlaceMode = MFVideoInterlace_Progressive;
        newFormat.isFixedSize = true;
        if (nativeFormat.subType == MFVideoFormat_MP43) //Unable to estimate FPS for MP43
            newFormat.frameRateNum = 0;
    }
    // we select native format first and then our requested format (related issue #12822)
    if (!newType.isEmpty()) // camera input
    {
        initStream(dwVideoStreamIndex, nativeFormat);
    }
    if (!initStream(dwVideoStreamIndex, newFormat))
    {
        return false;
    }
    outputVideoFormat = outFormat;
    return true;
}

bool CvCapture_MSMF::configureOutput()
{
    if (FAILED(videoFileSource->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, false)))
    {
        CV_LOG_WARNING(NULL, "Failed to reset streams");
        return false;
    }
    bool tmp = true;
    if (videoStream != -1)
        tmp = (!device_status)? configureVideoOutput(MediaType(), outputVideoFormat) : configureVideoOutput(MediaType::createDefault_Video(), outputVideoFormat);
    if (audioStream != -1)
        tmp &= (!device_status)? configureAudioOutput(MediaType()) : configureAudioOutput(MediaType::createDefault_Audio());
    return tmp;
}

bool CvCapture_MSMF::open(int index, const cv::VideoCaptureParameters* params)
{
    close();
    if (index < 0)
        return false;

    if (params)
    {
        configureHW(*params);
        if (!(configureStreams(*params) && setAudioProperties(*params)))
            return false;
    }
    if (videoStream != -1 && audioStream != -1 || videoStream == -1 && audioStream == -1)
    {
        CV_LOG_DEBUG(NULL, "Only one of the properties CAP_PROP_AUDIO_STREAM " << audioStream << " and " << CAP_PROP_VIDEO_STREAM << " must be different from -1");
        return false;
    }
    DeviceList devices;
    UINT32 count = 0;
    if (audioStream != -1)
        count = devices.read(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_GUID);
    if (videoStream != -1)
        count = devices.read(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (count == 0 || static_cast<UINT32>(index) > count)
    {
        CV_LOG_DEBUG(NULL, "Device " << index << " not found (total " << count << " devices)");
        return false;
    }
    _ComPtr<IMFAttributes> attr = getDefaultSourceConfig();
    _ComPtr<IMFSourceReaderCallback> cb = new SourceReaderCB();
    attr->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, cb.Get());
    _ComPtr<IMFMediaSource> src = devices.activateSource(index);
    if (!src.Get() || FAILED(MFCreateSourceReaderFromMediaSource(src.Get(), attr.Get(), &videoFileSource)))
    {
        CV_LOG_DEBUG(NULL, "Failed to create source reader");
        return false;
    }

    isOpen = true;
    device_status = true;
    camid = index;
    readCallback = cb;
    duration = 0;
    if (configureOutput())
    {
        frameStep = captureVideoFormat.getFrameStep();
    }
    if (isOpen && !openFinalize_(params))
    {
        close();
        return false;
    }
    if (isOpen)
    {
        if (audioStream != -1)
            if (!checkAudioProperties())
                return false;
    }

    return isOpen;
}

bool CvCapture_MSMF::open(const cv::String& _filename, const Ptr<IStreamReader>& stream, const cv::VideoCaptureParameters* params)
{
    close();
    if (_filename.empty() && !stream)
        return false;

    if (params)
    {
        configureHW(*params);
        if (!(configureStreams(*params) && setAudioProperties(*params)))
            return false;
    }
    // Set source reader parameters
    _ComPtr<IMFAttributes> attr = getDefaultSourceConfig();
    bool succeeded = false;
    if (!_filename.empty())
    {
        cv::AutoBuffer<wchar_t> unicodeFileName(_filename.length() + 1);
        MultiByteToWideChar(CP_ACP, 0, _filename.c_str(), -1, unicodeFileName.data(), (int)_filename.length() + 1);
        succeeded = SUCCEEDED(MFCreateSourceReaderFromURL(unicodeFileName.data(), attr.Get(), &videoFileSource));
    }
    else if (stream)
    {
        // TODO: implement read by chunks
        // FIXIT: save stream in field
        std::vector<char> data;
        data.resize((size_t)stream->seek(0, SEEK_END));
        stream->seek(0, SEEK_SET);
        stream->read(data.data(), data.size());
        IStream* s = SHCreateMemStream(reinterpret_cast<const BYTE*>(data.data()), static_cast<UINT32>(data.size()));
        if (!s)
            return false;

        succeeded = SUCCEEDED(MFCreateMFByteStreamOnStream(s, &byteStream));
        if (!succeeded)
            return false;
        if (!SUCCEEDED(MFStartup(MF_VERSION)))
            return false;
        succeeded = SUCCEEDED(MFCreateSourceReaderFromByteStream(byteStream.Get(), attr.Get(), &videoFileSource));
    }

    if (succeeded)
    {
        isOpen = true;
        usedVideoSampleTime = 0;
        if (configureOutput())
        {
            filename = _filename;
            frameStep = captureVideoFormat.getFrameStep();
            PROPVARIANT var;
            HRESULT hr;
            if (SUCCEEDED(hr = videoFileSource->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &var)) &&
                var.vt == VT_UI8)
            {
                duration = var.uhVal.QuadPart;
                PropVariantClear(&var);
            }
            else
                duration = 0;
        }
    }
    if (isOpen && !openFinalize_(params))
    {
        close();
        return false;
    }
    if (isOpen)
    {
        if (audioStream != -1)
        {
            if (!checkAudioProperties())
                return false;
            if (videoStream != -1)
            {
                isOpen = grabFrame();
                if (isOpen)
                    grabIsDone = true;
            }
        }
    }
    return isOpen;
}

bool CvCapture_MSMF::openFinalize_(const VideoCaptureParameters* params)
{
    if (params)
    {
        std::vector<int> unused_params = params->getUnused();
        for (int key : unused_params)
        {
            if (!setProperty(key, params->get<double>(key)))
            {
                CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: can't set property " << key);
                return false;
            }
        }
    }

    VideoAccelerationType actual_va_type = (captureMode == MODE_HW) ? VIDEO_ACCELERATION_D3D11 : VIDEO_ACCELERATION_NONE;
    if (va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
    {
        if (va_type != actual_va_type)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: Can't select requested video acceleration through CAP_PROP_HW_ACCELERATION: "
                    << va_type << " (actual is " << actual_va_type << "). Bailout");
            return false;
        }
    }
    else
    {
        va_type = actual_va_type;
    }

    return true;
}

bool CvCapture_MSMF::configureStreams(const cv::VideoCaptureParameters& params)
{
    if (params.has(CAP_PROP_VIDEO_STREAM))
    {
        double value = params.get<double>(CAP_PROP_VIDEO_STREAM);
        if (value == -1 || value == 0)
            videoStream = static_cast<int>(value);
        else
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: CAP_PROP_VIDEO_STREAM parameter value is invalid/unsupported: " << value);
            return false;
        }
    }
    if (params.has(CAP_PROP_AUDIO_STREAM))
    {
        double value = params.get<double>(CAP_PROP_AUDIO_STREAM);
        if (value == -1 || value > -1)
            audioStream = static_cast<int>(value);
        else
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: CAP_PROP_AUDIO_STREAM parameter value is invalid/unsupported: " << value);
            return false;
        }
    }
    return true;
}
bool CvCapture_MSMF::setAudioProperties(const cv::VideoCaptureParameters& params)
{
    if (params.has(CAP_PROP_AUDIO_DATA_DEPTH))
    {
        int value = static_cast<int>(params.get<double>(CAP_PROP_AUDIO_DATA_DEPTH));
        if (value != CV_8S && value != CV_16S && value != CV_32S && value != CV_32F)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: CAP_PROP_AUDIO_DATA_DEPTH parameter value is invalid/unsupported: " << value);
            return false;
        }
        else
        {
            outputAudioFormat = value;
        }
    }
    if (params.has(CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
    {
        int value = static_cast<int>(params.get<double>(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));
        if (value < 0)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: CAP_PROP_AUDIO_SAMPLES_PER_SECOND parameter can't be negative: " << value);
            return false;
        }
        else
        {
            audioSamplesPerSecond = value;
        }
    }
    if (params.has(CAP_PROP_AUDIO_SYNCHRONIZE))
    {
        int value = static_cast<UINT32>(params.get<double>(CAP_PROP_AUDIO_SYNCHRONIZE));
        syncLastFrame = (value != 0) ? true : false;
    }
    return true;
}
bool CvCapture_MSMF::checkAudioProperties()
{
    if (audioSamplesPerSecond != 0)
    {
        _ComPtr<IMFMediaType> type;
        UINT32 actualAudioSamplesPerSecond = 0;
        HRESULT hr = videoFileSource->GetCurrentMediaType(dwAudioStreamIndex, &type);
        if (SUCCEEDED(hr))
        {
            type->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND , &actualAudioSamplesPerSecond);
            if (actualAudioSamplesPerSecond != audioSamplesPerSecond)
            {
                CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: CAP_PROP_AUDIO_SAMPLES_PER_SECOND parameter value is invalid/unsupported: " << audioSamplesPerSecond
                            << ". Current value of CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << actualAudioSamplesPerSecond);
                close();
                return false;
            }
            return true;
        }
        return false;
    }
    return true;
}
bool CvCapture_MSMF::grabVideoFrame()
{
    DWORD streamIndex,  flags;
    HRESULT hr;
    usedVideoSample.Release();

    bool returnFlag = false;
    bool stopFlag = false;
    if (audioStream != -1)
    {
        usedVideoSample.swap(impendingVideoSample);
        std::swap(usedVideoSampleTime, impendingVideoSampleTime);
    }
    while (!stopFlag)
    {
        for (;;)
        {
            CV_TRACE_REGION("ReadSample");
            if (!SUCCEEDED(hr = videoFileSource->ReadSample(
                dwVideoStreamIndex, // Stream index.
                0,             // Flags.
                &streamIndex,  // Receives the actual stream index.
                &flags,        // Receives status flags.
                &impendingVideoSampleTime,   // Receives the time stamp.
                &impendingVideoSample   // Receives the sample or NULL.
            )))
                break;
            if (streamIndex != dwVideoStreamIndex)
                break;
            if (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ALLEFFECTSREMOVED | MF_SOURCE_READERF_ENDOFSTREAM))
                break;
            if (impendingVideoSample)
                break;
            if (flags & MF_SOURCE_READERF_STREAMTICK)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream tick detected. Retrying to grab the frame");
            }
        }
        if (SUCCEEDED(hr))
        {
            if (streamIndex != dwVideoStreamIndex)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Wrong stream read. Abort capturing");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ERROR)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream reading error. Abort capturing");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ALLEFFECTSREMOVED)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream decoding error. Abort capturing");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
            {
                vEOS = true;
                lastFrame = true;
                stopFlag = true;
                if (audioStream == -1)
                    returnFlag = false;
                else if (usedVideoSample)
                    returnFlag = true;
                CV_LOG_DEBUG(NULL, "videoio(MSMF): End of video stream detected");
            }
            else
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): got video frame with timestamp=" << impendingVideoSampleTime);
                if (audioStream != -1)
                {
                    if (!usedVideoSample)
                    {
                        usedVideoSample.swap(impendingVideoSample);
                        std::swap(usedVideoSampleTime, impendingVideoSampleTime);
                        videoStartOffset = usedVideoSampleTime;
                    }
                    else
                    {
                        stopFlag = true;
                    }
                    if (impendingVideoSample)
                    {
                        nFrame++;
                        videoSampleDuration = impendingVideoSampleTime - usedVideoSampleTime;
                        requiredAudioTime = impendingVideoSampleTime - givenAudioTime;
                        givenAudioTime += requiredAudioTime;
                    }
                }
                else
                {
                    usedVideoSample.swap(impendingVideoSample);
                    std::swap(usedVideoSampleTime, impendingVideoSampleTime);
                    stopFlag = true;
                    nFrame++;
                }
                if (flags & MF_SOURCE_READERF_NEWSTREAM)
                {
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): New stream detected");
                }
                if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
                {
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream native media type changed");
                }
                if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
                {
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream current media type changed");
                }
                returnFlag = true;
            }
        }
    }
    return returnFlag;
}

bool CvCapture_MSMF::configureAudioFrame()
{
    if (!audioSamples.empty() || !bufferAudioData.empty() && aEOS)
    {
        _ComPtr<IMFMediaBuffer> buf = NULL;
        std::vector<BYTE> audioDataInUse;
        BYTE* ptr = NULL;
        DWORD maxsize = 0, cursize = 0;
        CV_TRACE_REGION("get_contiguous_buffer");
        for (auto item : audioSamples)
        {
            if (!SUCCEEDED(item->ConvertToContiguousBuffer(&buf)))
            {
                CV_TRACE_REGION("get_buffer");
                DWORD bcnt = 0;
                if (!SUCCEEDED(item->GetBufferCount(&bcnt)))
                    break;
                if (bcnt == 0)
                    break;
                if (!SUCCEEDED(item->GetBufferByIndex(0, &buf)))
                    break;
            }
            if (!SUCCEEDED(buf->Lock(&ptr, &maxsize, &cursize)))
                break;
            size_t lastSize = bufferAudioData.size();
            bufferAudioData.resize(lastSize+cursize);
            for (unsigned int i = 0; i < cursize; i++)
            {
                bufferAudioData[lastSize+i]=*(ptr+i);
            }
            CV_TRACE_REGION_NEXT("unlock");
            buf->Unlock();
            buf = NULL;
        }
        audioSamples.clear();

        audioSamplePos += chunkLengthOfBytes/((captureAudioFormat.bit_per_sample/8)*captureAudioFormat.nChannels);
        chunkLengthOfBytes = (videoStream != -1) ? (LONGLONG)((requiredAudioTime*captureAudioFormat.nSamplesPerSec*captureAudioFormat.nChannels*(captureAudioFormat.bit_per_sample)/8)/1e7) : cursize;
        if ((videoStream != -1) && (chunkLengthOfBytes % ((int)(captureAudioFormat.bit_per_sample)/8* (int)captureAudioFormat.nChannels) != 0))
        {
            if ( (double)audioSamplePos/captureAudioFormat.nSamplesPerSec + audioStartOffset * 1e-7 - usedVideoSampleTime * 1e-7 >= 0 )
                chunkLengthOfBytes -= numberOfAdditionalAudioBytes;
            numberOfAdditionalAudioBytes = ((int)(captureAudioFormat.bit_per_sample)/8* (int)captureAudioFormat.nChannels)
                                        - chunkLengthOfBytes % ((int)(captureAudioFormat.bit_per_sample)/8* (int)captureAudioFormat.nChannels);
            chunkLengthOfBytes += numberOfAdditionalAudioBytes;
        }
        if (lastFrame && !syncLastFrame || aEOS && !vEOS)
        {
            chunkLengthOfBytes = bufferAudioData.size();
            audioSamplePos += chunkLengthOfBytes/((captureAudioFormat.bit_per_sample/8)*captureAudioFormat.nChannels);
        }
        CV_Check((double)chunkLengthOfBytes, chunkLengthOfBytes >= INT_MIN || chunkLengthOfBytes <= INT_MAX, "MSMF: The chunkLengthOfBytes is out of the allowed range");
        copy(bufferAudioData.begin(), bufferAudioData.begin() + (int)chunkLengthOfBytes, std::back_inserter(audioDataInUse));
        bufferAudioData.erase(bufferAudioData.begin(), bufferAudioData.begin() + (int)chunkLengthOfBytes);
        if (audioFrame.empty())
        {
            switch (outputAudioFormat)
            {
            case CV_8S:
                cv::Mat((int)chunkLengthOfBytes/(captureAudioFormat.nChannels), captureAudioFormat.nChannels, CV_8S, audioDataInUse.data()).copyTo(audioFrame);
                break;
            case CV_16S:
                cv::Mat((int)chunkLengthOfBytes/(2*captureAudioFormat.nChannels), captureAudioFormat.nChannels, CV_16S, audioDataInUse.data()).copyTo(audioFrame);
                break;
            case CV_32S:
                cv::Mat((int)chunkLengthOfBytes/(4*captureAudioFormat.nChannels), captureAudioFormat.nChannels, CV_32S, audioDataInUse.data()).copyTo(audioFrame);
                break;
            case CV_32F:
                cv::Mat((int)chunkLengthOfBytes/(4*captureAudioFormat.nChannels), captureAudioFormat.nChannels, CV_32F, audioDataInUse.data()).copyTo(audioFrame);
                break;
            default:
                break;
            }
        }
        audioDataInUse.clear();
        audioDataInUse.shrink_to_fit();
        return true;
    }
    else
    {
        return false;
    }
}

bool CvCapture_MSMF::grabAudioFrame()
{
    DWORD streamIndex,  flags;
    HRESULT hr;
    _ComPtr<IMFSample> audioSample = NULL;
    audioSamples.clear();

    bool returnFlag = false;
    audioTime = 0;
    int numberOfSamples = -1;
    if (bufferedAudioDuration*1e7 > requiredAudioTime)
        return true;
    while ((!vEOS) ? audioTime <= requiredAudioTime : !aEOS)
    {
        if (audioStartOffset - usedVideoSampleTime > videoSampleDuration)
            return true;
        for (;;)
        {
            CV_TRACE_REGION("ReadSample");
            if (!SUCCEEDED(hr = videoFileSource->ReadSample(
                dwAudioStreamIndex, // Stream index.
                0,             // Flags.
                &streamIndex,  // Receives the actual stream index.
                &flags,        // Receives status flags.
                &audioSampleTime,   // Receives the time stamp.
                &audioSample  // Receives the sample or NULL.
            )))
                break;
            if (streamIndex != dwAudioStreamIndex)
                break;
            if (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ALLEFFECTSREMOVED | MF_SOURCE_READERF_ENDOFSTREAM))
                break;
            if (audioSample)
                break;
            if (flags & MF_SOURCE_READERF_STREAMTICK)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream tick detected. Retrying to grab the frame");
            }
        }
        if (SUCCEEDED(hr))
        {
            if (streamIndex != dwAudioStreamIndex)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Wrong stream read. Abort capturing");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ERROR)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream reading error. Abort capturing");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ALLEFFECTSREMOVED)
            {
                CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream decoding error. Abort capturing");
                close();
            }
            else if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
            {
                aEOS = true;
                if (videoStream != -1 && !vEOS)
                    returnFlag = true;
                if (videoStream == -1)
                    audioSamplePos += chunkLengthOfBytes/((captureAudioFormat.bit_per_sample/8)*captureAudioFormat.nChannels);
                CV_LOG_DEBUG(NULL, "videoio(MSMF): End of audio stream detected");
                break;
            }
            else
            {
                audioSamples.push_back(audioSample);
                audioSample = NULL;
                numberOfSamples++;
                audioSamples[numberOfSamples]->GetSampleDuration(&audioSampleDuration);
                CV_LOG_DEBUG(NULL, "videoio(MSMF): got audio frame with timestamp=" << audioSampleTime << "  duration=" << audioSampleDuration);
                audioTime += (LONGLONG)(audioSampleDuration + bufferedAudioDuration*1e7);
                if (nFrame == 1 && audioStartOffset == -1)
                {
                    audioStartOffset = audioSampleTime - audioSampleDuration;
                    requiredAudioTime -= audioStartOffset;
                }
                if (flags & MF_SOURCE_READERF_NEWSTREAM)
                {
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): New stream detected");
                }
                if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
                {
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream native media type changed");
                }
                if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
                {
                    CV_LOG_DEBUG(NULL, "videoio(MSMF): Stream current media type changed");
                }
                returnFlag = true;
            }
        }
        else
        {
            CV_LOG_DEBUG(NULL, "videoio(MSMF): ReadSample() method is not succeeded");
            return false;
        }
    }

    returnFlag &= configureAudioFrame();
    return returnFlag;
}

bool CvCapture_MSMF::grabFrame()
{
    CV_TRACE_FUNCTION();

    if (grabIsDone)
    {
        grabIsDone = false;
        CV_LOG_DEBUG(NULL, "videoio(MSMF): return pre-grabbed frame " << usedVideoSampleTime);
        return true;
    }

    audioFrame = Mat();
    if (readCallback)  // async "live" capture mode
    {
        audioSamples.push_back(NULL);
        HRESULT hr = 0;
        SourceReaderCB* reader = ((SourceReaderCB*)readCallback.Get());
        DWORD dwStreamIndex = 0;
        if (videoStream != -1)
            dwStreamIndex = dwVideoStreamIndex;
        if (audioStream != -1)
            dwStreamIndex = dwAudioStreamIndex;
        if (!reader->m_reader)
        {
            // Initiate capturing with async callback
            reader->m_reader = videoFileSource.Get();
            reader->m_dwStreamIndex = dwStreamIndex;
            if (FAILED(hr = videoFileSource->ReadSample(dwStreamIndex, 0, NULL, NULL, NULL, NULL)))
            {
                CV_LOG_ERROR(NULL, "videoio(MSMF): can't grab frame - initial async ReadSample() call failed: " << hr);
                reader->m_reader = NULL;
                return false;
            }
        }
        BOOL bEOS = false;
        LONGLONG timestamp = 0;
        if (FAILED(hr = reader->Wait( videoStream == -1 ? INFINITE : 10000, (videoStream != -1) ? usedVideoSample : audioSamples[0], timestamp, bEOS)))  // 10 sec
        {
            CV_LOG_WARNING(NULL, "videoio(MSMF): can't grab frame. Error: " << hr);
            return false;
        }
        if (bEOS)
        {
            CV_LOG_WARNING(NULL, "videoio(MSMF): EOS signal. Capture stream is lost");
            return false;
        }
        if (videoStream != -1)
            usedVideoSampleTime = timestamp;
        if (audioStream != -1)
            return configureAudioFrame();

        CV_LOG_DEBUG(NULL, "videoio(MSMF): grabbed frame " << usedVideoSampleTime);
        return true;
    }
    else if (isOpen)
    {
        if (vEOS)
            return false;

        bool returnFlag = true;

        if (videoStream != -1)
        {
            if (!vEOS)
                returnFlag &= grabVideoFrame();
            if (!returnFlag)
                return false;
        }

        if (audioStream != -1)
        {
            bufferedAudioDuration = (double)(bufferAudioData.size()/((captureAudioFormat.bit_per_sample/8)*captureAudioFormat.nChannels))/captureAudioFormat.nSamplesPerSec;
            audioFrame.release();
            if (!aEOS)
                returnFlag &= grabAudioFrame();
        }

        return returnFlag;
    }
    return false;
}

bool CvCapture_MSMF::retrieveVideoFrame(cv::OutputArray frame)
{
    CV_TRACE_FUNCTION();
    CV_LOG_DEBUG(NULL, "videoio(MSMF): retrieve video frame start...");
    do
    {
        if (!usedVideoSample)
            break;

        _ComPtr<IMFMediaBuffer> buf = NULL;
        CV_TRACE_REGION("get_contiguous_buffer");
        if (!SUCCEEDED(usedVideoSample->ConvertToContiguousBuffer(&buf)))
        {
            CV_TRACE_REGION("get_buffer");
            DWORD bcnt = 0;
            if (!SUCCEEDED(usedVideoSample->GetBufferCount(&bcnt)))
                break;
            if (bcnt == 0)
                break;
            if (!SUCCEEDED(usedVideoSample->GetBufferByIndex(0, &buf)))
                break;
        }

        bool lock2d = false;
        BYTE* ptr = NULL;
        LONG pitch = 0;
        DWORD maxsize = 0, cursize = 0;

        // "For 2-D buffers, the Lock2D method is more efficient than the Lock method"
        // see IMFMediaBuffer::Lock method documentation: https://msdn.microsoft.com/en-us/library/windows/desktop/bb970366(v=vs.85).aspx
        _ComPtr<IMF2DBuffer> buffer2d;
        if (convertFormat)
        {
            if (SUCCEEDED(buf.As<IMF2DBuffer>(buffer2d)))
            {
                CV_TRACE_REGION_NEXT("lock2d");
                if (SUCCEEDED(buffer2d->Lock2D(&ptr, &pitch)))
                {
                    lock2d = true;
                }
            }
        }
        if (ptr == NULL)
        {
            CV_Assert(lock2d == false);
            CV_TRACE_REGION_NEXT("lock");
            if (!SUCCEEDED(buf->Lock(&ptr, &maxsize, &cursize)))
            {
                break;
            }
        }
        if (!ptr)
            break;
        if (convertFormat)
        {
            if (lock2d || (unsigned int)cursize == captureVideoFormat.sampleSize)
            {
                switch (outputVideoFormat)
                {
                case CV_CAP_MODE_YUYV:
                    cv::Mat(captureVideoFormat.height, captureVideoFormat.width, CV_8UC2, ptr, pitch).copyTo(frame);
                    break;
                case CV_CAP_MODE_BGR:
                    if (captureMode == MODE_HW)
                        cv::cvtColor(cv::Mat(captureVideoFormat.height, captureVideoFormat.width, CV_8UC4, ptr, pitch), frame, cv::COLOR_BGRA2BGR);
                    else
                        cv::Mat(captureVideoFormat.height, captureVideoFormat.width, CV_8UC3, ptr, pitch).copyTo(frame);
                    break;
                case CV_CAP_MODE_RGB:
                    if (captureMode == MODE_HW)
                        cv::cvtColor(cv::Mat(captureVideoFormat.height, captureVideoFormat.width, CV_8UC4, ptr, pitch), frame, cv::COLOR_BGRA2BGR);
                    else
                        cv::cvtColor(cv::Mat(captureVideoFormat.height, captureVideoFormat.width, CV_8UC3, ptr, pitch), frame, cv::COLOR_BGR2RGB);
                    break;
                case CV_CAP_MODE_GRAY:
                    cv::Mat(captureVideoFormat.height, captureVideoFormat.width, CV_8UC1, ptr, pitch).copyTo(frame);
                    break;
                default:
                    frame.release();
                    break;
                }
            }
            else
                frame.release();
        }
        else
        {
            cv::Mat(1, cursize, CV_8UC1, ptr, pitch).copyTo(frame);
        }
        CV_TRACE_REGION_NEXT("unlock");
        if (lock2d)
            buffer2d->Unlock2D();
        else
            buf->Unlock();
        CV_LOG_DEBUG(NULL, "videoio(MSMF): retrieve video frame done!");
        return !frame.empty();
    } while (0);

    frame.release();
    return false;
}

bool CvCapture_MSMF::retrieveAudioFrame(int index, cv::OutputArray frame)
{
    CV_TRACE_FUNCTION();
    if (audioStartOffset - usedVideoSampleTime > videoSampleDuration)
    {
        frame.release();
        return true;
    }
    do
    {
        if (audioFrame.empty())
        {
            frame.release();
            if (aEOS)
                return true;
        }
        cv::Mat data;
        switch (outputAudioFormat)
        {
        case CV_8S:
            data = cv::Mat(1, audioFrame.rows, CV_8S);
            for (int i = 0; i < audioFrame.rows; i++)
                data.at<char>(0,i) = audioFrame.at<char>(i,index-audioBaseIndex);
            break;
        case CV_16S:
            data = cv::Mat(1, audioFrame.rows, CV_16S);
            for (int i = 0; i < audioFrame.rows; i++)
                data.at<short>(0,i) = audioFrame.at<short>(i,index-audioBaseIndex);
            break;
        case CV_32S:
            data = cv::Mat(1, audioFrame.rows, CV_32S);
            for (int i = 0; i < audioFrame.rows; i++)
                data.at<int>(0,i) = audioFrame.at<int>(i,index-audioBaseIndex);
            break;
        case CV_32F:
            data = cv::Mat(1, audioFrame.rows, CV_32F);
            for (int i = 0; i < audioFrame.rows; i++)
                data.at<float>(0,i) = audioFrame.at<float>(i,index-audioBaseIndex);
            break;
        default:
            frame.release();
            break;
        }
        if (!data.empty())
            data.copyTo(frame);

        return !frame.empty();
    } while (0);

    return false;
}

bool CvCapture_MSMF::retrieveFrame(int index, cv::OutputArray frame)
{
    CV_TRACE_FUNCTION();
    if (index < 0)
        return false;
    if ((unsigned int)index < audioBaseIndex)
    {
        if (videoStream == -1)
        {
            frame.release();
            return false;
        }
        else
            return retrieveVideoFrame(frame);
    }
    else
    {
        if (audioStream == -1)
        {
            frame.release();
            return false;
        }
        else
            return retrieveAudioFrame(index, frame);
    }
}

bool CvCapture_MSMF::setTime(double time, bool rough)
{
    if (videoStream == -1)
        return false;
    if (videoStream != -1 && audioStream != -1)
        if (time != 0)
            return false;
    PROPVARIANT var;
    if (SUCCEEDED(videoFileSource->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_SOURCE_READER_MEDIASOURCE_CHARACTERISTICS, &var)) &&
        var.vt == VT_UI4 && var.ulVal & MFMEDIASOURCE_CAN_SEEK)
    {
        usedVideoSample.Release();
        bool useGrabbing = time > 0 && !rough && !(var.ulVal & MFMEDIASOURCE_HAS_SLOW_SEEK);
        PropVariantClear(&var);
        usedVideoSampleTime = (useGrabbing) ? 0 : (LONGLONG)floor(time + 0.5);
        nFrame = (useGrabbing) ? 0 : usedVideoSampleTime/frameStep;
        givenAudioTime = (useGrabbing) ? 0 : nFrame*frameStep;
        var.vt = VT_I8;
        var.hVal.QuadPart = usedVideoSampleTime;
        bool resOK = SUCCEEDED(videoFileSource->SetCurrentPosition(GUID_NULL, var));
        PropVariantClear(&var);
        if (resOK && useGrabbing)
        {
            LONGLONG timeborder = (LONGLONG)floor(time + 0.5) - frameStep / 2;
            do { resOK = grabFrame(); usedVideoSample.Release(); } while (resOK && usedVideoSampleTime < timeborder);
        }
        return resOK;
    }
    return false;
}

bool CvCapture_MSMF::setTime(int numberFrame)
{
    if (videoStream == -1)
        return false;
    if (videoStream != -1 && audioStream != -1)
        if (numberFrame != 0)
            return false;
    PROPVARIANT var;
    if (SUCCEEDED(videoFileSource->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_SOURCE_READER_MEDIASOURCE_CHARACTERISTICS, &var)) &&
        var.vt == VT_UI4 && var.ulVal & MFMEDIASOURCE_CAN_SEEK)
    {
        usedVideoSample.Release();
        PropVariantClear(&var);
        usedVideoSampleTime =  0;
        nFrame =  0;
        givenAudioTime =  0;
        var.vt = VT_I8;
        var.hVal.QuadPart = usedVideoSampleTime;
        bool resOK = SUCCEEDED(videoFileSource->SetCurrentPosition(GUID_NULL, var));
        PropVariantClear(&var);
        while (resOK && nFrame < numberFrame) { resOK = grabFrame(); usedVideoSample.Release(); };
        return resOK;
    }
    return false;
}

template <typename CtrlT>
bool CvCapture_MSMF::readComplexPropery(long prop, long & val) const
{
    _ComPtr<CtrlT> ctrl;
    if (FAILED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&ctrl))))
    {
        CV_LOG_DEBUG(NULL, "Failed to get service for stream");
        return false;
    }
    long paramVal, paramFlag;
    if (FAILED(ctrl->Get(prop, &paramVal, &paramFlag)))
    {
        CV_LOG_DEBUG(NULL, "Failed to get property " << prop);
        // we continue
    }
    // fallback - get default value
    long minVal, maxVal, stepVal;
    if (FAILED(ctrl->GetRange(prop, &minVal, &maxVal, &stepVal, &paramVal, &paramFlag)))
    {
        CV_LOG_DEBUG(NULL, "Failed to get default value for property " << prop);
        return false;
    }
    val = paramVal;
    return true;
}

double CvCapture_MSMF::getProperty( int property_id ) const
{
    long cVal = 0;
    if (isOpen)
        switch (property_id)
        {
        case CAP_PROP_MODE:
            return captureMode;
        case cv::CAP_PROP_HW_DEVICE:
            return hwDeviceIndex;
        case cv::CAP_PROP_HW_ACCELERATION:
            return static_cast<double>(va_type);
        case CAP_PROP_CONVERT_RGB:
                return convertFormat ? 1 : 0;
        case CAP_PROP_SAR_NUM:
                return captureVideoFormat.aspectRatioNum;
        case CAP_PROP_SAR_DEN:
                return captureVideoFormat.aspectRatioDenom;
        case CAP_PROP_FRAME_WIDTH:
            return captureVideoFormat.width;
        case CAP_PROP_FRAME_HEIGHT:
            return captureVideoFormat.height;
        case CAP_PROP_FOURCC:
            return captureVideoFormat.subType.Data1;
        case CAP_PROP_FPS:
            return captureVideoFormat.getFramerate();
        case CAP_PROP_FRAME_COUNT:
            if (duration != 0)
                return floor(((double)duration / 1e7)* captureVideoFormat.getFramerate() + 0.5);
            else
                break;
        case CAP_PROP_POS_FRAMES:
            return (double)nFrame;
        case CAP_PROP_POS_MSEC:
            return (double)usedVideoSampleTime / 1e4;
        case CAP_PROP_AUDIO_POS:
            return (double)audioSamplePos;
        case CAP_PROP_POS_AVI_RATIO:
            if (duration != 0)
                return (double)usedVideoSampleTime / duration;
            else
                break;
        case CAP_PROP_BRIGHTNESS:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Brightness, cVal))
                return cVal;
            break;
        case CAP_PROP_CONTRAST:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Contrast, cVal))
                return cVal;
            break;
        case CAP_PROP_SATURATION:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Saturation, cVal))
                return cVal;
            break;
        case CAP_PROP_HUE:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Hue, cVal))
                return cVal;
            break;
        case CAP_PROP_GAIN:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Gain, cVal))
                return cVal;
            break;
        case CAP_PROP_SHARPNESS:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Sharpness, cVal))
                return cVal;
            break;
        case CAP_PROP_GAMMA:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_Gamma, cVal))
                return cVal;
            break;
        case CAP_PROP_BACKLIGHT:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_BacklightCompensation, cVal))
                return cVal;
            break;
        case CAP_PROP_MONOCHROME:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_ColorEnable, cVal))
                return cVal == 0 ? 1 : 0;
            break;
        case CAP_PROP_TEMPERATURE:
            if (readComplexPropery<IAMVideoProcAmp>(VideoProcAmp_WhiteBalance, cVal))
                return cVal;
            break;
        case CAP_PROP_PAN:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Pan, cVal))
                return cVal;
            break;
        case CAP_PROP_TILT:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Tilt, cVal))
                return cVal;
            break;
        case CAP_PROP_ROLL:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Roll, cVal))
                return cVal;
            break;
        case CAP_PROP_IRIS:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Iris, cVal))
                return cVal;
            break;
        case CAP_PROP_EXPOSURE:
        case CAP_PROP_AUTO_EXPOSURE:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Exposure, cVal))
            {
                if (property_id == CAP_PROP_EXPOSURE)
                    return cVal;
                else
                    return cVal == VideoProcAmp_Flags_Auto;
            }
            break;
        case CAP_PROP_ZOOM:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Zoom, cVal))
                return cVal;
            break;
        case CAP_PROP_FOCUS:
        case CAP_PROP_AUTOFOCUS:
            if (readComplexPropery<IAMCameraControl>(CameraControl_Focus, cVal))
            {
                if (property_id == CAP_PROP_FOCUS)
                    return cVal;
                else
                    return cVal == VideoProcAmp_Flags_Auto;
            }
            break;
        case CAP_PROP_WHITE_BALANCE_BLUE_U:
        case CAP_PROP_WHITE_BALANCE_RED_V:
        case CAP_PROP_RECTIFICATION:
        case CAP_PROP_TRIGGER:
        case CAP_PROP_TRIGGER_DELAY:
        case CAP_PROP_GUID:
        case CAP_PROP_ISO_SPEED:
        case CAP_PROP_SETTINGS:
        case CAP_PROP_BUFFERSIZE:
        case CAP_PROP_AUDIO_BASE_INDEX:
            return audioBaseIndex;
        case CAP_PROP_AUDIO_TOTAL_STREAMS:
            return numberOfAudioStreams;
        case CAP_PROP_AUDIO_TOTAL_CHANNELS:
            return captureAudioFormat.nChannels;
        case CAP_PROP_AUDIO_SAMPLES_PER_SECOND:
            return captureAudioFormat.nSamplesPerSec;
        case CAP_PROP_AUDIO_DATA_DEPTH:
            return outputAudioFormat;
        case CAP_PROP_AUDIO_SHIFT_NSEC:
            return (double)(audioStartOffset - videoStartOffset)*1e2;
        default:
            break;
        }
    return -1;
}

template <typename CtrlT>
bool CvCapture_MSMF::writeComplexProperty(long prop, double val, long flags)
{
    _ComPtr<CtrlT> ctrl;
    if (FAILED(videoFileSource->GetServiceForStream((DWORD)MF_SOURCE_READER_MEDIASOURCE, GUID_NULL, IID_PPV_ARGS(&ctrl))))
    {
        CV_LOG_DEBUG(NULL, "Failed get service for stream");
        return false;
    }
    if (FAILED(ctrl->Set(prop, (long)val, flags)))
    {
        CV_LOG_DEBUG(NULL, "Failed to set property " << prop);
        return false;
    }
    return true;
}

bool CvCapture_MSMF::setProperty( int property_id, double value )
{
    MediaType newFormat = captureVideoFormat;
    if (isOpen)
        switch (property_id)
        {
        case CAP_PROP_MODE:
            switch ((MSMFCapture_Mode)((int)value))
            {
            case MODE_SW:
                return configureHW(false);
            case MODE_HW:
                return configureHW(true);
            default:
                return false;
            }
        case CAP_PROP_FOURCC:
            return configureVideoOutput(newFormat, (int)cvRound(value));
        case CAP_PROP_FORMAT:
            return configureVideoOutput(newFormat, (int)cvRound(value));
        case CAP_PROP_CONVERT_RGB:
            convertFormat = (value != 0);
            return configureVideoOutput(newFormat, outputVideoFormat);
        case CAP_PROP_SAR_NUM:
            if (value > 0)
            {
                newFormat.aspectRatioNum = (UINT32)cvRound(value);
                return configureVideoOutput(newFormat, outputVideoFormat);
            }
            break;
        case CAP_PROP_SAR_DEN:
            if (value > 0)
            {
                newFormat.aspectRatioDenom = (UINT32)cvRound(value);
                return configureVideoOutput(newFormat, outputVideoFormat);
            }
            break;
        case CAP_PROP_FRAME_WIDTH:
            if (value >= 0)
            {
                newFormat.width = (UINT32)cvRound(value);
                return configureVideoOutput(newFormat, outputVideoFormat);
            }
            break;
        case CAP_PROP_FRAME_HEIGHT:
            if (value >= 0)
            {
                newFormat.height = (UINT32)cvRound(value);
                return configureVideoOutput(newFormat, outputVideoFormat);
            }
            break;
        case CAP_PROP_FPS:
            if (value >= 0)
            {
                newFormat.setFramerate(value);
                return configureVideoOutput(newFormat, outputVideoFormat);
            }
            break;
        case CAP_PROP_FRAME_COUNT:
            break;
        case CAP_PROP_POS_AVI_RATIO:
            if (duration != 0)
                return setTime(duration * value, true);
            break;
        case CAP_PROP_POS_FRAMES:
            if (std::fabs(captureVideoFormat.getFramerate()) > 0)
                return setTime((int)value);
            break;
        case CAP_PROP_POS_MSEC:
                return setTime(value  * 1e4, false);
        case CAP_PROP_BRIGHTNESS:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Brightness, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_CONTRAST:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Contrast, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_SATURATION:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Saturation, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_HUE:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Hue, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_GAIN:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Gain, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_SHARPNESS:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Sharpness, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_GAMMA:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_Gamma, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_BACKLIGHT:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_BacklightCompensation, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_MONOCHROME:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_ColorEnable, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_TEMPERATURE:
            return writeComplexProperty<IAMVideoProcAmp>(VideoProcAmp_WhiteBalance, value, VideoProcAmp_Flags_Manual);
        case CAP_PROP_PAN:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Pan, value, CameraControl_Flags_Manual);
        case CAP_PROP_TILT:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Tilt, value, CameraControl_Flags_Manual);
        case CAP_PROP_ROLL:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Roll, value, CameraControl_Flags_Manual);
        case CAP_PROP_IRIS:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Iris, value, CameraControl_Flags_Manual);
        case CAP_PROP_EXPOSURE:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Exposure, value, CameraControl_Flags_Manual);
        case CAP_PROP_AUTO_EXPOSURE:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Exposure, value, value != 0 ? VideoProcAmp_Flags_Auto : VideoProcAmp_Flags_Manual);
        case CAP_PROP_ZOOM:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Zoom, value, CameraControl_Flags_Manual);
        case CAP_PROP_FOCUS:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Focus, value, CameraControl_Flags_Manual);
        case CAP_PROP_AUTOFOCUS:
            return writeComplexProperty<IAMCameraControl>(CameraControl_Focus, value, value != 0 ? CameraControl_Flags_Auto : CameraControl_Flags_Manual);
        case CAP_PROP_WHITE_BALANCE_BLUE_U:
        case CAP_PROP_WHITE_BALANCE_RED_V:
        case CAP_PROP_RECTIFICATION:
        case CAP_PROP_TRIGGER:
        case CAP_PROP_TRIGGER_DELAY:
        case CAP_PROP_GUID:
        case CAP_PROP_ISO_SPEED:
        case CAP_PROP_SETTINGS:
        case CAP_PROP_BUFFERSIZE:
        default:
            break;
        }
    return false;
}

cv::Ptr<cv::IVideoCapture> cv::cvCreateCapture_MSMF( int index, const cv::VideoCaptureParameters& params)
{
    cv::Ptr<CvCapture_MSMF> capture = cv::makePtr<CvCapture_MSMF>();
    if (capture)
    {
        capture->open(index, &params);
        if (capture->isOpened())
            return capture;
    }
    return cv::Ptr<cv::IVideoCapture>();
}

cv::Ptr<cv::IVideoCapture> cv::cvCreateCapture_MSMF(const cv::String& filename, const cv::VideoCaptureParameters& params)
{
    cv::Ptr<CvCapture_MSMF> capture = cv::makePtr<CvCapture_MSMF>();
    if (capture)
    {
        capture->open(filename, nullptr, &params);
        if (capture->isOpened())
            return capture;
    }
    return cv::Ptr<cv::IVideoCapture>();
}

cv::Ptr<cv::IVideoCapture> cv::cvCreateCapture_MSMF(const Ptr<IStreamReader>& stream, const cv::VideoCaptureParameters& params)
{
    cv::Ptr<CvCapture_MSMF> capture = cv::makePtr<CvCapture_MSMF>();
    if (capture)
    {
        capture->open(std::string(), stream, &params);
        if (capture->isOpened())
            return capture;
    }
    return cv::Ptr<cv::IVideoCapture>();
}

//
//
// Media Foundation-based Video Writer
//
//

class CvVideoWriter_MSMF : public cv::IVideoWriter
{
public:
    CvVideoWriter_MSMF();
    virtual ~CvVideoWriter_MSMF();
    virtual bool open(const cv::String& filename, int fourcc,
                      double fps, cv::Size frameSize, const cv::VideoWriterParameters& params);
    virtual void close();
    virtual void write(cv::InputArray);

    virtual double getProperty(int) const override;
    virtual bool setProperty(int, double) { return false; }
    virtual bool isOpened() const { return initiated; }

    int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_MSMF; }
private:
    Media_Foundation& MF;
    VideoAccelerationType va_type;
    int va_device;

    UINT32 videoWidth;
    UINT32 videoHeight;
    double fps;
    UINT32 bitRate;
    UINT32 frameSize;
    GUID   encodingFormat;
    GUID   inputFormat;

    DWORD  streamIndex;
    _ComPtr<IMFSinkWriter> sinkWriter;

    bool   initiated;

    LONGLONG rtStart;
    UINT64 rtDuration;

    static const GUID FourCC2GUID(int fourcc);
};

CvVideoWriter_MSMF::CvVideoWriter_MSMF():
    MF(Media_Foundation::getInstance()),
    va_type(VIDEO_ACCELERATION_NONE),
    va_device(-1),
    videoWidth(0),
    videoHeight(0),
    fps(0),
    bitRate(0),
    frameSize(0),
    encodingFormat(),
    inputFormat(),
    streamIndex(0),
    initiated(false),
    rtStart(0),
    rtDuration(0)
{
}

CvVideoWriter_MSMF::~CvVideoWriter_MSMF()
{
    close();
}

const GUID CvVideoWriter_MSMF::FourCC2GUID(int fourcc)
{
    switch(fourcc)
    {
        case CV_FOURCC_MACRO('d', 'v', '2', '5'):
            return MFVideoFormat_DV25; break;
        case CV_FOURCC_MACRO('d', 'v', '5', '0'):
            return MFVideoFormat_DV50; break;
        case CV_FOURCC_MACRO('d', 'v', 'c', ' '):
            return MFVideoFormat_DVC; break;
        case CV_FOURCC_MACRO('d', 'v', 'h', '1'):
            return MFVideoFormat_DVH1; break;
        case CV_FOURCC_MACRO('d', 'v', 'h', 'd'):
            return MFVideoFormat_DVHD; break;
        case CV_FOURCC_MACRO('d', 'v', 's', 'd'):
            return MFVideoFormat_DVSD; break;
        case CV_FOURCC_MACRO('d', 'v', 's', 'l'):
                return MFVideoFormat_DVSL; break;
#if (WINVER >= 0x0602)
        case CV_FOURCC_MACRO('H', '2', '6', '3'):   // Available only for Win 8 target.
                return MFVideoFormat_H263; break;
#endif
        case CV_FOURCC_MACRO('H', '2', '6', '4'):
                return MFVideoFormat_H264; break;
#if defined(NTDDI_WIN10)
        case CV_FOURCC_MACRO('H', '2', '6', '5'):
                return MFVideoFormat_H265;  break;
        case CV_FOURCC_MACRO('H', 'E', 'V', 'C'):
                return MFVideoFormat_HEVC;  break;
#endif
        case CV_FOURCC_MACRO('M', '4', 'S', '2'):
                return MFVideoFormat_M4S2; break;
        case CV_FOURCC_MACRO('M', 'J', 'P', 'G'):
                return MFVideoFormat_MJPG; break;
        case CV_FOURCC_MACRO('M', 'P', '4', '3'):
                return MFVideoFormat_MP43; break;
        case CV_FOURCC_MACRO('M', 'P', '4', 'S'):
                return MFVideoFormat_MP4S; break;
        case CV_FOURCC_MACRO('M', 'P', '4', 'V'):
                return MFVideoFormat_MP4V; break;
        case CV_FOURCC_MACRO('M', 'P', 'G', '1'):
                return MFVideoFormat_MPG1; break;
        case CV_FOURCC_MACRO('M', 'S', 'S', '1'):
                return MFVideoFormat_MSS1; break;
        case CV_FOURCC_MACRO('M', 'S', 'S', '2'):
                return MFVideoFormat_MSS2; break;
        case CV_FOURCC_MACRO('W', 'M', 'V', '1'):
                return MFVideoFormat_WMV1; break;
        case CV_FOURCC_MACRO('W', 'M', 'V', '2'):
                return MFVideoFormat_WMV2; break;
        case CV_FOURCC_MACRO('W', 'M', 'V', '3'):
                return MFVideoFormat_WMV3; break;
        case CV_FOURCC_MACRO('W', 'V', 'C', '1'):
                return MFVideoFormat_WVC1; break;
        default:
            return MFVideoFormat_H264;
    }
}

bool CvVideoWriter_MSMF::open( const cv::String& filename, int fourcc,
                               double _fps, cv::Size _frameSize, const cv::VideoWriterParameters& params)
{
    if (initiated)
        close();

    if (params.has(VIDEOWRITER_PROP_HW_ACCELERATION))
    {
        va_type = params.get<VideoAccelerationType>(VIDEOWRITER_PROP_HW_ACCELERATION);
        if (va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: MSMF backend doesn't support writer acceleration support. Can't handle VIDEOWRITER_PROP_HW_ACCELERATION parameter. Bailout");
            return false;
        }
    }
    if (params.has(VIDEOWRITER_PROP_HW_DEVICE))
    {
        va_device = params.get<int>(VIDEOWRITER_PROP_HW_DEVICE);
        if (va_type == VIDEO_ACCELERATION_NONE && va_device != -1)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: Invalid usage of VIDEOWRITER_PROP_HW_DEVICE without requested H/W acceleration. Bailout");
            return false;
        }
        if (va_type == VIDEO_ACCELERATION_ANY && va_device != -1)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: Invalid usage of VIDEOWRITER_PROP_HW_DEVICE with 'ANY' H/W acceleration. Bailout");
            return false;
        }
        if (va_device != -1)
        {
            CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: VIDEOWRITER_PROP_HW_DEVICE is not supported. Specify -1 (auto) value. Bailout");
            return false;
        }
    }

    videoWidth = _frameSize.width;
    videoHeight = _frameSize.height;
    fps = _fps;
    bitRate = (UINT32)fps*videoWidth*videoHeight; // 1-bit per pixel
    encodingFormat = FourCC2GUID(fourcc);
    inputFormat = MFVideoFormat_RGB32;

    _ComPtr<IMFMediaType>  mediaTypeOut;
    _ComPtr<IMFMediaType>  mediaTypeIn;
    _ComPtr<IMFAttributes> spAttr;
    if (// Set the output media type.
        SUCCEEDED(MFCreateMediaType(&mediaTypeOut)) &&
        SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
        SUCCEEDED(mediaTypeOut->SetGUID(MF_MT_SUBTYPE, encodingFormat)) &&
        SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_AVG_BITRATE, bitRate)) &&
        SUCCEEDED(mediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)) &&
        SUCCEEDED(MFSetAttributeSize(mediaTypeOut.Get(), MF_MT_FRAME_SIZE, videoWidth, videoHeight)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_FRAME_RATE, (UINT32)(fps * 1000), 1000)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeOut.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) &&
        // Set the input media type.
        SUCCEEDED(MFCreateMediaType(&mediaTypeIn)) &&
        SUCCEEDED(mediaTypeIn->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) &&
        SUCCEEDED(mediaTypeIn->SetGUID(MF_MT_SUBTYPE, inputFormat)) &&
        SUCCEEDED(mediaTypeIn->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)) &&
        SUCCEEDED(mediaTypeIn->SetUINT32(MF_MT_DEFAULT_STRIDE, 4 * videoWidth)) && //Assume BGR32 input
        SUCCEEDED(MFSetAttributeSize(mediaTypeIn.Get(), MF_MT_FRAME_SIZE, videoWidth, videoHeight)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeIn.Get(), MF_MT_FRAME_RATE, (UINT32)(fps * 1000), 1000)) &&
        SUCCEEDED(MFSetAttributeRatio(mediaTypeIn.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) &&
        // Set sink writer parameters
        SUCCEEDED(MFCreateAttributes(&spAttr, 10)) &&
        SUCCEEDED(spAttr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, true)) &&
        SUCCEEDED(spAttr->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, true))
        )
    {
        // Create the sink writer
        cv::AutoBuffer<wchar_t> unicodeFileName(filename.length() + 1);
        MultiByteToWideChar(CP_ACP, 0, filename.c_str(), -1, unicodeFileName.data(), (int)filename.length() + 1);
        HRESULT hr = MFCreateSinkWriterFromURL(unicodeFileName.data(), NULL, spAttr.Get(), &sinkWriter);
        if (SUCCEEDED(hr))
        {
            // Configure the sink writer and tell it start to start accepting data
            if (SUCCEEDED(sinkWriter->AddStream(mediaTypeOut.Get(), &streamIndex)) &&
                SUCCEEDED(sinkWriter->SetInputMediaType(streamIndex, mediaTypeIn.Get(), NULL)) &&
                SUCCEEDED(sinkWriter->BeginWriting()))
            {
                initiated = true;
                rtStart = 0;
                MFFrameRateToAverageTimePerFrame((UINT32)(fps * 1000), 1000, &rtDuration);

                VideoAccelerationType actual_va_type = VIDEO_ACCELERATION_NONE;
                if (va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
                {
                    if (va_type != actual_va_type)
                    {
                        CV_LOG_ERROR(NULL, "VIDEOIO/MSMF: Can't select requested video acceleration through VIDEOWRITER_PROP_HW_ACCELERATION: "
                                << va_type << " (actual is " << actual_va_type << "). Bailout");
                        close();
                        return false;
                    }
                }
                else
                {
                    va_type = actual_va_type;
                }

                return true;
            }
        }
    }

    return false;
}

void CvVideoWriter_MSMF::close()
{
    if (initiated)
    {
        initiated = false;
        sinkWriter->Finalize();
        sinkWriter.Release();
    }
}

void CvVideoWriter_MSMF::write(cv::InputArray img)
{
    if (img.empty() ||
        (img.channels() != 1 && img.channels() != 3 && img.channels() != 4) ||
        (UINT32)img.cols() != videoWidth || (UINT32)img.rows() != videoHeight)
        return;

    const LONG cbWidth = 4 * videoWidth;
    const DWORD cbBuffer = cbWidth * videoHeight;
    _ComPtr<IMFSample> sample;
    _ComPtr<IMFMediaBuffer> buffer;
    BYTE *pData = NULL;
    // Prepare a media sample.
    if (SUCCEEDED(MFCreateSample(&sample)) &&
        // Set sample time stamp and duration.
        SUCCEEDED(sample->SetSampleTime(rtStart)) &&
        SUCCEEDED(sample->SetSampleDuration(rtDuration)) &&
        // Create a memory buffer.
        SUCCEEDED(MFCreateMemoryBuffer(cbBuffer, &buffer)) &&
        // Set the data length of the buffer.
        SUCCEEDED(buffer->SetCurrentLength(cbBuffer)) &&
        // Add the buffer to the sample.
        SUCCEEDED(sample->AddBuffer(buffer.Get())) &&
        // Lock the buffer.
        SUCCEEDED(buffer->Lock(&pData, NULL, NULL)))
    {
        // Copy the video frame to the buffer.
        cv::cvtColor(img.getMat(), cv::Mat(videoHeight, videoWidth, CV_8UC4, pData, cbWidth), img.channels() > 1 ? cv::COLOR_BGR2BGRA : cv::COLOR_GRAY2BGRA);
        buffer->Unlock();
        // Send media sample to the Sink Writer.
        if (SUCCEEDED(sinkWriter->WriteSample(streamIndex, sample.Get())))
        {
            rtStart += rtDuration;
        }
    }
}


double CvVideoWriter_MSMF::getProperty(int propId) const
{
    if (propId == VIDEOWRITER_PROP_HW_ACCELERATION)
    {
        return static_cast<double>(va_type);
    }
    else if (propId == VIDEOWRITER_PROP_HW_DEVICE)
    {
        return static_cast<double>(va_device);
    }
    return 0;
}

cv::Ptr<cv::IVideoWriter> cv::cvCreateVideoWriter_MSMF( const std::string& filename, int fourcc,
                                                        double fps, const cv::Size& frameSize,
                                                        const VideoWriterParameters& params)
{
    cv::Ptr<CvVideoWriter_MSMF> writer = cv::makePtr<CvVideoWriter_MSMF>();
    if (writer)
    {
        writer->open(filename, fourcc, fps, frameSize, params);
        if (writer->isOpened())
            return writer;
    }
    return cv::Ptr<cv::IVideoWriter>();
}

#if defined(BUILD_PLUGIN)

#define NEW_PLUGIN

#ifndef NEW_PLUGIN
#define ABI_VERSION 0
#define API_VERSION 0
#include "plugin_api.hpp"
#else
#define CAPTURE_ABI_VERSION 1
#define CAPTURE_API_VERSION 2
#include "plugin_capture_api.hpp"
#define WRITER_ABI_VERSION 1
#define WRITER_API_VERSION 1
#include "plugin_writer_api.hpp"
#endif

namespace cv {

typedef CvCapture_MSMF CaptureT;
typedef CvVideoWriter_MSMF WriterT;

static
CvResult CV_API_CALL cv_capture_open_with_params(
    const char* filename, int camera_index,
    int* params, unsigned n_params,
    CV_OUT CvPluginCapture* handle
)
{
    if (!handle)
        return CV_ERROR_FAIL;
    *handle = NULL;
    CaptureT* cap = 0;
    try
    {
        cv::VideoCaptureParameters parameters(params, n_params);
        cap = new CaptureT();
        bool res;
        if (filename)
        {
            res = cap->open(std::string(filename), nullptr, &parameters);
        }
        else
            res = cap->open(camera_index, &parameters);
        if (res)
        {
            *handle = (CvPluginCapture)cap;
            return CV_ERROR_OK;
        }
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
    }
    if (cap)
        delete cap;
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_capture_open_buffer(
    void* opaque,
    long long(*read)(void* opaque, char* buffer, long long size),
    long long(*seek)(void* opaque, long long offset, int way),
    int* params, unsigned n_params,
    CV_OUT CvPluginCapture* handle
)
{
    if (!handle)
        return CV_ERROR_FAIL;

    *handle = NULL;
    CaptureT* cap = 0;
    try
    {
        cv::VideoCaptureParameters parameters(params, n_params);
        cap = new CaptureT();
        bool res = cap->open(std::string(), makePtr<PluginStreamReader>(opaque, read, seek), &parameters);
        if (res)
        {
            *handle = (CvPluginCapture)cap;
            return CV_ERROR_OK;
        }
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
    }
    if (cap)
        delete cap;
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_capture_open(const char* filename, int camera_index, CV_OUT CvPluginCapture* handle)
{
    return cv_capture_open_with_params(filename, camera_index, NULL, 0, handle);
}

static
CvResult CV_API_CALL cv_capture_release(CvPluginCapture handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    CaptureT* instance = (CaptureT*)handle;
    delete instance;
    return CV_ERROR_OK;
}


static
CvResult CV_API_CALL cv_capture_get_prop(CvPluginCapture handle, int prop, CV_OUT double* val)
{
    if (!handle)
        return CV_ERROR_FAIL;
    if (!val)
        return CV_ERROR_FAIL;
    try
    {
        CaptureT* instance = (CaptureT*)handle;
        *val = instance->getProperty(prop);
        return CV_ERROR_OK;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_capture_set_prop(CvPluginCapture handle, int prop, double val)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CaptureT* instance = (CaptureT*)handle;
        return instance->setProperty(prop, val) ? CV_ERROR_OK : CV_ERROR_FAIL;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_capture_grab(CvPluginCapture handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CaptureT* instance = (CaptureT*)handle;
        return instance->grabFrame() ? CV_ERROR_OK : CV_ERROR_FAIL;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_capture_retrieve(CvPluginCapture handle, int stream_idx, cv_videoio_capture_retrieve_cb_t callback, void* userdata)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CaptureT* instance = (CaptureT*)handle;
        Mat img;
        if (instance->retrieveFrame(stream_idx, img))
#ifndef NEW_PLUGIN
            return callback(stream_idx, img.data, (int)img.step, img.cols, img.rows, img.channels(), userdata);
#else
            return callback(stream_idx, img.data, (int)img.step, img.cols, img.rows, img.type(), userdata);
#endif
        return CV_ERROR_FAIL;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_writer_open_with_params(
    const char* filename, int fourcc, double fps, int width, int height,
    int* params, unsigned n_params,
    CV_OUT CvPluginWriter* handle)
{
    WriterT* wrt = 0;
    try
    {
        VideoWriterParameters parameters(params, n_params);
        wrt = new WriterT();
        Size sz(width, height);
        if (wrt && wrt->open(filename, fourcc, fps, sz, parameters))
        {
            *handle = (CvPluginWriter)wrt;
            return CV_ERROR_OK;
        }
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
    }
    if (wrt)
        delete wrt;
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_writer_open(const char* filename, int fourcc, double fps, int width, int height, int isColor,
    CV_OUT CvPluginWriter* handle)
{
    int params[2] = { VIDEOWRITER_PROP_IS_COLOR, isColor };
    return cv_writer_open_with_params(filename, fourcc, fps, width, height, params, 1, handle);
}

static
CvResult CV_API_CALL cv_writer_release(CvPluginWriter handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    WriterT* instance = (WriterT*)handle;
    delete instance;
    return CV_ERROR_OK;
}

static
CvResult CV_API_CALL cv_writer_get_prop(CvPluginWriter handle, int prop, CV_OUT double* val)
{
    if (!handle)
        return CV_ERROR_FAIL;
    if (!val)
        return CV_ERROR_FAIL;
    try
    {
        WriterT* instance = (WriterT*)handle;
        *val = instance->getProperty(prop);
        return CV_ERROR_OK;
    }
    catch (...)
    {
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_writer_set_prop(CvPluginWriter /*handle*/, int /*prop*/, double /*val*/)
{
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_writer_write(CvPluginWriter handle, const unsigned char* data, int step, int width, int height, int cn)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CV_Assert(step >= 0);
        WriterT* instance = (WriterT*)handle;
        Size sz(width, height);
        Mat img(sz, CV_MAKETYPE(CV_8U, cn), (void*)data, (size_t)step);
        instance->write(img);
        return CV_ERROR_OK;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "MSMF: Exception is raised: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_WARNING(NULL, "MSMF: Unknown C++ exception is raised");
        return CV_ERROR_FAIL;
    }
}

} // namespace

#ifndef NEW_PLUGIN

static const OpenCV_VideoIO_Plugin_API_preview plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Plugin_API_preview), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "Microsoft Media Foundation OpenCV Video I/O plugin"
    },
    {
        /*  1*/cv::CAP_MSMF,
        /*  2*/cv::cv_capture_open,
        /*  3*/cv::cv_capture_release,
        /*  4*/cv::cv_capture_get_prop,
        /*  5*/cv::cv_capture_set_prop,
        /*  6*/cv::cv_capture_grab,
        /*  7*/cv::cv_capture_retrieve,
        /*  8*/cv::cv_writer_open,
        /*  9*/cv::cv_writer_release,
        /* 10*/cv::cv_writer_get_prop,
        /* 11*/cv::cv_writer_set_prop,
        /* 12*/cv::cv_writer_write
    }
};

const OpenCV_VideoIO_Plugin_API_preview* opencv_videoio_plugin_init_v0(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == ABI_VERSION && requested_api_version <= API_VERSION)
        return &plugin_api;
    return NULL;
}

#else  // NEW_PLUGIN

static const OpenCV_VideoIO_Capture_Plugin_API capture_plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Capture_Plugin_API), CAPTURE_ABI_VERSION, CAPTURE_API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "Microsoft Media Foundation OpenCV Video I/O plugin"
    },
    {
        /*  1*/cv::CAP_MSMF,
        /*  2*/cv::cv_capture_open,
        /*  3*/cv::cv_capture_release,
        /*  4*/cv::cv_capture_get_prop,
        /*  5*/cv::cv_capture_set_prop,
        /*  6*/cv::cv_capture_grab,
        /*  7*/cv::cv_capture_retrieve,
    },
    {
        /*  8*/cv::cv_capture_open_with_params,
    },
    {
        /*  9*/cv::cv_capture_open_buffer,
    }
};

const OpenCV_VideoIO_Capture_Plugin_API* opencv_videoio_capture_plugin_init_v1(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == CAPTURE_ABI_VERSION && requested_api_version <= CAPTURE_API_VERSION)
        return &capture_plugin_api;
    return NULL;
}

static const OpenCV_VideoIO_Writer_Plugin_API writer_plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Writer_Plugin_API), WRITER_ABI_VERSION, WRITER_API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "Microsoft Media Foundation OpenCV Video I/O plugin"
    },
    {
        /*  1*/cv::CAP_MSMF,
        /*  2*/cv::cv_writer_open,
        /*  3*/cv::cv_writer_release,
        /*  4*/cv::cv_writer_get_prop,
        /*  5*/cv::cv_writer_set_prop,
        /*  6*/cv::cv_writer_write
    },
    {
        /*  7*/cv::cv_writer_open_with_params
    }
};

const OpenCV_VideoIO_Writer_Plugin_API* opencv_videoio_writer_plugin_init_v1(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == WRITER_ABI_VERSION && requested_api_version <= WRITER_API_VERSION)
        return &writer_plugin_api;
    return NULL;
}

#endif  // NEW_PLUGIN

#endif // BUILD_PLUGIN
