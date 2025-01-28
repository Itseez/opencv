// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include "opencv2/videoio.hpp"
#ifdef HAVE_OPENCL
#include "opencv2/core/ocl.hpp"
#endif
#if defined(__OPENCV_BUILD) && !defined(BUILD_PLUGIN)  // TODO Properly detect and add D3D11 / LIBVA dependencies for standalone plugins
#include "cvconfig.h"
#endif
#include <sstream>

#if defined(HAVE_MFX) && defined(HAVE_ONEVPL)
#undef HAVE_MFX  // libav's hwcontext_qsv.h doesn't expect oneVPL headers
#endif

#ifdef HAVE_D3D11
#define D3D11_NO_HELPERS
#include <d3d11.h>
#include <codecvt>
#include <locale>
#include "opencv2/core/directx.hpp"
#ifdef HAVE_OPENCL
#include <CL/cl_d3d11.h>
#endif
#endif // HAVE_D3D11

#ifdef HAVE_VA
#include <va/va_backend.h>
#ifdef HAVE_VA_INTEL
#include "opencv2/core/va_intel.hpp"
#include "opencv2/core/detail/va_intel_interop.hpp"
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef HAVE_VA_INTEL_OLD_HEADER
#include <CL/va_ext.h>
#else
#include <CL/cl_va_api_media_sharing_intel.h>
#endif
#endif
#endif // HAVE_VA

// FFMPEG "C" headers
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#ifdef HAVE_D3D11
#include <libavutil/hwcontext_d3d11va.h>
#endif
#ifdef HAVE_VA
#include <libavutil/hwcontext_vaapi.h>
#endif
#ifdef HAVE_MFX // dependency only on MFX header files, no linkage dependency
#include <libavutil/hwcontext_qsv.h>
#endif
}

#define HW_DEFAULT_POOL_SIZE    32
#define HW_DEFAULT_SW_FORMAT    AV_PIX_FMT_NV12

using namespace cv;

static AVCodec *hw_find_codec(AVCodecID id, AVHWDeviceType hw_type, int (*check_category)(const AVCodec *),
                              const char *disabled_codecs, AVPixelFormat *hw_pix_fmt);
static AVBufferRef* hw_create_device(AVHWDeviceType hw_type, int hw_device, const std::string& device_subname, bool use_opencl);
static AVBufferRef* hw_create_frames(struct AVCodecContext* ctx, AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format);
static AVPixelFormat hw_get_format_callback(struct AVCodecContext *ctx, const enum AVPixelFormat * fmt);
static VideoAccelerationType hw_type_to_va_type(AVHWDeviceType hw_type);

static
const char* getVideoAccelerationName(VideoAccelerationType va_type)
{
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: return "none";
    case VIDEO_ACCELERATION_ANY: return "any";
    case VIDEO_ACCELERATION_D3D11: return "d3d11";
    case VIDEO_ACCELERATION_VAAPI: return "vaapi";
    case VIDEO_ACCELERATION_MFX: return "mfx";
    }
    return "unknown";
}

static
std::string getDecoderConfiguration(VideoAccelerationType va_type, AVDictionary *dict)
{
    std::string va_name = getVideoAccelerationName(va_type);
    std::string key_name = std::string("hw_decoders_") + va_name;
    const char *hw_acceleration = NULL;
    if (dict)
    {
        AVDictionaryEntry* entry = av_dict_get(dict, key_name.c_str(), NULL, 0);
        if (entry)
            hw_acceleration = entry->value;
    }
    if (hw_acceleration)
        return hw_acceleration;

    // some default values (FFMPEG_DECODE_ACCELERATION_TYPES)
#ifdef _WIN32
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: return "";
    case VIDEO_ACCELERATION_ANY: return "d3d11va";
    case VIDEO_ACCELERATION_D3D11: return "d3d11va";
    case VIDEO_ACCELERATION_VAAPI: return "";
    case VIDEO_ACCELERATION_MFX: return ""; // "qsv" fails if non-Intel D3D11 device
    }
    return "";
#else
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: return "";
    case VIDEO_ACCELERATION_ANY: return "vaapi.iHD";
    case VIDEO_ACCELERATION_D3D11: return "";
    case VIDEO_ACCELERATION_VAAPI: return "vaapi.iHD";
    case VIDEO_ACCELERATION_MFX: return "qsv.iHD";
    }
    return "";
#endif
}

static
std::string getEncoderConfiguration(VideoAccelerationType va_type, AVDictionary *dict)
{
    std::string va_name = getVideoAccelerationName(va_type);
    std::string key_name = std::string("hw_encoders_") + va_name;
    const char *hw_acceleration = NULL;
    if (dict)
    {
        AVDictionaryEntry* entry = av_dict_get(dict, key_name.c_str(), NULL, 0);
        if (entry)
            hw_acceleration = entry->value;
    }
    if (hw_acceleration)
        return hw_acceleration;

    // some default values (FFMPEG_ENCODE_ACCELERATION_TYPES)
#ifdef _WIN32
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: return "";
    case VIDEO_ACCELERATION_ANY: return "qsv";
    case VIDEO_ACCELERATION_D3D11: return "";
    case VIDEO_ACCELERATION_VAAPI: return "";
    case VIDEO_ACCELERATION_MFX: return "qsv";
    }
    return "";
#else
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: return "";
    case VIDEO_ACCELERATION_ANY: return "qsv.iHD,vaapi.iHD";
    case VIDEO_ACCELERATION_D3D11: return "";
    case VIDEO_ACCELERATION_VAAPI: return "vaapi.iHD";
    case VIDEO_ACCELERATION_MFX: return "qsv.iHD";
    }
    return "unknown";
#endif
}

static
std::string getDecoderDisabledCodecs(AVDictionary *dict)
{
    std::string key_name = std::string("hw_disable_decoders");
    const char *disabled_codecs = NULL;
    if (dict)
    {
        AVDictionaryEntry* entry = av_dict_get(dict, key_name.c_str(), NULL, 0);
        if (entry)
            disabled_codecs = entry->value;
    }
    if (disabled_codecs)
        return disabled_codecs;

    // some default values (FFMPEG_DECODE_DISABLE_CODECS)
#ifdef _WIN32
    return "none";
#else
    return "av1.vaapi,av1_qsv,vp8.vaapi,vp8_qsv";  // "vp9_qsv"
#endif
}

static
std::string getEncoderDisabledCodecs(AVDictionary *dict)
{
    std::string key_name = std::string("hw_disabled_encoders");
    const char *disabled_codecs = NULL;
    if (dict)
    {
        AVDictionaryEntry* entry = av_dict_get(dict, key_name.c_str(), NULL, 0);
        if (entry)
            disabled_codecs = entry->value;
    }
    if (disabled_codecs)
        return disabled_codecs;

    // some default values (FFMPEG_ENCODE_DISABLE_CODECS)
#ifdef _WIN32
    return "mjpeg_qsv";
#else
    return "mjpeg_vaapi,mjpeg_qsv,vp8_vaapi";
#endif
}

static
bool hw_check_device(AVBufferRef* ctx, AVHWDeviceType hw_type, const std::string& device_subname) {
    if (!ctx)
        return false;
    AVHWDeviceContext* hw_device_ctx = (AVHWDeviceContext*)ctx->data;
    if (!hw_device_ctx->hwctx)
        return false;
    const char *hw_name = av_hwdevice_get_type_name(hw_type);
    if (hw_type == AV_HWDEVICE_TYPE_QSV)
        hw_name = "MFX";
    bool ret = true;
    std::string device_name;
#if defined(HAVE_D3D11)
    if (hw_device_ctx->type == AV_HWDEVICE_TYPE_D3D11VA) {
        ID3D11Device* device = ((AVD3D11VADeviceContext*)hw_device_ctx->hwctx)->device;
        IDXGIDevice* dxgiDevice = nullptr;
        if (device && SUCCEEDED(device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgiDevice)))) {
            IDXGIAdapter* adapter = nullptr;
            if (SUCCEEDED(dxgiDevice->GetAdapter(&adapter))) {
                DXGI_ADAPTER_DESC desc;
                if (SUCCEEDED(adapter->GetDesc(&desc))) {
                    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
                    device_name = conv.to_bytes(desc.Description);
                }
                adapter->Release();
            }
            dxgiDevice->Release();
        }
    }
#endif
    if (hw_device_ctx->type == AV_HWDEVICE_TYPE_VAAPI) {
#if defined(HAVE_VA) && (VA_MAJOR_VERSION >= 1)
        VADisplay display = ((AVVAAPIDeviceContext *) hw_device_ctx->hwctx)->display;
        if (display) {
            VADriverContext *va_ctx = ((VADisplayContext *) display)->pDriverContext;
            device_name = va_ctx->str_vendor;
            if (hw_type == AV_HWDEVICE_TYPE_QSV) {
                // Workaround for issue fixed in MediaSDK 21.x https://github.com/Intel-Media-SDK/MediaSDK/issues/2595
                // Checks VAAPI driver for support of VideoProc operation required by MediaSDK
                ret = false;
                int n_entrypoints = va_ctx->max_entrypoints;
                std::vector<VAEntrypoint> entrypoints(n_entrypoints);
                if (va_ctx->vtable->vaQueryConfigEntrypoints(va_ctx, VAProfileNone, entrypoints.data(), &n_entrypoints) == VA_STATUS_SUCCESS) {
                    for (int i = 0; i < n_entrypoints; i++) {
                        if (entrypoints[i] == VAEntrypointVideoProc) {
                            ret = true;
                            break;
                        }
                    }
                }
                if (!ret)
                    CV_LOG_INFO(NULL, "FFMPEG: Skipping MFX video acceleration as entrypoint VideoProc not found in: " << device_name);
            }
        }
#else
        ret = (hw_type != AV_HWDEVICE_TYPE_QSV); // disable MFX if we can't check VAAPI for VideoProc entrypoint
#endif
    }
    if (ret && !device_subname.empty() && device_name.find(device_subname) == std::string::npos)
    {
        CV_LOG_INFO(NULL, "FFMPEG: Skipping '" << hw_name <<
            "' video acceleration on the following device name as not matching substring '" << device_subname << "': " << device_name);
        ret = false;  // reject configuration
    }
    if (ret)
    {
        if (!device_name.empty()) {
            CV_LOG_INFO(NULL, "FFMPEG: Using " << hw_name << " video acceleration on device: " << device_name);
        } else {
            CV_LOG_INFO(NULL, "FFMPEG: Using " << hw_name << " video acceleration");
        }
    }
    return ret;
}

static
AVBufferRef* hw_create_derived_context(AVHWDeviceType hw_type, AVBufferRef* hw_device_ctx) {
    AVBufferRef* derived_ctx = NULL;
    const char* hw_name = av_hwdevice_get_type_name(hw_type);
    int err = av_hwdevice_ctx_create_derived(&derived_ctx, hw_type, hw_device_ctx, 0);
    if (!derived_ctx || err < 0)
    {
        if (derived_ctx)
            av_buffer_unref(&derived_ctx);
        CV_LOG_INFO(NULL, "FFMPEG: Failed to create derived video acceleration (av_hwdevice_ctx_create_derived) for " << hw_name << ". Error=" << err);
        return NULL;
    }
    else
    {
        // Store child context in 'user_opaque' field of parent context.
        struct FreeChildContext {
            static void free(struct AVHWDeviceContext* ctx) {
                AVBufferRef* child_ctx = (AVBufferRef*)ctx->user_opaque;
                if (child_ctx)
                    av_buffer_unref(&child_ctx);
            }
        };
        AVHWDeviceContext* ctx = (AVHWDeviceContext*)derived_ctx->data;
        ctx->user_opaque = av_buffer_ref(hw_device_ctx);
        ctx->free = FreeChildContext::free;
        CV_LOG_INFO(NULL, "FFMPEG: Created derived video acceleration context (av_hwdevice_ctx_create_derived) for " << hw_name);
        return derived_ctx;
    }
}

#ifdef HAVE_OPENCL // GPU buffer interop with cv::UMat

// FFmpeg context attached to OpenCL context
class OpenCL_FFMPEG_Context : public ocl::Context::UserContext {
public:
    OpenCL_FFMPEG_Context(AVBufferRef* ctx) {
        ctx_ = av_buffer_ref(ctx);
    }
    virtual ~OpenCL_FFMPEG_Context() {
        av_buffer_unref(&ctx_);
    }
    AVBufferRef* GetAVHWDevice() {
        return ctx_;
    }
private:
    AVBufferRef* ctx_;
};

#ifdef HAVE_MFX
static
int hw_find_qsv_surface_index(AVFrame* hw_frame)
{
    if (AV_PIX_FMT_QSV != hw_frame->format)
        return -1;
    mfxFrameSurface1* surface = (mfxFrameSurface1*)hw_frame->data[3]; // As defined by AV_PIX_FMT_QSV
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frame->hw_frames_ctx->data;
    AVQSVFramesContext* qsv_ctx = (AVQSVFramesContext*)frames_ctx->hwctx;
    for (int i = 0; i < qsv_ctx->nb_surfaces; i++) {
        if (surface == qsv_ctx->surfaces + i) {
            return i;
        }
    }
    return -1;
}
#endif

#ifdef HAVE_VA
static
VADisplay hw_get_va_display(AVHWDeviceContext* hw_device_ctx)
{
    if (hw_device_ctx->type == AV_HWDEVICE_TYPE_QSV) { // we stored pointer to child context in 'user_opaque' field
        AVBufferRef* ctx = (AVBufferRef*)hw_device_ctx->user_opaque;
        hw_device_ctx = (AVHWDeviceContext*)ctx->data;
    }
    if (hw_device_ctx && hw_device_ctx->type == AV_HWDEVICE_TYPE_VAAPI) {
        return ((AVVAAPIDeviceContext*)hw_device_ctx->hwctx)->display;
    }
    return NULL;
}
#endif // HAVE_VA

#ifdef HAVE_VA_INTEL
static
VASurfaceID hw_get_va_surface(AVFrame* hw_frame) {
    if (AV_PIX_FMT_VAAPI == hw_frame->format) {
        return (VASurfaceID)(size_t)hw_frame->data[3]; // As defined by AV_PIX_FMT_VAAPI
    }
#ifdef HAVE_MFX
    else if (AV_PIX_FMT_QSV == hw_frame->format) {
        int frame_idx = hw_find_qsv_surface_index(hw_frame);
        if (frame_idx >= 0) { // frame index is same in parent (QSV) and child (VAAPI) frame context
            AVHWFramesContext *frames_ctx = (AVHWFramesContext *) hw_frame->hw_frames_ctx->data;
            AVHWFramesContext *child_ctx = (AVHWFramesContext *) frames_ctx->user_opaque;
            if (child_ctx && AV_HWDEVICE_TYPE_VAAPI == child_ctx->device_ctx->type) {
                AVVAAPIFramesContext *vaapi_ctx = (AVVAAPIFramesContext *) child_ctx->hwctx;
                CV_Assert(frame_idx < vaapi_ctx->nb_surfaces);
                return vaapi_ctx->surface_ids[frame_idx];
            }
        }
    }
#endif // HAVE_MFX
    return VA_INVALID_SURFACE;
}
#endif // HAVE_VA_INTEL

#ifdef HAVE_D3D11
static
AVD3D11VADeviceContext* hw_get_d3d11_device_ctx(AVHWDeviceContext* hw_device_ctx) {
    if (AV_HWDEVICE_TYPE_QSV == hw_device_ctx->type) { // we stored pointer to child context in 'user_opaque' field
        AVBufferRef* ctx = (AVBufferRef*)hw_device_ctx->user_opaque;
        hw_device_ctx = (AVHWDeviceContext*)ctx->data;
    }
    if (AV_HWDEVICE_TYPE_D3D11VA == hw_device_ctx->type) {
        return (AVD3D11VADeviceContext*)hw_device_ctx->hwctx;
    }
    return NULL;
}

ID3D11Texture2D* hw_get_d3d11_texture(AVFrame* hw_frame, int* subresource) {
    ID3D11Texture2D* texture = NULL;
    if (AV_PIX_FMT_D3D11 == hw_frame->format) {
        texture = (ID3D11Texture2D*)hw_frame->data[0]; // As defined by AV_PIX_FMT_D3D11
        *subresource = (intptr_t)hw_frame->data[1]; // As defined by AV_PIX_FMT_D3D11
    }
#ifdef HAVE_MFX
    else if (AV_PIX_FMT_QSV == hw_frame->format) {
        AVHWFramesContext *frames_ctx = (AVHWFramesContext *) hw_frame->hw_frames_ctx->data;
        AVHWFramesContext *child_ctx = (AVHWFramesContext *) frames_ctx->user_opaque;
        if (child_ctx && AV_HWDEVICE_TYPE_D3D11VA == child_ctx->device_ctx->type) {
            texture = ((AVD3D11VAFramesContext*)child_ctx->hwctx)->texture;
        }
        *subresource = hw_find_qsv_surface_index(hw_frame);
        CV_Assert(*subresource >= 0);
    }
#endif
    return texture;
}

// In D3D11 case we allocate additional texture as single texture (not texture array) because
// OpenCL interop with D3D11 doesn't support/work with NV12 sub-texture of texture array.
ID3D11Texture2D* hw_get_d3d11_single_texture(AVFrame* hw_frame, AVD3D11VADeviceContext* d3d11_device_ctx, ID3D11Texture2D* texture) {
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frame->hw_frames_ctx->data;
    if (AV_HWDEVICE_TYPE_QSV == frames_ctx->device_ctx->type) {
        frames_ctx = (AVHWFramesContext*)frames_ctx->user_opaque; // we stored pointer to child context in 'user_opaque' field
    }
    if (!frames_ctx || AV_HWDEVICE_TYPE_D3D11VA != frames_ctx->device_ctx->type) {
        return NULL;
    }
    ID3D11Texture2D* singleTexture = (ID3D11Texture2D*)frames_ctx->user_opaque;
    if (!singleTexture && d3d11_device_ctx && texture) {
        D3D11_TEXTURE2D_DESC desc = {};
        texture->GetDesc(&desc);
        desc.ArraySize = 1;
        desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
        if (SUCCEEDED(d3d11_device_ctx->device->CreateTexture2D(&desc, NULL, &singleTexture))) {
            frames_ctx->user_opaque = singleTexture;
        }
    }
    return singleTexture;
}
#endif // HAVE_D3D11

static
AVHWDeviceType hw_check_opencl_context(AVHWDeviceContext* ctx) {
    ocl::OpenCLExecutionContext& ocl_context = ocl::OpenCLExecutionContext::getCurrentRef();
    if (!ctx || ocl_context.empty())
        return AV_HWDEVICE_TYPE_NONE;
#ifdef HAVE_VA_INTEL
    VADisplay vadisplay_ocl = ocl_context.getContext().getOpenCLContextProperty(CL_CONTEXT_VA_API_DISPLAY_INTEL);
    VADisplay vadisplay_ctx = hw_get_va_display(ctx);
    if (vadisplay_ocl && vadisplay_ocl == vadisplay_ctx)
        return AV_HWDEVICE_TYPE_VAAPI;
    else
       CV_Error_(cv::Error::StsBadArg, ("Can't interop with current OpenCL context - VA display mismatch: %p (hwcontext) vs %p (ocl_context).\ndid you call initializeContextFromVA before using VideoCapture/VideoWriter?", vadisplay_ctx, vadisplay_ocl));
#endif
#ifdef HAVE_D3D11
    ID3D11Device* d3d11device_ocl = (ID3D11Device*)ocl_context.getContext().getOpenCLContextProperty(CL_CONTEXT_D3D11_DEVICE_KHR);
    AVD3D11VADeviceContext* d3d11_device_ctx = hw_get_d3d11_device_ctx(ctx);
    if (d3d11_device_ctx && d3d11device_ocl && d3d11_device_ctx->device == d3d11device_ocl)
        return AV_HWDEVICE_TYPE_D3D11VA;
#endif
    return AV_HWDEVICE_TYPE_NONE;
}

static
void hw_init_opencl(AVBufferRef* ctx) {
    if (!ctx)
        return;
    AVHWDeviceContext* hw_device_ctx = (AVHWDeviceContext*)ctx->data;
    if (!hw_device_ctx)
        return;
    ocl::OpenCLExecutionContext& ocl_context = ocl::OpenCLExecutionContext::getCurrent();
#ifdef HAVE_VA_INTEL
    cv::va_intel::VAAPIInterop* interop = ocl_context.getContext().getUserContext<cv::va_intel::VAAPIInterop>().get();
    //only initialize the context automatically if it isn't already
    if(!interop) {
        VADisplay va_display = hw_get_va_display(hw_device_ctx);
        if (va_display) {
            va_intel::ocl::initializeContextFromVA(va_display);
        }
    } else {
        CV_LOG_DEBUG(NULL, "OpenCL/VA_INTEL: CL/VA interop already initialized. ")
    }
#endif
#ifdef HAVE_D3D11
    AVD3D11VADeviceContext* d3d11_device_ctx = hw_get_d3d11_device_ctx(hw_device_ctx);
    if (d3d11_device_ctx) {
        directx::ocl::initializeContextFromD3D11Device(d3d11_device_ctx->device);
    }
#endif
    if (hw_check_opencl_context(hw_device_ctx) != AV_HWDEVICE_TYPE_NONE) {
        // Attach AVHWDeviceContext to OpenCL context
        ocl_context.getContext().setUserContext(std::make_shared<OpenCL_FFMPEG_Context>(ctx));
    }
}

static
AVBufferRef* hw_create_context_from_opencl(ocl::OpenCLExecutionContext& ocl_context, AVHWDeviceType hw_type) {
    if (ocl_context.empty())
        return NULL;
    auto ocl_ffmpeg_context = ocl_context.getContext().getUserContext<OpenCL_FFMPEG_Context>();
    if (!ocl_ffmpeg_context)
        return NULL;
    AVBufferRef* ctx = ocl_ffmpeg_context->GetAVHWDevice();
    if (hw_type != ((AVHWDeviceContext*)ctx->data)->type) {
        ctx = hw_create_derived_context(hw_type, ctx);
    }
    else {
        ctx = av_buffer_ref(ctx);
    }
    if (ctx)
        CV_LOG_INFO(NULL, "FFMPEG: Using " << av_hwdevice_get_type_name(hw_type) << " video acceleration context attached to OpenCL context");
    return ctx;
}

#endif // HAVE_OPENCL

static
AVBufferRef* hw_create_device(AVHWDeviceType hw_type, int hw_device, const std::string& device_subname, bool use_opencl) {
    AVBufferRef* hw_device_ctx = NULL;
    if (AV_HWDEVICE_TYPE_NONE == hw_type)
        return NULL;

#ifdef HAVE_OPENCL
    // Check if OpenCL context has AVHWDeviceContext attached to it
    ocl::OpenCLExecutionContext& ocl_context = ocl::OpenCLExecutionContext::getCurrentRef();
    try {
        hw_device_ctx = hw_create_context_from_opencl(ocl_context, hw_type);
        if (hw_device_ctx) {
            if (hw_device >= 0)
                CV_LOG_ERROR(NULL, "VIDEOIO/FFMPEG: ignoring property HW_DEVICE as device context already created and attached to OpenCL context");
            return hw_device_ctx;
        }
    }
    catch (...) {
        CV_LOG_INFO(NULL, "FFMPEG: Exception creating Video Acceleration context using current OpenCL context");
    }
#endif

    // Create new media context. In QSV case, first create 'child' context.
    std::vector<AVHWDeviceType> child_types = { hw_type };
    if (hw_type == AV_HWDEVICE_TYPE_QSV) {
#ifdef _WIN32
        child_types = { AV_HWDEVICE_TYPE_D3D11VA, AV_HWDEVICE_TYPE_DXVA2 };
#else
        child_types = { AV_HWDEVICE_TYPE_VAAPI };
#endif
    }
    for (AVHWDeviceType child_type : child_types) {
        char device[128] = "";
        char* pdevice = NULL;
        if (hw_device >= 0 && hw_device < 100000) {
            if (child_type == AV_HWDEVICE_TYPE_VAAPI) {
                snprintf(device, sizeof(device), "/dev/dri/renderD%d", 128 + hw_device);
            }
            else {
                snprintf(device, sizeof(device), "%d", hw_device);
            }
            pdevice = device;
        }
        const char* hw_child_name = av_hwdevice_get_type_name(child_type);
        const char* device_name = pdevice ? pdevice : "'default'";
        int err = av_hwdevice_ctx_create(&hw_device_ctx, child_type, pdevice, NULL, 0);
        if (hw_device_ctx && err >= 0)
        {
            if (!hw_check_device(hw_device_ctx, hw_type, device_subname)) {
                av_buffer_unref(&hw_device_ctx);
                continue;
            }
            CV_LOG_INFO(NULL, "FFMPEG: Created video acceleration context (av_hwdevice_ctx_create) for " << hw_child_name << " on device " << device_name);
#ifdef HAVE_OPENCL
            // if OpenCL context not created yet or property HW_ACCELERATION_USE_OPENCL set, create OpenCL context with binding to video acceleration context
            if (ocl::haveOpenCL()) {
                if (ocl_context.empty() || use_opencl) {
                    try {
                        hw_init_opencl(hw_device_ctx);
                        ocl_context = ocl::OpenCLExecutionContext::getCurrentRef();
                        if (!ocl_context.empty()) {
                            CV_LOG_INFO(NULL, "FFMPEG: Created OpenCL context with " << hw_child_name <<
                                " video acceleration on OpenCL device: " << ocl_context.getDevice().name());
                        }
                    } catch (...) {
                        CV_LOG_INFO(NULL, "FFMPEG: Exception creating OpenCL context with " << hw_child_name << " video acceleration");
                    }
                }
                else {
                    CV_LOG_INFO(NULL, "FFMPEG: Can't bind " << hw_child_name << " video acceleration context to already created OpenCL context");
                }
            }
#else
            CV_UNUSED(use_opencl);
#endif
            if (hw_type != child_type) {
                AVBufferRef* derived_ctx = hw_create_derived_context(hw_type, hw_device_ctx);
                av_buffer_unref(&hw_device_ctx);
                return derived_ctx;
            } else {
                return hw_device_ctx;
            }
        }
        else
        {
            const char* hw_name = hw_child_name;
            CV_LOG_INFO(NULL, "FFMPEG: Failed to create " << hw_name << " video acceleration (av_hwdevice_ctx_create) on device " << device_name);
        }
    }
    return NULL;
}

static
AVBufferRef* hw_create_frames(struct AVCodecContext* codec_ctx, AVBufferRef *hw_device_ctx, int width, int height, AVPixelFormat hw_format)
{
    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)hw_device_ctx->data;
    AVBufferRef* child_ctx = hw_device_ctx;
    // In QSV case we first allocate child D3D11/VAAPI frames (except DXVA2 as no OpenCL interop), then derive to parent QSV frames
    if (AV_HWDEVICE_TYPE_QSV == device_ctx->type) {
        AVBufferRef *ctx = (AVBufferRef *) device_ctx->user_opaque; // child context stored during creation of derived context
        if (ctx && AV_HWDEVICE_TYPE_DXVA2 != ((AVHWDeviceContext *) ctx->data)->type) {
            child_ctx = ctx;
        }
    }
    AVBufferRef *hw_frames_ref = nullptr;
    if (codec_ctx)
    {
        int res = avcodec_get_hw_frames_parameters(codec_ctx, child_ctx, hw_format, &hw_frames_ref);
        if (res < 0)
        {
            CV_LOG_DEBUG(NULL, "FFMPEG: avcodec_get_hw_frames_parameters() call failed: " << res)
        }
    }
    if (!hw_frames_ref)
    {
        hw_frames_ref = av_hwframe_ctx_alloc(child_ctx);
    }
    if (!hw_frames_ref)
    {
        CV_LOG_INFO(NULL, "FFMPEG: Failed to create HW frame context (av_hwframe_ctx_alloc)");
        return NULL;
    }
    AVHWFramesContext *frames_ctx = (AVHWFramesContext *)(hw_frames_ref->data);
    frames_ctx->width = width;
    frames_ctx->height = height;
    if (frames_ctx->format == AV_PIX_FMT_NONE) {
        if (child_ctx == hw_device_ctx) {
            frames_ctx->format = hw_format;
        }
        else {
            AVHWFramesConstraints* constraints = av_hwdevice_get_hwframe_constraints(child_ctx, NULL);
            if (constraints) {
                frames_ctx->format = constraints->valid_hw_formats[0];
                av_hwframe_constraints_free(&constraints);
            }
        }
    }
    if (frames_ctx->sw_format == AV_PIX_FMT_NONE)
        frames_ctx->sw_format = HW_DEFAULT_SW_FORMAT;
    if (frames_ctx->initial_pool_size == 0)
        frames_ctx->initial_pool_size = HW_DEFAULT_POOL_SIZE;

#ifdef HAVE_D3D11
    if (frames_ctx->device_ctx && AV_HWDEVICE_TYPE_D3D11VA == frames_ctx->device_ctx->type) {
        // BindFlags
        AVD3D11VAFramesContext* frames_hwctx = (AVD3D11VAFramesContext*)frames_ctx->hwctx;
        frames_hwctx->BindFlags |= D3D11_BIND_DECODER | D3D11_BIND_VIDEO_ENCODER;
        // See function hw_get_d3d11_single_texture(), it allocates additional ID3D11Texture2D texture and
        // attaches it as 'user_opaque' field. We have to set free() callback before av_hwframe_ctx_init() call.
        struct D3D11SingleTexture {
            static void free(struct AVHWFramesContext* ctx) {
                ID3D11Texture2D* singleTexture = (ID3D11Texture2D*)ctx->user_opaque;
                if (ctx->user_opaque)
                    singleTexture->Release();
            }
        };
        frames_ctx->free = D3D11SingleTexture::free;
    }
#endif

    int res = av_hwframe_ctx_init(hw_frames_ref);
    if (res < 0)
    {
        CV_LOG_INFO(NULL, "FFMPEG: Failed to initialize HW frame context (av_hwframe_ctx_init): " << res);
        av_buffer_unref(&hw_frames_ref);
        return NULL;
    }

    if (child_ctx != hw_device_ctx) {
        AVBufferRef* derived_frame_ctx = NULL;
        int flags = AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_WRITE;
        res = av_hwframe_ctx_create_derived(&derived_frame_ctx, hw_format, hw_device_ctx, hw_frames_ref, flags);
        av_buffer_unref(&hw_frames_ref);
        if (res < 0)
        {
            CV_LOG_INFO(NULL, "FFMPEG: Failed to create derived HW frame context (av_hwframe_ctx_create_derived): " << res);
            return NULL;
        }
        else {
            ((AVHWFramesContext*)derived_frame_ctx->data)->user_opaque = frames_ctx;
            return derived_frame_ctx;
        }
    }
    else {
        return hw_frames_ref;
    }
}

static
bool hw_check_codec(AVCodec* codec, AVHWDeviceType hw_type, const char *disabled_codecs)
{
    CV_Assert(disabled_codecs);
    std::string hw_name = std::string(".") + av_hwdevice_get_type_name(hw_type);
    std::stringstream s_stream(disabled_codecs);
    while (s_stream.good()) {
        std::string name;
        getline(s_stream, name, ',');
        if (name == codec->name || name == hw_name || name == codec->name + hw_name || name == "hw") {
            CV_LOG_INFO(NULL, "FFMPEG: skipping codec " << codec->name << hw_name);
            return false;
        }
    }
    return true;
}

static
AVCodec *hw_find_codec(AVCodecID id, AVHWDeviceType hw_type, int (*check_category)(const AVCodec *), const char *disabled_codecs, AVPixelFormat *hw_pix_fmt) {
    AVCodec *c = 0;
    void *opaque = 0;

    while (NULL != (c = (AVCodec*)av_codec_iterate(&opaque)))
    {
        if (!check_category(c))
            continue;
        if (c->id != id)
            continue;
        if (c->capabilities & AV_CODEC_CAP_EXPERIMENTAL)
            continue;
        if (hw_type != AV_HWDEVICE_TYPE_NONE) {
            AVPixelFormat hw_native_fmt = AV_PIX_FMT_NONE;
#if LIBAVUTIL_BUILD < AV_VERSION_INT(56, 51, 100) // VAAPI encoders support avcodec_get_hw_config() starting ffmpeg 4.3
            if (hw_type == AV_HWDEVICE_TYPE_VAAPI)
                hw_native_fmt = AV_PIX_FMT_VAAPI_VLD;
#endif
            if (hw_type == AV_HWDEVICE_TYPE_CUDA) // CUDA encoders don't support avcodec_get_hw_config()
                hw_native_fmt = AV_PIX_FMT_CUDA;
            if (av_codec_is_encoder(c) && hw_native_fmt != AV_PIX_FMT_NONE && c->pix_fmts) {
                for (int i = 0; c->pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
                    if (c->pix_fmts[i] == hw_native_fmt) {
                        *hw_pix_fmt = hw_native_fmt;
                        if (hw_check_codec(c, hw_type, disabled_codecs))
                            return c;
                    }
                }
            }
            for (int i = 0;; i++) {
                const AVCodecHWConfig *hw_config = avcodec_get_hw_config(c, i);
                if (!hw_config)
                    break;
                if (hw_config->device_type == hw_type) {
                    *hw_pix_fmt = hw_config->pix_fmt;
                    if (hw_check_codec(c, hw_type, disabled_codecs))
                        return c;
                }
            }
        } else {
            return c;
        }
    }

    return NULL;
}

// Callback to select hardware pixel format (not software format) and allocate frame pool (hw_frames_ctx)
static
AVPixelFormat hw_get_format_callback(struct AVCodecContext *ctx, const enum AVPixelFormat * fmt) {
    if (!ctx->hw_device_ctx)
        return fmt[0];
    AVHWDeviceType hw_type = ((AVHWDeviceContext*)ctx->hw_device_ctx->data)->type;
    for (int j = 0;; j++) {
        const AVCodecHWConfig *hw_config = avcodec_get_hw_config(ctx->codec, j);
        if (!hw_config)
            break;
        if (hw_config->device_type == hw_type) {
            for (int i = 0; fmt[i] != AV_PIX_FMT_NONE; i++) {
                if (fmt[i] == hw_config->pix_fmt) {
                    if (hw_config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) {
                        ctx->sw_pix_fmt = HW_DEFAULT_SW_FORMAT;
                        ctx->hw_frames_ctx = hw_create_frames(ctx, ctx->hw_device_ctx, ctx->width, ctx->height, fmt[i]);
                        if (ctx->hw_frames_ctx) {
                            //ctx->sw_pix_fmt = ((AVHWFramesContext *)(ctx->hw_frames_ctx->data))->sw_format;
                            return fmt[i];
                        }
                    }
                }
            }
        }
    }
    CV_LOG_DEBUG(NULL, "FFMPEG: Can't select HW format in 'get_format()' callback, use default");
    return fmt[0];
}

// GPU color conversion NV12->BGRA via OpenCL extensions
static bool
hw_copy_frame_to_umat(AVBufferRef* ctx, AVFrame* hw_frame, cv::OutputArray output) {
    CV_UNUSED(hw_frame);
    CV_UNUSED(output);
    if (!ctx)
        return false;

#ifdef HAVE_OPENCL
    try {
        // check that current OpenCL context initilized with binding to same VAAPI/D3D11 context
        AVHWDeviceContext *hw_device_ctx = (AVHWDeviceContext *) ctx->data;
        AVHWDeviceType child_type = hw_check_opencl_context(hw_device_ctx);
        if (child_type == AV_HWDEVICE_TYPE_NONE)
            return false;

#ifdef HAVE_VA_INTEL
        if (child_type == AV_HWDEVICE_TYPE_VAAPI) {
            VADisplay va_display = hw_get_va_display(hw_device_ctx);
            VASurfaceID va_surface = hw_get_va_surface(hw_frame);
            if (va_display && va_surface != VA_INVALID_SURFACE) {
                va_intel::convertFromVASurface(va_display, va_surface, {hw_frame->width, hw_frame->height}, output);
                return true;
            }
        }
#endif

#ifdef HAVE_D3D11
        if (child_type == AV_HWDEVICE_TYPE_D3D11VA) {
            AVD3D11VADeviceContext* d3d11_device_ctx = hw_get_d3d11_device_ctx(hw_device_ctx);
            int subresource = 0;
            ID3D11Texture2D* texture = hw_get_d3d11_texture(hw_frame, &subresource);
            ID3D11Texture2D* singleTexture = hw_get_d3d11_single_texture(hw_frame, d3d11_device_ctx, texture);
            if (texture && singleTexture) {
                // Copy D3D11 sub-texture to D3D11 single texture
                d3d11_device_ctx->device_context->CopySubresourceRegion(singleTexture, 0, 0, 0, 0, texture, subresource, NULL);
                // Copy D3D11 single texture to cv::UMat
                directx::convertFromD3D11Texture2D(singleTexture, output);
                return true;
            }
        }
#endif
    }
    catch (...)
    {
        return false;
    }
#endif // HAVE_OPENCL

    return false;
}

// GPU color conversion BGRA->NV12 via OpenCL extensions
static bool
hw_copy_umat_to_frame(AVBufferRef* ctx, cv::InputArray input, AVFrame* hw_frame) {
    CV_UNUSED(input);
    CV_UNUSED(hw_frame);
    if (!ctx)
        return false;

#ifdef HAVE_OPENCL
    try {
        // check that current OpenCL context initilized with binding to same VAAPI/D3D11 context
        AVHWDeviceContext *hw_device_ctx = (AVHWDeviceContext *) ctx->data;
        AVHWDeviceType child_type = hw_check_opencl_context(hw_device_ctx);
        if (child_type == AV_HWDEVICE_TYPE_NONE)
            return false;

#ifdef HAVE_VA_INTEL
        if (child_type == AV_HWDEVICE_TYPE_VAAPI) {
            VADisplay va_display = hw_get_va_display(hw_device_ctx);
            VASurfaceID va_surface = hw_get_va_surface(hw_frame);
            if (va_display != NULL && va_surface != VA_INVALID_SURFACE) {
                va_intel::convertToVASurface(va_display, input, va_surface, {hw_frame->width, hw_frame->height});
                return true;
            }
        }
#endif

#ifdef HAVE_D3D11
        if (child_type == AV_HWDEVICE_TYPE_D3D11VA) {
            AVD3D11VADeviceContext* d3d11_device_ctx = hw_get_d3d11_device_ctx(hw_device_ctx);
            int subresource = 0;
            ID3D11Texture2D* texture = hw_get_d3d11_texture(hw_frame, &subresource);
            ID3D11Texture2D* singleTexture = hw_get_d3d11_single_texture(hw_frame, d3d11_device_ctx, texture);
            if (texture && singleTexture) {
                // Copy cv::UMat to D3D11 single texture
                directx::convertToD3D11Texture2D(input, singleTexture);
                // Copy D3D11 single texture to D3D11 sub-texture
                d3d11_device_ctx->device_context->CopySubresourceRegion(texture, subresource, 0, 0, 0, singleTexture, 0, NULL);
                return true;
            }
        }
#endif
    }
    catch (...)
    {
        return false;
    }
#endif // HAVE_OPENCL

    return false;
}

static
VideoAccelerationType hw_type_to_va_type(AVHWDeviceType hw_type) {
    struct HWTypeFFMPEG {
        AVHWDeviceType hw_type;
        VideoAccelerationType va_type;
    } known_hw_types[] = {
            { AV_HWDEVICE_TYPE_D3D11VA, VIDEO_ACCELERATION_D3D11 },
            { AV_HWDEVICE_TYPE_VAAPI, VIDEO_ACCELERATION_VAAPI },
            { AV_HWDEVICE_TYPE_QSV, VIDEO_ACCELERATION_MFX },
            { AV_HWDEVICE_TYPE_CUDA, (VideoAccelerationType)(1 << 11) },
    };
    for (const HWTypeFFMPEG& hw : known_hw_types) {
        if (hw_type == hw.hw_type)
            return hw.va_type;
    }
    return VIDEO_ACCELERATION_NONE;
}

class HWAccelIterator {
public:
    HWAccelIterator(VideoAccelerationType va_type, bool isEncoder, AVDictionary *dict)
        : hw_type_(AV_HWDEVICE_TYPE_NONE)
    {
        std::string accel_list;
        if (va_type != VIDEO_ACCELERATION_NONE)
        {
            updateAccelList_(accel_list, va_type, isEncoder, dict);
        }
        if (va_type == VIDEO_ACCELERATION_ANY)
        {
            if (!accel_list.empty())
                accel_list += ",";  // add no-acceleration case to the end of the list
        }
        CV_LOG_DEBUG(NULL, "FFMPEG: allowed acceleration types (" << getVideoAccelerationName(va_type) << "): '" << accel_list << "'");

        if (accel_list.empty() && va_type != VIDEO_ACCELERATION_NONE && va_type != VIDEO_ACCELERATION_ANY)
        {
            // broke stream
            std::string tmp;
            s_stream_ >> tmp;
        }
        else
        {
            s_stream_ = std::istringstream(accel_list);
        }

        if (va_type != VIDEO_ACCELERATION_NONE)
        {
            disabled_codecs_ = isEncoder
                    ? getEncoderDisabledCodecs(dict)
                    : getDecoderDisabledCodecs(dict);
            CV_LOG_DEBUG(NULL, "FFMPEG: disabled codecs: '" << disabled_codecs_ << "'");
        }
    }
    bool good() const
    {
        return s_stream_.good();
    }
    void parse_next()
    {
        getline(s_stream_, hw_type_device_string_, ',');
        size_t index = hw_type_device_string_.find('.');
        if (index != std::string::npos) {
            device_subname_ = hw_type_device_string_.substr(index + 1);
            hw_type_string_ = hw_type_device_string_.substr(0, index);
        } else {
            device_subname_.clear();
            hw_type_string_ = hw_type_device_string_;
        }
        hw_type_ = av_hwdevice_find_type_by_name(hw_type_string_.c_str());
    }
    const std::string& hw_type_device_string() const { return hw_type_device_string_; }
    const std::string& hw_type_string() const { return hw_type_string_; }
    AVHWDeviceType hw_type() const { return hw_type_; }
    const std::string& device_subname() const { return device_subname_; }
    const std::string& disabled_codecs() const { return disabled_codecs_; }
private:
    bool updateAccelList_(std::string& accel_list, VideoAccelerationType va_type, bool isEncoder, AVDictionary *dict)
    {
        std::string new_accels = isEncoder
                ? getEncoderConfiguration(va_type, dict)
                : getDecoderConfiguration(va_type, dict);
        if (new_accels.empty())
            return false;
        if (accel_list.empty())
            accel_list = new_accels;
        else
            accel_list = accel_list + "," + new_accels;
        return true;
    }
    std::istringstream s_stream_;
    std::string hw_type_device_string_;
    std::string hw_type_string_;
    AVHWDeviceType hw_type_;
    std::string device_subname_;

    std::string disabled_codecs_;
};
