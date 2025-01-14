// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef PLUGIN_CAPTURE_API_HPP
#define PLUGIN_CAPTURE_API_HPP

#include <opencv2/core/cvdef.h>
#include <opencv2/core/llapi/llapi.h>

#if !defined(BUILD_PLUGIN)

/// increased for backward-compatible changes, e.g. add new function
/// Caller API <= Plugin API -> plugin is fully compatible
/// Caller API > Plugin API -> plugin is not fully compatible, caller should use extra checks to use plugins with older API
#define CAPTURE_API_VERSION 2

/// increased for incompatible changes, e.g. remove function argument
/// Caller ABI == Plugin ABI -> plugin is compatible
/// Caller ABI > Plugin ABI -> plugin is not compatible, caller should use shim code to use old ABI plugins (caller may know how lower ABI works, so it is possible)
/// Caller ABI < Plugin ABI -> plugin can't be used (plugin should provide interface with lower ABI to handle that)
#define CAPTURE_ABI_VERSION 1

#else // !defined(BUILD_PLUGIN)

#if !defined(CAPTURE_ABI_VERSION) || !defined(CAPTURE_API_VERSION)
#error "Plugin must define CAPTURE_ABI_VERSION and CAPTURE_API_VERSION before including plugin_capture_api.hpp"
#endif

#endif // !defined(BUILD_PLUGIN)


#ifdef __cplusplus
extern "C" {
#endif

typedef CvResult (CV_API_CALL *cv_videoio_capture_retrieve_cb_t)(int stream_idx, unsigned const char* data, int step, int width, int height, int type, void* userdata);

typedef struct CvPluginCapture_t* CvPluginCapture;

struct OpenCV_VideoIO_Capture_Plugin_API_v1_0_api_entries
{
    /** OpenCV capture ID (VideoCaptureAPIs)
    @note API-ENTRY 1, API-Version == 0
     */
    int id;

    /** @brief Open video capture

    @param filename File name or NULL to use camera_index instead
    @param camera_index Camera index (used if filename == NULL)
    @param[out] handle pointer on Capture handle

    @note API-CALL 2, API-Version == 0
     */
    CvResult (CV_API_CALL *Capture_open)(const char* filename, int camera_index, CV_OUT CvPluginCapture* handle);

    /** @brief Release Capture handle

    @param handle Capture handle

    @note API-CALL 3, API-Version == 0
     */
    CvResult (CV_API_CALL *Capture_release)(CvPluginCapture handle);

    /** @brief Get property value

    @param handle Capture handle
    @param prop Property index
    @param[out] val property value

    @note API-CALL 4, API-Version == 0
     */
    CvResult (CV_API_CALL *Capture_getProperty)(CvPluginCapture handle, int prop, CV_OUT double* val);

    /** @brief Set property value

    @param handle Capture handle
    @param prop Property index
    @param val property value

    @note API-CALL 5, API-Version == 0
     */
    CvResult (CV_API_CALL *Capture_setProperty)(CvPluginCapture handle, int prop, double val);

    /** @brief Grab frame

    @param handle Capture handle

    @note API-CALL 6, API-Version == 0
     */
    CvResult (CV_API_CALL *Capture_grab)(CvPluginCapture handle);

    /** @brief Retrieve frame

    @param handle Capture handle
    @param stream_idx stream index to retrieve (BGR/IR/depth data)
    @param callback retrieve callback (synchronous)
    @param userdata callback context data

    @note API-CALL 7, API-Version == 0
     */
    CvResult (CV_API_CALL *Capture_retreive)(CvPluginCapture handle, int stream_idx, cv_videoio_capture_retrieve_cb_t callback, void* userdata);
}; // OpenCV_VideoIO_Capture_Plugin_API_v1_0_api_entries

struct OpenCV_VideoIO_Capture_Plugin_API_v1_1_api_entries
{
    /** @brief Open video capture with parameters

    @param filename File name or NULL to use camera_index instead
    @param camera_index Camera index (used if filename == NULL)
    @param params pointer on 2*n_params array of 'key,value' pairs
    @param n_params number of passed parameters
    @param[out] handle pointer on Capture handle

    @note API-CALL 8, API-Version == 1
     */
    CvResult (CV_API_CALL *Capture_open_with_params)(
        const char* filename, int camera_index,
        int* params, unsigned n_params,
        CV_OUT CvPluginCapture* handle);
}; // OpenCV_VideoIO_Capture_Plugin_API_v1_1_api_entries

struct OpenCV_VideoIO_Capture_Plugin_API_v1_2_api_entries
{
    /** @brief Open video capture from buffer with parameters

    @param opaque A pointer to user data
    @param read A pointer to a function that is called to reads @p size bytes to allocated @p buffer. Returns a number of bytes that were actually read
    @param seek A pointer to a function that is called to move starting position inside the stream buffer.
                @p offset is a number of bytes and @p way is one of the markers SEEK_SET, SEEK_CUR, SEEK_END.
                Function returns an absolute current position in bytes.
    @param params pointer on 2*n_params array of 'key,value' pairs
    @param n_params number of passed parameters
    @param[out] handle pointer on Capture handle

    @note API-CALL 9, API-Version == 2
     */
    CvResult (CV_API_CALL *Capture_open_stream)(
        void* opaque,
        long long(*read)(void* opaque, char* buffer, long long size),
        long long(*seek)(void* opaque, long long offset, int way),
        int* params, unsigned n_params,
        CV_OUT CvPluginCapture* handle);
}; // OpenCV_VideoIO_Capture_Plugin_API_v1_2_api_entries

typedef struct OpenCV_VideoIO_Capture_Plugin_API_v1_0
{
    OpenCV_API_Header api_header;
    struct OpenCV_VideoIO_Capture_Plugin_API_v1_0_api_entries v0;
} OpenCV_VideoIO_Capture_Plugin_API_v1_0;

typedef struct OpenCV_VideoIO_Capture_Plugin_API_v1_1
{
    OpenCV_API_Header api_header;
    struct OpenCV_VideoIO_Capture_Plugin_API_v1_0_api_entries v0;
    struct OpenCV_VideoIO_Capture_Plugin_API_v1_1_api_entries v1;
} OpenCV_VideoIO_Capture_Plugin_API_v1_1;

typedef struct OpenCV_VideoIO_Capture_Plugin_API_v1_2
{
    OpenCV_API_Header api_header;
    struct OpenCV_VideoIO_Capture_Plugin_API_v1_0_api_entries v0;
    struct OpenCV_VideoIO_Capture_Plugin_API_v1_1_api_entries v1;
    struct OpenCV_VideoIO_Capture_Plugin_API_v1_2_api_entries v2;
} OpenCV_VideoIO_Capture_Plugin_API_v1_2;

#if CAPTURE_ABI_VERSION == 1 && CAPTURE_API_VERSION == 2
typedef struct OpenCV_VideoIO_Capture_Plugin_API_v1_2 OpenCV_VideoIO_Capture_Plugin_API;
#elif CAPTURE_ABI_VERSION == 1 && CAPTURE_API_VERSION == 1
typedef struct OpenCV_VideoIO_Capture_Plugin_API_v1_1 OpenCV_VideoIO_Capture_Plugin_API;
#elif CAPTURE_ABI_VERSION == 1 && CAPTURE_API_VERSION == 0
typedef struct OpenCV_VideoIO_Capture_Plugin_API_v1_0 OpenCV_VideoIO_Capture_Plugin_API;
#else
#error "Not supported configuration: check CAPTURE_ABI_VERSION/CAPTURE_API_VERSION"
#endif

#ifdef BUILD_PLUGIN

CV_PLUGIN_EXPORTS
const OpenCV_VideoIO_Capture_Plugin_API* CV_API_CALL opencv_videoio_capture_plugin_init_v1
        (int requested_abi_version, int requested_api_version, void* reserved /*NULL*/) CV_NOEXCEPT;

#else  // BUILD_PLUGIN
typedef const OpenCV_VideoIO_Capture_Plugin_API* (CV_API_CALL *FN_opencv_videoio_capture_plugin_init_t)
        (int requested_abi_version, int requested_api_version, void* reserved /*NULL*/);
#endif  // BUILD_PLUGIN


#ifdef __cplusplus
}
#endif

#endif // PLUGIN_CAPTURE_API_HPP
