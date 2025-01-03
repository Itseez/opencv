OpenCV environment variables reference {#tutorial_env_reference}
======================================

@prev_tutorial{tutorial_config_reference}
@next_tutorial{tutorial_linux_install}

@tableofcontents

## Introduction

OpenCV can change its behavior depending on the runtime environment:
- enable extra debugging output or performance tracing
- modify default locations and search paths
- tune some algorithms or general behavior
- enable or disable workarounds, safety features and optimizations

**Notes:**
- ⭐ marks most popular variables
- variables with names like this `VAR_${NAME}` describes family of variables, where `${NAME}` should be changed to one of predefined values, e.g. `VAR_TBB`, `VAR_OPENMP`, ...

### Setting environment variable in Windows
In terminal or cmd-file (bat-file):
```.bat
set MY_ENV_VARIABLE=true
C:\my_app.exe
```
In GUI:
- Go to "Settings -> System -> About"
- Click on "Advanced system settings" in the right part
- In new window click on the "Environment variables" button
- Add an entry to the "User variables" list

### Setting environment variable in Linux

In terminal or shell script:
```.sh
export MY_ENV_VARIABLE=true
./my_app
```
or as a single command:
```.sh
MY_ENV_VARIABLE=true ./my_app
```

### Setting environment variable in Python

```.py
import os
os.environ["MY_ENV_VARIABLE"] = "True" # value must be a string
import cv2 # variables set after this may not have effect
```

@note This method may not work on all operating systems and/or Python distributions. For example, it works on Ubuntu Linux with system Python interpreter, but doesn't work on Windows 10 with the official Python package. It depends on the ability of a process to change its own environment (OpenCV uses `getenv` from C++ runtime to read variables).

@note See also:
- https://docs.python.org/3.12/library/os.html#os.environ
- https://stackoverflow.com/questions/69199708/setenvironmentvariable-does-not-seem-to-set-values-that-can-be-retrieved-by-ge


## Types

- _bool_ - `1`, `True`, `true`, `TRUE` / `0`, `False`, `false`, `FALSE`
- _number_/_size_ - unsigned number, suffixes `MB`, `Mb`, `mb`, `KB`, `Kb`, `kb`
- _string_ - plain string or can have a structure
- _path_ - to file, to directory
- _paths_ - `;`-separated on Windows, `:`-separated on others


## General, core
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_SKIP_CPU_BASELINE_CHECK | bool | false | do not check that current CPU supports all features used by the build (baseline) |
| OPENCV_CPU_DISABLE | `,` or `;`-separated | | disable code branches which use CPU features (dispatched code) |
| OPENCV_SETUP_TERMINATE_HANDLER | bool | true (Windows) | use std::set_terminate to install own termination handler |
| OPENCV_LIBVA_RUNTIME | file path | | libva for VA interoperability utils |
| OPENCV_ENABLE_MEMALIGN | bool | true (except static analysis, memory sanitizer, fuzzying, _WIN32?) | enable aligned memory allocations |
| OPENCV_BUFFER_AREA_ALWAYS_SAFE | bool | false | enable safe mode for multi-buffer allocations (each buffer separately) |
| OPENCV_KMEANS_PARALLEL_GRANULARITY | num | 1000 | tune algorithm parallel work distribution parameter `parallel_for_(..., ..., ..., granularity)` |
| OPENCV_DUMP_ERRORS | bool | true (Debug or Android), false (others) | print extra information on exception (log to Android) |
| OPENCV_DUMP_CONFIG | bool | false | print build configuration to stderr (`getBuildInformation`) |
| OPENCV_PYTHON_DEBUG | bool | false | enable extra warnings in Python bindings |
| OPENCV_TEMP_PATH | path | `/tmp/` (Linux), `/data/local/tmp/` (Android), `GetTempPathA` (Windows) | directory for temporary files |
| OPENCV_DATA_PATH_HINT | paths | | paths for findDataFile |
| OPENCV_DATA_PATH | paths | | paths for findDataFile |
| OPENCV_SAMPLES_DATA_PATH_HINT | paths | | paths for findDataFile |
| OPENCV_SAMPLES_DATA_PATH | paths | | paths for findDataFile |

Links:
- https://github.com/opencv/opencv/wiki/CPU-optimizations-build-options


## Logging
| name | type | default | description |
|------|------|---------|-------------|
| ⭐ OPENCV_LOG_LEVEL | string | | logging level (see accepted values below) |
| OPENCV_LOG_TIMESTAMP | bool | true | logging with timestamps |
| OPENCV_LOG_TIMESTAMP_NS | bool | false | add nsec to logging timestamps |

### Levels
- `0`, `O`, `OFF`, `S`, `SILENT`, `DISABLE`, `DISABLED`
- `F`, `FATAL`
- `E`, `ERROR`
- `W`, `WARNING`, `WARN`, `WARNINGS`
- `I`, `INFO`
- `D`, `DEBUG`
- `V`, `VERBOSE`


## core/parallel_for
| name | type | default | description |
|------|------|---------|-------------|
| ⭐ OPENCV_FOR_THREADS_NUM | num | 0 | set number of threads |
| OPENCV_THREAD_POOL_ACTIVE_WAIT_PAUSE_LIMIT | num | 16 | tune pthreads parallel_for backend |
| OPENCV_THREAD_POOL_ACTIVE_WAIT_WORKER | num | 2000 | tune pthreads parallel_for backend |
| OPENCV_THREAD_POOL_ACTIVE_WAIT_MAIN | num | 10000 | tune pthreads parallel_for backend |
| OPENCV_THREAD_POOL_ACTIVE_WAIT_THREADS_LIMIT | num | 0 | tune pthreads parallel_for backend |
| OPENCV_FOR_OPENMP_DYNAMIC_DISABLE | bool | false | use single OpenMP thread |


## backends
Some modules have multiple available backends, following variables allow choosing specific backend or changing default priorities in which backends will be probed (e.g. when opening a video file).

| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_PARALLEL_BACKEND | string | | choose specific paralel_for backend (one of `TBB`, `ONETBB`, `OPENMP`) |
| OPENCV_PARALLEL_PRIORITY_${NAME} | num | | set backend priority, default is 1000 |
| OPENCV_PARALLEL_PRIORITY_LIST | string, `,`-separated | | list of backends in priority order |
| OPENCV_UI_BACKEND | string | | choose highgui backend for window rendering (one of `GTK`, `GTK3`, `GTK2`, `QT`, `WIN32`) |
| OPENCV_UI_PRIORITY_${NAME} | num | | set highgui backend priority, default is 1000 |
| OPENCV_UI_PRIORITY_LIST | string, `,`-separated | | list of highgui backends in priority order |
| OPENCV_VIDEOIO_PRIORITY_${NAME} | num | | set videoio backend priority, default is 1000 |
| OPENCV_VIDEOIO_PRIORITY_LIST | string, `,`-separated | | list of videoio backends in priority order |


## plugins
Some external dependencies can be detached into a dynamic library, which will be loaded at runtime (plugin). Following variables allow changing default search locations and naming pattern for these plugins.
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_CORE_PLUGIN_PATH | paths | | directories to search for _core_ plugins |
| OPENCV_CORE_PARALLEL_PLUGIN_${NAME} | string, glob | | parallel_for plugin library name (glob), e.g. default for TBB is "opencv_core_parallel_tbb*.so" |
| OPENCV_DNN_PLUGIN_PATH | paths | | directories to search for _dnn_ plugins |
| OPENCV_DNN_PLUGIN_${NAME} | string, glob | | parallel_for plugin library name (glob), e.g. default for TBB is "opencv_core_parallel_tbb*.so" |
| OPENCV_CORE_PLUGIN_PATH | paths | | directories to search for _highgui_ plugins (YES it is CORE) |
| OPENCV_UI_PLUGIN_${NAME} | string, glob | | _highgui_ plugin library name (glob) |
| OPENCV_VIDEOIO_PLUGIN_PATH | paths | | directories to search for _videoio_ plugins |
| OPENCV_VIDEOIO_PLUGIN_${NAME} | string, glob | | _videoio_ plugin library name (glob) |

## OpenCL

**Note:** OpenCL device specification format is `<Platform>:<CPU|GPU|ACCELERATOR|nothing=GPU/CPU>:<deviceName>`, e.g. `AMD:GPU:`

| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_OPENCL_RUNTIME | filepath or `disabled` | | path to OpenCL runtime library (e.g. `OpenCL.dll`, `libOpenCL.so`) |
| ⭐ OPENCV_OPENCL_DEVICE | string or `disabled` | | choose specific OpenCL device. See specification format in the note above. See more details in the Links section. |
| OPENCV_OPENCL_RAISE_ERROR | bool | false | raise exception if something fails during OpenCL kernel preparation and execution (Release builds only) |
| OPENCV_OPENCL_ABORT_ON_BUILD_ERROR | bool | false | abort if OpenCL kernel compilation failed |
| OPENCV_OPENCL_CACHE_ENABLE | bool | true | enable OpenCL kernel cache |
| OPENCV_OPENCL_CACHE_WRITE | bool | true | allow writing to the cache, otherwise cache will be read-only |
| OPENCV_OPENCL_CACHE_LOCK_ENABLE | bool | true | use .lock files to synchronize between multiple applications using the same OpenCL cache (may not work on network drives) |
| OPENCV_OPENCL_CACHE_CLEANUP | bool | true | automatically remove old entries from cache (leftovers from older OpenCL runtimes) |
| OPENCV_OPENCL_VALIDATE_BINARY_PROGRAMS | bool | false | validate loaded binary OpenCL kernels |
| OPENCV_OPENCL_DISABLE_BUFFER_RECT_OPERATIONS | bool | true (Apple), false (others) | enable workaround for non-continuos data downloads |
| OPENCV_OPENCL_BUILD_EXTRA_OPTIONS | string | | pass extra options to OpenCL kernel compilation |
| OPENCV_OPENCL_ENABLE_MEM_USE_HOST_PTR | bool | true | workaround/optimization for buffer allocation |
| OPENCV_OPENCL_ALIGNMENT_MEM_USE_HOST_PTR | num | 4 | parameter for OPENCV_OPENCL_ENABLE_MEM_USE_HOST_PTR |
| OPENCV_OPENCL_DEVICE_MAX_WORK_GROUP_SIZE | num | 0 | allow to decrease maxWorkGroupSize |
| OPENCV_OPENCL_PROGRAM_CACHE | num | 0 | limit number of programs in OpenCL kernel cache |
| OPENCV_OPENCL_RAISE_ERROR_REUSE_ASYNC_KERNEL | bool | false | raise exception if async kernel failed |
| OPENCV_OPENCL_BUFFERPOOL_LIMIT | num | 1 << 27 (Intel device), 0 (others) | limit memory used by buffer bool |
| OPENCV_OPENCL_HOST_PTR_BUFFERPOOL_LIMIT | num | | same as OPENCV_OPENCL_BUFFERPOOL_LIMIT, but for HOST_PTR buffers |
| OPENCV_OPENCL_BUFFER_FORCE_MAPPING | bool | false | force clEnqueueMapBuffer |
| OPENCV_OPENCL_BUFFER_FORCE_COPYING | bool | false | force clEnqueueReadBuffer/clEnqueueWriteBuffer |
| OPENCV_OPENCL_FORCE | bool | false | force running OpenCL kernel even if usual conditions are not met (e.g. dst.isUMat) |
| OPENCV_OPENCL_PERF_CHECK_BYPASS | bool | false | force running OpenCL kernel even if usual performance-related conditions are not met (e.g. image is very small) |

### SVM (Shared Virtual Memory) - disabled by default
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_OPENCL_SVM_DISABLE | bool | false | disable SVM |
| OPENCV_OPENCL_SVM_FORCE_UMAT_USAGE | bool | false | |
| OPENCV_OPENCL_SVM_DISABLE_UMAT_USAGE | bool | false | |
| OPENCV_OPENCL_SVM_CAPABILITIES_MASK | num | | |
| OPENCV_OPENCL_SVM_BUFFERPOOL_LIMIT | num | | same as OPENCV_OPENCL_BUFFERPOOL_LIMIT, but for SVM buffers |

### Links:
- https://github.com/opencv/opencv/wiki/OpenCL-optimizations


## Tracing/Profiling
| name | type | default | description |
|------|------|---------|-------------|
| ⭐ OPENCV_TRACE | bool | false | enable trace |
| OPENCV_TRACE_LOCATION | string | `OpenCVTrace` | trace file name ("${name}-$03d.txt") |
| OPENCV_TRACE_DEPTH_OPENCV | num | 1 | |
| OPENCV_TRACE_MAX_CHILDREN_OPENCV | num | 1000 | |
| OPENCV_TRACE_MAX_CHILDREN | num | 1000 | |
| OPENCV_TRACE_SYNC_OPENCL | bool | false | wait for OpenCL kernels to finish |
| OPENCV_TRACE_ITT_ENABLE | bool | true | |
| OPENCV_TRACE_ITT_PARENT | bool | false | set parentID for ITT task |
| OPENCV_TRACE_ITT_SET_THREAD_NAME | bool | false | set name for OpenCV's threads "OpenCVThread-%03d" |

### Links:
- https://github.com/opencv/opencv/wiki/Profiling-OpenCV-Applications


## Cache
**Note:** Default tmp location is `%TMPDIR%` (Windows); `$XDG_CACHE_HOME`, `$HOME/.cache`, `/var/tmp`, `/tmp` (others)
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_CACHE_SHOW_CLEANUP_MESSAGE | bool | true | show cache cleanup message |
| OPENCV_DOWNLOAD_CACHE_DIR | path | default tmp location | cache directory for downloaded files (subdirectory `downloads`) |
| OPENCV_DNN_IE_GPU_CACHE_DIR | path | default tmp location | cache directory for OpenVINO OpenCL kernels (subdirectory `dnn_ie_cache_${device}`) |
| OPENCV_OPENCL_CACHE_DIR | path | default tmp location | cache directory for OpenCL kernels cache (subdirectory `opencl_cache`) |


## dnn
**Note:** In the table below `dump_base_name` equals to `ocv_dnn_net_%05d_%02d` where first argument is internal network ID and the second - dump level.
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_DNN_BACKEND_DEFAULT | num | 3 (OpenCV) | set default DNN backend, see dnn.hpp for backends enumeration |
| OPENCV_DNN_NETWORK_DUMP | num | 0 | level of information dumps, 0 - no dumps (default file name `${dump_base_name}.dot`) |
| OPENCV_DNN_DISABLE_MEMORY_OPTIMIZATIONS | bool | false |  |
| OPENCV_DNN_CHECK_NAN_INF | bool | false | check for NaNs in layer outputs |
| OPENCV_DNN_CHECK_NAN_INF_DUMP | bool | false | print layer data when NaN check has failed |
| OPENCV_DNN_CHECK_NAN_INF_RAISE_ERROR | bool | false | also raise exception when NaN check has failed |
| OPENCV_DNN_ONNX_USE_LEGACY_NAMES | bool | false | use ONNX node names as-is instead of "onnx_node!${node_name}" |
| OPENCV_DNN_CUSTOM_ONNX_TYPE_INCLUDE_DOMAIN_NAME | bool | true | prepend layer domain to layer types ("domain.type") |
| OPENCV_VULKAN_RUNTIME | file path | | set location of Vulkan runtime library for DNN Vulkan backend |
| OPENCV_DNN_IE_SERIALIZE | bool | false | dump intermediate OpenVINO graph (default file names `${dump_base_name}_ngraph.xml`, `${dump_base_name}_ngraph.bin`) |
| OPENCV_DNN_IE_EXTRA_PLUGIN_PATH | path | | path to extra OpenVINO plugins |
| OPENCV_DNN_IE_VPU_TYPE | string | | Force using specific OpenVINO VPU device type ("Myriad2" or "MyriadX") |
| OPENCV_TEST_DNN_IE_VPU_TYPE | string | | same as OPENCV_DNN_IE_VPU_TYPE, but for tests |
| OPENCV_DNN_INFERENCE_ENGINE_HOLD_PLUGINS | bool | true | always hold one existing OpenVINO instance to avoid crashes on unloading |
| OPENCV_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND | bool | true (Windows), false (other) | another OpenVINO lifetime workaround |
| OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES | bool | false | allow running on CPU devices, allow FP16 on non-Intel device |
| OPENCV_OCL4DNN_CONVOLUTION_IGNORE_INPUT_DIMS_4_CHECK | bool | false | workaround for OpenCL backend, see https://github.com/opencv/opencv/issues/20833 |
| OPENCV_OCL4DNN_WORKAROUND_IDLF | bool | true | another workaround for OpenCL backend |
| OPENCV_OCL4DNN_CONFIG_PATH | path | | path to kernel configuration cache for auto-tuning (must be existing directory), set this variable to enable auto-tuning |
| OPENCV_OCL4DNN_DISABLE_AUTO_TUNING | bool | false | disable auto-tuning |
| OPENCV_OCL4DNN_FORCE_AUTO_TUNING | bool | false | force auto-tuning |
| OPENCV_OCL4DNN_TEST_ALL_KERNELS | num | 0 | test convolution kernels, number of iterations (auto-tuning) |
| OPENCV_OCL4DNN_DUMP_FAILED_RESULT | bool | false | dump extra information on errors (auto-tuning) |
| OPENCV_OCL4DNN_TUNING_RAISE_CHECK_ERROR | bool | false | raise exception on errors (auto-tuning) |


## Tests
| name | type | default | description |
|------|------|---------|-------------|
| ⭐ OPENCV_TEST_DATA_PATH | dir path | | set test data search location (e.g. `/home/user/opencv_extra/testdata`) |
| ⭐ OPENCV_DNN_TEST_DATA_PATH | dir path | `$OPENCV_TEST_DATA_PATH/dnn` | set DNN model search location for tests (used by _dnn_, _gapi_, _objdetect_, _video_ modules)  |
| OPENCV_OPEN_MODEL_ZOO_DATA_PATH | dir path | `$OPENCV_DNN_TEST_DATA_PATH/omz_intel_models` | set OpenVINO models search location for tests (used by _dnn_, _gapi_ modules) |
| INTEL_CVSDK_DIR | | | some _dnn_ tests can search OpenVINO models here too |
| OPENCV_TEST_DEBUG | num | 0 | debug level for tests, same as `--test_debug` (0 - no debug (default), 1 - basic test debug information, >1 - extra debug information) |
| OPENCV_TEST_REQUIRE_DATA | bool | false | same as `--test_require_data` option (fail on missing non-required test data instead of skip) |
| OPENCV_TEST_CHECK_OPTIONAL_DATA | bool | false | assert when optional data is not found |
| OPENCV_IPP_CHECK | bool | false | default value for `--test_ipp_check` and `--perf_ipp_check` |
| OPENCV_PERF_VALIDATION_DIR | dir path | | location of files read/written by `--perf_read_validation_results`/`--perf_write_validation_results` |
| ⭐ OPENCV_PYTEST_FILTER | string (glob) | | test filter for Python tests |

### Links:
* https://github.com/opencv/opencv/wiki/QA_in_OpenCV


## videoio
**Note:** extra FFmpeg options should be pased in form `key;value|key;value|key;value`, for example `hwaccel;cuvid|video_codec;h264_cuvid|vsync;0` or `vcodec;x264|vprofile;high|vlevel;4.0`

| name | type | default | description |
|------|------|---------|-------------|
| ⭐ OPENCV_FFMPEG_CAPTURE_OPTIONS | string (see note) | | extra options for VideoCapture FFmpeg backend |
| ⭐ OPENCV_FFMPEG_WRITER_OPTIONS | string (see note) | | extra options for VideoWriter FFmpeg backend |
| OPENCV_FFMPEG_THREADS | num | | set FFmpeg thread count |
| OPENCV_FFMPEG_DEBUG | bool | false | enable logging messages from FFmpeg |
| OPENCV_FFMPEG_LOGLEVEL | num | | set FFmpeg logging level |
| OPENCV_FFMPEG_DLL_DIR | dir path | | directory with FFmpeg plugin (legacy) |
| OPENCV_FFMPEG_IS_THREAD_SAFE | bool | false | enabling this option will turn off thread safety locks in the FFmpeg backend (use only if you are sure FFmpeg is built with threading support, tested on Linux) |
| OPENCV_FFMPEG_READ_ATTEMPTS | num | 4096 | number of failed `av_read_frame` attempts before failing read procedure |
| OPENCV_FFMPEG_DECODE_ATTEMPTS | num | 64 | number of failed `avcodec_receive_frame` attempts before failing decoding procedure |
| OPENCV_VIDEOIO_GSTREAMER_CALL_DEINIT | bool | false | close GStreamer instance on end |
| OPENCV_VIDEOIO_GSTREAMER_START_MAINLOOP | bool | false | start GStreamer loop in separate thread |
| OPENCV_VIDEOIO_MFX_IMPL | num | | set specific MFX implementation (see MFX docs for enumeration) |
| OPENCV_VIDEOIO_MFX_EXTRA_SURFACE_NUM | num | 1 | add extra surfaces to the surface pool |
| OPENCV_VIDEOIO_MFX_POOL_TIMEOUT | num | 1 | timeout for waiting for free surface from the pool (in seconds) |
| OPENCV_VIDEOIO_MFX_BITRATE_DIVISOR | num | 300 | this option allows to tune encoding bitrate (video quality/size) |
| OPENCV_VIDEOIO_MFX_WRITER_TIMEOUT | num | 1 | timeout for encoding operation (in seconds) |
| OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS | bool | true | allow HW-accelerated transformations (DXVA) in MediaFoundation processing graph (may slow down camera probing process) |
| OPENCV_DSHOW_DEBUG | bool | false | enable verbose logging in the DShow backend |
| OPENCV_DSHOW_SAVEGRAPH_FILENAME | file path | | enable processing graph tump in the DShow backend |
| OPENCV_VIDEOIO_V4L_RANGE_NORMALIZED | bool | false | use (0, 1) range for properties (V4L) |
| OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT | num | 10 | timeout for select call (in seconds) (V4L) |
| OPENCV_VIDEOCAPTURE_DEBUG | bool | false | enable debug messages for VideoCapture |
| OPENCV_VIDEOWRITER_DEBUG | bool | false | enable debug messages for VideoWriter |
| ⭐ OPENCV_VIDEOIO_DEBUG | bool | false | debug messages for both VideoCapture and VideoWriter |

### videoio tests
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_TEST_VIDEOIO_BACKEND_REQUIRE_FFMPEG | bool | false | test app will exit if no FFmpeg backend is available |
| OPENCV_TEST_V4L2_VIVID_DEVICE | file path | | path to VIVID virtual camera device for V4L2 test (e.g. `/dev/video5`) |
| OPENCV_TEST_PERF_CAMERA_LIST | paths | | cameras to use in performance test (waitAny_V4L test) |
| OPENCV_TEST_CAMERA_%d_FPS | num | | fps to set for N-th camera (0-based index) (waitAny_V4L test) |


## highgui

| name | type | default | description |
|------|------|---------|-------------|
| $XDG_RUNTIME_DIR | | | Wayland backend specific - create shared memory-mapped file for interprocess communication (named `opencv-shared-??????`) |
| OPENCV_HIGHGUI_FB_MODE | string | `FB` | Selects output mode for the framebuffer backend (`FB` - regular frambuffer, `EMU` - emulation, perform internal checks but does nothing, `XVFB` - compatible with _xvfb_ virtual frambuffer) |
| OPENCV_HIGHGUI_FB_DEVICE | file path | | Path to frambuffer device to use (will be checked first) |
| FRAMEBUFFER | file path | `/dev/fb0` | Same as OPENCV_HIGHGUI_FB_DEVICE, commonly used variable for the same purpose (will be checked second) |


## imgproc
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_OPENCL_IMGPROC_MORPH_SPECIAL_KERNEL | bool | true (Apple), false (others) | use special OpenCL kernel for small morph kernel (Intel devices) |
| OPENCV_GAUSSIANBLUR_CHECK_BITEXACT_KERNELS | bool | false | validate Gaussian kernels before running (src is CV_16U, bit-exact version) |


## imgcodecs
| name | type | default | description |
|------|------|---------|-------------|
| OPENCV_IMGCODECS_AVIF_MAX_FILE_SIZE | num | 64MB | limit input AVIF size |
| OPENCV_IMGCODECS_WEBP_MAX_FILE_SIZE | num | 64MB | limit input WEBM size |
| OPENCV_IO_MAX_IMAGE_PARAMS | num | 50 | limit maximum allowed number of parameters in imwrite and imencode |
| OPENCV_IO_MAX_IMAGE_WIDTH | num | 1 << 20, limit input image size to avoid large memory allocations | |
| OPENCV_IO_MAX_IMAGE_HEIGHT | num | 1 << 20 | |
| OPENCV_IO_MAX_IMAGE_PIXELS | num | 1 << 30 | |
| OPENCV_IO_ENABLE_JASPER | bool | true (set build option OPENCV_IO_FORCE_JASPER), false (otherwise) | enable Jasper backend |

@note OPENCV_IO_ENABLE_OPENEXR is deprecated because bundled OpenEXR library had been removed.
