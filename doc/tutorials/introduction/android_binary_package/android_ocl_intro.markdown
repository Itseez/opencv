Use OpenCL in Android camera preview based CV application {#tutorial_android_ocl_intro}
=====================================

@prev_tutorial{tutorial_android_dnn_intro}
@next_tutorial{tutorial_macos_install}

|    |    |
| -: | :- |
| Original author | Andrey Pavlenko, Alexander Panov |
| Compatibility   | OpenCV >= 4.9 |

@tableofcontents

This guide was designed to help you in use of [OpenCL &trade;](https://www.khronos.org/opencl/) in Android camera preview based CV application.
Tutorial was written for [Android Studio](http://developer.android.com/tools/studio/index.html) 2022.2.1. It was tested with Ubuntu 22.04.

This tutorial assumes you have the following installed and configured:

-   Android Studio (2022.2.1.+)
-   JDK 17
-   Android SDK
-   Android NDK (25.2.9519653+)
-   download OpenCV source code from [github](git@github.com:opencv/opencv.git) or from [releases](https://opencv.org/releases/) and build by [instruction on wiki](https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build).

It also assumes that you are familiar with Android Java and JNI programming basics.
If you need help with anything of the above, you may refer to our @ref tutorial_android_dev_intro guide.

This tutorial also assumes you have an Android operated device with OpenCL enabled.

The related source code is located within OpenCV samples at
[opencv/samples/android/tutorial-4-opencl](https://github.com/opencv/opencv/tree/4.x/samples/android/tutorial-4-opencl/) directory.

How to build custom OpenCV Android SDK with OpenCL
--------------------------------------------------

1. __Assemble and configure Android OpenCL SDK.__
The JNI part of the sample depends on standard Khornos OpenCL headers, and C++ wrapper for OpenCL and libOpenCL.so.
The standard OpenCL headers may be copied from 3rdparty directory in OpenCV repository or you Linux distribution package.
C++ wrapper is available in [official Khronos reposiotry on Github](https://github.com/KhronosGroup/OpenCL-CLHPP).
Copy the header files to didicated directory in the following way:
@code{.bash}
cd your_path/ && mkdir ANDROID_OPENCL_SDK && mkdir ANDROID_OPENCL_SDK/include && cd ANDROID_OPENCL_SDK/include
cp -r path_to_opencv/opencv/3rdparty/include/opencl/1.2/CL . && cd CL
wget https://github.com/KhronosGroup/OpenCL-CLHPP/raw/main/include/CL/opencl.hpp
wget https://github.com/KhronosGroup/OpenCL-CLHPP/raw/main/include/CL/cl2.hpp
@endcode
libOpenCL.so may be provided with BSP or just downloaded from any OpenCL-cabaple Android device with relevant arhitecture.
@code{.bash}
cd your_path/ANDROID_OPENCL_SDK && mkdir lib && cd lib
adb pull /system/vendor/lib64/libOpenCL.so
@endcode
System verison of libOpenCL.so may have a lot of platform specific dependencies. `-Wl,--allow-shlib-undefined` flag allows
to ignore 3rdparty symbols if they are not used during the build.
The following CMake line allows to link the JNI part against standard OpenCL, but not include the loadLibrary into
application package. System OpenCL API is used in run-time.
@code
target_link_libraries(${target} -lOpenCL)
@endcode


2. __Build custom OpenCV Android SDK with OpenCL.__
OpenCL support (T-API) is disabled in OpenCV builds for Android OS by default.
but it's possible to rebuild locally OpenCV for Android with OpenCL/T-API enabled: use `-DWITH_OPENCL=ON` option for CMake.
You also need to specify the path to the Android OpenCL SDK: use `-DANDROID_OPENCL_SDK=path_to_your_Android_OpenCL_SDK` option for CMake.
If you are building OpenCV using `build_sdk.py` please follow [instruction on wiki](https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build).
Set these CMake parameters in your `.config.py`, e.g. `ndk-18-api-level-21.config.py`:
@code{.py}
ABI("3", "arm64-v8a", None, 21, cmake_vars=dict('WITH_OPENCL': 'ON', 'ANDROID_OPENCL_SDK': 'path_to_your_Android_OpenCL_SDK'))
@endcode
If you are building OpenCV using cmake/ninja, use this bash script (set your NDK_VERSION and your paths instead of examples of paths):
@code{.bash}
cd path_to_opencv && mkdir build && cd build
export NDK_VERSION=25.2.9519653
export ANDROID_SDK=/home/user/Android/Sdk/
export ANDROID_OPENCL_SDK=/path_to_ANDROID_OPENCL_SDK/
export ANDROID_HOME=$ANDROID_SDK
export ANDROID_NDK_HOME=$ANDROID_SDK/ndk/$NDK_VERSION/
cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -DANDROID_STL=c++_shared -DANDROID_NATIVE_API_LEVEL=24
-DANDROID_SDK=$ANDROID_SDK -DANDROID_NDK=$ANDROID_NDK_HOME -DBUILD_JAVA=ON -DANDROID_HOME=$ANDROID_SDK -DBUILD_ANDROID_EXAMPLES=ON
-DINSTALL_ANDROID_EXAMPLES=ON -DANDROID_ABI=arm64-v8a -DWITH_OPENCL=ON -DANDROID_OPENCL_SDK=$ANDROID_OPENCL_SDK ..
@endcode

Preface
-------

Using [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units)
via OpenCL for applications performance enhancements is quite a modern trend now.
Some CV algo-s (e.g. image filtering) run much faster on a GPU than on a CPU.
Recently it has become possible on Android OS.

The most popular CV application scenario for an Android operated device is starting camera in preview mode, applying some CV algo to every frame
and displaying the preview frames modified by that CV algo.

Let's consider how we can use OpenCL in this scenario. In particular let's try two ways: direct calls to OpenCL API and recently introduced OpenCV T-API
(aka [Transparent API](https://docs.google.com/presentation/d/1qoa29N_B-s297-fp0-b3rBirvpzJQp8dCtllLQ4DVCY/present)) - implicit OpenCL accelerations of some OpenCV algo-s.

Application structure
---------------------

Starting Android API level 11 (Android 3.0) [Camera API](http://developer.android.com/reference/android/hardware/Camera.html)
allows use of OpenGL texture as a target for preview frames.
Android API level 21 brings a new [Camera2 API](http://developer.android.com/reference/android/hardware/camera2/package-summary.html)
that provides much more control over the camera settings and usage modes,
it allows several targets for preview frames and OpenGL texture in particular.

Having a preview frame in an OpenGL texture is a good deal for using OpenCL because there is an
[OpenGL-OpenCL Interoperability API (cl_khr_gl_sharing)](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/cl_khr_gl_sharing.html),
allowing sharing OpenGL texture data with OpenCL functions without copying (with some restrictions of course).

Let's create a base for our application that just configures Android camera to send preview frames to OpenGL texture and displays these frames
on display without any processing.

A minimal `Activity` class for that purposes looks like following:

@code{.java}
public class Tutorial4Activity extends Activity {

    private MyGLSurfaceView mView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        mView = new MyGLSurfaceView(this);
        setContentView(mView);
    }

    @Override
    protected void onPause() {
        mView.onPause();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mView.onResume();
    }
}
@endcode

And a minimal `View` class respectively:

@snippet samples/android/tutorial-4-opencl/src/org/opencv/samples/tutorial4/MyGLSurfaceView.java minimal_surface_view

@note we use two renderer classes: one for legacy [Camera](http://developer.android.com/reference/android/hardware/Camera.html) API
and another for modern [Camera2](http://developer.android.com/reference/android/hardware/camera2/package-summary.html).

A minimal `Renderer` class can be implemented in Java (OpenGL ES 2.0 [available](http://developer.android.com/reference/android/opengl/GLES20.html) in Java),
but since we are going to modify the preview texture with OpenCL let's move OpenGL stuff to JNI.
Here is a simple Java wrapper for our JNI stuff:

@snippet samples/android/tutorial-4-opencl/src/org/opencv/samples/tutorial4/NativePart.java native_part

Since `Camera` and `Camera2` APIs differ significantly in camera setup and control, let's create a base class for the two corresponding renderers:

@code{.java}
public abstract class MyGLRendererBase implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {
    protected final String LOGTAG = "MyGLRendererBase";

    protected SurfaceTexture mSTex;
    protected MyGLSurfaceView mView;

    protected boolean mGLInit = false;
    protected boolean mTexUpdate = false;

    MyGLRendererBase(MyGLSurfaceView view) {
        mView = view;
    }

    protected abstract void openCamera();
    protected abstract void closeCamera();
    protected abstract void setCameraPreviewSize(int width, int height);

    public void onResume() {
        Log.i(LOGTAG, "onResume");
    }

    public void onPause() {
        Log.i(LOGTAG, "onPause");
        mGLInit = false;
        mTexUpdate = false;
        closeCamera();
        if(mSTex != null) {
            mSTex.release();
            mSTex = null;
            NativeGLRenderer.closeGL();
        }
    }

    @Override
    public synchronized void onFrameAvailable(SurfaceTexture surfaceTexture) {
        //Log.i(LOGTAG, "onFrameAvailable");
        mTexUpdate = true;
        mView.requestRender();
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        //Log.i(LOGTAG, "onDrawFrame");
        if (!mGLInit)
            return;

        synchronized (this) {
            if (mTexUpdate) {
                mSTex.updateTexImage();
                mTexUpdate = false;
            }
        }
        NativeGLRenderer.drawFrame();
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int surfaceWidth, int surfaceHeight) {
        Log.i(LOGTAG, "onSurfaceChanged("+surfaceWidth+"x"+surfaceHeight+")");
        NativeGLRenderer.changeSize(surfaceWidth, surfaceHeight);
        setCameraPreviewSize(surfaceWidth, surfaceHeight);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        Log.i(LOGTAG, "onSurfaceCreated");
        String strGLVersion = GLES20.glGetString(GLES20.GL_VERSION);
        if (strGLVersion != null)
            Log.i(LOGTAG, "OpenGL ES version: " + strGLVersion);

        int hTex = NativeGLRenderer.initGL();
        mSTex = new SurfaceTexture(hTex);
        mSTex.setOnFrameAvailableListener(this);
        openCamera();
        mGLInit = true;
    }
}
@endcode

As you can see, inheritors for `Camera` and `Camera2` APIs should implement the following abstract methods:
@code{.java}
    protected abstract void openCamera();
    protected abstract void closeCamera();
    protected abstract void setCameraPreviewSize(int width, int height);
@endcode

Let's leave the details of their implementation beyond of this tutorial, please refer the
[source code](https://github.com/opencv/opencv/tree/4.x/samples/android/tutorial-4-opencl/) to see them.

Preview Frames modification
---------------------------

The details OpenGL ES 2.0 initialization are also quite straightforward and noisy to be quoted here,
but the important point here is that the OpeGL texture to be the target for camera preview should be of type `GL_TEXTURE_EXTERNAL_OES`
(not `GL_TEXTURE_2D`), internally it keeps picture data in _YUV_ format.
That makes unable sharing it via CL-GL interop (`cl_khr_gl_sharing`) and accessing its pixel data via C/C++ code.
To overcome this restriction we have to perform an OpenGL rendering from this texture to another regular `GL_TEXTURE_2D` one
using _FrameBuffer Object_ (aka FBO).

### C/C++ code

After that we can read (_copy_) pixel data from C/C++ via `glReadPixels()` and write them back to texture after modification via `glTexSubImage2D()`.

### Direct OpenCL calls

Also that `GL_TEXTURE_2D` texture can be shared with OpenCL without copying, but we have to create OpenCL context with special way for that:

@snippet samples/android/tutorial-4-opencl/jni/CLprocessor.cpp init_opencl

Then the texture can be wrapped by a `cl::ImageGL` object and processed via OpenCL calls:

@snippet samples/android/tutorial-4-opencl/jni/CLprocessor.cpp process_pure_opencl

### OpenCV T-API

But instead of writing OpenCL code by yourselves you may want to use __OpenCV T-API__ that calls OpenCL implicitly.
All that you need is to pass the created OpenCL context to OpenCV (via `cv::ocl::attachContext()`) and somehow wrap OpenGL texture with `cv::UMat`.
Unfortunately `UMat` keeps OpenCL _buffer_ internally, that can't be wrapped over either OpenGL _texture_ or OpenCL _image_ - so we have to copy image data here:

@snippet samples/android/tutorial-4-opencl/jni/CLprocessor.cpp process_tapi

@note We have to make one more image data copy when placing back the modified image to the original OpenGL texture via OpenCL image wrapper.

Performance notes
-----------------

To compare the performance we measured FPS of the same preview frames modification (_Laplacian_) done by C/C++ code (call to `cv::Laplacian` with `cv::Mat`),
by direct OpenCL calls (using OpenCL _images_ for input and output), and by OpenCV _T-API_ (call to `cv::Laplacian` with `cv::UMat`) on _Sony Xperia Z3_ with 720p camera resolution:
* __C/C++ version__ shows __3-4 fps__
* __direct OpenCL calls__ shows __25-27 fps__
* __OpenCV T-API__ shows __11-13 fps__ (due to extra copying from `cl_image` to `cl_buffer` and back)
