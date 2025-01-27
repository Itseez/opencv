Android Development with OpenCV {#tutorial_dev_with_OCV_on_Android}
===============================

@prev_tutorial{tutorial_android_dev_intro}
@next_tutorial{tutorial_android_dnn_intro}

|    |    |
| -: | :- |
| Original authors | Alexander Panov, Rostislav Vasilikhin |
| Compatibility | OpenCV >= 4.9.0 |

@tableofcontents

This tutorial has been created to help you use OpenCV library within your Android project.

This guide was checked on Ubuntu but contains no platform-dependent parts, therefore should be compatible with any OS supported by Android Studio and OpenCV4Android SDK.

This tutorial assumes you have the following installed and configured:

-   Android Studio
-   JDK
-   Android SDK and NDK
-   Optional: OpenCV for Android SDK from official [release page on Github](https://github.com/opencv/opencv/releases)
    or [SourceForge](https://sourceforge.net/projects/opencvlibrary/). Advanced: as alternative the SDK may be
    built from source code by [instruction on wiki](https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build).

If you need help with anything of the above, you may refer to our @ref tutorial_android_dev_intro guide.

If you encounter any error after thoroughly following these steps, feel free to contact us via OpenCV [forum](https://forum.opencv.org). We'll do our best to help you out.


Hello OpenCV sample with SDK
----------------------------

In this section we're gonna create a simple app that does nothing but OpenCV loading. In next section we'll extend it to support camera.

In addition to this instruction you can use some video guide, for example [this one](https://www.youtube.com/watch?v=bR7lL886-uc&ab_channel=ProgrammingHut)

1. Open Android Studio and create empty project by choosing ***Empty Views Activity***

    ![](images/create_empty_project.png)

2. Setup the project:
    - Choose ***Java*** language
    - Choose ***Groovy DSL*** build configuration language
    - Choose ***Minumum SDK*** with the version number not less than was used during OpenCV 4 Android build
        - If you don't know it, you can find it in file `OpenCV-android-sdk/sdk/build.gradle` at `android -> defaultConfig -> minSdkVersion`

    ![](images/setup_project.png)


3. Click ***File -> New -> Import module...*** and select OpenCV SDK path

    ![](images/sdk_path.png)

4. Set module name as `OpenCV` and press `Finish`

    ![](images/module_name.png)

5. OpenCV also provides experiemental Kotlin support. Please add Android Kotlin plugin to `MyApplication/OpenCV/build.gradle` file:
    @code{.gradle}
    plugins {
        id 'org.jetbrains.kotlin.android' version '1.7.10' #version may differ for your setup
    }
    @endcode
    Like this:
    ![](images/gradle_ocv_fix.png)
    If you don't do this, you may get an error:
    @code
    Task failed with an exception.
    -----------
    * Where:
    Build file '/home/alexander/AndroidStudioProjects/MyApplication/opencv/build.gradle' line: 4

    * What went wrong:
    A problem occurred evaluating project ':opencv'.
    > Plugin with id 'kotlin-android' not found.
    @endcode
    The fix was found [here](https://stackoverflow.com/questions/73225714/import-opencv-sdk-to-android-studio-chipmunk)

6. OpenCV project uses `buildConfig` feature. Please enable it in
   `MyApplication/OpenCV/build.gradle` file to `android` block:
    @code{.gradle}
    buildFeatures{
        buildConfig true
    }

    @endcode
    Like this:
    ![](images/module_gradle_fix.png)
    If you don't do this, you may get an error:
    @code
    JavaCameraView.java:15: error: cannot find symbol import org.opencv.BuildConfig; ^ symbol: class BuildConfig location: package org.opencv
    @endcode
    The fix was found [here](https://stackoverflow.com/questions/76374886/error-cannot-find-symbol-import-org-opencv-buildconfig-android-studio) and [here](https://forum.opencv.org/t/task-compiledebugjavawithjavac-failed/13667/4)

7. Add the module to the project:
    - Click ***File -> Project structure... -> Dependencies -> All modules -> + (Add Dependency button) -> Module dependency***

    ![](images/add_module_1.png)

    - Choose `app`

    ![](images/add_module_2.png)

    - Select `OpenCV`

    ![](images/add_module_3.png)

8. Before using any OpenCV function you have to load the library first. If you application includes other OpenCV-dependent native libraries you should load them ***after*** OpenCV initialization.
    Add the folowing code to load the library at app start:
    @snippet samples/android/tutorial-1-camerapreview/src/org/opencv/samples/tutorial1/Tutorial1Activity.java ocv_loader_init
    Like this:
    ![](images/sample_code.png)

9. Choose a device to check the sample on and run the code by pressing `run` button

    ![](images/run_app.png)

Hello OpenCV sample with Maven Central
--------------------------------------

Since OpenCV 4.9.0 OpenCV for Android package is available with Maven Central and may be installed
automatically as Gradle dependency. In this section we're gonna create a simple app that does nothing
but OpenCV loading with Maven Central.

1. Open Android Studio and create empty project by choosing ***Empty Views Activity***

    ![](images/create_empty_project.png)

2. Setup the project:
    - Choose ***Java*** language
    - Choose ***Groovy DSL*** build configuration language
    - Choose ***Minumum SDK*** with the version number not less than OpenCV supports. For 4.9.0 minimal SDK version is 21.

    ![](images/setup_project.png)

3. Edit `build.gradle` and add OpenCV library to Dependencies list like this:
    @code{.gradle}
    dependencies {
        implementation 'org.opencv:opencv:4.9.0'
    }
    @endcode
   `4.9.0` may be replaced by any version available as [official release](https://central.sonatype.com/artifact/org.opencv/opencv).

4. Before using any OpenCV function you have to load the library first. If you application includes other
   OpenCV-dependent native libraries you should load them ***after*** OpenCV initialization. Add the folowing
   code to load the library at app start:
    @snippet samples/android/tutorial-1-camerapreview/src/org/opencv/samples/tutorial1/Tutorial1Activity.java ocv_loader_init
    Like this:
    ![](images/sample_code.png)

5. Choose a device to check the sample on and run the code by pressing `run` button

    ![](images/run_app.png)

Camera view sample
------------------

In this section we'll extend our empty OpenCV app created in the previous section to support camera. We'll take camera frames and display them on the screen.

1. Tell a system that we need camera permissions.
    Add the following code to the file `MyApplication/app/src/main/AndroidManifest.xml`:
    @snippet samples/android/tutorial-1-camerapreview/gradle/AndroidManifest.xml camera_permissions
    Like this:
    ![](images/camera_permissions.png)

2. Go to `activity_main.xml` layout and delete TextView with text "Hello World!"

    ![](images/delete_text.png)

    This can also be done in Code or Split mode by removing the `TextView` block from XML file.

3. Add camera view to the layout:
    1. Add a scheme into layout description:
    @code{.xml}
    xmlns:opencv="http://schemas.android.com/apk/res-auto"
    @endcode

    2. Replace `TextView` with `org.opencv.android.JavaCameraView` widget:
    @snippet /samples/android/tutorial-1-camerapreview/res/layout/tutorial1_surface_view.xml camera_view

    3. If you get a layout warning replace `fill_parent` values by `match_parent` for `android:layout_width` and `android:layout_height` properties

    You'll get a code like this:

    @include /samples/android/tutorial-1-camerapreview/res/layout/tutorial1_surface_view.xml

4. Inherit the main class from `org.opencv.android.CameraActivity`. CameraActivity implements
   camera perimission requiest and some other utilities needed for CV application. Methods we're
   interested in to override are `onCreate`, `onDestroy`, `onPause`, `onResume` and `getCameraViewList`

5. Implement the interface `org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2`
   `onCameraFrame` method should return the `Mat` object with content for render.
    The sample just returns camera frame for preview: `return inputFrame.rgba();`

6. Allocate `org.opencv.android.CameraBridgeViewBase` object:
    - It should be created at app start (`onCreate` method) and this class should be set as a listener
    - At pause/resume (`onPause`, `onResume` methods) it should be disabled/enabled
    - Should be disabled at app finish (`onDestroy` method)
    - Should be returned in `getCameraViewList`

7. Optionally you can forbid the phone to dim screen or lock:

    @snippet samples/android/tutorial-1-camerapreview/src/org/opencv/samples/tutorial1/Tutorial1Activity.java keep_screen

Finally you'll get source code similar to this:

@include samples/android/tutorial-1-camerapreview/src/org/opencv/samples/tutorial1/Tutorial1Activity.java

This is it! Now you can run the code on your device to check it.


Let's discuss some most important steps
---------------------------------------

Every Android application with UI must implement Activity and View. By the first steps we create blank
activity and default view layout. The simplest OpenCV-centric application must perform OpenCV
initialization, create a view to show preview from camera and implement `CvCameraViewListener2` interface
to get frames from camera and process them.

First of all we create our application view using XML layout. Our layout consists of the only one
full screen component of class `org.opencv.android.JavaCameraView`. This OpenCV class is inherited from
 `CameraBridgeViewBase` that extends `SurfaceView` and under the hood uses standard Android camera API.

The `CvCameraViewListener2` interface lets you add some processing steps after the frame is grabbed from
the camera and before it's rendered on the screen. The most important method is `onCameraFrame`. This is
a callback function and it's called on retrieving frame from camera. It expects that `onCameraFrame`
function returns RGBA frame that will be drawn on the screen.

The callback passes a frame from camera to our class as an object of `CvCameraViewFrame` class.
This object has `rgba()` and `gray()` methods that let a user get colored or one-channel grayscale
frame as a `Mat` class object.

@note Do not save or use `CvCameraViewFrame` object out of `onCameraFrame` callback. This object does
not have its own state and its behavior outside the callback is unpredictable!
