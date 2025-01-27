Getting Started with Images {#tutorial_display_image}
===========================

@prev_tutorial{tutorial_building_tegra_cuda}
@next_tutorial{tutorial_documentation}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.4.4 |

@tableofcontents

@warning
This tutorial can contain obsolete information.

Goal
----

In this tutorial you will learn how to:

-   Read an image from file (using @ref cv::imread)
-   Display an image in an OpenCV window (using @ref cv::imshow)
-   Write an image to a file (using @ref cv::imwrite)

Source Code
-----------

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/introduction/display_image/display_image.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/introduction/display_image/display_image.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/introduction/display_image/display_image.py
@end_toggle


Explanation
-----------

@add_toggle_cpp
In OpenCV 3 we have multiple modules. Each one takes care of a different area or approach towards
image processing. You could already observe this in the structure of the user guide of these
tutorials itself. Before you use any of them you first need to include the header files where the
content of each individual module is declared.

You'll almost always end up using the:

- @ref core "core" section, as here are defined the basic building blocks of the library
- @ref imgcodecs "imgcodecs" module, which provides functions for reading and writing
- @ref highgui "highgui" module, as this contains the functions to show an image in a window

We also include the *iostream* to facilitate console line output and input.

By declaring `using namespace cv;`, in the following, the library functions can be accessed without explicitly stating the namespace.

@snippet cpp/tutorial_code/introduction/display_image/display_image.cpp includes
@end_toggle

@add_toggle_python
As a first step, the OpenCV python library is imported.
The proper way to do this is to additionally assign it the name *cv*, which is used in the following to reference the library.

@snippet samples/python/tutorial_code/introduction/display_image/display_image.py imports
@end_toggle

Now, let's analyze the main code.
As a first step, we read the image "starry_night.jpg" from the OpenCV samples.
In order to do so, a call to the @ref cv::imread function loads the image using the file path specified by the first argument.
The second argument is optional and specifies the format in which we want the image. This may be:

-   IMREAD_COLOR loads the image in the BGR 8-bit format. This is the **default** that is used here.
-   IMREAD_UNCHANGED loads the image as is (including the alpha channel if present)
-   IMREAD_GRAYSCALE loads the image as an intensity one

After reading in the image data will be stored in a @ref cv::Mat object.

@add_toggle_cpp
@snippet cpp/tutorial_code/introduction/display_image/display_image.cpp imread
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/introduction/display_image/display_image.py imread
@end_toggle

@note
   OpenCV offers support for the image formats Windows bitmap (bmp), portable image formats (pbm,
    pgm, ppm) and Sun raster (sr, ras). With help of plugins (you need to specify to use them if you
    build yourself the library, nevertheless in the packages we ship present by default) you may
    also load image formats like JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 - codenamed in the CMake as
    Jasper), TIFF files (tiff, tif) and portable network graphics (png). Furthermore, OpenEXR is
    also a possibility.

Afterwards, a check is executed, if the image was loaded correctly.
@add_toggle_cpp
@snippet cpp/tutorial_code/introduction/display_image/display_image.cpp empty
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/introduction/display_image/display_image.py empty
@end_toggle

Then, the image is shown using a call to the @ref cv::imshow function.
The first argument is the title of the window and the second argument is the @ref cv::Mat object that will be shown.

Because we want our window to be displayed until the user presses a key (otherwise the program would
end far too quickly), we use the @ref cv::waitKey function whose only parameter is just how long
should it wait for a user input (measured in milliseconds). Zero means to wait forever.
The return value is the key that was pressed.

@add_toggle_cpp
@snippet cpp/tutorial_code/introduction/display_image/display_image.cpp imshow
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/introduction/display_image/display_image.py imshow
@end_toggle

In the end, the image is written to a file if the pressed key was the "s"-key.
For this the cv::imwrite function is called that has the file path and the cv::Mat object as an argument.

@add_toggle_cpp
@snippet cpp/tutorial_code/introduction/display_image/display_image.cpp imsave
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/introduction/display_image/display_image.py imsave
@end_toggle
