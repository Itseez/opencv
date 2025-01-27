Using OpenCV with gcc and CMake {#tutorial_linux_gcc_cmake}
===============================

@prev_tutorial{tutorial_linux_gdb_pretty_printer}
@next_tutorial{tutorial_linux_eclipse}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

@tableofcontents

@warning
This tutorial can contain obsolete information.

@note We assume that you have successfully installed OpenCV in your workstation.

-   The easiest way of using OpenCV in your code is to use [CMake](http://www.cmake.org/). A few
    advantages (taken from the Wiki):
    -#  No need to change anything when porting between Linux and Windows
    -#  Can easily be combined with other tools by CMake( i.e. Qt, ITK and VTK )
-   If you are not familiar with CMake, checkout the
    [tutorial](http://www.cmake.org/cmake/help/cmake_tutorial.html) on its website.

Steps
-----

### Create a program using OpenCV

Let's use a simple program such as DisplayImage.cpp shown below.
@code{.cpp}
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], IMREAD_COLOR );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    waitKey(0);

    return 0;
}
@endcode
### Create a CMake file

Now you have to create your CMakeLists.txt file. It should look like this:
@code{.cmake}
cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
@endcode
### Generate the executable

This part is easy, just proceed as with any other project using CMake:
@code{.bash}
cd <DisplayImage_directory>
cmake .
make
@endcode
### Result

By now you should have an executable (called DisplayImage in this case). You just have to run it
giving an image location as an argument, i.e.:
@code{.bash}
./DisplayImage lena.jpg
@endcode
You should get a nice window as the one shown below:

![](images/GCC_CMake_Example_Tutorial.jpg)
