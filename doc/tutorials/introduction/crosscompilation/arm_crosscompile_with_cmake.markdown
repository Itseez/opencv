Cross compilation for ARM based Linux systems {#tutorial_arm_crosscompile_with_cmake}
=============================================

@prev_tutorial{tutorial_macos_install}
@next_tutorial{tutorial_crosscompile_with_multiarch}

|    |    |
| -: | :- |
| Original author |  Alexander Smorkalov |
| Compatibility | OpenCV >= 3.0 |

@tableofcontents

@warning
This tutorial can contain obsolete information.

This steps are tested on Ubuntu Linux 12.04, but should work for other Linux distributions. I case
of other distributions package names and names of cross compilation tools may differ. There are
several popular EABI versions that are used on ARM platform. This tutorial is written for *gnueabi*
and *gnueabihf*, but other variants should work with minimal changes.

Prerequisites
-------------

-   Host computer with Linux;
-   Git;
-   CMake 2.6 or higher;
-   Cross compilation tools for ARM: gcc, libstc++, etc. Depending on target platform you need to
    choose *gnueabi* or *gnueabihf* tools. Install command for *gnueabi*:
    @code{.bash}
    sudo apt-get install gcc-arm-linux-gnueabi
    @endcode
    Install command for *gnueabihf*:
    @code{.bash}
    sudo apt-get install gcc-arm-linux-gnueabihf
    @endcode
-   pkgconfig;
-   Python 2.6 for host system;
-   [optional] ffmpeg or libav development packages for armeabi(hf): libavcodec-dev,
    libavformat-dev, libswscale-dev;
-   [optional] GTK+2.x or higher, including headers (libgtk2.0-dev) for armeabi(hf);
-   [optional] libdc1394 2.x;
-   [optional] libjpeg-dev, libpng-dev, libtiff-dev, libjasper-dev for armeabi(hf).

Getting OpenCV Source Code
--------------------------

You can use the latest stable OpenCV version available in *sourceforge* or you can grab the latest
snapshot from our [Git repository](https://github.com/opencv/opencv.git).

### Getting the Latest Stable OpenCV Version

-   Go to our [page on Sourceforge](http://sourceforge.net/projects/opencvlibrary);
-   Download the source tarball and unpack it.

### Getting the Cutting-edge OpenCV from the Git Repository

Launch Git client and clone [OpenCV repository](http://github.com/opencv/opencv)

In Linux it can be achieved with the following command in Terminal:
@code{.bash}
cd ~/<my_working _directory>
git clone https://github.com/opencv/opencv.git
@endcode

Building OpenCV
---------------

-#  Create a build directory, make it current and run the following command:
    @code{.bash}
    cmake [<some optional parameters>] -DCMAKE_TOOLCHAIN_FILE=<path to the OpenCV source directory>/platforms/linux/arm-gnueabi.toolchain.cmake <path to the OpenCV source directory>
    @endcode
    Toolchain uses *gnueabihf* EABI convention by default. Add -DSOFTFP=ON cmake argument to switch
    on softfp compiler.
    @code{.bash}
    cmake [<some optional parameters>] -DSOFTFP=ON -DCMAKE_TOOLCHAIN_FILE=<path to the OpenCV source directory>/platforms/linux/arm-gnueabi.toolchain.cmake <path to the OpenCV source directory>
    @endcode
    For example:
    @code{.bash}
    cd ~/opencv/platforms/linux
    mkdir -p build_hardfp
    cd build_hardfp

    cmake -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake ../../..
    @endcode

-#  Run make in build (\<cmake_binary_dir\>) directory:
    @code{.bash}
    make
    @endcode

@note
Optionally you can strip symbols info from the created library via install/strip make target.
This option produces smaller binary (\~ twice smaller) but makes further debugging harder.

### Enable hardware optimizations

Depending on target platform architecture different instruction sets can be used. By default
compiler generates code for armv5l without VFPv3 and NEON extensions. Add -DENABLE_VFPV3=ON to
cmake command line to enable code generation for VFPv3 and -DENABLE_NEON=ON for using NEON SIMD
extensions.

TBB is supported on multi core ARM SoCs also. Add -DWITH_TBB=ON and -DBUILD_TBB=ON to enable it.
Cmake scripts download TBB sources from official project site
<http://threadingbuildingblocks.org/> and build it.
