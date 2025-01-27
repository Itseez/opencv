# MultiArch cross-compilation with Ubuntu/Debian{#tutorial_crosscompile_with_multiarch}

@prev_tutorial{tutorial_arm_crosscompile_with_cmake}
@next_tutorial{tutorial_building_tegra_cuda}

[TOC]

|    |    |
| -: | :- |
| Original author | Kumataro |
| Compatibility   | Ubuntu >=23.04 |
|^                | OpenCV >=4.8.0 |

@tableofcontents

@warning
This tutorial may contain obsolete information.

## What is "MultiArch"

OpenCV may use a lot of 3rdparty libraries for video and image decoding, rendering, acceleration
and complex math algorithms. The 3rd party components are found by CMake on the build host
cross-compilation allows to build OpenCV for foreign architecture or OS, but we loose that large
world of components and have to cross-compile each dependency separately and point to it during
OpenCV build.

Debian/Ubuntu MultiArch helps to fix this. It allows to install several foreign architecture
libraries on host system and use them during OpenCV dependencies resolution.

@warning
- Following these steps will make your Linux environment a little dirty.
  If possible, it is better to use VMs or Container(e.g. Docker).
- This tutorial expects host and target uses same Ubuntu version.
   Do not use/mix different versions for external library dependency.
  - Good: Host and Target are 23.04.
  - Good: Host and Target are 23.10.
  - Not Good: Host is 23.04, and Target is 23.10.
  - Not Good: Host is 23.10, and Target is 23.04.
- This tutorial may be used for Debian and its derivatives like Raspberry Pi OS. Please make any
necessary changes.

## Download tools

Install necessary tools and toolchains for cross-compilation.

- git, cmake, pkgconf and build-essential are required basically.
- ninja-build is to reduce compilation time(option).
- crossbuild-essential-armhf is toolchain package for armv7 target.
- crossbuild-essential-arm64 is toolchain package for aarch64 target.

@code{.bash}
sudo apt update -y
sudo apt install -y \
    git \
    cmake \
    pkgconf \
    build-essential \
    ninja-build \
    crossbuild-essential-armhf \
    crossbuild-essential-arm64
@endcode

If you want to enable Python 3 wrapper, install these packages too.

@code{.bash}
sudo apt install -y \
    python3-minimal \
    python3-numpy
@endcode

## Working folder structure

In this tutorial, following working folder structure are used.

@code{.unparsed}
/home
  + kmtr                    - please replace your account name.
    + work
      + opencv              - source, cloned from github
      + opencv_contrib      - source, cloned from github
      + build4-full_arm64   - artifact(for aarch64 target), created by cmake
      + build4-full_armhf   - artifact(for armhf target), created by cmake
@endcode

1. Create working folder under your home directory.
2. Clone OpenCV and OpenCV Contrib from repository to work directory.

@code{.bash}
cd ~
mkdir work
cd work
git clone --depth=1 https://github.com/opencv/opencv.git
git clone --depth=1 https://github.com/opencv/opencv_contrib.git
@endcode

## Update apt and dpkg settings

These steps are on host.

`apt` and `dpkg` are package management systems used in Ubuntu and Debian.

Following are setup steps to use MultiArch.

### Step 1. Add apt source for arm64 and armhf

Execute `sudo apt edit-sources` to add foreign arch libraries at end of file.

Example 1: arm64 and armv7 for Ubuntu 23.04

@code{.unparsed}
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-updates main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-updates universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-updates multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-backports main restricted universe multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-security main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-security universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports lunar-security multiverse
@endcode

Example 2: arm64 and armv7 for Ubuntu 23.10

@code{.unparsed}
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-updates main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-updates universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-updates multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-backports main restricted universe multiverse
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-security main restricted
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-security universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports mantic-security multiverse
@endcode

### Step 2. Update apt database

Update apt database to apply new apt sources.

Execute `sudo apt update`.

@code{.bash}
sudo apt update
@endcode

### Step 3. Update dpkg settings

Update dpkg settings to support foreign architectures.

Execute `sudo dpkg --add-architecture arm64` and/or `sudo dpkg --add-architecture armhf`.

@code{.bash}
sudo dpkg --add-architecture arm64
sudo dpkg --add-architecture armhf
@endcode

`sudo dpkg --print-architecture` shows what is host architecture.

@code{.bash}
sudo dpkg --print-architecture
amd64
@endcode

And `sudo dpkg --print-foreign-architectures` shows what foreign architectures are supported.

@code{.bash}
sudo dpkg --print-foreign-architectures
arm64
armhf
@endcode

### Confirm working pkg-config

With MultiArch, several shared libraries and pkg-config information for each architectures are stored into /usr/lib.

@code{.unparsed}
/usr
  + lib
    + aarch64-linux-gnu   - shared libraries for arm64
      + pkgconfig         - pkg-config files for arm64 libraries
    + arm-linux-gnueabihf - shared libraries for armhf
      + pkgconfig         - pkg-config files for armhf libraries
  + share
    + pkgconfig         - pkg-config files(for header files)
@endcode

Confirm to work `pkg-config` using `PKG_CONFIG_PATH`, `PKG_CONFIG_LIBDIR` and `PKG_CONFIG_SYSROOT_DIR` options.

for aarch64:

@code{.bash}
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
      pkg-config --list-all
@endcode

for armv7:

@code{.bash}
PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig:/usr/share/pkgconfig \
  PKG_CONFIG_LIBDIR=/usr/lib/arm-linux-gnueabihf \
  PKG_CONFIG_SYSROOT_DIR=/ \
      pkg-config --list-all
@endcode

## Cross-compile for aarch64

Following is to compile for target (aarch64) at host (x86-64).

### Step 1. Install external libraries for target into host

This step is on host.

Install libfreetype-dev, libharfbuzz-dev and FFmpeg packages for target (arm64) into host (x86-64).

@code{.bash}
sudo apt install -y \
    libavcodec-dev:arm64 \
    libavformat-dev:arm64 \
    libavutil-dev:arm64 \
    libswscale-dev:arm64 \
    libfreetype-dev:arm64 \
    libharfbuzz-dev:arm64
@endcode

If you want to enable Python 3 wrapper, install these packages too.

@code{.bash}
sudo apt install -y \
    libpython3-dev:arm64
@endcode

If succeed, pkg-config can show information about these packages.

For Freetype2 and Harfbuzz:

@code{.bash}
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
       pkg-config freetype2 harfbuzz --cflags --libs
-I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/harfbuzz -I/usr/include/glib-2.0 -I/usr/lib/aarch64-linux-gnu/glib-2.0/include -L/usr/lib/aarch64-linux-gnu -lfreetype -lharfbuzz
@endcode

For FFmpeg:

@code{.bash}
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
       pkg-config libavcodec libavformat libavutil libswscale --cflags --libs
-I/usr/include/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu -lavcodec -lavformat -lavutil -lswscale
@endcode

### Step 2. Configure OpenCV Settings
This step is on host.

Execute `cmake` to make cross-compile configuration for aarch64.

@note `-DCMAKE_TOOLCHAIN_FILE` should be absolute/real file path, not relative path.

@code{.bash}
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
        cmake -S opencv \
              -B build4-full_arm64 \
              -DCMAKE_TOOLCHAIN_FILE=/home/kmtr/work/opencv/platforms/linux/aarch64-gnu.toolchain.cmake \
              -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
              -GNinja
@endcode

If you want to enable Python 3 wrapper, extra options are needed.

@code{.bash}
PYTHON3_REALPATH=`realpath /usr/bin/python3`
PYTHON3_BASENAME=`basename ${PYTHON3_REALPATH}`
PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
    PKG_CONFIG_SYSROOT_DIR=/ \
        cmake -S opencv \
              -B build4-full_arm64 \
              -DCMAKE_TOOLCHAIN_FILE=/home/kmtr/work/opencv/platforms/linux/aarch64-gnu.toolchain.cmake \
              -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
              -DPYTHON3_NUMPY_INCLUDE_DIRS="/usr/local/lib/${PYTHON3_BASENAME}/dist-packages/numpy/core/include/" \
              -DPYTHON3_INCLUDE_PATH="/usr/include/${PYTHON3_BASENAME};/usr/include/" \
              -DPYTHON3_LIBRARIES=`find /usr/lib/aarch64-linux-gnu/ -name libpython*.so` \
              -DPYTHON3_EXECUTABLE="/usr/bin/${PYTHON3_BASENAME}" \
              -DPYTHON3_CVPY_SUFFIX=".so" \
              -GNinja
@endcode

@note
@parblock
Lastly, "python3.XX" string is needed. So this script generate it.
- Get real path from "/usr/bin/python3" to "/usr/bin/python3.xx".
- Get base name from "/usr/bin/python3.xx" to "pyhton3.xx".
@endparblock

Following is cmake outputs.
- `Host` is `Linux x86_64`.
- `Target` is `Linux aarch64`.
- FFmpeg is available.

@code{.unparsed}
-- General configuration for OpenCV 4.8.0-dev =====================================
--   Version control:               408730b
--
--   Extra modules:
--     Location (extra):            /home/kmtr/work/opencv_contrib/modules
--     Version control (extra):     faa5468
--
--   Platform:
--     Timestamp:                   2023-12-01T22:02:14Z
--     Host:                        Linux 6.5.0-13-generic x86_64
--     Target:                      Linux 1 aarch64
--     CMake:                       3.27.4
--     CMake generator:             Ninja
--     CMake build tool:            /usr/bin/ninja
--     Configuration:               Release
--
--   CPU/HW features:
--     Baseline:                    NEON FP16
--       required:                  NEON
--       disabled:                  VFPV3
--     Dispatched code generation:  NEON_DOTPROD NEON_FP16 NEON_BF16
--       requested:                 NEON_FP16 NEON_BF16 NEON_DOTPROD
--       NEON_DOTPROD (1 files):    + NEON_DOTPROD
--       NEON_FP16 (2 files):       + NEON_FP16
--       NEON_BF16 (0 files):       + NEON_BF16
--
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/aarch64-linux-gnu-g++  (ver 13.2.0)

:
:

--
--   Video I/O:
--     DC1394:                      NO
--     FFMPEG:                      YES
--       avcodec:                   YES (60.3.100)
--       avformat:                  YES (60.3.100)
--       avutil:                    YES (58.2.100)
--       swscale:                   YES (7.1.100)
--       avresample:                NO
--     GStreamer:                   NO
--     v4l/v4l2:                    YES (linux/videodev2.h)
--
@endcode

If enabling Python 3 wrapper is succeeded, `Python 3:` section shows more.

@code{.unparsed}
--
--   Python 3:
--     Interpreter:                 /usr/bin/python3.11 (ver 3.11.6)
--     Libraries:                   /usr/lib/aarch64-linux-gnu/libpython3.11.so
--     numpy:                       /usr/local/lib/python3.11/dist-packages/numpy/core/include/ (ver undefined - cannot be probed because of the cross-compilation)
--     install path:                lib/python3.11/dist-packages/cv2/python-3.11
--
--   Python (for build):            /usr/bin/python3.11
--
@endcode

### Step 3. Build and archive OpenCV libraries and headers

This step in in host.

Build and install.
(This `install` means only that copying artifacts to `install` folder.)

@code{.bash}
     cmake --build   build4-full_arm64
sudo cmake --install build4-full_arm64
@endcode

Archive artifacts(built libraries and headers) to `opencv_arm64.tgz` with tar command.

@code{.bash}
tar czvf opencv_arm64.tgz -C build4-full_arm64/install .
@endcode

And send `opencv_arm64.tgz` to target.

### Step 4. Install dependency libraries at target

This step is executed on the target system.

Install dependency run-time libraries for OpenCV/OpenCV contrib libraries at target.

@code{.bash}
sudo apt install -y \
    libavcodec60 \
    libavformat60 \
    libavutil58 \
    libswscale7 \
    libfreetype6 \
    libharfbuzz0b

sudo ldconfig
@endcode

If you want to enable Python 3 wrapper, install these packages too.

@code{.bash}
sudo apt install -y \
    python3-minimal \
    python3-numpy
@endcode

@warning
@parblock
If version of runtime libraries and/or programs are incremented, apt package names may be changed
(e.g. `libswscale6` is used for Ubuntu 23.04, but `libswscale7` is used for Ubuntu 23.10).
Looking for it with `apt search` command or https://packages.ubuntu.com/ .
@endparblock

@warning
@parblock
External library version between host and target should be same.
Please update to the latest version libraries at the same time as possible.

Even if the OS versions are the same between the Host and Target,
the versions may differ due to additional updates to the libraries.
This will cause unexpected problems.

For example)
- On Host, OpenCV has been build with external libA (v1.0) for target.
- libA (v1.1) may be updated.
- On Target, libA (v1.1) is installed to use OpenCV.
- In this case, versions of libA is difference between compiling and running.
@endparblock

@warning
@parblock
If you forget/mismatch to install some necessary libraries, OpenCV will not works well.

`ldd` command can detect dependency. If there are any "not found", please install necessary libraries.

@code{.bash}
ldd /usr/local/lib/libopencv_freetype.so
@endcode

(Not Good) `freetype module` requires `libharfbuzz.so.0`, but it has not been installed.
@code{.unparsed}
        linux-vdso.so.1 (0xABCDEFG01234567)
        libopencv_imgproc.so.408 => /usr/local/lib/libopencv_imgproc.so.408 (0xABCDEF001234567)
        libfreetype.so.6 => /lib/aarch64-linux-gnu/libfreetype.so.6 (0xABCDEF001234567)
        libharfbuzz.so.0 => not found
        libopencv_core.so.408 => /usr/local/lib/libopencv_core.so.408 (0xABCDEF001234567)
        :
@endcode

(Good) All libraries which are required from `freetype modules` are installed.
@code{.unparsed}
        linux-vdso.so.1 (0xABCDEFG01234567)
        libopencv_imgproc.so.408 => /usr/local/lib/libopencv_imgproc.so.408 (0xABCDEF001234567)
        libfreetype.so.6 => /lib/aarch64-linux-gnu/libfreetype.so.6 (0xABCDEF001234567)
        libharfbuzz.so.0 => /lib/aarch64-linux-gnu/libharfbuzz.so.0 (0xABCDEF001234567)
        libopencv_core.so.408 => /usr/local/lib/libopencv_core.so.408 (0xABCDEF001234567)
        :
@endcode
@endparblock

### Step 5. Install OpenCV libraries to target

This step is on target.

Receive `opencv_arm64.tgz` from host (generated at Step3), and extract to `/usr/local`.

@code{.bash}
sudo tar zxvf opencv_arm64.tgz -C /usr/local
sudo ldconfig
@endcode

You can use OpenCV libraries same as self-compiling. Following is OpenCV sample code. Compile and
run it on target.

Makefile
@code{.make}
a.out : main.cpp
    g++ main.cpp -o a.out \
        -I/usr/local/include/opencv4 \
        -lopencv_core
@endcode

main.cpp
@code{.cpp}
#include <iostream>
#include <opencv2/core.hpp>
int main(void)
{
  std::cout << cv::getBuildInformation() << std::endl;
  return 0;
}
@endcode

Execute `make` and run it.
@code{.bash}
make a.out
./a.out
@endcode

If you want to enable Python 3 wrapper, execute following command to confirm.

@code{.bash}
python3 -c "import cv2; print(cv2.getBuildInformation())"
@endcode

## Cross-compile for armv7

Following is to compile for target (armhf) at host (x86-64).

- To resolve dependencies, `linux-libc-dev:armhf` is required.
- To optimize with neon, `-DENABLE_NEON=ON` is needed.

@code{.bash}
sudo apt install -y \
    linux-libc-dev:armhf \
    libavcodec-dev:armhf \
    libavformat-dev:armhf \
    libavutil-dev:armhf \
    libswscale-dev:armhf \
    libfreetype-dev:armhf \
    libharfbuzz-dev:armhf

PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig:/usr/share/pkgconfig \
    PKG_CONFIG_LIBDIR=/usr/lib/arm-linux-gnueabihf \
    PKG_CONFIG_SYSROOT_DIR=/ \
        cmake -S opencv \
              -B build4-full_armhf \
              -DENABLE_NEON=ON \
              -DCMAKE_TOOLCHAIN_FILE=/home/kmtr/work/opencv/platforms/linux/arm-gnueabi.toolchain.cmake \
              -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
              -GNinja

cmake      --build   build4-full_armhf
sudo cmake --install build4-full_armhf
tar czvf opencv_armhf.tgz -C build4-full_armhf/install .
@endcode

Following is cmake outputs.
- `Host` is `Linux x86_64`.
- `Target` is `Linux arm`.
- FFmpeg is available.

@code{.unparsed}
-- General configuration for OpenCV 4.8.0-dev =====================================
--   Version control:               408730b
--
--   Extra modules:
--     Location (extra):            /home/kmtr/work/opencv_contrib/modules
--     Version control (extra):     faa5468
--
--   Platform:
--     Timestamp:                   2023-12-02T03:39:58Z
--     Host:                        Linux 6.5.0-13-generic x86_64
--     Target:                      Linux 1 arm
--     CMake:                       3.27.4
--     CMake generator:             Ninja
--     CMake build tool:            /usr/bin/ninja
--     Configuration:               Release
--
--   CPU/HW features:
--     Baseline:                    NEON
--       requested:                 DETECT
--       required:                  NEON
--       disabled:                  VFPV3
--
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/arm-linux-gnueabihf-g++  (ver 13.2.0)

:
:

--
--   Video I/O:
--     DC1394:                      NO
--     FFMPEG:                      YES
--       avcodec:                   YES (60.3.100)
--       avformat:                  YES (60.3.100)
--       avutil:                    YES (58.2.100)
--       swscale:                   YES (7.1.100)
--       avresample:                NO
--     GStreamer:                   NO
--     v4l/v4l2:                    YES (linux/videodev2.h)
--

@endcode
