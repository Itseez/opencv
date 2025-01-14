Installation in MacOS {#tutorial_macos_install}
=====================

@prev_tutorial{tutorial_android_ocl_intro}
@next_tutorial{tutorial_arm_crosscompile_with_cmake}

|    |    |
| -: | :- |
| Original author | `@sajarindider` |
| Compatibility | OpenCV >= 3.4 |

The following steps have been tested for macOS (Mavericks) but should work with other versions as well.

Required Packages
-----------------

-   CMake 3.9 or higher
-   Git
-   Python 3.x and NumPy 1.5 or later

This tutorial will assume you have [Python](https://docs.python.org/3/using/mac.html),
[NumPy](https://numpy.org/install/) and
[Git](https://git-scm.com/downloads/mac) installed on your machine.

@note
-   macOS up to 12.2 (Monterey): Comes with Python 2.7 pre-installed.
-   macOS 12.3 and later: Python 2.7 has been removed, and no version of Python is included by default.

It is recommended to install the latest version of Python 3.x (at least Python 3.8) for compatibility with the latest OpenCV Python bindings.

@note
If you have Xcode and Xcode Command Line Tools installed, Git is already available on your machine.

Installing CMake
----------------
-# Find the version for your system and download CMake from their release's [page](https://cmake.org/download/)

-# Install the `.dmg` package and launch it from Applications. That will give you the UI app of CMake

-# From the CMake app window, choose menu Tools --> How to Install For Command Line Use. Then, follow the instructions from the pop-up there.

-# The install folder will be `/usr/local/bin/` by default. Complete the installation by choosing Install command line links.

-# Test that CMake is installed correctly by running:

    @code{.bash}
    cmake --version
    @endcode

@note You can use [Homebrew](https://brew.sh/) to install CMake with:

    @code{.bash}
    brew install cmake
    @endcode

Getting OpenCV Source Code
--------------------------

You can use the latest stable OpenCV version or you can grab the latest snapshot from our
[Git repository](https://github.com/opencv/opencv.git).

### Getting the Latest Stable OpenCV Version

-   Go to our [OpenCV releases page](https://opencv.org/releases).
-   Download the source archive of the latest version (e.g., OpenCV 4.x) and unpack it.

### Getting the Cutting-edge OpenCV from the Git Repository

Launch Git client and clone [OpenCV repository](https://github.com/opencv/opencv).
If you need modules from [OpenCV contrib repository](https://github.com/opencv/opencv_contrib) then clone it as well.

For example:

    @code{.bash}
    cd ~/<your_working_directory>
    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
    @endcode

Building OpenCV from Source Using CMake
---------------------------------------

-#  Create a temporary directory, which we denote as `build_opencv`, where you want to put
    the generated Makefiles, project files as well the object files and output binaries and enter
    there.

    For example:

    @code{.bash}
    mkdir build_opencv
    cd build_opencv
    @endcode

    @note It is good practice to keep your source code directories clean. Create the build directory outside of the source tree.

-#  Configuring. Run `cmake [<some optional parameters>] <path to the OpenCV source directory>`

    For example:

    @code{.bash}
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ../opencv
    @endcode

    Alternatively, you can use the CMake GUI (`cmake-gui`):

    -   set the OpenCV source code path to, e.g. `/Users/your_username/opencv`
    -   set the binary build path to your CMake build directory, e.g. `/Users/your_username/build_opencv`
    -   set optional parameters
    -   run: "Configure"
    -   run: "Generate"

-#  Description of some parameters
    -   build type: `-DCMAKE_BUILD_TYPE=Release` (or `Debug`).
    -   include Extra Modules: If you cloned the `opencv_contrib` repository and want to include its modules, set:

        @code{.bash}
        -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules
        @endcode
    -   set `-DBUILD_DOCS=ON` for building documents (doxygen is required)
    -   set `-DBUILD_EXAMPLES=ON` to build all examples

-#  [optional] Building python. Set the following python parameters:
    -   `-DPYTHON3_EXECUTABLE=$(which python3)`
    -   `-DPYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")`
    -   `-DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())")`

-#  Build. From build directory execute *make*, it is recommended to do this in several threads

    For example:

    @code{.bash}
    make -j$(sysctl -n hw.ncpu) # runs the build using all available CPU cores
    @endcode

-#  After building, you can install OpenCV system-wide using:

    @code{.bash}
    sudo make install
    @endcode

-#  To use OpenCV in your CMake-based projects through `find_package(OpenCV)`, specify the `OpenCV_DIR` variable pointing to the build or install directory.

    For example:

    @code{.bash}
    cmake -DOpenCV_DIR=~/build_opencv ..
    @endcode

### Verifying the OpenCV Installation

After building (and optionally installing) OpenCV, you can verify the installation by checking the version using Python:

    @code{.bash}
    python3 -c "import cv2; print(cv2.__version__)"
    @endcode

This command should output the version of OpenCV you have installed.

@note
You can also use a package manager like [Homebrew](https://brew.sh/)
or [pip](https://pip.pypa.io/en/stable/) to install releases of OpenCV only (Not the cutting edge).

- Installing via Homebrew:

    For example:

    @code{.bash}
    brew install opencv
    @endcode

- Installing via pip:

    For example:

    @code{.bash}
    pip install opencv-python
    @endcode

    @note To access the extra modules from `opencv_contrib`, install the `opencv-contrib-python` package using `pip install opencv-contrib-python`.
