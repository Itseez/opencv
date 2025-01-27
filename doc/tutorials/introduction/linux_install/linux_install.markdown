Installation in Linux {#tutorial_linux_install}
=====================

@prev_tutorial{tutorial_env_reference}
@next_tutorial{tutorial_linux_gdb_pretty_printer}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

@tableofcontents

# Quick start {#tutorial_linux_install_quick_start}


## Build core modules {#tutorial_linux_install_quick_build_core}

@snippet linux_quick_install.sh body


## Build with opencv_contrib {#tutorial_linux_install_quick_build_contrib}

@snippet linux_quick_install_contrib.sh body


# Detailed process {#tutorial_linux_install_detailed}

This section provides more details of the build process and describes alternative methods and tools. Please refer to the @ref tutorial_general_install tutorial for general installation details and to the @ref tutorial_config_reference for configuration options documentation.


## Install compiler and build tools {#tutorial_linux_install_detailed_basic_compiler}

- To compile OpenCV you will need a C++ compiler. Usually it is G++/GCC or Clang/LLVM:
    - Install GCC...
    @snippet linux_install_a.sh gcc
    - ... or Clang:
    @snippet linux_install_b.sh clang

- OpenCV uses CMake build configuration tool:
@snippet linux_install_a.sh cmake

- CMake can generate scripts for different build systems, e.g. _make_, _ninja_:

    - Install Make...
    @snippet linux_install_a.sh make
    - ... or Ninja:
    @snippet linux_install_b.sh ninja

- Install tool for getting and unpacking sources:

    - _wget_ and _unzip_...
    @snippet linux_install_a.sh wget
    - ... or _git_:
    @snippet linux_install_b.sh git


## Download sources {#tutorial_linux_install_detailed_basic_download}

There are two methods of getting OpenCV sources:

- Download snapshot of repository using web browser or any download tool (~80-90Mb) and unpack it...
@snippet linux_install_a.sh download
- ... or clone repository to local machine using _git_ to get full change history (>470Mb):
@snippet linux_install_b.sh download


@note
Snapshots of other branches, releases or commits can be found on the [GitHub](https://github.com/opencv/opencv) and the [official download page](https://opencv.org/releases).


## Configure and build {#tutorial_linux_install_detailed_basic_build}

- Create build directory:
@snippet linux_install_a.sh prepare

- Configure - generate build scripts for the preferred build system:
    - For _make_...
    @snippet linux_install_a.sh configure
    - ... or for _ninja_:
    @snippet linux_install_b.sh configure

- Build - run actual compilation process:
    - Using _make_...
    @snippet linux_install_a.sh build
    - ... or _ninja_:
    @snippet linux_install_b.sh build


@note
_Configure_ process can download some files from the internet to satisfy library dependencies, connection failures can cause some of modules or functionalities to be turned off or behave differently. Refer to the @ref tutorial_general_install and @ref tutorial_config_reference tutorials for details and full configuration options reference.

@note
If you experience problems with the build process, try to clean or recreate the build directory. Changes in the configuration like disabling a dependency, modifying build scripts or switching sources to another branch are not handled very well and can result in broken workspace.

@note
_Make_ can run multiple compilation processes in parallel, `-j<NUM>` option means "run <NUM> jobs simultaneously". _Ninja_ will automatically detect number of available processor cores and does not need `-j` option.


## Check build results {#tutorial_linux_install_detailed_basic_verify}

After successful build you will find libraries in the `build/lib` directory and executables (test, samples, apps) in the `build/bin` directory:
@snippet linux_install_a.sh check

CMake package files will be located in the build root:
@snippet linux_install_a.sh check cmake


## Install

@warning
The installation process only copies files to predefined locations and does minor patching. Installing using this method does not integrate opencv into the system package registry and thus, for example, opencv can not be uninstalled automatically. We do not recommend system-wide installation to regular users due to possible conflicts with system packages.

By default OpenCV will be installed to the `/usr/local` directory, all files will be copied to following locations:
* `/usr/local/bin` - executable files
* `/usr/local/lib` - libraries (.so)
* `/usr/local/cmake/opencv4` - cmake package
* `/usr/local/include/opencv4` - headers
* `/usr/local/share/opencv4` - other files (e.g. trained cascades in XML format)

Since `/usr/local` is owned by the root user, the installation should be performed with elevated privileges (`sudo`):
@snippet linux_install_a.sh install
or
@snippet linux_install_b.sh install

Installation root directory can be changed with `CMAKE_INSTALL_PREFIX` configuration parameter, e.g. `-DCMAKE_INSTALL_PREFIX=$HOME/.local` to install to current user's local directory. Installation layout can be changed with `OPENCV_*_INSTALL_PATH` parameters. See @ref tutorial_config_reference for details.
