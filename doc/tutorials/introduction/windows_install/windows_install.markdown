Installation in Windows {#tutorial_windows_install}
=======================

@prev_tutorial{tutorial_linux_eclipse}
@next_tutorial{tutorial_windows_visual_studio_opencv}

|    |    |
| -: | :- |
| Original author | Bernát Gábor |
| Compatibility | OpenCV >= 3.0 |

@tableofcontents

@warning
This tutorial can contain obsolete information.

The description here was tested on Windows 7 SP1. Nevertheless, it should also work on any other
relatively modern version of Windows OS. If you encounter errors after following the steps described
below, feel free to contact us via our [OpenCV Q&A forum](https://forum.opencv.org). We'll do our
best to help you out.

@note To use the OpenCV library you have two options: @ref tutorial_windows_install_prebuilt or
@ref tutorial_windows_install_build. While the first one is easier to complete, it only works if you are coding
with the latest Microsoft Visual Studio IDE and do not take advantage of the most advanced
technologies we integrate into our library. .. _Windows_Install_Prebuild:

## Installation by Using the Pre-built Libraries {#tutorial_windows_install_prebuilt}

-#  Launch a web browser of choice and go to our [page on
    Sourceforge](http://sourceforge.net/projects/opencvlibrary/files/).
-#  Choose a build you want to use and download it.
-#  Make sure you have admin rights. Unpack the self-extracting archive.
-#  You can check the installation at the chosen path as you can see below.

    ![](images/OpenCV_Install_Directory.png)

-#  To finalize the installation go to the @ref tutorial_windows_install_path section.

## Installation by Using git-bash (version>=2.14.1) and cmake (version >=3.9.1){#tutorial_windows_gitbash_build}

-#  You must download [cmake (version >=3.9.1)](https://cmake.org) and install it. You must add cmake to PATH variable during installation

-#  You must install [git-bash (version>=2.14.1)](https://git-for-windows.github.io/). Don't add git to PATH variable during installation

-#  Run git-bash. You observe a command line window.
Suppose you want to build opencv and opencv_contrib in c:/lib

-#  In git command line enter following command (if folder does not exist) :
@code{.bash}
mkdir /c/lib
cd /c/lib
@endcode

-#  save this script with name installOCV.sh in c:/lib
@code{.bash}
#!/bin/bash -e
myRepo=$(pwd)
CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 16 2019"
#CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 15 2017 Win64"
#CMAKE_GENERATOR_OPTIONS=(-G"Visual Studio 16 2019" -A x64)  # CMake 3.14+ is required
if [  ! -d "$myRepo/opencv"  ]; then
    echo "cloning opencv"
    git clone https://github.com/opencv/opencv.git
else
    cd opencv
    git pull --rebase
    cd ..
fi
if [  ! -d "$myRepo/opencv_contrib"  ]; then
    echo "cloning opencv_contrib"
    git clone https://github.com/opencv/opencv_contrib.git
else
    cd opencv_contrib
    git pull --rebase
    cd ..
fi
RepoSource=opencv
mkdir -p build_opencv
pushd build_opencv
CMAKE_OPTIONS=(-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON)
set -x
cmake "${CMAKE_GENERATOR_OPTIONS[@]}" "${CMAKE_OPTIONS[@]}" -DOPENCV_EXTRA_MODULES_PATH="$myRepo"/opencv_contrib/modules -DCMAKE_INSTALL_PREFIX="$myRepo/install/$RepoSource" "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->debug"
cmake --build .  --config debug
echo "************************* $Source_DIR -->release"
cmake --build .  --config release
cmake --build .  --target install --config release
cmake --build .  --target install --config debug
popd
@endcode
    In this script I suppose you use VS 2015 in 64 bits
@code{.bash}
CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 14 2015 Win64"
@endcode
    and opencv will be installed in c:/lib/install/opencv
@code{.bash}
-DCMAKE_INSTALL_PREFIX="$myRepo/install/$RepoSource"
@endcode
    with no Perf tests, no tests, no doc, no CUDA and no example
@code{.bash}
CMAKE_OPTIONS=(-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF)
@endcode
-#  In git command line enter following command :
@code{.bash}
./installOCV.sh
@endcode
-# Drink a coffee or two... opencv is ready : That's all!
-# Next time you run this script, opencv and opencv_contrib will be updated and rebuild


## Installation by Making Your Own Libraries from the Source Files {#tutorial_windows_install_build}

You may find the content of this tutorial also inside the following videos:
[Part 1](https://www.youtube.com/watch?v=NnovZ1cTlMs) and [Part 2](https://www.youtube.com/watch?v=qGNWMcfWwPU), hosted on YouTube.

@youtube{NnovZ1cTlMs}
@youtube{qGNWMcfWwPU}

**warning**

These videos above are long-obsolete and contain inaccurate information. Be careful, since
solutions described in those videos are no longer supported and may even break your install.

If you are building your own libraries you can take the source files from our [Git
repository](https://github.com/opencv/opencv.git).

Building the OpenCV library from scratch requires a couple of tools installed beforehand:

-   An IDE of choice (preferably), or just a CC++ compiler that will actually make the binary files.
    Here we will use the [Microsoft Visual Studio](https://www.microsoft.com/visualstudio/en-us).
    However, you can use any other IDE that has a valid CC++ compiler.
-   [CMake](http://www.cmake.org/cmake/resources/software.html), which is a neat tool to make the project files (for your chosen IDE) from the OpenCV
    source files. It will also allow an easy configuration of the OpenCV build files, in order to
    make binary files that fits exactly to your needs.
-   Git to acquire the OpenCV source files. A good tool for this is [TortoiseGit](http://code.google.com/p/tortoisegit/wiki/Download). Alternatively,
    you can just download an archived version of the source files from our [page on
    Sourceforge](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/)

OpenCV may come in multiple flavors. There is a "core" section that will work on its own.
Nevertheless, there is a couple of tools, libraries made by 3rd parties that offer services of which
the OpenCV may take advantage. These will improve its capabilities in many ways. In order to use any
of them, you need to download and install them on your system.

-   The [Python libraries](http://www.python.org/downloads/) are required to build the *Python interface* of OpenCV. For now use the
    version `2.7.{x}`. This is also a must if you want to build the *OpenCV documentation*.
-   [Numpy](http://numpy.scipy.org/) is a scientific computing package for Python. Required for the *Python interface*.
-   [Intel Threading Building Blocks (*TBB*)](http://threadingbuildingblocks.org/file.php?fid=77) is used inside OpenCV for parallel code
    snippets. Using this will make sure that the OpenCV library will take advantage of all the cores
    you have in your system's CPU.
-   [Intel Integrated Performance Primitives (*IPP*)](http://software.intel.com/en-us/articles/intel-ipp/) may be used to improve the performance
    of color conversion, Haar training and DFT functions of the OpenCV library. Watch out, since
    this is not a free service.
-   OpenCV offers a somewhat fancier and more useful graphical user interface, than the default one
    by using the [Qt framework](http://qt.nokia.com/downloads). For a quick overview of what this has to offer, look into the
    documentations *highgui* module, under the *Qt New Functions* section. Version 4.6 or later of
    the framework is required.
-   [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) is a C++ template library for linear algebra.
-   The latest [CUDA Toolkit](http://developer.nvidia.com/cuda-downloads) will allow you to use the power lying inside your GPU. This will
    drastically improve performance for some algorithms (e.g the HOG descriptor). Getting more and
    more of our algorithms to work on the GPUs is a constant effort of the OpenCV team.
-   [OpenEXR](http://www.openexr.com/downloads.html) source files are required for the library to work with this high dynamic range (HDR)
    image file format.
-   The OpenNI Framework contains a set of open source APIs that provide support for natural interaction with devices via methods such as voice command recognition, hand gestures, and body
    motion tracking. Prebuilt binaries can be found [here](http://structure.io/openni). The source code of [OpenNI](https://github.com/OpenNI/OpenNI) and [OpenNI2](https://github.com/OpenNI/OpenNI2) are also available on Github.
-   [Doxygen](http://www.doxygen.nl) is a documentation generator and is the tool that will actually create the
    *OpenCV documentation*.

Now we will describe the steps to follow for a full build (using all the above frameworks, tools and
libraries). If you do not need the support for some of these, you can just freely skip this section.

### Building the library

-#  Make sure you have a working IDE with a valid compiler. In case of the Microsoft Visual Studio
    just install it and make sure it starts up.
-#  Install [CMake](http://www.cmake.org/cmake/resources/software.html). Simply follow the wizard, no need to add it to the path. The default install
    options are OK.
-#  Download and install an up-to-date version of msysgit from its [official
    site](http://code.google.com/p/msysgit/downloads/list). There is also the portable version,
    which you need only to unpack to get access to the console version of Git. Supposing that for
    some of us it could be quite enough.
-#  Install [TortoiseGit](http://code.google.com/p/tortoisegit/wiki/Download). Choose the 32 or 64 bit version according to the type of OS you work in.
    While installing, locate your msysgit (if it does not do that automatically). Follow the
    wizard -- the default options are OK for the most part.
-#  Choose a directory in your file system, where you will download the OpenCV libraries to. I
    recommend creating a new one that has short path and no special characters in it, for example
    `D:/OpenCV`. For this tutorial, I will suggest you do so. If you use your own path and know, what
    you are doing -- it is OK.
    -#  Clone the repository to the selected directory. After clicking *Clone* button, a window will
        appear where you can select from what repository you want to download source files
        (<https://github.com/opencv/opencv.git>) and to what directory (`D:/OpenCV`).
    -#  Push the OK button and be patient as the repository is quite a heavy download. It will take
        some time depending on your Internet connection.

-#  In this section, I will cover installing the 3rd party libraries.
    -#  Download the [Python libraries](http://www.python.org/downloads/) and install it with the default options. You will need a
        couple other python extensions. Luckily installing all these may be automated by a nice tool
        called [Setuptools](http://pypi.python.org/pypi/setuptools#downloads). Download and install
        again.

    -#  The easiest way to install Numpy is to just download its binaries from the [sourceforge page](http://sourceforge.net/projects/numpy/files/NumPy/).
        Make sure your download and install
        exactly the binary for your python version (so for version `2.7`).

    -#  For the [Intel Threading Building Blocks (*TBB*)](http://threadingbuildingblocks.org/file.php?fid=77)
        download the source files and extract
        it inside a directory on your system. For example let there be `D:/OpenCV/dep`. For installing
        the [Intel Integrated Performance Primitives (*IPP*)](http://software.intel.com/en-us/articles/intel-ipp/)
        the story is the same. For
        extracting the archives, I recommend using the [7-Zip](http://www.7-zip.org/) application.

        ![](images/IntelTBB.png)

    -#  In case of the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) library it is again a case of download and extract to the
        `D:/OpenCV/dep` directory.
    -#  Same as above with [OpenEXR](http://www.openexr.com/downloads.html).
    -#  For the [OpenNI Framework](http://www.openni.org/) you need to install both the [development
        build](http://www.openni.org/downloadfiles/opennimodules/openni-binaries/21-stable) and the
        [PrimeSensor
        Module](http://www.openni.org/downloadfiles/opennimodules/openni-compliant-hardware-binaries/32-stable).
    -#  For the CUDA you need again two modules: the latest [CUDA Toolkit](http://developer.nvidia.com/cuda-downloads) and the *CUDA Tools SDK*.
        Download and install both of them with a *complete* option by using the 32 or 64 bit setups
        according to your OS.
    -#  In case of the Qt framework you need to build yourself the binary files (unless you use the
        Microsoft Visual Studio 2008 with 32 bit compiler). To do this go to the [Qt
        Downloads](http://qt.nokia.com/downloads) page. Download the source files (not the
        installers!!!):

        ![](images/qtDownloadThisPackage.png)

        Extract it into a nice and short named directory like `D:/OpenCV/dep/qt/` . Then you need to
        build it. Start up a *Visual* *Studio* *Command* *Prompt* (*2010*) by using the start menu
        search (or navigate through the start menu
        All Programs --\> Microsoft Visual Studio 2010 --\> Visual Studio Tools --\> Visual Studio Command Prompt (2010)).

        ![](images/visualstudiocommandprompt.jpg)

        Now navigate to the extracted folder and enter inside it by using this console window. You
        should have a folder containing files like *Install*, *Make* and so on. Use the *dir* command
        to list files inside your current directory. Once arrived at this directory enter the
        following command:
        @code{.bash}
        configure.exe -release -no-webkit -no-phonon -no-phonon-backend -no-script -no-scripttools
                      -no-qt3support -no-multimedia -no-ltcg
        @endcode
        Completing this will take around 10-20 minutes. Then enter the next command that will take a
        lot longer (can easily take even more than a full hour):
        @code{.bash}
        nmake
        @endcode
        After this set the Qt environment variables using the following command on Windows 7:
        @code{.bash}
        setx -m QTDIR D:/OpenCV/dep/qt/qt-everywhere-opensource-src-4.7.3
        @endcode
        Also, add the built binary files path to the system path by using the [PathEditor](http://www.redfernplace.com/software-projects/patheditor/). In our
        case this is `D:/OpenCV/dep/qt/qt-everywhere-opensource-src-4.7.3/bin`.

        @note
        If you plan on doing Qt application development you can also install at this point the *Qt
        Visual Studio Add-in*. After this you can make and build Qt applications without using the *Qt
        Creator*. Everything is nicely integrated into Visual Studio.

-#  Now start the *CMake (cmake-gui)*. You may again enter it in the start menu search or get it
    from the All Programs --\> CMake 2.8 --\> CMake (cmake-gui). First, select the directory for the
    source files of the OpenCV library (1). Then, specify a directory where you will build the
    binary files for OpenCV (2).

    ![](images/CMakeSelectBin.jpg)

    Press the Configure button to specify the compiler (and *IDE*) you want to use. Note that in
    case you can choose between different compilers for making either 64 bit or 32 bit libraries.
    Select the one you use in your application development.

    ![](images/CMake_Configure_Windows.jpg)

    CMake will start out and based on your system variables will try to automatically locate as many
    packages as possible. You can modify the packages to use for the build in the WITH --\> WITH_X
    menu points (where *X* is the package abbreviation). Here are a list of current packages you can
    turn on or off:

    ![](images/CMakeBuildWithWindowsGUI.jpg)

    Select all the packages you want to use and press again the *Configure* button. For an easier
    overview of the build options make sure the *Grouped* option under the binary directory
    selection is turned on. For some of the packages CMake may not find all of the required files or
    directories. In case of these, CMake will throw an error in its output window (located at the
    bottom of the GUI) and set its field values to not found constants. For example:

    ![](images/CMakePackageNotFoundWindows.jpg)

    ![](images/CMakeOutputPackageNotFound.jpg)

    For these you need to manually set the queried directories or files path. After this press again
    the *Configure* button to see if the value entered by you was accepted or not. Do this until all
    entries are good and you cannot see errors in the field/value or the output part of the GUI. Now
    I want to emphasize an option that you will definitely love:
    ENABLE --\> ENABLE_SOLUTION_FOLDERS. OpenCV will create many-many projects and turning this
    option will make sure that they are categorized inside directories in the *Solution Explorer*.
    It is a must have feature, if you ask me.

    ![](images/CMakeBuildOptionsOpenCV.jpg)

    Furthermore, you need to select what part of OpenCV you want to build.

    -   *BUILD_DOCS* -\> It creates two projects for building the documentation of OpenCV (there
        will be a separate project for building the HTML and the PDF files). Note that these are not
        built together with the solution. You need to make an explicit build project command on
        these to do so.
    -   *BUILD_EXAMPLES* -\> OpenCV comes with many example applications from which you may learn
        most of the libraries capabilities. This will also come handy to easily try out if OpenCV is
        fully functional on your computer.
    -   *BUILD_PACKAGE* -\> Prior to version 2.3 with this you could build a project that will
        build an OpenCV installer. With this, you can easily install your OpenCV flavor on other
        systems. For the latest source files of OpenCV, it generates a new project that simply
        creates a zip archive with OpenCV sources.
    -   *BUILD_SHARED_LIBS* -\> With this you can control to build DLL files (when turned on) or
        static library files (\*.lib) otherwise.
    -   *BUILD_TESTS* -\> Each module of OpenCV has a test project assigned to it. Building these
        test projects is also a good way to try out, that the modules work just as expected on your
        system too.
    -   *BUILD_PERF_TESTS* -\> There are also performance tests for many OpenCV functions. If
        you are concerned about performance, build them and run.
    -   *BUILD_opencv_python* -\> Self-explanatory. Create the binaries to use OpenCV from the
        Python language.
    -   *BUILD_opencv_world* -\> Generate a single "opencv_world" binary (a shared or static library, depending on *BUILD_SHARED_LIBS*) including all the modules instead of a collection of separate binaries, one binary per module.

    Press again the *Configure* button and ensure no errors are reported. If this is the case, you
    can tell CMake to create the project files by pushing the *Generate* button. Go to the build
    directory and open the created **OpenCV** solution. Depending on just how much of the above
    options you have selected the solution may contain quite a lot of projects so be tolerant on the
    IDE at the startup. Now you need to build both the *Release* and the *Debug* binaries. Use the
    drop-down menu on your IDE to change to another of these after building for one of them.

    ![](images/ChangeBuildVisualStudio.jpg)

    In the end, you can observe the built binary files inside the bin directory:

    ![](images/OpenCVBuildResultWindows.jpg)

    For the documentation, you need to explicitly issue the build commands on the *doxygen* project for
    the HTML documentation. It will call *Doxygen* to do
    all the hard work. You can find the generated documentation inside the `build/doc/doxygen/html`.

    To collect the header and the binary files, that you will use during your own projects, into a
    separate directory (similarly to how the pre-built binaries ship) you need to explicitly build
    the *Install* project.

    ![](images/WindowsBuildInstall.png)

    This will create an *Install* directory inside the *Build* one collecting all the built binaries
    into a single place. Use this only after you built both the *Release* and *Debug* versions.

    To test your build just go into the `Build/bin/Debug` or `Build/bin/Release` directory and start
    a couple of applications like the *contours.exe*. If they run, you are done. Otherwise,
    something definitely went awfully wrong. In this case you should contact us at our [Q&A forum](https://forum.opencv.org/).
    If everything is okay, the *contours.exe* output should resemble the following image (if
    built with Qt support):

    ![](images/WindowsQtContoursOutput.png)

@note
   If you use the GPU module (CUDA libraries), make sure you also upgrade to the latest drivers of
    your GPU. Error messages containing invalid entries in (or cannot find) the nvcuda.dll are
    caused mostly by old video card drivers. For testing the GPU (if built) run the
    *performance_gpu.exe* sample application.

## Set the OpenCV environment variable and add it to the systems path {#tutorial_windows_install_path}

First, we set an environment variable to make our work easier. This will hold the build directory of
our OpenCV library that we use in our projects. Start up a command window and enter:
@code
    setx OpenCV_DIR D:\OpenCV\build\x64\vc14     (suggested for Visual Studio 2015 - 64 bit Windows)
    setx OpenCV_DIR D:\OpenCV\build\x86\vc14     (suggested for Visual Studio 2015 - 32 bit Windows)

    setx OpenCV_DIR D:\OpenCV\build\x64\vc15     (suggested for Visual Studio 2017 - 64 bit Windows)
    setx OpenCV_DIR D:\OpenCV\build\x86\vc15     (suggested for Visual Studio 2017 - 32 bit Windows)

    setx OpenCV_DIR D:\OpenCV\build\x64\vc16     (suggested for Visual Studio 2019 - 64 bit Windows)
    setx OpenCV_DIR D:\OpenCV\build\x86\vc16     (suggested for Visual Studio 2019 - 32 bit Windows)

    setx OpenCV_DIR D:\OpenCV\build\x64\vc17     (suggested for Visual Studio 2022 - 64 bit Windows)
    setx OpenCV_DIR D:\OpenCV\build\x86\vc17     (suggested for Visual Studio 2022 - 32 bit Windows)
@endcode
Here the directory is where you have your OpenCV binaries (*extracted* or *built*). You can have
different platform (e.g. x64 instead of x86) or compiler type, so substitute appropriate value.
Inside this, you should have two folders called *lib* and *bin*.

If you built static libraries then you are done. Otherwise, you need to add the *bin* folders path
to the systems path. This is because you will use the OpenCV library in form of *"Dynamic-link
libraries"* (also known as **DLL**). Inside these are stored all the algorithms and information the
OpenCV library contains. The operating system will load them only on demand, during runtime.
However, to do this the operating system needs to know where they are. The systems **PATH** contains
a list of folders where DLLs can be found. Add the OpenCV library path to this and the OS will know
where to look if he ever needs the OpenCV binaries. Otherwise, you will need to copy the used DLLs
right beside the applications executable file (*exe*) for the OS to find it, which is highly
unpleasant if you work on many projects. To do this start up again the [PathEditor](http://www.redfernplace.com/software-projects/patheditor/) and add the
following new entry (right click in the application to bring up the menu):
@code
    %OPENCV_DIR%\bin
@endcode

![](images/PathEditorOpenCVInsertNew.png)

![](images/PathEditorOpenCVSetPath.png)

Save it to the registry and you are done. If you ever change the location of your build directories
or want to try out your application with a different build, all you will need to do is to update the
OPENCV_DIR variable via the *setx* command inside a command window.

Now you can continue reading the tutorials with the @ref tutorial_windows_visual_studio_opencv section.
There you will find out how to use the OpenCV library in your own projects with the help of the
Microsoft Visual Studio IDE.
