/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HIGHGUI_HPP
#define OPENCV_HIGHGUI_HPP

#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_IMGCODECS
#include "opencv2/imgcodecs.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOIO
#include "opencv2/videoio.hpp"
#endif

/**
@defgroup highgui High-level GUI

While OpenCV was designed for use in full-scale applications and can be used within functionally
rich UI frameworks (such as Qt\*, WinForms\*, or Cocoa\*) or without any UI at all, sometimes there
it is required to try functionality quickly and visualize the results. This is what the HighGUI
module has been designed for.

It provides easy interface to:

-   Create and manipulate windows that can display images and "remember" their content (no need to
    handle repaint events from OS).
-   Add trackbars to the windows, handle simple mouse events as well as keyboard commands.

@{
    @defgroup highgui_window_flags Flags related creating and manipulating HighGUI windows and mouse events
    @defgroup highgui_opengl OpenGL support
    @defgroup highgui_qt Qt New Functions

    ![image](pics/qtgui.png)

    This figure explains new functionality implemented with Qt\* GUI. The new GUI provides a statusbar,
    a toolbar, and a control panel. The control panel can have trackbars and buttonbars attached to it.
    If you cannot see the control panel, press Ctrl+P or right-click any Qt window and select **Display
    properties window**.

    -   To attach a trackbar, the window name parameter must be NULL.

    -   To attach a buttonbar, a button must be created. If the last bar attached to the control panel
        is a buttonbar, the new button is added to the right of the last button. If the last bar
        attached to the control panel is a trackbar, or the control panel is empty, a new buttonbar is
        created. Then, a new button is attached to it.

    See below the example used to generate the figure:

    @include highgui_qt.cpp

    @defgroup highgui_winrt WinRT support

    This figure explains new functionality implemented with WinRT GUI. The new GUI provides an Image control,
    and a slider panel. Slider panel holds trackbars attached to it.

    Sliders are attached below the image control. Every new slider is added below the previous one.

    See below the example used to generate the figure:
    @code
    void sample_app::MainPage::ShowWindow()
    {
        static cv::String windowName("sample");
        cv::winrt_initContainer(this->cvContainer);
        cv::namedWindow(windowName); // not required

        cv::Mat image = cv::imread("Assets/sample.jpg");
        cv::Mat converted = cv::Mat(image.rows, image.cols, CV_8UC4);
        cv::cvtColor(image, converted, COLOR_BGR2BGRA);
        cv::imshow(windowName, converted); // this will create window if it hasn't been created before

        int state = 42;
        cv::TrackbarCallback callback = [](int pos, void* userdata)
        {
            if (pos == 0) {
                cv::destroyWindow(windowName);
            }
        };
        cv::TrackbarCallback callbackTwin = [](int pos, void* userdata)
        {
            if (pos >= 70) {
                cv::destroyAllWindows();
            }
        };
        cv::createTrackbar("Sample trackbar", windowName, &state, 100, callback);
        cv::createTrackbar("Twin brother", windowName, &state, 100, callbackTwin);
    }
    @endcode
@}
*/

///////////////////////// graphical user interface //////////////////////////
namespace cv
{

//! @addtogroup highgui
//! @{

//! @addtogroup highgui_window_flags
//! @{

//! Flags for cv::namedWindow
enum WindowFlags {
       WINDOW_NORMAL     = 0x00000000, //!< the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size.
       WINDOW_AUTOSIZE   = 0x00000001, //!< the user cannot resize the window, the size is constrainted by the image displayed.
       WINDOW_OPENGL     = 0x00001000, //!< window with opengl support.

       WINDOW_FULLSCREEN = 1,          //!< change the window to fullscreen.
       WINDOW_FREERATIO  = 0x00000100, //!< the image expends as much as it can (no ratio constraint).
       WINDOW_KEEPRATIO  = 0x00000000, //!< the ratio of the image is respected.
       WINDOW_GUI_EXPANDED=0x00000000, //!< status bar and tool bar
       WINDOW_GUI_NORMAL = 0x00000010, //!< old fashious way
    };

//! Flags for cv::setWindowProperty / cv::getWindowProperty
enum WindowPropertyFlags {
       WND_PROP_FULLSCREEN   = 0, //!< fullscreen property    (can be WINDOW_NORMAL or WINDOW_FULLSCREEN).
       WND_PROP_AUTOSIZE     = 1, //!< autosize property      (can be WINDOW_NORMAL or WINDOW_AUTOSIZE).
       WND_PROP_ASPECT_RATIO = 2, //!< window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO).
       WND_PROP_OPENGL       = 3, //!< opengl support.
       WND_PROP_VISIBLE      = 4, //!< checks whether the window exists and is visible
       WND_PROP_TOPMOST      = 5, //!< property to toggle normal window being topmost or not
       WND_PROP_VSYNC        = 6  //!< enable or disable VSYNC (in OpenGL mode)
     };

//! Mouse Events see cv::MouseCallback
enum MouseEventTypes {
       EVENT_MOUSEMOVE      = 0, //!< indicates that the mouse pointer has moved over the window.
       EVENT_LBUTTONDOWN    = 1, //!< indicates that the left mouse button is pressed.
       EVENT_RBUTTONDOWN    = 2, //!< indicates that the right mouse button is pressed.
       EVENT_MBUTTONDOWN    = 3, //!< indicates that the middle mouse button is pressed.
       EVENT_LBUTTONUP      = 4, //!< indicates that left mouse button is released.
       EVENT_RBUTTONUP      = 5, //!< indicates that right mouse button is released.
       EVENT_MBUTTONUP      = 6, //!< indicates that middle mouse button is released.
       EVENT_LBUTTONDBLCLK  = 7, //!< indicates that left mouse button is double clicked.
       EVENT_RBUTTONDBLCLK  = 8, //!< indicates that right mouse button is double clicked.
       EVENT_MBUTTONDBLCLK  = 9, //!< indicates that middle mouse button is double clicked.
       EVENT_MOUSEWHEEL     = 10,//!< positive and negative values mean forward and backward scrolling, respectively.
       EVENT_MOUSEHWHEEL    = 11 //!< positive and negative values mean right and left scrolling, respectively.
     };

//! Mouse Event Flags see cv::MouseCallback
enum MouseEventFlags {
       EVENT_FLAG_LBUTTON   = 1, //!< indicates that the left mouse button is down.
       EVENT_FLAG_RBUTTON   = 2, //!< indicates that the right mouse button is down.
       EVENT_FLAG_MBUTTON   = 4, //!< indicates that the middle mouse button is down.
       EVENT_FLAG_CTRLKEY   = 8, //!< indicates that CTRL Key is pressed.
       EVENT_FLAG_SHIFTKEY  = 16,//!< indicates that SHIFT Key is pressed.
       EVENT_FLAG_ALTKEY    = 32 //!< indicates that ALT Key is pressed.
     };

//! @} highgui_window_flags

//! @addtogroup highgui_qt
//! @{

//! Qt font weight
enum QtFontWeights {
        QT_FONT_LIGHT           = 25, //!< Weight of 25
        QT_FONT_NORMAL          = 50, //!< Weight of 50
        QT_FONT_DEMIBOLD        = 63, //!< Weight of 63
        QT_FONT_BOLD            = 75, //!< Weight of 75
        QT_FONT_BLACK           = 87  //!< Weight of 87
     };

//! Qt font style
enum QtFontStyles {
        QT_STYLE_NORMAL         = 0, //!< Normal font.
        QT_STYLE_ITALIC         = 1, //!< Italic font.
        QT_STYLE_OBLIQUE        = 2  //!< Oblique font.
     };

//! Qt "button" type
enum QtButtonTypes {
       QT_PUSH_BUTTON   = 0,    //!< Push button.
       QT_CHECKBOX      = 1,    //!< Checkbox button.
       QT_RADIOBOX      = 2,    //!< Radiobox button.
       QT_NEW_BUTTONBAR = 1024  //!< Button should create a new buttonbar
     };

//! @} highgui_qt

/** @brief Callback function for mouse events. see cv::setMouseCallback
@param event one of the cv::MouseEventTypes constants.
@param x The x-coordinate of the mouse event.
@param y The y-coordinate of the mouse event.
@param flags one of the cv::MouseEventFlags constants.
@param userdata The optional parameter.
 */
typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);

/** @brief Callback function for Trackbar see cv::createTrackbar
@param pos current position of the specified trackbar.
@param userdata The optional parameter.
 */
typedef void (*TrackbarCallback)(int pos, void* userdata);

/** @brief Callback function defined to be called every frame. See cv::setOpenGlDrawCallback
@param userdata The optional parameter.
 */
typedef void (*OpenGlDrawCallback)(void* userdata);

/** @brief Callback function for a button created by cv::createButton
@param state current state of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button.
@param userdata The optional parameter.
 */
typedef void (*ButtonCallback)(int state, void* userdata);

/** @brief Creates a window.

The function namedWindow creates a window that can be used as a placeholder for images and
trackbars. Created windows are referred to by their names.

If a window with the same name already exists, the function does nothing.

You can call cv::destroyWindow or cv::destroyAllWindows to close the window and de-allocate any associated
memory usage. For a simple program, you do not really have to call these functions because all the
resources and windows of the application are closed automatically by the operating system upon exit.

@note Qt backend supports additional flags:
 -   **WINDOW_NORMAL or WINDOW_AUTOSIZE:** WINDOW_NORMAL enables you to resize the
     window, whereas WINDOW_AUTOSIZE adjusts automatically the window size to fit the
     displayed image (see imshow ), and you cannot change the window size manually.
 -   **WINDOW_FREERATIO or WINDOW_KEEPRATIO:** WINDOW_FREERATIO adjusts the image
     with no respect to its ratio, whereas WINDOW_KEEPRATIO keeps the image ratio.
 -   **WINDOW_GUI_NORMAL or WINDOW_GUI_EXPANDED:** WINDOW_GUI_NORMAL is the old way to draw the window
     without statusbar and toolbar, whereas WINDOW_GUI_EXPANDED is a new enhanced GUI.
By default, flags == WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED

@param winname Name of the window in the window caption that may be used as a window identifier.
@param flags Flags of the window. The supported flags are: (cv::WindowFlags)
 */
CV_EXPORTS_W void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);

/** @brief Destroys the specified window.

The function destroyWindow destroys the window with the given name.

@param winname Name of the window to be destroyed.
 */
CV_EXPORTS_W void destroyWindow(const String& winname);

/** @brief Destroys all of the HighGUI windows.

The function destroyAllWindows destroys all of the opened HighGUI windows.
 */
CV_EXPORTS_W void destroyAllWindows();


/** @brief HighGUI backend used.

The function returns HighGUI backend name used: could be COCOA, GTK2/3, QT, WAYLAND or WIN32.
Returns empty string if there is no available UI backend.
 */
CV_EXPORTS_W const std::string currentUIFramework();


CV_EXPORTS_W int startWindowThread();

/** @brief Similar to #waitKey, but returns full key code.

@note Key code is implementation specific and depends on used backend: QT/GTK/Win32/etc

*/
CV_EXPORTS_W int waitKeyEx(int delay = 0);

/** @brief Waits for a pressed key.

The function waitKey waits for a key event infinitely (when \f$\texttt{delay}\leq 0\f$ ) or for delay
milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the
function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is
running on your computer at that time. It returns the code of the pressed key or -1 if no key was
pressed before the specified time had elapsed. To check for a key press but not wait for it, use
#pollKey.

@note The functions #waitKey and #pollKey are the only methods in HighGUI that can fetch and handle
GUI events, so one of them needs to be called periodically for normal event processing unless
HighGUI is used within an environment that takes care of event processing.

@note The function only works if there is at least one HighGUI window created and the window is
active. If there are several HighGUI windows, any of them can be active.

@param delay Delay in milliseconds. 0 is the special value that means "forever".
 */
CV_EXPORTS_W int waitKey(int delay = 0);

/** @brief Polls for a pressed key.

The function pollKey polls for a key event without waiting. It returns the code of the pressed key
or -1 if no key was pressed since the last invocation. To wait until a key was pressed, use #waitKey.

@note The functions #waitKey and #pollKey are the only methods in HighGUI that can fetch and handle
GUI events, so one of them needs to be called periodically for normal event processing unless
HighGUI is used within an environment that takes care of event processing.

@note The function only works if there is at least one HighGUI window created and the window is
active. If there are several HighGUI windows, any of them can be active.
 */
CV_EXPORTS_W int pollKey();

/** @brief Displays an image in the specified window.

The function imshow displays an image in the specified window. If the window was created with the
cv::WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution.
Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:

-   If the image is 8-bit unsigned, it is displayed as is.
-   If the image is 16-bit unsigned, the pixels are divided by 256. That is, the
    value range [0,255\*256] is mapped to [0,255].
-   If the image is 32-bit or 64-bit floating-point, the pixel values are multiplied by 255. That is, the
    value range [0,1] is mapped to [0,255].
-   32-bit integer images are not processed anymore due to ambiguouty of required transform.
    Convert to 8-bit unsigned matrix using a custom preprocessing specific to image's context.

If window was created with OpenGL support, cv::imshow also support ogl::Buffer , ogl::Texture2D and
cuda::GpuMat as input.

If the window was not created before this function, it is assumed creating a window with cv::WINDOW_AUTOSIZE.

If you need to show an image that is bigger than the screen resolution, you will need to call namedWindow("", WINDOW_NORMAL) before the imshow.

@note This function should be followed by a call to cv::waitKey or cv::pollKey to perform GUI
housekeeping tasks that are necessary to actually show the given image and make the window respond
to mouse and keyboard events. Otherwise, it won't display the image and the window might lock up.
For example, **waitKey(0)** will display the window infinitely until any keypress (it is suitable
for image display). **waitKey(25)** will display a frame and wait approximately 25 ms for a key
press (suitable for displaying a video frame-by-frame). To remove the window, use cv::destroyWindow.

@note [__Windows Backend Only__] Pressing Ctrl+C will copy the image to the clipboard. Pressing Ctrl+S will show a dialog to save the image.
@note [__Wayland Backend Only__] Supoorting format is extended.
-   If the image is 8-bit signed, the pixels are biased by 128. That is, the
    value range [-128,127] is mapped to [0,255].
-   If the image is 16-bit signed, the pixels are divided by 256 and biased by 128. That is, the
    value range [-32768,32767] is mapped to [0,255].

@param winname Name of the window.
@param mat Image to be shown.
 */
CV_EXPORTS_W void imshow(const String& winname, InputArray mat);

/** @brief Resizes the window to the specified size

@note The specified window size is for the image area. Toolbars are not counted.
Only windows created without cv::WINDOW_AUTOSIZE flag can be resized.

@param winname Window name.
@param width The new window width.
@param height The new window height.
 */
CV_EXPORTS_W void resizeWindow(const String& winname, int width, int height);

/** @overload
@param winname Window name.
@param size The new window size.
*/
CV_EXPORTS_W void resizeWindow(const String& winname, const cv::Size& size);

/** @brief Moves the window to the specified position

@param winname Name of the window.
@param x The new x-coordinate of the window.
@param y The new y-coordinate of the window.

@note [__Wayland Backend Only__] This function is not supported by the Wayland protocol limitation.
 */
CV_EXPORTS_W void moveWindow(const String& winname, int x, int y);

/** @brief Changes parameters of a window dynamically.

The function setWindowProperty enables changing properties of a window.

@param winname Name of the window.
@param prop_id Window property to edit. The supported operation flags are: (cv::WindowPropertyFlags)
@param prop_value New value of the window property. The supported flags are: (cv::WindowFlags)

@note [__Wayland Backend Only__] This function is not supported.
 */
CV_EXPORTS_W void setWindowProperty(const String& winname, int prop_id, double prop_value);

/** @brief Updates window title
@param winname Name of the window.
@param title New title.
*/
CV_EXPORTS_W void setWindowTitle(const String& winname, const String& title);

/** @brief Provides parameters of a window.

The function getWindowProperty returns properties of a window.

@param winname Name of the window.
@param prop_id Window property to retrieve. The following operation flags are available: (cv::WindowPropertyFlags)

@sa setWindowProperty

@note [__Wayland Backend Only__] This function is not supported.
 */
CV_EXPORTS_W double getWindowProperty(const String& winname, int prop_id);

/** @brief Provides rectangle of image in the window.

The function getWindowImageRect returns the client screen coordinates, width and height of the image rendering area.

@param winname Name of the window.

@sa resizeWindow moveWindow

@note [__Wayland Backend Only__] This function is not supported by the Wayland protocol limitation.
 */
CV_EXPORTS_W Rect getWindowImageRect(const String& winname);

/** @example samples/cpp/snippets/create_mask.cpp
This program demonstrates using mouse events and how to make and use a mask image (black and white) .
*/

/** @brief Sets mouse handler for the specified window

@param winname Name of the window.
@param onMouse Callback function for mouse events. See OpenCV samples on how to specify and use the callback.
@param userdata The optional parameter passed to the callback.
 */
CV_EXPORTS void setMouseCallback(const String& winname, MouseCallback onMouse, void* userdata = 0);

/** @brief Gets the mouse-wheel motion delta, when handling mouse-wheel events cv::EVENT_MOUSEWHEEL and
cv::EVENT_MOUSEHWHEEL.

For regular mice with a scroll-wheel, delta will be a multiple of 120. The value 120 corresponds to
a one notch rotation of the wheel or the threshold for action to be taken and one such action should
occur for each delta. Some high-precision mice with higher-resolution freely-rotating wheels may
generate smaller values.

For cv::EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling,
respectively. For cv::EVENT_MOUSEHWHEEL, where available, positive and negative values mean right and
left scrolling, respectively.

@note Mouse-wheel events are currently supported only on Windows and Cocoa.

@param flags The mouse callback flags parameter.
 */
CV_EXPORTS int getMouseWheelDelta(int flags);

/** @brief Allows users to select a ROI on the given image.

The function creates a window and allows users to select a ROI using the mouse.
Controls: use `space` or `enter` to finish selection, use key `c` to cancel selection (function will return the zero cv::Rect).

@param windowName name of the window where selection process will be shown.
@param img image to select a ROI.
@param showCrosshair if true crosshair of selection rectangle will be shown.
@param fromCenter if true center of selection will match initial mouse position. In opposite case a corner of
selection rectangle will correspont to the initial mouse position.
@param printNotice if true a notice to select ROI or cancel selection will be printed in console.
@return selected ROI or empty rect if selection canceled.

@note The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).
After finish of work an empty callback will be set for the used window.
 */
CV_EXPORTS_W Rect selectROI(const String& windowName, InputArray img, bool showCrosshair = true, bool fromCenter = false, bool printNotice = true);

/** @overload
 */
CV_EXPORTS_W Rect selectROI(InputArray img, bool showCrosshair = true, bool fromCenter = false, bool printNotice = true);

/** @brief Allows users to select multiple ROIs on the given image.

The function creates a window and allows users to select multiple ROIs using the mouse.
Controls: use `space` or `enter` to finish current selection and start a new one,
use `esc` to terminate multiple ROI selection process.

@param windowName name of the window where selection process will be shown.
@param img image to select a ROI.
@param boundingBoxes selected ROIs.
@param showCrosshair if true crosshair of selection rectangle will be shown.
@param fromCenter if true center of selection will match initial mouse position. In opposite case a corner of
selection rectangle will correspont to the initial mouse position.
@param printNotice if true a notice to select ROI or cancel selection will be printed in console.

@note The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).
After finish of work an empty callback will be set for the used window.
 */
CV_EXPORTS_W void selectROIs(const String& windowName, InputArray img,
                             CV_OUT std::vector<Rect>& boundingBoxes, bool showCrosshair = true, bool fromCenter = false, bool printNotice = true);

/** @brief Creates a trackbar and attaches it to the specified window.

The function createTrackbar creates a trackbar (a slider or range control) with the specified name
and range, assigns a variable value to be a position synchronized with the trackbar and specifies
the callback function onChange to be called on the trackbar position change. The created trackbar is
displayed in the specified window winname.

@note [__Qt Backend Only__] winname can be empty if the trackbar should be attached to the
control panel.

Clicking the label of each trackbar enables editing the trackbar values manually.

@param trackbarname Name of the created trackbar.
@param winname Name of the window that will contain the trackbar.
@param value Pointer to the integer value that will be changed by the trackbar.
Pass `nullptr` if the value pointer is not used. In this case, manually handle
the trackbar position in the callback function.
@param count Maximum position of the trackbar.
@param onChange Pointer to the function to be called every time the slider changes position.
This function should have the prototype void Foo(int, void\*);, where the first parameter is
the trackbar position, and the second parameter is the user data (see the next parameter).
If the callback is a nullptr, no callbacks are called, but the trackbar's value will still be
updated automatically.
@param userdata Optional user data that is passed to the callback.
@note If the `value` pointer is `nullptr`, the trackbar position must be manually managed.
Call the callback function manually with the desired initial value to avoid runtime warnings.
@see \ref tutorial_trackbar
 */
CV_EXPORTS int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

/** @brief Returns the trackbar position.

The function returns the current position of the specified trackbar.

@note [__Qt Backend Only__] winname can be empty if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of the trackbar.
 */
CV_EXPORTS_W int getTrackbarPos(const String& trackbarname, const String& winname);

/** @brief Sets the trackbar position.

The function sets the position of the specified trackbar in the specified window.

@note [__Qt Backend Only__] winname can be empty if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param pos New position.
 */
CV_EXPORTS_W void setTrackbarPos(const String& trackbarname, const String& winname, int pos);

/** @brief Sets the trackbar maximum position.

The function sets the maximum position of the specified trackbar in the specified window.

@note [__Qt Backend Only__] winname can be empty if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param maxval New maximum position.
 */
CV_EXPORTS_W void setTrackbarMax(const String& trackbarname, const String& winname, int maxval);

/** @brief Sets the trackbar minimum position.

The function sets the minimum position of the specified trackbar in the specified window.

@note [__Qt Backend Only__] winname can be empty if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param minval New minimum position.
 */
CV_EXPORTS_W void setTrackbarMin(const String& trackbarname, const String& winname, int minval);

//! @addtogroup highgui_opengl OpenGL support
//! @{

/** @brief Displays OpenGL 2D texture in the specified window.

@param winname Name of the window.
@param tex OpenGL 2D texture data.
 */
CV_EXPORTS void imshow(const String& winname, const ogl::Texture2D& tex);

/** @brief Sets a callback function to be called to draw on top of displayed image.

The function setOpenGlDrawCallback can be used to draw 3D data on the window. See the example of
callback function below:
@code
    void on_opengl(void* param)
    {
        glLoadIdentity();

        glTranslated(0.0, 0.0, -1.0);

        glRotatef( 55, 1, 0, 0 );
        glRotatef( 45, 0, 1, 0 );
        glRotatef( 0, 0, 0, 1 );

        static const int coords[6][4][3] = {
            { { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
            { { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
            { { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
            { { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
            { { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
            { { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
        };

        for (int i = 0; i < 6; ++i) {
                    glColor3ub( i*20, 100+i*10, i*42 );
                    glBegin(GL_QUADS);
                    for (int j = 0; j < 4; ++j) {
                            glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
                    }
                    glEnd();
        }
    }
@endcode

@param winname Name of the window.
@param onOpenGlDraw Pointer to the function to be called every frame. This function should be
prototyped as void Foo(void\*) .
@param userdata Pointer passed to the callback function.(__Optional__)
 */
CV_EXPORTS void setOpenGlDrawCallback(const String& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0);

/** @brief Sets the specified window as current OpenGL context.

@param winname Name of the window.
 */
CV_EXPORTS void setOpenGlContext(const String& winname);

/** @brief Force window to redraw its context and call draw callback ( See cv::setOpenGlDrawCallback ).

@param winname Name of the window.
 */
CV_EXPORTS void updateWindow(const String& winname);

//! @} highgui_opengl

//! @addtogroup highgui_qt
//! @{

/** @brief QtFont available only for Qt. See cv::fontQt
 */
struct QtFont
{
    const char* nameFont;  //!< Name of the font
    Scalar      color;     //!< Color of the font. Scalar(blue_component, green_component, red_component[, alpha_component])
    int         font_face; //!< See cv::QtFontStyles
    const int*  ascii;     //!< font data and metrics
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;     //!< slope coefficient: 0 - normal, >0 - italic
    int         thickness; //!< See cv::QtFontWeights
    float       dx;        //!< horizontal interval between letters
    int         line_type; //!< PointSize
};

/** @brief Creates the font to draw a text on an image.

The function fontQt creates a cv::QtFont object. This cv::QtFont is not compatible with putText .

A basic usage of this function is the following: :
@code
    QtFont font = fontQt("Times");
    addText( img1, "Hello World !", Point(50,50), font);
@endcode

@param nameFont Name of the font. The name should match the name of a system font (such as
*Times*). If the font is not found, a default one is used.
@param pointSize Size of the font. If not specified, equal zero or negative, the point size of the
font is set to a system-dependent default value. Generally, this is 12 points.
@param color Color of the font in BGRA where A = 255 is fully transparent. Use the macro CV_RGB
for simplicity.
@param weight Font weight. Available operation flags are : cv::QtFontWeights You can also specify a positive integer for better control.
@param style Font style. Available operation flags are : cv::QtFontStyles
@param spacing Spacing between characters. It can be negative or positive.
 */
CV_EXPORTS QtFont fontQt(const String& nameFont, int pointSize = -1,
                         Scalar color = Scalar::all(0), int weight = QT_FONT_NORMAL,
                         int style = QT_STYLE_NORMAL, int spacing = 0);

/** @brief Draws a text on the image.

The function addText draws *text* on the image *img* using a specific font *font* (see example cv::fontQt
)

@param img 8-bit 3-channel image where the text should be drawn.
@param text Text to write on an image.
@param org Point(x,y) where the text should start on an image.
@param font Font to use to draw a text.
 */
CV_EXPORTS void addText( const Mat& img, const String& text, Point org, const QtFont& font);

/** @brief Draws a text on the image.

@param img 8-bit 3-channel image where the text should be drawn.
@param text Text to write on an image.
@param org Point(x,y) where the text should start on an image.
@param nameFont Name of the font. The name should match the name of a system font (such as
*Times*). If the font is not found, a default one is used.
@param pointSize Size of the font. If not specified, equal zero or negative, the point size of the
font is set to a system-dependent default value. Generally, this is 12 points.
@param color Color of the font in BGRA where A = 255 is fully transparent.
@param weight Font weight. Available operation flags are : cv::QtFontWeights You can also specify a positive integer for better control.
@param style Font style. Available operation flags are : cv::QtFontStyles
@param spacing Spacing between characters. It can be negative or positive.
 */
CV_EXPORTS_W void addText(const Mat& img, const String& text, Point org, const String& nameFont, int pointSize = -1, Scalar color = Scalar::all(0),
        int weight = QT_FONT_NORMAL, int style = QT_STYLE_NORMAL, int spacing = 0);

/** @brief Displays a text on a window image as an overlay for a specified duration.

The function displayOverlay displays useful information/tips on top of the window for a certain
amount of time *delayms*. The function does not modify the image, displayed in the window, that is,
after the specified delay the original content of the window is restored.

@param winname Name of the window.
@param text Overlay text to write on a window image.
@param delayms The period (in milliseconds), during which the overlay text is displayed. If this
function is called before the previous overlay text timed out, the timer is restarted and the text
is updated. If this value is zero, the text never disappears.
 */
CV_EXPORTS_W void displayOverlay(const String& winname, const String& text, int delayms = 0);

/** @brief Displays a text on the window statusbar during the specified period of time.

The function displayStatusBar displays useful information/tips on top of the window for a certain
amount of time *delayms* . This information is displayed on the window statusbar (the window must be
created with the cv::WINDOW_GUI_EXPANDED flags).

@param winname Name of the window.
@param text Text to write on the window statusbar.
@param delayms Duration (in milliseconds) to display the text. If this function is called before
the previous text timed out, the timer is restarted and the text is updated. If this value is
zero, the text never disappears.
 */
CV_EXPORTS_W void displayStatusBar(const String& winname, const String& text, int delayms = 0);

/** @brief Saves parameters of the specified window.

The function saveWindowParameters saves size, location, flags, trackbars value, zoom and panning
location of the window windowName.

@param windowName Name of the window.
 */
CV_EXPORTS void saveWindowParameters(const String& windowName);

/** @brief Loads parameters of the specified window.

The function loadWindowParameters loads size, location, flags, trackbars value, zoom and panning
location of the window windowName.

@param windowName Name of the window.
 */
CV_EXPORTS void loadWindowParameters(const String& windowName);

CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);

CV_EXPORTS  void stopLoop();

/** @brief Attaches a button to the control panel.

The function createButton attaches a button to the control panel. Each button is added to a
buttonbar to the right of the last button. A new buttonbar is created if nothing was attached to the
control panel before, or if the last element attached to the control panel was a trackbar or if the
QT_NEW_BUTTONBAR flag is added to the type.

See below various examples of the cv::createButton function call: :
@code
    createButton("",callbackButton);//create a push button "button 0", that will call callbackButton.
    createButton("button2",callbackButton,NULL,QT_CHECKBOX,0);
    createButton("button3",callbackButton,&value);
    createButton("button5",callbackButton1,NULL,QT_RADIOBOX);
    createButton("button6",callbackButton2,NULL,QT_PUSH_BUTTON,1);
    createButton("button6",callbackButton2,NULL,QT_PUSH_BUTTON|QT_NEW_BUTTONBAR);// create a push button in a new row
@endcode

@param  bar_name Name of the button.
@param on_change Pointer to the function to be called every time the button changes its state.
This function should be prototyped as void Foo(int state,\*void); . *state* is the current state
of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button.
@param userdata Pointer passed to the callback function.
@param type Optional type of the button. Available types are: (cv::QtButtonTypes)
@param initial_button_state Default state of the button. Use for checkbox and radiobox. Its
value could be 0 or 1. (__Optional__)
*/
CV_EXPORTS int createButton( const String& bar_name, ButtonCallback on_change,
                             void* userdata = 0, int type = QT_PUSH_BUTTON,
                             bool initial_button_state = false);

//! @} highgui_qt

//! @} highgui

} // cv

#endif
