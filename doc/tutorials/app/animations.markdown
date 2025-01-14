Handling Animated Image Files {#tutorial_animations}
===========================

@tableofcontents

|    |    |
| -: | :- |
| Original author | Suleyman Turkmen (with help of ChatGPT) |
| Compatibility | OpenCV >= 4.11 |

Goal
----
In this tutorial, you will learn how to:

- Use `cv::imreadanimation` to load frames from animated image files.
- Understand the structure and parameters of the `cv::Animation` structure.
- Display individual frames from an animation.
- Use `cv::imwriteanimation` to write `cv::Animation` to a file.

Source Code
-----------

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/imgcodecs/animations.cpp)

-   **Code at a glance:**
    @include samples/cpp/tutorial_code/imgcodecs/animations.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/imgcodecs/animations.py)

-   **Code at a glance:**
    @include samples/python/tutorial_code/imgcodecs/animations.py
@end_toggle

Explanation
-----------

## Initializing the Animation Structure

   Initialize a `cv::Animation` structure to hold the frames from the animated image file.

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp init_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py init_animation
@end_toggle

## Loading Frames

   Use `cv::imreadanimation` to load frames from the specified file. Here, we load all frames from an animated WebP image.

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp read_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py read_animation
@end_toggle

## Displaying Frames

   Each frame in the `animation.frames` vector can be displayed as a standalone image. This loop iterates through each frame, displaying it in a window with a short delay to simulate the animation.

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp show_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py show_animation
@end_toggle

## Saving Animation

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp write_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py write_animation
@end_toggle

## Summary

The `cv::imreadanimation` and `cv::imwriteanimation` functions make it easy to work with animated image files by loading frames into a `cv::Animation` structure, allowing frame-by-frame processing.
With these functions, you can load, process, and save frames from animated image files like GIF, AVIF, APNG, and WebP.
