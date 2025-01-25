Introduction to OpenCV.js and Tutorials {#tutorial_js_intro}
=======================================

OpenCV
------

OpenCV was created at Intel in 1999 by **Gary Bradski**. The first release came out in 2000. **Vadim Pisarevsky** joined Gary Bradski to manage Intel's Russian software OpenCV team. In 2005, OpenCV was used on Stanley, the vehicle that won the 2005 DARPA Grand Challenge. Later, its active development continued under the support of Willow Garage, with Gary Bradski and Vadim Pisarevsky leading the project. OpenCV now supports a multitude of algorithms related to Computer Vision and Machine Learning and is expanding day by day.

OpenCV supports a wide variety of programming languages such as C++, Python, and Java, and is available on different platforms including Windows, Linux, OS X, Android, and iOS. Interfaces for high-speed GPU operations based on CUDA and OpenCL are also under active development. OpenCV.js brings OpenCV to the open web platform and makes it available to JavaScript programmers.

OpenCV.js: OpenCV for the JavaScript programmer
-------------

The web is the most ubiquitous open computing platform. With HTML5 standards implemented in every browser, web applications can render online video with HTML5 video tags, capture webcam video via WebRTC API, and access each pixel of a video frame via the canvas API. With the abundance of available multimedia content, web developers need a wide array of image and vision processing algorithms in JavaScript to build innovative applications. This requirement is even more essential for emerging applications on the web, such as Web Virtual Reality (WebVR) and Augmented Reality (WebAR). All of these use cases demand efficient implementations of computation-intensive vision kernels on the web.

[Emscripten](https://emscripten.org/) is an LLVM-to-JavaScript compiler. It takes LLVM bitcode—which can be generated from C/C++ using clang—and compiles it into asm.js or WebAssembly that can execute directly inside web browsers. Asm.js is a highly optimizable, low-level subset of JavaScript. It enables ahead-of-time compilation and optimization in JavaScript engines, providing near-to-native execution speed. WebAssembly is a new portable, size- and load-time-efficient binary format suitable for compilation to the web. It aims to execute at native speed and is currently being designed as an open standard by W3C.

OpenCV.js is a JavaScript binding for a selected subset of OpenCV functions for the web platform. It allows emerging web applications with multimedia processing to benefit from the wide variety of vision functions available in OpenCV. OpenCV.js leverages Emscripten to compile OpenCV functions into asm.js or WebAssembly targets and provides JavaScript APIs for web applications to access them. Future versions of the library will take advantage of acceleration APIs available on the web, such as SIMD and multi-threaded execution.

OpenCV.js was initially created in the Parallel Architectures and Systems Group at the University of California Irvine (UCI) as a research project funded by Intel Corporation. It was further improved and integrated into the OpenCV project as part of the Google Summer of Code 2017 program.

OpenCV.js Tutorials
-----------------------

OpenCV introduces a new set of tutorials to guide you through various functions available in OpenCV.js. **This guide is mainly focused on OpenCV 3.x versions.**

The purpose of OpenCV.js tutorials is to:
1. Help with the adaptability of OpenCV in web development.
2. Assist the web community, developers, and computer vision researchers in interactively accessing a variety of web-based OpenCV examples to understand specific vision algorithms.

Because OpenCV.js runs directly inside browsers, the tutorial web pages are intuitive and interactive. For example, using the WebRTC API and evaluating JavaScript code allows developers to change parameters of CV functions and perform live CV coding on web pages to see results in real time.

Prior knowledge of JavaScript and web application development is recommended to understand this guide.

Contributors
------------

Below is the list of contributors to OpenCV.js bindings and tutorials:

- **Sajjad Taheri**: Architect of the initial version and GSoC mentor, University of California, Irvine.
- **Congxiang Pan**: GSoC student, Shanghai Jiao Tong University.
- **Gang Song**: GSoC student, Shanghai Jiao Tong University.
- **Wenyao Gan**: Student intern, Shanghai Jiao Tong University.
- **Mohammad Reza Haghighat**: Project initiator & sponsor, Intel Corporation.
- **Ningxin Hu**: Students' supervisor, Intel Corporation.
