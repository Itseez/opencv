// //////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

if (typeof Module.FS === 'undefined' && typeof FS !== 'undefined') {
    Module.FS = FS;
}

if (typeof cv === 'undefined') {
    var cv = Module;
}

Module['imread'] = function(imageSource) {
    var img = null;
    if (typeof imageSource === 'string') {
        img = document.getElementById(imageSource);
    } else {
        img = imageSource;
    }
    var canvas = null;
    var ctx = null;
    if (img instanceof HTMLImageElement) {
        canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, img.width, img.height);
    } else if (img instanceof HTMLCanvasElement || img instanceof OffscreenCanvas) {
        canvas = img;
        ctx = canvas.getContext('2d');
    } else {
        throw new Error('Please input the valid canvas or img id.');
    }

    var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return cv.matFromImageData(imgData);
};

Module['imshow'] = function(canvasSource, mat) {
    var canvas = null;
    if (typeof canvasSource === 'string') {
        canvas = document.getElementById(canvasSource);
    } else {
        canvas = canvasSource;
    }
    if (!(canvas instanceof HTMLCanvasElement)) {
        throw new Error('Please input the valid canvas element or id.');
    }
    if (!(mat instanceof cv.Mat)) {
        throw new Error('Please input the valid cv.Mat instance.');
    }

    // convert the mat type to cv.CV_8U
    var img = new cv.Mat();
    var depth = mat.type()%8;
    var scale = depth <= cv.CV_8S? 1.0 : (depth <= cv.CV_32S? 1.0/256.0 : 255.0);
    var shift = (depth === cv.CV_8S || depth === cv.CV_16S)? 128.0 : 0.0;
    mat.convertTo(img, cv.CV_8U, scale, shift);

    // convert the img type to cv.CV_8UC4
    switch (img.type()) {
        case cv.CV_8UC1:
            cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
            break;
        case cv.CV_8UC3:
            cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
            break;
        case cv.CV_8UC4:
            break;
        default:
            throw new Error('Bad number of channels (Source image must have 1, 3 or 4 channels)');
    }
    var imgData = new ImageData(new Uint8ClampedArray(img.data), img.cols, img.rows);
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    ctx.putImageData(imgData, 0, 0);
    img.delete();
};

Module['VideoCapture'] = function(videoSource) {
    var video = null;
    if (typeof videoSource === 'string') {
        video = document.getElementById(videoSource);
    } else {
        video = videoSource;
    }
    if (!(video instanceof HTMLVideoElement)) {
        throw new Error('Please input the valid video element or id.');
    }
    var canvas = document.createElement('canvas');
    canvas.width = video.width;
    canvas.height = video.height;
    var ctx = canvas.getContext('2d');
    this.video = video;
    this.read = function(frame) {
        if (!(frame instanceof cv.Mat)) {
            throw new Error('Please input the valid cv.Mat instance.');
        }
        if (frame.type() !== cv.CV_8UC4) {
            throw new Error('Bad type of input mat: the type should be cv.CV_8UC4.');
        }
        if (frame.cols !== video.width || frame.rows !== video.height) {
            throw new Error('Bad size of input mat: the size should be same as the video.');
        }
        ctx.drawImage(video, 0, 0, video.width, video.height);
        frame.data.set(ctx.getImageData(0, 0, video.width, video.height).data);
    };
};

function Range(start, end) {
    this.start = typeof(start) === 'undefined' ? 0 : start;
    this.end = typeof(end) === 'undefined' ? 0 : end;
}

Module['Range'] = Range;

function Point(x, y) {
    this.x = typeof(x) === 'undefined' ? 0 : x;
    this.y = typeof(y) === 'undefined' ? 0 : y;
}

Module['Point'] = Point;

function Size(width, height) {
    this.width = typeof(width) === 'undefined' ? 0 : width;
    this.height = typeof(height) === 'undefined' ? 0 : height;
}

Module['Size'] = Size;

function Rect() {
    switch (arguments.length) {
        case 0: {
            // new cv.Rect()
            this.x = 0;
            this.y = 0;
            this.width = 0;
            this.height = 0;
            break;
        }
        case 1: {
            // new cv.Rect(rect)
            var rect = arguments[0];
            this.x = rect.x;
            this.y = rect.y;
            this.width = rect.width;
            this.height = rect.height;
            break;
        }
        case 2: {
            // new cv.Rect(point, size)
            var point = arguments[0];
            var size = arguments[1];
            this.x = point.x;
            this.y = point.y;
            this.width = size.width;
            this.height = size.height;
            break;
        }
        case 4: {
            // new cv.Rect(x, y, width, height)
            this.x = arguments[0];
            this.y = arguments[1];
            this.width = arguments[2];
            this.height = arguments[3];
            break;
        }
        default: {
            throw new Error('Invalid arguments');
        }
    }
}

Module['Rect'] = Rect;

function RotatedRect() {
    switch (arguments.length) {
        case 0: {
            this.center = {x: 0, y: 0};
            this.size = {width: 0, height: 0};
            this.angle = 0;
            break;
        }
        case 3: {
            this.center = arguments[0];
            this.size = arguments[1];
            this.angle = arguments[2];
            break;
        }
        default: {
            throw new Error('Invalid arguments');
        }
    }
}

RotatedRect.points = function(obj) {
    return Module.rotatedRectPoints(obj);
};

RotatedRect.boundingRect = function(obj) {
    return Module.rotatedRectBoundingRect(obj);
};

RotatedRect.boundingRect2f = function(obj) {
    return Module.rotatedRectBoundingRect2f(obj);
};

Module['RotatedRect'] = RotatedRect;

function Scalar(v0, v1, v2, v3) {
    this.push(typeof(v0) === 'undefined' ? 0 : v0);
    this.push(typeof(v1) === 'undefined' ? 0 : v1);
    this.push(typeof(v2) === 'undefined' ? 0 : v2);
    this.push(typeof(v3) === 'undefined' ? 0 : v3);
}

Scalar.prototype = new Array; // eslint-disable-line no-array-constructor

Scalar.all = function(v) {
    return Scalar(v, v, v, v);
};

Module['Scalar'] = Scalar;

function MinMaxLoc() {
    switch (arguments.length) {
        case 0: {
            this.minVal = 0;
            this.maxVal = 0;
            this.minLoc = Point(0, 0);
            this.maxLoc = Point(0, 0);
            break;
        }
        case 4: {
            this.minVal = arguments[0];
            this.maxVal = arguments[1];
            this.minLoc = arguments[2];
            this.maxLoc = arguments[3];
            break;
        }
        default: {
            throw new Error('Invalid arguments');
        }
    }
}

Module['MinMaxLoc'] = MinMaxLoc;

function Circle() {
    switch (arguments.length) {
        case 0: {
            this.center = Point(0, 0);
            this.radius = 0;
            break;
        }
        case 2: {
            this.center = arguments[0];
            this.radius = arguments[1];
            break;
        }
        default: {
            throw new Error('Invalid arguments');
        }
    }
}

Module['Circle'] = Circle;

function TermCriteria() {
    switch (arguments.length) {
        case 0: {
            this.type = 0;
            this.maxCount = 0;
            this.epsilon = 0;
            break;
        }
        case 3: {
            this.type = arguments[0];
            this.maxCount = arguments[1];
            this.epsilon = arguments[2];
            break;
        }
        default: {
            throw new Error('Invalid arguments');
        }
    }
}

Module['TermCriteria'] = TermCriteria;

Module['matFromArray'] = function(rows, cols, type, array) {
    var mat = new cv.Mat(rows, cols, type);
    switch (type) {
        case cv.CV_8U:
        case cv.CV_8UC1:
        case cv.CV_8UC2:
        case cv.CV_8UC3:
        case cv.CV_8UC4: {
            mat.data.set(array);
            break;
        }
        case cv.CV_8S:
        case cv.CV_8SC1:
        case cv.CV_8SC2:
        case cv.CV_8SC3:
        case cv.CV_8SC4: {
            mat.data8S.set(array);
            break;
        }
        case cv.CV_16U:
        case cv.CV_16UC1:
        case cv.CV_16UC2:
        case cv.CV_16UC3:
        case cv.CV_16UC4: {
            mat.data16U.set(array);
            break;
        }
        case cv.CV_16S:
        case cv.CV_16SC1:
        case cv.CV_16SC2:
        case cv.CV_16SC3:
        case cv.CV_16SC4: {
            mat.data16S.set(array);
            break;
        }
        case cv.CV_32S:
        case cv.CV_32SC1:
        case cv.CV_32SC2:
        case cv.CV_32SC3:
        case cv.CV_32SC4: {
            mat.data32S.set(array);
            break;
        }
        case cv.CV_32F:
        case cv.CV_32FC1:
        case cv.CV_32FC2:
        case cv.CV_32FC3:
        case cv.CV_32FC4: {
            mat.data32F.set(array);
            break;
        }
        case cv.CV_64F:
        case cv.CV_64FC1:
        case cv.CV_64FC2:
        case cv.CV_64FC3:
        case cv.CV_64FC4: {
            mat.data64F.set(array);
            break;
        }
        default: {
            throw new Error('Type is unsupported');
        }
    }
    return mat;
};

Module['matFromImageData'] = function(imageData) {
    var mat = new cv.Mat(imageData.height, imageData.width, cv.CV_8UC4);
    mat.data.set(imageData.data);
    return mat;
};
