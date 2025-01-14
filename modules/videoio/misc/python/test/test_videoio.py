#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv
import io
import sys

from tests_common import NewOpenCVTests

class Bindings(NewOpenCVTests):

    def check_name(self, name):
        #print(name)
        self.assertFalse(name == None)
        self.assertFalse(name == "")

    def test_registry(self):
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_ANY));
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_FFMPEG))
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_OPENCV_MJPEG))
        backends = cv.videoio_registry.getBackends()
        for backend in backends:
            self.check_name(cv.videoio_registry.getBackendName(backend))

    def test_capture_stream_file(self):
        if sys.version_info[0] < 3:
            raise self.skipTest('Python 3.x required')

        api_pref = None
        for backend in cv.videoio_registry.getStreamBufferedBackends():
            if not cv.videoio_registry.hasBackend(backend):
                continue
            if not cv.videoio_registry.isBackendBuiltIn(backend):
                _, abi, api = cv.videoio_registry.getStreamBufferedBackendPluginVersion(backend)
                if (abi < 1 or (abi == 1 and api < 2)):
                    continue
            api_pref = backend
            break

        if not api_pref:
            raise self.skipTest("No available backends")

        with open(self.find_file("cv/video/768x576.avi"), "rb") as f:
            cap = cv.VideoCapture(f, api_pref, [])
            self.assertTrue(cap.isOpened())
            hasFrame, frame = cap.read()
            self.assertTrue(hasFrame)
            self.assertEqual(frame.shape, (576, 768, 3))

    def test_capture_stream_buffer(self):
        if sys.version_info[0] < 3:
            raise self.skipTest('Python 3.x required')

        api_pref = None
        for backend in cv.videoio_registry.getStreamBufferedBackends():
            if not cv.videoio_registry.hasBackend(backend):
                continue
            if not cv.videoio_registry.isBackendBuiltIn(backend):
                _, abi, api = cv.videoio_registry.getStreamBufferedBackendPluginVersion(backend)
                if (abi < 1 or (abi == 1 and api < 2)):
                    continue
            api_pref = backend
            break

        if not api_pref:
            raise self.skipTest("No available backends")

        class BufferStream(io.BufferedIOBase):
            def __init__(self, filepath):
                self.f = open(filepath, "rb")

            def read(self, size=-1):
                return self.f.read(size)

            def seek(self, offset, whence):
                return self.f.seek(offset, whence)

            def __del__(self):
                self.f.close()

        stream = BufferStream(self.find_file("cv/video/768x576.avi"))

        cap = cv.VideoCapture(stream, api_pref, [])
        self.assertTrue(cap.isOpened())
        hasFrame, frame = cap.read()
        self.assertTrue(hasFrame)
        self.assertEqual(frame.shape, (576, 768, 3))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
