// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
#define OPENCV_VIDEOIO_UTILS_PRIVATE_HPP

#include "opencv2/core/cvdef.h"
#include <string>

namespace cv {
CV_EXPORTS std::string icvExtractPattern(const std::string& filename, unsigned *offset);

class PluginStreamReader : public IStreamReader
{
public:
    PluginStreamReader(void* _opaque,
                       long long (*_read)(void* opaque, char* buffer, long long size),
                       long long (*_seek)(void* opaque, long long offset, int way))
    {
        opaque = _opaque;
        readCallback = _read;
        seekCallback = _seek;
    }

    virtual ~PluginStreamReader() {}

    long long read(char* buffer, long long size) override
    {
        return readCallback(opaque, buffer, size);
    }

    long long seek(long long offset, int way) override
    {
        return seekCallback(opaque, offset, way);
    }

private:
    void* opaque;
    long long (*readCallback)(void* opaque, char* buffer, long long size);
    long long (*seekCallback)(void* opaque, long long offset, int way);
};

}

#endif // OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
