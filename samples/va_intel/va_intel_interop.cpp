/* origin: libva-1.3.1/test/decode/mpeg2vldemo.cpp */

/*
 * Copyright (c) 2007-2008 Intel Corporation. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL PRECISION INSIGHT AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <stdexcept>
#include <string>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <va/va.h>

#include "display.cpp.inc"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/va_intel.hpp"

#define CHECK_VASTATUS(_status,_func) \
    if (_status != VA_STATUS_SUCCESS) \
    { \
        char str[256]; \
        snprintf(str, sizeof(str)-1, "%s:%s (%d) failed(status=0x%08x),exit\n", __func__, _func, __LINE__, _status); \
        throw std::runtime_error(str); \
    }

class CmdlineParser
{
public:
    enum { fnInput=0, fnOutput1, fnOutput2, _fnNumFiles }; // file name indices
    CmdlineParser(int argc, char** argv):
        m_argc(argc), m_argv(argv)
        {}
    void usage()
        {
            fprintf(stderr,
                    "Usage: va_intel_interop [-f] infile outfile1 outfile2\n\n"
                    "Interop ON/OFF version\n\n"
                    "where:  -f    option indicates interop is off (fallback mode); interop is on by default\n"
                    "        infile   is to be existing, contains input image data (bmp, jpg, png, tiff, etc)\n"
                    "        outfile1 is to be created, contains original surface data (NV12)\n"
                    "        outfile2 is to be created, contains processed surface data (NV12)\n");
        }
    // true => go, false => usage/exit; extra args/unknown options are ignored for simplicity
    bool run()
        {
            int n = 0;
            for (int i = 0; i < _fnNumFiles; ++i)
                m_files[i] = 0;
            m_interop = true;
            for (int i = 1; i < m_argc; ++i)
            {
                const char *arg = m_argv[i];
                if (arg[0] == '-') // option
                {
                    if (!strcmp(arg, "-f"))
                        m_interop = false;
                }
                else // parameter
                {
                    if (n < _fnNumFiles)
                        m_files[n++] = arg;
                }
            }
            return bool(n >= _fnNumFiles);
        }
    bool isInterop() const
        {
            return m_interop;
        }
    const char* getFile(int n) const
        {
            return ((n >= 0) && (n < _fnNumFiles)) ? m_files[n] : 0;
        }
private:
    int m_argc;
    char** m_argv;
    const char* m_files[_fnNumFiles];
    bool m_interop;
};

class Timer
{
public:
    enum UNITS
    {
        USEC = 0,
        MSEC,
        SEC
    };

    Timer() : m_t0(0), m_diff(0)
    {
        m_tick_frequency = (float)cv::getTickFrequency();

        m_unit_mul[USEC] = 1000000;
        m_unit_mul[MSEC] = 1000;
        m_unit_mul[SEC]  = 1;
    }

    void clear()
    {
        m_t0 = m_diff = 0;
    }

    void start()
    {
        m_t0 = cv::getTickCount();
    }

    void stop()
    {
        m_diff = cv::getTickCount() - m_t0;
    }

    float time(UNITS u = MSEC)
    {
        float sec = m_diff / m_tick_frequency;

        return sec * m_unit_mul[u];
    }

public:
    float m_tick_frequency;
    int64 m_t0;
    int64 m_diff;
    int   m_unit_mul[3];
};

static void checkIfAvailableYUV420()
{
    VAEntrypoint entrypoints[5];
    int num_entrypoints,vld_entrypoint;
    VAConfigAttrib attrib;
    VAStatus status;

    status = vaQueryConfigEntrypoints(va::display, VAProfileMPEG2Main, entrypoints, &num_entrypoints);
    CHECK_VASTATUS(status, "vaQueryConfigEntrypoints");

    for (vld_entrypoint = 0; vld_entrypoint < num_entrypoints; ++vld_entrypoint)
    {
        if (entrypoints[vld_entrypoint] == VAEntrypointVLD)
            break;
    }
    if (vld_entrypoint == num_entrypoints)
        throw std::runtime_error("Failed to find VLD entry point");

    attrib.type = VAConfigAttribRTFormat;
    vaGetConfigAttributes(va::display, VAProfileMPEG2Main, VAEntrypointVLD, &attrib, 1);
    if ((attrib.value & VA_RT_FORMAT_YUV420) == 0)
        throw std::runtime_error("Desired YUV420 RT format not found");
}

static cv::UMat readImage(const char* fileName)
{
    cv::Mat m = cv::imread(fileName);
    if (m.empty())
        throw std::runtime_error("Failed to load image: " + std::string(fileName));
    return m.getUMat(cv::ACCESS_RW);
}

static void writeImage(const cv::UMat& u, const char* fileName, bool doInterop)
{
    std::string fn = std::string(fileName) + std::string(doInterop ? ".on" : ".off") + std::string(".jpg");
    cv::imwrite(fn, u);
}

static float run(const char* infile, const char* outfile1, const char* outfile2, bool doInterop)
{
    VASurfaceID surface;
    VAStatus status;
    Timer t;
    if(doInterop) {
        // initialize CL context for CL/VA interop
        cv::va_intel::ocl::initializeContextFromVA(va::display);
    }
    // load input image
    cv::UMat u1 = readImage(infile);
    cv::Size size2 = u1.size();
    status = vaCreateSurfaces(va::display, VA_RT_FORMAT_YUV420, size2.width, size2.height, &surface, 1, NULL, 0);
    CHECK_VASTATUS(status, "vaCreateSurfaces");

    // transfer image into VA surface, make sure all CL initialization is done (kernels etc)
    cv::va_intel::convertToVASurface(va::display, u1, surface, size2);
    cv::va_intel::convertFromVASurface(va::display, surface, size2, u1);
    cv::UMat u2;
    cv::blur(u1, u2, cv::Size(7, 7), cv::Point(-3, -3));
    // measure performance on some image processing
    writeImage(u1, outfile1, doInterop);
    t.start();
    cv::va_intel::convertFromVASurface(va::display, surface, size2, u1);
    cv::blur(u1, u2, cv::Size(7, 7), cv::Point(-3, -3));
    cv::va_intel::convertToVASurface(va::display, u2, surface, size2);
    t.stop();
    writeImage(u2, outfile2, doInterop);

    vaDestroySurfaces(va::display, &surface,1);

    return t.time(Timer::MSEC);
}

int main(int argc, char** argv)
{
    try
    {
        CmdlineParser cmd(argc, argv);
        if (!cmd.run())
        {
            cmd.usage();
            return 0;
        }

        if (!va::openDisplay())
            throw std::runtime_error("Failed to open VA display for CL-VA interoperability");
        std::cout << "VA display opened successfully" << std::endl;

        checkIfAvailableYUV420();

        const char* infile = cmd.getFile(CmdlineParser::fnInput);
        const char* outfile1 = cmd.getFile(CmdlineParser::fnOutput1);
        const char* outfile2 = cmd.getFile(CmdlineParser::fnOutput2);
        bool doInterop = cmd.isInterop();

        float time = run(infile, outfile1, outfile2, doInterop);

        std::cout << "Interop " << (doInterop ? "ON " : "OFF") << ": processing time, msec: " << time << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << std::endl;
    }

    va::closeDisplay();
    return 0;
}
