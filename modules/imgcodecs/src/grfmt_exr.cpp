/*M///////////////////////////////////////////////////////////////////////////////////////
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

#include "precomp.hpp"

#ifdef HAVE_OPENEXR

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#if defined _MSC_VER && _MSC_VER >= 1200
#  pragma warning( disable: 4100 4244 4267 )
#endif

#if defined __GNUC__ && defined __APPLE__
#  pragma GCC diagnostic ignored "-Wshadow"
#endif

/// C++ Standard Libraries
#include <iostream>
#include <stdexcept>

#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfStandardAttributes.h>
#include <half.h>
#include "grfmt_exr.hpp"
#include "OpenEXRConfig.h"

#if defined _WIN32

#undef UINT
#define UINT ((Imf::PixelType)0)
#undef HALF
#define HALF ((Imf::PixelType)1)
#undef FLOAT
#define FLOAT ((Imf::PixelType)2)

#endif

namespace cv
{

static bool isOpenEXREnabled()
{
    static const bool PARAM_ENABLE_OPENEXR = utils::getConfigurationParameterBool("OPENCV_IO_ENABLE_OPENEXR",
#ifdef OPENCV_IMGCODECS_USE_OPENEXR
        true
#else
        false
#endif
    );
    return PARAM_ENABLE_OPENEXR;
}
static void initOpenEXR()
{
    if (!isOpenEXREnabled())
    {
        const char* message = "imgcodecs: OpenEXR codec is disabled. You can enable it via 'OPENCV_IO_ENABLE_OPENEXR' option. Refer for details and cautions here: https://github.com/opencv/opencv/issues/21326";
        CV_LOG_WARNING(NULL, message);
        CV_Error(Error::StsNotImplemented, message);
    }
}

/////////////////////// ExrDecoder ///////////////////

ExrDecoder::ExrDecoder()
{
    m_signature = "\x76\x2f\x31\x01";
    m_file = 0;
    m_red = m_green = m_blue = m_alpha = 0;
    m_type = ((Imf::PixelType)0);
    m_iscolor = false;
    m_bit_depth = 0;
    m_isfloat = false;
    m_ischroma = false;
    m_hasalpha = false;
    m_native_depth = false;

}


ExrDecoder::~ExrDecoder()
{
    close();
}


void  ExrDecoder::close()
{
    if( m_file )
    {
        delete m_file;
        m_file = 0;
    }
}


int  ExrDecoder::type() const
{
    return CV_MAKETYPE((m_isfloat ? CV_32F : CV_32S), ((m_iscolor && m_hasalpha) ? 4 : m_iscolor ? 3 : m_hasalpha ? 2 : 1));
}


bool  ExrDecoder::readHeader()
{
    bool result = false;

    m_file = new InputFile( m_filename.c_str() );

    if( !m_file ) // probably paranoid
        return false;

    m_datawindow = m_file->header().dataWindow();
    m_width = m_datawindow.max.x - m_datawindow.min.x + 1;
    m_height = m_datawindow.max.y - m_datawindow.min.y + 1;

    // the type HALF is converted to 32 bit float
    // and the other types supported by OpenEXR are 32 bit anyway
    m_bit_depth = 32;

    if( hasChromaticities( m_file->header() ))
        m_chroma = chromaticities( m_file->header() );

    const ChannelList &channels = m_file->header().channels();
    m_red = channels.findChannel( "R" );
    m_green = channels.findChannel( "G" );
    m_blue = channels.findChannel( "B" );
    m_alpha = channels.findChannel( "A" );

    if( m_alpha ) // alpha channel supported in RGB, Y, and YC scenarios
        m_hasalpha = true;

    if( m_red || m_green || m_blue )
    {
        m_iscolor = true;
        m_ischroma = false;
        result = true;
    }
    else
    {
        m_green = channels.findChannel( "Y" );
        if( !m_green )
        {
            m_green = channels.findChannel( "Z" ); // Distance of the front of a sample from the viewer
        }
        if( m_green )
        {
            m_ischroma = true;
            m_red = channels.findChannel( "RY" );
            m_blue = channels.findChannel( "BY" );
            m_iscolor = (m_blue || m_red);
            result = true;
        }
        else
            result = false;
    }

    if( result )
    {
        m_type = FLOAT;
        m_isfloat = ( m_type == FLOAT );
    }

    if( !result )
        close();

    return result;
}


bool  ExrDecoder::readData( Mat& img )
{
    m_native_depth = CV_MAT_DEPTH(type()) == img.depth();
    bool color = img.channels() > 2; // output mat has 3+ channels; Y or YA are the 1 and 2 channel scenario
    bool alphasupported = ( img.channels() % 2 == 0 );  // even number of channels indicates alpha
    int channels = 0;
    uchar* data = img.ptr();
    size_t step = img.step;
    bool justcopy = ( m_native_depth && (color == m_iscolor) );
    bool chromatorgb = ( m_ischroma && color );
    bool rgbtogray = ( !m_ischroma && m_iscolor && !color );
    bool result = true;
    FrameBuffer frame;
    const int defaultchannels = 3;
    int xsample[defaultchannels] = {1, 1, 1};
    char *buffer;
    CV_Assert(m_type == FLOAT);
    const size_t floatsize = sizeof(float);
    size_t xstep = m_native_depth ? floatsize : 1; // 4 bytes if native depth (FLOAT), otherwise converting to 1 byte U8 depth
    size_t ystep = 0;
    const int channelstoread = ( (m_iscolor && alphasupported) ? 4 :
                                ( (m_iscolor && !m_ischroma) || color) ? 3 : alphasupported ? 2 : 1 ); // number of channels to read may exceed channels in output img
    size_t xStride = floatsize * channelstoread;

    // See https://github.com/opencv/opencv/issues/26705
    // If ALGO_HINT_ACCURATE is set, read BGR and swap to RGB.
    // If ALGO_HINT_APPROX is set,   read RGB directly.
    bool doReadRGB = m_use_rgb;
    bool doPostColorSwap = false; // After decoding, swap BGR to RGB
    if(m_use_rgb && (getDefaultAlgorithmHint() == ALGO_HINT_ACCURATE) )
    {
        doReadRGB = false;
        doPostColorSwap = true;
    }

    AutoBuffer<char> copy_buffer;

    if( !justcopy )
    {
        copy_buffer.allocate(floatsize * m_width * defaultchannels);
        buffer = copy_buffer.data();
        ystep = 0;
    }
    else
    {
        buffer = (char *)data;
        ystep = step;
    }

    if( m_ischroma )
    {
        if( color )
        {
            if( m_blue )
            {
                frame.insert( "BY", Slice( m_type,
                                           buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep,
                                           xStride, ystep, m_blue->xSampling, m_blue->ySampling, 0.0 ));
                xsample[0] = m_blue->xSampling;
            }
            else
            {
                frame.insert( "BY", Slice( m_type,
                                           buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep,
                                           xStride, ystep, 1, 1, 0.0 ));
            }
            if( m_green )
            {
                frame.insert( "Y", Slice( m_type,
                                          buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + floatsize,
                                          xStride, ystep, m_green->xSampling, m_green->ySampling, 0.0 ));
                xsample[1] = m_green->xSampling;
            }
            else
            {
                frame.insert( "Y", Slice( m_type,
                                          buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + floatsize,
                                          xStride, ystep, 1, 1, 0.0 ));
            }
            if( m_red )
            {
                frame.insert( "RY", Slice( m_type,
                                           buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + (floatsize * 2),
                                           xStride, ystep, m_red->xSampling, m_red->ySampling, 0.0 ));
                xsample[2] = m_red->xSampling;
            }
            else
            {
                frame.insert( "RY", Slice( m_type,
                                           buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + (floatsize * 2),
                                           xStride, ystep, 1, 1, 0.0 ));
            }
        }
        else
        {
            frame.insert( "Y", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep,
                            xStride, ystep, m_green->xSampling, m_green->ySampling, 0.0 ));
            xsample[0] = m_green->xSampling;
        }
    }
    else
    {
        if( m_blue )
        {
            frame.insert( "B", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep,
                            xStride, ystep, m_blue->xSampling, m_blue->ySampling, 0.0 ));
            xsample[0] = m_blue->xSampling;
        }
        else
        {
            frame.insert( "B", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep,
                            xStride, ystep, 1, 1, 0.0 ));
        }
        if( m_green )
        {
            frame.insert( "G", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + floatsize,
                            xStride, ystep, m_green->xSampling, m_green->ySampling, 0.0 ));
            xsample[1] = m_green->xSampling;
        }
        else
        {
            frame.insert( "G", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + floatsize,
                            xStride, ystep, 1, 1, 0.0 ));
        }
        if( m_red )
        {
            frame.insert( "R", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + (floatsize * 2),
                            xStride, ystep, m_red->xSampling, m_red->ySampling, 0.0 ));
            xsample[2] = m_red->xSampling;
        }
        else
        {
            frame.insert( "R", Slice( m_type,
                            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + (floatsize * 2),
                            xStride, ystep, 1, 1, 0.0 ));
        }
    }

    if( justcopy && m_hasalpha && alphasupported )
    { // alpha preserved only in justcopy scenario where alpha is desired (alphasupported)
      // and present in original file (m_hasalpha)
        CV_Assert(channelstoread == img.channels());
        int offset = (channelstoread - 1) * floatsize;
        frame.insert( "A", Slice( m_type,
            buffer - m_datawindow.min.x * xStride - m_datawindow.min.y * ystep + offset,
            xStride, ystep, m_alpha->xSampling, m_alpha->ySampling, 0.0 ));
    }

    for (FrameBuffer::Iterator it = frame.begin(); it != frame.end(); it++) {
        channels++;
    }

    CV_Assert(channels == channelstoread);

    if( (channels != channelstoread) || (!justcopy && channels > defaultchannels) )
    { // safety checking what ought to be true here
        close();
        return false;
    }

    m_file->setFrameBuffer( frame );
    if( justcopy )
    {
        m_file->readPixels( m_datawindow.min.y, m_datawindow.max.y );

        if( m_iscolor )
        {
            if (doReadRGB)
            {
                if( m_red && (m_red->xSampling != 1 || m_red->ySampling != 1) )
                    UpSample( data, channelstoread, step / xstep, m_red->xSampling, m_red->ySampling );
                if( m_green && (m_green->xSampling != 1 || m_green->ySampling != 1) )
                    UpSample( data + xstep, channelstoread, step / xstep, m_green->xSampling, m_green->ySampling );
                if( m_blue && (m_blue->xSampling != 1 || m_blue->ySampling != 1) )
                    UpSample( data + 2 * xstep, channelstoread, step / xstep, m_blue->xSampling, m_blue->ySampling );
            }
            else
            {
                if( m_blue && (m_blue->xSampling != 1 || m_blue->ySampling != 1) )
                    UpSample( data, channelstoread, step / xstep, m_blue->xSampling, m_blue->ySampling );
                if( m_green && (m_green->xSampling != 1 || m_green->ySampling != 1) )
                    UpSample( data + xstep, channelstoread, step / xstep, m_green->xSampling, m_green->ySampling );
                if( m_red && (m_red->xSampling != 1 || m_red->ySampling != 1) )
                    UpSample( data + 2 * xstep, channelstoread, step / xstep, m_red->xSampling, m_red->ySampling );
            }
        }
        else if( m_green && (m_green->xSampling != 1 || m_green->ySampling != 1) )
            UpSample( data, channelstoread, step / xstep, m_green->xSampling, m_green->ySampling );

        if( chromatorgb )
        {
            if (doReadRGB)
                ChromaToRGB( (float *)data, m_height, channelstoread, step / xstep );
            else
                ChromaToBGR( (float *)data, m_height, channelstoread, step / xstep );
        }
    }
    else
    {
        uchar *out = data;
        int x, y;
        for( y = m_datawindow.min.y; y <= m_datawindow.max.y; y++ )
        {
            m_file->readPixels( y, y );

            for( int i = 0; i < channels; i++ )
            {
                if( xsample[i] != 1 )
                    UpSampleX( (float *)buffer + i, channels, xsample[i] );
            }
            if( rgbtogray )
            {
                RGBToGray( (float *)buffer, (float *)out );
            }
            else
            {
                if( chromatorgb )
                {
                    if (doReadRGB)
                        ChromaToRGB( (float *)buffer, 1, defaultchannels, step );
                    else
                        ChromaToBGR( (float *)buffer, 1, defaultchannels, step );
                }

                if( m_type == FLOAT )
                {
                    float *fi = (float *)buffer;
                    for( x = 0; x < m_width * img.channels(); x++)
                    {
                        out[x] = cv::saturate_cast<uchar>(fi[x]);
                    }
                }
                else
                {
                    unsigned *ui = (unsigned *)buffer;
                    for( x = 0; x < m_width * img.channels(); x++)
                    {
                        out[x] = cv::saturate_cast<uchar>(ui[x]);
                    }
                }
            }

            out += step;
        }
        if( color )
        {
            if (doReadRGB)
            {
                if( m_red && (m_red->xSampling != 1 || m_red->ySampling != 1) )
                    UpSampleY( data, defaultchannels, step / xstep, m_red->ySampling );
                if( m_green && (m_green->xSampling != 1 || m_green->ySampling != 1) )
                    UpSampleY( data + xstep, defaultchannels, step / xstep, m_green->ySampling );
                if( m_blue && (m_blue->xSampling != 1 || m_blue->ySampling != 1) )
                    UpSampleY( data + 2 * xstep, defaultchannels, step / xstep, m_blue->ySampling );
            }
            else
            {
                if( m_blue && (m_blue->xSampling != 1 || m_blue->ySampling != 1) )
                    UpSampleY( data, defaultchannels, step / xstep, m_blue->ySampling );
                if( m_green && (m_green->xSampling != 1 || m_green->ySampling != 1) )
                    UpSampleY( data + xstep, defaultchannels, step / xstep, m_green->ySampling );
                if( m_red && (m_red->xSampling != 1 || m_red->ySampling != 1) )
                    UpSampleY( data + 2 * xstep, defaultchannels, step / xstep, m_red->ySampling );
            }
        }
        else if( m_green && (m_green->xSampling != 1 || m_green->ySampling != 1) )
            UpSampleY( data, 1, step / xstep, m_green->ySampling );
    }

    close();

    if(doPostColorSwap)
    {
        cvtColor( img, img, cv::COLOR_BGR2RGB );
    }

    return result;
}

/**
// on entry pixel values are stored packed in the upper left corner of the image
// this functions expands them by duplication to cover the whole image
 */
void  ExrDecoder::UpSample( uchar *data, int xstep, int ystep, int xsample, int ysample )
{
    for( int y = (m_height - 1) / ysample, yre = m_height - ysample; y >= 0; y--, yre -= ysample )
    {
        for( int x = (m_width - 1) / xsample, xre = m_width - xsample; x >= 0; x--, xre -= xsample )
        {
            for( int i = 0; i < ysample; i++ )
            {
                for( int n = 0; n < xsample; n++ )
                {
                    if( !m_native_depth )
                        data[(yre + i) * ystep + (xre + n) * xstep] = data[y * ystep + x * xstep];
                    else if( m_type == FLOAT )
                        ((float *)data)[(yre + i) * ystep + (xre + n) * xstep] = ((float *)data)[y * ystep + x * xstep];
                    else
                        ((unsigned *)data)[(yre + i) * ystep + (xre + n) * xstep] = ((unsigned *)data)[y * ystep + x * xstep];
                }
            }
        }
    }
}

/**
// on entry pixel values are stored packed in the upper left corner of the image
// this functions expands them by duplication to cover the whole image
 */
void  ExrDecoder::UpSampleX( float *data, int xstep, int xsample )
{
    for( int x = (m_width - 1) / xsample, xre = m_width - xsample; x >= 0; x--, xre -= xsample )
    {
        for( int n = 0; n < xsample; n++ )
        {
            if( m_type == FLOAT )
                ((float *)data)[(xre + n) * xstep] = ((float *)data)[x * xstep];
            else
                ((unsigned *)data)[(xre + n) * xstep] = ((unsigned *)data)[x * xstep];
        }
    }
}

/**
// on entry pixel values are stored packed in the upper left corner of the image
// this functions expands them by duplication to cover the whole image
 */
void  ExrDecoder::UpSampleY( uchar *data, int xstep, int ystep, int ysample )
{
    for( int y = m_height - ysample, yre = m_height - ysample; y >= 0; y -= ysample, yre -= ysample )
    {
        for( int x = 0; x < m_width; x++ )
        {
            for( int i = 1; i < ysample; i++ )
            {
                if( !m_native_depth )
                    data[(yre + i) * ystep + x * xstep] = data[y * ystep + x * xstep];
                else if( m_type == FLOAT )
                    ((float *)data)[(yre + i) * ystep + x * xstep] = ((float *)data)[y * ystep + x * xstep];
                else
                    ((unsigned *)data)[(yre + i) * ystep + x * xstep] = ((unsigned *)data)[y * ystep + x * xstep];
            }
        }
    }
}

/**
// algorithm from ImfRgbaYca.cpp
 */
void  ExrDecoder::ChromaToBGR( float *data, int numlines, int xstep, int ystep )
{
    for( int y = 0; y < numlines; y++ )
    {
        for( int x = 0; x < m_width; x++ )
        {
            double b, Y, r;
            if( m_type == FLOAT )
            {
                b = data[y * ystep + x * xstep];
                Y = data[y * ystep + x * xstep + 1];
                r = data[y * ystep + x * xstep + 2];
            }
            else
            {
                b = ((unsigned *)data)[y * ystep + x * xstep];
                Y = ((unsigned *)data)[y * ystep + x * xstep + 1];
                r = ((unsigned *)data)[y * ystep + x * xstep + 2];
            }
            r = (r + 1) * Y;
            b = (b + 1) * Y;
            Y = (Y - b * m_chroma.blue[1] - r * m_chroma.red[1]) / m_chroma.green[1];

            if( m_type == FLOAT )
            {
                data[y * ystep + x * xstep] = (float)b;
                data[y * ystep + x * xstep + 1] = (float)Y;
                data[y * ystep + x * xstep + 2] = (float)r;
            }
            else
            {
                int t = cvRound(b);
                ((unsigned *)data)[y * ystep + x * xstep + 0] = (unsigned)MAX(t, 0);
                t = cvRound(Y);
                ((unsigned *)data)[y * ystep + x * xstep + 1] = (unsigned)MAX(t, 0);
                t = cvRound(r);
                ((unsigned *)data)[y * ystep + x * xstep + 2] = (unsigned)MAX(t, 0);
            }
        }
    }
}

void  ExrDecoder::ChromaToRGB(float *data, int numlines, int xstep, int ystep)
{
    for( int y = 0; y < numlines; y++ )
    {
        for( int x = 0; x < m_width; x++ )
        {
            double b, Y, r;
            if( m_type == FLOAT )
            {
                b = data[y * ystep + x * xstep];
                Y = data[y * ystep + x * xstep + 1];
                r = data[y * ystep + x * xstep + 2];
            }
            else
            {
                b = ((unsigned *)data)[y * ystep + x * xstep];
                Y = ((unsigned *)data)[y * ystep + x * xstep + 1];
                r = ((unsigned *)data)[y * ystep + x * xstep + 2];
            }
            r = (r + 1) * Y;
            b = (b + 1) * Y;
            Y = (Y - b * m_chroma.blue[1] - r * m_chroma.red[1]) / m_chroma.green[1];

            if( m_type == FLOAT )
            {
                data[y * ystep + x * xstep] = (float)r;
                data[y * ystep + x * xstep + 1] = (float)Y;
                data[y * ystep + x * xstep + 2] = (float)b;
            }
            else
            {
                int t = cvRound(r);
                ((unsigned *)data)[y * ystep + x * xstep + 0] = (unsigned)MAX(t, 0);
                t = cvRound(Y);
                ((unsigned *)data)[y * ystep + x * xstep + 1] = (unsigned)MAX(t, 0);
                t = cvRound(b);
                ((unsigned *)data)[y * ystep + x * xstep + 2] = (unsigned)MAX(t, 0);
            }
        }
    }
}

/**
// convert one row to gray
*/
void  ExrDecoder::RGBToGray( float *in, float *out )
{
    if( m_type == FLOAT )
    {
        if( m_native_depth )
        {
            for( int i = 0, n = 0; i < m_width; i++, n += 3 )
                out[i] = in[n] * m_chroma.blue[0] + in[n + 1] * m_chroma.green[0] + in[n + 2] * m_chroma.red[0];
        }
        else
        {
            uchar *o = (uchar *)out;
            for( int i = 0, n = 0; i < m_width; i++, n += 3 )
                o[i] = (uchar) (in[n] * m_chroma.blue[0] + in[n + 1] * m_chroma.green[0] + in[n + 2] * m_chroma.red[0]);
        }
    }
    else // UINT
    {
        if( m_native_depth )
        {
            unsigned *ui = (unsigned *)in;
            for( int i = 0; i < m_width * 3; i++ )
                ui[i] -= 0x80000000;
            int *si = (int *)in;
            for( int i = 0, n = 0; i < m_width; i++, n += 3 )
                ((int *)out)[i] = int(si[n] * m_chroma.blue[0] + si[n + 1] * m_chroma.green[0] + si[n + 2] * m_chroma.red[0]);
        }
        else // how to best convert float to uchar?
        {
            unsigned *ui = (unsigned *)in;
            for( int i = 0, n = 0; i < m_width; i++, n += 3 )
                ((uchar *)out)[i] = uchar((ui[n] * m_chroma.blue[0] + ui[n + 1] * m_chroma.green[0] + ui[n + 2] * m_chroma.red[0]) * (256.0 / 4294967296.0));
        }
    }
}


ImageDecoder ExrDecoder::newDecoder() const
{
    initOpenEXR();
    return makePtr<ExrDecoder>();
}

/////////////////////// ExrEncoder ///////////////////


ExrEncoder::ExrEncoder()
{
    m_description = "OpenEXR Image files (*.exr)";
}


ExrEncoder::~ExrEncoder()
{
}


bool  ExrEncoder::isFormatSupported( int depth ) const
{
    return ( CV_MAT_DEPTH(depth) == CV_32F );
}


bool  ExrEncoder::write( const Mat& img, const std::vector<int>& params )
{
    int width = img.cols, height = img.rows;
    int depth = img.depth();
    CV_Assert( depth == CV_32F );
    int channels = img.channels();
    bool result = false;
    Header header( width, height );
    Imf::PixelType type = FLOAT;

    for( size_t i = 0; i < params.size(); i += 2 )
    {
        if( params[i] == IMWRITE_EXR_TYPE )
        {
            switch( params[i+1] )
            {
            case IMWRITE_EXR_TYPE_HALF:
                type = HALF;
                break;
            case IMWRITE_EXR_TYPE_FLOAT:
                type = FLOAT;
                break;
            default:
                CV_Error(Error::StsBadArg, "IMWRITE_EXR_TYPE is invalid or not supported");
            }
        }
        if ( params[i] == IMWRITE_EXR_COMPRESSION )
        {
            switch ( params[i + 1] )
            {
            case IMWRITE_EXR_COMPRESSION_NO:
                header.compression() = NO_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_RLE:
                header.compression() = RLE_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_ZIPS:
                header.compression() = ZIPS_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_ZIP:
                header.compression() = ZIP_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_PIZ:
                header.compression() = PIZ_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_PXR24:
                header.compression() = PXR24_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_B44:
                header.compression() = B44_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_B44A:
                header.compression() = B44A_COMPRESSION;
                break;
// version macros introduced in openexr 2.0.1.
// - https://github.com/AcademySoftwareFoundation/openexr/commit/60cdff8a6f5c4e25a374e5f366d6e9b4efd869b3#diff-c4bae0726aebe410e407db9abd406d9cf2684f82dd8a08f46d84e8b7c35cf22aR67
#if defined(OPENEXR_VERSION_MAJOR) && defined(OPENEXR_VERSION_MINOR) && OPENEXR_VERSION_MAJOR * 1000 + OPENEXR_VERSION_MINOR >= 2 * 1000 + 2
            // available since version 2.2.0
            case IMWRITE_EXR_COMPRESSION_DWAA:
                header.compression() = DWAA_COMPRESSION;
                break;
            case IMWRITE_EXR_COMPRESSION_DWAB:
                header.compression() = DWAB_COMPRESSION;
                break;
#endif
            default:
                CV_Error(Error::StsBadArg, "IMWRITE_EXR_COMPRESSION is invalid or not supported");
            }
        }
        if (params[i] == IMWRITE_EXR_DWA_COMPRESSION_LEVEL)
        {
#if !defined(OPENEXR_VERSION_MAJOR)
            CV_LOG_ONCE_WARNING(NULL, "Setting `IMWRITE_EXR_DWA_COMPRESSION_LEVEL` not supported in unknown OpenEXR version possibly prior to 2.0.1 (version 3 is required)");
#elif OPENEXR_VERSION_MAJOR < 3
            CV_LOG_ONCE_WARNING(NULL, "Setting `IMWRITE_EXR_DWA_COMPRESSION_LEVEL` not supported in OpenEXR version " + std::to_string(OPENEXR_VERSION_MAJOR) + " (version 3 is required)");
#else
            header.dwaCompressionLevel() = params[i + 1];
#endif
        }
    }

    if( channels == 3 || channels == 4 )
    {
        header.channels().insert( "R", Channel( type ) );
        header.channels().insert( "G", Channel( type ) );
        header.channels().insert( "B", Channel( type ) );
        //printf("bunt\n");
    }
    else
    {
        header.channels().insert( "Y", Channel( type ) );
        //printf("gray\n");
    }

    if( channels % 2 == 0 )
    { // even number of channels indicates Alpha
        header.channels().insert( "A", Channel( type ) );
    }

    OutputFile file( m_filename.c_str(), header );

    FrameBuffer frame;

    char *buffer;
    size_t bufferstep;
    int size;
    Mat exrMat;
    if( type == HALF )
    {
        img.convertTo(exrMat, CV_16F);
        buffer = (char *)const_cast<uchar *>( exrMat.ptr() );
        bufferstep = exrMat.step;
        size = 2;
    }
    else
    {
        buffer = (char *)const_cast<uchar *>( img.ptr() );
        bufferstep = img.step;
        size = 4;
    }

    if( channels == 3 || channels == 4 )
    {
        frame.insert( "B", Slice( type, buffer, size * channels, bufferstep ));
        frame.insert( "G", Slice( type, buffer + size, size * channels, bufferstep ));
        frame.insert( "R", Slice( type, buffer + size * 2, size * channels, bufferstep ));
    }
    else
        frame.insert( "Y", Slice( type, buffer, size * channels, bufferstep ));

    if( channels % 2 == 0 )
    { // even channel count indicates Alpha channel
        frame.insert( "A", Slice( type, buffer + size * (channels - 1), size * channels, bufferstep ));
    }

    file.setFrameBuffer( frame );

    result = true;
    try
    {
        file.writePixels( height );
    }
    catch(...)
    {
        result = false;
    }

    return result;
}


ImageEncoder ExrEncoder::newEncoder() const
{
    initOpenEXR();
    return makePtr<ExrEncoder>();
}

}

#endif

/* End of file. */
