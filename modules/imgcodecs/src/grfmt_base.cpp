/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "grfmt_base.hpp"
#include "bitstrm.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv
{

BaseImageDecoder::BaseImageDecoder()
{
    m_width = m_height = 0;
    m_type = -1;
    m_buf_supported = false;
    m_scale_denom = 1;
    m_use_rgb = false;
    m_frame_count = 1;
}


ExifEntry_t BaseImageDecoder::getExifTag(const ExifTagName tag) const
{
    return m_exif.getTag(tag);
}
bool BaseImageDecoder::setSource( const String& filename )
{
    m_filename = filename;
    m_buf.release();
    return true;
}

bool BaseImageDecoder::setSource( const Mat& buf )
{
    if( !m_buf_supported )
        return false;
    m_filename = String();
    m_buf = buf;
    return true;
}

size_t BaseImageDecoder::signatureLength() const
{
    return m_signature.size();
}

bool BaseImageDecoder::checkSignature( const String& signature ) const
{
    size_t len = signatureLength();
    return signature.size() >= len && memcmp( signature.c_str(), m_signature.c_str(), len ) == 0;
}

int BaseImageDecoder::setScale( const int& scale_denom )
{
    int temp = m_scale_denom;
    m_scale_denom = scale_denom;
    return temp;
}

void BaseImageDecoder::setRGB(bool useRGB)
{
    m_use_rgb = useRGB;
}

ImageDecoder BaseImageDecoder::newDecoder() const
{
    return ImageDecoder();
}

BaseImageEncoder::BaseImageEncoder()
{
    m_buf = 0;
    m_buf_supported = false;
}

bool  BaseImageEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U;
}

String BaseImageEncoder::getDescription() const
{
    return m_description;
}

bool BaseImageEncoder::setDestination( const String& filename )
{
    m_filename = filename;
    m_buf = 0;
    return true;
}

bool BaseImageEncoder::setDestination( std::vector<uchar>& buf )
{
    if( !m_buf_supported )
        return false;
    m_buf = &buf;
    m_buf->clear();
    m_filename = String();
    return true;
}

bool BaseImageEncoder::writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params)
{
    if(img_vec.size() > 1)
        CV_LOG_INFO(NULL, "Multi page image will be written as animation with 1 second frame duration.");

    Animation animation;
    animation.frames = img_vec;

    for (size_t i = 0; i < animation.frames.size(); i++)
    {
        animation.durations.push_back(1000);
    }
    return writeanimation(animation, params);
}

bool BaseImageEncoder::writeanimation(const Animation&, const std::vector<int>& )
{
    return false;
}

ImageEncoder BaseImageEncoder::newEncoder() const
{
    return ImageEncoder();
}

void BaseImageEncoder::throwOnEror() const
{
    if(!m_last_error.empty())
    {
        String msg = "Raw image encoder error: " + m_last_error;
        CV_Error( Error::BadImageSize, msg.c_str() );
    }
}

}

/* End of file. */
