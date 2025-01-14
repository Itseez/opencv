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

#ifdef HAVE_WEBP

#include "precomp.hpp"

#include <stdio.h>
#include <limits.h>
#include "grfmt_webp.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include <webp/decode.h>
#include <webp/encode.h>
#include <webp/demux.h>
#include <webp/mux.h>

namespace cv
{

// 64Mb limit to avoid memory DDOS
static size_t param_maxFileSize = utils::getConfigurationParameterSizeT("OPENCV_IMGCODECS_WEBP_MAX_FILE_SIZE", 64*1024*1024);

static const size_t WEBP_HEADER_SIZE = 32;

WebPDecoder::WebPDecoder()
{
    m_buf_supported = true;
    fs_size = 0;
    m_has_animation = false;
    m_previous_timestamp = 0;
}

WebPDecoder::~WebPDecoder() {}

void WebPDecoder::UniquePtrDeleter::operator()(WebPAnimDecoder* decoder) const
{
    WebPAnimDecoderDelete(decoder);
}

size_t WebPDecoder::signatureLength() const
{
    return WEBP_HEADER_SIZE;
}

bool WebPDecoder::checkSignature(const String & signature) const
{
    bool ret = false;

    if(signature.size() >= WEBP_HEADER_SIZE)
    {
        WebPBitstreamFeatures features;
        if(VP8_STATUS_OK == WebPGetFeatures((uint8_t *)signature.c_str(),
                                            WEBP_HEADER_SIZE, &features))
        {
            ret = true;
        }
    }

    return ret;
}

ImageDecoder WebPDecoder::newDecoder() const
{
    return makePtr<WebPDecoder>();
}

bool WebPDecoder::readHeader()
{
    if (m_has_animation)
    {
        return true;
    }

    uint8_t header[WEBP_HEADER_SIZE] = { 0 };
    if (m_buf.empty())
    {
        fs.open(m_filename.c_str(), std::ios::binary);
        fs.seekg(0, std::ios::end);
        fs_size = safeCastToSizeT(fs.tellg(), "File is too large");
        fs.seekg(0, std::ios::beg);
        CV_Assert(fs && "File stream error");
        CV_CheckGE(fs_size, WEBP_HEADER_SIZE, "File is too small");
        CV_CheckLE(fs_size, param_maxFileSize, "File is too large. Increase OPENCV_IMGCODECS_WEBP_MAX_FILE_SIZE parameter if you want to process large files");

        fs.read((char*)header, sizeof(header));
        CV_Assert(fs && "Can't read WEBP_HEADER_SIZE bytes");
    }
    else
    {
        CV_CheckGE(m_buf.total(), WEBP_HEADER_SIZE, "Buffer is too small");
        memcpy(header, m_buf.ptr(), sizeof(header));
        data = m_buf;
    }

    WebPBitstreamFeatures features;
    if (VP8_STATUS_OK < WebPGetFeatures(header, sizeof(header), &features)) return false;

    m_has_animation = features.has_animation == 1;

    if (m_has_animation)
    {
        if (m_buf.empty())
        {
            fs.seekg(0, std::ios::beg); CV_Assert(fs && "File stream error");
            data.create(1, validateToInt(fs_size), CV_8UC1);
            fs.read((char*)data.ptr(), fs_size);
            CV_Assert(fs && "Can't read file data");
            fs.close();
        }

        CV_Assert(data.type() == CV_8UC1); CV_Assert(data.rows == 1);

        WebPData webp_data;
        webp_data.bytes = (const uint8_t*)data.ptr();
        webp_data.size = data.total();

        WebPAnimDecoderOptions dec_options;
        WebPAnimDecoderOptionsInit(&dec_options);

        dec_options.color_mode = m_use_rgb ? MODE_RGBA : MODE_BGRA;
        anim_decoder.reset(WebPAnimDecoderNew(&webp_data, &dec_options));
        CV_Assert(anim_decoder.get() && "Error parsing image");

        WebPAnimInfo anim_info;
        WebPAnimDecoderGetInfo(anim_decoder.get(), &anim_info);
        m_animation.loop_count = anim_info.loop_count;

        m_animation.bgcolor[0] = (anim_info.bgcolor >> 24) & 0xFF;
        m_animation.bgcolor[1] = (anim_info.bgcolor >> 16) & 0xFF;
        m_animation.bgcolor[2] = (anim_info.bgcolor >> 8) & 0xFF;
        m_animation.bgcolor[3] = anim_info.bgcolor & 0xFF;
        m_frame_count = anim_info.frame_count;
    }
    m_width = features.width;
    m_height = features.height;
    m_type = features.has_alpha ? CV_8UC4 : CV_8UC3;

    return true;
}

bool WebPDecoder::readData(Mat &img)
{
    CV_CheckGE(m_width, 0, ""); CV_CheckGE(m_height, 0, "");

    CV_CheckEQ(img.cols, m_width, "");
    CV_CheckEQ(img.rows, m_height, "");

    if (data.empty())
    {
        fs.seekg(0, std::ios::beg); CV_Assert(fs && "File stream error");
        data.create(1, validateToInt(fs_size), CV_8UC1);
        fs.read((char*)data.ptr(), fs_size);
        CV_Assert(fs && "Can't read file data");
        fs.close();
    }
    CV_Assert(data.type() == CV_8UC1); CV_Assert(data.rows == 1);

    Mat read_img;
    CV_CheckType(img.type(), img.type() == CV_8UC1 || img.type() == CV_8UC3 || img.type() == CV_8UC4, "");
    if (img.type() != m_type || img.cols != m_width || img.rows != m_height)
    {
        read_img.create(m_height, m_width, m_type);
    }
    else
    {
        read_img = img;  // copy header
    }

    uchar* out_data = read_img.ptr();
    size_t out_data_size = read_img.dataend - out_data;

    uchar* res_ptr = NULL;

    if (m_has_animation)
    {
        uint8_t* buf;
        int timestamp;

        WebPAnimDecoderGetNext(anim_decoder.get(), &buf, &timestamp);
        Mat tmp(Size(m_width, m_height), CV_8UC4, buf);

        if (img.type() == CV_8UC1)
        {
            cvtColor(tmp, img, COLOR_BGR2GRAY);
        }
        else
        if (img.type() == CV_8UC3)
        {
            cvtColor(tmp, img, COLOR_BGRA2BGR);
        }
        else
            tmp.copyTo(img);

        m_animation.durations.push_back(timestamp - m_previous_timestamp);
        m_previous_timestamp = timestamp;
        return true;
    }

    if (m_type == CV_8UC3)
    {
        CV_CheckTypeEQ(read_img.type(), CV_8UC3, "");
        if (m_use_rgb)
            res_ptr = WebPDecodeRGBInto(data.ptr(), data.total(), out_data,
                (int)out_data_size, (int)read_img.step);
        else
            res_ptr = WebPDecodeBGRInto(data.ptr(), data.total(), out_data,
                (int)out_data_size, (int)read_img.step);
    }
    else if (m_type == CV_8UC4)
    {
        CV_CheckTypeEQ(read_img.type(), CV_8UC4, "");
        if (m_use_rgb)
            res_ptr = WebPDecodeRGBAInto(data.ptr(), data.total(), out_data,
                (int)out_data_size, (int)read_img.step);
        else
            res_ptr = WebPDecodeBGRAInto(data.ptr(), data.total(), out_data,
                (int)out_data_size, (int)read_img.step);
    }

    if (res_ptr != out_data)
        return false;

    if (read_img.data == img.data && img.type() == m_type)
    {
        // nothing
    }
    else if (img.type() == CV_8UC1)
    {
        cvtColor(read_img, img, COLOR_BGR2GRAY);
    }
    else if (img.type() == CV_8UC3 && m_type == CV_8UC4)
    {
        cvtColor(read_img, img, COLOR_BGRA2BGR);
    }
    else
    {
        CV_Error(Error::StsInternal, "");
    }
    return true;
}

bool WebPDecoder::nextPage()
{
    // Prepare the next page, if any.
    return WebPAnimDecoderHasMoreFrames(anim_decoder.get()) > 0;
}

WebPEncoder::WebPEncoder()
{
    m_description = "WebP files (*.webp)";
    m_buf_supported = true;
}

WebPEncoder::~WebPEncoder() { }

ImageEncoder WebPEncoder::newEncoder() const
{
    return makePtr<WebPEncoder>();
}

bool WebPEncoder::write(const Mat& img, const std::vector<int>& params)
{
    CV_CheckDepthEQ(img.depth(), CV_8U, "WebP codec supports 8U images only");

    const int width = img.cols, height = img.rows;

    bool comp_lossless = true;
    float quality = 100.0f;

    if (params.size() > 1)
    {
        if (params[0] == IMWRITE_WEBP_QUALITY)
        {
            comp_lossless = false;
            quality = static_cast<float>(params[1]);
            if (quality < 1.0f)
            {
                quality = 1.0f;
            }
            if (quality > 100.0f)
            {
                comp_lossless = true;
            }
        }
    }

    int channels = img.channels();
    CV_Check(channels, channels == 1 || channels == 3 || channels == 4, "");

    const Mat *image = &img;
    Mat temp;

    if (channels == 1)
    {
        cvtColor(*image, temp, COLOR_GRAY2BGR);
        image = &temp;
        channels = 3;
    }

    uint8_t *out = NULL;
    size_t size = 0;
    if (comp_lossless)
    {
        if (channels == 3)
        {
            size = WebPEncodeLosslessBGR(image->ptr(), width, height, (int)image->step, &out);
        }
        else if (channels == 4)
        {
            size = WebPEncodeLosslessBGRA(image->ptr(), width, height, (int)image->step, &out);
        }
    }
    else
    {
        if (channels == 3)
        {
            size = WebPEncodeBGR(image->ptr(), width, height, (int)image->step, quality, &out);
        }
        else if (channels == 4)
        {
            size = WebPEncodeBGRA(image->ptr(), width, height, (int)image->step, quality, &out);
        }
    }
#if WEBP_DECODER_ABI_VERSION >= 0x0206
    Ptr<uint8_t> out_cleaner(out, WebPFree);
#else
    Ptr<uint8_t> out_cleaner(out, free);
#endif

    CV_Assert(size > 0);
    size_t bytes_written = 0;
    if (m_buf)
    {
        m_buf->resize(size);
        memcpy(&(*m_buf)[0], out, size);
        bytes_written = size;
    }
    else
    {
        FILE *fd = fopen(m_filename.c_str(), "wb");
        if (fd != NULL)
        {
            bytes_written = fwrite(out, sizeof(uint8_t), size, fd);
            if (size != bytes_written)
            {
                CV_LOG_ERROR(NULL, cv::format("Only %zu or %zu bytes are written\n",bytes_written, size));
            }
            fclose(fd); fd = NULL;
        }
    }

    return (size > 0) && (bytes_written == size);
}

bool WebPEncoder::writeanimation(const Animation& animation, const std::vector<int>& params)
{
    CV_CheckDepthEQ(animation.frames[0].depth(), CV_8U, "WebP codec supports only 8-bit unsigned images");
    int ok = 0;
    int timestamp = 0;
    const int width = animation.frames[0].cols, height = animation.frames[0].rows;

    WebPAnimEncoderOptions anim_config;
    WebPConfig config;
    WebPPicture pic;
    WebPData webp_data;

    WebPDataInit(&webp_data);
    if (!WebPAnimEncoderOptionsInit(&anim_config) ||
        !WebPConfigInit(&config) ||
        !WebPPictureInit(&pic)) {
        CV_LOG_ERROR(NULL, "Library version mismatch!\n");
        WebPDataClear(&webp_data);
        return false;
    }

    int bgvalue = (static_cast<int>(animation.bgcolor[0]) & 0xFF) << 24 |
        (static_cast<int>(animation.bgcolor[1]) & 0xFF) << 16 |
        (static_cast<int>(animation.bgcolor[2]) & 0xFF) << 8 |
        (static_cast<int>(animation.bgcolor[3]) & 0xFF);

    anim_config.anim_params.bgcolor = bgvalue;
    anim_config.anim_params.loop_count = animation.loop_count;

    if (params.size() > 1)
    {
        if (params[0] == IMWRITE_WEBP_QUALITY)
        {
            config.lossless = 0;
            config.quality = static_cast<float>(params[1]);
            if (config.quality < 1.0f)
            {
                config.quality = 1.0f;
            }
            if (config.quality >= 100.0f)
            {
                config.lossless = 1;
            }
        }
        anim_config.minimize_size = 0;
    }

    std::unique_ptr<WebPAnimEncoder, void (*)(WebPAnimEncoder*)> anim_encoder(
        WebPAnimEncoderNew(width, height, &anim_config), WebPAnimEncoderDelete);

    pic.width = width;
    pic.height = height;
    pic.use_argb = 1;
    pic.argb_stride = width;
    WebPEncode(&config, &pic);

    bool is_input_rgba = animation.frames[0].channels() == 4;
    Size canvas_size = Size(animation.frames[0].cols,animation.frames[0].rows);

    for (size_t i = 0; i < animation.frames.size(); i++)
    {
        Mat argb;
        CV_Assert(canvas_size == Size(animation.frames[i].cols,animation.frames[i].rows));
        if (is_input_rgba)
            pic.argb = (uint32_t*)animation.frames[i].data;
        else
        {
            cvtColor(animation.frames[i], argb, COLOR_BGR2BGRA);
            pic.argb = (uint32_t*)argb.data;
        }
        ok = WebPAnimEncoderAdd(anim_encoder.get(), &pic, timestamp, &config);
        timestamp += animation.durations[i];
    }

    // add a last fake frame to signal the last duration
    ok = ok & WebPAnimEncoderAdd(anim_encoder.get(), NULL, timestamp, NULL);
    ok = ok & WebPAnimEncoderAssemble(anim_encoder.get(), &webp_data);

    size_t bytes_written = 0;
    if (ok)
    {
        if (m_buf)
        {
            m_buf->resize(webp_data.size);
            memcpy(&(*m_buf)[0], webp_data.bytes, webp_data.size);
            bytes_written = webp_data.size;
        }
        else
        {
            FILE* fd = fopen(m_filename.c_str(), "wb");
            if (fd != NULL)
            {
                bytes_written = fwrite(webp_data.bytes, sizeof(uint8_t), webp_data.size, fd);
                if (webp_data.size != bytes_written)
                {
                    CV_LOG_ERROR(NULL, cv::format("Only %zu or %zu bytes are written\n",bytes_written, webp_data.size));
                }
                fclose(fd); fd = NULL;
            }
        }
    }

    bool status = (ok > 0) && (webp_data.size == bytes_written);

    // free resources
    WebPDataClear(&webp_data);

    return status;
}

}

#endif
