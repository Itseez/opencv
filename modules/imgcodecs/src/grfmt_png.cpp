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

#ifdef HAVE_PNG

/****************************************************************************************\
    This part of the file implements PNG codec on base of libpng library,
    in particular, this code is based on example.c from libpng
    (see otherlibs/_graphics/readme.txt for copyright notice)
    and png2bmp sample from libpng distribution (Copyright (C) 1999-2001 MIYASAKA Masaru)
\****************************************************************************************/

/****************************************************************************\
 *
 *  this file includes some modified part of apngasm and APNG Optimizer 1.4
 *  both have zlib license.
 *
 ****************************************************************************/


 /*  apngasm
 *
 *  The next generation of apngasm, the APNG Assembler.
 *  The apngasm CLI tool and library can assemble and disassemble APNG image files.
 *
 *  https://github.com/apngasm/apngasm


 * APNG Optimizer 1.4
 *
 * Makes APNG files smaller.
 *
 * http://sourceforge.net/projects/apng/files
 *
 * Copyright (c) 2011-2015 Max Stepin
 * maxst at users.sourceforge.net
 *
 * zlib license
 * ------------
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#ifndef _LFS64_LARGEFILE
#  define _LFS64_LARGEFILE 0
#endif
#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 0
#endif

#include "grfmt_png.hpp"
#include <opencv2/core/utils/logger.hpp>

#if defined _MSC_VER && _MSC_VER >= 1200
    // interaction between '_setjmp' and C++ object destruction is non-portable
    #pragma warning( disable: 4611 )
    #pragma warning( disable: 4244 )
#endif

// the following defines are a hack to avoid multiple problems with frame pointer handling and setjmp
// see http://gcc.gnu.org/ml/gcc/2011-10/msg00324.html for some details
#define mingw_getsp(...) 0
#define __builtin_frame_address(...) 0

namespace cv
{

const uint32_t id_IHDR = 0x52444849; // PNG header
const uint32_t id_acTL = 0x4C546361; // Animation control chunk
const uint32_t id_fcTL = 0x4C546366; // Frame control chunk
const uint32_t id_IDAT = 0x54414449; // first frame and/or default image
const uint32_t id_fdAT = 0x54416466; // Frame data chunk
const uint32_t id_PLTE = 0x45544C50;
const uint32_t id_bKGD = 0x44474B62;
const uint32_t id_tRNS = 0x534E5274;
const uint32_t id_IEND = 0x444E4549; // end/footer chunk

APNGFrame::APNGFrame()
{
    _pixels = NULL;
    _width = 0;
    _height = 0;
    _colorType = 0;
    _paletteSize = 0;
    _transparencySize = 0;
    _delayNum = 1;
    _delayDen = 1000;
}

APNGFrame::~APNGFrame() {}

bool APNGFrame::setMat(const cv::Mat& src, unsigned delayNum, unsigned delayDen)
{
    _delayNum = delayNum;
    _delayDen = delayDen;

    if (!src.empty())
    {
        png_uint_32 rowbytes = src.depth() == CV_16U ? src.cols * src.channels() * 2 : src.cols * src.channels();
        _width = src.cols;
        _height = src.rows;
        _colorType = src.channels() == 1 ? PNG_COLOR_TYPE_GRAY : src.channels() == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA;
        _pixels = src.data;
        _rows.resize(_height);

        for (unsigned int i = 0; i < _height; ++i)
            _rows[i] = _pixels + i * rowbytes;
        return true;
    }
    return false;
}

void APNGFrame::setWidth(unsigned int width) { _width = width; }
void APNGFrame::setHeight(unsigned int height) { _height = height;}
void APNGFrame::setColorType(unsigned char colorType) { _colorType = colorType; }
void APNGFrame::setPalette(const rgb* palette) { std::copy(palette, palette + 256, _palette); }
void APNGFrame::setTransparency(const unsigned char* transparency) { std::copy(transparency, transparency + 256, _transparency); }
void APNGFrame::setPaletteSize(int paletteSize) { _paletteSize = paletteSize; }
void APNGFrame::setTransparencySize(int transparencySize) { _transparencySize = transparencySize; }
void APNGFrame::setDelayNum(unsigned int delayNum) { _delayNum = delayNum; }
void APNGFrame::setDelayDen(unsigned int delayDen) { _delayDen = delayDen; }
void APNGFrame::setPixels(unsigned char* pixels) { _pixels = pixels; }

PngDecoder::PngDecoder()
{
    m_signature = "\x89\x50\x4e\x47\xd\xa\x1a\xa";
    m_color_type = 0;
    m_f = 0;
    m_buf_supported = true;
    m_buf_pos = 0;
    m_bit_depth = 0;
    m_frame_no = 0;
    w0 = 0;
    h0 = 0;
    x0 = 0;
    y0 = 0;
    delay_num = 0;
    delay_den = 0;
    dop = 0;
    bop = 0;
}

PngDecoder::~PngDecoder()
{
    if( m_f )
    {
        fclose( m_f );
        m_f = nullptr;
    }
}

ImageDecoder PngDecoder::newDecoder() const
{
    return makePtr<PngDecoder>();
}

void  PngDecoder::readDataFromBuf( void* _png_ptr, unsigned char* dst, size_t size )
{
    png_structp png_ptr = (png_structp)_png_ptr;
    PngDecoder* decoder = (PngDecoder*)(png_get_io_ptr(png_ptr));
    CV_Assert( decoder );
    const Mat& buf = decoder->m_buf;
    if( decoder->m_buf_pos + size > buf.cols*buf.rows*buf.elemSize() )
    {
        png_error(png_ptr, "PNG input buffer is incomplete");
        return;
    }
    memcpy( dst, decoder->m_buf.ptr() + decoder->m_buf_pos, size );
    decoder->m_buf_pos += size;
}

bool  PngDecoder::readHeader()
{
    volatile bool result = false;

    PngPtrs png_ptrs;
    png_structp png_ptr = png_ptrs.getPng();
    png_infop info_ptr = png_ptrs.getInfo();
    png_infop end_info = png_ptrs.getEndInfo();

    if( png_ptr && info_ptr && end_info )
    {
        m_buf_pos = 0;
        if( setjmp( png_jmpbuf( png_ptr ) ) == 0 )
        {
            unsigned char sig[8];
            uint32_t id = 0;
            Chunk chunk;

            if( !m_buf.empty() )
                png_set_read_fn(png_ptr, this, (png_rw_ptr)readDataFromBuf );
            else
            {
                m_f = fopen(m_filename.c_str(), "rb");
                if (!m_f)
                {
                    return false;
                }
                png_init_io(png_ptr, m_f);
            }

            // Read PNG header: 137 80 78 71 13 10 26 10
            if (!read_from_io(&sig, 8))
                return false;

            id = read_chunk(m_chunkIHDR);
            // 8=HDR+size, 13=size of IHDR chunk, 4=CRC
            // http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html#C.IHDR
            if (!(id == id_IHDR && m_chunkIHDR.p.size() == 8 + 13 + 4))
            {
                return false;
            }

            while (true)
            {
                m_is_fcTL_loaded = false;
                id = read_chunk(chunk);

                if (!id || (m_f && feof(m_f)) || (!m_buf.empty() && m_buf_pos > m_buf.total()))
                {
                    return false;
                }

                if (id == id_IDAT)
                {
                    if (m_f)
                        fseek(m_f, 0, SEEK_SET);
                    else
                        m_buf_pos = 0;
                    break;
                }

                if (id == id_acTL)
                {
                    // 8=HDR+size, 8=size of acTL chunk, 4=CRC
                    // https://wiki.mozilla.org/APNG_Specification#%60acTL%60:_The_Animation_Control_Chunk
                    if (chunk.p.size() != 8 + 8 + 4)
                        return false;
                    m_animation.loop_count = png_get_uint_32(&chunk.p[12]);

                    m_frame_count = png_get_uint_32(&chunk.p[8]);
                    if (m_frame_count == 0)
                        return false;
                }

                if (id == id_fcTL)
                {
                    // 8=HDR+size, 26=size of fcTL chunk, 4=CRC
                    // https://wiki.mozilla.org/APNG_Specification#%60fcTL%60:_The_Frame_Control_Chunk
                    if (chunk.p.size() != 8 + 26 + 4)
                        return false;
                    m_is_fcTL_loaded = true;
                    w0 = png_get_uint_32(&chunk.p[12]);
                    h0 = png_get_uint_32(&chunk.p[16]);
                    x0 = png_get_uint_32(&chunk.p[20]);
                    y0 = png_get_uint_32(&chunk.p[24]);
                    delay_num = png_get_uint_16(&chunk.p[28]);
                    delay_den = png_get_uint_16(&chunk.p[30]);
                    dop = chunk.p[32];
                    bop = chunk.p[33];
                }

                if (id == id_bKGD)
                {
                    // 8=HDR+size, ??=size of bKGD chunk, 4=CRC
                    // The spec is actually more complex: http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html#C.bKGD
                    // TODO: we only check that 4 bytes can be read from &chunk.p[8]. Fix.
                    if (chunk.p.size() < 8 + 4)
                        return false;
                    int bgcolor = png_get_uint_32(&chunk.p[8]);
                    m_animation.bgcolor[3] = (bgcolor >> 24) & 0xFF;
                    m_animation.bgcolor[2] = (bgcolor >> 16) & 0xFF;
                    m_animation.bgcolor[1] = (bgcolor >> 8) & 0xFF;
                    m_animation.bgcolor[0] = bgcolor & 0xFF;
                }

                if (id == id_PLTE || id == id_tRNS)
                    m_chunksInfo.push_back(chunk);
            }

            png_uint_32 wdth, hght;
            int bit_depth, color_type, num_trans=0;
            png_bytep trans;
            png_color_16p trans_values;

            // Free chunk in case png_read_info uses longjmp.
            chunk.p.clear();
            chunk.p.shrink_to_fit();

            png_read_info( png_ptr, info_ptr );
            png_get_IHDR(png_ptr, info_ptr, &wdth, &hght,
                &bit_depth, &color_type, 0, 0, 0);

            m_width = (int)wdth;
            m_height = (int)hght;
            m_color_type = color_type;
            m_bit_depth = bit_depth;

            if (bit_depth <= 8 || bit_depth == 16)
            {
                switch (color_type)
                {
                case PNG_COLOR_TYPE_RGB:
                case PNG_COLOR_TYPE_PALETTE:
                    png_get_tRNS(png_ptr, info_ptr, &trans, &num_trans, &trans_values);
                    if (num_trans > 0)
                        m_type = CV_8UC4;
                    else
                        m_type = CV_8UC3;
                    break;
                case PNG_COLOR_TYPE_GRAY_ALPHA:
                case PNG_COLOR_TYPE_RGB_ALPHA:
                    m_type = CV_8UC4;
                    break;
                default:
                    m_type = CV_8UC1;
                }
                if (bit_depth == 16)
                    m_type = CV_MAKETYPE(CV_16U, CV_MAT_CN(m_type));
                result = true;
            }
        }
    }

    if(result)
    {
        m_png_ptrs = std::move(png_ptrs);
    }

    return result;
}

bool  PngDecoder::readData( Mat& img )
{
    if (m_frame_count > 1)
    {
        Mat mat_cur = Mat::zeros(img.rows, img.cols, m_type);
        uint32_t id = 0;
        uint32_t j = 0;
        uint32_t imagesize = m_width * m_height * mat_cur.channels();
        m_is_IDAT_loaded = false;

        if (m_frame_no == 0)
        {
            m_mat_raw = Mat(img.rows, img.cols, m_type);
            m_mat_next = Mat(img.rows, img.cols, m_type);
            frameRaw.setMat(m_mat_raw);
            frameNext.setMat(m_mat_next);
            if (m_f)
                fseek(m_f, -8, SEEK_CUR);
            else
                m_buf_pos -= 8;
        }
        else
            m_mat_next.copyTo(mat_cur);

        frameCur.setMat(mat_cur);

        processing_start((void*)&frameRaw, mat_cur);
        png_structp png_ptr = m_png_ptrs.getPng();
        png_infop info_ptr = m_png_ptrs.getInfo();

        while (true)
        {
            Chunk chunk;
            id = read_chunk(chunk);
            if (!id)
                return false;

            if (id == id_fcTL && m_is_IDAT_loaded)
            {
                if (!m_is_fcTL_loaded)
                {
                    m_is_fcTL_loaded = true;
                    w0 = m_width;
                    h0 = m_height;
                }

                if (processing_finish())
                {
                    if (dop == 2)
                        memcpy(frameNext.getPixels(), frameCur.getPixels(), imagesize);

                    compose_frame(frameCur.getRows(), frameRaw.getRows(), bop, x0, y0, w0, h0, mat_cur);
                    if (!delay_den)
                        delay_den = 100;
                    m_animation.durations.push_back(cvRound(1000.*delay_num/delay_den));

                    if (mat_cur.channels() == img.channels())
                        mat_cur.copyTo(img);
                    else if (img.channels() == 1)
                        cvtColor(mat_cur, img, COLOR_BGRA2GRAY);
                    else if (img.channels() == 3)
                        cvtColor(mat_cur, img, COLOR_BGRA2BGR);

                    if (dop != 2)
                    {
                        memcpy(frameNext.getPixels(), frameCur.getPixels(), imagesize);
                        if (dop == 1)
                            for (j = 0; j < h0; j++)
                                memset(frameNext.getRows()[y0 + j] + x0 * img.channels(), 0, w0 * img.channels());
                    }
                }
                else
                {
                    return false;
                }

                w0 = png_get_uint_32(&chunk.p[12]);
                h0 = png_get_uint_32(&chunk.p[16]);
                x0 = png_get_uint_32(&chunk.p[20]);
                y0 = png_get_uint_32(&chunk.p[24]);
                delay_num = png_get_uint_16(&chunk.p[28]);
                delay_den = png_get_uint_16(&chunk.p[30]);
                dop = chunk.p[32];
                bop = chunk.p[33];

                if (int(x0 + w0) > img.cols || int(y0 + h0) > img.rows || dop > 2 || bop > 1)
                {
                    return false;
                }

                memcpy(&m_chunkIHDR.p[8], &chunk.p[12], 8);
                return true;
            }
            else if (id == id_IDAT)
            {
                m_is_IDAT_loaded = true;
                png_process_data(png_ptr, info_ptr, chunk.p.data(), chunk.p.size());
            }
            else if (id == id_fdAT && m_is_fcTL_loaded)
            {
                m_is_IDAT_loaded = true;
                png_save_uint_32(&chunk.p[4], static_cast<uint32_t>(chunk.p.size() - 16));
                memcpy(&chunk.p[8], "IDAT", 4);
                png_process_data(png_ptr, info_ptr, &chunk.p[4], chunk.p.size() - 4);
            }
            else if (id == id_IEND)
            {
                if (processing_finish())
                {
                    compose_frame(frameCur.getRows(), frameRaw.getRows(), bop, x0, y0, w0, h0, mat_cur);
                    if (!delay_den)
                        delay_den = 100;
                    m_animation.durations.push_back(cvRound(1000.*delay_num/delay_den));

                    if (mat_cur.channels() == img.channels())
                        mat_cur.copyTo(img);
                    else if (img.channels() == 1)
                        cvtColor(mat_cur, img, COLOR_BGRA2GRAY);
                    else if (img.channels() == 3)
                        cvtColor(mat_cur, img, COLOR_BGRA2BGR);
                }
                else
                    return false;

                return true;
            }
            else
                png_process_data(png_ptr, info_ptr, chunk.p.data(), chunk.p.size());
        }
        return false;
    }

    volatile bool result = false;
    AutoBuffer<unsigned char*> _buffer(m_height);
    unsigned char** buffer = _buffer.data();
    bool color = img.channels() > 1;

    png_structp png_ptr = m_png_ptrs.getPng();
    png_infop info_ptr = m_png_ptrs.getInfo();
    png_infop end_info = m_png_ptrs.getEndInfo();

    if( png_ptr && info_ptr && end_info && m_width && m_height )
    {
        if( setjmp( png_jmpbuf ( png_ptr ) ) == 0 )
        {
            int y;

            if( img.depth() == CV_8U && m_bit_depth == 16 )
                png_set_strip_16( png_ptr );
            else if( !isBigEndian() )
                png_set_swap( png_ptr );

            if(img.channels() < 4)
            {
                /* observation: png_read_image() writes 400 bytes beyond
                 * end of data when reading a 400x118 color png
                 * "mpplus_sand.png".  OpenCV crashes even with demo
                 * programs.  Looking at the loaded image I'd say we get 4
                 * bytes per pixel instead of 3 bytes per pixel.  Test
                 * indicate that it is a good idea to always ask for
                 * stripping alpha..  18.11.2004 Axel Walthelm
                 */
                 png_set_strip_alpha( png_ptr );
            } else
                png_set_tRNS_to_alpha( png_ptr );

            if( m_color_type == PNG_COLOR_TYPE_PALETTE )
                png_set_palette_to_rgb( png_ptr );

            if( (m_color_type & PNG_COLOR_MASK_COLOR) == 0 && m_bit_depth < 8 )
#if (PNG_LIBPNG_VER_MAJOR*10000 + PNG_LIBPNG_VER_MINOR*100 + PNG_LIBPNG_VER_RELEASE >= 10209) || \
    (PNG_LIBPNG_VER_MAJOR == 1 && PNG_LIBPNG_VER_MINOR == 0 && PNG_LIBPNG_VER_RELEASE >= 18)
                png_set_expand_gray_1_2_4_to_8( png_ptr );
#else
                png_set_gray_1_2_4_to_8( png_ptr );
#endif

            if( (m_color_type & PNG_COLOR_MASK_COLOR) && color && !m_use_rgb)
                png_set_bgr( png_ptr ); // convert RGB to BGR
            else if( color )
                png_set_gray_to_rgb( png_ptr ); // Gray->RGB
            else
                png_set_rgb_to_gray( png_ptr, 1, 0.299, 0.587 ); // RGB->Gray

            png_set_interlace_handling( png_ptr );
            png_read_update_info( png_ptr, info_ptr );

            for( y = 0; y < m_height; y++ )
                buffer[y] = img.data + y*img.step;

            png_read_image( png_ptr, buffer );
            png_read_end( png_ptr, end_info );

#ifdef PNG_eXIf_SUPPORTED
            png_uint_32 num_exif = 0;
            png_bytep exif = 0;

            // Exif info could be in info_ptr (intro_info) or end_info per specification
            if( png_get_valid(png_ptr, info_ptr, PNG_INFO_eXIf) )
                png_get_eXIf_1(png_ptr, info_ptr, &num_exif, &exif);
            else if( png_get_valid(png_ptr, end_info, PNG_INFO_eXIf) )
                png_get_eXIf_1(png_ptr, end_info, &num_exif, &exif);

            if( exif && num_exif > 0 )
            {
                m_exif.parseExif(exif, num_exif);
            }
#endif

            result = true;
        }
    }

    return result;
}

bool PngDecoder::nextPage() {
    return ++m_frame_no < (int)m_frame_count;
}

void PngDecoder::compose_frame(std::vector<png_bytep>& rows_dst, const std::vector<png_bytep>& rows_src, unsigned char _bop, uint32_t x, uint32_t y, uint32_t w, uint32_t h, Mat& img)
{
    int channels = img.channels();
    if (img.depth() == CV_16U)
        cv::parallel_for_(cv::Range(0, h), [&](const cv::Range& range) {
        for (int j = range.start; j < range.end; j++) {
            uint16_t* sp = reinterpret_cast<uint16_t*>(rows_src[j]);
            uint16_t* dp = reinterpret_cast<uint16_t*>(rows_dst[j + y]) + x * channels;

            if (_bop == 0) {
                // Overwrite mode: copy source row directly to destination
                memcpy(dp, sp, w * channels * sizeof(uint16_t));
            }
            else {
                // Blending mode
                for (unsigned int i = 0; i < w; i++, sp += channels, dp += channels) {
                    if (sp[3] == 65535) { // Fully opaque in 16-bit (max value)
                        memcpy(dp, sp, channels * sizeof(uint16_t));
                    }
                    else if (sp[3] != 0) { // Partially transparent
                        if (dp[3] != 0) { // Both source and destination have alpha
                            uint32_t u = sp[3] * 65535; // 16-bit max
                            uint32_t v = (65535 - sp[3]) * dp[3];
                            uint32_t al = u + v;
                            dp[0] = static_cast<uint16_t>((sp[0] * u + dp[0] * v) / al); // Red
                            dp[1] = static_cast<uint16_t>((sp[1] * u + dp[1] * v) / al); // Green
                            dp[2] = static_cast<uint16_t>((sp[2] * u + dp[2] * v) / al); // Blue
                            dp[3] = static_cast<uint16_t>(al / 65535);                  // Alpha
                        }
                        else {
                            // If destination alpha is 0, copy source pixel
                            memcpy(dp, sp, channels * sizeof(uint16_t));
                        }
                    }
                }
            }
        }
            });
    else
        cv::parallel_for_(cv::Range(0, h), [&](const cv::Range& range) {
        for (int j = range.start; j < range.end; j++) {
            unsigned char* sp = rows_src[j];
            unsigned char* dp = rows_dst[j + y] + x * channels;

            if (_bop == 0) {
                // Overwrite mode: copy source row directly to destination
                memcpy(dp, sp, w * channels);
            }
            else {
                // Blending mode
                for (unsigned int i = 0; i < w; i++, sp += channels, dp += channels) {
                    if (sp[3] == 255) {
                        // Fully opaque: copy source pixel directly
                        memcpy(dp, sp, channels);
                    }
                    else if (sp[3] != 0) {
                        // Alpha blending
                        if (dp[3] != 0) {
                            int u = sp[3] * 255;
                            int v = (255 - sp[3]) * dp[3];
                            int al = u + v;
                            dp[0] = (sp[0] * u + dp[0] * v) / al; // Red
                            dp[1] = (sp[1] * u + dp[1] * v) / al; // Green
                            dp[2] = (sp[2] * u + dp[2] * v) / al; // Blue
                            dp[3] = al / 255;                     // Alpha
                        }
                        else {
                            // If destination alpha is 0, copy source pixel
                            memcpy(dp, sp, channels);
                        }
                    }
                }
            }
        }
            });
}

bool PngDecoder::read_from_io(void* buffer, size_t num_bytes)
{
    if (m_f)
        return fread(buffer, 1, num_bytes, m_f) == num_bytes;

    if (m_buf_pos + num_bytes > m_buf.cols * m_buf.rows * m_buf.elemSize()) {
        CV_LOG_WARNING(NULL, "PNG input buffer is incomplete");
        return false;
    }

    memcpy( buffer, m_buf.ptr() + m_buf_pos, num_bytes );
    m_buf_pos += num_bytes;
    return true;
}

uint32_t PngDecoder::read_chunk(Chunk& chunk)
{
    unsigned char len[4];
    if (read_from_io(&len, 4))
    {
        const size_t size = static_cast<size_t>(png_get_uint_32(len)) + 12;
        if (size > PNG_USER_CHUNK_MALLOC_MAX)
        {
            CV_LOG_WARNING(NULL, "chunk data is too large");
            return 0;
        }
        chunk.p.resize(size);
        memcpy(chunk.p.data(), len, 4);
        if (read_from_io(&chunk.p[4], chunk.p.size() - 4))
            return *(uint32_t*)(&chunk.p[4]);
    }
    return 0;
}

bool PngDecoder::processing_start(void* frame_ptr, const Mat& img)
{
    static uint8_t header[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };

    PngPtrs png_ptrs;
    png_structp png_ptr = png_ptrs.getPng();
    png_infop info_ptr = png_ptrs.getInfo();

    if (!png_ptr || !info_ptr) {
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        return false;
    }

    m_png_ptrs = std::move(png_ptrs);
    png_set_crc_action(png_ptr, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);
    png_set_progressive_read_fn(png_ptr, frame_ptr, (png_progressive_info_ptr)info_fn, row_fn, NULL);

    if (img.channels() < 4)
        png_set_strip_alpha(png_ptr);
    else
        png_set_tRNS_to_alpha(png_ptr);

    png_process_data(png_ptr, info_ptr, header, 8);
    png_process_data(png_ptr, info_ptr, m_chunkIHDR.p.data(), m_chunkIHDR.p.size());

    if ((m_color_type & PNG_COLOR_MASK_COLOR) && img.channels() > 1 && !m_use_rgb)
        png_set_bgr(png_ptr); // convert RGB to BGR
    else if (img.channels() > 1)
        png_set_gray_to_rgb(png_ptr); // Gray->RGB
    else
        png_set_rgb_to_gray(png_ptr, 1, 0.299, 0.587); // RGB->Gray

    for (size_t i = 0; i < m_chunksInfo.size(); i++)
        png_process_data(png_ptr, info_ptr, m_chunksInfo[i].p.data(), m_chunksInfo[i].p.size());

    return true;
}

bool PngDecoder::processing_finish()
{
    static uint8_t footer[12] = { 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };

    png_structp png_ptr = m_png_ptrs.getPng();
    png_infop info_ptr = m_png_ptrs.getInfo();

    if (!png_ptr) {
        m_png_ptrs.clear();
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        m_png_ptrs.clear();
        return false;
    }

    png_process_data(png_ptr, info_ptr, footer, 12);
    m_png_ptrs.clear();

    return true;
}

void PngDecoder::info_fn(png_structp png_ptr, png_infop info_ptr)
{
    png_set_expand(png_ptr);
    (void)png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);
}

void PngDecoder::row_fn(png_structp png_ptr, png_bytep new_row, png_uint_32 row_num, int pass)
{
    CV_UNUSED(pass);
    APNGFrame* frame = (APNGFrame*)png_get_progressive_ptr(png_ptr);
    png_progressive_combine_row(png_ptr, frame->getRows()[row_num], new_row);
}

/////////////////////// PngEncoder ///////////////////

PngEncoder::PngEncoder()
{
    m_description = "Portable Network Graphics files (*.png)";
    m_buf_supported = true;
    op_zstream1.zalloc = NULL;
    op_zstream2.zalloc = NULL;
    next_seq_num = 0;
    trnssize = 0;
    palsize = 0;
    memset(palette, 0, sizeof(palette));
    memset(trns, 0, sizeof(trns));
    memset(op, 0, sizeof(op));
}

PngEncoder::~PngEncoder()
{
}

bool  PngEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U;
}

ImageEncoder PngEncoder::newEncoder() const
{
    return makePtr<PngEncoder>();
}

void PngEncoder::writeDataToBuf(void* _png_ptr, unsigned char* src, size_t size)
{
    if( size == 0 )
        return;
    png_structp png_ptr = (png_structp)_png_ptr;
    PngEncoder* encoder = (PngEncoder*)(png_get_io_ptr(png_ptr));
    CV_Assert( encoder && encoder->m_buf );
    size_t cursz = encoder->m_buf->size();
    encoder->m_buf->resize(cursz + size);
    memcpy( &(*encoder->m_buf)[cursz], src, size );
}

void PngEncoder::flushBuf(void*)
{
}

bool  PngEncoder::write( const Mat& img, const std::vector<int>& params )
{
    png_structp png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, 0, 0, 0 );
    png_infop info_ptr = 0;
    FILE * volatile f = 0;
    int y, width = img.cols, height = img.rows;
    int depth = img.depth(), channels = img.channels();
    volatile bool result = false;
    AutoBuffer<uchar*> buffer;

    if( depth != CV_8U && depth != CV_16U )
        return false;

    if( png_ptr )
    {
        info_ptr = png_create_info_struct( png_ptr );

        if( info_ptr )
        {
            if( setjmp( png_jmpbuf ( png_ptr ) ) == 0 )
            {
                if( m_buf )
                {
                    png_set_write_fn(png_ptr, this,
                        (png_rw_ptr)writeDataToBuf, (png_flush_ptr)flushBuf);
                }
                else
                {
                    f = fopen( m_filename.c_str(), "wb" );
                    if( f )
                        png_init_io( png_ptr, (png_FILE_p)f );
                }

                int compression_level = -1; // Invalid value to allow setting 0-9 as valid
                int compression_strategy = IMWRITE_PNG_STRATEGY_RLE; // Default strategy
                bool isBilevel = false;

                for( size_t i = 0; i < params.size(); i += 2 )
                {
                    if( params[i] == IMWRITE_PNG_COMPRESSION )
                    {
                        compression_strategy = IMWRITE_PNG_STRATEGY_DEFAULT; // Default strategy
                        compression_level = params[i+1];
                        compression_level = MIN(MAX(compression_level, 0), Z_BEST_COMPRESSION);
                    }
                    if( params[i] == IMWRITE_PNG_STRATEGY )
                    {
                        compression_strategy = params[i+1];
                        compression_strategy = MIN(MAX(compression_strategy, 0), Z_FIXED);
                    }
                    if( params[i] == IMWRITE_PNG_BILEVEL )
                    {
                        isBilevel = params[i+1] != 0;
                    }
                }

                if( m_buf || f )
                {
                    if( compression_level >= 0 )
                    {
                        png_set_compression_level( png_ptr, compression_level );
                    }
                    else
                    {
                        // tune parameters for speed
                        // (see http://wiki.linuxquestions.org/wiki/Libpng)
                        png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
                        png_set_compression_level(png_ptr, Z_BEST_SPEED);
                    }
                    png_set_compression_strategy(png_ptr, compression_strategy);

                    png_set_IHDR( png_ptr, info_ptr, width, height, depth == CV_8U ? isBilevel?1:8 : 16,
                        channels == 1 ? PNG_COLOR_TYPE_GRAY :
                        channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA,
                        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                        PNG_FILTER_TYPE_DEFAULT );

                    png_write_info( png_ptr, info_ptr );

                    if (isBilevel)
                        png_set_packing(png_ptr);

                    png_set_bgr( png_ptr );
                    if( !isBigEndian() )
                        png_set_swap( png_ptr );

                    buffer.allocate(height);
                    for( y = 0; y < height; y++ )
                        buffer[y] = img.data + y*img.step;

                    png_write_image( png_ptr, buffer.data() );
                    png_write_end( png_ptr, info_ptr );

                    result = true;
                }
            }
        }
    }

    png_destroy_write_struct( &png_ptr, &info_ptr );
    if(f) fclose( (FILE*)f );

    return result;
}

size_t PngEncoder::write_to_io(void const* _Buffer, size_t  _ElementSize, size_t _ElementCount, FILE * _Stream)
{
    if (_Stream)
        return fwrite(_Buffer, _ElementSize, _ElementCount, _Stream);

    size_t cursz = m_buf->size();
    m_buf->resize(cursz + _ElementCount);
    memcpy( &(*m_buf)[cursz], _Buffer, _ElementCount );
    return _ElementCount;
}

void PngEncoder::writeChunk(FILE* f, const char* name, unsigned char* data, uint32_t length)
{
    unsigned char buf[4];
    uint32_t crc = crc32(0, Z_NULL, 0);

    png_save_uint_32(buf, length);
    write_to_io(buf, 1, 4, f);
    write_to_io(name, 1, 4, f);
    crc = crc32(crc, (const Bytef*)name, 4);

    if (memcmp(name, "fdAT", 4) == 0)
    {
        png_save_uint_32(buf, next_seq_num++);
        write_to_io(buf, 1, 4, f);
        crc = crc32(crc, buf, 4);
        length -= 4;
    }

    if (data != NULL && length > 0)
    {
        write_to_io(data, 1, length, f);
        crc = crc32(crc, data, length);
    }

    png_save_uint_32(buf, crc);
    write_to_io(buf, 1, 4, f);
}

void PngEncoder::writeIDATs(FILE* f, int frame, unsigned char* data, uint32_t length, uint32_t idat_size)
{
    uint32_t z_cmf = data[0];
    if ((z_cmf & 0x0f) == 8 && (z_cmf & 0xf0) <= 0x70)
    {
        if (length >= 2)
        {
            uint32_t z_cinfo = z_cmf >> 4;
            uint32_t half_z_window_size = 1 << (z_cinfo + 7);
            while (idat_size <= half_z_window_size && half_z_window_size >= 256)
            {
                z_cinfo--;
                half_z_window_size >>= 1;
            }
            z_cmf = (z_cmf & 0x0f) | (z_cinfo << 4);
            if (data[0] != (unsigned char)z_cmf)
            {
                data[0] = (unsigned char)z_cmf;
                data[1] &= 0xe0;
                data[1] += (unsigned char)(0x1f - ((z_cmf << 8) + data[1]) % 0x1f);
            }
        }
    }

    while (length > 0)
    {
        uint32_t ds = length;
        if (ds > 32768)
            ds = 32768;

        if (frame == 0)
            writeChunk(f, "IDAT", data, ds);
        else
            writeChunk(f, "fdAT", data, ds + 4);

        data += ds;
        length -= ds;
    }
}

void PngEncoder::processRect(unsigned char* row, int rowbytes, int bpp, int stride, int h, unsigned char* rows)
{
    int i, j, v;
    int a, b, c, pa, pb, pc, p;
    unsigned char* prev = NULL;
    unsigned char* dp = rows;
    unsigned char* out;

    for (j = 0; j < h; j++)
    {
        uint32_t sum = 0;
        unsigned char* best_row = row_buf.data();
        uint32_t mins = ((uint32_t)(-1)) >> 1;

        out = row_buf.data() + 1;
        for (i = 0; i < rowbytes; i++)
        {
            v = out[i] = row[i];
            sum += (v < 128) ? v : 256 - v;
        }
        mins = sum;

        sum = 0;
        out = sub_row.data() + 1;
        for (i = 0; i < bpp; i++)
        {
            v = out[i] = row[i];
            sum += (v < 128) ? v : 256 - v;
        }
        for (i = bpp; i < rowbytes; i++)
        {
            v = out[i] = row[i] - row[i - bpp];
            sum += (v < 128) ? v : 256 - v;
            if (sum > mins)
                break;
        }
        if (sum < mins)
        {
            mins = sum;
            best_row = sub_row.data();
        }

        if (prev)
        {
            sum = 0;
            out = up_row.data() + 1;
            for (i = 0; i < rowbytes; i++)
            {
                v = out[i] = row[i] - prev[i];
                sum += (v < 128) ? v : 256 - v;
                if (sum > mins)
                    break;
            }
            if (sum < mins)
            {
                mins = sum;
                best_row = up_row.data();
            }

            sum = 0;
            out = avg_row.data() + 1;
            for (i = 0; i < bpp; i++)
            {
                v = out[i] = row[i] - prev[i] / 2;
                sum += (v < 128) ? v : 256 - v;
            }
            for (i = bpp; i < rowbytes; i++)
            {
                v = out[i] = row[i] - (prev[i] + row[i - bpp]) / 2;
                sum += (v < 128) ? v : 256 - v;
                if (sum > mins)
                    break;
            }
            if (sum < mins)
            {
                mins = sum;
                best_row = avg_row.data();
            }

            sum = 0;
            out = paeth_row.data() + 1;
            for (i = 0; i < bpp; i++)
            {
                v = out[i] = row[i] - prev[i];
                sum += (v < 128) ? v : 256 - v;
            }
            for (i = bpp; i < rowbytes; i++)
            {
                a = row[i - bpp];
                b = prev[i];
                c = prev[i - bpp];
                p = b - c;
                pc = a - c;
                pa = abs(p);
                pb = abs(pc);
                pc = abs(p + pc);
                p = (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b
                    : c;
                v = out[i] = row[i] - p;
                sum += (v < 128) ? v : 256 - v;
                if (sum > mins)
                    break;
            }
            if (sum < mins)
            {
                best_row = paeth_row.data();
            }
        }

        if (rows == NULL)
        {
            // deflate_rect_op()
            op_zstream1.next_in = row_buf.data();
            op_zstream1.avail_in = rowbytes + 1;
            deflate(&op_zstream1, Z_NO_FLUSH);

            op_zstream2.next_in = best_row;
            op_zstream2.avail_in = rowbytes + 1;
            deflate(&op_zstream2, Z_NO_FLUSH);
        }
        else
        {
            // deflate_rect_fin()
            memcpy(dp, best_row, rowbytes + 1);
            dp += rowbytes + 1;
        }

        prev = row;
        row += stride;
    }
}

void PngEncoder::deflateRectOp(unsigned char* pdata, int x, int y, int w, int h, int bpp, int stride, int zbuf_size, int n)
{
    unsigned char* row = pdata + y * stride + x * bpp;
    int rowbytes = w * bpp;

    op_zstream1.data_type = Z_BINARY;
    op_zstream1.next_out = op_zbuf1.data();
    op_zstream1.avail_out = zbuf_size;

    op_zstream2.data_type = Z_BINARY;
    op_zstream2.next_out = op_zbuf2.data();
    op_zstream2.avail_out = zbuf_size;

    processRect(row, rowbytes, bpp, stride, h, NULL);

    deflate(&op_zstream1, Z_FINISH);
    deflate(&op_zstream2, Z_FINISH);
    op[n].p = pdata;

    if (op_zstream1.total_out < op_zstream2.total_out)
    {
        op[n].size = op_zstream1.total_out;
        op[n].filters = 0;
    }
    else
    {
        op[n].size = op_zstream2.total_out;
        op[n].filters = 1;
    }
    op[n].x = x;
    op[n].y = y;
    op[n].w = w;
    op[n].h = h;
    op[n].valid = 1;
    deflateReset(&op_zstream1);
    deflateReset(&op_zstream2);
}

bool PngEncoder::getRect(uint32_t w, uint32_t h, unsigned char* pimage1, unsigned char* pimage2, unsigned char* ptemp, uint32_t bpp, uint32_t stride, int zbuf_size, uint32_t has_tcolor, uint32_t tcolor, int n)
{
    uint32_t i, j, x0, y0, w0, h0;
    uint32_t x_min = w - 1;
    uint32_t y_min = h - 1;
    uint32_t x_max = 0;
    uint32_t y_max = 0;
    uint32_t diffnum = 0;
    uint32_t over_is_possible = 1;

    if (!has_tcolor)
        over_is_possible = 0;

    if (bpp == 1)
    {
        unsigned char* pa = pimage1;
        unsigned char* pb = pimage2;
        unsigned char* pc = ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                unsigned char c = *pb++;
                if (*pa++ != c)
                {
                    diffnum++;
                    if (has_tcolor && c == tcolor)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c = tcolor;

                *pc++ = c;
            }
    }
    else if (bpp == 2)
    {
        unsigned short* pa = (unsigned short*)pimage1;
        unsigned short* pb = (unsigned short*)pimage2;
        unsigned short* pc = (unsigned short*)ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uint32_t c1 = *pa++;
                uint32_t c2 = *pb++;
                if ((c1 != c2) && ((c1 >> 8) || (c2 >> 8)))
                {
                    diffnum++;
                    if ((c2 >> 8) != 0xFF)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c2 = 0;

                *pc++ = c2;
            }
    }
    else if (bpp == 3)
    {
        unsigned char* pa = pimage1;
        unsigned char* pb = pimage2;
        unsigned char* pc = ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uint32_t c1 = (pa[2] << 16) + (pa[1] << 8) + pa[0];
                uint32_t c2 = (pb[2] << 16) + (pb[1] << 8) + pb[0];
                if (c1 != c2)
                {
                    diffnum++;
                    if (has_tcolor && c2 == tcolor)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c2 = tcolor;

                memcpy(pc, &c2, 3);
                pa += 3;
                pb += 3;
                pc += 3;
            }
    }
    else if (bpp == 4)
    {
        uint32_t* pa = (uint32_t*)pimage1;
        uint32_t* pb = (uint32_t*)pimage2;
        uint32_t* pc = (uint32_t*)ptemp;

        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
            {
                uint32_t c1 = *pa++;
                uint32_t c2 = *pb++;
                if ((c1 != c2) && ((c1 >> 24) || (c2 >> 24)))
                {
                    diffnum++;
                    if ((c2 >> 24) != 0xFF)
                        over_is_possible = 0;
                    if (i < x_min)
                        x_min = i;
                    if (i > x_max)
                        x_max = i;
                    if (j < y_min)
                        y_min = j;
                    if (j > y_max)
                        y_max = j;
                }
                else
                    c2 = 0;

                *pc++ = c2;
            }
    }

    if (diffnum == 0)
    {
        return false;
    }
    else
    {
        x0 = x_min;
        y0 = y_min;
        w0 = x_max - x_min + 1;
        h0 = y_max - y_min + 1;
    }

    if (n < 3)
    {
        deflateRectOp(pimage2, x0, y0, w0, h0, bpp, stride, zbuf_size, n * 2);

        if (over_is_possible)
            deflateRectOp(ptemp, x0, y0, w0, h0, bpp, stride, zbuf_size, n * 2 + 1);
    }

    return true;
}

void PngEncoder::deflateRectFin(unsigned char* zbuf, uint32_t* zsize, int bpp, int stride, unsigned char* rows, int zbuf_size, int n)
{
    unsigned char* row = op[n].p + op[n].y * stride + op[n].x * bpp;
    int rowbytes = op[n].w * bpp;

    if (op[n].filters == 0)
    {
        unsigned char* dp = rows;
        for (int j = 0; j < op[n].h; j++)
        {
            *dp++ = 0;
            memcpy(dp, row, rowbytes);
            dp += rowbytes;
            row += stride;
        }
    }
    else
        processRect(row, rowbytes, bpp, stride, op[n].h, rows);

    z_stream fin_zstream;
    fin_zstream.data_type = Z_BINARY;
    fin_zstream.zalloc = Z_NULL;
    fin_zstream.zfree = Z_NULL;
    fin_zstream.opaque = Z_NULL;
    deflateInit2(&fin_zstream, Z_BEST_COMPRESSION, 8, 15, 8, op[n].filters ? Z_FILTERED : Z_DEFAULT_STRATEGY);

    fin_zstream.next_out = zbuf;
    fin_zstream.avail_out = zbuf_size;
    fin_zstream.next_in = rows;
    fin_zstream.avail_in = op[n].h * (rowbytes + 1);
    deflate(&fin_zstream, Z_FINISH);
    *zsize = fin_zstream.total_out;
    deflateEnd(&fin_zstream);
}

bool PngEncoder::writeanimation(const Animation& animation, const std::vector<int>& params)
{
    int compression_level = 6;
    int compression_strategy = IMWRITE_PNG_STRATEGY_RLE; // Default strategy
    bool isBilevel = false;

    for (size_t i = 0; i < params.size(); i += 2)
    {
        if (params[i] == IMWRITE_PNG_COMPRESSION)
        {
            compression_strategy = IMWRITE_PNG_STRATEGY_DEFAULT; // Default strategy
            compression_level = params[i + 1];
            compression_level = MIN(MAX(compression_level, 0), Z_BEST_COMPRESSION);
        }
        if (params[i] == IMWRITE_PNG_STRATEGY)
        {
            compression_strategy = params[i + 1];
            compression_strategy = MIN(MAX(compression_strategy, 0), Z_FIXED);
        }
        if (params[i] == IMWRITE_PNG_BILEVEL)
        {
            isBilevel = params[i + 1] != 0;
        }
    }

    CV_UNUSED(isBilevel);
    uint32_t first =0;
    uint32_t loops= animation.loop_count;
    uint32_t coltype= animation.frames[0].channels() == 1 ? PNG_COLOR_TYPE_GRAY : animation.frames[0].channels() == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA;

    FILE* m_f = NULL;
    uint32_t i, j, k;
    uint32_t x0, y0, w0, h0, dop, bop;
    uint32_t idat_size, zbuf_size, zsize;
    unsigned char header[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    uint32_t num_frames = (int)animation.frames.size();
    uint32_t width = animation.frames[0].cols;
    uint32_t height = animation.frames[0].rows;
    uint32_t bpp = (coltype == 6) ? 4 : (coltype == 2) ? 3
        : (coltype == 4) ? 2
        : 1;
    uint32_t has_tcolor = (coltype >= 4 || (coltype <= 2 && trnssize)) ? 1 : 0;
    uint32_t tcolor = 0;
    uint32_t rowbytes = width * bpp;
    uint32_t imagesize = rowbytes * height;

    AutoBuffer<unsigned char> temp(imagesize);
    AutoBuffer<unsigned char> over1(imagesize);
    AutoBuffer<unsigned char> over2(imagesize);
    AutoBuffer<unsigned char> over3(imagesize);
    AutoBuffer<unsigned char> rest(imagesize);
    AutoBuffer<unsigned char> rows((rowbytes + 1) * height);

    std::vector<APNGFrame> frames;
    std::vector<Mat> tmpframes;

    for (i = 0; i < (uint32_t)animation.frames.size(); i++)
    {
        APNGFrame apngFrame;
        tmpframes.push_back(animation.frames[i].clone());
        // TO DO optimize BGR RGB conversations
        if (animation.frames[i].channels() == 4)
            cvtColor(animation.frames[i], tmpframes[i], COLOR_BGRA2RGBA);
        if (animation.frames[i].channels() == 3)
            cvtColor(animation.frames[i], tmpframes[i], COLOR_BGR2RGB);

        if (tmpframes[i].depth() != CV_8U)
            tmpframes[i].convertTo(tmpframes[i], CV_8U, 1.0 / 255);
        apngFrame.setMat(tmpframes[i], animation.durations[i]);

        if (i > 0 && !getRect(width, height, frames.back().getPixels(), apngFrame.getPixels(), over1.data(), bpp, rowbytes, 0, 0, 0, 3))
        {
            frames.back().setDelayNum(frames.back().getDelayNum() + apngFrame.getDelayNum());
            num_frames--;
        }
        else
            frames.push_back(apngFrame);
    }

    if (trnssize)
    {
        if (coltype == 0)
            tcolor = trns[1];
        else if (coltype == 2)
            tcolor = (((trns[5] << 8) + trns[3]) << 8) + trns[1];
        else if (coltype == 3)
        {
            for (i = 0; i < trnssize; i++)
                if (trns[i] == 0)
                {
                    has_tcolor = 1;
                    tcolor = i;
                    break;
                }
        }
    }

    if (m_buf || (m_f = fopen(m_filename.c_str(), "wb")) != 0)
    {
        unsigned char buf_IHDR[13];
        unsigned char buf_acTL[8];
        unsigned char buf_fcTL[26];

        png_save_uint_32(buf_IHDR, width);
        png_save_uint_32(buf_IHDR + 4, height);
        buf_IHDR[8] = 8;
        buf_IHDR[9] = coltype;
        buf_IHDR[10] = 0;
        buf_IHDR[11] = 0;
        buf_IHDR[12] = 0;

        png_save_uint_32(buf_acTL, num_frames - first);
        png_save_uint_32(buf_acTL + 4, loops);

        write_to_io(header, 1, 8, m_f);

        writeChunk(m_f, "IHDR", buf_IHDR, 13);

        if (num_frames > 1)
            writeChunk(m_f, "acTL", buf_acTL, 8);
        else
            first = 0;

        if (palsize > 0)
            writeChunk(m_f, "PLTE", (unsigned char*)(&palette), palsize * 3);

        if ((animation.bgcolor != Scalar()) && (animation.frames.size() > 1))
        {
            uint64_t bgvalue = (static_cast<int>(animation.bgcolor[0]) & 0xFF) << 24 |
                (static_cast<int>(animation.bgcolor[1]) & 0xFF) << 16 |
                (static_cast<int>(animation.bgcolor[2]) & 0xFF) << 8 |
                (static_cast<int>(animation.bgcolor[3]) & 0xFF);
            writeChunk(m_f, "bKGD", (unsigned char*)(&bgvalue), 6); //the bKGD chunk must precede the first IDAT chunk, and must follow the PLTE chunk.
        }

        if (trnssize > 0)
            writeChunk(m_f, "tRNS", trns, trnssize);

        op_zstream1.data_type = Z_BINARY;
        op_zstream1.zalloc = Z_NULL;
        op_zstream1.zfree = Z_NULL;
        op_zstream1.opaque = Z_NULL;
        deflateInit2(&op_zstream1, compression_level, 8, 15, 8, compression_strategy);

        op_zstream2.data_type = Z_BINARY;
        op_zstream2.zalloc = Z_NULL;
        op_zstream2.zfree = Z_NULL;
        op_zstream2.opaque = Z_NULL;
        deflateInit2(&op_zstream2, compression_level, 8, 15, 8, Z_FILTERED);

        idat_size = (rowbytes + 1) * height;
        zbuf_size = idat_size + ((idat_size + 7) >> 3) + ((idat_size + 63) >> 6) + 11;

        AutoBuffer<unsigned char> zbuf(zbuf_size);
        op_zbuf1.allocate(zbuf_size);
        op_zbuf2.allocate(zbuf_size);
        row_buf.allocate(rowbytes + 1);
        sub_row.allocate(rowbytes + 1);
        up_row.allocate(rowbytes + 1);
        avg_row.allocate(rowbytes + 1);
        paeth_row.allocate(rowbytes + 1);

        row_buf[0] = 0;
        sub_row[0] = 1;
        up_row[0] = 2;
        avg_row[0] = 3;
        paeth_row[0] = 4;

        x0 = 0;
        y0 = 0;
        w0 = width;
        h0 = height;
        bop = 0;
        next_seq_num = 0;

        for (j = 0; j < 6; j++)
            op[j].valid = 0;
        deflateRectOp(frames[0].getPixels(), x0, y0, w0, h0, bpp, rowbytes, zbuf_size, 0);
        deflateRectFin(zbuf.data(), &zsize, bpp, rowbytes, rows.data(), zbuf_size, 0);

        if (first)
        {
            writeIDATs(m_f, 0, zbuf.data(), zsize, idat_size);
            for (j = 0; j < 6; j++)
                op[j].valid = 0;
            deflateRectOp(frames[1].getPixels(), x0, y0, w0, h0, bpp, rowbytes, zbuf_size, 0);
            deflateRectFin(zbuf.data(), &zsize, bpp, rowbytes, rows.data(), zbuf_size, 0);
        }

        for (i = first; i < num_frames - 1; i++)
        {
            uint32_t op_min;
            int op_best;

            for (j = 0; j < 6; j++)
                op[j].valid = 0;

            /* dispose = none */
            getRect(width, height, frames[i].getPixels(), frames[i + 1].getPixels(), over1.data(), bpp, rowbytes, zbuf_size, has_tcolor, tcolor, 0);

            /* dispose = background */
            if (has_tcolor)
            {
                memcpy(temp.data(), frames[i].getPixels(), imagesize);
                if (coltype == 2)
                    for (j = 0; j < h0; j++)
                        for (k = 0; k < w0; k++)
                            memcpy(temp.data() + ((j + y0) * width + (k + x0)) * 3, &tcolor, 3);
                else
                    for (j = 0; j < h0; j++)
                        memset(temp.data() + ((j + y0) * width + x0) * bpp, tcolor, w0 * bpp);

                getRect(width, height, temp.data(), frames[i + 1].getPixels(), over2.data(), bpp, rowbytes, zbuf_size, has_tcolor, tcolor, 1);
            }

            /* dispose = previous */
            if (i > first)
                getRect(width, height, rest.data(), frames[i + 1].getPixels(), over3.data(), bpp, rowbytes, zbuf_size, has_tcolor, tcolor, 2);

            op_min = op[0].size;
            op_best = 0;
            for (j = 1; j < 6; j++)
                if (op[j].valid)
                {
                    if (op[j].size < op_min)
                    {
                        op_min = op[j].size;
                        op_best = j;
                    }
                }

            dop = op_best >> 1;

            png_save_uint_32(buf_fcTL, next_seq_num++);
            png_save_uint_32(buf_fcTL + 4, w0);
            png_save_uint_32(buf_fcTL + 8, h0);
            png_save_uint_32(buf_fcTL + 12, x0);
            png_save_uint_32(buf_fcTL + 16, y0);
            png_save_uint_16(buf_fcTL + 20, frames[i].getDelayNum());
            png_save_uint_16(buf_fcTL + 22, frames[i].getDelayDen());
            buf_fcTL[24] = dop;
            buf_fcTL[25] = bop;
            writeChunk(m_f, "fcTL", buf_fcTL, 26);

            writeIDATs(m_f, i, zbuf.data(), zsize, idat_size);

            /* process apng dispose - begin */
            if (dop != 2)
                memcpy(rest.data(), frames[i].getPixels(), imagesize);

            if (dop == 1)
            {
                if (coltype == 2)
                    for (j = 0; j < h0; j++)
                        for (k = 0; k < w0; k++)
                            memcpy(rest.data() + ((j + y0) * width + (k + x0)) * 3, &tcolor, 3);
                else
                    for (j = 0; j < h0; j++)
                        memset(rest.data() + ((j + y0) * width + x0) * bpp, tcolor, w0 * bpp);
            }
            /* process apng dispose - end */

            x0 = op[op_best].x;
            y0 = op[op_best].y;
            w0 = op[op_best].w;
            h0 = op[op_best].h;
            bop = op_best & 1;

            deflateRectFin(zbuf.data(), &zsize, bpp, rowbytes, rows.data(), zbuf_size, op_best);
        }

        if (num_frames > 1)
        {
            png_save_uint_32(buf_fcTL, next_seq_num++);
            png_save_uint_32(buf_fcTL + 4, w0);
            png_save_uint_32(buf_fcTL + 8, h0);
            png_save_uint_32(buf_fcTL + 12, x0);
            png_save_uint_32(buf_fcTL + 16, y0);
            png_save_uint_16(buf_fcTL + 20, frames[i].getDelayNum());
            png_save_uint_16(buf_fcTL + 22, frames[i].getDelayDen());
            buf_fcTL[24] = 0;
            buf_fcTL[25] = bop;
            writeChunk(m_f, "fcTL", buf_fcTL, 26);
        }

        writeIDATs(m_f, num_frames - 1, zbuf.data(), zsize, idat_size);

        writeChunk(m_f, "IEND", 0, 0);

        if (m_f)
            fclose(m_f);

        deflateEnd(&op_zstream1);
        deflateEnd(&op_zstream2);
    }

    return true;
}

}

#endif

/* End of file. */
