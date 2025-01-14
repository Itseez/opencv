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
#include "grfmt_bmp.hpp"

namespace cv
{

static const char* fmtSignBmp = "BM";

/************************ BMP decoder *****************************/

BmpDecoder::BmpDecoder()
{
    m_signature = fmtSignBmp;
    m_offset = -1;
    m_buf_supported = true;
    m_origin = ORIGIN_TL;
    m_bpp = 0;
    m_rle_code = BMP_RGB;
    initMask();
}


BmpDecoder::~BmpDecoder()
{
}


void  BmpDecoder::close()
{
    m_strm.close();
}

ImageDecoder BmpDecoder::newDecoder() const
{
    return makePtr<BmpDecoder>();
}

bool  BmpDecoder::readHeader()
{
    bool result = false;
    bool iscolor = false;

    if( !m_buf.empty() )
    {
        if( !m_strm.open( m_buf ) )
            return false;
    }
    else if( !m_strm.open( m_filename ))
        return false;

    try
    {
        m_strm.skip( 10 );
        m_offset = m_strm.getDWord();

        int  size = m_strm.getDWord();
        CV_Assert(size > 0); // overflow, 2Gb limit

        initMask();
        if( size >= 36 )
        {
            m_width  = m_strm.getDWord();
            m_height = m_strm.getDWord();
            m_bpp    = m_strm.getDWord() >> 16;
            int m_rle_code_ = m_strm.getDWord();
            CV_Assert(m_rle_code_ >= 0 && m_rle_code_ <= BMP_BITFIELDS);
            m_rle_code = (BmpCompression)m_rle_code_;
            m_strm.skip(12);
            int clrused = m_strm.getDWord();

            if( m_bpp == 32 && m_rle_code == BMP_BITFIELDS && size >= 56 )
            {
                m_strm.skip(4); //important colors
                //0 is Red channel bit mask, 1 is Green channel bit mask, 2 is Blue channel bit mask, 3 is Alpha channel bit mask
                for( int index_rgba = 0; index_rgba < 4; ++index_rgba )
                {
                    uint mask = m_strm.getDWord();
                    m_rgba_mask[index_rgba] = mask;
                    if(mask != 0)
                    {
                        int bit_count = 0;
                        while(!(mask & 1))
                        {
                            mask >>= 1;
                            ++bit_count;
                        }
                        m_rgba_bit_offset[index_rgba] = bit_count;
                        m_rgba_scale_factor[index_rgba] = 255.0f / mask;
                    }
                }
                m_strm.skip( size - 56 );
            }
            else
                m_strm.skip( size - 36 );

            if( m_width > 0 && m_height != 0 &&
             (((m_bpp == 1 || m_bpp == 4 || m_bpp == 8 ||
                m_bpp == 24 || m_bpp == 32 ) && m_rle_code == BMP_RGB) ||
               ((m_bpp == 16 || m_bpp == 32) && (m_rle_code == BMP_RGB || m_rle_code == BMP_BITFIELDS)) ||
               (m_bpp == 4 && m_rle_code == BMP_RLE4) ||
               (m_bpp == 8 && m_rle_code == BMP_RLE8)))
            {
                iscolor = true;
                result = true;

                if( m_bpp <= 8 )
                {
                    CV_Assert(clrused >= 0 && clrused <= 256);
                    memset(m_palette, 0, sizeof(m_palette));
                    m_strm.getBytes(m_palette, (clrused == 0? 1<<m_bpp : clrused)*4 );
                    iscolor = IsColorPalette( m_palette, m_bpp );
                }
                else if( m_bpp == 16 && m_rle_code == BMP_BITFIELDS )
                {
                    int redmask = m_strm.getDWord();
                    int greenmask = m_strm.getDWord();
                    int bluemask = m_strm.getDWord();

                    if( bluemask == 0x1f && greenmask == 0x3e0 && redmask == 0x7c00 )
                        m_bpp = 15;
                    else if( bluemask == 0x1f && greenmask == 0x7e0 && redmask == 0xf800 )
                        ;
                    else
                        result = false;
                }
                else if (m_bpp == 32 && m_rle_code == BMP_BITFIELDS)
                {
                    // 32bit BMP not require to check something - we can simply allow it to use
                    ;
                }
                else if( m_bpp == 16 && m_rle_code == BMP_RGB )
                    m_bpp = 15;
            }
        }
        else if( size == 12 )
        {
            m_width  = m_strm.getWord();
            m_height = m_strm.getWord();
            m_bpp    = m_strm.getDWord() >> 16;
            m_rle_code = BMP_RGB;

            if( m_width > 0 && m_height != 0 &&
               (m_bpp == 1 || m_bpp == 4 || m_bpp == 8 ||
                m_bpp == 24 || m_bpp == 32 ))
            {
                if( m_bpp <= 8 )
                {
                    uchar buffer[256*3];
                    int j, clrused = 1 << m_bpp;
                    m_strm.getBytes( buffer, clrused*3 );
                    for( j = 0; j < clrused; j++ )
                    {
                        m_palette[j].b = buffer[3*j+0];
                        m_palette[j].g = buffer[3*j+1];
                        m_palette[j].r = buffer[3*j+2];
                    }
                }
                result = true;
            }
        }
    }
    catch(...)
    {
        throw;
    }
    // in 32 bit case alpha channel is used - so require CV_8UC4 type
    m_type = iscolor ? ((m_bpp == 32 && m_rle_code != BMP_RGB) ? CV_8UC4 : CV_8UC3 ) : CV_8UC1;
    m_origin = m_height > 0 ? ORIGIN_BL : ORIGIN_TL;
    if ( m_height == std::numeric_limits<int>::min() ) {
        // abs(std::numeric_limits<int>::min()) is undefined behavior.
        result = false;
    } else {
        m_height = std::abs(m_height);
    }

    if( !result )
    {
        m_offset = -1;
        m_width = m_height = -1;
        m_strm.close();
    }
    return result;
}


bool  BmpDecoder::readData( Mat& img )
{
    uchar* data = img.ptr();
    int step = validateToInt(img.step);
    bool color = img.channels() > 1;
    uchar  gray_palette[256] = {0};
    bool   result = false;
    int  src_pitch = ((m_width*(m_bpp != 15 ? m_bpp : 16) + 7)/8 + 3) & -4;
    int  nch = color ? 3 : 1;
    int  y, width3 = m_width*nch;

    // FIXIT: use safe pointer arithmetic (avoid 'int'), use size_t, intptr_t, etc
    CV_Assert(((uint64)m_height * m_width * nch < (CV_BIG_UINT(1) << 30)) && "BMP reader implementation doesn't support large images >= 1Gb");

    if( m_offset < 0 || !m_strm.isOpened())
        return false;

    if( m_origin == ORIGIN_BL )
    {
        data += (m_height - 1)*(size_t)step;
        step = -step;
    }

    AutoBuffer<uchar> _src, _bgr;
    _src.allocate(src_pitch + 32);

    if( !color )
    {
        if( m_bpp <= 8 )
        {
            CvtPaletteToGray( m_palette, gray_palette, 1 << m_bpp );
        }
        _bgr.allocate(m_width*3 + 32);
    }
    uchar *src = _src.data(), *bgr = _bgr.data();

    try
    {
        m_strm.setPos( m_offset );

        switch( m_bpp )
        {
        /************************* 1 BPP ************************/
        case 1:
            for( y = 0; y < m_height; y++, data += step )
            {
                m_strm.getBytes( src, src_pitch );
                FillColorRow1( color ? data : bgr, src, m_width, m_palette );
                if( !color )
                    icvCvt_BGR2Gray_8u_C3C1R( bgr, 0, data, 0, Size(m_width,1) );
            }
            result = true;
            break;

        /************************* 4 BPP ************************/
        case 4:
            if( m_rle_code == BMP_RGB )
            {
                for( y = 0; y < m_height; y++, data += step )
                {
                    m_strm.getBytes( src, src_pitch );
                    if( color )
                        FillColorRow4( data, src, m_width, m_palette );
                    else
                        FillGrayRow4( data, src, m_width, gray_palette );
                }
                result = true;
            }
            else if( m_rle_code == BMP_RLE4 ) // rle4 compression
            {
                uchar* line_end = data + width3;
                y = 0;

                for(;;)
                {
                    int code = m_strm.getWord();
                    const int len = code & 255;
                    code >>= 8;
                    if( len != 0 ) // encoded mode
                    {
                        PaletteEntry clr[2];
                        uchar gray_clr[2];
                        int t = 0;

                        clr[0] = m_palette[code >> 4];
                        clr[1] = m_palette[code & 15];
                        gray_clr[0] = gray_palette[code >> 4];
                        gray_clr[1] = gray_palette[code & 15];

                        uchar* end = data + len*nch;
                        if( end > line_end ) goto decode_rle4_bad;
                        do
                        {
                            if( color )
                                WRITE_PIX( data, clr[t] );
                            else
                                *data = gray_clr[t];
                            t ^= 1;
                        }
                        while( (data += nch) < end );
                    }
                    else if( code > 2 ) // absolute mode
                    {
                        if( data + code*nch > line_end ) goto decode_rle4_bad;
                        int sz = (((code + 1)>>1) + 1) & (~1);
                        CV_Assert((size_t)sz < _src.size());
                        m_strm.getBytes(src, sz);
                        if( color )
                            data = FillColorRow4( data, src, code, m_palette );
                        else
                            data = FillGrayRow4( data, src, code, gray_palette );
                    }
                    else
                    {
                        int x_shift3 = (int)(line_end - data);

                        if( code == 2 )
                        {
                            x_shift3 = m_strm.getByte()*nch;
                            m_strm.getByte();
                        }

                        if( color )
                            data = FillUniColor( data, line_end, step, width3,
                                                 y, m_height, x_shift3,
                                                 m_palette[0] );
                        else
                            data = FillUniGray( data, line_end, step, width3,
                                                y, m_height, x_shift3,
                                                gray_palette[0] );

                        if( y >= m_height )
                            break;
                    }
                }

                result = true;
decode_rle4_bad: ;
            }
            break;

        /************************* 8 BPP ************************/
        case 8:
            if( m_rle_code == BMP_RGB )
            {
                for( y = 0; y < m_height; y++, data += step )
                {
                    m_strm.getBytes( src, src_pitch );
                    if( color )
                        FillColorRow8( data, src, m_width, m_palette );
                    else
                        FillGrayRow8( data, src, m_width, gray_palette );
                }
                result = true;
            }
            else if( m_rle_code == BMP_RLE8 ) // rle8 compression
            {
                uchar* line_end = data + width3;
                int line_end_flag = 0;
                y = 0;

                for(;;)
                {
                    int code = m_strm.getWord();
                    int len = code & 255;
                    code >>= 8;
                    if( len != 0 ) // encoded mode
                    {
                        int prev_y = y;
                        len *= nch;

                        if( data + len > line_end )
                            goto decode_rle8_bad;

                        if( color )
                            data = FillUniColor( data, line_end, step, width3,
                                                 y, m_height, len,
                                                 m_palette[code] );
                        else
                            data = FillUniGray( data, line_end, step, width3,
                                                y, m_height, len,
                                                gray_palette[code] );

                        line_end_flag = y - prev_y;

                        if( y >= m_height )
                            break;
                    }
                    else if( code > 2 ) // absolute mode
                    {
                        int prev_y = y;
                        int code3 = code*nch;

                        if( data + code3 > line_end )
                            goto decode_rle8_bad;
                        int sz = (code + 1) & (~1);
                        CV_Assert((size_t)sz < _src.size());
                        m_strm.getBytes(src, sz);
                        if( color )
                            data = FillColorRow8( data, src, code, m_palette );
                        else
                            data = FillGrayRow8( data, src, code, gray_palette );

                        line_end_flag = y - prev_y;
                    }
                    else
                    {
                        int x_shift3 = (int)(line_end - data);
                        int y_shift = m_height - y;

                        if( code || !line_end_flag || x_shift3 < width3 )
                        {
                            if( code == 2 )
                            {
                                x_shift3 = m_strm.getByte()*nch;
                                y_shift = m_strm.getByte();
                            }

                            x_shift3 += (y_shift * width3) & ((code == 0) - 1);

                            if( y >= m_height )
                                break;

                            if( color )
                                data = FillUniColor( data, line_end, step, width3,
                                                     y, m_height, x_shift3,
                                                     m_palette[0] );
                            else
                                data = FillUniGray( data, line_end, step, width3,
                                                    y, m_height, x_shift3,
                                                    gray_palette[0] );

                            if( y >= m_height )
                                break;
                        }

                        line_end_flag = 0;
                        if( y >= m_height )
                            break;
                    }
                }

                result = true;
decode_rle8_bad: ;
            }
            break;
        /************************* 15 BPP ************************/
        case 15:
            for( y = 0; y < m_height; y++, data += step )
            {
                m_strm.getBytes( src, src_pitch );
                if( !color )
                    icvCvt_BGR5552Gray_8u_C2C1R( src, 0, data, 0, Size(m_width,1) );
                else
                    icvCvt_BGR5552BGR_8u_C2C3R( src, 0, data, 0, Size(m_width,1) );
            }
            result = true;
            break;
        /************************* 16 BPP ************************/
        case 16:
            for( y = 0; y < m_height; y++, data += step )
            {
                m_strm.getBytes( src, src_pitch );
                if( !color )
                    icvCvt_BGR5652Gray_8u_C2C1R( src, 0, data, 0, Size(m_width,1) );
                else
                    icvCvt_BGR5652BGR_8u_C2C3R( src, 0, data, 0, Size(m_width,1) );
            }
            result = true;
            break;
        /************************* 24 BPP ************************/
        case 24:
            for( y = 0; y < m_height; y++, data += step )
            {
                m_strm.getBytes( src, src_pitch );
                if(!color)
                    icvCvt_BGR2Gray_8u_C3C1R( src, 0, data, 0, Size(m_width,1) );
                else
                    memcpy( data, src, m_width*3 );
            }
            result = true;
            break;
        /************************* 32 BPP ************************/
        case 32:
            {
                bool has_bit_mask = (m_rgba_bit_offset[0] >= 0) && (m_rgba_bit_offset[1] >= 0) && (m_rgba_bit_offset[2] >= 0);
                for( y = 0; y < m_height; y++, data += step )
                {
                    m_strm.getBytes( src, src_pitch );

                    if( !color )
                    {
                        if ( has_bit_mask )
                            maskBGRAtoGray(data, src, m_width);
                        else
                            icvCvt_BGRA2Gray_8u_C4C1R( src, 0, data, 0, Size(m_width,1) );
                    }
                    else if( img.channels() == 3 )
                    {
                        if ( has_bit_mask )
                            maskBGRA(data, src, m_width, false);
                        else
                            icvCvt_BGRA2BGR_8u_C4C3R(src, 0, data, 0, Size(m_width, 1));
                    }
                    else if ( img.channels() == 4 )
                    {
                        if ( has_bit_mask )
                            maskBGRA(data, src, m_width, true);
                        else
                            memcpy(data, src, m_width * 4);
                    }
                }
            }
            result = true;
            break;
        default:
            CV_Error(cv::Error::StsError, "Invalid/unsupported mode");
        }
    }
    catch(...)
    {
        throw;
    }

    if (m_use_rgb && color && img.channels() == 3)
    {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    return result;
}

void  BmpDecoder::initMask()
{
    memset(m_rgba_mask, 0, sizeof(m_rgba_mask));
    memset(m_rgba_bit_offset, -1, sizeof(m_rgba_bit_offset));
    for (size_t i = 0; i < 4; i++) {
        m_rgba_scale_factor[i] = 1.0f;
    }
}

void  BmpDecoder::maskBGRA(uchar* des, const uchar* src, int num, bool alpha_required)
{
    int dest_stride = alpha_required ? 4 : 3;
    for( int i = 0; i < num; i++, des += dest_stride, src += 4 )
    {
        uint data = *((uint*)src);
        des[0] = (uchar)(((m_rgba_mask[2] & data) >> m_rgba_bit_offset[2]) * m_rgba_scale_factor[2]);
        des[1] = (uchar)(((m_rgba_mask[1] & data) >> m_rgba_bit_offset[1]) * m_rgba_scale_factor[1]);
        des[2] = (uchar)(((m_rgba_mask[0] & data) >> m_rgba_bit_offset[0]) * m_rgba_scale_factor[0]);
        if (alpha_required)
        {
            if (m_rgba_bit_offset[3] >= 0)
                des[3] = (uchar)(((m_rgba_mask[3] & data) >> m_rgba_bit_offset[3]) * m_rgba_scale_factor[3]);
            else
                des[3] = 255;
        }
    }
}

void  BmpDecoder::maskBGRAtoGray(uchar* des, const uchar* src, int num)
{
    for( int i = 0; i < num; i++, des++, src += 4 )
    {
        uint data = *((uint*)src);
        int red = (uchar)(((m_rgba_mask[0] & data) >> m_rgba_bit_offset[0]) * m_rgba_scale_factor[0]);
        int green = (uchar)(((m_rgba_mask[1] & data) >> m_rgba_bit_offset[1]) * m_rgba_scale_factor[1]);
        int blue = (uchar)(((m_rgba_mask[2] & data) >> m_rgba_bit_offset[2]) * m_rgba_scale_factor[2]);

        *des = (uchar)(0.299f * red + 0.587f * green + 0.114f * blue);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////

BmpEncoder::BmpEncoder()
{
    m_description = "Windows bitmap (*.bmp;*.dib)";
    m_buf_supported = true;
}


BmpEncoder::~BmpEncoder()
{
}

ImageEncoder BmpEncoder::newEncoder() const
{
    return makePtr<BmpEncoder>();
}

bool  BmpEncoder::write( const Mat& img, const std::vector<int>& )
{
    int width = img.cols, height = img.rows, channels = img.channels();
    int fileStep = (width*channels + 3) & -4;
    uchar zeropad[] = "\0\0\0\0";
    WLByteStream strm;

    if( m_buf )
    {
        if( !strm.open( *m_buf ) )
            return false;
    }
    else if( !strm.open( m_filename ))
        return false;

    int  bitmapHeaderSize = 40;
    int  paletteSize = channels > 1 ? 0 : 1024;
    int  headerSize = 14 /* fileheader */ + bitmapHeaderSize + paletteSize;
    size_t fileSize = (size_t)fileStep*height + headerSize;
    PaletteEntry palette[256];

    if( m_buf )
        m_buf->reserve( alignSize(fileSize + 16, 256) );

    // write signature 'BM'
    CHECK_WRITE(strm.putBytes( fmtSignBmp, (int)strlen(fmtSignBmp) ));

    // write file header
    CHECK_WRITE(strm.putDWord( validateToInt(fileSize) )); // file size
    CHECK_WRITE(strm.putDWord( 0 ));
    CHECK_WRITE(strm.putDWord( headerSize ));

    // write bitmap header
    CHECK_WRITE(strm.putDWord( bitmapHeaderSize ));
    CHECK_WRITE(strm.putDWord( width ));
    CHECK_WRITE(strm.putDWord( height ));
    CHECK_WRITE(strm.putWord( 1 ));
    CHECK_WRITE(strm.putWord( channels << 3 ));
    CHECK_WRITE(strm.putDWord( BMP_RGB ));
    CHECK_WRITE(strm.putDWord( 0 ));
    CHECK_WRITE(strm.putDWord( 0 ));
    CHECK_WRITE(strm.putDWord( 0 ));
    CHECK_WRITE(strm.putDWord( 0 ));
    CHECK_WRITE(strm.putDWord( 0 ));

    if( channels == 1 )
    {
        FillGrayPalette( palette, 8 );
        CHECK_WRITE(strm.putBytes( palette, sizeof(palette)));
    }

    width *= channels;
    for( int y = height - 1; y >= 0; y-- )
    {
        CHECK_WRITE(strm.putBytes( img.ptr(y), width ));
        if( fileStep > width )
        {
            CHECK_WRITE(strm.putBytes( zeropad, fileStep - width ));
        }
    }

    strm.close();
    return true;
}

}
