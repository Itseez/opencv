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

/****************************************************************************************\
    A part of the file implements TIFF reader on base of libtiff library
    (see otherlibs/_graphics/readme.txt for copyright notice)
\****************************************************************************************/

#include "precomp.hpp"

#ifdef HAVE_TIFF
#include <opencv2/core/utils/logger.hpp>

#include "grfmt_tiff.hpp"
#include <limits>

#include "tiff.h"
#include "tiffio.h"

namespace cv
{

// to extend cvtColor() to support CV_8S, CV_16S, CV_32S and CV_64F.
static void extend_cvtColor( InputArray _src, OutputArray _dst, int code );

#define CV_TIFF_CHECK_CALL(call) \
    if (0 == (call)) { \
        CV_LOG_WARNING(NULL, "OpenCV TIFF(line " << __LINE__ << "): failed " #call); \
        CV_Error(Error::StsError, "OpenCV TIFF: failed " #call); \
    }

#define CV_TIFF_CHECK_CALL_DEBUG(call) \
    if (0 == (call)) { \
        CV_LOG_DEBUG(NULL, "OpenCV TIFF(line " << __LINE__ << "): failed optional call: " #call ", ignoring"); \
    }

static void cv_tiffCloseHandle(void* handle)
{
    TIFFClose((TIFF*)handle);
}

static void cv_tiffErrorHandler(const char* module, const char* fmt, va_list ap)
{
    if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_DEBUG)
        return;
    // TODO cv::vformat() with va_list parameter
    fprintf(stderr, "OpenCV TIFF: ");
    if (module != NULL)
        fprintf(stderr, "%s: ", module);
    fprintf(stderr, "Warning, ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, ".\n");
}

static bool cv_tiffSetErrorHandler_()
{
    TIFFSetErrorHandler(cv_tiffErrorHandler);
    TIFFSetWarningHandler(cv_tiffErrorHandler);
    return true;
}

static bool cv_tiffSetErrorHandler()
{
    static bool v = cv_tiffSetErrorHandler_();
    return v;
}

static const char fmtSignTiffII[] = "II\x2a\x00";
static const char fmtSignTiffMM[] = "MM\x00\x2a";
static const char fmtSignBigTiffII[] = "II\x2b\x00";
static const char fmtSignBigTiffMM[] = "MM\x00\x2b";

TiffDecoder::TiffDecoder()
{
    m_hdr = false;
    m_buf_supported = true;
    m_buf_pos = 0;
}


void TiffDecoder::close()
{
    m_tif.release();
}

TiffDecoder::~TiffDecoder()
{
    close();
}

size_t TiffDecoder::signatureLength() const
{
    return 4;
}

bool TiffDecoder::checkSignature( const String& signature ) const
{
    return signature.size() >= 4 &&
        (memcmp(signature.c_str(), fmtSignTiffII, 4) == 0 ||
        memcmp(signature.c_str(), fmtSignTiffMM, 4) == 0 ||
        memcmp(signature.c_str(), fmtSignBigTiffII, 4) == 0 ||
        memcmp(signature.c_str(), fmtSignBigTiffMM, 4) == 0);
}

int TiffDecoder::normalizeChannelsNumber(int channels) const
{
    CV_Check(channels, channels >= 1 && channels <= 5, "Unsupported number of channels");
    return channels;
}

ImageDecoder TiffDecoder::newDecoder() const
{
    cv_tiffSetErrorHandler();
    return makePtr<TiffDecoder>();
}

class TiffDecoderBufHelper
{
    Mat& m_buf;
    size_t& m_buf_pos;
public:
    TiffDecoderBufHelper(Mat& buf, size_t& buf_pos) :
        m_buf(buf), m_buf_pos(buf_pos)
    {}
    static tmsize_t read( thandle_t handle, void* buffer, tmsize_t n )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        const Mat& buf = helper->m_buf;
        const tmsize_t size = buf.cols*buf.rows*buf.elemSize();
        tmsize_t pos = helper->m_buf_pos;
        if ( n > (size - pos) )
        {
            n = size - pos;
        }
        std::memcpy(buffer, buf.ptr() + pos, n);
        helper->m_buf_pos += n;
        return n;
    }

    static tmsize_t write( thandle_t /*handle*/, void* /*buffer*/, tmsize_t /*n*/ )
    {
        // Not used for decoding.
        return 0;
    }

    static toff_t seek( thandle_t handle, toff_t offset, int whence )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        const Mat& buf = helper->m_buf;
        const toff_t size = buf.cols*buf.rows*buf.elemSize();
        toff_t new_pos = helper->m_buf_pos;
        switch (whence)
        {
            case SEEK_SET:
                new_pos = offset;
                break;
            case SEEK_CUR:
                new_pos += offset;
                break;
            case SEEK_END:
                new_pos = size + offset;
                break;
        }
        new_pos = std::min(new_pos, size);
        helper->m_buf_pos = (size_t)new_pos;
        return new_pos;
    }

    static int map( thandle_t handle, void** base, toff_t* size )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        Mat& buf = helper->m_buf;
        *base = buf.ptr();
        *size = buf.cols*buf.rows*buf.elemSize();
        return 0;
    }

    static toff_t size( thandle_t handle )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        const Mat& buf = helper->m_buf;
        return buf.cols*buf.rows*buf.elemSize();
    }

    static int close( thandle_t handle )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        delete helper;
        return 0;
    }
};

bool TiffDecoder::readHeader()
{
    bool result = false;
    TIFF* tif = static_cast<TIFF*>(m_tif.get());
    if (!tif)
    {
        // TIFFOpen() mode flags are different to fopen().  A 'b' in mode "rb" has no effect when reading.
        // http://www.simplesystems.org/libtiff/functions/TIFFOpen.html
        if ( !m_buf.empty() )
        {
            m_buf_pos = 0;
            TiffDecoderBufHelper* buf_helper = new TiffDecoderBufHelper(this->m_buf, this->m_buf_pos);
            tif = TIFFClientOpen( "", "r", reinterpret_cast<thandle_t>(buf_helper), &TiffDecoderBufHelper::read,
                                  &TiffDecoderBufHelper::write, &TiffDecoderBufHelper::seek,
                                  &TiffDecoderBufHelper::close, &TiffDecoderBufHelper::size,
                                  &TiffDecoderBufHelper::map, /*unmap=*/0 );
            if (!tif)
                delete buf_helper;
        }
        else
        {
            tif = TIFFOpen(m_filename.c_str(), "r");
        }
        if (tif)
            m_tif.reset(tif, cv_tiffCloseHandle);
        else
            m_tif.release();
    }

    if (tif)
    {
        uint32_t wdth = 0, hght = 0;
        uint16_t photometric = 0;

        CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &wdth));
        CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &hght));
        CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric));

        {
            bool isGrayScale = photometric == PHOTOMETRIC_MINISWHITE || photometric == PHOTOMETRIC_MINISBLACK;
            uint16_t bpp = 8, ncn = isGrayScale ? 1 : 3;
            if (0 == TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp))
            {
                // TIFF bi-level images don't require TIFFTAG_BITSPERSAMPLE tag
                bpp = 1;
            }
            CV_TIFF_CHECK_CALL_DEBUG(TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &ncn));

            m_width = wdth;
            m_height = hght;
            m_frame_count = TIFFNumberOfDirectories(tif);
            if (ncn == 3 && photometric == PHOTOMETRIC_LOGLUV)
            {
                m_type = CV_32FC3;
                m_hdr = true;
                return true;
            }
            m_hdr = false;

            if( bpp > 8 &&
               ((photometric > 2) ||
                (ncn != 1 && ncn != 3 && ncn != 4)))
                bpp = 8;

            uint16_t sample_format = SAMPLEFORMAT_UINT;
            TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);
            int wanted_channels = normalizeChannelsNumber(ncn);
            switch (bpp)
            {
            case 1:
            {
                CV_Check((int)sample_format, sample_format == SAMPLEFORMAT_UINT || sample_format == SAMPLEFORMAT_INT, "");
                int depth = sample_format == SAMPLEFORMAT_INT ? CV_8S : CV_8U;
                m_type = CV_MAKETYPE(depth, !isGrayScale ? wanted_channels : 1);
                result = true;
                break;
            }
            case 4:
                //support 4-bit palette.
                if (photometric == PHOTOMETRIC_PALETTE)
                {
                    CV_Check((int)sample_format, sample_format == SAMPLEFORMAT_UINT || sample_format == SAMPLEFORMAT_INT, "");
                    int depth = sample_format == SAMPLEFORMAT_INT ? CV_8S : CV_8U;
                    m_type = CV_MAKETYPE(depth, 3);
                    result = true;
                }
                else
                    CV_Error(cv::Error::StsError, "bitsperpixel value is 4 should be palette.");
                break;
            case 8:
            {
                //Palette color, the value of the component is used as an index into the red,
                //green and blue curves in the ColorMap field to retrieve an RGB triplet that defines the color.
                CV_Check((int)sample_format, sample_format == SAMPLEFORMAT_UINT || sample_format == SAMPLEFORMAT_INT, "");
                int depth = sample_format == SAMPLEFORMAT_INT ? CV_8S : CV_8U;
                if (photometric == PHOTOMETRIC_PALETTE)
                    m_type = CV_MAKETYPE(depth, 3);
                else
                    m_type = CV_MAKETYPE(depth, !isGrayScale ? wanted_channels : 1);
                result = true;
                break;
            }
            case 10:
            case 12:
            case 14:
            case 16:
            {
                CV_Check((int)sample_format, sample_format == SAMPLEFORMAT_UINT || sample_format == SAMPLEFORMAT_INT, "");
                int depth = sample_format == SAMPLEFORMAT_INT ? CV_16S : CV_16U;
                m_type = CV_MAKETYPE(depth, !isGrayScale ? wanted_channels : 1);
                result = true;
                break;
            }
            case 32:
            {
                CV_Check((int)sample_format, sample_format == SAMPLEFORMAT_IEEEFP || sample_format == SAMPLEFORMAT_INT, "");
                int depth = sample_format == SAMPLEFORMAT_IEEEFP ? CV_32F : CV_32S;
                m_type = CV_MAKETYPE(depth, wanted_channels);
                result = true;
                break;
            }
            case 64:
                CV_CheckEQ((int)sample_format, SAMPLEFORMAT_IEEEFP, "");
                m_type = CV_MAKETYPE(CV_64F, wanted_channels);
                result = true;
                break;
            default:
                CV_Error(cv::Error::StsError, "Invalid bitsperpixel value read from TIFF header! Must be 1, 8, 10, 12, 14, 16, 32 or 64.");
            }
        }
    }

    if( !result )
        close();

    return result;
}

bool TiffDecoder::nextPage()
{
    // Prepare the next page, if any.
    return !m_tif.empty() &&
           TIFFReadDirectory(static_cast<TIFF*>(m_tif.get())) &&
           readHeader();
}

static void fixOrientationPartial(Mat &img, uint16_t orientation)
{
    switch(orientation) {
        case ORIENTATION_RIGHTTOP:
        case ORIENTATION_LEFTBOT:
            flip(img, img, -1);
            /* fall through */

        case ORIENTATION_LEFTTOP:
        case ORIENTATION_RIGHTBOT:
            transpose(img, img);
            break;
    }
}

static void fixOrientationFull(Mat &img, int orientation)
{
    switch(orientation) {
        case ORIENTATION_TOPRIGHT:
            flip(img, img, 1);
            break;

        case ORIENTATION_BOTRIGHT:
            flip(img, img, -1);
            break;

        case ORIENTATION_BOTLEFT:
            flip(img, img, 0);
            break;

        case ORIENTATION_LEFTTOP:
            transpose(img, img);
            break;

        case ORIENTATION_RIGHTTOP:
            transpose(img, img);
            flip(img, img, 1);
            break;

        case ORIENTATION_RIGHTBOT:
            transpose(img, img);
            flip(img, img, -1);
            break;

        case ORIENTATION_LEFTBOT:
            transpose(img, img);
            flip(img, img, 0);
            break;
    }
}

/**
 * Fix orientation defined in tag 274.
 * For 8 bit some corrections are done by TIFFReadRGBAStrip/Tile already.
 * Not so for 16/32/64 bit.
 */
static void fixOrientation(Mat &img, uint16_t orientation, bool isOrientationFull)
{
    if( isOrientationFull )
    {
        fixOrientationFull(img, orientation);
    }
    else
    {
        fixOrientationPartial(img, orientation);
    }
}

static void _unpack10To16(const uchar* src, const uchar* srcEnd, ushort* dst, ushort* dstEnd, size_t expectedDstElements)
{
    //5*8b=4*10b : 5 src for 4 dst
    constexpr const size_t packedBitsCount = 10;
    constexpr const size_t packedBitsMask = ((1<<packedBitsCount)-1);
    constexpr const size_t srcElementsPerPacket = 5;
    constexpr const size_t dstElementsPerPacket = 4;
    constexpr const size_t bitsPerPacket = dstElementsPerPacket*packedBitsCount;
    const size_t fullPacketsCount = std::min({
      expectedDstElements/dstElementsPerPacket,
      (static_cast<size_t>(srcEnd-src)/srcElementsPerPacket),
      (static_cast<size_t>(dstEnd-dst)/dstElementsPerPacket)
    });
    union {
      uint64_t u64;
      uint8_t  u8[8];
    } buf = {0};
    for(size_t i = 0 ; i<fullPacketsCount ; ++i)
    {
        for(size_t j = 0 ; j<srcElementsPerPacket ; ++j)
          buf.u8[srcElementsPerPacket-1-j] = *src++;
        for(size_t j = 0 ; j<dstElementsPerPacket ; ++j)
        {
          dst[dstElementsPerPacket-1-j] = static_cast<ushort>(buf.u64 & packedBitsMask);
          buf.u64 >>= packedBitsCount;
        }
        dst += dstElementsPerPacket;
    }
    size_t remainingDstElements = std::min(
        expectedDstElements-fullPacketsCount*dstElementsPerPacket,
        static_cast<size_t>(dstEnd-dst)
    );
    bool stop = !remainingDstElements;
    while(!stop)
    {
        for(size_t j = 0 ; j<srcElementsPerPacket ; ++j)
            buf.u8[srcElementsPerPacket-1-j] = (src<srcEnd) ? *src++ : 0;
        for(size_t j = 0 ; j<dstElementsPerPacket ; ++j)
        {
            stop |= !(remainingDstElements--);
            if (!stop)
              *dst++ = static_cast<ushort>((buf.u64 >> (bitsPerPacket-(j+1)*packedBitsCount)) & packedBitsMask);
        }
    }//end while(!stop)
}
//end _unpack10To16()

static void _unpack12To16(const uchar* src, const uchar* srcEnd, ushort* dst, ushort* dstEnd, size_t expectedDstElements)
{
  //3*8b=2*12b : 3 src for 2 dst
  constexpr const size_t packedBitsCount = 12;
  constexpr const size_t packedBitsMask = ((1<<packedBitsCount)-1);
  constexpr const size_t srcElementsPerPacket = 3;
  constexpr const size_t dstElementsPerPacket = 2;
  constexpr const size_t bitsPerPacket = dstElementsPerPacket*packedBitsCount;
  const size_t fullPacketsCount = std::min({
    expectedDstElements/dstElementsPerPacket,
    (static_cast<size_t>(srcEnd-src)/srcElementsPerPacket),
    (static_cast<size_t>(dstEnd-dst)/dstElementsPerPacket)
  });
  union {
      uint32_t u32;
      uint8_t  u8[4];
  } buf = {0};
  for(size_t i = 0 ; i<fullPacketsCount ; ++i)
  {
      for(size_t j = 0 ; j<srcElementsPerPacket ; ++j)
          buf.u8[srcElementsPerPacket-1-j] = *src++;
      for(size_t j = 0 ; j<dstElementsPerPacket ; ++j)
      {
          dst[dstElementsPerPacket-1-j] = static_cast<ushort>(buf.u32 & packedBitsMask);
          buf.u32 >>= packedBitsCount;
      }
      dst += dstElementsPerPacket;
  }
  size_t remainingDstElements = std::min(
      expectedDstElements-fullPacketsCount*dstElementsPerPacket,
      static_cast<size_t>(dstEnd-dst)
  );
  bool stop = !remainingDstElements;
  while(!stop)
  {
      for(size_t j = 0 ; j<srcElementsPerPacket ; ++j)
          buf.u8[srcElementsPerPacket-1-j] = (src<srcEnd) ? *src++ : 0;
      for(size_t j = 0 ; j<dstElementsPerPacket ; ++j)
      {
          stop |= !(remainingDstElements--);
          if (!stop)
              *dst++ = static_cast<ushort>((buf.u32 >> (bitsPerPacket-(j+1)*packedBitsCount)) & packedBitsMask);
      }
  }//end while(!stop)
}
//end _unpack12To16()

static void _unpack14To16(const uchar* src, const uchar* srcEnd, ushort* dst, ushort* dstEnd, size_t expectedDstElements)
{
    //7*8b=4*14b : 7 src for 4 dst
    constexpr const size_t packedBitsCount = 14;
    constexpr const size_t packedBitsMask = ((1<<packedBitsCount)-1);
    constexpr const size_t srcElementsPerPacket = 7;
    constexpr const size_t dstElementsPerPacket = 4;
    constexpr const size_t bitsPerPacket = dstElementsPerPacket*packedBitsCount;
    const size_t fullPacketsCount = std::min({
      expectedDstElements/dstElementsPerPacket,
      (static_cast<size_t>(srcEnd-src)/srcElementsPerPacket),
      (static_cast<size_t>(dstEnd-dst)/dstElementsPerPacket)
    });
    union {
        uint64_t u64;
        uint8_t  u8[8];
    } buf = {0};
    for(size_t i = 0 ; i<fullPacketsCount ; ++i)
    {
        for(size_t j = 0 ; j<srcElementsPerPacket ; ++j)
            buf.u8[srcElementsPerPacket-1-j] = *src++;
        for(size_t j = 0 ; j<dstElementsPerPacket ; ++j)
        {
            dst[dstElementsPerPacket-1-j] = static_cast<ushort>(buf.u64 & packedBitsMask);
            buf.u64 >>= packedBitsCount;
        }
        dst += dstElementsPerPacket;
    }
    size_t remainingDstElements = std::min(
        expectedDstElements-fullPacketsCount*dstElementsPerPacket,
        static_cast<size_t>(dstEnd-dst)
    );
    bool stop = !remainingDstElements;
    while(!stop)
    {
        for(size_t j = 0 ; j<srcElementsPerPacket ; ++j)
            buf.u8[srcElementsPerPacket-1-j] = (src<srcEnd) ? *src++ : 0;
        for(size_t j = 0 ; j<dstElementsPerPacket ; ++j)
        {
            stop |= !(remainingDstElements--);
            if (!stop)
                *dst++ = static_cast<ushort>((buf.u64 >> (bitsPerPacket-(j+1)*packedBitsCount)) & packedBitsMask);
        }
    }//end while(!stop)
}
//end _unpack14To16()

bool  TiffDecoder::readData( Mat& img )
{
    int type = img.type();
    int depth = CV_MAT_DEPTH(type);

    CV_Assert(!m_tif.empty());
    TIFF* tif = (TIFF*)m_tif.get();

    uint16_t photometric = (uint16_t)-1;
    CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric));

    if (m_hdr && depth >= CV_32F)
    {
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_FLOAT));
    }

    bool color = img.channels() > 1;

    CV_CheckType(type, depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F, "");

    if (m_width && m_height)
    {
        int is_tiled = TIFFIsTiled(tif) != 0;
        bool isGrayScale = photometric == PHOTOMETRIC_MINISWHITE || photometric == PHOTOMETRIC_MINISBLACK;
        uint16_t bpp = 8, ncn = isGrayScale ? 1 : 3;
        if (0 == TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp))
        {
            // TIFF bi-level images don't require TIFFTAG_BITSPERSAMPLE tag
            bpp = 1;
        }
        CV_TIFF_CHECK_CALL_DEBUG(TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &ncn));
        uint16_t img_orientation = ORIENTATION_TOPLEFT;
        CV_TIFF_CHECK_CALL_DEBUG(TIFFGetField(tif, TIFFTAG_ORIENTATION, &img_orientation));
        constexpr const int bitsPerByte = 8;
        int dst_bpp = (int)(img.elemSize1() * bitsPerByte);
        bool vert_flip = dst_bpp == 8 &&
                        (img_orientation == ORIENTATION_BOTRIGHT || img_orientation == ORIENTATION_RIGHTBOT ||
                         img_orientation == ORIENTATION_BOTLEFT || img_orientation == ORIENTATION_LEFTBOT);
        int wanted_channels = normalizeChannelsNumber(img.channels());
        bool doReadScanline = false;

        uint32_t tile_width0 = m_width, tile_height0 = 0;

        if (is_tiled)
        {
            CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width0));
            CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_height0));
        }
        else
        {
            // optional
            CV_TIFF_CHECK_CALL_DEBUG(TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &tile_height0));
        }

        {
            if (tile_width0 == 0)
                tile_width0 = m_width;

            if (tile_height0 == 0 ||
                    (!is_tiled && tile_height0 == std::numeric_limits<uint32_t>::max()) )
                tile_height0 = m_height;

            const int TILE_MAX_WIDTH = (1 << 24);
            const int TILE_MAX_HEIGHT = (1 << 24);
            CV_Assert((int)tile_width0 > 0 && (int)tile_width0 <= TILE_MAX_WIDTH);
            CV_Assert((int)tile_height0 > 0 && (int)tile_height0 <= TILE_MAX_HEIGHT);
            const uint64_t MAX_TILE_SIZE = (CV_BIG_UINT(1) << 30);
            CV_CheckLE((int)ncn, 5, "");
            CV_CheckLE((int)bpp, 64, "");

            if (dst_bpp == 8)
            {
                const int _ncn = 4; // Read RGBA
                const int _bpp = 8; // Read 8bit

                // if buffer_size(as 32bit RGBA) >= MAX_TILE_SIZE*95%,
                // we will use TIFFReadScanline function.

                if (
                    (uint64_t)tile_width0 * tile_height0 * _ncn * std::max(1, (int)(_bpp / bitsPerByte))
                    >=
                    ( (uint64_t) MAX_TILE_SIZE * 95 / 100)
                )
                {
                    uint16_t planerConfig = (uint16_t)-1;
                    CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planerConfig));

                    doReadScanline = (!is_tiled) // no tile
                                     &&
                                     ( ( ncn == 1 ) || ( ncn == 3 ) || ( ncn == 4 ) )
                                     &&
                                     ( ( bpp == 8 ) || ( bpp == 16 ) )
                                     &&
                                     (tile_height0 == (uint32_t) m_height) // single strip
                                     &&
                                     (
                                         (photometric == PHOTOMETRIC_MINISWHITE)
                                         ||
                                         (photometric == PHOTOMETRIC_MINISBLACK)
                                         ||
                                         (photometric == PHOTOMETRIC_RGB)
                                     )
                                     &&
                                     (planerConfig != PLANARCONFIG_SEPARATE);

                    // Currently only EXTRASAMPLE_ASSOCALPHA is supported.
                    if ( doReadScanline && ( ncn == 4 ) )
                    {
                        uint16_t extra_samples_num;
                        uint16_t *extra_samples = NULL;
                        CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, &extra_samples_num, &extra_samples ));
                        doReadScanline = ( extra_samples_num == 1 ) && ( extra_samples[0] == EXTRASAMPLE_ASSOCALPHA );
                    }
                }

                if ( !doReadScanline )
                {
                    // we will use TIFFReadRGBA* functions, so allocate temporary buffer for 32bit RGBA
                    bpp = 8;
                    ncn = 4;

                    char errmsg[1024];
                    if (!TIFFRGBAImageOK(tif, errmsg))
                    {
                        CV_LOG_WARNING(NULL, "OpenCV TIFF: TIFFRGBAImageOK: " << errmsg);
                        close();
                        return false;
                    }
                }
            }
            else if (dst_bpp == 16)
            {
                // if buffer_size >= MAX_TILE_SIZE*95%,
                // we will use TIFFReadScanline function.
                if (
                    (uint64_t)tile_width0 * tile_height0 * ncn * std::max(1, (int)(bpp / bitsPerByte))
                    >=
                    MAX_TILE_SIZE * 95 / 100
                )
                {
                    uint16_t planerConfig = (uint16_t)-1;
                    CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planerConfig));

                    doReadScanline = (!is_tiled) // no tile
                                     &&
                                     ( ( ncn == 1 ) || ( ncn == 3 ) || ( ncn == 4 ) )
                                     &&
                                     ( ( bpp == 8 ) || ( bpp == 16 ) )
                                     &&
                                     (tile_height0 == (uint32_t) m_height) // single strip
                                     &&
                                     (
                                         (photometric == PHOTOMETRIC_MINISWHITE)
                                         ||
                                         (photometric == PHOTOMETRIC_MINISBLACK)
                                         ||
                                         (photometric == PHOTOMETRIC_RGB)
                                     )
                                     &&
                                     (planerConfig != PLANARCONFIG_SEPARATE);

                    // Currently only EXTRASAMPLE_ASSOCALPHA is supported.
                    if ( doReadScanline && ( ncn == 4 ) )
                    {
                        uint16_t extra_samples_num;
                        uint16_t *extra_samples = NULL;
                        CV_TIFF_CHECK_CALL(TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, &extra_samples_num, &extra_samples ));
                        doReadScanline = ( extra_samples_num == 1 ) && ( extra_samples[0] == EXTRASAMPLE_ASSOCALPHA );
                    }
                }
            }
            else if (dst_bpp == 32 || dst_bpp == 64)
            {
                CV_Assert(ncn == img.channels());
                CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP));
            }

            if ( doReadScanline )
            {
                // Read each scanlines.
                tile_height0 = 1;
            }

            const size_t src_buffer_bytes_per_row = divUp(static_cast<size_t>(ncn * tile_width0 * bpp), static_cast<size_t>(bitsPerByte));
            const size_t src_buffer_size = tile_height0 * src_buffer_bytes_per_row;
            CV_CheckLT(src_buffer_size, MAX_TILE_SIZE, "buffer_size is too large: >= 1Gb");
            const size_t src_buffer_unpacked_bytes_per_row = divUp(static_cast<size_t>(ncn * tile_width0 * dst_bpp), static_cast<size_t>(bitsPerByte));
            const size_t src_buffer_unpacked_size = tile_height0 * src_buffer_unpacked_bytes_per_row;
            const bool needsUnpacking = (bpp < dst_bpp);
            AutoBuffer<uchar> _src_buffer(src_buffer_size);
            uchar* src_buffer = _src_buffer.data();
            AutoBuffer<uchar> _src_buffer_unpacked(needsUnpacking ? src_buffer_unpacked_size : 0);
            uchar* src_buffer_unpacked = needsUnpacking ? _src_buffer_unpacked.data() : nullptr;

            if ( doReadScanline )
            {
                CV_CheckGE(src_buffer_size,
                           static_cast<size_t>(TIFFScanlineSize(tif)),
                           "src_buffer_size is smaller than TIFFScanlineSize().");
            }

            int tileidx = 0;

            #define MAKE_FLAG(a,b) ( (a << 8) | b )
            const int  convert_flag = MAKE_FLAG( ncn, wanted_channels );
            const bool isNeedConvert16to8 = ( doReadScanline ) && ( bpp == 16 ) && ( dst_bpp == 8);

            for (int y = 0; y < m_height; y += (int)tile_height0)
            {
                int tile_height = std::min((int)tile_height0, m_height - y);

                const int img_y = vert_flip ? m_height - y - tile_height : y;

                for(int x = 0; x < m_width; x += (int)tile_width0, tileidx++)
                {
                    int tile_width = std::min((int)tile_width0, m_width - x);

                    switch (dst_bpp)
                    {
                        case 8:
                        {
                            uchar* bstart = src_buffer;
                            if (doReadScanline)
                            {
                                CV_TIFF_CHECK_CALL((int)TIFFReadScanline(tif, (uint32_t*)src_buffer, y) >= 0);

                                if ( isNeedConvert16to8 )
                                {
                                    // Convert buffer image from 16bit to 8bit.
                                    int ix;
                                    for ( ix = 0 ; ix < tile_width * ncn - 4; ix += 4 )
                                    {
                                        src_buffer[ ix     ] = src_buffer[ ix * 2 + 1 ];
                                        src_buffer[ ix + 1 ] = src_buffer[ ix * 2 + 3 ];
                                        src_buffer[ ix + 2 ] = src_buffer[ ix * 2 + 5 ];
                                        src_buffer[ ix + 3 ] = src_buffer[ ix * 2 + 7 ];
                                    }

                                    for (        ; ix < tile_width * ncn ; ix ++ )
                                    {
                                        src_buffer[ ix ] = src_buffer[ ix * 2 + 1];
                                    }
                                }
                            }
                            else if (!is_tiled)
                            {
                                CV_TIFF_CHECK_CALL(TIFFReadRGBAStrip(tif, y, (uint32_t*)src_buffer));
                            }
                            else
                            {
                                CV_TIFF_CHECK_CALL(TIFFReadRGBATile(tif, x, y, (uint32_t*)src_buffer));
                                // Tiles fill the buffer from the bottom up
                                bstart += (tile_height0 - tile_height) * tile_width0 * 4;
                            }

                            uchar* img_line_buffer = (uchar*) img.ptr(y, 0);

                            for (int i = 0; i < tile_height; i++)
                            {
                                if (doReadScanline)
                                {
                                    switch ( convert_flag )
                                    {
                                    case MAKE_FLAG( 1, 1 ): // GRAY to GRAY
                                        std::memcpy( (void*) img_line_buffer,
                                                     (void*) bstart,
                                                     tile_width * sizeof(uchar) );
                                        break;

                                    case MAKE_FLAG( 1, 3 ): // GRAY to BGR
                                        icvCvt_Gray2BGR_8u_C1C3R( bstart, 0,
                                                img_line_buffer, 0,
                                                Size(tile_width, 1) );
                                        break;

                                    case MAKE_FLAG( 3, 1): // RGB to GRAY
                                        icvCvt_BGR2Gray_8u_C3C1R( bstart, 0,
                                                img_line_buffer, 0,
                                                Size(tile_width, 1) );
                                        break;

                                    case MAKE_FLAG( 3, 3 ): // RGB to BGR
                                        if (m_use_rgb)
                                            std::memcpy( (void*) img_line_buffer,
                                                         (void*) bstart,
                                                         tile_width * sizeof(uchar) );
                                        else
                                            icvCvt_BGR2RGB_8u_C3R( bstart, 0,
                                                    img_line_buffer, 0,
                                                    Size(tile_width, 1) );
                                        break;

                                    case MAKE_FLAG( 4, 1 ): // RGBA to GRAY
                                        icvCvt_BGRA2Gray_8u_C4C1R( bstart, 0,
                                                img_line_buffer, 0,
                                                Size(tile_width, 1) );
                                        break;

                                    case MAKE_FLAG( 4, 3 ): // RGBA to BGR
                                        icvCvt_BGRA2BGR_8u_C4C3R( bstart, 0,
                                                img_line_buffer, 0,
                                                Size(tile_width, 1), m_use_rgb ? 0 : 2);
                                        break;

                                    case MAKE_FLAG( 4, 4 ): // RGBA to BGRA
                                        icvCvt_BGRA2RGBA_8u_C4R(bstart, 0,
                                                img_line_buffer, 0,
                                                Size(tile_width, 1) );
                                        break;

                                    default:
                                        CV_LOG_ONCE_ERROR(NULL, "OpenCV TIFF(line " << __LINE__ << "): Unsupported convertion :"
                                                               << " bpp = " << bpp << " ncn = " << (int)ncn
                                                               << " wanted_channels =" << wanted_channels  );
                                        break;
                                    }
                                    #undef MAKE_FLAG
                                }
                                else if (color)
                                {
                                    if (wanted_channels == 4)
                                    {
                                        icvCvt_BGRA2RGBA_8u_C4R(bstart + i*tile_width0*4, 0,
                                                img.ptr(img_y + tile_height - i - 1, x), 0,
                                                Size(tile_width, 1) );
                                    }
                                    else
                                    {
                                        CV_CheckEQ(wanted_channels, 3, "TIFF-8bpp: BGR/BGRA images are supported only");
                                        icvCvt_BGRA2BGR_8u_C4C3R(bstart + i*tile_width0*4, 0,
                                                img.ptr(img_y + tile_height - i - 1, x), 0,
                                                Size(tile_width, 1), m_use_rgb ? 0 : 2);
                                    }
                                }
                                else
                                {
                                    CV_CheckEQ(wanted_channels, 1, "");
                                    icvCvt_BGRA2Gray_8u_C4C1R( bstart + i*tile_width0*4, 0,
                                            img.ptr(img_y + tile_height - i - 1, x), 0,
                                            Size(tile_width, 1), 2);
                                }
                            }
                            break;
                        }

                        case 16:
                        {
                            if (doReadScanline)
                            {
                                CV_TIFF_CHECK_CALL((int)TIFFReadScanline(tif, (uint32_t*)src_buffer, y) >= 0);
                            }
                            else if (!is_tiled)
                            {
                                CV_TIFF_CHECK_CALL((int)TIFFReadEncodedStrip(tif, tileidx, (uint32_t*)src_buffer, src_buffer_size) >= 0);
                            }
                            else
                            {
                                CV_TIFF_CHECK_CALL((int)TIFFReadEncodedTile(tif, tileidx, (uint32_t*)src_buffer, src_buffer_size) >= 0);
                            }

                            for (int i = 0; i < tile_height; i++)
                            {
                                ushort* buffer16 = (ushort*)(src_buffer+i*src_buffer_bytes_per_row);
                                if (needsUnpacking)
                                {
                                    const uchar* src_packed = src_buffer+i*src_buffer_bytes_per_row;
                                    uchar* dst_unpacked = src_buffer_unpacked+i*src_buffer_unpacked_bytes_per_row;
                                    if (bpp == 10)
                                        _unpack10To16(src_packed, src_packed+src_buffer_bytes_per_row,
                                                      (ushort*)dst_unpacked, (ushort*)(dst_unpacked+src_buffer_unpacked_bytes_per_row),
                                                      ncn * tile_width0);
                                    else if (bpp == 12)
                                        _unpack12To16(src_packed, src_packed+src_buffer_bytes_per_row,
                                                      (ushort*)dst_unpacked, (ushort*)(dst_unpacked+src_buffer_unpacked_bytes_per_row),
                                                      ncn * tile_width0);
                                    else if (bpp == 14)
                                        _unpack14To16(src_packed, src_packed+src_buffer_bytes_per_row,
                                                      (ushort*)dst_unpacked, (ushort*)(dst_unpacked+src_buffer_unpacked_bytes_per_row),
                                                      ncn * tile_width0);
                                    buffer16 = (ushort*)dst_unpacked;
                                }

                                if (color)
                                {
                                    if (ncn == 1)
                                    {
                                        CV_CheckEQ(wanted_channels, 3, "");
                                        icvCvt_Gray2BGR_16u_C1C3R(buffer16, 0,
                                                img.ptr<ushort>(img_y + i, x), 0,
                                                Size(tile_width, 1));
                                    }
                                    else if (ncn == 3)
                                    {
                                        CV_CheckEQ(wanted_channels, 3, "");
                                        if (m_use_rgb)
                                            std::memcpy(buffer16, img.ptr<ushort>(img_y + i, x), tile_width * sizeof(ushort));
                                        else
                                            icvCvt_RGB2BGR_16u_C3R(buffer16, 0,
                                                    img.ptr<ushort>(img_y + i, x), 0,
                                                    Size(tile_width, 1));
                                    }
                                    else if (ncn == 4)
                                    {
                                        if (wanted_channels == 4)
                                        {
                                            icvCvt_BGRA2RGBA_16u_C4R(buffer16, 0,
                                                img.ptr<ushort>(img_y + i, x), 0,
                                                Size(tile_width, 1));
                                        }
                                        else
                                        {
                                            CV_CheckEQ(wanted_channels, 3, "TIFF-16bpp: BGR/BGRA images are supported only");
                                            icvCvt_BGRA2BGR_16u_C4C3R(buffer16, 0,
                                                img.ptr<ushort>(img_y + i, x), 0,
                                                Size(tile_width, 1), m_use_rgb ? 0 : 2);
                                        }
                                    }
                                    else
                                    {
                                        CV_Error(Error::StsError, "Not supported");
                                    }
                                }
                                else
                                {
                                    CV_CheckEQ(wanted_channels, 1, "");
                                    if( ncn == 1 )
                                    {
                                        std::memcpy(img.ptr<ushort>(img_y + i, x),
                                                    buffer16,
                                                    tile_width*sizeof(ushort));
                                    }
                                    else
                                    {
                                        icvCvt_BGRA2Gray_16u_CnC1R(buffer16, 0,
                                                img.ptr<ushort>(img_y + i, x), 0,
                                                Size(tile_width, 1), ncn, 2);
                                    }
                                }
                            }
                            break;
                        }

                        case 32:
                        case 64:
                        {
                            if( !is_tiled )
                            {
                                CV_TIFF_CHECK_CALL((int)TIFFReadEncodedStrip(tif, tileidx, src_buffer, src_buffer_size) >= 0);
                            }
                            else
                            {
                                CV_TIFF_CHECK_CALL((int)TIFFReadEncodedTile(tif, tileidx, src_buffer, src_buffer_size) >= 0);
                            }

                            Mat m_tile(Size(tile_width0, tile_height0), CV_MAKETYPE((dst_bpp == 32) ? (depth == CV_32S ? CV_32S : CV_32F) : CV_64F, ncn), src_buffer);
                            Rect roi_tile(0, 0, tile_width, tile_height);
                            Rect roi_img(x, img_y, tile_width, tile_height);
                            if (!m_hdr && ncn == 3 && !m_use_rgb)
                                extend_cvtColor(m_tile(roi_tile), img(roi_img), COLOR_RGB2BGR);
                            else if (!m_hdr && ncn == 4)
                                extend_cvtColor(m_tile(roi_tile), img(roi_img), COLOR_RGBA2BGRA);
                            else
                                m_tile(roi_tile).copyTo(img(roi_img));
                            break;
                        }
                        default:
                        {
                            CV_Assert(0 && "OpenCV TIFF: unsupported depth");
                        }
                    }  // switch (dst_bpp)
                }  // for x
            }  // for y
        }
        if (bpp < dst_bpp)
          img *= (1<<(dst_bpp-bpp));

        // If TIFFReadRGBA* function is used -> fixOrientationPartial().
        // Otherwise                         -> fixOrientationFull().
        fixOrientation(img, img_orientation,
                       ( ( dst_bpp != 8 ) && ( !doReadScanline ) ) );
    }

    if (m_hdr && depth >= CV_32F)
    {
        CV_Assert(photometric == PHOTOMETRIC_LOGLUV);
        if (m_use_rgb)
            cvtColor(img, img, COLOR_XYZ2RGB);
        else
            cvtColor(img, img, COLOR_XYZ2BGR);
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////

TiffEncoder::TiffEncoder()
{
    m_description = "TIFF Files (*.tiff;*.tif)";
    m_buf_supported = true;
}

TiffEncoder::~TiffEncoder()
{
}

ImageEncoder TiffEncoder::newEncoder() const
{
    cv_tiffSetErrorHandler();
    return makePtr<TiffEncoder>();
}

bool TiffEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F;
}

class TiffEncoderBufHelper
{
public:

    TiffEncoderBufHelper(std::vector<uchar> *buf)
            : m_buf(buf), m_buf_pos(0)
    {}

    TIFF* open ()
    {
        // do NOT put "wb" as the mode, because the b means "big endian" mode, not "binary" mode.
        // http://www.simplesystems.org/libtiff/functions/TIFFOpen.html
        return TIFFClientOpen( "", "w", reinterpret_cast<thandle_t>(this), &TiffEncoderBufHelper::read,
                               &TiffEncoderBufHelper::write, &TiffEncoderBufHelper::seek,
                               &TiffEncoderBufHelper::close, &TiffEncoderBufHelper::size,
                               /*map=*/0, /*unmap=*/0 );
    }

    static tmsize_t read( thandle_t handle, void* buffer, tmsize_t n )
    {
        // Used for imencodemulti() to stores multi-images.
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        size_t begin = (size_t)helper->m_buf_pos;
        size_t end = begin + n;
        CV_CheckGT( helper->m_buf->size(), end , "do not be over-run buffer");
        std::memcpy(buffer, &(*helper->m_buf)[begin], n);
        helper->m_buf_pos = end;
        return n;
    }

    static tmsize_t write( thandle_t handle, void* buffer, tmsize_t n )
    {
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        size_t begin = (size_t)helper->m_buf_pos;
        size_t end = begin + n;
        if ( helper->m_buf->size() < end )
        {
            helper->m_buf->resize(end);
        }
        std::memcpy(&(*helper->m_buf)[begin], buffer, n);
        helper->m_buf_pos = end;
        return n;
    }

    static toff_t seek( thandle_t handle, toff_t offset, int whence )
    {
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        const toff_t size = helper->m_buf->size();
        toff_t new_pos = helper->m_buf_pos;
        switch (whence)
        {
            case SEEK_SET:
                new_pos = offset;
                break;
            case SEEK_CUR:
                new_pos += offset;
                break;
            case SEEK_END:
                new_pos = size + offset;
                break;
        }
        helper->m_buf_pos = new_pos;
        return new_pos;
    }

    static toff_t size( thandle_t handle )
    {
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        return helper->m_buf->size();
    }

    static int close( thandle_t /*handle*/ )
    {
        // Do nothing.
        return 0;
    }

private:

    std::vector<uchar>* m_buf;
    toff_t m_buf_pos;
};

static bool readParam(const std::vector<int>& params, int key, int& value)
{
    for (size_t i = 0; i + 1 < params.size(); i += 2)
    {
        if (params[i] == key)
        {
            value = params[i + 1];
            return true;
        }
    }
    return false;
}

bool TiffEncoder::writeLibTiff( const std::vector<Mat>& img_vec, const std::vector<int>& params)
{
    // do NOT put "wb" as the mode, because the b means "big endian" mode, not "binary" mode.
    // http://www.simplesystems.org/libtiff/functions/TIFFOpen.html
    TIFF* tif = NULL;

    TiffEncoderBufHelper buf_helper(m_buf);
    if ( m_buf )
    {
        tif = buf_helper.open();
    }
    else
    {
        tif = TIFFOpen(m_filename.c_str(), "w");
    }
    if (!tif)
    {
        return false;
    }
    cv::Ptr<void> tif_cleanup(tif, cv_tiffCloseHandle);

    //Settings that matter to all images
    int compression = COMPRESSION_LZW;
    int predictor = PREDICTOR_HORIZONTAL;
    int resUnit = -1, dpiX = -1, dpiY = -1;

    readParam(params, IMWRITE_TIFF_COMPRESSION, compression);
    readParam(params, IMWRITE_TIFF_PREDICTOR, predictor);
    readParam(params, IMWRITE_TIFF_RESUNIT, resUnit);
    readParam(params, IMWRITE_TIFF_XDPI, dpiX);
    readParam(params, IMWRITE_TIFF_YDPI, dpiY);

    //Iterate through each image in the vector and write them out as Tiff directories
    for (size_t page = 0; page < img_vec.size(); page++)
    {
        const Mat& img = img_vec[page];
        CV_Assert(!img.empty());
        int channels = img.channels();
        int width = img.cols, height = img.rows;
        int type = img.type();
        int depth = CV_MAT_DEPTH(type);
        CV_CheckType(type, depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F, "");
        CV_CheckType(type, channels >= 1 && channels <= 4, "");

        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width));
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height));

        if (img_vec.size() > 1)
        {
            CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE));
            CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_PAGENUMBER, page, img_vec.size()));
        }

        int compression_param = -1;  // OPENCV_FUTURE
        if (type == CV_32FC3 && (!readParam(params, IMWRITE_TIFF_COMPRESSION, compression_param) || compression_param == COMPRESSION_SGILOG))
        {
            if (!write_32FC3_SGILOG(img, tif))
                return false;
            continue;
        }

        int page_compression = compression;

        int bitsPerChannel = -1;
        uint16_t sample_format = SAMPLEFORMAT_INT;
        switch (depth)
        {
            case CV_8U:
                sample_format = SAMPLEFORMAT_UINT;
                /* FALLTHRU */
            case CV_8S:
            {
                bitsPerChannel = 8;
                break;
            }

            case CV_16U:
                sample_format = SAMPLEFORMAT_UINT;
                /* FALLTHRU */
            case CV_16S:
            {
                bitsPerChannel = 16;
                break;
            }

            case CV_32S:
            {
                bitsPerChannel = 32;
                sample_format = SAMPLEFORMAT_INT;
                break;
            }
            case CV_32F:
            {
                bitsPerChannel = 32;
                page_compression = COMPRESSION_NONE;
                sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            }
            case CV_64F:
            {
                bitsPerChannel = 64;
                page_compression = COMPRESSION_NONE;
                sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            }
            default:
            {
                return false;
            }
        }

        const int bitsPerByte = 8;
        size_t fileStep = (width * channels * bitsPerChannel) / bitsPerByte;
        CV_Assert(fileStep > 0);

        int rowsPerStrip = (int)((1 << 13) / fileStep);
        readParam(params, IMWRITE_TIFF_ROWSPERSTRIP, rowsPerStrip);
        rowsPerStrip = std::max(1, std::min(height, rowsPerStrip));

        int colorspace = channels > 1 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerChannel));
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_COMPRESSION, page_compression));
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, colorspace));
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels));
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG));
        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip));

        CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format));

        if (page_compression == COMPRESSION_LZW || page_compression == COMPRESSION_ADOBE_DEFLATE || page_compression == COMPRESSION_DEFLATE)
        {
            CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_PREDICTOR, predictor));
        }

        if (resUnit >= RESUNIT_NONE && resUnit <= RESUNIT_CENTIMETER)
        {
            CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, resUnit));
        }
        if (dpiX >= 0)
        {
            CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_XRESOLUTION, (float)dpiX));
        }
        if (dpiY >= 0)
        {
            CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_YRESOLUTION, (float)dpiY));
        }

        // row buffer, because TIFFWriteScanline modifies the original data!
        size_t scanlineSize = TIFFScanlineSize(tif);
        AutoBuffer<uchar> _buffer(scanlineSize + 32);
        uchar* buffer = _buffer.data(); CV_DbgAssert(buffer);
        Mat m_buffer(Size(width, 1), CV_MAKETYPE(depth, channels), buffer, (size_t)scanlineSize);

        for (int y = 0; y < height; ++y)
        {
            switch (channels)
            {
                case 1:
                {
                    std::memcpy(buffer, img.ptr(y), scanlineSize);
                    break;
                }

                case 3:
                {
                    extend_cvtColor(img(Rect(0, y, width, 1)), (const Mat&)m_buffer, COLOR_BGR2RGB);
                    break;
                }

                case 4:
                {
                    extend_cvtColor(img(Rect(0, y, width, 1)), (const Mat&)m_buffer, COLOR_BGRA2RGBA);
                    break;
                }

                default:
                {
                    CV_Assert(0);
                }
            }

            CV_TIFF_CHECK_CALL(TIFFWriteScanline(tif, buffer, y, 0) == 1);
        }

        CV_TIFF_CHECK_CALL(TIFFWriteDirectory(tif));
    }

    return true;
}

bool TiffEncoder::write_32FC3_SGILOG(const Mat& _img, void* tif_)
{
    TIFF* tif = (TIFF*)tif_;
    CV_Assert(tif);

    Mat img;
    cvtColor(_img, img, COLOR_BGR2XYZ);

    //done by caller: CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.cols));
    //done by caller: CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.rows));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_SGILOG));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_LOGLUV));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_FLOAT));
    CV_TIFF_CHECK_CALL(TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1));
    const int strip_size = 3 * img.cols;
    for (int i = 0; i < img.rows; i++)
    {
        CV_TIFF_CHECK_CALL(TIFFWriteEncodedStrip(tif, i, (tdata_t)img.ptr<float>(i), strip_size * sizeof(float)) != (tsize_t)-1);
    }
    CV_TIFF_CHECK_CALL(TIFFWriteDirectory(tif));
    return true;
}

bool TiffEncoder::writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params)
{
    return writeLibTiff(img_vec, params);
}

bool  TiffEncoder::write( const Mat& img, const std::vector<int>& params)
{
    int type = img.type();
    int depth = CV_MAT_DEPTH(type);

    CV_CheckType(type, depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S || depth == CV_32S || depth == CV_32F || depth == CV_64F, "");

    std::vector<Mat> img_vec;
    img_vec.push_back(img);
    return writeLibTiff(img_vec, params);
}

static void extend_cvtColor( InputArray _src, OutputArray _dst, int code )
{
    CV_Assert( !_src.empty() );
    CV_Assert( _src.dims() == 2 );

    // This function extend_cvtColor reorders the src channels with only thg limited condition.
    // Otherwise, it calls cvtColor.

    const int stype = _src.type();
    if(!
        (
            (
                ( stype == CV_8SC3  ) || ( stype == CV_8SC4  ) ||
                ( stype == CV_16SC3 ) || ( stype == CV_16SC4 ) ||
                ( stype == CV_32SC3 ) || ( stype == CV_32SC4 ) ||
                ( stype == CV_64FC3 ) || ( stype == CV_64FC4 )
            )
            &&
            (
                ( code == COLOR_BGR2RGB ) || ( code == COLOR_BGRA2RGBA )
            )
        )
    )
    {
        cvtColor( _src, _dst, code );
        return;
    }

    Mat src = _src.getMat();

    // cv::mixChannels requires the output arrays to be pre-allocated before calling the function.
    _dst.create( _src.size(), stype );
    Mat dst = _dst.getMat();

    // BGR to RGB or BGRA to RGBA
    //   src[0] -> dst[2]
    //   src[1] -> dst[1]
    //   src[2] -> dst[0]
    //   src[3] -> dst[3] if src has alpha channel.
    std::vector<int> fromTo;
    fromTo.push_back(0); fromTo.push_back(2);
    fromTo.push_back(1); fromTo.push_back(1);
    fromTo.push_back(2); fromTo.push_back(0);
    if ( code == COLOR_BGRA2RGBA )
    {
        fromTo.push_back(3); fromTo.push_back(3);
    }

    cv::mixChannels( src, dst, fromTo );
}

} // namespace

#endif
