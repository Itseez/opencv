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

#ifndef _GRFMT_PNG_H_
#define _GRFMT_PNG_H_

#ifdef HAVE_PNG

#include "grfmt_base.hpp"
#include "bitstrm.hpp"
#include <png.h>
#include <zlib.h>
#include <vector>

namespace cv
{

struct Chunk { std::vector<unsigned char> p; };
struct OP { unsigned char* p; uint32_t size; int x, y, w, h, valid, filters; };

typedef struct {
  unsigned char r, g, b;
} rgb;

class APNGFrame {
public:

    APNGFrame();

    // Destructor
    ~APNGFrame();

    bool setMat(const cv::Mat& src, unsigned delayNum = 1, unsigned delayDen = 1000);

    // Getters and Setters
    unsigned char* getPixels() const { return _pixels; }
    void setPixels(unsigned char* pixels);

    unsigned int getWidth() const { return _width; }
    void setWidth(unsigned int width);

    unsigned int getHeight() const { return _height; }
    void setHeight(unsigned int height);

    unsigned char getColorType() const { return _colorType; }
    void setColorType(unsigned char colorType);

    rgb* getPalette() { return _palette; }
    void setPalette(const rgb* palette);

    unsigned char* getTransparency() { return _transparency; }
    void setTransparency(const unsigned char* transparency);

    int getPaletteSize() const { return _paletteSize; }
    void setPaletteSize(int paletteSize);

    int getTransparencySize() const { return _transparencySize; }
    void setTransparencySize(int transparencySize);

    unsigned int getDelayNum() const { return _delayNum; }
    void setDelayNum(unsigned int delayNum);

    unsigned int getDelayDen() const { return _delayDen; }
    void setDelayDen(unsigned int delayDen);

    std::vector<png_bytep>& getRows() { return _rows; }

private:
    unsigned char* _pixels;
    unsigned int _width;
    unsigned int _height;
    unsigned char _colorType;
    rgb _palette[256];
    unsigned char _transparency[256];
    int _paletteSize;
    int _transparencySize;
    unsigned int _delayNum;
    unsigned int _delayDen;
    std::vector<png_bytep> _rows;
};

class PngDecoder CV_FINAL : public BaseImageDecoder
{
public:
    PngDecoder();
    virtual ~PngDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    bool  nextPage() CV_OVERRIDE;

    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:
    static void readDataFromBuf(void* png_ptr, uchar* dst, size_t size);
    static void info_fn(png_structp png_ptr, png_infop info_ptr);
    static void row_fn(png_structp png_ptr, png_bytep new_row, png_uint_32 row_num, int pass);
    bool processing_start(void* frame_ptr, const Mat& img);
    bool processing_finish();
    void compose_frame(std::vector<png_bytep>& rows_dst, const std::vector<png_bytep>& rows_src, unsigned char bop, uint32_t x, uint32_t y, uint32_t w, uint32_t h, Mat& img);
    bool read_from_io(void* buffer, size_t num_bytes);
    uint32_t  read_chunk(Chunk& chunk);

    struct PngPtrs {
        public:
            PngPtrs() {
                png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, 0, 0, 0 );
                if (png_ptr) {
                    info_ptr = png_create_info_struct( png_ptr );
                    end_info = png_create_info_struct( png_ptr );
                } else {
                    info_ptr = end_info = nullptr;
                }
            }
            ~PngPtrs() {
                clear();
            }
            PngPtrs& operator=(PngPtrs&& other) {
                clear();
                png_ptr = other.png_ptr;
                info_ptr = other.info_ptr;
                end_info = other.end_info;
                other.png_ptr = nullptr;
                other.info_ptr = other.end_info = nullptr;
                return *this;
            }
            void clear() {
                if (png_ptr) {
                    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
                    png_ptr = nullptr;
                    info_ptr = end_info = nullptr;
                }
            }
            png_structp getPng() const { return png_ptr; }
            png_infop getInfo() const { return info_ptr; }
            png_infop getEndInfo() const { return end_info; }
        private:
            png_structp png_ptr; // pointer to decompression structure
            png_infop info_ptr; // pointer to image information structure
            png_infop end_info; // pointer to one more image information structure
    };

    PngPtrs m_png_ptrs;
    int   m_bit_depth;
    FILE* m_f;
    int   m_color_type;
    Chunk m_chunkIHDR;
    int   m_frame_no;
    size_t m_buf_pos;
    std::vector<Chunk> m_chunksInfo;
    APNGFrame frameRaw;
    APNGFrame frameNext;
    APNGFrame frameCur;
    Mat m_mat_raw;
    Mat m_mat_next;
    uint32_t w0;
    uint32_t h0;
    uint32_t x0;
    uint32_t y0;
    uint32_t delay_num;
    uint32_t delay_den;
    uint32_t dop;
    uint32_t bop;
    bool m_is_fcTL_loaded;
    bool m_is_IDAT_loaded;
};


class PngEncoder CV_FINAL : public BaseImageEncoder
{
public:
    PngEncoder();
    virtual ~PngEncoder();

    bool isFormatSupported( int depth ) const CV_OVERRIDE;
    bool write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    bool writeanimation(const Animation& animinfo, const std::vector<int>& params) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    static void writeDataToBuf(void* png_ptr, unsigned char* src, size_t size);
    static void flushBuf(void* png_ptr);
    size_t write_to_io(void const* _Buffer, size_t  _ElementSize, size_t _ElementCount, FILE* _Stream);

private:
    void writeChunk(FILE* f, const char* name, unsigned char* data, uint32_t length);
    void writeIDATs(FILE* f, int frame, unsigned char* data, uint32_t length, uint32_t idat_size);
    void processRect(unsigned char* row, int rowbytes, int bpp, int stride, int h, unsigned char* rows);
    void deflateRectFin(unsigned char* zbuf, uint32_t* zsize, int bpp, int stride, unsigned char* rows, int zbuf_size, int n);
    void deflateRectOp(unsigned char* pdata, int x, int y, int w, int h, int bpp, int stride, int zbuf_size, int n);
    bool getRect(uint32_t w, uint32_t h, unsigned char* pimage1, unsigned char* pimage2, unsigned char* ptemp, uint32_t bpp, uint32_t stride, int zbuf_size, uint32_t has_tcolor, uint32_t tcolor, int n);

    AutoBuffer<unsigned char> op_zbuf1;
    AutoBuffer<unsigned char> op_zbuf2;
    AutoBuffer<unsigned char> row_buf;
    AutoBuffer<unsigned char> sub_row;
    AutoBuffer<unsigned char> up_row;
    AutoBuffer<unsigned char> avg_row;
    AutoBuffer<unsigned char> paeth_row;
    z_stream       op_zstream1;
    z_stream       op_zstream2;
    OP             op[6];
    rgb            palette[256];
    unsigned char  trns[256];
    uint32_t       palsize, trnssize;
    uint32_t       next_seq_num;
};

}

#endif

#endif/*_GRFMT_PNG_H_*/
