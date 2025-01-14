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

#ifndef _UTILS_H_
#define _UTILS_H_

namespace cv {

int validateToInt(size_t step);

template <typename _Tp> static inline
size_t safeCastToSizeT(const _Tp v_origin, const char* msg)
{
    const size_t value_cast = (size_t)v_origin;
    if ((_Tp)value_cast != v_origin)
        CV_Error(cv::Error::StsError, msg ? msg : "Can't cast value into size_t");
    return value_cast;
}

struct PaletteEntry
{
    unsigned char b, g, r, a;
};

#define WRITE_PIX( ptr, clr )       \
    (((uchar*)(ptr))[0] = (clr).b,  \
     ((uchar*)(ptr))[1] = (clr).g,  \
     ((uchar*)(ptr))[2] = (clr).r)

#define  descale(x,n)  (((x) + (1 << ((n)-1))) >> (n))
#define  saturate(x)   (uchar)(((x) & ~255) == 0 ? (x) : ~((x)>>31))

void icvCvt_BGR2Gray_8u_C3C1R( const uchar* bgr, int bgr_step,
                               uchar* gray, int gray_step,
                               Size size, int swap_rb=0 );
void icvCvt_BGRA2Gray_8u_C4C1R( const uchar* bgra, int bgra_step,
                                uchar* gray, int gray_step,
                                Size size, int swap_rb=0 );
void icvCvt_BGRA2Gray_16u_CnC1R( const ushort* bgra, int bgra_step,
                               ushort* gray, int gray_step,
                               Size size, int ncn, int swap_rb=0 );

void icvCvt_Gray2BGR_8u_C1C3R( const uchar* gray, int gray_step,
                               uchar* bgr, int bgr_step, Size size );
void icvCvt_Gray2BGR_16u_C1C3R( const ushort* gray, int gray_step,
                               ushort* bgr, int bgr_step, Size size );

void icvCvt_BGRA2BGR_8u_C4C3R( const uchar* bgra, int bgra_step,
                               uchar* bgr, int bgr_step,
                               Size size, int swap_rb=0 );
void icvCvt_BGRA2BGR_16u_C4C3R( const ushort* bgra, int bgra_step,
                               ushort* bgr, int bgr_step,
                               Size size, int _swap_rb );

void icvCvt_BGR2RGB_8u_C3R( const uchar* bgr, int bgr_step,
                            uchar* rgb, int rgb_step, Size size );
#define icvCvt_RGB2BGR_8u_C3R icvCvt_BGR2RGB_8u_C3R
void icvCvt_BGR2RGB_16u_C3R( const ushort* bgr, int bgr_step,
                             ushort* rgb, int rgb_step, Size size );
#define icvCvt_RGB2BGR_16u_C3R icvCvt_BGR2RGB_16u_C3R

void icvCvt_BGRA2RGBA_8u_C4R( const uchar* bgra, int bgra_step,
                              uchar* rgba, int rgba_step, Size size );
#define icvCvt_RGBA2BGRA_8u_C4R icvCvt_BGRA2RGBA_8u_C4R

void icvCvt_BGRA2RGBA_16u_C4R( const ushort* bgra, int bgra_step,
                               ushort* rgba, int rgba_step, Size size );
#define icvCvt_RGBA2BGRA_16u_C4R icvCvt_BGRA2RGBA_16u_C4R

void icvCvt_BGR5552Gray_8u_C2C1R( const uchar* bgr555, int bgr555_step,
                                  uchar* gray, int gray_step, Size size );
void icvCvt_BGR5652Gray_8u_C2C1R( const uchar* bgr565, int bgr565_step,
                                  uchar* gray, int gray_step, Size size );
void icvCvt_BGR5552BGR_8u_C2C3R( const uchar* bgr555, int bgr555_step,
                                 uchar* bgr, int bgr_step, Size size );
void icvCvt_BGR5652BGR_8u_C2C3R( const uchar* bgr565, int bgr565_step,
                                 uchar* bgr, int bgr_step, Size size );
void icvCvt_CMYK2BGR_8u_C4C3R( const uchar* cmyk, int cmyk_step,
                               uchar* bgr, int bgr_step, Size size );
void icvCvt_CMYK2RGB_8u_C4C3R( const uchar* cmyk, int cmyk_step,
                               uchar* rgb, int rgb_step, Size size );
void icvCvt_CMYK2Gray_8u_C4C1R( const uchar* ycck, int ycck_step,
                                uchar* gray, int gray_step, Size size );

void  FillGrayPalette( PaletteEntry* palette, int bpp, bool negative = false );
bool  IsColorPalette( PaletteEntry* palette, int bpp );
void  CvtPaletteToGray( const PaletteEntry* palette, uchar* grayPalette, int entries );
uchar* FillUniColor( uchar* data, uchar*& line_end, int step, int width3,
                     int& y, int height, int count3, PaletteEntry clr );
uchar* FillUniGray( uchar* data, uchar*& line_end, int step, int width3,
                     int& y, int height, int count3, uchar clr );

uchar* FillColorRow8( uchar* data, uchar* indices, int len, PaletteEntry* palette );
uchar* FillGrayRow8( uchar* data, uchar* indices, int len, uchar* palette );
uchar* FillColorRow4( uchar* data, uchar* indices, int len, PaletteEntry* palette );
uchar* FillGrayRow4( uchar* data, uchar* indices, int len, uchar* palette );
uchar* FillColorRow1( uchar* data, uchar* indices, int len, PaletteEntry* palette );
uchar* FillGrayRow1( uchar* data, uchar* indices, int len, uchar* palette );

CV_INLINE bool  isBigEndian( void )
{
#ifdef WORDS_BIGENDIAN
    return true;
#else
    return false;
#endif
}

}  // namespace

#endif/*_UTILS_H_*/
