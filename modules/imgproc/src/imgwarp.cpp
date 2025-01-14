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
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "hal_replacement.hpp"
#include <opencv2/core/utils/configuration.private.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/softfloat.hpp"
#include "imgwarp.hpp"

#include "warp_kernels.simd.hpp"
#include "warp_kernels.simd_declarations.hpp"

using namespace cv;

namespace cv
{

#if defined (HAVE_IPP) && (!IPP_DISABLE_WARPAFFINE || !IPP_DISABLE_WARPPERSPECTIVE || !IPP_DISABLE_REMAP)
typedef IppStatus (CV_STDCALL* ippiSetFunc)(const void*, void *, int, IppiSize);

template <int channels, typename Type>
bool IPPSetSimple(cv::Scalar value, void *dataPointer, int step, IppiSize &size, ippiSetFunc func)
{
    CV_INSTRUMENT_REGION_IPP();

    Type values[channels];
    for( int i = 0; i < channels; i++ )
        values[i] = saturate_cast<Type>(value[i]);
    return func(values, dataPointer, step, size) >= 0;
}

static bool IPPSet(const cv::Scalar &value, void *dataPointer, int step, IppiSize &size, int channels, int depth)
{
    CV_INSTRUMENT_REGION_IPP();

    if( channels == 1 )
    {
        switch( depth )
        {
        case CV_8U:
            return CV_INSTRUMENT_FUN_IPP(ippiSet_8u_C1R, saturate_cast<Ipp8u>(value[0]), (Ipp8u *)dataPointer, step, size) >= 0;
        case CV_16U:
            return CV_INSTRUMENT_FUN_IPP(ippiSet_16u_C1R, saturate_cast<Ipp16u>(value[0]), (Ipp16u *)dataPointer, step, size) >= 0;
        case CV_32F:
            return CV_INSTRUMENT_FUN_IPP(ippiSet_32f_C1R, saturate_cast<Ipp32f>(value[0]), (Ipp32f *)dataPointer, step, size) >= 0;
        }
    }
    else
    {
        if( channels == 3 )
        {
            switch( depth )
            {
            case CV_8U:
                return IPPSetSimple<3, Ipp8u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_8u_C3R);
            case CV_16U:
                return IPPSetSimple<3, Ipp16u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_16u_C3R);
            case CV_32F:
                return IPPSetSimple<3, Ipp32f>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_32f_C3R);
            }
        }
        else if( channels == 4 )
        {
            switch( depth )
            {
            case CV_8U:
                return IPPSetSimple<4, Ipp8u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_8u_C4R);
            case CV_16U:
                return IPPSetSimple<4, Ipp16u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_16u_C4R);
            case CV_32F:
                return IPPSetSimple<4, Ipp32f>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_32f_C4R);
            }
        }
    }
    return false;
}
#endif

/************** interpolation formulas and tables ***************/

const int INTER_REMAP_COEF_BITS=15;
const int INTER_REMAP_COEF_SCALE=1 << INTER_REMAP_COEF_BITS;

static uchar NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

#if CV_SIMD128
static short BilinearTab_iC4_buf[INTER_TAB_SIZE2+2][2][8];
static short (*BilinearTab_iC4)[2][8] = (short (*)[2][8])alignPtr(BilinearTab_iC4_buf, 16);
#endif

static float BicubicTab_f[INTER_TAB_SIZE2][4][4];
static short BicubicTab_i[INTER_TAB_SIZE2][4][4];

static float Lanczos4Tab_f[INTER_TAB_SIZE2][8][8];
static short Lanczos4Tab_i[INTER_TAB_SIZE2][8][8];

static inline void interpolateLinear( float x, float* coeffs )
{
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void interpolateLanczos4( float x, float* coeffs )
{
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2]=
    {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

    if( x < FLT_EPSILON )
    {
        for( int i = 0; i < 8; i++ )
            coeffs[i] = 0;
        coeffs[3] = 1;
        return;
    }

    float sum = 0;
    double y0=-(x+3)*CV_PI*0.25, s0 = std::sin(y0), c0= std::cos(y0);
    for(int i = 0; i < 8; i++ )
    {
        double y = -(x+3-i)*CV_PI*0.25;
        coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        sum += coeffs[i];
    }

    sum = 1.f/sum;
    for(int i = 0; i < 8; i++ )
        coeffs[i] *= sum;
}

static void initInterTab1D(int method, float* tab, int tabsz)
{
    float scale = 1.f/tabsz;
    if( method == INTER_LINEAR )
    {
        for( int i = 0; i < tabsz; i++, tab += 2 )
            interpolateLinear( i*scale, tab );
    }
    else if( method == INTER_CUBIC )
    {
        for( int i = 0; i < tabsz; i++, tab += 4 )
            interpolateCubic( i*scale, tab );
    }
    else if( method == INTER_LANCZOS4 )
    {
        for( int i = 0; i < tabsz; i++, tab += 8 )
            interpolateLanczos4( i*scale, tab );
    }
    else
        CV_Error( cv::Error::StsBadArg, "Unknown interpolation method" );
}


static const void* initInterTab2D( int method, bool fixpt )
{
    static bool inittab[INTER_MAX+1] = {false};
    float* tab = 0;
    short* itab = 0;
    int ksize = 0;
    if( method == INTER_LINEAR )
        tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize=2;
    else if( method == INTER_CUBIC )
        tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize=4;
    else if( method == INTER_LANCZOS4 )
        tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize=8;
    else
        CV_Error( cv::Error::StsBadArg, "Unknown/unsupported interpolation type" );

    if( !inittab[method] )
    {
        AutoBuffer<float> _tab(8*INTER_TAB_SIZE);
        int i, j, k1, k2;
        initInterTab1D(method, _tab.data(), INTER_TAB_SIZE);
        for( i = 0; i < INTER_TAB_SIZE; i++ )
            for( j = 0; j < INTER_TAB_SIZE; j++, tab += ksize*ksize, itab += ksize*ksize )
            {
                int isum = 0;
                NNDeltaTab_i[i*INTER_TAB_SIZE+j][0] = j < INTER_TAB_SIZE/2;
                NNDeltaTab_i[i*INTER_TAB_SIZE+j][1] = i < INTER_TAB_SIZE/2;

                for( k1 = 0; k1 < ksize; k1++ )
                {
                    float vy = _tab[i*ksize + k1];
                    for( k2 = 0; k2 < ksize; k2++ )
                    {
                        float v = vy*_tab[j*ksize + k2];
                        tab[k1*ksize + k2] = v;
                        isum += itab[k1*ksize + k2] = saturate_cast<short>(v*INTER_REMAP_COEF_SCALE);
                    }
                }

                if( isum != INTER_REMAP_COEF_SCALE )
                {
                    int diff = isum - INTER_REMAP_COEF_SCALE;
                    int ksize2 = ksize/2, Mk1=ksize2, Mk2=ksize2, mk1=ksize2, mk2=ksize2;
                    for( k1 = ksize2; k1 < ksize2+2; k1++ )
                        for( k2 = ksize2; k2 < ksize2+2; k2++ )
                        {
                            if( itab[k1*ksize+k2] < itab[mk1*ksize+mk2] )
                                mk1 = k1, mk2 = k2;
                            else if( itab[k1*ksize+k2] > itab[Mk1*ksize+Mk2] )
                                Mk1 = k1, Mk2 = k2;
                        }
                    if( diff < 0 )
                        itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
                    else
                        itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
                }
            }
        tab -= INTER_TAB_SIZE2*ksize*ksize;
        itab -= INTER_TAB_SIZE2*ksize*ksize;
#if CV_SIMD128
        if( method == INTER_LINEAR )
        {
            for( i = 0; i < INTER_TAB_SIZE2; i++ )
                for( j = 0; j < 4; j++ )
                {
                    BilinearTab_iC4[i][0][j*2] = BilinearTab_i[i][0][0];
                    BilinearTab_iC4[i][0][j*2+1] = BilinearTab_i[i][0][1];
                    BilinearTab_iC4[i][1][j*2] = BilinearTab_i[i][1][0];
                    BilinearTab_iC4[i][1][j*2+1] = BilinearTab_i[i][1][1];
                }
        }
#endif
        inittab[method] = true;
    }
    return fixpt ? (const void*)itab : (const void*)tab;
}

#ifndef __MINGW32__
static bool initAllInterTab2D()
{
    return  initInterTab2D( INTER_LINEAR, false ) &&
            initInterTab2D( INTER_LINEAR, true ) &&
            initInterTab2D( INTER_CUBIC, false ) &&
            initInterTab2D( INTER_CUBIC, true ) &&
            initInterTab2D( INTER_LANCZOS4, false ) &&
            initInterTab2D( INTER_LANCZOS4, true );
}

static volatile bool doInitAllInterTab2D = initAllInterTab2D();
#endif

template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

/****************************************************************************************\
*                       General warping (affine, perspective, remap)                     *
\****************************************************************************************/

template<typename T, bool isRelative>
static void remapNearest( const Mat& _src, Mat& _dst, const Mat& _xy,
                          int borderType, const Scalar& _borderValue, const Point& _offset )
{
    Size ssize = _src.size(), dsize = _dst.size();
    const int cn = _src.channels();
    const T* S0 = _src.ptr<T>();
    T cval[CV_CN_MAX];
    size_t sstep = _src.step/sizeof(S0[0]);

    for(int k = 0; k < cn; k++ )
        cval[k] = saturate_cast<T>(_borderValue[k & 3]);

    unsigned width1 = ssize.width, height1 = ssize.height;

    if( _dst.isContinuous() && _xy.isContinuous() && !isRelative )
    {
        dsize.width *= dsize.height;
        dsize.height = 1;
    }

    for(int dy = 0; dy < dsize.height; dy++ )
    {
        T* D = _dst.ptr<T>(dy);
        const short* XY = _xy.ptr<short>(dy);
        const int off_y = isRelative ? (_offset.y+dy) : 0;
        if( cn == 1 )
        {
            for(int dx = 0; dx < dsize.width; dx++ )
            {
                const int off_x = isRelative ? (_offset.x+dx) : 0;
                int sx = XY[dx*2]+off_x, sy = XY[dx*2+1]+off_y;
                if( (unsigned)sx < width1 && (unsigned)sy < height1 )
                    D[dx] = S0[sy*sstep + sx];
                else
                {
                    if( borderType == BORDER_REPLICATE )
                    {
                        sx = clip(sx, 0, ssize.width);
                        sy = clip(sy, 0, ssize.height);
                        D[dx] = S0[sy*sstep + sx];
                    }
                    else if( borderType == BORDER_CONSTANT )
                        D[dx] = cval[0];
                    else if( borderType != BORDER_TRANSPARENT )
                    {
                        sx = borderInterpolate(sx, ssize.width, borderType);
                        sy = borderInterpolate(sy, ssize.height, borderType);
                        D[dx] = S0[sy*sstep + sx];
                    }
                }
            }
        }
        else
        {
            for(int dx = 0; dx < dsize.width; dx++, D += cn )
            {
                const int off_x = isRelative ? (_offset.x+dx) : 0;
                int sx = XY[dx*2]+off_x, sy = XY[dx*2+1]+off_y;
                const T *S;
                if( (unsigned)sx < width1 && (unsigned)sy < height1 )
                {
                    if( cn == 3 )
                    {
                        S = S0 + sy*sstep + sx*3;
                        D[0] = S[0], D[1] = S[1], D[2] = S[2];
                    }
                    else if( cn == 4 )
                    {
                        S = S0 + sy*sstep + sx*4;
                        D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
                    }
                    else
                    {
                        S = S0 + sy*sstep + sx*cn;
                        for(int k = 0; k < cn; k++ )
                            D[k] = S[k];
                    }
                }
                else if( borderType != BORDER_TRANSPARENT )
                {
                    if( borderType == BORDER_REPLICATE )
                    {
                        sx = clip(sx, 0, ssize.width);
                        sy = clip(sy, 0, ssize.height);
                        S = S0 + sy*sstep + sx*cn;
                    }
                    else if( borderType == BORDER_CONSTANT )
                        S = &cval[0];
                    else
                    {
                        sx = borderInterpolate(sx, ssize.width, borderType);
                        sy = borderInterpolate(sy, ssize.height, borderType);
                        S = S0 + sy*sstep + sx*cn;
                    }
                    for(int k = 0; k < cn; k++ )
                        D[k] = S[k];
                }
            }
        }
    }
}

template<bool>
struct RemapNoVec
{
    int operator()( const Mat&, void*, const short*, const ushort*,
                    const void*, int, cv::Point& ) const { return 0; }
};

#if CV_SIMD128

typedef unsigned short CV_DECL_ALIGNED(1) unaligned_ushort;
typedef int CV_DECL_ALIGNED(1) unaligned_int;

template<bool isRelative>
struct RemapVec_8u
{
    int operator()( const Mat& _src, void* _dst, const short* XY,
                    const ushort* FXY, const void* _wtab, int width, const Point& _offset ) const
    {
        int cn = _src.channels(), x = 0, sstep = (int)_src.step;
        if( (cn != 1 && cn != 3 && cn != 4) || sstep >= 0x8000 )
            return 0;

        const uchar *S0 = _src.ptr(), *S1 = _src.ptr(1);
        const short* wtab = cn == 1 ? (const short*)_wtab : &BilinearTab_iC4[0][0][0];
        uchar* D = (uchar*)_dst;
        v_int32x4 delta = v_setall_s32(INTER_REMAP_COEF_SCALE / 2);
        v_int16x8 xy2ofs = v_reinterpret_as_s16(v_setall_s32(cn + (sstep << 16)));
        int CV_DECL_ALIGNED(16) iofs0[4], iofs1[4];
        const uchar* src_limit_8bytes = _src.datalimit - VTraits<v_int16x8>::vlanes();
#define CV_PICK_AND_PACK_RGB(ptr, offset, result)  \
        {                                          \
            const uchar* const p = ((const uchar*)ptr) + (offset); \
            if (p <= src_limit_8bytes)             \
            {                                      \
                v_uint8x16 rrggbb, dummy;          \
                v_uint16x8 rrggbb8, dummy8;        \
                v_uint8x16 rgb0 = v_reinterpret_as_u8(v_int32x4(*(unaligned_int*)(p), 0, 0, 0)); \
                v_uint8x16 rgb1 = v_reinterpret_as_u8(v_int32x4(*(unaligned_int*)(p + 3), 0, 0, 0)); \
                v_zip(rgb0, rgb1, rrggbb, dummy);  \
                v_expand(rrggbb, rrggbb8, dummy8); \
                result = v_reinterpret_as_s16(rrggbb8); \
            }                                      \
            else                                   \
            {                                      \
                result = v_int16x8((short)p[0], (short)p[3], /* r0r1 */ \
                                   (short)p[1], (short)p[4], /* g0g1 */ \
                                   (short)p[2], (short)p[5], /* b0b1 */ 0, 0); \
            }                                      \
        }
#define CV_PICK_AND_PACK_RGBA(ptr, offset, result) \
        {                                          \
            const uchar* const p = ((const uchar*)ptr) + (offset); \
            CV_DbgAssert(p <= src_limit_8bytes);   \
            v_uint8x16 rrggbbaa, dummy;            \
            v_uint16x8 rrggbbaa8, dummy8;          \
            v_uint8x16 rgba0 = v_reinterpret_as_u8(v_int32x4(*(unaligned_int*)(p), 0, 0, 0)); \
            v_uint8x16 rgba1 = v_reinterpret_as_u8(v_int32x4(*(unaligned_int*)(p + VTraits<v_int32x4>::vlanes()), 0, 0, 0)); \
            v_zip(rgba0, rgba1, rrggbbaa, dummy);  \
            v_expand(rrggbbaa, rrggbbaa8, dummy8); \
            result = v_reinterpret_as_s16(rrggbbaa8); \
        }
#define CV_PICK_AND_PACK4(base,offset)             \
            v_uint16x8(*(unaligned_ushort*)(base + offset[0]), *(unaligned_ushort*)(base + offset[1]), \
                       *(unaligned_ushort*)(base + offset[2]), *(unaligned_ushort*)(base + offset[3]), \
                       0, 0, 0, 0)

        const short _rel_offset_x = static_cast<short>(_offset.x);
        const short _rel_offset_y = static_cast<short>(_offset.y);
        v_int16x8 v_dxy0(_rel_offset_x, _rel_offset_y, _rel_offset_x, _rel_offset_y, _rel_offset_x, _rel_offset_y, _rel_offset_x, _rel_offset_y);
        v_int16x8 v_dxy1 = v_dxy0;
        v_dxy0 = v_add(v_dxy0, v_int16x8(0, 0, 1, 0, 2, 0, 3, 0));
        v_dxy1 = v_add(v_dxy1, v_int16x8(4, 0, 5, 0, 6, 0, 7, 0));
        if( cn == 1 )
        {
            for( ; x <= width - 8; x += 8 )
            {
                v_int16x8 _xy0 = v_load(XY + x*2);
                v_int16x8 _xy1 = v_load(XY + x*2 + 8);
                if (isRelative)
                {
                    const short x_s16 = static_cast<short>(x);
                    v_int16x8 v_dxy01(x_s16, 0, x_s16, 0, x_s16, 0, x_s16, 0);
                    _xy0 = v_add(_xy0, v_add(v_dxy01, v_dxy0));
                    _xy1 = v_add(_xy1, v_add(v_dxy01, v_dxy1));
                }
                v_int32x4 v0, v1, v2, v3, a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2;

                v_int32x4 xy0 = v_dotprod( _xy0, xy2ofs );
                v_int32x4 xy1 = v_dotprod( _xy1, xy2ofs );
                v_store( iofs0, xy0 );
                v_store( iofs1, xy1 );

                v_uint16x8 stub, dummy;
                v_uint16x8 vec16;
                vec16 = CV_PICK_AND_PACK4(S0, iofs0);
                v_expand(v_reinterpret_as_u8(vec16), stub, dummy);
                v0 = v_reinterpret_as_s32(stub);
                vec16 = CV_PICK_AND_PACK4(S1, iofs0);
                v_expand(v_reinterpret_as_u8(vec16), stub, dummy);
                v1 = v_reinterpret_as_s32(stub);

                v_zip(v_load_low((int*)(wtab + FXY[x]     * 4)), v_load_low((int*)(wtab + FXY[x + 1] * 4)), a0, a1);
                v_zip(v_load_low((int*)(wtab + FXY[x + 2] * 4)), v_load_low((int*)(wtab + FXY[x + 3] * 4)), b0, b1);
                v_recombine(a0, b0, a2, b2);
                v1 = v_dotprod(v_reinterpret_as_s16(v1), v_reinterpret_as_s16(b2), delta);
                v0 = v_dotprod(v_reinterpret_as_s16(v0), v_reinterpret_as_s16(a2), v1);

                vec16 = CV_PICK_AND_PACK4(S0, iofs1);
                v_expand(v_reinterpret_as_u8(vec16), stub, dummy);
                v2 = v_reinterpret_as_s32(stub);
                vec16 = CV_PICK_AND_PACK4(S1, iofs1);
                v_expand(v_reinterpret_as_u8(vec16), stub, dummy);
                v3 = v_reinterpret_as_s32(stub);

                v_zip(v_load_low((int*)(wtab + FXY[x + 4] * 4)), v_load_low((int*)(wtab + FXY[x + 5] * 4)), c0, c1);
                v_zip(v_load_low((int*)(wtab + FXY[x + 6] * 4)), v_load_low((int*)(wtab + FXY[x + 7] * 4)), d0, d1);
                v_recombine(c0, d0, c2, d2);
                v3 = v_dotprod(v_reinterpret_as_s16(v3), v_reinterpret_as_s16(d2), delta);
                v2 = v_dotprod(v_reinterpret_as_s16(v2), v_reinterpret_as_s16(c2), v3);

                v0 = v_shr<INTER_REMAP_COEF_BITS>(v0);
                v2 = v_shr<INTER_REMAP_COEF_BITS>(v2);
                v_pack_u_store(D + x, v_pack(v0, v2));
            }
        }
        else if( cn == 3 )
        {
            for( ; x <= width - 5; x += 4, D += 12 )
            {
                v_int16x8 u0, v0, u1, v1;
                v_int16x8 _xy0 = v_load(XY + x * 2);
                if (isRelative)
                {
                    const short x_s16 = static_cast<short>(x);
                    v_int16x8 v_dxy01(x_s16, 0, x_s16, 0, x_s16, 0, x_s16, 0);
                    _xy0 = v_add(_xy0, v_add(v_dxy01, v_dxy0));
                }

                v_int32x4 xy0 = v_dotprod(_xy0, xy2ofs);
                v_store(iofs0, xy0);

                int offset0 = FXY[x] * 16;
                int offset1 = FXY[x + 1] * 16;
                int offset2 = FXY[x + 2] * 16;
                int offset3 = FXY[x + 3] * 16;
                v_int16x8 w00 = v_load(wtab + offset0);
                v_int16x8 w01 = v_load(wtab + offset0 + 8);
                v_int16x8 w10 = v_load(wtab + offset1);
                v_int16x8 w11 = v_load(wtab + offset1 + 8);

                CV_PICK_AND_PACK_RGB(S0, iofs0[0], u0);
                CV_PICK_AND_PACK_RGB(S1, iofs0[0], v0);
                CV_PICK_AND_PACK_RGB(S0, iofs0[1], u1);
                CV_PICK_AND_PACK_RGB(S1, iofs0[1], v1);

                v_int32x4 result0 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u0, w00, v_dotprod(v0, w01, delta)));
                v_int32x4 result1 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u1, w10, v_dotprod(v1, w11, delta)));

                result0 = v_rotate_left<1>(result0);
                v_int16x8 result8 = v_pack(result0, result1);
                v_uint8x16 result16 = v_pack_u(result8, result8);
                v_store_low(D, v_rotate_right<1>(result16));


                w00 = v_load(wtab + offset2);
                w01 = v_load(wtab + offset2 + 8);
                w10 = v_load(wtab + offset3);
                w11 = v_load(wtab + offset3 + 8);
                CV_PICK_AND_PACK_RGB(S0, iofs0[2], u0);
                CV_PICK_AND_PACK_RGB(S1, iofs0[2], v0);
                CV_PICK_AND_PACK_RGB(S0, iofs0[3], u1);
                CV_PICK_AND_PACK_RGB(S1, iofs0[3], v1);

                result0 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u0, w00, v_dotprod(v0, w01, delta)));
                result1 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u1, w10, v_dotprod(v1, w11, delta)));

                result0 = v_rotate_left<1>(result0);
                result8 = v_pack(result0, result1);
                result16 = v_pack_u(result8, result8);
                v_store_low(D + 6, v_rotate_right<1>(result16));
            }
        }
        else if( cn == 4 )
        {
            for( ; x <= width - 4; x += 4, D += 16 )
            {
                v_int16x8 _xy0 = v_load(XY + x * 2);
                if (isRelative)
                {
                    const short x_s16 = static_cast<short>(x);
                    v_int16x8 v_dxy01(x_s16, 0, x_s16, 0, x_s16, 0, x_s16, 0);
                    _xy0 = v_add(_xy0, v_add(v_dxy01, v_dxy0));
                }
                v_int16x8 u0, v0, u1, v1;

                v_int32x4 xy0 = v_dotprod( _xy0, xy2ofs );
                v_store(iofs0, xy0);

                int offset0 = FXY[x] * 16;
                int offset1 = FXY[x + 1] * 16;
                int offset2 = FXY[x + 2] * 16;
                int offset3 = FXY[x + 3] * 16;

                v_int16x8 w00 = v_load(wtab + offset0);
                v_int16x8 w01 = v_load(wtab + offset0 + 8);
                v_int16x8 w10 = v_load(wtab + offset1);
                v_int16x8 w11 = v_load(wtab + offset1 + 8);
                CV_PICK_AND_PACK_RGBA(S0, iofs0[0], u0);
                CV_PICK_AND_PACK_RGBA(S1, iofs0[0], v0);
                CV_PICK_AND_PACK_RGBA(S0, iofs0[1], u1);
                CV_PICK_AND_PACK_RGBA(S1, iofs0[1], v1);

                v_int32x4 result0 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u0, w00, v_dotprod(v0, w01, delta)));
                v_int32x4 result1 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u1, w10, v_dotprod(v1, w11, delta)));
                v_int16x8 result8 = v_pack(result0, result1);
                v_pack_u_store(D, result8);

                w00 = v_load(wtab + offset2);
                w01 = v_load(wtab + offset2 + 8);
                w10 = v_load(wtab + offset3);
                w11 = v_load(wtab + offset3 + 8);
                CV_PICK_AND_PACK_RGBA(S0, iofs0[2], u0);
                CV_PICK_AND_PACK_RGBA(S1, iofs0[2], v0);
                CV_PICK_AND_PACK_RGBA(S0, iofs0[3], u1);
                CV_PICK_AND_PACK_RGBA(S1, iofs0[3], v1);

                result0 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u0, w00, v_dotprod(v0, w01, delta)));
                result1 = v_shr<INTER_REMAP_COEF_BITS>(v_dotprod(u1, w10, v_dotprod(v1, w11, delta)));
                result8 = v_pack(result0, result1);
                v_pack_u_store(D + 8, result8);
            }
        }

        return x;
    }
};

#else

template<bool isRelative> using RemapVec_8u = RemapNoVec<isRelative>;

#endif

template<class CastOp, class VecOp, typename AT, bool isRelative>
static void remapBilinear( const Mat& _src, Mat& _dst, const Mat& _xy,
                           const Mat& _fxy, const void* _wtab,
                           int borderType, const Scalar& _borderValue, const Point& _offset )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    Size ssize = _src.size(), dsize = _dst.size();
    const int cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr<T>();
    size_t sstep = _src.step/sizeof(S0[0]);
    T cval[CV_CN_MAX];
    CastOp castOp;
    VecOp vecOp;

    for(int k = 0; k < cn; k++ )
        cval[k] = saturate_cast<T>(_borderValue[k & 3]);

    unsigned width1 = std::max(ssize.width-1, 0), height1 = std::max(ssize.height-1, 0);
    CV_Assert( !ssize.empty() );
#if CV_SIMD128
    if( _src.type() == CV_8UC3 )
        width1 = std::max(ssize.width-2, 0);
#endif

    for(int dy = 0; dy < dsize.height; dy++ )
    {
        T* D = _dst.ptr<T>(dy);
        const short* XY = _xy.ptr<short>(dy);
        const ushort* FXY = _fxy.ptr<ushort>(dy);
        int X0 = 0;
        bool prevInlier = false;
        const int off_y = (isRelative ? (_offset.y+dy) : 0);
        for(int dx = 0; dx <= dsize.width; dx++ )
        {
            bool curInlier = dx < dsize.width ?
                (unsigned)XY[dx*2]+(isRelative ? (_offset.x+dx) : 0) < width1 &&
                (unsigned)XY[dx*2+1]+off_y < height1 : !prevInlier;
            if( curInlier == prevInlier )
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if( !curInlier )
            {
                Point subOffset(_offset.x+dx, _offset.y+dy);
                int len = vecOp( _src, D, XY + dx*2, FXY + dx, wtab, X1 - dx, subOffset );
                D += len*cn;
                dx += len;

                if( cn == 1 )
                {
                    for( ; dx < X1; dx++, D++ )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx;
                        *D = castOp(WT(S[0]*w[0] + S[1]*w[1] + S[sstep]*w[2] + S[sstep+1]*w[3]));
                    }
                }
                else if( cn == 2 )
                    for( ; dx < X1; dx++, D += 2 )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*2;
                        WT t0 = S[0]*w[0] + S[2]*w[1] + S[sstep]*w[2] + S[sstep+2]*w[3];
                        WT t1 = S[1]*w[0] + S[3]*w[1] + S[sstep+1]*w[2] + S[sstep+3]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1);
                    }
                else if( cn == 3 )
                    for( ; dx < X1; dx++, D += 3 )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*3;
                        WT t0 = S[0]*w[0] + S[3]*w[1] + S[sstep]*w[2] + S[sstep+3]*w[3];
                        WT t1 = S[1]*w[0] + S[4]*w[1] + S[sstep+1]*w[2] + S[sstep+4]*w[3];
                        WT t2 = S[2]*w[0] + S[5]*w[1] + S[sstep+2]*w[2] + S[sstep+5]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1); D[2] = castOp(t2);
                    }
                else if( cn == 4 )
                    for( ; dx < X1; dx++, D += 4 )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*4;
                        WT t0 = S[0]*w[0] + S[4]*w[1] + S[sstep]*w[2] + S[sstep+4]*w[3];
                        WT t1 = S[1]*w[0] + S[5]*w[1] + S[sstep+1]*w[2] + S[sstep+5]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1);
                        t0 = S[2]*w[0] + S[6]*w[1] + S[sstep+2]*w[2] + S[sstep+6]*w[3];
                        t1 = S[3]*w[0] + S[7]*w[1] + S[sstep+3]*w[2] + S[sstep+7]*w[3];
                        D[2] = castOp(t0); D[3] = castOp(t1);
                    }
                else
                    for( ; dx < X1; dx++, D += cn )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*cn;
                        for(int k = 0; k < cn; k++ )
                        {
                            WT t0 = S[k]*w[0] + S[k+cn]*w[1] + S[sstep+k]*w[2] + S[sstep+k+cn]*w[3];
                            D[k] = castOp(t0);
                        }
                    }
            }
            else
            {
                if (borderType == BORDER_TRANSPARENT) {
                    for (; dx < X1; dx++, D += cn) {
                        if (dx >= dsize.width) continue;
                        const int sx = XY[dx * 2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx * 2 + 1]+off_y;
                        // If the mapped point is still within bounds, it did not get computed
                        // because it lacked 4 neighbors. Still, it can be computed with an
                        // approximate formula. If it is outside, the point is left untouched.
                        if (sx >= 0 && sx <= ssize.width - 1 && sy >= 0 && sy <= ssize.height - 1) {
                            const AT* w = wtab + FXY[dx] * 4;
                            WT w_tot = 0;
                            if (sx >= 0 && sy >= 0) w_tot += w[0];
                            if (sy >= 0 && sx < ssize.width - 1) w_tot += w[1];
                            if (sx >= 0 && sy < ssize.height - 1) w_tot += w[2];
                            if (sx < ssize.width - 1 && sy < ssize.height - 1) w_tot += w[3];
                            if (w_tot == 0.f) continue;
                            const WT w_tot_ini = (WT)w[0] + w[1] + w[2] + w[3];
                            const T* S = S0 + sy * sstep + sx * cn;
                            for (int k = 0; k < cn; k++) {
                                WT t0 = 0;
                                if (sx >= 0 && sy >= 0) t0 += S[k] * w[0];
                                if (sy >= 0 && sx < ssize.width - 1) t0 += S[k + cn] * w[1];
                                if (sx >= 0 && sy < ssize.height - 1) t0 += S[sstep + k] * w[2];
                                if (sx < ssize.width - 1 && sy < ssize.height - 1) t0 += S[sstep + k + cn] * w[3];
                                t0 = (WT)(t0 * (float)w_tot_ini / w_tot);
                                D[k] = castOp(t0);
                            }
                        }
                    }
                    continue;
                }

                if( cn == 1 )
                    for( ; dx < X1; dx++, D++ )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        if( borderType == BORDER_CONSTANT &&
                            (sx >= ssize.width || sx+1 < 0 ||
                             sy >= ssize.height || sy+1 < 0) )
                        {
                            D[0] = cval[0];
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            T v0, v1, v2, v3;
                            const AT* w = wtab + FXY[dx]*4;
                            if( borderType == BORDER_REPLICATE )
                            {
                                sx0 = clip(sx, 0, ssize.width);
                                sx1 = clip(sx+1, 0, ssize.width);
                                sy0 = clip(sy, 0, ssize.height);
                                sy1 = clip(sy+1, 0, ssize.height);
                                v0 = S0[sy0*sstep + sx0];
                                v1 = S0[sy0*sstep + sx1];
                                v2 = S0[sy1*sstep + sx0];
                                v3 = S0[sy1*sstep + sx1];
                            }
                            else
                            {
                                sx0 = borderInterpolate(sx, ssize.width, borderType);
                                sx1 = borderInterpolate(sx+1, ssize.width, borderType);
                                sy0 = borderInterpolate(sy, ssize.height, borderType);
                                sy1 = borderInterpolate(sy+1, ssize.height, borderType);
                                v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx0] : cval[0];
                                v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx1] : cval[0];
                                v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx0] : cval[0];
                                v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx1] : cval[0];
                            }
                            D[0] = castOp(WT(v0*w[0] + v1*w[1] + v2*w[2] + v3*w[3]));
                        }
                    }
                else
                    for( ; dx < X1; dx++, D += cn )
                    {
                        int sx = XY[dx*2]+(isRelative ? (_offset.x+dx) : 0), sy = XY[dx*2+1]+off_y;
                        if( borderType == BORDER_CONSTANT &&
                            (sx >= ssize.width || sx+1 < 0 ||
                             sy >= ssize.height || sy+1 < 0) )
                        {
                            for(int k = 0; k < cn; k++ )
                                D[k] = cval[k];
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            const T *v0, *v1, *v2, *v3;
                            const AT* w = wtab + FXY[dx]*4;
                            if( borderType == BORDER_REPLICATE )
                            {
                                sx0 = clip(sx, 0, ssize.width);
                                sx1 = clip(sx+1, 0, ssize.width);
                                sy0 = clip(sy, 0, ssize.height);
                                sy1 = clip(sy+1, 0, ssize.height);
                                v0 = S0 + sy0*sstep + sx0*cn;
                                v1 = S0 + sy0*sstep + sx1*cn;
                                v2 = S0 + sy1*sstep + sx0*cn;
                                v3 = S0 + sy1*sstep + sx1*cn;
                            }
                            else
                            {
                                sx0 = borderInterpolate(sx, ssize.width, borderType);
                                sx1 = borderInterpolate(sx+1, ssize.width, borderType);
                                sy0 = borderInterpolate(sy, ssize.height, borderType);
                                sy1 = borderInterpolate(sy+1, ssize.height, borderType);
                                v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx0*cn : &cval[0];
                                v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx1*cn : &cval[0];
                                v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx0*cn : &cval[0];
                                v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx1*cn : &cval[0];
                            }
                            for(int k = 0; k < cn; k++ )
                                D[k] = castOp(WT(v0[k]*w[0] + v1[k]*w[1] + v2[k]*w[2] + v3[k]*w[3]));
                        }
                    }
            }
        }
    }
}


template<class CastOp, typename AT, int ONE, bool isRelative>
static void remapBicubic( const Mat& _src, Mat& _dst, const Mat& _xy,
                          const Mat& _fxy, const void* _wtab,
                          int borderType, const Scalar& _borderValue, const Point& _offset )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    Size ssize = _src.size(), dsize = _dst.size();
    const int cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr<T>();
    size_t sstep = _src.step/sizeof(S0[0]);
    T cval[CV_CN_MAX];
    CastOp castOp;

    for(int k = 0; k < cn; k++ )
        cval[k] = saturate_cast<T>(_borderValue[k & 3]);

    int borderType1 = borderType != BORDER_TRANSPARENT ? borderType : BORDER_REFLECT_101;

    unsigned width1 = std::max(ssize.width-3, 0), height1 = std::max(ssize.height-3, 0);

    if( _dst.isContinuous() && _xy.isContinuous() && _fxy.isContinuous() && !isRelative )
    {
        dsize.width *= dsize.height;
        dsize.height = 1;
    }

    for(int dy = 0; dy < dsize.height; dy++ )
    {
        T* D = _dst.ptr<T>(dy);
        const short* XY = _xy.ptr<short>(dy);
        const ushort* FXY = _fxy.ptr<ushort>(dy);
        const int off_y = isRelative ? (_offset.y+dy) : 0;
        for(int dx = 0; dx < dsize.width; dx++, D += cn )
        {
            const int off_x = isRelative ? (_offset.x+dx) : 0;
            int sx = XY[dx*2]-1+off_x, sy = XY[dx*2+1]-1+off_y;
            const AT* w = wtab + FXY[dx]*16;
            if( (unsigned)sx < width1 && (unsigned)sy < height1 )
            {
                const T* S = S0 + sy*sstep + sx*cn;
                for(int k = 0; k < cn; k++ )
                {
                    WT sum = S[0]*w[0] + S[cn]*w[1] + S[cn*2]*w[2] + S[cn*3]*w[3];
                    S += sstep;
                    sum += S[0]*w[4] + S[cn]*w[5] + S[cn*2]*w[6] + S[cn*3]*w[7];
                    S += sstep;
                    sum += S[0]*w[8] + S[cn]*w[9] + S[cn*2]*w[10] + S[cn*3]*w[11];
                    S += sstep;
                    sum += S[0]*w[12] + S[cn]*w[13] + S[cn*2]*w[14] + S[cn*3]*w[15];
                    S += 1 - sstep*3;
                    D[k] = castOp(sum);
                }
            }
            else
            {
                int x[4], y[4];
                if( borderType == BORDER_TRANSPARENT &&
                    ((unsigned)(sx+1) >= (unsigned)ssize.width ||
                    (unsigned)(sy+1) >= (unsigned)ssize.height) )
                    continue;

                if( borderType1 == BORDER_CONSTANT &&
                    (sx >= ssize.width || sx+4 <= 0 ||
                    sy >= ssize.height || sy+4 <= 0))
                {
                    for(int k = 0; k < cn; k++ )
                        D[k] = cval[k];
                    continue;
                }

                for(int i = 0; i < 4; i++ )
                {
                    x[i] = borderInterpolate(sx + i, ssize.width, borderType1)*cn;
                    y[i] = borderInterpolate(sy + i, ssize.height, borderType1);
                }

                for(int k = 0; k < cn; k++, S0++, w -= 16 )
                {
                    WT cv = cval[k], sum = cv*ONE;
                    for(int i = 0; i < 4; i++, w += 4 )
                    {
                        int yi = y[i];
                        const T* S = S0 + yi*sstep;
                        if( yi < 0 )
                            continue;
                        if( x[0] >= 0 )
                            sum += (S[x[0]] - cv)*w[0];
                        if( x[1] >= 0 )
                            sum += (S[x[1]] - cv)*w[1];
                        if( x[2] >= 0 )
                            sum += (S[x[2]] - cv)*w[2];
                        if( x[3] >= 0 )
                            sum += (S[x[3]] - cv)*w[3];
                    }
                    D[k] = castOp(sum);
                }
                S0 -= cn;
            }
        }
    }
}


template<class CastOp, typename AT, int ONE, bool isRelative>
static void remapLanczos4( const Mat& _src, Mat& _dst, const Mat& _xy,
                           const Mat& _fxy, const void* _wtab,
                           int borderType, const Scalar& _borderValue, const Point& _offset )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    Size ssize = _src.size(), dsize = _dst.size();
    const int cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr<T>();
    size_t sstep = _src.step/sizeof(S0[0]);
    T cval[CV_CN_MAX];
    CastOp castOp;

    for(int k = 0; k < cn; k++ )
        cval[k] = saturate_cast<T>(_borderValue[k & 3]);

    int borderType1 = borderType != BORDER_TRANSPARENT ? borderType : BORDER_REFLECT_101;

    unsigned width1 = std::max(ssize.width-7, 0), height1 = std::max(ssize.height-7, 0);

    if( _dst.isContinuous() && _xy.isContinuous() && _fxy.isContinuous() && !isRelative )
    {
        dsize.width *= dsize.height;
        dsize.height = 1;
    }

    for(int dy = 0; dy < dsize.height; dy++ )
    {
        T* D = _dst.ptr<T>(dy);
        const short* XY = _xy.ptr<short>(dy);
        const ushort* FXY = _fxy.ptr<ushort>(dy);
        const int off_y = isRelative ? (_offset.y+dy) : 0;
        for(int dx = 0; dx < dsize.width; dx++, D += cn )
        {
            const int off_x = isRelative ? (_offset.x+dx) : 0;
            int sx = XY[dx*2]-3+off_x, sy = XY[dx*2+1]-3+off_y;
            const AT* w = wtab + FXY[dx]*64;
            const T* S = S0 + sy*sstep + sx*cn;
            if( (unsigned)sx < width1 && (unsigned)sy < height1 )
            {
                for(int k = 0; k < cn; k++ )
                {
                    WT sum = 0;
                    for( int r = 0; r < 8; r++, S += sstep, w += 8 )
                        sum += S[0]*w[0] + S[cn]*w[1] + S[cn*2]*w[2] + S[cn*3]*w[3] +
                            S[cn*4]*w[4] + S[cn*5]*w[5] + S[cn*6]*w[6] + S[cn*7]*w[7];
                    w -= 64;
                    S -= sstep*8 - 1;
                    D[k] = castOp(sum);
                }
            }
            else
            {
                int x[8], y[8];
                if( borderType == BORDER_TRANSPARENT &&
                    ((unsigned)(sx+3) >= (unsigned)ssize.width ||
                    (unsigned)(sy+3) >= (unsigned)ssize.height) )
                    continue;

                if( borderType1 == BORDER_CONSTANT &&
                    (sx >= ssize.width || sx+8 <= 0 ||
                    sy >= ssize.height || sy+8 <= 0))
                {
                    for(int k = 0; k < cn; k++ )
                        D[k] = cval[k];
                    continue;
                }

                for(int i = 0; i < 8; i++ )
                {
                    x[i] = borderInterpolate(sx + i, ssize.width, borderType1)*cn;
                    y[i] = borderInterpolate(sy + i, ssize.height, borderType1);
                }

                for(int k = 0; k < cn; k++, S0++, w -= 64 )
                {
                    WT cv = cval[k], sum = cv*ONE;
                    for(int i = 0; i < 8; i++, w += 8 )
                    {
                        int yi = y[i];
                        const T* S1 = S0 + yi*sstep;
                        if( yi < 0 )
                            continue;
                        if( x[0] >= 0 )
                            sum += (S1[x[0]] - cv)*w[0];
                        if( x[1] >= 0 )
                            sum += (S1[x[1]] - cv)*w[1];
                        if( x[2] >= 0 )
                            sum += (S1[x[2]] - cv)*w[2];
                        if( x[3] >= 0 )
                            sum += (S1[x[3]] - cv)*w[3];
                        if( x[4] >= 0 )
                            sum += (S1[x[4]] - cv)*w[4];
                        if( x[5] >= 0 )
                            sum += (S1[x[5]] - cv)*w[5];
                        if( x[6] >= 0 )
                            sum += (S1[x[6]] - cv)*w[6];
                        if( x[7] >= 0 )
                            sum += (S1[x[7]] - cv)*w[7];
                    }
                    D[k] = castOp(sum);
                }
                S0 -= cn;
            }
        }
    }
}


typedef void (*RemapNNFunc)(const Mat& _src, Mat& _dst, const Mat& _xy,
                            int borderType, const Scalar& _borderValue, const Point& _offset);

typedef void (*RemapFunc)(const Mat& _src, Mat& _dst, const Mat& _xy,
                          const Mat& _fxy, const void* _wtab,
                          int borderType, const Scalar& _borderValue, const Point& _offset);

class RemapInvoker :
    public ParallelLoopBody
{
public:
    RemapInvoker(const Mat& _src, Mat& _dst, const Mat *_m1,
                 const Mat *_m2, int _borderType, const Scalar &_borderValue,
                 int _planar_input, RemapNNFunc _nnfunc, RemapFunc _ifunc, const void *_ctab) :
        ParallelLoopBody(), src(&_src), dst(&_dst), m1(_m1), m2(_m2),
        borderType(_borderType), borderValue(_borderValue),
        planar_input(_planar_input), nnfunc(_nnfunc), ifunc(_ifunc), ctab(_ctab)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int x, y, x1, y1;
        const int buf_size = 1 << 14;
        int brows0 = std::min(128, dst->rows), map_depth = m1->depth();
        int bcols0 = std::min(buf_size/brows0, dst->cols);
        brows0 = std::min(buf_size/bcols0, dst->rows);

        Mat _bufxy(brows0, bcols0, CV_16SC2), _bufa;
        if( !nnfunc )
            _bufa.create(brows0, bcols0, CV_16UC1);

        for( y = range.start; y < range.end; y += brows0 )
        {
            for( x = 0; x < dst->cols; x += bcols0 )
            {
                int brows = std::min(brows0, range.end - y);
                int bcols = std::min(bcols0, dst->cols - x);
                Mat dpart(*dst, Rect(x, y, bcols, brows));
                Mat bufxy(_bufxy, Rect(0, 0, bcols, brows));

                if( nnfunc )
                {
                    if( m1->type() == CV_16SC2 && m2->empty() ) // the data is already in the right format
                        bufxy = (*m1)(Rect(x, y, bcols, brows));
                    else if( map_depth != CV_32F )
                    {
                        for( y1 = 0; y1 < brows; y1++ )
                        {
                            short* XY = bufxy.ptr<short>(y1);
                            const short* sXY = m1->ptr<short>(y+y1) + x*2;
                            const ushort* sA = m2->ptr<ushort>(y+y1) + x;

                            for( x1 = 0; x1 < bcols; x1++ )
                            {
                                int a = sA[x1] & (INTER_TAB_SIZE2-1);
                                XY[x1*2] = sXY[x1*2] + NNDeltaTab_i[a][0];
                                XY[x1*2+1] = sXY[x1*2+1] + NNDeltaTab_i[a][1];
                            }
                        }
                    }
                    else if( !planar_input )
                        (*m1)(Rect(x, y, bcols, brows)).convertTo(bufxy, bufxy.depth());
                    else
                    {
                        for( y1 = 0; y1 < brows; y1++ )
                        {
                            short* XY = bufxy.ptr<short>(y1);
                            const float* sX = m1->ptr<float>(y+y1) + x;
                            const float* sY = m2->ptr<float>(y+y1) + x;
                            x1 = 0;

                            #if CV_SIMD128
                            {
                                int span = VTraits<v_float32x4>::vlanes();
                                for( ; x1 <= bcols - span * 2; x1 += span * 2 )
                                {
                                    v_int32x4 ix0 = v_round(v_load(sX + x1));
                                    v_int32x4 iy0 = v_round(v_load(sY + x1));
                                    v_int32x4 ix1 = v_round(v_load(sX + x1 + span));
                                    v_int32x4 iy1 = v_round(v_load(sY + x1 + span));

                                    v_int16x8 dx, dy;
                                    dx = v_pack(ix0, ix1);
                                    dy = v_pack(iy0, iy1);
                                    v_store_interleave(XY + x1 * 2, dx, dy);
                                }
                            }
                            #endif
                            for( ; x1 < bcols; x1++ )
                            {
                                XY[x1*2] = saturate_cast<short>(sX[x1]);
                                XY[x1*2+1] = saturate_cast<short>(sY[x1]);
                            }
                        }
                    }
                    nnfunc( *src, dpart, bufxy, borderType, borderValue, Point(x, y) );
                    continue;
                }

                Mat bufa(_bufa, Rect(0, 0, bcols, brows));
                for( y1 = 0; y1 < brows; y1++ )
                {
                    short* XY = bufxy.ptr<short>(y1);
                    ushort* A = bufa.ptr<ushort>(y1);

                    if( m1->type() == CV_16SC2 && (m2->type() == CV_16UC1 || m2->type() == CV_16SC1) )
                    {
                        bufxy = (*m1)(Rect(x, y, bcols, brows));

                        const ushort* sA = m2->ptr<ushort>(y+y1) + x;
                        x1 = 0;

                        #if CV_SIMD128
                        {
                            v_uint16x8 v_scale = v_setall_u16(INTER_TAB_SIZE2 - 1);
                            int span = VTraits<v_uint16x8>::vlanes();
                            for( ; x1 <= bcols - span; x1 += span )
                                v_store((unsigned short*)(A + x1), v_and(v_load(sA + x1), v_scale));
                        }
                        #endif
                        for( ; x1 < bcols; x1++ )
                            A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2-1));
                    }
                    else if( planar_input )
                    {
                        const float* sX = m1->ptr<float>(y+y1) + x;
                        const float* sY = m2->ptr<float>(y+y1) + x;

                        x1 = 0;
                        #if CV_SIMD128
                        {
                            v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
                            v_int32x4 v_scale2 = v_setall_s32(INTER_TAB_SIZE - 1);
                            int span = VTraits<v_float32x4>::vlanes();
                            for( ; x1 <= bcols - span * 2; x1 += span * 2 )
                            {
                                v_int32x4 v_sx0 = v_round(v_mul(v_scale, v_load(sX + x1)));
                                v_int32x4 v_sy0 = v_round(v_mul(v_scale, v_load(sY + x1)));
                                v_int32x4 v_sx1 = v_round(v_mul(v_scale, v_load(sX + x1 + span)));
                                v_int32x4 v_sy1 = v_round(v_mul(v_scale, v_load(sY + x1 + span)));
                                v_uint16x8 v_sx8 = v_reinterpret_as_u16(v_pack(v_and(v_sx0, v_scale2), v_and(v_sx1, v_scale2)));
                                v_uint16x8 v_sy8 = v_reinterpret_as_u16(v_pack(v_and(v_sy0, v_scale2), v_and(v_sy1, v_scale2)));
                                v_uint16x8 v_v = v_or(v_shl<INTER_BITS>(v_sy8), v_sx8);
                                v_store(A + x1, v_v);

                                v_int16x8 v_d0 = v_pack(v_shr<INTER_BITS>(v_sx0), v_shr<INTER_BITS>(v_sx1));
                                v_int16x8 v_d1 = v_pack(v_shr<INTER_BITS>(v_sy0), v_shr<INTER_BITS>(v_sy1));
                                v_store_interleave(XY + (x1 << 1), v_d0, v_d1);
                            }
                        }
                        #endif
                        for( ; x1 < bcols; x1++ )
                        {
                            int sx = cvRound(sX[x1]*static_cast<float>(INTER_TAB_SIZE));
                            int sy = cvRound(sY[x1]*static_cast<float>(INTER_TAB_SIZE));
                            int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
                            XY[x1*2] = saturate_cast<short>(sx >> INTER_BITS);
                            XY[x1*2+1] = saturate_cast<short>(sy >> INTER_BITS);
                            A[x1] = (ushort)v;
                        }
                    }
                    else
                    {
                        const float* sXY = m1->ptr<float>(y+y1) + x*2;
                        x1 = 0;

                        #if CV_SIMD128
                        {
                            v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
                            v_int32x4 v_scale2 = v_setall_s32(INTER_TAB_SIZE - 1), v_scale3 = v_setall_s32(INTER_TAB_SIZE);
                            int span = VTraits<v_float32x4>::vlanes();
                            for( ; x1 <= bcols - span * 2; x1 += span * 2 )
                            {
                                v_float32x4 v_fx, v_fy;
                                v_load_deinterleave(sXY + (x1 << 1), v_fx, v_fy);
                                v_int32x4 v_sx0 = v_round(v_mul(v_fx, v_scale));
                                v_int32x4 v_sy0 = v_round(v_mul(v_fy, v_scale));
                                v_load_deinterleave(sXY + ((x1 + span) << 1), v_fx, v_fy);
                                v_int32x4 v_sx1 = v_round(v_mul(v_fx, v_scale));
                                v_int32x4 v_sy1 = v_round(v_mul(v_fy, v_scale));
                                v_int32x4 v_v0 = v_muladd(v_scale3, (v_and(v_sy0, v_scale2)), (v_and(v_sx0, v_scale2)));
                                v_int32x4 v_v1 = v_muladd(v_scale3, (v_and(v_sy1, v_scale2)), (v_and(v_sx1, v_scale2)));
                                v_uint16x8 v_v8 = v_reinterpret_as_u16(v_pack(v_v0, v_v1));
                                v_store(A + x1, v_v8);
                                v_int16x8 v_dx = v_pack(v_shr<INTER_BITS>(v_sx0), v_shr<INTER_BITS>(v_sx1));
                                v_int16x8 v_dy = v_pack(v_shr<INTER_BITS>(v_sy0), v_shr<INTER_BITS>(v_sy1));
                                v_store_interleave(XY + (x1 << 1), v_dx, v_dy);
                            }
                        }
                        #endif

                        for( ; x1 < bcols; x1++ )
                        {
                            int sx = cvRound(sXY[x1*2]*static_cast<float>(INTER_TAB_SIZE));
                            int sy = cvRound(sXY[x1*2+1]*static_cast<float>(INTER_TAB_SIZE));
                            int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
                            XY[x1*2] = saturate_cast<short>(sx >> INTER_BITS);
                            XY[x1*2+1] = saturate_cast<short>(sy >> INTER_BITS);
                            A[x1] = (ushort)v;
                        }
                    }
                }
                ifunc(*src, dpart, bufxy, bufa, ctab, borderType, borderValue, Point(x, y));
            }
        }
    }

private:
    const Mat* src;
    Mat* dst;
    const Mat *m1, *m2;
    int borderType;
    Scalar borderValue;
    int planar_input;
    RemapNNFunc nnfunc;
    RemapFunc ifunc;
    const void *ctab;
};

#ifdef HAVE_OPENCL

static bool ocl_remap(InputArray _src, OutputArray _dst, InputArray _map1, InputArray _map2,
                      int interpolation, int borderType, const Scalar& borderValue)
{
    const bool hasRelativeFlag = ((interpolation & cv::WARP_RELATIVE_MAP) != 0);
    interpolation &= ~cv::WARP_RELATIVE_MAP;

    const ocl::Device & dev = ocl::Device::getDefault();
    int cn = _src.channels(), type = _src.type(), depth = _src.depth(),
            rowsPerWI = dev.isIntel() ? 4 : 1;

    if(!dev.hasFP64() && depth == CV_64F)
        return false;

    if (borderType == BORDER_TRANSPARENT || !(interpolation == INTER_LINEAR || interpolation == INTER_NEAREST)
            || _map1.type() == CV_16SC1 || _map2.type() == CV_16SC1)
        return false;

    UMat src = _src.getUMat(), map1 = _map1.getUMat(), map2 = _map2.getUMat();

    if( (map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.empty())) ||
        (map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.empty())) )
    {
        if (map1.type() != CV_16SC2)
            std::swap(map1, map2);
    }
    else
        CV_Assert( map1.type() == CV_32FC2 || (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) );

    _dst.create(map1.size(), type);
    UMat dst = _dst.getUMat();

    String kernelName = "remap";
    if (map1.type() == CV_32FC2 && map2.empty())
        kernelName += "_32FC2";
    else if (map1.type() == CV_16SC2)
    {
        kernelName += "_16SC2";
        if (!map2.empty())
            kernelName += "_16UC1";
    }
    else if (map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
        kernelName += "_2_32FC1";
    else
        CV_Error(Error::StsBadArg, "Unsupported map types");

    static const char * const interMap[] = { "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LINEAR", "INTER_LANCZOS" };
    static const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP",
                           "BORDER_REFLECT_101", "BORDER_TRANSPARENT" };
    String buildOptions = format("-D %s -D %s -D T=%s -D ROWS_PER_WI=%d -D WARP_RELATIVE=%d",
                                 interMap[interpolation], borderMap[borderType],
                                 ocl::typeToStr(type), rowsPerWI,
                                 hasRelativeFlag ? 1 : 0);

    if (interpolation != INTER_NEAREST)
    {
        char cvt[3][50];
        int wdepth = std::max(CV_32F, depth);
        buildOptions = buildOptions
                      + format(" -D WT=%s -D CONVERT_TO_T=%s -D CONVERT_TO_WT=%s"
                               " -D CONVERT_TO_WT2=%s -D WT2=%s",
                               ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)),
                               ocl::convertTypeStr(wdepth, depth, cn, cvt[0], sizeof(cvt[0])),
                               ocl::convertTypeStr(depth, wdepth, cn, cvt[1], sizeof(cvt[1])),
                               ocl::convertTypeStr(CV_32S, wdepth, 2, cvt[2], sizeof(cvt[2])),
                               ocl::typeToStr(CV_MAKE_TYPE(wdepth, 2)));
    }
    int scalarcn = cn == 3 ? 4 : cn;
    int sctype = CV_MAKETYPE(depth, scalarcn);
    buildOptions += format(" -D T=%s -D T1=%s -D CN=%d -D ST=%s -D SRC_DEPTH=%d",
                           ocl::typeToStr(type), ocl::typeToStr(depth),
                           cn, ocl::typeToStr(sctype), depth);

    ocl::Kernel k(kernelName.c_str(), ocl::imgproc::remap_oclsrc, buildOptions);

    Mat scalar(1, 1, sctype, borderValue);
    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src), dstarg = ocl::KernelArg::WriteOnly(dst),
            map1arg = ocl::KernelArg::ReadOnlyNoSize(map1),
            scalararg = ocl::KernelArg::Constant((void*)scalar.ptr(), scalar.elemSize());

    if (map2.empty())
        k.args(srcarg, dstarg, map1arg, scalararg);
    else
        k.args(srcarg, dstarg, map1arg, ocl::KernelArg::ReadOnlyNoSize(map2), scalararg);

    size_t globalThreads[2] = { (size_t)dst.cols, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalThreads, NULL, false);
}

#if 0
/**
@deprecated with old version of cv::linearPolar
*/
static bool ocl_linearPolar(InputArray _src, OutputArray _dst,
    Point2f center, double maxRadius, int flags)
{
    UMat src_with_border; // don't scope this variable (it holds image data)

    UMat mapx, mapy, r, cp_sp;
    UMat src = _src.getUMat();
    _dst.create(src.size(), src.type());
    Size dsize = src.size();
    r.create(Size(1, dsize.width), CV_32F);
    cp_sp.create(Size(1, dsize.height), CV_32FC2);

    mapx.create(dsize, CV_32F);
    mapy.create(dsize, CV_32F);
    size_t w = dsize.width;
    size_t h = dsize.height;
    String buildOptions;
    unsigned mem_size = 32;
    if (flags & cv::WARP_INVERSE_MAP)
    {
        buildOptions = "-D InverseMap";
    }
    else
    {
        buildOptions = format("-D ForwardMap  -D MEM_SIZE=%d", mem_size);
    }
    String retval;
    ocl::Program p(ocl::imgproc::linearPolar_oclsrc, buildOptions, retval);
    ocl::Kernel k("linearPolar", p);
    ocl::KernelArg ocl_mapx = ocl::KernelArg::PtrReadWrite(mapx), ocl_mapy = ocl::KernelArg::PtrReadWrite(mapy);
    ocl::KernelArg  ocl_cp_sp = ocl::KernelArg::PtrReadWrite(cp_sp);
    ocl::KernelArg ocl_r = ocl::KernelArg::PtrReadWrite(r);

    if (!(flags & cv::WARP_INVERSE_MAP))
    {



        ocl::Kernel computeAngleRadius_Kernel("computeAngleRadius", p);
        float PI2_height = (float) CV_2PI / dsize.height;
        float maxRadius_width = (float) maxRadius / dsize.width;
        computeAngleRadius_Kernel.args(ocl_cp_sp, ocl_r, maxRadius_width, PI2_height, (unsigned)dsize.width, (unsigned)dsize.height);
        size_t max_dim = max(h, w);
        computeAngleRadius_Kernel.run(1, &max_dim, NULL, false);
        k.args(ocl_mapx, ocl_mapy, ocl_cp_sp, ocl_r, center.x, center.y, (unsigned)dsize.width, (unsigned)dsize.height);
    }
    else
    {
        const int ANGLE_BORDER = 1;

        cv::copyMakeBorder(src, src_with_border, ANGLE_BORDER, ANGLE_BORDER, 0, 0, BORDER_WRAP);
        src = src_with_border;
        Size ssize = src_with_border.size();
        ssize.height -= 2 * ANGLE_BORDER;
        float ascale =  ssize.height / ((float)CV_2PI);
        float pscale =  ssize.width / ((float) maxRadius);

        k.args(ocl_mapx, ocl_mapy, ascale, pscale, center.x, center.y, ANGLE_BORDER, (unsigned)dsize.width, (unsigned)dsize.height);


    }
    size_t globalThreads[2] = { (size_t)dsize.width , (size_t)dsize.height };
    size_t localThreads[2] = { mem_size , mem_size };
    k.run(2, globalThreads, localThreads, false);
    remap(src, _dst, mapx, mapy, flags & cv::INTER_MAX, (flags & cv::WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT);
    return true;
}
static bool ocl_logPolar(InputArray _src, OutputArray _dst,
    Point2f center, double M, int flags)
{
    if (M <= 0)
        CV_Error(cv::Error::StsOutOfRange, "M should be >0");
    UMat src_with_border; // don't scope this variable (it holds image data)

    UMat mapx, mapy, r, cp_sp;
    UMat src = _src.getUMat();
    _dst.create(src.size(), src.type());
    Size dsize = src.size();
    r.create(Size(1, dsize.width), CV_32F);
    cp_sp.create(Size(1, dsize.height), CV_32FC2);

    mapx.create(dsize, CV_32F);
    mapy.create(dsize, CV_32F);
    size_t w = dsize.width;
    size_t h = dsize.height;
    String buildOptions;
    unsigned mem_size = 32;
    if (flags & cv::WARP_INVERSE_MAP)
    {
        buildOptions = "-D InverseMap";
    }
    else
    {
        buildOptions = format("-D ForwardMap  -D MEM_SIZE=%d", mem_size);
    }
    String retval;
    ocl::Program p(ocl::imgproc::logPolar_oclsrc, buildOptions, retval);
    //ocl::Program p(ocl::imgproc::my_linearPolar_oclsrc, buildOptions, retval);
    //printf("%s\n", retval);
    ocl::Kernel k("logPolar", p);
    ocl::KernelArg ocl_mapx = ocl::KernelArg::PtrReadWrite(mapx), ocl_mapy = ocl::KernelArg::PtrReadWrite(mapy);
    ocl::KernelArg  ocl_cp_sp = ocl::KernelArg::PtrReadWrite(cp_sp);
    ocl::KernelArg ocl_r = ocl::KernelArg::PtrReadWrite(r);

    if (!(flags & cv::WARP_INVERSE_MAP))
    {



        ocl::Kernel computeAngleRadius_Kernel("computeAngleRadius", p);
        float PI2_height = (float) CV_2PI / dsize.height;

        computeAngleRadius_Kernel.args(ocl_cp_sp, ocl_r, (float)M, PI2_height, (unsigned)dsize.width, (unsigned)dsize.height);
        size_t max_dim = max(h, w);
        computeAngleRadius_Kernel.run(1, &max_dim, NULL, false);
        k.args(ocl_mapx, ocl_mapy, ocl_cp_sp, ocl_r, center.x, center.y, (unsigned)dsize.width, (unsigned)dsize.height);
    }
    else
    {
        const int ANGLE_BORDER = 1;

        cv::copyMakeBorder(src, src_with_border, ANGLE_BORDER, ANGLE_BORDER, 0, 0, BORDER_WRAP);
        src = src_with_border;
        Size ssize = src_with_border.size();
        ssize.height -= 2 * ANGLE_BORDER;
        float ascale =  ssize.height / ((float)CV_2PI);


        k.args(ocl_mapx, ocl_mapy, ascale, (float)M, center.x, center.y, ANGLE_BORDER, (unsigned)dsize.width, (unsigned)dsize.height);


    }
    size_t globalThreads[2] = { (size_t)dsize.width , (size_t)dsize.height };
    size_t localThreads[2] = { mem_size , mem_size };
    k.run(2, globalThreads, localThreads, false);
    remap(src, _dst, mapx, mapy, flags & cv::INTER_MAX, (flags & cv::WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT);
    return true;
}
#endif

#endif

#if defined HAVE_IPP && !IPP_DISABLE_REMAP

typedef IppStatus (CV_STDCALL * ippiRemap)(const void * pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi,
                                           const Ipp32f* pxMap, int xMapStep, const Ipp32f* pyMap, int yMapStep,
                                           void * pDst, int dstStep, IppiSize dstRoiSize, int interpolation);

class IPPRemapInvoker :
        public ParallelLoopBody
{
public:
    IPPRemapInvoker(Mat & _src, Mat & _dst, Mat & _xmap, Mat & _ymap, ippiRemap _ippFunc,
                    int _ippInterpolation, int _borderType, const Scalar & _borderValue, bool * _ok) :
        ParallelLoopBody(), src(_src), dst(_dst), map1(_xmap), map2(_ymap), ippFunc(_ippFunc),
        ippInterpolation(_ippInterpolation), borderType(_borderType), borderValue(_borderValue), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range & range) const
    {
        IppiRect srcRoiRect = { 0, 0, src.cols, src.rows };
        Mat dstRoi = dst.rowRange(range);
        IppiSize dstRoiSize = ippiSize(dstRoi.size());
        int type = dst.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

        if (borderType == BORDER_CONSTANT &&
                !IPPSet(borderValue, dstRoi.ptr(), (int)dstRoi.step, dstRoiSize, cn, depth))
        {
            *ok = false;
            return;
        }

        if (CV_INSTRUMENT_FUN_IPP(ippFunc, src.ptr(), ippiSize(src.size()), (int)src.step, srcRoiRect,
                    map1.ptr<Ipp32f>(), (int)map1.step, map2.ptr<Ipp32f>(), (int)map2.step,
                    dstRoi.ptr(), (int)dstRoi.step, dstRoiSize, ippInterpolation) < 0)
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        }
    }

private:
    Mat & src, & dst, & map1, & map2;
    ippiRemap ippFunc;
    int ippInterpolation, borderType;
    Scalar borderValue;
    bool * ok;
};

#endif

}

void cv::remap( InputArray _src, OutputArray _dst,
                InputArray _map1, InputArray _map2,
                int interpolation, int borderType, const Scalar& borderValue,
                AlgorithmHint hint )
{
    CV_INSTRUMENT_REGION();

    if (hint == cv::ALGO_HINT_DEFAULT)
        hint = cv::getDefaultAlgorithmHint();

    CV_Assert( !_map1.empty() );
    CV_Assert( _map2.empty() || (_map2.size() == _map1.size()));

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_remap(_src, _dst, _map1, _map2, interpolation, borderType, borderValue))

    Mat src = _src.getMat(), map1 = _map1.getMat(), map2 = _map2.getMat();
    _dst.create( map1.size(), src.type() );
    Mat dst = _dst.getMat();


    CV_Assert( dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX );

    if( dst.data == src.data )
        src = src.clone();

    if ((map1.type() == CV_32FC1) && (map2.type() == CV_32FC1))
    {
        CALL_HAL(remap32f, cv_hal_remap32f, src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows,
                 map1.ptr<float>(), map1.step, map2.ptr<float>(), map2.step, interpolation, borderType, borderValue.val);
    }

    const bool hasRelativeFlag = ((interpolation & cv::WARP_RELATIVE_MAP) != 0);

    interpolation &= ~cv::WARP_RELATIVE_MAP;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    int type = src.type(), depth = CV_MAT_DEPTH(type);

    if (interpolation == INTER_NEAREST && map1.depth() == CV_32F) {
        const auto *src_data = src.ptr<const uchar>();
        auto *dst_data = dst.ptr<uchar>();
        size_t src_step = src.step, dst_step = dst.step,
                map1_step = map1.step, map2_step = map2.step;
        int src_rows = src.rows, src_cols = src.cols;
        int dst_rows = dst.rows, dst_cols = dst.cols;
        const float *map1_data = map1.ptr<const float>();
        const float *map2_data = map2.ptr<const float>();
        switch (src.type()) {
            case CV_8UC1: {
                CV_CPU_DISPATCH(remapNearestInvoker_8UC1, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_8UC3: {
                CV_CPU_DISPATCH(remapNearestInvoker_8UC3, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_8UC4: {
                CV_CPU_DISPATCH(remapNearestInvoker_8UC4, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC1: {
                CV_CPU_DISPATCH(remapNearestInvoker_16UC1, ((const uint16_t*)src_data, src_step, src_rows, src_cols, (uint16_t*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC3: {
                CV_CPU_DISPATCH(remapNearestInvoker_16UC3, ((const uint16_t*)src_data, src_step, src_rows, src_cols, (uint16_t*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC4: {
                CV_CPU_DISPATCH(remapNearestInvoker_16UC4, ((const uint16_t*)src_data, src_step, src_rows, src_cols, (uint16_t*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC1: {
                CV_CPU_DISPATCH(remapNearestInvoker_32FC1, ((const float*)src_data, src_step, src_rows, src_cols, (float*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC3: {
                CV_CPU_DISPATCH(remapNearestInvoker_32FC3, ((const float*)src_data, src_step, src_rows, src_cols, (float*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC4: {
                CV_CPU_DISPATCH(remapNearestInvoker_32FC4, ((const float*)src_data, src_step, src_rows, src_cols, (float*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            // no default
        }
    }

    if (interpolation == INTER_LINEAR) {
        if (map1.depth() == CV_32F) {
            const auto *src_data = src.ptr<const uint8_t>();
            auto *dst_data = dst.ptr<uint8_t>();
            size_t src_step = src.step, dst_step = dst.step,
                   map1_step = map1.step, map2_step = map2.step;
            int src_rows = src.rows, src_cols = src.cols;
            int dst_rows = dst.rows, dst_cols = dst.cols;
            const float *map1_data = map1.ptr<const float>();
            const float *map2_data = map2.ptr<const float>();
            switch (src.type()) {
                case CV_8UC1: {
                    if (hint == cv::ALGO_HINT_APPROX) {
                        CV_CPU_DISPATCH(remapLinearApproxInvoker_8UC1, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    } else {
                        CV_CPU_DISPATCH(remapLinearInvoker_8UC1, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    }
                    break;
                }
                case CV_8UC3: {
                    if (hint == cv::ALGO_HINT_APPROX) {
                        CV_CPU_DISPATCH(remapLinearApproxInvoker_8UC3, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    } else {
                        CV_CPU_DISPATCH(remapLinearInvoker_8UC3, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    }
                    break;
                }
                case CV_8UC4: {
                    if (hint == cv::ALGO_HINT_APPROX) {
                        CV_CPU_DISPATCH(remapLinearApproxInvoker_8UC4, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    } else {
                        CV_CPU_DISPATCH(remapLinearInvoker_8UC4, (src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    }
                    break;
                }
                case CV_16UC1: {
                    CV_CPU_DISPATCH(remapLinearInvoker_16UC1, ((const uint16_t*)src_data, src_step, src_rows, src_cols, (uint16_t*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
                case CV_16UC3: {
                    CV_CPU_DISPATCH(remapLinearInvoker_16UC3, ((const uint16_t*)src_data, src_step, src_rows, src_cols, (uint16_t*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
                case CV_16UC4: {
                    CV_CPU_DISPATCH(remapLinearInvoker_16UC4, ((const uint16_t*)src_data, src_step, src_rows, src_cols, (uint16_t*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
                case CV_32FC1: {
                    CV_CPU_DISPATCH(remapLinearInvoker_32FC1, ((const float*)src_data, src_step, src_rows, src_cols, (float*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
                case CV_32FC3: {
                    CV_CPU_DISPATCH(remapLinearInvoker_32FC3, ((const float*)src_data, src_step, src_rows, src_cols, (float*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
                case CV_32FC4: {
                    CV_CPU_DISPATCH(remapLinearInvoker_32FC4, ((const float*)src_data, src_step, src_rows, src_cols, (float*)dst_data, dst_step, dst_rows, dst_cols, borderType, borderValue.val, map1_data, map1_step, map2_data, map2_step, hasRelativeFlag), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
                // no default
            }
        }
    }

#if defined HAVE_IPP && !IPP_DISABLE_REMAP
    CV_IPP_CHECK()
    {
        if ((interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_NEAREST) &&
                map1.type() == CV_32FC1 && map2.type() == CV_32FC1 &&
                (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT))
        {
            int ippInterpolation =
                interpolation == INTER_NEAREST ? IPPI_INTER_NN :
                interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR : IPPI_INTER_CUBIC;

            ippiRemap ippFunc =
                type == CV_8UC1 ? (ippiRemap)ippiRemap_8u_C1R :
                type == CV_8UC3 ? (ippiRemap)ippiRemap_8u_C3R :
                type == CV_8UC4 ? (ippiRemap)ippiRemap_8u_C4R :
                type == CV_16UC1 ? (ippiRemap)ippiRemap_16u_C1R :
                type == CV_16UC3 ? (ippiRemap)ippiRemap_16u_C3R :
                type == CV_16UC4 ? (ippiRemap)ippiRemap_16u_C4R :
                type == CV_32FC1 ? (ippiRemap)ippiRemap_32f_C1R :
                type == CV_32FC3 ? (ippiRemap)ippiRemap_32f_C3R :
                type == CV_32FC4 ? (ippiRemap)ippiRemap_32f_C4R : 0;

            if (ippFunc)
            {
                bool ok;
                IPPRemapInvoker invoker(src, dst, map1, map2, ippFunc, ippInterpolation,
                                        borderType, borderValue, &ok);
                Range range(0, dst.rows);
                parallel_for_(range, invoker, dst.total() / (double)(1 << 16));

                if (ok)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                    return;
                }
                setIppErrorStatus();
            }
        }
    }
#endif

    RemapNNFunc nnfunc = 0;
    RemapFunc ifunc = 0;
    const void* ctab = 0;
    bool fixpt = depth == CV_8U;
    bool planar_input = false;

    static RemapNNFunc nn_tab[2][CV_DEPTH_MAX] =
    {
        {
            remapNearest<uchar, false>, remapNearest<schar, false>, remapNearest<ushort, false>, remapNearest<short, false>,
            remapNearest<int, false>, remapNearest<float, false>, remapNearest<double, false>, 0
        },
        {
            remapNearest<uchar, true>, remapNearest<schar, true>, remapNearest<ushort, true>, remapNearest<short, true>,
            remapNearest<int, true>, remapNearest<float, true>, remapNearest<double, true>, 0
        }
    };

    static RemapFunc linear_tab[2][CV_DEPTH_MAX] =
    {
        {
            remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, RemapVec_8u<false>, short, false>, 0,
            remapBilinear<Cast<float, ushort>, RemapNoVec<false>, float, false>,
            remapBilinear<Cast<float, short>, RemapNoVec<false>, float, false>, 0,
            remapBilinear<Cast<float, float>, RemapNoVec<false>, float, false>,
            remapBilinear<Cast<double, double>, RemapNoVec<false>, float, false>, 0
        },
        {
            remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, RemapVec_8u<true>, short, true>, 0,
            remapBilinear<Cast<float, ushort>, RemapNoVec<true>, float, true>,
            remapBilinear<Cast<float, short>, RemapNoVec<true>, float, true>, 0,
            remapBilinear<Cast<float, float>, RemapNoVec<true>, float, true>,
            remapBilinear<Cast<double, double>, RemapNoVec<true>, float, true>, 0
        }
    };

    static RemapFunc cubic_tab[2][CV_DEPTH_MAX] =
    {
        {
            remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE, false>, 0,
            remapBicubic<Cast<float, ushort>, float, 1, false>,
            remapBicubic<Cast<float, short>, float, 1, false>, 0,
            remapBicubic<Cast<float, float>, float, 1, false>,
            remapBicubic<Cast<double, double>, float, 1, false>, 0
        },
        {
            remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE, true>, 0,
            remapBicubic<Cast<float, ushort>, float, 1, true>,
            remapBicubic<Cast<float, short>, float, 1, true>, 0,
            remapBicubic<Cast<float, float>, float, 1, true>,
            remapBicubic<Cast<double, double>, float, 1, true>, 0
        }
    };

    static RemapFunc lanczos4_tab[2][8] =
    {
        {
            remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE, false>, 0,
            remapLanczos4<Cast<float, ushort>, float, 1, false>,
            remapLanczos4<Cast<float, short>, float, 1, false>, 0,
            remapLanczos4<Cast<float, float>, float, 1, false>,
            remapLanczos4<Cast<double, double>, float, 1, false>, 0
        },
        {
            remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE, true>, 0,
            remapLanczos4<Cast<float, ushort>, float, 1, true>,
            remapLanczos4<Cast<float, short>, float, 1, true>, 0,
            remapLanczos4<Cast<float, float>, float, 1, true>,
            remapLanczos4<Cast<double, double>, float, 1, true>, 0
        }
    };

    const int relativeOptionIndex = (hasRelativeFlag ? 1 : 0);
    if( interpolation == INTER_NEAREST )
    {
        nnfunc = nn_tab[relativeOptionIndex][depth];
        CV_Assert( nnfunc != 0 );
    }
    else
    {
        if( interpolation == INTER_LINEAR )
            ifunc = linear_tab[relativeOptionIndex][depth];
        else if( interpolation == INTER_CUBIC ){
            ifunc = cubic_tab[relativeOptionIndex][depth];
            CV_Assert( _src.channels() <= 4 );
        }
        else if( interpolation == INTER_LANCZOS4 ){
            ifunc = lanczos4_tab[relativeOptionIndex][depth];
            CV_Assert( _src.channels() <= 4 );
        }
        else
            CV_Error( cv::Error::StsBadArg, "Unknown interpolation method" );
        CV_Assert( ifunc != 0 );
        ctab = initInterTab2D( interpolation, fixpt );
    }

    const Mat *m1 = &map1, *m2 = &map2;

    if( (map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1 || map2.empty())) ||
        (map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.type() == CV_16SC1 || map1.empty())) )
    {
        if( map1.type() != CV_16SC2 )
            std::swap(m1, m2);
    }
    else
    {
        CV_Assert( ((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && map2.empty()) ||
            (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) );
        planar_input = map1.channels() == 1;
    }

    RemapInvoker invoker(src, dst, m1, m2,
                         borderType, borderValue, planar_input, nnfunc, ifunc,
                         ctab);
    parallel_for_(Range(0, dst.rows), invoker, dst.total()/(double)(1<<16));
}


void cv::convertMaps( InputArray _map1, InputArray _map2,
                      OutputArray _dstmap1, OutputArray _dstmap2,
                      int dstm1type, bool nninterpolate )
{
    CV_INSTRUMENT_REGION();

    Mat map1 = _map1.getMat(), map2 = _map2.getMat(), dstmap1, dstmap2;
    Size size = map1.size();
    const Mat *m1 = &map1, *m2 = &map2;
    int m1type = m1->type(), m2type = m2->type();

    CV_Assert( (m1type == CV_16SC2 && (nninterpolate || m2type == CV_16UC1 || m2type == CV_16SC1)) ||
               (m2type == CV_16SC2 && (nninterpolate || m1type == CV_16UC1 || m1type == CV_16SC1)) ||
               (m1type == CV_32FC1 && m2type == CV_32FC1) ||
               (m1type == CV_32FC2 && m2->empty()) );

    if( m2type == CV_16SC2 )
    {
        std::swap( m1, m2 );
        std::swap( m1type, m2type );
    }

    if( dstm1type <= 0 )
        dstm1type = m1type == CV_16SC2 ? CV_32FC2 : CV_16SC2;
    CV_Assert( dstm1type == CV_16SC2 || dstm1type == CV_32FC1 || dstm1type == CV_32FC2 );
    _dstmap1.create( size, dstm1type );
    dstmap1 = _dstmap1.getMat();

    if( !nninterpolate && dstm1type != CV_32FC2 )
    {
        _dstmap2.create( size, dstm1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
        dstmap2 = _dstmap2.getMat();
    }
    else
        _dstmap2.release();

    if( m1type == dstm1type || (nninterpolate &&
        ((m1type == CV_16SC2 && dstm1type == CV_32FC2) ||
        (m1type == CV_32FC2 && dstm1type == CV_16SC2))) )
    {
        m1->convertTo( dstmap1, dstmap1.type() );
        if( !dstmap2.empty() && dstmap2.type() == m2->type() )
            m2->copyTo( dstmap2 );
        return;
    }

    if( m1type == CV_32FC1 && dstm1type == CV_32FC2 )
    {
        Mat vdata[] = { *m1, *m2 };
        merge( vdata, 2, dstmap1 );
        return;
    }

    if( m1type == CV_32FC2 && dstm1type == CV_32FC1 )
    {
        Mat mv[] = { dstmap1, dstmap2 };
        split( *m1, mv );
        return;
    }

    if( m1->isContinuous() && (m2->empty() || m2->isContinuous()) &&
        dstmap1.isContinuous() && (dstmap2.empty() || dstmap2.isContinuous()) )
    {
        size.width *= size.height;
        size.height = 1;
    }

#if CV_TRY_SSE4_1
    bool useSSE4_1 = CV_CPU_HAS_SUPPORT_SSE4_1;
#endif

    const float scale = 1.f/static_cast<float>(INTER_TAB_SIZE);
    int x, y;
    for( y = 0; y < size.height; y++ )
    {
        const float* src1f = m1->ptr<float>(y);
        const float* src2f = m2->ptr<float>(y);
        const short* src1 = (const short*)src1f;
        const ushort* src2 = (const ushort*)src2f;

        float* dst1f = dstmap1.ptr<float>(y);
        float* dst2f = dstmap2.ptr<float>(y);
        short* dst1 = (short*)dst1f;
        ushort* dst2 = (ushort*)dst2f;
        x = 0;

        if( m1type == CV_32FC1 && dstm1type == CV_16SC2 )
        {
            if( nninterpolate )
            {
                #if CV_TRY_SSE4_1
                if (useSSE4_1)
                    opt_SSE4_1::convertMaps_nninterpolate32f1c16s_SSE41(src1f, src2f, dst1, size.width);
                else
                #endif
                {
                    #if CV_SIMD128
                    {
                        int span = VTraits<v_int16x8>::vlanes();
                        for( ; x <= size.width - span; x += span )
                        {
                            v_int16x8 v_dst[2];
                            #define CV_PACK_MAP(X) v_pack(v_round(v_load(X)), v_round(v_load((X)+4)))
                            v_dst[0] = CV_PACK_MAP(src1f + x);
                            v_dst[1] = CV_PACK_MAP(src2f + x);
                            #undef CV_PACK_MAP
                            v_store_interleave(dst1 + (x << 1), v_dst[0], v_dst[1]);
                        }
                    }
                    #endif
                    for( ; x < size.width; x++ )
                    {
                        dst1[x*2] = saturate_cast<short>(src1f[x]);
                        dst1[x*2+1] = saturate_cast<short>(src2f[x]);
                    }
                }
            }
            else
            {
                #if CV_TRY_SSE4_1
                if (useSSE4_1)
                    opt_SSE4_1::convertMaps_32f1c16s_SSE41(src1f, src2f, dst1, dst2, size.width);
                else
                #endif
                {
                    #if CV_SIMD128
                    {
                        v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
                        v_int32x4 v_mask = v_setall_s32(INTER_TAB_SIZE - 1);
                        v_int32x4 v_scale3 = v_setall_s32(INTER_TAB_SIZE);
                        int span = VTraits<v_float32x4>::vlanes();
                        for( ; x <= size.width - span * 2; x += span * 2 )
                        {
                            v_int32x4 v_ix0 = v_round(v_mul(v_scale, v_load(src1f + x)));
                            v_int32x4 v_ix1 = v_round(v_mul(v_scale, v_load(src1f + x + span)));
                            v_int32x4 v_iy0 = v_round(v_mul(v_scale, v_load(src2f + x)));
                            v_int32x4 v_iy1 = v_round(v_mul(v_scale, v_load(src2f + x + span)));

                            v_int16x8 v_dst[2];
                            v_dst[0] = v_pack(v_shr<INTER_BITS>(v_ix0), v_shr<INTER_BITS>(v_ix1));
                            v_dst[1] = v_pack(v_shr<INTER_BITS>(v_iy0), v_shr<INTER_BITS>(v_iy1));
                            v_store_interleave(dst1 + (x << 1), v_dst[0], v_dst[1]);

                            v_int32x4 v_dst0 = v_muladd(v_scale3, (v_and(v_iy0, v_mask)), (v_and(v_ix0, v_mask)));
                            v_int32x4 v_dst1 = v_muladd(v_scale3, (v_and(v_iy1, v_mask)), (v_and(v_ix1, v_mask)));
                            v_store(dst2 + x, v_pack_u(v_dst0, v_dst1));
                        }
                    }
                    #endif
                    for( ; x < size.width; x++ )
                    {
                        int ix = saturate_cast<int>(src1f[x]*static_cast<float>(INTER_TAB_SIZE));
                        int iy = saturate_cast<int>(src2f[x]*static_cast<float>(INTER_TAB_SIZE));
                        dst1[x*2] = saturate_cast<short>(ix >> INTER_BITS);
                        dst1[x*2+1] = saturate_cast<short>(iy >> INTER_BITS);
                        dst2[x] = (ushort)((iy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE-1)));
                    }
                }
            }
        }
        else if( m1type == CV_32FC2 && dstm1type == CV_16SC2 )
        {
            #if CV_TRY_SSE4_1
            if( useSSE4_1 )
                opt_SSE4_1::convertMaps_32f2c16s_SSE41(src1f, dst1, dst2, size.width);
            else
            #endif
            {
                #if CV_SIMD128
                {
                    v_float32x4 v_scale = v_setall_f32((float)INTER_TAB_SIZE);
                    v_int32x4 v_mask = v_setall_s32(INTER_TAB_SIZE - 1);
                    v_int32x4 v_scale3 = v_setall_s32(INTER_TAB_SIZE);
                    int span = VTraits<v_uint16x8>::vlanes();
                    for (; x <= size.width - span; x += span )
                    {
                        v_float32x4 v_src0[2], v_src1[2];
                        v_load_deinterleave(src1f + (x << 1), v_src0[0], v_src0[1]);
                        v_load_deinterleave(src1f + (x << 1) + span, v_src1[0], v_src1[1]);
                        v_int32x4 v_ix0 = v_round(v_mul(v_src0[0], v_scale));
                        v_int32x4 v_ix1 = v_round(v_mul(v_src1[0], v_scale));
                        v_int32x4 v_iy0 = v_round(v_mul(v_src0[1], v_scale));
                        v_int32x4 v_iy1 = v_round(v_mul(v_src1[1], v_scale));

                        v_int16x8 v_dst[2];
                        v_dst[0] = v_pack(v_shr<INTER_BITS>(v_ix0), v_shr<INTER_BITS>(v_ix1));
                        v_dst[1] = v_pack(v_shr<INTER_BITS>(v_iy0), v_shr<INTER_BITS>(v_iy1));
                        v_store_interleave(dst1 + (x << 1), v_dst[0], v_dst[1]);

                        v_store(dst2 + x, v_pack_u(
                            v_muladd(v_scale3, (v_and(v_iy0, v_mask)), (v_and(v_ix0, v_mask))),
                            v_muladd(v_scale3, (v_and(v_iy1, v_mask)), (v_and(v_ix1, v_mask)))));
                    }
                }
                #endif
                for( ; x < size.width; x++ )
                {
                    int ix = saturate_cast<int>(src1f[x*2]*static_cast<float>(INTER_TAB_SIZE));
                    int iy = saturate_cast<int>(src1f[x*2+1]*static_cast<float>(INTER_TAB_SIZE));
                    dst1[x*2] = saturate_cast<short>(ix >> INTER_BITS);
                    dst1[x*2+1] = saturate_cast<short>(iy >> INTER_BITS);
                    dst2[x] = (ushort)((iy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE-1)));
                }
            }
        }
        else if( m1type == CV_16SC2 && dstm1type == CV_32FC1 )
        {
            #if CV_SIMD128
            {
                v_uint16x8 v_mask2 =  v_setall_u16(INTER_TAB_SIZE2-1);
                v_uint32x4 v_zero =   v_setzero_u32(), v_mask = v_setall_u32(INTER_TAB_SIZE-1);
                v_float32x4 v_scale = v_setall_f32(scale);
                int span = VTraits<v_float32x4>::vlanes();
                for( ; x <= size.width - span * 2; x += span * 2 )
                {
                    v_uint32x4 v_fxy1, v_fxy2;
                    if ( src2 )
                    {
                        v_uint16x8 v_src2 = v_and(v_load(src2 + x), v_mask2);
                        v_expand(v_src2, v_fxy1, v_fxy2);
                    }
                    else
                        v_fxy1 = v_fxy2 = v_zero;

                    v_int16x8 v_src[2];
                    v_int32x4 v_src0[2], v_src1[2];
                    v_load_deinterleave(src1 + (x << 1), v_src[0], v_src[1]);
                    v_expand(v_src[0], v_src0[0], v_src0[1]);
                    v_expand(v_src[1], v_src1[0], v_src1[1]);
                    #define CV_COMPUTE_MAP_X(X, FXY)  v_muladd(v_scale, v_cvt_f32(v_reinterpret_as_s32(v_and((FXY), v_mask))),\
                                                                        v_cvt_f32(v_reinterpret_as_s32(X)))
                    #define CV_COMPUTE_MAP_Y(Y, FXY)  v_muladd(v_scale, v_cvt_f32(v_reinterpret_as_s32(v_shr<INTER_BITS>((FXY)))),\
                                                                        v_cvt_f32(v_reinterpret_as_s32(Y)))
                    v_float32x4 v_dst1 = CV_COMPUTE_MAP_X(v_src0[0], v_fxy1);
                    v_float32x4 v_dst2 = CV_COMPUTE_MAP_Y(v_src1[0], v_fxy1);
                    v_store(dst1f + x, v_dst1);
                    v_store(dst2f + x, v_dst2);

                    v_dst1 = CV_COMPUTE_MAP_X(v_src0[1], v_fxy2);
                    v_dst2 = CV_COMPUTE_MAP_Y(v_src1[1], v_fxy2);
                    v_store(dst1f + x + span, v_dst1);
                    v_store(dst2f + x + span, v_dst2);
                    #undef CV_COMPUTE_MAP_X
                    #undef CV_COMPUTE_MAP_Y
                }
            }
            #endif
            for( ; x < size.width; x++ )
            {
                int fxy = src2 ? src2[x] & (INTER_TAB_SIZE2-1) : 0;
                dst1f[x] = src1[x*2] + (fxy & (INTER_TAB_SIZE-1))*scale;
                dst2f[x] = src1[x*2+1] + (fxy >> INTER_BITS)*scale;
            }
        }
        else if( m1type == CV_16SC2 && dstm1type == CV_32FC2 )
        {
            #if CV_SIMD128
            {
                v_int16x8 v_mask2 = v_setall_s16(INTER_TAB_SIZE2-1);
                v_int32x4 v_zero = v_setzero_s32(), v_mask = v_setall_s32(INTER_TAB_SIZE-1);
                v_float32x4 v_scale = v_setall_f32(scale);
                int span = VTraits<v_int16x8>::vlanes();
                for( ; x <= size.width - span; x += span )
                {
                    v_int32x4 v_fxy1, v_fxy2;
                    if (src2)
                    {
                        v_int16x8 v_src2 = v_and(v_load((short *)src2 + x), v_mask2);
                        v_expand(v_src2, v_fxy1, v_fxy2);
                    }
                    else
                        v_fxy1 = v_fxy2 = v_zero;

                    v_int16x8 v_src[2];
                    v_int32x4 v_src0[2], v_src1[2];
                    v_float32x4 v_dst[2];
                    v_load_deinterleave(src1 + (x << 1), v_src[0], v_src[1]);
                    v_expand(v_src[0], v_src0[0], v_src0[1]);
                    v_expand(v_src[1], v_src1[0], v_src1[1]);

                    #define CV_COMPUTE_MAP_X(X, FXY) v_muladd(v_scale, v_cvt_f32(v_and((FXY), v_mask)), v_cvt_f32(X))
                    #define CV_COMPUTE_MAP_Y(Y, FXY) v_muladd(v_scale, v_cvt_f32(v_shr<INTER_BITS>((FXY))), v_cvt_f32(Y))
                    v_dst[0] = CV_COMPUTE_MAP_X(v_src0[0], v_fxy1);
                    v_dst[1] = CV_COMPUTE_MAP_Y(v_src1[0], v_fxy1);
                    v_store_interleave(dst1f + (x << 1), v_dst[0], v_dst[1]);

                    v_dst[0] = CV_COMPUTE_MAP_X(v_src0[1], v_fxy2);
                    v_dst[1] = CV_COMPUTE_MAP_Y(v_src1[1], v_fxy2);
                    v_store_interleave(dst1f + (x << 1) + span, v_dst[0], v_dst[1]);
                    #undef CV_COMPUTE_MAP_X
                    #undef CV_COMPUTE_MAP_Y
                }
            }
            #endif
            for( ; x < size.width; x++ )
            {
                int fxy = src2 ? src2[x] & (INTER_TAB_SIZE2-1): 0;
                dst1f[x*2] = src1[x*2] + (fxy & (INTER_TAB_SIZE-1))*scale;
                dst1f[x*2+1] = src1[x*2+1] + (fxy >> INTER_BITS)*scale;
            }
        }
        else
            CV_Error( cv::Error::StsNotImplemented, "Unsupported combination of input/output matrices" );
    }
}


namespace cv
{

class WarpAffineInvoker :
    public ParallelLoopBody
{
public:
    WarpAffineInvoker(const Mat &_src, Mat &_dst, int _interpolation, int _borderType,
                      const Scalar &_borderValue, int *_adelta, int *_bdelta, const double *_M) :
        ParallelLoopBody(), src(_src), dst(_dst), interpolation(_interpolation),
        borderType(_borderType), borderValue(_borderValue), adelta(_adelta), bdelta(_bdelta),
        M(_M)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int BLOCK_SZ = 64;
        AutoBuffer<short, 0> __XY(BLOCK_SZ * BLOCK_SZ * 2), __A(BLOCK_SZ * BLOCK_SZ);
        short *XY = __XY.data(), *A = __A.data();
        const int AB_BITS = MAX(10, (int)INTER_BITS);
        const int AB_SCALE = 1 << AB_BITS;
        int round_delta = interpolation == INTER_NEAREST ? AB_SCALE/2 : AB_SCALE/INTER_TAB_SIZE/2, x, y, y1;

        int bh0 = std::min(BLOCK_SZ/2, dst.rows);
        int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, dst.cols);
        bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, dst.rows);

        for( y = range.start; y < range.end; y += bh0 )
        {
            for( x = 0; x < dst.cols; x += bw0 )
            {
                int bw = std::min( bw0, dst.cols - x);
                int bh = std::min( bh0, range.end - y);

                Mat _XY(bh, bw, CV_16SC2, XY);
                Mat dpart(dst, Rect(x, y, bw, bh));

                for( y1 = 0; y1 < bh; y1++ )
                {
                    short* xy = XY + y1*bw*2;
                    int X0 = saturate_cast<int>((M[1]*(y + y1) + M[2])*AB_SCALE) + round_delta;
                    int Y0 = saturate_cast<int>((M[4]*(y + y1) + M[5])*AB_SCALE) + round_delta;

                    if( interpolation == INTER_NEAREST )
                        hal::warpAffineBlocklineNN(adelta + x, bdelta + x, xy, X0, Y0, bw);
                    else
                        hal::warpAffineBlockline(adelta + x, bdelta + x, xy, A + y1*bw, X0, Y0, bw);
                }

                if( interpolation == INTER_NEAREST )
                    remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
                else
                {
                    Mat _matA(bh, bw, CV_16U, A);
                    remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
                }
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int interpolation, borderType;
    Scalar borderValue;
    int *adelta, *bdelta;
    const double *M;
};


#if defined (HAVE_IPP) && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_WARPAFFINE
typedef IppStatus (CV_STDCALL* ippiWarpAffineBackFunc)(const void*, IppiSize, int, IppiRect, void *, int, IppiRect, double [2][3], int);

class IPPWarpAffineInvoker :
    public ParallelLoopBody
{
public:
    IPPWarpAffineInvoker(Mat &_src, Mat &_dst, double (&_coeffs)[2][3], int &_interpolation, int _borderType,
                         const Scalar &_borderValue, ippiWarpAffineBackFunc _func, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), mode(_interpolation), coeffs(_coeffs),
        borderType(_borderType), borderValue(_borderValue), func(_func), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        IppiSize srcsize = { src.cols, src.rows };
        IppiRect srcroi = { 0, 0, src.cols, src.rows };
        IppiRect dstroi = { 0, range.start, dst.cols, range.end - range.start };
        int cnn = src.channels();
        if( borderType == BORDER_CONSTANT )
        {
            IppiSize setSize = { dst.cols, range.end - range.start };
            void *dataPointer = dst.ptr(range.start);
            if( !IPPSet( borderValue, dataPointer, (int)dst.step[0], setSize, cnn, src.depth() ) )
            {
                *ok = false;
                return;
            }
        }

        // Aug 2013: problem in IPP 7.1, 8.0 : sometimes function return ippStsCoeffErr
        IppStatus status = CV_INSTRUMENT_FUN_IPP(func,( src.ptr(), srcsize, (int)src.step[0], srcroi, dst.ptr(),
                                (int)dst.step[0], dstroi, coeffs, mode ));
        if( status < 0)
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        }
    }
private:
    Mat &src;
    Mat &dst;
    int mode;
    double (&coeffs)[2][3];
    int borderType;
    Scalar borderValue;
    ippiWarpAffineBackFunc func;
    bool *ok;
    const IPPWarpAffineInvoker& operator= (const IPPWarpAffineInvoker&);
};
#endif

#ifdef HAVE_OPENCL

enum { OCL_OP_PERSPECTIVE = 1, OCL_OP_AFFINE = 0 };

static bool ocl_warpTransform_cols4(InputArray _src, OutputArray _dst, InputArray _M0,
                                    Size dsize, int flags, int borderType, const Scalar& borderValue,
                                    int op_type)
{
    CV_Assert(op_type == OCL_OP_AFFINE || op_type == OCL_OP_PERSPECTIVE);
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), dtype = _dst.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    if ( !dev.isIntel() || !(type == CV_8UC1) ||
         !(dtype == CV_8UC1) || !(_dst.cols() % 4 == 0) ||
         (op_type == OCL_OP_PERSPECTIVE && interpolation == INTER_LINEAR && (cn == 1 || cn == 3 || cn == 4)) ||
         !(borderType == cv::BORDER_CONSTANT &&
          (interpolation == cv::INTER_NEAREST || interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC)))
        return false;

    const char * const warp_op[2] = { "Affine", "Perspective" };
    const char * const interpolationMap[3] = { "nearest", "linear", "cubic" };
    ocl::ProgramSource program = ocl::imgproc::warp_transform_oclsrc;
    String kernelName = format("warp%s_%s_8u", warp_op[op_type], interpolationMap[interpolation]);

    bool is32f = (interpolation == INTER_CUBIC || interpolation == INTER_LINEAR) && op_type == OCL_OP_AFFINE;
    int wdepth = interpolation == INTER_NEAREST ? depth : std::max(is32f ? CV_32F : CV_32S, depth);
    int sctype = CV_MAKETYPE(wdepth, cn);

    ocl::Kernel k;
    String opts = format("-D ST=%s", ocl::typeToStr(sctype));

    k.create(kernelName.c_str(), program, opts);
    if (k.empty())
        return false;

    float borderBuf[] = { 0, 0, 0, 0 };
    scalarToRawData(borderValue, borderBuf, sctype);

    UMat src = _src.getUMat(), M0;
    _dst.create( dsize.empty() ? src.size() : dsize, src.type() );
    UMat dst = _dst.getUMat();

    if (src.u == dst.u)
        src = src.clone();

    float M[9] = {0};
    int matRows = (op_type == OCL_OP_AFFINE ? 2 : 3);
    Mat matM(matRows, 3, CV_32F, M), M1 = _M0.getMat();
    CV_Assert( (M1.type() == CV_32F || M1.type() == CV_64F) && M1.rows == matRows && M1.cols == 3 );
    M1.convertTo(matM, matM.type());

    if( !(flags & WARP_INVERSE_MAP) )
    {
        if (op_type == OCL_OP_PERSPECTIVE)
            invert(matM, matM);
        else
        {
            float D = M[0]*M[4] - M[1]*M[3];
            D = D != 0 ? 1.f/D : 0;
            float A11 = M[4]*D, A22=M[0]*D;
            M[0] = A11; M[1] *= -D;
            M[3] *= -D; M[4] = A22;
            float b1 = -M[0]*M[2] - M[1]*M[5];
            float b2 = -M[3]*M[2] - M[4]*M[5];
            M[2] = b1; M[5] = b2;
        }
    }
    matM.convertTo(M0, CV_32F);

    k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst), ocl::KernelArg::PtrReadOnly(M0),
           ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, borderBuf, CV_ELEM_SIZE(sctype)));

    size_t globalThreads[2];
    globalThreads[0] = (size_t)(dst.cols / 4);
    globalThreads[1] = (size_t)dst.rows;

    return k.run(2, globalThreads, NULL, false);
}

static bool ocl_warpTransform(InputArray _src, OutputArray _dst, InputArray _M0,
                              Size dsize, int flags, int borderType, const Scalar& borderValue,
                              int op_type)
{
    CV_Assert(op_type == OCL_OP_AFFINE || op_type == OCL_OP_PERSPECTIVE);
    const ocl::Device & dev = ocl::Device::getDefault();

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    const bool doubleSupport = dev.doubleFPConfig() > 0;

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;
    int rowsPerWI = dev.isIntel() && op_type == OCL_OP_AFFINE && interpolation <= INTER_LINEAR ? 4 : 1;

    if ( !(borderType == cv::BORDER_CONSTANT &&
           (interpolation == cv::INTER_NEAREST || interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC)) ||
         (!doubleSupport && depth == CV_64F) || cn > 4)
        return false;

    bool useDouble = depth == CV_64F;

    const char * const interpolationMap[3] = { "NEAREST", "LINEAR", "CUBIC" };
    ocl::ProgramSource program = op_type == OCL_OP_AFFINE ?
                ocl::imgproc::warp_affine_oclsrc : ocl::imgproc::warp_perspective_oclsrc;
    const char * const kernelName = op_type == OCL_OP_AFFINE ? "warpAffine" : "warpPerspective";

    int scalarcn = cn == 3 ? 4 : cn;
    bool is32f = op_type == OCL_OP_AFFINE ?
                 /* Affine*/ !dev.isAMD() && (interpolation == INTER_CUBIC || interpolation == INTER_LINEAR) :
                 /* Perspective*/ interpolation == INTER_LINEAR;
    int wdepth = interpolation == INTER_NEAREST ? depth : std::max(is32f ? CV_32F : CV_32S, depth);
    int sctype = CV_MAKETYPE(wdepth, scalarcn);

    ocl::Kernel k;
    String opts;
    if (interpolation == INTER_NEAREST)
    {
        opts = format("-D INTER_NEAREST -D T=%s%s -D CT=%s -D T1=%s -D ST=%s -D CN=%d -D ROWS_PER_WI=%d",
                      ocl::typeToStr(type),
                      doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                      useDouble ? "double" : "float",
                      ocl::typeToStr(CV_MAT_DEPTH(type)),
                      ocl::typeToStr(sctype), cn, rowsPerWI);
    }
    else {
        char cvt[2][50];
        opts = format("-D INTER_%s -D T=%s -D T1=%s -D ST=%s -D WT=%s -D SRC_DEPTH=%d"
                      " -D CONVERT_TO_WT=%s -D CONVERT_TO_T=%s%s -D CT=%s -D CN=%d -D ROWS_PER_WI=%d",
                      interpolationMap[interpolation], ocl::typeToStr(type),
                      ocl::typeToStr(CV_MAT_DEPTH(type)),
                      ocl::typeToStr(sctype),
                      ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)), depth,
                      ocl::convertTypeStr(depth, wdepth, cn, cvt[0], sizeof(cvt[0])),
                      ocl::convertTypeStr(wdepth, depth, cn, cvt[1], sizeof(cvt[1])),
                      doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                      useDouble ? "double" : "float",
                      cn, rowsPerWI);
    }

    k.create(kernelName, program, opts);
    if (k.empty())
        return false;

    double borderBuf[] = { 0, 0, 0, 0 };
    scalarToRawData(borderValue, borderBuf, sctype);

    UMat src = _src.getUMat(), M0;
    _dst.create( dsize.empty() ? src.size() : dsize, src.type() );
    UMat dst = _dst.getUMat();

    if (src.u == dst.u)
        src = src.clone();

    double M[9] = {0};
    int matRows = (op_type == OCL_OP_AFFINE ? 2 : 3);
    Mat matM(matRows, 3, CV_64F, M), M1 = _M0.getMat();
    CV_Assert( (M1.type() == CV_32F || M1.type() == CV_64F) &&
               M1.rows == matRows && M1.cols == 3 );
    M1.convertTo(matM, matM.type());

    if( !(flags & WARP_INVERSE_MAP) )
    {
        if (op_type == OCL_OP_PERSPECTIVE)
            invert(matM, matM);
        else
        {
            double D = M[0]*M[4] - M[1]*M[3];
            D = D != 0 ? 1./D : 0;
            double A11 = M[4]*D, A22=M[0]*D;
            M[0] = A11; M[1] *= -D;
            M[3] *= -D; M[4] = A22;
            double b1 = -M[0]*M[2] - M[1]*M[5];
            double b2 = -M[3]*M[2] - M[4]*M[5];
            M[2] = b1; M[5] = b2;
        }
    }
    matM.convertTo(M0, useDouble ? CV_64F : CV_32F);

    k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst), ocl::KernelArg::PtrReadOnly(M0),
           ocl::KernelArg(ocl::KernelArg::CONSTANT, 0, 0, 0, borderBuf, CV_ELEM_SIZE(sctype)));

    size_t globalThreads[2] = { (size_t)dst.cols, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalThreads, NULL, false);
}

#endif

#ifdef HAVE_IPP
#define IPP_WARPAFFINE_PARALLEL 1

#ifdef HAVE_IPP_IW

class ipp_warpAffineParallel: public ParallelLoopBody
{
public:
    ipp_warpAffineParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, IppiInterpolationType _inter, double (&_coeffs)[2][3], ::ipp::IwiBorderType _borderType, IwTransDirection _iwTransDirection, bool *_ok):m_src(src), m_dst(dst)
    {
        pOk = _ok;

        inter          = _inter;
        borderType     = _borderType;
        iwTransDirection = _iwTransDirection;

        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = _coeffs[i][j];

        *pOk = true;
    }
    ~ipp_warpAffineParallel() {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION_IPP();

        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, m_src, m_dst, coeffs, iwTransDirection, inter, ::ipp::IwiWarpAffineParams(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    IppiInterpolationType inter;
    double coeffs[2][3];
    ::ipp::IwiBorderType borderType;
    IwTransDirection iwTransDirection;

    bool  *pOk;
    const ipp_warpAffineParallel& operator= (const ipp_warpAffineParallel&);
};

#endif

static bool ipp_warpAffine( InputArray _src, OutputArray _dst, int interpolation, int borderType, const Scalar & borderValue, InputArray _M, int flags )
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    if (!cv::ipp::useIPP_NotExact())
        return false;

    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    if((int)ippInter < 0)
        return false;

    // Acquire data and begin processing
    try
    {
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        ::ipp::IwiImage        iwSrc = ippiGetImage(src);
        ::ipp::IwiImage        iwDst = ippiGetImage(dst);
        ::ipp::IwiBorderType   ippBorder(ippiGetBorderType(borderType), ippiGetValue(borderValue));
        IwTransDirection       iwTransDirection;
        if(!ippBorder)
            return false;

        if( !(flags & WARP_INVERSE_MAP) )
            iwTransDirection = iwTransForward;
        else
            iwTransDirection = iwTransInverse;

        Mat M = _M.getMat();
        double coeffs[2][3];
        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = M.at<double>(i, j);

        const int threads = ippiSuggestThreadsNum(iwDst, 2);

        if(IPP_WARPAFFINE_PARALLEL && threads > 1)
        {
            bool  ok      = true;
            Range range(0, (int)iwDst.m_size.height);
            ipp_warpAffineParallel invoker(iwSrc, iwDst, ippInter, coeffs, ippBorder, iwTransDirection, &ok);
            if(!ok)
                return false;

            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return false;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, iwSrc, iwDst, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpAffineParams(), ippBorder);
        }

    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }

    return true;
#else
    CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(interpolation);
    CV_UNUSED(borderType); CV_UNUSED(borderValue); CV_UNUSED(_M); CV_UNUSED(flags);
    return false;
#endif
}

#endif

namespace hal {

static void warpAffine(int src_type,
                       const uchar * src_data, size_t src_step, int src_width, int src_height,
                       uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                       const double M[6], int interpolation, int borderType, const double borderValue[4], AlgorithmHint hint)
{
    CALL_HAL(warpAffine, cv_hal_warpAffine, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue);

    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(Size(dst_width, dst_height), src_type, dst_data, dst_step);

    if (interpolation == INTER_NEAREST) {
        switch (src_type) {
            case CV_8UC1: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_8UC1, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_8UC3: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_8UC3, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_8UC4: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_8UC4, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC1: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_16UC1, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC3: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_16UC3, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC4: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_16UC4, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC1: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_32FC1, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC3: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_32FC3, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC4: {
                CV_CPU_DISPATCH(warpAffineNearestInvoker_32FC4, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            // no default
        }
    }

    if (interpolation == INTER_LINEAR) {
        switch (src_type) {
            case CV_8UC1: {
                if (hint == cv::ALGO_HINT_APPROX) {
                    CV_CPU_DISPATCH(warpAffineLinearApproxInvoker_8UC1, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                } else {
                    CV_CPU_DISPATCH(warpAffineLinearInvoker_8UC1, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                }
                break;
            }
            case CV_8UC3: {
                if (hint == cv::ALGO_HINT_APPROX) {
                    CV_CPU_DISPATCH(warpAffineLinearApproxInvoker_8UC3, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                } else {
                    CV_CPU_DISPATCH(warpAffineLinearInvoker_8UC3, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                }
                break;
            }
            case CV_8UC4: {
                if (hint == cv::ALGO_HINT_APPROX) {
                    CV_CPU_DISPATCH(warpAffineLinearApproxInvoker_8UC4, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                } else {
                    CV_CPU_DISPATCH(warpAffineLinearInvoker_8UC4, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                }
                break;
            }
            case CV_16UC1: {
                CV_CPU_DISPATCH(warpAffineLinearInvoker_16UC1, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC3: {
                CV_CPU_DISPATCH(warpAffineLinearInvoker_16UC3, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC4: {
                CV_CPU_DISPATCH(warpAffineLinearInvoker_16UC4, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC1: {
                CV_CPU_DISPATCH(warpAffineLinearInvoker_32FC1, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC3: {
                CV_CPU_DISPATCH(warpAffineLinearInvoker_32FC3, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC4: {
                CV_CPU_DISPATCH(warpAffineLinearInvoker_32FC4, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            // no default
        }
    }

    int x;
    AutoBuffer<int> _abdelta(dst.cols*2);
    int* adelta = &_abdelta[0], *bdelta = adelta + dst.cols;
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    const int AB_SCALE = 1 << AB_BITS;

    for( x = 0; x < dst.cols; x++ )
    {
        adelta[x] = saturate_cast<int>(M[0]*x*AB_SCALE);
        bdelta[x] = saturate_cast<int>(M[3]*x*AB_SCALE);
    }

    Range range(0, dst.rows);
    WarpAffineInvoker invoker(src, dst, interpolation, borderType,
                              Scalar(borderValue[0], borderValue[1], borderValue[2], borderValue[3]),
                              adelta, bdelta, M);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

void warpAffineBlocklineNN(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw)
{
    CALL_HAL(warpAffineBlocklineNN, cv_hal_warpAffineBlocklineNN, adelta, bdelta, xy, X0, Y0, bw);

    constexpr int AB_BITS = MAX(10, static_cast<int>(INTER_BITS));
    int x1 = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        const v_int32 v_X0 = vx_setall_s32(X0);
        const v_int32 v_Y0 = vx_setall_s32(Y0);
        const int step = VTraits<v_int16>::vlanes();
        for (; x1 <= bw - step; x1 += step)
        {
            v_int16 v_X = v_pack(v_shr<AB_BITS>(v_add(v_X0, vx_load(adelta + x1))),
                                 v_shr<AB_BITS>(v_add(v_X0, vx_load(adelta + x1 + step / 2))));
            v_int16 v_Y = v_pack(v_shr<AB_BITS>(v_add(v_Y0, vx_load(bdelta + x1))),
                                 v_shr<AB_BITS>(v_add(v_Y0, vx_load(bdelta + x1 + step / 2))));
            v_store_interleave(xy + 2 * x1, v_X, v_Y);
        }
    }
#endif
    for (; x1 < bw; x1++)
    {
        const int X = (X0 + adelta[x1]) >> AB_BITS;
        const int Y = (Y0 + bdelta[x1]) >> AB_BITS;
        xy[x1 * 2] = saturate_cast<short>(X);
        xy[x1 * 2 + 1] = saturate_cast<short>(Y);
    }
}

void warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw)
{
    CALL_HAL(warpAffineBlockline, cv_hal_warpAffineBlockline, adelta, bdelta, xy, alpha, X0, Y0, bw);

    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int x1 = 0;
    #if CV_TRY_AVX2
    bool useAVX2 = CV_CPU_HAS_SUPPORT_AVX2;
    if ( useAVX2 )
        x1 = opt_AVX2::warpAffineBlockline(adelta, bdelta, xy, alpha, X0, Y0, bw);
    #endif
    #if CV_TRY_LASX
    bool useLASX = CV_CPU_HAS_SUPPORT_LASX;
    if ( useLASX )
        x1 = opt_LASX::warpAffineBlockline(adelta, bdelta, xy, alpha, X0, Y0, bw);
    #endif
    {
        #if CV_SIMD128
        {
            v_int32x4 v__X0 = v_setall_s32(X0), v__Y0 = v_setall_s32(Y0);
            v_int32x4 v_mask = v_setall_s32(INTER_TAB_SIZE - 1);
            int span = VTraits<v_float32x4>::vlanes();
            for( ; x1 <= bw - span * 2; x1 += span * 2 )
            {
                v_int32x4 v_X0 = v_shr<AB_BITS - INTER_BITS>(v_add(v__X0, v_load(adelta + x1)));
                v_int32x4 v_Y0 = v_shr<AB_BITS - INTER_BITS>(v_add(v__Y0, v_load(bdelta + x1)));
                v_int32x4 v_X1 = v_shr<AB_BITS - INTER_BITS>(v_add(v__X0, v_load(adelta + x1 + span)));
                v_int32x4 v_Y1 = v_shr<AB_BITS - INTER_BITS>(v_add(v__Y0, v_load(bdelta + x1 + span)));

                v_int16x8 v_xy[2];
                v_xy[0] = v_pack(v_shr<INTER_BITS>(v_X0), v_shr<INTER_BITS>(v_X1));
                v_xy[1] = v_pack(v_shr<INTER_BITS>(v_Y0), v_shr<INTER_BITS>(v_Y1));
                v_store_interleave(xy + (x1 << 1), v_xy[0], v_xy[1]);

                v_int32x4 v_alpha0 = v_or(v_shl<INTER_BITS>(v_and(v_Y0, v_mask)), v_and(v_X0, v_mask));
                v_int32x4 v_alpha1 = v_or(v_shl<INTER_BITS>(v_and(v_Y1, v_mask)), v_and(v_X1, v_mask));
                v_store(alpha + x1, v_pack(v_alpha0, v_alpha1));
            }
        }
        #endif
        for( ; x1 < bw; x1++ )
        {
            int X = (X0 + adelta[x1]) >> (AB_BITS - INTER_BITS);
            int Y = (Y0 + bdelta[x1]) >> (AB_BITS - INTER_BITS);
            xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
            xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
            alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
                    (X & (INTER_TAB_SIZE-1)));
        }
    }
}

} // hal::
} // cv::


void cv::warpAffine( InputArray _src, OutputArray _dst,
                     InputArray _M0, Size dsize,
                     int flags, int borderType, const Scalar& borderValue,
                     AlgorithmHint hint )
{
    CV_INSTRUMENT_REGION();

    if (hint == cv::ALGO_HINT_DEFAULT)
        hint = cv::getDefaultAlgorithmHint();

    int interpolation = flags & INTER_MAX;
    CV_Assert( _src.channels() <= 4 || (interpolation != INTER_LANCZOS4 &&
                                        interpolation != INTER_CUBIC) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat() &&
               _src.cols() <= SHRT_MAX && _src.rows() <= SHRT_MAX,
               ocl_warpTransform_cols4(_src, _dst, _M0, dsize, flags, borderType,
                                       borderValue, OCL_OP_AFFINE))

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_warpTransform(_src, _dst, _M0, dsize, flags, borderType,
                                 borderValue, OCL_OP_AFFINE))

    Mat src = _src.getMat(), M0 = _M0.getMat();
    _dst.create( dsize.empty() ? src.size() : dsize, src.type() );
    Mat dst = _dst.getMat();
    CV_Assert( src.cols > 0 && src.rows > 0 );
    if( dst.data == src.data )
        src = src.clone();

    double M[6] = {0};
    Mat matM(2, 3, CV_64F, M);
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 2 && M0.cols == 3 );
    M0.convertTo(matM, matM.type());

    CV_IPP_RUN_FAST(ipp_warpAffine(src, dst, interpolation, borderType, borderValue, matM, flags));

    if( !(flags & WARP_INVERSE_MAP) )
    {
        double D = M[0]*M[4] - M[1]*M[3];
        D = D != 0 ? 1./D : 0;
        double A11 = M[4]*D, A22=M[0]*D;
        M[0] = A11; M[1] *= -D;
        M[3] *= -D; M[4] = A22;
        double b1 = -M[0]*M[2] - M[1]*M[5];
        double b2 = -M[3]*M[2] - M[4]*M[5];
        M[2] = b1; M[5] = b2;
    }

#if defined (HAVE_IPP) && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_WARPAFFINE
    CV_IPP_CHECK()
    {
        int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
        if( ( depth == CV_8U || depth == CV_16U || depth == CV_32F ) &&
           ( cn == 1 || cn == 3 || cn == 4 ) &&
           ( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC) &&
           ( borderType == cv::BORDER_TRANSPARENT || borderType == cv::BORDER_CONSTANT) )
        {
            ippiWarpAffineBackFunc ippFunc = 0;
            if ((flags & WARP_INVERSE_MAP) != 0)
            {
                ippFunc =
                type == CV_8UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_8u_C1R :
                type == CV_8UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_8u_C3R :
                type == CV_8UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_8u_C4R :
                type == CV_16UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_16u_C1R :
                type == CV_16UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_16u_C3R :
                type == CV_16UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_16u_C4R :
                type == CV_32FC1 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_32f_C1R :
                type == CV_32FC3 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_32f_C3R :
                type == CV_32FC4 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_32f_C4R :
                0;
            }
            else
            {
                ippFunc =
                type == CV_8UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffine_8u_C1R :
                type == CV_8UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffine_8u_C3R :
                type == CV_8UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffine_8u_C4R :
                type == CV_16UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffine_16u_C1R :
                type == CV_16UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffine_16u_C3R :
                type == CV_16UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffine_16u_C4R :
                type == CV_32FC1 ? (ippiWarpAffineBackFunc)ippiWarpAffine_32f_C1R :
                type == CV_32FC3 ? (ippiWarpAffineBackFunc)ippiWarpAffine_32f_C3R :
                type == CV_32FC4 ? (ippiWarpAffineBackFunc)ippiWarpAffine_32f_C4R :
                0;
            }
            int mode =
            interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR :
            interpolation == INTER_NEAREST ? IPPI_INTER_NN :
            interpolation == INTER_CUBIC ? IPPI_INTER_CUBIC :
            0;
            CV_Assert(mode && ippFunc);

            double coeffs[2][3];
            for( int i = 0; i < 2; i++ )
                for( int j = 0; j < 3; j++ )
                    coeffs[i][j] = matM.at<double>(i, j);

            bool ok;
            Range range(0, dst.rows);
            IPPWarpAffineInvoker invoker(src, dst, coeffs, mode, borderType, borderValue, ippFunc, &ok);
            parallel_for_(range, invoker, dst.total()/(double)(1<<16));
            if( ok )
            {
                CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                return;
            }
            setIppErrorStatus();
        }
    }
#endif

    hal::warpAffine(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows,
                    M, interpolation, borderType, borderValue.val, hint);
}


namespace cv
{
#if CV_SIMD128_64F
void WarpPerspectiveLine_ProcessNN_CV_SIMD(const double *M, short* xy, double X0, double Y0, double W0, int bw)
{
    const v_float64x2 v_M0 = v_setall_f64(M[0]);
    const v_float64x2 v_M3 = v_setall_f64(M[3]);
    const v_float64x2 v_M6 = v_setall_f64(M[6]);
    const v_float64x2 v_intmax = v_setall_f64((double)INT_MAX);
    const v_float64x2 v_intmin = v_setall_f64((double)INT_MIN);
    const v_float64x2 v_2 = v_setall_f64(2.0);
    const v_float64x2 v_zero = v_setzero_f64();
    const v_float64x2 v_1 = v_setall_f64(1.0);

    int x1 = 0;
    v_float64x2 v_X0d = v_setall_f64(X0);
    v_float64x2 v_Y0d = v_setall_f64(Y0);
    v_float64x2 v_W0 = v_setall_f64(W0);
    v_float64x2 v_x1(0.0, 1.0);

    for (; x1 <= bw - 16; x1 += 16)
    {
        // 0-3
        v_int32x4 v_X0, v_Y0;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X0 = v_round(v_fX0, v_fX1);
            v_Y0 = v_round(v_fY0, v_fY1);
        }

        // 4-7
        v_int32x4 v_X1, v_Y1;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X1 = v_round(v_fX0, v_fX1);
            v_Y1 = v_round(v_fY0, v_fY1);
        }

        // 8-11
        v_int32x4 v_X2, v_Y2;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X2 = v_round(v_fX0, v_fX1);
            v_Y2 = v_round(v_fY0, v_fY1);
        }

        // 12-15
        v_int32x4 v_X3, v_Y3;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_1, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X3 = v_round(v_fX0, v_fX1);
            v_Y3 = v_round(v_fY0, v_fY1);
        }

        // convert to 16s
        v_X0 = v_reinterpret_as_s32(v_pack(v_X0, v_X1));
        v_X1 = v_reinterpret_as_s32(v_pack(v_X2, v_X3));
        v_Y0 = v_reinterpret_as_s32(v_pack(v_Y0, v_Y1));
        v_Y1 = v_reinterpret_as_s32(v_pack(v_Y2, v_Y3));

        v_store_interleave(xy + x1 * 2, (v_reinterpret_as_s16)(v_X0), (v_reinterpret_as_s16)(v_Y0));
        v_store_interleave(xy + x1 * 2 + 16, (v_reinterpret_as_s16)(v_X1), (v_reinterpret_as_s16)(v_Y1));
    }

    for( ; x1 < bw; x1++ )
    {
        double W = W0 + M[6]*x1;
        W = W ? 1./W : 0;
        double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
        double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
        int X = saturate_cast<int>(fX);
        int Y = saturate_cast<int>(fY);

        xy[x1*2] = saturate_cast<short>(X);
        xy[x1*2+1] = saturate_cast<short>(Y);
    }
}

void WarpPerspectiveLine_Process_CV_SIMD(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw)
{
    const v_float64x2 v_M0 = v_setall_f64(M[0]);
    const v_float64x2 v_M3 = v_setall_f64(M[3]);
    const v_float64x2 v_M6 = v_setall_f64(M[6]);
    const v_float64x2 v_intmax = v_setall_f64((double)INT_MAX);
    const v_float64x2 v_intmin = v_setall_f64((double)INT_MIN);
    const v_float64x2 v_2 = v_setall_f64(2.0);
    const v_float64x2 v_zero = v_setzero_f64();
    const v_float64x2 v_its = v_setall_f64((double)INTER_TAB_SIZE);
    const v_int32x4 v_itsi1 = v_setall_s32(INTER_TAB_SIZE - 1);

    int x1 = 0;

    v_float64x2 v_X0d = v_setall_f64(X0);
    v_float64x2 v_Y0d = v_setall_f64(Y0);
    v_float64x2 v_W0 = v_setall_f64(W0);
    v_float64x2 v_x1(0.0, 1.0);

    for (; x1 <= bw - 16; x1 += 16)
    {
        // 0-3
        v_int32x4 v_X0, v_Y0;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X0 = v_round(v_fX0, v_fX1);
            v_Y0 = v_round(v_fY0, v_fY1);
        }

        // 4-7
        v_int32x4 v_X1, v_Y1;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X1 = v_round(v_fX0, v_fX1);
            v_Y1 = v_round(v_fY0, v_fY1);
        }

        // 8-11
        v_int32x4 v_X2, v_Y2;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X2 = v_round(v_fX0, v_fX1);
            v_Y2 = v_round(v_fY0, v_fY1);
        }

        // 12-15
        v_int32x4 v_X3, v_Y3;
        {
            v_float64x2 v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY0 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_W = v_muladd(v_M6, v_x1, v_W0);
            v_W = v_select(v_ne(v_W, v_zero), v_div(v_its, v_W), v_zero);
            v_float64x2 v_fX1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M0, v_x1, v_X0d), v_W)));
            v_float64x2 v_fY1 = v_max(v_intmin, v_min(v_intmax, v_mul(v_muladd(v_M3, v_x1, v_Y0d), v_W)));
            v_x1 = v_add(v_x1, v_2);

            v_X3 = v_round(v_fX0, v_fX1);
            v_Y3 = v_round(v_fY0, v_fY1);
        }

        // store alpha
        v_int32x4 v_alpha0 = v_add(v_shl<INTER_BITS>(v_and(v_Y0, v_itsi1)), v_and(v_X0, v_itsi1));
        v_int32x4 v_alpha1 = v_add(v_shl<INTER_BITS>(v_and(v_Y1, v_itsi1)), v_and(v_X1, v_itsi1));
        v_store((alpha + x1), v_pack(v_alpha0, v_alpha1));

        v_alpha0 = v_add(v_shl<INTER_BITS>(v_and(v_Y2, v_itsi1)), v_and(v_X2, v_itsi1));
        v_alpha1 = v_add(v_shl<INTER_BITS>(v_and(v_Y3, v_itsi1)), v_and(v_X3, v_itsi1));
        v_store((alpha + x1 + 8), v_pack(v_alpha0, v_alpha1));

        // convert to 16s
        v_X0 = v_reinterpret_as_s32(v_pack(v_shr<INTER_BITS>(v_X0), v_shr<INTER_BITS>(v_X1)));
        v_X1 = v_reinterpret_as_s32(v_pack(v_shr<INTER_BITS>(v_X2), v_shr<INTER_BITS>(v_X3)));
        v_Y0 = v_reinterpret_as_s32(v_pack(v_shr<INTER_BITS>(v_Y0), v_shr<INTER_BITS>(v_Y1)));
        v_Y1 = v_reinterpret_as_s32(v_pack(v_shr<INTER_BITS>(v_Y2), v_shr<INTER_BITS>(v_Y3)));

        v_store_interleave(xy + x1 * 2, (v_reinterpret_as_s16)(v_X0), (v_reinterpret_as_s16)(v_Y0));
        v_store_interleave(xy + x1 * 2 + 16, (v_reinterpret_as_s16)(v_X1), (v_reinterpret_as_s16)(v_Y1));
    }

    for( ; x1 < bw; x1++ )
    {
        double W = W0 + M[6]*x1;
        W = W ? static_cast<double>(INTER_TAB_SIZE)/W : 0;
        double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
        double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
        int X = saturate_cast<int>(fX);
        int Y = saturate_cast<int>(fY);

        xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
        xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
        alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
                            (X & (INTER_TAB_SIZE-1)));
    }
}
#endif

class WarpPerspectiveInvoker :
    public ParallelLoopBody
{
public:
    WarpPerspectiveInvoker(const Mat &_src, Mat &_dst, const double *_M, int _interpolation,
                           int _borderType, const Scalar &_borderValue) :
        ParallelLoopBody(), src(_src), dst(_dst), M(_M), interpolation(_interpolation),
        borderType(_borderType), borderValue(_borderValue)
    {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int BLOCK_SZ = 32;
        short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
        int x, y, y1, width = dst.cols, height = dst.rows;

        int bh0 = std::min(BLOCK_SZ/2, height);
        int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
        bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

        for( y = range.start; y < range.end; y += bh0 )
        {
            for( x = 0; x < width; x += bw0 )
            {
                int bw = std::min( bw0, width - x);
                int bh = std::min( bh0, range.end - y); // height

                Mat _XY(bh, bw, CV_16SC2, XY);
                Mat dpart(dst, Rect(x, y, bw, bh));

                for( y1 = 0; y1 < bh; y1++ )
                {
                    short* xy = XY + y1*bw*2;
                    double X0 = M[0]*x + M[1]*(y + y1) + M[2];
                    double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
                    double W0 = M[6]*x + M[7]*(y + y1) + M[8];

                    if( interpolation == INTER_NEAREST )
                        hal::warpPerspectiveBlocklineNN(M, xy, X0, Y0, W0, bw);
                    else
                        hal::warpPerspectiveBlockline(M, xy, A + y1*bw, X0, Y0, W0, bw);
                }

                if( interpolation == INTER_NEAREST )
                    remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
                else
                {
                    Mat _matA(bh, bw, CV_16U, A);
                    remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
                }
            }
        }
    }

private:
    Mat src;
    Mat dst;
    const double* M;
    int interpolation, borderType;
    Scalar borderValue;
};

#if defined (HAVE_IPP) && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_WARPPERSPECTIVE
typedef IppStatus (CV_STDCALL* ippiWarpPerspectiveFunc)(const void*, IppiSize, int, IppiRect, void *, int, IppiRect, double [3][3], int);

class IPPWarpPerspectiveInvoker :
    public ParallelLoopBody
{
public:
    IPPWarpPerspectiveInvoker(Mat &_src, Mat &_dst, double (&_coeffs)[3][3], int &_interpolation,
                              int &_borderType, const Scalar &_borderValue, ippiWarpPerspectiveFunc _func, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), mode(_interpolation), coeffs(_coeffs),
        borderType(_borderType), borderValue(_borderValue), func(_func), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        IppiSize srcsize = {src.cols, src.rows};
        IppiRect srcroi = {0, 0, src.cols, src.rows};
        IppiRect dstroi = {0, range.start, dst.cols, range.end - range.start};
        int cnn = src.channels();

        if( borderType == BORDER_CONSTANT )
        {
            IppiSize setSize = {dst.cols, range.end - range.start};
            void *dataPointer = dst.ptr(range.start);
            if( !IPPSet( borderValue, dataPointer, (int)dst.step[0], setSize, cnn, src.depth() ) )
            {
                *ok = false;
                return;
            }
        }

        IppStatus status = CV_INSTRUMENT_FUN_IPP(func,(src.ptr();, srcsize, (int)src.step[0], srcroi, dst.ptr(), (int)dst.step[0], dstroi, coeffs, mode));
        if (status != ippStsNoErr)
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        }
    }
private:
    Mat &src;
    Mat &dst;
    int mode;
    double (&coeffs)[3][3];
    int borderType;
    const Scalar borderValue;
    ippiWarpPerspectiveFunc func;
    bool *ok;

    const IPPWarpPerspectiveInvoker& operator= (const IPPWarpPerspectiveInvoker&);
};
#endif

namespace hal {

static void warpPerspective(int src_type,
                    const uchar * src_data, size_t src_step, int src_width, int src_height,
                    uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                    const double M[9], int interpolation, int borderType, const double borderValue[4], AlgorithmHint hint)
{
    CALL_HAL(warpPerspective, cv_hal_warpPerspective, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue);

    if (interpolation == INTER_NEAREST) {
        switch (src_type) {
            case CV_8UC1: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_8UC1, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_8UC3: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_8UC3, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_8UC4: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_8UC4, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC1: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_16UC1, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC3: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_16UC3, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC4: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_16UC4, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC1: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_32FC1, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC3: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_32FC3, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC4: {
                CV_CPU_DISPATCH(warpPerspectiveNearestInvoker_32FC4, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
        }
    }

    if (interpolation == INTER_LINEAR) {
        switch (src_type) {
            case CV_8UC1: {
                if (hint == cv::ALGO_HINT_APPROX) {
                    CV_CPU_DISPATCH(warpPerspectiveLinearApproxInvoker_8UC1, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                } else {
                    CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_8UC1, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
            }
            case CV_8UC3: {
                if (hint == cv::ALGO_HINT_APPROX) {
                    CV_CPU_DISPATCH(warpPerspectiveLinearApproxInvoker_8UC3, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                } else {
                    CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_8UC3, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
            }
            case CV_8UC4: {
                if (hint == cv::ALGO_HINT_APPROX) {
                    CV_CPU_DISPATCH(warpPerspectiveLinearApproxInvoker_8UC4, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                } else {
                    CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_8UC4, (src_data, src_step, src_height, src_width, dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                    break;
                }
            }
            case CV_16UC1: {
                CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_16UC1, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC3: {
                CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_16UC3, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_16UC4: {
                CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_16UC4, ((const uint16_t*)src_data, src_step, src_height, src_width, (uint16_t*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC1: {
                CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_32FC1, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC3: {
                CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_32FC3, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            case CV_32FC4: {
                CV_CPU_DISPATCH(warpPerspectiveLinearInvoker_32FC4, ((const float*)src_data, src_step, src_height, src_width, (float*)dst_data, dst_step, dst_height, dst_width, M, borderType, borderValue), CV_CPU_DISPATCH_MODES_ALL);
                break;
            }
            // no default
        }
    }

    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(Size(dst_width, dst_height), src_type, dst_data, dst_step);

    Range range(0, dst.rows);
    WarpPerspectiveInvoker invoker(src, dst, M, interpolation, borderType, Scalar(borderValue[0], borderValue[1], borderValue[2], borderValue[3]));
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

void warpPerspectiveBlocklineNN(const double *M, short* xy, double X0, double Y0, double W0, int bw)
{
    CALL_HAL(warpPerspectiveBlocklineNN, cv_hal_warpPerspectiveBlocklineNN, M, xy, X0, Y0, W0, bw);

    #if CV_TRY_SSE4_1
    Ptr<opt_SSE4_1::WarpPerspectiveLine_SSE4> pwarp_impl_sse4;
    if(CV_CPU_HAS_SUPPORT_SSE4_1)
        pwarp_impl_sse4 = opt_SSE4_1::WarpPerspectiveLine_SSE4::getImpl(M);

    if (pwarp_impl_sse4)
        pwarp_impl_sse4->processNN(M, xy, X0, Y0, W0, bw);
    else
    #endif
    {
        #if CV_SIMD128_64F
        WarpPerspectiveLine_ProcessNN_CV_SIMD(M, xy, X0, Y0, W0, bw);
        #else
        for( int x1 = 0; x1 < bw; x1++ )
        {
            double W = W0 + M[6]*x1;
            W = W ? 1./W : 0;
            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
            int X = saturate_cast<int>(fX);
            int Y = saturate_cast<int>(fY);

            xy[x1*2] = saturate_cast<short>(X);
            xy[x1*2+1] = saturate_cast<short>(Y);
        }
        #endif
    }
}

void warpPerspectiveBlockline(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw)
{
    CALL_HAL(warpPerspectiveBlockline, cv_hal_warpPerspectiveBlockline, M, xy, alpha, X0, Y0, W0, bw);

    #if CV_TRY_SSE4_1
    Ptr<opt_SSE4_1::WarpPerspectiveLine_SSE4> pwarp_impl_sse4;
    if(CV_CPU_HAS_SUPPORT_SSE4_1)
        pwarp_impl_sse4 = opt_SSE4_1::WarpPerspectiveLine_SSE4::getImpl(M);

    if (pwarp_impl_sse4)
        pwarp_impl_sse4->process(M, xy, alpha, X0, Y0, W0, bw);
    else
    #endif
    {
        #if CV_SIMD128_64F
        WarpPerspectiveLine_Process_CV_SIMD(M, xy, alpha, X0, Y0, W0, bw);
        #else
        for( int x1 = 0; x1 < bw; x1++ )
        {
            double W = W0 + M[6]*x1;
            W = W ? INTER_TAB_SIZE/W : 0;
            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
            int X = saturate_cast<int>(fX);
            int Y = saturate_cast<int>(fY);

            xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
            xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
            alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
                                (X & (INTER_TAB_SIZE-1)));
        }
        #endif
    }
}

} // hal::
} // cv::

void cv::warpPerspective( InputArray _src, OutputArray _dst, InputArray _M0,
                          Size dsize, int flags, int borderType, const Scalar& borderValue,
                          AlgorithmHint hint )
{
    CV_INSTRUMENT_REGION();

    if (hint == cv::ALGO_HINT_DEFAULT)
        hint = cv::getDefaultAlgorithmHint();

    CV_Assert( _src.total() > 0 );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat() &&
               _src.cols() <= SHRT_MAX && _src.rows() <= SHRT_MAX,
               ocl_warpTransform_cols4(_src, _dst, _M0, dsize, flags, borderType, borderValue,
                                       OCL_OP_PERSPECTIVE))

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_warpTransform(_src, _dst, _M0, dsize, flags, borderType, borderValue,
                              OCL_OP_PERSPECTIVE))

    Mat src = _src.getMat(), M0 = _M0.getMat();
    _dst.create( dsize.empty() ? src.size() : dsize, src.type() );
    Mat dst = _dst.getMat();

    if( dst.data == src.data )
        src = src.clone();

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3 );
    M0.convertTo(matM, matM.type());

#if defined (HAVE_IPP) && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_WARPPERSPECTIVE
    CV_IPP_CHECK()
    {
        int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
        if( (depth == CV_8U || depth == CV_16U || depth == CV_32F) &&
           (cn == 1 || cn == 3 || cn == 4) &&
           ( borderType == cv::BORDER_TRANSPARENT || borderType == cv::BORDER_CONSTANT ) &&
           (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC))
        {
            ippiWarpPerspectiveFunc ippFunc = 0;
            if ((flags & WARP_INVERSE_MAP) != 0)
            {
                ippFunc = type == CV_8UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_8u_C1R :
                type == CV_8UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_8u_C3R :
                type == CV_8UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_8u_C4R :
                type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_16u_C1R :
                type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_16u_C3R :
                type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_16u_C4R :
                type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_32f_C1R :
                type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_32f_C3R :
                type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_32f_C4R : 0;
            }
            else
            {
                ippFunc = type == CV_8UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_8u_C1R :
                type == CV_8UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_8u_C3R :
                type == CV_8UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_8u_C4R :
                type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_16u_C1R :
                type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_16u_C3R :
                type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_16u_C4R :
                type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_32f_C1R :
                type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_32f_C3R :
                type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_32f_C4R : 0;
            }
            int mode =
            interpolation == INTER_NEAREST ? IPPI_INTER_NN :
            interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR :
            interpolation == INTER_CUBIC ? IPPI_INTER_CUBIC : 0;
            CV_Assert(mode && ippFunc);

            double coeffs[3][3];
            for( int i = 0; i < 3; i++ )
                for( int j = 0; j < 3; j++ )
                    coeffs[i][j] = matM.at<double>(i, j);

            bool ok;
            Range range(0, dst.rows);
            IPPWarpPerspectiveInvoker invoker(src, dst, coeffs, mode, borderType, borderValue, ippFunc, &ok);
            parallel_for_(range, invoker, dst.total()/(double)(1<<16));
            if( ok )
            {
                CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                return;
            }
            setIppErrorStatus();
        }
    }
#endif

    if( !(flags & WARP_INVERSE_MAP) )
        invert(matM, matM);

    hal::warpPerspective(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows,
                        matM.ptr<double>(), interpolation, borderType, borderValue.val, hint);
}


cv::Matx23d cv::getRotationMatrix2D_(Point2f center, double angle, double scale)
{
    CV_INSTRUMENT_REGION();

    angle *= CV_PI/180;
    double alpha = std::cos(angle)*scale;
    double beta = std::sin(angle)*scale;

    Matx23d M(
        alpha, beta, (1-alpha)*center.x - beta*center.y,
        -beta, alpha, beta*center.x + (1-alpha)*center.y
    );
    return M;
}

/* Calculates coefficients of perspective transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   cij - matrix coefficients, c22 = 1
 */
cv::Mat cv::getPerspectiveTransform(const Point2f src[], const Point2f dst[], int solveMethod)
{
    CV_INSTRUMENT_REGION();

    Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
    double a[8][8], b[8];
    Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for( int i = 0; i < 4; ++i )
    {
        a[i][0] = a[i+4][3] = src[i].x;
        a[i][1] = a[i+4][4] = src[i].y;
        a[i][2] = a[i+4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] =
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
        a[i][6] = -src[i].x*dst[i].x;
        a[i][7] = -src[i].y*dst[i].x;
        a[i+4][6] = -src[i].x*dst[i].y;
        a[i+4][7] = -src[i].y*dst[i].y;
        b[i] = dst[i].x;
        b[i+4] = dst[i].y;
    }

    solve(A, B, X, solveMethod);
    M.ptr<double>()[8] = 1.;

    return M;
}

/* Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 | |c01| |u1|
 * | x2 y2  1  0  0  0 | |c02| |u2|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * \  0  0  0 x2 y2  1 / |c12| |v2|
 *
 * where:
 *   cij - matrix coefficients
 */

cv::Mat cv::getAffineTransform( const Point2f src[], const Point2f dst[] )
{
    Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.ptr());
    double a[6*6], b[6];
    Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);

    for( int i = 0; i < 3; i++ )
    {
        int j = i*12;
        int k = i*12+6;
        a[j] = a[k+3] = src[i].x;
        a[j+1] = a[k+4] = src[i].y;
        a[j+2] = a[k+5] = 1;
        a[j+3] = a[j+4] = a[j+5] = 0;
        a[k] = a[k+1] = a[k+2] = 0;
        b[i*2] = dst[i].x;
        b[i*2+1] = dst[i].y;
    }

    solve( A, B, X );
    return M;
}

void cv::invertAffineTransform(InputArray _matM, OutputArray __iM)
{
    Mat matM = _matM.getMat();
    CV_Assert(matM.rows == 2 && matM.cols == 3);
    __iM.create(2, 3, matM.type());
    Mat _iM = __iM.getMat();

    if( matM.type() == CV_32F )
    {
        const softfloat* M = matM.ptr<softfloat>();
        softfloat* iM = _iM.ptr<softfloat>();
        int step = (int)(matM.step/sizeof(M[0])), istep = (int)(_iM.step/sizeof(iM[0]));

        softdouble D = M[0]*M[step+1] - M[1]*M[step];
        D = D != 0. ? softdouble(1.)/D : softdouble(0.);
        softdouble A11 = M[step+1]*D, A22 = M[0]*D, A12 = -M[1]*D, A21 = -M[step]*D;
        softdouble b1 = -A11*M[2] - A12*M[step+2];
        softdouble b2 = -A21*M[2] - A22*M[step+2];

        iM[0] = A11; iM[1] = A12; iM[2] = b1;
        iM[istep] = A21; iM[istep+1] = A22; iM[istep+2] = b2;
    }
    else if( matM.type() == CV_64F )
    {
        const softdouble* M = matM.ptr<softdouble>();
        softdouble* iM = _iM.ptr<softdouble>();
        int step = (int)(matM.step/sizeof(M[0])), istep = (int)(_iM.step/sizeof(iM[0]));

        softdouble D = M[0]*M[step+1] - M[1]*M[step];
        D = D != 0. ? softdouble(1.)/D : softdouble(0.);
        softdouble A11 = M[step+1]*D, A22 = M[0]*D, A12 = -M[1]*D, A21 = -M[step]*D;
        softdouble b1 = -A11*M[2] - A12*M[step+2];
        softdouble b2 = -A21*M[2] - A22*M[step+2];

        iM[0] = A11; iM[1] = A12; iM[2] = b1;
        iM[istep] = A21; iM[istep+1] = A22; iM[istep+2] = b2;
    }
    else
        CV_Error( cv::Error::StsUnsupportedFormat, "" );
}

cv::Mat cv::getPerspectiveTransform(InputArray _src, InputArray _dst, int solveMethod)
{
    Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert(src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4);
    return getPerspectiveTransform((const Point2f*)src.data, (const Point2f*)dst.data, solveMethod);
}

cv::Mat cv::getAffineTransform(InputArray _src, InputArray _dst)
{
    Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert(src.checkVector(2, CV_32F) == 3 && dst.checkVector(2, CV_32F) == 3);
    return getAffineTransform((const Point2f*)src.data, (const Point2f*)dst.data);
}

/****************************************************************************************
PkLab.net 2018 based on cv::linearPolar from OpenCV by J.L. Blanco, Apr 2009
****************************************************************************************/
void cv::warpPolar(InputArray _src, OutputArray _dst, Size dsize,
                   Point2f center, double maxRadius, int flags)
{
    // if dest size is empty given than calculate using proportional setting
    // thus we calculate needed angles to keep same area as bounding circle
    if ((dsize.width <= 0) && (dsize.height <= 0))
    {
        dsize.width = cvRound(maxRadius);
        dsize.height = cvRound(maxRadius * CV_PI);
    }
    else if (dsize.height <= 0)
    {
        dsize.height = cvRound(dsize.width * CV_PI);
    }

    Mat mapx, mapy;
    mapx.create(dsize, CV_32F);
    mapy.create(dsize, CV_32F);
    bool semiLog = (flags & WARP_POLAR_LOG) != 0;

    if (!(flags & cv::WARP_INVERSE_MAP))
    {
        CV_Assert(!dsize.empty());
        double Kangle = CV_2PI / dsize.height;
        int phi, rho;

        // precalculate scaled rho
        Mat rhos = Mat(1, dsize.width, CV_32F);
        float* bufRhos = (float*)(rhos.data);
        if (semiLog)
        {
            double Kmag = std::log(maxRadius) / dsize.width;
            for (rho = 0; rho < dsize.width; rho++)
                bufRhos[rho] = (float)(std::exp(rho * Kmag) - 1.0);

        }
        else
        {
            double Kmag = maxRadius / dsize.width;
            for (rho = 0; rho < dsize.width; rho++)
                bufRhos[rho] = (float)(rho * Kmag);
        }

        for (phi = 0; phi < dsize.height; phi++)
        {
            double KKy = Kangle * phi;
            double cp = std::cos(KKy);
            double sp = std::sin(KKy);
            float* mx = (float*)(mapx.data + phi*mapx.step);
            float* my = (float*)(mapy.data + phi*mapy.step);

            for (rho = 0; rho < dsize.width; rho++)
            {
                double x = bufRhos[rho] * cp + center.x;
                double y = bufRhos[rho] * sp + center.y;

                mx[rho] = (float)x;
                my[rho] = (float)y;
            }
        }
        remap(_src, _dst, mapx, mapy, flags & cv::INTER_MAX, (flags & cv::WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT);
    }
    else
    {
        const int ANGLE_BORDER = 1;
        cv::copyMakeBorder(_src, _dst, ANGLE_BORDER, ANGLE_BORDER, 0, 0, BORDER_WRAP);
        Mat src = _dst.getMat();
        Size ssize = _dst.size();
        ssize.height -= 2 * ANGLE_BORDER;
        CV_Assert(!ssize.empty());
        const double Kangle = CV_2PI / ssize.height;
        double Kmag;
        if (semiLog)
            Kmag = std::log(maxRadius) / ssize.width;
        else
            Kmag = maxRadius / ssize.width;

        int x, y;
        Mat bufx, bufy, bufp, bufa;

        bufx = Mat(1, dsize.width, CV_32F);
        bufy = Mat(1, dsize.width, CV_32F);
        bufp = Mat(1, dsize.width, CV_32F);
        bufa = Mat(1, dsize.width, CV_32F);

        for (x = 0; x < dsize.width; x++)
            bufx.at<float>(0, x) = (float)x - center.x;

        for (y = 0; y < dsize.height; y++)
        {
            float* mx = (float*)(mapx.data + y*mapx.step);
            float* my = (float*)(mapy.data + y*mapy.step);

            for (x = 0; x < dsize.width; x++)
                bufy.at<float>(0, x) = (float)y - center.y;

            cartToPolar(bufx, bufy, bufp, bufa, 0);

            if (semiLog)
            {
                bufp += 1.f;
                log(bufp, bufp);
            }

            for (x = 0; x < dsize.width; x++)
            {
                double rho = bufp.at<float>(0, x) / Kmag;
                double phi = bufa.at<float>(0, x) / Kangle;
                mx[x] = (float)rho;
                my[x] = (float)phi + ANGLE_BORDER;
            }
        }
        remap(src, _dst, mapx, mapy, flags & cv::INTER_MAX,
              (flags & cv::WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT);
    }
}

void cv::linearPolar( InputArray _src, OutputArray _dst,
                      Point2f center, double maxRadius, int flags )
{
    warpPolar(_src, _dst, _src.size(), center, maxRadius, flags & ~WARP_POLAR_LOG);
}

void cv::logPolar( InputArray _src, OutputArray _dst,
                   Point2f center, double maxRadius, int flags )
{
    Size ssize = _src.size();
    double M = maxRadius > 0 ? std::exp(ssize.width / maxRadius) : 1;
    warpPolar(_src, _dst, ssize, center, M, flags | WARP_POLAR_LOG);
}

/* End of file. */
