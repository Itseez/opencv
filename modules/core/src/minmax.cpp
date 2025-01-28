// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "opencv2/core/openvx/ovx_defs.hpp"
#include "stat.hpp"
#include "opencv2/core/detail/dispatch_helper.impl.hpp"

#include <algorithm>

#ifndef OPENCV_IPP_MINMAX
#undef HAVE_IPP
#undef CV_IPP_RUN_FAST
#define CV_IPP_RUN_FAST(f, ...)
#undef CV_IPP_RUN
#define CV_IPP_RUN(c, f, ...)
#endif // OPENCV_IPP_MINMAX

#define IPP_DISABLE_MINMAXIDX_MANY_ROWS 1  // see Core_MinMaxIdx.rows_overflow test

/****************************************************************************************\
*                                       minMaxLoc                                        *
\****************************************************************************************/

namespace cv
{

template<typename T, typename WT> static void
minMaxIdx_( const T* src, const uchar* mask, WT* _minVal, WT* _maxVal,
            size_t* _minIdx, size_t* _maxIdx, int len, size_t startIdx )
{
    WT minVal = *_minVal, maxVal = *_maxVal;
    size_t minIdx = *_minIdx, maxIdx = *_maxIdx;

    if( !mask )
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }
    else
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( mask[i] && val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( mask[i] && val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }

    *_minIdx = minIdx;
    *_maxIdx = maxIdx;
    *_minVal = minVal;
    *_maxVal = maxVal;
}

#if CV_SIMD128
template<typename T, typename WT> CV_ALWAYS_INLINE void
minMaxIdx_init( const T* src, const uchar* mask, WT* minval, WT* maxval,
                size_t* minidx, size_t* maxidx, WT &minVal, WT &maxVal,
                size_t &minIdx, size_t &maxIdx, const WT minInit, const WT maxInit,
                const int nlanes, int len, size_t startidx, int &j, int &len0 )
{
    len0 = len & -nlanes;
    j = 0;

    minVal = *minval, maxVal = *maxval;
    minIdx = *minidx, maxIdx = *maxidx;

    // To handle start values out of range
    if ( minVal < minInit || maxVal < minInit || minVal > maxInit || maxVal > maxInit )
    {
        uchar done = 0x00;

        for ( ; (j < len) && (done != 0x03); j++ )
        {
            if ( !mask || mask[j] ) {
                T val = src[j];
                if ( val < minVal )
                {
                    minVal = val;
                    minIdx = startidx + j;
                    done |= 0x01;
                }
                if ( val > maxVal )
                {
                    maxVal = val;
                    maxIdx = startidx + j;
                    done |= 0x02;
                }
            }
        }

        len0 = j + ((len - j) & -nlanes);
    }
}

#if CV_SIMD128_64F
CV_ALWAYS_INLINE double v_reduce_min(const v_float64x2& a)
{
    double CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return std::min(idx[0], idx[1]);
}

CV_ALWAYS_INLINE double v_reduce_max(const v_float64x2& a)
{
    double CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return std::max(idx[0], idx[1]);
}

CV_ALWAYS_INLINE uint64_t v_reduce_min(const v_uint64x2& a)
{
    uint64_t CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return std::min(idx[0], idx[1]);
}

CV_ALWAYS_INLINE v_uint64x2 v_select(const v_uint64x2& mask, const v_uint64x2& a, const v_uint64x2& b)
{
    return v_xor(b, v_and(v_xor(a, b), mask));
}
#endif

#define MINMAXIDX_REDUCE(suffix, suffix2, maxLimit, IR) \
template<typename T, typename VT, typename IT> CV_ALWAYS_INLINE void \
minMaxIdx_reduce_##suffix( VT &valMin, VT &valMax, IT &idxMin, IT &idxMax, IT &none, \
                  T &minVal, T &maxVal, size_t &minIdx, size_t &maxIdx, \
                  size_t delta ) \
{ \
    if ( v_check_any(v_ne(idxMin, none)) ) \
    { \
        minVal = v_reduce_min(valMin); \
        minIdx = (size_t)v_reduce_min(v_select(v_reinterpret_as_##suffix2(v_eq(v_setall_##suffix((IR)minVal), valMin)), \
                     idxMin, v_setall_##suffix2(maxLimit))) + delta; \
    } \
    if ( v_check_any(v_ne(idxMax, none)) ) \
    { \
        maxVal = v_reduce_max(valMax); \
        maxIdx = (size_t)v_reduce_min(v_select(v_reinterpret_as_##suffix2(v_eq(v_setall_##suffix((IR)maxVal), valMax)), \
                     idxMax, v_setall_##suffix2(maxLimit))) + delta; \
    } \
}

MINMAXIDX_REDUCE(u8, u8, UCHAR_MAX, uchar)
MINMAXIDX_REDUCE(s8, u8, UCHAR_MAX, uchar)
MINMAXIDX_REDUCE(u16, u16, USHRT_MAX, ushort)
MINMAXIDX_REDUCE(s16, u16, USHRT_MAX, ushort)
MINMAXIDX_REDUCE(s32, u32, UINT_MAX, uint)
MINMAXIDX_REDUCE(f32, u32, (1 << 23) - 1, float)
#if CV_SIMD128_64F
MINMAXIDX_REDUCE(f64, u64, UINT_MAX, double)
#endif

template<typename T, typename WT> CV_ALWAYS_INLINE void
minMaxIdx_finish( const T* src, const uchar* mask, WT* minval, WT* maxval,
                  size_t* minidx, size_t* maxidx, WT minVal, WT maxVal,
                  size_t minIdx, size_t maxIdx, int len, size_t startidx,
                  int j )
{
    for ( ; j < len ; j++ )
    {
        if ( !mask || mask[j] )
        {
            T val = src[j];
            if ( val < minVal )
            {
                minVal = val;
                minIdx = startidx + j;
            }
            if ( val > maxVal )
            {
                maxVal = val;
                maxIdx = startidx + j;
            }
        }
    }

    *minidx = minIdx;
    *maxidx = maxIdx;
    *minval = minVal;
    *maxval = maxVal;
}
#endif

static void minMaxIdx_8u(const uchar* src, const uchar* mask, int* minval, int* maxval,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128
    if ( len >= VTraits<v_uint8x16>::vlanes() )
    {
        int j, len0;
        int minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        (int)0, (int)UCHAR_MAX, VTraits<v_uint8x16>::vlanes(), len, startidx, j, len0 );

        if ( j <= len0 - VTraits<v_uint8x16>::vlanes() )
        {
            v_uint8x16 inc = v_setall_u8((uchar)VTraits<v_uint8x16>::vlanes());
            v_uint8x16 none = v_reinterpret_as_u8(v_setall_s8(-1));
            v_uint8x16 idxStart(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

            do
            {
                v_uint8x16 valMin = v_setall_u8((uchar)minVal), valMax = v_setall_u8((uchar)maxVal);
                v_uint8x16 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 15 * VTraits<v_uint8x16>::vlanes()); k += VTraits<v_uint8x16>::vlanes() )
                    {
                        v_uint8x16 data = v_load(src + k);
                        v_uint8x16 cmpMin = (v_lt(data, valMin));
                        v_uint8x16 cmpMax = (v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 15 * VTraits<v_uint8x16>::vlanes()); k += VTraits<v_uint8x16>::vlanes() )
                    {
                        v_uint8x16 data = v_load(src + k);
                        v_uint8x16 maskVal = v_ne(v_load(mask + k), v_setzero_u8());
                        v_uint8x16 cmpMin = v_and(v_lt(data, valMin), maskVal);
                        v_uint8x16 cmpMax = v_and(v_gt(data, valMax), maskVal);
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(cmpMin, data, valMin);
                        valMax = v_select(cmpMax, data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_u8( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                     minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
#endif
}

static void minMaxIdx_8s(const schar* src, const uchar* mask, int* minval, int* maxval,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128
    if ( len >= VTraits<v_int8x16>::vlanes() )
    {
        int j, len0;
        int minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        (int)SCHAR_MIN, (int)SCHAR_MAX, VTraits<v_int8x16>::vlanes(), len, startidx, j, len0 );

        if ( j <= len0 - VTraits<v_int8x16>::vlanes() )
        {
            v_uint8x16 inc = v_setall_u8((uchar)VTraits<v_int8x16>::vlanes());
            v_uint8x16 none = v_reinterpret_as_u8(v_setall_s8(-1));
            v_uint8x16 idxStart(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

            do
            {
                v_int8x16 valMin = v_setall_s8((schar)minVal), valMax = v_setall_s8((schar)maxVal);
                v_uint8x16 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 15 * VTraits<v_int8x16>::vlanes()); k += VTraits<v_int8x16>::vlanes() )
                    {
                        v_int8x16 data = v_load(src + k);
                        v_uint8x16 cmpMin = v_reinterpret_as_u8(v_lt(data, valMin));
                        v_uint8x16 cmpMax = v_reinterpret_as_u8(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 15 * VTraits<v_int8x16>::vlanes()); k += VTraits<v_int8x16>::vlanes() )
                    {
                        v_int8x16 data = v_load(src + k);
                        v_uint8x16 maskVal = v_ne(v_load(mask + k), v_setzero_u8());
                        v_uint8x16 cmpMin = v_and(v_reinterpret_as_u8(v_lt(data, valMin)), maskVal);
                        v_uint8x16 cmpMax = v_and(v_reinterpret_as_u8(v_gt(data, valMax)), maskVal);
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_s8(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_s8(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_s8( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                     minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx );
#endif
}

static void minMaxIdx_16u(const ushort* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128
    if ( len >= VTraits<v_uint16x8>::vlanes() )
    {
        int j, len0;
        int minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        (int)0, (int)USHRT_MAX, VTraits<v_uint16x8>::vlanes(), len, startidx, j, len0 );

        if ( j <= len0 - VTraits<v_uint16x8>::vlanes() )
        {
            v_uint16x8 inc = v_setall_u16((uchar)VTraits<v_uint16x8>::vlanes());
            v_uint16x8 none = v_reinterpret_as_u16(v_setall_s16(-1));
            v_uint16x8 idxStart(0, 1, 2, 3, 4, 5, 6, 7);

            do
            {
                v_uint16x8 valMin = v_setall_u16((ushort)minVal), valMax = v_setall_u16((ushort)maxVal);
                v_uint16x8 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 8191 * VTraits<v_uint16x8>::vlanes()); k += VTraits<v_uint16x8>::vlanes() )
                    {
                        v_uint16x8 data = v_load(src + k);
                        v_uint16x8 cmpMin = (v_lt(data, valMin));
                        v_uint16x8 cmpMax = (v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 8191 * VTraits<v_uint16x8>::vlanes()); k += VTraits<v_uint16x8>::vlanes() )
                    {
                        v_uint16x8 data = v_load(src + k);
                        v_uint16x8 maskVal = v_ne(v_load_expand(mask + k), v_setzero_u16());
                        v_uint16x8 cmpMin = v_and(v_lt(data, valMin), maskVal);
                        v_uint16x8 cmpMax = v_and(v_gt(data, valMax), maskVal);
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(cmpMin, data, valMin);
                        valMax = v_select(cmpMax, data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_u16( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                      minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx );
#endif
}

static void minMaxIdx_16s(const short* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128
    if ( len >= VTraits<v_int16x8>::vlanes() )
    {
        int j, len0;
        int minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        (int)SHRT_MIN, (int)SHRT_MAX, VTraits<v_int16x8>::vlanes(), len, startidx, j, len0 );

        if ( j <= len0 - VTraits<v_int16x8>::vlanes() )
        {
            v_uint16x8 inc = v_setall_u16((uchar)VTraits<v_int16x8>::vlanes());
            v_uint16x8 none = v_reinterpret_as_u16(v_setall_s16(-1));
            v_uint16x8 idxStart(0, 1, 2, 3, 4, 5, 6, 7);

            do
            {
                v_int16x8 valMin = v_setall_s16((short)minVal), valMax = v_setall_s16((short)maxVal);
                v_uint16x8 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 8191 * VTraits<v_int16x8>::vlanes()); k += VTraits<v_int16x8>::vlanes() )
                    {
                        v_int16x8 data = v_load(src + k);
                        v_uint16x8 cmpMin = v_reinterpret_as_u16(v_lt(data, valMin));
                        v_uint16x8 cmpMax = v_reinterpret_as_u16(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 8191 * VTraits<v_int16x8>::vlanes()); k += VTraits<v_int16x8>::vlanes() )
                    {
                        v_int16x8 data = v_load(src + k);
                        v_uint16x8 maskVal = v_ne(v_load_expand(mask + k), v_setzero_u16());
                        v_uint16x8 cmpMin = v_and(v_reinterpret_as_u16(v_lt(data, valMin)), maskVal);
                        v_uint16x8 cmpMax = v_and(v_reinterpret_as_u16(v_gt(data, valMax)), maskVal);
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_s16(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_s16(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_s16( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                      minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx );
#endif
}

static void minMaxIdx_32s(const int* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128
    if ( len >= 2 * VTraits<v_int32x4>::vlanes() )
    {
        int j = 0, len0 = len & -(2 * VTraits<v_int32x4>::vlanes());
        int minVal = *minval, maxVal = *maxval;
        size_t minIdx = *minidx, maxIdx = *maxidx;

        {
            v_uint32x4 inc = v_setall_u32(VTraits<v_int32x4>::vlanes());
            v_uint32x4 none = v_reinterpret_as_u32(v_setall_s32(-1));
            v_uint32x4 idxStart(0, 1, 2, 3);

            do
            {
                v_int32x4 valMin = v_setall_s32(minVal), valMax = v_setall_s32(maxVal);
                v_uint32x4 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 32766 * 2 * VTraits<v_int32x4>::vlanes()); k += 2 * VTraits<v_int32x4>::vlanes() )
                    {
                        v_int32x4 data = v_load(src + k);
                        v_uint32x4 cmpMin = v_reinterpret_as_u32(v_lt(data, valMin));
                        v_uint32x4 cmpMax = v_reinterpret_as_u32(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + VTraits<v_int32x4>::vlanes());
                        cmpMin = v_reinterpret_as_u32(v_lt(data, valMin));
                        cmpMax = v_reinterpret_as_u32(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 32766 * 2 * VTraits<v_int32x4>::vlanes()); k += 2 * VTraits<v_int32x4>::vlanes() )
                    {
                        v_int32x4 data = v_load(src + k);
                        v_uint16x8 maskVal = v_ne(v_load_expand(mask + k), v_setzero_u16());
                        v_int32x4 maskVal1, maskVal2;
                        v_expand(v_reinterpret_as_s16(maskVal), maskVal1, maskVal2);
                        v_uint32x4 cmpMin = v_reinterpret_as_u32(v_and(v_lt(data, valMin), maskVal1));
                        v_uint32x4 cmpMax = v_reinterpret_as_u32(v_and(v_gt(data, valMax), maskVal1));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_s32(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_s32(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + VTraits<v_int32x4>::vlanes());
                        cmpMin = v_reinterpret_as_u32(v_and(v_lt(data, valMin), maskVal2));
                        cmpMax = v_reinterpret_as_u32(v_and(v_gt(data, valMax), maskVal2));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_s32(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_s32(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_s32( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                      minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx );
#endif
}

static void minMaxIdx_32f(const float* src, const uchar* mask, float* minval, float* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128
    if ( len >= 2 * VTraits<v_float32x4>::vlanes() )
    {
        int j, len0;
        float minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        FLT_MIN, FLT_MAX, 2 * VTraits<v_float32x4>::vlanes(), len, startidx, j, len0 );

        if ( j <= len0 - 2 * VTraits<v_float32x4>::vlanes() )
        {
            v_uint32x4 inc = v_setall_u32(VTraits<v_float32x4>::vlanes());
            v_uint32x4 none = v_reinterpret_as_u32(v_setall_s32(-1));
            v_uint32x4 idxStart(0, 1, 2, 3);

            do
            {
                v_float32x4 valMin = v_setall_f32(minVal), valMax = v_setall_f32(maxVal);
                v_uint32x4 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 32766 * 2 * VTraits<v_float32x4>::vlanes()); k += 2 * VTraits<v_float32x4>::vlanes() )
                    {
                        v_float32x4 data = v_load(src + k);
                        v_uint32x4 cmpMin = v_reinterpret_as_u32(v_lt(data, valMin));
                        v_uint32x4 cmpMax = v_reinterpret_as_u32(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + VTraits<v_float32x4>::vlanes());
                        cmpMin = v_reinterpret_as_u32(v_lt(data, valMin));
                        cmpMax = v_reinterpret_as_u32(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 32766 * 2 * VTraits<v_float32x4>::vlanes()); k += 2 * VTraits<v_float32x4>::vlanes() )
                    {
                        v_float32x4 data = v_load(src + k);
                        v_uint16x8 maskVal = v_ne(v_load_expand(mask + k), v_setzero_u16());
                        v_int32x4 maskVal1, maskVal2;
                        v_expand(v_reinterpret_as_s16(maskVal), maskVal1, maskVal2);
                        v_uint32x4 cmpMin = v_reinterpret_as_u32(v_and(v_reinterpret_as_s32(v_lt(data, valMin)), maskVal1));
                        v_uint32x4 cmpMax = v_reinterpret_as_u32(v_and(v_reinterpret_as_s32(v_gt(data, valMax)), maskVal1));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_f32(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_f32(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + VTraits<v_float32x4>::vlanes());
                        cmpMin = v_reinterpret_as_u32(v_and(v_reinterpret_as_s32(v_lt(data, valMin)), maskVal2));
                        cmpMax = v_reinterpret_as_u32(v_and(v_reinterpret_as_s32(v_gt(data, valMax)), maskVal2));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_f32(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_f32(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_f32( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                      minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx );
#endif
}

static void minMaxIdx_64f(const double* src, const uchar* mask, double* minval, double* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{
#if CV_SIMD128_64F
    if ( len >= 4 * VTraits<v_float64x2>::vlanes() )
    {
        int j, len0;
        double minVal, maxVal;
        size_t minIdx, maxIdx;

        minMaxIdx_init( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal, minIdx, maxIdx,
                        DBL_MIN, DBL_MAX, 4 * VTraits<v_float64x2>::vlanes(), len, startidx, j, len0 );

        if ( j <= len0 - 4 * VTraits<v_float64x2>::vlanes() )
        {
            v_uint64x2 inc = v_setall_u64(VTraits<v_float64x2>::vlanes());
            v_uint64x2 none = v_reinterpret_as_u64(v_setall_s64(-1));
            v_uint64x2 idxStart(0, 1);

            do
            {
                v_float64x2 valMin = v_setall_f64(minVal), valMax = v_setall_f64(maxVal);
                v_uint64x2 idx = idxStart, idxMin = none, idxMax = none;

                int k = j;
                size_t delta = startidx + j;

                if ( !mask )
                {
                    for( ; k < std::min(len0, j + 32764 * 4 * VTraits<v_float64x2>::vlanes()); k += 4 * VTraits<v_float64x2>::vlanes() )
                    {
                        v_float64x2 data = v_load(src + k);
                        v_uint64x2 cmpMin = v_reinterpret_as_u64(v_lt(data, valMin));
                        v_uint64x2 cmpMax = v_reinterpret_as_u64(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + VTraits<v_float64x2>::vlanes());
                        cmpMin = v_reinterpret_as_u64(v_lt(data, valMin));
                        cmpMax = v_reinterpret_as_u64(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + 2 * VTraits<v_float64x2>::vlanes());
                        cmpMin = v_reinterpret_as_u64(v_lt(data, valMin));
                        cmpMax = v_reinterpret_as_u64(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + 3 * VTraits<v_float64x2>::vlanes());
                        cmpMin = v_reinterpret_as_u64(v_lt(data, valMin));
                        cmpMax = v_reinterpret_as_u64(v_gt(data, valMax));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_min(data, valMin);
                        valMax = v_max(data, valMax);
                        idx = v_add(idx, inc);
                    }
                }
                else
                {
                    for( ; k < std::min(len0, j + 32764 * 4 * VTraits<v_float64x2>::vlanes()); k += 4 * VTraits<v_float64x2>::vlanes() )
                    {
                        v_float64x2 data = v_load(src + k);
                        v_uint16x8 maskVal = v_ne(v_load_expand(mask + k), v_setzero_u16());
                        v_int32x4 maskVal1, maskVal2;
                        v_expand(v_reinterpret_as_s16(maskVal), maskVal1, maskVal2);
                        v_int64x2 maskVal3, maskVal4;
                        v_expand(maskVal1, maskVal3, maskVal4);
                        v_uint64x2 cmpMin = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_lt(data, valMin)), maskVal3));
                        v_uint64x2 cmpMax = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_gt(data, valMax)), maskVal3));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_f64(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_f64(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + VTraits<v_float64x2>::vlanes());
                        cmpMin = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_lt(data, valMin)), maskVal4));
                        cmpMax = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_gt(data, valMax)), maskVal4));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_f64(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_f64(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + 2 * VTraits<v_float64x2>::vlanes());
                        v_expand(maskVal2, maskVal3, maskVal4);
                        cmpMin = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_lt(data, valMin)), maskVal3));
                        cmpMax = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_gt(data, valMax)), maskVal3));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_f64(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_f64(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                        data = v_load(src + k + 3 * VTraits<v_float64x2>::vlanes());
                        cmpMin = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_lt(data, valMin)), maskVal4));
                        cmpMax = v_reinterpret_as_u64(v_and(v_reinterpret_as_s64(v_gt(data, valMax)), maskVal4));
                        idxMin = v_select(cmpMin, idx, idxMin);
                        idxMax = v_select(cmpMax, idx, idxMax);
                        valMin = v_select(v_reinterpret_as_f64(cmpMin), data, valMin);
                        valMax = v_select(v_reinterpret_as_f64(cmpMax), data, valMax);
                        idx = v_add(idx, inc);
                    }
                }

                j = k;

                minMaxIdx_reduce_f64( valMin, valMax, idxMin, idxMax, none, minVal, maxVal,
                                      minIdx, maxIdx, delta );
            }
            while ( j < len0 );
        }

        minMaxIdx_finish( src, mask, minval, maxval, minidx, maxidx, minVal, maxVal,
                          minIdx, maxIdx, len, startidx, j );
    }
    else
    {
        minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx);
    }
#else
    minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx );
#endif
}

typedef void (*MinMaxIdxFunc)(const uchar*, const uchar*, int*, int*, size_t*, size_t*, int, size_t);

static MinMaxIdxFunc getMinmaxTab(int depth)
{
    static MinMaxIdxFunc minmaxTab[CV_DEPTH_MAX] =
    {
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_8u), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_8s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_16u), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_16s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_32s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_32f), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_64f),
        0
    };

    return minmaxTab[depth];
}

// The function expects 1-based indexing for ofs
// Zero is treated as invalid offset (not found)
static void ofs2idx(const Mat& a, size_t ofs, int* idx)
{
    int i, d = a.dims;
    if( ofs > 0 )
    {
        ofs--;
        for( i = d-1; i >= 0; i-- )
        {
            int sz = a.size[i];
            idx[i] = (int)(ofs % sz);
            ofs /= sz;
        }
    }
    else
    {
        for( i = d-1; i >= 0; i-- )
            idx[i] = -1;
    }
}

#ifdef HAVE_OPENCL

#define MINMAX_STRUCT_ALIGNMENT 8 // sizeof double

template <typename T>
void getMinMaxRes(const Mat & db, double * minVal, double * maxVal,
                  int* minLoc, int* maxLoc,
                  int groupnum, int cols, double * maxVal2)
{
    uint index_max = std::numeric_limits<uint>::max();
    T minval = std::numeric_limits<T>::max();
    T maxval = std::numeric_limits<T>::min() > 0 ? -std::numeric_limits<T>::max() : std::numeric_limits<T>::min(), maxval2 = maxval;
    uint minloc = index_max, maxloc = index_max;

    size_t index = 0;
    const T * minptr = NULL, * maxptr = NULL, * maxptr2 = NULL;
    const uint * minlocptr = NULL, * maxlocptr = NULL;
    if (minVal || minLoc)
    {
        minptr = db.ptr<T>();
        index += sizeof(T) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (maxVal || maxLoc)
    {
        maxptr = (const T *)(db.ptr() + index);
        index += sizeof(T) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (minLoc)
    {
        minlocptr = (const uint *)(db.ptr() + index);
        index += sizeof(uint) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (maxLoc)
    {
        maxlocptr = (const uint *)(db.ptr() + index);
        index += sizeof(uint) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (maxVal2)
        maxptr2 = (const T *)(db.ptr() + index);

    for (int i = 0; i < groupnum; i++)
    {
        if (minptr && minptr[i] <= minval)
        {
            if (minptr[i] == minval)
            {
                if (minlocptr)
                    minloc = std::min(minlocptr[i], minloc);
            }
            else
            {
                if (minlocptr)
                    minloc = minlocptr[i];
                minval = minptr[i];
            }
        }
        if (maxptr && maxptr[i] >= maxval)
        {
            if (maxptr[i] == maxval)
            {
                if (maxlocptr)
                    maxloc = std::min(maxlocptr[i], maxloc);
            }
            else
            {
                if (maxlocptr)
                    maxloc = maxlocptr[i];
                maxval = maxptr[i];
            }
        }
        if (maxptr2 && maxptr2[i] > maxval2)
            maxval2 = maxptr2[i];
    }
    bool zero_mask = (minLoc && minloc == index_max) ||
            (maxLoc && maxloc == index_max);

    if (minVal)
        *minVal = zero_mask ? 0 : (double)minval;
    if (maxVal)
        *maxVal = zero_mask ? 0 : (double)maxval;
    if (maxVal2)
        *maxVal2 = zero_mask ? 0 : (double)maxval2;

    if (minLoc)
    {
        minLoc[0] = zero_mask ? -1 : minloc / cols;
        minLoc[1] = zero_mask ? -1 : minloc % cols;
    }
    if (maxLoc)
    {
        maxLoc[0] = zero_mask ? -1 : maxloc / cols;
        maxLoc[1] = zero_mask ? -1 : maxloc % cols;
    }
}

typedef void (*getMinMaxResFunc)(const Mat & db, double * minVal, double * maxVal,
                                 int * minLoc, int *maxLoc, int gropunum, int cols, double * maxVal2);

bool ocl_minMaxIdx( InputArray _src, double* minVal, double* maxVal, int* minLoc, int* maxLoc, InputArray _mask,
                           int ddepth, bool absValues, InputArray _src2, double * maxVal2)
{
    const ocl::Device & dev = ocl::Device::getDefault();

#ifdef __ANDROID__
    if (dev.isNVidia())
        return false;
#endif

    if (dev.deviceVersionMajor() == 1 && dev.deviceVersionMinor() < 2)
    {
        // 'static' storage class specifier used by "minmaxloc" is available from OpenCL 1.2+ only
        return false;
    }

    bool doubleSupport = dev.doubleFPConfig() > 0, haveMask = !_mask.empty(),
        haveSrc2 = _src2.kind() != _InputArray::NONE;
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            kercn = haveMask ? cn : std::min(4, ocl::predictOptimalVectorWidth(_src, _src2));

    if (depth >= CV_16F)
        return false;

    // disabled following modes since it occasionally fails on AMD devices (e.g. A10-6800K, sep. 2014)
    if ((haveMask || type == CV_32FC1) && dev.isAMD())
        return false;

    CV_Assert( (cn == 1 && (!haveMask || _mask.type() == CV_8U)) ||
              (cn >= 1 && !minLoc && !maxLoc) );

    if (ddepth < 0)
        ddepth = depth;

    CV_Assert(!haveSrc2 || _src2.type() == type);

    if (depth == CV_32S)
        return false;

    if ((depth == CV_64F || ddepth == CV_64F) && !doubleSupport)
        return false;

    int groupnum = dev.maxComputeUnits();
    size_t wgs = dev.maxWorkGroupSize();

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    bool needMinVal = minVal || minLoc, needMinLoc = minLoc != NULL,
            needMaxVal = maxVal || maxLoc, needMaxLoc = maxLoc != NULL;

    // in case of mask we must know whether mask is filled with zeros or not
    // so let's calculate min or max location, if it's undefined, so mask is zeros
    if (!(needMaxLoc || needMinLoc) && haveMask)
    {
        if (needMinVal)
            needMinLoc = true;
        else
            needMaxLoc = true;
    }

    char cvt[2][50];
    String opts = format("-D DEPTH_%d -D srcT1=%s%s -D WGS=%d -D srcT=%s"
                         " -D WGS2_ALIGNED=%d%s%s%s -D kercn=%d%s%s%s%s"
                         " -D dstT1=%s -D dstT=%s -D convertToDT=%s%s%s%s%s -D wdepth=%d -D convertFromU=%s"
                         " -D MINMAX_STRUCT_ALIGNMENT=%d",
                         depth, ocl::typeToStr(depth), haveMask ? " -D HAVE_MASK" : "", (int)wgs,
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)), wgs2_aligned,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : "",
                         _mask.isContinuous() ? " -D HAVE_MASK_CONT" : "", kercn,
                         needMinVal ? " -D NEED_MINVAL" : "", needMaxVal ? " -D NEED_MAXVAL" : "",
                         needMinLoc ? " -D NEED_MINLOC" : "", needMaxLoc ? " -D NEED_MAXLOC" : "",
                         ocl::typeToStr(ddepth), ocl::typeToStr(CV_MAKE_TYPE(ddepth, kercn)),
                         ocl::convertTypeStr(depth, ddepth, kercn, cvt[0], sizeof(cvt[0])),
                         absValues ? " -D OP_ABS" : "",
                         haveSrc2 ? " -D HAVE_SRC2" : "", maxVal2 ? " -D OP_CALC2" : "",
                         haveSrc2 && _src2.isContinuous() ? " -D HAVE_SRC2_CONT" : "", ddepth,
                         depth <= CV_32S && ddepth == CV_32S ? ocl::convertTypeStr(CV_8U, ddepth, kercn, cvt[1], sizeof(cvt[1])) : "noconvert",
                         MINMAX_STRUCT_ALIGNMENT);

    ocl::Kernel k("minmaxloc", ocl::core::minmaxloc_oclsrc, opts);
    if (k.empty())
        return false;

    int esz = CV_ELEM_SIZE(ddepth), esz32s = CV_ELEM_SIZE1(CV_32S),
            dbsize = groupnum * ((needMinVal ? esz : 0) + (needMaxVal ? esz : 0) +
                                 (needMinLoc ? esz32s : 0) + (needMaxLoc ? esz32s : 0) +
                                 (maxVal2 ? esz : 0))
                     + 5 * MINMAX_STRUCT_ALIGNMENT;
    UMat src = _src.getUMat(), src2 = _src2.getUMat(), db(1, dbsize, CV_8UC1), mask = _mask.getUMat();

    if (cn > 1 && !haveMask)
    {
        src = src.reshape(1);
        src2 = src2.reshape(1);
    }

    if (haveSrc2)
    {
        if (!haveMask)
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db), ocl::KernelArg::ReadOnlyNoSize(src2));
        else
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db), ocl::KernelArg::ReadOnlyNoSize(mask),
                   ocl::KernelArg::ReadOnlyNoSize(src2));
    }
    else
    {
        if (!haveMask)
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db));
        else
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db), ocl::KernelArg::ReadOnlyNoSize(mask));
    }

    size_t globalsize = groupnum * wgs;
    if (!k.run(1, &globalsize, &wgs, true))
        return false;

    static const getMinMaxResFunc functab[7] =
    {
        getMinMaxRes<uchar>,
        getMinMaxRes<char>,
        getMinMaxRes<ushort>,
        getMinMaxRes<short>,
        getMinMaxRes<int>,
        getMinMaxRes<float>,
        getMinMaxRes<double>
    };

    CV_Assert(ddepth <= CV_64F);
    getMinMaxResFunc func = functab[ddepth];

    int locTemp[2];
    func(db.getMat(ACCESS_READ), minVal, maxVal,
         needMinLoc ? minLoc ? minLoc : locTemp : minLoc,
         needMaxLoc ? maxLoc ? maxLoc : locTemp : maxLoc,
         groupnum, src.cols, maxVal2);

    return true;
}

#endif

#ifdef HAVE_OPENVX
namespace ovx {
    template <> inline bool skipSmallImages<VX_KERNEL_MINMAXLOC>(int w, int h) { return w*h < 3840 * 2160; }
}
static bool openvx_minMaxIdx(Mat &src, double* minVal, double* maxVal, int* minIdx, int* maxIdx, Mat &mask)
{
    int stype = src.type();
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size / rows) : 0;
    if ((stype != CV_8UC1 && stype != CV_16SC1) || !mask.empty() ||
        (src.dims != 2 && !(src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size))
        )
        return false;

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();
        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, stype == CV_8UC1 ? VX_DF_IMAGE_U8 : VX_DF_IMAGE_S16,
                ivx::Image::createAddressing(cols, rows, stype == CV_8UC1 ? 1 : 2, (vx_int32)(src.step[0])), src.ptr());

        ivx::Scalar vxMinVal = ivx::Scalar::create(ctx, stype == CV_8UC1 ? VX_TYPE_UINT8 : VX_TYPE_INT16, 0);
        ivx::Scalar vxMaxVal = ivx::Scalar::create(ctx, stype == CV_8UC1 ? VX_TYPE_UINT8 : VX_TYPE_INT16, 0);
        ivx::Array vxMinInd, vxMaxInd;
        ivx::Scalar vxMinCount, vxMaxCount;
        if (minIdx)
        {
            vxMinInd = ivx::Array::create(ctx, VX_TYPE_COORDINATES2D, 1);
            vxMinCount = ivx::Scalar::create(ctx, VX_TYPE_UINT32, 0);
        }
        if (maxIdx)
        {
            vxMaxInd = ivx::Array::create(ctx, VX_TYPE_COORDINATES2D, 1);
            vxMaxCount = ivx::Scalar::create(ctx, VX_TYPE_UINT32, 0);
        }

        ivx::IVX_CHECK_STATUS(vxuMinMaxLoc(ctx, ia, vxMinVal, vxMaxVal, vxMinInd, vxMaxInd, vxMinCount, vxMaxCount));

        if (minVal)
        {
            *minVal = stype == CV_8UC1 ? vxMinVal.getValue<vx_uint8>() : vxMinVal.getValue<vx_int16>();
        }
        if (maxVal)
        {
            *maxVal = stype == CV_8UC1 ? vxMaxVal.getValue<vx_uint8>() : vxMaxVal.getValue<vx_int16>();
        }
        if (minIdx)
        {
            if(vxMinCount.getValue<vx_uint32>()<1) throw ivx::RuntimeError(VX_ERROR_INVALID_VALUE, std::string(__func__) + "(): minimum value location not found");
            vx_coordinates2d_t loc;
            vxMinInd.copyRangeTo(0, 1, &loc);
            size_t minidx = loc.y * cols + loc.x + 1;
            ofs2idx(src, minidx, minIdx);
        }
        if (maxIdx)
        {
            if (vxMaxCount.getValue<vx_uint32>()<1) throw ivx::RuntimeError(VX_ERROR_INVALID_VALUE, std::string(__func__) + "(): maximum value location not found");
            vx_coordinates2d_t loc;
            vxMaxInd.copyRangeTo(0, 1, &loc);
            size_t maxidx = loc.y * cols + loc.x + 1;
            ofs2idx(src, maxidx, maxIdx);
        }
    }
    catch (const ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (const ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
#endif

#ifdef HAVE_IPP
static IppStatus ipp_minMaxIndex_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint* pMinIndex, IppiPoint* pMaxIndex, const Ipp8u*, int)
{
    switch(dataType)
    {
    case ipp8u:  return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp16u: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    default:     return ippStsDataTypeErr;
    }
}

static IppStatus ipp_minMaxIndexMask_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint* pMinIndex, IppiPoint* pMaxIndex, const Ipp8u* pMask, int maskStep)
{
    switch(dataType)
    {
    case ipp8u:  return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_8u_C1MR, (const Ipp8u*)pSrc, srcStep, pMask, maskStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp16u: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_16u_C1MR, (const Ipp16u*)pSrc, srcStep, pMask, maskStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_32f_C1MR, (const Ipp32f*)pSrc, srcStep, pMask, maskStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    default:     return ippStsDataTypeErr;
    }
}

static IppStatus ipp_minMax_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint*, IppiPoint*, const Ipp8u*, int)
{
    IppStatus status;

    switch(dataType)
    {
#if IPP_VERSION_X100 > 201701 // wrong min values
    case ipp8u:
    {
        Ipp8u val[2];
        status = CV_INSTRUMENT_FUN_IPP(ippiMinMax_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, &val[0], &val[1]);
        *pMinVal = val[0];
        *pMaxVal = val[1];
        return status;
    }
#endif
    case ipp16u:
    {
        Ipp16u val[2];
        status = CV_INSTRUMENT_FUN_IPP(ippiMinMax_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, &val[0], &val[1]);
        *pMinVal = val[0];
        *pMaxVal = val[1];
        return status;
    }
    case ipp16s:
    {
        Ipp16s val[2];
        status = CV_INSTRUMENT_FUN_IPP(ippiMinMax_16s_C1R, (const Ipp16s*)pSrc, srcStep, size, &val[0], &val[1]);
        *pMinVal = val[0];
        *pMaxVal = val[1];
        return status;
    }
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinMax_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMinVal, pMaxVal);
    default:     return ipp_minMaxIndex_wrap(pSrc, srcStep, size, dataType, pMinVal, pMaxVal, NULL, NULL, NULL, 0);
    }
}

static IppStatus ipp_minIdx_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float*, IppiPoint* pMinIndex, IppiPoint*, const Ipp8u*, int)
{
    IppStatus status;

    switch(dataType)
    {
    case ipp8u:
    {
        Ipp8u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMinIndx_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, &val, &pMinIndex->x, &pMinIndex->y);
        *pMinVal = val;
        return status;
    }
    case ipp16u:
    {
        Ipp16u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMinIndx_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, &val, &pMinIndex->x, &pMinIndex->y);
        *pMinVal = val;
        return status;
    }
    case ipp16s:
    {
        Ipp16s val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMinIndx_16s_C1R, (const Ipp16s*)pSrc, srcStep, size, &val, &pMinIndex->x, &pMinIndex->y);
        *pMinVal = val;
        return status;
    }
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinIndx_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMinVal, &pMinIndex->x, &pMinIndex->y);
    default:     return ipp_minMaxIndex_wrap(pSrc, srcStep, size, dataType, pMinVal, NULL, pMinIndex, NULL, NULL, 0);
    }
}

static IppStatus ipp_maxIdx_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float*, float* pMaxVal, IppiPoint*, IppiPoint* pMaxIndex, const Ipp8u*, int)
{
    IppStatus status;

    switch(dataType)
    {
    case ipp8u:
    {
        Ipp8u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, &val, &pMaxIndex->x, &pMaxIndex->y);
        *pMaxVal = val;
        return status;
    }
    case ipp16u:
    {
        Ipp16u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, &val, &pMaxIndex->x, &pMaxIndex->y);
        *pMaxVal = val;
        return status;
    }
    case ipp16s:
    {
        Ipp16s val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_16s_C1R, (const Ipp16s*)pSrc, srcStep, size, &val, &pMaxIndex->x, &pMaxIndex->y);
        *pMaxVal = val;
        return status;
    }
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMaxVal, &pMaxIndex->x, &pMaxIndex->y);
    default:     return ipp_minMaxIndex_wrap(pSrc, srcStep, size, dataType, NULL, pMaxVal, NULL, pMaxIndex, NULL, 0);
    }
}

typedef IppStatus (*IppMinMaxSelector)(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint* pMinIndex, IppiPoint* pMaxIndex, const Ipp8u* pMask, int maskStep);

static bool ipp_minMaxIdx(Mat &src, double* _minVal, double* _maxVal, int* _minIdx, int* _maxIdx, Mat &mask)
{
#if IPP_VERSION_X100 >= 700
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201800
    // cv::minMaxIdx problem with NaN input
    // Disable 32F processing only
    if(src.depth() == CV_32F && cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

#if IPP_VERSION_X100 < 201801
    // cv::minMaxIdx problem with index positions on AVX
    if(!mask.empty() && _maxIdx && cv::ipp::getIppTopFeatures() != ippCPUID_SSE42)
        return false;
#endif

    IppStatus   status;
    IppDataType dataType = ippiGetDataType(src.depth());
    float       minVal = 0;
    float       maxVal = 0;
    IppiPoint   minIdx = {-1, -1};
    IppiPoint   maxIdx = {-1, -1};

    float       *pMinVal = (_minVal || _minIdx)?&minVal:NULL;
    float       *pMaxVal = (_maxVal || _maxIdx)?&maxVal:NULL;
    IppiPoint   *pMinIdx = (_minIdx)?&minIdx:NULL;
    IppiPoint   *pMaxIdx = (_maxIdx)?&maxIdx:NULL;

    IppMinMaxSelector ippMinMaxFun = ipp_minMaxIndexMask_wrap;
    if(mask.empty())
    {
        if(_maxVal && _maxIdx && !_minVal && !_minIdx)
            ippMinMaxFun = ipp_maxIdx_wrap;
        else if(!_maxVal && !_maxIdx && _minVal && _minIdx)
            ippMinMaxFun = ipp_minIdx_wrap;
        else if(_maxVal && !_maxIdx && _minVal && !_minIdx)
            ippMinMaxFun = ipp_minMax_wrap;
        else if(!_maxVal && !_maxIdx && !_minVal && !_minIdx)
            return false;
        else
            ippMinMaxFun = ipp_minMaxIndex_wrap;
    }

    if(src.dims <= 2)
    {
        IppiSize size = ippiSize(src.size());
#if defined(_WIN32) && !defined(_WIN64) && IPP_VERSION_X100 == 201900 && IPP_DISABLE_MINMAXIDX_MANY_ROWS
        if (size.height > 65536)
            return false;  // test: Core_MinMaxIdx.rows_overflow
#endif
        size.width *= src.channels();

        status = ippMinMaxFun(src.ptr(), (int)src.step, size, dataType, pMinVal, pMaxVal, pMinIdx, pMaxIdx, (Ipp8u*)mask.ptr(), (int)mask.step);
        if(status < 0)
            return false;
        if(_minVal)
            *_minVal = minVal;
        if(_maxVal)
            *_maxVal = maxVal;
        if(_minIdx)
        {
#if IPP_VERSION_X100 < 201801
            // Should be just ippStsNoOperation check, but there is a bug in the function so we need additional checks
            if(status == ippStsNoOperation && !mask.empty() && !pMinIdx->x && !pMinIdx->y)
#else
            if(status == ippStsNoOperation)
#endif
            {
                _minIdx[0] = -1;
                _minIdx[1] = -1;
            }
            else
            {
                _minIdx[0] = minIdx.y;
                _minIdx[1] = minIdx.x;
            }
        }
        if(_maxIdx)
        {
#if IPP_VERSION_X100 < 201801
            // Should be just ippStsNoOperation check, but there is a bug in the function so we need additional checks
            if(status == ippStsNoOperation && !mask.empty() && !pMaxIdx->x && !pMaxIdx->y)
#else
            if(status == ippStsNoOperation)
#endif
            {
                _maxIdx[0] = -1;
                _maxIdx[1] = -1;
            }
            else
            {
                _maxIdx[0] = maxIdx.y;
                _maxIdx[1] = maxIdx.x;
            }
        }
    }
    else
    {
        const Mat *arrays[] = {&src, mask.empty()?NULL:&mask, NULL};
        uchar     *ptrs[3]  = {NULL};
        NAryMatIterator it(arrays, ptrs);
        IppiSize size = ippiSize(it.size*src.channels(), 1);
        int srcStep      = (int)(size.width*src.elemSize1());
        int maskStep     = size.width;
        size_t idxPos    = 1;
        size_t minIdxAll = 0;
        size_t maxIdxAll = 0;
        float  minValAll = IPP_MAXABS_32F;
        float  maxValAll = -IPP_MAXABS_32F;

        for(size_t i = 0; i < it.nplanes; i++, ++it, idxPos += size.width)
        {
            status = ippMinMaxFun(ptrs[0], srcStep, size, dataType, pMinVal, pMaxVal, pMinIdx, pMaxIdx, ptrs[1], maskStep);
            if(status < 0)
                return false;
#if IPP_VERSION_X100 > 201701
            // Zero-mask check, function should return ippStsNoOperation warning
            if(status == ippStsNoOperation)
                    continue;
#else
            // Crude zero-mask check, waiting for fix in IPP function
            if(ptrs[1])
            {
                Mat localMask(Size(size.width, 1), CV_8U, ptrs[1], maskStep);
                if(!cv::countNonZero(localMask))
                    continue;
            }
#endif

            if(_minVal && minVal < minValAll)
            {
                minValAll = minVal;
                minIdxAll = idxPos+minIdx.x;
            }
            if(_maxVal && maxVal > maxValAll)
            {
                maxValAll = maxVal;
                maxIdxAll = idxPos+maxIdx.x;
            }
        }
        if(!src.empty() && mask.empty())
        {
            if(minIdxAll == 0)
                minIdxAll = 1;
            if(maxValAll == 0)
                maxValAll = 1;
        }

        if(_minVal)
            *_minVal = minValAll;
        if(_maxVal)
            *_maxVal = maxValAll;
        if(_minIdx)
            ofs2idx(src, minIdxAll, _minIdx);
        if(_maxIdx)
            ofs2idx(src, maxIdxAll, _maxIdx);
    }

    return true;
#else
    CV_UNUSED(src); CV_UNUSED(minVal); CV_UNUSED(maxVal); CV_UNUSED(minIdx); CV_UNUSED(maxIdx); CV_UNUSED(mask);
    return false;
#endif
}
#endif

}

void cv::minMaxIdx(InputArray _src, double* minVal,
                   double* maxVal, int* minIdx, int* maxIdx,
                   InputArray _mask)
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( (cn == 1 && (_mask.empty() || _mask.type() == CV_8U)) ||
        (cn > 1 && _mask.empty() && !minIdx && !maxIdx) );

    CV_OCL_RUN(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2  && (_mask.empty() || _src.size() == _mask.size()),
               ocl_minMaxIdx(_src, minVal, maxVal, minIdx, maxIdx, _mask))

    Mat src = _src.getMat(), mask = _mask.getMat();

    if (src.dims <= 2)
    {
        if ((size_t)src.step == (size_t)mask.step)
        {
            CALL_HAL(minMaxIdx, cv_hal_minMaxIdx, src.data, src.step, src.cols*cn, src.rows,
                     src.depth(), minVal, maxVal, minIdx, maxIdx, mask.data);
        }
        else
        {
            CALL_HAL(minMaxIdxMaskStep, cv_hal_minMaxIdxMaskStep, src.data, src.step, src.cols*cn, src.rows,
                     src.depth(), minVal, maxVal, minIdx, maxIdx, mask.data, mask.step);
        }

    }
    else if (src.isContinuous() && mask.isContinuous())
    {
        int res = cv_hal_minMaxIdx(src.data, 0, (int)src.total()*cn, 1, src.depth(),
                                   minVal, maxVal, minIdx, maxIdx, mask.data);

        if (res == CV_HAL_ERROR_OK)
        {
            // minIdx[0] and minIdx[0] are always 0 for "flatten" version
            if (minIdx)
                ofs2idx(src, minIdx[1]+1, minIdx);
            if (maxIdx)
                ofs2idx(src, maxIdx[1]+1, maxIdx);
            return;
        }
        else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED)
        {
            CV_Error_(cv::Error::StsInternal,
            ("HAL implementation minMaxIdx ==> " CVAUX_STR(cv_hal_minMaxIdx) " returned %d (0x%08x)", res, res));
        }
    }

    CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_MINMAXLOC>(src.cols, src.rows),
               openvx_minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask))

    CV_IPP_RUN_FAST(ipp_minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask))

    MinMaxIdxFunc func = getMinmaxTab(depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);

    size_t minidx = 0, maxidx = 0;
    int iminval = INT_MAX, imaxval = INT_MIN;
    float  fminval = std::numeric_limits<float>::infinity(),  fmaxval = -fminval;
    double dminval = std::numeric_limits<double>::infinity(), dmaxval = -dminval;
    size_t startidx = 1;
    int *minval = &iminval, *maxval = &imaxval;
    int planeSize = (int)it.size*cn;

    if( depth == CV_32F )
        minval = (int*)&fminval, maxval = (int*)&fmaxval;
    else if( depth == CV_64F )
        minval = (int*)&dminval, maxval = (int*)&dmaxval;

    for( size_t i = 0; i < it.nplanes; i++, ++it, startidx += planeSize )
        func( ptrs[0], ptrs[1], minval, maxval, &minidx, &maxidx, planeSize, startidx );

    if (!src.empty() && mask.empty())
    {
        if( minidx == 0 )
            minidx = 1;
        if( maxidx == 0 )
            maxidx = 1;
    }

    if( minidx == 0 )
        dminval = dmaxval = 0;
    else if( depth == CV_32F )
        dminval = fminval, dmaxval = fmaxval;
    else if( depth <= CV_32S )
        dminval = iminval, dmaxval = imaxval;

    if( minVal )
        *minVal = dminval;
    if( maxVal )
        *maxVal = dmaxval;

    if( minIdx )
        ofs2idx(src, minidx, minIdx);
    if( maxIdx )
        ofs2idx(src, maxidx, maxIdx);
}

void cv::minMaxLoc( InputArray _img, double* minVal, double* maxVal,
                    Point* minLoc, Point* maxLoc, InputArray mask )
{
    CV_INSTRUMENT_REGION();

    int dims = _img.dims();
    CV_CheckLE(dims, 2, "");

    minMaxIdx(_img, minVal, maxVal, (int*)minLoc, (int*)maxLoc, mask);
    if( minLoc )
    {
        if (dims == 2)
            std::swap(minLoc->x, minLoc->y);
        else
            minLoc->y = 0;
    }
    if( maxLoc )
    {
        if (dims == 2)
            std::swap(maxLoc->x, maxLoc->y);
        else
            maxLoc->y = 0;
    }
}

enum class ReduceMode
{
    FIRST_MIN = 0, //!< get index of first min occurrence
    LAST_MIN  = 1, //!< get index of last min occurrence
    FIRST_MAX = 2, //!< get index of first max occurrence
    LAST_MAX  = 3, //!< get index of last max occurrence
};

template <typename T>
struct reduceMinMaxImpl
{
    void operator()(const cv::Mat& src, cv::Mat& dst, ReduceMode mode, const int axis) const
    {
        switch(mode)
        {
        case ReduceMode::FIRST_MIN:
            reduceMinMaxApply<std::less>(src, dst, axis);
            break;
        case ReduceMode::LAST_MIN:
            reduceMinMaxApply<std::less_equal>(src, dst, axis);
            break;
        case ReduceMode::FIRST_MAX:
            reduceMinMaxApply<std::greater>(src, dst, axis);
            break;
        case ReduceMode::LAST_MAX:
            reduceMinMaxApply<std::greater_equal>(src, dst, axis);
            break;
        }
    }

    template <template<class> class Cmp>
    static void reduceMinMaxApply(const cv::Mat& src, cv::Mat& dst, const int axis)
    {
        Cmp<T> cmp;

        const auto *src_ptr = src.ptr<T>();
        auto *dst_ptr = dst.ptr<int32_t>();

        const size_t outer_size = src.total(0, axis);
        const auto mid_size = static_cast<size_t>(src.size[axis]);

        const size_t outer_step = src.total(axis);
        const size_t dst_step = dst.total(axis);

        const size_t mid_step = src.total(axis + 1);

        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            const size_t outer_offset = outer * outer_step;
            const size_t dst_offset = outer * dst_step;
            for (size_t mid = 0; mid != mid_size; ++mid)
            {
                const size_t src_offset = outer_offset + mid * mid_step;
                for (size_t inner = 0; inner < mid_step; inner++)
                {
                    int32_t& index = dst_ptr[dst_offset + inner];

                    const size_t prev = outer_offset + index * mid_step + inner;
                    const size_t curr = src_offset + inner;

                    if (cmp(src_ptr[curr], src_ptr[prev]))
                    {
                        index = static_cast<int32_t>(mid);
                    }
                }
            }
        }
    }
};

static void reduceMinMax(cv::InputArray src, cv::OutputArray dst, ReduceMode mode, int axis)
{
    CV_INSTRUMENT_REGION();

    cv::Mat srcMat = src.getMat();
    axis = (axis + srcMat.dims) % srcMat.dims;
    CV_Assert(srcMat.channels() == 1 && axis >= 0 && axis < srcMat.dims);

    std::vector<int> sizes(srcMat.dims);
    std::copy(srcMat.size.p, srcMat.size.p + srcMat.dims, sizes.begin());
    sizes[axis] = 1;

    dst.create(srcMat.dims, sizes.data(), CV_32SC1); // indices
    cv::Mat dstMat = dst.getMat();
    dstMat.setTo(cv::Scalar::all(0));

    if (!srcMat.isContinuous())
    {
        srcMat = srcMat.clone();
    }

    bool needs_copy = !dstMat.isContinuous();
    if (needs_copy)
    {
        dstMat = dstMat.clone();
    }

    cv::detail::depthDispatch<reduceMinMaxImpl>(srcMat.depth(), srcMat, dstMat, mode, axis);

    if (needs_copy)
    {
        dstMat.copyTo(dst);
    }
}

void cv::reduceArgMin(InputArray src, OutputArray dst, int axis, bool lastIndex)
{
    reduceMinMax(src, dst, lastIndex ? ReduceMode::LAST_MIN : ReduceMode::FIRST_MIN, axis);
}

void cv::reduceArgMax(InputArray src, OutputArray dst, int axis, bool lastIndex)
{
    reduceMinMax(src, dst, lastIndex ? ReduceMode::LAST_MAX : ReduceMode::FIRST_MAX, axis);
}
