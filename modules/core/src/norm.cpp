// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "stat.hpp"

/****************************************************************************************\
*                                         norm                                           *
\****************************************************************************************/

namespace cv { namespace hal {

extern const uchar popCountTable[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

static const uchar popCountTable2[] =
{
    0, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4
};

static const uchar popCountTable4[] =
{
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
};


int normHamming(const uchar* a, int n, int cellSize)
{
    int output;
    CALL_HAL_RET(normHamming8u, cv_hal_normHamming8u, output, a, n, cellSize);

    if( cellSize == 1 )
        return normHamming(a, n);
    const uchar* tab = 0;
    if( cellSize == 2 )
        tab = popCountTable2;
    else if( cellSize == 4 )
        tab = popCountTable4;
    else
        return -1;
    int i = 0;
    int result = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_uint64 t = vx_setzero_u64();
    if ( cellSize == 2)
    {
        v_uint16 mask = v_reinterpret_as_u16(vx_setall_u8(0x55));
        for(; i <= n - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes())
        {
            v_uint16 a0 = v_reinterpret_as_u16(vx_load(a + i));
            t = v_add(t, v_popcount(v_reinterpret_as_u64(v_and(v_or(a0, v_shr<1>(a0)), mask))));
        }
    }
    else    // cellSize == 4
    {
        v_uint16 mask = v_reinterpret_as_u16(vx_setall_u8(0x11));
        for(; i <= n - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes())
        {
            v_uint16 a0 = v_reinterpret_as_u16(vx_load(a + i));
            v_uint16 a1 = v_or(a0, v_shr<2>(a0));
            t = v_add(t, v_popcount(v_reinterpret_as_u64(v_and(v_or(a1, v_shr<1>(a1)), mask))));

        }
    }
    result += (int)v_reduce_sum(t);
    vx_cleanup();
#elif CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
        result += tab[a[i]] + tab[a[i+1]] + tab[a[i+2]] + tab[a[i+3]];
#endif
    for( ; i < n; i++ )
        result += tab[a[i]];
    return result;
}

int normHamming(const uchar* a, const uchar* b, int n, int cellSize)
{
    int output;
    CALL_HAL_RET(normHammingDiff8u, cv_hal_normHammingDiff8u, output, a, b, n, cellSize);

    if( cellSize == 1 )
        return normHamming(a, b, n);
    const uchar* tab = 0;
    if( cellSize == 2 )
        tab = popCountTable2;
    else if( cellSize == 4 )
        tab = popCountTable4;
    else
        return -1;
    int i = 0;
    int result = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_uint64 t = vx_setzero_u64();
    if ( cellSize == 2)
    {
        v_uint16 mask = v_reinterpret_as_u16(vx_setall_u8(0x55));
        for(; i <= n - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes())
        {
            v_uint16 ab0 = v_reinterpret_as_u16(v_xor(vx_load(a + i), vx_load(b + i)));
            t = v_add(t, v_popcount(v_reinterpret_as_u64(v_and(v_or(ab0, v_shr<1>(ab0)), mask))));
        }
    }
    else    // cellSize == 4
    {
        v_uint16 mask = v_reinterpret_as_u16(vx_setall_u8(0x11));
        for(; i <= n - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes())
        {
            v_uint16 ab0 = v_reinterpret_as_u16(v_xor(vx_load(a + i), vx_load(b + i)));
            v_uint16 ab1 = v_or(ab0, v_shr<2>(ab0));
            t = v_add(t, v_popcount(v_reinterpret_as_u64(v_and(v_or(ab1, v_shr<1>(ab1)), mask))));
        }
    }
    result += (int)v_reduce_sum(t);
    vx_cleanup();
#elif CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
        result += tab[a[i] ^ b[i]] + tab[a[i+1] ^ b[i+1]] +
                tab[a[i+2] ^ b[i+2]] + tab[a[i+3] ^ b[i+3]];
#endif
    for( ; i < n; i++ )
        result += tab[a[i] ^ b[i]];
    return result;
}

float normL2Sqr_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 v_d0 = vx_setzero_f32(), v_d1 = vx_setzero_f32();
    v_float32 v_d2 = vx_setzero_f32(), v_d3 = vx_setzero_f32();
    for (; j <= n - 4 * VTraits<v_float32>::vlanes(); j += 4 * VTraits<v_float32>::vlanes())
    {
        v_float32 t0 = v_sub(vx_load(a + j), vx_load(b + j));
        v_float32 t1 = v_sub(vx_load(a + j + VTraits<v_float32>::vlanes()), vx_load(b + j + VTraits<v_float32>::vlanes()));
        v_d0 = v_muladd(t0, t0, v_d0);
        v_float32 t2 = v_sub(vx_load(a + j + 2 * VTraits<v_float32>::vlanes()), vx_load(b + j + 2 * VTraits<v_float32>::vlanes()));
        v_d1 = v_muladd(t1, t1, v_d1);
        v_float32 t3 = v_sub(vx_load(a + j + 3 * VTraits<v_float32>::vlanes()), vx_load(b + j + 3 * VTraits<v_float32>::vlanes()));
        v_d2 = v_muladd(t2, t2, v_d2);
        v_d3 = v_muladd(t3, t3, v_d3);
    }
    d = v_reduce_sum(v_add(v_add(v_add(v_d0, v_d1), v_d2), v_d3));
#endif
    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
    return d;
}


float normL1_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 v_d0 = vx_setzero_f32(), v_d1 = vx_setzero_f32();
    v_float32 v_d2 = vx_setzero_f32(), v_d3 = vx_setzero_f32();
    for (; j <= n - 4 * VTraits<v_float32>::vlanes(); j += 4 * VTraits<v_float32>::vlanes())
    {
        v_d0 = v_add(v_d0, v_absdiff(vx_load(a + j), vx_load(b + j)));
        v_d1 = v_add(v_d1, v_absdiff(vx_load(a + j + VTraits<v_float32>::vlanes()), vx_load(b + j + VTraits<v_float32>::vlanes())));
        v_d2 = v_add(v_d2, v_absdiff(vx_load(a + j + 2 * VTraits<v_float32>::vlanes()), vx_load(b + j + 2 * VTraits<v_float32>::vlanes())));
        v_d3 = v_add(v_d3, v_absdiff(vx_load(a + j + 3 * VTraits<v_float32>::vlanes()), vx_load(b + j + 3 * VTraits<v_float32>::vlanes())));
    }
    d = v_reduce_sum(v_add(v_add(v_add(v_d0, v_d1), v_d2), v_d3));
#endif
    for( ; j < n; j++ )
        d += std::abs(a[j] - b[j]);
    return d;
}

int normL1_(const uchar* a, const uchar* b, int n)
{
    int j = 0, d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; j <= n - 4 * VTraits<v_uint8>::vlanes(); j += 4 * VTraits<v_uint8>::vlanes())
        d += v_reduce_sad(vx_load(a + j), vx_load(b + j)) +
             v_reduce_sad(vx_load(a + j + VTraits<v_uint8>::vlanes()), vx_load(b + j + VTraits<v_uint8>::vlanes())) +
             v_reduce_sad(vx_load(a + j + 2 * VTraits<v_uint8>::vlanes()), vx_load(b + j + 2 * VTraits<v_uint8>::vlanes())) +
             v_reduce_sad(vx_load(a + j + 3 * VTraits<v_uint8>::vlanes()), vx_load(b + j + 3 * VTraits<v_uint8>::vlanes()));
#endif
    for( ; j < n; j++ )
        d += std::abs(a[j] - b[j]);
    return d;
}

} //cv::hal

//==================================================================================================

template<typename T, typename ST> int
normInf_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result = std::max(result, normInf<T, ST>(src, len*cn));
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result = std::max(result, ST(cv_abs(src[k])));
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL1_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL1<T, ST>(src, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result += cv_abs(src[k]);
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL2_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL2Sqr<T, ST>(src, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    T v = src[k];
                    result += (ST)v*v;
                }
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffInf_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result = std::max(result, normInf<T, ST>(src1, src2, len*cn));
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result = std::max(result, (ST)std::abs(src1[k] - src2[k]));
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL1_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL1<T, ST>(src1, src2, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result += std::abs(src1[k] - src2[k]);
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL2_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL2Sqr<T, ST>(src1, src2, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    ST v = src1[k] - src2[k];
                    result += v*v;
                }
            }
    }
    *_result = result;
    return 0;
}

#define CV_DEF_NORM_FUNC(L, suffix, type, ntype) \
    static int norm##L##_##suffix(const type* src, const uchar* mask, ntype* r, int len, int cn) \
{ return norm##L##_(src, mask, r, len, cn); } \
    static int normDiff##L##_##suffix(const type* src1, const type* src2, \
    const uchar* mask, ntype* r, int len, int cn) \
{ return normDiff##L##_(src1, src2, mask, r, (int)len, cn); }

#define CV_DEF_NORM_ALL(suffix, type, inftype, l1type, l2type) \
    CV_DEF_NORM_FUNC(Inf, suffix, type, inftype) \
    CV_DEF_NORM_FUNC(L1, suffix, type, l1type) \
    CV_DEF_NORM_FUNC(L2, suffix, type, l2type)

CV_DEF_NORM_ALL(8u, uchar, int, int, int)
CV_DEF_NORM_ALL(8s, schar, int, int, int)
CV_DEF_NORM_ALL(16u, ushort, int, int, double)
CV_DEF_NORM_ALL(16s, short, int, int, double)
CV_DEF_NORM_ALL(32s, int, int, double, double)
CV_DEF_NORM_ALL(32f, float, float, double, double)
CV_DEF_NORM_ALL(64f, double, double, double, double)


typedef int (*NormFunc)(const uchar*, const uchar*, uchar*, int, int);
typedef int (*NormDiffFunc)(const uchar*, const uchar*, const uchar*, uchar*, int, int);

static NormFunc getNormFunc(int normType, int depth)
{
    static NormFunc normTab[3][8] =
    {
        {
            (NormFunc)GET_OPTIMIZED(normInf_8u), (NormFunc)GET_OPTIMIZED(normInf_8s), (NormFunc)GET_OPTIMIZED(normInf_16u), (NormFunc)GET_OPTIMIZED(normInf_16s),
            (NormFunc)GET_OPTIMIZED(normInf_32s), (NormFunc)GET_OPTIMIZED(normInf_32f), (NormFunc)normInf_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL1_8u), (NormFunc)GET_OPTIMIZED(normL1_8s), (NormFunc)GET_OPTIMIZED(normL1_16u), (NormFunc)GET_OPTIMIZED(normL1_16s),
            (NormFunc)GET_OPTIMIZED(normL1_32s), (NormFunc)GET_OPTIMIZED(normL1_32f), (NormFunc)normL1_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL2_8u), (NormFunc)GET_OPTIMIZED(normL2_8s), (NormFunc)GET_OPTIMIZED(normL2_16u), (NormFunc)GET_OPTIMIZED(normL2_16s),
            (NormFunc)GET_OPTIMIZED(normL2_32s), (NormFunc)GET_OPTIMIZED(normL2_32f), (NormFunc)normL2_64f, 0
        }
    };

    return normTab[normType][depth];
}

static NormDiffFunc getNormDiffFunc(int normType, int depth)
{
    static NormDiffFunc normDiffTab[3][8] =
    {
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffInf_8u), (NormDiffFunc)normDiffInf_8s,
            (NormDiffFunc)normDiffInf_16u, (NormDiffFunc)normDiffInf_16s,
            (NormDiffFunc)normDiffInf_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffInf_32f),
            (NormDiffFunc)normDiffInf_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL1_8u), (NormDiffFunc)normDiffL1_8s,
            (NormDiffFunc)normDiffL1_16u, (NormDiffFunc)normDiffL1_16s,
            (NormDiffFunc)normDiffL1_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL1_32f),
            (NormDiffFunc)normDiffL1_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL2_8u), (NormDiffFunc)normDiffL2_8s,
            (NormDiffFunc)normDiffL2_16u, (NormDiffFunc)normDiffL2_16s,
            (NormDiffFunc)normDiffL2_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL2_32f),
            (NormDiffFunc)normDiffL2_64f, 0
        }
    };

    return normDiffTab[normType][depth];
}

#ifdef HAVE_OPENCL

static bool ocl_norm( InputArray _src, int normType, InputArray _mask, double & result )
{
    const ocl::Device & d = ocl::Device::getDefault();

#ifdef __ANDROID__
    if (d.isNVidia())
        return false;
#endif
    const int cn = _src.channels();
    if (cn > 4)
        return false;
    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    bool doubleSupport = d.doubleFPConfig() > 0,
            haveMask = _mask.kind() != _InputArray::NONE;

    if (depth >= CV_16F)
        return false;  // TODO: support FP16

    if ( !(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR) ||
         (!doubleSupport && depth == CV_64F))
        return false;

    UMat src = _src.getUMat();

    if (normType == NORM_INF)
    {
        if (!ocl_minMaxIdx(_src, NULL, &result, NULL, NULL, _mask,
                           std::max(depth, CV_32S), depth != CV_8U && depth != CV_16U))
            return false;
    }
    else if (normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR)
    {
        Scalar sc;
        bool unstype = depth == CV_8U || depth == CV_16U;

        if ( !ocl_sum(haveMask ? src : src.reshape(1), sc, normType == NORM_L2 || normType == NORM_L2SQR ?
                    OCL_OP_SUM_SQR : (unstype ? OCL_OP_SUM : OCL_OP_SUM_ABS), _mask) )
            return false;

        double s = 0.0;
        for (int i = 0; i < (haveMask ? cn : 1); ++i)
            s += sc[i];

        result = normType == NORM_L1 || normType == NORM_L2SQR ? s : std::sqrt(s);
    }

    return true;
}

#endif

#ifdef HAVE_IPP
static bool ipp_norm(Mat &src, int normType, Mat &mask, double &result)
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 >= 700
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;

    if( (src.dims == 2 || (src.isContinuous() && mask.isContinuous()))
        && cols > 0 && (size_t)rows*cols == total_size )
    {
        if( !mask.empty() )
        {
            IppiSize sz = { cols, rows };
            int type = src.type();

            typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormFuncC1 ippiNorm_C1MR =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_32f_C1MR :
                0) :
            normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_32f_C1MR :
                0) :
            normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_32f_C1MR :
                0) : 0;
            if( ippiNorm_C1MR )
            {
                Ipp64f norm;
                if( CV_INSTRUMENT_FUN_IPP(ippiNorm_C1MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, &norm) >= 0 )
                {
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskNormFuncC3 ippiNorm_C3CMR =
                normType == NORM_INF ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_32f_C3CMR :
                0) :
            normType == NORM_L1 ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_32f_C3CMR :
                0) :
            normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_32f_C3CMR :
                0) : 0;
            if( ippiNorm_C3CMR )
            {
                Ipp64f norm1, norm2, norm3;
                if( CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 1, &norm1) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 2, &norm2) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 3, &norm3) >= 0)
                {
                    Ipp64f norm =
                        normType == NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                        normType == NORM_L1 ? norm1 + norm2 + norm3 :
                        normType == NORM_L2 || normType == NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                        0;
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
        }
        else
        {
            IppiSize sz = { cols*src.channels(), rows };
            int type = src.depth();

            typedef IppStatus (CV_STDCALL* ippiNormFuncHint)(const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            typedef IppStatus (CV_STDCALL* ippiNormFuncNoHint)(const void *, int, IppiSize, Ipp64f *);
            ippiNormFuncHint ippiNormHint =
                normType == NORM_L1 ?
                (type == CV_32FC1 ? (ippiNormFuncHint)ippiNorm_L1_32f_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_32FC1 ? (ippiNormFuncHint)ippiNorm_L2_32f_C1R :
                0) : 0;
            ippiNormFuncNoHint ippiNorm =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_8u_C1R :
                type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_16u_C1R :
                type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_16s_C1R :
                type == CV_32FC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_32f_C1R :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_L1_8u_C1R :
                type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_L1_16u_C1R :
                type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_L1_16s_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_L2_8u_C1R :
                type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_L2_16u_C1R :
                type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_L2_16s_C1R :
                0) : 0;
            if( ippiNormHint || ippiNorm )
            {
                Ipp64f norm;
                IppStatus ret = ippiNormHint ? CV_INSTRUMENT_FUN_IPP(ippiNormHint, src.ptr(), (int)src.step[0], sz, &norm, ippAlgHintAccurate) :
                                CV_INSTRUMENT_FUN_IPP(ippiNorm, src.ptr(), (int)src.step[0], sz, &norm);
                if( ret >= 0 )
                {
                    result = (normType == NORM_L2SQR) ? norm * norm : norm;
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(normType); CV_UNUSED(mask); CV_UNUSED(result);
#endif
    return false;
}  // ipp_norm()
#endif  // HAVE_IPP

double norm( InputArray _src, int normType, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    normType &= NORM_TYPE_MASK;
    CV_Assert( normType == NORM_INF || normType == NORM_L1 ||
               normType == NORM_L2 || normType == NORM_L2SQR ||
               ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && _src.type() == CV_8U) );

#if defined HAVE_OPENCL || defined HAVE_IPP
    double _result = 0;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_norm(_src, normType, _mask, _result),
                _result)
#endif

    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_norm(src, normType, mask, _result), _result);

    int depth = src.depth(), cn = src.channels();
    if( src.isContinuous() && mask.empty() )
    {
        size_t len = src.total()*cn;
        if( len == (size_t)(int)len )
        {
            if( depth == CV_32F )
            {
                const float* data = src.ptr<float>();

                if( normType == NORM_L2 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL2_32f)(data, 0, &result, (int)len, 1);
                    return std::sqrt(result);
                }
                if( normType == NORM_L2SQR )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL2_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_L1 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL1_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_INF )
                {
                    float result = 0;
                    GET_OPTIMIZED(normInf_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
            }
            if( depth == CV_8U )
            {
                const uchar* data = src.ptr<uchar>();

                if( normType == NORM_HAMMING )
                {
                    return hal::normHamming(data, (int)len);
                }

                if( normType == NORM_HAMMING2 )
                {
                    return hal::normHamming(data, (int)len, 2);
                }
            }
        }
    }

    CV_Assert( mask.empty() || mask.type() == CV_8U );

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_and(src, mask, temp);
            return norm(temp, normType);
        }
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        const Mat* arrays[] = {&src, 0};
        uchar* ptrs[1] = {};
        NAryMatIterator it(arrays, ptrs);
        int total = (int)it.size;
        int result = 0;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            result += hal::normHamming(ptrs[0], total, cellSize);
        }

        return result;
    }

    NormFunc func = getNormFunc(normType >> 1, depth == CV_16F ? CV_32F : depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2] = {};
    union
    {
        double d;
        int i;
        float f;
    }
    result;
    result.d = 0;
    NAryMatIterator it(arrays, ptrs);
    CV_CheckLT((size_t)it.size, (size_t)INT_MAX, "");

    if ((normType == NORM_L1 && depth <= CV_16S) ||
        ((normType == NORM_L2 || normType == NORM_L2SQR) && depth <= CV_8S))
    {
        // special case to handle "integer" overflow in accumulator
        const size_t esz = src.elemSize();
        const int total = (int)it.size;
        const int intSumBlockSize = (normType == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15))/cn;
        const int blockSize = std::min(total, intSumBlockSize);
        int isum = 0;
        int count = 0;

        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            for (int j = 0; j < total; j += blockSize)
            {
                int bsz = std::min(total - j, blockSize);
                func(ptrs[0], ptrs[1], (uchar*)&isum, bsz, cn);
                count += bsz;
                if (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total))
                {
                    result.d += isum;
                    isum = 0;
                    count = 0;
                }
                ptrs[0] += bsz*esz;
                if (ptrs[1])
                    ptrs[1] += bsz;
            }
        }
    }
    else if (depth == CV_16F)
    {
        const size_t esz = src.elemSize();
        const int total = (int)it.size;
        const int blockSize = std::min(total, divUp(1024, cn));
        AutoBuffer<float, 1026/*divUp(1024,3)*3*/> fltbuf(blockSize * cn);
        float* data0 = fltbuf.data();
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            for (int j = 0; j < total; j += blockSize)
            {
                int bsz = std::min(total - j, blockSize);
                hal::cvt16f32f((const hfloat*)ptrs[0], data0, bsz * cn);
                func((uchar*)data0, ptrs[1], (uchar*)&result.f, bsz, cn);
                ptrs[0] += bsz*esz;
                if (ptrs[1])
                    ptrs[1] += bsz;
            }
        }
    }
    else
    {
        // generic implementation
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            func(ptrs[0], ptrs[1], (uchar*)&result, (int)it.size, cn);
        }
    }

    if( normType == NORM_INF )
    {
        if(depth == CV_64F)
            return result.d;
        else if (depth == CV_32F || depth == CV_16F)
            return result.f;
        else
            return result.i;
    }
    else if( normType == NORM_L2 )
        return std::sqrt(result.d);

    return result.d;
}

//==================================================================================================

#ifdef HAVE_OPENCL
static bool ocl_norm( InputArray _src1, InputArray _src2, int normType, InputArray _mask, double & result )
{
#ifdef __ANDROID__
    if (ocl::Device::getDefault().isNVidia())
        return false;
#endif

    Scalar sc1, sc2;
    int cn = _src1.channels();
    if (cn > 4)
        return false;
    int type = _src1.type(), depth = CV_MAT_DEPTH(type);
    bool relative = (normType & NORM_RELATIVE) != 0;
    normType &= ~NORM_RELATIVE;
    bool normsum = normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR;

#ifdef __APPLE__
    if(normType == NORM_L1 && type == CV_16UC3 && !_mask.empty())
        return false;
#endif

    if (normsum)
    {
        if (!ocl_sum(_src1, sc1, normType == NORM_L2 || normType == NORM_L2SQR ?
                     OCL_OP_SUM_SQR : OCL_OP_SUM, _mask, _src2, relative, sc2))
            return false;
    }
    else
    {
        if (!ocl_minMaxIdx(_src1, NULL, &sc1[0], NULL, NULL, _mask, std::max(CV_32S, depth),
                           false, _src2, relative ? &sc2[0] : NULL))
            return false;
        cn = 1;
    }

    double s2 = 0;
    for (int i = 0; i < cn; ++i)
    {
        result += sc1[i];
        if (relative)
            s2 += sc2[i];
    }

    if (normType == NORM_L2)
    {
        result = std::sqrt(result);
        if (relative)
            s2 = std::sqrt(s2);
    }

    if (relative)
        result /= (s2 + DBL_EPSILON);

    return true;
}  // ocl_norm()
#endif  // HAVE_OPENCL

#ifdef HAVE_IPP
static bool ipp_norm(InputArray _src1, InputArray _src2, int normType, InputArray _mask, double &result)
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 >= 700
    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();

    if( normType & CV_RELATIVE )
    {
        normType &= NORM_TYPE_MASK;

        size_t total_size = src1.total();
        int rows = src1.size[0], cols = rows ? (int)(total_size/rows) : 0;
        if( (src1.dims == 2 || (src1.isContinuous() && src2.isContinuous() && mask.isContinuous()))
            && cols > 0 && (size_t)rows*cols == total_size )
        {
            if( !mask.empty() )
            {
                IppiSize sz = { cols, rows };
                int type = src1.type();

                typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
                ippiMaskNormDiffFuncC1 ippiNormRel_C1MR =
                    normType == NORM_INF ?
                    (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_8u_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_32f_C1MR :
                    0) :
                    normType == NORM_L1 ?
                    (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_8u_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_32f_C1MR :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_8u_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_32f_C1MR :
                    0) : 0;
                if( ippiNormRel_C1MR )
                {
                    Ipp64f norm;
                    if( CV_INSTRUMENT_FUN_IPP(ippiNormRel_C1MR, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], mask.ptr(), (int)mask.step[0], sz, &norm) >= 0 )
                    {
                        result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                        return true;
                    }
                }
            }
            else
            {
                IppiSize sz = { cols*src1.channels(), rows };
                int type = src1.depth();

                typedef IppStatus (CV_STDCALL* ippiNormRelFuncHint)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
                typedef IppStatus (CV_STDCALL* ippiNormRelFuncNoHint)(const void *, int, const void *, int, IppiSize, Ipp64f *);
                ippiNormRelFuncHint ippiNormRelHint =
                    normType == NORM_L1 ?
                    (type == CV_32F ? (ippiNormRelFuncHint)ippiNormRel_L1_32f_C1R :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_32F ? (ippiNormRelFuncHint)ippiNormRel_L2_32f_C1R :
                    0) : 0;
                ippiNormRelFuncNoHint ippiNormRel =
                    normType == NORM_INF ?
                    (type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_8u_C1R :
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_16s_C1R :
                    type == CV_32F ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_32f_C1R :
                    0) :
                    normType == NORM_L1 ?
                    (type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_L1_8u_C1R :
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_L1_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_L1_16s_C1R :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_L2_8u_C1R :
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_L2_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_L2_16s_C1R :
                    0) : 0;
                if( ippiNormRelHint || ippiNormRel )
                {
                    Ipp64f norm;
                    IppStatus ret = ippiNormRelHint ? CV_INSTRUMENT_FUN_IPP(ippiNormRelHint, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm, ippAlgHintAccurate) :
                                    CV_INSTRUMENT_FUN_IPP(ippiNormRel, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm);
                    if( ret >= 0 )
                    {
                        result = (normType == NORM_L2SQR) ? norm * norm : norm;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    normType &= NORM_TYPE_MASK;

    size_t total_size = src1.total();
    int rows = src1.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( (src1.dims == 2 || (src1.isContinuous() && src2.isContinuous() && mask.isContinuous()))
        && cols > 0 && (size_t)rows*cols == total_size )
    {
        if( !mask.empty() )
        {
            IppiSize sz = { cols, rows };
            int type = src1.type();

            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormDiffFuncC1 ippiNormDiff_C1MR =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_32f_C1MR :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_32f_C1MR :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_32f_C1MR :
                0) : 0;
            if( ippiNormDiff_C1MR )
            {
                Ipp64f norm;
                if( CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C1MR, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], mask.ptr(), (int)mask.step[0], sz, &norm) >= 0 )
                {
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC3)(const void *, int, const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskNormDiffFuncC3 ippiNormDiff_C3CMR =
                normType == NORM_INF ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_32f_C3CMR :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_32f_C3CMR :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_32f_C3CMR :
                0) : 0;
            if( ippiNormDiff_C3CMR )
            {
                Ipp64f norm1, norm2, norm3;
                if( CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 1, &norm1) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 2, &norm2) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 3, &norm3) >= 0)
                {
                    Ipp64f norm =
                        normType == NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                        normType == NORM_L1 ? norm1 + norm2 + norm3 :
                        normType == NORM_L2 || normType == NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                        0;
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
        }
        else
        {
            IppiSize sz = { cols*src1.channels(), rows };
            int type = src1.depth();

            typedef IppStatus (CV_STDCALL* ippiNormDiffFuncHint)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            typedef IppStatus (CV_STDCALL* ippiNormDiffFuncNoHint)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiNormDiffFuncHint ippiNormDiffHint =
                normType == NORM_L1 ?
                (type == CV_32F ? (ippiNormDiffFuncHint)ippiNormDiff_L1_32f_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_32F ? (ippiNormDiffFuncHint)ippiNormDiff_L2_32f_C1R :
                0) : 0;
            ippiNormDiffFuncNoHint ippiNormDiff =
                normType == NORM_INF ?
                (type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_8u_C1R :
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_16s_C1R :
                type == CV_32F ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_32f_C1R :
                0) :
                normType == NORM_L1 ?
                (type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_8u_C1R :
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_16s_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_8u_C1R :
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_16s_C1R :
                0) : 0;
            if( ippiNormDiffHint || ippiNormDiff )
            {
                Ipp64f norm;
                IppStatus ret = ippiNormDiffHint ? CV_INSTRUMENT_FUN_IPP(ippiNormDiffHint, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm, ippAlgHintAccurate) :
                                CV_INSTRUMENT_FUN_IPP(ippiNormDiff, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm);
                if( ret >= 0 )
                {
                    result = (normType == NORM_L2SQR) ? norm * norm : norm;
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(_src1); CV_UNUSED(_src2); CV_UNUSED(normType); CV_UNUSED(_mask); CV_UNUSED(result);
#endif
    return false;
}  // ipp_norm
#endif  // HAVE_IPP


double norm( InputArray _src1, InputArray _src2, int normType, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    CV_CheckTypeEQ(_src1.type(), _src2.type(), "Input type mismatch");
    CV_Assert(_src1.sameSize(_src2));

#if defined HAVE_OPENCL || defined HAVE_IPP
    double _result = 0;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src1.isUMat()),
                ocl_norm(_src1, _src2, normType, _mask, _result),
                _result)
#endif

    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_norm(_src1, _src2, normType, _mask, _result), _result);

    if( normType & CV_RELATIVE )
    {
        return norm(_src1, _src2, normType & ~CV_RELATIVE, _mask)/(norm(_src2, normType, _mask) + DBL_EPSILON);
    }

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    int depth = src1.depth(), cn = src1.channels();

    normType &= 7;
    CV_Assert( normType == NORM_INF || normType == NORM_L1 ||
               normType == NORM_L2 || normType == NORM_L2SQR ||
              ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && src1.type() == CV_8U) );

    if( src1.isContinuous() && src2.isContinuous() && mask.empty() )
    {
        size_t len = src1.total()*src1.channels();
        if( len == (size_t)(int)len )
        {
            if( src1.depth() == CV_32F )
            {
                const float* data1 = src1.ptr<float>();
                const float* data2 = src2.ptr<float>();

                if( normType == NORM_L2 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL2_32f)(data1, data2, 0, &result, (int)len, 1);
                    return std::sqrt(result);
                }
                if( normType == NORM_L2SQR )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL2_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_L1 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL1_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_INF )
                {
                    float result = 0;
                    GET_OPTIMIZED(normDiffInf_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
            }
        }
    }

    CV_Assert( mask.empty() || mask.type() == CV_8U );

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_xor(src1, src2, temp);
            bitwise_and(temp, mask, temp);
            return norm(temp, normType);
        }
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        const Mat* arrays[] = {&src1, &src2, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        int total = (int)it.size;
        int result = 0;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            result += hal::normHamming(ptrs[0], ptrs[1], total, cellSize);
        }

        return result;
    }

    NormDiffFunc func = getNormDiffFunc(normType >> 1, depth == CV_16F ? CV_32F : depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src1, &src2, &mask, 0};
    uchar* ptrs[3] = {};
    union
    {
        double d;
        float f;
        int i;
        unsigned u;
    }
    result;
    result.d = 0;
    NAryMatIterator it(arrays, ptrs);
    CV_CheckLT((size_t)it.size, (size_t)INT_MAX, "");

    if ((normType == NORM_L1 && depth <= CV_16S) ||
        ((normType == NORM_L2 || normType == NORM_L2SQR) && depth <= CV_8S))
    {
        // special case to handle "integer" overflow in accumulator
        const size_t esz = src1.elemSize();
        const int total = (int)it.size;
        const int intSumBlockSize = (normType == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15))/cn;
        const int blockSize = std::min(total, intSumBlockSize);
        int isum = 0;
        int count = 0;

        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            for (int j = 0; j < total; j += blockSize)
            {
                int bsz = std::min(total - j, blockSize);
                func(ptrs[0], ptrs[1], ptrs[2], (uchar*)&isum, bsz, cn);
                count += bsz;
                if (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total))
                {
                    result.d += isum;
                    isum = 0;
                    count = 0;
                }
                ptrs[0] += bsz*esz;
                ptrs[1] += bsz*esz;
                if (ptrs[2])
                    ptrs[2] += bsz;
            }
        }
    }
    else if (depth == CV_16F)
    {
        const size_t esz = src1.elemSize();
        const int total = (int)it.size;
        const int blockSize = std::min(total, divUp(512, cn));
        AutoBuffer<float, 1026/*divUp(512,3)*3*2*/> fltbuf(blockSize * cn * 2);
        float* data0 = fltbuf.data();
        float* data1 = fltbuf.data() + blockSize * cn;
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            for (int j = 0; j < total; j += blockSize)
            {
                int bsz = std::min(total - j, blockSize);
                hal::cvt16f32f((const hfloat*)ptrs[0], data0, bsz * cn);
                hal::cvt16f32f((const hfloat*)ptrs[1], data1, bsz * cn);
                func((uchar*)data0, (uchar*)data1, ptrs[2], (uchar*)&result.f, bsz, cn);
                ptrs[0] += bsz*esz;
                ptrs[1] += bsz*esz;
                if (ptrs[2])
                    ptrs[2] += bsz;
            }
        }
    }
    else
    {
        // generic implementation
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            func(ptrs[0], ptrs[1], ptrs[2], (uchar*)&result, (int)it.size, cn);
        }
    }

    if( normType == NORM_INF )
    {
        if (depth == CV_64F)
            return result.d;
        else if (depth == CV_32F || depth == CV_16F)
            return result.f;
        else
            return result.u;
    }
    else if( normType == NORM_L2 )
        return std::sqrt(result.d);

    return result.d;
}

cv::Hamming::ResultType Hamming::operator()( const unsigned char* a, const unsigned char* b, int size ) const
{
    return cv::hal::normHamming(a, b, size);
}

double PSNR(InputArray _src1, InputArray _src2, double R)
{
    CV_INSTRUMENT_REGION();

    //Input arrays must have depth CV_8U
    CV_Assert( _src1.type() == _src2.type() );

    double diff = std::sqrt(norm(_src1, _src2, NORM_L2SQR)/(_src1.total()*_src1.channels()));
    return 20*log10(R/(diff+DBL_EPSILON));
}


#ifdef HAVE_OPENCL
static bool ocl_normalize( InputArray _src, InputOutputArray _dst, InputArray _mask, int dtype,
                           double scale, double delta )
{
    UMat src = _src.getUMat();

    if( _mask.empty() )
        src.convertTo( _dst, dtype, scale, delta );
    else if (src.channels() <= 4)
    {
        const ocl::Device & dev = ocl::Device::getDefault();

        int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
                ddepth = CV_MAT_DEPTH(dtype), wdepth = std::max(CV_32F, std::max(sdepth, ddepth)),
                rowsPerWI = dev.isIntel() ? 4 : 1;

        float fscale = static_cast<float>(scale), fdelta = static_cast<float>(delta);
        bool haveScale = std::fabs(scale - 1) > DBL_EPSILON,
                haveZeroScale = !(std::fabs(scale) > DBL_EPSILON),
                haveDelta = std::fabs(delta) > DBL_EPSILON,
                doubleSupport = dev.doubleFPConfig() > 0;

        if (!haveScale && !haveDelta && stype == dtype)
        {
            _src.copyTo(_dst, _mask);
            return true;
        }
        if (haveZeroScale)
        {
            _dst.setTo(Scalar(delta), _mask);
            return true;
        }

        if ((sdepth == CV_64F || ddepth == CV_64F) && !doubleSupport)
            return false;

        char cvt[2][50];
        String opts = format("-D srcT=%s -D dstT=%s -D convertToWT=%s -D cn=%d -D rowsPerWI=%d"
                             " -D convertToDT=%s -D workT=%s%s%s%s -D srcT1=%s -D dstT1=%s",
                             ocl::typeToStr(stype), ocl::typeToStr(dtype),
                             ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0], sizeof(cvt[0])), cn,
                             rowsPerWI, ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1], sizeof(cvt[1])),
                             ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)),
                             doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                             haveScale ? " -D HAVE_SCALE" : "",
                             haveDelta ? " -D HAVE_DELTA" : "",
                             ocl::typeToStr(sdepth), ocl::typeToStr(ddepth));

        ocl::Kernel k("normalizek", ocl::core::normalize_oclsrc, opts);
        if (k.empty())
            return false;

        UMat mask = _mask.getUMat(), dst = _dst.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
                dstarg = ocl::KernelArg::ReadWrite(dst);

        if (haveScale)
        {
            if (haveDelta)
                k.args(srcarg, maskarg, dstarg, fscale, fdelta);
            else
                k.args(srcarg, maskarg, dstarg, fscale);
        }
        else
        {
            if (haveDelta)
                k.args(srcarg, maskarg, dstarg, fdelta);
            else
                k.args(srcarg, maskarg, dstarg);
        }

        size_t globalsize[2] = { (size_t)src.cols, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
        return k.run(2, globalsize, NULL, false);
    }
    else
    {
        UMat temp;
        src.convertTo( temp, dtype, scale, delta );
        temp.copyTo( _dst, _mask );
    }

    return true;
}  // ocl_normalize
#endif  // HAVE_OPENCL


inline static bool rounds_outside_interval(double smin, double smax, float scale_float, float shift_float, double dmin, double dmax)
{
    return smin * scale_float + shift_float < dmin || smax * scale_float + shift_float > dmax;
}

void normalize(InputArray _src, InputOutputArray _dst, double a, double b,
               int norm_type, int rtype, InputArray _mask)
{
    CV_INSTRUMENT_REGION();

    double scale = 1, shift = 0;
    int type = _src.type(), depth = CV_MAT_DEPTH(type);

    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : depth;

    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxIdx( _src, &smin, &smax, 0, 0, _mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        if( rtype == CV_32F )
        {
            // Cast to float and enforce conditions
            float scale_float = static_cast<float>(scale);
            float shift_float = (float)dmin - (float)(smin * scale);

            // check if a rounding error will occur, we want the values to be in [dmin, dmax]
            if (rounds_outside_interval(smin, smax, scale_float, shift_float, dmin, dmax)) {
                while (rounds_outside_interval(smin, smax, scale_float, shift_float, dmin, dmax))
                {
                    scale_float = nextafterf(scale_float, -INFINITY);
                    shift_float = (float)dmin - (float)(smin * scale_float);
                }
            }            
            scale = scale_float;
            shift = shift_float;
        }
        else
            shift = dmin - smin*scale;
    }
    else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( _src, norm_type, _mask );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
        shift = 0;
    }
    else
        CV_Error( cv::Error::StsBadArg, "Unknown/unsupported norm type" );

    CV_OCL_RUN(_dst.isUMat(),
               ocl_normalize(_src, _dst, _mask, rtype, scale, shift))

    Mat src = _src.getMat();
    if ( _mask.empty() )
        src.convertTo( _dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( _dst, _mask );
    }
}

}  // namespace
