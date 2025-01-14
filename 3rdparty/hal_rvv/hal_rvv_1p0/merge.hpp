// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MERGE_HPP_INCLUDED
#define OPENCV_HAL_RVV_MERGE_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_merge8u
#define cv_hal_merge8u cv::cv_hal_rvv::merge8u
#undef cv_hal_merge16u
#define cv_hal_merge16u cv::cv_hal_rvv::merge16u
#undef cv_hal_merge32s
#define cv_hal_merge32s cv::cv_hal_rvv::merge32s
#undef cv_hal_merge64s
#define cv_hal_merge64s cv::cv_hal_rvv::merge64s

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int merge8u(const uchar** src, uchar* dst, int len, int cn ) {
    int k = cn % 4 ? cn % 4 : 4;
    int i = 0;
    int vl = __riscv_vsetvlmax_e8m1();
    if( k == 1 )
    {
        const uchar* src0 = src[0];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle8_v_u8m1(src0 + i, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn, sizeof(uchar)*cn, a, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++)
            dst[i*cn] = src0[i];
    }
    else if( k == 2 )
    {
        const uchar *src0 = src[0], *src1 = src[1];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle8_v_u8m1(src0 + i, vl);
            auto b = __riscv_vle8_v_u8m1(src1 + i, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn, sizeof(uchar)*cn, a, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn + 1, sizeof(uchar)*cn, b, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[i*cn] = src0[i];
            dst[i*cn+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const uchar *src0 = src[0], *src1 = src[1], *src2 = src[2];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle8_v_u8m1(src0 + i, vl);
            auto b = __riscv_vle8_v_u8m1(src1 + i, vl);
            auto c = __riscv_vle8_v_u8m1(src2 + i, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn, sizeof(uchar)*cn, a, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn + 1, sizeof(uchar)*cn, b, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn + 2, sizeof(uchar)*cn, c, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[i*cn] = src0[i];
            dst[i*cn+1] = src1[i];
            dst[i*cn+2] = src2[i];
        }
    }
    else
    {
        const uchar *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle8_v_u8m1(src0 + i, vl);
            auto b = __riscv_vle8_v_u8m1(src1 + i, vl);
            auto c = __riscv_vle8_v_u8m1(src2 + i, vl);
            auto d = __riscv_vle8_v_u8m1(src3 + i, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn, sizeof(uchar)*cn, a, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn + 1, sizeof(uchar)*cn, b, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn + 2, sizeof(uchar)*cn, c, vl);
            __riscv_vsse8_v_u8m1(dst + i*cn + 3, sizeof(uchar)*cn, d, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[i*cn] = src0[i];
            dst[i*cn+1] = src1[i];
            dst[i*cn+2] = src2[i];
            dst[i*cn+3] = src3[i];
        }
    }
    #if defined(__clang__)
    #pragma clang loop vectorize(disable)
    #endif
    for( ; k < cn; k += 4 )
    {
        const uchar *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        i = 0;
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle8_v_u8m1(src0 + i, vl);
            auto b = __riscv_vle8_v_u8m1(src1 + i, vl);
            auto c = __riscv_vle8_v_u8m1(src2 + i, vl);
            auto d = __riscv_vle8_v_u8m1(src3 + i, vl);
            __riscv_vsse8_v_u8m1(dst + k+i*cn, sizeof(uchar)*cn, a, vl);
            __riscv_vsse8_v_u8m1(dst + k+i*cn + 1, sizeof(uchar)*cn, b, vl);
            __riscv_vsse8_v_u8m1(dst + k+i*cn + 2, sizeof(uchar)*cn, c, vl);
            __riscv_vsse8_v_u8m1(dst + k+i*cn + 3, sizeof(uchar)*cn, d, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[k+i*cn] = src0[i];
            dst[k+i*cn+1] = src1[i];
            dst[k+i*cn+2] = src2[i];
            dst[k+i*cn+3] = src3[i];
        }
    }
    return CV_HAL_ERROR_OK;
}

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int merge16u(const ushort** src, ushort* dst, int len, int cn ) {
    int k = cn % 4 ? cn % 4 : 4;
    int i = 0;
    int vl = __riscv_vsetvlmax_e16m1();
    if( k == 1 )
    {
        const ushort* src0 = src[0];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle16_v_u16m1(src0 + i, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn, sizeof(ushort)*cn, a, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++)
            dst[i*cn] = src0[i];
    }
    else if( k == 2 )
    {
        const ushort *src0 = src[0], *src1 = src[1];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle16_v_u16m1(src0 + i, vl);
            auto b = __riscv_vle16_v_u16m1(src1 + i, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn, sizeof(ushort)*cn, a, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn + 1, sizeof(ushort)*cn, b, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[i*cn] = src0[i];
            dst[i*cn+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const ushort *src0 = src[0], *src1 = src[1], *src2 = src[2];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle16_v_u16m1(src0 + i, vl);
            auto b = __riscv_vle16_v_u16m1(src1 + i, vl);
            auto c = __riscv_vle16_v_u16m1(src2 + i, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn, sizeof(ushort)*cn, a, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn + 1, sizeof(ushort)*cn, b, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn + 2, sizeof(ushort)*cn, c, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[i*cn] = src0[i];
            dst[i*cn+1] = src1[i];
            dst[i*cn+2] = src2[i];
        }
    }
    else
    {
        const ushort *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle16_v_u16m1(src0 + i, vl);
            auto b = __riscv_vle16_v_u16m1(src1 + i, vl);
            auto c = __riscv_vle16_v_u16m1(src2 + i, vl);
            auto d = __riscv_vle16_v_u16m1(src3 + i, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn, sizeof(ushort)*cn, a, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn + 1, sizeof(ushort)*cn, b, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn + 2, sizeof(ushort)*cn, c, vl);
            __riscv_vsse16_v_u16m1(dst + i*cn + 3, sizeof(ushort)*cn, d, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++ )
        {
            dst[i*cn] = src0[i];
            dst[i*cn+1] = src1[i];
            dst[i*cn+2] = src2[i];
            dst[i*cn+3] = src3[i];
        }
    }
    #if defined(__clang__)
    #pragma clang loop vectorize(disable)
    #endif
    for( ; k < cn; k += 4 )
    {
        const uint16_t *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        i = 0;
        for( ; i <= len - vl; i += vl)
        {
            auto a = __riscv_vle16_v_u16m1(src0 + i, vl);
            auto b = __riscv_vle16_v_u16m1(src1 + i, vl);
            auto c = __riscv_vle16_v_u16m1(src2 + i, vl);
            auto d = __riscv_vle16_v_u16m1(src3 + i, vl);
            __riscv_vsse16_v_u16m1(dst + k+i*cn, sizeof(ushort)*cn, a, vl);
            __riscv_vsse16_v_u16m1(dst + k+i*cn + 1, sizeof(ushort)*cn, b, vl);
            __riscv_vsse16_v_u16m1(dst + k+i*cn + 2, sizeof(ushort)*cn, c, vl);
            __riscv_vsse16_v_u16m1(dst + k+i*cn + 3, sizeof(ushort)*cn, d, vl);
        }
        for( ; i < len; i++ )
        {
            dst[k+i*cn] = src0[i];
            dst[k+i*cn+1] = src1[i];
            dst[k+i*cn+2] = src2[i];
            dst[k+i*cn+3] = src3[i];
        }
    }
    return CV_HAL_ERROR_OK;
}

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int merge32s(const int** src, int* dst, int len, int cn ) {
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        const int* src0 = src[0];
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( i = j = 0; i < len; i++, j += cn )
            dst[j] = src0[i];
    }
    else if( k == 2 )
    {
        const int *src0 = src[0], *src1 = src[1];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const int *src0 = src[0], *src1 = src[1], *src2 = src[2];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const int *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    #if defined(__clang__)
    #pragma clang loop vectorize(disable)
    #endif
    for( ; k < cn; k += 4 )
    {
        const int *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    return CV_HAL_ERROR_OK;
}

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int merge64s(const int64** src, int64* dst, int len, int cn ) {
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        const int64* src0 = src[0];
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( i = j = 0; i < len; i++, j += cn )
            dst[j] = src0[i];
    }
    else if( k == 2 )
    {
        const int64 *src0 = src[0], *src1 = src[1];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const int64 *src0 = src[0], *src1 = src[1], *src2 = src[2];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const int64 *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    #if defined(__clang__)
    #pragma clang loop vectorize(disable)
    #endif
    for( ; k < cn; k += 4 )
    {
        const int64 *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    return CV_HAL_ERROR_OK;
}

}}

#endif
