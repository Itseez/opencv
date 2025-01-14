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

#include "precomp.hpp"

#include <vector>

#include "opencv2/core/hal/intrin.hpp"

#include "filter.hpp"

#include "opencv2/core/softfloat.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
template <typename RFT>
void GaussianBlurFixedPoint(const Mat& src, Mat& dst,
                            const RFT* fkx, int fkx_size,
                            const RFT* fky, int fky_size,
                            int borderType);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if defined(CV_CPU_BASELINE_MODE)
// included in dispatch.cpp
#else
#include "fixedpoint.inl.hpp"
#endif

namespace {

template <typename ET, typename FT>
void hlineSmooth1N(const ET* src, int cn, const FT* m, int, FT* dst, int len, int)
{
    for (int i = 0; i < len*cn; i++, src++, dst++)
        *dst = (*m) * (*src);
}
template <>
void hlineSmooth1N<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int)
{
    int lencn = len*cn;
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 vmul = vx_setall_u16(*((uint16_t*)m));
    for (; i <= lencn - VECSZ; i += VECSZ)
        v_store((uint16_t*)dst + i, v_mul(vmul, vx_load_expand(src + i)));
#endif
    for (; i < lencn; i++)
        dst[i] = m[0] * src[i];
}
template <typename ET, typename FT>
void hlineSmooth1N1(const ET* src, int cn, const FT*, int, FT* dst, int len, int)
{
    for (int i = 0; i < len*cn; i++, src++, dst++)
        *dst = *src;
}
template <>
void hlineSmooth1N1<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16*, int, ufixedpoint16* dst, int len, int)
{
    int lencn = len*cn;
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    for (; i <= lencn - VECSZ; i += VECSZ)
        v_store((uint16_t*)dst + i, v_shl<8>(vx_load_expand(src + i)));
#endif
    for (; i < lencn; i++)
        dst[i] = src[i];
}
template <typename ET, typename FT>
void hlineSmooth3N(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[1] * src[k] + m[2] * src[cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[0] * src[src_idx*cn + k];
        }

        src += cn; dst += cn;
        for (int i = cn; i < (len - 1)*cn; i++, src++, dst++)
            *dst = m[0] * src[-cn] + m[1] * src[0] + m[2] * src[cn];

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[2] * src[src_idx + k];
        }
    }
}
template <>
void hlineSmooth3N<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[1] * src[k] + m[2] * src[cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[0] * src[src_idx*cn + k];
        }

        src += cn; dst += cn;
        int i = cn, lencn = (len - 1)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = VTraits<v_uint16>::vlanes();
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        v_uint16 v_mul2 = vx_setall_u16(_m[2]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_add(v_add(v_mul(vx_load_expand(src - cn), v_mul0), v_mul(vx_load_expand(src), v_mul1)), v_mul(vx_load_expand(src + cn), v_mul2)));
#endif
        for (; i < lencn; i++, src++, dst++)
            *dst = m[0] * src[-cn] + m[1] * src[0] + m[2] * src[cn];

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[2] * src[src_idx + k];
        }
    }
}

template <typename ET, typename FT, typename VFT>
void hlineSmooth3N121Impl(const ET* src, int cn, const FT*, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        if(borderType != BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = FT(src[k]);
        else
            for (int k = 0; k < cn; k++)
                dst[k] = FT(src[k])>>1;
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = (FT(src[k])>>1) + (FT(src[cn + k])>>2);
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + (FT(src[src_idx*cn + k])>>2);
        }

        src += cn; dst += cn;
        int i = cn, lencn = (len - 1)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ = VTraits<VFT>::vlanes();
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((typename FT::raw_t*)dst, v_shl<(FT::fixedShift-2)>(v_add(vx_load_expand(src - cn), vx_load_expand(src + cn), v_shl<1>((vx_load_expand(src))))));
#endif
        for (; i < lencn; i++, src++, dst++)
            *dst = (FT(src[-cn])>>2) + (FT(src[cn])>>2) + (FT(src[0])>>1);

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = (FT(src[k - cn])>>2) + (FT(src[k])>>1);
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + (FT(src[src_idx + k])>>2);
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth3N121(const ET* src, int cn, const FT*, int, FT* dst, int len, int borderType);
template <>
void hlineSmooth3N121<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* _m, int _n, ufixedpoint16* dst, int len, int borderType)
{
    hlineSmooth3N121Impl<uint8_t, ufixedpoint16, v_uint16>(src, cn, _m, _n, dst, len, borderType);
}
template <>
void hlineSmooth3N121<uint16_t, ufixedpoint32>(const uint16_t* src, int cn, const ufixedpoint32* _m, int _n, ufixedpoint32* dst, int len, int borderType)
{
    hlineSmooth3N121Impl<uint16_t, ufixedpoint32, v_uint32>(src, cn, _m, _n, dst, len, borderType);
}

template <typename ET, typename FT>
void hlineSmooth3Naba(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? (m[0]<<1) + m[1] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[cn + k] + m[0] * src[src_idx*cn + k];
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[cn + k];
        }

        src += cn; dst += cn;
        for (int i = cn; i < (len - 1)*cn; i++, src++, dst++)
            *dst = m[1] * src[0] + m[0] * src[-cn] + m[0] * src[cn];

        // Point that fall right from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[k - cn] + m[0] * src[src_idx + k];
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        }
    }
}
template <>
void hlineSmooth3Naba<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? (m[0]<<1) + m[1] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[1] * (uint32_t)(src[k]) + ((uint16_t*)m)[0] * ((uint32_t)(src[cn + k]) + (uint32_t)(src[src_idx*cn + k])));
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[cn + k];
        }

        src += cn; dst += cn;
        int i = cn, lencn = (len - 1)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = VTraits<v_uint16>::vlanes();
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_add(v_mul(v_add(  vx_load_expand(src - cn), vx_load_expand(src + cn)),  v_mul0), v_mul(vx_load_expand(src), v_mul1)));
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = saturate_cast<uint16_t>(((uint16_t*)m)[1] * (uint32_t)(src[0]) + ((uint16_t*)m)[0] * ((uint32_t)(src[-cn]) + (uint32_t)(src[cn])));

        // Point that fall right from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[1] * (uint32_t)(src[k]) + ((uint16_t*)m)[0] * ((uint32_t)(src[k - cn]) + (uint32_t)(src[src_idx + k])));
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth5N(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] + m[3] + m[4] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k   ] = m[2] * src[k] + m[3] * src[k+cn];
                dst[k+cn] = m[1] * src[k] + m[2] * src[k+cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k     ] = m[1] * src[k + idxm1] + m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + idxp1] + m[0] * src[k + idxm2];
                dst[k + cn] = m[0] * src[k + idxm1] + m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k       ] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2*cn];
                dst[k +   cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2*cn];
                dst[k + 2*cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2*cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k       ] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2*cn] + m[0] * src[k + idxm2] + m[1] * src[k + idxm1];
                dst[k +   cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2*cn] + m[0] * src[k + idxm1] + m[4] * src[k + idxp1];
                dst[k + 2*cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2*cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[2] * src[k] + m[3] * src[cn + k] + m[4] * src[2*cn + k];
            dst[k + cn] = m[1] * src[k] + m[2] * src[cn + k] + m[3] * src[2*cn + k] + m[4] * src[3*cn + k];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxm2 + k] + m[1] * src[idxm1 + k];
                dst[k + cn] = dst[k + cn] + m[0] * src[idxm1 + k];
            }
        }

        src += 2*cn; dst += 2*cn;
        for (int i = 2*cn; i < (len - 2)*cn; i++, src++, dst++)
            *dst = m[0] * src[-2*cn] + m[1] * src[-cn] + m[2] * src[0] + m[3] * src[cn] + m[4] * src[2*cn];

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[0] * src[k - 2*cn] + m[1] * src[k - cn] + m[2] * src[k] + m[3] * src[k + cn];
            dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len+1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[4] * src[idxp1 + k];
                dst[k + cn] = dst[k + cn] + m[3] * src[idxp1 + k] + m[4] * src[idxp2 + k];
            }
        }
    }
}
template <>
void hlineSmooth5N<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] + m[3] + m[4] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[3] * src[k + cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[1] * src[k + idxm1] + m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + idxp1] + m[0] * src[k + idxm2];
                dst[k + cn] = m[0] * src[k + idxm1] + m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2 * cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2 * cn];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2 * cn] + m[0] * src[k + idxm2] + m[1] * src[k + idxm1];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2 * cn] + m[0] * src[k + idxm1] + m[4] * src[k + idxp1];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[2] * src[k] + m[3] * src[cn + k] + m[4] * src[2 * cn + k];
            dst[k + cn] = m[1] * src[k] + m[2] * src[cn + k] + m[3] * src[2 * cn + k] + m[4] * src[3 * cn + k];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxm2 + k] + m[1] * src[idxm1 + k];
                dst[k + cn] = dst[k + cn] + m[0] * src[idxm1 + k];
            }
        }

        src += 2 * cn; dst += 2 * cn;
        int i = 2*cn, lencn = (len - 2)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = VTraits<v_uint16>::vlanes();
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        v_uint16 v_mul2 = vx_setall_u16(_m[2]);
        v_uint16 v_mul3 = vx_setall_u16(_m[3]);
        v_uint16 v_mul4 = vx_setall_u16(_m[4]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_add(v_add(v_add(v_add(v_mul(vx_load_expand(src - 2 * cn), v_mul0), v_mul(vx_load_expand(src - cn), v_mul1)), v_mul(vx_load_expand(src), v_mul2)), v_mul(vx_load_expand(src + cn), v_mul3)), v_mul(vx_load_expand(src + 2 * cn), v_mul4)));
#endif
        for (; i < lencn; i++, src++, dst++)
            *dst = m[0] * src[-2*cn] + m[1] * src[-cn] + m[2] * src[0] + m[3] * src[cn] + m[4] * src[2*cn];

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[0] * src[k - 2 * cn] + m[1] * src[k - cn] + m[2] * src[k] + m[3] * src[k + cn];
            dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[4] * src[idxp1 + k];
                dst[k + cn] = dst[k + cn] + m[3] * src[idxp1 + k] + m[4] * src[idxp2 + k];
            }
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth5N14641(const ET* src, int cn, const FT*, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = (FT(src[k])>>3)*(uint8_t)3;
        else
            for (int k = 0; k < cn; k++)
                dst[k] = src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2);
                dst[k + cn] = (FT(src[k]) >> 2) + (FT(src[k + cn])>>4)*(uint8_t)6;
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + idxm1])>>2) + (FT(src[k + cn])>>2) + (FT(src[k + idxp1])>>4) + (FT(src[k + idxm2])>>4);
                dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k + idxp1])>>2) + (FT(src[k + idxm1])>>4) + (FT(src[k + idxp2])>>4);
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k + 2 * cn])>>4);
                dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k + 2 * cn])>>2);
                dst[k + 2 * cn] = (FT(src[k + 2 * cn])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k])>>4);
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k + idxm1])>>2) + (FT(src[k + 2 * cn])>>4) + (FT(src[k + idxm2])>>4);
                dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k + 2 * cn])>>2) + (FT(src[k + idxm1])>>4) + (FT(src[k + idxp1])>>4);
                dst[k + 2 * cn] = (FT(src[k + 2 * cn])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k + idxp1])>>2) + (FT(src[k])>>4) + (FT(src[k + idxp2])>>4);
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[cn + k])>>2) + (FT(src[2 * cn + k])>>4);
            dst[k + cn] = (FT(src[cn + k])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[2 * cn + k])>>2) + (FT(src[3 * cn + k])>>4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (FT(src[idxm2 + k])>>4) + (FT(src[idxm1 + k])>>2);
                dst[k + cn] = dst[k + cn] + (FT(src[idxm1 + k])>>4);
            }
        }

        src += 2 * cn; dst += 2 * cn;
        for (int i = 2 * cn; i < (len - 2)*cn; i++, src++, dst++)
            *dst = (FT(src[0])>>4)*(uint8_t)6 + (FT(src[-cn])>>2) + (FT(src[cn])>>2) + (FT(src[-2 * cn])>>4) + (FT(src[2 * cn])>>4);

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k - cn])>>2) + (FT(src[k + cn])>>2) + (FT(src[k - 2 * cn])>>4);
            dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k - cn])>>4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (FT(src[idxp1 + k])>>4);
                dst[k + cn] = dst[k + cn] + (FT(src[idxp1 + k])>>2) + (FT(src[idxp2 + k])>>4);
            }
        }
    }
}
template <>
void hlineSmooth5N14641<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16*, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = (ufixedpoint16(src[k])>>3) * (uint8_t)3;
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = src[k];
        }
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2);
                dst[k + cn] = (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6;
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + idxm1]) >> 2) + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + idxp1]) >> 4) + (ufixedpoint16(src[k + idxm2]) >> 4);
                dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + idxp1]) >> 2) + (ufixedpoint16(src[k + idxm1]) >> 4) + (ufixedpoint16(src[k + idxp2]) >> 4);
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 4);
                dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 2);
                dst[k + 2 * cn] = (ufixedpoint16(src[k + 2 * cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k]) >> 4);
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + idxm1]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 4) + (ufixedpoint16(src[k + idxm2]) >> 4);
                dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 2) + (ufixedpoint16(src[k + idxm1]) >> 4) + (ufixedpoint16(src[k + idxp1]) >> 4);
                dst[k + 2 * cn] = (ufixedpoint16(src[k + 2 * cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + idxp1]) >> 2) + (ufixedpoint16(src[k]) >> 4) + (ufixedpoint16(src[k + idxp2]) >> 4);
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[cn + k]) >> 2) + (ufixedpoint16(src[2 * cn + k]) >> 4);
            dst[k + cn] = (ufixedpoint16(src[cn + k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[2 * cn + k]) >> 2) + (ufixedpoint16(src[3 * cn + k]) >> 4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (ufixedpoint16(src[idxm2 + k]) >> 4) + (ufixedpoint16(src[idxm1 + k]) >> 2);
                dst[k + cn] = dst[k + cn] + (ufixedpoint16(src[idxm1 + k]) >> 4);
            }
        }

        src += 2 * cn; dst += 2 * cn;
        int i = 2 * cn, lencn = (len - 2)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ = VTraits<v_uint16>::vlanes();
        v_uint16 v_6 = vx_setall_u16(6);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_shl<4>(v_add(v_add(v_add(v_mul(vx_load_expand(src), v_6), v_shl<2>(v_add(vx_load_expand(src - cn), vx_load_expand(src + cn)))), vx_load_expand(src - 2 * cn)), vx_load_expand(src + 2 * cn))));
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = (uint16_t(src[0]) * 6 + ((uint16_t(src[-cn]) + uint16_t(src[cn])) << 2) + uint16_t(src[-2 * cn]) + uint16_t(src[2 * cn])) << 4;

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k - cn]) >> 2) + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k - 2 * cn]) >> 4);
            dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k - cn]) >> 4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (ufixedpoint16(src[idxp1 + k]) >> 4);
                dst[k + cn] = dst[k + cn] + (ufixedpoint16(src[idxp1 + k]) >> 2) + (ufixedpoint16(src[idxp2 + k]) >> 4);
            }
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth5Nabcba(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? ((m[0] + m[1])<<1) + m[2] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[1] * src[k + idxm1] + m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + idxp1] + m[0] * src[k + idxm2];
                dst[k + cn] = m[0] * src[k + idxm1] + m[1] * src[k] + m[2] * src[k + cn] + m[1] * src[k + idxp1] + m[0] * src[k + idxp2];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + 2 * cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[1] * src[k + 2 * cn];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + 2 * cn] + m[0] * src[k + idxm2] + m[1] * src[k + idxm1];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[1] * src[k + 2 * cn] + m[0] * src[k + idxm1] + m[0] * src[k + idxp1];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn] + m[1] * src[k + idxp1] + m[0] * src[k + idxp2];
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[2] * src[k] + m[1] * src[cn + k] + m[0] * src[2 * cn + k];
            dst[k + cn] = m[1] * src[k] + m[2] * src[cn + k] + m[1] * src[2 * cn + k] + m[0] * src[3 * cn + k];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxm2 + k] + m[1] * src[idxm1 + k];
                dst[k + cn] = dst[k + cn] + m[0] * src[idxm1 + k];
            }
        }

        src += 2 * cn; dst += 2 * cn;
        for (int i = 2 * cn; i < (len - 2)*cn; i++, src++, dst++)
            *dst = m[0] * src[-2 * cn] + m[1] * src[-cn] + m[2] * src[0] + m[3] * src[cn] + m[4] * src[2 * cn];

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[0] * src[k - 2 * cn] + m[1] * src[k - cn] + m[2] * src[k] + m[3] * src[k + cn];
            dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxp1 + k];
                dst[k + cn] = dst[k + cn] + m[1] * src[idxp1 + k] + m[0] * src[idxp2 + k];
            }
        }
    }
}
template <>
void hlineSmooth5Nabcba<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? ((m[0] + m[1]) << 1) + m[2] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[1] * ((uint32_t)(src[k + idxm1]) + (uint32_t)(src[k + cn])) + ((uint16_t*)m)[2] * (uint32_t)(src[k]) + ((uint16_t*)m)[0] * ((uint32_t)(src[k + idxp1]) + (uint32_t)(src[k + idxm2])));
                ((uint16_t*)dst)[k + cn] = saturate_cast<uint16_t>(((uint16_t*)m)[0] * ((uint32_t)(src[k + idxm1]) + (uint32_t)(src[k + idxp2])) + ((uint16_t*)m)[1] * ((uint32_t)(src[k]) + (uint32_t)(src[k + idxp1])) + ((uint16_t*)m)[2] * (uint32_t)(src[k + cn]));
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + 2 * cn];
                ((uint16_t*)dst)[k + cn] = saturate_cast<uint16_t>(((uint16_t*)m)[1] * ((uint32_t)(src[k]) + (uint32_t)(src[k + 2 * cn])) + ((uint16_t*)m)[2] * (uint32_t)(src[k + cn]));
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[2] * (uint32_t)(src[k]) + ((uint16_t*)m)[1] * ((uint32_t)(src[k + cn]) + (uint32_t)(src[k + idxm1])) + ((uint16_t*)m)[0] * ((uint32_t)(src[k + 2 * cn]) + (uint32_t)(src[k + idxm2])));
                ((uint16_t*)dst)[k + cn] = saturate_cast<uint16_t>(((uint16_t*)m)[2] * (uint32_t)(src[k + cn]) + ((uint16_t*)m)[1] * ((uint32_t)(src[k]) + (uint32_t)(src[k + 2 * cn])) + ((uint16_t*)m)[0] * ((uint32_t)(src[k + idxm1]) + (uint32_t)(src[k + idxp1])));
                ((uint16_t*)dst)[k + 2 * cn] = saturate_cast<uint16_t>(((uint16_t*)m)[0] * ((uint32_t)(src[k]) + (uint32_t)(src[k + idxp2])) + ((uint16_t*)m)[1] * ((uint32_t)(src[k + cn]) + (uint32_t)(src[k + idxp1])) + ((uint16_t*)m)[2] * (uint32_t)(src[k + 2 * cn]));
            }
        }
    }
    else
    {
        // Points that fall left from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[2] * (uint32_t)(src[k]) + ((uint16_t*)m)[1] * ((uint32_t)(src[cn + k]) + (uint32_t)(src[idxm1 + k])) + ((uint16_t*)m)[0] * ((uint32_t)(src[2 * cn + k]) + (uint32_t)(src[idxm2 + k])));
                ((uint16_t*)dst)[k + cn] = saturate_cast<uint16_t>(((uint16_t*)m)[1] * ((uint32_t)(src[k]) + (uint32_t)(src[2 * cn + k])) + ((uint16_t*)m)[2] * (uint32_t)(src[cn + k]) + ((uint16_t*)m)[0] * ((uint32_t)(src[3 * cn + k]) + (uint32_t)(src[idxm1 + k])));
            }
        }
        else
        {
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[cn + k] + m[0] * src[2 * cn + k];
                ((uint16_t*)dst)[k + cn] = saturate_cast<uint16_t>(((uint16_t*)m)[1] * ((uint32_t)(src[k]) + (uint32_t)(src[2 * cn + k])) + ((uint16_t*)m)[2] * (uint32_t)(src[cn + k]) + ((uint16_t*)m)[0] * (uint32_t)(src[3 * cn + k]));
            }
        }

        src += 2 * cn; dst += 2 * cn;
        int i = 2 * cn, lencn = (len - 2)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = VTraits<v_uint16>::vlanes();
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        v_uint16 v_mul2 = vx_setall_u16(_m[2]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_add(v_add(v_mul(v_add(vx_load_expand(src - 2 * cn), vx_load_expand(src + 2 * cn)), v_mul0), v_mul(v_add(vx_load_expand(src - cn), vx_load_expand(src + cn)), v_mul1)), v_mul(vx_load_expand(src), v_mul2)));
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = saturate_cast<uint16_t>(((uint16_t*)m)[0] * ((uint32_t)(src[-2 * cn]) + (uint32_t)(src[2 * cn])) + ((uint16_t*)m)[1] * ((uint32_t)(src[-cn]) + (uint32_t)(src[cn])) + ((uint16_t*)m)[2] * (uint32_t)(src[0]));

        // Points that fall right from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[0] * ((uint32_t)(src[k - 2 * cn]) + (uint32_t)(src[idxp1 + k])) + ((uint16_t*)m)[1] * ((uint32_t)(src[k - cn]) + (uint32_t)(src[k + cn])) + ((uint16_t*)m)[2] * (uint32_t)(src[k]));
                ((uint16_t*)dst)[k + cn] = saturate_cast<uint16_t>(((uint16_t*)m)[0] * ((uint32_t)(src[k - cn]) + (uint32_t)(src[idxp2 + k])) + ((uint16_t*)m)[1] * ((uint32_t)(src[k]) + (uint32_t)(src[idxp1 + k])) + ((uint16_t*)m)[2] * (uint32_t)(src[k + cn]));
            }
        }
        else
        {
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = saturate_cast<uint16_t>(((uint16_t*)m)[0] * (uint32_t)(src[k - 2 * cn]) + ((uint16_t*)m)[1] * ((uint32_t)(src[k - cn]) + (uint32_t)(src[k + cn])) + ((uint16_t*)m)[2] * (uint32_t)(src[k]));
                dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
            }
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth(const ET* src, int cn, const FT* m, int n, FT* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift-i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    for (; i < (len - post_shift + 1)*cn; i++, src++, dst++)
    {
        *dst = m[0] * src[0];
        for (int j = 1; j < n; j++)
            *dst = *dst + m[j] * src[j*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <>
void hlineSmooth<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int n, ufixedpoint16* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    int lencn = (len - post_shift + 1)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    for (; i <= lencn - VECSZ; i+=VECSZ, src+=VECSZ, dst+=VECSZ)
    {
        v_uint16 v_res0 = v_mul(vx_load_expand(src), vx_setall_u16(*((uint16_t*)m)));
        for (int j = 1; j < n; j++)
            v_res0 = v_add(v_res0, v_mul(vx_load_expand(src + j * cn), vx_setall_u16(*((uint16_t *)(m + j)))));
        v_store((uint16_t*)dst, v_res0);
    }
#endif
    for (; i < lencn; i++, src++, dst++)
    {
            *dst = m[0] * src[0];
            for (int j = 1; j < n; j++)
                *dst = *dst + m[j] * src[j*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <typename ET, typename FT>
void hlineSmoothONa_yzy_a(const ET* src, int cn, const FT* m, int n, FT* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    for (; i < (len - post_shift + 1)*cn; i++, src++, dst++)
    {
        *dst = m[pre_shift] * src[pre_shift*cn];
        for (int j = 0; j < pre_shift; j++)
            *dst = *dst + m[j] * src[j*cn] + m[j] * src[(n-1-j)*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <>
void hlineSmoothONa_yzy_a<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int n, ufixedpoint16* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    int lencn = (len - post_shift + 1)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
    {
        v_uint16 v_res0 = v_mul(vx_load_expand(src + pre_shift * cn), vx_setall_u16(*((uint16_t*)(m + pre_shift))));
        for (int j = 0; j < pre_shift; j ++)
            v_res0 = v_add(v_res0, v_mul(v_add(vx_load_expand(src + j * cn), vx_load_expand(src + (n - 1 - j) * cn)), vx_setall_u16(*((uint16_t *)(m + j)))));
        v_store((uint16_t*)dst, v_res0);
    }
#endif
    for (; i < lencn; i++, src++, dst++)
    {
        *dst = m[pre_shift] * src[pre_shift*cn];
        for (int j = 0; j < pre_shift; j++)
            *dst = *dst + m[j] * src[j*cn] + m[j] * src[(n - 1 - j)*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <>
void hlineSmoothONa_yzy_a<uint16_t, ufixedpoint32>(const uint16_t* src, int cn, const ufixedpoint32* m, int n, ufixedpoint32* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    int lencn = (len - post_shift + 1)*cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint32>::vlanes();
    for (; i <= lencn - VECSZ * 2; i += VECSZ * 2, src += VECSZ * 2, dst += VECSZ * 2)
    {
        v_uint32 v_res0, v_res1;
        v_mul_expand(vx_load(src + pre_shift * cn), vx_setall_u16((uint16_t) *((uint32_t*)(m + pre_shift))), v_res0, v_res1);
        for (int j = 0; j < pre_shift; j ++)
        {
            v_uint16 v_weight = vx_setall_u16((uint16_t) *((uint32_t*)(m + j)));
            v_uint32 v_add0, v_add1;
            v_mul_expand(vx_load(src + j * cn), v_weight, v_add0, v_add1);
            v_res0 = v_add(v_res0, v_add0);
            v_res1 = v_add(v_res1, v_add1);
            v_mul_expand(vx_load(src + (n - 1 - j)*cn), v_weight, v_add0, v_add1);
            v_res0 = v_add(v_res0, v_add0);
            v_res1 = v_add(v_res1, v_add1);
        }
        v_store((uint32_t*)dst, v_res0);
        v_store((uint32_t*)dst + VECSZ, v_res1);
    }
#endif
    for (; i < lencn; i++, src++, dst++)
    {
        *dst = m[pre_shift] * src[pre_shift*cn];
        for (int j = 0; j < pre_shift; j++)
            *dst = *dst + m[j] * src[j*cn] + m[j] * src[(n - 1 - j)*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <typename ET, typename FT>
void vlineSmooth1N(const FT* const * src, const FT* m, int, ET* dst, int len)
{
    const FT* src0 = src[0];
    for (int i = 0; i < len; i++)
        dst[i] = *m * src0[i];
}
template <>
void vlineSmooth1N<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int, uint8_t* dst, int len)
{
    const ufixedpoint16* src0 = src[0];
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_mul = vx_setall_u16(*((uint16_t*)m)<<1);
    for (; i <= len - VECSZ; i += VECSZ)
        v_rshr_pack_store<1>(dst + i, v_mul_hi(vx_load((uint16_t*)src0 + i), v_mul));
#endif
    for (; i < len; i++)
        dst[i] = m[0] * src0[i];
}
template <typename ET, typename FT>
void vlineSmooth1N1(const FT* const * src, const FT*, int, ET* dst, int len)
{
    const FT* src0 = src[0];
    for (int i = 0; i < len; i++)
        dst[i] = src0[i];
}
template <>
void vlineSmooth1N1<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16*, int, uint8_t* dst, int len)
{
    const ufixedpoint16* src0 = src[0];
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    for (; i <= len - VECSZ; i += VECSZ)
        v_rshr_pack_store<8>(dst + i, vx_load((uint16_t*)(src0 + i)));
#endif
    for (; i < len; i++)
        dst[i] = src0[i];
}
template <typename ET, typename FT>
void vlineSmooth3N(const FT* const * src, const FT* m, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i];
}
template <>
void vlineSmooth3N<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int, uint8_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
    v_int32 v_128_4 = vx_setall_s32(128 << 16);
    const int VECSZ = VTraits<v_uint16>::vlanes();
    if (len >= VECSZ)
    {
        ufixedpoint32 val[] = { (m[0] + m[1] + m[2]) * ufixedpoint16((uint8_t)128) };
        v_128_4 = vx_setall_s32(*((int32_t*)val));
    }
    uint32_t val01;
    std::memcpy(&val01, m, sizeof(val01));
    v_int16 v_mul01 = v_reinterpret_as_s16(vx_setall_u32(val01));
    v_int16 v_mul2 = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + 2))));
    for (; i <= len - 4*VECSZ; i += 4*VECSZ)
    {
        v_int16 v_src00, v_src10, v_src01, v_src11, v_src02, v_src12, v_src03, v_src13;
        v_int16 v_tmp0, v_tmp1;

        const int16_t* src0 = (const int16_t*)src[0] + i;
        const int16_t* src1 = (const int16_t*)src[1] + i;
        v_src00 = vx_load(src0);
        v_src01 = vx_load(src0 + VECSZ);
        v_src02 = vx_load(src0 + 2*VECSZ);
        v_src03 = vx_load(src0 + 3*VECSZ);
        v_src10 = vx_load(src1);
        v_src11 = vx_load(src1 + VECSZ);
        v_src12 = vx_load(src1 + 2*VECSZ);
        v_src13 = vx_load(src1 + 3*VECSZ);
        v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
        v_int32 v_res0 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res1 = v_dotprod(v_tmp1, v_mul01);
        v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
        v_int32 v_res2 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res3 = v_dotprod(v_tmp1, v_mul01);
        v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
        v_int32 v_res4 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res5 = v_dotprod(v_tmp1, v_mul01);
        v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
        v_int32 v_res6 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res7 = v_dotprod(v_tmp1, v_mul01);

        v_int32 v_resj0, v_resj1;
        const int16_t* src2 = (const int16_t*)src[2] + i;
        v_src00 = vx_load(src2);
        v_src01 = vx_load(src2 + VECSZ);
        v_src02 = vx_load(src2 + 2*VECSZ);
        v_src03 = vx_load(src2 + 3*VECSZ);
        v_mul_expand(v_add_wrap(v_src00, v_128), v_mul2, v_resj0, v_resj1);
        v_res0 = v_add(v_res0, v_resj0);
        v_res1 = v_add(v_res1, v_resj1);
        v_mul_expand(v_add_wrap(v_src01, v_128), v_mul2, v_resj0, v_resj1);
        v_res2 = v_add(v_res2, v_resj0);
        v_res3 = v_add(v_res3, v_resj1);
        v_mul_expand(v_add_wrap(v_src02, v_128), v_mul2, v_resj0, v_resj1);
        v_res4 = v_add(v_res4, v_resj0);
        v_res5 = v_add(v_res5, v_resj1);
        v_mul_expand(v_add_wrap(v_src03, v_128), v_mul2, v_resj0, v_resj1);
        v_res6 = v_add(v_res6, v_resj0);
        v_res7 = v_add(v_res7, v_resj1);

        v_res0 = v_add(v_res0, v_128_4);
        v_res1 = v_add(v_res1, v_128_4);
        v_res2 = v_add(v_res2, v_128_4);
        v_res3 = v_add(v_res3, v_128_4);
        v_res4 = v_add(v_res4, v_128_4);
        v_res5 = v_add(v_res5, v_128_4);
        v_res6 = v_add(v_res6, v_128_4);
        v_res7 = v_add(v_res7, v_128_4);

        v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
        v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
    }
#endif
    for (; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i];
}
template <typename ET, typename FT>
void vlineSmooth3N121(const FT* const * src, const FT*, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = (FT::WT(src[0][i]) >> 2) + (FT::WT(src[2][i]) >> 2) + (FT::WT(src[1][i]) >> 1);
}
template <>
void vlineSmooth3N121<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16*, int, uint8_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint32 v_src00, v_src01, v_src02, v_src03, v_src10, v_src11, v_src12, v_src13, v_src20, v_src21, v_src22, v_src23;
        v_expand(vx_load((uint16_t*)(src[0]) + i), v_src00, v_src01);
        v_expand(vx_load((uint16_t*)(src[0]) + i + VECSZ), v_src02, v_src03);
        v_expand(vx_load((uint16_t*)(src[1]) + i), v_src10, v_src11);
        v_expand(vx_load((uint16_t*)(src[1]) + i + VECSZ), v_src12, v_src13);
        v_expand(vx_load((uint16_t*)(src[2]) + i), v_src20, v_src21);
        v_expand(vx_load((uint16_t*)(src[2]) + i + VECSZ), v_src22, v_src23);
        v_store(dst + i, v_pack(v_rshr_pack<10>(v_add(v_add(v_src00, v_src20), v_add(v_src10, v_src10)), v_add(v_add(v_src01, v_src21), v_add(v_src11, v_src11))),
                                v_rshr_pack<10>(v_add(v_add(v_src02, v_src22), v_add(v_src12, v_src12)), v_add(v_add(v_src03, v_src23), v_add(v_src13, v_src13)))));
    }
#endif
    for (; i < len; i++)
        dst[i] = (((uint32_t)(((uint16_t*)(src[0]))[i]) + (uint32_t)(((uint16_t*)(src[2]))[i]) + ((uint32_t)(((uint16_t*)(src[1]))[i]) << 1)) + (1 << 9)) >> 10;
}
template <>
void vlineSmooth3N121<uint16_t, ufixedpoint32>(const ufixedpoint32* const * src, const ufixedpoint32*, int, uint16_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint32>::vlanes();
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint64 v_src00, v_src01, v_src02, v_src03, v_src10, v_src11, v_src12, v_src13, v_src20, v_src21, v_src22, v_src23;
        v_expand(vx_load((uint32_t*)(src[0]) + i), v_src00, v_src01);
        v_expand(vx_load((uint32_t*)(src[0]) + i + VECSZ), v_src02, v_src03);
        v_expand(vx_load((uint32_t*)(src[1]) + i), v_src10, v_src11);
        v_expand(vx_load((uint32_t*)(src[1]) + i + VECSZ), v_src12, v_src13);
        v_expand(vx_load((uint32_t*)(src[2]) + i), v_src20, v_src21);
        v_expand(vx_load((uint32_t*)(src[2]) + i + VECSZ), v_src22, v_src23);
        v_store(dst + i, v_pack(v_rshr_pack<18>(v_add(v_add(v_src00, v_src20), v_add(v_src10, v_src10)), v_add(v_add(v_src01, v_src21), v_add(v_src11, v_src11))),
                                v_rshr_pack<18>(v_add(v_add(v_src02, v_src22), v_add(v_src12, v_src12)), v_add(v_add(v_src03, v_src23), v_add(v_src13, v_src13)))));
    }
#endif
    for (; i < len; i++)
        dst[i] = (((uint64_t)((uint32_t*)(src[0]))[i]) + (uint64_t)(((uint32_t*)(src[2]))[i]) + ((uint64_t(((uint32_t*)(src[1]))[i]) << 1)) + (1 << 17)) >> 18;
}
template <typename ET, typename FT>
void vlineSmooth5N(const FT* const * src, const FT* m, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i] + m[3] * src[3][i] + m[4] * src[4][i];
}
template <>
void vlineSmooth5N<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int, uint8_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    if (len >= 4 * VECSZ)
    {
        ufixedpoint32 val[] = { (m[0] + m[1] + m[2] + m[3] + m[4]) * ufixedpoint16((uint8_t)128) };
        v_int32 v_128_4 = vx_setall_s32(*((int32_t*)val));
        const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
        v_int16 v_mul01 = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)m)));
        v_int16 v_mul23 = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)(m + 2))));
        v_int16 v_mul4 = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + 4))));
        for (; i <= len - 4*VECSZ; i += 4*VECSZ)
        {
            v_int16 v_src00, v_src10, v_src01, v_src11, v_src02, v_src12, v_src03, v_src13;
            v_int16 v_tmp0, v_tmp1;

            const int16_t* src0 = (const int16_t*)src[0] + i;
            const int16_t* src1 = (const int16_t*)src[1] + i;
            v_src00 = vx_load(src0);
            v_src01 = vx_load(src0 + VECSZ);
            v_src02 = vx_load(src0 + 2*VECSZ);
            v_src03 = vx_load(src0 + 3*VECSZ);
            v_src10 = vx_load(src1);
            v_src11 = vx_load(src1 + VECSZ);
            v_src12 = vx_load(src1 + 2*VECSZ);
            v_src13 = vx_load(src1 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
            v_int32 v_res0 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res1 = v_dotprod(v_tmp1, v_mul01);
            v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
            v_int32 v_res2 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res3 = v_dotprod(v_tmp1, v_mul01);
            v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
            v_int32 v_res4 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res5 = v_dotprod(v_tmp1, v_mul01);
            v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
            v_int32 v_res6 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res7 = v_dotprod(v_tmp1, v_mul01);

            const int16_t* src2 = (const int16_t*)src[2] + i;
            const int16_t* src3 = (const int16_t*)src[3] + i;
            v_src00 = vx_load(src2);
            v_src01 = vx_load(src2 + VECSZ);
            v_src02 = vx_load(src2 + 2*VECSZ);
            v_src03 = vx_load(src2 + 3*VECSZ);
            v_src10 = vx_load(src3);
            v_src11 = vx_load(src3 + VECSZ);
            v_src12 = vx_load(src3 + 2*VECSZ);
            v_src13 = vx_load(src3 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
            v_res0 = v_add(v_res0, v_dotprod(v_tmp0, v_mul23));
            v_res1 = v_add(v_res1, v_dotprod(v_tmp1, v_mul23));
            v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
            v_res2 = v_add(v_res2, v_dotprod(v_tmp0, v_mul23));
            v_res3 = v_add(v_res3, v_dotprod(v_tmp1, v_mul23));
            v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
            v_res4 = v_add(v_res4, v_dotprod(v_tmp0, v_mul23));
            v_res5 = v_add(v_res5, v_dotprod(v_tmp1, v_mul23));
            v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
            v_res6 = v_add(v_res6, v_dotprod(v_tmp0, v_mul23));
            v_res7 = v_add(v_res7, v_dotprod(v_tmp1, v_mul23));

            v_int32 v_resj0, v_resj1;
            const int16_t* src4 = (const int16_t*)src[4] + i;
            v_src00 = vx_load(src4);
            v_src01 = vx_load(src4 + VECSZ);
            v_src02 = vx_load(src4 + 2*VECSZ);
            v_src03 = vx_load(src4 + 3*VECSZ);
            v_mul_expand(v_add_wrap(v_src00, v_128), v_mul4, v_resj0, v_resj1);
            v_res0 = v_add(v_res0, v_resj0);
            v_res1 = v_add(v_res1, v_resj1);
            v_mul_expand(v_add_wrap(v_src01, v_128), v_mul4, v_resj0, v_resj1);
            v_res2 = v_add(v_res2, v_resj0);
            v_res3 = v_add(v_res3, v_resj1);
            v_mul_expand(v_add_wrap(v_src02, v_128), v_mul4, v_resj0, v_resj1);
            v_res4 = v_add(v_res4, v_resj0);
            v_res5 = v_add(v_res5, v_resj1);
            v_mul_expand(v_add_wrap(v_src03, v_128), v_mul4, v_resj0, v_resj1);
            v_res6 = v_add(v_res6, v_resj0);
            v_res7 = v_add(v_res7, v_resj1);

            v_res0 = v_add(v_res0, v_128_4);
            v_res1 = v_add(v_res1, v_128_4);
            v_res2 = v_add(v_res2, v_128_4);
            v_res3 = v_add(v_res3, v_128_4);
            v_res4 = v_add(v_res4, v_128_4);
            v_res5 = v_add(v_res5, v_128_4);
            v_res6 = v_add(v_res6, v_128_4);
            v_res7 = v_add(v_res7, v_128_4);

            v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                              v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
            v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                              v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
        }
    }
#endif
    for (; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i] + m[3] * src[3][i] + m[4] * src[4][i];
}
template <typename ET, typename FT>
void vlineSmooth5N14641(const FT* const * src, const FT*, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = (FT::WT(src[2][i])*(uint8_t)6 + ((FT::WT(src[1][i]) + FT::WT(src[3][i]))<<2) + FT::WT(src[0][i]) + FT::WT(src[4][i])) >> 4;
}
template <>
void vlineSmooth5N14641<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16*, int, uint8_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_uint32 v_6 = vx_setall_u32(6);
    const int VECSZ = VTraits<v_uint16>::vlanes();
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint32 v_src00, v_src10, v_src20, v_src30, v_src40;
        v_uint32 v_src01, v_src11, v_src21, v_src31, v_src41;
        v_uint32 v_src02, v_src12, v_src22, v_src32, v_src42;
        v_uint32 v_src03, v_src13, v_src23, v_src33, v_src43;
        v_expand(vx_load((uint16_t*)(src[0]) + i), v_src00, v_src01);
        v_expand(vx_load((uint16_t*)(src[0]) + i + VECSZ), v_src02, v_src03);
        v_expand(vx_load((uint16_t*)(src[1]) + i), v_src10, v_src11);
        v_expand(vx_load((uint16_t*)(src[1]) + i + VECSZ), v_src12, v_src13);
        v_expand(vx_load((uint16_t*)(src[2]) + i), v_src20, v_src21);
        v_expand(vx_load((uint16_t*)(src[2]) + i + VECSZ), v_src22, v_src23);
        v_expand(vx_load((uint16_t*)(src[3]) + i), v_src30, v_src31);
        v_expand(vx_load((uint16_t*)(src[3]) + i + VECSZ), v_src32, v_src33);
        v_expand(vx_load((uint16_t*)(src[4]) + i), v_src40, v_src41);
        v_expand(vx_load((uint16_t*)(src[4]) + i + VECSZ), v_src42, v_src43);
        v_store(dst + i, v_pack(v_rshr_pack<12>(v_add(v_add(v_add(v_mul(v_src20, v_6), v_shl<2>(v_add(v_src10, v_src30))), v_src00), v_src40),
                                                v_add(v_add(v_add(v_mul(v_src21, v_6), v_shl<2>(v_add(v_src11, v_src31))), v_src01), v_src41)),
                                v_rshr_pack<12>(v_add(v_add(v_add(v_mul(v_src22, v_6), v_shl<2>(v_add(v_src12, v_src32))), v_src02), v_src42),
                                                v_add(v_add(v_add(v_mul(v_src23, v_6), v_shl<2>(v_add(v_src13, v_src33))), v_src03), v_src43))));
    }
#endif
    for (; i < len; i++)
        dst[i] = ((uint32_t)(((uint16_t*)(src[2]))[i]) * 6 +
                  (((uint32_t)(((uint16_t*)(src[1]))[i]) + (uint32_t)(((uint16_t*)(src[3]))[i])) << 2) +
                  (uint32_t)(((uint16_t*)(src[0]))[i]) + (uint32_t)(((uint16_t*)(src[4]))[i]) + (1 << 11)) >> 12;
}
template <>
void vlineSmooth5N14641<uint16_t, ufixedpoint32>(const ufixedpoint32* const * src, const ufixedpoint32*, int, uint16_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint32>::vlanes();
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint64 v_src00, v_src10, v_src20, v_src30, v_src40;
        v_uint64 v_src01, v_src11, v_src21, v_src31, v_src41;
        v_uint64 v_src02, v_src12, v_src22, v_src32, v_src42;
        v_uint64 v_src03, v_src13, v_src23, v_src33, v_src43;
        v_expand(vx_load((uint32_t*)(src[0]) + i), v_src00, v_src01);
        v_expand(vx_load((uint32_t*)(src[0]) + i + VECSZ), v_src02, v_src03);
        v_expand(vx_load((uint32_t*)(src[1]) + i), v_src10, v_src11);
        v_expand(vx_load((uint32_t*)(src[1]) + i + VECSZ), v_src12, v_src13);
        v_expand(vx_load((uint32_t*)(src[2]) + i), v_src20, v_src21);
        v_expand(vx_load((uint32_t*)(src[2]) + i + VECSZ), v_src22, v_src23);
        v_expand(vx_load((uint32_t*)(src[3]) + i), v_src30, v_src31);
        v_expand(vx_load((uint32_t*)(src[3]) + i + VECSZ), v_src32, v_src33);
        v_expand(vx_load((uint32_t*)(src[4]) + i), v_src40, v_src41);
        v_expand(vx_load((uint32_t*)(src[4]) + i + VECSZ), v_src42, v_src43);
        v_store(dst + i, v_pack(v_rshr_pack<20>(v_add(v_add(v_add(v_add(v_shl<2>(v_src20), v_shl<1>(v_src20)), v_shl<2>(v_add(v_src10, v_src30))), v_src00), v_src40),
                                                v_add(v_add(v_add(v_add(v_shl<2>(v_src21), v_shl<1>(v_src21)), v_shl<2>(v_add(v_src11, v_src31))), v_src01), v_src41)),
                                v_rshr_pack<20>(v_add(v_add(v_add(v_add(v_shl<2>(v_src22), v_shl<1>(v_src22)), v_shl<2>(v_add(v_src12, v_src32))), v_src02), v_src42),
                                                v_add(v_add(v_add(v_add(v_shl<2>(v_src23), v_shl<1>(v_src23)), v_shl<2>(v_add(v_src13, v_src33))), v_src03), v_src43))));
    }
#endif
    for (; i < len; i++)
        dst[i] = ((uint64_t)(((uint32_t*)(src[2]))[i]) * 6 +
                  (((uint64_t)(((uint32_t*)(src[1]))[i]) + (uint64_t)(((uint32_t*)(src[3]))[i])) << 2) +
                  (uint64_t)(((uint32_t*)(src[0]))[i]) + (uint64_t)(((uint32_t*)(src[4]))[i]) + (1 << 19)) >> 20;
}
template <typename ET, typename FT>
void vlineSmooth(const FT* const * src, const FT* m, int n, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
    {
        typename FT::WT val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
            val = val + m[j] * src[j][i];
        dst[i] = val;
    }
}

inline uint32_t read_pair_as_u32(const ufixedpoint16 * mem)
{
    union Cv32sufX2 { uint32_t v32; int16_t v16[2]; } res;
    res.v16[0] = mem->raw();
    res.v16[1] = (mem + 1)->raw();
    return res.v32;
}

template <>
void vlineSmooth<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int n, uint8_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
    v_int32 v_128_4 = vx_setall_s32(128 << 16);
    const int VECSZ = VTraits<v_uint16>::vlanes();
    if (len >= VECSZ)
    {
        ufixedpoint16 msum = m[0] + m[1];
        for (int j = 2; j < n; j++)
            msum = msum + m[j];
        ufixedpoint32 val[] = { msum * ufixedpoint16((uint8_t)128) };
        v_128_4 = vx_setall_s32(*((int32_t*)val));
    }
    for (; i <= len - 4*VECSZ; i += 4*VECSZ)
    {
        v_int16 v_src00, v_src10, v_src01, v_src11, v_src02, v_src12, v_src03, v_src13;
        v_int16 v_tmp0, v_tmp1;

        v_int16 v_mul = v_reinterpret_as_s16(vx_setall_u32(read_pair_as_u32(m)));

        const int16_t* src0 = (const int16_t*)src[0] + i;
        const int16_t* src1 = (const int16_t*)src[1] + i;
        v_src00 = vx_load(src0);
        v_src01 = vx_load(src0 + VECSZ);
        v_src02 = vx_load(src0 + 2*VECSZ);
        v_src03 = vx_load(src0 + 3*VECSZ);
        v_src10 = vx_load(src1);
        v_src11 = vx_load(src1 + VECSZ);
        v_src12 = vx_load(src1 + 2*VECSZ);
        v_src13 = vx_load(src1 + 3*VECSZ);
        v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
        v_int32 v_res0 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res1 = v_dotprod(v_tmp1, v_mul);
        v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
        v_int32 v_res2 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res3 = v_dotprod(v_tmp1, v_mul);
        v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
        v_int32 v_res4 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res5 = v_dotprod(v_tmp1, v_mul);
        v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
        v_int32 v_res6 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res7 = v_dotprod(v_tmp1, v_mul);

        int j = 2;
        for (; j < n - 1; j+=2)
        {
            v_mul = v_reinterpret_as_s16(vx_setall_u32(read_pair_as_u32(m + j)));

            const int16_t* srcj0 = (const int16_t*)src[j] + i;
            const int16_t* srcj1 = (const int16_t*)src[j + 1] + i;
            v_src00 = vx_load(srcj0);
            v_src01 = vx_load(srcj0 + VECSZ);
            v_src02 = vx_load(srcj0 + 2*VECSZ);
            v_src03 = vx_load(srcj0 + 3*VECSZ);
            v_src10 = vx_load(srcj1);
            v_src11 = vx_load(srcj1 + VECSZ);
            v_src12 = vx_load(srcj1 + 2*VECSZ);
            v_src13 = vx_load(srcj1 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
            v_res0 = v_add(v_res0, v_dotprod(v_tmp0, v_mul));
            v_res1 = v_add(v_res1, v_dotprod(v_tmp1, v_mul));
            v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
            v_res2 = v_add(v_res2, v_dotprod(v_tmp0, v_mul));
            v_res3 = v_add(v_res3, v_dotprod(v_tmp1, v_mul));
            v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
            v_res4 = v_add(v_res4, v_dotprod(v_tmp0, v_mul));
            v_res5 = v_add(v_res5, v_dotprod(v_tmp1, v_mul));
            v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
            v_res6 = v_add(v_res6, v_dotprod(v_tmp0, v_mul));
            v_res7 = v_add(v_res7, v_dotprod(v_tmp1, v_mul));
        }
        if(j < n)
        {
            v_int32 v_resj0, v_resj1;
            v_mul = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + j))));
            const int16_t* srcj = (const int16_t*)src[j] + i;
            v_src00 = vx_load(srcj);
            v_src01 = vx_load(srcj + VECSZ);
            v_src02 = vx_load(srcj + 2*VECSZ);
            v_src03 = vx_load(srcj + 3*VECSZ);
            v_mul_expand(v_add_wrap(v_src00, v_128), v_mul, v_resj0, v_resj1);
            v_res0 = v_add(v_res0, v_resj0);
            v_res1 = v_add(v_res1, v_resj1);
            v_mul_expand(v_add_wrap(v_src01, v_128), v_mul, v_resj0, v_resj1);
            v_res2 = v_add(v_res2, v_resj0);
            v_res3 = v_add(v_res3, v_resj1);
            v_mul_expand(v_add_wrap(v_src02, v_128), v_mul, v_resj0, v_resj1);
            v_res4 = v_add(v_res4, v_resj0);
            v_res5 = v_add(v_res5, v_resj1);
            v_mul_expand(v_add_wrap(v_src03, v_128), v_mul, v_resj0, v_resj1);
            v_res6 = v_add(v_res6, v_resj0);
            v_res7 = v_add(v_res7, v_resj1);
        }
        v_res0 = v_add(v_res0, v_128_4);
        v_res1 = v_add(v_res1, v_128_4);
        v_res2 = v_add(v_res2, v_128_4);
        v_res3 = v_add(v_res3, v_128_4);
        v_res4 = v_add(v_res4, v_128_4);
        v_res5 = v_add(v_res5, v_128_4);
        v_res6 = v_add(v_res6, v_128_4);
        v_res7 = v_add(v_res7, v_128_4);

        v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
        v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
    }
#endif
    for (; i < len; i++)
    {
        ufixedpoint32 val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
        {
            val = val + m[j] * src[j][i];
        }
        dst[i] = val;
    }
}
template <typename ET, typename FT>
void vlineSmoothONa_yzy_a(const FT* const * src, const FT* m, int n, ET* dst, int len)
{
    int pre_shift = n / 2;
    for (int i = 0; i < len; i++)
    {
        typename FT::WT val = m[pre_shift] * src[pre_shift][i];
        for (int j = 0; j < pre_shift; j++)
            val = val + m[j] * src[j][i] + m[j] * src[(n - 1 - j)][i];
        dst[i] = val;
    }
}
template <>
void vlineSmoothONa_yzy_a<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int n, uint8_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    int pre_shift = n / 2;
    const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
    v_int32 v_128_4 = vx_setall_s32(128 << 16);
    const int VECSZ = VTraits<v_uint16>::vlanes();
    if (len >= VECSZ)
    {
        ufixedpoint16 msum = m[0] + m[pre_shift] + m[n - 1];
        for (int j = 1; j < pre_shift; j++)
            msum = msum + m[j] + m[n - 1 - j];
        ufixedpoint32 val[] = { msum * ufixedpoint16((uint8_t)128) };
        v_128_4 = vx_setall_s32(*((int32_t*)val));
    }
    for (; i <= len - 4*VECSZ; i += 4*VECSZ)
    {
        v_int16 v_src00, v_src10, v_src20, v_src30, v_src01, v_src11, v_src21, v_src31;
        v_int32 v_res0, v_res1, v_res2, v_res3, v_res4, v_res5, v_res6, v_res7;
        v_int16 v_tmp0, v_tmp1, v_tmp2, v_tmp3, v_tmp4, v_tmp5, v_tmp6, v_tmp7;

        v_int16 v_mul = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + pre_shift))));
        const int16_t* srcp = (const int16_t*)src[pre_shift] + i;
        v_src00 = vx_load(srcp);
        v_src10 = vx_load(srcp + VECSZ);
        v_src20 = vx_load(srcp + 2*VECSZ);
        v_src30 = vx_load(srcp + 3*VECSZ);
        v_mul_expand(v_add_wrap(v_src00, v_128), v_mul, v_res0, v_res1);
        v_mul_expand(v_add_wrap(v_src10, v_128), v_mul, v_res2, v_res3);
        v_mul_expand(v_add_wrap(v_src20, v_128), v_mul, v_res4, v_res5);
        v_mul_expand(v_add_wrap(v_src30, v_128), v_mul, v_res6, v_res7);

        int j = 0;
        for (; j < pre_shift; j++)
        {
            v_mul = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + j))));

            const int16_t* srcj0 = (const int16_t*)src[j] + i;
            const int16_t* srcj1 = (const int16_t*)src[n - 1 - j] + i;
            v_src00 = vx_load(srcj0);
            v_src10 = vx_load(srcj0 + VECSZ);
            v_src20 = vx_load(srcj0 + 2*VECSZ);
            v_src30 = vx_load(srcj0 + 3*VECSZ);
            v_src01 = vx_load(srcj1);
            v_src11 = vx_load(srcj1 + VECSZ);
            v_src21 = vx_load(srcj1 + 2*VECSZ);
            v_src31 = vx_load(srcj1 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src01, v_128), v_tmp0, v_tmp1);
            v_res0 = v_add(v_res0, v_dotprod(v_tmp0, v_mul));
            v_res1 = v_add(v_res1, v_dotprod(v_tmp1, v_mul));
            v_zip(v_add_wrap(v_src10, v_128), v_add_wrap(v_src11, v_128), v_tmp2, v_tmp3);
            v_res2 = v_add(v_res2, v_dotprod(v_tmp2, v_mul));
            v_res3 = v_add(v_res3, v_dotprod(v_tmp3, v_mul));
            v_zip(v_add_wrap(v_src20, v_128), v_add_wrap(v_src21, v_128), v_tmp4, v_tmp5);
            v_res4 = v_add(v_res4, v_dotprod(v_tmp4, v_mul));
            v_res5 = v_add(v_res5, v_dotprod(v_tmp5, v_mul));
            v_zip(v_add_wrap(v_src30, v_128), v_add_wrap(v_src31, v_128), v_tmp6, v_tmp7);
            v_res6 = v_add(v_res6, v_dotprod(v_tmp6, v_mul));
            v_res7 = v_add(v_res7, v_dotprod(v_tmp7, v_mul));
        }

        v_res0 = v_add(v_res0, v_128_4);
        v_res1 = v_add(v_res1, v_128_4);
        v_res2 = v_add(v_res2, v_128_4);
        v_res3 = v_add(v_res3, v_128_4);
        v_res4 = v_add(v_res4, v_128_4);
        v_res5 = v_add(v_res5, v_128_4);
        v_res6 = v_add(v_res6, v_128_4);
        v_res7 = v_add(v_res7, v_128_4);

        v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
        v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
    }
#endif
    for (; i < len; i++)
    {
        ufixedpoint32 val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
        {
            val = val + m[j] * src[j][i];
        }
        dst[i] = val;
    }
}
template <>
void vlineSmoothONa_yzy_a<uint16_t, ufixedpoint32>(const ufixedpoint32* const * src, const ufixedpoint32* m, int n, uint16_t* dst, int len)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    int pre_shift = n / 2;
    const int VECSZ = VTraits<v_uint32>::vlanes();
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint32 v_src00, v_src10, v_src01, v_src11;
        v_uint64 v_res0, v_res1, v_res2, v_res3;
        v_uint64 v_tmp0, v_tmp1, v_tmp2, v_tmp3, v_tmp4, v_tmp5, v_tmp6, v_tmp7;

        v_uint32 v_mul = vx_setall_u32(*((uint32_t*)(m + pre_shift)));
        const uint32_t* srcp = (const uint32_t*)src[pre_shift] + i;
        v_src00 = vx_load(srcp);
        v_src10 = vx_load(srcp + VECSZ);
        v_mul_expand(v_src00, v_mul, v_res0, v_res1);
        v_mul_expand(v_src10, v_mul, v_res2, v_res3);

        int j = 0;
        for (; j < pre_shift; j++)
        {
            v_mul = vx_setall_u32(*((uint32_t*)(m + j)));

            const uint32_t* srcj0 = (const uint32_t*)src[j] + i;
            const uint32_t* srcj1 = (const uint32_t*)src[n - 1 - j] + i;
            v_src00 = vx_load(srcj0);
            v_src01 = vx_load(srcj1);
            v_mul_expand(v_src00, v_mul, v_tmp0, v_tmp1);
            v_mul_expand(v_src01, v_mul, v_tmp2, v_tmp3);
            v_res0 = v_add(v_res0, v_add(v_tmp0, v_tmp2));
            v_res1 = v_add(v_res1, v_add(v_tmp1, v_tmp3));

            v_src10 = vx_load(srcj0 + VECSZ);
            v_src11 = vx_load(srcj1 + VECSZ);
            v_mul_expand(v_src10, v_mul, v_tmp4, v_tmp5);
            v_mul_expand(v_src11, v_mul, v_tmp6, v_tmp7);
            v_res2 = v_add(v_res2, v_add(v_tmp4, v_tmp6));
            v_res3 = v_add(v_res3, v_add(v_tmp5, v_tmp7));
        }

        v_store(dst + i, v_pack(v_rshr_pack<32>(v_res0, v_res1),
                                v_rshr_pack<32>(v_res2, v_res3)));
    }
#endif
    for (; i < len; i++)
    {
        ufixedpoint64 val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
        {
            val = val + m[j] * src[j][i];
        }
        dst[i] = (uint16_t)val;
    }
}
template <typename ET, typename FT>
class fixedSmoothInvoker : public ParallelLoopBody
{
public:
    fixedSmoothInvoker(const ET* _src, size_t _src_stride, ET* _dst, size_t _dst_stride,
                       int _width, int _height, int _cn, const FT* _kx, int _kxlen, const FT* _ky, int _kylen, int _borderType) : ParallelLoopBody(),
                       src(_src), dst(_dst), src_stride(_src_stride), dst_stride(_dst_stride),
                       width(_width), height(_height), cn(_cn), kx(_kx), ky(_ky), kxlen(_kxlen), kylen(_kylen), borderType(_borderType)
    {
        if (kxlen == 1)
        {
            if (kx[0] == FT::one())
                hlineSmoothFunc = hlineSmooth1N1;
            else
                hlineSmoothFunc = hlineSmooth1N;
        }
        else if (kxlen == 3)
        {
            if (kx[0] == (FT::one()>>2)&&kx[1] == (FT::one()>>1)&&kx[2] == (FT::one()>>2))
                hlineSmoothFunc = hlineSmooth3N121;
            else if ((kx[0] - kx[2]).isZero())
                    hlineSmoothFunc = hlineSmooth3Naba;
            else
                hlineSmoothFunc = hlineSmooth3N;
        }
        else if (kxlen == 5)
        {
            if (kx[2] == (FT::one()*(uint8_t)3>>3) &&
                kx[1] == (FT::one()>>2) && kx[3] == (FT::one()>>2) &&
                kx[0] == (FT::one()>>4) && kx[4] == (FT::one()>>4))
                hlineSmoothFunc = hlineSmooth5N14641;
            else if (kx[0] == kx[4] && kx[1] == kx[3])
                hlineSmoothFunc = hlineSmooth5Nabcba;
            else
                hlineSmoothFunc = hlineSmooth5N;
        }
        else if (kxlen % 2 == 1)
        {
            if (kx[(kxlen - 1)/ 2] == FT::one())
                hlineSmoothFunc = hlineSmooth1N1;
            else
                hlineSmoothFunc = hlineSmoothONa_yzy_a;
            for (int i = 0; i < kxlen / 2; i++)
                if (!(kx[i] == kx[kxlen - 1 - i]))
                {
                    hlineSmoothFunc = hlineSmooth;
                    break;
                }
        }
        else
            hlineSmoothFunc = hlineSmooth;
        if (kylen == 1)
        {
            if (ky[0] == FT::one())
                vlineSmoothFunc = vlineSmooth1N1;
            else
                vlineSmoothFunc = vlineSmooth1N;
        }
        else if (kylen == 3)
        {
            if (ky[0] == (FT::one() >> 2) && ky[1] == (FT::one() >> 1) && ky[2] == (FT::one() >> 2))
                vlineSmoothFunc = vlineSmooth3N121;
            else
                vlineSmoothFunc = vlineSmooth3N;
        }
        else if (kylen == 5)
        {
            if (ky[2] == (FT::one() * (uint8_t)3 >> 3) &&
                ky[1] == (FT::one() >> 2) && ky[3] == (FT::one() >> 2) &&
                ky[0] == (FT::one() >> 4) && ky[4] == (FT::one() >> 4))
                vlineSmoothFunc = vlineSmooth5N14641;
            else
                vlineSmoothFunc = vlineSmooth5N;
        }
        else if (kylen % 2 == 1)
        {
            vlineSmoothFunc = vlineSmoothONa_yzy_a;
            for (int i = 0; i < kylen / 2; i++)
                if (!(ky[i] == ky[kylen - 1 - i]))
                {
                    vlineSmoothFunc = vlineSmooth;
                    break;
                }
        }
        else
            vlineSmoothFunc = vlineSmooth;
    }
    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        AutoBuffer<FT> _buf(width*cn*kylen);
        FT* buf = _buf.data();
        AutoBuffer<FT*> _ptrs(kylen*2);
        FT** ptrs = _ptrs.data();

        if (kylen == 1)
        {
            ptrs[0] = buf;
            for (int i = range.start; i < range.end; i++)
            {
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[0], width, borderType);
                vlineSmoothFunc(ptrs, ky, kylen, dst + i * dst_stride, width*cn);
            }
        }
        else if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int pre_shift = kylen / 2;
            int post_shift = kylen - pre_shift - 1;
            // First line evaluation
            int idst = range.start;
            int ifrom = max(0, idst - pre_shift);
            int ito = idst + post_shift + 1;
            int i = ifrom;
            int bufline = 0;
            for (; i < min(ito, height); i++, bufline++)
            {
                ptrs[bufline+kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
            }
            for (; i < ito; i++, bufline++)
            {
                int src_idx = borderInterpolate(i, height, borderType);
                if (src_idx < ifrom)
                {
                    ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                    hlineSmoothFunc(src + src_idx * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                }
                else
                {
                    ptrs[bufline + kylen] = ptrs[bufline] = ptrs[src_idx - ifrom];
                }
            }
            for (int j = idst - pre_shift; j < 0; j++)
            {
                int src_idx = borderInterpolate(j, height, borderType);
                if (src_idx >= ito)
                {
                    ptrs[2*kylen + j] = ptrs[kylen + j] = buf + (kylen + j) * width*cn;
                    hlineSmoothFunc(src + src_idx * src_stride, cn, kx, kxlen, ptrs[kylen + j], width, borderType);
                }
                else
                {
                    ptrs[2*kylen + j] = ptrs[kylen + j] = ptrs[src_idx];
                }
            }
            vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn); idst++;

            // border mode dependent part evaluation
            // i points to last src row to evaluate in convolution
            bufline %= kylen; ito = min(height, range.end + post_shift);
            for (; i < min(kylen, ito); i++, idst++)
            {
                ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                bufline = (bufline + 1) % kylen;
                vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
            }
            // Points inside the border
            for (; i < ito; i++, idst++)
            {
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                bufline = (bufline + 1) % kylen;
                vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
            }
            // Points that could fall below border
            for (; i < range.end + post_shift; i++, idst++)
            {
                int src_idx = borderInterpolate(i, height, borderType);
                if ((i - src_idx) > kylen)
                    hlineSmoothFunc(src + src_idx * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                else
                    ptrs[bufline + kylen] = ptrs[bufline] = ptrs[(bufline + kylen - (i - src_idx)) % kylen];
                bufline = (bufline + 1) % kylen;
                vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
            }
        }
        else
        {
            int pre_shift = kylen / 2;
            int post_shift = kylen - pre_shift - 1;
            // First line evaluation
            int idst = range.start;
            int ifrom = idst - pre_shift;
            int ito = min(idst + post_shift + 1, height);
            int i = max(0, ifrom);
            int bufline = 0;
            for (; i < ito; i++, bufline++)
            {
                ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
            }

            if (bufline == 1)
                vlineSmooth1N(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            else if (bufline == 3)
                vlineSmooth3N(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            else if (bufline == 5)
                vlineSmooth5N(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            else
                vlineSmooth(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            idst++;

            // border mode dependent part evaluation
            // i points to last src row to evaluate in convolution
            bufline %= kylen; ito = min(height, range.end + post_shift);
            for (; i < min(kylen, ito); i++, idst++)
            {
                ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                bufline++;
                if (bufline == 3)
                    vlineSmooth3N(ptrs, ky + kylen - bufline, i + 1, dst + idst*dst_stride, width*cn);
                else if (bufline == 5)
                    vlineSmooth5N(ptrs, ky + kylen - bufline, i + 1, dst + idst*dst_stride, width*cn);
                else
                    vlineSmooth(ptrs, ky + kylen - bufline, i + 1, dst + idst*dst_stride, width*cn);
                bufline %= kylen;
            }
            // Points inside the border
            if (i - max(0, ifrom) >= kylen)
            {
                for (; i < ito; i++, idst++)
                {
                    hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                    bufline = (bufline + 1) % kylen;
                    vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
                }

                // Points that could fall below border
                // i points to first src row to evaluate in convolution
                bufline = (bufline + 1) % kylen;
                for (i = idst - pre_shift; i < range.end - pre_shift; i++, idst++, bufline++)
                    if (height - i == 3)
                        vlineSmooth3N(ptrs + bufline, ky, height - i, dst + idst*dst_stride, width*cn);
                    else if (height - i == 5)
                        vlineSmooth5N(ptrs + bufline, ky, height - i, dst + idst*dst_stride, width*cn);
                    else
                        vlineSmooth(ptrs + bufline, ky, height - i, dst + idst*dst_stride, width*cn);
            }
            else
            {
                // i points to first src row to evaluate in convolution
                for (i = idst - pre_shift; i < min(range.end - pre_shift, 0); i++, idst++)
                    if (height == 3)
                        vlineSmooth3N(ptrs, ky - i, height, dst + idst*dst_stride, width*cn);
                    else if (height == 5)
                        vlineSmooth5N(ptrs, ky - i, height, dst + idst*dst_stride, width*cn);
                    else
                        vlineSmooth(ptrs, ky - i, height, dst + idst*dst_stride, width*cn);
                for (; i < range.end - pre_shift; i++, idst++)
                    if (height - i == 3)
                        vlineSmooth3N(ptrs + i - max(0, ifrom), ky, height - i, dst + idst*dst_stride, width*cn);
                    else if (height - i == 5)
                        vlineSmooth5N(ptrs + i - max(0, ifrom), ky, height - i, dst + idst*dst_stride, width*cn);
                    else
                        vlineSmooth(ptrs + i - max(0, ifrom), ky, height - i, dst + idst*dst_stride, width*cn);
            }
        }
    }
private:
    const ET* src;
    ET* dst;
    size_t src_stride, dst_stride;
    int width, height, cn;
    const FT *kx, *ky;
    int kxlen, kylen;
    int borderType;
    void(*hlineSmoothFunc)(const ET* src, int cn, const FT* m, int n, FT* dst, int len, int borderType);
    void(*vlineSmoothFunc)(const FT* const * src, const FT* m, int n, ET* dst, int len);

    fixedSmoothInvoker(const fixedSmoothInvoker&);
    fixedSmoothInvoker& operator=(const fixedSmoothInvoker&);
};

}  // namespace anon

template <typename RFT, typename ET, typename FT>
void GaussianBlurFixedPointImpl(const Mat& src, /*const*/ Mat& dst,
                                const RFT* fkx, int fkx_size,
                                const RFT* fky, int fky_size,
                                int borderType)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(src.depth() == DataType<ET>::depth && ((borderType & BORDER_ISOLATED) || !src.isSubmatrix()));
    fixedSmoothInvoker<ET, FT> invoker(
            src.ptr<ET>(), src.step1(),
            dst.ptr<ET>(), dst.step1(), dst.cols, dst.rows, dst.channels(),
            (const FT*)fkx, fkx_size, (const FT*)fky, fky_size,
            borderType & ~BORDER_ISOLATED);
    {
        // TODO AVX guard (external call)
        parallel_for_(Range(0, dst.rows), invoker, std::max(1, std::min(getNumThreads(), getNumberOfCPUs())));
    }
}
template <>
void GaussianBlurFixedPoint<uint16_t>(const Mat& src, /*const*/ Mat& dst,
                                      const uint16_t/*ufixedpoint16*/* fkx, int fkx_size,
                                      const uint16_t/*ufixedpoint16*/* fky, int fky_size,
                                      int borderType)
{
    GaussianBlurFixedPointImpl<uint16_t, uint8_t, ufixedpoint16>(src, dst, fkx, fkx_size, fky, fky_size, borderType);
}

template <>
void GaussianBlurFixedPoint<uint32_t>(const Mat& src, /*const*/ Mat& dst,
                                      const uint32_t/*ufixedpoint32*/* fkx, int fkx_size,
                                      const uint32_t/*ufixedpoint32*/* fky, int fky_size,
                                      int borderType)
{
    GaussianBlurFixedPointImpl<uint32_t, uint16_t, ufixedpoint32>(src, dst, fkx, fkx_size, fky, fky_size, borderType);
}
#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
