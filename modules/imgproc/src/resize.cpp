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
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
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
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"
#include "resize.hpp"

#include "opencv2/core/softfloat.hpp"
#include "fixedpoint.inl.hpp"

using namespace cv;

namespace
{

template <typename ET, bool needsign> struct fixedtype { typedef fixedpoint64 type; };
template <> struct fixedtype<uint32_t, false> { typedef ufixedpoint64 type; };
template <bool needsign> struct fixedtype<int16_t, needsign> { typedef fixedpoint32 type; };
template <> struct fixedtype<uint16_t, false> { typedef ufixedpoint32 type; };
template <bool needsign> struct fixedtype<int8_t, needsign> { typedef fixedpoint32 type; };
template <> struct fixedtype<uint8_t, false> { typedef ufixedpoint16 type; };

//FT is fixedtype<ET, needsign>::type
template <typename ET, typename FT, int n, bool mulall>
static void hlineResize(ET* src, int cn, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    for (; i < dst_min; i++, m += n) // Points that fall left from src image so became equal to leftmost src point
    {
        for (int j = 0; j < cn; j++, dst++)
        {
            *dst = src[j];
        }
    }
    for (; i < dst_max; i++, m += n)
    {
        ET* src_ofst = src + cn*ofst[i];
        for (int j = 0; j < cn; j++, dst++)
        {
            *dst = (mulall || !m[0].isZero()) ? m[0] * src_ofst[j] : FT::zero();
            for (int k = 1; k < n; k++)
            {
                *dst = *dst + ((mulall || !m[k].isZero()) ? m[k] * src_ofst[j+k*cn] : FT::zero());
            }
        }
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ET* src_last = src + cn*ofst[dst_width - 1];
    for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
    {
        for (int j = 0; j < cn; j++, dst++)
        {
            *dst = src_last[j];
        }
    }
}
template <typename ET, typename FT, int n, bool mulall, int cncnt> struct hline
{
    static void ResizeCn(ET* src, int cn, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        hlineResize<ET, FT, n, mulall>(src, cn, ofst, m, dst, dst_min, dst_max, dst_width);
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 1>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[1];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + ofst[dst_width - 1])[0];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 2>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + 2*ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[2];
            *(dst++) = m[0] * px[1] + m[1] * px[3];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 2*ofst[dst_width - 1])[0];
        src1 = (src + 2*ofst[dst_width - 1])[1];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 3>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + 3*ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[3];
            *(dst++) = m[0] * px[1] + m[1] * px[4];
            *(dst++) = m[0] * px[2] + m[1] * px[5];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 3*ofst[dst_width - 1])[0];
        src1 = (src + 3*ofst[dst_width - 1])[1];
        src2 = (src + 3*ofst[dst_width - 1])[2];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 2, true, 4>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]), src3(src[3]);
        for (; i < dst_min; i++, m += 2) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
        for (; i < dst_max; i++, m += 2)
        {
            ET* px = src + 4*ofst[i];
            *(dst++) = m[0] * px[0] + m[1] * px[4];
            *(dst++) = m[0] * px[1] + m[1] * px[5];
            *(dst++) = m[0] * px[2] + m[1] * px[6];
            *(dst++) = m[0] * px[3] + m[1] * px[7];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 4*ofst[dst_width - 1])[0];
        src1 = (src + 4*ofst[dst_width - 1])[1];
        src2 = (src + 4*ofst[dst_width - 1])[2];
        src3 = (src + 4*ofst[dst_width - 1])[3];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 1>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[1] + m[2] * src[2] + m[3] * src[3];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + ofst[dst_width - 1])[0];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 2>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + 2*ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[2] + m[2] * src[4] + m[3] * src[6];
            *(dst++) = m[0] * src[1] + m[1] * src[3] + m[2] * src[5] + m[3] * src[7];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 2*ofst[dst_width - 1])[0];
        src1 = (src + 2*ofst[dst_width - 1])[1];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 3>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + 3*ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[3] + m[2] * src[6] + m[3] * src[ 9];
            *(dst++) = m[0] * src[1] + m[1] * src[4] + m[2] * src[7] + m[3] * src[10];
            *(dst++) = m[0] * src[2] + m[1] * src[5] + m[2] * src[8] + m[3] * src[11];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 3*ofst[dst_width - 1])[0];
        src1 = (src + 3*ofst[dst_width - 1])[1];
        src2 = (src + 3*ofst[dst_width - 1])[2];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
        }
    }
};
template <typename ET, typename FT> struct hline<ET, FT, 4, true, 4>
{
    static void ResizeCn(ET* src, int, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
    {
        int i = 0;
        FT src0(src[0]), src1(src[1]), src2(src[2]), src3(src[3]);
        for (; i < dst_min; i++, m += 4) // Points that fall left from src image so became equal to leftmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
        for (; i < dst_max; i++, m += 4)
        {
            ET* px = src + 4*ofst[i];
            *(dst++) = m[0] * src[0] + m[1] * src[4] + m[2] * src[ 8] + m[3] * src[12];
            *(dst++) = m[0] * src[1] + m[1] * src[5] + m[2] * src[ 9] + m[3] * src[13];
            *(dst++) = m[0] * src[2] + m[1] * src[6] + m[2] * src[10] + m[3] * src[14];
            *(dst++) = m[0] * src[3] + m[1] * src[7] + m[2] * src[11] + m[3] * src[15];
        }
        // Avoid reading a potentially unset ofst, leading to a random memory read.
        if (i >= dst_width) {
            return;
        }
        src0 = (src + 4*ofst[dst_width - 1])[0];
        src1 = (src + 4*ofst[dst_width - 1])[1];
        src2 = (src + 4*ofst[dst_width - 1])[2];
        src3 = (src + 4*ofst[dst_width - 1])[3];
        for (; i < dst_width; i++) // Points that fall right from src image so became equal to rightmost src point
        {
            *(dst++) = src0;
            *(dst++) = src1;
            *(dst++) = src2;
            *(dst++) = src3;
        }
    }
};
template <typename ET, typename FT, int n, bool mulall, int cncnt>
static void hlineResizeCn(ET* src, int cn, int *ofst, FT* m, FT* dst, int dst_min, int dst_max, int dst_width)
{
    hline<ET, FT, n, mulall, cncnt>::ResizeCn(src, cn, ofst, m, dst, dst_min, dst_max, dst_width);
}

template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 1>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    ufixedpoint16 src_0(src[0]);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_src_0 = vx_setall_u16(*((uint16_t*)&src_0));
    for (; i <= dst_min - VECSZ; i += VECSZ, m += 2*VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = src_0;
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - 2*VECSZ; i += 2*VECSZ, m += 4*VECSZ, dst += 2*VECSZ)
    {
        v_uint16 v_src0, v_src1;
        v_expand(vx_lut_pairs(src, ofst + i), v_src0, v_src1);
        v_store((uint16_t*)dst      , v_pack(v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), vx_load((int16_t*)m))),
                                             v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), vx_load((int16_t*)m + VECSZ)))));
        v_expand(vx_lut_pairs(src, ofst + i + VECSZ), v_src0, v_src1);
        v_store((uint16_t*)dst+VECSZ, v_pack(v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), vx_load((int16_t*)m + 2*VECSZ))),
                                             v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), vx_load((int16_t*)m + 3*VECSZ)))));
    }
    if (i <= dst_max - VECSZ)
    {
        v_uint16 v_src0, v_src1;
        v_expand(vx_lut_pairs(src, ofst + i), v_src0, v_src1);
        v_store((uint16_t*)dst, v_pack(v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), vx_load((int16_t*)m))),
                                       v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), vx_load((int16_t*)m + VECSZ)))));
        i += VECSZ; m += 2*VECSZ; dst += VECSZ;
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[1];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    src_0 = (src + ofst[dst_width - 1])[0];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_src_0 = vx_setall_u16(*((uint16_t*)&src_0));
    for (; i <= dst_width - VECSZ; i += VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = src_0;
    }
}
template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 2>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    union {
        uint32_t d;
        uint16_t w[2];
    } srccn;
    ((ufixedpoint16*)(srccn.w))[0] = src[0];
    ((ufixedpoint16*)(srccn.w))[1] = src[1];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_srccn = v_reinterpret_as_u16(vx_setall_u32(srccn.d));
    for (; i <= dst_min - VECSZ/2; i += VECSZ/2, m += VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - VECSZ/2; i += VECSZ/2, m += VECSZ, dst += VECSZ)
    {
        v_uint16 v_src0, v_src1;
        v_expand(v_interleave_pairs(v_reinterpret_as_u8(vx_lut_pairs((uint16_t*)src, ofst + i))), v_src0, v_src1);

        v_uint32 v_mul = vx_load((uint32_t*)m);//AaBbCcDd
        v_uint32 v_zip0, v_zip1;
        v_zip(v_mul, v_mul, v_zip0, v_zip1);//AaAaBbBb CcCcDdDd
        v_uint32 v_res0 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), v_reinterpret_as_s16(v_zip0)));
        v_uint32 v_res1 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), v_reinterpret_as_s16(v_zip1)));
        v_store((uint16_t*)dst, v_pack(v_res0, v_res1));//AB1AB2CD1CD2
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + 2 * ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[2];
        *(dst++) = m[0] * px[1] + m[1] * px[3];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ((ufixedpoint16*)(srccn.w))[0] = (src + 2 * ofst[dst_width - 1])[0]; ((ufixedpoint16*)(srccn.w))[1] = (src + 2 * ofst[dst_width - 1])[1];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_srccn = v_reinterpret_as_u16(vx_setall_u32(srccn.d));
    for (; i <= dst_width - VECSZ/2; i += VECSZ/2, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
    }
}
template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 3>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    union {
        uint64_t q;
        uint16_t w[4];
    } srccn;
    ((ufixedpoint16*)(srccn.w))[0] = src[0];
    ((ufixedpoint16*)(srccn.w))[1] = src[1];
    ((ufixedpoint16*)(srccn.w))[2] = src[2];
    ((ufixedpoint16*)(srccn.w))[3] = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_srccn = v_pack_triplets(v_reinterpret_as_u16(vx_setall_u64(srccn.q)));
    for (; i <= dst_min - (VECSZ+2)/3; i += VECSZ/4, m += VECSZ/2, dst += 3*VECSZ/4) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    CV_DECL_ALIGNED(CV_SIMD_WIDTH) int ofst3[VTraits<v_uint16>::max_nlanes/2];
    for (; i <= dst_max - (3*VECSZ/4 + (VECSZ+2)/3); i += VECSZ/2, m += VECSZ, dst += 3*VECSZ/2)
    {
        v_store(ofst3, v_mul(vx_load(ofst + i), vx_setall_s32(3)));
        v_uint8 v_src01, v_src23;
        v_uint16 v_src0, v_src1, v_src2, v_src3;
        v_zip(vx_lut_quads(src, ofst3), v_reinterpret_as_u8(v_shr<8>(v_reinterpret_as_u32(vx_lut_quads(src+2, ofst3)))), v_src01, v_src23);
        v_expand(v_src01, v_src0, v_src1);
        v_expand(v_src23, v_src2, v_src3);

        v_uint32 v_mul0, v_mul1, v_mul2, v_mul3, v_tmp;
        v_mul0 = vx_load((uint32_t*)m);//AaBbCcDd
        v_zip(v_mul0, v_mul0, v_mul3, v_tmp );//AaAaBbBb CcCcDdDd
        v_zip(v_mul3, v_mul3, v_mul0, v_mul1);//AaAaAaAa BbBbBbBb
        v_zip(v_tmp , v_tmp , v_mul2, v_mul3);//CcCcCcCc DdDdDdDd

        v_uint32 v_res0 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), v_reinterpret_as_s16(v_mul0)));
        v_uint32 v_res1 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), v_reinterpret_as_s16(v_mul1)));
        v_uint32 v_res2 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src2), v_reinterpret_as_s16(v_mul2)));
        v_uint32 v_res3 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src3), v_reinterpret_as_s16(v_mul3)));
        v_store((uint16_t*)dst            , v_pack_triplets(v_pack(v_res0, v_res1)));
        v_store((uint16_t*)dst + 3*VECSZ/4, v_pack_triplets(v_pack(v_res2, v_res3)));
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + 3 * ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[3];
        *(dst++) = m[0] * px[1] + m[1] * px[4];
        *(dst++) = m[0] * px[2] + m[1] * px[5];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ((ufixedpoint16*)(srccn.w))[0] = (src + 3*ofst[dst_width - 1])[0];
    ((ufixedpoint16*)(srccn.w))[1] = (src + 3*ofst[dst_width - 1])[1];
    ((ufixedpoint16*)(srccn.w))[2] = (src + 3*ofst[dst_width - 1])[2];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_srccn = v_pack_triplets(v_reinterpret_as_u16(vx_setall_u64(srccn.q)));
    for (; i <= dst_width - (VECSZ+2)/3; i += VECSZ/4, dst += 3*VECSZ/4) // Points that fall right from src image so became equal to rightmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
    }
}
template <>
void hlineResizeCn<uint8_t, ufixedpoint16, 2, true, 4>(uint8_t* src, int, int *ofst, ufixedpoint16* m, ufixedpoint16* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    union {
        uint64_t q;
        uint16_t w[4];
    } srccn;
    ((ufixedpoint16*)(srccn.w))[0] = src[0];
    ((ufixedpoint16*)(srccn.w))[1] = src[1];
    ((ufixedpoint16*)(srccn.w))[2] = src[2];
    ((ufixedpoint16*)(srccn.w))[3] = src[3];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint16>::vlanes();
    v_uint16 v_srccn = v_reinterpret_as_u16(vx_setall_u64(srccn.q));
    for (; i <= dst_min - VECSZ/4; i += VECSZ/4, m += VECSZ/2, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[3];
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - VECSZ/2; i += VECSZ/2, m += VECSZ, dst += 2*VECSZ)
    {
        v_uint16 v_src0, v_src1, v_src2, v_src3;
        v_expand(v_interleave_quads(v_reinterpret_as_u8(vx_lut_pairs((uint32_t*)src, ofst + i))), v_src0, v_src1);
        v_expand(v_interleave_quads(v_reinterpret_as_u8(vx_lut_pairs((uint32_t*)src, ofst + i + VECSZ/4))), v_src2, v_src3);

        v_uint32 v_mul0, v_mul1, v_mul2, v_mul3, v_tmp;
        v_mul0 = vx_load((uint32_t*)m);//AaBbCcDd
        v_zip(v_mul0, v_mul0, v_mul3, v_tmp );//AaAaBbBb CcCcDdDd
        v_zip(v_mul3, v_mul3, v_mul0, v_mul1);//AaAaAaAa BbBbBbBb
        v_zip(v_tmp , v_tmp , v_mul2, v_mul3);//CcCcCcCc DdDdDdDd

        v_uint32 v_res0 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src0), v_reinterpret_as_s16(v_mul0)));
        v_uint32 v_res1 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src1), v_reinterpret_as_s16(v_mul1)));
        v_uint32 v_res2 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src2), v_reinterpret_as_s16(v_mul2)));
        v_uint32 v_res3 = v_reinterpret_as_u32(v_dotprod(v_reinterpret_as_s16(v_src3), v_reinterpret_as_s16(v_mul3)));
        v_store((uint16_t*)dst        , v_pack(v_res0, v_res1));
        v_store((uint16_t*)dst + VECSZ, v_pack(v_res2, v_res3));
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint8_t* px = src + 4 * ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[4];
        *(dst++) = m[0] * px[1] + m[1] * px[5];
        *(dst++) = m[0] * px[2] + m[1] * px[6];
        *(dst++) = m[0] * px[3] + m[1] * px[7];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    ((ufixedpoint16*)(srccn.w))[0] = (src + 4 * ofst[dst_width - 1])[0]; ((ufixedpoint16*)(srccn.w))[1] = (src + 4 * ofst[dst_width - 1])[1];
    ((ufixedpoint16*)(srccn.w))[2] = (src + 4 * ofst[dst_width - 1])[2]; ((ufixedpoint16*)(srccn.w))[3] = (src + 4 * ofst[dst_width - 1])[3];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_srccn = v_reinterpret_as_u16(vx_setall_u64(srccn.q));
    for (; i <= dst_width - VECSZ/4; i += VECSZ/4, dst += VECSZ) // Points that fall right from src image so became equal to rightmost src point
    {
        v_store((uint16_t*)dst, v_srccn);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = ((ufixedpoint16*)(srccn.w))[0];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[1];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[2];
        *(dst++) = ((ufixedpoint16*)(srccn.w))[3];
    }
}
template <>
void hlineResizeCn<uint16_t, ufixedpoint32, 2, true, 1>(uint16_t* src, int, int *ofst, ufixedpoint32* m, ufixedpoint32* dst, int dst_min, int dst_max, int dst_width)
{
    int i = 0;
    ufixedpoint32 src_0(src[0]);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint32>::vlanes();
    v_uint32 v_src_0 = vx_setall_u32(*((uint32_t*)&src_0));
    for (; i <= dst_min - VECSZ; i += VECSZ, m += 2*VECSZ, dst += VECSZ) // Points that fall left from src image so became equal to leftmost src point
    {
        v_store((uint32_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_min; i++, m += 2)
    {
        *(dst++) = src_0;
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    for (; i <= dst_max - VECSZ; i += VECSZ, m += 2*VECSZ, dst += VECSZ)
    {
        v_uint32 v_src0, v_src1;
        v_expand(vx_lut_pairs(src, ofst + i), v_src0, v_src1);

        v_uint64 v_res0 = v_reinterpret_as_u64(v_mul(v_src0, vx_load((uint32_t *)m)));
        v_uint64 v_res1 = v_reinterpret_as_u64(v_mul(v_src1, vx_load((uint32_t *)m + VECSZ)));
        v_store((uint32_t*)dst, v_pack(v_add(v_and(v_res0, vx_setall_u64(0xFFFFFFFF)), v_shr<32>(v_res0)),
                                       v_add(v_and(v_res1, vx_setall_u64(0xFFFFFFFF)), v_shr<32>(v_res1))));
    }
#endif
    for (; i < dst_max; i += 1, m += 2)
    {
        uint16_t* px = src + ofst[i];
        *(dst++) = m[0] * px[0] + m[1] * px[1];
    }
    // Avoid reading a potentially unset ofst, leading to a random memory read.
    if (i >= dst_width) {
        return;
    }
    src_0 = (src + ofst[dst_width - 1])[0];
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_src_0 = vx_setall_u32(*((uint32_t*)&src_0));
    for (; i <= dst_width - VECSZ; i += VECSZ, dst += VECSZ)
    {
        v_store((uint32_t*)dst, v_src_0);
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = src_0;
    }
}

template <typename ET, typename FT>
void vlineSet(FT* src, ET* dst, int dst_width)
{
    for (int i = 0; i < dst_width; i++)
        dst[i] = src[i];
}
template <>
void vlineSet<uint8_t, ufixedpoint16>(ufixedpoint16* src, uint8_t* dst, int dst_width)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint8>::vlanes();
    const v_uint16 v_fixedRound = vx_setall_u16((uint16_t)((1U << 8) >> 1));
    for (; i <= dst_width - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
    {
        v_uint16 v_src0 = vx_load((uint16_t*)src);
        v_uint16 v_src1 = vx_load((uint16_t*)src + VECSZ/2);

        v_uint16 v_res0 = v_shr<8>(v_add(v_src0, v_fixedRound));
        v_uint16 v_res1 = v_shr<8>(v_add(v_src1, v_fixedRound));

        v_store(dst, v_pack(v_res0, v_res1));
    }
#endif
    for (; i < dst_width; i++)
        *(dst++) = *(src++);
}

template <typename ET, typename FT, int n>
void vlineResize(FT* src, size_t src_step, FT* m, ET* dst, int dst_width)
{
    for (int i = 0; i < dst_width; i++)
    {
        typename FT::WT res = src[i] * m[0];
        for (int k = 1; k < n; k++)
            res = res + src[i + k*src_step] * m[k];
        dst[i] = res;
    }
}
template <>
void vlineResize<uint8_t, ufixedpoint16, 2>(ufixedpoint16* src, size_t src_step, ufixedpoint16* m, uint8_t* dst, int dst_width)
{
    int i = 0;
    ufixedpoint16* src1 = src + src_step;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_uint8>::vlanes();
    const v_int32 v_fixedRound = vx_setall_s32((int32_t)((1 << 16) >> 1));
    const v_int16 v_128    = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1<<15));
    const v_int8  v_128_16 = v_reinterpret_as_s8 (vx_setall_u8 ((uint8_t) 1<<7));

    v_int16 v_mul = v_reinterpret_as_s16(vx_setall_u32(((uint32_t*)m)[0]));
    for (; i <= dst_width - VECSZ; i += VECSZ, src += VECSZ, src1 += VECSZ, dst += VECSZ)
    {
        v_int16 v_src00 = vx_load((int16_t*)src);
        v_int16 v_src10 = vx_load((int16_t*)src1);
        v_int16 v_tmp0, v_tmp1;
        v_zip(v_add_wrap(v_src00,v_128), v_add_wrap(v_src10,v_128), v_tmp0, v_tmp1);

        v_int32 v_res0 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res1 = v_dotprod(v_tmp1, v_mul);

        v_int16 v_src01 = vx_load((int16_t*)src + VECSZ/2);
        v_int16 v_src11 = vx_load((int16_t*)src1 + VECSZ/2);
        v_zip(v_add_wrap(v_src01,v_128), v_add_wrap(v_src11,v_128), v_tmp0, v_tmp1);
        v_int32 v_res2 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res3 = v_dotprod(v_tmp1, v_mul);

        v_int8 v_res = v_pack(v_pack(v_shr<16>(v_add(v_res0, v_fixedRound)),
                                     v_shr<16>(v_add(v_res1, v_fixedRound))),
                              v_pack(v_shr<16>(v_add(v_res2, v_fixedRound)),
                                     v_shr<16>(v_add(v_res3, v_fixedRound))));

        v_store(dst, v_reinterpret_as_u8(v_sub_wrap(v_res, v_128_16)));
    }
#endif
    for (; i < dst_width; i++)
    {
        *(dst++) = (uint8_t)(*(src++) * m[0] + *(src1++) * m[1]);
    }
}

template <typename ET> class interpolationLinear
{
public:
    static const int len = 2;
    static const bool needsign = false;
    interpolationLinear(double inv_scale, int srcsize, int dstsize) : scale(softdouble::one() / softdouble(inv_scale)), maxsize(srcsize), minofst(0), maxofst(dstsize) {}
    void getCoeffs(int val, int* offset, typename fixedtype<ET, needsign>::type* coeffs)
    {
        typedef typename fixedtype<ET, needsign>::type fixedpoint;
        softdouble fval = scale*(softdouble(val)+softdouble(0.5))-softdouble(0.5);
        int ival = cvFloor(fval);
        if (ival >= 0 && maxsize > 1)
        {
            if (ival < maxsize - 1)
            {
                *offset = ival;
                coeffs[1] = fval - softdouble(ival);
                coeffs[0] = fixedpoint::one() - coeffs[1];
            }
            else
            {
                *offset = maxsize - 1;
                maxofst = min(maxofst, val);
            }
        }
        else
        {
            minofst = max(minofst, val + 1);
        }
    }
    void getMinMax(int &min, int &max)
    {
        min = minofst;
        max = maxofst;
    }
protected:
    softdouble scale;
    int maxsize;
    int minofst, maxofst;
};

template <typename ET, typename FT, int interp_y_len>
class resize_bitExactInvoker :
    public ParallelLoopBody
{
public:
    typedef FT fixedpoint;
    typedef void(*hResizeFunc)(ET* src, int cn, int *ofst, fixedpoint* m, fixedpoint* dst, int dst_min, int dst_max, int dst_width);
    resize_bitExactInvoker(const uchar* _src, size_t _src_step, int _src_width, int _src_height,
                           uchar* _dst, size_t _dst_step, int _dst_width, int _dst_height,
                           int _cn, int *_xoffsets, int *_yoffsets, fixedpoint *_xcoeffs, fixedpoint *_ycoeffs,
                           int _min_x, int _max_x, int _min_y, int _max_y, hResizeFunc _hResize) : ParallelLoopBody(),
                           src(_src), src_step(_src_step), src_width(_src_width), src_height(_src_height),
                           dst(_dst), dst_step(_dst_step), dst_width(_dst_width), dst_height(_dst_height),
                           cn(_cn), xoffsets(_xoffsets), yoffsets(_yoffsets), xcoeffs(_xcoeffs), ycoeffs(_ycoeffs),
                           min_x(_min_x), max_x(_max_x), min_y(_min_y), max_y(_max_y), hResize(_hResize) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        AutoBuffer<fixedpoint> linebuf(interp_y_len * dst_width * cn);
        int last_eval = - interp_y_len;
        int evalbuf_start = 0;
        int rmin_y = max(min_y, range.start);
        int rmax_y = min(max_y, range.end);
        if (range.start < min_y)
        {
            last_eval = 1 - interp_y_len;
            evalbuf_start = 1;
            hResize((ET*)src, cn, xoffsets, xcoeffs, linebuf.data(), min_x, max_x, dst_width);
        }
        int dy = range.start;
        for (; dy < rmin_y; dy++)
            vlineSet<ET, FT>(linebuf.data(), (ET*)(dst + dst_step * dy), dst_width*cn);
        for (; dy < rmax_y; dy++)
        {
            int &iy = yoffsets[dy];

            int i;
            for (i = max(iy, last_eval + interp_y_len); i < min(iy + interp_y_len, src_height); i++, evalbuf_start = (evalbuf_start + 1) % interp_y_len)
                hResize((ET*)(src + i * src_step), cn, xoffsets, xcoeffs, linebuf.data() + evalbuf_start*(dst_width * cn), min_x, max_x, dst_width);
            evalbuf_start = (evalbuf_start + max(iy, src_height - interp_y_len) - max(last_eval, src_height - interp_y_len)) % interp_y_len;
            last_eval = iy;

            fixedpoint curcoeffs[interp_y_len];
            for (i = 0; i < evalbuf_start; i++)
                curcoeffs[i] = ycoeffs[ dy*interp_y_len - evalbuf_start + interp_y_len + i];
            for (; i < interp_y_len; i++)
                curcoeffs[i] = ycoeffs[ dy*interp_y_len - evalbuf_start + i];

            vlineResize<ET, FT, interp_y_len>(linebuf.data(), dst_width*cn, curcoeffs, (ET*)(dst + dst_step * dy), dst_width*cn);
        }
        fixedpoint *endline = linebuf.data();
        if (last_eval + interp_y_len > src_height)
            endline += dst_width*cn*((evalbuf_start + src_height - 1 - last_eval) % interp_y_len);
        else
            hResize((ET*)(src + (src_height - 1) * src_step), cn, xoffsets, xcoeffs, endline, min_x, max_x, dst_width);
        for (; dy < range.end; dy++)
            vlineSet<ET, FT>(endline, (ET*)(dst + dst_step * dy), dst_width*cn);
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vx_cleanup();
#endif
    }

private:
    const uchar* src;
    size_t src_step;
    int src_width, src_height;
    uchar* dst;
    size_t dst_step;
    int dst_width, dst_height, cn;
    int *xoffsets, *yoffsets;
    fixedpoint *xcoeffs, *ycoeffs;
    int min_x, max_x, min_y, max_y;
    hResizeFunc hResize;

    resize_bitExactInvoker(const resize_bitExactInvoker&);
    resize_bitExactInvoker& operator=(const resize_bitExactInvoker&);
};

template <typename ET, typename interpolation>
void resize_bitExact(const uchar* src, size_t src_step, int src_width, int src_height,
                           uchar* dst, size_t dst_step, int dst_width, int dst_height,
                     int cn, double inv_scale_x, double inv_scale_y)
{
    typedef typename fixedtype<ET, interpolation::needsign>::type fixedpoint;
    void(*hResize)(ET* src, int cn, int *ofst, fixedpoint* m, fixedpoint* dst, int dst_min, int dst_max, int dst_width);
    switch (cn)
    {
    case  1: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 1> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 1>; break;
    case  2: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 2> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 2>; break;
    case  3: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 3> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 3>; break;
    case  4: hResize = src_width > interpolation::len ? hlineResizeCn<ET, fixedpoint, interpolation::len, true, 4> : hlineResizeCn<ET, fixedpoint, interpolation::len, false, 4>; break;
    default: hResize = src_width > interpolation::len ? hlineResize<ET, fixedpoint, interpolation::len, true>      : hlineResize<ET, fixedpoint, interpolation::len, false>     ; break;
    }

    interpolation interp_x(inv_scale_x, src_width, dst_width);
    interpolation interp_y(inv_scale_y, src_height, dst_height);

    AutoBuffer<uchar> buf( dst_width * sizeof(int) +
                           dst_height * sizeof(int) +
                           dst_width * interp_x.len*sizeof(fixedpoint) +
                           dst_height * interp_y.len * sizeof(fixedpoint) );
    int* xoffsets = (int*)buf.data();
    int* yoffsets = xoffsets + dst_width;
    fixedpoint* xcoeffs = (fixedpoint*)(yoffsets + dst_height);
    fixedpoint* ycoeffs = xcoeffs + dst_width * interp_x.len;

    int min_x, max_x, min_y, max_y;
    for (int dx = 0; dx < dst_width; dx++)
        interp_x.getCoeffs(dx, xoffsets+dx, xcoeffs+dx*interp_x.len);
    interp_x.getMinMax(min_x, max_x);
    for (int dy = 0; dy < dst_height; dy++)
        interp_y.getCoeffs(dy, yoffsets+dy, ycoeffs+dy*interp_y.len);
    interp_y.getMinMax(min_y, max_y);

    resize_bitExactInvoker<ET, fixedpoint, interpolation::len> invoker(src, src_step, src_width, src_height, dst, dst_step, dst_width, dst_height, cn,
                                                                       xoffsets, yoffsets, xcoeffs, ycoeffs, min_x, max_x, min_y, max_y, hResize);
    Range range(0, dst_height);
    parallel_for_(range, invoker, dst_width * dst_height / (double)(1 << 16));
}

typedef void(*be_resize_func)(const uchar* src, size_t src_step, int src_width, int src_height,
                                    uchar* dst, size_t dst_step, int dst_width, int dst_height,
                              int cn, double inv_scale_x, double inv_scale_y);

}

namespace cv
{

/************** interpolation formulas and tables ***************/

const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

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

    float sum = 0;
    double y0=-(x+3)*CV_PI*0.25, s0 = std::sin(y0), c0= std::cos(y0);
    for(int i = 0; i < 8; i++ )
    {
        float y0_ = (x+3-i);
        if (fabs(y0_) >= 1e-6f)
        {
            double y = -y0_*CV_PI*0.25;
            coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        }
        else
        {
            // special handling for 'x' values:
            // - ~0.0: 0 0 0 1 0 0 0 0
            // - ~1.0: 0 0 0 0 1 0 0 0
            coeffs[i] = 1e30f;
        }
        sum += coeffs[i];
    }

    sum = 1.f/sum;
    for(int i = 0; i < 8; i++ )
        coeffs[i] *= sum;
}

/**
 * the coordiante transformation from dst to src is linear
 * and can be written as: x_org = f(x) = a * x + b.
 * note: scale may be user input and not equal to (src / dst).
 * ref to onnx, length_resized is src * scale (float), not dst (int).
 */
static Vec2f interCoordinate(int coordinate, int dst, int src, double scale)
{
    float a, b;
    if (coordinate == INTER_HALF_PIXEL
        || coordinate == INTER_HALF_PIXEL_SYMMETRIC
        || coordinate == INTER_HALF_PIXEL_PYTORCH)
    {
        a = static_cast<float>(1.0 / scale);
        b = static_cast<float>(0.5 / scale - 0.5);
        if (coordinate == INTER_HALF_PIXEL_SYMMETRIC)
            b += static_cast<float>(0.5 * (src - dst / scale));
        if (coordinate == INTER_HALF_PIXEL_PYTORCH && dst <= 1)
        {
            a = 0.f;
            b = -0.5f;
        }
    }
    else if (coordinate == INTER_ALIGN_CORNERS)
    {
        a = static_cast<float>((src - 1.0) / (src * scale - 1.0));
        b = 0.f;
    }
    else if (coordinate == INTER_ASYMMETRIC)
    {
        a = static_cast<float>(1.0 / scale);
        b = 0.f;
    }
    else
        CV_Error(Error::StsBadArg, format("Unknown coordinate transformation mode %d", coordinate));
    return Vec2f(a, b);
}

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

/****************************************************************************************\
*                                         Resize                                         *
\****************************************************************************************/

class resizeNNInvoker :
    public ParallelLoopBody
{
public:
    resizeNNInvoker(const Mat& _src, Mat &_dst, int *_x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        ify(_ify)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x, pix_size = (int)src.elemSize();

        for( y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.data + dst.step*y;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.ptr(sy);

            switch( pix_size )
            {
            case 1:
                for( x = 0; x <= dsize.width - 2; x += 2 )
                {
                    uchar t0 = S[x_ofs[x]];
                    uchar t1 = S[x_ofs[x+1]];
                    D[x] = t0;
                    D[x+1] = t1;
                }

                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
                for( x = 0; x < dsize.width; x++ )
                    *(ushort*)(D + x*2) = *(ushort*)(S + x_ofs[x]);
                break;
            case 3:
                for( x = 0; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
                for( x = 0; x < dsize.width; x++ )
                    *(int*)(D + x*4) = *(int*)(S + x_ofs[x]);
                break;
            case 6:
                for( x = 0; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                for( x = 0; x < dsize.width; x++, D += 8 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1];
                }
                break;
            case 12:
                for( x = 0; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofs[x];
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    double ify;

    resizeNNInvoker(const resizeNNInvoker&);
    resizeNNInvoker& operator=(const resizeNNInvoker&);
};

static void
resizeNN( const Mat& src, Mat& dst, double fx, double fy )
{
    Size ssize = src.size(), dsize = dst.size();
    AutoBuffer<int> _x_ofs(dsize.width);
    int* x_ofs = _x_ofs.data();
    int pix_size = (int)src.elemSize();
    double ifx = 1./fx, ify = 1./fy;
    int x;

    for( x = 0; x < dsize.width; x++ )
    {
        int sx = cvFloor(x*ifx);
        x_ofs[x] = std::min(sx, ssize.width-1)*pix_size;
    }

    Range range(0, dsize.height);
#if CV_TRY_AVX2
    if(CV_CPU_HAS_SUPPORT_AVX2 && ((pix_size == 2) || (pix_size == 4)))
    {
        if(pix_size == 2)
            opt_AVX2::resizeNN2_AVX2(range, src, dst, x_ofs, ify);
        else
            opt_AVX2::resizeNN4_AVX2(range, src, dst, x_ofs, ify);
    }
    else
#endif
#if CV_TRY_SSE4_1
    if(CV_CPU_HAS_SUPPORT_SSE4_1 && ((pix_size == 2) || (pix_size == 4)))
    {
        if(pix_size == 2)
            opt_SSE4_1::resizeNN2_SSE4_1(range, src, dst, x_ofs, ify);
        else
            opt_SSE4_1::resizeNN4_SSE4_1(range, src, dst, x_ofs, ify);
    }
    else
#endif
#if CV_TRY_LASX
    if(CV_CPU_HAS_SUPPORT_LASX && ((pix_size == 2) || (pix_size == 4)))
    {
        if(pix_size == 2)
            opt_LASX::resizeNN2_LASX(range, src, dst, x_ofs, ify);
        else
            opt_LASX::resizeNN4_LASX(range, src, dst, x_ofs, ify);
    }
    else
#endif
    {
        resizeNNInvoker invoker(src, dst, x_ofs, ify);
        parallel_for_(range, invoker, dst.total()/(double)(1<<16));
    }
}

class resizeNN_bitexactInvoker : public ParallelLoopBody
{
public:
    resizeNN_bitexactInvoker(const Mat& _src, Mat& _dst, int* _x_ofse, int _ify, int _ify0)
        : src(_src), dst(_dst), x_ofse(_x_ofse), ify(_ify), ify0(_ify0) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int pix_size = (int)src.elemSize();
        for( int y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.ptr(y);
            int _sy = (ify * y + ify0) >> 16;
            int sy = std::min(_sy, ssize.height-1);
            const uchar* S = src.ptr(sy);

            int x = 0;
            switch( pix_size )
            {
            case 1:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes() )
                    v_store(D + x, vx_lut(S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofse[x]];
                break;
            case 2:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, vx_lut((ushort*)S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    *((ushort*)D + x) = *((ushort*)S + x_ofse[x]);
                break;
            case 3:
                for( ; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofse[x] * 3;
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes() )
                    v_store((uint32_t*)D + x, vx_lut((uint32_t*)S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    *((uint32_t*)D + x) = *((uint32_t*)S + x_ofse[x]);
                break;
            case 6:
                for( ; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofse[x]*6);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint64>::vlanes(); x += VTraits<v_uint64>::vlanes() )
                    v_store((uint64_t*)D + x, vx_lut((uint64_t*)S, x_ofse + x));
#endif
                for( ; x < dsize.width; x++ )
                    *((uint64_t*)D + x) = *((uint64_t*)S + x_ofse[x]);
                break;
            case 12:
                for( ; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofse[x]*12);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofse[x] * pix_size;
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }
private:
    const Mat& src;
    Mat& dst;
    int* x_ofse;
    const int ify;
    const int ify0;
};

static void resizeNN_bitexact( const Mat& src, Mat& dst, double /*fx*/, double /*fy*/ )
{
    Size ssize = src.size(), dsize = dst.size();
    int ifx = ((ssize.width << 16) + dsize.width / 2) / dsize.width; // 16bit fixed-point arithmetic
    int ifx0 = ifx / 2 - ssize.width % 2;                       // This method uses center pixel coordinate as Pillow and scikit-images do.
    int ify = ((ssize.height << 16) + dsize.height / 2) / dsize.height;
    int ify0 = ify / 2 - ssize.height % 2;

    cv::utils::BufferArea area;
    int* x_ofse = 0;
    area.allocate(x_ofse, dsize.width, CV_SIMD_WIDTH);
    area.commit();

    for( int x = 0; x < dsize.width; x++ )
    {
        int sx = (ifx * x + ifx0) >> 16;
        x_ofse[x] = std::min(sx, ssize.width-1);    // offset in element (not byte)
    }
    Range range(0, dsize.height);
    resizeNN_bitexactInvoker invoker(src, dst, x_ofse, ify, ify0);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

class ResizeOnnxNNInvoker : public ParallelLoopBody
{
    Mat src;
    Mat& dst;
    Matx22f M;
    int mode;
    float offset;
    AutoBuffer<size_t> x_ofs;
    ResizeOnnxNNInvoker(const ResizeOnnxNNInvoker&);
    ResizeOnnxNNInvoker& operator=(const ResizeOnnxNNInvoker&);

    int srcIndex(int x, float a, float b) const
    {
        // offset can not add to M(0, 1) and M(1, 1) directly
        // due to the small float error near integer
        float f = fmaf(static_cast<float>(x), a, b);
        if (mode == INTER_NEAREST_PREFER_FLOOR ||
            mode == INTER_NEAREST_CEIL)
            x = cvCeil(f + offset);
        else
            x = cvFloor(f + offset);
        return x;
    }

public:
    ResizeOnnxNNInvoker(Mat const& _src, Mat& _dst, const Matx22f& _M, int _mode)
        : src(_src), dst(_dst), M(_M), mode(_mode)
    {
        offset = 0.f;
        if (mode == INTER_NEAREST_PREFER_FLOOR)
            offset = -0.5f;
        if (mode == INTER_NEAREST_PREFER_CEIL)
            offset = +0.5f;

        x_ofs.allocate(dst.cols);
        size_t pix_size = src.elemSize();
        for (int x = 0; x < dst.cols; ++x)
        {
            int sx = srcIndex(x, M(0, 0), M(0, 1));
            sx = min(max(sx, 0), src.cols - 1);
            x_ofs[x] = sx * pix_size;
        }
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int width = dst.cols;
        size_t pix_size = src.elemSize();
        for (int y = range.start; y < range.end; ++y)
        {
            uchar* D = dst.ptr(y);
            int sy = srcIndex(y, M(1, 0), M(1, 1));
            sy = min(max(sy, 0), src.rows - 1);
            uchar const* S = src.ptr(sy);
            int x = 0;

            switch (pix_size)
            {
            case 1:
                for (; x <= width - 2; x += 2)
                {
                    uchar t0 = S[x_ofs[x    ]];
                    uchar t1 = S[x_ofs[x + 1]];
                    D[x    ] = t0;
                    D[x + 1] = t1;
                }
                for (; x < width; ++x)
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
                for (; x < width; ++x)
                    reinterpret_cast<ushort*>(D)[x] = *(reinterpret_cast<ushort const*>(S + x_ofs[x]));
                break;
            case 3:
                for (; x < width; ++x, D += 3)
                {
                    uchar const* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
                for (; x < width; ++x)
                    reinterpret_cast<uint*>(D)[x] = *(reinterpret_cast<uint const*>(S + x_ofs[x]));
                break;
            case 6:
                for (; x < width; ++x, D += 6)
                {
                    ushort const* _tS = reinterpret_cast<ushort const*>(S + x_ofs[x]);
                    ushort* _tD = reinterpret_cast<ushort*>(D);
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                for (; x < width; ++x)
                    reinterpret_cast<uint64*>(D)[x] = *(reinterpret_cast<uint64 const*>(S + x_ofs[x]));
                break;
            case 12:
                for (; x < width; ++x, D += 12)
                {
                    uint const* _tS = reinterpret_cast<uint const*>(S + x_ofs[x]);
                    uint* _tD = reinterpret_cast<uint*>(D);
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
#if CV_SIMD128
            case 16:
                for (; x < width; ++x, D += 16)
                    v_store(D, v_load(S + x_ofs[x]));
                break;
#endif
            default:
                for (; x < width; ++x, D += pix_size)
                {
                    uchar const* _tS = S + x_ofs[x];
                    for (size_t k = 0; k < pix_size; ++k)
                        D[k] = _tS[k];
                }
            }
        }
    }
};

struct VResizeNoVec
{
    template<typename WT, typename T, typename BT>
    int operator()(const WT**, T*, const BT*, int ) const
    {
        return 0;
    }
};

struct HResizeNoVec
{
    template<typename T, typename WT, typename AT> inline
    int operator()(const T**, WT**, int, const int*,
        const AT*, int, int, int, int, int) const
    {
        return 0;
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)

struct VResizeLinearVec_32s8u
{
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1];
        int x = 0;
        v_int16 b0 = vx_setall_s16(beta[0]), b1 = vx_setall_s16(beta[1]);

        if( (((size_t)S0|(size_t)S1)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                v_store(dst + x, v_rshr_pack_u<2>(v_add(v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S0 + x)), v_shr<4>(vx_load_aligned(S0 + x + VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S1 + x)), v_shr<4>(vx_load_aligned(S1 + x + VTraits<v_int32>::vlanes()))), b1)),
                                                  v_add(v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S0 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load_aligned(S0 + x + 3 * VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S1 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load_aligned(S1 + x + 3 * VTraits<v_int32>::vlanes()))), b1))));
        else
            for( ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                v_store(dst + x, v_rshr_pack_u<2>(v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x)), v_shr<4>(vx_load(S0 + x + VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x)), v_shr<4>(vx_load(S1 + x + VTraits<v_int32>::vlanes()))), b1)),
                                                  v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load(S0 + x + 3 * VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x + 2 * VTraits<v_int32>::vlanes())), v_shr<4>(vx_load(S1 + x + 3 * VTraits<v_int32>::vlanes()))), b1))));

        for( ; x < width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
            v_rshr_pack_u_store<2>(dst + x, v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x)), v_shr<4>(vx_load(S0 + x + VTraits<v_int32>::vlanes()))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x)), v_shr<4>(vx_load(S1 + x + VTraits<v_int32>::vlanes()))), b1)));

        return x;
    }
};

struct VResizeLinearVec_32f16u
{
    int operator()(const float** src, ushort* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1];
        int x = 0;

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]);

        if( (((size_t)S0|(size_t)S1)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
                v_store(dst + x, v_pack_u(v_round(v_muladd(vx_load_aligned(S0 + x                    ), b0, v_mul(vx_load_aligned(S1 + x), b1))),
                                          v_round(v_muladd(vx_load_aligned(S0 + x + VTraits<v_float32>::vlanes()), b0, v_mul(vx_load_aligned(S1 + x + VTraits<v_float32>::vlanes()), b1)))));
        else
            for (; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
                v_store(dst + x, v_pack_u(v_round(v_muladd(vx_load(S0 + x                    ), b0, v_mul(vx_load(S1 + x), b1))),
                                          v_round(v_muladd(vx_load(S0 + x + VTraits<v_float32>::vlanes()), b0, v_mul(vx_load(S1 + x + VTraits<v_float32>::vlanes()), b1)))));
        for( ; x < width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
        {
            v_int32 t0 = v_round(v_muladd(vx_load(S0 + x), b0, v_mul(vx_load(S1 + x), b1)));
            v_store_low(dst + x, v_pack_u(t0, t0));
        }

        return x;
    }
};

struct VResizeLinearVec_32f16s
{
    int operator()(const float** src, short* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1];
        int x = 0;

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]);

        if( (((size_t)S0|(size_t)S1)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
                v_store(dst + x, v_pack(v_round(v_muladd(vx_load_aligned(S0 + x                    ), b0, v_mul(vx_load_aligned(S1 + x), b1))),
                                        v_round(v_muladd(vx_load_aligned(S0 + x + VTraits<v_float32>::vlanes()), b0, v_mul(vx_load_aligned(S1 + x + VTraits<v_float32>::vlanes()), b1)))));
        else
            for (; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
                v_store(dst + x, v_pack(v_round(v_muladd(vx_load(S0 + x                    ), b0, v_mul(vx_load(S1 + x), b1))),
                                        v_round(v_muladd(vx_load(S0 + x + VTraits<v_float32>::vlanes()), b0, v_mul(vx_load(S1 + x + VTraits<v_float32>::vlanes()), b1)))));
        for( ; x < width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
        {
            v_int32 t0 = v_round(v_muladd(vx_load(S0 + x), b0, v_mul(vx_load(S1 + x), b1)));
            v_store_low(dst + x, v_pack(t0, t0));
        }

        return x;
    }
};

struct VResizeLinearVec_32f
{
    int operator()(const float** src, float* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1];
        int x = 0;

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]);

        if( (((size_t)S0|(size_t)S1)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
                v_store(dst + x, v_muladd(vx_load_aligned(S0 + x), b0, v_mul(vx_load_aligned(S1 + x), b1)));
        else
            for( ; x <= width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
                v_store(dst + x, v_muladd(vx_load(S0 + x), b0, v_mul(vx_load(S1 + x), b1)));

        return x;
    }
};


struct VResizeCubicVec_32s8u
{
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        float scale = 1.f/(INTER_RESIZE_COEF_SCALE*INTER_RESIZE_COEF_SCALE);

        v_float32 b0 = vx_setall_f32(beta[0] * scale), b1 = vx_setall_f32(beta[1] * scale),
                  b2 = vx_setall_f32(beta[2] * scale), b3 = vx_setall_f32(beta[3] * scale);

        if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&(VTraits<v_uint8>::vlanes() - 1)) == 0 )
            for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
                v_pack_u_store(dst + x, v_pack(v_round(v_muladd(v_cvt_f32(vx_load_aligned(S0 + x                    )),  b0,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S1 + x                    )),  b1,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S2 + x                    )),  b2,
                                                                v_mul(v_cvt_f32(vx_load_aligned(S3 + x)), b3))))),
                                               v_round(v_muladd(v_cvt_f32(vx_load_aligned(S0 + x + VTraits<v_float32>::vlanes())),  b0,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S1 + x + VTraits<v_float32>::vlanes())),  b1,
                                                       v_muladd(v_cvt_f32(vx_load_aligned(S2 + x + VTraits<v_float32>::vlanes())),  b2,
                                                                v_mul(v_cvt_f32(vx_load_aligned(S3 + x + VTraits<v_float32>::vlanes())), b3)))))));
        else
            for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
                v_pack_u_store(dst + x, v_pack(v_round(v_muladd(v_cvt_f32(vx_load(S0 + x                    )),  b0,
                                                       v_muladd(v_cvt_f32(vx_load(S1 + x                    )),  b1,
                                                       v_muladd(v_cvt_f32(vx_load(S2 + x                    )),  b2,
                                                                v_mul(v_cvt_f32(vx_load(S3 + x)), b3))))),
                                               v_round(v_muladd(v_cvt_f32(vx_load(S0 + x + VTraits<v_float32>::vlanes())),  b0,
                                                       v_muladd(v_cvt_f32(vx_load(S1 + x + VTraits<v_float32>::vlanes())),  b1,
                                                       v_muladd(v_cvt_f32(vx_load(S2 + x + VTraits<v_float32>::vlanes())),  b2,
                                                                v_mul(v_cvt_f32(vx_load(S3 + x + VTraits<v_float32>::vlanes())), b3)))))));
        return x;
    }
};

struct VResizeCubicVec_32f16u
{
    int operator()(const float** src, ushort* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]);

        for (; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
            v_store(dst + x, v_pack_u(v_round(v_muladd(vx_load(S0 + x                    ),  b0,
                                              v_muladd(vx_load(S1 + x                    ),  b1,
                                              v_muladd(vx_load(S2 + x                    ),  b2,
                                                       v_mul(vx_load(S3 + x), b3))))),
                                      v_round(v_muladd(vx_load(S0 + x + VTraits<v_float32>::vlanes()),  b0,
                                              v_muladd(vx_load(S1 + x + VTraits<v_float32>::vlanes()),  b1,
                                              v_muladd(vx_load(S2 + x + VTraits<v_float32>::vlanes()),  b2,
                                                       v_mul(vx_load(S3 + x + VTraits<v_float32>::vlanes()), b3)))))));

        return x;
    }
};

struct VResizeCubicVec_32f16s
{
    int operator()(const float** src, short* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]);

        for (; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
            v_store(dst + x, v_pack(v_round(v_muladd(vx_load(S0 + x                    ),  b0,
                                            v_muladd(vx_load(S1 + x                    ),  b1,
                                            v_muladd(vx_load(S2 + x                    ),  b2,
                                                     v_mul(vx_load(S3 + x), b3))))),
                                    v_round(v_muladd(vx_load(S0 + x + VTraits<v_float32>::vlanes()),  b0,
                                            v_muladd(vx_load(S1 + x + VTraits<v_float32>::vlanes()),  b1,
                                            v_muladd(vx_load(S2 + x + VTraits<v_float32>::vlanes()),  b2,
                                                     v_mul(vx_load(S3 + x + VTraits<v_float32>::vlanes()), b3)))))));

        return x;
    }
};

struct VResizeCubicVec_32f
{
    int operator()(const float** src, float* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]);

        for( ; x <= width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
            v_store(dst + x, v_muladd(vx_load(S0 + x),  b0,
                             v_muladd(vx_load(S1 + x),  b1,
                             v_muladd(vx_load(S2 + x),  b2,
                                      v_mul(vx_load(S3 + x), b3)))));

        return x;
    }
};


#if CV_TRY_SSE4_1

struct VResizeLanczos4Vec_32f16u
{
    int operator()(const float** src, ushort* dst, const float* beta, int width) const
    {
        if (CV_CPU_HAS_SUPPORT_SSE4_1)
            return opt_SSE4_1::VResizeLanczos4Vec_32f16u_SSE41(src, dst, beta, width);
        else
            return 0;
    }
};

#else

struct VResizeLanczos4Vec_32f16u
{
    int operator()(const float** src, ushort* dst, const float* beta, int width ) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
                    *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
        int x = 0;
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]),
                  b4 = vx_setall_f32(beta[4]), b5 = vx_setall_f32(beta[5]),
                  b6 = vx_setall_f32(beta[6]), b7 = vx_setall_f32(beta[7]);

        for( ; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
            v_store(dst + x, v_pack_u(v_round(v_muladd(vx_load(S0 + x                    ),  b0,
                                              v_muladd(vx_load(S1 + x                    ),  b1,
                                              v_muladd(vx_load(S2 + x                    ),  b2,
                                              v_muladd(vx_load(S3 + x                    ),  b3,
                                              v_muladd(vx_load(S4 + x                    ),  b4,
                                              v_muladd(vx_load(S5 + x                    ),  b5,
                                              v_muladd(vx_load(S6 + x                    ),  b6,
                                                       v_mul(vx_load(S7 + x                    ), b7))))))))),
                                      v_round(v_muladd(vx_load(S0 + x + VTraits<v_float32>::vlanes()),  b0,
                                              v_muladd(vx_load(S1 + x + VTraits<v_float32>::vlanes()),  b1,
                                              v_muladd(vx_load(S2 + x + VTraits<v_float32>::vlanes()),  b2,
                                              v_muladd(vx_load(S3 + x + VTraits<v_float32>::vlanes()),  b3,
                                              v_muladd(vx_load(S4 + x + VTraits<v_float32>::vlanes()),  b4,
                                              v_muladd(vx_load(S5 + x + VTraits<v_float32>::vlanes()),  b5,
                                              v_muladd(vx_load(S6 + x + VTraits<v_float32>::vlanes()),  b6,
                                                       v_mul(vx_load(S7 + x + VTraits<v_float32>::vlanes()), b7)))))))))));

        return x;
    }
};

#endif

struct VResizeLanczos4Vec_32f16s
{
    int operator()(const float** src, short* dst, const float* beta, int width ) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
                    *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
        int x = 0;
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]),
                  b4 = vx_setall_f32(beta[4]), b5 = vx_setall_f32(beta[5]),
                  b6 = vx_setall_f32(beta[6]), b7 = vx_setall_f32(beta[7]);

        for( ; x <= width - VTraits<v_int16>::vlanes(); x += VTraits<v_int16>::vlanes())
            v_store(dst + x, v_pack(v_round(v_muladd(vx_load(S0 + x                    ),  b0,
                                            v_muladd(vx_load(S1 + x                    ),  b1,
                                            v_muladd(vx_load(S2 + x                    ),  b2,
                                            v_muladd(vx_load(S3 + x                    ),  b3,
                                            v_muladd(vx_load(S4 + x                    ),  b4,
                                            v_muladd(vx_load(S5 + x                    ),  b5,
                                            v_muladd(vx_load(S6 + x                    ),  b6,
                                                     v_mul(vx_load(S7 + x), b7))))))))),
                                    v_round(v_muladd(vx_load(S0 + x + VTraits<v_float32>::vlanes()),  b0,
                                            v_muladd(vx_load(S1 + x + VTraits<v_float32>::vlanes()),  b1,
                                            v_muladd(vx_load(S2 + x + VTraits<v_float32>::vlanes()),  b2,
                                            v_muladd(vx_load(S3 + x + VTraits<v_float32>::vlanes()),  b3,
                                            v_muladd(vx_load(S4 + x + VTraits<v_float32>::vlanes()),  b4,
                                            v_muladd(vx_load(S5 + x + VTraits<v_float32>::vlanes()),  b5,
                                            v_muladd(vx_load(S6 + x + VTraits<v_float32>::vlanes()),  b6,
                                                     v_mul(vx_load(S7 + x + VTraits<v_float32>::vlanes()), b7)))))))))));

        return x;
    }
};

struct VResizeLanczos4Vec_32f
{
    int operator()(const float** src, float* dst, const float* beta, int width ) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
                    *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
        int x = 0;

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]),
                  b4 = vx_setall_f32(beta[4]), b5 = vx_setall_f32(beta[5]),
                  b6 = vx_setall_f32(beta[6]), b7 = vx_setall_f32(beta[7]);

        for( ; x <= width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
            v_store(dst + x, v_muladd(vx_load(S0 + x),  b0,
                             v_muladd(vx_load(S1 + x),  b1,
                             v_muladd(vx_load(S2 + x),  b2,
                             v_muladd(vx_load(S3 + x),  b3,
                             v_muladd(vx_load(S4 + x),  b4,
                             v_muladd(vx_load(S5 + x),  b5,
                             v_muladd(vx_load(S6 + x),  b6,
                                      v_mul(vx_load(S7 + x), b7)))))))));

        return x;
    }
};

#else

typedef VResizeNoVec VResizeLinearVec_32s8u;
typedef VResizeNoVec VResizeLinearVec_32f16u;
typedef VResizeNoVec VResizeLinearVec_32f16s;
typedef VResizeNoVec VResizeLinearVec_32f;

typedef VResizeNoVec VResizeCubicVec_32s8u;
typedef VResizeNoVec VResizeCubicVec_32f16u;
typedef VResizeNoVec VResizeCubicVec_32f16s;
typedef VResizeNoVec VResizeCubicVec_32f;

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;
typedef VResizeNoVec VResizeLanczos4Vec_32f16s;
typedef VResizeNoVec VResizeLanczos4Vec_32f;

#endif

#if CV_SIMD128

template<typename ST, typename DT, typename AT, typename DVT>
struct HResizeLinearVec_X4
{
    int operator()(const ST** src, DT** dst, int count, const int* xofs,
        const AT* alpha, int, int, int cn, int, int xmax) const
    {
        const int nlanes = 4;
        const int len0 = xmax & -nlanes;
        int dx = 0, k = 0;

        for( ; k <= (count - 2); k+=2 )
        {
            const ST *S0 = src[k];
            DT *D0 = dst[k];
            const ST *S1 = src[k+1];
            DT *D1 = dst[k+1];

            for( dx = 0; dx < len0; dx += nlanes )
            {
                int sx0 = xofs[dx+0];
                int sx1 = xofs[dx+1];
                int sx2 = xofs[dx+2];
                int sx3 = xofs[dx+3];
                DVT a_even;
                DVT a_odd;

                v_load_deinterleave(&alpha[dx*2], a_even, a_odd);
                DVT s0(S0[sx0], S0[sx1], S0[sx2], S0[sx3]);
                DVT s1(S0[sx0+cn], S0[sx1+cn], S0[sx2+cn], S0[sx3+cn]);
                DVT s0_u(S1[sx0], S1[sx1], S1[sx2], S1[sx3]);
                DVT s1_u(S1[sx0+cn], S1[sx1+cn], S1[sx2+cn], S1[sx3+cn]);
                v_store(&D1[dx], v_add(v_mul(s0_u, a_even), v_mul(s1_u, a_odd)));
                v_store(&D0[dx], v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
            }
        }
        for( ; k < count; k++ )
        {
            const ST *S = src[k];
            DT *D = dst[k];
            for( dx = 0; dx < len0; dx += nlanes )
            {
                int sx0 = xofs[dx+0];
                int sx1 = xofs[dx+1];
                int sx2 = xofs[dx+2];
                int sx3 = xofs[dx+3];
                DVT a_even;
                DVT a_odd;

                v_load_deinterleave(&alpha[dx*2], a_even, a_odd);
                DVT s0(S[sx0], S[sx1], S[sx2], S[sx3]);
                DVT s1(S[sx0+cn], S[sx1+cn], S[sx2+cn], S[sx3+cn]);
                v_store(&D[dx], v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
            }
        }
        return dx;
    }
};

struct HResizeLinearVecU8_X4
{
    int operator()(const uchar** src, int** dst, int count, const int* xofs,
        const short* alpha/*[xmax]*/, int /*smax*/, int dmax, int cn, int /*xmin*/, int xmax) const
    {
        int dx = 0, k = 0;

        if(cn == 1)
        {
            const int step = 8;
            const int len0 = xmax & -step;
            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_lut_pairs(S0, xofs+dx), sl, sh);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D0[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                    v_expand(v_lut_pairs(S1, xofs+dx), sl, sh);
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D1[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_lut_pairs(S, xofs+dx), sl, sh);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
        }
        else if(cn == 2)
        {
            const int step = 8;
            const int len0 = xmax & -step;
            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; dx < len0; dx += step )
                {
                    int ofs[4] = { xofs[dx], xofs[dx + 2], xofs[dx + 4], xofs[dx + 6] };
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_interleave_pairs(v_lut_quads(S0, ofs)), sl, sh);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D0[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                    v_expand(v_interleave_pairs(v_lut_quads(S1, ofs)), sl, sh);
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D1[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; dx < len0; dx += step )
                {
                    int ofs[4] = { xofs[dx], xofs[dx + 2], xofs[dx + 4], xofs[dx + 6] };
                    v_int16x8 al = v_load(alpha+dx*2);
                    v_int16x8 ah = v_load(alpha+dx*2+8);
                    v_uint16x8 sl, sh;
                    v_expand(v_interleave_pairs(v_lut_quads(S, ofs)), sl, sh);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(sl), al));
                    v_store(&D[dx+4], v_dotprod(v_reinterpret_as_s16(sh), ah));
                }
            }
        }
        else if(cn == 3)
        {
            /* Peek at the last x offset to find the maximal s offset.  We know the loop
               will terminate prior to value which may be 1 or more elements prior to the
               final valid offset. xofs[] is constucted to be an array of increasingly
               large offsets (i.e xofs[x] <= xofs[x+1] for x < xmax). */
            int smax = xofs[dmax-cn];

            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; (xofs[dx] + cn) < smax; dx += cn )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S0 + xofs[dx]), v_shl<16>(v_load_expand_q(S0 + xofs[dx] + cn)))), a));
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S1 + xofs[dx]), v_shl<16>(v_load_expand_q(S1 + xofs[dx] + cn)))), a));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; (xofs[dx] + cn) < smax; dx += cn )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S + xofs[dx]), v_shl<16>(v_load_expand_q(S + xofs[dx] + cn)))), a));
                }
            }
            /* Debug check to ensure truthiness that we never vector the final value. */
            CV_DbgAssert(dx < dmax);
        }
        else if(cn == 4)
        {
            const int step = 4;
            const int len0 = xmax & -step;
            for( ; k <= (count - 2); k+=2 )
            {
                const uchar *S0 = src[k];
                int *D0 = dst[k];
                const uchar *S1 = src[k+1];
                int *D1 = dst[k+1];

                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D0[dx], v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S0+xofs[dx]))), a));
                    v_store(&D1[dx], v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S1+xofs[dx]))), a));
                }
            }
            for( ; k < count; k++ )
            {
                const uchar *S = src[k];
                int *D = dst[k];
                for( dx = 0; dx < len0; dx += step )
                {
                    v_int16x8 a = v_load(alpha+dx*2);
                    v_store(&D[dx], v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S+xofs[dx]))), a));
                }
            }
        }
        else
        {
            return 0;  // images with channels >4 are out of optimization scope
        }
        return dx;
    }
};

typedef HResizeLinearVec_X4<float,float,float,v_float32x4> HResizeLinearVec_32f;
typedef HResizeLinearVec_X4<ushort,float,float,v_float32x4> HResizeLinearVec_16u32f;
typedef HResizeLinearVec_X4<short,float,float,v_float32x4> HResizeLinearVec_16s32f;
typedef HResizeLinearVecU8_X4 HResizeLinearVec_8u32s;

#else

typedef HResizeNoVec HResizeLinearVec_8u32s;
typedef HResizeNoVec HResizeLinearVec_16u32f;
typedef HResizeNoVec HResizeLinearVec_16s32f;
typedef HResizeNoVec HResizeLinearVec_32f;

#endif

typedef HResizeNoVec HResizeLinearVec_64f;


template<typename T, typename WT, typename AT, int ONE, class VecOp>
struct HResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        int dx, k;
        VecOp vecOp;

        int dx0 = vecOp(src, dst, count,
            xofs, alpha, swidth, dwidth, cn, xmin, xmax );

        for( k = 0; k <= count - 2; k+=2 )
        {
            const T *S0 = src[k], *S1 = src[k+1];
            WT *D0 = dst[k], *D1 = dst[k+1];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                WT a0 = alpha[dx*2], a1 = alpha[dx*2+1];
                WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
                WT t1 = S1[sx]*a0 + S1[sx + cn]*a1;
                D0[dx] = t0; D1[dx] = t1;
            }

            for( ; dx < dwidth; dx++ )
            {
                int sx = xofs[dx];
                D0[dx] = WT(S0[sx]*ONE); D1[dx] = WT(S1[sx]*ONE);
            }
        }

        for( ; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                D[dx] = S[sx]*alpha[dx*2] + S[sx+cn]*alpha[dx*2+1];
            }

            for( ; dx < dwidth; dx++ )
                D[dx] = WT(S[xofs[dx]]*ONE);
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1];
        const WT *S0 = src[0], *S1 = src[1];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT t0, t1;
            t0 = S0[x]*b0 + S1[x]*b1;
            t1 = S0[x+1]*b0 + S1[x+1]*b1;
            dst[x] = castOp(t0); dst[x+1] = castOp(t1);
            t0 = S0[x+2]*b0 + S1[x+2]*b1;
            t1 = S0[x+3]*b0 + S1[x+3]*b1;
            dst[x+2] = castOp(t0); dst[x+3] = castOp(t1);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1);
    }
};

template<>
struct VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>, VResizeLinearVec_32s8u>
{
    typedef uchar value_type;
    typedef int buf_type;
    typedef short alpha_type;

    void operator()(const buf_type** src, value_type* dst, const alpha_type* beta, int width ) const
    {
        alpha_type b0 = beta[0], b1 = beta[1];
        const buf_type *S0 = src[0], *S1 = src[1];
        VResizeLinearVec_32s8u vecOp;

        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            dst[x+0] = uchar(( ((b0 * (S0[x+0] >> 4)) >> 16) + ((b1 * (S1[x+0] >> 4)) >> 16) + 2)>>2);
            dst[x+1] = uchar(( ((b0 * (S0[x+1] >> 4)) >> 16) + ((b1 * (S1[x+1] >> 4)) >> 16) + 2)>>2);
            dst[x+2] = uchar(( ((b0 * (S0[x+2] >> 4)) >> 16) + ((b1 * (S1[x+2] >> 4)) >> 16) + 2)>>2);
            dst[x+3] = uchar(( ((b0 * (S0[x+3] >> 4)) >> 16) + ((b1 * (S1[x+3] >> 4)) >> 16) + 2)>>2);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
    }
};


template<typename T, typename WT, typename AT>
struct HResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += 4 )
                {
                    int j, sx = xofs[dx] - cn;
                    WT v = 0;
                    for( j = 0; j < 4; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += 4 )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx-cn]*alpha[0] + S[sx]*alpha[1] +
                        S[sx+cn]*alpha[2] + S[sx+cn*2]*alpha[3];
                }
                limit = dwidth;
            }
            alpha -= dwidth*4;
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp(src, dst, beta, width);
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1 + S2[x]*b2 + S3[x]*b3);
    }
};


template<typename T, typename WT, typename AT>
struct HResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += 8 )
                {
                    int j, sx = xofs[dx] - cn*3;
                    WT v = 0;
                    for( j = 0; j < 8; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += 8 )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx-cn*3]*alpha[0] + S[sx-cn*2]*alpha[1] +
                        S[sx-cn]*alpha[2] + S[sx]*alpha[3] +
                        S[sx+cn]*alpha[4] + S[sx+cn*2]*alpha[5] +
                        S[sx+cn*3]*alpha[6] + S[sx+cn*4]*alpha[7];
                }
                limit = dwidth;
            }
            alpha -= dwidth*8;
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        CastOp castOp;
        VecOp vecOp;
        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT b = beta[0];
            const WT* S = src[0];
            WT s0 = S[x]*b, s1 = S[x+1]*b, s2 = S[x+2]*b, s3 = S[x+3]*b;

            for( int k = 1; k < 8; k++ )
            {
                b = beta[k]; S = src[k];
                s0 += S[x]*b; s1 += S[x+1]*b;
                s2 += S[x+2]*b; s3 += S[x+3]*b;
            }

            dst[x] = castOp(s0); dst[x+1] = castOp(s1);
            dst[x+2] = castOp(s2); dst[x+3] = castOp(s3);
        }
        #endif
        for( ; x < width; x++ )
        {
            dst[x] = castOp(src[0][x]*beta[0] + src[1][x]*beta[1] +
                src[2][x]*beta[2] + src[3][x]*beta[3] + src[4][x]*beta[4] +
                src[5][x]*beta[5] + src[6][x]*beta[6] + src[7][x]*beta[7]);
        }
    }
};


static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

static const int MAX_ESIZE=16;

template <typename HResize, typename VResize>
class resizeGeneric_Invoker :
    public ParallelLoopBody
{
public:
    typedef typename HResize::value_type T;
    typedef typename HResize::buf_type WT;
    typedef typename HResize::alpha_type AT;

    resizeGeneric_Invoker(const Mat& _src, Mat &_dst, const int *_xofs, const int *_yofs,
        const AT* _alpha, const AT* __beta, const Size& _ssize, const Size &_dsize,
        int _ksize, int _xmin, int _xmax) :
        ParallelLoopBody(), src(_src), dst(_dst), xofs(_xofs), yofs(_yofs),
        alpha(_alpha), _beta(__beta), ssize(_ssize), dsize(_dsize),
        ksize(_ksize), xmin(_xmin), xmax(_xmax)
    {
        CV_Assert(ksize <= MAX_ESIZE);
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int dy, cn = src.channels();
        HResize hresize;
        VResize vresize;

        int bufstep = (int)alignSize(dsize.width, 16);
        AutoBuffer<WT> _buffer(bufstep*ksize);
        const T* srows[MAX_ESIZE]={0};
        WT* rows[MAX_ESIZE]={0};
        int prev_sy[MAX_ESIZE];

        for(int k = 0; k < ksize; k++ )
        {
            prev_sy[k] = -1;
            rows[k] = _buffer.data() + bufstep*k;
        }

        const AT* beta = _beta + ksize * range.start;

        for( dy = range.start; dy < range.end; dy++, beta += ksize )
        {
            int sy0 = yofs[dy], k0=ksize, k1=0, ksize2 = ksize/2;

            for(int k = 0; k < ksize; k++ )
            {
                int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
                for( k1 = std::max(k1, k); k1 < ksize; k1++ )
                {
                    if( k1 < MAX_ESIZE && sy == prev_sy[k1] ) // if the sy-th row has been computed already, reuse it.
                    {
                        if( k1 > k )
                            memcpy( rows[k], rows[k1], bufstep*sizeof(rows[0][0]) );
                        break;
                    }
                }
                if( k1 == ksize )
                    k0 = std::min(k0, k); // remember the first row that needs to be computed
                srows[k] = src.template ptr<T>(sy);
                prev_sy[k] = sy;
            }

            if( k0 < ksize )
                hresize( (const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha),
                        ssize.width, dsize.width, cn, xmin, xmax );
            vresize( (const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width );
        }
    }

private:
    Mat src;
    Mat dst;
    const int* xofs, *yofs;
    const AT* alpha, *_beta;
    Size ssize, dsize;
    const int ksize, xmin, xmax;

    resizeGeneric_Invoker& operator = (const resizeGeneric_Invoker&);
};

template<class HResize, class VResize>
static void resizeGeneric_( const Mat& src, Mat& dst,
                            const int* xofs, const void* _alpha,
                            const int* yofs, const void* _beta,
                            int xmin, int xmax, int ksize )
{
    typedef typename HResize::alpha_type AT;

    const AT* beta = (const AT*)_beta;
    Size ssize = src.size(), dsize = dst.size();
    int cn = src.channels();
    ssize.width *= cn;
    dsize.width *= cn;
    xmin *= cn;
    xmax *= cn;
    // image resize is a separable operation. In case of not too strong

    Range range(0, dsize.height);
    resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
        ssize, dsize, ksize, xmin, xmax);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

template <typename T, typename WT>
struct ResizeAreaFastNoVec
{
    ResizeAreaFastNoVec(int, int) { }
    ResizeAreaFastNoVec(int, int, int, int) { }
    int operator() (const T*, T*, int) const
    { return 0; }
};

#if CV_NEON

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const uchar* S, uchar* D, int w) const
    {
        int dx = 0;
        const uchar* S0 = S, * S1 = S0 + step;

        uint16x8_t v_2 = vdupq_n_u16(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 16; dx += 16, S0 += 32, S1 += 32, D += 16)
            {
                uint8x16x2_t v_row0 = vld2q_u8(S0), v_row1 = vld2q_u8(S1);

                uint16x8_t v_dst0 = vaddl_u8(vget_low_u8(v_row0.val[0]), vget_low_u8(v_row0.val[1]));
                v_dst0 = vaddq_u16(v_dst0, vaddl_u8(vget_low_u8(v_row1.val[0]), vget_low_u8(v_row1.val[1])));
                v_dst0 = vshrq_n_u16(vaddq_u16(v_dst0, v_2), 2);

                uint16x8_t v_dst1 = vaddl_u8(vget_high_u8(v_row0.val[0]), vget_high_u8(v_row0.val[1]));
                v_dst1 = vaddq_u16(v_dst1, vaddl_u8(vget_high_u8(v_row1.val[0]), vget_high_u8(v_row1.val[1])));
                v_dst1 = vshrq_n_u16(vaddq_u16(v_dst1, v_2), 2);

                vst1q_u8(D, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint8x16_t v_row0 = vld1q_u8(S0), v_row1 = vld1q_u8(S1);

                uint16x8_t v_row00 = vmovl_u8(vget_low_u8(v_row0));
                uint16x8_t v_row01 = vmovl_u8(vget_high_u8(v_row0));
                uint16x8_t v_row10 = vmovl_u8(vget_low_u8(v_row1));
                uint16x8_t v_row11 = vmovl_u8(vget_high_u8(v_row1));

                uint16x4_t v_p0 = vadd_u16(vadd_u16(vget_low_u16(v_row00), vget_high_u16(v_row00)),
                                           vadd_u16(vget_low_u16(v_row10), vget_high_u16(v_row10)));
                uint16x4_t v_p1 = vadd_u16(vadd_u16(vget_low_u16(v_row01), vget_high_u16(v_row01)),
                                           vadd_u16(vget_low_u16(v_row11), vget_high_u16(v_row11)));
                uint16x8_t v_dst = vshrq_n_u16(vaddq_u16(vcombine_u16(v_p0, v_p1), v_2), 2);

                vst1_u8(D, vmovn_u16(v_dst));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const ushort * S, ushort * D, int w) const
    {
        int dx = 0;
        const ushort * S0 = S, * S1 = (const ushort *)((const uchar *)(S0) + step);

        uint32x4_t v_2 = vdupq_n_u32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint16x8x2_t v_row0 = vld2q_u16(S0), v_row1 = vld2q_u16(S1);

                uint32x4_t v_dst0 = vaddl_u16(vget_low_u16(v_row0.val[0]), vget_low_u16(v_row0.val[1]));
                v_dst0 = vaddq_u32(v_dst0, vaddl_u16(vget_low_u16(v_row1.val[0]), vget_low_u16(v_row1.val[1])));
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_2), 2);

                uint32x4_t v_dst1 = vaddl_u16(vget_high_u16(v_row0.val[0]), vget_high_u16(v_row0.val[1]));
                v_dst1 = vaddq_u32(v_dst1, vaddl_u16(vget_high_u16(v_row1.val[0]), vget_high_u16(v_row1.val[1])));
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_2), 2);

                vst1q_u16(D, vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                uint16x8_t v_row0 = vld1q_u16(S0), v_row1 = vld1q_u16(S1);
                uint32x4_t v_dst = vaddq_u32(vaddl_u16(vget_low_u16(v_row0), vget_high_u16(v_row0)),
                                             vaddl_u16(vget_low_u16(v_row1), vget_high_u16(v_row1)));
                vst1_u16(D, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const short * S, short * D, int w) const
    {
        int dx = 0;
        const short * S0 = S, * S1 = (const short *)((const uchar *)(S0) + step);

        int32x4_t v_2 = vdupq_n_s32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                int16x8x2_t v_row0 = vld2q_s16(S0), v_row1 = vld2q_s16(S1);

                int32x4_t v_dst0 = vaddl_s16(vget_low_s16(v_row0.val[0]), vget_low_s16(v_row0.val[1]));
                v_dst0 = vaddq_s32(v_dst0, vaddl_s16(vget_low_s16(v_row1.val[0]), vget_low_s16(v_row1.val[1])));
                v_dst0 = vshrq_n_s32(vaddq_s32(v_dst0, v_2), 2);

                int32x4_t v_dst1 = vaddl_s16(vget_high_s16(v_row0.val[0]), vget_high_s16(v_row0.val[1]));
                v_dst1 = vaddq_s32(v_dst1, vaddl_s16(vget_high_s16(v_row1.val[0]), vget_high_s16(v_row1.val[1])));
                v_dst1 = vshrq_n_s32(vaddq_s32(v_dst1, v_2), 2);

                vst1q_s16(D, vcombine_s16(vmovn_s32(v_dst0), vmovn_s32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                int16x8_t v_row0 = vld1q_s16(S0), v_row1 = vld1q_s16(S1);
                int32x4_t v_dst = vaddq_s32(vaddl_s16(vget_low_s16(v_row0), vget_high_s16(v_row0)),
                                            vaddl_s16(vget_low_s16(v_row1), vget_high_s16(v_row1)));
                vst1_s16(D, vmovn_s32(vshrq_n_s32(vaddq_s32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        float32x4_t v_025 = vdupq_n_f32(0.25f);

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4x2_t v_row0 = vld2q_f32(S0), v_row1 = vld2q_f32(S1);

                float32x4_t v_dst0 = vaddq_f32(v_row0.val[0], v_row0.val[1]);
                float32x4_t v_dst1 = vaddq_f32(v_row1.val[0], v_row1.val[1]);

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4_t v_dst0 = vaddq_f32(vld1q_f32(S0), vld1q_f32(S0 + 4));
                float32x4_t v_dst1 = vaddq_f32(vld1q_f32(S1), vld1q_f32(S1 + 4));

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#elif CV_SIMD

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step) {}

    int operator() (const uchar* S, uchar* D, int w) const
    {
        int dx = 0;
        const uchar* S0 = S;
        const uchar* S1 = S0 + step;

        if (cn == 1)
        {
            v_uint16 masklow = vx_setall_u16(0x00ff);
            for ( ; dx <= w - VTraits<v_uint16>::vlanes(); dx += VTraits<v_uint16>::vlanes(), S0 += VTraits<v_uint8>::vlanes(), S1 += VTraits<v_uint8>::vlanes(), D += VTraits<v_uint16>::vlanes())
            {
                v_uint16 r0 = v_reinterpret_as_u16(vx_load(S0));
                v_uint16 r1 = v_reinterpret_as_u16(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<8>(r0), v_and(r0, masklow)), v_shr<8>(r1)), v_and(r1, masklow)));
            }
        }
        else if (cn == 3)
        {
            if (CV_SIMD_WIDTH > 64)
                return 0;
            for ( ; dx <= w - 3*VTraits<v_uint8>::vlanes(); dx += 3*VTraits<v_uint8>::vlanes(), S0 += 6*VTraits<v_uint8>::vlanes(), S1 += 6*VTraits<v_uint8>::vlanes(), D += 3*VTraits<v_uint8>::vlanes())
            {
                v_uint16 t0, t1, t2, t3, t4, t5;
                v_uint16 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + VTraits<v_uint16>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_uint16>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_uint16>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_uint16>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_uint16>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_uint16 bl, gl, rl;
#if CV_SIMD_WIDTH == 16
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#elif CV_SIMD_WIDTH == 32
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#elif CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_uint16>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_uint16>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_uint16>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_uint16>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_uint16>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_uint16>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_uint16 bh, gh, rh;
#if CV_SIMD_WIDTH == 16
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#elif CV_SIMD_WIDTH == 32
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#elif CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
        }
        else
        {
            CV_Assert(cn == 4);
            for ( ; dx <= w - VTraits<v_uint8>::vlanes(); dx += VTraits<v_uint8>::vlanes(), S0 += 2*VTraits<v_uint8>::vlanes(), S1 += 2*VTraits<v_uint8>::vlanes(), D += VTraits<v_uint8>::vlanes())
            {
                v_uint32 r00, r01, r10, r11;
                v_load_deinterleave((uint32_t*)S0, r00, r01);
                v_load_deinterleave((uint32_t*)S1, r10, r11);

                v_uint16 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_u8(r00), r00l, r00h);
                v_expand(v_reinterpret_as_u8(r01), r01l, r01h);
                v_expand(v_reinterpret_as_u8(r10), r10l, r10h);
                v_expand(v_reinterpret_as_u8(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(v_add(v_add(r00l, r01l), r10l), r11l), v_add(v_add(v_add(r00h, r01h), r10h), r11h)));
            }
        }

        return dx;
    }

private:
    int cn;
    int step;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step) {}

    int operator() (const ushort* S, ushort* D, int w) const
    {
        int dx = 0;
        const ushort* S0 = (const ushort*)S;
        const ushort* S1 = (const ushort*)((const uchar*)(S) + step);

        if (cn == 1)
        {
            v_uint32 masklow = vx_setall_u32(0x0000ffff);
            for (; dx <= w - VTraits<v_uint32>::vlanes(); dx += VTraits<v_uint32>::vlanes(), S0 += VTraits<v_uint16>::vlanes(), S1 += VTraits<v_uint16>::vlanes(), D += VTraits<v_uint32>::vlanes())
            {
                v_uint32 r0 = v_reinterpret_as_u32(vx_load(S0));
                v_uint32 r1 = v_reinterpret_as_u32(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<16>(r0), v_and(r0, masklow)), v_shr<16>(r1)), v_and(r1, masklow)));
            }
        }
        else if (cn == 3)
        {
#if CV_SIMD_WIDTH == 16
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
#if CV_SSE4_1
            {
                v_uint32 r0, r1, r2, r3;
                v_expand(vx_load(S0), r0, r1);
                v_expand(vx_load(S1), r2, r3);
                r0 = v_add(r0, r2); r1 = v_add(r1, r3);
                v_rshr_pack_store<2>(D, v_add(r0, v_rotate_left<1>(r1, r0)));
            }
#else
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_load_expand(S0), v_load_expand(S0 + 3)), v_load_expand(S1)), v_load_expand(S1 + 3)));
#endif
#elif CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
            for ( ; dx <= w - 3*VTraits<v_uint16>::vlanes(); dx += 3*VTraits<v_uint16>::vlanes(), S0 += 6*VTraits<v_uint16>::vlanes(), S1 += 6*VTraits<v_uint16>::vlanes(), D += 3*VTraits<v_uint16>::vlanes())
            {
                v_uint32 t0, t1, t2, t3, t4, t5;
                v_uint32 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + VTraits<v_uint32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_uint32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_uint32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_uint32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_uint32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_uint32 bl, gl, rl;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_uint32>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_uint32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_uint32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_uint32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_uint32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_uint32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_uint32 bh, gh, rh;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#elif CV_SIMD_WIDTH >= 64
            v_uint32 masklow = vx_setall_u32(0x0000ffff);
            for ( ; dx <= w - 3*VTraits<v_uint16>::vlanes(); dx += 3*VTraits<v_uint16>::vlanes(), S0 += 6*VTraits<v_uint16>::vlanes(), S1 += 6*VTraits<v_uint16>::vlanes(), D += 3*VTraits<v_uint16>::vlanes())
            {
                v_uint16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_uint32 bl = (v_reinterpret_as_u32(b0) >> 16) + (v_reinterpret_as_u32(b0) & masklow) + (v_reinterpret_as_u32(b1) >> 16) + (v_reinterpret_as_u32(b1) & masklow);
                v_uint32 gl = (v_reinterpret_as_u32(g0) >> 16) + (v_reinterpret_as_u32(g0) & masklow) + (v_reinterpret_as_u32(g1) >> 16) + (v_reinterpret_as_u32(g1) & masklow);
                v_uint32 rl = (v_reinterpret_as_u32(r0) >> 16) + (v_reinterpret_as_u32(r0) & masklow) + (v_reinterpret_as_u32(r1) >> 16) + (v_reinterpret_as_u32(r1) & masklow);
                v_load_deinterleave(S0 + 3*VTraits<v_uint16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_uint16>::vlanes(), b1, g1, r1);
                v_uint32 bh = (v_reinterpret_as_u32(b0) >> 16) + (v_reinterpret_as_u32(b0) & masklow) + (v_reinterpret_as_u32(b1) >> 16) + (v_reinterpret_as_u32(b1) & masklow);
                v_uint32 gh = (v_reinterpret_as_u32(g0) >> 16) + (v_reinterpret_as_u32(g0) & masklow) + (v_reinterpret_as_u32(g1) >> 16) + (v_reinterpret_as_u32(g1) & masklow);
                v_uint32 rh = (v_reinterpret_as_u32(r0) >> 16) + (v_reinterpret_as_u32(r0) & masklow) + (v_reinterpret_as_u32(r1) >> 16) + (v_reinterpret_as_u32(r1) & masklow);
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#endif
        }
        else
        {
            CV_Assert(cn == 4);
#if CV_SIMD_WIDTH >= 64
            for ( ; dx <= w - VTraits<v_uint16>::vlanes(); dx += VTraits<v_uint16>::vlanes(), S0 += 2*VTraits<v_uint16>::vlanes(), S1 += 2*VTraits<v_uint16>::vlanes(), D += VTraits<v_uint16>::vlanes())
            {
                v_uint64 r00, r01, r10, r11;
                v_load_deinterleave((uint64_t*)S0, r00, r01);
                v_load_deinterleave((uint64_t*)S1, r10, r11);

                v_uint32 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_u16(r00), r00l, r00h);
                v_expand(v_reinterpret_as_u16(r01), r01l, r01h);
                v_expand(v_reinterpret_as_u16(r10), r10l, r10h);
                v_expand(v_reinterpret_as_u16(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(r00l, r01l, r10l, r11l), v_add(r00h, r01h, r10h, r11h)));
            }
#else
            for ( ; dx <= w - VTraits<v_uint32>::vlanes(); dx += VTraits<v_uint32>::vlanes(), S0 += VTraits<v_uint16>::vlanes(), S1 += VTraits<v_uint16>::vlanes(), D += VTraits<v_uint32>::vlanes())
            {
                v_uint32 r0, r1, r2, r3;
                v_expand(vx_load(S0), r0, r1);
                v_expand(vx_load(S1), r2, r3);
                r0 = v_add(r0, r2); r1 = v_add(r1, r3);
                v_uint32 v_d;
#if CV_SIMD_WIDTH == 16
                v_d = v_add(r0, r1);
#elif CV_SIMD_WIDTH == 32
                v_uint32 t0, t1;
                v_recombine(r0, r1, t0, t1);
                v_d = v_add(t0, t1);
#endif
                v_rshr_pack_store<2>(D, v_d);
            }
#endif
        }

        return dx;
    }

private:
    int cn;
    int step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) :
        cn(_cn), step(_step) {}

    int operator() (const short* S, short* D, int w) const
    {
        int dx = 0;
        const short* S0 = (const short*)S;
        const short* S1 = (const short*)((const uchar*)(S) + step);

        if (cn == 1)
        {
            v_int32 masklow = vx_setall_s32(0x0000ffff);
            for (; dx <= w - VTraits<v_int32>::vlanes(); dx += VTraits<v_int32>::vlanes(), S0 += VTraits<v_int16>::vlanes(), S1 += VTraits<v_int16>::vlanes(), D += VTraits<v_int32>::vlanes())
            {
                v_int32 r0 = v_reinterpret_as_s32(vx_load(S0));
                v_int32 r1 = v_reinterpret_as_s32(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<16>(r0), v_shr<16>(v_shl<16>(v_and(r0, masklow)))), v_shr<16>(r1)), v_shr<16>(v_shl<16>(v_and(r1, masklow)))));
            }
        }
        else if (cn == 3)
        {
#if CV_SIMD_WIDTH == 16
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_load_expand(S0), v_load_expand(S0 + 3)), v_load_expand(S1)), v_load_expand(S1 + 3)));
#elif CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
            for ( ; dx <= w - 3*VTraits<v_int16>::vlanes(); dx += 3*VTraits<v_int16>::vlanes(), S0 += 6*VTraits<v_int16>::vlanes(), S1 += 6*VTraits<v_int16>::vlanes(), D += 3*VTraits<v_int16>::vlanes())
            {
                v_int32 t0, t1, t2, t3, t4, t5;
                v_int32 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_int32>::vlanes()), vx_load_expand(S1 + VTraits<v_int32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_int32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_int32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_int32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_int32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_int32 bl, gl, rl;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_int32>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_int32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_int32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_int32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_int32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_int32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_int32 bh, gh, rh;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#elif CV_SIMD_WIDTH >= 64
            for ( ; dx <= w - 3*VTraits<v_int16>::vlanes(); dx += 3*VTraits<v_int16>::vlanes(), S0 += 6*VTraits<v_int16>::vlanes(), S1 += 6*VTraits<v_int16>::vlanes(), D += 3*VTraits<v_int16>::vlanes())
            {
                v_int16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_int32 bl = (v_reinterpret_as_s32(b0) >> 16) + ((v_reinterpret_as_s32(b0) << 16) >> 16) + (v_reinterpret_as_s32(b1) >> 16) + ((v_reinterpret_as_s32(b1) << 16) >> 16);
                v_int32 gl = (v_reinterpret_as_s32(g0) >> 16) + ((v_reinterpret_as_s32(g0) << 16) >> 16) + (v_reinterpret_as_s32(g1) >> 16) + ((v_reinterpret_as_s32(g1) << 16) >> 16);
                v_int32 rl = (v_reinterpret_as_s32(r0) >> 16) + ((v_reinterpret_as_s32(r0) << 16) >> 16) + (v_reinterpret_as_s32(r1) >> 16) + ((v_reinterpret_as_s32(r1) << 16) >> 16);
                v_load_deinterleave(S0 + 3*VTraits<v_int16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_int16>::vlanes(), b1, g1, r1);
                v_int32 bh = (v_reinterpret_as_s32(b0) >> 16) + ((v_reinterpret_as_s32(b0) << 16) >> 16) + (v_reinterpret_as_s32(b1) >> 16) + ((v_reinterpret_as_s32(b1) << 16) >> 16);
                v_int32 gh = (v_reinterpret_as_s32(g0) >> 16) + ((v_reinterpret_as_s32(g0) << 16) >> 16) + (v_reinterpret_as_s32(g1) >> 16) + ((v_reinterpret_as_s32(g1) << 16) >> 16);
                v_int32 rh = (v_reinterpret_as_s32(r0) >> 16) + ((v_reinterpret_as_s32(r0) << 16) >> 16) + (v_reinterpret_as_s32(r1) >> 16) + ((v_reinterpret_as_s32(r1) << 16) >> 16);
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#endif
        }
        else
        {
            CV_Assert(cn == 4);
            for (; dx <= w - VTraits<v_int16>::vlanes(); dx += VTraits<v_int16>::vlanes(), S0 += 2 * VTraits<v_int16>::vlanes(), S1 += 2 * VTraits<v_int16>::vlanes(), D += VTraits<v_int16>::vlanes())
            {
#if CV_SIMD_WIDTH >= 64
                v_int64 r00, r01, r10, r11;
                v_load_deinterleave((int64_t*)S0, r00, r01);
                v_load_deinterleave((int64_t*)S1, r10, r11);

                v_int32 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_s16(r00), r00l, r00h);
                v_expand(v_reinterpret_as_s16(r01), r01l, r01h);
                v_expand(v_reinterpret_as_s16(r10), r10l, r10h);
                v_expand(v_reinterpret_as_s16(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(r00l, r01l, r10l, r11l), v_add(r00h, r01h, r10h, r11h)));
#else
                v_int32 r0, r1, r2, r3;
                r0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                r1 = v_add(vx_load_expand(S0 + VTraits<v_int32>::vlanes()), vx_load_expand(S1 + VTraits<v_int32>::vlanes()));
                r2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_int32>::vlanes()));
                r3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_int32>::vlanes()));
                v_int32 dl, dh;
#if CV_SIMD_WIDTH == 16
                dl = v_add(r0, r1); dh = v_add(r2, r3);
#elif CV_SIMD_WIDTH == 32
                v_int32 t0, t1, t2, t3;
                v_recombine(r0, r1, t0, t1); v_recombine(r2, r3, t2, t3);
                dl = v_add(t0, t1); dh = v_add(t2, t3);
#endif
                v_store(D, v_rshr_pack<2>(dl, dh));
#endif
            }
        }

        return dx;
    }

private:
    int cn;
    int step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        if (cn == 1)
        {
            v_float32 v_025 = vx_setall_f32(0.25f);
            for ( ; dx <= w - VTraits<v_float32>::vlanes(); dx += VTraits<v_float32>::vlanes(), S0 += 2*VTraits<v_float32>::vlanes(), S1 += 2*VTraits<v_float32>::vlanes(), D += VTraits<v_float32>::vlanes())
            {
                v_float32 v_row00, v_row01, v_row10, v_row11;
                v_load_deinterleave(S0, v_row00, v_row01);
                v_load_deinterleave(S1, v_row10, v_row11);
                v_store(D, v_mul(v_add(v_add(v_row00, v_row01), v_add(v_row10, v_row11)), v_025));
            }
        }
        else if (cn == 4)
        {
#if CV_SIMD_WIDTH == 16
            v_float32 v_025 = vx_setall_f32(0.25f);
            for (; dx <= w - VTraits<v_float32>::vlanes(); dx += VTraits<v_float32>::vlanes(), S0 += 2*VTraits<v_float32>::vlanes(), S1 += 2*VTraits<v_float32>::vlanes(), D += VTraits<v_float32>::vlanes())
                v_store(D, v_mul(v_add(v_add(vx_load(S0), vx_load(S0 + VTraits<v_float32>::vlanes())), v_add(vx_load(S1), vx_load(S1 + VTraits<v_float32>::vlanes()))), v_025));
#elif CV_SIMD256
            v_float32x8 v_025 = v256_setall_f32(0.25f);
            for (; dx <= w - VTraits<v_float32x8>::vlanes(); dx += VTraits<v_float32x8>::vlanes(), S0 += 2*VTraits<v_float32x8>::vlanes(), S1 += 2*VTraits<v_float32x8>::vlanes(), D += VTraits<v_float32x8>::vlanes())
            {
                v_float32x8 dst0, dst1;
                v_recombine(v_add(v256_load(S0), v256_load(S1)), v_add(v256_load(S0 + VTraits<v_float32x8>::vlanes()), v256_load(S1 + VTraits<v_float32x8>::vlanes())), dst0, dst1);
                v_store(D, v_mul(v_add(dst0, dst1), v_025));
            }
#endif
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#else

typedef ResizeAreaFastNoVec<uchar, uchar> ResizeAreaFastVec_SIMD_8u;
typedef ResizeAreaFastNoVec<ushort, ushort> ResizeAreaFastVec_SIMD_16u;
typedef ResizeAreaFastNoVec<short, short> ResizeAreaFastVec_SIMD_16s;
typedef ResizeAreaFastNoVec<float, float> ResizeAreaFastVec_SIMD_32f;

#endif

template<typename T, typename SIMDVecOp>
struct ResizeAreaFastVec
{
    ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step), vecOp(_cn, _step)
    {
        fast_mode = scale_x == 2 && scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator() (const T* S, T* D, int w) const
    {
        if (!fast_mode)
            return 0;

        const T* nextS = (const T*)((const uchar*)S + step);
        int dx = vecOp(S, D, w);

        if (cn == 1)
            for( ; dx < w; ++dx )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+1] + nextS[index] + nextS[index+1] + 2) >> 2);
            }
        else if (cn == 3)
            for( ; dx < w; dx += 3 )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+3] + nextS[index] + nextS[index+3] + 2) >> 2);
                D[dx+1] = (T)((S[index+1] + S[index+4] + nextS[index+1] + nextS[index+4] + 2) >> 2);
                D[dx+2] = (T)((S[index+2] + S[index+5] + nextS[index+2] + nextS[index+5] + 2) >> 2);
            }
        else
            {
                CV_Assert(cn == 4);
                for( ; dx < w; dx += 4 )
                {
                    int index = dx*2;
                    D[dx] = (T)((S[index] + S[index+4] + nextS[index] + nextS[index+4] + 2) >> 2);
                    D[dx+1] = (T)((S[index+1] + S[index+5] + nextS[index+1] + nextS[index+5] + 2) >> 2);
                    D[dx+2] = (T)((S[index+2] + S[index+6] + nextS[index+2] + nextS[index+6] + 2) >> 2);
                    D[dx+3] = (T)((S[index+3] + S[index+7] + nextS[index+3] + nextS[index+7] + 2) >> 2);
                }
            }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    bool fast_mode;
    int step;
    SIMDVecOp vecOp;
};

template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker :
    public ParallelLoopBody
{
public:
    resizeAreaFast_Invoker(const Mat &_src, Mat &_dst,
        int _scale_x, int _scale_y, const int* _ofs, const int* _xofs) :
        ParallelLoopBody(), src(_src), dst(_dst), scale_x(_scale_x),
        scale_y(_scale_y), ofs(_ofs), xofs(_xofs)
    {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        int area = scale_x * scale_y;
        float scale = 1.f / area;
        int dwidth1 = ssize.width / scale_x * cn;
        dsize.width *= cn;
        ssize.width *= cn;
        int dy, dx, k = 0;

        VecOp vop(scale_x, scale_y, src.channels(), (int)src.step/*, area_ofs*/);

        for( dy = range.start; dy < range.end; dy++ )
        {
            T* D = (T*)(dst.data + dst.step*dy);
            int sy0 = dy*scale_y;
            int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

            if( sy0 >= ssize.height )
            {
                for( dx = 0; dx < dsize.width; dx++ )
                    D[dx] = 0;
                continue;
            }

            dx = vop(src.template ptr<T>(sy0), D, w);
            for( ; dx < w; dx++ )
            {
                const T* S = src.template ptr<T>(sy0) + xofs[dx];
                WT sum = 0;
                k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= area - 4; k += 4 )
                    sum += S[ofs[k]] + S[ofs[k+1]] + S[ofs[k+2]] + S[ofs[k+3]];
                #endif
                for( ; k < area; k++ )
                    sum += S[ofs[k]];

                D[dx] = saturate_cast<T>(sum * scale);
            }

            for( ; dx < dsize.width; dx++ )
            {
                WT sum = 0;
                int count = 0, sx0 = xofs[dx];
                if( sx0 >= ssize.width )
                    D[dx] = 0;

                for( int sy = 0; sy < scale_y; sy++ )
                {
                    if( sy0 + sy >= ssize.height )
                        break;
                    const T* S = src.template ptr<T>(sy0 + sy) + sx0;
                    for( int sx = 0; sx < scale_x*cn; sx += cn )
                    {
                        if( sx0 + sx >= ssize.width )
                            break;
                        sum += S[sx];
                        count++;
                    }
                }
                // sum maybe double, converting it to float will decrease precision
                // when count < 2^23, converting it to float is fine
                D[dx] = saturate_cast<T>(sum / static_cast<float>(count));
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int scale_x, scale_y;
    const int *ofs, *xofs;
};

template<typename T, typename WT, typename VecOp>
static void resizeAreaFast_( const Mat& src, Mat& dst, const int* ofs, const int* xofs,
                             int scale_x, int scale_y )
{
    Range range(0, dst.rows);
    resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x,
        scale_y, ofs, xofs);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

struct DecimateAlpha
{
    int si, di;
    float alpha;
};


namespace inter_area {
#if (CV_SIMD || CV_SIMD_SCALABLE)
inline void saturate_store(const float* src, uchar* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp2 = v_round(vx_load(src + 2 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp3 = v_round(vx_load(src + 3 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(v_pack_u(tmp0, tmp1), v_pack_u(tmp2, tmp3)));
}

inline void saturate_store(const float* src, schar* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp2 = v_round(vx_load(src + 2 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp3 = v_round(vx_load(src + 3 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(v_pack(tmp0, tmp1), v_pack(tmp2, tmp3)));
}

inline void saturate_store(const float* src, ushort* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack_u(tmp0, tmp1));
}

inline void saturate_store(const float* src, short* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(tmp0, tmp1));
}

static inline v_float32 vx_setall(float coeff) { return vx_setall_f32(coeff); }

template <typename T>
struct VArea {};

template <>
struct VArea<float> {
    typedef v_float32 vWT;
};
#endif

#if (CV_SIMD128_64F || CV_SIMD_SCALABLE_64F)
static inline v_float64 vx_setall(double coeff) { return vx_setall_f64(coeff); }

template <>
struct VArea<double> {
    typedef v_float64 vWT;
};

inline void saturate_store(const double* sum, int width, int* D) {
    const int step = VTraits<v_float64>::vlanes() * sizeof(double) / sizeof(int);
    int dx = 0, limit = width - step;
    for (; dx <= limit; dx += step)
    {
        v_store(D + dx, v_round(
            vx_load(sum + dx + 0 * VTraits<v_float64>::vlanes()),
            vx_load(sum + dx + 1 * VTraits<v_float64>::vlanes())));
    }
    for (; dx < width; ++dx)
        D[dx] = saturate_cast<int>(sum[dx]);
}
#else
inline void mul(const double* buf, int width, double beta, double* sum) {
    for (int dx = 0; dx < width; ++dx) {
        sum[dx] = beta * buf[dx];
    }
}

inline void muladd(const double* buf, int width, double beta, double* sum) {
    for (int dx = 0; dx < width; ++dx) {
        sum[dx] += beta * buf[dx];
    }
}

inline void saturate_store(const double* sum, int width, int* D) {
    for (int dx = 0; dx < width; ++dx)
        D[dx] = saturate_cast<int>(sum[dx]);
}
#endif

template <typename T, typename WT>
inline void saturate_store(const WT* sum, int width, T* D) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes() * sizeof(WT) / sizeof(T);
    for (; dx + step < width; dx += step) {
        saturate_store(sum + dx, D + dx);
    }
#endif
    for (; dx < width; ++dx) {
        D[dx] = saturate_cast<T>(sum[dx]);
    }
}

// Optimization when T == WT.
template <typename WT>
inline void saturate_store(const WT* sum, int width, WT* D) {
    std::copy(sum, sum + width, D);
}

template <typename WT>
inline void mul(const WT* buf, int width, WT beta, WT* sum) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes();
    const typename VArea<WT>::vWT vbeta = vx_setall(beta);
    int limit = width - step;
    for (; dx <= limit; dx += step) {
        vx_store(sum + dx, v_mul(vbeta, vx_load(buf + dx)));
    }
#endif
    for (; dx < width; ++dx) {
        sum[dx] = beta * buf[dx];
    }
}

template <typename WT>
inline void muladd(const WT* buf, int width, WT beta, WT* sum) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes();
    const typename VArea<WT>::vWT vbeta = vx_setall(beta);
    int limit = width - step;
    for (; dx <= limit; dx += step) {
        vx_store(sum + dx, v_add(vx_load(sum + dx), v_mul(vbeta, vx_load(buf + dx))));
    }
#endif
    for (; dx < width; ++dx) {
        sum[dx] += beta * buf[dx];
    }
}

}  // namespace inter_area

template<typename T, typename WT> class ResizeArea_Invoker :
    public ParallelLoopBody
{
public:
    ResizeArea_Invoker( const Mat& _src, Mat& _dst,
                        const DecimateAlpha* _xtab, int _xtab_size,
                        const DecimateAlpha* _ytab, int _ytab_size,
                        const int* _tabofs )
    {
        src = &_src;
        dst = &_dst;
        xtab0 = _xtab;
        xtab_size0 = _xtab_size;
        ytab = _ytab;
        ytab_size = _ytab_size;
        tabofs = _tabofs;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst->size();
        int cn = dst->channels();
        dsize.width *= cn;
        AutoBuffer<WT> _buffer(dsize.width*2);
        const DecimateAlpha* xtab = xtab0;
        int xtab_size = xtab_size0;
        WT *buf = _buffer.data(), *sum = buf + dsize.width;
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, k, dx, prev_dy = ytab[j_start].di;

        for( dx = 0; dx < dsize.width; dx++ )
            sum[dx] = (WT)0;

        for( j = j_start; j < j_end; j++ )
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = src->template ptr<T>(sy);
                for( dx = 0; dx < dsize.width; dx++ )
                    buf[dx] = (WT)0;

                if( cn == 1 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        buf[dxn] += S[xtab[k].si]*alpha;
                    }
                else if( cn == 2 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                    }
                else if( cn == 3 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        WT t2 = buf[dxn+2] + S[sxn+2]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1; buf[dxn+2] = t2;
                    }
                else if( cn == 4 )
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                        t0 = buf[dxn+2] + S[sxn+2]*alpha;
                        t1 = buf[dxn+3] + S[sxn+3]*alpha;
                        buf[dxn+2] = t0; buf[dxn+3] = t1;
                    }
                }
                else
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        for( int c = 0; c < cn; c++ )
                            buf[dxn + c] += S[sxn + c]*alpha;
                    }
                }
            }

            if( dy != prev_dy )
            {
                inter_area::saturate_store(sum, dsize.width, dst->template ptr<T>(prev_dy));
                inter_area::mul(buf, dsize.width, beta, sum);
                prev_dy = dy;
            }
            else
            {
                inter_area::muladd(buf, dsize.width, beta, sum);
            }
        }

        inter_area::saturate_store(sum, dsize.width, dst->template ptr<T>(prev_dy));
    }

private:
    const Mat* src;
    Mat* dst;
    const DecimateAlpha* xtab0;
    const DecimateAlpha* ytab;
    int xtab_size0, ytab_size;
    const int* tabofs;
};


template <typename T, typename WT>
static void resizeArea_( const Mat& src, Mat& dst,
                         const DecimateAlpha* xtab, int xtab_size,
                         const DecimateAlpha* ytab, int ytab_size,
                         const int* tabofs )
{
    parallel_for_(Range(0, dst.rows),
                 ResizeArea_Invoker<T, WT>(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs),
                 dst.total()/((double)(1 << 16)));
}


class ResizeOnnxCtrl
{
   utils::BufferArea area;

public:
    struct TabIdx
    {
        int si, di; // index on src / dst by elem1
        union { float f; double d; }; // coefficient / weight

        void as(float&  v) { v = f; }
        void as(double& v) { v = d; }
    };

    /* resize parameter */
    bool is_fixpt, is_double;
    int sampler, antialias;
    /* only meaningful when do bi-cubic or antialias resampling.
        For nearest neighbor, it will have no pixel to select.
        For linear without antialias,
            the two sample pixels are at least one inside and at most one outside.
            So exclude_outside is simply equivalent to clamp.
    */
    int exclude_outside;
    int ksize, xkanti, ykanti;
    Point2f scalef;

    /* for antialias resize */
    TabIdx* xtab;
    TabIdx* ytab;
    /* for generic resize */
    int* xofs;
    int* yofs;
    double* xcoeffs;
    double* ycoeffs;
    int xmin, xmax;

private:
    void cubicCoeffsAntiAlias(
        int dstlen, int cn, float srcpos, float scale, int srclen, float A, TabIdx* elem)
    {
        scale = min(scale, 1.f);
        int index = cvFloor(srcpos);
        float ratio = srcpos - index;
        int start = cvFloor(-2.f / scale) + 1;
        int end = 2 - start;
        int len = end - start;
        // no need to add FLT_EPSILON.
        // in antialias cubic resize, we will have at least ceil(2 / scale) pixels inside
        float sum = 0;
        for (int i = start; i < end; ++i)
        {
            float x = fabsf(i - ratio) * scale;
            if (x <= 1)
                x = ((A + 2) * x - (A + 3)) * x * x + 1;
            else if (x <= 2)
                x = A * (((x - 5) * x + 8) * x - 4);
            else
                x = 0;
            int sx = index + i;
            if (exclude_outside && static_cast<unsigned>(sx) >= static_cast<unsigned>(srclen))
                x = 0;
            elem[i - start].di = cn * dstlen;
            elem[i - start].si = cn * min(max(sx, 0), srclen - 1);
            elem[i - start].f = x;
            sum += x;
        }
        for (int i = 0; i < len; ++i)
        {
            if (is_double)
                elem[i].d = elem[i].f / sum;
            else
                elem[i].f = elem[i].f / sum;
        }
    }

    void cubicCoeffs(float x, float A, float* coeffs)
    {
        coeffs[0] = A * ((((x + 1) - 5) * (x + 1) + 8) * (x + 1) - 4);
        coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
        coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
        coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
    }

    void linearCoeffsAntialias(
        int dstlen, int cn, float srcpos, float scale, int srclen, TabIdx* elem)
    {
        scale = min(scale, 1.f);
        int index = cvFloor(srcpos);
        float ratio = srcpos - index;
        int start = cvFloor(-1.f / scale) + 1;
        int end = 2 - start;
        int len = end - start;
        float sum = 0.f;
        for (int i = start; i < end; ++i)
        {
            float x = fabsf(i - ratio) * scale;
            x = min(max(1.f - x, 0.f), 1.f);
            int sx = index + i;
            if (exclude_outside && static_cast<unsigned>(sx) >= static_cast<unsigned>(srclen))
                x = 0;
            elem[i - start].di = cn * dstlen;
            elem[i - start].si = cn * min(max(sx, 0), srclen - 1);
            elem[i - start].f = x;
            sum += x;
        }
        for (int i = 0; i < len; ++i)
        {
            if (is_double)
                elem[i].d = elem[i].f / sum;
            else
                elem[i].f = elem[i].f / sum;
        }
    }

    void linearCoeffs(float x, float* coeffs)
    {
        coeffs[0] = 1.f - x;
        coeffs[1] = x;
    }

   public:
    ResizeOnnxCtrl(int interpolation, int type, float cubicCoeff,
        Size ssize, Size dsize, Point2d const& scaled, Matx22f const& M)
    {
        sampler = interpolation & INTER_SAMPLER_MASK;
        antialias = interpolation & INTER_ANTIALIAS_MASK;
        exclude_outside = interpolation & INTER_EXCLUDE_OUTSIDE_MASK;
        CV_CheckGE(cubicCoeff, -1.f, "cubic coefficient should range [-1, 0)");
        CV_CheckLT(cubicCoeff, +0.f, "cubic coefficient should range [-1, 0)");
        CV_Check(sampler, sampler == INTER_LINEAR || sampler == INTER_CUBIC,
            "should not error");

        int cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
        scalef = static_cast<Point2f>(scaled);
        ksize = (sampler == INTER_LINEAR ? 2 : 4);
        is_double = (depth == CV_64F);
        is_fixpt = (depth == CV_8U || depth == CV_8S);
        is_double = (depth == CV_32S || depth == CV_64F);
        xtab = ytab = nullptr;
        xofs = yofs = nullptr;
        xcoeffs = ycoeffs = nullptr;
        int khalf = ksize / 2;
        xkanti = 2 * cvCeil(khalf / min(scalef.x, 1.f));
        ykanti = 2 * cvCeil(khalf / min(scalef.y, 1.f));
        area.allocate(xtab, xkanti * dsize.width );
        area.allocate(ytab, ykanti * dsize.height);
        area.allocate(xofs, dsize.width * cn + 1);
        area.allocate(yofs, dsize.height * 1 + 1);
        area.allocate(xcoeffs, ksize * dsize.width * cn);
        area.allocate(ycoeffs, ksize * dsize.height * 1);
        area.commit();
        float cbuf[MAX_ESIZE] = { 0 };
        CV_CheckLE(ksize, MAX_ESIZE, "resampler kernel's size is too larger");

        // when upsampling, `antialias` is same as `generic`, use `generic` to speed up
        if (antialias && scaled.x < 1.0)
        {
            float a = M(0, 0), b = M(0, 1);
            for (int d = 0; d < dsize.width; ++d)
            {
                float f = fmaf(static_cast<float>(d), a, b);
                if (sampler == INTER_LINEAR)
                    linearCoeffsAntialias(d, cn, f, scalef.x, ssize.width, xtab + d * xkanti);
                else // if (sampler == INTER_CUBIC)
                    cubicCoeffsAntiAlias(d, cn, f, scalef.x, ssize.width, cubicCoeff, xtab + d * xkanti);
            }
        }
        else
        {
            xkanti = 0;
            xmin = 0;
            xmax = dsize.width;
            float a = M(0, 0), b = M(0, 1);
            for (int d = 0; d < dsize.width; ++d)
            {
                float f = fmaf(static_cast<float>(d), a, b);
                int s = cvFloor(f);
                f -= s;
                if (s < khalf - 1) {
                    xmin = d + 1;
                    if (s < 0 && sampler == INTER_LINEAR)
                        f = 0, s = 0;
                }
                if (s + khalf >= ssize.width)
                {
                    xmax = min(xmax, d);
                    if (s >= ssize.width - 1 && sampler == INTER_LINEAR)
                        f = 0, s = ssize.width - 1;
                }
                for (int k = 0; k < cn; ++k)
                    xofs[cn * d + k] = cn * s + k;
                if (sampler == INTER_LINEAR)
                    linearCoeffs(f, cbuf);
                else // if (sampler == INTER_CUBIC)
                {
                    cubicCoeffs(f, cubicCoeff, cbuf);
                    if (exclude_outside && (s < 1 || s + 2 >= ssize.width))
                    {
                        // no need to add FLT_EPSILON.
                        // in cubic without antialias, we will have at least 2 pixels inside
                        float sum = 0;
                        for (int k = 0; k < 4; ++k)
                        {
                            if (static_cast<unsigned>(s + k - 1) >= static_cast<unsigned>(ssize.width))
                                cbuf[k] = 0;
                            sum += cbuf[k];
                        }
                        for (int k = 0; k < 4; ++k)
                            cbuf[k] /= sum;
                    }
                }
                if (is_fixpt)
                {
                    short* coeffs = reinterpret_cast<short*>(xcoeffs) + cn * ksize * d;
                    for (int k = 0; k < ksize; ++k)
                        coeffs[k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
                    for (int k = ksize; k < cn * ksize; ++k)
                        coeffs[k] = coeffs[k - ksize];
                }
                else if (is_double)
                {
                    double* coeffs = xcoeffs + cn * ksize * d;
                    for (int k = 0; k < ksize; ++k)
                        coeffs[k] = cbuf[k];
                    for (int k = ksize; k < cn * ksize; ++k)
                        coeffs[k] = coeffs[k - ksize];
                }
                else
                {
                    float* coeffs = reinterpret_cast<float*>(xcoeffs) + cn * ksize * d;
                    for (int k = 0; k < ksize; ++k)
                        coeffs[k] = cbuf[k];
                    for (int k = ksize; k < cn * ksize; ++k)
                        coeffs[k] = coeffs[k - ksize];
                }
            }
        }

        if (antialias && scaled.y < 1.0)
        {
            float a = M(1, 0), b = M(1, 1);
            for (int d = 0; d < dsize.height; ++d)
            {
                float f = fmaf(static_cast<float>(d), a, b);
                if (sampler == INTER_LINEAR)
                    linearCoeffsAntialias(d, 1, f, scalef.y, ssize.height, ytab + d * ykanti);
                else // if (sampler == INTER_CUBIC)
                    cubicCoeffsAntiAlias(d, 1, f, scalef.y, ssize.height, cubicCoeff, ytab + d * ykanti);
            }
        }
        else
        {
            ykanti = 0;
            float a = M(1, 0), b = M(1, 1);
            for (int d = 0; d < dsize.height; ++d)
            {
                float f = fmaf(static_cast<float>(d), a, b);
                int s = cvFloor(f);
                f -= s;
                yofs[d] = s;
                if (sampler == INTER_LINEAR)
                    linearCoeffs(f, cbuf);
                else // if (sampler == INTER_CUBIC)
                {
                    cubicCoeffs(f, cubicCoeff, cbuf);
                    if (exclude_outside && (s < 1 || s + 2 >= ssize.height))
                    {
                        float sum = 0;
                        for (int k = 0; k < 4; ++k)
                        {
                            if (static_cast<unsigned>(s + k - 1) >= static_cast<unsigned>(ssize.height))
                                cbuf[k] = 0;
                            sum += cbuf[k];
                        }
                        for (int k = 0; k < 4; ++k)
                            cbuf[k] /= sum;
                    }
                }
                if (is_fixpt)
                {
                    short* coeffs = reinterpret_cast<short*>(ycoeffs) + 1 * ksize * d;
                    for (int k = 0; k < ksize; ++k)
                        coeffs[k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
                }
                else if (is_double)
                {
                    double* coeffs = ycoeffs + 1 * ksize * d;
                    for (int k = 0; k < ksize; ++k)
                        coeffs[k] = cbuf[k];
                }
                else
                {
                    float* coeffs = reinterpret_cast<float*>(ycoeffs) + 1 * ksize * d;
                    for (int k = 0; k < ksize; ++k)
                        coeffs[k] = cbuf[k];
                }
            }
        }
    }
};

template <typename HResize, typename VResize, typename IdxT>
class ResizeOnnxInvoker : public ParallelLoopBody
{
    Mat const& src;
    Mat& dst;
    ResizeOnnxCtrl const& ctrl;
    HResize hresize;
    VResize vresize;

    ResizeOnnxInvoker& operator =(ResizeOnnxInvoker const&);

public:
    typedef typename HResize::value_type T;
    typedef typename HResize::buf_type WT;
    typedef typename HResize::alpha_type AT;

    ResizeOnnxInvoker(const Mat& _src, Mat& _dst, ResizeOnnxCtrl const& _ctrl) :
        src(_src), dst(_dst), ctrl(_ctrl)
    {
        static_assert(sizeof(WT) == sizeof(IdxT), "expected");
        static_assert(std::is_same<IdxT, typename std::common_type<IdxT, WT>::type>::value,
            "IdxT double : WT double | IdxT float : WT float / int");

        CV_CheckLE(ctrl.ksize, MAX_ESIZE, "resampler kernel's size is too larger");
        CV_Check(ctrl.is_fixpt, !(ctrl.is_fixpt && ctrl.is_double), "can not be both types");
        // prefer static_assert, but how ?
        // check generic resize
        if (ctrl.is_fixpt)
        {
            CV_Check(ctrl.is_fixpt, (std::is_same<AT, short>::value),
                "when use fixpt / short coeffs, AT is expected to be short");
            CV_Check(sizeof(T) * 10 + sizeof(WT),
                (std::is_same<WT, int>::value
                    && (std::is_same<T, uchar>::value || std::is_same<T, schar>::value)),
                "fixpt works when T is uchar or schar");
        }
        else if (ctrl.is_double)
        {
            CV_Check(ctrl.is_double, (std::is_same<AT, double>::value),
                "when use double coeffs, AT is expected to be double");
            CV_Check(sizeof(T) * 10 + sizeof(WT),
                (std::is_same<WT, double>::value &&
                    (std::is_same<T, int>::value || std::is_same<T, double>::value)),
                "double WT works when T is int or double");
         }
        else
        {
            CV_Check(sizeof(AT), (std::is_same<AT, float>::value),
                "when use float coeffs, AT is expected to be float");
            CV_Check(sizeof(T) * 10 + sizeof(WT),
                (std::is_same<WT, float>::value
                    && (std::is_same<T, short>::value || std::is_same<T, ushort>::value
                        || std::is_same<T, float>::value)),
                "float WT works for other types");
        }
        // check antialias resize
        if (ctrl.is_double)
        {
            CV_Check(ctrl.is_double, (std::is_same<IdxT, double>::value),
                "when use double coeffs, IdxT is expected to be double");
        }
        else
        {
            CV_Check(ctrl.is_double, (std::is_same<IdxT, float>::value),
                "when use float coeffs, IdxT is expected to be float");
        }
    }

    void horiAntialiasAccumulate(T const* S, IdxT* L) const
    {
        IdxT alpha;
        int const cn = dst.channels();
        int const len = ctrl.xkanti * dst.cols;
        if (cn == 1)
            for (int k = 0; k < len; ++k)
            {
                int di = ctrl.xtab[k].di;
                int si = ctrl.xtab[k].si;
                ctrl.xtab[k].as(alpha);
                L[di] += S[si] * alpha;
            }
        else if (cn == 2)
            for (int k = 0; k < len; ++k)
            {
                int di = ctrl.xtab[k].di;
                int si = ctrl.xtab[k].si;
                ctrl.xtab[k].as(alpha);
                L[di    ] += S[si    ] * alpha;
                L[di + 1] += S[si + 1] * alpha;
            }
        else if (cn == 3)
            for (int k = 0; k < len; ++k)
            {
                int di = ctrl.xtab[k].di;
                int si = ctrl.xtab[k].si;
                ctrl.xtab[k].as(alpha);
                L[di    ] += S[si    ] * alpha;
                L[di + 1] += S[si + 1] * alpha;
                L[di + 2] += S[si + 2] * alpha;
            }
        else if (cn == 4)
            for (int k = 0; k < len; ++k)
            {
                int di = ctrl.xtab[k].di;
                int si = ctrl.xtab[k].si;
                ctrl.xtab[k].as(alpha);
                L[di    ] += S[si    ] * alpha;
                L[di + 1] += S[si + 1] * alpha;
                L[di + 2] += S[si + 2] * alpha;
                L[di + 3] += S[si + 3] * alpha;
            }
        else
            for (int k = 0; k < len; ++k)
            {
                int di = ctrl.xtab[k].di;
                int si = ctrl.xtab[k].si;
                ctrl.xtab[k].as(alpha);
                for (int c = 0; c < cn; ++c)
                    L[di + c] += S[si + c] * alpha;
            }
    }

    void horiAntialiasLines(T const** srcptr, WT** dstptr, IdxT* L, int count) const
    {
        int cn = dst.channels();
        int dwidth = dst.cols * cn;
        bool const same_wt_idxt = std::is_same<WT, IdxT>::value;
        for (int i = 0; i < count; ++i)
        {
            T const* S = srcptr[i];
            // reinterpret_cast makes compiler happy
            if (same_wt_idxt)
                L = reinterpret_cast<IdxT*>(dstptr[i]);
            memset(L, 0, sizeof(IdxT) * dwidth);
            horiAntialiasAccumulate(S, L);
            if (!same_wt_idxt)
            {
                // only when is_fixpt, wt (int) and idxt (float) can be different
                CV_DbgCheck(ctrl.is_fixpt, ctrl.is_fixpt
                    && (std::is_same<IdxT, float>::value)
                    && (std::is_same<WT, int>::value), "");
                float* Lf = reinterpret_cast<float*>(L);
                int* D = reinterpret_cast<int*>(dstptr[i]);
                float const alpha = INTER_RESIZE_COEF_SCALE;
                int k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 valpha = vx_setall_f32(alpha);
                int limit = dwidth - VTraits<v_float32>::vlanes();
                for (; k <= limit; k += VTraits<v_float32>::vlanes())
                    v_store(D + k, v_round(v_mul(vx_load(Lf + k), valpha)));
#endif
                for (; k < dwidth; ++k)
                    D[k] = cvRound(Lf[k] * alpha);
            }
        }
    }

    void horiGenericLines(T const** srcptr, WT** dstptr, int count) const
    {
        int cn = src.channels();
        int ssize = src.cols * cn;
        int dsize = dst.cols * cn;
        int xmin = ctrl.xmin * cn;
        int xmax = ctrl.xmax * cn;
        // just call hresize
        hresize(srcptr, dstptr, count, ctrl.xofs,
            reinterpret_cast<AT const*>(ctrl.xcoeffs), ssize, dsize, cn, xmin, xmax);
    }

    void vertAntialias(Range const& range) const
    {
        int cn = dst.channels();
        int dwidth = dst.cols * cn;
        // the sample lines on src of the i-th and (i + 1)-th dst-row
        // will overlap at most bufrow src-rows
        int bstart = 0, bufrow = ctrl.ykanti - cvFloor(1.f / ctrl.scalef.y);
        // a ring buffer, have bufrow lines, begin with bstart
        Mat buffer(bufrow + 1, dwidth * sizeof(IdxT), CV_8U);
        AutoBuffer<int> ysrc(bufrow);
        IdxT* A = buffer.template ptr<IdxT>(bufrow);
        for (int i = 0; i < bufrow; ++i)
            ysrc[i] = -1;
        for (int dy = range.start; dy < range.end; ++dy)
        {
            int tidx = dy * ctrl.ykanti;
            for (int t = 0; t < ctrl.ykanti; ++t, ++tidx)
            {
                CV_DbgCheckEQ(dy, ctrl.ytab[tidx].di, "something wrong");
                IdxT beta;
                ctrl.ytab[tidx].as(beta);
                int sy = ctrl.ytab[tidx].si;
                IdxT* L = nullptr;
                // if the sy-th row has been computed already, reuse it.
                for (int i = 0; i < bufrow; ++i)
                    if (ysrc[i] == sy)
                    {
                        L = buffer.template ptr<IdxT>(i);
                        break;
                    }
                // else, compute and save to the buffer line with the minimum ysrc
                if (!L)
                {
                    T const* S = src.template ptr<T>(sy);
                    L = buffer.template ptr<IdxT>(bstart);
                    ysrc[bstart] = sy;
                    bstart = (bstart + 1) % bufrow;
                    if (ctrl.xkanti)
                    {
                        memset(L, 0, buffer.cols * sizeof(uchar));
                        horiAntialiasAccumulate(S, L);
                    }
                    else
                    {
                        WT* Lw = reinterpret_cast<WT*>(L);
                        horiGenericLines(&S, &Lw, 1);
                    }
                }
                if (ctrl.xkanti)
                {
                    if (t == 0)
                        inter_area::mul(L, dwidth, beta, A);
                    else
                        inter_area::muladd(L, dwidth, beta, A);
                }
                else
                {
                    // A & Lw (IdxT / WT) maybe different type, can not use inter_area
                    WT* Lw = reinterpret_cast<WT*>(L);
                    if (ctrl.is_fixpt)
                        beta /= INTER_RESIZE_COEF_SCALE;
                    if (t == 0)
                        for (int w = 0; w < dwidth; ++w)
                            A[w] = saturate_cast<IdxT>(Lw[w] * beta);
                    else
                        for (int w = 0; w < dwidth; ++w)
                            A[w] += Lw[w] * beta;
                }
            }
            inter_area::saturate_store(A, dwidth, dst.template ptr<T>(dy));
        }
    }

    void vertGeneric(Range const& range) const
    {
        int ksize = ctrl.ksize, ksize2 = ksize / 2;
        int cn = src.channels();
        int dwidth = dst.cols * cn;
        size_t bufstep = alignSize(dwidth, CV_SIMD_WIDTH / sizeof(IdxT));
        AutoBuffer<IdxT> _buffer(bufstep * (ksize + 1));
        T const* srows[MAX_ESIZE] = {0};
        WT* rows[MAX_ESIZE] = {0};
        int prev_sy[MAX_ESIZE];
        IdxT* L = _buffer.data() + bufstep * ksize;
        for (int k = 0; k < ksize; ++k)
        {
            prev_sy[k] = -1;
            rows[k] = reinterpret_cast<WT*>(_buffer.data() + bufstep * k);
        }
        AT const* beta = reinterpret_cast<AT const*>(ctrl.ycoeffs) + ksize * range.start;
        for (int dy = range.start; dy < range.end; ++dy, beta += ksize)
        {
            int sy0 = ctrl.yofs[dy], k0 = ksize, k1 = 0;
            for(int k = 0; k < ksize; k++ )
            {
                int sy = min(max(sy0 - ksize2 + 1 + k, 0), src.rows - 1);
                for (k1 = max(k1, k); k1 < ksize; ++k1)
                {
                    // if the sy-th row has been computed already, reuse it.
                    if (sy == prev_sy[k1])
                    {
                        if (k1 > k)
                            memcpy(rows[k], rows[k1], bufstep * sizeof(WT));
                        break;
                    }
                }
                // remember the first row that needs to be computed
                if (k1 == ksize)
                    k0 = min(k0, k);
                srows[k] = src.template ptr<T>(sy);
                prev_sy[k] = sy;
            }

            if (k0 < ksize)
            {
                if (ctrl.xkanti)
                    horiAntialiasLines(srows + k0, rows + k0, L, ksize - k0);
                else
                    horiGenericLines(srows + k0, rows + k0, ksize - k0);
            }
            vresize(const_cast<WT const**>(rows), dst.template ptr<T>(dy), beta, dwidth);
        }
    }

    virtual void operator() (Range const& range) const CV_OVERRIDE
    {
        if (ctrl.ykanti)
            vertAntialias(range);
        else
            vertGeneric(range);
    }
};

template <typename HResize, typename VResize, typename IdxT>
static void resizeOnnx_(Mat const& src, Mat& dst, ResizeOnnxCtrl const& ctrl)
{
    /* The complexity of resize is relate to ksize and:
    - non-antialias and NN: dstsize, same as that in cv::resize.
    - antialias: dstsize and ceil(1.0 / scale). */
    double nstripes = static_cast<double>(dst.rows) * dst.cols / (1 << 16);
    // only parallel by rows
    if (ctrl.ykanti)
        nstripes *= ceil(1.0 / ctrl.scalef.y);
    // do not wake too many threads, really use the cache lines
    nstripes = min(nstripes, 2.0 * getNumberOfCPUs());
    parallel_for_(Range(0, dst.rows),
        ResizeOnnxInvoker<HResize, VResize, IdxT>(src, dst, ctrl), nstripes);
}


typedef void (*ResizeFunc)( const Mat& src, Mat& dst,
                            const int* xofs, const void* alpha,
                            const int* yofs, const void* beta,
                            int xmin, int xmax, int ksize );

typedef void (*ResizeAreaFastFunc)( const Mat& src, Mat& dst,
                                    const int* ofs, const int *xofs,
                                    int scale_x, int scale_y );

typedef void (*ResizeAreaFunc)( const Mat& src, Mat& dst,
                                const DecimateAlpha* xtab, int xtab_size,
                                const DecimateAlpha* ytab, int ytab_size,
                                const int* yofs);

typedef void (*ResizeOnnxFunc)(Mat const& src, Mat& dst, ResizeOnnxCtrl const& ctrl);


static int computeResizeAreaTab( int ssize, int dsize, int cn, double scale, DecimateAlpha* tab )
{
    int k = 0;
    for(int dx = 0; dx < dsize; dx++ )
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if( sx1 - fsx1 > 1e-3 )
        {
            CV_Assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }

        for(int sx = sx1; sx < sx2; sx++ )
        {
            CV_Assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k++].alpha = float(1.0 / cellWidth);
        }

        if( fsx2 - sx2 > 1e-3 )
        {
            CV_Assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            tab[k++].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}


#ifdef HAVE_OPENCL

static void ocl_computeResizeAreaTabs(int ssize, int dsize, double scale, int * const map_tab,
                                      float * const alpha_tab, int * const ofs_tab)
{
    int k = 0, dx = 0;
    for ( ; dx < dsize; dx++)
    {
        ofs_tab[dx] = k;

        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if (sx1 - fsx1 > 1e-3)
        {
            map_tab[k] = sx1 - 1;
            alpha_tab[k++] = (float)((sx1 - fsx1) / cellWidth);
        }

        for (int sx = sx1; sx < sx2; sx++)
        {
            map_tab[k] = sx;
            alpha_tab[k++] = float(1.0 / cellWidth);
        }

        if (fsx2 - sx2 > 1e-3)
        {
            map_tab[k] = sx2;
            alpha_tab[k++] = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    ofs_tab[dx] = k;
}

static bool ocl_resize( InputArray _src, OutputArray _dst, Size dsize,
                        double fx, double fy, int interpolation)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    double inv_fx = 1.0 / fx, inv_fy = 1.0 / fy;
    float inv_fxf = (float)inv_fx, inv_fyf = (float)inv_fy;
    int iscale_x = saturate_cast<int>(inv_fx), iscale_y = saturate_cast<int>(inv_fx);
    bool is_area_fast = std::abs(inv_fx - iscale_x) < DBL_EPSILON &&
        std::abs(inv_fy - iscale_y) < DBL_EPSILON;

    // in case of scale_x && scale_y is equal to 2
    // INTER_AREA (fast) also is equal to INTER_LINEAR
    if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
        /*interpolation = INTER_AREA*/CV_UNUSED(0); // INTER_AREA is slower

    if( !(cn <= 4 &&
           (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR ||
            (interpolation == INTER_AREA && inv_fx >= 1 && inv_fy >= 1) )) )
        return false;

    UMat src = _src.getUMat();
    _dst.create(dsize, type);
    UMat dst = _dst.getUMat();

    Size ssize = src.size();
    ocl::Kernel k;
    size_t globalsize[] = { (size_t)dst.cols, (size_t)dst.rows };

    ocl::Image2D srcImage;

    // See if this could be done with a sampler.  We stick with integer
    // datatypes because the observed error is low.
    bool useSampler = (interpolation == INTER_LINEAR && ocl::Device::getDefault().imageSupport() &&
                       ocl::Image2D::canCreateAlias(src) && depth <= 4 &&
                       ocl::Image2D::isFormatSupported(depth, cn, true) &&
                       src.offset==0);
    if (useSampler)
    {
        int wdepth = std::max(depth, CV_32S);
        char buf[2][50];
        cv::String compileOpts = format("-D USE_SAMPLER -D SRC_DEPTH=%d -D T=%s -D T1=%s "
                        "-D CONVERT_TO_DT=%s -D CN=%d",
                        depth, ocl::typeToStr(type), ocl::typeToStr(depth),
                        ocl::convertTypeStr(wdepth, depth, cn, buf[1], sizeof(buf[1])),
                        cn);
        k.create("resizeSampler", ocl::imgproc::resize_oclsrc, compileOpts);

        if (k.empty())
            useSampler = false;
        else
        {
            // Convert the input into an OpenCL image type, using normalized channel data types
            // and aliasing the UMat.
            srcImage = ocl::Image2D(src, true, true);
            k.args(srcImage, ocl::KernelArg::WriteOnly(dst),
                   (float)inv_fx, (float)inv_fy);
        }
    }

    if (interpolation == INTER_LINEAR && !useSampler)
    {
        char buf[2][50];

        // integer path is slower because of CPU part, so it's disabled
        if (depth == CV_8U && ((void)0, 0))
        {
            AutoBuffer<uchar> _buffer((dsize.width + dsize.height)*(sizeof(int) + sizeof(short)*2));
            int* xofs = (int*)_buffer.data(), * yofs = xofs + dsize.width;
            short* ialpha = (short*)(yofs + dsize.height), * ibeta = ialpha + dsize.width*2;
            float fxx, fyy;
            int sx, sy;

            for (int dx = 0; dx < dsize.width; dx++)
            {
                fxx = (float)((dx+0.5)*inv_fx - 0.5);
                sx = cvFloor(fxx);
                fxx -= sx;

                if (sx < 0)
                    fxx = 0, sx = 0;

                if (sx >= ssize.width-1)
                    fxx = 0, sx = ssize.width-1;

                xofs[dx] = sx;
                ialpha[dx*2 + 0] = saturate_cast<short>((1.f - fxx) * INTER_RESIZE_COEF_SCALE);
                ialpha[dx*2 + 1] = saturate_cast<short>(fxx         * INTER_RESIZE_COEF_SCALE);
            }

            for (int dy = 0; dy < dsize.height; dy++)
            {
                fyy = (float)((dy+0.5)*inv_fy - 0.5);
                sy = cvFloor(fyy);
                fyy -= sy;

                yofs[dy] = sy;
                ibeta[dy*2 + 0] = saturate_cast<short>((1.f - fyy) * INTER_RESIZE_COEF_SCALE);
                ibeta[dy*2 + 1] = saturate_cast<short>(fyy         * INTER_RESIZE_COEF_SCALE);
            }

            int wdepth = std::max(depth, CV_32S), wtype = CV_MAKETYPE(wdepth, cn);
            UMat coeffs;
            Mat(1, static_cast<int>(_buffer.size()), CV_8UC1, _buffer.data()).copyTo(coeffs);

            k.create("resizeLN", ocl::imgproc::resize_oclsrc,
                     format("-D INTER_LINEAR_INTEGER -D SRC_DEPTH=%d -D T=%s -D T1=%s "
                            "-D WT=%s -D CONVERT_TO_WT=%s -D CONVERT_TO_DT=%s -D CN=%d "
                            "-D INTER_RESIZE_COEF_BITS=%d",
                            depth, ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                            ocl::convertTypeStr(depth, wdepth, cn, buf[0], sizeof(buf[0])),
                            ocl::convertTypeStr(wdepth, depth, cn, buf[1], sizeof(buf[1])),
                            cn, INTER_RESIZE_COEF_BITS));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
                   ocl::KernelArg::PtrReadOnly(coeffs));
        }
        else
        {
            int wdepth = depth <= CV_8S ? CV_32S : std::max(depth, CV_32F);
            int wtype = CV_MAKETYPE(wdepth, cn);
            k.create("resizeLN", ocl::imgproc::resize_oclsrc,
                     format("-D INTER_LINEAR -D SRC_DEPTH=%d -D T=%s -D T1=%s "
                            "-D WT=%s -D CONVERT_TO_WT=%s -D CONVERT_TO_DT=%s -D CN=%d "
                            "-D INTER_RESIZE_COEF_BITS=%d",
                            depth, ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                            ocl::convertTypeStr(depth, wdepth, cn, buf[0], sizeof(buf[0])),
                            ocl::convertTypeStr(wdepth, depth, cn, buf[1], sizeof(buf[1])),
                            cn, INTER_RESIZE_COEF_BITS));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
                   (float)inv_fx, (float)inv_fy);
        }
    }
    else if (interpolation == INTER_NEAREST)
    {
        k.create("resizeNN", ocl::imgproc::resize_oclsrc,
                 format("-D INTER_NEAREST -D T=%s -D T1=%s -D CN=%d",
                        ocl::vecopTypeToStr(type), ocl::vecopTypeToStr(depth), cn));
        if (k.empty())
            return false;

        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
               (float)inv_fx, (float)inv_fy);
    }
    else if (interpolation == INTER_AREA)
    {
        int wdepth = std::max(depth, is_area_fast ? CV_32S : CV_32F);
        int wtype = CV_MAKE_TYPE(wdepth, cn);

        char cvt[2][50];
        String buildOption = format("-D INTER_AREA -D T=%s -D T1=%s -D WTV=%s -D CONVERT_TO_WTV=%s -D CN=%d",
                                    ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                                    ocl::convertTypeStr(depth, wdepth, cn, cvt[0], sizeof(cvt[0])), cn);

        UMat alphaOcl, tabofsOcl, mapOcl;
        UMat dmap, smap;

        if (is_area_fast)
        {
            int wdepth2 = std::max(CV_32F, depth), wtype2 = CV_MAKE_TYPE(wdepth2, cn);
            buildOption = buildOption + format(" -D CONVERT_TO_T=%s -D WT2V=%s -D CONVERT_TO_WT2V=%s -D INTER_AREA_FAST"
                                                " -D XSCALE=%d -D YSCALE=%d -D SCALE=%ff",
                                                ocl::convertTypeStr(wdepth2, depth, cn, cvt[0], sizeof(cvt[0])),
                                                ocl::typeToStr(wtype2), ocl::convertTypeStr(wdepth, wdepth2, cn, cvt[1], sizeof(cvt[1])),
                                    iscale_x, iscale_y, 1.0f / (iscale_x * iscale_y));

            k.create("resizeAREA_FAST", ocl::imgproc::resize_oclsrc, buildOption);
            if (k.empty())
                return false;
        }
        else
        {
            buildOption = buildOption + format(" -D CONVERT_TO_T=%s", ocl::convertTypeStr(wdepth, depth, cn, cvt[0], sizeof(cvt[0])));
            k.create("resizeAREA", ocl::imgproc::resize_oclsrc, buildOption);
            if (k.empty())
                return false;

            int xytab_size = (ssize.width + ssize.height) << 1;
            int tabofs_size = dsize.height + dsize.width + 2;

            AutoBuffer<int> _xymap_tab(xytab_size), _xyofs_tab(tabofs_size);
            AutoBuffer<float> _xyalpha_tab(xytab_size);
            int * xmap_tab = _xymap_tab.data(), * ymap_tab = _xymap_tab.data() + (ssize.width << 1);
            float * xalpha_tab = _xyalpha_tab.data(), * yalpha_tab = _xyalpha_tab.data() + (ssize.width << 1);
            int * xofs_tab = _xyofs_tab.data(), * yofs_tab = _xyofs_tab.data() + dsize.width + 1;

            ocl_computeResizeAreaTabs(ssize.width, dsize.width, inv_fx, xmap_tab, xalpha_tab, xofs_tab);
            ocl_computeResizeAreaTabs(ssize.height, dsize.height, inv_fy, ymap_tab, yalpha_tab, yofs_tab);

            // loading precomputed arrays to GPU
            Mat(1, xytab_size, CV_32FC1, _xyalpha_tab.data()).copyTo(alphaOcl);
            Mat(1, xytab_size, CV_32SC1, _xymap_tab.data()).copyTo(mapOcl);
            Mat(1, tabofs_size, CV_32SC1, _xyofs_tab.data()).copyTo(tabofsOcl);
        }

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src), dstarg = ocl::KernelArg::WriteOnly(dst);

        if (is_area_fast)
            k.args(srcarg, dstarg);
        else
            k.args(srcarg, dstarg, inv_fxf, inv_fyf, ocl::KernelArg::PtrReadOnly(tabofsOcl),
                   ocl::KernelArg::PtrReadOnly(mapOcl), ocl::KernelArg::PtrReadOnly(alphaOcl));

        return k.run(2, globalsize, NULL, false);
    }

    return k.run(2, globalsize, 0, false);
}

static void ocl_resizeOnnxTable(int srclen, int dstlen, int esz, int exclude_outside,
    int sampler, float a, float b, float A, float scale, int* offset, float* coeff)
{
    // maybe want do linear resize in this way?
    CV_Assert(sampler == INTER_LINEAR || sampler == INTER_CUBIC);
    scale = min(scale, 1.f);
    int start = cvFloor((sampler == INTER_LINEAR ? -1.f : -2.f) / scale) + 1;
    int end = 2 - start;
    int kanti = end - start;
    for (int d = 0; d < dstlen; ++d)
    {
        float spos = fmaf(static_cast<float>(d), a, b);
        int index = cvFloor(spos);
        float ratio = spos - index;
        float sum = 0.f;
        for (int i = start; i < end; ++i)
        {
            float x = fabsf(i - ratio) * scale;
            if (sampler == INTER_LINEAR)
                x = min(max(x, 0.f), 1.f);
            else
            {
                if (x <= 1)
                    x = ((A + 2) * x - (A + 3)) * x * x + 1;
                else if (x <= 2)
                    x = A * (((x - 5) * x + 8) * x - 4);
                else
                    x = 0;
            }
            int sx = index + i;
            if (exclude_outside && static_cast<unsigned>(sx) >= static_cast<unsigned>(srclen))
                x = 0;
            // make work-item(s) in a work-group load offset / coeff in one / fewer memory transaction
            // offsets & coeffs are arranged like
            //      00     10     20     ... n0
            //      01     11     21     ... n1     ...
            //      0(k-1) 1(k-1) 2(k-1) ... n(k-1)
            int to = d + (i - start) * dstlen;
            offset[to] = min(max(sx, 0), srclen - 1) * esz;
            coeff [to] = x;
            sum += x;
        }
        for (int i = 0; i < kanti; ++i)
            coeff[d + i * dstlen] /= sum;
    }
}

static char const* ocl_resizeOnnx_typeToString(int type, char* buf, size_t size)
{
    // typeToStr CV_Assert will failed
    static char const* tab[CV_64F + 1] =
        { "uchar", "char", "ushort", "short", "int", "float", "double" };
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if (cn == 1)
        return tab[depth];
    CV_Assert(size >= 18);
    snprintf(buf, size, "%s%d", tab[depth], cn);
    return buf;
}

static char const* ocl_resizeOnnx_convertTypeString(int sdepth, int ddepth, int cn, char* buf, size_t size)
{
    if( sdepth == ddepth )
        return "noconvert";
    char dtype[32];
    const char *typestr = ocl_resizeOnnx_typeToString(CV_MAKETYPE(ddepth, cn), dtype, sizeof(dtype));
    if ((ddepth >= CV_32F) ||
        (ddepth == CV_32S && sdepth < CV_32S) ||
        (ddepth == CV_16S && sdepth <= CV_8S) ||
        (ddepth == CV_16U && sdepth == CV_8U))
        snprintf(buf, size, "convert_%s", typestr);
    else if (sdepth >= CV_32F)
        snprintf(buf, size, "convert_%s%s_rte", typestr, (ddepth < CV_32S ? "_sat" : ""));
    else
        snprintf(buf, size, "convert_%s_sat", typestr);
    return buf;
}

static bool ocl_resizeOnnx(InputArray _src, OutputArray _dst,
    Matx22f const& M, Point2d const& scaled, int interpolation, float cubicCoeff)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int sampler = interpolation & INTER_SAMPLER_MASK;
    int nearest = interpolation & INTER_NEAREST_MODE_MASK;
    int antialias = interpolation & INTER_ANTIALIAS_MASK;
    int exclude_outside = interpolation & INTER_EXCLUDE_OUTSIDE_MASK;
    Point2f scale = static_cast<Point2f>(scaled);
    int khalf = (sampler == INTER_LINEAR ? 2 : 4) / 2;
    float xscale = min(scale.x, 1.f), yscale = min(scale.y, 1.f);
    int xstart = cvFloor(-khalf / xscale) + 1, xend = 2 - xstart;
    int ystart = cvFloor(-khalf / yscale) + 1, yend = 2 - ystart;

    ocl::Kernel k;
    UMat src = _src.getUMat(), dst = _dst.getUMat();
    size_t globalsize[] = {static_cast<size_t>(dst.cols), static_cast<size_t>(dst.rows)};
    char buf[6][64];
    int pixel_size = static_cast<int>(src.elemSize());
    int T = depth, VT = type;
    String buildopts, errmsg;
    // opencv ocl kernel use int for step and offset
    if (depth > CV_64F || src.size[0] * src.step[0] > INT_MAX)
        return false;

    if (sampler == INTER_NEAREST)
    {
        int W = depth, VW = type;
        float offset = (nearest == INTER_NEAREST_PREFER_FLOOR) ? -0.5f :
                (nearest == INTER_NEAREST_PREFER_CEIL) ? 0.5f : 0.f;
        static char const *nearest_name[4] = {
            "INTER_NEAREST_PREFER_FLOOR", "INTER_NEAREST_PREFER_CEIL",
            "INTER_NEAREST_FLOOR", "INTER_NEAREST_CEIL"
        };
        buildopts = format(
            "-D INTER_NEAREST -D %s "
            "-D T=%s -D W=%s -D CN=%d -D PIXEL_SIZE=%d -D VT=%s -D VW=%s "
            "-D TO_WORK=%s -D TO_VEC_WORK=%s -D TO_TYPE=%s -D TO_VEC_TYPE=%s ",
            nearest_name[nearest >> INTER_NEAREST_MODE_SHIFT],
            ocl_resizeOnnx_typeToString(T, nullptr, 0),
            ocl_resizeOnnx_typeToString(W, nullptr, 0),
            cn, pixel_size,
            ocl_resizeOnnx_typeToString(VT, buf[0], sizeof(buf[0])),
            ocl_resizeOnnx_typeToString(VW, buf[1], sizeof(buf[1])),
            ocl_resizeOnnx_convertTypeString(T, W, 1, buf[2], sizeof(buf[2])),
            ocl_resizeOnnx_convertTypeString(T, W, cn, buf[3], sizeof(buf[3])),
            ocl_resizeOnnx_convertTypeString(W, T, 1, buf[4], sizeof(buf[4])),
            ocl_resizeOnnx_convertTypeString(W, T, cn, buf[5], sizeof(buf[5]))
        );
        k.create("resizeOnnx_nearest", ocl::imgproc::resize_onnx_oclsrc,
            buildopts, &errmsg);
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
            pixel_size, offset, M(0, 0), M(0, 1), M(1, 0), M(1, 1));
    }
    else if (sampler == INTER_LINEAR && !antialias)
    {
        int W = (T < CV_32S || T == CV_32F) ? CV_32F : CV_64F, VW = CV_MAKETYPE(W, cn);
        buildopts = format(
            "-D INTER_LINEAR "
            "-D T=%s -D W=%s -D CN=%d -D PIXEL_SIZE=%d -D VT=%s -D VW=%s "
            "-D TO_WORK=%s -D TO_VEC_WORK=%s -D TO_TYPE=%s -D TO_VEC_TYPE=%s ",
            ocl_resizeOnnx_typeToString(T, nullptr, 0),
            ocl_resizeOnnx_typeToString(W, nullptr, 0),
            cn, pixel_size,
            ocl_resizeOnnx_typeToString(VT, buf[0], sizeof(buf[0])),
            ocl_resizeOnnx_typeToString(VW, buf[1], sizeof(buf[1])),
            ocl_resizeOnnx_convertTypeString(T, W, 1, buf[2], sizeof(buf[2])),
            ocl_resizeOnnx_convertTypeString(T, W, cn, buf[3], sizeof(buf[3])),
            ocl_resizeOnnx_convertTypeString(W, T, 1, buf[4], sizeof(buf[4])),
            ocl_resizeOnnx_convertTypeString(W, T, cn, buf[5], sizeof(buf[5]))
        );
        k.create("resizeOnnx_linear", ocl::imgproc::resize_onnx_oclsrc,
            buildopts, &errmsg);
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
            pixel_size, cn, M(0, 0), M(0, 1), M(1, 0), M(1, 1));
    }
    else if (sampler == INTER_LINEAR && antialias)
    {
        int W = (T < CV_32S || T == CV_32F) ? CV_32F : CV_64F, VW = CV_MAKETYPE(W, cn);
        buildopts = format(
            "-D INTER_LINEAR -D INTER_ANTIALIAS -D EXCLUDE_OUTSIDE=%d "
            "-D T=%s -D W=%s -D CN=%d -D PIXEL_SIZE=%d -D VT=%s -D VW=%s "
            "-D TO_WORK=%s -D TO_VEC_WORK=%s -D TO_TYPE=%s -D TO_VEC_TYPE=%s ",
            exclude_outside,
            ocl_resizeOnnx_typeToString(T, nullptr, 0),
            ocl_resizeOnnx_typeToString(W, nullptr, 0),
            cn, pixel_size,
            ocl_resizeOnnx_typeToString(VT, buf[0], sizeof(buf[0])),
            ocl_resizeOnnx_typeToString(VW, buf[1], sizeof(buf[1])),
            ocl_resizeOnnx_convertTypeString(T, W, 1, buf[2], sizeof(buf[2])),
            ocl_resizeOnnx_convertTypeString(T, W, cn, buf[3], sizeof(buf[3])),
            ocl_resizeOnnx_convertTypeString(W, T, 1, buf[4], sizeof(buf[4])),
            ocl_resizeOnnx_convertTypeString(W, T, cn, buf[5], sizeof(buf[5]))
        );
        k.create("resizeOnnx_linear_antialias", ocl::imgproc::resize_onnx_oclsrc,
            buildopts, &errmsg);
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
            pixel_size, cn, M(0, 0), M(0, 1), M(1, 0), M(1, 1),
            xscale, yscale, xstart, ystart, xend, yend);
    }
    else if (sampler == INTER_CUBIC && !antialias)
    {
        int W = (T < CV_32S || T == CV_32F) ? CV_32F : CV_64F, VW = CV_MAKETYPE(W, cn);
        buildopts = format(
            "-D INTER_CUBIC -D EXCLUDE_OUTSIDE=%d "
            "-D T=%s -D W=%s -D CN=%d -D PIXEL_SIZE=%d -D VT=%s -D VW=%s "
            "-D TO_WORK=%s -D TO_VEC_WORK=%s -D TO_TYPE=%s -D TO_VEC_TYPE=%s ",
            exclude_outside,
            ocl_resizeOnnx_typeToString(T, nullptr, 0),
            ocl_resizeOnnx_typeToString(W, nullptr, 0),
            cn, pixel_size,
            ocl_resizeOnnx_typeToString(VT, buf[0], sizeof(buf[0])),
            ocl_resizeOnnx_typeToString(VW, buf[1], sizeof(buf[1])),
            ocl_resizeOnnx_convertTypeString(T, W, 1, buf[2], sizeof(buf[2])),
            ocl_resizeOnnx_convertTypeString(T, W, cn, buf[3], sizeof(buf[3])),
            ocl_resizeOnnx_convertTypeString(W, T, 1, buf[4], sizeof(buf[4])),
            ocl_resizeOnnx_convertTypeString(W, T, cn, buf[5], sizeof(buf[5]))
        );
        k.create("resizeOnnx_cubic", ocl::imgproc::resize_onnx_oclsrc,
            buildopts, &errmsg);
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
            pixel_size, cn, M(0, 0), M(0, 1), M(1, 0), M(1, 1), cubicCoeff);
    }
    else if (sampler == INTER_CUBIC && antialias)
    {
        int xkanti = xend - xstart, xstride = xkanti * dst.cols;
        int ykanti = yend - ystart, ystride = ykanti * dst.rows;
        int tabsize = (xstride + ystride) * 2;
        AutoBuffer<int> table(tabsize);
        int* xoffset = table.data();
        int* yoffset = xoffset + xstride;
        float* xcoeff = reinterpret_cast<float*>(yoffset + ystride);
        float* ycoeff = reinterpret_cast<float*>(xcoeff + xstride);
        // use table coeffs, no need to define `-D EXCLUDE_OUTSIDE=%d`
        ocl_resizeOnnxTable(src.cols, dst.cols, pixel_size,
            exclude_outside, sampler, M(0, 0), M(0, 1), cubicCoeff, scale.x, xoffset, xcoeff);
        ocl_resizeOnnxTable(src.rows, dst.rows, static_cast<int>(src.step[0]),
            exclude_outside, sampler, M(1, 0), M(1, 1), cubicCoeff, scale.y, yoffset, ycoeff);
        UMat utable;
        Mat(1, tabsize, CV_32S, table.data()).copyTo(utable);
        int W = (T < CV_32S || T == CV_32F) ? CV_32F : CV_64F, VW = CV_MAKETYPE(W, cn);
        buildopts = format(
            "-D INTER_CUBIC -D INTER_ANTIALIAS "
            "-D T=%s -D W=%s -D CN=%d -D PIXEL_SIZE=%d -D VT=%s -D VW=%s "
            "-D TO_WORK=%s -D TO_VEC_WORK=%s -D TO_TYPE=%s -D TO_VEC_TYPE=%s ",
            ocl_resizeOnnx_typeToString(T, nullptr, 0),
            ocl_resizeOnnx_typeToString(W, nullptr, 0),
            cn, pixel_size,
            ocl_resizeOnnx_typeToString(VT, buf[0], sizeof(buf[0])),
            ocl_resizeOnnx_typeToString(VW, buf[1], sizeof(buf[1])),
            ocl_resizeOnnx_convertTypeString(T, W, 1, buf[2], sizeof(buf[2])),
            ocl_resizeOnnx_convertTypeString(T, W, cn, buf[3], sizeof(buf[3])),
            ocl_resizeOnnx_convertTypeString(W, T, 1, buf[4], sizeof(buf[4])),
            ocl_resizeOnnx_convertTypeString(W, T, cn, buf[5], sizeof(buf[5]))
        );
        k.create("resizeOnnx_table", ocl::imgproc::resize_onnx_oclsrc,
            buildopts, &errmsg);
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
            pixel_size, cn, xkanti, ykanti, xstride, ystride,
            ocl::KernelArg::PtrReadOnly(utable));
    }
    else
        CV_Error(cv::Error::StsError, "should not got here");

    return k.run(2, globalsize, 0, false);
}

#endif

#ifdef HAVE_IPP
#define IPP_RESIZE_PARALLEL 1

#ifdef HAVE_IPP_IW
class ipp_resizeParallel: public ParallelLoopBody
{
public:
    ipp_resizeParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, bool &ok):
        m_src(src), m_dst(dst), m_ok(ok) {}
    ~ipp_resizeParallel()
    {
    }

    void Init(IppiInterpolationType inter)
    {
        iwiResize.InitAlloc(m_src.m_size, m_dst.m_size, m_src.m_dataType, m_src.m_channels, inter, ::ipp::IwiResizeParams(0, 0, 0.75, 4), ippBorderRepl);

        m_ok = true;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION_IPP();

        if(!m_ok)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(iwiResize, m_src, m_dst, ippBorderRepl, tile);
        }
        catch(const ::ipp::IwException &)
        {
            m_ok = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    mutable ::ipp::IwiResize iwiResize;

    volatile bool &m_ok;
    const ipp_resizeParallel& operator= (const ipp_resizeParallel&);
};

class ipp_resizeAffineParallel: public ParallelLoopBody
{
public:
    ipp_resizeAffineParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, bool &ok):
        m_src(src), m_dst(dst), m_ok(ok) {}
    ~ipp_resizeAffineParallel()
    {
    }

    void Init(IppiInterpolationType inter, double scaleX, double scaleY)
    {
        double shift = (inter == ippNearest)?-1e-10:-0.5;
        double coeffs[2][3] = {
            {scaleX, 0,      shift+0.5*scaleX},
            {0,      scaleY, shift+0.5*scaleY}
        };

        iwiWarpAffine.InitAlloc(m_src.m_size, m_dst.m_size, m_src.m_dataType, m_src.m_channels, coeffs, iwTransForward, inter, ::ipp::IwiWarpAffineParams(0, 0, 0.75), ippBorderRepl);

        m_ok = true;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION_IPP();

        if(!m_ok)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(iwiWarpAffine, m_src, m_dst, tile);
        }
        catch(const ::ipp::IwException &)
        {
            m_ok = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    mutable ::ipp::IwiWarpAffine iwiWarpAffine;

    volatile bool &m_ok;
    const ipp_resizeAffineParallel& operator= (const ipp_resizeAffineParallel&);
};
#endif

static bool ipp_resize(const uchar * src_data, size_t src_step, int src_width, int src_height,
            uchar * dst_data, size_t dst_step, int dst_width, int dst_height, double inv_scale_x, double inv_scale_y,
            int depth, int channels, int interpolation)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    IppDataType           ippDataType = ippiGetDataType(depth);
    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    if((int)ippInter < 0)
        return false;

    // Resize which doesn't match OpenCV exactly
    if (!cv::ipp::useIPP_NotExact())
    {
        if (ippInter == ippNearest || ippInter == ippSuper || (ippDataType == ipp8u && ippInter == ippLinear))
            return false;
    }

    if(ippInter != ippLinear && ippDataType == ipp64f)
        return false;

#if IPP_VERSION_X100 < 201801
    // Degradations on int^2 linear downscale
    if (ippDataType != ipp64f && ippInter == ippLinear && inv_scale_x < 1 && inv_scale_y < 1) // if downscale
    {
        int scale_x = (int)(1 / inv_scale_x);
        int scale_y = (int)(1 / inv_scale_y);
        if (1 / inv_scale_x - scale_x < DBL_EPSILON && 1 / inv_scale_y - scale_y < DBL_EPSILON) // if integer
        {
            if (!(scale_x&(scale_x - 1)) && !(scale_y&(scale_y - 1))) // if power of 2
                return false;
        }
    }
#endif

    bool  affine = false;
    const double IPP_RESIZE_EPS = (depth == CV_64F)?0:1e-10;
    double ex = fabs((double)dst_width / src_width  - inv_scale_x) / inv_scale_x;
    double ey = fabs((double)dst_height / src_height - inv_scale_y) / inv_scale_y;

    // Use affine transform resize to allow sub-pixel accuracy
    if(ex > IPP_RESIZE_EPS || ey > IPP_RESIZE_EPS)
        affine = true;

    // Affine doesn't support Lanczos and Super interpolations
    if(affine && (ippInter == ippLanczos || ippInter == ippSuper))
        return false;

    try
    {
        ::ipp::IwiImage iwSrc(::ipp::IwiSize(src_width, src_height), ippDataType, channels, 0, (void*)src_data, src_step);
        ::ipp::IwiImage iwDst(::ipp::IwiSize(dst_width, dst_height), ippDataType, channels, 0, (void*)dst_data, dst_step);

        bool  ok;
        int   threads = ippiSuggestThreadsNum(iwDst, 1+((double)(src_width*src_height)/(dst_width*dst_height)));
        Range range(0, dst_height);
        ipp_resizeParallel       invokerGeneral(iwSrc, iwDst, ok);
        ipp_resizeAffineParallel invokerAffine(iwSrc, iwDst, ok);
        ParallelLoopBody        *pInvoker = NULL;
        if(affine)
        {
            pInvoker = &invokerAffine;
            invokerAffine.Init(ippInter, inv_scale_x, inv_scale_y);
        }
        else
        {
            pInvoker = &invokerGeneral;
            invokerGeneral.Init(ippInter);
        }

        if(IPP_RESIZE_PARALLEL && threads > 1)
            parallel_for_(range, *pInvoker, threads*4);
        else
            pInvoker->operator()(range);

        if(!ok)
            return false;
    }
    catch(const ::ipp::IwException &)
    {
        return false;
    }
    return true;
#else
    CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(src_width); CV_UNUSED(src_height); CV_UNUSED(dst_data); CV_UNUSED(dst_step);
    CV_UNUSED(dst_width); CV_UNUSED(dst_height); CV_UNUSED(inv_scale_x); CV_UNUSED(inv_scale_y); CV_UNUSED(depth);
    CV_UNUSED(channels); CV_UNUSED(interpolation);
    return false;
#endif
}
#endif

//==================================================================================================

namespace hal {

void resize(int src_type,
            const uchar * src_data, size_t src_step, int src_width, int src_height,
            uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
            double inv_scale_x, double inv_scale_y, int interpolation)
{
    CV_INSTRUMENT_REGION();

    CV_Assert((dst_width > 0 && dst_height > 0) || (inv_scale_x > 0 && inv_scale_y > 0));
    if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
    {
        inv_scale_x = static_cast<double>(dst_width) / src_width;
        inv_scale_y = static_cast<double>(dst_height) / src_height;
    }

    CALL_HAL(resize, cv_hal_resize, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);

    int  depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);
    Size dsize = Size(saturate_cast<int>(src_width*inv_scale_x),
                        saturate_cast<int>(src_height*inv_scale_y));
    CV_Assert( !dsize.empty() );

    CV_IPP_RUN_FAST(ipp_resize(src_data, src_step, src_width, src_height, dst_data, dst_step, dsize.width, dsize.height, inv_scale_x, inv_scale_y, depth, cn, interpolation))

    static ResizeFunc linear_tab[] =
    {
        resizeGeneric_<
            HResizeLinear<uchar, int, short,
                INTER_RESIZE_COEF_SCALE,
                HResizeLinearVec_8u32s>,
            VResizeLinear<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeLinearVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeLinear<ushort, float, float, 1,
                HResizeLinearVec_16u32f>,
            VResizeLinear<ushort, float, float, Cast<float, ushort>,
                VResizeLinearVec_32f16u> >,
        resizeGeneric_<
            HResizeLinear<short, float, float, 1,
                HResizeLinearVec_16s32f>,
            VResizeLinear<short, float, float, Cast<float, short>,
                VResizeLinearVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeLinear<float, float, float, 1,
                HResizeLinearVec_32f>,
            VResizeLinear<float, float, float, Cast<float, float>,
                VResizeLinearVec_32f> >,
        resizeGeneric_<
            HResizeLinear<double, double, float, 1,
                HResizeNoVec>,
            VResizeLinear<double, double, float, Cast<double, double>,
                VResizeNoVec> >,
        0
    };

    static ResizeFunc cubic_tab[] =
    {
        resizeGeneric_<
            HResizeCubic<uchar, int, short>,
            VResizeCubic<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeCubicVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeCubic<ushort, float, float>,
            VResizeCubic<ushort, float, float, Cast<float, ushort>,
            VResizeCubicVec_32f16u> >,
        resizeGeneric_<
            HResizeCubic<short, float, float>,
            VResizeCubic<short, float, float, Cast<float, short>,
            VResizeCubicVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeCubic<float, float, float>,
            VResizeCubic<float, float, float, Cast<float, float>,
            VResizeCubicVec_32f> >,
        resizeGeneric_<
            HResizeCubic<double, double, float>,
            VResizeCubic<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeFunc lanczos4_tab[] =
    {
        resizeGeneric_<HResizeLanczos4<uchar, int, short>,
            VResizeLanczos4<uchar, int, short,
            FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
            VResizeNoVec> >,
        0,
        resizeGeneric_<HResizeLanczos4<ushort, float, float>,
            VResizeLanczos4<ushort, float, float, Cast<float, ushort>,
            VResizeLanczos4Vec_32f16u> >,
        resizeGeneric_<HResizeLanczos4<short, float, float>,
            VResizeLanczos4<short, float, float, Cast<float, short>,
            VResizeLanczos4Vec_32f16s> >,
        0,
        resizeGeneric_<HResizeLanczos4<float, float, float>,
            VResizeLanczos4<float, float, float, Cast<float, float>,
            VResizeLanczos4Vec_32f> >,
        resizeGeneric_<HResizeLanczos4<double, double, float>,
            VResizeLanczos4<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeAreaFastFunc areafast_tab[] =
    {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar, ResizeAreaFastVec_SIMD_8u> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort, ResizeAreaFastVec_SIMD_16u> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short, ResizeAreaFastVec_SIMD_16s> >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastVec_SIMD_32f>,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0
    };

    static ResizeAreaFunc area_tab[] =
    {
        resizeArea_<uchar, float>, 0, resizeArea_<ushort, float>,
        resizeArea_<short, float>, 0, resizeArea_<float, float>,
        resizeArea_<double, double>, 0
    };

    static be_resize_func linear_exact_tab[] =
    {
        resize_bitExact<uchar, interpolationLinear<uchar> >,
        resize_bitExact<schar, interpolationLinear<schar> >,
        resize_bitExact<ushort, interpolationLinear<ushort> >,
        resize_bitExact<short, interpolationLinear<short> >,
        resize_bitExact<int, interpolationLinear<int> >,
        0,
        0,
        0
    };

    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;

    int iscale_x = saturate_cast<int>(scale_x);
    int iscale_y = saturate_cast<int>(scale_y);

    bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
            std::abs(scale_y - iscale_y) < DBL_EPSILON;

    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(dsize, src_type, dst_data, dst_step);

    if (interpolation == INTER_LINEAR_EXACT)
    {
        // in case of inv_scale_x && inv_scale_y is equal to 0.5
        // INTER_AREA (fast) is equal to bit exact INTER_LINEAR
        if (is_area_fast && iscale_x == 2 && iscale_y == 2 && cn != 2)//Area resize implementation for 2-channel images isn't bit-exact
            interpolation = INTER_AREA;
        else
        {
            be_resize_func func = linear_exact_tab[depth];
            CV_Assert(func != 0);
            func(src_data, src_step, src_width, src_height,
                 dst_data, dst_step, dst_width, dst_height,
                 cn, inv_scale_x, inv_scale_y);
            return;
        }
    }

    if( interpolation == INTER_NEAREST )
    {
        resizeNN( src, dst, inv_scale_x, inv_scale_y );
        return;
    }

    if( interpolation == INTER_NEAREST_EXACT )
    {
        resizeNN_bitexact( src, dst, inv_scale_x, inv_scale_y );
        return;
    }

    int k, sx, sy, dx, dy;


    {
        // in case of scale_x && scale_y is equal to 2
        // INTER_AREA (fast) also is equal to INTER_LINEAR
        if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
            interpolation = INTER_AREA;

        // true "area" interpolation is only implemented for the case (scale_x >= 1 && scale_y >= 1).
        // In other cases it is emulated using some variant of bilinear interpolation
        if( interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1 )
        {
            if( is_area_fast )
            {
                int area = iscale_x*iscale_y;
                size_t srcstep = src_step / src.elemSize1();
                AutoBuffer<int> _ofs(area + dsize.width*cn);
                int* ofs = _ofs.data();
                int* xofs = ofs + area;
                ResizeAreaFastFunc func = areafast_tab[depth];
                CV_Assert( func != 0 );

                for( sy = 0, k = 0; sy < iscale_y; sy++ )
                    for( sx = 0; sx < iscale_x; sx++ )
                        ofs[k++] = (int)(sy*srcstep + sx*cn);

                for( dx = 0; dx < dsize.width; dx++ )
                {
                    int j = dx * cn;
                    sx = iscale_x * j;
                    for( k = 0; k < cn; k++ )
                        xofs[j + k] = sx + k;
                }

                func( src, dst, ofs, xofs, iscale_x, iscale_y );
                return;
            }

            ResizeAreaFunc func = area_tab[depth];
            CV_Assert( func != 0 && cn <= 4 );

            AutoBuffer<DecimateAlpha> _xytab((src_width + src_height)*2);
            DecimateAlpha* xtab = _xytab.data(), *ytab = xtab + src_width*2;

            int xtab_size = computeResizeAreaTab(src_width, dsize.width, cn, scale_x, xtab);
            int ytab_size = computeResizeAreaTab(src_height, dsize.height, 1, scale_y, ytab);

            AutoBuffer<int> _tabofs(dsize.height + 1);
            int* tabofs = _tabofs.data();
            for( k = 0, dy = 0; k < ytab_size; k++ )
            {
                if( k == 0 || ytab[k].di != ytab[k-1].di )
                {
                    CV_Assert( ytab[k].di == dy );
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;

            func( src, dst, xtab, xtab_size, ytab, ytab_size, tabofs );
            return;
        }
    }

    int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == CV_8U;
    float fx, fy;
    ResizeFunc func=0;
    int ksize=0, ksize2;
    if( interpolation == INTER_CUBIC )
        ksize = 4, func = cubic_tab[depth];
    else if( interpolation == INTER_LANCZOS4 )
        ksize = 8, func = lanczos4_tab[depth];
    else if( interpolation == INTER_LINEAR || interpolation == INTER_AREA )
        ksize = 2, func = linear_tab[depth];
    else
        CV_Error( cv::Error::StsBadArg, "Unknown interpolation method" );
    ksize2 = ksize/2;

    CV_Assert( func != 0 );

    AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
    int* xofs = (int*)_buffer.data();
    int* yofs = xofs + width;
    float* alpha = (float*)(yofs + dsize.height);
    short* ialpha = (short*)alpha;
    float* beta = alpha + width*ksize;
    short* ibeta = ialpha + width*ksize;
    float cbuf[MAX_ESIZE] = {0};

    for( dx = 0; dx < dsize.width; dx++ )
    {
        if( !area_mode )
        {
            fx = (float)((dx+0.5)*scale_x - 0.5);
            sx = cvFloor(fx);
            fx -= sx;
        }
        else
        {
            sx = cvFloor(dx*scale_x);
            fx = (float)((dx+1) - (sx+1)*inv_scale_x);
            fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
        }

        if( sx < ksize2-1 )
        {
            xmin = dx+1;
            if( sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = 0;
        }

        if( sx + ksize2 >= src_width )
        {
            xmax = std::min( xmax, dx );
            if( sx >= src_width-1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = src_width-1;
        }

        for( k = 0, sx *= cn; k < cn; k++ )
            xofs[dx*cn + k] = sx + k;

        if( interpolation == INTER_CUBIC )
            interpolateCubic( fx, cbuf );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fx, cbuf );
        else
        {
            cbuf[0] = 1.f - fx;
            cbuf[1] = fx;
        }
        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            for( ; k < cn*ksize; k++ )
                ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                alpha[dx*cn*ksize + k] = cbuf[k];
            for( ; k < cn*ksize; k++ )
                alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
        }
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        if( !area_mode )
        {
            fy = (float)((dy+0.5)*scale_y - 0.5);
            sy = cvFloor(fy);
            fy -= sy;
        }
        else
        {
            sy = cvFloor(dy*scale_y);
            fy = (float)((dy+1) - (sy+1)*inv_scale_y);
            fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
        }

        yofs[dy] = sy;
        if( interpolation == INTER_CUBIC )
            interpolateCubic( fy, cbuf );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fy, cbuf );
        else
        {
            cbuf[0] = 1.f - fy;
            cbuf[1] = fy;
        }

        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                beta[dy*ksize + k] = cbuf[k];
        }
    }

    func( src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs,
          fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize );
}

} // cv::hal::
} // cv::

//==================================================================================================

void cv::resize( InputArray _src, OutputArray _dst, Size dsize,
                 double inv_scale_x, double inv_scale_y, int interpolation )
{
    CV_INSTRUMENT_REGION();

    Size ssize = _src.size();

    CV_Assert( !ssize.empty() );
    if( dsize.empty() )
    {
        CV_Assert(inv_scale_x > 0); CV_Assert(inv_scale_y > 0);
        dsize = Size(saturate_cast<int>(ssize.width*inv_scale_x),
                     saturate_cast<int>(ssize.height*inv_scale_y));
        CV_Assert( !dsize.empty() );
    }
    else
    {
        inv_scale_x = (double)dsize.width/ssize.width;
        inv_scale_y = (double)dsize.height/ssize.height;
        CV_Assert(inv_scale_x > 0); CV_Assert(inv_scale_y > 0);
    }

    if (interpolation == INTER_LINEAR_EXACT && (_src.depth() == CV_32F || _src.depth() == CV_64F))
        interpolation = INTER_LINEAR; // If depth isn't supported fallback to generic resize

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat() && _src.cols() > 10 && _src.rows() > 10,
               ocl_resize(_src, _dst, dsize, inv_scale_x, inv_scale_y, interpolation))

    // Fake reference to source. Resolves issue 13577 in case of src == dst.
    UMat srcUMat;
    if (_src.isUMat())
        srcUMat = _src.getUMat();

    Mat src = _src.getMat();
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    if (dsize == ssize)
    {
        // Source and destination are of same size. Use simple copy.
        src.copyTo(dst);
        return;
    }

    hal::resize(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows, inv_scale_x, inv_scale_y, interpolation);
}

#ifndef OPENCV_EXCLUDE_C_API

void cv::resizeOnnx(InputArray _src, OutputArray _dst,
    Size dsize, Point2d scale, int interpolation, float cubicCoeff)
{
    static_assert((1 << INTER_SAMPLER_BIT) >= INTER_MAX, "");
    CV_INSTRUMENT_REGION();

    Size ssize = _src.size();
    CV_CheckEQ(_src.dims(), 2, "only 2 dim image is support now");
    CV_CheckFalse(ssize.empty(), "src size must not be empty");
    // allow user input both dsize and scale
    if (dsize.empty())
    {
        CV_CheckGT(scale.x, 0.0, "scale must > 0 if no dsize given");
        CV_CheckGT(scale.y, 0.0, "scale must > 0 if no dsize given");
        // https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_resize.py
        // output_size = (scale_factors * np.array(data.shape)).astype(int)
        dsize.width  = static_cast<int>(scale.x * ssize.width );
        dsize.height = static_cast<int>(scale.y * ssize.height);
    }
    if (scale.x == 0 || scale.y == 0)
    {
        scale.x = static_cast<double>(dsize.width ) / ssize.width;
        scale.y = static_cast<double>(dsize.height) / ssize.height;
    }
    CV_CheckFalse(dsize.empty(), "dst size must not empty");
    CV_CheckGT(scale.x, 0.0, "require computed or given scale > 0");
    CV_CheckGT(scale.y, 0.0, "require computed or given scale > 0");

    int sampler = interpolation & INTER_SAMPLER_MASK;
    int nearest = interpolation & INTER_NEAREST_MODE_MASK;
    int coordinate = interpolation & INTER_COORDINATE_MASK;
    CV_Assert(
        sampler == INTER_NEAREST ||
        sampler == INTER_LINEAR ||
        sampler == INTER_CUBIC);
    CV_Assert(
        nearest == INTER_NEAREST_PREFER_FLOOR ||
        nearest == INTER_NEAREST_PREFER_CEIL ||
        nearest == INTER_NEAREST_FLOOR ||
        nearest == INTER_NEAREST_CEIL);
    CV_Assert(
        coordinate == INTER_HALF_PIXEL ||
        coordinate == INTER_HALF_PIXEL_PYTORCH ||
        coordinate == INTER_HALF_PIXEL_SYMMETRIC ||
        coordinate == INTER_ALIGN_CORNERS ||
        coordinate == INTER_ASYMMETRIC);

    // x_org = x * a + b
    Matx22f M;
    Vec2f xcoef = interCoordinate(coordinate, dsize.width, ssize.width, scale.x);
    Vec2f ycoef = interCoordinate(coordinate, dsize.height, ssize.height, scale.y);
    M(0, 0) = xcoef[0];
    M(0, 1) = xcoef[1];
    M(1, 0) = ycoef[0];
    M(1, 1) = ycoef[1];

    _dst.create(dsize, _src.type());
    if (dsize == ssize)
    {
        // Source and destination are of same size. Use simple copy.
        _src.copyTo(_dst);
        return;
    }
    // Antialias is applied when downsampling
    if (scale.x >= 1.0 && scale.y >= 1.0)
        interpolation &= ~INTER_ANTIALIAS_MASK;

    // Fake reference to source. Resolves issue 13577 in case of src == dst.
    UMat srcUMat;
    if (_src.isUMat())
        srcUMat = _src.getUMat();

    CV_OCL_RUN(_src.isUMat() && _dst.isUMat(),
        ocl_resizeOnnx(_src, _dst, M, scale, interpolation, cubicCoeff))
    // if (cv::ocl::isOpenCLActivated() && _src.isUMat() && _dst.isUMat())
        // CV_Assert(ocl_resizeOnnx(_src, _dst, M, scale, interpolation, cubicCoeff));

    Mat src = _src.getMat(), dst = _dst.getMat();

    if (sampler == INTER_NEAREST)
    {
        parallel_for_(Range(0, dsize.height),
            ResizeOnnxNNInvoker(src, dst, M, nearest),
            static_cast<double>(dsize.height) * dsize.width / (1 << 16));
        return;
    }

    static ResizeOnnxFunc linear_tab[CV_DEPTH_MAX] =
    {
        resizeOnnx_<
            HResizeLinear<uchar, int, short, INTER_RESIZE_COEF_SCALE, HResizeLinearVec_8u32s>,
            VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
                VResizeLinearVec_32s8u>,
            float>,
        resizeOnnx_<
            HResizeLinear<schar, int, short, INTER_RESIZE_COEF_SCALE, HResizeNoVec>,
            VResizeLinear<schar, int, short, FixedPtCast<int, schar, INTER_RESIZE_COEF_BITS * 2>,
                VResizeNoVec>,
            float>,
        resizeOnnx_<
            HResizeLinear<ushort, float, float, 1, HResizeLinearVec_16u32f>,
            VResizeLinear<ushort, float, float, Cast<float, ushort>, VResizeLinearVec_32f16u>,
            float>,
        resizeOnnx_<
            HResizeLinear<short, float, float, 1, HResizeLinearVec_16s32f>,
            VResizeLinear<short, float, float, Cast<float, short>, VResizeLinearVec_32f16s>,
            float>,
        resizeOnnx_<
            HResizeLinear<int, double, double, 1, HResizeNoVec>,
            VResizeLinear<int, double, double, Cast<double, int>, VResizeNoVec>,
            double>,
        resizeOnnx_<
            HResizeLinear<float, float, float, 1, HResizeLinearVec_32f>,
            VResizeLinear<float, float, float, Cast<float, float>, VResizeLinearVec_32f>,
            float>,
        resizeOnnx_<
            HResizeLinear<double, double, double, 1, HResizeNoVec>,
            VResizeLinear<double, double, double, Cast<double, double>, VResizeNoVec>,
            double>,
        nullptr
    };

    static ResizeOnnxFunc cubic_tab[CV_DEPTH_MAX] =
    {
        resizeOnnx_<
            HResizeCubic<uchar, int, short>,
            VResizeCubic<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
                VResizeCubicVec_32s8u>,
            float>,
        resizeOnnx_<
            HResizeCubic<schar, int, short>,
            VResizeCubic<schar, int, short, FixedPtCast<int, schar, INTER_RESIZE_COEF_BITS * 2>,
                VResizeNoVec>,
            float>,
        resizeOnnx_<
            HResizeCubic<ushort, float, float>,
            VResizeCubic<ushort, float, float, Cast<float, ushort>, VResizeCubicVec_32f16u>,
            float>,
        resizeOnnx_<
            HResizeCubic<short, float, float>,
            VResizeCubic<short, float, float, Cast<float, short>, VResizeCubicVec_32f16s>,
            float>,
        resizeOnnx_<
            HResizeCubic<int, double, double>,
            VResizeCubic<int, double, double, Cast<double, int>, VResizeNoVec>,
            double>,
        resizeOnnx_<
            HResizeCubic<float, float, float>,
            VResizeCubic<float, float, float, Cast<float, float>, VResizeCubicVec_32f>,
            float>,
        resizeOnnx_<
            HResizeCubic<double, double, double>,
            VResizeCubic<double, double, double, Cast<double, double>, VResizeNoVec>,
            double>,
        nullptr
    };

    int depth = src.depth(), type = src.type();
    ResizeOnnxCtrl ctrl(interpolation, type, cubicCoeff, ssize, dsize, scale, M);
    ResizeOnnxFunc func = linear_tab[depth];
    if (sampler == INTER_LINEAR)
        func = linear_tab[depth];
    else if (sampler == INTER_CUBIC)
        func = cubic_tab[depth];
    else
        CV_Error(CV_StsBadArg, format("Unknown sampler %d", sampler));

    func(src, dst, ctrl);
}


CV_IMPL void
cvResize( const CvArr* srcarr, CvArr* dstarr, int method )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() );
    cv::resize( src, dst, dst.size(), (double)dst.cols/src.cols,
        (double)dst.rows/src.rows, method );
}

#endif
/* End of file. */
