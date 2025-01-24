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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
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

/********************************* COPYRIGHT NOTICE *******************************\
  Original code for Bayer->BGR/RGB conversion is provided by Dirk Schaefer
  from MD-Mathematische Dienste GmbH. Below is the copyright notice:

    IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
    By downloading, copying, installing or using the software you agree
    to this license. If you do not agree to this license, do not download,
    install, copy or use the software.

    Contributors License Agreement:

      Copyright (c) 2002,
      MD-Mathematische Dienste GmbH
      Im Defdahl 5-10
      44141 Dortmund
      Germany
      www.md-it.de

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain
    the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    The name of Contributor may not be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************/


#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include <limits>

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

namespace cv
{


//////////////////////////// Bayer Pattern -> RGB conversion /////////////////////////////

template<typename T>
class SIMDBayerStubInterpolator_
{
public:
    int bayer2Gray(const T*, int, T*, int, int, int, int) const
    {
        return 0;
    }

    int bayer2RGB(const T*, int, T*, int, int) const
    {
        return 0;
    }

    int bayer2RGBA(const T*, int, T*, int, int, const T) const
    {
        return 0;
    }

    int bayer2RGB_EA(const T*, int, T*, int, int) const
    {
        return 0;
    }
};

#if CV_SIMD256
class SIMDBayerInterpolator_8u
{
public:
    int bayer2Gray(const uchar* bayer, int bayer_step, uchar* dst,
                   int width, int bcoeff, int gcoeff, int rcoeff) const
    {
        v_uint16x16 v255 = v256_setall_u16(255);
        v_int16x16 v_descale = v256_setall_s16(static_cast<short>(1 << 14));
        v_int16x16 dummy;
        v_int16x16 cxrb;
        v_int16x16 cxg2;
        v_zip(v256_setall_s16(static_cast<short>(rcoeff)),
              v256_setall_s16(static_cast<short>(bcoeff)),
              cxrb,
              dummy);
        v_zip(v256_setall_s16(static_cast<short>(gcoeff)),
              v256_setall_s16(static_cast<short>(2)),
              cxg2,
              dummy);

        const uchar* bayer_end = bayer + width;

        for (; bayer < bayer_end - 34; bayer += 30, dst += 30)
        {
            v_uint16x16 first_line = v_reinterpret_as_u16(v256_load(bayer));
            v_uint16x16 second_line = v_reinterpret_as_u16(v256_load(bayer + bayer_step));
            v_uint16x16 third_line = v_reinterpret_as_u16(v256_load(bayer + bayer_step * 2));

            // bayer[0]
            v_uint16x16 first_line0 = v_and(first_line, v255);
            // bayer[bayer_step*2]
            v_uint16x16 third_line0 = v_and(third_line, v255);
            // bayer[0] + bayer[bayer_step*2]
            v_uint16x16 first_third_line0 = v_add(first_line0, third_line0);
            // bayer[2] + bayer[bayer_step*2+2]
            v_uint16x16 first_third_line2 = v_rotate_right<1>(first_third_line0);
            // bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2]
            v_int16x16 r0 = v_reinterpret_as_s16(v_add(first_third_line0, first_third_line2));
            // (bayer[2] + bayer[bayer_step*2+2]) * 2
            v_int16x16 r1 = v_reinterpret_as_s16(v_shl<1>(first_third_line2));

            // bayer[bayer_step+1]
            v_uint16x16 second_line1 = v_shr<8>(second_line);
            // bayer[bayer_step+1] * 4
            v_int16x16 b0 = v_reinterpret_as_s16(v_shl<2>(second_line1));
            // bayer[bayer_step+3]
            v_uint16x16 second_line3 = v_rotate_right<1>(second_line1);
            // bayer[bayer_step+1] + bayer[bayer_step+3]
            v_uint16x16 second_line13 = v_add(second_line1, second_line3);
            // (bayer[bayer_step+1] + bayer[bayer_step+3]) * 2
            v_int16x16 b1 = v_reinterpret_as_s16(v_shl(second_line13, 1));

            // bayer[1]
            v_uint16x16 first_line1 = v_shr<8>(first_line);
            // bayer[bayer_step]
            v_uint16x16 second_line0 = v_and(second_line, v255);
            // bayer[bayer_step+2]
            v_uint16x16 second_line2 = v_rotate_right<1>(second_line0);
            // bayer[bayer_step] + bayer[bayer_step+2]
            v_uint16x16 second_line02 = v_add(second_line0, second_line2);
            // bayer[bayer_step*2+1]
            v_uint16x16 third_line1 = v_shr<8>(third_line);
            // bayer[1] + bayer[bayer_step*2+1]
            v_uint16x16 first_third_line1 = v_add(first_line1, third_line1);
            // bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1]
            v_int16x16 g0 = v_reinterpret_as_s16(v_add(first_third_line1, second_line02));
            // bayer[bayer_step+2] * 4
            v_int16x16 g1 = v_reinterpret_as_s16(v_shl<2>(second_line2));

            v_int16x16 rb0;
            v_int16x16 rb1;
            v_int16x16 rb2;
            v_int16x16 rb3;
            v_zip(r0, b0, rb0, rb1);
            v_zip(r1, b1, rb2, rb3);

            v_int16x16 gd0;
            v_int16x16 gd1;
            v_int16x16 gd2;
            v_int16x16 gd3;
            v_zip(g0, v_descale, gd0, gd1);
            v_zip(g1, v_descale, gd2, gd3);

            v_int32x8 gray_even0 = v_shr<16>(v_add(v_dotprod(rb0, cxrb), v_dotprod(gd0, cxg2)));
            v_int32x8 gray_even1 = v_shr<16>(v_add(v_dotprod(rb1, cxrb), v_dotprod(gd1, cxg2)));
            v_int32x8 gray_odd0 = v_shr<16>(v_add(v_dotprod(rb2, cxrb), v_dotprod(gd2, cxg2)));
            v_int32x8 gray_odd1 = v_shr<16>(v_add(v_dotprod(rb3, cxrb), v_dotprod(gd3, cxg2)));

            v_int16x16 gray_even = v_pack(gray_even0, gray_even1);
            v_int16x16 gray_odd = v_pack(gray_odd0, gray_odd1);

            v_int16x16 gray_d0;
            v_int16x16 gray_d1;
            v_zip(gray_even, gray_odd, gray_d0, gray_d1);

            v_uint8x32 gray = v_pack(v_reinterpret_as_u16(gray_d0), v_reinterpret_as_u16(gray_d1));

            v_store(dst, gray);
        }
        return (int)(bayer - (bayer_end - width));
    }

    int bayer2RGB(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue) const
    {
        v_uint16x16 delta1 = v256_setall_u16(1), delta2 = v256_setall_u16(2);
        v_uint16x16 mask = v256_setall_u16(blue < 0 ? (ushort)(-1) : 0);
        v_uint16x16 masklo = v256_setall_u16(0x00ff);
        v_uint8x32 z = v256_setzero_u8();
        const uchar* bayer_end = bayer + width;
        for( ; bayer <= bayer_end - 34; bayer += 28, dst += 84 )
        {
            v_uint16x16 r0 = v_reinterpret_as_u16(v256_load(bayer));
            v_uint16x16 r1 = v_reinterpret_as_u16(v256_load(bayer+bayer_step));
            v_uint16x16 r2 = v_reinterpret_as_u16(v256_load(bayer+bayer_step*2));

            v_uint16x16 b1 = v_add(v_and(r0, masklo), v_and(r2, masklo));
            v_uint16x16 nextb1 = v_rotate_right<1>(b1);
            v_uint16x16 b0 = v_add(b1, nextb1);
            b1 = v_shr<1>(v_add(nextb1, delta1));
            b0 = v_shr<2>(v_add(b0, delta2));
            // b0 b2 ... b30 b1 b3 ... b31
            b0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1)));

            v_uint16x16 g0 = v_add(v_shr<8>(r0), v_shr<8>(r2));
            v_uint16x16 g1 = v_and(r1, masklo);
            g0 = v_add(g0, v_add(v_rotate_right<1>(g1), g1));
            g1 = v_rotate_right<1>(g1);
            g0 = v_shr<2>(v_add(g0, delta2));
            // g0 g2 ... g30 g1 g3 ... g31
            g0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(g0), v_reinterpret_as_s16(g1)));

            r0 = v_shr<8>(r1);
            r1 = v_add(v_rotate_right<1>(r0), r0);
            r1 = v_shr<1>(v_add(r1, delta1));
            // r0 r2 ... r30 r1 r3 ... r31
            r0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(r0), v_reinterpret_as_s16(r1)));

            b1 = v_and(v_xor(b0, r0), mask);
            b0 = v_xor(b0, b1);
            r0 = v_xor(r0, b1);

            // b1 g1 b3 g3 b5 g5...
            v_uint8x32 pack_lo, pack_hi;
            v_zip(v_reinterpret_as_u8(b0), v_reinterpret_as_u8(g0), pack_lo, pack_hi);
            b1 = v_reinterpret_as_u16(pack_hi);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = v_reinterpret_as_u16(pack_lo);

            // r1 0 r3 0 r5 0 ...
            v_zip(v_reinterpret_as_u8(r0), z, pack_lo, pack_hi);
            r1 = v_reinterpret_as_u16(pack_hi);
            // r0 0 r2 0 r4 0 ...
            r0 = v_reinterpret_as_u16(pack_lo);

            // 0 b0 g0 r0 0 b2 g2 r2 ...
            v_zip(b0, r0, g0, g1);
            g0 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g0)));
            // 0 b16 g16 r16 0 b18 g18 r18 ...
            g1 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g1)));

            // b1 g1 r1 0 b3 g3 r3 0 ...
            v_zip(b1, r1, r0, r1);
            // b17 g17 r17 0 b19 g19 r19 0 ...

            // 0 b0 g0 r0 b1 g1 r1 0 ...
            v_uint32x8 pack32_lo, pack32_hi;
            v_zip(v_reinterpret_as_u32(g0), v_reinterpret_as_u32(r0), pack32_lo, pack32_hi);
            b0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b8 g8 r8 b9 g9 r9 0 ...
            b1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            // store x[63:0] to ptr
            #define v_store_1_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm256_castsi256_si128(x.val))
            // store x[127:64] to ptr
            #define v_store_2_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm_unpackhi_epi64(_mm256_castsi256_si128(x.val), _mm256_castsi256_si128(x.val)))
            // store x[191:128] to ptr
            #define v_store_3_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm256_extracti128_si256(x.val, 1))
            // store x[255:192] to ptr
            #define v_store_4_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm_unpackhi_epi64(_mm256_extracti128_si256(x.val, 1), _mm256_extracti128_si256(x.val, 1)))

            v_store_1_4(dst-1+0, b0);
            v_store_2_4(dst-1+6*1, b0);
            v_store_3_4(dst-1+6*2, b0);
            v_store_4_4(dst-1+6*3, b0);
            v_store_1_4(dst-1+6*4, b1);
            v_store_2_4(dst-1+6*5, b1);
            v_store_3_4(dst-1+6*6, b1);
            v_store_4_4(dst-1+6*7, b1);

            // 0 b8 g8 r8 b9 g9 r9 0 ...
            v_zip(v_reinterpret_as_u32(g1), v_reinterpret_as_u32(r1), pack32_lo, pack32_hi);
            g0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b12 g12 r12 b13 g13 r13 0 ...
            g1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            v_store_1_4(dst-1+6*8, g0);
            v_store_2_4(dst-1+6*9, g0);
            v_store_3_4(dst-1+6*10, g0);
            v_store_4_4(dst-1+6*11, g0);
            v_store_1_4(dst-1+6*12, g1);
            v_store_2_4(dst-1+6*13, g1);
        }

        return (int)(bayer - (bayer_end - width));
    }

    int bayer2RGBA(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue, const uchar alpha) const
    {
        v_uint16x16 delta1 = v256_setall_u16(1), delta2 = v256_setall_u16(2);
        v_uint16x16 mask = v256_setall_u16(blue < 0 ? (ushort)(-1) : 0);
        v_uint16x16 masklo = v256_setall_u16(0x00ff);
        v_uint8x32 a = v256_setall_u8(alpha);
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 34; bayer += 28, dst += 112 )
        {
            v_uint16x16 r0 = v_reinterpret_as_u16(v256_load(bayer));
            v_uint16x16 r1 = v_reinterpret_as_u16(v256_load(bayer+bayer_step));
            v_uint16x16 r2 = v_reinterpret_as_u16(v256_load(bayer+bayer_step*2));

            v_uint16x16 b1 = v_add(v_and(r0, masklo), v_and(r2, masklo));
            v_uint16x16 nextb1 = v_rotate_right<1>(b1);
            v_uint16x16 b0 = v_add(b1, nextb1);
            b1 = v_shr<1>(v_add(nextb1, delta1));
            b0 = v_shr<2>(v_add(b0, delta2));
            // b0 b2 ... b30 b1 b3 ... b31
            b0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1)));

            v_uint16x16 g0 = v_add(v_shr<8>(r0), v_shr<8>(r2));
            v_uint16x16 g1 = v_and(r1, masklo);
            g0 = v_add(g0, v_add(v_rotate_right<1>(g1), g1));
            g1 = v_rotate_right<1>(g1);
            g0 = v_shr<2>(v_add(g0, delta2));
            // g0 g2 ... g30 g1 g3 ... g31
            g0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(g0), v_reinterpret_as_s16(g1)));

            r0 = v_shr<8>(r1);
            r1 = v_add(v_rotate_right<1>(r0), r0);
            r1 = v_shr<1>(v_add(r1, delta1));
            // r0 r2 ... r30 r1 r3 ... r31
            r0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(r0), v_reinterpret_as_s16(r1)));

            b1 = v_and(v_xor(b0, r0), mask);
            b0 = v_xor(b0, b1);
            r0 = v_xor(r0, b1);

            // b1 g1 b3 g3 b5 g5...
            v_uint8x32 pack_lo, pack_hi;
            v_zip(v_reinterpret_as_u8(b0), v_reinterpret_as_u8(g0), pack_lo, pack_hi);
            b1 = v_reinterpret_as_u16(pack_hi);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = v_reinterpret_as_u16(pack_lo);

            // r1 a r3 a r5 a ...
            v_zip(v_reinterpret_as_u8(r0), a, pack_lo, pack_hi);
            r1 = v_reinterpret_as_u16(pack_hi);
            // r0 a r2 a r4 a ...
            r0 = v_reinterpret_as_u16(pack_lo);

            // b0 g0 r0 a b2 g2 r2 a ...
            v_zip(b0, r0, g0, g1);
            // b16 g16 r16 a b18 g18 r18 a ...

            // b1 g1 r1 a b3 g3 r3 a ...
            v_zip(b1, r1, r0, r1);
            // b17 g17 r17 a b19 g19 r19 a ...

            v_uint32x8 pack32_lo, pack32_hi;
            v_zip(v_reinterpret_as_u32(g0), v_reinterpret_as_u32(r0), pack32_lo, pack32_hi);
            // b0 g0 r0 a b1 g1 r1 a ...
            b0 = v_reinterpret_as_u16(pack32_lo);
            b1 = v_reinterpret_as_u16(pack32_hi);

            v_store_low(dst-1+0, v_reinterpret_as_u8(b0));
            v_store_high(dst-1+16*1, v_reinterpret_as_u8(b0));
            v_store_low(dst-1+16*2, v_reinterpret_as_u8(b1));
            v_store_high(dst-1+16*3, v_reinterpret_as_u8(b1));

            // b16 g16 r16 a b17 g17 r17 a ...
            v_zip(v_reinterpret_as_u32(g1), v_reinterpret_as_u32(r1), pack32_lo, pack32_hi);
            g0 = v_reinterpret_as_u16(pack32_lo);
            g1 = v_reinterpret_as_u16(pack32_hi);

            v_store_low(dst-1+16*4, v_reinterpret_as_u8(g0));
            v_store_high(dst-1+16*5, v_reinterpret_as_u8(g0));

            v_store_low(dst-1+16*6, v_reinterpret_as_u8(g1));
        }

        return (int)(bayer - (bayer_end - width));
    }

    int bayer2RGB_EA(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue) const
    {
        const uchar* bayer_end = bayer + width;
        v_uint16x16 masklow = v256_setall_u16(0x00ff);
        v_uint16x16 delta1 = v256_setall_u16(1), delta2 = v256_setall_u16(2);
        v_uint16x16 full = v256_setall_u16((ushort)(-1));
        v_uint8x32 z = v256_setzero_u8();
        v_uint16x16 mask = v256_setall_u16(blue > 0 ? (ushort)(-1) : 0);

        for ( ; bayer <= bayer_end - 34; bayer += 28, dst += 84)
        {
            v_uint16x16 r0 = v_reinterpret_as_u16(v256_load(bayer));
            v_uint16x16 r1 = v_reinterpret_as_u16(v256_load(bayer+bayer_step));
            v_uint16x16 r2 = v_reinterpret_as_u16(v256_load(bayer+bayer_step*2));

            v_uint16x16 b1 = v_add(v_and(r0, masklow), v_and(r2, masklow));
            v_uint16x16 nextb1 = v_rotate_right<1>(b1);
            v_uint16x16 b0 = v_add(b1, nextb1);
            b1 = v_shr<1>(v_add(nextb1, delta1));
            b0 = v_shr<2>(v_add(b0, delta2));
            // b0 b2 ... b30 b1 b3 ... b31
            b0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1)));

            // vertical sum
            v_uint16x16 r0g = v_shr<8>(r0);
            v_uint16x16 r2g = v_shr<8>(r2);
            v_uint16x16 sumv = v_shr<1>(v_add(v_add(r0g, r2g), delta1));
            // horizontal sum
            v_uint16x16 g1 = v_and(r1, masklow);
            v_uint16x16 nextg1 = v_rotate_right<1>(g1);
            v_uint16x16 sumg = v_shr<1>(v_add(v_add(g1, nextg1), delta1));

            // gradients
            v_uint16x16 gradv = v_add(v_sub(r0g, r2g), v_sub(r2g, r0g));
            v_uint16x16 gradg = v_add(v_sub(nextg1, g1), v_sub(g1, nextg1));
            v_uint16x16 gmask = v_gt(gradg, gradv);
            v_uint16x16 g0 = v_add(v_and(gmask, sumv), v_and(sumg, v_xor(gmask, full)));
            // g0 g2 ... g14 g1 g3 ...
            g0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(g0), v_reinterpret_as_s16(nextg1)));

            r0 = v_shr<8>(r1);
            r1 = v_add(v_rotate_right<1>(r0), r0);
            r1 = v_shr<1>(v_add(r1, delta1));
            // r0 r2 ... r30 r1 r3 ... r31
            r0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(r0), v_reinterpret_as_s16(r1)));

            b1 = v_and(v_xor(b0, r0), mask);
            b0 = v_xor(b0, b1);
            r0 = v_xor(r0, b1);

            // b1 g1 b3 g3 b5 g5...
            v_uint8x32 pack_lo, pack_hi;
            v_zip(v_reinterpret_as_u8(b0), v_reinterpret_as_u8(g0), pack_lo, pack_hi);
            b1 = v_reinterpret_as_u16(pack_hi);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = v_reinterpret_as_u16(pack_lo);

            // r1 0 r3 0 r5 0 ...
            v_zip(v_reinterpret_as_u8(r0), z, pack_lo, pack_hi);
            r1 = v_reinterpret_as_u16(pack_hi);
            // r0 0 r2 0 r4 0 ...
            r0 = v_reinterpret_as_u16(pack_lo);

            // 0 b0 g0 r0 0 b2 g2 r2 ...
            v_zip(b0, r0, g0, g1);
            g0 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g0)));
            // 0 b16 g16 r16 0 b18 g18 r18 ...
            g1 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g1)));

            // b1 g1 r1 0 b3 g3 r3 0 ...
            v_zip(b1, r1, r0, r1);
            // b17 g17 r17 0 b19 g19 r19 0 ...

            // 0 b0 g0 r0 b1 g1 r1 0 ...
            v_uint32x8 pack32_lo, pack32_hi;
            v_zip(v_reinterpret_as_u32(g0), v_reinterpret_as_u32(r0), pack32_lo, pack32_hi);
            b0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b8 g8 r8 b9 g9 r9 0 ...
            b1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            // store x[63:0] to ptr
            #define v_store_1_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm256_castsi256_si128(x.val))
            // store x[127:64] to ptr
            #define v_store_2_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm_unpackhi_epi64(_mm256_castsi256_si128(x.val), _mm256_castsi256_si128(x.val)))
            // store x[191:128] to ptr
            #define v_store_3_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm256_extracti128_si256(x.val, 1))
            // store x[255:192] to ptr
            #define v_store_4_4(ptr, x) _mm_storel_epi64((__m128i*)(ptr), _mm_unpackhi_epi64(_mm256_extracti128_si256(x.val, 1), _mm256_extracti128_si256(x.val, 1)))

            v_store_1_4(dst-1+0, b0);
            v_store_2_4(dst-1+6*1, b0);
            v_store_3_4(dst-1+6*2, b0);
            v_store_4_4(dst-1+6*3, b0);
            v_store_1_4(dst-1+6*4, b1);
            v_store_2_4(dst-1+6*5, b1);
            v_store_3_4(dst-1+6*6, b1);
            v_store_4_4(dst-1+6*7, b1);

            // 0 b8 g8 r8 b9 g9 r9 0 ...
            v_zip(v_reinterpret_as_u32(g1), v_reinterpret_as_u32(r1), pack32_lo, pack32_hi);
            g0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b12 g12 r12 b13 g13 r13 0 ...
            g1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            v_store_1_4(dst-1+6*8, g0);
            v_store_2_4(dst-1+6*9, g0);
            v_store_3_4(dst-1+6*10, g0);
            v_store_4_4(dst-1+6*11, g0);
            v_store_1_4(dst-1+6*12, g1);
            v_store_2_4(dst-1+6*13, g1);
        }

        return int(bayer - (bayer_end - width));
    }
};
#elif CV_SIMD128
class SIMDBayerInterpolator_8u
{
public:
    int bayer2Gray(const uchar* bayer, int bayer_step, uchar* dst,
                   int width, int bcoeff, int gcoeff, int rcoeff) const
    {
#if CV_NEON
        uint16x8_t masklo = vdupq_n_u16(255);
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 14 )
        {
            uint16x8_t r0 = vld1q_u16((const ushort*)bayer);
            uint16x8_t r1 = vld1q_u16((const ushort*)(bayer + bayer_step));
            uint16x8_t r2 = vld1q_u16((const ushort*)(bayer + bayer_step*2));

            uint16x8_t b1_ = vaddq_u16(vandq_u16(r0, masklo), vandq_u16(r2, masklo));
            uint16x8_t b1 = vextq_u16(b1_, b1_, 1);
            uint16x8_t b0 = vaddq_u16(b1_, b1);
            // b0 = b0 b2 b4 ...
            // b1 = b1 b3 b5 ...

            uint16x8_t g0 = vaddq_u16(vshrq_n_u16(r0, 8), vshrq_n_u16(r2, 8));
            uint16x8_t g1 = vandq_u16(r1, masklo);
            g0 = vaddq_u16(g0, vaddq_u16(g1, vextq_u16(g1, g1, 1)));
            uint16x8_t rot = vextq_u16(g1, g1, 1);
            g1 = vshlq_n_u16(rot, 2);
            // g0 = b0 b2 b4 ...
            // g1 = b1 b3 b5 ...

            r0 = vshrq_n_u16(r1, 8);
            r1 = vaddq_u16(r0, vextq_u16(r0, r0, 1));
            r0 = vshlq_n_u16(r0, 2);
            // r0 = r0 r2 r4 ...
            // r1 = r1 r3 r5 ...

            b0 = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(b0), (short)(rcoeff*2)));
            b1 = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(b1), (short)(rcoeff*4)));

            g0 = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(g0), (short)(gcoeff*2)));
            g1 = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(g1), (short)(gcoeff*2)));

            r0 = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(r0), (short)(bcoeff*2)));
            r1 = vreinterpretq_u16_s16(vqdmulhq_n_s16(vreinterpretq_s16_u16(r1), (short)(bcoeff*4)));

            g0 = vaddq_u16(vaddq_u16(g0, b0), r0);
            g1 = vaddq_u16(vaddq_u16(g1, b1), r1);

            uint8x8x2_t p = vzip_u8(vrshrn_n_u16(g0, 2), vrshrn_n_u16(g1, 2));
            vst1_u8(dst, p.val[0]);
            vst1_u8(dst + 8, p.val[1]);
        }
#else
        v_uint16x8 v255 = v_setall_u16(255);
        v_int16x8 v_descale = v_setall_s16(static_cast<short>(1 << 14));
        v_int16x8 dummy;
        v_int16x8 cxrb;
        v_int16x8 cxg2;
        v_zip(v_setall_s16(static_cast<short>(rcoeff)),
              v_setall_s16(static_cast<short>(bcoeff)),
              cxrb,
              dummy);
        v_zip(v_setall_s16(static_cast<short>(gcoeff)),
              v_setall_s16(static_cast<short>(2)),
              cxg2,
              dummy);

        const uchar* bayer_end = bayer + width;

        for (; bayer < bayer_end - 14; bayer += 14, dst += 14)
        {
            v_uint16x8 first_line = v_reinterpret_as_u16(v_load(bayer));
            v_uint16x8 second_line = v_reinterpret_as_u16(v_load(bayer + bayer_step));
            v_uint16x8 third_line = v_reinterpret_as_u16(v_load(bayer + bayer_step * 2));

            // bayer[0]
            v_uint16x8 first_line0 = v_and(first_line, v255);
            // bayer[bayer_step*2]
            v_uint16x8 third_line0 = v_and(third_line, v255);
            // bayer[0] + bayer[bayer_step*2]
            v_uint16x8 first_third_line0 = v_add(first_line0, third_line0);
            // bayer[2] + bayer[bayer_step*2+2]
            v_uint16x8 first_third_line2 = v_rotate_right<1>(first_third_line0);
            // bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2]
            v_int16x8 r0 = v_reinterpret_as_s16(v_add(first_third_line0, first_third_line2));
            // (bayer[2] + bayer[bayer_step*2+2]) * 2
            v_int16x8 r1 = v_reinterpret_as_s16(v_shl<1>(first_third_line2));

            // bayer[bayer_step+1]
            v_uint16x8 second_line1 = v_shr<8>(second_line);
            // bayer[bayer_step+1] * 4
            v_int16x8 b0 = v_reinterpret_as_s16(v_shl<2>(second_line1));
            // bayer[bayer_step+3]
            v_uint16x8 second_line3 = v_rotate_right<1>(second_line1);
            // bayer[bayer_step+1] + bayer[bayer_step+3]
            v_uint16x8 second_line13 = v_add(second_line1, second_line3);
            // (bayer[bayer_step+1] + bayer[bayer_step+3]) * 2
            v_int16x8 b1 = v_reinterpret_as_s16(v_shl(second_line13, 1));

            // bayer[1]
            v_uint16x8 first_line1 = v_shr<8>(first_line);
            // bayer[bayer_step]
            v_uint16x8 second_line0 = v_and(second_line, v255);
            // bayer[bayer_step+2]
            v_uint16x8 second_line2 = v_rotate_right<1>(second_line0);
            // bayer[bayer_step] + bayer[bayer_step+2]
            v_uint16x8 second_line02 = v_add(second_line0, second_line2);
            // bayer[bayer_step*2+1]
            v_uint16x8 third_line1 = v_shr<8>(third_line);
            // bayer[1] + bayer[bayer_step*2+1]
            v_uint16x8 first_third_line1 = v_add(first_line1, third_line1);
            // bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1]
            v_int16x8 g0 = v_reinterpret_as_s16(v_add(first_third_line1, second_line02));
            // bayer[bayer_step+2] * 4
            v_int16x8 g1 = v_reinterpret_as_s16(v_shl<2>(second_line2));

            v_int16x8 rb0;
            v_int16x8 rb1;
            v_int16x8 rb2;
            v_int16x8 rb3;
            v_zip(r0, b0, rb0, rb1);
            v_zip(r1, b1, rb2, rb3);

            v_int16x8 gd0;
            v_int16x8 gd1;
            v_int16x8 gd2;
            v_int16x8 gd3;
            v_zip(g0, v_descale, gd0, gd1);
            v_zip(g1, v_descale, gd2, gd3);

            v_int32x4 gray_even0 = v_shr<16>(v_add(v_dotprod(rb0, cxrb), v_dotprod(gd0, cxg2)));
            v_int32x4 gray_even1 = v_shr<16>(v_add(v_dotprod(rb1, cxrb), v_dotprod(gd1, cxg2)));
            v_int32x4 gray_odd0 = v_shr<16>(v_add(v_dotprod(rb2, cxrb), v_dotprod(gd2, cxg2)));
            v_int32x4 gray_odd1 = v_shr<16>(v_add(v_dotprod(rb3, cxrb), v_dotprod(gd3, cxg2)));

            v_int16x8 gray_even = v_pack(gray_even0, gray_even1);
            v_int16x8 gray_odd = v_pack(gray_odd0, gray_odd1);

            v_int16x8 gray_d0;
            v_int16x8 gray_d1;
            v_zip(gray_even, gray_odd, gray_d0, gray_d1);

            v_uint8x16 gray = v_pack(v_reinterpret_as_u16(gray_d0), v_reinterpret_as_u16(gray_d1));

            v_store(dst, gray);
        }
#endif

        return static_cast<int>(bayer - (bayer_end - width));
    }

    int bayer2RGB(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue) const
    {
        /*
         B G B G | B G B G | B G B G | B G B G
         G R G R | G R G R | G R G R | G R G R
         B G B G | B G B G | B G B G | B G B G
         */

#if CV_NEON
        uint16x8_t masklo = vdupq_n_u16(255);
        uint8x16x3_t pix;
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 42 )
        {
            uint16x8_t r0 = vld1q_u16((const ushort*)bayer);
            uint16x8_t r1 = vld1q_u16((const ushort*)(bayer + bayer_step));
            uint16x8_t r2 = vld1q_u16((const ushort*)(bayer + bayer_step*2));

            uint16x8_t b1 = vaddq_u16(vandq_u16(r0, masklo), vandq_u16(r2, masklo));
            uint16x8_t nextb1 = vextq_u16(b1, b1, 1);
            uint16x8_t b0 = vaddq_u16(b1, nextb1);
            // b0 b1 b2 ...
            uint8x8x2_t bb = vzip_u8(vrshrn_n_u16(b0, 2), vrshrn_n_u16(nextb1, 1));
            pix.val[1-blue] = vcombine_u8(bb.val[0], bb.val[1]);

            uint16x8_t g0 = vaddq_u16(vshrq_n_u16(r0, 8), vshrq_n_u16(r2, 8));
            uint16x8_t g1 = vandq_u16(r1, masklo);
            g0 = vaddq_u16(g0, vaddq_u16(g1, vextq_u16(g1, g1, 1)));
            g1 = vextq_u16(g1, g1, 1);
            // g0 g1 g2 ...
            uint8x8x2_t gg = vzip_u8(vrshrn_n_u16(g0, 2), vmovn_u16(g1));
            pix.val[1] = vcombine_u8(gg.val[0], gg.val[1]);

            r0 = vshrq_n_u16(r1, 8);
            r1 = vaddq_u16(r0, vextq_u16(r0, r0, 1));
            // r0 r1 r2 ...
            uint8x8x2_t rr = vzip_u8(vmovn_u16(r0), vrshrn_n_u16(r1, 1));
            pix.val[1+blue] = vcombine_u8(rr.val[0], rr.val[1]);

            vst3q_u8(dst-1, pix);
        }
#else
        v_uint16x8 delta1 = v_setall_u16(1), delta2 = v_setall_u16(2);
        v_uint16x8 mask = v_setall_u16(blue < 0 ? (ushort)(-1) : 0);
        v_uint16x8 masklo = v_setall_u16(0x00ff);
        v_uint8x16 z = v_setzero_u8();
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 42 )
        {
            v_uint16x8 r0 = v_reinterpret_as_u16(v_load(bayer));
            v_uint16x8 r1 = v_reinterpret_as_u16(v_load(bayer+bayer_step));
            v_uint16x8 r2 = v_reinterpret_as_u16(v_load(bayer+bayer_step*2));

            v_uint16x8 b1 = v_add(v_and(r0, masklo), v_and(r2, masklo));
            v_uint16x8 nextb1 = v_rotate_right<1>(b1);
            v_uint16x8 b0 = v_add(b1, nextb1);
            b1 = v_shr<1>(v_add(nextb1, delta1));
            b0 = v_shr<2>(v_add(b0, delta2));
            // b0 b2 ... b14 b1 b3 ... b15
            b0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1)));

            v_uint16x8 g0 = v_add(v_shr<8>(r0), v_shr<8>(r2));
            v_uint16x8 g1 = v_and(r1, masklo);
            g0 = v_add(g0, v_add(v_rotate_right<1>(g1), g1));
            g1 = v_rotate_right<1>(g1);
            g0 = v_shr<2>(v_add(g0, delta2));
            // g0 g2 ... g14 g1 g3 ... g15
            g0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(g0), v_reinterpret_as_s16(g1)));

            r0 = v_shr<8>(r1);
            r1 = v_add(v_rotate_right<1>(r0), r0);
            r1 = v_shr<1>(v_add(r1, delta1));
            // r0 r2 ... r14 r1 r3 ... r15
            r0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(r0), v_reinterpret_as_s16(r1)));

            b1 = v_and(v_xor(b0, r0), mask);
            b0 = v_xor(b0, b1);
            r0 = v_xor(r0, b1);

            // b1 g1 b3 g3 b5 g5...
            v_uint8x16 pack_lo, pack_hi;
            v_zip(v_reinterpret_as_u8(b0), v_reinterpret_as_u8(g0), pack_lo, pack_hi);
            b1 = v_reinterpret_as_u16(pack_hi);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = v_reinterpret_as_u16(pack_lo);

            // r1 0 r3 0 r5 0 ...
            v_zip(v_reinterpret_as_u8(r0), z, pack_lo, pack_hi);
            r1 = v_reinterpret_as_u16(pack_hi);
            // r0 0 r2 0 r4 0 ...
            r0 = v_reinterpret_as_u16(pack_lo);

            // 0 b0 g0 r0 0 b2 g2 r2 ...
            v_zip(b0, r0, g0, g1);
            g0 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g0)));
            // 0 b8 g8 r8 0 b10 g10 r10 ...
            g1 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g1)));

            // b1 g1 r1 0 b3 g3 r3 0 ...
            v_zip(b1, r1, r0, r1);
            // b9 g9 r9 0 b11 g11 r11 0 ...

            // 0 b0 g0 r0 b1 g1 r1 0 ...
            v_uint32x4 pack32_lo, pack32_hi;
            v_zip(v_reinterpret_as_u32(g0), v_reinterpret_as_u32(r0), pack32_lo, pack32_hi);
            b0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b4 g4 r4 b5 g5 r5 0 ...
            b1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            v_store_low(dst-1+0, v_reinterpret_as_u8(b0));
            v_store_high(dst-1+6*1, v_reinterpret_as_u8(b0));
            v_store_low(dst-1+6*2, v_reinterpret_as_u8(b1));
            v_store_high(dst-1+6*3, v_reinterpret_as_u8(b1));

            // 0 b8 g8 r8 b9 g9 r9 0 ...
            v_zip(v_reinterpret_as_u32(g1), v_reinterpret_as_u32(r1), pack32_lo, pack32_hi);
            g0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b12 g12 r12 b13 g13 r13 0 ...
            g1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            v_store_low(dst-1+6*4, v_reinterpret_as_u8(g0));
            v_store_high(dst-1+6*5, v_reinterpret_as_u8(g0));

            v_store_low(dst-1+6*6, v_reinterpret_as_u8(g1));
        }
#endif

        return (int)(bayer - (bayer_end - width));
    }

    int bayer2RGBA(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue, const uchar alpha) const
    {
        /*
         B G B G | B G B G | B G B G | B G B G
         G R G R | G R G R | G R G R | G R G R
         B G B G | B G B G | B G B G | B G B G
         */

#if CV_NEON
        uint16x8_t masklo = vdupq_n_u16(255);
        uint8x16x4_t pix;
        const uchar* bayer_end = bayer + width;
        pix.val[3] = vdupq_n_u8(alpha);

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 56 )
        {
            uint16x8_t r0 = vld1q_u16((const ushort*)bayer);
            uint16x8_t r1 = vld1q_u16((const ushort*)(bayer + bayer_step));
            uint16x8_t r2 = vld1q_u16((const ushort*)(bayer + bayer_step*2));

            uint16x8_t b1 = vaddq_u16(vandq_u16(r0, masklo), vandq_u16(r2, masklo));
            uint16x8_t nextb1 = vextq_u16(b1, b1, 1);
            uint16x8_t b0 = vaddq_u16(b1, nextb1);
            // b0 b1 b2 ...
            uint8x8x2_t bb = vzip_u8(vrshrn_n_u16(b0, 2), vrshrn_n_u16(nextb1, 1));
            pix.val[1-blue] = vcombine_u8(bb.val[0], bb.val[1]);

            uint16x8_t g0 = vaddq_u16(vshrq_n_u16(r0, 8), vshrq_n_u16(r2, 8));
            uint16x8_t g1 = vandq_u16(r1, masklo);
            g0 = vaddq_u16(g0, vaddq_u16(g1, vextq_u16(g1, g1, 1)));
            g1 = vextq_u16(g1, g1, 1);
            // g0 g1 g2 ...
            uint8x8x2_t gg = vzip_u8(vrshrn_n_u16(g0, 2), vmovn_u16(g1));
            pix.val[1] = vcombine_u8(gg.val[0], gg.val[1]);

            r0 = vshrq_n_u16(r1, 8);
            r1 = vaddq_u16(r0, vextq_u16(r0, r0, 1));
            // r0 r1 r2 ...
            uint8x8x2_t rr = vzip_u8(vmovn_u16(r0), vrshrn_n_u16(r1, 1));
            pix.val[1+blue] = vcombine_u8(rr.val[0], rr.val[1]);

            vst4q_u8(dst-1, pix);
        }
#else
        v_uint16x8 delta1 = v_setall_u16(1), delta2 = v_setall_u16(2);
        v_uint16x8 mask = v_setall_u16(blue < 0 ? (ushort)(-1) : 0);
        v_uint16x8 masklo = v_setall_u16(0x00ff);
        v_uint8x16 a = v_setall_u8(alpha);
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 56 )
        {
            v_uint16x8 r0 = v_reinterpret_as_u16(v_load(bayer));
            v_uint16x8 r1 = v_reinterpret_as_u16(v_load(bayer+bayer_step));
            v_uint16x8 r2 = v_reinterpret_as_u16(v_load(bayer+bayer_step*2));

            v_uint16x8 b1 = v_add(v_and(r0, masklo), v_and(r2, masklo));
            v_uint16x8 nextb1 = v_rotate_right<1>(b1);
            v_uint16x8 b0 = v_add(b1, nextb1);
            b1 = v_shr<1>(v_add(nextb1, delta1));
            b0 = v_shr<2>(v_add(b0, delta2));
            // b0 b2 ... b14 b1 b3 ... b15
            b0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1)));

            v_uint16x8 g0 = v_add(v_shr<8>(r0), v_shr<8>(r2));
            v_uint16x8 g1 = v_and(r1, masklo);
            g0 = v_add(g0, v_add(v_rotate_right<1>(g1), g1));
            g1 = v_rotate_right<1>(g1);
            g0 = v_shr<2>(v_add(g0, delta2));
            // g0 g2 ... g14 g1 g3 ... g15
            g0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(g0), v_reinterpret_as_s16(g1)));

            r0 = v_shr<8>(r1);
            r1 = v_add(v_rotate_right<1>(r0), r0);
            r1 = v_shr<1>(v_add(r1, delta1));
            // r0 r2 ... r14 r1 r3 ... r15
            r0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(r0), v_reinterpret_as_s16(r1)));

            b1 = v_and(v_xor(b0, r0), mask);
            b0 = v_xor(b0, b1);
            r0 = v_xor(r0, b1);

            // b1 g1 b3 g3 b5 g5...
            v_uint8x16 pack_lo, pack_hi;
            v_zip(v_reinterpret_as_u8(b0), v_reinterpret_as_u8(g0), pack_lo, pack_hi);
            b1 = v_reinterpret_as_u16(pack_hi);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = v_reinterpret_as_u16(pack_lo);

            // r1 a r3 a r5 a ...
            v_zip(v_reinterpret_as_u8(r0), a, pack_lo, pack_hi);
            r1 = v_reinterpret_as_u16(pack_hi);
            // r0 a r2 a r4 a ...
            r0 = v_reinterpret_as_u16(pack_lo);

            // a b0 g0 r0 a b2 g2 r2 ...
            v_zip(b0, r0, g0, g1);
            // a b8 g8 r8 a b10 g10 r10 ...

            // b1 g1 r1 a b3 g3 r3 a ...
            v_zip(b1, r1, r0, r1);
            // b9 g9 r9 a b11 g11 r11 a ...

            // a b0 g0 r0 b1 g1 r1 a ...
            v_uint32x4 pack32_lo, pack32_hi;
            v_zip(v_reinterpret_as_u32(g0), v_reinterpret_as_u32(r0), pack32_lo, pack32_hi);
            b0 = v_reinterpret_as_u16(pack32_lo);
            // a b4 g4 r4 b5 g5 r5 a ...
            b1 = v_reinterpret_as_u16(pack32_hi);

            v_store_low(dst-1+0, v_reinterpret_as_u8(b0));
            v_store_high(dst-1+8*1, v_reinterpret_as_u8(b0));
            v_store_low(dst-1+8*2, v_reinterpret_as_u8(b1));
            v_store_high(dst-1+8*3, v_reinterpret_as_u8(b1));

            // a b8 g8 r8 b9 g9 r9 a ...
            v_zip(v_reinterpret_as_u32(g1), v_reinterpret_as_u32(r1), pack32_lo, pack32_hi);
            g0 = v_reinterpret_as_u16(pack32_lo);
            // a b12 g12 r12 b13 g13 r13 a ...
            g1 = v_reinterpret_as_u16(pack32_hi);

            v_store_low(dst-1+8*4, v_reinterpret_as_u8(g0));
            v_store_high(dst-1+8*5, v_reinterpret_as_u8(g0));

            v_store_low(dst-1+8*6, v_reinterpret_as_u8(g1));
        }
#endif

        return (int)(bayer - (bayer_end - width));
    }

    int bayer2RGB_EA(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue) const
    {
        const uchar* bayer_end = bayer + width;
        v_uint16x8 masklow = v_setall_u16(0x00ff);
        v_uint16x8 delta1 = v_setall_u16(1), delta2 = v_setall_u16(2);
        v_uint16x8 full = v_setall_u16((ushort)(-1));
        v_uint8x16 z = v_setzero_u8();
        v_uint16x8 mask = v_setall_u16(blue > 0 ? (ushort)(-1) : 0);

        for ( ; bayer <= bayer_end - 18; bayer += 14, dst += 42)
        {
            /*
             B G B G | B G B G | B G B G | B G B G
             G R G R | G R G R | G R G R | G R G R
             B G B G | B G B G | B G B G | B G B G
             */

            v_uint16x8 r0 = v_reinterpret_as_u16(v_load(bayer));
            v_uint16x8 r1 = v_reinterpret_as_u16(v_load(bayer+bayer_step));
            v_uint16x8 r2 = v_reinterpret_as_u16(v_load(bayer+bayer_step*2));

            v_uint16x8 b1 = v_add(v_and(r0, masklow), v_and(r2, masklow));
            v_uint16x8 nextb1 = v_rotate_right<1>(b1);
            v_uint16x8 b0 = v_add(b1, nextb1);
            b1 = v_shr<1>(v_add(nextb1, delta1));
            b0 = v_shr<2>(v_add(b0, delta2));
            // b0 b2 ... b14 b1 b3 ... b15
            b0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1)));

            // vertical sum
            v_uint16x8 r0g = v_shr<8>(r0);
            v_uint16x8 r2g = v_shr<8>(r2);
            v_uint16x8 sumv = v_shr<1>(v_add(v_add(r0g, r2g), delta1));
            // horizontal sum
            v_uint16x8 g1 = v_and(r1, masklow);
            v_uint16x8 nextg1 = v_rotate_right<1>(g1);
            v_uint16x8 sumg = v_shr<1>(v_add(v_add(g1, nextg1), delta1));

            // gradients
            v_uint16x8 gradv = v_add(v_sub(r0g, r2g), v_sub(r2g, r0g));
            v_uint16x8 gradg = v_add(v_sub(nextg1, g1), v_sub(g1, nextg1));
            v_uint16x8 gmask = v_gt(gradg, gradv);
            v_uint16x8 g0 = v_add(v_and(gmask, sumv), v_and(sumg, v_xor(gmask, full)));
            // g0 g2 ... g14 g1 g3 ...
            g0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(g0), v_reinterpret_as_s16(nextg1)));

            r0 = v_shr<8>(r1);
            r1 = v_add(v_rotate_right<1>(r0), r0);
            r1 = v_shr<1>(v_add(r1, delta1));
            // r0 r2 ... r14 r1 r3 ... r15
            r0 = v_reinterpret_as_u16(v_pack_u(v_reinterpret_as_s16(r0), v_reinterpret_as_s16(r1)));

            b1 = v_and(v_xor(b0, r0), mask);
            b0 = v_xor(b0, b1);
            r0 = v_xor(r0, b1);

            // b1 g1 b3 g3 b5 g5...
            v_uint8x16 pack_lo, pack_hi;
            v_zip(v_reinterpret_as_u8(b0), v_reinterpret_as_u8(g0), pack_lo, pack_hi);
            b1 = v_reinterpret_as_u16(pack_hi);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = v_reinterpret_as_u16(pack_lo);

            // r1 0 r3 0 r5 0 ...
            v_zip(v_reinterpret_as_u8(r0), z, pack_lo, pack_hi);
            r1 = v_reinterpret_as_u16(pack_hi);
            // r0 0 r2 0 r4 0 ...
            r0 = v_reinterpret_as_u16(pack_lo);

            // 0 b0 g0 r0 0 b2 g2 r2 ...
            v_zip(b0, r0, g0, g1);
            g0 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g0)));
            // 0 b8 g8 r8 0 b10 g10 r10 ...
            g1 = v_reinterpret_as_u16(v_rotate_left<1>(v_reinterpret_as_u8(g1)));

            // b1 g1 r1 0 b3 g3 r3 0 ...
            v_zip(b1, r1, r0, r1);
            // b9 g9 r9 0 b11 g11 r11 0 ...

            // 0 b0 g0 r0 b1 g1 r1 0 ...
            v_uint32x4 pack32_lo, pack32_hi;
            v_zip(v_reinterpret_as_u32(g0), v_reinterpret_as_u32(r0), pack32_lo, pack32_hi);
            b0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b4 g4 r4 b5 g5 r5 0 ...
            b1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            v_store_low(dst+0, v_reinterpret_as_u8(b0));
            v_store_high(dst+6*1, v_reinterpret_as_u8(b0));
            v_store_low(dst+6*2, v_reinterpret_as_u8(b1));
            v_store_high(dst+6*3, v_reinterpret_as_u8(b1));

            // 0 b8 g8 r8 b9 g9 r9 0 ...
            v_zip(v_reinterpret_as_u32(g1), v_reinterpret_as_u32(r1), pack32_lo, pack32_hi);
            g0 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_lo)));
            // 0 b12 g12 r12 b13 g13 r13 0 ...
            g1 = v_reinterpret_as_u16(v_rotate_right<1>(v_reinterpret_as_u8(pack32_hi)));

            v_store_low(dst+6*4, v_reinterpret_as_u8(g0));
            v_store_high(dst+6*5, v_reinterpret_as_u8(g0));

            v_store_low(dst+6*6, v_reinterpret_as_u8(g1));
        }

        return int(bayer - (bayer_end - width));
    }
};
#else
typedef SIMDBayerStubInterpolator_<uchar> SIMDBayerInterpolator_8u;
#endif


template<typename T, class SIMDInterpolator>
class Bayer2Gray_Invoker :
    public ParallelLoopBody
{
public:
    Bayer2Gray_Invoker(const Mat& _srcmat, Mat& _dstmat, int _start_with_green,
        const Size& _size, int _bcoeff, int _rcoeff) :
        ParallelLoopBody(), srcmat(_srcmat), dstmat(_dstmat), Start_with_green(_start_with_green),
        size(_size), Bcoeff(_bcoeff), Rcoeff(_rcoeff)
    {
    }

    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        SIMDInterpolator vecOp;
        const unsigned G2Y = 9617;
        const int SHIFT = 14;

        const T* bayer0 = srcmat.ptr<T>();
        int bayer_step = (int)(srcmat.step/sizeof(T));
        T* dst0 = (T*)dstmat.data;
        int dst_step = (int)(dstmat.step/sizeof(T));
        int bcoeff = Bcoeff, rcoeff = Rcoeff;
        int start_with_green = Start_with_green;

        dst0 += dst_step + 1;

        if (range.start % 2)
        {
            std::swap(bcoeff, rcoeff);
            start_with_green = !start_with_green;
        }

        bayer0 += range.start * bayer_step;
        dst0 += range.start * dst_step;

        for(int i = range.start ; i < range.end; ++i, bayer0 += bayer_step, dst0 += dst_step )
        {
            unsigned t0, t1, t2;
            const T* bayer = bayer0;
            T* dst = dst0;
            const T* bayer_end = bayer + size.width;

            if( size.width <= 0 )
            {
                dst[-1] = dst[size.width] = 0;
                continue;
            }

            if( start_with_green )
            {
                t0 = (bayer[1] + bayer[bayer_step*2+1])*rcoeff;
                t1 = (bayer[bayer_step] + bayer[bayer_step+2])*bcoeff;
                t2 = bayer[bayer_step+1]*(2*G2Y);

                dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+1);
                bayer++;
                dst++;
            }

            int delta = vecOp.bayer2Gray(bayer, bayer_step, dst, size.width, bcoeff, G2Y, rcoeff);
            bayer += delta;
            dst += delta;

            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 2 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2])*rcoeff;
                t1 = (bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1])*G2Y;
                t2 = bayer[bayer_step+1]*(4*bcoeff);
                dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+2);

                t0 = (bayer[2] + bayer[bayer_step*2+2])*rcoeff;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3])*bcoeff;
                t2 = bayer[bayer_step+2]*(2*G2Y);
                dst[1] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+1);
            }

            if( bayer < bayer_end )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2])*rcoeff;
                t1 = (bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1])*G2Y;
                t2 = bayer[bayer_step+1]*(4*bcoeff);
                dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+2);
                bayer++;
                dst++;
            }

            dst0[-1] = dst0[0];
            dst0[size.width] = dst0[size.width-1];

            std::swap(bcoeff, rcoeff);
            start_with_green = !start_with_green;
        }
    }

private:
    Mat srcmat;
    Mat dstmat;
    int Start_with_green;
    Size size;
    int Bcoeff, Rcoeff;
};

template<typename T, typename SIMDInterpolator>
static void Bayer2Gray_( const Mat& srcmat, Mat& dstmat, int code )
{
    const int R2Y = 4899;
    const int B2Y = 1868;

    Size size = srcmat.size();
    int bcoeff = B2Y, rcoeff = R2Y;
    int start_with_green = code == COLOR_BayerGB2GRAY || code == COLOR_BayerGR2GRAY;

    if( code != COLOR_BayerBG2GRAY && code != COLOR_BayerGB2GRAY )
    {
        std::swap(bcoeff, rcoeff);
    }
    size.height -= 2;
    size.width -= 2;

    if (size.height > 0)
    {
        Range range(0, size.height);
        Bayer2Gray_Invoker<T, SIMDInterpolator> invoker(srcmat, dstmat,
            start_with_green, size, bcoeff, rcoeff);
        parallel_for_(range, invoker, dstmat.total()/static_cast<double>(1<<16));
    }

    size = dstmat.size();
    T* dst0 = dstmat.ptr<T>();
    int dst_step = (int)(dstmat.step/sizeof(T));
    if( size.height > 2 )
        for( int i = 0; i < size.width; i++ )
        {
            dst0[i] = dst0[i + dst_step];
            dst0[i + (size.height-1)*dst_step] = dst0[i + (size.height-2)*dst_step];
        }
    else
        for( int i = 0; i < size.width; i++ )
            dst0[i] = dst0[i + (size.height-1)*dst_step] = 0;
}

template <typename T>
struct Alpha
{
    static T value() { return std::numeric_limits<T>::max(); }
};

template <>
struct Alpha<float>
{
    static float value() { return 1.0f; }
};

template <typename T, typename SIMDInterpolator>
class Bayer2RGB_Invoker :
    public ParallelLoopBody
{
public:
    Bayer2RGB_Invoker(const Mat& _srcmat, Mat& _dstmat, int _start_with_green, int _blue, const Size& _size) :
        ParallelLoopBody(),
        srcmat(_srcmat), dstmat(_dstmat), Start_with_green(_start_with_green), Blue(_blue), size(_size)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        SIMDInterpolator vecOp;
        T alpha = Alpha<T>::value();
        int dcn = dstmat.channels();
        int dcn2 = dcn << 1;

        int bayer_step = (int)(srcmat.step/sizeof(T));
        const T* bayer0 = srcmat.ptr<T>() + bayer_step * range.start;

        int dst_step = (int)(dstmat.step/sizeof(T));
        T* dst0 = reinterpret_cast<T*>(dstmat.data) + (range.start + 1) * dst_step + dcn + 1;

        int blue = Blue, start_with_green = Start_with_green;
        if (range.start % 2)
        {
            blue = -blue;
            start_with_green = !start_with_green;
        }

        for (int i = range.start; i < range.end; bayer0 += bayer_step, dst0 += dst_step, ++i )
        {
            int t0, t1;
            const T* bayer = bayer0;
            T* dst = dst0;
            const T* bayer_end = bayer + size.width;

            // in case of when size.width <= 2
            if( size.width <= 0 )
            {
                if (dcn == 3)
                {
                    dst[-4] = dst[-3] = dst[-2] = dst[size.width*dcn-1] =
                    dst[size.width*dcn] = dst[size.width*dcn+1] = 0;
                }
                else
                {
                    dst[-5] = dst[-4] = dst[-3] = dst[size.width*dcn-1] =
                    dst[size.width*dcn] = dst[size.width*dcn+1] = 0;
                    dst[-2] = dst[size.width*dcn+2] = alpha;
                }
                continue;
            }

            if( start_with_green )
            {
                t0 = (bayer[1] + bayer[bayer_step*2+1] + 1) >> 1;
                t1 = (bayer[bayer_step] + bayer[bayer_step+2] + 1) >> 1;

                dst[-blue] = (T)t0;
                dst[0] = bayer[bayer_step+1];
                dst[blue] = (T)t1;
                if (dcn == 4)
                    dst[2] = alpha; // alpha channel

                bayer++;
                dst += dcn;
            }

            // simd optimization only for dcn == 3
            int delta = dcn == 4 ?
                vecOp.bayer2RGBA(bayer, bayer_step, dst, size.width, blue, alpha) :
                vecOp.bayer2RGB(bayer, bayer_step, dst, size.width, blue);
            bayer += delta;
            dst += delta*dcn;

            if (dcn == 3) // Bayer to BGR
            {
                if( blue > 0 )
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[-1] = (T)t0;
                        dst[0] = (T)t1;
                        dst[1] = bayer[bayer_step+1];

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[2] = (T)t0;
                        dst[3] = bayer[bayer_step+2];
                        dst[4] = (T)t1;
                    }
                }
                else
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[1] = (T)t0;
                        dst[0] = (T)t1;
                        dst[-1] = bayer[bayer_step+1];

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[4] = (T)t0;
                        dst[3] = bayer[bayer_step+2];
                        dst[2] = (T)t1;
                    }
                }
            }
            else // Bayer to BGRA
            {
                // if current row does not contain Blue pixels
                if( blue > 0 )
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[-1] = (T)t0;
                        dst[0] = (T)t1;
                        dst[1] = bayer[bayer_step+1];
                        dst[2] = alpha; // alpha channel

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[3] = (T)t0;
                        dst[4] = bayer[bayer_step+2];
                        dst[5] = (T)t1;
                        dst[6] = alpha; // alpha channel
                    }
                }
                else // if current row contains Blue pixels
                {
                    for( ; bayer <= bayer_end - 2; bayer += 2, dst += dcn2 )
                    {
                        t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                              bayer[bayer_step*2+2] + 2) >> 2;
                        t1 = (bayer[1] + bayer[bayer_step] +
                              bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                        dst[-1] = bayer[bayer_step+1];
                        dst[0] = (T)t1;
                        dst[1] = (T)t0;
                        dst[2] = alpha; // alpha channel

                        t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                        t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                        dst[3] = (T)t1;
                        dst[4] = bayer[bayer_step+2];
                        dst[5] = (T)t0;
                        dst[6] = alpha; // alpha channel
                    }
                }
            }

            // if skip one pixel at the end of row
            if( bayer < bayer_end )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[-blue] = (T)t0;
                dst[0] = (T)t1;
                dst[blue] = bayer[bayer_step+1];
                if (dcn == 4)
                    dst[2] = alpha; // alpha channel
                bayer++;
                dst += dcn;
            }

            // fill the last and the first pixels of row accordingly
            if (dcn == 3)
            {
                dst0[-4] = dst0[-1];
                dst0[-3] = dst0[0];
                dst0[-2] = dst0[1];
                dst0[size.width*dcn-1] = dst0[size.width*dcn-4];
                dst0[size.width*dcn] = dst0[size.width*dcn-3];
                dst0[size.width*dcn+1] = dst0[size.width*dcn-2];
            }
            else
            {
                dst0[-5] = dst0[-1];
                dst0[-4] = dst0[0];
                dst0[-3] = dst0[1];
                dst0[-2] = dst0[2]; // alpha channel
                dst0[size.width*dcn-1] = dst0[size.width*dcn-5];
                dst0[size.width*dcn] = dst0[size.width*dcn-4];
                dst0[size.width*dcn+1] = dst0[size.width*dcn-3];
                dst0[size.width*dcn+2] = dst0[size.width*dcn-2]; // alpha channel
            }

            blue = -blue;
            start_with_green = !start_with_green;
        }
    }

private:
    Mat srcmat;
    Mat dstmat;
    int Start_with_green, Blue;
    Size size;
};

template<typename T, class SIMDInterpolator>
static void Bayer2RGB_( const Mat& srcmat, Mat& dstmat, int code )
{
    int dst_step = (int)(dstmat.step/sizeof(T));
    Size size = srcmat.size();
    int blue = (code == COLOR_BayerBG2BGR || code == COLOR_BayerGB2BGR ||
                code == COLOR_BayerBG2BGRA || code == COLOR_BayerGB2BGRA ) ? -1 : 1;
    int start_with_green = (code == COLOR_BayerGB2BGR || code == COLOR_BayerGR2BGR ||
                            code == COLOR_BayerGB2BGRA || code == COLOR_BayerGR2BGRA);

    int dcn = dstmat.channels();
    size.height -= 2;
    size.width -= 2;

    if (size.height > 0)
    {
        Range range(0, size.height);
        Bayer2RGB_Invoker<T, SIMDInterpolator> invoker(srcmat, dstmat, start_with_green, blue, size);
        parallel_for_(range, invoker, dstmat.total()/static_cast<double>(1<<16));
    }

    // filling the first and the last rows
    size = dstmat.size();
    T* dst0 = dstmat.ptr<T>();
    if( size.height > 2 )
        for( int i = 0; i < size.width*dcn; i++ )
        {
            dst0[i] = dst0[i + dst_step];
            dst0[i + (size.height-1)*dst_step] = dst0[i + (size.height-2)*dst_step];
        }
    else
        for( int i = 0; i < size.width*dcn; i++ )
            dst0[i] = dst0[i + (size.height-1)*dst_step] = 0;
}


/////////////////// Demosaicing using Variable Number of Gradients ///////////////////////

static void Bayer2RGB_VNG_8u( const Mat& srcmat, Mat& dstmat, int code )
{
    const uchar* bayer = srcmat.ptr();
    int bstep = (int)srcmat.step;
    uchar* dst = dstmat.ptr();
    int dststep = (int)dstmat.step;
    Size size = srcmat.size();

    int blueIdx = code == COLOR_BayerBG2BGR_VNG || code == COLOR_BayerGB2BGR_VNG ? 0 : 2;
    bool greenCell0 = code != COLOR_BayerBG2BGR_VNG && code != COLOR_BayerRG2BGR_VNG;

    // for too small images use the simple interpolation algorithm
    if( MIN(size.width, size.height) < 8 )
    {
        Bayer2RGB_<uchar, SIMDBayerInterpolator_8u>( srcmat, dstmat, code );
        return;
    }

    const int brows = 3, bcn = 7;
    int N = size.width, N2 = N*2, N3 = N*3, N4 = N*4, N5 = N*5, N6 = N*6, N7 = N*7;
    int i, bufstep = N7*bcn;
    cv::AutoBuffer<ushort> _buf(bufstep*brows);
    ushort* buf = _buf.data();

    bayer += bstep*2;

    for( int y = 2; y < size.height - 4; y++ )
    {
        uchar* dstrow = dst + dststep*y + 6;
        const uchar* srow;

        for( int dy = (y == 2 ? -1 : 1); dy <= 1; dy++ )
        {
            ushort* brow = buf + ((y + dy - 1)%brows)*bufstep + 1;
            srow = bayer + (y+dy)*bstep + 1;

            for( i = 0; i < bcn; i++ )
                brow[N*i-1] = brow[(N-2) + N*i] = 0;

            i = 1;
#if CV_SIMD256
            for( ; i <= N-17; i += 16, srow += 16, brow += 16)
            {
                v_uint16x16 s1, s2, s3, s4, s6, s7, s8, s9;

                s1 = v256_load_expand(srow-1-bstep);
                s2 = v256_load_expand(srow-bstep);
                s3 = v256_load_expand(srow+1-bstep);

                s4 = v256_load_expand(srow-1);
                s6 = v256_load_expand(srow+1);

                s7 = v256_load_expand(srow-1+bstep);
                s8 = v256_load_expand(srow+bstep);
                s9 = v256_load_expand(srow+1+bstep);

                v_uint16x16 b0, b1, b2, b3, b4, b5, b6;

                b0 = v_add(v_add(v_shl<1>(v_absdiff(s2, s8)), v_absdiff(s1, s7)), v_absdiff(s3, s9));
                b1 = v_add(v_add(v_shl<1>(v_absdiff(s4, s6)), v_absdiff(s1, s3)), v_absdiff(s7, s9));
                b2 = v_shl<1>(v_absdiff(s3, s7));
                b3 = v_shl<1>(v_absdiff(s1, s9));

                v_store(brow, b0);
                v_store(brow + N, b1);
                v_store(brow + N2, b2);
                v_store(brow + N3, b3);

                b4 = v_add(v_add(b2, v_absdiff(s2, s4)), v_absdiff(s6, s8));
                b5 = v_add(v_add(b3, v_absdiff(s2, s6)), v_absdiff(s4, s8));
                b6 = v_shr<1>(v_add(v_add(v_add(s2, s4), s6), s8));

                v_store(brow + N4, b4);
                v_store(brow + N5, b5);
                v_store(brow + N6, b6);
            }
#elif CV_SIMD128
            for( ; i <= N-9; i += 8, srow += 8, brow += 8 )
            {
                v_uint16x8 s1, s2, s3, s4, s6, s7, s8, s9;

                s1 = v_load_expand(srow-1-bstep);
                s2 = v_load_expand(srow-bstep);
                s3 = v_load_expand(srow+1-bstep);

                s4 = v_load_expand(srow-1);
                s6 = v_load_expand(srow+1);

                s7 = v_load_expand(srow-1+bstep);
                s8 = v_load_expand(srow+bstep);
                s9 = v_load_expand(srow+1+bstep);

                v_uint16x8 b0, b1, b2, b3, b4, b5, b6;

                b0 = v_add(v_add(v_shl<1>(v_absdiff(s2, s8)), v_absdiff(s1, s7)), v_absdiff(s3, s9));
                b1 = v_add(v_add(v_shl<1>(v_absdiff(s4, s6)), v_absdiff(s1, s3)), v_absdiff(s7, s9));
                b2 = v_shl<1>(v_absdiff(s3, s7));
                b3 = v_shl<1>(v_absdiff(s1, s9));

                v_store(brow, b0);
                v_store(brow + N, b1);
                v_store(brow + N2, b2);
                v_store(brow + N3, b3);

                b4 = v_add(v_add(b2, v_absdiff(s2, s4)), v_absdiff(s6, s8));
                b5 = v_add(v_add(b3, v_absdiff(s2, s6)), v_absdiff(s4, s8));
                b6 = v_shr<1>(v_add(v_add(v_add(s2, s4), s6), s8));

                v_store(brow + N4, b4);
                v_store(brow + N5, b5);
                v_store(brow + N6, b6);
            }
#endif

            for( ; i < N-1; i++, srow++, brow++ )
            {
                brow[0] = (ushort)(std::abs(srow[-1-bstep] - srow[-1+bstep]) +
                                   std::abs(srow[-bstep] - srow[+bstep])*2 +
                                   std::abs(srow[1-bstep] - srow[1+bstep]));
                brow[N] = (ushort)(std::abs(srow[-1-bstep] - srow[1-bstep]) +
                                   std::abs(srow[-1] - srow[1])*2 +
                                   std::abs(srow[-1+bstep] - srow[1+bstep]));
                brow[N2] = (ushort)(std::abs(srow[+1-bstep] - srow[-1+bstep])*2);
                brow[N3] = (ushort)(std::abs(srow[-1-bstep] - srow[1+bstep])*2);
                brow[N4] = (ushort)(brow[N2] + std::abs(srow[-bstep] - srow[-1]) +
                                    std::abs(srow[+bstep] - srow[1]));
                brow[N5] = (ushort)(brow[N3] + std::abs(srow[-bstep] - srow[1]) +
                                    std::abs(srow[+bstep] - srow[-1]));
                brow[N6] = (ushort)((srow[-bstep] + srow[-1] + srow[1] + srow[+bstep])>>1);
            }
        }

        const ushort* brow0 = buf + ((y - 2) % brows)*bufstep + 2;
        const ushort* brow1 = buf + ((y - 1) % brows)*bufstep + 2;
        const ushort* brow2 = buf + (y % brows)*bufstep + 2;
        static const float scale[] = { 0.f, 0.5f, 0.25f, 0.1666666666667f, 0.125f, 0.1f, 0.08333333333f, 0.0714286f, 0.0625f };
        srow = bayer + y*bstep + 2;
        bool greenCell = greenCell0;

        i = 2;
#if CV_SIMD256 || CV_SIMD128
        int limit = greenCell ? std::min(3, N-2) : 2;
#else
        int limit = N - 2;
#endif

        do
        {
            for( ; i < limit; i++, srow++, brow0++, brow1++, brow2++, dstrow += 3 )
            {
                int gradN = brow0[0] + brow1[0];
                int gradS = brow1[0] + brow2[0];
                int gradW = brow1[N-1] + brow1[N];
                int gradE = brow1[N] + brow1[N+1];
                int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                int R, G, B;

                if( !greenCell )
                {
                    int gradNE = brow0[N4+1] + brow1[N4];
                    int gradSW = brow1[N4] + brow2[N4-1];
                    int gradNW = brow0[N5-1] + brow1[N5];
                    int gradSE = brow1[N5] + brow2[N5+1];

                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + MAX(maxGrad/2, 1);

                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2] + srow[0];
                        Gs += srow[-bstep]*2;
                        Bs += srow[-bstep-1] + srow[-bstep+1];
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2] + srow[0];
                        Gs += srow[bstep]*2;
                        Bs += srow[bstep-1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-2] + srow[0];
                        Gs += srow[-1]*2;
                        Bs += srow[-bstep-1] + srow[bstep-1];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[2] + srow[0];
                        Gs += srow[1]*2;
                        Bs += srow[-bstep+1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+2] + srow[0];
                        Gs += brow0[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-2] + srow[0];
                        Gs += brow2[N6-1];
                        Bs += srow[bstep-1]*2;
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-2] + srow[0];
                        Gs += brow0[N6-1];
                        Bs += srow[-bstep-1]*2;
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+2] + srow[0];
                        Gs += brow2[N6+1];
                        Bs += srow[bstep+1]*2;
                        ng++;
                    }
                    R = srow[0];
                    G = R + cvRound((Gs - Rs)*scale[ng]);
                    B = R + cvRound((Bs - Rs)*scale[ng]);
                }
                else
                {
                    int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                    int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                    int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                    int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];

                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + MAX(maxGrad/2, 1);

                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-bstep*2+1];
                        Gs += srow[-bstep*2] + srow[0];
                        Bs += srow[-bstep]*2;
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2-1] + srow[bstep*2+1];
                        Gs += srow[bstep*2] + srow[0];
                        Bs += srow[bstep]*2;
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-1]*2;
                        Gs += srow[-2] + srow[0];
                        Bs += srow[-bstep-2]+srow[bstep-2];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[1]*2;
                        Gs += srow[2] + srow[0];
                        Bs += srow[-bstep+2]+srow[bstep+2];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+1] + srow[1];
                        Gs += srow[-bstep+1]*2;
                        Bs += srow[-bstep] + srow[-bstep+2];
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-1] + srow[-1];
                        Gs += srow[bstep-1]*2;
                        Bs += srow[bstep] + srow[bstep-2];
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-1];
                        Gs += srow[-bstep-1]*2;
                        Bs += srow[-bstep-2]+srow[-bstep];
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+1] + srow[1];
                        Gs += srow[bstep+1]*2;
                        Bs += srow[bstep+2]+srow[bstep];
                        ng++;
                    }
                    G = srow[0];
                    R = G + cvRound((Rs - Gs)*scale[ng]);
                    B = G + cvRound((Bs - Gs)*scale[ng]);
                }
                dstrow[blueIdx] = cv::saturate_cast<uchar>(B);
                dstrow[1] = cv::saturate_cast<uchar>(G);
                dstrow[blueIdx^2] = cv::saturate_cast<uchar>(R);
                greenCell = !greenCell;
            }

#if CV_SIMD256
            v_uint32x8 emask = v256_setall_u32(0x0000ffff), omask = v256_setall_u32(0xffff0000);
            v_uint16x16 one = v256_setall_u16(1), z = v256_setzero_u16();
            v_float32x8 _0_5 = v256_setall_f32(0.5f);

            //(aA_aA_aA_aA_aA_aA_aA_aA) * (bB_bB_bB_bB_bB_bB_bB_bB) => (bA_bA_bA_bA_bA_bA_bA_bA)
            #define v_merge_u16(a, b) (v_or((v_and((a), v_reinterpret_as_u16(emask))), (v_and((b), v_reinterpret_as_u16(omask)))))

            //(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) => (1f,2f,3f,4f,5f,6f,7f,8f)
            #define v_cvt_s16f32_lo(a)  v_cvt_f32(v_expand_low(v_reinterpret_as_s16(a)))

            //(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) => (9f,10f,11f,12f,13f,14f,15f,16f)
            #define v_cvt_s16f32_hi(a)  v_cvt_f32(v_expand_high(v_reinterpret_as_s16(a)))

            // process 16 pixels at once
            for( ; i <= N - 18; i += 16, srow += 16, brow0 += 16, brow1 += 16, brow2 += 16 )
            {
                //int gradN = brow0[0] + brow1[0];
                v_uint16x16 gradN = v_add(v256_load(brow0), v256_load(brow1));

                //int gradS = brow1[0] + brow2[0];
                v_uint16x16 gradS = v_add(v256_load(brow1), v256_load(brow2));

                //int gradW = brow1[N-1] + brow1[N];
                v_uint16x16 gradW = v_add(v256_load(brow1 + N - 1), v256_load(brow1 + N));

                //int gradE = brow1[N+1] + brow1[N];
                v_uint16x16 gradE = v_add(v256_load(brow1 + N + 1), v256_load(brow1 + N));

                //int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                //int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                v_uint16x16 minGrad = v_min(v_min(gradN, gradS), v_min(gradW, gradE));
                v_uint16x16 maxGrad = v_max(v_max(gradN, gradS), v_max(gradW, gradE));

                v_uint16x16 grad0, grad1;

                //int gradNE = brow0[N4+1] + brow1[N4];
                //int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                grad0 = v_add(v256_load(brow0 + N4 + 1), v256_load(brow1 + N4));
                grad1 = v_add(v_add(v_add(v256_load(brow0 + N2), v256_load(brow0 + N2 + 1)), v256_load(brow1 + N2)), v256_load(brow1 + N2 + 1));
                v_uint16x16 gradNE = v_merge_u16(grad0, grad1);

                //int gradSW = brow1[N4] + brow2[N4-1];
                //int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                grad0 = v_add(v256_load(brow2 + N4 - 1), v256_load(brow1 + N4));
                grad1 = v_add(v_add(v_add(v256_load(brow2 + N2), v256_load(brow2 + N2 - 1)), v256_load(brow1 + N2)), v256_load(brow1 + N2 - 1));
                v_uint16x16 gradSW = v_merge_u16(grad0, grad1);

                minGrad = v_min(v_min(minGrad, gradNE), gradSW);
                maxGrad = v_max(v_max(maxGrad, gradNE), gradSW);

                //int gradNW = brow0[N5-1] + brow1[N5];
                //int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                grad0 = v_add(v256_load(brow0 + N5 - 1), v256_load(brow1 + N5));
                grad1 = v_add(v_add(v_add(v256_load(brow0 + N3), v256_load(brow0 + N3 - 1)), v256_load(brow1 + N3)), v256_load(brow1 + N3 - 1));
                v_uint16x16 gradNW = v_merge_u16(grad0, grad1);

                //int gradSE = brow1[N5] + brow2[N5+1];
                //int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];
                grad0 = v_add(v256_load(brow2 + N5 + 1), v256_load(brow1 + N5));
                grad1 = v_add(v_add(v_add(v256_load(brow2 + N3), v256_load(brow2 + N3 + 1)), v256_load(brow1 + N3)), v256_load(brow1 + N3 + 1));
                v_uint16x16 gradSE = v_merge_u16(grad0, grad1);

                minGrad = v_min(v_min(minGrad, gradNW), gradSE);
                maxGrad = v_max(v_max(maxGrad, gradNW), gradSE);

                //int T = minGrad + maxGrad/2;
                v_uint16x16 T = v_add(v_max((v_shr<1>(maxGrad)), one), minGrad);

                v_uint16x16 RGs = z, GRs = z, Bs = z, ng = z;

                v_uint16x16 x0  = v256_load_expand(srow +0);
                v_uint16x16 x1  = v256_load_expand(srow -1 - bstep);
                v_uint16x16 x2  = v256_load_expand(srow -1 - bstep*2);
                v_uint16x16 x3  = v256_load_expand(srow    - bstep);
                v_uint16x16 x4  = v256_load_expand(srow +1 - bstep*2);
                v_uint16x16 x5  = v256_load_expand(srow +1 - bstep);
                v_uint16x16 x6  = v256_load_expand(srow +2 - bstep);
                v_uint16x16 x7  = v256_load_expand(srow +1);
                v_uint16x16 x8  = v256_load_expand(srow +2 + bstep);
                v_uint16x16 x9  = v256_load_expand(srow +1 + bstep);
                v_uint16x16 x10 = v256_load_expand(srow +1 + bstep*2);
                v_uint16x16 x11 = v256_load_expand(srow    + bstep);
                v_uint16x16 x12 = v256_load_expand(srow -1 + bstep*2);
                v_uint16x16 x13 = v256_load_expand(srow -1 + bstep);
                v_uint16x16 x14 = v256_load_expand(srow -2 + bstep);
                v_uint16x16 x15 = v256_load_expand(srow -1);
                v_uint16x16 x16 = v256_load_expand(srow -2 - bstep);

                v_uint16x16 t0, t1, mask;

                // gradN ***********************************************
                mask = (v_gt(T, gradN)); // mask = T>gradN
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradN)

                t0 = (v_shl<1>(x3));                                 // srow[-bstep]*2
                t1 = v_add(v256_load_expand(srow - bstep * 2), x0);  // srow[-bstep*2] + srow[0]

                // RGs += (srow[-bstep*2] + srow[0]) * (T>gradN)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += {srow[-bstep]*2; (srow[-bstep*2-1] + srow[-bstep*2+1])} * (T>gradN)
                GRs = v_add(GRs, (v_and(v_merge_u16(t0, v_add(x2, x4)), mask)));
                // Bs  += {(srow[-bstep-1]+srow[-bstep+1]); srow[-bstep]*2 } * (T>gradN)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x1, x5), t0), mask));

                // gradNE **********************************************
                mask = (v_gt(T, gradNE)); // mask = T>gradNE
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradNE)

                t0 = (v_shl<1>(x5));                                    // srow[-bstep+1]*2
                t1 = v_add(v256_load_expand(srow - bstep * 2 + 2), x0);   // srow[-bstep*2+2] + srow[0]

                // RGs += {(srow[-bstep*2+2] + srow[0]); srow[-bstep+1]*2} * (T>gradNE)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow0[N6+1]; (srow[-bstep*2+1] + srow[1])} * (T>gradNE)
                GRs = v_add(GRs, v_and(v_merge_u16(v256_load(brow0+N6+1), v_add(x4, x7)), mask));
                // Bs  += {srow[-bstep+1]*2; (srow[-bstep] + srow[-bstep+2])}  * (T>gradNE)
                Bs = v_add(Bs, v_and(v_merge_u16(t0, v_add(x3, x6)), mask));

                // gradE ***********************************************
                mask = (v_gt(T, gradE));  // mask = T>gradE
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradE)

                t0 = (v_shl<1>(x7));                         // srow[1]*2
                t1 = v_add(v256_load_expand(srow + 2), x0); // srow[2] + srow[0]

                // RGs += (srow[2] + srow[0]) * (T>gradE)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += (srow[1]*2) * (T>gradE)
                GRs = v_add(GRs, v_and(t0, mask));
                // Bs  += {(srow[-bstep+1]+srow[bstep+1]); (srow[-bstep+2]+srow[bstep+2])} * (T>gradE)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x5, x9), v_add(x6, x8)), mask));

                // gradSE **********************************************
                mask = (v_gt(T, gradSE));  // mask = T>gradSE
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradSE)

                t0 = (v_shl<1>(x9));                                 // srow[bstep+1]*2
                t1 = v_add(v256_load_expand(srow + bstep * 2 + 2), x0); // srow[bstep*2+2] + srow[0]

                // RGs += {(srow[bstep*2+2] + srow[0]); srow[bstep+1]*2} * (T>gradSE)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow2[N6+1]; (srow[1]+srow[bstep*2+1])} * (T>gradSE)
                GRs = v_add(GRs, v_and(v_merge_u16(v256_load(brow2+N6+1), v_add(x7, x10)), mask));
                // Bs  += {srow[bstep+1]*2; (srow[bstep+2]+srow[bstep])} * (T>gradSE)
                Bs = v_add(Bs, v_and(v_merge_u16((v_shl<1>(x9)), v_add(x8, x11)), mask));

                // gradS ***********************************************
                mask = (v_gt(T, gradS));  // mask = T>gradS
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradS)

                t0 = (v_shl<1>(x11));                             // srow[bstep]*2
                t1 = v_add(v256_load_expand(srow + bstep * 2), x0); // srow[bstep*2]+srow[0]

                // RGs += (srow[bstep*2]+srow[0]) * (T>gradS)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += {srow[bstep]*2; (srow[bstep*2+1]+srow[bstep*2-1])} * (T>gradS)
                GRs = v_add(GRs, v_and(v_merge_u16(t0, v_add(x10, x12)), mask));
                // Bs  += {(srow[bstep+1]+srow[bstep-1]); srow[bstep]*2} * (T>gradS)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x9, x13), t0), mask));

                // gradSW **********************************************
                mask = (v_gt(T, gradSW));  // mask = T>gradSW
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradSW)

                t0 = (v_shl<1>(x13));                                // srow[bstep-1]*2
                t1 = v_add(v256_load_expand(srow + bstep * 2 - 2), x0); // srow[bstep*2-2]+srow[0]

                // RGs += {(srow[bstep*2-2]+srow[0]); srow[bstep-1]*2} * (T>gradSW)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow2[N6-1]; (srow[bstep*2-1]+srow[-1])} * (T>gradSW)
                GRs = v_add(GRs, v_and(v_merge_u16(v256_load(brow2+N6-1), v_add(x12, x15)), mask));
                // Bs  += {srow[bstep-1]*2; (srow[bstep]+srow[bstep-2])} * (T>gradSW)
                Bs = v_add(Bs, v_and(v_merge_u16(t0, v_add(x11, x14)), mask));

                // gradW ***********************************************
                mask = (v_gt(T, gradW));  // mask = T>gradW
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradW)

                t0 = (v_shl<1>(x15));                         // srow[-1]*2
                t1 = v_add(v256_load_expand(srow - 2), x0); // srow[-2]+srow[0]

                // RGs += (srow[-2]+srow[0]) * (T>gradW)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += (srow[-1]*2) * (T>gradW)
                GRs = v_add(GRs, v_and(t0, mask));
                // Bs  += {(srow[-bstep-1]+srow[bstep-1]); (srow[bstep-2]+srow[-bstep-2])} * (T>gradW)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x1, x13), v_add(x14, x16)), mask));

                // gradNW **********************************************
                mask = (v_gt(T, gradNW));  // mask = T>gradNW
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradNW)

                t0 = (v_shl<1>(x1));                                 // srow[-bstep-1]*2
                t1 = v_add(v256_load_expand(srow - bstep * 2 - 2), x0); // srow[-bstep*2-2]+srow[0]

                // RGs += {(srow[-bstep*2-2]+srow[0]); srow[-bstep-1]*2} * (T>gradNW)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow0[N6-1]; (srow[-bstep*2-1]+srow[-1])} * (T>gradNW)
                GRs = v_add(GRs, v_and(v_merge_u16(v256_load(brow0+N6-1), v_add(x2, x15)), mask));
                // Bs  += {srow[-bstep-1]*2; (srow[-bstep]+srow[-bstep-2])} * (T>gradNW)
                Bs = v_add(Bs, v_and(v_merge_u16(v_shl<1>(x1), v_add(x3, x16)), mask));

                v_float32x8 ngf0 = v_div(_0_5, v_cvt_s16f32_lo(ng));
                v_float32x8 ngf1 = v_div(_0_5, v_cvt_s16f32_hi(ng));

                // now interpolate r, g & b
                t0 = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(GRs), v_reinterpret_as_s16(RGs)));
                t1 = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(Bs), v_reinterpret_as_s16(RGs)));

                t0 = v_reinterpret_as_u16(
                    v_add(v_reinterpret_as_s16(x0),
                        v_pack(
                            v_round(v_mul(v_cvt_s16f32_lo(t0), ngf0)),
                            v_round(v_mul(v_cvt_s16f32_hi(t0), ngf1)))));

                t1 = v_reinterpret_as_u16(
                    v_add(v_reinterpret_as_s16(x0),
                        v_pack(
                            v_round(v_mul(v_cvt_s16f32_lo(t1), ngf0)),
                            v_round(v_mul(v_cvt_s16f32_hi(t1), ngf1)))));

                x1 = v_merge_u16(x0, t0);
                x2 = v_merge_u16(t0, x0);

                uchar R[16], G[16], B[16];

                v_store_low(blueIdx ? B : R, v_pack_u(v_reinterpret_as_s16(x1), v_reinterpret_as_s16(z)));
                v_store_low(G, v_pack_u(v_reinterpret_as_s16(x2), v_reinterpret_as_s16(z)));
                v_store_low(blueIdx ? R : B, v_pack_u(v_reinterpret_as_s16(t1), v_reinterpret_as_s16(z)));

                for( int j = 0; j < 16; j++, dstrow += 3 )
                {
                    dstrow[0] = B[j]; dstrow[1] = G[j]; dstrow[2] = R[j];
                }
            }
#elif CV_SIMD128
            v_uint32x4 emask = v_setall_u32(0x0000ffff), omask = v_setall_u32(0xffff0000);
            v_uint16x8 one = v_setall_u16(1), z = v_setzero_u16();
            v_float32x4 _0_5 = v_setall_f32(0.5f);

            #define v_merge_u16(a, b) (v_or((v_and((a), v_reinterpret_as_u16(emask))), (v_and((b), v_reinterpret_as_u16(omask))))) //(aA_aA_aA_aA) * (bB_bB_bB_bB) => (bA_bA_bA_bA)
            #define v_cvt_s16f32_lo(a)  v_cvt_f32(v_expand_low(v_reinterpret_as_s16(a)))   //(1,2,3,4,5,6,7,8) => (1f,2f,3f,4f)
            #define v_cvt_s16f32_hi(a)  v_cvt_f32(v_expand_high(v_reinterpret_as_s16(a)))   //(1,2,3,4,5,6,7,8) => (5f,6f,7f,8f)

            // process 8 pixels at once
            for( ; i <= N - 10; i += 8, srow += 8, brow0 += 8, brow1 += 8, brow2 += 8 )
            {
                //int gradN = brow0[0] + brow1[0];
                v_uint16x8 gradN = v_add(v_load(brow0), v_load(brow1));

                //int gradS = brow1[0] + brow2[0];
                v_uint16x8 gradS = v_add(v_load(brow1), v_load(brow2));

                //int gradW = brow1[N-1] + brow1[N];
                v_uint16x8 gradW = v_add(v_load(brow1 + N - 1), v_load(brow1 + N));

                //int gradE = brow1[N+1] + brow1[N];
                v_uint16x8 gradE = v_add(v_load(brow1 + N + 1), v_load(brow1 + N));

                //int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                //int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                v_uint16x8 minGrad = v_min(v_min(gradN, gradS), v_min(gradW, gradE));
                v_uint16x8 maxGrad = v_max(v_max(gradN, gradS), v_max(gradW, gradE));

                v_uint16x8 grad0, grad1;

                //int gradNE = brow0[N4+1] + brow1[N4];
                //int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                grad0 = v_add(v_load(brow0 + N4 + 1), v_load(brow1 + N4));
                grad1 = v_add(v_add(v_add(v_load(brow0 + N2), v_load(brow0 + N2 + 1)), v_load(brow1 + N2)), v_load(brow1 + N2 + 1));
                v_uint16x8 gradNE = v_merge_u16(grad0, grad1);

                //int gradSW = brow1[N4] + brow2[N4-1];
                //int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                grad0 = v_add(v_load(brow2 + N4 - 1), v_load(brow1 + N4));
                grad1 = v_add(v_add(v_add(v_load(brow2 + N2), v_load(brow2 + N2 - 1)), v_load(brow1 + N2)), v_load(brow1 + N2 - 1));
                v_uint16x8 gradSW = v_merge_u16(grad0, grad1);

                minGrad = v_min(v_min(minGrad, gradNE), gradSW);
                maxGrad = v_max(v_max(maxGrad, gradNE), gradSW);

                //int gradNW = brow0[N5-1] + brow1[N5];
                //int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                grad0 = v_add(v_load(brow0 + N5 - 1), v_load(brow1 + N5));
                grad1 = v_add(v_add(v_add(v_load(brow0 + N3), v_load(brow0 + N3 - 1)), v_load(brow1 + N3)), v_load(brow1 + N3 - 1));
                v_uint16x8 gradNW = v_merge_u16(grad0, grad1);

                //int gradSE = brow1[N5] + brow2[N5+1];
                //int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];
                grad0 = v_add(v_load(brow2 + N5 + 1), v_load(brow1 + N5));
                grad1 = v_add(v_add(v_add(v_load(brow2 + N3), v_load(brow2 + N3 + 1)), v_load(brow1 + N3)), v_load(brow1 + N3 + 1));
                v_uint16x8 gradSE = v_merge_u16(grad0, grad1);

                minGrad = v_min(v_min(minGrad, gradNW), gradSE);
                maxGrad = v_max(v_max(maxGrad, gradNW), gradSE);

                //int T = minGrad + maxGrad/2;
                v_uint16x8 T = v_add(v_max((v_shr<1>(maxGrad)), one), minGrad);

                v_uint16x8 RGs = z, GRs = z, Bs = z, ng = z;

                v_uint16x8 x0  = v_load_expand(srow +0);
                v_uint16x8 x1  = v_load_expand(srow -1 - bstep);
                v_uint16x8 x2  = v_load_expand(srow -1 - bstep*2);
                v_uint16x8 x3  = v_load_expand(srow    - bstep);
                v_uint16x8 x4  = v_load_expand(srow +1 - bstep*2);
                v_uint16x8 x5  = v_load_expand(srow +1 - bstep);
                v_uint16x8 x6  = v_load_expand(srow +2 - bstep);
                v_uint16x8 x7  = v_load_expand(srow +1);
                v_uint16x8 x8  = v_load_expand(srow +2 + bstep);
                v_uint16x8 x9  = v_load_expand(srow +1 + bstep);
                v_uint16x8 x10 = v_load_expand(srow +1 + bstep*2);
                v_uint16x8 x11 = v_load_expand(srow    + bstep);
                v_uint16x8 x12 = v_load_expand(srow -1 + bstep*2);
                v_uint16x8 x13 = v_load_expand(srow -1 + bstep);
                v_uint16x8 x14 = v_load_expand(srow -2 + bstep);
                v_uint16x8 x15 = v_load_expand(srow -1);
                v_uint16x8 x16 = v_load_expand(srow -2 - bstep);

                v_uint16x8 t0, t1, mask;

                // gradN ***********************************************
                mask = (v_gt(T, gradN)); // mask = T>gradN
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradN)

                t0 = (v_shl<1>(x3));                                 // srow[-bstep]*2
                t1 = v_add(v_load_expand(srow - bstep * 2), x0);  // srow[-bstep*2] + srow[0]

                // RGs += (srow[-bstep*2] + srow[0]) * (T>gradN)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += {srow[-bstep]*2; (srow[-bstep*2-1] + srow[-bstep*2+1])} * (T>gradN)
                GRs = v_add(GRs, (v_and(v_merge_u16(t0, v_add(x2, x4)), mask)));
                // Bs  += {(srow[-bstep-1]+srow[-bstep+1]); srow[-bstep]*2 } * (T>gradN)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x1, x5), t0), mask));

                // gradNE **********************************************
                mask = (v_gt(T, gradNE)); // mask = T>gradNE
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradNE)

                t0 = (v_shl<1>(x5));                                    // srow[-bstep+1]*2
                t1 = v_add(v_load_expand(srow - bstep * 2 + 2), x0);   // srow[-bstep*2+2] + srow[0]

                // RGs += {(srow[-bstep*2+2] + srow[0]); srow[-bstep+1]*2} * (T>gradNE)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow0[N6+1]; (srow[-bstep*2+1] + srow[1])} * (T>gradNE)
                GRs = v_add(GRs, v_and(v_merge_u16(v_load(brow0+N6+1), v_add(x4, x7)), mask));
                // Bs  += {srow[-bstep+1]*2; (srow[-bstep] + srow[-bstep+2])}  * (T>gradNE)
                Bs = v_add(Bs, v_and(v_merge_u16(t0, v_add(x3, x6)), mask));

                // gradE ***********************************************
                mask = (v_gt(T, gradE));  // mask = T>gradE
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradE)

                t0 = (v_shl<1>(x7));                         // srow[1]*2
                t1 = v_add(v_load_expand(srow + 2), x0); // srow[2] + srow[0]

                // RGs += (srow[2] + srow[0]) * (T>gradE)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += (srow[1]*2) * (T>gradE)
                GRs = v_add(GRs, v_and(t0, mask));
                // Bs  += {(srow[-bstep+1]+srow[bstep+1]); (srow[-bstep+2]+srow[bstep+2])} * (T>gradE)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x5, x9), v_add(x6, x8)), mask));

                // gradSE **********************************************
                mask = (v_gt(T, gradSE));  // mask = T>gradSE
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradSE)

                t0 = (v_shl<1>(x9));                                 // srow[bstep+1]*2
                t1 = v_add(v_load_expand(srow + bstep * 2 + 2), x0); // srow[bstep*2+2] + srow[0]

                // RGs += {(srow[bstep*2+2] + srow[0]); srow[bstep+1]*2} * (T>gradSE)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow2[N6+1]; (srow[1]+srow[bstep*2+1])} * (T>gradSE)
                GRs = v_add(GRs, v_and(v_merge_u16(v_load(brow2+N6+1), v_add(x7, x10)), mask));
                // Bs  += {srow[bstep+1]*2; (srow[bstep+2]+srow[bstep])} * (T>gradSE)
                Bs = v_add(Bs, v_and(v_merge_u16((v_shl<1>(x9)), v_add(x8, x11)), mask));

                // gradS ***********************************************
                mask = (v_gt(T, gradS));  // mask = T>gradS
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradS)

                t0 = (v_shl<1>(x11));                             // srow[bstep]*2
                t1 = v_add(v_load_expand(srow + bstep * 2), x0); // srow[bstep*2]+srow[0]

                // RGs += (srow[bstep*2]+srow[0]) * (T>gradS)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += {srow[bstep]*2; (srow[bstep*2+1]+srow[bstep*2-1])} * (T>gradS)
                GRs = v_add(GRs, v_and(v_merge_u16(t0, v_add(x10, x12)), mask));
                // Bs  += {(srow[bstep+1]+srow[bstep-1]); srow[bstep]*2} * (T>gradS)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x9, x13), t0), mask));

                // gradSW **********************************************
                mask = (v_gt(T, gradSW));  // mask = T>gradSW
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradSW)

                t0 = (v_shl<1>(x13));                                // srow[bstep-1]*2
                t1 = v_add(v_load_expand(srow + bstep * 2 - 2), x0); // srow[bstep*2-2]+srow[0]

                // RGs += {(srow[bstep*2-2]+srow[0]); srow[bstep-1]*2} * (T>gradSW)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow2[N6-1]; (srow[bstep*2-1]+srow[-1])} * (T>gradSW)
                GRs = v_add(GRs, v_and(v_merge_u16(v_load(brow2+N6-1), v_add(x12, x15)), mask));
                // Bs  += {srow[bstep-1]*2; (srow[bstep]+srow[bstep-2])} * (T>gradSW)
                Bs = v_add(Bs, v_and(v_merge_u16(t0, v_add(x11, x14)), mask));

                // gradW ***********************************************
                mask = (v_gt(T, gradW));  // mask = T>gradW
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradW)

                t0 = (v_shl<1>(x15));                         // srow[-1]*2
                t1 = v_add(v_load_expand(srow - 2), x0); // srow[-2]+srow[0]

                // RGs += (srow[-2]+srow[0]) * (T>gradW)
                RGs = v_add(RGs, v_and(t1, mask));
                // GRs += (srow[-1]*2) * (T>gradW)
                GRs = v_add(GRs, v_and(t0, mask));
                // Bs  += {(srow[-bstep-1]+srow[bstep-1]); (srow[bstep-2]+srow[-bstep-2])} * (T>gradW)
                Bs = v_add(Bs, v_and(v_merge_u16(v_add(x1, x13), v_add(x14, x16)), mask));

                // gradNW **********************************************
                mask = (v_gt(T, gradNW));  // mask = T>gradNW
                ng = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(ng), v_reinterpret_as_s16(mask)));     // ng += (T>gradNW)

                t0 = (v_shl<1>(x1));                                 // srow[-bstep-1]*2
                t1 = v_add(v_load_expand(srow - bstep * 2 - 2), x0); // srow[-bstep*2-2]+srow[0]

                // RGs += {(srow[-bstep*2-2]+srow[0]); srow[-bstep-1]*2} * (T>gradNW)
                RGs = v_add(RGs, v_and(v_merge_u16(t1, t0), mask));
                // GRs += {brow0[N6-1]; (srow[-bstep*2-1]+srow[-1])} * (T>gradNW)
                GRs = v_add(GRs, v_and(v_merge_u16(v_load(brow0+N6-1), v_add(x2, x15)), mask));
                // Bs  += {srow[-bstep-1]*2; (srow[-bstep]+srow[-bstep-2])} * (T>gradNW)
                Bs = v_add(Bs, v_and(v_merge_u16(v_shl<1>(x1), v_add(x3, x16)), mask));

                v_float32x4 ngf0 = v_div(_0_5, v_cvt_s16f32_lo(ng));
                v_float32x4 ngf1 = v_div(_0_5, v_cvt_s16f32_hi(ng));

                // now interpolate r, g & b
                t0 = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(GRs), v_reinterpret_as_s16(RGs)));
                t1 = v_reinterpret_as_u16(v_sub(v_reinterpret_as_s16(Bs), v_reinterpret_as_s16(RGs)));

                t0 = v_reinterpret_as_u16(
                    v_add(v_reinterpret_as_s16(x0),
                        v_pack(
                            v_round(v_mul(v_cvt_s16f32_lo(t0), ngf0)),
                            v_round(v_mul(v_cvt_s16f32_hi(t0), ngf1)))));

                t1 = v_reinterpret_as_u16(
                    v_add(v_reinterpret_as_s16(x0),
                        v_pack(
                            v_round(v_mul(v_cvt_s16f32_lo(t1), ngf0)),
                            v_round(v_mul(v_cvt_s16f32_hi(t1), ngf1)))));

                x1 = v_merge_u16(x0, t0);
                x2 = v_merge_u16(t0, x0);

                uchar R[8], G[8], B[8];

                v_store_low(blueIdx ? B : R, v_pack_u(v_reinterpret_as_s16(x1), v_reinterpret_as_s16(z)));
                v_store_low(G, v_pack_u(v_reinterpret_as_s16(x2), v_reinterpret_as_s16(z)));
                v_store_low(blueIdx ? R : B, v_pack_u(v_reinterpret_as_s16(t1), v_reinterpret_as_s16(z)));

                for( int j = 0; j < 8; j++, dstrow += 3 )
                {
                    dstrow[0] = B[j]; dstrow[1] = G[j]; dstrow[2] = R[j];
                }
            }
#endif

            limit = N - 2;
        }
        while( i < N - 2 );

        for( i = 0; i < 6; i++ )
        {
            dst[dststep*y + 5 - i] = dst[dststep*y + 8 - i];
            dst[dststep*y + (N - 2)*3 + i] = dst[dststep*y + (N - 3)*3 + i];
        }

        greenCell0 = !greenCell0;
        blueIdx ^= 2;
    }

    for( i = 0; i < size.width*3; i++ )
    {
        dst[i] = dst[i + dststep] = dst[i + dststep*2];
        dst[i + dststep*(size.height-4)] =
        dst[i + dststep*(size.height-3)] =
        dst[i + dststep*(size.height-2)] =
        dst[i + dststep*(size.height-1)] = dst[i + dststep*(size.height-5)];
    }
}

//////////////////////////////// Edge-Aware Demosaicing //////////////////////////////////

template <typename T, typename SIMDInterpolator>
class Bayer2RGB_EdgeAware_T_Invoker :
    public cv::ParallelLoopBody
{
public:
    Bayer2RGB_EdgeAware_T_Invoker(const Mat& _src, Mat& _dst, const Size& _size,
        int _blue, int _start_with_green) :
        ParallelLoopBody(),
        src(_src), dst(_dst), size(_size), Blue(_blue), Start_with_green(_start_with_green)
    {
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        int dcn = dst.channels();
        int dcn2 = dcn<<1;
        int start_with_green = Start_with_green, blue = Blue;
        int sstep = int(src.step / src.elemSize1()), dstep = int(dst.step / dst.elemSize1());
        SIMDInterpolator vecOp;

        const T* S = src.ptr<T>(range.start + 1) + 1;
        T* D = reinterpret_cast<T*>(dst.data + (range.start + 1) * dst.step) + dcn;

        if (range.start % 2)
        {
            start_with_green ^= 1;
            blue ^= 1;
        }

        // to BGR
        for (int y = range.start; y < range.end; ++y)
        {
            int x = 1;
            if (start_with_green)
            {
                D[blue<<1] = (S[-sstep] + S[sstep] + 1) >> 1;
                D[1] = S[0];
                D[2-(blue<<1)] = (S[-1] + S[1] + 1) >> 1;
                D += dcn;
                ++S;
                ++x;
            }

            int delta = vecOp.bayer2RGB_EA(S - sstep - 1, sstep, D, size.width, blue);
            x += delta;
            S += delta;
            D += dcn * delta;

            if (blue)
                for (; x < size.width; x += 2, S += 2, D += dcn2)
                {
                    D[0] = S[0];
                    D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[sstep] - S[-sstep]) ? (S[sstep] + S[-sstep] + 1) : (S[-1] + S[1] + 1)) >> 1;
                    D[2] = (S[-sstep-1] + S[-sstep+1] + S[sstep-1] + S[sstep+1] + 2) >> 2;

                    D[3] = (S[0] + S[2] + 1) >> 1;
                    D[4] = S[1];
                    D[5] = (S[-sstep+1] + S[sstep+1] + 1) >> 1;
                }
            else
                for (; x < size.width; x += 2, S += 2, D += dcn2)
                {
                    D[0] = (S[-sstep-1] + S[-sstep+1] + S[sstep-1] + S[sstep+1] + 2) >> 2;
                    D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[sstep] - S[-sstep]) ? (S[sstep] + S[-sstep] + 1) : (S[-1] + S[1] + 1)) >> 1;
                    D[2] = S[0];

                    D[3] = (S[-sstep+1] + S[sstep+1] + 1) >> 1;
                    D[4] = S[1];
                    D[5] = (S[0] + S[2] + 1) >> 1;
                }

            if (x <= size.width)
            {
                D[blue<<1] = (S[-sstep-1] + S[-sstep+1] + S[sstep-1] + S[sstep+1] + 2) >> 2;
                D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[sstep] - S[-sstep]) ? (S[sstep] + S[-sstep] + 1) : (S[-1] + S[1] + 1)) >> 1;
                D[2-(blue<<1)] = S[0];
                D += dcn;
                ++S;
            }

            for (int i = 0; i < dcn; ++i)
            {
                D[i] = D[-dcn + i];
                D[-dstep+dcn+i] = D[-dstep+(dcn<<1)+i];
            }

            start_with_green ^= 1;
            blue ^= 1;
            S += 2;
            D += dcn2;
        }
    }

private:
    Mat src;
    Mat dst;
    Size size;
    int Blue, Start_with_green;
};

template <typename T, typename SIMDInterpolator>
static void Bayer2RGB_EdgeAware_T(const Mat& src, Mat& dst, int code)
{
    Size size = src.size();

    // for small sizes
    if (size.width <= 2 || size.height <= 2)
    {
        dst = Scalar::all(0);
        return;
    }

    size.width -= 2;
    size.height -= 2;

    int start_with_green = code == COLOR_BayerGB2BGR_EA || code == COLOR_BayerGR2BGR_EA ? 1 : 0;
    int blue = code == COLOR_BayerGB2BGR_EA || code == COLOR_BayerBG2BGR_EA ? 1 : 0;

    if (size.height > 0)
    {
        Bayer2RGB_EdgeAware_T_Invoker<T, SIMDInterpolator> invoker(src, dst, size, blue, start_with_green);
        Range range(0, size.height);
        parallel_for_(range, invoker, dst.total()/static_cast<double>(1<<16));
    }
    size = dst.size();
    size.width *= dst.channels();
    size_t dstep = dst.step / dst.elemSize1();
    T* firstRow = dst.ptr<T>();
    T* lastRow = dst.ptr<T>() + (size.height-1) * dstep;

    if (size.height > 2)
    {
        for (int x = 0; x < size.width; ++x)
        {
            firstRow[x] = (firstRow+dstep)[x];
            lastRow[x] = (lastRow-dstep)[x];
        }
    }
    else
        for (int x = 0; x < size.width; ++x)
            firstRow[x] = lastRow[x] = 0;
}

} // end namespace cv

//////////////////////////////////////////////////////////////////////////////////////////
//                           The main Demosaicing function                              //
//////////////////////////////////////////////////////////////////////////////////////////

void cv::demosaicing(InputArray _src, OutputArray _dst, int code, int dcn)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), dst;
    Size sz = src.size();
    int scn = src.channels(), depth = src.depth();

    CV_Assert(depth == CV_8U || depth == CV_16U);
    CV_Assert(!src.empty());

    switch (code)
    {
    case COLOR_BayerBG2GRAY: case COLOR_BayerGB2GRAY: case COLOR_BayerRG2GRAY: case COLOR_BayerGR2GRAY:
        if (dcn <= 0)
            dcn = 1;
        CV_Assert( scn == 1 && dcn == 1 );

        _dst.create(sz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getMat();

        if( depth == CV_8U )
            Bayer2Gray_<uchar, SIMDBayerInterpolator_8u>(src, dst, code);
        else if( depth == CV_16U )
            Bayer2Gray_<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst, code);
        else
            CV_Error(cv::Error::StsUnsupportedFormat, "Bayer->Gray demosaicing only supports 8u and 16u types");
        break;

    case COLOR_BayerBG2BGRA: case COLOR_BayerGB2BGRA: case COLOR_BayerRG2BGRA: case COLOR_BayerGR2BGRA:
        if (dcn <= 0)
          dcn = 4;
        /* fallthrough */
    case COLOR_BayerBG2BGR: case COLOR_BayerGB2BGR: case COLOR_BayerRG2BGR: case COLOR_BayerGR2BGR:
    case COLOR_BayerBG2BGR_VNG: case COLOR_BayerGB2BGR_VNG: case COLOR_BayerRG2BGR_VNG: case COLOR_BayerGR2BGR_VNG:
        {
            if (dcn <= 0)
                dcn = 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4) );

            _dst.create(sz, CV_MAKE_TYPE(depth, dcn));
            Mat dst_ = _dst.getMat();

            if( code == COLOR_BayerBG2BGR || code == COLOR_BayerBG2BGRA ||
                code == COLOR_BayerGB2BGR || code == COLOR_BayerGB2BGRA ||
                code == COLOR_BayerRG2BGR || code == COLOR_BayerRG2BGRA ||
                code == COLOR_BayerGR2BGR || code == COLOR_BayerGR2BGRA )
            {
                if( depth == CV_8U )
                    Bayer2RGB_<uchar, SIMDBayerInterpolator_8u>(src, dst_, code);
                else if( depth == CV_16U )
                    Bayer2RGB_<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst_, code);
                else
                    CV_Error(cv::Error::StsUnsupportedFormat, "Bayer->RGB demosaicing only supports 8u and 16u types");
            }
            else
            {
                CV_Assert( depth == CV_8U );
                Bayer2RGB_VNG_8u(src, dst_, code);
            }
        }
        break;

    case COLOR_BayerBG2BGR_EA: case COLOR_BayerGB2BGR_EA: case COLOR_BayerRG2BGR_EA: case COLOR_BayerGR2BGR_EA:
        if (dcn <= 0)
            dcn = 3;

        CV_Assert(scn == 1 && dcn == 3);
        _dst.create(sz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getMat();

        if (depth == CV_8U)
            Bayer2RGB_EdgeAware_T<uchar, SIMDBayerInterpolator_8u>(src, dst, code);
        else if (depth == CV_16U)
            Bayer2RGB_EdgeAware_T<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst, code);
        else
            CV_Error(cv::Error::StsUnsupportedFormat, "Bayer->RGB Edge-Aware demosaicing only currently supports 8u and 16u types");

        break;

    default:
        CV_Error( cv::Error::StsBadFlag, "Unknown / unsupported color conversion code" );
    }
}
