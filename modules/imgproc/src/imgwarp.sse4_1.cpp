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
#include "imgwarp.hpp"

namespace cv
{

    namespace opt_SSE4_1
{
inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_g1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
}

void convertMaps_nninterpolate32f1c16s_SSE41(const float* src1f, const float* src2f, short* dst1, int width)
{
    int x = 0;
    for (; x <= width - 16; x += 16)
    {
        __m128i v_dst0 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_loadu_ps(src1f + x)),
            _mm_cvtps_epi32(_mm_loadu_ps(src1f + x + 4)));
        __m128i v_dst1 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_loadu_ps(src1f + x + 8)),
            _mm_cvtps_epi32(_mm_loadu_ps(src1f + x + 12)));

        __m128i v_dst2 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_loadu_ps(src2f + x)),
            _mm_cvtps_epi32(_mm_loadu_ps(src2f + x + 4)));
        __m128i v_dst3 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_loadu_ps(src2f + x + 8)),
            _mm_cvtps_epi32(_mm_loadu_ps(src2f + x + 12)));

        _mm_interleave_epi16(v_dst0, v_dst1, v_dst2, v_dst3);

        _mm_storeu_si128((__m128i *)(dst1 + x * 2), v_dst0);
        _mm_storeu_si128((__m128i *)(dst1 + x * 2 + 8), v_dst1);
        _mm_storeu_si128((__m128i *)(dst1 + x * 2 + 16), v_dst2);
        _mm_storeu_si128((__m128i *)(dst1 + x * 2 + 24), v_dst3);
    }

    for (; x < width; x++)
    {
        dst1[x * 2] = saturate_cast<short>(src1f[x]);
        dst1[x * 2 + 1] = saturate_cast<short>(src2f[x]);
    }
}

void convertMaps_32f1c16s_SSE41(const float* src1f, const float* src2f, short* dst1, ushort* dst2, int width)
{
    int x = 0;
    __m128 v_its = _mm_set1_ps(INTER_TAB_SIZE);
    __m128i v_its1 = _mm_set1_epi32(INTER_TAB_SIZE - 1);

    for (; x <= width - 16; x += 16)
    {
        __m128i v_ix0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src1f + x), v_its));
        __m128i v_ix1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src1f + x + 4), v_its));
        __m128i v_iy0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src2f + x), v_its));
        __m128i v_iy1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src2f + x + 4), v_its));

        __m128i v_dst10 = _mm_packs_epi32(_mm_srai_epi32(v_ix0, INTER_BITS),
            _mm_srai_epi32(v_ix1, INTER_BITS));
        __m128i v_dst12 = _mm_packs_epi32(_mm_srai_epi32(v_iy0, INTER_BITS),
            _mm_srai_epi32(v_iy1, INTER_BITS));
        __m128i v_dst20 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_iy0, v_its1), INTER_BITS),
            _mm_and_si128(v_ix0, v_its1));
        __m128i v_dst21 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_iy1, v_its1), INTER_BITS),
            _mm_and_si128(v_ix1, v_its1));
        _mm_storeu_si128((__m128i *)(dst2 + x), _mm_packus_epi32(v_dst20, v_dst21));

        v_ix0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src1f + x + 8), v_its));
        v_ix1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src1f + x + 12), v_its));
        v_iy0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src2f + x + 8), v_its));
        v_iy1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src2f + x + 12), v_its));

        __m128i v_dst11 = _mm_packs_epi32(_mm_srai_epi32(v_ix0, INTER_BITS),
            _mm_srai_epi32(v_ix1, INTER_BITS));
        __m128i v_dst13 = _mm_packs_epi32(_mm_srai_epi32(v_iy0, INTER_BITS),
            _mm_srai_epi32(v_iy1, INTER_BITS));
        v_dst20 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_iy0, v_its1), INTER_BITS),
            _mm_and_si128(v_ix0, v_its1));
        v_dst21 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_iy1, v_its1), INTER_BITS),
            _mm_and_si128(v_ix1, v_its1));
        _mm_storeu_si128((__m128i *)(dst2 + x + 8), _mm_packus_epi32(v_dst20, v_dst21));

        _mm_interleave_epi16(v_dst10, v_dst11, v_dst12, v_dst13);

        _mm_storeu_si128((__m128i *)(dst1 + x * 2), v_dst10);
        _mm_storeu_si128((__m128i *)(dst1 + x * 2 + 8), v_dst11);
        _mm_storeu_si128((__m128i *)(dst1 + x * 2 + 16), v_dst12);
        _mm_storeu_si128((__m128i *)(dst1 + x * 2 + 24), v_dst13);
    }
    for (; x < width; x++)
    {
        int ix = saturate_cast<int>(src1f[x] * static_cast<float>(INTER_TAB_SIZE));
        int iy = saturate_cast<int>(src2f[x] * static_cast<float>(INTER_TAB_SIZE));
        dst1[x * 2] = saturate_cast<short>(ix >> INTER_BITS);
        dst1[x * 2 + 1] = saturate_cast<short>(iy >> INTER_BITS);
        dst2[x] = (ushort)((iy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE - 1)));
    }
}

void convertMaps_32f2c16s_SSE41(const float* src1f, short* dst1, ushort* dst2, int width)
{
    int x = 0;
    __m128 v_its = _mm_set1_ps(INTER_TAB_SIZE);
    __m128i v_its1 = _mm_set1_epi32(INTER_TAB_SIZE - 1);
    __m128i v_y_mask = _mm_set1_epi32((INTER_TAB_SIZE - 1) << 16);

    for (; x <= width - 4; x += 4)
    {
        __m128i v_src0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src1f + x * 2), v_its));
        __m128i v_src1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(src1f + x * 2 + 4), v_its));

        __m128i v_dst1 = _mm_packs_epi32(_mm_srai_epi32(v_src0, INTER_BITS),
            _mm_srai_epi32(v_src1, INTER_BITS));
        _mm_storeu_si128((__m128i *)(dst1 + x * 2), v_dst1);

        // x0 y0 x1 y1 . . .
        v_src0 = _mm_packs_epi32(_mm_and_si128(v_src0, v_its1),
            _mm_and_si128(v_src1, v_its1));
        __m128i v_dst2 = _mm_or_si128(_mm_srli_epi32(_mm_and_si128(v_src0, v_y_mask), 16 - INTER_BITS), // y0 0 y1 0 . . .
            _mm_and_si128(v_src0, v_its1)); // 0 x0 0 x1 . . .
        _mm_storel_epi64((__m128i *)(dst2 + x), _mm_packus_epi32(v_dst2, v_dst2));
    }
    for (; x < width; x++)
    {
        int ix = saturate_cast<int>(src1f[x * 2] * static_cast<float>(INTER_TAB_SIZE));
        int iy = saturate_cast<int>(src1f[x * 2 + 1] * static_cast<float>(INTER_TAB_SIZE));
        dst1[x * 2] = saturate_cast<short>(ix >> INTER_BITS);
        dst1[x * 2 + 1] = saturate_cast<short>(iy >> INTER_BITS);
        dst2[x] = (ushort)((iy & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE - 1)));
    }
}

class WarpPerspectiveLine_SSE4_Impl CV_FINAL : public WarpPerspectiveLine_SSE4
{
public:
    WarpPerspectiveLine_SSE4_Impl(const double *M)
    {
        CV_UNUSED(M);
    }
    virtual void processNN(const double *M, short* xy, double X0, double Y0, double W0, int bw) CV_OVERRIDE
    {
        const __m128d v_M0 = _mm_set1_pd(M[0]);
        const __m128d v_M3 = _mm_set1_pd(M[3]);
        const __m128d v_M6 = _mm_set1_pd(M[6]);
        const __m128d v_intmax = _mm_set1_pd((double)INT_MAX);
        const __m128d v_intmin = _mm_set1_pd((double)INT_MIN);
        const __m128d v_2 = _mm_set1_pd(2);
        const __m128d v_zero = _mm_setzero_pd();
        const __m128d v_1 = _mm_set1_pd(1);

        int x1 = 0;
        __m128d v_X0d = _mm_set1_pd(X0);
        __m128d v_Y0d = _mm_set1_pd(Y0);
        __m128d v_W0 = _mm_set1_pd(W0);
        __m128d v_x1 = _mm_set_pd(1, 0);

        for (; x1 <= bw - 16; x1 += 16)
        {
            // 0-3
            __m128i v_X0, v_Y0;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // 4-8
            __m128i v_X1, v_Y1;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X1 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y1 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // 8-11
            __m128i v_X2, v_Y2;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // 12-15
            __m128i v_X3, v_Y3;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_1, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X3 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y3 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // convert to 16s
            v_X0 = _mm_packs_epi32(v_X0, v_X1);
            v_X1 = _mm_packs_epi32(v_X2, v_X3);
            v_Y0 = _mm_packs_epi32(v_Y0, v_Y1);
            v_Y1 = _mm_packs_epi32(v_Y2, v_Y3);

            _mm_interleave_epi16(v_X0, v_X1, v_Y0, v_Y1);

            _mm_storeu_si128((__m128i *)(xy + x1 * 2), v_X0);
            _mm_storeu_si128((__m128i *)(xy + x1 * 2 + 8), v_X1);
            _mm_storeu_si128((__m128i *)(xy + x1 * 2 + 16), v_Y0);
            _mm_storeu_si128((__m128i *)(xy + x1 * 2 + 24), v_Y1);
        }

        for (; x1 < bw; x1++)
        {
            double W = W0 + M[6] * x1;
            W = W ? 1. / W : 0;
            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1)*W));
            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1)*W));
            int X = saturate_cast<int>(fX);
            int Y = saturate_cast<int>(fY);

            xy[x1 * 2] = saturate_cast<short>(X);
            xy[x1 * 2 + 1] = saturate_cast<short>(Y);
        }
    }
    virtual void process(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw) CV_OVERRIDE
    {
        const __m128d v_M0 = _mm_set1_pd(M[0]);
        const __m128d v_M3 = _mm_set1_pd(M[3]);
        const __m128d v_M6 = _mm_set1_pd(M[6]);
        const __m128d v_intmax = _mm_set1_pd((double)INT_MAX);
        const __m128d v_intmin = _mm_set1_pd((double)INT_MIN);
        const __m128d v_2 = _mm_set1_pd(2);
        const __m128d v_zero = _mm_setzero_pd();
        const __m128d v_its = _mm_set1_pd(INTER_TAB_SIZE);
        const __m128i v_itsi1 = _mm_set1_epi32(INTER_TAB_SIZE - 1);

        int x1 = 0;

        __m128d v_X0d = _mm_set1_pd(X0);
        __m128d v_Y0d = _mm_set1_pd(Y0);
        __m128d v_W0 = _mm_set1_pd(W0);
        __m128d v_x1 = _mm_set_pd(1, 0);

        for (; x1 <= bw - 16; x1 += 16)
        {
            // 0-3
            __m128i v_X0, v_Y0;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y0 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // 4-8
            __m128i v_X1, v_Y1;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X1 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y1 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // 8-11
            __m128i v_X2, v_Y2;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y2 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // 12-15
            __m128i v_X3, v_Y3;
            {
                __m128d v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY0 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_W = _mm_add_pd(_mm_mul_pd(v_M6, v_x1), v_W0);
                v_W = _mm_andnot_pd(_mm_cmpeq_pd(v_W, v_zero), _mm_div_pd(v_its, v_W));
                __m128d v_fX1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_X0d, _mm_mul_pd(v_M0, v_x1)), v_W)));
                __m128d v_fY1 = _mm_max_pd(v_intmin, _mm_min_pd(v_intmax, _mm_mul_pd(_mm_add_pd(v_Y0d, _mm_mul_pd(v_M3, v_x1)), v_W)));
                v_x1 = _mm_add_pd(v_x1, v_2);

                v_X3 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fX0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fX1))));
                v_Y3 = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_fY0)),
                    _mm_castsi128_ps(_mm_cvtpd_epi32(v_fY1))));
            }

            // store alpha
            __m128i v_alpha0 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_Y0, v_itsi1), INTER_BITS),
                _mm_and_si128(v_X0, v_itsi1));
            __m128i v_alpha1 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_Y1, v_itsi1), INTER_BITS),
                _mm_and_si128(v_X1, v_itsi1));
            _mm_storeu_si128((__m128i *)(alpha + x1), _mm_packs_epi32(v_alpha0, v_alpha1));

            v_alpha0 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_Y2, v_itsi1), INTER_BITS),
                _mm_and_si128(v_X2, v_itsi1));
            v_alpha1 = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(v_Y3, v_itsi1), INTER_BITS),
                _mm_and_si128(v_X3, v_itsi1));
            _mm_storeu_si128((__m128i *)(alpha + x1 + 8), _mm_packs_epi32(v_alpha0, v_alpha1));

            // convert to 16s
            v_X0 = _mm_packs_epi32(_mm_srai_epi32(v_X0, INTER_BITS), _mm_srai_epi32(v_X1, INTER_BITS));
            v_X1 = _mm_packs_epi32(_mm_srai_epi32(v_X2, INTER_BITS), _mm_srai_epi32(v_X3, INTER_BITS));
            v_Y0 = _mm_packs_epi32(_mm_srai_epi32(v_Y0, INTER_BITS), _mm_srai_epi32(v_Y1, INTER_BITS));
            v_Y1 = _mm_packs_epi32(_mm_srai_epi32(v_Y2, INTER_BITS), _mm_srai_epi32(v_Y3, INTER_BITS));

            _mm_interleave_epi16(v_X0, v_X1, v_Y0, v_Y1);

            _mm_storeu_si128((__m128i *)(xy + x1 * 2), v_X0);
            _mm_storeu_si128((__m128i *)(xy + x1 * 2 + 8), v_X1);
            _mm_storeu_si128((__m128i *)(xy + x1 * 2 + 16), v_Y0);
            _mm_storeu_si128((__m128i *)(xy + x1 * 2 + 24), v_Y1);
        }
        for (; x1 < bw; x1++)
        {
            double W = W0 + M[6] * x1;
            W = W ? static_cast<double>(INTER_TAB_SIZE) / W : 0;
            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1)*W));
            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1)*W));
            int X = saturate_cast<int>(fX);
            int Y = saturate_cast<int>(fY);

            xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
            xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
            alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1))*INTER_TAB_SIZE +
                (X & (INTER_TAB_SIZE - 1)));
        }
    }
    virtual ~WarpPerspectiveLine_SSE4_Impl() CV_OVERRIDE {}
};

Ptr<WarpPerspectiveLine_SSE4> WarpPerspectiveLine_SSE4::getImpl(const double *M)
{
    return Ptr<WarpPerspectiveLine_SSE4>(new WarpPerspectiveLine_SSE4_Impl(M));
}

}
}
/* End of file. */
