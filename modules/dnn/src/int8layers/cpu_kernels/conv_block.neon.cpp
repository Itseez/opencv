// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "convolution.hpp"

namespace cv {
namespace dnn {

namespace opt_NEON
{
#if CV_NEON && CV_NEON_AARCH64 && defined(__ARM_FEATURE_DOTPROD)
void convBlock_INT8(int np, const char* _a, const char* _b, int* c, int ldc, bool init_c, const int width, const int convMR, const int convNR)
{
    CV_Assert(convMR == 4);
    const int8_t* a = (const int8_t*)_a;
    const int8_t* b = (const int8_t*)_b;

    if (width > 12)
    {
        int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00, c04 = c00, c05 = c00;
        int32x4_t c10 = vdupq_n_s32(0), c11 = c10, c12 = c10, c13 = c10, c14 = c10, c15 = c10;
        int32x4_t c20 = vdupq_n_s32(0), c21 = c20, c22 = c20, c23 = c20, c24 = c20, c25 = c20;
        int32x4_t c30 = vdupq_n_s32(0), c31 = c30, c32 = c30, c33 = c30, c34 = c30, c35 = c30;

        for (int p = 0; p < np; p += 4, a += 4 * convMR, b += 4 * convNR)
        {
            int8x16_t a0 = vld1q_s8(a), b0, b1, b2;

            b0 = vld1q_s8(b); b1 = vld1q_s8(b + 16); b2 = vld1q_s8(b + 32);
            c00 = vdotq_laneq_s32(c00, b0, a0, 0);
            c01 = vdotq_laneq_s32(c01, b1, a0, 0);
            c02 = vdotq_laneq_s32(c02, b2, a0, 0);
            c10 = vdotq_laneq_s32(c10, b0, a0, 1);
            c11 = vdotq_laneq_s32(c11, b1, a0, 1);
            c12 = vdotq_laneq_s32(c12, b2, a0, 1);
            c20 = vdotq_laneq_s32(c20, b0, a0, 2);
            c21 = vdotq_laneq_s32(c21, b1, a0, 2);
            c22 = vdotq_laneq_s32(c22, b2, a0, 2);
            c30 = vdotq_laneq_s32(c30, b0, a0, 3);
            c31 = vdotq_laneq_s32(c31, b1, a0, 3);
            c32 = vdotq_laneq_s32(c32, b2, a0, 3);

            b0 = vld1q_s8(b + 48); b1 = vld1q_s8(b + 64); b2 = vld1q_s8(b + 80);
            c03 = vdotq_laneq_s32(c03, b0, a0, 0);
            c04 = vdotq_laneq_s32(c04, b1, a0, 0);
            c05 = vdotq_laneq_s32(c05, b2, a0, 0);
            c13 = vdotq_laneq_s32(c13, b0, a0, 1);
            c14 = vdotq_laneq_s32(c14, b1, a0, 1);
            c15 = vdotq_laneq_s32(c15, b2, a0, 1);
            c23 = vdotq_laneq_s32(c23, b0, a0, 2);
            c24 = vdotq_laneq_s32(c24, b1, a0, 2);
            c25 = vdotq_laneq_s32(c25, b2, a0, 2);
            c33 = vdotq_laneq_s32(c33, b0, a0, 3);
            c34 = vdotq_laneq_s32(c34, b1, a0, 3);
            c35 = vdotq_laneq_s32(c35, b2, a0, 3);
        }

        if (!init_c)
        {
#undef NEON_UPDATE_QCONV_BLOCK
#define NEON_UPDATE_QCONV_BLOCK(i) \
            c##i##0 = vaddq_s32(c##i##0, vld1q_s32(c+i*ldc)); \
            c##i##1 = vaddq_s32(c##i##1, vld1q_s32(c+i*ldc+4)); \
            c##i##2 = vaddq_s32(c##i##2, vld1q_s32(c+i*ldc+8)); \
            c##i##3 = vaddq_s32(c##i##3, vld1q_s32(c+i*ldc+12)); \
            c##i##4 = vaddq_s32(c##i##4, vld1q_s32(c+i*ldc+16)); \
            c##i##5 = vaddq_s32(c##i##5, vld1q_s32(c+i*ldc+20))

            NEON_UPDATE_QCONV_BLOCK(0);
            NEON_UPDATE_QCONV_BLOCK(1);
            NEON_UPDATE_QCONV_BLOCK(2);
            NEON_UPDATE_QCONV_BLOCK(3);
        }

#undef NEON_STORE_QCONV_BLOCK
#define NEON_STORE_QCONV_BLOCK(i) \
        vst1q_s32(c+i*ldc, c##i##0); \
        vst1q_s32(c+i*ldc+4, c##i##1); \
        vst1q_s32(c+i*ldc+8, c##i##2); \
        vst1q_s32(c+i*ldc+12, c##i##3); \
        vst1q_s32(c+i*ldc+16, c##i##4); \
        vst1q_s32(c+i*ldc+20, c##i##5)

        NEON_STORE_QCONV_BLOCK(0);
        NEON_STORE_QCONV_BLOCK(1);
        NEON_STORE_QCONV_BLOCK(2);
        NEON_STORE_QCONV_BLOCK(3);
    }
    else
    {
        int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00;
        int32x4_t c10 = vdupq_n_s32(0), c11 = c10, c12 = c10;
        int32x4_t c20 = vdupq_n_s32(0), c21 = c20, c22 = c20;
        int32x4_t c30 = vdupq_n_s32(0), c31 = c30, c32 = c30;

        for (int p = 0; p < np; p += 4, a += 4 * convMR, b += 4 * convNR)
        {
            int8x16_t a0 = vld1q_s8(a), b0, b1, b2;

            b0 = vld1q_s8(b); b1 = vld1q_s8(b + 16); b2 = vld1q_s8(b + 32);
            c00 = vdotq_laneq_s32(c00, b0, a0, 0);
            c01 = vdotq_laneq_s32(c01, b1, a0, 0);
            c02 = vdotq_laneq_s32(c02, b2, a0, 0);
            c10 = vdotq_laneq_s32(c10, b0, a0, 1);
            c11 = vdotq_laneq_s32(c11, b1, a0, 1);
            c12 = vdotq_laneq_s32(c12, b2, a0, 1);
            c20 = vdotq_laneq_s32(c20, b0, a0, 2);
            c21 = vdotq_laneq_s32(c21, b1, a0, 2);
            c22 = vdotq_laneq_s32(c22, b2, a0, 2);
            c30 = vdotq_laneq_s32(c30, b0, a0, 3);
            c31 = vdotq_laneq_s32(c31, b1, a0, 3);
            c32 = vdotq_laneq_s32(c32, b2, a0, 3);
        }

        if (!init_c)
        {
#undef NEON_UPDATE_QCONV_BLOCK
#define NEON_UPDATE_QCONV_BLOCK(i) \
            c##i##0 = vaddq_s32(c##i##0, vld1q_s32(c+i*ldc)); \
            c##i##1 = vaddq_s32(c##i##1, vld1q_s32(c+i*ldc+4)); \
            c##i##2 = vaddq_s32(c##i##2, vld1q_s32(c+i*ldc+8))

            NEON_UPDATE_QCONV_BLOCK(0);
            NEON_UPDATE_QCONV_BLOCK(1);
            NEON_UPDATE_QCONV_BLOCK(2);
            NEON_UPDATE_QCONV_BLOCK(3);
        }

#undef NEON_STORE_QCONV_BLOCK
#define NEON_STORE_QCONV_BLOCK(i) \
        vst1q_s32(c+i*ldc, c##i##0); \
        vst1q_s32(c+i*ldc+4, c##i##1); \
        vst1q_s32(c+i*ldc+8, c##i##2)

        NEON_STORE_QCONV_BLOCK(0);
        NEON_STORE_QCONV_BLOCK(1);
        NEON_STORE_QCONV_BLOCK(2);
        NEON_STORE_QCONV_BLOCK(3);
    }
}
#else
void convBlock_INT8(int , const char* , const char* , int* , int , bool , const int , const int , const int )
{
    CV_Error(Error::StsUnsupportedFormat, "opt_NEON::convBlock_INT8 func is not supported!");
}
#endif
}
}} // namespace cv::dnn
