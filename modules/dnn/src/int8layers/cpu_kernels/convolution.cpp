// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpConv_Quantized.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "convolution.hpp"

namespace cv { namespace dnn {
enum { VEC_ALIGN = 32}; // Memory alignment.

void convBlock_INT8(int np, const char* a, const char* b, int* c, int ldc, bool init_c, const int outLen,
                    const int convMR, const int convNR);

Ptr<FastQConv> initFastQConv(
            InputArray _weightsMat,
            int* srcBias,
            int ngroups,
            int K, int C,
            const std::vector<size_t>& kernel_size,
            const std::vector<size_t>& strides,
            const std::vector<size_t>& dilations,
            const std::vector<size_t>& pads_begin,
            const std::vector<size_t>& pads_end,
            int conv_dim,
            const std::vector<float>& outputMultiplier,
            float input_sc,
            int input_zp,
            float output_sc,
            int output_zp,
            bool per_channel)
{
    Ptr<FastQConv> conv = makePtr<FastQConv>();

    CV_Assert(ngroups > 0 && K > 0 && C > 0 && K % ngroups == 0);

    // Weight shape, [K, C, Dk, Hk, Wk] for Conv3D, [K, C, Hk, Wk] for Conv2D, [K, C, Wk] for Conv1D.
    int Dk = conv_dim == CONV_3D ? (int)kernel_size[0] : 1;
    int Hk = conv_dim == CONV_1D ? 1 : (int)kernel_size[kernel_size.size() - 2];
    int Wk = (int)kernel_size.back();
    int karea = Wk*Hk*Dk;

    conv->pad_front = conv_dim == CONV_3D ? (int)pads_begin[0] : 0;
    conv->pad_top = conv_dim == CONV_1D ? 0 : (int)pads_begin[pads_begin.size() - 2];
    conv->pad_left = (int)pads_begin.back();

    conv->pad_behind = conv_dim == CONV_3D ? (int)pads_end[0] : 0;
    conv->pad_bottom = conv_dim == CONV_1D ? 0 : (int)pads_end[pads_end.size() - 2];
    conv->pad_right = (int)pads_end.back();

    int stride_d = conv_dim == CONV_3D ? (int)strides[0] : 1;
    int stride_h = conv_dim == CONV_1D ? 1 : (int)strides[strides.size() - 2];
    int stride_w = (int)strides.back();

    int dilation_d = conv_dim == CONV_3D ? (int)dilations[0] : 1;
    int dilation_h = conv_dim == CONV_1D ? 1 : (int)dilations[dilations.size() - 2];
    int dilation_w = (int)dilations.back();

    CV_Assert(Dk > 0 && Hk > 0 && Wk > 0);
    CV_Assert(stride_d >= 0 && stride_h >= 0 && stride_w > 0);
    CV_Assert(dilation_d > 0 && dilation_h > 0 && dilation_w > 0);

    conv->K = K; conv->C = C; conv->Hk = Hk; conv->Wk = Wk, conv->Dk = Dk;

    conv->stride_d = stride_d;
    conv->stride_h = stride_h;
    conv->stride_w = stride_w;

    conv->dilation_d = dilation_d;
    conv->dilation_h = dilation_h;
    conv->dilation_w = dilation_w;
    conv->conv_dim = conv_dim;
    conv->ngroups = ngroups;

    conv->outputMultiplier.assign(outputMultiplier.begin(), outputMultiplier.end());
    conv->input_sc = input_sc;
    conv->input_zp = input_zp;
    conv->output_sc = output_sc;
    conv->output_zp = output_zp;

    conv->per_channel = per_channel;

    bool ifRunDepthWise = ngroups > 1 && ngroups == K && ngroups == C;

    // TODO, support CONV_1D and CONV_3D at depth_wise branch.
    ifRunDepthWise &= conv_dim == CONV_2D && Hk == 3 && Wk == 3 && ((stride_w == 1) || (stride_w == 2 && dilation_w == 1)) &&
                              max(stride_w, dilation_w) >= conv->pad_left && max(stride_h, dilation_h) >= conv->pad_top
                              && conv->pad_left <= 1 && conv->pad_top <= 1;

    conv->conv_type = ifRunDepthWise? QCONV_TYPE_DEPTHWISE : QCONV_TYPE_GENERIC;

    Mat weightsMat = _weightsMat.getMat();
    auto wShape = shape(weightsMat);
    const size_t wstep = weightsMat.step1();

    // TODO! for ARM platform, we use block layout. We need re-pack the weight to [K/4, CgHkWk/4, 4, 4]
    // And for AVX, or other platform, should we keep the same pack data layout? TBD.
    char *srcWeights = (char *)weightsMat.data;

    if (conv->conv_type == QCONV_TYPE_DEPTHWISE)
    {
        // Handle the Conv1D, Conv2D and Conv3D depth-wise.
        // for depth-wise convolutions on NCHW data we just preserve the weights in KCHW layout,
        // but add some padding to make the weights array layout more SIMD-friendly
        int ksize = karea;

        // TODO: simplify the following code with std::copy.
        // this code aims to let memory fit with vector size.
        int padded_ksize = ((ksize + VEC_ALIGN-1) / VEC_ALIGN) * VEC_ALIGN;
        int nweights = C*padded_ksize;
        conv->weightsBuf.reserve(nweights + VEC_ALIGN);
        conv->weightsBufPtr = alignPtr(conv->weightsBuf.data(), VEC_ALIGN);
        memset(conv->weightsBufPtr, 0, nweights*sizeof(conv->weightsBufPtr[0]));
        auto weightsBufPtr = conv->weightsBufPtr;

        parallel_for_(Range(0, C), [&](const Range& r0){
        for(int c = r0.start; c < r0.end; c++)
        {
            for (int k = 0; k < ksize; k++)
                weightsBufPtr[c*padded_ksize + k] = srcWeights[c*wstep + k];
        }});
    }
    else if (conv->conv_type == QCONV_TYPE_GENERIC)
    {
        // The weights are packed as
        // ngroups x (ceil((K/ngroups)/CONV_MR)*CONV_MR) x (Cg*Hk*Wk*Dk) x CONV_MR tensor
        int Kg = K/ngroups, Cg = max(C/ngroups, 1);
        int numStripsMR = (Kg + CONV_MR - 1) / CONV_MR;
        int Kg_aligned = numStripsMR * CONV_MR;
        int DkHkWk = Dk*Hk*Wk;

        // Block layout branch.
        int Cg_strips = (Cg + CONV_PACKN - 1) / CONV_PACKN;
        int Cg_aligned = Cg_strips * CONV_PACKN;
        int DkHkWkCg_aligned = Dk*Hk*Wk*Cg_aligned;

        size_t nweights = ngroups*Kg_aligned*DkHkWkCg_aligned;
        conv->weightsBuf.reserve(nweights + VEC_ALIGN);
        conv->weightsBufPtr = alignPtr(conv->weightsBuf.data(), VEC_ALIGN);
        char* weightsBufPtr = conv->weightsBufPtr;
        memset(weightsBufPtr, 0, nweights * sizeof(weightsBufPtr[0]));

        parallel_for_(Range(0, ngroups * numStripsMR), [&](const Range& r0){
        for (int gsi = r0.start; gsi < r0.end; gsi++)
        {
            int g = gsi / numStripsMR;
            int si = gsi - g * numStripsMR;

            int startK = si * CONV_MR;
            CV_Assert(startK < Kg_aligned);

            int dk = Kg - startK < CONV_MR ? Kg - startK : CONV_MR; // check if we need zero padding.

            char* packed_wptr = weightsBufPtr + DkHkWkCg_aligned * (startK + g * Kg_aligned);
            int k_idx = g*Kg + startK;

            for (int hwd = 0; hwd < DkHkWk; hwd++)
            {
                for (int c = 0; c < Cg; c += CONV_PACKN, packed_wptr += CONV_PACKN * CONV_MR)
                {
                    const char * wptr = srcWeights + wstep * k_idx + c*Hk*Wk*Dk + hwd;

                    int k = 0;
                    for(; k < dk; k++, wptr += wstep)
                    {
                        for (int ci = 0; ci < CONV_PACKN; ci++)
                        {
                            char w = (c + ci) < Cg ? wptr[DkHkWk * ci] : 0;
                            packed_wptr[k * CONV_PACKN + ci] = w;
                        }
                    }

                    for(; k < CONV_MR; k++)
                    {
                        for (int ci = 0; ci < CONV_PACKN; ci++)
                        {
                            packed_wptr[k * CONV_PACKN + ci] = 0;
                        }
                    }
                }
            }
        }});
    }
    else
        CV_Error(CV_StsUnsupportedFormat, "Unknown convolution type.");

    // store bias; append some zero's to make sure that
    // we can always read MR elements starting from any valid index
    {
        int k = 0, nbias = K + VEC_ALIGN;
        conv->biasBuf.resize(nbias);
        int* biasBufPtr = conv->biasBuf.data();
        for(; k < K; k++)
            biasBufPtr[k] = srcBias ? srcBias[k] : 0;
        for(; k < nbias; k++)
            biasBufPtr[k] = 0;
    }
    return conv;
}

#if CONV_PACKN == 4 || CONV_PACKN == 8
static inline void store_block_layout_24(const int8_t* src, int8_t* dst, const int src_planesize)
{

#if CV_SIMD128 && CONV_PACKN == 4
    v_int8x16 v0 = v_load(src);
    v_int8x16 v1 = v_load(src + src_planesize);
    v_int8x16 v2 = v_load(src + 2 * src_planesize);
    v_int8x16 v3 = v_load(src + 3 * src_planesize);

    v_store_interleave(dst, v0, v1, v2, v3);

    auto src2 = src+16;

    v0 = v_load_halves(src2                    , src2 +     src_planesize);
    v1 = v_load_halves(src2 + 2 * src_planesize, src2 + 3 * src_planesize);

    v_zip(v0, v1, v2, v3);
    v_store_interleave(dst + 64, v2, v3);

#elif CV_SIMD256 && CONV_PACKN == 8
    v_int8x16 v0 = v_load(src);
    v_int8x16 v1 = v_load(src + src_planesize);
    v_int8x16 v2 = v_load(src + 2 * src_planesize);
    v_int8x16 v3 = v_load(src + 3 * src_planesize);
    v_int8x16 v4 = v_load(src + 4 * src_planesize);
    v_int8x16 v5 = v_load(src + 5 * src_planesize);
    v_int8x16 v6 = v_load(src + 6 * src_planesize);
    v_int8x16 v7 = v_load(src + 7 * src_planesize);

    // Pack 8
    /*
     * We have the [A0, B0, C0, D0,...] ...[A7, B7, C7, D7,...]
     * And we want to get the [A0, A1, A2, A3, .., B14, B15],...., [...]
     * Use v_zip, combine two by two.
     * First:
     * Combine the 04, 26, 15, 37
     * Then, combine the 0246, 1357.
     * Finally, we get 01234567.
     * */

    v_int8x16 v04_0, v04_1, v26_0, v26_1, v15_0, v15_1, v37_0, v37_1;
    v_zip(v0, v4, v04_0, v04_1);
    v_zip(v2, v6, v26_0, v26_1);
    v_zip(v1, v5, v15_0, v15_1);
    v_zip(v3, v7, v37_0, v37_1);

    v_int8x16 v0246_0, v0246_1, v0246_2, v0246_3, v1357_0, v1357_1, v1357_2, v1357_3;
    v_zip(v04_0, v26_0, v0246_0, v0246_1);
    v_zip(v04_1, v26_1, v0246_2, v0246_3);
    v_zip(v15_0, v37_0, v1357_0, v1357_1);
    v_zip(v15_1, v37_1, v1357_2, v1357_3);

    v_store_interleave(dst     , v0246_0, v1357_0);
    v_store_interleave(dst + 32, v0246_1, v1357_1);
    v_store_interleave(dst + 64, v0246_2, v1357_2);
    v_store_interleave(dst + 96, v0246_3, v1357_3);

    auto src2 = src+16;

    v_int8x16 _v01 = v_load_halves(src2                    , src2 + 1 * src_planesize);
    v_int8x16 _v23 = v_load_halves(src2 + 2 * src_planesize, src2 + 3 * src_planesize);
    v_int8x16 _v45 = v_load_halves(src2 + 4 * src_planesize, src2 + 5 * src_planesize);
    v_int8x16 _v67 = v_load_halves(src2 + 6 * src_planesize, src2 + 7 * src_planesize);

    v_int8x16 v04, v15, v26, v37;
    v_zip(_v01, _v45, v04, v15);
    v_zip(_v23, _v67, v26, v37);

    v_store_interleave(dst + 128, v04, v15, v26, v37);
#else
    for (int i = 0; i < CONV_PACKN; i++)
    {
        for (int j = 0; j < 24; j++)
        {
            dst[j * CONV_PACKN + i] = dst[i * src_planesize + j];
        }
    }
#endif
}

#undef PACK_I8_LOAD8
#define PACK_I8_LOAD8(c, stride, ptr)     \
    int8_t t0##c = *((ptr) + stride*0);    \
    int8_t t1##c = *((ptr) + stride*1);    \
    int8_t t2##c = *((ptr) + stride*2);    \
    int8_t t3##c = *((ptr) + stride*3);    \
    int8_t t4##c = *((ptr) + stride*4);    \
    int8_t t5##c = *((ptr) + stride*5);    \
    int8_t t6##c = *((ptr) + stride*6);    \
    int8_t t7##c = *((ptr) + stride*7)

#undef PACK4_I8_STORE4
#define PACK4_I8_STORE4(group, ptr)          \
    *((ptr) + group*4 + 0) = t##group##0;   \
    *((ptr) + group*4 + 1) = t##group##1;   \
    *((ptr) + group*4 + 2) = t##group##2;   \
    *((ptr) + group*4 + 3) = t##group##3

#undef PACK8_I8_STORE8
#define PACK8_I8_STORE8(group, ptr)         \
    *((ptr) + group*8 + 0) = t##group##0;   \
    *((ptr) + group*8 + 1) = t##group##1;   \
    *((ptr) + group*8 + 2) = t##group##2;   \
    *((ptr) + group*8 + 3) = t##group##3;   \
    *((ptr) + group*8 + 4) = t##group##4;   \
    *((ptr) + group*8 + 5) = t##group##5;   \
    *((ptr) + group*8 + 6) = t##group##6;   \
    *((ptr) + group*8 + 7) = t##group##7

#undef PACK4_I8_LOAD4
#define PACK4_I8_LOAD4(c, ptr)     \
    int8_t t0##c = *((ptr) + 0);    \
    int8_t t1##c = *((ptr) + 1);    \
    int8_t t2##c = *((ptr) + 2);    \
    int8_t t3##c = *((ptr) + 3);

#if CONV_NR == 12
static inline void store_block_layout_12(const int8_t* src, int8_t* dst, const int src_planesize)
{
#if CV_SIMD128
    CV_Assert(CONV_PACKN == 4);
    v_int8x16 v0 = v_load_halves(src                    , src +     src_planesize);
    v_int8x16 v1 = v_load_halves(src + 2 * src_planesize, src + 3 * src_planesize);

    v_int8x16 v_p0, v_p1;

    v_zip(v0, v1, v_p0, v_p1);
    v_store_interleave(dst, v_p0, v_p1);

    PACK4_I8_LOAD4(0, src + 8);
    PACK4_I8_LOAD4(1, src + 8 + src_planesize);
    PACK4_I8_LOAD4(2, src + 8 + src_planesize * 2);
    PACK4_I8_LOAD4(3, src + 8 + src_planesize * 3);

    PACK4_I8_STORE4(0, dst + 32);
    PACK4_I8_STORE4(1, dst + 32);
    PACK4_I8_STORE4(2, dst + 32);
    PACK4_I8_STORE4(3, dst + 32);
#else
    for (int i = 0; i < CONV_PACKN; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            dst[j * CONV_PACKN + i] = dst[i * src_planesize + j];
        }
    }
#endif
}
#endif

static inline void store_block_layout_8(const int8_t* src, int8_t* dst, const int src_planesize)
{
#if CV_SIMD128 && CONV_PACKN == 4
    v_int8x16 v0 = v_load_halves(src                    , src +     src_planesize);
    v_int8x16 v1 = v_load_halves(src + 2 * src_planesize, src + 3 * src_planesize);

    v_int8x16 v_p0, v_p1;

    v_zip(v0, v1, v_p0, v_p1);
    v_store_interleave(dst, v_p0, v_p1);
#elif CONV_PACKN == 8
    v_int8x16 _v01 = v_load_halves(src                    , src + 1 * src_planesize);
    v_int8x16 _v23 = v_load_halves(src + 2 * src_planesize, src + 3 * src_planesize);
    v_int8x16 _v45 = v_load_halves(src + 4 * src_planesize, src + 5 * src_planesize);
    v_int8x16 _v67 = v_load_halves(src + 6 * src_planesize, src + 7 * src_planesize);

    v_int8x16 v04, v15, v26, v37;
    v_zip(_v01, _v45, v04, v15);
    v_zip(_v23, _v67, v26, v37);

    v_store_interleave(dst, v04, v15, v26, v37);
#else
    for (int i = 0; i < CONV_PACKN; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            dst[j * CONV_PACKN + i] = dst[i * src_planesize + j];
        }
    }
#endif
}
#endif

// Do im2col and pack input data. And convert NCHW to NC4HW4.
static void packInputData(char* inpbuf_task, char* inp, const int* ofstab, const int* dhwTab, int zyx0, int zyx_limit,
                                int ksize, int stride_d, int stride_h, int stride_w, int pad_front, int pad_top, int pad_left, int Di, int Hi, int Wi,
                                int H0, int W0, int Cg, int stripesize, int inp_plane_ofs, int inp_planesize, bool fast_1x1, const int8_t input_zp)
{
#if CONV_PACKN == 4 || CONV_PACKN == 8
    AutoBuffer<int8_t> buffer, zbuffer;

    buffer.allocate(CONV_PACKN * CONV_NR + VEC_ALIGN);
    int8_t* bufferPtr = alignPtr(buffer.data(), VEC_ALIGN);

    zbuffer.allocate(CONV_NR + VEC_ALIGN);
    int8_t* zbufPtr = alignPtr(zbuffer.data(), VEC_ALIGN);
    memset(zbufPtr, input_zp, CONV_NR * sizeof(int8_t));
#endif

    for (int stripe = 0; zyx0 < zyx_limit; stripe++, zyx0 += CONV_NR)
    {
        int8_t *inpbuf = (int8_t *)inpbuf_task + stripe * stripesize;
        int8_t *inptr = (int8_t *)inp + inp_plane_ofs;

        if (fast_1x1)
        {
            int slice_len = zyx_limit - zyx0;
            bool partial = slice_len < CONV_NR;

            // Superfast branch for 1x1 convolutions with sy=sx=1.
            // in this case each feature plane can be safely treated
            // as 1D array, and we just extract next portion
            // of CONV_NR elements from each feature plane and
            // put it together.
            inptr += zyx0;

            if (!partial)
            {
#if CONV_PACKN == 4 || CONV_PACKN == 8
                int c = 0;
                for (; c <= Cg - CONV_PACKN; c += CONV_PACKN, inptr += CONV_PACKN * inp_planesize,
                        inpbuf += CONV_PACKN * CONV_NR)
                {
#if CONV_NR == 24
                    store_block_layout_24(inptr, inpbuf, inp_planesize);
#elif CONV_NR == 12
                    store_block_layout_12(inptr, inpbuf, inp_planesize);
#endif
                }

                if (c < Cg)
                {
                    memset(bufferPtr, input_zp, buffer.size() * sizeof(char));
                    for (int i = 0; c < Cg; c++, i++)
                    {
                        memcpy(bufferPtr + i * CONV_NR, inptr + i * inp_planesize, CONV_NR * sizeof(char));
                    }
#if CONV_NR == 24
                    store_block_layout_24(bufferPtr, inpbuf, CONV_NR);
#elif CONV_NR == 12
                    store_block_layout_12(bufferPtr, inpbuf, CONV_NR);
#endif
                }
#elif CONV_PACKN == 1
                for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR)
                    memcpy(inpbuf, inptr, CONV_NR * sizeof(inpbuf[0]));
#endif
            }
            else
            {
#if CONV_PACKN == 4 || CONV_PACKN == 8
                int c = 0;
                for (; c <= Cg - CONV_PACKN; c += CONV_PACKN, inptr += CONV_PACKN * inp_planesize,
                        inpbuf += CONV_PACKN * CONV_NR)
                {
                    memset(bufferPtr, input_zp, buffer.size() * sizeof(char));
                    for (int i = 0; i < CONV_PACKN; i++)
                    {
                        memcpy(bufferPtr + CONV_NR * i, inptr + i * inp_planesize, slice_len * sizeof(char));
                    }
#if CONV_NR == 24
                    store_block_layout_24(bufferPtr, inpbuf, CONV_NR);
#elif CONV_NR == 12
                    store_block_layout_12(bufferPtr, inpbuf, CONV_NR);
#endif
                }

                if (c < Cg)
                {
                    memset(bufferPtr, input_zp, buffer.size() * sizeof(char));
                    for (int i = 0; c < Cg; c++, i++)
                    {
                        memcpy(bufferPtr + i * CONV_NR, inptr + i * inp_planesize, slice_len * sizeof(char));
                    }

#if CONV_NR == 24
                    store_block_layout_24(bufferPtr, inpbuf, CONV_NR);
#elif CONV_NR == 12
                    store_block_layout_12(bufferPtr, inpbuf, CONV_NR);
#endif
                }
#elif CONV_PACKN == 1
                for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR)
                {
                    memcpy(inpbuf, inptr, slice_len * sizeof(inpbuf[0]));
                    memset(inpbuf + slice_len, input_zp, (CONV_NR - slice_len) * sizeof(inpbuf[0]));
                }
#endif
            }
        }
        else
        {
            const int HW0 = H0 * W0;
            const int HWi = Hi * Wi;
            int z0_ = zyx0 / HW0, yx0 = zyx0 - z0_ * HW0;
            int y0_ = yx0 / W0, x0_ = yx0 - y0_ * W0;
            int Cg_aligned = (Cg + CONV_PACKN -1) / CONV_PACKN * CONV_PACKN;

            for (int k = 0; k < ksize; k++)
            {
                int dz = dhwTab[k * 3], dy = dhwTab[k * 3 + 1], dx = dhwTab[k * 3 + 2];
                int i = 0, z0 = z0_, y0 = y0_, x0 = x0_;

                for (; i < CONV_NR;)
                {
                    int8_t* inpbuf_ki = inpbuf + k * CONV_NR * Cg_aligned + i * CONV_PACKN;
                    int zi = z0 * stride_d + dz - pad_front;
                    int yi = y0 * stride_h + dy - pad_top;
                    int xi = x0 * stride_w + dx - pad_left;

                    if ((unsigned) zi < (unsigned) Di && (unsigned) yi < (unsigned) Hi &&
                        (unsigned) xi < (unsigned) Wi)
                    {
                        const int8_t *inptr_ki = inptr + zi * HWi + yi * Wi + xi;
                        if (i + 8 <= CONV_NR && x0 + 8 <= W0 && xi + stride_w * 8 <= Wi)
                        {
                            if (stride_w == 1)
                            {
#if CONV_PACKN == 4 || CONV_PACKN == 8
                                int c = 0;
                                for (; c <= Cg - CONV_PACKN; c += CONV_PACKN, inptr_ki += CONV_PACKN * inp_planesize,
                                        inpbuf_ki += CONV_PACKN * CONV_NR)
                                {
                                    store_block_layout_8(inptr_ki, inpbuf_ki, inp_planesize);
                                }

                                if (c < Cg)
                                {
                                    memset(bufferPtr, input_zp, buffer.size() * sizeof(char));
                                    for (int i = 0; c < Cg; c++, i++)
                                    {
                                        memcpy(bufferPtr + i * CONV_NR, inptr_ki + i * inp_planesize, 8 * sizeof(char));
                                    }
                                    store_block_layout_8(bufferPtr, inpbuf_ki, CONV_NR);
                                }
#elif CONV_PACKN == 1
                                for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                {
                                    const int* inptr_ki_int = (const int*)inptr_ki;
                                    int* inpbuf_ki_int = (int*)inpbuf_ki;

                                    int t0 = inptr_ki_int[0];
                                    int t1 = inptr_ki_int[1];

                                    inpbuf_ki_int[0] = t0;
                                    inpbuf_ki_int[1] = t1;
                                }
#endif
                            }
                            else
                            {
#if CONV_PACKN == 4 || CONV_PACKN == 8
                                int c = 0;
                                for (; c <= Cg - CONV_PACKN; c += CONV_PACKN, inptr_ki += CONV_PACKN * inp_planesize, inpbuf_ki += CONV_PACKN * CONV_NR)
                                {
                                    PACK_I8_LOAD8(0, stride_w, inptr_ki);
                                    PACK_I8_LOAD8(1, stride_w, inptr_ki + inp_planesize);
                                    PACK_I8_LOAD8(2, stride_w, inptr_ki + inp_planesize*2);
                                    PACK_I8_LOAD8(3, stride_w, inptr_ki + inp_planesize*3);

                                    #if CONV_PACKN == 8
                                    PACK_I8_LOAD8(4, stride_w, inptr_ki + inp_planesize * 4);
                                    PACK_I8_LOAD8(5, stride_w, inptr_ki + inp_planesize * 5);
                                    PACK_I8_LOAD8(6, stride_w, inptr_ki + inp_planesize * 6);
                                    PACK_I8_LOAD8(7, stride_w, inptr_ki + inp_planesize * 7);
                                    #endif

                                    #if CONV_PACKN == 4
                                    PACK4_I8_STORE4(0, inpbuf_ki);
                                    PACK4_I8_STORE4(1, inpbuf_ki);
                                    PACK4_I8_STORE4(2, inpbuf_ki);
                                    PACK4_I8_STORE4(3, inpbuf_ki);
                                    PACK4_I8_STORE4(4, inpbuf_ki);
                                    PACK4_I8_STORE4(5, inpbuf_ki);
                                    PACK4_I8_STORE4(6, inpbuf_ki);
                                    PACK4_I8_STORE4(7, inpbuf_ki);
                                    #elif CONV_PACKN == 8
                                    PACK8_I8_STORE8(0, inpbuf_ki);
                                    PACK8_I8_STORE8(1, inpbuf_ki);
                                    PACK8_I8_STORE8(2, inpbuf_ki);
                                    PACK8_I8_STORE8(3, inpbuf_ki);
                                    PACK8_I8_STORE8(4, inpbuf_ki);
                                    PACK8_I8_STORE8(5, inpbuf_ki);
                                    PACK8_I8_STORE8(6, inpbuf_ki);
                                    PACK8_I8_STORE8(7, inpbuf_ki);
                                    #endif
                                }

                                if (c < Cg)
                                {
                                    const int8_t* inptr0 = inptr_ki;
                                    const int8_t* inptr1 = c+1 < Cg ? inptr_ki + inp_planesize : zbufPtr;
                                    const int8_t* inptr2 = c+2 < Cg ? inptr_ki + inp_planesize*2 : zbufPtr;
                                    const int8_t* inptr3 = c+3 < Cg ? inptr_ki + inp_planesize*3 : zbufPtr;

                                    PACK_I8_LOAD8(0, stride_w, inptr0);
                                    PACK_I8_LOAD8(1, stride_w, inptr1);
                                    PACK_I8_LOAD8(2, stride_w, inptr2);
                                    PACK_I8_LOAD8(3, stride_w, inptr3);

                                    #if CONV_PACKN == 8
                                    const int8_t* inptr4 = c+4 < Cg ? inptr_ki + inp_planesize*4 : zbufPtr;
                                    const int8_t* inptr5 = c+5 < Cg ? inptr_ki + inp_planesize*5 : zbufPtr;
                                    const int8_t* inptr6 = c+6 < Cg ? inptr_ki + inp_planesize*6 : zbufPtr;
                                    const int8_t* inptr7 = c+7 < Cg ? inptr_ki + inp_planesize*7 : zbufPtr;

                                    PACK_I8_LOAD8(4, stride_w, inptr4);
                                    PACK_I8_LOAD8(5, stride_w, inptr5);
                                    PACK_I8_LOAD8(6, stride_w, inptr6);
                                    PACK_I8_LOAD8(7, stride_w, inptr7);
                                    #endif

                                    #if CONV_PACKN == 4
                                    PACK4_I8_STORE4(0, inpbuf_ki);
                                    PACK4_I8_STORE4(1, inpbuf_ki);
                                    PACK4_I8_STORE4(2, inpbuf_ki);
                                    PACK4_I8_STORE4(3, inpbuf_ki);
                                    PACK4_I8_STORE4(4, inpbuf_ki);
                                    PACK4_I8_STORE4(5, inpbuf_ki);
                                    PACK4_I8_STORE4(6, inpbuf_ki);
                                    PACK4_I8_STORE4(7, inpbuf_ki);
                                    #elif CONV_PACKN == 8
                                    PACK8_I8_STORE8(0, inpbuf_ki);
                                    PACK8_I8_STORE8(1, inpbuf_ki);
                                    PACK8_I8_STORE8(2, inpbuf_ki);
                                    PACK8_I8_STORE8(3, inpbuf_ki);
                                    PACK8_I8_STORE8(4, inpbuf_ki);
                                    PACK8_I8_STORE8(5, inpbuf_ki);
                                    PACK8_I8_STORE8(6, inpbuf_ki);
                                    PACK8_I8_STORE8(7, inpbuf_ki);
                                    #endif
                                }
#elif CONV_PACKN == 1
                                for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                {
                                    int8_t t0 = inptr_ki[0], t1 = inptr_ki[stride_w];
                                    int8_t t2 = inptr_ki[stride_w * 2], t3 = inptr_ki[stride_w * 3];
                                    int8_t t4 = inptr_ki[stride_w * 4], t5 = inptr_ki[stride_w * 5];
                                    int8_t t6 = inptr_ki[stride_w * 6], t7 = inptr_ki[stride_w * 7];
                                    inpbuf_ki[0] = t0;
                                    inpbuf_ki[1] = t1;
                                    inpbuf_ki[2] = t2;
                                    inpbuf_ki[3] = t3;
                                    inpbuf_ki[4] = t4;
                                    inpbuf_ki[5] = t5;
                                    inpbuf_ki[6] = t6;
                                    inpbuf_ki[7] = t7;
                                }
#endif
                            }
                            i += 8;
                            x0 += 8;
                        }
                        else
                        {
#if CONV_PACKN == 4 || CONV_PACKN == 8
                            int c = 0;
                            for (; c <= Cg - CONV_PACKN; c += CONV_PACKN, inptr_ki += CONV_PACKN * inp_planesize, inpbuf_ki += CONV_PACKN * CONV_NR)
                            {
                                int8_t t0 = inptr_ki[0];
                                int8_t t1 = inptr_ki[inp_planesize];
                                int8_t t2 = inptr_ki[inp_planesize*2];
                                int8_t t3 = inptr_ki[inp_planesize*3];
                                inpbuf_ki[0] = t0;
                                inpbuf_ki[1] = t1;
                                inpbuf_ki[2] = t2;
                                inpbuf_ki[3] = t3;

                                #if CONV_PACKN == 8
                                int8_t t4 = inptr_ki[inp_planesize*4];
                                int8_t t5 = inptr_ki[inp_planesize*5];
                                int8_t t6 = inptr_ki[inp_planesize*6];
                                int8_t t7 = inptr_ki[inp_planesize*7];
                                inpbuf_ki[4] = t4;
                                inpbuf_ki[5] = t5;
                                inpbuf_ki[6] = t6;
                                inpbuf_ki[7] = t7;
                                #endif
                            }

                            if (c < Cg)
                            {
                                int8_t t0 = inptr_ki[0];
                                int8_t t1 = c + 1 < Cg ? inptr_ki[inp_planesize  ] : input_zp;
                                int8_t t2 = c + 2 < Cg ? inptr_ki[inp_planesize*2] : input_zp;
                                int8_t t3 = c + 3 < Cg ? inptr_ki[inp_planesize*3] : input_zp;
                                inpbuf_ki[0] = t0;
                                inpbuf_ki[1] = t1;
                                inpbuf_ki[2] = t2;
                                inpbuf_ki[3] = t3;

                                #if CONV_PACKN == 8
                                int8_t t4 = c + 4 < Cg ? inptr_ki[inp_planesize*4] : input_zp;
                                int8_t t5 = c + 5 < Cg ? inptr_ki[inp_planesize*5] : input_zp;
                                int8_t t6 = c + 6 < Cg ? inptr_ki[inp_planesize*6] : input_zp;
                                int8_t t7 = c + 7 < Cg ? inptr_ki[inp_planesize*7] : input_zp;
                                inpbuf_ki[4] = t4;
                                inpbuf_ki[5] = t5;
                                inpbuf_ki[6] = t6;
                                inpbuf_ki[7] = t7;
                                #endif
                            }
#elif CONV_PACKN == 1
                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                *inpbuf_ki = *inptr_ki;
#endif
                            i++;
                            x0++;
                        }
                    }
                    else
                    {
#if CONV_PACKN == 4 || CONV_PACKN == 8
                        for (int c = 0; c < Cg; c += CONV_PACKN, inpbuf_ki += CONV_NR * CONV_PACKN)
                        {
                            inpbuf_ki[0] = inpbuf_ki[1] = inpbuf_ki[2] = inpbuf_ki[3] = input_zp;

                            #if CONV_PACKN == 8
                            inpbuf_ki[4] = inpbuf_ki[5] = inpbuf_ki[6] = inpbuf_ki[7] = input_zp;
                            #endif
                        }
#elif CONV_PACKN == 1
                        for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR)
                            inpbuf_ki[0] = input_zp;
#endif
                        i++;
                        x0++;
                    }

                    int mask = x0 >= W0;
                    y0 += mask;
                    x0 &= mask - 1;

                    mask = y0 >= H0;
                    z0 += mask;
                    y0 &= mask - 1;
                }
            }
        }
    }
}


void runFastQConv(InputArray _input, OutputArray _output, const Ptr<FastQConv>& conv, int ntasks,
                  const Ptr<ActivationLayerInt8>& actLayer)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    int conv_dim = conv->conv_dim;

    CV_Assert_N(input.dims == output.dims,
                input.size[0] == output.size[0],
                conv->C == input.size[1],
                conv->K == output.size[1],
                input.type() == output.type(),
                input.isContinuous(),
                output.isContinuous());

    ActivationLayerInt8* activ_INT8 = nullptr;
    Mat activationLUT;
    if (actLayer)
    {
        Ptr<ActivationLayerInt8> activ_Instance = actLayer.dynamicCast<ActivationLayerInt8>();

        if (!activ_Instance.empty())
        {
            activ_INT8 = activ_Instance.get();
            if (!activ_Instance->blobs.empty())
                activationLUT = activ_Instance->blobs[0];
        }
    }

    if (conv->conv_type == QCONV_TYPE_DEPTHWISE)
    {
        // Depthwise-Convolution layer should not be followed by Add layer.
        CV_Assert(conv_dim == CONV_2D);
        return runDepthwise(input, output, conv, activ_INT8, activationLUT);
    }

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);

    CV_Assert(inputShape.size() == outputShape.size());

    int N = inputShape[0], C = inputShape[1];

    // input shape: [N, C, D, H, W] for Conv3D, [N, C, H, W] for Conv2D, [N, C, W] for Conv1D.
    int Di = conv_dim == CONV_3D ? inputShape[2] : 1;
    int Hi = conv_dim == CONV_1D ? 1 : inputShape[inputShape.size() - 2];
    int Wi = inputShape[inputShape.size() - 1];

    int ngroups = conv->ngroups;
    int K = conv->K, Dk = conv->Dk, Hk = conv->Hk, Wk = conv->Wk;

    int D0 = conv_dim == CONV_3D ? outputShape[2] : 1;
    int H0 = conv_dim == CONV_1D ? 1 : outputShape[outputShape.size() - 2];
    int W0 = outputShape[outputShape.size() - 1];

    int Cg = C/ngroups, Kg = K/ngroups;

    const size_t inp_planesize = (size_t)Di*Hi*Wi;
    const size_t out_planesize = (size_t)D0*H0*W0;

    int pad_front = conv->pad_front;
    int pad_top = conv->pad_top;
    int pad_left = conv->pad_left;

    int stride_d = conv->stride_d, stride_h = conv->stride_h, stride_w = conv->stride_w;
    int dilation_d = conv->dilation_d, dilation_h = conv->dilation_h, dilation_w = conv->dilation_w;

    int ksize = Dk*Hk*Wk;
    bool fast_1x1 = ksize == 1 && stride_d == 1 && stride_w == 1 && stride_h == 1
            && pad_front == 0 && pad_left == 0 && pad_top == 0;

    std::vector<int> ofstab_(Hk*Wk*Dk*4, 0);
    int* ofstab = ofstab_.data();
    int* dhwTab = ofstab + Hk*Wk*Dk;

    if (conv_dim == CONV_1D)
    {
        for( int w = 0; w < Wk; w++)
        {
            int dw = w*dilation_w;
            dhwTab[w*3+2] = dw;
            ofstab[w] = dw;
        }
    }
    else if (conv_dim == CONV_2D)
    {
        for (int h = 0; h < Hk; h++)
            for( int w = 0; w < Wk; w++)
            {
                int k = h*Wk + w;
                int dh = h*dilation_h, dw = w*dilation_w;
                dhwTab[k*3+1] = dh;
                dhwTab[k*3+2] = dw;
                ofstab[k] = dh*Wi + dw;
            }
    }
    else
    {
        for (int d = 0; d < Dk; d++)
            for (int h = 0; h < Hk; h++)
            {
                for (int w = 0; w < Wk; w++)
                {
                    int k = d*Hk*Wk + h*Wk + w;
                    int dd = d*dilation_d, dh = h*dilation_h, dw = w*dilation_w;
                    dhwTab[k*3] = dd;
                    dhwTab[k*3+1] = dh;
                    dhwTab[k*3+2] = dw;
                    ofstab[k] = dd*Hi*Wi + dh*Wi + dw;
                }
            }
    }

    int MAX_STRIPES = (56 + CONV_NR - 1)/CONV_NR;

    // Friendly to L1 cache
    const int K_BLOCK_SIZE = 32;   // Must be a multiple of 4.
    const int C_BLOCK_SIZE = 1024; // Must be a multiple of 4 and 8 (CONV_NPACK).

    int Kg_nblocks = (Kg + CONV_MR-1)/CONV_MR, Kg_aligned = Kg_nblocks * CONV_MR;
    int stripes_per_sample0 = ((int)out_planesize + CONV_NR - 1) / CONV_NR;
    int stripes_per_sample = stripes_per_sample0;

    if (stripes_per_sample < ntasks * 4)
    {
        MAX_STRIPES = 1;
        stripes_per_sample = 1;
    }
    else
        Kg_nblocks = 1;

    int Kstripes = Kg_nblocks*stripes_per_sample;
    int nsubtasks = N*ngroups*Kstripes; // total subtask number.

    // Currently, since we do not support the block layout (NC4HW4), to use the NEON instruction of vdotq_laneq_s32,
    // we need to do the convert of NCHW to NC4HW4 at data packing stage. So, only seperate im2col is used here.
    // When we totally support the block layout, we should change the following code.

    int Cg_strips = (Cg + CONV_PACKN - 1) / CONV_PACKN;
    int Cg_aligned = Cg_strips * CONV_PACKN;

    int DkHkWkCg_aligned = Dk*Hk*Wk*Cg_aligned;
    size_t stripesize = alignSize(CONV_NR * ksize * Cg_aligned, VEC_ALIGN);
    size_t cbufsize = alignSize(CONV_NR * (K_BLOCK_SIZE) * MAX_STRIPES, VEC_ALIGN);

    size_t taskbufsize = cbufsize;
    size_t all_taskbufsize = cbufsize * ntasks;
    size_t totalbufsize = N * ngroups * stripes_per_sample0 * stripesize;

    AutoBuffer<char> inpbuf_pack_;
    AutoBuffer<int> all_taskbuf_;

    inpbuf_pack_.allocate(totalbufsize + VEC_ALIGN * sizeof(char ));
    char* inpbuf_pack = alignPtr(inpbuf_pack_.data(), (int)(VEC_ALIGN * sizeof(char )));

    all_taskbuf_.allocate(all_taskbufsize + VEC_ALIGN);
    int* all_taskbuf = alignPtr(all_taskbuf_.data(), (int)(VEC_ALIGN * sizeof(int ))); // used for input im2col and packing.

    char* inp = input.ptr<char>();
    char* out = output.ptr<char>();

    const int8_t* lutptr_ = !activationLUT.empty() ? activationLUT.ptr<int8_t>() : 0;
    const char inp_zp = conv->input_zp;

    const size_t in_wstep = input.step1();
    const size_t out_wstep = output.step1();

    const int all_pack_task = N*ngroups*Cg_strips;
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        if (fast_1x1 && all_pack_task >= ntasks)
        {
            int ngc0 = task_id*all_pack_task/ntasks, ngc1 = (task_id+1)*all_pack_task/ntasks, dc = 0;

            for (; ngc0 < ngc1; ngc0 += dc)
            {
                int n = ngc0/(ngroups*Cg_strips), gc = ngc0 - n*ngroups*Cg_strips;
                int g = gc / Cg_strips;
                int s = gc - g * Cg_strips;

                dc = Cg_strips - s <= ngc1 - ngc0 ? Cg_strips - s : ngc1 - ngc0;

                int c0 = s * CONV_PACKN;
                int c1 = dc * CONV_PACKN < Cg ?  dc * CONV_PACKN : Cg;
                char * inptr_ = inp + n * in_wstep + (g * Cg + c0) * inp_planesize;
                char* inpbuf_ = inpbuf_pack + (n*ngroups + g) * stripes_per_sample0 * stripesize + c0 * CONV_NR;

                packInputData(inpbuf_, inptr_, ofstab, dhwTab, 0, out_planesize, ksize, stride_d, stride_h,
                              stride_w, pad_front, pad_top, pad_left, Di, Hi, Wi, H0, W0, c1, stripesize, 0,
                              inp_planesize, fast_1x1, inp_zp);
            }
        }
        else
        {
            const int allTasks = N * ngroups * stripes_per_sample0;
            int ngs0 = task_id*allTasks/ntasks, ngs1 = (task_id+1)*allTasks/ntasks, ds = 0;

            for (; ngs0 < ngs1; ngs0 += ds)
            {
                int n = ngs0 / (ngroups * stripes_per_sample0), gs0 = ngs0 - n*ngroups * stripes_per_sample0;
                int g = gs0 / stripes_per_sample0, s0 = gs0 - g * stripes_per_sample0;

                ds = stripes_per_sample0 - s0 <= ngs1 - ngs0 ? stripes_per_sample0 - s0 : ngs1 - ngs0;

                int zyx = s0 * CONV_NR;
                int zyx_limit = (s0 + ds) * CONV_NR < out_planesize ? (s0 + ds) * CONV_NR : out_planesize;

                char * inptr_ = inp + n * in_wstep + g * Cg * inp_planesize;
                char* inpbuf_ = inpbuf_pack + (n * ngroups + g) * stripes_per_sample0 * stripesize + s0 * stripesize;

                packInputData(inpbuf_, inptr_, ofstab, dhwTab, zyx, zyx_limit, ksize, stride_d, stride_h,
                              stride_w, pad_front, pad_top, pad_left, Di, Hi, Wi, H0, W0, Cg, stripesize, 0,
                              inp_planesize, fast_1x1, inp_zp);
            }
        }
    }
    });

    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        int* cbuf_task = all_taskbuf + taskbufsize * task_id;

        int ngs0 = (int)((size_t)nsubtasks * task_id / ntasks);
        int ngs1 = (int)((size_t)nsubtasks * (task_id+1) / ntasks);

        for (int subtask = ngs0; subtask < ngs1; )
        {
            int ng = subtask / Kstripes;
            int kzyx0 = subtask - ng * Kstripes;
            int kzyx1 = kzyx0 + (ngs1 - subtask);
            int n = ng / ngroups, g = ng % ngroups;

            kzyx1 = kzyx1 <= Kstripes ? kzyx1 : Kstripes;
            subtask += kzyx1 - kzyx0;
            int k0, k1;
            int zyx0, zyx_limit, zyx_block_limit = 0;

            if (stripes_per_sample == 1)
            {
                k0 = kzyx0 * CONV_MR;
                k1 = kzyx1 * CONV_MR;
                k1 = k1 <= Kg ? k1 : Kg;
                zyx0 = 0;
                zyx_limit = (int)out_planesize;
            }
            else
            {
                k0 = 0;
                k1 = Kg;
                zyx0 = kzyx0 * CONV_NR;
                zyx_limit = kzyx1 * CONV_NR;
                zyx_limit = zyx_limit < out_planesize ? zyx_limit : (int)out_planesize;
            }

            for (; zyx0 < zyx_limit; zyx0 = zyx_block_limit)
            {
                // step 1. extract part of input tensor and represent it in zigzag form
                zyx_block_limit = zyx0 + CONV_NR * MAX_STRIPES;
                zyx_block_limit = zyx_block_limit < zyx_limit ? zyx_block_limit : zyx_limit;

                int nstripes = (zyx_block_limit - zyx0 + CONV_NR - 1) / CONV_NR;

                CV_Assert(nstripes <= MAX_STRIPES);

                char *weights = conv->weightsBufPtr + g * Kg_aligned * DkHkWkCg_aligned;
                float* multiplier = conv->outputMultiplier.data() + Kg * g;;

                const int *biasptr = conv->biasBuf.data() + Kg * g;
                int ldc = nstripes * CONV_NR;

                // 2. do convolution, compute Kg x (zyx_block_limit - zyx0) part of the output tensor
                int out_width = zyx_block_limit - zyx0;
                for (int k0_block = k0; k0_block < k1; k0_block += K_BLOCK_SIZE)
                {
                    int k1_block = k0_block + K_BLOCK_SIZE < k1 ? k0_block + K_BLOCK_SIZE : k1;
                    for (int c0 = 0; c0 < DkHkWkCg_aligned; c0 += C_BLOCK_SIZE)
                    {
                        int c1 = c0 + C_BLOCK_SIZE < DkHkWkCg_aligned ? c0 + C_BLOCK_SIZE : DkHkWkCg_aligned;
                        const char *inptr = inpbuf_pack + (ng * stripes_per_sample0 + zyx0/CONV_NR)*stripesize + c0 * CONV_NR;

                        for (int stripe = 0; stripe < nstripes; stripe++, inptr += stripesize)
                        {
                            const int outLen = std::min(out_width - stripe * CONV_NR, CONV_NR);
                            const char *wptr = weights + k0_block * DkHkWkCg_aligned + c0 * CONV_MR;

                            int *cptr = cbuf_task + stripe * CONV_NR;

                            for (int k = k0_block; k < k1_block; k += CONV_MR,
                                    wptr += DkHkWkCg_aligned * CONV_MR, cptr += CONV_MR * ldc)
                            {
                                // TODO add AVX/AVX2 compute branch.
#if CV_NEON
                                if (conv->useNEON && CONV_PACKN == 4)
                                    opt_NEON::convBlock_INT8(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                                else
#endif
                                convBlock_INT8(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                            }
                        }
                    }

                    size_t outofs = n * out_wstep + (g * Kg + k0_block) * out_planesize + zyx0;
                    const int *cptr = cbuf_task;
                    int8_t *outptr = (int8_t *)out + outofs;

                    for (int k = k0_block; k < k1_block; k++,
                            cptr += ldc, outptr += out_planesize)
                    {
                        float biasval = (float)biasptr[k];
                        float mult = multiplier[k];

                        int j = 0;
#if CV_SIMD128
                        v_float32x4 vbias = v_setall_f32(biasval);
                        v_float32x4 vmult = v_setall_f32(mult);

                        v_int32x4 voutzp = v_setall_s32(conv->output_zp);
                        v_int32x4 outmin = v_setall_s32(-128), outmax = v_setall_s32(127);

                        for (; j + 15 < out_width; j += 16)
                        {

                            v_float32x4 v0 = v_add(v_cvt_f32(v_load(cptr + j)), vbias);
                            v_float32x4 v1 = v_add(v_cvt_f32(v_load(cptr + j + 4)), vbias);
                            v_float32x4 v2 = v_add(v_cvt_f32(v_load(cptr + j + 8)), vbias);
                            v_float32x4 v3 = v_add(v_cvt_f32(v_load(cptr + j + 12)), vbias);

                            v0 = v_mul(vmult, v0);
                            v1 = v_mul(vmult, v1);
                            v2 = v_mul(vmult, v2);
                            v3 = v_mul(vmult, v3);

                            v_int32x4 vv0 = v_add(v_round(v0), voutzp);
                            v_int32x4 vv1 = v_add(v_round(v1), voutzp);
                            v_int32x4 vv2 = v_add(v_round(v2), voutzp);
                            v_int32x4 vv3 = v_add(v_round(v3), voutzp);

                            vv0 = v_min(v_max(vv0, outmin), outmax);
                            vv1 = v_min(v_max(vv1, outmin), outmax);
                            vv2 = v_min(v_max(vv2, outmin), outmax);
                            vv3 = v_min(v_max(vv3, outmin), outmax);

                            v_store(outptr + j, v_pack(v_pack(vv0, vv1), v_pack(vv2, vv3)));
                        }
#endif
                        for (; j < out_width; j++)
                        {
                            float v = (cptr[j] + biasval) * mult + conv->output_zp;
                            outptr[j] = (int8_t)std::min(std::max((int)std::round(v), -128), 127);
                        }

                        if (activ_INT8 && lutptr_)
                            activ_INT8->forwardSlice(outptr, lutptr_, outptr, out_width,
                                                     out_planesize, Kg * g + k, Kg * g + k + 1);
                    }
                }
            }
        }
    }
    });
}


/****************************************************************************************\
                                    SIMD and no-SIMD code for convBlock
\****************************************************************************************/
void convBlock_INT8(int np, const char* _a, const char* _b, int* c, int ldc, bool init_c, const int outLen,
               const int convMR, const int convNR)
{
    const int8_t* a = (const int8_t*)_a;
    const int8_t* b = (const int8_t*)_b;

    CV_Assert(convMR == 4 && (convNR == 24 || convNR == 12));
#if CV_SIMD128 && CONV_PACKN == 4
    if (outLen > 12)
    {
        v_int32x4 c00  = v_setzero_s32(), c01 = c00, c02 = c00, c03 = c00, c04 = c00, c05 = c00;
        v_int32x4 c10  = v_setzero_s32(), c11 = c10, c12 = c10, c13 = c10, c14 = c10, c15 = c10;
        v_int32x4 c20  = v_setzero_s32(), c21 = c00, c22 = c20, c23 = c20, c24 = c20, c25 = c20;
        v_int32x4 c30  = v_setzero_s32(), c31 = c00, c32 = c30, c33 = c30, c34 = c30, c35 = c30;

        for (int p = 0; p < np; p += CONV_PACKN, a += CONV_PACKN * convMR, b += CONV_PACKN * convNR)
        {
            v_int8x16 b0 = v_load(b);
            v_int8x16 b1 = v_load(b + 16);
            v_int8x16 b2 = v_load(b + 32);

            v_int8x16 a0_8, a1_8, a2_8, a3_8;
            v_int32x4 a0_32, a1_32;
            v_int16x8 a0 = v_load_expand(a);
            v_expand(a0, a0_32, a1_32);

            v_int16x8 a0_16 = v_pack(a0_32, a0_32);
            a0_8 = v_pack(a0_16, a0_16);

            v_int16x8 a1_16 = v_pack(a1_32, a1_32);
            a1_8 = v_pack(a1_16, a1_16);

            a0 = v_load_expand(a + 8);
            v_expand(a0, a0_32, a1_32);

            a0_16 = v_pack(a0_32, a0_32);
            a2_8 = v_pack(a0_16, a0_16);

            a1_16 = v_pack(a1_32, a1_32);
            a3_8 = v_pack(a1_16, a1_16);

            c00 = v_dotprod_expand(a0_8, b0, c00);
            c01 = v_dotprod_expand(a0_8, b1, c01);
            c02 = v_dotprod_expand(a0_8, b2, c02);
            c10 = v_dotprod_expand(a1_8, b0, c10);
            c11 = v_dotprod_expand(a1_8, b1, c11);
            c12 = v_dotprod_expand(a1_8, b2, c12);
            c20 = v_dotprod_expand(a2_8, b0, c20);
            c21 = v_dotprod_expand(a2_8, b1, c21);
            c22 = v_dotprod_expand(a2_8, b2, c22);
            c30 = v_dotprod_expand(a3_8, b0, c30);
            c31 = v_dotprod_expand(a3_8, b1, c31);
            c32 = v_dotprod_expand(a3_8, b2, c32);

            b0 = v_load(b + 48); b1 = v_load(b + 64); b2 = v_load(b + 80);

            c03 = v_dotprod_expand(a0_8, b0, c03);
            c04 = v_dotprod_expand(a0_8, b1, c04);
            c05 = v_dotprod_expand(a0_8, b2, c05);
            c13 = v_dotprod_expand(a1_8, b0, c13);
            c14 = v_dotprod_expand(a1_8, b1, c14);
            c15 = v_dotprod_expand(a1_8, b2, c15);
            c23 = v_dotprod_expand(a2_8, b0, c23);
            c24 = v_dotprod_expand(a2_8, b1, c24);
            c25 = v_dotprod_expand(a2_8, b2, c25);
            c33 = v_dotprod_expand(a3_8, b0, c33);
            c34 = v_dotprod_expand(a3_8, b1, c34);
            c35 = v_dotprod_expand(a3_8, b2, c35);
        }

        if (!init_c)
        {
#undef SIMD_UPDATE_QCONV_BLOCK
#define SIMD_UPDATE_QCONV_BLOCK(i) \
        c##i##0 = v_add(c##i##0, v_load(c+i*ldc)); \
        c##i##1 = v_add(c##i##1, v_load(c+i*ldc+4)); \
        c##i##2 = v_add(c##i##2, v_load(c+i*ldc+8)); \
        c##i##3 = v_add(c##i##3, v_load(c+i*ldc+12)); \
        c##i##4 = v_add(c##i##4, v_load(c+i*ldc+16)); \
        c##i##5 = v_add(c##i##5, v_load(c+i*ldc+20));

            SIMD_UPDATE_QCONV_BLOCK(0);
            SIMD_UPDATE_QCONV_BLOCK(1);
            SIMD_UPDATE_QCONV_BLOCK(2);
            SIMD_UPDATE_QCONV_BLOCK(3);
        }

#undef SIMD_UPDATE_QCONV_BLOCK
#define SIMD_UPDATE_QCONV_BLOCK(i) \
        v_store(c+i*ldc, c##i##0); \
        v_store(c+i*ldc+4, c##i##1); \
        v_store(c+i*ldc+8, c##i##2); \
        v_store(c+i*ldc+12, c##i##3); \
        v_store(c+i*ldc+16, c##i##4); \
        v_store(c+i*ldc+20, c##i##5);

        SIMD_UPDATE_QCONV_BLOCK(0);
        SIMD_UPDATE_QCONV_BLOCK(1);
        SIMD_UPDATE_QCONV_BLOCK(2);
        SIMD_UPDATE_QCONV_BLOCK(3);
    }
    else
    {
        v_int32x4 c00  = v_setzero_s32(), c01 = c00, c02 = c00;
        v_int32x4 c10  = v_setzero_s32(), c11 = c10, c12 = c10;
        v_int32x4 c20  = v_setzero_s32(), c21 = c00, c22 = c20;
        v_int32x4 c30  = v_setzero_s32(), c31 = c00, c32 = c30;

        for (int p = 0; p < np; p += CONV_PACKN, a += CONV_PACKN * convMR, b += CONV_PACKN * convNR)
        {
            v_int8x16 b0 = v_load(b);
            v_int8x16 b1 = v_load(b + 16);
            v_int8x16 b2 = v_load(b + 32);

            v_int8x16 a0_8, a1_8, a2_8, a3_8;
            v_int32x4 a0_32, a1_32;
            v_int16x8 a0 = v_load_expand(a);
            v_expand(a0, a0_32, a1_32);

            v_int16x8 a0_16 = v_pack(a0_32, a0_32);
            a0_8 = v_pack(a0_16, a0_16);

            v_int16x8 a1_16 = v_pack(a1_32, a1_32);
            a1_8 = v_pack(a1_16, a1_16);

            a0 = v_load_expand(a + 8);
            v_expand(a0, a0_32, a1_32);

            a0_16 = v_pack(a0_32, a0_32);
            a2_8 = v_pack(a0_16, a0_16);

            a1_16 = v_pack(a1_32, a1_32);
            a3_8 = v_pack(a1_16, a1_16);

            c00 = v_dotprod_expand(a0_8, b0, c00);
            c01 = v_dotprod_expand(a0_8, b1, c01);
            c02 = v_dotprod_expand(a0_8, b2, c02);
            c10 = v_dotprod_expand(a1_8, b0, c10);
            c11 = v_dotprod_expand(a1_8, b1, c11);
            c12 = v_dotprod_expand(a1_8, b2, c12);
            c20 = v_dotprod_expand(a2_8, b0, c20);
            c21 = v_dotprod_expand(a2_8, b1, c21);
            c22 = v_dotprod_expand(a2_8, b2, c22);
            c30 = v_dotprod_expand(a3_8, b0, c30);
            c31 = v_dotprod_expand(a3_8, b1, c31);
            c32 = v_dotprod_expand(a3_8, b2, c32);
        }

        if (!init_c)
        {
#undef SIMD_UPDATE_QCONV_BLOCK
#define SIMD_UPDATE_QCONV_BLOCK(i)                   \
        c##i##0 = v_add(c##i##0, v_load(c+i*ldc));   \
        c##i##1 = v_add(c##i##1, v_load(c+i*ldc+4)); \
        c##i##2 = v_add(c##i##2, v_load(c+i*ldc+8))

            SIMD_UPDATE_QCONV_BLOCK(0);
            SIMD_UPDATE_QCONV_BLOCK(1);
            SIMD_UPDATE_QCONV_BLOCK(2);
            SIMD_UPDATE_QCONV_BLOCK(3);
        }

#undef SIMD_UPDATE_QCONV_BLOCK
#define SIMD_UPDATE_QCONV_BLOCK(i)   \
        v_store(c+i*ldc, c##i##0);   \
        v_store(c+i*ldc+4, c##i##1); \
        v_store(c+i*ldc+8, c##i##2)

        SIMD_UPDATE_QCONV_BLOCK(0);
        SIMD_UPDATE_QCONV_BLOCK(1);
        SIMD_UPDATE_QCONV_BLOCK(2);
        SIMD_UPDATE_QCONV_BLOCK(3);
    }
#else
    // NO_SIMD_implemantation.
    {
        std::vector<int> cbuffer(convMR * outLen, 0);
        int* cbuf = cbuffer.data();
        for (int p = 0; p < np; p += CONV_PACKN)
        {
#if CONV_PACKN == 4 || CONV_PACKN == 8
            for (int i = 0; i < convMR; i++)
            {
                int8_t ai0 = a[convMR*p + i * CONV_PACKN];
                int8_t ai1 = a[convMR*p + i * CONV_PACKN + 1];
                int8_t ai2 = a[convMR*p + i * CONV_PACKN + 2];
                int8_t ai3 = a[convMR*p + i * CONV_PACKN + 3];
                #if CONV_PACKN == 8
                int8_t ai4 = a[convMR*p + i * CONV_PACKN + 4];
                int8_t ai5 = a[convMR*p + i * CONV_PACKN + 5];
                int8_t ai6 = a[convMR*p + i * CONV_PACKN + 6];
                int8_t ai7 = a[convMR*p + i * CONV_PACKN + 7];
                #endif

                for( int j = 0; j < outLen; j++ )
                {
                    cbuf[i * outLen+j] += b[convNR*p + j * CONV_PACKN]     * ai0 +
                                          b[convNR*p + j * CONV_PACKN + 1] * ai1 +
                                          b[convNR*p + j * CONV_PACKN + 2] * ai2 +
                                          b[convNR*p + j * CONV_PACKN + 3] * ai3
                                          #if CONV_PACKN == 8
                                          +
                                          b[convNR*p + j * CONV_PACKN + 4] * ai4 +
                                          b[convNR*p + j * CONV_PACKN + 5] * ai5 +
                                          b[convNR*p + j * CONV_PACKN + 6] * ai6 +
                                          b[convNR*p + j * CONV_PACKN + 7] * ai7
                                          #endif
                                          ;
                }
            }
#elif CONV_PACKN == 1
            for( int i = 0; i < convMR; i++ )
            {
                int ai = a[convMR*p + i];
                for( int j = 0; j < outLen; j++ )
                    cbuf[i * outLen+j] += (int)b[convNR*p + j] * ai;
            }
#endif
        }

        if (!init_c)
        {
            for(int i = 0; i < convMR; i++)
            {
                for(int j = 0; j < outLen; j++)
                    c[i*ldc + j] += cbuf[i*outLen + j];
            }
        }
        else
        {
            for(int i = 0; i < convMR; i++)
            {
                for(int j = 0; j < outLen; j++)
                    c[i*ldc + j] = cbuf[i*outLen + j];
            }
        }
    }
#endif

}

}} // namespace cv::dnn
