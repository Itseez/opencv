// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "convolution.hpp"

namespace cv { namespace dnn {
enum { VEC_ALIGN = 32}; // Memory alignment.

#if CV_SIMD
static inline void v_expand_mul_add(const v_int8x16& a, const v_int8x16& b,
                                    v_int32x4& out0, v_int32x4& out1, v_int32x4& out2, v_int32x4& out3)
{
    v_int16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);

    v_int32x4 t0, t1;
    v_mul_expand(a0, b0, t0, t1);
    out0 += t0; out1 += t1;

    v_mul_expand(a1, b1, t0, t1);
    out2 += t0; out3 += t1;
}
#endif

void fastDepthwiseConv( const int8_t* wptr,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w,
                        int dilation_h, int dilation_w,
                        int pad_t, int pad_l,
                        const int* biasptr, const float* multptr,
                        const int8_t* inptr_,
                        int height, int width,
                        int8_t* outptr_,
                        int out_d, int outH, int outW,
                        int inZp, int outZp);

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastQConv>& conv, ActivationLayerInt8* activ_INT8,
                  const  Mat& activationLUT)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);

    CV_Assert(inputShape.size() == 4 && conv->conv_dim == CONV_2D && "Currently, only Conv 2D depthwise is supported.");
    CV_Assert(inputShape.size() == outputShape.size());

    int N = inputShape[0], C = inputShape[1];
    int Hi = inputShape[inputShape.size() - 2];
    int Wi = inputShape[inputShape.size() - 1];

    int K = conv->K, Hk = conv->Hk, Wk = conv->Wk;
    int H0 = outputShape[outputShape.size() - 2];
    int W0 = outputShape[outputShape.size() - 1];
    int ngroups = conv->ngroups;

    const size_t inp_planesize = (size_t) Hi * Wi;
    const size_t out_planesize = (size_t) H0 * W0;

    CV_Assert(ngroups > 1 && ngroups == K && ngroups == C);

    int stride_h = conv->stride_h, stride_w = conv->stride_w;
    int dilation_h = conv->dilation_h, dilation_w = conv->dilation_w;

    int pad_top = conv->pad_top, pad_bottom = conv->pad_bottom;
    int pad_left = conv->pad_left, pad_right = conv->pad_right;

    int ksize = Hk * Wk;

    const int VEC_NLANES = 32;
    int padded_ksize = ((ksize + VEC_NLANES-1) / VEC_NLANES) * VEC_NLANES;

    std::vector<int> ofstab_(3 * ksize, 0);
    int *ofstab = ofstab_.data();
    int *yxtab = ofstab + ksize;

    for (int k = 0; k < ksize; k++)
    {
        int y = k < ksize ? k / Wk : 0;
        int x = k < ksize ? k % Wk : 0;
        int dy = y * dilation_h, dx = x * dilation_w;
        yxtab[k * 2] = dy;
        yxtab[k * 2 + 1] = dx;
        ofstab[k] = dy * Wi + dx;
    }

    const int8_t *weights0 = (const int8_t *)conv->weightsBufPtr;
    const int *bias = conv->biasBuf.data();
    CV_Assert(ksize > 1 || (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0));

    const int8_t* lutptr_ = !activationLUT.empty() ? activationLUT.ptr<int8_t>() : 0;
    const char inp_zp = conv->input_zp;

    const int8_t* inp = input.ptr<int8_t>();
    int8_t* out = output.ptr<int8_t>();
    float* multiplier = conv->outputMultiplier.data();

    const size_t in_wstep = input.step1();
    const size_t out_wstep = output.step1();

    parallel_for_(Range(0, N * C), [&](const Range &r0) {
    for (int nc = r0.start; nc < r0.end; nc++)
    {
        int n = nc / C;
        int c = nc % C; // c is equal to group in depthwise convolution.
        const int8_t *inptr0 = inp + n * in_wstep + inp_planesize * c;
        int8_t *outptr0 = out + n * out_wstep + out_planesize * c;
        const int8_t *weights = weights0 + c * padded_ksize;

        // TODO, optimize the AVX2 branch
        fastDepthwiseConv(weights, Hk, Wk, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left,
                          bias, multiplier, inptr0, Hi, Wi, outptr0, c, H0, W0, inp_zp, conv->output_zp);

        if (activ_INT8 && lutptr_)
            activ_INT8->forwardSlice(outptr0, lutptr_, outptr0, (int) out_planesize, (int) out_planesize, c, c+1);
    }});
}

void fastDepthwiseConv( const int8_t* wptr,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w,
                        int dilation_h, int dilation_w,
                        int pad_t, int pad_l,
                        const int* biasptr, const float* multptr,
                        const int8_t* inptr_,
                        int height, int width,
                        int8_t* outptr_,
                        int out_d, int outH, int outW,
                        int inpZp, int outZp)
{
    const int w00_ = (int)wptr[0], w01_ = (int)wptr[1], w02_ = (int)wptr[2],
            w10 = (int)wptr[3], w11 = (int)wptr[4], w12 = (int)wptr[5],
            w20_ = (int)wptr[6], w21_ = (int)wptr[7], w22_ = (int)wptr[8];

    const int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    int bias = biasptr[out_d], biasCopy;
    float multiplier = multptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const int8_t* imgptr0 = inptr_ + in_i*width;
        const int8_t* imgptr1 = imgptr0 + dilation_h*width;
        const int8_t* imgptr2 = imgptr0 + (dilation_h*2)*width;

        int out, w00 = w00_, w01 = w01_, w02 = w02_;
        int w20 = w20_, w21 = w21_, w22 = w22_;

        // Bias has a fused offset component. bias = bias_quantized - input_zeropoint*sum_of_weights.
        // In some cases below, certain weights are not used for convolution or set to zero.
        // So we create a copy of bias at the start and remove the weight's components as necessary.
        biasCopy = bias;

        if (in_i < 0)
        {
            biasCopy += inpZp * (w00 + w01 + w02);
            w00 = w01 = w02 = 0;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            biasCopy += inpZp * (w20 + w21 + w22);
            w20 = w21 = w22 = 0;
            imgptr2 = imgptr1;
        }

        int8_t* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = std::round(float(imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                                   imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                                   imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + biasCopy +
                                   inpZp*(w00 + w10 + w20)) * multiplier) + outZp;

            outptr[0] = (int8_t)std::min(std::max(out, -128), 127);
            out_j = 1;
        }

#if CV_SIMD128
        const int VEC_NLANES = 16;
        if ((stride_w == 1 || (stride_w == 2 && dilation_w == 1)) && (outW1 - out_j) >= VEC_NLANES)
        {
            v_int8x16 vw00 = v_setall_s8(w00), vw01 = v_setall_s8(w01), vw02 = v_setall_s8(w02),
                    vw10 = v_setall_s8(w10), vw11 = v_setall_s8(w11), vw12 = v_setall_s8(w12),
                    vw20 = v_setall_s8(w20), vw21 = v_setall_s8(w21), vw22 = v_setall_s8(w22);

            v_int32x4 vout0, vout1, vout2, vout3;
            v_int32x4 vbias = v_setall_s32(biasCopy);
            v_float32x4 vmult = v_setall_f32(multiplier);
            v_int32x4 voutzp = v_setall_s32(outZp);
            v_int32x4 outmin = v_setall_s32(-128), outmax = v_setall_s32(127);

            if (stride_w == 1)
            {
                for (; out_j < outW1; out_j += VEC_NLANES)
                {
                    // Tail processing.
                    if (out_j > outW1 - VEC_NLANES)
                    {
                        if (out_j <= pad_l)
                            break;
                        out_j = outW1 - VEC_NLANES;
                    }

                    int in_j = out_j * stride_w - pad_l;

                    v_int8x16 v00 = v_load(imgptr0 + in_j),
                            v01 = v_load(imgptr0 + in_j + dilation_w),
                            v02 = v_load(imgptr0 + in_j + dilation_w*2),
                            v10 = v_load(imgptr1 + in_j),
                            v11 = v_load(imgptr1 + in_j + dilation_w),
                            v12 = v_load(imgptr1 + in_j + dilation_w*2),
                            v20 = v_load(imgptr2 + in_j),
                            v21 = v_load(imgptr2 + in_j + dilation_w),
                            v22 = v_load(imgptr2 + in_j + dilation_w*2);

                    vout0 = vout1 = vout2 = vout3 = vbias;
                    v_expand_mul_add(v00, vw00, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v01, vw01, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v02, vw02, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v10, vw10, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v11, vw11, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v12, vw12, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v20, vw20, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v21, vw21, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v22, vw22, vout0, vout1, vout2, vout3);

                    vout0 = v_add(voutzp, v_round(v_cvt_f32(vout0)*vmult));
                    vout1 = v_add(voutzp, v_round(v_cvt_f32(vout1)*vmult));
                    vout2 = v_add(voutzp, v_round(v_cvt_f32(vout2)*vmult));
                    vout3 = v_add(voutzp, v_round(v_cvt_f32(vout3)*vmult));

                    vout0 = v_min(v_max(vout0, outmin), outmax);
                    vout1 = v_min(v_max(vout1, outmin), outmax);
                    vout2 = v_min(v_max(vout2, outmin), outmax);
                    vout3 = v_min(v_max(vout3, outmin), outmax);

                    v_store(outptr + out_j, v_pack(v_pack(vout0, vout1), v_pack(vout2, vout3)));
                }
            }
            else // (stride_w == 2 && dilation_wdilation_w == 1)
            {
                for (; out_j < outW1; out_j += VEC_NLANES)
                {
                    // Tail processing.
                    if (out_j > outW1 - VEC_NLANES)
                    {
                        if (out_j <= pad_l)
                            break;
                        out_j = outW1 - VEC_NLANES;
                    }

                    int in_j = out_j * stride_w - pad_l;

                    v_int8x16 unused;
                    v_int8x16 v00, v01, v02,
                              v10, v11, v12,
                              v20, v21, v22;

                    v_load_deinterleave(imgptr0 + in_j, v00, v01);
                    v_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    v_load_deinterleave(imgptr1 + in_j, v10, v11);
                    v_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    v_load_deinterleave(imgptr2 + in_j, v20, v21);
                    v_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    vout0 = vout1 = vout2 = vout3 = vbias;

                    v_expand_mul_add(v00, vw00, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v01, vw01, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v02, vw02, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v10, vw10, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v11, vw11, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v12, vw12, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v20, vw20, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v21, vw21, vout0, vout1, vout2, vout3);
                    v_expand_mul_add(v22, vw22, vout0, vout1, vout2, vout3);

                    vout0 = voutzp + v_round(v_cvt_f32(vout0)*vmult);
                    vout1 = voutzp + v_round(v_cvt_f32(vout1)*vmult);
                    vout2 = voutzp + v_round(v_cvt_f32(vout2)*vmult);
                    vout3 = voutzp + v_round(v_cvt_f32(vout3)*vmult);

                    vout0 = v_min(v_max(vout0, outmin), outmax);
                    vout1 = v_min(v_max(vout1, outmin), outmax);
                    vout2 = v_min(v_max(vout2, outmin), outmax);
                    vout3 = v_min(v_max(vout3, outmin), outmax);

                    v_store(outptr + out_j, v_pack(v_pack(vout0, vout1), v_pack(vout2, vout3)));
                }
            }
        }
#endif

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = std::round(float(imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                                   imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                                   imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + biasCopy)
                             * multiplier) + outZp;
            outptr[out_j] = (int8_t)std::min(std::max(out, -128), 127);
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            int s0 = 1, s1 = 1, s2 = 1;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0;
                biasCopy += inpZp*(w00 + w10 + w20);
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0;
                biasCopy += inpZp*(w01 + w11 + w21);
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0;
                biasCopy += inpZp*(w02 + w12 + w22);
            }

            out = std::round(float(imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                                   imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                                   imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + biasCopy) * multiplier) + outZp;

            outptr[out_j] = (int8_t)std::min(std::max(out, -128), 127);
        }
    }
}

}} // namespace cv::dnn
