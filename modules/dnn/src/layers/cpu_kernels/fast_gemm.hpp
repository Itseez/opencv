// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#ifndef OPENCV_DNN_FAST_GEMM_HPP
#define OPENCV_DNN_FAST_GEMM_HPP

#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

struct FastGemmOpt {
    bool use_avx;
    bool use_avx2;
    bool use_neon;
    bool use_lasx;

    FastGemmOpt() {
        use_avx = false;
        use_avx2 = false;
        use_neon = false;
        use_lasx = false;
    }

    void init() {
        use_avx = checkHardwareSupport(CPU_AVX);
        use_avx2 = checkHardwareSupport(CPU_AVX2);
        use_neon = checkHardwareSupport(CPU_NEON);
        use_lasx = checkHardwareSupport(CPU_LASX);
    }

    bool all() {
        return use_avx || use_avx2 || use_neon || use_lasx;
    }
};

struct MatMulHelper {
    std::vector<size_t> A_offsets;
    std::vector<size_t> B_offsets;
    std::vector<size_t> C_offsets;
    size_t batch;

    int lda0, lda1;
    int ldb0, ldb1;
    int ldc;

    int M, N, K;

    void compute(bool trans_a, bool trans_b, MatShape A_shape, MatShape B_shape, MatShape C_shape) {
        auto A_ndims = A_shape.size(), B_ndims = B_shape.size(), C_ndims =  C_shape.size();
        int ma = A_shape[A_ndims - 2], na = A_shape.back();
        int mb = B_shape[B_ndims - 2], nb = B_shape.back();
        lda0 = na, lda1 = 1;
        ldb0 = nb, ldb1 = 1;
        ldc = C_shape.back();

        M = trans_a ? na : ma;
        N = trans_b ? mb : nb;
        K = trans_a ? ma : na;

        if (trans_a) {
            std::swap(lda0, lda1);
        }
        if (trans_b) {
            std::swap(ldb0, ldb1);
        }

        // compute offsets
        auto batch_ndims = C_ndims - 2;

        batch = total(C_shape, 0, batch_ndims);

        A_offsets.resize(batch, 0);
        B_offsets.resize(batch, 0);
        C_offsets.resize(batch, 0);

        // build C_offsets
        size_t C_step = total(C_shape, C_ndims - 2);

        MatShape A_broadcast_shape(C_ndims, 1);
        std::memcpy(A_broadcast_shape.data() + (C_ndims - A_ndims), A_shape.data(), A_ndims * sizeof(int));
        MatShape B_broadcast_shape(C_shape.size(), 1);
        std::memcpy(B_broadcast_shape.data() + (C_ndims - B_ndims), B_shape.data(), B_shape.size() * sizeof(int));
        std::vector<size_t> A_steps(C_ndims, 1), B_steps(C_ndims, 1);
        for (int i = C_ndims - 2; i >= 0; i--) {
            A_steps[i] = A_steps[i + 1] * A_broadcast_shape[i + 1];
            B_steps[i] = B_steps[i + 1] * B_broadcast_shape[i + 1];
        }
        size_t t, idx;
        for (size_t i = 0; i < batch; i++) {
            C_offsets[i] = i * C_step;

            size_t A_offset = 0, B_offset = 0;
            t = i;
            for (int j = batch_ndims - 1; j >= 0; j--) {
                idx = t / C_shape[j];
                int idx_offset = (int)(t - idx * C_shape[j]);
                A_offset += A_broadcast_shape[j] == 1 ? 0 : idx_offset * A_steps[j];
                B_offset += B_broadcast_shape[j] == 1 ? 0 : idx_offset * B_steps[j];
                t = idx;
            }
            A_offsets[i] = A_offset;
            B_offsets[i] = B_offset;
        }
    }
};

void fastGemmPackB(const Mat &m, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt);

void fastGemm(bool trans_a, int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *packed_B, float beta,
              float *C, int ldc, FastGemmOpt &opt);
void fastGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
              float alpha, const float *A, int lda0, int lda1, const float *B, int ldb0, int ldb1,
              float beta, float *C, int ldc, FastGemmOpt &opt);
void fastGemm(bool trans_a, bool trans_b,
              float alpha, const Mat &A, const Mat &B,
              float beta, Mat &C, FastGemmOpt &opt);

void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                   int M, int N, int K, float alpha, const float *A, int lda0, int lda1, const float *B,
                   int ldb0, int ldb1, float beta, float *C, int ldc, FastGemmOpt &opt);
void fastGemmBatch(bool trans_a, bool trans_b, float alpha, const Mat &A,
                   const Mat &B, float beta, Mat &C, FastGemmOpt &opt);

}} // cv::dnn

#endif // OPENCV_DNN_FAST_GEMM_HPP
