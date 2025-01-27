// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_CONVERT_SCALE_HPP_INCLUDED
#define OPENCV_HAL_RVV_CONVERT_SCALE_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_convertScale
#define cv_hal_convertScale cv::cv_hal_rvv::convertScale

inline int convertScale_8U8U(const uchar* src, size_t src_step, uchar* dst, size_t dst_step, int width, int height, double alpha, double beta)
{
    int vlmax = __riscv_vsetvlmax_e32m8();
    auto vec_b = __riscv_vfmv_v_f_f32m8(beta, vlmax);
    float a = alpha;

    for (int i = 0; i < height; i++)
    {
        const uchar* src_row = src + i * src_step;
        uchar* dst_row = dst + i * dst_step;
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
            auto vec_src_u16 = __riscv_vzext_vf2(vec_src, vl);
            auto vec_src_f32 = __riscv_vfwcvt_f(vec_src_u16, vl);
            auto vec_fma = __riscv_vfmadd(vec_src_f32, a, vec_b, vl);
            auto vec_dst_u16 = __riscv_vfncvt_xu(vec_fma, vl);
            auto vec_dst = __riscv_vnclipu(vec_dst_u16, 0, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_u8m2(dst_row + j, vec_dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int convertScale_8U32F(const uchar* src, size_t src_step, uchar* dst, size_t dst_step, int width, int height, double alpha, double beta)
{
    int vlmax = __riscv_vsetvlmax_e32m8();
    auto vec_b = __riscv_vfmv_v_f_f32m8(beta, vlmax);
    float a = alpha;

    for (int i = 0; i < height; i++)
    {
        const uchar* src_row = src + i * src_step;
        float* dst_row = reinterpret_cast<float*>(dst + i * dst_step);
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
            auto vec_src_u16 = __riscv_vzext_vf2(vec_src, vl);
            auto vec_src_f32 = __riscv_vfwcvt_f(vec_src_u16, vl);
            auto vec_fma = __riscv_vfmadd(vec_src_f32, a, vec_b, vl);
            __riscv_vse32_v_f32m8(dst_row + j, vec_fma, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int convertScale_32F32F(const uchar* src, size_t src_step, uchar* dst, size_t dst_step, int width, int height, double alpha, double beta)
{
    int vlmax = __riscv_vsetvlmax_e32m8();
    auto vec_b = __riscv_vfmv_v_f_f32m8(beta, vlmax);
    float a = alpha;

    for (int i = 0; i < height; i++)
    {
        const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
        float* dst_row = reinterpret_cast<float*>(dst + i * dst_step);
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m8(width - j);
            auto vec_src = __riscv_vle32_v_f32m8(src_row + j, vl);
            auto vec_fma = __riscv_vfmadd(vec_src, a, vec_b, vl);
            __riscv_vse32_v_f32m8(dst_row + j, vec_fma, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int convertScale(const uchar* src, size_t src_step, uchar* dst, size_t dst_step, int width, int height,
                        int sdepth, int ddepth, double alpha, double beta)
{
    if (!dst)
        return CV_HAL_ERROR_OK;

    switch (sdepth)
    {
    case CV_8U:
        switch (ddepth)
        {
        case CV_8U:
            return convertScale_8U8U(src, src_step, dst, dst_step, width, height, alpha, beta);
        case CV_32F:
            return convertScale_8U32F(src, src_step, dst, dst_step, width, height, alpha, beta);
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    case CV_32F:
        switch (ddepth)
        {
        case CV_32F:
            return convertScale_32F32F(src, src_step, dst, dst_step, width, height, alpha, beta);
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}

#endif
