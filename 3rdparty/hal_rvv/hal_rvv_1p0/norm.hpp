// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_NORM_HPP_INCLUDED
#define OPENCV_HAL_RVV_NORM_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_norm
#define cv_hal_norm cv::cv_hal_rvv::norm

inline int normInf_8UC1(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e8m8();
    auto vec_max = __riscv_vmv_v_x_u8m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m8(width - j);
                auto vec_src = __riscv_vle8_v_u8m8(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m8(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                vec_max = __riscv_vmaxu_tumu(bool_mask, vec_max, vec_max, vec_src, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m8(width - j);
                auto vec_src = __riscv_vle8_v_u8m8(src_row + j, vl);
                vec_max = __riscv_vmaxu_tu(vec_max, vec_max, vec_src, vl);
            }
        }
    }
    auto sc_max = __riscv_vmv_s_x_u8m1(0, vlmax);
    sc_max = __riscv_vredmaxu(vec_max, sc_max, vlmax);
    *result = __riscv_vmv_x(sc_max);

    return CV_HAL_ERROR_OK;
}

inline int normL1_8UC1(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e8m2();
    auto vec_sum = __riscv_vmv_v_x_u32m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m2(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto vec_zext = __riscv_vzext_vf4_u32m8_m(bool_mask, vec_src, vl);
                vec_sum = __riscv_vadd_tumu(bool_mask, vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_zext = __riscv_vzext_vf4(vec_src, vl);
                vec_sum = __riscv_vadd_tu(vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    auto sc_sum = __riscv_vmv_s_x_u32m1(0, vlmax);
    sc_sum = __riscv_vredsum(vec_sum, sc_sum, vlmax);
    *result = __riscv_vmv_x(sc_sum);

    return CV_HAL_ERROR_OK;
}

inline int normL2Sqr_8UC1(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e8m2();
    auto vec_sum = __riscv_vmv_v_x_u32m8(0, vlmax);
    int cnt = 0;
    auto reduce = [&](int vl) {
        if ((cnt += vl) < (1 << 16))
            return;
        cnt = vl;
        for (int i = 0; i < vlmax; i++)
        {
            *result += __riscv_vmv_x(vec_sum);
            vec_sum = __riscv_vslidedown(vec_sum, 1, vlmax);
        }
        vec_sum = __riscv_vmv_v_x_u32m8(0, vlmax);
    };

    *result = 0;
    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                reduce(vl);

                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m2(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto vec_mul = __riscv_vwmulu_vv_u16m4_m(bool_mask, vec_src, vec_src, vl);
                auto vec_zext = __riscv_vzext_vf2_u32m8_m(bool_mask, vec_mul, vl);
                vec_sum = __riscv_vadd_tumu(bool_mask, vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                reduce(vl);

                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_mul = __riscv_vwmulu(vec_src, vec_src, vl);
                auto vec_zext = __riscv_vzext_vf2(vec_mul, vl);
                vec_sum = __riscv_vadd_tu(vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    reduce(1 << 16);

    return CV_HAL_ERROR_OK;
}

inline int normInf_8UC4(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e8m8();
    auto vec_max = __riscv_vmv_v_x_u8m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int vl, vlm;
            for (int j = 0, jm = 0; j < width * 4; j += vl, jm += vlm)
            {
                vl = __riscv_vsetvl_e8m8(width * 4 - j);
                vlm = __riscv_vsetvl_e8m2(width - jm);
                auto vec_src = __riscv_vle8_v_u8m8(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m2(mask_row + jm, vlm);
                auto vec_mask_ext = __riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(vec_mask, 1, vlm), vlm), 0x01010101, vlm);
                auto bool_mask_ext = __riscv_vmsne(__riscv_vreinterpret_u8m8(vec_mask_ext), 0, vl);
                vec_max = __riscv_vmaxu_tumu(bool_mask_ext, vec_max, vec_max, vec_src, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            int vl;
            for (int j = 0; j < width * 4; j += vl)
            {
                vl = __riscv_vsetvl_e8m8(width * 4 - j);
                auto vec_src = __riscv_vle8_v_u8m8(src_row + j, vl);
                vec_max = __riscv_vmaxu_tu(vec_max, vec_max, vec_src, vl);
            }
        }
    }
    auto sc_max = __riscv_vmv_s_x_u8m1(0, vlmax);
    sc_max = __riscv_vredmaxu(vec_max, sc_max, vlmax);
    *result = __riscv_vmv_x(sc_max);

    return CV_HAL_ERROR_OK;
}

inline int normL1_8UC4(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e8m2();
    auto vec_sum = __riscv_vmv_v_x_u32m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int vl, vlm;
            for (int j = 0, jm = 0; j < width * 4; j += vl, jm += vlm)
            {
                vl = __riscv_vsetvl_e8m2(width * 4 - j);
                vlm = __riscv_vsetvl_e8mf2(width - jm);
                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8mf2(mask_row + jm, vlm);
                auto vec_mask_ext = __riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(vec_mask, 1, vlm), vlm), 0x01010101, vlm);
                auto bool_mask_ext = __riscv_vmsne(__riscv_vreinterpret_u8m2(vec_mask_ext), 0, vl);
                auto vec_zext = __riscv_vzext_vf4_u32m8_m(bool_mask_ext, vec_src, vl);
                vec_sum = __riscv_vadd_tumu(bool_mask_ext, vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            int vl;
            for (int j = 0; j < width * 4; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width * 4 - j);
                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_zext = __riscv_vzext_vf4(vec_src, vl);
                vec_sum = __riscv_vadd_tu(vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    auto sc_sum = __riscv_vmv_s_x_u32m1(0, vlmax);
    sc_sum = __riscv_vredsum(vec_sum, sc_sum, vlmax);
    *result = __riscv_vmv_x(sc_sum);

    return CV_HAL_ERROR_OK;
}

inline int normL2Sqr_8UC4(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e8m2();
    auto vec_sum = __riscv_vmv_v_x_u32m8(0, vlmax);
    int cnt = 0;
    auto reduce = [&](int vl) {
        if ((cnt += vl) < (1 << 16))
            return;
        cnt = vl;
        for (int i = 0; i < vlmax; i++)
        {
            *result += __riscv_vmv_x(vec_sum);
            vec_sum = __riscv_vslidedown(vec_sum, 1, vlmax);
        }
        vec_sum = __riscv_vmv_v_x_u32m8(0, vlmax);
    };

    *result = 0;
    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int vl, vlm;
            for (int j = 0, jm = 0; j < width * 4; j += vl, jm += vlm)
            {
                vl = __riscv_vsetvl_e8m2(width * 4 - j);
                vlm = __riscv_vsetvl_e8mf2(width - jm);
                reduce(vl);

                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8mf2(mask_row + jm, vlm);
                auto vec_mask_ext = __riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(vec_mask, 1, vlm), vlm), 0x01010101, vlm);
                auto bool_mask_ext = __riscv_vmsne(__riscv_vreinterpret_u8m2(vec_mask_ext), 0, vl);
                auto vec_mul = __riscv_vwmulu_vv_u16m4_m(bool_mask_ext, vec_src, vec_src, vl);
                auto vec_zext = __riscv_vzext_vf2_u32m8_m(bool_mask_ext, vec_mul, vl);
                vec_sum = __riscv_vadd_tumu(bool_mask_ext, vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const uchar* src_row = src + i * src_step;
            int vl;
            for (int j = 0; j < width * 4; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width * 4 - j);
                reduce(vl);

                auto vec_src = __riscv_vle8_v_u8m2(src_row + j, vl);
                auto vec_mul = __riscv_vwmulu(vec_src, vec_src, vl);
                auto vec_zext = __riscv_vzext_vf2(vec_mul, vl);
                vec_sum = __riscv_vadd_tu(vec_sum, vec_sum, vec_zext, vl);
            }
        }
    }
    reduce(1 << 16);

    return CV_HAL_ERROR_OK;
}

inline int normInf_32FC1(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e32m8();
    auto vec_max = __riscv_vfmv_v_f_f32m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m8(width - j);
                auto vec_src = __riscv_vle32_v_f32m8(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m2(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto vec_abs = __riscv_vfabs_v_f32m8_m(bool_mask, vec_src, vl);
                vec_max = __riscv_vfmax_tumu(bool_mask, vec_max, vec_max, vec_abs, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m8(width - j);
                auto vec_src = __riscv_vle32_v_f32m8(src_row + j, vl);
                auto vec_abs = __riscv_vfabs(vec_src, vl);
                vec_max = __riscv_vfmax_tu(vec_max, vec_max, vec_abs, vl);
            }
        }
    }
    auto sc_max = __riscv_vfmv_s_f_f32m1(0, vlmax);
    sc_max = __riscv_vfredmax(vec_max, sc_max, vlmax);
    *result = __riscv_vfmv_f(sc_max);

    return CV_HAL_ERROR_OK;
}

inline int normL1_32FC1(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e32m4();
    auto vec_sum = __riscv_vfmv_v_f_f64m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(width - j);
                auto vec_src = __riscv_vle32_v_f32m4(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m1(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto vec_abs = __riscv_vfabs_v_f32m4_m(bool_mask, vec_src, vl);
                auto vec_fext = __riscv_vfwcvt_f_f_v_f64m8_m(bool_mask, vec_abs, vl);
                vec_sum = __riscv_vfadd_tumu(bool_mask, vec_sum, vec_sum, vec_fext, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(width - j);
                auto vec_src = __riscv_vle32_v_f32m4(src_row + j, vl);
                auto vec_abs = __riscv_vfabs(vec_src, vl);
                auto vec_fext = __riscv_vfwcvt_f_f_v_f64m8(vec_abs, vl);
                vec_sum = __riscv_vfadd_tu(vec_sum, vec_sum, vec_fext, vl);
            }
        }
    }
    auto sc_sum = __riscv_vfmv_s_f_f64m1(0, vlmax);
    sc_sum = __riscv_vfredosum(vec_sum, sc_sum, vlmax);
    *result = __riscv_vfmv_f(sc_sum);

    return CV_HAL_ERROR_OK;
}

inline int normL2Sqr_32FC1(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width, int height, double* result)
{
    int vlmax = __riscv_vsetvlmax_e32m4();
    auto vec_sum = __riscv_vfmv_v_f_f64m8(0, vlmax);

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(width - j);
                auto vec_src = __riscv_vle32_v_f32m4(src_row + j, vl);
                auto vec_mask = __riscv_vle8_v_u8m1(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto vec_mul = __riscv_vfwmul_vv_f64m8_m(bool_mask, vec_src, vec_src, vl);
                vec_sum = __riscv_vfadd_tumu(bool_mask, vec_sum, vec_sum, vec_mul, vl);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const float* src_row = reinterpret_cast<const float*>(src + i * src_step);
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(width - j);
                auto vec_src = __riscv_vle32_v_f32m4(src_row + j, vl);
                auto vec_mul = __riscv_vfwmul(vec_src, vec_src, vl);
                vec_sum = __riscv_vfadd_tu(vec_sum, vec_sum, vec_mul, vl);
            }
        }
    }
    auto sc_sum = __riscv_vfmv_s_f_f64m1(0, vlmax);
    sc_sum = __riscv_vfredosum(vec_sum, sc_sum, vlmax);
    *result = __riscv_vfmv_f(sc_sum);

    return CV_HAL_ERROR_OK;
}

inline int norm(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width,
                int height, int type, int norm_type, double* result)
{
    if (!result)
        return CV_HAL_ERROR_OK;

    switch (type)
    {
    case CV_8UC1:
        switch (norm_type)
        {
        case NORM_INF:
            return normInf_8UC1(src, src_step, mask, mask_step, width, height, result);
        case NORM_L1:
            return normL1_8UC1(src, src_step, mask, mask_step, width, height, result);
        case NORM_L2SQR:
            return normL2Sqr_8UC1(src, src_step, mask, mask_step, width, height, result);
        case NORM_L2:
            int ret = normL2Sqr_8UC1(src, src_step, mask, mask_step, width, height, result);
            *result = std::sqrt(*result);
            return ret;
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    case CV_8UC4:
        switch (norm_type)
        {
        case NORM_INF:
            return normInf_8UC4(src, src_step, mask, mask_step, width, height, result);
        case NORM_L1:
            return normL1_8UC4(src, src_step, mask, mask_step, width, height, result);
        case NORM_L2SQR:
            return normL2Sqr_8UC4(src, src_step, mask, mask_step, width, height, result);
        case NORM_L2:
            int ret = normL2Sqr_8UC4(src, src_step, mask, mask_step, width, height, result);
            *result = std::sqrt(*result);
            return ret;
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    case CV_32FC1:
        switch (norm_type)
        {
        case NORM_INF:
            return normInf_32FC1(src, src_step, mask, mask_step, width, height, result);
        case NORM_L1:
            return normL1_32FC1(src, src_step, mask, mask_step, width, height, result);
        case NORM_L2SQR:
            return normL2Sqr_32FC1(src, src_step, mask, mask_step, width, height, result);
        case NORM_L2:
            int ret = normL2Sqr_32FC1(src, src_step, mask, mask_step, width, height, result);
            *result = std::sqrt(*result);
            return ret;
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}

#endif
