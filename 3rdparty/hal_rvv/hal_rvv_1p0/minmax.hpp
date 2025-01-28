// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MINMAXIDX_HPP_INCLUDED
#define OPENCV_HAL_RVV_MINMAXIDX_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_minMaxIdx
#define cv_hal_minMaxIdx cv::cv_hal_rvv::minMaxIdx
#undef cv_hal_minMaxIdxMaskStep
#define cv_hal_minMaxIdxMaskStep cv::cv_hal_rvv::minMaxIdx

#define HAL_RVV_MINMAXIDX_READTWICE_GENERATOR(D_TYPE, V_TYPE, EEW, EMUL, IS_U, IS_F, F_OR_X, F_OR_S, M_EMUL) \
inline int minMaxIdx_##V_TYPE(const uchar* src_data, size_t src_step, int width, int height, double* minVal, double* maxVal, \
                              int* minIdx, int* maxIdx, uchar* mask, size_t mask_step) \
{ \
    int vlmax = __riscv_vsetvlmax_##EEW##EMUL(); \
    auto vec_min = __riscv_v##IS_F##mv_v_##F_OR_X##_##V_TYPE##EMUL(std::numeric_limits<D_TYPE>::max(), vlmax); \
    auto vec_max = __riscv_v##IS_F##mv_v_##F_OR_X##_##V_TYPE##EMUL(std::numeric_limits<D_TYPE>::lowest(), vlmax); \
    D_TYPE val_min, val_max; \
 \
    if (mask) \
    { \
        for (int i = 0; i < height; i++) \
        { \
            const D_TYPE* src_row = reinterpret_cast<const D_TYPE*>(src_data + i * src_step); \
            const uchar* mask_row = mask + i * mask_step; \
            int vl; \
            for (int j = 0; j < width; j += vl) \
            { \
                vl = __riscv_vsetvl_##EEW##EMUL(width - j); \
                auto vec_src = __riscv_vl##EEW##_v_##V_TYPE##EMUL(src_row + j, vl); \
                auto vec_mask = __riscv_vle8_v_u8##M_EMUL(mask_row + j, vl); \
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl); \
                vec_min = __riscv_v##IS_F##min##IS_U##_tumu(bool_mask, vec_min, vec_min, vec_src, vl); \
                vec_max = __riscv_v##IS_F##max##IS_U##_tumu(bool_mask, vec_max, vec_max, vec_src, vl); \
            } \
        } \
 \
        auto sc_minval = __riscv_v##IS_F##mv_s_##F_OR_X##_##V_TYPE##m1(std::numeric_limits<D_TYPE>::max(), vlmax); \
        auto sc_maxval = __riscv_v##IS_F##mv_s_##F_OR_X##_##V_TYPE##m1(std::numeric_limits<D_TYPE>::lowest(), vlmax); \
        sc_minval = __riscv_v##IS_F##redmin##IS_U(vec_min, sc_minval, vlmax); \
        sc_maxval = __riscv_v##IS_F##redmax##IS_U(vec_max, sc_maxval, vlmax); \
        val_min = __riscv_v##IS_F##mv_##F_OR_X(sc_minval); \
        val_max = __riscv_v##IS_F##mv_##F_OR_X(sc_maxval); \
 \
        bool found_min = !minIdx, found_max = !maxIdx; \
        for (int i = 0; i < height && (!found_min || !found_max); i++) \
        { \
            const D_TYPE* src_row = reinterpret_cast<const D_TYPE*>(src_data + i * src_step); \
            const uchar* mask_row = mask + i * mask_step; \
            int vl; \
            for (int j = 0; j < width && (!found_min || !found_max); j += vl) \
            { \
                vl = __riscv_vsetvl_##EEW##EMUL(width - j); \
                auto vec_src = __riscv_vl##EEW##_v_##V_TYPE##EMUL(src_row + j, vl); \
                auto vec_mask = __riscv_vle8_v_u8##M_EMUL(mask_row + j, vl); \
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl); \
                auto bool_zero = __riscv_vmxor(bool_mask, bool_mask, vl); \
                if (!found_min) \
                { \
                    auto bool_minpos = __riscv_vm##F_OR_S##eq_mu(bool_mask, bool_zero, vec_src, val_min, vl); \
                    int index = __riscv_vfirst(bool_minpos, vl); \
                    if (index != -1) \
                    { \
                        found_min = true; \
                        minIdx[0] = i; \
                        minIdx[1] = j + index; \
                    } \
                } \
                if (!found_max) \
                { \
                    auto bool_maxpos = __riscv_vm##F_OR_S##eq_mu(bool_mask, bool_zero, vec_src, val_max, vl); \
                    int index = __riscv_vfirst(bool_maxpos, vl); \
                    if (index != -1) \
                    { \
                        found_max = true; \
                        maxIdx[0] = i; \
                        maxIdx[1] = j + index; \
                    } \
                } \
            } \
        } \
    } \
    else \
    { \
        for (int i = 0; i < height; i++) \
        { \
            const D_TYPE* src_row = reinterpret_cast<const D_TYPE*>(src_data + i * src_step); \
            int vl; \
            for (int j = 0; j < width; j += vl) \
            { \
                vl = __riscv_vsetvl_##EEW##EMUL(width - j); \
                auto vec_src = __riscv_vl##EEW##_v_##V_TYPE##EMUL(src_row + j, vl); \
                vec_min = __riscv_v##IS_F##min##IS_U##_tu(vec_min, vec_min, vec_src, vl); \
                vec_max = __riscv_v##IS_F##max##IS_U##_tu(vec_max, vec_max, vec_src, vl); \
            } \
        } \
 \
        auto sc_minval = __riscv_v##IS_F##mv_s_##F_OR_X##_##V_TYPE##m1(std::numeric_limits<D_TYPE>::max(), vlmax); \
        auto sc_maxval = __riscv_v##IS_F##mv_s_##F_OR_X##_##V_TYPE##m1(std::numeric_limits<D_TYPE>::lowest(), vlmax); \
        sc_minval = __riscv_v##IS_F##redmin##IS_U(vec_min, sc_minval, vlmax); \
        sc_maxval = __riscv_v##IS_F##redmax##IS_U(vec_max, sc_maxval, vlmax); \
        val_min = __riscv_v##IS_F##mv_##F_OR_X(sc_minval); \
        val_max = __riscv_v##IS_F##mv_##F_OR_X(sc_maxval); \
 \
        bool found_min = !minIdx, found_max = !maxIdx; \
        for (int i = 0; i < height && (!found_min || !found_max); i++) \
        { \
            const D_TYPE* src_row = reinterpret_cast<const D_TYPE*>(src_data + i * src_step); \
            int vl; \
            for (int j = 0; j < width && (!found_min || !found_max); j += vl) \
            { \
                vl = __riscv_vsetvl_##EEW##EMUL(width - j); \
                auto vec_src = __riscv_vl##EEW##_v_##V_TYPE##EMUL(src_row + j, vl); \
                if (!found_min) \
                { \
                    auto bool_minpos = __riscv_vm##F_OR_S##eq(vec_src, val_min, vl); \
                    int index = __riscv_vfirst(bool_minpos, vl); \
                    if (index != -1) \
                    { \
                        found_min = true; \
                        minIdx[0] = i; \
                        minIdx[1] = j + index; \
                    } \
                } \
                if (!found_max) \
                { \
                    auto bool_maxpos = __riscv_vm##F_OR_S##eq(vec_src, val_max, vl); \
                    int index = __riscv_vfirst(bool_maxpos, vl); \
                    if (index != -1) \
                    { \
                        found_max = true; \
                        maxIdx[0] = i; \
                        maxIdx[1] = j + index; \
                    } \
                } \
            } \
        } \
    } \
    if (minVal) \
    { \
        *minVal = val_min; \
    } \
    if (maxVal) \
    { \
        *maxVal = val_max; \
    } \
 \
    return CV_HAL_ERROR_OK; \
}

#define HAL_RVV_MINMAXIDX_READONCE_GENERATOR(D_TYPE, V_TYPE, EEW, EMUL, IS_U, IS_F, F_OR_X, F_OR_S, M_EMUL, P_EMUL) \
inline int minMaxIdx_##V_TYPE(const uchar* src_data, size_t src_step, int width, int height, double* minVal, double* maxVal, \
                              int* minIdx, int* maxIdx, uchar* mask, size_t mask_step) \
{ \
    int vlmax = __riscv_vsetvlmax_##EEW##EMUL(); \
    auto vec_min = __riscv_v##IS_F##mv_v_##F_OR_X##_##V_TYPE##EMUL(std::numeric_limits<D_TYPE>::max(), vlmax); \
    auto vec_max = __riscv_v##IS_F##mv_v_##F_OR_X##_##V_TYPE##EMUL(std::numeric_limits<D_TYPE>::lowest(), vlmax); \
    auto vec_pos = __riscv_vid_v_u32##P_EMUL(vlmax); \
    auto vec_minpos = __riscv_vundefined_u32##P_EMUL(), vec_maxpos = __riscv_vundefined_u32##P_EMUL(); \
    D_TYPE val_min, val_max; \
 \
    if (mask) \
    { \
        for (int i = 0; i < height; i++) \
        { \
            const D_TYPE* src_row = reinterpret_cast<const D_TYPE*>(src_data + i * src_step); \
            const uchar* mask_row = mask + i * mask_step; \
            int vl; \
            for (int j = 0; j < width; j += vl) \
            { \
                vl = __riscv_vsetvl_##EEW##EMUL(width - j); \
                auto vec_src = __riscv_vl##EEW##_v_##V_TYPE##EMUL(src_row + j, vl); \
                auto vec_mask = __riscv_vle8_v_u8##M_EMUL(mask_row + j, vl); \
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl); \
                auto bool_zero = __riscv_vmxor(bool_mask, bool_mask, vl); \
 \
                auto bool_minpos = __riscv_vm##F_OR_S##lt##IS_U##_mu(bool_mask, bool_zero, vec_src, vec_min, vl); \
                auto bool_maxpos = __riscv_vm##F_OR_S##gt##IS_U##_mu(bool_mask, bool_zero, vec_src, vec_max, vl); \
                vec_minpos = __riscv_vmerge_tu(vec_minpos, vec_minpos, vec_pos, bool_minpos, vl); \
                vec_maxpos = __riscv_vmerge_tu(vec_maxpos, vec_maxpos, vec_pos, bool_maxpos, vl); \
 \
                vec_min = __riscv_vmerge_tu(vec_min, vec_min, vec_src, bool_minpos, vl); \
                vec_max = __riscv_vmerge_tu(vec_max, vec_max, vec_src, bool_maxpos, vl); \
                vec_pos = __riscv_vadd(vec_pos, vl, vlmax); \
            } \
        } \
    } \
    else \
    { \
        for (int i = 0; i < height; i++) \
        { \
            const D_TYPE* src_row = reinterpret_cast<const D_TYPE*>(src_data + i * src_step); \
            int vl; \
            for (int j = 0; j < width; j += vl) \
            { \
                vl = __riscv_vsetvl_##EEW##EMUL(width - j); \
                auto vec_src = __riscv_vl##EEW##_v_##V_TYPE##EMUL(src_row + j, vl); \
 \
                auto bool_minpos = __riscv_vm##F_OR_S##lt##IS_U(vec_src, vec_min, vl); \
                auto bool_maxpos = __riscv_vm##F_OR_S##gt##IS_U(vec_src, vec_max, vl); \
                vec_minpos = __riscv_vmerge_tu(vec_minpos, vec_minpos, vec_pos, bool_minpos, vl); \
                vec_maxpos = __riscv_vmerge_tu(vec_maxpos, vec_maxpos, vec_pos, bool_maxpos, vl); \
 \
                vec_min = __riscv_vmerge_tu(vec_min, vec_min, vec_src, bool_minpos, vl); \
                vec_max = __riscv_vmerge_tu(vec_max, vec_max, vec_src, bool_maxpos, vl); \
                vec_pos = __riscv_vadd(vec_pos, vl, vlmax); \
            } \
        } \
    } \
 \
    val_min = std::numeric_limits<D_TYPE>::max(); \
    val_max = std::numeric_limits<D_TYPE>::lowest(); \
    for (int i = 0; i < vlmax; i++) \
    { \
        if (val_min > __riscv_v##IS_F##mv_##F_OR_X(vec_min)) \
        { \
            val_min = __riscv_v##IS_F##mv_##F_OR_X(vec_min); \
            if (minIdx) \
            { \
                minIdx[0] = __riscv_vmv_x(vec_minpos) / width; \
                minIdx[1] = __riscv_vmv_x(vec_minpos) % width; \
            } \
        } \
        if (val_max < __riscv_v##IS_F##mv_##F_OR_X(vec_max)) \
        { \
            val_max = __riscv_v##IS_F##mv_##F_OR_X(vec_max); \
            if (maxIdx) \
            { \
                maxIdx[0] = __riscv_vmv_x(vec_maxpos) / width; \
                maxIdx[1] = __riscv_vmv_x(vec_maxpos) % width; \
            } \
        } \
        vec_min = __riscv_vslidedown(vec_min, 1, vlmax); \
        vec_max = __riscv_vslidedown(vec_max, 1, vlmax); \
        vec_minpos = __riscv_vslidedown(vec_minpos, 1, vlmax); \
        vec_maxpos = __riscv_vslidedown(vec_maxpos, 1, vlmax); \
    } \
    if (minVal) \
    { \
        *minVal = val_min; \
    } \
    if (maxVal) \
    { \
        *maxVal = val_max; \
    } \
 \
    return CV_HAL_ERROR_OK; \
}

HAL_RVV_MINMAXIDX_READTWICE_GENERATOR(uchar, u8, e8, m1, u, , x, s, m1)
HAL_RVV_MINMAXIDX_READTWICE_GENERATOR(schar, i8, e8, m1, , , x, s, m1)
HAL_RVV_MINMAXIDX_READTWICE_GENERATOR(ushort, u16, e16, m1, u, , x, s, mf2)
HAL_RVV_MINMAXIDX_READTWICE_GENERATOR(short, i16, e16, m1, , , x, s, mf2)
HAL_RVV_MINMAXIDX_READONCE_GENERATOR(int, i32, e32, m4, , , x, s, m1, m4)
HAL_RVV_MINMAXIDX_READONCE_GENERATOR(float, f32, e32, m4, , f, f, f, m1, m4)
HAL_RVV_MINMAXIDX_READONCE_GENERATOR(double, f64, e64, m4, , f, f, f, mf2, m2)
#undef HAL_RVV_MINMAXIDX_READTWICE_GENERATOR
#undef HAL_RVV_MINMAXIDX_READONCE_GENERATOR

inline int minMaxIdx(const uchar* src_data, size_t src_step, int width, int height, int depth, double* minVal, double* maxVal,
                     int* minIdx, int* maxIdx, uchar* mask, size_t mask_step = 0)
{
    if (!mask_step)
        mask_step = src_step;

    switch (depth)
    {
    case CV_8UC1:
        return minMaxIdx_u8(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_8SC1:
        return minMaxIdx_i8(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_16UC1:
        return minMaxIdx_u16(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_16SC1:
        return minMaxIdx_i16(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_32SC1:
        return minMaxIdx_i32(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_32FC1:
        return minMaxIdx_f32(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_64FC1:
        return minMaxIdx_f64(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}

#endif
