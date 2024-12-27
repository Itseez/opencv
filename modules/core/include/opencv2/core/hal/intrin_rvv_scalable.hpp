// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// The original implementation is contributed by HAN Liutong.
// Copyright (C) 2022, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP
#define OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP

#include <opencv2/core/base.hpp>

#if defined(__GNUC__) && !defined(__clang__)
// FIXIT: eliminate massive warnigs from templates
// GCC from 'rvv-next': riscv64-unknown-linux-gnu-g++ (g42df3464463) 12.0.1 20220505 (prerelease)
// doesn't work: #pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#ifndef CV_RVV_MAX_VLEN
#define CV_RVV_MAX_VLEN 1024
#endif

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD_SCALABLE 1
#define CV_SIMD_SCALABLE_64F 1
#if defined(__riscv_zvfh) && __riscv_zvfh
    #define CV_SIMD_SCALABLE_FP16 1
#else
    #define CV_SIMD_SCALABLE_FP16 0
#endif


using v_uint8 = vuint8m2_t;
using v_int8 = vint8m2_t;
using v_uint16 = vuint16m2_t;
using v_int16 = vint16m2_t;
using v_uint32 = vuint32m2_t;
using v_int32 = vint32m2_t;
using v_uint64 = vuint64m2_t;
using v_int64 = vint64m2_t;

#if CV_SIMD_SCALABLE_FP16
using v_float16 = vfloat16m2_t;
#endif
using v_float32 = vfloat32m2_t;
#if CV_SIMD_SCALABLE_64F
using v_float64 = vfloat64m2_t;
#endif

using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using uint64 = unsigned long int;
using int64 = long int;


template <class T>
struct VTraits;

#define OPENCV_HAL_IMPL_RVV_TRAITS(REG, TYP, SUF, SZ) \
template <> \
struct VTraits<REG> \
{ \
    static inline int vlanes() { return __riscv_vsetvlmax_##SUF(); } \
    using lane_type = TYP; \
    static const int max_nlanes = CV_RVV_MAX_VLEN/SZ; \
};

OPENCV_HAL_IMPL_RVV_TRAITS(vint8m1_t, int8_t, e8m1, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vint8m2_t, int8_t, e8m2, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vint8m4_t, int8_t, e8m4, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vint8m8_t, int8_t, e8m8, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint8m1_t, uint8_t, e8m1, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint8m2_t, uint8_t, e8m2, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint8m4_t, uint8_t, e8m4, 8)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint8m8_t, uint8_t, e8m8, 8)

OPENCV_HAL_IMPL_RVV_TRAITS(vint16m1_t, int16_t, e16m1, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vint16m2_t, int16_t, e16m2, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vint16m4_t, int16_t, e16m4, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vint16m8_t, int16_t, e16m8, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint16m1_t, uint16_t, e16m1, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint16m2_t, uint16_t, e16m2, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint16m4_t, uint16_t, e16m4, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint16m8_t, uint16_t, e16m8, 16)

OPENCV_HAL_IMPL_RVV_TRAITS(vint32m1_t, int32_t, e32m1, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vint32m2_t, int32_t, e32m2, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vint32m4_t, int32_t, e32m4, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vint32m8_t, int32_t, e32m8, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint32m1_t, uint32_t, e32m1, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint32m2_t, uint32_t, e32m2, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint32m4_t, uint32_t, e32m4, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint32m8_t, uint32_t, e32m8, 32)

OPENCV_HAL_IMPL_RVV_TRAITS(vint64m1_t, int64_t, e64m1, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vint64m2_t, int64_t, e64m2, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vint64m4_t, int64_t, e64m4, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vint64m8_t, int64_t, e64m8, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint64m1_t, uint64_t, e64m1, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint64m2_t, uint64_t, e64m2, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint64m4_t, uint64_t, e64m4, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vuint64m8_t, uint64_t, e64m8, 64)

#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat16m1_t, hfloat, e16m1, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat16m2_t, hfloat, e16m2, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat16m4_t, hfloat, e16m4, 16)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat16m8_t, hfloat, e16m8, 16)
#endif

OPENCV_HAL_IMPL_RVV_TRAITS(vfloat32m1_t, float, e32m1, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat32m2_t, float, e32m2, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat32m4_t, float, e32m4, 32)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat32m8_t, float, e32m8, 32)

#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat64m1_t, double, e64m1, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat64m2_t, double, e64m2, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat64m4_t, double, e64m4, 64)
OPENCV_HAL_IMPL_RVV_TRAITS(vfloat64m8_t, double, e64m8, 64)
#endif


// LLVM/Clang defines "overloaded intrinsics" e.g. 'vand(op1, op2)'
// GCC does not have these functions, so we need to implement them manually
// We implement only selected subset required to build current state of the code
// Included inside namespace cv::
// #ifndef __riscv_v_intrinsic_overloading
// #include "intrin_rvv_compat_overloaded.hpp"
// #endif // __riscv_v_intrinsic_overloading


//////////// get0 ////////////
#define OPENCV_HAL_IMPL_RVV_GRT0_INT(_Tpvec, _Tp) \
inline _Tp v_get0(const v_##_Tpvec& v) \
{ \
    return __riscv_vmv_x(v); \
}

OPENCV_HAL_IMPL_RVV_GRT0_INT(uint8, uchar)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int8, schar)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint16, ushort)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int16, short)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint32, unsigned)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int32, int)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint64, uint64)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int64, int64)

#if CV_SIMD_SCALABLE_FP16
inline hfloat v_get0(const v_float16& v) \
{ \
    return (hfloat)__riscv_vfmv_f(v); \
}
#endif
inline float v_get0(const v_float32& v) \
{ \
    return __riscv_vfmv_f(v); \
}
#if CV_SIMD_SCALABLE_64F
inline double v_get0(const v_float64& v) \
{ \
    return __riscv_vfmv_f(v); \
}
#endif

//////////// Initial ////////////

#define OPENCV_HAL_IMPL_RVV_INIT_INTEGER(_Tpvec, _Tp, suffix1, suffix2, vl) \
inline v_##_Tpvec v_setzero_##suffix1() \
{ \
    return __riscv_vmv_v_x_##suffix2##m2(0, vl); \
} \
inline v_##_Tpvec v_setall_##suffix1(_Tp v) \
{ \
    return __riscv_vmv_v_x_##suffix2##m2(v, vl); \
} \
template <> inline v_##_Tpvec v_setzero_() \
{ \
    return v_setzero_##suffix1(); \
} \
template <> inline v_##_Tpvec v_setall_(_Tp v) \
{ \
    return v_setall_##suffix1(v); \
}

OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint8, uchar, u8, u8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int8, schar, s8, i8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint16, ushort, u16, u16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int16, short, s16, i16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint32, uint, u32, u32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int32, int, s32, i32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint64, uint64, u64, u64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int64, int64, s64, i64, VTraits<v_int64>::vlanes())

#define OPENCV_HAL_IMPL_RVV_INIT_FP(_Tpv, _Tp, suffix, vl) \
inline v_##_Tpv v_setzero_##suffix() \
{ \
    return __riscv_vfmv_v_f_##suffix##m2(0, vl); \
} \
inline v_##_Tpv v_setall_##suffix(_Tp v) \
{ \
    return __riscv_vfmv_v_f_##suffix##m2(v, vl); \
} \
template <> inline v_##_Tpv v_setzero_() \
{ \
    return v_setzero_##suffix(); \
} \
template <> inline v_##_Tpv v_setall_(_Tp v) \
{ \
    return v_setall_##suffix(v); \
}

#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_setzero_f16()
{
    return __riscv_vfmv_v_f_f16m2(0, VTraits<v_float16>::vlanes());
}
inline v_float16 v_setall_f16(float v) // In some cases we may use v_setall_f16(1.0f)
{
    return __riscv_vfmv_v_f_f16m2((_Float16)v, VTraits<v_float16>::vlanes());
}
inline v_float16 v_setall_f16(hfloat v)
{
    return __riscv_vfmv_v_f_f16m2((_Float16)v, VTraits<v_float16>::vlanes());
}
template <> inline v_float16 v_setzero_() \
{ \
    return v_setzero_f16(); \
} \
template <> inline v_float16 v_setall_(hfloat v) \
{ \
    return v_setall_f16(v); \
}
#endif
OPENCV_HAL_IMPL_RVV_INIT_FP(float32, float, f32, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_INIT_FP(float64, double, f64, VTraits<v_float64>::vlanes())
#endif

//////////// Reinterpret ////////////
#define OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(_Tpvec1, suffix1) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec1& v) \
{ \
    return v;\
}
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint8, u8)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint16, u16)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint32, u32)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint64, u64)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int8, s8)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int16, s16)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int32, s32)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int64, s64)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(float16, f16)
#endif
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(float32, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(float64, f64)
#endif
// TODO: can be simplified by using overloaded RV intrinsic
#define OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, nsuffix1, nsuffix2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) \
{ \
    return v_##_Tpvec1(__riscv_vreinterpret_v_##nsuffix2##m2_##nsuffix1##m2(v));\
} \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) \
{ \
    return v_##_Tpvec2(__riscv_vreinterpret_v_##nsuffix1##m2_##nsuffix2##m2(v));\
}

OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, int8, u8, s8, u8, i8)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, int16, u16, s16, u16, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, int32, u32, s32, u32, i32)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, float16, u16, f16, u16, f16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, float16, s16, f16, i16, f16)
#endif
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, float32, u32, f32, u32, f32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int32, float32, s32, f32, i32, f32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint64, int64, u64, s64, u64, i64)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint64, float64, u64, f64, u64, f64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int64, float64, s64, f64, i64, f64)
#endif
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint16, u8, u16, u8, u16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint32, u8, u32, u8, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint64, u8, u64, u8, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, uint32, u16, u32, u16, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, uint64, u16, u64, u16, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, uint64, u32, u64, u32, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int16, s8, s16, i8, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int32, s8, s32, i8, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int64, s8, s64, i8, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, int32, s16, s32, i16, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, int64, s16, s64, i16, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int32, int64, s32, s64, i32, i64)


#define OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, nsuffix1, nsuffix2, width1, width2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) \
{ \
    return __riscv_vreinterpret_v_##nsuffix1##width2##m2_##nsuffix1##width1##m2(__riscv_vreinterpret_v_##nsuffix2##width2##m2_##nsuffix1##width2##m2(v));\
} \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) \
{ \
    return __riscv_vreinterpret_v_##nsuffix1##width2##m2_##nsuffix2##width2##m2(__riscv_vreinterpret_v_##nsuffix1##width1##m2_##nsuffix1##width2##m2(v));\
}

OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int16, u8, s16, u, i, 8, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int32, u8, s32, u, i, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int64, u8, s64, u, i, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int8, u16, s8, u, i, 16, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int32, u16, s32, u, i, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int64, u16, s64, u, i, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int8, u32, s8, u, i, 32, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int16, u32, s16, u, i, 32, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int64, u32, s64, u, i, 32, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int8, u64, s8, u, i, 64, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int16, u64, s16, u, i, 64, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int32, u64, s32, u, i, 64, 32)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float16, u8, f16, u, f, 8, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, float16, u32, f16, u, f, 32, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, float16, u64, f16, u, f, 64, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float16, s8, f16, i, f, 8, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int32, float16, s32, f16, i, f, 32, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int64, float16, s64, f16, i, f, 64, 16)
#endif
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float32, u8, f32, u, f, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, float32, u16, f32, u, f, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, float32, u64, f32, u, f, 64, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float32, s8, f32, i, f, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16, float32, s16, f32, i, f, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int64, float32, s64, f32, i, f, 64, 32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float64, u8, f64, u, f, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, float64, u16, f64, u, f, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, float64, u32, f64, u, f, 32, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float64, s8, f64, i, f, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16, float64, s16, f64, i, f, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int32, float64, s32, f64, i, f, 32, 64)
// Three times reinterpret
#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_reinterpret_as_f16(const v_float64& v) \
{ \
    return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vreinterpret_v_u64m2_u16m2(__riscv_vreinterpret_v_f64m2_u64m2(v)));\
}

inline v_float64 v_reinterpret_as_f64(const v_float16& v) \
{ \
    return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vreinterpret_v_u16m2_u64m2(__riscv_vreinterpret_v_f16m2_u16m2(v)));\
}
#endif
inline v_float32 v_reinterpret_as_f32(const v_float64& v) \
{ \
    return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vreinterpret_v_u64m2_u32m2(__riscv_vreinterpret_v_f64m2_u64m2(v)));\
}

inline v_float64 v_reinterpret_as_f64(const v_float32& v) \
{ \
    return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vreinterpret_v_u32m2_u64m2(__riscv_vreinterpret_v_f32m2_u32m2(v)));\
}
#endif

//////////// Extract //////////////

#define OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(_Tpvec, _Tp, vl) \
template <int s = 0> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b, int i = s) \
{ \
    return __riscv_vslideup(__riscv_vslidedown(a, i, vl), b, VTraits<_Tpvec>::vlanes() - i, vl); \
} \
template<int s = 0> inline _Tp v_extract_n(_Tpvec v, int i = s) \
{ \
    return __riscv_vmv_x(__riscv_vslidedown(v, i, vl)); \
}

OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint8, uchar, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int8, schar, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint16, ushort, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int16, short, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint32, unsigned int, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int32, int, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint64, uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int64, int64, VTraits<v_int64>::vlanes())

#define OPENCV_HAL_IMPL_RVV_EXTRACT_FP(_Tpvec, _Tp, vl) \
template <int s = 0> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b, int i = s) \
{ \
    return __riscv_vslideup(__riscv_vslidedown(a, i, vl), b, VTraits<_Tpvec>::vlanes() - i, vl); \
} \
template<int s = 0> inline _Tp v_extract_n(_Tpvec v, int i = s) \
{ \
    return (_Tp)__riscv_vfmv_f(__riscv_vslidedown(v, i, vl)); \
}

#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float16, hfloat, VTraits<v_float16>::vlanes())
#endif
OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float32, float, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float64, double, VTraits<v_float64>::vlanes())
#endif

#define OPENCV_HAL_IMPL_RVV_EXTRACT(_Tpvec, _Tp, vl) \
inline _Tp v_extract_highest(_Tpvec v) \
{ \
    return (_Tp)v_extract_n(v, vl-1); \
}

OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint8, uchar, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int8, schar, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint16, ushort, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int16, short, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint32, unsigned int, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int32, int, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint64, uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int64, int64, VTraits<v_int64>::vlanes())
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float16, hfloat, VTraits<v_float16>::vlanes())
#endif
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float32, float, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float64, double, VTraits<v_float64>::vlanes())
#endif


////////////// Load/Store //////////////
#define OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(_Tpvec, _nTpvec, _Tp, hvl, vl, width, suffix) \
inline _Tpvec v_load(const _Tp* ptr) \
{ \
    return __riscv_vle##width##_v_##suffix##m2(ptr, vl); \
} \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ \
    return __riscv_vle##width##_v_##suffix##m2(ptr, vl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ \
    __riscv_vse##width##_v_##suffix##m2(ptr, a, vl); \
} \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    return __riscv_vle##width##_v_##suffix##m2(ptr, hvl); \
} \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return __riscv_vslideup(__riscv_vle##width##_v_##suffix##m2(ptr0, hvl), __riscv_vle##width##_v_##suffix##m2(ptr1, hvl), hvl, vl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width(ptr, a, vl); \
} \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width(ptr, a, vl); \
} \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width(ptr, a, vl); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width(ptr, a, hvl); \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width(ptr, __riscv_vslidedown_vx_##suffix##m2(a, hvl, vl), hvl); \
} \
template<typename... Targs> \
_Tpvec v_load_##suffix(Targs... nScalars) \
{ \
    return v_load({nScalars...}); \
}

#define OPENCV_HAL_IMPL_RVV_LOADSTORE_OP_FP16(_Tpvec, _nTpvec, _Tp, hvl, vl, width, suffix) \
inline _Tpvec v_load(const _Tp* ptr) \
{ \
    return __riscv_vle##width##_v_##suffix##m2((_Float16*)ptr, vl); \
} \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ \
    return __riscv_vle##width##_v_##suffix##m2((_Float16*)ptr, vl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ \
    __riscv_vse##width##_v_##suffix##m2((_Float16*)ptr, a, vl); \
} \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    return __riscv_vle##width##_v_##suffix##m2((_Float16*)ptr, hvl); \
} \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return __riscv_vslideup(__riscv_vle##width##_v_##suffix##m2((_Float16*)ptr0, hvl), __riscv_vle##width##_v_##suffix##m2((_Float16*)ptr1, hvl), hvl, vl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width((_Float16*)ptr, a, vl); \
} \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width((_Float16*)ptr, a, vl); \
} \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width((_Float16*)ptr, a, vl); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width((_Float16*)ptr, a, hvl); \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    __riscv_vse##width((_Float16*)ptr, __riscv_vslidedown_vx_##suffix##m2(a, hvl, vl), hvl); \
}

OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint8, vuint8m2_t, uchar, VTraits<v_uint8>::vlanes() / 2, VTraits<v_uint8>::vlanes(), 8, u8)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int8, vint8m2_t, schar, VTraits<v_int8>::vlanes() / 2, VTraits<v_int8>::vlanes(), 8, i8)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint16, vuint16m2_t, ushort, VTraits<v_uint16>::vlanes() / 2, VTraits<v_uint16>::vlanes(), 16, u16)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int16, vint16m2_t, short, VTraits<v_int16>::vlanes() / 2, VTraits<v_int16>::vlanes(), 16, i16)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint32, vuint32m2_t, unsigned int, VTraits<v_uint32>::vlanes() / 2, VTraits<v_uint32>::vlanes(), 32, u32)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int32, vint32m2_t, int, VTraits<v_int32>::vlanes() / 2, VTraits<v_int32>::vlanes(), 32, i32)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint64, vuint64m2_t, uint64, VTraits<v_uint64>::vlanes() / 2, VTraits<v_uint64>::vlanes(), 64, u64)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int64, vint64m2_t, int64, VTraits<v_int64>::vlanes() / 2, VTraits<v_int64>::vlanes(), 64, i64)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP_FP16(v_float16, vfloat16m2_t, hfloat, VTraits<v_float16>::vlanes() /2 , VTraits<v_float16>::vlanes(), 16, f16)
#endif
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float32, vfloat32m2_t, float, VTraits<v_float32>::vlanes() /2 , VTraits<v_float32>::vlanes(), 32, f32)

#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float64, vfloat64m2_t, double, VTraits<v_float64>::vlanes() / 2, VTraits<v_float64>::vlanes(), 64, f64)
#endif

template <int N = VTraits<v_uint16>::max_nlanes>
inline void v_store(ushort* ptr, const v_uint16& a)
{
    ushort buf[VTraits<v_uint16>::max_nlanes];
    v_store(buf, a);
    for (int i = 0; i < N; i++) {
        ptr[i] = buf[i];
    }
}
template <> inline void v_store<8>(ushort* ptr, const v_uint16& a)
{
    ushort buf[VTraits<v_uint16>::max_nlanes];
    v_store(buf, a);
    ptr[0] = buf[0]; ptr[1] = buf[1]; ptr[2] = buf[2]; ptr[3] = buf[3];
    ptr[4] = buf[4]; ptr[5] = buf[5]; ptr[6] = buf[6]; ptr[7] = buf[7];
}

template <int N = VTraits<v_uint32>::max_nlanes>
inline void v_store(unsigned* ptr, const v_uint32& a)
{
    unsigned buf[VTraits<v_uint32>::max_nlanes];
    v_store(buf, a);
    for (int i = 0; i < N; i++) {
        ptr[i] = buf[i];
    }
}

template <> inline void v_store<4>(unsigned* ptr, const v_uint32& a)
{
    unsigned buf[VTraits<v_uint32>::max_nlanes];
    v_store(buf, a);
    ptr[0] = buf[0]; ptr[1] = buf[1];
    ptr[2] = buf[2]; ptr[3] = buf[3];
}

template <int N = VTraits<v_int32>::max_nlanes>
inline void v_store(int* ptr, const v_int32& a)
{
    int buf[VTraits<v_int32>::max_nlanes];
    v_store(buf, a);
    for (int i = 0; i < N; i++) {
        ptr[i] = buf[i];
    }
}

template <> inline void v_store<4>(int* ptr, const v_int32& a)
{
    int buf[VTraits<v_int32>::max_nlanes];
    v_store(buf, a);
    ptr[0] = buf[0]; ptr[1] = buf[1];
    ptr[2] = buf[2]; ptr[3] = buf[3];
}

template <int N = VTraits<v_float32>::max_nlanes>
inline void v_store(float* ptr, const v_float32& a)
{
    float buf[VTraits<v_float32>::max_nlanes];
    v_store(buf, a);
    for (int i = 0; i < N; i++) {
        ptr[i] = buf[i];
    }
}
template <> inline void v_store<4>(float* ptr, const v_float32& a)
{
    float buf[VTraits<v_float32>::max_nlanes];
    v_store(buf, a);
    ptr[0] = buf[0]; ptr[1] = buf[1];
    ptr[2] = buf[2]; ptr[3] = buf[3];
}

////////////// Lookup table access ////////////////////
#define OPENCV_HAL_IMPL_RVV_LUT(_Tpvec, _Tp, suffix) \
inline _Tpvec v_lut(const _Tp* tab, const int* idx) \
{ \
    auto vidx = __riscv_vmul(__riscv_vreinterpret_u32##suffix(__riscv_vle32_v_i32##suffix(idx, VTraits<_Tpvec>::vlanes())), sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return __riscv_vloxei32(tab, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_LUT(v_int8, schar, m8)
#define OPENCV_HAL_IMPL_RVV_LUT_FP16(_Tpvec, _Tp, suffix) \
inline _Tpvec v_lut(const _Tp* tab, const int* idx) \
{ \
    auto vidx = __riscv_vmul(__riscv_vreinterpret_u32##suffix(__riscv_vle32_v_i32##suffix(idx, VTraits<_Tpvec>::vlanes())), sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return __riscv_vloxei32((_Float16*)tab, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_LUT(v_int16, short, m4)
OPENCV_HAL_IMPL_RVV_LUT(v_int32, int, m2)
OPENCV_HAL_IMPL_RVV_LUT(v_int64, int64_t, m1)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_LUT_FP16(v_float16, hfloat, m4)
#endif
OPENCV_HAL_IMPL_RVV_LUT(v_float32, float, m2)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_LUT(v_float64, double, m1)
#endif

#define OPENCV_HAL_IMPL_RVV_LUT_PAIRS(_Tpvec, _Tp, _TpCast, suffix1, suffix2, v_trunc) \
inline _Tpvec v_lut_pairs(const _Tp* tab, const int* idx) \
{ \
    auto v0 = __riscv_vle32_v_u32##suffix1((unsigned*)idx, VTraits<_Tpvec>::vlanes()/2); \
    auto v1 = __riscv_vadd(v0, 1, VTraits<_Tpvec>::vlanes()/2); \
    auto w0 = __riscv_vwcvtu_x(v0, VTraits<_Tpvec>::vlanes()/2); \
    auto w1 = __riscv_vwcvtu_x(v1, VTraits<_Tpvec>::vlanes()/2); \
    auto sh1 = __riscv_vslide1up(v_trunc(__riscv_vreinterpret_u32##suffix2(w1)),0, VTraits<_Tpvec>::vlanes()); \
    auto vid = __riscv_vor(sh1, v_trunc(__riscv_vreinterpret_u32##suffix2(w0)), VTraits<_Tpvec>::vlanes()); \
    auto vidx = __riscv_vmul(vid, sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return __riscv_vloxei32((_TpCast *)tab, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_int8, schar, schar, m4, m8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_int16, short, short, m2, m4, OPENCV_HAL_NOP)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_float16, hfloat, _Float16, m2, m4, OPENCV_HAL_NOP)
#endif
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_int32, int, int, m1, m2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_float32, float, float, m1, m2, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_int64, int64_t, int64_t, m1, m2, __riscv_vlmul_trunc_u32m1)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_LUT_PAIRS(v_float64, double, double, m1, m2, __riscv_vlmul_trunc_u32m1)
#endif


#define OPENCV_HAL_IMPL_RVV_LUT_QUADS(_Tpvec, _Tp, _TpCast, suffix0, suffix1, suffix2, v_trunc) \
inline _Tpvec v_lut_quads(const _Tp* tab, const int* idx) \
{ \
    auto v0 = __riscv_vle32_v_u32##suffix0((unsigned*)idx, VTraits<_Tpvec>::vlanes()/4); \
    auto v1 = __riscv_vadd(v0, 1, VTraits<_Tpvec>::vlanes()/4); \
    auto v2 = __riscv_vadd(v0, 2, VTraits<_Tpvec>::vlanes()/4); \
    auto v3 = __riscv_vadd(v0, 3, VTraits<_Tpvec>::vlanes()/4); \
    auto w0 = __riscv_vwcvtu_x(v0, VTraits<_Tpvec>::vlanes()/4); \
    auto w1 = __riscv_vwcvtu_x(v1, VTraits<_Tpvec>::vlanes()/4); \
    auto w2 = __riscv_vwcvtu_x(v2, VTraits<_Tpvec>::vlanes()/4); \
    auto w3 = __riscv_vwcvtu_x(v3, VTraits<_Tpvec>::vlanes()/4); \
    auto sh2 = __riscv_vslide1up(__riscv_vreinterpret_u32##suffix1(w2),0, VTraits<_Tpvec>::vlanes()/2); \
    auto sh3 = __riscv_vslide1up(__riscv_vreinterpret_u32##suffix1(w3),0, VTraits<_Tpvec>::vlanes()/2); \
    auto vid0 = __riscv_vor(sh2, __riscv_vreinterpret_u32##suffix1(w0), VTraits<_Tpvec>::vlanes()/2); \
    auto vid1 = __riscv_vor(sh3, __riscv_vreinterpret_u32##suffix1(w1), VTraits<_Tpvec>::vlanes()/2); \
    auto wid0 = __riscv_vwcvtu_x(v_trunc(vid0), VTraits<_Tpvec>::vlanes()/2); \
    auto wid1 = __riscv_vwcvtu_x(v_trunc(vid1), VTraits<_Tpvec>::vlanes()/2); \
    auto shwid1 = __riscv_vslide1up(__riscv_vreinterpret_u32##suffix2(wid1),0, VTraits<_Tpvec>::vlanes()); \
    auto vid = __riscv_vor(shwid1, __riscv_vreinterpret_u32##suffix2(wid0), VTraits<_Tpvec>::vlanes()); \
    auto vidx = __riscv_vmul(vid, sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return __riscv_vloxei32((_TpCast *)tab, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_LUT_QUADS(v_int8, schar, schar, m2, m4, m8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_LUT_QUADS(v_int16, short, short, m1 , m2, m4, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_LUT_QUADS(v_int32, int, int, m1, m2, m2, __riscv_vlmul_trunc_u32m1)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_LUT_QUADS(v_float16, hfloat, _Float16, m1 , m2, m4, OPENCV_HAL_NOP)
#endif
OPENCV_HAL_IMPL_RVV_LUT_QUADS(v_float32, float, float, m1, m2, m2, __riscv_vlmul_trunc_u32m1)

#define OPENCV_HAL_IMPL_RVV_LUT_VEC(_Tpvec, _Tp) \
inline _Tpvec v_lut(const _Tp* tab, const v_int32& vidx) \
{ \
    v_uint32 vidx_ = __riscv_vmul(__riscv_vreinterpret_u32m2(vidx), sizeof(_Tp), VTraits<v_int32>::vlanes()); \
    return __riscv_vloxei32(tab, vidx_, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_LUT_VEC(v_float32, float)
OPENCV_HAL_IMPL_RVV_LUT_VEC(v_int32, int)
OPENCV_HAL_IMPL_RVV_LUT_VEC(v_uint32, unsigned)

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_lut(const double* tab, const v_int32& vidx) \
{ \
    vuint32m1_t vidx_ = __riscv_vmul(__riscv_vlmul_trunc_u32m1(__riscv_vreinterpret_u32m2(vidx)), sizeof(double), VTraits<v_float64>::vlanes()); \
    return __riscv_vloxei32(tab, vidx_, VTraits<v_float64>::vlanes()); \
}
#endif


inline v_uint8 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }
inline v_uint16 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }
inline v_uint32 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }
inline v_uint64 v_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64 v_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

////////////// Pack boolean ////////////////////
inline v_uint8 v_pack_b(const v_uint16& a, const v_uint16& b)
{
    return __riscv_vnsrl(__riscv_vset(__riscv_vlmul_ext_v_u16m2_u16m4(a),1,b), 0, VTraits<v_uint8>::vlanes());
}

inline v_uint8 v_pack_b(const v_uint32& a, const v_uint32& b,
                           const v_uint32& c, const v_uint32& d)
{

    return __riscv_vnsrl(__riscv_vnsrl(__riscv_vset(__riscv_vset(__riscv_vset(__riscv_vlmul_ext_u32m8(a),1,b),2,c),3,d), 0, VTraits<v_uint8>::vlanes()), 0, VTraits<v_uint8>::vlanes());
}

inline v_uint8 v_pack_b(const v_uint64& a, const v_uint64& b, const v_uint64& c,
                           const v_uint64& d, const v_uint64& e, const v_uint64& f,
                           const v_uint64& g, const v_uint64& h)
{
    vuint8m1_t t0 = __riscv_vnsrl(__riscv_vnsrl(__riscv_vnsrl(__riscv_vset(__riscv_vset(__riscv_vset(__riscv_vlmul_ext_u64m8(a),1,b),2,c),3,d), 0, VTraits<v_uint8>::vlanes()), 0, VTraits<v_uint8>::vlanes()), 0, VTraits<v_uint8>::vlanes());
    vuint8m1_t t1 = __riscv_vnsrl(__riscv_vnsrl(__riscv_vnsrl(__riscv_vset(__riscv_vset(__riscv_vset(__riscv_vlmul_ext_u64m8(e),1,f),2,g),3,h), 0, VTraits<v_uint8>::vlanes()), 0, VTraits<v_uint8>::vlanes()), 0, VTraits<v_uint8>::vlanes());

    return __riscv_vset(__riscv_vlmul_ext_u8m2(t0), 1, t1);
}

////////////// Arithmetics //////////////
#define OPENCV_HAL_IMPL_RVV_BIN_OP(_Tpvec, ocv_intrin, rvv_intrin) \
inline _Tpvec v_##ocv_intrin(const _Tpvec& a, const _Tpvec& b) \
{ \
    return rvv_intrin(a, b, VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, add, __riscv_vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, sub, __riscv_vssubu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, add, __riscv_vsadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, sub, __riscv_vssub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, add, __riscv_vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, sub, __riscv_vssubu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, add, __riscv_vsadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, sub, __riscv_vssub)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float16, add, __riscv_vfadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float16, sub, __riscv_vfsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float16, mul, __riscv_vfmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float16, div, __riscv_vfdiv)
#endif
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, add, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, sub, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, mul, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, add, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, sub, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, mul, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, add, __riscv_vfadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, sub, __riscv_vfsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, mul, __riscv_vfmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, div, __riscv_vfdiv)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint64, add, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint64, sub, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int64, add, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int64, sub, __riscv_vsub)

#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, add, __riscv_vfadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, sub, __riscv_vfsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, mul, __riscv_vfmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, div, __riscv_vfdiv)
#endif

#define OPENCV_HAL_IMPL_RVV_BIN_MADD(_Tpvec, rvv_add) \
template<typename... Args> \
inline _Tpvec v_add(const _Tpvec& f1, const _Tpvec& f2, const Args&... vf) { \
    return v_add(rvv_add(f1, f2, VTraits<_Tpvec>::vlanes()), vf...); \
}
#define OPENCV_HAL_IMPL_RVV_BIN_MMUL(_Tpvec, rvv_mul) \
template<typename... Args> \
inline _Tpvec v_mul(const _Tpvec& f1, const _Tpvec& f2, const Args&... vf) { \
    return v_mul(rvv_mul(f1, f2, VTraits<_Tpvec>::vlanes()), vf...); \
}
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint8, __riscv_vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int8, __riscv_vsadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint16, __riscv_vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int16, __riscv_vsadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint32, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int32, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_float32, __riscv_vfadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint64, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int64, __riscv_vadd)

OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_uint32, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_int32, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_float32, __riscv_vfmul)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_float16, __riscv_vfadd)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_float16, __riscv_vfmul)
#endif
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_float64, __riscv_vfadd)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_float64, __riscv_vfmul)
#endif

#define OPENCV_HAL_IMPL_RVV_MUL_EXPAND(_Tpvec, _Tpwvec, _TpwvecM2, suffix, wmul) \
inline void v_mul_expand(const _Tpvec& a, const _Tpvec& b, _Tpwvec& c, _Tpwvec& d) \
{ \
    _TpwvecM2 temp = wmul(a, b, VTraits<_Tpvec>::vlanes()); \
    c = __riscv_vget_##suffix##m2(temp, 0); \
    d = __riscv_vget_##suffix##m2(temp, 1); \
}

OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint8, v_uint16, vuint16m4_t, u16, __riscv_vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int8, v_int16, vint16m4_t, i16, __riscv_vwmul)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint16, v_uint32, vuint32m4_t, u32, __riscv_vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int16, v_int32, vint32m4_t, i32, __riscv_vwmul)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint32, v_uint64, vuint64m4_t, u64, __riscv_vwmulu)

inline v_int16 v_mul_hi(const v_int16& a, const v_int16& b)
{
    return __riscv_vmulh(a, b, VTraits<v_int16>::vlanes());
}
inline v_uint16 v_mul_hi(const v_uint16& a, const v_uint16& b)
{
    return __riscv_vmulhu(a, b, VTraits<v_uint16>::vlanes());
}

////////////// Arithmetics (wrap)//////////////
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, add_wrap, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, add_wrap, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, add_wrap, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, add_wrap, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, sub_wrap, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, sub_wrap, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, sub_wrap, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, sub_wrap, __riscv_vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, mul_wrap, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, mul_wrap, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, mul_wrap, __riscv_vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, mul_wrap, __riscv_vmul)

//////// Saturating Multiply ////////
#define OPENCV_HAL_IMPL_RVV_MUL_SAT(_Tpvec, _clip, _wmul) \
inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _clip(_wmul(a, b, VTraits<_Tpvec>::vlanes()), 0, 0, VTraits<_Tpvec>::vlanes()); \
} \
template<typename... Args> \
inline _Tpvec v_mul(const _Tpvec& a1, const _Tpvec& a2, const Args&... va) { \
    return v_mul(_clip(_wmul(a1, a2, VTraits<_Tpvec>::vlanes()), 0, 0, VTraits<_Tpvec>::vlanes()), va...); \
}

OPENCV_HAL_IMPL_RVV_MUL_SAT(v_uint8, __riscv_vnclipu, __riscv_vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_int8, __riscv_vnclip, __riscv_vwmul)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_uint16, __riscv_vnclipu, __riscv_vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_int16, __riscv_vnclip, __riscv_vwmul)

////////////// Bitwise logic //////////////

#define OPENCV_HAL_IMPL_RVV_LOGIC_OP(_Tpvec, vl) \
inline _Tpvec v_and(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vand(a, b, vl); \
} \
inline _Tpvec v_or(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vor(a, b, vl); \
} \
inline _Tpvec v_xor(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vxor(a, b, vl); \
} \
inline _Tpvec v_not (const _Tpvec& a) \
{ \
    return __riscv_vnot(a, vl); \
}

OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int64, VTraits<v_int64>::vlanes())

#if CV_SIMD_SCALABLE_FP16
#define OPENCV_HAL_IMPL_RVV_FLT16_BIT_OP(intrin) \
inline v_float16 intrin (const v_float16& a, const v_float16& b) \
{ \
    return __riscv_vreinterpret_f16m2(intrin(__riscv_vreinterpret_i16m2(a), __riscv_vreinterpret_i16m2(b))); \
}
OPENCV_HAL_IMPL_RVV_FLT16_BIT_OP(v_and)
OPENCV_HAL_IMPL_RVV_FLT16_BIT_OP(v_or)
OPENCV_HAL_IMPL_RVV_FLT16_BIT_OP(v_xor)

inline v_float16 v_not (const v_float16& a) \
{ \
    return __riscv_vreinterpret_f16m2(v_not(__riscv_vreinterpret_i16m2(a))); \
}
#endif

#define OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(intrin) \
inline v_float32 intrin (const v_float32& a, const v_float32& b) \
{ \
    return __riscv_vreinterpret_f32m2(intrin(__riscv_vreinterpret_i32m2(a), __riscv_vreinterpret_i32m2(b))); \
}
OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(v_and)
OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(v_or)
OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(v_xor)

inline v_float32 v_not (const v_float32& a) \
{ \
    return __riscv_vreinterpret_f32m2(v_not(__riscv_vreinterpret_i32m2(a))); \
}

#if CV_SIMD_SCALABLE_64F
#define OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(intrin) \
inline v_float64 intrin (const v_float64& a, const v_float64& b) \
{ \
    return __riscv_vreinterpret_f64m2(intrin(__riscv_vreinterpret_i64m2(a), __riscv_vreinterpret_i64m2(b))); \
}
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(v_and)
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(v_or)
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(v_xor)

inline v_float64 v_not (const v_float64& a) \
{ \
    return __riscv_vreinterpret_f64m2(v_not(__riscv_vreinterpret_i64m2(a))); \
}
#endif


////////////// Bitwise shifts //////////////
/*  Usage
1. v_shl<N>(vec);
2. v_shl(vec, N); // instead of vec << N, when N is non-constant.
*/

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(_Tpvec, vl) \
template<int s = 0> inline _Tpvec v_shl(const _Tpvec& a, int n = s) \
{ \
    return _Tpvec(__riscv_vsll(a, uint8_t(n), vl)); \
} \
template<int s = 0> inline _Tpvec v_shr(const _Tpvec& a, int n = s) \
{ \
    return _Tpvec(__riscv_vsrl(a, uint8_t(n), vl)); \
}

#define OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(_Tpvec, vl) \
template<int s = 0> inline _Tpvec v_shl(const _Tpvec& a, int n = s) \
{ \
    return _Tpvec(__riscv_vsll(a, uint8_t(n), vl)); \
} \
template<int s = 0> inline _Tpvec v_shr(const _Tpvec& a, int n = s) \
{ \
    return _Tpvec(__riscv_vsra(a, uint8_t(n), vl)); \
}

OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int64, VTraits<v_int64>::vlanes())

////////////// Comparison //////////////
#define OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, op, intrin, suffix) \
inline _Tpvec v_##op(const _Tpvec& a, const _Tpvec& b) \
{ \
    size_t VLEN = VTraits<_Tpvec>::vlanes(); \
    uint64_t ones = -1; \
    return __riscv_vmerge(__riscv_vmv_v_x_##suffix##m2(0, VLEN), ones, intrin(a, b, VLEN), VLEN); \
}

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, op, intrin, suffix) \
inline _Tpvec v_##op (const _Tpvec& a, const _Tpvec& b) \
{ \
    size_t VLEN = VTraits<_Tpvec>::vlanes(); \
    union { uint64_t u; VTraits<_Tpvec>::lane_type d; } ones; \
    ones.u = -1; \
    auto diff = intrin(a, b, VLEN); \
    auto z = __riscv_vfmv_v_f_##suffix##m2(0, VLEN); \
    auto res = __riscv_vfmerge(z, ones.d, diff, VLEN); \
    return _Tpvec(res); \
} //TODO

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, op, intrin, suffix) \
inline _Tpvec v_##op (const _Tpvec& a, const _Tpvec& b) \
{ \
    size_t VLEN = VTraits<_Tpvec>::vlanes(); \
    union { uint64_t u; _Float16 d; } ones; \
    ones.u = -1; \
    auto diff = intrin(a, b, VLEN); \
    auto z = __riscv_vfmv_v_f_##suffix##m2(0, VLEN); \
    auto res = __riscv_vfmerge(z, ones.d, diff, VLEN); \
    return _Tpvec(res); \
} //TODO

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(_Tpvec, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, eq, __riscv_vmseq, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ne, __riscv_vmsne, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, lt, __riscv_vmsltu, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, gt, __riscv_vmsgtu, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, le, __riscv_vmsleu, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ge, __riscv_vmsgeu, suffix)

#define OPENCV_HAL_IMPL_RVV_SIGNED_CMP(_Tpvec, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, eq, __riscv_vmseq, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ne, __riscv_vmsne, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, lt, __riscv_vmslt, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, gt, __riscv_vmsgt, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, le, __riscv_vmsle, suffix) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ge, __riscv_vmsge, suffix)

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP(_Tpvec, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, eq, __riscv_vmfeq, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, ne, __riscv_vmfne, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, lt, __riscv_vmflt, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, gt, __riscv_vmfgt, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, le, __riscv_vmfle, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, ge, __riscv_vmfge, suffix)

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP_FP16(_Tpvec, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, eq, __riscv_vmfeq, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, ne, __riscv_vmfne, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, lt, __riscv_vmflt, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, gt, __riscv_vmfgt, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, le, __riscv_vmfle, suffix) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP_FP16(_Tpvec, ge, __riscv_vmfge, suffix)

OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint8, u8)
OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint16, u16)
OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint64, u64)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int8, i8)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int16, i16)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int32, i32)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int64, i64)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_FP16(v_float16, f16)
#endif
OPENCV_HAL_IMPL_RVV_FLOAT_CMP(v_float32, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_FLOAT_CMP(v_float64, f64)
#endif

#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_not_nan(const v_float16& a)
{ return v_eq(a, a); }
#endif

inline v_float32 v_not_nan(const v_float32& a)
{ return v_eq(a, a); }

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_not_nan(const v_float64& a)
{ return v_eq(a, a); }
#endif

////////////// Min/Max //////////////

#define OPENCV_HAL_IMPL_RVV_BIN_FUNC(_Tpvec, func, intrin, vl) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return intrin(a, b, vl); \
}

OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8, v_min, __riscv_vminu, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8, v_max, __riscv_vmaxu, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8, v_min, __riscv_vmin, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8, v_max, __riscv_vmax, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16, v_min, __riscv_vminu, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16, v_max, __riscv_vmaxu, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16, v_min, __riscv_vmin, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16, v_max, __riscv_vmax, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32, v_min, __riscv_vminu, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32, v_max, __riscv_vmaxu, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32, v_min, __riscv_vmin, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32, v_max, __riscv_vmax, VTraits<v_int32>::vlanes())
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float16, v_min, __riscv_vfmin, VTraits<v_float16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float16, v_max, __riscv_vfmax, VTraits<v_float16>::vlanes())
#endif
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32, v_min, __riscv_vfmin, VTraits<v_float32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32, v_max, __riscv_vfmax, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_min, __riscv_vfmin, VTraits<v_float64>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_max, __riscv_vfmax, VTraits<v_float64>::vlanes())
#endif

////////////// Transpose4x4 //////////////
#define OPENCV_HAL_IMPL_RVV_ZIP4(_Tpvec, _wTpvec, suffix, convert2u, convert) \
inline void v_zip4(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) { \
    int vl = 4; \
    _wTpvec temp = __riscv_vreinterpret_##suffix##m4(convert2u( \
        __riscv_vor(__riscv_vzext_vf2(convert(a0), vl), \
            __riscv_vreinterpret_u64m4(__riscv_vslide1up(__riscv_vreinterpret_u32m4(__riscv_vzext_vf2(convert(a1), vl)), 0, vl*2)), \
            vl))); \
    b0 = __riscv_vget_##suffix##m2(temp, 0); \
    b1 = __riscv_vget_##suffix##m2(__riscv_vrgather(temp, __riscv_vadd(__riscv_vid_v_u32m4(vl), 4, vl)/*{4,5,6,7} */, vl) ,0); \
}

OPENCV_HAL_IMPL_RVV_ZIP4(v_uint32, vuint32m4_t, u32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP4(v_int32, vint32m4_t, i32, __riscv_vreinterpret_u32m4, __riscv_vreinterpret_u32m2)
OPENCV_HAL_IMPL_RVV_ZIP4(v_float32, vfloat32m4_t, f32, __riscv_vreinterpret_u32m4, __riscv_vreinterpret_u32m2)


#define OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(_Tpvec, suffix) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, const _Tpvec& a2, const _Tpvec& a3, _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3) { \
    _Tpvec t0,t1,t2,t3; \
    v_zip4(a0, a2, t0, t2); \
    v_zip4(a1, a3, t1, t3); \
    v_zip4(t0, t1, b0, b1); \
    v_zip4(t2, t3, b2, b3); \
}

OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(v_int32, i32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(v_float32, f32)

////////////// Reduce //////////////

#define OPENCV_HAL_IMPL_RVV_REDUCE_SUM(_Tpvec, _wTpvec, _nwTpvec, scalartype, wsuffix, vl, red) \
inline scalartype v_reduce_sum(const _Tpvec& a)  \
{ \
    _nwTpvec zero = __riscv_vmv_v_x_##wsuffix##m1(0, vl); \
    _nwTpvec res = __riscv_vmv_v_x_##wsuffix##m1(0, vl); \
    res = __riscv_v##red(a, zero, vl); \
    return (scalartype)__riscv_vmv_x(res); \
}
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint8, v_uint16, vuint16m1_t, unsigned, u16, VTraits<v_uint8>::vlanes(), wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int8, v_int16, vint16m1_t, int, i16, VTraits<v_int8>::vlanes(), wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint16, v_uint32, vuint32m1_t, unsigned, u32, VTraits<v_uint16>::vlanes(), wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int16, v_int32, vint32m1_t, int, i32, VTraits<v_int16>::vlanes(), wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint32, v_uint64, vuint64m1_t, unsigned, u64, VTraits<v_uint32>::vlanes(), wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int32, v_int64, vint64m1_t, int, i64, VTraits<v_int32>::vlanes(), wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint64, v_uint64, vuint64m1_t, uint64, u64, VTraits<v_uint64>::vlanes(), redsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int64, v_int64, vint64m1_t, int64, i64, VTraits<v_int64>::vlanes(), redsum)


#define OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(_Tpvec, _wTpvec, _nwTpvec, scalartype, wsuffix, vl) \
inline scalartype v_reduce_sum(const _Tpvec& a)  \
{ \
    _nwTpvec zero = __riscv_vfmv_v_f_##wsuffix##m1(0, vl); \
    _nwTpvec res = __riscv_vfmv_v_f_##wsuffix##m1(0, vl); \
    res = __riscv_vfredusum(a, zero, vl); \
    return (scalartype)__riscv_vfmv_f(res); \
}
OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(v_float32, v_float32, vfloat32m1_t, float, f32, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(v_float64, v_float64, vfloat64m1_t, float, f64, VTraits<v_float64>::vlanes())
#endif

#define OPENCV_HAL_IMPL_RVV_REDUCE(_Tpvec, _nTpvec, func, scalartype, suffix, vl, red) \
inline scalartype v_reduce_##func(const _Tpvec& a)  \
{ \
    _nTpvec narrowM1 = __riscv_vlmul_trunc_##suffix##m1(a); \
    return (scalartype)__riscv_vmv_x(__riscv_v##red(a, narrowM1, vl)); \
}

#define OPENCV_HAL_IMPL_RVV_REDUCE_FP(_Tpvec, _nTpvec, func, scalartype, suffix, vl, red) \
inline scalartype v_reduce_##func(const _Tpvec& a)  \
{ \
    _nTpvec narrowM1 = __riscv_vlmul_trunc_##suffix##m1(a); \
    return (scalartype)__riscv_vfmv_f(__riscv_v##red(a, narrowM1, vl)); \
}

OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8, vuint8m1_t, min, uchar, u8, VTraits<v_uint8>::vlanes(), redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8, vint8m1_t, min, schar, i8, VTraits<v_int8>::vlanes(), redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16, vuint16m1_t, min, ushort, u16, VTraits<v_uint16>::vlanes(), redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16, vint16m1_t, min, short, i16, VTraits<v_int16>::vlanes(), redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32, vuint32m1_t, min, unsigned, u32, VTraits<v_uint32>::vlanes(), redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32, vint32m1_t, min, int, i32, VTraits<v_int32>::vlanes(), redmin)
OPENCV_HAL_IMPL_RVV_REDUCE_FP(v_float32, vfloat32m1_t, min, float, f32, VTraits<v_float32>::vlanes(), fredmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8,  vuint8m1_t, max, uchar, u8, VTraits<v_uint8>::vlanes(), redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8,  vint8m1_t, max, schar, i8, VTraits<v_int8>::vlanes(), redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16,  vuint16m1_t, max, ushort, u16, VTraits<v_uint16>::vlanes(), redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16,  vint16m1_t, max, short, i16, VTraits<v_int16>::vlanes(), redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32,  vuint32m1_t, max, unsigned, u32, VTraits<v_uint32>::vlanes(), redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32,  vint32m1_t, max, int, i32, VTraits<v_int32>::vlanes(), redmax)
OPENCV_HAL_IMPL_RVV_REDUCE_FP(v_float32,  vfloat32m1_t, max, float, f32, VTraits<v_float32>::vlanes(), fredmax)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_REDUCE_FP(v_float16, vfloat16m1_t, max, hfloat, f16, VTraits<v_float16>::vlanes(), fredmax)
OPENCV_HAL_IMPL_RVV_REDUCE_FP(v_float16, vfloat16m1_t, min, hfloat, f16, VTraits<v_float16>::vlanes(), fredmin)
#endif

inline v_float32 v_reduce_sum4(const v_float32& a, const v_float32& b,
                                 const v_float32& c, const v_float32& d)
{
    // 0000 1111 2222 3333 ....
    vuint64m4_t vid1 = __riscv_vid_v_u64m4(VTraits<vuint32m2_t>::vlanes());
    vuint16m4_t t1 = __riscv_vreinterpret_u16m4(vid1);
    vuint16m4_t t2 = __riscv_vslide1up(t1, 0, VTraits<vuint8m2_t>::vlanes());
    vuint16m4_t t3 = __riscv_vslide1up(t2, 0, VTraits<vuint8m2_t>::vlanes());
    vuint16m4_t t4 = __riscv_vslide1up(t3, 0, VTraits<vuint8m2_t>::vlanes());
    t1 = __riscv_vor(
        __riscv_vor(t1, t2, VTraits<vuint8m2_t>::vlanes()),
        __riscv_vor(t3, t4, VTraits<vuint8m2_t>::vlanes()),
        VTraits<vuint8m2_t>::vlanes()
    );

    // index for transpose4X4
    vuint16m4_t vidx0 = __riscv_vmul(t1, 12, VTraits<vuint8m2_t>::vlanes());
    vidx0 = __riscv_vadd(vidx0, __riscv_vid_v_u16m4(VTraits<vuint8m2_t>::vlanes()), VTraits<vuint8m2_t>::vlanes());
    vuint16m4_t vidx1 = __riscv_vadd(vidx0, 4, VTraits<vuint8m2_t>::vlanes());
    vuint16m4_t vidx2 = __riscv_vadd(vidx0, 8, VTraits<vuint8m2_t>::vlanes());
    vuint16m4_t vidx3 = __riscv_vadd(vidx0, 12, VTraits<vuint8m2_t>::vlanes());

    // zip
    vuint32m4_t tempA = __riscv_vreinterpret_u32m4( \
        __riscv_vor(__riscv_vzext_vf2(__riscv_vreinterpret_u32m2(a), VTraits<vuint16m2_t>::vlanes()), \
            __riscv_vreinterpret_u64m4(__riscv_vslide1up(__riscv_vreinterpret_u32m4(__riscv_vzext_vf2(__riscv_vreinterpret_u32m2(c), VTraits<vuint16m2_t>::vlanes())), 0, VTraits<vuint16m2_t>::vlanes())), \
            VTraits<vuint32m2_t>::vlanes())); \
    vuint32m4_t tempB = __riscv_vreinterpret_u32m4( \
        __riscv_vor(__riscv_vzext_vf2(__riscv_vreinterpret_u32m2(b), VTraits<vuint16m2_t>::vlanes()), \
            __riscv_vreinterpret_u64m4(__riscv_vslide1up(__riscv_vreinterpret_u32m4(__riscv_vzext_vf2(__riscv_vreinterpret_u32m2(d), VTraits<vuint16m2_t>::vlanes())), 0, VTraits<vuint16m2_t>::vlanes())), \
            VTraits<vuint32m2_t>::vlanes())); \
    vfloat32m8_t temp = __riscv_vreinterpret_f32m8(__riscv_vreinterpret_u32m8( \
        __riscv_vor(__riscv_vzext_vf2(tempA, VTraits<vuint8m2_t>::vlanes()), \
            __riscv_vreinterpret_u64m8(__riscv_vslide1up(__riscv_vreinterpret_u32m8(__riscv_vzext_vf2(tempB, VTraits<vuint8m2_t>::vlanes())), 0, VTraits<vuint8m2_t>::vlanes())), \
            VTraits<vuint16m2_t>::vlanes())));

    // transpose
    vfloat32m2_t b0 = __riscv_vlmul_trunc_f32m2(__riscv_vrgatherei16(temp, vidx0, VTraits<vuint8m2_t>::vlanes()));
    vfloat32m2_t b1 = __riscv_vlmul_trunc_f32m2(__riscv_vrgatherei16(temp, vidx1, VTraits<vuint8m2_t>::vlanes()));
    vfloat32m2_t b2 = __riscv_vlmul_trunc_f32m2(__riscv_vrgatherei16(temp, vidx2, VTraits<vuint8m2_t>::vlanes()));
    vfloat32m2_t b3 = __riscv_vlmul_trunc_f32m2(__riscv_vrgatherei16(temp, vidx3, VTraits<vuint8m2_t>::vlanes()));

    // vector add
    v_float32 res = __riscv_vfadd(
        __riscv_vfadd(b0, b1, VTraits<vfloat32m2_t>::vlanes()),
        __riscv_vfadd(b2, b3, VTraits<vfloat32m2_t>::vlanes()),
        VTraits<vfloat32m2_t>::vlanes()
    );
    return res;
}

////////////// Square-Root //////////////
#define OPENCV_HAL_IMPL_RVV_SQR_FP(_Tpvec, _setAllFunc) \
inline _Tpvec v_sqrt(const _Tpvec& x) \
{ \
    return __riscv_vfsqrt(x, VTraits<_Tpvec>::vlanes()); \
} \
inline _Tpvec v_invsqrt(const _Tpvec& x) \
{ \
    return v_div(_setAllFunc(1.0f), v_sqrt(x)); \
} \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpvec x = __riscv_vfmacc(__riscv_vfmul(a, a, VTraits<_Tpvec>::vlanes()), b, b, VTraits<_Tpvec>::vlanes()); \
    return v_sqrt(x); \
} \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vfmacc(__riscv_vfmul(a, a, VTraits<_Tpvec>::vlanes()), b, b, VTraits<_Tpvec>::vlanes()); \
}

#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_SQR_FP(v_float16, v_setall_f16)
#endif
OPENCV_HAL_IMPL_RVV_SQR_FP(v_float32, v_setall_f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_SQR_FP(v_float64, v_setall_f64)
#endif

////////////// Multiply-Add //////////////

inline v_float32 v_fma(const v_float32& a, const v_float32& b, const v_float32& c)
{
    return __riscv_vfmacc(c, a, b, VTraits<v_float32>::vlanes());
}
inline v_int32 v_fma(const v_int32& a, const v_int32& b, const v_int32& c)
{
    return __riscv_vmacc(c, a, b, VTraits<v_float32>::vlanes());
}

inline v_float32 v_muladd(const v_float32& a, const v_float32& b, const v_float32& c)
{
    return v_fma(a, b, c);
}

inline v_int32 v_muladd(const v_int32& a, const v_int32& b, const v_int32& c)
{
    return v_fma(a, b, c);
}

#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_fma(const v_float16& a, const v_float16& b, const v_float16& c)
{
    return __riscv_vfmacc(c, a, b, VTraits<v_float16>::vlanes());
}

inline v_float16 v_muladd(const v_float16& a, const v_float16& b, const v_float16& c)
{
    return v_fma(a, b, c);
}
#endif

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_fma(const v_float64& a, const v_float64& b, const v_float64& c)
{
    return __riscv_vfmacc_vv_f64m2(c, a, b, VTraits<v_float64>::vlanes());
}

inline v_float64 v_muladd(const v_float64& a, const v_float64& b, const v_float64& c)
{
    return v_fma(a, b, c);
}
#endif

////////////// Check all/any //////////////

#define OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(_Tpvec, vl) \
inline bool v_check_all(const _Tpvec& a) \
{ \
    return (int)__riscv_vcpop(__riscv_vmslt(a, 0, vl), vl) == vl; \
} \
inline bool v_check_any(const _Tpvec& a) \
{ \
    return (int)__riscv_vcpop(__riscv_vmslt(a, 0, vl), vl) != 0; \
}

OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int64, VTraits<v_int64>::vlanes())


inline bool v_check_all(const v_uint8& a)
{ return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint8& a)
{ return v_check_any(v_reinterpret_as_s8(a)); }

inline bool v_check_all(const v_uint16& a)
{ return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint16& a)
{ return v_check_any(v_reinterpret_as_s16(a)); }

#if CV_SIMD_SCALABLE_FP16
inline bool v_check_all(const v_float16& a)
{ return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_float16& a)
{ return v_check_any(v_reinterpret_as_s16(a)); }
#endif

inline bool v_check_all(const v_uint32& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_uint32& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }

inline bool v_check_all(const v_float32& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float32& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }

inline bool v_check_all(const v_uint64& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_uint64& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }

#if CV_SIMD_SCALABLE_64F
inline bool v_check_all(const v_float64& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_float64& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }
#endif

////////////// abs //////////////

#define OPENCV_HAL_IMPL_RVV_ABSDIFF(_Tpvec, abs) \
inline _Tpvec v_##abs(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_sub(v_max(a, b), v_min(a, b)); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint8, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint16, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint32, absdiff)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float16, absdiff)
#endif
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float32, absdiff)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float64, absdiff)
#endif
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int8, absdiffs)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int16, absdiffs)

#define OPENCV_HAL_IMPL_RVV_ABSDIFF_S(_Tpvec, _rTpvec, width) \
inline _rTpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vnclipu(__riscv_vreinterpret_u##width##m4(__riscv_vwsub_vv(v_max(a, b), v_min(a, b), VTraits<_Tpvec>::vlanes())), 0, 0, VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int8, v_uint8, 16)
OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int16, v_uint16, 32)
OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int32, v_uint32, 64)

#define OPENCV_HAL_IMPL_RVV_ABS(_Tprvec, _Tpvec, suffix) \
inline _Tprvec v_abs(const _Tpvec& a) \
{ \
    return v_absdiff(a, v_setzero_##suffix()); \
}

OPENCV_HAL_IMPL_RVV_ABS(v_uint8, v_int8, s8)
OPENCV_HAL_IMPL_RVV_ABS(v_uint16, v_int16, s16)
OPENCV_HAL_IMPL_RVV_ABS(v_uint32, v_int32, s32)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_ABS(v_float16, v_float16, f16)
#endif
OPENCV_HAL_IMPL_RVV_ABS(v_float32, v_float32, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_ABS(v_float64, v_float64, f64)
#endif


#define OPENCV_HAL_IMPL_RVV_REDUCE_SAD(_Tpvec, scalartype) \
inline scalartype v_reduce_sad(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_reduce_sum(v_absdiff(a, b)); \
}

OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint32, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int32, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_float32, float)

////////////// Select //////////////

#define OPENCV_HAL_IMPL_RVV_SELECT(_Tpvec, vl) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vmerge(b, a, __riscv_vmsne(mask, 0, vl), vl); \
}

OPENCV_HAL_IMPL_RVV_SELECT(v_uint8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_uint16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_uint32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_int8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_int32, VTraits<v_int32>::vlanes())
#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_select(const v_float16& mask, const v_float16& a, const v_float16& b) \
{ \
    return __riscv_vmerge(b, a, __riscv_vmfne(mask, 0, VTraits<v_float16>::vlanes()), VTraits<v_float16>::vlanes()); \
}
#endif

inline v_float32 v_select(const v_float32& mask, const v_float32& a, const v_float32& b) \
{ \
    return __riscv_vmerge(b, a, __riscv_vmfne(mask, 0, VTraits<v_float32>::vlanes()), VTraits<v_float32>::vlanes()); \
}

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_select(const v_float64& mask, const v_float64& a, const v_float64& b) \
{ \
    return __riscv_vmerge(b, a, __riscv_vmfne(mask, 0, VTraits<v_float64>::vlanes()), VTraits<v_float64>::vlanes()); \
}
#endif

////////////// Rotate shift //////////////

#define OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(_Tpvec, suffix, vl) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return __riscv_vslidedown(a, n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return __riscv_vslideup(__riscv_vmv_v_x_##suffix##m2(0, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vslideup(__riscv_vslidedown(a, n, vl), b, VTraits<_Tpvec>::vlanes() - n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vslideup(__riscv_vslidedown(b, VTraits<_Tpvec>::vlanes() - n, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ CV_UNUSED(b); return a; }

OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint8, u8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int8, i8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint16, u16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int16, i16,  VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint32, u32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int32, i32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint64, u64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int64, i64, VTraits<v_int64>::vlanes())

#define OPENCV_HAL_IMPL_RVV_ROTATE_FP(_Tpvec, suffix, vl) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return __riscv_vslidedown(a, n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return __riscv_vslideup(__riscv_vfmv_v_f_##suffix##m2(0, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vslideup(__riscv_vslidedown(a, n, vl), b, VTraits<_Tpvec>::vlanes() - n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vslideup(__riscv_vslidedown(b, VTraits<_Tpvec>::vlanes() - n, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ CV_UNUSED(b); return a; }

#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float16, f16, VTraits<v_float16>::vlanes())
#endif
OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float32, f32, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float64, f64,  VTraits<v_float64>::vlanes())
#endif

////////////// Convert to float //////////////

#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_cvt_f16(const v_float32 &a)
{
    return __riscv_vfncvt_f(__riscv_vlmul_ext_f32m4(a), VTraits<v_float32>::vlanes());
}
inline v_float16 v_cvt_f16(const v_float32 &a, const v_float32 &b)
{
    return __riscv_vfncvt_f(__riscv_vset(__riscv_vlmul_ext_f32m4(a),1,b), VTraits<v_float16>::vlanes());
}
inline v_float16 v_cvt_f16(const v_int16 &a)
{
    return __riscv_vfcvt_f(a, VTraits<v_float16>::vlanes());
}
inline v_float32 v_cvt_f32(const v_float16 &a)
{
    return __riscv_vget_f32m2(__riscv_vfwcvt_f(a, VTraits<v_float16>::vlanes()), 0);
}
inline v_float32 v_cvt_f32_high(const v_float16 &a)
{
    return __riscv_vget_f32m2(__riscv_vfwcvt_f(a, VTraits<v_float16>::vlanes()), 1);
}
#endif

inline v_float32 v_cvt_f32(const v_int32& a)
{
    return __riscv_vfcvt_f_x_v_f32m2(a, VTraits<v_float32>::vlanes());
}

#if CV_SIMD_SCALABLE_64F
inline v_float32 v_cvt_f32(const v_float64& a)
{
    return __riscv_vfncvt_f(__riscv_vlmul_ext_f64m4(a), VTraits<v_float64>::vlanes());
}

inline v_float32 v_cvt_f32(const v_float64& a, const v_float64& b)
{
    return __riscv_vfncvt_f(__riscv_vset(__riscv_vlmul_ext_f64m4(a),1,b), VTraits<v_float32>::vlanes());
}

inline v_float64 v_cvt_f64(const v_int32& a)
{
    return __riscv_vget_f64m2(__riscv_vfwcvt_f(a, VTraits<v_int32>::vlanes()), 0);
}

inline v_float64 v_cvt_f64_high(const v_int32& a)
{
    return __riscv_vget_f64m2(__riscv_vfwcvt_f(a, VTraits<v_int32>::vlanes()), 1);
}

inline v_float64 v_cvt_f64(const v_float32& a)
{
    return __riscv_vget_f64m2(__riscv_vfwcvt_f(a, VTraits<v_float32>::vlanes()), 0);
}

inline v_float64 v_cvt_f64_high(const v_float32& a)
{
    return __riscv_vget_f64m2(__riscv_vfwcvt_f(a, VTraits<v_float32>::vlanes()), 1);
}

inline v_float64 v_cvt_f64(const v_int64& a)
{
    return __riscv_vfcvt_f(a, VTraits<v_int64>::vlanes());
}
#endif

//////////// Broadcast //////////////

#define OPENCV_HAL_IMPL_RVV_BROADCAST(_Tpvec, suffix) \
template<int s = 0> inline _Tpvec v_broadcast_element(_Tpvec v, int i = s) \
{ \
    return v_setall_##suffix((_Float16)v_extract_n(v, i)); \
} \
inline _Tpvec v_broadcast_highest(_Tpvec v) \
{ \
    return v_setall_##suffix((_Float16)v_extract_n(v, VTraits<_Tpvec>::vlanes()-1)); \
}

#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float16, f16)
#endif
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int32, s32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float32, f32)


////////////// Reverse //////////////
#define OPENCV_HAL_IMPL_RVV_REVERSE(_Tpvec, width) \
inline _Tpvec v_reverse(const _Tpvec& a)  \
{ \
    vuint##width##m2_t vidx = __riscv_vrsub(__riscv_vid_v_u##width##m2(VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()-1, VTraits<_Tpvec>::vlanes()); \
    return __riscv_vrgather(a, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint8, 8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int8, 8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint16, 16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int16, 16)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_REVERSE(v_float16, 16)
#endif
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint32, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int32, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_float32, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint64, 64)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int64, 64)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_REVERSE(v_float64, 64)
#endif

//////////// Value reordering ////////////

#define OPENCV_HAL_IMPL_RVV_EXPAND(_Tp, _Tpwvec, _Tpwvec_m2, _Tpvec, width, suffix, suffix2, cvt) \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
{ \
    _Tpwvec_m2 temp = cvt(a, VTraits<_Tpvec>::vlanes()); \
    b0 = __riscv_vget_##suffix##m2(temp, 0); \
    b1 = __riscv_vget_##suffix##m2(temp, 1); \
} \
inline _Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _Tpwvec_m2 temp = cvt(a, VTraits<_Tpvec>::vlanes()); \
    return __riscv_vget_##suffix##m2(temp, 0); \
} \
inline _Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _Tpwvec_m2 temp = cvt(a, VTraits<_Tpvec>::vlanes()); \
    return __riscv_vget_##suffix##m2(temp, 1); \
} \
inline _Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    return cvt(__riscv_vle##width##_v_##suffix2##m1(ptr, VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_EXPAND(uchar, v_uint16, vuint16m4_t, v_uint8, 8, u16, u8, __riscv_vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(schar, v_int16, vint16m4_t, v_int8, 8, i16, i8, __riscv_vwcvt_x)
OPENCV_HAL_IMPL_RVV_EXPAND(ushort, v_uint32, vuint32m4_t, v_uint16, 16, u32, u16, __riscv_vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(short, v_int32, vint32m4_t, v_int16, 16, i32, i16, __riscv_vwcvt_x)
OPENCV_HAL_IMPL_RVV_EXPAND(uint, v_uint64, vuint64m4_t, v_uint32, 32, u64, u32, __riscv_vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(int, v_int64, vint64m4_t, v_int32, 32, i64, i32, __riscv_vwcvt_x)

template <int N = VTraits<v_uint32>::max_nlanes>
inline v_uint32 v_load(const unsigned* ptr)
{
    unsigned buf[VTraits<v_uint32>::max_nlanes];
    v_store(buf, v_setzero_u32());
    for (int i = 0; i < N; i++) {
        buf[i] = ptr[i];
    }
    return v_load(buf);
}
template <> inline v_uint32 v_load<4>(const unsigned* ptr)
{
    unsigned buf[VTraits<v_uint32>::max_nlanes];
    v_store(buf, v_setzero_u32());
    buf[0] = ptr[0]; buf[1] = ptr[1]; buf[2] = ptr[2]; buf[3] = ptr[3];
    return v_load(buf);
}

template <int N = VTraits<v_int32>::max_nlanes>
inline v_int32 v_load(const int* ptr)
{
    int buf[VTraits<v_int32>::max_nlanes];
    v_store(buf, v_setzero_s32());
    for (int i = 0; i < N; i++) {
        buf[i] = ptr[i];
    }
    return v_load(buf);
}
template <> inline v_int32 v_load<4>(const int* ptr)
{
    int buf[VTraits<v_int32>::max_nlanes];
    v_store(buf, v_setzero_s32());
    buf[0] = ptr[0]; buf[1] = ptr[1]; buf[2] = ptr[2]; buf[3] = ptr[3];
    return v_load(buf);
}

template <int N = VTraits<v_float32>::max_nlanes>
inline v_float32 v_load(const float* ptr)
{
    float buf[VTraits<v_float32>::max_nlanes];
    v_store(buf, v_setzero_f32());
    for (int i = 0; i < N; i++) {
        buf[i] = ptr[i];
    }
    return v_load(buf);
}
template <> inline v_float32 v_load<4>(const float* ptr)
{
    float buf[VTraits<v_float32>::max_nlanes];
    v_store(buf, v_setzero_f32());
    buf[0] = ptr[0]; buf[1] = ptr[1]; buf[2] = ptr[2]; buf[3] = ptr[3];
    return v_load(buf);
}

template <int N = VTraits<v_uint32>::max_nlanes>
inline v_uint32 v_load_expand(const ushort* ptr)
{
    ushort buf[VTraits<v_uint16>::max_nlanes];
    v_store(buf, v_setzero_u16());
    for (int i = 0; i < N; i++) {
        buf[i] = ptr[i];
    }
    return v_load_expand(buf);
}
template <> inline v_uint32 v_load_expand<4>(const ushort* ptr)
{
    ushort buf[VTraits<v_uint16>::max_nlanes];
    v_store(buf, v_setzero_u16());
    buf[0] = ptr[0]; buf[1] = ptr[1]; buf[2] = ptr[2]; buf[3] = ptr[3];
    return v_load_expand(buf);
}

inline v_uint32 v_load_expand_q(const uchar* ptr)
{
    return __riscv_vwcvtu_x(__riscv_vwcvtu_x(__riscv_vle8_v_u8mf2(ptr, VTraits<v_uint32>::vlanes()), VTraits<v_uint32>::vlanes()), VTraits<v_uint32>::vlanes());
}

inline v_int32 v_load_expand_q(const schar* ptr)
{
    return __riscv_vwcvt_x(__riscv_vwcvt_x(__riscv_vle8_v_i8mf2(ptr, VTraits<v_int32>::vlanes()), VTraits<v_int32>::vlanes()), VTraits<v_int32>::vlanes());
}

template <int N = VTraits<v_uint32>::max_nlanes>
inline v_uint32 v_load_expand_q(const uchar* ptr)
{
    uchar buf[VTraits<v_uint8>::max_nlanes];
    v_store(buf, v_setzero_u8());
    for (int i = 0; i < N; i++) {
        buf[i] = ptr[i];
    }
    return v_load_expand_q(buf);
}
template <> inline v_uint32 v_load_expand_q<4>(const uchar* ptr)
{
    uchar buf[VTraits<v_uint8>::max_nlanes];
    v_store(buf, v_setzero_u8());
    buf[0] = ptr[0]; buf[1] = ptr[1]; buf[2] = ptr[2]; buf[3] = ptr[3];
    return v_load_expand_q(buf);
}

#define OPENCV_HAL_IMPL_RVV_PACK(_Tpvec, _Tp, _wTpvec, hwidth, hsuffix, suffix, rshr, shr) \
inline _Tpvec v_pack(const _wTpvec& a, const _wTpvec& b) \
{ \
    return shr(__riscv_vset(__riscv_vlmul_ext_##suffix##m4(a), 1, b), 0, 0, VTraits<_Tpvec>::vlanes()); \
} \
inline void v_pack_store(_Tp* ptr, const _wTpvec& a) \
{ \
    __riscv_vse##hwidth##_v_##hsuffix##m1(ptr, shr(a, 0, 0, VTraits<_Tpvec>::vlanes()), VTraits<_wTpvec>::vlanes()); \
} \
template<int n = 0> inline \
_Tpvec v_rshr_pack(const _wTpvec& a, const _wTpvec& b, int N = n) \
{ \
    return rshr(__riscv_vset(__riscv_vlmul_ext_##suffix##m4(a), 1, b), N, 0, VTraits<_Tpvec>::vlanes()); \
} \
template<int n = 0> inline \
void v_rshr_pack_store(_Tp* ptr, const _wTpvec& a, int N = n) \
{ \
    __riscv_vse##hwidth##_v_##hsuffix##m1(ptr, rshr(a, N, 0, VTraits<_Tpvec>::vlanes()), VTraits<_wTpvec>::vlanes()); \
}

#define OPENCV_HAL_IMPL_RVV_PACK_32(_Tpvec, _Tp, _wTpvec, hwidth, hsuffix, suffix, rshr, shr) \
inline _Tpvec v_pack(const _wTpvec& a, const _wTpvec& b) \
{ \
    return shr(__riscv_vset(__riscv_vlmul_ext_##suffix##m4(a), 1, b), 0, VTraits<_Tpvec>::vlanes()); \
} \
inline void v_pack_store(_Tp* ptr, const _wTpvec& a) \
{ \
    __riscv_vse##hwidth##_v_##hsuffix##m1(ptr, shr(a, 0, VTraits<_Tpvec>::vlanes()), VTraits<_wTpvec>::vlanes()); \
} \
template<int n = 0> inline \
_Tpvec v_rshr_pack(const _wTpvec& a, const _wTpvec& b, int N = n) \
{ \
    return rshr(__riscv_vset(__riscv_vlmul_ext_##suffix##m4(a), 1, b), N, 0, VTraits<_Tpvec>::vlanes()); \
} \
template<int n = 0> inline \
void v_rshr_pack_store(_Tp* ptr, const _wTpvec& a, int N = n) \
{ \
    __riscv_vse##hwidth##_v_##hsuffix##m1(ptr, rshr(a, N, 0, VTraits<_Tpvec>::vlanes()), VTraits<_wTpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_PACK(v_uint8, uchar, v_uint16, 8, u8, u16, __riscv_vnclipu, __riscv_vnclipu)
OPENCV_HAL_IMPL_RVV_PACK(v_int8, schar, v_int16, 8,  i8, i16, __riscv_vnclip, __riscv_vnclip)
OPENCV_HAL_IMPL_RVV_PACK(v_uint16, ushort, v_uint32, 16, u16, u32, __riscv_vnclipu, __riscv_vnclipu)
OPENCV_HAL_IMPL_RVV_PACK(v_int16, short, v_int32, 16, i16, i32, __riscv_vnclip, __riscv_vnclip)
OPENCV_HAL_IMPL_RVV_PACK_32(v_uint32, unsigned, v_uint64, 32, u32, u64, __riscv_vnclipu, __riscv_vnsrl)
OPENCV_HAL_IMPL_RVV_PACK_32(v_int32, int, v_int64, 32, i32, i64, __riscv_vnclip, __riscv_vnsra)

template <int N = VTraits<v_uint16>::max_nlanes>
inline v_uint16 v_pack(const v_uint32& a, const v_uint32& b)
{
    ushort bufa[N];
    ushort bufb[N];
    v_pack_store(bufa, a);
    v_pack_store(bufb, b);
    ushort buf[N];
    for (int i = 0; i < N; i++) {
        buf[i] = bufa[i];
        buf[i+N/2] = bufb[i];
    }
    return v_load(buf);
}

template <> inline v_uint16 v_pack<4>(const v_uint32& a, const v_uint32& b)
{
    constexpr int N = VTraits<v_uint16>::max_nlanes;
    ushort bufa[N];
    ushort bufb[N];
    v_pack_store(bufa, a);
    v_pack_store(bufb, b);

    ushort buf[N];
    buf[0] = bufa[0]; buf[1] = bufa[1]; buf[2] = bufa[2]; buf[3] = bufa[3];
    buf[4] = bufb[0]; buf[5] = bufb[1]; buf[6] = bufb[2]; buf[7] = bufb[3];
    return v_load(buf);
}

#define OPENCV_HAL_IMPL_RVV_PACK_U(_Tpvec, _Tp, _wTpvec, _wTp, hwidth, width, hsuffix, suffix, cast, hvl, vl) \
inline _Tpvec v_pack_u(const _wTpvec& a, const _wTpvec& b) \
{ \
    return __riscv_vnclipu(cast(__riscv_vmax(__riscv_vset(__riscv_vlmul_ext_##suffix##m4(a), 1, b), 0, vl)), 0, 0, vl); \
} \
inline void v_pack_u_store(_Tp* ptr, const _wTpvec& a) \
{ \
    __riscv_vse##hwidth##_v_##hsuffix##m1(ptr, __riscv_vnclipu(__riscv_vreinterpret_u##width##m2(__riscv_vmax(a, 0, vl)), 0, 0, vl), hvl); \
} \
template<int N = 0> inline \
_Tpvec v_rshr_pack_u(const _wTpvec& a, const _wTpvec& b, int n = N) \
{ \
    return __riscv_vnclipu(cast(__riscv_vmax(__riscv_vset(__riscv_vlmul_ext_##suffix##m4(a), 1, b), 0, vl)), n, 0, vl); \
} \
template<int N = 0> inline \
void v_rshr_pack_u_store(_Tp* ptr, const _wTpvec& a, int n = N) \
{ \
    __riscv_vse##hwidth##_v_##hsuffix##m1(ptr, __riscv_vnclipu(__riscv_vreinterpret_u##width##m2(__riscv_vmax(a, 0, vl)), n, 0, vl), hvl); \
}

OPENCV_HAL_IMPL_RVV_PACK_U(v_uint8, uchar, v_int16, short, 8, 16, u8, i16, __riscv_vreinterpret_v_i16m4_u16m4, VTraits<v_int16>::vlanes(), VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_PACK_U(v_uint16, ushort, v_int32, int, 16, 32, u16, i32,  __riscv_vreinterpret_v_i32m4_u32m4, VTraits<v_int32>::vlanes(), VTraits<v_uint16>::vlanes())

template <int N = VTraits<v_uint16>::max_nlanes>
inline v_uint16 v_pack_u(const v_int32& a, const v_int32& b)
{
    ushort bufa[N];
    ushort bufb[N];
    v_pack_u_store(bufa, a);
    v_pack_u_store(bufb, b);
    ushort buf[N];
    for (int i = 0; i < N; i++) {
        buf[i] = bufa[i];
        buf[i+N/2] = bufb[i];
    }
    return v_load(buf);
}

template <> inline v_uint16 v_pack_u<4>(const v_int32& a, const v_int32& b)
{
    constexpr int N = VTraits<v_uint16>::max_nlanes;
    ushort bufa[N];
    ushort bufb[N];
    v_pack_u_store(bufa, a);
    v_pack_u_store(bufb, b);

    ushort buf[N];
    buf[0] = bufa[0]; buf[1] = bufa[1]; buf[2] = bufa[2]; buf[3] = bufa[3];
    buf[4] = bufb[0]; buf[5] = bufb[1]; buf[6] = bufb[2]; buf[7] = bufb[3];
    return v_load(buf);
}

template <int N = VTraits<v_int16>::max_nlanes>
inline void v_pack_store(uchar* ptr, const v_uint16& a)
{
    uchar buf[VTraits<v_uint8>::max_nlanes];
    v_pack_store(buf, a);
    for (int i = 0; i < N; i++) {
        ptr[i] = buf[i];
    }
}
template <> inline void v_pack_store<8>(uchar* ptr, const v_uint16& a)
{
    uchar buf[VTraits<v_uint8>::max_nlanes];
    v_pack_store(buf, a);
    ptr[0] = buf[0]; ptr[1] = buf[1]; ptr[2] = buf[2]; ptr[3] = buf[3];
    ptr[4] = buf[4]; ptr[5] = buf[5]; ptr[6] = buf[6]; ptr[7] = buf[7];
}


/* void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1)
  a0 = {A1 A2 A3 A4}
  a1 = {B1 B2 B3 B4}
---------------
  {A1 B1 A2 B2} and {A3 B3 A4 B4}
*/

#define OPENCV_HAL_IMPL_RVV_ZIP(_Tpvec, _wTpvec, suffix, width, width2, convert2um2, convert2um1) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) { \
    _wTpvec temp = __riscv_vreinterpret_##suffix##m4(convert2um2( \
        __riscv_vor(__riscv_vzext_vf2(convert2um1(a0), VTraits<_Tpvec>::vlanes()*2), \
            __riscv_vreinterpret_u##width2##m4(__riscv_vslide1up(__riscv_vreinterpret_u##width##m4(__riscv_vzext_vf2(convert2um1(a1), VTraits<_Tpvec>::vlanes()*2)), 0, VTraits<_Tpvec>::vlanes()*2)), \
            VTraits<_Tpvec>::vlanes()))); \
    b0 = __riscv_vget_##suffix##m2(temp, 0); \
    b1 = __riscv_vget_##suffix##m2(temp, 1); \
}
OPENCV_HAL_IMPL_RVV_ZIP(v_uint8, vuint8m4_t, u8, 8, 16, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP(v_int8, vint8m4_t, i8, 8, 16, __riscv_vreinterpret_u8m4, __riscv_vreinterpret_u8m2)
OPENCV_HAL_IMPL_RVV_ZIP(v_uint16, vuint16m4_t, u16, 16, 32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP(v_int16, vint16m4_t, i16, 16, 32, __riscv_vreinterpret_u16m4, __riscv_vreinterpret_u16m2)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_ZIP(v_float16, vfloat16m4_t, f16, 16, 32, __riscv_vreinterpret_u16m4, __riscv_vreinterpret_u16m2)
#endif
OPENCV_HAL_IMPL_RVV_ZIP(v_uint32, vuint32m4_t, u32, 32, 64, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP(v_int32, vint32m4_t, i32, 32, 64, __riscv_vreinterpret_u32m4, __riscv_vreinterpret_u32m2)
OPENCV_HAL_IMPL_RVV_ZIP(v_float32, vfloat32m4_t, f32, 32, 64, __riscv_vreinterpret_u32m4, __riscv_vreinterpret_u32m2)

#if CV_SIMD_SCALABLE_64F
inline void v_zip(const v_float64& a0, const v_float64& a1, v_float64& b0, v_float64& b1) { \
    vuint16mf2_t idx0 = __riscv_vid_v_u16mf2(VTraits<v_float64>::vlanes());
    vuint16mf2_t idx1 = __riscv_vadd(idx0, VTraits<v_float64>::vlanes(), VTraits<v_float64>::vlanes());
    vuint16m1_t idx = __riscv_vreinterpret_u16m1(( \
        __riscv_vor(__riscv_vzext_vf2(idx0, VTraits<v_float64>::vlanes()), \
            __riscv_vreinterpret_u32m1(__riscv_vslide1up(__riscv_vreinterpret_u16m1(__riscv_vzext_vf2(idx1, VTraits<v_float64>::vlanes())), 0, VTraits<v_uint32>::vlanes())), \
            VTraits<v_uint32>::vlanes())));
#if 0
    vfloat64m4_t temp = __riscv_vcreate_v_f64m2_f64m4(a0, a1);
#else // TODO: clean up when RVV Intrinsic is frozen.
    vfloat64m4_t temp = __riscv_vlmul_ext_f64m4(a0);
    temp = __riscv_vset(temp, 1, a1);
#endif
    temp = __riscv_vrgatherei16(temp, idx, VTraits<v_float64>::vlanes()*2);
    b0 = __riscv_vget_f64m2(temp, 0); \
    b1 = __riscv_vget_f64m2(temp, 1); \
}
#endif

#define OPENCV_HAL_IMPL_RVV_UNPACKS(_Tpvec, width) \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vslideup(a, b, VTraits<_Tpvec>::vlanes()/2, VTraits<_Tpvec>::vlanes());\
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    return __riscv_vslideup( \
            __riscv_vslidedown(a, VTraits<_Tpvec>::vlanes()/2, VTraits<_Tpvec>::vlanes()), \
            __riscv_vslidedown(b, VTraits<_Tpvec>::vlanes()/2, VTraits<_Tpvec>::vlanes()), \
            VTraits<_Tpvec>::vlanes()/2, \
            VTraits<_Tpvec>::vlanes()); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    c = v_combine_low(a, b); \
    d = v_combine_high(a, b); \
}

OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint8, 8)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int8, 8)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint16, 16)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int16, 16)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint32, 32)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int32, 32)
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float16, 16)
#endif
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float32, 32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float64, 64)
#endif

#define OPENCV_HAL_IMPL_RVV_INTERLEAVED(_Tpvec, _Tp, _TpCast, suffix, width, hwidth, vl) \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b) \
{ \
    a = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)ptr  , sizeof(_Tp)*2, VTraits<v_##_Tpvec>::vlanes()); \
    b = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)(ptr+1), sizeof(_Tp)*2, VTraits<v_##_Tpvec>::vlanes()); \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, v_##_Tpvec& c) \
{ \
    a = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)ptr  , sizeof(_Tp)*3, VTraits<v_##_Tpvec>::vlanes()); \
    b = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)(ptr+1), sizeof(_Tp)*3, VTraits<v_##_Tpvec>::vlanes()); \
    c = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)(ptr+2), sizeof(_Tp)*3, VTraits<v_##_Tpvec>::vlanes()); \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, \
                                v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    \
    a = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)ptr  , sizeof(_Tp)*4, VTraits<v_##_Tpvec>::vlanes()); \
    b = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)(ptr+1), sizeof(_Tp)*4, VTraits<v_##_Tpvec>::vlanes()); \
    c = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)(ptr+2), sizeof(_Tp)*4, VTraits<v_##_Tpvec>::vlanes()); \
    d = __riscv_vlse##width##_v_##suffix##m2((_TpCast *)(ptr+3), sizeof(_Tp)*4, VTraits<v_##_Tpvec>::vlanes()); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    __riscv_vsse##width((_TpCast *)ptr, sizeof(_Tp)*2, a, VTraits<v_##_Tpvec>::vlanes()); \
    __riscv_vsse##width((_TpCast *)(ptr+1), sizeof(_Tp)*2, b, VTraits<v_##_Tpvec>::vlanes()); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    __riscv_vsse##width((_TpCast *)ptr, sizeof(_Tp)*3, a, VTraits<v_##_Tpvec>::vlanes()); \
    __riscv_vsse##width((_TpCast *)(ptr+1), sizeof(_Tp)*3, b, VTraits<v_##_Tpvec>::vlanes()); \
    __riscv_vsse##width((_TpCast *)(ptr+2), sizeof(_Tp)*3, c, VTraits<v_##_Tpvec>::vlanes()); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, const v_##_Tpvec& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    __riscv_vsse##width((_TpCast *)ptr, sizeof(_Tp)*4, a, VTraits<v_##_Tpvec>::vlanes()); \
    __riscv_vsse##width((_TpCast *)(ptr+1), sizeof(_Tp)*4, b, VTraits<v_##_Tpvec>::vlanes()); \
    __riscv_vsse##width((_TpCast *)(ptr+2), sizeof(_Tp)*4, c, VTraits<v_##_Tpvec>::vlanes()); \
    __riscv_vsse##width((_TpCast *)(ptr+3), sizeof(_Tp)*4, d, VTraits<v_##_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint8, uchar, uchar, u8, 8, 4, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int8, schar, schar, i8, 8, 4, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint16, ushort, ushort, u16, 16, 8, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int16, short, short, i16, 16, 8, VTraits<v_int16>::vlanes())
#if CV_SIMD_SCALABLE_FP16
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float16, hfloat, _Float16, f16, 16, 8, VTraits<v_float16>::vlanes())
#endif
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint32, unsigned, unsigned, u32, 32, 16, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int32, int, int, i32, 32, 16, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float32, float, float, f32, 32, 16, VTraits<v_float32>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint64, uint64, uint64, u64, 64, 32, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int64, int64, int64, i64, 64, 32, VTraits<v_int64>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float64, double, double, f64, 64, 32, VTraits<v_float64>::vlanes())
#endif

static uint64_t idx_interleave_pairs[] = { \
    0x0705060403010200, 0x0f0d0e0c0b090a08, 0x1715161413111210, 0x1f1d1e1c1b191a18, \
    0x2725262423212220, 0x2f2d2e2c2b292a28, 0x3735363433313230, 0x3f3d3e3c3b393a38, \
    0x4745464443414240, 0x4f4d4e4c4b494a48, 0x5755565453515250, 0x5f5d5e5c5b595a58, \
    0x6765666463616260, 0x6f6d6e6c6b696a68, 0x7775767473717270, 0x7f7d7e7c7b797a78};

static uint64_t idx_interleave_quads[] = { \
    0x0703060205010400, 0x0f0b0e0a0d090c08, 0x1713161215111410, 0x1f1b1e1a1d191c18, \
    0x2723262225212420, 0x2f2b2e2a2d292c28, 0x3733363235313430, 0x3f3b3e3a3d393c38, \
    0x4743464245414440, 0x4f4b4e4a4d494c48, 0x5753565255515450, 0x5f5b5e5a5d595c58, \
    0x6763666265616460, 0x6f6b6e6a6d696c68, 0x7773767275717470, 0x7f7b7e7a7d797c78};

#define OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(_Tpvec, func) \
inline _Tpvec v_interleave_##func(const _Tpvec& vec) { \
    CV_CheckLE(VTraits<_Tpvec>::vlanes(), VTraits<_Tpvec>::max_nlanes, "RVV implementation only supports VLEN in the range [128, 1024]"); \
    vuint8m2_t vidx = __riscv_vundefined_u8m2();\
    vidx = __riscv_vreinterpret_u8m2(__riscv_vle64_v_u64m2(idx_interleave_##func, 16)); \
    return __riscv_vrgather(vec, vidx, VTraits<v_uint8>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_uint8, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_int8, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_uint8, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_int8, quads)

#define OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(_Tpvec, width, vzext_vfx, func) \
inline _Tpvec v_interleave_##func(const _Tpvec& vec) { \
    CV_CheckLE(VTraits<_Tpvec>::vlanes(), VTraits<_Tpvec>::max_nlanes, "RVV implementation only supports VLEN in the range [128, 1024]"); \
    vuint##width##m2_t vidx = __riscv_vundefined_u##width##m2();\
    vidx = __riscv_vget_u##width##m2(vzext_vfx(__riscv_vreinterpret_u8m2(__riscv_vle64_v_u64m2(idx_interleave_##func, 16)), VTraits<v_uint8>::vlanes()), 0); \
    return __riscv_vrgather(vec, vidx, VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint16, 16, __riscv_vzext_vf2, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int16, 16, __riscv_vzext_vf2, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint32, 32, __riscv_vzext_vf4, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int32, 32, __riscv_vzext_vf4, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_float32, 32, __riscv_vzext_vf4, pairs)

OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint16, 16, __riscv_vzext_vf2, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int16, 16, __riscv_vzext_vf2, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint32, 32, __riscv_vzext_vf4, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int32, 32, __riscv_vzext_vf4, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_float32, 32, __riscv_vzext_vf4, quads)

//////////// PopCount //////////
static const unsigned char popCountTable[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};
#define OPENCV_HAL_IMPL_RVV_HADD(_Tpvec, _Tpvec2, _Tm2, width, width2, suffix, add) \
static inline _Tpvec2 v_hadd(_Tpvec a) { \
    vuint##width2##m2_t oneX2 = __riscv_vmv_v_x_u##width2##m2(1, VTraits<v_uint##width2>::vlanes()); \
    vuint##width##m2_t one = __riscv_vreinterpret_u##width##m2(oneX2); \
    _Tm2 res = add(a, __riscv_vslide1down(a, 0, VTraits<v_uint##width>::vlanes()), VTraits<v_uint##width>::vlanes()); \
    return __riscv_vget_##suffix##m2(__riscv_vcompress(res, __riscv_vmseq(one, 1, VTraits<v_uint##width>::vlanes()), VTraits<v_uint##width>::vlanes()), 0); \
}
OPENCV_HAL_IMPL_RVV_HADD(v_uint8, v_uint16, vuint16m4_t, 8, 16, u16, __riscv_vwaddu_vv)
OPENCV_HAL_IMPL_RVV_HADD(v_uint16, v_uint32, vuint32m4_t, 16, 32, u32, __riscv_vwaddu_vv)
OPENCV_HAL_IMPL_RVV_HADD(v_uint32, v_uint64, vuint64m4_t, 32, 64, u64, __riscv_vwaddu_vv)
OPENCV_HAL_IMPL_RVV_HADD(v_int8, v_int16, vint16m4_t, 8, 16, i16, __riscv_vwadd_vv)
OPENCV_HAL_IMPL_RVV_HADD(v_int16, v_int32, vint32m4_t, 16, 32, i32, __riscv_vwadd_vv)
OPENCV_HAL_IMPL_RVV_HADD(v_int32, v_int64, vint64m4_t, 32, 64, i64, __riscv_vwadd_vv)

OPENCV_HAL_IMPL_RVV_HADD(vint32m4_t, v_int32, vint32m4_t, 16, 32, i32, __riscv_vadd)
OPENCV_HAL_IMPL_RVV_HADD(vint64m4_t, v_int64, vint64m4_t, 32, 64, i64, __riscv_vadd)

inline v_uint8 v_popcount(const v_uint8& a)
{
    return __riscv_vloxei8(popCountTable, a, VTraits<v_uint8>::vlanes());
}
inline v_uint16 v_popcount(const v_uint16& a)
{
    return v_hadd(v_popcount(__riscv_vreinterpret_u8m2(a)));
}
inline v_uint32 v_popcount(const v_uint32& a)
{
    return v_hadd(v_hadd(v_popcount(__riscv_vreinterpret_u8m2(a))));
}
inline v_uint64 v_popcount(const v_uint64& a)
{
    return v_hadd(v_hadd(v_hadd(v_popcount(__riscv_vreinterpret_u8m2(a)))));
}

inline v_uint8 v_popcount(const v_int8& a)
{
    return v_popcount(v_abs(a));\
}
inline v_uint16 v_popcount(const v_int16& a)
{
    return v_popcount(v_abs(a));\
}
inline v_uint32 v_popcount(const v_int32& a)
{
    return v_popcount(v_abs(a));\
}
inline v_uint64 v_popcount(const v_int64& a)
{
    // max(0 - a) is used, since v_abs does not support 64-bit integers.
    return v_popcount(v_reinterpret_as_u64(__riscv_vmax(a, v_sub(v_setzero_s64(), a), VTraits<v_int64>::vlanes())));
}


//////////// SignMask ////////////
#define OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(_Tpvec) \
inline int v_signmask(const _Tpvec& a) \
{ \
    uint8_t ans[4] = {0}; \
    __riscv_vsm(ans, __riscv_vmslt(a, 0, VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()); \
    return *(reinterpret_cast<int*>(ans)) & (((__int128_t)1 << VTraits<_Tpvec>::vlanes()) - 1); \
} \
inline int v_scan_forward(const _Tpvec& a) \
{ \
    return (int)__riscv_vfirst(__riscv_vmslt(a, 0, VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int8)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int16)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int32)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int64)

inline int64 v_signmask(const v_uint8& a)
{ return v_signmask(v_reinterpret_as_s8(a)); }
inline int64 v_signmask(const v_uint16& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }
#if CV_SIMD_SCALABLE_FP16
inline int v_signmask(const v_float16& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }
#endif
inline int v_signmask(const v_uint32& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_float32& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_uint64& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
#if CV_SIMD_SCALABLE_64F
inline int v_signmask(const v_float64& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
#endif

//////////// Scan forward ////////////
inline int v_scan_forward(const v_uint8& a)
{ return v_scan_forward(v_reinterpret_as_s8(a)); }
inline int v_scan_forward(const v_uint16& a)
{ return v_scan_forward(v_reinterpret_as_s16(a)); }
inline int v_scan_forward(const v_uint32& a)
{ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_float32& a)
{ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_uint64& a)
{ return v_scan_forward(v_reinterpret_as_s64(a)); }
#if CV_SIMD_SCALABLE_64F
inline int v_scan_forward(const v_float64& a)
{ return v_scan_forward(v_reinterpret_as_s64(a)); }
#endif

//////////// Pack triplets ////////////
// {A0, A1, A2, A3, B0, B1, B2, B3, C0 ...} --> {A0, A1, A2, B0, B1, B2, C0 ...}
// mask: {0,0,0,1, ...} -> {T,T,T,F, ...}
#define OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(_Tpvec, v_trunc) \
inline _Tpvec v_pack_triplets(const _Tpvec& vec) { \
    size_t vl = VTraits<v_uint8>::vlanes(); \
    vuint32m2_t one = __riscv_vmv_v_x_u32m2(1, VTraits<v_uint32>::vlanes()); \
    vuint8m2_t zero = __riscv_vmv_v_x_u8m2(0, vl); \
    vuint8m2_t mask = __riscv_vreinterpret_u8m2(one); \
    return __riscv_vcompress(vec, __riscv_vmseq(v_trunc(__riscv_vslideup(zero, mask, 3, vl)), 0, vl), VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint16, __riscv_vlmul_trunc_u8m1)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int16, __riscv_vlmul_trunc_u8m1)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint32, __riscv_vlmul_trunc_u8mf2)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int32, __riscv_vlmul_trunc_u8mf2)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_float32, __riscv_vlmul_trunc_u8mf2)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint64, __riscv_vlmul_trunc_u8mf4)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int64, __riscv_vlmul_trunc_u8mf4)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_float64, __riscv_vlmul_trunc_u8mf4)
#endif


////// FP16 support ///////

#if defined(__riscv_zfh) && __riscv_zfh
inline v_float32 v_load_expand(const hfloat* ptr)
{
    return __riscv_vfwcvt_f(__riscv_vle16_v_f16m1((_Float16*)ptr, VTraits<v_float32>::vlanes()) ,VTraits<v_float32>::vlanes());;
}

inline void v_pack_store(hfloat* ptr, const v_float32& v)
{
    __riscv_vse16_v_f16m1((_Float16*)ptr, __riscv_vfncvt_f_f_w_f16m1(v, VTraits<v_float32>::vlanes()), VTraits<v_float32>::vlanes());
}
#else
inline v_float32 v_load_expand(const hfloat* ptr)
{
    float buf[32];
    for( int i = 0; i < VTraits<v_float32>::vlanes(); i++ ) buf[i] = (float)ptr[i];
    return v_load(buf);
}

inline void v_pack_store(hfloat* ptr, const v_float32& v)
{
    float buf[32];
    v_store(buf, v);
    for( int i = 0; i < VTraits<v_float32>::vlanes(); i++ ) ptr[i] = hfloat(buf[i]);
}
#endif
////////////// Rounding //////////////
#if CV_SIMD_SCALABLE_FP16
inline v_int16 v_round(const v_float16& a)
{
    return __riscv_vfcvt_x(a, VTraits<v_float16>::vlanes());
}

inline v_int16 v_floor(const v_float16& a)
{
    return __riscv_vfcvt_x_f_v_i16m2_rm(a, 1 /*RNE, round-to-nearest-even*/, VTraits<v_float16>::vlanes());
}

inline v_int16 v_ceil(const v_float16& a)
{
    return __riscv_vfcvt_x_f_v_i16m2_rm(a, 3 /*ROD, round-to-odd*/, VTraits<v_float16>::vlanes());
}

inline v_int16 v_trunc(const v_float16& a)
{
    return __riscv_vfcvt_rtz_x(a, VTraits<v_float16>::vlanes());
}
#endif
inline v_int32 v_round(const v_float32& a)
{
    // return vfcvt_x(vfadd(a, 1e-6, VTraits<v_float32>::vlanes()), VTraits<v_float32>::vlanes());
    return __riscv_vfcvt_x(a, VTraits<v_float32>::vlanes());
}

inline v_int32 v_floor(const v_float32& a)
{
    return __riscv_vfcvt_x(__riscv_vfsub(a, 0.5f - 1e-5, VTraits<v_float32>::vlanes()), VTraits<v_float32>::vlanes());
    // return vfcvt_x(a, VTraits<v_float32>::vlanes());
}

inline v_int32 v_ceil(const v_float32& a)
{
    return __riscv_vfcvt_x(__riscv_vfadd(a, 0.5f - 1e-5, VTraits<v_float32>::vlanes()), VTraits<v_float32>::vlanes());
}

inline v_int32 v_trunc(const v_float32& a)
{
    return __riscv_vfcvt_rtz_x(a, VTraits<v_float32>::vlanes());
}
#if CV_SIMD_SCALABLE_64F
inline v_int32 v_round(const v_float64& a)
{
    return __riscv_vfncvt_x(__riscv_vlmul_ext_f64m4(a), VTraits<v_float32>::vlanes());
}

inline v_int32 v_round(const v_float64& a, const v_float64& b)
{
    // return vfncvt_x(vset(vlmul_ext_f64m2(vfadd(a, 1e-6, VTraits<v_float64>::vlanes())), 1, b), VTraits<v_float32>::vlanes());
    // Fix https://github.com/opencv/opencv/issues/24746
    return __riscv_vfncvt_x(__riscv_vset(__riscv_vlmul_ext_f64m4(a), 1, b), VTraits<v_float32>::vlanes());
}

inline v_int32 v_floor(const v_float64& a)
{
    return __riscv_vfncvt_x(__riscv_vlmul_ext_f64m4(__riscv_vfsub(a, 0.5f - 1e-6, VTraits<v_float64>::vlanes())), VTraits<v_float32>::vlanes());
}

inline v_int32 v_ceil(const v_float64& a)
{
    return __riscv_vfncvt_x(__riscv_vlmul_ext_f64m4(__riscv_vfadd(a, 0.5f - 1e-6, VTraits<v_float64>::vlanes())), VTraits<v_float32>::vlanes());
}

inline v_int32 v_trunc(const v_float64& a)
{
    return __riscv_vfncvt_rtz_x(__riscv_vlmul_ext_f64m4(a), VTraits<v_float32>::vlanes());
}
#endif

//////// Dot Product ////////

// 16 >> 32
inline v_int32 v_dotprod(const v_int16& a, const v_int16& b)
{
    vint32m4_t temp1 = __riscv_vwmul(a, b, VTraits<v_int16>::vlanes());
    return v_hadd(temp1);
}

inline v_int32 v_dotprod(const v_int16& a, const v_int16& b, const v_int32& c)
{
    vint32m4_t temp1 = __riscv_vwmul(a, b, VTraits<v_int16>::vlanes());
    return __riscv_vadd(v_hadd(temp1), c, VTraits<v_int32>::vlanes());
}

// 32 >> 64
inline v_int64 v_dotprod(const v_int32& a, const v_int32& b)
{
    vuint64m2_t one64 = __riscv_vmv_v_x_u64m2(1, VTraits<v_uint64>::vlanes()); \
    vuint32m2_t one32 = __riscv_vreinterpret_u32m2(one64); \
    vbool16_t mask = __riscv_vmseq(one32, 1, VTraits<v_uint32>::vlanes()); \
    vint64m4_t temp1 = __riscv_vwmul(a, b, VTraits<v_int32>::vlanes()); \
    vint64m4_t temp2 = __riscv_vslide1down(temp1, 0, VTraits<v_int32>::vlanes());
    vint64m4_t res = __riscv_vadd(temp1, temp2, VTraits<v_int32>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_int32>::vlanes()); \
    return __riscv_vlmul_trunc_i64m2(res); \
}
inline v_int64 v_dotprod(const v_int32& a, const v_int32& b, const v_int64& c)
{
    vuint64m2_t one64 = __riscv_vmv_v_x_u64m2(1, VTraits<v_uint64>::vlanes()); \
    vuint32m2_t one32 = __riscv_vreinterpret_u32m2(one64); \
    vbool16_t mask = __riscv_vmseq(one32, 1, VTraits<v_uint32>::vlanes()); \
    vint64m4_t temp1 = __riscv_vwmul(a, b, VTraits<v_int32>::vlanes()); \
    vint64m4_t temp2 = __riscv_vslide1down(temp1, 0, VTraits<v_int32>::vlanes());
    vint64m4_t res = __riscv_vadd(temp1, temp2, VTraits<v_int32>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_int32>::vlanes()); \
    return __riscv_vadd(__riscv_vlmul_trunc_i64m2(res), c, VTraits<v_int64>::vlanes()); \
}

// 8 >> 32
inline v_uint32 v_dotprod_expand(const v_uint8& a, const v_uint8& b)
{
    vuint32m2_t one32 = __riscv_vmv_v_x_u32m2(1, VTraits<v_uint32>::vlanes()); \
    vuint8m2_t one8 = __riscv_vreinterpret_u8m2(one32); \
    vbool4_t mask = __riscv_vmseq(one8, 1, VTraits<v_uint8>::vlanes()); \
    vuint16m4_t t0 = __riscv_vwmulu(a, b, VTraits<v_uint8>::vlanes()); \
    vuint16m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_uint8>::vlanes());
    vuint16m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_uint8>::vlanes());
    vuint16m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_uint8>::vlanes());
    vuint32m8_t res = __riscv_vadd(__riscv_vwaddu_vv(t2, t3, VTraits<v_uint8>::vlanes()), __riscv_vwaddu_vv(t0, t1, VTraits<v_uint8>::vlanes()), VTraits<v_uint8>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_uint8>::vlanes()); \
    return __riscv_vlmul_trunc_u32m2(res);
}

inline v_uint32 v_dotprod_expand(const v_uint8& a, const v_uint8& b,
                                  const v_uint32& c)
{
    vuint32m2_t one32 = __riscv_vmv_v_x_u32m2(1, VTraits<v_uint32>::vlanes()); \
    vuint8m2_t one8 = __riscv_vreinterpret_u8m2(one32); \
    vbool4_t mask = __riscv_vmseq(one8, 1, VTraits<v_uint8>::vlanes()); \
    vuint16m4_t t0 = __riscv_vwmulu(a, b, VTraits<v_uint8>::vlanes()); \
    vuint16m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_uint8>::vlanes());
    vuint16m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_uint8>::vlanes());
    vuint16m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_uint8>::vlanes());
    vuint32m8_t res = __riscv_vadd(__riscv_vwaddu_vv(t2, t3, VTraits<v_uint8>::vlanes()), __riscv_vwaddu_vv(t0, t1, VTraits<v_uint8>::vlanes()), VTraits<v_uint8>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_uint8>::vlanes()); \
    return __riscv_vadd(__riscv_vlmul_trunc_u32m2(res), c, VTraits<v_uint8>::vlanes());
}

inline v_int32 v_dotprod_expand(const v_int8& a, const v_int8& b)
{
    vuint32m2_t one32 = __riscv_vmv_v_x_u32m2(1, VTraits<v_uint32>::vlanes()); \
    vuint8m2_t one8 = __riscv_vreinterpret_u8m2(one32); \
    vbool4_t mask = __riscv_vmseq(one8, 1, VTraits<v_uint8>::vlanes()); \
    vint16m4_t t0 = __riscv_vwmul(a, b, VTraits<v_int8>::vlanes()); \
    vint16m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_int8>::vlanes());
    vint16m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_int8>::vlanes());
    vint16m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_int8>::vlanes());
    vint32m8_t res = __riscv_vadd(__riscv_vwadd_vv(t2, t3, VTraits<v_int8>::vlanes()), __riscv_vwadd_vv(t0, t1, VTraits<v_int8>::vlanes()), VTraits<v_int8>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_int8>::vlanes()); \
    return __riscv_vlmul_trunc_i32m2(res);
}

inline v_int32 v_dotprod_expand(const v_int8& a, const v_int8& b,
                                  const v_int32& c)
{
    vuint32m2_t one32 = __riscv_vmv_v_x_u32m2(1, VTraits<v_uint32>::vlanes()); \
    vuint8m2_t one8 = __riscv_vreinterpret_u8m2(one32); \
    vbool4_t mask = __riscv_vmseq(one8, 1, VTraits<v_uint8>::vlanes()); \
    vint16m4_t t0 = __riscv_vwmul(a, b, VTraits<v_int8>::vlanes()); \
    vint16m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_int8>::vlanes());
    vint16m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_int8>::vlanes());
    vint16m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_int8>::vlanes());
    vint32m8_t res = __riscv_vadd(__riscv_vwadd_vv(t2, t3, VTraits<v_int8>::vlanes()), __riscv_vwadd_vv(t0, t1, VTraits<v_int8>::vlanes()), VTraits<v_int8>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_int8>::vlanes()); \
    return __riscv_vadd(__riscv_vlmul_trunc_i32m2(res), c, VTraits<v_int8>::vlanes());
}


// // 16 >> 64
inline v_uint64 v_dotprod_expand(const v_uint16& a, const v_uint16& b)
{
    vuint64m2_t one64 = __riscv_vmv_v_x_u64m2(1, VTraits<v_uint64>::vlanes()); \
    vuint16m2_t one16 = __riscv_vreinterpret_u16m2(one64); \
    vbool8_t mask = __riscv_vmseq(one16, 1, VTraits<v_uint16>::vlanes()); \
    vuint32m4_t t0 = __riscv_vwmulu(a, b, VTraits<v_uint16>::vlanes()); \
    vuint32m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_uint16>::vlanes());
    vuint32m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_uint16>::vlanes());
    vuint32m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_uint16>::vlanes());
    vuint64m8_t res = __riscv_vadd(__riscv_vwaddu_vv(t2, t3, VTraits<v_uint16>::vlanes()), __riscv_vwaddu_vv(t0, t1, VTraits<v_uint16>::vlanes()), VTraits<v_uint16>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_uint16>::vlanes()); \
    return __riscv_vlmul_trunc_u64m2(res);
}
inline v_uint64 v_dotprod_expand(const v_uint16& a, const v_uint16& b, const v_uint64& c)
{
    vuint64m2_t one64 = __riscv_vmv_v_x_u64m2(1, VTraits<v_uint64>::vlanes()); \
    vuint16m2_t one16 = __riscv_vreinterpret_u16m2(one64); \
    vbool8_t mask = __riscv_vmseq(one16, 1, VTraits<v_uint16>::vlanes()); \
    vuint32m4_t t0 = __riscv_vwmulu(a, b, VTraits<v_uint16>::vlanes()); \
    vuint32m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_uint16>::vlanes());
    vuint32m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_uint16>::vlanes());
    vuint32m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_uint16>::vlanes());
    vuint64m8_t res = __riscv_vadd(__riscv_vwaddu_vv(t2, t3, VTraits<v_uint16>::vlanes()), __riscv_vwaddu_vv(t0, t1, VTraits<v_uint16>::vlanes()), VTraits<v_uint16>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_uint16>::vlanes()); \
    return __riscv_vadd(__riscv_vlmul_trunc_u64m2(res), c, VTraits<v_uint16>::vlanes());
}

inline v_int64 v_dotprod_expand(const v_int16& a, const v_int16& b)
{
    vuint64m2_t one64 = __riscv_vmv_v_x_u64m2(1, VTraits<v_uint64>::vlanes()); \
    vuint16m2_t one16 = __riscv_vreinterpret_u16m2(one64); \
    vbool8_t mask = __riscv_vmseq(one16, 1, VTraits<v_uint16>::vlanes()); \
    vint32m4_t t0 = __riscv_vwmul(a, b, VTraits<v_int16>::vlanes()); \
    vint32m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_int16>::vlanes());
    vint32m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_int16>::vlanes());
    vint32m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_int16>::vlanes());
    vint64m8_t res = __riscv_vadd(__riscv_vwadd_vv(t2, t3, VTraits<v_int16>::vlanes()), __riscv_vwadd_vv(t0, t1, VTraits<v_int16>::vlanes()), VTraits<v_int16>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_int16>::vlanes()); \
    return __riscv_vlmul_trunc_i64m2(res);
}
inline v_int64 v_dotprod_expand(const v_int16& a, const v_int16& b,
                                  const v_int64& c)
{
    vuint64m2_t one64 = __riscv_vmv_v_x_u64m2(1, VTraits<v_uint64>::vlanes()); \
    vuint16m2_t one16 = __riscv_vreinterpret_u16m2(one64); \
    vbool8_t mask = __riscv_vmseq(one16, 1, VTraits<v_uint16>::vlanes()); \
    vint32m4_t t0 = __riscv_vwmul(a, b, VTraits<v_int16>::vlanes()); \
    vint32m4_t t1= __riscv_vslide1down(t0, 0, VTraits<v_int16>::vlanes());
    vint32m4_t t2= __riscv_vslide1down(t1, 0, VTraits<v_int16>::vlanes());
    vint32m4_t t3= __riscv_vslide1down(t2, 0, VTraits<v_int16>::vlanes());
    vint64m8_t res = __riscv_vadd(__riscv_vwadd_vv(t2, t3, VTraits<v_int16>::vlanes()), __riscv_vwadd_vv(t0, t1, VTraits<v_int16>::vlanes()), VTraits<v_int16>::vlanes());
    res = __riscv_vcompress(res, mask, VTraits<v_int16>::vlanes()); \
    return __riscv_vadd(__riscv_vlmul_trunc_i64m2(res), c, VTraits<v_int16>::vlanes());
}

// // 32 >> 64f
#if CV_SIMD_SCALABLE_64F
inline v_float64 v_dotprod_expand(const v_int32& a, const v_int32& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64 v_dotprod_expand(const v_int32& a,   const v_int32& b,
                                    const v_float64& c)
{ return v_add(v_dotprod_expand(a, b) , c); }
#endif

//////// Fast Dot Product ////////
// 16 >> 32
inline v_int32 v_dotprod_fast(const v_int16& a, const v_int16& b)
{
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, VTraits<vint32m1_t>::vlanes());
    return __riscv_vset(__riscv_vmv_v_x_i32m2(0, VTraits<v_int32>::vlanes()), 0, __riscv_vredsum_tu(zero, __riscv_vwmul(a, b, VTraits<v_int16>::vlanes()), zero,  VTraits<v_int16>::vlanes()));
}
inline v_int32 v_dotprod_fast(const v_int16& a, const v_int16& b, const v_int32& c)
{
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, VTraits<vint32m1_t>::vlanes());
    return  __riscv_vadd(c, __riscv_vset(__riscv_vmv_v_x_i32m2(0, VTraits<v_int32>::vlanes()), 0, __riscv_vredsum_tu(zero, __riscv_vwmul(a, b, VTraits<v_int16>::vlanes()), zero,  VTraits<v_int16>::vlanes())), VTraits<v_int32>::vlanes());
}

// 32 >> 64
inline v_int64 v_dotprod_fast(const v_int32& a, const v_int32& b)
{
    vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, VTraits<vint64m1_t>::vlanes());
    return __riscv_vset(__riscv_vmv_v_x_i64m2(0, VTraits<v_int64>::vlanes()), 0, __riscv_vredsum_tu(zero, __riscv_vwmul(a, b, VTraits<v_int32>::vlanes()), zero,  VTraits<v_int32>::vlanes()));
}
inline v_int64 v_dotprod_fast(const v_int32& a, const v_int32& b, const v_int64& c)
{
    vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, VTraits<vint64m1_t>::vlanes());
    return  __riscv_vadd(c, __riscv_vset(__riscv_vmv_v_x_i64m2(0, VTraits<v_int64>::vlanes()), 0, __riscv_vredsum_tu(zero, __riscv_vwmul(a, b, VTraits<v_int32>::vlanes()), zero,  VTraits<v_int32>::vlanes())), VTraits<v_int64>::vlanes());
}


// 8 >> 32
inline v_uint32 v_dotprod_expand_fast(const v_uint8& a, const v_uint8& b)
{
    vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, VTraits<vuint32m1_t>::vlanes());
    auto res = __riscv_vwredsumu_tu(zero, __riscv_vwmulu(a, b, VTraits<v_uint8>::vlanes()), zero,   VTraits<v_uint8>::vlanes());
    return __riscv_vset(__riscv_vmv_v_x_u32m2(0, VTraits<v_uint32>::vlanes()), 0, res);
}
inline v_uint32 v_dotprod_expand_fast(const v_uint8& a, const v_uint8& b, const v_uint32& c)
{
    vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, VTraits<vuint32m1_t>::vlanes());
    auto res = __riscv_vwredsumu_tu(zero, __riscv_vwmulu(a, b, VTraits<v_uint8>::vlanes()), zero,   VTraits<v_uint8>::vlanes());
    return __riscv_vadd(c, __riscv_vset(__riscv_vmv_v_x_u32m2(0, VTraits<v_uint32>::vlanes()), 0, res), VTraits<v_uint32>::vlanes());
}
inline v_int32 v_dotprod_expand_fast(const v_int8& a, const v_int8& b)
{
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, VTraits<vint32m1_t>::vlanes());
    return __riscv_vset(__riscv_vmv_v_x_i32m2(0, VTraits<v_uint32>::vlanes()), 0, __riscv_vwredsum_tu(zero,  __riscv_vwmul(a, b, VTraits<v_int8>::vlanes()), zero,  VTraits<v_int8>::vlanes()));
}
inline v_int32 v_dotprod_expand_fast(const v_int8& a, const v_int8& b, const v_int32& c)
{
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, VTraits<vint32m1_t>::vlanes());
    return __riscv_vadd(c, __riscv_vset(__riscv_vmv_v_x_i32m2(0, VTraits<v_uint32>::vlanes()), 0, __riscv_vwredsum_tu(zero, __riscv_vwmul(a, b, VTraits<v_int8>::vlanes()), zero,  VTraits<v_int8>::vlanes())), VTraits<v_int32>::vlanes());
}

// 16 >> 64
inline v_uint64 v_dotprod_expand_fast(const v_uint16& a, const v_uint16& b)
{
    vuint64m1_t zero = __riscv_vmv_v_x_u64m1(0, VTraits<vuint64m1_t>::vlanes());
    return __riscv_vset(__riscv_vmv_v_x_u64m2(0, VTraits<v_uint64>::vlanes()), 0, __riscv_vwredsumu_tu(zero,  __riscv_vwmulu(a, b, VTraits<v_uint16>::vlanes()), zero,  VTraits<v_uint16>::vlanes()));
}
inline v_uint64 v_dotprod_expand_fast(const v_uint16& a, const v_uint16& b, const v_uint64& c)
{
    vuint64m1_t zero = __riscv_vmv_v_x_u64m1(0, VTraits<vuint64m1_t>::vlanes());
    return __riscv_vadd(c, __riscv_vset(__riscv_vmv_v_x_u64m2(0, VTraits<v_uint64>::vlanes()), 0, __riscv_vwredsumu_tu(zero,  __riscv_vwmulu(a, b, VTraits<v_uint16>::vlanes()), zero,  VTraits<v_uint16>::vlanes())), VTraits<v_uint64>::vlanes());
}
inline v_int64 v_dotprod_expand_fast(const v_int16& a, const v_int16& b)
{
    vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, VTraits<vint64m1_t>::vlanes());
    return __riscv_vset(__riscv_vmv_v_x_i64m2(0, VTraits<v_int64>::vlanes()), 0, __riscv_vwredsum_tu(zero,  __riscv_vwmul(a, b, VTraits<v_int16>::vlanes()), zero,  VTraits<v_int16>::vlanes()));
}
inline v_int64 v_dotprod_expand_fast(const v_int16& a, const v_int16& b, const v_int64& c)
{
    vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, VTraits<vint64m1_t>::vlanes());
    return __riscv_vadd(c, __riscv_vset(__riscv_vmv_v_x_i64m2(0, VTraits<v_int64>::vlanes()), 0, __riscv_vwredsum_tu(zero,  __riscv_vwmul(a, b, VTraits<v_int16>::vlanes()), zero,  VTraits<v_int16>::vlanes())), VTraits<v_int64>::vlanes());
}

// 32 >> 64f
#if CV_SIMD_SCALABLE_64F
inline v_float64 v_dotprod_expand_fast(const v_int32& a, const v_int32& b)
{ return v_cvt_f64(v_dotprod_fast(a, b)); }
inline v_float64 v_dotprod_expand_fast(const v_int32& a, const v_int32& b, const v_float64& c)
{ return v_add(v_dotprod_expand_fast(a, b) , c); }
#endif

// TODO: only 128 bit now.
#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_matmul(  const v_float16 &v,
                            const v_float16 &m0, const v_float16 &m1,
                            const v_float16 &m2, const v_float16 &m3,
                            const v_float16 &m4, const v_float16 &m5,
                            const v_float16 &m6, const v_float16 &m7) {
    vfloat16m2_t res;
    res = __riscv_vfmul_vf_f16m2(m0, (_Float16)v_extract_n(v, 0), VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 1), m1, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 2), m2, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 3), m3, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 4), m4, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 5), m5, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 6), m6, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 7), m7, VTraits<v_float16>::vlanes());
    return res;
}
inline v_float16 v_matmuladd(  const v_float16 &v,
                               const v_float16 &m0, const v_float16 &m1,
                               const v_float16 &m2, const v_float16 &m3,
                               const v_float16 &m4, const v_float16 &m5,
                               const v_float16 &m6,
                               const v_float16 &a) {
    vfloat16m2_t res;
    res = __riscv_vfmul_vf_f16m2(m0, (_Float16)v_extract_n(v, 0), VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 1), m1, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 2), m2, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 3), m3, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 4), m4, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 5), m5, VTraits<v_float16>::vlanes());
    res = __riscv_vfmacc_vf_f16m2(res, (_Float16)v_extract_n(v, 6), m6, VTraits<v_float16>::vlanes());
    return  __riscv_vfadd(res, a, VTraits<v_float16>::vlanes());
}
#endif

inline v_float32 v_matmul(const v_float32& v, const v_float32& mat0,
                            const v_float32& mat1, const v_float32& mat2,
                            const v_float32& mat3)
{
    vfloat32m2_t res;
    res = __riscv_vfmul_vf_f32m2(mat0, v_extract_n(v, 0), VTraits<v_float32>::vlanes());
    res = __riscv_vfmacc_vf_f32m2(res, v_extract_n(v, 1), mat1, VTraits<v_float32>::vlanes());
    res = __riscv_vfmacc_vf_f32m2(res, v_extract_n(v, 2), mat2, VTraits<v_float32>::vlanes());
    res = __riscv_vfmacc_vf_f32m2(res, v_extract_n(v, 3), mat3, VTraits<v_float32>::vlanes());
    return res;
}

// TODO: only 128 bit now.
inline v_float32 v_matmuladd(const v_float32& v, const v_float32& mat0,
                               const v_float32& mat1, const v_float32& mat2,
                               const v_float32& a)
{
    vfloat32m2_t res = __riscv_vfmul_vf_f32m2(mat0, v_extract_n(v,0), VTraits<v_float32>::vlanes());
    res = __riscv_vfmacc_vf_f32m2(res, v_extract_n(v,1), mat1, VTraits<v_float32>::vlanes());
    res = __riscv_vfmacc_vf_f32m2(res, v_extract_n(v,2), mat2, VTraits<v_float32>::vlanes());
    return __riscv_vfadd(res, a, VTraits<v_float32>::vlanes());
}

inline void v_cleanup() {}

#include "intrin_math.hpp"
#if CV_SIMD_SCALABLE_FP16
inline v_float16 v_exp(const v_float16& x) { return v_exp_default_16f<v_float16, v_int16>(x); }
inline v_float16 v_log(const v_float16& x) { return v_log_default_16f<v_float16, v_int16>(x); }
inline void v_sincos(const v_float16& x, v_float16& s, v_float16& c) { v_sincos_default_16f<v_float16, v_int16>(x, s, c); }
inline v_float16 v_sin(const v_float16& x) { return v_sin_default_16f<v_float16, v_int16>(x); }
inline v_float16 v_cos(const v_float16& x) { return v_cos_default_16f<v_float16, v_int16>(x); }
#endif
inline v_float32 v_exp(const v_float32& x) { return v_exp_default_32f<v_float32, v_int32>(x); }
inline v_float32 v_log(const v_float32& x) { return v_log_default_32f<v_float32, v_int32>(x); }
inline void v_sincos(const v_float32& x, v_float32& s, v_float32& c) { v_sincos_default_32f<v_float32, v_int32>(x, s, c); }
inline v_float32 v_sin(const v_float32& x) { return v_sin_default_32f<v_float32, v_int32>(x); }
inline v_float32 v_cos(const v_float32& x) { return v_cos_default_32f<v_float32, v_int32>(x); }
inline v_float32 v_erf(const v_float32& x) { return v_erf_default_32f<v_float32, v_int32>(x); }

inline v_float64 v_exp(const v_float64& x) { return v_exp_default_64f<v_float64, v_int64>(x); }
inline v_float64 v_log(const v_float64& x) { return v_log_default_64f<v_float64, v_int64>(x); }
inline void v_sincos(const v_float64& x, v_float64& s, v_float64& c) { v_sincos_default_64f<v_float64, v_int64>(x, s, c); }
inline v_float64 v_sin(const v_float64& x) { return v_sin_default_64f<v_float64, v_int64>(x); }
inline v_float64 v_cos(const v_float64& x) { return v_cos_default_64f<v_float64, v_int64>(x); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} //namespace cv

#endif //OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP
