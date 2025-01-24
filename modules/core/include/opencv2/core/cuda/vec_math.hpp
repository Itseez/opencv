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

#ifndef OPENCV_CUDA_VECMATH_HPP
#define OPENCV_CUDA_VECMATH_HPP

#include "vec_traits.hpp"
#include "saturate_cast.hpp"

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{

// saturate_cast

namespace vec_math_detail
{
    template <int cn, typename VecD> struct SatCastHelper;
    template <typename VecD> struct SatCastHelper<1, VecD>
    {
        template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
        {
            typedef typename VecTraits<VecD>::elem_type D;
            return VecTraits<VecD>::make(saturate_cast<D>(v.x));
        }
    };
    template <typename VecD> struct SatCastHelper<2, VecD>
    {
        template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
        {
            typedef typename VecTraits<VecD>::elem_type D;
            return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y));
        }
    };
    template <typename VecD> struct SatCastHelper<3, VecD>
    {
        template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
        {
            typedef typename VecTraits<VecD>::elem_type D;
            return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z));
        }
    };
    template <typename VecD> struct SatCastHelper<4, VecD>
    {
        template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
        {
            typedef typename VecTraits<VecD>::elem_type D;
            return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z), saturate_cast<D>(v.w));
        }
    };

    template <typename VecD, typename VecS> static __device__ __forceinline__ VecD saturate_cast_helper(const VecS& v)
    {
        return SatCastHelper<VecTraits<VecD>::cn, VecD>::cast(v);
    }
}

template<typename T> static __device__ __forceinline__ T saturate_cast(const uchar1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const char1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const ushort1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const short1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const uint1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const int1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const float1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const double1& v) {return vec_math_detail::saturate_cast_helper<T>(v);}

template<typename T> static __device__ __forceinline__ T saturate_cast(const uchar2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const char2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const ushort2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const short2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const uint2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const int2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const float2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const double2& v) {return vec_math_detail::saturate_cast_helper<T>(v);}

template<typename T> static __device__ __forceinline__ T saturate_cast(const uchar3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const char3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const ushort3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const short3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const uint3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const int3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const float3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const double3& v) {return vec_math_detail::saturate_cast_helper<T>(v);}

template<typename T> static __device__ __forceinline__ T saturate_cast(const uchar4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const char4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const ushort4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const short4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const uint4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const int4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const float4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}
template<typename T> static __device__ __forceinline__ T saturate_cast(const double4& v) {return vec_math_detail::saturate_cast_helper<T>(v);}

// unary operators

#define CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(op, input_type, output_type) \
    __device__ __forceinline__ output_type ## 1 operator op(const input_type ## 1 & a) \
    { \
        return VecTraits<output_type ## 1>::make(op (a.x)); \
    } \
    __device__ __forceinline__ output_type ## 2 operator op(const input_type ## 2 & a) \
    { \
        return VecTraits<output_type ## 2>::make(op (a.x), op (a.y)); \
    } \
    __device__ __forceinline__ output_type ## 3 operator op(const input_type ## 3 & a) \
    { \
        return VecTraits<output_type ## 3>::make(op (a.x), op (a.y), op (a.z)); \
    } \
    __device__ __forceinline__ output_type ## 4 operator op(const input_type ## 4 & a) \
    { \
        return VecTraits<output_type ## 4>::make(op (a.x), op (a.y), op (a.z), op (a.w)); \
    }

CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(-, char, char)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(-, short, short)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(-, int, int)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(-, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(-, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(!, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(~, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(~, char, char)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(~, ushort, ushort)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(~, short, short)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(~, int, int)
CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(~, uint, uint)

#undef CV_CUDEV_IMPLEMENT_VEC_UNARY_OP

// unary functions

#define CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(func_name, func, input_type, output_type) \
    __device__ __forceinline__ output_type ## 1 func_name(const input_type ## 1 & a) \
    { \
        return VecTraits<output_type ## 1>::make(func (a.x)); \
    } \
    __device__ __forceinline__ output_type ## 2 func_name(const input_type ## 2 & a) \
    { \
        return VecTraits<output_type ## 2>::make(func (a.x), func (a.y)); \
    } \
    __device__ __forceinline__ output_type ## 3 func_name(const input_type ## 3 & a) \
    { \
        return VecTraits<output_type ## 3>::make(func (a.x), func (a.y), func (a.z)); \
    } \
    __device__ __forceinline__ output_type ## 4 func_name(const input_type ## 4 & a) \
    { \
        return VecTraits<output_type ## 4>::make(func (a.x), func (a.y), func (a.z), func (a.w)); \
    }

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(abs, ::fabsf, float, float)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrtf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sqrt, ::sqrt, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::expf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp, ::exp, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2f, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp2, ::exp2, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10f, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(exp10, ::exp10, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::logf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log, ::log, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2f, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log2, ::log2, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10f, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(log10, ::log10, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sinf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sin, ::sin, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cosf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cos, ::cos, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tanf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tan, ::tan, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asinf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asin, ::asin, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acosf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acos, ::acos, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atanf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atan, ::atan, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinhf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(sinh, ::sinh, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::coshf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(cosh, ::cosh, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanhf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(tanh, ::tanh, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinhf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(asinh, ::asinh, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acoshf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(acosh, ::acosh, double, double)

CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, char, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, short, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, int, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanhf, float, float)
CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(atanh, ::atanh, double, double)

#undef CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC

// a += b

#define CV_CUDEV_IMPLEMENT_VEC_BINARY_OP_ASSIGN(op, input_type, output_type) \
    __device__ __forceinline__ output_type ## 1 & operator op ## = (output_type ## 1 & a, const input_type ## 1 & b) \
    { \
        return a = VecTraits<output_type ## 1>::make(a.x op b.x); \
    } \
    __device__ __forceinline__ output_type ## 2 & operator op ## = (output_type ## 2 & a, const input_type ## 2 & b) \
    { \
        return a = VecTraits<output_type ## 2>::make(a.x op b.x, a.y op b.y); \
    } \
    __device__ __forceinline__ output_type ## 3 & operator op ## = (output_type ## 3 & a, const input_type ## 3 & b) \
    { \
        return a = VecTraits<output_type ## 3>::make(a.x op b.x, a.y op b.y, a.z op b.z); \
    } \
    __device__ __forceinline__ output_type ## 4 & operator op ## = (output_type ## 4 & a, const input_type ## 4 & b) \
    { \
        return a = VecTraits<output_type ## 4>::make(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w); \
    }

// binary operators (vec & vec)

#define CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(op, input_type, output_type) \
    __device__ __forceinline__ output_type ## 1 operator op(const input_type ## 1 & a, const input_type ## 1 & b) \
    { \
        return VecTraits<output_type ## 1>::make(a.x op b.x); \
    } \
    __device__ __forceinline__ output_type ## 2 operator op(const input_type ## 2 & a, const input_type ## 2 & b) \
    { \
        return VecTraits<output_type ## 2>::make(a.x op b.x, a.y op b.y); \
    } \
    __device__ __forceinline__ output_type ## 3 operator op(const input_type ## 3 & a, const input_type ## 3 & b) \
    { \
        return VecTraits<output_type ## 3>::make(a.x op b.x, a.y op b.y, a.z op b.z); \
    } \
    __device__ __forceinline__ output_type ## 4 operator op(const input_type ## 4 & a, const input_type ## 4 & b) \
    { \
        return VecTraits<output_type ## 4>::make(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w); \
    } \
    CV_CUDEV_IMPLEMENT_VEC_BINARY_OP_ASSIGN(op, input_type, output_type)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, uchar, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, char, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, ushort, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, short, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, uint, uint)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(+, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, uchar, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, char, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, ushort, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, short, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, uint, uint)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(-, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, uchar, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, char, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, ushort, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, short, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, uint, uint)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(*, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, uchar, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, char, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, ushort, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, short, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, uint, uint)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(/, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&, char, char)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&, ushort, ushort)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&, short, short)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&, uint, uint)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(|, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(|, char, char)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(|, ushort, ushort)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(|, short, short)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(|, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(|, uint, uint)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(^, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(^, char, char)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(^, ushort, ushort)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(^, short, short)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(^, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(^, uint, uint)

#undef CV_CUDEV_IMPLEMENT_VEC_BINARY_OP_ASSIGN
#define CV_CUDEV_IMPLEMENT_VEC_BINARY_OP_ASSIGN(op, input_type, output_type)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(==, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(!=, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(>=, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(<=, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(&&, double, uchar)

CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, char, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, ushort, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, short, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, int, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, uint, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, float, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(||, double, uchar)

#undef CV_CUDEV_IMPLEMENT_VEC_BINARY_OP
#undef CV_CUDEV_IMPLEMENT_VEC_BINARY_OP_ASSIGN

// binary operators (vec & scalar)

#define CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(op, input_type, scalar_type, output_type) \
    __device__ __forceinline__ output_type ## 1 operator op(const input_type ## 1 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 1>::make(a.x op s); \
    } \
    __device__ __forceinline__ output_type ## 1 operator op(scalar_type s, const input_type ## 1 & b) \
    { \
        return VecTraits<output_type ## 1>::make(s op b.x); \
    } \
    __device__ __forceinline__ output_type ## 2 operator op(const input_type ## 2 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 2>::make(a.x op s, a.y op s); \
    } \
    __device__ __forceinline__ output_type ## 2 operator op(scalar_type s, const input_type ## 2 & b) \
    { \
        return VecTraits<output_type ## 2>::make(s op b.x, s op b.y); \
    } \
    __device__ __forceinline__ output_type ## 3 operator op(const input_type ## 3 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 3>::make(a.x op s, a.y op s, a.z op s); \
    } \
    __device__ __forceinline__ output_type ## 3 operator op(scalar_type s, const input_type ## 3 & b) \
    { \
        return VecTraits<output_type ## 3>::make(s op b.x, s op b.y, s op b.z); \
    } \
    __device__ __forceinline__ output_type ## 4 operator op(const input_type ## 4 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 4>::make(a.x op s, a.y op s, a.z op s, a.w op s); \
    } \
    __device__ __forceinline__ output_type ## 4 operator op(scalar_type s, const input_type ## 4 & b) \
    { \
        return VecTraits<output_type ## 4>::make(s op b.x, s op b.y, s op b.z, s op b.w); \
    }

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, uchar, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, char, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, ushort, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, short, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, uint, uint, uint)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(+, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, uchar, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, char, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, ushort, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, short, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, uint, uint, uint)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(-, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, uchar, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, char, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, ushort, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, short, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, uint, uint, uint)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(*, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, uchar, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, char, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, ushort, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, short, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, uint, uint, uint)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(/, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(==, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(!=, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(>=, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(<=, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&&, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, char, char, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, ushort, ushort, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, short, short, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, int, int, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, uint, uint, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, float, float, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(||, double, double, uchar)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&, char, char, char)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&, ushort, ushort, ushort)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&, short, short, short)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(&, uint, uint, uint)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(|, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(|, char, char, char)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(|, ushort, ushort, ushort)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(|, short, short, short)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(|, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(|, uint, uint, uint)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(^, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(^, char, char, char)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(^, ushort, ushort, ushort)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(^, short, short, short)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(^, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(^, uint, uint, uint)

#undef CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP

// binary function (vec & vec)

#define CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(func_name, func, input_type, output_type) \
    __device__ __forceinline__ output_type ## 1 func_name(const input_type ## 1 & a, const input_type ## 1 & b) \
    { \
        return VecTraits<output_type ## 1>::make(func (a.x, b.x)); \
    } \
    __device__ __forceinline__ output_type ## 2 func_name(const input_type ## 2 & a, const input_type ## 2 & b) \
    { \
        return VecTraits<output_type ## 2>::make(func (a.x, b.x), func (a.y, b.y)); \
    } \
    __device__ __forceinline__ output_type ## 3 func_name(const input_type ## 3 & a, const input_type ## 3 & b) \
    { \
        return VecTraits<output_type ## 3>::make(func (a.x, b.x), func (a.y, b.y), func (a.z, b.z)); \
    } \
    __device__ __forceinline__ output_type ## 4 func_name(const input_type ## 4 & a, const input_type ## 4 & b) \
    { \
        return VecTraits<output_type ## 4>::make(func (a.x, b.x), func (a.y, b.y), func (a.z, b.z), func (a.w, b.w)); \
    }

CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::max, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::max, char, char)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::max, ushort, ushort)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::max, short, short)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::max, uint, uint)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::max, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::fmaxf, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(max, ::fmax, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::min, uchar, uchar)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::min, char, char)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::min, ushort, ushort)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::min, short, short)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::min, uint, uint)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::min, int, int)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::fminf, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(min, ::fmin, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, char, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, short, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, uint, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, int, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypotf, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(hypot, ::hypot, double, double)

CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, uchar, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, char, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, ushort, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, short, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, uint, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, int, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2f, float, float)
CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(atan2, ::atan2, double, double)

#undef CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC

// binary function (vec & scalar)

#define CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(func_name, func, input_type, scalar_type, output_type) \
    __device__ __forceinline__ output_type ## 1 func_name(const input_type ## 1 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 1>::make(func ((output_type) a.x, (output_type) s)); \
    } \
    __device__ __forceinline__ output_type ## 1 func_name(scalar_type s, const input_type ## 1 & b) \
    { \
        return VecTraits<output_type ## 1>::make(func ((output_type) s, (output_type) b.x)); \
    } \
    __device__ __forceinline__ output_type ## 2 func_name(const input_type ## 2 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 2>::make(func ((output_type) a.x, (output_type) s), func ((output_type) a.y, (output_type) s)); \
    } \
    __device__ __forceinline__ output_type ## 2 func_name(scalar_type s, const input_type ## 2 & b) \
    { \
        return VecTraits<output_type ## 2>::make(func ((output_type) s, (output_type) b.x), func ((output_type) s, (output_type) b.y)); \
    } \
    __device__ __forceinline__ output_type ## 3 func_name(const input_type ## 3 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 3>::make(func ((output_type) a.x, (output_type) s), func ((output_type) a.y, (output_type) s), func ((output_type) a.z, (output_type) s)); \
    } \
    __device__ __forceinline__ output_type ## 3 func_name(scalar_type s, const input_type ## 3 & b) \
    { \
        return VecTraits<output_type ## 3>::make(func ((output_type) s, (output_type) b.x), func ((output_type) s, (output_type) b.y), func ((output_type) s, (output_type) b.z)); \
    } \
    __device__ __forceinline__ output_type ## 4 func_name(const input_type ## 4 & a, scalar_type s) \
    { \
        return VecTraits<output_type ## 4>::make(func ((output_type) a.x, (output_type) s), func ((output_type) a.y, (output_type) s), func ((output_type) a.z, (output_type) s), func ((output_type) a.w, (output_type) s)); \
    } \
    __device__ __forceinline__ output_type ## 4 func_name(scalar_type s, const input_type ## 4 & b) \
    { \
        return VecTraits<output_type ## 4>::make(func ((output_type) s, (output_type) b.x), func ((output_type) s, (output_type) b.y), func ((output_type) s, (output_type) b.z), func ((output_type) s, (output_type) b.w)); \
    }

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::max, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::max, char, char, char)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::max, ushort, ushort, ushort)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::max, short, short, short)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::max, uint, uint, uint)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::max, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmaxf, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(max, ::fmax, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::min, uchar, uchar, uchar)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::min, char, char, char)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::min, ushort, ushort, ushort)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::min, short, short, short)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::min, uint, uint, uint)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::min, int, int, int)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fminf, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(min, ::fmin, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypotf, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(hypot, ::hypot, double, double, double)

CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, uchar, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, uchar, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, char, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, char, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, ushort, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, ushort, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, short, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, short, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, uint, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, uint, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, int, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, int, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2f, float, float, float)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, float, double, double)
CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(atan2, ::atan2, double, double, double)

#undef CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC

}}} // namespace cv { namespace cuda { namespace device

//! @endcond

#endif // OPENCV_CUDA_VECMATH_HPP
