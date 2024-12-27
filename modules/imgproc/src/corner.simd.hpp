/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of OpenCV project.
// See opencv/LICENSE for the actual licensing terms.
// Contributed by Giles Payne
//
//M*/

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "corner.simd_helpers.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void doCalcMinEigenValLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst);

void doCalcHarrisLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k);

void doCornerEigenValsVecsLine(int& j, int width, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// currently there is no universal intrinsic implementation for AVX - so handle AVX separately to make optimal use of available instructions
#if CV_AVX && !CV_AVX2

struct v_float32x8
{
    enum { nlanes = 8 };
    __m256 val;
    explicit v_float32x8(__m256 v) : val(v) {}
    v_float32x8() {}
};

static v_float32x8 operator+(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_add_ps(lhs.val, rhs.val));
}

static v_float32x8 operator-(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_sub_ps(lhs.val, rhs.val));
}

static v_float32x8 operator*(const v_float32x8& lhs, const v_float32x8& rhs) {
    return v_float32x8(_mm256_mul_ps(lhs.val, rhs.val));
}

static v_float32x8 v_load(const float* ptr) {
    return v_float32x8(_mm256_load_ps(ptr));
}

static void v_store(float* ptr, const v_float32x8& vec) {
    _mm256_storeu_ps(ptr, vec.val);
}

static v_float32x8 v_setall(float val) {
    return v_float32x8(_mm256_set1_ps(val));
}

static v_float32x8 v_sqrt(const v_float32x8& vec) {
    return v_float32x8(_mm256_sqrt_ps(vec.val));
}

static v_float32x8 v_muladd(const v_float32x8& v1, const v_float32x8& v2, const v_float32x8& v3) {
    return (v1 * v2) + v3;
}

void doCalcMinEigenValLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst) {
    calcMinEigenValLine<v_float32x8, v_load, v_store, v_setall, v_sqrt, v_muladd>(j, width, v_float32x8::nlanes, cov_x2, cov_xy, cov_y2, dst);
}

void doCalcHarrisLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k) {
    calcHarrisLine<v_float32x8, v_load, v_store, v_setall>(j, width, v_float32x8::nlanes, cov_x2, cov_xy, cov_y2, dst, k);
}

void doCornerEigenValsVecsLine(int& j, int width, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2) {
    cornerEigenValsVecsLine<v_float32x8, v_load, v_store>(j, width, v_float32x8::nlanes, dxdata, dydata, cov_x2, cov_xy, cov_y2);
}

#else  // universal intrinsics

void doCalcMinEigenValLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst) {
    calcMinEigenValLine<v_float32, vx_load, v_store, vx_setall_f32, v_sqrt, v_muladd>(j, width, VTraits<v_float32>::vlanes(), cov_x2, cov_xy, cov_y2, dst);
}

void doCalcHarrisLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k) {
    calcHarrisLine<v_float32, vx_load, v_store, vx_setall_f32>(j, width, VTraits<v_float32>::vlanes(), cov_x2, cov_xy, cov_y2, dst, k);
}

void doCornerEigenValsVecsLine(int& j, int width, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2) {
    cornerEigenValsVecsLine<v_float32, vx_load, v_store>(j, width, VTraits<v_float32>::vlanes(), dxdata, dydata, cov_x2, cov_xy, cov_y2);
}

#endif

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
