/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of OpenCV project.
// See opencv/LICENSE for the actual licensing terms.
// Contributed by Giles Payne
//
//M*/

#include "precomp.hpp"

#include <vector>

#include "opencv2/core/hal/intrin.hpp"

#include "corner.simd.hpp"
#include "corner.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

#include "corner.hpp"

namespace cv {

static void doCalcMinEigenValLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst)
{
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(doCalcMinEigenValLine, (j, width, cov_x2, cov_xy, cov_y2, dst),
        CV_CPU_DISPATCH_MODES_ALL);
}

static void doCalcHarrisLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k) {
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(doCalcHarrisLine, (j, width, cov_x2, cov_xy, cov_y2, dst, k),
        CV_CPU_DISPATCH_MODES_ALL);
}

static void doCornerEigenValsVecsLine(int& j, int width, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2) {
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(doCornerEigenValsVecsLine, (j, width, dxdata, dydata, cov_x2, cov_xy, cov_y2),
        CV_CPU_DISPATCH_MODES_ALL);
}

static float float_load(const float* addr) {
    return *addr;
}

static void float_store(float* addr, const float& val) {
    *addr = val;
}

static float float_setall(float val) {
    return val;
}

static float float_sqrt(const float& val) {
    return std::sqrt(val);
}

static void doCalcMinEigenValLine_NOSIMD(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst) {
    calcMinEigenValLine<float, float_load, float_store, float_setall, float_sqrt>(j, width, 1, cov_x2, cov_xy, cov_y2, dst );
}

static void doCalcHarrisLine_NOSIMD(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k) {
    calcHarrisLine<float, float_load, float_store, float_setall>(j, width, 1, cov_x2, cov_xy, cov_y2, dst, k);
}

static void doCornerEigenValsVecsLine_NOSIMD(int& j, int width, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2) {
    cornerEigenValsVecsLine<float, float_load, float_store>(j, width, 1, dxdata, dydata, cov_x2, cov_xy, cov_y2);
}

void calcMinEigenValLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst) {
    doCalcMinEigenValLine(j, width, cov_x2, cov_xy, cov_y2, dst);
    doCalcMinEigenValLine_NOSIMD(j, width, cov_x2, cov_xy, cov_y2, dst);
}

void calcHarrisLine(int& j, int width, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k) {
    doCalcHarrisLine(j, width, cov_x2, cov_xy, cov_y2, dst, k);
    doCalcHarrisLine_NOSIMD(j, width, cov_x2, cov_xy, cov_y2, dst, k);
}

void cornerEigenValsVecsLine(int& j, int width, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2) {
    doCornerEigenValsVecsLine(j, width, dxdata, dydata, cov_x2, cov_xy, cov_y2);
    doCornerEigenValsVecsLine_NOSIMD(j, width, dxdata, dydata, cov_x2, cov_xy, cov_y2);
}

} // namespace
