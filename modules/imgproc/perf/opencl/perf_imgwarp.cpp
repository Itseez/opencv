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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// WarpAffine ////////////////////////

CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC)

typedef tuple<Size, MatType, InterType> WarpAffineParams;
typedef TestBaseWithParam<WarpAffineParams> WarpAffineFixture;

OCL_PERF_TEST_P(WarpAffineFixture, WarpAffine,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, InterType::all()))
{
    static const double coeffs[2][3] =
    {
        { cos(CV_PI / 6), -sin(CV_PI / 6), 100.0  },
        { sin(CV_PI / 6), cos(CV_PI / 6) , -100.0 }
    };
    Mat M(2, 3, CV_64F, (void *)coeffs);

    const WarpAffineParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : interpolation == INTER_CUBIC ? 2e-3 : 1e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::warpAffine(src, dst, M, srcSize, interpolation);

    SANITY_CHECK(dst, eps);
}

///////////// WarpPerspective ////////////////////////

typedef WarpAffineParams WarpPerspectiveParams;
typedef TestBaseWithParam<WarpPerspectiveParams> WarpPerspectiveFixture;

OCL_PERF_TEST_P(WarpPerspectiveFixture, WarpPerspective,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134,
                                   OCL_PERF_ENUM(InterType(INTER_NEAREST), InterType(INTER_LINEAR))))
{
    static const double coeffs[3][3] =
    {
        {cos(CV_PI / 6), -sin(CV_PI / 6), 100.0},
        {sin(CV_PI / 6), cos(CV_PI / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);

    const WarpPerspectiveParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::warpPerspective(src, dst, M, srcSize, interpolation);

    SANITY_CHECK(dst, eps);
}

///////////// Resize ////////////////////////

typedef tuple<Size, MatType, InterType, double> ResizeParams;
typedef TestBaseWithParam<ResizeParams> ResizeFixture;

OCL_PERF_TEST_P(ResizeFixture, Resize,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134,
                               OCL_PERF_ENUM(InterType(INTER_NEAREST), InterType(INTER_LINEAR)),
                               ::testing::Values(0.5, 2.0)))
{
    const ResizeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interType = get<2>(params);
    double scale = get<3>(params);
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    UMat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, interType);

    SANITY_CHECK(dst, eps);
}

typedef tuple<Size, MatType, double> ResizeAreaParams;
typedef TestBaseWithParam<ResizeAreaParams> ResizeAreaFixture;

OCL_PERF_TEST_P(ResizeAreaFixture, Resize,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, ::testing::Values(0.3, 0.5, 0.6)))
{
    const ResizeAreaParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    double scale = get<2>(params);
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    UMat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, cv::INTER_AREA);

    SANITY_CHECK(dst, eps);
}

typedef ResizeAreaParams ResizeLinearExactParams;
typedef TestBaseWithParam<ResizeLinearExactParams> ResizeLinearExactFixture;

OCL_PERF_TEST_P(ResizeLinearExactFixture, Resize,
            ::testing::Combine(OCL_TEST_SIZES, ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4), ::testing::Values(0.5, 2.0)))
{
    const ResizeAreaParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    double scale = get<2>(params);
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));
    const double eps = 1e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    UMat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, cv::INTER_LINEAR_EXACT);

    SANITY_CHECK(dst, eps);
}

typedef tuple<Size, MatType, double> ResizeOnnxParams;
typedef TestBaseWithParam<ResizeOnnxParams> ResizeOnnxFixture;

OCL_PERF_TEST_P(ResizeOnnxFixture, LinearAntialias,
    Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, Values(0.3, 0.5, 0.6)))
{
    const ResizeOnnxParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    double scale = get<2>(params);
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    UMat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::resizeOnnx(src, dst, dstSize, Point2d(), INTER_LINEAR | INTER_ANTIALIAS);

    SANITY_CHECK_NOTHING();
}

///////////// Remap ////////////////////////

typedef tuple<Size, MatType, InterType> RemapParams;
typedef TestBaseWithParam<RemapParams> RemapFixture;

OCL_PERF_TEST_P(RemapFixture, Remap,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134,
                               OCL_PERF_ENUM(InterType(INTER_NEAREST), InterType(INTER_LINEAR))))
{
    const RemapParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params), borderMode = BORDER_CONSTANT;
    //const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    UMat xmap(srcSize, CV_32FC1), ymap(srcSize, CV_32FC1);

    {
        Mat _xmap = xmap.getMat(ACCESS_WRITE), _ymap = ymap.getMat(ACCESS_WRITE);
        for (int i = 0; i < srcSize.height; ++i)
        {
            float * const xmap_row = _xmap.ptr<float>(i);
            float * const ymap_row = _ymap.ptr<float>(i);

            for (int j = 0; j < srcSize.width; ++j)
            {
                xmap_row[j] = (j - srcSize.width * 0.5f) * 0.75f + srcSize.width * 0.5f;
                ymap_row[j] = (i - srcSize.height * 0.5f) * 0.75f + srcSize.height * 0.5f;
            }
        }
    }
    declare.in(src, WARMUP_RNG).in(xmap, ymap, WARMUP_READ).out(dst);

    OCL_TEST_CYCLE() cv::remap(src, dst, xmap, ymap, interpolation, borderMode);

    SANITY_CHECK_NOTHING();
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
