/*M///////////////////////////////////////////////////////////////////////////////////////
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "perf_precomp.hpp"
#include <algorithm>
#include <functional>

namespace opencv_test
{
using namespace perf;

CV_ENUM(Method, RANSAC, LMEDS)
typedef tuple<int, double, Method, size_t> AffineParams;
typedef TestBaseWithParam<AffineParams> EstimateAffine;
#define ESTIMATE_PARAMS Combine(Values(100000, 5000, 100), Values(0.99, 0.95, 0.9), Method::all(), Values(10, 0))

static float rngIn(float from, float to) { return from + (to-from) * (float)theRNG(); }

static Mat rngPartialAffMat() {
    double theta = rngIn(0, (float)CV_PI*2.f);
    double scale = rngIn(0, 3);
    double tx = rngIn(-2, 2);
    double ty = rngIn(-2, 2);
    double aff[2*3] = { std::cos(theta) * scale, -std::sin(theta) * scale, tx,
                        std::sin(theta) * scale,  std::cos(theta) * scale, ty };
    return Mat(2, 3, CV_64F, aff).clone();
}

static Mat rngPartialAffRbMat() {
    double theta = rngIn(0, (float)CV_PI*2.f);
    double tx = rngIn(-2, 2);
    double ty = rngIn(-2, 2);
    double aff[2*3] = { std::cos(theta), -std::sin(theta), tx,
                        std::sin(theta),  std::cos(theta), ty };
    return Mat(2, 3, CV_64F, aff).clone();
}

PERF_TEST_P( EstimateAffine, EstimateAffine2D, ESTIMATE_PARAMS )
{
    AffineParams params = GetParam();
    const int n = get<0>(params);
    const double confidence = get<1>(params);
    const int method = get<2>(params);
    const size_t refining = get<3>(params);

    Mat aff(2, 3, CV_64F);
    cv::randu(aff, -2., 2.);

    // LMEDS can't handle more than 50% outliers (by design)
    int m;
    if (method == LMEDS)
        m = 3*n/5;
    else
        m = 2*n/5;
    const float shift_outl = 15.f;
    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC2);
    Mat tpts(1, n, CV_32FC2);

    randu(fpts, 0., 100.);
    transform(fpts, tpts, aff);

    /* adding noise to some points */
    Mat outliers = tpts.colRange(m, n);
    outliers.reshape(1) += shift_outl;

    Mat noise (outliers.size(), outliers.type());
    randu(noise, 0., noise_level);
    outliers += noise;

    Mat aff_est;
    vector<uchar> inliers (n);

    warmup(inliers, WARMUP_WRITE);
    warmup(fpts, WARMUP_READ);
    warmup(tpts, WARMUP_READ);

    TEST_CYCLE()
    {
        aff_est = estimateAffine2D(fpts, tpts, inliers, method, 3, 2000, confidence, refining);
    }

    // we already have accuracy tests
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( EstimateAffine, EstimateAffinePartial2D, ESTIMATE_PARAMS )
{
    AffineParams params = GetParam();
    const int n = get<0>(params);
    const double confidence = get<1>(params);
    const int method = get<2>(params);
    const size_t refining = get<3>(params);

    Mat aff = rngPartialAffMat();

    int m;
    // LMEDS can't handle more than 50% outliers (by design)
    if (method == LMEDS)
        m = 3*n/5;
    else
        m = 2*n/5;
    const float shift_outl = 15.f;    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC2);
    Mat tpts(1, n, CV_32FC2);

    randu(fpts, 0., 100.);
    transform(fpts, tpts, aff);

    /* adding noise*/
    Mat outliers = tpts.colRange(m, n);
    outliers.reshape(1) += shift_outl;

    Mat noise (outliers.size(), outliers.type());
    randu(noise, 0., noise_level);
    outliers += noise;

    Mat aff_est;
    vector<uchar> inliers (n);

    warmup(inliers, WARMUP_WRITE);
    warmup(fpts, WARMUP_READ);
    warmup(tpts, WARMUP_READ);

    TEST_CYCLE()
    {
        aff_est = estimateAffinePartial2D(fpts, tpts, inliers, method, 3, 2000, confidence, refining);
    }

    // we already have accuracy tests
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( EstimateAffine, estimateAffineRigidBody2D, ESTIMATE_PARAMS )
{
    AffineParams params = GetParam();
    const int n = get<0>(params);
    const double confidence = get<1>(params);
    const int method = get<2>(params);
    const size_t refining = get<3>(params);

    Mat aff = rngPartialAffRbMat();

    int m;
    // LMEDS can't handle more than 50% outliers (by design)
    if (method == LMEDS)
        m = 3*n/5;
    else
        m = 2*n/5;
    const float shift_outl = 15.f;    const float noise_level = 20.f;

    Mat fpts(1, n, CV_32FC2);
    Mat tpts(1, n, CV_32FC2);

    randu(fpts, 0., 100.);
    transform(fpts, tpts, aff);

    /* adding noise*/
    Mat outliers = tpts.colRange(m, n);
    outliers.reshape(1) += shift_outl;

    Mat noise (outliers.size(), outliers.type());
    randu(noise, 0., noise_level);
    outliers += noise;

    Mat aff_est;
    vector<uchar> inliers (n);

    warmup(inliers, WARMUP_WRITE);
    warmup(fpts, WARMUP_READ);
    warmup(tpts, WARMUP_READ);

    TEST_CYCLE()
    {
        aff_est = estimateAffineRigidBody2D(fpts, tpts, inliers, method, 3, 2000, confidence, refining);
    }

    // we already have accuracy tests
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
