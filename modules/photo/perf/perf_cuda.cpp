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

#include "perf_precomp.hpp"

#include "opencv2/photo/cuda.hpp"
#include "opencv2/ts/cuda_perf.hpp"

#include "opencv2/opencv_modules.hpp"

#if defined (HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAIMGPROC)

namespace opencv_test { namespace {
using namespace perf;

#define CUDA_DENOISING_IMAGE_SIZES testing::Values(perf::szVGA, perf::sz720p)

//////////////////////////////////////////////////////////////////////
// nonLocalMeans

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth, MatCn, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, CUDA_NonLocalMeans,
            Combine(CUDA_DENOISING_IMAGE_SIZES,
                    Values<MatDepth>(CV_8U),
                    CUDA_CHANNELS_1_3,
                    Values(21),
                    Values(5)))
{
    declare.time(600.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int search_window_size = GET_PARAM(3);
    const int block_size = GET_PARAM(4);

    const float h = 10;
    const int borderMode = cv::BORDER_REFLECT101;

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::nonLocalMeans(d_src, dst, h, search_window_size, block_size, borderMode);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}


//////////////////////////////////////////////////////////////////////
// fastNonLocalMeans

DEF_PARAM_TEST(Sz_Depth_Cn_WinSz_BlockSz, cv::Size, MatDepth, MatCn, int, int);

PERF_TEST_P(Sz_Depth_Cn_WinSz_BlockSz, CUDA_FastNonLocalMeans,
            Combine(CUDA_DENOISING_IMAGE_SIZES,
                    Values<MatDepth>(CV_8U),
                    CUDA_CHANNELS_1_3,
                    Values(21),
                    Values(7)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int search_window_size = GET_PARAM(2);
    const int block_size = GET_PARAM(3);

    const float h = 10;
    const int type = CV_MAKE_TYPE(depth, 1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::fastNlMeansDenoising(d_src, dst, h, search_window_size, block_size);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::fastNlMeansDenoising(src, dst, h, block_size, search_window_size);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// fastNonLocalMeans (colored)

DEF_PARAM_TEST(Sz_Depth_WinSz_BlockSz, cv::Size, MatDepth, int, int);

PERF_TEST_P(Sz_Depth_WinSz_BlockSz, CUDA_FastNonLocalMeansColored,
            Combine(CUDA_DENOISING_IMAGE_SIZES,
                    Values<MatDepth>(CV_8U),
                    Values(21),
                    Values(7)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int search_window_size = GET_PARAM(2);
    const int block_size = GET_PARAM(3);

    const float h = 10;
    const int type = CV_MAKE_TYPE(depth, 3);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::fastNlMeansDenoisingColored(d_src, dst, h, h, search_window_size, block_size);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::fastNlMeansDenoisingColored(src, dst, h, h, block_size, search_window_size);

        CPU_SANITY_CHECK(dst);
    }
}

}} // namespace
#endif
