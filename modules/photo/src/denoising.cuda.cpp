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

#include "precomp.hpp"

#include "opencv2/photo/cuda.hpp"
#include "opencv2/core/private.cuda.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_CUDAARITHM
#  include "opencv2/cudaarithm.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAIMGPROC
#  include "opencv2/cudaimgproc.hpp"
#endif

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || !defined(HAVE_OPENCV_CUDAARITHM) || !defined(HAVE_OPENCV_CUDAIMGPROC)

void cv::cuda::nonLocalMeans(InputArray, OutputArray, float, int, int, int, Stream&) { throw_no_cuda(); }
void cv::cuda::fastNlMeansDenoising(InputArray, OutputArray, float, int, int, Stream&) { throw_no_cuda(); }
void cv::cuda::fastNlMeansDenoisingColored(InputArray, OutputArray, float, float, int, int, Stream&) { throw_no_cuda(); }

#else

//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing (brute force)

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template<typename T>
        void nlm_bruteforce_gpu(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream);
    }
}}}

void cv::cuda::nonLocalMeans(InputArray _src, OutputArray _dst, float h, int search_window, int block_window, int borderMode, Stream& stream)
{
    using cv::cuda::device::imgproc::nlm_bruteforce_gpu;
    typedef void (*func_t)(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream);

    static const func_t funcs[] = {nlm_bruteforce_gpu<uchar>, nlm_bruteforce_gpu<uchar2>, nlm_bruteforce_gpu<uchar3>,
                                    nullptr/*nlm_bruteforce_gpu<uchar4>,*/, nlm_bruteforce_gpu<ushort>};

    const GpuMat src = _src.getGpuMat();

    CV_Assert(src.type() == CV_8U || src.type() == CV_8UC2 || src.type() == CV_8UC3 || src.type() == CV_16UC1);

    int func_idx = src.type() == CV_16U ? 4 : src.channels() - 1;
    const func_t func = funcs[func_idx];
    CV_Assert(func);

    int b = borderMode;
    CV_Assert(b == BORDER_REFLECT101 || b == BORDER_REPLICATE || b == BORDER_CONSTANT || b == BORDER_REFLECT || b == BORDER_WRAP);

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    func(src, dst, search_window/2, block_window/2, h, borderMode, StreamAccessor::getStream(stream));
}

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void nln_fast_get_buffer_size(const PtrStepSzb& src, int search_window, int block_window, int& buffer_cols, int& buffer_rows);

        template<typename T>
        void nlm_fast_gpu(const PtrStepSzb& src, PtrStepSzb dst, PtrStepi buffer,
                          int search_window, int block_window, float h, cudaStream_t stream);

        void fnlm_split_channels(const PtrStepSz<uchar3>& lab, PtrStepb l, PtrStep<uchar2> ab, cudaStream_t stream);
        void fnlm_merge_channels(const PtrStepb& l, const PtrStep<uchar2>& ab, PtrStepSz<uchar3> lab, cudaStream_t stream);
     }
}}}

void cv::cuda::fastNlMeansDenoising(InputArray _src, OutputArray _dst, float h, int search_window, int block_window, Stream& stream)
{
    const GpuMat src = _src.getGpuMat();

    CV_Assert((src.depth() == CV_8U && src.channels() < 4) || (src.depth() == CV_16U && src.channels() == 1));

    int border_size = search_window/2 + block_window/2;
    Size esize = src.size() + Size(border_size, border_size) * 2;

    BufferPool pool(stream);

    GpuMat extended_src = pool.getBuffer(esize, src.type());
    cv::cuda::copyMakeBorder(src, extended_src, border_size, border_size, border_size, border_size, cv::BORDER_DEFAULT, Scalar(), stream);
    GpuMat src_hdr = extended_src(Rect(Point2i(border_size, border_size), src.size()));

    int bcols, brows;
    device::imgproc::nln_fast_get_buffer_size(src_hdr, search_window, block_window, bcols, brows);
    GpuMat buffer = pool.getBuffer(brows, bcols, CV_32S);

    using namespace cv::cuda::device::imgproc;
    typedef void (*nlm_fast_t)(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);
    static const nlm_fast_t funcs[] = {nlm_fast_gpu<uchar>, nlm_fast_gpu<uchar2>, nlm_fast_gpu<uchar3>,
                                       nullptr/*nlm_fast_gpu<uchar4>,*/, nlm_fast_gpu<ushort>};

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    int func_idx = src.type() == CV_16U ? 4 : src.channels() - 1;
    funcs[func_idx](src_hdr, dst, buffer, search_window, block_window, h, StreamAccessor::getStream(stream));
}

void cv::cuda::fastNlMeansDenoisingColored(InputArray _src, OutputArray _dst, float h_luminance, float h_color, int search_window, int block_window, Stream& stream)
{
    const GpuMat src = _src.getGpuMat();

    CV_Assert(src.type() == CV_8UC3);

    BufferPool pool(stream);

    GpuMat lab = pool.getBuffer(src.size(), src.type());
    cv::cuda::cvtColor(src, lab, cv::COLOR_BGR2Lab, 0, stream);

    GpuMat l = pool.getBuffer(src.size(), CV_8U);
    GpuMat ab = pool.getBuffer(src.size(), CV_8UC2);
    device::imgproc::fnlm_split_channels(lab, l, ab, StreamAccessor::getStream(stream));

    fastNlMeansDenoising(l, l, h_luminance, search_window, block_window, stream);
    fastNlMeansDenoising(ab, ab, h_color, search_window, block_window, stream);

    device::imgproc::fnlm_merge_channels(l, ab, lab, StreamAccessor::getStream(stream));
    cv::cuda::cvtColor(lab, _dst, cv::COLOR_Lab2BGR, 0, stream);
}

#endif
