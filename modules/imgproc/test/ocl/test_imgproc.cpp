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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan, lyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Wu Zailong, bullet@yeah.net
//    Xu Pang, pangxu010@163.com
//    Sen Liu, swjtuls1987@126.com
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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(ImgprocTestBase, MatType,
                int, // blockSize
                int, // border type
                bool) // roi or not
{
    int type, borderType, blockSize;
    bool useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        blockSize = GET_PARAM(1);
        borderType = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }

    void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0, bool relative = false)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

//////////////////////////////// copyMakeBorder ////////////////////////////////////////////

PARAM_TEST_CASE(CopyMakeBorder, MatDepth, // depth
                Channels, // channels
                bool, // isolated or not
                BorderType, // border type
                bool) // roi or not
{
    int type, borderType;
    bool useRoi;

    TestUtils::Border border;
    Scalar val;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        borderType = GET_PARAM(3);

        if (GET_PARAM(2))
            borderType |= BORDER_ISOLATED;

        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        border = randomBorder(0, MAX_VALUE << 2);
        val = randomScalar(-MAX_VALUE, MAX_VALUE);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        dstBorder.top += border.top;
        dstBorder.lef += border.lef;
        dstBorder.rig += border.rig;
        dstBorder.bot += border.bot;

        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near()
    {
        OCL_EXPECT_MATS_NEAR(dst, 0);
    }
};

OCL_TEST_P(CopyMakeBorder, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::copyMakeBorder(src_roi, dst_roi, border.top, border.bot, border.lef, border.rig, borderType, val));
        OCL_ON(cv::copyMakeBorder(usrc_roi, udst_roi, border.top, border.bot, border.lef, border.rig, borderType, val));

        Near();
    }
}

//////////////////////////////// equalizeHist //////////////////////////////////////////////

typedef ImgprocTestBase EqualizeHist;

OCL_TEST_P(EqualizeHist, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::equalizeHist(src_roi, dst_roi));
        OCL_ON(cv::equalizeHist(usrc_roi, udst_roi));

        Near(1);
    }
}

//////////////////////////////// Corners test //////////////////////////////////////////

struct CornerTestBase :
        public ImgprocTestBase
{
    void random_roi()
    {
        Mat image = readImageType("../gpu/stereobm/aloe-L.png", type);
        ASSERT_FALSE(image.empty());

        bool isFP = CV_MAT_DEPTH(type) >= CV_32F;
        float val = 255.0f;
        if (isFP)
        {
            image.convertTo(image, -1, 1.0 / 255);
            val /= 255.0f;
        }

        Size roiSize = image.size();
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);

        Size wholeSize = Size(roiSize.width + srcBorder.lef + srcBorder.rig, roiSize.height + srcBorder.top + srcBorder.bot);
        src = randomMat(wholeSize, type, -val, val, false);
        src_roi = src(Rect(srcBorder.lef, srcBorder.top, roiSize.width, roiSize.height));
        image.copyTo(src_roi);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, CV_32FC1, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }
};

typedef CornerTestBase CornerMinEigenVal;

OCL_TEST_P(CornerMinEigenVal, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        int apertureSize = 3;

        OCL_OFF(cv::cornerMinEigenVal(src_roi, dst_roi, blockSize, apertureSize, borderType));
        OCL_ON(cv::cornerMinEigenVal(usrc_roi, udst_roi, blockSize, apertureSize, borderType));

        // The corner kernel uses native_sqrt() which has implementation defined accuracy.
        // If we're using a CL implementation that isn't intel, test with relaxed accuracy.
        if (!ocl::useOpenCL() || ocl::Device::getDefault().isIntel())
            Near(1e-5, true);
        else
            Near(0.1, true);
    }
}

//////////////////////////////// cornerHarris //////////////////////////////////////////

typedef CornerTestBase CornerHarris;

OCL_TEST_P(CornerHarris, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        int apertureSize = 3;
        double k = randomDouble(0.01, 0.9);

        OCL_OFF(cv::cornerHarris(src_roi, dst_roi, blockSize, apertureSize, k, borderType));
        OCL_ON(cv::cornerHarris(usrc_roi, udst_roi, blockSize, apertureSize, k, borderType));

        Near(1e-6, true);
    }
}

//////////////////////////////// preCornerDetect //////////////////////////////////////////

typedef ImgprocTestBase PreCornerDetect;

OCL_TEST_P(PreCornerDetect, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        const int apertureSize = blockSize;

        OCL_OFF(cv::preCornerDetect(src_roi, dst_roi, apertureSize, borderType));
        OCL_ON(cv::preCornerDetect(usrc_roi, udst_roi, apertureSize, borderType));

        Near(1e-6, true);
    }
}


////////////////////////////////// integral /////////////////////////////////////////////////

struct Integral :
        public ImgprocTestBase
{
    int sdepth, sqdepth;

    TEST_DECLARE_OUTPUT_PARAMETER(dst2);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        sdepth = GET_PARAM(1);
        sqdepth = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }

    void random_roi()
    {
        ASSERT_EQ(CV_MAT_CN(type), 1);

        Size roiSize = randomSize(1, MAX_VALUE), isize = Size(roiSize.width + 1, roiSize.height + 1);
        Border srcBorder = randomBorder(0, useRoi ? 2 : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? 2 : 0);
        randomSubMat(dst, dst_roi, isize, dstBorder, sdepth, 5, 16);

        Border dst2Border = randomBorder(0, useRoi ? 2 : 0);
        randomSubMat(dst2, dst2_roi, isize, dst2Border, sqdepth, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst2);
    }

    void Near2(double threshold = 0.0, bool relative = false)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst2, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst2, threshold);
    }
};

OCL_TEST_P(Integral, Mat1)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::integral(src_roi, dst_roi, sdepth));
        OCL_ON(cv::integral(usrc_roi, udst_roi, sdepth));

        Near();
    }
}

OCL_TEST_P(Integral, Mat2)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::integral(src_roi, dst_roi, dst2_roi, sdepth, sqdepth));
        OCL_ON(cv::integral(usrc_roi, udst_roi, udst2_roi, sdepth, sqdepth));

        Near();
        sqdepth == CV_32F ? Near2(1e-6, true) : Near2();
    }
}

////////////////////////////////////////  threshold //////////////////////////////////////////////

struct Threshold :
        public ImgprocTestBase
{
    int thresholdType;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        thresholdType = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }
};

OCL_TEST_P(Threshold, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        double maxVal = randomDouble(20.0, 127.0);
        double thresh = randomDouble(0.0, maxVal);

        OCL_OFF(cv::threshold(src_roi, dst_roi, thresh, maxVal, thresholdType));
        OCL_ON(cv::threshold(usrc_roi, udst_roi, thresh, maxVal, thresholdType));

        Near(1);
    }
}

struct Threshold_Dryrun :
        public ImgprocTestBase
{
    int thresholdType;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        thresholdType = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }
};

OCL_TEST_P(Threshold_Dryrun, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        double maxVal = randomDouble(20.0, 127.0);
        double thresh = randomDouble(0.0, maxVal);

        const int _thresholdType = thresholdType | THRESH_DRYRUN;

        src_roi.copyTo(dst_roi);
        usrc_roi.copyTo(udst_roi);

        OCL_OFF(cv::threshold(src_roi, dst_roi, thresh, maxVal, _thresholdType));
        OCL_ON(cv::threshold(usrc_roi, udst_roi, thresh, maxVal, _thresholdType));

        OCL_EXPECT_MATS_NEAR(dst, 0);
    }
}

struct Threshold_masked :
    public ImgprocTestBase
{
    int thresholdType;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        thresholdType = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }
};

OCL_TEST_P(Threshold_masked, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        double maxVal = randomDouble(20.0, 127.0);
        double thresh = randomDouble(0.0, maxVal);

        const int _thresholdType = thresholdType;

        cv::Size sz = src_roi.size();
        cv::Mat mask_roi = cv::Mat::zeros(sz, CV_8UC1);
        cv::RotatedRect ellipseRect((cv::Point2f)cv::Point(sz.width/2, sz.height/2), (cv::Size2f)sz, 0);
        cv::ellipse(mask_roi, ellipseRect, cv::Scalar::all(255), cv::FILLED);//for very different mask alignments

        cv::UMat umask_roi(mask_roi.size(), mask_roi.type());
        mask_roi.copyTo(umask_roi.getMat(cv::AccessFlag::ACCESS_WRITE));

        src_roi.copyTo(dst_roi);
        usrc_roi.copyTo(udst_roi);

        OCL_OFF(cv::threshold(src_roi, dst_roi, mask_roi, thresh, maxVal, _thresholdType));
        OCL_ON(cv::threshold(usrc_roi, udst_roi, umask_roi, thresh, maxVal, _thresholdType));

        OCL_EXPECT_MATS_NEAR(dst, 0);
    }
}


/////////////////////////////////////////// CLAHE //////////////////////////////////////////////////

PARAM_TEST_CASE(CLAHETest, Size, double, bool)
{
    Size gridSize;
    double clipLimit;
    bool useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        gridSize = GET_PARAM(0);
        clipLimit = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }

    void random_roi()
    {
        Size roiSize = randomSize(std::max(gridSize.height, gridSize.width), MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, CV_8UC1, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, CV_8UC1, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0)
    {
        OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

OCL_TEST_P(CLAHETest, Accuracy)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        Ptr<CLAHE> clahe = cv::createCLAHE(clipLimit, gridSize);

        OCL_OFF(clahe->apply(src_roi, dst_roi));
        OCL_ON(clahe->apply(usrc_roi, udst_roi));

        Near(1.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, EqualizeHist, Combine(
                            Values((MatType)CV_8UC1),
                            Values(0), // not used
                            Values(0), // not used
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, CornerMinEigenVal, Combine(
                            Values((MatType)CV_8UC1, (MatType)CV_32FC1),
                            Values(3, 5),
                            Values((BorderType)BORDER_CONSTANT, (BorderType)BORDER_REPLICATE,
                                   (BorderType)BORDER_REFLECT, (BorderType)BORDER_REFLECT101),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, CornerHarris, Combine(
                            Values((MatType)CV_8UC1, CV_32FC1),
                            Values(3, 5),
                            Values( (BorderType)BORDER_CONSTANT, (BorderType)BORDER_REPLICATE,
                                    (BorderType)BORDER_REFLECT, (BorderType)BORDER_REFLECT_101),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, PreCornerDetect, Combine(
                            Values((MatType)CV_8UC1, CV_32FC1),
                            Values(3, 5),
                            Values( (BorderType)BORDER_CONSTANT, (BorderType)BORDER_REPLICATE,
                                    (BorderType)BORDER_REFLECT, (BorderType)BORDER_REFLECT_101),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, Integral, Combine(
                            Values((MatType)CV_8UC1), // TODO does not work with CV_32F, CV_64F
                            Values(CV_32SC1, CV_32FC1), // desired sdepth
                            Values(CV_32FC1, CV_64FC1), // desired sqdepth
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, Threshold, Combine(
                            Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4,
                                   CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                                   CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4),
                            Values(0),
                            Values(ThreshOp(THRESH_BINARY),
                                   ThreshOp(THRESH_BINARY_INV), ThreshOp(THRESH_TRUNC),
                                   ThreshOp(THRESH_TOZERO), ThreshOp(THRESH_TOZERO_INV)),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, Threshold_Dryrun, Combine(
                            Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4,
                                   CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                                   CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4),
                            Values(0),
                            Values(ThreshOp(THRESH_BINARY),
                                   ThreshOp(THRESH_BINARY_INV), ThreshOp(THRESH_TRUNC),
                                   ThreshOp(THRESH_TOZERO), ThreshOp(THRESH_TOZERO_INV)),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, Threshold_masked, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_16SC1, CV_16SC3, CV_16UC1, CV_16UC3, CV_32FC1, CV_32FC3, CV_64FC1, CV_64FC3),
                            Values(0),
                            Values(ThreshOp(THRESH_BINARY),
                                   ThreshOp(THRESH_BINARY_INV), ThreshOp(THRESH_TRUNC),
                                   ThreshOp(THRESH_TOZERO), ThreshOp(THRESH_TOZERO_INV)),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, CLAHETest, Combine(
                            Values(Size(4, 4), Size(32, 8), Size(8, 64)),
                            Values(0.0, 10.0, 62.0, 300.0),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgprocTestBase, CopyMakeBorder, Combine(
                            testing::Values((MatDepth)CV_8U, (MatDepth)CV_16S, (MatDepth)CV_32S, (MatDepth)CV_32F),
                            testing::Values(Channels(1), Channels(3), (Channels)4),
                            Bool(), // border isolated or not
                            Values((BorderType)BORDER_CONSTANT, (BorderType)BORDER_REPLICATE, (BorderType)BORDER_REFLECT,
                                   (BorderType)BORDER_WRAP, (BorderType)BORDER_REFLECT_101),
                            Bool()));

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
