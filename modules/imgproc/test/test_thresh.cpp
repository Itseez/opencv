/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_ThreshTest : public cvtest::ArrayTest
{
public:
    CV_ThreshTest(int test_type = 0);

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int thresh_type;
    double thresh_val;
    double max_val;
    int extra_type;
};


CV_ThreshTest::CV_ThreshTest(int test_type)
{
    CV_Assert( (test_type & cv::THRESH_MASK) == 0 );
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    optional_mask = false;
    element_wise_relative_error = true;
    extra_type = test_type;
    // Reduce number of test with automated thresholding
    if (extra_type != 0)
        test_case_count = 250;
}


void CV_ThreshTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 5, cn = cvtest::randInt(rng) % 4 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16S : depth == 2 ? CV_16U : depth == 3 ? CV_32F : CV_64F;

    if ( extra_type == cv::THRESH_OTSU )
    {
        depth = cvtest::randInt(rng) % 2 == 0 ? CV_8U : CV_16U;
        cn = 1;
    }

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth,cn);
    thresh_type = cvtest::randInt(rng) % 5;

    if( depth == CV_8U )
    {
        thresh_val = (cvtest::randReal(rng)*350. - 50.);
        max_val = (cvtest::randReal(rng)*350. - 50.);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = 255.f;
    }
    else if( depth == CV_16S )
    {
        double min_val = SHRT_MIN-100.f;
        max_val = SHRT_MAX+100.f;
        thresh_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        max_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = (double)SHRT_MAX;
    }
    else if( depth == CV_16U )
    {
        double min_val = -100.f;
        max_val = USHRT_MAX+100.f;
        thresh_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        max_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = (double)USHRT_MAX;
    }
    else
    {
        thresh_val = (cvtest::randReal(rng)*1000. - 500.);
        max_val = (cvtest::randReal(rng)*1000. - 500.);
    }
}


double CV_ThreshTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return FLT_EPSILON*10;
}


void CV_ThreshTest::run_func()
{
    cvThreshold( test_array[INPUT][0], test_array[OUTPUT][0],
                 thresh_val, max_val, thresh_type | extra_type);
}


static double compute_otsu_thresh(const Mat& _src)
{
    int depth = _src.depth();
    int width = _src.cols, height = _src.rows;
    const int N = 65536;
    std::vector<int> h(N, 0);
    int i, j;
    double mu = 0, scale = 1./(width*height);
    for(i = 0; i < height; ++i)
    {
        for(j = 0; j < width; ++j)
        {
          const int val  = depth == CV_16UC1 ? (int)_src.at<ushort>(i, j) : (int)_src.at<uchar>(i,j);
          h[val]++;
        }
    }
    for( i = 0; i < N; i++ )
    {
        mu += i*(double)h[i];
    }

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

static void test_threshold( const Mat& _src, Mat& _dst,
                            double thresh, double maxval, int thresh_type, int extra_type )
{
    int i, j;
    int depth = _src.depth(), cn = _src.channels();
    int width_n = _src.cols*cn, height = _src.rows;
    int ithresh = cvFloor(thresh);
    int imaxval, ithresh2;
    if (extra_type == cv::THRESH_OTSU)
    {
        thresh = compute_otsu_thresh(_src);
        ithresh = cvFloor(thresh);
    }

    if( depth == CV_8U )
    {
        ithresh2 = saturate_cast<uchar>(ithresh);
        imaxval = saturate_cast<uchar>(maxval);
    }
    else if( depth == CV_16S )
    {
        ithresh2 = saturate_cast<short>(ithresh);
        imaxval = saturate_cast<short>(maxval);
    }
    else if( depth == CV_16U )
    {
        ithresh2 = saturate_cast<ushort>(ithresh);
        imaxval = saturate_cast<ushort>(maxval);
    }
    else
    {
        ithresh2 = cvRound(ithresh);
        imaxval = cvRound(maxval);
    }

    CV_Assert( depth == CV_8U || depth == CV_16S || depth == CV_16U || depth == CV_32F || depth == CV_64F );

    switch( thresh_type )
    {
    case cv::THRESH_BINARY:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (short)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (ushort)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (float)(src[j] > thresh ? maxval : 0.f);
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = src[j] > thresh ? maxval : 0.0;
            }
        }
        break;
    case cv::THRESH_BINARY_INV:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (short)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (ushort)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (float)(src[j] > thresh ? 0.f : maxval);
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = src[j] > thresh ? 0.0 : maxval;
            }
        }
        break;
    case cv::THRESH_TRUNC:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (ushort)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = (float)(s > thresh ? thresh : s);
                }
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = s > thresh ? thresh : s;
                }
            }
        }
        break;
    case cv::THRESH_TOZERO:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (ushort)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = s > thresh ? s : 0.f;
                }
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = s > thresh ? s : 0.0;
                }
            }
        }
        break;
    case cv::THRESH_TOZERO_INV:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? 0 : s);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? 0 : s);
                }
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (ushort)(s > ithresh ? 0 : s);
                }
            }
            else if (depth == CV_32F)
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = s > thresh ? 0.f : s;
                }
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = s > thresh ? 0.0 : s;
                }
            }
        }
        break;
    default:
        CV_Assert(0);
    }
}


void CV_ThreshTest::prepare_to_validation( int /*test_case_idx*/ )
{
    test_threshold( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                   thresh_val, max_val, thresh_type, extra_type );
}

TEST(Imgproc_Threshold, accuracy) { CV_ThreshTest test; test.safe_run(); }
TEST(Imgproc_Threshold, accuracyOtsu) { CV_ThreshTest test(cv::THRESH_OTSU); test.safe_run(); }

BIGDATA_TEST(Imgproc_Threshold, huge)
{
    Mat m(65000, 40000, CV_8U);
    ASSERT_FALSE(m.isContinuous());

    uint64 i, n = (uint64)m.rows*m.cols;
    for( i = 0; i < n; i++ )
        m.data[i] = (uchar)(i & 255);

    cv::threshold(m, m, 127, 255, cv::THRESH_BINARY);
    int nz = cv::countNonZero(m);  // FIXIT 'int' is not enough here (overflow is possible with other inputs)
    ASSERT_EQ((uint64)nz, n / 2);
}

TEST(Imgproc_Threshold, threshold_dryrun)
{
    Size sz(16, 16);
    Mat input_original(sz, CV_8U, Scalar::all(2));
    Mat input = input_original.clone();
    std::vector<int> threshTypes = {THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV};
    std::vector<int> threshFlags = {0, THRESH_OTSU, THRESH_TRIANGLE};
    for(int threshType : threshTypes)
    {
        for(int threshFlag : threshFlags)
        {
            const int _threshType = threshType | threshFlag | THRESH_DRYRUN;
            cv::threshold(input, input, 2.0, 0.0, _threshType);
            EXPECT_MAT_NEAR(input, input_original, 0);
        }
    }
}

typedef tuple < bool, int, int, int, int > Imgproc_Threshold_Masked_Params_t;

typedef testing::TestWithParam< Imgproc_Threshold_Masked_Params_t > Imgproc_Threshold_Masked;

TEST_P(Imgproc_Threshold_Masked, threshold_mask)
{
    bool useROI = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int cn = get<2>(GetParam());
    int threshType = get<3>(GetParam());
    int threshFlag = get<4>(GetParam());
    const bool isValidConfig = ((depth == CV_8U) || ((threshFlag != THRESH_OTSU) && (threshFlag != THRESH_TRIANGLE)));
    if (isValidConfig)
    {
        const int _threshType = threshType | threshFlag;
        Size sz(127, 127);
        Size wrapperSize = useROI ? Size(sz.width+4, sz.height+4) : sz;
        Mat wrapper(wrapperSize, CV_MAKETYPE(depth, cn));
        Mat input = useROI ? Mat(wrapper, Rect(Point(), sz)) : wrapper;
        cv::randu(input, cv::Scalar::all(0), cv::Scalar::all(255));

        Mat mask = cv::Mat::zeros(sz, CV_8UC1);
        cv::RotatedRect ellipseRect((cv::Point2f)cv::Point(sz.width/2, sz.height/2), (cv::Size2f)sz, 0);
        cv::ellipse(mask, ellipseRect, cv::Scalar::all(255), cv::FILLED);//for very different mask alignments

        Mat output_with_mask = cv::Mat::zeros(sz, input.type());
        cv::threshold(input, output_with_mask, mask, 127, 255, _threshType);

        cv::bitwise_not(mask, mask);
        input.copyTo(output_with_mask, mask);

        Mat output_without_mask;
        cv::threshold(input, output_without_mask, 127, 255, _threshType);
        input.copyTo(output_without_mask, mask);

        EXPECT_MAT_NEAR(output_with_mask, output_without_mask, 0);
    }
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgproc_Threshold_Masked,
    testing::Combine(
        testing::Values(false, true),//use roi
        testing::Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),//depth
        testing::Values(1, 3),//channels
        testing::Values(THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV),// threshTypes
        testing::Values(0)//threshFlags there is no way to compare masked to non-masked OTSU/TRIANGLE since the threshold will be different
    )
);


TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_16085)
{
    Size sz(16, 16);
    Mat input(sz, CV_32F, Scalar::all(2));
    Mat result;
    cv::threshold(input, result, 2.0, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_21258)
{
    Size sz(16, 16);
    float val = nextafterf(16.0f, 0.0f);  // 0x417fffff, all bits in mantissa are 1
    Mat input(sz, CV_32F, Scalar::all(val));
    Mat result;
    cv::threshold(input, result, val, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_21258_Min)
{
    Size sz(16, 16);
    float min_val = -std::numeric_limits<float>::max();
    Mat input(sz, CV_32F, Scalar::all(min_val));
    Mat result;
    cv::threshold(input, result, min_val, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_21258_Max)
{
    Size sz(16, 16);
    float max_val = std::numeric_limits<float>::max();
    Mat input(sz, CV_32F, Scalar::all(max_val));
    Mat result;
    cv::threshold(input, result, max_val, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, mean)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 8);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold1.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, mean_inv)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 8);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold1.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    gt = Mat(gt.rows, gt.cols, CV_8UC1, cv::Scalar(255)) - gt;
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, gauss)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 200, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, -5);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold2.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, gauss_inv)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 200, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, -5);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold2.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    gt = Mat(gt.rows, gt.cols, CV_8UC1, cv::Scalar(200)) - gt;
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

}} // namespace
