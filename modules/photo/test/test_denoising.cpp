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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

//#define DUMP_RESULTS

#ifdef DUMP_RESULTS
#  define DUMP(image, path) imwrite(path, image)
#else
#  define DUMP(image, path)
#endif


TEST(Photo_DenoisingGrayscale, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    string expected_path = folder + "lena_noised_denoised_grayscale_tw=7_sw=21_h=10.png";

    Mat original = imread(original_path, IMREAD_GRAYSCALE);
    Mat expected = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    Mat result;
    fastNlMeansDenoising(original, result, 10);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingColored, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    string expected_path = folder + "lena_noised_denoised_lab12_tw=7_sw=21_h=10_h2=10.png";

    Mat original = imread(original_path, IMREAD_COLOR);
    Mat expected = imread(expected_path, IMREAD_COLOR);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    Mat result;
    fastNlMeansDenoisingColored(original, result, 10, 10);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingGrayscaleMulti, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    string expected_path = folder + "lena_noised_denoised_multi_tw=7_sw=21_h=15.png";
    Mat expected = imread(expected_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    vector<Mat> original(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original[i] = imread(original_path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(original[i].empty()) << "Could not load input image " << original_path;
    }

    Mat result;
    fastNlMeansDenoisingMulti(original, result, imgs_count / 2, imgs_count, 15);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingColoredMulti, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    string expected_path = folder + "lena_noised_denoised_multi_lab12_tw=7_sw=21_h=10_h2=15.png";
    Mat expected = imread(expected_path, IMREAD_COLOR);
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    vector<Mat> original(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original[i] = imread(original_path, IMREAD_COLOR);
        ASSERT_FALSE(original[i].empty()) << "Could not load input image " << original_path;
    }

    Mat result;
    fastNlMeansDenoisingColoredMulti(original, result, imgs_count / 2, imgs_count, 10, 15);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_White, issue_2646)
{
    cv::Mat img(50, 50, CV_8UC1, cv::Scalar::all(255));
    cv::Mat filtered;
    cv::fastNlMeansDenoising(img, filtered);

    int nonWhitePixelsCount = (int)img.total() - cv::countNonZero(filtered == img);

    ASSERT_EQ(0, nonWhitePixelsCount);
}

TEST(Photo_Denoising, speed)
{
    string imgname = string(cvtest::TS::ptr()->get_data_path()) + "shared/5MP.png";
    Mat src = imread(imgname, IMREAD_GRAYSCALE), dst;

    double t = (double)getTickCount();
    fastNlMeansDenoising(src, dst, 5, 7, 21);
    t = (double)getTickCount() - t;
    printf("execution time: %gms\n", t*1000./getTickFrequency());
}
// Related issue :
// - https://github.com/opencv/opencv/issues/26582
class Photo_DenoisingGrayscaleMulti16Bit : public ::testing::Test {
protected:
    struct TestConfig {
        int width = 127;
        int height = 129;
        int imgs_count = 3;
        float h = 15.0f;
        int templateWindowSize = 7;
        int searchWindowSize = 21;
    };

    static double computePSNR(const cv::Mat& I1, const cv::Mat& I2) {
        CV_Assert(I1.type() == I2.type() && I1.size() == I2.size());
        cv::Mat s1;
        cv::absdiff(I1, I2, s1);
        s1.convertTo(s1, CV_32F);
        s1 = s1.mul(s1);
        cv::Scalar s = cv::sum(s1);
        double mse = s[0] / static_cast<double>(I1.total());

        if (mse == 0) return INFINITY;

        double max_pixel = 65535.0;
        return 10.0 * log10((max_pixel * max_pixel) / mse);
    }

    static std::vector<cv::Mat> generateTestImages(const TestConfig& config, int minVal, int maxVal) {
        std::vector<cv::Mat> images(config.imgs_count);
        for (int i = 0; i < config.imgs_count; i++) {
            images[i] = cv::Mat(config.height, config.width, CV_16UC1);
            cv::randu(images[i], cv::Scalar::all(minVal), cv::Scalar::all(maxVal));
        }
        return images;
    }

    void runDenoisingTest(const std::vector<cv::Mat>& inputImages, const TestConfig& config, const std::string& testCaseName, int expectedMinVal, int expectedMaxVal, double psnrThreshold = -1) {
        cv::Mat result;
        std::vector<float> h_vec = {config.h};

        ASSERT_NO_THROW({
            cv::fastNlMeansDenoisingMulti(inputImages, result, static_cast<int>(inputImages.size() / 2), static_cast<int>(inputImages.size()), h_vec, config.templateWindowSize, config.searchWindowSize, cv::NORM_L1);
        }) << "fastNlMeansDenoisingMulti threw an unexpected exception in " << testCaseName;

        ASSERT_FALSE(result.empty()) << "Denoising result is empty in " << testCaseName;
        ASSERT_EQ(result.type(), CV_16UC1) << "Incorrect result type in " << testCaseName;
        ASSERT_EQ(result.size(), inputImages[0].size()) << "Incorrect result size in " << testCaseName;

        double minVal, maxVal;
        cv::minMaxLoc(result, &minVal, &maxVal);
        EXPECT_GE(minVal, expectedMinVal) << "Minimum value out of range in " << testCaseName;
        EXPECT_LE(maxVal, expectedMaxVal) << "Maximum value out of range in " << testCaseName;

        if (psnrThreshold > 0) {
            cv::Mat groundTruth = cv::Mat::ones(config.height, config.width, CV_16UC1) * 10000;
            double psnr = computePSNR(result, groundTruth);
            EXPECT_GT(psnr, psnrThreshold) << "PSNR is too low in " << testCaseName;
        }
    }
};

TEST_F(Photo_DenoisingGrayscaleMulti16Bit, ComprehensiveDenoisingTest)
{
    TestConfig config;

    auto randomImages = generateTestImages(config, 9500, 10500);
    runDenoisingTest(randomImages, config, "RandomValuesDenoising", 0, 65535, 30.0);

    auto maxImages = std::vector<cv::Mat>(config.imgs_count, cv::Mat::ones(config.height, config.width, CV_16UC1) * 65535);
    runDenoisingTest(maxImages, config, "MaxValueDenoising", 65535, 65535);

    auto zeroImages = std::vector<cv::Mat>(config.imgs_count, cv::Mat::zeros(config.height, config.width, CV_16UC1));
    runDenoisingTest(zeroImages, config, "ZeroValueDenoising", 0, 0);
}

}} // namespace
