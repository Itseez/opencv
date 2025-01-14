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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

void loadImage(string path, Mat &img)
{
    img = imread(path, -1);
    ASSERT_FALSE(img.empty()) << "Could not load input image " << path;
}

void checkEqual(Mat img0, Mat img1, double threshold, const string& name)
{
    double max = 1.0;
    minMaxLoc(abs(img0 - img1), NULL, &max);
    ASSERT_FALSE(max > threshold) << "max=" << max << " threshold=" << threshold << " method=" << name;
}

static vector<float> DEFAULT_VECTOR;
void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times = DEFAULT_VECTOR)
{
    std::ifstream list_file((path + "list.txt").c_str());
    ASSERT_TRUE(list_file.is_open());
    string name;
    float val;
    while(list_file >> name >> val) {
        Mat img = imread(path + name);
        ASSERT_FALSE(img.empty()) << "Could not load input image " << path + name;
        images.push_back(img);
        times.push_back(1 / val);
    }
    list_file.close();
}

void loadResponseCSV(String path, Mat& response)
{
    response = Mat(256, 1, CV_32FC3);
    std::ifstream resp_file(path.c_str());
    for(int i = 0; i < 256; i++) {
        for(int c = 0; c < 3; c++) {
            resp_file >> response.at<Vec3f>(i)[c];
            resp_file.ignore(1);
        }
    }
    resp_file.close();
}

TEST(Photo_Tonemap, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/tonemap/";

    Mat img, expected, result;
    loadImage(test_path + "image.hdr", img);
    float gamma = 2.2f;

    Ptr<Tonemap> linear = createTonemap(gamma);
    linear->process(img, result);
    loadImage(test_path + "linear.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Simple");

    Ptr<TonemapDrago> drago = createTonemapDrago(gamma);
    drago->process(img, result);
    loadImage(test_path + "drago.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Drago");

    Ptr<TonemapReinhard> reinhard = createTonemapReinhard(gamma);
    reinhard->process(img, result);
    loadImage(test_path + "reinhard.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Reinhard");

    Ptr<TonemapMantiuk> mantiuk = createTonemapMantiuk(gamma);
    mantiuk->process(img, result);
    loadImage(test_path + "mantiuk.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Mantiuk");
}

TEST(Photo_AlignMTB, regression)
{
    const int TESTS_COUNT = 100;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "shared/";

    string file_name = folder + "lena.png";
    Mat img;
    loadImage(file_name, img);
    cvtColor(img, img, COLOR_RGB2GRAY);

    int max_bits = 5;
    int max_shift = 32;
    srand(static_cast<unsigned>(time(0)));
    int errors = 0;

    Ptr<AlignMTB> align = createAlignMTB(max_bits);
    RNG rng = theRNG();

    for(int i = 0; i < TESTS_COUNT; i++) {
        Point shift(rng.uniform(0, max_shift), rng.uniform(0, max_shift));
        Mat res;
        align->shiftMat(img, res, shift);
        Point calc = align->calculateShift(img, res);
        errors += (calc != -shift);
    }
    ASSERT_TRUE(errors < 5) << errors << " errors";
}

TEST(Photo_MergeMertens, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    loadExposureSeq((test_path + "exposures/").c_str() , images);

    Ptr<MergeMertens> merge = createMergeMertens();

    Mat result, expected;
    loadImage(test_path + "merge/mertens.png", expected);
    merge->process(images, result);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(expected, result, 3, "Mertens");

    Mat uniform(100, 100, CV_8UC3);
    uniform = Scalar(0, 255, 0);

    images.clear();
    images.push_back(uniform);

    merge->process(images, result);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(uniform, result, 1e-2f, "Mertens");
}

TEST(Photo_MergeDebevec, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response;
    loadExposureSeq(test_path + "exposures/", images, times);
    loadResponseCSV(test_path + "exposures/response.csv", response);

    Ptr<MergeDebevec> merge = createMergeDebevec();

    Mat result, expected;
    loadImage(test_path + "merge/debevec.hdr", expected);
    merge->process(images, result, times, response);

    Ptr<Tonemap> map = createTonemap();
    map->process(result, result);
    map->process(expected, expected);

    checkEqual(expected, result, 1e-2f, "Debevec");
}

TEST(Photo_MergeRobertson, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images, times);
    Ptr<MergeRobertson> merge = createMergeRobertson();
    Mat result, expected;
    loadImage(test_path + "merge/robertson.hdr", expected);
    merge->process(images, result, times);

    const float eps = 6.f;
    checkEqual(expected, result, eps, "MergeRobertson");
}

TEST(Photo_CalibrateDebevec, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response, expected;
    loadExposureSeq(test_path + "exposures/", images, times);
    loadResponseCSV(test_path + "calibrate/debevec.csv", expected);
    Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();

    calibrate->process(images, response, times);
    Mat diff = abs(response - expected);
    diff = diff.mul(1.0f / response);
    double max;
    minMaxLoc(diff, NULL, &max);
#if defined(__arm__) || defined(__aarch64__)
    ASSERT_LT(max, 0.2);
#else
    ASSERT_LT(max, 0.1);
#endif
}

TEST(Photo_CalibrateRobertson, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response, expected;
    loadExposureSeq(test_path + "exposures/", images, times);
    loadResponseCSV(test_path + "calibrate/robertson.csv", expected);

    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson();
    calibrate->process(images, response, times);
    checkEqual(expected, response, 1e-1f, "CalibrateRobertson");
}

TEST(Photo_CalibrateRobertson, bug_18180)
{
    vector<Mat> images;
    vector<cv::String> fn;
    string test_path = cvtest::TS::ptr()->get_data_path() + "hdr/exposures/bug_18180/";
    for(int i = 1; i <= 4; ++i)
        images.push_back(imread(test_path + std::to_string(i) + ".jpg"));
    vector<float> times {15.0f, 2.5f, 0.25f, 0.33f};
    Mat response, expected;
    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson(2, 0.01f);
    calibrate->process(images, response, times);
    Mat response_no_nans = response.clone();
    patchNaNs(response_no_nans);
    // since there should be no NaNs, original response vs. response with NaNs patched should be identical
    EXPECT_EQ(0.0, cv::norm(response, response_no_nans, NORM_L2));
}

}} // namespace
