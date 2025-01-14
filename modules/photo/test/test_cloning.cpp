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

#define OUTPUT_SAVING 0
#if OUTPUT_SAVING
#define SAVE(x) std::vector<int> params;\
                params.push_back(16);\
                params.push_back(0);\
                imwrite(folder + "output.png", x ,params);
#else
#define SAVE(x)
#endif

static const double numerical_precision = 0.05; // 95% of pixels should have exact values

TEST(Photo_SeamlessClone_normal, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Normal_Cloning/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "destination1.png";
    string original_path3 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat destination = imread(original_path2, IMREAD_COLOR);
    Mat mask = imread(original_path3, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(destination.empty()) << "Could not load destination image " << original_path2;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path3;

    Mat result;
    Point p;
    p.x = destination.size().width/2;
    p.y = destination.size().height/2;
    seamlessClone(source, destination, mask, p, result, NORMAL_CLONE);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    SAVE(result);

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();

    mask = Scalar(0, 0, 0);
    seamlessClone(source, destination, mask, p, result, NORMAL_CLONE);

    reference = destination;
    errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_mixed, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Mixed_Cloning/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "destination1.png";
    string original_path3 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat destination = imread(original_path2, IMREAD_COLOR);
    Mat mask = imread(original_path3, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(destination.empty()) << "Could not load destination image " << original_path2;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path3;

    Mat result;
    Point p;
    p.x = destination.size().width/2;
    p.y = destination.size().height/2;
    seamlessClone(source, destination, mask, p, result, MIXED_CLONE);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_featureExchange, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Monochrome_Transfer/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "destination1.png";
    string original_path3 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat destination = imread(original_path2, IMREAD_COLOR);
    Mat mask = imread(original_path3, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(destination.empty()) << "Could not load destination image " << original_path2;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path3;

    Mat result;
    Point p;
    p.x = destination.size().width/2;
    p.y = destination.size().height/2;
    seamlessClone(source, destination, mask, p, result, MONOCHROME_TRANSFER);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_colorChange, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/color_change/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat mask = imread(original_path2, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path2;

    Mat result;
    colorChange(source, mask, result, 1.5, .5, .5);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_illuminationChange, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Illumination_Change/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat mask = imread(original_path2, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path2;

    Mat result;
    illuminationChange(source, mask, result, 0.2f, 0.4f);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_textureFlattening, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Texture_Flattening/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat mask = imread(original_path2, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path2;

    Mat result;
    textureFlattening(source, mask, result, 30, 45, 3);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

}} // namespace
