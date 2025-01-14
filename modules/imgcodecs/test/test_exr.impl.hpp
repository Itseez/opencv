// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

//#define GENERATE_DATA

namespace opencv_test { namespace {

size_t getFileSize(const string& filename)
{
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (ifs.is_open())
    {
        ifs.seekg(0, std::ios::end);
        return (size_t)ifs.tellg();
    }
    return 0;
}

TEST(Imgcodecs_EXR, readWrite_32FC1)
{ // Y channels
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1.exr";
    const string filenameOutput = cv::tempfile(".exr");
#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC1, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1,img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    // Check generated file size to ensure that it's compressed with proper options
    ASSERT_LE(396u, getFileSize(filenameOutput));       // OpenEXR 2
    ASSERT_LE(      getFileSize(filenameOutput), 440u); // OpenEXR 3.2+
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, readWrite_32FC3)
{ // RGB channels
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC3.exr";
    const string filenameOutput = cv::tempfile(".exr");
#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC3, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}


TEST(Imgcodecs_EXR, readWrite_32FC1_half)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1_half.exr";
    const string filenameOutput = cv::tempfile(".exr");

    std::vector<int> params;
    params.push_back(IMWRITE_EXR_TYPE);
    params.push_back(IMWRITE_EXR_TYPE_HALF);

#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC1, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img, params));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1,img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img, params));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, readWrite_32FC3_half)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC3_half.exr";
    const string filenameOutput = cv::tempfile(".exr");

    std::vector<int> params;
    params.push_back(IMWRITE_EXR_TYPE);
    params.push_back(IMWRITE_EXR_TYPE_HALF);

#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC3, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img, params));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img, params));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, readWrite_32FC1_PIZ)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1.exr";
    const string filenameOutput = cv::tempfile(".exr");


    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1, img.type());

    std::vector<int> params;
    params.push_back(IMWRITE_EXR_COMPRESSION);
    params.push_back(IMWRITE_EXR_COMPRESSION_PIZ);
    ASSERT_TRUE(cv::imwrite(filenameOutput, img, params));
    // Check generated file size to ensure that it's compressed with proper options
    ASSERT_EQ(849u, getFileSize(filenameOutput));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

// Note: YC to GRAYSCALE (IMREAD_GRAYSCALE | IMREAD_ANYDEPTH)
// outputs a black image,
// as does Y to RGB (IMREAD_COLOR | IMREAD_ANYDEPTH).
// This behavoir predates adding EXR alpha support issue
// 16115.

TEST(Imgcodecs_EXR, read_YA_ignore_alpha)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_YA.exr";

    const Mat img = cv::imread(filenameInput, IMREAD_GRAYSCALE | IMREAD_ANYDEPTH);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1, img.type());

    // Writing Y covered by test 32FC1
}

TEST(Imgcodecs_EXR, read_YA_unchanged)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_YA.exr";

    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC2, img.type());

    // Cannot test writing, 2 channel writing not suppported by loadsave
}

TEST(Imgcodecs_EXR, read_YC_changeDepth)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_YRYBY.exr";

    const Mat img = cv::imread(filenameInput, IMREAD_COLOR);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC3, img.type());

    const Mat img_rgb = cv::imread(filenameInput, IMREAD_COLOR_RGB);

    ASSERT_FALSE(img_rgb.empty());
    ASSERT_EQ(CV_8UC3, img_rgb.type());

    cvtColor(img_rgb, img_rgb, COLOR_RGB2BGR);

    // See https://github.com/opencv/opencv/issues/26705
    // If ALGO_HINT_ACCURATE is set, norm should be 0.
    // If ALGO_HINT_APPROX is set,   norm should be 1(or 0).
    EXPECT_LE(cvtest::norm(img, img_rgb, NORM_INF),
             (cv::getDefaultAlgorithmHint() == ALGO_HINT_ACCURATE)?0:1);

    // Cannot test writing, EXR encoder doesn't support 8U depth
}

TEST(Imgcodecs_EXR, readwrite_YCA_ignore_alpha)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_YRYBYA.exr";
    const string filenameOutput = cv::tempfile(".exr");

    const Mat img = cv::imread(filenameInput, IMREAD_COLOR | IMREAD_ANYDEPTH);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, read_YC_unchanged)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_YRYBY.exr";

    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    // Writing YC covered by test readwrite_YCA_ignore_alpha
}

TEST(Imgcodecs_EXR, readwrite_YCA_unchanged)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_YRYBYA.exr";
    const string filenameOutput = cv::tempfile(".exr");

    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC4, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, readwrite_RGBA_togreyscale)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_GeneratedRGBA.exr";
    const string filenameOutput = cv::tempfile(".exr");

    const Mat img = cv::imread(filenameInput, IMREAD_GRAYSCALE | IMREAD_ANYDEPTH);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, read_RGBA_ignore_alpha)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_GeneratedRGBA.exr";

    const Mat img = cv::imread(filenameInput, IMREAD_COLOR | IMREAD_ANYDEPTH);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    // Writing RGB covered by test 32FC3
}

TEST(Imgcodecs_EXR, read_RGBA_unchanged)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_GeneratedRGBA.exr";
    const string filenameOutput = cv::tempfile(".exr");

#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC4, Scalar(0.5, 0.1, 1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0, 1));
    img(Rect(10, 20, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img));
#endif

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC4, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

// See https://github.com/opencv/opencv/pull/26211
// ( related with https://github.com/opencv/opencv/issues/26207 )
TEST(Imgcodecs_EXR, imencode_regression_26207_extra)
{
    // CV_8U is not supported depth for EXR Encoder.
    const cv::Mat src(100, 100, CV_8UC1, cv::Scalar::all(0));
    std::vector<uchar> buf;
    bool ret = false;
    EXPECT_ANY_THROW(ret = imencode(".exr", src, buf));
    EXPECT_FALSE(ret);
}
TEST(Imgcodecs_EXR, imwrite_regression_26207_extra)
{
    // CV_8U is not supported depth for EXR Encoder.
    const cv::Mat src(100, 100, CV_8UC1, cv::Scalar::all(0));
    const string filename = cv::tempfile(".exr");
    bool ret = false;
    EXPECT_ANY_THROW(ret = imwrite(filename, src));
    EXPECT_FALSE(ret);
    remove(filename.c_str());
}


}} // namespace
