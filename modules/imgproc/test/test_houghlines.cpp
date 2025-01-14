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
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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

//#define GENERATE_DATA // generate data in debug mode via CPU code path (without IPP / OpenCL and other accelerators)

namespace opencv_test { namespace {

template<typename T>
struct SimilarWith
{
    T value;
    float theta_eps;
    float rho_eps;
    SimilarWith(T val, float e, float r_e): value(val), theta_eps(e), rho_eps(r_e) { }
    bool operator()(const T& other);
};

template<>
bool SimilarWith<Vec2f>::operator()(const Vec2f& other)
{
    return std::abs(other[0] - value[0]) < rho_eps && std::abs(other[1] - value[1]) < theta_eps;
}

template<>
bool SimilarWith<Vec3f>::operator()(const Vec3f& other)
{
    return std::abs(other[0] - value[0]) < rho_eps && std::abs(other[1] - value[1]) < theta_eps;
}

template<>
bool SimilarWith<Vec4i>::operator()(const Vec4i& other)
{
    return cv::norm(value, other) < theta_eps;
}

template <typename T>
int countMatIntersection(const Mat& expect, const Mat& actual, float eps, float rho_eps)
{
    int count = 0;
    if (!expect.empty() && !actual.empty())
    {
        for (MatConstIterator_<T> it=expect.begin<T>(); it!=expect.end<T>(); it++)
        {
            MatConstIterator_<T> f = std::find_if(actual.begin<T>(), actual.end<T>(), SimilarWith<T>(*it, eps, rho_eps));
            if (f != actual.end<T>())
                count++;
        }
    }
    return count;
}

String getTestCaseName(String filename)
{
    string temp(filename);
    size_t pos = temp.find_first_of("\\/.");
    while ( pos != string::npos ) {
       temp.replace( pos, 1, "_" );
       pos = temp.find_first_of("\\/.");
    }
    return String(temp);
}

class BaseHoughLineTest
{
public:
    enum {STANDART = 0, PROBABILISTIC};
protected:
    template<typename LinesType, typename LineType>
    void run_test(int type, const char* xml_name);

    string picture_name;
    double rhoStep;
    double thetaStep;
    int threshold;
    int minLineLength;
    int maxGap;
};

typedef tuple<string, double, double, int> Image_RhoStep_ThetaStep_Threshold_t;
class StandartHoughLinesTest : public BaseHoughLineTest, public testing::TestWithParam<Image_RhoStep_ThetaStep_Threshold_t>
{
public:
    StandartHoughLinesTest()
    {
        picture_name = get<0>(GetParam());
        rhoStep = get<1>(GetParam());
        thetaStep = get<2>(GetParam());
        threshold = get<3>(GetParam());
        minLineLength = 0;
        maxGap = 0;
    }
};

typedef tuple<string, double, double, int, int, int> Image_RhoStep_ThetaStep_Threshold_MinLine_MaxGap_t;
class ProbabilisticHoughLinesTest : public BaseHoughLineTest, public testing::TestWithParam<Image_RhoStep_ThetaStep_Threshold_MinLine_MaxGap_t>
{
public:
    ProbabilisticHoughLinesTest()
    {
        picture_name = get<0>(GetParam());
        rhoStep = get<1>(GetParam());
        thetaStep = get<2>(GetParam());
        threshold = get<3>(GetParam());
        minLineLength = get<4>(GetParam());
        maxGap = get<5>(GetParam());
    }
};

typedef tuple<double, double, double, double> HoughLinesPointSetInput_t;
class HoughLinesPointSetTest : public testing::TestWithParam<HoughLinesPointSetInput_t>
{
protected:
    void run_test();
    double Rho;
    double Theta;
    double rhoMin, rhoMax, rhoStep;
    double thetaMin, thetaMax, thetaStep;
public:
    HoughLinesPointSetTest()
    {
        rhoMin = get<0>(GetParam());
        rhoMax = get<1>(GetParam());
        rhoStep = (rhoMax - rhoMin) / 360.0f;
        thetaMin = get<2>(GetParam());
        thetaMax = get<3>(GetParam());
        thetaStep = CV_PI / 180.0f;
        Rho = 320.00000;
        Theta = 1.04719;
    }
};

template<typename LinesType, typename LineType>
void BaseHoughLineTest::run_test(int type, const char* xml_name)
{
    string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Invalid test image: " << filename;

    string xml = string(cvtest::TS::ptr()->get_data_path()) + "imgproc/" + xml_name;

    Mat dst;
    Canny(src, dst, 100, 150, 3);
    ASSERT_FALSE(dst.empty()) << "Failed Canny edge detector";

    LinesType lines;
    if (type == STANDART)
        HoughLines(dst, lines, rhoStep, thetaStep, threshold, 0, 0);
    else if (type == PROBABILISTIC)
        HoughLinesP(dst, lines, rhoStep, thetaStep, threshold, minLineLength, maxGap);

    String test_case_name = format("lines_%s_%.0f_%.2f_%d_%d_%d", picture_name.c_str(), rhoStep, thetaStep,
                                   threshold, minLineLength, maxGap);
    test_case_name = getTestCaseName(test_case_name);

#ifdef GENERATE_DATA
    {
        FileStorage fs(xml, FileStorage::READ);
        ASSERT_TRUE(!fs.isOpened() || fs[test_case_name].empty());
    }
    {
        FileStorage fs(xml, FileStorage::APPEND);
        EXPECT_TRUE(fs.isOpened()) << "Cannot open sanity data file: " << xml;
        fs << test_case_name << Mat(lines);
    }
#else
    FileStorage fs(xml, FileStorage::READ);
    FileNode node = fs[test_case_name];
    ASSERT_FALSE(node.empty()) << "Missing test data: " << test_case_name << std::endl << "XML: " << xml;

    Mat exp_lines_;
    read(fs[test_case_name], exp_lines_, Mat());
    fs.release();
    LinesType exp_lines;
    exp_lines_.copyTo(exp_lines);

    int count = -1;
    if (type == STANDART)
        count = countMatIntersection<LineType>(Mat(exp_lines), Mat(lines), (float) thetaStep + FLT_EPSILON, (float) rhoStep + FLT_EPSILON);
    else if (type == PROBABILISTIC)
        count = countMatIntersection<LineType>(Mat(exp_lines), Mat(lines), 1e-4f, 0.f);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    EXPECT_LE(std::abs((double)count - Mat(exp_lines).total()), Mat(exp_lines).total() * 0.25)
        << "count=" << count << " expected=" << Mat(exp_lines).total();
#else
    EXPECT_EQ(count, (int)Mat(exp_lines).total());
#endif
#endif // GENERATE_DATA
}

void HoughLinesPointSetTest::run_test(void)
{
    Mat lines_f, lines_i;
    vector<Point2f> pointf;
    vector<Point2i> pointi;
    vector<Vec3d> line_polar_f, line_polar_i;
    const float Points[20][2] = {
    { 0.0f,   369.0f }, { 10.0f,  364.0f }, { 20.0f,  358.0f }, { 30.0f,  352.0f },
    { 40.0f,  346.0f }, { 50.0f,  341.0f }, { 60.0f,  335.0f }, { 70.0f,  329.0f },
    { 80.0f,  323.0f }, { 90.0f,  318.0f }, { 100.0f, 312.0f }, { 110.0f, 306.0f },
    { 120.0f, 300.0f }, { 130.0f, 295.0f }, { 140.0f, 289.0f }, { 150.0f, 284.0f },
    { 160.0f, 277.0f }, { 170.0f, 271.0f }, { 180.0f, 266.0f }, { 190.0f, 260.0f }
    };

    // Float
    for (int i = 0; i < 20; i++)
    {
        pointf.push_back(Point2f(Points[i][0],Points[i][1]));
    }

    HoughLinesPointSet(pointf, lines_f, 20, 1,
                       rhoMin, rhoMax, rhoStep,
                       thetaMin, thetaMax, thetaStep);

    lines_f.copyTo( line_polar_f );

    // Integer
    for( int i = 0; i < 20; i++ )
    {
        pointi.push_back( Point2i( (int)Points[i][0], (int)Points[i][1] ) );
    }

    HoughLinesPointSet( pointi, lines_i, 20, 1,
                        rhoMin, rhoMax, rhoStep,
                        thetaMin, thetaMax, thetaStep );

    lines_i.copyTo( line_polar_i );

    EXPECT_EQ((int)(line_polar_f.at(0).val[1] * 100000.0f), (int)(Rho * 100000.0f));
    EXPECT_EQ((int)(line_polar_f.at(0).val[2] * 100000.0f), (int)(Theta * 100000.0f));
    EXPECT_EQ((int)(line_polar_i.at(0).val[1] * 100000.0f), (int)(Rho * 100000.0f));
    EXPECT_EQ((int)(line_polar_i.at(0).val[2] * 100000.0f), (int)(Theta * 100000.0f));
}

TEST_P(StandartHoughLinesTest, regression)
{
    run_test<Mat, Vec2f>(STANDART, "HoughLines.xml");
}

TEST_P(ProbabilisticHoughLinesTest, regression)
{
    run_test<Mat, Vec4i>(PROBABILISTIC, "HoughLinesP.xml");
}

TEST_P(StandartHoughLinesTest, regression_Vec2f)
{
    run_test<std::vector<Vec2f>, Vec2f>(STANDART, "HoughLines2f.xml");
}

TEST_P(StandartHoughLinesTest, regression_Vec3f)
{
    run_test<std::vector<Vec3f>, Vec3f>(STANDART, "HoughLines3f.xml");
}

TEST_P(HoughLinesPointSetTest, regression)
{
    run_test();
}

TEST(HoughLinesPointSet, regression_21029)
{
    std::vector<Point2f> points;
    points.push_back(Point2f(100, 100));
    points.push_back(Point2f(1000, 1000));
    points.push_back(Point2f(10000, 10000));
    points.push_back(Point2f(100000, 100000));

    double rhoMin = 0;
    double rhoMax = 10;
    double rhoStep = 0.1;

    double thetaMin = 85 * CV_PI / 180.0;
    double thetaMax = 95 * CV_PI / 180.0;
    double thetaStep = 1 * CV_PI / 180.0;

    int lines_max = 5;
    int threshold = 100;

    Mat lines;

    HoughLinesPointSet(points, lines,
        lines_max, threshold,
        rhoMin, rhoMax, rhoStep,
        thetaMin, thetaMax, thetaStep
    );

    EXPECT_TRUE(lines.empty());
}

TEST(HoughLines, regression_21983)
{
    Mat img(200, 200, CV_8UC1, Scalar(0));
    line(img, Point(0, 100), Point(100, 100), Scalar(255));
    std::vector<Vec2f> lines;
    HoughLines(img, lines, 1, CV_PI/180, 90, 0, 0, 0.001, 1.58);
    ASSERT_EQ(lines.size(), 1U);
    EXPECT_EQ(lines[0][0], 100);
    EXPECT_NEAR(lines[0][1], 1.57179642, 1e-4);
}

TEST(WeightedHoughLines, horizontal)
{
    Mat img(25, 25, CV_8UC1, Scalar(0));
    // draw lines. from top to bottom, stronger to weaker.
    line(img, Point(0, 6), Point(25, 6), Scalar(240));
    line(img, Point(0, 12), Point(25, 12), Scalar(255));
    line(img, Point(0, 18), Point(25, 18), Scalar(220));

    // detect lines
    std::vector<Vec2f> lines;
    int threshold{220*25-1};
    bool use_edgeval{true};
    HoughLines(img, lines, 1, CV_PI/180, threshold, 0, 0, 0.0, CV_PI, use_edgeval);

    // check results
    ASSERT_EQ(3U, lines.size());
    // detected lines is assumed sorted from stronger to weaker.
    EXPECT_EQ(12, lines[0][0]);
    EXPECT_EQ(6, lines[1][0]);
    EXPECT_EQ(18, lines[2][0]);
    EXPECT_NEAR(CV_PI/2, lines[0][1], CV_PI/180 + 1e-6);
    EXPECT_NEAR(CV_PI/2, lines[1][1], CV_PI/180 + 1e-6);
    EXPECT_NEAR(CV_PI/2, lines[2][1], CV_PI/180 + 1e-6);
}

TEST(WeightedHoughLines, diagonal)
{
    Mat img(25, 25, CV_8UC1, Scalar(0));
    // draw lines.
    line(img, Point(0, 0), Point(25, 25), Scalar(128));
    line(img, Point(0, 25), Point(25, 0), Scalar(255));

    // detect lines
    std::vector<Vec2f> lines;
    int threshold{128*25-1};
    bool use_edgeval{true};
    HoughLines(img, lines, 1, CV_PI/180, threshold, 0, 0, 0.0, CV_PI, use_edgeval);

    // check results
    ASSERT_EQ(2U, lines.size());
    // detected lines is assumed sorted from stronger to weaker.
    EXPECT_EQ(18, lines[0][0]); // 25*sqrt(2)/2 = 17.67 ~ 18
    EXPECT_EQ(0, lines[1][0]);
    EXPECT_NEAR(CV_PI/4, lines[0][1], CV_PI/180 + 1e-6);
    EXPECT_NEAR(CV_PI*3/4, lines[1][1], CV_PI/180 + 1e-6);
}

INSTANTIATE_TEST_CASE_P( ImgProc, StandartHoughLinesTest, testing::Combine(testing::Values( "shared/pic5.png", "../stitching/a1.png" ),
                                                                           testing::Values( 1, 10 ),
                                                                           testing::Values( 0.05, 0.1 ),
                                                                           testing::Values( 80, 150 )
                                                                           ));

INSTANTIATE_TEST_CASE_P( ImgProc, ProbabilisticHoughLinesTest, testing::Combine(testing::Values( "shared/pic5.png", "shared/pic1.png" ),
                                                                                testing::Values( 5, 10 ),
                                                                                testing::Values( 0.05, 0.1 ),
                                                                                testing::Values( 75, 150 ),
                                                                                testing::Values( 0, 10 ),
                                                                                testing::Values( 0, 4 )
                                                                                ));

INSTANTIATE_TEST_CASE_P( Imgproc, HoughLinesPointSetTest, testing::Combine(testing::Values( 0.0f, 120.0f ),
                                                                           testing::Values( 360.0f, 480.0f ),
                                                                           testing::Values( 0.0f, (CV_PI / 18.0f) ),
                                                                           testing::Values( (CV_PI / 2.0f), (CV_PI * 5.0f / 12.0f) )
                                                                           ));

}} // namespace
