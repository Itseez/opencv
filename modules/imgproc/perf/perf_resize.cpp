// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<MatType, Size, Size> MatInfo_Size_Size_t;
typedef TestBaseWithParam<MatInfo_Size_Size_t> MatInfo_Size_Size;
typedef tuple<Size,Size> Size_Size_t;
typedef tuple<MatType, Size_Size_t> MatInfo_SizePair_t;
typedef TestBaseWithParam<MatInfo_SizePair_t> MatInfo_SizePair;

#define MATTYPE_NE_VALUES CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4,     \
                          CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, \
                          CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4

// For gradient-ish testing of the other matrix formats
template<typename T>
static void fillFPGradient(Mat& img)
{
    const int ch = img.channels();

    int r, c, i;
    for(r=0; r<img.rows; r++)
    {
        for(c=0; c<img.cols; c++)
        {
            T vals[] = {(T)r, (T)c, (T)(r*c), (T)(r*c/(r+c+1))};
            T *p = (T*)img.ptr(r, c);
            for(i=0; i<ch; i++) p[i] = (T)vals[i];
        }
    }
}

PERF_TEST_P(MatInfo_Size_Size, resizeUpLinear,
            testing::Values(
                MatInfo_Size_Size_t(CV_8UC1, szVGA, szqHD),
                MatInfo_Size_Size_t(CV_8UC2, szVGA, szqHD),
                MatInfo_Size_Size_t(CV_8UC3, szVGA, szqHD),
                MatInfo_Size_Size_t(CV_8UC4, szVGA, szqHD),
                MatInfo_Size_Size_t(CV_8UC1, szVGA, sz720p),
                MatInfo_Size_Size_t(CV_8UC2, szVGA, sz720p),
                MatInfo_Size_Size_t(CV_8UC3, szVGA, sz720p),
                MatInfo_Size_Size_t(CV_8UC4, szVGA, sz720p)
                )
            )
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    Size to = get<2>(GetParam());

    cv::Mat src(from, matType), dst(to, matType);
    cvtest::fillGradient(src);
    declare.in(src).out(dst);

    TEST_CYCLE_MULTIRUN(10) resize(src, dst, to, 0, 0, INTER_LINEAR_EXACT);

#ifdef __ANDROID__
    SANITY_CHECK(dst, 5);
#else
    SANITY_CHECK(dst, 1 + 1e-6);
#endif
}

PERF_TEST_P(MatInfo_SizePair, resizeUpLinearNonExact,
            testing::Combine
                (
                testing::Values( MATTYPE_NE_VALUES ),
                testing::Values( Size_Size_t(szVGA, szqHD), Size_Size_t(szVGA, sz720p) )
                )
             )
{
    int matType = get<0>(GetParam());
    Size_Size_t sizes = get<1>(GetParam());
    Size from = get<0>(sizes);
    Size to = get<1>(sizes);

    cv::Mat src(from, matType), dst(to, matType);
    switch(src.depth())
    {
        case CV_8U: cvtest::fillGradient(src); break;
        case CV_16U: fillFPGradient<ushort>(src); break;
        case CV_32F: fillFPGradient<float>(src); break;
    }
    declare.in(src).out(dst);

    TEST_CYCLE_MULTIRUN(10) resize(src, dst, to, 0, 0, INTER_LINEAR);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MatInfo_Size_Size, resizeDownLinear,
            testing::Values(
                MatInfo_Size_Size_t(CV_8UC1, szVGA, szQVGA),
                MatInfo_Size_Size_t(CV_8UC2, szVGA, szQVGA),
                MatInfo_Size_Size_t(CV_8UC3, szVGA, szQVGA),
                MatInfo_Size_Size_t(CV_8UC4, szVGA, szQVGA),
                MatInfo_Size_Size_t(CV_8UC1, szqHD, szVGA),
                MatInfo_Size_Size_t(CV_8UC2, szqHD, szVGA),
                MatInfo_Size_Size_t(CV_8UC3, szqHD, szVGA),
                MatInfo_Size_Size_t(CV_8UC4, szqHD, szVGA),
                MatInfo_Size_Size_t(CV_8UC1, sz720p, Size(120 * sz720p.width / sz720p.height, 120)),//face detection min_face_size = 20%
                MatInfo_Size_Size_t(CV_8UC2, sz720p, Size(120 * sz720p.width / sz720p.height, 120)),//face detection min_face_size = 20%
                MatInfo_Size_Size_t(CV_8UC3, sz720p, Size(120 * sz720p.width / sz720p.height, 120)),//face detection min_face_size = 20%
                MatInfo_Size_Size_t(CV_8UC4, sz720p, Size(120 * sz720p.width / sz720p.height, 120)),//face detection min_face_size = 20%
                MatInfo_Size_Size_t(CV_8UC1, sz720p, szVGA),
                MatInfo_Size_Size_t(CV_8UC2, sz720p, szVGA),
                MatInfo_Size_Size_t(CV_8UC3, sz720p, szVGA),
                MatInfo_Size_Size_t(CV_8UC4, sz720p, szVGA),
                MatInfo_Size_Size_t(CV_8UC1, sz720p, szQVGA),
                MatInfo_Size_Size_t(CV_8UC2, sz720p, szQVGA),
                MatInfo_Size_Size_t(CV_8UC3, sz720p, szQVGA),
                MatInfo_Size_Size_t(CV_8UC4, sz720p, szQVGA)
                )
            )
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    Size to = get<2>(GetParam());

    cv::Mat src(from, matType), dst(to, matType);
    cvtest::fillGradient(src);
    declare.in(src).out(dst);

    TEST_CYCLE_MULTIRUN(10) resize(src, dst, to, 0, 0, INTER_LINEAR_EXACT);

#ifdef __ANDROID__
    SANITY_CHECK(dst, 5);
#else
    SANITY_CHECK(dst, 1 + 1e-6);
#endif
}

PERF_TEST_P(MatInfo_SizePair, resizeDownLinearNonExact,
            testing::Combine
                (
                testing::Values( MATTYPE_NE_VALUES ),
                testing::Values
                    (
                    Size_Size_t(szVGA, szQVGA),
                    Size_Size_t(szqHD, szVGA),
                    Size_Size_t(sz720p, Size(120 * sz720p.width / sz720p.height, 120)),
                    Size_Size_t(sz720p, szVGA),
                    Size_Size_t(sz720p, szQVGA)
                    )
                )
            )
{
    int matType = get<0>(GetParam());
    Size_Size_t sizes = get<1>(GetParam());
    Size from = get<0>(sizes);
    Size to = get<1>(sizes);

    cv::Mat src(from, matType), dst(to, matType);
    switch(src.depth())
    {
        case CV_8U: cvtest::fillGradient(src); break;
        case CV_16U: fillFPGradient<ushort>(src); break;
        case CV_32F: fillFPGradient<float>(src); break;
    }
    declare.in(src).out(dst);

    TEST_CYCLE_MULTIRUN(10) resize(src, dst, to, 0, 0, INTER_LINEAR);

    SANITY_CHECK_NOTHING();
}


typedef tuple<MatType, Size, int> MatInfo_Size_Scale_t;
typedef TestBaseWithParam<MatInfo_Size_Scale_t> MatInfo_Size_Scale;

PERF_TEST_P(MatInfo_Size_Scale, ResizeAreaFast,
            testing::Combine(
                testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4),
                testing::Values(szVGA, szqHD, sz720p, sz1080p),
                testing::Values(2)
                )
            )
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    int scale = get<2>(GetParam());

    from.width = (from.width/scale)*scale;
    from.height = (from.height/scale)*scale;

    cv::Mat src(from, matType);
    cv::Mat dst(from.height / scale, from.width / scale, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    int runs = 15;
    TEST_CYCLE_MULTIRUN(runs) resize(src, dst, dst.size(), 0, 0, INTER_AREA);

    //difference equal to 1 is allowed because of different possible rounding modes: round-to-nearest vs bankers' rounding
    SANITY_CHECK(dst, 1);
}


typedef TestBaseWithParam<tuple<MatType, Size, double> > MatInfo_Size_Scale_Area;

PERF_TEST_P(MatInfo_Size_Scale_Area, ResizeArea,
            testing::Combine(
                testing::Values(CV_8UC1, CV_8UC3, CV_8UC4),
                testing::Values(szVGA, szqHD, sz720p, sz1080p, sz2160p),
                testing::Values(0.1, 0.25, 0.81)
                )
            )
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    double scale = get<2>(GetParam());

    cv::Mat src(from, matType);

    Size to(cvRound(from.width * scale), cvRound(from.height * scale));
    cv::Mat dst(to, matType);

    declare.in(src, WARMUP_RNG).out(dst);
    declare.time(100);

    TEST_CYCLE() resize(src, dst, dst.size(), 0, 0, INTER_AREA);

    //difference equal to 1 is allowed because of different possible rounding modes: round-to-nearest vs bankers' rounding
    SANITY_CHECK(dst, 1);
}

typedef MatInfo_Size_Scale_Area MatInfo_Size_Scale_NN;

PERF_TEST_P(MatInfo_Size_Scale_NN, ResizeNN,
    testing::Combine(
        testing::Values(CV_8UC1, CV_8UC2, CV_8UC4),
        testing::Values(szVGA, szqHD, sz720p, sz1080p, sz2160p),
        testing::Values(2.4, 3.4, 1.3)
    )
)
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    double scale = get<2>(GetParam());

    cv::Mat src(from, matType);

    Size to(cvRound(from.width * scale), cvRound(from.height * scale));
    cv::Mat dst(to, matType);

    declare.in(src, WARMUP_RNG).out(dst);
    declare.time(100);

    TEST_CYCLE() resize(src, dst, dst.size(), 0, 0, INTER_NEAREST);

    EXPECT_GT(countNonZero(dst.reshape(1)), 0);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MatInfo_Size_Scale_NN, ResizeNNExact,
    testing::Combine(
        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4),
        testing::Values(sz720p, sz1080p),
        testing::Values(0.25, 0.5, 2.0)
    )
)
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    double scale = get<2>(GetParam());

    cv::Mat src(from, matType);

    Size to(cvRound(from.width * scale), cvRound(from.height * scale));
    cv::Mat dst(to, matType);

    declare.in(src, WARMUP_RNG).out(dst);
    declare.time(100);

    TEST_CYCLE() resize(src, dst, dst.size(), 0, 0, INTER_NEAREST_EXACT);

    EXPECT_GT(countNonZero(dst.reshape(1)), 0);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MatInfo_Size_Size, ResizeOnnxLinearAntialias, Combine(
    Values(CV_8UC1, CV_8UC3, CV_8UC4),
    Values(sz1440p),
    Values(szVGA, szqHD, sz720p, sz1080p)
))
{
    int matType = get<0>(GetParam());
    Size from = get<1>(GetParam());
    Size to = get<2>(GetParam());

    cv::Mat src(from, matType), dst(to, matType);
    declare.in(src).out(dst);
    declare.time(100);

    TEST_CYCLE() resizeOnnx(src, dst, to, Point2d(), INTER_LINEAR | INTER_ANTIALIAS);

    SANITY_CHECK_NOTHING();
}

}
