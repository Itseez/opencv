#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

PERF_TEST_P(Size_MatType, Mat_Eye,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat diagonalMatrix(size.height, size.width, type);

    declare.out(diagonalMatrix);

    int runs = (size.width <= 640) ? 15 : 5;
    TEST_CYCLE_MULTIRUN(runs)
    {
        diagonalMatrix = Mat::eye(size, type);
    }

    SANITY_CHECK(diagonalMatrix, 1);
}

PERF_TEST_P(Size_MatType, Mat_Zeros,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES, CV_32FC3))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat zeroMatrix(size.height, size.width, type);

    declare.out(zeroMatrix);

    int runs = (size.width <= 640) ? 15 : 5;
    TEST_CYCLE_MULTIRUN(runs)
    {
        zeroMatrix = Mat::zeros(size, type);
    }

    SANITY_CHECK(zeroMatrix, 1);
}

PERF_TEST_P(Size_MatType, Mat_Clone,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat source(size.height, size.width, type);
    Mat destination(size.height, size.width, type);

    declare.in(source, WARMUP_RNG).out(destination);

    TEST_CYCLE()
    {
        Mat tmp = source.clone();
        CV_UNUSED(tmp);
    }
    destination = source.clone();

    SANITY_CHECK(destination, 1);
}

PERF_TEST_P(Size_MatType, Mat_Clone_Roi,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());

    unsigned int width = size.width;
    unsigned int height = size.height;
    Mat source(height, width, type);
    Mat destination(size.height/2, size.width/2, type);

    declare.in(source, WARMUP_RNG).out(destination);

    Mat roi(source, Rect(width/4, height/4, 3*width/4, 3*height/4));

    TEST_CYCLE()
    {
        Mat tmp = roi.clone();
        CV_UNUSED(tmp);
    }
    destination = roi.clone();

    SANITY_CHECK(destination, 1);
}

PERF_TEST_P(Size_MatType, Mat_CopyToWithMask,
            testing::Combine(testing::Values(::perf::sz1080p, ::perf::szODD),
                             testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_16UC1, CV_32SC1, CV_32FC4))
            )
{
    const Size_MatType_t params = GetParam();
    const Size size = get<0>(params);
    const int type = get<1>(params);

    Mat src(size, type), dst(size, type), mask(size, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG).out(dst);

    TEST_CYCLE()
    {
        src.copyTo(dst, mask);
    }

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType, Mat_SetToWithMask,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(CV_8UC1, CV_8UC2))
            )
{
    const Size_MatType_t params = GetParam();
    const Size size = get<0>(params);
    const int type = get<1>(params);
    const Scalar sc = Scalar::all(27);

    Mat src(size, type), mask(size, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG).out(src);

    TEST_CYCLE()
    {
        src.setTo(sc, mask);
    }

    SANITY_CHECK(src);
}

///////////// Transform ////////////////////////

PERF_TEST_P(Size_MatType, Mat_Transform,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(CV_8UC3, CV_8SC3, CV_16UC3, CV_16SC3, CV_32SC3, CV_32FC3, CV_64FC3))
            )
{
    const Size_MatType_t params = GetParam();
    const Size srcSize0 = get<0>(params);
    const Size srcSize = Size(1, srcSize0.width*srcSize0.height);
    const int type = get<1>(params);
    const float transform[] = { 0.5f,           0.f, 0.86602540378f, 128,
                                0.f,            1.f, 0.f,            -64,
                                0.86602540378f, 0.f, 0.5f,            32,};
    Mat mtx(Size(4, 3), CV_32FC1, (void*)transform);

    Mat src(srcSize, type), dst(srcSize, type);
    randu(src, 0, 30);
    declare.in(src).out(dst);

    TEST_CYCLE()
    {
        cv::transform(src, dst, mtx);
    }

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType, Mat_Transform_Diagonal,
            testing::Combine(testing::Values(szVGA, sz720p, sz1080p),
                             testing::Values(CV_8UC3, CV_8SC3, CV_16UC3, CV_16SC3, CV_32SC3, CV_32FC3, CV_64FC3))
            )
{
    const Size_MatType_t params = GetParam();
    const Size srcSize0 = get<0>(params);
    const Size srcSize = Size(1, srcSize0.width*srcSize0.height);
    const int type = get<1>(params);
    const float transform[] = { 0.5f,            0.f,           0.f, 128,
                                 0.f, 0.86602540378f,           0.f, -64,
                                 0.f,            0.f, 0.4330127019f,  32, };
    Mat mtx(Size(4, 3), CV_32FC1, (void*)transform);

    Mat src(srcSize, type), dst(srcSize, type);
    randu(src, 0, 30);
    declare.in(src).out(dst);

    TEST_CYCLE()
    {
        cv::transform(src, dst, mtx);
    }

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

} // namespace
