// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{

#if defined(HAVE_PNG) || defined(HAVE_SPNG)

using namespace perf;

typedef perf::TestBaseWithParam<std::string> PNG;

PERF_TEST(PNG, decode)
{
    String filename = getDataPath("perf/2560x1600.png");

    FILE *f = fopen(filename.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<uchar> file_buf((size_t)len);
    EXPECT_EQ(len, (long)fread(&file_buf[0], 1, (size_t)len, f));
    fclose(f); f = NULL;

    TEST_CYCLE() imdecode(file_buf, IMREAD_UNCHANGED);

    SANITY_CHECK_NOTHING();
}

PERF_TEST(PNG, decode_rgb)
{
    String filename = getDataPath("perf/2560x1600.png");

    FILE *f = fopen(filename.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<uchar> file_buf((size_t)len);
    EXPECT_EQ(len, (long)fread(&file_buf[0], 1, (size_t)len, f));
    fclose(f); f = NULL;

    TEST_CYCLE() imdecode(file_buf, IMREAD_COLOR_RGB);

    SANITY_CHECK_NOTHING();
}

PERF_TEST(PNG, encode)
{
    String filename = getDataPath("perf/2560x1600.png");
    cv::Mat src = imread(filename);

    vector<uchar> buf;
    TEST_CYCLE() imencode(".png", src, buf);

    SANITY_CHECK_NOTHING();
}

PERF_TEST(PNG, encodemulti)
{
    String filename = getDataPath("perf/2560x1600.png");
    Mat src = imread(filename);
    vector<Mat> vec;
    vec.push_back(src);
    vec.push_back(src.clone());

    vector<uchar> buf;
    TEST_CYCLE() imencode(".png", vec, buf);

    SANITY_CHECK_NOTHING();
}
#endif // HAVE_PNG

} // namespace
