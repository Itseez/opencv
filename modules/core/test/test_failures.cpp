#include "opencv2/core.hpp"
#include "opencv2/ts.hpp"
#include "test_precomp.hpp"

int a_function()
{
    return cvtest::randInt(cv::theRNG());
}

namespace opencv_test { namespace {


TEST(Dummy, AlwaysFails) {
    EXPECT_EQ(-1, a_function());
}

}}
