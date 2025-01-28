#include "test_precomp.hpp"
#include <unistd.h>

namespace opencv_test { namespace {

int a_function()
{
    return cvtest::randInt(cv::theRNG());
}

TEST(Dummy, Hangs) {
    while (a_function() != 0)
    {
        sleep(30);
        printf("Hang...\n");
    }
}

}}
