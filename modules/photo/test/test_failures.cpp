#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Dummy, CausesCrash) {
    std::cerr << "Crashing now!!!" << std::endl;
    int* p = nullptr;
    *p = 42;
}

}}
