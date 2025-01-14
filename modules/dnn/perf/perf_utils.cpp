// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

using Utils_blobFromImage = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImage, HWC_TO_NCHW) {
    std::vector<int> input_shape = GetParam();

    Mat input(input_shape, CV_32FC3);
    randu(input, -10.0f, 10.f);

    TEST_CYCLE() {
        Mat blob = blobFromImage(input);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImage,
    Values(std::vector<int>{  32,   32},
           std::vector<int>{  64,   64},
           std::vector<int>{ 128,  128},
           std::vector<int>{ 256,  256},
           std::vector<int>{ 512,  512},
           std::vector<int>{1024, 1024},
           std::vector<int>{2048, 2048})
);

using Utils_blobFromImages = TestBaseWithParam<std::vector<int>>;
PERF_TEST_P_(Utils_blobFromImages, HWC_TO_NCHW) {
    std::vector<int> input_shape = GetParam();
    int batch = input_shape.front();
    std::vector<int> input_shape_no_batch(input_shape.begin()+1, input_shape.end());

    std::vector<Mat> inputs;
    for (int i = 0; i < batch; i++) {
        Mat input(input_shape_no_batch, CV_32FC3);
        randu(input, -10.0f, 10.f);
        inputs.push_back(input);
    }

    TEST_CYCLE() {
        Mat blobs = blobFromImages(inputs);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Utils_blobFromImages,
    Values(std::vector<int>{16,   32,   32},
           std::vector<int>{16,   64,   64},
           std::vector<int>{16,  128,  128},
           std::vector<int>{16,  256,  256},
           std::vector<int>{16,  512,  512},
           std::vector<int>{16, 1024, 1024},
           std::vector<int>{16, 2048, 2048})
);

}
