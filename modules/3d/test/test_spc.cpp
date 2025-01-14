// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Zhangjie Chen <zhangjiec01@gmail.com>
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(SpC, FunctionTest) {
    auto folder = cvtest::TS::ptr()->get_data_path();
    String test_file_path = folder + "pointcloudio/two_cubes.obj";
    std::vector<cv::Point3f> vertices;
    std::vector<std::vector<int32_t>> indices;
    std::vector<int> results;

    cv::loadMesh(test_file_path, vertices, indices);
    // init class instance
    cv::SpectralCluster cluster(0.1f, 0.1f);

    cluster.cluster(vertices, indices, 2, results);

    std::vector<int> truth_value = {0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0};

    CV_Assert(results == truth_value);
}
}
}
