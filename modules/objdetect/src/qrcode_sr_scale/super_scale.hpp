// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __SCALE_SUPER_SCALE_HPP_
#define __SCALE_SUPER_SCALE_HPP_

#include <stdio.h>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;

namespace cv {
constexpr static float MAX_SCALE = 4.0f;

class SuperScale {
public:
    SuperScale(){};
    ~SuperScale(){};
    int init(const std::string &config_path);
    std::vector<float> getScaleList(const int width, const int height);
    void processImageScale(const Mat &src, Mat &dst, float scale, bool use_sr, int sr_max_size = 160);

private:
    std::shared_ptr<dnn::Net> qbar_sr;
    int superResolutionScale(const Mat &src, Mat &dst);
};
}  // namesapce cv
#endif  // __SCALE_SUPER_SCALE_HPP_