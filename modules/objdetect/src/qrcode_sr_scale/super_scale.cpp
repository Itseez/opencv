// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "super_scale.hpp"
#include <opencv2/dnn.hpp>
#include <iostream>

#define CLIP(x, x1, x2) max(x1, min(x, x2))
namespace cv {
int SuperScale::init(const std::string &sr_path) {
    dnn::Net network = dnn::readNetFromONNX(sr_path);
    this->qbar_sr = std::make_shared<dnn::Net>(network);

    return 0;
}

std::vector<float> SuperScale::getScaleList(const int width, const int height) {
    float min_side = min(width, height);
    if (min_side <= 450.f) {
        return {1.0f, 300.0f / min_side, 2.0f};
    }
    else
        return {1.0f, 450.f / min_side};
}

void SuperScale::processImageScale(const Mat &src, Mat &dst, float scale, bool use_sr,
                                  int sr_max_size) {
    scale = min(scale, MAX_SCALE);
    if (scale > .0 && scale < 1.0)
    {  // down sample
        resize(src, dst, Size(), scale, scale, INTER_AREA);
    }
    else if (scale >= 1.0 && scale < 2.0)
    {
        resize(src, dst, Size(), scale, scale, INTER_CUBIC);
    }
    else if (scale >= 2.0)
    {
        int width = src.cols;
        int height = src.rows;
        if (use_sr && (int) sqrt(width * height * 1.0) < sr_max_size && !qbar_sr->empty())
        {
            superResolutionScale(src, dst);
            if (scale > 2.0)
            {
                processImageScale(dst, dst, scale / 2.0f, use_sr);
            }
        }
        else
        { resize(src, dst, Size(), scale, scale, INTER_CUBIC); }
    }
}

int SuperScale::superResolutionScale(const Mat &src, Mat &dst) {
   
    Mat blob;
    dnn::blobFromImage(src, blob, 1.0, Size(src.cols, src.rows), {0.0f}, false, false);

    qbar_sr->setInput(blob);
    auto prob = qbar_sr->forward();

    dst = Mat(prob.size[2], prob.size[3], CV_32F, prob.ptr<float>());
    dst.convertTo(dst, CV_8UC1);
    
    return 0;
}
}  // namesapce cv