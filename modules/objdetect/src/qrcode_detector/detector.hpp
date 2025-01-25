// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <memory>

namespace cv {
    typedef struct BoxInfo
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } BoxInfo;

    struct DetectInfo
    {
        int class_id;
        float prob;
        int x, y, width, height;
        
        void printInfo()
        {
            printf("class %d, prob %.2f, x %d, y %d, width %d, height %d\n", class_id, prob, x, y, width, height);
        }
        
        bool operator<(const DetectInfo &other) const
        {
            return other.prob < prob;
        }
    };

    class Detector
    {
        public:
            Detector(){};
            ~Detector(){};
            int init(const std::string &config_path);
            bool detect(const Mat &image,std::vector<DetectInfo> &bboxes);
            void setReferenceSize(int reference_size_) {this->reference_size = reference_size_;}
            void setScoreThres(float score_thres_) {this->score_thres = score_thres_;}
            void setIouThres(float iou_thres_) {this->iou_thres = iou_thres_;}
    
        private:
            bool post_process_det(std::vector<Mat> outputs,float inputWidth,float inputHeight,std::vector<BoxInfo>& dets);
            bool pre_process_det(const Mat &image,Mat &out_blob);
            void multiclass_nms(std::vector<BoxInfo> &input_boxes, std::vector<BoxInfo> &output_boxes, float thr, int inputWidth, int inputHeight);
            void decode_infer(float *clsPred, float *disPred, int stride, std::vector<std::vector<BoxInfo>> &results, const std::vector<int> &outShapeCls, const std::vector<int> &outShapeDis, float scoreThres,float inputHeight,float inputWidth);
            void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH);
    
        private:
            std::shared_ptr<dnn::Net> qbar_detector;
            int reference_size;
            float score_thres, iou_thres;
    };
}  // namespace cv