/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "tracklet.hpp"

#include <sstream>
#include <memory>

namespace vas {
namespace ot {

Tracklet::Tracklet()
    : id(0), label(-1), association_idx(kNoMatchDetection), status(ST_DEAD), age(0), confidence(0.f),
      occlusion_ratio(0.f), association_delta_t(0.f), association_fail_count(0),
      rgb_features_(std::make_shared<std::deque<cv::Mat>>()) {
}

Tracklet::~Tracklet() {
}

void Tracklet::ClearTrajectory() {
    trajectory.clear();
    trajectory_filtered.clear();
}

void Tracklet::InitTrajectory(const cv::Rect2f &bounding_box) {
    trajectory.push_back(bounding_box);
    trajectory_filtered.push_back(bounding_box);
}

void Tracklet::AddUpdatedTrajectory(const cv::Rect2f &bounding_box, const cv::Rect2f &corrected_box) {
    trajectory.push_back(bounding_box);
    trajectory_filtered.push_back(corrected_box);
}

void Tracklet::UpdateLatestTrajectory(const cv::Rect2f &bounding_box, const cv::Rect2f &corrected_box) {
    trajectory.back() = bounding_box;
    trajectory_filtered.back() = corrected_box;
}

void Tracklet::RenewTrajectory(const cv::Rect2f &bounding_box) {
    ClearTrajectory();
    trajectory.push_back(bounding_box);
    trajectory_filtered.push_back(bounding_box);
}

std::deque<cv::Mat> *Tracklet::GetRgbFeatures() {
    return rgb_features_.get(); // Return the raw pointer from the shared_ptr
}

void Tracklet::AddRgbFeature(const cv::Mat &feature) {
    rgb_features_->push_back(feature);
}

std::string Tracklet::Serialize() const {
#ifdef DUMP_OTAV
    DEFINE_STRING_VAR(s_id, id);
    DEFINE_STRING_VAR(s_label, label);
    DEFINE_STRING_VAR(s_association_idx, association_idx);
    DEFINE_STRING_VAR(s_status, static_cast<int>(status));
    DEFINE_STRING_VAR(s_age, age);
    DEFINE_STRING_VAR(s_confidence, ROUND_F(confidence, 100.0));
    DEFINE_STRING_VAR(s_occlusion_ratio, ROUND_F(occlusion_ratio, 100.0));
    DEFINE_STRING_VAR(s_association_delta_t, association_delta_t);
    DEFINE_STRING_VAR(s_association_fail_count, association_fail_count);
    DEFINE_STRING_VAR(t_x, ROUND_F(trajectory.back().x, 10.0));
    DEFINE_STRING_VAR(t_y, ROUND_F(trajectory.back().y, 10.0));
    DEFINE_STRING_VAR(t_w, ROUND_F(trajectory.back().width, 10.0));
    DEFINE_STRING_VAR(t_h, ROUND_F(trajectory.back().height, 10.0));
    DEFINE_STRING_VAR(tf_x, ROUND_F(trajectory_filtered.back().x, 10.0));
    DEFINE_STRING_VAR(tf_y, ROUND_F(trajectory_filtered.back().y, 10.0));
    DEFINE_STRING_VAR(tf_w, ROUND_F(trajectory_filtered.back().width, 10.0));
    DEFINE_STRING_VAR(tf_h, ROUND_F(trajectory_filtered.back().height, 10.0));
    DEFINE_STRING_VAR(p_x, ROUND_F(predicted.x, 10.0));
    DEFINE_STRING_VAR(p_y, ROUND_F(predicted.y, 10.0));
    DEFINE_STRING_VAR(p_w, ROUND_F(predicted.width, 10.0));
    DEFINE_STRING_VAR(p_h, ROUND_F(predicted.height, 10.0));
    std::string formatted_msg = "meta:\"" + s_id + "," + s_label + "," + s_association_idx + "," + s_status + "," +
                                s_age + "," + s_confidence + "," + s_occlusion_ratio + "," + s_association_delta_t +
                                "," + s_association_fail_count + "\", roi:\"" + t_x + "," + t_y + "," + t_w + "," +
                                t_h + "\", roif:\"" + tf_x + "," + tf_y + "," + tf_w + "," + tf_h + "\", roip:\"" +
                                p_x + "," + p_y + "," + p_w + "," + p_h + "\"";

    std::string free_msg;
    if (otav_msg.size() > 0) {
        free_msg = ", msg: [";
        for (auto line : otav_msg) {
            if (line.size() > 0)
                free_msg += "\n\"" + line + "\",";
        }
        free_msg += "]";
        otav_msg.clear();
    }
    return formatted_msg + free_msg;
#else
    return "";
#endif
}

ZeroTermImagelessTracklet::ZeroTermImagelessTracklet() : Tracklet(), birth_count(1) {
}

ZeroTermImagelessTracklet::~ZeroTermImagelessTracklet() {
}

void ZeroTermImagelessTracklet::RenewTrajectory(const cv::Rect2f &bounding_box) {
    float velo_x = bounding_box.x - trajectory.back().x;
    float velo_y = bounding_box.y - trajectory.back().y;
    cv::Rect rect_predict(int(bounding_box.x + velo_x / 3), int(bounding_box.y + velo_y / 3),
                          int(bounding_box.width), int(bounding_box.height));

    ClearTrajectory();
    kalman_filter.reset(new KalmanFilterNoOpencv(bounding_box));
    kalman_filter->Predict();
    kalman_filter->Correct(rect_predict);

    trajectory.push_back(bounding_box);
    trajectory_filtered.push_back(bounding_box);
}

ShortTermImagelessTracklet::ShortTermImagelessTracklet() : Tracklet() {
}

ShortTermImagelessTracklet::~ShortTermImagelessTracklet() {
}

void ShortTermImagelessTracklet::RenewTrajectory(const cv::Rect2f &bounding_box) {
    float velo_x = bounding_box.x - trajectory.back().x;
    float velo_y = bounding_box.y - trajectory.back().y;
    cv::Rect rect_predict(int(bounding_box.x + velo_x / 3), int(bounding_box.y + velo_y / 3),
                          int(bounding_box.width), int(bounding_box.height));

    ClearTrajectory();
    kalman_filter.reset(new KalmanFilterNoOpencv(bounding_box));
    kalman_filter->Predict();
    kalman_filter->Correct(rect_predict);

    trajectory.push_back(bounding_box);
    trajectory_filtered.push_back(bounding_box);
}

}; // namespace ot
}; // namespace vas
