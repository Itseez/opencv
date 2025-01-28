/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_TRACKET_HPP
#define VAS_OT_TRACKET_HPP

#include "kalman_filter/kalman_filter_no_opencv.hpp"

#include <vas/common.hpp>

#include <cstdint>
#include <deque>
#include <memory>

namespace vas {
namespace ot {

const int32_t kNoMatchDetection = -1;

enum Status {
    ST_DEAD = -1,   // dead
    ST_NEW = 0,     // new
    ST_TRACKED = 1, // tracked
    ST_LOST = 2     // lost but still alive (in the detection phase if it configured)
};

struct Detection {
    cv::Rect2f rect;
    int32_t class_label = -1;
    int32_t index = -1;
};

class Tracklet {
  public:
    Tracklet();
    virtual ~Tracklet();

  public:
    void ClearTrajectory();
    void InitTrajectory(const cv::Rect2f &bounding_box);
    void AddUpdatedTrajectory(const cv::Rect2f &bounding_box, const cv::Rect2f &corrected_box);
    void UpdateLatestTrajectory(const cv::Rect2f &bounding_box, const cv::Rect2f &corrected_box);
    virtual void RenewTrajectory(const cv::Rect2f &bounding_box);

    virtual std::deque<cv::Mat> *GetRgbFeatures();
    void AddRgbFeature(const cv::Mat &feature);
    virtual std::string Serialize() const; // Returns key:value with comma separated format

  public:
    int32_t id; // If hasnot been assigned : -1 to 0
    int32_t label;
    int32_t association_idx;
    Status status;
    int32_t age;
    float confidence;

    float occlusion_ratio;
    float association_delta_t;
    int32_t association_fail_count;

    std::deque<cv::Rect2f> trajectory;
    std::deque<cv::Rect2f> trajectory_filtered;
    cv::Rect2f predicted;                      // Result from Kalman prediction. It is for debugging (OTAV)
    mutable std::vector<std::string> otav_msg; // Messages for OTAV

  private:
    std::shared_ptr<std::deque<cv::Mat>> rgb_features_;
};

class ZeroTermImagelessTracklet : public Tracklet {
  public:
    ZeroTermImagelessTracklet();
    virtual ~ZeroTermImagelessTracklet();

    void RenewTrajectory(const cv::Rect2f &bounding_box) override;

  public:
    int32_t birth_count;
    std::unique_ptr<KalmanFilterNoOpencv> kalman_filter;
};

class ShortTermImagelessTracklet : public Tracklet {
  public:
    ShortTermImagelessTracklet();
    virtual ~ShortTermImagelessTracklet();

    void RenewTrajectory(const cv::Rect2f &bounding_box) override;

  public:
    std::unique_ptr<KalmanFilterNoOpencv> kalman_filter;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_TRACKET_HPP
