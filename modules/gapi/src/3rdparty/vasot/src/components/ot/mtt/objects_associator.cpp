/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "objects_associator.hpp"
#include "hungarian_wrap.hpp"
#include "rgb_histogram.hpp"
#include "../prof_def.hpp"
#include "../../../common/exception.hpp"

namespace vas {
namespace ot {

const float kAssociationCostThreshold = 1.0f;
const float kRgbHistDistScale = 0.25f;
const float kNormCenterDistScale = 0.5f;
const float kNormShapeDistScale = 0.75f;

ObjectsAssociator::ObjectsAssociator(bool tracking_per_class) : tracking_per_class_(tracking_per_class) {
}

ObjectsAssociator::~ObjectsAssociator() {
}

std::pair<std::vector<bool>, std::vector<int32_t>>
ObjectsAssociator::Associate(const std::vector<Detection> &detections,
                             const std::vector<std::shared_ptr<Tracklet>> &tracklets,
                             const std::vector<cv::Mat> *detection_rgb_features) {
    PROF_START(PROF_COMPONENTS_OT_ASSOCIATE_COMPUTE_DIST_TABLE);
    std::vector<std::vector<float>> d2t_rgb_dist_table;

    if (detection_rgb_features != nullptr) {
        d2t_rgb_dist_table = ComputeRgbDistance(detections, tracklets, detection_rgb_features);
    }

    auto n_detections = detections.size();
    auto n_tracklets = tracklets.size();

    std::vector<bool> d_is_associated(n_detections, false);
    std::vector<int32_t> t_associated_d_index(n_tracklets, -1);

    // Compute detection-tracklet normalized position distance table
    std::vector<std::vector<float>> d2t_pos_dist_table(n_detections, std::vector<float>(n_tracklets, 1000.0f));
    for (std::size_t d = 0; d < n_detections; ++d) {
        TRACE("input detect(%.0f,%.0f %.0fx%.0f)", detections[d].rect.x, detections[d].rect.y, detections[d].rect.width,
              detections[d].rect.height);
        for (std::size_t t = 0; t < n_tracklets; ++t) {
            if (tracking_per_class_ && (detections[d].class_label != tracklets[t]->label))
                continue;

            d2t_pos_dist_table[d][t] = NormalizedCenterDistance(detections[d].rect, tracklets[t]->trajectory.back());
        }
    }

    // Compute detection-tracklet normalized shape distance table
    std::vector<std::vector<float>> d2t_shape_dist_table(n_detections, std::vector<float>(n_tracklets, 1000.0f));
    for (std::size_t d = 0; d < n_detections; ++d) {
        for (std::size_t t = 0; t < n_tracklets; ++t) {
            if (tracking_per_class_ && (detections[d].class_label != tracklets[t]->label))
                continue;

            d2t_shape_dist_table[d][t] = NormalizedShapeDistance(detections[d].rect, tracklets[t]->trajectory.back());
        }
    }
    PROF_END(PROF_COMPONENTS_OT_ASSOCIATE_COMPUTE_DIST_TABLE);

    PROF_START(PROF_COMPONENTS_OT_ASSOCIATE_COMPUTE_COST_TABLE);
    // Compute detection-tracklet association cost table
    cv::Mat_<float> d2t_cost_table;
    d2t_cost_table.create(static_cast<int32_t>(detections.size()),
                          static_cast<int32_t>(tracklets.size() + detections.size()));
    d2t_cost_table = kAssociationCostThreshold + 1.0f;

    for (std::size_t t = 0; t < n_tracklets; ++t) {
        const auto &tracklet = tracklets[t];
        float rgb_hist_dist_scale = kRgbHistDistScale;

        float const_ratio = 0.95f;
        float norm_center_dist_scale =
            (1.0f - const_ratio) * kNormCenterDistScale * tracklet->association_delta_t / 0.033f +
            const_ratio * kNormCenterDistScale; // adaptive to delta_t
        float norm_shape_dist_scale =
            (1.0f - const_ratio) * kNormShapeDistScale * tracklet->association_delta_t / 0.033f +
            const_ratio * kNormShapeDistScale; // adaptive to delta_t
        float log_term = logf(rgb_hist_dist_scale * norm_center_dist_scale * norm_shape_dist_scale);

        for (std::size_t d = 0; d < n_detections; ++d) {
            if (tracking_per_class_ && (detections[d].class_label != tracklets[t]->label))
                continue;

            d2t_cost_table(static_cast<int32_t>(d), static_cast<int32_t>(t)) =
                log_term + d2t_pos_dist_table[d][t] / norm_center_dist_scale +
                d2t_shape_dist_table[d][t] / norm_shape_dist_scale;

            if (d2t_rgb_dist_table.empty() == false) {
                d2t_cost_table(static_cast<int32_t>(d), static_cast<int32_t>(t)) +=
                    d2t_rgb_dist_table[d][t] / kRgbHistDistScale;
            }
        }
    }

    for (std::size_t d = 0; d < n_detections; ++d) {
        d2t_cost_table(static_cast<int32_t>(d), static_cast<int32_t>(d + n_tracklets)) =
            kAssociationCostThreshold;
    }
    PROF_END(PROF_COMPONENTS_OT_ASSOCIATE_COMPUTE_COST_TABLE);

    // Solve detection-tracking association using Hungarian algorithm
    PROF_START(PROF_COMPONENTS_OT_ASSOCIATE_WITH_HUNGARIAN);
    HungarianAlgo hungarian(d2t_cost_table);
    cv::Mat_<uint8_t> d2t_assign_table = hungarian.Solve();
    PROF_END(PROF_COMPONENTS_OT_ASSOCIATE_WITH_HUNGARIAN);

    for (std::size_t d = 0; d < n_detections; ++d) {
        for (std::size_t t = 0; t < n_tracklets; ++t) {
            if (d2t_assign_table(static_cast<int32_t>(d), static_cast<int32_t>(t))) {
                d_is_associated[d] = true;
                t_associated_d_index[t] = static_cast<int32_t>(d);
                break;
            }
        }
    }

    return std::make_pair(d_is_associated, t_associated_d_index);
}

std::vector<std::vector<float>>
ObjectsAssociator::ComputeRgbDistance(const std::vector<Detection> &detections,
                                      const std::vector<std::shared_ptr<Tracklet>> &tracklets,
                                      const std::vector<cv::Mat> *detection_rgb_features) {
    auto n_detections = detections.size();
    auto n_tracklets = tracklets.size();

    // Compute detection-tracklet RGB feature distance table
    std::vector<std::vector<float>> d2t_rgb_dist_table(n_detections, std::vector<float>(n_tracklets, 1000.0f));
    for (std::size_t d = 0; d < n_detections; ++d) {
        const auto &d_rgb_feature = (*detection_rgb_features)[d];
        for (std::size_t t = 0; t < n_tracklets; ++t) {
            if (tracking_per_class_ && (detections[d].class_label != tracklets[t]->label))
                continue;

             // Check if RGB features are available
        auto t_rgb_features = tracklets[t]->GetRgbFeatures();
        if (!t_rgb_features || t_rgb_features->empty()) {
            continue; // Skip if no RGB features are available
        }

            // Find best match in rgb feature history
            float min_dist = 1000.0f;
            for (const auto &t_rgb_feature : *(tracklets[t]->GetRgbFeatures())) {
                min_dist = std::min(min_dist, 1.0f - RgbHistogram::ComputeSimilarity(d_rgb_feature, t_rgb_feature));
            }
            d2t_rgb_dist_table[d][t] = min_dist;
        }
    }

    return d2t_rgb_dist_table;
}

float ObjectsAssociator::NormalizedCenterDistance(const cv::Rect2f &r1, const cv::Rect2f &r2) {
    float normalizer = std::min(0.5f * (r1.width + r1.height), 0.5f * (r2.width + r2.height));

    float r1x = r1.x + 0.5f * r1.width;
    float r1y = r1.y + 0.5f * r1.height;
    float r2x = r2.x + 0.5f * r2.width;
    float r2y = r2.y + 0.5f * r2.height;
    float dx = (r2x - r1x) / normalizer;
    float dy = (r2y - r1y) / normalizer;
    return std::sqrt(dx * dx + dy * dy);
}

float ObjectsAssociator::NormalizedShapeDistance(const cv::Rect2f &r1, const cv::Rect2f &r2) {
    int32_t normalize_w = int32_t(r1.width);
    int32_t normalize_h = int32_t(r1.height);

    if (r2.width + r2.height < r1.width + r1.height) {
        normalize_w = int32_t(r2.width);
        normalize_h = int32_t(r2.height);
    }

    float dw = (r2.width - r1.width) / normalize_w;
    float dh = (r2.height - r1.height) / normalize_h;
    return std::sqrt(dw * dw + dh * dh);
}

}; // namespace ot
}; // namespace vas
