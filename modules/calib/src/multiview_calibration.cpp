// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/softfloat.hpp"
#include "fisheye.hpp"
#include <iomanip>

namespace cv {
namespace multiview {
class RobustFunction : public Algorithm {
public:
    virtual float getError(float err) const = 0;
};

#define USE_FAST_EXP 0

// TODO: should we require a least number of visible points? Now I set to three since this should be the minimal number of points to do the PNP
#define MINIMUM_OBSERVATION 4
#define MINIMUM_AREA_RATIO 0.005

#if USE_FAST_EXP
class RobustExpFunction : public RobustFunction {
private:
    const float over_scale, pow_23 = 1 << 23;
public:
    explicit RobustExpFunction (float scale_=30.0f) : over_scale(-1.442695040f /scale_) {}
    // err > 0
    float getError(float err) const override {
        const float under_exp = err * over_scale;
        if (under_exp < -20) return 0; // prevent overflow further
        // http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
        softfloat vexp = softfloat::fromRaw(static_cast<uint32_t>(pow_23 * (under_exp + 126.94269504f)));
        return float(vexp);
    }
};
#else
class RobustExpFunction : public RobustFunction {
private:
    const float minvScale;
public:
    explicit RobustExpFunction (float scale_=30.0f) : minvScale(-1.f / scale_) {}
    // err > 0
    float getError(float err) const override
    {
        return exp(minvScale * err);
    }
};
#endif

// TODO: the function here is still incorrect: W should not directly be considered as a diagonal
static double robustWrapper (const Mat& ptsErrors, Mat& weights, const RobustFunction &fnc) {
    Mat errs;
    ptsErrors.convertTo(errs, CV_32F);
    weights.create((int)ptsErrors.total()*ptsErrors.channels(), 1, CV_32FC1);
    const Point2f * errs_ptr = errs.ptr<Point2f>();
    float * weights_ptr = weights.ptr<float>();
    double robust_sum_sqr_errs = 0.0;
    for (int pt = 0; pt < (int)errs.total(); pt++) {
        Point2f p = errs_ptr[pt];
        float sqr_err = p.dot(p);
        float w = fnc.getError(sqr_err);
        weights_ptr[pt*2 + 0] = w;
        weights_ptr[pt*2 + 1] = w;
        robust_sum_sqr_errs += w * sqr_err;
    }
    return robust_sum_sqr_errs;
}

static double computeReprojectionMSE(const Mat &obj_points_, const Mat &img_points_, const Matx33d &K, const Mat &distortion,
               const Mat &rvec, const Mat &tvec, InputArray rvec2, InputArray tvec2, bool is_fisheye) {
    Mat r, t;
    if (!rvec2.empty() && !tvec2.empty()) {
        composeRT(rvec, tvec, rvec2, tvec2, r, t);
    } else {
        r = rvec; t = tvec;
    }

    Mat tmpImagePoints, obj_points = obj_points_, img_points = img_points_;
    if (is_fisheye) {
        obj_points = obj_points.reshape(3); // must be 3 channels
        fisheye::projectPoints(obj_points, tmpImagePoints, r, t, K, distortion);
    } else {
        projectPoints(obj_points, r, t, K, distortion, tmpImagePoints);
    }

    if (img_points.channels() != tmpImagePoints.channels())
        img_points = img_points.reshape(tmpImagePoints.channels());

    if (img_points.rows != tmpImagePoints.rows)
        img_points = img_points.t();

    subtract(tmpImagePoints, img_points, tmpImagePoints);

    return norm(tmpImagePoints, NORM_L2SQR) / tmpImagePoints.rows;
}

static void establishValidPointMap(const std::vector<std::vector<Mat>>& imagePoints,
                        const std::vector<Size> &imageSize,
                        const std::vector<std::vector<bool>>& detection_mask_mat,
                        std::vector<std::vector<std::vector<bool>>>& is_valid_imgpt) {

    int NUM_CAMERAS = int(imagePoints.size());
    int NUM_FRAMES = int(imagePoints[0].size());
    int NUM_PATTERN_PTS = 0;
    for (int c = 0; c < NUM_CAMERAS; c++) {
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!detection_mask_mat[c][f])
                continue;
            if (NUM_PATTERN_PTS == 0)
                NUM_PATTERN_PTS = imagePoints[c][f].rows;
            for (int p = 0; p < NUM_PATTERN_PTS; p++) {
                if (imagePoints[c][f].type() == CV_32F && imagePoints[c][f].cols == 2) {
                    if (std::min((imagePoints[c][f].at<float>(p, 0)), imagePoints[c][f].at<float>(p, 1)) < 0)
                        continue;

                    if (imageSize[c].height > 0 && imageSize[c].width > 0) {
                        if ((imagePoints[c][f].at<float>(p, 0) > imageSize[c].width) || (imagePoints[c][f].at<float>(p, 1) > imageSize[c].height))
                            continue;
                    }
                } else { // imagePoints[c][f].type() == CV_32FC2
                    if (std::min((imagePoints[c][f].at<Point2f>(p, 0).x), imagePoints[c][f].at<Point2f>(p, 0).y) < 0)
                        continue;

                    if (imageSize[c].height > 0 && imageSize[c].width > 0) {
                        if ((imagePoints[c][f].at<Point2f>(p, 0).x > imageSize[c].width) || (imagePoints[c][f].at<Point2f>(p, 0).y > imageSize[c].height))
                            continue;
                    }
                }
                is_valid_imgpt[c][f][p] = true;
            }
        }
    }
}

static bool maximumSpanningTree (int NUM_CAMERAS, int NUM_FRAMES, const std::vector<std::vector<bool>> &detection_mask,
          std::vector<int> &parent, std::vector<std::vector<int>> &overlap,
          std::vector<std::vector<Vec3d>> &opt_axes,
          const std::vector<std::vector<bool>> &is_valid_angle2pattern,
          const std::vector<std::vector<float>> &points_area_ratio,
          double WEIGHT_ANGLE_PATTERN, double WEIGHT_CAMERAS_ANGLES) {
    const double THR_CAMERAS_ANGLES = 160*M_PI/180;
    // build weights matrix
    overlap = std::vector<std::vector<int>>(NUM_CAMERAS, std::vector<int>(NUM_CAMERAS, 0));
    std::vector<std::vector<double>> weights(NUM_CAMERAS, std::vector<double>(NUM_CAMERAS, DBL_MIN));
    for (int c1 = 0; c1 < NUM_CAMERAS; c1++) {
        for (int c2 = c1+1; c2 < NUM_CAMERAS; c2++) {
            double weight = 0;
            int overlaps = 0;
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (detection_mask[c1][f] && detection_mask[c2][f]) {
                    overlaps += 1;
                    weight += points_area_ratio[c1][f] + points_area_ratio[c2][f];
                    weight += WEIGHT_ANGLE_PATTERN * ((int)is_valid_angle2pattern[c1][f] + (int)is_valid_angle2pattern[c2][f]);
                    if (WEIGHT_CAMERAS_ANGLES > 0) {
                        // angle between cameras optical axes
                        weight += WEIGHT_CAMERAS_ANGLES * int(acos(opt_axes[c1][f].dot(opt_axes[c2][f])) < THR_CAMERAS_ANGLES);
                    }
                }
            }
            if (overlaps > 0) {
                overlap[c1][c2] = overlap[c2][c1] = overlaps;
                weights[c1][c2] = weights[c2][c1] = overlaps + weight;
            }
        }
    }

    // find maximum spanning tree using Prim's algorithm
    std::vector<bool> visited(NUM_CAMERAS, false);
    std::vector<double> weight(NUM_CAMERAS, DBL_MIN);
    parent = std::vector<int>(NUM_CAMERAS, -1);
    weight[0] = DBL_MAX;
    for (int cam =  0; cam < NUM_CAMERAS-1; cam++) {
        int max_weight_idx = -1;
        auto max_weight = DBL_MIN;
        for (int cam2 = 0; cam2 < NUM_CAMERAS; cam2++) {
            if (!visited[cam2] && max_weight < weight[cam2]) {
                max_weight = weight[cam2];
                max_weight_idx = cam2;
            }
        }
        if (max_weight_idx == -1)
            return false;
        visited[max_weight_idx] = true;
        for (int cam2 = 0; cam2 < NUM_CAMERAS; cam2++) {
            if (!visited[cam2] && overlap[max_weight_idx][cam2] > 0) {
                if (weight[cam2] < weights[max_weight_idx][cam2]) {
                    weight[cam2] = weights[max_weight_idx][cam2];
                    parent[cam2] = max_weight_idx;
                }
            }
        }
    }
    return true;
}

static double imagePointsAreaFrame (const Size& imageSize, const Mat& imagePoints) {
    std::vector<int> hull;
    const auto img_area = (float)(imageSize.width * imageSize.height);
    const auto * const image_pts_ptr = (float *) imagePoints.data;
    convexHull(imagePoints, hull, true/*has to be clockwise*/, false/*indices*/);
    float area = 0;
    int j = hull.back();
    // http://alienryderflex.com/polygon_area/
    for (int i : hull) {
        area += (image_pts_ptr[j*2] + image_pts_ptr[i*2])*(image_pts_ptr[j*2+1] - image_pts_ptr[i*2+1]);
        j = i;
    }
    return area*.5f / img_area;
}

static void selectPairsBFS (std::vector<std::pair<int,int>> &pairs, int NUM_CAMERAS, const std::vector<int> &parent) {
    // find pairs using Breadth First Search graph traversing
    // it is important to keep this order of pairs, since it is easier
    // to find relative views wrt to 0-th camera.
    std::vector<int> nodes = {0};
    pairs.reserve(NUM_CAMERAS-1);
    while (!nodes.empty()) {
        std::vector<int> new_nodes;
        for (int n : nodes) {
            for (int c = 0; c < NUM_CAMERAS; c++) {
                if (parent[c] == n) {
                    pairs.emplace_back(std::make_pair(n, c));
                    new_nodes.emplace_back(c);
                }
            }
        }
        nodes = new_nodes;
    }
}

static double getScaleOfObjPoints (int NUM_PATTERN_PTS, const Mat &obj_pts, bool obj_points_in_rows) {
    double scale_3d_pts = 0.0;
    // compute scale of 3D points as the maximum pairwise distance
    for (int i = 0; i < NUM_PATTERN_PTS; i++) {
        for (int j = i+1; j < NUM_PATTERN_PTS; j++) {
            double dist;
            if (obj_points_in_rows) {
                dist = norm(obj_pts.row(i)-obj_pts.row(j), NORM_L2SQR);
            } else {
                dist = norm(obj_pts.col(i)-obj_pts.col(j), NORM_L2SQR);
            }
            if (scale_3d_pts < dist) {
                scale_3d_pts = dist;
            }
        }
    }
    return scale_3d_pts;
}

static void thresholdPatternCameraAngles (int NUM_PATTERN_PTS, double THR_PATTERN_CAMERA_ANGLES,
        const std::vector<Mat> &objPoints_norm, const std::vector<std::vector<Vec3d>> &rvecs_all,
        std::vector<std::vector<Vec3d>> &opt_axes, std::vector<std::vector<bool>> &is_valid_angle2pattern) {
    const int NUM_FRAMES = (int)objPoints_norm.size(), NUM_CAMERAS = (int)rvecs_all.size();
    is_valid_angle2pattern = std::vector<std::vector<bool>>(NUM_CAMERAS, std::vector<bool>(NUM_FRAMES, true));
    int pattern1 = -1, pattern2 = -1, pattern3 = -1;
    for (int f = 0; f < NUM_FRAMES; f++) {
        double norm_normal = 0;
        if (pattern1 == -1) {
            // take non colinear 3 points and save them
            for (int p1 = 0; p1 < NUM_PATTERN_PTS; p1++) {
                for (int p2 = p1+1; p2 < NUM_PATTERN_PTS; p2++) {
                    for (int p3 = NUM_PATTERN_PTS-1; p3 > p2; p3--) { // start from the last point
                        Mat pattern_normal = (objPoints_norm[f].row(p2)-objPoints_norm[f].row(p1))
                                    .cross(objPoints_norm[f].row(p3)-objPoints_norm[f].row(p1));
                        norm_normal = norm(pattern_normal, NORM_L2SQR);
                        if (norm_normal > 1e-6) {
                            pattern1 = p1;
                            pattern2 = p2;
                            pattern3 = p3;
                            norm_normal = sqrt(norm_normal);
                            break;
                        }
                    }
                    if (pattern1 != -1) break;
                }
                if (pattern1 != -1) break;
            }
            if (pattern1 == -1) {
                CV_Error(Error::StsBadArg, "Pattern points are collinear!");
            }
        }
        Vec3d pattern_normal = (objPoints_norm[f].row(pattern2)-objPoints_norm[f].row(pattern1)).
                  cross(objPoints_norm[f].row(pattern3)-objPoints_norm[f].row(pattern1));
        norm_normal = norm(pattern_normal);
        pattern_normal /= norm_normal;

        for (int c = 0; c < NUM_CAMERAS; c++) {
            Matx33d R;
            Rodrigues(rvecs_all[c][f], R);
            opt_axes[c][f] = Vec3d(Mat(R.row(2)));
            const double angle = acos(opt_axes[c][f].dot(pattern_normal));
            is_valid_angle2pattern[c][f] = min(M_PI-angle, angle) < THR_PATTERN_CAMERA_ANGLES;
        }
    }
}

static void pairwiseCalibration (const std::vector<std::pair<int,int>> &pairs,
        const std::vector<bool> &is_fisheye_vec, const std::vector<std::vector<Mat>> &objPoints_norm,
        const std::vector<std::vector<Mat>> &imagePoints, const std::vector<std::vector<int>> &overlaps,
        const std::vector<std::vector<bool>> &detection_mask_mat,
        const std::vector<Mat> &Ks,
        const std::vector<Mat> &distortions, std::vector<Matx33d> &Rs_vec, std::vector<Vec3d> &Ts_vec, bool useExtrinsicsGuess) {
    const int NUM_FRAMES = (int) detection_mask_mat[0].size();
    int NUM_CAMERAS = int(Rs_vec.size());

    std::vector<Matx33d> Rs_prior;
    std::vector<Vec3d> Ts_prior;
    if (useExtrinsicsGuess) {
        Rs_prior.resize(NUM_CAMERAS);
        Ts_prior.resize(NUM_CAMERAS);
        for (int i = 0; i < NUM_CAMERAS; i++) {
            Rs_vec[i].copyTo(Rs_prior[i]);
            Ts_vec[i].copyTo(Ts_prior[i]);
        }
    }

    std::vector<CameraModel> camera_models(NUM_CAMERAS);
    for (int camera = 0; camera < NUM_CAMERAS; camera++) {
        if (is_fisheye_vec[camera]) {
            camera_models[camera] = CALIB_MODEL_FISHEYE;
        } else {
            camera_models[camera] = CALIB_MODEL_PINHOLE;
        }
    }

    for (const auto &pair : pairs) {
        const int c1 = pair.first, c2 = pair.second, overlap = overlaps[c1][c2];
        // prepare image points of two cameras and grid points
        std::vector<Mat> image_points1, image_points2, grid_points1, grid_points2;
        grid_points1.reserve(overlap);
        grid_points2.reserve(overlap);
        image_points1.reserve(overlap);
        image_points2.reserve(overlap);
        int cnt_valid_frame1 = 0, cnt_valid_frame2 = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (detection_mask_mat[c1][f] && detection_mask_mat[c2][f]) {
                grid_points1.emplace_back(objPoints_norm[c1][cnt_valid_frame1]);
                grid_points2.emplace_back(objPoints_norm[c2][cnt_valid_frame2]);
                image_points1.emplace_back(imagePoints[c1][cnt_valid_frame1]);
                image_points2.emplace_back(imagePoints[c2][cnt_valid_frame2]);
            }
            if (detection_mask_mat[c1][f])
                cnt_valid_frame1++;
            if (detection_mask_mat[c2][f])
                cnt_valid_frame2++;

        }
        Matx33d R;
        Vec3d T;
        if (useExtrinsicsGuess) {
            R = Rs_prior[c2] * Rs_prior[c1].t();
            T = -R * Ts_prior[c1] + Ts_prior[c2];
        }
        // TODO: what flags do we need to perform the stereo calibration?
        // image size does not matter since intrinsics are used
        int flags_extrinsics = CALIB_FIX_INTRINSIC;

        if (useExtrinsicsGuess) {
            flags_extrinsics += CALIB_USE_EXTRINSIC_GUESS;
        }

        registerCameras(grid_points1, grid_points2, image_points1, image_points2,
                        Ks[c1], distortions[c1], camera_models[c1],
                        Ks[c2], distortions[c2], camera_models[c2],
                        R, T, noArray(), noArray(), noArray(), noArray(), noArray(), flags_extrinsics);

        // R_0 = I
        // R_ij = R_i R_j^T     =>  R_i = R_ij R_j
        // t_ij = ti - R_ij tj  =>  t_i = t_ij + R_ij t_j
        if (c1 == 0) {
            Rs_vec[c2] = R;
            Ts_vec[c2] = T;
        } else {
            Rs_vec[c2] = Matx33d(Mat(R * Rs_vec[c1]));
            Ts_vec[c2] = Vec3d(Mat(T + R * Ts_vec[c1]));
        }
    }
}

static void optimizeLM (std::vector<double> &param,
                const RobustFunction &robust_fnc,
                const TermCriteria &termCrit,
                const std::vector<bool> &valid_frames,
                const std::vector<std::vector<bool>> &detection_mask_mat,
                const std::vector<std::vector<Mat>> &objPoints_norm,
                const std::vector<std::vector<Mat>> &imagePoints,
                const std::vector<Mat> &Ks,
                const std::vector<Mat> &distortions,
                const std::vector<bool> &is_fisheye_vec) {
    const int NUM_FRAMES = (int) detection_mask_mat[0].size(), NUM_CAMERAS = (int)detection_mask_mat.size();
    int iters_lm = 0, cnt_valid_frame = 0;
    auto lmcallback = [&](InputOutputArray _param, OutputArray JtErr_, OutputArray JtJ_, double& errnorm) {
        auto * param_p = _param.getMat().ptr<double>();
        errnorm = 0;
        cnt_valid_frame = 0;
        std::vector<int> frame_head(NUM_CAMERAS, 0);
        for (int i = 0; i < NUM_FRAMES; i++ ) {
            if (!valid_frames[i]) continue;
            for (int k = 0; k < NUM_CAMERAS; k++ ) {
                // Pose for camera #0 is not optimized, but it's re-projection error is taken into account
                if (!detection_mask_mat[k][i]) continue;
                int f = frame_head[k];
                frame_head[k]++;

                int NUM_PATTERN_PTS = (int) objPoints_norm[k][f].rows;
                const int cam_idx = (k-1)*6; // camera extrinsics
                const auto * const pose_k = (k > 0)? (param_p + cam_idx) : nullptr;
                Vec3d om_0ToK = (k > 0)? Vec3d(pose_k[0], pose_k[1], pose_k[2]) : Vec3d(0., 0., 0.), om[2];
                Vec3d T_0ToK = (k > 0)? Vec3d(pose_k[3], pose_k[4], pose_k[5]) : Vec3d(0., 0., 0.), T[2];
                Matx33d dr3dr1, dr3dr2, dt3dr2, dt3dt1, dt3dt2;

                auto * pi = param_p + (cnt_valid_frame+NUM_CAMERAS-1)*6; // get rvecs / tvecs for frame pose
                om[0] = Vec3d(pi[0], pi[1], pi[2]);
                T[0] = Vec3d(pi[3], pi[4], pi[5]);

                if( JtJ_.needed() || JtErr_.needed() )
                    composeRT( om[0], T[0], om_0ToK, T_0ToK, om[1], T[1], dr3dr1, noArray(),
                               dr3dr2, noArray(), noArray(), dt3dt1, dt3dr2, dt3dt2 );
                else
                    composeRT( om[0], T[0], om_0ToK, T_0ToK, om[1], T[1] );

                // get object points
                Mat objpt_i = objPoints_norm[k][f].reshape(3, 1);
                objpt_i.convertTo(objpt_i, CV_64FC3);

                Mat err( NUM_PATTERN_PTS*2, 1, CV_64F ), tmpImagePoints = err.reshape(2, 1);
                Mat Je( NUM_PATTERN_PTS*2, 6, CV_64F ), J_0ToK( NUM_PATTERN_PTS*2, 6, CV_64F );
                Mat dpdrot = Je.colRange(0, 3), dpdt = Je.colRange(3, 6); // num_points*2 x 3 each
                // get image points
                Mat imgpt_ik = imagePoints[k][f].reshape(2, 1);
                imgpt_ik.convertTo(imgpt_ik, CV_64FC2);

                if (is_fisheye_vec[k]) {
                    if( JtJ_.needed() || JtErr_.needed() ) {
                        Mat jacobian; // of size num_points*2  x  15 (2 + 2 + 4 + 3 + 3 + 1; // f, c, k, om, T, alpha)
                        fisheye::projectPoints(objpt_i, tmpImagePoints, om[1], T[1], Ks[k], distortions[k], 0, jacobian);
                        jacobian.colRange(8,11).copyTo(dpdrot);
                        jacobian.colRange(11,14).copyTo(dpdt);
                    } else
                        fisheye::projectPoints(objpt_i, tmpImagePoints, om[1], T[1], Ks[k], distortions[k]);
                } else {
                    if( JtJ_.needed() || JtErr_.needed() )
                        projectPoints(objpt_i, om[1], T[1], Ks[k], distortions[k],
                                        tmpImagePoints, dpdrot, dpdt, noArray(), noArray(), noArray(), noArray());
                    else
                        projectPoints(objpt_i, om[1], T[1], Ks[k], distortions[k], tmpImagePoints);
                }
                subtract( tmpImagePoints, imgpt_ik, tmpImagePoints);
                Mat weights;
                const double robust_l2_norm = multiview::robustWrapper(tmpImagePoints, weights, robust_fnc);
                errnorm += robust_l2_norm;

                if (JtJ_.needed()) {
                    Mat JtErr = JtErr_.getMat(), JtJ = JtJ_.getMat();
                    const int eofs = (cnt_valid_frame+NUM_CAMERAS-1)*6;
                    assert( JtJ_.needed() && JtErr_.needed() );
                    // JtJ : NUM_PARAMS x NUM_PARAMS, JtErr : NUM_PARAMS x 1
                    // d(err_{x|y}R) ~ de3
                    // convert de3/{dr3,dt3} => de3{dr1,dt1} & de3{dr2,dt2}

                    Mat wd;
                    Mat::diag(weights).convertTo(wd, CV_64F);
                    if (k > 0) { // if not camera #0
                        for (int p = 0; p < NUM_PATTERN_PTS * 2; p++) {
                            Matx13d de3dr3, de3dt3, de3dr2, de3dt2, de3dr1, de3dt1;
                            for (int j = 0; j < 3; j++)
                                de3dr3(j) = Je.at<double>(p, j);

                            for (int j = 0; j < 3; j++)
                                de3dt3(j) = Je.at<double>(p, 3 + j);

                            for (int j = 0; j < 3; j++)
                                de3dr2(j) = J_0ToK.at<double>(p, j);

                            for (int j = 0; j < 3; j++)
                                de3dt2(j) = J_0ToK.at<double>(p, 3 + j);

                            de3dr1 = de3dr3 * dr3dr1;
                            de3dt1 = de3dt3 * dt3dt1;
                            de3dr2 = de3dr3 * dr3dr2 + de3dt3 * dt3dr2;
                            de3dt2 = de3dt3 * dt3dt2;

                            for (int j = 0; j < 3; j++)
                                Je.at<double>(p, j) = de3dr1(j);

                            for (int j = 0; j < 3; j++)
                                Je.at<double>(p, 3 + j) = de3dt1(j);

                            for (int j = 0; j < 3; j++)
                                J_0ToK.at<double>(p, j) = de3dr2(j);

                            for (int j = 0; j < 3; j++)
                                J_0ToK.at<double>(p, 3 + j) = de3dt2(j);
                        }

                        // 6 x (ni*2) * (ni*2 x ni*2) * (ni*2) x 6
                        JtJ(Rect((k - 1) * 6, (k - 1) * 6, 6, 6)) += (J_0ToK.t() * wd * J_0ToK);
                        JtJ(Rect(eofs, (k - 1) * 6, 6, 6)) = (J_0ToK.t() * wd * Je);
                        JtErr.rowRange((k - 1) * 6, (k - 1) * 6 + 6) += (J_0ToK.t() * wd * err);
                    }
                    JtJ(Rect(eofs, eofs, 6, 6)) += Je.t() * wd * Je;
                    JtErr.rowRange(eofs, eofs + 6) += Je.t() * wd * err;
                }
            }
            cnt_valid_frame++;
        }
        iters_lm += 1;
        return true;
    };
    LevMarq solver(param, lmcallback,
       LevMarq::Settings()
               .setMaxIterations(termCrit.maxCount)
               .setStepNormTolerance(termCrit.epsilon)
               .setSmallEnergyTolerance(termCrit.epsilon * termCrit.epsilon),
           noArray()/*mask, all variables to optimize*/);
    solver.optimize();
}

static void checkConnected (const std::vector<std::vector<bool>> &detection_mask_mat) {
    const int NUM_CAMERAS = (int)detection_mask_mat.size(), NUM_FRAMES = (int)detection_mask_mat[0].size();
    std::vector<bool> visited(NUM_CAMERAS, false);
    std::function<void(int)> dfs_search;
    dfs_search = [&] (int cam) {
        visited[cam] = true;
        for (int cam2 = 0; cam2 < NUM_CAMERAS; cam2++) {
            if (!visited[cam2]) {
                for (int f = 0; f < NUM_FRAMES; f++) {
                    if (detection_mask_mat[cam][f] && detection_mask_mat[cam2][f]) {
                        dfs_search(cam2);
                        break;
                    }
                }
            }
        }
    };
    dfs_search(0);
    for (int c = 0; c < NUM_CAMERAS; c++) {
        if (! visited[c]) {
            std::string isolated_cameras = "", visited_str = "";
            for (int i = 0; i < NUM_CAMERAS; i++) {
                if (!visited[i]) {
                    if (isolated_cameras != "")
                        isolated_cameras += ", ";
                    isolated_cameras += std::to_string(i);
                } else {
                    if (visited_str != "")
                        visited_str += ", ";
                    visited_str += std::to_string(i);
                }
            }
            CV_Error(Error::StsBadArg, "Isolated cameras (or components) "+isolated_cameras+" from the connected component "+visited_str+"!");
        }
    }
}
}

//TODO: use Input/OutputArrays for imagePoints, imageSize(?), Ks, distortions
double calibrateMultiview (InputArrayOfArrays objPoints, const std::vector<std::vector<Mat>> &imagePoints,
        const std::vector<Size> &imageSize, InputArray detectionMask,
        InputOutputArrayOfArrays Rs, InputOutputArrayOfArrays Ts, std::vector<Mat> &Ks, std::vector<Mat> &distortions,
        OutputArrayOfArrays rvecs0, OutputArrayOfArrays tvecs0, InputArray isFisheye,
        OutputArray perFrameErrors, OutputArray initializationPairs, bool useIntrinsicsGuess, InputArray flagsForIntrinsics, bool useExtrinsicsGuess) {

    CV_CheckEQ((int)objPoints.empty(), 0, "Objects points must not be empty!");
    CV_CheckEQ((int)imagePoints.empty(), 0, "Image points must not be empty!");
    CV_CheckEQ((int)imageSize.empty(), 0, "Image size per camera must not be empty!");
    CV_CheckEQ((int)detectionMask.empty(), 0, "detectionMask matrix must not be empty!");
    CV_CheckEQ((int)isFisheye.empty(), 0, "Fisheye mask must not be empty!");

    Mat detection_mask_ = detectionMask.getMat(), is_fisheye_mat = isFisheye.getMat();
    CV_CheckEQ(detection_mask_.type(), CV_8U, "detectionMask must be of type CV_8U");
    CV_CheckEQ(is_fisheye_mat.type(), CV_8U, "isFisheye must be of type CV_8U");


    // equal number of cameras
    CV_Assert(imageSize.size() == imagePoints.size());
    CV_Assert(detection_mask_.rows == std::max(isFisheye.rows(), isFisheye.cols()));
    CV_Assert(detection_mask_.rows == (int)imageSize.size());
    CV_Assert(detection_mask_.cols == std::max(objPoints.rows(), objPoints.cols())); // equal number of frames
    CV_Assert(Rs.isMatVector() == Ts.isMatVector());
    if (useIntrinsicsGuess) {
        CV_Assert(Ks.size() == distortions.size() && Ks.size() == imageSize.size());
    }

    if (useExtrinsicsGuess) {
        CV_Assert(Rs.isMatVector() && Ts.isMatVector());
        CV_Assert(Rs.total() == Ts.total() && Rs.total() == imageSize.size());
    }

    // normalize object points
    const Mat obj_pts_0 = objPoints.getMat(0);
    CV_Assert((obj_pts_0.type() == CV_32F && (obj_pts_0.rows == 3 || obj_pts_0.cols == 3)) ||
              (obj_pts_0.type() == CV_32FC3 && (obj_pts_0.rows == 1 || obj_pts_0.cols == 1)));
    const bool obj_points_in_rows = obj_pts_0.cols == 3;
    const int NUM_CAMERAS = (int)detection_mask_.rows, NUM_FRAMES = (int)detection_mask_.cols;
    CV_Assert((NUM_CAMERAS > 1) && (NUM_FRAMES > 0));

    // TODO: should we allow varient number of pattern points in the calibration?
    const int NUM_PATTERN_PTS = obj_points_in_rows ? obj_pts_0.rows : obj_pts_0.cols;
    const double scale_3d_pts = multiview::getScaleOfObjPoints(NUM_PATTERN_PTS, obj_pts_0, obj_points_in_rows);

    Mat flagsForIntrinsics_mat = flagsForIntrinsics.getMat();
    if (flagsForIntrinsics_mat.empty())
    {
        flagsForIntrinsics_mat = Mat(Size(1, NUM_CAMERAS), CV_32SC1, cv::Scalar(0));
        // set the flag for fisheye camera to be cv::CALIB_RECOMPUTE_EXTRINSIC+cv::CALIB_FIX_SKEW;
        auto * const is_fisheye_ptr = is_fisheye_mat.data;
        for (int c = 0; c < NUM_CAMERAS; c++) {
            if (is_fisheye_ptr[c] != 0) {
                flagsForIntrinsics_mat.at<int>(c) = cv::CALIB_RECOMPUTE_EXTRINSIC+cv::CALIB_FIX_SKEW;
            }
        }
    }

    CV_Assert(flagsForIntrinsics_mat.total() == size_t(NUM_CAMERAS));
    CV_CheckEQ(flagsForIntrinsics_mat.type(), CV_32S, "flagsForIntrinsics should be of type 32SC1");
    CV_CheckEQ(flagsForIntrinsics_mat.channels(), 1, "flagsForIntrinsics should be of type 32SC1");

    std::vector<Mat> objPoints_norm;
    objPoints_norm.reserve(NUM_FRAMES);
    for (int i = 0; i < NUM_FRAMES; i++) {
        if (obj_points_in_rows)
            objPoints_norm.emplace_back(objPoints.getMat(i)*(1/scale_3d_pts));
        else
            objPoints_norm.emplace_back(objPoints.getMat(i).t()*(1/scale_3d_pts));
        objPoints_norm[i] = objPoints_norm[i].reshape(1);
    }

    ////////////////////////////////////////////////
    std::vector<int> num_visible_frames_per_camera(NUM_CAMERAS);
    std::vector<bool> valid_frames(NUM_FRAMES, false);
    // process input and count all visible frames and points

    std::vector<bool> is_fisheye_vec(NUM_CAMERAS);
    std::vector<std::vector<bool>> detection_mask_mat(NUM_CAMERAS, std::vector<bool>(NUM_FRAMES));
    const auto * const detection_mask_ptr = detection_mask_.data, * const is_fisheye_ptr = is_fisheye_mat.data;
    for (int c = 0; c < NUM_CAMERAS; c++) {
        for (int f = 0; f < NUM_FRAMES; f++) {
            detection_mask_mat[c][f] = detection_mask_ptr[c*NUM_FRAMES + f] != 0;
        }
    }

    // Establish the valid point vector
    std::vector<std::vector<std::vector<bool>>> is_valid_imgpt(NUM_CAMERAS);
    for (int k = 0; k < NUM_CAMERAS; k++) {
        is_valid_imgpt[k].resize(NUM_FRAMES);
        for (int f = 0; f < NUM_FRAMES; f++) {
            is_valid_imgpt[k][f].resize(NUM_PATTERN_PTS, false);
        }
    }
    multiview::establishValidPointMap(imagePoints, imageSize, detection_mask_mat, is_valid_imgpt);

    std::vector<std::vector<Mat>> obj_points_valid(NUM_CAMERAS), img_points_valid(NUM_CAMERAS);
    std::vector<std::vector<float>> points_ratio_area(NUM_CAMERAS, std::vector<float>(NUM_FRAMES, 0));
    for (int camera = 0; camera < NUM_CAMERAS; camera++) {
        std::vector<Mat>& obj_points_ = obj_points_valid[camera];
        std::vector<Mat>& img_points_ = img_points_valid[camera];
        obj_points_.reserve(num_visible_frames_per_camera[camera]);
        img_points_.reserve(num_visible_frames_per_camera[camera]);
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!detection_mask_mat[camera][f])
                continue;
            Mat obj_points_frame, img_points_frame;
            for (int i = 0; i < NUM_PATTERN_PTS; i++) {
                if (is_valid_imgpt[camera][f][i]){
                    obj_points_frame.push_back(objPoints_norm[f].row(i).reshape(3));
                    img_points_frame.push_back(imagePoints[camera][f].row(i).reshape(2));
                }
            }
            double area_ratio = img_points_frame.empty() ? 0 : multiview::imagePointsAreaFrame(imageSize[camera], img_points_frame);

            // // Only put it if there are more than some points pr covers a non-degenerate region
            // CV_LOG_IF_WARNING(NULL, obj_points_frame.rows < MINIMUM_OBSERVATION || area_ratio < MINIMUM_AREA_RATIO, "Warning! Fewer than " + std::to_string(MINIMUM_OBSERVATION) + " object points are visible or observation covers less than " + std::to_string(MINIMUM_AREA_RATIO) + " area  in the image frame " + std::to_string(f) + " for camera " + std::to_string(camera));

            // Refine the detection mask by removing the frames with only degenerate observation (when all images points are collinear, or are very close together)
            if (obj_points_frame.rows >= MINIMUM_OBSERVATION && area_ratio >= MINIMUM_AREA_RATIO) {
                obj_points_.emplace_back(obj_points_frame);
                img_points_.emplace_back(img_points_frame);
                points_ratio_area[camera][f] = float(area_ratio);
            } else
                detection_mask_mat[camera][f] = false;
        }
    }

    // Establish the new valid frame count from the refined detection mask
    for (int c = 0; c < NUM_CAMERAS; c++) {
        is_fisheye_vec[c] = is_fisheye_ptr[c] != 0;
        int num_visible_frames = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (detection_mask_mat[c][f]) {
                num_visible_frames++;
                valid_frames[f] = true; // if frame is visible by at least one camera then count is as a valid one
            }
        }
        if (num_visible_frames == 0) {
            CV_Error(Error::StsBadArg, "camera "+std::to_string(c)+" has no visible frames!");
        }
        num_visible_frames_per_camera[c] = num_visible_frames;
    }

    multiview::checkConnected(detection_mask_mat);

    // constant threshold for angle between two camera axes in radians (=160*M_PI/180).
    // if angle exceeds this threshold then a weight of a camera pair is lowered.
    const double THR_PATTERN_CAMERA_ANGLES = 160*M_PI/180;
    std::vector<std::vector<Vec3d>> rvecs_all(NUM_CAMERAS, std::vector<Vec3d>(NUM_FRAMES)),
        tvecs_all(NUM_CAMERAS, std::vector<Vec3d>(NUM_FRAMES)),
        opt_axes(NUM_CAMERAS, std::vector<Vec3d>(NUM_FRAMES));

    std::vector<int> camera_rt_best(NUM_FRAMES, -1);
    std::vector<double> camera_rt_errors(NUM_FRAMES, std::numeric_limits<double>::max());
    const double WARNING_RMSE = 15.;
    if (!useIntrinsicsGuess) {
        // calibrate each camera independently to find intrinsic parameters - K and distortion coefficients
        distortions = std::vector<Mat>(NUM_CAMERAS);
        Ks = std::vector<Mat>(NUM_CAMERAS);
        for (int camera = 0; camera < NUM_CAMERAS; camera++) {
            Mat rvecs, tvecs;
            std::vector<Mat>& obj_points_ = obj_points_valid[camera];
            std::vector<Mat>& img_points_ = img_points_valid[camera];
            std::vector<double> errors_per_view;
            double repr_err;
            if (is_fisheye_vec[camera]) {
                repr_err = fisheye::calibrate(obj_points_, img_points_, imageSize[camera],
                    Ks[camera], distortions[camera], rvecs, tvecs, flagsForIntrinsics_mat.at<int>(camera));
                // calibrate does not compute error per view, so compute it manually
                errors_per_view = std::vector<double>(obj_points_.size());
                for (int f = 0; f < (int) obj_points_.size(); f++) {
                    double err2 = multiview::computeReprojectionMSE(obj_points_[f],
                        img_points_[f], Ks[camera], distortions[camera], rvecs.row(f), tvecs.row(f), noArray(), noArray(), true);
                    errors_per_view[f] = sqrt(err2);
                }
            } else {
                repr_err = calibrateCamera(obj_points_, img_points_, imageSize[camera], Ks[camera], distortions[camera],
                   rvecs, tvecs, noArray(), noArray(), errors_per_view, flagsForIntrinsics_mat.at<int>(camera));
            }
            CV_LOG_IF_WARNING(NULL, repr_err > WARNING_RMSE, "Warning! Mean RMSE of intrinsics calibration for camera "+std::to_string(camera)+" is higher than "+std::to_string(WARNING_RMSE)+" pixels!");
            int cnt_visible_frame = 0;
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (detection_mask_mat[camera][f]) {
                    rvecs_all[camera][f] = Vec3d(Mat(3, 1, CV_64F, rvecs.row(cnt_visible_frame).data));
                    tvecs_all[camera][f] = Vec3d(Mat(3, 1, CV_64F, tvecs.row(cnt_visible_frame).data));
                    double err = errors_per_view[cnt_visible_frame];
                    double err2 = err * err;
                    if (camera_rt_errors[f] > err2) {
                        camera_rt_errors[f] = err2;
                        camera_rt_best[f] = camera;
                    }
                    cnt_visible_frame++;
                }
            }
        }
    } else {
        // use PnP to compute rvecs and tvecs
        for (int k = 0; k < NUM_CAMERAS; k++) {
            int cnt_valid_frame = 0;
            for (int i = 0; i < NUM_FRAMES; i++) {
                if (!detection_mask_mat[k][i]) continue;

                if(is_fisheye_vec[k])
                    fisheye::solvePnP(obj_points_valid[k][cnt_valid_frame], img_points_valid[k][cnt_valid_frame], Ks[k], distortions[k], rvecs_all[k][i], tvecs_all[k][i], false, SOLVEPNP_ITERATIVE);
                else
                    solvePnP(obj_points_valid[k][cnt_valid_frame], img_points_valid[k][cnt_valid_frame], Ks[k], distortions[k], rvecs_all[k][i], tvecs_all[k][i], false, SOLVEPNP_ITERATIVE);

                // TODO: Add reprojection error check after solvePnP

                const double err2 = multiview::computeReprojectionMSE(obj_points_valid[k][cnt_valid_frame], img_points_valid[k][cnt_valid_frame],
                                                                      Ks[k], distortions[k], Mat(rvecs_all[k][i]), Mat(tvecs_all[k][i]),
                                                                      noArray(), noArray(), is_fisheye_vec[k]);
                if (camera_rt_errors[i] > err2) {
                    camera_rt_errors[i] = err2;
                    camera_rt_best[i] = k;
                }
                cnt_valid_frame++;
            }
        }
    }

    std::vector<std::vector<bool>> is_valid_angle2pattern;
    multiview::thresholdPatternCameraAngles(NUM_PATTERN_PTS, THR_PATTERN_CAMERA_ANGLES, objPoints_norm, rvecs_all, opt_axes, is_valid_angle2pattern);

    std::vector<Matx33d> Rs_vec(NUM_CAMERAS);
    std::vector<Vec3d> Ts_vec(NUM_CAMERAS);
    Rs_vec[0] = Matx33d::eye();
    Ts_vec[0] = Vec3d::zeros();

    if (useExtrinsicsGuess) {
        for (int k = 1; k < NUM_CAMERAS; k++) {
            Rs.getMat(k).copyTo(Rs_vec[k]);
            Ts.getMat(k).copyTo(Ts_vec[k]);
            Ts_vec[k] /= scale_3d_pts;
        }
    }

    if (!useExtrinsicsGuess) {
        std::vector<int> parent;
        std::vector<std::vector<int>> overlaps;
        if (! multiview::maximumSpanningTree(NUM_CAMERAS, NUM_FRAMES, detection_mask_mat, parent, overlaps, opt_axes,
                is_valid_angle2pattern, points_ratio_area, .5, 1.0)) {
            // failed to find suitable pairs with constraints!
            CV_Error(Error::StsInternal, "Failed to build tree for stereo calibration.");
        }

        std::vector<std::pair<int,int>> pairs;
        multiview::selectPairsBFS (pairs, NUM_CAMERAS, parent);

        if ((int)pairs.size() != NUM_CAMERAS-1) {
            CV_Error(Error::StsInternal, "Failed to build tree for stereo calibration. Incorrect number of pairs.");
        }
        if (initializationPairs.needed()) {
            Mat pairs_mat = Mat_<int>(NUM_CAMERAS-1, 2);
            auto * pairs_ptr = (int *) pairs_mat.data;
            for (const auto &p : pairs) {
                (*pairs_ptr++) = p.first;
                (*pairs_ptr++) = p.second;
            }
            pairs_mat.copyTo(initializationPairs);
        }
        multiview::pairwiseCalibration(pairs, is_fisheye_vec, obj_points_valid, img_points_valid,
            overlaps, detection_mask_mat, Ks, distortions, Rs_vec, Ts_vec, useExtrinsicsGuess);
    }

    const int NUM_VALID_FRAMES = countNonZero(valid_frames);
    const int nparams = (NUM_VALID_FRAMES + NUM_CAMERAS - 1) * 6; // rvecs + tvecs (6)
    std::vector<double> param(nparams, 0.);

    // use found relative extrinsics to initialize parameters
    for (int c = 1; c < NUM_CAMERAS; c++) {
        Vec3d rvec;
        Rodrigues(Rs_vec[c], rvec);
        memcpy(&param[0]+(c-1)*6  , rvec.val, 3*sizeof(double));
        memcpy(&param[0]+(c-1)*6+3, Ts_vec[c].val, 3*sizeof(double));
    }

    // use found rvecs / tvecs or estimate them to initialize rest of parameters
    int cnt_valid_frame = 0;
    for (int i = 0; i < NUM_FRAMES; i++ ) {
        if (!valid_frames[i]) continue;
        Vec3d rvec_0, tvec_0;
        if (camera_rt_best[i] != 0) {
            // convert rvecs / tvecs from k-th camera to the first one

            // formulas for relative rotation / translation
            // R = R_k R0^T       => R_k = R R_0
            // t = t_k - R t_0    => t_k = t + R t_0

            // initial camera R_0 = I, t_0 = 0 is fixed to R(rvec_0) and tvec_0
            // R_0 = R(rvec_0)
            // t_0 = tvec_0

            // R'_k = R(rvec_k) = R_k R_0       => R_0 = R_k^T R(rvec_k)
            // t'_k = tvec_k = t_k + R_k t_0    => t_0 = R_k^T (tvec_k - t_k)
            const int rt_best_idx = camera_rt_best[i];
            Matx33d R_k;
            Rodrigues(rvecs_all[rt_best_idx][i], R_k);
            tvec_0 = Rs_vec[rt_best_idx].t() * (tvecs_all[rt_best_idx][i] - Ts_vec[rt_best_idx]);
            Rodrigues(Rs_vec[rt_best_idx].t() * R_k, rvec_0);
        } else {
            rvec_0 = rvecs_all[0][i];
            tvec_0 = tvecs_all[0][i];
        }

        // save rvecs0 / tvecs0 parameters
        memcpy(&param[0]+(cnt_valid_frame+NUM_CAMERAS-1)*6  , rvec_0.val, 3*sizeof(double));
        memcpy(&param[0]+(cnt_valid_frame+NUM_CAMERAS-1)*6+3, tvec_0.val, 3*sizeof(double));
        cnt_valid_frame++;
    }

    TermCriteria termCrit (TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-12);
    const float RBS_FNC_SCALE = 30;
    multiview::RobustExpFunction robust_fnc(RBS_FNC_SCALE);
    multiview::optimizeLM(param, robust_fnc, termCrit, valid_frames, detection_mask_mat, obj_points_valid, img_points_valid, Ks, distortions, is_fisheye_vec);
    const auto * const params = &param[0];

    // extract extrinsics (R_i, t_i) for i = 1 ... NUM_CAMERAS:
    if (!useExtrinsicsGuess) {
        Rs.create(NUM_CAMERAS, 1, CV_64F);
        Ts.create(NUM_CAMERAS, 1, CV_64F);
    }
    for (int c = 0; c < NUM_CAMERAS; c++) {
        Mat r_store, t_store;
        if (!useExtrinsicsGuess) {
            Rs.create(3, 3, CV_64F, c, true);
            Ts.create(3, 1, CV_64F, c, true);
        }
        r_store.create(3, 1, CV_64F);
        t_store = Ts.getMat(c);
        if (c == 0) {
            memcpy(r_store.ptr(), Vec3d(0,0,0).val, 3*sizeof(double));
            memcpy(t_store.ptr(), Vec3d(0,0,0).val, 3*sizeof(double));
        } else {
            memcpy(r_store.ptr(), params + (c-1)*6, 3*sizeof(double));
            memcpy(t_store.ptr(), params + (c-1)*6+3, 3*sizeof(double)); // and de-normalize translation
            t_store *= scale_3d_pts;
        }
        Mat R = Rs.getMat(c);
        Rodrigues(r_store, R);
    }
    Mat rvecs0_, tvecs0_;
    if (rvecs0.needed() || perFrameErrors.needed()) {
        const bool is_mat_vec = rvecs0.needed() && rvecs0.isMatVector();
        if (is_mat_vec) {
            rvecs0.create(NUM_FRAMES, 1, CV_64F);
        } else {
            rvecs0_ = Mat_<double>(NUM_FRAMES, 3);
        }
        cnt_valid_frame = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!valid_frames[f]) continue;
            if (is_mat_vec)
                rvecs0.create(3, 1, CV_64F, f, true);
            Mat store = is_mat_vec ? rvecs0.getMat(f) : rvecs0_.row(f);
            memcpy(store.ptr(), params + (cnt_valid_frame + NUM_CAMERAS - 1)*6, 3*sizeof(double));
            cnt_valid_frame += 1;
        }
        if (!is_mat_vec && rvecs0.needed())
            rvecs0_.copyTo(rvecs0);
    }

    if (tvecs0.needed() || perFrameErrors.needed()) {
        const bool is_mat_vec = tvecs0.needed() && tvecs0.isMatVector();
        if (is_mat_vec) {
            tvecs0.create(NUM_FRAMES, 1, CV_64F);
        } else {
            tvecs0_ = Mat_<double>(NUM_FRAMES, 3);
        }
        cnt_valid_frame = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!valid_frames[f]) continue;
            if (is_mat_vec)
                tvecs0.create(3, 1, CV_64F, f, true);
            Mat store = is_mat_vec ? tvecs0.getMat(f) : tvecs0_.row(f);
            memcpy(store.ptr(), params + (cnt_valid_frame + NUM_CAMERAS - 1)*6+3, 3*sizeof(double));
            store *= scale_3d_pts;
            cnt_valid_frame += 1;
        }
        if (!is_mat_vec && tvecs0.needed())
            tvecs0_.copyTo(tvecs0);
    }

    double sum_errors = 0, cnt_errors = 0;
    if (perFrameErrors.needed()) {
        const bool rvecs_mat_vec = rvecs0.needed() && rvecs0.isMatVector(), tvecs_mat_vec = tvecs0.needed() && tvecs0.isMatVector();
        Mat errs = Mat_<double>(NUM_CAMERAS, NUM_FRAMES);
        auto * errs_ptr = (double *) errs.data;
        for (int c = 0; c < NUM_CAMERAS; c++) {
            Mat rvec;
            Rodrigues(Rs.getMat(c), rvec);
            const Mat tvec = Ts.getMat(c);
            cnt_valid_frame = 0;
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (detection_mask_mat[c][f]) {
                    const Mat rvec0 = rvecs_mat_vec ? rvecs0.getMat(f) : rvecs0_.row(f).t();
                    const Mat tvec0 = tvecs_mat_vec ? tvecs0.getMat(f) : tvecs0_.row(f).t();
                    const double err2 = multiview::computeReprojectionMSE(obj_points_valid[c][cnt_valid_frame], img_points_valid[c][cnt_valid_frame], Ks[c],
                         distortions[c], rvec0, tvec0 / scale_3d_pts, rvec, tvec / scale_3d_pts, is_fisheye_vec[c]);
                    (*errs_ptr++) = sqrt(err2);
                    sum_errors += err2;
                    cnt_errors += 1;
                    cnt_valid_frame++;
                } else (*errs_ptr++) = -1.0;
            }
        }
        errs.copyTo(perFrameErrors);
    }

    return sqrt(sum_errors / cnt_errors);
}
}
