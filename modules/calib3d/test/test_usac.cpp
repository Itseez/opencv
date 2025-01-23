// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

enum TestSolver { Homogr, Fundam, Essen, PnP, Affine, SE2, SIM2, SO3, SE3, SIM3, };
/*
* rng -- reference to random generator
* pts1 -- 2xN image points
* pts2 -- for PnP is 3xN object points, otherwise 2xN image points.
* two_calib -- True if two cameras have different calibration.
* K1 -- intrinsic matrix of the first camera. For PnP only one camera.
* K2 -- only if two_calib is True.
* pts_size -- required size of points.
* inlier_ratio -- required inlier ratio
* noise_std -- standard deviation of Gaussian noise of image points.
* gt_inliers -- has size of number of inliers. Contains indices of inliers.
*/
static int generatePoints (cv::RNG &rng, cv::Mat &pts1, cv::Mat &pts2, cv::Mat &K1, cv::Mat &K2,
                    bool two_calib, int pts_size, TestSolver test_case, double inlier_ratio, double noise_std,
                    std::vector<int> &gt_inliers) {

    auto eulerAnglesToRotationMatrix = [] (double pitch, double yaw, double roll) {
        // Calculate rotation about x axis
        cv::Matx33d R_x (1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
        // Calculate rotation about y axis
        cv::Matx33d R_y (cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));
        // Calculate rotation about z axis
        cv::Matx33d R_z (cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);
        return cv::Mat(R_z * R_y * R_x); // Combined rotation matrix
    };

    const double pitch_min = -CV_PI / 6, pitch_max = CV_PI / 6; // 30 degrees
    const double yaw_min = -CV_PI / 6, yaw_max = CV_PI / 6;
    const double roll_min = -CV_PI / 6, roll_max = CV_PI / 6;

    cv::Mat R = eulerAnglesToRotationMatrix(rng.uniform(pitch_min, pitch_max),
            rng.uniform(yaw_min, yaw_max), rng.uniform(roll_min, roll_max));

    // generate random translation,
    // if test for homography fails try to fix translation to zero vec so H is related by transl.
    cv::Vec3d t (rng.uniform(-0.5f, 0.5f), rng.uniform(-0.5f, 0.5f), rng.uniform(1.0f, 2.0f));

    // generate random calibration
    auto getRandomCalib = [&] () {
        return cv::Mat(cv::Matx33d(rng.uniform(100.0, 1000.0), 0, rng.uniform(100.0, 100.0),
                       0, rng.uniform(100.0, 1000.0), rng.uniform(-100.0, 100.0),
                       0, 0, 1.));
    };
    K1 = getRandomCalib();
    K2 = two_calib ? getRandomCalib() : K1.clone();

    auto updateTranslation = [] (const cv::Mat &pts, const cv::Mat &R_, cv::Vec3d &t_) {
        // Make sure the shape is in front of the camera
        cv::Mat points3d_transformed = R_ * pts + t_ * cv::Mat::ones(1, pts.cols, pts.type());
        double min_dist, max_dist;
        cv::minMaxIdx(points3d_transformed.row(2), &min_dist, &max_dist);
        if (min_dist < 0) t_(2) -= min_dist + 1.0;
    };

    // compute size of inliers and outliers
    const int inl_size = static_cast<int>(inlier_ratio * pts_size);
    const int out_size = pts_size - inl_size;

    // all points will have top 'inl_size' of their points inliers
    gt_inliers.clear(); gt_inliers.reserve(inl_size);
    for (int i = 0; i < inl_size; i++)
        gt_inliers.emplace_back(i);

    // double precision to multiply points by models
    const int pts_type = CV_64F;
    cv::Mat points3d;
    if (test_case == TestSolver::Homogr) {
        points3d.create(2, inl_size, pts_type);
        rng.fill(points3d, cv::RNG::UNIFORM, 0.0, 1.0); // keep small range
        // inliers must be planar points, let their 3D coordinate be 1
        cv::vconcat(points3d, cv::Mat::ones(1, inl_size, points3d.type()), points3d);
    } else if (test_case == TestSolver::Fundam || test_case == TestSolver::Essen) {
        // create 3D points which are inliers
        points3d.create(3, inl_size, pts_type);
        rng.fill(points3d, cv::RNG::UNIFORM, 0.0, 1.0);
    } else if (test_case == TestSolver::PnP) {
        //pts1 are image points, pts2 are object points
        pts2.create(3, inl_size, pts_type); // 3D inliers
        rng.fill(pts2, cv::RNG::UNIFORM, 0, 1);

        updateTranslation(pts2, R, t);

        // project 3D points (pts2) on image plane (pts1)
        pts1 = K1 * (R * pts2 + t * cv::Mat::ones(1, pts2.cols, pts2.type()));
        cv::divide(pts1.row(0), pts1.row(2), pts1.row(0));
        cv::divide(pts1.row(1), pts1.row(2), pts1.row(1));
        // make 2D points
        pts1 = pts1.rowRange(0, 2);

        // create random outliers
        cv::Mat pts_outliers = cv::Mat(5, out_size, pts2.type());
        rng.fill(pts_outliers, cv::RNG::UNIFORM, 0, 1000);

        // merge inliers with random image points = outliers
        cv::hconcat(pts1, pts_outliers.rowRange(0, 2), pts1);
        // merge 3D inliers with 3D outliers
        cv::hconcat(pts2, pts_outliers.rowRange(2, 5), pts2);

        // add Gaussian noise to image points
        cv::Mat noise(pts1.rows, pts1.cols, pts1.type());
        rng.fill(noise, cv::RNG::NORMAL, 0, noise_std);
        pts1 += noise;
        return inl_size;
    } else if (test_case == TestSolver::Affine ||
               test_case == TestSolver::SE2 || test_case == TestSolver::SIM2 ||
               test_case == TestSolver::SO3 || test_case == TestSolver::SE3 || test_case == TestSolver::SIM3) {
    } else
        CV_Error(cv::Error::StsBadArg, "Unknown solver!");

    if (test_case != TestSolver::PnP) {
        // project 3D point on image plane
        // use two relative scenes. The first camera is P1 = K1 [I | 0], the second P2 = K2 [R | t]
        if (test_case == TestSolver::SE2 || test_case == TestSolver::SIM2) {
            pts1 = cv::Mat(2, inl_size, pts_type);
            rng.fill(pts1, cv::RNG::UNIFORM, 0, 1000);
            cv::Mat sc = cv::Mat::eye(3, 3, pts_type);
            if (test_case == TestSolver::SIM2) {
                sc.at<double>(0,0) = sc.at<double>(1,1) = rng.uniform(1., 5.);
            }
            cv::Matx33d tr(1,0,rng.uniform(50., 500.),0,1,rng.uniform(50., 500.), 0, 0, 1);
            const double phi = rng.uniform(0., CV_PI);
            cv::Matx33d rot(cos(phi), -sin(phi),0, sin(phi), cos(phi),0, 0, 0, 1);
            cv::Mat A = sc * tr * rot;
            cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), points3d);
            pts2 = A * points3d;
            // get 2D points
            pts1 = pts1.rowRange(0,2); pts2 = pts2.rowRange(0,2);
        } else if (test_case == TestSolver::SO3 || test_case == TestSolver::SE3 || test_case == TestSolver::SIM3) {
            pts1 = cv::Mat(3, inl_size, pts_type);
            rng.fill(pts1, cv::RNG::UNIFORM, 0, 1000);
            cv::Mat sc = cv::Mat::eye(4, 4, pts_type);
            if (test_case == TestSolver::SIM3) {
                sc.at<double>(0,0) = sc.at<double>(1,1) = sc.at<double>(2,2) = rng.uniform(1., 5.);
            }
            cv::Matx44d tr(1,0,0,rng.uniform(50., 500.), 0,1,0,rng.uniform(50., 500.), 0,0,1,rng.uniform(50., 500.), 0,0,0,1);
            cv::Mat rot = cv::Mat::eye(4, 4, pts_type);
            cv::Matx31d rvec(rng.uniform(0., CV_PI), rng.uniform(0., CV_PI), rng.uniform(0., CV_PI));
            cv::Rodrigues(rvec, rot(cv::Rect(0,0,3,3)));
            cv::Mat A;
            if (test_case == TestSolver::SO3) {
                A = rot;
            } else {
                A = sc * tr * rot;
            }
            cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), points3d);
            pts2 = A * points3d;
            // get 3D points
            pts1 = pts1.rowRange(0,3); pts2 = pts2.rowRange(0,3);
        } else if (test_case != TestSolver::Affine) {
            updateTranslation(points3d, R, t);

            pts1 = K1 * points3d;
            pts2 = K2 * (R * points3d + t * cv::Mat::ones(1, points3d.cols, points3d.type()));

            // normalize by 3 coordinate
            cv::divide(pts1.row(0), pts1.row(2), pts1.row(0));
            cv::divide(pts1.row(1), pts1.row(2), pts1.row(1));
            cv::divide(pts2.row(0), pts2.row(2), pts2.row(0));
            cv::divide(pts2.row(1), pts2.row(2), pts2.row(1));
            // get 2D points
            pts1 = pts1.rowRange(0,2); pts2 = pts2.rowRange(0,2);
        } else {
            // TestSolver::Affine
            pts1 = cv::Mat(2, inl_size, pts_type);
            rng.fill(pts1, cv::RNG::UNIFORM, 0, 1000);
            cv::Matx33d sc(rng.uniform(1., 5.),0,0,rng.uniform(1., 4.),0,0, 0, 0, 1);
            cv::Matx33d tr(1,0,rng.uniform(50., 500.),0,1,rng.uniform(50., 500.), 0, 0, 1);
            const double phi = rng.uniform(0., CV_PI);
            cv::Matx33d rot(cos(phi), -sin(phi),0, sin(phi), cos(phi),0, 0, 0, 1);
            cv::Matx33d A = sc * tr * rot;
            cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), points3d);
            pts2 = A * points3d;
            // get 2D points
            pts1 = pts1.rowRange(0,2); pts2 = pts2.rowRange(0,2);
        }

        // generate random outliers as 2D image points
        cv::Mat pts1_outliers(pts1.rows, out_size, pts1.type()),
                pts2_outliers(pts2.rows, out_size, pts2.type());
        rng.fill(pts1_outliers, cv::RNG::UNIFORM, 0, 1000);
        rng.fill(pts2_outliers, cv::RNG::UNIFORM, 0, 1000);
        // merge inliers and outliers
        cv::hconcat(pts1, pts1_outliers, pts1);
        cv::hconcat(pts2, pts2_outliers, pts2);

        // add normal / Gaussian noise to image points
        cv::Mat noise1 (pts1.rows, pts1.cols, pts1.type()), noise2 (pts2.rows, pts2.cols, pts2.type());
        rng.fill(noise1, cv::RNG::NORMAL, 0, noise_std); pts1 += noise1;
        rng.fill(noise2, cv::RNG::NORMAL, 0, noise_std); pts2 += noise2;
    }

    return inl_size;
}

/*
* for test case = 0, 1, 2 (homography and epipolar geometry): pts1 and pts2 are 3xN
* for test_case = 3 (PnP): pts1 are 3xN and pts2 are 4xN
* all points are of the same type as model
*/
static double getError (TestSolver test_case, int pt_idx, const cv::Mat &pts1, const cv::Mat &pts2, const cv::Mat &model) {
    cv::Mat pt1 = pts1.col(pt_idx), pt2 = pts2.col(pt_idx);
    if (test_case == TestSolver::Homogr) { // reprojection error
        // compute Euclidean distance between given and reprojected points
        cv::Mat est_pt2 = model * pt1; est_pt2 /= est_pt2.at<double>(est_pt2.rows-1);
        if (false) {
            cv::Mat est_pt1 = model.inv() * pt2; est_pt1 /= est_pt1.at<double>(est_pt1.rows-1);
            return (cv::norm(est_pt1 - pt1) + cv::norm(est_pt2 - pt2)) / 2;
        }
        return cv::norm(est_pt2 - pt2);
    } else
    if (test_case == TestSolver::Fundam || test_case == TestSolver::Essen) {
        cv::Mat l2 = model     * pt1;
        cv::Mat l1 = model.t() * pt2;
        // Sampson error
        return fabs(pt2.dot(l2)) / sqrt(pow(l1.at<double>(0), 2) + pow(l1.at<double>(1), 2) +
                                  pow(l2.at<double>(0), 2) + pow(l2.at<double>(1), 2));
    } else
    if (test_case == TestSolver::PnP) { // PnP, reprojection error
        cv::Mat img_pt = model * pt2; img_pt /= img_pt.at<double>(2);
        return cv::norm(pt1 - img_pt);
    } else
        CV_Error(cv::Error::StsBadArg, "Undefined test case!");
}

/*
* inl_size -- number of ground truth inliers
* pts1 and pts2 are of the same size as from function generatePoints(...)
*/
static void checkInliersMask (TestSolver test_case, int inl_size, double thr, const cv::Mat &pts1_,
                       const cv::Mat &pts2_, const cv::Mat &model, const cv::Mat &mask) {
    ASSERT_TRUE(!model.empty() && !mask.empty());

    cv::Mat pts1 = pts1_, pts2 = pts2_;
    if (pts1.type() != model.type()) {
        pts1.convertTo(pts1, model.type());
        pts2.convertTo(pts2, model.type());
    }
    // convert to homogeneous
    cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), pts1);
    cv::vconcat(pts2, cv::Mat::ones(1, pts2.cols, pts2.type()), pts2);

    thr *= 1.001; // increase a little threshold due to numerical imprecisions
    const auto * const mask_ptr = mask.ptr<uchar>();
    int num_found_inliers = 0;
    for (int i = 0; i < pts1.cols; i++)
        if (mask_ptr[i]) {
            ASSERT_LT(getError(test_case, i, pts1, pts2, model), thr);
            num_found_inliers++;
        }
    // check if RANSAC found at least 80% of inliers
    ASSERT_GT(num_found_inliers, 0.8 * inl_size);
}

TEST(usac_Homography, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 1500;
    cv::RNG &rng = cv::theRNG();
    // do not test USAC_PARALLEL, because it is not deterministic
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
           pts_size, TestSolver ::Homogr, inl_ratio/*inl ratio*/, 0.1 /*noise std*/, gt_inliers);
        // compute max_iters with standard upper bound rule for RANSAC with 1.5x tolerance
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                 log(1 - pow(inl_ratio, 4 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, H = cv::findHomography(pts1, pts2,flag, thr, mask,
                                                       int(max_iters), conf);
            checkInliersMask(TestSolver::Homogr, inl_size, thr, pts1, pts2, H, mask);
        }
    }
}

TEST(usac_Fundamental, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::RNG &rng = cv::theRNG();
    // start from 25% otherwise max_iters will be too big
    const std::vector<int> flags = {USAC_DEFAULT, USAC_FM_8PTS, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    const double conf = 0.99, thr = 1.;
    for (double inl_ratio = 0.25; inl_ratio < 0.91; inl_ratio += 0.1) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
          pts_size, TestSolver ::Fundam, inl_ratio, 0.1 /*noise std*/, gt_inliers);

        for (auto flag : flags) {
            const int sample_size = flag == USAC_FM_8PTS ? 8 : 7;
            const double max_iters = 1.25 * log(1 - conf) /
                    log(1 - pow(inl_ratio, sample_size));
            cv::Mat mask, F = cv::findFundamentalMat(pts1, pts2,flag, thr, conf,
                                                           int(max_iters), mask);
            checkInliersMask(TestSolver::Fundam, inl_size, thr, pts1, pts2, F, mask);
        }
    }
}

TEST(usac_Fundamental, regression_19639)
{
    double x_[] = {
        941, 890,
        596, 940,
        898, 941,
        894, 933,
        586, 938,
        902, 933,
        887, 935
    };
    Mat x(7, 1, CV_64FC2, x_);

    double y_[] = {
        1416,  806,
        1157,  852,
        1380,  855,
        1378,  843,
        1145,  849,
        1378,  843,
        1378,  843
    };
    Mat y(7, 1, CV_64FC2, y_);

    //std::cout << x << std::endl;
    //std::cout << y << std::endl;

    Mat m = cv::findFundamentalMat(x, y, USAC_MAGSAC, 3, 0.99);
    EXPECT_TRUE(m.empty());
}

CV_ENUM(UsacMethod, USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC)
typedef TestWithParam<UsacMethod> usac_Essential;

TEST_P(usac_Essential, accuracy) {
    int method = GetParam();
    std::vector<int> gt_inliers;
    const int pts_size = 1500;
    cv::RNG &rng = cv::theRNG();
    // findEssentilaMat has by default number of maximum iterations equal to 1000.
    // It means that with 99% confidence we assume at least 34.08% of inliers
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.35; inl_ratio < 0.91; inl_ratio += 0.1) {
        cv::Mat pts1, pts2, K1, K2;
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
          pts_size, TestSolver ::Fundam, inl_ratio, 0.01 /*noise std, works bad with high noise*/, gt_inliers);
        const double conf = 0.99, thr = 1.;
        cv::Mat mask, E;
        try {
            E = cv::findEssentialMat(pts1, pts2, K1, method, conf, thr, mask);
        } catch (cv::Exception &e) {
            if (e.code != cv::Error::StsNotImplemented)
                FAIL() << "Essential matrix estimation failed!\n";
            else continue;
        }
        // calibrate points
        cv::Mat cpts1_3d, cpts2_3d;
        cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), cpts1_3d);
        cv::vconcat(pts2, cv::Mat::ones(1, pts2.cols, pts2.type()), cpts2_3d);
        cpts1_3d = K1.inv() * cpts1_3d; cpts2_3d = K1.inv() * cpts2_3d;
        checkInliersMask(TestSolver::Essen, inl_size, thr / ((K1.at<double>(0,0) + K1.at<double>(1,1)) / 2),
                            cpts1_3d.rowRange(0,2), cpts2_3d.rowRange(0,2), E, mask);
    }
}

TEST_P(usac_Essential, maxiters) {
    int method = GetParam();
    cv::RNG &rng = cv::theRNG();
    cv::Mat mask;
    cv::Mat K1 = cv::Mat(cv::Matx33d(1, 0, 0,
                                     0, 1, 0,
                                     0, 0, 1.));
    const double conf = 0.99, thr = 0.25;
    int roll_results_sum = 0;

    for (int iters = 0; iters < 10; iters++) {
        cv::Mat E1, E2;
        try {
            cv::Mat pts1 = cv::Mat(2, 50, CV_64F);
            cv::Mat pts2 = cv::Mat(2, 50, CV_64F);
            rng.fill(pts1, cv::RNG::UNIFORM, 0.0, 1.0);
            rng.fill(pts2, cv::RNG::UNIFORM, 0.0, 1.0);

            E1 = cv::findEssentialMat(pts1, pts2, K1, method, conf, thr, 1, mask);
            E2 = cv::findEssentialMat(pts1, pts2, K1, method, conf, thr, 1000, mask);

            if (E1.dims != E2.dims) { continue; }
            roll_results_sum += cv::norm(E1, E2, NORM_L1) != 0;
        } catch (cv::Exception &e) {
            if (e.code != cv::Error::StsNotImplemented)
                FAIL() << "Essential matrix estimation failed!\n";
            else continue;
        }
    }
    EXPECT_NE(roll_results_sum, 0);
}

INSTANTIATE_TEST_CASE_P(Calib3d, usac_Essential, UsacMethod::all());

TEST(usac_P3P, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 3000;
    cv::Mat img_pts, obj_pts, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, img_pts, obj_pts, K1, K2, false /*two calib*/,
                                      pts_size, TestSolver ::PnP, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                   log(1 - pow(inl_ratio, 3 /* sample size */));

        for (auto flag : flags) {
            std::vector<int> inliers;
            cv::Mat rvec, tvec, mask, R, P;
            CV_Assert(cv::solvePnPRansac(obj_pts, img_pts, K1, cv::noArray(), rvec, tvec,
                    false, (int)max_iters, (float)thr, conf, inliers, flag));
            cv::Rodrigues(rvec, R);
            cv::hconcat(K1 * R, K1 * tvec, P);
            mask.create(pts_size, 1, CV_8U);
            mask.setTo(Scalar::all(0));
            for (auto inl : inliers)
                mask.at<uchar>(inl) = true;
            checkInliersMask(TestSolver ::PnP, inl_size, thr, img_pts, obj_pts, P, mask);
        }
    }
}

TEST (usac_Affine2D, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::Mat pts1, pts2, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
                  pts_size, TestSolver ::Affine, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, A = cv::estimateAffine2D(pts1, pts2, mask, flag, thr, (size_t)max_iters, conf, 0);
            cv::vconcat(A, cv::Mat(cv::Matx13d(0,0,1)), A);
            checkInliersMask(TestSolver::Homogr /*use homography error*/, inl_size, thr, pts1, pts2, A, mask);
        }
    }
}

TEST (usac_SE2, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::Mat pts1, pts2, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
                  pts_size, TestSolver ::SE2, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, A = cv::estimateSE2(pts1, pts2, mask, flag, thr, (size_t)max_iters, conf, 0);
            cv::vconcat(A, cv::Mat(cv::Matx13d(0,0,1)), A);
            checkInliersMask(TestSolver::Homogr /*use homography error*/, inl_size, thr, pts1, pts2, A, mask);
        }
    }
}

TEST (usac_SIM2, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::Mat pts1, pts2, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
                  pts_size, TestSolver ::SIM2, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, A = cv::estimateSIM2(pts1, pts2, mask, flag, thr, (size_t)max_iters, conf, 0);
            cv::vconcat(A, cv::Mat(cv::Matx13d(0,0,1)), A);
            checkInliersMask(TestSolver::Homogr /*use homography error*/, inl_size, thr, pts1, pts2, A, mask);
        }
    }
}

TEST (usac_SO3, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::Mat pts1, pts2, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
                  pts_size, TestSolver ::SO3, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, A = cv::estimateSO3(pts1, pts2, mask, flag, thr, (size_t)max_iters, conf, 0);
            cv::vconcat(A, cv::Mat(cv::Matx14d(0,0,0,1)), A);
            checkInliersMask(TestSolver::Homogr /*use homography error*/, inl_size, thr, pts1, pts2, A, mask);
        }
    }
}

TEST (usac_SE3, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::Mat pts1, pts2, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
                  pts_size, TestSolver ::SE3, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, A = cv::estimateSE3(pts1, pts2, mask, flag, thr, (size_t)max_iters, conf, 0);
            cv::vconcat(A, cv::Mat(cv::Matx14d(0,0,0,1)), A);
            checkInliersMask(TestSolver::Homogr /*use homography error*/, inl_size, thr, pts1, pts2, A, mask);
        }
    }
}

TEST (usac_SIM3, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 2000;
    cv::Mat pts1, pts2, K1, K2;
    cv::RNG &rng = cv::theRNG();
    const std::vector<int> flags = {USAC_DEFAULT, USAC_ACCURATE, USAC_PROSAC, USAC_FAST, USAC_MAGSAC};
    for (double inl_ratio = 0.1; inl_ratio < 0.91; inl_ratio += 0.1) {
        int inl_size = generatePoints(rng, pts1, pts2, K1, K2, false /*two calib*/,
                  pts_size, TestSolver ::SIM3, inl_ratio, 0.15 /*noise std*/, gt_inliers);
        const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
        for (auto flag : flags) {
            cv::Mat mask, A = cv::estimateSIM3(pts1, pts2, mask, flag, thr, (size_t)max_iters, conf, 0);
            cv::vconcat(A, cv::Mat(cv::Matx14d(0,0,0,1)), A);
            checkInliersMask(TestSolver::Homogr /*use homography error*/, inl_size, thr, pts1, pts2, A, mask);
        }
    }
}

TEST(usac_testUsacParams, accuracy) {
    std::vector<int> gt_inliers;
    const int pts_size = 150000;
    cv::RNG &rng = cv::theRNG();
    const cv::UsacParams usac_params = cv::UsacParams();
    cv::Mat pts1, pts2, K1, K2, mask, model, rvec, tvec, R;
    int inl_size;
    auto getInlierRatio = [] (int max_iters, int sample_size, double conf) {
        return std::pow(1 - exp(log(1 - conf)/(double)max_iters), 1 / (double)sample_size);
    };
    cv::Vec4d dist_coeff (0, 0, 0, 0); // test with 0 distortion

    // Homography matrix
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::Homogr,
    getInlierRatio(usac_params.maxIterations, 4, usac_params.confidence), 0.1, gt_inliers);
    model = cv::findHomography(pts1, pts2, mask, usac_params);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // Fundamental matrix
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::Fundam,
    getInlierRatio(usac_params.maxIterations, 7, usac_params.confidence), 0.1, gt_inliers);
    model = cv::findFundamentalMat(pts1, pts2, mask, usac_params);
    checkInliersMask(TestSolver::Fundam, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // Essential matrix
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, true, pts_size, TestSolver::Essen,
    getInlierRatio(usac_params.maxIterations, 5, usac_params.confidence), 0.01, gt_inliers);
    try {
        model = cv::findEssentialMat(pts1, pts2, K1, K2, dist_coeff, dist_coeff, mask, usac_params);
        cv::Mat cpts1_3d, cpts2_3d;
        cv::vconcat(pts1, cv::Mat::ones(1, pts1.cols, pts1.type()), cpts1_3d);
        cv::vconcat(pts2, cv::Mat::ones(1, pts2.cols, pts2.type()), cpts2_3d);
        cpts1_3d = K1.inv() * cpts1_3d; cpts2_3d = K2.inv() * cpts2_3d;
        checkInliersMask(TestSolver::Essen, inl_size, usac_params.threshold /
        ((K1.at<double>(0,0) + K1.at<double>(1,1) + K2.at<double>(0,0) + K2.at<double>(1,1)) / 4),
        cpts1_3d.rowRange(0,2), cpts2_3d.rowRange(0,2), model, mask);
    } catch (cv::Exception &e) {
        if (e.code != cv::Error::StsNotImplemented)
            FAIL() << "Essential matrix estimation failed!\n";
            // CV_Error(cv::Error::StsError, "Essential matrix estimation failed!");
    }

    std::vector<int> inliers(pts_size);
    // P3P
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::PnP,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.01, gt_inliers);
    CV_Assert(cv::solvePnPRansac(pts2, pts1, K1, dist_coeff, rvec, tvec, inliers, usac_params));
    cv::Rodrigues(rvec, R); cv::hconcat(K1 * R, K1 * tvec, model);
    mask.create(pts_size, 1, CV_8U);
    mask.setTo(Scalar::all(0));
    for (auto inl : inliers)
        mask.at<uchar>(inl) = true;
    checkInliersMask(TestSolver::PnP, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // P6P
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::PnP,
    getInlierRatio(usac_params.maxIterations, 6, usac_params.confidence), 0.1, gt_inliers);
    cv::Mat K_est;
    CV_Assert(cv::solvePnPRansac(pts2, pts1, K_est, dist_coeff, rvec, tvec, inliers, usac_params));
    cv::Rodrigues(rvec, R); cv::hconcat(K_est * R, K_est * tvec, model);
    mask.setTo(Scalar::all(0));
    for (auto inl : inliers)
        mask.at<uchar>(inl) = true;
    checkInliersMask(TestSolver::PnP, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // Affine2D
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::Affine,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.1, gt_inliers);
    model = cv::estimateAffine2D(pts1, pts2, mask, usac_params);
    cv::vconcat(model, cv::Mat(cv::Matx13d(0,0,1)), model);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // SE2
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::SE2,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.1, gt_inliers);
    model = cv::estimateSE2(pts1, pts2, mask, usac_params);
    cv::vconcat(model, cv::Mat(cv::Matx13d(0,0,1)), model);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // SIM2
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::SIM2,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.1, gt_inliers);
    model = cv::estimateSIM2(pts1, pts2, mask, usac_params);
    cv::vconcat(model, cv::Mat(cv::Matx13d(0,0,1)), model);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // SO3
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::SO3,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.1, gt_inliers);
    model = cv::estimateSO3(pts1, pts2, mask, usac_params);
    cv::vconcat(model, cv::Mat(cv::Matx14d(0,0,0,1)), model);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // SE3
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::SE3,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.1, gt_inliers);
    model = cv::estimateSE3(pts1, pts2, mask, usac_params);
    cv::vconcat(model, cv::Mat(cv::Matx14d(0,0,0,1)), model);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);

    // SIM3
    inl_size = generatePoints(rng, pts1, pts2, K1, K2, false, pts_size, TestSolver::SIM3,
    getInlierRatio(usac_params.maxIterations, 3, usac_params.confidence), 0.1, gt_inliers);
    model = cv::estimateSIM3(pts1, pts2, mask, usac_params);
    cv::vconcat(model, cv::Mat(cv::Matx14d(0,0,0,1)), model);
    checkInliersMask(TestSolver::Homogr, inl_size, usac_params.threshold, pts1, pts2, model, mask);
}

TEST(usac_solvePnPRansac, regression_21105) {
    std::vector<int> gt_inliers;
    const int pts_size = 100;
    double inl_ratio = 0.1;
    cv::Mat img_pts, obj_pts, K1, K2;
    cv::RNG &rng = cv::theRNG();
    generatePoints(rng, img_pts, obj_pts, K1, K2, false /*two calib*/,
                   pts_size, TestSolver ::PnP, inl_ratio, 0.15 /*noise std*/, gt_inliers);
    const double conf = 0.99, thr = 2., max_iters = 1.3 * log(1 - conf) /
                log(1 - pow(inl_ratio, 3 /* sample size */));
    const int flag = USAC_DEFAULT;
    std::vector<int> inliers;
    cv::Matx31d rvec, tvec;
    CV_Assert(cv::solvePnPRansac(obj_pts, img_pts, K1, cv::noArray(), rvec, tvec,
            false, (int)max_iters, (float)thr, conf, inliers, flag));

    cv::Mat zero_column = cv::Mat::zeros(3, 1, K1.type());
    cv::hconcat(K1, zero_column, K1);
    cv::Mat K1_copy = K1.colRange(0, 3);
    std::vector<int> inliers_copy;
    cv::Matx31d rvec_copy, tvec_copy;
    CV_Assert(cv::solvePnPRansac(obj_pts, img_pts, K1_copy, cv::noArray(), rvec_copy, tvec_copy,
              false, (int)max_iters, (float)thr, conf, inliers_copy, flag));
    EXPECT_EQ(rvec, rvec_copy);
    EXPECT_EQ(tvec, tvec_copy);
    EXPECT_EQ(inliers, inliers_copy);
}

}}  // namespace
