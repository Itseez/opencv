// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2016, Itseez, Inc, all rights reserved.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// return true if point lies inside ellipse
static bool check_pt_in_ellipse(const Point2f& pt, const RotatedRect& el) {
    Point2f to_pt = pt - el.center;
    double pt_angle = atan2(to_pt.y, to_pt.x);
    double el_angle = el.angle * CV_PI / 180;
    double x_dist = 0.5 * el.size.width * cos(pt_angle + el_angle);
    double y_dist = 0.5 * el.size.height * sin(pt_angle + el_angle);
    double el_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    return cv::norm(to_pt) < el_dist;
}

// Return true if mass center of fitted points lies inside ellipse
static bool fit_and_check_ellipse(const vector<Point2f>& pts) {
    RotatedRect ellipse = fitEllipseDirect(pts); // fitEllipseAMS() also works fine

    Point2f mass_center;
    for (size_t i = 0; i < pts.size(); i++) {
        mass_center += pts[i];
    }
    mass_center /= (float)pts.size();

    return check_pt_in_ellipse(mass_center, ellipse);
}

TEST(Imgproc_FitEllipse_Issue_4515, accuracy) {
    vector<Point2f> pts;
    pts.push_back(Point2f(327, 317));
    pts.push_back(Point2f(328, 316));
    pts.push_back(Point2f(329, 315));
    pts.push_back(Point2f(330, 314));
    pts.push_back(Point2f(331, 314));
    pts.push_back(Point2f(332, 314));
    pts.push_back(Point2f(333, 315));
    pts.push_back(Point2f(333, 316));
    pts.push_back(Point2f(333, 317));
    pts.push_back(Point2f(333, 318));
    pts.push_back(Point2f(333, 319));
    pts.push_back(Point2f(333, 320));

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

TEST(Imgproc_FitEllipse_Issue_6544, accuracy) {
    vector<Point2f> pts;
    pts.push_back(Point2f(924.784f, 764.160f));
    pts.push_back(Point2f(928.388f, 615.903f));
    pts.push_back(Point2f(847.4f,   888.014f));
    pts.push_back(Point2f(929.406f, 741.675f));
    pts.push_back(Point2f(904.564f, 825.605f));
    pts.push_back(Point2f(926.742f, 760.746f));
    pts.push_back(Point2f(863.479f, 873.406f));
    pts.push_back(Point2f(910.987f, 808.863f));
    pts.push_back(Point2f(929.145f, 744.976f));
    pts.push_back(Point2f(917.474f, 791.823f));

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

TEST(Imgproc_FitEllipse_Issue_10270, accuracy) {
    vector<Point2f> pts;
    float scale = 1;
    Point2f shift(0, 0);
    pts.push_back(Point2f(0, 1)*scale+shift);
    pts.push_back(Point2f(0, 2)*scale+shift);
    pts.push_back(Point2f(0, 3)*scale+shift);
    pts.push_back(Point2f(2, 3)*scale+shift);
    pts.push_back(Point2f(0, 4)*scale+shift);

    // check that we get almost vertical ellipse centered around (1, 3)
    RotatedRect e = fitEllipse(pts);
    EXPECT_LT(std::min(fabs(e.angle-180), fabs(e.angle)), 10.);
    EXPECT_NEAR(e.center.x, 1, 1);
    EXPECT_NEAR(e.center.y, 3, 1);
    EXPECT_LT(e.size.width*3, e.size.height);
}

TEST(Imgproc_FitEllipse_JavaCase, accuracy) {
    vector<Point2f> pts;
    float scale = 1;
    Point2f shift(0, 0);
    pts.push_back(Point2f(0, 0)*scale+shift);
    pts.push_back(Point2f(1, 1)*scale+shift);
    pts.push_back(Point2f(-1, 1)*scale+shift);
    pts.push_back(Point2f(-1, -1)*scale+shift);
    pts.push_back(Point2f(1, -1)*scale+shift);

    // check that we get almost circle centered around (0, 0)
    RotatedRect e = fitEllipse(pts);
    EXPECT_NEAR(e.center.x, 0, 0.01);
    EXPECT_NEAR(e.center.y, 0, 0.01);
    EXPECT_NEAR(e.size.width, sqrt(2.)*2, 0.4);
    EXPECT_NEAR(e.size.height, sqrt(2.)*2, 0.4);
}

template<typename T>
static float get_ellipse_fitting_error(const std::vector<T>& points, const Mat& closest_points) {
    float mse = 0.0f;
    for (int i = 0; i < static_cast<int>(points.size()); i++)
    {
        Point2f pt_err = Point2f(static_cast<float>(points[i].x), static_cast<float>(points[i].y)) - closest_points.at<Point2f>(i);
        mse += pt_err.x*pt_err.x + pt_err.y*pt_err.y;
    }
    return mse / points.size();
}

TEST(Imgproc_getClosestEllipsePoints, ellipse_mse) {
    // https://github.com/opencv/opencv/issues/26078
    std::vector<Point2i> points_list;

    // [1434, 308], [1434, 309], [1433, 310], [1427, 310], [1427, 312], [1426, 313], [1422, 313], [1422, 314],
    points_list.push_back(Point2i(1434, 308));
    points_list.push_back(Point2i(1434, 309));
    points_list.push_back(Point2i(1433, 310));
    points_list.push_back(Point2i(1427, 310));
    points_list.push_back(Point2i(1427, 312));
    points_list.push_back(Point2i(1426, 313));
    points_list.push_back(Point2i(1422, 313));
    points_list.push_back(Point2i(1422, 314));

    // [1421, 315], [1415, 315], [1415, 316], [1414, 317], [1408, 317], [1408, 319], [1407, 320], [1403, 320],
    points_list.push_back(Point2i(1421, 315));
    points_list.push_back(Point2i(1415, 315));
    points_list.push_back(Point2i(1415, 316));
    points_list.push_back(Point2i(1414, 317));
    points_list.push_back(Point2i(1408, 317));
    points_list.push_back(Point2i(1408, 319));
    points_list.push_back(Point2i(1407, 320));
    points_list.push_back(Point2i(1403, 320));

    // [1403, 321], [1402, 322], [1396, 322], [1396, 323], [1395, 324], [1389, 324], [1389, 326], [1388, 327],
    points_list.push_back(Point2i(1403, 321));
    points_list.push_back(Point2i(1402, 322));
    points_list.push_back(Point2i(1396, 322));
    points_list.push_back(Point2i(1396, 323));
    points_list.push_back(Point2i(1395, 324));
    points_list.push_back(Point2i(1389, 324));
    points_list.push_back(Point2i(1389, 326));
    points_list.push_back(Point2i(1388, 327));

    // [1382, 327], [1382, 328], [1381, 329], [1376, 329], [1376, 330], [1375, 331], [1369, 331], [1369, 333],
    points_list.push_back(Point2i(1382, 327));
    points_list.push_back(Point2i(1382, 328));
    points_list.push_back(Point2i(1381, 329));
    points_list.push_back(Point2i(1376, 329));
    points_list.push_back(Point2i(1376, 330));
    points_list.push_back(Point2i(1375, 331));
    points_list.push_back(Point2i(1369, 331));
    points_list.push_back(Point2i(1369, 333));

    // [1368, 334], [1362, 334], [1362, 335], [1361, 336], [1359, 336], [1359, 1016], [1365, 1016], [1366, 1017],
    points_list.push_back(Point2i(1368, 334));
    points_list.push_back(Point2i(1362, 334));
    points_list.push_back(Point2i(1362, 335));
    points_list.push_back(Point2i(1361, 336));
    points_list.push_back(Point2i(1359, 336));
    points_list.push_back(Point2i(1359, 1016));
    points_list.push_back(Point2i(1365, 1016));
    points_list.push_back(Point2i(1366, 1017));

    // [1366, 1019], [1430, 1019], [1430, 1017], [1431, 1016], [1440, 1016], [1440, 308]
    points_list.push_back(Point2i(1366, 1019));
    points_list.push_back(Point2i(1430, 1019));
    points_list.push_back(Point2i(1430, 1017));
    points_list.push_back(Point2i(1431, 1016));
    points_list.push_back(Point2i(1440, 1016));
    points_list.push_back(Point2i(1440, 308));

    RotatedRect fit_ellipse_params(
        Point2f(1442.97900390625, 662.1879272460938),
        Size2f(579.5570678710938, 730.834228515625),
        20.190902709960938
    );

    // Point2i
    {
        Mat pointsi(points_list);
        Mat closest_pts;
        getClosestEllipsePoints(fit_ellipse_params, pointsi, closest_pts);
        EXPECT_TRUE(pointsi.rows == closest_pts.rows);
        EXPECT_TRUE(pointsi.cols == closest_pts.cols);
        EXPECT_TRUE(pointsi.channels() == closest_pts.channels());

        float fit_ellipse_mse = get_ellipse_fitting_error(points_list, closest_pts);
        EXPECT_NEAR(fit_ellipse_mse, 1.61994, 1e-4);
    }

    // Point2f
    {
        Mat pointsf;
        Mat(points_list).convertTo(pointsf, CV_32F);

        Mat closest_pts;
        getClosestEllipsePoints(fit_ellipse_params, pointsf, closest_pts);
        EXPECT_TRUE(pointsf.rows == closest_pts.rows);
        EXPECT_TRUE(pointsf.cols == closest_pts.cols);
        EXPECT_TRUE(pointsf.channels() == closest_pts.channels());

        float fit_ellipse_mse = get_ellipse_fitting_error(points_list, closest_pts);
        EXPECT_NEAR(fit_ellipse_mse, 1.61994, 1e-4);
    }
}

static std::vector<Point2f> sample_ellipse_pts(const RotatedRect& ellipse_params) {
    // Sample N points using the ellipse parametric form
    float xc = ellipse_params.center.x;
    float yc = ellipse_params.center.y;
    float a = ellipse_params.size.width / 2;
    float b = ellipse_params.size.height / 2;
    float theta = static_cast<float>(ellipse_params.angle * M_PI / 180);

    float cos_th = std::cos(theta);
    float sin_th = std::sin(theta);
    int nb_samples = 180;
    std::vector<Point2f> ellipse_pts(nb_samples);
    for (int i = 0; i < nb_samples; i++) {
        float ax = a * cos_th;
        float ay = a * sin_th;
        float bx = -b * sin_th;
        float by = b * cos_th;

        float t = static_cast<float>(i / static_cast<float>(nb_samples) * 2*M_PI);
        float cos_t = std::cos(t);
        float sin_t = std::sin(t);

        ellipse_pts[i].x = xc + ax*cos_t + bx*sin_t;
        ellipse_pts[i].y = yc + ay*cos_t + by*sin_t;
    }

    return ellipse_pts;
}

TEST(Imgproc_getClosestEllipsePoints, ellipse_mse_2) {
    const float tol = 1e-3f;

    // bb height > width
    // Chech correctness of the minor/major axes swapping and updated angle in getClosestEllipsePoints
    {
        RotatedRect ellipse_params(
            Point2f(-142.97f, -662.1878f),
            Size2f(539.557f, 730.83f),
            27.09960938f
        );
        std::vector<Point2f> ellipse_pts = sample_ellipse_pts(ellipse_params);

        Mat pointsf, closest_pts;
        Mat(ellipse_pts).convertTo(pointsf, CV_32F);
        getClosestEllipsePoints(ellipse_params, pointsf, closest_pts);

        float ellipse_pts_mse = get_ellipse_fitting_error(ellipse_pts, closest_pts);
        EXPECT_NEAR(ellipse_pts_mse, 0, tol);
    }

    // bb height > width + negative angle
    {
        RotatedRect ellipse_params(
            Point2f(-142.97f, 562.1878f),
            Size2f(53.557f, 730.83f),
            -75.09960938f
        );
        std::vector<Point2f> ellipse_pts = sample_ellipse_pts(ellipse_params);

        Mat pointsf, closest_pts;
        Mat(ellipse_pts).convertTo(pointsf, CV_32F);
        getClosestEllipsePoints(ellipse_params, pointsf, closest_pts);

        float ellipse_pts_mse = get_ellipse_fitting_error(ellipse_pts, closest_pts);
        EXPECT_NEAR(ellipse_pts_mse, 0, tol);
    }

    // Negative angle
    {
        RotatedRect ellipse_params(
            Point2f(742.97f, -462.1878f),
            Size2f(535.57f, 130.83f),
            -75.09960938f
        );
        std::vector<Point2f> ellipse_pts = sample_ellipse_pts(ellipse_params);

        Mat pointsf, closest_pts;
        Mat(ellipse_pts).convertTo(pointsf, CV_32F);
        getClosestEllipsePoints(ellipse_params, pointsf, closest_pts);

        float ellipse_pts_mse = get_ellipse_fitting_error(ellipse_pts, closest_pts);
        EXPECT_NEAR(ellipse_pts_mse, 0, tol);
    }
}

}} // namespace
