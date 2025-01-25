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
    Point2f mass_center;
    for (size_t i = 0; i < pts.size(); i++) {
        mass_center += pts[i];
    }
    mass_center /= (float)pts.size();

    for (const auto& fit_ellipse : {fitEllipse, fitEllipseAMS, fitEllipseDirect}) {
        const RotatedRect ellipse = fit_ellipse(pts);
        if (!check_pt_in_ellipse(mass_center, ellipse)) {
            return false;
        }
    }
    return true;
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

TEST(Imgproc_FitEllipse_HorizontalLine, accuracy) {
    vector<Point2f> pts({{-300, 100}, {-200, 100}, {-100, 100}, {0, 100}, {100, 100}, {200, 100}, {300, 100}});
    const RotatedRect el = fitEllipse(pts);

    EXPECT_NEAR(el.center.x, 0, 100);
    EXPECT_NEAR(el.center.y, 100, 1);
    EXPECT_NEAR(el.size.width, 2, 2);
    EXPECT_NEAR(el.size.height, 600, 100);
    EXPECT_NEAR(el.angle, 90, 0.1);
}

TEST(Imgproc_FitEllipse_Issue_26078, accuracy) {
    vector<Point2f> pts({
        {1434, 308}, {1434, 309}, {1433, 310}, {1427, 310}, {1427, 312}, {1426, 313}, {1422, 313}, {1422, 314},
        {1421, 315}, {1415, 315}, {1415, 316}, {1414, 317}, {1408, 317}, {1408, 319}, {1407, 320}, {1403, 320},
        {1403, 321}, {1402, 322}, {1396, 322}, {1396, 323}, {1395, 324}, {1389, 324}, {1389, 326}, {1388, 327},
        {1382, 327}, {1382, 328}, {1381, 329}, {1376, 329}, {1376, 330}, {1375, 331}, {1369, 331}, {1369, 333},
        {1368, 334}, {1362, 334}, {1362, 335}, {1361, 336}, {1359, 336}, {1359, 1016}, {1365, 1016}, {1366, 1017},
        {1366, 1019}, {1430, 1019}, {1430, 1017}, {1431, 1016}, {1440, 1016}, {1440, 308},
    });

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

TEST(Imgproc_FitEllipse_Issue_26360_1, accuracy) {
    vector<Point2f> pts({
        {37, 111}, {37, 112}, {36, 113}, {37, 114}, {37, 115}, {37, 116}, {37, 117}, {37, 118},
        {36, 119}, {35, 120}, {34, 120}, {33, 121}, {32, 121}, {31, 121}, {30, 122}, {29, 123},
        {28, 123}, {27, 124}, {26, 124}, {25, 125}, {25, 126}, {25, 127}, {25, 128}, {25, 129},
        {254, 121}, {255, 120}, {255, 119}, {256, 118}, {256, 117}, {256, 116}, {256, 115}, {256, 114},
        {256, 113}, {256, 112}, {256, 111}, {256, 110}, {256, 109}, {256, 108}, {256, 107}, {257, 106},
        {257, 105}, {256, 104}, {257, 103}, {257, 102}, {257, 101},
    });

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

TEST(Imgproc_FitEllipse_Issue_26360_2, accuracy) {
    vector<Point2f> pts({
        {37, 105}, {38, 106}, {38, 107}, {25, 129}, {254, 121}, {257, 103}, {257, 102}, {257, 101},
    });

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

TEST(Imgproc_FitEllipse_Issue_26360_3, accuracy) {
    vector<Point2f> pts({
        {37, 105}, {38, 106}, {38, 107}, {257, 103}, {257, 102}, {257, 101},
    });

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

TEST(Imgproc_FitEllipse_Issue_26360_4, accuracy) {
    vector<Point2f> pts({
        {30, 105}, {25, 110}, {30, 115}, {250, 105}, {255, 110}, {250, 115},
    });

    EXPECT_TRUE(fit_and_check_ellipse(pts));
}

}} // namespace
