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

}} // namespace
