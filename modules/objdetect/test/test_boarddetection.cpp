// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "test_precomp.hpp"
#include "test_aruco_utils.hpp"

namespace opencv_test { namespace {

enum class ArucoAlgParams
{
    USE_DEFAULT = 0,
    USE_ARUCO3 = 1
};

/**
 * @brief Check pose estimation of aruco board
 */
class CV_ArucoBoardPose : public cvtest::BaseTest {
    public:
    CV_ArucoBoardPose(ArucoAlgParams arucoAlgParams)
    {
        aruco::DetectorParameters params;
        aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
        params.minDistanceToBorder = 3;
        if (arucoAlgParams == ArucoAlgParams::USE_ARUCO3) {
            params.useAruco3Detection = true;
            params.cornerRefinementMethod = (int)aruco::CORNER_REFINE_SUBPIX;
            params.minSideLengthCanonicalImg = 16;
            params.errorCorrectionRate = 0.8;
        }
        detector = aruco::ArucoDetector(dictionary, params);
    }

    protected:
    aruco::ArucoDetector detector;
    void run(int);
};


void CV_ArucoBoardPose::run(int) {
    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    const int sizeX = 3, sizeY = 3;
    aruco::DetectorParameters detectorParameters = detector.getDetectorParameters();

    // for different perspectives
    for(double distance : {0.2, 0.35}) {
        for(int yaw = -55; yaw <= 50; yaw += 25) {
            for(int pitch = -55; pitch <= 50; pitch += 25) {
                vector<int> tmpIds;
                for(int i = 0; i < sizeX*sizeY; i++)
                    tmpIds.push_back((iter + int(i)) % 250);
                aruco::GridBoard gridboard(Size(sizeX, sizeY), 0.02f, 0.005f, detector.getDictionary(), tmpIds);
                int markerBorder = iter % 2 + 1;
                iter++;
                // create synthetic image
                Mat img = projectBoard(gridboard, cameraMatrix, deg2rad(yaw), deg2rad(pitch), distance,
                                       imgSize, markerBorder);
                vector<vector<Point2f> > corners;
                vector<int> ids;
                detectorParameters.markerBorderBits = markerBorder;
                detector.setDetectorParameters(detectorParameters);
                detector.detectMarkers(img, corners, ids);

                ASSERT_EQ(ids.size(), gridboard.getIds().size());

                // estimate pose
                Mat rvec, tvec;
                {
                    Mat objPoints, imgPoints; // get object and image points for the solvePnP function
                    gridboard.matchImagePoints(corners, ids, objPoints, imgPoints);
                    solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
                }

                // check axes
                vector<Point2f> axes = getAxis(cameraMatrix, distCoeffs, rvec, tvec, gridboard.getRightBottomCorner().x);
                vector<Point2f> topLeft = getMarkerById(gridboard.getIds()[0], corners, ids);
                ASSERT_NEAR(topLeft[0].x, axes[0].x, 2.f);
                ASSERT_NEAR(topLeft[0].y, axes[0].y, 2.f);
                vector<Point2f> topRight = getMarkerById(gridboard.getIds()[2], corners, ids);
                ASSERT_NEAR(topRight[1].x, axes[1].x, 2.f);
                ASSERT_NEAR(topRight[1].y, axes[1].y, 2.f);
                vector<Point2f> bottomLeft = getMarkerById(gridboard.getIds()[6], corners, ids);
                ASSERT_NEAR(bottomLeft[3].x, axes[2].x, 2.f);
                ASSERT_NEAR(bottomLeft[3].y, axes[2].y, 2.f);

                // check estimate result
                for(unsigned int i = 0; i < ids.size(); i++) {
                    int foundIdx = -1;
                    for(unsigned int j = 0; j < gridboard.getIds().size(); j++) {
                        if(gridboard.getIds()[j] == ids[i]) {
                            foundIdx = int(j);
                            break;
                        }
                    }

                    if(foundIdx == -1) {
                        ts->printf(cvtest::TS::LOG, "Marker detected with wrong ID in Board test");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    vector< Point2f > projectedCorners;
                    projectPoints(gridboard.getObjPoints()[foundIdx], rvec, tvec, cameraMatrix, distCoeffs,
                                  projectedCorners);

                    for(int c = 0; c < 4; c++) {
                        double repError = cv::norm(projectedCorners[c] - corners[i][c]);  // TODO cvtest
                        if(repError > 5.) {
                            ts->printf(cvtest::TS::LOG, "Corner reprojection error too high");
                            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                            return;
                        }
                    }
                }
            }
        }
    }
}



/**
 * @brief Check refine strategy
 */
class CV_ArucoRefine : public cvtest::BaseTest {
    public:
    CV_ArucoRefine(ArucoAlgParams arucoAlgParams)
    {
        aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
        aruco::DetectorParameters params;
        params.minDistanceToBorder = 3;
        params.cornerRefinementMethod = (int)aruco::CORNER_REFINE_SUBPIX;
        if (arucoAlgParams == ArucoAlgParams::USE_ARUCO3)
            params.useAruco3Detection = true;
        aruco::RefineParameters refineParams(10.f, 3.f, true);
        detector = aruco::ArucoDetector(dictionary, params, refineParams);
    }

    protected:
    aruco::ArucoDetector detector;
    void run(int);
};


void CV_ArucoRefine::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    aruco::DetectorParameters detectorParameters = detector.getDetectorParameters();

    // for different perspectives
    for(double distance : {0.2, 0.4}) {
        for(int yaw = -60; yaw < 60; yaw += 30) {
            for(int pitch = -60; pitch <= 60; pitch += 30) {
                aruco::GridBoard gridboard(Size(3, 3), 0.02f, 0.005f, detector.getDictionary());
                int markerBorder = iter % 2 + 1;
                iter++;

                // create synthetic image
                Mat img = projectBoard(gridboard, cameraMatrix, deg2rad(yaw), deg2rad(pitch), distance,
                                       imgSize, markerBorder);
                // detect markers
                vector<vector<Point2f> > corners, rejected;
                vector<int> ids;
                detectorParameters.markerBorderBits = markerBorder;
                detector.setDetectorParameters(detectorParameters);
                detector.detectMarkers(img, corners, ids, rejected);

                // remove a marker from detection
                int markersBeforeDelete = (int)ids.size();
                if(markersBeforeDelete < 2) continue;

                rejected.push_back(corners[0]);
                corners.erase(corners.begin(), corners.begin() + 1);
                ids.erase(ids.begin(), ids.begin() + 1);

                // try to refind the erased marker
                detector.refineDetectedMarkers(img, gridboard, corners, ids, rejected, cameraMatrix,
                                               distCoeffs, noArray());

                // check result
                if((int)ids.size() < markersBeforeDelete) {
                    ts->printf(cvtest::TS::LOG, "Error in refine detected markers");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }
            }
        }
    }
}

TEST(CV_ArucoBoardPose, accuracy) {
    CV_ArucoBoardPose test(ArucoAlgParams::USE_DEFAULT);
    test.safe_run();
}

typedef CV_ArucoBoardPose CV_Aruco3BoardPose;
TEST(CV_Aruco3BoardPose, accuracy) {
    CV_Aruco3BoardPose test(ArucoAlgParams::USE_ARUCO3);
    test.safe_run();
}

typedef CV_ArucoRefine CV_Aruco3Refine;

TEST(CV_ArucoRefine, accuracy) {
    CV_ArucoRefine test(ArucoAlgParams::USE_DEFAULT);
    test.safe_run();
}

TEST(CV_Aruco3Refine, accuracy) {
    CV_Aruco3Refine test(ArucoAlgParams::USE_ARUCO3);
    test.safe_run();
}

TEST(CV_ArucoBoardPose, CheckNegativeZ)
{
    double matrixData[9] = { -3.9062571886921410e+02, 0., 4.2350000000000000e+02,
                              0., 3.9062571886921410e+02, 2.3950000000000000e+02,
                              0., 0., 1 };
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, matrixData);

    vector<cv::Point3f> pts3d1, pts3d2;
    pts3d1.push_back(cv::Point3f(0.326198f, -0.030621f, 0.303620f));
    pts3d1.push_back(cv::Point3f(0.325340f, -0.100594f, 0.301862f));
    pts3d1.push_back(cv::Point3f(0.255859f, -0.099530f, 0.293416f));
    pts3d1.push_back(cv::Point3f(0.256717f, -0.029557f, 0.295174f));

    pts3d2.push_back(cv::Point3f(-0.033144f, -0.034819f, 0.245216f));
    pts3d2.push_back(cv::Point3f(-0.035507f, -0.104705f, 0.241987f));
    pts3d2.push_back(cv::Point3f(-0.105289f, -0.102120f, 0.237120f));
    pts3d2.push_back(cv::Point3f(-0.102926f, -0.032235f, 0.240349f));

    vector<int> tmpIds = {0, 1};
    vector<vector<Point3f> > tmpObjectPoints = {pts3d1, pts3d2};
    aruco::Board board(tmpObjectPoints, aruco::getPredefinedDictionary(0), tmpIds);

    vector<vector<Point2f> > corners;
    vector<Point2f> pts2d;
    pts2d.push_back(cv::Point2f(37.7f, 203.3f));
    pts2d.push_back(cv::Point2f(38.5f, 120.5f));
    pts2d.push_back(cv::Point2f(105.5f, 115.8f));
    pts2d.push_back(cv::Point2f(104.2f, 202.7f));
    corners.push_back(pts2d);
    pts2d.clear();
    pts2d.push_back(cv::Point2f(476.0f, 184.2f));
    pts2d.push_back(cv::Point2f(479.6f, 73.8f));
    pts2d.push_back(cv::Point2f(590.9f, 77.0f));
    pts2d.push_back(cv::Point2f(587.5f, 188.1f));
    corners.push_back(pts2d);

    Vec3d rvec, tvec;
    int nUsed = 0;
    {
        Mat objPoints, imgPoints; // get object and image points for the solvePnP function
        board.matchImagePoints(corners, board.getIds(), objPoints, imgPoints);
        nUsed = (int)objPoints.total()/4;
        solvePnP(objPoints, imgPoints, cameraMatrix, Mat(), rvec, tvec);
    }
    ASSERT_EQ(nUsed, 2);

    cv::Matx33d rotm; cv::Point3d out;
    cv::Rodrigues(rvec, rotm);
    out = cv::Point3d(tvec) + rotm*Point3d(board.getObjPoints()[0][0]);
    ASSERT_GT(out.z, 0);

    corners.clear(); pts2d.clear();
    pts2d.push_back(cv::Point2f(38.4f, 204.5f));
    pts2d.push_back(cv::Point2f(40.0f, 124.7f));
    pts2d.push_back(cv::Point2f(102.0f, 119.1f));
    pts2d.push_back(cv::Point2f(99.9f, 203.6f));
    corners.push_back(pts2d);
    pts2d.clear();
    pts2d.push_back(cv::Point2f(476.0f, 184.3f));
    pts2d.push_back(cv::Point2f(479.2f, 75.1f));
    pts2d.push_back(cv::Point2f(588.7f, 79.2f));
    pts2d.push_back(cv::Point2f(586.3f, 188.5f));
    corners.push_back(pts2d);

    nUsed = 0;
    {
        Mat objPoints, imgPoints; // get object and image points for the solvePnP function
        board.matchImagePoints(corners, board.getIds(), objPoints, imgPoints);
        nUsed = (int)objPoints.total()/4;
        solvePnP(objPoints, imgPoints, cameraMatrix, Mat(), rvec, tvec, true);
    }
    ASSERT_EQ(nUsed, 2);

    cv::Rodrigues(rvec, rotm);
    out = cv::Point3d(tvec) + rotm*Point3d(board.getObjPoints()[0][0]);
    ASSERT_GT(out.z, 0);
}

TEST(CV_ArucoGenerateBoard, regression_1226) {
    int bwidth = 1600;
    int bheight = 1200;

    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::CharucoBoard board(Size(7, 5), 1.0, 0.75, dict);
    cv::Size sz(bwidth, bheight);
    cv::Mat mat;

    ASSERT_NO_THROW(
    {
        board.generateImage(sz, mat, 0, 1);
    });
}

TEST(CV_ArucoDictionary, extendDictionary) {
    aruco::Dictionary base_dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
    aruco::Dictionary custom_dictionary = aruco::extendDictionary(150, 4, base_dictionary);

    ASSERT_EQ(custom_dictionary.bytesList.rows, 150);
    ASSERT_EQ(cv::norm(custom_dictionary.bytesList, base_dictionary.bytesList.rowRange(0, 150)), 0.);
}
TEST(CV_ArucoBoardGenerateImage_RotationTest, HandlesRotatedMarkersWithoutBoundingBoxError)
{
    using namespace cv;
    using namespace cv::aruco;
    Dictionary dict = getPredefinedDictionary(DICT_4X4_50);
    DetectorParameters detectorParams;
    ArucoDetector detector(dict, detectorParams);
    std::vector<float> angles = {0.0f, 45.0f, 90.0f, 135.0f};
    for (auto angle_deg : angles)
    {
        float angle_rad = angle_deg * static_cast<float>(CV_PI) / 180.0f;
        float c = cos(angle_rad);
        float s = sin(angle_rad);
        std::vector<Point3f> markerCorners(4);
        markerCorners[0] = Point3f(0.f, 0.f, 0.f);
        markerCorners[1] = Point3f(1.f, 0.f, 0.f);
        markerCorners[2] = Point3f(1.f, 1.f, 0.f);
        markerCorners[3] = Point3f(0.f, 1.f, 0.f);
        for (auto &p : markerCorners)
        {
            float xNew = p.x * c - p.y * s;
            float yNew = p.x * s + p.y * c;
            p.x = xNew;
            p.y = yNew;
        }
        std::vector<std::vector<Point3f>> allObjPoints{markerCorners};
        std::vector<int> ids{0};
        Board board(allObjPoints, dict, ids);
        float markerSize = 1.0f;
        float rotatedSize = markerSize * std::sqrt(2.0f);
        int borderBits = 1;
        int marginSize = 20;
        int sidePixels = static_cast<int>((rotatedSize + 2.0f * borderBits) * 500) + 2 * marginSize;
        Mat outImg;
        Size outSize(sidePixels, sidePixels);
        ASSERT_NO_THROW(board.generateImage(outSize, outImg, marginSize, borderBits))
            << "board.generateImage() threw an exception at angle " << angle_deg;
        std::vector<int> detectedIds;
        std::vector<std::vector<Point2f>> detectedCorners;
        detector.detectMarkers(outImg, detectedCorners, detectedIds);
        ASSERT_EQ(detectedIds.size(), (size_t)1)
            << "Failed to detect single marker at angle: " << angle_deg;
        EXPECT_EQ(detectedIds[0], 0)
            << "Marker ID mismatch at angle: " << angle_deg;
    }
}

}} // namespace
