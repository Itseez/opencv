// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "opencv2/objdetect/charuco_detector.hpp"
#include "aruco_utils.hpp"

namespace cv {
namespace aruco {

using namespace std;

struct CharucoDetector::CharucoDetectorImpl {
    CharucoBoard board;
    CharucoParameters charucoParameters;
    ArucoDetector arucoDetector;

    CharucoDetectorImpl(const CharucoBoard& _board, const CharucoParameters _charucoParameters,
                        const ArucoDetector& _arucoDetector): board(_board), charucoParameters(_charucoParameters),
                                                              arucoDetector(_arucoDetector)
    {}

    bool checkBoard(InputArrayOfArrays markerCorners, InputArray markerIds, InputArray charucoCorners, InputArray charucoIds) {
        vector<Mat> mCorners;
        markerCorners.getMatVector(mCorners);
        const Mat mIds = markerIds.getMat();
        const Mat chCorners = charucoCorners.getMat();
        const Mat chIds = charucoIds.getMat();
        const vector<int>& boardIds = board.getIds();

        const vector<vector<int> > nearestMarkerIdx = board.getNearestMarkerIdx();
        vector<Point2f> distance(board.getNearestMarkerIdx().size(), Point2f(0.f, std::numeric_limits<float>::max()));
        // distance[i].x: max distance from the i-th charuco corner to charuco corner-forming markers.
        // The two charuco corner-forming markers of i-th charuco corner are defined in getNearestMarkerIdx()[i]
        // distance[i].y: min distance from the charuco corner to other markers.
        for (size_t i = 0ull; i < chIds.total(); i++) {
            int chId = chIds.ptr<int>(0)[i];
            Point2f charucoCorner(chCorners.ptr<Point2f>(0)[i]);
            for (size_t j = 0ull; j < mIds.total(); j++) {
                int idMaker = mIds.ptr<int>(0)[j];
                // skip the check if the marker is not in the current board.
                if (find(boardIds.begin(), boardIds.end(), idMaker) == boardIds.end())
                    continue;
                Point2f centerMarker((mCorners[j].ptr<Point2f>(0)[0] + mCorners[j].ptr<Point2f>(0)[1] +
                                      mCorners[j].ptr<Point2f>(0)[2] + mCorners[j].ptr<Point2f>(0)[3]) / 4.f);
                float dist = sqrt(normL2Sqr<float>(centerMarker - charucoCorner));
                // nearestMarkerIdx contains for each charuco corner, nearest marker index in ids array
                const int nearestMarkerId1 = boardIds[nearestMarkerIdx[chId][0]];
                const int nearestMarkerId2 = boardIds[nearestMarkerIdx[chId][1]];
                if (nearestMarkerId1 == idMaker || nearestMarkerId2 == idMaker) {
                    int nearestCornerId = nearestMarkerId1 == idMaker ? board.getNearestMarkerCorners()[chId][0] : board.getNearestMarkerCorners()[chId][1];
                    Point2f nearestCorner = mCorners[j].ptr<Point2f>(0)[nearestCornerId];
                    // distToNearest: distance from the charuco corner to charuco corner-forming markers
                    float distToNearest = sqrt(normL2Sqr<float>(nearestCorner - charucoCorner));
                    distance[chId].x = max(distance[chId].x, distToNearest);
                    // check that nearestCorner is nearest point
                    {
                        Point2f mid1 = (mCorners[j].ptr<Point2f>(0)[(nearestCornerId + 1) % 4]+nearestCorner)*0.5f;
                        Point2f mid2 = (mCorners[j].ptr<Point2f>(0)[(nearestCornerId + 3) % 4]+nearestCorner)*0.5f;
                        float tmpDist = min(sqrt(normL2Sqr<float>(mid1 - charucoCorner)), sqrt(normL2Sqr<float>(mid2 - charucoCorner)));
                        if (tmpDist < distToNearest)
                            return false;
                    }
                }
                // check distance from the charuco corner to other markers
                else
                    distance[chId].y = min(distance[chId].y, dist);
            }
            // if distance from the charuco corner to charuco corner-forming markers more then distance from the charuco corner to other markers,
            // then a false board is found.
            if (distance[chId].x > 0.f && distance[chId].y < std::numeric_limits<float>::max() && distance[chId].x > distance[chId].y)
                return false;
        }
        return true;
    }

    /** Calculate the maximum window sizes for corner refinement for each charuco corner based on the distance
     * to their closest markers */
    vector<Size> getMaximumSubPixWindowSizes(InputArrayOfArrays markerCorners, InputArray markerIds,
                                               InputArray charucoCorners) {
        size_t nCharucoCorners = charucoCorners.getMat().total();

        CV_Assert(board.getNearestMarkerIdx().size() == nCharucoCorners);

        vector<Size> winSizes(nCharucoCorners, Size(-1, -1));
        for(size_t i = 0ull; i < nCharucoCorners; i++) {
            if(charucoCorners.getMat().at<Point2f>((int)i) == Point2f(-1.f, -1.f)) continue;
            if(board.getNearestMarkerIdx()[i].empty()) continue;
                double minDist = -1;
                int counter = 0;
                // calculate the distance to each of the closest corner of each closest marker
                for(size_t j = 0; j < board.getNearestMarkerIdx()[i].size(); j++) {
                    // find marker
                    int markerId = board.getIds()[board.getNearestMarkerIdx()[i][j]];
                    int markerIdx = -1;
                    for(size_t k = 0; k < markerIds.getMat().total(); k++) {
                        if(markerIds.getMat().at<int>((int)k) == markerId) {
                            markerIdx = (int)k;
                                break;
                            }
                        }
                    if(markerIdx == -1) continue;
                    Point2f markerCorner =
                        markerCorners.getMat(markerIdx).at<Point2f>(board.getNearestMarkerCorners()[i][j]);
                    Point2f charucoCorner = charucoCorners.getMat().at<Point2f>((int)i);
                    double dist = norm(markerCorner - charucoCorner);
                    if(minDist == -1) minDist = dist; // if first distance, just assign it
                    minDist = min(dist, minDist);
                    counter++;
                }
                // if this is the first closest marker, dont do anything
                if(counter == 0)
                    continue;
                else {
                    // else, calculate the maximum window size
                    int winSizeInt = int(minDist - 2); // remove 2 pixels for safety
                    if(winSizeInt < 1) winSizeInt = 1; // minimum size is 1
                    if(winSizeInt > 10) winSizeInt = 10; // maximum size is 10
                    winSizes[i] = Size(winSizeInt, winSizeInt);
                }
            }
        return winSizes;
    }

    /** @brief From all projected chessboard corners, select those inside the image and apply subpixel refinement */
    void selectAndRefineChessboardCorners(InputArray allCorners, InputArray image, OutputArray selectedCorners,
                                          OutputArray selectedIds, const vector<Size> &winSizes) {
        const int minDistToBorder = 2; // minimum distance of the corner to the image border
        // remaining corners, ids and window refinement sizes after removing corners outside the image
        vector<Point2f> filteredChessboardImgPoints;
        vector<Size> filteredWinSizes;
        vector<int> filteredIds;
        // filter corners outside the image
        Rect innerRect(minDistToBorder, minDistToBorder, image.getMat().cols - 2 * minDistToBorder,
                       image.getMat().rows - 2 * minDistToBorder);
        for(unsigned int i = 0; i < allCorners.getMat().total(); i++) {
            if(innerRect.contains(allCorners.getMat().at<Point2f>(i))) {
                filteredChessboardImgPoints.push_back(allCorners.getMat().at<Point2f>(i));
                filteredIds.push_back(i);
                filteredWinSizes.push_back(winSizes[i]);
            }
        }
        // if none valid, return 0
        if(filteredChessboardImgPoints.empty()) return;
        // corner refinement, first convert input image to grey
        Mat grey;
        if(image.type() == CV_8UC3)
            cvtColor(image, grey, COLOR_BGR2GRAY);
        else
            grey = image.getMat();
        //// For each of the charuco corners, apply subpixel refinement using its correspondind winSize
        parallel_for_(Range(0, (int)filteredChessboardImgPoints.size()), [&](const Range& range) {
            const int begin = range.start;
            const int end = range.end;
            for (int i = begin; i < end; i++) {
                vector<Point2f> in;
                in.push_back(filteredChessboardImgPoints[i] - Point2f(0.5, 0.5)); // adjust sub-pixel coordinates for cornerSubPix
                Size winSize = filteredWinSizes[i];
                if (winSize.height == -1 || winSize.width == -1)
                    winSize = Size(arucoDetector.getDetectorParameters().cornerRefinementWinSize,
                                   arucoDetector.getDetectorParameters().cornerRefinementWinSize);
                cornerSubPix(grey, in, winSize, Size(),
                             TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                          arucoDetector.getDetectorParameters().cornerRefinementMaxIterations,
                                          arucoDetector.getDetectorParameters().cornerRefinementMinAccuracy));
                filteredChessboardImgPoints[i] = in[0] + Point2f(0.5, 0.5);
            }
        });
        // parse output
        Mat(filteredChessboardImgPoints).copyTo(selectedCorners);
        Mat(filteredIds).copyTo(selectedIds);
    }

    /** Interpolate charuco corners using approximated pose estimation */
    void interpolateCornersCharucoApproxCalib(InputArrayOfArrays markerCorners, InputArray markerIds,
                                              InputArray image, OutputArray charucoCorners, OutputArray charucoIds) {
        CV_Assert(image.getMat().channels() == 1 || image.getMat().channels() == 3);
        CV_Assert(markerCorners.total() == markerIds.getMat().total());

        // approximated pose estimation using marker corners
        Mat approximatedRvec, approximatedTvec;
        Mat objPoints, imgPoints; // object and image points for the solvePnP function
        Board simpleBoard(board.getObjPoints(), board.getDictionary(), board.getIds());
        simpleBoard.matchImagePoints(markerCorners, markerIds, objPoints, imgPoints);
        if (objPoints.total() < 4ull)  // need, at least, 4 corners
            return;

        solvePnP(objPoints, imgPoints, charucoParameters.cameraMatrix, charucoParameters.distCoeffs, approximatedRvec, approximatedTvec);

        // project chessboard corners
        vector<Point2f> allChessboardImgPoints;
        projectPoints(board.getChessboardCorners(), approximatedRvec, approximatedTvec, charucoParameters.cameraMatrix,
                      charucoParameters.distCoeffs, allChessboardImgPoints);
        // calculate maximum window sizes for subpixel refinement. The size is limited by the distance
        // to the closes marker corner to avoid erroneous displacements to marker corners
        vector<Size> subPixWinSizes = getMaximumSubPixWindowSizes(markerCorners, markerIds, allChessboardImgPoints);
        // filter corners outside the image and subpixel-refine charuco corners
        selectAndRefineChessboardCorners(allChessboardImgPoints, image, charucoCorners, charucoIds, subPixWinSizes);
    }

    /** Interpolate charuco corners using local homography */
    void interpolateCornersCharucoLocalHom(InputArrayOfArrays markerCorners, InputArray markerIds, InputArray image,
                                           OutputArray charucoCorners, OutputArray charucoIds) {
        CV_Assert(image.getMat().channels() == 1 || image.getMat().channels() == 3);
        CV_Assert(markerCorners.total() == markerIds.getMat().total());
        size_t nMarkers = markerIds.getMat().total();
        // calculate local homographies for each marker
        vector<Mat> transformations(nMarkers);
        vector<bool> validTransform(nMarkers, false);
        const auto& ids = board.getIds();
        for(size_t i = 0ull; i < nMarkers; i++) {
            vector<Point2f> markerObjPoints2D;
            int markerId = markerIds.getMat().at<int>((int)i);
            auto it = find(ids.begin(), ids.end(), markerId);
            if(it == ids.end()) continue;
            auto boardIdx = it - ids.begin();
            markerObjPoints2D.resize(4ull);
            for(size_t j = 0ull; j < 4ull; j++)
                markerObjPoints2D[j] =
                    Point2f(board.getObjPoints()[boardIdx][j].x, board.getObjPoints()[boardIdx][j].y);
            transformations[i] = getPerspectiveTransform(markerObjPoints2D, markerCorners.getMat((int)i));
            // set transform as valid if transformation is non-singular
            double det = determinant(transformations[i]);
            validTransform[i] = std::abs(det) > 1e-6;
        }
        size_t nCharucoCorners = (size_t)board.getChessboardCorners().size();
        vector<Point2f> allChessboardImgPoints(nCharucoCorners, Point2f(-1, -1));
        // for each charuco corner, calculate its interpolation position based on the closest markers
        // homographies
        for(size_t i = 0ull; i < nCharucoCorners; i++) {
            Point2f objPoint2D = Point2f(board.getChessboardCorners()[i].x, board.getChessboardCorners()[i].y);
            vector<Point2f> interpolatedPositions;
            for(size_t j = 0ull; j < board.getNearestMarkerIdx()[i].size(); j++) {
                int markerId = board.getIds()[board.getNearestMarkerIdx()[i][j]];
                int markerIdx = -1;
                for(size_t k = 0ull; k < markerIds.getMat().total(); k++) {
                    if(markerIds.getMat().at<int>((int)k) == markerId) {
                        markerIdx = (int)k;
                        break;
                    }
                }
                if (markerIdx != -1 &&
                    validTransform[markerIdx])
                {
                    vector<Point2f> in, out;
                    in.push_back(objPoint2D);
                    perspectiveTransform(in, out, transformations[markerIdx]);
                    interpolatedPositions.push_back(out[0]);
                }
            }
            // none of the closest markers detected
            if(interpolatedPositions.empty()) continue;
            // more than one closest marker detected, take middle point
            if(interpolatedPositions.size() > 1ull) {
                allChessboardImgPoints[i] = (interpolatedPositions[0] + interpolatedPositions[1]) / 2.;
            }
            // a single closest marker detected
            else allChessboardImgPoints[i] = interpolatedPositions[0];
        }
        // calculate maximum window sizes for subpixel refinement. The size is limited by the distance
        // to the closes marker corner to avoid erroneous displacements to marker corners
        vector<Size> subPixWinSizes = getMaximumSubPixWindowSizes(markerCorners, markerIds, allChessboardImgPoints);
        // filter corners outside the image and subpixel-refine charuco corners
        selectAndRefineChessboardCorners(allChessboardImgPoints, image, charucoCorners, charucoIds, subPixWinSizes);
    }

    /** Remove charuco corners if any of their minMarkers closest markers has not been detected */
    int filterCornersWithoutMinMarkers(InputArray _allCharucoCorners, InputArray allCharucoIds, InputArray allArucoIds,
                                       OutputArray _filteredCharucoCorners, OutputArray _filteredCharucoIds) {
        CV_Assert(charucoParameters.minMarkers >= 0 && charucoParameters.minMarkers <= 2);
        vector<Point2f> filteredCharucoCorners;
        vector<int> filteredCharucoIds;
        // for each charuco corner
        for(unsigned int i = 0; i < allCharucoIds.getMat().total(); i++) {
            int currentCharucoId = allCharucoIds.getMat().at<int>(i);
            int totalMarkers = 0; // nomber of closest marker detected
            // look for closest markers
            for(unsigned int m = 0; m < board.getNearestMarkerIdx()[currentCharucoId].size(); m++) {
                int markerId = board.getIds()[board.getNearestMarkerIdx()[currentCharucoId][m]];
                bool found = false;
                for(unsigned int k = 0; k < allArucoIds.getMat().total(); k++) {
                    if(allArucoIds.getMat().at<int>(k) == markerId) {
                        found = true;
                        break;
                    }
                }
                if(found) totalMarkers++;
            }
            // if enough markers detected, add the charuco corner to the final list
            if(totalMarkers >= charucoParameters.minMarkers) {
                filteredCharucoIds.push_back(currentCharucoId);
                filteredCharucoCorners.push_back(_allCharucoCorners.getMat().at<Point2f>(i));
            }
        }
        // parse output
        Mat(filteredCharucoCorners).copyTo(_filteredCharucoCorners);
        Mat(filteredCharucoIds).copyTo(_filteredCharucoIds);
        return (int)_filteredCharucoIds.total();
    }

    void detectBoard(InputArray image, OutputArray charucoCorners, OutputArray charucoIds,
                     InputOutputArrayOfArrays markerCorners, InputOutputArray markerIds) {
        CV_Assert((markerCorners.empty() && markerIds.empty() && !image.empty()) || (markerCorners.total() == markerIds.total()));
        vector<vector<Point2f>> tmpMarkerCorners;
        vector<int> tmpMarkerIds;
        InputOutputArrayOfArrays _markerCorners = markerCorners.needed() ? markerCorners : tmpMarkerCorners;
        InputOutputArray _markerIds = markerIds.needed() ? markerIds : tmpMarkerIds;

        if (markerCorners.empty() && markerIds.empty()) {
            vector<vector<Point2f> > rejectedMarkers;
            arucoDetector.detectMarkers(image, _markerCorners, _markerIds, rejectedMarkers);
            if (charucoParameters.tryRefineMarkers)
                arucoDetector.refineDetectedMarkers(image, board, _markerCorners, _markerIds, rejectedMarkers);
            if (_markerCorners.empty() && _markerIds.empty())
                return;
        }
        // if camera parameters are avaible, use approximated calibration
        if(!charucoParameters.cameraMatrix.empty())
            interpolateCornersCharucoApproxCalib(_markerCorners, _markerIds, image, charucoCorners, charucoIds);
        // else use local homography
        else
            interpolateCornersCharucoLocalHom(_markerCorners, _markerIds, image, charucoCorners, charucoIds);
        // to return a charuco corner, its closest aruco markers should have been detected
        filterCornersWithoutMinMarkers(charucoCorners, charucoIds, _markerIds, charucoCorners, charucoIds);
    }

    void detectBoardWithCheck(InputArray image, OutputArray charucoCorners, OutputArray charucoIds,
                              InputOutputArrayOfArrays markerCorners, InputOutputArray markerIds) {
        vector<vector<Point2f>> tmpMarkerCorners;
        vector<int> tmpMarkerIds;
        InputOutputArrayOfArrays _markerCorners = markerCorners.needed() ? markerCorners : tmpMarkerCorners;
        InputOutputArray _markerIds = markerIds.needed() ? markerIds : tmpMarkerIds;
        detectBoard(image, charucoCorners, charucoIds, _markerCorners, _markerIds);
        if (charucoParameters.checkMarkers && checkBoard(_markerCorners, _markerIds, charucoCorners, charucoIds) == false) {
            CV_LOG_DEBUG(NULL, "ChArUco board is built incorrectly");
            charucoCorners.release();
            charucoIds.release();
        }
    }
};

CharucoDetector::CharucoDetector(const CharucoBoard &board, const CharucoParameters &charucoParams,
                                 const DetectorParameters &detectorParams, const RefineParameters& refineParams) {
    this->charucoDetectorImpl = makePtr<CharucoDetectorImpl>(board, charucoParams, ArucoDetector(board.getDictionary(), detectorParams, refineParams));
}

const CharucoBoard& CharucoDetector::getBoard() const {
    return charucoDetectorImpl->board;
}

void CharucoDetector::setBoard(const CharucoBoard& board) {
     this->charucoDetectorImpl->board = board;
      charucoDetectorImpl->arucoDetector.setDictionary(board.getDictionary());
}

const CharucoParameters &CharucoDetector::getCharucoParameters() const {
    return charucoDetectorImpl->charucoParameters;
}

void CharucoDetector::setCharucoParameters(CharucoParameters &charucoParameters) {
    charucoDetectorImpl->charucoParameters = charucoParameters;
}

const DetectorParameters& CharucoDetector::getDetectorParameters() const {
    return charucoDetectorImpl->arucoDetector.getDetectorParameters();
}

void CharucoDetector::setDetectorParameters(const DetectorParameters& detectorParameters) {
    charucoDetectorImpl->arucoDetector.setDetectorParameters(detectorParameters);
}

const RefineParameters& CharucoDetector::getRefineParameters() const {
    return charucoDetectorImpl->arucoDetector.getRefineParameters();
}

void CharucoDetector::setRefineParameters(const RefineParameters& refineParameters) {
    charucoDetectorImpl->arucoDetector.setRefineParameters(refineParameters);
}

void CharucoDetector::detectBoard(InputArray image, OutputArray charucoCorners, OutputArray charucoIds,
                                  InputOutputArrayOfArrays markerCorners, InputOutputArray markerIds) const {
    charucoDetectorImpl->detectBoardWithCheck(image, charucoCorners, charucoIds, markerCorners, markerIds);
}

void CharucoDetector::detectDiamonds(InputArray image, OutputArrayOfArrays _diamondCorners, OutputArray _diamondIds,
                                     InputOutputArrayOfArrays inMarkerCorners, InputOutputArray inMarkerIds) const {
    CV_Assert(getBoard().getChessboardSize() == Size(3, 3));
    CV_Assert((inMarkerCorners.empty() && inMarkerIds.empty() && !image.empty()) || (inMarkerCorners.total() == inMarkerIds.total()));

    vector<vector<Point2f>> tmpMarkerCorners;
    vector<int> tmpMarkerIds;
    InputOutputArrayOfArrays _markerCorners = inMarkerCorners.needed() ? inMarkerCorners : tmpMarkerCorners;
    InputOutputArray _markerIds = inMarkerIds.needed() ? inMarkerIds : tmpMarkerIds;
    if (_markerCorners.empty() && _markerIds.empty()) {
        charucoDetectorImpl->arucoDetector.detectMarkers(image, _markerCorners, _markerIds);
    }

    const float minRepDistanceRate = 1.302455f;
    vector<vector<Point2f>> diamondCorners;
    vector<Vec4i> diamondIds;

    // stores if the detected markers have been assigned or not to a diamond
    vector<bool> assigned(_markerIds.total(), false);
    if(_markerIds.total() < 4ull) return; // a diamond need at least 4 markers

    // convert input image to grey
    Mat grey;
    if(image.type() == CV_8UC3)
        cvtColor(image, grey, COLOR_BGR2GRAY);
    else
        grey = image.getMat();
    auto board = getBoard();

    // for each of the detected markers, try to find a diamond
    for(unsigned int i = 0; i < (unsigned int)_markerIds.total(); i++) {
        if(assigned[i]) continue;

        // calculate marker perimeter
        float perimeterSq = 0;
        Mat corners = _markerCorners.getMat(i);
        for(int c = 0; c < 4; c++) {
          Point2f edge = corners.at<Point2f>(c) - corners.at<Point2f>((c + 1) % 4);
          perimeterSq += edge.x*edge.x + edge.y*edge.y;
        }
        // maximum reprojection error relative to perimeter
        float minRepDistance = sqrt(perimeterSq) * minRepDistanceRate;

        int currentId = _markerIds.getMat().at<int>(i);

        // prepare data to call refineDetectedMarkers()
        // detected markers (only the current one)
        vector<Mat> currentMarker;
        vector<int> currentMarkerId;
        currentMarker.push_back(_markerCorners.getMat(i));
        currentMarkerId.push_back(currentId);

        // marker candidates (the rest of markers if they have not been assigned)
        vector<Mat> candidates;
        vector<int> candidatesIdxs;
        for(unsigned int k = 0; k < assigned.size(); k++) {
            if(k == i) continue;
            if(!assigned[k]) {
                candidates.push_back(_markerCorners.getMat(k));
                candidatesIdxs.push_back(k);
            }
        }
        if(candidates.size() < 3ull) break; // we need at least 3 free markers
        // modify charuco layout id to make sure all the ids are different than current id
        vector<int> tmpIds(4ull);
        for(int k = 1; k < 4; k++)
            tmpIds[k] = currentId + 1 + k;
        // current id is assigned to [0], so it is the marker on the top
        tmpIds[0] = currentId;

        // create Charuco board layout for diamond (3x3 layout)
        charucoDetectorImpl->board = CharucoBoard(Size(3, 3), board.getSquareLength(),
                                                  board.getMarkerLength(), board.getDictionary(), tmpIds);

        // try to find the rest of markers in the diamond
        vector<int> acceptedIdxs;
        if (currentMarker.size() != 4ull)
        {
            RefineParameters refineParameters(minRepDistance, -1.f, false);
            RefineParameters tmp = charucoDetectorImpl->arucoDetector.getRefineParameters();
            charucoDetectorImpl->arucoDetector.setRefineParameters(refineParameters);
            charucoDetectorImpl->arucoDetector.refineDetectedMarkers(grey, getBoard(), currentMarker, currentMarkerId,
                                                                     candidates,
                                                                     noArray(), noArray(), acceptedIdxs);
            charucoDetectorImpl->arucoDetector.setRefineParameters(tmp);
        }

        // if found, we have a diamond
        if(currentMarker.size() == 4ull) {
            assigned[i] = true;
            // calculate diamond id, acceptedIdxs array indicates the markers taken from candidates array
            Vec4i markerId;
            markerId[0] = currentId;
            for(int k = 1; k < 4; k++) {
                int currentMarkerIdx = candidatesIdxs[acceptedIdxs[k - 1]];
                markerId[k] = _markerIds.getMat().at<int>(currentMarkerIdx);
                assigned[currentMarkerIdx] = true;
            }

            // interpolate the charuco corners of the diamond
            vector<Point2f> currentMarkerCorners;
            Mat aux;
            charucoDetectorImpl->detectBoardWithCheck(grey, currentMarkerCorners, aux, currentMarker, currentMarkerId);

            // if everything is ok, save the diamond
            if(currentMarkerCorners.size() > 0ull) {
                // reorder corners
                vector<Point2f> currentMarkerCornersReorder;
                currentMarkerCornersReorder.resize(4);
                currentMarkerCornersReorder[0] = currentMarkerCorners[0];
                currentMarkerCornersReorder[1] = currentMarkerCorners[1];
                currentMarkerCornersReorder[2] = currentMarkerCorners[3];
                currentMarkerCornersReorder[3] = currentMarkerCorners[2];

                diamondCorners.push_back(currentMarkerCornersReorder);
                diamondIds.push_back(markerId);
            }
        }
    }
    charucoDetectorImpl->board = board;

    if(diamondIds.size() > 0ull) {
        // parse output
        Mat(diamondIds).copyTo(_diamondIds);

        _diamondCorners.create((int)diamondCorners.size(), 1, CV_32FC2);
        for(unsigned int i = 0; i < diamondCorners.size(); i++) {
            _diamondCorners.create(4, 1, CV_32FC2, i, true);
            for(int j = 0; j < 4; j++) {
                _diamondCorners.getMat(i).at<Point2f>(j) = diamondCorners[i][j];
            }
        }
    }
}

void drawDetectedCornersCharuco(InputOutputArray _image, InputArray _charucoCorners,
                                InputArray _charucoIds, Scalar cornerColor) {
    CV_Assert(!_image.getMat().empty() &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert((_charucoCorners.total() == _charucoIds.total()) ||
              _charucoIds.total() == 0);
    CV_Assert(_charucoCorners.channels() == 2);

    Mat charucoCorners = _charucoCorners.getMat();
    if (charucoCorners.type() != CV_32SC2)
        charucoCorners.convertTo(charucoCorners, CV_32SC2);
    Mat charucoIds;
    if (!_charucoIds.empty())
        charucoIds = _charucoIds.getMat();
    size_t nCorners = charucoCorners.total();
    for(size_t i = 0; i < nCorners; i++) {
        Point corner = charucoCorners.at<Point>((int)i);
        // draw first corner mark
        rectangle(_image, corner - Point(3, 3), corner + Point(3, 3), cornerColor, 1, LINE_AA);
        // draw ID
        if(!_charucoIds.empty()) {
            int id = charucoIds.at<int>((int)i);
            stringstream s;
            s << "id=" << id;
            putText(_image, s.str(), corner + Point(5, -5), FONT_HERSHEY_SIMPLEX, 0.5,
                    cornerColor, 2);
        }
    }
}

void drawDetectedDiamonds(InputOutputArray _image, InputArrayOfArrays _corners, InputArray _ids, Scalar borderColor) {
    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert((_corners.total() == _ids.total()) || _ids.total() == 0);

    // calculate colors
    Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    int nMarkers = (int)_corners.total();
    for(int i = 0; i < nMarkers; i++) {
        Mat currentMarker = _corners.getMat(i);
        CV_Assert(currentMarker.total() == 4 && currentMarker.channels() == 2);
        if (currentMarker.type() != CV_32SC2)
            currentMarker.convertTo(currentMarker, CV_32SC2);

        // draw marker sides
        for(int j = 0; j < 4; j++) {
            Point p0, p1;
            p0 = currentMarker.at<Point>(j);
            p1 = currentMarker.at<Point>((j + 1) % 4);
            line(_image, p0, p1, borderColor, 1);
        }

        // draw first corner mark
        rectangle(_image, currentMarker.at<Point>(0) - Point(3, 3),
                  currentMarker.at<Point>(0) + Point(3, 3), cornerColor, 1, LINE_AA);

        // draw id composed by four numbers
        if(_ids.total() != 0) {
            Point cent(0, 0);
            for(int p = 0; p < 4; p++)
                cent += currentMarker.at<Point>(p);
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().at< Vec4i >(i);
            putText(_image, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}

}
}
