/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_TRACKING_HPP
#define OPENCV_TRACKING_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

//! @addtogroup video_track
//! @{

enum { OPTFLOW_USE_INITIAL_FLOW     = 4,
       OPTFLOW_LK_GET_MIN_EIGENVALS = 8,
       OPTFLOW_FARNEBACK_GAUSSIAN   = 256
     };

/** @brief Finds an object center, size, and orientation.

@param probImage Back projection of the object histogram. See calcBackProject.
@param window Initial search window.
@param criteria Stop criteria for the underlying meanShift.
returns
(in old interfaces) Number of iterations CAMSHIFT took to converge
The function implements the CAMSHIFT object tracking algorithm @cite Bradski98 . First, it finds an
object center using meanShift and then adjusts the window size and finds the optimal rotation. The
function returns the rotated rectangle structure that includes the object position, size, and
orientation. The next position of the search window can be obtained with RotatedRect::boundingRect()

See the OpenCV sample camshiftdemo.c that tracks colored objects.

@note
-   (Python) A sample explaining the camshift tracking algorithm can be found at
    opencv_source_code/samples/python/camshift.py
 */
CV_EXPORTS_W RotatedRect CamShift( InputArray probImage, CV_IN_OUT Rect& window,
                                   TermCriteria criteria );
/** @example samples/cpp/camshiftdemo.cpp
An example using the mean-shift tracking algorithm
*/

/** @brief Finds an object on a back projection image.

@param probImage Back projection of the object histogram. See calcBackProject for details.
@param window Initial search window.
@param criteria Stop criteria for the iterative search algorithm.
returns
:   Number of iterations CAMSHIFT took to converge.
The function implements the iterative object search algorithm. It takes the input back projection of
an object and the initial position. The mass center in window of the back projection image is
computed and the search window center shifts to the mass center. The procedure is repeated until the
specified number of iterations criteria.maxCount is done or until the window center shifts by less
than criteria.epsilon. The algorithm is used inside CamShift and, unlike CamShift , the search
window size or orientation do not change during the search. You can simply pass the output of
calcBackProject to this function. But better results can be obtained if you pre-filter the back
projection and remove the noise. For example, you can do this by retrieving connected components
with findContours , throwing away contours with small area ( contourArea ), and rendering the
remaining contours with drawContours.

 */
CV_EXPORTS_W int meanShift( InputArray probImage, CV_IN_OUT Rect& window, TermCriteria criteria );

/** @brief Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.

@param img 8-bit input image.
@param pyramid output pyramid.
@param winSize window size of optical flow algorithm. Must be not less than winSize argument of
calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
@param maxLevel 0-based maximal pyramid level number.
@param withDerivatives set to precompute gradients for the every pyramid level. If pyramid is
constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
@param pyrBorder the border mode for pyramid layers.
@param derivBorder the border mode for gradients.
@param tryReuseInputImage put ROI of input image into the pyramid if possible. You can pass false
to force data copying.
@return number of levels in constructed pyramid. Can be less than maxLevel.
 */
CV_EXPORTS_W int buildOpticalFlowPyramid( InputArray img, OutputArrayOfArrays pyramid,
                                          Size winSize, int maxLevel, bool withDerivatives = true,
                                          int pyrBorder = BORDER_REFLECT_101,
                                          int derivBorder = BORDER_CONSTANT,
                                          bool tryReuseInputImage = true );

/** @example samples/cpp/lkdemo.cpp
An example using the Lucas-Kanade optical flow algorithm
*/

/** @brief Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with
pyramids.

@param prevImg first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
@param nextImg second input image or pyramid of the same size and the same type as prevImg.
@param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
single-precision floating-point numbers.
@param nextPts output vector of 2D points (with single-precision floating-point coordinates)
containing the calculated new positions of input features in the second image; when
OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
@param status output status vector (of unsigned chars); each element of the vector is set to 1 if
the flow for the corresponding features has been found, otherwise, it is set to 0.
@param err output vector of errors; each element of the vector is set to an error for the
corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
found then the error is not defined (use the status parameter to find such cases).
@param winSize size of the search window at each pyramid level.
@param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
algorithm will use as many levels as pyramids have but no more than maxLevel.
@param criteria parameter, specifying the termination criteria of the iterative search algorithm
(after the specified maximum number of iterations criteria.maxCount or when the search window
moves by less than criteria.epsilon.
@param flags operation flags:
 -   **OPTFLOW_USE_INITIAL_FLOW** uses initial estimations, stored in nextPts; if the flag is
     not set, then prevPts is copied to nextPts and is considered the initial estimate.
 -   **OPTFLOW_LK_GET_MIN_EIGENVALS** use minimum eigen values as an error measure (see
     minEigThreshold description); if the flag is not set, then L1 distance between patches
     around the original and a moved point, divided by number of pixels in a window, is used as a
     error measure.
@param minEigThreshold the algorithm calculates the minimum eigen value of a 2x2 normal matrix of
optical flow equations (this matrix is called a spatial gradient matrix in @cite Bouguet00), divided
by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
performance boost.

The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
@cite Bouguet00 . The function is parallelized with the TBB library.

@note Some examples:

-   An example using the Lucas-Kanade optical flow algorithm can be found at
    opencv_source_code/samples/cpp/lkdemo.cpp
-   (Python) An example using the Lucas-Kanade optical flow algorithm can be found at
    opencv_source_code/samples/python/lk_track.py
-   (Python) An example using the Lucas-Kanade tracker for homography matching can be found at
    opencv_source_code/samples/python/lk_homography.py
 */
CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                        InputArray prevPts, InputOutputArray nextPts,
                                        OutputArray status, OutputArray err,
                                        Size winSize = Size(21,21), int maxLevel = 3,
                                        TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                        int flags = 0, double minEigThreshold = 1e-4 );

/** @brief Computes a dense optical flow using the Gunnar Farneback's algorithm.

@param prev first 8-bit single-channel input image.
@param next second input image of the same size and the same type as prev.
@param flow computed flow image that has the same size as prev and type CV_32FC2.
@param pyr_scale parameter, specifying the image scale (\<1) to build pyramids for each image;
pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous
one.
@param levels number of pyramid layers including the initial image; levels=1 means that no extra
layers are created and only the original images are used.
@param winsize averaging window size; larger values increase the algorithm robustness to image
noise and give more chances for fast motion detection, but yield more blurred motion field.
@param iterations number of iterations the algorithm does at each pyramid level.
@param poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel;
larger values mean that the image will be approximated with smoother surfaces, yielding more
robust algorithm and more blurred motion field, typically poly_n =5 or 7.
@param poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a
basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a
good value would be poly_sigma=1.5.
@param flags operation flags that can be a combination of the following:
 -   **OPTFLOW_USE_INITIAL_FLOW** uses the input flow as an initial flow approximation.
 -   **OPTFLOW_FARNEBACK_GAUSSIAN** uses the Gaussian \f$\texttt{winsize}\times\texttt{winsize}\f$
     filter instead of a box filter of the same size for optical flow estimation; usually, this
     option gives z more accurate flow than with a box filter, at the cost of lower speed;
     normally, winsize for a Gaussian window should be set to a larger value to achieve the same
     level of robustness.

The function finds an optical flow for each prev pixel using the @cite Farneback2003 algorithm so that

\f[\texttt{prev} (y,x)  \sim \texttt{next} ( y + \texttt{flow} (y,x)[1],  x + \texttt{flow} (y,x)[0])\f]

@note Some examples:

-   An example using the optical flow algorithm described by Gunnar Farneback can be found at
    opencv_source_code/samples/cpp/fback.cpp
-   (Python) An example using the optical flow algorithm described by Gunnar Farneback can be
    found at opencv_source_code/samples/python/opt_flow.py
 */
CV_EXPORTS_W void calcOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                            double pyr_scale, int levels, int winsize,
                                            int iterations, int poly_n, double poly_sigma,
                                            int flags );

/** @brief Computes an optimal affine transformation between two 2D point sets.

@param src First input 2D point set stored in std::vector or Mat, or an image stored in Mat.
@param dst Second input 2D point set of the same size and the same type as A, or another image.
@param fullAffine If true, the function finds an optimal affine transformation with no additional
restrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is
limited to combinations of translation, rotation, and uniform scaling (4 degrees of freedom).

The function finds an optimal affine transform *[A|b]* (a 2 x 3 floating-point matrix) that
approximates best the affine transformation between:

*   Two point sets
*   Two raster images. In this case, the function first finds some features in the src image and
    finds the corresponding features in dst image. After that, the problem is reduced to the first
    case.
In case of point sets, the problem is formulated as follows: you need to find a 2x2 matrix *A* and
2x1 vector *b* so that:

\f[[A^*|b^*] = arg  \min _{[A|b]}  \sum _i  \| \texttt{dst}[i] - A { \texttt{src}[i]}^T - b  \| ^2\f]
where src[i] and dst[i] are the i-th points in src and dst, respectively
\f$[A|b]\f$ can be either arbitrary (when fullAffine=true ) or have a form of
\f[\begin{bmatrix} a_{11} & a_{12} & b_1  \\ -a_{12} & a_{11} & b_2  \end{bmatrix}\f]
when fullAffine=false.

@deprecated Use cv::estimateAffine2D, cv::estimateAffinePartial2D instead. If you are using this function
with images, extract points using cv::calcOpticalFlowPyrLK and then use the estimation functions.

@sa
estimateAffine2D, estimateAffinePartial2D, getAffineTransform, getPerspectiveTransform, findHomography
 */
CV_DEPRECATED CV_EXPORTS Mat estimateRigidTransform( InputArray src, InputArray dst, bool fullAffine );

enum
{
    MOTION_TRANSLATION = 0,
    MOTION_EUCLIDEAN   = 1,
    MOTION_AFFINE      = 2,
    MOTION_HOMOGRAPHY  = 3
};

/** @brief Computes the Enhanced Correlation Coefficient value between two images @cite EP08 .

@param templateImage single-channel template image; CV_8U or CV_32F array.
@param inputImage single-channel input image to be warped to provide an image similar to
 templateImage, same type as templateImage.
@param inputMask An optional mask to indicate valid values of inputImage.

@sa
findTransformECC
 */

CV_EXPORTS_W double computeECC(InputArray templateImage, InputArray inputImage, InputArray inputMask = noArray());

/** @example samples/cpp/image_alignment.cpp
An example using the image alignment ECC algorithm
*/

/** @brief Finds the geometric transform (warp) between two images in terms of the ECC criterion @cite EP08 .

@param templateImage single-channel template image; CV_8U or CV_32F array.
@param inputImage single-channel input image which should be warped with the final warpMatrix in
order to provide an image similar to templateImage, same type as templateImage.
@param warpMatrix floating-point \f$2\times 3\f$ or \f$3\times 3\f$ mapping matrix (warp).
@param motionType parameter, specifying the type of motion:
 -   **MOTION_TRANSLATION** sets a translational motion model; warpMatrix is \f$2\times 3\f$ with
     the first \f$2\times 2\f$ part being the unity matrix and the rest two parameters being
     estimated.
 -   **MOTION_EUCLIDEAN** sets a Euclidean (rigid) transformation as motion model; three
     parameters are estimated; warpMatrix is \f$2\times 3\f$.
 -   **MOTION_AFFINE** sets an affine motion model (DEFAULT); six parameters are estimated;
     warpMatrix is \f$2\times 3\f$.
 -   **MOTION_HOMOGRAPHY** sets a homography as a motion model; eight parameters are
     estimated;\`warpMatrix\` is \f$3\times 3\f$.
@param criteria parameter, specifying the termination criteria of the ECC algorithm;
criteria.epsilon defines the threshold of the increment in the correlation coefficient between two
iterations (a negative criteria.epsilon makes criteria.maxcount the only termination criterion).
Default values are shown in the declaration above.
@param inputMask An optional mask to indicate valid values of inputImage.
@param gaussFiltSize An optional value indicating size of gaussian blur filter; (DEFAULT: 5)

The function estimates the optimum transformation (warpMatrix) with respect to ECC criterion
(@cite EP08), that is

\f[\texttt{warpMatrix} = \arg\max_{W} \texttt{ECC}(\texttt{templateImage}(x,y),\texttt{inputImage}(x',y'))\f]

where

\f[\begin{bmatrix} x' \\ y' \end{bmatrix} = W \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}\f]

(the equation holds with homogeneous coordinates for homography). It returns the final enhanced
correlation coefficient, that is the correlation coefficient between the template image and the
final warped input image. When a \f$3\times 3\f$ matrix is given with motionType =0, 1 or 2, the third
row is ignored.

Unlike findHomography and estimateRigidTransform, the function findTransformECC implements an
area-based alignment that builds on intensity similarities. In essence, the function updates the
initial transformation that roughly aligns the images. If this information is missing, the identity
warp (unity matrix) is used as an initialization. Note that if images undergo strong
displacements/rotations, an initial transformation that roughly aligns the images is necessary
(e.g., a simple euclidean/similarity transform that allows for the images showing the same image
content approximately). Use inverse warping in the second image to take an image close to the first
one, i.e. use the flag WARP_INVERSE_MAP with warpAffine or warpPerspective. See also the OpenCV
sample image_alignment.cpp that demonstrates the use of the function. Note that the function throws
an exception if algorithm does not converges.

@sa
computeECC, estimateAffine2D, estimateAffinePartial2D, findHomography
 */
CV_EXPORTS_W double findTransformECC( InputArray templateImage, InputArray inputImage,
                                      InputOutputArray warpMatrix, int motionType,
                                      TermCriteria criteria,
                                      InputArray inputMask, int gaussFiltSize);

/** @overload */
CV_EXPORTS_W
double findTransformECC(InputArray templateImage, InputArray inputImage,
    InputOutputArray warpMatrix, int motionType = MOTION_AFFINE,
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001),
    InputArray inputMask = noArray());

/** @example samples/cpp/kalman.cpp
An example using the standard Kalman filter
*/

/** @brief Kalman filter class.

The class implements a standard Kalman filter <http://en.wikipedia.org/wiki/Kalman_filter>,
@cite Welch95 . However, you can modify transitionMatrix, controlMatrix, and measurementMatrix to get
an extended Kalman filter functionality.
@note In C API when CvKalman\* kalmanFilter structure is not needed anymore, it should be released
with cvReleaseKalman(&kalmanFilter)
 */
class CV_EXPORTS_W KalmanFilter
{
public:
    CV_WRAP KalmanFilter();
    /** @overload
    @param dynamParams Dimensionality of the state.
    @param measureParams Dimensionality of the measurement.
    @param controlParams Dimensionality of the control vector.
    @param type Type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP KalmanFilter( int dynamParams, int measureParams, int controlParams = 0, int type = CV_32F );

    /** @brief Re-initializes Kalman filter. The previous content is destroyed.

    @param dynamParams Dimensionality of the state.
    @param measureParams Dimensionality of the measurement.
    @param controlParams Dimensionality of the control vector.
    @param type Type of the created matrices that should be CV_32F or CV_64F.
     */
    void init( int dynamParams, int measureParams, int controlParams = 0, int type = CV_32F );

    /** @brief Computes a predicted state.

    @param control The optional input control
     */
    CV_WRAP const Mat& predict( const Mat& control = Mat() );

    /** @brief Updates the predicted state from the measurement.

    @param measurement The measured system parameters
     */
    CV_WRAP const Mat& correct( const Mat& measurement );

    CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
    CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
    CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
    CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
    CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
    CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
    CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
    CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)

    // temporary matrices
    Mat temp1;
    Mat temp2;
    Mat temp3;
    Mat temp4;
    Mat temp5;
};


/** @brief Read a .flo file

 @param path Path to the file to be loaded

 The function readOpticalFlow loads a flow field from a file and returns it as a single matrix.
 Resulting Mat has a type CV_32FC2 - floating-point, 2-channel. First channel corresponds to the
 flow in the horizontal direction (u), second - vertical (v).
 */
CV_EXPORTS_W Mat readOpticalFlow( const String& path );
/** @brief Write a .flo to disk

 @param path Path to the file to be written
 @param flow Flow field to be stored

 The function stores a flow field in a file, returns true on success, false otherwise.
 The flow field must be a 2-channel, floating-point matrix (CV_32FC2). First channel corresponds
 to the flow in the horizontal direction (u), second - vertical (v).
 */
CV_EXPORTS_W bool writeOpticalFlow( const String& path, InputArray flow );

/**
   Base class for dense optical flow algorithms
*/
class CV_EXPORTS_W DenseOpticalFlow : public Algorithm
{
public:
    /** @brief Calculates an optical flow.

    @param I0 first 8-bit single-channel input image.
    @param I1 second input image of the same size and the same type as prev.
    @param flow computed flow image that has the same size as prev and type CV_32FC2.
     */
    CV_WRAP virtual void calc( InputArray I0, InputArray I1, InputOutputArray flow ) = 0;
    /** @brief Releases all inner buffers.
    */
    CV_WRAP virtual void collectGarbage() = 0;
};

/** @brief Base interface for sparse optical flow algorithms.
 */
class CV_EXPORTS_W SparseOpticalFlow : public Algorithm
{
public:
    /** @brief Calculates a sparse optical flow.

    @param prevImg First input image.
    @param nextImg Second input image of the same size and the same type as prevImg.
    @param prevPts Vector of 2D points for which the flow needs to be found.
    @param nextPts Output vector of 2D points containing the calculated new positions of input features in the second image.
    @param status Output status vector. Each element of the vector is set to 1 if the
                  flow for the corresponding features has been found. Otherwise, it is set to 0.
    @param err Optional output vector that contains error response for each point (inverse confidence).
     */
    CV_WRAP virtual void calc(InputArray prevImg, InputArray nextImg,
                      InputArray prevPts, InputOutputArray nextPts,
                      OutputArray status,
                      OutputArray err = cv::noArray()) = 0;
};


/** @brief Class computing a dense optical flow using the Gunnar Farneback's algorithm.
 */
class CV_EXPORTS_W FarnebackOpticalFlow : public DenseOpticalFlow
{
public:
    CV_WRAP virtual int getNumLevels() const = 0;
    CV_WRAP virtual void setNumLevels(int numLevels) = 0;

    CV_WRAP virtual double getPyrScale() const = 0;
    CV_WRAP virtual void setPyrScale(double pyrScale) = 0;

    CV_WRAP virtual bool getFastPyramids() const = 0;
    CV_WRAP virtual void setFastPyramids(bool fastPyramids) = 0;

    CV_WRAP virtual int getWinSize() const = 0;
    CV_WRAP virtual void setWinSize(int winSize) = 0;

    CV_WRAP virtual int getNumIters() const = 0;
    CV_WRAP virtual void setNumIters(int numIters) = 0;

    CV_WRAP virtual int getPolyN() const = 0;
    CV_WRAP virtual void setPolyN(int polyN) = 0;

    CV_WRAP virtual double getPolySigma() const = 0;
    CV_WRAP virtual void setPolySigma(double polySigma) = 0;

    CV_WRAP virtual int getFlags() const = 0;
    CV_WRAP virtual void setFlags(int flags) = 0;

    CV_WRAP static Ptr<FarnebackOpticalFlow> create(
            int numLevels = 5,
            double pyrScale = 0.5,
            bool fastPyramids = false,
            int winSize = 13,
            int numIters = 10,
            int polyN = 5,
            double polySigma = 1.1,
            int flags = 0);
};

/** @brief Variational optical flow refinement

This class implements variational refinement of the input flow field, i.e.
it uses input flow to initialize the minimization of the following functional:
\f$E(U) = \int_{\Omega} \delta \Psi(E_I) + \gamma \Psi(E_G) + \alpha \Psi(E_S) \f$,
where \f$E_I,E_G,E_S\f$ are color constancy, gradient constancy and smoothness terms
respectively. \f$\Psi(s^2)=\sqrt{s^2+\epsilon^2}\f$ is a robust penalizer to limit the
influence of outliers. A complete formulation and a description of the minimization
procedure can be found in @cite Brox2004
*/
class CV_EXPORTS_W VariationalRefinement : public DenseOpticalFlow
{
public:
    /** @brief @ref calc function overload to handle separate horizontal (u) and vertical (v) flow components
    (to avoid extra splits/merges) */
    CV_WRAP virtual void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v) = 0;

    /** @brief Number of outer (fixed-point) iterations in the minimization procedure.
    @see setFixedPointIterations */
    CV_WRAP virtual int getFixedPointIterations() const = 0;
    /** @copybrief getFixedPointIterations @see getFixedPointIterations */
    CV_WRAP virtual void setFixedPointIterations(int val) = 0;

    /** @brief Number of inner successive over-relaxation (SOR) iterations
        in the minimization procedure to solve the respective linear system.
    @see setSorIterations */
    CV_WRAP virtual int getSorIterations() const = 0;
    /** @copybrief getSorIterations @see getSorIterations */
    CV_WRAP virtual void setSorIterations(int val) = 0;

    /** @brief Relaxation factor in SOR
    @see setOmega */
    CV_WRAP virtual float getOmega() const = 0;
    /** @copybrief getOmega @see getOmega */
    CV_WRAP virtual void setOmega(float val) = 0;

    /** @brief Weight of the smoothness term
    @see setAlpha */
    CV_WRAP virtual float getAlpha() const = 0;
    /** @copybrief getAlpha @see getAlpha */
    CV_WRAP virtual void setAlpha(float val) = 0;

    /** @brief Weight of the color constancy term
    @see setDelta */
    CV_WRAP virtual float getDelta() const = 0;
    /** @copybrief getDelta @see getDelta */
    CV_WRAP virtual void setDelta(float val) = 0;

    /** @brief Weight of the gradient constancy term
    @see setGamma */
    CV_WRAP virtual float getGamma() const = 0;
    /** @copybrief getGamma @see getGamma */
    CV_WRAP virtual void setGamma(float val) = 0;

    /** @brief Norm value shift for robust penalizer
    @see setEpsilon */
    CV_WRAP virtual float getEpsilon() const = 0;
    /** @copybrief getEpsilon @see getEpsilon */
    CV_WRAP virtual void setEpsilon(float val) = 0;

    /** @brief Creates an instance of VariationalRefinement
    */
    CV_WRAP static Ptr<VariationalRefinement> create();
};

/** @brief DIS optical flow algorithm.

This class implements the Dense Inverse Search (DIS) optical flow algorithm. More
details about the algorithm can be found at @cite Kroeger2016 . Includes three presets with preselected
parameters to provide reasonable trade-off between speed and quality. However, even the slowest preset is
still relatively fast, use DeepFlow if you need better quality and don't care about speed.

This implementation includes several additional features compared to the algorithm described in the paper,
including spatial propagation of flow vectors (@ref getUseSpatialPropagation), as well as an option to
utilize an initial flow approximation passed to @ref calc (which is, essentially, temporal propagation,
if the previous frame's flow field is passed).
*/
class CV_EXPORTS_W DISOpticalFlow : public DenseOpticalFlow
{
public:
    enum
    {
        PRESET_ULTRAFAST = 0,
        PRESET_FAST = 1,
        PRESET_MEDIUM = 2
    };

    /** @brief Finest level of the Gaussian pyramid on which the flow is computed (zero level
        corresponds to the original image resolution). The final flow is obtained by bilinear upscaling.
        @see setFinestScale */
    CV_WRAP virtual int getFinestScale() const = 0;
    /** @copybrief getFinestScale @see getFinestScale */
    CV_WRAP virtual void setFinestScale(int val) = 0;

    /** @brief Size of an image patch for matching (in pixels). Normally, default 8x8 patches work well
        enough in most cases.
        @see setPatchSize */
    CV_WRAP virtual int getPatchSize() const = 0;
    /** @copybrief getPatchSize @see getPatchSize */
    CV_WRAP virtual void setPatchSize(int val) = 0;

    /** @brief Stride between neighbor patches. Must be less than patch size. Lower values correspond
        to higher flow quality.
        @see setPatchStride */
    CV_WRAP virtual int getPatchStride() const = 0;
    /** @copybrief getPatchStride @see getPatchStride */
    CV_WRAP virtual void setPatchStride(int val) = 0;

    /** @brief Maximum number of gradient descent iterations in the patch inverse search stage. Higher values
        may improve quality in some cases.
        @see setGradientDescentIterations */
    CV_WRAP virtual int getGradientDescentIterations() const = 0;
    /** @copybrief getGradientDescentIterations @see getGradientDescentIterations */
    CV_WRAP virtual void setGradientDescentIterations(int val) = 0;

    /** @brief Number of fixed point iterations of variational refinement per scale. Set to zero to
        disable variational refinement completely. Higher values will typically result in more smooth and
        high-quality flow.
    @see setGradientDescentIterations */
    CV_WRAP virtual int getVariationalRefinementIterations() const = 0;
    /** @copybrief getGradientDescentIterations @see getGradientDescentIterations */
    CV_WRAP virtual void setVariationalRefinementIterations(int val) = 0;

    /** @brief Weight of the smoothness term
    @see setVariationalRefinementAlpha */
    CV_WRAP virtual float getVariationalRefinementAlpha() const = 0;
    /** @copybrief getVariationalRefinementAlpha @see getVariationalRefinementAlpha */
    CV_WRAP virtual void setVariationalRefinementAlpha(float val) = 0;

    /** @brief Weight of the color constancy term
    @see setVariationalRefinementDelta */
    CV_WRAP virtual float getVariationalRefinementDelta() const = 0;
    /** @copybrief getVariationalRefinementDelta @see getVariationalRefinementDelta */
    CV_WRAP virtual void setVariationalRefinementDelta(float val) = 0;

    /** @brief Weight of the gradient constancy term
    @see setVariationalRefinementGamma */
    CV_WRAP virtual float getVariationalRefinementGamma() const = 0;
    /** @copybrief getVariationalRefinementGamma @see getVariationalRefinementGamma */
    CV_WRAP virtual void setVariationalRefinementGamma(float val) = 0;

    /** @brief Norm value shift for robust penalizer
    @see setVariationalRefinementEpsilon */
    CV_WRAP virtual float getVariationalRefinementEpsilon() const = 0;
    /** @copybrief getVariationalRefinementEpsilon @see getVariationalRefinementEpsilon */
    CV_WRAP virtual void setVariationalRefinementEpsilon(float val) = 0;


    /** @brief Whether to use mean-normalization of patches when computing patch distance. It is turned on
        by default as it typically provides a noticeable quality boost because of increased robustness to
        illumination variations. Turn it off if you are certain that your sequence doesn't contain any changes
        in illumination.
    @see setUseMeanNormalization */
    CV_WRAP virtual bool getUseMeanNormalization() const = 0;
    /** @copybrief getUseMeanNormalization @see getUseMeanNormalization */
    CV_WRAP virtual void setUseMeanNormalization(bool val) = 0;

    /** @brief Whether to use spatial propagation of good optical flow vectors. This option is turned on by
        default, as it tends to work better on average and can sometimes help recover from major errors
        introduced by the coarse-to-fine scheme employed by the DIS optical flow algorithm. Turning this
        option off can make the output flow field a bit smoother, however.
    @see setUseSpatialPropagation */
    CV_WRAP virtual bool getUseSpatialPropagation() const = 0;
    /** @copybrief getUseSpatialPropagation @see getUseSpatialPropagation */
    CV_WRAP virtual void setUseSpatialPropagation(bool val) = 0;

    /** @brief Creates an instance of DISOpticalFlow

    @param preset one of PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
    */
    CV_WRAP static Ptr<DISOpticalFlow> create(int preset = DISOpticalFlow::PRESET_FAST);
};

/** @brief Class used for calculating a sparse optical flow.

The class can calculate an optical flow for a sparse feature set using the
iterative Lucas-Kanade method with pyramids.

@sa calcOpticalFlowPyrLK

*/
class CV_EXPORTS_W SparsePyrLKOpticalFlow : public SparseOpticalFlow
{
public:
    CV_WRAP virtual Size getWinSize() const = 0;
    CV_WRAP virtual void setWinSize(Size winSize) = 0;

    CV_WRAP virtual int getMaxLevel() const = 0;
    CV_WRAP virtual void setMaxLevel(int maxLevel) = 0;

    CV_WRAP virtual TermCriteria getTermCriteria() const = 0;
    CV_WRAP virtual void setTermCriteria(TermCriteria& crit) = 0;

    CV_WRAP virtual int getFlags() const = 0;
    CV_WRAP virtual void setFlags(int flags) = 0;

    CV_WRAP virtual double getMinEigThreshold() const = 0;
    CV_WRAP virtual void setMinEigThreshold(double minEigThreshold) = 0;

    CV_WRAP static Ptr<SparsePyrLKOpticalFlow> create(
            Size winSize = Size(21, 21),
            int maxLevel = 3, TermCriteria crit =
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
            int flags = 0,
            double minEigThreshold = 1e-4);
};




/** @brief Base abstract class for the long-term tracker
 */
class CV_EXPORTS_W Tracker
{
protected:
    Tracker();
public:
    virtual ~Tracker();

    /** @brief Initialize the tracker with a known bounding box that surrounded the target
    @param image The initial frame
    @param boundingBox The initial bounding box
    */
    CV_WRAP virtual
    void init(InputArray image, const Rect& boundingBox) = 0;

    /** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The bounding box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
    */
    CV_WRAP virtual
    bool update(InputArray image, CV_OUT Rect& boundingBox) = 0;
};



/** @brief The MIL algorithm trains a classifier in an online manner to separate the object from the
background.

Multiple Instance Learning avoids the drift problem for a robust tracking. The implementation is
based on @cite MIL .

Original code can be found here <http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml>
 */
class CV_EXPORTS_W TrackerMIL : public Tracker
{
protected:
    TrackerMIL();  // use ::create()
public:
    virtual ~TrackerMIL() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        //parameters for sampler
        CV_PROP_RW float samplerInitInRadius;  //!< radius for gathering positive instances during init
        CV_PROP_RW int samplerInitMaxNegNum;  //!< # negative samples to use during init
        CV_PROP_RW float samplerSearchWinSize;  //!< size of search window
        CV_PROP_RW float samplerTrackInRadius;  //!< radius for gathering positive instances during tracking
        CV_PROP_RW int samplerTrackMaxPosNum;  //!< # positive samples to use during tracking
        CV_PROP_RW int samplerTrackMaxNegNum;  //!< # negative samples to use during tracking
        CV_PROP_RW int featureSetNumFeatures;  //!< # features
    };

    /** @brief Create MIL tracker instance
     *  @param parameters MIL parameters TrackerMIL::Params
     */
    static CV_WRAP
    Ptr<TrackerMIL> create(const TrackerMIL::Params &parameters = TrackerMIL::Params());

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};



/** @brief the GOTURN (Generic Object Tracking Using Regression Networks) tracker
 *
 *  GOTURN (@cite GOTURN) is kind of trackers based on Convolutional Neural Networks (CNN). While taking all advantages of CNN trackers,
 *  GOTURN is much faster due to offline training without online fine-tuning nature.
 *  GOTURN tracker addresses the problem of single target tracking: given a bounding box label of an object in the first frame of the video,
 *  we track that object through the rest of the video. NOTE: Current method of GOTURN does not handle occlusions; however, it is fairly
 *  robust to viewpoint changes, lighting changes, and deformations.
 *  Inputs of GOTURN are two RGB patches representing Target and Search patches resized to 227x227.
 *  Outputs of GOTURN are predicted bounding box coordinates, relative to Search patch coordinate system, in format X1,Y1,X2,Y2.
 *  Original paper is here: <http://davheld.github.io/GOTURN/GOTURN.pdf>
 *  As long as original authors implementation: <https://github.com/davheld/GOTURN#train-the-tracker>
 *  Implementation of training algorithm is placed in separately here due to 3d-party dependencies:
 *  <https://github.com/Auron-X/GOTURN_Training_Toolkit>
 *  GOTURN architecture goturn.prototxt and trained model goturn.caffemodel are accessible on opencv_extra GitHub repository.
 */
class CV_EXPORTS_W TrackerGOTURN : public Tracker
{
protected:
    TrackerGOTURN();  // use ::create()
public:
    virtual ~TrackerGOTURN() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string modelTxt;
        CV_PROP_RW std::string modelBin;
    };

    /** @brief Constructor
    @param parameters GOTURN parameters TrackerGOTURN::Params
    */
    static CV_WRAP
    Ptr<TrackerGOTURN> create(const TrackerGOTURN::Params& parameters = TrackerGOTURN::Params());

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

class CV_EXPORTS_W TrackerDaSiamRPN : public Tracker
{
protected:
    TrackerDaSiamRPN();  // use ::create()
public:
    virtual ~TrackerDaSiamRPN() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string model;
        CV_PROP_RW std::string kernel_cls1;
        CV_PROP_RW std::string kernel_r1;
        CV_PROP_RW int backend;
        CV_PROP_RW int target;
    };

    /** @brief Constructor
    @param parameters DaSiamRPN parameters TrackerDaSiamRPN::Params
    */
    static CV_WRAP
    Ptr<TrackerDaSiamRPN> create(const TrackerDaSiamRPN::Params& parameters = TrackerDaSiamRPN::Params());

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

/** @brief the Nano tracker is a super lightweight dnn-based general object tracking.
 *
 *  Nano tracker is much faster and extremely lightweight due to special model structure, the whole model size is about 1.9 MB.
 *  Nano tracker needs two models: one for feature extraction (backbone) and the another for localization (neckhead).
 *  Model download link: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
 *  Original repo is here: https://github.com/HonglinChu/NanoTrack
 *  Author: HongLinChu, 1628464345@qq.com
 */
class CV_EXPORTS_W TrackerNano : public Tracker
{
protected:
    TrackerNano();  // use ::create()
public:
    virtual ~TrackerNano() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string backbone;
        CV_PROP_RW std::string neckhead;
        CV_PROP_RW int backend;
        CV_PROP_RW int target;
    };

    /** @brief Constructor
    @param parameters NanoTrack parameters TrackerNano::Params
    */
    static CV_WRAP
    Ptr<TrackerNano> create(const TrackerNano::Params& parameters = TrackerNano::Params());

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

/** @brief the VIT tracker is a super lightweight dnn-based general object tracking.
 *
 *  VIT tracker is much faster and extremely lightweight due to special model structure, the model file is about 767KB.
 *  Model download link: https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
 *  Author: PengyuLiu, 1872918507@qq.com
 */
class CV_EXPORTS_W TrackerVit : public Tracker
{
protected:
    TrackerVit();  // use ::create()
public:
    virtual ~TrackerVit() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string net;
        CV_PROP_RW std::vector<uchar> modelBuffer;
        CV_PROP_RW int backend;
        CV_PROP_RW int target;
        CV_PROP_RW Scalar meanvalue;
        CV_PROP_RW Scalar stdvalue;
        CV_PROP_RW float tracking_score_threshold;
    };

    /** @brief Constructor
    @param parameters vit tracker parameters TrackerVit::Params
    */
    static CV_WRAP
    Ptr<TrackerVit> create(const TrackerVit::Params& parameters = TrackerVit::Params());

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    // void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    // bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

//! @} video_track

} // cv

#endif
