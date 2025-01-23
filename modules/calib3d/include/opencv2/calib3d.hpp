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

#ifndef OPENCV_CALIB3D_HPP
#define OPENCV_CALIB3D_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
#include "opencv2/core/utils/logger.hpp"

/**
  @defgroup calib3d Camera Calibration and 3D Reconstruction

The functions in this section use a so-called pinhole camera model. The view of a scene
is obtained by projecting a scene's 3D point \f$P_w\f$ into the image plane using a perspective
transformation which forms the corresponding pixel \f$p\f$. Both \f$P_w\f$ and \f$p\f$ are
represented in homogeneous coordinates, i.e. as 3D and 2D homogeneous vector respectively. You will
find a brief introduction to projective geometry, homogeneous vectors and homogeneous
transformations at the end of this section's introduction. For more succinct notation, we often drop
the 'homogeneous' and say vector instead of homogeneous vector.

The distortion-free projective transformation given by a  pinhole camera model is shown below.

\f[s \; p = A \begin{bmatrix} R|t \end{bmatrix} P_w,\f]

where \f$P_w\f$ is a 3D point expressed with respect to the world coordinate system,
\f$p\f$ is a 2D pixel in the image plane, \f$A\f$ is the camera intrinsic matrix,
\f$R\f$ and \f$t\f$ are the rotation and translation that describe the change of coordinates from
world to camera coordinate systems (or camera frame) and \f$s\f$ is the projective transformation's
arbitrary scaling and not part of the camera model.

The camera intrinsic matrix \f$A\f$ (notation used as in @cite Zhang2000 and also generally notated
as \f$K\f$) projects 3D points given in the camera coordinate system to 2D pixel coordinates, i.e.

\f[p = A P_c.\f]

The camera intrinsic matrix \f$A\f$ is composed of the focal lengths \f$f_x\f$ and \f$f_y\f$, which are
expressed in pixel units, and the principal point \f$(c_x, c_y)\f$, that is usually close to the
image center:

\f[A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1},\f]

and thus

\f[s \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1} \vecthree{X_c}{Y_c}{Z_c}.\f]

The matrix of intrinsic parameters does not depend on the scene viewed. So, once estimated, it can
be re-used as long as the focal length is fixed (in case of a zoom lens). Thus, if an image from the
camera is scaled by a factor, all of these parameters need to be scaled (multiplied/divided,
respectively) by the same factor.

The joint rotation-translation matrix \f$[R|t]\f$ is the matrix product of a projective
transformation and a homogeneous transformation. The 3-by-4 projective transformation maps 3D points
represented in camera coordinates to 2D points in the image plane and represented in normalized
camera coordinates \f$x' = X_c / Z_c\f$ and \f$y' = Y_c / Z_c\f$:

\f[Z_c \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1
\end{bmatrix}.\f]

The homogeneous transformation is encoded by the extrinsic parameters \f$R\f$ and \f$t\f$ and
represents the change of basis from world coordinate system \f$w\f$ to the camera coordinate sytem
\f$c\f$. Thus, given the representation of the point \f$P\f$ in world coordinates, \f$P_w\f$, we
obtain \f$P\f$'s representation in the camera coordinate system, \f$P_c\f$, by

\f[P_c = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} P_w,\f]

This homogeneous transformation is composed out of \f$R\f$, a 3-by-3 rotation matrix, and \f$t\f$, a
3-by-1 translation vector:

\f[\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix},
\f]

and therefore

\f[\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}.\f]

Combining the projective transformation and the homogeneous transformation, we obtain the projective
transformation that maps 3D points in world coordinates into 2D points in the image plane and in
normalized camera coordinates:

\f[Z_c \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} = \begin{bmatrix} R|t \end{bmatrix} \begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix},\f]

with \f$x' = X_c / Z_c\f$ and \f$y' = Y_c / Z_c\f$. Putting the equations for instrincs and extrinsics together, we can write out
\f$s \; p = A \begin{bmatrix} R|t \end{bmatrix} P_w\f$ as

\f[s \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}.\f]

If \f$Z_c \ne 0\f$, the transformation above is equivalent to the following,

\f[\begin{bmatrix}
u \\
v
\end{bmatrix} = \begin{bmatrix}
f_x X_c/Z_c + c_x \\
f_y Y_c/Z_c + c_y
\end{bmatrix}\f]

with

\f[\vecthree{X_c}{Y_c}{Z_c} = \begin{bmatrix}
R|t
\end{bmatrix} \begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}.\f]

The following figure illustrates the pinhole camera model.

![Pinhole camera model](pics/pinhole_camera_model.png)

Real lenses usually have some distortion, mostly radial distortion, and slight tangential distortion.
So, the above model is extended as:

\f[\begin{bmatrix}
u \\
v
\end{bmatrix} = \begin{bmatrix}
f_x x'' + c_x \\
f_y y'' + c_y
\end{bmatrix}\f]

where

\f[\begin{bmatrix}
x'' \\
y''
\end{bmatrix} = \begin{bmatrix}
x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4 \\
y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
\end{bmatrix}\f]

with

\f[r^2 = x'^2 + y'^2\f]

and

\f[\begin{bmatrix}
x'\\
y'
\end{bmatrix} = \begin{bmatrix}
X_c/Z_c \\
Y_c/Z_c
\end{bmatrix},\f]

if \f$Z_c \ne 0\f$.

The distortion parameters are the radial coefficients \f$k_1\f$, \f$k_2\f$, \f$k_3\f$, \f$k_4\f$, \f$k_5\f$, and \f$k_6\f$
,\f$p_1\f$ and \f$p_2\f$ are the tangential distortion coefficients, and \f$s_1\f$, \f$s_2\f$, \f$s_3\f$, and \f$s_4\f$,
are the thin prism distortion coefficients. Higher-order coefficients are not considered in OpenCV.

The next figures show two common types of radial distortion: barrel distortion
(\f$ 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \f$ monotonically decreasing)
and pincushion distortion (\f$ 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \f$ monotonically increasing).
Radial distortion is always monotonic for real lenses,
and if the estimator produces a non-monotonic result,
this should be considered a calibration failure.
More generally, radial distortion must be monotonic and the distortion function must be bijective.
A failed estimation result may look deceptively good near the image center
but will work poorly in e.g. AR/SFM applications.
The optimization method used in OpenCV camera calibration does not include these constraints as
the framework does not support the required integer programming and polynomial inequalities.
See [issue #15992](https://github.com/opencv/opencv/issues/15992) for additional information.

![](pics/distortion_examples.png)
![](pics/distortion_examples2.png)

In some cases, the image sensor may be tilted in order to focus an oblique plane in front of the
camera (Scheimpflug principle). This can be useful for particle image velocimetry (PIV) or
triangulation with a laser fan. The tilt causes a perspective distortion of \f$x''\f$ and
\f$y''\f$. This distortion can be modeled in the following way, see e.g. @cite Louhichi07.

\f[\begin{bmatrix}
u \\
v
\end{bmatrix} = \begin{bmatrix}
f_x x''' + c_x \\
f_y y''' + c_y
\end{bmatrix},\f]

where

\f[s\vecthree{x'''}{y'''}{1} =
\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}(\tau_x, \tau_y)}
{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\f]

and the matrix \f$R(\tau_x, \tau_y)\f$ is defined by two rotations with angular parameter
\f$\tau_x\f$ and \f$\tau_y\f$, respectively,

\f[
R(\tau_x, \tau_y) =
\vecthreethree{\cos(\tau_y)}{0}{-\sin(\tau_y)}{0}{1}{0}{\sin(\tau_y)}{0}{\cos(\tau_y)}
\vecthreethree{1}{0}{0}{0}{\cos(\tau_x)}{\sin(\tau_x)}{0}{-\sin(\tau_x)}{\cos(\tau_x)} =
\vecthreethree{\cos(\tau_y)}{\sin(\tau_y)\sin(\tau_x)}{-\sin(\tau_y)\cos(\tau_x)}
{0}{\cos(\tau_x)}{\sin(\tau_x)}
{\sin(\tau_y)}{-\cos(\tau_y)\sin(\tau_x)}{\cos(\tau_y)\cos(\tau_x)}.
\f]

In the functions below the coefficients are passed or returned as

\f[(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f]

vector. That is, if the vector contains four elements, it means that \f$k_3=0\f$ . The distortion
coefficients do not depend on the scene viewed. Thus, they also belong to the intrinsic camera
parameters. And they remain the same regardless of the captured image resolution. If, for example, a
camera has been calibrated on images of 320 x 240 resolution, absolutely the same distortion
coefficients can be used for 640 x 480 images from the same camera while \f$f_x\f$, \f$f_y\f$,
\f$c_x\f$, and \f$c_y\f$ need to be scaled appropriately.

The functions below use the above model to do the following:

-   Project 3D points to the image plane given intrinsic and extrinsic parameters.
-   Compute extrinsic parameters given intrinsic parameters, a few 3D points, and their
projections.
-   Estimate intrinsic and extrinsic camera parameters from several views of a known calibration
pattern (every view is described by several 3D-2D point correspondences).
-   Estimate the relative position and orientation of the stereo camera "heads" and compute the
*rectification* transformation that makes the camera optical axes parallel.

<B> Homogeneous Coordinates </B><br>
Homogeneous Coordinates are a system of coordinates that are used in projective geometry. Their use
allows to represent points at infinity by finite coordinates and simplifies formulas when compared
to the cartesian counterparts, e.g. they have the advantage that affine transformations can be
expressed as linear homogeneous transformation.

One obtains the homogeneous vector \f$P_h\f$ by appending a 1 along an n-dimensional cartesian
vector \f$P\f$ e.g. for a 3D cartesian vector the mapping \f$P \rightarrow P_h\f$ is:

\f[\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix} \rightarrow \begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}.\f]

For the inverse mapping \f$P_h \rightarrow P\f$, one divides all elements of the homogeneous vector
by its last element, e.g. for a 3D homogeneous vector one gets its 2D cartesian counterpart by:

\f[\begin{bmatrix}
X \\
Y \\
W
\end{bmatrix} \rightarrow \begin{bmatrix}
X / W \\
Y / W
\end{bmatrix},\f]

if \f$W \ne 0\f$.

Due to this mapping, all multiples \f$k P_h\f$, for \f$k \ne 0\f$, of a homogeneous point represent
the same point \f$P_h\f$. An intuitive understanding of this property is that under a projective
transformation, all multiples of \f$P_h\f$ are mapped to the same point. This is the physical
observation one does for pinhole cameras, as all points along a ray through the camera's pinhole are
projected to the same image point, e.g. all points along the red ray in the image of the pinhole
camera model above would be mapped to the same image coordinate. This property is also the source
for the scale ambiguity s in the equation of the pinhole camera model.

As mentioned, by using homogeneous coordinates we can express any change of basis parameterized by
\f$R\f$ and \f$t\f$ as a linear transformation, e.g. for the change of basis from coordinate system
0 to coordinate system 1 becomes:

\f[P_1 = R P_0 + t \rightarrow P_{h_1} = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} P_{h_0}.\f]

@note
    -   Many functions in this module take a camera intrinsic matrix as an input parameter. Although all
        functions assume the same structure of this parameter, they may name it differently. The
        parameter's description, however, will be clear in that a camera intrinsic matrix with the structure
        shown above is required.
    -   A calibration sample for 3 cameras in a horizontal position can be found at
        opencv_source_code/samples/cpp/3calibration.cpp
    -   A calibration sample based on a sequence of images can be found at
        opencv_source_code/samples/cpp/calibration.cpp
    -   A calibration sample in order to do 3D reconstruction can be found at
        opencv_source_code/samples/cpp/build3dmodel.cpp
    -   A calibration example on stereo calibration can be found at
        opencv_source_code/samples/cpp/stereo_calib.cpp
    -   A calibration example on stereo matching can be found at
        opencv_source_code/samples/cpp/stereo_match.cpp
    -   (Python) A camera calibration sample can be found at
        opencv_source_code/samples/python/calibrate.py

  @{
    @defgroup calib3d_fisheye Fisheye camera model

    Definitions: Let P be a point in 3D of coordinates X in the world reference frame (stored in the
    matrix X) The coordinate vector of P in the camera reference frame is:

    \f[Xc = R X + T\f]

    where R is the rotation matrix corresponding to the rotation vector om: R = rodrigues(om); call x, y
    and z the 3 coordinates of Xc:

    \f[\begin{array}{l} x = Xc_1 \\ y = Xc_2 \\ z = Xc_3 \end{array} \f]

    The pinhole projection coordinates of P is [a; b] where

    \f[\begin{array}{l} a = x / z \ and \ b = y / z \\ r^2 = a^2 + b^2 \\ \theta = atan(r) \end{array} \f]

    Fisheye distortion:

    \f[\theta_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8)\f]

    The distorted point coordinates are [x'; y'] where

    \f[\begin{array}{l} x' = (\theta_d / r) a \\ y' = (\theta_d / r) b \end{array} \f]

    Finally, conversion into pixel coordinates: The final pixel coordinates vector [u; v] where:

    \f[\begin{array}{l} u = f_x (x' + \alpha y') + c_x \\
    v = f_y y' + c_y \end{array} \f]

    Summary:
    Generic camera model @cite Kannala2006 with perspective projection and without distortion correction

  @}
 */

namespace cv
{

//! @addtogroup calib3d
//! @{

//! type of the robust estimation algorithm
enum { LMEDS  = 4,  //!< least-median of squares algorithm
       RANSAC = 8,  //!< RANSAC algorithm
       RHO    = 16, //!< RHO algorithm
       USAC_DEFAULT  = 32, //!< USAC algorithm, default settings
       USAC_PARALLEL = 33, //!< USAC, parallel version
       USAC_FM_8PTS = 34,  //!< USAC, fundamental matrix 8 points
       USAC_FAST = 35,     //!< USAC, fast settings
       USAC_ACCURATE = 36, //!< USAC, accurate settings
       USAC_PROSAC = 37,   //!< USAC, sorted points, runs PROSAC
       USAC_MAGSAC = 38    //!< USAC, runs MAGSAC++
     };

enum SolvePnPMethod {
    SOLVEPNP_ITERATIVE   = 0, //!< Pose refinement using non-linear Levenberg-Marquardt minimization scheme @cite Madsen04 @cite Eade13 \n
                              //!< Initial solution for non-planar "objectPoints" needs at least 6 points and uses the DLT algorithm. \n
                              //!< Initial solution for planar "objectPoints" needs at least 4 points and uses pose from homography decomposition.
    SOLVEPNP_EPNP        = 1, //!< EPnP: Efficient Perspective-n-Point Camera Pose Estimation @cite lepetit2009epnp
    SOLVEPNP_P3P         = 2, //!< Complete Solution Classification for the Perspective-Three-Point Problem @cite gao2003complete
    SOLVEPNP_DLS         = 3, //!< **Broken implementation. Using this flag will fallback to EPnP.** \n
                              //!< A Direct Least-Squares (DLS) Method for PnP @cite hesch2011direct
    SOLVEPNP_UPNP        = 4, //!< **Broken implementation. Using this flag will fallback to EPnP.** \n
                              //!< Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation @cite penate2013exhaustive
    SOLVEPNP_AP3P        = 5, //!< An Efficient Algebraic Solution to the Perspective-Three-Point Problem @cite Ke17
    SOLVEPNP_IPPE        = 6, //!< Infinitesimal Plane-Based Pose Estimation @cite Collins14 \n
                              //!< Object points must be coplanar.
    SOLVEPNP_IPPE_SQUARE = 7, //!< Infinitesimal Plane-Based Pose Estimation @cite Collins14 \n
                              //!< This is a special case suitable for marker pose estimation.\n
                              //!< 4 coplanar object points must be defined in the following order:
                              //!<   - point 0: [-squareLength / 2,  squareLength / 2, 0]
                              //!<   - point 1: [ squareLength / 2,  squareLength / 2, 0]
                              //!<   - point 2: [ squareLength / 2, -squareLength / 2, 0]
                              //!<   - point 3: [-squareLength / 2, -squareLength / 2, 0]
    SOLVEPNP_SQPNP       = 8, //!< SQPnP: A Consistently Fast and Globally OptimalSolution to the Perspective-n-Point Problem @cite Terzakis2020SQPnP
#ifndef CV_DOXYGEN
    SOLVEPNP_MAX_COUNT        //!< Used for count
#endif
};

enum { CALIB_CB_ADAPTIVE_THRESH = 1,
       CALIB_CB_NORMALIZE_IMAGE = 2,
       CALIB_CB_FILTER_QUADS    = 4,
       CALIB_CB_FAST_CHECK      = 8,
       CALIB_CB_EXHAUSTIVE      = 16,
       CALIB_CB_ACCURACY        = 32,
       CALIB_CB_LARGER          = 64,
       CALIB_CB_MARKER          = 128,
       CALIB_CB_PLAIN           = 256
     };

enum { CALIB_CB_SYMMETRIC_GRID  = 1,
       CALIB_CB_ASYMMETRIC_GRID = 2,
       CALIB_CB_CLUSTERING      = 4
     };

enum { CALIB_NINTRINSIC          = 18,
       CALIB_USE_INTRINSIC_GUESS = 0x00001,
       CALIB_FIX_ASPECT_RATIO    = 0x00002,
       CALIB_FIX_PRINCIPAL_POINT = 0x00004,
       CALIB_ZERO_TANGENT_DIST   = 0x00008,
       CALIB_FIX_FOCAL_LENGTH    = 0x00010,
       CALIB_FIX_K1              = 0x00020,
       CALIB_FIX_K2              = 0x00040,
       CALIB_FIX_K3              = 0x00080,
       CALIB_FIX_K4              = 0x00800,
       CALIB_FIX_K5              = 0x01000,
       CALIB_FIX_K6              = 0x02000,
       CALIB_RATIONAL_MODEL      = 0x04000,
       CALIB_THIN_PRISM_MODEL    = 0x08000,
       CALIB_FIX_S1_S2_S3_S4     = 0x10000,
       CALIB_TILTED_MODEL        = 0x40000,
       CALIB_FIX_TAUX_TAUY       = 0x80000,
       CALIB_USE_QR              = 0x100000, //!< use QR instead of SVD decomposition for solving. Faster but potentially less precise
       CALIB_FIX_TANGENT_DIST    = 0x200000,
       // only for stereo
       CALIB_FIX_INTRINSIC       = 0x00100,
       CALIB_SAME_FOCAL_LENGTH   = 0x00200,
       // for stereo rectification
       CALIB_ZERO_DISPARITY      = 0x00400,
       CALIB_USE_LU              = (1 << 17), //!< use LU instead of SVD decomposition for solving. much faster but potentially less precise
       CALIB_USE_EXTRINSIC_GUESS = (1 << 22)  //!< for stereoCalibrate
     };

//! the algorithm for finding fundamental matrix
enum { FM_7POINT = 1, //!< 7-point algorithm
       FM_8POINT = 2, //!< 8-point algorithm
       FM_LMEDS  = 4, //!< least-median algorithm. 7-point algorithm is used.
       FM_RANSAC = 8  //!< RANSAC algorithm. It needs at least 15 points. 7-point algorithm is used.
     };

enum HandEyeCalibrationMethod
{
    CALIB_HAND_EYE_TSAI         = 0, //!< A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration @cite Tsai89
    CALIB_HAND_EYE_PARK         = 1, //!< Robot Sensor Calibration: Solving AX = XB on the Euclidean Group @cite Park94
    CALIB_HAND_EYE_HORAUD       = 2, //!< Hand-eye Calibration @cite Horaud95
    CALIB_HAND_EYE_ANDREFF      = 3, //!< On-line Hand-Eye Calibration @cite Andreff99
    CALIB_HAND_EYE_DANIILIDIS   = 4  //!< Hand-Eye Calibration Using Dual Quaternions @cite Daniilidis98
};

enum RobotWorldHandEyeCalibrationMethod
{
    CALIB_ROBOT_WORLD_HAND_EYE_SHAH = 0, //!< Solving the robot-world/hand-eye calibration problem using the kronecker product @cite Shah2013SolvingTR
    CALIB_ROBOT_WORLD_HAND_EYE_LI   = 1  //!< Simultaneous robot-world and hand-eye calibration using dual-quaternions and kronecker product @cite Li2010SimultaneousRA
};

enum SamplingMethod { SAMPLING_UNIFORM=0, SAMPLING_PROGRESSIVE_NAPSAC=1, SAMPLING_NAPSAC=2,
        SAMPLING_PROSAC=3 };
enum LocalOptimMethod {LOCAL_OPTIM_NULL=0, LOCAL_OPTIM_INNER_LO=1, LOCAL_OPTIM_INNER_AND_ITER_LO=2,
        LOCAL_OPTIM_GC=3, LOCAL_OPTIM_SIGMA=4};
enum ScoreMethod {SCORE_METHOD_RANSAC=0, SCORE_METHOD_MSAC=1, SCORE_METHOD_MAGSAC=2, SCORE_METHOD_LMEDS=3};
enum NeighborSearchMethod { NEIGH_FLANN_KNN=0, NEIGH_GRID=1, NEIGH_FLANN_RADIUS=2 };
enum PolishingMethod { NONE_POLISHER=0, LSQ_POLISHER=1, MAGSAC=2, COV_POLISHER=3 };

struct CV_EXPORTS_W_SIMPLE UsacParams
{ // in alphabetical order
    CV_WRAP UsacParams();
    CV_PROP_RW double confidence;
    CV_PROP_RW bool isParallel;
    CV_PROP_RW int loIterations;
    CV_PROP_RW LocalOptimMethod loMethod;
    CV_PROP_RW int loSampleSize;
    CV_PROP_RW int maxIterations;
    CV_PROP_RW NeighborSearchMethod neighborsSearch;
    CV_PROP_RW int randomGeneratorState;
    CV_PROP_RW SamplingMethod sampler;
    CV_PROP_RW ScoreMethod score;
    CV_PROP_RW double threshold;
    CV_PROP_RW PolishingMethod final_polisher;
    CV_PROP_RW int final_polisher_iterations;
};

/** @brief Converts a rotation matrix to a rotation vector or vice versa.

@param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
@param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
@param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial
derivatives of the output array components with respect to the input array components.

\f[\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos(\theta) I + (1- \cos{\theta} ) r r^T +  \sin(\theta) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\f]

Inverse transformation can be also done easily, since

\f[\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\f]

A rotation vector is a convenient and most compact representation of a rotation matrix (since any
rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry
optimization procedures like @ref calibrateCamera, @ref stereoCalibrate, or @ref solvePnP .

@note More information about the computation of the derivative of a 3D rotation matrix with respect to its exponential coordinate
can be found in:
    - A Compact Formula for the Derivative of a 3-D Rotation in Exponential Coordinates, Guillermo Gallego, Anthony J. Yezzi @cite Gallego2014ACF

@note Useful information on SE(3) and Lie Groups can be found in:
    - A tutorial on SE(3) transformation parameterizations and on-manifold optimization, Jose-Luis Blanco @cite blanco2010tutorial
    - Lie Groups for 2D and 3D Transformation, Ethan Eade @cite Eade17
    - A micro Lie theory for state estimation in robotics, Joan Solà, Jérémie Deray, Dinesh Atchuthan @cite Sol2018AML
 */
CV_EXPORTS_W void Rodrigues( InputArray src, OutputArray dst, OutputArray jacobian = noArray() );



/** Levenberg-Marquardt solver. Starting with the specified vector of parameters it
    optimizes the target vector criteria "err"
    (finds local minima of each target vector component absolute value).

    When needed, it calls user-provided callback.
*/
class CV_EXPORTS LMSolver : public Algorithm
{
public:
    class CV_EXPORTS Callback
    {
    public:
        virtual ~Callback() {}
        /**
         computes error and Jacobian for the specified vector of parameters

         @param param the current vector of parameters
         @param err output vector of errors: err_i = actual_f_i - ideal_f_i
         @param J output Jacobian: J_ij = d(ideal_f_i)/d(param_j)

         when J=noArray(), it means that it does not need to be computed.
         Dimensionality of error vector and param vector can be different.
         The callback should explicitly allocate (with "create" method) each output array
         (unless it's noArray()).
        */
        virtual bool compute(InputArray param, OutputArray err, OutputArray J) const = 0;
    };

    /**
       Runs Levenberg-Marquardt algorithm using the passed vector of parameters as the start point.
       The final vector of parameters (whether the algorithm converged or not) is stored at the same
       vector. The method returns the number of iterations used. If it's equal to the previously specified
       maxIters, there is a big chance the algorithm did not converge.

       @param param initial/final vector of parameters.

       Note that the dimensionality of parameter space is defined by the size of param vector,
       and the dimensionality of optimized criteria is defined by the size of err vector
       computed by the callback.
    */
    virtual int run(InputOutputArray param) const = 0;

    /**
       Sets the maximum number of iterations
       @param maxIters the number of iterations
    */
    virtual void setMaxIters(int maxIters) = 0;
    /**
       Retrieves the current maximum number of iterations
    */
    virtual int getMaxIters() const = 0;

    /**
       Creates Levenberg-Marquard solver

       @param cb callback
       @param maxIters maximum number of iterations that can be further
         modified using setMaxIters() method.
    */
    static Ptr<LMSolver> create(const Ptr<LMSolver::Callback>& cb, int maxIters);
    static Ptr<LMSolver> create(const Ptr<LMSolver::Callback>& cb, int maxIters, double eps);
};



/** @example samples/cpp/tutorial_code/features2D/Homography/pose_from_homography.cpp
An example program about pose estimation from coplanar points

Check @ref tutorial_homography "the corresponding tutorial" for more details
*/

/** @brief Finds a perspective transformation between two planes.

@param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
or vector\<Point2f\> .
@param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
a vector\<Point2f\> .
@param method Method used to compute a homography matrix. The following methods are possible:
-   **0** - a regular method using all the points, i.e., the least squares method
-   @ref RANSAC - RANSAC-based robust method
-   @ref LMEDS - Least-Median robust method
-   @ref RHO - PROSAC-based robust method
@param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
(used in the RANSAC and RHO methods only). That is, if
\f[\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} \cdot \texttt{srcPoints} _i) \|_2  >  \texttt{ransacReprojThreshold}\f]
then the point \f$i\f$ is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
it usually makes sense to set this parameter somewhere in the range of 1 to 10.
@param mask Optional output mask set by a robust method ( RANSAC or LMeDS ). Note that the input
mask values are ignored.
@param maxIters The maximum number of RANSAC iterations.
@param confidence Confidence level, between 0 and 1.

The function finds and returns the perspective transformation \f$H\f$ between the source and the
destination planes:

\f[s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\f]

so that the back-projection error

\f[\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\f]

is minimized. If the parameter method is set to the default value 0, the function uses all the point
pairs to compute an initial homography estimate with a simple least-squares scheme.

However, if not all of the point pairs ( \f$srcPoints_i\f$, \f$dstPoints_i\f$ ) fit the rigid perspective
transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
the mask of inliers/outliers.

Regardless of the method, robust or not, the computed homography matrix is refined further (using
inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
re-projection error even more.

The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
noise is rather small, use the default method (method=0).

The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
determined up to a scale. If \f$h_{33}\f$ is non-zero, the matrix is normalized so that \f$h_{33}=1\f$.
@note Whenever an \f$H\f$ matrix cannot be estimated, an empty one will be returned.

@sa
getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
perspectiveTransform
 */
CV_EXPORTS_W Mat findHomography( InputArray srcPoints, InputArray dstPoints,
                                 int method = 0, double ransacReprojThreshold = 3,
                                 OutputArray mask=noArray(), const int maxIters = 2000,
                                 const double confidence = 0.995);

/** @overload */
CV_EXPORTS Mat findHomography( InputArray srcPoints, InputArray dstPoints,
                               OutputArray mask, int method = 0, double ransacReprojThreshold = 3 );


CV_EXPORTS_W Mat findHomography(InputArray srcPoints, InputArray dstPoints, OutputArray mask,
                   const UsacParams &params);

/** @brief Computes an RQ decomposition of 3x3 matrices.

@param src 3x3 input matrix.
@param mtxR Output 3x3 upper-triangular matrix.
@param mtxQ Output 3x3 orthogonal matrix.
@param Qx Optional output 3x3 rotation matrix around x-axis.
@param Qy Optional output 3x3 rotation matrix around y-axis.
@param Qz Optional output 3x3 rotation matrix around z-axis.

The function computes a RQ decomposition using the given rotations. This function is used in
#decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
and a rotation matrix.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
sequence of rotations about the three principal axes that results in the same orientation of an
object, e.g. see @cite Slabaugh . Returned three rotation matrices and corresponding three Euler angles
are only one of the possible solutions.
 */
CV_EXPORTS_W Vec3d RQDecomp3x3( InputArray src, OutputArray mtxR, OutputArray mtxQ,
                                OutputArray Qx = noArray(),
                                OutputArray Qy = noArray(),
                                OutputArray Qz = noArray());

/** @brief Decomposes a projection matrix into a rotation matrix and a camera intrinsic matrix.

@param projMatrix 3x4 input projection matrix P.
@param cameraMatrix Output 3x3 camera intrinsic matrix \f$\cameramatrix{A}\f$.
@param rotMatrix Output 3x3 external rotation matrix R.
@param transVect Output 4x1 translation vector T.
@param rotMatrixX Optional 3x3 rotation matrix around x-axis.
@param rotMatrixY Optional 3x3 rotation matrix around y-axis.
@param rotMatrixZ Optional 3x3 rotation matrix around z-axis.
@param eulerAngles Optional three-element vector containing three Euler angles of rotation in
degrees.

The function computes a decomposition of a projection matrix into a calibration and a rotation
matrix and the position of a camera.

It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
be used in OpenGL. Note, there is always more than one sequence of rotations about the three
principal axes that results in the same orientation of an object, e.g. see @cite Slabaugh . Returned
three rotation matrices and corresponding three Euler angles are only one of the possible solutions.

The function is based on #RQDecomp3x3 .
 */
CV_EXPORTS_W void decomposeProjectionMatrix( InputArray projMatrix, OutputArray cameraMatrix,
                                             OutputArray rotMatrix, OutputArray transVect,
                                             OutputArray rotMatrixX = noArray(),
                                             OutputArray rotMatrixY = noArray(),
                                             OutputArray rotMatrixZ = noArray(),
                                             OutputArray eulerAngles =noArray() );

/** @brief Computes partial derivatives of the matrix product for each multiplied matrix.

@param A First multiplied matrix.
@param B Second multiplied matrix.
@param dABdA First output derivative matrix d(A\*B)/dA of size
\f$\texttt{A.rows*B.cols} \times {A.rows*A.cols}\f$ .
@param dABdB Second output derivative matrix d(A\*B)/dB of size
\f$\texttt{A.rows*B.cols} \times {B.rows*B.cols}\f$ .

The function computes partial derivatives of the elements of the matrix product \f$A*B\f$ with regard to
the elements of each of the two input matrices. The function is used to compute the Jacobian
matrices in #stereoCalibrate but can also be used in any other similar optimization function.
 */
CV_EXPORTS_W void matMulDeriv( InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB );

/** @brief Combines two rotation-and-shift transformations.

@param rvec1 First rotation vector.
@param tvec1 First translation vector.
@param rvec2 Second rotation vector.
@param tvec2 Second translation vector.
@param rvec3 Output rotation vector of the superposition.
@param tvec3 Output translation vector of the superposition.
@param dr3dr1 Optional output derivative of rvec3 with regard to rvec1
@param dr3dt1 Optional output derivative of rvec3 with regard to tvec1
@param dr3dr2 Optional output derivative of rvec3 with regard to rvec2
@param dr3dt2 Optional output derivative of rvec3 with regard to tvec2
@param dt3dr1 Optional output derivative of tvec3 with regard to rvec1
@param dt3dt1 Optional output derivative of tvec3 with regard to tvec1
@param dt3dr2 Optional output derivative of tvec3 with regard to rvec2
@param dt3dt2 Optional output derivative of tvec3 with regard to tvec2

The functions compute:

\f[\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\f]

where \f$\mathrm{rodrigues}\f$ denotes a rotation vector to a rotation matrix transformation, and
\f$\mathrm{rodrigues}^{-1}\f$ denotes the inverse transformation. See #Rodrigues for details.

Also, the functions can compute the derivatives of the output vectors with regards to the input
vectors (see #matMulDeriv ). The functions are used inside #stereoCalibrate but can also be used in
your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
function that contains a matrix multiplication.
 */
CV_EXPORTS_W void composeRT( InputArray rvec1, InputArray tvec1,
                             InputArray rvec2, InputArray tvec2,
                             OutputArray rvec3, OutputArray tvec3,
                             OutputArray dr3dr1 = noArray(), OutputArray dr3dt1 = noArray(),
                             OutputArray dr3dr2 = noArray(), OutputArray dr3dt2 = noArray(),
                             OutputArray dt3dr1 = noArray(), OutputArray dt3dt1 = noArray(),
                             OutputArray dt3dr2 = noArray(), OutputArray dt3dt2 = noArray() );

/** @brief Projects 3D points to an image plane.

@param objectPoints Array of object points expressed wrt. the world coordinate frame. A 3xN/Nx3
1-channel or 1xN/Nx1 3-channel (or vector\<Point3f\> ), where N is the number of points in the view.
@param rvec The rotation vector (@ref Rodrigues) that, together with tvec, performs a change of
basis from world to camera coordinate system, see @ref calibrateCamera for details.
@param tvec The translation vector, see parameter description above.
@param cameraMatrix Camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$ . If the vector is empty, the zero distortion coefficients are assumed.
@param imagePoints Output array of image points, 1xN/Nx1 2-channel, or
vector\<Point2f\> .
@param jacobian Optional output 2Nx(10+\<numDistCoeffs\>) jacobian matrix of derivatives of image
points with respect to components of the rotation vector, translation vector, focal lengths,
coordinates of the principal point and the distortion coefficients. In the old interface different
components of the jacobian are returned via different output parameters.
@param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the
function assumes that the aspect ratio (\f$f_x / f_y\f$) is fixed and correspondingly adjusts the
jacobian matrix.

The function computes the 2D projections of 3D points to the image plane, given intrinsic and
extrinsic camera parameters. Optionally, the function computes Jacobians -matrices of partial
derivatives of image points coordinates (as functions of all the input parameters) with respect to
the particular parameters, intrinsic and/or extrinsic. The Jacobians are used during the global
optimization in @ref calibrateCamera, @ref solvePnP, and @ref stereoCalibrate. The function itself
can also be used to compute a re-projection error, given the current intrinsic and extrinsic
parameters.

@note By setting rvec = tvec = \f$[0, 0, 0]\f$, or by setting cameraMatrix to a 3x3 identity matrix,
or by passing zero distortion coefficients, one can get various useful partial cases of the
function. This means, one can compute the distorted coordinates for a sparse set of points or apply
a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
 */
CV_EXPORTS_W void projectPoints( InputArray objectPoints,
                                 InputArray rvec, InputArray tvec,
                                 InputArray cameraMatrix, InputArray distCoeffs,
                                 OutputArray imagePoints,
                                 OutputArray jacobian = noArray(),
                                 double aspectRatio = 0 );

/** @example samples/cpp/tutorial_code/features2D/Homography/homography_from_camera_displacement.cpp
An example program about homography from the camera displacement

Check @ref tutorial_homography "the corresponding tutorial" for more details
*/

/** @brief Finds an object pose from 3D-2D point correspondences.

@see @ref calib3d_solvePnP

This function returns the rotation and the translation vectors that transform a 3D point expressed in the object
coordinate frame to the camera coordinate frame, using different methods:
- P3P methods (@ref SOLVEPNP_P3P, @ref SOLVEPNP_AP3P): need 4 input points to return a unique solution.
- @ref SOLVEPNP_IPPE Input points must be >= 4 and object points must be coplanar.
- @ref SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
Number of input points must be 4. Object points must be defined in the following order:
  - point 0: [-squareLength / 2,  squareLength / 2, 0]
  - point 1: [ squareLength / 2,  squareLength / 2, 0]
  - point 2: [ squareLength / 2, -squareLength / 2, 0]
  - point 3: [-squareLength / 2, -squareLength / 2, 0]
- for all the other flags, number of input points must be >= 4 and object points can be in any configuration.

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2d\> can be also passed here.
@param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
the model coordinate system to the camera coordinate system.
@param tvec Output translation vector.
@param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param flags Method for solving a PnP problem: see @ref calib3d_solvePnP_flags

More information about Perspective-n-Points is described in @ref calib3d_solvePnP

@note
   -   An example of how to use solvePnP for planar augmented reality can be found at
        opencv_source_code/samples/python/plane_ar.py
   -   If you are using Python:
        - Numpy array slices won't work as input because solvePnP requires contiguous
        arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
        modules/calib3d/src/solvepnp.cpp version 2.4.9)
        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
        to its calling of #undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
        which requires 2-channel information.
        - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
        it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
        np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
   -   The methods @ref SOLVEPNP_DLS and @ref SOLVEPNP_UPNP cannot be used as the current implementations are
       unstable and sometimes give completely wrong results. If you pass one of these two
       flags, @ref SOLVEPNP_EPNP method will be used instead.
   -   The minimum number of points is 4 in the general case. In the case of @ref SOLVEPNP_P3P and @ref SOLVEPNP_AP3P
       methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
       of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
   -   With @ref SOLVEPNP_ITERATIVE method and `useExtrinsicGuess=true`, the minimum number of points is 3 (3 points
       are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
       global solution to converge.
   -   With @ref SOLVEPNP_IPPE input points must be >= 4 and object points must be coplanar.
   -   With @ref SOLVEPNP_IPPE_SQUARE this is a special case suitable for marker pose estimation.
       Number of input points must be 4. Object points must be defined in the following order:
         - point 0: [-squareLength / 2,  squareLength / 2, 0]
         - point 1: [ squareLength / 2,  squareLength / 2, 0]
         - point 2: [ squareLength / 2, -squareLength / 2, 0]
         - point 3: [-squareLength / 2, -squareLength / 2, 0]
    -  With @ref SOLVEPNP_SQPNP input points must be >= 3
 */
CV_EXPORTS_W bool solvePnP( InputArray objectPoints, InputArray imagePoints,
                            InputArray cameraMatrix, InputArray distCoeffs,
                            OutputArray rvec, OutputArray tvec,
                            bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE );

/** @brief Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

@see @ref calib3d_solvePnP

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2d\> can be also passed here.
@param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
the model coordinate system to the camera coordinate system.
@param tvec Output translation vector.
@param useExtrinsicGuess Parameter used for @ref SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param iterationsCount Number of iterations.
@param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
is the maximum allowed distance between the observed and computed point projections to consider it
an inlier.
@param confidence The probability that the algorithm produces a useful result.
@param inliers Output vector that contains indices of inliers in objectPoints and imagePoints .
@param flags Method for solving a PnP problem (see @ref solvePnP ).

The function estimates an object pose given a set of object points, their corresponding image
projections, as well as the camera intrinsic matrix and the distortion coefficients. This function finds such
a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
projections imagePoints and the projected (using @ref projectPoints ) objectPoints. The use of RANSAC
makes the function resistant to outliers.

@note
   -   An example of how to use solvePNPRansac for object detection can be found at
        opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
   -   The default method used to estimate the camera pose for the Minimal Sample Sets step
       is #SOLVEPNP_EPNP. Exceptions are:
         - if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
         - if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
   -   The method used to estimate the camera pose using all the inliers is defined by the
       flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
       the method #SOLVEPNP_EPNP will be used instead.
 */
CV_EXPORTS_W bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
                                  InputArray cameraMatrix, InputArray distCoeffs,
                                  OutputArray rvec, OutputArray tvec,
                                  bool useExtrinsicGuess = false, int iterationsCount = 100,
                                  float reprojectionError = 8.0, double confidence = 0.99,
                                  OutputArray inliers = noArray(), int flags = SOLVEPNP_ITERATIVE );


/*
Finds rotation and translation vector.
If cameraMatrix is given then run P3P. Otherwise run linear P6P and output cameraMatrix too.
*/
CV_EXPORTS_W bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
                     InputOutputArray cameraMatrix, InputArray distCoeffs,
                     OutputArray rvec, OutputArray tvec, OutputArray inliers,
                     const UsacParams &params=UsacParams());

/** @brief Finds an object pose from 3 3D-2D point correspondences.

@see @ref calib3d_solvePnP

@param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or
1x3/3x1 3-channel. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.
 vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvecs Output rotation vectors (see @ref Rodrigues ) that, together with tvecs, brings points from
the model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.
@param tvecs Output translation vectors.
@param flags Method for solving a P3P problem:
-   @ref SOLVEPNP_P3P Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).
-   @ref SOLVEPNP_AP3P Method is based on the paper of T. Ke and S. Roumeliotis.
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).

The function estimates the object pose given 3 object points, their corresponding image
projections, as well as the camera intrinsic matrix and the distortion coefficients.

@note
The solutions are sorted by reprojection errors (lowest to highest).
 */
CV_EXPORTS_W int solveP3P( InputArray objectPoints, InputArray imagePoints,
                           InputArray cameraMatrix, InputArray distCoeffs,
                           OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                           int flags );

/** @brief Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.

@see @ref calib3d_solvePnP

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
where N is the number of points. vector\<Point3d\> can also be passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2d\> can also be passed here.
@param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Input/Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
@param tvec Input/Output translation vector. Input values are used as an initial solution.
@param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.

The function refines the object pose given at least 3 object points, their corresponding image
projections, an initial solution for the rotation and translation vector,
as well as the camera intrinsic matrix and the distortion coefficients.
The function minimizes the projection error with respect to the rotation and the translation vectors, according
to a Levenberg-Marquardt iterative minimization @cite Madsen04 @cite Eade13 process.
 */
CV_EXPORTS_W void solvePnPRefineLM( InputArray objectPoints, InputArray imagePoints,
                                    InputArray cameraMatrix, InputArray distCoeffs,
                                    InputOutputArray rvec, InputOutputArray tvec,
                                    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, FLT_EPSILON));

/** @brief Refine a pose (the translation and the rotation that transform a 3D point expressed in the object coordinate frame
to the camera coordinate frame) from a 3D-2D point correspondences and starting from an initial solution.

@see @ref calib3d_solvePnP

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
where N is the number of points. vector\<Point3d\> can also be passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2d\> can also be passed here.
@param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Input/Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
the model coordinate system to the camera coordinate system. Input values are used as an initial solution.
@param tvec Input/Output translation vector. Input values are used as an initial solution.
@param criteria Criteria when to stop the Levenberg-Marquard iterative algorithm.
@param VVSlambda Gain for the virtual visual servoing control law, equivalent to the \f$\alpha\f$
gain in the Damped Gauss-Newton formulation.

The function refines the object pose given at least 3 object points, their corresponding image
projections, an initial solution for the rotation and translation vector,
as well as the camera intrinsic matrix and the distortion coefficients.
The function minimizes the projection error with respect to the rotation and the translation vectors, using a
virtual visual servoing (VVS) @cite Chaumette06 @cite Marchand16 scheme.
 */
CV_EXPORTS_W void solvePnPRefineVVS( InputArray objectPoints, InputArray imagePoints,
                                     InputArray cameraMatrix, InputArray distCoeffs,
                                     InputOutputArray rvec, InputOutputArray tvec,
                                     TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, FLT_EPSILON),
                                     double VVSlambda = 1);

/** @brief Finds an object pose from 3D-2D point correspondences.

@see @ref calib3d_solvePnP

This function returns a list of all the possible solutions (a solution is a <rotation vector, translation vector>
couple), depending on the number of input points and the chosen method:
- P3P methods (@ref SOLVEPNP_P3P, @ref SOLVEPNP_AP3P): 3 or 4 input points. Number of returned solutions can be between 0 and 4 with 3 input points.
- @ref SOLVEPNP_IPPE Input points must be >= 4 and object points must be coplanar. Returns 2 solutions.
- @ref SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
Number of input points must be 4 and 2 solutions are returned. Object points must be defined in the following order:
  - point 0: [-squareLength / 2,  squareLength / 2, 0]
  - point 1: [ squareLength / 2,  squareLength / 2, 0]
  - point 2: [ squareLength / 2, -squareLength / 2, 0]
  - point 3: [-squareLength / 2, -squareLength / 2, 0]
- for all the other flags, number of input points must be >= 4 and object points can be in any configuration.
Only 1 solution is returned.

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2d\> can be also passed here.
@param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvecs Vector of output rotation vectors (see @ref Rodrigues ) that, together with tvecs, brings points from
the model coordinate system to the camera coordinate system.
@param tvecs Vector of output translation vectors.
@param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param flags Method for solving a PnP problem: see @ref calib3d_solvePnP_flags
@param rvec Rotation vector used to initialize an iterative PnP refinement algorithm, when flag is @ref SOLVEPNP_ITERATIVE
and useExtrinsicGuess is set to true.
@param tvec Translation vector used to initialize an iterative PnP refinement algorithm, when flag is @ref SOLVEPNP_ITERATIVE
and useExtrinsicGuess is set to true.
@param reprojectionError Optional vector of reprojection error, that is the RMS error
(\f$ \text{RMSE} = \sqrt{\frac{\sum_{i}^{N} \left ( \hat{y_i} - y_i \right )^2}{N}} \f$) between the input image points
and the 3D object points projected with the estimated pose.

More information is described in @ref calib3d_solvePnP

@note
   -   An example of how to use solvePnP for planar augmented reality can be found at
        opencv_source_code/samples/python/plane_ar.py
   -   If you are using Python:
        - Numpy array slices won't work as input because solvePnP requires contiguous
        arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
        modules/calib3d/src/solvepnp.cpp version 2.4.9)
        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
        to its calling of #undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
        which requires 2-channel information.
        - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
        it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
        np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
   -   The methods @ref SOLVEPNP_DLS and @ref SOLVEPNP_UPNP cannot be used as the current implementations are
       unstable and sometimes give completely wrong results. If you pass one of these two
       flags, @ref SOLVEPNP_EPNP method will be used instead.
   -   The minimum number of points is 4 in the general case. In the case of @ref SOLVEPNP_P3P and @ref SOLVEPNP_AP3P
       methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
       of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
   -   With @ref SOLVEPNP_ITERATIVE method and `useExtrinsicGuess=true`, the minimum number of points is 3 (3 points
       are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
       global solution to converge.
   -   With @ref SOLVEPNP_IPPE input points must be >= 4 and object points must be coplanar.
   -   With @ref SOLVEPNP_IPPE_SQUARE this is a special case suitable for marker pose estimation.
       Number of input points must be 4. Object points must be defined in the following order:
         - point 0: [-squareLength / 2,  squareLength / 2, 0]
         - point 1: [ squareLength / 2,  squareLength / 2, 0]
         - point 2: [ squareLength / 2, -squareLength / 2, 0]
         - point 3: [-squareLength / 2, -squareLength / 2, 0]
 */
CV_EXPORTS_W int solvePnPGeneric( InputArray objectPoints, InputArray imagePoints,
                                  InputArray cameraMatrix, InputArray distCoeffs,
                                  OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                  bool useExtrinsicGuess = false, SolvePnPMethod flags = SOLVEPNP_ITERATIVE,
                                  InputArray rvec = noArray(), InputArray tvec = noArray(),
                                  OutputArray reprojectionError = noArray() );

/** @brief Finds an initial camera intrinsic matrix from 3D-2D point correspondences.

@param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
coordinate space. In the old interface all the per-view vectors are concatenated. See
#calibrateCamera for details.
@param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
old interface all the per-view vectors are concatenated.
@param imageSize Image size in pixels used to initialize the principal point.
@param aspectRatio If it is zero or negative, both \f$f_x\f$ and \f$f_y\f$ are estimated independently.
Otherwise, \f$f_x = f_y \cdot \texttt{aspectRatio}\f$ .

The function estimates and returns an initial camera intrinsic matrix for the camera calibration process.
Currently, the function only supports planar calibration patterns, which are patterns where each
object point has z-coordinate =0.
 */
CV_EXPORTS_W Mat initCameraMatrix2D( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints,
                                     Size imageSize, double aspectRatio = 1.0 );

/** @brief Finds the positions of internal corners of the chessboard.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   @ref CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black
and white, rather than a fixed threshold level (computed from the average image brightness).
-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with #equalizeHist before
applying fixed or adaptive thresholding.
-   @ref CALIB_CB_FILTER_QUADS Use additional criteria (like contour area, perimeter,
square-like shape) to filter out false quads extracted at the contour retrieval stage.
-   @ref CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners,
and shortcut the call if none is found. This can drastically speed up the call in the
degenerate condition when no chessboard is observed.
-   @ref CALIB_CB_PLAIN All other flags are ignored. The input image is taken as is.
No image processing is done to improve to find the checkerboard. This has the effect of speeding up the
execution of the function but could lead to not recognizing the checkerboard if the image
is not previously binarized in the appropriate manner.

The function attempts to determine whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. The function returns a non-zero value if all of the corners
are found and they are placed in a certain order (row by row, left to right in every row).
Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
squares touch each other. The detected coordinates are approximate, and to determine their positions
more accurately, the function calls #cornerSubPix. You also may use the function #cornerSubPix with
different parameters if returned coordinates are not accurate enough.

Sample usage of detecting and drawing chessboard corners: :
@code
    Size patternsize(8,6); //interior number of corners
    Mat gray = ....; //source image
    vector<Point2f> corners; //this will be filled by the detected corners

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if(patternfound)
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments. Otherwise, if there is no
border and the background is dark, the outer black squares cannot be segmented properly and so the
square grouping and ordering algorithm fails.

Use gen_pattern.py (@ref tutorial_camera_calibration_pattern) to create checkerboard.
 */
CV_EXPORTS_W bool findChessboardCorners( InputArray image, Size patternSize, OutputArray corners,
                                         int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

/*
   Checks whether the image contains chessboard of the specific size or not.
   If yes, nonzero value is returned.
*/
CV_EXPORTS_W bool checkChessboard(InputArray img, Size size);

/** @brief Finds the positions of internal corners of the chessboard using a sector based approach.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before detection.
-   @ref CALIB_CB_EXHAUSTIVE Run an exhaustive search to improve detection rate.
-   @ref CALIB_CB_ACCURACY Up sample input image to improve sub-pixel accuracy due to aliasing effects.
-   @ref CALIB_CB_LARGER The detected pattern is allowed to be larger than patternSize (see description).
-   @ref CALIB_CB_MARKER The detected pattern must have a marker (see description).
This should be used if an accurate camera calibration is required.
@param meta Optional output arrray of detected corners (CV_8UC1 and size = cv::Size(columns,rows)).
Each entry stands for one corner of the pattern and can have one of the following values:
-   0 = no meta data attached
-   1 = left-top corner of a black cell
-   2 = left-top corner of a white cell
-   3 = left-top corner of a black cell with a white marker dot
-   4 = left-top corner of a white cell with a black marker dot (pattern origin in case of markers otherwise first corner)

The function is analog to #findChessboardCorners but uses a localized radon
transformation approximated by box filters being more robust to all sort of
noise, faster on larger images and is able to directly return the sub-pixel
position of the internal chessboard corners. The Method is based on the paper
@cite duda2018 "Accurate Detection and Localization of Checkerboard Corners for
Calibration" demonstrating that the returned sub-pixel positions are more
accurate than the one returned by cornerSubPix allowing a precise camera
calibration for demanding applications.

In the case, the flags @ref CALIB_CB_LARGER or @ref CALIB_CB_MARKER are given,
the result can be recovered from the optional meta array. Both flags are
helpful to use calibration patterns exceeding the field of view of the camera.
These oversized patterns allow more accurate calibrations as corners can be
utilized, which are as close as possible to the image borders.  For a
consistent coordinate system across all images, the optional marker (see image
below) can be used to move the origin of the board to the location where the
black circle is located.

@note The function requires a white boarder with roughly the same width as one
of the checkerboard fields around the whole board to improve the detection in
various environments. In addition, because of the localized radon
transformation it is beneficial to use round corners for the field corners
which are located on the outside of the board. The following figure illustrates
a sample checkerboard optimized for the detection. However, any other checkerboard
can be used as well.

Use gen_pattern.py (@ref tutorial_camera_calibration_pattern) to create checkerboard.
![Checkerboard](pics/checkerboard_radon.png)
 */
CV_EXPORTS_AS(findChessboardCornersSBWithMeta)
bool findChessboardCornersSB(InputArray image,Size patternSize, OutputArray corners,
                             int flags,OutputArray meta);
/** @overload */
CV_EXPORTS_W inline
bool findChessboardCornersSB(InputArray image, Size patternSize, OutputArray corners,
                             int flags = 0)
{
    return findChessboardCornersSB(image, patternSize, corners, flags, noArray());
}

/** @brief Estimates the sharpness of a detected chessboard.

Image sharpness, as well as brightness, are a critical parameter for accuracte
camera calibration. For accessing these parameters for filtering out
problematic calibraiton images, this method calculates edge profiles by traveling from
black to white chessboard cell centers. Based on this, the number of pixels is
calculated required to transit from black to white. This width of the
transition area is a good indication of how sharp the chessboard is imaged
and should be below ~3.0 pixels.

@param image Gray image used to find chessboard corners
@param patternSize Size of a found chessboard pattern
@param corners Corners found by #findChessboardCornersSB
@param rise_distance Rise distance 0.8 means 10% ... 90% of the final signal strength
@param vertical By default edge responses for horizontal lines are calculated
@param sharpness Optional output array with a sharpness value for calculated edge responses (see description)

The optional sharpness array is of type CV_32FC1 and has for each calculated
profile one row with the following five entries:
* 0 = x coordinate of the underlying edge in the image
* 1 = y coordinate of the underlying edge in the image
* 2 = width of the transition area (sharpness)
* 3 = signal strength in the black cell (min brightness)
* 4 = signal strength in the white cell (max brightness)

@return Scalar(average sharpness, average min brightness, average max brightness,0)
*/
CV_EXPORTS_W Scalar estimateChessboardSharpness(InputArray image, Size patternSize, InputArray corners,
                                                float rise_distance=0.8F,bool vertical=false,
                                                OutputArray sharpness=noArray());


//! finds subpixel-accurate positions of the chessboard corners
CV_EXPORTS_W bool find4QuadCornerSubpix( InputArray img, InputOutputArray corners, Size region_size );

/** @brief Renders the detected chessboard corners.

@param image Destination image. It must be an 8-bit color image.
@param patternSize Number of inner corners per a chessboard row and column
(patternSize = cv::Size(points_per_row,points_per_column)).
@param corners Array of detected corners, the output of #findChessboardCorners.
@param patternWasFound Parameter indicating whether the complete board was found or not. The
return value of #findChessboardCorners should be passed here.

The function draws individual chessboard corners detected either as red circles if the board was not
found, or as colored corners connected with lines if the board was found.
 */
CV_EXPORTS_W void drawChessboardCorners( InputOutputArray image, Size patternSize,
                                         InputArray corners, bool patternWasFound );

/** @brief Draw axes of the world/object coordinate system from pose estimation. @sa solvePnP

@param image Input/output image. It must have 1 or 3 channels. The number of channels is not altered.
@param cameraMatrix Input 3x3 floating-point matrix of camera intrinsic parameters.
\f$\cameramatrix{A}\f$
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is empty, the zero distortion coefficients are assumed.
@param rvec Rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
the model coordinate system to the camera coordinate system.
@param tvec Translation vector.
@param length Length of the painted axes in the same unit than tvec (usually in meters).
@param thickness Line thickness of the painted axes.

This function draws the axes of the world/object coordinate system w.r.t. to the camera frame.
OX is drawn in red, OY in green and OZ in blue.
 */
CV_EXPORTS_W void drawFrameAxes(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                                InputArray rvec, InputArray tvec, float length, int thickness=3);

struct CV_EXPORTS_W_SIMPLE CirclesGridFinderParameters
{
    CV_WRAP CirclesGridFinderParameters();
    CV_PROP_RW cv::Size2f densityNeighborhoodSize;
    CV_PROP_RW float minDensity;
    CV_PROP_RW int kmeansAttempts;
    CV_PROP_RW int minDistanceToAddKeypoint;
    CV_PROP_RW int keypointScale;
    CV_PROP_RW float minGraphConfidence;
    CV_PROP_RW float vertexGain;
    CV_PROP_RW float vertexPenalty;
    CV_PROP_RW float existingVertexGain;
    CV_PROP_RW float edgeGain;
    CV_PROP_RW float edgePenalty;
    CV_PROP_RW float convexHullFactor;
    CV_PROP_RW float minRNGEdgeSwitchDist;

    enum GridType
    {
      SYMMETRIC_GRID, ASYMMETRIC_GRID
    };
    GridType gridType;

    CV_PROP_RW float squareSize; //!< Distance between two adjacent points. Used by CALIB_CB_CLUSTERING.
    CV_PROP_RW float maxRectifiedDistance; //!< Max deviation from prediction. Used by CALIB_CB_CLUSTERING.
};

#ifndef DISABLE_OPENCV_3_COMPATIBILITY
typedef CirclesGridFinderParameters CirclesGridFinderParameters2;
#endif

/** @brief Finds centers in the grid of circles.

@param image grid view of input circles; it must be an 8-bit grayscale or color image.
@param patternSize number of circles per row and column
( patternSize = Size(points_per_row, points_per_colum) ).
@param centers output array of detected centers.
@param flags various operation flags that can be one of the following values:
-   @ref CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
-   @ref CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
-   @ref CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to
perspective distortions but much more sensitive to background clutter.
@param blobDetector feature detector that finds blobs like dark circles on light background.
                    If `blobDetector` is NULL then `image` represents Point2f array of candidates.
@param parameters struct for finding circles in a grid pattern.

The function attempts to determine whether the input image contains a grid of circles. If it is, the
function locates centers of the circles. The function returns a non-zero value if all of the centers
have been found and they have been placed in a certain order (row by row, left to right in every
row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0.

Sample usage of detecting and drawing the centers of circles: :
@code
    Size patternsize(7,7); //number of centers
    Mat gray = ...; //source image
    vector<Point2f> centers; //this will be filled by the detected centers

    bool patternfound = findCirclesGrid(gray, patternsize, centers);

    drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments.
 */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags,
                                   const Ptr<FeatureDetector> &blobDetector,
                                   const CirclesGridFinderParameters& parameters);

/** @overload */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags = CALIB_CB_SYMMETRIC_GRID,
                                   const Ptr<FeatureDetector> &blobDetector = SimpleBlobDetector::create());

/** @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration
pattern.

@param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
the calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer
vector contains as many elements as the number of pattern views. If the same calibration pattern
is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
possible to use partially occluded patterns or even different patterns in different views. Then,
the vectors will be different. Although the points are 3D, they all lie in the calibration pattern's
XY coordinate plane (thus 0 in the Z-coordinate), if the used calibration pattern is a planar rig.
In the old interface all the vectors of object points from different views are concatenated
together.
@param imagePoints In the new interface it is a vector of vectors of the projections of calibration
pattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and
objectPoints.size(), and imagePoints[i].size() and objectPoints[i].size() for each i, must be equal,
respectively. In the old interface all the vectors of object points from different views are
concatenated together.
@param imageSize Size of the image used only to initialize the camera intrinsic matrix.
@param cameraMatrix Input/output 3x3 floating-point camera intrinsic matrix
\f$\cameramatrix{A}\f$ . If @ref CALIB_USE_INTRINSIC_GUESS
and/or @ref CALIB_FIX_ASPECT_RATIO, @ref CALIB_FIX_PRINCIPAL_POINT or @ref CALIB_FIX_FOCAL_LENGTH
are specified, some or all of fx, fy, cx, cy must be initialized before calling the function.
@param distCoeffs Input/output vector of distortion coefficients
\f$\distcoeffs\f$.
@param rvecs Output vector of rotation vectors (@ref Rodrigues ) estimated for each pattern view
(e.g. std::vector<cv::Mat>>). That is, each i-th rotation vector together with the corresponding
i-th translation vector (see the next output parameter description) brings the calibration pattern
from the object coordinate space (in which object points are specified) to the camera coordinate
space. In more technical terms, the tuple of the i-th rotation and translation vector performs
a change of basis from object coordinate space to camera coordinate space. Due to its duality, this
tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate
space.
@param tvecs Output vector of translation vectors estimated for each pattern view, see parameter
describtion above.
@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
parameters. Order of deviations values:
\f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
parameters. Order of deviations values: \f$(R_0, T_0, \dotsc , R_{M - 1}, T_{M - 1})\f$ where M is
the number of pattern views. \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   @ref CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of
fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
center ( imageSize is used), and focal distances are computed in a least-squares fashion.
Note, that if intrinsic parameters are known, there is no need to use this function just to
estimate extrinsic parameters. Use @ref solvePnP instead.
-   @ref CALIB_FIX_PRINCIPAL_POINT The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when
 @ref CALIB_USE_INTRINSIC_GUESS is set too.
-   @ref CALIB_FIX_ASPECT_RATIO The functions consider only fy as a free parameter. The
ratio fx/fy stays the same as in the input cameraMatrix . When
 @ref CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
ignored, only their ratio is computed and used further.
-   @ref CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients \f$(p_1, p_2)\f$ are set
to zeros and stay zero.
-   @ref CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global optimization if
 @ref CALIB_USE_INTRINSIC_GUESS is set.
-   @ref CALIB_FIX_K1,..., @ref CALIB_FIX_K6 The corresponding radial distortion
coefficient is not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is
set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_RATIONAL_MODEL Coefficients k4, k5, and k6 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the rational model and return 8 coefficients or more.
-   @ref CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients or more.
-   @ref CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients.
-   @ref CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

@return the overall RMS re-projection error.

The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
views. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object
points and their corresponding 2D projections in each view must be specified. That may be achieved
by using an object with known geometry and easily detectable feature points. Such an object is
called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
a calibration rig (see @ref findChessboardCorners). Currently, initialization of intrinsic
parameters (when @ref CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
be used as long as initial cameraMatrix is provided.

The algorithm performs the following steps:

-   Compute the initial intrinsic parameters (the option only available for planar calibration
    patterns) or read them from the input parameters. The distortion coefficients are all set to
    zeros initially unless some of CALIB_FIX_K? are specified.

-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
    done using @ref solvePnP .

-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
    that is, the total sum of squared distances between the observed feature points imagePoints and
    the projected (using the current estimates for camera parameters and the poses) object points
    objectPoints. See @ref projectPoints for details.

@note
    If you use a non-square (i.e. non-N-by-N) grid and @ref findChessboardCorners for calibration,
    and @ref calibrateCamera returns bad values (zero distortion coefficients, \f$c_x\f$ and
    \f$c_y\f$ very far from the image center, and/or large differences between \f$f_x\f$ and
    \f$f_y\f$ (ratios of 10:1 or more)), then you are probably using patternSize=cvSize(rows,cols)
    instead of using patternSize=cvSize(cols,rows) in @ref findChessboardCorners.

@note
    The function may throw exceptions, if unsupported combination of parameters is provided or
    the system is underconstrained.

@sa
   calibrateCameraRO, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate,
   undistort
 */
CV_EXPORTS_AS(calibrateCameraExtended) double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @overload */
CV_EXPORTS_W double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

This function is an extension of #calibrateCamera with the method of releasing object which was
proposed in @cite strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar
targets (calibration plates), this method can dramatically improve the precision of the estimated
camera parameters. Both the object-releasing method and standard method are supported by this
function. Use the parameter **iFixedPoint** for method selection. In the internal implementation,
#calibrateCamera is a wrapper for this function.

@param objectPoints Vector of vectors of calibration pattern points in the calibration pattern
coordinate space. See #calibrateCamera for details. If the method of releasing object to be used,
the identical calibration board must be used in each view and it must be fully visible, and all
objectPoints[i] must be the same and all points should be roughly close to a plane. **The calibration
target has to be rigid, or at least static if the camera (rather than the calibration target) is
shifted for grabbing images.**
@param imagePoints Vector of vectors of the projections of calibration pattern points. See
#calibrateCamera for details.
@param imageSize Size of the image used only to initialize the intrinsic camera matrix.
@param iFixedPoint The index of the 3D object point in objectPoints[0] to be fixed. It also acts as
a switch for calibration method selection. If object-releasing method to be used, pass in the
parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will
make standard calibration method selected. Usually the top-right corner point of the calibration
board grid is recommended to be fixed when object-releasing method being utilized. According to
\cite strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front
and objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and
newObjPoints are only possible if coordinates of these three fixed points are accurate enough.
@param cameraMatrix Output 3x3 floating-point camera matrix. See #calibrateCamera for details.
@param distCoeffs Output vector of distortion coefficients. See #calibrateCamera for details.
@param rvecs Output vector of rotation vectors estimated for each pattern view. See #calibrateCamera
for details.
@param tvecs Output vector of translation vectors estimated for each pattern view.
@param newObjPoints The updated output vector of calibration pattern points. The coordinates might
be scaled based on three fixed points. The returned coordinates are accurate only if the above
mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter
is ignored with standard calibration method.
@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
See #calibrateCamera for details.
@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
See #calibrateCamera for details.
@param stdDeviationsObjPoints Output vector of standard deviations estimated for refined coordinates
of calibration pattern points. It has the same size and order as objectPoints[0] vector. This
parameter is ignored with standard calibration method.
 @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of some predefined values. See
#calibrateCamera for details. If the method of releasing object is used, the calibration time may
be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially
less precise and less stable in some rare cases.
@param criteria Termination criteria for the iterative optimization algorithm.

@return the overall RMS re-projection error.

The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
views. The algorithm is based on @cite Zhang2000, @cite BouguetMCT and @cite strobl2011iccv. See
#calibrateCamera for other detailed explanations.
@sa
   calibrateCamera, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
 */
CV_EXPORTS_AS(calibrateCameraROExtended) double calibrateCameraRO( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize, int iFixedPoint,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray newObjPoints,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray stdDeviationsObjPoints,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @overload */
CV_EXPORTS_W double calibrateCameraRO( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize, int iFixedPoint,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray newObjPoints,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @brief Computes useful camera characteristics from the camera intrinsic matrix.

@param cameraMatrix Input camera intrinsic matrix that can be estimated by #calibrateCamera or
#stereoCalibrate .
@param imageSize Input image size in pixels.
@param apertureWidth Physical width in mm of the sensor.
@param apertureHeight Physical height in mm of the sensor.
@param fovx Output field of view in degrees along the horizontal sensor axis.
@param fovy Output field of view in degrees along the vertical sensor axis.
@param focalLength Focal length of the lens in mm.
@param principalPoint Principal point in mm.
@param aspectRatio \f$f_y/f_x\f$

The function computes various useful camera characteristics from the previously estimated camera
matrix.

@note
   Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for
    the chessboard pitch (it can thus be any value).
 */
CV_EXPORTS_W void calibrationMatrixValues( InputArray cameraMatrix, Size imageSize,
                                           double apertureWidth, double apertureHeight,
                                           CV_OUT double& fovx, CV_OUT double& fovy,
                                           CV_OUT double& focalLength, CV_OUT Point2d& principalPoint,
                                           CV_OUT double& aspectRatio );

/** @brief Calibrates a stereo camera set up. This function finds the intrinsic parameters
for each of the two cameras and the extrinsic parameters between the two cameras.

@param objectPoints Vector of vectors of the calibration pattern points. The same structure as
in @ref calibrateCamera. For each pattern view, both cameras need to see the same object
points. Therefore, objectPoints.size(), imagePoints1.size(), and imagePoints2.size() need to be
equal as well as objectPoints[i].size(), imagePoints1[i].size(), and imagePoints2[i].size() need to
be equal for each i.
@param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
observed by the first camera. The same structure as in @ref calibrateCamera.
@param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
observed by the second camera. The same structure as in @ref calibrateCamera.
@param cameraMatrix1 Input/output camera intrinsic matrix for the first camera, the same as in
@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
@param distCoeffs1 Input/output vector of distortion coefficients, the same as in
@ref calibrateCamera.
@param cameraMatrix2 Input/output second camera intrinsic matrix for the second camera. See description for
cameraMatrix1.
@param distCoeffs2 Input/output lens distortion coefficients for the second camera. See
description for distCoeffs1.
@param imageSize Size of the image used only to initialize the camera intrinsic matrices.
@param R Output rotation matrix. Together with the translation vector T, this matrix brings
points given in the first camera's coordinate system to points in the second camera's
coordinate system. In more technical terms, the tuple of R and T performs a change of basis
from the first camera's coordinate system to the second camera's coordinate system. Due to its
duality, this tuple is equivalent to the position of the first camera with respect to the
second camera coordinate system.
@param T Output translation vector, see description above.
@param E Output essential matrix.
@param F Output fundamental matrix.
@param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
coordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each
i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter
description) brings the calibration pattern from the object coordinate space (in which object points are
specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms,
the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space
to camera coordinate space of the first camera of the stereo pair.
@param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
of previous output parameter ( rvecs ).
@param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   @ref CALIB_FIX_INTRINSIC Fix cameraMatrix? and distCoeffs? so that only R, T, E, and F
matrices are estimated.
-   @ref CALIB_USE_INTRINSIC_GUESS Optimize some or all of the intrinsic parameters
according to the specified flags. Initial values are provided by the user.
-   @ref CALIB_USE_EXTRINSIC_GUESS R and T contain valid initial values that are optimized further.
Otherwise R and T are initialized to the median value of the pattern views (each dimension separately).
-   @ref CALIB_FIX_PRINCIPAL_POINT Fix the principal points during the optimization.
-   @ref CALIB_FIX_FOCAL_LENGTH Fix \f$f^{(j)}_x\f$ and \f$f^{(j)}_y\f$ .
-   @ref CALIB_FIX_ASPECT_RATIO Optimize \f$f^{(j)}_y\f$ . Fix the ratio \f$f^{(j)}_x/f^{(j)}_y\f$
.
-   @ref CALIB_SAME_FOCAL_LENGTH Enforce \f$f^{(0)}_x=f^{(1)}_x\f$ and \f$f^{(0)}_y=f^{(1)}_y\f$ .
-   @ref CALIB_ZERO_TANGENT_DIST Set tangential distortion coefficients for each camera to
zeros and fix there.
-   @ref CALIB_FIX_K1,..., @ref CALIB_FIX_K6 Do not change the corresponding radial
distortion coefficient during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set,
the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_RATIONAL_MODEL Enable coefficients k4, k5, and k6. To provide the backward
compatibility, this extra flag should be explicitly specified to make the calibration
function use the rational model and return 8 coefficients. If the flag is not set, the
function computes and returns only 5 distortion coefficients.
-   @ref CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   @ref CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   @ref CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

The function estimates the transformation between two cameras making a stereo pair. If one computes
the poses of an object relative to the first camera and to the second camera,
( \f$R_1\f$,\f$T_1\f$ ) and (\f$R_2\f$,\f$T_2\f$), respectively, for a stereo camera where the
relative position and orientation between the two cameras are fixed, then those poses definitely
relate to each other. This means, if the relative position and orientation (\f$R\f$,\f$T\f$) of the
two cameras is known, it is possible to compute (\f$R_2\f$,\f$T_2\f$) when (\f$R_1\f$,\f$T_1\f$) is
given. This is what the described function does. It computes (\f$R\f$,\f$T\f$) such that:

\f[R_2=R R_1\f]
\f[T_2=R T_1 + T.\f]

Therefore, one can compute the coordinate representation of a 3D point for the second camera's
coordinate system when given the point's coordinate representation in the first camera's coordinate
system:

\f[\begin{bmatrix}
X_2 \\
Y_2 \\
Z_2 \\
1
\end{bmatrix} = \begin{bmatrix}
R & T \\
0 & 1
\end{bmatrix} \begin{bmatrix}
X_1 \\
Y_1 \\
Z_1 \\
1
\end{bmatrix}.\f]


Optionally, it computes the essential matrix E:

\f[E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} R\f]

where \f$T_i\f$ are components of the translation vector \f$T\f$ : \f$T=[T_0, T_1, T_2]^T\f$ .
And the function can also compute the fundamental matrix F:

\f[F = cameraMatrix2^{-T}\cdot E \cdot cameraMatrix1^{-1}\f]

Besides the stereo-related information, the function can also perform a full calibration of each of
the two cameras. However, due to the high dimensionality of the parameter space and noise in the
input data, the function can diverge from the correct solution. If the intrinsic parameters can be
estimated with high accuracy for each of the cameras individually (for example, using
#calibrateCamera ), you are recommended to do so and then pass @ref CALIB_FIX_INTRINSIC flag to the
function along with the computed intrinsic parameters. Otherwise, if all the parameters are
estimated at once, it makes sense to restrict some parameters, for example, pass
 @ref CALIB_SAME_FOCAL_LENGTH and @ref CALIB_ZERO_TANGENT_DIST flags, which is usually a
reasonable assumption.

Similarly to #calibrateCamera, the function minimizes the total re-projection error for all the
points in all the available views from both cameras. The function returns the final value of the
re-projection error.
 */
CV_EXPORTS_AS(stereoCalibrateExtended) double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray perViewErrors, int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );

/// @overload
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, OutputArray R,OutputArray T, OutputArray E, OutputArray F,
                                     int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );

/// @overload
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArray perViewErrors, int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );

/** @brief Computes rectification transforms for each head of a calibrated stereo camera.

@param cameraMatrix1 First camera intrinsic matrix.
@param distCoeffs1 First camera distortion parameters.
@param cameraMatrix2 Second camera intrinsic matrix.
@param distCoeffs2 Second camera distortion parameters.
@param imageSize Size of the image used for stereo calibration.
@param R Rotation matrix from the coordinate system of the first camera to the second camera,
see @ref stereoCalibrate.
@param T Translation vector from the coordinate system of the first camera to the second camera,
see @ref stereoCalibrate.
@param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
brings points given in the unrectified first camera's coordinate system to points in the rectified
first camera's coordinate system. In more technical terms, it performs a change of basis from the
unrectified first camera's coordinate system to the rectified first camera's coordinate system.
@param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
brings points given in the unrectified second camera's coordinate system to points in the rectified
second camera's coordinate system. In more technical terms, it performs a change of basis from the
unrectified second camera's coordinate system to the rectified second camera's coordinate system.
@param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
camera, i.e. it projects points given in the rectified first camera coordinate system into the
rectified first camera's image.
@param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
camera, i.e. it projects points given in the rectified first camera coordinate system into the
rectified second camera's image.
@param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see @ref reprojectImageTo3D).
@param flags Operation flags that may be zero or @ref CALIB_ZERO_DISPARITY . If the flag is set,
the function makes the principal points of each camera have the same pixel coordinates in the
rectified views. And if the flag is not set, the function may still shift the images in the
horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
useful image area.
@param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
images are zoomed and shifted so that only valid pixels are visible (no black areas after
rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
pixels from the original images from the cameras are retained in the rectified images (no source
image pixels are lost). Any intermediate value yields an intermediate result between
those two extreme cases.
@param newImageSize New image resolution after rectification. The same size should be passed to
#initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
preserve details in the original image, especially when there is a big radial distortion.
@param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).
@param validPixROI2 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).

The function computes the rotation matrices for each camera that (virtually) make both camera image
planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
the dense stereo correspondence problem. The function takes the matrices computed by #stereoCalibrate
as input. As output, it provides two rotation matrices and also two projection matrices in the new
coordinates. The function distinguishes the following two cases:

-   **Horizontal stereo**: the first and the second camera views are shifted relative to each other
    mainly along the x-axis (with possible small vertical shift). In the rectified images, the
    corresponding epipolar lines in the left and right cameras are horizontal and have the same
    y-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix}
                        f & 0 & cx_1 & 0 \\
                        0 & f & cy & 0 \\
                        0 & 0 & 1 & 0
                     \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix}
                        f & 0 & cx_2 & T_x \cdot f \\
                        0 & f & cy & 0 \\
                        0 & 0 & 1 & 0
                     \end{bmatrix} ,\f]

    \f[\texttt{Q} = \begin{bmatrix}
                        1 & 0 & 0 & -cx_1 \\
                        0 & 1 & 0 & -cy \\
                        0 & 0 & 0 & f \\
                        0 & 0 & -\frac{1}{T_x} & \frac{cx_1 - cx_2}{T_x}
                    \end{bmatrix} \f]

    where \f$T_x\f$ is a horizontal shift between the cameras and \f$cx_1=cx_2\f$ if
    @ref CALIB_ZERO_DISPARITY is set.

-   **Vertical stereo**: the first and the second camera views are shifted relative to each other
    mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
    lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix}
                        f & 0 & cx & 0 \\
                        0 & f & cy_1 & 0 \\
                        0 & 0 & 1 & 0
                     \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix}
                        f & 0 & cx & 0 \\
                        0 & f & cy_2 & T_y \cdot f \\
                        0 & 0 & 1 & 0
                     \end{bmatrix},\f]

    \f[\texttt{Q} = \begin{bmatrix}
                        1 & 0 & 0 & -cx \\
                        0 & 1 & 0 & -cy_1 \\
                        0 & 0 & 0 & f \\
                        0 & 0 & -\frac{1}{T_y} & \frac{cy_1 - cy_2}{T_y}
                    \end{bmatrix} \f]

    where \f$T_y\f$ is a vertical shift between the cameras and \f$cy_1=cy_2\f$ if
    @ref CALIB_ZERO_DISPARITY is set.

As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
matrices. The matrices, together with R1 and R2 , can then be passed to #initUndistortRectifyMap to
initialize the rectification map for each camera.

See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
the corresponding image regions. This means that the images are well rectified, which is what most
stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
their interiors are all valid pixels.

![image](pics/stereo_undistort.jpg)
 */
CV_EXPORTS_W void stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 Size imageSize, InputArray R, InputArray T,
                                 OutputArray R1, OutputArray R2,
                                 OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags = CALIB_ZERO_DISPARITY,
                                 double alpha = -1, Size newImageSize = Size(),
                                 CV_OUT Rect* validPixROI1 = 0, CV_OUT Rect* validPixROI2 = 0 );

/** @brief Computes a rectification transform for an uncalibrated stereo camera.

@param points1 Array of feature points in the first image.
@param points2 The corresponding points in the second image. The same formats as in
#findFundamentalMat are supported.
@param F Input fundamental matrix. It can be computed from the same set of point pairs using
#findFundamentalMat .
@param imgSize Size of the image.
@param H1 Output rectification homography matrix for the first image.
@param H2 Output rectification homography matrix for the second image.
@param threshold Optional threshold used to filter out the outliers. If the parameter is greater
than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points
for which \f$|\texttt{points2[i]}^T \cdot \texttt{F} \cdot \texttt{points1[i]}|>\texttt{threshold}\f$ )
are rejected prior to computing the homographies. Otherwise, all the points are considered inliers.

The function computes the rectification transformations without knowing intrinsic parameters of the
cameras and their relative position in the space, which explains the suffix "uncalibrated". Another
related difference from #stereoRectify is that the function outputs not the rectification
transformations in the object (3D) space, but the planar perspective transformations encoded by the
homography matrices H1 and H2 . The function implements the algorithm @cite Hartley99 .

@note
   While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily
    depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion,
    it would be better to correct it before computing the fundamental matrix and calling this
    function. For example, distortion coefficients can be estimated for each head of stereo camera
    separately by using #calibrateCamera . Then, the images can be corrected using #undistort , or
    just the point coordinates can be corrected with #undistortPoints .
 */
CV_EXPORTS_W bool stereoRectifyUncalibrated( InputArray points1, InputArray points2,
                                             InputArray F, Size imgSize,
                                             OutputArray H1, OutputArray H2,
                                             double threshold = 5 );

//! computes the rectification transformations for 3-head camera, where all the heads are on the same line.
CV_EXPORTS_W float rectify3Collinear( InputArray cameraMatrix1, InputArray distCoeffs1,
                                      InputArray cameraMatrix2, InputArray distCoeffs2,
                                      InputArray cameraMatrix3, InputArray distCoeffs3,
                                      InputArrayOfArrays imgpt1, InputArrayOfArrays imgpt3,
                                      Size imageSize, InputArray R12, InputArray T12,
                                      InputArray R13, InputArray T13,
                                      OutputArray R1, OutputArray R2, OutputArray R3,
                                      OutputArray P1, OutputArray P2, OutputArray P3,
                                      OutputArray Q, double alpha, Size newImgSize,
                                      CV_OUT Rect* roi1, CV_OUT Rect* roi2, int flags );

/** @brief Returns the new camera intrinsic matrix based on the free scaling parameter.

@param cameraMatrix Input camera intrinsic matrix.
@param distCoeffs Input vector of distortion coefficients
\f$\distcoeffs\f$. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param imageSize Original image size.
@param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
valid) and 1 (when all the source image pixels are retained in the undistorted image). See
#stereoRectify for details.
@param newImgSize Image size after rectification. By default, it is set to imageSize .
@param validPixROI Optional output rectangle that outlines all-good-pixels region in the
undistorted image. See roi1, roi2 description in #stereoRectify .
@param centerPrincipalPoint Optional flag that indicates whether in the new camera intrinsic matrix the
principal point should be at the image center or not. By default, the principal point is chosen to
best fit a subset of the source image (determined by alpha) to the corrected image.
@return new_camera_matrix Output new camera intrinsic matrix.

The function computes and returns the optimal new camera intrinsic matrix based on the free scaling parameter.
By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
image pixels if there is valuable information in the corners alpha=1 , or get something in between.
When alpha\>0 , the undistorted result is likely to have some black pixels corresponding to
"virtual" pixels outside of the captured distorted image. The original camera intrinsic matrix, distortion
coefficients, the computed new camera intrinsic matrix, and newImageSize should be passed to
#initUndistortRectifyMap to produce the maps for #remap .
 */
CV_EXPORTS_W Mat getOptimalNewCameraMatrix( InputArray cameraMatrix, InputArray distCoeffs,
                                            Size imageSize, double alpha, Size newImgSize = Size(),
                                            CV_OUT Rect* validPixROI = 0,
                                            bool centerPrincipalPoint = false);

/** @brief Computes Hand-Eye calibration: \f$_{}^{g}\textrm{T}_c\f$

@param[in] R_gripper2base Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the robot base frame (\f$_{}^{b}\textrm{T}_g\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from gripper frame to robot base frame.
@param[in] t_gripper2base Translation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the robot base frame (\f$_{}^{b}\textrm{T}_g\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from gripper frame to robot base frame.
@param[in] R_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the target frame to the camera frame (\f$_{}^{c}\textrm{T}_t\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from calibration target frame to camera frame.
@param[in] t_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the target frame to the camera frame (\f$_{}^{c}\textrm{T}_t\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from calibration target frame to camera frame.
@param[out] R_cam2gripper Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a point
expressed in the camera frame to the gripper frame (\f$_{}^{g}\textrm{T}_c\f$).
@param[out] t_cam2gripper Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a point
expressed in the camera frame to the gripper frame (\f$_{}^{g}\textrm{T}_c\f$).
@param[in] method One of the implemented Hand-Eye calibration method, see cv::HandEyeCalibrationMethod

The function performs the Hand-Eye calibration using various methods. One approach consists in estimating the
rotation then the translation (separable solutions) and the following methods are implemented:
  - R. Tsai, R. Lenz A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/EyeCalibration \cite Tsai89
  - F. Park, B. Martin Robot Sensor Calibration: Solving AX = XB on the Euclidean Group \cite Park94
  - R. Horaud, F. Dornaika Hand-Eye Calibration \cite Horaud95

Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions),
with the following implemented methods:
  - N. Andreff, R. Horaud, B. Espiau On-line Hand-Eye Calibration \cite Andreff99
  - K. Daniilidis Hand-Eye Calibration Using Dual Quaternions \cite Daniilidis98

The following picture describes the Hand-Eye calibration problem where the transformation between a camera ("eye")
mounted on a robot gripper ("hand") has to be estimated. This configuration is called eye-in-hand.

The eye-to-hand configuration consists in a static camera observing a calibration pattern mounted on the robot
end-effector. The transformation from the camera to the robot base frame can then be estimated by inputting
the suitable transformations to the function, see below.

![](pics/hand-eye_figure.png)

The calibration procedure is the following:
  - a static calibration pattern is used to estimate the transformation between the target frame
  and the camera frame
  - the robot gripper is moved in order to acquire several poses
  - for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for
  instance the robot kinematics
\f[
    \begin{bmatrix}
    X_b\\
    Y_b\\
    Z_b\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{b}\textrm{R}_g & _{}^{b}\textrm{t}_g \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
\f]
  - for each pose, the homogeneous transformation between the calibration target frame and the camera frame is recorded using
  for instance a pose estimation method (PnP) from 2D-3D point correspondences
\f[
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{c}\textrm{R}_t & _{}^{c}\textrm{t}_t \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_t\\
    Y_t\\
    Z_t\\
    1
    \end{bmatrix}
\f]

The Hand-Eye calibration procedure returns the following homogeneous transformation
\f[
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{g}\textrm{R}_c & _{}^{g}\textrm{t}_c \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
\f]

This problem is also known as solving the \f$\mathbf{A}\mathbf{X}=\mathbf{X}\mathbf{B}\f$ equation:
  - for an eye-in-hand configuration
\f[
    \begin{align*}
    ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &=
    \hspace{0.1em} ^{b}{\textrm{T}_g}^{(2)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\

    (^{b}{\textrm{T}_g}^{(2)})^{-1} \hspace{0.2em} ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c &=
    \hspace{0.1em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\

    \textrm{A}_i \textrm{X} &= \textrm{X} \textrm{B}_i \\
    \end{align*}
\f]

  - for an eye-to-hand configuration
\f[
    \begin{align*}
    ^{g}{\textrm{T}_b}^{(1)} \hspace{0.2em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &=
    \hspace{0.1em} ^{g}{\textrm{T}_b}^{(2)} \hspace{0.2em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\

    (^{g}{\textrm{T}_b}^{(2)})^{-1} \hspace{0.2em} ^{g}{\textrm{T}_b}^{(1)} \hspace{0.2em} ^{b}\textrm{T}_c &=
    \hspace{0.1em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\

    \textrm{A}_i \textrm{X} &= \textrm{X} \textrm{B}_i \\
    \end{align*}
\f]

\note
Additional information can be found on this [website](http://campar.in.tum.de/Chair/HandEyeCalibration).
\note
A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation.
So at least 3 different poses are required, but it is strongly recommended to use many more poses.

 */
CV_EXPORTS_W void calibrateHandEye( InputArrayOfArrays R_gripper2base, InputArrayOfArrays t_gripper2base,
                                    InputArrayOfArrays R_target2cam, InputArrayOfArrays t_target2cam,
                                    OutputArray R_cam2gripper, OutputArray t_cam2gripper,
                                    HandEyeCalibrationMethod method=CALIB_HAND_EYE_TSAI );

/** @brief Computes Robot-World/Hand-Eye calibration: \f$_{}^{w}\textrm{T}_b\f$ and \f$_{}^{c}\textrm{T}_g\f$

@param[in] R_world2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the world frame to the camera frame (\f$_{}^{c}\textrm{T}_w\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from world frame to the camera frame.
@param[in] t_world2cam Translation part extracted from the homogeneous matrix that transforms a point
expressed in the world frame to the camera frame (\f$_{}^{c}\textrm{T}_w\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from world frame to the camera frame.
@param[in] R_base2gripper Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the gripper frame (\f$_{}^{g}\textrm{T}_b\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from robot base frame to the gripper frame.
@param[in] t_base2gripper Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the gripper frame (\f$_{}^{g}\textrm{T}_b\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from robot base frame to the gripper frame.
@param[out] R_base2world Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the world frame (\f$_{}^{w}\textrm{T}_b\f$).
@param[out] t_base2world Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the world frame (\f$_{}^{w}\textrm{T}_b\f$).
@param[out] R_gripper2cam Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the camera frame (\f$_{}^{c}\textrm{T}_g\f$).
@param[out] t_gripper2cam Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the camera frame (\f$_{}^{c}\textrm{T}_g\f$).
@param[in] method One of the implemented Robot-World/Hand-Eye calibration method, see cv::RobotWorldHandEyeCalibrationMethod

The function performs the Robot-World/Hand-Eye calibration using various methods. One approach consists in estimating the
rotation then the translation (separable solutions):
  - M. Shah, Solving the robot-world/hand-eye calibration problem using the kronecker product \cite Shah2013SolvingTR

Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions),
with the following implemented method:
  - A. Li, L. Wang, and D. Wu, Simultaneous robot-world and hand-eye calibration using dual-quaternions and kronecker product \cite Li2010SimultaneousRA

The following picture describes the Robot-World/Hand-Eye calibration problem where the transformations between a robot and a world frame
and between a robot gripper ("hand") and a camera ("eye") mounted at the robot end-effector have to be estimated.

![](pics/robot-world_hand-eye_figure.png)

The calibration procedure is the following:
  - a static calibration pattern is used to estimate the transformation between the target frame
  and the camera frame
  - the robot gripper is moved in order to acquire several poses
  - for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for
  instance the robot kinematics
\f[
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{g}\textrm{R}_b & _{}^{g}\textrm{t}_b \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_b\\
    Y_b\\
    Z_b\\
    1
    \end{bmatrix}
\f]
  - for each pose, the homogeneous transformation between the calibration target frame (the world frame) and the camera frame is recorded using
  for instance a pose estimation method (PnP) from 2D-3D point correspondences
\f[
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{c}\textrm{R}_w & _{}^{c}\textrm{t}_w \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_w\\
    Y_w\\
    Z_w\\
    1
    \end{bmatrix}
\f]

The Robot-World/Hand-Eye calibration procedure returns the following homogeneous transformations
\f[
    \begin{bmatrix}
    X_w\\
    Y_w\\
    Z_w\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{w}\textrm{R}_b & _{}^{w}\textrm{t}_b \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_b\\
    Y_b\\
    Z_b\\
    1
    \end{bmatrix}
\f]
\f[
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{c}\textrm{R}_g & _{}^{c}\textrm{t}_g \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
\f]

This problem is also known as solving the \f$\mathbf{A}\mathbf{X}=\mathbf{Z}\mathbf{B}\f$ equation, with:
  - \f$\mathbf{A} \Leftrightarrow \hspace{0.1em} _{}^{c}\textrm{T}_w\f$
  - \f$\mathbf{X} \Leftrightarrow \hspace{0.1em} _{}^{w}\textrm{T}_b\f$
  - \f$\mathbf{Z} \Leftrightarrow \hspace{0.1em} _{}^{c}\textrm{T}_g\f$
  - \f$\mathbf{B} \Leftrightarrow \hspace{0.1em} _{}^{g}\textrm{T}_b\f$

\note
At least 3 measurements are required (input vectors size must be greater or equal to 3).

 */
CV_EXPORTS_W void calibrateRobotWorldHandEye( InputArrayOfArrays R_world2cam, InputArrayOfArrays t_world2cam,
                                              InputArrayOfArrays R_base2gripper, InputArrayOfArrays t_base2gripper,
                                              OutputArray R_base2world, OutputArray t_base2world,
                                              OutputArray R_gripper2cam, OutputArray t_gripper2cam,
                                              RobotWorldHandEyeCalibrationMethod method=CALIB_ROBOT_WORLD_HAND_EYE_SHAH );

/** @brief Converts points from Euclidean to homogeneous space.

@param src Input vector of N-dimensional points.
@param dst Output vector of N+1-dimensional points.

The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of
point coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).
 */
CV_EXPORTS_W void convertPointsToHomogeneous( InputArray src, OutputArray dst );

/** @brief Converts points from homogeneous to Euclidean space.

@param src Input vector of N-dimensional points.
@param dst Output vector of N-1-dimensional points.

The function converts points homogeneous to Euclidean space using perspective projection. That is,
each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the
output point coordinates will be (0,0,0,...).
 */
CV_EXPORTS_W void convertPointsFromHomogeneous( InputArray src, OutputArray dst );

/** @brief Converts points to/from homogeneous coordinates.

@param src Input array or vector of 2D, 3D, or 4D points.
@param dst Output vector of 2D, 3D, or 4D points.

The function converts 2D or 3D points from/to homogeneous coordinates by calling either
#convertPointsToHomogeneous or #convertPointsFromHomogeneous.

@note The function is obsolete. Use one of the previous two functions instead.
 */
CV_EXPORTS void convertPointsHomogeneous( InputArray src, OutputArray dst );

/** @brief Calculates a fundamental matrix from the corresponding points in two images.

@param points1 Array of N points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param method Method for computing a fundamental matrix.
-   @ref FM_7POINT for a 7-point algorithm. \f$N = 7\f$
-   @ref FM_8POINT for an 8-point algorithm. \f$N \ge 8\f$
-   @ref FM_RANSAC for the RANSAC algorithm. \f$N \ge 8\f$
-   @ref FM_LMEDS for the LMedS algorithm. \f$N \ge 8\f$
@param ransacReprojThreshold Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param confidence Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level
of confidence (probability) that the estimated matrix is correct.
@param[out] mask optional output mask
@param maxIters The maximum number of robust method iterations.

The epipolar geometry is described by the following equation:

\f[[p_2; 1]^T F [p_1; 1] = 0\f]

where \f$F\f$ is a fundamental matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the
second images, respectively.

The function calculates the fundamental matrix using one of four methods listed above and returns
the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point
algorithm, the function may return up to 3 solutions ( \f$9 \times 3\f$ matrix that stores all 3
matrices sequentially).

The calculated fundamental matrix may be passed further to #computeCorrespondEpilines that finds the
epipolar lines corresponding to the specified points. It can also be passed to
#stereoRectifyUncalibrated to compute the rectification transformation. :
@code
    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    Mat fundamental_matrix =
     findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
@endcode
 */
CV_EXPORTS_W Mat findFundamentalMat( InputArray points1, InputArray points2,
                                     int method, double ransacReprojThreshold, double confidence,
                                     int maxIters, OutputArray mask = noArray() );

/** @overload */
CV_EXPORTS_W Mat findFundamentalMat( InputArray points1, InputArray points2,
                                     int method = FM_RANSAC,
                                     double ransacReprojThreshold = 3., double confidence = 0.99,
                                     OutputArray mask = noArray() );

/** @overload */
CV_EXPORTS Mat findFundamentalMat( InputArray points1, InputArray points2,
                                   OutputArray mask, int method = FM_RANSAC,
                                   double ransacReprojThreshold = 3., double confidence = 0.99 );


CV_EXPORTS_W Mat findFundamentalMat( InputArray points1, InputArray points2,
                        OutputArray mask, const UsacParams &params);

/** @brief Calculates an essential matrix from the corresponding points in two images.

@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should
be floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1.
@param cameraMatrix Camera intrinsic matrix \f$\cameramatrix{A}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera intrinsic matrix. If this assumption does not hold for your use case, use another
function overload or #undistortPoints with `P = cv::NoArray()` for both cameras to transform image
points to normalized image coordinates, which are valid for the identity camera intrinsic matrix.
When passing these coordinates, pass the identity matrix for this parameter.
@param method Method for computing an essential matrix.
-   @ref RANSAC for the RANSAC algorithm.
-   @ref LMEDS for the LMedS algorithm.
@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
confidence (probability) that the estimated matrix is correct.
@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
for the other points. The array is computed only in the RANSAC and LMedS methods.
@param maxIters The maximum number of robust method iterations.

This function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 .
@cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation:

\f[[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\f]

where \f$E\f$ is an essential matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the
second images, respectively. The result of this function may be passed further to
#decomposeEssentialMat or #recoverPose to recover the relative pose between cameras.
 */
CV_EXPORTS_W
Mat findEssentialMat(
    InputArray points1, InputArray points2,
    InputArray cameraMatrix, int method = RANSAC,
    double prob = 0.999, double threshold = 1.0,
    int maxIters = 1000, OutputArray mask = noArray()
);

/** @overload */
CV_EXPORTS
Mat findEssentialMat(
    InputArray points1, InputArray points2,
    InputArray cameraMatrix, int method,
    double prob, double threshold,
    OutputArray mask
);  // TODO remove from OpenCV 5.0

/** @overload
@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should
be floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param focal focal length of the camera. Note that this function assumes that points1 and points2
are feature points from cameras with same focal length and principal point.
@param pp principal point of the camera.
@param method Method for computing a fundamental matrix.
-   @ref RANSAC for the RANSAC algorithm.
-   @ref LMEDS for the LMedS algorithm.
@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
confidence (probability) that the estimated matrix is correct.
@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
for the other points. The array is computed only in the RANSAC and LMedS methods.
@param maxIters The maximum number of robust method iterations.

This function differs from the one above that it computes camera intrinsic matrix from focal length and
principal point:

\f[A =
\begin{bmatrix}
f & 0 & x_{pp}  \\
0 & f & y_{pp}  \\
0 & 0 & 1
\end{bmatrix}\f]
 */
CV_EXPORTS_W
Mat findEssentialMat(
    InputArray points1, InputArray points2,
    double focal = 1.0, Point2d pp = Point2d(0, 0),
    int method = RANSAC, double prob = 0.999,
    double threshold = 1.0, int maxIters = 1000,
    OutputArray mask = noArray()
);

/** @overload */
CV_EXPORTS
Mat findEssentialMat(
    InputArray points1, InputArray points2,
    double focal, Point2d pp,
    int method, double prob,
    double threshold, OutputArray mask
);  // TODO remove from OpenCV 5.0

/** @brief Calculates an essential matrix from the corresponding points in two images from potentially two different cameras.

@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should
be floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1.
@param cameraMatrix1 Camera matrix for the first camera \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param cameraMatrix2 Camera matrix for the second camera \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs1 Input vector of distortion coefficients for the first camera
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param distCoeffs2 Input vector of distortion coefficients for the second camera
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param method Method for computing an essential matrix.
-   @ref RANSAC for the RANSAC algorithm.
-   @ref LMEDS for the LMedS algorithm.
@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
confidence (probability) that the estimated matrix is correct.
@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
for the other points. The array is computed only in the RANSAC and LMedS methods.

This function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 .
@cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation:

\f[[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\f]

where \f$E\f$ is an essential matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the
second images, respectively. The result of this function may be passed further to
#decomposeEssentialMat or  #recoverPose to recover the relative pose between cameras.
 */
CV_EXPORTS_W Mat findEssentialMat( InputArray points1, InputArray points2,
                                 InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 int method = RANSAC,
                                 double prob = 0.999, double threshold = 1.0,
                                 OutputArray mask = noArray() );


CV_EXPORTS_W Mat findEssentialMat( InputArray points1, InputArray points2,
                      InputArray cameraMatrix1, InputArray cameraMatrix2,
                      InputArray dist_coeff1, InputArray dist_coeff2, OutputArray mask,
                      const UsacParams &params);

/** @brief Decompose an essential matrix to possible rotations and translation.

@param E The input essential matrix.
@param R1 One possible rotation matrix.
@param R2 Another possible rotation matrix.
@param t One possible translation.

This function decomposes the essential matrix E using svd decomposition @cite HartleyZ00. In
general, four possible poses exist for the decomposition of E. They are \f$[R_1, t]\f$,
\f$[R_1, -t]\f$, \f$[R_2, t]\f$, \f$[R_2, -t]\f$.

If E gives the epipolar constraint \f$[p_2; 1]^T A^{-T} E A^{-1} [p_1; 1] = 0\f$ between the image
points \f$p_1\f$ in the first image and \f$p_2\f$ in second image, then any of the tuples
\f$[R_1, t]\f$, \f$[R_1, -t]\f$, \f$[R_2, t]\f$, \f$[R_2, -t]\f$ is a change of basis from the first
camera's coordinate system to the second camera's coordinate system. However, by decomposing E, one
can only get the direction of the translation. For this reason, the translation t is returned with
unit length.
 */
CV_EXPORTS_W void decomposeEssentialMat( InputArray E, OutputArray R1, OutputArray R2, OutputArray t );

/** @brief Recovers the relative camera rotation and the translation from corresponding points in two images from two different cameras, using cheirality check. Returns the number of
inliers that pass the check.

@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param cameraMatrix1 Input/output camera matrix for the first camera, the same as in
@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
@param distCoeffs1 Input/output vector of distortion coefficients, the same as in
@ref calibrateCamera.
@param cameraMatrix2 Input/output camera matrix for the first camera, the same as in
@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
@param distCoeffs2 Input/output vector of distortion coefficients, the same as in
@ref calibrateCamera.
@param E The output essential matrix.
@param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
that performs a change of basis from the first camera's coordinate system to the second camera's
coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
described below.
@param t Output translation vector. This vector is obtained by @ref decomposeEssentialMat and
therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
length.
@param method Method for computing an essential matrix.
-   @ref RANSAC for the RANSAC algorithm.
-   @ref LMEDS for the LMedS algorithm.
@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
confidence (probability) that the estimated matrix is correct.
@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
recover pose. In the output mask only inliers which pass the cheirality check.

This function decomposes an essential matrix using @ref decomposeEssentialMat and then verifies
possible pose hypotheses by doing cheirality check. The cheirality check means that the
triangulated 3D points should have positive depth. Some details can be found in @cite Nister03.

This function can be used to process the output E and mask from @ref findEssentialMat. In this
scenario, points1 and points2 are the same input for findEssentialMat.:
@code
    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    // Input: camera calibration of both cameras, for example using intrinsic chessboard calibration.
    Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;

    // Output: Essential matrix, relative rotation and relative translation.
    Mat E, R, t, mask;

    recoverPose(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, E, R, t, mask);
@endcode
 */
CV_EXPORTS_W int recoverPose( InputArray points1, InputArray points2,
                            InputArray cameraMatrix1, InputArray distCoeffs1,
                            InputArray cameraMatrix2, InputArray distCoeffs2,
                            OutputArray E, OutputArray R, OutputArray t,
                            int method = cv::RANSAC, double prob = 0.999, double threshold = 1.0,
                            InputOutputArray mask = noArray());

/** @brief Recovers the relative camera rotation and the translation from an estimated essential
matrix and the corresponding points in two images, using chirality check. Returns the number of
inliers that pass the check.

@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param cameraMatrix Camera intrinsic matrix \f$\cameramatrix{A}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera intrinsic matrix.
@param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
that performs a change of basis from the first camera's coordinate system to the second camera's
coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
described below.
@param t Output translation vector. This vector is obtained by @ref decomposeEssentialMat and
therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
length.
@param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
inliers in points1 and points2 for the given essential matrix E. Only these inliers will be used to
recover pose. In the output mask only inliers which pass the chirality check.

This function decomposes an essential matrix using @ref decomposeEssentialMat and then verifies
possible pose hypotheses by doing chirality check. The chirality check means that the
triangulated 3D points should have positive depth. Some details can be found in @cite Nister03.

This function can be used to process the output E and mask from @ref findEssentialMat. In this
scenario, points1 and points2 are the same input for #findEssentialMat :
@code
    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    // cametra matrix with both focal lengths = 1, and principal point = (0, 0)
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    Mat E, R, t, mask;

    E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
@endcode
 */
CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            InputArray cameraMatrix, OutputArray R, OutputArray t,
                            InputOutputArray mask = noArray() );

/** @overload
@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
that performs a change of basis from the first camera's coordinate system to the second camera's
coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
description below.
@param t Output translation vector. This vector is obtained by @ref decomposeEssentialMat and
therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
length.
@param focal Focal length of the camera. Note that this function assumes that points1 and points2
are feature points from cameras with same focal length and principal point.
@param pp principal point of the camera.
@param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
inliers in points1 and points2 for the given essential matrix E. Only these inliers will be used to
recover pose. In the output mask only inliers which pass the chirality check.

This function differs from the one above that it computes camera intrinsic matrix from focal length and
principal point:

\f[A =
\begin{bmatrix}
f & 0 & x_{pp}  \\
0 & f & y_{pp}  \\
0 & 0 & 1
\end{bmatrix}\f]
 */
CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            OutputArray R, OutputArray t,
                            double focal = 1.0, Point2d pp = Point2d(0, 0),
                            InputOutputArray mask = noArray() );

/** @overload
@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1.
@param cameraMatrix Camera intrinsic matrix \f$\cameramatrix{A}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera intrinsic matrix.
@param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
that performs a change of basis from the first camera's coordinate system to the second camera's
coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
description below.
@param t Output translation vector. This vector is obtained by @ref decomposeEssentialMat and
therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
length.
@param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite
points).
@param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
inliers in points1 and points2 for the given essential matrix E. Only these inliers will be used to
recover pose. In the output mask only inliers which pass the chirality check.
@param triangulatedPoints 3D points which were reconstructed by triangulation.

This function differs from the one above that it outputs the triangulated 3D point that are used for
the chirality check.
 */
CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            InputArray cameraMatrix, OutputArray R, OutputArray t, double distanceThresh, InputOutputArray mask = noArray(),
                            OutputArray triangulatedPoints = noArray());

/** @brief For points in an image of a stereo pair, computes the corresponding epilines in the other image.

@param points Input points. \f$N \times 1\f$ or \f$1 \times N\f$ matrix of type CV_32FC2 or
vector\<Point2f\> .
@param whichImage Index of the image (1 or 2) that contains the points .
@param F Fundamental matrix that can be estimated using #findFundamentalMat or #stereoRectify .
@param lines Output vector of the epipolar lines corresponding to the points in the other image.
Each line \f$ax + by + c=0\f$ is encoded by 3 numbers \f$(a, b, c)\f$ .

For every point in one of the two images of a stereo pair, the function finds the equation of the
corresponding epipolar line in the other image.

From the fundamental matrix definition (see #findFundamentalMat ), line \f$l^{(2)}_i\f$ in the second
image for the point \f$p^{(1)}_i\f$ in the first image (when whichImage=1 ) is computed as:

\f[l^{(2)}_i = F p^{(1)}_i\f]

And vice versa, when whichImage=2, \f$l^{(1)}_i\f$ is computed from \f$p^{(2)}_i\f$ as:

\f[l^{(1)}_i = F^T p^{(2)}_i\f]

Line coefficients are defined up to a scale. They are normalized so that \f$a_i^2+b_i^2=1\f$ .
 */
CV_EXPORTS_W void computeCorrespondEpilines( InputArray points, int whichImage,
                                             InputArray F, OutputArray lines );

/** @brief This function reconstructs 3-dimensional points (in homogeneous coordinates) by using
their observations with a stereo camera.

@param projMatr1 3x4 projection matrix of the first camera, i.e. this matrix projects 3D points
given in the world's coordinate system into the first image.
@param projMatr2 3x4 projection matrix of the second camera, i.e. this matrix projects 3D points
given in the world's coordinate system into the second image.
@param projPoints1 2xN array of feature points in the first image. In the case of the c++ version,
it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
@param projPoints2 2xN array of corresponding points in the second image. In the case of the c++
version, it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
@param points4D 4xN array of reconstructed points in homogeneous coordinates. These points are
returned in the world's coordinate system.

@note
   Keep in mind that all input data should be of float type in order for this function to work.

@note
   If the projection matrices from @ref stereoRectify are used, then the returned points are
   represented in the first camera's rectified coordinate system.

@sa
   reprojectImageTo3D
 */
CV_EXPORTS_W void triangulatePoints( InputArray projMatr1, InputArray projMatr2,
                                     InputArray projPoints1, InputArray projPoints2,
                                     OutputArray points4D );

/** @brief Refines coordinates of corresponding points.

@param F 3x3 fundamental matrix.
@param points1 1xN array containing the first set of points.
@param points2 1xN array containing the second set of points.
@param newPoints1 The optimized points1.
@param newPoints2 The optimized points2.

The function implements the Optimal Triangulation Method (see Multiple View Geometry @cite HartleyZ00 for details).
For each given point correspondence points1[i] \<-\> points2[i], and a fundamental matrix F, it
computes the corrected correspondences newPoints1[i] \<-\> newPoints2[i] that minimize the geometric
error \f$d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2\f$ (where \f$d(a,b)\f$ is the
geometric distance between points \f$a\f$ and \f$b\f$ ) subject to the epipolar constraint
\f$newPoints2^T \cdot F \cdot newPoints1 = 0\f$ .
 */
CV_EXPORTS_W void correctMatches( InputArray F, InputArray points1, InputArray points2,
                                  OutputArray newPoints1, OutputArray newPoints2 );

/** @brief Filters off small noise blobs (speckles) in the disparity map

@param img The input 16-bit signed disparity image
@param newVal The disparity value used to paint-off the speckles
@param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not
affected by the algorithm
@param maxDiff Maximum difference between neighbor disparity pixels to put them into the same
blob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point
disparity map, where disparity values are multiplied by 16, this scale factor should be taken into
account when specifying this parameter value.
@param buf The optional temporary buffer to avoid memory allocation within the function.
 */
CV_EXPORTS_W void filterSpeckles( InputOutputArray img, double newVal,
                                  int maxSpeckleSize, double maxDiff,
                                  InputOutputArray buf = noArray() );

//! computes valid disparity ROI from the valid ROIs of the rectified images (that are returned by #stereoRectify)
CV_EXPORTS_W Rect getValidDisparityROI( Rect roi1, Rect roi2,
                                        int minDisparity, int numberOfDisparities,
                                        int blockSize );

//! validates disparity using the left-right check. The matrix "cost" should be computed by the stereo correspondence algorithm
CV_EXPORTS_W void validateDisparity( InputOutputArray disparity, InputArray cost,
                                     int minDisparity, int numberOfDisparities,
                                     int disp12MaxDisp = 1 );

/** @brief Reprojects a disparity image to 3D space.

@param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
floating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no
fractional bits. If the disparity is 16-bit signed format, as computed by @ref StereoBM or
@ref StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before
being used here.
@param _3dImage Output 3-channel floating-point image of the same size as disparity. Each element of
_3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. If one
uses Q obtained by @ref stereoRectify, then the returned points are represented in the first
camera's rectified coordinate system.
@param Q \f$4 \times 4\f$ perspective transformation matrix that can be obtained with
@ref stereoRectify.
@param handleMissingValues Indicates, whether the function should handle missing values (i.e.
points where the disparity was not computed). If handleMissingValues=true, then pixels with the
minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
to 3D points with a very large Z value (currently set to 10000).
@param ddepth The optional output array depth. If it is -1, the output image will have CV_32F
depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.

The function transforms a single-channel disparity map to a 3-channel image representing a 3D
surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
computes:

\f[\begin{bmatrix}
X \\
Y \\
Z \\
W
\end{bmatrix} = Q \begin{bmatrix}
x \\
y \\
\texttt{disparity} (x,y) \\
1
\end{bmatrix}.\f]

@sa
   To reproject a sparse set of points {(x,y,d),...} to 3D space, use perspectiveTransform.
 */
CV_EXPORTS_W void reprojectImageTo3D( InputArray disparity,
                                      OutputArray _3dImage, InputArray Q,
                                      bool handleMissingValues = false,
                                      int ddepth = -1 );

/** @brief Calculates the Sampson Distance between two points.

The function cv::sampsonDistance calculates and returns the first order approximation of the geometric error as:
\f[
sd( \texttt{pt1} , \texttt{pt2} )=
\frac{(\texttt{pt2}^t \cdot \texttt{F} \cdot \texttt{pt1})^2}
{((\texttt{F} \cdot \texttt{pt1})(0))^2 +
((\texttt{F} \cdot \texttt{pt1})(1))^2 +
((\texttt{F}^t \cdot \texttt{pt2})(0))^2 +
((\texttt{F}^t \cdot \texttt{pt2})(1))^2}
\f]
The fundamental matrix may be calculated using the #findFundamentalMat function. See @cite HartleyZ00 11.4.3 for details.
@param pt1 first homogeneous 2d point
@param pt2 second homogeneous 2d point
@param F fundamental matrix
@return The computed Sampson distance.
*/
CV_EXPORTS_W double sampsonDistance(InputArray pt1, InputArray pt2, InputArray F);

/** @brief Computes an optimal affine transformation between two 3D point sets.

It computes
\f[
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix}
=
\begin{bmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
X\\
Y\\
Z\\
\end{bmatrix}
+
\begin{bmatrix}
b_1\\
b_2\\
b_3\\
\end{bmatrix}
\f]

@param src First input 3D point set containing \f$(X,Y,Z)\f$.
@param dst Second input 3D point set containing \f$(x,y,z)\f$.
@param out Output 3D affine transformation matrix \f$3 \times 4\f$ of the form
\f[
\begin{bmatrix}
a_{11} & a_{12} & a_{13} & b_1\\
a_{21} & a_{22} & a_{23} & b_2\\
a_{31} & a_{32} & a_{33} & b_3\\
\end{bmatrix}
\f]
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as
an inlier.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.

The function estimates an optimal 3D affine transformation between two 3D point sets using the
RANSAC algorithm.
 */
CV_EXPORTS_W  int estimateAffine3D(InputArray src, InputArray dst,
                                   OutputArray out, OutputArray inliers,
                                   double ransacThreshold = 3, double confidence = 0.99);

/** @brief Computes an optimal affine transformation between two 3D point sets.

It computes \f$R,s,t\f$ minimizing \f$\sum{i} dst_i - c \cdot R \cdot src_i \f$
where \f$R\f$ is a 3x3 rotation matrix, \f$t\f$ is a 3x1 translation vector and \f$s\f$ is a
scalar size value. This is an implementation of the algorithm by Umeyama \cite umeyama1991least .
The estimated affine transform has a homogeneous scale which is a subclass of affine
transformations with 7 degrees of freedom. The paired point sets need to comprise at least 3
points each.

@param src First input 3D point set.
@param dst Second input 3D point set.
@param scale If null is passed, the scale parameter c will be assumed to be 1.0.
Else the pointed-to variable will be set to the optimal scale.
@param force_rotation If true, the returned rotation will never be a reflection.
This might be unwanted, e.g. when optimizing a transform between a right- and a
left-handed coordinate system.
@return 3D affine transformation matrix \f$3 \times 4\f$ of the form
\f[T =
\begin{bmatrix}
R & t\\
\end{bmatrix}
\f]

 */
CV_EXPORTS_W   cv::Mat estimateAffine3D(InputArray src, InputArray dst,
                                        CV_OUT double* scale = nullptr, bool force_rotation = true);

/** @brief Computes an optimal translation between two 3D point sets.
 *
 * It computes
 * \f[
 * \begin{bmatrix}
 * x\\
 * y\\
 * z\\
 * \end{bmatrix}
 * =
 * \begin{bmatrix}
 * X\\
 * Y\\
 * Z\\
 * \end{bmatrix}
 * +
 * \begin{bmatrix}
 * b_1\\
 * b_2\\
 * b_3\\
 * \end{bmatrix}
 * \f]
 *
 * @param src First input 3D point set containing \f$(X,Y,Z)\f$.
 * @param dst Second input 3D point set containing \f$(x,y,z)\f$.
 * @param out Output 3D translation vector \f$3 \times 1\f$ of the form
 * \f[
 * \begin{bmatrix}
 * b_1 \\
 * b_2 \\
 * b_3 \\
 * \end{bmatrix}
 * \f]
 * @param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
 * @param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as
 * an inlier.
 * @param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
 * between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
 * significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
 *
 * The function estimates an optimal 3D translation between two 3D point sets using the
 * RANSAC algorithm.
 *  */
CV_EXPORTS_W  int estimateTranslation3D(InputArray src, InputArray dst,
                                        OutputArray out, OutputArray inliers,
                                        double ransacThreshold = 3, double confidence = 0.99);

/** @brief Computes an optimal affine transformation between two 2D point sets.

It computes
\f[
\begin{bmatrix}
x\\
y\\
\end{bmatrix}
=
\begin{bmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}\\
\end{bmatrix}
\begin{bmatrix}
X\\
Y\\
\end{bmatrix}
+
\begin{bmatrix}
b_1\\
b_2\\
\end{bmatrix}
\f]

@param from First input 2D point set containing \f$(X,Y)\f$.
@param to Second input 2D point set containing \f$(x,y)\f$.
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref RANSAC - RANSAC-based robust method
-   @ref LMEDS - Least-Median robust method
RANSAC is the default method.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output 2D affine transformation matrix \f$2 \times 3\f$ or empty matrix if transformation
could not be estimated. The returned matrix has the following form:
\f[
\begin{bmatrix}
a_{11} & a_{12} & b_1\\
a_{21} & a_{22} & b_2\\
\end{bmatrix}
\f]

The function estimates an optimal 2D affine transformation between two 2D point sets using the
selected robust algorithm.

The computed transformation is then refined further (using only inliers) with the
Levenberg-Marquardt method to reduce the re-projection error even more.

@note
The RANSAC method can handle practically any ratio of outliers but needs a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers.

@sa estimateAffinePartial2D, getAffineTransform
*/
CV_EXPORTS_W cv::Mat estimateAffine2D(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = RANSAC, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);


CV_EXPORTS_W cv::Mat estimateAffine2D(InputArray pts1, InputArray pts2, OutputArray inliers,
                     const UsacParams &params);

/** @brief Computes an optimal limited affine transformation with 4 degrees of freedom between
two 2D point sets.

@param from First input 2D point set.
@param to Second input 2D point set.
@param inliers Output vector indicating which points are inliers.
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref RANSAC - RANSAC-based robust method
-   @ref LMEDS - Least-Median robust method
RANSAC is the default method.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output 2D affine transformation (4 degrees of freedom) matrix \f$2 \times 3\f$ or
empty matrix if transformation could not be estimated.

The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
estimation.

The computed transformation is then refined further (using only inliers) with the
Levenberg-Marquardt method to reduce the re-projection error even more.

Estimated transformation matrix is:
\f[ \begin{bmatrix} \cos(\theta) \cdot s & -\sin(\theta) \cdot s & t_x \\
                \sin(\theta) \cdot s & \cos(\theta) \cdot s & t_y
\end{bmatrix} \f]
Where \f$ \theta \f$ is the rotation angle, \f$ s \f$ the scaling factor and \f$ t_x, t_y \f$ are
translations in \f$ x, y \f$ axes respectively.

@note
The RANSAC method can handle practically any ratio of outliers but need a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers.

@sa estimateAffine2D, getAffineTransform
*/
CV_EXPORTS_W cv::Mat estimateAffinePartial2D(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = RANSAC, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);


/** @brief Computes an optimal limited \f$SE(2)\f$ transformation with 3 degrees of freedom between
two 2D point sets.

@param from First input 2D point set containing \f$(X,Y)\f$.
@param to Second input 2D point set containing \f$(x,y)\f$.
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref USAC_DEFAULT – has standard LO-RANSAC.
-   @ref USAC_PARALLEL – has LO-RANSAC and RANSACs run in parallel.
-   @ref USAC_ACCURATE – has GC-RANSAC.
-   @ref USAC_FAST – has LO-RANSAC with smaller number iterations in local optimization step. Uses RANSAC score to maximize number of inliers and terminate earlier.
-   @ref USAC_PROSAC – has PROSAC sampling. Note, points must be sorted.
-   @ref USAC_MAGSAC – has MAGSAC++.
USAC_DEFAULT is the default method.
The legacy methods of @ref RANSAC and @ref LMEDS are not supported. Use @sa estimateAffinePartial2D for them.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output \f$SE(2)\f$ transformation (3 degrees of freedom) matrix \f$2 \times 3\f$ or
empty matrix if transformation could not be estimated.

The function estimates an optimal \f$SE(2)\f$ transformation with 3 degrees of freedom limited to
combinations of translation and rotation.
@sa estimateAffine2D
*/
CV_EXPORTS_W cv::Mat estimateSE2(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = USAC_DEFAULT, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

CV_EXPORTS_W cv::Mat estimateSE2(InputArray pts1, InputArray pts2, OutputArray inliers,
                     const UsacParams &params);


/** @brief Computes an optimal limited \f$Sim(2)\f$ transformation with 4 degrees of freedom between
two 2D point sets.

@param from First input 2D point set containing \f$(X,Y)\f$.
@param to Second input 2D point set containing \f$(x,y)\f$.
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref USAC_DEFAULT – has standard LO-RANSAC.
-   @ref USAC_PARALLEL – has LO-RANSAC and RANSACs run in parallel.
-   @ref USAC_ACCURATE – has GC-RANSAC.
-   @ref USAC_FAST – has LO-RANSAC with smaller number iterations in local optimization step. Uses RANSAC score to maximize number of inliers and terminate earlier.
-   @ref USAC_PROSAC – has PROSAC sampling. Note, points must be sorted.
-   @ref USAC_MAGSAC – has MAGSAC++.
USAC_DEFAULT is the default method.
The legacy methods of @ref RANSAC and @ref LMEDS are not supported. Use @sa estimateAffinePartial2D for them.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output \f$Sim(2)\f$ transformation (4 degrees of freedom) matrix \f$2 \times 3\f$ or
empty matrix if transformation could not be estimated.

The function estimates an optimal \f$Sim(2)\f$ transformation with 4 degrees of freedom limited to
combinations of translation, rotation, and uniform scaling.
@sa estimateAffine2D
*/
CV_EXPORTS_W cv::Mat estimateSIM2(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = USAC_DEFAULT, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

CV_EXPORTS_W cv::Mat estimateSIM2(InputArray pts1, InputArray pts2, OutputArray inliers,
                     const UsacParams &params);

/** @brief Computes an optimal limited \f$SO(3)\f$ transformation with 3 degrees of freedom between
@return Output \f$SO(3)\f$ transformation (3 degrees of freedom) matrix \f$3 \times 4\f$ or
empty matrix if transformation could not be estimated.

@param from First input 3D point set containing \f$(X,Y,Z)\f$.
@param to Second input 3D point set containing \f$(x,y,z)\f$.
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref USAC_DEFAULT – has standard LO-RANSAC.
-   @ref USAC_PARALLEL – has LO-RANSAC and RANSACs run in parallel.
-   @ref USAC_ACCURATE – has GC-RANSAC.
-   @ref USAC_FAST – has LO-RANSAC with smaller number iterations in local optimization step. Uses RANSAC score to maximize number of inliers and terminate earlier.
-   @ref USAC_PROSAC – has PROSAC sampling. Note, points must be sorted.
-   @ref USAC_MAGSAC – has MAGSAC++.
USAC_DEFAULT is the default method.
The legacy methods of @ref RANSAC and @ref LMEDS are not supported. Use @sa estimateAffine3D for them.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output \f$SO(3)\f$ transformation (3 degrees of freedom) matrix \f$3 \times 4\f$ or
empty matrix if transformation could not be estimated.
The function estimates an optimal \f$SO(3)\f$ transformation with 3 degrees of freedom limited to
combinations of translation and rotation.
@sa estimateAffine3D
*/
CV_EXPORTS_W cv::Mat estimateSO3(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = USAC_DEFAULT, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

CV_EXPORTS_W cv::Mat estimateSO3(InputArray pts1, InputArray pts2, OutputArray inliers,
                     const UsacParams &params);

/** @brief Computes an optimal limited \f$SE(3)\f$ transformation with 6 degrees of freedom between
two 3D point sets.

@param from First input 3D point set containing \f$(X,Y,Z)\f$.
@param to Second input 3D point set containing \f$(x,y,z)\f$.
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref USAC_DEFAULT – has standard LO-RANSAC.
-   @ref USAC_PARALLEL – has LO-RANSAC and RANSACs run in parallel.
-   @ref USAC_ACCURATE – has GC-RANSAC.
-   @ref USAC_FAST – has LO-RANSAC with smaller number iterations in local optimization step. Uses RANSAC score to maximize number of inliers and terminate earlier.
-   @ref USAC_PROSAC – has PROSAC sampling. Note, points must be sorted.
-   @ref USAC_MAGSAC – has MAGSAC++.
USAC_DEFAULT is the default method.
The legacy methods of @ref RANSAC and @ref LMEDS are not supported. Use @sa estimateAffine3D for them.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output \f$SE(3)\f$ transformation (6 degrees of freedom) matrix \f$3 \times 4\f$ or
empty matrix if transformation could not be estimated.
The function estimates an optimal \f$SE(3)\f$ transformation with 6 degrees of freedom limited to
combinations of translation and rotation.
@sa estimateAffine3D
*/
CV_EXPORTS_W cv::Mat estimateSE3(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = USAC_DEFAULT, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

CV_EXPORTS_W cv::Mat estimateSE3(InputArray pts1, InputArray pts2, OutputArray inliers,
                     const UsacParams &params);

/** @brief Computes an optimal limited \f$Sim(3)\f$ transformation with 7 degrees of freedom between
two 3D point sets.

@param from First input 3D point set containing \f$(X,Y,Z)\f$.
@param to Second input 3D point set containing \f$(x,y,z)\f$.
@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).
@param method Robust method used to compute transformation. The following methods are possible:
-   @ref USAC_DEFAULT – has standard LO-RANSAC.
-   @ref USAC_PARALLEL – has LO-RANSAC and RANSACs run in parallel.
-   @ref USAC_ACCURATE – has GC-RANSAC.
-   @ref USAC_FAST – has LO-RANSAC with smaller number iterations in local optimization step. Uses RANSAC score to maximize number of inliers and terminate earlier.
-   @ref USAC_PROSAC – has PROSAC sampling. Note, points must be sorted.
-   @ref USAC_MAGSAC – has MAGSAC++.
USAC_DEFAULT is the default method.
The legacy methods of @ref RANSAC and @ref LMEDS are not supported. Use @sa estimateAffine3D for them.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output \f$Sim(3)\f$ transformation (7 degrees of freedom) matrix \f$3 \times 4\f$ or
empty matrix if transformation could not be estimated.

The function estimates an optimal \f$Sim(3)\f$ transformation with 7 degrees of freedom limited to
combinations of translation, rotation, and uniform scaling.
@sa estimateAffine3D
*/CV_EXPORTS_W cv::Mat estimateSIM3(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = USAC_DEFAULT, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

CV_EXPORTS_W cv::Mat estimateSIM3(InputArray pts1, InputArray pts2, OutputArray inliers,
                     const UsacParams &params);

/** @example samples/cpp/tutorial_code/features2D/Homography/decompose_homography.cpp
An example program with homography decomposition.

Check @ref tutorial_homography "the corresponding tutorial" for more details.
*/

/** @brief Decompose a homography matrix to rotation(s), translation(s) and plane normal(s).

@param H The input homography matrix between two images.
@param K The input camera intrinsic matrix.
@param rotations Array of rotation matrices.
@param translations Array of translation matrices.
@param normals Array of plane normal matrices.

This function extracts relative camera motion between two views of a planar object and returns up to
four mathematical solution tuples of rotation, translation, and plane normal. The decomposition of
the homography matrix H is described in detail in @cite Malis2007.

If the homography H, induced by the plane, gives the constraint
\f[s_i \vecthree{x'_i}{y'_i}{1} \sim H \vecthree{x_i}{y_i}{1}\f] on the source image points
\f$p_i\f$ and the destination image points \f$p'_i\f$, then the tuple of rotations[k] and
translations[k] is a change of basis from the source camera's coordinate system to the destination
camera's coordinate system. However, by decomposing H, one can only get the translation normalized
by the (typically unknown) depth of the scene, i.e. its direction but with normalized length.

If point correspondences are available, at least two solutions may further be invalidated, by
applying positive depth constraint, i.e. all points must be in front of the camera.
 */
CV_EXPORTS_W int decomposeHomographyMat(InputArray H,
                                        InputArray K,
                                        OutputArrayOfArrays rotations,
                                        OutputArrayOfArrays translations,
                                        OutputArrayOfArrays normals);

/** @brief Filters homography decompositions based on additional information.

@param rotations Vector of rotation matrices.
@param normals Vector of plane normal matrices.
@param beforePoints Vector of (rectified) visible reference points before the homography is applied
@param afterPoints Vector of (rectified) visible reference points after the homography is applied
@param possibleSolutions Vector of int indices representing the viable solution set after filtering
@param pointsMask optional Mat/Vector of 8u type representing the mask for the inliers as given by the #findHomography function

This function is intended to filter the output of the #decomposeHomographyMat based on additional
information as described in @cite Malis2007 . The summary of the method: the #decomposeHomographyMat function
returns 2 unique solutions and their "opposites" for a total of 4 solutions. If we have access to the
sets of points visible in the camera frame before and after the homography transformation is applied,
we can determine which are the true potential solutions and which are the opposites by verifying which
homographies are consistent with all visible reference points being in front of the camera. The inputs
are left unchanged; the filtered solution set is returned as indices into the existing one.

*/
CV_EXPORTS_W void filterHomographyDecompByVisibleRefpoints(InputArrayOfArrays rotations,
                                                           InputArrayOfArrays normals,
                                                           InputArray beforePoints,
                                                           InputArray afterPoints,
                                                           OutputArray possibleSolutions,
                                                           InputArray pointsMask = noArray());

/** @brief The base class for stereo correspondence algorithms.
 */
class CV_EXPORTS_W StereoMatcher : public Algorithm
{
public:
    enum { DISP_SHIFT = 4,
           DISP_SCALE = (1 << DISP_SHIFT)
         };

    /** @brief Computes disparity map for the specified stereo pair

    @param left Left 8-bit single-channel image.
    @param right Right image of the same size and the same type as the left one.
    @param disparity Output disparity map. It has the same size as the input images. Some algorithms,
    like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value
    has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map.
     */
    CV_WRAP virtual void compute( InputArray left, InputArray right,
                                  OutputArray disparity ) = 0;

    CV_WRAP virtual int getMinDisparity() const = 0;
    CV_WRAP virtual void setMinDisparity(int minDisparity) = 0;

    CV_WRAP virtual int getNumDisparities() const = 0;
    CV_WRAP virtual void setNumDisparities(int numDisparities) = 0;

    CV_WRAP virtual int getBlockSize() const = 0;
    CV_WRAP virtual void setBlockSize(int blockSize) = 0;

    CV_WRAP virtual int getSpeckleWindowSize() const = 0;
    CV_WRAP virtual void setSpeckleWindowSize(int speckleWindowSize) = 0;

    CV_WRAP virtual int getSpeckleRange() const = 0;
    CV_WRAP virtual void setSpeckleRange(int speckleRange) = 0;

    CV_WRAP virtual int getDisp12MaxDiff() const = 0;
    CV_WRAP virtual void setDisp12MaxDiff(int disp12MaxDiff) = 0;
};


/** @brief Class for computing stereo correspondence using the block matching algorithm, introduced and
contributed to OpenCV by K. Konolige.
 */
class CV_EXPORTS_W StereoBM : public StereoMatcher
{
public:
    enum { PREFILTER_NORMALIZED_RESPONSE = 0,
           PREFILTER_XSOBEL              = 1
         };

    CV_WRAP virtual int getPreFilterType() const = 0;
    CV_WRAP virtual void setPreFilterType(int preFilterType) = 0;

    CV_WRAP virtual int getPreFilterSize() const = 0;
    CV_WRAP virtual void setPreFilterSize(int preFilterSize) = 0;

    CV_WRAP virtual int getPreFilterCap() const = 0;
    CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;

    CV_WRAP virtual int getTextureThreshold() const = 0;
    CV_WRAP virtual void setTextureThreshold(int textureThreshold) = 0;

    CV_WRAP virtual int getUniquenessRatio() const = 0;
    CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;

    CV_WRAP virtual int getSmallerBlockSize() const = 0;
    CV_WRAP virtual void setSmallerBlockSize(int blockSize) = 0;

    CV_WRAP virtual Rect getROI1() const = 0;
    CV_WRAP virtual void setROI1(Rect roi1) = 0;

    CV_WRAP virtual Rect getROI2() const = 0;
    CV_WRAP virtual void setROI2(Rect roi2) = 0;

    /** @brief Creates StereoBM object

    @param numDisparities the disparity search range. For each pixel algorithm will find the best
    disparity from 0 (default minimum disparity) to numDisparities. The search range can then be
    shifted by changing the minimum disparity.
    @param blockSize the linear size of the blocks compared by the algorithm. The size should be odd
    (as the block is centered at the current pixel). Larger block size implies smoother, though less
    accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher
    chance for algorithm to find a wrong correspondence.

    The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for
    a specific stereo pair.
     */
    CV_WRAP static Ptr<StereoBM> create(int numDisparities = 0, int blockSize = 21);
};

/** @brief The class implements the modified H. Hirschmuller algorithm @cite HH08 that differs from the original
one as follows:

-   By default, the algorithm is single-pass, which means that you consider only 5 directions
instead of 8. Set mode=StereoSGBM::MODE_HH in createStereoSGBM to run the full variant of the
algorithm but beware that it may consume a lot of memory.
-   The algorithm matches blocks, not individual pixels. Though, setting blockSize=1 reduces the
blocks to single pixels.
-   Mutual information cost function is not implemented. Instead, a simpler Birchfield-Tomasi
sub-pixel metric from @cite BT98 is used. Though, the color images are supported as well.
-   Some pre- and post- processing steps from K. Konolige algorithm StereoBM are included, for
example: pre-filtering (StereoBM::PREFILTER_XSOBEL type) and post-filtering (uniqueness
check, quadratic interpolation and speckle filtering).

@note
   -   (Python) An example illustrating the use of the StereoSGBM matching algorithm can be found
        at opencv_source_code/samples/python/stereo_match.py
 */
class CV_EXPORTS_W StereoSGBM : public StereoMatcher
{
public:
    enum
    {
        MODE_SGBM = 0,
        MODE_HH   = 1,
        MODE_SGBM_3WAY = 2,
        MODE_HH4  = 3
    };

    CV_WRAP virtual int getPreFilterCap() const = 0;
    CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;

    CV_WRAP virtual int getUniquenessRatio() const = 0;
    CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;

    CV_WRAP virtual int getP1() const = 0;
    CV_WRAP virtual void setP1(int P1) = 0;

    CV_WRAP virtual int getP2() const = 0;
    CV_WRAP virtual void setP2(int P2) = 0;

    CV_WRAP virtual int getMode() const = 0;
    CV_WRAP virtual void setMode(int mode) = 0;

    /** @brief Creates StereoSGBM object

    @param minDisparity Minimum possible disparity value. Normally, it is zero but sometimes
    rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    @param numDisparities Maximum disparity minus minimum disparity. The value is always greater than
    zero. In the current implementation, this parameter must be divisible by 16.
    @param blockSize Matched block size. It must be an odd number \>=1 . Normally, it should be
    somewhere in the 3..11 range.
    @param P1 The first parameter controlling the disparity smoothness. See below.
    @param P2 The second parameter controlling the disparity smoothness. The larger the values are,
    the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1
    between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor
    pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good
    P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and
    32\*number_of_image_channels\*blockSize\*blockSize , respectively).
    @param disp12MaxDiff Maximum allowed difference (in integer pixel units) in the left-right
    disparity check. Set it to a non-positive value to disable the check.
    @param preFilterCap Truncation value for the prefiltered image pixels. The algorithm first
    computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.
    The result values are passed to the Birchfield-Tomasi pixel cost function.
    @param uniquenessRatio Margin in percentage by which the best (minimum) computed cost function
    value should "win" the second best value to consider the found match correct. Normally, a value
    within the 5-15 range is good enough.
    @param speckleWindowSize Maximum size of smooth disparity regions to consider their noise speckles
    and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the
    50-200 range.
    @param speckleRange Maximum disparity variation within each connected component. If you do speckle
    filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    Normally, 1 or 2 is good enough.
    @param mode Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming
    algorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and
    huge for HD-size pictures. By default, it is set to false .

    The first constructor initializes StereoSGBM with all the default parameters. So, you only have to
    set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter
    to a custom value.
     */
    CV_WRAP static Ptr<StereoSGBM> create(int minDisparity = 0, int numDisparities = 16, int blockSize = 3,
                                          int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
                                          int preFilterCap = 0, int uniquenessRatio = 0,
                                          int speckleWindowSize = 0, int speckleRange = 0,
                                          int mode = StereoSGBM::MODE_SGBM);
};


//! cv::undistort mode
enum UndistortTypes
{
    PROJ_SPHERICAL_ORTHO  = 0,
    PROJ_SPHERICAL_EQRECT = 1
};

/** @brief Transforms an image to compensate for lens distortion.

The function transforms an image to compensate radial and tangential lens distortion.

The function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap
(with bilinear interpolation). See the former function for details of the transformation being
performed.

Those pixels in the destination image, for which there is no correspondent pixels in the source
image, are filled with zeros (black color).

A particular subset of the source image that will be visible in the corrected image can be regulated
by newCameraMatrix. You can use #getOptimalNewCameraMatrix to compute the appropriate
newCameraMatrix depending on your requirements.

The camera matrix and the distortion parameters can be determined using #calibrateCamera. If
the resolution of images is different from the resolution used at the calibration stage, \f$f_x,
f_y, c_x\f$ and \f$c_y\f$ need to be scaled accordingly, while the distortion coefficients remain
the same.

@param src Input (distorted) image.
@param dst Output (corrected) image that has the same size and type as src .
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param newCameraMatrix Camera matrix of the distorted image. By default, it is the same as
cameraMatrix but you may additionally scale and shift the result by using a different matrix.
 */
CV_EXPORTS_W void undistort( InputArray src, OutputArray dst,
                             InputArray cameraMatrix,
                             InputArray distCoeffs,
                             InputArray newCameraMatrix = noArray() );

/** @brief Computes the undistortion and rectification transformation map.

The function computes the joint undistortion and rectification transformation and represents the
result in the form of maps for #remap. The undistorted image looks like original, as if it is
captured with a camera using the camera matrix =newCameraMatrix and zero distortion. In case of a
monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by
#getOptimalNewCameraMatrix for a better control over scaling. In case of a stereo camera,
newCameraMatrix is normally set to P1 or P2 computed by #stereoRectify .

Also, this new camera is oriented differently in the coordinate space, according to R. That, for
example, helps to align two heads of a stereo camera so that the epipolar lines on both images
become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).

The function actually builds the maps for the inverse mapping algorithm that is used by #remap. That
is, for each pixel \f$(u, v)\f$ in the destination (corrected and rectified) image, the function
computes the corresponding coordinates in the source image (that is, in the original image from
camera). The following process is applied:
\f[
\begin{array}{l}
x  \leftarrow (u - {c'}_x)/{f'}_x  \\
y  \leftarrow (v - {c'}_y)/{f'}_y  \\
{[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\
x'  \leftarrow X/W  \\
y'  \leftarrow Y/W  \\
r^2  \leftarrow x'^2 + y'^2 \\
x''  \leftarrow x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
+ 2p_1 x' y' + p_2(r^2 + 2 x'^2)  + s_1 r^2 + s_2 r^4\\
y''  \leftarrow y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
+ p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
s\vecthree{x'''}{y'''}{1} =
\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}((\tau_x, \tau_y)}
{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\
map_x(u,v)  \leftarrow x''' f_x + c_x  \\
map_y(u,v)  \leftarrow y''' f_y + c_y
\end{array}
\f]
where \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
are the distortion coefficients.

In case of a stereo camera, this function is called twice: once for each camera head, after
#stereoRectify, which in its turn is called after #stereoCalibrate. But if the stereo camera
was not calibrated, it is still possible to compute the rectification transformations directly from
the fundamental matrix using #stereoRectifyUncalibrated. For each camera, the function computes
homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D
space. R can be computed from H as
\f[\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}\f]
where cameraMatrix can be chosen arbitrarily.

@param cameraMatrix Input camera matrix \f$A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param R Optional rectification transformation in the object space (3x3 matrix). R1 or R2 ,
computed by #stereoRectify can be passed here. If the matrix is empty, the identity transformation
is assumed. In #initUndistortRectifyMap R assumed to be an identity matrix.
@param newCameraMatrix New camera matrix \f$A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}\f$.
@param size Undistorted image size.
@param m1type Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see #convertMaps
@param map1 The first output map.
@param map2 The second output map.
 */
CV_EXPORTS_W
void initUndistortRectifyMap(InputArray cameraMatrix, InputArray distCoeffs,
                             InputArray R, InputArray newCameraMatrix,
                             Size size, int m1type, OutputArray map1, OutputArray map2);

/** @brief Computes the projection and inverse-rectification transformation map. In essense, this is the inverse of
#initUndistortRectifyMap to accomodate stereo-rectification of projectors ('inverse-cameras') in projector-camera pairs.

The function computes the joint projection and inverse rectification transformation and represents the
result in the form of maps for #remap. The projected image looks like a distorted version of the original which,
once projected by a projector, should visually match the original. In case of a monocular camera, newCameraMatrix
is usually equal to cameraMatrix, or it can be computed by
#getOptimalNewCameraMatrix for a better control over scaling. In case of a projector-camera pair,
newCameraMatrix is normally set to P1 or P2 computed by #stereoRectify .

The projector is oriented differently in the coordinate space, according to R. In case of projector-camera pairs,
this helps align the projector (in the same manner as #initUndistortRectifyMap for the camera) to create a stereo-rectified pair. This
allows epipolar lines on both images to become horizontal and have the same y-coordinate (in case of a horizontally aligned projector-camera pair).

The function builds the maps for the inverse mapping algorithm that is used by #remap. That
is, for each pixel \f$(u, v)\f$ in the destination (projected and inverse-rectified) image, the function
computes the corresponding coordinates in the source image (that is, in the original digital image). The following process is applied:

\f[
\begin{array}{l}
\text{newCameraMatrix}\\
x  \leftarrow (u - {c'}_x)/{f'}_x  \\
y  \leftarrow (v - {c'}_y)/{f'}_y  \\

\\\text{Undistortion}
\\\scriptsize{\textit{though equation shown is for radial undistortion, function implements cv::undistortPoints()}}\\
r^2  \leftarrow x^2 + y^2 \\
\theta \leftarrow \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}\\
x' \leftarrow \frac{x}{\theta} \\
y'  \leftarrow \frac{y}{\theta} \\

\\\text{Rectification}\\
{[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\
x''  \leftarrow X/W  \\
y''  \leftarrow Y/W  \\

\\\text{cameraMatrix}\\
map_x(u,v)  \leftarrow x'' f_x + c_x  \\
map_y(u,v)  \leftarrow y'' f_y + c_y
\end{array}
\f]
where \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
are the distortion coefficients vector distCoeffs.

In case of a stereo-rectified projector-camera pair, this function is called for the projector while #initUndistortRectifyMap is called for the camera head.
This is done after #stereoRectify, which in turn is called after #stereoCalibrate. If the projector-camera pair
is not calibrated, it is still possible to compute the rectification transformations directly from
the fundamental matrix using #stereoRectifyUncalibrated. For the projector and camera, the function computes
homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D
space. R can be computed from H as
\f[\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}\f]
where cameraMatrix can be chosen arbitrarily.

@param cameraMatrix Input camera matrix \f$A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param R Optional rectification transformation in the object space (3x3 matrix). R1 or R2,
computed by #stereoRectify can be passed here. If the matrix is empty, the identity transformation
is assumed.
@param newCameraMatrix New camera matrix \f$A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}\f$.
@param size Distorted image size.
@param m1type Type of the first output map. Can be CV_32FC1, CV_32FC2 or CV_16SC2, see #convertMaps
@param map1 The first output map for #remap.
@param map2 The second output map for #remap.
 */
CV_EXPORTS_W
void initInverseRectificationMap( InputArray cameraMatrix, InputArray distCoeffs,
                           InputArray R, InputArray newCameraMatrix,
                           const Size& size, int m1type, OutputArray map1, OutputArray map2 );

//! initializes maps for #remap for wide-angle
CV_EXPORTS
float initWideAngleProjMap(InputArray cameraMatrix, InputArray distCoeffs,
                           Size imageSize, int destImageWidth,
                           int m1type, OutputArray map1, OutputArray map2,
                           enum UndistortTypes projType = PROJ_SPHERICAL_EQRECT, double alpha = 0);
static inline
float initWideAngleProjMap(InputArray cameraMatrix, InputArray distCoeffs,
                           Size imageSize, int destImageWidth,
                           int m1type, OutputArray map1, OutputArray map2,
                           int projType, double alpha = 0)
{
    return initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize, destImageWidth,
                                m1type, map1, map2, (UndistortTypes)projType, alpha);
}

/** @brief Returns the default new camera matrix.

The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when
centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true).

In the latter case, the new camera matrix will be:

\f[\begin{bmatrix} f_x && 0 && ( \texttt{imgSize.width} -1)*0.5  \\ 0 && f_y && ( \texttt{imgSize.height} -1)*0.5  \\ 0 && 0 && 1 \end{bmatrix} ,\f]

where \f$f_x\f$ and \f$f_y\f$ are \f$(0,0)\f$ and \f$(1,1)\f$ elements of cameraMatrix, respectively.

By default, the undistortion functions in OpenCV (see #initUndistortRectifyMap, #undistort) do not
move the principal point. However, when you work with stereo, it is important to move the principal
points in both views to the same y-coordinate (which is required by most of stereo correspondence
algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for
each view where the principal points are located at the center.

@param cameraMatrix Input camera matrix.
@param imgsize Camera view image size in pixels.
@param centerPrincipalPoint Location of the principal point in the new camera matrix. The
parameter indicates whether this location should be at the image center or not.
 */
CV_EXPORTS_W
Mat getDefaultNewCameraMatrix(InputArray cameraMatrix, Size imgsize = Size(),
                              bool centerPrincipalPoint = false);

/** @brief Computes the ideal point coordinates from the observed point coordinates.

The function is similar to #undistort and #initUndistortRectifyMap but it operates on a
sparse set of points instead of a raster image. Also the function performs a reverse transformation
to  #projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a
planar object, it does, up to a translation vector, if the proper R is specified.

For each observed point coordinate \f$(u, v)\f$ the function computes:
\f[
\begin{array}{l}
x^{"}  \leftarrow (u - c_x)/f_x  \\
y^{"}  \leftarrow (v - c_y)/f_y  \\
(x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\
{[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\
x  \leftarrow X/W  \\
y  \leftarrow Y/W  \\
\text{only performed if P is specified:} \\
u'  \leftarrow x {f'}_x + {c'}_x  \\
v'  \leftarrow y {f'}_y + {c'}_y
\end{array}
\f]

where *undistort* is an approximate iterative algorithm that estimates the normalized original
point coordinates out of the normalized distorted point coordinates ("normalized" means that the
coordinates do not depend on the camera matrix).

The function can be used for both a stereo camera head or a monocular camera (when R is empty).
@param src Observed point coordinates, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or
vector\<Point2f\> ).
@param dst Output ideal point coordinates (1xN/Nx1 2-channel or vector\<Point2f\> ) after undistortion and reverse perspective
transformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.
@param cameraMatrix Camera matrix \f$\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by
#stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.
@param P New camera matrix (3x3) or new projection matrix (3x4) \f$\begin{bmatrix} {f'}_x & 0 & {c'}_x & t_x \\ 0 & {f'}_y & {c'}_y & t_y \\ 0 & 0 & 1 & t_z \end{bmatrix}\f$. P1 or P2 computed by
#stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.
 */
CV_EXPORTS_W
void undistortPoints(InputArray src, OutputArray dst,
                     InputArray cameraMatrix, InputArray distCoeffs,
                     InputArray R = noArray(), InputArray P = noArray());
/** @overload
    @note Default version of #undistortPoints does 5 iterations to compute undistorted points.
 */
CV_EXPORTS_AS(undistortPointsIter)
void undistortPoints(InputArray src, OutputArray dst,
                     InputArray cameraMatrix, InputArray distCoeffs,
                     InputArray R, InputArray P, TermCriteria criteria);

/**
 * @brief Compute undistorted image points position
 *
 * @param src Observed points position, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or
CV_64FC2) (or vector\<Point2f\> ).
 * @param dst Output undistorted points position (1xN/Nx1 2-channel or vector\<Point2f\> ).
 * @param cameraMatrix Camera matrix \f$\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
 * @param distCoeffs Distortion coefficients
 */
CV_EXPORTS_W
void undistortImagePoints(InputArray src, OutputArray dst, InputArray cameraMatrix,
                          InputArray distCoeffs,
                          TermCriteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5,
                                                      0.01));

//! @} calib3d

/** @brief The methods in this namespace use a so-called fisheye camera model.
  @ingroup calib3d_fisheye
*/
namespace fisheye
{
//! @addtogroup calib3d_fisheye
//! @{

    enum{
        CALIB_USE_INTRINSIC_GUESS   = 1 << 0,
        CALIB_RECOMPUTE_EXTRINSIC   = 1 << 1,
        CALIB_CHECK_COND            = 1 << 2,
        CALIB_FIX_SKEW              = 1 << 3,
        CALIB_FIX_K1                = 1 << 4,
        CALIB_FIX_K2                = 1 << 5,
        CALIB_FIX_K3                = 1 << 6,
        CALIB_FIX_K4                = 1 << 7,
        CALIB_FIX_INTRINSIC         = 1 << 8,
        CALIB_FIX_PRINCIPAL_POINT   = 1 << 9,
        CALIB_ZERO_DISPARITY        = 1 << 10,
        CALIB_FIX_FOCAL_LENGTH      = 1 << 11
    };

    /** @brief Projects points using fisheye model

    @param objectPoints Array of object points, 1xN/Nx1 3-channel (or vector\<Point3f\> ), where N is
    the number of points in the view.
    @param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or
    vector\<Point2f\>.
    @param affine
    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param alpha The skew coefficient.
    @param jacobian Optional output 2Nx15 jacobian matrix of derivatives of image points with respect
    to components of the focal lengths, coordinates of the principal point, distortion coefficients,
    rotation vector, translation vector, and the skew. In the old interface different components of
    the jacobian are returned via different output parameters.

    The function computes projections of 3D points to the image plane given intrinsic and extrinsic
    camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
    image points coordinates (as functions of all the input parameters) with respect to the particular
    parameters, intrinsic and/or extrinsic.
     */
    CV_EXPORTS void projectPoints(InputArray objectPoints, OutputArray imagePoints, const Affine3d& affine,
        InputArray K, InputArray D, double alpha = 0, OutputArray jacobian = noArray());

    /** @overload */
    CV_EXPORTS_W void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec,
        InputArray K, InputArray D, double alpha = 0, OutputArray jacobian = noArray());

    /** @brief Distorts 2D points using fisheye model.

    @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is
    the number of points in the view.
    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param alpha The skew coefficient.
    @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .

    Note that the function assumes the camera intrinsic matrix of the undistorted points to be identity.
    This means if you want to distort image points you have to multiply them with \f$K^{-1}\f$ or
    use another function overload.
     */
    CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double alpha = 0);

    /** @overload
    Overload of distortPoints function to handle cases when undistorted points are obtained with non-identity
    camera matrix, e.g. output of #estimateNewCameraMatrixForUndistortRectify.
    @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is
    the number of points in the view.
    @param Kundistorted Camera intrinsic matrix used as new camera matrix for undistortion.
    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param alpha The skew coefficient.
    @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
    @sa estimateNewCameraMatrixForUndistortRectify
    */
    CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray Kundistorted, InputArray K, InputArray D, double alpha = 0);

    /** @brief Undistorts 2D points using fisheye model

    @param distorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is the
    number of points in the view.
    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    @param criteria Termination criteria
    @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
     */
    CV_EXPORTS_W void undistortPoints(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray R = noArray(), InputArray P  = noArray(),
                TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1e-8));

    /** @brief Computes undistortion and rectification maps for image transform by #remap. If D is empty zero
    distortion is used, if R or P is empty identity matrixes are used.

    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See #convertMaps
    for details.
    @param map1 The first output map.
    @param map2 The second output map.
     */
    CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, InputArray R, InputArray P,
        const cv::Size& size, int m1type, OutputArray map1, OutputArray map2);

    /** @brief Transforms an image to compensate for fisheye lens distortion.

    @param distorted image with fisheye lens distortion.
    @param undistorted Output image with compensated fisheye lens distortion.
    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param Knew Camera intrinsic matrix of the distorted image. By default, it is the identity matrix but you
    may additionally scale and shift the result by using a different matrix.
    @param new_size the new size

    The function transforms an image to compensate radial and tangential lens distortion.

    The function is simply a combination of #fisheye::initUndistortRectifyMap (with unity R ) and #remap
    (with bilinear interpolation). See the former function for details of the transformation being
    performed.

    See below the results of undistortImage.
       -   a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
            k_4, k_5, k_6) of distortion were optimized under calibration)
        -   b\) result of #fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
            k_3, k_4) of fisheye distortion were optimized under calibration)
        -   c\) original image was captured with fisheye lens

    Pictures a) and b) almost the same. But if we consider points of image located far from the center
    of image, we can notice that on image a) these points are distorted.

    ![image](pics/fisheye_undistorted.jpg)
     */
    CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray Knew = cv::noArray(), const Size& new_size = Size());

    /** @brief Estimates new camera intrinsic matrix for undistortion or rectification.

    @param K Camera intrinsic matrix \f$\cameramatrix{K}\f$.
    @param image_size Size of the image
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    @param balance Sets the new focal length in range between the min focal length and the max focal
    length. Balance is in range of [0, 1].
    @param new_size the new size
    @param fov_scale Divisor for new focal length.
     */
    CV_EXPORTS_W void estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, const Size &image_size, InputArray R,
        OutputArray P, double balance = 0.0, const Size& new_size = Size(), double fov_scale = 1.0);

    /** @brief Performs camera calibration

    @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
    coordinate space.
    @param imagePoints vector of vectors of the projections of calibration pattern points.
    imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
    objectPoints[i].size() for each i.
    @param image_size Size of the image used only to initialize the camera intrinsic matrix.
    @param K Output 3x3 floating-point camera intrinsic matrix
    \f$\cameramatrix{A}\f$ . If
    @ref fisheye::CALIB_USE_INTRINSIC_GUESS is specified, some or all of fx, fy, cx, cy must be
    initialized before calling the function.
    @param D Output vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param rvecs Output vector of rotation vectors (see @ref Rodrigues ) estimated for each pattern view.
    That is, each k-th rotation vector together with the corresponding k-th translation vector (see
    the next output parameter description) brings the calibration pattern from the model coordinate
    space (in which object points are specified) to the world coordinate space, that is, a real
    position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
    @param tvecs Output vector of translation vectors estimated for each pattern view.
    @param flags Different flags that may be zero or a combination of the following values:
    -    @ref fisheye::CALIB_USE_INTRINSIC_GUESS  cameraMatrix contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center ( imageSize is used), and focal distances are computed in a least-squares fashion.
    -    @ref fisheye::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -    @ref fisheye::CALIB_CHECK_COND  The functions will check validity of condition number.
    -    @ref fisheye::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
    -    @ref fisheye::CALIB_FIX_K1,..., @ref fisheye::CALIB_FIX_K4 Selected distortion coefficients
    are set to zeros and stay zero.
    -    @ref fisheye::CALIB_FIX_PRINCIPAL_POINT  The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when @ref fisheye::CALIB_USE_INTRINSIC_GUESS is set too.
    -    @ref fisheye::CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global
optimization. It is the \f$max(width,height)/\pi\f$ or the provided \f$f_x\f$, \f$f_y\f$ when @ref fisheye::CALIB_USE_INTRINSIC_GUESS is set too.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, const Size& image_size,
        InputOutputArray K, InputOutputArray D, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0,
            TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

    /** @brief Stereo rectification for fisheye camera model

    @param K1 First camera intrinsic matrix.
    @param D1 First camera distortion parameters.
    @param K2 Second camera intrinsic matrix.
    @param D2 Second camera distortion parameters.
    @param imageSize Size of the image used for stereo calibration.
    @param R Rotation matrix between the coordinate systems of the first and the second
    cameras.
    @param tvec Translation vector between coordinate systems of the cameras.
    @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
    camera.
    @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
    camera.
    @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see #reprojectImageTo3D ).
    @param flags Operation flags that may be zero or @ref fisheye::CALIB_ZERO_DISPARITY . If the flag is set,
    the function makes the principal points of each camera have the same pixel coordinates in the
    rectified views. And if the flag is not set, the function may still shift the images in the
    horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
    useful image area.
    @param newImageSize New image resolution after rectification. The same size should be passed to
    #initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
    is passed (default), it is set to the original imageSize . Setting it to larger value can help you
    preserve details in the original image, especially when there is a big radial distortion.
    @param balance Sets the new focal length in range between the min focal length and the max focal
    length. Balance is in range of [0, 1].
    @param fov_scale Divisor for new focal length.
     */
    CV_EXPORTS_W void stereoRectify(InputArray K1, InputArray D1, InputArray K2, InputArray D2, const Size &imageSize, InputArray R, InputArray tvec,
        OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags, const Size &newImageSize = Size(),
        double balance = 0.0, double fov_scale = 1.0);

    /** @brief Performs stereo calibration

    @param objectPoints Vector of vectors of the calibration pattern points.
    @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
    observed by the first camera.
    @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
    observed by the second camera.
    @param K1 Input/output first camera intrinsic matrix:
    \f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
    any of @ref fisheye::CALIB_USE_INTRINSIC_GUESS , @ref fisheye::CALIB_FIX_INTRINSIC are specified,
    some or all of the matrix components must be initialized.
    @param D1 Input/output vector of distortion coefficients \f$\distcoeffsfisheye\f$ of 4 elements.
    @param K2 Input/output second camera intrinsic matrix. The parameter is similar to K1 .
    @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
    similar to D1 .
    @param imageSize Size of the image used only to initialize camera intrinsic matrix.
    @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
    @param T Output translation vector between the coordinate systems of the cameras.
    @param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
    coordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each
    i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter
    description) brings the calibration pattern from the object coordinate space (in which object points are
    specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms,
    the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space
    to camera coordinate space of the first camera of the stereo pair.
    @param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
    of previous output parameter ( rvecs ).
    @param flags Different flags that may be zero or a combination of the following values:
    -    @ref fisheye::CALIB_FIX_INTRINSIC  Fix K1, K2? and D1, D2? so that only R, T matrices
    are estimated.
    -    @ref fisheye::CALIB_USE_INTRINSIC_GUESS  K1, K2 contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center (imageSize is used), and focal distances are computed in a least-squares fashion.
    -    @ref fisheye::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -    @ref fisheye::CALIB_CHECK_COND  The functions will check validity of condition number.
    -    @ref fisheye::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
    -   @ref fisheye::CALIB_FIX_K1,..., @ref fisheye::CALIB_FIX_K4 Selected distortion coefficients are set to zeros and stay
    zero.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                  InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                  OutputArray R, OutputArray T, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = fisheye::CALIB_FIX_INTRINSIC,
                                  TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

    /// @overload
    CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                  InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                  OutputArray R, OutputArray T, int flags = fisheye::CALIB_FIX_INTRINSIC,
                                  TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

    /**
    @brief Finds an object pose from 3D-2D point correspondences for fisheye camera moodel.

    @param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
    1xN/Nx1 3-channel, where N is the number of points. vector\<Point3d\> can be also passed here.
    @param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
    where N is the number of points. vector\<Point2d\> can be also passed here.
    @param cameraMatrix Input camera intrinsic matrix \f$\cameramatrix{A}\f$ .
    @param distCoeffs Input vector of distortion coefficients (4x1/1x4).
    @param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec, brings points from
    the model coordinate system to the camera coordinate system.
    @param tvec Output translation vector.
    @param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
    the provided rvec and tvec values as initial approximations of the rotation and translation
    vectors, respectively, and further optimizes them.
    @param flags Method for solving a PnP problem: see @ref calib3d_solvePnP_flags
    This function returns the rotation and the translation vectors that transform a 3D point expressed in the object
    coordinate frame to the camera coordinate frame, using different methods:
    - P3P methods (@ref SOLVEPNP_P3P, @ref SOLVEPNP_AP3P): need 4 input points to return a unique solution.
    - @ref SOLVEPNP_IPPE Input points must be >= 4 and object points must be coplanar.
    - @ref SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation.
    Number of input points must be 4. Object points must be defined in the following order:
    - point 0: [-squareLength / 2,  squareLength / 2, 0]
    - point 1: [ squareLength / 2,  squareLength / 2, 0]
    - point 2: [ squareLength / 2, -squareLength / 2, 0]
    - point 3: [-squareLength / 2, -squareLength / 2, 0]
    - for all the other flags, number of input points must be >= 4 and object points can be in any configuration.
    @param criteria Termination criteria for internal undistortPoints call.
    The function interally undistorts points with @ref undistortPoints and call @ref cv::solvePnP,
    thus the input are very similar. More information about Perspective-n-Points is described in @ref calib3d_solvePnP
    for more information.
    */
    CV_EXPORTS_W bool solvePnP( InputArray objectPoints, InputArray imagePoints,
                                InputArray cameraMatrix, InputArray distCoeffs,
                                OutputArray rvec, OutputArray tvec,
                                bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE,
                                TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1e-8)
                              );

//! @} calib3d_fisheye
} // end namespace fisheye

} //end namespace cv

#if 0 //def __cplusplus
//////////////////////////////////////////////////////////////////////////////////////////
class CV_EXPORTS CvLevMarq
{
public:
    CvLevMarq();
    CvLevMarq( int nparams, int nerrs, CvTermCriteria criteria=
              cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    ~CvLevMarq();
    void init( int nparams, int nerrs, CvTermCriteria criteria=
              cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    bool update( const CvMat*& param, CvMat*& J, CvMat*& err );
    bool updateAlt( const CvMat*& param, CvMat*& JtJ, CvMat*& JtErr, double*& errNorm );

    void clear();
    void step();
    enum { DONE=0, STARTED=1, CALC_J=2, CHECK_ERR=3 };

    cv::Ptr<CvMat> mask;
    cv::Ptr<CvMat> prevParam;
    cv::Ptr<CvMat> param;
    cv::Ptr<CvMat> J;
    cv::Ptr<CvMat> err;
    cv::Ptr<CvMat> JtJ;
    cv::Ptr<CvMat> JtJN;
    cv::Ptr<CvMat> JtErr;
    cv::Ptr<CvMat> JtJV;
    cv::Ptr<CvMat> JtJW;
    double prevErrNorm, errNorm;
    int lambdaLg10;
    CvTermCriteria criteria;
    int state;
    int iters;
    bool completeSymmFlag;
    int solveMethod;
};
#endif

#endif
