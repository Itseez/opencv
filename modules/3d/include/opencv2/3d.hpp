// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_HPP
#define OPENCV_3D_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "opencv2/3d/depth.hpp"
#include "opencv2/3d/odometry.hpp"
#include "opencv2/3d/odometry_frame.hpp"
#include "opencv2/3d/odometry_settings.hpp"
#include "opencv2/3d/volume.hpp"
#include "opencv2/3d/ptcloud.hpp"

/**
  @defgroup _3d 3D vision functionality

Most of the functions in this section use a so-called pinhole camera model. The view of a scene
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

 */

namespace cv {

//! @addtogroup _3d
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

//! the algorithm for finding fundamental matrix
enum { FM_7POINT = 1, //!< 7-point algorithm
       FM_8POINT = 2, //!< 8-point algorithm
       FM_LMEDS  = 4, //!< least-median algorithm. 7-point algorithm is used.
       FM_RANSAC = 8  //!< RANSAC algorithm. It needs at least 15 points. 7-point algorithm is used.
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


/** @brief Type of matrix used in LevMarq solver

Matrix type can be dense, sparse or chosen automatically based on a matrix size, performance considerations or backend availability.

Note: only dense matrix is now supported
*/
enum class MatrixType
{
    AUTO = 0,
    DENSE = 1,
    SPARSE = 2
};

/** @brief Type of variables used in LevMarq solver

Variables can be linear, rotation (SO(3) group) or rigid transformation (SE(3) group) with corresponding jacobians and exponential updates.

Note: only linear variables are now supported
*/
enum class VariableType
{
    LINEAR = 0,
    SO3 = 1,
    SE3 = 2
};

/** @brief Levenberg-Marquadt solver

A Levenberg-Marquadt algorithm locally minimizes an objective function value (aka energy, cost or error) starting from
current param vector.
To do that, at each iteration it repeatedly calculates the energy at probe points until it's reduced.
To calculate a probe point, a linear equation is solved: (J^T*J + lambda*D)*dx = -J^T*b where J is a function jacobian,
b is a vector of residuals (aka errors or energy terms), D is a diagonal matrix generated from J^T*J diagonal
and lambda changes for each probe point. Then the resulting dx is "added" to current variable and it forms
a probe value. "Added" is quoted because in some groups (e.g. SO(3) group) such an increment can be a non-trivial operation.

For more details, please refer to Wikipedia page (https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).

This solver supports fixed variables and two forms of callback function:
1. Generating ordinary jacobian J and residual vector err ("long")
2. Generating normal equation matrix J^T*J and gradient vector J^T*err

Currently the solver supports dense jacobian matrix and linear parameter increment.
*/
class CV_EXPORTS LevMarq
{
public:
    /** @brief Optimization report

    The structure is returned when optimization is over.
    */
    struct CV_EXPORTS Report
    {
        Report(bool isFound, int nIters, double finalEnergy) :
            found(isFound), iters(nIters), energy(finalEnergy)
        { }
        // true if the cost function converged to a local minimum which is checked by check* fields, thresholds and other options
        // false if the cost function failed to converge because of error, amount of iterations exhausted or lambda explosion
        bool found;
        // amount of iterations elapsed until the optimization stopped
        int iters;
        // energy value reached by the optimization
        double energy;
    };

    /** @brief Structure to keep LevMarq settings

    The structure allows a user to pass algorithm parameters along with their names like this:
    @code
    MySolver solver(nVars, callback, MySolver::Settings().geodesicS(true).geoScale(1.0));
    @endcode
    */
    struct CV_EXPORTS Settings
    {
        Settings();

        inline Settings& setJacobiScaling          (bool   v) { jacobiScaling = v; return *this; }
        inline Settings& setUpDouble               (bool   v) { upDouble = v; return *this; }
        inline Settings& setUseStepQuality         (bool   v) { useStepQuality = v; return *this; }
        inline Settings& setClampDiagonal          (bool   v) { clampDiagonal = v; return *this; }
        inline Settings& setStepNormInf            (bool   v) { stepNormInf = v; return *this; }
        inline Settings& setCheckRelEnergyChange   (bool   v) { checkRelEnergyChange = v; return *this; }
        inline Settings& setCheckMinGradient       (bool   v) { checkMinGradient = v; return *this; }
        inline Settings& setCheckStepNorm          (bool   v) { checkStepNorm = v; return *this; }
        inline Settings& setGeodesic               (bool   v) { geodesic = v; return *this; }
        inline Settings& setHGeo                   (double v) { hGeo = v; return *this; }
        inline Settings& setGeoScale               (double v) { geoScale = v; return *this; }
        inline Settings& setStepNormTolerance      (double v) { stepNormTolerance = v; return *this; }
        inline Settings& setRelEnergyDeltaTolerance(double v) { relEnergyDeltaTolerance = v; return *this; }
        inline Settings& setMinGradientTolerance   (double v) { minGradientTolerance = v; return *this; }
        inline Settings& setSmallEnergyTolerance   (double v) { smallEnergyTolerance = v; return *this; }
        inline Settings& setMaxIterations          (int    v) { maxIterations = (unsigned int)v; return *this; }
        inline Settings& setInitialLambda          (double v) { initialLambda = v; return *this; }
        inline Settings& setInitialLmUpFactor      (double v) { initialLmUpFactor = v; return *this; }
        inline Settings& setInitialLmDownFactor    (double v) { initialLmDownFactor = v; return *this; }

        // normalize jacobian columns for better conditioning
        // slows down sparse solver, but maybe this'd be useful for some other solver
        bool jacobiScaling;
        // double upFactor until the probe is successful
        bool upDouble;
        // use stepQuality metrics for steps down
        bool useStepQuality;
        // clamp diagonal values added to J^T*J to pre-defined range of values
        bool clampDiagonal;
        // to use squared L2 norm or Inf norm for step size estimation
        bool stepNormInf;
        // to use relEnergyDeltaTolerance or not
        bool checkRelEnergyChange;
        // to use minGradientTolerance or not
        bool checkMinGradient;
        // to use stepNormTolerance or not
        bool checkStepNorm;
        // to use geodesic acceleration or not
        bool geodesic;
        // second directional derivative approximation step for geodesic acceleration
        double hGeo;
        // how much of geodesic acceleration is used
        double geoScale;
        // optimization stops when norm2(dx) drops below this value
        double stepNormTolerance;
        // optimization stops when relative energy change drops below this value
        double relEnergyDeltaTolerance;
        // optimization stops when max gradient value (J^T*b vector) drops below this value
        double minGradientTolerance;
        // optimization stops when energy drops below this value
        double smallEnergyTolerance;
        // optimization stops after a number of iterations performed
        unsigned int maxIterations;

        // LevMarq up and down params
        double initialLambda;
        double initialLmUpFactor;
        double initialLmDownFactor;
    };

    /** "Long" callback: f(param, &err, &J) -> bool
    Computes error and Jacobian for the specified vector of parameters,
    returns true on success.

    param: the current vector of parameters
    err: output vector of errors: err_i = actual_f_i - ideal_f_i
    J: output Jacobian: J_ij = d(ideal_f_i)/d(param_j)

    Param vector values may be changed by the callback only if they are fixed.
    Changing non-fixed variables may lead to incorrect results.
    When J=noArray(), it means that it does not need to be computed.
    Dimensionality of error vector and param vector can be different.
    The callback should explicitly allocate (with "create" method) each output array
    (unless it's noArray()).
    */
    typedef std::function<bool(InputOutputArray, OutputArray, OutputArray)> LongCallback;

    /** Normal callback: f(param, &JtErr, &JtJ, &errnorm) -> bool

        Computes squared L2 error norm, normal equation matrix J^T*J and J^T*err vector
        where J is MxN Jacobian: J_ij = d(err_i)/d(param_j)
        err is Mx1 vector of errors: err_i = actual_f_i - ideal_f_i
        M is a number of error terms, N is a number of variables to optimize.
        Make sense to use this class instead of usual Callback if the number
        of error terms greatly exceeds the number of variables.

        param: the current Nx1 vector of parameters
        JtErr: output Nx1 vector J^T*err
        JtJ: output NxN matrix J^T*J
        errnorm: output total error: dot(err, err)

        Param vector values may be changed by the callback only if they are fixed.
        Changing non-fixed variables may lead to incorrect results.
        If JtErr or JtJ are empty, they don't have to be computed.
        The callback should explicitly allocate (with "create" method) each output array
        (unless it's noArray()).
    */
    typedef std::function<bool(InputOutputArray, OutputArray, OutputArray, double&)> NormalCallback;

    /**
        Creates a solver

        @param nvars Number of variables in a param vector
        @param callback "Long" callback, produces jacobian and residuals for each energy term, returns true on success
        @param settings LevMarq settings structure, see LevMarqBase class for details
        @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
        @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
        @param paramType Type of optimized parameters; only LINEAR is supported now
        @param nerrs Energy terms amount. If zero, callback-generated jacobian size is used instead
        @param solveMethod What method to use for linear system solving
    */
    LevMarq(int nvars, LongCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
            MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, int nerrs = 0, int solveMethod = DECOMP_SVD);
    /**
        Creates a solver

        @param nvars Number of variables in a param vector
        @param callback Normal callback, produces J^T*J and J^T*b directly instead of J and b, returns true on success
        @param settings LevMarq settings structure, see LevMarqBase class for details
        @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
        @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
        @param paramType Type of optimized parameters; only LINEAR is supported now
        @param LtoR Indicates what part of symmetric matrix to copy to another part: lower or upper. Used only with alt. callback
        @param solveMethod What method to use for linear system solving
    */
    LevMarq(int nvars, NormalCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
            MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, bool LtoR = false, int solveMethod = DECOMP_SVD);

    /**
        Creates a solver

        @param param Input/output vector containing starting param vector and resulting optimized params
        @param callback "Long" callback, produces jacobian and residuals for each energy term, returns true on success
        @param settings LevMarq settings structure, see LevMarqBase class for details
        @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
        @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
        @param paramType Type of optimized parameters; only LINEAR is supported now
        @param nerrs Energy terms amount. If zero, callback-generated jacobian size is used instead
        @param solveMethod What method to use for linear system solving
    */
    LevMarq(InputOutputArray param, LongCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
            MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, int nerrs = 0, int solveMethod = DECOMP_SVD);
    /**
        Creates a solver

        @param param Input/output vector containing starting param vector and resulting optimized params
        @param callback Normal callback, produces J^T*J and J^T*b directly instead of J and b, returns true on success
        @param settings LevMarq settings structure, see LevMarqBase class for details
        @param mask Indicates what variables are fixed during optimization (zeros) and what vars to optimize (non-zeros)
        @param matrixType Type of matrix used in the solver; only DENSE and AUTO are supported now
        @param paramType Type of optimized parameters; only LINEAR is supported now
        @param LtoR Indicates what part of symmetric matrix to copy to another part: lower or upper. Used only with alt. callback
        @param solveMethod What method to use for linear system solving
    */
    LevMarq(InputOutputArray param, NormalCallback callback, const Settings& settings = Settings(), InputArray mask = noArray(),
            MatrixType matrixType = MatrixType::AUTO, VariableType paramType = VariableType::LINEAR, bool LtoR = false, int solveMethod = DECOMP_SVD);

    /**
        Runs Levenberg-Marquadt algorithm using current settings and given parameters vector.
        The method returns the optimization report.
    */
    Report optimize();

    /** @brief Runs optimization using the passed vector of parameters as the start point.

        The final vector of parameters (whether the algorithm converged or not) is stored at the same
        vector.
        This method can be used instead of the optimize() method if rerun with different start points is required.
        The method returns the optimization report.

        @param param initial/final vector of parameters.

        Note that the dimensionality of parameter space is defined by the size of param vector,
        and the dimensionality of optimized criteria is defined by the size of err vector
        computed by the callback.
    */
    Report run(InputOutputArray param);

private:
    class Impl;
    Ptr<Impl> pImpl;
};


/** @example samples/cpp/tutorial_code/features/Homography/pose_from_homography.cpp
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
                                 double aspectRatio = 0);

/** @overload */
CV_EXPORTS_AS(projectPointsSepJ) void projectPoints(
                    InputArray objectPoints,
                    InputArray rvec, InputArray tvec,
                    InputArray cameraMatrix, InputArray distCoeffs,
                    OutputArray imagePoints, OutputArray dpdr,
                    OutputArray dpdt, OutputArray dpdf=noArray(),
                    OutputArray dpdc=noArray(), OutputArray dpdk=noArray(),
                    OutputArray dpdo=noArray(), double aspectRatio=0.);

/** @example samples/cpp/tutorial_code/features/Homography/homography_from_camera_displacement.cpp
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
        modules/3d/src/solvepnp.cpp version 2.4.9)
        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
        to its calling of #undistortPoints (around line 75 of modules/3d/src/solvepnp.cpp version 2.4.9)
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
        opencv_source_code/samples/cpp/tutorial_code/3d/real_time_pose_estimation/
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
                                    TermCriteria criteria = TermCriteria(TermCriteria::EPS +
                                        TermCriteria::COUNT, 20, FLT_EPSILON));

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
                                     TermCriteria criteria = TermCriteria(TermCriteria::EPS +
                                         TermCriteria::COUNT, 20, FLT_EPSILON),
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
        modules/3d/src/solvepnp.cpp version 2.4.9)
        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
        to its calling of #undistortPoints (around line 75 of modules/3d/src/solvepnp.cpp version 2.4.9)
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
                                  bool useExtrinsicGuess = false,
                                  int flags = SOLVEPNP_ITERATIVE,
                                  InputArray rvec = noArray(), InputArray tvec = noArray(),
                                  OutputArray reprojectionError = noArray() );

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

/** @brief Converts points from Euclidean to homogeneous space.

@param src Input vector of N-dimensional points.
@param dst Output vector of N+1-dimensional points.
@param dtype The desired output array depth (either CV_32F or CV_64F are currently supported).
    If it's -1, then it's set automatically to CV_32F or CV_64F, depending on the input depth.

The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of
point coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).
 */
CV_EXPORTS_W void convertPointsToHomogeneous( InputArray src, OutputArray dst, int dtype=-1 );

/** @brief Converts points from homogeneous to Euclidean space.

@param src Input vector of N-dimensional points.
@param dst Output vector of N-1-dimensional points.
@param dtype The desired output array depth (either CV_32F or CV_64F are currently supported).
    If it's -1, then it's set automatically to CV_32F or CV_64F, depending on the input depth.

The function converts points homogeneous to Euclidean space using perspective projection. That is,
each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the
output point coordinates will be (0,0,0,...).
 */
CV_EXPORTS_W void convertPointsFromHomogeneous( InputArray src, OutputArray dst, int dtype=-1 );

/** @brief Converts points to/from homogeneous coordinates.

@param src Input array or vector of 2D, 3D, or 4D points.
@param dst Output vector of 2D, 3D, or 4D points.

The function converts 2D or 3D points from/to homogeneous coordinates by calling either
#convertPointsToHomogeneous or #convertPointsFromHomogeneous.

@note The function is obsolete. Use one of the previous two functions instead.
 */
CV_EXPORTS void convertPointsHomogeneous( InputArray src, OutputArray dst );

/** @example samples/cpp/snippets/epipolar_lines.cpp
An example using the findFundamentalMat function
*/
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

/** @brief Recovers the relative camera rotation and the translation from corresponding points in two images from two different cameras, using chirality check. Returns the number of
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
inliers in points1 and points2 for the given essential matrix E. Only these inliers will be used to
recover pose. In the output mask only inliers which pass the chirality check.

This function decomposes an essential matrix using @ref decomposeEssentialMat and then verifies
possible pose hypotheses by doing chirality check. The chirality check means that the
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
                            InputArray cameraMatrix, OutputArray R, OutputArray t,
                            double distanceThresh, InputOutputArray mask = noArray(),
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
CV_EXPORTS_W Mat estimateAffine2D(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = RANSAC, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);


CV_EXPORTS_W Mat estimateAffine2D(InputArray pts1, InputArray pts2, OutputArray inliers,
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

/** @example samples/cpp/tutorial_code/features/Homography/decompose_homography.cpp
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

/** @brief Returns the inscribed and bounding rectangles for the "undisorted" image plane.

The functions emulates undistortion of the image plane using the specified camera matrix,
distortion coefficients, the optional 3D rotation and the "new" camera matrix. In the case of
noticeable radial (or maybe pinclusion) distortion the rectangular image plane is distorted and
turns into some convex or concave shape. The function computes approximate inscribed (inner) and
bounding (outer) rectangles after such undistortion. The rectangles can be used to adjust
the newCameraMatrix so that the result image, for example, fits all the data from the original image
(at the expense of possibly big "black" areas) or, for another example, gets rid of black areas at the expense
some lost data near the original image edge. The function #getOptimalNewCameraMatrix uses this function
to compute the optimal new camera matrix.

@param cameraMatrix the original camera matrix.
@param distCoeffs distortion coefficients.
@param R the optional 3D rotation, applied before projection (see stereoRectify etc.)
@param newCameraMatrix the new camera matrix after undistortion. Usually it matches the original cameraMatrix.
@param imgSize the size of the image plane.
@param inner the output maximal inscribed rectangle of the undistorted image plane.
@param outer the output minimal bounding rectangle of the undistorted image plane.
 */
CV_EXPORTS void getUndistortRectangles(InputArray cameraMatrix, InputArray distCoeffs,
                                       InputArray R, InputArray newCameraMatrix, Size imgSize,
                                       Rect_<double>& inner, Rect_<double>& outer );

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
@param criteria termination criteria for the iterative point undistortion algorithm
 */
CV_EXPORTS_W
void undistortPoints(InputArray src, OutputArray dst,
                     InputArray cameraMatrix, InputArray distCoeffs,
                     InputArray R = noArray(), InputArray P = noArray(),
                     TermCriteria criteria=TermCriteria(TermCriteria::MAX_ITER, 5, 0.01));


/**
 * @brief Compute undistorted image points position
 *
 * @param src Observed points position, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or vector\<Point2f\> ).
 * @param dst Output undistorted points position (1xN/Nx1 2-channel or vector\<Point2f\> ).
 * @param cameraMatrix Camera matrix \f$\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
 * @param distCoeffs Distortion coefficients
 */
CV_EXPORTS_W
void undistortImagePoints(InputArray src, OutputArray dst, InputArray cameraMatrix,
                          InputArray distCoeffs,
                          TermCriteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 0.01));

namespace fisheye {

/** @brief Projects points using fisheye model

@param objectPoints Array of object points, 1xN/Nx1 3-channel (or vector\<Point3f\> ), where N is
the number of points in the view.
@param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or
vector\<Point2f\>.
@param affine
@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
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
@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
@param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
@param alpha The skew coefficient.
@param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .

Note that the function assumes the camera intrinsic matrix of the undistorted points to be identity.
This means if you want to distort image points you have to multiply them with \f$K^{-1}\f$ or
use another function overload.
 */
CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double alpha = 0);

/** @overload
Overload of distortPoints function to handle cases when undistorted points are got with non-identity
camera matrix, e.g. output of #estimateNewCameraMatrixForUndistortRectify.
@param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is
the number of points in the view.
@param Kundistorted Camera intrinsic matrix used as new camera matrix for undistortion.
@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
@param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
@param alpha The skew coefficient.
@param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
@sa estimateNewCameraMatrixForUndistortRectify
*/
CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray Kundistorted, InputArray K, InputArray D, double alpha = 0);

/** @brief Undistorts 2D points using fisheye model

@param distorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is the
number of points in the view.
@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
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

/** @brief Estimates new camera intrinsic matrix for undistortion or rectification.

@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
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

/** @brief Computes undistortion and rectification maps for image transform by cv::remap(). If D is empty zero
distortion is used, if R or P is empty identity matrixes are used.

@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
@param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
@param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
1-channel or 1x1 3-channel
@param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
@param size Undistorted image size.
@param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
for details.
@param map1 The first output map.
@param map2 The second output map.
 */
CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, InputArray R, InputArray P,
    const cv::Size& size, int m1type, OutputArray map1, OutputArray map2);

/** @brief Transforms an image to compensate for fisheye lens distortion.

@param distorted image with fisheye lens distortion.
@param undistorted Output image with compensated fisheye lens distortion.
@param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
@param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
@param Knew Camera intrinsic matrix of the distorted image. By default, it is the identity matrix but you
may additionally scale and shift the result by using a different matrix.
@param new_size the new size

The function transforms an image to compensate radial and tangential lens distortion.

The function is simply a combination of fisheye::initUndistortRectifyMap (with unity R ) and remap
(with bilinear interpolation). See the former function for details of the transformation being
performed.

See below the results of undistortImage.
   -   a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
        k_4, k_5, k_6) of distortion were optimized under calibration)
    -   b\) result of fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
        k_3, k_4) of fisheye distortion were optimized under calibration)
    -   c\) original image was captured with fisheye lens

Pictures a) and b) almost the same. But if we consider points of image located far from the center
of image, we can notice that on image a) these points are distorted.

![image](pics/fisheye_undistorted.jpg)
 */
CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted,
    InputArray K, InputArray D, InputArray Knew = cv::noArray(), const Size& new_size = Size());

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
thus the input are very similar. Check there and Perspective-n-Points is described in @ref calib3d_solvePnP
for more information.
*/
CV_EXPORTS_W bool solvePnP( InputArray objectPoints, InputArray imagePoints,
                            InputArray cameraMatrix, InputArray distCoeffs,
                            OutputArray rvec, OutputArray tvec,
                            bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE,
                            TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1e-8)
                          );

} // namespace fisheye

/** @brief Octree for 3D vision.
 *
 * In 3D vision filed, the Octree is used to process and accelerate the pointcloud data. The class Octree represents
 * the Octree data structure. Each Octree will have a fixed depth. The depth of Octree refers to the distance from
 * the root node to the leaf node.All OctreeNodes will not exceed this depth.Increasing the depth will increase
 * the amount of calculation exponentially. And the small number of depth refers low resolution of Octree.
 * Each node contains 8 children, which are used to divide the space cube into eight parts. Each octree node represents
 * a cube. And these eight children will have a fixed order, the order is described as follows:
 *
 * For illustration, assume,
 *
 * rootNode: origin == (0, 0, 0), size == 2
 *
 * Then,
 *
 * children[0]: origin == (0, 0, 0), size == 1
 *
 * children[1]: origin == (1, 0, 0), size == 1, along X-axis next to child 0
 *
 * children[2]: origin == (0, 1, 0), size == 1, along Y-axis next to child 0
 *
 * children[3]: origin == (1, 1, 0), size == 1, in X-Y plane
 *
 * children[4]: origin == (0, 0, 1), size == 1, along Z-axis next to child 0
 *
 * children[5]: origin == (1, 0, 1), size == 1, in X-Z plane
 *
 * children[6]: origin == (0, 1, 1), size == 1, in Y-Z plane
 *
 * children[7]: origin == (1, 1, 1), size == 1, furthest from child 0
 */

class CV_EXPORTS_W Octree
{
public:
    //! Default constructor.
    Octree();

    /** @overload
     * @brief Creates an empty Octree with given maximum depth
     *
     * @param maxDepth The max depth of the Octree
     * @param size bounding box size for the Octree
     * @param origin Initial center coordinate
     * @param withColors Whether to keep per-point colors or not
     * @return resulting Octree
     */
    CV_WRAP static Ptr<Octree> createWithDepth(int maxDepth, double size, const Point3f& origin = { }, bool withColors = false);

    /** @overload
     * @brief Create an Octree from the PointCloud data with the specific maxDepth
     *
     * @param maxDepth Max depth of the octree
     * @param pointCloud point cloud data, should be 3-channel float array
     * @param colors color attribute of point cloud in the same 3-channel float format
     * @return resulting Octree
     */
    CV_WRAP static Ptr<Octree> createWithDepth(int maxDepth, InputArray pointCloud, InputArray colors = noArray());

    /** @overload
     * @brief Creates an empty Octree with given resolution
     *
     * @param resolution The size of the octree leaf node
     * @param size bounding box size for the Octree
     * @param origin Initial center coordinate
     * @param withColors Whether to keep per-point colors or not
     * @return resulting Octree
     */
    CV_WRAP static Ptr<Octree> createWithResolution(double resolution, double size, const Point3f& origin = { }, bool withColors = false);

     /** @overload
     * @brief Create an Octree from the PointCloud data with the specific resolution
     *
     * @param resolution The size of the octree leaf node
     * @param pointCloud point cloud data, should be 3-channel float array
     * @param colors color attribute of point cloud in the same 3-channel float format
     * @return resulting octree
     */
    CV_WRAP static Ptr<Octree> createWithResolution(double resolution, InputArray pointCloud, InputArray colors = noArray());

    //! Default destructor
    ~Octree();

    /** @overload
    * @brief Insert a point data with color to a OctreeNode.
    *
    * @param point The point data in Point3f format.
    * @param color The color attribute of point in Point3f format.
    * @return Returns whether the insertion is successful.
    */
    CV_WRAP bool insertPoint(const Point3f& point, const Point3f& color = { });

    /** @brief Determine whether the point is within the space range of the specific cube.
     *
     * @param point The point coordinates.
     * @return If point is in bound, return ture. Otherwise, false.
     */
    CV_WRAP bool isPointInBound(const Point3f& point) const;

    //! returns true if the rootnode is NULL.
    CV_WRAP bool empty() const;

    /** @brief Reset all octree parameter.
    *
    *  Clear all the nodes of the octree and initialize the parameters.
    */
    CV_WRAP void clear();

    /** @brief Delete a given point from the Octree.
    *
    * Delete the corresponding element from the pointList in the corresponding leaf node. If the leaf node
    * does not contain other points after deletion, this node will be deleted. In the same way,
    * its parent node may also be deleted if its last child is deleted.
    * @param point The point coordinates, comparison is epsilon-based
    * @return return ture if the point is deleted successfully.
    */
    CV_WRAP bool deletePoint(const Point3f& point);

    /** @brief restore point cloud data from Octree.
    *
    * Restore the point cloud data from existing octree. The points in same leaf node will be seen as the same point.
    * This point is the center of the leaf node. If the resolution is small, it will work as a downSampling function.
    * @param restoredPointCloud The output point cloud data, can be replaced by noArray() if not needed
    * @param restoredColor The color attribute of point cloud data, can be omitted if not needed
    */
    CV_WRAP void getPointCloudByOctree(OutputArray restoredPointCloud, OutputArray restoredColor = noArray());

    /** @brief Radius Nearest Neighbor Search in Octree.
    *
    * Search all points that are less than or equal to radius.
    * And return the number of searched points.
    * @param query Query point.
    * @param radius Retrieved radius value.
    * @param points Point output. Contains searched points in 3-float format, and output vector is not in order,
    * can be replaced by noArray() if not needed
    * @param squareDists Dist output. Contains searched squared distance in floats, and output vector is not in order,
    * can be omitted if not needed
    * @return the number of searched points.
    */
    CV_WRAP int radiusNNSearch(const Point3f& query, float radius, OutputArray points, OutputArray squareDists = noArray()) const;

    /** @overload
    *  @brief Radius Nearest Neighbor Search in Octree.
    *
    * Search all points that are less than or equal to radius.
    * And return the number of searched points.
    * @param query Query point.
    * @param radius Retrieved radius value.
    * @param points Point output. Contains searched points in 3-float format, and output vector is not in order,
    * can be replaced by noArray() if not needed
    * @param colors Color output. Contains colors corresponding to points in pointSet, can be replaced by noArray() if not needed
    * @param squareDists Dist output. Contains searched squared distance in floats, and output vector is not in order,
    * can be replaced by noArray() if not needed
    * @return the number of searched points.
    */
    CV_WRAP int radiusNNSearch(const Point3f& query, float radius, OutputArray points, OutputArray colors, OutputArray squareDists) const;

    /** @brief K Nearest Neighbor Search in Octree.
    *
    * Find the K nearest neighbors to the query point.
    * @param query Query point.
    * @param K amount of nearest neighbors to find
    * @param points Point output. Contains K points in 3-float format, arranged in order of distance from near to far,
    * can be replaced by noArray() if not needed
    * @param squareDists Dist output. Contains K squared distance in floats, arranged in order of distance from near to far,
    * can be omitted if not needed
    */
    CV_WRAP void KNNSearch(const Point3f& query, const int K, OutputArray points, OutputArray squareDists = noArray()) const;

    /** @overload
    *  @brief K Nearest Neighbor Search in Octree.
    *
    * Find the K nearest neighbors to the query point.
    * @param query Query point.
    * @param K amount of nearest neighbors to find
    * @param points Point output. Contains K points in 3-float format, arranged in order of distance from near to far,
    * can be replaced by noArray() if not needed
    * @param colors Color output. Contains colors corresponding to points in pointSet, can be replaced by noArray() if not needed
    * @param squareDists Dist output. Contains K squared distance in floats, arranged in order of distance from near to far,
    * can be replaced by noArray() if not needed
    */
    CV_WRAP void KNNSearch(const Point3f& query, const int K, OutputArray points, OutputArray colors, OutputArray squareDists) const;

protected:
    struct Impl;
    Ptr<Impl> p;
};


/** @brief Loads a point cloud from a file.
*
* The function loads point cloud from the specified file and returns it.
* If the cloud cannot be read, throws an error.
* Vertex coordinates, normals and colors are returned as they are saved in the file
* even if these arrays have different sizes and their elements do not correspond to each other
* (which is typical for OBJ files for example)
*
* Currently, the following file formats are supported:
* -  [Wavefront obj file *.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
* -  [Polygon File Format *.ply](https://en.wikipedia.org/wiki/PLY_(file_format))
*
* @param filename Name of the file
* @param vertices vertex coordinates, each value contains 3 floats
* @param normals per-vertex normals, each value contains 3 floats
* @param rgb per-vertex colors, each value contains 3 floats
*/
CV_EXPORTS_W void loadPointCloud(const String &filename, OutputArray vertices, OutputArray normals = noArray(), OutputArray rgb = noArray());

/** @brief Saves a point cloud to a specified file.
*
* The function saves point cloud to the specified file.
* File format is chosen based on the filename extension.
*
* @param filename Name of the file
* @param vertices vertex coordinates, each value contains 3 floats
* @param normals per-vertex normals, each value contains 3 floats
* @param rgb per-vertex colors, each value contains 3 floats
*/
CV_EXPORTS_W void savePointCloud(const String &filename, InputArray vertices, InputArray normals = noArray(), InputArray rgb = noArray());

/** @brief Loads a mesh from a file.
*
* The function loads mesh from the specified file and returns it.
* If the mesh cannot be read, throws an error
* Vertex attributes (i.e. space and texture coodinates, normals and colors) are returned in same-sized
* arrays with corresponding elements having the same indices.
* This means that if a face uses a vertex with a normal or a texture coordinate with different indices
* (which is typical for OBJ files for example), this vertex will be duplicated for each face it uses.
*
* Currently, the following file formats are supported:
* -  [Wavefront obj file *.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file) (ONLY TRIANGULATED FACES)
* -  [Polygon File Format *.ply](https://en.wikipedia.org/wiki/PLY_(file_format))
* @param filename Name of the file
* @param vertices vertex coordinates, each value contains 3 floats
* @param indices per-face list of vertices, each value is a vector of ints
* @param normals per-vertex normals, each value contains 3 floats
* @param colors per-vertex colors, each value contains 3 floats
* @param texCoords per-vertex texture coordinates, each value contains 2 or 3 floats
*/
CV_EXPORTS_W void loadMesh(const String &filename, OutputArray vertices, OutputArrayOfArrays indices,
                           OutputArray normals = noArray(), OutputArray colors = noArray(),
                           OutputArray texCoords = noArray());

/** @brief Saves a mesh to a specified file.
*
* The function saves mesh to the specified file.
* File format is chosen based on the filename extension.
*
* @param filename Name of the file.
* @param vertices vertex coordinates, each value contains 3 floats
* @param indices per-face list of vertices, each value is a vector of ints
* @param normals per-vertex normals, each value contains 3 floats
* @param colors per-vertex colors, each value contains 3 floats
* @param texCoords per-vertex texture coordinates, each value contains 2 or 3 floats
*/
CV_EXPORTS_W void saveMesh(const String &filename, InputArray vertices, InputArrayOfArrays indices,
                           InputArray normals = noArray(), InputArray colors = noArray(), InputArray texCoords = noArray());


//! Triangle fill settings
enum TriangleShadingType
{
    RASTERIZE_SHADING_WHITE  = 0, //!< a white color is used for the whole triangle
    RASTERIZE_SHADING_FLAT   = 1, //!< a color of 1st vertex of each triangle is used
    RASTERIZE_SHADING_SHADED = 2  //!< a color is interpolated between 3 vertices with perspective correction
};

//! Face culling settings: what faces are drawn after face culling
enum TriangleCullingMode
{
    RASTERIZE_CULLING_NONE = 0, //!< all faces are drawn, no culling is actually performed
    RASTERIZE_CULLING_CW   = 1, //!< triangles which vertices are given in clockwork order are drawn
    RASTERIZE_CULLING_CCW  = 2  //!< triangles which vertices are given in counterclockwork order are drawn
};

//! GL compatibility settings
enum TriangleGlCompatibleMode
{
    RASTERIZE_COMPAT_DISABLED = 0, //!< Color and depth have their natural values and converted to internal formats if needed
    RASTERIZE_COMPAT_INVDEPTH = 1  //!< Color is natural, Depth is transformed from [-zNear; -zFar] to [0; 1]
                                   //!< by the following formula: \f$ \frac{z_{far} \left(z + z_{near}\right)}{z \left(z_{far} - z_{near}\right)} \f$ \n
                                   //!< In this mode the input/output depthBuf is considered to be in this format,
                                   //!< therefore it's faster since there're no conversions performed
};

/**
 * @brief Structure to keep settings for rasterization
 */
struct CV_EXPORTS_W_SIMPLE TriangleRasterizeSettings
{
    TriangleRasterizeSettings();

    CV_WRAP TriangleRasterizeSettings& setShadingType(TriangleShadingType st) { shadingType = st; return *this; }
    CV_WRAP TriangleRasterizeSettings& setCullingMode(TriangleCullingMode cm) { cullingMode = cm; return *this; }
    CV_WRAP TriangleRasterizeSettings& setGlCompatibleMode(TriangleGlCompatibleMode gm) { glCompatibleMode = gm; return *this; }

    TriangleShadingType shadingType;
    TriangleCullingMode cullingMode;
    TriangleGlCompatibleMode glCompatibleMode;
};


/** @brief Renders a set of triangles on a depth and color image

Triangles can be drawn white (1.0, 1.0, 1.0), flat-shaded or with a color interpolation between vertices.
In flat-shaded mode the 1st vertex color of each triangle is used to fill the whole triangle.

The world2cam is an inverted camera pose matrix in fact. It transforms vertices from world to
camera coordinate system.

The camera coordinate system emulates the OpenGL's coordinate system having coordinate origin in a screen center,
X axis pointing right, Y axis pointing up and Z axis pointing towards the viewer
except that image is vertically flipped after the render.
This means that all visible objects are placed in z-negative area, or exactly in -zNear > z > -zFar since
zNear and zFar are positive.
For example, at fovY = PI/2 the point (0, 1, -1) will be projected to (width/2, 0) screen point,
(1, 0, -1) to (width/2 + height/2, height/2). Increasing fovY makes projection smaller and vice versa.

The function does not create or clear output images before the rendering. This means that it can be used
for drawing over an existing image or for rendering a model into a 3D scene using pre-filled Z-buffer.

Empty scene results in a depth buffer filled by the maximum value since every pixel is infinitely far from the camera.
Therefore, before rendering anything from scratch the depthBuf should be filled by zFar values (or by ones in INVDEPTH mode).

There are special versions of this function named triangleRasterizeDepth and triangleRasterizeColor
for cases if a user needs a color image or a depth image alone; they may run slightly faster.

@param vertices vertices coordinates array. Should contain values of CV_32FC3 type or a compatible one (e.g. cv::Vec3f, etc.)
@param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array.
Should contain CV_32SC3 values or compatible
@param colors per-vertex colors of CV_32FC3 type or compatible. Can be empty or the same size as vertices array.
If the values are out of [0; 1] range, the result correctness is not guaranteed
@param colorBuf an array representing the final rendered image. Should containt CV_32FC3 values and be the same size as depthBuf.
Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
@param depthBuf an array of floats containing resulting Z buffer. Should contain float values and be the same size as colorBuf.
Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
Empty scene corresponds to all values set to zFar (or to 1.0 in INVDEPTH mode)
@param world2cam a 4x3 or 4x4 float or double matrix containing inverted (sic!) camera pose
@param fovY field of view in vertical direction, given in radians
@param zNear minimum Z value to render, everything closer is clipped
@param zFar maximum Z value to render, everything farther is clipped
@param settings see TriangleRasterizeSettings. By default the smooth shading is on,
with CW culling and with disabled GL compatibility
*/
CV_EXPORTS_W void triangleRasterize(InputArray vertices, InputArray indices, InputArray colors,
                                    InputOutputArray colorBuf, InputOutputArray depthBuf,
                                    InputArray world2cam, double fovY, double zNear, double zFar,
                                    const TriangleRasterizeSettings& settings = TriangleRasterizeSettings());

/** @brief Overloaded version of triangleRasterize() with depth-only rendering

@param vertices vertices coordinates array. Should contain values of CV_32FC3 type or a compatible one (e.g. cv::Vec3f, etc.)
@param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array.
Should contain CV_32SC3 values or compatible
@param depthBuf an array of floats containing resulting Z buffer. Should contain float values and be the same size as colorBuf.
Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
Empty scene corresponds to all values set to zFar (or to 1.0 in INVDEPTH mode)
@param world2cam a 4x3 or 4x4 float or double matrix containing inverted (sic!) camera pose
@param fovY field of view in vertical direction, given in radians
@param zNear minimum Z value to render, everything closer is clipped
@param zFar maximum Z value to render, everything farther is clipped
@param settings see TriangleRasterizeSettings. By default the smooth shading is on,
with CW culling and with disabled GL compatibility
*/
CV_EXPORTS_W void triangleRasterizeDepth(InputArray vertices, InputArray indices, InputOutputArray depthBuf,
                                         InputArray world2cam, double fovY, double zNear, double zFar,
                                         const TriangleRasterizeSettings& settings = TriangleRasterizeSettings());

/** @brief Overloaded version of triangleRasterize() with color-only rendering

@param vertices vertices coordinates array. Should contain values of CV_32FC3 type or a compatible one (e.g. cv::Vec3f, etc.)
@param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array.
Should contain CV_32SC3 values or compatible
@param colors per-vertex colors of CV_32FC3 type or compatible. Can be empty or the same size as vertices array.
If the values are out of [0; 1] range, the result correctness is not guaranteed
@param colorBuf an array representing the final rendered image. Should containt CV_32FC3 values and be the same size as depthBuf.
Not cleared before rendering, i.e. the content is reused as there is some pre-rendered scene.
@param world2cam a 4x3 or 4x4 float or double matrix containing inverted (sic!) camera pose
@param fovY field of view in vertical direction, given in radians
@param zNear minimum Z value to render, everything closer is clipped
@param zFar maximum Z value to render, everything farther is clipped
@param settings see TriangleRasterizeSettings. By default the smooth shading is on,
with CW culling and with disabled GL compatibility
*/
CV_EXPORTS_W void triangleRasterizeColor(InputArray vertices, InputArray indices, InputArray colors, InputOutputArray colorBuf,
                                         InputArray world2cam, double fovY, double zNear, double zFar,
                                         const TriangleRasterizeSettings& settings = TriangleRasterizeSettings());

/** @brief Spectral Cluster for Mesh Segmentation
 *
 * The SpectralCluster class implements a method for segmenting a 3D mesh into k segments of triangular faces using
 * spectral clustering.
 * This technique combines constructing the affinity matrix from the input mesh, and using eigen decomposition to
 * project data into a k-dimensional space. The K-Means algorithm is then applied to the eigenvectors to generate
 * segment labels.
 *
 * For more detailed description, see https://github.com/opencv/opencv/pull/25597
*/
class CV_EXPORTS SpectralCluster {
public:
    /** @brief Mesh Spectral Cluster

    @param delta A float number between 0 and 1 controlling the relative importance of geodesic distance and angle distance.
    Usually, 0 < delta < 0.2
    @param eta A float number between 0 and 1, which gives weight to concavity in clustering. Usually, 0 < eta < 0.1
    */
    SpectralCluster();
    SpectralCluster(float delta, float eta);

    /** @brief Cluster the input mesh into k segments
    @param vertices vertices coordinates array, should contain values of Point3f
    @param indices triangle vertices index array, 3 per triangle. Each index indicates a vertex in a vertices array
    @param k integer, indicates number of segments
    @param result output labels corresponding to each triangular face
    */
    void cluster(const std::vector<cv::Point3f> &vertices, const std::vector<std::vector<int32_t>> &indices, int k,
                 OutputArray result);

private:
    class Impl;
    Ptr<Impl> pImpl;
};

//! @} _3d
} //end namespace cv

#endif
