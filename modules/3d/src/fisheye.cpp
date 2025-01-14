// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {

struct JacobianRow
{
    Vec2d df, dc;
    Vec4d dk;
    Vec3d dom, dT;
    double dalpha;
};

void cv::fisheye::projectPoints(InputArray objectPoints, OutputArray imagePoints, const Affine3d& affine,
    InputArray K, InputArray D, double alpha, OutputArray jacobian)
{
    CV_INSTRUMENT_REGION();

    projectPoints(objectPoints, imagePoints, affine.rvec(), affine.translation(), K, D, alpha, jacobian);
}

void cv::fisheye::projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray _rvec,
        InputArray _tvec, InputArray _K, InputArray _D, double alpha, OutputArray jacobian)
{
    CV_INSTRUMENT_REGION();

    // will support only 3-channel data now for points
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));
    size_t n = objectPoints.total();

    CV_Assert(_rvec.total() * _rvec.channels() == 3 && (_rvec.depth() == CV_32F || _rvec.depth() == CV_64F));
    CV_Assert(_tvec.total() * _tvec.channels() == 3 && (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));
    CV_Assert(_tvec.getMat().isContinuous() && _rvec.getMat().isContinuous());

    Vec3d om = _rvec.depth() == CV_32F ? (Vec3d)*_rvec.getMat().ptr<Vec3f>() : *_rvec.getMat().ptr<Vec3d>();
    Vec3d T  = _tvec.depth() == CV_32F ? (Vec3d)*_tvec.getMat().ptr<Vec3f>() : *_tvec.getMat().ptr<Vec3d>();

    CV_Assert(_K.size() == Size(3,3) && (_K.type() == CV_32F || _K.type() == CV_64F) && _D.type() == _K.type() && _D.total() == 4);

    Vec2d f, c;
    if (_K.depth() == CV_32F)
    {

        Matx33f K = _K.getMat();
        f = Vec2f(K(0, 0), K(1, 1));
        c = Vec2f(K(0, 2), K(1, 2));
    }
    else
    {
        Matx33d K = _K.getMat();
        f = Vec2d(K(0, 0), K(1, 1));
        c = Vec2d(K(0, 2), K(1, 2));
    }

    Vec4d k = _D.depth() == CV_32F ? (Vec4d)*_D.getMat().ptr<Vec4f>(): *_D.getMat().ptr<Vec4d>();

    const bool isJacobianNeeded = jacobian.needed();
    JacobianRow *Jn = 0;
    if (isJacobianNeeded)
    {
        int nvars = 2 + 2 + 1 + 4 + 3 + 3; // f, c, alpha, k, om, T,
        jacobian.create(2*(int)n, nvars, CV_64F);
        Jn = jacobian.getMat().ptr<JacobianRow>(0);
    }

    Matx33d R;
    Matx<double, 3, 9> dRdom;
    Rodrigues(om, R, dRdom);
    Affine3d aff(om, T);

    const Vec3f* Xf = objectPoints.getMat().ptr<Vec3f>();
    const Vec3d* Xd = objectPoints.getMat().ptr<Vec3d>();
    Vec2f *xpf = imagePoints.getMat().ptr<Vec2f>();
    Vec2d *xpd = imagePoints.getMat().ptr<Vec2d>();

    for(size_t i = 0; i < n; ++i)
    {
        Vec3d Xi = objectPoints.depth() == CV_32F ? (Vec3d)Xf[i] : Xd[i];
        Vec3d Y = aff*Xi;
        if (fabs(Y[2]) < DBL_MIN)
            Y[2] = 1;
        Vec2d x(Y[0]/Y[2], Y[1]/Y[2]);

        double r2 = x.dot(x);
        double r = std::sqrt(r2);

        // Angle of the incoming ray:
        double theta = std::atan(r);

        double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
                theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

        double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;

        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        Vec2d xd1 = x * cdist;
        Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
        Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

        if (objectPoints.depth() == CV_32F)
            xpf[i] = final_point;
        else
            xpd[i] = final_point;

        if (isJacobianNeeded)
        {
            //Vec3d Xi = pdepth == CV_32F ? (Vec3d)Xf[i] : Xd[i];
            //Vec3d Y = aff*Xi;
            double dYdR[] = { Xi[0], Xi[1], Xi[2], 0, 0, 0, 0, 0, 0,
                              0, 0, 0, Xi[0], Xi[1], Xi[2], 0, 0, 0,
                              0, 0, 0, 0, 0, 0, Xi[0], Xi[1], Xi[2] };

            Matx33d dYdom_data = Matx<double, 3, 9>(dYdR) * dRdom.t();
            const Vec3d *dYdom = (Vec3d*)dYdom_data.val;

            Matx33d dYdT_data = Matx33d::eye();
            const Vec3d *dYdT = (Vec3d*)dYdT_data.val;

            //Vec2d x(Y[0]/Y[2], Y[1]/Y[2]);
            Vec3d dxdom[2];
            dxdom[0] = (1.0/Y[2]) * dYdom[0] - x[0]/Y[2] * dYdom[2];
            dxdom[1] = (1.0/Y[2]) * dYdom[1] - x[1]/Y[2] * dYdom[2];

            Vec3d dxdT[2];
            dxdT[0]  = (1.0/Y[2]) * dYdT[0] - x[0]/Y[2] * dYdT[2];
            dxdT[1]  = (1.0/Y[2]) * dYdT[1] - x[1]/Y[2] * dYdT[2];

            //double r2 = x.dot(x);
            Vec3d dr2dom = 2 * x[0] * dxdom[0] + 2 * x[1] * dxdom[1];
            Vec3d dr2dT  = 2 * x[0] *  dxdT[0] + 2 * x[1] *  dxdT[1];

            //double r = std::sqrt(r2);
            double drdr2 = r > 1e-8 ? 1.0/(2*r) : 1;
            Vec3d drdom = drdr2 * dr2dom;
            Vec3d drdT  = drdr2 * dr2dT;

            // Angle of the incoming ray:
            //double theta = atan(r);
            double dthetadr = 1.0/(1+r2);
            Vec3d dthetadom = dthetadr * drdom;
            Vec3d dthetadT  = dthetadr *  drdT;

            //double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;
            double dtheta_ddtheta = 1 + 3*k[0]*theta2 + 5*k[1]*theta4 + 7*k[2]*theta6 + 9*k[3]*theta8;
            Vec3d dtheta_ddom = dtheta_ddtheta * dthetadom;
            Vec3d dtheta_ddT  = dtheta_ddtheta * dthetadT;
            Vec4d dtheta_ddk  = Vec4d(theta3, theta5, theta7, theta9);

            //double inv_r = r > 1e-8 ? 1.0/r : 1;
            //double cdist = r > 1e-8 ? theta_d / r : 1;
            Vec3d dcdistdom = inv_r * (dtheta_ddom - cdist*drdom);
            Vec3d dcdistdT  = inv_r * (dtheta_ddT  - cdist*drdT);
            Vec4d dcdistdk  = inv_r *  dtheta_ddk;

            //Vec2d xd1 = x * cdist;
            Vec4d dxd1dk[2];
            Vec3d dxd1dom[2], dxd1dT[2];
            dxd1dom[0] = x[0] * dcdistdom + cdist * dxdom[0];
            dxd1dom[1] = x[1] * dcdistdom + cdist * dxdom[1];
            dxd1dT[0]  = x[0] * dcdistdT  + cdist * dxdT[0];
            dxd1dT[1]  = x[1] * dcdistdT  + cdist * dxdT[1];
            dxd1dk[0]  = x[0] * dcdistdk;
            dxd1dk[1]  = x[1] * dcdistdk;

            //Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
            Vec4d dxd3dk[2];
            Vec3d dxd3dom[2], dxd3dT[2];
            dxd3dom[0] = dxd1dom[0] + alpha * dxd1dom[1];
            dxd3dom[1] = dxd1dom[1];
            dxd3dT[0]  = dxd1dT[0]  + alpha * dxd1dT[1];
            dxd3dT[1]  = dxd1dT[1];
            dxd3dk[0]  = dxd1dk[0]  + alpha * dxd1dk[1];
            dxd3dk[1]  = dxd1dk[1];

            Vec2d dxd3dalpha(xd1[1], 0);

            //final jacobian
            Jn[0].dom = f[0] * dxd3dom[0];
            Jn[1].dom = f[1] * dxd3dom[1];

            Jn[0].dT = f[0] * dxd3dT[0];
            Jn[1].dT = f[1] * dxd3dT[1];

            Jn[0].dk = f[0] * dxd3dk[0];
            Jn[1].dk = f[1] * dxd3dk[1];

            Jn[0].dalpha = f[0] * dxd3dalpha[0];
            Jn[1].dalpha = 0; //f[1] * dxd3dalpha[1];

            Jn[0].df = Vec2d(xd3[0], 0);
            Jn[1].df = Vec2d(0, xd3[1]);

            Jn[0].dc = Vec2d(1, 0);
            Jn[1].dc = Vec2d(0, 1);

            //step to jacobian rows for next point
            Jn += 2;
        }
    }
}

void cv::fisheye::distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double alpha)
{
    CV_INSTRUMENT_REGION();

    // will support only 2-channel data now for points
    CV_Assert(undistorted.type() == CV_32FC2 || undistorted.type() == CV_64FC2);
    distorted.create(undistorted.size(), undistorted.type());
    size_t n = undistorted.total();

    CV_Assert(K.size() == Size(3,3) && (K.type() == CV_32F || K.type() == CV_64F) && D.total() == 4);

    Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0 ,2), camMat(1, 2));
    }

    Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    const Vec2f* Xf = undistorted.getMat().ptr<Vec2f>();
    const Vec2d* Xd = undistorted.getMat().ptr<Vec2d>();
    Vec2f *xpf = distorted.getMat().ptr<Vec2f>();
    Vec2d *xpd = distorted.getMat().ptr<Vec2d>();

    for(size_t i = 0; i < n; ++i)
    {
        Vec2d x = undistorted.depth() == CV_32F ? (Vec2d)Xf[i] : Xd[i];

        double r2 = x.dot(x);
        double r = std::sqrt(r2);

        // Angle of the incoming ray:
        double theta = std::atan(r);

        double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
                theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

        double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;

        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        Vec2d xd1 = x * cdist;
        Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
        Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

        if (undistorted.depth() == CV_32F)
            xpf[i] = final_point;
        else
            xpd[i] = final_point;
    }
}

void cv::fisheye::distortPoints(InputArray _undistorted, OutputArray distorted, InputArray Kundistorted, InputArray K, InputArray D, double alpha)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(_undistorted.type() == CV_32FC2 || _undistorted.type() == CV_64FC2);
    CV_Assert(Kundistorted.size() == Size(3,3) && (Kundistorted.type() == CV_32F || Kundistorted.type() == CV_64F));

    cv::Mat undistorted = _undistorted.getMat();
    cv::Mat normalized(undistorted.size(), CV_64FC2);

    Mat Knew = Kundistorted.getMat();

    double cx, cy, fx, fy;
    if (Knew.depth() == CV_32F)
    {
        fx = (double)Knew.at<float>(0, 0);
        fy = (double)Knew.at<float>(1, 1);
        cx = (double)Knew.at<float>(0, 2);
        cy = (double)Knew.at<float>(1, 2);
    }
    else
    {
        fx = Knew.at<double>(0, 0);
        fy = Knew.at<double>(1, 1);
        cx = Knew.at<double>(0, 2);
        cy = Knew.at<double>(1, 2);
    }

    size_t n = undistorted.total();
    const Vec2f* Xf = undistorted.ptr<Vec2f>();
    const Vec2d* Xd = undistorted.ptr<Vec2d>();
    Vec2d* normXd = normalized.ptr<Vec2d>();
    for (size_t i = 0; i < n; i++)
    {
        Vec2d p = undistorted.depth() == CV_32F ? (Vec2d)Xf[i] : Xd[i];
        normXd[i][0] = (p[0] - cx) / fx;
        normXd[i][1] = (p[1] - cy) / fy;
    }

    cv::fisheye::distortPoints(normalized, distorted, K, D, alpha);
}

void cv::fisheye::undistortPoints( InputArray distorted, OutputArray undistorted, InputArray K, InputArray D,
                                   InputArray R, InputArray P, TermCriteria criteria)
{
    CV_INSTRUMENT_REGION();

    // will support only 2-channel data now for points
    CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());

    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

    CV_Assert(criteria.isValid());

    Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
    }

    Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    Matx33d RR = Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    if(!P.empty())
    {
        Matx33d PP;
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }

    // start undistorting
    const Vec2f* srcf = distorted.getMat().ptr<Vec2f>();
    const Vec2d* srcd = distorted.getMat().ptr<Vec2d>();
    Vec2f* dstf = undistorted.getMat().ptr<Vec2f>();
    Vec2d* dstd = undistorted.getMat().ptr<Vec2d>();

    size_t n = distorted.total();
    int sdepth = distorted.depth();

    const bool isEps = (criteria.type & TermCriteria::EPS) != 0;

    /* Define max count for solver iterations */
    int maxCount = std::numeric_limits<int>::max();
    if (criteria.type & TermCriteria::MAX_ITER) {
        maxCount = criteria.maxCount;
    }


    for(size_t i = 0; i < n; i++ )
    {
        Vec2d pi = sdepth == CV_32F ? (Vec2d)srcf[i] : srcd[i];  // image point
        Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

        // the current camera model is only valid up to 180 FOV
        // for larger FOV the loop below does not converge
        // clip values so we still get plausible results for super fisheye images > 180 grad
        theta_d = min(max(-CV_PI/2., theta_d), CV_PI/2.);

        bool converged = false;
        double theta = theta_d;

        double scale = 0.0;

        if (!isEps || fabs(theta_d) > criteria.epsilon)
        {
            // compensate distortion iteratively using Newton method

            for (int j = 0; j < maxCount; j++)
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                double k0_theta2 = k[0] * theta2, k1_theta4 = k[1] * theta4, k2_theta6 = k[2] * theta6, k3_theta8 = k[3] * theta8;
                /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
                double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                   (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
                theta = theta - theta_fix;

                if (isEps && (fabs(theta_fix) < criteria.epsilon))
                {
                    converged = true;
                    break;
                }
            }

            scale = std::tan(theta) / theta_d;
        }
        else
        {
            converged = true;
        }

        // theta is monotonously increasing or decreasing depending on the sign of theta
        // if theta has flipped, it might converge due to symmetry but on the opposite of the camera center
        // so we can check whether theta has changed the sign during the optimization
        bool theta_flipped = ((theta_d < 0 && theta > 0) || (theta_d > 0 && theta < 0));

        if ((converged || !isEps) && !theta_flipped)
        {
            Vec2d pu = pw * scale; //undistorted point

            // reproject
            Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
            Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

            if( sdepth == CV_32F )
                dstf[i] = fi;
            else
                dstd[i] = fi;
        }
        else
        {
            // Vec2d fi(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
            Vec2d fi(-1000000.0, -1000000.0);

            if( sdepth == CV_32F )
                dstf[i] = fi;
            else
                dstd[i] = fi;
        }
    }
}

void cv::fisheye::initUndistortRectifyMap( InputArray K, InputArray D, InputArray R, InputArray P,
    const cv::Size& size, int m1type, OutputArray map1, OutputArray map2 )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( m1type == CV_16SC2 || m1type == CV_32F || m1type <=0 );
    map1.create( size, m1type <= 0 ? CV_16SC2 : m1type );
    map2.create( size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F );

    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert((P.empty() || P.depth() == CV_32F || P.depth() == CV_64F) && (R.empty() || R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(K.size() == Size(3, 3) && (D.empty() || D.total() == 4));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));

    Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
    }

    Vec4d k = Vec4d::all(0);
    if (!D.empty())
        k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    Matx33d RR  = Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        RR = Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    Matx33d PP = Matx33d::eye();
    if (!P.empty())
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);

    Matx33d iR = (PP * RR).inv(cv::DECOMP_SVD);

    for( int i = 0; i < size.height; ++i)
    {
        float* m1f = map1.getMat().ptr<float>(i);
        float* m2f = map2.getMat().ptr<float>(i);
        short*  m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;

        double _x = i*iR(0, 1) + iR(0, 2),
               _y = i*iR(1, 1) + iR(1, 2),
               _w = i*iR(2, 1) + iR(2, 2);

        for( int j = 0; j < size.width; ++j)
        {
            double u, v;
            if( _w <= 0)
            {
                u = (_x > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                v = (_y > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
            }
            else
            {
                double x = _x/_w, y = _y/_w;

                double r = sqrt(x*x + y*y);
                double theta = std::atan(r);

                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta4*theta4;
                double theta_d = theta * (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8);

                double scale = (r == 0) ? 1.0 : theta_d / r;
                u = f[0]*x*scale + c[0];
                v = f[1]*y*scale + c[1];
            }

            if( m1type == CV_16SC2 )
            {
                int iu = cv::saturate_cast<int>(u*static_cast<double>(cv::INTER_TAB_SIZE));
                int iv = cv::saturate_cast<int>(v*static_cast<double>(cv::INTER_TAB_SIZE));
                m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
            }
            else if( m1type == CV_32FC1 )
            {
                m1f[j] = (float)u;
                m2f[j] = (float)v;
            }

            _x += iR(0, 0);
            _y += iR(1, 0);
            _w += iR(2, 0);
        }
    }
}

void cv::fisheye::estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, const Size &image_size, InputArray R,
    OutputArray P, double balance, const Size& new_size, double fov_scale)
{
    CV_INSTRUMENT_REGION();

    CV_Assert( K.size() == Size(3, 3)       && (K.depth() == CV_32F || K.depth() == CV_64F));
    CV_Assert(D.empty() || ((D.total() == 4) && (D.depth() == CV_32F || D.depth() == CV_64F)));

    int w = image_size.width, h = image_size.height;
    balance = std::min(std::max(balance, 0.0), 1.0);

    Mat points(1, 4, CV_64FC2);
    Vec2d* pptr = points.ptr<Vec2d>();
    pptr[0] = Vec2d(w/2, 0);
    pptr[1] = Vec2d(w, h/2);
    pptr[2] = Vec2d(w/2, h);
    pptr[3] = Vec2d(0, h/2);

    fisheye::undistortPoints(points, points, K, D, R);
    cv::Scalar center_mass = mean(points);
    Vec2d cn(center_mass.val);

    double aspect_ratio = (K.depth() == CV_32F) ? K.getMat().at<float >(0,0)/K.getMat().at<float> (1,1)
                                                : K.getMat().at<double>(0,0)/K.getMat().at<double>(1,1);

    // convert to identity ratio
    cn[1] *= aspect_ratio;
    for(size_t i = 0; i < points.total(); ++i)
        pptr[i][1] *= aspect_ratio;

    double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
    for(size_t i = 0; i < points.total(); ++i)
    {
        miny = std::min(miny, pptr[i][1]);
        maxy = std::max(maxy, pptr[i][1]);
        minx = std::min(minx, pptr[i][0]);
        maxx = std::max(maxx, pptr[i][0]);
    }

    double f1 = w * 0.5/(cn[0] - minx);
    double f2 = w * 0.5/(maxx - cn[0]);
    double f3 = h * 0.5 * aspect_ratio/(cn[1] - miny);
    double f4 = h * 0.5 * aspect_ratio/(maxy - cn[1]);

    double fmin = std::min(f1, std::min(f2, std::min(f3, f4)));
    double fmax = std::max(f1, std::max(f2, std::max(f3, f4)));

    double f = balance * fmin + (1.0 - balance) * fmax;
    f *= fov_scale > 0 ? 1.0/fov_scale : 1.0;

    Vec2d new_f(f, f), new_c = -cn * f + Vec2d(w, h * aspect_ratio) * 0.5;

    // restore aspect ratio
    new_f[1] /= aspect_ratio;
    new_c[1] /= aspect_ratio;

    if (!new_size.empty())
    {
        double rx = new_size.width /(double)image_size.width;
        double ry = new_size.height/(double)image_size.height;

        new_f[0] *= rx;  new_f[1] *= ry;
        new_c[0] *= rx;  new_c[1] *= ry;
    }

    Mat(Matx33d(new_f[0], 0, new_c[0],
                0, new_f[1], new_c[1],
                0,        0,       1)).convertTo(P, P.empty() ? K.type() : P.type());
}

void cv::fisheye::undistortImage(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray Knew, const Size& new_size)
{
    CV_INSTRUMENT_REGION();

    Size size = !new_size.empty() ? new_size : distorted.size();

    Mat map1, map2;
    fisheye::initUndistortRectifyMap(K, D, Matx33d::eye(), Knew, size, CV_16SC2, map1, map2 );
    cv::remap(distorted, undistorted, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}

bool cv::fisheye::solvePnP( InputArray opoints, InputArray ipoints,
               InputArray cameraMatrix, InputArray distCoeffs,
               OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess,
               int flags, TermCriteria criteria)
{

    Mat imagePointsNormalized;
    cv::fisheye::undistortPoints(ipoints, imagePointsNormalized, cameraMatrix, distCoeffs, noArray(), cameraMatrix, criteria);
    return cv::solvePnP(opoints, imagePointsNormalized, cameraMatrix, noArray(), rvec, tvec, useExtrinsicGuess, flags);
}

} // namespace cv
