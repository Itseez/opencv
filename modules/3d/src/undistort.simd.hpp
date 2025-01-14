/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
Ptr<ParallelLoopBody> getInitUndistortRectifyMapComputer(Size _size, Mat &_map1, Mat &_map2, int _m1type,
                                                         const double* _ir, Matx33d &_matTilt,
                                                         double _u0, double _v0, double _fx, double _fy,
                                                         double _k1, double _k2, double _p1, double _p2,
                                                         double _k3, double _k4, double _k5, double _k6,
                                                         double _s1, double _s2, double _s3, double _s4);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
namespace
{
class initUndistortRectifyMapComputer : public ParallelLoopBody
{
public:
    initUndistortRectifyMapComputer(
        Size _size, Mat &_map1, Mat &_map2, int _m1type,
        const double* _ir, Matx33d &_matTilt,
        double _u0, double _v0, double _fx, double _fy,
        double _k1, double _k2, double _p1, double _p2,
        double _k3, double _k4, double _k5, double _k6,
        double _s1, double _s2, double _s3, double _s4)
      : size(_size),
        map1(_map1),
        map2(_map2),
        m1type(_m1type),
        ir(_ir),
        matTilt(_matTilt),
        u0(_u0),
        v0(_v0),
        fx(_fx),
        fy(_fy),
        k1(_k1),
        k2(_k2),
        p1(_p1),
        p2(_p2),
        k3(_k3),
        k4(_k4),
        k5(_k5),
        k6(_k6),
        s1(_s1),
        s2(_s2),
        s3(_s3),
        s4(_s4) {
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        for (int i = 0; i < 2 * VTraits<v_float64>::vlanes(); ++i)
        {
            s_x[i] = ir[0] * i;
            s_y[i] = ir[3] * i;
            s_w[i] = ir[6] * i;
        }
#endif
    }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        const int begin = range.start;
        const int end = range.end;

        for( int i = begin; i < end; i++ )
        {
            float* m1f = map1.ptr<float>(i);
            float* m2f = map2.empty() ? 0 : map2.ptr<float>(i);
            short* m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;
            double _x = i*ir[1] + ir[2], _y = i*ir[4] + ir[5], _w = i*ir[7] + ir[8];

            int j = 0;

            if (m1type == CV_16SC2)
                CV_Assert(m1 != NULL && m2 != NULL);
            else if (m1type == CV_32FC1)
                CV_Assert(m1f != NULL && m2f != NULL);
            else
                CV_Assert(m1 != NULL);

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
            const v_float64 v_one = vx_setall_f64(1.0);
            for (; j <= size.width - 2*VTraits<v_float64>::vlanes(); j += 2*VTraits<v_float64>::vlanes(), _x += 2*VTraits<v_float64>::vlanes() * ir[0], _y += 2*VTraits<v_float64>::vlanes() * ir[3], _w += 2*VTraits<v_float64>::vlanes() * ir[6])
            {
                v_float64 m_0, m_1, m_2, m_3;
                m_2 = v_div(v_one, v_add(vx_setall_f64(_w), vx_load(this->s_w)));
                m_3 = v_div(v_one, v_add(vx_setall_f64(_w), vx_load(this->s_w + VTraits<v_float64>::vlanes())));
                m_0 = vx_setall_f64(_x); m_1 = vx_setall_f64(_y);
                v_float64 x_0 = v_mul(v_add(m_0, vx_load(this->s_x)), m_2);
                v_float64 x_1 = v_mul(v_add(m_0, vx_load(this->s_x + VTraits<v_float64>::vlanes())), m_3);
                v_float64 y_0 = v_mul(v_add(m_1, vx_load(this->s_y)), m_2);
                v_float64 y_1 = v_mul(v_add(m_1, vx_load(this->s_y + VTraits<v_float64>::vlanes())), m_3);

                v_float64 xd_0 = v_mul(x_0, x_0);
                v_float64 yd_0 = v_mul(y_0, y_0);
                v_float64 xd_1 = v_mul(x_1, x_1);
                v_float64 yd_1 = v_mul(y_1, y_1);

                v_float64 r2_0 = v_add(xd_0, yd_0);
                v_float64 r2_1 = v_add(xd_1, yd_1);

                m_1 = vx_setall_f64(k3);
                m_2 = vx_setall_f64(k2);
                m_3 = vx_setall_f64(k1);
                m_0 = v_muladd(v_muladd(v_muladd(m_1, r2_0, m_2), r2_0, m_3), r2_0, v_one);
                m_1 = v_muladd(v_muladd(v_muladd(m_1, r2_1, m_2), r2_1, m_3), r2_1, v_one);
                m_3 = vx_setall_f64(k6);
                m_2 = vx_setall_f64(k5);
                m_0 = v_div(m_0, v_muladd(v_muladd(v_muladd(m_3, r2_0, m_2), r2_0, vx_setall_f64(this->k4)), r2_0, v_one));
                m_1 = v_div(m_1, v_muladd(v_muladd(v_muladd(m_3, r2_1, m_2), r2_1, vx_setall_f64(this->k4)), r2_1, v_one));

                m_3 = vx_setall_f64(2.0);
                xd_0 = v_muladd(m_3, xd_0, r2_0);
                yd_0 = v_muladd(m_3, yd_0, r2_0);
                xd_1 = v_muladd(m_3, xd_1, r2_1);
                yd_1 = v_muladd(m_3, yd_1, r2_1);
                m_2 = v_mul(v_mul(x_0, y_0), m_3);
                m_3 = v_mul(v_mul(x_1, y_1), m_3);

                x_0 = v_mul(x_0, m_0); y_0 = v_mul(y_0, m_0); x_1 = v_mul(x_1, m_1); y_1 = v_mul(y_1, m_1);

                m_0 = vx_setall_f64(p1);
                m_1 = vx_setall_f64(p2);
                xd_0 = v_muladd(xd_0, m_1, x_0);
                yd_0 = v_muladd(yd_0, m_0, y_0);
                xd_1 = v_muladd(xd_1, m_1, x_1);
                yd_1 = v_muladd(yd_1, m_0, y_1);

                xd_0 = v_muladd(m_0, m_2, xd_0);
                yd_0 = v_muladd(m_1, m_2, yd_0);
                xd_1 = v_muladd(m_0, m_3, xd_1);
                yd_1 = v_muladd(m_1, m_3, yd_1);

                m_0 = v_mul(r2_0, r2_0);
                m_1 = v_mul(r2_1, r2_1);
                m_2 = vx_setall_f64(s2);
                m_3 = vx_setall_f64(s1);
                xd_0 = v_muladd(m_3, r2_0, v_muladd(m_2, m_0, xd_0));
                xd_1 = v_muladd(m_3, r2_1, v_muladd(m_2, m_1, xd_1));
                m_2 = vx_setall_f64(s4);
                m_3 = vx_setall_f64(s3);
                yd_0 = v_muladd(m_3, r2_0, v_muladd(m_2, m_0, yd_0));
                yd_1 = v_muladd(m_3, r2_1, v_muladd(m_2, m_1, yd_1));

                m_0 = vx_setall_f64(matTilt.val[0]);
                m_1 = vx_setall_f64(matTilt.val[1]);
                m_2 = vx_setall_f64(matTilt.val[2]);
                x_0 = v_muladd(m_0, xd_0, v_muladd(m_1, yd_0, m_2));
                x_1 = v_muladd(m_0, xd_1, v_muladd(m_1, yd_1, m_2));
                m_0 = vx_setall_f64(matTilt.val[3]);
                m_1 = vx_setall_f64(matTilt.val[4]);
                m_2 = vx_setall_f64(matTilt.val[5]);
                y_0 = v_muladd(m_0, xd_0, v_muladd(m_1, yd_0, m_2));
                y_1 = v_muladd(m_0, xd_1, v_muladd(m_1, yd_1, m_2));
                m_0 = vx_setall_f64(matTilt.val[6]);
                m_1 = vx_setall_f64(matTilt.val[7]);
                m_2 = vx_setall_f64(matTilt.val[8]);
                r2_0 = v_muladd(m_0, xd_0, v_muladd(m_1, yd_0, m_2));
                r2_1 = v_muladd(m_0, xd_1, v_muladd(m_1, yd_1, m_2));
                m_0 = vx_setzero_f64();
                r2_0 = v_select(v_eq(r2_0, m_0), v_one, v_div(v_one, r2_0));
                r2_1 = v_select(v_eq(r2_1, m_0), v_one, v_div(v_one, r2_1));

                m_0 = vx_setall_f64(fx);
                m_1 = vx_setall_f64(u0);
                m_2 = vx_setall_f64(fy);
                m_3 = vx_setall_f64(v0);
                x_0 = v_muladd(v_mul(m_0, r2_0), x_0, m_1);
                y_0 = v_muladd(v_mul(m_2, r2_0), y_0, m_3);
                x_1 = v_muladd(v_mul(m_0, r2_1), x_1, m_1);
                y_1 = v_muladd(v_mul(m_2, r2_1), y_1, m_3);

                if (m1type == CV_32FC1)
                {
                    v_store(&m1f[j], v_cvt_f32(x_0, x_1));
                    v_store(&m2f[j], v_cvt_f32(y_0, y_1));
                }
                else if (m1type == CV_32FC2)
                {
                    v_float32 mf0, mf1;
                    v_zip(v_cvt_f32(x_0, x_1), v_cvt_f32(y_0, y_1), mf0, mf1);
                    v_store(&m1f[j * 2], mf0);
                    v_store(&m1f[j * 2 + VTraits<v_float32>::vlanes()], mf1);
                }
                else // m1type == CV_16SC2
                {
                    m_0 = vx_setall_f64(INTER_TAB_SIZE);
                    x_0 = v_mul(x_0, m_0); x_1 = v_mul(x_1, m_0); y_0 = v_mul(y_0, m_0); y_1 = v_mul(y_1, m_0);

                    v_int32 mask = vx_setall_s32(INTER_TAB_SIZE - 1);
                    v_int32 iu = v_round(x_0, x_1);
                    v_int32 iv = v_round(y_0, y_1);

                    v_pack_u_store(&m2[j], v_add(v_and(iu, mask), v_mul(v_and(iv, mask), vx_setall_s32(INTER_TAB_SIZE))));
                    v_int32 out0, out1;
                    v_zip(v_shr<INTER_BITS>(iu), v_shr<INTER_BITS>(iv), out0, out1);
                    v_store(&m1[j * 2], v_pack(out0, out1));
                }
            }

            vx_cleanup();
#endif
            for( ; j < size.width; j++, _x += ir[0], _y += ir[3], _w += ir[6] )
            {
                double w = 1./_w, x = _x*w, y = _y*w;
                double x2 = x*x, y2 = y*y;
                double r2 = x2 + y2, _2xy = 2*x*y;
                double kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2)/(1 + ((k6*r2 + k5)*r2 + k4)*r2);
                double xd = (x*kr + p1*_2xy + p2*(r2 + 2*x2) + s1*r2+s2*r2*r2);
                double yd = (y*kr + p1*(r2 + 2*y2) + p2*_2xy + s3*r2+s4*r2*r2);
                Vec3d vecTilt = matTilt*cv::Vec3d(xd, yd, 1);
                double invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
                double u = fx*invProj*vecTilt(0) + u0;
                double v = fy*invProj*vecTilt(1) + v0;
                if( m1type == CV_16SC2 )
                {
                    int iu = saturate_cast<int>(u*static_cast<double>(INTER_TAB_SIZE));
                    int iv = saturate_cast<int>(v*static_cast<double>(INTER_TAB_SIZE));
                    m1[j*2] = (short)(iu >> INTER_BITS);
                    m1[j*2+1] = (short)(iv >> INTER_BITS);
                    m2[j] = (ushort)((iv & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE-1)));
                }
                else if( m1type == CV_32FC1 )
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }
                else
                {
                    m1f[j*2] = (float)u;
                    m1f[j*2+1] = (float)v;
                }
            }
        }
    }

private:
    Size size;
    Mat &map1;
    Mat &map2;
    int m1type;
    const double* ir;
    Matx33d &matTilt;
    double u0;
    double v0;
    double fx;
    double fy;
    double k1;
    double k2;
    double p1;
    double p2;
    double k3;
    double k4;
    double k5;
    double k6;
    double s1;
    double s2;
    double s3;
    double s4;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    double s_x[2*VTraits<v_float64>::max_nlanes];
    double s_y[2*VTraits<v_float64>::max_nlanes];
    double s_w[2*VTraits<v_float64>::max_nlanes];
#endif
};
}

Ptr<ParallelLoopBody> getInitUndistortRectifyMapComputer(Size _size, Mat &_map1, Mat &_map2, int _m1type,
                                                         const double* _ir, Matx33d &_matTilt,
                                                         double _u0, double _v0, double _fx, double _fy,
                                                         double _k1, double _k2, double _p1, double _p2,
                                                         double _k3, double _k4, double _k5, double _k6,
                                                         double _s1, double _s2, double _s3, double _s4)
{
    CV_INSTRUMENT_REGION();

    return Ptr<initUndistortRectifyMapComputer>(new initUndistortRectifyMapComputer(_size, _map1, _map2, _m1type, _ir, _matTilt, _u0, _v0, _fx, _fy,
                                                                                    _k1, _k2, _p1, _p2, _k3, _k4, _k5, _k6, _s1, _s2, _s3, _s4));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
}
/*  End of file  */
