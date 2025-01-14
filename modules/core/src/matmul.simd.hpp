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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#ifdef HAVE_LAPACK
#define CV_GEMM_BASELINE_ONLY
#endif
#define CV_MAHALANOBIS_BASELINE_ONLY
#define CV_MULTRANSPOSED_BASELINE_ONLY

namespace cv {

// forward declarations
typedef void (*TransformFunc)(const uchar* src, uchar* dst, const uchar* m, int len, int scn, int dcn);
typedef void (*ScaleAddFunc)(const uchar* src1, const uchar* src2, uchar* dst, int len, const void* alpha);
typedef void (*MulTransposedFunc)(const Mat& src, const/*preallocated*/ Mat& dst, const Mat& delta, double scale);
typedef double (*MahalanobisImplFunc)(const Mat& v1, const Mat& v2, const Mat& icovar, double *diff_buffer /*[len]*/, int len /*=v1.total()*/);

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// forward declarations

void gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
             float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags);
void gemm64f(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
             double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags);
void gemm32fc(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
              float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
              int m_a, int n_a, int n_d, int flags);
void gemm64fc(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
              double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
              int m_a, int n_a, int n_d, int flags);

TransformFunc getTransformFunc(int depth);
TransformFunc getDiagTransformFunc(int depth);
TransformFunc getPerspectiveTransform(int depth);

ScaleAddFunc getScaleAddFunc(int depth);

MahalanobisImplFunc getMahalanobisImplFunc(int depth);

MulTransposedFunc getMulTransposedFunc(int stype, int dtype, bool ata);

double dotProd_8u(const uchar* src1, const uchar* src2, int len);
double dotProd_8s(const schar* src1, const schar* src2, int len);
double dotProd_16u(const ushort* src1, const ushort* src2, int len);
double dotProd_16s(const short* src1, const short* src2, int len);
double dotProd_32s(const int* src1, const int* src2, int len);
double dotProd_32f(const float* src1, const float* src2, int len);
double dotProd_64f(const double* src1, const double* src2, int len);



#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if !defined(CV_GEMM_BASELINE_ONLY) || defined(CV_CPU_BASELINE_MODE)
/****************************************************************************************\
*                                         GEMM                                           *
\****************************************************************************************/

static void
GEMM_CopyBlock( const uchar* src, size_t src_step,
                uchar* dst, size_t dst_step,
                Size size, size_t pix_size )
{
    int j;
    size.width *= (int)(pix_size / sizeof(int));

    for( ; size.height--; src += src_step, dst += dst_step )
    {
        j=0;
         #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int t0 = ((const int*)src)[j];
            int t1 = ((const int*)src)[j+1];
            ((int*)dst)[j] = t0;
            ((int*)dst)[j+1] = t1;
            t0 = ((const int*)src)[j+2];
            t1 = ((const int*)src)[j+3];
            ((int*)dst)[j+2] = t0;
            ((int*)dst)[j+3] = t1;
        }
        #endif
        for( ; j < size.width; j++ )
            ((int*)dst)[j] = ((const int*)src)[j];
    }
}


static void
GEMM_TransposeBlock( const uchar* src, size_t src_step,
                     uchar* dst, size_t dst_step,
                     Size size, size_t pix_size )
{
    int i, j;
    for( i = 0; i < size.width; i++, dst += dst_step, src += pix_size )
    {
        const uchar* _src = src;
        switch( pix_size )
        {
        case sizeof(int):
            for( j = 0; j < size.height; j++, _src += src_step )
                ((int*)dst)[j] = ((int*)_src)[0];
            break;
        case sizeof(int)*2:
            for( j = 0; j < size.height*2; j += 2, _src += src_step )
            {
                int t0 = ((int*)_src)[0];
                int t1 = ((int*)_src)[1];
                ((int*)dst)[j] = t0;
                ((int*)dst)[j+1] = t1;
            }
            break;
        case sizeof(int)*4:
            for( j = 0; j < size.height*4; j += 4, _src += src_step )
            {
                int t0 = ((int*)_src)[0];
                int t1 = ((int*)_src)[1];
                ((int*)dst)[j] = t0;
                ((int*)dst)[j+1] = t1;
                t0 = ((int*)_src)[2];
                t1 = ((int*)_src)[3];
                ((int*)dst)[j+2] = t0;
                ((int*)dst)[j+3] = t1;
            }
            break;
        default:
            CV_Assert(0);
            return;
        }
    }
}


template<typename T, typename WT> static void
GEMMSingleMul( const T* a_data, size_t a_step,
               const T* b_data, size_t b_step,
               const T* c_data, size_t c_step,
               T* d_data, size_t d_step,
               Size a_size, Size d_size,
               double alpha, double beta, int flags )
{
    int i, j, k, n = a_size.width, m = d_size.width, drows = d_size.height;
    const T *_a_data = a_data, *_b_data = b_data, *_c_data = c_data;
    cv::AutoBuffer<T> _a_buf;
    T* a_buf = 0;
    size_t a_step0, a_step1, c_step0, c_step1, t_step;

    a_step /= sizeof(a_data[0]);
    b_step /= sizeof(b_data[0]);
    c_step /= sizeof(c_data[0]);
    d_step /= sizeof(d_data[0]);
    a_step0 = a_step;
    a_step1 = 1;

    if( !c_data )
        c_step0 = c_step1 = 0;
    else if( !(flags & GEMM_3_T) )
        c_step0 = c_step, c_step1 = 1;
    else
        c_step0 = 1, c_step1 = c_step;

    if( flags & GEMM_1_T )
    {
        CV_SWAP( a_step0, a_step1, t_step );
        n = a_size.height;
        if( a_step > 1 && n > 1 )
        {
            _a_buf.allocate(n);
            a_buf = _a_buf.data();
        }
    }

    if( n == 1 ) /* external product */
    {
        cv::AutoBuffer<T> _b_buf;
        T* b_buf = 0;

        if( a_step > 1 && a_size.height > 1 )
        {
            _a_buf.allocate(drows);
            a_buf = _a_buf.data();
            for( k = 0; k < drows; k++ )
                a_buf[k] = a_data[a_step*k];
            a_data = a_buf;
        }

        if( b_step > 1 )
        {
            _b_buf.allocate(d_size.width);
            b_buf = _b_buf.data();
            for( j = 0; j < d_size.width; j++ )
                b_buf[j] = b_data[j*b_step];
            b_data = b_buf;
        }

        for( i = 0; i < drows; i++, _c_data += c_step0, d_data += d_step )
        {
            WT al = WT(a_data[i])*alpha;
            c_data = _c_data;
            for( j = 0; j <= d_size.width - 2; j += 2, c_data += 2*c_step1 )
            {
                WT s0 = al*WT(b_data[j]);
                WT s1 = al*WT(b_data[j+1]);
                if( !c_data )
                {
                    d_data[j] = T(s0);
                    d_data[j+1] = T(s1);
                }
                else
                {
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
                    d_data[j+1] = T(s1 + WT(c_data[c_step1])*beta);
                }
            }

            for( ; j < d_size.width; j++, c_data += c_step1 )
            {
                WT s0 = al*WT(b_data[j]);
                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else if( flags & GEMM_2_T ) /* A * Bt */
    {
        for( i = 0; i < drows; i++, _a_data += a_step0, _c_data += c_step0, d_data += d_step )
        {
            a_data = _a_data;
            b_data = _b_data;
            c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < d_size.width; j++, b_data += b_step,
                                               c_data += c_step1 )
            {
                WT s0(0), s1(0), s2(0), s3(0);
                k = 0;
                 #if CV_ENABLE_UNROLLED
                for( ; k <= n - 4; k += 4 )
                {
                    s0 += WT(a_data[k])*WT(b_data[k]);
                    s1 += WT(a_data[k+1])*WT(b_data[k+1]);
                    s2 += WT(a_data[k+2])*WT(b_data[k+2]);
                    s3 += WT(a_data[k+3])*WT(b_data[k+3]);
                }
                #endif
                for( ; k < n; k++ )
                    s0 += WT(a_data[k])*WT(b_data[k]);
                s0 = (s0+s1+s2+s3)*alpha;

                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else if( d_size.width*sizeof(d_data[0]) <= 1600 )
    {
        for( i = 0; i < drows; i++, _a_data += a_step0,
                                    _c_data += c_step0,
                                    d_data += d_step )
        {
            a_data = _a_data, c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j <= m - 4; j += 4, c_data += 4*c_step1 )
            {
                const T* b = _b_data + j;
                WT s0(0), s1(0), s2(0), s3(0);

                for( k = 0; k < n; k++, b += b_step )
                {
                    WT a(a_data[k]);
                    s0 += a * WT(b[0]); s1 += a * WT(b[1]);
                    s2 += a * WT(b[2]); s3 += a * WT(b[3]);
                }

                if( !c_data )
                {
                    d_data[j] = T(s0*alpha);
                    d_data[j+1] = T(s1*alpha);
                    d_data[j+2] = T(s2*alpha);
                    d_data[j+3] = T(s3*alpha);
                }
                else
                {
                    s0 = s0*alpha; s1 = s1*alpha;
                    s2 = s2*alpha; s3 = s3*alpha;
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
                    d_data[j+1] = T(s1 + WT(c_data[c_step1])*beta);
                    d_data[j+2] = T(s2 + WT(c_data[c_step1*2])*beta);
                    d_data[j+3] = T(s3 + WT(c_data[c_step1*3])*beta);
                }
            }

            for( ; j < m; j++, c_data += c_step1 )
            {
                const T* b = _b_data + j;
                WT s0(0);

                for( k = 0; k < n; k++, b += b_step )
                    s0 += WT(a_data[k]) * WT(b[0]);

                s0 = s0*alpha;
                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else
    {
        cv::AutoBuffer<WT> _d_buf(m);
        WT* d_buf = _d_buf.data();

        for( i = 0; i < drows; i++, _a_data += a_step0, _c_data += c_step0, d_data += d_step )
        {
            a_data = _a_data;
            b_data = _b_data;
            c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = _a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < m; j++ )
                d_buf[j] = WT(0);

            for( k = 0; k < n; k++, b_data += b_step )
            {
                WT al(a_data[k]);
                j=0;
                for( ; j < m; j++ )
                    d_buf[j] += WT(b_data[j])*al;
            }

            if( !c_data )
                for( j = 0; j < m; j++ )
                    d_data[j] = T(d_buf[j]*alpha);
            else
                for( j = 0; j < m; j++, c_data += c_step1 )
                {
                    WT t = d_buf[j]*alpha;
                    d_data[j] = T(t + WT(c_data[0])*beta);
                }
        }
    }
}


template<typename T, typename WT> static void
GEMMBlockMul( const T* a_data, size_t a_step,
              const T* b_data, size_t b_step,
              WT* d_data, size_t d_step,
              Size a_size, Size d_size, int flags )
{
    int i, j, k, n = a_size.width, m = d_size.width;
    const T *_a_data = a_data, *_b_data = b_data;
    cv::AutoBuffer<T> _a_buf;
    T* a_buf = 0;
    size_t a_step0, a_step1, t_step;
    int do_acc = flags & 16;

    a_step /= sizeof(a_data[0]);
    b_step /= sizeof(b_data[0]);
    d_step /= sizeof(d_data[0]);

    a_step0 = a_step;
    a_step1 = 1;

    if( flags & GEMM_1_T )
    {
        CV_SWAP( a_step0, a_step1, t_step );
        n = a_size.height;
        _a_buf.allocate(n);
        a_buf = _a_buf.data();
    }

    if( flags & GEMM_2_T )
    {
        /* second operand is transposed */
        for( i = 0; i < d_size.height; i++, _a_data += a_step0, d_data += d_step )
        {
            a_data = _a_data; b_data = _b_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < d_size.width; j++, b_data += b_step )
            {
                WT s0 = do_acc ? d_data[j]:WT(0), s1(0);
                for( k = 0; k <= n - 2; k += 2 )
                {
                    s0 += WT(a_data[k])*WT(b_data[k]);
                    s1 += WT(a_data[k+1])*WT(b_data[k+1]);
                }

                for( ; k < n; k++ )
                    s0 += WT(a_data[k])*WT(b_data[k]);

                d_data[j] = s0 + s1;
            }
        }
    }
    else
    {
        for( i = 0; i < d_size.height; i++, _a_data += a_step0, d_data += d_step )
        {
            a_data = _a_data, b_data = _b_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j <= m - 4; j += 4 )
            {
                WT s0, s1, s2, s3;
                const T* b = b_data + j;

                if( do_acc )
                {
                    s0 = d_data[j]; s1 = d_data[j+1];
                    s2 = d_data[j+2]; s3 = d_data[j+3];
                }
                else
                    s0 = s1 = s2 = s3 = WT(0);

                for( k = 0; k < n; k++, b += b_step )
                {
                    WT a(a_data[k]);
                    s0 += a * WT(b[0]); s1 += a * WT(b[1]);
                    s2 += a * WT(b[2]); s3 += a * WT(b[3]);
                }

                d_data[j] = s0; d_data[j+1] = s1;
                d_data[j+2] = s2; d_data[j+3] = s3;
            }

            for( ; j < m; j++ )
            {
                const T* b = b_data + j;
                WT s0 = do_acc ? d_data[j] : WT(0);

                for( k = 0; k < n; k++, b += b_step )
                    s0 += WT(a_data[k]) * WT(b[0]);

                d_data[j] = s0;
            }
        }
    }
}


template<typename T, typename WT> static void
GEMMStore( const T* c_data, size_t c_step,
           const WT* d_buf, size_t d_buf_step,
           T* d_data, size_t d_step, Size d_size,
           double alpha, double beta, int flags )
{
    const T* _c_data = c_data;
    int j;
    size_t c_step0, c_step1;

    c_step /= sizeof(c_data[0]);
    d_buf_step /= sizeof(d_buf[0]);
    d_step /= sizeof(d_data[0]);

    if( !c_data )
        c_step0 = c_step1 = 0;
    else if( !(flags & GEMM_3_T) )
        c_step0 = c_step, c_step1 = 1;
    else
        c_step0 = 1, c_step1 = c_step;

    for( ; d_size.height--; _c_data += c_step0, d_buf += d_buf_step, d_data += d_step )
    {
        if( _c_data )
        {
            c_data = _c_data;
            j=0;
             #if CV_ENABLE_UNROLLED
            for(; j <= d_size.width - 4; j += 4, c_data += 4*c_step1 )
            {
                WT t0 = alpha*d_buf[j];
                WT t1 = alpha*d_buf[j+1];
                t0 += beta*WT(c_data[0]);
                t1 += beta*WT(c_data[c_step1]);
                d_data[j] = T(t0);
                d_data[j+1] = T(t1);
                t0 = alpha*d_buf[j+2];
                t1 = alpha*d_buf[j+3];
                t0 += beta*WT(c_data[c_step1*2]);
                t1 += beta*WT(c_data[c_step1*3]);
                d_data[j+2] = T(t0);
                d_data[j+3] = T(t1);
            }
            #endif
            for( ; j < d_size.width; j++, c_data += c_step1 )
            {
                WT t0 = alpha*d_buf[j];
                d_data[j] = T(t0 + WT(c_data[0])*beta);
            }
        }
        else
        {
            j = 0;
             #if CV_ENABLE_UNROLLED
            for( ; j <= d_size.width - 4; j += 4 )
            {
                WT t0 = alpha*d_buf[j];
                WT t1 = alpha*d_buf[j+1];
                d_data[j] = T(t0);
                d_data[j+1] = T(t1);
                t0 = alpha*d_buf[j+2];
                t1 = alpha*d_buf[j+3];
                d_data[j+2] = T(t0);
                d_data[j+3] = T(t1);
            }
            #endif
            for( ; j < d_size.width; j++ )
                d_data[j] = T(alpha*d_buf[j]);
        }
    }
}


typedef void (*GEMMSingleMulFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, const void* src3, size_t step3,
                   void* dst, size_t dststep, Size srcsize, Size dstsize,
                   double alpha, double beta, int flags );

typedef void (*GEMMBlockMulFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, void* dst, size_t dststep,
                   Size srcsize, Size dstsize, int flags );

typedef void (*GEMMStoreFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, void* dst, size_t dststep,
                   Size dstsize, double alpha, double beta, int flags );

static void GEMMSingleMul_32f( const float* a_data, size_t a_step,
              const float* b_data, size_t b_step,
              const float* c_data, size_t c_step,
              float* d_data, size_t d_step,
              Size a_size, Size d_size,
              double alpha, double beta, int flags )
{
    GEMMSingleMul<float,double>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}

static void GEMMSingleMul_64f( const double* a_data, size_t a_step,
                              const double* b_data, size_t b_step,
                              const double* c_data, size_t c_step,
                              double* d_data, size_t d_step,
                              Size a_size, Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<double,double>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}


static void GEMMSingleMul_32fc( const Complexf* a_data, size_t a_step,
                              const Complexf* b_data, size_t b_step,
                              const Complexf* c_data, size_t c_step,
                              Complexf* d_data, size_t d_step,
                              Size a_size, Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<Complexf,Complexd>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}

static void GEMMSingleMul_64fc( const Complexd* a_data, size_t a_step,
                              const Complexd* b_data, size_t b_step,
                              const Complexd* c_data, size_t c_step,
                              Complexd* d_data, size_t d_step,
                              Size a_size, Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<Complexd,Complexd>(a_data, a_step, b_data, b_step, c_data,
                                 c_step, d_data, d_step, a_size, d_size,
                                 alpha, beta, flags);
}

static void GEMMBlockMul_32f( const float* a_data, size_t a_step,
             const float* b_data, size_t b_step,
             double* d_data, size_t d_step,
             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_64f( const double* a_data, size_t a_step,
                             const double* b_data, size_t b_step,
                             double* d_data, size_t d_step,
                             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_32fc( const Complexf* a_data, size_t a_step,
                             const Complexf* b_data, size_t b_step,
                             Complexd* d_data, size_t d_step,
                             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_64fc( const Complexd* a_data, size_t a_step,
                             const Complexd* b_data, size_t b_step,
                             Complexd* d_data, size_t d_step,
                             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMStore_32f( const float* c_data, size_t c_step,
          const double* d_buf, size_t d_buf_step,
          float* d_data, size_t d_step, Size d_size,
          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_64f( const double* c_data, size_t c_step,
                      const double* d_buf, size_t d_buf_step,
                      double* d_data, size_t d_step, Size d_size,
                      double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_32fc( const Complexf* c_data, size_t c_step,
                          const Complexd* d_buf, size_t d_buf_step,
                          Complexf* d_data, size_t d_step, Size d_size,
                          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_64fc( const Complexd* c_data, size_t c_step,
                          const Complexd* d_buf, size_t d_buf_step,
                          Complexd* d_data, size_t d_step, Size d_size,
                          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}

static void gemmImpl( Mat A, Mat B, double alpha,
           Mat C, double beta, Mat D, int flags )
{
    CV_INSTRUMENT_REGION();

    const int block_lin_size = 128;
    const int block_size = block_lin_size * block_lin_size;

    static double zero[] = {0,0,0,0};
    static float zerof[] = {0,0,0,0};

    Size a_size = A.size(), d_size;
    int i, len = 0, type = A.type();

    switch( flags & (GEMM_1_T|GEMM_2_T) )
    {
    case 0:
        d_size = Size( B.cols, a_size.height );
        len = B.rows;
        break;
    case 1:
        d_size = Size( B.cols, a_size.width );
        len = B.rows;
        break;
    case 2:
        d_size = Size( B.rows, a_size.height );
        len = B.cols;
        break;
    case 3:
        d_size = Size( B.rows, a_size.width );
        len = B.cols;
        break;
    }

    if( flags == 0 && 2 <= len && len <= 4 && (len == d_size.width || len == d_size.height) )
    {
        if( type == CV_32F )
        {
            float* d = D.ptr<float>();
            const float *a = A.ptr<float>(),
                        *b = B.ptr<float>(),
                        *c = (const float*)C.data;
            size_t d_step = D.step/sizeof(d[0]),
                a_step = A.step/sizeof(a[0]),
                b_step = B.step/sizeof(b[0]),
                c_step = C.data ? C.step/sizeof(c[0]) : 0;

            if( !c )
                c = zerof;

            switch( len )
            {
            case 2:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                    }
                }
                else
                    break;
                return;
            case 3:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1];
                        float t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                        d[2] = (float)(t2*alpha + c[2]*beta);
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] + a[a_step+2]*b[b_step*2];
                        float t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] + a[a_step*2+2]*b[b_step*2];

                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                        d[d_step*2] = (float)(t2*alpha + c[c_step*2]*beta);
                    }
                }
                else
                    break;
                return;
            case 4:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1] + a[3]*b[b_step*3+1];
                        float t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2] + a[3]*b[b_step*3+2];
                        float t3 = a[0]*b[3] + a[1]*b[b_step+3] + a[2]*b[b_step*2+3] + a[3]*b[b_step*3+3];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                        d[2] = (float)(t2*alpha + c[2]*beta);
                        d[3] = (float)(t3*alpha + c[3]*beta);
                    }
                }
                else if( len <= 16 && a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] +
                                   a[a_step+2]*b[b_step*2] + a[a_step+3]*b[b_step*3];
                        float t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] +
                                   a[a_step*2+2]*b[b_step*2] + a[a_step*2+3]*b[b_step*3];
                        float t3 = a[a_step*3]*b[0] + a[a_step*3+1]*b[b_step] +
                                   a[a_step*3+2]*b[b_step*2] + a[a_step*3+3]*b[b_step*3];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                        d[d_step*2] = (float)(t2*alpha + c[c_step*2]*beta);
                        d[d_step*3] = (float)(t3*alpha + c[c_step*3]*beta);
                    }
                }
                else
                    break;
                return;
            }
        }

        if( type == CV_64F )
        {
            double* d = D.ptr<double>();
            const double *a = A.ptr<double>(),
                         *b = B.ptr<double>(),
                         *c = (const double*)C.data;
            size_t d_step = D.step/sizeof(d[0]),
                a_step = A.step/sizeof(a[0]),
                b_step = B.step/sizeof(b[0]),
                c_step = C.data ? C.step/sizeof(c[0]) : 0;
            if( !c )
                c = zero;

            switch( len )
            {
            case 2:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step];
                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                    }
                }
                else
                    break;
                return;
            case 3:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1];
                        double t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                        d[2] = t2*alpha + c[2]*beta;
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] + a[a_step+2]*b[b_step*2];
                        double t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] + a[a_step*2+2]*b[b_step*2];

                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                        d[d_step*2] = t2*alpha + c[c_step*2]*beta;
                    }
                }
                else
                    break;
                return;
            case 4:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1] + a[3]*b[b_step*3+1];
                        double t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2] + a[3]*b[b_step*3+2];
                        double t3 = a[0]*b[3] + a[1]*b[b_step+3] + a[2]*b[b_step*2+3] + a[3]*b[b_step*3+3];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                        d[2] = t2*alpha + c[2]*beta;
                        d[3] = t3*alpha + c[3]*beta;
                    }
                }
                else if( d_size.width <= 16 && a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] +
                                    a[a_step+2]*b[b_step*2] + a[a_step+3]*b[b_step*3];
                        double t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] +
                                    a[a_step*2+2]*b[b_step*2] + a[a_step*2+3]*b[b_step*3];
                        double t3 = a[a_step*3]*b[0] + a[a_step*3+1]*b[b_step] +
                                    a[a_step*3+2]*b[b_step*2] + a[a_step*3+3]*b[b_step*3];
                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                        d[d_step*2] = t2*alpha + c[c_step*2]*beta;
                        d[d_step*3] = t3*alpha + c[c_step*3]*beta;
                    }
                }
                else
                    break;
                return;
            }
        }
    }

    {
    size_t b_step = B.step;
    GEMMSingleMulFunc singleMulFunc;
    GEMMBlockMulFunc blockMulFunc;
    GEMMStoreFunc storeFunc;
    Mat *matD = &D;
    const uchar* Cdata = C.data;
    size_t Cstep = C.data ? (size_t)C.step : 0;
    AutoBuffer<uchar> buf;

    if( type == CV_32FC1 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_32f;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_32f;
        storeFunc = (GEMMStoreFunc)GEMMStore_32f;
    }
    else if( type == CV_64FC1 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_64f;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_64f;
        storeFunc = (GEMMStoreFunc)GEMMStore_64f;
    }
    else if( type == CV_32FC2 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_32fc;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_32fc;
        storeFunc = (GEMMStoreFunc)GEMMStore_32fc;
    }
    else
    {
        CV_Assert( type == CV_64FC2 );
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_64fc;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_64fc;
        storeFunc = (GEMMStoreFunc)GEMMStore_64fc;
    }

    if( (d_size.width == 1 || len == 1) && !(flags & GEMM_2_T) && B.isContinuous() )
    {
        b_step = d_size.width == 1 ? 0 : CV_ELEM_SIZE(type);
        flags |= GEMM_2_T;
    }

    /*if( (d_size.width | d_size.height | len) >= 16 && icvBLAS_GEMM_32f_p != 0 )
    {
        blas_func = type == CV_32FC1 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_32f_p :
                    type == CV_64FC1 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_64f_p :
                    type == CV_32FC2 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_32fc_p :
                    type == CV_64FC2 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_64fc_p : 0;
    }

    if( blas_func )
    {
        const char* transa = flags & GEMM_1_T ? "t" : "n";
        const char* transb = flags & GEMM_2_T ? "t" : "n";
        int lda, ldb, ldd;

        if( C->data.ptr )
        {
            if( C->data.ptr != D->data.ptr )
            {
                if( !(flags & GEMM_3_T) )
                    cvCopy( C, D );
                else
                    cvTranspose( C, D );
            }
        }

        if( CV_MAT_DEPTH(type) == CV_32F )
        {
            Complex32f _alpha, _beta;

            lda = A->step/sizeof(float);
            ldb = b_step/sizeof(float);
            ldd = D->step/sizeof(float);
            _alpha.re = (float)alpha;
            _alpha.im = 0;
            _beta.re = C->data.ptr ? (float)beta : 0;
            _beta.im = 0;
            if( CV_MAT_CN(type) == 2 )
                lda /= 2, ldb /= 2, ldd /= 2;

            blas_func( transb, transa, &d_size.width, &d_size.height, &len,
                   &_alpha, B->data.ptr, &ldb, A->data.ptr, &lda,
                   &_beta, D->data.ptr, &ldd );
        }
        else
        {
            CvComplex64f _alpha, _beta;

            lda = A->step/sizeof(double);
            ldb = b_step/sizeof(double);
            ldd = D->step/sizeof(double);
            _alpha.re = alpha;
            _alpha.im = 0;
            _beta.re = C->data.ptr ? beta : 0;
            _beta.im = 0;
            if( CV_MAT_CN(type) == 2 )
                lda /= 2, ldb /= 2, ldd /= 2;

            blas_func( transb, transa, &d_size.width, &d_size.height, &len,
                   &_alpha, B->data.ptr, &ldb, A->data.ptr, &lda,
                   &_beta, D->data.ptr, &ldd );
        }
    }
    else*/ if( ((d_size.height <= block_lin_size/2 || d_size.width <= block_lin_size/2) &&
        len <= 10000) || len <= 10 ||
        (d_size.width <= block_lin_size &&
        d_size.height <= block_lin_size && len <= block_lin_size) )
    {
        singleMulFunc( A.ptr(), A.step, B.ptr(), b_step, Cdata, Cstep,
                       matD->ptr(), matD->step, a_size, d_size, alpha, beta, flags );
    }
    else
    {
        int is_a_t = flags & GEMM_1_T;
        int is_b_t = flags & GEMM_2_T;
        int elem_size = CV_ELEM_SIZE(type);
        int dk0_1, dk0_2;
        size_t a_buf_size = 0, b_buf_size, d_buf_size;
        uchar* a_buf = 0;
        uchar* b_buf = 0;
        uchar* d_buf = 0;
        int j, k, di = 0, dj = 0, dk = 0;
        int dm0, dn0, dk0;
        size_t a_step0, a_step1, b_step0, b_step1, c_step0, c_step1;
        int work_elem_size = elem_size << (CV_MAT_DEPTH(type) == CV_32F ? 1 : 0);

        if( !is_a_t )
            a_step0 = A.step, a_step1 = elem_size;
        else
            a_step0 = elem_size, a_step1 = A.step;

        if( !is_b_t )
            b_step0 = b_step, b_step1 = elem_size;
        else
            b_step0 = elem_size, b_step1 = b_step;

        if( C.empty() )
        {
            c_step0 = c_step1 = 0;
            flags &= ~GEMM_3_T;
        }
        else if( !(flags & GEMM_3_T) )
            c_step0 = C.step, c_step1 = elem_size;
        else
            c_step0 = elem_size, c_step1 = C.step;

        dm0 = std::min( block_lin_size, d_size.height );
        dn0 = std::min( block_lin_size, d_size.width );
        dk0_1 = block_size / dm0;
        dk0_2 = block_size / dn0;
        dk0 = std::min( dk0_1, dk0_2 );
        dk0 = std::min( dk0, len );
        if( dk0*dm0 > block_size )
            dm0 = block_size / dk0;
        if( dk0*dn0 > block_size )
            dn0 = block_size / dk0;

        dk0_1 = (dn0+dn0/8+2) & -2;
        b_buf_size = (size_t)(dk0+dk0/8+1)*dk0_1*elem_size;
        d_buf_size = (size_t)(dk0+dk0/8+1)*dk0_1*work_elem_size;

        if( is_a_t )
        {
            a_buf_size = (size_t)(dm0+dm0/8+1)*((dk0+dk0/8+2)&-2)*elem_size;
            flags &= ~GEMM_1_T;
        }

        buf.allocate(d_buf_size + b_buf_size + a_buf_size);
        d_buf = buf.data();
        b_buf = d_buf + d_buf_size;

        if( is_a_t )
            a_buf = b_buf + b_buf_size;

        for( i = 0; i < d_size.height; i += di )
        {
            di = dm0;
            if( i + di >= d_size.height || 8*(i + di) + di > 8*d_size.height )
                di = d_size.height - i;

            for( j = 0; j < d_size.width; j += dj )
            {
                uchar* _d = matD->ptr() + i*matD->step + j*elem_size;
                const uchar* _c = Cdata + i*c_step0 + j*c_step1;
                size_t _d_step = matD->step;
                dj = dn0;

                if( j + dj >= d_size.width || 8*(j + dj) + dj > 8*d_size.width )
                    dj = d_size.width - j;

                flags &= 15;
                if( dk0 < len )
                {
                    _d = d_buf;
                    _d_step = dj*work_elem_size;
                }

                for( k = 0; k < len; k += dk )
                {
                    const uchar* _a = A.ptr() + i*a_step0 + k*a_step1;
                    size_t _a_step = A.step;
                    const uchar* _b = B.ptr() + k*b_step0 + j*b_step1;
                    size_t _b_step = b_step;
                    Size a_bl_size;

                    dk = dk0;
                    if( k + dk >= len || 8*(k + dk) + dk > 8*len )
                        dk = len - k;

                    if( !is_a_t )
                        a_bl_size.width = dk, a_bl_size.height = di;
                    else
                        a_bl_size.width = di, a_bl_size.height = dk;

                    if( a_buf && is_a_t )
                    {
                        _a_step = dk*elem_size;
                        GEMM_TransposeBlock( _a, A.step, a_buf, _a_step, a_bl_size, elem_size );
                        std::swap( a_bl_size.width, a_bl_size.height );
                        _a = a_buf;
                    }

                    if( dj < d_size.width )
                    {
                        Size b_size;
                        if( !is_b_t )
                            b_size.width = dj, b_size.height = dk;
                        else
                            b_size.width = dk, b_size.height = dj;

                        _b_step = b_size.width*elem_size;
                        GEMM_CopyBlock( _b, b_step, b_buf, _b_step, b_size, elem_size );
                        _b = b_buf;
                    }

                    if( dk0 < len )
                        blockMulFunc( _a, _a_step, _b, _b_step, _d, _d_step,
                                      a_bl_size, Size(dj,di), flags );
                    else
                        singleMulFunc( _a, _a_step, _b, _b_step, _c, Cstep,
                                       _d, _d_step, a_bl_size, Size(dj,di), alpha, beta, flags );
                    flags |= 16;
                }

                if( dk0 < len )
                    storeFunc( _c, Cstep, _d, _d_step,
                               matD->ptr(i) + j*elem_size,
                               matD->step, Size(dj,di), alpha, beta, flags );
            }
        }
    }
    }
}

template <typename fptype>inline static void
callGemmImpl(const fptype *src1, size_t src1_step, const fptype *src2, size_t src2_step, fptype alpha,
          const fptype *src3, size_t src3_step, fptype beta, fptype *dst, size_t dst_step, int m_a, int n_a, int n_d, int flags, int type)
{
    CV_StaticAssert(GEMM_1_T == CV_HAL_GEMM_1_T, "Incompatible GEMM_1_T flag in HAL");
    CV_StaticAssert(GEMM_2_T == CV_HAL_GEMM_2_T, "Incompatible GEMM_2_T flag in HAL");
    CV_StaticAssert(GEMM_3_T == CV_HAL_GEMM_3_T, "Incompatible GEMM_3_T flag in HAL");

    int b_m, b_n, c_m, c_n, m_d;

    if(flags & GEMM_2_T)
    {
        b_m = n_d;
        if(flags & GEMM_1_T )
        {
            b_n = m_a;
            m_d = n_a;
        }
        else
        {
            b_n = n_a;
            m_d = m_a;
        }
    }
    else
    {
        b_n = n_d;
        if(flags & GEMM_1_T )
        {
            b_m = m_a;
            m_d = n_a;
        }
        else
        {
            m_d = m_a;
            b_m = n_a;
        }
    }

    if(flags & GEMM_3_T)
    {
        c_m = n_d;
        c_n = m_d;
    }
    else
    {
        c_m = m_d;
        c_n = n_d;
    }

    Mat A, B, C;
    if(src1 != NULL)
        A = Mat(m_a, n_a, type, (void*)src1, src1_step);
    if(src2 != NULL)
        B = Mat(b_m, b_n, type, (void*)src2, src2_step);
    if(src3 != NULL && beta != 0.0)
        C = Mat(c_m, c_n, type, (void*)src3, src3_step);
    Mat D(m_d, n_d, type, (void*)dst, dst_step);

    gemmImpl(A, B, alpha, C, beta, D, flags);
}

void gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
             float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    callGemmImpl(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags, CV_32F);
}

void gemm64f(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
             double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    callGemmImpl(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags, CV_64F);
}

void gemm32fc(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
              float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
              int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    callGemmImpl(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags, CV_32FC2);
}

void gemm64fc(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
              double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
              int m_a, int n_a, int n_d, int flags)
{
    CV_INSTRUMENT_REGION();
    callGemmImpl(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m_a, n_a, n_d, flags, CV_64FC2);
}

#endif // !defined(CV_GEMM_BASELINE_ONLY) || defined(CV_CPU_BASELINE_MODE)



/****************************************************************************************\
*                                        Transform                                       *
\****************************************************************************************/

template<typename T, typename WT> static void
transform_( const T* src, T* dst, const WT* m, int len, int scn, int dcn )
{
    int x;

    if( scn == 2 && dcn == 2 )
    {
        for( x = 0; x < len*2; x += 2 )
        {
            WT v0 = src[x], v1 = src[x+1];
            T t0 = saturate_cast<T>(m[0]*v0 + m[1]*v1 + m[2]);
            T t1 = saturate_cast<T>(m[3]*v0 + m[4]*v1 + m[5]);
            dst[x] = t0; dst[x+1] = t1;
        }
    }
    else if( scn == 3 && dcn == 3 )
    {
        for( x = 0; x < len*3; x += 3 )
        {
            WT v0 = src[x], v1 = src[x+1], v2 = src[x+2];
            T t0 = saturate_cast<T>(m[0]*v0 + m[1]*v1 + m[2]*v2 + m[3]);
            T t1 = saturate_cast<T>(m[4]*v0 + m[5]*v1 + m[6]*v2 + m[7]);
            T t2 = saturate_cast<T>(m[8]*v0 + m[9]*v1 + m[10]*v2 + m[11]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
    }
    else if( scn == 3 && dcn == 1 )
    {
        for( x = 0; x < len; x++, src += 3 )
            dst[x] = saturate_cast<T>(m[0]*src[0] + m[1]*src[1] + m[2]*src[2] + m[3]);
    }
    else if( scn == 4 && dcn == 4 )
    {
        for( x = 0; x < len*4; x += 4 )
        {
            WT v0 = src[x], v1 = src[x+1], v2 = src[x+2], v3 = src[x+3];
            T t0 = saturate_cast<T>(m[0]*v0 + m[1]*v1 + m[2]*v2 + m[3]*v3 + m[4]);
            T t1 = saturate_cast<T>(m[5]*v0 + m[6]*v1 + m[7]*v2 + m[8]*v3 + m[9]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<T>(m[10]*v0 + m[11]*v1 + m[12]*v2 + m[13]*v3 + m[14]);
            t1 = saturate_cast<T>(m[15]*v0 + m[16]*v1 + m[17]*v2 + m[18]*v3 + m[19]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
    }
    else
    {
        for( x = 0; x < len; x++, src += scn, dst += dcn )
        {
            const WT* _m = m;
            int j, k;
            for( j = 0; j < dcn; j++, _m += scn + 1 )
            {
                WT s = _m[scn];
                for( k = 0; k < scn; k++ )
                    s += _m[k]*src[k];
                dst[j] = saturate_cast<T>(s);
            }
        }
    }
}

static void
transform_8u( const uchar* src, uchar* dst, const float* m, int len, int scn, int dcn )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int BITS = 10, SCALE = 1 << BITS;
    const float MAX_M = (float)(1 << (15 - BITS));

    if( scn == 3 && dcn == 3 &&
        std::abs(m[0]) < MAX_M && std::abs(m[1]) < MAX_M && std::abs(m[ 2]) < MAX_M*256 && std::abs(m[ 3]) < MAX_M*256 &&
        std::abs(m[4]) < MAX_M && std::abs(m[5]) < MAX_M && std::abs(m[ 6]) < MAX_M*256 && std::abs(m[ 7]) < MAX_M*256 &&
        std::abs(m[8]) < MAX_M && std::abs(m[9]) < MAX_M && std::abs(m[10]) < MAX_M*256 && std::abs(m[11]) < MAX_M*256 )
    {
        const int nChannels = 3;

        union {
            short s[6];
            int p[3];
        } m16;
        m16.s[0] = saturate_cast<short>(m[0] * SCALE); m16.s[1] = saturate_cast<short>(m[1] * SCALE);
        m16.s[2] = saturate_cast<short>(m[4] * SCALE); m16.s[3] = saturate_cast<short>(m[5] * SCALE);
        m16.s[4] = saturate_cast<short>(m[8] * SCALE); m16.s[5] = saturate_cast<short>(m[9] * SCALE);
        int m32[] = {saturate_cast<int>(m[ 2] * SCALE), saturate_cast<int>(m[ 3] * SCALE),
                     saturate_cast<int>(m[ 6] * SCALE), saturate_cast<int>(m[ 7] * SCALE),
                     saturate_cast<int>(m[10] * SCALE), saturate_cast<int>(m[11] * SCALE)};
        v_int16 m01 = v_reinterpret_as_s16(vx_setall_s32(m16.p[0]));
        v_int32 m2 = vx_setall_s32(m32[0]);
        v_int32 m3 = vx_setall_s32(m32[1]);
        v_int16 m45 = v_reinterpret_as_s16(vx_setall_s32(m16.p[1]));
        v_int32 m6 = vx_setall_s32(m32[2]);
        v_int32 m7 = vx_setall_s32(m32[3]);
        v_int16 m89 = v_reinterpret_as_s16(vx_setall_s32(m16.p[2]));
        v_int32 m10 = vx_setall_s32(m32[4]);
        v_int32 m11 = vx_setall_s32(m32[5]);
        int x = 0;
        for (; x <= (len - VTraits<v_uint8>::vlanes()) * nChannels; x += VTraits<v_uint8>::vlanes() * nChannels)
        {
            v_uint8 b, g, r;
            v_load_deinterleave(src + x, b, g, r);
            v_uint8 bgl, bgh;
            v_zip(b, g, bgl, bgh);
            v_uint16 rl, rh;
            v_expand(r, rl, rh);

            v_int16 dbl, dbh, dgl, dgh, drl, drh;
            v_uint16 p0, p2;
            v_int32 p1, p3;
            v_expand(bgl, p0, p2);
            v_expand(v_reinterpret_as_s16(rl), p1, p3);
            dbl = v_rshr_pack<BITS>(v_add(v_add(v_dotprod(v_reinterpret_as_s16(p0), m01), v_mul(p1, m2)), m3),
                                    v_add(v_add(v_dotprod(v_reinterpret_as_s16(p2), m01), v_mul(p3, m2)), m3));
            dgl = v_rshr_pack<BITS>(v_add(v_add(v_dotprod(v_reinterpret_as_s16(p0), m45), v_mul(p1, m6)), m7),
                                    v_add(v_add(v_dotprod(v_reinterpret_as_s16(p2), m45), v_mul(p3, m6)), m7));
            drl = v_rshr_pack<BITS>(v_add(v_add(v_dotprod(v_reinterpret_as_s16(p0), m89), v_mul(p1, m10)), m11),
                                    v_add(v_add(v_dotprod(v_reinterpret_as_s16(p2), m89), v_mul(p3, m10)), m11));
            v_expand(bgh, p0, p2);
            v_expand(v_reinterpret_as_s16(rh), p1, p3);
            dbh = v_rshr_pack<BITS>(v_add(v_add(v_dotprod(v_reinterpret_as_s16(p0), m01), v_mul(p1, m2)), m3),
                                    v_add(v_add(v_dotprod(v_reinterpret_as_s16(p2), m01), v_mul(p3, m2)), m3));
            dgh = v_rshr_pack<BITS>(v_add(v_add(v_dotprod(v_reinterpret_as_s16(p0), m45), v_mul(p1, m6)), m7),
                                    v_add(v_add(v_dotprod(v_reinterpret_as_s16(p2), m45), v_mul(p3, m6)), m7));
            drh = v_rshr_pack<BITS>(v_add(v_add(v_dotprod(v_reinterpret_as_s16(p0), m89), v_mul(p1, m10)), m11),
                                    v_add(v_add(v_dotprod(v_reinterpret_as_s16(p2), m89), v_mul(p3, m10)), m11));
            v_store_interleave(dst + x, v_pack_u(dbl, dbh), v_pack_u(dgl, dgh), v_pack_u(drl, drh));
        }
        m32[1] = saturate_cast<int>((m[3] + 0.5f)*SCALE);
        m32[3] = saturate_cast<int>((m[7] + 0.5f)*SCALE);
        m32[5] = saturate_cast<int>((m[11] + 0.5f)*SCALE);
        for( ; x < len * nChannels; x += nChannels )
        {
            int v0 = src[x], v1 = src[x+1], v2 = src[x+2];
            uchar t0 = saturate_cast<uchar>((m16.s[0] * v0 + m16.s[1] * v1 + m32[0] * v2 + m32[1]) >> BITS);
            uchar t1 = saturate_cast<uchar>((m16.s[2] * v0 + m16.s[3] * v1 + m32[2] * v2 + m32[3]) >> BITS);
            uchar t2 = saturate_cast<uchar>((m16.s[4] * v0 + m16.s[5] * v1 + m32[4] * v2 + m32[5]) >> BITS);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
        vx_cleanup();
        return;
    }
#endif

    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_16u( const ushort* src, ushort* dst, const float* m, int len, int scn, int dcn )
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if( scn == 3 && dcn == 3 )
    {
        int x = 0;
#if CV_SIMD_WIDTH > 16
        v_float32 m0  = vx_setall_f32(m[ 0]);
        v_float32 m1  = vx_setall_f32(m[ 1]);
        v_float32 m2  = vx_setall_f32(m[ 2]);
        v_float32 m3  = vx_setall_f32(m[ 3] - 32768.f);
        v_float32 m4  = vx_setall_f32(m[ 4]);
        v_float32 m5  = vx_setall_f32(m[ 5]);
        v_float32 m6  = vx_setall_f32(m[ 6]);
        v_float32 m7  = vx_setall_f32(m[ 7] - 32768.f);
        v_float32 m8  = vx_setall_f32(m[ 8]);
        v_float32 m9  = vx_setall_f32(m[ 9]);
        v_float32 m10 = vx_setall_f32(m[10]);
        v_float32 m11 = vx_setall_f32(m[11] - 32768.f);
        v_int16 delta = vx_setall_s16(-32768);
        for (; x <= (len - VTraits<v_uint16>::vlanes())*3; x +=  VTraits<v_uint16>::vlanes()*3)
        {
            v_uint16 b, g, r;
            v_load_deinterleave(src + x, b, g, r);
            v_uint32 bl, bh, gl, gh, rl, rh;
            v_expand(b, bl, bh);
            v_expand(g, gl, gh);
            v_expand(r, rl, rh);

            v_int16 db, dg, dr;
            db = v_add_wrap(v_pack(v_round(v_muladd(v_cvt_f32(v_reinterpret_as_s32(bl)), m0, v_muladd(v_cvt_f32(v_reinterpret_as_s32(gl)), m1, v_muladd(v_cvt_f32(v_reinterpret_as_s32(rl)),  m2,  m3)))),
                                   v_round(v_muladd(v_cvt_f32(v_reinterpret_as_s32(bh)), m0, v_muladd(v_cvt_f32(v_reinterpret_as_s32(gh)), m1, v_muladd(v_cvt_f32(v_reinterpret_as_s32(rh)),  m2,  m3))))), delta);
            dg = v_add_wrap(v_pack(v_round(v_muladd(v_cvt_f32(v_reinterpret_as_s32(bl)), m4, v_muladd(v_cvt_f32(v_reinterpret_as_s32(gl)), m5, v_muladd(v_cvt_f32(v_reinterpret_as_s32(rl)),  m6,  m7)))),
                                   v_round(v_muladd(v_cvt_f32(v_reinterpret_as_s32(bh)), m4, v_muladd(v_cvt_f32(v_reinterpret_as_s32(gh)), m5, v_muladd(v_cvt_f32(v_reinterpret_as_s32(rh)),  m6,  m7))))), delta);
            dr = v_add_wrap(v_pack(v_round(v_muladd(v_cvt_f32(v_reinterpret_as_s32(bl)), m8, v_muladd(v_cvt_f32(v_reinterpret_as_s32(gl)), m9, v_muladd(v_cvt_f32(v_reinterpret_as_s32(rl)), m10, m11)))),
                                   v_round(v_muladd(v_cvt_f32(v_reinterpret_as_s32(bh)), m8, v_muladd(v_cvt_f32(v_reinterpret_as_s32(gh)), m9, v_muladd(v_cvt_f32(v_reinterpret_as_s32(rh)), m10, m11))))), delta);
            v_store_interleave(dst + x, v_reinterpret_as_u16(db), v_reinterpret_as_u16(dg), v_reinterpret_as_u16(dr));
        }
#endif
#if CV_SIMD128
        v_float32x4 _m0l(m[0], m[4], m[ 8], 0.f);
        v_float32x4 _m1l(m[1], m[5], m[ 9], 0.f);
        v_float32x4 _m2l(m[2], m[6], m[10], 0.f);
        v_float32x4 _m3l(m[3] - 32768.f, m[7] - 32768.f, m[11] - 32768.f, 0.f);
        v_float32x4 _m0h = v_rotate_left<1>(_m0l);
        v_float32x4 _m1h = v_rotate_left<1>(_m1l);
        v_float32x4 _m2h = v_rotate_left<1>(_m2l);
        v_float32x4 _m3h = v_rotate_left<1>(_m3l);
        v_int16x8 _delta(0, -32768, -32768, -32768, -32768, -32768, -32768, 0);
        for( ; x <= len*3 - VTraits<v_uint16x8>::vlanes(); x += 3*VTraits<v_uint16x8>::vlanes()/4 )
            v_store(dst + x, v_rotate_right<1>(v_reinterpret_as_u16(v_add_wrap(v_pack(
                             v_round(v_matmuladd(v_cvt_f32(v_reinterpret_as_s32(v_load_expand(src + x    ))), _m0h, _m1h, _m2h, _m3h)),
                             v_round(v_matmuladd(v_cvt_f32(v_reinterpret_as_s32(v_load_expand(src + x + 3))), _m0l, _m1l, _m2l, _m3l))), _delta))));
#endif //CV_SIMD128
        for( ; x < len * 3; x += 3 )
        {
            float v0 = src[x], v1 = src[x + 1], v2 = src[x + 2];
            ushort t0 = saturate_cast<ushort>(m[0] * v0 + m[1] * v1 + m[ 2] * v2 + m[ 3]);
            ushort t1 = saturate_cast<ushort>(m[4] * v0 + m[5] * v1 + m[ 6] * v2 + m[ 7]);
            ushort t2 = saturate_cast<ushort>(m[8] * v0 + m[9] * v1 + m[10] * v2 + m[11]);
            dst[x] = t0; dst[x + 1] = t1; dst[x + 2] = t2;
        }
        vx_cleanup();
        return;
    }
#endif

    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_32f( const float* src, float* dst, const float* m, int len, int scn, int dcn )
{
// Disabled for RISC-V Vector (scalable), because of:
// 1. v_matmuladd for RVV is 128-bit only but not scalable, this will fail the test `Core_Transform.accuracy`.
// 2. Both gcc and clang can autovectorize this, with better performance than using Universal intrinsic.
#if (CV_SIMD || CV_SIMD_SCALABLE) && !defined(__aarch64__) && !defined(_M_ARM64) && !(CV_TRY_RVV && CV_RVV)
    int x = 0;
    if( scn == 3 && dcn == 3 )
    {
        int idx[VTraits<v_float32>::max_nlanes/2];
        for( int i = 0; i < VTraits<v_float32>::vlanes()/4; i++ )
        {
            idx[i] = 3*i;
            idx[i + VTraits<v_float32>::vlanes()/4] = 0;
        }
        float _m[] = { m[0], m[4], m[ 8], 0.f,
                       m[1], m[5], m[ 9], 0.f,
                       m[2], m[6], m[10], 0.f,
                       m[3], m[7], m[11], 0.f };
        v_float32 m0 = vx_lut_quads(_m     , idx + VTraits<v_float32>::vlanes()/4);
        v_float32 m1 = vx_lut_quads(_m +  4, idx + VTraits<v_float32>::vlanes()/4);
        v_float32 m2 = vx_lut_quads(_m +  8, idx + VTraits<v_float32>::vlanes()/4);
        v_float32 m3 = vx_lut_quads(_m + 12, idx + VTraits<v_float32>::vlanes()/4);
        for( ; x <= len*3 - VTraits<v_float32>::vlanes(); x += 3*VTraits<v_float32>::vlanes()/4 )
            v_store(dst + x, v_pack_triplets(v_matmuladd(vx_lut_quads(src + x, idx), m0, m1, m2, m3)));
        for( ; x < len*3; x += 3 )
        {
            float v0 = src[x], v1 = src[x+1], v2 = src[x+2];
            float t0 = saturate_cast<float>(m[0]*v0 + m[1]*v1 + m[ 2]*v2 + m[ 3]);
            float t1 = saturate_cast<float>(m[4]*v0 + m[5]*v1 + m[ 6]*v2 + m[ 7]);
            float t2 = saturate_cast<float>(m[8]*v0 + m[9]*v1 + m[10]*v2 + m[11]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
        vx_cleanup();
        return;
    }

    if( scn == 4 && dcn == 4 )
    {
#if CV_SIMD_WIDTH > 16
        int idx[VTraits<v_float32>::max_nlanes/4];
        for( int i = 0; i < VTraits<v_float32>::vlanes()/4; i++ )
            idx[i] = 0;
        float _m[] = { m[4], m[9], m[14], m[19] };
        v_float32 m0 = vx_lut_quads(m   , idx);
        v_float32 m1 = vx_lut_quads(m+ 5, idx);
        v_float32 m2 = vx_lut_quads(m+10, idx);
        v_float32 m3 = vx_lut_quads(m+15, idx);
        v_float32 m4 = vx_lut_quads(_m, idx);
        for( ; x <= len*4 - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes() )
        {
            v_float32 v_src = vx_load(src + x);
            v_store(dst + x, v_add(v_reduce_sum4(v_mul(v_src, m0), v_mul(v_src, m1), v_mul(v_src, m2), v_mul(v_src, m3)), m4));
        }
#endif
#if CV_SIMD128
        v_float32x4 _m0 = v_load(m     );
        v_float32x4 _m1 = v_load(m +  5);
        v_float32x4 _m2 = v_load(m + 10);
        v_float32x4 _m3 = v_load(m + 15);
        v_float32x4 _m4(m[4], m[9], m[14], m[19]);
        for( ; x < len*4; x += VTraits<v_float32x4>::vlanes() )
        {
            v_float32x4 v_src = v_load(src + x);
            v_store(dst + x, v_add(v_reduce_sum4(v_mul(v_src, _m0), v_mul(v_src, _m1), v_mul(v_src, _m2), v_mul(v_src, _m3)), _m4));
        }
#else // CV_SIMD_WIDTH >= 16 && !CV_SIMD128
        for( ; x < len*4; x += 4 )
        {
            float v0 = src[x], v1 = src[x+1], v2 = src[x+2], v3 = src[x+3];
            float t0 = saturate_cast<float>(m[0]*v0 + m[1]*v1 + m[ 2]*v2 + m[ 3]*v3 + m[ 4]);
            float t1 = saturate_cast<float>(m[5]*v0 + m[6]*v1 + m[ 7]*v2 + m[ 8]*v3 + m[ 9]);
            float t2 = saturate_cast<float>(m[10]*v0 + m[11]*v1 + m[12]*v2 + m[13]*v3 + m[14]);
            float t3 = saturate_cast<float>(m[15]*v0 + m[16]*v1 + m[17]*v2 + m[18]*v3 + m[19]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2; dst[x+3] = t3;
        }
#endif
        vx_cleanup();
        return;
    }
#endif

    transform_(src, dst, m, len, scn, dcn);
}


static void
transform_8s(const schar* src, schar* dst, const float* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_16s(const short* src, short* dst, const float* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_32s(const int* src, int* dst, const double* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_64f(const double* src, double* dst, const double* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

template<typename T, typename WT> static void
diagtransform_( const T* src, T* dst, const WT* m, int len, int cn, int )
{
    int x;

    if( cn == 2 )
    {
        for( x = 0; x < len*2; x += 2 )
        {
            T t0 = saturate_cast<T>(m[0]*src[x] + m[2]);
            T t1 = saturate_cast<T>(m[4]*src[x+1] + m[5]);
            dst[x] = t0; dst[x+1] = t1;
        }
    }
    else if( cn == 3 )
    {
        for( x = 0; x < len*3; x += 3 )
        {
            T t0 = saturate_cast<T>(m[0]*src[x] + m[3]);
            T t1 = saturate_cast<T>(m[5]*src[x+1] + m[7]);
            T t2 = saturate_cast<T>(m[10]*src[x+2] + m[11]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
    }
    else if( cn == 4 )
    {
        for( x = 0; x < len*4; x += 4 )
        {
            T t0 = saturate_cast<T>(m[0]*src[x] + m[4]);
            T t1 = saturate_cast<T>(m[6]*src[x+1] + m[9]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<T>(m[12]*src[x+2] + m[14]);
            t1 = saturate_cast<T>(m[18]*src[x+3] + m[19]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
    }
    else
    {
        for( x = 0; x < len; x++, src += cn, dst += cn )
        {
            const WT* _m = m;
            for( int j = 0; j < cn; j++, _m += cn + 1 )
                dst[j] = saturate_cast<T>(src[j]*_m[j] + _m[cn]);
        }
    }
}

static void
diagtransform_8u(const uchar* src, uchar* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_8s(const schar* src, schar* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_16u(const ushort* src, ushort* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_16s(const short* src, short* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_32s(const int* src, int* dst, const double* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_32f(const float* src, float* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_64f(const double* src, double* dst, const double* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}


TransformFunc getTransformFunc(int depth)
{
    static TransformFunc transformTab[CV_DEPTH_MAX] =
    {
        (TransformFunc)transform_8u, (TransformFunc)transform_8s, (TransformFunc)transform_16u,
        (TransformFunc)transform_16s, (TransformFunc)transform_32s, (TransformFunc)transform_32f,
        (TransformFunc)transform_64f, 0
    };

    return transformTab[depth];
}

TransformFunc getDiagTransformFunc(int depth)
{
    static TransformFunc diagTransformTab[CV_DEPTH_MAX] =
    {
        (TransformFunc)diagtransform_8u, (TransformFunc)diagtransform_8s, (TransformFunc)diagtransform_16u,
        (TransformFunc)diagtransform_16s, (TransformFunc)diagtransform_32s, (TransformFunc)diagtransform_32f,
        (TransformFunc)diagtransform_64f, 0
    };

    return diagTransformTab[depth];
}



/****************************************************************************************\
*                                  Perspective Transform                                 *
\****************************************************************************************/

template<typename T> static void
perspectiveTransform_( const T* src, T* dst, const double* m, int len, int scn, int dcn )
{
    const double eps = FLT_EPSILON;
    int i;

    if( scn == 2 && dcn == 2 )
    {
        for( i = 0; i < len*2; i += 2 )
        {
            T x = src[i], y = src[i + 1];
            double w = x*m[6] + y*m[7] + m[8];

            if( fabs(w) > eps )
            {
                w = 1./w;
                dst[i] = (T)((x*m[0] + y*m[1] + m[2])*w);
                dst[i+1] = (T)((x*m[3] + y*m[4] + m[5])*w);
            }
            else
                dst[i] = dst[i+1] = (T)0;
        }
    }
    else if( scn == 3 && dcn == 3 )
    {
        for( i = 0; i < len*3; i += 3 )
        {
            T x = src[i], y = src[i + 1], z = src[i + 2];
            double w = x*m[12] + y*m[13] + z*m[14] + m[15];

            if( fabs(w) > eps )
            {
                w = 1./w;
                dst[i] = (T)((x*m[0] + y*m[1] + z*m[2] + m[3]) * w);
                dst[i+1] = (T)((x*m[4] + y*m[5] + z*m[6] + m[7]) * w);
                dst[i+2] = (T)((x*m[8] + y*m[9] + z*m[10] + m[11]) * w);
            }
            else
                dst[i] = dst[i+1] = dst[i+2] = (T)0;
        }
    }
    else if( scn == 3 && dcn == 2 )
    {
        for( i = 0; i < len; i++, src += 3, dst += 2 )
        {
            T x = src[0], y = src[1], z = src[2];
            double w = x*m[8] + y*m[9] + z*m[10] + m[11];

            if( fabs(w) > eps )
            {
                w = 1./w;
                dst[0] = (T)((x*m[0] + y*m[1] + z*m[2] + m[3])*w);
                dst[1] = (T)((x*m[4] + y*m[5] + z*m[6] + m[7])*w);
            }
            else
                dst[0] = dst[1] = (T)0;
        }
    }
    else
    {
        for( i = 0; i < len; i++, src += scn, dst += dcn )
        {
            const double* _m = m + dcn*(scn + 1);
            double w = _m[scn];
            int j, k;
            for( k = 0; k < scn; k++ )
                w += _m[k]*src[k];
            if( fabs(w) > eps )
            {
                _m = m;
                for( j = 0; j < dcn; j++, _m += scn + 1 )
                {
                    double s = _m[scn];
                    for( k = 0; k < scn; k++ )
                        s += _m[k]*src[k];
                    dst[j] = (T)(s*w);
                }
            }
            else
                for( j = 0; j < dcn; j++ )
                    dst[j] = 0;
        }
    }
}

static void
perspectiveTransform_32f(const float* src, float* dst, const double* m, int len, int scn, int dcn)
{
    perspectiveTransform_(src, dst, m, len, scn, dcn);
}

static void
perspectiveTransform_64f(const double* src, double* dst, const double* m, int len, int scn, int dcn)
{
    perspectiveTransform_(src, dst, m, len, scn, dcn);
}

TransformFunc getPerspectiveTransform(int depth)
{
    if (depth == CV_32F)
        return (TransformFunc)perspectiveTransform_32f;
    if (depth == CV_64F)
        return (TransformFunc)perspectiveTransform_64f;
    CV_Assert(0 && "Not supported");
}



/****************************************************************************************\
*                                       ScaleAdd                                         *
\****************************************************************************************/

static void scaleAdd_32f(const float* src1, const float* src2, float* dst,
                         int len, float* _alpha)
{
    float alpha = *_alpha;
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 v_alpha = vx_setall_f32(alpha);
    const int cWidth = VTraits<v_float32>::vlanes();
    for (; i <= len - cWidth; i += cWidth)
        v_store(dst + i, v_muladd(vx_load(src1 + i), v_alpha, vx_load(src2 + i)));
    vx_cleanup();
#endif
    for (; i < len; i++)
        dst[i] = src1[i] * alpha + src2[i];
}


static void scaleAdd_64f(const double* src1, const double* src2, double* dst,
                         int len, double* _alpha)
{
    double alpha = *_alpha;
    int i = 0;
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    v_float64 a2 = vx_setall_f64(alpha);
    const int cWidth = VTraits<v_float64>::vlanes();
    for (; i <= len - cWidth; i += cWidth)
        v_store(dst + i, v_muladd(vx_load(src1 + i), a2, vx_load(src2 + i)));
    vx_cleanup();
#endif
    for (; i < len; i++)
        dst[i] = src1[i] * alpha + src2[i];
}

ScaleAddFunc getScaleAddFunc(int depth)
{
    if (depth == CV_32F)
        return (ScaleAddFunc)scaleAdd_32f;
    if (depth == CV_64F)
        return (ScaleAddFunc)scaleAdd_64f;
    CV_Assert(0 && "Not supported");
}



/****************************************************************************************\
*                                        Mahalanobis                                     *
\****************************************************************************************/

template<typename T> static inline
double MahalanobisImpl(const Mat& v1, const Mat& v2, const Mat& icovar, double *diff_buffer /*[len]*/, int len /*=v1.total()*/)
{
    CV_INSTRUMENT_REGION();

    Size sz = v1.size();
    double result = 0;

    sz.width *= v1.channels();
    if (v1.isContinuous() && v2.isContinuous())
    {
        sz.width *= sz.height;
        sz.height = 1;
    }

    {
        const T* src1 = v1.ptr<T>();
        const T* src2 = v2.ptr<T>();
        size_t step1 = v1.step/sizeof(src1[0]);
        size_t step2 = v2.step/sizeof(src2[0]);
        double* diff = diff_buffer;
        const T* mat = icovar.ptr<T>();
        size_t matstep = icovar.step/sizeof(mat[0]);

        for (; sz.height--; src1 += step1, src2 += step2, diff += sz.width)
        {
            for (int i = 0; i < sz.width; i++)
                diff[i] = src1[i] - src2[i];
        }

        diff = diff_buffer;
        for (int i = 0; i < len; i++, mat += matstep)
        {
            double row_sum = 0;
            int j = 0;
#if CV_ENABLE_UNROLLED
            for(; j <= len - 4; j += 4 )
                row_sum += diff[j]*mat[j] + diff[j+1]*mat[j+1] +
                           diff[j+2]*mat[j+2] + diff[j+3]*mat[j+3];
#endif
            for (; j < len; j++)
                row_sum += diff[j]*mat[j];
            result += row_sum * diff[i];
        }
    }
    return result;
}

MahalanobisImplFunc getMahalanobisImplFunc(int depth)
{
    if (depth == CV_32F)
        return (MahalanobisImplFunc)MahalanobisImpl<float>;
    if (depth == CV_64F)
        return (MahalanobisImplFunc)MahalanobisImpl<double>;
    CV_Assert(0 && "Not supported");
}


#if !defined(CV_MULTRANSPOSED_BASELINE_ONLY) || defined(CV_CPU_BASELINE_MODE)
/****************************************************************************************\
*                                        MulTransposed                                   *
\****************************************************************************************/

template<typename sT, typename dT> static void
MulTransposedR(const Mat& srcmat, const Mat& dstmat, const Mat& deltamat, double scale)
{
    int i, j, k;
    const sT* src = srcmat.ptr<sT>();
    dT* dst = (dT*)dstmat.ptr<dT>();
    const dT* delta = deltamat.ptr<dT>();
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    Size size = srcmat.size();
    dT* tdst = dst;
    dT* col_buf = 0;
    dT* delta_buf = 0;
    int buf_size = size.height*sizeof(dT);
    AutoBuffer<uchar> buf;

    if( delta && delta_cols < size.width )
    {
        CV_Assert( delta_cols == 1 );
        buf_size *= 5;
    }
    buf.allocate(buf_size);
    col_buf = (dT*)buf.data();

    if( delta && delta_cols < size.width )
    {
        delta_buf = col_buf + size.height;
        for( i = 0; i < size.height; i++ )
            delta_buf[i*4] = delta_buf[i*4+1] =
                delta_buf[i*4+2] = delta_buf[i*4+3] = delta[i*deltastep];
        delta = delta_buf;
        deltastep = deltastep ? 4 : 0;
    }

#if CV_SIMD128_64F
    v_float64x2 v_scale = v_setall_f64(scale);
#endif

    if( !delta )
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            for( k = 0; k < size.height; k++ )
                col_buf[k] = src[k*srcstep+i];

            for( j = i; j <= size.width - 4; j += 4 )
            {
#if CV_SIMD128_64F
                if (DataType<sT>::depth == CV_64F && DataType<dT>::depth == CV_64F)
                {
                    v_float64x2 s0 = v_setzero_f64(), s1 = v_setzero_f64();
                    const double *tsrc = (double*)(src + j);

                    for( k = 0; k < size.height; k++, tsrc += srcstep )
                    {
                        v_float64x2 a = v_setall_f64((double)col_buf[k]);
                        s0 = v_add(s0, v_mul(a, v_load(tsrc + 0)));
                        s1 = v_add(s1, v_mul(a, v_load(tsrc + 2)));
                    }

                    v_store((double*)(tdst+j), v_mul(s0, v_scale));
                    v_store((double*)(tdst+j+2), v_mul(s1, v_scale));
                } else
#endif
                {
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    const sT *tsrc = src + j;

                    for( k = 0; k < size.height; k++, tsrc += srcstep )
                    {
                        double a = col_buf[k];
                        s0 += a * tsrc[0];
                        s1 += a * tsrc[1];
                        s2 += a * tsrc[2];
                        s3 += a * tsrc[3];
                    }

                    tdst[j] = (dT)(s0*scale);
                    tdst[j+1] = (dT)(s1*scale);
                    tdst[j+2] = (dT)(s2*scale);
                    tdst[j+3] = (dT)(s3*scale);
                }
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                    s0 += (double)col_buf[k] * tsrc[0];

                tdst[j] = (dT)(s0*scale);
            }
        }
    else
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            if( !delta_buf )
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta[k*deltastep+i];
            else
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta_buf[k*deltastep];

            for( j = i; j <= size.width - 4; j += 4 )
            {
#if CV_SIMD128_64F
                if (DataType<sT>::depth == CV_64F && DataType<dT>::depth == CV_64F)
                {
                    v_float64x2 s0 = v_setzero_f64(), s1 = v_setzero_f64();
                    const double *tsrc = (double*)(src + j);
                    const double *d = (double*)(delta_buf ? delta_buf : delta + j);

                    for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                    {
                        v_float64x2 a = v_setall_f64((double)col_buf[k]);
                        s0 = v_add(s0, v_mul(a, v_sub(v_load(tsrc + 0), v_load(d + 0))));
                        s1 = v_add(s1, v_mul(a, v_sub(v_load(tsrc + 2), v_load(d + 2))));
                    }

                    v_store((double*)(tdst+j), v_mul(s0, v_scale));
                    v_store((double*)(tdst+j+2), v_mul(s1, v_scale));
                }
                else
#endif

                {
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    const sT *tsrc = src + j;
                    const dT *d = delta_buf ? delta_buf : delta + j;

                    for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                    {
                        double a = col_buf[k];
                        s0 += a * (tsrc[0] - d[0]);
                        s1 += a * (tsrc[1] - d[1]);
                        s2 += a * (tsrc[2] - d[2]);
                        s3 += a * (tsrc[3] - d[3]);
                    }

                    tdst[j] = (dT)(s0*scale);
                    tdst[j+1] = (dT)(s1*scale);
                    tdst[j+2] = (dT)(s2*scale);
                    tdst[j+3] = (dT)(s3*scale);
                }
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                    s0 += (double)col_buf[k] * (tsrc[0] - d[0]);

                tdst[j] = (dT)(s0*scale);
            }
        }
}


template<typename sT, typename dT> static void
MulTransposedL(const Mat& srcmat, const Mat& dstmat, const Mat& deltamat, double scale)
{
    int i, j, k;
    const sT* src = srcmat.ptr<sT>();
    dT* dst = (dT*)dstmat.ptr<dT>();
    const dT* delta = deltamat.ptr<dT>();
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    Size size = srcmat.size();
    dT* tdst = dst;

    if( !delta )
        for( i = 0; i < size.height; i++, tdst += dststep )
            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc1 = src + i*srcstep;
                const sT *tsrc2 = src + j*srcstep;
#if CV_SIMD128_64F
                if (DataType<sT>::depth == CV_64F && DataType<dT>::depth == CV_64F)
                {
                    const double *v_tsrc1 = (double *)(tsrc1);
                    const double *v_tsrc2 = (double *)(tsrc2);
                    v_float64x2 v_s = v_setzero_f64();

                    for( k = 0; k <= size.width - 4; k += 4 )
                        v_s = v_add(v_s, v_add(v_mul(v_load(v_tsrc1 + k), v_load(v_tsrc2 + k)), v_mul(v_load(v_tsrc1 + k + 2), v_load(v_tsrc2 + k + 2))));
                    s += v_reduce_sum(v_s);
                }
                else
#endif
                {
                    for( k = 0; k <= size.width - 4; k += 4 )
                        s += (double)tsrc1[k]*tsrc2[k] + (double)tsrc1[k+1]*tsrc2[k+1] +
                             (double)tsrc1[k+2]*tsrc2[k+2] + (double)tsrc1[k+3]*tsrc2[k+3];
                }
                for( ; k < size.width; k++ )
                    s += (double)tsrc1[k] * tsrc2[k];
                tdst[j] = (dT)(s*scale);
            }
    else
    {
        dT delta_buf[4];
        int delta_shift = delta_cols == size.width ? 4 : 0;
        AutoBuffer<uchar> buf(size.width*sizeof(dT));
        dT* row_buf = (dT*)buf.data();

        for( i = 0; i < size.height; i++, tdst += dststep )
        {
            const sT *tsrc1 = src + i*srcstep;
            const dT *tdelta1 = delta + i*deltastep;

            if( delta_cols < size.width )
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[0];
            else
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[k];

            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc2 = src + j*srcstep;
                const dT *tdelta2 = delta + j*deltastep;
                if( delta_cols < size.width )
                {
                    delta_buf[0] = delta_buf[1] =
                        delta_buf[2] = delta_buf[3] = tdelta2[0];
                    tdelta2 = delta_buf;
                }
#if CV_SIMD128_64F
                if (DataType<sT>::depth == CV_64F && DataType<dT>::depth == CV_64F)
                {
                    const double *v_tsrc2 = (double *)(tsrc2);
                    const double *v_tdelta2 = (double *)(tdelta2);
                    const double *v_row_buf = (double *)(row_buf);
                    v_float64x2 v_s = v_setzero_f64();

                    for( k = 0; k <= size.width - 4; k += 4, v_tdelta2 += delta_shift )
                        v_s = v_add(v_s, v_add(v_mul(v_sub(v_load(v_tsrc2 + k), v_load(v_tdelta2)), v_load(v_row_buf + k)), v_mul(v_sub(v_load(v_tsrc2 + k + 2), v_load(v_tdelta2 + 2)), v_load(v_row_buf + k + 2))));
                    s += v_reduce_sum(v_s);

                    tdelta2 = (const dT *)(v_tdelta2);
                }
                else
#endif
                {
                    for( k = 0; k <= size.width-4; k += 4, tdelta2 += delta_shift )
                        s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]) +
                             (double)row_buf[k+1]*(tsrc2[k+1] - tdelta2[1]) +
                             (double)row_buf[k+2]*(tsrc2[k+2] - tdelta2[2]) +
                             (double)row_buf[k+3]*(tsrc2[k+3] - tdelta2[3]);
                }
                for( ; k < size.width; k++, tdelta2++ )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]);
                tdst[j] = (dT)(s*scale);
            }
        }
    }
}

MulTransposedFunc getMulTransposedFunc(int stype, int dtype, bool ata)
{
    MulTransposedFunc func = NULL;
    if (stype == CV_8U && dtype == CV_32F)
    {
        func = ata ? MulTransposedR<uchar,float>
                   : MulTransposedL<uchar,float>;
    }
    else if (stype == CV_8U && dtype == CV_64F)
    {
        func = ata ? MulTransposedR<uchar,double>
                   : MulTransposedL<uchar,double>;
    }
    else if(stype == CV_16U && dtype == CV_32F)
    {
        func = ata ? MulTransposedR<ushort,float>
                   : MulTransposedL<ushort,float>;
    }
    else if(stype == CV_16U && dtype == CV_64F)
    {
        func = ata ? MulTransposedR<ushort,double>
                   : MulTransposedL<ushort,double>;
    }
    else if(stype == CV_16S && dtype == CV_32F)
    {
        func = ata ? MulTransposedR<short,float>
                   : MulTransposedL<short,float>;
    }
    else if(stype == CV_16S && dtype == CV_64F)
    {
        func = ata ? MulTransposedR<short,double>
                   : MulTransposedL<short,double>;
    }
    else if(stype == CV_32F && dtype == CV_32F)
    {
        func = ata ? MulTransposedR<float,float>
                   : MulTransposedL<float,float>;
    }
    else if(stype == CV_32F && dtype == CV_64F)
    {
        func = ata ? MulTransposedR<float,double>
                   : MulTransposedL<float,double>;
    }
    else if(stype == CV_64F && dtype == CV_64F)
    {
        func = ata ? MulTransposedR<double,double>
                   : MulTransposedL<double,double>;
    }
    CV_Assert(func && "Not supported");
    return func;
}
#endif // !defined(CV_MULTRANSPOSED_BASELINE_ONLY) || defined(CV_CPU_BASELINE_MODE)


/****************************************************************************************\
*                                      Dot Product                                       *
\****************************************************************************************/

template<typename T> static inline
double dotProd_(const T* src1, const T* src2, int len)
{
    int i = 0;
    double result = 0;

#if CV_ENABLE_UNROLLED
    for( ; i <= len - 4; i += 4 )
        result += (double)src1[i]*src2[i] + (double)src1[i+1]*src2[i+1] +
            (double)src1[i+2]*src2[i+2] + (double)src1[i+3]*src2[i+3];
#endif
    for( ; i < len; i++ )
        result += (double)src1[i]*src2[i];

    return result;
}


double dotProd_8u(const uchar* src1, const uchar* src2, int len)
{
    double r = 0;
    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_uint16>::vlanes(), blockSize0 = (1 << 15), blockSize;

    while (i < len0)
    {
        blockSize = std::min(len0 - i, blockSize0);
        v_uint32 v_sum = vx_setzero_u32();
        const int cWidth = VTraits<v_uint16>::vlanes();

        int j = 0;
        for (; j <= blockSize - cWidth * 2; j += cWidth * 2)
        {
            v_uint8 v_src1 = vx_load(src1 + j);
            v_uint8 v_src2 = vx_load(src2 + j);
            v_sum = v_dotprod_expand_fast(v_src1, v_src2, v_sum);
        }

        for (; j <= blockSize - cWidth; j += cWidth)
        {
            v_int16 v_src10 = v_reinterpret_as_s16(vx_load_expand(src1 + j));
            v_int16 v_src20 = v_reinterpret_as_s16(vx_load_expand(src2 + j));
            v_sum = v_add(v_sum, v_reinterpret_as_u32(v_dotprod_fast(v_src10, v_src20)));
        }
        r += (double)v_reduce_sum(v_sum);

        src1 += blockSize;
        src2 += blockSize;
        i += blockSize;
    }
    vx_cleanup();
#endif
    return r + dotProd_(src1, src2, len - i);
}


double dotProd_8s(const schar* src1, const schar* src2, int len)
{
    double r = 0.0;
    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_int16>::vlanes(), blockSize0 = (1 << 14), blockSize;

    while (i < len0)
    {
        blockSize = std::min(len0 - i, blockSize0);
        v_int32 v_sum = vx_setzero_s32();
        const int cWidth = VTraits<v_int16>::vlanes();

        int j = 0;
        for (; j <= blockSize - cWidth * 2; j += cWidth * 2)
        {
            v_int8 v_src1 = vx_load(src1 + j);
            v_int8 v_src2 = vx_load(src2 + j);
            v_sum = v_dotprod_expand_fast(v_src1, v_src2, v_sum);
        }

        for (; j <= blockSize - cWidth; j += cWidth)
        {
            v_int16 v_src1 = vx_load_expand(src1 + j);
            v_int16 v_src2 = vx_load_expand(src2 + j);
            v_sum = v_dotprod_fast(v_src1, v_src2, v_sum);
        }
        r += (double)v_reduce_sum(v_sum);

        src1 += blockSize;
        src2 += blockSize;
        i += blockSize;
    }
    vx_cleanup();
#endif

    return r + dotProd_(src1, src2, len - i);
}

double dotProd_16u(const ushort* src1, const ushort* src2, int len)
{
    double r = 0.0;
    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_uint16>::vlanes(), blockSize0 = (1 << 24), blockSize;

    while (i < len0)
    {
        blockSize = std::min(len0 - i, blockSize0);
        v_uint64 v_sum = vx_setzero_u64();
        const int cWidth = VTraits<v_uint16>::vlanes();

        int j = 0;
        for (; j <= blockSize - cWidth; j += cWidth)
        {
            v_uint16 v_src1 = vx_load(src1 + j);
            v_uint16 v_src2 = vx_load(src2 + j);
            v_sum = v_dotprod_expand_fast(v_src1, v_src2, v_sum);
        }
        r += (double)v_reduce_sum(v_sum);

        src1 += blockSize;
        src2 += blockSize;
        i += blockSize;
    }
    vx_cleanup();
#endif
    return r + dotProd_(src1, src2, len - i);
}

double dotProd_16s(const short* src1, const short* src2, int len)
{
    double r = 0.0;
    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_int16>::vlanes(), blockSize0 = (1 << 24), blockSize;

    while (i < len0)
    {
        blockSize = std::min(len0 - i, blockSize0);
        v_int64 v_sum = vx_setzero_s64();
        const int cWidth = VTraits<v_int16>::vlanes();

        int j = 0;
        for (; j <= blockSize - cWidth; j += cWidth)
        {
            v_int16 v_src1 = vx_load(src1 + j);
            v_int16 v_src2 = vx_load(src2 + j);
            v_sum = v_dotprod_expand_fast(v_src1, v_src2, v_sum);
        }
        r += (double)v_reduce_sum(v_sum);

        src1 += blockSize;
        src2 += blockSize;
        i += blockSize;
    }
    vx_cleanup();
#endif
    return r + dotProd_(src1, src2, len - i);
}

double dotProd_32s(const int* src1, const int* src2, int len)
{
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    double r = .0;
    int i = 0;
    const int step  = VTraits<v_int32>::vlanes();
    v_float64 v_sum0 = vx_setzero_f64();
#if CV_SIMD_WIDTH == 16
    const int wstep = step * 2;
    v_float64 v_sum1 = vx_setzero_f64();
    for (; i < len - wstep; i += wstep, src1 += wstep, src2 += wstep)
    {
        v_int32 v_src10 = vx_load(src1);
        v_int32 v_src20 = vx_load(src2);
        v_int32 v_src11 = vx_load(src1 + step);
        v_int32 v_src21 = vx_load(src2 + step);
        v_sum0 = v_dotprod_expand_fast(v_src10, v_src20, v_sum0);
        v_sum1 = v_dotprod_expand_fast(v_src11, v_src21, v_sum1);
    }
    v_sum0 = v_add(v_sum0, v_sum1);
#endif
    for (; i < len - step; i += step, src1 += step, src2 += step)
    {
        v_int32 v_src1 = vx_load(src1);
        v_int32 v_src2 = vx_load(src2);
        v_sum0 = v_dotprod_expand_fast(v_src1, v_src2, v_sum0);
    }
    r = v_reduce_sum(v_sum0);
    vx_cleanup();
    return r + dotProd_(src1, src2, len - i);
#else
    return dotProd_(src1, src2, len);
#endif
}

double dotProd_32f(const float* src1, const float* src2, int len)
{
    double r = 0.0;
    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int len0 = len & -VTraits<v_float32>::vlanes(), blockSize0 = (1 << 13), blockSize;

    while (i < len0)
    {
        blockSize = std::min(len0 - i, blockSize0);
        v_float32 v_sum = vx_setzero_f32();

        int j = 0;
        int cWidth = VTraits<v_float32>::vlanes();

#if CV_ENABLE_UNROLLED
        v_float32 v_sum1 = vx_setzero_f32();
        v_float32 v_sum2 = vx_setzero_f32();
        v_float32 v_sum3 = vx_setzero_f32();

        for (; j <= blockSize - (cWidth * 4); j += (cWidth * 4))
        {
            v_sum  = v_muladd(vx_load(src1 + j),
                              vx_load(src2 + j), v_sum);
            v_sum1 = v_muladd(vx_load(src1 + j + cWidth),
                              vx_load(src2 + j + cWidth), v_sum1);
            v_sum2 = v_muladd(vx_load(src1 + j + (cWidth * 2)),
                              vx_load(src2 + j + (cWidth * 2)), v_sum2);
            v_sum3 = v_muladd(vx_load(src1 + j + (cWidth * 3)),
                              vx_load(src2 + j + (cWidth * 3)), v_sum3);
        }

        v_sum = v_add(v_sum, v_add(v_add(v_sum1, v_sum2), v_sum3));
#endif

        for (; j <= blockSize - cWidth; j += cWidth)
            v_sum = v_muladd(vx_load(src1 + j), vx_load(src2 + j), v_sum);

        r += v_reduce_sum(v_sum);

        src1 += blockSize;
        src2 += blockSize;
        i += blockSize;
    }
    vx_cleanup();
#endif
    return r + dotProd_(src1, src2, len - i);
}

double dotProd_64f(const double* src1, const double* src2, int len)
{
    return dotProd_(src1, src2, len);
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
