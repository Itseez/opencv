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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "corner.hpp"

namespace cv
{

static void calcMinEigenVal( const Mat& _cov_x2, const Mat& _cov_xy, const Mat& _cov_y2, Mat& _dst )
{
    int i, j;
    Size size = _cov_x2.size();

    if( _cov_x2.isContinuous() && _cov_xy.isContinuous() && _cov_y2.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const float* cov_x2 = _cov_x2.ptr<float>(i);
        const float* cov_xy = _cov_xy.ptr<float>(i);
        const float* cov_y2 = _cov_y2.ptr<float>(i);
        float* dst = _dst.ptr<float>(i);
        j = 0;
        calcMinEigenValLine(j, size.width, cov_x2, cov_xy, cov_y2, dst );
    }
}

static void calcHarris( const Mat& _cov_x2, const Mat& _cov_xy, const Mat& _cov_y2, Mat& _dst, double k )
{
    int i, j;
    Size size = _cov_x2.size();

    if( _cov_x2.isContinuous() && _cov_xy.isContinuous() && _cov_y2.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const float* cov_x2 = _cov_x2.ptr<float>(i);
        const float* cov_xy = _cov_xy.ptr<float>(i);
        const float* cov_y2 = _cov_y2.ptr<float>(i);
        float* dst = _dst.ptr<float>(i);
        j = 0;
        calcHarrisLine(j, size.width, cov_x2, cov_xy, cov_y2, dst, k );
    }
}

static void eigen2x2( const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, int n )
{
    for( int j = 0; j < n; j++ )
    {
        double a = cov_x2[j];
        double b = cov_xy[j];
        double c = cov_y2[j];

        double u = (a + c)*0.5;
        double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
        double l1 = u + v;
        double l2 = u - v;

        double x = b;
        double y = l1 - a;
        double e = fabs(x);

        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l1 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }

        double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j] = (float)l1;
        dst[6*j + 2] = (float)(x*d);
        dst[6*j + 3] = (float)(y*d);

        x = b;
        y = l2 - a;
        e = fabs(x);

        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l2 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }

        d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j + 1] = (float)l2;
        dst[6*j + 4] = (float)(x*d);
        dst[6*j + 5] = (float)(y*d);
    }
}

static void calcEigenValsVecs( const Mat& _cov_x2, const Mat& _cov_xy, const Mat& _cov_y2, Mat& _dst )
{
    Size size = _cov_x2.size();
    if( _cov_x2.isContinuous() && _cov_xy.isContinuous() && _cov_y2.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( int i = 0; i < size.height; i++ )
    {
        const float* cov_x2 = _cov_x2.ptr<float>(i);
        const float* cov_xy = _cov_xy.ptr<float>(i);
        const float* cov_y2 = _cov_y2.ptr<float>(i);
        float* dst = _dst.ptr<float>(i);

        eigen2x2(cov_x2, cov_xy, cov_y2, dst, size.width);
    }
}

enum { MINEIGENVAL=0, HARRIS=1, EIGENVALSVECS=2 };


static void
cornerEigenValsVecs( const Mat& src, Mat& eigenv, int block_size,
                     int aperture_size, int op_type, double k=0.,
                     int borderType=BORDER_DEFAULT )
{
    int depth = src.depth();
    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if( aperture_size < 0 )
        scale *= 2.0;
    if( depth == CV_8U )
        scale *= 255.0;
    scale = 1.0/scale;

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );

    Mat Dx, Dy;
    if( aperture_size > 0 )
    {
        Sobel( src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType );
        Sobel( src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType );
    }
    else
    {
        Scharr( src, Dx, CV_32F, 1, 0, scale, 0, borderType );
        Scharr( src, Dy, CV_32F, 0, 1, scale, 0, borderType );
    }

    Size size = src.size();
    Mat cov_x2( size, CV_32F );
    Mat cov_xy( size, CV_32F );
    Mat cov_y2( size, CV_32F );
    int i, j;

    for( i = 0; i < size.height; i++ )
    {
        float* cov_data_x2 = cov_x2.ptr<float>(i);
        float* cov_data_xy = cov_xy.ptr<float>(i);
        float* cov_data_y2 = cov_y2.ptr<float>(i);
        const float* dxdata = Dx.ptr<float>(i);
        const float* dydata = Dy.ptr<float>(i);
        j = 0;
        cornerEigenValsVecsLine(j, size.width, dxdata, dydata, cov_data_x2, cov_data_xy, cov_data_y2);
    }

    boxFilter(cov_x2, cov_x2, cov_x2.depth(), Size(block_size, block_size),
        Point(-1,-1), false, borderType );
    boxFilter(cov_xy, cov_xy, cov_xy.depth(), Size(block_size, block_size),
        Point(-1,-1), false, borderType );
    boxFilter(cov_y2, cov_y2, cov_y2.depth(), Size(block_size, block_size),
        Point(-1,-1), false, borderType );

    if( op_type == MINEIGENVAL )
        calcMinEigenVal( cov_x2, cov_xy, cov_y2, eigenv );
    else if( op_type == HARRIS )
        calcHarris( cov_x2, cov_xy, cov_y2, eigenv, k );
    else if( op_type == EIGENVALSVECS )
        calcEigenValsVecs( cov_x2, cov_xy, cov_y2, eigenv );
}

#ifdef HAVE_OPENCL

static bool extractCovData(InputArray _src, UMat & Dx, UMat & Dy, int depth,
                           float scale, int aperture_size, int borderType)
{
    UMat src = _src.getUMat();

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);

    const int sobel_lsz = 16;
    if ((aperture_size == 3 || aperture_size == 5 || aperture_size == 7 || aperture_size == -1) &&
        wholeSize.height > sobel_lsz + (aperture_size >> 1) &&
        wholeSize.width > sobel_lsz + (aperture_size >> 1))
    {
        CV_Assert(depth == CV_8U || depth == CV_32F);

        Dx.create(src.size(), CV_32FC1);
        Dy.create(src.size(), CV_32FC1);

        size_t localsize[2] = { (size_t)sobel_lsz, (size_t)sobel_lsz };
        size_t globalsize[2] = { localsize[0] * (1 + (src.cols - 1) / localsize[0]),
                                 localsize[1] * (1 + (src.rows - 1) / localsize[1]) };

        int src_offset_x = (int)((src.offset % src.step) / src.elemSize());
        int src_offset_y = (int)(src.offset / src.step);

        const char * const borderTypes[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
                                             "BORDER_WRAP", "BORDER_REFLECT101" };

        ocl::Kernel k(format("sobel%d", aperture_size).c_str(), ocl::imgproc::covardata_oclsrc,
                      cv::format("-D BLK_X=%d -D BLK_Y=%d -D %s -D SRCTYPE=%s%s",
                                 (int)localsize[0], (int)localsize[1], borderTypes[borderType], ocl::typeToStr(depth),
                                 aperture_size < 0 ? " -D SCHARR" : ""));
        if (k.empty())
            return false;

        k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, src_offset_x, src_offset_y,
               ocl::KernelArg::WriteOnlyNoSize(Dx), ocl::KernelArg::WriteOnly(Dy),
               wholeSize.height, wholeSize.width, scale);

        return k.run(2, globalsize, localsize, false);
    }
    else
    {
        if (aperture_size > 0)
        {
            Sobel(_src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType);
            Sobel(_src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType);
        }
        else
        {
            Scharr(_src, Dx, CV_32F, 1, 0, scale, 0, borderType);
            Scharr(_src, Dy, CV_32F, 0, 1, scale, 0, borderType);
        }
    }

    return true;
}

static bool ocl_cornerMinEigenValVecs(InputArray _src, OutputArray _dst, int block_size,
                                      int aperture_size, double k, int borderType, int op_type)
{
    CV_Assert(op_type == HARRIS || op_type == MINEIGENVAL);

    if ( !(borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE ||
           borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101) )
        return false;

    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    if ( !(type == CV_8UC1 || type == CV_32FC1) )
        return false;

    const char * const borderTypes[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
                                         "BORDER_WRAP", "BORDER_REFLECT101" };
    const char * const cornerType[] = { "CORNER_MINEIGENVAL", "CORNER_HARRIS", 0 };


    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if (aperture_size < 0)
        scale *= 2.0;
    if (depth == CV_8U)
        scale *= 255.0;
    scale = 1.0 / scale;

    UMat Dx, Dy;
    if (!extractCovData(_src, Dx, Dy, depth, (float)scale, aperture_size, borderType))
        return false;

    ocl::Kernel cornelKernel("corner", ocl::imgproc::corner_oclsrc,
                             format("-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s -D %s",
                                    block_size / 2, block_size / 2, block_size, block_size,
                                    borderTypes[borderType], cornerType[op_type]));
    if (cornelKernel.empty())
        return false;

    _dst.createSameSize(_src, CV_32FC1);
    UMat dst = _dst.getUMat();

    cornelKernel.args(ocl::KernelArg::ReadOnly(Dx), ocl::KernelArg::ReadOnly(Dy),
                      ocl::KernelArg::WriteOnly(dst), (float)k);

    size_t blockSizeX = 256, blockSizeY = 1;
    size_t gSize = blockSizeX - block_size / 2 * 2;
    size_t globalSizeX = (Dx.cols) % gSize == 0 ? Dx.cols / gSize * blockSizeX : (Dx.cols / gSize + 1) * blockSizeX;
    size_t rows_per_thread = 2;
    size_t globalSizeY = ((Dx.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ?
                         ((Dx.rows + rows_per_thread - 1) / rows_per_thread) :
                         (((Dx.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;

    size_t globalsize[2] = { globalSizeX, globalSizeY }, localsize[2] = { blockSizeX, blockSizeY };
    return cornelKernel.run(2, globalsize, localsize, false);
}

static bool ocl_preCornerDetect( InputArray _src, OutputArray _dst, int ksize, int borderType, int depth )
{
    UMat Dx, Dy, D2x, D2y, Dxy;

    if (!extractCovData(_src, Dx, Dy, depth, 1, ksize, borderType))
        return false;

    Sobel( _src, D2x, CV_32F, 2, 0, ksize, 1, 0, borderType );
    Sobel( _src, D2y, CV_32F, 0, 2, ksize, 1, 0, borderType );
    Sobel( _src, Dxy, CV_32F, 1, 1, ksize, 1, 0, borderType );

    _dst.create( _src.size(), CV_32FC1 );
    UMat dst = _dst.getUMat();

    double factor = 1 << (ksize - 1);
    if( depth == CV_8U )
        factor *= 255;
    factor = 1./(factor * factor * factor);

    ocl::Kernel k("preCornerDetect", ocl::imgproc::precornerdetect_oclsrc);
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(Dx), ocl::KernelArg::ReadOnlyNoSize(Dy),
           ocl::KernelArg::ReadOnlyNoSize(D2x), ocl::KernelArg::ReadOnlyNoSize(D2y),
           ocl::KernelArg::ReadOnlyNoSize(Dxy), ocl::KernelArg::WriteOnly(dst), (float)factor);

    size_t globalsize[2] = { (size_t)dst.cols, (size_t)dst.rows };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

#if 0 //defined(HAVE_IPP)
namespace cv
{
static bool ipp_cornerMinEigenVal( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
#if IPP_VERSION_X100 >= 800
    CV_INSTRUMENT_REGION_IPP();

    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32FC1 );
    Mat dst = _dst.getMat();

    {
        typedef IppStatus (CV_STDCALL * ippiMinEigenValGetBufferSize)(IppiSize, int, int, int*);
        typedef IppStatus (CV_STDCALL * ippiMinEigenVal)(const void*, int, Ipp32f*, int, IppiSize, IppiKernelType, int, int, Ipp8u*);
        IppiKernelType kerType;
        int kerSize = ksize;
        if (ksize < 0)
        {
            kerType = ippKernelScharr;
            kerSize = 3;
        } else
        {
            kerType = ippKernelSobel;
        }
        bool isolated = (borderType & BORDER_ISOLATED) != 0;
        int borderTypeNI = borderType & ~BORDER_ISOLATED;
        if ((borderTypeNI == BORDER_REPLICATE && (!src.isSubmatrix() || isolated)) &&
            (kerSize == 3 || kerSize == 5) && (blockSize == 3 || blockSize == 5))
        {
            ippiMinEigenValGetBufferSize getBufferSizeFunc = 0;
            ippiMinEigenVal ippiMinEigenVal_C1R = 0;
            float norm_coef = 0.f;

            if (src.type() == CV_8UC1)
            {
                getBufferSizeFunc = (ippiMinEigenValGetBufferSize) ippiMinEigenValGetBufferSize_8u32f_C1R;
                ippiMinEigenVal_C1R = (ippiMinEigenVal) ippiMinEigenVal_8u32f_C1R;
                norm_coef = 1.f / 255.f;
            } else if (src.type() == CV_32FC1)
            {
                getBufferSizeFunc = (ippiMinEigenValGetBufferSize) ippiMinEigenValGetBufferSize_32f_C1R;
                ippiMinEigenVal_C1R = (ippiMinEigenVal) ippiMinEigenVal_32f_C1R;
                norm_coef = 255.f;
            }
            norm_coef = kerType == ippKernelSobel ? norm_coef : norm_coef / 2.45f;

            if (getBufferSizeFunc && ippiMinEigenVal_C1R)
            {
                int bufferSize;
                IppiSize srcRoi = { src.cols, src.rows };
                IppStatus ok = getBufferSizeFunc(srcRoi, kerSize, blockSize, &bufferSize);
                if (ok >= 0)
                {
                    AutoBuffer<uchar> buffer(bufferSize);
                    ok = CV_INSTRUMENT_FUN_IPP(ippiMinEigenVal_C1R, src.ptr(), (int) src.step, dst.ptr<Ipp32f>(), (int) dst.step, srcRoi, kerType, kerSize, blockSize, buffer.data());
                    CV_SUPPRESS_DEPRECATED_START
                    if (ok >= 0) ok = CV_INSTRUMENT_FUN_IPP(ippiMulC_32f_C1IR, norm_coef, dst.ptr<Ipp32f>(), (int) dst.step, srcRoi);
                    CV_SUPPRESS_DEPRECATED_END
                    if (ok >= 0)
                    {
                        CV_IMPL_ADD(CV_IMPL_IPP);
                        return true;
                    }
                }
            }
        }
    }
#else
    CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(blockSize); CV_UNUSED(borderType);
#endif
    return false;
}
}
#endif

void cv::cornerMinEigenVal( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_cornerMinEigenValVecs(_src, _dst, blockSize, ksize, 0.0, borderType, MINEIGENVAL))

/*#ifdef HAVE_IPP
    int kerSize = (ksize < 0)?3:ksize;
    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    int borderTypeNI = borderType & ~BORDER_ISOLATED;
#endif
    CV_IPP_RUN(((borderTypeNI == BORDER_REPLICATE && (!_src.isSubmatrix() || isolated)) &&
            (kerSize == 3 || kerSize == 5) && (blockSize == 3 || blockSize == 5)) && IPP_VERSION_X100 >= 800,
        ipp_cornerMinEigenVal( _src, _dst, blockSize, ksize, borderType ));
    */

    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32FC1 );
    Mat dst = _dst.getMat();

    cornerEigenValsVecs( src, dst, blockSize, ksize, MINEIGENVAL, 0, borderType );
}


#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_cornerHarris( Mat &src, Mat &dst, int blockSize, int ksize, double k, int borderType )
{
#if IPP_VERSION_X100 >= 810
    CV_INSTRUMENT_REGION_IPP();

    {
        int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
        int borderTypeNI = borderType & ~BORDER_ISOLATED;
        bool isolated = (borderType & BORDER_ISOLATED) != 0;

        if ( (ksize == 3 || ksize == 5) && (type == CV_8UC1 || type == CV_32FC1) &&
            (borderTypeNI == BORDER_CONSTANT || borderTypeNI == BORDER_REPLICATE) && cn == 1 && (!src.isSubmatrix() || isolated) )
        {
            IppiSize roisize = { src.cols, src.rows };
            IppiMaskSize masksize = ksize == 5 ? ippMskSize5x5 : ippMskSize3x3;
            IppDataType datatype = type == CV_8UC1 ? ipp8u : ipp32f;
            Ipp32s bufsize = 0;

            double scale = (double)(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;
            if (ksize < 0)
                scale *= 2.0;
            if (depth == CV_8U)
                scale *= 255.0;
            scale = std::pow(scale, -4.0);

            if (ippiHarrisCornerGetBufferSize(roisize, masksize, blockSize, datatype, cn, &bufsize) >= 0)
            {
                Ipp8u * buffer = (Ipp8u*)CV_IPP_MALLOC(bufsize);
                IppiDifferentialKernel filterType = ksize > 0 ? ippFilterSobel : ippFilterScharr;
                IppiBorderType borderTypeIpp = borderTypeNI == BORDER_CONSTANT ? ippBorderConst : ippBorderRepl;
                IppStatus status = (IppStatus)-1;

                if (depth == CV_8U)
                    status = CV_INSTRUMENT_FUN_IPP(ippiHarrisCorner_8u32f_C1R, (const Ipp8u *)src.data, (int)src.step, (Ipp32f *)dst.data, (int)dst.step, roisize,
                        filterType, masksize, blockSize, (Ipp32f)k, (Ipp32f)scale, borderTypeIpp, 0, buffer);
                else if (depth == CV_32F)
                    status = CV_INSTRUMENT_FUN_IPP(ippiHarrisCorner_32f_C1R, (const Ipp32f *)src.data, (int)src.step, (Ipp32f *)dst.data, (int)dst.step, roisize,
                        filterType, masksize, blockSize, (Ipp32f)k, (Ipp32f)scale, borderTypeIpp, 0, buffer);
                ippsFree(buffer);

                if (status >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(blockSize);  CV_UNUSED(ksize); CV_UNUSED(k); CV_UNUSED(borderType);
#endif
    return false;
}
}
#endif

void cv::cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_cornerMinEigenValVecs(_src, _dst, blockSize, ksize, k, borderType, HARRIS))

    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32FC1 );
    Mat dst = _dst.getMat();

#ifdef HAVE_IPP
    int borderTypeNI = borderType & ~BORDER_ISOLATED;
    bool isolated = (borderType & BORDER_ISOLATED) != 0;
#endif
    CV_IPP_RUN(((ksize == 3 || ksize == 5) && (_src.type() == CV_8UC1 || _src.type() == CV_32FC1) &&
        (borderTypeNI == BORDER_CONSTANT || borderTypeNI == BORDER_REPLICATE) && CV_MAT_CN(_src.type()) == 1 &&
        (!_src.isSubmatrix() || isolated)) && IPP_VERSION_X100 >= 810, ipp_cornerHarris( src, dst, blockSize, ksize, k, borderType ));

    cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
}


void cv::cornerEigenValsAndVecs( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    Size dsz = _dst.size();
    int dtype = _dst.type();

    if( dsz.height != src.rows || dsz.width*CV_MAT_CN(dtype) != src.cols*6 || CV_MAT_DEPTH(dtype) != CV_32F )
        _dst.create( src.size(), CV_32FC(6) );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, EIGENVALSVECS, 0, borderType );
}


void cv::preCornerDetect( InputArray _src, OutputArray _dst, int ksize, int borderType )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type();
    CV_Assert( type == CV_8UC1 || type == CV_32FC1 );

    CV_OCL_RUN( _src.dims() <= 2 && _dst.isUMat(),
                ocl_preCornerDetect(_src, _dst, ksize, borderType, CV_MAT_DEPTH(type)))

    Mat Dx, Dy, D2x, D2y, Dxy, src = _src.getMat();
    _dst.create( src.size(), CV_32FC1 );
    Mat dst = _dst.getMat();

    Sobel( src, Dx, CV_32F, 1, 0, ksize, 1, 0, borderType );
    Sobel( src, Dy, CV_32F, 0, 1, ksize, 1, 0, borderType );
    Sobel( src, D2x, CV_32F, 2, 0, ksize, 1, 0, borderType );
    Sobel( src, D2y, CV_32F, 0, 2, ksize, 1, 0, borderType );
    Sobel( src, Dxy, CV_32F, 1, 1, ksize, 1, 0, borderType );

    double factor = 1 << (ksize - 1);
    if( src.depth() == CV_8U )
        factor *= 255;
    factor = 1./(factor * factor * factor);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    float factor_f = (float)factor;
    v_float32 v_factor = vx_setall_f32(factor_f), v_m2 = vx_setall_f32(-2.0f);
#endif

    Size size = src.size();
    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        float* dstdata = dst.ptr<float>(i);
        const float* dxdata = Dx.ptr<float>(i);
        const float* dydata = Dy.ptr<float>(i);
        const float* d2xdata = D2x.ptr<float>(i);
        const float* d2ydata = D2y.ptr<float>(i);
        const float* dxydata = Dxy.ptr<float>(i);

        j = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            for( ; j <= size.width - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes() )
            {
                v_float32 v_dx = vx_load(dxdata + j);
                v_float32 v_dy = vx_load(dydata + j);

                v_float32 v_s1 = v_mul(v_mul(v_dx, v_dx), vx_load(d2ydata + j));
                v_float32 v_s2 = v_muladd((v_mul(v_dy, v_dy)),  vx_load(d2xdata + j), v_s1);
                v_float32 v_s3 = v_muladd(v_mul(v_mul(v_dy, v_dx), vx_load(dxydata + j)), v_m2, v_s2);

                v_store(dstdata + j, v_mul(v_s3, v_factor));
            }
        }
#endif

        for( ; j < size.width; j++ )
        {
            float dx = dxdata[j];
            float dy = dydata[j];
            dstdata[j] = (float)(factor*(dx*dx*d2ydata[j] + dy*dy*d2xdata[j] - 2*dx*dy*dxydata[j]));
        }
    }
}

CV_IMPL void
cvCornerMinEigenVal( const CvArr* srcarr, CvArr* dstarr,
                     int block_size, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::cornerMinEigenVal( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}

CV_IMPL void
cvCornerHarris( const CvArr* srcarr, CvArr* dstarr,
                int block_size, int aperture_size, double k )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::cornerHarris( src, dst, block_size, aperture_size, k, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvCornerEigenValsAndVecs( const void* srcarr, void* dstarr,
                          int block_size, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.rows && src.cols*6 == dst.cols*dst.channels() && dst.depth() == CV_32F );
    cv::cornerEigenValsAndVecs( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvPreCornerDetect( const void* srcarr, void* dstarr, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::preCornerDetect( src, dst, aperture_size, cv::BORDER_REPLICATE );
}

/* End of file */
