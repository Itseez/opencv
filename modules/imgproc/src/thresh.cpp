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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv
{

template <typename T>
static inline T threshBinary(const T& src, const T& thresh, const T& maxval)
{
    return src > thresh ? maxval : 0;
}

template <typename T>
static inline T threshBinaryInv(const T& src, const T& thresh, const T& maxval)
{
    return src <= thresh ? maxval : 0;
}

template <typename T>
static inline T threshTrunc(const T& src, const T& thresh)
{
    return std::min(src, thresh);
}

template <typename T>
static inline T threshToZero(const T& src, const T& thresh)
{
    return src > thresh ? src : 0;
}

template <typename T>
static inline T threshToZeroInv(const T& src, const T& thresh)
{
    return src <= thresh ? src : 0;
}

template <typename T, bool useMask>
static void threshGeneric(Size roi, const int cn, const T* src, size_t src_step,
                          T* dst, size_t dst_step,
                          const uchar* mask, size_t mask_step,
                          T thresh, T maxval, int type)
{
    int i = 0, j;
    switch (type)
    {
    case THRESH_BINARY:
        for (; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
            for (j = 0; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinary<T>(src[j], thresh, maxval);
        return;

    case THRESH_BINARY_INV:
        for (; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
            for (j = 0; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinaryInv<T>(src[j], thresh, maxval);
        return;

    case THRESH_TRUNC:
        for (; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
            for (j = 0; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshTrunc<T>(src[j], thresh);
        return;

    case THRESH_TOZERO:
        for (; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
            for (j = 0; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZero<T>(src[j], thresh);
        return;

    case THRESH_TOZERO_INV:
        for (; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
            for (j = 0; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZeroInv<T>(src[j], thresh);
        return;

    default:
        CV_Error( cv::Error::StsBadArg, "" ); return;
    }
}

template<bool useMask>
static void
thresh_8u( const Mat& _src, Mat& _dst, const Mat& _mask, uchar thresh, uchar maxval, int type )
{
    Size roi = _src.size();
    const int cn = _src.channels();
    roi.width *= cn;
    size_t src_step = _src.step;
    size_t dst_step = _dst.step;
    size_t mask_step = _mask.step;

    if( _src.isContinuous() && _dst.isContinuous() && (!useMask || _mask.empty() || _mask.isContinuous()) )
    {
        roi.width *= roi.height;
        roi.height = 1;
        src_step = dst_step = roi.width;
        if (useMask)
            mask_step = _mask.total();
    }

    if (!useMask)
    {
        #if defined(HAVE_IPP)
        CV_IPP_CHECK()
        {
            IppiSize sz = { roi.width, roi.height };
            CV_SUPPRESS_DEPRECATED_START
            switch( type )
            {
            case THRESH_TRUNC:
                if (_src.data == _dst.data && CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_8u_C1IR, _dst.ptr(), (int)dst_step, sz, thresh) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                if (CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_8u_C1R, _src.ptr(), (int)src_step, _dst.ptr(), (int)dst_step, sz, thresh) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            case THRESH_TOZERO:
                if (_src.data == _dst.data && CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_8u_C1IR, _dst.ptr(), (int)dst_step, sz, thresh+1, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                if (CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_8u_C1R, _src.ptr(), (int)src_step, _dst.ptr(), (int)dst_step, sz, thresh + 1, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            case THRESH_TOZERO_INV:
                if (_src.data == _dst.data && CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_8u_C1IR, _dst.ptr(), (int)dst_step, sz, thresh, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                if (CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_8u_C1R, _src.ptr(), (int)src_step, _dst.ptr(), (int)dst_step, sz, thresh, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            }
            CV_SUPPRESS_DEPRECATED_END
        }
        #endif
    }

    int j = 0;
    const uchar* src = _src.ptr();
    uchar* dst = _dst.ptr();
    const uchar* mask = nullptr;
    if (useMask)
    {
        mask = _mask.ptr();
    }

    #if (CV_SIMD || CV_SIMD_SCALABLE)
    v_uint8 thresh_u = vx_setall_u8( thresh );
    v_uint8 maxval16 = vx_setall_u8( maxval );

    v_uint8 vm0;
    int maskReadLUT[16];
    int maskRepeat = 0;
    if (useMask)
        maskRepeat = static_cast<int>(sizeof(uint8_t)*cn);
    switch( type )
    {
    case THRESH_BINARY:
        for( int i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            const int srcReadLanes = VTraits<v_uint8>::vlanes();
            const int maskReadLanes = !useMask ? 0 : VTraits<v_uint8>::vlanes();
            for( j = 0; j <= roi.width - srcReadLanes; j += srcReadLanes)
            {
                v_uint8 v0, v2;
                v0 = vx_load( src + j );
                if (useMask)
                {
                    if (cn == 1)
                        vm0 = v_load(mask + j);
                    else
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                    }
                    v2 = v_and(v0, v_not(vm0));
                }
                v0 = v_lt(thresh_u, v0);
                v0 = v_and(v0, maxval16);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, vm0), v2);
                }
                v_store( dst + j, v0 );
            }
        }
        break;

    case THRESH_BINARY_INV:
        for( int i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            const int srcReadLanes = VTraits<v_uint8>::vlanes();
            const int maskReadLanes = !useMask ? 0 : VTraits<v_uint8>::vlanes();
            for( j = 0; j <= roi.width - srcReadLanes; j += srcReadLanes)
            {
                v_uint8 v0, v2;
                v0 = vx_load( src + j );
                if (useMask)
                {
                    if (cn == 1)
                        vm0 = v_load(mask + j);
                    else
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                    }
                    v2 = v_and(v0, v_not(vm0));
                }
                v0 = v_le(v0, thresh_u);
                v0 = v_and(v0, maxval16);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, vm0), v2);
                }
                v_store( dst + j, v0 );
            }
        }
        break;

    case THRESH_TRUNC:
        for( int i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            const int srcReadLanes = VTraits<v_uint8>::vlanes();
            const int maskReadLanes = !useMask ? 0 : VTraits<v_uint8>::vlanes();
            for( j = 0; j <= roi.width - srcReadLanes; j += srcReadLanes)
            {
                v_uint8 v0, v2;
                v0 = vx_load( src + j );
                if (useMask)
                {
                    if (cn == 1)
                        vm0 = v_load(mask + j);
                    else
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                    }
                    v2 = v_and(v0, v_not(vm0));
                }
                v0 = v_sub(v0, v_sub(v0, thresh_u));
                if (useMask)
                {
                    v0 = v_or(v_and(v0, vm0), v2);
                }
                v_store( dst + j, v0 );
            }
        }
        break;

    case THRESH_TOZERO:
        for( int i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            const int srcReadLanes = VTraits<v_uint8>::vlanes();
            const int maskReadLanes = !useMask ? 0 : VTraits<v_uint8>::vlanes();
            for( j = 0; j <= roi.width - srcReadLanes; j += srcReadLanes)
            {
                v_uint8 v0, v2;
                v0 = vx_load( src + j );
                if (useMask)
                {
                    if (cn == 1)
                        vm0 = v_load(mask + j);
                    else
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                    }
                    v2 = v_and(v0, v_not(vm0));
                }
                v0 = v_and(v_lt(thresh_u, v0), v0);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, vm0), v2);
                }
                v_store( dst + j, v0 );
            }
        }
        break;

    case THRESH_TOZERO_INV:
        for( int i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            const int srcReadLanes = VTraits<v_uint8>::vlanes();
            const int maskReadLanes = !useMask ? 0 : VTraits<v_uint8>::vlanes();
            for( j = 0; j <= roi.width - srcReadLanes; j += srcReadLanes)
            {
                v_uint8 v0, v2;
                v0 = vx_load( src + j );
                if (useMask)
                {
                    if (cn == 1)
                        vm0 = v_load(mask + j);
                    else
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                    }
                    v2 = v_and(v0, v_not(vm0));
                }
                v0 = v_and(v_le(v0, thresh_u), v0);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, vm0), v2);
                }
                v_store( dst + j, v0 );
            }
        }
        break;
    }
#endif

    int j_scalar = j;
    if( j_scalar < roi.width )
    {
        const int thresh_pivot = thresh + 1;
        uchar tab[256] = {0};
        switch( type )
        {
        case THRESH_BINARY:
            memset(tab, 0, thresh_pivot);
            if (thresh_pivot < 256) {
                memset(tab + thresh_pivot, maxval, 256 - thresh_pivot);
            }
            break;
        case THRESH_BINARY_INV:
            memset(tab, maxval, thresh_pivot);
            if (thresh_pivot < 256) {
                memset(tab + thresh_pivot, 0, 256 - thresh_pivot);
            }
            break;
        case THRESH_TRUNC:
            for( int i = 0; i <= thresh; i++ )
                tab[i] = (uchar)i;
            if (thresh_pivot < 256) {
                memset(tab + thresh_pivot, thresh, 256 - thresh_pivot);
            }
            break;
        case THRESH_TOZERO:
            memset(tab, 0, thresh_pivot);
            for( int i = thresh_pivot; i < 256; i++ )
                tab[i] = (uchar)i;
            break;
        case THRESH_TOZERO_INV:
            for( int i = 0; i <= thresh; i++ )
                tab[i] = (uchar)i;
            if (thresh_pivot < 256) {
                memset(tab + thresh_pivot, 0, 256 - thresh_pivot);
            }
            break;
        }

        src = _src.ptr();
        dst = _dst.ptr();
        if (useMask)
          mask = _mask.ptr();
        for( int i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = j_scalar;
#if CV_ENABLE_UNROLLED
            for( ; j <= roi.width - 4; j += 4 )
            {
                uchar t0 = tab[src[j]];
                uchar t1 = tab[src[j+1]];
                uchar m0 = useMask ? mask[j/cn] : 0xFF;
                uchar m1 = useMask ? mask[(j+1)/cn] : 0xFF;

                if (!useMask || (m0 != 0))
                    dst[j] = t0;
                if (!useMask || (m1 != 0))
                    dst[j+1] = t1;

                t0 = tab[src[j+2]];
                t1 = tab[src[j+3]];
                m0 = useMask ? mask[(j+2)/cn] : 0xFF;
                m1 = useMask ? mask[(j+3)/cn] : 0xFF;

                if (!useMask || (m0 != 0))
                    dst[j+2] = t0;
                if (!useMask || (m1 != 0))
                    dst[j+3] = t1;
            }
#endif
            for( ; j < roi.width; j++ )
            {
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = tab[src[j]];
            }
        }
    }
}

template<bool useMask>
static void
thresh_16u(const Mat& _src, Mat& _dst, const Mat& _mask, ushort thresh, ushort maxval, int type)
{
    Size roi = _src.size();
    const int cn = _src.channels();
    roi.width *= cn;
    size_t src_step = _src.step / _src.elemSize1();
    size_t dst_step = _dst.step / _dst.elemSize1();

    size_t mask_step = _mask.step;

    if( _src.isContinuous() && _dst.isContinuous() && (!useMask || _mask.empty() || _mask.isContinuous()) )
    {
        roi.width *= roi.height;
        roi.height = 1;
        src_step = dst_step = roi.width;
        if (useMask)
            mask_step = _mask.total();
    }

    // HAVE_IPP not supported

    const ushort* src = _src.ptr<ushort>();
    ushort* dst = _dst.ptr<ushort>();
    const uchar* mask = nullptr;
    if (useMask)
        mask = _mask.ptr();

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int i, j;
    v_uint16 thresh_u = vx_setall_u16(thresh);
    v_uint16 maxval16 = vx_setall_u16(maxval);

    v_uint8 vm0,vm1;
    int maskReadLUT[32];
    int maskRepeat = 0;
    if (useMask)
        maskRepeat = static_cast<int>(sizeof(uint16_t)*cn);
    switch( type )
    {
    case THRESH_BINARY:
        for (i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_uint16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_uint16 v0, v1, v2, v3;
                v0 = vx_load(src + j);
                v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_u16(v_not(vm1)));
                }
                v0 = v_lt(thresh_u, v0);
                v1 = v_lt(thresh_u, v1);
                v0 = v_and(v0, maxval16);
                v1 = v_and(v1, maxval16);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_u16(vm1)), v3);
                }
                v_store(dst + j, v0);
                v_store(dst + j + VTraits<v_uint16>::vlanes(), v1);
            }
            if ( j <= roi.width - VTraits<v_uint16>::vlanes() )
            {
                v_uint16 v0 = vx_load(src + j);
                v_uint16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                }
                v0 = v_lt(thresh_u, v0);
                v0 = v_and(v0, maxval16);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                v_store(dst + j, v0);
                j += VTraits<v_uint16>::vlanes();
            }

            for (; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinary<ushort>(src[j], thresh, maxval);
        }
        break;

    case THRESH_BINARY_INV:
        for (i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_uint16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_uint16 v0, v1, v2, v3;
                v0 = vx_load(src + j);
                v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_u16(v_not(vm1)));
                }
                v0 = v_le(v0, thresh_u);
                v1 = v_le(v1, thresh_u);
                v0 = v_and(v0, maxval16);
                v1 = v_and(v1, maxval16);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_u16(vm1)), v3);
                }
                v_store(dst + j, v0);
                v_store(dst + j + VTraits<v_uint16>::vlanes(), v1);
            }
            if ( j <= roi.width - VTraits<v_uint16>::vlanes() )
            {
                v_uint16 v0 = vx_load(src + j);
                v_uint16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                }
                v0 = v_le(v0, thresh_u);
                v0 = v_and(v0, maxval16);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                v_store(dst + j, v0);
                j += VTraits<v_uint16>::vlanes();
            }

            for (; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinaryInv<ushort>(src[j], thresh, maxval);
        }
        break;

    case THRESH_TRUNC:
        for (i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_uint16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_uint16 v0, v1, v2, v3;
                v0 = vx_load(src + j);
                v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_u16(v_not(vm1)));
                }
                v0 = v_min(v0, thresh_u);
                v1 = v_min(v1, thresh_u);
                if (useMask)
                {
                    v0 = v_or(v_and(v0,v_reinterpret_as_u16(vm0)), v2);
                    v1 = v_or(v_and(v1,v_reinterpret_as_u16(vm1)), v3);
                }
                v_store(dst + j, v0);
                v_store(dst + j + VTraits<v_uint16>::vlanes(), v1);
            }
            if ( j <= roi.width - VTraits<v_uint16>::vlanes() )
            {
                v_uint16 v0 = vx_load(src + j);
                v_uint16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                }
                v0 = v_min(v0, thresh_u);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                v_store(dst + j, v0);
                j += VTraits<v_uint16>::vlanes();
            }

            for (; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshTrunc<ushort>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO:
        for (i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_uint16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_uint16 v0, v1, v2, v3;
                v0 = vx_load(src + j);
                v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_u16(v_not(vm1)));
                }
                v0 = v_and(v_lt(thresh_u, v0), v0);
                v1 = v_and(v_lt(thresh_u, v1), v1);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_u16(vm1)), v3);
                }
                v_store(dst + j, v0);
                v_store(dst + j + VTraits<v_uint16>::vlanes(), v1);
            }
            if ( j <= roi.width - VTraits<v_uint16>::vlanes() )
            {
                v_uint16 v0 = vx_load(src + j);
                v_uint16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                }
                v0 = v_and(v_lt(thresh_u, v0), v0);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                v_store(dst + j, v0);
                j += VTraits<v_uint16>::vlanes();
            }

            for (; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZero<ushort>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO_INV:
        for (i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0)
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_uint16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_uint16 v0, v1, v2, v3;
                v0 = vx_load(src + j);
                v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_u16(v_not(vm1)));
                }
                v0 = v_and(v_le(v0, thresh_u), v0);
                v1 = v_and(v_le(v1, thresh_u), v1);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_u16(vm1)), v3);
                }
                v_store(dst + j, v0);
                v_store(dst + j + VTraits<v_uint16>::vlanes(), v1);
            }
            if ( j <= roi.width - VTraits<v_uint16>::vlanes() )
            {
                v_uint16 v0 = vx_load(src + j);
                v_uint16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_u16(v_not(vm0)));
                }
                v0 = v_and(v_le(v0, thresh_u), v0);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_u16(vm0)), v2);
                v_store(dst + j, v0);
                j += VTraits<v_uint16>::vlanes();
            }

            for (; j < roi.width; j++)
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZeroInv<ushort>(src[j], thresh);
        }
        break;
    }
#else
    threshGeneric<ushort, useMask>(roi, cn, src, src_step, dst, dst_step, mask, mask_step, thresh, maxval, type);
#endif
}

template<bool useMask>
static void
thresh_16s( const Mat& _src, Mat& _dst, const Mat& _mask, short thresh, short maxval, int type )
{
    Size roi = _src.size();
    const int cn = _src.channels();
    roi.width *= cn;
    const short* src = _src.ptr<short>();
    short* dst = _dst.ptr<short>();
    size_t src_step = _src.step/sizeof(src[0]);
    size_t dst_step = _dst.step/sizeof(dst[0]);
    size_t mask_step = _mask.step;

    if( _src.isContinuous() && _dst.isContinuous() && (!useMask || _mask.empty() || _mask.isContinuous()) )
    {
        roi.width *= roi.height;
        roi.height = 1;
        src_step = dst_step = roi.width;
        if (useMask)
            mask_step = _mask.total();
    }

    const uchar* mask = nullptr;
    if (useMask)
        mask = _mask.ptr();

    if (!useMask)
    {
        #if defined(HAVE_IPP)
        CV_IPP_CHECK()
        {
            IppiSize sz = { roi.width, roi.height };
            CV_SUPPRESS_DEPRECATED_START
            switch( type )
            {
            case THRESH_TRUNC:
                if (_src.data == _dst.data && CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_16s_C1IR, dst, (int)dst_step*sizeof(dst[0]), sz, thresh) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                if (CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_16s_C1R, src, (int)src_step*sizeof(src[0]), dst, (int)dst_step*sizeof(dst[0]), sz, thresh) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            case THRESH_TOZERO:
                if (_src.data == _dst.data && CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_16s_C1IR, dst, (int)dst_step*sizeof(dst[0]), sz, thresh + 1, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                if (CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_16s_C1R, src, (int)src_step*sizeof(src[0]), dst, (int)dst_step*sizeof(dst[0]), sz, thresh + 1, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            case THRESH_TOZERO_INV:
                if (_src.data == _dst.data && CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_16s_C1IR, dst, (int)dst_step*sizeof(dst[0]), sz, thresh, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                if (CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_16s_C1R, src, (int)src_step*sizeof(src[0]), dst, (int)dst_step*sizeof(dst[0]), sz, thresh, 0) >= 0)
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            }
            CV_SUPPRESS_DEPRECATED_END
        }
        #endif
    }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int i, j;
    v_int16 thresh8 = vx_setall_s16( thresh );
    v_int16 maxval8 = vx_setall_s16( maxval );

    v_uint8 vm0, vm1;
    int maskReadLUT[32];
    int maskRepeat = 0;
    if (useMask)
        maskRepeat = static_cast<int>(sizeof(int16_t)*cn);
    switch( type )
    {
    case THRESH_BINARY:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_int16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_int16 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_int16>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_s16(v_not(vm1)));
                }
                v0 = v_lt(thresh8, v0);
                v1 = v_lt(thresh8, v1);
                v0 = v_and(v0, maxval8);
                v1 = v_and(v1, maxval8);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_s16(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_int16>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_int16>::vlanes() )
            {
                v_int16 v0 = vx_load( src + j );
                v_int16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                }
                v0 = v_lt(thresh8, v0);
                v0 = v_and(v0, maxval8);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_int16>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinary<short>(src[j], thresh, maxval);
        }
        break;

    case THRESH_BINARY_INV:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_int16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_int16 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_int16>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_s16(v_not(vm1)));
                }
                v0 = v_le(v0, thresh8);
                v1 = v_le(v1, thresh8);
                v0 = v_and(v0, maxval8);
                v1 = v_and(v1, maxval8);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_s16(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_int16>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_int16>::vlanes() )
            {
                v_int16 v0 = vx_load( src + j );
                v_int16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                }
                v0 = v_le(v0, thresh8);
                v0 = v_and(v0, maxval8);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_int16>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinaryInv<short>(src[j], thresh, maxval);
        }
        break;

    case THRESH_TRUNC:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_int16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_int16 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_int16>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_s16(v_not(vm1)));
                }
                v0 = v_min( v0, thresh8 );
                v1 = v_min( v1, thresh8 );
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_s16(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_int16>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_int16>::vlanes() )
            {
                v_int16 v0 = vx_load( src + j );
                v_int16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                }
                v0 = v_min( v0, thresh8 );
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_int16>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshTrunc<short>( src[j], thresh );
        }
        break;

    case THRESH_TOZERO:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_int16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_int16 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_int16>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_s16(v_not(vm1)));
                }
                v0 = v_and(v_lt(thresh8, v0), v0);
                v1 = v_and(v_lt(thresh8, v1), v1);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_s16(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_int16>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_int16>::vlanes() )
            {
                v_int16 v0 = vx_load( src + j );
                v_int16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                }
                v0 = v_and(v_lt(thresh8, v0), v0);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_int16>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZero<short>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO_INV:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_int16>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_int16 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_int16>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_s16(v_not(vm1)));
                }
                v0 = v_and(v_le(v0, thresh8), v0);
                v1 = v_and(v_le(v1, thresh8), v1);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_s16(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_int16>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_int16>::vlanes() )
            {
                v_int16 v0 = vx_load( src + j );
                v_int16 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_s16(v_not(vm0)));
                }
                v0 = v_and(v_le(v0, thresh8), v0);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_s16(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_int16>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZeroInv<short>(src[j], thresh);
        }
        break;
    default:
        CV_Error( cv::Error::StsBadArg, "" ); return;
    }
#else
    threshGeneric<short, useMask>(roi, cn, src, src_step, dst, dst_step, mask, mask_step, thresh, maxval, type);
#endif
}

template<bool useMask>
static void
thresh_32f( const Mat& _src, Mat& _dst, const Mat& _mask, float thresh, float maxval, int type )
{
    Size roi = _src.size();
    const int cn = _src.channels();
    roi.width *= cn;
    const float* src = _src.ptr<float>();
    float* dst = _dst.ptr<float>();
    size_t src_step = _src.step/sizeof(src[0]);
    size_t dst_step = _dst.step/sizeof(dst[0]);
    size_t mask_step = _mask.step;

    if( _src.isContinuous() && _dst.isContinuous() && (!useMask || _mask.empty() || _mask.isContinuous()) )
    {
        roi.width *= roi.height;
        roi.height = 1;
        if (useMask)
            mask_step = _mask.total();
    }

    const uchar* mask = nullptr;
    if (useMask)
        mask = _mask.ptr();

    if (!useMask)
    {
        #if defined(HAVE_IPP)
        CV_IPP_CHECK()
        {
            IppiSize sz = { roi.width, roi.height };
            switch( type )
            {
            case THRESH_TRUNC:
                if (0 <= CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_32f_C1R, src, (int)src_step*sizeof(src[0]), dst, (int)dst_step*sizeof(dst[0]), sz, thresh))
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            case THRESH_TOZERO:
                if (0 <= CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_32f_C1R, src, (int)src_step*sizeof(src[0]), dst, (int)dst_step*sizeof(dst[0]), sz, nextafterf(thresh, std::numeric_limits<float>::infinity()), 0))
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            case THRESH_TOZERO_INV:
                if (0 <= CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_32f_C1R, src, (int)src_step*sizeof(src[0]), dst, (int)dst_step*sizeof(dst[0]), sz, thresh, 0))
                {
                    CV_IMPL_ADD(CV_IMPL_IPP);
                    return;
                }
                setIppErrorStatus();
                break;
            }
        }
        #endif
    }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int i, j;
    v_float32 thresh4 = vx_setall_f32( thresh );
    v_float32 maxval4 = vx_setall_f32( maxval );

    v_uint8 vm0, vm1;
    int maskReadLUT[32];
    int maskRepeat = 0;
    if (useMask)
        maskRepeat = static_cast<int>(sizeof(float)*cn);
    switch( type )
    {
        case THRESH_BINARY:
            for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
            {
                j = 0;
                const int srcReadLanes = 2*VTraits<v_float32>::vlanes();
                const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
                for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
                {
                    v_float32 v0, v1, v2, v3;
                    v0 = vx_load( src + j );
                    v1 = vx_load( src + j + VTraits<v_float32>::vlanes() );
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        vm1 = vx_lut(mask, maskReadLUT+16);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                        v3 = v_and(v1, v_reinterpret_as_f32(v_not(vm1)));
                    }
                    v0 = v_lt(thresh4, v0);
                    v1 = v_lt(thresh4, v1);
                    v0 = v_and(v0, maxval4);
                    v1 = v_and(v1, maxval4);
                    if (useMask)
                    {
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                        v1 = v_or(v_and(v1, v_reinterpret_as_f32(vm1)), v3);
                    }
                    v_store( dst + j, v0 );
                    v_store( dst + j + VTraits<v_float32>::vlanes(), v1 );
                }
                if ( j <= roi.width - VTraits<v_float32>::vlanes() )
                {
                    v_float32 v0 = vx_load( src + j );
                    v_float32 v2;
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                    }
                    v0 = v_lt(thresh4, v0);
                    v0 = v_and(v0, maxval4);
                    if (useMask)
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                    v_store( dst + j, v0 );
                    j += VTraits<v_float32>::vlanes();
                }

                for( ; j < roi.width; j++ )
                    if (!useMask || (mask[j/cn] != 0))
                        dst[j] = threshBinary<float>(src[j], thresh, maxval);
            }
            break;

        case THRESH_BINARY_INV:
            for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
            {
                j = 0;
                const int srcReadLanes = 2*VTraits<v_float32>::vlanes();
                const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
                for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
                {
                    v_float32 v0, v1, v2, v3;
                    v0 = vx_load( src + j );
                    v1 = vx_load( src + j + VTraits<v_float32>::vlanes() );
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        vm1 = vx_lut(mask, maskReadLUT+16);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                        v3 = v_and(v1, v_reinterpret_as_f32(v_not(vm1)));
                    }
                    v0 = v_le(v0, thresh4);
                    v1 = v_le(v1, thresh4);
                    v0 = v_and(v0, maxval4);
                    v1 = v_and(v1, maxval4);
                    if (useMask)
                    {
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                        v1 = v_or(v_and(v1, v_reinterpret_as_f32(vm1)), v3);
                    }
                    v_store( dst + j, v0 );
                    v_store( dst + j + VTraits<v_float32>::vlanes(), v1 );
                }
                if ( j <= roi.width - VTraits<v_float32>::vlanes() )
                {
                    v_float32 v0 = vx_load( src + j );
                    v_float32 v2;
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                    }
                    v0 = v_le(v0, thresh4);
                    v0 = v_and(v0, maxval4);
                    if (useMask)
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                    v_store( dst + j, v0 );
                    j += VTraits<v_float32>::vlanes();
                }

                for( ; j < roi.width; j++ )
                    if (!useMask || (mask[j/cn] != 0))
                        dst[j] = threshBinaryInv<float>(src[j], thresh, maxval);
            }
            break;

        case THRESH_TRUNC:
            for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
            {
                j = 0;
                const int srcReadLanes = 2*VTraits<v_float32>::vlanes();
                const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
                for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
                {
                    v_float32 v0, v1, v2, v3;
                    v0 = vx_load( src + j );
                    v1 = vx_load( src + j + VTraits<v_float32>::vlanes() );
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        vm1 = vx_lut(mask, maskReadLUT+16);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                        v3 = v_and(v1, v_reinterpret_as_f32(v_not(vm1)));
                    }
                    v0 = v_min( v0, thresh4 );
                    v1 = v_min( v1, thresh4 );
                    if (useMask)
                    {
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                        v1 = v_or(v_and(v1, v_reinterpret_as_f32(vm1)), v3);
                    }
                    v_store( dst + j, v0 );
                    v_store( dst + j + VTraits<v_float32>::vlanes(), v1 );
                }
                if ( j <= roi.width - VTraits<v_float32>::vlanes() )
                {
                    v_float32 v0 = vx_load( src + j );
                    v_float32 v2;
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                    }
                    v0 = v_min( v0, thresh4 );
                    if (useMask)
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                    v_store( dst + j, v0 );
                    j += VTraits<v_float32>::vlanes();
                }

                for( ; j < roi.width; j++ )
                    if (!useMask || (mask[j/cn] != 0))
                        dst[j] = threshTrunc<float>(src[j], thresh);
            }
            break;

        case THRESH_TOZERO:
            for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
            {
                j = 0;
                const int srcReadLanes = 2*VTraits<v_float32>::vlanes();
                const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
                for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
                {
                    v_float32 v0, v1, v2, v3;
                    v0 = vx_load( src + j );
                    v1 = vx_load( src + j + VTraits<v_float32>::vlanes() );
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        vm1 = vx_lut(mask, maskReadLUT+16);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                        v3 = v_and(v1, v_reinterpret_as_f32(v_not(vm1)));
                    }
                    v0 = v_and(v_lt(thresh4, v0), v0);
                    v1 = v_and(v_lt(thresh4, v1), v1);
                    if (useMask)
                    {
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                        v1 = v_or(v_and(v1, v_reinterpret_as_f32(vm1)), v3);
                    }
                    v_store( dst + j, v0 );
                    v_store( dst + j + VTraits<v_float32>::vlanes(), v1 );
                }
                if ( j <= roi.width - VTraits<v_float32>::vlanes() )
                {
                    v_float32 v0 = vx_load( src + j );
                    v_float32 v2;
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                    }
                    v0 = v_and(v_lt(thresh4, v0), v0);
                    if (useMask)
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                    v_store( dst + j, v0 );
                    j += VTraits<v_float32>::vlanes();
                }

                for( ; j < roi.width; j++ )
                    if (!useMask || (mask[j/cn] != 0))
                        dst[j] = threshToZero<float>(src[j], thresh);
            }
            break;

        case THRESH_TOZERO_INV:
            for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
            {
                j = 0;
                const int srcReadLanes = 2*VTraits<v_float32>::vlanes();
                const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
                for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
                {
                    v_float32 v0, v1, v2, v3;
                    v0 = vx_load( src + j );
                    v1 = vx_load( src + j + VTraits<v_float32>::vlanes() );
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        vm1 = vx_lut(mask, maskReadLUT+16);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                        v3 = v_and(v1, v_reinterpret_as_f32(v_not(vm1)));
                    }
                    v0 = v_and(v_le(v0, thresh4), v0);
                    v1 = v_and(v_le(v1, thresh4), v1);
                    if (useMask)
                    {
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                        v1 = v_or(v_and(v1, v_reinterpret_as_f32(vm1)), v3);
                    }
                    v_store( dst + j, v0 );
                    v_store( dst + j + VTraits<v_float32>::vlanes(), v1 );
                }
                if ( j <= roi.width - VTraits<v_float32>::vlanes() )
                {
                    v_float32 v0 = vx_load( src + j );
                    v_float32 v2;
                    if (useMask)
                    {
                        for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                            maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                        vm0 = vx_lut(mask, maskReadLUT);
                        v2 = v_and(v0, v_reinterpret_as_f32(v_not(vm0)));
                    }
                    v0 = v_and(v_le(v0, thresh4), v0);
                    if (useMask)
                        v0 = v_or(v_and(v0, v_reinterpret_as_f32(vm0)), v2);
                    v_store( dst + j, v0 );
                    j += VTraits<v_float32>::vlanes();
                }

                for( ; j < roi.width; j++ )
                    if (!useMask || (mask[j/cn] != 0))
                        dst[j] = threshToZeroInv<float>(src[j], thresh);
            }
            break;
        default:
            CV_Error( cv::Error::StsBadArg, "" ); return;
    }
#else
    threshGeneric<float, useMask>(roi, cn, src, src_step, dst, dst_step, mask, mask_step, thresh, maxval, type);
#endif
}

template<bool useMask>
static void
thresh_64f(const Mat& _src, Mat& _dst, const Mat& _mask, double thresh, double maxval, int type)
{
    Size roi = _src.size();
    const int cn = _src.channels();
    roi.width *= cn;
    const double* src = _src.ptr<double>();
    double* dst = _dst.ptr<double>();
    size_t src_step = _src.step / sizeof(src[0]);
    size_t dst_step = _dst.step / sizeof(dst[0]);
    size_t mask_step = _mask.step;

    if( _src.isContinuous() && _dst.isContinuous() && (!useMask || _mask.empty() || _mask.isContinuous()) )
    {
        roi.width *= roi.height;
        roi.height = 1;
        if (useMask)
            mask_step = _mask.total();
    }

    const uchar* mask = nullptr;
    if (useMask)
        mask = _mask.ptr();

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    int i, j;
    v_float64 thresh2 = vx_setall_f64( thresh );
    v_float64 maxval2 = vx_setall_f64( maxval );

    v_uint8 vm0, vm1;
    int maskReadLUT[32];
    int maskRepeat = 0;
    if (useMask)
        maskRepeat = static_cast<int>(sizeof(double)*cn);
    switch( type )
    {
    case THRESH_BINARY:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_float64>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_float64 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_float64>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_f64(v_not(vm1)));
                }
                v0 = v_lt(thresh2, v0);
                v1 = v_lt(thresh2, v1);
                v0 = v_and(v0, maxval2);
                v1 = v_and(v1, maxval2);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_f64(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_float64>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_float64>::vlanes() )
            {
                v_float64 v0 = vx_load( src + j );
                v_float64 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                }
                v0 = v_lt(thresh2, v0);
                v0 = v_and(v0, maxval2);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_float64>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinary<double>(src[j], thresh, maxval);
        }
        break;

    case THRESH_BINARY_INV:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_float64>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_float64 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_float64>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_f64(v_not(vm1)));
                }
                v0 = v_le(v0, thresh2);
                v1 = v_le(v1, thresh2);
                v0 = v_and(v0, maxval2);
                v1 = v_and(v1, maxval2);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_f64(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_float64>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_float64>::vlanes() )
            {
                v_float64 v0 = vx_load( src + j );
                v_float64 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                }
                v0 = v_le(v0, thresh2);
                v0 = v_and(v0, maxval2);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_float64>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshBinaryInv<double>(src[j], thresh, maxval);
        }
        break;

    case THRESH_TRUNC:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_float64>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_float64 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_float64>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_f64(v_not(vm1)));
                }
                v0 = v_min( v0, thresh2 );
                v1 = v_min( v1, thresh2 );
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_f64(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_float64>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_float64>::vlanes() )
            {
                v_float64 v0 = vx_load( src + j );
                v_float64 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                }
                v0 = v_min( v0, thresh2 );
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_float64>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshTrunc<double>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_float64>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_float64 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_float64>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_f64(v_not(vm1)));
                }
                v0 = v_and(v_lt(thresh2, v0), v0);
                v1 = v_and(v_lt(thresh2, v1), v1);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_f64(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_float64>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_float64>::vlanes() )
            {
                v_float64 v0 = vx_load( src + j );
                v_float64 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                }
                v0 = v_and(v_lt(thresh2, v0), v0);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_float64>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZero<double>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO_INV:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step, mask += useMask ? mask_step : 0 )
        {
            j = 0;
            const int srcReadLanes = 2*VTraits<v_float64>::vlanes();
            const int maskReadLanes = !useMask ? 0 : 2*VTraits<v_uint8>::vlanes();
            for( ; j <= roi.width - srcReadLanes; j += srcReadLanes )
            {
                v_float64 v0, v1, v2, v3;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_float64>::vlanes() );
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    vm1 = vx_lut(mask, maskReadLUT+16);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                    v3 = v_and(v1, v_reinterpret_as_f64(v_not(vm1)));
                }
                v0 = v_and(v_le(v0, thresh2), v0);
                v1 = v_and(v_le(v1, thresh2), v1);
                if (useMask)
                {
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                    v1 = v_or(v_and(v1, v_reinterpret_as_f64(vm1)), v3);
                }
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_float64>::vlanes(), v1 );
            }
            if ( j <= roi.width - VTraits<v_float64>::vlanes() )
            {
                v_float64 v0 = vx_load( src + j );
                v_float64 v2;
                if (useMask)
                {
                    for(int k = 0 ; k<static_cast<int>(sizeof(maskReadLUT)/sizeof(int)) ; ++k)
                        maskReadLUT[k] = (k+(j/srcReadLanes)*maskReadLanes)/maskRepeat;
                    vm0 = vx_lut(mask, maskReadLUT);
                    v2 = v_and(v0, v_reinterpret_as_f64(v_not(vm0)));
                }
                v0 = v_and(v_le(v0, thresh2), v0);
                if (useMask)
                    v0 = v_or(v_and(v0, v_reinterpret_as_f64(vm0)), v2);
                v_store( dst + j, v0 );
                j += VTraits<v_float64>::vlanes();
            }

            for( ; j < roi.width; j++ )
                if (!useMask || (mask[j/cn] != 0))
                    dst[j] = threshToZeroInv<double>(src[j], thresh);
        }
        break;
    default:
        CV_Error(cv::Error::StsBadArg, ""); return;
    }
#else
    threshGeneric<double, useMask>(roi, cn, src, src_step, dst, dst_step, mask, mask_step, thresh, maxval, type);
#endif
}

#ifdef HAVE_IPP
static bool ipp_getThreshVal_Otsu_8u( const unsigned char* _src, int step, Size size, unsigned char &thresh)
{
    CV_INSTRUMENT_REGION_IPP();

// Performance degradations
#if IPP_VERSION_X100 >= 201800
    IppiSize srcSize = { size.width, size.height };

    if(CV_INSTRUMENT_FUN_IPP(ippiComputeThreshold_Otsu_8u_C1R, _src, step, srcSize, &thresh) < 0)
        return false;

    return true;
#else
    CV_UNUSED(_src); CV_UNUSED(step); CV_UNUSED(size); CV_UNUSED(thresh);
    return false;
#endif
}
#endif

template<typename T, size_t BinsOnStack = 0u, bool useMask = false>
static double getThreshVal_Otsu( const Mat& _src, const Mat& _mask, const Size& size)
{
    const int N = std::numeric_limits<T>::max() + 1;
    int i, j;
    #if CV_ENABLE_UNROLLED
    AutoBuffer<int, 4 * BinsOnStack> hBuf(4 * N);
    #else
    AutoBuffer<int, BinsOnStack> hBuf(N);
    #endif
    memset(hBuf.data(), 0, hBuf.size() * sizeof(int));
    int* h = hBuf.data();
    #if CV_ENABLE_UNROLLED
    int* h_unrolled[3] = {h + N, h + 2 * N, h + 3 * N };
    #endif
    int maskCount = 0;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = _src.ptr<T>(i, 0);
        const unsigned char* pMask = nullptr;
        if (useMask)
            pMask = _mask.ptr<unsigned char>(i, 0);
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            if (useMask)
              h[v0] += (pMask[j] != 0) ? 1 : 0;
            else
              h[v0]++;
            if (useMask)
                h_unrolled[0][v1] += (pMask[j+1] != 0) ? ++maskCount,1 : 0;
            else
                h_unrolled[0][v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            if (useMask)
                h_unrolled[1][v0] += (pMask[j+2] != 0) ? ++maskCount,1 : 0;
            else
                h_unrolled[1][v0]++;
            if (useMask)
                h_unrolled[2][v1] += (pMask[j+3] != 0) ? ++maskCount,1 : 0;
            else
                h_unrolled[2][v1]++;
        }
        #endif
        for( ; j < size.width; j++ )
        {
            if (useMask)
                h[src[j]] += (pMask[j] != 0) ? ++maskCount,1 : 0;
            else
                h[src[j]]++;
        }
    }

    double mu = 0, scale = 1./(useMask ? maskCount : (size.width*size.height));
    for( i = 0; i < N; i++ )
    {
        #if CV_ENABLE_UNROLLED
        h[i] += h_unrolled[0][i] + h_unrolled[1][i] + h_unrolled[2][i];
        #endif
        mu += i*(double)h[i];
    }

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for(i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }
    return max_val;
}

static double
getThreshVal_Otsu_8u( const Mat& _src, const Mat& _mask )
{
    Size size = _src.size();
    int step = (int) _src.step;
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }

    if (_mask.empty())
    {
        #ifdef HAVE_IPP
        unsigned char thresh = 0;
        CV_IPP_RUN_FAST(ipp_getThreshVal_Otsu_8u(_src.ptr(), step, size, thresh), thresh);
        #else
        CV_UNUSED(step);
        #endif
    }

    if (!_mask.empty())
        return getThreshVal_Otsu<uchar, 256u, true>(_src, _mask, size);
    else
        return getThreshVal_Otsu<uchar, 256u, false>(_src, _mask, size);
}

static double
getThreshVal_Otsu_16u( const Mat& _src, const Mat& _mask )
{
    Size size = _src.size();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    if (!_mask.empty())
        return getThreshVal_Otsu<ushort, true>(_src, _mask, size);
    else
        return getThreshVal_Otsu<ushort, false>(_src, _mask, size);
}

template<bool useMask>
static double
getThreshVal_Triangle_8u( const Mat& _src, const Mat& _mask )
{
    Size size = _src.size();
    int step = (int) _src.step;
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }

    const int N = 256;
    int i, j, h[N] = {0};
    #if CV_ENABLE_UNROLLED
    int h_unrolled[3][N] = {};
    #endif
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.ptr() + step*i;
        const uchar* mask = nullptr;
        if (useMask)
          mask = _mask.ptr<unsigned char>(i);
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            if (useMask)
                h[v0] += (mask[j] != 0) ? 1 : 0;
            else
                h[v0]++;
            if (useMask)
                h_unrolled[0][v1] += (mask[j+1] != 0) ? 1 : 0;
            else
                h_unrolled[0][v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            if (useMask)
                h_unrolled[1][v0] += (mask[j+2] != 0) ? 1 : 0;
            else
                h_unrolled[1][v0]++;
            if (useMask)
                h_unrolled[2][v0] += (mask[j+3] != 0) ? 1 : 0;
            else
                h_unrolled[2][v1]++;
        }
        #endif
        for( ; j < size.width; j++ )
        {
            if (useMask)
                h[src[j]] += (mask[j] != 0) ? 1 : 0;
            else
                h[src[j]]++;
        }
    }

    int left_bound = 0, right_bound = 0, max_ind = 0, max = 0;
    int temp;
    bool isflipped = false;

    #if CV_ENABLE_UNROLLED
    for( i = 0; i < N; i++ )
    {
        h[i] += h_unrolled[0][i] + h_unrolled[1][i] + h_unrolled[2][i];
    }
    #endif

    for( i = 0; i < N; i++ )
    {
        if( h[i] > 0 )
        {
            left_bound = i;
            break;
        }
    }
    if( left_bound > 0 )
        left_bound--;

    for( i = N-1; i > 0; i-- )
    {
        if( h[i] > 0 )
        {
            right_bound = i;
            break;
        }
    }
    if( right_bound < N-1 )
        right_bound++;

    for( i = 0; i < N; i++ )
    {
        if( h[i] > max)
        {
            max = h[i];
            max_ind = i;
        }
    }

    if( max_ind-left_bound < right_bound-max_ind)
    {
        isflipped = true;
        i = 0, j = N-1;
        while( i < j )
        {
            temp = h[i]; h[i] = h[j]; h[j] = temp;
            i++; j--;
        }
        left_bound = N-1-right_bound;
        max_ind = N-1-max_ind;
    }

    double thresh = left_bound;
    double a, b, dist = 0, tempdist;

    /*
     * We do not need to compute precise distance here. Distance is maximized, so some constants can
     * be omitted. This speeds up a computation a bit.
     */
    a = max; b = left_bound-max_ind;
    for( i = left_bound+1; i <= max_ind; i++ )
    {
        tempdist = a*i + b*h[i];
        if( tempdist > dist)
        {
            dist = tempdist;
            thresh = i;
        }
    }
    thresh--;

    if( isflipped )
        thresh = N-1-thresh;

    return thresh;
}

class ThresholdRunner : public ParallelLoopBody
{
public:
    ThresholdRunner(Mat _src, Mat _dst, const cv::Mat& _mask, double _thresh, double _maxval, int _thresholdType)
    {
        src = _src;
        dst = _dst;
        mask = _mask;
        useMask = !mask.empty();

        thresh = _thresh;
        maxval = _maxval;
        thresholdType = _thresholdType;
    }

    void operator () (const Range& range) const CV_OVERRIDE
    {
        int row0 = range.start;
        int row1 = range.end;

        Mat srcStripe = src.rowRange(row0, row1);
        Mat dstStripe = dst.rowRange(row0, row1);
        Mat maskStripe;

        if (useMask)
        {
            maskStripe = mask.rowRange(row0, row1);
            CALL_HAL(threshold_mask, cv_hal_threshold_mask, srcStripe.data, srcStripe.step, dstStripe.data, dstStripe.step,
                     maskStripe.data, maskStripe.step,
                     srcStripe.cols, srcStripe.rows, srcStripe.depth(), srcStripe.channels(),
                     thresh, maxval, thresholdType);
            if (srcStripe.depth() == CV_8U)
            {
                thresh_8u<true>( srcStripe, dstStripe, maskStripe, (uchar)thresh, (uchar)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_16S )
            {
                thresh_16s<true>( srcStripe, dstStripe, maskStripe, (short)thresh, (short)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_16U )
            {
                thresh_16u<true>( srcStripe, dstStripe, maskStripe, (ushort)thresh, (ushort)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_32F )
            {
                thresh_32f<true>( srcStripe, dstStripe, maskStripe, (float)thresh, (float)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_64F )
            {
                thresh_64f<true>(srcStripe, dstStripe, maskStripe, thresh, maxval, thresholdType);
            }
        }//end if (useMask)
        else//if (!useMask)
        {
            CALL_HAL(threshold, cv_hal_threshold, srcStripe.data, srcStripe.step, dstStripe.data, dstStripe.step,
                     srcStripe.cols, srcStripe.rows, srcStripe.depth(), srcStripe.channels(),
                     thresh, maxval, thresholdType);
            if (srcStripe.depth() == CV_8U)
            {
                thresh_8u<false>( srcStripe, dstStripe, maskStripe, (uchar)thresh, (uchar)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_16S )
            {
                thresh_16s<false>( srcStripe, dstStripe, maskStripe, (short)thresh, (short)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_16U )
            {
                thresh_16u<false>( srcStripe, dstStripe, maskStripe, (ushort)thresh, (ushort)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_32F )
            {
                thresh_32f<false>( srcStripe, dstStripe, maskStripe, (float)thresh, (float)maxval, thresholdType );
            }
            else if( srcStripe.depth() == CV_64F )
            {
                thresh_64f<false>(srcStripe, dstStripe, maskStripe, thresh, maxval, thresholdType);
            }
        }//end if (!useMask)
    }

private:
    Mat src;
    Mat dst;
    Mat mask;

    double thresh;
    double maxval;
    int thresholdType;
    bool useMask;
};

#ifdef HAVE_OPENCL

static bool ocl_threshold( InputArray _src, OutputArray _dst, InputArray _mask, double & thresh, double maxval, int thresh_type )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
        kercn = ocl::predictOptimalVectorWidth(_src, _dst), ktype = CV_MAKE_TYPE(depth, kercn);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    const bool isDisabled = ((thresh_type & THRESH_DRYRUN) != 0);
    thresh_type &= ~THRESH_DRYRUN;

    if ( isDisabled ||
        !(thresh_type == THRESH_BINARY || thresh_type == THRESH_BINARY_INV || thresh_type == THRESH_TRUNC ||
         thresh_type == THRESH_TOZERO || thresh_type == THRESH_TOZERO_INV) ||
        (!doubleSupport && depth == CV_64F))
        return false;

    const char * const thresholdMap[] = { "THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC",
                                          "THRESH_TOZERO", "THRESH_TOZERO_INV" };
    ocl::Device dev = ocl::Device::getDefault();
    int stride_size = dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU) ? 4 : 1;

    const bool useMask = !_mask.empty();

    ocl::Kernel k =
        !useMask ?
            ocl::Kernel("threshold", ocl::imgproc::threshold_oclsrc,
                        format("-D %s -D T=%s -D T1=%s -D STRIDE_SIZE=%d%s", thresholdMap[thresh_type],
                        ocl::typeToStr(ktype), ocl::typeToStr(depth), stride_size,
                        doubleSupport ? " -D DOUBLE_SUPPORT" : "")) :
            ocl::Kernel("threshold_mask", ocl::imgproc::threshold_mask_oclsrc,
                        format("-D %s -D T=%s -D T1=%s -D CN=%d -D STRIDE_SIZE=%d%s", thresholdMap[thresh_type],
                        ocl::typeToStr(ktype), ocl::typeToStr(depth), cn, stride_size,
                        doubleSupport ? " -D DOUBLE_SUPPORT" : ""));

    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(src.size(), type);
    UMat dst = _dst.getUMat();
    UMat mask = !useMask ? cv::UMat() : _mask.getUMat();

    if (depth <= CV_32S)
        thresh = cvFloor(thresh);

    const double min_vals[] = { 0, CHAR_MIN, 0, SHRT_MIN, INT_MIN, -FLT_MAX, -DBL_MAX, 0 };
    double min_val = min_vals[depth];

    if (!useMask)
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst, cn, kercn),
               ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(thresh))),
               ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(maxval))),
               ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(min_val))));
    else
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst, cn, kercn),
            ocl::KernelArg::ReadOnlyNoSize(mask),
            ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(thresh))),
            ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(maxval))),
            ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(min_val))));

    size_t globalsize[2] = { (size_t)dst.cols * cn / kercn, (size_t)dst.rows };
    globalsize[1] = (globalsize[1] + stride_size - 1) / stride_size;
    return k.run(2, globalsize, NULL, false);
}

#endif


#ifdef HAVE_OPENVX
#define IMPL_OPENVX_TOZERO 1
static bool openvx_threshold(Mat src, Mat dst, const Mat& mask, int thresh, int maxval, int type)
{
    CV_Assert(mask.empty());
    Mat a = src;

    int trueVal, falseVal;
    switch (type)
    {
    case THRESH_BINARY:
#ifndef VX_VERSION_1_1
        if (maxval != 255)
            return false;
#endif
        trueVal = maxval;
        falseVal = 0;
        break;
    case THRESH_TOZERO:
#if IMPL_OPENVX_TOZERO
        trueVal = 255;
        falseVal = 0;
        if (dst.data == src.data)
        {
            a = Mat(src.size(), src.type());
            src.copyTo(a);
        }
        break;
#endif
    case THRESH_BINARY_INV:
#ifdef VX_VERSION_1_1
        trueVal = 0;
        falseVal = maxval;
        break;
#endif
    case THRESH_TOZERO_INV:
#ifdef VX_VERSION_1_1
#if IMPL_OPENVX_TOZERO
        trueVal = 0;
        falseVal = 255;
        if (dst.data == src.data)
        {
            a = Mat(src.size(), src.type());
            src.copyTo(a);
        }
        break;
#endif
#endif
    case THRESH_TRUNC:
    default:
        return false;
    }

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();

        ivx::Threshold thh = ivx::Threshold::createBinary(ctx, VX_TYPE_UINT8, thresh);
        thh.setValueTrue(trueVal);
        thh.setValueFalse(falseVal);

        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(a.cols*a.channels(), a.rows, 1, (vx_int32)(a.step)), src.data),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(dst.cols*dst.channels(), dst.rows, 1, (vx_int32)(dst.step)), dst.data);

        ivx::IVX_CHECK_STATUS(vxuThreshold(ctx, ia, thh, ib));
#if IMPL_OPENVX_TOZERO
        if (type == THRESH_TOZERO || type == THRESH_TOZERO_INV)
        {
            ivx::Image
                ic = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                    ivx::Image::createAddressing(dst.cols*dst.channels(), dst.rows, 1, (vx_int32)(dst.step)), dst.data);
            ivx::IVX_CHECK_STATUS(vxuAnd(ctx, ib, ia, ic));
        }
#endif
    }
    catch (const ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (const ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
#endif

}

double cv::threshold( InputArray _src, OutputArray _dst, double thresh, double maxval, int type )
{
  return cv::threshold(_src, _dst, cv::noArray(), thresh, maxval, type);
}

double cv::threshold( InputArray _src, OutputArray _dst, InputArray _mask, double thresh, double maxval, int type )
{
    CV_INSTRUMENT_REGION();
    CV_Assert(_mask.empty() || ((_mask.type() == CV_8UC1) && (_mask.size() == _src.size())));

    CV_OCL_RUN_(_src.dims() <= 2 && _dst.isUMat(),
                ocl_threshold(_src, _dst, _mask, thresh, maxval, type), thresh)

    const bool isDisabled = ((type & THRESH_DRYRUN) != 0);
    type &= ~THRESH_DRYRUN;

    Mat src = _src.getMat();
    Mat mask = _mask.getMat();

    const bool hasMask = !mask.empty();
    const bool isMaskValid = hasMask && ((mask.type() == CV_8UC1) && (mask.size() == src.size()));

    if (!isDisabled)
        _dst.create( src.size(), src.type() );
    Mat dst = isDisabled ? cv::Mat() : _dst.getMat();

    int automatic_thresh = (type & ~cv::THRESH_MASK);
    type &= THRESH_MASK;

    CV_Assert( automatic_thresh != (cv::THRESH_OTSU | cv::THRESH_TRIANGLE) );
    if( automatic_thresh == cv::THRESH_OTSU )
    {
        int src_type = src.type();
        CV_CheckType(src_type, src_type == CV_8UC1 || src_type == CV_16UC1, "THRESH_OTSU mode");

        if (isMaskValid)
        {
            CALL_HAL_RET(thresholdMaskOtsu, cv_hal_threshold_mask_otsu, thresh, src.data, src.step, dst.data, dst.step, mask.data, mask.step,
                src.cols, src.rows, src_type, maxval, type);
        }
        else
        {
            CALL_HAL_RET(thresholdOtsu, cv_hal_threshold_otsu, thresh, src.data, src.step, dst.data, dst.step,
                         src.cols, src.rows, src_type, maxval, type);
        }

        thresh = src.type() == CV_8UC1 ? getThreshVal_Otsu_8u( src, mask )
                                       : getThreshVal_Otsu_16u( src, mask );
    }
    else if( automatic_thresh == cv::THRESH_TRIANGLE )
    {
        CV_Assert( src.type() == CV_8UC1 );
        if (isMaskValid)
            thresh = getThreshVal_Triangle_8u<true>( src, mask );
        else
            thresh = getThreshVal_Triangle_8u<false>( src, mask );
    }

    if( src.depth() == CV_8U )
    {
        int ithresh = cvFloor(thresh);
        thresh = ithresh;
        if (isDisabled)
            return thresh;

        int imaxval = cvRound(maxval);
        if( type == THRESH_TRUNC )
            imaxval = ithresh;
        imaxval = saturate_cast<uchar>(imaxval);

        if( ithresh < 0 || ithresh >= 255 )
        {
            if( type == THRESH_BINARY || type == THRESH_BINARY_INV ||
                ((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < 0) ||
                (type == THRESH_TOZERO && ithresh >= 255) )
            {
                int v = type == THRESH_BINARY ? (ithresh >= 255 ? 0 : imaxval) :
                        type == THRESH_BINARY_INV ? (ithresh >= 255 ? imaxval : 0) :
                        /*type == THRESH_TRUNC ? imaxval :*/ 0;
                dst.setTo(v, mask);
            }
            else
                src.copyTo(dst, mask);
            return thresh;
        }

       CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_THRESHOLD>(src.cols, src.rows),
                  openvx_threshold(src, dst, mask, ithresh, imaxval, type), (double)ithresh)

        thresh = ithresh;
        maxval = imaxval;
    }
    else if( src.depth() == CV_16S )
    {
        int ithresh = cvFloor(thresh);
        thresh = ithresh;
        if (isDisabled)
            return thresh;

        int imaxval = cvRound(maxval);
        if( type == THRESH_TRUNC )
            imaxval = ithresh;
        imaxval = saturate_cast<short>(imaxval);

        if( ithresh < SHRT_MIN || ithresh >= SHRT_MAX )
        {
            if( type == THRESH_BINARY || type == THRESH_BINARY_INV ||
               ((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < SHRT_MIN) ||
               (type == THRESH_TOZERO && ithresh >= SHRT_MAX) )
            {
                int v = type == THRESH_BINARY ? (ithresh >= SHRT_MAX ? 0 : imaxval) :
                type == THRESH_BINARY_INV ? (ithresh >= SHRT_MAX ? imaxval : 0) :
                /*type == THRESH_TRUNC ? imaxval :*/ 0;
                dst.setTo(v, mask);
            }
            else
                src.copyTo(dst, mask);
            return thresh;
        }
        thresh = ithresh;
        maxval = imaxval;
    }
    else if (src.depth() == CV_16U )
    {
        int ithresh = cvFloor(thresh);
        thresh = ithresh;
        if (isDisabled)
            return thresh;

        int imaxval = cvRound(maxval);
        if (type == THRESH_TRUNC)
            imaxval = ithresh;
        imaxval = saturate_cast<ushort>(imaxval);

        int ushrt_min = 0;
        if (ithresh < ushrt_min || ithresh >= (int)USHRT_MAX)
        {
            if (type == THRESH_BINARY || type == THRESH_BINARY_INV ||
               ((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < ushrt_min) ||
               (type == THRESH_TOZERO && ithresh >= (int)USHRT_MAX))
            {
                int v = type == THRESH_BINARY ? (ithresh >= (int)USHRT_MAX ? 0 : imaxval) :
                        type == THRESH_BINARY_INV ? (ithresh >= (int)USHRT_MAX ? imaxval : 0) :
                  /*type == THRESH_TRUNC ? imaxval :*/ 0;
                dst.setTo(v, mask);
            }
            else
                src.copyTo(dst, mask);
            return thresh;
        }
        thresh = ithresh;
        maxval = imaxval;
    }
    else if( src.depth() == CV_32F )
        ;
    else if( src.depth() == CV_64F )
        ;
    else
        CV_Error( cv::Error::StsUnsupportedFormat, "" );

    if (isDisabled)
        return thresh;

    parallel_for_(Range(0, dst.rows),
                  ThresholdRunner(src, dst, mask, thresh, maxval, type),
                  dst.total()/(double)(1<<16));
    return thresh;
}


void cv::adaptiveThreshold( InputArray _src, OutputArray _dst, double maxValue,
                            int method, int type, int blockSize, double delta )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
    Size size = src.size();

    _dst.create( size, src.type() );
    Mat dst = _dst.getMat();

    if( maxValue < 0 )
    {
        dst = Scalar(0);
        return;
    }

    CALL_HAL(adaptiveThreshold, cv_hal_adaptiveThreshold, src.data, src.step, dst.data, dst.step, src.cols, src.rows,
             maxValue, method, type, blockSize, delta);

    Mat mean;

    if( src.data != dst.data )
        mean = dst;

    if (method == ADAPTIVE_THRESH_MEAN_C)
        boxFilter( src, mean, src.type(), Size(blockSize, blockSize),
                   Point(-1,-1), true, BORDER_REPLICATE|BORDER_ISOLATED );
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        Mat srcfloat,meanfloat;
        src.convertTo(srcfloat,CV_32F);
        meanfloat=srcfloat;
        GaussianBlur(srcfloat, meanfloat, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE|BORDER_ISOLATED);
        meanfloat.convertTo(mean, src.type());
    }
    else
        CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported adaptive threshold method" );

    int i, j;
    uchar imaxval = saturate_cast<uchar>(maxValue);
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
    uchar tab[768];

    if( type == cv::THRESH_BINARY )
        for( i = 0; i < 768; i++ )
            tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
    else if( type == cv::THRESH_BINARY_INV )
        for( i = 0; i < 768; i++ )
            tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
    else
        CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported threshold type" );

    if( src.isContinuous() && mean.isContinuous() && dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const uchar* sdata = src.ptr(i);
        const uchar* mdata = mean.ptr(i);
        uchar* ddata = dst.ptr(i);

        for( j = 0; j < size.width; j++ )
            ddata[j] = tab[sdata[j] - mdata[j] + 255];
    }
}

CV_IMPL double
cvThreshold( const void* srcarr, void* dstarr, double thresh, double maxval, int type )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), dst0 = dst;

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() &&
        (src.depth() == dst.depth() || dst.depth() == CV_8U));

    thresh = cv::threshold( src, dst, thresh, maxval, type );
    if( dst0.data != dst.data )
        dst.convertTo( dst0, dst0.depth() );
    return thresh;
}


CV_IMPL void
cvAdaptiveThreshold( const void *srcIm, void *dstIm, double maxValue,
                     int method, int type, int blockSize, double delta )
{
    cv::Mat src = cv::cvarrToMat(srcIm), dst = cv::cvarrToMat(dstIm);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    cv::adaptiveThreshold( src, dst, maxValue, method, type, blockSize, delta );
}

/* End of file. */
