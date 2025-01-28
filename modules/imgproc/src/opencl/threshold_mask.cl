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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Zhang Ying, zhangying913@gmail.com
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
// This software is provided by the copyright holders and contributors as is and
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void threshold_mask(__global const uchar * srcptr, int src_step, int src_offset,
                             __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
                             __global const uchar * maskptr, int mask_step, int mask_offset,
                             T1 thresh, T1 max_val, T1 min_val)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1) * STRIDE_SIZE;

    if (gx < cols)
    {
        int src_index = mad24(gy, src_step, mad24(gx, (int)sizeof(T), src_offset));
        int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));
        int mask_index = mad24(gy, mask_step, mad24(gx/CN, (int)sizeof(uchar), mask_offset));

        #pragma unroll
        for (int i = 0; i < STRIDE_SIZE; i++)
        {
            if (gy < rows)
            {
                T sdata = *(__global const T *)(srcptr + src_index);
                const uchar mdata = *(maskptr + mask_index);
                if (mdata != 0)
                {
                    __global T * dst = (__global T *)(dstptr + dst_index);

                    #ifdef THRESH_BINARY
                            dst[0] = sdata > (thresh) ? (T)(max_val) : (T)(0);
                    #elif defined THRESH_BINARY_INV
                            dst[0] = sdata > (thresh) ? (T)(0) : (T)(max_val);
                    #elif defined THRESH_TRUNC
                            dst[0] = clamp(sdata, (T)min_val, (T)(thresh));
                    #elif defined THRESH_TOZERO
                            dst[0] = sdata > (thresh) ? sdata : (T)(0);
                    #elif defined THRESH_TOZERO_INV
                            dst[0] = sdata > (thresh) ? (T)(0) : sdata;
                    #endif
                }
                gy++;
                src_index += src_step;
                dst_index += dst_step;
                mask_index += mask_step;
            }
        }
    }
}
