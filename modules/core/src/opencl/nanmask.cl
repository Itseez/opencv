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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//
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
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

__kernel void nanMask(__global const uchar * srcptr, int srcstep, int srcoffset,
                      __global uchar * dstptr, int dststep, int dstoffset,
                      int rows, int cols )
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < cols)
    {
        int src_index = mad24(y0, srcstep, mad24(x, (int)sizeof(srcT) * cn, srcoffset));
        int dst_index = mad24(y0, dststep, x + dstoffset);

        for (int y = y0, y1 = min(rows, y0 + rowsPerWI); y < y1; ++y, src_index += srcstep, dst_index += dststep)
        {
#ifdef MASK_ALL
            bool vnan = true;
#else
            bool vnan = false;
#endif
            for (int c = 0; c < cn; c++)
            {
                srcT val = *(__global srcT *)(srcptr + src_index + c * (int)sizeof(srcT));

                bool v = false;
#ifdef MASK_NANS
                v = v || isnan(val);
#endif

#ifdef MASK_INFS
                v = v || isinf(val);
#endif

#ifdef MASK_ALL
                vnan = vnan && v;
#else
                vnan = vnan || v;
#endif
            }
#ifdef INVERT
            vnan = !vnan;
#endif

            *(dstptr + dst_index) = vnan ? 255 : 0;
        }
    }
}