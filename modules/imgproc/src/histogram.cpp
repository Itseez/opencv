/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//            Intel License Agreement
//        For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "opencv2/core/utils/tls.hpp"

namespace cv
{

////////////////// Helper functions //////////////////////

#define CV_CLAMP_INT(v, vmin, vmax) (v < vmin ? vmin : (vmax < v ? vmax : v))

static const size_t OUT_OF_RANGE = (size_t)1 << (sizeof(size_t)*8 - 2);

static void
calcHistLookupTables_8u( const Mat& hist, const SparseMat& shist,
                         int dims, const float** ranges, const double* uniranges,
                         bool uniform, bool issparse, std::vector<size_t>& _tab )
{
    const int low = 0, high = 256;
    int i, j;
    _tab.resize((high-low)*dims);
    size_t* tab = &_tab[0];

    if( uniform )
    {
        for( i = 0; i < dims; i++ )
        {
            double a = uniranges[i*2];
            double b = uniranges[i*2+1];
            int sz = !issparse ? hist.size[i] : shist.size(i);
            size_t step = !issparse ? hist.step[i] : 1;

            double v_lo = ranges ? ranges[i][0] : 0;
            double v_hi = ranges ? ranges[i][1] : 256;

            for( j = low; j < high; j++ )
            {
                int idx = cvFloor(j*a + b);
                size_t written_idx = OUT_OF_RANGE;
                if (j >= v_lo && j < v_hi)
                {
                    idx = CV_CLAMP_INT(idx, 0, sz - 1);
                    written_idx = idx*step;
                }
                tab[i*(high - low) + j - low] = written_idx;
            }
        }
    }
    else if (ranges)
    {
        for( i = 0; i < dims; i++ )
        {
            int limit = std::min(cvCeil(ranges[i][0]), high);
            int idx = -1, sz = !issparse ? hist.size[i] : shist.size(i);
            size_t written_idx = OUT_OF_RANGE;
            size_t step = !issparse ? hist.step[i] : 1;

            for(j = low;;)
            {
                for( ; j < limit; j++ )
                    tab[i*(high - low) + j - low] = written_idx;

                if( (unsigned)(++idx) < (unsigned)sz )
                {
                    limit = std::min(cvCeil(ranges[i][idx+1]), high);
                    written_idx = idx*step;
                }
                else
                {
                    for( ; j < high; j++ )
                        tab[i*(high - low) + j - low] = OUT_OF_RANGE;
                    break;
                }
            }
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Either ranges, either uniform ranges should be provided");
    }
}


static void histPrepareImages( const Mat* images, int nimages, const int* channels,
                               const Mat& mask, int dims, const int* histSize,
                               const float** ranges, bool uniform,
                               std::vector<uchar*>& ptrs, std::vector<int>& deltas,
                               Size& imsize, std::vector<double>& uniranges )
{
    int i, j, c;
    CV_Assert( channels != 0 || nimages == dims );

    imsize = images[0].size();
    int depth = images[0].depth(), esz1 = (int)images[0].elemSize1();
    bool isContinuous = true;

    ptrs.resize(dims + 1);
    deltas.resize((dims + 1)*2);

    for( i = 0; i < dims; i++ )
    {
        if(!channels)
        {
            j = i;
            c = 0;
            CV_Assert( images[j].channels() == 1 );
        }
        else
        {
            c = channels[i];
            CV_Assert( c >= 0 );
            for( j = 0; j < nimages; c -= images[j].channels(), j++ )
                if( c < images[j].channels() )
                    break;
            CV_Assert( j < nimages );
        }

        CV_Assert( images[j].size() == imsize && images[j].depth() == depth );
        if( !images[j].isContinuous() )
            isContinuous = false;
        ptrs[i] = images[j].data + c*esz1;
        deltas[i*2] = images[j].channels();
        deltas[i*2+1] = (int)(images[j].step/esz1 - imsize.width*deltas[i*2]);
    }

    if( !mask.empty() )
    {
        CV_Assert( mask.size() == imsize && mask.channels() == 1 );
        isContinuous = isContinuous && mask.isContinuous();
        ptrs[dims] = mask.data;
        deltas[dims*2] = 1;
        deltas[dims*2 + 1] = (int)(mask.step/mask.elemSize1());
    }

    if( isContinuous )
    {
        imsize.width *= imsize.height;
        imsize.height = 1;
    }

    if( !ranges ) // implicit uniform ranges for 8U
    {
        CV_Assert( depth == CV_8U );

        uniranges.resize( dims*2 );
        for( i = 0; i < dims; i++ )
        {
            uniranges[i*2] = histSize[i]/256.;
            uniranges[i*2+1] = 0;
        }
    }
    else if( uniform )
    {
        uniranges.resize( dims*2 );
        for( i = 0; i < dims; i++ )
        {
            CV_Assert( ranges[i] && ranges[i][0] < ranges[i][1] );
            double low = ranges[i][0], high = ranges[i][1];
            double t = histSize[i]/(high - low);
            uniranges[i*2] = t;
            uniranges[i*2+1] = -t*low;
#if 0  // This should be true by math, but it is not accurate numerically
            CV_Assert(cvFloor(low * uniranges[i*2] + uniranges[i*2+1]) == 0);
            CV_Assert((high * uniranges[i*2] + uniranges[i*2+1]) < histSize[i]);
#endif
        }
    }
    else
    {
        for( i = 0; i < dims; i++ )
        {
            size_t n = histSize[i];
            for(size_t k = 0; k < n; k++ )
                CV_Assert( ranges[i][k] < ranges[i][k+1] );
        }
    }
}


////////////////////////////////// C A L C U L A T E    H I S T O G R A M ////////////////////////////////////

template<typename T> static void
calcHist_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
           Size imsize, Mat& hist, int dims, const float** _ranges,
           const double* _uniranges, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.ptr();
    int i, x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    int size[CV_MAX_DIM];
    size_t hstep[CV_MAX_DIM];

    for( i = 0; i < dims; i++ )
    {
        size[i] = hist.size[i];
        hstep[i] = hist.step[i];
    }

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];

        if( dims == 1 )
        {
            double a = uniranges[0], b = uniranges[1];
            int sz = size[0], d0 = deltas[0], step0 = deltas[1];
            const T* p0 = (const T*)ptrs[0];

            double v0_lo = _ranges[0][0];
            double v0_hi = _ranges[0][1];

            for( ; imsize.height--; p0 += step0, mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++, p0 += d0 )
                    {
                        double v0 = (double)*p0;
                        int idx = cvFloor(v0*a + b);
                        if (v0 < v0_lo || v0 >= v0_hi)
                            continue;
                        idx = CV_CLAMP_INT(idx, 0, sz - 1);
                        CV_DbgAssert((unsigned)idx < (unsigned)sz);
                        ((int*)H)[idx]++;
                    }
                else
                    for( x = 0; x < imsize.width; x++, p0 += d0 )
                        if( mask[x] )
                        {
                            double v0 = (double)*p0;
                            int idx = cvFloor(v0*a + b);
                            if (v0 < v0_lo || v0 >= v0_hi)
                                continue;
                            idx = CV_CLAMP_INT(idx, 0, sz - 1);
                            CV_DbgAssert((unsigned)idx < (unsigned)sz);
                            ((int*)H)[idx]++;
                        }
            }
            return;
        }
        else if( dims == 2 )
        {
            double a0 = uniranges[0], b0 = uniranges[1], a1 = uniranges[2], b1 = uniranges[3];
            int sz0 = size[0], sz1 = size[1];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3];
            size_t hstep0 = hstep[0];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];

            double v0_lo = _ranges[0][0];
            double v0_hi = _ranges[0][1];
            double v1_lo = _ranges[1][0];
            double v1_hi = _ranges[1][1];

            for( ; imsize.height--; p0 += step0, p1 += step1, mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                    {
                        double v0 = (double)*p0;
                        double v1 = (double)*p1;
                        int idx0 = cvFloor(v0*a0 + b0);
                        int idx1 = cvFloor(v1*a1 + b1);
                        if (v0 < v0_lo || v0 >= v0_hi)
                            continue;
                        if (v1 < v1_lo || v1 >= v1_hi)
                            continue;
                        idx0 = CV_CLAMP_INT(idx0, 0, sz0 - 1);
                        idx1 = CV_CLAMP_INT(idx1, 0, sz1 - 1);
                        CV_DbgAssert((unsigned)idx0 < (unsigned)sz0 && (unsigned)idx1 < (unsigned)sz1);
                        ((int*)(H + hstep0*idx0))[idx1]++;
                    }
                else
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                        if( mask[x] )
                        {
                            double v0 = (double)*p0;
                            double v1 = (double)*p1;
                            int idx0 = cvFloor(v0*a0 + b0);
                            int idx1 = cvFloor(v1*a1 + b1);
                            if (v0 < v0_lo || v0 >= v0_hi)
                                continue;
                            if (v1 < v1_lo || v1 >= v1_hi)
                                continue;
                            idx0 = CV_CLAMP_INT(idx0, 0, sz0 - 1);
                            idx1 = CV_CLAMP_INT(idx1, 0, sz1 - 1);
                            CV_DbgAssert((unsigned)idx0 < (unsigned)sz0 && (unsigned)idx1 < (unsigned)sz1);
                            ((int*)(H + hstep0*idx0))[idx1]++;
                        }
            }
            return;
        }
        else if( dims == 3 )
        {
            double a0 = uniranges[0], b0 = uniranges[1],
                   a1 = uniranges[2], b1 = uniranges[3],
                   a2 = uniranges[4], b2 = uniranges[5];
            int sz0 = size[0], sz1 = size[1], sz2 = size[2];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3],
                d2 = deltas[4], step2 = deltas[5];
            size_t hstep0 = hstep[0], hstep1 = hstep[1];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];
            const T* p2 = (const T*)ptrs[2];

            double v0_lo = _ranges[0][0];
            double v0_hi = _ranges[0][1];
            double v1_lo = _ranges[1][0];
            double v1_hi = _ranges[1][1];
            double v2_lo = _ranges[2][0];
            double v2_hi = _ranges[2][1];

            for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                    {
                        double v0 = (double)*p0;
                        double v1 = (double)*p1;
                        double v2 = (double)*p2;
                        int idx0 = cvFloor(v0*a0 + b0);
                        int idx1 = cvFloor(v1*a1 + b1);
                        int idx2 = cvFloor(v2*a2 + b2);
                        if (v0 < v0_lo || v0 >= v0_hi)
                            continue;
                        if (v1 < v1_lo || v1 >= v1_hi)
                            continue;
                        if (v2 < v2_lo || v2 >= v2_hi)
                            continue;
                        idx0 = CV_CLAMP_INT(idx0, 0, sz0 - 1);
                        idx1 = CV_CLAMP_INT(idx1, 0, sz1 - 1);
                        idx2 = CV_CLAMP_INT(idx2, 0, sz2 - 1);
                        CV_DbgAssert(
                            (unsigned)idx0 < (unsigned)sz0 &&
                            (unsigned)idx1 < (unsigned)sz1 &&
                            (unsigned)idx2 < (unsigned)sz2);
                        ((int*)(H + hstep0*idx0 + hstep1*idx1))[idx2]++;
                    }
                else
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                        if( mask[x] )
                        {
                            double v0 = (double)*p0;
                            double v1 = (double)*p1;
                            double v2 = (double)*p2;
                            int idx0 = cvFloor(v0*a0 + b0);
                            int idx1 = cvFloor(v1*a1 + b1);
                            int idx2 = cvFloor(v2*a2 + b2);
                            if (v0 < v0_lo || v0 >= v0_hi)
                                continue;
                            if (v1 < v1_lo || v1 >= v1_hi)
                                continue;
                            if (v2 < v2_lo || v2 >= v2_hi)
                                continue;
                            idx0 = CV_CLAMP_INT(idx0, 0, sz0 - 1);
                            idx1 = CV_CLAMP_INT(idx1, 0, sz1 - 1);
                            idx2 = CV_CLAMP_INT(idx2, 0, sz2 - 1);
                            CV_DbgAssert(
                                (unsigned)idx0 < (unsigned)sz0 &&
                                (unsigned)idx1 < (unsigned)sz1 &&
                                (unsigned)idx2 < (unsigned)sz2);
                            ((int*)(H + hstep0*idx0 + hstep1*idx1))[idx2]++;
                        }
            }
        }
        else
        {
            for( ; imsize.height--; mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++ )
                    {
                        uchar* Hptr = H;
                        for( i = 0; i < dims; i++ )
                        {
                            double v_lo = _ranges[i][0];
                            double v_hi = _ranges[i][1];
                            double v = *ptrs[i];
                            if (v < v_lo || v >= v_hi)
                                break;
                            int idx = cvFloor(v*uniranges[i*2] + uniranges[i*2+1]);
                            idx = CV_CLAMP_INT(idx, 0, size[i] - 1);
                            CV_DbgAssert((unsigned)idx < (unsigned)size[i]);
                            ptrs[i] += deltas[i*2];
                            Hptr += idx*hstep[i];
                        }

                        if( i == dims )
                            ++*((int*)Hptr);
                        else
                            for( ; i < dims; i++ )
                                ptrs[i] += deltas[i*2];
                    }
                else
                    for( x = 0; x < imsize.width; x++ )
                    {
                        uchar* Hptr = H;
                        i = 0;
                        if( mask[x] )
                            for( ; i < dims; i++ )
                            {
                                double v_lo = _ranges[i][0];
                                double v_hi = _ranges[i][1];
                                double v = *ptrs[i];
                                if (v < v_lo || v >= v_hi)
                                    break;
                                int idx = cvFloor(v*uniranges[i*2] + uniranges[i*2+1]);
                                idx = CV_CLAMP_INT(idx, 0, size[i] - 1);
                                CV_DbgAssert((unsigned)idx < (unsigned)size[i]);
                                ptrs[i] += deltas[i*2];
                                Hptr += idx*hstep[i];
                            }

                        if( i == dims )
                            ++*((int*)Hptr);
                        else
                            for( ; i < dims; i++ )
                                ptrs[i] += deltas[i*2];
                    }
                for( i = 0; i < dims; i++ )
                    ptrs[i] += deltas[i*2 + 1];
            }
        }
    }
    else if (_ranges)
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; mask += mstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                uchar* Hptr = H;
                i = 0;

                if( !mask || mask[x] )
                    for( ; i < dims; i++ )
                    {
                        float v = (float)*ptrs[i];
                        const float* R = ranges[i];
                        int idx = -1, sz = size[i];

                        while( v >= R[idx+1] && ++idx < sz )
                            ; // nop

                        if( (unsigned)idx >= (unsigned)sz )
                            break;

                        ptrs[i] += deltas[i*2];
                        Hptr += idx*hstep[i];
                    }

                if( i == dims )
                    ++*((int*)Hptr);
                else
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Either ranges, either uniform ranges should be provided");
    }
}


static void
calcHist_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
             Size imsize, Mat& hist, int dims, const float** _ranges,
             const double* _uniranges, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.ptr();
    int x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    std::vector<size_t> _tab;

    calcHistLookupTables_8u( hist, SparseMat(), dims, _ranges, _uniranges, uniform, false, _tab );
    const size_t* tab = &_tab[0];

    if( dims == 1 )
    {
        int d0 = deltas[0], step0 = deltas[1];
        int matH[256] = { 0, };
        const uchar* p0 = (const uchar*)ptrs[0];

        for( ; imsize.height--; p0 += step0, mask += mstep )
        {
            if( !mask )
            {
                if( d0 == 1 )
                {
                    for( x = 0; x <= imsize.width - 4; x += 4 )
                    {
                        int t0 = p0[x], t1 = p0[x+1];
                        matH[t0]++; matH[t1]++;
                        t0 = p0[x+2]; t1 = p0[x+3];
                        matH[t0]++; matH[t1]++;
                    }
                    p0 += x;
                }
                else
                    for( x = 0; x <= imsize.width - 4; x += 4 )
                    {
                        int t0 = p0[0], t1 = p0[d0];
                        matH[t0]++; matH[t1]++;
                        p0 += d0*2;
                        t0 = p0[0]; t1 = p0[d0];
                        matH[t0]++; matH[t1]++;
                        p0 += d0*2;
                    }

                for( ; x < imsize.width; x++, p0 += d0 )
                    matH[*p0]++;
            }
            else
                for( x = 0; x < imsize.width; x++, p0 += d0 )
                    if( mask[x] )
                        matH[*p0]++;
        }

        for(int i = 0; i < 256; i++ )
        {
            size_t hidx = tab[i];
            if( hidx < OUT_OF_RANGE )
                *(int*)(H + hidx) += matH[i];
        }
    }
    else if( dims == 2 )
    {
        int d0 = deltas[0], step0 = deltas[1],
            d1 = deltas[2], step1 = deltas[3];
        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];

        for( ; imsize.height--; p0 += step0, p1 += step1, mask += mstep )
        {
            if( !mask )
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                {
                    size_t idx = tab[*p0] + tab[*p1 + 256];
                    if( idx < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
            else
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                {
                    size_t idx;
                    if( mask[x] && (idx = tab[*p0] + tab[*p1 + 256]) < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
        }
    }
    else if( dims == 3 )
    {
        int d0 = deltas[0], step0 = deltas[1],
            d1 = deltas[2], step1 = deltas[3],
            d2 = deltas[4], step2 = deltas[5];

        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];
        const uchar* p2 = (const uchar*)ptrs[2];

        for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, mask += mstep )
        {
            if( !mask )
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                {
                    size_t idx = tab[*p0] + tab[*p1 + 256] + tab[*p2 + 512];
                    if( idx < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
            else
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                {
                    size_t idx;
                    if( mask[x] && (idx = tab[*p0] + tab[*p1 + 256] + tab[*p2 + 512]) < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
        }
    }
    else
    {
        for( ; imsize.height--; mask += mstep )
        {
            if( !mask )
                for( x = 0; x < imsize.width; x++ )
                {
                    uchar* Hptr = H;
                    int i = 0;
                    for( ; i < dims; i++ )
                    {
                        size_t idx = tab[*ptrs[i] + i*256];
                        if( idx >= OUT_OF_RANGE )
                            break;
                        Hptr += idx;
                        ptrs[i] += deltas[i*2];
                    }

                    if( i == dims )
                        ++*((int*)Hptr);
                    else
                        for( ; i < dims; i++ )
                            ptrs[i] += deltas[i*2];
                }
            else
                for( x = 0; x < imsize.width; x++ )
                {
                    uchar* Hptr = H;
                    int i = 0;
                    if( mask[x] )
                        for( ; i < dims; i++ )
                        {
                            size_t idx = tab[*ptrs[i] + i*256];
                            if( idx >= OUT_OF_RANGE )
                                break;
                            Hptr += idx;
                            ptrs[i] += deltas[i*2];
                        }

                    if( i == dims )
                        ++*((int*)Hptr);
                    else
                        for( ; i < dims; i++ )
                            ptrs[i] += deltas[i*2];
                }
            for(int i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}

#ifdef HAVE_IPP

typedef IppStatus(CV_STDCALL * IppiHistogram_C1)(const void* pSrc, int srcStep,
    IppiSize roiSize, Ipp32u* pHist, const IppiHistogramSpec* pSpec, Ipp8u* pBuffer);

static IppiHistogram_C1 getIppiHistogramFunction_C1(int type)
{
    IppiHistogram_C1 ippFunction =
        (type == CV_8UC1) ? (IppiHistogram_C1)ippiHistogram_8u_C1R :
        (type == CV_16UC1) ? (IppiHistogram_C1)ippiHistogram_16u_C1R :
        (type == CV_32FC1) ? (IppiHistogram_C1)ippiHistogram_32f_C1R :
        NULL;

    return ippFunction;
}

class ipp_calcHistParallelTLS
{
public:
    ipp_calcHistParallelTLS() {}

    IppAutoBuffer<IppiHistogramSpec> spec;
    IppAutoBuffer<Ipp8u>  buffer;
    IppAutoBuffer<Ipp32u> thist;
};

class ipp_calcHistParallel: public ParallelLoopBody
{
public:
    ipp_calcHistParallel(const Mat &src, Mat &hist, Ipp32s histSize, const float *ranges, bool uniform, bool &ok):
        ParallelLoopBody(), m_src(src), m_hist(hist), m_ok(ok)
    {
        ok = true;

        m_uniform        = uniform;
        m_ranges         = ranges;
        m_histSize       = histSize;
        m_type           = ippiGetDataType(src.type());
        m_levelsNum      = histSize+1;
        ippiHistogram_C1 = getIppiHistogramFunction_C1(src.type());
        m_fullRoi    = ippiSize(src.size());
        m_bufferSize = 0;
        m_specSize   = 0;
        if(!ippiHistogram_C1)
        {
            ok = false;
            return;
        }

        if(ippiHistogramGetBufferSize(m_type, m_fullRoi, &m_levelsNum, 1, 1, &m_specSize, &m_bufferSize) < 0)
        {
            ok = false;
            return;
        }

        hist.setTo(0);
    }

    virtual void operator() (const Range & range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION_IPP();

        if(!m_ok)
            return;

        ipp_calcHistParallelTLS *pTls = m_tls.get();

        IppiSize roi = {m_src.cols, range.end - range.start };
        bool     mtLoop = false;
        if(m_fullRoi.height != roi.height)
            mtLoop = true;

        if(!pTls->spec)
        {
            pTls->spec.allocate(m_specSize);
            if(!pTls->spec.get())
            {
                m_ok = false;
                return;
            }

            pTls->buffer.allocate(m_bufferSize);
            if(!pTls->buffer.get() && m_bufferSize)
            {
                m_ok = false;
                return;
            }

            if(m_uniform)
            {
                if(ippiHistogramUniformInit(m_type, (Ipp32f*)&m_ranges[0], (Ipp32f*)&m_ranges[1], (Ipp32s*)&m_levelsNum, 1, pTls->spec) < 0)
                {
                    m_ok = false;
                    return;
                }
            }
            else
            {
                if(ippiHistogramInit(m_type, (const Ipp32f**)&m_ranges, (Ipp32s*)&m_levelsNum, 1, pTls->spec) < 0)
                {
                    m_ok = false;
                    return;
                }
            }

            pTls->thist.allocate(m_histSize*sizeof(Ipp32u));
        }

        if(CV_INSTRUMENT_FUN_IPP(ippiHistogram_C1, m_src.ptr(range.start), (int)m_src.step, roi, pTls->thist, pTls->spec, pTls->buffer) < 0)
        {
            m_ok = false;
            return;
        }

        if(mtLoop)
        {
            for(int i = 0; i < m_histSize; i++)
                CV_XADD((int*)(m_hist.ptr(i)), *(int*)((Ipp32u*)pTls->thist + i));
        }
        else
            ippiCopy_32s_C1R((Ipp32s*)pTls->thist.get(), sizeof(Ipp32u), (Ipp32s*)m_hist.ptr(), (int)m_hist.step, ippiSize(1, m_histSize));
    }

private:
    const Mat      &m_src;
    Mat            &m_hist;
    Ipp32s          m_histSize;
    const float    *m_ranges;
    bool            m_uniform;

    IppiHistogram_C1    ippiHistogram_C1;
    IppiSize            m_fullRoi;
    IppDataType         m_type;
    Ipp32s              m_levelsNum;
    int                 m_bufferSize;
    int                 m_specSize;

    mutable Mutex                    m_syncMutex;
    TLSData<ipp_calcHistParallelTLS> m_tls;

    volatile bool &m_ok;
    const ipp_calcHistParallel & operator = (const ipp_calcHistParallel & );
};

#endif

}

#ifdef HAVE_IPP
#define IPP_HISTOGRAM_PARALLEL 1
namespace cv
{
static bool ipp_calchist(const Mat &image, Mat &hist, int histSize, const float** ranges, bool uniform, bool accumulate)
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201801
    // No SSE42 optimization for uniform 32f
    if(uniform && image.depth() == CV_32F && cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

    // IPP_DISABLE_HISTOGRAM - https://github.com/opencv/opencv/issues/11544
    // and https://github.com/opencv/opencv/issues/21595
    if ((uniform && (ranges[0][1] - ranges[0][0]) != histSize) || abs(ranges[0][0]) != cvFloor(ranges[0][0]))
        return false;

    Mat ihist = hist;
    if(accumulate)
        ihist.create(1, &histSize, CV_32S);

    bool  ok      = true;
    int   threads = ippiSuggestThreadsNum(image, (1+((double)ihist.total()/image.total()))*2);
    Range range(0, image.rows);
    ipp_calcHistParallel invoker(image, ihist, histSize, ranges[0], uniform, ok);
    if(!ok)
        return false;

    if(IPP_HISTOGRAM_PARALLEL && threads > 1)
        parallel_for_(range, invoker, threads*2);
    else
        invoker(range);

    if(ok)
    {
        if(accumulate)
        {
            IppiSize histRoi = ippiSize(1, histSize);
            IppAutoBuffer<Ipp32f> fhist(histSize*sizeof(Ipp32f));
            CV_INSTRUMENT_FUN_IPP(ippiConvert_32s32f_C1R, (Ipp32s*)ihist.ptr(), (int)ihist.step, (Ipp32f*)fhist, sizeof(Ipp32f), histRoi);
            CV_INSTRUMENT_FUN_IPP(ippiAdd_32f_C1IR, (Ipp32f*)fhist, sizeof(Ipp32f), (Ipp32f*)hist.ptr(), (int)hist.step, histRoi);
        }
        else
            CV_INSTRUMENT_FUN_IPP(ippiConvert_32s32f_C1R, (Ipp32s*)ihist.ptr(), (int)ihist.step, (Ipp32f*)hist.ptr(), (int)hist.step, ippiSize(1, histSize));
    }
    return ok;
}
}
#endif

void cv::calcHist( const Mat* images, int nimages, const int* channels,
                   InputArray _mask, OutputArray _hist, int dims, const int* histSize,
                   const float** ranges, bool uniform, bool accumulate )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(images && nimages > 0);

    Mat mask = _mask.getMat();

    CV_Assert(dims > 0 && histSize);

    const uchar* const histdata = _hist.getMat().ptr();
    _hist.create(dims, histSize, CV_32F);
    Mat hist = _hist.getMat();

    if(histdata != hist.data)
        accumulate = false;

    CV_IPP_RUN(
        nimages == 1 && dims == 1 && channels && channels[0] == 0
            && _mask.empty() && images[0].dims <= 2 && ranges && ranges[0],
        ipp_calchist(images[0], hist, histSize[0], ranges, uniform, accumulate));

    Mat ihist = hist;
    ihist.flags = (ihist.flags & ~CV_MAT_TYPE_MASK)|CV_32S;

    if(!accumulate)
        hist = Scalar(0.);
    else
        hist.convertTo(ihist, CV_32S);

    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;

    CV_Assert( mask.empty() || mask.type() == CV_8UC1 );
    histPrepareImages( images, nimages, channels, mask, dims, hist.size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();

    if( depth == CV_8U )
        calcHist_8u(ptrs, deltas, imsize, ihist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_16U )
        calcHist_<ushort>(ptrs, deltas, imsize, ihist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_32F )
        calcHist_<float>(ptrs, deltas, imsize, ihist, dims, ranges, _uniranges, uniform );
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");

    ihist.convertTo(hist, CV_32F);
}


namespace cv
{

template<typename T> static void
calcSparseHist_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                 Size imsize, SparseMat& hist, int dims, const float** _ranges,
                 const double* _uniranges, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    const int* size = hist.hdr->size;
    int idx[CV_MAX_DIM];

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];

        for( ; imsize.height--; mask += mstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                i = 0;
                if( !mask || mask[x] )
                    for( ; i < dims; i++ )
                    {
                        idx[i] = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                        if( (unsigned)idx[i] >= (unsigned)size[i] )
                            break;
                        ptrs[i] += deltas[i*2];
                    }

                if( i == dims )
                    ++*(int*)hist.ptr(idx, true);
                else
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
            }
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else if (_ranges)
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; mask += mstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                i = 0;

                if( !mask || mask[x] )
                    for( ; i < dims; i++ )
                    {
                        float v = (float)*ptrs[i];
                        const float* R = ranges[i];
                        int j = -1, sz = size[i];

                        while( v >= R[j+1] && ++j < sz )
                            ; // nop

                        if( (unsigned)j >= (unsigned)sz )
                            break;
                        ptrs[i] += deltas[i*2];
                        idx[i] = j;
                    }

                if( i == dims )
                    ++*(int*)hist.ptr(idx, true);
                else
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Either ranges, either uniform ranges should be provided");
    }
}


static void
calcSparseHist_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                   Size imsize, SparseMat& hist, int dims, const float** _ranges,
                   const double* _uniranges, bool uniform )
{
    uchar** ptrs = (uchar**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    int idx[CV_MAX_DIM];
    std::vector<size_t> _tab;

    calcHistLookupTables_8u( Mat(), hist, dims, _ranges, _uniranges, uniform, true, _tab );
    const size_t* tab = &_tab[0];

    for( ; imsize.height--; mask += mstep )
    {
        for( x = 0; x < imsize.width; x++ )
        {
            int i = 0;
            if( !mask || mask[x] )
                for( ; i < dims; i++ )
                {
                    size_t hidx = tab[*ptrs[i] + i*256];
                    if( hidx >= OUT_OF_RANGE )
                        break;
                    ptrs[i] += deltas[i*2];
                    idx[i] = (int)hidx;
                }

            if( i == dims )
                ++*(int*)hist.ptr(idx,true);
            else
                for( ; i < dims; i++ )
                    ptrs[i] += deltas[i*2];
        }
        for(int i = 0; i < dims; i++ )
            ptrs[i] += deltas[i*2 + 1];
    }
}


static void calcHist( const Mat* images, int nimages, const int* channels,
                      const Mat& mask, SparseMat& hist, int dims, const int* histSize,
                      const float** ranges, bool uniform, bool accumulate, bool keepInt )
{
    size_t i, N;

    if( !accumulate )
        hist.create(dims, histSize, CV_32F);
    else
    {
        SparseMatIterator it = hist.begin();
        for( i = 0, N = hist.nzcount(); i < N; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            Cv32suf* val = (Cv32suf*)it.ptr;
            val->i = cvRound(val->f);
        }
    }

    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;

    CV_Assert( mask.empty() || mask.type() == CV_8UC1 );
    histPrepareImages( images, nimages, channels, mask, dims, hist.hdr->size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();
    if( depth == CV_8U )
        calcSparseHist_8u(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_16U )
        calcSparseHist_<ushort>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_32F )
        calcSparseHist_<float>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, uniform );
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");

    if( !keepInt )
    {
        SparseMatIterator it = hist.begin();
        for( i = 0, N = hist.nzcount(); i < N; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            Cv32suf* val = (Cv32suf*)it.ptr;
            val->f = (float)val->i;
        }
    }
}

#ifdef HAVE_OPENCL

enum
{
    BINS = 256
};

static bool ocl_calcHist1(InputArray _src, OutputArray _hist, int ddepth = CV_32S)
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int compunits = dev.maxComputeUnits();
    size_t wgs = dev.maxWorkGroupSize();
    Size size = _src.size();
    bool use16 = size.width % 16 == 0 && _src.offset() % 16 == 0 && _src.step() % 16 == 0;
    int kercn = dev.isAMD() && use16 ? 16 : std::min(4, ocl::predictOptimalVectorWidth(_src));

    ocl::Kernel k1("calculate_histogram", ocl::imgproc::histogram_oclsrc,
                   format("-D BINS=%d -D HISTS_COUNT=%d -D WGS=%zu -D kercn=%d -D T=%s%s",
                          BINS, compunits, wgs, kercn,
                          kercn == 4 ? "int" : ocl::typeToStr(CV_8UC(kercn)),
                          _src.isContinuous() ? " -D HAVE_SRC_CONT" : ""));
    if (k1.empty())
        return false;

    int hsz = BINS;
    _hist.create(1, &hsz, ddepth);
    UMat src = _src.getUMat(), ghist(1, BINS * compunits, CV_32SC1),
            hist = _hist.getUMat().reshape(1, hsz);

    k1.args(ocl::KernelArg::ReadOnly(src),
            ocl::KernelArg::PtrWriteOnly(ghist), (int)src.total());

    size_t globalsize = compunits * wgs;
    if (!k1.run(1, &globalsize, &wgs, false))
        return false;

    wgs = std::min<size_t>(ocl::Device::getDefault().maxWorkGroupSize(), BINS);
    char cvt[50];
    ocl::Kernel k2("merge_histogram", ocl::imgproc::histogram_oclsrc,
                   format("-D BINS=%d -D HISTS_COUNT=%d -D WGS=%d -D convertToHT=%s -D HT=%s",
                          BINS, compunits, (int)wgs, ocl::convertTypeStr(CV_32S, ddepth, 1, cvt, sizeof(cvt)),
                          ocl::typeToStr(ddepth)));
    if (k2.empty())
        return false;

    k2.args(ocl::KernelArg::PtrReadOnly(ghist),
            ocl::KernelArg::WriteOnlyNoSize(hist));

    return k2.run(1, &wgs, &wgs, false);
}

static bool ocl_calcHist(InputArrayOfArrays images, OutputArray hist)
{
    std::vector<UMat> v;
    images.getUMatVector(v);

    return ocl_calcHist1(v[0], hist, CV_32F);
}

#endif

}

void cv::calcHist( const Mat* images, int nimages, const int* channels,
               InputArray _mask, SparseMat& hist, int dims, const int* histSize,
               const float** ranges, bool uniform, bool accumulate )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(images && nimages > 0);

    Mat mask = _mask.getMat();
    calcHist( images, nimages, channels, mask, hist, dims, histSize,
              ranges, uniform, accumulate, false );
}


void cv::calcHist( InputArrayOfArrays images, const std::vector<int>& channels,
                   InputArray mask, OutputArray hist,
                   const std::vector<int>& histSize,
                   const std::vector<float>& ranges,
                   bool accumulate )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(images.total() == 1 && channels.size() == 1 && images.channels(0) == 1 &&
               channels[0] == 0 && images.isUMatVector() && mask.empty() && !accumulate &&
               histSize.size() == 1 && histSize[0] == BINS && ranges.size() == 2 &&
               ranges[0] == 0 && ranges[1] == static_cast<float>(BINS),
               ocl_calcHist(images, hist))

    int i, dims = (int)histSize.size(), rsz = (int)ranges.size(), csz = (int)channels.size();
    int nimages = (int)images.total();

    CV_Assert(nimages > 0 && dims > 0);
    CV_Assert(rsz == dims*2 || (rsz == 0 && images.depth(0) == CV_8U));
    CV_Assert(csz == 0 || csz == dims);
    float* _ranges[CV_MAX_DIM];
    if( rsz > 0 )
    {
        for( i = 0; i < rsz/2; i++ )
            _ranges[i] = (float*)&ranges[i*2];
    }

    AutoBuffer<Mat> buf(nimages);
    for( i = 0; i < nimages; i++ )
        buf[i] = images.getMat(i);

    calcHist(&buf[0], nimages, csz ? &channels[0] : 0,
            mask, hist, dims, &histSize[0], rsz ? (const float**)_ranges : 0,
            true, accumulate);
}


/////////////////////////////////////// B A C K   P R O J E C T ////////////////////////////////////

namespace cv
{

template<typename T, typename BT> static void
calcBackProj_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
               Size imsize, const Mat& hist, int dims, const float** _ranges,
               const double* _uniranges, float scale, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    const uchar* H = hist.ptr();
    int i, x;
    BT* bproj = (BT*)_ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    int size[CV_MAX_DIM];
    size_t hstep[CV_MAX_DIM];

    for( i = 0; i < dims; i++ )
    {
        size[i] = hist.size[i];
        hstep[i] = hist.step[i];
    }

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];

        if( dims == 1 )
        {
            double a = uniranges[0], b = uniranges[1];
            int sz = size[0], d0 = deltas[0], step0 = deltas[1];
            const T* p0 = (const T*)ptrs[0];

            for( ; imsize.height--; p0 += step0, bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++, p0 += d0 )
                {
                    int idx = cvFloor(*p0*a + b);
                    bproj[x] = (unsigned)idx < (unsigned)sz ? saturate_cast<BT>(((const float*)H)[idx]*scale) : 0;
                }
            }
        }
        else if( dims == 2 )
        {
            double a0 = uniranges[0], b0 = uniranges[1],
                   a1 = uniranges[2], b1 = uniranges[3];
            int sz0 = size[0], sz1 = size[1];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3];
            size_t hstep0 = hstep[0];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];

            for( ; imsize.height--; p0 += step0, p1 += step1, bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                {
                    int idx0 = cvFloor(*p0*a0 + b0);
                    int idx1 = cvFloor(*p1*a1 + b1);
                    bproj[x] = (unsigned)idx0 < (unsigned)sz0 &&
                               (unsigned)idx1 < (unsigned)sz1 ?
                        saturate_cast<BT>(((const float*)(H + hstep0*idx0))[idx1]*scale) : 0;
                }
            }
        }
        else if( dims == 3 )
        {
            double a0 = uniranges[0], b0 = uniranges[1],
                   a1 = uniranges[2], b1 = uniranges[3],
                   a2 = uniranges[4], b2 = uniranges[5];
            int sz0 = size[0], sz1 = size[1], sz2 = size[2];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3],
                d2 = deltas[4], step2 = deltas[5];
            size_t hstep0 = hstep[0], hstep1 = hstep[1];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];
            const T* p2 = (const T*)ptrs[2];

            for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                {
                    int idx0 = cvFloor(*p0*a0 + b0);
                    int idx1 = cvFloor(*p1*a1 + b1);
                    int idx2 = cvFloor(*p2*a2 + b2);
                    bproj[x] = (unsigned)idx0 < (unsigned)sz0 &&
                               (unsigned)idx1 < (unsigned)sz1 &&
                               (unsigned)idx2 < (unsigned)sz2 ?
                        saturate_cast<BT>(((const float*)(H + hstep0*idx0 + hstep1*idx1))[idx2]*scale) : 0;
                }
            }
        }
        else
        {
            for( ; imsize.height--; bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++ )
                {
                    const uchar* Hptr = H;
                    for( i = 0; i < dims; i++ )
                    {
                        int idx = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                        if( (unsigned)idx >= (unsigned)size[i] || (_ranges && *ptrs[i] >= _ranges[i][1]))
                            break;
                        ptrs[i] += deltas[i*2];
                        Hptr += idx*hstep[i];
                    }

                    if( i == dims )
                        bproj[x] = saturate_cast<BT>(*(const float*)Hptr*scale);
                    else
                    {
                        bproj[x] = 0;
                        for( ; i < dims; i++ )
                            ptrs[i] += deltas[i*2];
                    }
                }
                for( i = 0; i < dims; i++ )
                    ptrs[i] += deltas[i*2 + 1];
            }
        }
    }
    else if (_ranges)
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                const uchar* Hptr = H;
                for( i = 0; i < dims; i++ )
                {
                    float v = (float)*ptrs[i];
                    const float* R = ranges[i];
                    int idx = -1, sz = size[i];

                    while( v >= R[idx+1] && ++idx < sz )
                        ; // nop

                    if( (unsigned)idx >= (unsigned)sz )
                        break;

                    ptrs[i] += deltas[i*2];
                    Hptr += idx*hstep[i];
                }

                if( i == dims )
                    bproj[x] = saturate_cast<BT>(*(const float*)Hptr*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Either ranges, either uniform ranges should be provided");
    }
}


static void
calcBackProj_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                 Size imsize, const Mat& hist, int dims, const float** _ranges,
                 const double* _uniranges, float scale, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    const uchar* H = hist.ptr();
    int i, x;
    uchar* bproj = _ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    std::vector<size_t> _tab;

    calcHistLookupTables_8u( hist, SparseMat(), dims, _ranges, _uniranges, uniform, false, _tab );
    const size_t* tab = &_tab[0];

    if( dims == 1 )
    {
        int d0 = deltas[0], step0 = deltas[1];
        uchar matH[256] = {0};
        const uchar* p0 = (const uchar*)ptrs[0];

        for( i = 0; i < 256; i++ )
        {
            size_t hidx = tab[i];
            if( hidx < OUT_OF_RANGE )
                matH[i] = saturate_cast<uchar>(*(float*)(H + hidx)*scale);
        }

        for( ; imsize.height--; p0 += step0, bproj += bpstep )
        {
            if( d0 == 1 )
            {
                for( x = 0; x <= imsize.width - 4; x += 4 )
                {
                    uchar t0 = matH[p0[x]], t1 = matH[p0[x+1]];
                    bproj[x] = t0; bproj[x+1] = t1;
                    t0 = matH[p0[x+2]]; t1 = matH[p0[x+3]];
                    bproj[x+2] = t0; bproj[x+3] = t1;
                }
                p0 += x;
            }
            else
                for( x = 0; x <= imsize.width - 4; x += 4 )
                {
                    uchar t0 = matH[p0[0]], t1 = matH[p0[d0]];
                    bproj[x] = t0; bproj[x+1] = t1;
                    p0 += d0*2;
                    t0 = matH[p0[0]]; t1 = matH[p0[d0]];
                    bproj[x+2] = t0; bproj[x+3] = t1;
                    p0 += d0*2;
                }

            for( ; x < imsize.width; x++, p0 += d0 )
                bproj[x] = matH[*p0];
        }
    }
    else if( dims == 2 )
    {
        int d0 = deltas[0], step0 = deltas[1],
            d1 = deltas[2], step1 = deltas[3];
        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];

        for( ; imsize.height--; p0 += step0, p1 += step1, bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
            {
                size_t idx = tab[*p0] + tab[*p1 + 256];
                bproj[x] = idx < OUT_OF_RANGE ? saturate_cast<uchar>(*(const float*)(H + idx)*scale) : 0;
            }
        }
    }
    else if( dims == 3 )
    {
        int d0 = deltas[0], step0 = deltas[1],
        d1 = deltas[2], step1 = deltas[3],
        d2 = deltas[4], step2 = deltas[5];
        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];
        const uchar* p2 = (const uchar*)ptrs[2];

        for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
            {
                size_t idx = tab[*p0] + tab[*p1 + 256] + tab[*p2 + 512];
                bproj[x] = idx < OUT_OF_RANGE ? saturate_cast<uchar>(*(const float*)(H + idx)*scale) : 0;
            }
        }
    }
    else
    {
        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                const uchar* Hptr = H;
                for( i = 0; i < dims; i++ )
                {
                    size_t idx = tab[*ptrs[i] + i*256];
                    if( idx >= OUT_OF_RANGE )
                        break;
                    ptrs[i] += deltas[i*2];
                    Hptr += idx;
                }

                if( i == dims )
                    bproj[x] = saturate_cast<uchar>(*(const float*)Hptr*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}

}

void cv::calcBackProject( const Mat* images, int nimages, const int* channels,
                          InputArray _hist, OutputArray _backProject,
                          const float** ranges, double scale, bool uniform )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(images && nimages > 0);

    Mat hist = _hist.getMat();
    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;
    if (hist.dims == 2 && (hist.rows == 1 || hist.cols == 1)) {
        CV_Assert(hist.isContinuous());
        std::vector<int> hist_size = {hist.rows + hist.cols - 1};
        hist = hist.reshape(1, hist_size);
    }
    int dims = hist.dims;

    CV_Assert( dims > 0 && !hist.empty() );
    _backProject.create( images[0].size(), images[0].depth() );
    Mat backProject = _backProject.getMat();
    histPrepareImages( images, nimages, channels, backProject, dims, hist.size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();
    if( depth == CV_8U )
        calcBackProj_8u(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, (float)scale, uniform);
    else if( depth == CV_16U )
        calcBackProj_<ushort, ushort>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, (float)scale, uniform );
    else if( depth == CV_32F )
        calcBackProj_<float, float>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, (float)scale, uniform );
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");
}


namespace cv
{

template<typename T, typename BT> static void
calcSparseBackProj_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                     Size imsize, const SparseMat& hist, int dims, const float** _ranges,
                     const double* _uniranges, float scale, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x;
    BT* bproj = (BT*)_ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    const int* size = hist.hdr->size;
    int idx[CV_MAX_DIM];
    const SparseMat_<float>& hist_ = (const SparseMat_<float>&)hist;

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];
        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                for( i = 0; i < dims; i++ )
                {
                    idx[i] = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                    if( (unsigned)idx[i] >= (unsigned)size[i] )
                        break;
                    ptrs[i] += deltas[i*2];
                }

                if( i == dims )
                    bproj[x] = saturate_cast<BT>(hist_(idx)*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else if (_ranges)
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                for( i = 0; i < dims; i++ )
                {
                    float v = (float)*ptrs[i];
                    const float* R = ranges[i];
                    int j = -1, sz = size[i];

                    while( v >= R[j+1] && ++j < sz )
                        ; // nop

                    if( (unsigned)j >= (unsigned)sz )
                        break;
                    idx[i] = j;
                    ptrs[i] += deltas[i*2];
                }

                if( i == dims )
                    bproj[x] = saturate_cast<BT>(hist_(idx)*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Either ranges, either uniform ranges should be provided");
    }
}


static void
calcSparseBackProj_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                       Size imsize, const SparseMat& hist, int dims, const float** _ranges,
                       const double* _uniranges, float scale, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x;
    uchar* bproj = _ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    std::vector<size_t> _tab;
    int idx[CV_MAX_DIM];

    calcHistLookupTables_8u( Mat(), hist, dims, _ranges, _uniranges, uniform, true, _tab );
    const size_t* tab = &_tab[0];

    for( ; imsize.height--; bproj += bpstep )
    {
        for( x = 0; x < imsize.width; x++ )
        {
            for( i = 0; i < dims; i++ )
            {
                size_t hidx = tab[*ptrs[i] + i*256];
                if( hidx >= OUT_OF_RANGE )
                    break;
                idx[i] = (int)hidx;
                ptrs[i] += deltas[i*2];
            }

            if( i == dims )
                bproj[x] = saturate_cast<uchar>(hist.value<float>(idx)*scale);
            else
            {
                bproj[x] = 0;
                for( ; i < dims; i++ )
                    ptrs[i] += deltas[i*2];
            }
        }
        for( i = 0; i < dims; i++ )
            ptrs[i] += deltas[i*2 + 1];
    }
}

}

void cv::calcBackProject( const Mat* images, int nimages, const int* channels,
                          const SparseMat& hist, OutputArray _backProject,
                          const float** ranges, double scale, bool uniform )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(images && nimages > 0);

    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;
    int dims = hist.dims();

    CV_Assert( dims > 0 );
    _backProject.create( images[0].size(), images[0].depth() );
    Mat backProject = _backProject.getMat();
    histPrepareImages( images, nimages, channels, backProject,
                       dims, hist.hdr->size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;
    int depth = images[0].depth();
    if( depth == CV_8U )
        calcSparseBackProj_8u(ptrs, deltas, imsize, hist, dims, ranges,
                              _uniranges, (float)scale, uniform);
    else if( depth == CV_16U )
        calcSparseBackProj_<ushort, ushort>(ptrs, deltas, imsize, hist, dims, ranges,
                                          _uniranges, (float)scale, uniform );
    else if( depth == CV_32F )
        calcSparseBackProj_<float, float>(ptrs, deltas, imsize, hist, dims, ranges,
                                          _uniranges, (float)scale, uniform );
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");
}

#ifdef HAVE_OPENCL

namespace cv {

static void getUMatIndex(const std::vector<UMat> & um, int cn, int & idx, int & cnidx)
{
    int totalChannels = 0;
    for (size_t i = 0, size = um.size(); i < size; ++i)
    {
        int ccn = um[i].channels();
        totalChannels += ccn;

        if (totalChannels == cn)
        {
            idx = (int)(i + 1);
            cnidx = 0;
            return;
        }
        else if (totalChannels > cn)
        {
            idx = (int)i;
            cnidx = i == 0 ? cn : (cn - totalChannels + ccn);
            return;
        }
    }

    idx = cnidx = -1;
}

static bool ocl_calcBackProject( InputArrayOfArrays _images, std::vector<int> channels,
                                 InputArray _hist, OutputArray _dst,
                                 const std::vector<float>& ranges,
                                 float scale, size_t histdims )
{
    std::vector<UMat> images;
    _images.getUMatVector(images);

    size_t nimages = images.size(), totalcn = images[0].channels();

    CV_Assert(nimages > 0);
    Size size = images[0].size();
    int depth = images[0].depth();

    //kernels are valid for this type only
    if (depth != CV_8U)
        return false;

    for (size_t i = 1; i < nimages; ++i)
    {
        const UMat & m = images[i];
        totalcn += m.channels();
        CV_Assert(size == m.size() && depth == m.depth());
    }

    std::sort(channels.begin(), channels.end());
    for (size_t i = 0; i < histdims; ++i)
        CV_Assert(channels[i] < (int)totalcn);

    if (histdims == 1)
    {
        int idx, cnidx;
        getUMatIndex(images, channels[0], idx, cnidx);
        CV_Assert(idx >= 0);
        UMat im = images[idx];

        String opts = format("-D histdims=1 -D scn=%d", im.channels());
        ocl::Kernel lutk("calcLUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (lutk.empty())
            return false;

        size_t lsize = 256;
        UMat lut(1, (int)lsize, CV_32SC1);
        UMat hist = _hist.getUMat();
        UMat uranges; Mat(ranges, false).copyTo(uranges);
        hist = hist.reshape(1, hist.rows + hist.cols-1);

        lutk.args(ocl::KernelArg::ReadOnlyNoSize(hist), hist.rows,
                  ocl::KernelArg::PtrWriteOnly(lut), scale, ocl::KernelArg::PtrReadOnly(uranges));
        if (!lutk.run(1, &lsize, NULL, false))
            return false;

        ocl::Kernel mapk("LUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (mapk.empty())
            return false;

        _dst.create(size, depth);
        UMat dst = _dst.getUMat();

        im.offset += cnidx;
        mapk.args(ocl::KernelArg::ReadOnlyNoSize(im), ocl::KernelArg::PtrReadOnly(lut),
                  ocl::KernelArg::WriteOnly(dst));

        size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };
        return mapk.run(2, globalsize, NULL, false);
    }
    else if (histdims == 2)
    {
        int idx0, idx1, cnidx0, cnidx1;
        getUMatIndex(images, channels[0], idx0, cnidx0);
        getUMatIndex(images, channels[1], idx1, cnidx1);
        CV_Assert(idx0 >= 0 && idx1 >= 0);
        UMat im0 = images[idx0], im1 = images[idx1];

        // Lut for the first dimension
        String opts = format("-D histdims=2 -D scn1=%d -D scn2=%d", im0.channels(), im1.channels());
        ocl::Kernel lutk1("calcLUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (lutk1.empty())
            return false;

        size_t lsize = 256;
        UMat lut(1, (int)lsize<<1, CV_32SC1);
        UMat hist = _hist.getUMat();
        UMat uranges; Mat(ranges, false).copyTo(uranges);

        lutk1.args(hist.rows, ocl::KernelArg::PtrWriteOnly(lut), (int)0, ocl::KernelArg::PtrReadOnly(uranges), (int)0);
        if (!lutk1.run(1, &lsize, NULL, false))
            return false;

        // lut for the second dimension
        ocl::Kernel lutk2("calcLUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (lutk2.empty())
            return false;

        lut.offset += lsize * sizeof(int);
        lutk2.args(hist.cols, ocl::KernelArg::PtrWriteOnly(lut), (int)256, ocl::KernelArg::PtrReadOnly(uranges), (int)2);
        if (!lutk2.run(1, &lsize, NULL, false))
            return false;

        // perform lut
        ocl::Kernel mapk("LUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (mapk.empty())
            return false;

        _dst.create(size, depth);
        UMat dst = _dst.getUMat();

        im0.offset += cnidx0;
        im1.offset += cnidx1;
        mapk.args(ocl::KernelArg::ReadOnlyNoSize(im0), ocl::KernelArg::ReadOnlyNoSize(im1),
               ocl::KernelArg::ReadOnlyNoSize(hist), ocl::KernelArg::PtrReadOnly(lut), scale, ocl::KernelArg::WriteOnly(dst));

        size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };
        return mapk.run(2, globalsize, NULL, false);
    }
    return false;
}

}

#endif

void cv::calcBackProject( InputArrayOfArrays images, const std::vector<int>& channels,
                          InputArray hist, OutputArray dst,
                          const std::vector<float>& ranges,
                          double scale )
{
    CV_INSTRUMENT_REGION();
    if (hist.dims() <= 2)
    {
#ifdef HAVE_OPENCL
        Size histSize = hist.size();
        bool _1D = histSize.height == 1 || histSize.width == 1;
        size_t histdims = _1D ? 1 : hist.dims();
#endif

        CV_OCL_RUN(dst.isUMat() && hist.type() == CV_32FC1 &&
            histdims <= 2 && ranges.size() == histdims * 2 && histdims == channels.size(),
            ocl_calcBackProject(images, channels, hist, dst, ranges, (float)scale, histdims))
    }
    Mat H0 = hist.getMat(), H;
    int hcn = H0.channels();

    if( hcn > 1 )
    {
        CV_Assert( H0.isContinuous() );
        int hsz[CV_CN_MAX+1];
        memcpy(hsz, &H0.size[0], H0.dims*sizeof(hsz[0]));
        hsz[H0.dims] = hcn;
        H = Mat(H0.dims+1, hsz, H0.depth(), H0.ptr());
    }
    else
        H = H0;

    bool _1d = H.rows == 1 || H.cols == 1;
    int i, dims = H.dims, rsz = (int)ranges.size(), csz = (int)channels.size();
    int nimages = (int)images.total();

    CV_Assert(nimages > 0);
    CV_Assert(rsz == dims*2 || (rsz == 2 && _1d) || (rsz == 0 && images.depth(0) == CV_8U));
    CV_Assert(csz == 0 || csz == dims || (csz == 1 && _1d));

    float* _ranges[CV_MAX_DIM];
    if( rsz > 0 )
    {
        for( i = 0; i < rsz/2; i++ )
            _ranges[i] = (float*)&ranges[i*2];
    }

    AutoBuffer<Mat> buf(nimages);
    for( i = 0; i < nimages; i++ )
        buf[i] = images.getMat(i);

    calcBackProject(&buf[0], nimages, csz ? &channels[0] : 0,
        hist, dst, rsz ? (const float**)_ranges : 0, scale, true);
}


////////////////// C O M P A R E   H I S T O G R A M S ////////////////////////

double cv::compareHist( InputArray _H1, InputArray _H2, int method )
{
    CV_INSTRUMENT_REGION();

    Mat H1 = _H1.getMat(), H2 = _H2.getMat();
    const Mat* arrays[] = {&H1, &H2, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    double result = 0;
    int j;

    CV_Assert( H1.type() == H2.type() && H1.depth() == CV_32F );

    double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

    CV_Assert( it.planes[0].isContinuous() && it.planes[1].isContinuous() );

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        const float* h1 = it.planes[0].ptr<float>();
        const float* h2 = it.planes[1].ptr<float>();
        const int len = it.planes[0].rows*it.planes[0].cols*H1.channels();
        j = 0;

        if( (method == cv::HISTCMP_CHISQR) || (method == cv::HISTCMP_CHISQR_ALT))
        {
            for( ; j < len; j++ )
            {
                double a = h1[j] - h2[j];
                double b = (method == cv::HISTCMP_CHISQR) ? h1[j] : h1[j] + h2[j];
                if( fabs(b) > DBL_EPSILON )
                    result += a*a/b;
            }
        }
        else if( method == cv::HISTCMP_CORREL )
        {
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
            v_float64 v_s1 = vx_setzero_f64();
            v_float64 v_s2 = vx_setzero_f64();
            v_float64 v_s11 = vx_setzero_f64();
            v_float64 v_s12 = vx_setzero_f64();
            v_float64 v_s22 = vx_setzero_f64();
            for ( ; j <= len - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
            {
                v_float32 v_a = vx_load(h1 + j);
                v_float32 v_b = vx_load(h2 + j);

                // 0-1
                v_float64 v_ad = v_cvt_f64(v_a);
                v_float64 v_bd = v_cvt_f64(v_b);
                v_s12 = v_muladd(v_ad, v_bd, v_s12);
                v_s11 = v_muladd(v_ad, v_ad, v_s11);
                v_s22 = v_muladd(v_bd, v_bd, v_s22);
                v_s1 = v_add(v_s1, v_ad);
                v_s2 = v_add(v_s2, v_bd);

                // 2-3
                v_ad = v_cvt_f64_high(v_a);
                v_bd = v_cvt_f64_high(v_b);
                v_s12 = v_muladd(v_ad, v_bd, v_s12);
                v_s11 = v_muladd(v_ad, v_ad, v_s11);
                v_s22 = v_muladd(v_bd, v_bd, v_s22);
                v_s1 = v_add(v_s1, v_ad);
                v_s2 = v_add(v_s2, v_bd);
            }
            s12 += v_reduce_sum(v_s12);
            s11 += v_reduce_sum(v_s11);
            s22 += v_reduce_sum(v_s22);
            s1 += v_reduce_sum(v_s1);
            s2 += v_reduce_sum(v_s2);
#elif CV_SIMD && 0 //Disable vectorization for cv::HISTCMP_CORREL if f64 is unsupported due to low precision
            v_float32 v_s1 = vx_setzero_f32();
            v_float32 v_s2 = vx_setzero_f32();
            v_float32 v_s11 = vx_setzero_f32();
            v_float32 v_s12 = vx_setzero_f32();
            v_float32 v_s22 = vx_setzero_f32();
            for (; j <= len - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
            {
                v_float32 v_a = vx_load(h1 + j);
                v_float32 v_b = vx_load(h2 + j);

                v_s12 = v_muladd(v_a, v_b, v_s12);
                v_s11 = v_muladd(v_a, v_a, v_s11);
                v_s22 = v_muladd(v_b, v_b, v_s22);
                v_s1 += v_a;
                v_s2 += v_b;
            }
            s12 += v_reduce_sum(v_s12);
            s11 += v_reduce_sum(v_s11);
            s22 += v_reduce_sum(v_s22);
            s1 += v_reduce_sum(v_s1);
            s2 += v_reduce_sum(v_s2);
#endif
            for( ; j < len; j++ )
            {
                double a = h1[j];
                double b = h2[j];

                s12 += a*b;
                s1 += a;
                s11 += a*a;
                s2 += b;
                s22 += b*b;
            }
        }
        else if( method == cv::HISTCMP_INTERSECT )
        {
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
            v_float64 v_result = vx_setzero_f64();
            for ( ; j <= len - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
            {
                v_float32 v_src = v_min(vx_load(h1 + j), vx_load(h2 + j));
                v_result = v_add(v_result, v_add(v_cvt_f64(v_src), v_cvt_f64_high(v_src)));
            }
            result += v_reduce_sum(v_result);
#elif CV_SIMD
            v_float32 v_result = vx_setzero_f32();
            for (; j <= len - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
            {
                v_float32 v_src = v_min(vx_load(h1 + j), vx_load(h2 + j));
                v_result = v_add(v_result, v_src);
            }
            result += v_reduce_sum(v_result);
#endif
            for( ; j < len; j++ )
                result += std::min(h1[j], h2[j]);
        }
        else if( method == cv::HISTCMP_BHATTACHARYYA )
        {
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
            v_float64 v_s1 = vx_setzero_f64();
            v_float64 v_s2 = vx_setzero_f64();
            v_float64 v_result = vx_setzero_f64();
            for ( ; j <= len - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
            {
                v_float32 v_a = vx_load(h1 + j);
                v_float32 v_b = vx_load(h2 + j);

                v_float64 v_ad = v_cvt_f64(v_a);
                v_float64 v_bd = v_cvt_f64(v_b);
                v_s1 = v_add(v_s1, v_ad);
                v_s2 = v_add(v_s2, v_bd);
                v_result = v_add(v_result, v_sqrt(v_mul(v_ad, v_bd)));

                v_ad = v_cvt_f64_high(v_a);
                v_bd = v_cvt_f64_high(v_b);
                v_s1 = v_add(v_s1, v_ad);
                v_s2 = v_add(v_s2, v_bd);
                v_result = v_add(v_result, v_sqrt(v_mul(v_ad, v_bd)));
            }
            s1 += v_reduce_sum(v_s1);
            s2 += v_reduce_sum(v_s2);
            result += v_reduce_sum(v_result);
#elif CV_SIMD && 0 //Disable vectorization for cv::HISTCMP_BHATTACHARYYA if f64 is unsupported due to low precision
            v_float32 v_s1 = vx_setzero_f32();
            v_float32 v_s2 = vx_setzero_f32();
            v_float32 v_result = vx_setzero_f32();
            for (; j <= len - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
            {
                v_float32 v_a = vx_load(h1 + j);
                v_float32 v_b = vx_load(h2 + j);
                v_s1 += v_a;
                v_s2 += v_b;
                v_result += v_sqrt(v_a * v_b);
            }
            s1 += v_reduce_sum(v_s1);
            s2 += v_reduce_sum(v_s2);
            result += v_reduce_sum(v_result);
#endif
            for( ; j < len; j++ )
            {
                double a = h1[j];
                double b = h2[j];
                result += std::sqrt(a*b);
                s1 += a;
                s2 += b;
            }
        }
        else if( method == cv::HISTCMP_KL_DIV )
        {
            for( ; j < len; j++ )
            {
                double p = h1[j];
                double q = h2[j];
                if( fabs(p) <= DBL_EPSILON ) {
                    continue;
                }
                if(  fabs(q) <= DBL_EPSILON ) {
                    q = 1e-10;
                }
                result += p * std::log( p / q );
            }
        }
        else
            CV_Error( cv::Error::StsBadArg, "Unknown comparison method" );
    }

    if( method == cv::HISTCMP_CHISQR_ALT )
        result *= 2;
    else if( method == cv::HISTCMP_CORREL )
    {
        size_t total = H1.total();
        double scale = 1./total;
        double num = s12 - s1*s2*scale;
        double denom2 = (s11 - s1*s1*scale)*(s22 - s2*s2*scale);
        result = std::abs(denom2) > DBL_EPSILON ? num/std::sqrt(denom2) : 1.;
    }
    else if( method == cv::HISTCMP_BHATTACHARYYA )
    {
        s1 *= s2;
        s1 = fabs(s1) > FLT_EPSILON ? 1./std::sqrt(s1) : 1.;
        result = std::sqrt(std::max(1. - result*s1, 0.));
    }

    return result;
}


double cv::compareHist( const SparseMat& H1, const SparseMat& H2, int method )
{
    CV_INSTRUMENT_REGION();

    double result = 0;
    int i, dims = H1.dims();

    CV_Assert( dims > 0 && dims == H2.dims() && H1.type() == H2.type() && H1.type() == CV_32F );
    for( i = 0; i < dims; i++ )
        CV_Assert( H1.size(i) == H2.size(i) );

    const SparseMat *PH1 = &H1, *PH2 = &H2;
    if( PH1->nzcount() > PH2->nzcount() && method != cv::HISTCMP_CHISQR && method != cv::HISTCMP_CHISQR_ALT && method != cv::HISTCMP_KL_DIV )
        std::swap(PH1, PH2);

    SparseMatConstIterator it = PH1->begin();

    int N1 = (int)PH1->nzcount(), N2 = (int)PH2->nzcount();

    if( (method == cv::HISTCMP_CHISQR) || (method == cv::HISTCMP_CHISQR_ALT) )
    {
        for( i = 0; i < N1; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            float v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            float v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            double a = v1 - v2;
            double b = (method == cv::HISTCMP_CHISQR) ? v1 : v1 + v2;
            if( fabs(b) > DBL_EPSILON )
                result += a*a/b;
        }
    }
    else if( method == cv::HISTCMP_CORREL )
    {
        double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

        for( i = 0; i < N1; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            double v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            s12 += v1*PH2->value<float>(node->idx, (size_t*)&node->hashval);
            s1 += v1;
            s11 += v1*v1;
        }

        it = PH2->begin();
        for( i = 0; i < N2; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            double v2 = it.value<float>();
            s2 += v2;
            s22 += v2*v2;
        }

        size_t total = 1;
        for( i = 0; i < H1.dims(); i++ )
            total *= H1.size(i);
        double scale = 1./total;
        double num = s12 - s1*s2*scale;
        double denom2 = (s11 - s1*s1*scale)*(s22 - s2*s2*scale);
        result = std::abs(denom2) > DBL_EPSILON ? num/std::sqrt(denom2) : 1.;
    }
    else if( method == cv::HISTCMP_INTERSECT )
    {
        for( i = 0; i < N1; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            float v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            float v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            if( v2 )
                result += std::min(v1, v2);
        }
    }
    else if( method == cv::HISTCMP_BHATTACHARYYA )
    {
        double s1 = 0, s2 = 0;

        for( i = 0; i < N1; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            double v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            double v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            result += std::sqrt(v1*v2);
            s1 += v1;
        }

        it = PH2->begin();
        for( i = 0; i < N2; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            s2 += it.value<float>();
        }

        s1 *= s2;
        s1 = fabs(s1) > FLT_EPSILON ? 1./std::sqrt(s1) : 1.;
        result = std::sqrt(std::max(1. - result*s1, 0.));
    }
    else if( method == cv::HISTCMP_KL_DIV )
    {
        for( i = 0; i < N1; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            double v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            double v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            if( !v2 )
                v2 = 1e-10;
            result += v1 * std::log( v1 / v2 );
        }
    }
    else
        CV_Error( cv::Error::StsBadArg, "Unknown comparison method" );

    if( method == cv::HISTCMP_CHISQR_ALT )
        result *= 2;

    return result;
}



class EqualizeHistCalcHist_Invoker : public cv::ParallelLoopBody
{
public:
    enum {HIST_SZ = 256};

    EqualizeHistCalcHist_Invoker(cv::Mat& src, int* histogram, cv::Mutex* histogramLock)
        : src_(src), globalHistogram_(histogram), histogramLock_(histogramLock)
    { }

    void operator()( const cv::Range& rowRange ) const CV_OVERRIDE
    {
        int localHistogram[HIST_SZ] = {0, };

        const size_t sstep = src_.step;

        int width = src_.cols;
        int height = rowRange.end - rowRange.start;

        if (src_.isContinuous())
        {
            width *= height;
            height = 1;
        }

        for (const uchar* ptr = src_.ptr<uchar>(rowRange.start); height--; ptr += sstep)
        {
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                int t0 = ptr[x], t1 = ptr[x+1];
                localHistogram[t0]++; localHistogram[t1]++;
                t0 = ptr[x+2]; t1 = ptr[x+3];
                localHistogram[t0]++; localHistogram[t1]++;
            }

            for (; x < width; ++x)
                localHistogram[ptr[x]]++;
        }

        cv::AutoLock lock(*histogramLock_);

        for( int i = 0; i < HIST_SZ; i++ )
            globalHistogram_[i] += localHistogram[i];
    }

    static bool isWorthParallel( const cv::Mat& src )
    {
        return ( src.total() >= 640*480 );
    }

private:
    EqualizeHistCalcHist_Invoker& operator=(const EqualizeHistCalcHist_Invoker&);

    cv::Mat& src_;
    int* globalHistogram_;
    cv::Mutex* histogramLock_;
};

class EqualizeHistLut_Invoker : public cv::ParallelLoopBody
{
public:
    EqualizeHistLut_Invoker( cv::Mat& src, cv::Mat& dst, int* lut )
        : src_(src),
          dst_(dst),
          lut_(lut)
    { }

    void operator()( const cv::Range& rowRange ) const CV_OVERRIDE
    {
        const size_t sstep = src_.step;
        const size_t dstep = dst_.step;

        int width = src_.cols;
        int height = rowRange.end - rowRange.start;
        int* lut = lut_;

        if (src_.isContinuous() && dst_.isContinuous())
        {
            width *= height;
            height = 1;
        }

        const uchar* sptr = src_.ptr<uchar>(rowRange.start);
        uchar* dptr = dst_.ptr<uchar>(rowRange.start);

        for (; height--; sptr += sstep, dptr += dstep)
        {
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                int v0 = sptr[x];
                int v1 = sptr[x+1];
                int x0 = lut[v0];
                int x1 = lut[v1];
                dptr[x] = (uchar)x0;
                dptr[x+1] = (uchar)x1;

                v0 = sptr[x+2];
                v1 = sptr[x+3];
                x0 = lut[v0];
                x1 = lut[v1];
                dptr[x+2] = (uchar)x0;
                dptr[x+3] = (uchar)x1;
            }

            for (; x < width; ++x)
                dptr[x] = (uchar)lut[sptr[x]];
        }
    }

    static bool isWorthParallel( const cv::Mat& src )
    {
        return ( src.total() >= 640*480 );
    }

private:
    EqualizeHistLut_Invoker& operator=(const EqualizeHistLut_Invoker&);

    cv::Mat& src_;
    cv::Mat& dst_;
    int* lut_;
};

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_equalizeHist(InputArray _src, OutputArray _dst)
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int compunits = dev.maxComputeUnits();
    size_t wgs = dev.maxWorkGroupSize();
    Size size = _src.size();
    bool use16 = size.width % 16 == 0 && _src.offset() % 16 == 0 && _src.step() % 16 == 0;
    int kercn = dev.isAMD() && use16 ? 16 : std::min(4, ocl::predictOptimalVectorWidth(_src));

    ocl::Kernel k1("calculate_histogram", ocl::imgproc::histogram_oclsrc,
                   format("-D BINS=%d -D HISTS_COUNT=%d -D WGS=%zu -D kercn=%d -D T=%s%s",
                          BINS, compunits, wgs, kercn,
                          kercn == 4 ? "int" : ocl::typeToStr(CV_8UC(kercn)),
                          _src.isContinuous() ? " -D HAVE_SRC_CONT" : ""));
    if (k1.empty())
        return false;

    UMat src = _src.getUMat(), ghist(1, BINS * compunits, CV_32SC1);

    k1.args(ocl::KernelArg::ReadOnly(src),
            ocl::KernelArg::PtrWriteOnly(ghist), (int)src.total());

    size_t globalsize = compunits * wgs;
    if (!k1.run(1, &globalsize, &wgs, false))
        return false;

    wgs = std::min<size_t>(ocl::Device::getDefault().maxWorkGroupSize(), BINS);
    UMat lut(1, 256, CV_8UC1);
    ocl::Kernel k2("calcLUT", ocl::imgproc::histogram_oclsrc,
                  format("-D BINS=%d -D HISTS_COUNT=%d -D WGS=%d",
                         BINS, compunits, (int)wgs));
    k2.args(ocl::KernelArg::PtrWriteOnly(lut),
           ocl::KernelArg::PtrReadOnly(ghist), (int)_src.total());

    // calculation of LUT
    if (!k2.run(1, &wgs, &wgs, false))
        return false;

    // execute LUT transparently
    LUT(_src, lut, _dst);
    return true;
}

}

#endif

void cv::equalizeHist( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( _src.type() == CV_8UC1 );

    if (_src.empty())
        return;

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_equalizeHist(_src, _dst))

    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    CALL_HAL(equalizeHist, cv_hal_equalize_hist, src.data, src.step, dst.data, dst.step, src.cols, src.rows);

    Mutex histogramLockInstance;

    const int hist_sz = EqualizeHistCalcHist_Invoker::HIST_SZ;
    int hist[hist_sz] = {0,};
    int lut[hist_sz];

    EqualizeHistCalcHist_Invoker calcBody(src, hist, &histogramLockInstance);
    EqualizeHistLut_Invoker      lutBody(src, dst, lut);
    cv::Range heightRange(0, src.rows);

    if(EqualizeHistCalcHist_Invoker::isWorthParallel(src))
        parallel_for_(heightRange, calcBody);
    else
        calcBody(heightRange);

    int i = 0;
    while (!hist[i]) ++i;

    int total = (int)src.total();
    if (hist[i] == total)
    {
        dst.setTo(i);
        return;
    }

    float scale = (hist_sz - 1.f)/(total - hist[i]);
    int sum = 0;

    for (lut[i++] = 0; i < hist_sz; ++i)
    {
        sum += hist[i];
        lut[i] = saturate_cast<uchar>(sum * scale);
    }

    if(EqualizeHistLut_Invoker::isWorthParallel(src))
        parallel_for_(heightRange, lutBody);
    else
        lutBody(heightRange);
}

/* End of file. */
