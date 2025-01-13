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
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
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

#include <vector>

#include "opencv2/core/hal/intrin.hpp"

#ifdef _MSC_VER
#pragma warning(disable: 4244)  // warning C4244: 'argument': conversion from 'int' to 'ushort', possible loss of data
                                // triggered on intrinsic code from medianBlur_8u_O1()
#endif

/*
 * This file includes the code, contributed by Simon Perreault
 * (the function icvMedianBlur_8u_O1)
 *
 * Constant-time median filtering -- http://nomis80.org/ctmf.html
 * Copyright (C) 2006 Simon Perreault
 *
 * Contact:
 *  Laboratoire de vision et systemes numeriques
 *  Pavillon Adrien-Pouliot
 *  Universite Laval
 *  Sainte-Foy, Quebec, Canada
 *  G1K 7P4
 *
 *  perreaul@gel.ulaval.ca
 */

/****************************************************************************************\
                                      Median Filter
\****************************************************************************************/

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
void medianBlur(const Mat& src0, /*const*/ Mat& dst, int ksize);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if 0 // disabled for now
static void
medianBlur_8u_O1( const Mat& _src, Mat& _dst, int ksize )
{
    CV_INSTRUMENT_REGION();

    typedef ushort HT;

    /**
     * This structure represents a two-tier histogram. The first tier (known as the
     * "coarse" level) is 4 bit wide and the second tier (known as the "fine" level)
     * is 8 bit wide. Pixels inserted in the fine level also get inserted into the
     * coarse bucket designated by the 4 MSBs of the fine bucket value.
     *
     * The structure is aligned on 16 bits, which is a prerequisite for SIMD
     * instructions. Each bucket is 16 bit wide, which means that extra care must be
     * taken to prevent overflow.
     */
    typedef struct
    {
        HT coarse[16];
        HT fine[16][16];
    } Histogram;

/**
 * HOP is short for Histogram OPeration. This macro makes an operation \a op on
 * histogram \a h for pixel value \a x. It takes care of handling both levels.
 */
#define HOP(h,x,op) \
    h.coarse[x>>4] op, \
    *((HT*)h.fine + x) op

#define COP(c,j,x,op) \
    h_coarse[ 16*(n*c+j) + (x>>4) ] op, \
    h_fine[ 16 * (n*(16*c+(x>>4)) + j) + (x & 0xF) ] op

    int cn = _dst.channels(), m = _dst.rows, r = (ksize-1)/2;
    CV_Assert(cn > 0 && cn <= 4);
    size_t sstep = _src.step, dstep = _dst.step;

    int STRIPE_SIZE = std::min( _dst.cols, 512/cn );

#if defined(CV_SIMD_WIDTH) && CV_SIMD_WIDTH >= 16
# define CV_ALIGNMENT CV_SIMD_WIDTH
#else
# define CV_ALIGNMENT 16
#endif

    std::vector<HT> _h_coarse(1 * 16 * (STRIPE_SIZE + 2*r) * cn + CV_ALIGNMENT);
    std::vector<HT> _h_fine(16 * 16 * (STRIPE_SIZE + 2*r) * cn + CV_ALIGNMENT);
    HT* h_coarse = alignPtr(&_h_coarse[0], CV_ALIGNMENT);
    HT* h_fine = alignPtr(&_h_fine[0], CV_ALIGNMENT);

    for( int x = 0; x < _dst.cols; x += STRIPE_SIZE )
    {
        int i, j, k, c, n = std::min(_dst.cols - x, STRIPE_SIZE) + r*2;
        const uchar* src = _src.ptr() + x*cn;
        uchar* dst = _dst.ptr() + (x - r)*cn;

        memset( h_coarse, 0, 16*n*cn*sizeof(h_coarse[0]) );
        memset( h_fine, 0, 16*16*n*cn*sizeof(h_fine[0]) );

        // First row initialization
        for( c = 0; c < cn; c++ )
        {
            for( j = 0; j < n; j++ )
                COP( c, j, src[cn*j+c], += (HT)(r+2) );

            for( i = 1; i < r; i++ )
            {
                const uchar* p = src + sstep*std::min(i, m-1);
                for ( j = 0; j < n; j++ )
                    COP( c, j, p[cn*j+c], ++ );
            }
        }

        for( i = 0; i < m; i++ )
        {
            const uchar* p0 = src + sstep * std::max( 0, i-r-1 );
            const uchar* p1 = src + sstep * std::min( m-1, i+r );

            for( c = 0; c < cn; c++ )
            {
                Histogram CV_DECL_ALIGNED(CV_ALIGNMENT) H;
                HT CV_DECL_ALIGNED(CV_ALIGNMENT) luc[16];

                memset(&H, 0, sizeof(H));
                memset(luc, 0, sizeof(luc));

                // Update column histograms for the entire row.
                for( j = 0; j < n; j++ )
                {
                    COP( c, j, p0[j*cn + c], -- );
                    COP( c, j, p1[j*cn + c], ++ );
                }

                // First column initialization
                for (k = 0; k < 16; ++k)
                {
#if CV_SIMD256
                    v_store(H.fine[k], v_mul_wrap(v256_load(h_fine + 16 * n*(16 * c + k)), v_add(v256_setall_u16(2 * r + 1), v256_load(H.fine[k]))));
#elif CV_SIMD128
                    v_store(H.fine[k], v_add(v_mul_wrap(v_load(h_fine + 16 * n * (16 * c + k)), v_setall_u16((ushort)(2 * r + 1))), v_load(H.fine[k])));
                    v_store(H.fine[k] + 8, v_add(v_mul_wrap(v_load(h_fine + 16 * n * (16 * c + k) + 8), v_setall_u16((ushort)(2 * r + 1))), v_load(H.fine[k] + 8)));
#else
                    for (int ind = 0; ind < 16; ++ind)
                        H.fine[k][ind] = (HT)(H.fine[k][ind] + (2 * r + 1) * h_fine[16 * n*(16 * c + k) + ind]);
#endif
                }

#if CV_SIMD256
                v_uint16x16 v_coarse = v256_load(H.coarse);
#elif CV_SIMD128
                v_uint16x8 v_coarsel = v_load(H.coarse);
                v_uint16x8 v_coarseh = v_load(H.coarse + 8);
#endif
                HT* px = h_coarse + 16 * n*c;
                for( j = 0; j < 2*r; ++j, px += 16 )
                {
#if CV_SIMD256
                    v_coarse = v_add(v_coarse, v256_load(px));
#elif CV_SIMD128
                    v_coarsel = v_add(v_coarsel, v_load(px));
                    v_coarseh = v_add(v_coarseh, v_load(px + 8));
#else
                    for (int ind = 0; ind < 16; ++ind)
                        H.coarse[ind] += px[ind];
#endif
                }

                for( j = r; j < n-r; j++ )
                {
                    int t = 2*r*r + 2*r, b, sum = 0;
                    HT* segment;

                    px = h_coarse + 16 * (n*c + std::min(j + r, n - 1));
#if CV_SIMD256
                    v_coarse = v_add(v_coarse, v256_load(px));
                    v_store(H.coarse, v_coarse);
#elif CV_SIMD128
                    v_coarsel = v_add(v_coarsel, v_load(px));
                    v_coarseh = v_add(v_coarseh, v_load(px + 8));
                    v_store(H.coarse, v_coarsel);
                    v_store(H.coarse + 8, v_coarseh);
#else
                    for (int ind = 0; ind < 16; ++ind)
                        H.coarse[ind] += px[ind];
#endif

                    // Find median at coarse level
                    for ( k = 0; k < 16 ; ++k )
                    {
                        sum += H.coarse[k];
                        if ( sum > t )
                        {
                            sum -= H.coarse[k];
                            break;
                        }
                    }
                    CV_Assert( k < 16 );

                    /* Update corresponding histogram segment */
#if CV_SIMD256
                    v_uint16x16 v_fine;
#elif CV_SIMD128
                    v_uint16x8 v_finel;
                    v_uint16x8 v_fineh;
#endif
                    if ( luc[k] <= j-r )
                    {
#if CV_SIMD256
                        v_fine = v256_setzero_u16();
#elif CV_SIMD128
                        v_finel = v_setzero_u16();
                        v_fineh = v_setzero_u16();
#else
                        memset(&H.fine[k], 0, 16 * sizeof(HT));
#endif
                        px = h_fine + 16 * (n*(16 * c + k) + j - r);
                        for (luc[k] = HT(j - r); luc[k] < MIN(j + r + 1, n); ++luc[k], px += 16)
                        {
#if CV_SIMD256
                            v_fine = v_add(v_fine, v256_load(px));
#elif CV_SIMD128
                            v_finel = v_add(v_finel, v_load(px));
                            v_fineh = v_add(v_fineh, v_load(px + 8));
#else
                            for (int ind = 0; ind < 16; ++ind)
                                H.fine[k][ind] += px[ind];
#endif
                        }

                        if ( luc[k] < j+r+1 )
                        {
                            px = h_fine + 16 * (n*(16 * c + k) + (n - 1));
#if CV_SIMD256
                            v_fine = v_add(v_fine, v_mul_wrap(v256_load(px), v256_setall_u16(j + r + 1 - n)));
#elif CV_SIMD128
                            v_finel = v_add(v_finel, v_mul_wrap(v_load(px), v_setall_u16((ushort)(j + r + 1 - n))));
                            v_fineh = v_add(v_fineh, v_mul_wrap(v_load(px + 8), v_setall_u16((ushort)(j + r + 1 - n))));
#else
                            for (int ind = 0; ind < 16; ++ind)
                                H.fine[k][ind] = (HT)(H.fine[k][ind] + (j + r + 1 - n) * px[ind]);
#endif
                            luc[k] = (HT)(j+r+1);
                        }
                    }
                    else
                    {
#if CV_SIMD256
                        v_fine = v256_load(H.fine[k]);
#elif CV_SIMD128
                        v_finel = v_load(H.fine[k]);
                        v_fineh = v_load(H.fine[k] + 8);
#endif
                        px = h_fine + 16*n*(16 * c + k);
                        for ( ; luc[k] < j+r+1; ++luc[k] )
                        {
#if CV_SIMD256
                            v_fine = v_sub(v_add(v_fine, v256_load(px + 16 * MIN(luc[k], n - 1))), v256_load(px + 16 * MAX(luc[k] - 2 * r - 1, 0)));
#elif CV_SIMD128
                            v_finel = v_sub(v_add(v_finel, v_load(px + 16 * MIN(luc[k], n - 1)    )), v_load(px + 16 * MAX(luc[k] - 2 * r - 1, 0)));
                            v_fineh = v_sub(v_add(v_fineh, v_load(px + 16 * MIN(luc[k], n - 1) + 8)), v_load(px + 16 * MAX(luc[k] - 2 * r - 1, 0) + 8));
#else
                            for (int ind = 0; ind < 16; ++ind)
                                H.fine[k][ind] += px[16 * MIN(luc[k], n - 1) + ind] - px[16 * MAX(luc[k] - 2 * r - 1, 0) + ind];
#endif
                        }
                    }

                    px = h_coarse + 16 * (n*c + MAX(j - r, 0));
#if CV_SIMD256
                    v_store(H.fine[k], v_fine);
                    v_coarse = v_sub(v_coarse, v256_load(px));
#elif CV_SIMD128
                    v_store(H.fine[k], v_finel);
                    v_store(H.fine[k] + 8, v_fineh);
                    v_coarsel = v_sub(v_coarsel, v_load(px));
                    v_coarseh = v_sub(v_coarseh, v_load(px + 8));
#else
                    for (int ind = 0; ind < 16; ++ind)
                        H.coarse[ind] -= px[ind];
#endif

                    /* Find median in segment */
                    segment = H.fine[k];
                    for ( b = 0; b < 16 ; b++ )
                    {
                        sum += segment[b];
                        if ( sum > t )
                        {
                            dst[dstep*i+cn*j+c] = (uchar)(16*k + b);
                            break;
                        }
                    }
                    CV_Assert( b < 16 );
                }
            }
        }
    }

#undef HOP
#undef COP
}

static void
medianBlur_8u_Om( const Mat& _src, Mat& _dst, int m )
{
    CV_INSTRUMENT_REGION();

    #define N  16
    int     zone0[4][N];
    int     zone1[4][N*N];
    int     x, y;
    int     n2 = m*m/2;
    Size    size = _dst.size();
    const uchar* src = _src.ptr();
    uchar*  dst = _dst.ptr();
    int     src_step = (int)_src.step, dst_step = (int)_dst.step;
    int     cn = _src.channels();
    const uchar*  src_max = src + size.height*src_step;
    CV_Assert(cn > 0 && cn <= 4);

    #define UPDATE_ACC01( pix, cn, op ) \
    {                                   \
        int p = (pix);                  \
        zone1[cn][p] op;                \
        zone0[cn][p >> 4] op;           \
    }

    //CV_Assert( size.height >= nx && size.width >= nx );
    for( x = 0; x < size.width; x++, src += cn, dst += cn )
    {
        uchar* dst_cur = dst;
        const uchar* src_top = src;
        const uchar* src_bottom = src;
        int k, c;
        int src_step1 = src_step, dst_step1 = dst_step;

        if( x % 2 != 0 )
        {
            src_bottom = src_top += src_step*(size.height-1);
            dst_cur += dst_step*(size.height-1);
            src_step1 = -src_step1;
            dst_step1 = -dst_step1;
        }

        // init accumulator
        memset( zone0, 0, sizeof(zone0[0])*cn );
        memset( zone1, 0, sizeof(zone1[0])*cn );

        for( y = 0; y <= m/2; y++ )
        {
            for( c = 0; c < cn; c++ )
            {
                if( y > 0 )
                {
                    for( k = 0; k < m*cn; k += cn )
                        UPDATE_ACC01( src_bottom[k+c], c, ++ );
                }
                else
                {
                    for( k = 0; k < m*cn; k += cn )
                        UPDATE_ACC01( src_bottom[k+c], c, += m/2+1 );
                }
            }

            if( (src_step1 > 0 && y < size.height-1) ||
                (src_step1 < 0 && size.height-y-1 > 0) )
                src_bottom += src_step1;
        }

        for( y = 0; y < size.height; y++, dst_cur += dst_step1 )
        {
            // find median
            for( c = 0; c < cn; c++ )
            {
                int s = 0;
                for( k = 0; ; k++ )
                {
                    int t = s + zone0[c][k];
                    if( t > n2 ) break;
                    s = t;
                }

                for( k *= N; ;k++ )
                {
                    s += zone1[c][k];
                    if( s > n2 ) break;
                }

                dst_cur[c] = (uchar)k;
            }

            if( y+1 == size.height )
                break;

            if( cn == 1 )
            {
                for( k = 0; k < m; k++ )
                {
                    int p = src_top[k];
                    int q = src_bottom[k];
                    zone1[0][p]--;
                    zone0[0][p>>4]--;
                    zone1[0][q]++;
                    zone0[0][q>>4]++;
                }
            }
            else if( cn == 3 )
            {
                for( k = 0; k < m*3; k += 3 )
                {
                    UPDATE_ACC01( src_top[k], 0, -- );
                    UPDATE_ACC01( src_top[k+1], 1, -- );
                    UPDATE_ACC01( src_top[k+2], 2, -- );

                    UPDATE_ACC01( src_bottom[k], 0, ++ );
                    UPDATE_ACC01( src_bottom[k+1], 1, ++ );
                    UPDATE_ACC01( src_bottom[k+2], 2, ++ );
                }
            }
            else
            {
                CV_Assert( cn == 4 );
                for( k = 0; k < m*4; k += 4 )
                {
                    UPDATE_ACC01( src_top[k], 0, -- );
                    UPDATE_ACC01( src_top[k+1], 1, -- );
                    UPDATE_ACC01( src_top[k+2], 2, -- );
                    UPDATE_ACC01( src_top[k+3], 3, -- );

                    UPDATE_ACC01( src_bottom[k], 0, ++ );
                    UPDATE_ACC01( src_bottom[k+1], 1, ++ );
                    UPDATE_ACC01( src_bottom[k+2], 2, ++ );
                    UPDATE_ACC01( src_bottom[k+3], 3, ++ );
                }
            }

            if( (src_step1 > 0 && src_bottom + src_step1 < src_max) ||
                (src_step1 < 0 && src_bottom + src_step1 >= src) )
                src_bottom += src_step1;

            if( y >= m/2 )
                src_top += src_step1;
        }
    }
#undef N
#undef UPDATE_ACC
}
#endif

// Binary search to find the median offset in the histogram
static inline void
get_median_ofs256(int &ofs, const uint16_t *hist, const int halfsum) {
    int s = 0, step = 128;
    uint16_t ds, m;

#if CV_SIMD || CV_SIMD_SCALABLE
    const int lanes = VTraits<v_uint16>::vlanes();
#endif

    ofs = 0;
    while (step) {
#if CV_SIMD || CV_SIMD_SCALABLE
        // Use SIMD instructions when the step is larger than or equal to the lane size
        if (step >= lanes) {
            v_uint16 v_ds = vx_load(hist + ofs);
            for (int i = lanes; i <= step - lanes; i += lanes)
                v_ds = v_add(v_ds, vx_load(hist + ofs + i));
            ds = v_reduce_sum(v_ds);
        }
            // For smaller steps, use scalar accumulation
        else
#endif
        {
            ds = hist[ofs];
            for (int i = 1; i < step; i++)
                ds += hist[ofs + i];
        }

        // Determine if the cumulative sum has reached or surpassed half the total sum
        m = (s + ds) <= halfsum;
        ofs += m * step;
        s += ds & -m;
        step = step >> 1;
    }
}

static void
medianBlur_8u(const Mat &_src, Mat &_dst, int ksize) {
    CV_INSTRUMENT_REGION();

    const int channels = _src.channels();
    int radius = ksize / 2;
    CV_Assert(ksize % 2 == 1);

    const int rows = _src.rows;
    const int cols = _src.cols;
    const int win_half_size = ksize * ksize / 2;

    const uint8_t *src_data = _src.ptr<uint8_t>();
    size_t src_step = _src.step;
    uint8_t *dst_data = _dst.ptr<uint8_t>();
    size_t dst_step = _dst.step;

    double nstripes = (double) (rows * cols) / 1024.0;
    parallel_for_(Range(0, rows), [&](const Range &r) {
        int y_start = r.start;
        int y_end = r.end;

        std::vector<std::vector<uint16_t>> hist256(channels, std::vector<uint16_t>(256, 0));
        std::vector<const uint8_t *> win_row_ptrs(ksize);

        // initial hist for the first pixel
        for (int dy = -radius; dy <= radius; dy++) {
            int y = std::clamp(y_start + dy, 0, rows - 1);
            const uint8_t *row_ptr = src_data + y * src_step;
            win_row_ptrs[dy + radius] = row_ptr;

            // always start from col 0, so need to add radius to first pixel
            for (int c = 0; c < channels; ++c)
                hist256[c][row_ptr[c]] += radius + 1;
            // add the rest of the row
            for (int dx = 1; dx <= radius; dx++) {
                int x = std::min(dx, cols - 1);
                const uint8_t *pixel_ptr = row_ptr + x * channels;
                for (int c = 0; c < channels; ++c) {
                    hist256[c][pixel_ptr[c]]++;
                }
            }
        }

        int x_start = 0, x_end = cols, x_step = 1;
        // slide the window across the row
        for (int y = y_start; y < y_end; y++) {
            uint8_t *dst_pixel_ptr = dst_data + y * dst_step + x_start * channels;
            for (int x = x_start; x != x_end; x += x_step, dst_pixel_ptr += x_step * channels) {
                if (x != x_start) {
                    // update hist by removing and adding pixel values as the window moves
                    int x_remove = std::clamp(x - (radius + 1) * x_step, 0, cols - 1) * channels;
                    int x_add = std::clamp(x + radius * x_step, 0, cols - 1) * channels;
                    for (int dy = 0; dy < ksize; dy++) {
                        const uint8_t *win_pixel_ptr = win_row_ptrs[dy];
                        for (int c = 0; c < channels; ++c) {
                            hist256[c][win_pixel_ptr[x_remove + c]]--;
                            hist256[c][win_pixel_ptr[x_add + c]]++;
                        }
                    }
                }

                for (int c = 0; c < channels; ++c) {
                    int ofs = 0;
                    get_median_ofs256(ofs, hist256[c].data(), win_half_size);
                    dst_pixel_ptr[c] = static_cast<uint8_t>(ofs);
                }
            }

            // update hist for the next row
            if (y + 1 < y_end) {
                const uint8_t *old_row_ptr = win_row_ptrs[0];
                int y_new = std::min(y + radius + 1, rows - 1);
                const uint8_t *new_row_ptr = src_data + y_new * src_step;

                for (int dy = 1; dy < ksize; dy++)
                    win_row_ptrs[dy - 1] = win_row_ptrs[dy];
                win_row_ptrs[ksize - 1] = new_row_ptr;

                int dx = (x_end - x_step) * channels;

                for (int c = 0; c < channels; ++c) {
                    hist256[c][old_row_ptr[dx + c]] -= radius + 1;
                    hist256[c][new_row_ptr[dx + c]] += radius + 1;
                }

                for (dx = 2 * channels; dx <= (radius + 1) * channels; dx += channels) {
                    int x = std::clamp(x_end * channels - dx * x_step, 0, (cols - 1) * channels);

                    for (int c = 0; c < channels; ++c) {
                        hist256[c][old_row_ptr[x + c]]--;
                        hist256[c][new_row_ptr[x + c]]++;
                    }
                }

                // reverse the traversal direction for efficient memory access (zigzag pattern)
                dx = x_start;
                x_step = -x_step;
                x_start = x_end + x_step;
                x_end = dx + x_step;
            }
        }
    }, nstripes);
}

namespace {

struct MinMax8u
{
    typedef uchar value_type;
    typedef int arg_type;
    arg_type load(const uchar* ptr) { return *ptr; }
    void store(uchar* ptr, arg_type val) { *ptr = (uchar)val; }
    void operator()(arg_type& a, arg_type& b) const
    {
        int t = CV_FAST_CAST_8U(a - b);
        b += t; a -= t;
    }
};

struct MinMax16u
{
    typedef ushort value_type;
    typedef int arg_type;
    arg_type load(const ushort* ptr) { return *ptr; }
    void store(ushort* ptr, arg_type val) { *ptr = (ushort)val; }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = std::min(a, b);
        b = std::max(b, t);
    }
};

struct MinMax16s
{
    typedef short value_type;
    typedef int arg_type;
    arg_type load(const short* ptr) { return *ptr; }
    void store(short* ptr, arg_type val) { *ptr = (short)val; }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = std::min(a, b);
        b = std::max(b, t);
    }
};

struct MinMax32f
{
    typedef float value_type;
    typedef float arg_type;
    arg_type load(const float* ptr) { return *ptr; }
    void store(float* ptr, arg_type val) { *ptr = val; }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = std::min(a, b);
        b = std::max(b, t);
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)

struct MinMaxVec8u
{
    typedef uchar value_type;
    typedef v_uint8 arg_type;
    arg_type load(const uchar* ptr) { return vx_load(ptr); }
    void store(uchar* ptr, const arg_type &val) { v_store(ptr, val); }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = v_min(a, b);
        b = v_max(b, t);
    }
};


struct MinMaxVec16u
{
    typedef ushort value_type;
    typedef v_uint16 arg_type;
    arg_type load(const ushort* ptr) { return vx_load(ptr); }
    void store(ushort* ptr, const arg_type &val) { v_store(ptr, val); }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = v_min(a, b);
        b = v_max(b, t);
    }
};


struct MinMaxVec16s
{
    typedef short value_type;
    typedef v_int16 arg_type;
    arg_type load(const short* ptr) { return vx_load(ptr); }
    void store(short* ptr, const arg_type &val) { v_store(ptr, val); }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = v_min(a, b);
        b = v_max(b, t);
    }
};


struct MinMaxVec32f
{
    typedef float value_type;
    typedef v_float32 arg_type;
    arg_type load(const float* ptr) { return vx_load(ptr); }
    void store(float* ptr, const arg_type &val) { v_store(ptr, val); }
    void operator()(arg_type& a, arg_type& b) const
    {
        arg_type t = a;
        a = v_min(a, b);
        b = v_max(b, t);
    }
};

#else

typedef MinMax8u MinMaxVec8u;
typedef MinMax16u MinMaxVec16u;
typedef MinMax16s MinMaxVec16s;
typedef MinMax32f MinMaxVec32f;

#endif

template<class Op, class VecOp>
static void
medianBlur_SortNet( const Mat& _src, Mat& _dst, int m )
{
    CV_INSTRUMENT_REGION();

    typedef typename Op::value_type T;
    typedef typename Op::arg_type WT;
    typedef typename VecOp::arg_type VT;

    const T* src = _src.ptr<T>();
    T* dst = _dst.ptr<T>();
    int sstep = (int)(_src.step/sizeof(T));
    int dstep = (int)(_dst.step/sizeof(T));
    Size size = _dst.size();
    int i, j, k, cn = _src.channels();
    Op op;
    VecOp vop;

    if( m == 3 )
    {
        if( size.width == 1 || size.height == 1 )
        {
            int len = size.width + size.height - 1;
            int sdelta = size.height == 1 ? cn : sstep;
            int sdelta0 = size.height == 1 ? 0 : sstep - cn;
            int ddelta = size.height == 1 ? cn : dstep;

            for( i = 0; i < len; i++, src += sdelta0, dst += ddelta )
                for( j = 0; j < cn; j++, src++ )
                {
                    WT p0 = src[i > 0 ? -sdelta : 0];
                    WT p1 = src[0];
                    WT p2 = src[i < len - 1 ? sdelta : 0];

                    op(p0, p1); op(p1, p2); op(p0, p1);
                    dst[j] = (T)p1;
                }
            return;
        }

        size.width *= cn;
        for( i = 0; i < size.height; i++, dst += dstep )
        {
            const T* row0 = src + std::max(i - 1, 0)*sstep;
            const T* row1 = src + i*sstep;
            const T* row2 = src + std::min(i + 1, size.height-1)*sstep;
            int limit = cn;

            for(j = 0;; )
            {
                for( ; j < limit; j++ )
                {
                    int j0 = j >= cn ? j - cn : j;
                    int j2 = j < size.width - cn ? j + cn : j;
                    WT p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
                    WT p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
                    WT p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

                    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
                    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
                    op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
                    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
                    op(p4, p2); op(p6, p4); op(p4, p2);
                    dst[j] = (T)p4;
                }

                if( limit == size.width )
                    break;

#if (CV_SIMD || CV_SIMD_SCALABLE)
                int nlanes = VTraits<typename VecOp::arg_type>::vlanes();
#else
                int nlanes = 1;
#endif
                for( ; j <= size.width - nlanes - cn; j += nlanes )
                {
                    VT p0 = vop.load(row0+j-cn), p1 = vop.load(row0+j), p2 = vop.load(row0+j+cn);
                    VT p3 = vop.load(row1+j-cn), p4 = vop.load(row1+j), p5 = vop.load(row1+j+cn);
                    VT p6 = vop.load(row2+j-cn), p7 = vop.load(row2+j), p8 = vop.load(row2+j+cn);

                    vop(p1, p2); vop(p4, p5); vop(p7, p8); vop(p0, p1);
                    vop(p3, p4); vop(p6, p7); vop(p1, p2); vop(p4, p5);
                    vop(p7, p8); vop(p0, p3); vop(p5, p8); vop(p4, p7);
                    vop(p3, p6); vop(p1, p4); vop(p2, p5); vop(p4, p7);
                    vop(p4, p2); vop(p6, p4); vop(p4, p2);
                    vop.store(dst+j, p4);
                }

                limit = size.width;
            }
        }
    }
    else if( m == 5 )
    {
        if( size.width == 1 || size.height == 1 )
        {
            int len = size.width + size.height - 1;
            int sdelta = size.height == 1 ? cn : sstep;
            int sdelta0 = size.height == 1 ? 0 : sstep - cn;
            int ddelta = size.height == 1 ? cn : dstep;

            for( i = 0; i < len; i++, src += sdelta0, dst += ddelta )
                for( j = 0; j < cn; j++, src++ )
                {
                    int i1 = i > 0 ? -sdelta : 0;
                    int i0 = i > 1 ? -sdelta*2 : i1;
                    int i3 = i < len-1 ? sdelta : 0;
                    int i4 = i < len-2 ? sdelta*2 : i3;
                    WT p0 = src[i0], p1 = src[i1], p2 = src[0], p3 = src[i3], p4 = src[i4];

                    op(p0, p1); op(p3, p4); op(p2, p3); op(p3, p4); op(p0, p2);
                    op(p2, p4); op(p1, p3); op(p1, p2);
                    dst[j] = (T)p2;
                }
            return;
        }

        size.width *= cn;
        for( i = 0; i < size.height; i++, dst += dstep )
        {
            const T* row[5];
            row[0] = src + std::max(i - 2, 0)*sstep;
            row[1] = src + std::max(i - 1, 0)*sstep;
            row[2] = src + i*sstep;
            row[3] = src + std::min(i + 1, size.height-1)*sstep;
            row[4] = src + std::min(i + 2, size.height-1)*sstep;
            int limit = cn*2;

            for(j = 0;; )
            {
                for( ; j < limit; j++ )
                {
                    WT p[25];
                    int j1 = j >= cn ? j - cn : j;
                    int j0 = j >= cn*2 ? j - cn*2 : j1;
                    int j3 = j < size.width - cn ? j + cn : j;
                    int j4 = j < size.width - cn*2 ? j + cn*2 : j3;
                    for( k = 0; k < 5; k++ )
                    {
                        const T* rowk = row[k];
                        p[k*5] = rowk[j0]; p[k*5+1] = rowk[j1];
                        p[k*5+2] = rowk[j]; p[k*5+3] = rowk[j3];
                        p[k*5+4] = rowk[j4];
                    }

                    op(p[1], p[2]); op(p[0], p[1]); op(p[1], p[2]); op(p[4], p[5]); op(p[3], p[4]);
                    op(p[4], p[5]); op(p[0], p[3]); op(p[2], p[5]); op(p[2], p[3]); op(p[1], p[4]);
                    op(p[1], p[2]); op(p[3], p[4]); op(p[7], p[8]); op(p[6], p[7]); op(p[7], p[8]);
                    op(p[10], p[11]); op(p[9], p[10]); op(p[10], p[11]); op(p[6], p[9]); op(p[8], p[11]);
                    op(p[8], p[9]); op(p[7], p[10]); op(p[7], p[8]); op(p[9], p[10]); op(p[0], p[6]);
                    op(p[4], p[10]); op(p[4], p[6]); op(p[2], p[8]); op(p[2], p[4]); op(p[6], p[8]);
                    op(p[1], p[7]); op(p[5], p[11]); op(p[5], p[7]); op(p[3], p[9]); op(p[3], p[5]);
                    op(p[7], p[9]); op(p[1], p[2]); op(p[3], p[4]); op(p[5], p[6]); op(p[7], p[8]);
                    op(p[9], p[10]); op(p[13], p[14]); op(p[12], p[13]); op(p[13], p[14]); op(p[16], p[17]);
                    op(p[15], p[16]); op(p[16], p[17]); op(p[12], p[15]); op(p[14], p[17]); op(p[14], p[15]);
                    op(p[13], p[16]); op(p[13], p[14]); op(p[15], p[16]); op(p[19], p[20]); op(p[18], p[19]);
                    op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[21], p[23]); op(p[22], p[24]);
                    op(p[22], p[23]); op(p[18], p[21]); op(p[20], p[23]); op(p[20], p[21]); op(p[19], p[22]);
                    op(p[22], p[24]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[12], p[18]);
                    op(p[16], p[22]); op(p[16], p[18]); op(p[14], p[20]); op(p[20], p[24]); op(p[14], p[16]);
                    op(p[18], p[20]); op(p[22], p[24]); op(p[13], p[19]); op(p[17], p[23]); op(p[17], p[19]);
                    op(p[15], p[21]); op(p[15], p[17]); op(p[19], p[21]); op(p[13], p[14]); op(p[15], p[16]);
                    op(p[17], p[18]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[0], p[12]);
                    op(p[8], p[20]); op(p[8], p[12]); op(p[4], p[16]); op(p[16], p[24]); op(p[12], p[16]);
                    op(p[2], p[14]); op(p[10], p[22]); op(p[10], p[14]); op(p[6], p[18]); op(p[6], p[10]);
                    op(p[10], p[12]); op(p[1], p[13]); op(p[9], p[21]); op(p[9], p[13]); op(p[5], p[17]);
                    op(p[13], p[17]); op(p[3], p[15]); op(p[11], p[23]); op(p[11], p[15]); op(p[7], p[19]);
                    op(p[7], p[11]); op(p[11], p[13]); op(p[11], p[12]);
                    dst[j] = (T)p[12];
                }

                if( limit == size.width )
                    break;

#if (CV_SIMD || CV_SIMD_SCALABLE)
                int nlanes = VTraits<typename VecOp::arg_type>::vlanes();
#else
                int nlanes = 1;
#endif
                for( ; j <= size.width - nlanes - cn*2; j += nlanes )
                {
                    VT p0 = vop.load(row[0]+j-cn*2), p5 = vop.load(row[1]+j-cn*2), p10 = vop.load(row[2]+j-cn*2), p15 = vop.load(row[3]+j-cn*2), p20 = vop.load(row[4]+j-cn*2);
                    VT p1 = vop.load(row[0]+j-cn*1), p6 = vop.load(row[1]+j-cn*1), p11 = vop.load(row[2]+j-cn*1), p16 = vop.load(row[3]+j-cn*1), p21 = vop.load(row[4]+j-cn*1);
                    VT p2 = vop.load(row[0]+j-cn*0), p7 = vop.load(row[1]+j-cn*0), p12 = vop.load(row[2]+j-cn*0), p17 = vop.load(row[3]+j-cn*0), p22 = vop.load(row[4]+j-cn*0);
                    VT p3 = vop.load(row[0]+j+cn*1), p8 = vop.load(row[1]+j+cn*1), p13 = vop.load(row[2]+j+cn*1), p18 = vop.load(row[3]+j+cn*1), p23 = vop.load(row[4]+j+cn*1);
                    VT p4 = vop.load(row[0]+j+cn*2), p9 = vop.load(row[1]+j+cn*2), p14 = vop.load(row[2]+j+cn*2), p19 = vop.load(row[3]+j+cn*2), p24 = vop.load(row[4]+j+cn*2);

                    vop(p1, p2); vop(p0, p1); vop(p1, p2); vop(p4, p5); vop(p3, p4);
                    vop(p4, p5); vop(p0, p3); vop(p2, p5); vop(p2, p3); vop(p1, p4);
                    vop(p1, p2); vop(p3, p4); vop(p7, p8); vop(p6, p7); vop(p7, p8);
                    vop(p10, p11); vop(p9, p10); vop(p10, p11); vop(p6, p9); vop(p8, p11);
                    vop(p8, p9); vop(p7, p10); vop(p7, p8); vop(p9, p10); vop(p0, p6);
                    vop(p4, p10); vop(p4, p6); vop(p2, p8); vop(p2, p4); vop(p6, p8);
                    vop(p1, p7); vop(p5, p11); vop(p5, p7); vop(p3, p9); vop(p3, p5);
                    vop(p7, p9); vop(p1, p2); vop(p3, p4); vop(p5, p6); vop(p7, p8);
                    vop(p9, p10); vop(p13, p14); vop(p12, p13); vop(p13, p14); vop(p16, p17);
                    vop(p15, p16); vop(p16, p17); vop(p12, p15); vop(p14, p17); vop(p14, p15);
                    vop(p13, p16); vop(p13, p14); vop(p15, p16); vop(p19, p20); vop(p18, p19);
                    vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p21, p23); vop(p22, p24);
                    vop(p22, p23); vop(p18, p21); vop(p20, p23); vop(p20, p21); vop(p19, p22);
                    vop(p22, p24); vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p12, p18);
                    vop(p16, p22); vop(p16, p18); vop(p14, p20); vop(p20, p24); vop(p14, p16);
                    vop(p18, p20); vop(p22, p24); vop(p13, p19); vop(p17, p23); vop(p17, p19);
                    vop(p15, p21); vop(p15, p17); vop(p19, p21); vop(p13, p14); vop(p15, p16);
                    vop(p17, p18); vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p0, p12);
                    vop(p8, p20); vop(p8, p12); vop(p4, p16); vop(p16, p24); vop(p12, p16);
                    vop(p2, p14); vop(p10, p22); vop(p10, p14); vop(p6, p18); vop(p6, p10);
                    vop(p10, p12); vop(p1, p13); vop(p9, p21); vop(p9, p13); vop(p5, p17);
                    vop(p13, p17); vop(p3, p15); vop(p11, p23); vop(p11, p15); vop(p7, p19);
                    vop(p7, p11); vop(p11, p13); vop(p11, p12);
                    vop.store(dst+j, p12);
                }

                limit = size.width;
            }
        }
    }
}

} // namespace anon

void medianBlur(const Mat& src0, /*const*/ Mat& dst, int ksize)
{
    CV_INSTRUMENT_REGION();

    bool useSortNet = ksize == 3 || (ksize == 5
#if !((CV_SIMD || CV_SIMD_SCALABLE))
            && ( src0.depth() > CV_8U || src0.channels() == 2 || src0.channels() > 4 )
#endif
        );

    Mat src;
    if( useSortNet )
    {
        if( dst.data != src0.data )
            src = src0;
        else
            src0.copyTo(src);

        if( src.depth() == CV_8U )
            medianBlur_SortNet<MinMax8u, MinMaxVec8u>( src, dst, ksize );
        else if( src.depth() == CV_16U )
            medianBlur_SortNet<MinMax16u, MinMaxVec16u>( src, dst, ksize );
        else if( src.depth() == CV_16S )
            medianBlur_SortNet<MinMax16s, MinMaxVec16s>( src, dst, ksize );
        else if( src.depth() == CV_32F )
            medianBlur_SortNet<MinMax32f, MinMaxVec32f>( src, dst, ksize );
        else
            CV_Error(cv::Error::StsUnsupportedFormat, "");

        return;
    }
    else
    {
        int cn = src0.channels();
        CV_Assert(src.depth() == CV_8U && (cn == 1 || cn == 3 || cn == 4));

        medianBlur_8u(src0, dst, ksize);
    }
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
