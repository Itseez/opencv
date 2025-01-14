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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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
#include <algorithm>
#include <iterator>

namespace cv
{

// Classical Hough Transform
struct LinePolar
{
    float rho;
    float angle;
};


struct hough_cmp_gt
{
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

static inline int
computeNumangle( double min_theta, double max_theta, double theta_step )
{
    int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;
    // If the distance between the first angle and the last angle is
    // approximately equal to pi, then the last angle will be removed
    // in order to prevent a line to be detected twice.
    if ( numangle > 1 && fabs(CV_PI - (numangle-1)*theta_step) < theta_step/2 )
        --numangle;
    return numangle;
}

static void
createTrigTable( int numangle, double min_theta, double theta_step,
                 float irho, float *tabSin, float *tabCos )
{
    float ang = static_cast<float>(min_theta);
    for(int n = 0; n < numangle; ang += (float)theta_step, n++ )
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }
}

static void
findLocalMaximums( int numrho, int numangle, int threshold,
                   const int *accum, std::vector<int>& sort_buf )
{
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf.push_back(base);
        }
}

/*
Here image is an input raster;
step is it's step; size characterizes it's ROI;
rho and theta are discretization steps (in pixels and radians correspondingly).
threshold is the minimum number of pixels in the feature for it
to be a candidate for line. lines is the output
array of (rho, theta) pairs. linesMax is the buffer size (number of pairs).
Functions return the actual number of found lines.
*/
static void
HoughLinesStandard( InputArray src, OutputArray lines, int type,
                    float rho, float theta,
                    int threshold, int linesMax,
                    double min_theta, double max_theta, bool use_edgeval = false )
{
    CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Internal error");

    Mat img = src.getMat();

    int i, j;
    float irho = 1 / rho;

    CV_Assert( img.type() == CV_8UC1 );
    CV_Assert( linesMax > 0 );

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;

    int max_rho = width + height;
    int min_rho = -max_rho;

    CV_CheckGE(max_theta, min_theta, "max_theta must be greater than min_theta");

    int numangle = computeNumangle(min_theta, max_theta, theta);
    int numrho = cvRound(((max_rho - min_rho) + 1) / rho);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    if (type == CV_32FC2 && CV_IPP_CHECK_COND)
    {
        IppiSize srcSize = { width, height };
        IppPointPolar delta = { rho, theta };
        IppPointPolar dstRoi[2] = {{(Ipp32f) min_rho, (Ipp32f) min_theta},{(Ipp32f) max_rho, (Ipp32f) max_theta}};
        int bufferSize;
        int nz = countNonZero(img);
        int ipp_linesMax = std::min(linesMax, nz*numangle/threshold);
        int linesCount = 0;
        std::vector<Vec2f> _lines(ipp_linesMax);
        IppStatus ok = ippiHoughLineGetSize_8u_C1R(srcSize, delta, ipp_linesMax, &bufferSize);
        Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
        if (ok >= 0) {ok = CV_INSTRUMENT_FUN_IPP(ippiHoughLine_Region_8u32f_C1R, image, step, srcSize, (IppPointPolar*) &_lines[0], dstRoi, ipp_linesMax, &linesCount, delta, threshold, buffer);};
        ippsFree(buffer);
        if (ok >= 0)
        {
            lines.create(linesCount, 1, CV_32FC2);
            Mat(linesCount, 1, CV_32FC2, &_lines[0]).copyTo(lines);
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif


    Mat _accum = Mat::zeros( (numangle+2), (numrho+2), CV_32SC1 );
    std::vector<int> _sort_buf;
    AutoBuffer<float> _tabSin(numangle);
    AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum.ptr<int>();
    float *tabSin = _tabSin.data(), *tabCos = _tabCos.data();

    // create sin and cos table
    createTrigTable( numangle, min_theta, theta,
                     irho, tabSin, tabCos);

    // stage 1. fill accumulator
    if (use_edgeval) {
        for( i = 0; i < height; i++ )
            for( j = 0; j < width; j++ )
            {
                if( image[i * step + j] != 0 )
                    for(int n = 0; n < numangle; n++ )
                    {
                        int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                        r += (numrho - 1) / 2;
                        accum[(n + 1) * (numrho + 2) + r + 1] += image[i * step + j];
                     }
             }
    } else {
        for( i = 0; i < height; i++ )
            for( j = 0; j < width; j++ )
            {
                if( image[i * step + j] != 0 )
                    for(int n = 0; n < numangle; n++ )
                    {
                        int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                        r += (numrho - 1) / 2;
                        accum[(n + 1) * (numrho + 2) + r + 1]++;
                    }
            }
     }

    // stage 2. find local maximums
    findLocalMaximums( numrho, numangle, threshold, accum, _sort_buf );

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    double scale = 1./(numrho+2);

    lines.create(linesMax, 1, type);
    Mat _lines = lines.getMat();
    for( i = 0; i < linesMax; i++ )
    {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = (r - (numrho - 1)*0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        if (type == CV_32FC2)
        {
            _lines.at<Vec2f>(i) = Vec2f(line.rho, line.angle);
        }
        else
        {
            CV_DbgAssert(type == CV_32FC3);
            _lines.at<Vec3f>(i) = Vec3f(line.rho, line.angle, (float)accum[idx]);
        }
    }
}


// Multi-Scale variant of Classical Hough Transform

struct hough_index
{
    hough_index() : value(0), rho(0.f), theta(0.f) {}
    hough_index(int _val, float _rho, float _theta)
    : value(_val), rho(_rho), theta(_theta) {}

    int value;
    float rho, theta;
};


static void
HoughLinesSDiv( InputArray image, OutputArray lines, int type,
                float rho, float theta, int threshold,
                int srn, int stn, int linesMax,
                double min_theta, double max_theta )
{
    CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Internal error");

    #define _POINT(row, column)\
        (image_src[(row)*step+(column)])

    Mat img = image.getMat();
    int index, i;
    int ri, ti, ti1, ti0;
    int row, col;
    float r, t;                 /* Current rho and theta */
    float rv;                   /* Some temporary rho value */

    int fn = 0;
    float xc, yc;

    const float d2r = (float)(CV_PI / 180);
    int sfn = srn * stn;
    int fi;
    int count;
    int cmax = 0;

    std::vector<hough_index> lst;

    CV_Assert( img.type() == CV_8UC1 );
    CV_Assert( linesMax > 0 );

    threshold = MIN( threshold, 255 );

    const uchar* image_src = img.ptr();
    int step = (int)img.step;
    int w = img.cols;
    int h = img.rows;

    float irho = 1 / rho;
    float itheta = 1 / theta;
    float srho = rho / srn;
    float stheta = theta / stn;
    float isrho = 1 / srho;
    float istheta = 1 / stheta;

    int rn = cvFloor( std::sqrt( (double)w * w + (double)h * h ) * irho );
    int tn = cvFloor( 2 * CV_PI * itheta );

    lst.push_back(hough_index(threshold, -1.f, 0.f));

    // Precalculate sin table
    std::vector<float> _sinTable( 5 * tn * stn );
    float* sinTable = &_sinTable[0];

    for( index = 0; index < 5 * tn * stn; index++ )
        sinTable[index] = (float)cos( stheta * index * 0.2f );

    std::vector<uchar> _caccum(rn * tn, (uchar)0);
    uchar* caccum = &_caccum[0];

    // Counting all feature pixels
    for( row = 0; row < h; row++ )
        for( col = 0; col < w; col++ )
            fn += _POINT( row, col ) != 0;

    std::vector<int> _x(fn), _y(fn);
    int* x = &_x[0], *y = &_y[0];

    // Full Hough Transform (it's accumulator update part)
    fi = 0;
    for( row = 0; row < h; row++ )
    {
        for( col = 0; col < w; col++ )
        {
            if( _POINT( row, col ))
            {
                int halftn;
                float r0;
                float scale_factor;
                int iprev = -1;
                float phi, phi1;
                float theta_it;     // Value of theta for iterating

                // Remember the feature point
                x[fi] = col;
                y[fi] = row;
                fi++;

                yc = (float) row + 0.5f;
                xc = (float) col + 0.5f;

                /* Update the accumulator */
                t = (float) fabs( fastAtan2( yc, xc ) * d2r );
                r = (float) std::sqrt( (double)xc * xc + (double)yc * yc );
                r0 = r * irho;
                ti0 = cvFloor( (t + CV_PI*0.5) * itheta );

                caccum[ti0]++;

                theta_it = rho / r;
                theta_it = theta_it < theta ? theta_it : theta;
                scale_factor = theta_it * itheta;
                halftn = cvFloor( CV_PI / theta_it );
                for( ti1 = 1, phi = theta_it - (float)(CV_PI*0.5), phi1 = (theta_it + t) * itheta;
                     ti1 < halftn; ti1++, phi += theta_it, phi1 += scale_factor )
                {
                    rv = r0 * std::cos( phi );
                    i = (int)rv * tn;
                    i += cvFloor( phi1 );
                    CV_Assert( i >= 0 );
                    CV_Assert( i < rn * tn );
                    caccum[i] = (uchar) (caccum[i] + ((i ^ iprev) != 0));
                    iprev = i;
                    if( cmax < caccum[i] )
                        cmax = caccum[i];
                }
            }
        }
    }

    // Starting additional analysis
    count = 0;
    for( ri = 0; ri < rn; ri++ )
    {
        for( ti = 0; ti < tn; ti++ )
        {
            if( caccum[ri * tn + ti] > threshold )
                count++;
        }
    }

    if( count * 100 > rn * tn )
    {
        HoughLinesStandard( image, lines, type, rho, theta, threshold, linesMax, min_theta, max_theta );
        return;
    }

    std::vector<uchar> _buffer(srn * stn + 2);
    uchar* buffer = &_buffer[0];
    uchar* mcaccum = buffer + 1;

    count = 0;
    for( ri = 0; ri < rn; ri++ )
    {
        for( ti = 0; ti < tn; ti++ )
        {
            if( caccum[ri * tn + ti] > threshold )
            {
                count++;
                memset( mcaccum, 0, sfn * sizeof( uchar ));

                for( index = 0; index < fn; index++ )
                {
                    int ti2;
                    float r0;

                    yc = (float) y[index] + 0.5f;
                    xc = (float) x[index] + 0.5f;

                    // Update the accumulator
                    t = (float) fabs( fastAtan2( yc, xc ) * d2r );
                    r = (float) std::sqrt( (double)xc * xc + (double)yc * yc ) * isrho;
                    ti0 = cvFloor( (t + CV_PI * 0.5) * istheta );
                    ti2 = (ti * stn - ti0) * 5;
                    r0 = (float) ri *srn;

                    for( ti1 = 0; ti1 < stn; ti1++, ti2 += 5 )
                    {
                        rv = r * sinTable[(int) (std::abs( ti2 ))] - r0;
                        i = cvFloor( rv ) * stn + ti1;

                        i = CV_IMAX( i, -1 );
                        i = CV_IMIN( i, sfn );
                        mcaccum[i]++;
                        CV_Assert( i >= -1 );
                        CV_Assert( i <= sfn );
                    }
                }

                // Find peaks in maccum...
                for( index = 0; index < sfn; index++ )
                {
                    int pos = (int)(lst.size() - 1);
                    if( pos < 0 || lst[pos].value < mcaccum[index] )
                    {
                        hough_index vi(mcaccum[index],
                                       index / stn * srho + ri * rho,
                                       index % stn * stheta + ti * theta - (float)(CV_PI*0.5));
                        lst.push_back(vi);
                        for( ; pos >= 0; pos-- )
                        {
                            if( lst[pos].value > vi.value )
                                break;
                            lst[pos+1] = lst[pos];
                        }
                        lst[pos+1] = vi;
                        if( (int)lst.size() > linesMax )
                            lst.pop_back();
                    }
                }
            }
        }
    }

    int pos = (int)(lst.size() - 1);
    if( pos >= 0 && lst[pos].rho < 0 )
        lst.pop_back();

    lines.create((int)lst.size(), 1, type);
    Mat _lines = lines.getMat();
    for( size_t idx = 0; idx < lst.size(); idx++ )
    {
        if (type == CV_32FC2)
        {
            _lines.at<Vec2f>((int)idx) = Vec2f(lst[idx].rho, lst[idx].theta);
        }
        else
        {
            CV_DbgAssert(type == CV_32FC3);
            _lines.at<Vec3f>((int)idx) = Vec3f(lst[idx].rho, lst[idx].theta, (float)lst[idx].value);
        }
    }
}


/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

static void
HoughLinesProbabilistic( Mat& image,
                         float rho, float theta, int threshold,
                         int lineLength, int lineGap,
                         std::vector<Vec4i>& lines, int linesMax )
{
    Point pt;
    float irho = 1 / rho;
    RNG rng((uint64)-1);

    CV_Assert( image.type() == CV_8UC1 );

    int width = image.cols;
    int height = image.rows;

    int numangle = computeNumangle(0.0, CV_PI, theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    CV_IPP_CHECK()
    {
        IppiSize srcSize = { width, height };
        IppPointPolar delta = { rho, theta };
        IppiHoughProbSpec* pSpec;
        int bufferSize, specSize;
        int ipp_linesMax = std::min(linesMax, numangle*numrho);
        int linesCount = 0;
        lines.resize(ipp_linesMax);
        IppStatus ok = ippiHoughProbLineGetSize_8u_C1R(srcSize, delta, &specSize, &bufferSize);
        Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
        pSpec = (IppiHoughProbSpec*) ippsMalloc_8u_L(specSize);
        if (ok >= 0) ok = ippiHoughProbLineInit_8u32f_C1R(srcSize, delta, ippAlgHintNone, pSpec);
        if (ok >= 0) {ok = CV_INSTRUMENT_FUN_IPP(ippiHoughProbLine_8u32f_C1R, image.data, (int)image.step, srcSize, threshold, lineLength, lineGap, (IppiPoint*) &lines[0], ipp_linesMax, &linesCount, buffer, pSpec);};

        ippsFree(pSpec);
        ippsFree(buffer);
        if (ok >= 0)
        {
            lines.resize(linesCount);
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        lines.clear();
        setIppErrorStatus();
    }
#endif

    Mat accum = Mat::zeros( numangle, numrho, CV_32SC1 );
    Mat mask( height, width, CV_8UC1 );
    std::vector<float> trigtab(numangle*2);

    for( int n = 0; n < numangle; n++ )
    {
        trigtab[n*2] = (float)(cos((double)n*theta) * irho);
        trigtab[n*2+1] = (float)(sin((double)n*theta) * irho);
    }
    const float* ttab = &trigtab[0];
    uchar* mdata0 = mask.ptr();
    std::vector<Point> nzloc;

    // stage 1. collect non-zero image points
    for( pt.y = 0; pt.y < height; pt.y++ )
    {
        const uchar* data = image.ptr(pt.y);
        uchar* mdata = mask.ptr(pt.y);
        for( pt.x = 0; pt.x < width; pt.x++ )
        {
            if( data[pt.x] )
            {
                mdata[pt.x] = (uchar)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();

    // stage 2. process all the points in random order
    for( ; count > 0; count-- )
    {
        // choose random point out of the remaining ones
        int idx = rng.uniform(0, count);
        int max_val = threshold-1, max_n = 0;
        Point point = nzloc[idx];
        Point line_end[2];
        float a, b;
        int* adata = accum.ptr<int>();
        int i = point.y, j = point.x, k,  xflag;
        int64_t x0, y0, dx0, dy0;
        int good_line;
        const int shift = 16;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count-1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if( !mdata0[i*width + j] )
            continue;

        // update accumulator, find the most probable line
        for( int n = 0; n < numangle; n++, adata += numrho )
        {
            int r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if( max_val < threshold )
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n*2+1];
        b = ttab[max_n*2];
        x0 = j;
        y0 = i;
        if( fabs(a) > fabs(b) )
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for( k = 0; k < 2; k++ )
        {
            int gap = 0;
            int64_t x = x0, y = y0, dx = dx0, dy = dy0 ;
            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int64_t i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    gap = 0;
                    line_end[k].y = static_cast<int>(i1);
                    line_end[k].x = static_cast<int>(j1);
                }
                else if( ++gap > lineGap )
                    break;
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    std::abs(line_end[1].y - line_end[0].y) >= lineLength;

        for( k = 0; k < 2; k++ )
        {
            int64_t x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int64_t i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    if( good_line )
                    {
                        adata = accum.ptr<int>();
                        for( int n = 0; n < numangle; n++, adata += numrho )
                        {
                            int r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }

        if( good_line )
        {
            Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
            lines.push_back(lr);
            if( (int)lines.size() >= linesMax )
                return;
        }
    }
}

#ifdef HAVE_OPENCL

#define OCL_MAX_LINES 4096

static bool ocl_makePointsList(InputArray _src, OutputArray _pointsList, InputOutputArray _counters)
{
    UMat src = _src.getUMat();
    _pointsList.create(1, (int) src.total(), CV_32SC1);
    UMat pointsList = _pointsList.getUMat();
    UMat counters = _counters.getUMat();
    ocl::Device dev = ocl::Device::getDefault();

    const int pixPerWI = 16;
    int workgroup_size = min((int) dev.maxWorkGroupSize(), (src.cols + pixPerWI - 1)/pixPerWI);
    ocl::Kernel pointListKernel("make_point_list", ocl::imgproc::hough_lines_oclsrc,
                                format("-D MAKE_POINTS_LIST -D GROUP_SIZE=%d -D LOCAL_SIZE=%d", workgroup_size, src.cols));
    if (pointListKernel.empty())
        return false;

    pointListKernel.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnlyNoSize(pointsList),
                         ocl::KernelArg::PtrWriteOnly(counters));

    size_t localThreads[2]  = { (size_t)workgroup_size, 1 };
    size_t globalThreads[2] = { (size_t)workgroup_size, (size_t)src.rows };

    return pointListKernel.run(2, globalThreads, localThreads, false);
}

static bool ocl_fillAccum(InputArray _pointsList, OutputArray _accum, int total_points, double rho, double theta, int numrho, int numangle)
{
    UMat pointsList = _pointsList.getUMat();
    _accum.create(numangle + 2, numrho + 2, CV_32SC1);
    UMat accum = _accum.getUMat();
    ocl::Device dev = ocl::Device::getDefault();

    float irho = (float) (1 / rho);
    int workgroup_size = min((int) dev.maxWorkGroupSize(), total_points);

    ocl::Kernel fillAccumKernel;
    size_t localThreads[2];
    size_t globalThreads[2];

    size_t local_memory_needed = (numrho + 2)*sizeof(int);
    if (local_memory_needed > dev.localMemSize())
    {
        accum.setTo(Scalar::all(0));
        fillAccumKernel.create("fill_accum_global", ocl::imgproc::hough_lines_oclsrc,
                                format("-D FILL_ACCUM_GLOBAL"));
        if (fillAccumKernel.empty())
            return false;
        globalThreads[0] = workgroup_size; globalThreads[1] = numangle;
        fillAccumKernel.args(ocl::KernelArg::ReadOnlyNoSize(pointsList), ocl::KernelArg::WriteOnlyNoSize(accum),
                        total_points, irho, (float) theta, numrho, numangle);
        return fillAccumKernel.run(2, globalThreads, NULL, false);
    }
    else
    {
        fillAccumKernel.create("fill_accum_local", ocl::imgproc::hough_lines_oclsrc,
                                format("-D FILL_ACCUM_LOCAL -D LOCAL_SIZE=%d -D BUFFER_SIZE=%d", workgroup_size, numrho + 2));
        if (fillAccumKernel.empty())
            return false;
        localThreads[0] = workgroup_size; localThreads[1] = 1;
        globalThreads[0] = workgroup_size; globalThreads[1] = numangle+2;
        fillAccumKernel.args(ocl::KernelArg::ReadOnlyNoSize(pointsList), ocl::KernelArg::WriteOnlyNoSize(accum),
                        total_points, irho, (float) theta, numrho, numangle);
        return fillAccumKernel.run(2, globalThreads, localThreads, false);
    }
}

static bool ocl_HoughLines(InputArray _src, OutputArray _lines, double rho, double theta, int threshold,
                           double min_theta, double max_theta)
{
    CV_Assert(_src.type() == CV_8UC1);

    if (max_theta < 0 || max_theta > CV_PI ) {
        CV_Error( Error::StsBadArg, "max_theta must fall between 0 and pi" );
    }
    if (min_theta < 0 || min_theta > max_theta ) {
        CV_Error( Error::StsBadArg, "min_theta must fall between 0 and max_theta" );
    }
    if (!(rho > 0 && theta > 0)) {
        CV_Error( Error::StsBadArg, "rho and theta must be greater 0" );
    }

    UMat src = _src.getUMat();
    int numangle = computeNumangle(min_theta, max_theta, theta);
    int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

    UMat pointsList;
    UMat counters(1, 2, CV_32SC1, Scalar::all(0));

    if (!ocl_makePointsList(src, pointsList, counters))
        return false;

    int total_points = counters.getMat(ACCESS_READ).at<int>(0, 0);
    if (total_points <= 0)
    {
        _lines.release();
        return true;
    }

    UMat accum;
    if (!ocl_fillAccum(pointsList, accum, total_points, rho, theta, numrho, numangle))
        return false;

    const int pixPerWI = 8;
    ocl::Kernel getLinesKernel("get_lines", ocl::imgproc::hough_lines_oclsrc,
                               format("-D GET_LINES"));
    if (getLinesKernel.empty())
        return false;

    int linesMax = threshold > 0 ? min(total_points*numangle/threshold, OCL_MAX_LINES) : OCL_MAX_LINES;
    UMat lines(linesMax, 1, CV_32FC2);

    getLinesKernel.args(ocl::KernelArg::ReadOnly(accum), ocl::KernelArg::WriteOnlyNoSize(lines),
                        ocl::KernelArg::PtrWriteOnly(counters), linesMax, threshold, (float) rho, (float) theta);

    size_t globalThreads[2] = { ((size_t)numrho + pixPerWI - 1)/pixPerWI, (size_t)numangle };
    if (!getLinesKernel.run(2, globalThreads, NULL, false))
        return false;

    int total_lines = min(counters.getMat(ACCESS_READ).at<int>(0, 1), linesMax);
    if (total_lines > 0)
        _lines.assign(lines.rowRange(Range(0, total_lines)));
    else
        _lines.release();
    return true;
}

static bool ocl_HoughLinesP(InputArray _src, OutputArray _lines, double rho, double theta, int threshold,
                           double minLineLength, double maxGap)
{
    CV_Assert(_src.type() == CV_8UC1);

    if (!(rho > 0 && theta > 0)) {
        CV_Error( Error::StsBadArg, "rho and theta must be greater 0" );
    }

    UMat src = _src.getUMat();
    int numangle = computeNumangle(0.0, CV_PI, theta);
    int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

    UMat pointsList;
    UMat counters(1, 2, CV_32SC1, Scalar::all(0));

    if (!ocl_makePointsList(src, pointsList, counters))
        return false;

    int total_points = counters.getMat(ACCESS_READ).at<int>(0, 0);
    if (total_points <= 0)
    {
        _lines.release();
        return true;
    }

    UMat accum;
    if (!ocl_fillAccum(pointsList, accum, total_points, rho, theta, numrho, numangle))
        return false;

    ocl::Kernel getLinesKernel("get_lines", ocl::imgproc::hough_lines_oclsrc,
                               format("-D GET_LINES_PROBABOLISTIC"));
    if (getLinesKernel.empty())
        return false;

    int linesMax = threshold > 0 ? min(total_points*numangle/threshold, OCL_MAX_LINES) : OCL_MAX_LINES;
    UMat lines(linesMax, 1, CV_32SC4);

    getLinesKernel.args(ocl::KernelArg::ReadOnly(accum), ocl::KernelArg::ReadOnly(src),
                        ocl::KernelArg::WriteOnlyNoSize(lines), ocl::KernelArg::PtrWriteOnly(counters),
                        linesMax, threshold, (int) minLineLength, (int) maxGap, (float) rho, (float) theta);

    size_t globalThreads[2] = { (size_t)numrho, (size_t)numangle };
    if (!getLinesKernel.run(2, globalThreads, NULL, false))
        return false;

    int total_lines = min(counters.getMat(ACCESS_READ).at<int>(0, 1), linesMax);
    if (total_lines > 0)
        _lines.assign(lines.rowRange(Range(0, total_lines)));
    else
        _lines.release();

    return true;
}

#endif /* HAVE_OPENCL */

void HoughLines( InputArray _image, OutputArray lines,
                 double rho, double theta, int threshold,
                 double srn, double stn, double min_theta, double max_theta, bool use_edgeval )
{
    CV_INSTRUMENT_REGION();

    int type = CV_32FC2;
    if (lines.fixedType())
    {
        type = lines.type();
        CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Wrong type of output lines");
    }

    CV_OCL_RUN(srn == 0 && stn == 0 && _image.isUMat() && lines.isUMat() && type == CV_32FC2,
               ocl_HoughLines(_image, lines, rho, theta, threshold, min_theta, max_theta));

    if( srn == 0 && stn == 0 )
        HoughLinesStandard(_image, lines, type, (float)rho, (float)theta, threshold, INT_MAX, min_theta, max_theta, use_edgeval );
    else
        HoughLinesSDiv(_image, lines, type, (float)rho, (float)theta, threshold, cvRound(srn), cvRound(stn), INT_MAX, min_theta, max_theta);
}


void HoughLinesP(InputArray _image, OutputArray _lines,
                 double rho, double theta, int threshold,
                 double minLineLength, double maxGap )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_image.isUMat() && _lines.isUMat(),
               ocl_HoughLinesP(_image, _lines, rho, theta, threshold, minLineLength, maxGap));

    Mat image = _image.getMat();
    std::vector<Vec4i> lines;
    HoughLinesProbabilistic(image, (float)rho, (float)theta, threshold, cvRound(minLineLength), cvRound(maxGap), lines, INT_MAX);
    Mat(lines).copyTo(_lines);
}

void HoughLinesPointSet( InputArray _point, OutputArray _lines, int lines_max, int threshold,
                         double min_rho, double max_rho, double rho_step,
                         double min_theta, double max_theta, double theta_step )
{
    std::vector<Vec3d> lines;
    std::vector<Point2f> point;
    _point.copyTo(point);

    CV_Assert( _point.type() == CV_32FC2 || _point.type() == CV_32SC2 );
    if( lines_max <= 0 ) {
        CV_Error( Error::StsBadArg, "lines_max must be greater than 0" );
    }
    if( threshold < 0) {
        CV_Error( Error::StsBadArg, "threshold must be greater than 0" );
    }
    if( ((max_rho - min_rho) <= 0) || ((max_theta - min_theta) <= 0) ) {
        CV_Error( Error::StsBadArg, "max must be greater than min" );
    }
    if( ((rho_step <= 0)) || ((theta_step <= 0)) ) {
        CV_Error( Error::StsBadArg, "step must be greater than 0" );
    }

    int i;
    float irho = 1 / (float)rho_step;
    float irho_min = ((float)min_rho * irho);
    int numangle = computeNumangle(min_theta, max_theta, theta_step);
    int numrho = cvRound((max_rho - min_rho + 1) / rho_step);

    Mat _accum = Mat::zeros( (numangle+2), (numrho+2), CV_32SC1 );
    std::vector<int> _sort_buf;
    AutoBuffer<float> _tabSin(numangle);
    AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum.ptr<int>();
    float *tabSin = _tabSin.data(), *tabCos = _tabCos.data();

    // create sin and cos table
    createTrigTable( numangle, min_theta, theta_step,
                     irho, tabSin, tabCos );

    // stage 1. fill accumulator
    for( i = 0; i < (int)point.size(); i++ )
        for(int n = 0; n < numangle; n++ )
        {
            int r = cvRound( point.at(i).x  * tabCos[n] + point.at(i).y * tabSin[n] - irho_min);
            if ( r >= 0 && r <= numrho) {
                accum[(n+1) * (numrho+2) + r+1]++;
            }
        }

    // stage 2. find local maximums
    findLocalMaximums( numrho, numangle, threshold, accum, _sort_buf );

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    lines_max = std::min(lines_max, (int)_sort_buf.size());
    double scale = 1./(numrho+2);
    for( i = 0; i < lines_max; i++ )
    {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = static_cast<float>(min_rho) + r * (float)rho_step;
        line.angle = static_cast<float>(min_theta) + n * (float)theta_step;
        lines.push_back(Vec3d((double)accum[idx], (double)line.rho, (double)line.angle));
    }

    Mat(lines).copyTo(_lines);
}

/****************************************************************************************\
*                                     Circle Detection                                   *
\****************************************************************************************/

struct EstimatedCircle
{
    EstimatedCircle() { accum = 0; }
    EstimatedCircle(Vec3f _c, int _accum) :
        c(_c), accum(_accum) {}
    Vec3f c;
    int accum;
};

static bool cmpAccum(const EstimatedCircle& left, const EstimatedCircle& right)
{
    // Compare everything so the order is completely deterministic
    // Larger accum first
    if (left.accum > right.accum)
        return true;
    else if (left.accum < right.accum)
        return false;
    // Larger radius first
    else if (left.c[2] > right.c[2])
        return true;
    else if (left.c[2] < right.c[2])
        return false;
    // Smaller X
    else if (left.c[0] < right.c[0])
        return true;
    else if (left.c[0] > right.c[0])
        return false;
    // Smaller Y
    else if (left.c[1] < right.c[1])
        return true;
    else if (left.c[1] > right.c[1])
        return false;
    // Identical - neither object is less than the other
    else
        return false;
}

static inline Vec3f GetCircle(const EstimatedCircle& est)
{
    return est.c;
}

static inline Vec4f GetCircle4f(const EstimatedCircle& est)
{
    return Vec4f(est.c[0], est.c[1], est.c[2], (float)est.accum);
}

class NZPointList : public std::vector<Point>
{
private:
    NZPointList(const NZPointList& other); // non-copyable

public:
    NZPointList(int reserveSize = 256)
    {
        reserve(reserveSize);
    }
};

class NZPointSet
{
private:
    NZPointSet(const NZPointSet& other); // non-copyable

public:
    Mat_<uchar> positions;

    NZPointSet(int rows, int cols) :
        positions(rows, cols, (uchar)0)
    {
    }

    void insert(const Point& pt)
    {
        positions(pt) = 1;
    }

    void insert(const NZPointSet& from)
    {
        cv::bitwise_or(from.positions, positions, positions);
    }

    void toList(NZPointList& list) const
    {
        for (int y = 0; y < positions.rows; y++)
        {
            const uchar *ptr = positions.ptr<uchar>(y, 0);
            for (int x = 0; x < positions.cols; x++)
            {
                if (ptr[x])
                {
                    list.push_back(Point(x, y));
                }
            }
        }
    }
};

class HoughCirclesAccumInvoker : public ParallelLoopBody
{
public:
    HoughCirclesAccumInvoker(const Mat &_edges, const Mat &_dx, const Mat &_dy, int _minRadius, int _maxRadius, float _idp,
                             std::vector<Mat>& _accumVec, NZPointSet& _nz, Mutex& _mtx) :
        edges(_edges), dx(_dx), dy(_dy), minRadius(_minRadius), maxRadius(_maxRadius), idp(_idp),
        accumVec(_accumVec), nz(_nz), mutex(_mtx)
    {
        acols = cvCeil(edges.cols * idp), arows = cvCeil(edges.rows * idp);
        astep = acols + 2;
    }

    ~HoughCirclesAccumInvoker() { }

    void operator()(const Range &boundaries) const CV_OVERRIDE
    {
        Mat accumLocal = Mat(arows + 2, acols + 2, CV_32SC1, Scalar::all(0));
        int *adataLocal = accumLocal.ptr<int>();
        NZPointSet nzLocal(nz.positions.rows, nz.positions.cols);
        int startRow = boundaries.start;
        int endRow = boundaries.end;
        int numCols = edges.cols;

        if(edges.isContinuous() && dx.isContinuous() && dy.isContinuous())
        {
            numCols *= (boundaries.end - boundaries.start);
            endRow = boundaries.start + 1;
        }

        // Accumulate circle evidence for each edge pixel
        for(int y = startRow; y < endRow; ++y )
        {
            const uchar* edgeData = edges.ptr<const uchar>(y);
            const short* dxData = dx.ptr<const short>(y);
            const short* dyData = dy.ptr<const short>(y);
            int x = 0;

            for(; x < numCols; ++x )
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                {
                    v_uint8 v_zero = vx_setzero_u8();

                    for(; x <= numCols - 2*VTraits<v_uint8>::vlanes(); x += 2*VTraits<v_uint8>::vlanes()) {
                        v_uint8 v_edge1 = (v_ne(vx_load(edgeData + x), v_zero));
                        v_uint8 v_edge2 = (v_ne(vx_load(edgeData + x + VTraits<v_uint8>::vlanes()), v_zero));

                        if(v_check_any(v_edge1))
                        {
                            x += v_scan_forward(v_edge1);
                            goto _next_step;
                        }

                        if(v_check_any(v_edge2))
                        {
                            x += VTraits<v_uint8>::vlanes() + v_scan_forward(v_edge2);
                            goto _next_step;
                        }
                    }
                }
#endif
                for(; x < numCols && !edgeData[x]; ++x)
                    ;

                if(x == numCols)
                    continue;
#if (CV_SIMD || CV_SIMD_SCALABLE)
_next_step:
#endif
                float vx, vy;
                int sx, sy, x0, y0, x1, y1;

                vx = dxData[x];
                vy = dyData[x];

                if(vx == 0 && vy == 0)
                    continue;

                float mag = std::sqrt(vx*vx+vy*vy);

                if(mag < 1.0f)
                    continue;

                Point pt = Point(x % edges.cols, y + x / edges.cols);
                nzLocal.insert(pt);

                sx = cvRound((vx * idp) * 1024 / mag);
                sy = cvRound((vy * idp) * 1024 / mag);

                x0 = cvRound((pt.x * idp) * 1024);
                y0 = cvRound((pt.y * idp) * 1024);

                // Step from min_radius to max_radius in both directions of the gradient
                for(int k1 = 0; k1 < 2; k1++ )
                {
                    x1 = x0 + minRadius * sx;
                    y1 = y0 + minRadius * sy;

                    for(int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, r++ )
                    {
                        int x2 = x1 >> 10, y2 = y1 >> 10;
                        if( (unsigned)x2 >= (unsigned)acols ||
                            (unsigned)y2 >= (unsigned)arows )
                            break;

                        adataLocal[y2*astep + x2]++;
                    }

                    sx = -sx; sy = -sy;
                }
            }
        }

        { // TODO Try using TLSContainers
            AutoLock lock(mutex);
            accumVec.push_back(accumLocal);
            nz.insert(nzLocal);
        }
    }

private:
    const Mat &edges, &dx, &dy;
    int minRadius, maxRadius;
    float idp;
    std::vector<Mat>& accumVec;
    NZPointSet& nz;

    int acols, arows, astep;

    Mutex& mutex;
};

class HoughCirclesFindCentersInvoker : public ParallelLoopBody
{
public:
    HoughCirclesFindCentersInvoker(const Mat &_accum, std::vector<int> &_centers, int _accThreshold, Mutex& _mutex) :
        accum(_accum), centers(_centers), accThreshold(_accThreshold), _lock(_mutex)
    {
        acols = accum.cols;
        arows = accum.rows;
        adata = accum.ptr<int>();
    }

    ~HoughCirclesFindCentersInvoker() {}

    void operator()(const Range &boundaries) const CV_OVERRIDE
    {
        int startRow = boundaries.start;
        int endRow = boundaries.end;
        std::vector<int> centersLocal;
        bool singleThread = (boundaries == Range(1, accum.rows - 1));

        startRow = max(1, startRow);
        endRow = min(arows - 1, endRow);

        //Find possible circle centers
        for(int y = startRow; y < endRow; ++y )
        {
            int x = 1;
            int base = y * acols + x;

            for(; x < acols - 1; ++x, ++base )
            {
                if( adata[base] > accThreshold &&
                    adata[base] > adata[base-1] && adata[base] >= adata[base+1] &&
                    adata[base] > adata[base-acols] && adata[base] >= adata[base+acols] )
                    centersLocal.push_back(base);
            }
        }

        if(!centersLocal.empty())
        {
            if(singleThread)
                centers = centersLocal;
            else
            {
                AutoLock alock(_lock);
                centers.insert(centers.end(), centersLocal.begin(), centersLocal.end());
            }
        }
    }

private:
    const Mat &accum;
    std::vector<int> &centers;
    int accThreshold;

    int acols, arows;
    const int *adata;
    Mutex& _lock;
};

template<typename T>
static bool CheckDistance(const std::vector<T> &circles, size_t endIdx, const T& circle, float minDist2)
{
    bool goodPoint = true;
    for (uint j = 0; j < endIdx; ++j)
    {
        T pt = circles[j];
        float distX = circle[0] - pt[0], distY = circle[1] - pt[1];
        if (distX * distX + distY * distY < minDist2)
        {
            goodPoint = false;
            break;
        }
    }
    return goodPoint;
}

static void GetCircleCenters(const std::vector<int> &centers, std::vector<Vec3f> &circles, int acols, float minDist, float dr)
{
    size_t centerCnt = centers.size();
    float minDist2 = minDist * minDist;
    for (size_t i = 0; i < centerCnt; ++i)
    {
        int center = centers[i];
        int y = center / acols;
        int x = center - y * acols;
        Vec3f circle = Vec3f((x + 0.5f) * dr, (y + 0.5f) * dr, 0);

        bool goodPoint = CheckDistance(circles, circles.size(), circle, minDist2);
        if (goodPoint)
            circles.push_back(circle);
    }
}

static void GetCircleCenters(const std::vector<int> &centers, std::vector<Vec4f> &circles, int acols, float minDist, float dr)
{
    size_t centerCnt = centers.size();
    float minDist2 = minDist * minDist;
    for (size_t i = 0; i < centerCnt; ++i)
    {
        int center = centers[i];
        int y = center / acols;
        int x = center - y * acols;
        Vec4f circle = Vec4f((x + 0.5f) * dr, (y + 0.5f) * dr, 0, (float)center);

        bool goodPoint = CheckDistance(circles, circles.size(), circle, minDist2);
        if (goodPoint)
            circles.push_back(circle);
    }
}

template<typename T>
static void RemoveOverlaps(std::vector<T>& circles, float minDist)
{
    if (circles.size() <= 1u)
        return;
    float minDist2 = minDist * minDist;
    size_t endIdx = 1;
    for (size_t i = 1; i < circles.size(); ++i)
    {
        T circle = circles[i];
        if (CheckDistance(circles, endIdx, circle, minDist2))
        {
            circles[endIdx] = circle;
            ++endIdx;
        }
    }
    circles.resize(endIdx);
}

static void CreateCircles(const std::vector<EstimatedCircle>& circlesEst, std::vector<Vec3f>& circles)
{
    std::transform(circlesEst.begin(), circlesEst.end(), std::back_inserter(circles), GetCircle);
}

static void CreateCircles(const std::vector<EstimatedCircle>& circlesEst, std::vector<Vec4f>& circles)
{
    std::transform(circlesEst.begin(), circlesEst.end(), std::back_inserter(circles), GetCircle4f);
}

template<class NZPoints>
class HoughCircleEstimateRadiusInvoker : public ParallelLoopBody
{
public:
    HoughCircleEstimateRadiusInvoker(const NZPoints &_nz, int _nzSz, const std::vector<int> &_centers, std::vector<EstimatedCircle> &_circlesEst,
                                     int _acols, int _accThreshold, int _minRadius, int _maxRadius,
                                     float _dp, Mutex& _mutex) :
        nz(_nz), nzSz(_nzSz), centers(_centers), circlesEst(_circlesEst), acols(_acols), accThreshold(_accThreshold),
        minRadius(_minRadius), maxRadius(_maxRadius), dr(_dp), _lock(_mutex)
    {
        minRadius2 = (float)minRadius * minRadius;
        maxRadius2 = (float)maxRadius * maxRadius;
        centerSz = (int)centers.size();
        CV_Assert(nzSz > 0);
    }

    ~HoughCircleEstimateRadiusInvoker() {}

protected:
    inline int filterCircles(const Point2f& curCenter, float* ddata) const;

    void operator()(const Range &boundaries) const CV_OVERRIDE
    {
        std::vector<EstimatedCircle> circlesLocal;
        const int nBinsPerDr = 10;
        int nBins = cvRound((maxRadius - minRadius)/dr*nBinsPerDr);
        AutoBuffer<int> bins(nBins);
        AutoBuffer<float> distBuf(nzSz), distSqrtBuf(nzSz);
        float *ddata = distBuf.data();
        float *dSqrtData = distSqrtBuf.data();

        bool singleThread = (boundaries == Range(0, centerSz));
        int i = boundaries.start;

        // For each found possible center
        // Estimate radius and check support
        for(; i < boundaries.end; ++i)
        {
            int ofs = centers[i];
            int y = ofs / acols;
            int x = ofs - y * acols;

            //Calculate circle's center in pixels
            Point2f curCenter = Point2f((x + 0.5f) * dr, (y + 0.5f) * dr);
            int nzCount = filterCircles(curCenter, ddata);

            int maxCount = 0;
            float rBest = 0;
            if(nzCount)
            {
                Mat_<float> distMat(1, nzCount, ddata);
                Mat_<float> distSqrtMat(1, nzCount, dSqrtData);
                sqrt(distMat, distSqrtMat);

                memset(bins.data(), 0, sizeof(bins[0])*bins.size());
                for(int k = 0; k < nzCount; k++)
                {
                    int bin = std::max(0, std::min(nBins-1, cvRound((dSqrtData[k] - minRadius)/dr*nBinsPerDr)));
                    bins[bin]++;
                }

                for(int j = nBins - 1; j > 0; j--)
                {
                    if(bins[j])
                    {
                        int upbin = j;
                        int curCount = 0;
                        for(; j > upbin - nBinsPerDr && j >= 0; j--)
                        {
                            curCount += bins[j];
                        }
                        float rCur = (upbin + j)/2.f /nBinsPerDr * dr + minRadius;
                        if((curCount * rBest >= maxCount * rCur) ||
                            (rBest < FLT_EPSILON && curCount >= maxCount))
                        {
                            rBest = rCur;
                            maxCount = curCount;
                        }
                    }
                }
            }

            // Check if the circle has enough support
            if(maxCount > accThreshold)
            {
                circlesLocal.push_back(EstimatedCircle(Vec3f(curCenter.x, curCenter.y, rBest), maxCount));
            }
        }

        if(!circlesLocal.empty())
        {
            std::sort(circlesLocal.begin(), circlesLocal.end(), cmpAccum);
            if(singleThread)
            {
                std::swap(circlesEst, circlesLocal);
            }
            else
            {
                AutoLock alock(_lock);
                if (circlesEst.empty())
                    std::swap(circlesEst, circlesLocal);
                else
                    circlesEst.insert(circlesEst.end(), circlesLocal.begin(), circlesLocal.end());
            }
        }
    }

private:
    const NZPoints &nz;
    int nzSz;
    const std::vector<int> &centers;
    std::vector<EstimatedCircle> &circlesEst;
    int acols, accThreshold, minRadius, maxRadius;
    float dr;
    int centerSz;
    float minRadius2, maxRadius2;
    Mutex& _lock;
};

template<>
inline int HoughCircleEstimateRadiusInvoker<NZPointList>::filterCircles(const Point2f& curCenter, float* ddata) const
{
    int nzCount = 0;
    const Point* nz_ = &nz[0];
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        const v_float32 v_minRadius2 = vx_setall_f32(minRadius2);
        const v_float32 v_maxRadius2 = vx_setall_f32(maxRadius2);

        v_float32 v_curCenterX = vx_setall_f32(curCenter.x);
        v_float32 v_curCenterY = vx_setall_f32(curCenter.y);

        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rbuf[VTraits<v_float32>::max_nlanes];
        int CV_DECL_ALIGNED(CV_SIMD_WIDTH) rmask[VTraits<v_int32>::max_nlanes];
        for(; j <= nzSz - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
        {
            v_float32 v_nzX, v_nzY;
            v_load_deinterleave((const float*)&nz_[j], v_nzX, v_nzY); // FIXIT use proper datatype

            v_float32 v_x = v_cvt_f32(v_reinterpret_as_s32(v_nzX));
            v_float32 v_y = v_cvt_f32(v_reinterpret_as_s32(v_nzY));

            v_float32 v_dx = v_sub(v_x, v_curCenterX);
            v_float32 v_dy = v_sub(v_y, v_curCenterY);

            v_float32 v_r2 = v_add(v_mul(v_dx, v_dx), v_mul(v_dy, v_dy));
            v_float32 vmask = v_and(v_le(v_minRadius2, v_r2), v_le(v_r2, v_maxRadius2));
            if (v_check_any(vmask))
            {
                v_store_aligned(rmask, v_reinterpret_as_s32(vmask));
                v_store_aligned(rbuf, v_r2);
                for (int i = 0; i < VTraits<v_int32>::vlanes(); ++i)
                    if (rmask[i]) ddata[nzCount++] = rbuf[i];
            }
        }
    }
#endif

    // Estimate best radius
    for(; j < nzSz; ++j)
    {
        const Point pt = nz_[j];
        float _dx = curCenter.x - pt.x, _dy = curCenter.y - pt.y;
        float _r2 = _dx * _dx + _dy * _dy;

        if(minRadius2 <= _r2 && _r2 <= maxRadius2)
        {
            ddata[nzCount++] = _r2;
        }
    }
    return nzCount;
}

template<>
inline int HoughCircleEstimateRadiusInvoker<NZPointSet>::filterCircles(const Point2f& curCenter, float* ddata) const
{
    int nzCount = 0;
    const Mat_<uchar>& positions = nz.positions;

    const int rOuter = maxRadius + 1;
    const Range xOuter = Range(std::max(int(curCenter.x - rOuter), 0), std::min(int(curCenter.x + rOuter), positions.cols));
    const Range yOuter = Range(std::max(int(curCenter.y - rOuter), 0), std::min(int(curCenter.y + rOuter), positions.rows));

#if (CV_SIMD || CV_SIMD_SCALABLE)
    float v_seq[VTraits<v_float32>::max_nlanes];
    for (int i = 0; i < VTraits<v_float32>::vlanes(); ++i)
        v_seq[i] = (float)i;
    const v_float32 v_minRadius2 = vx_setall_f32(minRadius2);
    const v_float32 v_maxRadius2 = vx_setall_f32(maxRadius2);
    const v_float32 v_curCenterX_0123 = v_sub(vx_setall_f32(curCenter.x), vx_load(v_seq));
#endif

    for (int y = yOuter.start; y < yOuter.end; y++)
    {
        const uchar* ptr = positions.ptr(y, 0);
        float dy = curCenter.y - y;
        float dy2 = dy * dy;

        int x = xOuter.start;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            const v_float32 v_dy2 = vx_setall_f32(dy2);
            const v_uint32 v_zero_u32 = vx_setall_u32(0);
            float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rbuf[VTraits<v_float32>::max_nlanes];
            int CV_DECL_ALIGNED(CV_SIMD_WIDTH) rmask[VTraits<v_int32>::max_nlanes];
            for (; x <= xOuter.end - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
            {
                v_uint32 v_mask = vx_load_expand_q(ptr + x);
                v_mask = v_ne(v_mask, v_zero_u32);

                v_float32 v_x = v_cvt_f32(vx_setall_s32(x));
                v_float32 v_dx = v_sub(v_x, v_curCenterX_0123);

                v_float32 v_r2 = v_add(v_mul(v_dx, v_dx), v_dy2);
                v_float32 vmask = v_and(v_and(v_le(v_minRadius2, v_r2), v_le(v_r2, v_maxRadius2)), v_reinterpret_as_f32(v_mask));
                if (v_check_any(vmask))
                {
                    v_store_aligned(rmask, v_reinterpret_as_s32(vmask));
                    v_store_aligned(rbuf, v_r2);
                    for (int i = 0; i < VTraits<v_int32>::vlanes(); ++i)
                        if (rmask[i]) ddata[nzCount++] = rbuf[i];
                }
            }
        }
#endif
        for (; x < xOuter.end; x++)
        {
            if (ptr[x])
            {
                float _dx = curCenter.x - x;
                float _r2 = _dx * _dx + dy2;
                if(minRadius2 <= _r2 && _r2 <= maxRadius2)
                {
                    ddata[nzCount++] = _r2;
                }
            }
        }
    }
    return nzCount;
}

template <typename CircleType>
static void HoughCirclesGradient(InputArray _image, OutputArray _circles,
                                 float dp, float minDist,
                                 int minRadius, int maxRadius, int cannyThreshold,
                                 int accThreshold, int maxCircles, int kernelSize, bool centersOnly)
{
    CV_Assert(kernelSize == -1 || kernelSize == 3 || kernelSize == 5 || kernelSize == 7);

    dp = max(dp, 1.f);
    float idp = 1.f/dp;

    Mat edges, dx, dy;

    Sobel(_image, dx, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE);
    Sobel(_image, dy, CV_16S, 0, 1, kernelSize, 1, 0, BORDER_REPLICATE);
    Canny(dx, dy, edges, std::max(1, cannyThreshold / 2), cannyThreshold, false);

    Mutex mtx;
    int numThreads = std::max(1, getNumThreads());
    std::vector<Mat> accumVec;
    NZPointSet nz(_image.rows(), _image.cols());
    parallel_for_(Range(0, edges.rows),
                  HoughCirclesAccumInvoker(edges, dx, dy, minRadius, maxRadius, idp, accumVec, nz, mtx),
                  numThreads);
    int nzSz = cv::countNonZero(nz.positions);
    if(nzSz <= 0)
        return;

    Mat accum = accumVec[0];
    for(size_t i = 1; i < accumVec.size(); i++)
    {
        accum += accumVec[i];
    }
    accumVec.clear();

    std::vector<int> centers;

    // 4 rows when multithreaded because there is a bit overhead
    // and on the other side there are some row ranges where centers are concentrated
    parallel_for_(Range(1, accum.rows - 1),
                  HoughCirclesFindCentersInvoker(accum, centers, accThreshold, mtx),
                  (numThreads > 1) ? ((accum.rows - 2) / 4) : 1);

    int centerCnt = (int)centers.size();
    if(centerCnt == 0)
        return;

    std::sort(centers.begin(), centers.end(), hough_cmp_gt(accum.ptr<int>()));

    std::vector<CircleType> circles;
    circles.reserve(256);
    if (centersOnly)
    {
        // Just get the circle centers
        GetCircleCenters(centers, circles, accum.cols, minDist, dp);
    }
    else
    {
        std::vector<EstimatedCircle> circlesEst;
        if (nzSz < maxRadius * maxRadius)
        {
            // Faster to use a list
            NZPointList nzList(nzSz);
            nz.toList(nzList);
            // One loop iteration per thread if multithreaded.
            parallel_for_(Range(0, centerCnt),
                HoughCircleEstimateRadiusInvoker<NZPointList>(nzList, nzSz, centers, circlesEst, accum.cols,
                    accThreshold, minRadius, maxRadius, dp, mtx),
                numThreads);
        }
        else
        {
            // Faster to use a matrix
            // One loop iteration per thread if multithreaded.
            parallel_for_(Range(0, centerCnt),
                HoughCircleEstimateRadiusInvoker<NZPointSet>(nz, nzSz, centers, circlesEst, accum.cols,
                    accThreshold, minRadius, maxRadius, dp, mtx),
                numThreads);
        }

        // Sort by accumulator value
        std::sort(circlesEst.begin(), circlesEst.end(), cmpAccum);

        // Create Circles
        CreateCircles(circlesEst, circles);
        RemoveOverlaps(circles, minDist);
    }

    if (circles.size() > 0)
    {
        int numCircles = std::min(maxCircles, int(circles.size()));
        Mat(1, numCircles, cv::traits::Type<CircleType>::value, &circles[0]).copyTo(_circles);
        return;
    }
}

static int circle_popcnt(uint64 val)
{
    #ifdef CV_POPCNT_U64
        return CV_POPCNT_U64(val);
    #else
        val -= (val >> 1) & 0x5555555555555555ULL;
        val = (val & 0x3333333333333333ULL) + ((val >> 2) & 0x3333333333333333ULL);
        val = (val + (val >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
        return (int)((val * 0x0101010101010101ULL) >> 56);
    #endif
}

// The structure describes the circle "candidate" that is composed by one or more circle-like arcs.
// * rw - the sum of radiuses multiplied by the corresponding arc lengths.
// * weight - the arc length (the number of pixels it contains)
// * mask - bit mask of 64 elements that shows the coverage of the whole 0..360 degrees angular range of the circle.
//     The mask of all 1's means that the whole circle is completely covered. 0's show the uncovered segments.
struct CircleData
{
    CircleData() { rw = 0; weight = 0; mask = 0; }
    double rw;
    int weight;
    uint64 mask;
};

enum
{
    HOUGH_CIRCLES_ALT_BLOCK_SIZE = 10,
    HOUGH_CIRCLES_ALT_MAX_CLUSTERS = 10
};

static void HoughCirclesAlt( const Mat& img, std::vector<EstimatedCircle>& circles, double dp, double rdMinDist,
                             double minRadius, double maxRadius, double cannyThreshold, double minCos2 )
{
    const int MIN_COUNT = 10;
    const int RAY_FP_BITS = 10;
    const int RAY_FP_SCALE = 1 << RAY_FP_BITS;
    const int ACCUM_FP_BITS = 6;
    const int RAY_SHIFT2 = ACCUM_FP_BITS/2;
    const int ACCUM_ALPHA_ONE = 1 << RAY_SHIFT2;
    const int ACCUM_ALPHA_MASK = ACCUM_ALPHA_ONE - 1;
    const int RAY_SHIFT1 = RAY_FP_BITS - RAY_SHIFT2;
    const int RAY_DELTA1 = 1 << (RAY_SHIFT1 - 1);

    const double ARC_DELTA = 80;
    const double ARC_EPS = 0.03;
    const double CIRCLE_AREA_OFFSET = 4000;
    const double ARC2CLUSTER_EPS = 0.06;
    const double CLUSTER_MERGE_EPS = 0.075;
    const double FINAL_MERGE_DIST_EPS = 0.01;
    const double FINAL_MERGE_AREA_EPS = CLUSTER_MERGE_EPS;

    if( maxRadius <= 0 )
        maxRadius = std::min(img.cols, img.rows)*0.5;
    if( minRadius > maxRadius )
        std::swap(minRadius, maxRadius);
    maxRadius = std::min(maxRadius, std::min(img.cols, img.rows)*0.5);
    maxRadius = std::max(maxRadius, 1.);
    minRadius = std::max(minRadius, 1.);
    minRadius = std::min(minRadius, maxRadius);
    cannyThreshold = std::max(cannyThreshold, 1.);
    dp = std::max(dp, 1.);

    Mat Dx, Dy, edges;
    Scharr(img, Dx, CV_16S, 1, 0);
    Scharr(img, Dy, CV_16S, 0, 1);
    Canny(Dx, Dy, edges, cannyThreshold/2, cannyThreshold, true);
    Mat mask(img.rows + 2, img.cols + 2, CV_8U, Scalar::all(0));
    double idp = 1./dp;
    int minR = cvFloor(minRadius*idp);
    int maxR = cvCeil(maxRadius*idp);
    int acols = cvRound(img.cols*idp);
    int arows = cvRound(img.rows*idp);
    Mat accum(arows + 1, acols + 1, CV_32S, Scalar::all(0));
    int* adata = accum.ptr<int>();
    int astep = (int)accum.step1();
    minR = std::max(minR, 1);
    maxR = std::max(maxR, 1);

    const uchar* edgeData = edges.ptr<uchar>();
    int estep = (int)edges.step1();
    const short* dxData = Dx.ptr<short>();
    const short* dyData = Dy.ptr<short>();
    int dxystep = (int)Dx.step1();
    uchar* mdata = mask.ptr<uchar>();
    int mstep = (int)mask.step1();

    circles.clear();
    std::vector<Vec4f> nz;

    std::vector<Point> stack;
    const int n33[][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};

    for( int x = 0; x < mask.cols; x++ ) mdata[x] = mdata[(mask.rows-1)*mstep + x] = (uchar)1;
    for( int y = 0; y < mask.rows; y++ ) mdata[y*mstep] = mdata[y*mstep + mask.cols-1] = (uchar)1;
    mdata += mstep + 1;

    for( int y = 0; y < edges.rows; y++ )
    {
        for( int x = 0; x < edges.cols; x++ )
        {
            if(!edgeData[y*estep + x] || mdata[y*mstep + x])
                continue;

            mdata[y*mstep + x] = (uchar)1;
            stack.push_back(Point(x, y));
            bool backtrace_mode = false;

            do
            {
                Point p = stack.back();
                stack.pop_back();
                int vx = dxData[p.y*dxystep + p.x];
                int vy = dyData[p.y*dxystep + p.x];

                float mag = std::sqrt((float)vx*vx+(float)vy*vy);
                nz.push_back(Vec4f((float)p.x, (float)p.y, (float)vx, (float)vy));
                CV_Assert(mdata[p.y*mstep + p.x] == 1);

                int sx = cvRound(vx * RAY_FP_SCALE / mag);
                int sy = cvRound(vy * RAY_FP_SCALE / mag);

                int x0 = cvRound((p.x * idp) * RAY_FP_SCALE);
                int y0 = cvRound((p.y * idp) * RAY_FP_SCALE);

                // Step from min_radius to max_radius in both directions of the gradient
                for(int k1 = 0; k1 < 2; k1++ )
                {
                    int x1 = x0 + minR * sx;
                    int y1 = y0 + minR * sy;

                    for(int r = minR; r <= maxR; x1 += sx, y1 += sy, r++ )
                    {
                        int x2a = (x1 + RAY_DELTA1) >> RAY_SHIFT1, y2a = (y1 + RAY_DELTA1) >> RAY_SHIFT1;
                        int x2 = x2a >> RAY_SHIFT2, y2 = y2a >> RAY_SHIFT2;
                        if( (unsigned)x2 >= (unsigned)acols ||
                            (unsigned)y2 >= (unsigned)arows )
                            break;

                        // instead of giving everything to the computed pixel of the accumulator,
                        // do a weighted update of 4 neighbor (2x2) pixels using bilinear interpolation.
                        // we do it to reduce the aliasing effect, even though it's slower
                        int* ptr = adata + y2*astep + x2;
                        int a = (x2a & ACCUM_ALPHA_MASK), b = (y2a & ACCUM_ALPHA_MASK);
                        ptr[0] += (ACCUM_ALPHA_ONE - a)*(ACCUM_ALPHA_ONE - b);
                        ptr[1] += a*(ACCUM_ALPHA_ONE - b);
                        ptr[astep] += (ACCUM_ALPHA_ONE - a)*b;
                        ptr[astep+1] += a*b;
                    }

                    sx = -sx; sy = -sy;
                }

                int neighbors = 0;
                for( int k = 0; k < 8; k++ )
                {
                    int dy = n33[k][0], dx = n33[k][1];
                    int y_ = p.y + dy, x_ = p.x + dx;
                    if( mdata[y_*mstep + x_] || !edgeData[y_*estep + x_])
                        continue;
                    mdata[y_*mstep + x_] = (uchar)1;
                    stack.push_back(Point(x_, y_));
                    neighbors++;
                }

                if( neighbors == 0 )
                {
                    if( backtrace_mode )
                        nz.pop_back();
                    backtrace_mode = true;
                }
                else
                    backtrace_mode = false;
            } while(!stack.empty());
            // insert a special "stop marker" in the end of each
            // connected component to make sure we
            // finalize and analyze the arc segment
            nz.push_back(Vec4f(0.f, 0.f, 0.f, 0.f));
        }
    }

    if( nz.empty() )
        return;

    // use dilation with massive ((rdMinDisp/dp)*2+1) x ((rdMinDisp/dp)*2+1) kernel.
    // this trick helps us quickly find the local maxima of accumulator value
    // that are at least within the specified distance from each other.
    Mat accum_f, accum_max;
    accum.convertTo(accum_f, CV_32F);
    int niters = std::max(cvCeil(rdMinDist*idp), 1);
    dilate(accum_f, accum_max, Mat(), Point(-1, -1), niters, BORDER_CONSTANT, Scalar::all(0));
    std::vector<Point2f> centers;

    // find the possible circle centers
    for( int y = 0; y < arows; y++ )
    {
        const float* adataf = accum_f.ptr<float>(y);
        const float* amaxdata = accum_max.ptr<float>(y);
        int left = -1;
        for( int x = 0; x < acols; x++ )
        {
            if(adataf[x] == amaxdata[x] && adataf[x] > adataf[x+astep])
            {
                if(left < 0) left=x;
            }
            else if(left >= 0)
            {
                float cx = (float)((left + x - 1)*dp*0.5f);
                float cy = (float)(y*dp);
                centers.push_back(Point2f(cx, cy));
                left = -1;
            }
        }
    }

    if(centers.empty())
        return;

    float minR2 = (float)(minRadius*minRadius);
    float maxR2 = (float)(maxRadius*maxRadius);
    int nstripes = (int)((centers.size() + HOUGH_CIRCLES_ALT_BLOCK_SIZE-1)/HOUGH_CIRCLES_ALT_BLOCK_SIZE);
    const int nnz = (int)nz.size();
    Mutex cmutex;

    // Check each possible pair (edge_pixel[i], circle_center[j]).
    // For each circle form the clusters to identify possible radius values.
    // Several clusters (up to 10) are maintained to help to filter out false alarms and
    // to support the concentric circle cases.

    // inside parallel for we process the next "HOUGH_CIRCLES_ALT_BLOCK_SIZE" circles
    parallel_for_(Range(0, nstripes), [&](const Range& r)
    {
    CircleData cdata[HOUGH_CIRCLES_ALT_BLOCK_SIZE*HOUGH_CIRCLES_ALT_MAX_CLUSTERS];
    CircleData arc[HOUGH_CIRCLES_ALT_BLOCK_SIZE];
    int prev_idx[HOUGH_CIRCLES_ALT_BLOCK_SIZE];

    std::vector<EstimatedCircle> local_circles;
    for(int j0 = r.start*HOUGH_CIRCLES_ALT_BLOCK_SIZE; j0 < r.end*HOUGH_CIRCLES_ALT_BLOCK_SIZE; j0 += HOUGH_CIRCLES_ALT_BLOCK_SIZE)
    {
        const Vec4f* nzdata = &nz[0];
        const Point2f* cc = &centers[j0];
        int nc = std::min((int)(centers.size() - j0), (int)HOUGH_CIRCLES_ALT_BLOCK_SIZE);
        if(nc <= 0) break;

        // reset the statistics about the clusters
        for( int j = 0; j < nc; j++ )
        {
            for( int k = 0; k < HOUGH_CIRCLES_ALT_BLOCK_SIZE; k++ )
                cdata[j*HOUGH_CIRCLES_ALT_MAX_CLUSTERS + k] = CircleData();
            arc[j] = CircleData();
            arc[j].weight = 1; // avoid division by zero
            prev_idx[j] = -2; // we compare the current index "i" with prev_idx[j]+1
                              // to check whether we are still at the current Canny
                              // connected component. so we initially set it to -2
                              // to make sure that the initial check gives "false".
        }

        for( int i = 0; i < nnz; i++ )
        {
            Vec4f v = nzdata[i];
            float x = v[0], y = v[1], vx = v[2], vy = v[3], mag2 = vx*vx + vy*vy;
            bool stop_marker = x == 0.f && y == 0.f && vx == 0.f && vy == 0.f;

            for( int j = 0; j < nc; j++ )
            {
                float cx = cc[j].x, cy = cc[j].y;
                float dx = x - cx, dy = y - cy;
                float rij2 = dx*dx + dy*dy;
                // check that i-th pixel is within the specified distance range from the center
                if( (rij2 > maxR2 || rij2 < minR2) && i < nnz-1 ) continue;
                float dv = dx*vx + dy*vy;
                // check that the line segment connecting the edge pixel and the center and
                // the gradient at the edge pixel are almost collinear
                if( (double)dv*dv < (double)minCos2*mag2*rij2 && i < nnz-1 ) continue;
                float rij = std::sqrt(rij2);

                CircleData& arc_j = arc[j];
                double r_arc = arc_j.rw/arc_j.weight;
                int di0 = 0;
                int prev = prev_idx[j];
                prev_idx[j] = i;

                // update the arc statistics if it still looks like an arc
                if( std::abs(rij - r_arc) < (r_arc + ARC_DELTA)*ARC_EPS && prev+1 == i && !stop_marker )
                {
                    arc_j.rw += rij;
                    arc_j.weight++;
                    di0 = 1;
                    r_arc = arc_j.rw/arc_j.weight;
                    if( i < nnz -1 )
                        continue;
                }

                // otherwise (or in the very end) store the arc in the cluster collection,
                // if the arc is long enough.
                if( arc_j.weight >= MIN_COUNT && arc_j.weight >= r_arc*0.15 )
                {
                    // before doing it, compute the angular range coverage (the mask).
                    uint64 mval = 0;
                    for( int di = 0; di < arc_j.weight; di++ )
                    {
                        int i1 = prev + di0 - di;
                        Vec4f u = nz[i1];
                        float x1 = u[0], y1 = u[1];
                        float dx1 = x1 - cx, dy1 = y1 - cy;
                        float af = fastAtan2(dy1, dx1)*(64.f/360.f);
                        int a = (cvFloor(af) & 63);
                        int b = (a + 1) & 63;
                        af -= a;
                        // this is another protection from aliasing effects
                        if( af <= 0.25f )
                            mval |= (uint64)1 << a;
                        else if( af > 0.75f )
                            mval |= (uint64)1 << b;
                        else
                            mval |= ((uint64)1 << a) | ((uint64)1 << b);
                    }

                    double min_eps = DBL_MAX;
                    int min_mval = (int)(sizeof(mval)*8+1);
                    int k = 0, best_k = -1, subst_k = -1;
                    CircleData* cdata_j = &cdata[j*HOUGH_CIRCLES_ALT_MAX_CLUSTERS];

                    for( ; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
                    {
                        CircleData& cjk = cdata_j[k];
                        if( cjk.weight == 0 )
                            break;  // it means that there is no more valid clusters
                        double rk = cjk.rw/cjk.weight;
                        // Compute and use the weighted "cluster with arc" area instead of
                        // just cluster area or just arc area or their sum. This is because the cluster can
                        // be small and the arc can be big, or vice versa. Weighted area is more robust.
                        double r2avg = (rk*rk*cjk.weight + r_arc*r_arc*arc_j.weight)/(cjk.weight + arc_j.weight);
                        // It seems to be more robust to compare circle areas (without "pi" scale)
                        // instead of radiuses. When we compare radiuses, when depending on the ALPHA,
                        // different big circles are merged too easily, or different small circles stay different.
                        if( std::abs(rk*rk - r_arc*r_arc) < (r2avg + CIRCLE_AREA_OFFSET)*ARC2CLUSTER_EPS )
                        {
                            double eps = std::abs(rk - r_arc)/rk;
                            if( eps < min_eps )
                            {
                                min_eps = eps;
                                best_k = k;
                            }
                        }
                        else
                        {
                            // Select the cluster with the worst angular coverage.
                            // We use the angular coverage instead of the arc weight
                            // in order to protect real small circles
                            // from "fake" bigger circles with bigger "support".
                            int pcnt = circle_popcnt(cjk.mask);
                            if( pcnt < min_mval )
                            {
                                min_mval = pcnt;
                                subst_k = k;
                            }
                        }
                    }

                    if( best_k >= 0 ) // if found the match, merge the arc into the cluster
                    {
                        CircleData& cjk = cdata_j[best_k];
                        cjk.rw += arc_j.rw;
                        cjk.weight += arc_j.weight;
                        cjk.mask |= mval;
                    }
                    else
                    {
                        if( k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS )
                            subst_k = k; // if we have empty space, just add the new cluster, do not throw anything
                        CircleData& cjk0 = cdata_j[subst_k];

                        // here was the code that attempts to merge the thrown-away cluster with others,
                        // but apparently it does not have any noticeable effect,
                        // so we removed it for the sake of simplicity ...

                        // add the new cluster
                        cjk0.rw = arc_j.rw;
                        cjk0.weight = arc_j.weight;
                        cjk0.mask = mval;
                    }
                }
                // reset the arc statistics.
                arc_j.rw = stop_marker ? 0. : rij;
                arc_j.weight = 1;
                // do not clean arc_j.mval, because we do not alter it.
            }
        }

        // now merge the final clusters for each particular circle center (cx, cy)
        for( int j = 0; j < nc; j++ )
        {
            CircleData* cdata_j = &cdata[j*HOUGH_CIRCLES_ALT_MAX_CLUSTERS];
            float cx = cc[j].x, cy = cc[j].y;

            for( int k = 0; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
            {
                CircleData& cjk = cdata_j[k];
                if( cjk.weight == 0 )
                    continue;

                // Let in only more or less significant clusters.
                // Small clusters more likely correspond to a noise
                // (otherwise they would grew more substantial during the
                // cluster construction phase).
                // Processing those noisy clusters takes time and
                // potentially decreases accuracy of computed radiuses
                // of good clusters.
                double rjk = cjk.rw/cjk.weight;
                if( cjk.weight < rjk || circle_popcnt(cjk.mask) < 15 )
                    cjk.weight = 0;
            }

            // extensive O(nclusters^2) cluster merge algorithm, but since the number
            // of clusters is limited with a modest constant HOUGH_CIRCLES_ALT_MAX_CLUSTERS,
            // it's still O(1) algorithm :)
            for( int k = 0; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
            {
                CircleData& cjk = cdata_j[k];
                if( cjk.weight == 0 )
                    continue;
                double rk = cjk.rw/cjk.weight;

                int l = k+1;
                for( ; l < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; l++ )
                {
                    CircleData& cjl = cdata_j[l];
                    if( l == k || cjl.weight == 0 )
                        continue;
                    double rl = cjl.rw/cjl.weight;
                    // Here we use a simple sum of areas (without "pi" scale) instead of weighted
                    // sum just for simplicity and potentially for better accuracy.
                    if( std::abs(rk*rk - rl*rl) < (rk*rk + rl*rl + CIRCLE_AREA_OFFSET)*CLUSTER_MERGE_EPS)
                    {
                        cjk.rw += cjl.rw;
                        cjk.weight += cjl.weight;
                        cjk.mask |= cjl.mask;
                        rk = cjk.rw/cjk.weight;
                        cjl.weight = 0;
                        l = -1; // try to merge other clusters again with the updated k-th cluster
                    }
                }
            }

            for( int k = 0; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
            {
                CircleData& cjk = cdata_j[k];
                if( cjk.weight == 0 )
                    continue;
                double rk = cjk.rw/cjk.weight;
                uint64 mask_jk = cjk.mask, mask_jk0 = (mask_jk + 1) ^ mask_jk;
                int count = 0, count0 = -1, runlen = 0, max_runlen = 0;
                int prev_bit = 0;
                for( int b = 0; b < 64; b++, mask_jk >>= 1, mask_jk0 >>= 1 )
                {
                    int bit_k = (mask_jk & 1) != 0;
                    count += bit_k;
                    count0 += (mask_jk0 & 1) != 0;
                    if(bit_k == prev_bit) { runlen++; continue; }
                    if(prev_bit == 1)
                        max_runlen = std::max(max_runlen, runlen);
                    runlen = 1;
                    prev_bit = bit_k;
                }
                if( prev_bit == 1)
                    max_runlen = std::max(max_runlen, runlen + (count < 64 ? count0 : 0));

                // Those constants are the results of fine-tuning.
                // Basically, by lowering thresholds more real circles, as well as fake circles, are accepted.
                // By raising the thresholds you get less real circles and less false alarms.
                // A better and more safe way to obtain better detection results is to regulate
                // [minRadius, maxRadius] range and to play with minCos2 parameter.
                // May be some classifier can be trained that takes the weight,
                // circle radius and the bit mask as inputs and produces the verdict.
                bool accepted = (cjk.weight >= rk*3 && count >= 35 && max_runlen >= 20) || count >= 55;
                //if(debug)
                //printf("[%c]. cx=%.1f, cy=%.1f, r=%.1f, weight=%d, count=%d, max_runlen=%d, mask=%016llx\n",
                //       (accepted ? '+' : '-'), cx, cy, rk, cjk.weight, count, max_runlen, cjk.mask);

                if( accepted )
                    local_circles.push_back(EstimatedCircle(Vec3f(cx, cy, (float)rk), cjk.weight));
            }
        }
    }
    if(!local_circles.empty())
    {
        cmutex.lock();
        std::copy(local_circles.begin(), local_circles.end(), std::back_inserter(circles));
        cmutex.unlock();
    }
    });

    // The final circle merge procedure.
    // This is O(ncircles^2) algorithm
    // and it can take a long time in some specific scenarious.
    // But most of the time it's very fast.
    size_t i0 = 0, nc = circles.size();
    for( size_t i = 0; i < nc; i++ )
    {
        if( circles[i].accum == 0 ) continue;
        EstimatedCircle& ci = circles[i0] = circles[i];
        for( size_t j = i+1; j < nc; j++ )
        {
            EstimatedCircle cj = circles[j];
            if( cj.accum == 0 ) continue;
            float dx = ci.c[0] - cj.c[0], dy = ci.c[1] - cj.c[1];
            float r2 = dx*dx + dy*dy;
            float rs = ci.c[2] + cj.c[2];
            if( r2 > rs*rs*FINAL_MERGE_DIST_EPS)
                continue;
            if( std::abs(ci.c[2]*ci.c[2] - cj.c[2]*cj.c[2]) <
                (ci.c[2]*ci.c[2] + cj.c[2]*cj.c[2] + CIRCLE_AREA_OFFSET)*FINAL_MERGE_AREA_EPS )
            {
                int wi = ci.accum, wj = cj.accum;
                if( wi < wj ) std::swap(ci, cj);
                circles[j].accum = 0;
            }
        }
        i0++;
    }
    circles.resize(i0);
}

static void HoughCircles( InputArray _image, OutputArray _circles,
                          int method, double dp, double minDist,
                          double param1, double param2,
                          int minRadius, int maxRadius,
                          int maxCircles, double param3 )
{
    CV_INSTRUMENT_REGION();

    int type = CV_32FC3;
    if( _circles.fixedType() )
    {
        type = _circles.type();
        CV_CheckType(type, type == CV_32FC3 || type == CV_32FC4, "Wrong type of output circles");
    }

    CV_Assert(!_image.empty() && _image.type() == CV_8UC1 && (_image.isMat() || _image.isUMat()));

    if( dp <= 0 || minDist <= 0 || param1 <= 0)
        CV_Error( Error::StsOutOfRange, "dp, min_dist and canny_threshold must be all positive numbers" );

    switch( method )
    {
    case HOUGH_GRADIENT:
        {
        int cannyThresh = cvRound(param1), accThresh = cvRound(param2), kernelSize = cvRound(param3);
        minRadius = std::max(0, minRadius);

        if( param2 <= 0 )
            CV_Error( Error::StsOutOfRange, "acc_threshold must be a positive number" );

        if(maxCircles < 0)
            maxCircles = INT_MAX;

        bool centersOnly = (maxRadius < 0);

        if( maxRadius <= 0 )
            maxRadius = std::max( _image.rows(), _image.cols() );
        else if( maxRadius <= minRadius )
            maxRadius = minRadius + 2;

        if (type == CV_32FC3)
            HoughCirclesGradient<Vec3f>(_image, _circles, (float)dp, (float)minDist,
                                        minRadius, maxRadius, cannyThresh,
                                        accThresh, maxCircles, kernelSize, centersOnly);
        else if (type == CV_32FC4)
            HoughCirclesGradient<Vec4f>(_image, _circles, (float)dp, (float)minDist,
                                        minRadius, maxRadius, cannyThresh,
                                        accThresh, maxCircles, kernelSize, centersOnly);
        else
            CV_Error(Error::StsError, "Internal error");
        }
        break;
    case HOUGH_GRADIENT_ALT:
        {
            if( param2 >= 1 )
                CV_Error( Error::StsOutOfRange, "when using HOUGH_GRADIENT_ALT method, param2 parameter must be smaller than 1.0" );

            std::vector<EstimatedCircle> circles;
            Mat image = _image.getMat();
            HoughCirclesAlt(image, circles, dp, minDist, minRadius, maxRadius, param1, param2);
            std::sort(circles.begin(), circles.end(), cmpAccum);
            size_t i, ncircles = circles.size();

            if( type == CV_32FC4 )
            {
                std::vector<Vec4f> cw(ncircles);
                for( i = 0; i < ncircles; i++ )
                    cw[i] = GetCircle4f(circles[i]);
                if (ncircles > 0)
                    Mat(1, (int)ncircles, cv::traits::Type<Vec4f>::value, &cw[0]).copyTo(_circles);
            }
            else if( type == CV_32FC3 )
            {
                std::vector<Vec3f> cwow(ncircles);
                for( i = 0; i < ncircles; i++ )
                    cwow[i] = GetCircle(circles[i]);
                if (ncircles > 0)
                    Mat(1, (int)ncircles, cv::traits::Type<Vec3f>::value, &cwow[0]).copyTo(_circles);
            }
            else
                CV_Error(Error::StsError, "Internal error");
        }
        break;
    default:
        CV_Error( Error::StsBadArg, "Unrecognized method id. Actually supported methods are HOUGH_GRADIENT and HOUGH_GRADIENT_ALT" );
    }
}

void HoughCircles( InputArray _image, OutputArray _circles,
                   int method, double dp, double minDist,
                   double param1, double param2,
                   int minRadius, int maxRadius )
{
    HoughCircles(_image, _circles, method, dp, minDist, param1, param2, minRadius, maxRadius, -1, 3);
}
} // \namespace cv
