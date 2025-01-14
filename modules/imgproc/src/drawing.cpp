/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
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
using namespace cv;

namespace cv
{

enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1<<12) - 256 };

static const int MAX_THICKNESS = 32767;

struct PolyEdge
{
    PolyEdge() : y0(0), y1(0), x(0), dx(0), next(0) {}
    //PolyEdge(int _y0, int _y1, int _x, int _dx) : y0(_y0), y1(_y1), x(_x), dx(_dx) {}

    int y0, y1;
    int64 x, dx;
    PolyEdge *next;
};

static void
CollectPolyEdges( Mat& img, const Point2l* v, int npts,
                  std::vector<PolyEdge>& edges, const void* color, int line_type,
                  int shift, Point offset=Point() );

static void
FillEdgeCollection( Mat& img, std::vector<PolyEdge>& edges, const void* color );

static void
PolyLine( Mat& img, const Point2l* v, int npts, bool closed,
          const void* color, int thickness, int line_type, int shift );

static void
FillConvexPoly( Mat& img, const Point2l* v, int npts,
                const void* color, int line_type, int shift );

/****************************************************************************************\
*                                   Lines                                                *
\****************************************************************************************/

bool clipLine( Size img_size, Point& pt1, Point& pt2 )
{
    Point2l p1(pt1);
    Point2l p2(pt2);
    bool inside = clipLine(Size2l(img_size.width, img_size.height), p1, p2);
    pt1.x = (int)p1.x;
    pt1.y = (int)p1.y;
    pt2.x = (int)p2.x;
    pt2.y = (int)p2.y;
    return inside;
}

bool clipLine( Size2l img_size, Point2l& pt1, Point2l& pt2 )
{
    CV_INSTRUMENT_REGION();

    int c1, c2;
    int64 right = img_size.width-1, bottom = img_size.height-1;

    if( img_size.width <= 0 || img_size.height <= 0 )
        return false;

    int64 &x1 = pt1.x, &y1 = pt1.y, &x2 = pt2.x, &y2 = pt2.y;
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;

    if( (c1 & c2) == 0 && (c1 | c2) != 0 )
    {
        int64 a;
        if( c1 & 12 )
        {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int64)((double)(a - y1) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if( c2 & 12 )
        {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int64)((double)(a - y2) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if( (c1 & c2) == 0 && (c1 | c2) != 0 )
        {
            if( c1 )
            {
                a = c1 == 1 ? 0 : right;
                y1 += (int64)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if( c2 )
            {
                a = c2 == 1 ? 0 : right;
                y2 += (int64)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }

        CV_Assert( (c1 & c2) != 0 || (x1 | y1 | x2 | y2) >= 0 );
    }

    return (c1 | c2) == 0;
}

bool clipLine( Rect img_rect, Point& pt1, Point& pt2 )
{
    CV_INSTRUMENT_REGION();

    Point tl = img_rect.tl();
    pt1 -= tl; pt2 -= tl;
    bool inside = clipLine(img_rect.size(), pt1, pt2);
    pt1 += tl; pt2 += tl;

    return inside;
}

void LineIterator::init( const Mat* img, Rect rect, Point pt1_, Point pt2_, int connectivity, bool leftToRight )
{
    CV_Assert( connectivity == 8 || connectivity == 4 );

    count = -1;
    p = Point(0, 0);
    ptr0 = ptr = 0;
    step = elemSize = 0;
    ptmode = !img;

    Point pt1 = pt1_ - rect.tl();
    Point pt2 = pt2_ - rect.tl();

    if( (unsigned)pt1.x >= (unsigned)(rect.width) ||
        (unsigned)pt2.x >= (unsigned)(rect.width) ||
        (unsigned)pt1.y >= (unsigned)(rect.height) ||
        (unsigned)pt2.y >= (unsigned)(rect.height) )
    {
        if( !clipLine(Size(rect.width, rect.height), pt1, pt2) )
        {
            err = plusDelta = minusDelta = plusStep = minusStep = plusShift = minusShift = count = 0;
            return;
        }
    }

    pt1 += rect.tl();
    pt2 += rect.tl();

    int delta_x = 1, delta_y = 1;
    int dx = pt2.x - pt1.x;
    int dy = pt2.y - pt1.y;

    if( dx < 0 )
    {
        if( leftToRight )
        {
            dx = -dx;
            dy = -dy;
            pt1 = pt2;
        }
        else
        {
            dx = -dx;
            delta_x = -1;
        }
    }

    if( dy < 0 )
    {
        dy = -dy;
        delta_y = -1;
    }

    bool vert = dy > dx;
    if( vert )
    {
        std::swap(dx, dy);
        std::swap(delta_x, delta_y);
    }

    CV_Assert( dx >= 0 && dy >= 0 );

    if( connectivity == 8 )
    {
        err = dx - (dy + dy);
        plusDelta = dx + dx;
        minusDelta = -(dy + dy);
        minusShift = delta_x;
        plusShift = 0;
        minusStep = 0;
        plusStep = delta_y;
        count = dx + 1;
    }
    else /* connectivity == 4 */
    {
        err = 0;
        plusDelta = (dx + dx) + (dy + dy);
        minusDelta = -(dy + dy);
        minusShift = delta_x;
        plusShift = -delta_x;
        minusStep = 0;
        plusStep = delta_y;
        count = dx + dy + 1;
    }

    if( vert )
    {
        std::swap(plusStep, plusShift);
        std::swap(minusStep, minusShift);
    }

    p = pt1;
    if( !ptmode )
    {
        ptr0 = img->ptr();
        step = (int)img->step;
        elemSize = (int)img->elemSize();
        ptr = (uchar*)ptr0 + (size_t)p.y*step + (size_t)p.x*elemSize;
        plusStep = plusStep*step + plusShift*elemSize;
        minusStep = minusStep*step + minusShift*elemSize;
    }
}

static void
Line( Mat& img, Point pt1, Point pt2,
      const void* _color, int connectivity = 8 )
{
    if( connectivity == 0 )
        connectivity = 8;
    else if( connectivity == 1 )
        connectivity = 4;

    LineIterator iterator(img, pt1, pt2, connectivity, true);
    int i, count = iterator.count;
    int pix_size = (int)img.elemSize();
    const uchar* color = (const uchar*)_color;

    if( pix_size == 3 )
    {
        for( i = 0; i < count; i++, ++iterator )
        {
            uchar* ptr = *iterator;
            ptr[0] = color[0];
            ptr[1] = color[1];
            ptr[2] = color[2];
        }
    }
    else
    {
        for( i = 0; i < count; i++, ++iterator )
        {
            uchar* ptr = *iterator;
            if( pix_size == 1 )
                ptr[0] = color[0];
            else
                memcpy( *iterator, color, pix_size );
        }
    }
}


/* Correction table depent on the slope */
static const uchar SlopeCorrTable[] = {
    181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201,
    203, 206, 209, 211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254
};

/* Gaussian for antialiasing filter */
static const int FilterTable[] = {
    168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254,
    254, 254, 252, 249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168,
    158, 149, 140, 131, 122, 114, 105, 97, 89, 82, 75, 68, 62, 56, 50, 45,
    40, 36, 32, 28, 25, 22, 19, 16, 14, 12, 11, 9, 8, 7, 5, 5
};

static void
LineAA( Mat& img, Point2l pt1, Point2l pt2, const void* color )
{
    int64 dx, dy;
    int ecount, scount = 0;
    int slope;
    int64 ax, ay;
    int64 x_step, y_step;
    int64 i, j;
    int ep_table[9];
    int cb = ((uchar*)color)[0], cg = ((uchar*)color)[1], cr = ((uchar*)color)[2], ca = ((uchar*)color)[3];
    int _cb, _cg, _cr, _ca;
    int nch = img.channels();
    uchar* ptr = img.ptr();
    size_t step = img.step;
    Size2l size0(img.size()), size = size0;

    if( !((nch == 1 || nch == 3 || nch == 4) && img.depth() == CV_8U) )
    {
        Line(img, Point((int)(pt1.x>>XY_SHIFT), (int)(pt1.y>>XY_SHIFT)), Point((int)(pt2.x>>XY_SHIFT), (int)(pt2.y>>XY_SHIFT)), color);
        return;
    }

    size.width <<= XY_SHIFT;
    size.height <<= XY_SHIFT;
    if( !clipLine( size, pt1, pt2 ))
        return;

    dx = pt2.x - pt1.x;
    dy = pt2.y - pt1.y;

    j = dx < 0 ? -1 : 0;
    ax = (dx ^ j) - j;
    i = dy < 0 ? -1 : 0;
    ay = (dy ^ i) - i;

    if( ax > ay )
    {
        dy = (dy ^ j) - j;
        pt1.x ^= pt2.x & j;
        pt2.x ^= pt1.x & j;
        pt1.x ^= pt2.x & j;
        pt1.y ^= pt2.y & j;
        pt2.y ^= pt1.y & j;
        pt1.y ^= pt2.y & j;

        x_step = XY_ONE;
        y_step = (int64)((uint64_t)dy << XY_SHIFT) / (ax | 1);
        pt2.x += XY_ONE;
        ecount = (int)((pt2.x >> XY_SHIFT) - (pt1.x >> XY_SHIFT));
        j = -(pt1.x & (XY_ONE - 1));
        pt1.y += ((y_step * j) >> XY_SHIFT) + (XY_ONE >> 1);
        slope = (y_step >> (XY_SHIFT - 5)) & 0x3f;
        slope ^= (y_step < 0 ? 0x3f : 0);

        /* Get 4-bit fractions for end-point adjustments */
        i = (pt1.x >> (XY_SHIFT - 7)) & 0x78;
        j = (pt2.x >> (XY_SHIFT - 7)) & 0x78;
    }
    else
    {
        dx = (dx ^ i) - i;
        pt1.x ^= pt2.x & i;
        pt2.x ^= pt1.x & i;
        pt1.x ^= pt2.x & i;
        pt1.y ^= pt2.y & i;
        pt2.y ^= pt1.y & i;
        pt1.y ^= pt2.y & i;

        x_step = (int64)((uint64_t)dx << XY_SHIFT) / (ay | 1);
        y_step = XY_ONE;
        pt2.y += XY_ONE;
        ecount = (int)((pt2.y >> XY_SHIFT) - (pt1.y >> XY_SHIFT));
        j = -(pt1.y & (XY_ONE - 1));
        pt1.x += ((x_step * j) >> XY_SHIFT) + (XY_ONE >> 1);
        slope = (x_step >> (XY_SHIFT - 5)) & 0x3f;
        slope ^= (x_step < 0 ? 0x3f : 0);

        /* Get 4-bit fractions for end-point adjustments */
        i = (pt1.y >> (XY_SHIFT - 7)) & 0x78;
        j = (pt2.y >> (XY_SHIFT - 7)) & 0x78;
    }

    slope = (slope & 0x20) ? 0x100 : SlopeCorrTable[slope];

    /* Calc end point correction table */
    {
        int t0 = slope << 7;
        int t1 = ((0x78 - (int)i) | 4) * slope;
        int t2 = ((int)j | 4) * slope;

        ep_table[0] = 0;
        ep_table[8] = slope;
        ep_table[1] = ep_table[3] = ((((j - i) & 0x78) | 4) * slope >> 8) & 0x1ff;
        ep_table[2] = (t1 >> 8) & 0x1ff;
        ep_table[4] = ((((j - i) + 0x80) | 4) * slope >> 8) & 0x1ff;
        ep_table[5] = ((t1 + t0) >> 8) & 0x1ff;
        ep_table[6] = (t2 >> 8) & 0x1ff;
        ep_table[7] = ((t2 + t0) >> 8) & 0x1ff;
    }

    if( nch == 3 )
    {
        #define  ICV_PUT_POINT(x, y)        \
        {                                   \
            uchar* tptr = ptr + (x)*3 + (y)*step; \
            _cb = tptr[0];                  \
            _cb += ((cb - _cb)*a + 127)>> 8;\
            _cb += ((cb - _cb)*a + 127)>> 8;\
            _cg = tptr[1];                  \
            _cg += ((cg - _cg)*a + 127)>> 8;\
            _cg += ((cg - _cg)*a + 127)>> 8;\
            _cr = tptr[2];                  \
            _cr += ((cr - _cr)*a + 127)>> 8;\
            _cr += ((cr - _cr)*a + 127)>> 8;\
            tptr[0] = (uchar)_cb;           \
            tptr[1] = (uchar)_cg;           \
            tptr[2] = (uchar)_cr;           \
        }
        if( ax > ay )
        {
            int x = (int)(pt1.x >> XY_SHIFT);

            for( ; ecount >= 0; x++, pt1.y += y_step, scount++, ecount-- )
            {
                if( (unsigned)x >= (unsigned)size0.width )
                    continue;
                int y = (int)((pt1.y >> XY_SHIFT) - 1);

                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.y >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * FilterTable[dist + 32] >> 8) & 0xff;
                if( (unsigned)y < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y)

                a = (ep_corr * FilterTable[dist] >> 8) & 0xff;
                if( (unsigned)(y+1) < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y+1)

                a = (ep_corr * FilterTable[63 - dist] >> 8) & 0xff;
                if( (unsigned)(y+2) < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y+2)
            }
        }
        else
        {
            int y = (int)(pt1.y >> XY_SHIFT);

            for( ; ecount >= 0; y++, pt1.x += x_step, scount++, ecount-- )
            {
                if( (unsigned)y >= (unsigned)size0.height )
                    continue;
                int x = (int)((pt1.x >> XY_SHIFT) - 1);
                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.x >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * FilterTable[dist + 32] >> 8) & 0xff;
                if( (unsigned)x < (unsigned)size0.width )
                    ICV_PUT_POINT(x, y)

                a = (ep_corr * FilterTable[dist] >> 8) & 0xff;
                if( (unsigned)(x+1) < (unsigned)size0.width )
                    ICV_PUT_POINT(x+1, y)

                a = (ep_corr * FilterTable[63 - dist] >> 8) & 0xff;
                if( (unsigned)(x+2) < (unsigned)size0.width )
                    ICV_PUT_POINT(x+2, y)
            }
        }
        #undef ICV_PUT_POINT
    }
    else if(nch == 1)
    {
        #define ICV_PUT_POINT(x, y)         \
        {                                   \
            uchar* tptr = ptr + (x) + (y) * step; \
            _cb = tptr[0];                  \
            _cb += ((cb - _cb)*a + 127)>> 8;\
            _cb += ((cb - _cb)*a + 127)>> 8;\
            tptr[0] = (uchar)_cb;           \
        }

        if( ax > ay )
        {
            int x = (int)(pt1.x >> XY_SHIFT);

            for( ; ecount >= 0; x++, pt1.y += y_step, scount++, ecount-- )
            {
                if( (unsigned)x >= (unsigned)size0.width )
                    continue;
                int y = (int)((pt1.y >> XY_SHIFT) - 1);

                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.y >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * FilterTable[dist + 32] >> 8) & 0xff;
                if( (unsigned)y < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y)

                a = (ep_corr * FilterTable[dist] >> 8) & 0xff;
                if( (unsigned)(y+1) < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y+1)

                a = (ep_corr * FilterTable[63 - dist] >> 8) & 0xff;
                if( (unsigned)(y+2) < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y+2)
            }
        }
        else
        {
            int y = (int)(pt1.y >> XY_SHIFT);

            for( ; ecount >= 0; y++, pt1.x += x_step, scount++, ecount-- )
            {
                if( (unsigned)y >= (unsigned)size0.height )
                    continue;
                int x = (int)((pt1.x >> XY_SHIFT) - 1);
                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.x >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * FilterTable[dist + 32] >> 8) & 0xff;
                if( (unsigned)x < (unsigned)size0.width )
                    ICV_PUT_POINT(x, y)

                a = (ep_corr * FilterTable[dist] >> 8) & 0xff;
                if( (unsigned)(x+1) < (unsigned)size0.width )
                    ICV_PUT_POINT(x+1, y)

                a = (ep_corr * FilterTable[63 - dist] >> 8) & 0xff;
                if( (unsigned)(x+2) < (unsigned)size0.width )
                    ICV_PUT_POINT(x+2, y)
            }
        }
        #undef ICV_PUT_POINT
    }
    else
    {
        #define  ICV_PUT_POINT(x, y)        \
        {                                   \
            uchar* tptr = ptr + (x)*4 + (y)*step; \
            _cb = tptr[0];                  \
            _cb += ((cb - _cb)*a + 127)>> 8;\
            _cb += ((cb - _cb)*a + 127)>> 8;\
            _cg = tptr[1];                  \
            _cg += ((cg - _cg)*a + 127)>> 8;\
            _cg += ((cg - _cg)*a + 127)>> 8;\
            _cr = tptr[2];                  \
            _cr += ((cr - _cr)*a + 127)>> 8;\
            _cr += ((cr - _cr)*a + 127)>> 8;\
            _ca = tptr[3];                  \
            _ca += ((ca - _ca)*a + 127)>> 8;\
            _ca += ((ca - _ca)*a + 127)>> 8;\
            tptr[0] = (uchar)_cb;           \
            tptr[1] = (uchar)_cg;           \
            tptr[2] = (uchar)_cr;           \
            tptr[3] = (uchar)_ca;           \
        }
        if( ax > ay )
        {
            int x = (int)(pt1.x >> XY_SHIFT);

            for( ; ecount >= 0; x++, pt1.y += y_step, scount++, ecount-- )
            {
                if( (unsigned)x >= (unsigned)size0.width )
                    continue;
                int y = (int)((pt1.y >> XY_SHIFT) - 1);

                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.y >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * FilterTable[dist + 32] >> 8) & 0xff;
                if( (unsigned)y < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y)

                a = (ep_corr * FilterTable[dist] >> 8) & 0xff;
                if( (unsigned)(y+1) < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y+1)

                a = (ep_corr * FilterTable[63 - dist] >> 8) & 0xff;
                if( (unsigned)(y+2) < (unsigned)size0.height )
                    ICV_PUT_POINT(x, y+2)
            }
        }
        else
        {
            int y = (int)(pt1.y >> XY_SHIFT);

            for( ; ecount >= 0; y++, pt1.x += x_step, scount++, ecount-- )
            {
                if( (unsigned)y >= (unsigned)size0.height )
                    continue;
                int x = (int)((pt1.x >> XY_SHIFT) - 1);
                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.x >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * FilterTable[dist + 32] >> 8) & 0xff;
                if( (unsigned)x < (unsigned)size0.width )
                    ICV_PUT_POINT(x, y)

                a = (ep_corr * FilterTable[dist] >> 8) & 0xff;
                if( (unsigned)(x+1) < (unsigned)size0.width )
                    ICV_PUT_POINT(x+1, y)

                a = (ep_corr * FilterTable[63 - dist] >> 8) & 0xff;
                if( (unsigned)(x+2) < (unsigned)size0.width )
                    ICV_PUT_POINT(x+2, y)
            }
        }
        #undef ICV_PUT_POINT
    }
}


static void
Line2( Mat& img, Point2l pt1, Point2l pt2, const void* color)
{
    int64 dx, dy;
    int ecount;
    int64 ax, ay;
    int64 i, j;
    int x, y;
    int64 x_step, y_step;
    int cb = ((uchar*)color)[0];
    int cg = ((uchar*)color)[1];
    int cr = ((uchar*)color)[2];
    int pix_size = (int)img.elemSize();
    uchar *ptr = img.ptr(), *tptr;
    size_t step = img.step;
    Size size = img.size();

    //CV_Assert( img && (nch == 1 || nch == 3) && img.depth() == CV_8U );

    Size2l sizeScaled(((int64)size.width) << XY_SHIFT, ((int64)size.height) << XY_SHIFT);
    if( !clipLine( sizeScaled, pt1, pt2 ))
        return;

    dx = pt2.x - pt1.x;
    dy = pt2.y - pt1.y;

    j = dx < 0 ? -1 : 0;
    ax = (dx ^ j) - j;
    i = dy < 0 ? -1 : 0;
    ay = (dy ^ i) - i;

    if( ax > ay )
    {
        dy = (dy ^ j) - j;
        pt1.x ^= pt2.x & j;
        pt2.x ^= pt1.x & j;
        pt1.x ^= pt2.x & j;
        pt1.y ^= pt2.y & j;
        pt2.y ^= pt1.y & j;
        pt1.y ^= pt2.y & j;

        x_step = XY_ONE;
        y_step = dy * (1 << XY_SHIFT) / (ax | 1);
        ecount = (int)((pt2.x - pt1.x) >> XY_SHIFT);
    }
    else
    {
        dx = (dx ^ i) - i;
        pt1.x ^= pt2.x & i;
        pt2.x ^= pt1.x & i;
        pt1.x ^= pt2.x & i;
        pt1.y ^= pt2.y & i;
        pt2.y ^= pt1.y & i;
        pt1.y ^= pt2.y & i;

        x_step = dx * (1 << XY_SHIFT) / (ay | 1);
        y_step = XY_ONE;
        ecount = (int)((pt2.y - pt1.y) >> XY_SHIFT);
    }

    pt1.x += (XY_ONE >> 1);
    pt1.y += (XY_ONE >> 1);

    if( pix_size == 3 )
    {
        #define  ICV_PUT_POINT(_x,_y)   \
        x = (_x); y = (_y);             \
        if( 0 <= x && x < size.width && \
            0 <= y && y < size.height ) \
        {                               \
            tptr = ptr + y*step + x*3;  \
            tptr[0] = (uchar)cb;        \
            tptr[1] = (uchar)cg;        \
            tptr[2] = (uchar)cr;        \
        }

        ICV_PUT_POINT((int)((pt2.x + (XY_ONE >> 1)) >> XY_SHIFT),
                      (int)((pt2.y + (XY_ONE >> 1)) >> XY_SHIFT));

        if( ax > ay )
        {
            pt1.x >>= XY_SHIFT;

            while( ecount >= 0 )
            {
                ICV_PUT_POINT((int)(pt1.x), (int)(pt1.y >> XY_SHIFT));
                pt1.x++;
                pt1.y += y_step;
                ecount--;
            }
        }
        else
        {
            pt1.y >>= XY_SHIFT;

            while( ecount >= 0 )
            {
                ICV_PUT_POINT((int)(pt1.x >> XY_SHIFT), (int)(pt1.y));
                pt1.x += x_step;
                pt1.y++;
                ecount--;
            }
        }

        #undef ICV_PUT_POINT
    }
    else if( pix_size == 1 )
    {
        #define  ICV_PUT_POINT(_x,_y) \
        x = (_x); y = (_y);           \
        if( 0 <= x && x < size.width && \
            0 <= y && y < size.height ) \
        {                           \
            tptr = ptr + y*step + x;\
            tptr[0] = (uchar)cb;    \
        }

        ICV_PUT_POINT((int)((pt2.x + (XY_ONE >> 1)) >> XY_SHIFT),
                      (int)((pt2.y + (XY_ONE >> 1)) >> XY_SHIFT));

        if( ax > ay )
        {
            pt1.x >>= XY_SHIFT;

            while( ecount >= 0 )
            {
                ICV_PUT_POINT((int)(pt1.x), (int)(pt1.y >> XY_SHIFT));
                pt1.x++;
                pt1.y += y_step;
                ecount--;
            }
        }
        else
        {
            pt1.y >>= XY_SHIFT;

            while( ecount >= 0 )
            {
                ICV_PUT_POINT((int)(pt1.x >> XY_SHIFT), (int)(pt1.y));
                pt1.x += x_step;
                pt1.y++;
                ecount--;
            }
        }

        #undef ICV_PUT_POINT
    }
    else
    {
        #define  ICV_PUT_POINT(_x,_y)   \
        x = (_x); y = (_y);             \
        if( 0 <= x && x < size.width && \
            0 <= y && y < size.height ) \
        {                               \
            tptr = ptr + y*step + x*pix_size;\
            for( j = 0; j < pix_size; j++ ) \
                tptr[j] = ((uchar*)color)[j]; \
        }

        ICV_PUT_POINT((int)((pt2.x + (XY_ONE >> 1)) >> XY_SHIFT),
                      (int)((pt2.y + (XY_ONE >> 1)) >> XY_SHIFT));

        if( ax > ay )
        {
            pt1.x >>= XY_SHIFT;

            while( ecount >= 0 )
            {
                ICV_PUT_POINT((int)(pt1.x), (int)(pt1.y >> XY_SHIFT));
                pt1.x++;
                pt1.y += y_step;
                ecount--;
            }
        }
        else
        {
            pt1.y >>= XY_SHIFT;

            while( ecount >= 0 )
            {
                ICV_PUT_POINT((int)(pt1.x >> XY_SHIFT), (int)(pt1.y));
                pt1.x += x_step;
                pt1.y++;
                ecount--;
            }
        }

        #undef ICV_PUT_POINT
    }
}


/****************************************************************************************\
*                   Antialiazed Elliptic Arcs via Antialiazed Lines                      *
\****************************************************************************************/

static const float SinTable[] =
    { 0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f,
    0.1045285f, 0.1218693f, 0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f,
    0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f, 0.2756374f, 0.2923717f,
    0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
    0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f,
    0.5000000f, 0.5150381f, 0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f,
    0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f, 0.6427876f, 0.6560590f,
    0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
    0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f,
    0.8090170f, 0.8191520f, 0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f,
    0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f, 0.8987940f, 0.9063078f,
    0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
    0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f,
    0.9781476f, 0.9816272f, 0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f,
    0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f, 0.9993908f, 0.9998477f,
    1.0000000f, 0.9998477f, 0.9993908f, 0.9986295f, 0.9975641f, 0.9961947f,
    0.9945219f, 0.9925462f, 0.9902681f, 0.9876883f, 0.9848078f, 0.9816272f,
    0.9781476f, 0.9743701f, 0.9702957f, 0.9659258f, 0.9612617f, 0.9563048f,
    0.9510565f, 0.9455186f, 0.9396926f, 0.9335804f, 0.9271839f, 0.9205049f,
    0.9135455f, 0.9063078f, 0.8987940f, 0.8910065f, 0.8829476f, 0.8746197f,
    0.8660254f, 0.8571673f, 0.8480481f, 0.8386706f, 0.8290376f, 0.8191520f,
    0.8090170f, 0.7986355f, 0.7880108f, 0.7771460f, 0.7660444f, 0.7547096f,
    0.7431448f, 0.7313537f, 0.7193398f, 0.7071068f, 0.6946584f, 0.6819984f,
    0.6691306f, 0.6560590f, 0.6427876f, 0.6293204f, 0.6156615f, 0.6018150f,
    0.5877853f, 0.5735764f, 0.5591929f, 0.5446390f, 0.5299193f, 0.5150381f,
    0.5000000f, 0.4848096f, 0.4694716f, 0.4539905f, 0.4383711f, 0.4226183f,
    0.4067366f, 0.3907311f, 0.3746066f, 0.3583679f, 0.3420201f, 0.3255682f,
    0.3090170f, 0.2923717f, 0.2756374f, 0.2588190f, 0.2419219f, 0.2249511f,
    0.2079117f, 0.1908090f, 0.1736482f, 0.1564345f, 0.1391731f, 0.1218693f,
    0.1045285f, 0.0871557f, 0.0697565f, 0.0523360f, 0.0348995f, 0.0174524f,
    0.0000000f, -0.0174524f, -0.0348995f, -0.0523360f, -0.0697565f, -0.0871557f,
    -0.1045285f, -0.1218693f, -0.1391731f, -0.1564345f, -0.1736482f, -0.1908090f,
    -0.2079117f, -0.2249511f, -0.2419219f, -0.2588190f, -0.2756374f, -0.2923717f,
    -0.3090170f, -0.3255682f, -0.3420201f, -0.3583679f, -0.3746066f, -0.3907311f,
    -0.4067366f, -0.4226183f, -0.4383711f, -0.4539905f, -0.4694716f, -0.4848096f,
    -0.5000000f, -0.5150381f, -0.5299193f, -0.5446390f, -0.5591929f, -0.5735764f,
    -0.5877853f, -0.6018150f, -0.6156615f, -0.6293204f, -0.6427876f, -0.6560590f,
    -0.6691306f, -0.6819984f, -0.6946584f, -0.7071068f, -0.7193398f, -0.7313537f,
    -0.7431448f, -0.7547096f, -0.7660444f, -0.7771460f, -0.7880108f, -0.7986355f,
    -0.8090170f, -0.8191520f, -0.8290376f, -0.8386706f, -0.8480481f, -0.8571673f,
    -0.8660254f, -0.8746197f, -0.8829476f, -0.8910065f, -0.8987940f, -0.9063078f,
    -0.9135455f, -0.9205049f, -0.9271839f, -0.9335804f, -0.9396926f, -0.9455186f,
    -0.9510565f, -0.9563048f, -0.9612617f, -0.9659258f, -0.9702957f, -0.9743701f,
    -0.9781476f, -0.9816272f, -0.9848078f, -0.9876883f, -0.9902681f, -0.9925462f,
    -0.9945219f, -0.9961947f, -0.9975641f, -0.9986295f, -0.9993908f, -0.9998477f,
    -1.0000000f, -0.9998477f, -0.9993908f, -0.9986295f, -0.9975641f, -0.9961947f,
    -0.9945219f, -0.9925462f, -0.9902681f, -0.9876883f, -0.9848078f, -0.9816272f,
    -0.9781476f, -0.9743701f, -0.9702957f, -0.9659258f, -0.9612617f, -0.9563048f,
    -0.9510565f, -0.9455186f, -0.9396926f, -0.9335804f, -0.9271839f, -0.9205049f,
    -0.9135455f, -0.9063078f, -0.8987940f, -0.8910065f, -0.8829476f, -0.8746197f,
    -0.8660254f, -0.8571673f, -0.8480481f, -0.8386706f, -0.8290376f, -0.8191520f,
    -0.8090170f, -0.7986355f, -0.7880108f, -0.7771460f, -0.7660444f, -0.7547096f,
    -0.7431448f, -0.7313537f, -0.7193398f, -0.7071068f, -0.6946584f, -0.6819984f,
    -0.6691306f, -0.6560590f, -0.6427876f, -0.6293204f, -0.6156615f, -0.6018150f,
    -0.5877853f, -0.5735764f, -0.5591929f, -0.5446390f, -0.5299193f, -0.5150381f,
    -0.5000000f, -0.4848096f, -0.4694716f, -0.4539905f, -0.4383711f, -0.4226183f,
    -0.4067366f, -0.3907311f, -0.3746066f, -0.3583679f, -0.3420201f, -0.3255682f,
    -0.3090170f, -0.2923717f, -0.2756374f, -0.2588190f, -0.2419219f, -0.2249511f,
    -0.2079117f, -0.1908090f, -0.1736482f, -0.1564345f, -0.1391731f, -0.1218693f,
    -0.1045285f, -0.0871557f, -0.0697565f, -0.0523360f, -0.0348995f, -0.0174524f,
    -0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f,
    0.1045285f, 0.1218693f, 0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f,
    0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f, 0.2756374f, 0.2923717f,
    0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
    0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f,
    0.5000000f, 0.5150381f, 0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f,
    0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f, 0.6427876f, 0.6560590f,
    0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
    0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f,
    0.8090170f, 0.8191520f, 0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f,
    0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f, 0.8987940f, 0.9063078f,
    0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
    0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f,
    0.9781476f, 0.9816272f, 0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f,
    0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f, 0.9993908f, 0.9998477f,
    1.0000000f
};


static void
sincos( int angle, float& cosval, float& sinval )
{
    angle += (angle < 0 ? 360 : 0);
    sinval = SinTable[angle];
    cosval = SinTable[450 - angle];
}

/*
   constructs polygon that represents elliptic arc.
*/
void ellipse2Poly( Point center, Size axes, int angle,
                   int arcStart, int arcEnd,
                   int delta, CV_OUT std::vector<Point>& pts )
{
    std::vector<Point2d> _pts;
    ellipse2Poly(Point2d(center.x, center.y), Size2d(axes.width, axes.height), angle,
                 arcStart, arcEnd, delta, _pts);
    Point prevPt(INT_MIN, INT_MIN);
    pts.resize(0);
    for (unsigned int i = 0; i < _pts.size(); ++i)
    {
        Point pt;
        pt.x = cvRound(_pts[i].x);
        pt.y = cvRound(_pts[i].y);
        if (pt != prevPt) {
            pts.push_back(pt);
            prevPt = pt;
        }
    }

    // If there are no points, it's a zero-size polygon
    CV_Assert( !pts.empty() );
    if (pts.size() == 1) {
        pts.assign(2, center);
    }
}

void ellipse2Poly( Point2d center, Size2d axes, int angle,
                   int arc_start, int arc_end,
                   int delta, std::vector<Point2d>& pts )
{
    CV_INSTRUMENT_REGION();
    CV_Assert(0 < delta && delta <= 180);

    float alpha, beta;
    int i;

    while( angle < 0 )
        angle += 360;
    while( angle > 360 )
        angle -= 360;

    if( arc_start > arc_end )
    {
        i = arc_start;
        arc_start = arc_end;
        arc_end = i;
    }
    while( arc_start < 0 )
    {
        arc_start += 360;
        arc_end += 360;
    }
    while( arc_end > 360 )
    {
        arc_end -= 360;
        arc_start -= 360;
    }
    if( arc_end - arc_start > 360 )
    {
        arc_start = 0;
        arc_end = 360;
    }
    sincos( angle, alpha, beta );
    pts.resize(0);

    for( i = arc_start; i < arc_end + delta; i += delta )
    {
        double x, y;
        angle = i;
        if( angle > arc_end )
            angle = arc_end;
        if( angle < 0 )
            angle += 360;

        x = axes.width * SinTable[450-angle];
        y = axes.height * SinTable[angle];
        Point2d pt;
        pt.x = center.x + x * alpha - y * beta;
        pt.y = center.y + x * beta + y * alpha;
        pts.push_back(pt);
    }

    // If there are no points, it's a zero-size polygon
    CV_Assert( !pts.empty() );
    if( pts.size() == 1) {
        pts.assign(2,center);
    }
}


static void
EllipseEx( Mat& img, Point2l center, Size2l axes,
           int angle, int arc_start, int arc_end,
           const void* color, int thickness, int line_type )
{
    axes.width = std::abs(axes.width), axes.height = std::abs(axes.height);
    int delta = (int)((std::max(axes.width,axes.height)+(XY_ONE>>1))>>XY_SHIFT);
    delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;

    std::vector<Point2d> _v;
    ellipse2Poly( Point2d((double)center.x, (double)center.y), Size2d((double)axes.width, (double)axes.height), angle, arc_start, arc_end, delta, _v );

    std::vector<Point2l> v;
    Point2l prevPt(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    for (unsigned int i = 0; i < _v.size(); ++i)
    {
        Point2l pt;
        pt.x = (int64)cvRound(_v[i].x / static_cast<double>(XY_ONE)) << XY_SHIFT;
        pt.y = (int64)cvRound(_v[i].y / static_cast<double>(XY_ONE)) << XY_SHIFT;
        pt.x += cvRound(_v[i].x - pt.x);
        pt.y += cvRound(_v[i].y - pt.y);
        if (pt != prevPt) {
            v.push_back(pt);
            prevPt = pt;
        }
    }

    // If there are no points, it's a zero-size polygon
    if (v.size() <= 1) {
        v.assign(2, center);
    }

    if( thickness >= 0 )
        PolyLine( img, &v[0], (int)v.size(), false, color, thickness, line_type, XY_SHIFT );
    else if( arc_end - arc_start >= 360 )
        FillConvexPoly( img, &v[0], (int)v.size(), color, line_type, XY_SHIFT );
    else
    {
        v.push_back(center);
        std::vector<PolyEdge> edges;
        CollectPolyEdges( img,  &v[0], (int)v.size(), edges, color, line_type, XY_SHIFT );
        FillEdgeCollection( img, edges, color );
    }
}


/****************************************************************************************\
*                                Polygons filling                                        *
\****************************************************************************************/

static inline void ICV_HLINE_X(uchar* ptr, int64_t xl, int64_t xr, const uchar* color, int pix_size)
{
    uchar* hline_min_ptr = (uchar*)(ptr) + (xl)*(pix_size);
    uchar* hline_end_ptr = (uchar*)(ptr) + (xr+1)*(pix_size);
    uchar* hline_ptr = hline_min_ptr;
    if (pix_size == 1)
      memset(hline_min_ptr, *color, hline_end_ptr-hline_min_ptr);
    else//if (pix_size != 1)
    {
      if (hline_min_ptr < hline_end_ptr)
      {
        memcpy(hline_ptr, color, pix_size);
        hline_ptr += pix_size;
      }//end if (hline_min_ptr < hline_end_ptr)
      size_t sizeToCopy = pix_size;
      while(hline_ptr < hline_end_ptr)
      {
        memcpy(hline_ptr, hline_min_ptr, sizeToCopy);
        hline_ptr += sizeToCopy;
        sizeToCopy = std::min(2*sizeToCopy, static_cast<size_t>(hline_end_ptr-hline_ptr));
      }//end while(hline_ptr < hline_end_ptr)
    }//end if (pix_size != 1)
}
//end ICV_HLINE_X()

static inline void ICV_HLINE(uchar* ptr, int64_t xl, int64_t xr, const void* color, int pix_size)
{
  ICV_HLINE_X(ptr, xl, xr, reinterpret_cast<const uchar*>(color), pix_size);
}
//end ICV_HLINE()

/* filling convex polygon. v - array of vertices, ntps - number of points */
static void
FillConvexPoly( Mat& img, const Point2l* v, int npts, const void* color, int line_type, int shift )
{
    struct
    {
        int idx, di;
        int64 x, dx;
        int ye;
    }
    edge[2];

    int delta = 1 << shift >> 1;
    int i, y, imin = 0;
    int edges = npts;
    int64 xmin, xmax, ymin, ymax;
    uchar* ptr = img.ptr();
    Size size = img.size();
    int pix_size = (int)img.elemSize();
    Point2l p0;
    int delta1, delta2;

    if( line_type < cv::LINE_AA )
        delta1 = delta2 = XY_ONE >> 1;
    else
        delta1 = XY_ONE - 1, delta2 = 0;

    p0 = v[npts - 1];
    p0.x <<= XY_SHIFT - shift;
    p0.y <<= XY_SHIFT - shift;

    CV_Assert( 0 <= shift && shift <= XY_SHIFT );
    xmin = xmax = v[0].x;
    ymin = ymax = v[0].y;

    for( i = 0; i < npts; i++ )
    {
        Point2l p = v[i];
        if( p.y < ymin )
        {
            ymin = p.y;
            imin = i;
        }

        ymax = std::max( ymax, p.y );
        xmax = std::max( xmax, p.x );
        xmin = MIN( xmin, p.x );

        p.x <<= XY_SHIFT - shift;
        p.y <<= XY_SHIFT - shift;

        if( line_type <= 8 )
        {
            if( shift == 0 )
            {
                Point pt0, pt1;
                pt0.x = (int)(p0.x >> XY_SHIFT);
                pt0.y = (int)(p0.y >> XY_SHIFT);
                pt1.x = (int)(p.x >> XY_SHIFT);
                pt1.y = (int)(p.y >> XY_SHIFT);
                Line( img, pt0, pt1, color, line_type );
            }
            else
                Line2( img, p0, p, color );
        }
        else
            LineAA( img, p0, p, color );
        p0 = p;
    }

    xmin = (xmin + delta) >> shift;
    xmax = (xmax + delta) >> shift;
    ymin = (ymin + delta) >> shift;
    ymax = (ymax + delta) >> shift;

    if( npts < 3 || (int)xmax < 0 || (int)ymax < 0 || (int)xmin >= size.width || (int)ymin >= size.height )
        return;

    ymax = MIN( ymax, size.height - 1 );
    edge[0].idx = edge[1].idx = imin;

    edge[0].ye = edge[1].ye = y = (int)ymin;
    edge[0].di = 1;
    edge[1].di = npts - 1;

    edge[0].x = edge[1].x = -XY_ONE;
    edge[0].dx = edge[1].dx = 0;

    ptr += (int64_t)img.step*y;

    do
    {
        if( line_type < cv::LINE_AA || y < (int)ymax || y == (int)ymin )
        {
            for( i = 0; i < 2; i++ )
            {
                if( y >= edge[i].ye )
                {
                    int idx0 = edge[i].idx, di = edge[i].di;
                    int idx = idx0 + di;
                    if (idx >= npts) idx -= npts;
                    int ty = 0;

                    for (; edges-- > 0; )
                    {
                        ty = (int)((v[idx].y + delta) >> shift);
                        if (ty > y)
                        {
                            int64 xs = v[idx0].x;
                            int64 xe = v[idx].x;
                            if (shift != XY_SHIFT)
                            {
                                xs <<= XY_SHIFT - shift;
                                xe <<= XY_SHIFT - shift;
                            }

                            edge[i].ye = ty;
                            edge[i].dx = ((xe - xs)*2 + ((int64_t)ty - y)) / (2 * ((int64_t)ty - y));
                            edge[i].x = xs;
                            edge[i].idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx += di;
                        if (idx >= npts) idx -= npts;
                    }
                }
            }
        }

        if (edges < 0)
            break;

        if (y >= 0)
        {
            int left = 0, right = 1;
            if (edge[0].x > edge[1].x)
            {
                left = 1, right = 0;
            }

            int xx1 = (int)((edge[left].x + delta1) >> XY_SHIFT);
            int xx2 = (int)((edge[right].x + delta2) >> XY_SHIFT);

            if( xx2 >= 0 && xx1 < size.width )
            {
                if( xx1 < 0 )
                    xx1 = 0;
                if( xx2 >= size.width )
                    xx2 = size.width - 1;
                ICV_HLINE( ptr, xx1, xx2, color, pix_size );
            }
        }
        else
        {
            // TODO optimize scan for negative y
        }

        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        ptr += img.step;
    }
    while( ++y <= (int)ymax );
}


/******** Arbitrary polygon **********/

static void
CollectPolyEdges( Mat& img, const Point2l* v, int count, std::vector<PolyEdge>& edges,
                  const void* color, int line_type, int shift, Point offset )
{
    int i, delta = offset.y + ((1 << shift) >> 1);
    Point2l pt0 = v[count-1], pt1;
    pt0.x = (pt0.x + offset.x) << (XY_SHIFT - shift);
    pt0.y = (pt0.y + delta) >> shift;

    edges.reserve( edges.size() + count );

    for( i = 0; i < count; i++, pt0 = pt1 )
    {
        Point2l t0, t1;
        PolyEdge edge;

        pt1 = v[i];
        pt1.x = (pt1.x + offset.x) << (XY_SHIFT - shift);
        pt1.y = (pt1.y + delta) >> shift;

        Point2l pt0c(pt0), pt1c(pt1);

        if (line_type < cv::LINE_AA)
        {
            t0.y = pt0.y; t1.y = pt1.y;
            t0.x = (pt0.x + (XY_ONE >> 1)) >> XY_SHIFT;
            t1.x = (pt1.x + (XY_ONE >> 1)) >> XY_SHIFT;
            Line(img, t0, t1, color, line_type);

            // use clipped endpoints to create a more accurate PolyEdge
            if ((unsigned)t0.x >= (unsigned)(img.cols) ||
                (unsigned)t1.x >= (unsigned)(img.cols) ||
                (unsigned)t0.y >= (unsigned)(img.rows) ||
                (unsigned)t1.y >= (unsigned)(img.rows))
            {
                clipLine(img.size(), t0, t1);

                if (t0.y != t1.y)
                {
                    pt0c.y = t0.y; pt1c.y = t1.y;
                }
            }

            pt0c.x = (int64)(t0.x) << XY_SHIFT;
            pt1c.x = (int64)(t1.x) << XY_SHIFT;
        }
        else
        {
            t0.x = pt0.x; t1.x = pt1.x;
            t0.y = pt0.y << XY_SHIFT;
            t1.y = pt1.y << XY_SHIFT;
            LineAA(img, t0, t1, color);
        }

        if (pt0.y == pt1.y)
            continue;

        edge.dx = (pt1c.x - pt0c.x) / (pt1c.y - pt0c.y);
        if (pt0.y < pt1.y)
        {
            edge.y0 = (int)(pt0.y);
            edge.y1 = (int)(pt1.y);
            edge.x = pt0c.x + (pt0.y - pt0c.y) * edge.dx; // correct starting point for clipped lines
        }
        else
        {
            edge.y0 = (int)(pt1.y);
            edge.y1 = (int)(pt0.y);
            edge.x = pt1c.x + (pt1.y - pt1c.y) * edge.dx; // correct starting point for clipped lines
        }
        edges.push_back(edge);
    }
}

struct CmpEdges
{
    bool operator ()(const PolyEdge& e1, const PolyEdge& e2)
    {
        return e1.y0 - e2.y0 ? e1.y0 < e2.y0 :
            e1.x - e2.x ? e1.x < e2.x : e1.dx < e2.dx;
    }
};

/**************** helper macros and functions for sequence/contour processing ***********/

static void
FillEdgeCollection( Mat& img, std::vector<PolyEdge>& edges, const void* color )
{
    PolyEdge tmp;
    int i, y, total = (int)edges.size();
    Size size = img.size();
    PolyEdge* e;
    int y_max = INT_MIN, y_min = INT_MAX;
    int64 x_max = 0xFFFFFFFFFFFFFFFF, x_min = 0x7FFFFFFFFFFFFFFF;
    int pix_size = (int)img.elemSize();
    int delta = XY_ONE - 1;

    if( total < 2 )
        return;

    for( i = 0; i < total; i++ )
    {
        PolyEdge& e1 = edges[i];
        CV_Assert( e1.y0 < e1.y1 );
        // Determine x-coordinate of the end of the edge.
        // (This is not necessary x-coordinate of any vertex in the array.)
        int64 x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
        y_min = std::min( y_min, e1.y0 );
        y_max = std::max( y_max, e1.y1 );
        x_min = std::min( x_min, e1.x );
        x_max = std::max( x_max, e1.x );
        x_min = std::min( x_min, x1 );
        x_max = std::max( x_max, x1 );
    }

    if( y_max < 0 || y_min >= size.height || x_max < 0 || x_min >= ((int64)size.width<<XY_SHIFT) )
        return;

    std::sort( edges.begin(), edges.end(), CmpEdges() );

    // start drawing
    tmp.y0 = INT_MAX;
    edges.push_back(tmp); // after this point we do not add
                          // any elements to edges, thus we can use pointers
    i = 0;
    tmp.next = 0;
    e = &edges[i];
    y_max = MIN( y_max, size.height );

    for( y = e->y0; y < y_max; y++ )
    {
        PolyEdge *last, *prelast, *keep_prelast;
        int draw = 0;
        int clipline = y < 0;

        prelast = &tmp;
        last = tmp.next;
        while( last || e->y0 == y )
        {
            if( last && last->y1 == y )
            {
                // exclude edge if y reaches its lower point
                prelast->next = last->next;
                last = last->next;
                continue;
            }
            keep_prelast = prelast;
            if( last && (e->y0 > y || last->x < e->x) )
            {
                // go to the next edge in active list
                prelast = last;
                last = last->next;
            }
            else if( i < total )
            {
                // insert new edge into active list if y reaches its upper point
                prelast->next = e;
                e->next = last;
                prelast = e;
                e = &edges[++i];
            }
            else
                break;

            if( draw )
            {
                if( !clipline )
                {
                    // convert x's from fixed-point to image coordinates
                    uchar *timg = img.ptr(y);
                    int x1, x2;

                    if (keep_prelast->x > prelast->x)
                    {
                        x1 = (int)((prelast->x + delta) >> XY_SHIFT);
                        x2 = (int)(keep_prelast->x >> XY_SHIFT);
                    }
                    else
                    {
                        x1 = (int)((keep_prelast->x + delta) >> XY_SHIFT);
                        x2 = (int)(prelast->x >> XY_SHIFT);
                    }

                    // clip and draw the line
                    if( x1 < size.width && x2 >= 0 )
                    {
                        if( x1 < 0 )
                            x1 = 0;
                        if( x2 >= size.width )
                            x2 = size.width - 1;
                        ICV_HLINE( timg, x1, x2, color, pix_size );
                    }
                }
                keep_prelast->x += keep_prelast->dx;
                prelast->x += prelast->dx;
            }
            draw ^= 1;
        }

        // sort edges (using bubble sort)
        keep_prelast = 0;

        do
        {
            prelast = &tmp;
            last = tmp.next;
            PolyEdge *last_exchange = 0;

            while( last != keep_prelast && last->next != 0 )
            {
                PolyEdge *te = last->next;

                // swap edges
                if( last->x > te->x )
                {
                    prelast->next = te;
                    last->next = te->next;
                    te->next = last;
                    prelast = te;
                    last_exchange = prelast;
                }
                else
                {
                    prelast = last;
                    last = te;
                }
            }
            if (last_exchange == NULL)
                break;
            keep_prelast = last_exchange;
        } while( keep_prelast != tmp.next && keep_prelast != &tmp );
    }
}


/* draws simple or filled circle */
static void
Circle( Mat& img, Point center, int radius, const void* color, int fill )
{
    Size size = img.size();
    size_t step = img.step;
    int pix_size = (int)img.elemSize();
    uchar* ptr = img.ptr();
    int64_t err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1;
    int inside = center.x >= radius && center.x < size.width - radius &&
        center.y >= radius && center.y < size.height - radius;

    #define ICV_PUT_POINT( ptr, x )     \
        memcpy( ptr + (x)*pix_size, color, pix_size );

    while( dx >= dy )
    {
        int mask;
        int64_t y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
        int64_t x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

        if( inside )
        {
            uchar *tptr0 = ptr + y11 * step;
            uchar *tptr1 = ptr + y12 * step;

            if( !fill )
            {
                ICV_PUT_POINT( tptr0, x11 );
                ICV_PUT_POINT( tptr1, x11 );
                ICV_PUT_POINT( tptr0, x12 );
                ICV_PUT_POINT( tptr1, x12 );
            }
            else
            {
                ICV_HLINE( tptr0, x11, x12, color, pix_size );
                ICV_HLINE( tptr1, x11, x12, color, pix_size );
            }

            tptr0 = ptr + y21 * step;
            tptr1 = ptr + y22 * step;

            if( !fill )
            {
                ICV_PUT_POINT( tptr0, x21 );
                ICV_PUT_POINT( tptr1, x21 );
                ICV_PUT_POINT( tptr0, x22 );
                ICV_PUT_POINT( tptr1, x22 );
            }
            else
            {
                ICV_HLINE( tptr0, x21, x22, color, pix_size );
                ICV_HLINE( tptr1, x21, x22, color, pix_size );
            }
        }
        else if( x11 < size.width && x12 >= 0 && y21 < size.height && y22 >= 0)
        {
            if( fill )
            {
                x11 = std::max( x11, (int64_t)0 );
                x12 = MIN( x12, size.width - 1 );
            }

            if( y11 >= 0 && y11 < size.height )
            {
                uchar *tptr = ptr + y11 * step;

                if( !fill )
                {
                    if( x11 >= 0 )
                        ICV_PUT_POINT( tptr, x11 );
                    if( x12 < size.width )
                        ICV_PUT_POINT( tptr, x12 );
                }
                else
                    ICV_HLINE( tptr, x11, x12, color, pix_size );
            }

            if( y12 >= 0 && y12 < size.height )
            {
                uchar *tptr = ptr + y12 * step;

                if( !fill )
                {
                    if( x11 >= 0 )
                        ICV_PUT_POINT( tptr, x11 );
                    if( x12 < size.width )
                        ICV_PUT_POINT( tptr, x12 );
                }
                else
                    ICV_HLINE( tptr, x11, x12, color, pix_size );
            }

            if( x21 < size.width && x22 >= 0 )
            {
                if( fill )
                {
                    x21 = std::max( x21, (int64_t)0 );
                    x22 = MIN( x22, size.width - 1 );
                }

                if( y21 >= 0 && y21 < size.height )
                {
                    uchar *tptr = ptr + y21 * step;

                    if( !fill )
                    {
                        if( x21 >= 0 )
                            ICV_PUT_POINT( tptr, x21 );
                        if( x22 < size.width )
                            ICV_PUT_POINT( tptr, x22 );
                    }
                    else
                        ICV_HLINE( tptr, x21, x22, color, pix_size );
                }

                if( y22 >= 0 && y22 < size.height )
                {
                    uchar *tptr = ptr + y22 * step;

                    if( !fill )
                    {
                        if( x21 >= 0 )
                            ICV_PUT_POINT( tptr, x21 );
                        if( x22 < size.width )
                            ICV_PUT_POINT( tptr, x22 );
                    }
                    else
                        ICV_HLINE( tptr, x21, x22, color, pix_size );
                }
            }
        }
        dy++;
        err += plus;
        plus += 2;

        mask = (err <= 0) - 1;

        err -= minus & mask;
        dx += mask;
        minus -= mask & 2;
    }

    #undef  ICV_PUT_POINT
}


static void
ThickLine( Mat& img, Point2l p0, Point2l p1, const void* color,
           int thickness, int line_type, int flags, int shift )
{
    static const double INV_XY_ONE = 1./static_cast<double>(XY_ONE);

    p0.x <<= XY_SHIFT - shift;
    p0.y <<= XY_SHIFT - shift;
    p1.x <<= XY_SHIFT - shift;
    p1.y <<= XY_SHIFT - shift;

    if( thickness <= 1 )
    {
        if( line_type < cv::LINE_AA )
        {
            if( line_type == 1 || line_type == 4 || shift == 0 )
            {
                p0.x = (p0.x + (XY_ONE>>1)) >> XY_SHIFT;
                p0.y = (p0.y + (XY_ONE>>1)) >> XY_SHIFT;
                p1.x = (p1.x + (XY_ONE>>1)) >> XY_SHIFT;
                p1.y = (p1.y + (XY_ONE>>1)) >> XY_SHIFT;
                Line( img, p0, p1, color, line_type );
            }
            else
                Line2( img, p0, p1, color );
        }
        else
            LineAA( img, p0, p1, color );
    }
    else
    {
        Point2l pt[4], dp = Point2l(0,0);
        double dx = (p0.x - p1.x)*INV_XY_ONE, dy = (p1.y - p0.y)*INV_XY_ONE;
        double r = dx * dx + dy * dy;
        int i, oddThickness = thickness & 1;
        thickness <<= XY_SHIFT - 1;

        if( fabs(r) > DBL_EPSILON )
        {
            r = (thickness + oddThickness*XY_ONE*0.5)/std::sqrt(r);
            dp.x = cvRound( dy * r );
            dp.y = cvRound( dx * r );

            pt[0].x = p0.x + dp.x;
            pt[0].y = p0.y + dp.y;
            pt[1].x = p0.x - dp.x;
            pt[1].y = p0.y - dp.y;
            pt[2].x = p1.x - dp.x;
            pt[2].y = p1.y - dp.y;
            pt[3].x = p1.x + dp.x;
            pt[3].y = p1.y + dp.y;

            FillConvexPoly( img, pt, 4, color, line_type, XY_SHIFT );
        }

        for( i = 0; i < 2; i++ )
        {
            if( flags & (i+1) )
            {
                if( line_type < cv::LINE_AA )
                {
                    Point center;
                    center.x = (int)((p0.x + (XY_ONE>>1)) >> XY_SHIFT);
                    center.y = (int)((p0.y + (XY_ONE>>1)) >> XY_SHIFT);
                    Circle( img, center, (thickness + (XY_ONE>>1)) >> XY_SHIFT, color, 1 );
                }
                else
                {
                    EllipseEx( img, p0, Size2l(thickness, thickness),
                               0, 0, 360, color, -1, line_type );
                }
            }
            p0 = p1;
        }
    }
}


static void
PolyLine( Mat& img, const Point2l* v, int count, bool is_closed,
          const void* color, int thickness,
          int line_type, int shift )
{
    if( !v || count <= 0 )
        return;

    int i = is_closed ? count - 1 : 0;
    int flags = 2 + !is_closed;
    Point2l p0;
    CV_Assert( 0 <= shift && shift <= XY_SHIFT && thickness >= 0 );

    p0 = v[i];
    for( i = !is_closed; i < count; i++ )
    {
        Point2l p = v[i];
        ThickLine( img, p0, p, color, thickness, line_type, flags, shift );
        p0 = p;
        flags = 2;
    }
}

/* ----------------------------------------------------------------------------------------- */
/* ADDING A SET OF PREDEFINED MARKERS WHICH COULD BE USED TO HIGHLIGHT POSITIONS IN AN IMAGE */
/* ----------------------------------------------------------------------------------------- */

void drawMarker(InputOutputArray img, Point position, const Scalar& color, int markerType, int markerSize, int thickness, int line_type)
{
    switch(markerType)
    {
    // The cross marker case
    case MARKER_CROSS:
        line(img, Point(position.x-(markerSize/2), position.y), Point(position.x+(markerSize/2), position.y), color, thickness, line_type);
        line(img, Point(position.x, position.y-(markerSize/2)), Point(position.x, position.y+(markerSize/2)), color, thickness, line_type);
        break;

    // The tilted cross marker case
    case MARKER_TILTED_CROSS:
        line(img, Point(position.x-(markerSize/2), position.y-(markerSize/2)), Point(position.x+(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y-(markerSize/2)), Point(position.x-(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        break;

    // The star marker case
    case MARKER_STAR:
        line(img, Point(position.x-(markerSize/2), position.y), Point(position.x+(markerSize/2), position.y), color, thickness, line_type);
        line(img, Point(position.x, position.y-(markerSize/2)), Point(position.x, position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x-(markerSize/2), position.y-(markerSize/2)), Point(position.x+(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y-(markerSize/2)), Point(position.x-(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        break;

    // The diamond marker case
    case MARKER_DIAMOND:
        line(img, Point(position.x, position.y-(markerSize/2)), Point(position.x+(markerSize/2), position.y), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y), Point(position.x, position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x, position.y+(markerSize/2)), Point(position.x-(markerSize/2), position.y), color, thickness, line_type);
        line(img, Point(position.x-(markerSize/2), position.y), Point(position.x, position.y-(markerSize/2)), color, thickness, line_type);
        break;

    // The square marker case
    case MARKER_SQUARE:
        line(img, Point(position.x-(markerSize/2), position.y-(markerSize/2)), Point(position.x+(markerSize/2), position.y-(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y-(markerSize/2)), Point(position.x+(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y+(markerSize/2)), Point(position.x-(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x-(markerSize/2), position.y+(markerSize/2)), Point(position.x-(markerSize/2), position.y-(markerSize/2)), color, thickness, line_type);
        break;

    // The triangle up marker case
    case MARKER_TRIANGLE_UP:
        line(img, Point(position.x-(markerSize/2), position.y+(markerSize/2)), Point(position.x+(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y+(markerSize/2)), Point(position.x, position.y-(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x, position.y-(markerSize/2)), Point(position.x-(markerSize/2), position.y+(markerSize/2)), color, thickness, line_type);
        break;

    // The triangle down marker case
    case MARKER_TRIANGLE_DOWN:
        line(img, Point(position.x-(markerSize/2), position.y-(markerSize/2)), Point(position.x+(markerSize/2), position.y-(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x+(markerSize/2), position.y-(markerSize/2)), Point(position.x, position.y+(markerSize/2)), color, thickness, line_type);
        line(img, Point(position.x, position.y+(markerSize/2)), Point(position.x-(markerSize/2), position.y-(markerSize/2)), color, thickness, line_type);
        break;

    // If any number that doesn't exist is entered as marker type, draw a cross marker, to avoid crashes
    default:
        drawMarker(img, position, color, MARKER_CROSS, markerSize, thickness, line_type);
        break;
    }
}

/****************************************************************************************\
*                              External functions                                        *
\****************************************************************************************/

void line( InputOutputArray _img, Point pt1, Point pt2, const Scalar& color,
           int thickness, int line_type, int shift )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( line_type == cv::LINE_AA && img.depth() != CV_8U )
        line_type = 8;

    CV_Assert( 0 < thickness && thickness <= MAX_THICKNESS );
    CV_Assert( 0 <= shift && shift <= XY_SHIFT );

    double buf[4];
    scalarToRawData( color, buf, img.type(), 0 );
    ThickLine( img, pt1, pt2, buf, thickness, line_type, 3, shift );
}

void arrowedLine(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,
           int thickness, int line_type, int shift, double tipLength)
{
    CV_INSTRUMENT_REGION();

    const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow

    line(img, pt1, pt2, color, thickness, line_type, shift);

    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );

    Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
        cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
    line(img, p, pt2, color, thickness, line_type, shift);

    p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
}

void rectangle( InputOutputArray _img, Point pt1, Point pt2,
                const Scalar& color, int thickness,
                int lineType, int shift )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( lineType == cv::LINE_AA && img.depth() != CV_8U )
        lineType = 8;

    CV_Assert( thickness <= MAX_THICKNESS );
    CV_Assert( 0 <= shift && shift <= XY_SHIFT );

    double buf[4];
    scalarToRawData(color, buf, img.type(), 0);

    Point2l pt[4];

    pt[0] = pt1;
    pt[1].x = pt2.x;
    pt[1].y = pt1.y;
    pt[2] = pt2;
    pt[3].x = pt1.x;
    pt[3].y = pt2.y;

    if( thickness >= 0 )
        PolyLine( img, pt, 4, true, buf, thickness, lineType, shift );
    else
        FillConvexPoly( img, pt, 4, buf, lineType, shift );
}


void rectangle( InputOutputArray img, Rect rec,
                const Scalar& color, int thickness,
                int lineType, int shift )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( 0 <= shift && shift <= XY_SHIFT );

    // Crop the rectangle to right around the mat.
    rec &= Rect(-(1 << shift), -(1 << shift), ((img.cols() + 2) << shift),
                ((img.rows() + 2) << shift));

    if( !rec.empty() )
        rectangle( img, rec.tl(), rec.br() - Point(1<<shift,1<<shift),
                   color, thickness, lineType, shift );
}


void circle( InputOutputArray _img, Point center, int radius,
             const Scalar& color, int thickness, int line_type, int shift )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( line_type == cv::LINE_AA && img.depth() != CV_8U )
        line_type = 8;

    CV_Assert( radius >= 0 && thickness <= MAX_THICKNESS &&
        0 <= shift && shift <= XY_SHIFT );

    double buf[4];
    scalarToRawData(color, buf, img.type(), 0);

    if( thickness > 1 || line_type != LINE_8 || shift > 0 )
    {
        Point2l _center(center);
        int64 _radius(radius);
        _center.x <<= XY_SHIFT - shift;
        _center.y <<= XY_SHIFT - shift;
        _radius <<= XY_SHIFT - shift;
        EllipseEx( img, _center, Size2l(_radius, _radius),
                   0, 0, 360, buf, thickness, line_type );
    }
    else
        Circle( img, center, radius, buf, thickness < 0 );
}


void ellipse( InputOutputArray _img, Point center, Size axes,
              double angle, double start_angle, double end_angle,
              const Scalar& color, int thickness, int line_type, int shift )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( line_type == cv::LINE_AA && img.depth() != CV_8U )
        line_type = 8;

    CV_Assert( axes.width >= 0 && axes.height >= 0 &&
        thickness <= MAX_THICKNESS && 0 <= shift && shift <= XY_SHIFT );

    double buf[4];
    scalarToRawData(color, buf, img.type(), 0);

    int _angle = cvRound(angle);
    int _start_angle = cvRound(start_angle);
    int _end_angle = cvRound(end_angle);
    Point2l _center(center);
    Size2l _axes(axes);
    _center.x <<= XY_SHIFT - shift;
    _center.y <<= XY_SHIFT - shift;
    _axes.width <<= XY_SHIFT - shift;
    _axes.height <<= XY_SHIFT - shift;

    EllipseEx( img, _center, _axes, _angle, _start_angle,
               _end_angle, buf, thickness, line_type );
}

void ellipse(InputOutputArray _img, const RotatedRect& box, const Scalar& color,
             int thickness, int lineType)
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( lineType == cv::LINE_AA && img.depth() != CV_8U )
        lineType = 8;

    CV_Assert( box.size.width >= 0 && box.size.height >= 0 &&
               thickness <= MAX_THICKNESS );

    double buf[4];
    scalarToRawData(color, buf, img.type(), 0);

    int _angle = cvRound(box.angle);
    Point2l center(cvRound(box.center.x),
                 cvRound(box.center.y));
    center.x = (center.x << XY_SHIFT) + cvRound((box.center.x - center.x)*static_cast<float>(XY_ONE));
    center.y = (center.y << XY_SHIFT) + cvRound((box.center.y - center.y)*static_cast<float>(XY_ONE));
    Size2l axes(cvRound(box.size.width),
              cvRound(box.size.height));
    axes.width  = (axes.width  << (XY_SHIFT - 1)) + cvRound((box.size.width - axes.width)*(XY_ONE>>1));
    axes.height = (axes.height << (XY_SHIFT - 1)) + cvRound((box.size.height - axes.height)*(XY_ONE>>1));
    EllipseEx( img, center, axes, _angle, 0, 360, buf, thickness, lineType );
}

void fillConvexPoly( InputOutputArray _img, const Point* pts, int npts,
                     const Scalar& color, int line_type, int shift )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( !pts || npts <= 0 )
        return;

    if( line_type == cv::LINE_AA && img.depth() != CV_8U )
        line_type = 8;

    double buf[4];
    CV_Assert( 0 <= shift && shift <=  XY_SHIFT );
    scalarToRawData(color, buf, img.type(), 0);
    std::vector<Point2l> _pts(pts, pts + npts);
    FillConvexPoly( img, _pts.data(), npts, buf, line_type, shift );
}

void fillPoly( InputOutputArray _img, const Point** pts, const int* npts, int ncontours,
               const Scalar& color, int line_type,
               int shift, Point offset )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( line_type == cv::LINE_AA && img.depth() != CV_8U )
        line_type = 8;

    CV_Assert( pts && npts && ncontours >= 0 && 0 <= shift && shift <= XY_SHIFT );

    double buf[4];
    scalarToRawData(color, buf, img.type(), 0);

    std::vector<PolyEdge> edges;

    int i, total = 0;
    for( i = 0; i < ncontours; i++ )
        total += npts[i];

    edges.reserve( total + 1 );
    for (i = 0; i < ncontours; i++)
    {
        if (npts[i] > 0 && pts[i])
        {
            std::vector<Point2l> _pts(pts[i], pts[i] + npts[i]);
            CollectPolyEdges(img, _pts.data(), npts[i], edges, buf, line_type, shift, offset);
        }
    }

    FillEdgeCollection(img, edges, buf);
}

void polylines( InputOutputArray _img, const Point* const* pts, const int* npts, int ncontours, bool isClosed,
                const Scalar& color, int thickness, int line_type, int shift )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();

    if( line_type == cv::LINE_AA && img.depth() != CV_8U )
        line_type = 8;

    CV_Assert( pts && npts && ncontours >= 0 &&
               0 <= thickness && thickness <= MAX_THICKNESS &&
               0 <= shift && shift <= XY_SHIFT );

    double buf[4];
    scalarToRawData( color, buf, img.type(), 0 );

    for( int i = 0; i < ncontours; i++ )
    {
        std::vector<Point2l> _pts(pts[i], pts[i]+npts[i]);
        PolyLine( img, _pts.data(), npts[i], isClosed, buf, thickness, line_type, shift );
    }
}

}

void cv::fillConvexPoly(InputOutputArray img, InputArray _points,
                        const Scalar& color, int lineType, int shift)
{
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();
    CV_Assert(points.checkVector(2, CV_32S) >= 0);
    fillConvexPoly(img, points.ptr<Point>(), points.rows*points.cols*points.channels()/2, color, lineType, shift);
}

void cv::fillPoly(InputOutputArray img, InputArrayOfArrays pts,
                  const Scalar& color, int lineType, int shift, Point offset)
{
    CV_INSTRUMENT_REGION();

    bool manyContours = pts.kind() == _InputArray::STD_VECTOR_VECTOR ||
                        pts.kind() == _InputArray::STD_VECTOR_MAT;
    int i, ncontours = manyContours ? (int)pts.total() : 1;
    if( ncontours == 0 )
        return;
    AutoBuffer<Point*> _ptsptr(ncontours);
    AutoBuffer<int> _npts(ncontours);
    Point** ptsptr = _ptsptr.data();
    int* npts = _npts.data();

    for( i = 0; i < ncontours; i++ )
    {
        Mat p = pts.getMat(manyContours ? i : -1);
        CV_Assert(p.checkVector(2, CV_32S) > 0);
        ptsptr[i] = p.ptr<Point>();
        npts[i] = p.rows*p.cols*p.channels()/2;
    }
    fillPoly(img, (const Point**)ptsptr, npts, (int)ncontours, color, lineType, shift, offset);
}

void cv::polylines(InputOutputArray img, InputArrayOfArrays pts,
                   bool isClosed, const Scalar& color,
                   int thickness, int lineType, int shift)
{
    CV_INSTRUMENT_REGION();

    bool manyContours = pts.kind() == _InputArray::STD_VECTOR_VECTOR ||
                        pts.kind() == _InputArray::STD_VECTOR_MAT;
    int i, ncontours = manyContours ? (int)pts.total() : 1;
    if( ncontours == 0 )
        return;
    AutoBuffer<Point*> _ptsptr(ncontours);
    AutoBuffer<int> _npts(ncontours);
    Point** ptsptr = _ptsptr.data();
    int* npts = _npts.data();

    for( i = 0; i < ncontours; i++ )
    {
        Mat p = pts.getMat(manyContours ? i : -1);
        if( p.total() == 0 )
        {
            ptsptr[i] = NULL;
            npts[i] = 0;
            continue;
        }
        CV_Assert(p.checkVector(2, CV_32S) >= 0);
        ptsptr[i] = p.ptr<Point>();
        npts[i] = p.rows*p.cols*p.channels()/2;
    }
    polylines(img, (const Point**)ptsptr, npts, (int)ncontours, isClosed, color, thickness, lineType, shift);
}



void cv::drawContours( InputOutputArray _image, InputArrayOfArrays _contours,
                   int contourIdx, const Scalar& color, int thickness,
                   int lineType, InputArray _hierarchy,
                   int maxLevel, Point offset )
{
    CV_INSTRUMENT_REGION();
    CV_Assert( thickness <= MAX_THICKNESS );
    const size_t ncontours = _contours.total();
    if (!ncontours)
        return;
    CV_Assert(ncontours <= (size_t)std::numeric_limits<int>::max());
    if (lineType == cv::LINE_AA && _image.depth() != CV_8U)
        lineType = 8;
    Mat image = _image.getMat();
    Mat_<Vec4i> hierarchy = _hierarchy.getMat();

    int i = 0, end = (int)ncontours;
    if (contourIdx >= 0)
    {
        i = contourIdx;
        end = i + 1;
    }
    std::vector<int> indexesToFill;
    if (hierarchy.empty() || maxLevel == 0)
    {
        indexesToFill.resize(end - i);
        std::iota(indexesToFill.begin(), indexesToFill.end(), i);
    }
    else
    {
        std::stack<int> indexes;
        for (; i != end; ++i)
        {
            // either all from the top level or a single contour
            if (hierarchy(i)[3] < 0 || contourIdx >= 0)
                indexes.push(i);
        }
        while (!indexes.empty())
        {
            // get current element
            const int cur = indexes.top();
            indexes.pop();

            //  check current element depth
            int curLevel = -1;
            int par = cur;
            while (par >= 0)
            {
                par = hierarchy(par)[3]; // parent
                ++curLevel;
            }
            if (curLevel <= maxLevel)
            {
                indexesToFill.push_back(cur);
            }

            int next = hierarchy(cur)[2]; // first child
            while (next > 0)
            {
                indexes.push(next);
                next = hierarchy(next)[0]; // next sibling
            }
        }
    }
    std::vector<Mat> contoursToFill;
    contoursToFill.reserve(indexesToFill.size());
    for (const int& idx : indexesToFill)
        contoursToFill.emplace_back(_contours.getMat(idx));

    if (thickness < 0)
        fillPoly(image, contoursToFill, color, lineType, 0, offset);
    else
    {
        double color_buf[4]{};
        scalarToRawData(color, color_buf, _image.type(), 0);
        for (const Mat& cnt : contoursToFill)
        {
            if (cnt.empty())
                continue;
            const int npoints = cnt.checkVector(2, CV_32S);
            CV_Assert(npoints > 0);
            for (int j = 0; j < npoints; ++j)
            {
                const bool isLastIter = j == npoints - 1;
                const Point pt1 = cnt.at<Point>(j);
                const Point pt2 = cnt.at<Point>(isLastIter ? 0 : j + 1);
                cv::ThickLine(image, pt1 + offset, pt2 + offset, color_buf, thickness, lineType, 2, 0);
            }
        }
    }
}
