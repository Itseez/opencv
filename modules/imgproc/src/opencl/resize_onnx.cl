// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef DOUBLE_SUPPORT
#   ifdef cl_amd_fp64
#       pragma OPENCL EXTENSION cl_amd_fp64:enable
#   elif defined (cl_khr_fp64)
#       pragma OPENCL EXTENSION cl_khr_fp64:enable
#   endif
#endif

#define noconvert(x) (x)

// for debug and intellisense
#ifndef T
#   define INTER_NEAREST1
#   define INTER_LINEAR1
#   define INTER_CUBIC
#   define INTER_ANTIALIAS1
#   define EXCLUDE_OUTSIDE 1
#   define T int
#   define W double
#   define CN 3
#   define PIXEL_SIZE 12
#   define VT int3
#   define VW double3
#   define TO_WORK     convert_double
#   define TO_VEC_WORK convert_double3
#   define TO_TYPE     convert_int_sat_rte
#   define TO_VEC_TYPE convert_int3_sat_rte
#endif

// use parameter `channel' to reduce the number of kernels
#if CN != 3
#   define loadpix(addr)        *(__global const VT*)(addr)
#   define storepix(val, addr)  *(__global VT*)(addr) = val
#else
#   define loadpix(addr)       vload3(0, (__global const T*)(addr))
#   define storepix(val, addr) vstore3(val, 0, (__global T*)(addr))
#endif

#if defined(INTER_NEAREST)

__kernel void resizeOnnx_nearest(
    __global uchar const* srcptr, int src_step, int src_offset, int src_rows, int src_cols,
    __global uchar      * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
    int pixel_size, float offset, float m00, float m01, float m10, float m11)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    if (dx < dst_cols && dy < dst_rows)
    {
        float fx = fma(dx, m00 , m01), fy = fma(dy, m10, m11);
#if defined(INTER_NEAREST_PREFER_FLOOR) || defined(INTER_NEAREST_CEIL)
        // x, y will >= 0, so `round toward positive infinity' is equivalent to ceil
        int sx = convert_int_rtp(fx + offset);
        int sy = convert_int_rtp(fy + offset);
#else
        // x, y will >= 0, so `round toward negative infinity' is equivalent to floor
        int sx = convert_int_rtn(fx + offset);
        int sy = convert_int_rtn(fy + offset);
#endif
        sx = clamp(sx, 0, src_cols - 1);
        sy = clamp(sy, 0, src_rows - 1);
        // maybe step >= 8M, so do not use `mad24' for y
        __global uchar const* S = srcptr + (sy * src_step + mad24(sx, pixel_size, src_offset));
        __global uchar      * D = dstptr + (dy * dst_step + mad24(dx, pixel_size, dst_offset));

#if PIXEL_SIZE == 1
        *D = *S;
#elif PIXEL_SIZE == 2
        *(__global ushort*)(D) = *(__global const ushort*)(S);
#elif PIXEL_SIZE == 3
        vstore3(vload3(0, S), 0, D);
#elif PIXEL_SIZE == 4
        *(__global uint*)(D) = *(__global const uint*)(S);
#elif PIXEL_SIZE == 6
        vstore3(vload3(0, (__global ushort const*)(S)), 0, (__global ushort*)(D));
#elif PIXEL_SIZE == 8
        *(__global uint2*)(D) = *(__global const uint2*)(S);
#elif PIXEL_SIZE == 12
        vstore3(vload3(0, (__global const uint*)(S)), 0, (__global uint*)(D));
#elif PIXEL_SIZE == 16
        *(__global uint4*)(D) = *(__global const uint4*)(S);
#else
        for (int i = 0; i < pixel_size; ++i)
            D[i] = S[i];
#endif
    }
}

#elif defined(INTER_LINEAR) && !defined(INTER_ANTIALIAS)

__kernel void resizeOnnx_linear(
    __global uchar const* srcptr, int src_step, int src_offset, int src_rows, int src_cols,
    __global uchar      * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
    int pixel_size, int channel, float m00, float m01, float m10, float m11)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    if (dx < dst_cols && dy < dst_rows)
    {
        float fx = fma(dx, m00, m01), fy = fma(dy, m10, m11);
        int ix = convert_int_rtn(fx), iy = convert_int_rtn(fy);
        float u1 = fx - ix, v1 = fy - iy;
        float u0 = 1.f - u1, v0 = 1.f - v1;
        int x0 = max(ix, 0);
        int y0 = max(iy, 0);
        int x1 = min(ix + 1, src_cols - 1);
        int y1 = min(iy + 1, src_rows - 1);
        __global uchar const* S0 = srcptr + (y0 * src_step + mad24(x0, pixel_size, src_offset));
        __global uchar const* S1 = srcptr + (y0 * src_step + mad24(x1, pixel_size, src_offset));
        __global uchar const* S2 = srcptr + (y1 * src_step + mad24(x0, pixel_size, src_offset));
        __global uchar const* S3 = srcptr + (y1 * src_step + mad24(x1, pixel_size, src_offset));
        __global uchar      * D  = dstptr + (dy * dst_step + mad24(dx, pixel_size, dst_offset));
#if CN == 1 || CN == 2 || CN == 3 || CN == 4
        VW s0 = TO_VEC_WORK(loadpix(S0)); VW s1 = TO_VEC_WORK(loadpix(S1));
        VW s2 = TO_VEC_WORK(loadpix(S2)); VW s3 = TO_VEC_WORK(loadpix(S3));
        VT d0 = TO_VEC_TYPE((u0 * v0) * s0 + (u1 * v0) * s1 + (u0 * v1) * s2 + (u1 * v1) * s3);
        storepix(d0, D);
#else
        W coeff[4] = { u0 * v0, u1 * v0, u0 * v1, u1 * v1 };
        for (int i = 0; i < channel; ++i)
        {
            W s0 = TO_WORK(((__global T const*)(S0))[i]);
            W s1 = TO_WORK(((__global T const*)(S1))[i]);
            W s2 = TO_WORK(((__global T const*)(S2))[i]);
            W s3 = TO_WORK(((__global T const*)(S3))[i]);
            W d0 = coeff[0] * s0 + coeff[1] * s1 + coeff[2] * s2 + coeff[3] * s3;
            ((__global T*)(D))[i] = TO_TYPE(d0);
        }
#endif
    }
}

#elif defined(INTER_LINEAR) && defined(INTER_ANTIALIAS)

__kernel void resizeOnnx_linear_antialias(
    __global uchar const* srcptr, int src_step, int src_offset, int src_rows, int src_cols,
    __global uchar      * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
    int pixel_size, int channel, float m00, float m01, float m10, float m11,
    float xscale, float yscale, int xstart, int ystart, int xend, int yend)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    if (dx < dst_cols && dy < dst_rows)
    {
        float fx = fma(dx, m00, m01), fy = fma(dy, m10, m11);
        int ix = convert_int_rtn(fx), iy = convert_int_rtn(fy);
        float rx = fx - ix, ry = fy - iy;
        __global uchar* D = dstptr + dy * dst_step + mad24(dx, pixel_size, dst_offset);
#if CN == 1 || CN == 2 || CN == 3 || CN == 4
        VW sumval = (VW)(0);
        float weight = 0;
        for (int h = ystart; h < yend; ++h)
        {
            VW sline = (VW)(0);
            float wline = 0;
            int sy = iy + h;
#if EXCLUDE_OUTSIDE
            if ((unsigned)(sy) >= (unsigned)(src_rows))
                continue;
#else
            sy = clamp(sy, 0, src_rows - 1);
#endif
            __global uchar const* S = srcptr + sy * src_step + src_offset;
            for (int w = xstart; w < xend; ++w)
            {
                int sx = ix + w;
#if EXCLUDE_OUTSIDE
                if ((unsigned)(sx) >= (unsigned)(src_cols))
                    continue;
#else
                sx = clamp(sx, 0, src_cols - 1);
#endif
                // the computation of linear's weights is trival, so do it in kernel
                float t = fabs(w - rx) * xscale;
                t = clamp(1.f - t, 0.f, 1.f);
                wline += t;
                sline += t * TO_VEC_WORK(loadpix(S + sx * pixel_size));
            }
            float u = fabs(h - ry) * yscale;
            u = clamp(1.f - u, 0.f, 1.f);
            weight += u * wline;
            sumval += u * sline;
        }
        storepix(TO_VEC_TYPE(sumval / weight), D);
#else
        W sumval = 0;
        float weight = 0;
        for (int h = ystart; h < yend; ++h)
        {
            W sline = 0;
            float wline = 0;
            int sy = iy + h;
#if EXCLUDE_OUTSIDE
            if ((unsigned)(sy) >= (unsigned)(src_rows))
                continue;
#else
            sy = clamp(sy, 0, src_rows - 1);
#endif
            __global uchar const* S = srcptr + sy * src_step + src_offset;
            for (int w = xstart; w < xend; ++w)
            {
                int sx = ix + w;
#if EXCLUDE_OUTSIDE
                if ((unsigned)(sx) >= (unsigned)(src_cols))
                    continue;
#else
                sx = clamp(sx, 0, src_cols - 1);
#endif
                float t = fabs(w - rx) * xscale;
                t = clamp(1.f - t, 0.f, 1.f);
                wline += t;
                sline += t * TO_WORK(((__global T const*)(S + sx * pixel_size))[0]);
            }
            float u = fabs(h - ry) * yscale;
            u = clamp(1.f - u, 0.f, 1.f);
            weight += u * wline;
            sumval += u * sline;
        }
        ((__global T*)(D))[0] = TO_TYPE(sumval / weight);

        for (int i = 1; i < channel; ++i)
        {
            sumval = 0;
            for (int h = ystart; h < yend; ++h)
            {
                W sline = 0;
                int sy = iy + h;
#if EXCLUDE_OUTSIDE
                if ((unsigned)(sy) >= (unsigned)(src_rows))
                    continue;
#else
                sy = clamp(sy, 0, src_rows - 1);
#endif
               __global uchar const* S = srcptr + sy * src_step + src_offset;
                for (int w = xstart; w < xend; ++w)
                {
                    int sx = ix + w;
#if EXCLUDE_OUTSIDE
                    if ((unsigned)(sx) >= (unsigned)(src_cols))
                       continue;
#else
                    sx = clamp(sx, 0, src_cols - 1);
#endif
                    float t = fabs(w - rx) * xscale;
                    t = clamp(1.f - t, 0.f, 1.f);
                    sline += t * TO_WORK(((__global T const*)(S + sx * pixel_size))[i]);
                }
                float u = fabs(h - ry) * yscale;
                u = clamp(1.f - u, 0.f, 1.f);
                sumval += u * sline;
            }
            ((__global T*)(D))[i] = TO_TYPE(sumval / weight);
        }
#endif
    }
}

#elif defined(INTER_CUBIC) && !defined(INTER_ANTIALIAS)

float cubicCoeff(float A, float A2, float A3, float x)
{
    x = fabs(x);
    if (x <= 1)
        x = (A2 * x - A3) * x * x + 1;
    else if (x <= 2)
        x = A * (((x - 5) * x + 8) * x - 4);
    else
        x = 0;
    return x;
}

__kernel void resizeOnnx_cubic(
    __global uchar const* srcptr, int src_step, int src_offset, int src_rows, int src_cols,
    __global uchar      * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
    int pixel_size, int channel, float m00, float m01, float m10, float m11, float A)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    float A2 = A + 2, A3 = A + 3;
    if (dx < dst_cols && dy < dst_rows)
    {
        float fx = fma(dx, m00, m01), fy = fma(dy, m10, m11);
        int xstart = convert_int_rtn(fx) - 1;
        int ystart = convert_int_rtn(fy) - 1;
        int xlimit = xstart + 3;
        int ylimit = ystart + 3;
        int xoffset[4];
        float xcoeff[4], xcoeffsum = 0;
        for (int x = xstart; x <= xlimit; ++x)
        {
            xoffset[x - xstart] = clamp(x, 0, src_cols - 1) * pixel_size;
            xcoeff [x - xstart] = cubicCoeff(A, A2, A3, x - fx);
#if EXCLUDE_OUTSIDE
            if ((unsigned)(x) >= (unsigned)(src_cols))
                xcoeff[x - xstart] = 0;
            xcoeffsum += xcoeff[x - xstart];
#endif
        }
        __global uchar* D = dstptr + (dy * dst_step + mad24(dx, pixel_size, dst_offset));
#if CN == 1 || CN == 2 || CN == 3 || CN == 4
        VW sum = (VW)(0);
#if EXCLUDE_OUTSIDE
        float ycoeffsum = 0;
#endif
        for (int y = ystart; y <= ylimit; ++y)
        {
#if EXCLUDE_OUTSIDE
            if ((unsigned)(y) >= (unsigned)(src_rows))
                continue;
            int yoffset = y * src_step + src_offset;
#else
            int yoffset = clamp(y, 0, src_rows - 1) * src_step + src_offset;
#endif
            VW sline = (VW)(0);
            for (int x = 0; x < 4; ++x)
                sline += (VW)(xcoeff[x]) * TO_VEC_WORK(loadpix(srcptr + yoffset + xoffset[x]));
            float u = cubicCoeff(A, A2, A3, y - fy);
#if EXCLUDE_OUTSIDE
            ycoeffsum += u;
#endif
            sum += sline * u;
        }
#if EXCLUDE_OUTSIDE
        storepix(TO_VEC_TYPE(sum / (ycoeffsum * xcoeffsum)), D);
#else
        storepix(TO_VEC_TYPE(sum), D);
#endif
#else
        int yoffset[4];
        float ycoeff[4], weight = 0;
        for (int y = ystart; y <= ylimit; ++y)
        {
            yoffset[y - ystart] = clamp(y, 0, src_rows - 1) * src_step + src_offset;
            ycoeff [y - ystart] = cubicCoeff(A, A2, A3, y - fy);
#if EXCLUDE_OUTSIDE
            if ((unsigned)(y) >= (unsigned)(src_rows))
                ycoeff[y - ystart] = 0;
            weight += ycoeff[y - ystart] * xcoeffsum;
#endif
        }
        for (int i = 0; i < channel; ++i)
        {
            W sum = 0;
            for (int y = 0; y < 4; ++y)
            {
                W sline = 0;
                for (int x = 0; x < 4; ++x)
                    sline += xcoeff[x] * TO_WORK(((__global T const*)
                                                (srcptr + yoffset[y] + xoffset[x]))[i]);
                sum += sline * ycoeff[y];
            }
#if EXCLUDE_OUTSIDE
            ((__global T*)(D))[i] = TO_TYPE(sum / weight);
#else
            ((__global T*)(D))[i] = TO_TYPE(sum);
#endif
        }
#endif
    }
}

#elif defined(INTER_CUBIC) && defined(INTER_ANTIALIAS)

// the computation of cubic's weight is heavy(?), so do it outside
// maybe it is also ok for linear antialias resize?
__kernel void resizeOnnx_table(
    __global uchar const* srcptr, int src_step, int src_offset, int src_rows, int src_cols,
    __global uchar      * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
    int pixel_size, int channel, int xkanti, int ykanti, int xstride, int ystride,
    __global int const* table)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);
    if (dx < dst_cols && dy < dst_rows)
    {
        __global uchar* D = dstptr + (dy * dst_step + mad24(dx, pixel_size, dst_offset));
        __global int const* xoffset = table;
        __global int const* yoffset = xoffset + xstride;
        __global float const* xcoeff = (__global float const*)(yoffset + ystride);
        __global float const* ycoeff = (__global float const*)(xcoeff + xstride);
#if CN == 1 || CN == 2 || CN == 3 || CN == 4
        VW sum = (VW)(0);
        // exact ykanti / xkanti loops
        for (int y = dy; y < ystride; y += dst_rows)
        {
            // offset is already clamped
            // xoffset is given by uchar, yoffset already multiply by src_step
            __global const uchar* S = srcptr + yoffset[y] + src_offset;
            VW sline = (VW)(0);
            for (int x = dx; x < xstride; x += dst_cols)
                sline += xcoeff[x] * TO_VEC_WORK(loadpix(S + xoffset[x]));
            sum += sline * ycoeff[y];
        }
        storepix(TO_VEC_TYPE(sum), D);
#else
        for (int i = 0; i < channel; ++i)
        {
            W sum = 0;
            for (int y = dy; y < ystride; y += dst_rows)
            {
                __global const uchar* S = (srcptr + yoffset[y] + src_offset);
                W sline = 0;
                for (int x = dx; x < xstride; x += dst_cols)
                    sline += xcoeff[x] * TO_WORK(((__global T const*)(S + xoffset[x]))[i]);
                sum += sline * ycoeff[y];
            }
            ((__global T*)(D))[i] = TO_TYPE(sum);
        }
#endif
    }
}

#else

#error "empty kernel"

#endif
