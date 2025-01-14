// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/********************************* COPYRIGHT NOTICE *******************************\
  The function for RGB to Lab conversion is based on the MATLAB script
  RGB2Lab.m translated by Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
  See the page [http://vision.stanford.edu/~ruzon/software/rgblab.html]
\**********************************************************************************/

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/softfloat.hpp"

#include "color.hpp"

using cv::softfloat;

static const float * splineBuild(const softfloat* f, size_t n)
{
    float* tab = cv::allocSingleton<float>(n * 4);
    const softfloat f2(2), f3(3), f4(4);
    softfloat cn(0);
    softfloat* sftab = reinterpret_cast<softfloat*>(tab);
    tab[0] = tab[1] = 0.0f;

    for(size_t i = 1; i < n; i++)
    {
        softfloat t = (f[i+1] - f[i]*f2 + f[i-1])*f3;
        softfloat l = softfloat::one()/(f4 - sftab[(i-1)*4]);
        sftab[i*4] = l; sftab[i*4+1] = (t - sftab[(i-1)*4+1])*l;
    }

    for(size_t j = 0; j < n; ++j)
    {
        size_t i = n - j - 1;
        softfloat c = sftab[i*4+1] - sftab[i*4]*cn;
        softfloat b = f[i+1] - f[i] - (cn + c*f2)/f3;
        softfloat d = (cn - c)/f3;
        sftab[i*4] = f[i]; sftab[i*4+1] = b;
        sftab[i*4+2] = c; sftab[i*4+3] = d;
        cn = c;
    }
    return tab;
}


// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<typename _Tp> static inline cv::v_float32 splineInterpolate(const cv::v_float32& x, const _Tp* tab, int n)
{
    using namespace cv;
    v_int32 ix = v_min(v_max(v_trunc(x), vx_setzero_s32()), vx_setall_s32(n-1));
    cv::v_float32 xx = v_sub(x, v_cvt_f32(ix));
    ix = v_shl<2>(ix);

    v_float32 t0, t1, t2, t3;
    // assume that VTraits<v_float32>::vlanes() == VTraits<v_int32>::vlanes()
    if(VTraits<v_float32>::vlanes() == 4)
    {
        int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) idx[4];
        v_store_aligned(idx, ix);
        v_float32 tt0, tt1, tt2, tt3;
        tt0 = vx_load(tab + idx[0]);
        tt1 = vx_load(tab + idx[1]);
        tt2 = vx_load(tab + idx[2]);
        tt3 = vx_load(tab + idx[3]);
        v_transpose4x4(tt0, tt1, tt2, tt3,
                        t0,  t1,  t2,  t3);
    }
    else
    {
        t0 = v_lut(tab + 0, ix);
        t1 = v_lut(tab + 1, ix);
        t2 = v_lut(tab + 2, ix);
        t3 = v_lut(tab + 3, ix);
    }

    return v_fma(v_fma(v_fma(t3, xx, t2), xx, t1), xx, t0);
}

#endif

namespace cv
{

////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

// 0.412453, 0.357580, 0.180423,
// 0.212671, 0.715160, 0.072169,
// 0.019334, 0.119193, 0.950227
static const softdouble sRGB2XYZ_D65[] =
{
    softdouble::fromRaw(0x3fda65a14488c60d),
    softdouble::fromRaw(0x3fd6e297396d0918),
    softdouble::fromRaw(0x3fc71819d2391d58),
    softdouble::fromRaw(0x3fcb38cda6e75ff6),
    softdouble::fromRaw(0x3fe6e297396d0918),
    softdouble::fromRaw(0x3fb279aae6c8f755),
    softdouble::fromRaw(0x3f93cc4ac6cdaf4b),
    softdouble::fromRaw(0x3fbe836eb4e98138),
    softdouble::fromRaw(0x3fee68427418d691)
};

//  3.240479, -1.53715, -0.498535,
// -0.969256, 1.875991, 0.041556,
//  0.055648, -0.204043, 1.057311
static const softdouble XYZ2sRGB_D65[] =
{
    softdouble::fromRaw(0x4009ec804102ff8f),
    softdouble::fromRaw(0xbff8982a9930be0e),
    softdouble::fromRaw(0xbfdfe7ff583a53b9),
    softdouble::fromRaw(0xbfef042528ae74f3),
    softdouble::fromRaw(0x3ffe040f23897204),
    softdouble::fromRaw(0x3fa546d3f9e7b80b),
    softdouble::fromRaw(0x3fac7de5082cf52c),
    softdouble::fromRaw(0xbfca1e14bdfd2631),
    softdouble::fromRaw(0x3ff0eabef06b3786)
};

static const int sRGB2XYZ_D65_i[] =
{
    1689,    1465,    739,
    871,     2929,    296,
    79,      488,     3892
};

static const int XYZ2sRGB_D65_i[] =
{
    13273,  -6296,  -2042,
    -3970,   7684,    170,
      228,   -836,   4331
};

template<typename _Tp> struct RGB2XYZ_f
{
    typedef _Tp channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)sRGB2XYZ_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp X = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Y = saturate_cast<_Tp>(src[0]*C3 + src[1]*C4 + src[2]*C5);
            _Tp Z = saturate_cast<_Tp>(src[0]*C6 + src[1]*C7 + src[2]*C8);
            dst[i] = X; dst[i+1] = Y; dst[i+2] = Z;
        }
    }
    int srccn;
    float coeffs[9];
};


template <>
struct RGB2XYZ_f<float>
{
    typedef float channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)sRGB2XYZ_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_float32>::vlanes();
        v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
        v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
        v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += vsize*3)
        {
            v_float32 b, g, r, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            v_float32 x, y, z;
            x = v_fma(b, vc0, v_fma(g, vc1, v_mul(r, vc2)));
            y = v_fma(b, vc3, v_fma(g, vc4, v_mul(r, vc5)));
            z = v_fma(b, vc6, v_fma(g, vc7, v_mul(r, vc8)));

            v_store_interleave(dst, x, y, z);
        }
#endif
        for( ; i < n; i++, src += scn, dst += 3)
        {
            float b = src[0], g = src[1], r = src[2];

            float X = saturate_cast<float>(b*C0 + g*C1 + r*C2);
            float Y = saturate_cast<float>(b*C3 + g*C4 + r*C5);
            float Z = saturate_cast<float>(b*C6 + g*C7 + r*C8);

            dst[0] = X; dst[1] = Y; dst[2] = Z;
        }
    }

    int srccn;
    float coeffs[9];
};


template<typename _Tp> struct RGB2XYZ_i
{
    typedef _Tp channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : sRGB2XYZ_D65_i[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for(int i = 0; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<_Tp>(X); dst[i+1] = saturate_cast<_Tp>(Y);
            dst[i+2] = saturate_cast<_Tp>(Z);
        }
    }
    int srccn;
    int coeffs[9];
};


template <>
struct RGB2XYZ_i<uchar>
{
    typedef uchar channel_type;
    static const int shift = xyz_shift;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : sRGB2XYZ_D65_i[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const uchar * src, uchar * dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int scn = srccn, i = 0;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        int descaleShift = 1 << (shift-1);
        v_int16 vdescale = vx_setall_s16((short)descaleShift);
        v_int16 cxbg, cxr1, cybg, cyr1, czbg, czr1;
        v_int16 dummy;
        v_zip(vx_setall_s16((short)C0), vx_setall_s16((short)C1), cxbg, dummy);
        v_zip(vx_setall_s16((short)C2), vx_setall_s16(        1), cxr1, dummy);
        v_zip(vx_setall_s16((short)C3), vx_setall_s16((short)C4), cybg, dummy);
        v_zip(vx_setall_s16((short)C5), vx_setall_s16(        1), cyr1, dummy);
        v_zip(vx_setall_s16((short)C6), vx_setall_s16((short)C7), czbg, dummy);
        v_zip(vx_setall_s16((short)C8), vx_setall_s16(        1), czr1, dummy);

        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_uint8 b, g, r, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            v_uint16 b0, b1, g0, g1, r0, r1;
            v_expand(b, b0, b1);
            v_expand(g, g0, g1);
            v_expand(r, r0, r1);

            v_int16 sb0, sb1, sg0, sg1, sr0, sr1;
            sr0 = v_reinterpret_as_s16(r0); sr1 = v_reinterpret_as_s16(r1);
            sg0 = v_reinterpret_as_s16(g0); sg1 = v_reinterpret_as_s16(g1);
            sb0 = v_reinterpret_as_s16(b0); sb1 = v_reinterpret_as_s16(b1);

            v_int16 bg0, bg1, bg2, bg3, rd0, rd1, rd2, rd3;
            v_zip(sb0, sg0, bg0, bg1);
            v_zip(sb1, sg1, bg2, bg3);
            v_zip(sr0, vdescale, rd0, rd1);
            v_zip(sr1, vdescale, rd2, rd3);

            v_uint32 vx0, vx1, vx2, vx3;
            v_uint32 vy0, vy1, vy2, vy3;
            v_uint32 vz0, vz1, vz2, vz3;

            vx0 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg0, cxbg), v_dotprod(rd0, cxr1))));
            vy0 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg0, cybg), v_dotprod(rd0, cyr1))));
            vz0 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg0, czbg), v_dotprod(rd0, czr1))));
            vx1 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg1, cxbg), v_dotprod(rd1, cxr1))));
            vy1 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg1, cybg), v_dotprod(rd1, cyr1))));
            vz1 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg1, czbg), v_dotprod(rd1, czr1))));
            vx2 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg2, cxbg), v_dotprod(rd2, cxr1))));
            vy2 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg2, cybg), v_dotprod(rd2, cyr1))));
            vz2 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg2, czbg), v_dotprod(rd2, czr1))));
            vx3 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg3, cxbg), v_dotprod(rd3, cxr1))));
            vy3 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg3, cybg), v_dotprod(rd3, cyr1))));
            vz3 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg3, czbg), v_dotprod(rd3, czr1))));

            v_uint16 x0, x1, y0, y1, z0, z1;
            x0 = v_pack(vx0, vx1);
            x1 = v_pack(vx2, vx3);
            y0 = v_pack(vy0, vy1);
            y1 = v_pack(vy2, vy3);
            z0 = v_pack(vz0, vz1);
            z1 = v_pack(vz2, vz3);

            v_uint8 x, y, z;
            x = v_pack(x0, x1);
            y = v_pack(y0, y1);
            z = v_pack(z0, z1);

            v_store_interleave(dst, x, y, z);
        }
#endif

        for ( ; i < n; i++, src += scn, dst += 3)
        {
            uchar b = src[0], g = src[1], r = src[2];

            int X = CV_DESCALE(b*C0 + g*C1 + r*C2, shift);
            int Y = CV_DESCALE(b*C3 + g*C4 + r*C5, shift);
            int Z = CV_DESCALE(b*C6 + g*C7 + r*C8, shift);
            dst[0] = saturate_cast<uchar>(X);
            dst[1] = saturate_cast<uchar>(Y);
            dst[2] = saturate_cast<uchar>(Z);
        }
    }

    int srccn, coeffs[9];
};


template <>
struct RGB2XYZ_i<ushort>
{
    typedef ushort channel_type;
    static const int shift = xyz_shift;
    static const int fix_shift = (int)(sizeof(short)*8 - shift);

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : sRGB2XYZ_D65_i[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }

    void operator()(const ushort * src, ushort * dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int scn = srccn, i = 0;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();
        const int descaleShift = 1 << (shift-1);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 vc0 = vx_setall_s16((short)C0), vc1 = vx_setall_s16((short)C1), vc2 = vx_setall_s16((short)C2);
        v_int16 vc3 = vx_setall_s16((short)C3), vc4 = vx_setall_s16((short)C4), vc5 = vx_setall_s16((short)C5);
        v_int16 vc6 = vx_setall_s16((short)C6), vc7 = vx_setall_s16((short)C7), vc8 = vx_setall_s16((short)C8);
        v_int16 zero = vx_setzero_s16(), one = vx_setall_s16(1);
        v_int16 cxbg, cxr1, cybg, cyr1, czbg, czr1;
        v_int16 dummy;
        v_zip(vc0, vc1, cxbg, dummy);
        v_zip(vc2, one, cxr1, dummy);
        v_zip(vc3, vc4, cybg, dummy);
        v_zip(vc5, one, cyr1, dummy);
        v_zip(vc6, vc7, czbg, dummy);
        v_zip(vc8, one, czr1, dummy);

        for (; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_uint16 b, g, r, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            v_int16 sb, sg, sr;
            sr = v_reinterpret_as_s16(r);
            sg = v_reinterpret_as_s16(g);
            sb = v_reinterpret_as_s16(b);

            // fixing 16bit signed multiplication
            v_int16 xmr, xmg, xmb;
            v_int16 ymr, ymg, ymb;
            v_int16 zmr, zmg, zmb;

            v_int16 mr = v_lt(sr, zero), mg = v_lt(sg, zero), mb = v_lt(sb, zero);

            xmb = v_and(mb, vc0);
            xmg = v_and(mg, vc1);
            xmr = v_and(mr, vc2);
            ymb = v_and(mb, vc3);
            ymg = v_and(mg, vc4);
            ymr = v_and(mr, vc5);
            zmb = v_and(mb, vc6);
            zmg = v_and(mg, vc7);
            zmr = v_and(mr, vc8);

            v_int32 xfix0, xfix1, yfix0, yfix1, zfix0, zfix1;
            v_expand(v_add(v_add(xmr, xmg), xmb), xfix0, xfix1);
            v_expand(v_add(v_add(ymr, ymg), ymb), yfix0, yfix1);
            v_expand(v_add(v_add(zmr, zmg), zmb), zfix0, zfix1);

            xfix0 = v_shl<16>(xfix0);
            xfix1 = v_shl<16>(xfix1);
            yfix0 = v_shl<16>(yfix0);
            yfix1 = v_shl<16>(yfix1);
            zfix0 = v_shl<16>(zfix0);
            zfix1 = v_shl<16>(zfix1);

            v_int16 bg0, bg1, rd0, rd1;
            v_zip(sb, sg, bg0, bg1);
            v_zip(sr, vdescale, rd0, rd1);

            v_uint32 x0, x1, y0, y1, z0, z1;

            x0 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_add(v_dotprod(bg0, cxbg), v_dotprod(rd0, cxr1)), xfix0)));
            x1 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_add(v_dotprod(bg1, cxbg), v_dotprod(rd1, cxr1)), xfix1)));
            y0 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_add(v_dotprod(bg0, cybg), v_dotprod(rd0, cyr1)), yfix0)));
            y1 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_add(v_dotprod(bg1, cybg), v_dotprod(rd1, cyr1)), yfix1)));
            z0 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_add(v_dotprod(bg0, czbg), v_dotprod(rd0, czr1)), zfix0)));
            z1 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_add(v_dotprod(bg1, czbg), v_dotprod(rd1, czr1)), zfix1)));

            v_uint16 x, y, z;
            x = v_pack(x0, x1);
            y = v_pack(y0, y1);
            z = v_pack(z0, z1);

            v_store_interleave(dst, x, y, z);
        }
#endif
        for ( ; i < n; i++, src += scn, dst += 3)
        {
            ushort b = src[0], g = src[1], r = src[2];
            int X = CV_DESCALE(b*C0 + g*C1 + r*C2, shift);
            int Y = CV_DESCALE(b*C3 + g*C4 + r*C5, shift);
            int Z = CV_DESCALE(b*C6 + g*C7 + r*C8, shift);
            dst[0] = saturate_cast<ushort>(X);
            dst[1] = saturate_cast<ushort>(Y);
            dst[2] = saturate_cast<ushort>(Z);
        }
    }

    int srccn, coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_f
{
    typedef _Tp channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)XYZ2sRGB_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp B = saturate_cast<_Tp>(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2);
            _Tp G = saturate_cast<_Tp>(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5);
            _Tp R = saturate_cast<_Tp>(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[9];
};


template <>
struct XYZ2RGB_f<float>
{
    typedef float channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)XYZ2sRGB_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int dcn = dstcn;
        float alpha = ColorChannel<float>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_float32>::vlanes();
        v_float32 valpha = vx_setall_f32(alpha);
        v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
        v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
        v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
        for( ; i <= n-vsize;
             i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_float32 x, y, z;
            v_load_deinterleave(src, x, y, z);

            v_float32 b, g, r;
            b = v_fma(x, vc0, v_fma(y, vc1, v_mul(z, vc2)));
            g = v_fma(x, vc3, v_fma(y, vc4, v_mul(z, vc5)));
            r = v_fma(x, vc6, v_fma(y, vc7, v_mul(z, vc8)));

            if(dcn == 4)
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, b, g, r);
            }
        }
#endif
        for( ; i < n; i++, src += 3, dst += dcn)
        {
            float x = src[0], y = src[1], z = src[2];
            float B = saturate_cast<float>(x*C0 + y*C1 + z*C2);
            float G = saturate_cast<float>(x*C3 + y*C4 + z*C5);
            float R = saturate_cast<float>(x*C6 + y*C7 + z*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_i
{
    typedef _Tp channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : XYZ2sRGB_D65_i[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<_Tp>(B); dst[1] = saturate_cast<_Tp>(G);
            dst[2] = saturate_cast<_Tp>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


template <>
struct XYZ2RGB_i<uchar>
{
    typedef uchar channel_type;
    static const int shift = xyz_shift;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : XYZ2sRGB_D65_i[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int dcn = dstcn, i = 0;
        uchar alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        const int descaleShift = 1 << (shift - 1);
        v_uint8 valpha = vx_setall_u8(alpha);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 cbxy, cbz1, cgxy, cgz1, crxy, crz1;
        v_int16 dummy;
        v_zip(vx_setall_s16((short)C0), vx_setall_s16((short)C1), cbxy, dummy);
        v_zip(vx_setall_s16((short)C2), vx_setall_s16(        1), cbz1, dummy);
        v_zip(vx_setall_s16((short)C3), vx_setall_s16((short)C4), cgxy, dummy);
        v_zip(vx_setall_s16((short)C5), vx_setall_s16(        1), cgz1, dummy);
        v_zip(vx_setall_s16((short)C6), vx_setall_s16((short)C7), crxy, dummy);
        v_zip(vx_setall_s16((short)C8), vx_setall_s16(        1), crz1, dummy);

        for ( ; i <= n-vsize;
              i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_uint8 x, y, z;
            v_load_deinterleave(src, x, y, z);

            v_uint16 ux0, ux1, uy0, uy1, uz0, uz1;
            v_expand(x, ux0, ux1);
            v_expand(y, uy0, uy1);
            v_expand(z, uz0, uz1);
            v_int16 x0, x1, y0, y1, z0, z1;
            x0 = v_reinterpret_as_s16(ux0);
            x1 = v_reinterpret_as_s16(ux1);
            y0 = v_reinterpret_as_s16(uy0);
            y1 = v_reinterpret_as_s16(uy1);
            z0 = v_reinterpret_as_s16(uz0);
            z1 = v_reinterpret_as_s16(uz1);

            v_int32 bb0, bb1, bb2, bb3,
                    gg0, gg1, gg2, gg3,
                    rr0, rr1, rr2, rr3;

            v_int16 xy0, xy1, xy2, xy3;
            v_int16 zd0, zd1, zd2, zd3;

            v_zip(x0, y0, xy0, xy1);
            v_zip(x1, y1, xy2, xy3);
            v_zip(z0, vdescale, zd0, zd1);
            v_zip(z1, vdescale, zd2, zd3);

            bb0 = v_shr<shift>(v_add(v_dotprod(xy0, cbxy), v_dotprod(zd0, cbz1)));
            gg0 = v_shr<shift>(v_add(v_dotprod(xy0, cgxy), v_dotprod(zd0, cgz1)));
            rr0 = v_shr<shift>(v_add(v_dotprod(xy0, crxy), v_dotprod(zd0, crz1)));
            bb1 = v_shr<shift>(v_add(v_dotprod(xy1, cbxy), v_dotprod(zd1, cbz1)));
            gg1 = v_shr<shift>(v_add(v_dotprod(xy1, cgxy), v_dotprod(zd1, cgz1)));
            rr1 = v_shr<shift>(v_add(v_dotprod(xy1, crxy), v_dotprod(zd1, crz1)));
            bb2 = v_shr<shift>(v_add(v_dotprod(xy2, cbxy), v_dotprod(zd2, cbz1)));
            gg2 = v_shr<shift>(v_add(v_dotprod(xy2, cgxy), v_dotprod(zd2, cgz1)));
            rr2 = v_shr<shift>(v_add(v_dotprod(xy2, crxy), v_dotprod(zd2, crz1)));
            bb3 = v_shr<shift>(v_add(v_dotprod(xy3, cbxy), v_dotprod(zd3, cbz1)));
            gg3 = v_shr<shift>(v_add(v_dotprod(xy3, cgxy), v_dotprod(zd3, cgz1)));
            rr3 = v_shr<shift>(v_add(v_dotprod(xy3, crxy), v_dotprod(zd3, crz1)));

            v_uint16 b0, b1, g0, g1, r0, r1;
            b0 = v_pack_u(bb0, bb1); b1 = v_pack_u(bb2, bb3);
            g0 = v_pack_u(gg0, gg1); g1 = v_pack_u(gg2, gg3);
            r0 = v_pack_u(rr0, rr1); r1 = v_pack_u(rr2, rr3);

            v_uint8 bb, gg, rr;
            bb = v_pack(b0, b1);
            gg = v_pack(g0, g1);
            rr = v_pack(r0, r1);

            if(dcn == 4)
            {
                v_store_interleave(dst, bb, gg, rr, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, bb, gg, rr);
            }
        }
#endif
        for ( ; i < n; i++, src += 3, dst += dcn)
        {
            uchar x = src[0], y = src[1], z = src[2];
            int B = CV_DESCALE(x*C0 + y*C1 + z*C2, shift);
            int G = CV_DESCALE(x*C3 + y*C4 + z*C5, shift);
            int R = CV_DESCALE(x*C6 + y*C7 + z*C8, shift);
            dst[0] = saturate_cast<uchar>(B); dst[1] = saturate_cast<uchar>(G);
            dst[2] = saturate_cast<uchar>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


template <>
struct XYZ2RGB_i<ushort>
{
    typedef ushort channel_type;
    static const int shift = xyz_shift;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : XYZ2sRGB_D65_i[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int dcn = dstcn, i = 0;
        ushort alpha = ColorChannel<ushort>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();
        const int descaleShift = 1 << (shift-1);
        v_uint16 valpha = vx_setall_u16(alpha);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 vc0 = vx_setall_s16((short)C0), vc1 = vx_setall_s16((short)C1), vc2 = vx_setall_s16((short)C2);
        v_int16 vc3 = vx_setall_s16((short)C3), vc4 = vx_setall_s16((short)C4), vc5 = vx_setall_s16((short)C5);
        v_int16 vc6 = vx_setall_s16((short)C6), vc7 = vx_setall_s16((short)C7), vc8 = vx_setall_s16((short)C8);
        v_int16 zero = vx_setzero_s16(), one = vx_setall_s16(1);
        v_int16 cbxy, cbz1, cgxy, cgz1, crxy, crz1;
        v_int16 dummy;
        v_zip(vc0, vc1, cbxy, dummy);
        v_zip(vc2, one, cbz1, dummy);
        v_zip(vc3, vc4, cgxy, dummy);
        v_zip(vc5, one, cgz1, dummy);
        v_zip(vc6, vc7, crxy, dummy);
        v_zip(vc8, one, crz1, dummy);

        for( ; i <= n-vsize;
             i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_uint16 x, y, z;
            v_load_deinterleave(src, x, y, z);

            v_int16 sx, sy, sz;
            sx = v_reinterpret_as_s16(x);
            sy = v_reinterpret_as_s16(y);
            sz = v_reinterpret_as_s16(z);

            // fixing 16bit signed multiplication
            v_int16 mx = v_lt(sx, zero), my = v_lt(sy, zero), mz = v_lt(sz, zero);

            v_int16 bmx, bmy, bmz;
            v_int16 gmx, gmy, gmz;
            v_int16 rmx, rmy, rmz;

            bmx = v_and(mx, vc0);
            bmy = v_and(my, vc1);
            bmz = v_and(mz, vc2);
            gmx = v_and(mx, vc3);
            gmy = v_and(my, vc4);
            gmz = v_and(mz, vc5);
            rmx = v_and(mx, vc6);
            rmy = v_and(my, vc7);
            rmz = v_and(mz, vc8);

            v_int32 bfix0, bfix1, gfix0, gfix1, rfix0, rfix1;
            v_expand(v_add(v_add(bmx, bmy), bmz), bfix0, bfix1);
            v_expand(v_add(v_add(gmx, gmy), gmz), gfix0, gfix1);
            v_expand(v_add(v_add(rmx, rmy), rmz), rfix0, rfix1);

            bfix0 = v_shl<16>(bfix0); bfix1 = v_shl<16>(bfix1);
            gfix0 = v_shl<16>(gfix0); gfix1 = v_shl<16>(gfix1);
            rfix0 = v_shl<16>(rfix0); rfix1 = v_shl<16>(rfix1);

            v_int16 xy0, xy1, zd0, zd1;
            v_zip(sx, sy, xy0, xy1);
            v_zip(sz, vdescale, zd0, zd1);

            v_int32 b0, b1, g0, g1, r0, r1;

            b0 = v_shr<shift>(v_add(v_add(v_dotprod(xy0, cbxy), v_dotprod(zd0, cbz1)), bfix0));
            b1 = v_shr<shift>(v_add(v_add(v_dotprod(xy1, cbxy), v_dotprod(zd1, cbz1)), bfix1));
            g0 = v_shr<shift>(v_add(v_add(v_dotprod(xy0, cgxy), v_dotprod(zd0, cgz1)), gfix0));
            g1 = v_shr<shift>(v_add(v_add(v_dotprod(xy1, cgxy), v_dotprod(zd1, cgz1)), gfix1));
            r0 = v_shr<shift>(v_add(v_add(v_dotprod(xy0, crxy), v_dotprod(zd0, crz1)), rfix0));
            r1 = v_shr<shift>(v_add(v_add(v_dotprod(xy1, crxy), v_dotprod(zd1, crz1)), rfix1));

            v_uint16 b, g, r;
            b = v_pack_u(b0, b1); g = v_pack_u(g0, g1); r = v_pack_u(r0, r1);

            if(dcn == 4)
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, b, g, r);
            }
        }
#endif
        for ( ; i < n; i++, src += 3, dst += dcn)
        {
            ushort x = src[0], y = src[1], z = src[2];
            int B = CV_DESCALE(x*C0 + y*C1 + z*C2, shift);
            int G = CV_DESCALE(x*C3 + y*C4 + z*C5, shift);
            int R = CV_DESCALE(x*C6 + y*C7 + z*C8, shift);
            dst[0] = saturate_cast<ushort>(B); dst[1] = saturate_cast<ushort>(G);
            dst[2] = saturate_cast<ushort>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

//0.950456, 1., 1.088754
static const softdouble D65[] = {softdouble::fromRaw(0x3fee6a22b3892ee8),
                                 softdouble::one(),
                                 softdouble::fromRaw(0x3ff16b8950763a19)};

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static const float *LabCbrtTab = 0;
static const float LabCbrtTabScale = softfloat(LAB_CBRT_TAB_SIZE*2)/softfloat(3);

static const float *sRGBGammaTab = 0;
static const float *sRGBInvGammaTab = 0;
static const float GammaTabScale((int)GAMMA_TAB_SIZE);

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
enum { inv_gamma_shift = 12, INV_GAMMA_TAB_SIZE = (1 << inv_gamma_shift) };
static ushort sRGBInvGammaTab_b[INV_GAMMA_TAB_SIZE], linearInvGammaTab_b[INV_GAMMA_TAB_SIZE];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static const bool enableBitExactness = true;
static const bool enableRGB2LabInterpolation = true;

#if CV_SIMD
static const bool enablePackedLab = true;
#endif

enum
{
    lab_lut_shift = 5,
    LAB_LUT_DIM = (1 << lab_lut_shift)+1,
    lab_base_shift = 14,
    LAB_BASE = (1 << lab_base_shift),
    LUT_BASE = (1 << 14),
    trilinear_shift = 8 - lab_lut_shift + 1,
    TRILINEAR_BASE = (1 << trilinear_shift)
};
static int16_t trilinearLUT[TRILINEAR_BASE*TRILINEAR_BASE*TRILINEAR_BASE*8];
static ushort LabToYF_b[256*2];
static const int minABvalue = -8145;
static const int *abToXZ_b;
// Luv constants
static const bool enableRGB2LuvInterpolation = true;

#if CV_SIMD
static const bool enablePackedRGB2Luv = true;
static const bool enablePackedLuv2RGB = true;
#endif

static const softfloat uLow(-134), uHigh(220), uRange(uHigh-uLow);
static const softfloat vLow(-140), vHigh(122), vRange(vHigh-vLow);

static struct LABLUVLUT_s16_t {
    const int16_t *RGB2LabLUT_s16;
    const int16_t *RGB2LuvLUT_s16;
} LABLUVLUTs16 = {0, 0};

static struct LUVLUT_T {
    const int *LuToUp_b;
    const int *LvToVp_b;
    const long long int *LvToVpl_b;
} LUVLUT = {0, 0, 0};

/* NB: no NaN propagation guarantee */
#define clip(value) \
    value < 0.0f ? 0.0f : value <= 1.0f ? value : 1.0f;

//all constants should be presented through integers to keep bit-exactness
static const softdouble gammaThreshold    = softdouble(809)/softdouble(20000);    //  0.04045
static const softdouble gammaInvThreshold = softdouble(7827)/softdouble(2500000); //  0.0031308
static const softdouble gammaLowScale     = softdouble(323)/softdouble(25);       // 12.92
static const softdouble gammaPower        = softdouble(12)/softdouble(5);         //  2.4
static const softdouble gammaXshift       = softdouble(11)/softdouble(200);       //  0.055

static const softfloat lthresh = softfloat(216) / softfloat(24389); // 0.008856f = (6/29)^3
static const softfloat lscale  = softfloat(841) / softfloat(108); // 7.787f = (29/3)^3/(29*4)
static const softfloat lbias = softfloat(16) / softfloat(116);
static const softfloat f255(255);


static inline softfloat applyGamma(softfloat x)
{
    //return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
    softdouble xd = x;
    return (xd <= gammaThreshold ?
                xd/gammaLowScale :
                pow((xd + gammaXshift)/(softdouble::one()+gammaXshift), gammaPower));
}


static inline softfloat applyInvGamma(softfloat x)
{
    //return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
    softdouble xd = x;
    return (xd <= gammaInvThreshold ?
                xd*gammaLowScale :
                pow(xd, softdouble::one()/gammaPower)*(softdouble::one()+gammaXshift) - gammaXshift);
}


static LUVLUT_T initLUTforLUV(const softfloat &un, const softfloat &vn)
{
    //when XYZ are limited to [0, 2]
    /*
        up: [-402, 1431.57]
        min abs diff up: 0.010407
        vp: [-0.25, 0.25]
        min abs(vp): 0.00034207
    */

    const softfloat oneof4 = softfloat::one()/softfloat(4);
    int *LuToUp_b = cv::allocSingleton<int>(256*256);
    int *LvToVp_b = cv::allocSingleton<int>(256*256);
    long long int *LvToVpl_b = cv::allocSingleton<long long int>(256*256);
    for(int LL = 0; LL < 256; LL++)
    {
        softfloat L = softfloat(LL*100)/f255;
        for(int uu = 0; uu < 256; uu++)
        {
            softfloat u = softfloat(uu)*uRange/f255 + uLow;
            softfloat up = softfloat(9)*(u + L*un);
            LuToUp_b[LL*256+uu] = cvRound(up*softfloat(LUT_BASE/1024));//1024 is OK, 2048 gave maxerr 3
        }
        for(int vv = 0; vv < 256; vv++)
        {
            softfloat v = softfloat(vv)*vRange/f255 + vLow;
            softfloat vp = oneof4/(v + L*vn);
            if(vp >  oneof4) vp =  oneof4;
            if(vp < -oneof4) vp = -oneof4;
            int ivp = cvRound(vp*softfloat(LUT_BASE*1024));
            LvToVp_b[LL*256+vv] = ivp;
            int vpl = ivp*LL;
            LvToVpl_b[LL*256+vv] = (12*13*100*(LUT_BASE/1024))*(long long)vpl;
        }
    }
    LUVLUT_T res;
    res.LuToUp_b = LuToUp_b;
    res.LvToVp_b = LvToVp_b;
    res.LvToVpl_b = LvToVpl_b;
    return res;
}


static int * initLUTforABXZ()
{
    int * res = cv::allocSingleton<int>(LAB_BASE*9/4);
    for(int i = minABvalue; i < LAB_BASE*9/4+minABvalue; i++)
    {
        int v;
        //6.f/29.f*BASE = 3389.730
        if(i <= 3390)
        {
            //fxz[k] = (fxz[k] - 16.0f / 116.0f) / 7.787f;
            // 7.787f = (29/3)^3/(29*4)
            v = i*108/841 - LUT_BASE*16/116*108/841;
        }
        else
        {
            //fxz[k] = fxz[k] * fxz[k] * fxz[k];
            v = i*i/LUT_BASE*i/LUT_BASE;
        }
        res[i-minABvalue] = v; // -1335 <= v <= 88231
    }
    return res;
}


inline void fill_one(int16_t *LAB, const int16_t *LAB_prev, int16_t *LUV, const int16_t *LUV_prev, int p, int q, int r, int _p, int _q, int _r)
{
    int idxold = 0;
    idxold += min(p+(_p), (int)(LAB_LUT_DIM-1))*3;
    idxold += min(q+(_q), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*3;
    idxold += min(r+(_r), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*LAB_LUT_DIM*3;
    int idxnew = p*3*8 + q*LAB_LUT_DIM*3*8 + r*LAB_LUT_DIM*LAB_LUT_DIM*3*8+4*(_p)+2*(_q)+(_r);
    LAB[idxnew]    = LAB_prev[idxold];
    LAB[idxnew+8]  = LAB_prev[idxold+1];
    LAB[idxnew+16] = LAB_prev[idxold+2];
    LUV[idxnew]    = LUV_prev[idxold];
    LUV[idxnew+8]  = LUV_prev[idxold+1];
    LUV[idxnew+16] = LUV_prev[idxold+2];
}


static LABLUVLUT_s16_t initLUTforLABLUVs16(const softfloat & un, const softfloat & vn)
{
    int i;
    softfloat scaledCoeffs[9], coeffs[9];

    //RGB2Lab coeffs
    softdouble scaleWhite[] = { softdouble::one()/D65[0],
                                softdouble::one(),
                                softdouble::one()/D65[2] };

    for(i = 0; i < 3; i++ )
    {
        coeffs[i*3+2] = sRGB2XYZ_D65[i*3+0];
        coeffs[i*3+1] = sRGB2XYZ_D65[i*3+1];
        coeffs[i*3+0] = sRGB2XYZ_D65[i*3+2];
        scaledCoeffs[i*3+0] = sRGB2XYZ_D65[i*3+2] * scaleWhite[i];
        scaledCoeffs[i*3+1] = sRGB2XYZ_D65[i*3+1] * scaleWhite[i];
        scaledCoeffs[i*3+2] = sRGB2XYZ_D65[i*3+0] * scaleWhite[i];
    }

    softfloat S0 = scaledCoeffs[0], S1 = scaledCoeffs[1], S2 = scaledCoeffs[2],
              S3 = scaledCoeffs[3], S4 = scaledCoeffs[4], S5 = scaledCoeffs[5],
              S6 = scaledCoeffs[6], S7 = scaledCoeffs[7], S8 = scaledCoeffs[8];
    softfloat C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

    //u, v: [-134.0, 220.0], [-140.0, 122.0]
    static const softfloat lld(LAB_LUT_DIM - 1), f116(116), f16(16), f500(500), f200(200);
    static const softfloat f100(100), f128(128), f256(256), lbase((int)LAB_BASE);
    //903.3f = (29/3)^3
    static const softfloat f9033 = softfloat(29*29*29)/softfloat(27);
    static const softfloat f9of4 = softfloat(9)/softfloat(4);
    static const softfloat f15(15), f3(3);

    AutoBuffer<int16_t> RGB2Labprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
    AutoBuffer<int16_t> RGB2Luvprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
    for(int p = 0; p < LAB_LUT_DIM; p++)
    {
        for(int q = 0; q < LAB_LUT_DIM; q++)
        {
            for(int r = 0; r < LAB_LUT_DIM; r++)
            {
                int idx = p*3 + q*LAB_LUT_DIM*3 + r*LAB_LUT_DIM*LAB_LUT_DIM*3;
                softfloat R = softfloat(p)/lld;
                softfloat G = softfloat(q)/lld;
                softfloat B = softfloat(r)/lld;

                R = applyGamma(R);
                G = applyGamma(G);
                B = applyGamma(B);

                //RGB 2 Lab LUT building
                {
                    softfloat X = R*S0 + G*S1 + B*S2;
                    softfloat Y = R*S3 + G*S4 + B*S5;
                    softfloat Z = R*S6 + G*S7 + B*S8;

                    softfloat FX = X > lthresh ? cbrt(X) : mulAdd(X, lscale, lbias);
                    softfloat FY = Y > lthresh ? cbrt(Y) : mulAdd(Y, lscale, lbias);
                    softfloat FZ = Z > lthresh ? cbrt(Z) : mulAdd(Z, lscale, lbias);

                    softfloat L = Y > lthresh ? (f116*FY - f16) : (f9033*Y);
                    softfloat a = f500 * (FX - FY);
                    softfloat b = f200 * (FY - FZ);

                    RGB2Labprev[idx]   = (int16_t)(cvRound(lbase*L/f100));
                    RGB2Labprev[idx+1] = (int16_t)(cvRound(lbase*(a + f128)/f256));
                    RGB2Labprev[idx+2] = (int16_t)(cvRound(lbase*(b + f128)/f256));
                }

                //RGB 2 Luv LUT building
                {
                    softfloat X = R*C0 + G*C1 + B*C2;
                    softfloat Y = R*C3 + G*C4 + B*C5;
                    softfloat Z = R*C6 + G*C7 + B*C8;

                    softfloat L = Y < lthresh ? mulAdd(Y, lscale, lbias) : cbrt(Y);
                    L = L*f116 - f16;

                    softfloat d = softfloat(4*13)/max(X + f15 * Y + f3 * Z, softfloat(FLT_EPSILON));
                    softfloat u = L*(X*d - un);
                    softfloat v = L*(f9of4*Y*d - vn);

                    RGB2Luvprev[idx  ] = (int16_t)cvRound(lbase*L/f100);
                    RGB2Luvprev[idx+1] = (int16_t)cvRound(lbase*(u-uLow)/uRange);
                    RGB2Luvprev[idx+2] = (int16_t)cvRound(lbase*(v-vLow)/vRange);
                }
            }
        }
    }

    int16_t *RGB2LabLUT_s16 = cv::allocSingleton<int16_t>(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8);
    int16_t *RGB2LuvLUT_s16 = cv::allocSingleton<int16_t>(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8);
    for(int p = 0; p < LAB_LUT_DIM; p++)
        for(int q = 0; q < LAB_LUT_DIM; q++)
            for(int r = 0; r < LAB_LUT_DIM; r++)
                for (int p_ = 0; p_ < 2; ++p_)
                    for (int q_ = 0; q_ < 2; ++q_)
                        for (int r_ = 0; r_ < 2; ++r_)
                            fill_one(RGB2LabLUT_s16, RGB2Labprev.data(), RGB2LuvLUT_s16, RGB2Luvprev.data(), p, q, r, p_, q_, r_);
    LABLUVLUT_s16_t res;
    res.RGB2LabLUT_s16 = RGB2LabLUT_s16;
    res.RGB2LuvLUT_s16 = RGB2LuvLUT_s16;
    return res;
}


static bool createLabTabs()
{
    softfloat f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1];
    softfloat scale = softfloat::one()/softfloat(LabCbrtTabScale);
    int i;
    for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
    {
        softfloat x = scale*softfloat(i);
        f[i] = x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x);
    }
    LabCbrtTab = splineBuild(f, LAB_CBRT_TAB_SIZE);

    scale = softfloat::one()/softfloat(GammaTabScale);
    for(i = 0; i <= GAMMA_TAB_SIZE; i++)
    {
        softfloat x = scale*softfloat(i);
        g[i] = applyGamma(x);
        ig[i] = applyInvGamma(x);
    }

    sRGBGammaTab = splineBuild(g, GAMMA_TAB_SIZE);
    sRGBInvGammaTab = splineBuild(ig, GAMMA_TAB_SIZE);

    static const softfloat intScale(255*(1 << gamma_shift));
    for(i = 0; i < 256; i++)
    {
        softfloat x = softfloat(i)/f255;
        sRGBGammaTab_b[i] = (ushort)(cvRound(intScale*applyGamma(x)));
        linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
    }
    static const softfloat invScale = softfloat::one()/softfloat((int)INV_GAMMA_TAB_SIZE);
    for(i = 0; i < INV_GAMMA_TAB_SIZE; i++)
    {
        softfloat x = invScale*softfloat(i);
        sRGBInvGammaTab_b[i] = (ushort)(cvRound(f255*applyInvGamma(x)));
        linearInvGammaTab_b[i] = (ushort)(cvTrunc(f255*x));
    }

    static const softfloat cbTabScale(softfloat::one()/(f255*(1 << gamma_shift)));
    static const softfloat lshift2(1 << lab_shift2);
    for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
    {
        softfloat x = cbTabScale*softfloat(i);
        LabCbrtTab_b[i] = (ushort)(cvRound(lshift2 * (x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x))));
    }

    //Lookup table for L to y and ify calculations
    for(i = 0; i < 256; i++)
    {
        int y, ify;
        //8 * 255.0 / 100.0 == 20.4
        if( i <= 20)
        {
            //yy = li / 903.3f;
            //y = L*100/903.3f; 903.3f = (29/3)^3, 255 = 17*3*5
            y = cvRound(softfloat(i*LUT_BASE*20*9)/softfloat(17*29*29*29));
            //fy = 7.787f * yy + 16.0f / 116.0f; 7.787f = (29/3)^3/(29*4)
            ify = cvRound(softfloat((int)LUT_BASE)*(softfloat(16)/softfloat(116) + softfloat(i*5)/softfloat(3*17*29)));
        }
        else
        {
            //fy = (li + 16.0f) / 116.0f;
            softfloat fy = (softfloat(i*100*LUT_BASE)/softfloat(255*116) +
                            softfloat(16*LUT_BASE)/softfloat(116));
            ify = cvRound(fy);
            //yy = fy * fy * fy;
            y = cvRound(fy*fy*fy/softfloat(LUT_BASE*LUT_BASE));
        }

        LabToYF_b[i*2  ] = (ushort)y;   // 0 <= y <= BASE
        LabToYF_b[i*2+1] = (ushort)ify; // 2260 <= ify <= BASE
    }

    //Lookup table for a,b to x,z conversion
    abToXZ_b = initLUTforABXZ();

    softfloat dd = D65[0] + D65[1]*softdouble(15) + D65[2]*softdouble(3);
    dd = softfloat::one()/max(dd, softfloat::eps());
    softfloat un = dd*softfloat(13*4)*D65[0];
    softfloat vn = dd*softfloat(13*9)*D65[1];

    //Luv LUT
    LUVLUT = initLUTforLUV(un, vn);

    //try to suppress warning
    static const bool calcLUT = enableRGB2LabInterpolation || enableRGB2LuvInterpolation;
    if(calcLUT)
    {

        LABLUVLUTs16 = initLUTforLABLUVs16(un, vn);

        for(int16_t p = 0; p < TRILINEAR_BASE; p++)
        {
            int16_t pp = TRILINEAR_BASE - p;
            for(int16_t q = 0; q < TRILINEAR_BASE; q++)
            {
                int16_t qq = TRILINEAR_BASE - q;
                for(int16_t r = 0; r < TRILINEAR_BASE; r++)
                {
                    int16_t rr = TRILINEAR_BASE - r;
                    int16_t* w = &trilinearLUT[8*p + 8*TRILINEAR_BASE*q + 8*TRILINEAR_BASE*TRILINEAR_BASE*r];
                    w[0]  = pp * qq * rr; w[1]  = pp * qq * r ; w[2]  = pp * q  * rr; w[3]  = pp * q  * r ;
                    w[4]  = p  * qq * rr; w[5]  = p  * qq * r ; w[6]  = p  * q  * rr; w[7]  = p  * q  * r ;
                }
            }
        }
    }
    return true;
}

static bool initLabTabs()
{
    static bool initialized = createLabTabs();
    return initialized;
}


// cx, cy, cz are in [0; LAB_BASE]
static inline void trilinearInterpolate(int cx, int cy, int cz, const int16_t* LUT,
                                        int& a, int& b, int& c)
{
    //LUT idx of origin pt of cube
    int tx = cx >> (lab_base_shift - lab_lut_shift);
    int ty = cy >> (lab_base_shift - lab_lut_shift);
    int tz = cz >> (lab_base_shift - lab_lut_shift);

    CV_DbgCheck(tx, tx >= 0 && tx < LAB_LUT_DIM, "");
    CV_DbgCheck(ty, ty >= 0 && ty < LAB_LUT_DIM, "");
    CV_DbgCheck(tz, tz >= 0 && tz < LAB_LUT_DIM, "");

    const int16_t* baseLUT = &LUT[3*8*tx + (3*8*LAB_LUT_DIM)*ty + (3*8*LAB_LUT_DIM*LAB_LUT_DIM)*tz];
    int aa[8], bb[8], cc[8];
    for(int i = 0; i < 8; i++)
    {
        aa[i] = baseLUT[i]; bb[i] = baseLUT[i+8]; cc[i] = baseLUT[i+16];
    }

    //x, y, z are [0; TRILINEAR_BASE)
    static const int bitMask = (1 << trilinear_shift) - 1;
    int x = (cx >> (lab_base_shift - 8 - 1)) & bitMask;
    int y = (cy >> (lab_base_shift - 8 - 1)) & bitMask;
    int z = (cz >> (lab_base_shift - 8 - 1)) & bitMask;

    int w[8];
    for(int i = 0; i < 8; i++)
    {
        w[i] = trilinearLUT[8*x + 8*TRILINEAR_BASE*y + 8*TRILINEAR_BASE*TRILINEAR_BASE*z + i];
    }

    a = aa[0]*w[0]+aa[1]*w[1]+aa[2]*w[2]+aa[3]*w[3]+aa[4]*w[4]+aa[5]*w[5]+aa[6]*w[6]+aa[7]*w[7];
    b = bb[0]*w[0]+bb[1]*w[1]+bb[2]*w[2]+bb[3]*w[3]+bb[4]*w[4]+bb[5]*w[5]+bb[6]*w[6]+bb[7]*w[7];
    c = cc[0]*w[0]+cc[1]*w[1]+cc[2]*w[2]+cc[3]*w[3]+cc[4]*w[4]+cc[5]*w[5]+cc[6]*w[6]+cc[7]*w[7];

    a = CV_DESCALE(a, trilinear_shift*3);
    b = CV_DESCALE(b, trilinear_shift*3);
    c = CV_DESCALE(c, trilinear_shift*3);
}

#if (CV_SIMD && CV_SIMD_WIDTH == 16)

// 8 inValues are in [0; LAB_BASE]
static inline void trilinearPackedInterpolate(const v_uint16x8& inX, const v_uint16x8& inY, const v_uint16x8& inZ,
                                              const int16_t* LUT,
                                              v_uint16x8& outA, v_uint16x8& outB, v_uint16x8& outC)
{
    //LUT idx of origin pt of cube
    v_uint16x8 idxsX = v_shr<lab_base_shift - lab_lut_shift>(inX);
    v_uint16x8 idxsY = v_shr<lab_base_shift - lab_lut_shift>(inY);
    v_uint16x8 idxsZ = v_shr<lab_base_shift - lab_lut_shift>(inZ);

    //x, y, z are [0; TRILINEAR_BASE)
    const uint16_t bitMask = (1 << trilinear_shift) - 1;
    v_uint16x8 bitMaskReg = v_setall_u16(bitMask);
    v_uint16x8 fracX = v_and(v_shr<lab_base_shift - 8 - 1>(inX), bitMaskReg);
    v_uint16x8 fracY = v_and(v_shr<lab_base_shift - 8 - 1>(inY), bitMaskReg);
    v_uint16x8 fracZ = v_and(v_shr<lab_base_shift - 8 - 1>(inZ), bitMaskReg);

    //load values to interpolate for pix0, pix1, .., pix7
    v_int16x8 a0, a1, a2, a3, a4, a5, a6, a7;
    v_int16x8 b0, b1, b2, b3, b4, b5, b6, b7;
    v_int16x8 c0, c1, c2, c3, c4, c5, c6, c7;

    v_uint32x4 addrDw0, addrDw1, addrDw10, addrDw11;
    v_mul_expand(v_setall_u16(3*8), idxsX, addrDw0, addrDw1);
    v_mul_expand(v_setall_u16(3*8*LAB_LUT_DIM), idxsY, addrDw10, addrDw11);
    addrDw0 = v_add(addrDw0, addrDw10); addrDw1 = v_add(addrDw1, addrDw11);
    v_mul_expand(v_setall_u16(3*8*LAB_LUT_DIM*LAB_LUT_DIM), idxsZ, addrDw10, addrDw11);
    addrDw0 = v_add(addrDw0, addrDw10); addrDw1 = v_add(addrDw1, addrDw11);

    uint32_t CV_DECL_ALIGNED(16) addrofs[8];
    v_store_aligned(addrofs, addrDw0);
    v_store_aligned(addrofs + 4, addrDw1);

    const int16_t* ptr;
#define LOAD_ABC(n) ptr = LUT + addrofs[n]; a##n = v_load(ptr); b##n = v_load(ptr + 8); c##n = v_load(ptr + 16)
    LOAD_ABC(0);
    LOAD_ABC(1);
    LOAD_ABC(2);
    LOAD_ABC(3);
    LOAD_ABC(4);
    LOAD_ABC(5);
    LOAD_ABC(6);
    LOAD_ABC(7);
#undef LOAD_ABC

    //interpolation weights for pix0, pix1, .., pix7
    v_int16x8 w0, w1, w2, w3, w4, w5, w6, w7;
    v_mul_expand(v_setall_u16(8), fracX, addrDw0, addrDw1);
    v_mul_expand(v_setall_u16(8*TRILINEAR_BASE), fracY, addrDw10, addrDw11);
    addrDw0 = v_add(addrDw0, addrDw10); addrDw1 = v_add(addrDw1, addrDw11);
    v_mul_expand(v_setall_u16(8*TRILINEAR_BASE*TRILINEAR_BASE), fracZ, addrDw10, addrDw11);
    addrDw0 = v_add(addrDw0, addrDw10); addrDw1 = v_add(addrDw1, addrDw11);

    v_store_aligned(addrofs, addrDw0);
    v_store_aligned(addrofs + 4, addrDw1);

#define LOAD_W(n) ptr = trilinearLUT + addrofs[n]; w##n = v_load(ptr)
    LOAD_W(0);
    LOAD_W(1);
    LOAD_W(2);
    LOAD_W(3);
    LOAD_W(4);
    LOAD_W(5);
    LOAD_W(6);
    LOAD_W(7);
#undef LOAD_W

    //outA = descale(v_reg<8>(sum(dot(ai, wi))))
    v_uint32x4 part0, part1;
#define DOT_SHIFT_PACK(l, ll) \
    part0 = v_uint32x4(v_reduce_sum(v_dotprod(l##0, w0)),\
                       v_reduce_sum(v_dotprod(l##1, w1)),\
                       v_reduce_sum(v_dotprod(l##2, w2)),\
                       v_reduce_sum(v_dotprod(l##3, w3)));\
    part1 = v_uint32x4(v_reduce_sum(v_dotprod(l##4, w4)),\
                       v_reduce_sum(v_dotprod(l##5, w5)),\
                       v_reduce_sum(v_dotprod(l##6, w6)),\
                       v_reduce_sum(v_dotprod(l##7, w7)));\
    (ll) = v_rshr_pack<trilinear_shift*3>(part0, part1)

    DOT_SHIFT_PACK(a, outA);
    DOT_SHIFT_PACK(b, outB);
    DOT_SHIFT_PACK(c, outC);

#undef DOT_SHIFT_PACK
}

#elif CV_SIMD // Fixed size v_int16x8 used below, CV_SIMD_SCALABLE is disabled.

// inValues are in [0; LAB_BASE]
static inline void trilinearPackedInterpolate(const v_uint16& inX, const v_uint16& inY, const v_uint16& inZ,
                                              const int16_t* LUT,
                                              v_uint16& outA, v_uint16& outB, v_uint16& outC)
{
    const int vsize = VTraits<v_uint16>::vlanes();
    const int vsize_max = VTraits<v_uint16>::max_nlanes;

    // LUT idx of origin pt of cube
    v_uint16 tx = v_shr<lab_base_shift - lab_lut_shift>(inX);
    v_uint16 ty = v_shr<lab_base_shift - lab_lut_shift>(inY);
    v_uint16 tz = v_shr<lab_base_shift - lab_lut_shift>(inZ);

    v_uint32 btmp00, btmp01, btmp10, btmp11, btmp20, btmp21;
    v_uint32 baseIdx0, baseIdx1;
    // baseIdx = tx*(3*8)+ty*(3*8*LAB_LUT_DIM)+tz*(3*8*LAB_LUT_DIM*LAB_LUT_DIM)
    v_mul_expand(tx, vx_setall_u16(3*8), btmp00, btmp01);
    v_mul_expand(ty, vx_setall_u16(3*8*LAB_LUT_DIM), btmp10, btmp11);
    v_mul_expand(tz, vx_setall_u16(3*8*LAB_LUT_DIM*LAB_LUT_DIM), btmp20, btmp21);
    baseIdx0 = v_add(v_add(btmp00, btmp10), btmp20);
    baseIdx1 = v_add(v_add(btmp01, btmp11), btmp21);

    uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vbaseIdx[vsize_max];
    v_store_aligned(vbaseIdx + 0*vsize/2, baseIdx0);
    v_store_aligned(vbaseIdx + 1*vsize/2, baseIdx1);

    // fracX, fracY, fracZ are [0; TRILINEAR_BASE)
    const uint16_t bitMask = (1 << trilinear_shift) - 1;
    v_uint16 bitMaskReg = vx_setall_u16(bitMask);
    v_uint16 fracX = v_and(v_shr<lab_base_shift - 8 - 1>(inX), bitMaskReg);
    v_uint16 fracY = v_and(v_shr<lab_base_shift - 8 - 1>(inY), bitMaskReg);
    v_uint16 fracZ = v_and(v_shr<lab_base_shift - 8 - 1>(inZ), bitMaskReg);

    // trilinearIdx = 8*x + 8*TRILINEAR_BASE*y + 8*TRILINEAR_BASE*TRILINEAR_BASE*z
    v_uint32 trilinearIdx0, trilinearIdx1;
    v_uint32 fracX0, fracX1, fracY0, fracY1, fracZ0, fracZ1;
    v_expand(fracX, fracX0, fracX1);
    v_expand(fracY, fracY0, fracY1);
    v_expand(fracZ, fracZ0, fracZ1);

    trilinearIdx0 = v_add(v_add(v_shl<3>(fracX0), v_shl<3 + trilinear_shift>(fracY0)), v_shl<3 + trilinear_shift * 2>(fracZ0));
    trilinearIdx1 = v_add(v_add(v_shl<3>(fracX1), v_shl<3 + trilinear_shift>(fracY1)), v_shl<3 + trilinear_shift * 2>(fracZ1));

    uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vtrilinearIdx[vsize_max];
    v_store_aligned(vtrilinearIdx + 0*vsize/2, trilinearIdx0);
    v_store_aligned(vtrilinearIdx + 1*vsize/2, trilinearIdx1);

    v_uint32 a0, a1, b0, b1, c0, c1;

    uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) va[vsize_max], vb[vsize_max], vc[vsize_max];
    for(int j = 0; j < vsize; j++)
    {
        const int16_t* baseLUT = LUT + vbaseIdx[j];

        v_int16x8 aa, bb, cc;
        aa = v_load(baseLUT);
        bb = v_load(baseLUT + 8);
        cc = v_load(baseLUT + 16);

        v_int16x8 w = v_load(trilinearLUT + vtrilinearIdx[j]);

        va[j] = v_reduce_sum(v_dotprod(aa, w));
        vb[j] = v_reduce_sum(v_dotprod(bb, w));
        vc[j] = v_reduce_sum(v_dotprod(cc, w));
    }

    a0 = vx_load_aligned(va + 0*vsize/2);
    a1 = vx_load_aligned(va + 1*vsize/2);
    b0 = vx_load_aligned(vb + 0*vsize/2);
    b1 = vx_load_aligned(vb + 1*vsize/2);
    c0 = vx_load_aligned(vc + 0*vsize/2);
    c1 = vx_load_aligned(vc + 1*vsize/2);

    // CV_DESCALE
    const v_uint32 descaleShift = vx_setall_u32(1 << (trilinear_shift*3 - 1));
    a0 = v_shr<trilinear_shift * 3>(v_add(a0, descaleShift));
    a1 = v_shr<trilinear_shift * 3>(v_add(a1, descaleShift));
    b0 = v_shr<trilinear_shift * 3>(v_add(b0, descaleShift));
    b1 = v_shr<trilinear_shift * 3>(v_add(b1, descaleShift));
    c0 = v_shr<trilinear_shift * 3>(v_add(c0, descaleShift));
    c1 = v_shr<trilinear_shift * 3>(v_add(c1, descaleShift));

    outA = v_pack(a0, a1); outB = v_pack(b0, b1); outC = v_pack(c0, c1);
}

#endif // CV_SIMD




struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble(_whitept[i]);
            else
                whitePt[i] = D65[i];

        static const softdouble lshift(1 << lab_shift);
        for( int i = 0; i < 3; i++ )
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i*3+j]);
                else
                    c[j] = sRGB2XYZ_D65[i*3+j];
            coeffs[i*3+(blueIdx^2)] = cvRound(lshift*c[0]/whitePt[i]);
            coeffs[i*3+1]           = cvRound(lshift*c[1]/whitePt[i]);
            coeffs[i*3+blueIdx]     = cvRound(lshift*c[2]/whitePt[i]);

            CV_Assert(coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift));
        }
    }

#if CV_NEON
    template <int n>
    inline void rgb2lab_batch(const ushort* tab,
                              const v_uint8 vRi, const v_uint8 vGi, const v_uint8 vBi,
                              v_int32& vL, v_int32& va, v_int32& vb) const
    {
        // Define some scalar constants which we will make use of later
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const int xyzDescaleShift = (1 << (lab_shift - 1));
        const int labDescaleShift = (1 << (lab_shift2 - 1));
        const int abShift = 128*(1 << lab_shift2);

        const int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                  C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                  C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        // int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
        v_int32 vR(tab[v_extract_n<4*n+0>(vRi)], tab[v_extract_n<4*n+1>(vRi)],
                   tab[v_extract_n<4*n+2>(vRi)], tab[v_extract_n<4*n+3>(vRi)]);
        v_int32 vG(tab[v_extract_n<4*n+0>(vGi)], tab[v_extract_n<4*n+1>(vGi)],
                   tab[v_extract_n<4*n+2>(vGi)], tab[v_extract_n<4*n+3>(vGi)]);
        v_int32 vB(tab[v_extract_n<4*n+0>(vBi)], tab[v_extract_n<4*n+1>(vBi)],
                   tab[v_extract_n<4*n+2>(vBi)], tab[v_extract_n<4*n+3>(vBi)]);

        /* int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];*/
        v_int32 vfX = v_fma(vR, v_setall_s32(C0), v_setall_s32(xyzDescaleShift));
        vfX = v_fma(vG, v_setall_s32(C1), vfX);
        vfX = v_fma(vB, v_setall_s32(C2), vfX);
        vfX = v_shr<lab_shift>(vfX);
        vfX = v_int32(LabCbrtTab_b[v_extract_n<0>(vfX)], LabCbrtTab_b[v_extract_n<1>(vfX)],
                      LabCbrtTab_b[v_extract_n<2>(vfX)], LabCbrtTab_b[v_extract_n<3>(vfX)]);

        /* int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)]; */
        v_int32 vfY = v_fma(vR, v_setall_s32(C3), v_setall_s32(xyzDescaleShift));
        vfY = v_fma(vG, v_setall_s32(C4), vfY);
        vfY = v_fma(vB, v_setall_s32(C5), vfY);
        vfY = v_shr<lab_shift>(vfY);
        vfY = v_int32(LabCbrtTab_b[v_extract_n<0>(vfY)], LabCbrtTab_b[v_extract_n<1>(vfY)],
                      LabCbrtTab_b[v_extract_n<2>(vfY)], LabCbrtTab_b[v_extract_n<3>(vfY)]);

        /* int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];*/
        v_int32 vfZ = v_fma(vR, v_setall_s32(C6), v_setall_s32(xyzDescaleShift));
        vfZ = v_fma(vG, v_setall_s32(C7), vfZ);
        vfZ = v_fma(vB, v_setall_s32(C8), vfZ);
        vfZ = v_shr<lab_shift>(vfZ);
        vfZ = v_int32(LabCbrtTab_b[v_extract_n<0>(vfZ)], LabCbrtTab_b[v_extract_n<1>(vfZ)],
                      LabCbrtTab_b[v_extract_n<2>(vfZ)], LabCbrtTab_b[v_extract_n<3>(vfZ)]);

        /* int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );*/
        vL = v_fma(vfY, v_setall_s32(Lscale), v_setall_s32(Lshift+labDescaleShift));
        vL = v_shr<lab_shift2>(vL);

        /* int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );*/
        va = v_fma(v_sub(vfX, vfY), v_setall_s32(500), v_setall_s32(abShift+labDescaleShift));
        va = v_shr<lab_shift2>(va);

        /* int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );*/
        vb = v_fma(v_sub(vfY, vfZ), v_setall_s32(200), v_setall_s32(abShift+labDescaleShift));
        vb = v_shr<lab_shift2>(vb);
    }
#endif // CV_NEON

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        i = 0;

#if CV_NEON
        // On each loop, we load nlanes of RGB/A v_uint8s and store nlanes of
        // Lab v_uint8s
        for(; i <= n - VTraits<v_uint8>::vlanes(); i += VTraits<v_uint8>::vlanes(),
                src += scn*VTraits<v_uint8>::vlanes(), dst += 3*VTraits<v_uint8>::vlanes() )
        {
            // Load 4 batches of 4 src
            v_uint8 vRi, vGi, vBi;
            if(scn == 4)
            {
                v_uint8 vAi;
                v_load_deinterleave(src, vRi, vGi, vBi, vAi);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, vRi, vGi, vBi);
            }

            // Do 4 batches of 4 RGB2Labs
            v_int32 vL0, va0, vb0;
            rgb2lab_batch<0>(tab, vRi, vGi, vBi, vL0, va0, vb0);
            v_int32 vL1, va1, vb1;
            rgb2lab_batch<1>(tab, vRi, vGi, vBi, vL1, va1, vb1);
            v_int32 vL2, va2, vb2;
            rgb2lab_batch<2>(tab, vRi, vGi, vBi, vL2, va2, vb2);
            v_int32 vL3, va3, vb3;
            rgb2lab_batch<3>(tab, vRi, vGi, vBi, vL3, va3, vb3);

            // Saturate, combine and store all batches
            // dst[0] = saturate_cast<uchar>(L);
            // dst[1] = saturate_cast<uchar>(a);
            // dst[2] = saturate_cast<uchar>(b);
            v_store_interleave(dst,
                v_pack(v_pack_u(vL0, vL1), v_pack_u(vL2, vL3)),
                v_pack(v_pack_u(va0, va1), v_pack_u(va2, va3)),
                v_pack(v_pack_u(vb0, vb1), v_pack_u(vb2, vb3)));
        }
#endif // CV_NEON

#if CV_SIMD
        const int vsize = VTraits<v_uint8>::vlanes();
        const int xyzDescaleShift = 1 << (lab_shift - 1);
        v_int16 vXYZdescale = vx_setall_s16(xyzDescaleShift);
        v_int16 cxrg, cxb1, cyrg, cyb1, czrg, czb1;
        v_int16 dummy;
        v_zip(vx_setall_s16((short)C0), vx_setall_s16((short)C1), cxrg, dummy);
        v_zip(vx_setall_s16((short)C2), vx_setall_s16(        1), cxb1, dummy);
        v_zip(vx_setall_s16((short)C3), vx_setall_s16((short)C4), cyrg, dummy);
        v_zip(vx_setall_s16((short)C5), vx_setall_s16(        1), cyb1, dummy);
        v_zip(vx_setall_s16((short)C6), vx_setall_s16((short)C7), czrg, dummy);
        v_zip(vx_setall_s16((short)C8), vx_setall_s16(        1), czb1, dummy);
        const int labDescaleShift = 1 << (lab_shift2 - 1);

        for( ; i <= n - vsize;
             i += vsize , src += scn*vsize, dst += 3*vsize)
        {
            v_uint8 R, G, B, A;
            if(scn == 4)
            {
                v_load_deinterleave(src, R, G, B, A);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, R, G, B);
            }

            // gamma substitution using tab
            v_uint16 drgb[6];
            // [0 1 2 3 4 5 6] => [R0 R1 G0 G1 B0 B1]
            v_expand(R, drgb[0], drgb[1]);
            v_expand(G, drgb[2], drgb[3]);
            v_expand(B, drgb[4], drgb[5]);

            // [0 1 2 3 4 5 6 7 8 9 10 11 12] => [4 per R, 4 per G, 4 per B]
            v_uint32 qrgb[12];
            for(int k = 0; k < 6; k++)
            {
                v_expand(drgb[k], qrgb[k*2+0], qrgb[k*2+1]);
            }

            uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vdrgb[VTraits<v_uint8>::max_nlanes*3];
            for(int k = 0; k < 12; k++)
            {
                v_store_aligned(vdrgb + k*vsize/4, qrgb[k]);
            }

            v_uint16 trgb[6];
            for(int k = 0; k < 6; k++)
            {
                trgb[k] = vx_lut(tab, (const int*)vdrgb + k*vsize/2);
            }

            v_int16 rgbs[6];
            for(int k = 0; k < 6; k++)
            {
                rgbs[k] = v_reinterpret_as_s16(trgb[k]);
            }
            v_int16 sB0, sB1, sG0, sG1, sR0, sR1;
            sR0 = rgbs[0]; sR1 = rgbs[1];
            sG0 = rgbs[2]; sG1 = rgbs[3];
            sB0 = rgbs[4]; sB1 = rgbs[5];

            v_int16 rg[4], bd[4];
            v_zip(sR0, sG0, rg[0], rg[1]);
            v_zip(sR1, sG1, rg[2], rg[3]);
            v_zip(sB0, vXYZdescale, bd[0], bd[1]);
            v_zip(sB1, vXYZdescale, bd[2], bd[3]);

            // [X, Y, Z] = CV_DESCALE(R*C_ + G*C_ + B*C_, lab_shift)
            v_uint32 x[4], y[4], z[4];
            for(int j = 0; j < 4; j++)
            {
                x[j] = v_shr<xyz_shift>(v_reinterpret_as_u32(v_add(v_dotprod(rg[j], cxrg), v_dotprod(bd[j], cxb1))));
                y[j] = v_shr<xyz_shift>(v_reinterpret_as_u32(v_add(v_dotprod(rg[j], cyrg), v_dotprod(bd[j], cyb1))));
                z[j] = v_shr<xyz_shift>(v_reinterpret_as_u32(v_add(v_dotprod(rg[j], czrg), v_dotprod(bd[j], czb1))));
            }

            // [fX, fY, fZ] = LabCbrtTab_b[vx, vy, vz]
            // [4 per X, 4 per Y, 4 per Z]
            uint32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vxyz[VTraits<v_uint8>::max_nlanes*3];
            for(int j = 0; j < 4; j++)
            {
                v_store_aligned(vxyz + (0*4+j)*vsize/4, x[j]);
                v_store_aligned(vxyz + (1*4+j)*vsize/4, y[j]);
                v_store_aligned(vxyz + (2*4+j)*vsize/4, z[j]);
            }
            // [X0, X1, Y0, Y1, Z0, Z1]
            v_uint16 fxyz[2*3];
            for(int j = 0; j < 2*3; j++)
            {
                fxyz[j] = vx_lut(LabCbrtTab_b, (const int*)vxyz + j*vsize/2);
            }

            v_int16 fX0, fX1, fY0, fY1, fZ0, fZ1;
            fX0 = v_reinterpret_as_s16(fxyz[0]), fX1 = v_reinterpret_as_s16(fxyz[1]);
            fY0 = v_reinterpret_as_s16(fxyz[2]), fY1 = v_reinterpret_as_s16(fxyz[3]);
            fZ0 = v_reinterpret_as_s16(fxyz[4]), fZ1 = v_reinterpret_as_s16(fxyz[5]);

            v_uint16 Ldiff0 = fxyz[2], Ldiff1 = fxyz[3];

            v_uint8 L, a, b;

            // L = (Lscale*Ldiff + (Lshift + labDescaleShift)) >> lab_shift2;
            v_uint32 vL[4];
            v_uint16 vLscale = vx_setall_u16(Lscale);
            v_mul_expand(Ldiff0, vLscale, vL[0], vL[1]);
            v_mul_expand(Ldiff1, vLscale, vL[2], vL[3]);
            v_uint32 vLshift = vx_setall_u32((uint32_t)(Lshift + labDescaleShift));
            for(int k = 0; k < 4; k++)
            {
                vL[k] = v_shr<lab_shift2>(v_add(vL[k], vLshift));
            }
            v_uint16 L0, L1;
            L0 = v_pack(vL[0], vL[1]);
            L1 = v_pack(vL[2], vL[3]);

            L = v_pack(L0, L1);

            // a = (500*(fX - fY) + (128*(1 << lab_shift2) + labDescaleShift)) >> lab_shift2;
            // b = (200*(fY - fZ) + (128*(1 << lab_shift2) + labDescaleShift)) >> lab_shift2;
            v_int16 adiff0 = v_sub_wrap(fX0, fY0), adiff1 = v_sub_wrap(fX1, fY1);
            v_int16 bdiff0 = v_sub_wrap(fY0, fZ0), bdiff1 = v_sub_wrap(fY1, fZ1);

            // [4 for a, 4 for b]
            v_int32 ab[8];
            v_int16 v500 = vx_setall_s16(500);
            v_mul_expand(adiff0, v500, ab[0], ab[1]);
            v_mul_expand(adiff1, v500, ab[2], ab[3]);
            v_int16 v200 = vx_setall_s16(200);
            v_mul_expand(bdiff0, v200, ab[4], ab[5]);
            v_mul_expand(bdiff1, v200, ab[6], ab[7]);
            v_int32 abShift = vx_setall_s32(128*(1 << lab_shift2) + labDescaleShift);
            for(int k = 0; k < 8; k++)
            {
                ab[k] = v_shr<lab_shift2>(v_add(ab[k], abShift));
            }
            v_int16 a0, a1, b0, b1;
            a0 = v_pack(ab[0], ab[1]); a1 = v_pack(ab[2], ab[3]);
            b0 = v_pack(ab[4], ab[5]); b1 = v_pack(ab[6], ab[7]);

            a = v_pack_u(a0, a1);
            b = v_pack_u(b0, b1);

            v_store_interleave(dst, L, a, b);
        }
#endif

        for(; i < n; i++, src += scn, dst += 3 )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

            int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
            int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
            int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

            dst[0] = saturate_cast<uchar>(L);
            dst[1] = saturate_cast<uchar>(a);
            dst[2] = saturate_cast<uchar>(b);
        }
    }

    int srccn;
    int coeffs[9];
    bool srgb;
};


struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb), blueIdx(_blueIdx)
    {
        initLabTabs();

        useInterpolation = (!_coeffs && !_whitept && srgb && enableRGB2LabInterpolation);

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble((double)_whitept[i]);
            else
                whitePt[i] = D65[i];

        softdouble scale[] = { softdouble::one() / whitePt[0],
                               softdouble::one(),
                               softdouble::one() / whitePt[2] };

        for( int i = 0; i < 3; i++ )
        {
            softfloat c[3];
            for(int k = 0; k < 3; k++)
                if(_coeffs)
                    c[k] = scale[i] * softdouble((double)_coeffs[i*3 + k]);
                else
                    c[k] = scale[i] * sRGB2XYZ_D65[i*3 + k];
            coeffs[i*3 + (blueIdx ^ 2)] = c[0];
            coeffs[i*3 + 1]             = c[1];
            coeffs[i*3 + blueIdx]       = c[2];

            CV_Assert( c[0] >= 0 && c[1] >= 0 && c[2] >= 0 &&
                       c[0] + c[1] + c[2] < softfloat((int)LAB_CBRT_TAB_SIZE) );
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int scn = srccn, bIdx = blueIdx;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        if(useInterpolation)
        {
            int i = 0;
            n *= 3;

#if CV_SIMD
            if(enablePackedLab)
            {
                const int vsize = VTraits<v_float32>::vlanes();
                static const int nPixels = vsize*2;
                for(; i < n - 3*nPixels; i += 3*nPixels, src += scn*nPixels)
                {
                    v_float32 rvec0, gvec0, bvec0, rvec1, gvec1, bvec1;
                    if(scn == 3)
                    {
                        v_load_deinterleave(src + 0*vsize, rvec0, gvec0, bvec0);
                        v_load_deinterleave(src + 3*vsize, rvec1, gvec1, bvec1);
                    }
                    else // scn == 4
                    {
                        v_float32 dummy0, dummy1;
                        v_load_deinterleave(src + 0*vsize, rvec0, gvec0, bvec0, dummy0);
                        v_load_deinterleave(src + 4*vsize, rvec1, gvec1, bvec1, dummy1);
                    }

                    if(bIdx)
                    {
                        swap(rvec0, bvec0);
                        swap(rvec1, bvec1);
                    }

                    v_float32 zerof = vx_setzero_f32(), onef = vx_setall_f32(1.0f);
                    /* clip() */
                    #define clipv(r) (r) = v_min(v_max((r), zerof), onef)
                    clipv(rvec0); clipv(rvec1);
                    clipv(gvec0); clipv(gvec1);
                    clipv(bvec0); clipv(bvec1);
                    #undef clipv
                    /* int iR = R*LAB_BASE, iG = G*LAB_BASE, iB = B*LAB_BASE, iL, ia, ib; */
                    v_float32 basef = vx_setall_f32(static_cast<float>(LAB_BASE));
                    rvec0 = v_mul(rvec0, basef), gvec0 = v_mul(gvec0, basef), bvec0 = v_mul(bvec0, basef);
                    rvec1 = v_mul(rvec1, basef), gvec1 = v_mul(gvec1, basef), bvec1 = v_mul(bvec1, basef);

                    v_int32 irvec0, igvec0, ibvec0, irvec1, igvec1, ibvec1;
                    irvec0 = v_round(rvec0); irvec1 = v_round(rvec1);
                    igvec0 = v_round(gvec0); igvec1 = v_round(gvec1);
                    ibvec0 = v_round(bvec0); ibvec1 = v_round(bvec1);

                    v_uint16 uirvec = v_pack_u(irvec0, irvec1);
                    v_uint16 uigvec = v_pack_u(igvec0, igvec1);
                    v_uint16 uibvec = v_pack_u(ibvec0, ibvec1);

                    v_uint16 ui_lvec, ui_avec, ui_bvec;
                    trilinearPackedInterpolate(uirvec, uigvec, uibvec, LABLUVLUTs16.RGB2LabLUT_s16, ui_lvec, ui_avec, ui_bvec);
                    v_int16 i_lvec = v_reinterpret_as_s16(ui_lvec);
                    v_int16 i_avec = v_reinterpret_as_s16(ui_avec);
                    v_int16 i_bvec = v_reinterpret_as_s16(ui_bvec);

                    /* float L = iL*1.0f/LAB_BASE, a = ia*1.0f/LAB_BASE, b = ib*1.0f/LAB_BASE; */
                    v_int32 i_lvec0, i_avec0, i_bvec0, i_lvec1, i_avec1, i_bvec1;
                    v_expand(i_lvec, i_lvec0, i_lvec1);
                    v_expand(i_avec, i_avec0, i_avec1);
                    v_expand(i_bvec, i_bvec0, i_bvec1);

                    v_float32 l_vec0, a_vec0, b_vec0, l_vec1, a_vec1, b_vec1;
                    l_vec0 = v_cvt_f32(i_lvec0); l_vec1 = v_cvt_f32(i_lvec1);
                    a_vec0 = v_cvt_f32(i_avec0); a_vec1 = v_cvt_f32(i_avec1);
                    b_vec0 = v_cvt_f32(i_bvec0); b_vec1 = v_cvt_f32(i_bvec1);

                    /* dst[i] = L*100.0f */
                    v_float32 v100dBase = vx_setall_f32(100.0f / static_cast<float>(LAB_BASE));
                    l_vec0 = v_mul(l_vec0, v100dBase);
                    l_vec1 = v_mul(l_vec1, v100dBase);
                    /*
                    dst[i + 1] = a*256.0f - 128.0f;
                    dst[i + 2] = b*256.0f - 128.0f;
                    */
                    v_float32 v256dBase = vx_setall_f32(256.0f / static_cast<float>(LAB_BASE)), vm128 = vx_setall_f32(-128.f);
                    a_vec0 = v_fma(a_vec0, v256dBase, vm128);
                    a_vec1 = v_fma(a_vec1, v256dBase, vm128);
                    b_vec0 = v_fma(b_vec0, v256dBase, vm128);
                    b_vec1 = v_fma(b_vec1, v256dBase, vm128);

                    v_store_interleave(dst + i + 0*vsize, l_vec0, a_vec0, b_vec0);
                    v_store_interleave(dst + i + 3*vsize, l_vec1, a_vec1, b_vec1);
                }
            }
#endif // CV_SIMD

            for(; i < n; i += 3, src += scn)
            {
                float R = clip(src[bIdx]);
                float G = clip(src[1]);
                float B = clip(src[bIdx^2]);

                int iR = cvRound(R*static_cast<float>(LAB_BASE)), iG = cvRound(G*static_cast<float>(LAB_BASE)), iB = cvRound(B*static_cast<float>(LAB_BASE));
                int iL, ia, ib;
                trilinearInterpolate(iR, iG, iB, LABLUVLUTs16.RGB2LabLUT_s16, iL, ia, ib);
                float L = iL*1.0f/static_cast<float>(LAB_BASE), a = ia*1.0f/static_cast<float>(LAB_BASE), b = ib*1.0f/static_cast<float>(LAB_BASE);

                dst[i] = L*100.0f;
                dst[i + 1] = a*256.0f - 128.0f;
                dst[i + 2] = b*256.0f - 128.0f;
            }
        }
        else
        {
            static const float _a = (softfloat(16) / softfloat(116));
            int i = 0;
#if CV_SIMD
            const int vsize = VTraits<v_float32>::vlanes();
            const int nrepeats = VTraits<v_float32>::nlanes == 4 ? 2 : 1;
            v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
            v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
            v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
            for( ; i <= n - vsize*nrepeats;
                 i += vsize*nrepeats, src += scn*vsize*nrepeats, dst += 3*vsize*nrepeats)
            {
                v_float32 R[nrepeats], G[nrepeats], B[nrepeats], A;
                if(scn == 4)
                {
                    for (int k = 0; k < nrepeats; k++)
                    {
                        v_load_deinterleave(src + k*4*vsize, R[k], G[k], B[k], A);
                    }
                }
                else // scn == 3
                {
                    for (int k = 0; k < nrepeats; k++)
                    {
                        v_load_deinterleave(src + k*3*vsize, R[k], G[k], B[k]);
                    }
                }

                v_float32 one = vx_setall_f32(1.0f), z = vx_setzero_f32();
                for (int k = 0; k < nrepeats; k++)
                {
                    R[k] = v_max(z, v_min(R[k], one));
                    G[k] = v_max(z, v_min(G[k], one));
                    B[k] = v_max(z, v_min(B[k], one));
                }

                if(gammaTab)
                {
                    v_float32 vgscale = vx_setall_f32(gscale);
                    for (int k = 0; k < nrepeats; k++)
                    {
                        R[k] = splineInterpolate(v_mul(R[k], vgscale), gammaTab, GAMMA_TAB_SIZE);
                        G[k] = splineInterpolate(v_mul(G[k], vgscale), gammaTab, GAMMA_TAB_SIZE);
                        B[k] = splineInterpolate(v_mul(B[k], vgscale), gammaTab, GAMMA_TAB_SIZE);
                    }
                }

                v_float32 X[nrepeats], Y[nrepeats], Z[nrepeats];
                v_float32 FX[nrepeats], FY[nrepeats], FZ[nrepeats];
                for (int k = 0; k < nrepeats; k++)
                {
                    X[k] = v_fma(R[k], vc0, v_fma(G[k], vc1, v_mul(B[k], vc2)));
                    Y[k] = v_fma(R[k], vc3, v_fma(G[k], vc4, v_mul(B[k], vc5)));
                    Z[k] = v_fma(R[k], vc6, v_fma(G[k], vc7, v_mul(B[k], vc8)));

                    // use spline interpolation instead of direct calculation
                    v_float32 vTabScale = vx_setall_f32(LabCbrtTabScale);
                    FX[k] = splineInterpolate(v_mul(X[k], vTabScale), LabCbrtTab, LAB_CBRT_TAB_SIZE);
                    FY[k] = splineInterpolate(v_mul(Y[k], vTabScale), LabCbrtTab, LAB_CBRT_TAB_SIZE);
                    FZ[k] = splineInterpolate(v_mul(Z[k], vTabScale), LabCbrtTab, LAB_CBRT_TAB_SIZE);
                }

                v_float32 L[nrepeats], a[nrepeats], b[nrepeats];
                for (int k = 0; k < nrepeats; k++)
                {
                    // 7.787f = (29/3)^3/(29*4), 0.008856f = (6/29)^3, 903.3 = (29/3)^3
                    v_float32 mask = v_gt(Y[k], (vx_setall_f32(0.008856f)));
                    v_float32 v116 = vx_setall_f32(116.f), vm16 = vx_setall_f32(-16.f);
                    L[k] = v_select(mask, v_fma(v116, FY[k], vm16), v_mul(vx_setall_f32(903.3f),Y[k]));
                    a[k] = v_mul(vx_setall_f32(500.F), v_sub(FX[k], FY[k]));
                    b[k] = v_mul(vx_setall_f32(200.F), v_sub(FY[k], FZ[k]));

                    v_store_interleave(dst + k*3*vsize, L[k], a[k], b[k]);
                }
            }
#endif

            for (; i < n; i++, src += scn, dst += 3 )
            {
                float R = clip(src[0]);
                float G = clip(src[1]);
                float B = clip(src[2]);

                if (gammaTab)
                {
                    R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                    G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                    B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
                }
                float X = R*C0 + G*C1 + B*C2;
                float Y = R*C3 + G*C4 + B*C5;
                float Z = R*C6 + G*C7 + B*C8;
                // 7.787f = (29/3)^3/(29*4), 0.008856f = (6/29)^3, 903.3 = (29/3)^3
                float FX = X > 0.008856f ? cubeRoot(X) : (7.787f * X + _a);
                float FY = Y > 0.008856f ? cubeRoot(Y) : (7.787f * Y + _a);
                float FZ = Z > 0.008856f ? cubeRoot(Z) : (7.787f * Z + _a);

                float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
                float a = 500.f * (FX - FY);
                float b = 200.f * (FY - FZ);

                dst[0] = L;
                dst[1] = a;
                dst[2] = b;
            }
        }
    }

    int srccn;
    float coeffs[9];
    bool srgb;
    bool useInterpolation;
    int blueIdx;
};


// Performs conversion in floats
struct Lab2RGBfloat
{
    typedef float channel_type;

    Lab2RGBfloat( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb), blueIdx(_blueIdx)
    {
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble((double)_whitept[i]);
            else
                whitePt[i] = D65[i];

        for( int i = 0; i < 3; i++ )
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i+j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i+j*3];

            coeffs[i+(blueIdx^2)*3] = (float)(c[0]*whitePt[i]);
            coeffs[i+3]             = (float)(c[1]*whitePt[i]);
            coeffs[i+blueIdx*3]     = (float)(c[2]*whitePt[i]);
        }

        lThresh = softfloat(8); // 0.008856f * 903.3f  = (6/29)^3*(29/3)^3 = 8
        fThresh = softfloat(6)/softfloat(29); // 7.787f * 0.008856f + 16.0f / 116.0f = 6/29
    }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();

#if CV_SIMD
        const int vsize = VTraits<v_float32>::vlanes();
        const int nrepeats = 2;
        v_float32 v16_116 = vx_setall_f32(16.0f / 116.0f);
        for( ; i <= n-vsize*nrepeats;
               i += vsize*nrepeats, src += 3*vsize*nrepeats, dst += dcn*vsize*nrepeats)
        {
            v_float32 li[nrepeats], ai[nrepeats], bi[nrepeats];
            for(int k = 0; k < nrepeats; k++)
            {
                v_load_deinterleave(src + k*3*vsize, li[k], ai[k], bi[k]);
            }

            v_float32 x[nrepeats], y[nrepeats], z[nrepeats], fy[nrepeats];
            v_float32 limask[nrepeats];
            v_float32 vlThresh = vx_setall_f32(lThresh);
            for(int k = 0; k < nrepeats; k++)
            {
                limask[k] = v_le(li[k], vlThresh);
            }
            v_float32 ylo[nrepeats], yhi[nrepeats], fylo[nrepeats], fyhi[nrepeats];
            // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
            v_float32 vinv903 = vx_setall_f32(1.f/903.3f);
            for(int k = 0; k < nrepeats; k++)
            {
                ylo[k] = v_mul(li[k], vinv903);
            }
            v_float32 v7787 = vx_setall_f32(7.787f);
            for(int k = 0; k < nrepeats; k++)
            {
                fylo[k] = v_fma(v7787, ylo[k], v16_116);
            }
            v_float32 v16 = vx_setall_f32(16.0f), vinv116 = vx_setall_f32(1.f/116.0f);
            for(int k = 0; k < nrepeats; k++)
            {
                fyhi[k] = v_mul(v_add(li[k], v16), vinv116);
            }
            for(int k = 0; k < nrepeats; k++)
            {
                yhi[k] = v_mul(fyhi[k], fyhi[k], fyhi[k]);
            }
            for(int k = 0; k < nrepeats; k++)
            {
                y[k]  = v_select(limask[k], ylo[k],  yhi[k]);
                fy[k] = v_select(limask[k], fylo[k], fyhi[k]);
            }

            v_float32 fxz[nrepeats*2];
            v_float32 vpinv500 = vx_setall_f32( 1.f/500.f);
            v_float32 vninv200 = vx_setall_f32(-1.f/200.f);
            for(int k = 0; k < nrepeats; k++)
            {
                fxz[k*2+0] = v_fma(ai[k], vpinv500, fy[k]);
                fxz[k*2+1] = v_fma(bi[k], vninv200, fy[k]);
            }
            v_float32 vfTresh = vx_setall_f32(fThresh);
            v_float32 vinv7787 = vx_setall_f32(1.f/7.787f);
            for(int k = 0; k < nrepeats; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    v_float32 f = fxz[k*2+j];
                    v_float32 fmask = v_le(f, vfTresh);
                    v_float32 flo = v_mul(v_sub(f, v16_116), vinv7787);
                    v_float32 fhi = v_mul(v_mul(f, f), f);
                    fxz[k*2+j] = v_select(fmask, flo, fhi);
                }
            }
            for(int k = 0; k < nrepeats; k++)
            {
                x[k] = fxz[k*2+0], z[k] = fxz[k*2+1];
            }
            v_float32 ro[nrepeats], go[nrepeats], bo[nrepeats];
            v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
            v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
            v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
            for(int k = 0; k < nrepeats; k++)
            {
                ro[k] = v_fma(vc0, x[k], v_fma(vc1, y[k], v_mul(vc2, z[k])));
                go[k] = v_fma(vc3, x[k], v_fma(vc4, y[k], v_mul(vc5, z[k])));
                bo[k] = v_fma(vc6, x[k], v_fma(vc7, y[k], v_mul(vc8, z[k])));
            }
            v_float32 one = vx_setall_f32(1.f), zero = vx_setzero_f32();
            for(int k = 0; k < nrepeats; k++)
            {
                ro[k] = v_max(zero, v_min(ro[k], one));
                go[k] = v_max(zero, v_min(go[k], one));
                bo[k] = v_max(zero, v_min(bo[k], one));
            }

            if (gammaTab)
            {
                v_float32 vgscale = vx_setall_f32(gscale);
                for(int k = 0; k < nrepeats; k++)
                {
                    ro[k] = v_mul(ro[k], vgscale);
                    go[k] = v_mul(go[k], vgscale);
                    bo[k] = v_mul(bo[k], vgscale);
                }

                for(int k = 0; k < nrepeats; k++)
                {
                    ro[k] = splineInterpolate(ro[k], gammaTab, GAMMA_TAB_SIZE);
                    go[k] = splineInterpolate(go[k], gammaTab, GAMMA_TAB_SIZE);
                    bo[k] = splineInterpolate(bo[k], gammaTab, GAMMA_TAB_SIZE);
                }
            }

            if(dcn == 4)
            {
                v_float32 valpha = vx_setall_f32(alpha);
                for(int k = 0; k < nrepeats; k++)
                {
                    v_store_interleave(dst + 4*vsize*k, ro[k], go[k], bo[k], valpha);
                }
            }
            else // dcn == 3
            {
                for(int k = 0; k < nrepeats; k++)
                {
                    v_store_interleave(dst + 3*vsize*k, ro[k], go[k], bo[k]);
                }
            }
        }
#endif
        for (; i < n; i++, src += 3, dst += dcn)
        {
            float li = src[0];
            float ai = src[1];
            float bi = src[2];

            // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
            float y, fy;
            if (li <= lThresh)
            {
                y = li / 903.3f;
                fy = 7.787f * y + 16.0f / 116.0f;
            }
            else
            {
                fy = (li + 16.0f) / 116.0f;
                y = fy * fy * fy;
            }

            float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

            for (int j = 0; j < 2; j++)
                if (fxz[j] <= fThresh)
                    fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                else
                    fxz[j] = fxz[j] * fxz[j] * fxz[j];

            float x = fxz[0], z = fxz[1];
            float ro = C0 * x + C1 * y + C2 * z;
            float go = C3 * x + C4 * y + C5 * z;
            float bo = C6 * x + C7 * y + C8 * z;
            ro = clip(ro);
            go = clip(go);
            bo = clip(bo);

            if (gammaTab)
            {
                ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = ro, dst[1] = go, dst[2] = bo;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9];
    bool srgb;
    float lThresh;
    float fThresh;
    int blueIdx;
};


// Performs conversion in integers
struct Lab2RGBinteger
{
    typedef uchar channel_type;

    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    // lThresh == (6/29)^3 * (29/3)^3 * BASE/100
    static const int lThresh = 1311;
    // fThresh == ((29/3)^3/(29*4) * (6/29)^3 + 16/116)*BASE
    static const int fThresh = 3390;
    // base16_116 == BASE*16/116
    static const int base16_116 = 2260;
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);

    Lab2RGBinteger( int _dstcn, int blueIdx, const float* _coeffs,
                    const float* _whitept, bool srgb )
    : dstcn(_dstcn), issRGB(srgb)
    {
        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble(_whitept[i]);
            else
                whitePt[i] = D65[i];

        static const softdouble lshift(1 << lab_shift);
        for(int i = 0; i < 3; i++)
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i+j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i+j*3];

            coeffs[i+(blueIdx)*3]   = cvRound(lshift*c[0]*whitePt[i]);
            coeffs[i+3]             = cvRound(lshift*c[1]*whitePt[i]);
            coeffs[i+(blueIdx^2)*3] = cvRound(lshift*c[2]*whitePt[i]);
        }
    }

    // L, a, b should be in their natural range
    inline void process(const uchar LL, const uchar aa, const uchar bb, int& ro, int& go, int& bo) const
    {
        int x, y, z;
        int ify;

        y   = LabToYF_b[LL*2  ];
        ify = LabToYF_b[LL*2+1];

        //float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };
        int adiv, bdiv;
        //adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;
        //approximations with reasonable precision
        adiv = ((5*aa*53687 + (1 << 7)) >> 13) - 128*BASE/500;
        bdiv = ((  bb*41943 + (1 << 4)) >>  9) - 128*BASE/200+1;

        int ifxz[] = {ify + adiv, ify - bdiv};

        for(int k = 0; k < 2; k++)
        {
            int& v = ifxz[k];
            v = abToXZ_b[v-minABvalue];
        }
        x = ifxz[0]; /* y = y */; z = ifxz[1];

        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2];
        int C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5];
        int C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

        ro = max(0, min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = max(0, min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = max(0, min((int)INV_GAMMA_TAB_SIZE-1, bo));

        if(issRGB)
        {
            ushort* tab = sRGBInvGammaTab_b;
            ro = tab[ro];
            go = tab[go];
            bo = tab[bo];
        }
        else
        {
            // rgb = (rgb*255) >> inv_gamma_shift
            ro = ((ro << 8) - ro) >> inv_gamma_shift;
            go = ((go << 8) - go) >> inv_gamma_shift;
            bo = ((bo << 8) - bo) >> inv_gamma_shift;
        }
    }

#if CV_SIMD
    inline void processLabToXYZ(const v_uint8& l, const v_uint8& a, const v_uint8& b,
                                v_int32 (&xiv)[4], v_int32 (&y)[4], v_int32 (&ziv)[4]) const
    {
        v_uint16 l0, l1;
        v_expand(l, l0, l1);
        v_int32 lq[4];
        v_expand(v_reinterpret_as_s16(l0), lq[0], lq[1]);
        v_expand(v_reinterpret_as_s16(l1), lq[2], lq[3]);

        // Load Y and IFY values from lookup-table
        // y = LabToYF_b[L_value*2], ify = LabToYF_b[L_value*2 + 1]
        // LabToYF_b[i*2  ] = y;   // 0 <= y <= BASE
        // LabToYF_b[i*2+1] = ify; // 2260 <= ify <= BASE
        v_int32 yf[4];
        v_int32 ify[4];
        v_int32 mask16 = vx_setall_s32(0xFFFF);
        for(int k = 0; k < 4; k++)
        {
            yf[k] = v_lut((const int*)LabToYF_b, lq[k]);
            y[k]   = v_and(yf[k], mask16);
            ify[k] = v_reinterpret_as_s32(v_shr(v_reinterpret_as_u32(yf[k]), 16));
        }

        v_int16 ify0, ify1;
        ify0 = v_pack(ify[0], ify[1]);
        ify1 = v_pack(ify[2], ify[3]);

        v_int16 adiv0, adiv1, bdiv0, bdiv1;
        v_uint16 a0, a1, b0, b1;
        v_expand(a, a0, a1); v_expand(b, b0, b1);
        //adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;
        //approximations with reasonable precision
        v_uint16 mulA = vx_setall_u16(53687);
        v_uint32 ma[4];
        v_uint32 addA = vx_setall_u32(1 << 7);
        v_mul_expand((v_add(a0, v_shl<2>(a0))), mulA, ma[0], ma[1]);
        v_mul_expand((v_add(a1, v_shl<2>(a1))), mulA, ma[2], ma[3]);
        adiv0 = v_reinterpret_as_s16(v_pack((v_shr<13>(v_add(ma[0], addA))), (v_shr<13>(v_add(ma[1], addA)))));
        adiv1 = v_reinterpret_as_s16(v_pack((v_shr<13>(v_add(ma[2], addA))), (v_shr<13>(v_add(ma[3], addA)))));

        v_uint16 mulB = vx_setall_u16(41943);
        v_uint32 mb[4];
        v_uint32 addB = vx_setall_u32(1 << 4);
        v_mul_expand(b0, mulB, mb[0], mb[1]);
        v_mul_expand(b1, mulB, mb[2], mb[3]);
        bdiv0 = v_reinterpret_as_s16(v_pack(v_shr<9>(v_add(mb[0], addB)), v_shr<9>(v_add(mb[1], addB))));
        bdiv1 = v_reinterpret_as_s16(v_pack(v_shr<9>(v_add(mb[2], addB)), v_shr<9>(v_add(mb[3], addB))));

        // 0 <= adiv <= 8356, 0 <= bdiv <= 20890
        /* x = ifxz[0]; y = y; z = ifxz[1]; */
        v_uint16 xiv0, xiv1, ziv0, ziv1;
        v_int16 vSubA = vx_setall_s16(-128*BASE/500 - minABvalue), vSubB = vx_setall_s16(128*BASE/200-1 - minABvalue);

        // int ifxz[] = {ify + adiv, ify - bdiv};
        // ifxz[k] = abToXZ_b[ifxz[k]-minABvalue];
        xiv0 = v_reinterpret_as_u16(v_add_wrap(v_add_wrap(ify0, adiv0), vSubA));
        xiv1 = v_reinterpret_as_u16(v_add_wrap(v_add_wrap(ify1, adiv1), vSubA));
        ziv0 = v_reinterpret_as_u16(v_add_wrap(v_sub_wrap(ify0, bdiv0), vSubB));
        ziv1 = v_reinterpret_as_u16(v_add_wrap(v_sub_wrap(ify1, bdiv1), vSubB));

        v_uint32 uxiv[4], uziv[4];
        v_expand(xiv0, uxiv[0], uxiv[1]);
        v_expand(xiv1, uxiv[2], uxiv[3]);
        v_expand(ziv0, uziv[0], uziv[1]);
        v_expand(ziv1, uziv[2], uziv[3]);

        for(int k = 0; k < 4; k++)
        {
            xiv[k] = v_lut(abToXZ_b, v_reinterpret_as_s32(uxiv[k]));
            ziv[k] = v_lut(abToXZ_b, v_reinterpret_as_s32(uziv[k]));
        }
        // abToXZ_b[i-minABvalue] = v; // -1335 <= v <= 88231
    }
#endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();

        i = 0;

#if CV_SIMD
        if(enablePackedLab)
        {
            bool srgb = issRGB;
            ushort* tab = sRGBInvGammaTab_b;
            const int vsize = VTraits<v_uint8>::vlanes();
            v_uint8 valpha = vx_setall_u8(alpha);
            v_int32 vc[9];
            for(int k = 0; k < 9; k++)
            {
                vc[k] = vx_setall_s32(coeffs[k]);
            }
            const int descaleShift = 1 << (shift-1);
            v_int32 vdescale = vx_setall_s32(descaleShift);
            for ( ; i <= n-vsize;
                  i += vsize, src += 3*vsize, dst += dcn*vsize)
            {
                v_uint8 l, a, b;
                v_load_deinterleave(src, l, a, b);

                v_int32 xq[4], yq[4], zq[4];
                processLabToXYZ(l, a, b, xq, yq, zq);

                // x, y, z exceed 2^16 so we cannot do v_mul_expand or v_dotprod
                v_int32 rq[4], gq[4], bq[4];
                for(int k = 0; k < 4; k++)
                {
                    rq[k] = v_shr<shift>(v_add(v_add(v_add(v_mul(vc[0], xq[k]), v_mul(vc[1], yq[k])), v_mul(vc[2], zq[k])), vdescale));
                    gq[k] = v_shr<shift>(v_add(v_add(v_add(v_mul(vc[3], xq[k]), v_mul(vc[4], yq[k])), v_mul(vc[5], zq[k])), vdescale));
                    bq[k] = v_shr<shift>(v_add(v_add(v_add(v_mul(vc[6], xq[k]), v_mul(vc[7], yq[k])), v_mul(vc[8], zq[k])), vdescale));
                }

                //limit indices in table and then substitute
                //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                v_int32 z = vx_setzero_s32(), up = vx_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                for (int k = 0; k < 4; k++)
                {
                    rq[k] = v_max(z, v_min(up, rq[k]));
                    gq[k] = v_max(z, v_min(up, gq[k]));
                    bq[k] = v_max(z, v_min(up, bq[k]));
                }

                v_uint16 rgb[6];
                if(srgb)
                {
                    // [RRR... , GGG... , BBB...]
                    int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vidx[VTraits<v_uint8>::max_nlanes*3];
                    for (int k = 0; k < 4; k++)
                        v_store_aligned(vidx + 0*vsize + k*vsize/4, rq[k]);
                    for (int k = 0; k < 4; k++)
                        v_store_aligned(vidx + 1*vsize + k*vsize/4, gq[k]);
                    for (int k = 0; k < 4; k++)
                        v_store_aligned(vidx + 2*vsize + k*vsize/4, bq[k]);

                    rgb[0] = vx_lut(tab, vidx + 0*vsize/2);
                    rgb[1] = vx_lut(tab, vidx + 1*vsize/2);
                    rgb[2] = vx_lut(tab, vidx + 2*vsize/2);
                    rgb[3] = vx_lut(tab, vidx + 3*vsize/2);
                    rgb[4] = vx_lut(tab, vidx + 4*vsize/2);
                    rgb[5] = vx_lut(tab, vidx + 5*vsize/2);
                }
                else
                {
                    // rgb = (rgb*255) >> inv_gamma_shift
                    for(int k = 0; k < 4; k++)
                    {
                        rq[k] = v_shr((v_sub(v_shl(rq[k], 8), rq[k])), inv_gamma_shift);
                        gq[k] = v_shr((v_sub(v_shl(gq[k], 8), gq[k])), inv_gamma_shift);
                        bq[k] = v_shr((v_sub(v_shl(bq[k], 8), bq[k])), inv_gamma_shift);
                    }
                    rgb[0] = v_reinterpret_as_u16(v_pack(rq[0], rq[1]));
                    rgb[1] = v_reinterpret_as_u16(v_pack(rq[2], rq[3]));
                    rgb[2] = v_reinterpret_as_u16(v_pack(gq[0], gq[1]));
                    rgb[3] = v_reinterpret_as_u16(v_pack(gq[2], gq[3]));
                    rgb[4] = v_reinterpret_as_u16(v_pack(bq[0], bq[1]));
                    rgb[5] = v_reinterpret_as_u16(v_pack(bq[2], bq[3]));
                }

                v_uint16 R0, R1, G0, G1, B0, B1;

                v_uint8 R, G, B;
                R = v_pack(rgb[0], rgb[1]);
                G = v_pack(rgb[2], rgb[3]);
                B = v_pack(rgb[4], rgb[5]);

                if(dcn == 4)
                {
                    v_store_interleave(dst, B, G, R, valpha);
                }
                else // dcn == 3
                {
                    v_store_interleave(dst, B, G, R);
                }
            }
        }
#endif

        for (; i < n; i++, src += 3, dst += dcn)
        {
            int ro, go, bo;
            process(src[0], src[1], src[2], ro, go, bo);

            dst[0] = saturate_cast<uchar>(bo);
            dst[1] = saturate_cast<uchar>(go);
            dst[2] = saturate_cast<uchar>(ro);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    int coeffs[9];
    bool issRGB;
};


struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : fcvt(_dstcn, _blueIdx, _coeffs, _whitept, _srgb), dstcn(_dstcn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    Lab2RGBfloat fcvt;
    int dstcn;
};


struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int _blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : fcvt(3, _blueIdx, _coeffs, _whitept, _srgb ), icvt(_dstcn, _blueIdx, _coeffs, _whitept, _srgb), dstcn(_dstcn)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        if(enableBitExactness)
        {
            icvt(src, dst, n);
            return;
        }

        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
#if CV_SIMD
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[3*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
#endif

        static const softfloat fl = softfloat(100)/f255;

#if CV_SIMD
        const int fsize = VTraits<v_float32>::vlanes();
        v_float32 vl = vx_setall_f32((float)fl);
        v_float32 va = vx_setall_f32(1.f);
        v_float32 vb = vx_setall_f32(1.f);
        v_float32 vaLow = vx_setall_f32(-128.f), vbLow = vx_setall_f32(-128.f);
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[VTraits<v_float32>::max_nlanes*3], interTmpA[VTraits<v_float32>::max_nlanes*3];
        v_store_interleave(interTmpM, vl, va, vb);
        v_store_interleave(interTmpA, vx_setzero_f32(), vaLow, vbLow);
        v_float32 mluv[3], aluv[3];
        for(int k = 0; k < 3; k++)
        {
            mluv[k] = vx_load_aligned(interTmpM + k*fsize);
            aluv[k] = vx_load_aligned(interTmpA + k*fsize);
        }
#endif

        i = 0;
        for(; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

#if CV_SIMD
            const int vsize = VTraits<v_uint8>::vlanes();
            for( ; j <= (dn - vsize)*3; j += 3*vsize )
            {
                v_uint8 s0, s1, s2;
                s0 = vx_load(src + j + 0*vsize);
                s1 = vx_load(src + j + 1*vsize);
                s2 = vx_load(src + j + 2*vsize);

                v_uint16 ss[6];
                v_expand(s0, ss[0], ss[1]);
                v_expand(s1, ss[2], ss[3]);
                v_expand(s2, ss[4], ss[5]);
                v_int32 vs[12];
                for(int k = 0; k < 6; k++)
                {
                    v_expand(v_reinterpret_as_s16(ss[k]), vs[k*2+0], vs[k*2+1]);
                }

                for(int bufp = 0; bufp < 12; bufp++)
                {
                    v_store_aligned(buf + j + bufp, v_muladd(v_cvt_f32(vs[bufp]), mluv[bufp%3], aluv[bufp%3]));
                }
            }
#endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*((float)fl);
                buf[j+1] = (float)(src[j+1] - 128.f);
                buf[j+2] = (float)(src[j+2] - 128.f);
            }

            fcvt(buf, buf, dn);

            j = 0;

#if CV_SIMD
            static const int nBlock = 4*fsize;
            v_float32 v255 = vx_setall_f32(255.f);
            if(dcn == 4)
            {
                v_uint8 valpha = vx_setall_u8(alpha);
                for( ; j <= (dn-nBlock)*3;
                     j += nBlock*3, dst += nBlock)
                {
                    v_float32 vf[4*3];
                    for(int k = 0; k < 4; k++)
                    {
                        v_load_deinterleave(buf + j, vf[k*3+0], vf[k*3+1], vf[k*3+2]);
                    }

                    v_int32 vi[4*3];
                    for(int k = 0; k < 4*3; k++)
                    {
                        vi[k] = v_round(v_mul(vf[k], v255));
                    }

                    v_uint8 rgb[3];
                    for(int k = 0; k < 3; k++)
                    {
                        rgb[k] = v_pack_u(v_pack(vi[0*3+k], vi[1*3+k]),
                                          v_pack(vi[2*3+k], vi[3*3+k]));
                    }

                    v_store_interleave(dst, rgb[0], rgb[1], rgb[2], valpha);
                }
            }
            else // dcn == 3
            {
                for(; j < dn*3 - nBlock; j += nBlock, dst += nBlock)
                {
                    v_float32 vf[4];
                    v_int32 vi[4];
                    for(int k = 0; k < 4; k++)
                    {
                        vf[k] = vx_load_aligned(buf + j + k*fsize);
                        vi[k] = v_round(v_mul(vf[k], v255));
                    }
                    v_store(dst, v_pack_u(v_pack(vi[0], vi[1]), v_pack(vi[2], vi[3])));
                }
            }
#endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    Lab2RGBfloat   fcvt;
    Lab2RGBinteger icvt;
    int dstcn;
};

///////////////////////////////////// RGB <-> L*u*v* /////////////////////////////////////

struct RGB2Luvfloat
{
    typedef float channel_type;

    RGB2Luvfloat( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : srccn(_srccn), srgb(_srgb)
    {
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++ )
            if(whitept)
                whitePt[i] = softdouble(whitept[i]);
            else
                whitePt[i] = D65[i];

        for(int i = 0; i < 3; i++ )
        {
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    coeffs[i*3+j] = _coeffs[i*3+j];
                else
                    coeffs[i*3+j] = (float)(sRGB2XYZ_D65[i*3+j]);

            if( blueIdx == 0 )
                std::swap(coeffs[i*3], coeffs[i*3+2]);
            CV_Assert( coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      softfloat(coeffs[i*3]) +
                      softfloat(coeffs[i*3+1]) +
                      softfloat(coeffs[i*3+2]) < softfloat(1.5f) );
        }

        softfloat d = whitePt[0] +
                      whitePt[1]*softdouble(15) +
                      whitePt[2]*softdouble(3);
        d = softfloat::one()/max(d, softfloat::eps());
        un = d*softfloat(13*4)*whitePt[0];
        vn = d*softfloat(13*9)*whitePt[1];

        CV_Assert(whitePt[1] == softdouble::one());
    }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

#if CV_SIMD
        const int vsize = VTraits<v_float32>::vlanes();
        const int nrepeats = VTraits<v_float32>::nlanes == 4 ? 2 : 1;
        for( ; i <= n-vsize*nrepeats;
             i+= vsize*nrepeats, src += scn*vsize*nrepeats, dst += 3*vsize*nrepeats)
        {
            v_float32 R[nrepeats], G[nrepeats], B[nrepeats], A;
            if(scn == 4)
            {
                for (int k = 0; k < nrepeats; k++)
                {
                    v_load_deinterleave(src + k*4*vsize, R[k], G[k], B[k], A);
                }
            }
            else // scn == 3
            {
                for (int k = 0; k < nrepeats; k++)
                {
                    v_load_deinterleave(src + k*3*vsize, R[k], G[k], B[k]);
                }
            }

            v_float32 zero = vx_setzero_f32(), one = vx_setall_f32(1.f);
            for (int k = 0; k < nrepeats; k++)
            {
                R[k] = v_min(v_max(R[k], zero), one);
                G[k] = v_min(v_max(G[k], zero), one);
                B[k] = v_min(v_max(B[k], zero), one);
            }

            if(gammaTab)
            {
                v_float32 vgscale = vx_setall_f32(gscale);
                for (int k = 0; k < nrepeats; k++)
                {
                    R[k] = v_mul(R[k], vgscale);
                    G[k] = v_mul(G[k], vgscale);
                    B[k] = v_mul(B[k], vgscale);
                }

                for (int k = 0; k < nrepeats; k++)
                {
                    R[k] = splineInterpolate(R[k], gammaTab, GAMMA_TAB_SIZE);
                    G[k] = splineInterpolate(G[k], gammaTab, GAMMA_TAB_SIZE);
                    B[k] = splineInterpolate(B[k], gammaTab, GAMMA_TAB_SIZE);
                }
            }

            v_float32 X[nrepeats], Y[nrepeats], Z[nrepeats];
            v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
            v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
            v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
            for (int k = 0; k < nrepeats; k++)
            {
                X[k] = v_fma(R[k], vc0, v_fma(G[k], vc1, v_mul(B[k], vc2)));
                Y[k] = v_fma(R[k], vc3, v_fma(G[k], vc4, v_mul(B[k], vc5)));
                Z[k] = v_fma(R[k], vc6, v_fma(G[k], vc7, v_mul(B[k], vc8)));
            }

            v_float32 L[nrepeats], u[nrepeats], v[nrepeats];
            v_float32 vmun = vx_setall_f32(-un), vmvn = vx_setall_f32(-vn);
            for (int k = 0; k < nrepeats; k++)
            {
                L[k] = splineInterpolate(v_mul(Y[k], vx_setall_f32(LabCbrtTabScale)), LabCbrtTab, LAB_CBRT_TAB_SIZE);
                // L = 116.f*L - 16.f;
                L[k] = v_fma(L[k], vx_setall_f32(116.f), vx_setall_f32(-16.f));

                v_float32 d;
                // d = (4*13) / max(X + 15 * Y + 3 * Z, FLT_EPSILON)
                d = v_fma(Y[k], vx_setall_f32(15.f), v_fma(Z[k], vx_setall_f32(3.f), X[k]));
                d = v_div(vx_setall_f32(4.F * 13.F), v_max(d, vx_setall_f32(FLT_EPSILON)));
                // u = L*(X*d - un)
                u[k] = v_mul(L[k], v_fma(X[k], d, vmun));
                // v = L*((9*0.25f)*Y*d - vn);
                v[k] = v_mul(L[k], v_fma(v_mul(vx_setall_f32(9.F * 0.25F), Y[k]), d, vmvn));
            }

            for (int k = 0; k < nrepeats; k++)
            {
                v_store_interleave(dst + k*3*vsize, L[k], u[k], v[k]);
            }
        }
#endif

        for( ; i < n; i++, src += scn, dst += 3 )
        {
            float R = src[0], G = src[1], B = src[2];
            R = clip(R);
            G = clip(G);
            B = clip(B);
            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            L = 116.f*L - 16.f;

            float d = (4*13) / std::max(X + 15 * Y + 3 * Z, FLT_EPSILON);
            float u = L*(X*d - un);
            float v = L*((9*0.25f)*Y*d - vn);

            dst[0] = L; dst[1] = u; dst[2] = v;
        }
    }

    int srccn;
    float coeffs[9], un, vn;
    bool srgb;
};

struct RGB2Luv_f
{
    typedef float channel_type;

    RGB2Luv_f( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : fcvt(_srccn, blueIdx, _coeffs, whitept, _srgb), srccn(_srccn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    RGB2Luvfloat fcvt;
    int srccn;
};

struct Luv2RGBfloat
{
    typedef float channel_type;

    Luv2RGBfloat( int _dstcn, int blueIdx, const float* _coeffs,
                  const float* whitept, bool _srgb )
    : dstcn(_dstcn),  srgb(_srgb)
    {
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(whitept)
                whitePt[i] = softdouble(whitept[i]);
            else
                whitePt[i] = D65[i];

        for( int i = 0; i < 3; i++ )
        {
            softfloat c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softfloat(_coeffs[i+j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i+j*3];

            coeffs[i+(blueIdx^2)*3] = c[0];
            coeffs[i+3]             = c[1];
            coeffs[i+blueIdx*3]     = c[2];
        }

        softfloat d = whitePt[0] +
                      whitePt[1]*softdouble(15) +
                      whitePt[2]*softdouble(3);
        d = softfloat::one()/max(d, softfloat::eps());
        un = softfloat(4*13)*d*whitePt[0];
        vn = softfloat(9*13)*d*whitePt[1];

        CV_Assert(whitePt[1] == softdouble::one());
    }

    void operator()(const float* src, float* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        float _un = un, _vn = vn;

#if CV_SIMD
        const int vsize = VTraits<v_float32>::vlanes();
        const int nrepeats = VTraits<v_float32>::nlanes == 4 ? 2 : 1;
        for( ; i <= n - vsize*nrepeats;
             i += vsize*nrepeats, src += vsize*3*nrepeats, dst += dcn*vsize*nrepeats)
        {
            v_float32 L[nrepeats], u[nrepeats], v[nrepeats];
            for (int k = 0; k < nrepeats; k++)
            {
                v_load_deinterleave(src + k*vsize*3, L[k], u[k], v[k]);
            }

            v_float32 X[nrepeats], Y[nrepeats], Z[nrepeats];

            v_float32 v16 = vx_setall_f32(16.f);
            v_float32 v116inv = vx_setall_f32(1.f/116.f);
            v_float32 v903inv = vx_setall_f32(1.0f/903.296296f); //(3./29.)^3
            for (int k = 0; k < nrepeats; k++)
            {
                v_float32 Ylo, Yhi;

                // ((L + 16)/116)^3
                Ylo = v_mul(v_add(L[k], v16), v116inv);
                Ylo = v_mul(v_mul(Ylo, Ylo), Ylo);
                // L*(3./29.)^3
                Yhi = v_mul(L[k], v903inv);

                // Y = (L <= 8) ? Y0 : Y1;
                Y[k] = v_select(v_ge(L[k], vx_setall_f32(8.f)), Ylo, Yhi);
            }

            v_float32 v4inv = vx_setall_f32(0.25f), v3 = vx_setall_f32(3.f);
            for(int k = 0; k < nrepeats; k++)
            {
                v_float32 up, vp;

                // up = 3*(u + L*_un);
                up = v_mul(v3, v_fma(L[k], vx_setall_f32(_un), u[k]));
                // vp = 0.25/(v + L*_vn);
                vp = v_div(v4inv, v_fma(L[k], vx_setall_f32(_vn), v[k]));

                // vp = max(-0.25, min(0.25, vp));
                vp = v_max(vx_setall_f32(-0.25f), v_min(v4inv, vp));

                //X = 3*up*vp; // (*Y) is done later
                X[k] = v_mul(v_mul(v3, up), vp);
                //Z = ((12*13*L - up)*vp - 5); // (*Y) is done later
                // xor flips the sign, works like unary minus
                Z[k] = v_fma(v_fma(L[k], vx_setall_f32(12.f*13.f), (v_xor(vx_setall_f32(-0.F), up))), vp, vx_setall_f32(-5.f));
            }

            v_float32 R[nrepeats], G[nrepeats], B[nrepeats];
            v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
            v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
            v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
            for(int k = 0; k < nrepeats; k++)
            {
                // R = (X*C0 + C1 + Z*C2)*Y; // here (*Y) is done
                R[k] = v_mul(v_fma(Z[k], vc2, v_fma(X[k], vc0, vc1)), Y[k]);
                G[k] = v_mul(v_fma(Z[k], vc5, v_fma(X[k], vc3, vc4)), Y[k]);
                B[k] = v_mul(v_fma(Z[k], vc8, v_fma(X[k], vc6, vc7)), Y[k]);
            }

            v_float32 vzero = vx_setzero_f32(), v1 = vx_setall_f32(1.f);
            for(int k = 0; k < nrepeats; k++)
            {
                R[k] = v_min(v_max(R[k], vzero), v1);
                G[k] = v_min(v_max(G[k], vzero), v1);
                B[k] = v_min(v_max(B[k], vzero), v1);
            }

            if(gammaTab)
            {
                v_float32 vgscale = vx_setall_f32(gscale);
                for(int k = 0; k < nrepeats; k++)
                {
                    R[k] = v_mul(R[k], vgscale);
                    G[k] = v_mul(G[k], vgscale);
                    B[k] = v_mul(B[k], vgscale);
                }
                for(int k = 0; k < nrepeats; k++)
                {
                    R[k] = splineInterpolate(R[k], gammaTab, GAMMA_TAB_SIZE);
                    G[k] = splineInterpolate(G[k], gammaTab, GAMMA_TAB_SIZE);
                    B[k] = splineInterpolate(B[k], gammaTab, GAMMA_TAB_SIZE);
                }
            }
            for(int k = 0; k < nrepeats; k++)
            {
                if(dcn == 4)
                {
                    v_store_interleave(dst + k*vsize*4, R[k], G[k], B[k], vx_setall_f32(alpha));
                }
                else // dcn == 3
                {
                    v_store_interleave(dst + k*vsize*3, R[k], G[k], B[k]);
                }
            }
        }
#endif

        for( ; i < n; i++, src += 3,  dst += dcn )
        {
            float L = src[0], u = src[1], v = src[2], X, Y, Z;
            if(L >= 8)
            {
                Y = (L + 16.f) * (1.f/116.f);
                Y = Y*Y*Y;
            }
            else
            {
                Y = L * (1.0f/903.3f); // L*(3./29.)^3
            }
            float up = 3.f*(u + L*_un);
            float vp = 0.25f/(v + L*_vn);
            if(vp >  0.25f) vp =  0.25f;
            if(vp < -0.25f) vp = -0.25f;
            X = Y*3.f*up*vp;
            Z = Y*(((12.f*13.f)*L - up)*vp - 5.f);

            float R = X*C0 + Y*C1 + Z*C2;
            float G = X*C3 + Y*C4 + Z*C5;
            float B = X*C6 + Y*C7 + Z*C8;

            R = clip(R);
            G = clip(G);
            B = clip(B);

            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = R; dst[1] = G; dst[2] = B;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct Luv2RGB_f
{
    typedef float channel_type;

    Luv2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* whitept, bool _srgb )
    : fcvt(_dstcn, blueIdx, _coeffs, whitept, _srgb), dstcn(_dstcn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    Luv2RGBfloat fcvt;
    int dstcn;
};

struct RGB2Luvinterpolate
{
    typedef uchar channel_type;

    RGB2Luvinterpolate( int _srccn, int _blueIdx, const float* /* _coeffs */,
                        const float* /* _whitept */, bool /*_srgb*/ )
    : srccn(_srccn), blueIdx(_blueIdx)
    {
        initLabTabs();
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i, scn = srccn, bIdx = blueIdx;

        i = 0; n *= 3;

#if CV_SIMD
        if(enablePackedRGB2Luv)
        {
            const int vsize = VTraits<v_uint16>::vlanes();
            static const int nPixels = vsize*2;
            for(; i < n - 3*nPixels; i += 3*nPixels, src += scn*nPixels)
            {
                /*
                    int R = src[bIdx], G = src[1], B = src[bIdx^2];
                */
                v_uint8 r, g, b, dummy;
                if(scn == 3)
                {
                    v_load_deinterleave(src, r, g, b);
                }
                else // scn == 4
                {
                    v_load_deinterleave(src, r, g, b, dummy);
                }

                if(bIdx)
                {
                    swap(r, b);
                }

                /*
                    static const int baseDiv = LAB_BASE/256;
                    R = R*baseDiv, G = G*baseDiv, B = B*baseDiv;
                */
                v_uint16 r0, r1, g0, g1, b0, b1;
                v_expand(r, r0, r1);
                v_expand(g, g0, g1);
                v_expand(b, b0, b1);
                r0 = v_shl<lab_base_shift - 8>(r0); r1 = v_shl<lab_base_shift - 8>(r1);
                g0 = v_shl<lab_base_shift - 8>(g0); g1 = v_shl<lab_base_shift - 8>(g1);
                b0 = v_shl<lab_base_shift - 8>(b0); b1 = v_shl<lab_base_shift - 8>(b1);

                /*
                    int L, u, v;
                    trilinearInterpolate(R, G, B, RGB2LuvLUT_s16, L, u, v);
                 */
                v_uint16 l0, u0, v0, l1, u1, v1;
                trilinearPackedInterpolate(r0, g0, b0, LABLUVLUTs16.RGB2LuvLUT_s16, l0, u0, v0);
                trilinearPackedInterpolate(r1, g1, b1, LABLUVLUTs16.RGB2LuvLUT_s16, l1, u1, v1);

                /*
                    dst[i]   = saturate_cast<uchar>(L/baseDiv);
                    dst[i+1] = saturate_cast<uchar>(u/baseDiv);
                    dst[i+2] = saturate_cast<uchar>(v/baseDiv);
                 */
                l0 = v_shr<lab_base_shift - 8>(l0); l1 = v_shr<lab_base_shift - 8>(l1);
                u0 = v_shr<lab_base_shift - 8>(u0); u1 = v_shr<lab_base_shift - 8>(u1);
                v0 = v_shr<lab_base_shift - 8>(v0); v1 = v_shr<lab_base_shift - 8>(v1);
                v_uint8 l = v_pack(l0, l1);
                v_uint8 u = v_pack(u0, u1);
                v_uint8 v = v_pack(v0, v1);
                v_store_interleave(dst + i, l, u, v);
            }
        }
#endif // CV_SIMD

        for(; i < n; i += 3, src += scn)
        {
            int R = src[bIdx], G = src[1], B = src[bIdx^2];

            // (LAB_BASE/255) gives more accuracy but not very much
            static const int baseDiv = LAB_BASE/256;
            R = R*baseDiv, G = G*baseDiv, B = B*baseDiv;

            int L, u, v;
            trilinearInterpolate(R, G, B, LABLUVLUTs16.RGB2LuvLUT_s16, L, u, v);

            dst[i] = saturate_cast<uchar>(L/baseDiv);
            dst[i+1] = saturate_cast<uchar>(u/baseDiv);
            dst[i+2] = saturate_cast<uchar>(v/baseDiv);
        }
    }

    int srccn;
    int blueIdx;
};


struct RGB2Luv_b
{
    typedef uchar channel_type;
    static const int bufChannels = 3;

    RGB2Luv_b( int _srccn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : srccn(_srccn),
      fcvt(bufChannels, blueIdx, _coeffs, _whitept, _srgb),
      icvt(_srccn, blueIdx, _coeffs, _whitept, _srgb)
    {
        // using interpolation for LRGB gives error up to 8 of 255, don't use it
        useInterpolation = (!_coeffs && !_whitept && _srgb
                            && enableBitExactness
                            && enableRGB2LuvInterpolation);
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        if(useInterpolation)
        {
            icvt(src, dst, n);
            return;
        }

        int scn = srccn;
#if CV_SIMD
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[bufChannels*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[bufChannels*BLOCK_SIZE];
#endif

        static const softfloat fL = f255/softfloat(100);
        static const softfloat fu = f255/uRange;
        static const softfloat fv = f255/vRange;
        static const softfloat su = -uLow*f255/uRange;
        static const softfloat sv = -vLow*f255/vRange;
#if CV_SIMD
        const int fsize = VTraits<v_float32>::vlanes();
        v_float32 ml = vx_setall_f32((float)fL), al = vx_setzero_f32();
        v_float32 mu = vx_setall_f32((float)fu), au = vx_setall_f32((float)su);
        v_float32 mv = vx_setall_f32((float)fv), av = vx_setall_f32((float)sv);
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[VTraits<v_float32>::max_nlanes*3], interTmpA[VTraits<v_float32>::max_nlanes*3];
        v_store_interleave(interTmpM, ml, mu, mv);
        v_store_interleave(interTmpA, al, au, av);
        v_float32 mluv[3], aluv[3];
        for(int k = 0; k < 3; k++)
        {
            mluv[k] = vx_load_aligned(interTmpM + k*fsize);
            aluv[k] = vx_load_aligned(interTmpA + k*fsize);
        }
#endif

        for(int i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*bufChannels )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            static const softfloat f255inv = softfloat::one()/f255;
#if CV_SIMD
            v_float32 v255inv = vx_setall_f32((float)f255inv);
            if(scn == 4)
            {
                int j = 0;
                static const int nBlock = fsize*4;
                for( ; j <= dn*bufChannels - nBlock*3;
                     j += nBlock*3, src += nBlock*4)
                {
                    v_uint8 rgb[3], dummy;
                    v_load_deinterleave(src, rgb[0], rgb[1], rgb[2], dummy);

                    v_uint16 d[3*2];
                    for(int k = 0; k < 3; k++)
                    {
                        v_expand(rgb[k], d[k*2+0], d[k*2+1]);
                    }
                    v_int32 q[3*4];
                    for(int k = 0; k < 3*2; k++)
                    {
                        v_expand(v_reinterpret_as_s16(d[k]), q[k*2+0], q[k*2+1]);
                    }

                    v_float32 f[3*4];
                    for(int k = 0; k < 3*4; k++)
                    {
                        f[k] = v_mul(v_cvt_f32(q[k]), v255inv);
                    }

                    for(int k = 0; k < 4; k++)
                    {
                        v_store_interleave(buf + j + k*3*fsize, f[0*4+k], f[1*4+k], f[2*4+k]);
                    }
                }
                for( ; j < dn*bufChannels; j += bufChannels, src += 4 )
                {
                    buf[j  ] = (float)(src[0]*((float)f255inv));
                    buf[j+1] = (float)(src[1]*((float)f255inv));
                    buf[j+2] = (float)(src[2]*((float)f255inv));
                }
            }
            else // scn == 3
            {
                int j = 0;
                static const int nBlock = fsize*2;
                for( ; j <= dn*bufChannels - nBlock;
                     j += nBlock, src += nBlock)
                {
                    v_uint16 d = vx_load_expand(src);
                    v_int32 q0, q1;
                    v_expand(v_reinterpret_as_s16(d), q0, q1);

                    v_store_aligned(buf + j + 0*fsize, v_mul(v_cvt_f32(q0), v255inv));
                    v_store_aligned(buf + j + 1*fsize, v_mul(v_cvt_f32(q1), v255inv));
                }
                for( ; j < dn*bufChannels; j++, src++ )
                {
                    buf[j] = (float)(src[0]*((float)f255inv));
                }
            }
#else
            for(int j = 0; j < dn*bufChannels; j += bufChannels, src += scn )
            {
                buf[j  ] = (float)(src[0]*((float)f255inv));
                buf[j+1] = (float)(src[1]*((float)f255inv));
                buf[j+2] = (float)(src[2]*((float)f255inv));
            }
#endif

            fcvt(buf, buf, dn);

            int j = 0;

#if CV_SIMD
            for( ; j <= dn*3 - fsize*3*4; j += fsize*3*4)
            {
                v_float32 f[3*4];
                for(int k = 0; k < 3*4; k++)
                    f[k] = vx_load_aligned(buf + j + k*fsize);

                for(int k = 0; k < 4; k++)
                {
                    f[k*3+0] = v_fma(f[k*3+0], mluv[0], aluv[0]);
                    f[k*3+1] = v_fma(f[k*3+1], mluv[1], aluv[1]);
                    f[k*3+2] = v_fma(f[k*3+2], mluv[2], aluv[2]);
                }

                v_int32 q[3*4];
                for(int k = 0; k < 3*4; k++)
                {
                    q[k] = v_round(f[k]);
                }

                for(int k = 0; k < 3; k++)
                {
                    v_store(dst + j + k*fsize*4, v_pack_u(v_pack(q[k*4+0], q[k*4+1]),
                                                          v_pack(q[k*4+2], q[k*4+3])));
                }
            }
#endif
            for( ; j < dn*3; j += 3 )
            {
                dst[j+0] = saturate_cast<uchar>(buf[j+0]*(float)fL);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*(float)fu + (float)su);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*(float)fv + (float)sv);
            }
        }
    }

    int srccn;
    RGB2Luvfloat fcvt;
    RGB2Luvinterpolate icvt;

    bool useInterpolation;
};


struct Luv2RGBinteger
{
    typedef uchar channel_type;

    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);

    // whitept is fixed for int calculations
    Luv2RGBinteger( int _dstcn, int blueIdx, const float* _coeffs,
                    const float* /*_whitept*/, bool _srgb )
    : dstcn(_dstcn), issRGB(_srgb)
    {
        initLabTabs();

        static const softdouble lshift(1 << lab_shift);
        for(int i = 0; i < 3; i++)
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i + j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i + j*3];

            coeffs[i+blueIdx*3]     = cvRound(lshift*c[0]);
            coeffs[i+3]             = cvRound(lshift*c[1]);
            coeffs[i+(blueIdx^2)*3] = cvRound(lshift*c[2]);
        }
    }

    // L, u, v should be in their natural range
    inline void process(const uchar LL, const uchar uu, const uchar vv, int& ro, int& go, int& bo) const
    {
        ushort y = LabToYF_b[LL*2];

        // y : [0, BASE]
        // up: [-402, 1431.57]*(BASE/1024)
        // vp: +/- 0.25*BASE*1024
        int up = LUVLUT.LuToUp_b[LL*256+uu];
        int vp = LUVLUT.LvToVp_b[LL*256+vv];
        // X = y*3.f* up/((float)BASE/1024) *vp/((float)BASE*1024);
        // Z = y*(((12.f*13.f)*((float)LL)*100.f/255.f - up/((float)BASE))*vp/((float)BASE*1024) - 5.f);

        long long int xv = ((int)up)*(long long)vp;
        int x = (int)(xv/BASE);
        x = (int)(((long long int)y)*x/BASE);

        long long int vpl = LUVLUT.LvToVpl_b[LL*256+vv];
        long long int zp = vpl - xv*(255/3);
        zp /= BASE;
        long long int zq = zp - (long long)(5*255*BASE);
        int zm = (int)(y*zq/BASE);
        int z = zm/256 + zm/65536;

        //limit X, Y, Z to [0, 2] to fit white point
        x = max(0, min(2*BASE, x)); z = max(0, min(2*BASE, z));

        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2];
        int C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5];
        int C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

        ro = max(0, min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = max(0, min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = max(0, min((int)INV_GAMMA_TAB_SIZE-1, bo));

        if(issRGB)
        {
            ushort* tab = sRGBInvGammaTab_b;
            ro = tab[ro];
            go = tab[go];
            bo = tab[bo];
        }
        else
        {
            // rgb = (rgb*255) >> inv_gamma_shift
            ro = ((ro << 8) - ro) >> inv_gamma_shift;
            go = ((go << 8) - go) >> inv_gamma_shift;
            bo = ((bo << 8) - bo) >> inv_gamma_shift;
        }
    }

#if CV_SIMD
    inline void processLuvToXYZ(const v_uint8& lv, const v_uint8& uv, const v_uint8& vv,
                                v_int32 (&x)[4], v_int32 (&y)[4], v_int32 (&z)[4]) const
    {
        const int vsize = VTraits<v_uint8>::vlanes();
        const int vsize_max = VTraits<v_uint8>::max_nlanes;

        v_uint16 lv0, lv1;
        v_expand(lv, lv0, lv1);
        v_uint32 lq[4];
        v_expand(lv0, lq[0], lq[1]);
        v_expand(lv1, lq[2], lq[3]);

        // y = LabToYF_b[LL*2];
        // load int32 instead of int16 then cut unused part by masking
        v_int32 mask16 = vx_setall_s32(0xFFFF);
        for(int k = 0; k < 4; k++)
        {
            y[k] = v_and(v_lut((const int *)LabToYF_b, v_reinterpret_as_s32(lq[k])), mask16);
        }

        v_int32 up[4], vp[4];
        // int up = LUVLUT.LuToUp_b[LL*256+u];
        // int vp = LUVLUT.LvToVp_b[LL*256+v];
        v_uint16 uv0, uv1, vv0, vv1;
        v_expand(uv, uv0, uv1);
        v_expand(vv, vv0, vv1);
        // LL*256
        v_uint16 ll0, ll1;
        ll0 = v_shl<8>(lv0); ll1 = v_shl<8>(lv1);
        v_uint16 upidx0, upidx1, vpidx0, vpidx1;
        upidx0 = v_add(ll0, uv0); upidx1 = v_add(ll1, uv1);
        vpidx0 = v_add(ll0, vv0); vpidx1 = v_add(ll1, vv1);
        v_uint32 upidx[4], vpidx[4];
        v_expand(upidx0, upidx[0], upidx[1]); v_expand(upidx1, upidx[2], upidx[3]);
        v_expand(vpidx0, vpidx[0], vpidx[1]); v_expand(vpidx1, vpidx[2], vpidx[3]);
        for(int k = 0; k < 4; k++)
        {
            up[k] = v_lut(LUVLUT.LuToUp_b, v_reinterpret_as_s32(upidx[k]));
            vp[k] = v_lut(LUVLUT.LvToVp_b, v_reinterpret_as_s32(vpidx[k]));
        }

        // long long int vpl = LUVLUT.LvToVpl_b[LL*256+v];
        v_int64 vpl[8];
        int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vpidxstore[vsize_max];
        for(int k = 0; k < 4; k++)
        {
            v_store_aligned(vpidxstore + k*vsize/4, v_reinterpret_as_s32(vpidx[k]));
        }
        for(int k = 0; k < 8; k++)
        {
            vpl[k] = vx_lut((const int64_t*)LUVLUT.LvToVpl_b, vpidxstore + k*vsize/8);
        }

        // not all 64-bit arithmetic is available in univ. intrinsics
        // need to handle it with scalar code
        int64_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vvpl[vsize_max];
        for(int k = 0; k < 8; k++)
        {
            v_store_aligned(vvpl + k*vsize/8, vpl[k]);
        }
        int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) vup[vsize_max], vvp[vsize_max],
                                               vx[vsize_max], vy[vsize_max], vzm[vsize_max];
        for(int k = 0; k < 4; k++)
        {
            v_store_aligned(vup + k*vsize/4, up[k]);
            v_store_aligned(vvp + k*vsize/4, vp[k]);
            v_store_aligned(vy + k*vsize/4, y[k]);
        }
        for(int i = 0; i < vsize; i++)
        {
            int32_t y_ = vy[i];
            int32_t up_ = vup[i];
            int32_t vp_ = vvp[i];

            int64_t vpl_ = vvpl[i];
            int64_t xv = up_*(int64_t)vp_;

            int64_t zp = vpl_ - xv*(255/3);
            zp = zp >> base_shift;
            int64_t zq = zp - (5*255*BASE);
            int32_t zm = (int32_t)((y_*zq) >> base_shift);
            vzm[i] = zm;

            vx[i] = (int32_t)(xv >> base_shift);
            vx[i] = (int32_t)((((int64_t)y_)*vx[i]) >> base_shift);
        }
        v_int32 zm[4];
        for(int k = 0; k < 4; k++)
        {
            x[k] = vx_load_aligned(vx + k*vsize/4);
            zm[k] = vx_load_aligned(vzm + k*vsize/4);
        }

        // z = zm/256 + zm/65536;
        for (int k = 0; k < 4; k++)
        {
            z[k] = v_add(v_shr<8>(zm[k]), v_shr<16>(zm[k]));
        }

        // (x, z) = clip((x, z), min=0, max=2*BASE)
        v_int32 zero = vx_setzero_s32(), base2 = vx_setall_s32(2*BASE);
        for(int k = 0; k < 4; k++)
        {
            x[k] = v_max(zero, v_min(base2, x[k]));
            z[k] = v_max(zero, v_min(base2, z[k]));
        }
    }
#endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        int i, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();

        i = 0;

#if CV_SIMD
        if(enablePackedLuv2RGB)
        {
            ushort* tab = sRGBInvGammaTab_b;
            bool srgb = issRGB;
            static const int vsize = VTraits<v_uint8>::vlanes();
            const int descaleShift = 1 << (shift-1);
            v_int16 vdescale = vx_setall_s16(descaleShift);
            v_int16 vc[9];
            for(int k = 0; k < 9; k++)
            {
                vc[k] = vx_setall_s16((short)coeffs[k]);
            }
            v_int16 one = vx_setall_s16(1);
            v_int16 cbxy, cbz1, cgxy, cgz1, crxy, crz1;
            v_int16 dummy;
            v_zip(vc[0], vc[1], crxy, dummy);
            v_zip(vc[2],   one, crz1, dummy);
            v_zip(vc[3], vc[4], cgxy, dummy);
            v_zip(vc[5],   one, cgz1, dummy);
            v_zip(vc[6], vc[7], cbxy, dummy);
            v_zip(vc[8],   one, cbz1, dummy);
            // fixing 16bit signed multiplication
            // by subtracting 2^(base_shift-1) and then adding result back
            v_int32 dummy32, fm[3];
            v_expand(v_add(vc[0],vc[1],vc[2]), fm[0], dummy32);
            v_expand(v_add(vc[3],vc[4],vc[5]), fm[1], dummy32);
            v_expand(v_add(vc[6],vc[7],vc[8]), fm[2], dummy32);
            fm[0] = v_shl(fm[0], (base_shift-1));
            fm[1] = v_shl(fm[1], (base_shift-1));
            fm[2] = v_shl(fm[2], (base_shift-1));

            for (; i <= n-vsize; i += vsize, src += 3*vsize, dst += dcn*vsize)
            {
                v_uint8 u8l, u8u, u8v;
                v_load_deinterleave(src, u8l, u8u, u8v);

                v_int32 xiv[4], yiv[4], ziv[4];

                processLuvToXYZ(u8l, u8u, u8v, xiv, yiv, ziv);

                // [xxyyzz]
                v_uint16 xyz[6];
                xyz[0] = v_pack_u(xiv[0], xiv[1]); xyz[1] = v_pack_u(xiv[2], xiv[3]);
                xyz[2] = v_pack_u(yiv[0], yiv[1]); xyz[3] = v_pack_u(yiv[2], yiv[3]);
                xyz[4] = v_pack_u(ziv[0], ziv[1]); xyz[5] = v_pack_u(ziv[2], ziv[3]);

                // ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                // go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                // bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

                // fix 16bit multiplication: c_i*v = c_i*(v-fixmul) + c_i*fixmul
                v_uint16 fixmul = vx_setall_u16(1 << (base_shift-1));
                v_int16 sxyz[6];
                for(int k = 0; k < 6; k++)
                {
                    sxyz[k] = v_reinterpret_as_s16(v_sub_wrap(xyz[k], fixmul));
                }

                v_int16 xy[4], zd[4];
                v_zip(sxyz[0], sxyz[2], xy[0], xy[1]);
                v_zip(sxyz[4], vdescale, zd[0], zd[1]);
                v_zip(sxyz[1], sxyz[3], xy[2], xy[3]);
                v_zip(sxyz[5], vdescale, zd[2], zd[3]);

                // [rrrrggggbbbb]
                v_int32 i_rgb[4*3];
                // a bit faster than one loop for all
                for(int k = 0; k < 4; k++)
                {
                    i_rgb[k+4*0] = v_shr<shift>(v_add(v_add(v_dotprod(xy[k], crxy), v_dotprod(zd[k], crz1)), fm[0]));
                }
                for(int k = 0; k < 4; k++)
                {
                    i_rgb[k+4*1] = v_shr<shift>(v_add(v_add(v_dotprod(xy[k], cgxy), v_dotprod(zd[k], cgz1)), fm[1]));
                }
                for(int k = 0; k < 4; k++)
                {
                    i_rgb[k+4*2] = v_shr<shift>(v_add(v_add(v_dotprod(xy[k], cbxy), v_dotprod(zd[k], cbz1)), fm[2]));
                }

                // [rrggbb]
                v_uint16 u_rgbvec[6];

                // limit indices in table and then substitute
                v_int32 z32 = vx_setzero_s32();
                v_int32 tabsz = vx_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                for(int k = 0; k < 12; k++)
                {
                    i_rgb[k] = v_max(z32, v_min(tabsz, i_rgb[k]));
                }

                // ro = tab[ro]; go = tab[go]; bo = tab[bo];
                if(srgb)
                {
                    // [rr.., gg.., bb..]
                    int32_t CV_DECL_ALIGNED(CV_SIMD_WIDTH) rgbshifts[3*VTraits<v_uint8>::max_nlanes];
                    for(int k = 0; k < 12; k++)
                    {
                        v_store_aligned(rgbshifts + k*vsize/4, i_rgb[k]);
                    }
                    for(int k = 0; k < 6; k++)
                    {
                        u_rgbvec[k] = vx_lut(tab, rgbshifts + k*vsize/2);
                    }
                }
                else
                {
                    // rgb = (rgb*255) >> inv_gamma_shift
                    for(int k = 0; k < 12; k++)
                    {
                        i_rgb[k] = v_shr((v_sub((v_shl(i_rgb[k], 8)), i_rgb[k])), inv_gamma_shift);
                    }

                    for(int k = 0; k < 6; k++)
                    {
                        u_rgbvec[k] = v_reinterpret_as_u16(v_pack(i_rgb[k*2+0], i_rgb[k*2+1]));
                    }
                }

                v_uint8 u8_b, u8_g, u8_r;
                u8_r = v_pack(u_rgbvec[0], u_rgbvec[1]);
                u8_g = v_pack(u_rgbvec[2], u_rgbvec[3]);
                u8_b = v_pack(u_rgbvec[4], u_rgbvec[5]);

                if(dcn == 4)
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r, vx_setall_u8(alpha));
                }
                else
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r);
                }
            }
        }
#endif

        for (; i < n; i++, src += 3, dst += dcn)
        {
            int ro, go, bo;
            process(src[0], src[1], src[2], ro, go, bo);

            dst[0] = saturate_cast<uchar>(bo);
            dst[1] = saturate_cast<uchar>(go);
            dst[2] = saturate_cast<uchar>(ro);
            if( dcn == 4 )
                dst[3] = alpha;
        }

    }

    int dstcn;
    int coeffs[9];
    bool issRGB;
};


struct Luv2RGB_b
{
    typedef uchar channel_type;

    Luv2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn),
      fcvt(3, blueIdx, _coeffs, _whitept, _srgb),
      icvt(_dstcn, blueIdx, _coeffs, _whitept, _srgb)
    {
        // whitept is fixed for int calculations
        useBitExactness = (!_whitept && enableBitExactness);
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        CV_INSTRUMENT_REGION();

        if(useBitExactness)
        {
            icvt(src, dst, n);
            return;
        }

        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
#if CV_SIMD
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[3*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
#endif

        static const softfloat fl = softfloat(100)/f255;
        static const softfloat fu = uRange/f255;
        static const softfloat fv = vRange/f255;

#if CV_SIMD
        const int fsize = VTraits<v_float32>::vlanes();
        v_float32 vl = vx_setall_f32((float)fl);
        v_float32 vu = vx_setall_f32((float)fu);
        v_float32 vv = vx_setall_f32((float)fv);
        v_float32 vuLow = vx_setall_f32((float)uLow), vvLow = vx_setall_f32((float)vLow);
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(CV_SIMD_WIDTH) interTmpM[VTraits<v_float32>::max_nlanes*3], interTmpA[VTraits<v_float32>::max_nlanes*3];
        v_store_interleave(interTmpM, vl, vu, vv);
        v_store_interleave(interTmpA, vx_setzero_f32(), vuLow, vvLow);
        v_float32 mluv[3], aluv[3];
        for(int k = 0; k < 3; k++)
        {
            mluv[k] = vx_load_aligned(interTmpM + k*fsize);
            aluv[k] = vx_load_aligned(interTmpA + k*fsize);
        }
#endif

        i = 0;
        for( ; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

#if CV_SIMD
            const int vsize = VTraits<v_uint8>::vlanes();
            for( ; j <= (dn - vsize)*3; j += 3*vsize )
            {
                v_uint8 s0, s1, s2;
                s0 = vx_load(src + j + 0*vsize);
                s1 = vx_load(src + j + 1*vsize);
                s2 = vx_load(src + j + 2*vsize);

                v_uint16 ss[6];
                v_expand(s0, ss[0], ss[1]);
                v_expand(s1, ss[2], ss[3]);
                v_expand(s2, ss[4], ss[5]);
                v_int32 vs[12];
                for(int k = 0; k < 6; k++)
                {
                    v_expand(v_reinterpret_as_s16(ss[k]), vs[k*2+0], vs[k*2+1]);
                }

                for(int bufp = 0; bufp < 12; bufp++)
                {
                    v_store_aligned(buf + j + bufp, v_muladd(v_cvt_f32(vs[bufp]), mluv[bufp%3], aluv[bufp%3]));
                }
            }
#endif
            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*((float)fl);
                buf[j+1] = (float)(src[j+1]*(float)fu + (float)uLow);
                buf[j+2] = (float)(src[j+2]*(float)fv + (float)vLow);
            }

            fcvt(buf, buf, dn);

            j = 0;

#if CV_SIMD
            static const int nBlock = 4*fsize;
            v_float32 v255 = vx_setall_f32(255.f);
            if(dcn == 4)
            {
                v_uint8 valpha = vx_setall_u8(alpha);
                for( ; j <= (dn-nBlock)*3;
                     j += nBlock*3, dst += nBlock)
                {
                    v_float32 vf[4*3];
                    for(int k = 0; k < 4; k++)
                    {
                        v_load_deinterleave(buf + j, vf[k*3+0], vf[k*3+1], vf[k*3+2]);
                    }

                    v_int32 vi[4*3];
                    for(int k = 0; k < 4*3; k++)
                    {
                        vi[k] = v_round(v_mul(vf[k], v255));
                    }

                    v_uint8 rgb[3];
                    for(int k = 0; k < 3; k++)
                    {
                        rgb[k] = v_pack_u(v_pack(vi[0*3+k], vi[1*3+k]),
                                          v_pack(vi[2*3+k], vi[3*3+k]));
                    }

                    v_store_interleave(dst, rgb[0], rgb[1], rgb[2], valpha);
                }
            }
            else // dcn == 3
            {
                for(; j < dn*3 - nBlock; j += nBlock, dst += nBlock)
                {
                    v_float32 vf[4];
                    v_int32 vi[4];
                    for(int k = 0; k < 4; k++)
                    {
                        vf[k] = vx_load_aligned(buf + j + k*fsize);
                        vi[k] = v_round(v_mul(vf[k], v255));
                    }
                    v_store(dst, v_pack_u(v_pack(vi[0], vi[1]), v_pack(vi[2], vi[3])));
                }
            }
#endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Luv2RGBfloat   fcvt;
    Luv2RGBinteger icvt;

    bool useBitExactness;
};

//
// IPP functions
//

#if NEED_IPP

#if !IPP_DISABLE_RGB_XYZ
static ippiGeneralFunc ippiRGB2XYZTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToXYZ_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToXYZ_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToXYZ_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_XYZ_RGB
static ippiGeneralFunc ippiXYZ2RGBTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiXYZToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiXYZToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiXYZToRGB_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_RGB_LAB
static ippiGeneralFunc ippiRGBToLUVTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiRGBToLUV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToLUV_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToLUV_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_LAB_RGB
static ippiGeneralFunc ippiLUVToRGBTab[CV_DEPTH_MAX] =
{
    (ippiGeneralFunc)ippiLUVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiLUVToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiLUVToRGB_32f_C3R, 0, 0
};
#endif

#endif


//
// HAL functions
//

namespace hal
{

void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoXYZ, cv_hal_cvtBGRtoXYZ, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
#if !IPP_DISABLE_RGB_XYZ
    CV_IPP_CHECK()
    {
        if(scn == 3 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                return;
        }
        else if(scn == 4 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                return;
        }
        else if(scn == 3 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                    IPPGeneralFunctor(ippiRGB2XYZTab[depth])) )
                return;
        }
        else if(scn == 4 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 0, 1, 2, depth)) )
                return;
        }
    }
#endif
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_i<uchar>(scn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_i<ushort>(scn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_f<float>(scn, blueIdx, 0));
}


void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtXYZtoBGR, cv_hal_cvtXYZtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
#if !IPP_DISABLE_XYZ_RGB
    CV_IPP_CHECK()
    {
        if(dcn == 3 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                return;
        }
        else if(dcn == 4 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                return;
        }
        if(dcn == 3 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                    IPPGeneralFunctor(ippiXYZ2RGBTab[depth])) )
                return;
        }
        else if(dcn == 4 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                return;
        }
    }
#endif
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_i<uchar>(dcn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_i<ushort>(dcn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_f<float>(dcn, blueIdx, 0));
}


// 8u, 32f
void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoLab, cv_hal_cvtBGRtoLab, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isLab, srgb);

#if defined(HAVE_IPP) && !IPP_DISABLE_RGB_LAB
    CV_IPP_CHECK()
    {
        if (!srgb)
        {
            if (isLab)
            {
                if (scn == 3 && depth == CV_8U && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor((ippiGeneralFunc)ippiBGRToLab_8u_C3R)))
                        return;
                }
                else if (scn == 4 && depth == CV_8U && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 0, 1, 2, depth)))
                        return;
                }
                else if (scn == 3 && depth == CV_8U && swapBlue) // slower than OpenCV
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                                 (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 2, 1, 0, depth)))
                        return;
                }
                else if (scn == 4 && depth == CV_8U && swapBlue) // slower than OpenCV
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 2, 1, 0, depth)))
                        return;
                }
            }
            else
            {
                if (scn == 3 && swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor(ippiRGBToLUVTab[depth])))
                        return;
                }
                else if (scn == 4 && swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 ippiRGBToLUVTab[depth], 0, 1, 2, depth)))
                        return;
                }
                else if (scn == 3 && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                                 ippiRGBToLUVTab[depth], 2, 1, 0, depth)))
                        return;
                }
                else if (scn == 4 && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 ippiRGBToLUVTab[depth], 2, 1, 0, depth)))
                        return;
                }
            }
        }
    }
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if(isLab)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_b(scn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_f(scn, blueIdx, 0, 0, srgb));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Luv_b(scn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Luv_f(scn, blueIdx, 0, 0, srgb));
    }
}


// 8u, 32f
void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtLabtoBGR, cv_hal_cvtLabtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isLab, srgb);

#if defined(HAVE_IPP) && !IPP_DISABLE_LAB_RGB
    CV_IPP_CHECK()
    {
        if (!srgb)
        {
            if (isLab)
            {
                if( dcn == 3 && depth == CV_8U && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R)) )
                        return;
                }
                else if( dcn == 4 && depth == CV_8U && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                                 ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                if( dcn == 3 && depth == CV_8U && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                                 ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( dcn == 4 && depth == CV_8U && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                                 ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
            }
            else
            {
                if( dcn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor(ippiLUVToRGBTab[depth])) )
                        return;
                }
                else if( dcn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                                 ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                if( dcn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                                 ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( dcn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                                 ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
            }
        }
    }
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if(isLab)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Lab2RGB_b(dcn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Lab2RGB_f(dcn, blueIdx, 0, 0, srgb));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Luv2RGB_b(dcn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Luv2RGB_f(dcn, blueIdx, 0, 0, srgb));
    }
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorBGR2Luv( InputArray _src, OutputArray _dst, int bidx, bool srgb)
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("BGR2Luv", ocl::imgproc::color_lab_oclsrc,
                       format("-D DCN=3 -D BIDX=%d%s", bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare additional arguments

    initLabTabs();

    static UMat usRGBGammaTab, ucoeffs, uLabCbrtTab;

    if (srgb && usRGBGammaTab.empty())
        Mat(1, GAMMA_TAB_SIZE * 4, CV_32FC1, const_cast<float*>(sRGBGammaTab)).copyTo(usRGBGammaTab);
    if (uLabCbrtTab.empty())
        Mat(1, LAB_CBRT_TAB_SIZE * 4, CV_32FC1, const_cast<float*>(LabCbrtTab)).copyTo(uLabCbrtTab);

    float coeffs[9];
    softdouble whitePt[3];
    for(int i = 0; i < 3; i++)
        whitePt[i] = D65[i];

    for (int i = 0; i < 3; i++)
    {
        int j = i * 3;

        softfloat c0 = sRGB2XYZ_D65[j    ];
        softfloat c1 = sRGB2XYZ_D65[j + 1];
        softfloat c2 = sRGB2XYZ_D65[j + 2];

        coeffs[j + (bidx ^ 2)] = c0;
        coeffs[j + 1]          = c1;
        coeffs[j + bidx]       = c2;

        CV_Assert( c0 >= 0 && c1 >= 0 && c2 >= 0 &&
                   c0 + c1 + c2 < softfloat(3)/softfloat(2));
    }

    softfloat d = whitePt[0] +
                  whitePt[1]*softdouble(15) +
                  whitePt[2]*softdouble(3);
    d = softfloat::one()/max(d, softfloat(FLT_EPSILON));
    float un = d*softfloat(13*4)*whitePt[0];
    float vn = d*softfloat(13*9)*whitePt[1];

    Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

    ocl::KernelArg ucoeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

    ocl::KernelArg LabCbrtTabarg = ocl::KernelArg::PtrReadOnly(uLabCbrtTab);

    // Setup additional arguments and run

    if(srgb)
        h.setArg(ocl::KernelArg::PtrReadOnly(usRGBGammaTab));

    h.setArg(LabCbrtTabarg);
    h.setArg(ucoeffsarg);
    h.setArg(un); h.setArg(vn);

    return h.run();
}


bool oclCvtColorBGR2Lab( InputArray _src, OutputArray _dst, int bidx, bool srgb )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("BGR2Lab", ocl::imgproc::color_lab_oclsrc,
                       format("-D DCN=3 -D BIDX=%d%s", bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare and set additional arguments

    initLabTabs();

    if (_src.depth() == CV_8U)
    {
        static UMat usRGBGammaTab, ulinearGammaTab, uLabCbrtTab, ucoeffs;

        if (srgb && usRGBGammaTab.empty())
            Mat(1, 256, CV_16UC1, sRGBGammaTab_b).copyTo(usRGBGammaTab);
        else if (ulinearGammaTab.empty())
            Mat(1, 256, CV_16UC1, linearGammaTab_b).copyTo(ulinearGammaTab);
        if (uLabCbrtTab.empty())
            Mat(1, LAB_CBRT_TAB_SIZE_B, CV_16UC1, LabCbrtTab_b).copyTo(uLabCbrtTab);

        int coeffs[9];
        static const softfloat lshift(1 << lab_shift);
        for( int i = 0; i < 3; i++ )
        {
            coeffs[i*3+(bidx^2)] = cvRound(lshift*sRGB2XYZ_D65[i*3  ]/D65[i]);
            coeffs[i*3+1]        = cvRound(lshift*sRGB2XYZ_D65[i*3+1]/D65[i]);
            coeffs[i*3+bidx]     = cvRound(lshift*sRGB2XYZ_D65[i*3+2]/D65[i]);

            CV_Assert(coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift));
        }
        Mat(1, 9, CV_32SC1, coeffs).copyTo(ucoeffs);

        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);

        h.setArg(ocl::KernelArg::PtrReadOnly(srgb ? usRGBGammaTab : ulinearGammaTab));
        h.setArg(ocl::KernelArg::PtrReadOnly(uLabCbrtTab));
        h.setArg(ocl::KernelArg::PtrReadOnly(ucoeffs));
        h.setArg(Lscale); h.setArg(Lshift);
    }
    else
    {
        static UMat usRGBGammaTab, ucoeffs;

        if (srgb && usRGBGammaTab.empty())
            Mat(1, GAMMA_TAB_SIZE * 4, CV_32FC1, const_cast<float*>(sRGBGammaTab)).copyTo(usRGBGammaTab);

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            whitePt[i] = D65[i];

        softdouble scale[] = { softdouble::one() / whitePt[0],
                               softdouble::one(),
                               softdouble::one() / whitePt[2] };

        float coeffs[9];
        for (int i = 0; i < 3; i++)
        {
            int j = i * 3;

            softfloat c0 = scale[i] * sRGB2XYZ_D65[j    ];
            softfloat c1 = scale[i] * sRGB2XYZ_D65[j + 1];
            softfloat c2 = scale[i] * sRGB2XYZ_D65[j + 2];

            coeffs[j + (bidx ^ 2)] = c0;
            coeffs[j + 1]          = c1;
            coeffs[j + bidx]       = c2;

            CV_Assert( c0 >= 0 && c1 >= 0 && c2 >= 0 &&
                       c0 + c1 + c2 < softfloat((int)LAB_CBRT_TAB_SIZE));
        }

        Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

        static const float _a = softfloat(16)/softfloat(116);
        static const float _1_3f = softfloat::one()/softfloat(3);
        ocl::KernelArg ucoeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

        if (srgb)
            h.setArg(ocl::KernelArg::PtrReadOnly(usRGBGammaTab));

        h.setArg(ucoeffsarg);
        h.setArg(_1_3f); h.setArg(_a);
    }

    return h.run();
}


bool oclCvtColorLab2BGR(InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb)
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("Lab2BGR", ocl::imgproc::color_lab_oclsrc,
                       format("-D DCN=%d -D BIDX=%d%s", dcn, bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare additional arguments

    initLabTabs();

    static UMat ucoeffs, usRGBInvGammaTab;

    if (srgb && usRGBInvGammaTab.empty())
        Mat(1, GAMMA_TAB_SIZE*4, CV_32FC1, const_cast<float*>(sRGBInvGammaTab)).copyTo(usRGBInvGammaTab);

    float coeffs[9];
    softdouble whitePt[3];
    for(int i = 0; i < 3; i++)
        whitePt[i] = D65[i];

    for( int i = 0; i < 3; i++ )
    {
        coeffs[i+(bidx^2)*3] = (float)(XYZ2sRGB_D65[i  ]*whitePt[i]);
        coeffs[i+3]          = (float)(XYZ2sRGB_D65[i+3]*whitePt[i]);
        coeffs[i+bidx*3]     = (float)(XYZ2sRGB_D65[i+6]*whitePt[i]);
    }

    Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

    float lThresh = softfloat(8); // 0.008856f * 903.3f  = (6/29)^3*(29/3)^3 = 8
    float fThresh = softfloat(6)/softfloat(29); // 7.787f * 0.008856f + 16.0f / 116.0f = 6/29

    ocl::KernelArg coeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

    // Set additional arguments and run

    if(srgb)
        h.setArg(ocl::KernelArg::PtrReadOnly(usRGBInvGammaTab));

    h.setArg(coeffsarg);
    h.setArg(lThresh);
    h.setArg(fThresh);

    return h.run();
}


bool oclCvtColorLuv2BGR(InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb)
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("Luv2BGR", ocl::imgproc::color_lab_oclsrc,
                       format("-D DCN=%d -D BIDX=%d%s", dcn, bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare additional arguments

    initLabTabs();

    static UMat ucoeffs, usRGBInvGammaTab;

    if (srgb && usRGBInvGammaTab.empty())
        Mat(1, GAMMA_TAB_SIZE*4, CV_32FC1, const_cast<float*>(sRGBInvGammaTab)).copyTo(usRGBInvGammaTab);

    float coeffs[9];
    softdouble whitePt[3];
    for(int i = 0; i < 3; i++)
        whitePt[i] = D65[i];

    for( int i = 0; i < 3; i++ )
    {
        coeffs[i+(bidx^2)*3] = (float)(XYZ2sRGB_D65[i  ]);
        coeffs[i+3]          = (float)(XYZ2sRGB_D65[i+3]);
        coeffs[i+bidx*3]     = (float)(XYZ2sRGB_D65[i+6]);
    }

    softfloat d = whitePt[0] +
            whitePt[1]*softdouble(15) +
            whitePt[2]*softdouble(3);
    d = softfloat::one()/max(d, softfloat(FLT_EPSILON));
    float un = softfloat(4*13)*d*whitePt[0];
    float vn = softfloat(9*13)*d*whitePt[1];

    Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

    ocl::KernelArg coeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

    // Set additional arguments and run

    if(srgb)
        h.setArg(ocl::KernelArg::PtrReadOnly(usRGBInvGammaTab));

    h.setArg(coeffsarg);
    h.setArg(un); h.setArg(vn);

    return h.run();
}


bool oclCvtColorBGR2XYZ( InputArray _src, OutputArray _dst, int bidx )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("RGB2XYZ", ocl::imgproc::color_lab_oclsrc,
                       format("-D DCN=3 -D BIDX=%d", bidx)))
    {
        return false;
    }

    // Prepare additional arguments

    UMat c;
    if (_src.depth() == CV_32F)
    {
        float coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = (float)sRGB2XYZ_D65[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
        Mat(1, 9, CV_32FC1, &coeffs[0]).copyTo(c);
    }
    else
    {
        int coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = sRGB2XYZ_D65_i[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
        Mat(1, 9, CV_32SC1, &coeffs[0]).copyTo(c);
    }

    // Set additional arguments and run

    h.setArg(ocl::KernelArg::PtrReadOnly(c));

    return h.run();
}


bool oclCvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("XYZ2RGB", ocl::imgproc::color_lab_oclsrc,
                       format("-D DCN=%d -D BIDX=%d", dcn, bidx)))
    {
        return false;
    }

    // Prepare additional arguments

    UMat c;
    if (_src.depth() == CV_32F)
    {
        float coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = (float)XYZ2sRGB_D65[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
        Mat(1, 9, CV_32FC1, &coeffs[0]).copyTo(c);
    }
    else
    {
        int coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = XYZ2sRGB_D65_i[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
        Mat(1, 9, CV_32SC1, &coeffs[0]).copyTo(c);
    }

    // Set additional arguments and run

    h.setArg(ocl::KernelArg::PtrReadOnly(c));

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2Lab( InputArray _src, OutputArray _dst, bool swapb, bool srgb)
{
    CvtHelper<Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoLab(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, true, srgb);
}


void cvtColorBGR2Luv( InputArray _src, OutputArray _dst, bool swapb, bool srgb)
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoLab(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, false, srgb);
}


void cvtColorLab2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb )
{
    if( dcn <= 0 ) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtLabtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, true, srgb);
}


void cvtColorLuv2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb )
{
    if( dcn <= 0 ) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtLabtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, false, srgb);
}


void cvtColorBGR2XYZ( InputArray _src, OutputArray _dst, bool swapb )
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoXYZ(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, h.depth, h.scn, swapb);
}


void cvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb )
{
    if( dcn <= 0 ) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtXYZtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, h.depth, dcn, swapb);
}

} // namespace cv
