// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_VSX_HPP
#define OPENCV_HAL_VSX_HPP

#include <algorithm>
#include "opencv2/core/utility.hpp"

#define CV_SIMD128 1
#define CV_SIMD128_64F 1

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Types ////////////

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };
    vec_uchar16 val;

    explicit v_uint8x16(const vec_uchar16& v) : val(v)
    {}
    v_uint8x16()
    {}
    v_uint8x16(vec_bchar16 v) : val(vec_uchar16_c(v))
    {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
        : val(vec_uchar16_set(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15))
    {}

    static inline v_uint8x16 zero() { return v_uint8x16(vec_uchar16_z); }

    uchar get0() const
    { return vec_extract(val, 0); }
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };
    vec_char16 val;

    explicit v_int8x16(const vec_char16& v) : val(v)
    {}
    v_int8x16()
    {}
    v_int8x16(vec_bchar16 v) : val(vec_char16_c(v))
    {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
              schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
        : val(vec_char16_set(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15))
    {}

    static inline v_int8x16 zero() { return v_int8x16(vec_char16_z); }

    schar get0() const
    { return vec_extract(val, 0); }
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };
    vec_ushort8 val;

    explicit v_uint16x8(const vec_ushort8& v) : val(v)
    {}
    v_uint16x8()
    {}
    v_uint16x8(vec_bshort8 v) : val(vec_ushort8_c(v))
    {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
        : val(vec_ushort8_set(v0, v1, v2, v3, v4, v5, v6, v7))
    {}

    static inline v_uint16x8 zero() { return v_uint16x8(vec_ushort8_z); }

    ushort get0() const
    { return vec_extract(val, 0); }
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };
    vec_short8 val;

    explicit v_int16x8(const vec_short8& v) : val(v)
    {}
    v_int16x8()
    {}
    v_int16x8(vec_bshort8 v) : val(vec_short8_c(v))
    {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
        : val(vec_short8_set(v0, v1, v2, v3, v4, v5, v6, v7))
    {}

    static inline v_int16x8 zero() { return v_int16x8(vec_short8_z); }

    short get0() const
    { return vec_extract(val, 0); }
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };
    vec_uint4 val;

    explicit v_uint32x4(const vec_uint4& v) : val(v)
    {}
    v_uint32x4()
    {}
    v_uint32x4(vec_bint4 v) : val(vec_uint4_c(v))
    {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3) : val(vec_uint4_set(v0, v1, v2, v3))
    {}

    static inline v_uint32x4 zero() { return v_uint32x4(vec_uint4_z); }

    uint get0() const
    { return vec_extract(val, 0); }
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };
    vec_int4 val;

    explicit v_int32x4(const vec_int4& v) : val(v)
    {}
    v_int32x4()
    {}
    v_int32x4(vec_bint4 v) : val(vec_int4_c(v))
    {}
    v_int32x4(int v0, int v1, int v2, int v3) : val(vec_int4_set(v0, v1, v2, v3))
    {}

    static inline v_int32x4 zero() { return v_int32x4(vec_int4_z); }

    int get0() const
    { return vec_extract(val, 0); }
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };
    vec_float4 val;

    explicit v_float32x4(const vec_float4& v) : val(v)
    {}
    v_float32x4()
    {}
    v_float32x4(vec_bint4 v) : val(vec_float4_c(v))
    {}
    v_float32x4(float v0, float v1, float v2, float v3) : val(vec_float4_set(v0, v1, v2, v3))
    {}

    static inline v_float32x4 zero() { return v_float32x4(vec_float4_z); }

    float get0() const
    { return vec_extract(val, 0); }
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };
    vec_udword2 val;

    explicit v_uint64x2(const vec_udword2& v) : val(v)
    {}
    v_uint64x2()
    {}
    v_uint64x2(vec_bdword2 v) : val(vec_udword2_c(v))
    {}
    v_uint64x2(uint64 v0, uint64 v1) : val(vec_udword2_set(v0, v1))
    {}

    static inline v_uint64x2 zero() { return v_uint64x2(vec_udword2_z); }

    uint64 get0() const
    { return vec_extract(val, 0); }
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };
    vec_dword2 val;

    explicit v_int64x2(const vec_dword2& v) : val(v)
    {}
    v_int64x2()
    {}
    v_int64x2(vec_bdword2 v) : val(vec_dword2_c(v))
    {}
    v_int64x2(int64 v0, int64 v1) : val(vec_dword2_set(v0, v1))
    {}

    static inline v_int64x2 zero() { return v_int64x2(vec_dword2_z); }

    int64 get0() const
    { return vec_extract(val, 0); }
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };
    vec_double2 val;

    explicit v_float64x2(const vec_double2& v) : val(v)
    {}
    v_float64x2()
    {}
    v_float64x2(vec_bdword2 v) : val(vec_double2_c(v))
    {}
    v_float64x2(double v0, double v1) : val(vec_double2_set(v0, v1))
    {}

    static inline v_float64x2 zero() { return v_float64x2(vec_double2_z); }

    double get0() const
    { return vec_extract(val, 0); }
};

#define OPENCV_HAL_IMPL_VSX_EXTRACT_N(_Tpvec, _Tp) \
template<int i> inline _Tp v_extract_n(VSX_UNUSED(_Tpvec v)) { return vec_extract(v.val, i); }

OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_uint8x16, uchar)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_int8x16, schar)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_uint16x8, ushort)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_int16x8, short)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_uint32x4, uint)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_int32x4, int)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_uint64x2, uint64)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_int64x2, int64)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_float32x4, float)
OPENCV_HAL_IMPL_VSX_EXTRACT_N(v_float64x2, double)

//////////////// Load and store operations ///////////////

/*
 * clang-5 aborted during parse "vec_xxx_c" only if it's
 * inside a function template which is defined by preprocessor macro.
 *
 * if vec_xxx_c defined as C++ cast, clang-5 will pass it
*/
#define OPENCV_HAL_IMPL_VSX_INITVEC(_Tpvec, _Tp, suffix, cast)                        \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(vec_splats((_Tp)0)); }             \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(vec_splats((_Tp)v));}          \
template <> inline _Tpvec v_setzero_() { return v_setzero_##suffix(); }               \
template <> inline _Tpvec v_setall_(_Tp v) { return v_setall_##suffix(v); }       \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0 &a)  \
{ return _Tpvec((cast)a.val); }

OPENCV_HAL_IMPL_VSX_INITVEC(v_uint8x16, uchar, u8, vec_uchar16)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int8x16, schar, s8, vec_char16)
OPENCV_HAL_IMPL_VSX_INITVEC(v_uint16x8, ushort, u16, vec_ushort8)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int16x8, short, s16, vec_short8)
OPENCV_HAL_IMPL_VSX_INITVEC(v_uint32x4, uint, u32, vec_uint4)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int32x4, int, s32, vec_int4)
OPENCV_HAL_IMPL_VSX_INITVEC(v_uint64x2, uint64, u64, vec_udword2)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int64x2, int64, s64, vec_dword2)
OPENCV_HAL_IMPL_VSX_INITVEC(v_float32x4, float, f32, vec_float4)
OPENCV_HAL_IMPL_VSX_INITVEC(v_float64x2, double, f64, vec_double2)

#define OPENCV_HAL_IMPL_VSX_LOADSTORE_C(_Tpvec, _Tp, ld, ld_a, st, st_a)    \
inline _Tpvec v_load(const _Tp* ptr)                                        \
{ return _Tpvec(ld(0, ptr)); }                                              \
inline _Tpvec v_load_aligned(VSX_UNUSED(const _Tp* ptr))                    \
{ return _Tpvec(ld_a(0, ptr)); }                                            \
inline _Tpvec v_load_low(const _Tp* ptr)                                    \
{ return _Tpvec(vec_ld_l8(ptr)); }                                          \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1)               \
{ return _Tpvec(vec_mergesqh(vec_ld_l8(ptr0), vec_ld_l8(ptr1))); }          \
inline void v_store(_Tp* ptr, const _Tpvec& a)                              \
{ st(a.val, 0, ptr); }                                                      \
inline void v_store_aligned(VSX_UNUSED(_Tp* ptr), const _Tpvec& a)          \
{ st_a(a.val, 0, ptr); }                                                    \
inline void v_store_aligned_nocache(VSX_UNUSED(_Tp* ptr), const _Tpvec& a)  \
{ st_a(a.val, 0, ptr); }                                                    \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode)         \
{ if(mode == hal::STORE_UNALIGNED) st(a.val, 0, ptr); else st_a(a.val, 0, ptr); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a)                          \
{ vec_st_l8(a.val, ptr); }                                                  \
inline void v_store_high(_Tp* ptr, const _Tpvec& a)                         \
{ vec_st_h8(a.val, ptr); }

// working around gcc bug for aligned ld/st
// if runtime check for vec_ld/st fail we failback to unaligned ld/st
// https://github.com/opencv/opencv/issues/13211
#ifdef CV_COMPILER_VSX_BROKEN_ALIGNED
    #define OPENCV_HAL_IMPL_VSX_LOADSTORE(_Tpvec, _Tp) \
    OPENCV_HAL_IMPL_VSX_LOADSTORE_C(_Tpvec, _Tp, vsx_ld, vsx_ld, vsx_st, vsx_st)
#else
    #define OPENCV_HAL_IMPL_VSX_LOADSTORE(_Tpvec, _Tp) \
    OPENCV_HAL_IMPL_VSX_LOADSTORE_C(_Tpvec, _Tp, vsx_ld, vec_ld, vsx_st, vec_st)
#endif

OPENCV_HAL_IMPL_VSX_LOADSTORE(v_uint8x16,  uchar)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_int8x16,   schar)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_uint16x8,  ushort)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_int16x8,   short)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_uint32x4,  uint)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_int32x4,   int)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_float32x4, float)

OPENCV_HAL_IMPL_VSX_LOADSTORE_C(v_float64x2, double, vsx_ld,  vsx_ld,  vsx_st,  vsx_st)
OPENCV_HAL_IMPL_VSX_LOADSTORE_C(v_uint64x2,  uint64, vsx_ld2, vsx_ld2, vsx_st2, vsx_st2)
OPENCV_HAL_IMPL_VSX_LOADSTORE_C(v_int64x2,    int64, vsx_ld2, vsx_ld2, vsx_st2, vsx_st2)

//////////////// Value reordering ///////////////

/* de&interleave */
#define OPENCV_HAL_IMPL_VSX_INTERLEAVE(_Tp, _Tpvec)                          \
inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a, _Tpvec& b)        \
{ vec_ld_deinterleave(ptr, a.val, b.val);}                                   \
inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a,                   \
                                _Tpvec& b, _Tpvec& c)                        \
{ vec_ld_deinterleave(ptr, a.val, b.val, c.val); }                           \
inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a, _Tpvec& b,        \
                                                _Tpvec& c, _Tpvec& d)        \
{ vec_ld_deinterleave(ptr, a.val, b.val, c.val, d.val); }                    \
inline void v_store_interleave(_Tp* ptr, const _Tpvec& a, const _Tpvec& b,   \
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ vec_st_interleave(a.val, b.val, ptr); }                                    \
inline void v_store_interleave(_Tp* ptr, const _Tpvec& a,                    \
                               const _Tpvec& b, const _Tpvec& c,             \
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ vec_st_interleave(a.val, b.val, c.val, ptr); }                             \
inline void v_store_interleave(_Tp* ptr, const _Tpvec& a, const _Tpvec& b,   \
                                         const _Tpvec& c, const _Tpvec& d,   \
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ vec_st_interleave(a.val, b.val, c.val, d.val, ptr); }

OPENCV_HAL_IMPL_VSX_INTERLEAVE(uchar, v_uint8x16)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(schar, v_int8x16)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(ushort, v_uint16x8)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(short, v_int16x8)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(uint, v_uint32x4)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(int, v_int32x4)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(float, v_float32x4)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(double, v_float64x2)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(int64, v_int64x2)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(uint64, v_uint64x2)

/* Expand */
#define OPENCV_HAL_IMPL_VSX_EXPAND(_Tpvec, _Tpwvec, _Tp, fl, fh)  \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1)   \
{                                                                 \
    b0.val = fh(a.val);                                           \
    b1.val = fl(a.val);                                           \
}                                                                 \
inline _Tpwvec v_expand_low(const _Tpvec& a)                      \
{ return _Tpwvec(fh(a.val)); }                                    \
inline _Tpwvec v_expand_high(const _Tpvec& a)                     \
{ return _Tpwvec(fl(a.val)); }                                    \
inline _Tpwvec v_load_expand(const _Tp* ptr)                      \
{ return _Tpwvec(fh(vec_ld_l8(ptr))); }

OPENCV_HAL_IMPL_VSX_EXPAND(v_uint8x16, v_uint16x8, uchar, vec_unpacklu, vec_unpackhu)
OPENCV_HAL_IMPL_VSX_EXPAND(v_int8x16, v_int16x8, schar, vec_unpackl, vec_unpackh)
OPENCV_HAL_IMPL_VSX_EXPAND(v_uint16x8, v_uint32x4, ushort, vec_unpacklu, vec_unpackhu)
OPENCV_HAL_IMPL_VSX_EXPAND(v_int16x8, v_int32x4, short, vec_unpackl, vec_unpackh)
OPENCV_HAL_IMPL_VSX_EXPAND(v_uint32x4, v_uint64x2, uint, vec_unpacklu, vec_unpackhu)
OPENCV_HAL_IMPL_VSX_EXPAND(v_int32x4, v_int64x2, int, vec_unpackl, vec_unpackh)

/* Load and zero expand a 4 byte value into the second dword, first is don't care. */
#if !defined(CV_COMPILER_VSX_BROKEN_ASM)
    #define _LXSIWZX(out, ptr, T) __asm__ ("lxsiwzx %x0, 0, %1\r\n" : "=wa"(out) : "r" (ptr) : "memory");
#else
    /* This is compiler-agnostic, but will introduce an unneeded splat on the critical path. */
    #define _LXSIWZX(out, ptr, T) out = (T)vec_udword2_sp(*(uint32_t*)(ptr));
#endif

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    // Zero-extend the extra 24B instead of unpacking. Usually faster in small kernel
    // Likewise note, value is zero extended and upper 4 bytes are zero'ed.
    vec_uchar16 pmu = {8, 12, 12, 12, 9, 12, 12, 12, 10, 12, 12, 12, 11, 12, 12, 12};
    vec_uchar16 out;

    _LXSIWZX(out, ptr, vec_uchar16);
    out = vec_perm(out, out, pmu);
    return v_uint32x4((vec_uint4)out);
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    vec_char16 out;
    vec_short8 outs;
    vec_int4 outw;

    _LXSIWZX(out, ptr, vec_char16);
    outs = vec_unpackl(out);
    outw = vec_unpackh(outs);
    return v_int32x4(outw);
}

/* pack */
#define OPENCV_HAL_IMPL_VSX_PACK(_Tpvec, _Tp, _Tpwvec, _Tpvn, _Tpdel, sfnc, pkfnc, addfnc, pack)    \
inline _Tpvec v_##pack(const _Tpwvec& a, const _Tpwvec& b)                                          \
{                                                                                                   \
    return _Tpvec(pkfnc(a.val, b.val));                                                             \
}                                                                                                   \
inline void v_##pack##_store(_Tp* ptr, const _Tpwvec& a)                                            \
{                                                                                                   \
    vec_st_l8(pkfnc(a.val, a.val), ptr);                                                            \
}                                                                                                   \
template<int n>                                                                                     \
inline _Tpvec v_rshr_##pack(const _Tpwvec& a, const _Tpwvec& b)                                     \
{                                                                                                   \
    const __vector _Tpvn vn = vec_splats((_Tpvn)n);                                                 \
    const __vector _Tpdel delta = vec_splats((_Tpdel)((_Tpdel)1 << (n-1)));                         \
    return _Tpvec(pkfnc(sfnc(addfnc(a.val, delta), vn), sfnc(addfnc(b.val, delta), vn)));           \
}                                                                                                   \
template<int n>                                                                                     \
inline void v_rshr_##pack##_store(_Tp* ptr, const _Tpwvec& a)                                       \
{                                                                                                   \
    const __vector _Tpvn vn = vec_splats((_Tpvn)n);                                                 \
    const __vector _Tpdel delta = vec_splats((_Tpdel)((_Tpdel)1 << (n-1)));                         \
    vec_st_l8(pkfnc(sfnc(addfnc(a.val, delta), vn), delta), ptr);                                   \
}

OPENCV_HAL_IMPL_VSX_PACK(v_uint8x16, uchar, v_uint16x8, unsigned short, unsigned short,
                         vec_sr, vec_packs, vec_adds, pack)
OPENCV_HAL_IMPL_VSX_PACK(v_int8x16, schar, v_int16x8, unsigned short, short,
                         vec_sra, vec_packs, vec_adds, pack)

OPENCV_HAL_IMPL_VSX_PACK(v_uint16x8, ushort, v_uint32x4, unsigned int, unsigned int,
                         vec_sr, vec_packs, vec_add, pack)
OPENCV_HAL_IMPL_VSX_PACK(v_int16x8, short, v_int32x4, unsigned int, int,
                         vec_sra, vec_packs, vec_add, pack)

OPENCV_HAL_IMPL_VSX_PACK(v_uint32x4, uint, v_uint64x2, unsigned long long, unsigned long long,
                         vec_sr, vec_pack, vec_add, pack)
OPENCV_HAL_IMPL_VSX_PACK(v_int32x4, int, v_int64x2, unsigned long long, long long,
                         vec_sra, vec_pack, vec_add, pack)

OPENCV_HAL_IMPL_VSX_PACK(v_uint8x16, uchar, v_int16x8, unsigned short, short,
                         vec_sra, vec_packsu, vec_adds, pack_u)
OPENCV_HAL_IMPL_VSX_PACK(v_uint16x8, ushort, v_int32x4, unsigned int, int,
                         vec_sra, vec_packsu, vec_add, pack_u)
// Following variant is not implemented on other platforms:
//OPENCV_HAL_IMPL_VSX_PACK(v_uint32x4, uint, v_int64x2, unsigned long long, long long,
//                         vec_sra, vec_packsu, vec_add, pack_u)

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    vec_uchar16 ab = vec_pack(a.val, b.val);
    return v_uint8x16(ab);
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    vec_ushort8 ab = vec_pack(a.val, b.val);
    vec_ushort8 cd = vec_pack(c.val, d.val);
    return v_uint8x16(vec_pack(ab, cd));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    vec_uint4 ab = vec_pack(a.val, b.val);
    vec_uint4 cd = vec_pack(c.val, d.val);
    vec_uint4 ef = vec_pack(e.val, f.val);
    vec_uint4 gh = vec_pack(g.val, h.val);

    vec_ushort8 abcd = vec_pack(ab, cd);
    vec_ushort8 efgh = vec_pack(ef, gh);
    return v_uint8x16(vec_pack(abcd, efgh));
}

/* Recombine */
template <typename _Tpvec>
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1)
{
    b0.val = vec_mergeh(a0.val, a1.val);
    b1.val = vec_mergel(a0.val, a1.val);
}

template <typename _Tpvec>
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(vec_mergesql(a.val, b.val)); }

template <typename _Tpvec>
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(vec_mergesqh(a.val, b.val)); }

template <typename _Tpvec>
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d)
{
    c.val = vec_mergesqh(a.val, b.val);
    d.val = vec_mergesql(a.val, b.val);
}

////////// Arithmetic, bitwise and comparison operations /////////

/* Element-wise binary and unary operations */
/** Arithmetics **/
#define OPENCV_HAL_IMPL_VSX_BIN_OP(bin_op, _Tpvec, intrin)       \
inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_uint8x16, vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_uint8x16, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_int8x16,  vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_int8x16, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_uint16x8, vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_uint16x8, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_int16x8, vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_int16x8, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_uint32x4, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_uint32x4, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_mul, v_uint32x4, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_int32x4, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_int32x4, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_mul, v_int32x4, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_float32x4, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_float32x4, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_mul, v_float32x4, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_div, v_float32x4, vec_div)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_float64x2, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_float64x2, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_mul, v_float64x2, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_div, v_float64x2, vec_div)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_uint64x2, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_uint64x2, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_add, v_int64x2, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(v_sub, v_int64x2, vec_sub)

// saturating multiply
#define OPENCV_HAL_IMPL_VSX_MUL_SAT(_Tpvec, _Tpwvec)             \
    inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b)        \
    {                                                            \
        _Tpwvec c, d;                                            \
        v_mul_expand(a, b, c, d);                                \
        return v_pack(c, d);                                     \
    }

OPENCV_HAL_IMPL_VSX_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_VSX_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_VSX_MUL_SAT(v_int16x8,  v_int32x4)
OPENCV_HAL_IMPL_VSX_MUL_SAT(v_uint16x8, v_uint32x4)

template<typename Tvec, typename Twvec>
inline void v_mul_expand(const Tvec& a, const Tvec& b, Twvec& c, Twvec& d)
{
    Twvec p0 = Twvec(vec_mule(a.val, b.val));
    Twvec p1 = Twvec(vec_mulo(a.val, b.val));
    v_zip(p0, p1, c, d);
}

inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    vec_int4 p0 = vec_mule(a.val, b.val);
    vec_int4 p1 = vec_mulo(a.val, b.val);
    static const vec_uchar16 perm = {2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
    return v_int16x8(vec_perm(vec_short8_c(p0), vec_short8_c(p1), perm));
}
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    vec_uint4 p0 = vec_mule(a.val, b.val);
    vec_uint4 p1 = vec_mulo(a.val, b.val);
    static const vec_uchar16 perm = {2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
    return v_uint16x8(vec_perm(vec_ushort8_c(p0), vec_ushort8_c(p1), perm));
}

/** Non-saturating arithmetics **/
#define OPENCV_HAL_IMPL_VSX_BIN_FUNC(func, intrin)    \
template<typename _Tpvec>                             \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b)  \
{ return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_add_wrap, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_sub_wrap, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_mul_wrap, vec_mul)

/** Bitwise shifts **/
#define OPENCV_HAL_IMPL_VSX_SHIFT_OP(_Tpvec, shr, splfunc)   \
inline _Tpvec v_shl(const _Tpvec& a, int imm)                \
{ return _Tpvec(vec_sl(a.val, splfunc(imm))); }              \
inline _Tpvec v_shr(const _Tpvec& a, int imm)                \
{ return _Tpvec(shr(a.val, splfunc(imm))); }                 \
template<int imm> inline _Tpvec v_shl(const _Tpvec& a)       \
{ return _Tpvec(vec_sl(a.val, splfunc(imm))); }              \
template<int imm> inline _Tpvec v_shr(const _Tpvec& a)       \
{ return _Tpvec(shr(a.val, splfunc(imm))); }

OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint8x16, vec_sr, vec_uchar16_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint16x8, vec_sr, vec_ushort8_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint32x4, vec_sr, vec_uint4_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint64x2, vec_sr, vec_udword2_sp)
// algebraic right shift
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int8x16, vec_sra, vec_uchar16_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int16x8, vec_sra, vec_ushort8_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int32x4, vec_sra, vec_uint4_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int64x2, vec_sra, vec_udword2_sp)

/** Bitwise logic **/
#define OPENCV_HAL_IMPL_VSX_LOGIC_OP(_Tpvec)    \
OPENCV_HAL_IMPL_VSX_BIN_OP(v_and, _Tpvec, vec_and)  \
OPENCV_HAL_IMPL_VSX_BIN_OP(v_or, _Tpvec, vec_or)    \
OPENCV_HAL_IMPL_VSX_BIN_OP(v_xor, _Tpvec, vec_xor)  \
inline _Tpvec v_not(const _Tpvec& a)                \
{ return _Tpvec(vec_not(a.val)); }

OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint8x16)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int8x16)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint16x8)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int16x8)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint32x4)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int32x4)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint64x2)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int64x2)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_float32x4)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_float64x2)

/** Bitwise select **/
#define OPENCV_HAL_IMPL_VSX_SELECT(_Tpvec, cast)                             \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vec_sel(b.val, a.val, cast(mask.val))); }

OPENCV_HAL_IMPL_VSX_SELECT(v_uint8x16, vec_bchar16_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_int8x16, vec_bchar16_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_uint16x8, vec_bshort8_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_int16x8, vec_bshort8_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_uint32x4, vec_bint4_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_int32x4, vec_bint4_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_float32x4, vec_bint4_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_float64x2, vec_bdword2_c)

/** Comparison **/
#define OPENCV_HAL_IMPL_VSX_INT_CMP_OP(_Tpvec)                 \
inline _Tpvec v_eq(const _Tpvec& a, const _Tpvec& b)           \
{ return _Tpvec(vec_cmpeq(a.val, b.val)); }                    \
inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b)           \
{ return _Tpvec(vec_cmpne(a.val, b.val)); }                    \
inline _Tpvec v_lt(const _Tpvec& a, const _Tpvec& b)           \
{ return _Tpvec(vec_cmplt(a.val, b.val)); }                    \
inline _Tpvec v_gt(const _Tpvec& a, const _Tpvec& b)           \
{ return _Tpvec(vec_cmpgt(a.val, b.val)); }                    \
inline _Tpvec v_le(const _Tpvec& a, const _Tpvec& b)           \
{ return _Tpvec(vec_cmple(a.val, b.val)); }                    \
inline _Tpvec v_ge(const _Tpvec& a, const _Tpvec& b)           \
{ return _Tpvec(vec_cmpge(a.val, b.val)); }

OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint8x16)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int8x16)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint16x8)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int16x8)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint32x4)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int32x4)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_float32x4)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_float64x2)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint64x2)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int64x2)

inline v_float32x4 v_not_nan(const v_float32x4& a)
{ return v_float32x4(vec_cmpeq(a.val, a.val)); }
inline v_float64x2 v_not_nan(const v_float64x2& a)
{ return v_float64x2(vec_cmpeq(a.val, a.val)); }

/** min/max **/
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_min, vec_min)
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_max, vec_max)

/** Rotate **/
#define OPENCV_IMPL_VSX_ROTATE(_Tpvec, suffix, shf, cast)                       \
template<int imm>                                                               \
inline _Tpvec v_rotate_##suffix(const _Tpvec& a)                                \
{                                                                               \
    const int wd = imm * sizeof(typename _Tpvec::lane_type);                    \
    if (wd > 15)                                                                \
        return _Tpvec::zero();                                                  \
    return _Tpvec((cast)shf(vec_uchar16_c(a.val), vec_uchar16_sp(wd << 3)));    \
}

#define OPENCV_IMPL_VSX_ROTATE_LR(_Tpvec, cast)     \
OPENCV_IMPL_VSX_ROTATE(_Tpvec, left, vec_slo, cast) \
OPENCV_IMPL_VSX_ROTATE(_Tpvec, right, vec_sro, cast)

OPENCV_IMPL_VSX_ROTATE_LR(v_uint8x16, vec_uchar16)
OPENCV_IMPL_VSX_ROTATE_LR(v_int8x16,  vec_char16)
OPENCV_IMPL_VSX_ROTATE_LR(v_uint16x8, vec_ushort8)
OPENCV_IMPL_VSX_ROTATE_LR(v_int16x8,  vec_short8)
OPENCV_IMPL_VSX_ROTATE_LR(v_uint32x4, vec_uint4)
OPENCV_IMPL_VSX_ROTATE_LR(v_int32x4,  vec_int4)
OPENCV_IMPL_VSX_ROTATE_LR(v_float32x4, vec_float4)
OPENCV_IMPL_VSX_ROTATE_LR(v_uint64x2, vec_udword2)
OPENCV_IMPL_VSX_ROTATE_LR(v_int64x2,  vec_dword2)
OPENCV_IMPL_VSX_ROTATE_LR(v_float64x2, vec_double2)

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b)
{
    enum { CV_SHIFT = 16 - imm * (sizeof(typename _Tpvec::lane_type)) };
    if (CV_SHIFT == 16)
        return a;
#ifdef __IBMCPP__
    return _Tpvec(vec_sld(b.val, a.val, CV_SHIFT & 15));
#else
    return _Tpvec(vec_sld(b.val, a.val, CV_SHIFT));
#endif
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b)
{
    enum { CV_SHIFT = imm * (sizeof(typename _Tpvec::lane_type)) };
    if (CV_SHIFT == 16)
        return b;
    return _Tpvec(vec_sld(a.val, b.val, CV_SHIFT));
}

#define OPENCV_IMPL_VSX_ROTATE_64_2RG(_Tpvec, suffix, rg1, rg2)   \
template<int imm>                                                 \
inline _Tpvec v_rotate_##suffix(const _Tpvec& a, const _Tpvec& b) \
{                                                                 \
    if (imm == 1)                                                 \
        return _Tpvec(vec_permi(rg1.val, rg2.val, 2));            \
    return imm ? b : a;                                           \
}

#define OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(_Tpvec)    \
OPENCV_IMPL_VSX_ROTATE_64_2RG(_Tpvec, left,  b, a)  \
OPENCV_IMPL_VSX_ROTATE_64_2RG(_Tpvec, right, a, b)

OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(v_float64x2)
OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(v_uint64x2)
OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(v_int64x2)

/* Reverse */
inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
    static const vec_uchar16 perm = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    vec_uchar16 vec = (vec_uchar16)a.val;
    return v_uint8x16(vec_perm(vec, vec, perm));
}

inline v_int8x16 v_reverse(const v_int8x16 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
    static const vec_uchar16 perm = {14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1};
    vec_uchar16 vec = (vec_uchar16)a.val;
    return v_reinterpret_as_u16(v_uint8x16(vec_perm(vec, vec, perm)));
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{
    static const vec_uchar16 perm = {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3};
    vec_uchar16 vec = (vec_uchar16)a.val;
    return v_reinterpret_as_u32(v_uint8x16(vec_perm(vec, vec, perm)));
}

inline v_int32x4 v_reverse(const v_int32x4 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{
    static const vec_uchar16 perm = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};
    vec_uchar16 vec = (vec_uchar16)a.val;
    return v_reinterpret_as_u64(v_uint8x16(vec_perm(vec, vec, perm)));
}

inline v_int64x2 v_reverse(const v_int64x2 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x2 v_reverse(const v_float64x2 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }

/* Extract */
template<int s, typename _Tpvec>
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)
{ return v_rotate_right<s>(a, b); }

////////// Reduce and mask /////////

/** Reduce **/
inline uint v_reduce_sum(const v_uint8x16& a)
{
    const vec_uint4 zero4 = vec_uint4_z;
    vec_uint4 sum4 = vec_sum4s(a.val, zero4);
    return (uint)vec_extract(vec_sums(vec_int4_c(sum4), vec_int4_c(zero4)), 3);
}
inline int v_reduce_sum(const v_int8x16& a)
{
    const vec_int4 zero4 = vec_int4_z;
    vec_int4 sum4 = vec_sum4s(a.val, zero4);
    return (int)vec_extract(vec_sums(sum4, zero4), 3);
}
inline int v_reduce_sum(const v_int16x8& a)
{
    const vec_int4 zero = vec_int4_z;
    return saturate_cast<int>(vec_extract(vec_sums(vec_sum4s(a.val, zero), zero), 3));
}
inline uint v_reduce_sum(const v_uint16x8& a)
{
    const vec_int4 v4 = vec_int4_c(vec_unpackhu(vec_adds(a.val, vec_sld(a.val, a.val, 8))));
    return saturate_cast<uint>(vec_extract(vec_sums(v4, vec_int4_z), 3));
}

#define OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(_Tpvec, _Tpvec2, scalartype, suffix, func) \
inline scalartype v_reduce_##suffix(const _Tpvec& a)                               \
{                                                                                  \
    const _Tpvec2 rs = func(a.val, vec_sld(a.val, a.val, 8));                      \
    return vec_extract(func(rs, vec_sld(rs, rs, 4)), 0);                           \
}
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_uint32x4, vec_uint4, uint, sum, vec_add)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_uint32x4, vec_uint4, uint, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_uint32x4, vec_uint4, uint, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_int32x4, vec_int4, int, sum, vec_add)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_int32x4, vec_int4, int, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_int32x4, vec_int4, int, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_float32x4, vec_float4, float, sum, vec_add)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_float32x4, vec_float4, float, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_float32x4, vec_float4, float, min, vec_min)

inline uint64 v_reduce_sum(const v_uint64x2& a)
{
    return vec_extract(vec_add(a.val, vec_permi(a.val, a.val, 3)), 0);
}
inline int64 v_reduce_sum(const v_int64x2& a)
{
    return vec_extract(vec_add(a.val, vec_permi(a.val, a.val, 3)), 0);
}
inline double v_reduce_sum(const v_float64x2& a)
{
    return vec_extract(vec_add(a.val, vec_permi(a.val, a.val, 3)), 0);
}

#define OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(_Tpvec, _Tpvec2, scalartype, suffix, func) \
inline scalartype v_reduce_##suffix(const _Tpvec& a)                               \
{                                                                                  \
    _Tpvec2 rs = func(a.val, vec_sld(a.val, a.val, 8));                            \
    rs = func(rs, vec_sld(rs, rs, 4));                                             \
    return vec_extract(func(rs, vec_sld(rs, rs, 2)), 0);                           \
}
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_uint16x8, vec_ushort8, ushort, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_uint16x8, vec_ushort8, ushort, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_int16x8, vec_short8, short, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_int16x8, vec_short8, short, min, vec_min)

#define OPENCV_HAL_IMPL_VSX_REDUCE_OP_16(_Tpvec, _Tpvec2, scalartype, suffix, func) \
inline scalartype v_reduce_##suffix(const _Tpvec& a)                               \
{                                                                                  \
    _Tpvec2 rs = func(a.val, vec_sld(a.val, a.val, 8));                            \
    rs = func(rs, vec_sld(rs, rs, 4));                                             \
    rs = func(rs, vec_sld(rs, rs, 2));                                             \
    return vec_extract(func(rs, vec_sld(rs, rs, 1)), 0);                           \
}
OPENCV_HAL_IMPL_VSX_REDUCE_OP_16(v_uint8x16, vec_uchar16, uchar, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_16(v_uint8x16, vec_uchar16, uchar, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_16(v_int8x16, vec_char16, schar, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_16(v_int8x16, vec_char16, schar, min, vec_min)

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    vec_float4 ac = vec_add(vec_mergel(a.val, c.val), vec_mergeh(a.val, c.val));
    ac = vec_add(ac, vec_sld(ac, ac, 8));

    vec_float4 bd = vec_add(vec_mergel(b.val, d.val), vec_mergeh(b.val, d.val));
    bd = vec_add(bd, vec_sld(bd, bd, 8));
    return v_float32x4(vec_mergeh(ac, bd));
}

inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
{
    const vec_uint4 zero4 = vec_uint4_z;
    vec_uint4 sum4 = vec_sum4s(vec_absd(a.val, b.val), zero4);
    return (unsigned)vec_extract(vec_sums(vec_int4_c(sum4), vec_int4_c(zero4)), 3);
}
inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
{
    const vec_int4 zero4 = vec_int4_z;
    vec_char16 ad = vec_abss(vec_subs(a.val, b.val));
    vec_int4 sum4 = vec_sum4s(ad, zero4);
    return (unsigned)vec_extract(vec_sums(sum4, zero4), 3);
}
inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
{
    vec_ushort8 ad = vec_absd(a.val, b.val);
    VSX_UNUSED(vec_int4) sum = vec_sums(vec_int4_c(vec_unpackhu(ad)) + vec_int4_c(vec_unpacklu(ad)), vec_int4_z);
    return (unsigned)vec_extract(sum, 3);
}
inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
{
    const vec_int4 zero4 = vec_int4_z;
    vec_short8 ad = vec_abss(vec_subs(a.val, b.val));
    vec_int4 sum4 = vec_sum4s(ad, zero4);
    return (unsigned)vec_extract(vec_sums(sum4, zero4), 3);
}
inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
{
    const vec_uint4 ad = vec_absd(a.val, b.val);
    const vec_uint4 rd = vec_add(ad, vec_sld(ad, ad, 8));
    return vec_extract(vec_add(rd, vec_sld(rd, rd, 4)), 0);
}
inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
{
    vec_int4 ad = vec_abss(vec_sub(a.val, b.val));
    return (unsigned)vec_extract(vec_sums(ad, vec_int4_z), 3);
}
inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    const vec_float4 ad = vec_abs(vec_sub(a.val, b.val));
    const vec_float4 rd = vec_add(ad, vec_sld(ad, ad, 8));
    return vec_extract(vec_add(rd, vec_sld(rd, rd, 4)), 0);
}

/** Popcount **/
inline v_uint8x16 v_popcount(const v_uint8x16& a)
{ return v_uint8x16(vec_popcntu(a.val)); }
inline v_uint8x16 v_popcount(const v_int8x16& a)
{ return v_uint8x16(vec_popcntu(a.val)); }
inline v_uint16x8 v_popcount(const v_uint16x8& a)
{ return v_uint16x8(vec_popcntu(a.val)); }
inline v_uint16x8 v_popcount(const v_int16x8& a)
{ return v_uint16x8(vec_popcntu(a.val)); }
inline v_uint32x4 v_popcount(const v_uint32x4& a)
{ return v_uint32x4(vec_popcntu(a.val)); }
inline v_uint32x4 v_popcount(const v_int32x4& a)
{ return v_uint32x4(vec_popcntu(a.val)); }
inline v_uint64x2 v_popcount(const v_uint64x2& a)
{ return v_uint64x2(vec_popcntu(a.val)); }
inline v_uint64x2 v_popcount(const v_int64x2& a)
{ return v_uint64x2(vec_popcntu(a.val)); }

/** Mask **/
inline int v_signmask(const v_uint8x16& a)
{
    static const vec_uchar16 qperm = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    return vec_extract((vec_int4)vec_vbpermq(v_reinterpret_as_u8(a).val, qperm), 2);
}
inline int v_signmask(const v_int8x16& a)
{ return v_signmask(v_reinterpret_as_u8(a)); }

inline int v_signmask(const v_int16x8& a)
{
    static const vec_uchar16 qperm = {112, 96, 80, 64, 48, 32, 16, 0, 128, 128, 128, 128, 128, 128, 128, 128};
    return vec_extract((vec_int4)vec_vbpermq(v_reinterpret_as_u8(a).val, qperm), 2);
}
inline int v_signmask(const v_uint16x8& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }

inline int v_signmask(const v_int32x4& a)
{
    static const vec_uchar16 qperm = {96, 64, 32, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    return vec_extract((vec_int4)vec_vbpermq(v_reinterpret_as_u8(a).val, qperm), 2);
}
inline int v_signmask(const v_uint32x4& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_float32x4& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }

inline int v_signmask(const v_int64x2& a)
{
    VSX_UNUSED(const vec_dword2) sv = vec_sr(a.val, vec_udword2_sp(63));
    return (int)vec_extract(sv, 0) | (int)vec_extract(sv, 1) << 1;
}
inline int v_signmask(const v_uint64x2& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
inline int v_signmask(const v_float64x2& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }

inline int v_scan_forward(const v_int8x16& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint8x16& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_int16x8& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint16x8& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_int32x4& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint32x4& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_float32x4& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_int64x2& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint64x2& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_float64x2& a) { return trailingZeros32(v_signmask(a)); }

template<typename _Tpvec>
inline bool v_check_all(const _Tpvec& a)
{ return vec_all_lt(a.val, _Tpvec::zero().val); }
inline bool v_check_all(const v_uint8x16& a)
{ return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_all(const v_uint16x8& a)
{ return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_all(const v_uint32x4& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_uint64x2& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_all(const v_float32x4& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_float64x2& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }

template<typename _Tpvec>
inline bool v_check_any(const _Tpvec& a)
{ return vec_any_lt(a.val, _Tpvec::zero().val); }
inline bool v_check_any(const v_uint8x16& a)
{ return v_check_any(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint16x8& a)
{ return v_check_any(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint32x4& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_uint64x2& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_float32x4& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float64x2& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }

////////// Other math /////////

/** Some frequent operations **/
inline v_float32x4 v_sqrt(const v_float32x4& x)
{ return v_float32x4(vec_sqrt(x.val)); }
inline v_float64x2 v_sqrt(const v_float64x2& x)
{ return v_float64x2(vec_sqrt(x.val)); }

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{ return v_float32x4(vec_rsqrt(x.val)); }
inline v_float64x2 v_invsqrt(const v_float64x2& x)
{ return v_float64x2(vec_rsqrt(x.val)); }

#define OPENCV_HAL_IMPL_VSX_MULADD(_Tpvec)                                  \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b)                 \
{ return _Tpvec(vec_sqrt(vec_madd(a.val, a.val, vec_mul(b.val, b.val)))); } \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b)             \
{ return _Tpvec(vec_madd(a.val, a.val, vec_mul(b.val, b.val))); }           \
inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)      \
{ return _Tpvec(vec_madd(a.val, b.val, c.val)); }                           \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)   \
{ return _Tpvec(vec_madd(a.val, b.val, c.val)); }

OPENCV_HAL_IMPL_VSX_MULADD(v_float32x4)
OPENCV_HAL_IMPL_VSX_MULADD(v_float64x2)

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{ return v_add(v_mul(a,  b), c); }

// TODO: exp, log, sin, cos

/** Absolute values **/
inline v_uint8x16 v_abs(const v_int8x16& x)
{ return v_uint8x16(vec_uchar16_c(vec_abs(x.val))); }

inline v_uint16x8 v_abs(const v_int16x8& x)
{ return v_uint16x8(vec_ushort8_c(vec_abs(x.val))); }

inline v_uint32x4 v_abs(const v_int32x4& x)
{ return v_uint32x4(vec_uint4_c(vec_abs(x.val))); }

inline v_float32x4 v_abs(const v_float32x4& x)
{ return v_float32x4(vec_abs(x.val)); }

inline v_float64x2 v_abs(const v_float64x2& x)
{ return v_float64x2(vec_abs(x.val)); }

/** Absolute difference **/
// unsigned
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_absdiff, vec_absd)

inline v_uint8x16 v_absdiff(const v_int8x16& a, const v_int8x16& b)
{ return v_reinterpret_as_u8(v_sub_wrap(v_max(a, b), v_min(a, b))); }
inline v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b)
{ return v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b))); }
inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{ return v_reinterpret_as_u32(v_sub(v_max(a, b), v_min(a, b))); }

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{ return v_abs(v_sub(a, b)); }
inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{ return v_abs(v_sub(a, b)); }

/** Absolute difference for signed integers **/
inline v_int8x16 v_absdiffs(const v_int8x16& a, const v_int8x16& b)
{ return v_int8x16(vec_abss(vec_subs(a.val, b.val))); }
inline v_int16x8 v_absdiffs(const v_int16x8& a, const v_int16x8& b)
{ return v_int16x8(vec_abss(vec_subs(a.val, b.val))); }

////////// Conversions /////////

/** Rounding **/
inline v_int32x4 v_round(const v_float32x4& a)
{ return v_int32x4(vec_cts(vec_rint(a.val))); }

inline v_int32x4 v_round(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_rint(a.val)), vec_int4_z)); }

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_rint(a.val)), vec_ctso(vec_rint(b.val)))); }

inline v_int32x4 v_floor(const v_float32x4& a)
{ return v_int32x4(vec_cts(vec_floor(a.val))); }

inline v_int32x4 v_floor(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_floor(a.val)), vec_int4_z)); }

inline v_int32x4 v_ceil(const v_float32x4& a)
{ return v_int32x4(vec_cts(vec_ceil(a.val))); }

inline v_int32x4 v_ceil(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_ceil(a.val)), vec_int4_z)); }

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(vec_cts(a.val)); }

inline v_int32x4 v_trunc(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(a.val), vec_int4_z)); }

/** To float **/
inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{ return v_float32x4(vec_ctf(a.val)); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{ return v_float32x4(vec_mergesqo(vec_cvfo(a.val), vec_float4_z)); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{ return v_float32x4(vec_mergesqo(vec_cvfo(a.val), vec_cvfo(b.val))); }

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{ return v_float64x2(vec_ctdo(vec_mergeh(a.val, a.val))); }

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{ return v_float64x2(vec_ctdo(vec_mergel(a.val, a.val))); }

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{ return v_float64x2(vec_cvfo(vec_mergeh(a.val, a.val))); }

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{ return v_float64x2(vec_cvfo(vec_mergel(a.val, a.val))); }

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{ return v_float64x2(vec_ctd(a.val)); }

////////////// Lookup table access ////////////////////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
    return v_int8x16(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]], tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]],
                     tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]);
}
inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
    return v_reinterpret_as_s8(v_int16x8(*(const short*)(tab+idx[0]), *(const short*)(tab+idx[1]), *(const short*)(tab+idx[2]), *(const short*)(tab+idx[3]),
                                       *(const short*)(tab+idx[4]), *(const short*)(tab+idx[5]), *(const short*)(tab+idx[6]), *(const short*)(tab+idx[7])));
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    return v_reinterpret_as_s8(v_int32x4(*(const int*)(tab+idx[0]), *(const int*)(tab+idx[1]), *(const int*)(tab+idx[2]), *(const int*)(tab+idx[3])));
}
inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((const schar*)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((const schar*)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((const schar*)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    return v_int16x8(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]], tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]);
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    return v_reinterpret_as_s16(v_int32x4(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]), *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    return v_reinterpret_as_s16(v_int64x2(*(const int64*)(tab + idx[0]), *(const int64*)(tab + idx[1])));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((const short*)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((const short*)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((const short*)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    return v_int32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    return v_reinterpret_as_s32(v_int64x2(*(const int64*)(tab + idx[0]), *(const int64*)(tab + idx[1])));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(vsx_ld(0, tab + idx[0]));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((const int*)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((const int*)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((const int*)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    return v_int64x2(tab[idx[0]], tab[idx[1]]);
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(vsx_ld2(0, tab + idx[0]));
}
inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    return v_float32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_pairs((const int*)tab, idx)); }
inline v_float32x4 v_lut_quads(const float* tab, const int* idx) { return v_load(tab + *idx); }

inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    return v_float64x2(tab[idx[0]], tab[idx[1]]);
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx) { return v_load(tab + *idx); }

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    const int idx[4] = {
        vec_extract(idxvec.val, 0),
        vec_extract(idxvec.val, 1),
        vec_extract(idxvec.val, 2),
        vec_extract(idxvec.val, 3)
    };
    return v_int32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    const int idx[4] = {
        vec_extract(idxvec.val, 0),
        vec_extract(idxvec.val, 1),
        vec_extract(idxvec.val, 2),
        vec_extract(idxvec.val, 3)
    };
    return v_uint32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    const int idx[4] = {
        vec_extract(idxvec.val, 0),
        vec_extract(idxvec.val, 1),
        vec_extract(idxvec.val, 2),
        vec_extract(idxvec.val, 3)
    };
    return v_float32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    const int idx[2] = {
        vec_extract(idxvec.val, 0),
        vec_extract(idxvec.val, 1)
    };
    return v_float64x2(tab[idx[0]], tab[idx[1]]);
}

inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    vec_float4 xy0 = vec_ld_l8(tab + vec_extract(idxvec.val, 0));
    vec_float4 xy1 = vec_ld_l8(tab + vec_extract(idxvec.val, 1));
    vec_float4 xy2 = vec_ld_l8(tab + vec_extract(idxvec.val, 2));
    vec_float4 xy3 = vec_ld_l8(tab + vec_extract(idxvec.val, 3));
    vec_float4 xy02 = vec_mergeh(xy0, xy2); // x0, x2, y0, y2
    vec_float4 xy13 = vec_mergeh(xy1, xy3); // x1, x3, y1, y3
    x.val = vec_mergeh(xy02, xy13);
    y.val = vec_mergel(xy02, xy13);
}
inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    vec_double2 xy0 = vsx_ld(vec_extract(idxvec.val, 0), tab);
    vec_double2 xy1 = vsx_ld(vec_extract(idxvec.val, 1), tab);
    x.val = vec_mergeh(xy0, xy1);
    y.val = vec_mergel(xy0, xy1);
}

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
    static const vec_uchar16 perm = {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15};
    return v_int8x16(vec_perm(vec.val, vec.val, perm));
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }

inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
    static const vec_uchar16 perm = {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15};
    return v_int8x16(vec_perm(vec.val, vec.val, perm));
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
    static const vec_uchar16 perm = {0,1, 4,5, 2,3, 6,7, 8,9, 12,13, 10,11, 14,15};
    return v_int16x8(vec_perm(vec.val, vec.val, perm));
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec)
{ return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }

inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
    static const vec_uchar16 perm = {0,1, 8,9, 2,3, 10,11, 4,5, 12,13, 6,7, 14,15};
    return v_int16x8(vec_perm(vec.val, vec.val, perm));
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec)
{ return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    static const vec_uchar16 perm = {0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15};
    return v_int32x4(vec_perm(vec.val, vec.val, perm));
}
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec)
{ return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec)
{ return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
    static const vec_uchar16 perm = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 15, 15, 15};
    return v_int8x16(vec_perm(vec.val, vec.val, perm));
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
    static const vec_uchar16 perm = {0,1, 2,3, 4,5, 8,9, 10,11, 12,13, 14,15, 14,15};
    return v_int16x8(vec_perm(vec.val, vec.val, perm));
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec)
{ return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec)
{ return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec)
{ return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec)
{ return vec; }

/////// FP16 support ////////

inline v_float32x4 v_load_expand(const hfloat* ptr)
{
    vec_ushort8 vf16 = vec_ld_l8((const ushort*)ptr);
#if CV_VSX3 && defined(vec_extract_fp_from_shorth)
    return v_float32x4(vec_extract_fp_from_shorth(vf16));
#elif CV_VSX3 && !defined(CV_COMPILER_VSX_BROKEN_ASM)
    vec_float4 vf32;
    __asm__ __volatile__ ("xvcvhpsp %x0,%x1" : "=wa" (vf32) : "wa" (vec_mergeh(vf16, vf16)));
    return v_float32x4(vf32);
#else
    const vec_int4 z = vec_int4_z, delta = vec_int4_sp(0x38000000);
    const vec_int4 signmask = vec_int4_sp(0x80000000);
    const vec_int4 maxexp = vec_int4_sp(0x7c000000);
    const vec_float4 deltaf = vec_float4_c(vec_int4_sp(0x38800000));

    vec_int4 bits = vec_int4_c(vec_mergeh(vec_short8_c(z), vec_short8_c(vf16)));
    vec_int4 e = vec_and(bits, maxexp), sign = vec_and(bits, signmask);
    vec_int4 t = vec_add(vec_sr(vec_xor(bits, sign), vec_uint4_sp(3)), delta); // ((h & 0x7fff) << 13) + delta
    vec_int4 zt = vec_int4_c(vec_sub(vec_float4_c(vec_add(t, vec_int4_sp(1 << 23))), deltaf));

    t = vec_add(t, vec_and(delta, vec_cmpeq(maxexp, e)));
    vec_bint4 zmask = vec_cmpeq(e, z);
    vec_int4 ft = vec_sel(t, zt, zmask);
    return v_float32x4(vec_float4_c(vec_or(ft, sign)));
#endif
}

inline void v_pack_store(hfloat* ptr, const v_float32x4& v)
{
// fixme: Is there any builtin op or intrinsic that cover "xvcvsphp"?
#if CV_VSX3 && !defined(CV_COMPILER_VSX_BROKEN_ASM)
    vec_ushort8 vf16;
    __asm__ __volatile__ ("xvcvsphp %x0,%x1" : "=wa" (vf16) : "wa" (v.val));
    vec_st_l8(vec_mergesqe(vf16, vf16), ptr);
#else
    const vec_int4 signmask = vec_int4_sp(0x80000000);
    const vec_int4 rval = vec_int4_sp(0x3f000000);

    vec_int4 t = vec_int4_c(v.val);
    vec_int4 sign = vec_sra(vec_and(t, signmask), vec_uint4_sp(16));
    t = vec_and(vec_nor(signmask, signmask), t);

    vec_bint4 finitemask = vec_cmpgt(vec_int4_sp(0x47800000), t);
    vec_bint4 isnan = vec_cmpgt(t, vec_int4_sp(0x7f800000));
    vec_int4 naninf = vec_sel(vec_int4_sp(0x7c00), vec_int4_sp(0x7e00), isnan);
    vec_bint4 tinymask = vec_cmpgt(vec_int4_sp(0x38800000), t);
    vec_int4 tt = vec_int4_c(vec_add(vec_float4_c(t), vec_float4_c(rval)));
    tt = vec_sub(tt, rval);
    vec_int4 odd = vec_and(vec_sr(t, vec_uint4_sp(13)), vec_int4_sp(1));
    vec_int4 nt = vec_add(t, vec_int4_sp(0xc8000fff));
    nt = vec_sr(vec_add(nt, odd), vec_uint4_sp(13));
    t = vec_sel(nt, tt, tinymask);
    t = vec_sel(naninf, t, finitemask);
    t = vec_or(t, sign);
    vec_st_l8(vec_packs(t, t), ptr);
#endif
}

inline void v_cleanup() {}


/** Reinterpret **/
/** its up there with load and store operations **/

////////// Matrix operations /////////

//////// Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{ return v_int32x4(vec_msum(a.val, b.val, vec_int4_z)); }
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_int32x4(vec_msum(a.val, b.val, c.val)); }

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    vec_dword2 even = vec_mule(a.val, b.val);
    vec_dword2 odd = vec_mulo(a.val, b.val);
    return v_int64x2(vec_add(even, odd));
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_add(v_dotprod(a, b), c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_uint32x4(vec_msum(a.val, b.val, c.val)); }
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{ return v_uint32x4(vec_msum(a.val, b.val, vec_uint4_z)); }

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    const vec_ushort8 eight = vec_ushort8_sp(8);
    vec_short8 a0 = vec_sra((vec_short8)vec_sld(a.val, a.val, 1), eight); // even
    vec_short8 a1 = vec_sra((vec_short8)a.val, eight); // odd
    vec_short8 b0 = vec_sra((vec_short8)vec_sld(b.val, b.val, 1), eight);
    vec_short8 b1 = vec_sra((vec_short8)b.val, eight);
    return v_int32x4(vec_msum(a0, b0, vec_msum(a1, b1, vec_int4_z)));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{
    const vec_ushort8 eight = vec_ushort8_sp(8);
    vec_short8 a0 = vec_sra((vec_short8)vec_sld(a.val, a.val, 1), eight); // even
    vec_short8 a1 = vec_sra((vec_short8)a.val, eight); // odd
    vec_short8 b0 = vec_sra((vec_short8)vec_sld(b.val, b.val, 1), eight);
    vec_short8 b1 = vec_sra((vec_short8)b.val, eight);
    return v_int32x4(vec_msum(a0, b0, vec_msum(a1, b1, c.val)));
}

// 16 >> 64
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    const vec_uint4 zero = vec_uint4_z;
    vec_uint4 even = vec_mule(a.val, b.val);
    vec_uint4 odd  = vec_mulo(a.val, b.val);
    vec_udword2 e0 = (vec_udword2)vec_mergee(even, zero);
    vec_udword2 e1 = (vec_udword2)vec_mergeo(even, zero);
    vec_udword2 o0 = (vec_udword2)vec_mergee(odd, zero);
    vec_udword2 o1 = (vec_udword2)vec_mergeo(odd, zero);
    vec_udword2 s0 = vec_add(e0, o0);
    vec_udword2 s1 = vec_add(e1, o1);
    return v_uint64x2(vec_add(s0, s1));
}
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    v_int32x4 prod = v_dotprod(a, b);
    v_int64x2 c, d;
    v_expand(prod, c, d);
    return v_int64x2(vec_add(vec_mergeh(c.val, d.val), vec_mergel(c.val, d.val)));
}
inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod(a, b); }
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_add(v_int32x4(vec_msum(a.val, b.val, vec_int4_z)), c); }
// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod(a, b); }
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_dotprod(a, b, c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_add(v_uint32x4(vec_msum(a.val, b.val, vec_uint4_z)), c); }

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
    vec_short8 a0 = vec_unpackh(a.val);
    vec_short8 a1 = vec_unpackl(a.val);
    vec_short8 b0 = vec_unpackh(b.val);
    vec_short8 b1 = vec_unpackl(b.val);
    return v_int32x4(vec_msum(a0, b0, vec_msum(a1, b1, vec_int4_z)));
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{ return v_dotprod_expand(a, b); }
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_dotprod_expand(a, b, c); }

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    v_int32x4 prod = v_dotprod(a, b);
    v_int64x2 c, d;
    v_expand(prod, c, d);
    return v_add(c, d);
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand(a, b, c); }

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    const vec_float4 v0 = vec_splat(v.val, 0);
    const vec_float4 v1 = vec_splat(v.val, 1);
    const vec_float4 v2 = vec_splat(v.val, 2);
    VSX_UNUSED(const vec_float4) v3 = vec_splat(v.val, 3);
    return v_float32x4(vec_madd(v0, m0.val, vec_madd(v1, m1.val, vec_madd(v2, m2.val, vec_mul(v3, m3.val)))));
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    const vec_float4 v0 = vec_splat(v.val, 0);
    const vec_float4 v1 = vec_splat(v.val, 1);
    const vec_float4 v2 = vec_splat(v.val, 2);
    return v_float32x4(vec_madd(v0, m0.val, vec_madd(v1, m1.val, vec_madd(v2, m2.val, a.val))));
}

#define OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(_Tpvec, _Tpvec2)                        \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1,                   \
                           const _Tpvec& a2, const _Tpvec& a3,                   \
                           _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3)       \
{                                                                                \
    _Tpvec2 a02 = vec_mergeh(a0.val, a2.val);                                    \
    _Tpvec2 a13 = vec_mergeh(a1.val, a3.val);                                    \
    b0.val = vec_mergeh(a02, a13);                                               \
    b1.val = vec_mergel(a02, a13);                                               \
    a02 = vec_mergel(a0.val, a2.val);                                            \
    a13 = vec_mergel(a1.val, a3.val);                                            \
    b2.val  = vec_mergeh(a02, a13);                                              \
    b3.val  = vec_mergel(a02, a13);                                              \
}
OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(v_uint32x4, vec_uint4)
OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(v_int32x4, vec_int4)
OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(v_float32x4, vec_float4)

template<int i, typename Tvec>
inline Tvec v_broadcast_element(const Tvec& v)
{ return Tvec(vec_splat(v.val, i)); }

#include "intrin_math.hpp"
inline v_float32x4 v_exp(const v_float32x4& x) { return v_exp_default_32f<v_float32x4, v_int32x4>(x); }
inline v_float32x4 v_log(const v_float32x4& x) { return v_log_default_32f<v_float32x4, v_int32x4>(x); }
inline void v_sincos(const v_float32x4& x, v_float32x4& s, v_float32x4& c) { v_sincos_default_32f<v_float32x4, v_int32x4>(x, s, c); }
inline v_float32x4 v_sin(const v_float32x4& x) { return v_sin_default_32f<v_float32x4, v_int32x4>(x); }
inline v_float32x4 v_cos(const v_float32x4& x) { return v_cos_default_32f<v_float32x4, v_int32x4>(x); }
inline v_float32x4 v_erf(const v_float32x4& x) { return v_erf_default_32f<v_float32x4, v_int32x4>(x); }

inline v_float64x2 v_exp(const v_float64x2& x) { return v_exp_default_64f<v_float64x2, v_int64x2>(x); }
inline v_float64x2 v_log(const v_float64x2& x) { return v_log_default_64f<v_float64x2, v_int64x2>(x); }
inline void v_sincos(const v_float64x2& x, v_float64x2& s, v_float64x2& c) { v_sincos_default_64f<v_float64x2, v_int64x2>(x, s, c); }
inline v_float64x2 v_sin(const v_float64x2& x) { return v_sin_default_64f<v_float64x2, v_int64x2>(x); }
inline v_float64x2 v_cos(const v_float64x2& x) { return v_cos_default_64f<v_float64x2, v_int64x2>(x); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif // OPENCV_HAL_VSX_HPP
