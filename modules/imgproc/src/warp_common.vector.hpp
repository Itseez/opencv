// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Shuffle (all pixels within image)
#define CV_WARP_NEAREST_VECTOR_SHUFFLE_ALLWITHIN_C1(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0];\
    }
#define CV_WARP_NEAREST_VECTOR_SHUFFLE_ALLWITHIN_C3(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[3*i] = srcptr[0];\
        pixbuf[3*i + 1] = srcptr[1]; \
        pixbuf[3*i + 2] = srcptr[2]; \
    }
#define CV_WARP_NEAREST_VECTOR_SHUFFLE_ALLWITHIN_C4(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[4*i] = srcptr[0];\
        pixbuf[4*i + 1] = srcptr[1]; \
        pixbuf[4*i + 2] = srcptr[2]; \
        pixbuf[4*i + 3] = srcptr[3]; \
    }
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_C1(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0]; \
        pixbuf[i + uf] = srcptr[1]; \
        pixbuf[i + uf*2] = srcptr[srcstep]; \
        pixbuf[i + uf*3] = srcptr[srcstep + 1]; \
    }
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_C3(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0]; \
        pixbuf[i + uf*4] = srcptr[1]; \
        pixbuf[i + uf*8] = srcptr[2]; \
        pixbuf[i + uf] = srcptr[3]; \
        pixbuf[i + uf*5] = srcptr[4]; \
        pixbuf[i + uf*9] = srcptr[5]; \
        pixbuf[i + uf*2] = srcptr[srcstep]; \
        pixbuf[i + uf*6] = srcptr[srcstep + 1]; \
        pixbuf[i + uf*10] = srcptr[srcstep + 2]; \
        pixbuf[i + uf*3] = srcptr[srcstep + 3]; \
        pixbuf[i + uf*7] = srcptr[srcstep + 4]; \
        pixbuf[i + uf*11] = srcptr[srcstep + 5]; \
    }
#define CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_C4(dtype) \
    for (int i = 0; i < uf; i++) { \
        const dtype* srcptr = src + addr[i]; \
        pixbuf[i] = srcptr[0]; \
        pixbuf[i + uf*4] = srcptr[1]; \
        pixbuf[i + uf*8] = srcptr[2]; \
        pixbuf[i + uf*12] = srcptr[3]; \
        pixbuf[i + uf] = srcptr[4]; \
        pixbuf[i + uf*5] = srcptr[5]; \
        pixbuf[i + uf*9] = srcptr[6]; \
        pixbuf[i + uf*13] = srcptr[7]; \
        pixbuf[i + uf*2] = srcptr[srcstep]; \
        pixbuf[i + uf*6] = srcptr[srcstep + 1]; \
        pixbuf[i + uf*10] = srcptr[srcstep + 2]; \
        pixbuf[i + uf*14] = srcptr[srcstep + 3]; \
        pixbuf[i + uf*3] = srcptr[srcstep + 4]; \
        pixbuf[i + uf*7] = srcptr[srcstep + 5]; \
        pixbuf[i + uf*11] = srcptr[srcstep + 6]; \
        pixbuf[i + uf*15] = srcptr[srcstep + 7]; \
    }
#define CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_8U(INTER, CN) \
    CV_WARP_##INTER##_VECTOR_SHUFFLE_ALLWITHIN_##CN(uint8_t)
#define CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_16U(INTER, CN) \
    CV_WARP_##INTER##_VECTOR_SHUFFLE_ALLWITHIN_##CN(uint16_t)
#define CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_32F(INTER, CN) \
    CV_WARP_##INTER##_VECTOR_SHUFFLE_ALLWITHIN_##CN(float)
#define CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(INTER, CN, DEPTH) \
    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_##DEPTH(INTER, CN)

// Shuffle (ARM NEON)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    uint8x8x4_t t00 = { \
        vld1_u8(src + addr[0]), \
        vld1_u8(src + addr[1]), \
        vld1_u8(src + addr[2]), \
        vld1_u8(src + addr[3]) \
    }; \
    uint8x8x4_t t01 = { \
        vld1_u8(src + addr[4]), \
        vld1_u8(src + addr[5]), \
        vld1_u8(src + addr[6]), \
        vld1_u8(src + addr[7]) \
    }; \
    uint8x8x4_t t10 = { \
        vld1_u8(src + addr[0] + srcstep), \
        vld1_u8(src + addr[1] + srcstep), \
        vld1_u8(src + addr[2] + srcstep), \
        vld1_u8(src + addr[3] + srcstep) \
    }; \
    uint8x8x4_t t11 = { \
        vld1_u8(src + addr[4] + srcstep), \
        vld1_u8(src + addr[5] + srcstep), \
        vld1_u8(src + addr[6] + srcstep), \
        vld1_u8(src + addr[7] + srcstep) \
    }; \
    uint32x2_t p00_, p01_, p10_, p11_;
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(coords, cn) \
    p00_ = vreinterpret_u32_u8(vtbl4_u8(t00, coords)); \
    p01_ = vreinterpret_u32_u8(vtbl4_u8(t01, coords)); \
    p10_ = vreinterpret_u32_u8(vtbl4_u8(t10, coords)); \
    p11_ = vreinterpret_u32_u8(vtbl4_u8(t11, coords)); \
    p00##cn = vreinterpret_u8_u32(vtrn1_u32(p00_, p01_)); \
    p01##cn = vreinterpret_u8_u32(vtrn2_u32(p00_, p01_)); \
    p10##cn = vreinterpret_u8_u32(vtrn1_u32(p10_, p11_)); \
    p11##cn = vreinterpret_u8_u32(vtrn2_u32(p10_, p11_));
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_C1() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(grays, g)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_C3() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(reds, r) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(greens, g) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(blues, b)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_C4() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_LOAD() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(reds, r) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(greens, g) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(blues, b) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_TRN(alphas, a)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(CN) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8_##CN()


// Shuffle (not all pixels within image)
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_8UC1() \
    v_store_low(dstptr + x, bval_v0);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_8UC3() \
    v_store_low(dstptr + x*3,        bval_v0); \
    v_store_low(dstptr + x*3 + uf,   bval_v1); \
    v_store_low(dstptr + x*3 + uf*2, bval_v2);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_8UC4() \
    v_store_low(dstptr + x*4,        bval_v0); \
    v_store_low(dstptr + x*4 + uf,   bval_v1); \
    v_store_low(dstptr + x*4 + uf*2, bval_v2); \
    v_store_low(dstptr + x*4 + uf*3, bval_v3);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_16UC1() \
    v_store(dstptr + x, bval_v0);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_16UC3() \
    v_store(dstptr + x*3,        bval_v0); \
    v_store(dstptr + x*3 + uf,   bval_v1); \
    v_store(dstptr + x*3 + uf*2, bval_v2);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_16UC4() \
    v_store(dstptr + x*4,        bval_v0); \
    v_store(dstptr + x*4 + uf,   bval_v1); \
    v_store(dstptr + x*4 + uf*2, bval_v2); \
    v_store(dstptr + x*4 + uf*3, bval_v3);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_32FC1() \
    v_store(dstptr + x,             bval_v0_l); \
    v_store(dstptr + x + vlanes_32, bval_v0_h);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_32FC3() \
    v_store(dstptr + x*3,                    bval_v0_l); \
    v_store(dstptr + x*3 + vlanes_32,        bval_v0_h); \
    v_store(dstptr + x*3 + uf,               bval_v1_l); \
    v_store(dstptr + x*3 + uf + vlanes_32,   bval_v1_h); \
    v_store(dstptr + x*3 + uf*2,             bval_v2_l); \
    v_store(dstptr + x*3 + uf*2 + vlanes_32, bval_v2_h);
#define CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_32FC4() \
    v_store(dstptr + x*4,                    bval_v0_l); \
    v_store(dstptr + x*4 + vlanes_32,        bval_v0_h); \
    v_store(dstptr + x*4 + uf,               bval_v1_l); \
    v_store(dstptr + x*4 + uf + vlanes_32,   bval_v1_h); \
    v_store(dstptr + x*4 + uf*2,             bval_v2_l); \
    v_store(dstptr + x*4 + uf*2 + vlanes_32, bval_v2_h); \
    v_store(dstptr + x*4 + uf*3,             bval_v3_l); \
    v_store(dstptr + x*4 + uf*3 + vlanes_32, bval_v3_h);

#define CV_WARP_VECTOR_FETCH_PIXEL_C1(dy, dx, pixbuf_ofs0, pixbuf_ofs1) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx; \
        pixbuf[i + pixbuf_ofs0] = src[addr_i]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs0] = bval[0]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs0] = dstptr[x + i]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t addr_i = iy_*srcstep + ix_; \
        pixbuf[i + pixbuf_ofs0] = src[addr_i]; \
    }
#define CV_WARP_VECTOR_FETCH_PIXEL_C3(dy, dx, pixbuf_ofs0, pixbuf_ofs1) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx*3; \
        pixbuf[i + pixbuf_ofs0] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = src[addr_i+2]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs0] = bval[0]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = bval[1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = bval[2]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs0] = dstptr[(x + i)*3]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = dstptr[(x + i)*3 + 1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = dstptr[(x + i)*3 + 2]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t addr_i = iy_*srcstep + ix_*3; \
        pixbuf[i + pixbuf_ofs0] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = src[addr_i+2]; \
    }
#define CV_WARP_VECTOR_FETCH_PIXEL_C4(dy, dx, pixbuf_ofs0, pixbuf_ofs1) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t addr_i = addr[i] + dy*srcstep + dx*4; \
        pixbuf[i + pixbuf_ofs0] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = src[addr_i+2]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*3] = src[addr_i+3]; \
    } else if (border_type == BORDER_CONSTANT) { \
        pixbuf[i + pixbuf_ofs0] = bval[0]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = bval[1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = bval[2]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*3] = bval[3]; \
    } else if (border_type == BORDER_TRANSPARENT) { \
        pixbuf[i + pixbuf_ofs0] = dstptr[(x + i)*4]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = dstptr[(x + i)*4 + 1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = dstptr[(x + i)*4 + 2]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*3] = dstptr[(x + i)*4 + 3]; \
    } else { \
        int ix_ = borderInterpolate_fast(ix + dx, srccols, border_type_x); \
        int iy_ = borderInterpolate_fast(iy + dy, srcrows, border_type_y); \
        size_t addr_i = iy_*srcstep + ix_*4; \
        pixbuf[i + pixbuf_ofs0] = src[addr_i]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1] = src[addr_i+1]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*2] = src[addr_i+2]; \
        pixbuf[i + pixbuf_ofs0 + pixbuf_ofs1*3] = src[addr_i+3]; \
    }
#define CV_WARP_NEAREST_VECTOR_FETCH_PIXEL_C1() \
    CV_WARP_VECTOR_FETCH_PIXEL_C1(0, 0, 0, 1);
#define CV_WARP_NEAREST_VECTOR_FETCH_PIXEL_C3() \
    CV_WARP_VECTOR_FETCH_PIXEL_C3(0, 0, 2*i, 1);
#define CV_WARP_NEAREST_VECTOR_FETCH_PIXEL_C4() \
    CV_WARP_VECTOR_FETCH_PIXEL_C4(0, 0, 3*i, 1);
#define CV_WARP_NEAREST_VECTOR_FETCH_PIXEL(CN) \
    CV_WARP_NEAREST_VECTOR_FETCH_PIXEL_##CN()
#define CV_WARP_LINEAR_VECTOR_FETCH_PIXEL(CN) \
    CV_WARP_VECTOR_FETCH_PIXEL_##CN(0, 0, 0,    uf*4); \
    CV_WARP_VECTOR_FETCH_PIXEL_##CN(0, 1, uf,   uf*4); \
    CV_WARP_VECTOR_FETCH_PIXEL_##CN(1, 0, uf*2, uf*4); \
    CV_WARP_VECTOR_FETCH_PIXEL_##CN(1, 1, uf*3, uf*4);

#define CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(INTER, CN, DEPTH) \
    if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) { \
        mask_0 = v_lt(v_reinterpret_as_u32(v_add(src_ix0, one)), outer_scols); \
        mask_1 = v_lt(v_reinterpret_as_u32(v_add(src_ix1, one)), outer_scols); \
        mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(v_add(src_iy0, one)), outer_srows)); \
        mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(v_add(src_iy1, one)), outer_srows)); \
        v_uint16 outer_mask = v_pack(mask_0, mask_1); \
        if (v_reduce_max(outer_mask) == 0) { \
            if (border_type == BORDER_CONSTANT) { \
                CV_WARP_VECTOR_SHUFFLE_STORE_CONSTANT_BORDER_##DEPTH##CN() \
            } \
            continue; \
        } \
    } \
    vx_store(src_ix, src_ix0); \
    vx_store(src_iy, src_iy0); \
    vx_store(src_ix + vlanes_32, src_ix1); \
    vx_store(src_iy + vlanes_32, src_iy1); \
    for (int i = 0; i < uf; i++) { \
        int ix = src_ix[i], iy = src_iy[i]; \
        CV_WARP_##INTER##_VECTOR_FETCH_PIXEL(CN) \
    }

// Shuffle (not all pixels within image) (ARM NEON)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(cn, offset)\
    p00##cn = vld1_u8(pixbuf + offset);      \
    p01##cn = vld1_u8(pixbuf + offset + 8);  \
    p10##cn = vld1_u8(pixbuf + offset + 16); \
    p11##cn = vld1_u8(pixbuf + offset + 24);
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_C1() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(g, 0)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_C3() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(r, 0) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(g, 32) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(b, 64)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_C4() \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(r, 0) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(g, 32) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(b, 64) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_LOAD(a, 96)
#define CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(CN) \
    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8_##CN()

// [New] Load pixels for interpolation
#define CV_WARP_VECTOR_NEAREST_LOAD_CN_8U_16U(cn, i) \
    v_uint16 f00##cn = vx_load_expand(pixbuf + uf * i);
#define CV_WARP_VECTOR_NEAREST_LOAD_CN_16U_16U(cn, i) \
    v_uint16 f00##cn = vx_load(pixbuf + uf * i);
#define CV_WARP_VECTOR_NEAREST_LOAD_CN_32F_32F(cn, i) \
    v_float32 f00##cn##l = vx_load(pixbuf + uf * i); \
    v_float32 f00##cn##h = vx_load(pixbuf + uf * i + vlanes_32);
#define CV_WARP_VECTOR_LINEAR_LOAD_CN_8U_16U(cn, i) \
    v_uint16  f00##cn = vx_load_expand(pixbuf + uf *  4*i), \
              f01##cn = vx_load_expand(pixbuf + uf * (4*i+1)), \
              f10##cn = vx_load_expand(pixbuf + uf * (4*i+2)), \
              f11##cn = vx_load_expand(pixbuf + uf * (4*i+3));
#define CV_WARP_VECTOR_LINEAR_LOAD_CN_16U_16U(cn, i) \
    v_uint16 f00##cn = vx_load(pixbuf + uf *  4*i), \
             f01##cn = vx_load(pixbuf + uf * (4*i+1)), \
             f10##cn = vx_load(pixbuf + uf * (4*i+2)), \
             f11##cn = vx_load(pixbuf + uf * (4*i+3));
#define CV_WARP_VECTOR_LINEAR_LOAD_CN_32F_32F(cn, i) \
    v_float32 f00##cn##l = vx_load(pixbuf + uf *  4*i), \
              f00##cn##h = vx_load(pixbuf + uf *  4*i    + vlanes_32); \
    v_float32 f01##cn##l = vx_load(pixbuf + uf * (4*i+1)), \
              f01##cn##h = vx_load(pixbuf + uf * (4*i+1) + vlanes_32); \
    v_float32 f10##cn##l = vx_load(pixbuf + uf * (4*i+2)), \
              f10##cn##h = vx_load(pixbuf + uf * (4*i+2) + vlanes_32); \
    v_float32 f11##cn##l = vx_load(pixbuf + uf * (4*i+3)), \
              f11##cn##h = vx_load(pixbuf + uf * (4*i+3) + vlanes_32);
#define CV_WARP_VECTOR_INTER_LOAD_C1(INTER, SDEPTH, DDEPTH) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(g, 0)
#define CV_WARP_VECTOR_INTER_LOAD_C3(INTER, SDEPTH, DDEPTH) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(r, 0) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(g, 1) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(b, 2)
#define CV_WARP_VECTOR_INTER_LOAD_C4(INTER, SDEPTH, DDEPTH) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(r, 0) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(g, 1) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(b, 2) \
    CV_WARP_VECTOR_##INTER##_LOAD_CN_##SDEPTH##_##DDEPTH(a, 3)
#define CV_WARP_VECTOR_INTER_LOAD(INTER, CN, SDEPTH, DDEPTH) \
    CV_WARP_VECTOR_INTER_LOAD_##CN(INTER, SDEPTH, DDEPTH)

// [New] Store
#define CV_WARP_VECTOR_NEAREST_STORE_C1_16U_8U() \
    v_pack_store(dstptr + x, f00g);
#define CV_WARP_VECTOR_NEAREST_STORE_C3_16U_8U() \
    v_pack_store(dstptr + 3*x,        f00r); \
    v_pack_store(dstptr + 3*x + uf,   f00g); \
    v_pack_store(dstptr + 3*x + uf*2, f00b);
#define CV_WARP_VECTOR_NEAREST_STORE_C4_16U_8U() \
    v_pack_store(dstptr + 4*x,        f00r); \
    v_pack_store(dstptr + 4*x + uf,   f00g); \
    v_pack_store(dstptr + 4*x + uf*2, f00b); \
    v_pack_store(dstptr + 4*x + uf*3, f00a);
#define CV_WARP_VECTOR_NEAREST_STORE_C1_16U_16U() \
    vx_store(dstptr + x, f00g);
#define CV_WARP_VECTOR_NEAREST_STORE_C3_16U_16U() \
    vx_store(dstptr + 3*x,        f00r); \
    vx_store(dstptr + 3*x + uf,   f00g); \
    vx_store(dstptr + 3*x + uf*2, f00b);
#define CV_WARP_VECTOR_NEAREST_STORE_C4_16U_16U() \
    vx_store(dstptr + 4*x,        f00r); \
    vx_store(dstptr + 4*x + uf,   f00g); \
    vx_store(dstptr + 4*x + uf*2, f00b); \
    vx_store(dstptr + 4*x + uf*3, f00a);
#define CV_WARP_VECTOR_NEAREST_STORE_C1_32F_32F() \
    vx_store(dstptr + x,             f00gl); \
    vx_store(dstptr + x + vlanes_32, f00gh);
#define CV_WARP_VECTOR_NEAREST_STORE_C3_32F_32F() \
    vx_store(dstptr + 3*x,                    f00rl); \
    vx_store(dstptr + 3*x        + vlanes_32, f00rh); \
    vx_store(dstptr + 3*x + uf,               f00gl); \
    vx_store(dstptr + 3*x + uf   + vlanes_32, f00gh); \
    vx_store(dstptr + 3*x + uf*2,             f00bl); \
    vx_store(dstptr + 3*x + uf*2 + vlanes_32, f00bh);
#define CV_WARP_VECTOR_NEAREST_STORE_C4_32F_32F() \
    vx_store(dstptr + 4*x,                    f00rl); \
    vx_store(dstptr + 4*x        + vlanes_32, f00rh); \
    vx_store(dstptr + 4*x + uf,               f00gl); \
    vx_store(dstptr + 4*x + uf   + vlanes_32, f00gh); \
    vx_store(dstptr + 4*x + uf*2,             f00bl); \
    vx_store(dstptr + 4*x + uf*2 + vlanes_32, f00bh); \
    vx_store(dstptr + 4*x + uf*3,             f00al); \
    vx_store(dstptr + 4*x + uf*3 + vlanes_32, f00ah);
#define CV_WARP_VECTOR_INTER_STORE(INTER, CN, SDEPTH, DDEPTH) \
    CV_WARP_VECTOR_##INTER##_STORE_##CN##_##SDEPTH##_##DDEPTH()


// Load pixels for linear interpolation (uint8_t -> uint16_t) (ARM NEON)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(cn) \
    v_uint16 f00##cn = v_uint16(vmovl_u8(p00##cn)), \
             f01##cn = v_uint16(vmovl_u8(p01##cn)), \
             f10##cn = v_uint16(vmovl_u8(p10##cn)), \
             f11##cn = v_uint16(vmovl_u8(p11##cn));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(g)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(b)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(b) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8U16_NEON(a)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON_##CN();

// Load pixels for linear interpolation (uint16_t -> float)
#define CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(cn) \
    v_float32 f00##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f00##cn))), f00##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f00##cn))), \
              f01##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f01##cn))), f01##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f01##cn))), \
              f10##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f10##cn))), f10##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f10##cn))), \
              f11##cn##l = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(f11##cn))), f11##cn##h = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(f11##cn)));
#define CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_CN_U16F32(a)
#define CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32_##CN()

// Load pixels for linear interpolation (uint8_t -> float16)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(cn) \
    v_float16 f00##cn = v_float16(vcvtq_f16_u16(vmovl_u8(p00##cn))), \
              f01##cn = v_float16(vcvtq_f16_u16(vmovl_u8(p01##cn))), \
              f10##cn = v_float16(vcvtq_f16_u16(vmovl_u8(p10##cn))), \
              f11##cn = v_float16(vcvtq_f16_u16(vmovl_u8(p11##cn)));
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(g)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(b)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(r) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(g) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(b) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_CN_U8F16(a)
#define CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16_##CN()

// Linear interpolation calculation (F32)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(cn) \
    f00##cn##l = v_fma(alphal, v_sub(f01##cn##l, f00##cn##l), f00##cn##l); f00##cn##h = v_fma(alphah, v_sub(f01##cn##h, f00##cn##h), f00##cn##h); \
    f10##cn##l = v_fma(alphal, v_sub(f11##cn##l, f10##cn##l), f10##cn##l); f10##cn##h = v_fma(alphah, v_sub(f11##cn##h, f10##cn##h), f10##cn##h);
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32(a)

#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(cn) \
    f00##cn##l = v_fma(betal,  v_sub(f10##cn##l, f00##cn##l), f00##cn##l); f00##cn##h = v_fma(betah,  v_sub(f10##cn##h, f00##cn##h), f00##cn##h);
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32(a)

#define CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(CN) \
    v_float32 alphal = src_x0, alphah = src_x1, \
              betal = src_y0, betah = src_y1; \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F32_##CN() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F32_##CN()

// Linear interpolation calculation (F16)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(cn) \
    f00##cn = v_fma(alpha, v_sub(f01##cn, f00##cn), f00##cn); \
    f10##cn = v_fma(alpha, v_sub(f11##cn, f10##cn), f10##cn);
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16(a)

#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(cn) \
    f00##cn = v_fma(beta,  v_sub(f10##cn, f00##cn), f00##cn);
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16_C1() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(g)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16_C3() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(b)
#define CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16_C4() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(r) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(g) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(b) \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16(a)

#define CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(CN) \
    v_float16 alpha = v_cvt_f16(src_x0, src_x1), \
              beta = v_cvt_f16(src_y0, src_y1); \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_ALPHA_F16_##CN() \
    CV_WARP_LINEAR_VECTOR_INTER_CALC_BETA_F16_##CN()


// Store
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_C1() \
    v_uint16 f00_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)); \
    v_uint8 f00_u8 = v_pack(f00_u16, vx_setall_u16(0)); \
    v_store_low(dstptr + x, f00_u8);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_C3() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)); \
    uint16_t tbuf[max_vlanes_16*3]; \
    v_store_interleave(tbuf, f00r_u16, f00g_u16, f00b_u16); \
    v_pack_store(dstptr + x*3, vx_load(tbuf)); \
    v_pack_store(dstptr + x*3 + vlanes_16, vx_load(tbuf + vlanes_16)); \
    v_pack_store(dstptr + x*3 + vlanes_16*2, vx_load(tbuf + vlanes_16*2));
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_C4() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)), \
             f00a_u16 = v_pack_u(v_round(f00al), v_round(f00ah)); \
    uint16_t tbuf[max_vlanes_16*4]; \
    v_store_interleave(tbuf, f00r_u16, f00g_u16, f00b_u16, f00a_u16); \
    v_pack_store(dstptr + x*4, vx_load(tbuf)); \
    v_pack_store(dstptr + x*4 + vlanes_16, vx_load(tbuf + vlanes_16)); \
    v_pack_store(dstptr + x*4 + vlanes_16*2, vx_load(tbuf + vlanes_16*2)); \
    v_pack_store(dstptr + x*4 + vlanes_16*3, vx_load(tbuf + vlanes_16*3));
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8_##CN()

#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_C1() \
    v_uint16 f00_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)); \
    v_store(dstptr + x, f00_u16);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_C3() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)); \
    v_store_interleave(dstptr + x*3, f00r_u16, f00g_u16, f00b_u16);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_C4() \
    v_uint16 f00r_u16 = v_pack_u(v_round(f00rl), v_round(f00rh)), \
             f00g_u16 = v_pack_u(v_round(f00gl), v_round(f00gh)), \
             f00b_u16 = v_pack_u(v_round(f00bl), v_round(f00bh)), \
             f00a_u16 = v_pack_u(v_round(f00al), v_round(f00ah)); \
    v_store_interleave(dstptr + x*4, f00r_u16, f00g_u16, f00b_u16, f00a_u16);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16_##CN()

#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_C1() \
    vx_store(dstptr + x, f00gl); \
    vx_store(dstptr + x + vlanes_32, f00gh);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_C3() \
    v_store_interleave(dstptr + x*3, f00rl, f00gl, f00bl); \
    v_store_interleave(dstptr + x*3 + vlanes_32*3, f00rh, f00gh, f00bh);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_C4() \
    v_store_interleave(dstptr + x*4, f00rl, f00gl, f00bl, f00al); \
    v_store_interleave(dstptr + x*4 + vlanes_32*4, f00rh, f00gh, f00bh, f00ah);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32_##CN()

#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8_C1() \
    uint8x8_t result = { \
        vqmovun_s16(vcvtnq_s16_f16(f00g.val)), \
    }; \
    vst1_u8(dstptr + x, result);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8_C3() \
    uint8x8x3_t result = { \
        vqmovun_s16(vcvtnq_s16_f16(f00r.val)), \
        vqmovun_s16(vcvtnq_s16_f16(f00g.val)), \
        vqmovun_s16(vcvtnq_s16_f16(f00b.val)), \
    }; \
    vst3_u8(dstptr + x*3, result);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8_C4() \
    uint8x8x4_t result = { \
        vqmovun_s16(vcvtnq_s16_f16(f00r.val)), \
        vqmovun_s16(vcvtnq_s16_f16(f00g.val)), \
        vqmovun_s16(vcvtnq_s16_f16(f00b.val)), \
        vqmovun_s16(vcvtnq_s16_f16(f00a.val)), \
    }; \
    vst4_u8(dstptr + x*4, result);
#define CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(CN) \
    CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8_##CN()

// Special case for C4 shuffle, interpolation and store
// SIMD128, c4, nearest
#define CV_WARP_SIMD128_NEAREST_SHUFFLE_INTER_8UC4_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[i+ofs]; \
    v_uint32 i##ofs##_pix0 = vx_load_expand_q(srcptr##ofs);
#define CV_WARP_SIMD128_NEAREST_SHUFFLE_INTER_16UC4_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[i+ofs]; \
    v_uint32 i##ofs##_pix0 = vx_load_expand(srcptr##ofs);
#define CV_WARP_SIMD128_NEAREST_SHUFFLE_INTER_32FC4_I(ofs) \
    const float *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_pix0 = vx_load(srcptr##ofs);
#define CV_WARP_SIMD128_NEAREST_STORE_8UC4_I() \
    v_pack_store(dstptr + 4*(x+i), v_pack(i0_pix0, i1_pix0)); \
    v_pack_store(dstptr + 4*(x+i+2), v_pack(i2_pix0, i3_pix0));
#define CV_WARP_SIMD128_NEAREST_STORE_16UC4_I() \
    vx_store(dstptr + 4*(x+i), v_pack(i0_pix0, i1_pix0)); \
    vx_store(dstptr + 4*(x+i+2), v_pack(i2_pix0, i3_pix0));
#define CV_WARP_SIMD128_NEAREST_STORE_32FC4_I() \
    vx_store(dstptr + 4*(x+i),   i0_pix0); \
    vx_store(dstptr + 4*(x+i+1), i1_pix0); \
    vx_store(dstptr + 4*(x+i+2), i2_pix0); \
    vx_store(dstptr + 4*(x+i+3), i3_pix0);
// SIMD128, c4, bilinear
#define CV_WARP_SIMD128_LINEAR_SHUFFLE_INTER_8UC4_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_pix0 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs))); \
    v_float32 i##ofs##_pix1 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs+4))); \
    v_float32 i##ofs##_pix2 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs+srcstep))); \
    v_float32 i##ofs##_pix3 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs+srcstep+4))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]);  \
    i##ofs##_pix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix1, i##ofs##_pix0), i##ofs##_pix0); \
    i##ofs##_pix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix3, i##ofs##_pix2), i##ofs##_pix2); \
    i##ofs##_pix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_pix2, i##ofs##_pix0), i##ofs##_pix0);
#define CV_WARP_SIMD128_LINEAR_SHUFFLE_INTER_16UC4_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_pix0 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs))); \
    v_float32 i##ofs##_pix1 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs+4))); \
    v_float32 i##ofs##_pix2 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs+srcstep))); \
    v_float32 i##ofs##_pix3 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs+srcstep+4))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]);  \
    i##ofs##_pix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix1, i##ofs##_pix0), i##ofs##_pix0); \
    i##ofs##_pix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix3, i##ofs##_pix2), i##ofs##_pix2); \
    i##ofs##_pix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_pix2, i##ofs##_pix0), i##ofs##_pix0);
#define CV_WARP_SIMD128_LINEAR_SHUFFLE_INTER_32FC4_I(ofs) \
    const float *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_pix0 = vx_load(srcptr##ofs); \
    v_float32 i##ofs##_pix1 = vx_load(srcptr##ofs+4); \
    v_float32 i##ofs##_pix2 = vx_load(srcptr##ofs+srcstep); \
    v_float32 i##ofs##_pix3 = vx_load(srcptr##ofs+srcstep+4); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]);  \
    i##ofs##_pix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix1, i##ofs##_pix0), i##ofs##_pix0); \
    i##ofs##_pix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix3, i##ofs##_pix2), i##ofs##_pix2); \
    i##ofs##_pix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_pix2, i##ofs##_pix0), i##ofs##_pix0);
#define CV_WARP_SIMD128_LINEAR_STORE_8UC4_I() \
    v_uint16 i01_pix = v_pack_u(v_round(i0_pix0), v_round(i1_pix0)); \
    v_uint16 i23_pix = v_pack_u(v_round(i2_pix0), v_round(i3_pix0)); \
    v_pack_store(dstptr + 4*(x+i),   i01_pix); \
    v_pack_store(dstptr + 4*(x+i+2), i23_pix);
#define CV_WARP_SIMD128_LINEAR_STORE_16UC4_I() \
    v_uint16 i01_pix = v_pack_u(v_round(i0_pix0), v_round(i1_pix0)); \
    v_uint16 i23_pix = v_pack_u(v_round(i2_pix0), v_round(i3_pix0)); \
    vx_store(dstptr + 4*(x+i),   i01_pix); \
    vx_store(dstptr + 4*(x+i+2), i23_pix);
#define CV_WARP_SIMD128_LINEAR_STORE_32FC4_I() \
    vx_store(dstptr + 4*(x+i),   i0_pix0); \
    vx_store(dstptr + 4*(x+i+1), i1_pix0); \
    vx_store(dstptr + 4*(x+i+2), i2_pix0); \
    vx_store(dstptr + 4*(x+i+3), i3_pix0);
#define CV_WARP_SIMD128_SHUFFLE_INTER_STORE_C4(INTER, DEPTH) \
    for (int i = 0; i < uf; i+=vlanes_32) { \
        CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(0) \
        CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(1) \
        CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(2) \
        CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(3) \
        CV_WARP_SIMD128_##INTER##_STORE_##DEPTH##C4_I() \
    }
// SIMD256, c4, nearest
#define CV_WARP_SIMD256_NEAREST_SHUFFLE_INTER_8UC4_I(ofs0, ofs1) \
    const uint8_t *srcptr##ofs0 = src + addr[i+ofs0]; \
    const uint8_t *srcptr##ofs1 = src + addr[i+ofs1]; \
    v_uint32 i##ofs0##_pix0x = v256_load_expand_q(srcptr##ofs0); \
    v_uint32 i##ofs1##_pix0x = v256_load_expand_q(srcptr##ofs1); \
    v_uint32 i##ofs0##ofs1##_pix00 = v_combine_low(i##ofs0##_pix0x, i##ofs1##_pix0x);
#define CV_WARP_SIMD256_NEAREST_SHUFFLE_INTER_16UC4_I(ofs0, ofs1) \
    const uint16_t *srcptr##ofs0 = src + addr[i+ofs0]; \
    const uint16_t *srcptr##ofs1 = src + addr[i+ofs1]; \
    v_uint32 i##ofs0##_pix0x = v256_load_expand(srcptr##ofs0); \
    v_uint32 i##ofs1##_pix0x = v256_load_expand(srcptr##ofs1); \
    v_uint32 i##ofs0##ofs1##_pix00 = v_combine_low(i##ofs0##_pix0x, i##ofs1##_pix0x);
#define CV_WARP_SIMD256_NEAREST_SHUFFLE_INTER_32FC4_I(ofs0, ofs1) \
    const float *srcptr##ofs0 = src + addr[i+ofs0]; \
    const float *srcptr##ofs1 = src + addr[i+ofs1]; \
    v_float32 i##ofs0##ofs1##_fpix00 = vx_load_halves(srcptr##ofs0, srcptr##ofs1);
#define CV_WARP_SIMD256_NEAREST_STORE_8UC4_I() \
    v_pack_store(dstptr + 4*(x+i),   v_pack(i01_pix00, i23_pix00)); \
    v_pack_store(dstptr + 4*(x+i+4), v_pack(i45_pix00, i67_pix00));
#define CV_WARP_SIMD256_NEAREST_STORE_16UC4_I() \
    vx_store(dstptr + 4*(x+i),   v_pack(i01_pix00, i23_pix00)); \
    vx_store(dstptr + 4*(x+i+4), v_pack(i45_pix00, i67_pix00));
#define CV_WARP_SIMD256_NEAREST_STORE_32FC4_I() \
    vx_store(dstptr + 4*(x+i),    i01_fpix00); \
    vx_store(dstptr + 4*(x+i+2),  i23_fpix00); \
    vx_store(dstptr + 4*(x+i+4),  i45_fpix00); \
    vx_store(dstptr + 4*(x+i+6),  i67_fpix00);
// SIMD256, c4, bilinear
#define CV_WARP_SIMD256_LINEAR_SHUFFLE_INTER_8UC4_I(ofs0, ofs1) \
    const uint8_t *srcptr##ofs0 = src + addr[i+ofs0]; \
    const uint8_t *srcptr##ofs1 = src + addr[i+ofs1]; \
    v_int32 i##ofs0##_pix01 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs0)), \
            i##ofs0##_pix23 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs0+srcstep)); \
    v_int32 i##ofs1##_pix01 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs1)), \
            i##ofs1##_pix23 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs1+srcstep)); \
    v_float32 i##ofs0##_fpix01 = v_cvt_f32(i##ofs0##_pix01), i##ofs0##_fpix23 = v_cvt_f32(i##ofs0##_pix23); \
    v_float32 i##ofs1##_fpix01 = v_cvt_f32(i##ofs1##_pix01), i##ofs1##_fpix23 = v_cvt_f32(i##ofs1##_pix23); \
    v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11, \
              i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
    v_recombine(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
    v_recombine(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
    v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[i+ofs0]), \
              i##ofs1##_alpha = vx_setall_f32(valpha[i+ofs1]), \
              i##ofs0##_beta  = vx_setall_f32(vbeta[i+ofs0]), \
              i##ofs1##_beta  = vx_setall_f32(vbeta[i+ofs1]); \
    v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
              i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    auto i##ofs0##ofs1##_pix00 = v_round(i##ofs0##ofs1##_fpix00);
#define CV_WARP_SIMD256_LINEAR_SHUFFLE_INTER_16UC4_I(ofs0, ofs1) \
    const uint16_t *srcptr##ofs0 = src + addr[i+ofs0]; \
    const uint16_t *srcptr##ofs1 = src + addr[i+ofs1]; \
    v_int32 i##ofs0##_pix01 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs0)), \
            i##ofs0##_pix23 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs0+srcstep)); \
    v_int32 i##ofs1##_pix01 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs1)), \
            i##ofs1##_pix23 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs1+srcstep)); \
    v_float32 i##ofs0##_fpix01 = v_cvt_f32(i##ofs0##_pix01), i##ofs0##_fpix23 = v_cvt_f32(i##ofs0##_pix23); \
    v_float32 i##ofs1##_fpix01 = v_cvt_f32(i##ofs1##_pix01), i##ofs1##_fpix23 = v_cvt_f32(i##ofs1##_pix23); \
    v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11, \
              i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
    v_recombine(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
    v_recombine(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
    v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[i+ofs0]), \
              i##ofs1##_alpha = vx_setall_f32(valpha[i+ofs1]), \
              i##ofs0##_beta  = vx_setall_f32(vbeta[i+ofs0]), \
              i##ofs1##_beta  = vx_setall_f32(vbeta[i+ofs1]); \
    v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
              i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    auto i##ofs0##ofs1##_pix00 = v_round(i##ofs0##ofs1##_fpix00);
#define CV_WARP_SIMD256_LINEAR_SHUFFLE_INTER_32FC4_I(ofs0, ofs1) \
    const float *srcptr##ofs0 = src + addr[i+ofs0]; \
    const float *srcptr##ofs1 = src + addr[i+ofs1]; \
    v_float32 i##ofs0##_fpix01 = v256_load(srcptr##ofs0), \
              i##ofs0##_fpix23 = v256_load(srcptr##ofs0+srcstep); \
    v_float32 i##ofs1##_fpix01 = v256_load(srcptr##ofs1), \
              i##ofs1##_fpix23 = v256_load(srcptr##ofs1+srcstep); \
    v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11, \
              i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
    v_recombine(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
    v_recombine(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
    v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[i+ofs0]), \
              i##ofs1##_alpha = vx_setall_f32(valpha[i+ofs1]), \
              i##ofs0##_beta  = vx_setall_f32(vbeta[i+ofs0]), \
              i##ofs1##_beta  = vx_setall_f32(vbeta[i+ofs1]); \
    v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
              i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00);
#define CV_WARP_SIMD256_LINEAR_STORE_8UC4_I() \
    v_pack_store(dstptr + 4*(x+i), v_pack_u(i01_pix00, i23_pix00)); \
    v_pack_store(dstptr + 4*(x+i+4), v_pack_u(i45_pix00, i67_pix00));
#define CV_WARP_SIMD256_LINEAR_STORE_16UC4_I() \
    vx_store(dstptr + 4*(x+i), v_pack_u(i01_pix00, i23_pix00)); \
    vx_store(dstptr + 4*(x+i+4), v_pack_u(i45_pix00, i67_pix00));
#define CV_WARP_SIMD256_LINEAR_STORE_32FC4_I() \
    vx_store(dstptr + 4*(x+i),    i01_fpix00); \
    vx_store(dstptr + 4*(x+i)+8,  i23_fpix00); \
    vx_store(dstptr + 4*(x+i)+16, i45_fpix00); \
    vx_store(dstptr + 4*(x+i)+24, i67_fpix00);
#define CV_WARP_SIMD256_SHUFFLE_INTER_STORE_C4(INTER, DEPTH) \
    for (int i = 0; i < uf; i+=vlanes_32) { \
        CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(0, 1) \
        CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(2, 3) \
        CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(4, 5) \
        CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(6, 7) \
        CV_WARP_SIMD256_##INTER##_STORE_##DEPTH##C4_I() \
    }
// SIMD_SCALABLE (SIMDX), c4, nearest
#define CV_WARP_SIMDX_NEAREST_SHUFFLE_INTER_8UC4_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[i+ofs]; \
    v_uint32 i##ofs##_pix0 = v_load_expand_q<4>(srcptr##ofs);
#define CV_WARP_SIMDX_NEAREST_SHUFFLE_INTER_16UC4_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[i+ofs]; \
    v_uint32 i##ofs##_pix0 = v_load_expand<4>(srcptr##ofs);
#define CV_WARP_SIMDX_NEAREST_SHUFFLE_INTER_32FC4_I(ofs) \
    const float *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_fpix0 = v_load<4>(srcptr##ofs);
#define CV_WARP_SIMDX_NEAREST_STORE_8UC4_I() \
    v_pack_store<8>(dstptr + 4*(x+i), v_pack<4>(i0_pix0, i1_pix0)); \
    v_pack_store<8>(dstptr + 4*(x+i+2), v_pack<4>(i2_pix0, i3_pix0));
#define CV_WARP_SIMDX_NEAREST_STORE_16UC4_I() \
    v_store<8>(dstptr + 4*(x+i), v_pack<4>(i0_pix0, i1_pix0)); \
    v_store<8>(dstptr + 4*(x+i+2), v_pack<4>(i2_pix0, i3_pix0));
#define CV_WARP_SIMDX_NEAREST_STORE_32FC4_I() \
    v_store<4>(dstptr + 4*(x+i),    i0_fpix0); \
    v_store<4>(dstptr + 4*(x+i)+4,  i1_fpix0); \
    v_store<4>(dstptr + 4*(x+i)+8,  i2_fpix0); \
    v_store<4>(dstptr + 4*(x+i)+12, i3_fpix0);
// SIMD_SCALABLE (SIMDX), c4, bilinear
#define CV_WARP_SIMDX_LINEAR_SHUFFLE_INTER_8UC4_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs))), \
              i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs+4))), \
              i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs+srcstep))), \
              i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs+srcstep+4))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]); \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0); \
    auto i##ofs##_pix0 = v_round(i##ofs##_fpix0);
#define CV_WARP_SIMDX_LINEAR_SHUFFLE_INTER_16UC4_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs))), \
              i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs+4))), \
              i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs+srcstep))), \
              i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs+srcstep+4))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]); \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0); \
    auto i##ofs##_pix0 = v_round(i##ofs##_fpix0);
#define CV_WARP_SIMDX_LINEAR_SHUFFLE_INTER_32FC4_I(ofs) \
    const float *srcptr##ofs = src + addr[i+ofs]; \
    v_float32 i##ofs##_fpix0 = v_load<4>(srcptr##ofs), \
              i##ofs##_fpix1 = v_load<4>(srcptr##ofs+4), \
              i##ofs##_fpix2 = v_load<4>(srcptr##ofs+srcstep), \
              i##ofs##_fpix3 = v_load<4>(srcptr##ofs+srcstep+4); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]); \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0);
#define CV_WARP_SIMDX_LINEAR_STORE_8UC4_I() \
    v_pack_store<8>(dstptr + 4*(x+i),   v_pack_u<4>(i0_pix0, i1_pix0)); \
    v_pack_store<8>(dstptr + 4*(x+i+2), v_pack_u<4>(i2_pix0, i3_pix0));
#define CV_WARP_SIMDX_LINEAR_STORE_16UC4_I() \
    v_store<8>(dstptr + 4*(x+i),   v_pack_u<4>(i0_pix0, i1_pix0)); \
    v_store<8>(dstptr + 4*(x+i+2), v_pack_u<4>(i2_pix0, i3_pix0));
#define CV_WARP_SIMDX_LINEAR_STORE_32FC4_I() \
    v_store<4>(dstptr + 4*(x+i),    i0_fpix0); \
    v_store<4>(dstptr + 4*(x+i)+4,  i1_fpix0); \
    v_store<4>(dstptr + 4*(x+i)+8,  i2_fpix0); \
    v_store<4>(dstptr + 4*(x+i)+12, i3_fpix0);
#define CV_WARP_SIMDX_SHUFFLE_INTER_STORE_C4(INTER, DEPTH) \
    for (int i = 0; i < uf; i+=4) { \
        CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(0); \
        CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(1); \
        CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(2); \
        CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C4_I(3); \
        CV_WARP_SIMDX_##INTER##_STORE_##DEPTH##C4_I(); \
    }

// Special case for C3 shuffle, interpolation and store
// SIMD128, c3, nearest
#define CV_WARP_SIMD128_NEAREST_SHUFFLE_INTER_8UC3_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[ofs]; \
    v_uint32 i##ofs##_pix0 = vx_load_expand_q(srcptr##ofs);
#define CV_WARP_SIMD128_NEAREST_SHUFFLE_INTER_16UC3_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[ofs]; \
    v_uint32 i##ofs##_pix0 = vx_load_expand(srcptr##ofs);
#define CV_WARP_SIMD128_NEAREST_SHUFFLE_INTER_32FC3_I(ofs) \
    const float *srcptr##ofs = src + addr[ofs]; \
  v_float32 i##ofs##_pix0 = vx_load(srcptr##ofs);
// #define CV_WARP_SIMD128_NEAREST_STORE_8UC3_I() \
//     v_pack_store(dstptr + 3*(x), v_rotate_right<1>(v_pack(v_rotate_left<1>(i0_pix0), i1_pix0))); \
//     v_pack_store(dstptr + 3*(x+2), v_rotate_right<1>(v_pack(v_rotate_left<1>(i2_pix0), i3_pix0))); \
//     v_pack_store(dstptr + 3*(x+4), v_rotate_right<1>(v_pack(v_rotate_left<1>(i4_pix0), i5_pix0))); \
//     v_pack_store(dstptr + 3*(x+6), v_rotate_right<1>(v_pack(v_rotate_left<1>(i6_pix0), i7_pix0)));
// #define CV_WARP_SIMD128_NEAREST_STORE_16UC3_I() \
//     vx_store(dstptr + 3*(x), v_rotate_right<1>(v_pack(v_rotate_left<1>(i0_pix0), i1_pix0))); \
//     vx_store(dstptr + 3*(x+2), v_rotate_right<1>(v_pack(v_rotate_left<1>(i2_pix0), i3_pix0))); \
//     vx_store(dstptr + 3*(x+4), v_rotate_right<1>(v_pack(v_rotate_left<1>(i4_pix0), i5_pix0))); \
//     vx_store(dstptr + 3*(x+6), v_rotate_right<1>(v_pack(v_rotate_left<1>(i6_pix0), i7_pix0)));
#define CV_WARP_SIMD128_NEAREST_STORE_8UC3_I() \
    uint32_t tmp_buf[max_vlanes_16*4]; \
    vx_store(tmp_buf,       i0_pix0); \
    vx_store(tmp_buf + 3,   i1_pix0); \
    vx_store(tmp_buf + 3*2, i2_pix0); \
    vx_store(tmp_buf + 3*3, i3_pix0); \
    vx_store(tmp_buf + 3*4, i4_pix0); \
    vx_store(tmp_buf + 3*5, i5_pix0); \
    vx_store(tmp_buf + 3*6, i6_pix0); \
    vx_store(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
    v_pack_store(dstptr + 3*x,             pix0); \
    v_pack_store(dstptr + 3*x+vlanes_16,   pix1); \
    v_pack_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD128_NEAREST_STORE_16UC3_I() \
    uint32_t tmp_buf[max_vlanes_16*4]; \
    vx_store(tmp_buf,       i0_pix0); \
    vx_store(tmp_buf + 3,   i1_pix0); \
    vx_store(tmp_buf + 3*2, i2_pix0); \
    vx_store(tmp_buf + 3*3, i3_pix0); \
    vx_store(tmp_buf + 3*4, i4_pix0); \
    vx_store(tmp_buf + 3*5, i5_pix0); \
    vx_store(tmp_buf + 3*6, i6_pix0); \
    vx_store(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
    vx_store(dstptr + 3*x,             pix0); \
    vx_store(dstptr + 3*x+vlanes_16,   pix1); \
    vx_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD128_NEAREST_STORE_32FC3_I() \
    if (rightmost) { \
        float tmp_buf[max_vlanes_32*4]; \
        vx_store(tmp_buf,       i0_pix0); \
        vx_store(tmp_buf + 3,   i1_pix0); \
        vx_store(tmp_buf + 3*2, i2_pix0); \
        vx_store(tmp_buf + 3*3, i3_pix0); \
        vx_store(dstptr + 3*x,             vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32,   vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*2, vx_load(tmp_buf + vlanes_32*2)); \
        vx_store(tmp_buf,       i4_pix0); \
        vx_store(tmp_buf + 3,   i5_pix0); \
        vx_store(tmp_buf + 3*2, i6_pix0); \
        vx_store(tmp_buf + 3*3, i7_pix0); \
        vx_store(dstptr + 3*x+vlanes_32*3, vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32*4, vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*5, vx_load(tmp_buf + vlanes_32*2)); \
    } else { \
        vx_store(dstptr + 3*(x),   i0_pix0); \
        vx_store(dstptr + 3*(x+1), i1_pix0); \
        vx_store(dstptr + 3*(x+2), i2_pix0); \
        vx_store(dstptr + 3*(x+3), i3_pix0); \
        vx_store(dstptr + 3*(x+4), i4_pix0); \
        vx_store(dstptr + 3*(x+5), i5_pix0); \
        vx_store(dstptr + 3*(x+6), i6_pix0); \
        vx_store(dstptr + 3*(x+7), i7_pix0); \
    }
// SIMD128, c3, bilinear
#define CV_WARP_SIMD128_LINEAR_SHUFFLE_INTER_8UC3_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs))); \
    v_float32 i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs+3))); \
    v_float32 i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs+srcstep))); \
    v_float32 i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(srcptr##ofs+srcstep+3))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[ofs]);  \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0); \
    auto i##ofs##_pix0 = v_round(i##ofs##_fpix0);
#define CV_WARP_SIMD128_LINEAR_SHUFFLE_INTER_16UC3_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs))); \
    v_float32 i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs+3))); \
    v_float32 i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs+srcstep))); \
    v_float32 i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(srcptr##ofs+srcstep+3))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[ofs]);  \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0); \
    auto i##ofs##_pix0 = v_round(i##ofs##_fpix0);
#define CV_WARP_SIMD128_LINEAR_SHUFFLE_INTER_32FC3_I(ofs) \
    const float *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_pix0 = vx_load(srcptr##ofs); \
    v_float32 i##ofs##_pix1 = vx_load(srcptr##ofs+3); \
    v_float32 i##ofs##_pix2 = vx_load(srcptr##ofs+srcstep); \
    v_float32 i##ofs##_pix3 = vx_load(srcptr##ofs+srcstep+3); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[ofs]);  \
    i##ofs##_pix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix1, i##ofs##_pix0), i##ofs##_pix0); \
    i##ofs##_pix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix3, i##ofs##_pix2), i##ofs##_pix2); \
    i##ofs##_pix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_pix2, i##ofs##_pix0), i##ofs##_pix0);
// #define CV_WARP_SIMD128_LINEAR_STORE_8UC3_I() \
//     v_uint16 i01_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(i0_pix0), i1_pix0)); \
//     v_uint16 i23_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(i2_pix0), i3_pix0)); \
//     v_uint16 i45_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(i4_pix0), i5_pix0)); \
//     v_uint16 i67_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(i6_pix0), i7_pix0)); \
//     v_pack_store(dstptr + 3*(x),   i01_pix); \
//     v_pack_store(dstptr + 3*(x+2), i23_pix); \
//     v_pack_store(dstptr + 3*(x+4), i45_pix); \
//     v_pack_store(dstptr + 3*(x+6), i67_pix);
// #define CV_WARP_SIMD128_LINEAR_STORE_16UC3_I() \
//     v_uint16 i01_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(v_round(i0_pix0)), v_round(i1_pix0))); \
//     v_uint16 i23_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(v_round(i2_pix0)), v_round(i3_pix0))); \
//     v_uint16 i45_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(v_round(i4_pix0)), v_round(i5_pix0))); \
//     v_uint16 i67_pix = v_rotate_right<1>(v_pack_u(v_rotate_left<1>(v_round(i6_pix0)), v_round(i7_pix0))); \
//     vx_store(dstptr + 3*(x),   i01_pix); \
//     vx_store(dstptr + 3*(x+2), i23_pix); \
//     vx_store(dstptr + 3*(x+4), i45_pix); \
//     vx_store(dstptr + 3*(x+6), i67_pix);
#define CV_WARP_SIMD128_LINEAR_STORE_8UC3_I() \
    int32_t tmp_buf[max_vlanes_16*4]; \
    vx_store(tmp_buf,       i0_pix0); \
    vx_store(tmp_buf + 3,   i1_pix0); \
    vx_store(tmp_buf + 3*2, i2_pix0); \
    vx_store(tmp_buf + 3*3, i3_pix0); \
    vx_store(tmp_buf + 3*4, i4_pix0); \
    vx_store(tmp_buf + 3*5, i5_pix0); \
    vx_store(tmp_buf + 3*6, i6_pix0); \
    vx_store(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack_u(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack_u(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack_u(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
    v_pack_store(dstptr + 3*x,             pix0); \
    v_pack_store(dstptr + 3*x+vlanes_16,   pix1); \
    v_pack_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD128_LINEAR_STORE_16UC3_I() \
    int32_t tmp_buf[max_vlanes_16*4]; \
    vx_store(tmp_buf,       i0_pix0); \
    vx_store(tmp_buf + 3,   i1_pix0); \
    vx_store(tmp_buf + 3*2, i2_pix0); \
    vx_store(tmp_buf + 3*3, i3_pix0); \
    vx_store(tmp_buf + 3*4, i4_pix0); \
    vx_store(tmp_buf + 3*5, i5_pix0); \
    vx_store(tmp_buf + 3*6, i6_pix0); \
    vx_store(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack_u(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack_u(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack_u(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
    vx_store(dstptr + 3*x,             pix0); \
    vx_store(dstptr + 3*x+vlanes_16,   pix1); \
    vx_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD128_LINEAR_STORE_32FC3_I() \
    if (rightmost) { \
        float tmp_buf[max_vlanes_32*4]; \
        vx_store(tmp_buf,       i0_pix0); \
        vx_store(tmp_buf + 3,   i1_pix0); \
        vx_store(tmp_buf + 3*2, i2_pix0); \
        vx_store(tmp_buf + 3*3, i3_pix0); \
        vx_store(dstptr + 3*x,             vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32,   vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*2, vx_load(tmp_buf + vlanes_32*2)); \
        vx_store(tmp_buf,       i4_pix0); \
        vx_store(tmp_buf + 3,   i5_pix0); \
        vx_store(tmp_buf + 3*2, i6_pix0); \
        vx_store(tmp_buf + 3*3, i7_pix0); \
        vx_store(dstptr + 3*x+vlanes_32*3, vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32*4, vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*5, vx_load(tmp_buf + vlanes_32*2)); \
    } else { \
        vx_store(dstptr + 3*(x),   i0_pix0); \
        vx_store(dstptr + 3*(x+1), i1_pix0); \
        vx_store(dstptr + 3*(x+2), i2_pix0); \
        vx_store(dstptr + 3*(x+3), i3_pix0); \
        vx_store(dstptr + 3*(x+4), i4_pix0); \
        vx_store(dstptr + 3*(x+5), i5_pix0); \
        vx_store(dstptr + 3*(x+6), i6_pix0); \
        vx_store(dstptr + 3*(x+7), i7_pix0); \
    }
#define CV_WARP_SIMD128_SHUFFLE_INTER_STORE_C3(INTER, DEPTH) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(0) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(1) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(2) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(3) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(4) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(5) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(6) \
    CV_WARP_SIMD128_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(7) \
    CV_WARP_SIMD128_##INTER##_STORE_##DEPTH##C3_I() \
// SIMD256, c3, nearest
#define CV_WARP_SIMD256_NEAREST_SHUFFLE_INTER_8UC3_I(ofs0, ofs1) \
    const uint8_t *srcptr##ofs0 = src + addr[ofs0]; \
    const uint8_t *srcptr##ofs1 = src + addr[ofs1]; \
    v_uint32 i##ofs0##_pix0x = v256_load_expand_q(srcptr##ofs0); \
    v_uint32 i##ofs1##_pix0x = v256_load_expand_q(srcptr##ofs1); \
    i##ofs0##_pix0x = v_rotate_left<1>(i##ofs0##_pix0x); \
    v_uint32 i##ofs0##ofs1##_pix00 = v_combine_low(i##ofs0##_pix0x, i##ofs1##_pix0x); \
    i##ofs0##ofs1##_pix00 = v_rotate_right<1>(i##ofs0##ofs1##_pix00);
#define CV_WARP_SIMD256_NEAREST_SHUFFLE_INTER_16UC3_I(ofs0, ofs1) \
    const uint16_t *srcptr##ofs0 = src + addr[ofs0]; \
    const uint16_t *srcptr##ofs1 = src + addr[ofs1]; \
    v_uint32 i##ofs0##_pix0x = v256_load_expand(srcptr##ofs0); \
    v_uint32 i##ofs1##_pix0x = v256_load_expand(srcptr##ofs1); \
    i##ofs0##_pix0x = v_rotate_left<1>(i##ofs0##_pix0x); \
    v_uint32 i##ofs0##ofs1##_pix00 = v_combine_low(i##ofs0##_pix0x, i##ofs1##_pix0x); \
    i##ofs0##ofs1##_pix00 = v_rotate_right<1>(i##ofs0##ofs1##_pix00);
#define CV_WARP_SIMD256_NEAREST_SHUFFLE_INTER_32FC3_I(ofs0, ofs1) \
    const float *srcptr##ofs0 = src + addr[ofs0]; \
    const float *srcptr##ofs1 = src + addr[ofs1]; \
    v_float32 i##ofs0##ofs1##_pix00 = vx_load_halves(srcptr##ofs0, srcptr##ofs1); \
    v_float32 i##ofs0##ofs1##_pix00_rl1 = v_rotate_left<1>(i##ofs0##ofs1##_pix00); \
    i##ofs0##ofs1##_pix00 = v256_combine_diagonal(i##ofs0##ofs1##_pix00_rl1, i##ofs0##ofs1##_pix00); \
    i##ofs0##ofs1##_pix00 = v_rotate_right<1>(i##ofs0##ofs1##_pix00);
#define CV_WARP_SIMD256_NEAREST_STORE_8UC3_I() \
        uint32_t tmp_buf[max_vlanes_16*4]; \
        vx_store(tmp_buf,        i01_pix00); \
        vx_store(tmp_buf + 3*2,  i23_pix00); \
        vx_store(tmp_buf + 3*4,  i45_pix00); \
        vx_store(tmp_buf + 3*6,  i67_pix00); \
        vx_store(tmp_buf + 3*8,  i89_pix00); \
        vx_store(tmp_buf + 3*10, i1011_pix00); \
        vx_store(tmp_buf + 3*12, i1213_pix00); \
        vx_store(tmp_buf + 3*14, i1415_pix00); \
        v_uint16 pix0 = v_pack(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
        v_uint16 pix1 = v_pack(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
        v_uint16 pix2 = v_pack(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
        v_pack_store(dstptr + 3*x,             pix0); \
        v_pack_store(dstptr + 3*x+vlanes_16,   pix1); \
        v_pack_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD256_NEAREST_STORE_16UC3_I() \
        uint32_t tmp_buf[max_vlanes_16*4]; \
        vx_store(tmp_buf,        i01_pix00); \
        vx_store(tmp_buf + 3*2,  i23_pix00); \
        vx_store(tmp_buf + 3*4,  i45_pix00); \
        vx_store(tmp_buf + 3*6,  i67_pix00); \
        vx_store(tmp_buf + 3*8,  i89_pix00); \
        vx_store(tmp_buf + 3*10, i1011_pix00); \
        vx_store(tmp_buf + 3*12, i1213_pix00); \
        vx_store(tmp_buf + 3*14, i1415_pix00); \
        v_uint16 pix0 = v_pack(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
        v_uint16 pix1 = v_pack(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
        v_uint16 pix2 = v_pack(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
        vx_store(dstptr + 3*x,             pix0); \
        vx_store(dstptr + 3*x+vlanes_16,   pix1); \
        vx_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD256_NEAREST_STORE_32FC3_I() \
    if (rightmost) { \
        float tmp_buf[max_vlanes_32*4]; \
        vx_store(tmp_buf,      i01_pix00); \
        vx_store(tmp_buf + 6,  i23_pix00); \
        vx_store(tmp_buf + 12, i45_pix00); \
        vx_store(tmp_buf + 18, i67_pix00); \
        vx_store(dstptr + 3*x,             vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32,   vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*2, vx_load(tmp_buf + vlanes_32*2)); \
        vx_store(tmp_buf,      i89_pix00); \
        vx_store(tmp_buf + 6,  i1011_pix00); \
        vx_store(tmp_buf + 12, i1213_pix00); \
        vx_store(tmp_buf + 18, i1415_pix00); \
        vx_store(dstptr + 3*x+vlanes_32*3, vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32*4, vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*5, vx_load(tmp_buf + vlanes_32*2)); \
    } else { \
        vx_store(dstptr + 3*(x),   i01_pix00); \
        vx_store(dstptr + 3*(x+2), i23_pix00); \
        vx_store(dstptr + 3*(x+4), i45_pix00); \
        vx_store(dstptr + 3*(x+6), i67_pix00); \
        vx_store(dstptr + 3*(x+vlanes_32),   i89_pix00); \
        vx_store(dstptr + 3*(x+vlanes_32+2), i1011_pix00); \
        vx_store(dstptr + 3*(x+vlanes_32+4), i1213_pix00); \
        vx_store(dstptr + 3*(x+vlanes_32+6), i1415_pix00); \
    }
// SIMD256, c3, bilinear
#define CV_WARP_SIMD256_LINEAR_SHUFFLE_INTER_8UC3_I(ofs0, ofs1) \
    const uint8_t *srcptr##ofs0 = src + addr[ofs0]; \
    const uint8_t *srcptr##ofs1 = src + addr[ofs1]; \
    v_int32 i##ofs0##_pix01 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs0)), \
            i##ofs0##_pix23 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs0+srcstep)); \
    v_int32 i##ofs1##_pix01 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs1)), \
            i##ofs1##_pix23 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs1+srcstep)); \
    v_int32 i##ofs0##_pix01_rl1 = v_rotate_left<1>(i##ofs0##_pix01); \
    v_int32 i##ofs0##_pix23_rl1 = v_rotate_left<1>(i##ofs0##_pix23); \
    v_int32 i##ofs1##_pix01_rl1 = v_rotate_left<1>(i##ofs1##_pix01); \
    v_int32 i##ofs1##_pix23_rl1 = v_rotate_left<1>(i##ofs1##_pix23); \
    i##ofs0##_pix01 = v256_combine_diagonal(i##ofs0##_pix01, i##ofs0##_pix01_rl1); \
    i##ofs0##_pix23 = v256_combine_diagonal(i##ofs0##_pix23, i##ofs0##_pix23_rl1); \
    i##ofs1##_pix01 = v256_combine_diagonal(i##ofs1##_pix01, i##ofs1##_pix01_rl1); \
    i##ofs1##_pix23 = v256_combine_diagonal(i##ofs1##_pix23, i##ofs1##_pix23_rl1); \
    v_float32 i##ofs0##_fpix01 = v_cvt_f32(i##ofs0##_pix01), i##ofs0##_fpix23 = v_cvt_f32(i##ofs0##_pix23); \
    v_float32 i##ofs1##_fpix01 = v_cvt_f32(i##ofs1##_pix01), i##ofs1##_fpix23 = v_cvt_f32(i##ofs1##_pix23); \
    v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11, \
              i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
    v_recombine(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
    v_recombine(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
    v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[ofs0]), \
              i##ofs1##_alpha = vx_setall_f32(valpha[ofs1]), \
              i##ofs0##_beta  = vx_setall_f32(vbeta[ofs0]), \
              i##ofs1##_beta  = vx_setall_f32(vbeta[ofs1]); \
    v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
              i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    v_float32 i##ofs0##ofs1##_fpix00_rl1 = v_rotate_left<1>(i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix00 = v256_combine_diagonal(i##ofs0##ofs1##_fpix00_rl1, i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix00 = v_rotate_right<1>(i##ofs0##ofs1##_fpix00); \
    auto i##ofs0##ofs1##_pix00 = v_round(i##ofs0##ofs1##_fpix00);
#define CV_WARP_SIMD256_LINEAR_SHUFFLE_INTER_16UC3_I(ofs0, ofs1) \
    const uint16_t *srcptr##ofs0 = src + addr[ofs0]; \
    const uint16_t *srcptr##ofs1 = src + addr[ofs1]; \
    v_int32 i##ofs0##_pix01 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs0)), \
            i##ofs0##_pix23 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs0+srcstep)); \
    v_int32 i##ofs1##_pix01 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs1)), \
            i##ofs1##_pix23 = v_reinterpret_as_s32(v256_load_expand(srcptr##ofs1+srcstep)); \
    v_int32 i##ofs0##_pix01_rl1 = v_rotate_left<1>(i##ofs0##_pix01); \
    v_int32 i##ofs0##_pix23_rl1 = v_rotate_left<1>(i##ofs0##_pix23); \
    v_int32 i##ofs1##_pix01_rl1 = v_rotate_left<1>(i##ofs1##_pix01); \
    v_int32 i##ofs1##_pix23_rl1 = v_rotate_left<1>(i##ofs1##_pix23); \
    i##ofs0##_pix01 = v256_combine_diagonal(i##ofs0##_pix01, i##ofs0##_pix01_rl1); \
    i##ofs0##_pix23 = v256_combine_diagonal(i##ofs0##_pix23, i##ofs0##_pix23_rl1); \
    i##ofs1##_pix01 = v256_combine_diagonal(i##ofs1##_pix01, i##ofs1##_pix01_rl1); \
    i##ofs1##_pix23 = v256_combine_diagonal(i##ofs1##_pix23, i##ofs1##_pix23_rl1); \
    v_float32 i##ofs0##_fpix01 = v_cvt_f32(i##ofs0##_pix01), i##ofs0##_fpix23 = v_cvt_f32(i##ofs0##_pix23); \
    v_float32 i##ofs1##_fpix01 = v_cvt_f32(i##ofs1##_pix01), i##ofs1##_fpix23 = v_cvt_f32(i##ofs1##_pix23); \
    v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11, \
            i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
    v_recombine(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
    v_recombine(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
    v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[ofs0]), \
              i##ofs1##_alpha = vx_setall_f32(valpha[ofs1]), \
              i##ofs0##_beta  = vx_setall_f32(vbeta[ofs0]), \
              i##ofs1##_beta  = vx_setall_f32(vbeta[ofs1]); \
    v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
              i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    v_float32 i##ofs0##ofs1##_fpix00_rl1 = v_rotate_left<1>(i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix00 = v256_combine_diagonal(i##ofs0##ofs1##_fpix00_rl1, i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix00 = v_rotate_right<1>(i##ofs0##ofs1##_fpix00); \
    auto i##ofs0##ofs1##_pix00 = v_round(i##ofs0##ofs1##_fpix00);
#define CV_WARP_SIMD256_LINEAR_SHUFFLE_INTER_32FC3_I(ofs0, ofs1) \
    const float *srcptr##ofs0 = src + addr[ofs0]; \
    const float *srcptr##ofs1 = src + addr[ofs1]; \
    v_float32 i##ofs0##_fpix01 = v256_load(srcptr##ofs0); \
    v_float32 i##ofs0##_fpix23 = v256_load(srcptr##ofs0+srcstep); \
    v_float32 i##ofs1##_fpix01 = v256_load(srcptr##ofs1); \
    v_float32 i##ofs1##_fpix23 = v256_load(srcptr##ofs1+srcstep); \
    v_float32 i##ofs0##_fpix01_rl1 = v_rotate_left<1>(i##ofs0##_fpix01); \
    v_float32 i##ofs0##_fpix23_rl1 = v_rotate_left<1>(i##ofs0##_fpix23); \
    v_float32 i##ofs1##_fpix01_rl1 = v_rotate_left<1>(i##ofs1##_fpix01); \
    v_float32 i##ofs1##_fpix23_rl1 = v_rotate_left<1>(i##ofs1##_fpix23); \
    i##ofs0##_fpix01 = v256_combine_diagonal(i##ofs0##_fpix01, i##ofs0##_fpix01_rl1); \
    i##ofs0##_fpix23 = v256_combine_diagonal(i##ofs0##_fpix23, i##ofs0##_fpix23_rl1); \
    i##ofs1##_fpix01 = v256_combine_diagonal(i##ofs1##_fpix01, i##ofs1##_fpix01_rl1); \
    i##ofs1##_fpix23 = v256_combine_diagonal(i##ofs1##_fpix23, i##ofs1##_fpix23_rl1); \
    v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11; \
    v_float32 i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
    v_recombine(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
    v_recombine(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
    v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[ofs0]), \
              i##ofs1##_alpha = vx_setall_f32(valpha[ofs1]), \
              i##ofs0##_beta  = vx_setall_f32(vbeta[ofs0]), \
              i##ofs1##_beta  = vx_setall_f32(vbeta[ofs1]); \
    v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
              i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
    i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
    v_float32 i##ofs0##ofs1##_fpix00_rl1 = v_rotate_left<1>(i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix00 = v256_combine_diagonal(i##ofs0##ofs1##_fpix00_rl1, i##ofs0##ofs1##_fpix00); \
    i##ofs0##ofs1##_fpix00 = v_rotate_right<1>(i##ofs0##ofs1##_fpix00);
#define CV_WARP_SIMD256_LINEAR_STORE_8UC3_I() \
        int32_t tmp_buf[max_vlanes_16*4]; \
        vx_store(tmp_buf,        i01_pix00); \
        vx_store(tmp_buf + 3*2,  i23_pix00); \
        vx_store(tmp_buf + 3*4,  i45_pix00); \
        vx_store(tmp_buf + 3*6,  i67_pix00); \
        vx_store(tmp_buf + 3*8,  i89_pix00); \
        vx_store(tmp_buf + 3*10, i1011_pix00); \
        vx_store(tmp_buf + 3*12, i1213_pix00); \
        vx_store(tmp_buf + 3*14, i1415_pix00); \
        v_uint16 pix0 = v_pack_u(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
        v_uint16 pix1 = v_pack_u(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
        v_uint16 pix2 = v_pack_u(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
        v_pack_store(dstptr + 3*x,             pix0); \
        v_pack_store(dstptr + 3*x+vlanes_16,   pix1); \
        v_pack_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD256_LINEAR_STORE_16UC3_I() \
        int32_t tmp_buf[max_vlanes_16*4]; \
        vx_store(tmp_buf,        i01_pix00); \
        vx_store(tmp_buf + 3*2,  i23_pix00); \
        vx_store(tmp_buf + 3*4,  i45_pix00); \
        vx_store(tmp_buf + 3*6,  i67_pix00); \
        vx_store(tmp_buf + 3*8,  i89_pix00); \
        vx_store(tmp_buf + 3*10, i1011_pix00); \
        vx_store(tmp_buf + 3*12, i1213_pix00); \
        vx_store(tmp_buf + 3*14, i1415_pix00); \
        v_uint16 pix0 = v_pack_u(vx_load(tmp_buf),             vx_load(tmp_buf+vlanes_32)); \
        v_uint16 pix1 = v_pack_u(vx_load(tmp_buf+vlanes_32*2), vx_load(tmp_buf+vlanes_32*3)); \
        v_uint16 pix2 = v_pack_u(vx_load(tmp_buf+vlanes_32*4), vx_load(tmp_buf+vlanes_32*5)); \
        vx_store(dstptr + 3*x,             pix0); \
        vx_store(dstptr + 3*x+vlanes_16,   pix1); \
        vx_store(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMD256_LINEAR_STORE_32FC3_I() \
    if (rightmost) { \
        float tmp_buf[max_vlanes_32*4]; \
        vx_store(tmp_buf,      i01_fpix00); \
        vx_store(tmp_buf + 6,  i23_fpix00); \
        vx_store(tmp_buf + 12, i45_fpix00); \
        vx_store(tmp_buf + 18, i67_fpix00); \
        vx_store(dstptr + 3*x,             vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32,   vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*2, vx_load(tmp_buf + vlanes_32*2)); \
        vx_store(tmp_buf,      i89_fpix00); \
        vx_store(tmp_buf + 6,  i1011_fpix00); \
        vx_store(tmp_buf + 12, i1213_fpix00); \
        vx_store(tmp_buf + 18, i1415_fpix00); \
        vx_store(dstptr + 3*x+vlanes_32*3, vx_load(tmp_buf)); \
        vx_store(dstptr + 3*x+vlanes_32*4, vx_load(tmp_buf + vlanes_32)); \
        vx_store(dstptr + 3*x+vlanes_32*5, vx_load(tmp_buf + vlanes_32*2)); \
    } else { \
        vx_store(dstptr + 3*(x),   i01_fpix00); \
        vx_store(dstptr + 3*(x+2), i23_fpix00); \
        vx_store(dstptr + 3*(x+4), i45_fpix00); \
        vx_store(dstptr + 3*(x+6), i67_fpix00); \
        vx_store(dstptr + 3*(x+vlanes_32),   i89_fpix00); \
        vx_store(dstptr + 3*(x+vlanes_32+2), i1011_fpix00); \
        vx_store(dstptr + 3*(x+vlanes_32+4), i1213_fpix00); \
        vx_store(dstptr + 3*(x+vlanes_32+6), i1415_fpix00); \
    }
#define CV_WARP_SIMD256_SHUFFLE_INTER_STORE_C3(INTER, DEPTH) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(0, 1) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(2, 3) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(4, 5) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(6, 7) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(8, 9) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(10, 11) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(12, 13) \
    CV_WARP_SIMD256_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(14, 15) \
    CV_WARP_SIMD256_##INTER##_STORE_##DEPTH##C3_I()
// SIMD_SCALABLE (SIMDX), c3, nearest
#define CV_WARP_SIMDX_NEAREST_SHUFFLE_INTER_8UC3_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[ofs]; \
    v_uint32 i##ofs##_pix0 = v_load_expand_q<4>(srcptr##ofs);
#define CV_WARP_SIMDX_NEAREST_SHUFFLE_INTER_16UC3_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[ofs]; \
    v_uint32 i##ofs##_pix0 = v_load_expand<4>(srcptr##ofs);
#define CV_WARP_SIMDX_NEAREST_SHUFFLE_INTER_32FC3_I(ofs) \
    const float *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_fpix0 = v_load<4>(srcptr##ofs);
#define CV_WARP_SIMDX_NEAREST_STORE_8UC3_I() \
    uint32_t tmp_buf[max_vlanes_16*4]; \
    v_store<4>(tmp_buf,       i0_pix0); \
    v_store<4>(tmp_buf + 3,   i1_pix0); \
    v_store<4>(tmp_buf + 3*2, i2_pix0); \
    v_store<4>(tmp_buf + 3*3, i3_pix0); \
    v_store<4>(tmp_buf + 3*4, i4_pix0); \
    v_store<4>(tmp_buf + 3*5, i5_pix0); \
    v_store<4>(tmp_buf + 3*6, i6_pix0); \
    v_store<4>(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack(v_load<4>(tmp_buf),             v_load<4>(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack(v_load<4>(tmp_buf+vlanes_32*2), v_load<4>(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack(v_load<4>(tmp_buf+vlanes_32*4), v_load<4>(tmp_buf+vlanes_32*5)); \
    v_pack_store<8>(dstptr + 3*x,             pix0); \
    v_pack_store<8>(dstptr + 3*x+vlanes_16,   pix1); \
    v_pack_store<8>(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMDX_NEAREST_STORE_16UC3_I() \
    uint32_t tmp_buf[max_vlanes_16*4]; \
    v_store<4>(tmp_buf,       i0_pix0); \
    v_store<4>(tmp_buf + 3,   i1_pix0); \
    v_store<4>(tmp_buf + 3*2, i2_pix0); \
    v_store<4>(tmp_buf + 3*3, i3_pix0); \
    v_store<4>(tmp_buf + 3*4, i4_pix0); \
    v_store<4>(tmp_buf + 3*5, i5_pix0); \
    v_store<4>(tmp_buf + 3*6, i6_pix0); \
    v_store<4>(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack(v_load<4>(tmp_buf),             v_load<4>(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack(v_load<4>(tmp_buf+vlanes_32*2), v_load<4>(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack(v_load<4>(tmp_buf+vlanes_32*4), v_load<4>(tmp_buf+vlanes_32*5)); \
    v_store<8>(dstptr + 3*x,             pix0); \
    v_store<8>(dstptr + 3*x+vlanes_16,   pix1); \
    v_store<8>(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMDX_NEAREST_STORE_32FC3_I() \
    if (rightmost) { \
        float tmp_buf[max_vlanes_32*4]; \
        v_store<4>(tmp_buf,       i0_fpix0); \
        v_store<4>(tmp_buf + 3,   i1_fpix0); \
        v_store<4>(tmp_buf + 3*2, i2_fpix0); \
        v_store<4>(tmp_buf + 3*3, i3_fpix0); \
        v_store<4>(dstptr + 3*x,             v_load<4>(tmp_buf)); \
        v_store<4>(dstptr + 3*x+vlanes_32,   v_load<4>(tmp_buf + vlanes_32)); \
        v_store<4>(dstptr + 3*x+vlanes_32*2, v_load<4>(tmp_buf + vlanes_32*2)); \
        v_store<4>(tmp_buf,       i4_fpix0); \
        v_store<4>(tmp_buf + 3,   i5_fpix0); \
        v_store<4>(tmp_buf + 3*2, i6_fpix0); \
        v_store<4>(tmp_buf + 3*3, i7_fpix0); \
        v_store<4>(dstptr + 3*x+vlanes_32*3, v_load<4>(tmp_buf)); \
        v_store<4>(dstptr + 3*x+vlanes_32*4, v_load<4>(tmp_buf + vlanes_32)); \
        v_store<4>(dstptr + 3*x+vlanes_32*5, v_load<4>(tmp_buf + vlanes_32*2)); \
    } else { \
        v_store<4>(dstptr + 3*(x),   i0_fpix0); \
        v_store<4>(dstptr + 3*(x+1), i1_fpix0); \
        v_store<4>(dstptr + 3*(x+2), i2_fpix0); \
        v_store<4>(dstptr + 3*(x+3), i3_fpix0); \
        v_store<4>(dstptr + 3*(x+4), i4_fpix0); \
        v_store<4>(dstptr + 3*(x+5), i5_fpix0); \
        v_store<4>(dstptr + 3*(x+6), i6_fpix0); \
        v_store<4>(dstptr + 3*(x+7), i7_fpix0); \
    }
// SIMD_SCALABLE (SIMDX), c3, bilinear
#define CV_WARP_SIMDX_LINEAR_SHUFFLE_INTER_8UC3_I(ofs) \
    const uint8_t *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs))), \
              i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs+3))), \
              i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs+srcstep))), \
              i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q<4>(srcptr##ofs+srcstep+3))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[ofs]); \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0); \
    auto i##ofs##_pix0 = v_round(i##ofs##_fpix0);
#define CV_WARP_SIMDX_LINEAR_SHUFFLE_INTER_16UC3_I(ofs) \
    const uint16_t *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs))), \
              i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs+3))), \
              i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs+srcstep))), \
              i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(v_load_expand<4>(srcptr##ofs+srcstep+3))); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[ofs]); \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0); \
    auto i##ofs##_pix0 = v_round(i##ofs##_fpix0);
#define CV_WARP_SIMDX_LINEAR_SHUFFLE_INTER_32FC3_I(ofs) \
    const float *srcptr##ofs = src + addr[ofs]; \
    v_float32 i##ofs##_fpix0 = v_load<4>(srcptr##ofs), \
              i##ofs##_fpix1 = v_load<4>(srcptr##ofs+3), \
              i##ofs##_fpix2 = v_load<4>(srcptr##ofs+srcstep), \
              i##ofs##_fpix3 = v_load<4>(srcptr##ofs+srcstep+3); \
    v_float32 i##ofs##_alpha = vx_setall_f32(valpha[ofs]), \
              i##ofs##_beta  = vx_setall_f32(vbeta[ofs]); \
    i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
    i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
    i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0);
#define CV_WARP_SIMDX_LINEAR_STORE_8UC3_I() \
    int32_t tmp_buf[max_vlanes_16*4]; \
    v_store<4>(tmp_buf,       i0_pix0); \
    v_store<4>(tmp_buf + 3,   i1_pix0); \
    v_store<4>(tmp_buf + 3*2, i2_pix0); \
    v_store<4>(tmp_buf + 3*3, i3_pix0); \
    v_store<4>(tmp_buf + 3*4, i4_pix0); \
    v_store<4>(tmp_buf + 3*5, i5_pix0); \
    v_store<4>(tmp_buf + 3*6, i6_pix0); \
    v_store<4>(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack_u(v_load<4>(tmp_buf),             v_load<4>(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack_u(v_load<4>(tmp_buf+vlanes_32*2), v_load<4>(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack_u(v_load<4>(tmp_buf+vlanes_32*4), v_load<4>(tmp_buf+vlanes_32*5)); \
    v_pack_store<8>(dstptr + 3*x,             pix0); \
    v_pack_store<8>(dstptr + 3*x+vlanes_16,   pix1); \
    v_pack_store<8>(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMDX_LINEAR_STORE_16UC3_I() \
    int32_t tmp_buf[max_vlanes_16*4]; \
    v_store<4>(tmp_buf,       i0_pix0); \
    v_store<4>(tmp_buf + 3,   i1_pix0); \
    v_store<4>(tmp_buf + 3*2, i2_pix0); \
    v_store<4>(tmp_buf + 3*3, i3_pix0); \
    v_store<4>(tmp_buf + 3*4, i4_pix0); \
    v_store<4>(tmp_buf + 3*5, i5_pix0); \
    v_store<4>(tmp_buf + 3*6, i6_pix0); \
    v_store<4>(tmp_buf + 3*7, i7_pix0); \
    v_uint16 pix0 = v_pack_u(v_load<4>(tmp_buf),             v_load<4>(tmp_buf+vlanes_32)); \
    v_uint16 pix1 = v_pack_u(v_load<4>(tmp_buf+vlanes_32*2), v_load<4>(tmp_buf+vlanes_32*3)); \
    v_uint16 pix2 = v_pack_u(v_load<4>(tmp_buf+vlanes_32*4), v_load<4>(tmp_buf+vlanes_32*5)); \
    v_store<8>(dstptr + 3*x,             pix0); \
    v_store<8>(dstptr + 3*x+vlanes_16,   pix1); \
    v_store<8>(dstptr + 3*x+vlanes_16*2, pix2);
#define CV_WARP_SIMDX_LINEAR_STORE_32FC3_I() \
    if (rightmost) { \
        float tmp_buf[max_vlanes_32*4]; \
        v_store<4>(tmp_buf,       i0_fpix0); \
        v_store<4>(tmp_buf + 3,   i1_fpix0); \
        v_store<4>(tmp_buf + 3*2, i2_fpix0); \
        v_store<4>(tmp_buf + 3*3, i3_fpix0); \
        v_store<4>(dstptr + 3*x,             v_load<4>(tmp_buf)); \
        v_store<4>(dstptr + 3*x+vlanes_32,   v_load<4>(tmp_buf + vlanes_32)); \
        v_store<4>(dstptr + 3*x+vlanes_32*2, v_load<4>(tmp_buf + vlanes_32*2)); \
        v_store<4>(tmp_buf,       i4_fpix0); \
        v_store<4>(tmp_buf + 3,   i5_fpix0); \
        v_store<4>(tmp_buf + 3*2, i6_fpix0); \
        v_store<4>(tmp_buf + 3*3, i7_fpix0); \
        v_store<4>(dstptr + 3*x+vlanes_32*3, v_load<4>(tmp_buf)); \
        v_store<4>(dstptr + 3*x+vlanes_32*4, v_load<4>(tmp_buf + vlanes_32)); \
        v_store<4>(dstptr + 3*x+vlanes_32*5, v_load<4>(tmp_buf + vlanes_32*2)); \
    } else { \
        v_store<4>(dstptr + 3*(x),   i0_fpix0); \
        v_store<4>(dstptr + 3*(x+1), i1_fpix0); \
        v_store<4>(dstptr + 3*(x+2), i2_fpix0); \
        v_store<4>(dstptr + 3*(x+3), i3_fpix0); \
        v_store<4>(dstptr + 3*(x+4), i4_fpix0); \
        v_store<4>(dstptr + 3*(x+5), i5_fpix0); \
        v_store<4>(dstptr + 3*(x+6), i6_fpix0); \
        v_store<4>(dstptr + 3*(x+7), i7_fpix0); \
    }
#define CV_WARP_SIMDX_SHUFFLE_INTER_STORE_C3(INTER, DEPTH) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(0) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(1) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(2) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(3) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(4) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(5) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(6) \
    CV_WARP_SIMDX_##INTER##_SHUFFLE_INTER_##DEPTH##C3_I(7) \
    CV_WARP_SIMDX_##INTER##_STORE_##DEPTH##C3_I() \

#define CV_WARP_VECTOR_SHUFFLE_INTER_STORE(SIMD, INTER, DEPTH, CN) \
    CV_WARP_##SIMD##_SHUFFLE_INTER_STORE_##CN(INTER, DEPTH)
