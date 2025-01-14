/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_HAL_CORE_HPP_INCLUDED
#define OPENCV_FASTCV_HAL_CORE_HPP_INCLUDED

#include <opencv2/core/base.hpp>

#undef  cv_hal_lut
#define cv_hal_lut                  fastcv_hal_lut
#undef  cv_hal_normHammingDiff8u
#define cv_hal_normHammingDiff8u    fastcv_hal_normHammingDiff8u
#undef  cv_hal_mul8u16u
#define cv_hal_mul8u16u             fastcv_hal_mul8u16u
#undef  cv_hal_sub8u32f
#define cv_hal_sub8u32f             fastcv_hal_sub8u32f
#undef  cv_hal_transpose2d
#define cv_hal_transpose2d          fastcv_hal_transpose2d
#undef  cv_hal_meanStdDev
#define cv_hal_meanStdDev           fastcv_hal_meanStdDev
#undef  cv_hal_flip
#define cv_hal_flip                 fastcv_hal_flip
#undef  cv_hal_rotate90
#define cv_hal_rotate90             fastcv_hal_rotate
#undef  cv_hal_addWeighted8u
#define cv_hal_addWeighted8u        fastcv_hal_addWeighted8u
#undef  cv_hal_mul8u
#define cv_hal_mul8u                fastcv_hal_mul8u
#undef  cv_hal_mul16s
#define cv_hal_mul16s               fastcv_hal_mul16s
#undef  cv_hal_mul32f
#define cv_hal_mul32f               fastcv_hal_mul32f

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief look-up table transform of an array.
/// @param src_data Source image data
/// @param src_step Source image step
/// @param src_type Source image type
/// @param lut_data Pointer to lookup table
/// @param lut_channel_size Size of each channel in bytes
/// @param lut_channels Number of channels in lookup table
/// @param dst_data Destination data
/// @param dst_step Destination step
/// @param width Width of images
/// @param height Height of images
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_lut(
    const uchar*    src_data,
    size_t          src_step,
    size_t          src_type,
    const uchar*    lut_data,
    size_t          lut_channel_size,
    size_t          lut_channels,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Hamming distance between two vectors
/// @param a pointer to first vector data
/// @param b pointer to second vector data
/// @param n length of vectors
/// @param cellSize how many bits of the vectors will be added and treated as a single bit, can be 1 (standard Hamming distance), 2 or 4
/// @param result pointer to result output
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_normHammingDiff8u(const uchar* a, const uchar* b, int n, int cellSize, int* result);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_mul8u16u(
    const uchar    * src1_data,
    size_t           src1_step,
    const uchar    * src2_data,
    size_t           src2_step,
    ushort         * dst_data,
    size_t           dst_step,
    int              width,
    int              height,
    double           scale);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_sub8u32f(
    const uchar     *src1_data,
    size_t           src1_step,
    const uchar     *src2_data,
    size_t           src2_step,
    float           *dst_data,
    size_t           dst_step,
    int              width,
    int              height);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_transpose2d(
    const uchar*     src_data,
    size_t           src_step,
    uchar*           dst_data,
    size_t           dst_step,
    int              src_width,
    int              src_height,
    int              element_size);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_meanStdDev(
    const uchar     * src_data,
    size_t            src_step,
    int               width,
    int               height,
    int               src_type,
    double          * mean_val,
    double          * stddev_val,
    uchar           * mask,
    size_t            mask_step);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Flips a 2D array around vertical, horizontal, or both axes
/// @param src_type source and destination image type
/// @param src_data source image data
/// @param src_step source image step
/// @param src_width source and destination image width
/// @param src_height source and destination image height
/// @param dst_data destination image data
/// @param dst_step destination image step
/// @param flip_mode 0 flips around x-axis, 1 around y-axis, -1 both
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_flip(
    int             src_type,
    const uchar*    src_data,
    size_t          src_step,
    int             src_width,
    int             src_height,
    uchar*          dst_data,
    size_t          dst_step,
    int             flip_mode);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Rotates a 2D array in multiples of 90 degrees.
/// @param src_type source and destination image type
/// @param src_data source image data
/// @param src_step source image step
/// @param src_width source image width
///   @If angle has value [180] it is also destination image width
///   If angle has values [90, 270] it is also destination image height
/// @param src_height source and destination image height (destination image width for angles [90, 270])
///   If angle has value [180] it is also destination image height
///   If angle has values [90, 270] it is also destination image width
/// @param dst_data destination image data
/// @param dst_step destination image step
/// @param angle clockwise angle for rotation in degrees from set [90, 180, 270]
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_rotate(
    int             src_type,
    const uchar*    src_data,
    size_t          src_step,
    int             src_width,
    int             src_height,
    uchar*          dst_data,
    size_t          dst_step,
    int             angle);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief weighted sum of two arrays using formula: dst[i] = a * src1[i] + b * src2[i]
/// @param src1_data first source image data
/// @param src1_step first source image step
/// @param src2_data second source image data
/// @param src2_step second source image step
/// @param dst_data  destination image data
/// @param dst_step  destination image step
/// @param width     width of the images
/// @param height    height of the images
/// @param scalars   numbers a, b, and c
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_addWeighted8u(
    const uchar*    src1_data,
    size_t          src1_step,
    const uchar*    src2_data,
    size_t          src2_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    const double    scalars[3]);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_mul8u(
    const uchar     *src1_data,
    size_t          src1_step,
    const uchar     *src2_data,
    size_t          src2_step,
    uchar           *dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    double          scale);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_mul16s(
    const short     *src1_data,
    size_t          src1_step,
    const short     *src2_data,
    size_t          src2_step,
    short           *dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    double          scale);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_mul32f(
    const float    *src1_data,
    size_t          src1_step,
    const float    *src2_data,
    size_t          src2_step,
    float          *dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    double          scale);

#endif
