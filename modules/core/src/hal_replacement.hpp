/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_CORE_HAL_REPLACEMENT_HPP
#define OPENCV_CORE_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

#if defined(__clang__)  // clang or MSVC clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

//! @addtogroup core_hal_interface
//! @note Define your functions to override default implementations:
//! @code
//! #undef hal_add8u
//! #define hal_add8u my_add8u
//! @endcode
//! @{

/**
Add: _dst[i] = src1[i] + src2[i]_ @n
Sub: _dst[i] = src1[i] - src2[i]_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
*/
//! @addtogroup core_hal_interface_addsub Element-wise add and subtract
//! @{
inline int hal_ni_add8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

inline int hal_ni_sub8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

inline int hal_ni_sub8u32f(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub8s32f(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

/**
Add scalar: _dst[i] = src[i] + scalar

@param src_data source image data
@param src_step source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
@param scalar_data pointer to scalar value
@param nChannels number of channels per element
*/
inline int hal_ni_addScalar32f32f(const float*   src_data, size_t src_step, float*   dst_data, size_t dst_step, int width, int height, const float*   scalar_data, int nChannels) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addScalar16s16s(const int16_t* src_data, size_t src_step, int16_t* dst_data, size_t dst_step, int width, int height, const int16_t* scalar_data, int nChannels) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Minimum: _dst[i] = min(src1[i], src2[i])_ @n
Maximum: _dst[i] = max(src1[i], src2[i])_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
*/
//! @addtogroup core_hal_interface_minmax Element-wise minimum or maximum
//! @{
inline int hal_ni_max8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

inline int hal_ni_min8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Absolute difference: _dst[i] = | src1[i] - src2[i] |_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
*/
//! @addtogroup core_hal_interface_absdiff Element-wise absolute difference
//! @{
inline int hal_ni_absdiff8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

/*
Absolute difference with scalar: _dst[i] = | src[i] - scalar |_

@param src_data source image data
@param src_step source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
@param scalar_data pointer to scalar value
@param nChannels number of channels per element
*/
inline int hal_ni_absDiffScalar32f32f(const float* src_data, size_t src_step, float*    dst_data, size_t dst_step, int width, int height, const float* scalar_data, int nChannels) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absDiffScalar32s32u(const int*   src_data, size_t src_step, uint32_t* dst_data, size_t dst_step, int width, int height, const int*   scalar_data, int nChannels) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absDiffScalar8u8u  (const uchar* src_data, size_t src_step, uchar*    dst_data, size_t dst_step, int width, int height, const uchar* scalar_data, int nChannels) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @}

/**
Bitwise AND: _dst[i] = src1[i] & src2[i]_ @n
Bitwise OR: _dst[i] = src1[i] | src2[i]_ @n
Bitwise XOR: _dst[i] = src1[i] ^ src2[i]_ @n
Bitwise NOT: _dst[i] = !src[i]_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
 */
//! @addtogroup core_hal_interface_logical Bitwise logical operations
//! @{
inline int hal_ni_and8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_or8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_xor8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_not8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_add8u hal_ni_add8u
#define cv_hal_add8s hal_ni_add8s
#define cv_hal_add16u hal_ni_add16u
#define cv_hal_add16s hal_ni_add16s
#define cv_hal_add32s hal_ni_add32s
#define cv_hal_add32f hal_ni_add32f
#define cv_hal_add64f hal_ni_add64f
#define cv_hal_sub8u hal_ni_sub8u
#define cv_hal_sub8s hal_ni_sub8s
#define cv_hal_sub16u hal_ni_sub16u
#define cv_hal_sub16s hal_ni_sub16s
#define cv_hal_sub32s hal_ni_sub32s
#define cv_hal_sub32f hal_ni_sub32f
#define cv_hal_sub64f hal_ni_sub64f
#define cv_hal_sub8u32f hal_ni_sub8u32f
#define cv_hal_sub8s32f hal_ni_sub8s32f
#define cv_hal_addScalar32f32f hal_ni_addScalar32f32f
#define cv_hal_addScalar16s16s hal_ni_addScalar16s16s
#define cv_hal_max8u hal_ni_max8u
#define cv_hal_max8s hal_ni_max8s
#define cv_hal_max16u hal_ni_max16u
#define cv_hal_max16s hal_ni_max16s
#define cv_hal_max32s hal_ni_max32s
#define cv_hal_max32f hal_ni_max32f
#define cv_hal_max64f hal_ni_max64f
#define cv_hal_min8u hal_ni_min8u
#define cv_hal_min8s hal_ni_min8s
#define cv_hal_min16u hal_ni_min16u
#define cv_hal_min16s hal_ni_min16s
#define cv_hal_min32s hal_ni_min32s
#define cv_hal_min32f hal_ni_min32f
#define cv_hal_min64f hal_ni_min64f
#define cv_hal_absdiff8u hal_ni_absdiff8u
#define cv_hal_absdiff8s hal_ni_absdiff8s
#define cv_hal_absdiff16u hal_ni_absdiff16u
#define cv_hal_absdiff16s hal_ni_absdiff16s
#define cv_hal_absdiff32s hal_ni_absdiff32s
#define cv_hal_absdiff32f hal_ni_absdiff32f
#define cv_hal_absdiff64f hal_ni_absdiff64f
#define cv_hal_absDiffScalar32f32f hal_ni_absDiffScalar32f32f
#define cv_hal_absDiffScalar32s32u hal_ni_absDiffScalar32s32u
#define cv_hal_absDiffScalar8u8u   hal_ni_absDiffScalar8u8u
#define cv_hal_and8u hal_ni_and8u
#define cv_hal_or8u hal_ni_or8u
#define cv_hal_xor8u hal_ni_xor8u
#define cv_hal_not8u hal_ni_not8u
//! @endcond

/**
Lookup table replacement
Table consists of 256 elements of a size from 1 to 8 bytes having 1 channel or src_channels
For 8s input type 128 is added to LUT index
Destination should have the same element type and number of channels as lookup table elements
@param src_data Source image data
@param src_step Source image step
@param src_type Sorce image type
@param lut_data Pointer to lookup table
@param lut_channel_size Size of each channel in bytes
@param lut_channels Number of channels in lookup table
@param dst_data Destination data
@param dst_step Destination step
@param width Width of images
@param height Height of images
@sa LUT
*/
//! @addtogroup core_hal_interface_lut Lookup table
//! @{
inline int hal_ni_lut(const uchar *src_data, size_t src_step, size_t src_type, const uchar* lut_data, size_t lut_channel_size, size_t lut_channels, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_lut hal_ni_lut
//! @endcond

/**
Hamming norm of a vector
@param a pointer to vector data
@param n length of a vector
@param cellSize how many bits of the vector will be added and treated as a single bit, can be 1 (standard Hamming distance), 2 or 4
@param result pointer to result output
*/
//! @addtogroup core_hal_interface_hamming Hamming distance
//! @{
inline int hal_ni_normHamming8u(const uchar* a, int n, int cellSize, int* result) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Hamming distance between two vectors
@param a pointer to first vector data
@param b pointer to second vector data
@param n length of vectors
@param cellSize how many bits of the vectors will be added and treated as a single bit, can be 1 (standard Hamming distance), 2 or 4
@param result pointer to result output
*/
//! @addtogroup core_hal_interface_hamming Hamming distance
//! @{
inline int hal_ni_normHammingDiff8u(const uchar* a, const uchar* b, int n, int cellSize, int* result) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
@brief Generic norm of an array.
@param src Source image
@param src_step Source image
@param mask Specified array region.
@param mask_step Mask array step.
@param width Source image dimensions
@param height Source image dimensions
@param type Element type of source image
@param norm_type Type of the norm
@param result Pointer to result output
*/
//! @addtogroup core_hal_interface_norm Absolute norm
//! @{
inline int hal_ni_norm(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step, int width,
                       int height, int type, int norm_type, double* result) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
@brief Generic norm between two arrays.
@param src1 First source image
@param src1_step First source image
@param src2 Second source image
@param src2_step Second source image
@param mask Specified array region.
@param mask_step Mask array step.
@param width Source image dimensions
@param height Source image dimensions
@param type Element type of source image
@param norm_type Type of the norm
@param result Pointer to result output
*/
//! @addtogroup core_hal_interface_norm Absolute norm
//! @{
inline int hal_ni_normDiff(const uchar* src1, size_t src1_step, const uchar* src2, size_t src2_step, const uchar* mask,
                           size_t mask_step, int width, int height, int type, int norm_type, double* result) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
@brief Convert array to another with specified type.
@param src Source image
@param src_step Source image
@param dst Destination image
@param dst_step Destination image
@param width Source image dimensions
@param height Source image dimensions
@param sdepth Depth of source image
@param ddepth Depth of destination image
@param alpha Scale value
@param beta Shift value
*/
//! @addtogroup core_hal_interface_convert Array convert
//! @{
inline int hal_ni_convertScale(const uchar* src, size_t src_step, uchar* dst, size_t dst_step, int width, int height,
                               int sdepth, int ddepth, double alpha, double beta) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_normHamming8u hal_ni_normHamming8u
#define cv_hal_normHammingDiff8u hal_ni_normHammingDiff8u
#define cv_hal_norm hal_ni_norm
#define cv_hal_normDiff hal_ni_normDiff
#define cv_hal_convertScale hal_ni_convertScale
//! @endcond

/**
Compare: _dst[i] = src1[i] op src2[i]_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
@param operation one of (CV_HAL_CMP_EQ, CV_HAL_CMP_GT, ...)
*/
//! @addtogroup core_hal_interface_compare Element-wise compare
//! @{
inline int hal_ni_cmp8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_cmp8u hal_ni_cmp8u
#define cv_hal_cmp8s hal_ni_cmp8s
#define cv_hal_cmp16u hal_ni_cmp16u
#define cv_hal_cmp16s hal_ni_cmp16s
#define cv_hal_cmp32s hal_ni_cmp32s
#define cv_hal_cmp32f hal_ni_cmp32f
#define cv_hal_cmp64f hal_ni_cmp64f
//! @endcond

/**
Multiply: _dst[i] = scale * src1[i] * src2[i]_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
@param scale additional multiplier
*/
//! @addtogroup core_hal_interface_multiply Element-wise multiply
//! @{
inline int hal_ni_mul8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul8u16u(const uchar* src1_data, size_t src1_step, const uchar* src2_data, size_t src2_step, ushort* dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul8s16s(const schar* src1_data, size_t src1_step, const schar* src2_data, size_t src2_step, short* dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Divide: _dst[i] = scale * src1[i] / src2[i]_
@param src1_data first source image data and step
@param src1_step first source image data and step
@param src2_data second source image data and step
@param src2_step second source image data and step
@param dst_data destination image data and step
@param dst_step destination image data and step
@param width dimensions of the images
@param height dimensions of the images
@param scale additional multiplier
*/
//! @addtogroup core_hal_interface_divide Element-wise divide
//! @{
inline int hal_ni_div8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Computes reciprocial: _dst[i] = scale / src[i]_
@param src_data source image data
@param src_step source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
@param scale additional multiplier
 */
//! @addtogroup core_hal_interface_reciprocial Element-wise reciprocial
//! @{
inline int hal_ni_recip8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip8s(const schar *src_data, size_t src_step, schar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip16u(const ushort *src_data, size_t src_step, ushort *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip16s(const short *src_data, size_t src_step, short *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip32s(const int *src_data, size_t src_step, int *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip32f(const float *src_data, size_t src_step, float *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip64f(const double *src_data, size_t src_step, double *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_mul8u hal_ni_mul8u
#define cv_hal_mul8s hal_ni_mul8s
#define cv_hal_mul16u hal_ni_mul16u
#define cv_hal_mul16s hal_ni_mul16s
#define cv_hal_mul32s hal_ni_mul32s
#define cv_hal_mul32f hal_ni_mul32f
#define cv_hal_mul64f hal_ni_mul64f
#define cv_hal_mul8u16u hal_ni_mul8u16u
#define cv_hal_mul8s16s hal_ni_mul8s16s
#define cv_hal_div8u hal_ni_div8u
#define cv_hal_div8s hal_ni_div8s
#define cv_hal_div16u hal_ni_div16u
#define cv_hal_div16s hal_ni_div16s
#define cv_hal_div32s hal_ni_div32s
#define cv_hal_div32f hal_ni_div32f
#define cv_hal_div64f hal_ni_div64f
#define cv_hal_recip8u hal_ni_recip8u
#define cv_hal_recip8s hal_ni_recip8s
#define cv_hal_recip16u hal_ni_recip16u
#define cv_hal_recip16s hal_ni_recip16s
#define cv_hal_recip32s hal_ni_recip32s
#define cv_hal_recip32f hal_ni_recip32f
#define cv_hal_recip64f hal_ni_recip64f
//! @endcond

/**
Computes weighted sum of two arrays using formula: _dst[i] = a * src1[i] + b * src2[i] + c_
@param src1_data first source image data
@param src1_step first source image step
@param src2_data second source image data
@param src2_step second source image step
@param dst_data destination image data
@param dst_step destination image step
@param width width of the images
@param height height of the images
@param scalars numbers _a_, _b_, and _c_
 */
//! @addtogroup core_hal_interface_addWeighted Element-wise weighted sum
//! @{
inline int hal_ni_addWeighted8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_addWeighted8u hal_ni_addWeighted8u
#define cv_hal_addWeighted8s hal_ni_addWeighted8s
#define cv_hal_addWeighted16u hal_ni_addWeighted16u
#define cv_hal_addWeighted16s hal_ni_addWeighted16s
#define cv_hal_addWeighted32s hal_ni_addWeighted32s
#define cv_hal_addWeighted32f hal_ni_addWeighted32f
#define cv_hal_addWeighted64f hal_ni_addWeighted64f
//! @endcond

/**
@param src_data array of interleaved values (__len__ x __cn__ items) [ B, G, R, B, G, R, ...]
@param dst_data array of pointers to destination arrays (__cn__ items x __len__ items) [ [B, B, ...], [G, G, ...], [R, R, ...] ]
@param len number of elements
@param cn number of channels
 */
//! @addtogroup core_hal_interface_split Channel split
//! @{
inline int hal_ni_split8u(const uchar *src_data, uchar **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split16u(const ushort *src_data, ushort **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split32s(const int *src_data, int **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split64s(const int64 *src_data, int64 **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_split8u hal_ni_split8u
#define cv_hal_split16u hal_ni_split16u
#define cv_hal_split32s hal_ni_split32s
#define cv_hal_split64s hal_ni_split64s
//! @endcond

/**
@param src_data array of pointers to source arrays (__cn__ items x __len__ items) [ [B, B, ...], [G, G, ...], [R, R, ...] ]
@param dst_data destination array of interleaved values (__len__ x __cn__ items) [ B, G, R, B, G, R, ...]
@param len number of elements
@param cn number of channels
 */
//! @addtogroup core_hal_interface_merge Channel merge
//! @{
inline int hal_ni_merge8u(const uchar **src_data, uchar *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge16u(const ushort **src_data, ushort *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge32s(const int **src_data, int *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge64s(const int64 **src_data, int64 *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_merge8u hal_ni_merge8u
#define cv_hal_merge16u hal_ni_merge16u
#define cv_hal_merge32s hal_ni_merge32s
#define cv_hal_merge64s hal_ni_merge64s
//! @endcond

/**
@param x source X arrays
@param y source Y arrays
@param mag destination magnitude array
@param angle destination angle array
@param len length of arrays
@param angleInDegrees if set to true return angles in degrees, otherwise in radians
*/
//! @addtogroup core_hal_interface_fastAtan Atan calculation
//! @{
inline int hal_ni_cartToPolar32f(const float* x, const float* y, float* mag, float* angle, int len, bool angleInDegrees) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cartToPolar64f(const double* x, const double* y, double* mag, double* angle, int len, bool angleInDegrees) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_cartToPolar32f hal_ni_cartToPolar32f
#define cv_hal_cartToPolar64f hal_ni_cartToPolar64f
//! @endcond

/**
@param y source Y arrays
@param x source X arrays
@param dst destination array
@param len length of arrays
@param angleInDegrees if set to true return angles in degrees, otherwise in radians
 */
//! @addtogroup core_hal_interface_fastAtan Atan calculation
//! @{
inline int hal_ni_fastAtan32f(const float* y, const float* x, float* dst, int len, bool angleInDegrees) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_fastAtan64f(const double* y, const double* x, double* dst, int len, bool angleInDegrees) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_fastAtan32f hal_ni_fastAtan32f
#define cv_hal_fastAtan64f hal_ni_fastAtan64f
//! @endcond


/**
@param x source X array
@param y source Y array
@param dst destination array
@param len length of arrays
 */
//! @addtogroup core_hal_interface_magnitude Magnitude calculation
//! @{
inline int hal_ni_magnitude32f(const float *x, const float *y, float *dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_magnitude64f(const double *x, const double  *y, double *dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_magnitude32f hal_ni_magnitude32f
#define cv_hal_magnitude64f hal_ni_magnitude64f
//! @endcond

/**
@param mag source magnitude arrays
@param mag source angle arrays
@param x destination X array
@param y destination Y array
@param len length of arrays
@param angleInDegrees if set to true interpret angles from degrees, otherwise from radians
*/
//! @addtogroup core_hal_interface_fastAtan Atan calculation
//! @{
inline int hal_ni_polarToCart32f(const float* mag, const float* angle, float* x, float* y, int len, bool angleInDegrees) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_polarToCart64f(const double* mag, const double* angle, double* x, double* y, int len, bool angleInDegrees) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_polarToCart32f hal_ni_polarToCart32f
#define cv_hal_polarToCart64f hal_ni_polarToCart64f
//! @endcond

/**
@param src source array
@param dst destination array
@param len length of arrays
 */
//! @addtogroup core_hal_interface_invSqrt Inverse square root calculation
//! @{
inline int hal_ni_invSqrt32f(const float* src, float* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_invSqrt64f(const double* src, double* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_invSqrt32f hal_ni_invSqrt32f
#define cv_hal_invSqrt64f hal_ni_invSqrt64f
//! @endcond


/**
@param src source array
@param dst destination array
@param len length of arrays
 */
//! @addtogroup core_hal_interface_sqrt Square root calculation
//! @{
inline int hal_ni_sqrt32f(const float* src, float* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sqrt64f(const double* src, double* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_sqrt32f hal_ni_sqrt32f
#define cv_hal_sqrt64f hal_ni_sqrt64f
//! @endcond


/**
@param src source array
@param dst destination array
@param len length of arrays
 */
//! @addtogroup core_hal_interface_log Natural logarithm calculation
//! @{
inline int hal_ni_log32f(const float* src, float* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_log64f(const double* src, double* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_log32f hal_ni_log32f
#define cv_hal_log64f hal_ni_log64f
//! @endcond


/**
@param src source array
@param dst destination array
@param len length of arrays
 */
//! @addtogroup core_hal_interface_exp Exponent calculation
//! @{
inline int hal_ni_exp32f(const float* src, float* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_exp64f(const double* src, double* dst, int len) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_exp32f hal_ni_exp32f
#define cv_hal_exp64f hal_ni_exp64f
//! @endcond


/**
@brief Dummy structure storing DFT/DCT context

Users can convert this pointer to any type they want. Initialisation and destruction should be made in Init and Free function implementations correspondingly.
Example:
@code{.cpp}
int my_hal_dftInit2D(cvhalDFT **context, ...) {
    *context = static_cast<cvhalDFT*>(new MyFilterData());
    //... init
}

int my_hal_dftFree2D(cvhalDFT *context) {
    MyFilterData *c = static_cast<MyFilterData*>(context);
    delete c;
}
@endcode
 */
struct cvhalDFT {};

/**
@param context double pointer to context storing all necessary data
@param len transformed array length
@param count estimated transformation count
@param depth array type (CV_32F or CV_64F)
@param flags algorithm options (combination of CV_HAL_DFT_INVERSE, CV_HAL_DFT_SCALE, ...)
@param needBuffer pointer to boolean variable, if valid pointer provided, then variable value should be set to true to signal that additional memory buffer is needed for operations
 */
inline int hal_ni_dftInit1D(cvhalDFT **context, int len, int count, int depth, int flags, bool *needBuffer) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
@param src source data
@param dst destination data
 */
inline int hal_ni_dft1D(cvhalDFT *context, const uchar *src, uchar *dst) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
 */
inline int hal_ni_dftFree1D(cvhalDFT *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dftInit1D hal_ni_dftInit1D
#define cv_hal_dft1D hal_ni_dft1D
#define cv_hal_dftFree1D hal_ni_dftFree1D
//! @endcond

/**
@param context double pointer to context storing all necessary data
@param width image width
@param height image height
@param depth image type (CV_32F or CV_64F)
@param src_channels number of channels in input image
@param dst_channels number of channels in output image
@param flags algorithm options (combination of CV_HAL_DFT_INVERSE, ...)
@param nonzero_rows number of nonzero rows in image, can be used for optimization
 */
inline int hal_ni_dftInit2D(cvhalDFT **context, int width, int height, int depth, int src_channels, int dst_channels, int flags, int nonzero_rows) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
@param src_data source image data
@param src_step source image step
@param dst_data destination image data
@param dst_step destination image step
 */
inline int hal_ni_dft2D(cvhalDFT *context, const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
 */
inline int hal_ni_dftFree2D(cvhalDFT *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dftInit2D hal_ni_dftInit2D
#define cv_hal_dft2D hal_ni_dft2D
#define cv_hal_dftFree2D hal_ni_dftFree2D
//! @endcond

/**
@param context double pointer to context storing all necessary data
@param width image width
@param height image height
@param depth image type (CV_32F or CV_64F)
@param flags algorithm options (combination of CV_HAL_DFT_INVERSE, ...)
 */
inline int hal_ni_dctInit2D(cvhalDFT **context, int width, int height, int depth, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
@param src_data source image data
@param src_step source image step
@param dst_data destination image data
@param dst_step destination image step
 */
inline int hal_ni_dct2D(cvhalDFT *context, const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
 */
inline int hal_ni_dctFree2D(cvhalDFT *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dctInit2D hal_ni_dctInit2D
#define cv_hal_dct2D hal_ni_dct2D
#define cv_hal_dctFree2D hal_ni_dctFree2D
//! @endcond


/**
Performs \f$LU\f$ decomposition of square matrix \f$A=P*L*U\f$ (where \f$P\f$ is permutation matrix) and solves matrix equation \f$A*X=B\f$.
Function returns the \f$sign\f$ of permutation \f$P\f$ via parameter info.
@param src1 pointer to input matrix \f$A\f$ stored in row major order. After finish of work src1 contains at least \f$U\f$ part of \f$LU\f$
decomposition which is appropriate for determainant calculation: \f$det(A)=sign*\prod_{j=1}^{M}a_{jj}\f$.
@param src1_step number of bytes between two consequent rows of matrix \f$A\f$.
@param m size of square matrix \f$A\f$.
@param src2 pointer to \f$M\times N\f$ matrix \f$B\f$ which is the right-hand side of system \f$A*X=B\f$. \f$B\f$ stored in row major order.
If src2 is null pointer only \f$LU\f$ decomposition will be performed. After finish of work src2 contains solution \f$X\f$ of system \f$A*X=B\f$.
@param src2_step number of bytes between two consequent rows of matrix \f$B\f$.
@param n number of right-hand vectors in \f$M\times N\f$ matrix \f$B\f$.
@param info indicates success of decomposition. If *info is equals to zero decomposition failed, othervise *info is equals to \f$sign\f$.
 */
//! @addtogroup core_hal_interface_decomp_lu LU matrix decomposition
//! @{
inline int hal_ni_LU32f(float* src1, size_t src1_step, int m, float* src2, size_t src2_step, int n, int* info) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_LU64f(double* src1, size_t src1_step, int m, double* src2, size_t src2_step, int n, int* info) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Performs Cholesky decomposition of matrix \f$A = L*L^T\f$ and solves matrix equation \f$A*X=B\f$.
@param src1 pointer to input matrix \f$A\f$ stored in row major order. After finish of work src1 contains lower triangular matrix \f$L\f$.
@param src1_step number of bytes between two consequent rows of matrix \f$A\f$.
@param m size of square matrix \f$A\f$.
@param src2 pointer to \f$M\times N\f$ matrix \f$B\f$ which is the right-hand side of system \f$A*X=B\f$. B stored in row major order.
If src2 is null pointer only Cholesky decomposition will be performed. After finish of work src2 contains solution \f$X\f$ of system \f$A*X=B\f$.
@param src2_step number of bytes between two consequent rows of matrix \f$B\f$.
@param n number of right-hand vectors in \f$M\times N\f$ matrix \f$B\f$.
@param info indicates success of decomposition. If *info is false decomposition failed.
 */

//! @addtogroup core_hal_interface_decomp_cholesky Cholesky matrix decomposition
//! @{
inline int hal_ni_Cholesky32f(float* src1, size_t src1_step, int m, float* src2, size_t src2_step, int n, bool* info) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_Cholesky64f(double* src1, size_t src1_step, int m, double* src2, size_t src2_step, int n, bool* info) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Performs singular value decomposition of \f$M\times N\f$(\f$M>N\f$) matrix \f$A = U*\Sigma*V^T\f$.
@param src pointer to input \f$M\times N\f$ matrix \f$A\f$ stored in column major order.
After finish of work src will be filled with rows of \f$U\f$ or not modified (depends of flag CV_HAL_SVD_MODIFY_A).
@param src_step number of bytes between two consequent columns of matrix \f$A\f$.
@param w pointer to array for singular values of matrix \f$A\f$ (i. e. first \f$N\f$ diagonal elements of matrix \f$\Sigma\f$).
@param u pointer to output \f$M\times N\f$ or \f$M\times M\f$ matrix \f$U\f$ (size depends of flags). Pointer must be valid if flag CV_HAL_SVD_MODIFY_A not used.
@param u_step number of bytes between two consequent rows of matrix \f$U\f$.
@param vt pointer to array for \f$N\times N\f$ matrix \f$V^T\f$.
@param vt_step number of bytes between two consequent rows of matrix \f$V^T\f$.
@param m number fo rows in matrix \f$A\f$.
@param n number of columns in matrix \f$A\f$.
@param flags algorithm options (combination of CV_HAL_SVD_FULL_UV, ...).
 */
//! @addtogroup core_hal_interface_decomp_svd Singular value matrix decomposition
//! @{
inline int hal_ni_SVD32f(float* src, size_t src_step, float* w, float* u, size_t u_step, float* vt, size_t vt_step, int m, int n, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_SVD64f(double* src, size_t src_step, double* w, double* u, size_t u_step, double* vt, size_t vt_step, int m, int n, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Performs QR decomposition of \f$M\times N\f$(\f$M>N\f$) matrix \f$A = Q*R\f$ and solves matrix equation \f$A*X=B\f$.
@param src1 pointer to input matrix \f$A\f$ stored in row major order. After finish of work src1 contains upper triangular \f$N\times N\f$ matrix \f$R\f$.
Lower triangle of src1 will be filled with vectors of elementary reflectors. See @cite VandLec and Lapack's DGEQRF documentation for details.
@param src1_step number of bytes between two consequent rows of matrix \f$A\f$.
@param m number fo rows in matrix \f$A\f$.
@param n number of columns in matrix \f$A\f$.
@param k number of right-hand vectors in \f$M\times K\f$ matrix \f$B\f$.
@param src2 pointer to \f$M\times K\f$ matrix \f$B\f$ which is the right-hand side of system \f$A*X=B\f$. \f$B\f$ stored in row major order.
If src2 is null pointer only QR decomposition will be performed. Otherwise system will be solved and src1 will be used as temporary buffer, so
after finish of work src2 contains solution \f$X\f$ of system \f$A*X=B\f$.
@param src2_step number of bytes between two consequent rows of matrix \f$B\f$.
@param dst pointer to continiuos \f$N\times 1\f$ array for scalar factors of elementary reflectors. See @cite VandLec for details.
@param info indicates success of decomposition. If *info is zero decomposition failed.
*/
//! @addtogroup core_hal_interface_decomp_qr QR matrix decomposition
//! @{
inline int hal_ni_QR32f(float* src1, size_t src1_step, int m, int n, int k, float* src2, size_t src2_step, float* dst, int* info) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_QR64f(double* src1, size_t src1_step, int m, int n, int k, double* src2, size_t src2_step, double* dst, int* info) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}



//! @cond IGNORED
#define cv_hal_LU32f hal_ni_LU32f
#define cv_hal_LU64f hal_ni_LU64f
#define cv_hal_Cholesky32f hal_ni_Cholesky32f
#define cv_hal_Cholesky64f hal_ni_Cholesky64f
#define cv_hal_SVD32f hal_ni_SVD32f
#define cv_hal_SVD64f hal_ni_SVD64f
#define cv_hal_QR32f hal_ni_QR32f
#define cv_hal_QR64f hal_ni_QR64f
//! @endcond


/**
The function performs generalized matrix multiplication similar to the gemm functions in BLAS level 3:
\f$D = \alpha*AB+\beta*C\f$

@param src1 pointer to input \f$M\times N\f$ matrix \f$A\f$ or \f$A^T\f$ stored in row major order.
@param src1_step number of bytes between two consequent rows of matrix \f$A\f$ or \f$A^T\f$.
@param src2 pointer to input \f$N\times K\f$ matrix \f$B\f$ or \f$B^T\f$ stored in row major order.
@param src2_step number of bytes between two consequent rows of matrix \f$B\f$ or \f$B^T\f$.
@param alpha \f$\alpha\f$ multiplier before \f$AB\f$
@param src3 pointer to input \f$M\times K\f$ matrix \f$C\f$ or \f$C^T\f$ stored in row major order.
@param src3_step number of bytes between two consequent rows of matrix \f$C\f$ or \f$C^T\f$.
@param beta \f$\beta\f$ multiplier before \f$C\f$
@param dst pointer to input \f$M\times K\f$ matrix \f$D\f$ stored in row major order.
@param dst_step number of bytes between two consequent rows of matrix \f$D\f$.
@param m number of rows in matrix \f$A\f$ or \f$A^T\f$, equals to number of rows in matrix \f$D\f$
@param n number of columns in matrix \f$A\f$ or \f$A^T\f$
@param k number of columns in matrix \f$B\f$ or \f$B^T\f$, equals to number of columns in matrix \f$D\f$
@param flags algorithm options (combination of CV_HAL_GEMM_1_T, ...).
 */

//! @addtogroup core_hal_interface_matrix_multiplication Matrix multiplication
//! @{
inline int hal_ni_gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                          float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
                          int m, int n, int k, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_gemm64f(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
                          double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
                          int m, int n, int k, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_gemm32fc(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                          float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
                          int m, int n, int k, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_gemm64fc(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
                          double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
                          int m, int n, int k, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_gemm32f hal_ni_gemm32f
#define cv_hal_gemm64f hal_ni_gemm64f
#define cv_hal_gemm32fc hal_ni_gemm32fc
#define cv_hal_gemm64fc hal_ni_gemm64fc
//! @endcond

/**
   @brief Finds the global minimum and maximum in an array.
   @param src_data Source image
   @param src_step Source image
   @param width Source image dimensions
   @param height Source image dimensions
   @param depth Depth of source image
   @param minVal Pointer to the returned global minimum and maximum in an array.
   @param maxVal Pointer to the returned global minimum and maximum in an array.
   @param minIdx Pointer to the returned minimum and maximum location.
   @param maxIdx Pointer to the returned minimum and maximum location.
   @param mask Specified array region.
*/
inline int hal_ni_minMaxIdx(const uchar* src_data, size_t src_step, int width, int height, int depth, double* minVal, double* maxVal,
                            int* minIdx, int* maxIdx, uchar* mask) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_minMaxIdx hal_ni_minMaxIdx
//! @endcond

/**
   @brief calculates the mean and the standard deviation of array elements independently for each channel
   @param src_data Source image
   @param src_step Source image
   @param width Source image dimensions
   @param height Source image dimensions
   @param src_type Type of source image
   @param mean_val Array of per-channel mean values. May be nullptr, if mean value is not required.
   @param stddev_val Array of per-channel standard deviation values. May be nullptr, if stddev value is not required.
   @param mask Specified array region.
   @param mask_step Mask array step.
   @sa meanStdDev
*/
inline int hal_ni_meanStdDev(const uchar* src_data, size_t src_step, int width, int height,
                             int src_type, double* mean_val, double* stddev_val, uchar* mask, size_t mask_step)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_meanStdDev hal_ni_meanStdDev
//! @endcond

/**
 * @brief calculates dot product of two vectors (represented as 2d images)
 *
 * @param a_data Pointer to 1st 2nd image data
 * @param a_step Stride of 1st 2nd image
 * @param b_data Pointer to 1st 2nd image data
 * @param b_step Stride of 1st 2nd image
 * @param width Width of both images
 * @param height Height of both images
 * @param type Data type of both images, for example CV_8U or CV_32F
 * @param dot_val Pointer to resulting dot product value
 * @return int
 */
inline int hal_ni_dotProduct(const uchar* a_data, size_t a_step, const uchar* b_data, size_t b_step, int width, int height,
                             int type, double *dot_val)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dotProduct hal_ni_dotProduct
//! @endcond

/**
   @brief hal_flip
   @param src_type source and destination image type
   @param src_data source image data
   @param src_step source image step
   @param src_width source and destination image width
   @param src_height source and destination image height
   @param dst_data destination image data
   @param dst_step destination image step
   @param flip_mode 0 flips around x-axis, positive around y-axis, negative both
 */
inline int hal_ni_flip(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
                       uchar* dst_data, size_t dst_step, int flip_mode) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_flip hal_ni_flip
//! @endcond

/**
   @brief rotate90
   @param src_type source and destination image type
   @param src_data source image data
   @param src_step source image step
   @param src_width source image width
   If angle has value [180] it is also destination image width
   If angle has values [90, 270] it is also destination image height
   @param src_height source and destination image height (destination image width for angles [90, 270])
   If angle has value [180] it is also destination image height
   If angle has values [90, 270] it is also destination image width
   @param dst_data destination image data
   @param dst_step destination image step
   @param angle clockwise angle for rotation in degrees from set [90, 180, 270]
 */
inline int hal_ni_rotate90(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
                           uchar* dst_data, size_t dst_step, int angle) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_rotate90 hal_ni_rotate90
//! @endcond

/**
   @brief Transpose2d
   @param src_data,src_step Source image
   @param dst_data,dst_step Destination image
   @param src_width,src_height Source image dimensions
   @param element_size Size of an element in bytes
*/
inline int hal_ni_transpose2d(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width,
                              int src_height, int element_size) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_transpose2d hal_ni_transpose2d
//! @endcond

//! @}


#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "hal_internal.hpp"
#include "custom_hal.hpp"

//! @cond IGNORED
#define CALL_HAL_RET(name, fun, retval, ...) \
{ \
    int res = __CV_EXPAND(fun(__VA_ARGS__, &retval)); \
    if (res == CV_HAL_ERROR_OK) \
        return retval; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res)); \
}


#define CALL_HAL(name, fun, ...) \
{ \
    int res = __CV_EXPAND(fun(__VA_ARGS__)); \
    if (res == CV_HAL_ERROR_OK) \
        return; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res)); \
}
//! @endcond

#endif
