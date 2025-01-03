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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_DNN_DNN_SHAPE_UTILS_HPP
#define OPENCV_DNN_DNN_SHAPE_UTILS_HPP

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/cvdef.h>  // CV_MAX_DIM
#include <iostream>
#include <ostream>
#include <sstream>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

//Slicing

struct _Range : public cv::Range
{
    _Range(const Range &r) : cv::Range(r) {}
    _Range(int start_, int size_ = 1) : cv::Range(start_, start_ + size_) {}
};

static inline Mat slice(const Mat &m, const _Range &r0)
{
    Range ranges[CV_MAX_DIM];
    for (int i = 1; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    return m(&ranges[0]);
}

static inline Mat slice(const Mat &m, const _Range &r0, const _Range &r1)
{
    CV_Assert(m.dims >= 2);
    Range ranges[CV_MAX_DIM];
    for (int i = 2; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    ranges[1] = r1;
    return m(&ranges[0]);
}

static inline Mat slice(const Mat &m, const _Range &r0, const _Range &r1, const _Range &r2)
{
    CV_Assert(m.dims >= 3);
    Range ranges[CV_MAX_DIM];
    for (int i = 3; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    ranges[1] = r1;
    ranges[2] = r2;
    return m(&ranges[0]);
}

static inline Mat slice(const Mat &m, const _Range &r0, const _Range &r1, const _Range &r2, const _Range &r3)
{
    CV_Assert(m.dims >= 4);
    Range ranges[CV_MAX_DIM];
    for (int i = 4; i < m.dims; i++)
        ranges[i] = Range::all();
    ranges[0] = r0;
    ranges[1] = r1;
    ranges[2] = r2;
    ranges[3] = r3;
    return m(&ranges[0]);
}

static inline UMat getPlane(UMat& m, int n, int ch) {
    CV_Assert(m.dims > 2);
    int offset = (n * m.step.p[0] + ch * m.step.p[1]) / m.elemSize();
    int total_length = 1;
    int plane_length = 1;
    for(int i = 0; i < m.dims; ++i) {
        total_length *= m.size[i];
    }

    for(int i = 2; i < m.dims; ++i) {
        plane_length *= m.size[i];
    }

    const int newShape[1] { total_length };
    UMat reshaped = m.reshape(1, 1, newShape);
    UMat roi = reshaped(Rect(0, offset, 1, plane_length));
    return roi.reshape(m.channels(), m.size[2]);
}

static inline Mat getPlane(const Mat &m, int n, int cn)
{
    CV_Assert(m.dims > 2);
    int sz[CV_MAX_DIM];
    for(int i = 2; i < m.dims; i++)
    {
        sz[i-2] = m.size.p[i];
    }
    return Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));
}

static inline MatShape shape(const int* dims, const int n)
{
    MatShape shape;
    shape.assign(dims, dims + n);
    return shape;
}

static inline MatShape shape(const Mat& mat)
{
    return shape(mat.size.p, mat.dims);
}

static inline MatShape shape(const MatSize& sz)
{
    return shape(sz.p, sz.dims());
}

static inline MatShape shape(const UMat& mat)
{
    return shape(mat.size.p, mat.dims);
}

#if 0  // issues with MatExpr wrapped into InputArray
static inline
MatShape shape(InputArray input)
{
    int sz[CV_MAX_DIM];
    int ndims = input.sizend(sz);
    return shape(sz, ndims);
}
#endif

namespace {inline bool is_neg(int i) { return i < 0; }}

static inline MatShape shape(int a0, int a1=-1, int a2=-1, int a3=-1)
{
    int dims[] = {a0, a1, a2, a3};
    MatShape s = shape(dims, 4);
    s.erase(std::remove_if(s.begin(), s.end(), is_neg), s.end());
    return s;
}

static inline int total(const MatShape& shape, int start = -1, int end = -1)
{
    if (shape.empty())
        return 0;

    int dims = (int)shape.size();

    if (start == -1) start = 0;
    if (end == -1) end = dims;

    CV_CheckLE(0, start, "");
    CV_CheckLE(start, end, "");
    CV_CheckLE(end, dims, "");

    int elems = 1;
    for (int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

// TODO: rename to countDimsElements()
static inline int total(const Mat& mat, int start = -1, int end = -1)
{
    if (mat.empty())
        return 0;

    int dims = mat.dims;

    if (start == -1) start = 0;
    if (end == -1) end = dims;

    CV_CheckLE(0, start, "");
    CV_CheckLE(start, end, "");
    CV_CheckLE(end, dims, "");

    int elems = 1;
    for (int i = start; i < end; i++)
    {
        elems *= mat.size[i];
    }
    return elems;
}

static inline MatShape concat(const MatShape& a, const MatShape& b)
{
    MatShape c = a;
    c.insert(c.end(), b.begin(), b.end());

    return c;
}

template<typename _Tp>
static inline std::string toString(const std::vector<_Tp>& shape, const String& name = "")
{
    std::ostringstream ss;
    if (!name.empty())
        ss << name << ' ';
    ss << '[';
    for(size_t i = 0, n = shape.size(); i < n; ++i)
        ss << ' ' << shape[i];
    ss << " ]";
    return ss.str();
}

template<typename _Tp>
static inline void print(const std::vector<_Tp>& shape, const String& name = "")
{
    std::cout << toString(shape, name) << std::endl;
}
template<typename _Tp>
static inline std::ostream& operator<<(std::ostream &out, const std::vector<_Tp>& shape)
{
    out << toString(shape);
    return out;
}

/// @brief Converts axis from `[-dims; dims)` (similar to Python's slice notation) to `[0; dims)` range.
static inline
int normalize_axis(int axis, int dims)
{
    CV_Check(axis, axis >= -dims && axis < dims, "");
    axis = (axis < 0) ? (dims + axis) : axis;
    CV_DbgCheck(axis, axis >= 0 && axis < dims, "");
    return axis;
}

static inline
int normalize_axis(int axis, const MatShape& shape)
{
    return normalize_axis(axis, (int)shape.size());
}

static inline
Range normalize_axis_range(const Range& r, int axisSize)
{
    if (r == Range::all())
        return Range(0, axisSize);
    CV_CheckGE(r.start, 0, "");
    Range clamped(r.start,
                  r.end > 0 ? std::min(r.end, axisSize) : axisSize + r.end + 1);
    CV_DbgCheckGE(clamped.start, 0, "");
    CV_CheckLT(clamped.start, clamped.end, "");
    CV_CheckLE(clamped.end, axisSize, "");
    return clamped;
}

static inline
bool isAllOnes(const MatShape &inputShape, int startPos, int endPos)
{
    CV_Assert(!inputShape.empty());

    CV_CheckGE((int) inputShape.size(), startPos, "");
    CV_CheckGE(startPos, 0, "");
    CV_CheckLE(startPos, endPos, "");
    CV_CheckLE((size_t)endPos, inputShape.size(), "");

    for (size_t i = startPos; i < endPos; i++)
    {
        if (inputShape[i] != 1)
            return false;
    }
    return true;
}

CV__DNN_INLINE_NS_END
}
}
#endif
