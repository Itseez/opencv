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

#ifndef OPENCV_CORE_MATRIX_OPERATIONS_HPP
#define OPENCV_CORE_MATRIX_OPERATIONS_HPP

#ifndef __cplusplus
#  error mat.inl.hpp header must be compiled as C++
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4127 5054 )
#endif

#if defined(CV_SKIP_DISABLE_CLANG_ENUM_WARNINGS)
  // nothing
#elif defined(CV_FORCE_DISABLE_CLANG_ENUM_WARNINGS)
  #define CV_DISABLE_CLANG_ENUM_WARNINGS
#elif defined(__clang__) && defined(__has_warning)
  #if __has_warning("-Wdeprecated-enum-enum-conversion") && __has_warning("-Wdeprecated-anon-enum-enum-conversion")
    #define CV_DISABLE_CLANG_ENUM_WARNINGS
  #endif
#endif
#ifdef CV_DISABLE_CLANG_ENUM_WARNINGS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif

namespace cv
{
CV__DEBUG_NS_BEGIN


//! @cond IGNORED

////////////////////////// Custom (raw) type wrapper //////////////////////////

template<typename _Tp> static inline
int rawType()
{
    CV_StaticAssert(sizeof(_Tp) <= CV_CN_MAX, "sizeof(_Tp) is too large");
    const int elemSize = sizeof(_Tp);
    return (int)CV_MAKETYPE(CV_8U, elemSize);
}

//////////////////////// Input/Output Arrays ////////////////////////

inline void _InputArray::init(int _flags, const void* _obj)
{ flags = _flags; obj = (void*)_obj; }

inline void _InputArray::init(int _flags, const void* _obj, Size _sz)
{ flags = _flags; obj = (void*)_obj; sz = _sz; }

inline void* _InputArray::getObj() const { return obj; }
inline int _InputArray::getFlags() const { return flags; }
inline Size _InputArray::getSz() const { return sz; }

inline _InputArray::_InputArray() { init(0 + NONE, 0); }
inline _InputArray::_InputArray(int _flags, void* _obj) { init(_flags, _obj); }
inline _InputArray::_InputArray(const Mat& m) { init(+MAT+ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<Mat>& vec) { init(+STD_VECTOR_MAT+ACCESS_READ, &vec); }
inline _InputArray::_InputArray(const UMat& m) { init(+UMAT+ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<UMat>& vec) { init(+STD_VECTOR_UMAT+ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_READ, &vec); }

template<typename _Tp, std::size_t _Nm> inline
_InputArray::_InputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_READ, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputArray::_InputArray(const std::array<Mat, _Nm>& arr)
{ init(+STD_ARRAY_MAT + ACCESS_READ, arr.data(), Size(1, _Nm)); }

inline
_InputArray::_InputArray(const std::vector<bool>& vec)
{ init(FIXED_TYPE + STD_BOOL_VECTOR + traits::Type<bool>::value + ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_READ, &vec); }

template<typename _Tp, int m, int n> inline
_InputArray::_InputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_READ, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputArray::_InputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_READ, vec, Size(n, 1)); }

template<typename _Tp> inline
_InputArray::_InputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + traits::Type<_Tp>::value + ACCESS_READ, &m); }

inline _InputArray::_InputArray(const double& val)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + CV_64F + ACCESS_READ, &val, Size(1,1)); }

inline _InputArray::_InputArray(const cuda::GpuMat& d_mat)
{ init(+CUDA_GPU_MAT + ACCESS_READ, &d_mat); }

inline _InputArray::_InputArray(const std::vector<cuda::GpuMat>& d_mat)
{	init(+STD_VECTOR_CUDA_GPU_MAT + ACCESS_READ, &d_mat);}

inline _InputArray::_InputArray(const ogl::Buffer& buf)
{ init(+OPENGL_BUFFER + ACCESS_READ, &buf); }

inline _InputArray::_InputArray(const cuda::HostMem& cuda_mem)
{ init(+CUDA_HOST_MEM + ACCESS_READ, &cuda_mem); }

template<typename _Tp> inline
_InputArray _InputArray::rawIn(const std::vector<_Tp>& vec)
{
    _InputArray v;
    v.flags = _InputArray::FIXED_TYPE + _InputArray::STD_VECTOR + rawType<_Tp>() + ACCESS_READ;
    v.obj = (void*)&vec;
    return v;
}

template<typename _Tp, std::size_t _Nm> inline
_InputArray _InputArray::rawIn(const std::array<_Tp, _Nm>& arr)
{
    _InputArray v;
    v.flags = FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_READ;
    v.obj = (void*)arr.data();
    v.sz = Size(1, _Nm);
    return v;
}

inline _InputArray::~_InputArray() {}

inline Mat _InputArray::getMat(int i) const
{
    if( kind() == MAT && i < 0 )
        return *(const Mat*)obj;
    return getMat_(i);
}

inline bool _InputArray::isMat() const { return kind() == _InputArray::MAT; }
inline bool _InputArray::isUMat() const  { return kind() == _InputArray::UMAT; }
inline bool _InputArray::isMatVector() const { return kind() == _InputArray::STD_VECTOR_MAT; }
inline bool _InputArray::isUMatVector() const  { return kind() == _InputArray::STD_VECTOR_UMAT; }
inline bool _InputArray::isMatx() const { return kind() == _InputArray::MATX; }
inline bool _InputArray::isVector() const { return kind() == _InputArray::STD_VECTOR ||
                                                   kind() == _InputArray::STD_BOOL_VECTOR ||
                                                   (kind() == _InputArray::MATX && (sz.width <= 1 || sz.height <= 1)); }
inline bool _InputArray::isGpuMat() const { return kind() == _InputArray::CUDA_GPU_MAT; }
inline bool _InputArray::isGpuMatVector() const { return kind() == _InputArray::STD_VECTOR_CUDA_GPU_MAT; }

////////////////////////////////////////////////////////////////////////////////////////

inline _OutputArray::_OutputArray() { init(+NONE + ACCESS_WRITE, 0); }
inline _OutputArray::_OutputArray(int _flags, void* _obj) { init(_flags + ACCESS_WRITE, _obj); }
inline _OutputArray::_OutputArray(Mat& m) { init(+MAT+ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<Mat>& vec) { init(+STD_VECTOR_MAT + ACCESS_WRITE, &vec); }
inline _OutputArray::_OutputArray(UMat& m) { init(+UMAT + ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<UMat>& vec) { init(+STD_VECTOR_UMAT + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp, std::size_t _Nm> inline
_OutputArray::_OutputArray(std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_OutputArray::_OutputArray(std::array<Mat, _Nm>& arr)
{ init(+STD_ARRAY_MAT + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &m); }

template<typename _Tp, int m, int n> inline
_OutputArray::_OutputArray(Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, &mtx, Size(n, m)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(_Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, vec, Size(n, 1)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp, std::size_t _Nm> inline
_OutputArray::_OutputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_OutputArray::_OutputArray(const std::array<Mat, _Nm>& arr)
{ init(FIXED_SIZE + STD_ARRAY_MAT + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &m); }

template<typename _Tp, int m, int n> inline
_OutputArray::_OutputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, &mtx, Size(n, m)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, vec, Size(n, 1)); }

inline _OutputArray::_OutputArray(cuda::GpuMat& d_mat)
{ init(+CUDA_GPU_MAT + ACCESS_WRITE, &d_mat); }

inline _OutputArray::_OutputArray(std::vector<cuda::GpuMat>& d_mat)
{	init(+STD_VECTOR_CUDA_GPU_MAT + ACCESS_WRITE, &d_mat);}

inline _OutputArray::_OutputArray(ogl::Buffer& buf)
{ init(+OPENGL_BUFFER + ACCESS_WRITE, &buf); }

inline _OutputArray::_OutputArray(cuda::HostMem& cuda_mem)
{ init(+CUDA_HOST_MEM + ACCESS_WRITE, &cuda_mem); }

inline _OutputArray::_OutputArray(const Mat& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<Mat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_WRITE, &vec); }

inline _OutputArray::_OutputArray(const UMat& m)
{ init(FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<UMat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_UMAT + ACCESS_WRITE, &vec); }

inline _OutputArray::_OutputArray(const cuda::GpuMat& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_GPU_MAT + ACCESS_WRITE, &d_mat); }


inline _OutputArray::_OutputArray(const ogl::Buffer& buf)
{ init(FIXED_TYPE + FIXED_SIZE + OPENGL_BUFFER + ACCESS_WRITE, &buf); }

inline _OutputArray::_OutputArray(const cuda::HostMem& cuda_mem)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_HOST_MEM + ACCESS_WRITE, &cuda_mem); }

template<typename _Tp> inline
_OutputArray _OutputArray::rawOut(std::vector<_Tp>& vec)
{
    _OutputArray v;
    v.flags = _InputArray::FIXED_TYPE + _InputArray::STD_VECTOR + rawType<_Tp>() + ACCESS_WRITE;
    v.obj = (void*)&vec;
    return v;
}

template<typename _Tp, std::size_t _Nm> inline
_OutputArray _OutputArray::rawOut(std::array<_Tp, _Nm>& arr)
{
    _OutputArray v;
    v.flags = FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE;
    v.obj = (void*)arr.data();
    v.sz = Size(1, _Nm);
    return v;
}

///////////////////////////////////////////////////////////////////////////////////////////

inline _InputOutputArray::_InputOutputArray() { init(0+ACCESS_RW, 0); }
inline _InputOutputArray::_InputOutputArray(int _flags, void* _obj) { init(_flags+ACCESS_RW, _obj); }
inline _InputOutputArray::_InputOutputArray(Mat& m) { init(+MAT+ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<Mat>& vec) { init(+STD_VECTOR_MAT+ACCESS_RW, &vec); }
inline _InputOutputArray::_InputOutputArray(UMat& m) { init(+UMAT+ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<UMat>& vec) { init(+STD_VECTOR_UMAT+ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(std::array<Mat, _Nm>& arr)
{ init(+STD_ARRAY_MAT + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + traits::Type<_Tp>::value + ACCESS_RW, &m); }

template<typename _Tp, int m, int n> inline
_InputOutputArray::_InputOutputArray(Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(_Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, vec, Size(n, 1)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(const std::array<Mat, _Nm>& arr)
{ init(FIXED_SIZE + STD_ARRAY_MAT + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + traits::Type<_Tp>::value + ACCESS_RW, &m); }

template<typename _Tp, int m, int n> inline
_InputOutputArray::_InputOutputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, vec, Size(n, 1)); }

inline _InputOutputArray::_InputOutputArray(cuda::GpuMat& d_mat)
{ init(+CUDA_GPU_MAT + ACCESS_RW, &d_mat); }

inline _InputOutputArray::_InputOutputArray(ogl::Buffer& buf)
{ init(+OPENGL_BUFFER + ACCESS_RW, &buf); }

inline _InputOutputArray::_InputOutputArray(cuda::HostMem& cuda_mem)
{ init(+CUDA_HOST_MEM + ACCESS_RW, &cuda_mem); }

inline _InputOutputArray::_InputOutputArray(const Mat& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<Mat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_RW, &vec); }

inline _InputOutputArray::_InputOutputArray(const UMat& m)
{ init(FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<UMat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_UMAT + ACCESS_RW, &vec); }

inline _InputOutputArray::_InputOutputArray(const cuda::GpuMat& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_GPU_MAT + ACCESS_RW, &d_mat); }

inline _InputOutputArray::_InputOutputArray(const std::vector<cuda::GpuMat>& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_CUDA_GPU_MAT + ACCESS_RW, &d_mat);}

template<> inline _InputOutputArray::_InputOutputArray(std::vector<cuda::GpuMat>& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_CUDA_GPU_MAT + ACCESS_RW, &d_mat);}

inline _InputOutputArray::_InputOutputArray(const ogl::Buffer& buf)
{ init(FIXED_TYPE + FIXED_SIZE + OPENGL_BUFFER + ACCESS_RW, &buf); }

inline _InputOutputArray::_InputOutputArray(const cuda::HostMem& cuda_mem)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_HOST_MEM + ACCESS_RW, &cuda_mem); }

template<typename _Tp> inline
_InputOutputArray _InputOutputArray::rawInOut(std::vector<_Tp>& vec)
{
    _InputOutputArray v;
    v.flags = _InputArray::FIXED_TYPE + _InputArray::STD_VECTOR + rawType<_Tp>() + ACCESS_RW;
    v.obj = (void*)&vec;
    return v;
}

template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray _InputOutputArray::rawInOut(std::array<_Tp, _Nm>& arr)
{
    _InputOutputArray v;
    v.flags = FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW;
    v.obj = (void*)arr.data();
    v.sz = Size(1, _Nm);
    return v;
}


template<typename _Tp> static inline _InputArray rawIn(_Tp& v) { return _InputArray::rawIn(v); }
template<typename _Tp> static inline _OutputArray rawOut(_Tp& v) { return _OutputArray::rawOut(v); }
template<typename _Tp> static inline _InputOutputArray rawInOut(_Tp& v) { return _InputOutputArray::rawInOut(v); }

CV__DEBUG_NS_END

//////////////////////////////////////////// Mat //////////////////////////////////////////

template<typename _Tp> inline
Mat::Mat(const std::vector<_Tp>& vec, bool copyData)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows((int)vec.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(vec.empty())
        return;
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&vec[0];
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)vec.size(), 1, traits::Type<_Tp>::value, (uchar*)&vec[0]).copyTo(*this);
}

template<typename _Tp, typename> inline
Mat::Mat(const std::initializer_list<_Tp> list)
    : Mat()
{
    CV_Assert(list.size() != 0);
    Mat((int)list.size(), 1, traits::Type<_Tp>::value, (uchar*)list.begin()).copyTo(*this);
}

template<typename _Tp> inline
Mat::Mat(const std::initializer_list<int> sizes, const std::initializer_list<_Tp> list)
    : Mat()
{
    size_t size_total = 1;
    for(auto s : sizes)
        size_total *= s;
    CV_Assert(list.size() != 0);
    CV_Assert(size_total == list.size());
    Mat((int)sizes.size(), (int*)sizes.begin(), traits::Type<_Tp>::value, (uchar*)list.begin()).copyTo(*this);
}

template<typename _Tp, std::size_t _Nm> inline
Mat::Mat(const std::array<_Tp, _Nm>& arr, bool copyData)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows((int)arr.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(arr.empty())
        return;
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)arr.data();
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)arr.size(), 1, traits::Type<_Tp>::value, (uchar*)arr.data()).copyTo(*this);
}

template<typename _Tp, int n> inline
Mat::Mat(const Vec<_Tp, n>& vec, bool copyData)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows(n), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)vec.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(n, 1, traits::Type<_Tp>::value, (void*)vec.val).copyTo(*this);
}


template<typename _Tp, int m, int n> inline
Mat::Mat(const Matx<_Tp,m,n>& M, bool copyData)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows(m), cols(n), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = cols * sizeof(_Tp);
        step[1] = sizeof(_Tp);
        datastart = data = (uchar*)M.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(m, n, traits::Type<_Tp>::value, (uchar*)M.val).copyTo(*this);
}

template<typename _Tp> inline
Mat::Mat(const Point_<_Tp>& pt, bool copyData)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows(2), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(2, 1, traits::Type<_Tp>::value);
        ((_Tp*)data)[0] = pt.x;
        ((_Tp*)data)[1] = pt.y;
    }
}

template<typename _Tp> inline
Mat::Mat(const Point3_<_Tp>& pt, bool copyData)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows(3), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(3, 1, traits::Type<_Tp>::value);
        ((_Tp*)data)[0] = pt.x;
        ((_Tp*)data)[1] = pt.y;
        ((_Tp*)data)[2] = pt.z;
    }
}

template<typename _Tp> inline
Mat::Mat(const MatCommaInitializer_<_Tp>& commaInitializer)
    : flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(0), rows(0), cols(0), data(0),
      datastart(0), dataend(0), allocator(0), u(0), size(&rows)
{
    *this = commaInitializer.operator Mat_<_Tp>();
}

inline
Mat Mat::row(int y) const
{
    return Mat(*this, Range(y, y + 1), Range::all());
}

inline
Mat Mat::col(int x) const
{
    return Mat(*this, Range::all(), Range(x, x + 1));
}

inline
Mat Mat::rowRange(int startrow, int endrow) const
{
    return Mat(*this, Range(startrow, endrow), Range::all());
}

inline
Mat Mat::rowRange(const Range& r) const
{
    return Mat(*this, r, Range::all());
}

inline
Mat Mat::colRange(int startcol, int endcol) const
{
    return Mat(*this, Range::all(), Range(startcol, endcol));
}

inline
Mat Mat::colRange(const Range& r) const
{
    return Mat(*this, Range::all(), r);
}

inline
Mat Mat::operator()( Range _rowRange, Range _colRange ) const
{
    return Mat(*this, _rowRange, _colRange);
}

inline
Mat Mat::operator()( const Rect& roi ) const
{
    return Mat(*this, roi);
}

inline
Mat Mat::operator()(const Range* ranges) const
{
    return Mat(*this, ranges);
}

inline
Mat Mat::operator()(const std::vector<Range>& ranges) const
{
    return Mat(*this, ranges);
}

inline
bool Mat::isContinuous() const
{
    return (flags & CONTINUOUS_FLAG) != 0;
}

inline
bool Mat::isSubmatrix() const
{
    return (flags & SUBMATRIX_FLAG) != 0;
}

inline
size_t Mat::elemSize() const
{
    size_t res = dims > 0 ? step.p[dims - 1] : 0;
    CV_DbgAssert(res != 0);
    return res;
}

inline
size_t Mat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int Mat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int Mat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int Mat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
uchar* Mat::ptr(int y)
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return data + step.p[0] * y;
}

inline
const uchar* Mat::ptr(int y) const
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return data + step.p[0] * y;
}

template<typename _Tp> inline
_Tp* Mat::ptr(int y)
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    _Tp* p = (_Tp*)((void*)(data + step.p[0] * y));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int y) const
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    const _Tp* p = (const _Tp*)((const void*)(data + step.p[0] * y));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

inline
uchar* Mat::ptr(int i0, int i1)
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

inline
const uchar* Mat::ptr(int i0, int i1) const
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

template<typename _Tp> inline
_Tp* Mat::ptr(int i0, int i1)
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);

    _Tp* p = (_Tp*)((void*)(data + i0 * step.p[0] + i1 * step.p[1]));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int i0, int i1) const
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);

    const _Tp* p = (const _Tp*)((const void*)(data + i0 * step.p[0] + i1 * step.p[1]));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

inline
uchar* Mat::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

inline
const uchar* Mat::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

template<typename _Tp> inline
_Tp* Mat::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);

    _Tp* p = (_Tp*)((void*)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);

    const _Tp* p = (const _Tp*)((const void*)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

inline
uchar* Mat::ptr(const int* idx)
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return p;
}

inline
const uchar* Mat::ptr(const int* idx) const
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return p;
}

template<typename _Tp> inline
_Tp* Mat::ptr(const int* idx)
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }

    _Tp* tPtr = (_Tp*)((void*)p);
    CV_DbgAssert(((uintptr_t)tPtr & (alignof(_Tp) - 1)) == 0);
    return tPtr;
}

template<typename _Tp> inline
const _Tp* Mat::ptr(const int* idx) const
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }

    const _Tp* tPtr = (const _Tp*)((const void*)p);
    CV_DbgAssert(((uintptr_t)tPtr & (alignof(_Tp) - 1)) == 0);
    return tPtr;
}

template<int n> inline
uchar* Mat::ptr(const Vec<int, n>& idx)
{
    return Mat::ptr(idx.val);
}

template<int n> inline
const uchar* Mat::ptr(const Vec<int, n>& idx) const
{
    return Mat::ptr(idx.val);
}

template<typename _Tp, int n> inline
_Tp* Mat::ptr(const Vec<int, n>& idx)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return Mat::ptr<_Tp>(idx.val);
}

template<typename _Tp, int n> inline
const _Tp* Mat::ptr(const Vec<int, n>& idx) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return Mat::ptr<_Tp>(idx.val);
}


template<typename _Tp> inline
_Tp& Mat::at(int i0, int i1)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());

    _Tp* p = ((_Tp*)((void*)(data + step.p[0] * i0))) + i1;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0, int i1) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());

    const _Tp* p = ((const _Tp*)((const void*)(data + step.p[0] * i0))) + i1;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& Mat::at(Point pt)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());

    _Tp* p = ((_Tp*)((void*)(data + step.p[0] * pt.y))) + pt.x;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat::at(Point pt) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());

    const _Tp* p = ((const _Tp*)((const void*)(data + step.p[0] * pt.y))) + pt.x;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& Mat::at(int i0)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    CV_DbgAssert(elemSize() == sizeof(_Tp));
    if( isContinuous() || size.p[0] == 1 )
    {
        _Tp* p = ((_Tp*)((void*)(data))) + i0;
        CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
        return *p;
    }
    if( size.p[1] == 1 )
    {
        _Tp* p = (_Tp*)((void*)(data + step.p[0] * i0));
        CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
        return *p;
    }
    int i = i0 / cols, j = i0 - i * cols;
    _Tp* p = ((_Tp*)((void*)(data + step.p[0] * i))) + j;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    CV_DbgAssert(elemSize() == sizeof(_Tp));
    if( isContinuous() || size.p[0] == 1 )
    {
        const _Tp* p = ((const _Tp*)((const void*)data)) + i0;
        CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
        return *p;
    }
    if( size.p[1] == 1 )
    {
        const _Tp* p = (const _Tp*)((const void*)(data + step.p[0] * i0));
        CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
        return *p;
    }
    int i = i0 / cols, j = i0 - i * cols;
    const _Tp* p = ((const _Tp*)((const void*)(data + step.p[0] * i))) + j;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& Mat::at(int i0, int i1, int i2)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    _Tp* p = (_Tp*)((void*)ptr(i0, i1, i2));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0, int i1, int i2) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    const _Tp* p = (const _Tp*)((const void*)ptr(i0, i1, i2));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& Mat::at(const int* idx)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    _Tp* p = (_Tp*)((void*)ptr(idx));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat::at(const int* idx) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    const _Tp* p = (const _Tp*)((const void*)ptr(idx));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp, int n> inline
_Tp& Mat::at(const Vec<int, n>& idx)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *ptr<_Tp>(idx.val);
}

template<typename _Tp, int n> inline
const _Tp& Mat::at(const Vec<int, n>& idx) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *ptr<_Tp>(idx.val);
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat::begin() const
{
    if (empty())
        return MatConstIterator_<_Tp>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatConstIterator_<_Tp>((const Mat_<_Tp>*)this);
}

template<typename _Tp> inline
std::reverse_iterator<MatConstIterator_<_Tp>> Mat::rbegin() const
{
    if (empty())
        return std::reverse_iterator<MatConstIterator_<_Tp>>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatConstIterator_<_Tp> it((const Mat_<_Tp>*)this);
    it += total();
    return std::reverse_iterator<MatConstIterator_<_Tp>> (it);
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat::end() const
{
    if (empty())
        return MatConstIterator_<_Tp>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatConstIterator_<_Tp> it((const Mat_<_Tp>*)this);
    it += total();
    return it;
}

template<typename _Tp> inline
std::reverse_iterator<MatConstIterator_<_Tp>> Mat::rend() const
{
    if (empty())
        return std::reverse_iterator<MatConstIterator_<_Tp>>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return std::reverse_iterator<MatConstIterator_<_Tp>>((const Mat_<_Tp>*)this);
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat::begin()
{
    if (empty())
        return MatIterator_<_Tp>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatIterator_<_Tp>((Mat_<_Tp>*)this);
}

template<typename _Tp> inline
std::reverse_iterator<MatIterator_<_Tp>> Mat::rbegin()
{
    if (empty())
        return std::reverse_iterator<MatIterator_<_Tp>>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatIterator_<_Tp> it((Mat_<_Tp>*)this);
    it += total();
    return std::reverse_iterator<MatIterator_<_Tp>>(it);
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat::end()
{
    if (empty())
        return MatIterator_<_Tp>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatIterator_<_Tp> it((Mat_<_Tp>*)this);
    it += total();
    return it;
}

template<typename _Tp> inline
std::reverse_iterator<MatIterator_<_Tp>> Mat::rend()
{
    if (empty())
        return std::reverse_iterator<MatIterator_<_Tp>>();
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return std::reverse_iterator<MatIterator_<_Tp>>(MatIterator_<_Tp>((Mat_<_Tp>*)this));
}

template<typename _Tp, typename Functor> inline
void Mat::forEach(const Functor& operation) {
    this->forEach_impl<_Tp>(operation);
}

template<typename _Tp, typename Functor> inline
void Mat::forEach(const Functor& operation) const {
    // call as not const
    (const_cast<Mat*>(this))->forEach<_Tp>(operation);
}

template<typename _Tp> inline
Mat::operator std::vector<_Tp>() const
{
    std::vector<_Tp> v;
    copyTo(v);
    return v;
}

template<typename _Tp, std::size_t _Nm> inline
Mat::operator std::array<_Tp, _Nm>() const
{
    std::array<_Tp, _Nm> v;
    copyTo(v);
    return v;
}

template<typename _Tp, int n> inline
Mat::operator Vec<_Tp, n>() const
{
    CV_Assert( data && dims <= 2 && (rows == 1 || cols == 1) &&
               rows + cols - 1 == n && channels() == 1 );

    if( isContinuous() && type() == traits::Type<_Tp>::value )
        return Vec<_Tp, n>((_Tp*)data);
    Vec<_Tp, n> v;
    Mat tmp(rows, cols, traits::Type<_Tp>::value, v.val);
    convertTo(tmp, tmp.type());
    return v;
}

template<typename _Tp, int m, int n> inline
Mat::operator Matx<_Tp, m, n>() const
{
    CV_Assert( data && dims <= 2 && rows == m && cols == n && channels() == 1 );

    if( isContinuous() && type() == traits::Type<_Tp>::value )
        return Matx<_Tp, m, n>((_Tp*)data);
    Matx<_Tp, m, n> mtx;
    Mat tmp(rows, cols, traits::Type<_Tp>::value, mtx.val);
    convertTo(tmp, tmp.type());
    return mtx;
}

template<typename _Tp> inline
void Mat::push_back(const _Tp& elem)
{
    if( !data )
    {
        *this = Mat(1, 1, traits::Type<_Tp>::value, (void*)&elem).clone();
        return;
    }
    CV_Assert(traits::Type<_Tp>::value == type() && cols == 1
              /* && dims == 2 (cols == 1 implies dims == 2) */);
    const uchar* tmp = dataend + step[0];
    if( !isSubmatrix() && isContinuous() && tmp <= datalimit )
    {
        *(_Tp*)(data + (size.p[0]++) * step.p[0]) = elem;
        dataend = tmp;
    }
    else
        push_back_(&elem);
}

template<typename _Tp> inline
void Mat::push_back(const Mat_<_Tp>& m)
{
    push_back((const Mat&)m);
}

template<> inline
void Mat::push_back(const MatExpr& expr)
{
    push_back(static_cast<Mat>(expr));
}


template<typename _Tp> inline
void Mat::push_back(const std::vector<_Tp>& v)
{
    push_back(Mat(v));
}


///////////////////////////// MatSize ////////////////////////////

inline
MatSize::MatSize(int* _p) CV_NOEXCEPT
    : p(_p) {}

inline
int MatSize::dims() const CV_NOEXCEPT
{
    return (p - 1)[0];
}

inline
Size MatSize::operator()() const
{
    CV_DbgAssert(dims() <= 2);
    return Size(p[1], p[0]);
}

inline
const int& MatSize::operator[](int i) const
{
    CV_DbgAssert(i < dims());
#ifdef __OPENCV_BUILD
    CV_DbgAssert(i >= 0);
#endif
    return p[i];
}

inline
int& MatSize::operator[](int i)
{
    CV_DbgAssert(i < dims());
#ifdef __OPENCV_BUILD
    CV_DbgAssert(i >= 0);
#endif
    return p[i];
}

inline
MatSize::operator const int*() const CV_NOEXCEPT
{
    return p;
}

inline
bool MatSize::operator != (const MatSize& sz) const CV_NOEXCEPT
{
    return !(*this == sz);
}



///////////////////////////// MatStep ////////////////////////////

inline
MatStep::MatStep() CV_NOEXCEPT
{
    p = buf; p[0] = p[1] = 0;
}

inline
MatStep::MatStep(size_t s) CV_NOEXCEPT
{
    p = buf; p[0] = s; p[1] = 0;
}

inline
const size_t& MatStep::operator[](int i) const CV_NOEXCEPT
{
    return p[i];
}

inline
size_t& MatStep::operator[](int i) CV_NOEXCEPT
{
    return p[i];
}

inline MatStep::operator size_t() const
{
    CV_DbgAssert( p == buf );
    return buf[0];
}

inline MatStep& MatStep::operator = (size_t s)
{
    CV_DbgAssert( p == buf );
    buf[0] = s;
    return *this;
}



////////////////////////////// Mat_<_Tp> ////////////////////////////

template<typename _Tp> inline
Mat_<_Tp>::Mat_() CV_NOEXCEPT
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) + traits::Type<_Tp>::value;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols)
    : Mat(_rows, _cols, traits::Type<_Tp>::value)
{
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols, const _Tp& value)
    : Mat(_rows, _cols, traits::Type<_Tp>::value)
{
    *this = value;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Size _sz)
    : Mat(_sz.height, _sz.width, traits::Type<_Tp>::value)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Size _sz, const _Tp& value)
    : Mat(_sz.height, _sz.width, traits::Type<_Tp>::value)
{
    *this = value;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _dims, const int* _sz)
    : Mat(_dims, _sz, traits::Type<_Tp>::value)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _dims, const int* _sz, const _Tp& _s)
    : Mat(_dims, _sz, traits::Type<_Tp>::value, Scalar(_s))
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _dims, const int* _sz, _Tp* _data, const size_t* _steps)
    : Mat(_dims, _sz, traits::Type<_Tp>::value, _data, _steps)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_<_Tp>& m, const Range* ranges)
    : Mat(m, ranges)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_<_Tp>& m, const std::vector<Range>& ranges)
    : Mat(m, ranges)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat& m)
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) + traits::Type<_Tp>::value;
    *this = m;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_& m)
    : Mat(m)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols, _Tp* _data, size_t steps)
    : Mat(_rows, _cols, traits::Type<_Tp>::value, _data, steps)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_& m, const Range& _rowRange, const Range& _colRange)
    : Mat(m, _rowRange, _colRange)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_& m, const Rect& roi)
    : Mat(m, roi)
{}

template<typename _Tp> template<int n> inline
Mat_<_Tp>::Mat_(const Vec<typename DataType<_Tp>::channel_type, n>& vec, bool copyData)
    : Mat(n / DataType<_Tp>::channels, 1, traits::Type<_Tp>::value, (void*)&vec)
{
    CV_Assert(n%DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> template<int m, int n> inline
Mat_<_Tp>::Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n>& M, bool copyData)
    : Mat(m, n / DataType<_Tp>::channels, traits::Type<_Tp>::value, (void*)&M)
{
    CV_Assert(n % DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Point_<typename DataType<_Tp>::channel_type>& pt, bool copyData)
    : Mat(2 / DataType<_Tp>::channels, 1, traits::Type<_Tp>::value, (void*)&pt)
{
    CV_Assert(2 % DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Point3_<typename DataType<_Tp>::channel_type>& pt, bool copyData)
    : Mat(3 / DataType<_Tp>::channels, 1, traits::Type<_Tp>::value, (void*)&pt)
{
    CV_Assert(3 % DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const MatCommaInitializer_<_Tp>& commaInitializer)
    : Mat(commaInitializer)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const std::vector<_Tp>& vec, bool copyData)
    : Mat(vec, copyData)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(std::initializer_list<_Tp> list)
    : Mat(list)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const std::initializer_list<int> sizes, std::initializer_list<_Tp> list)
    : Mat(sizes, list)
{}

template<typename _Tp> template<std::size_t _Nm> inline
Mat_<_Tp>::Mat_(const std::array<_Tp, _Nm>& arr, bool copyData)
    : Mat(arr, copyData)
{}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const Mat& m)
{
    if (m.empty())
    {
        release();
        return *this;
    }
    if( traits::Type<_Tp>::value == m.type() )
    {
        Mat::operator = (m);
        return *this;
    }
    if( traits::Depth<_Tp>::value == m.depth() )
    {
        return (*this = m.reshape(DataType<_Tp>::channels, m.dims, 0));
    }
    CV_Assert(DataType<_Tp>::channels == m.channels() || m.empty());
    m.convertTo(*this, type());
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const Mat_& m)
{
    Mat::operator=(m);
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const _Tp& s)
{
    typedef typename DataType<_Tp>::vec_type VT;
    Mat::operator=(Scalar((const VT&)s));
    return *this;
}

template<typename _Tp> inline
void Mat_<_Tp>::create(int _rows, int _cols)
{
    Mat::create(_rows, _cols, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
void Mat_<_Tp>::create(Size _sz)
{
    Mat::create(_sz, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
void Mat_<_Tp>::create(int _dims, const int* _sz)
{
    Mat::create(_dims, _sz, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
void Mat_<_Tp>::release()
{
    Mat::release();
    flags = (flags & ~CV_MAT_TYPE_MASK) + traits::Type<_Tp>::value;
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::cross(const Mat_& m) const
{
    return Mat_<_Tp>(Mat::cross(m));
}

template<typename _Tp> template<typename T2> inline
Mat_<_Tp>::operator Mat_<T2>() const
{
    return Mat_<T2>(static_cast<const Mat&>(*this));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::row(int y) const
{
    return Mat_(*this, Range(y, y+1), Range::all());
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::col(int x) const
{
    return Mat_(*this, Range::all(), Range(x, x+1));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::diag(int d) const
{
    return Mat_(Mat::diag(d));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::clone() const
{
    return Mat_(Mat::clone());
}

template<typename _Tp> inline
size_t Mat_<_Tp>::elemSize() const
{
    CV_DbgAssert( Mat::elemSize() == sizeof(_Tp) );
    return sizeof(_Tp);
}

template<typename _Tp> inline
size_t Mat_<_Tp>::elemSize1() const
{
    CV_DbgAssert( Mat::elemSize1() == sizeof(_Tp) / DataType<_Tp>::channels );
    return sizeof(_Tp) / DataType<_Tp>::channels;
}

template<typename _Tp> inline
int Mat_<_Tp>::type() const
{
    CV_DbgAssert( Mat::type() == traits::Type<_Tp>::value );
    return traits::Type<_Tp>::value;
}

template<typename _Tp> inline
int Mat_<_Tp>::depth() const
{
    CV_DbgAssert( Mat::depth() == traits::Depth<_Tp>::value );
    return traits::Depth<_Tp>::value;
}

template<typename _Tp> inline
int Mat_<_Tp>::channels() const
{
    CV_DbgAssert( Mat::channels() == DataType<_Tp>::channels );
    return DataType<_Tp>::channels;
}

template<typename _Tp> inline
size_t Mat_<_Tp>::stepT(int i) const
{
    return step.p[i] / elemSize();
}

template<typename _Tp> inline
size_t Mat_<_Tp>::step1(int i) const
{
    return step.p[i] / elemSize1();
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    return (Mat_<_Tp>&)(Mat::adjustROI(dtop, dbottom, dleft, dright));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()( const Range& _rowRange, const Range& _colRange ) const
{
    return Mat_<_Tp>(*this, _rowRange, _colRange);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()( const Rect& roi ) const
{
    return Mat_<_Tp>(*this, roi);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()( const Range* ranges ) const
{
    return Mat_<_Tp>(*this, ranges);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()(const std::vector<Range>& ranges) const
{
    return Mat_<_Tp>(*this, ranges);
}

template<typename _Tp> inline
_Tp* Mat_<_Tp>::operator [](int y)
{
    CV_DbgAssert( 0 <= y && y < size.p[0] );
    _Tp* p = (_Tp*)((void*)(data + y*step.p[0]));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* Mat_<_Tp>::operator [](int y) const
{
    CV_DbgAssert( 0 <= y && y < size.p[0] );
    const _Tp* p = (const _Tp*)((const void*)(data + y*step.p[0]));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(int i0, int i1)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert(type() == traits::Type<_Tp>::value);

    _Tp* p = ((_Tp*)((void*)(data + step.p[0] * i0))) + i1;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(int i0, int i1) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert(type() == traits::Type<_Tp>::value);

    const _Tp* p = ((const _Tp*)((const void*)(data + step.p[0] * i0))) + i1;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(Point pt)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)pt.x < (unsigned)size.p[1]);
    CV_DbgAssert(type() == traits::Type<_Tp>::value);

    _Tp* p = ((_Tp*)((void*)(data + step.p[0] * pt.y))) + pt.x;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(Point pt) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)pt.x < (unsigned)size.p[1]);
    CV_DbgAssert(type() == traits::Type<_Tp>::value);

    const _Tp* p = ((const _Tp*)((const void*)(data + step.p[0] * pt.y))) + pt.x;
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(const int* idx)
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(const int* idx) const
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> template<int n> inline
_Tp& Mat_<_Tp>::operator ()(const Vec<int, n>& idx)
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> template<int n> inline
const _Tp& Mat_<_Tp>::operator ()(const Vec<int, n>& idx) const
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(int i0)
{
    return this->at<_Tp>(i0);
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(int i0) const
{
    return this->at<_Tp>(i0);
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(int i0, int i1, int i2)
{
    return this->at<_Tp>(i0, i1, i2);
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(int i0, int i1, int i2) const
{
    return this->at<_Tp>(i0, i1, i2);
}

template<typename _Tp> inline
Mat_<_Tp>::operator std::vector<_Tp>() const
{
    std::vector<_Tp> v;
    copyTo(v);
    return v;
}

template<typename _Tp> template<std::size_t _Nm> inline
Mat_<_Tp>::operator std::array<_Tp, _Nm>() const
{
    std::array<_Tp, _Nm> a;
    copyTo(a);
    return a;
}

template<typename _Tp> template<int n> inline
Mat_<_Tp>::operator Vec<typename DataType<_Tp>::channel_type, n>() const
{
    CV_Assert(n % DataType<_Tp>::channels == 0);

#if defined _MSC_VER
    const Mat* pMat = (const Mat*)this; // workaround for MSVS <= 2012 compiler bugs (but GCC 4.6 dislikes this workaround)
    return pMat->operator Vec<typename DataType<_Tp>::channel_type, n>();
#else
    return this->Mat::operator Vec<typename DataType<_Tp>::channel_type, n>();
#endif
}

template<typename _Tp> template<int m, int n> inline
Mat_<_Tp>::operator Matx<typename DataType<_Tp>::channel_type, m, n>() const
{
    CV_Assert(n % DataType<_Tp>::channels == 0);

#if defined _MSC_VER
    const Mat* pMat = (const Mat*)this; // workaround for MSVS <= 2012 compiler bugs (but GCC 4.6 dislikes this workaround)
    Matx<typename DataType<_Tp>::channel_type, m, n> res = pMat->operator Matx<typename DataType<_Tp>::channel_type, m, n>();
    return res;
#else
    Matx<typename DataType<_Tp>::channel_type, m, n> res = this->Mat::operator Matx<typename DataType<_Tp>::channel_type, m, n>();
    return res;
#endif
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat_<_Tp>::begin() const
{
    return Mat::begin<_Tp>();
}

template<typename _Tp> inline
std::reverse_iterator<MatConstIterator_<_Tp>> Mat_<_Tp>::rbegin() const
{
    return Mat::rbegin<_Tp>();
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat_<_Tp>::end() const
{
    return Mat::end<_Tp>();
}

template<typename _Tp> inline
std::reverse_iterator<MatConstIterator_<_Tp>> Mat_<_Tp>::rend() const
{
    return Mat::rend<_Tp>();
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat_<_Tp>::begin()
{
    return Mat::begin<_Tp>();
}

template<typename _Tp> inline
std::reverse_iterator<MatIterator_<_Tp>> Mat_<_Tp>::rbegin()
{
    return Mat::rbegin<_Tp>();
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat_<_Tp>::end()
{
    return Mat::end<_Tp>();
}

template<typename _Tp> inline
std::reverse_iterator<MatIterator_<_Tp>> Mat_<_Tp>::rend()
{
    return Mat::rend<_Tp>();
}

template<typename _Tp> template<typename Functor> inline
void Mat_<_Tp>::forEach(const Functor& operation) {
    Mat::forEach<_Tp, Functor>(operation);
}

template<typename _Tp> template<typename Functor> inline
void Mat_<_Tp>::forEach(const Functor& operation) const {
    Mat::forEach<_Tp, Functor>(operation);
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Mat_&& m)
    : Mat(std::move(m))
{
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (Mat_&& m)
{
    Mat::operator = (std::move(m));
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Mat&& m)
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) + traits::Type<_Tp>::value;
    *this = std::move(m);
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (Mat&& m)
{
    if (m.empty())
    {
        release();
        return *this;
    }
    if( traits::Type<_Tp>::value == m.type() )
    {
        Mat::operator = ((Mat&&)m);
        return *this;
    }
    if( traits::Depth<_Tp>::value == m.depth() )
    {
        Mat::operator = ((Mat&&)m.reshape(DataType<_Tp>::channels, m.dims, 0));
        return *this;
    }
    CV_DbgAssert(DataType<_Tp>::channels == m.channels());
    m.convertTo(*this, type());
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(MatExpr&& e)
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) + traits::Type<_Tp>::value;
    *this = Mat(e);
}


///////////////////////////// SparseMat /////////////////////////////

inline
SparseMat SparseMat::clone() const
{
    SparseMat temp;
    this->copyTo(temp);
    return temp;
}

inline
size_t SparseMat::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t SparseMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int SparseMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int SparseMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int SparseMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
const int* SparseMat::size() const
{
    return hdr ? hdr->size : 0;
}

inline
int SparseMat::size(int i) const
{
    if( hdr )
    {
        CV_DbgAssert((unsigned)i < (unsigned)hdr->dims);
        return hdr->size[i];
    }
    return 0;
}

inline
int SparseMat::dims() const
{
    return hdr ? hdr->dims : 0;
}

inline
size_t SparseMat::nzcount() const
{
    return hdr ? hdr->nodeCount : 0;
}

template<typename _Tp> inline
_Tp& SparseMat::ref(int i0, size_t* hashval)
{
    _Tp* p = (_Tp*)((void*)(((SparseMat*)this)->ptr(i0, true, hashval)));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& SparseMat::ref(int i0, int i1, size_t* hashval)
{
    _Tp* p = (_Tp*)((void*)(((SparseMat*)this)->ptr(i0, i1, true, hashval)));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& SparseMat::ref(int i0, int i1, int i2, size_t* hashval)
{
    _Tp* p = (_Tp*)((void*)(((SparseMat*)this)->ptr(i0, i1, i2, true, hashval)));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp& SparseMat::ref(const int* idx, size_t* hashval)
{
    _Tp* p = (_Tp*)((void*)(((SparseMat*)this)->ptr(idx, true, hashval)));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
_Tp SparseMat::value(int i0, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(i0, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
_Tp SparseMat::value(int i0, int i1, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(i0, i1, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
_Tp SparseMat::value(int i0, int i1, int i2, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(i0, i1, i2, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
_Tp SparseMat::value(const int* idx, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(idx, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
const _Tp* SparseMat::find(int i0, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(i0, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* SparseMat::find(int i0, int i1, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((void*)((SparseMat*)this)->ptr(i0, i1, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* SparseMat::find(int i0, int i1, int i2, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(i0, i1, i2, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
const _Tp* SparseMat::find(const int* idx, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((const void*)((SparseMat*)this)->ptr(idx, false, hashval));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return p;
}

template<typename _Tp> inline
_Tp& SparseMat::value(Node* n)
{
    _Tp* p = (_Tp*)((void*)((uchar*)n + hdr->valueOffset));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

template<typename _Tp> inline
const _Tp& SparseMat::value(const Node* n) const
{
    const _Tp* p = (const _Tp*)((const void*)((const uchar*)n + hdr->valueOffset));
    CV_DbgAssert(((uintptr_t)p & (alignof(_Tp) - 1)) == 0);
    return *p;
}

inline
SparseMat::Node* SparseMat::node(size_t nidx)
{
    return (Node*)(void*)&hdr->pool[nidx];
}

inline
const SparseMat::Node* SparseMat::node(size_t nidx) const
{
    return (const Node*)(const void*)&hdr->pool[nidx];
}

inline
SparseMatIterator SparseMat::begin()
{
    return SparseMatIterator(this);
}

inline
SparseMatConstIterator SparseMat::begin() const
{
    return SparseMatConstIterator(this);
}

inline
SparseMatIterator SparseMat::end()
{
    SparseMatIterator it(this);
    it.seekEnd();
    return it;
}

inline
SparseMatConstIterator SparseMat::end() const
{
    SparseMatConstIterator it(this);
    it.seekEnd();
    return it;
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat::begin()
{
    return SparseMatIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat::begin() const
{
    return SparseMatConstIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat::end()
{
    SparseMatIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat::end() const
{
    SparseMatConstIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}



///////////////////////////// SparseMat_ ////////////////////////////

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_()
{
    flags = +MAGIC_VAL + traits::Type<_Tp>::value;
}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(int _dims, const int* _sizes)
    : SparseMat(_dims, _sizes, traits::Type<_Tp>::value)
{}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(const SparseMat& m)
{
    if( m.type() == traits::Type<_Tp>::value )
        *this = (const SparseMat_<_Tp>&)m;
    else
        m.convertTo(*this, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(const SparseMat_<_Tp>& m)
{
    this->flags = m.flags;
    this->hdr = m.hdr;
    if( this->hdr )
        CV_XADD(&this->hdr->refcount, 1);
}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(const Mat& m)
{
    SparseMat sm(m);
    *this = sm;
}

template<typename _Tp> inline
SparseMat_<_Tp>& SparseMat_<_Tp>::operator = (const SparseMat_<_Tp>& m)
{
    if( this != &m )
    {
        if( m.hdr ) CV_XADD(&m.hdr->refcount, 1);
        release();
        flags = m.flags;
        hdr = m.hdr;
    }
    return *this;
}

template<typename _Tp> inline
SparseMat_<_Tp>& SparseMat_<_Tp>::operator = (const SparseMat& m)
{
    if( m.type() == traits::Type<_Tp>::value )
        return (*this = (const SparseMat_<_Tp>&)m);
    m.convertTo(*this, traits::Type<_Tp>::value);
    return *this;
}

template<typename _Tp> inline
SparseMat_<_Tp>& SparseMat_<_Tp>::operator = (const Mat& m)
{
    return (*this = SparseMat(m));
}

template<typename _Tp> inline
SparseMat_<_Tp> SparseMat_<_Tp>::clone() const
{
    SparseMat_<_Tp> m;
    this->copyTo(m);
    return m;
}

template<typename _Tp> inline
void SparseMat_<_Tp>::create(int _dims, const int* _sizes)
{
    SparseMat::create(_dims, _sizes, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
int SparseMat_<_Tp>::type() const
{
    return traits::Type<_Tp>::value;
}

template<typename _Tp> inline
int SparseMat_<_Tp>::depth() const
{
    return traits::Depth<_Tp>::value;
}

template<typename _Tp> inline
int SparseMat_<_Tp>::channels() const
{
    return DataType<_Tp>::channels;
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(int i0, size_t* hashval)
{
    return SparseMat::ref<_Tp>(i0, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(int i0, size_t* hashval) const
{
    return SparseMat::value<_Tp>(i0, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(int i0, int i1, size_t* hashval)
{
    return SparseMat::ref<_Tp>(i0, i1, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(int i0, int i1, size_t* hashval) const
{
    return SparseMat::value<_Tp>(i0, i1, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(int i0, int i1, int i2, size_t* hashval)
{
    return SparseMat::ref<_Tp>(i0, i1, i2, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(int i0, int i1, int i2, size_t* hashval) const
{
    return SparseMat::value<_Tp>(i0, i1, i2, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(const int* idx, size_t* hashval)
{
    return SparseMat::ref<_Tp>(idx, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(const int* idx, size_t* hashval) const
{
    return SparseMat::value<_Tp>(idx, hashval);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat_<_Tp>::begin()
{
    return SparseMatIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat_<_Tp>::begin() const
{
    return SparseMatConstIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat_<_Tp>::end()
{
    SparseMatIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat_<_Tp>::end() const
{
    SparseMatConstIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}



////////////////////////// MatConstIterator /////////////////////////

inline
MatConstIterator::MatConstIterator()
    : m(0), elemSize(0), ptr(0), sliceStart(0), sliceEnd(0)
{}

inline
MatConstIterator::MatConstIterator(const Mat* _m)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    if( m && m->isContinuous() )
    {
        CV_Assert(!m->empty());
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total()*elemSize;
    }
    seek((const int*)0);
}

inline
MatConstIterator::MatConstIterator(const Mat* _m, int _row, int _col)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    CV_Assert(m && m->dims <= 2);
    if( m->isContinuous() )
    {
        CV_Assert(!m->empty());
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total()*elemSize;
    }
    int idx[] = {_row, _col};
    seek(idx);
}

inline
MatConstIterator::MatConstIterator(const Mat* _m, Point _pt)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    CV_Assert(m && m->dims <= 2);
    if( m->isContinuous() )
    {
        CV_Assert(!m->empty());
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total()*elemSize;
    }
    int idx[] = {_pt.y, _pt.x};
    seek(idx);
}

inline
MatConstIterator::MatConstIterator(const MatConstIterator& it)
    : m(it.m), elemSize(it.elemSize), ptr(it.ptr), sliceStart(it.sliceStart), sliceEnd(it.sliceEnd)
{}

inline
MatConstIterator& MatConstIterator::operator = (const MatConstIterator& it )
{
    m = it.m; elemSize = it.elemSize; ptr = it.ptr;
    sliceStart = it.sliceStart; sliceEnd = it.sliceEnd;
    return *this;
}

inline
const uchar* MatConstIterator::operator *() const
{
    return ptr;
}

inline MatConstIterator& MatConstIterator::operator += (ptrdiff_t ofs)
{
    if( !m || ofs == 0 )
        return *this;
    ptrdiff_t ofsb = ofs*elemSize;
    ptr += ofsb;
    if( ptr < sliceStart || sliceEnd <= ptr )
    {
        ptr -= ofsb;
        seek(ofs, true);
    }
    return *this;
}

inline
MatConstIterator& MatConstIterator::operator -= (ptrdiff_t ofs)
{
    return (*this += -ofs);
}

inline
MatConstIterator& MatConstIterator::operator --()
{
    if( m && (ptr -= elemSize) < sliceStart )
    {
        ptr += elemSize;
        seek(-1, true);
    }
    return *this;
}

inline
MatConstIterator MatConstIterator::operator --(int)
{
    MatConstIterator b = *this;
    *this += -1;
    return b;
}

inline
MatConstIterator& MatConstIterator::operator ++()
{
    if( m && (ptr += elemSize) >= sliceEnd )
    {
        ptr -= elemSize;
        seek(1, true);
    }
    return *this;
}

inline MatConstIterator MatConstIterator::operator ++(int)
{
    MatConstIterator b = *this;
    *this += 1;
    return b;
}


static inline
bool operator == (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

static inline
bool operator != (const MatConstIterator& a, const MatConstIterator& b)
{
    return !(a == b);
}

static inline
bool operator < (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr < b.ptr;
}

static inline
bool operator > (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr > b.ptr;
}

static inline
bool operator <= (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr <= b.ptr;
}

static inline
bool operator >= (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr >= b.ptr;
}

static inline
ptrdiff_t operator - (const MatConstIterator& b, const MatConstIterator& a)
{
    if( a.m != b.m )
        return ((size_t)(-1) >> 1);
    if( a.sliceEnd == b.sliceEnd )
        return (b.ptr - a.ptr)/static_cast<ptrdiff_t>(b.elemSize);

    return b.lpos() - a.lpos();
}

static inline
MatConstIterator operator + (const MatConstIterator& a, ptrdiff_t ofs)
{
    MatConstIterator b = a;
    return b += ofs;
}

static inline
MatConstIterator operator + (ptrdiff_t ofs, const MatConstIterator& a)
{
    MatConstIterator b = a;
    return b += ofs;
}

static inline
MatConstIterator operator - (const MatConstIterator& a, ptrdiff_t ofs)
{
    MatConstIterator b = a;
    return b += -ofs;
}


inline
const uchar* MatConstIterator::operator [](ptrdiff_t i) const
{
    return *(*this + i);
}



///////////////////////// MatConstIterator_ /////////////////////////

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_()
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m)
    : MatConstIterator(_m)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col)
    : MatConstIterator(_m, _row, _col)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m, Point _pt)
    : MatConstIterator(_m, _pt)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const MatConstIterator_& it)
    : MatConstIterator(it)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator = (const MatConstIterator_& it )
{
    MatConstIterator::operator = (it);
    return *this;
}

template<typename _Tp> inline
const _Tp& MatConstIterator_<_Tp>::operator *() const
{
    return *(_Tp*)((void*)(this->ptr));
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator += (ptrdiff_t ofs)
{
    MatConstIterator::operator += (ofs);
    return *this;
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator -= (ptrdiff_t ofs)
{
    return (*this += -ofs);
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator --()
{
    MatConstIterator::operator --();
    return *this;
}

template<typename _Tp> inline
MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator --(int)
{
    MatConstIterator_ b = *this;
    MatConstIterator::operator --();
    return b;
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator ++()
{
    MatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator ++(int)
{
    MatConstIterator_ b = *this;
    MatConstIterator::operator ++();
    return b;
}


template<typename _Tp> inline
Point MatConstIterator_<_Tp>::pos() const
{
    if( !m )
        return Point();
    CV_DbgAssert( m->dims <= 2 );
    if( m->isContinuous() )
    {
        ptrdiff_t ofs = (const _Tp*)ptr - (const _Tp*)m->data;
        int y = (int)(ofs / m->cols);
        int x = (int)(ofs - (ptrdiff_t)y * m->cols);
        return Point(x, y);
    }
    else
    {
        ptrdiff_t ofs = (uchar*)ptr - m->data;
        int y = (int)(ofs / m->step);
        int x = (int)((ofs - y * m->step)/sizeof(_Tp));
        return Point(x, y);
    }
}


template<typename _Tp> static inline
bool operator == (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

template<typename _Tp> static inline
bool operator != (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{
    return a.m != b.m || a.ptr != b.ptr;
}

template<typename _Tp> static inline
MatConstIterator_<_Tp> operator + (const MatConstIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatConstIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatConstIterator_<_Tp> operator + (ptrdiff_t ofs, const MatConstIterator_<_Tp>& a)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatConstIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatConstIterator_<_Tp> operator - (const MatConstIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a - ofs;
    return (MatConstIterator_<_Tp>&)t;
}

template<typename _Tp> inline
const _Tp& MatConstIterator_<_Tp>::operator [](ptrdiff_t i) const
{
    return *(_Tp*)((void*)MatConstIterator::operator [](i));
}



//////////////////////////// MatIterator_ ///////////////////////////

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_()
    : MatConstIterator_<_Tp>()
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m)
    : MatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, int _row, int _col)
    : MatConstIterator_<_Tp>(_m, _row, _col)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, Point _pt)
    : MatConstIterator_<_Tp>(_m, _pt)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, const int* _idx)
    : MatConstIterator_<_Tp>(_m, _idx)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(const MatIterator_& it)
    : MatConstIterator_<_Tp>(it)
{}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator = (const MatIterator_<_Tp>& it )
{
    MatConstIterator::operator = (it);
    return *this;
}

template<typename _Tp> inline
_Tp& MatIterator_<_Tp>::operator *() const
{
    return *(_Tp*)((void*)(this->ptr));
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator += (ptrdiff_t ofs)
{
    MatConstIterator::operator += (ofs);
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator -= (ptrdiff_t ofs)
{
    MatConstIterator::operator += (-ofs);
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator --()
{
    MatConstIterator::operator --();
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp> MatIterator_<_Tp>::operator --(int)
{
    MatIterator_ b = *this;
    MatConstIterator::operator --();
    return b;
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator ++()
{
    MatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp> MatIterator_<_Tp>::operator ++(int)
{
    MatIterator_ b = *this;
    MatConstIterator::operator ++();
    return b;
}

template<typename _Tp> inline
_Tp& MatIterator_<_Tp>::operator [](ptrdiff_t i) const
{
    return *(*this + i);
}


template<typename _Tp> static inline
bool operator == (const MatIterator_<_Tp>& a, const MatIterator_<_Tp>& b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

template<typename _Tp> static inline
bool operator != (const MatIterator_<_Tp>& a, const MatIterator_<_Tp>& b)
{
    return a.m != b.m || a.ptr != b.ptr;
}

template<typename _Tp> static inline
MatIterator_<_Tp> operator + (const MatIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatIterator_<_Tp> operator + (ptrdiff_t ofs, const MatIterator_<_Tp>& a)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatIterator_<_Tp> operator - (const MatIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a - ofs;
    return (MatIterator_<_Tp>&)t;
}



/////////////////////// SparseMatConstIterator //////////////////////

inline
SparseMatConstIterator::SparseMatConstIterator()
    : m(0), hashidx(0), ptr(0)
{}

inline
SparseMatConstIterator::SparseMatConstIterator(const SparseMatConstIterator& it)
    : m(it.m), hashidx(it.hashidx), ptr(it.ptr)
{}

inline SparseMatConstIterator& SparseMatConstIterator::operator = (const SparseMatConstIterator& it)
{
    if( this != &it )
    {
        m = it.m;
        hashidx = it.hashidx;
        ptr = it.ptr;
    }
    return *this;
}

template<typename _Tp> inline
const _Tp& SparseMatConstIterator::value() const
{
    return *(const _Tp*)((const void*)ptr);
}

inline
const SparseMat::Node* SparseMatConstIterator::node() const
{
    return (ptr && m && m->hdr) ? (const SparseMat::Node*)(const void*)(ptr - m->hdr->valueOffset) : 0;
}

inline
SparseMatConstIterator SparseMatConstIterator::operator ++(int)
{
    SparseMatConstIterator it = *this;
    ++*this;
    return it;
}

inline
void SparseMatConstIterator::seekEnd()
{
    if( m && m->hdr )
    {
        hashidx = m->hdr->hashtab.size();
        ptr = 0;
    }
}


static inline
bool operator == (const SparseMatConstIterator& it1, const SparseMatConstIterator& it2)
{
    return it1.m == it2.m && it1.ptr == it2.ptr;
}

static inline
bool operator != (const SparseMatConstIterator& it1, const SparseMatConstIterator& it2)
{
    return !(it1 == it2);
}



///////////////////////// SparseMatIterator /////////////////////////

inline
SparseMatIterator::SparseMatIterator()
{}

inline
SparseMatIterator::SparseMatIterator(SparseMat* _m)
    : SparseMatConstIterator(_m)
{}

inline
SparseMatIterator::SparseMatIterator(const SparseMatIterator& it)
    : SparseMatConstIterator(it)
{}

inline
SparseMatIterator& SparseMatIterator::operator = (const SparseMatIterator& it)
{
    (SparseMatConstIterator&)*this = it;
    return *this;
}

template<typename _Tp> inline
_Tp& SparseMatIterator::value() const
{
    return *(_Tp*)((void*)ptr);
}

inline
SparseMat::Node* SparseMatIterator::node() const
{
    return (SparseMat::Node*)SparseMatConstIterator::node();
}

inline
SparseMatIterator& SparseMatIterator::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

inline
SparseMatIterator SparseMatIterator::operator ++(int)
{
    SparseMatIterator it = *this;
    ++*this;
    return it;
}



////////////////////// SparseMatConstIterator_ //////////////////////

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_()
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMat_<_Tp>* _m)
    : SparseMatConstIterator(_m)
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMat* _m)
    : SparseMatConstIterator(_m)
{
    CV_Assert( _m->type() == traits::Type<_Tp>::value );
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMatConstIterator_<_Tp>& it)
    : SparseMatConstIterator(it)
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>& SparseMatConstIterator_<_Tp>::operator = (const SparseMatConstIterator_<_Tp>& it)
{
    return reinterpret_cast<SparseMatConstIterator_<_Tp>&>
         (*reinterpret_cast<SparseMatConstIterator*>(this) =
           reinterpret_cast<const SparseMatConstIterator&>(it));
}

template<typename _Tp> inline
const _Tp& SparseMatConstIterator_<_Tp>::operator *() const
{
    return *(const _Tp*)((const void*)this->ptr);
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>& SparseMatConstIterator_<_Tp>::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMatConstIterator_<_Tp>::operator ++(int)
{
    SparseMatConstIterator_<_Tp> it = *this;
    SparseMatConstIterator::operator ++();
    return it;
}



///////////////////////// SparseMatIterator_ ////////////////////////

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_()
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(SparseMat_<_Tp>* _m)
    : SparseMatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(SparseMat* _m)
    : SparseMatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(const SparseMatIterator_<_Tp>& it)
    : SparseMatConstIterator_<_Tp>(it)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>& SparseMatIterator_<_Tp>::operator = (const SparseMatIterator_<_Tp>& it)
{
    return reinterpret_cast<SparseMatIterator_<_Tp>&>
         (*reinterpret_cast<SparseMatConstIterator*>(this) =
           reinterpret_cast<const SparseMatConstIterator&>(it));
}

template<typename _Tp> inline
_Tp& SparseMatIterator_<_Tp>::operator *() const
{
    return *(_Tp*)((void*)this->ptr);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp>& SparseMatIterator_<_Tp>::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMatIterator_<_Tp>::operator ++(int)
{
    SparseMatIterator_<_Tp> it = *this;
    SparseMatConstIterator::operator ++();
    return it;
}



//////////////////////// MatCommaInitializer_ ///////////////////////

template<typename _Tp> inline
MatCommaInitializer_<_Tp>::MatCommaInitializer_(Mat_<_Tp>* _m)
    : it(_m)
{}

template<typename _Tp> template<typename T2> inline
MatCommaInitializer_<_Tp>& MatCommaInitializer_<_Tp>::operator , (T2 v)
{
    CV_DbgAssert( this->it < ((const Mat_<_Tp>*)this->it.m)->end() );
    *this->it = _Tp(v);
    ++this->it;
    return *this;
}

template<typename _Tp> inline
MatCommaInitializer_<_Tp>::operator Mat_<_Tp>() const
{
    CV_DbgAssert( this->it == ((const Mat_<_Tp>*)this->it.m)->end() );
    return Mat_<_Tp>(*this->it.m);
}


template<typename _Tp, typename T2> static inline
MatCommaInitializer_<_Tp> operator << (const Mat_<_Tp>& m, T2 val)
{
    MatCommaInitializer_<_Tp> commaInitializer((Mat_<_Tp>*)&m);
    return (commaInitializer, val);
}



///////////////////////// Matrix Expressions ////////////////////////

inline
Mat& Mat::operator = (const MatExpr& e)
{
    e.op->assign(e, *this);
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const MatExpr& e)
{
    e.op->assign(e, *this, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const MatExpr& e)
{
    e.op->assign(e, *this, traits::Type<_Tp>::value);
    return *this;
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::zeros(int rows, int cols)
{
    return Mat::zeros(rows, cols, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::zeros(Size sz)
{
    return Mat::zeros(sz, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::ones(int rows, int cols)
{
    return Mat::ones(rows, cols, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::ones(Size sz)
{
    return Mat::ones(sz, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::eye(int rows, int cols)
{
    return Mat::eye(rows, cols, traits::Type<_Tp>::value);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::eye(Size sz)
{
    return Mat::eye(sz, traits::Type<_Tp>::value);
}

inline
MatExpr::MatExpr()
    : op(0), flags(0), a(Mat()), b(Mat()), c(Mat()), alpha(0), beta(0), s()
{}

inline
MatExpr::MatExpr(const MatOp* _op, int _flags, const Mat& _a, const Mat& _b,
                 const Mat& _c, double _alpha, double _beta, const Scalar& _s)
    : op(_op), flags(_flags), a(_a), b(_b), c(_c), alpha(_alpha), beta(_beta), s(_s)
{}

inline
MatExpr::operator Mat() const
{
    Mat m;
    op->assign(*this, m);
    return m;
}

template<typename _Tp> inline
MatExpr::operator Mat_<_Tp>() const
{
    Mat_<_Tp> m;
    op->assign(*this, m, traits::Type<_Tp>::value);
    return m;
}


template<typename _Tp> static inline
MatExpr min(const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    return cv::min((const Mat&)a, (const Mat&)b);
}

template<typename _Tp> static inline
MatExpr min(const Mat_<_Tp>& a, double s)
{
    return cv::min((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr min(double s, const Mat_<_Tp>& a)
{
    return cv::min((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr max(const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    return cv::max((const Mat&)a, (const Mat&)b);
}

template<typename _Tp> static inline
MatExpr max(const Mat_<_Tp>& a, double s)
{
    return cv::max((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr max(double s, const Mat_<_Tp>& a)
{
    return cv::max((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr abs(const Mat_<_Tp>& m)
{
    return cv::abs((const Mat&)m);
}


static inline
Mat& operator += (Mat& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, a);
    return a;
}

static inline
const Mat& operator += (const Mat& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator += (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator += (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, (Mat&)a);
    return a;
}

static inline
Mat& operator -= (Mat& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, a);
    return a;
}

static inline
const Mat& operator -= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator -= (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator -= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, (Mat&)a);
    return a;
}

static inline
Mat& operator *= (Mat& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, a);
    return a;
}

static inline
const Mat& operator *= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator *= (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator *= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, (Mat&)a);
    return a;
}

static inline
Mat& operator /= (Mat& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, a);
    return a;
}

static inline
const Mat& operator /= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator /= (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator /= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, (Mat&)a);
    return a;
}


//////////////////////////////// UMat ////////////////////////////////

template<typename _Tp> inline
UMat::UMat(const std::vector<_Tp>& vec, bool copyData)
: flags(+MAGIC_VAL + traits::Type<_Tp>::value + CV_MAT_CONT_FLAG), dims(2), rows((int)vec.size()),
cols(1), allocator(0), usageFlags(USAGE_DEFAULT), u(0), offset(0), size(&rows)
{
    if(vec.empty())
        return;
    if( !copyData )
    {
        // !!!TODO!!!
        CV_Error(Error::StsNotImplemented, "");
    }
    else
        Mat((int)vec.size(), 1, traits::Type<_Tp>::value, (uchar*)&vec[0]).copyTo(*this);
}

inline
UMat UMat::row(int y) const
{
    return UMat(*this, Range(y, y + 1), Range::all());
}

inline
UMat UMat::col(int x) const
{
    return UMat(*this, Range::all(), Range(x, x + 1));
}

inline
UMat UMat::rowRange(int startrow, int endrow) const
{
    return UMat(*this, Range(startrow, endrow), Range::all());
}

inline
UMat UMat::rowRange(const Range& r) const
{
    return UMat(*this, r, Range::all());
}

inline
UMat UMat::colRange(int startcol, int endcol) const
{
    return UMat(*this, Range::all(), Range(startcol, endcol));
}

inline
UMat UMat::colRange(const Range& r) const
{
    return UMat(*this, Range::all(), r);
}

inline
UMat UMat::operator()( Range _rowRange, Range _colRange ) const
{
    return UMat(*this, _rowRange, _colRange);
}

inline
UMat UMat::operator()( const Rect& roi ) const
{
    return UMat(*this, roi);
}

inline
UMat UMat::operator()(const Range* ranges) const
{
    return UMat(*this, ranges);
}

inline
UMat UMat::operator()(const std::vector<Range>& ranges) const
{
    return UMat(*this, ranges);
}

inline
bool UMat::isContinuous() const
{
    return (flags & CONTINUOUS_FLAG) != 0;
}

inline
bool UMat::isSubmatrix() const
{
    return (flags & SUBMATRIX_FLAG) != 0;
}

inline
size_t UMat::elemSize() const
{
    size_t res = dims > 0 ? step.p[dims - 1] : 0;
    CV_DbgAssert(res != 0);
    return res;
}

inline
size_t UMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int UMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int UMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int UMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t UMat::step1(int i) const
{
    return step.p[i] / elemSize1();
}


inline bool UMatData::hostCopyObsolete() const { return (flags & HOST_COPY_OBSOLETE) != 0; }
inline bool UMatData::deviceCopyObsolete() const { return (flags & DEVICE_COPY_OBSOLETE) != 0; }
inline bool UMatData::deviceMemMapped() const { return (flags & DEVICE_MEM_MAPPED) != 0; }
inline bool UMatData::copyOnMap() const { return (flags & COPY_ON_MAP) != 0; }
inline bool UMatData::tempUMat() const { return (flags & TEMP_UMAT) != 0; }
inline bool UMatData::tempCopiedUMat() const { return (flags & TEMP_COPIED_UMAT) == TEMP_COPIED_UMAT; }

inline void UMatData::markDeviceMemMapped(bool flag)
{
  if(flag)
    flags |= DEVICE_MEM_MAPPED;
  else
    flags &= ~DEVICE_MEM_MAPPED;
}

inline void UMatData::markHostCopyObsolete(bool flag)
{
    if(flag)
        flags |= HOST_COPY_OBSOLETE;
    else
        flags &= ~HOST_COPY_OBSOLETE;
}
inline void UMatData::markDeviceCopyObsolete(bool flag)
{
    if(flag)
        flags |= DEVICE_COPY_OBSOLETE;
    else
        flags &= ~DEVICE_COPY_OBSOLETE;
}

//! @endcond

static inline
void swap(MatExpr& a, MatExpr& b) { a.swap(b); }

} //cv

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#ifdef CV_DISABLE_CLANG_ENUM_WARNINGS
#undef CV_DISABLE_CLANG_ENUM_WARNINGS
#pragma clang diagnostic pop
#endif

#endif
