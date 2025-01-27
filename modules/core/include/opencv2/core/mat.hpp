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

#ifndef OPENCV_CORE_MAT_HPP
#define OPENCV_CORE_MAT_HPP

#ifndef __cplusplus
#  error mat.hpp header must be compiled as C++
#endif

#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"

#include "opencv2/core/bufferpool.hpp"

#include <array>
#include <type_traits>

namespace cv
{

//! @addtogroup core_basic
//! @{

enum AccessFlag { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
    ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };
CV_ENUM_FLAGS(AccessFlag)
__CV_ENUM_FLAGS_BITWISE_AND(AccessFlag, int, AccessFlag)

CV__DEBUG_NS_BEGIN

class CV_EXPORTS _OutputArray;

//////////////////////// Input/Output Array Arguments /////////////////////////////////

/** @brief This is the proxy class for passing read-only input arrays into OpenCV functions.

It is defined as:
@code
    typedef const _InputArray& InputArray;
@endcode
where \ref cv::_InputArray is a class that can be constructed from \ref cv::Mat, \ref cv::Mat_<T>,
\ref cv::Matx<T, m, n>, std::vector<T>, std::vector<std::vector<T>>, std::vector<Mat>,
std::vector<Mat_<T>>, \ref cv::UMat, std::vector<UMat> or `double`. It can also be constructed from
a matrix expression.

Since this is mostly implementation-level class, and its interface may change in future versions, we
do not describe it in details. There are a few key things, though, that should be kept in mind:

-   When you see in the reference manual or in OpenCV source code a function that takes
    InputArray, it means that you can actually pass `Mat`, `Matx`, `vector<T>` etc. (see above the
    complete list).
-   Optional input arguments: If some of the input arrays may be empty, pass cv::noArray() (or
    simply cv::Mat() as you probably did before).
-   The class is designed solely for passing parameters. That is, normally you *should not*
    declare class members, local and global variables of this type.
-   If you want to design your own function or a class method that can operate of arrays of
    multiple types, you can use InputArray (or OutputArray) for the respective parameters. Inside
    a function you should use _InputArray::getMat() method to construct a matrix header for the
    array (without copying data). _InputArray::kind() can be used to distinguish Mat from
    `vector<>` etc., but normally it is not needed.

Here is how you can use a function that takes InputArray :
@code
    std::vector<Point2f> vec;
    // points or a circle
    for( int i = 0; i < 30; i++ )
        vec.push_back(Point2f((float)(100 + 30*cos(i*CV_PI*2/5)),
                              (float)(100 - 30*sin(i*CV_PI*2/5))));
    cv::transform(vec, vec, cv::Matx23f(0.707, -0.707, 10, 0.707, 0.707, 20));
@endcode
That is, we form an STL vector containing points, and apply in-place affine transformation to the
vector using the 2x3 matrix created inline as `Matx<float, 2, 3>` instance.

Here is how such a function can be implemented (for simplicity, we implement a very specific case of
it, according to the assertion statement inside) :
@code
    void myAffineTransform(InputArray _src, OutputArray _dst, InputArray _m)
    {
        // get Mat headers for input arrays. This is O(1) operation,
        // unless _src and/or _m are matrix expressions.
        Mat src = _src.getMat(), m = _m.getMat();
        CV_Assert( src.type() == CV_32FC2 && m.type() == CV_32F && m.size() == Size(3, 2) );

        // [re]create the output array so that it has the proper size and type.
        // In case of Mat it calls Mat::create, in case of STL vector it calls vector::resize.
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        for( int i = 0; i < src.rows; i++ )
            for( int j = 0; j < src.cols; j++ )
            {
                Point2f pt = src.at<Point2f>(i, j);
                dst.at<Point2f>(i, j) = Point2f(m.at<float>(0, 0)*pt.x +
                                                m.at<float>(0, 1)*pt.y +
                                                m.at<float>(0, 2),
                                                m.at<float>(1, 0)*pt.x +
                                                m.at<float>(1, 1)*pt.y +
                                                m.at<float>(1, 2));
            }
    }
@endcode
There is another related type, InputArrayOfArrays, which is currently defined as a synonym for
InputArray:
@code
    typedef InputArray InputArrayOfArrays;
@endcode
It denotes function arguments that are either vectors of vectors or vectors of matrices. A separate
synonym is needed to generate Python/Java etc. wrappers properly. At the function implementation
level their use is similar, but _InputArray::getMat(idx) should be used to get header for the
idx-th component of the outer vector and _InputArray::size().area() should be used to find the
number of components (vectors/matrices) of the outer vector.

In general, type support is limited to cv::Mat types. Other types are forbidden.
But in some cases we need to support passing of custom non-general Mat types, like arrays of cv::KeyPoint, cv::DMatch, etc.
This data is not intended to be interpreted as an image data, or processed somehow like regular cv::Mat.
To pass such custom type use rawIn() / rawOut() / rawInOut() wrappers.
Custom type is wrapped as Mat-compatible `CV_8UC<N>` values (N = sizeof(T), N <= CV_CN_MAX).
 */
class CV_EXPORTS _InputArray
{
public:
    enum KindFlag {
        KIND_SHIFT = 16,
        FIXED_TYPE = 0x8000 << KIND_SHIFT,
        FIXED_SIZE = 0x4000 << KIND_SHIFT,
        KIND_MASK = 31 << KIND_SHIFT,

        NONE              = 0 << KIND_SHIFT,
        MAT               = 1 << KIND_SHIFT,
        MATX              = 2 << KIND_SHIFT,
        STD_VECTOR        = 3 << KIND_SHIFT,
        STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
        STD_VECTOR_MAT    = 5 << KIND_SHIFT,
#if OPENCV_ABI_COMPATIBILITY < 500
        EXPR              = 6 << KIND_SHIFT,  //!< removed: https://github.com/opencv/opencv/pull/17046
#endif
        OPENGL_BUFFER     = 7 << KIND_SHIFT,
        CUDA_HOST_MEM     = 8 << KIND_SHIFT,
        CUDA_GPU_MAT      = 9 << KIND_SHIFT,
        UMAT              =10 << KIND_SHIFT,
        STD_VECTOR_UMAT   =11 << KIND_SHIFT,
        STD_BOOL_VECTOR   =12 << KIND_SHIFT,
        STD_VECTOR_CUDA_GPU_MAT = 13 << KIND_SHIFT,
#if OPENCV_ABI_COMPATIBILITY < 500
        STD_ARRAY         =14 << KIND_SHIFT,  //!< removed: https://github.com/opencv/opencv/issues/18897
#endif
        STD_ARRAY_MAT     =15 << KIND_SHIFT
    };

    _InputArray();
    _InputArray(int _flags, void* _obj);
    _InputArray(const Mat& m);
    _InputArray(const MatExpr& expr);
    _InputArray(const std::vector<Mat>& vec);
    template<typename _Tp> _InputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputArray(const std::vector<_Tp>& vec);
    _InputArray(const std::vector<bool>& vec);
    template<typename _Tp> _InputArray(const std::vector<std::vector<_Tp> >& vec);
    _InputArray(const std::vector<std::vector<bool> >&) = delete;  // not supported
    template<typename _Tp> _InputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputArray(const Matx<_Tp, m, n>& matx);
    _InputArray(const double& val);
    _InputArray(const cuda::GpuMat& d_mat);
    _InputArray(const std::vector<cuda::GpuMat>& d_mat_array);
    _InputArray(const ogl::Buffer& buf);
    _InputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputArray(const cudev::GpuMat_<_Tp>& m);
    _InputArray(const UMat& um);
    _InputArray(const std::vector<UMat>& umv);

    template<typename _Tp, std::size_t _Nm> _InputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _InputArray(const std::array<Mat, _Nm>& arr);

    template<typename _Tp> static _InputArray rawIn(const std::vector<_Tp>& vec);
    template<typename _Tp, std::size_t _Nm> static _InputArray rawIn(const std::array<_Tp, _Nm>& arr);

    Mat getMat(int idx=-1) const;
    Mat getMat_(int idx=-1) const;
    UMat getUMat(int idx=-1) const;
    void getMatVector(std::vector<Mat>& mv) const;
    void getUMatVector(std::vector<UMat>& umv) const;
    void getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const;
    cuda::GpuMat getGpuMat() const;
    ogl::Buffer getOGlBuffer() const;

    int getFlags() const;
    void* getObj() const;
    Size getSz() const;

    _InputArray::KindFlag kind() const;
    int dims(int i=-1) const;
    int cols(int i=-1) const;
    int rows(int i=-1) const;
    Size size(int i=-1) const;
    int sizend(int* sz, int i=-1) const;
    bool sameSize(const _InputArray& arr) const;
    size_t total(int i=-1) const;
    int type(int i=-1) const;
    int depth(int i=-1) const;
    int channels(int i=-1) const;
    bool isContinuous(int i=-1) const;
    bool isSubmatrix(int i=-1) const;
    bool empty() const;
    void copyTo(const _OutputArray& arr) const;
    void copyTo(const _OutputArray& arr, const _InputArray & mask) const;
    size_t offset(int i=-1) const;
    size_t step(int i=-1) const;
    bool isMat() const;
    bool isUMat() const;
    bool isMatVector() const;
    bool isUMatVector() const;
    bool isMatx() const;
    bool isVector() const;
    bool isGpuMat() const;
    bool isGpuMatVector() const;
    ~_InputArray();

protected:
    int flags;
    void* obj;
    Size sz;

    void init(int _flags, const void* _obj);
    void init(int _flags, const void* _obj, Size _sz);
};
CV_ENUM_FLAGS(_InputArray::KindFlag)
__CV_ENUM_FLAGS_BITWISE_AND(_InputArray::KindFlag, int, _InputArray::KindFlag)

/** @brief This type is very similar to InputArray except that it is used for input/output and output function
parameters.

Just like with InputArray, OpenCV users should not care about OutputArray, they just pass `Mat`,
`vector<T>` etc. to the functions. The same limitation as for `InputArray`: *Do not explicitly
create OutputArray instances* applies here too.

If you want to make your function polymorphic (i.e. accept different arrays as output parameters),
it is also not very difficult. Take the sample above as the reference. Note that
_OutputArray::create() needs to be called before _OutputArray::getMat(). This way you guarantee
that the output array is properly allocated.

Optional output parameters. If you do not need certain output array to be computed and returned to
you, pass cv::noArray(), just like you would in the case of optional input array. At the
implementation level, use _OutputArray::needed() to check if certain output array needs to be
computed or not.

There are several synonyms for OutputArray that are used to assist automatic Python/Java/... wrapper
generators:
@code
    typedef OutputArray OutputArrayOfArrays;
    typedef OutputArray InputOutputArray;
    typedef OutputArray InputOutputArrayOfArrays;
@endcode
 */
class CV_EXPORTS _OutputArray : public _InputArray
{
public:
    enum DepthMask
    {
        DEPTH_MASK_8U = 1 << CV_8U,
        DEPTH_MASK_8S = 1 << CV_8S,
        DEPTH_MASK_16U = 1 << CV_16U,
        DEPTH_MASK_16S = 1 << CV_16S,
        DEPTH_MASK_32S = 1 << CV_32S,
        DEPTH_MASK_32F = 1 << CV_32F,
        DEPTH_MASK_64F = 1 << CV_64F,
        DEPTH_MASK_16F = 1 << CV_16F,
        DEPTH_MASK_ALL = (DEPTH_MASK_64F<<1)-1,
        DEPTH_MASK_ALL_BUT_8S = DEPTH_MASK_ALL & ~DEPTH_MASK_8S,
        DEPTH_MASK_ALL_16F = (DEPTH_MASK_16F<<1)-1,
        DEPTH_MASK_FLT = DEPTH_MASK_32F + DEPTH_MASK_64F
    };

    _OutputArray();
    _OutputArray(int _flags, void* _obj);
    _OutputArray(Mat& m);
    _OutputArray(std::vector<Mat>& vec);
    _OutputArray(cuda::GpuMat& d_mat);
    _OutputArray(std::vector<cuda::GpuMat>& d_mat);
    _OutputArray(ogl::Buffer& buf);
    _OutputArray(cuda::HostMem& cuda_mem);
    template<typename _Tp> _OutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(std::vector<_Tp>& vec);
    _OutputArray(std::vector<bool>& vec) = delete;  // not supported
    template<typename _Tp> _OutputArray(std::vector<std::vector<_Tp> >& vec);
    _OutputArray(std::vector<std::vector<bool> >&) = delete;  // not supported
    template<typename _Tp> _OutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(Matx<_Tp, m, n>& matx);
    _OutputArray(UMat& m);
    _OutputArray(std::vector<UMat>& vec);

    _OutputArray(const Mat& m);
    _OutputArray(const std::vector<Mat>& vec);
    _OutputArray(const cuda::GpuMat& d_mat);
    _OutputArray(const std::vector<cuda::GpuMat>& d_mat);
    _OutputArray(const ogl::Buffer& buf);
    _OutputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _OutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _OutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(const Matx<_Tp, m, n>& matx);
    _OutputArray(const UMat& m);
    _OutputArray(const std::vector<UMat>& vec);

    template<typename _Tp, std::size_t _Nm> _OutputArray(std::array<_Tp, _Nm>& arr);
    template<typename _Tp, std::size_t _Nm> _OutputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _OutputArray(std::array<Mat, _Nm>& arr);
    template<std::size_t _Nm> _OutputArray(const std::array<Mat, _Nm>& arr);

    template<typename _Tp> static _OutputArray rawOut(std::vector<_Tp>& vec);
    template<typename _Tp, std::size_t _Nm> static _OutputArray rawOut(std::array<_Tp, _Nm>& arr);

    bool fixedSize() const;
    bool fixedType() const;
    bool needed() const;
    Mat& getMatRef(int i=-1) const;
    UMat& getUMatRef(int i=-1) const;
    cuda::GpuMat& getGpuMatRef() const;
    std::vector<cuda::GpuMat>& getGpuMatVecRef() const;
    ogl::Buffer& getOGlBufferRef() const;
    cuda::HostMem& getHostMemRef() const;
    void create(Size sz, int type, int i=-1, bool allowTransposed=false, _OutputArray::DepthMask fixedDepthMask=static_cast<_OutputArray::DepthMask>(0)) const;
    void create(int rows, int cols, int type, int i=-1, bool allowTransposed=false, _OutputArray::DepthMask fixedDepthMask=static_cast<_OutputArray::DepthMask>(0)) const;
    void create(int dims, const int* size, int type, int i=-1, bool allowTransposed=false, _OutputArray::DepthMask fixedDepthMask=static_cast<_OutputArray::DepthMask>(0)) const;
    void createSameSize(const _InputArray& arr, int mtype) const;
    void release() const;
    void clear() const;
    void setTo(const _InputArray& value, const _InputArray & mask = _InputArray()) const;

    void assign(const UMat& u) const;
    void assign(const Mat& m) const;

    void assign(const std::vector<UMat>& v) const;
    void assign(const std::vector<Mat>& v) const;

    void move(UMat& u) const;
    void move(Mat& m) const;
};


class CV_EXPORTS _InputOutputArray : public _OutputArray
{
public:
    _InputOutputArray();
    _InputOutputArray(int _flags, void* _obj);
    _InputOutputArray(Mat& m);
    _InputOutputArray(std::vector<Mat>& vec);
    _InputOutputArray(cuda::GpuMat& d_mat);
    _InputOutputArray(ogl::Buffer& buf);
    _InputOutputArray(cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(std::vector<_Tp>& vec);
    _InputOutputArray(std::vector<bool>& vec) = delete;  // not supported
    template<typename _Tp> _InputOutputArray(std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(Matx<_Tp, m, n>& matx);
    _InputOutputArray(UMat& m);
    _InputOutputArray(std::vector<UMat>& vec);

    _InputOutputArray(const Mat& m);
    _InputOutputArray(const std::vector<Mat>& vec);
    _InputOutputArray(const cuda::GpuMat& d_mat);
    _InputOutputArray(const std::vector<cuda::GpuMat>& d_mat);
    _InputOutputArray(const ogl::Buffer& buf);
    _InputOutputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(const Matx<_Tp, m, n>& matx);
    _InputOutputArray(const UMat& m);
    _InputOutputArray(const std::vector<UMat>& vec);

    template<typename _Tp, std::size_t _Nm> _InputOutputArray(std::array<_Tp, _Nm>& arr);
    template<typename _Tp, std::size_t _Nm> _InputOutputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _InputOutputArray(std::array<Mat, _Nm>& arr);
    template<std::size_t _Nm> _InputOutputArray(const std::array<Mat, _Nm>& arr);

    template<typename _Tp> static _InputOutputArray rawInOut(std::vector<_Tp>& vec);
    template<typename _Tp, std::size_t _Nm> _InputOutputArray rawInOut(std::array<_Tp, _Nm>& arr);

};

/** Helper to wrap custom types. @see InputArray */
template<typename _Tp> static inline _InputArray rawIn(_Tp& v);
/** Helper to wrap custom types. @see InputArray */
template<typename _Tp> static inline _OutputArray rawOut(_Tp& v);
/** Helper to wrap custom types. @see InputArray */
template<typename _Tp> static inline _InputOutputArray rawInOut(_Tp& v);

CV__DEBUG_NS_END

typedef const _InputArray& InputArray;
typedef InputArray InputArrayOfArrays;
typedef const _OutputArray& OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray& InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

/** @brief Returns an empty InputArray or OutputArray.

 This function is used to provide an "empty" or "null" array when certain functions
 take optional input or output arrays that you don't want to provide.

 Many OpenCV functions accept optional arguments as `cv::InputArray` or `cv::OutputArray`.
 When you don't want to pass any data for these optional parameters, you can use `cv::noArray()`
 to indicate that you are omitting them.

 @return An empty `cv::InputArray` or `cv::OutputArray` that can be used as a placeholder.

 @note This is often used when a function has optional arrays, and you do not want to
 provide a specific input or output array.

 @see cv::InputArray, cv::OutputArray
 */
CV_EXPORTS InputOutputArray noArray();

/////////////////////////////////// MatAllocator //////////////////////////////////////

/** @brief  Usage flags for allocator

 @warning  All flags except `USAGE_DEFAULT` are experimental.

 @warning  For the OpenCL allocator, `USAGE_ALLOCATE_SHARED_MEMORY` depends on
 OpenCV's optional, experimental integration with OpenCL SVM. To enable this
 integration, build OpenCV using the `WITH_OPENCL_SVM=ON` CMake option and, at
 runtime, call `cv::ocl::Context::getDefault().setUseSVM(true);` or similar
 code. Note that SVM is incompatible with OpenCL 1.x.
*/
enum UMatUsageFlags
{
    USAGE_DEFAULT = 0,

    // buffer allocation policy is platform and usage specific
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
    USAGE_ALLOCATE_SHARED_MEMORY = 1 << 2, // It is not equal to: USAGE_ALLOCATE_HOST_MEMORY | USAGE_ALLOCATE_DEVICE_MEMORY

    __UMAT_USAGE_FLAGS_32BIT = 0x7fffffff // Binary compatibility hint
};

struct CV_EXPORTS UMatData;

/** @brief  Custom array allocator
*/
class CV_EXPORTS MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    // let's comment it off for now to detect and fix all the uses of allocator
    //virtual void allocate(int dims, const int* sizes, int type, int*& refcount,
    //                      uchar*& datastart, uchar*& data, size_t* step) = 0;
    //virtual void deallocate(int* refcount, uchar* datastart, uchar* data) = 0;
    virtual UMatData* allocate(int dims, const int* sizes, int type,
                               void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const = 0;
    virtual bool allocate(UMatData* data, AccessFlag accessflags, UMatUsageFlags usageFlags) const = 0;
    virtual void deallocate(UMatData* data) const = 0;
    virtual void map(UMatData* data, AccessFlag accessflags) const;
    virtual void unmap(UMatData* data) const;
    virtual void download(UMatData* data, void* dst, int dims, const size_t sz[],
                          const size_t srcofs[], const size_t srcstep[],
                          const size_t dststep[]) const;
    virtual void upload(UMatData* data, const void* src, int dims, const size_t sz[],
                        const size_t dstofs[], const size_t dststep[],
                        const size_t srcstep[]) const;
    virtual void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
                      const size_t srcofs[], const size_t srcstep[],
                      const size_t dstofs[], const size_t dststep[], bool sync) const;

    // default implementation returns DummyBufferPoolController
    virtual BufferPoolController* getBufferPoolController(const char* id = NULL) const;
};


//////////////////////////////// MatCommaInitializer //////////////////////////////////

/** @brief  Comma-separated Matrix Initializer

 The class instances are usually not created explicitly.
 Instead, they are created on "matrix << firstValue" operator.

 The sample below initializes 2x2 rotation matrix:

 \code
 double angle = 30, a = cos(angle*CV_PI/180), b = sin(angle*CV_PI/180);
 Mat R = (Mat_<double>(2,2) << a, -b, b, a);
 \endcode
*/
template<typename _Tp> class MatCommaInitializer_
{
public:
    //! the constructor, created by "matrix << firstValue" operator, where matrix is cv::Mat
    MatCommaInitializer_(Mat_<_Tp>* _m);
    //! the operator that takes the next value and put it to the matrix
    template<typename T2> MatCommaInitializer_<_Tp>& operator , (T2 v);
    //! another form of conversion operator
    operator Mat_<_Tp>() const;
protected:
    MatIterator_<_Tp> it;
};


/////////////////////////////////////// Mat ///////////////////////////////////////////

// note that umatdata might be allocated together
// with the matrix data, not as a separate object.
// therefore, it does not have constructor or destructor;
// it should be explicitly initialized using init().
struct CV_EXPORTS UMatData
{
    enum MemoryFlag { COPY_ON_MAP=1, HOST_COPY_OBSOLETE=2,
        DEVICE_COPY_OBSOLETE=4, TEMP_UMAT=8, TEMP_COPIED_UMAT=24,
        USER_ALLOCATED=32, DEVICE_MEM_MAPPED=64,
        ASYNC_CLEANUP=128
    };
    UMatData(const MatAllocator* allocator);
    ~UMatData();

    // provide atomic access to the structure
    void lock();
    void unlock();

    bool hostCopyObsolete() const;
    bool deviceCopyObsolete() const;
    bool deviceMemMapped() const;
    bool copyOnMap() const;
    bool tempUMat() const;
    bool tempCopiedUMat() const;
    void markHostCopyObsolete(bool flag);
    void markDeviceCopyObsolete(bool flag);
    void markDeviceMemMapped(bool flag);

    const MatAllocator* prevAllocator;
    const MatAllocator* currAllocator;
    int urefcount;
    int refcount;
    uchar* data;
    uchar* origdata;
    size_t size;

    UMatData::MemoryFlag flags;
    void* handle;
    void* userdata;
    int allocatorFlags_;
    int mapcount;
    UMatData* originalUMatData;
    std::shared_ptr<void> allocatorContext;
};
CV_ENUM_FLAGS(UMatData::MemoryFlag)


struct CV_EXPORTS MatSize
{
    explicit MatSize(int* _p) CV_NOEXCEPT;
    int dims() const CV_NOEXCEPT;
    Size operator()() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    operator const int*() const CV_NOEXCEPT;  // TODO OpenCV 4.0: drop this
    bool operator == (const MatSize& sz) const CV_NOEXCEPT;
    bool operator != (const MatSize& sz) const CV_NOEXCEPT;

    int* p;
};

struct CV_EXPORTS MatStep
{
    MatStep() CV_NOEXCEPT;
    explicit MatStep(size_t s) CV_NOEXCEPT;
    const size_t& operator[](int i) const CV_NOEXCEPT;
    size_t& operator[](int i) CV_NOEXCEPT;
    operator size_t() const;
    MatStep& operator = (size_t s);

    size_t* p;
    size_t buf[2];
protected:
    MatStep& operator = (const MatStep&);
};

/** @example samples/cpp/cout_mat.cpp
An example demonstrating the serial out capabilities of cv::Mat
*/

 /** @brief n-dimensional dense array class \anchor CVMat_Details

The class Mat represents an n-dimensional dense numerical single-channel or multi-channel array. It
can be used to store real or complex-valued vectors and matrices, grayscale or color images, voxel
volumes, vector fields, point clouds, tensors, histograms (though, very high-dimensional histograms
may be better stored in a SparseMat ). The data layout of the array `M` is defined by the array
`M.step[]`, so that the address of element \f$(i_0,...,i_{M.dims-1})\f$, where \f$0\leq i_k<M.size[k]\f$, is
computed as:
\f[addr(M_{i_0,...,i_{M.dims-1}}) = M.data + M.step[0]*i_0 + M.step[1]*i_1 + ... + M.step[M.dims-1]*i_{M.dims-1}\f]
In case of a 2-dimensional array, the above formula is reduced to:
\f[addr(M_{i,j}) = M.data + M.step[0]*i + M.step[1]*j\f]
Note that `M.step[i] >= M.step[i+1]` (in fact, `M.step[i] >= M.step[i+1]*M.size[i+1]` ). This means
that 2-dimensional matrices are stored row-by-row, 3-dimensional matrices are stored plane-by-plane,
and so on. M.step[M.dims-1] is minimal and always equal to the element size M.elemSize() .

So, the data layout in Mat is compatible with the majority of dense array types from the standard
toolkits and SDKs, such as Numpy (ndarray), Win32 (independent device bitmaps), and others,
that is, with any array that uses *steps* (or *strides*) to compute the position of a pixel.
Due to this compatibility, it is possible to make a Mat header for user-allocated data and process
it in-place using OpenCV functions.

There are many different ways to create a Mat object. The most popular options are listed below:

- Use the create(nrows, ncols, type) method or the similar Mat(nrows, ncols, type[, fillValue])
constructor. A new array of the specified size and type is allocated. type has the same meaning as
in the cvCreateMat method. For example, CV_8UC1 means a 8-bit single-channel array, CV_32FC2
means a 2-channel (complex) floating-point array, and so on.
@code
    // make a 7x7 complex matrix filled with 1+3j.
    Mat M(7,7,CV_32FC2,Scalar(1,3));
    // and now turn M to a 100x60 15-channel 8-bit matrix.
    // The old content will be deallocated
    M.create(100,60,CV_8UC(15));
@endcode
As noted in the introduction to this chapter, create() allocates only a new array when the shape
or type of the current array are different from the specified ones.

- Create a multi-dimensional array:
@code
    // create a 100x100x100 8-bit array
    int sz[] = {100, 100, 100};
    Mat bigCube(3, sz, CV_8U, Scalar::all(0));
@endcode
It passes the number of dimensions =1 to the Mat constructor but the created array will be
2-dimensional with the number of columns set to 1. So, Mat::dims is always \>= 2 (can also be 0
when the array is empty).

- Use a copy constructor or assignment operator where there can be an array or expression on the
right side (see below). As noted in the introduction, the array assignment is an O(1) operation
because it only copies the header and increases the reference counter. The Mat::clone() method can
be used to get a full (deep) copy of the array when you need it.

- Construct a header for a part of another array. It can be a single row, single column, several
rows, several columns, rectangular region in the array (called a *minor* in algebra) or a
diagonal. Such operations are also O(1) because the new header references the same data. You can
actually modify a part of the array using this feature, for example:
@code
    // add the 5-th row, multiplied by 3 to the 3rd row
    M.row(3) = M.row(3) + M.row(5)*3;
    // now copy the 7-th column to the 1-st column
    // M.col(1) = M.col(7); // this will not work
    Mat M1 = M.col(1);
    M.col(7).copyTo(M1);
    // create a new 320x240 image
    Mat img(Size(320,240),CV_8UC3);
    // select a ROI
    Mat roi(img, Rect(10,10,100,100));
    // fill the ROI with (0,255,0) (which is green in RGB space);
    // the original 320x240 image will be modified
    roi = Scalar(0,255,0);
@endcode
Due to the additional datastart and dataend members, it is possible to compute a relative
sub-array position in the main *container* array using locateROI():
@code
    Mat A = Mat::eye(10, 10, CV_32S);
    // extracts A columns, 1 (inclusive) to 3 (exclusive).
    Mat B = A(Range::all(), Range(1, 3));
    // extracts B rows, 5 (inclusive) to 9 (exclusive).
    // that is, C \~ A(Range(5, 9), Range(1, 3))
    Mat C = B(Range(5, 9), Range::all());
    Size size; Point ofs;
    C.locateROI(size, ofs);
    // size will be (width=10,height=10) and the ofs will be (x=1, y=5)
@endcode
As in case of whole matrices, if you need a deep copy, use the `clone()` method of the extracted
sub-matrices.

- Make a header for user-allocated data. It can be useful to do the following:
    -# Process "foreign" data using OpenCV (for example, when you implement a DirectShow\* filter or
    a processing module for gstreamer, and so on). For example:
    @code
        Mat process_video_frame(const unsigned char* pixels,
                                int width, int height, int step)
        {
            // wrap input buffer
            Mat img(height, width, CV_8UC3, (unsigned char*)pixels, step);

            Mat result;
            GaussianBlur(img, result, Size(7, 7), 1.5, 1.5);

            return result;
        }
    @endcode
    -# Quickly initialize small matrices and/or get a super-fast element access.
    @code
        double m[3][3] = {{a, b, c}, {d, e, f}, {g, h, i}};
        Mat M = Mat(3, 3, CV_64F, m).inv();
    @endcode
    .

- Use MATLAB-style array initializers, zeros(), ones(), eye(), for example:
@code
    // create a double-precision identity matrix and add it to M.
    M += Mat::eye(M.rows, M.cols, CV_64F);
@endcode

- Use a comma-separated initializer:
@code
    // create a 3x3 double-precision identity matrix
    Mat M = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
@endcode
With this approach, you first call a constructor of the Mat class with the proper parameters, and
then you just put `<< operator` followed by comma-separated values that can be constants,
variables, expressions, and so on. Also, note the extra parentheses required to avoid compilation
errors.

Once the array is created, it is automatically managed via a reference-counting mechanism. If the
array header is built on top of user-allocated data, you should handle the data by yourself. The
array data is deallocated when no one points to it. If you want to release the data pointed by a
array header before the array destructor is called, use Mat::release().

The next important thing to learn about the array class is element access. This manual already
described how to compute an address of each array element. Normally, you are not required to use the
formula directly in the code. If you know the array element type (which can be retrieved using the
method Mat::type() ), you can access the element \f$M_{ij}\f$ of a 2-dimensional array as:
@code
    M.at<double>(i,j) += 1.f;
@endcode
assuming that `M` is a double-precision floating-point array. There are several variants of the method
at for a different number of dimensions.

If you need to process a whole row of a 2D array, the most efficient way is to get the pointer to
the row first, and then just use the plain C operator [] :
@code
    // compute sum of positive matrix elements
    // (assuming that M is a double-precision matrix)
    double sum=0;
    for(int i = 0; i < M.rows; i++)
    {
        const double* Mi = M.ptr<double>(i);
        for(int j = 0; j < M.cols; j++)
            sum += std::max(Mi[j], 0.);
    }
@endcode
Some operations, like the one above, do not actually depend on the array shape. They just process
elements of an array one by one (or elements from multiple arrays that have the same coordinates,
for example, array addition). Such operations are called *element-wise*. It makes sense to check
whether all the input/output arrays are continuous, namely, have no gaps at the end of each row. If
yes, process them as a long single row:
@code
    // compute the sum of positive matrix elements, optimized variant
    double sum=0;
    int cols = M.cols, rows = M.rows;
    if(M.isContinuous())
    {
        cols *= rows;
        rows = 1;
    }
    for(int i = 0; i < rows; i++)
    {
        const double* Mi = M.ptr<double>(i);
        for(int j = 0; j < cols; j++)
            sum += std::max(Mi[j], 0.);
    }
@endcode
In case of the continuous matrix, the outer loop body is executed just once. So, the overhead is
smaller, which is especially noticeable in case of small matrices.

Finally, there are STL-style iterators that are smart enough to skip gaps between successive rows:
@code
    // compute sum of positive matrix elements, iterator-based variant
    double sum=0;
    MatConstIterator_<double> it = M.begin<double>(), it_end = M.end<double>();
    for(; it != it_end; ++it)
        sum += std::max(*it, 0.);
@endcode
The matrix iterators are random-access iterators, so they can be passed to any STL algorithm,
including std::sort().

@note Matrix Expressions and arithmetic see MatExpr
*/
class CV_EXPORTS Mat
{
public:
    /**
    These are various constructors that form a matrix. As noted in the AutomaticAllocation, often
    the default constructor is enough, and the proper matrix will be allocated by an OpenCV function.
    The constructed matrix can further be assigned to another matrix or matrix expression or can be
    allocated with Mat::create . In the former case, the old content is de-referenced.
     */
    Mat() CV_NOEXCEPT;

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(int rows, int cols, int type);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
      */
    Mat(Size size, int type);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int rows, int cols, int type, const Scalar& s);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
      */
    Mat(Size size, int type, const Scalar& s);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(int ndims, const int* sizes, int type);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(const std::vector<int>& sizes, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int ndims, const int* sizes, int type, const Scalar& s);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(const std::vector<int>& sizes, int type, const Scalar& s);


    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    */
    Mat(const Mat& m);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param step Number of bytes each matrix row occupies. The value should include the padding bytes at
    the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    */
    Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param step Number of bytes each matrix row occupies. The value should include the padding bytes at
    the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    */
    Mat(Size size, int type, void* data, size_t step=AUTO_STEP);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
    set to the element size). If not specified, the matrix is assumed to be continuous.
    */
    Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
    set to the element size). If not specified, the matrix is assumed to be continuous.
    */
    Mat(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param rowRange Range of the m rows to take. As usual, the range start is inclusive and the range
    end is exclusive. Use Range::all() to take all the rows.
    @param colRange Range of the m columns to take. Use Range::all() to take all the columns.
    */
    Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param roi Region of interest.
    */
    Mat(const Mat& m, const Rect& roi);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param ranges Array of selected ranges of m along each dimensionality.
    */
    Mat(const Mat& m, const Range* ranges);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param ranges Array of selected ranges of m along each dimensionality.
    */
    Mat(const Mat& m, const std::vector<Range>& ranges);

    /** @overload
    @param vec STL vector whose elements form the matrix. The matrix has a single column and the number
    of rows equal to the number of vector elements. Type of the matrix matches the type of vector
    elements. The constructor can handle arbitrary types, for which there is a properly declared
    DataType . This means that the vector elements must be primitive numbers or uni-type numerical
    tuples of numbers. Mixed-type structures are not supported. The corresponding constructor is
    explicit. Since STL vectors are not automatically converted to Mat instances, you should write
    Mat(vec) explicitly. Unless you copy the data into the matrix ( copyData=true ), no new elements
    will be added to the vector because it can potentially yield vector data reallocation, and, thus,
    the matrix data pointer will be invalid.
    @param copyData Flag to specify whether the underlying data of the STL vector should be copied
    to (true) or shared with (false) the newly constructed matrix. When the data is copied, the
    allocated buffer is managed using Mat reference counting mechanism. While the data is shared,
    the reference counter is NULL, and you should not deallocate the data until the matrix is
    destructed.
    */
    template<typename _Tp> explicit Mat(const std::vector<_Tp>& vec, bool copyData=false);

    /** @overload
    */
    template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    explicit Mat(const std::initializer_list<_Tp> list);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const std::initializer_list<int> sizes, const std::initializer_list<_Tp> list);

    /** @overload
    */
    template<typename _Tp, size_t _Nm> explicit Mat(const std::array<_Tp, _Nm>& arr, bool copyData=false);

    /** @overload
    */
    template<typename _Tp, int n> explicit Mat(const Vec<_Tp, n>& vec, bool copyData=true);

    /** @overload
    */
    template<typename _Tp, int m, int n> explicit Mat(const Matx<_Tp, m, n>& mtx, bool copyData=true);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const Point_<_Tp>& pt, bool copyData=true);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const Point3_<_Tp>& pt, bool copyData=true);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const MatCommaInitializer_<_Tp>& commaInitializer);

    //! download data from GpuMat
    explicit Mat(const cuda::GpuMat& m);

    //! destructor - calls release()
    ~Mat();

    /** @brief assignment operators

    These are available assignment operators. Since they all are very different, make sure to read the
    operator parameters description.
    @param m Assigned, right-hand-side matrix. Matrix assignment is an O(1) operation. This means that
    no data is copied but the data is shared and the reference counter, if any, is incremented. Before
    assigning new data, the old data is de-referenced via Mat::release .
     */
    Mat& operator = (const Mat& m);

    /** @overload
    @param expr Assigned matrix expression object. As opposite to the first form of the assignment
    operation, the second form can reuse already allocated matrix if it has the right size and type to
    fit the matrix expression result. It is automatically handled by the real function that the matrix
    expressions is expanded to. For example, C=A+B is expanded to add(A, B, C), and add takes care of
    automatic C reallocation.
    */
    Mat& operator = (const MatExpr& expr);

    //! retrieve UMat from Mat
    UMat getUMat(AccessFlag accessFlags, UMatUsageFlags usageFlags = USAGE_DEFAULT) const;

    /** @brief Creates a matrix header for the specified matrix row.

    The method makes a new header for the specified matrix row and returns it. This is an O(1)
    operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
    original matrix. Here is the example of one of the classical basic matrix processing operations,
    axpy, used by LU and many other algorithms:
    @code
        inline void matrix_axpy(Mat& A, int i, int j, double alpha)
        {
            A.row(i) += A.row(j)*alpha;
        }
    @endcode
    @note In the current implementation, the following code does not work as expected:
    @code
        Mat A;
        ...
        A.row(i) = A.row(j); // will not work
    @endcode
    This happens because A.row(i) forms a temporary header that is further assigned to another header.
    Remember that each of these operations is O(1), that is, no data is copied. Thus, the above
    assignment is not true if you may have expected the j-th row to be copied to the i-th row. To
    achieve that, you should either turn this simple assignment into an expression or use the
    Mat::copyTo method:
    @code
        Mat A;
        ...
        // works, but looks a bit obscure.
        A.row(i) = A.row(j) + 0;
        // this is a bit longer, but the recommended method.
        A.row(j).copyTo(A.row(i));
    @endcode
    @param y A 0-based row index.
     */
    Mat row(int y) const;

    /** @brief Creates a matrix header for the specified matrix column.

    The method makes a new header for the specified matrix column and returns it. This is an O(1)
    operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
    original matrix. See also the Mat::row description.
    @param x A 0-based column index.
     */
    Mat col(int x) const;

    /** @brief Creates a matrix header for the specified row span.

    The method makes a new header for the specified row span of the matrix. Similarly to Mat::row and
    Mat::col , this is an O(1) operation.
    @param startrow An inclusive 0-based start index of the row span.
    @param endrow An exclusive 0-based ending index of the row span.
     */
    Mat rowRange(int startrow, int endrow) const;

    /** @overload
    @param r Range structure containing both the start and the end indices.
    */
    Mat rowRange(const Range& r) const;

    /** @brief Creates a matrix header for the specified column span.

    The method makes a new header for the specified column span of the matrix. Similarly to Mat::row and
    Mat::col , this is an O(1) operation.
    @param startcol An inclusive 0-based start index of the column span.
    @param endcol An exclusive 0-based ending index of the column span.
     */
    Mat colRange(int startcol, int endcol) const;

    /** @overload
    @param r Range structure containing both the start and the end indices.
    */
    Mat colRange(const Range& r) const;

    /** @brief Extracts a diagonal from a matrix

    The method makes a new header for the specified matrix diagonal. The new matrix is represented as a
    single-column matrix. Similarly to Mat::row and Mat::col, this is an O(1) operation.
    @param d index of the diagonal, with the following values:
    - `d=0` is the main diagonal.
    - `d<0` is a diagonal from the lower half. For example, d=-1 means the diagonal is set
      immediately below the main one.
    - `d>0` is a diagonal from the upper half. For example, d=1 means the diagonal is set
      immediately above the main one.
    For example:
    @code
        Mat m = (Mat_<int>(3,3) <<
                    1,2,3,
                    4,5,6,
                    7,8,9);
        Mat d0 = m.diag(0);
        Mat d1 = m.diag(1);
        Mat d_1 = m.diag(-1);
    @endcode
    The resulting matrices are
    @code
     d0 =
       [1;
        5;
        9]
     d1 =
       [2;
        6]
     d_1 =
       [4;
        8]
    @endcode
     */
    Mat diag(int d=0) const;

    /** @brief creates a diagonal matrix

    The method creates a square diagonal matrix from specified main diagonal.
    @param d One-dimensional matrix that represents the main diagonal.
     */
    CV_NODISCARD_STD static Mat diag(const Mat& d);

    /** @brief Creates a full copy of the array and the underlying data.

    The method creates a full copy of the array. The original step[] is not taken into account. So, the
    array copy is a continuous array occupying total()*elemSize() bytes.
     */
    CV_NODISCARD_STD Mat clone() const;

    /** @brief Copies the matrix to another one.

    The method copies the matrix data to another matrix. Before copying the data, the method invokes :
    @code
        m.create(this->size(), this->type());
    @endcode
    so that the destination matrix is reallocated if needed. While m.copyTo(m); works flawlessly, the
    function does not handle the case of a partial overlap between the source and the destination
    matrices.

    When the operation mask is specified, if the Mat::create call shown above reallocates the matrix,
    the newly allocated matrix is initialized with all zeros before copying the data.
    @param m Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
     */
    void copyTo( OutputArray m ) const;

    /** @overload
    @param m Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
    @param mask Operation mask of the same size as \*this. Its non-zero elements indicate which matrix
    elements need to be copied. The mask has to be of type CV_8U and can have 1 or multiple channels.
    */
    void copyTo( OutputArray m, InputArray mask ) const;

    /** @brief Converts an array to another data type with optional scaling.

    The method converts source pixel values to the target data type. saturate_cast\<\> is applied at
    the end to avoid possible overflows:

    \f[m(x,y) = saturate \_ cast<rType>( \alpha (*this)(x,y) +  \beta )\f]
    @param m output matrix; if it does not have a proper size or type before the operation, it is
    reallocated.
    @param rtype desired output matrix type or, rather, the depth since the number of channels are the
    same as the input has; if rtype is negative, the output matrix will have the same type as the input.
    @param alpha optional scale factor.
    @param beta optional delta added to the scaled values.
     */
    void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;

    /** @brief Provides a functional form of convertTo.

    This is an internally used method called by the @ref MatrixExpressions engine.
    @param m Destination array.
    @param type Desired destination array depth (or -1 if it should be the same as the source type).
     */
    void assignTo( Mat& m, int type=-1 ) const;

    /** @brief Sets all or some of the array elements to the specified value.
    @param s Assigned scalar converted to the actual array type.
    */
    Mat& operator = (const Scalar& s);

    /** @brief Sets all or some of the array elements to the specified value.

    This is an advanced variant of the Mat::operator=(const Scalar& s) operator.
    @param value Assigned scalar converted to the actual array type.
    @param mask Operation mask of the same size as \*this. Its non-zero elements indicate which matrix
    elements need to be copied. The mask has to be of type CV_8U and can have 1 or multiple channels
     */
    Mat& setTo(InputArray value, InputArray mask=noArray());

    /** @brief Changes the shape and/or the number of channels of a 2D matrix without copying the data.

    The method makes a new matrix header for \*this elements. The new matrix may have a different size
    and/or different number of channels. Any combination is possible if:
    -   No extra elements are included into the new matrix and no elements are excluded. Consequently,
        the product rows\*cols\*channels() must stay the same after the transformation.
    -   No data is copied. That is, this is an O(1) operation. Consequently, if you change the number of
        rows, or the operation changes the indices of elements row in some other way, the matrix must be
        continuous. See Mat::isContinuous .

    For example, if there is a set of 3D points stored as an STL vector, and you want to represent the
    points as a 3xN matrix, do the following:
    @code
        std::vector<Point3f> vec;
        ...
        Mat pointMat = Mat(vec). // convert vector to Mat, O(1) operation
                          reshape(1). // make Nx3 1-channel matrix out of Nx1 3-channel.
                                      // Also, an O(1) operation
                             t(); // finally, transpose the Nx3 matrix.
                                  // This involves copying all the elements
    @endcode
    3-channel 2x2 matrix reshaped to 1-channel 4x3 matrix, each column has values from one of original channels:
    @code
    Mat m(Size(2, 2), CV_8UC3, Scalar(1, 2, 3));
    vector<int> new_shape {4, 3};
    m = m.reshape(1, new_shape);
    @endcode
    or:
    @code
    Mat m(Size(2, 2), CV_8UC3, Scalar(1, 2, 3));
    const int new_shape[] = {4, 3};
    m = m.reshape(1, 2, new_shape);
    @endcode
    @param cn New number of channels. If the parameter is 0, the number of channels remains the same.
    @param rows New number of rows. If the parameter is 0, the number of rows remains the same.
     */
    Mat reshape(int cn, int rows=0) const;

    /** @overload
     * @param cn New number of channels. If the parameter is 0, the number of channels remains the same.
     * @param newndims New number of dimentions.
     * @param newsz Array with new matrix size by all dimentions. If some sizes are zero,
     * the original sizes in those dimensions are presumed.
     */
    Mat reshape(int cn, int newndims, const int* newsz) const;

    /** @overload
     * @param cn New number of channels. If the parameter is 0, the number of channels remains the same.
     * @param newshape Vector with new matrix size by all dimentions. If some sizes are zero,
     * the original sizes in those dimensions are presumed.
     */
    Mat reshape(int cn, const std::vector<int>& newshape) const;

    /** @brief Transposes a matrix.

    The method performs matrix transposition by means of matrix expressions. It does not perform the
    actual transposition but returns a temporary matrix transposition object that can be further used as
    a part of more complex matrix expressions or can be assigned to a matrix:
    @code
        Mat A1 = A + Mat::eye(A.size(), A.type())*lambda;
        Mat C = A1.t()*A1; // compute (A + lambda*I)^t * (A + lamda*I)
    @endcode
     */
    MatExpr t() const;

    /** @brief Inverses a matrix.

    The method performs a matrix inversion by means of matrix expressions. This means that a temporary
    matrix inversion object is returned by the method and can be used further as a part of more complex
    matrix expressions or can be assigned to a matrix.
    @param method Matrix inversion method. One of cv::DecompTypes
     */
    MatExpr inv(int method=DECOMP_LU) const;

    /** @brief Performs an element-wise multiplication or division of the two matrices.

    The method returns a temporary object encoding per-element array multiplication, with optional
    scale. Note that this is not a matrix multiplication that corresponds to a simpler "\*" operator.

    Example:
    @code
        Mat C = A.mul(5/B); // equivalent to divide(A, B, C, 5)
    @endcode
    @param m Another array of the same type and the same size as \*this, or a matrix expression.
    @param scale Optional scale factor.
     */
    MatExpr mul(InputArray m, double scale=1) const;

    /** @brief Computes a cross-product of two 3-element vectors.

    The method computes a cross-product of two 3-element vectors. The vectors must be 3-element
    floating-point vectors of the same shape and size. The result is another 3-element vector of the
    same shape and type as operands.
    @param m Another cross-product operand.
     */
    Mat cross(InputArray m) const;

    /** @brief Computes a dot-product of two vectors.

    The method computes a dot-product of two matrices. If the matrices are not single-column or
    single-row vectors, the top-to-bottom left-to-right scan ordering is used to treat them as 1D
    vectors. The vectors must have the same size and type. If the matrices have more than one channel,
    the dot products from all the channels are summed together.
    @param m another dot-product operand.
     */
    double dot(InputArray m) const;

    /** @brief Returns a zero array of the specified size and type.

    The method returns a Matlab-style zero array initializer. It can be used to quickly form a constant
    array as a function parameter, part of a matrix expression, or as a matrix initializer:
    @code
        Mat A;
        A = Mat::zeros(3, 3, CV_32F);
    @endcode
    In the example above, a new matrix is allocated only if A is not a 3x3 floating-point matrix.
    Otherwise, the existing matrix A is filled with zeros.
    @param rows Number of rows.
    @param cols Number of columns.
    @param type Created matrix type.
     */
    CV_NODISCARD_STD static MatExpr zeros(int rows, int cols, int type);

    /** @overload
    @param size Alternative to the matrix size specification Size(cols, rows) .
    @param type Created matrix type.
    */
    CV_NODISCARD_STD static MatExpr zeros(Size size, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sz Array of integers specifying the array shape.
    @param type Created matrix type.
    */
    CV_NODISCARD_STD static MatExpr zeros(int ndims, const int* sz, int type);

    /** @brief Returns an array of all 1's of the specified size and type.

    The method returns a Matlab-style 1's array initializer, similarly to Mat::zeros. Note that using
    this method you can initialize an array with an arbitrary value, using the following Matlab idiom:
    @code
        Mat A = Mat::ones(100, 100, CV_8U)*3; // make 100x100 matrix filled with 3.
    @endcode
    The above operation does not form a 100x100 matrix of 1's and then multiply it by 3. Instead, it
    just remembers the scale factor (3 in this case) and use it when actually invoking the matrix
    initializer.
    @note In case of multi-channels type, only the first channel will be initialized with 1's, the
    others will be set to 0's.
    @param rows Number of rows.
    @param cols Number of columns.
    @param type Created matrix type.
     */
    CV_NODISCARD_STD static MatExpr ones(int rows, int cols, int type);

    /** @overload
    @param size Alternative to the matrix size specification Size(cols, rows) .
    @param type Created matrix type.
    */
    CV_NODISCARD_STD static MatExpr ones(Size size, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sz Array of integers specifying the array shape.
    @param type Created matrix type.
    */
    CV_NODISCARD_STD static MatExpr ones(int ndims, const int* sz, int type);

    /** @brief Returns an identity matrix of the specified size and type.

    The method returns a Matlab-style identity matrix initializer, similarly to Mat::zeros. Similarly to
    Mat::ones, you can use a scale operation to create a scaled identity matrix efficiently:
    @code
        // make a 4x4 diagonal matrix with 0.1's on the diagonal.
        Mat A = Mat::eye(4, 4, CV_32F)*0.1;
    @endcode
    @note In case of multi-channels type, identity matrix will be initialized only for the first channel,
    the others will be set to 0's
    @param rows Number of rows.
    @param cols Number of columns.
    @param type Created matrix type.
     */
    CV_NODISCARD_STD static MatExpr eye(int rows, int cols, int type);

    /** @overload
    @param size Alternative matrix size specification as Size(cols, rows) .
    @param type Created matrix type.
    */
    CV_NODISCARD_STD static MatExpr eye(Size size, int type);

    /** @brief Allocates new array data if needed.

    This is one of the key Mat methods. Most new-style OpenCV functions and methods that produce arrays
    call this method for each output array. The method uses the following algorithm:

    -# If the current array shape and the type match the new ones, return immediately. Otherwise,
       de-reference the previous data by calling Mat::release.
    -# Initialize the new header.
    -# Allocate the new data of total()\*elemSize() bytes.
    -# Allocate the new, associated with the data, reference counter and set it to 1.

    Such a scheme makes the memory management robust and efficient at the same time and helps avoid
    extra typing for you. This means that usually there is no need to explicitly allocate output arrays.
    That is, instead of writing:
    @code
        Mat color;
        ...
        Mat gray(color.rows, color.cols, color.depth());
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    you can simply write:
    @code
        Mat color;
        ...
        Mat gray;
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    because cvtColor, as well as the most of OpenCV functions, calls Mat::create() for the output array
    internally.
    @param rows New number of rows.
    @param cols New number of columns.
    @param type New matrix type.
     */
    void create(int rows, int cols, int type);

    /** @overload
    @param size Alternative new matrix size specification: Size(cols, rows)
    @param type New matrix type.
    */
    void create(Size size, int type);

    /** @overload
    @param ndims New array dimensionality.
    @param sizes Array of integers specifying a new array shape.
    @param type New matrix type.
    */
    void create(int ndims, const int* sizes, int type);

    /** @overload
    @param sizes Array of integers specifying a new array shape.
    @param type New matrix type.
    */
    void create(const std::vector<int>& sizes, int type);

    /** @brief Increments the reference counter.

    The method increments the reference counter associated with the matrix data. If the matrix header
    points to an external data set (see Mat::Mat ), the reference counter is NULL, and the method has no
    effect in this case. Normally, to avoid memory leaks, the method should not be called explicitly. It
    is called implicitly by the matrix assignment operator. The reference counter increment is an atomic
    operation on the platforms that support it. Thus, it is safe to operate on the same matrices
    asynchronously in different threads.
     */
    void addref();

    /** @brief Decrements the reference counter and deallocates the matrix if needed.

    The method decrements the reference counter associated with the matrix data. When the reference
    counter reaches 0, the matrix data is deallocated and the data and the reference counter pointers
    are set to NULL's. If the matrix header points to an external data set (see Mat::Mat ), the
    reference counter is NULL, and the method has no effect in this case.

    This method can be called manually to force the matrix data deallocation. But since this method is
    automatically called in the destructor, or by any other method that changes the data pointer, it is
    usually not needed. The reference counter decrement and check for 0 is an atomic operation on the
    platforms that support it. Thus, it is safe to operate on the same matrices asynchronously in
    different threads.
     */
    void release();

    //! internal use function, consider to use 'release' method instead; deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const Mat& m);

    /** @brief Reserves space for the certain number of rows.

    The method reserves space for sz rows. If the matrix already has enough space to store sz rows,
    nothing happens. If the matrix is reallocated, the first Mat::rows rows are preserved. The method
    emulates the corresponding method of the STL vector class.
    @param sz Number of rows.
     */
    void reserve(size_t sz);

    /** @brief Reserves space for the certain number of bytes.

    The method reserves space for sz bytes. If the matrix already has enough space to store sz bytes,
    nothing happens. If matrix has to be reallocated its previous content could be lost.
    @param sz Number of bytes.
    */
    void reserveBuffer(size_t sz);

    /** @brief Changes the number of matrix rows.

    The methods change the number of matrix rows. If the matrix is reallocated, the first
    min(Mat::rows, sz) rows are preserved. The methods emulate the corresponding methods of the STL
    vector class.
    @param sz New number of rows.
     */
    void resize(size_t sz);

    /** @overload
    @param sz New number of rows.
    @param s Value assigned to the newly added elements.
     */
    void resize(size_t sz, const Scalar& s);

    //! internal function
    void push_back_(const void* elem);

    /** @brief Adds elements to the bottom of the matrix.

    The methods add one or more elements to the bottom of the matrix. They emulate the corresponding
    method of the STL vector class. When elem is Mat , its type and the number of columns must be the
    same as in the container matrix.
    @param elem Added element(s).
     */
    template<typename _Tp> void push_back(const _Tp& elem);

    /** @overload
    @param elem Added element(s).
    */
    template<typename _Tp> void push_back(const Mat_<_Tp>& elem);

    /** @overload
    @param elem Added element(s).
    */
    template<typename _Tp> void push_back(const std::vector<_Tp>& elem);

    /** @overload
    @param m Added line(s).
    */
    void push_back(const Mat& m);

    /** @brief Removes elements from the bottom of the matrix.

    The method removes one or more rows from the bottom of the matrix.
    @param nelems Number of removed rows. If it is greater than the total number of rows, an exception
    is thrown.
     */
    void pop_back(size_t nelems=1);

    /** @brief Locates the matrix header within a parent matrix.

    After you extracted a submatrix from a matrix using Mat::row, Mat::col, Mat::rowRange,
    Mat::colRange, and others, the resultant submatrix points just to the part of the original big
    matrix. However, each submatrix contains information (represented by datastart and dataend
    fields) that helps reconstruct the original matrix size and the position of the extracted
    submatrix within the original matrix. The method locateROI does exactly that.
    @param wholeSize Output parameter that contains the size of the whole matrix containing *this*
    as a part.
    @param ofs Output parameter that contains an offset of *this* inside the whole matrix.
     */
    void locateROI( Size& wholeSize, Point& ofs ) const;

    /** @brief Adjusts a submatrix size and position within the parent matrix.

    The method is complimentary to Mat::locateROI . The typical use of these functions is to determine
    the submatrix position within the parent matrix and then shift the position somehow. Typically, it
    can be required for filtering operations when pixels outside of the ROI should be taken into
    account. When all the method parameters are positive, the ROI needs to grow in all directions by the
    specified amount, for example:
    @code
        A.adjustROI(2, 2, 2, 2);
    @endcode
    In this example, the matrix size is increased by 4 elements in each direction. The matrix is shifted
    by 2 elements to the left and 2 elements up, which brings in all the necessary pixels for the
    filtering with the 5x5 kernel.

    adjustROI forces the adjusted ROI to be inside of the parent matrix that is boundaries of the
    adjusted ROI are constrained by boundaries of the parent matrix. For example, if the submatrix A is
    located in the first row of a parent matrix and you called A.adjustROI(2, 2, 2, 2) then A will not
    be increased in the upward direction.

    The function is used internally by the OpenCV filtering functions, like filter2D , morphological
    operations, and so on.
    @param dtop Shift of the top submatrix boundary upwards.
    @param dbottom Shift of the bottom submatrix boundary downwards.
    @param dleft Shift of the left submatrix boundary to the left.
    @param dright Shift of the right submatrix boundary to the right.
    @sa copyMakeBorder
     */
    Mat& adjustROI( int dtop, int dbottom, int dleft, int dright );

    /** @brief Extracts a rectangular submatrix.

    The operators make a new header for the specified sub-array of \*this . They are the most
    generalized forms of Mat::row, Mat::col, Mat::rowRange, and Mat::colRange . For example,
    `A(Range(0, 10), Range::all())` is equivalent to `A.rowRange(0, 10)`. Similarly to all of the above,
    the operators are O(1) operations, that is, no matrix data is copied.
    @param rowRange Start and end row of the extracted submatrix. The upper boundary is not included. To
    select all the rows, use Range::all().
    @param colRange Start and end column of the extracted submatrix. The upper boundary is not included.
    To select all the columns, use Range::all().
     */
    Mat operator()( Range rowRange, Range colRange ) const;

    /** @overload
    @param roi Extracted submatrix specified as a rectangle.
    */
    Mat operator()( const Rect& roi ) const;

    /** @overload
    @param ranges Array of selected ranges along each array dimension.
    */
    Mat operator()( const Range* ranges ) const;

    /** @overload
    @param ranges Array of selected ranges along each array dimension.
    */
    Mat operator()(const std::vector<Range>& ranges) const;

    template<typename _Tp> operator std::vector<_Tp>() const;
    template<typename _Tp, int n> operator Vec<_Tp, n>() const;
    template<typename _Tp, int m, int n> operator Matx<_Tp, m, n>() const;

    template<typename _Tp, std::size_t _Nm> operator std::array<_Tp, _Nm>() const;

    /** @brief Reports whether the matrix is continuous or not.

    The method returns true if the matrix elements are stored continuously without gaps at the end of
    each row. Otherwise, it returns false. Obviously, 1x1 or 1xN matrices are always continuous.
    Matrices created with Mat::create are always continuous. But if you extract a part of the matrix
    using Mat::col, Mat::diag, and so on, or constructed a matrix header for externally allocated data,
    such matrices may no longer have this property.

    The continuity flag is stored as a bit in the Mat::flags field and is computed automatically when
    you construct a matrix header. Thus, the continuity check is a very fast operation, though
    theoretically it could be done as follows:
    @code
        // alternative implementation of Mat::isContinuous()
        bool myCheckMatContinuity(const Mat& m)
        {
            //return (m.flags & Mat::CONTINUOUS_FLAG) != 0;
            return m.rows == 1 || m.step == m.cols*m.elemSize();
        }
    @endcode
    The method is used in quite a few of OpenCV functions. The point is that element-wise operations
    (such as arithmetic and logical operations, math functions, alpha blending, color space
    transformations, and others) do not depend on the image geometry. Thus, if all the input and output
    arrays are continuous, the functions can process them as very long single-row vectors. The example
    below illustrates how an alpha-blending function can be implemented:
    @code
        template<typename T>
        void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
        {
            const float alpha_scale = (float)std::numeric_limits<T>::max(),
                        inv_scale = 1.f/alpha_scale;

            CV_Assert( src1.type() == src2.type() &&
                       src1.type() == CV_MAKETYPE(traits::Depth<T>::value, 4) &&
                       src1.size() == src2.size());
            Size size = src1.size();
            dst.create(size, src1.type());

            // here is the idiom: check the arrays for continuity and,
            // if this is the case,
            // treat the arrays as 1D vectors
            if( src1.isContinuous() && src2.isContinuous() && dst.isContinuous() )
            {
                size.width *= size.height;
                size.height = 1;
            }
            size.width *= 4;

            for( int i = 0; i < size.height; i++ )
            {
                // when the arrays are continuous,
                // the outer loop is executed only once
                const T* ptr1 = src1.ptr<T>(i);
                const T* ptr2 = src2.ptr<T>(i);
                T* dptr = dst.ptr<T>(i);

                for( int j = 0; j < size.width; j += 4 )
                {
                    float alpha = ptr1[j+3]*inv_scale, beta = ptr2[j+3]*inv_scale;
                    dptr[j] = saturate_cast<T>(ptr1[j]*alpha + ptr2[j]*beta);
                    dptr[j+1] = saturate_cast<T>(ptr1[j+1]*alpha + ptr2[j+1]*beta);
                    dptr[j+2] = saturate_cast<T>(ptr1[j+2]*alpha + ptr2[j+2]*beta);
                    dptr[j+3] = saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale);
                }
            }
        }
    @endcode
    This approach, while being very simple, can boost the performance of a simple element-operation by
    10-20 percents, especially if the image is rather small and the operation is quite simple.

    Another OpenCV idiom in this function, a call of Mat::create for the destination array, that
    allocates the destination array unless it already has the proper size and type. And while the newly
    allocated arrays are always continuous, you still need to check the destination array because
    Mat::create does not always allocate a new matrix.
     */
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    /** @brief Returns the matrix element size in bytes.

    The method returns the matrix element size in bytes. For example, if the matrix type is CV_16SC3 ,
    the method returns 3\*sizeof(short) or 6.
     */
    size_t elemSize() const;

    /** @brief Returns the size of each matrix element channel in bytes.

    The method returns the matrix element channel size in bytes, that is, it ignores the number of
    channels. For example, if the matrix type is CV_16SC3 , the method returns sizeof(short) or 2.
     */
    size_t elemSize1() const;

    /** @brief Returns the type of a matrix element.

    The method returns a matrix element type. This is an identifier compatible with the CvMat type
    system, like CV_16SC3 or 16-bit signed 3-channel array, and so on.
     */
    int type() const;

    /** @brief Returns the depth of a matrix element.

    The method returns the identifier of the matrix element depth (the type of each individual channel).
    For example, for a 16-bit signed element array, the method returns CV_16S . A complete list of
    matrix types contains the following values:
    -   CV_8U - 8-bit unsigned integers ( 0..255 )
    -   CV_8S - 8-bit signed integers ( -128..127 )
    -   CV_16U - 16-bit unsigned integers ( 0..65535 )
    -   CV_16S - 16-bit signed integers ( -32768..32767 )
    -   CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
    -   CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    -   CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
     */
    int depth() const;

    /** @brief Returns the number of matrix channels.

    The method returns the number of matrix channels.
     */
    int channels() const;

    /** @brief Returns a normalized step.

    The method returns a matrix step divided by Mat::elemSize1() . It can be useful to quickly access an
    arbitrary matrix element.
     */
    size_t step1(int i=0) const;

    /** @brief Returns true if the array has no elements.

    The method returns true if Mat::total() is 0 or if Mat::data is NULL. Because of pop_back() and
    resize() methods `M.total() == 0` does not imply that `M.data == NULL`.
     */
    bool empty() const;

    /** @brief Returns the total number of array elements.

    The method returns the number of array elements (a number of pixels if the array represents an
    image).
     */
    size_t total() const;

    /** @brief Returns the total number of array elements.

     The method returns the number of elements within a certain sub-array slice with startDim <= dim < endDim
     */
    size_t total(int startDim, int endDim=INT_MAX) const;

    /**
     * @param elemChannels Number of channels or number of columns the matrix should have.
     *                     For a 2-D matrix, when the matrix has only 1 column, then it should have
     *                     elemChannels channels; When the matrix has only 1 channel,
     *                     then it should have elemChannels columns.
     *                     For a 3-D matrix, it should have only one channel. Furthermore,
     *                     if the number of planes is not one, then the number of rows
     *                     within every plane has to be 1; if the number of rows within
     *                     every plane is not 1, then the number of planes has to be 1.
     * @param depth The depth the matrix should have. Set it to -1 when any depth is fine.
     * @param requireContinuous Set it to true to require the matrix to be continuous
     * @return -1 if the requirement is not satisfied.
     *         Otherwise, it returns the number of elements in the matrix. Note
     *         that an element may have multiple channels.
     *
     * The following code demonstrates its usage for a 2-d matrix:
     * @snippet snippets/core_mat_checkVector.cpp example-2d
     *
     * The following code demonstrates its usage for a 3-d matrix:
     * @snippet snippets/core_mat_checkVector.cpp example-3d
     */
    int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

    /** @brief Returns a pointer to the specified matrix row.

    The methods return `uchar*` or typed pointer to the specified matrix row. See the sample in
    Mat::isContinuous to know how to use these methods.
    @param i0 A 0-based row index.
     */
    uchar* ptr(int i0=0);
    /** @overload */
    const uchar* ptr(int i0=0) const;

    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    uchar* ptr(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    const uchar* ptr(int row, int col) const;

    /** @overload */
    uchar* ptr(int i0, int i1, int i2);
    /** @overload */
    const uchar* ptr(int i0, int i1, int i2) const;

    /** @overload */
    uchar* ptr(const int* idx);
    /** @overload */
    const uchar* ptr(const int* idx) const;
    /** @overload */
    template<int n> uchar* ptr(const Vec<int, n>& idx);
    /** @overload */
    template<int n> const uchar* ptr(const Vec<int, n>& idx) const;

    /** @overload */
    template<typename _Tp> _Tp* ptr(int i0=0);
    /** @overload */
    template<typename _Tp> const _Tp* ptr(int i0=0) const;
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> _Tp* ptr(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> const _Tp* ptr(int row, int col) const;
    /** @overload */
    template<typename _Tp> _Tp* ptr(int i0, int i1, int i2);
    /** @overload */
    template<typename _Tp> const _Tp* ptr(int i0, int i1, int i2) const;
    /** @overload */
    template<typename _Tp> _Tp* ptr(const int* idx);
    /** @overload */
    template<typename _Tp> const _Tp* ptr(const int* idx) const;
    /** @overload */
    template<typename _Tp, int n> _Tp* ptr(const Vec<int, n>& idx);
    /** @overload */
    template<typename _Tp, int n> const _Tp* ptr(const Vec<int, n>& idx) const;

    /** @brief Returns a reference to the specified array element.

    The template methods return a reference to the specified array element. For the sake of higher
    performance, the index range checks are only performed in the Debug configuration.

    Note that the variants with a single index (i) can be used to access elements of single-row or
    single-column 2-dimensional arrays. That is, if, for example, A is a 1 x N floating-point matrix and
    B is an M x 1 integer matrix, you can simply write `A.at<float>(k+4)` and `B.at<int>(2*i+1)`
    instead of `A.at<float>(0,k+4)` and `B.at<int>(2*i+1,0)`, respectively.

    The example below initializes a Hilbert matrix:
    @code
        Mat H(100, 100, CV_64F);
        for(int i = 0; i < H.rows; i++)
            for(int j = 0; j < H.cols; j++)
                H.at<double>(i,j)=1./(i+j+1);
    @endcode

    Keep in mind that the size identifier used in the at operator cannot be chosen at random. It depends
    on the image from which you are trying to retrieve the data. The table below gives a better insight in this:
     - If matrix is of type `CV_8U` then use `Mat.at<uchar>(y,x)`.
     - If matrix is of type `CV_8S` then use `Mat.at<schar>(y,x)`.
     - If matrix is of type `CV_16U` then use `Mat.at<ushort>(y,x)`.
     - If matrix is of type `CV_16S` then use `Mat.at<short>(y,x)`.
     - If matrix is of type `CV_32S`  then use `Mat.at<int>(y,x)`.
     - If matrix is of type `CV_32F`  then use `Mat.at<float>(y,x)`.
     - If matrix is of type `CV_64F` then use `Mat.at<double>(y,x)`.

    @param i0 Index along the dimension 0
     */
    template<typename _Tp> _Tp& at(int i0=0);
    /** @overload
    @param i0 Index along the dimension 0
    */
    template<typename _Tp> const _Tp& at(int i0=0) const;
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> _Tp& at(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> const _Tp& at(int row, int col) const;

    /** @overload
    @param i0 Index along the dimension 0
    @param i1 Index along the dimension 1
    @param i2 Index along the dimension 2
    */
    template<typename _Tp> _Tp& at(int i0, int i1, int i2);
    /** @overload
    @param i0 Index along the dimension 0
    @param i1 Index along the dimension 1
    @param i2 Index along the dimension 2
    */
    template<typename _Tp> const _Tp& at(int i0, int i1, int i2) const;

    /** @overload
    @param idx Array of Mat::dims indices.
    */
    template<typename _Tp> _Tp& at(const int* idx);
    /** @overload
    @param idx Array of Mat::dims indices.
    */
    template<typename _Tp> const _Tp& at(const int* idx) const;

    /** @overload */
    template<typename _Tp, int n> _Tp& at(const Vec<int, n>& idx);
    /** @overload */
    template<typename _Tp, int n> const _Tp& at(const Vec<int, n>& idx) const;

    /** @overload
    special versions for 2D arrays (especially convenient for referencing image pixels)
    @param pt Element position specified as Point(j,i) .
    */
    template<typename _Tp> _Tp& at(Point pt);
    /** @overload
    special versions for 2D arrays (especially convenient for referencing image pixels)
    @param pt Element position specified as Point(j,i) .
    */
    template<typename _Tp> const _Tp& at(Point pt) const;

    /** @brief Returns the matrix iterator and sets it to the first matrix element.

    The methods return the matrix read-only or read-write iterators. The use of matrix iterators is very
    similar to the use of bi-directional STL iterators. In the example below, the alpha blending
    function is rewritten using the matrix iterators:
    @code
        template<typename T>
        void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
        {
            typedef Vec<T, 4> VT;

            const float alpha_scale = (float)std::numeric_limits<T>::max(),
                        inv_scale = 1.f/alpha_scale;

            CV_Assert( src1.type() == src2.type() &&
                       src1.type() == traits::Type<VT>::value &&
                       src1.size() == src2.size());
            Size size = src1.size();
            dst.create(size, src1.type());

            MatConstIterator_<VT> it1 = src1.begin<VT>(), it1_end = src1.end<VT>();
            MatConstIterator_<VT> it2 = src2.begin<VT>();
            MatIterator_<VT> dst_it = dst.begin<VT>();

            for( ; it1 != it1_end; ++it1, ++it2, ++dst_it )
            {
                VT pix1 = *it1, pix2 = *it2;
                float alpha = pix1[3]*inv_scale, beta = pix2[3]*inv_scale;
                *dst_it = VT(saturate_cast<T>(pix1[0]*alpha + pix2[0]*beta),
                             saturate_cast<T>(pix1[1]*alpha + pix2[1]*beta),
                             saturate_cast<T>(pix1[2]*alpha + pix2[2]*beta),
                             saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale));
            }
        }
    @endcode
     */
    template<typename _Tp> MatIterator_<_Tp> begin();
    template<typename _Tp> MatConstIterator_<_Tp> begin() const;

    /** @brief Same as begin() but for inverse traversal
     */
    template<typename _Tp> std::reverse_iterator<MatIterator_<_Tp>> rbegin();
    template<typename _Tp> std::reverse_iterator<MatConstIterator_<_Tp>> rbegin() const;

    /** @brief Returns the matrix iterator and sets it to the after-last matrix element.

    The methods return the matrix read-only or read-write iterators, set to the point following the last
    matrix element.
     */
    template<typename _Tp> MatIterator_<_Tp> end();
    template<typename _Tp> MatConstIterator_<_Tp> end() const;

    /** @brief Same as end() but for inverse traversal
     */
    template<typename _Tp> std::reverse_iterator< MatIterator_<_Tp>> rend();
    template<typename _Tp> std::reverse_iterator< MatConstIterator_<_Tp>> rend() const;


    /** @brief Runs the given functor over all matrix elements in parallel.

    The operation passed as argument has to be a function pointer, a function object or a lambda(C++11).

    Example 1. All of the operations below put 0xFF the first channel of all matrix elements:
    @code
        Mat image(1920, 1080, CV_8UC3);
        typedef cv::Point3_<uint8_t> Pixel;

        // first. raw pointer access.
        for (int r = 0; r < image.rows; ++r) {
            Pixel* ptr = image.ptr<Pixel>(r, 0);
            const Pixel* ptr_end = ptr + image.cols;
            for (; ptr != ptr_end; ++ptr) {
                ptr->x = 255;
            }
        }

        // Using MatIterator. (Simple but there are a Iterator's overhead)
        for (Pixel &p : cv::Mat_<Pixel>(image)) {
            p.x = 255;
        }

        // Parallel execution with function object.
        struct Operator {
            void operator ()(Pixel &pixel, const int * position) {
                pixel.x = 255;
            }
        };
        image.forEach<Pixel>(Operator());

        // Parallel execution using C++11 lambda.
        image.forEach<Pixel>([](Pixel &p, const int * position) -> void {
            p.x = 255;
        });
    @endcode
    Example 2. Using the pixel's position:
    @code
        // Creating 3D matrix (255 x 255 x 255) typed uint8_t
        // and initialize all elements by the value which equals elements position.
        // i.e. pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).

        int sizes[] = { 255, 255, 255 };
        typedef cv::Point3_<uint8_t> Pixel;

        Mat_<Pixel> image = Mat::zeros(3, sizes, CV_8UC3);

        image.forEach<Pixel>([](Pixel& pixel, const int position[]) -> void {
            pixel.x = position[0];
            pixel.y = position[1];
            pixel.z = position[2];
        });
    @endcode
     */
    template<typename _Tp, typename Functor> void forEach(const Functor& operation);
    /** @overload */
    template<typename _Tp, typename Functor> void forEach(const Functor& operation) const;

    Mat(Mat&& m) CV_NOEXCEPT;
    Mat& operator = (Mat&& m);

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
    enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;
    //! pointer to the data
    uchar* data;

    //! helper fields used in locateROI and adjustROI
    const uchar* datastart;
    const uchar* dataend;
    const uchar* datalimit;

    //! custom allocator
    MatAllocator* allocator;
    //! and the standard allocator
    static MatAllocator* getStdAllocator();
    static MatAllocator* getDefaultAllocator();
    static void setDefaultAllocator(MatAllocator* allocator);

    //! internal use method: updates the continuity flag
    void updateContinuityFlag();

    //! interaction with UMat
    UMatData* u;

    MatSize size;
    MatStep step;

protected:
    template<typename _Tp, typename Functor> void forEach_impl(const Functor& operation);
};


///////////////////////////////// Mat_<_Tp> ////////////////////////////////////

/** @brief Template matrix class derived from Mat

@code{.cpp}
    template<typename _Tp> class Mat_ : public Mat
    {
    public:
        // ... some specific methods
        //         and
        // no new extra fields
    };
@endcode
The class `Mat_<_Tp>` is a *thin* template wrapper on top of the Mat class. It does not have any
extra data fields. Nor this class nor Mat has any virtual methods. Thus, references or pointers to
these two classes can be freely but carefully converted one to another. For example:
@code{.cpp}
    // create a 100x100 8-bit matrix
    Mat M(100,100,CV_8U);
    // this will be compiled fine. no any data conversion will be done.
    Mat_<float>& M1 = (Mat_<float>&)M;
    // the program is likely to crash at the statement below
    M1(99,99) = 1.f;
@endcode
While Mat is sufficient in most cases, Mat_ can be more convenient if you use a lot of element
access operations and if you know matrix type at the compilation time. Note that
`Mat::at(int y,int x)` and `Mat_::operator()(int y,int x)` do absolutely the same
and run at the same speed, but the latter is certainly shorter:
@code{.cpp}
    Mat_<double> M(20,20);
    for(int i = 0; i < M.rows; i++)
        for(int j = 0; j < M.cols; j++)
            M(i,j) = 1./(i+j+1);
    Mat E, V;
    eigen(M,E,V);
    cout << E.at<double>(0,0)/E.at<double>(M.rows-1,0);
@endcode
To use Mat_ for multi-channel images/matrices, pass Vec as a Mat_ parameter:
@code{.cpp}
    // allocate a 320x240 color image and fill it with green (in RGB space)
    Mat_<Vec3b> img(240, 320, Vec3b(0,255,0));
    // now draw a diagonal white line
    for(int i = 0; i < 100; i++)
        img(i,i)=Vec3b(255,255,255);
    // and now scramble the 2nd (red) channel of each pixel
    for(int i = 0; i < img.rows; i++)
        for(int j = 0; j < img.cols; j++)
            img(i,j)[2] ^= (uchar)(i ^ j);
@endcode
Mat_ is fully compatible with C++11 range-based for loop. For example such loop
can be used to safely apply look-up table:
@code{.cpp}
void applyTable(Mat_<uchar>& I, const uchar* const table)
{
    for(auto& pixel : I)
    {
        pixel = table[pixel];
    }
}
@endcode
 */
template<typename _Tp> class Mat_ : public Mat
{
public:
    typedef _Tp value_type;
    typedef typename DataType<_Tp>::channel_type channel_type;
    typedef MatIterator_<_Tp> iterator;
    typedef MatConstIterator_<_Tp> const_iterator;

    //! default constructor
    Mat_() CV_NOEXCEPT;
    //! equivalent to Mat(_rows, _cols, DataType<_Tp>::type)
    Mat_(int _rows, int _cols);
    //! constructor that sets each matrix element to specified value
    Mat_(int _rows, int _cols, const _Tp& value);
    //! equivalent to Mat(_size, DataType<_Tp>::type)
    explicit Mat_(Size _size);
    //! constructor that sets each matrix element to specified value
    Mat_(Size _size, const _Tp& value);
    //! n-dim array constructor
    Mat_(int _ndims, const int* _sizes);
    //! n-dim array constructor that sets each matrix element to specified value
    Mat_(int _ndims, const int* _sizes, const _Tp& value);
    //! copy/conversion constructor. If m is of different type, it's converted
    Mat_(const Mat& m);
    //! copy constructor
    Mat_(const Mat_& m);
    //! constructs a matrix on top of user-allocated data. step is in bytes(!!!), regardless of the type
    Mat_(int _rows, int _cols, _Tp* _data, size_t _step=AUTO_STEP);
    //! constructs n-dim matrix on top of user-allocated data. steps are in bytes(!!!), regardless of the type
    Mat_(int _ndims, const int* _sizes, _Tp* _data, const size_t* _steps=0);
    //! selects a submatrix
    Mat_(const Mat_& m, const Range& rowRange, const Range& colRange=Range::all());
    //! selects a submatrix
    Mat_(const Mat_& m, const Rect& roi);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_& m, const Range* ranges);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_& m, const std::vector<Range>& ranges);
    //! from a matrix expression
    explicit Mat_(const MatExpr& e);
    //! makes a matrix out of Vec, std::vector, Point_ or Point3_. The matrix will have a single column
    explicit Mat_(const std::vector<_Tp>& vec, bool copyData=false);
    template<int n> explicit Mat_(const Vec<typename DataType<_Tp>::channel_type, n>& vec, bool copyData=true);
    template<int m, int n> explicit Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n>& mtx, bool copyData=true);
    explicit Mat_(const Point_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const Point3_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const MatCommaInitializer_<_Tp>& commaInitializer);

    Mat_(std::initializer_list<_Tp> values);
    explicit Mat_(const std::initializer_list<int> sizes, const std::initializer_list<_Tp> values);

    template <std::size_t _Nm> explicit Mat_(const std::array<_Tp, _Nm>& arr, bool copyData=false);

    Mat_& operator = (const Mat& m);
    Mat_& operator = (const Mat_& m);
    //! set all the elements to s.
    Mat_& operator = (const _Tp& s);
    //! assign a matrix expression
    Mat_& operator = (const MatExpr& e);

    //! iterators; they are smart enough to skip gaps in the end of rows
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    //reverse iterators
    std::reverse_iterator<iterator> rbegin();
    std::reverse_iterator<iterator> rend();
    std::reverse_iterator<const_iterator> rbegin() const;
    std::reverse_iterator<const_iterator> rend() const;

    //! template methods for operation over all matrix elements.
    // the operations take care of skipping gaps in the end of rows (if any)
    template<typename Functor> void forEach(const Functor& operation);
    template<typename Functor> void forEach(const Functor& operation) const;

    //! equivalent to Mat::create(_rows, _cols, DataType<_Tp>::type)
    void create(int _rows, int _cols);
    //! equivalent to Mat::create(_size, DataType<_Tp>::type)
    void create(Size _size);
    //! equivalent to Mat::create(_ndims, _sizes, DatType<_Tp>::type)
    void create(int _ndims, const int* _sizes);
    //! equivalent to Mat::release()
    void release();
    //! cross-product
    Mat_ cross(const Mat_& m) const;
    //! data type conversion
    template<typename T2> operator Mat_<T2>() const;
    //! overridden forms of Mat::row() etc.
    Mat_ row(int y) const;
    Mat_ col(int x) const;
    Mat_ diag(int d=0) const;
    CV_NODISCARD_STD Mat_ clone() const;

    //! overridden forms of Mat::elemSize() etc.
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1(int i=0) const;
    //! returns step()/sizeof(_Tp)
    size_t stepT(int i=0) const;

    //! overridden forms of Mat::zeros() etc. Data type is omitted, of course
    CV_NODISCARD_STD static MatExpr zeros(int rows, int cols);
    CV_NODISCARD_STD static MatExpr zeros(Size size);
    CV_NODISCARD_STD static MatExpr zeros(int _ndims, const int* _sizes);
    CV_NODISCARD_STD static MatExpr ones(int rows, int cols);
    CV_NODISCARD_STD static MatExpr ones(Size size);
    CV_NODISCARD_STD static MatExpr ones(int _ndims, const int* _sizes);
    CV_NODISCARD_STD static MatExpr eye(int rows, int cols);
    CV_NODISCARD_STD static MatExpr eye(Size size);

    //! some more overridden methods
    Mat_& adjustROI( int dtop, int dbottom, int dleft, int dright );
    Mat_ operator()( const Range& rowRange, const Range& colRange ) const;
    Mat_ operator()( const Rect& roi ) const;
    Mat_ operator()( const Range* ranges ) const;
    Mat_ operator()(const std::vector<Range>& ranges) const;

    //! more convenient forms of row and element access operators
    _Tp* operator [](int y);
    const _Tp* operator [](int y) const;

    //! returns reference to the specified element
    _Tp& operator ()(const int* idx);
    //! returns read-only reference to the specified element
    const _Tp& operator ()(const int* idx) const;

    //! returns reference to the specified element
    template<int n> _Tp& operator ()(const Vec<int, n>& idx);
    //! returns read-only reference to the specified element
    template<int n> const _Tp& operator ()(const Vec<int, n>& idx) const;

    //! returns reference to the specified element (1D case)
    _Tp& operator ()(int idx0);
    //! returns read-only reference to the specified element (1D case)
    const _Tp& operator ()(int idx0) const;
    //! returns reference to the specified element (2D case)
    _Tp& operator ()(int row, int col);
    //! returns read-only reference to the specified element (2D case)
    const _Tp& operator ()(int row, int col) const;
    //! returns reference to the specified element (3D case)
    _Tp& operator ()(int idx0, int idx1, int idx2);
    //! returns read-only reference to the specified element (3D case)
    const _Tp& operator ()(int idx0, int idx1, int idx2) const;

    _Tp& operator ()(Point pt);
    const _Tp& operator ()(Point pt) const;

    //! conversion to vector.
    operator std::vector<_Tp>() const;

    //! conversion to array.
    template<std::size_t _Nm> operator std::array<_Tp, _Nm>() const;

    //! conversion to Vec
    template<int n> operator Vec<typename DataType<_Tp>::channel_type, n>() const;
    //! conversion to Matx
    template<int m, int n> operator Matx<typename DataType<_Tp>::channel_type, m, n>() const;

    Mat_(Mat_&& m);
    Mat_& operator = (Mat_&& m);

    Mat_(Mat&& m);
    Mat_& operator = (Mat&& m);

    Mat_(MatExpr&& e);
};

typedef Mat_<uchar> Mat1b;
typedef Mat_<Vec2b> Mat2b;
typedef Mat_<Vec3b> Mat3b;
typedef Mat_<Vec4b> Mat4b;

typedef Mat_<short> Mat1s;
typedef Mat_<Vec2s> Mat2s;
typedef Mat_<Vec3s> Mat3s;
typedef Mat_<Vec4s> Mat4s;

typedef Mat_<ushort> Mat1w;
typedef Mat_<Vec2w> Mat2w;
typedef Mat_<Vec3w> Mat3w;
typedef Mat_<Vec4w> Mat4w;

typedef Mat_<int>   Mat1i;
typedef Mat_<Vec2i> Mat2i;
typedef Mat_<Vec3i> Mat3i;
typedef Mat_<Vec4i> Mat4i;

typedef Mat_<float> Mat1f;
typedef Mat_<Vec2f> Mat2f;
typedef Mat_<Vec3f> Mat3f;
typedef Mat_<Vec4f> Mat4f;

typedef Mat_<double> Mat1d;
typedef Mat_<Vec2d> Mat2d;
typedef Mat_<Vec3d> Mat3d;
typedef Mat_<Vec4d> Mat4d;

/** @todo document */
class CV_EXPORTS UMat
{
public:
    //! default constructor
    UMat(UMatUsageFlags usageFlags = USAGE_DEFAULT) CV_NOEXCEPT;
    //! constructs 2D matrix of the specified size and type
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
    UMat(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constructs 2D matrix and fills it with the specified value _s.
    UMat(int rows, int cols, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(Size size, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! constructs n-dimensional matrix
    UMat(int ndims, const int* sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(int ndims, const int* sizes, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! copy constructor
    UMat(const UMat& m);

    //! creates a matrix header for a part of the bigger matrix
    UMat(const UMat& m, const Range& rowRange, const Range& colRange=Range::all());
    UMat(const UMat& m, const Rect& roi);
    UMat(const UMat& m, const Range* ranges);
    UMat(const UMat& m, const std::vector<Range>& ranges);

    // FIXIT copyData=false is not implemented, drop this in favor of cv::Mat (OpenCV 5.0)
    //! builds matrix from std::vector with or without copying the data
    template<typename _Tp> explicit UMat(const std::vector<_Tp>& vec, bool copyData=false);

    //! destructor - calls release()
    ~UMat();
    //! assignment operators
    UMat& operator = (const UMat& m);

    Mat getMat(AccessFlag flags) const;

    //! returns a new matrix header for the specified row
    UMat row(int y) const;
    //! returns a new matrix header for the specified column
    UMat col(int x) const;
    //! ... for the specified row span
    UMat rowRange(int startrow, int endrow) const;
    UMat rowRange(const Range& r) const;
    //! ... for the specified column span
    UMat colRange(int startcol, int endcol) const;
    UMat colRange(const Range& r) const;
    //! ... for the specified diagonal
    //! (d=0 - the main diagonal,
    //!  >0 - a diagonal from the upper half,
    //!  <0 - a diagonal from the lower half)
    UMat diag(int d=0) const;
    //! constructs a square diagonal matrix which main diagonal is vector "d"
    CV_NODISCARD_STD static UMat diag(const UMat& d, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat diag(const UMat& d) { return diag(d, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload

    //! returns deep copy of the matrix, i.e. the data is copied
    CV_NODISCARD_STD UMat clone() const;
    //! copies the matrix content to "m".
    // It calls m.create(this->size(), this->type()).
    void copyTo( OutputArray m ) const;
    //! copies those matrix elements to "m" that are marked with non-zero mask elements.
    void copyTo( OutputArray m, InputArray mask ) const;
    //! converts matrix to another datatype with optional scaling. See cvConvertScale.
    void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;

    void assignTo( UMat& m, int type=-1 ) const;

    //! sets every matrix element to s
    UMat& operator = (const Scalar& s);
    //! sets some of the matrix elements to s, according to the mask
    UMat& setTo(InputArray value, InputArray mask=noArray());
    //! creates alternative matrix header for the same data, with different
    // number of channels and/or different number of rows. see cvReshape.
    UMat reshape(int cn, int rows=0) const;
    UMat reshape(int cn, int newndims, const int* newsz) const;

    //! matrix transposition by means of matrix expressions
    UMat t() const;
    //! matrix inversion by means of matrix expressions
    UMat inv(int method=DECOMP_LU) const;
    //! per-element matrix multiplication by means of matrix expressions
    UMat mul(InputArray m, double scale=1) const;

    //! computes dot-product
    double dot(InputArray m) const;

    //! Matlab-style matrix initialization
    CV_NODISCARD_STD static UMat zeros(int rows, int cols, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat zeros(Size size, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat zeros(int ndims, const int* sz, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat zeros(int rows, int cols, int type) { return zeros(rows, cols, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat zeros(Size size, int type) { return zeros(size, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat zeros(int ndims, const int* sz, int type) { return zeros(ndims, sz, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat ones(int rows, int cols, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat ones(Size size, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat ones(int ndims, const int* sz, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat ones(int rows, int cols, int type) { return ones(rows, cols, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat ones(Size size, int type) { return ones(size, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat ones(int ndims, const int* sz, int type) { return ones(ndims, sz, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat eye(int rows, int cols, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat eye(Size size, int type, UMatUsageFlags usageFlags /*= USAGE_DEFAULT*/);
    CV_NODISCARD_STD static UMat eye(int rows, int cols, int type) { return eye(rows, cols, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload
    CV_NODISCARD_STD static UMat eye(Size size, int type) { return eye(size, type, USAGE_DEFAULT); }  // OpenCV 5.0: remove abi compatibility overload

    //! allocates new matrix data unless the matrix already has specified size and type.
    // previous data is unreferenced if needed.
    void create(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(int ndims, const int* sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(const std::vector<int>& sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! increases the reference counter; use with care to avoid memleaks
    void addref();
    //! decreases reference counter;
    // deallocates the data when reference counter reaches 0.
    void release();

    //! deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const UMat& m);

    //! locates matrix header within a parent matrix. See below
    void locateROI( Size& wholeSize, Point& ofs ) const;
    //! moves/resizes the current matrix ROI inside the parent matrix.
    UMat& adjustROI( int dtop, int dbottom, int dleft, int dright );
    //! extracts a rectangular sub-matrix
    // (this is a generalized form of row, rowRange etc.)
    UMat operator()( Range rowRange, Range colRange ) const;
    UMat operator()( const Rect& roi ) const;
    UMat operator()( const Range* ranges ) const;
    UMat operator()(const std::vector<Range>& ranges) const;

    //! returns true iff the matrix data is continuous
    // (i.e. when there are no gaps between successive rows).
    // similar to CV_IS_MAT_CONT(cvmat->type)
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    //! returns element size in bytes,
    // similar to CV_ELEM_SIZE(cvmat->type)
    size_t elemSize() const;
    //! returns the size of element channel in bytes.
    size_t elemSize1() const;
    //! returns element type, similar to CV_MAT_TYPE(cvmat->type)
    int type() const;
    //! returns element type, similar to CV_MAT_DEPTH(cvmat->type)
    int depth() const;
    //! returns element type, similar to CV_MAT_CN(cvmat->type)
    int channels() const;
    //! returns step/elemSize1()
    size_t step1(int i=0) const;
    //! returns true if matrix data is NULL
    bool empty() const;
    //! returns the total number of matrix elements
    size_t total() const;

    //! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
    int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

    UMat(UMat&& m);
    UMat& operator = (UMat&& m);

    /*! Returns the OpenCL buffer handle on which UMat operates on.
        The UMat instance should be kept alive during the use of the handle to prevent the buffer to be
        returned to the OpenCV buffer pool.
     */
    void* handle(AccessFlag accessFlags) const;
    void ndoffset(size_t* ofs) const;

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
    enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;

    //! the matrix dimensionality, >= 2
    int dims;

    //! number of rows in the matrix; -1 when the matrix has more than 2 dimensions
    int rows;

    //! number of columns in the matrix; -1 when the matrix has more than 2 dimensions
    int cols;

    //! custom allocator
    MatAllocator* allocator;

    //! usage flags for allocator; recommend do not set directly, instead set during construct/create/getUMat
    UMatUsageFlags usageFlags;

    //! and the standard allocator
    static MatAllocator* getStdAllocator();

    //! internal use method: updates the continuity flag
    void updateContinuityFlag();

    //! black-box container of UMat data
    UMatData* u;

    //! offset of the submatrix (or 0)
    size_t offset;

    //! dimensional size of the matrix; accessible in various formats
    MatSize size;

    //! number of bytes each matrix element/row/plane/dimension occupies
    MatStep step;

protected:
};


/////////////////////////// multi-dimensional sparse matrix //////////////////////////

/** @brief The class SparseMat represents multi-dimensional sparse numerical arrays.

Such a sparse array can store elements of any type that Mat can store. *Sparse* means that only
non-zero elements are stored (though, as a result of operations on a sparse matrix, some of its
stored elements can actually become 0. It is up to you to detect such elements and delete them
using SparseMat::erase ). The non-zero elements are stored in a hash table that grows when it is
filled so that the search time is O(1) in average (regardless of whether element is there or not).
Elements can be accessed using the following methods:
-   Query operations (SparseMat::ptr and the higher-level SparseMat::ref, SparseMat::value and
    SparseMat::find), for example:
    @code
        const int dims = 5;
        int size[5] = {10, 10, 10, 10, 10};
        SparseMat sparse_mat(dims, size, CV_32F);
        for(int i = 0; i < 1000; i++)
        {
            int idx[dims];
            for(int k = 0; k < dims; k++)
                idx[k] = rand() % size[k];
            sparse_mat.ref<float>(idx) += 1.f;
        }
        cout << "nnz = " << sparse_mat.nzcount() << endl;
    @endcode
-   Sparse matrix iterators. They are similar to MatIterator but different from NAryMatIterator.
    That is, the iteration loop is familiar to STL users:
    @code
        // prints elements of a sparse floating-point matrix
        // and the sum of elements.
        SparseMatConstIterator_<float>
            it = sparse_mat.begin<float>(),
            it_end = sparse_mat.end<float>();
        double s = 0;
        int dims = sparse_mat.dims();
        for(; it != it_end; ++it)
        {
            // print element indices and the element value
            const SparseMat::Node* n = it.node();
            printf("(");
            for(int i = 0; i < dims; i++)
                printf("%d%s", n->idx[i], i < dims-1 ? ", " : ")");
            printf(": %g\n", it.value<float>());
            s += *it;
        }
        printf("Element sum is %g\n", s);
    @endcode
    If you run this loop, you will notice that elements are not enumerated in a logical order
    (lexicographical, and so on). They come in the same order as they are stored in the hash table
    (semi-randomly). You may collect pointers to the nodes and sort them to get the proper ordering.
    Note, however, that pointers to the nodes may become invalid when you add more elements to the
    matrix. This may happen due to possible buffer reallocation.
-   Combination of the above 2 methods when you need to process 2 or more sparse matrices
    simultaneously. For example, this is how you can compute unnormalized cross-correlation of the 2
    floating-point sparse matrices:
    @code
        double cross_corr(const SparseMat& a, const SparseMat& b)
        {
            const SparseMat *_a = &a, *_b = &b;
            // if b contains less elements than a,
            // it is faster to iterate through b
            if(_a->nzcount() > _b->nzcount())
                std::swap(_a, _b);
            SparseMatConstIterator_<float> it = _a->begin<float>(),
                                           it_end = _a->end<float>();
            double ccorr = 0;
            for(; it != it_end; ++it)
            {
                // take the next element from the first matrix
                float avalue = *it;
                const Node* anode = it.node();
                // and try to find an element with the same index in the second matrix.
                // since the hash value depends only on the element index,
                // reuse the hash value stored in the node
                float bvalue = _b->value<float>(anode->idx,&anode->hashval);
                ccorr += avalue*bvalue;
            }
            return ccorr;
        }
    @endcode
 */
class CV_EXPORTS SparseMat
{
public:
    typedef SparseMatIterator iterator;
    typedef SparseMatConstIterator const_iterator;

    enum { MAGIC_VAL=0x42FD0000, MAX_DIM=32, HASH_SCALE=0x5bd1e995, HASH_BIT=0x80000000 };

    //! the sparse matrix header
    struct CV_EXPORTS Hdr
    {
        Hdr(int _dims, const int* _sizes, int _type);
        void clear();
        int refcount;
        int dims;
        int valueOffset;
        size_t nodeSize;
        size_t nodeCount;
        size_t freeList;
        std::vector<uchar> pool;
        std::vector<size_t> hashtab;
        int size[MAX_DIM];
    };

    //! sparse matrix node - element of a hash table
    struct CV_EXPORTS Node
    {
        //! hash value
        size_t hashval;
        //! index of the next node in the same hash table entry
        size_t next;
        //! index of the matrix element
        int idx[MAX_DIM];
    };

    /** @brief Various SparseMat constructors.
     */
    SparseMat();

    /** @overload
    @param dims Array dimensionality.
    @param _sizes Sparce matrix size on all dementions.
    @param _type Sparse matrix data type.
    */
    SparseMat(int dims, const int* _sizes, int _type);

    /** @overload
    @param m Source matrix for copy constructor. If m is dense matrix (ocvMat) then it will be converted
    to sparse representation.
    */
    SparseMat(const SparseMat& m);

    /** @overload
    @param m Source matrix for copy constructor. If m is dense matrix (ocvMat) then it will be converted
    to sparse representation.
    */
    explicit SparseMat(const Mat& m);

    //! the destructor
    ~SparseMat();

    //! assignment operator. This is O(1) operation, i.e. no data is copied
    SparseMat& operator = (const SparseMat& m);
    //! equivalent to the corresponding constructor
    SparseMat& operator = (const Mat& m);

    //! creates full copy of the matrix
    CV_NODISCARD_STD SparseMat clone() const;

    //! copies all the data to the destination matrix. All the previous content of m is erased
    void copyTo( SparseMat& m ) const;
    //! converts sparse matrix to dense matrix.
    void copyTo( Mat& m ) const;
    //! multiplies all the matrix elements by the specified scale factor alpha and converts the results to the specified data type
    void convertTo( SparseMat& m, int rtype, double alpha=1 ) const;
    //! converts sparse matrix to dense n-dim matrix with optional type conversion and scaling.
    /*!
        @param [out] m - output matrix; if it does not have a proper size or type before the operation,
            it is reallocated
        @param [in] rtype - desired output matrix type or, rather, the depth since the number of channels
            are the same as the input has; if rtype is negative, the output matrix will have the
            same type as the input.
        @param [in] alpha - optional scale factor
        @param [in] beta - optional delta added to the scaled values
    */
    void convertTo( Mat& m, int rtype, double alpha=1, double beta=0 ) const;

    // not used now
    void assignTo( SparseMat& m, int type=-1 ) const;

    //! reallocates sparse matrix.
    /*!
        If the matrix already had the proper size and type,
        it is simply cleared with clear(), otherwise,
        the old matrix is released (using release()) and the new one is allocated.
    */
    void create(int dims, const int* _sizes, int _type);
    //! sets all the sparse matrix elements to 0, which means clearing the hash table.
    void clear();
    //! manually increments the reference counter to the header.
    void addref();
    // decrements the header reference counter. When the counter reaches 0, the header and all the underlying data are deallocated.
    void release();

    //! converts sparse matrix to the old-style representation; all the elements are copied.
    //operator CvSparseMat*() const;
    //! returns the size of each element in bytes (not including the overhead - the space occupied by SparseMat::Node elements)
    size_t elemSize() const;
    //! returns elemSize()/channels()
    size_t elemSize1() const;

    //! returns type of sparse matrix elements
    int type() const;
    //! returns the depth of sparse matrix elements
    int depth() const;
    //! returns the number of channels
    int channels() const;

    //! returns the array of sizes, or NULL if the matrix is not allocated
    const int* size() const;
    //! returns the size of i-th matrix dimension (or 0)
    int size(int i) const;
    //! returns the matrix dimensionality
    int dims() const;
    //! returns the number of non-zero elements (=the number of hash table nodes)
    size_t nzcount() const;

    //! computes the element hash value (1D case)
    size_t hash(int i0) const;
    //! computes the element hash value (2D case)
    size_t hash(int i0, int i1) const;
    //! computes the element hash value (3D case)
    size_t hash(int i0, int i1, int i2) const;
    //! computes the element hash value (nD case)
    size_t hash(const int* idx) const;

    //!@{
    /*!
     specialized variants for 1D, 2D, 3D cases and the generic_type one for n-D case.
     return pointer to the matrix element.
      - if the element is there (it's non-zero), the pointer to it is returned
      - if it's not there and createMissing=false, NULL pointer is returned
      - if it's not there and createMissing=true, then the new element
        is created and initialized with 0. Pointer to it is returned
      - if the optional hashval pointer is not NULL, the element hash value is
        not computed, but *hashval is taken instead.
    */
    //! returns pointer to the specified element (1D case)
    uchar* ptr(int i0, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (2D case)
    uchar* ptr(int i0, int i1, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (3D case)
    uchar* ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (nD case)
    uchar* ptr(const int* idx, bool createMissing, size_t* hashval=0);
    //!@}

    //!@{
    /*!
     return read-write reference to the specified sparse matrix element.

     `ref<_Tp>(i0,...[,hashval])` is equivalent to `*(_Tp*)ptr(i0,...,true[,hashval])`.
     The methods always return a valid reference.
     If the element did not exist, it is created and initialized with 0.
    */
    //! returns reference to the specified element (1D case)
    template<typename _Tp> _Tp& ref(int i0, size_t* hashval=0);
    //! returns reference to the specified element (2D case)
    template<typename _Tp> _Tp& ref(int i0, int i1, size_t* hashval=0);
    //! returns reference to the specified element (3D case)
    template<typename _Tp> _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    //! returns reference to the specified element (nD case)
    template<typename _Tp> _Tp& ref(const int* idx, size_t* hashval=0);
    //!@}

    //!@{
    /*!
     return value of the specified sparse matrix element.

     `value<_Tp>(i0,...[,hashval])` is equivalent to
     @code
     { const _Tp* p = find<_Tp>(i0,...[,hashval]); return p ? *p : _Tp(); }
     @endcode

     That is, if the element did not exist, the methods return 0.
     */
    //! returns value of the specified element (1D case)
    template<typename _Tp> _Tp value(int i0, size_t* hashval=0) const;
    //! returns value of the specified element (2D case)
    template<typename _Tp> _Tp value(int i0, int i1, size_t* hashval=0) const;
    //! returns value of the specified element (3D case)
    template<typename _Tp> _Tp value(int i0, int i1, int i2, size_t* hashval=0) const;
    //! returns value of the specified element (nD case)
    template<typename _Tp> _Tp value(const int* idx, size_t* hashval=0) const;
    //!@}

    //!@{
    /*!
     Return pointer to the specified sparse matrix element if it exists

     `find<_Tp>(i0,...[,hashval])` is equivalent to `(_const Tp*)ptr(i0,...false[,hashval])`.

     If the specified element does not exist, the methods return NULL.
    */
    //! returns pointer to the specified element (1D case)
    template<typename _Tp> const _Tp* find(int i0, size_t* hashval=0) const;
    //! returns pointer to the specified element (2D case)
    template<typename _Tp> const _Tp* find(int i0, int i1, size_t* hashval=0) const;
    //! returns pointer to the specified element (3D case)
    template<typename _Tp> const _Tp* find(int i0, int i1, int i2, size_t* hashval=0) const;
    //! returns pointer to the specified element (nD case)
    template<typename _Tp> const _Tp* find(const int* idx, size_t* hashval=0) const;
    //!@}

    //! erases the specified element (2D case)
    void erase(int i0, int i1, size_t* hashval=0);
    //! erases the specified element (3D case)
    void erase(int i0, int i1, int i2, size_t* hashval=0);
    //! erases the specified element (nD case)
    void erase(const int* idx, size_t* hashval=0);

    //!@{
    /*!
       return the sparse matrix iterator pointing to the first sparse matrix element
    */
    //! returns the sparse matrix iterator at the matrix beginning
    SparseMatIterator begin();
    //! returns the sparse matrix iterator at the matrix beginning
    template<typename _Tp> SparseMatIterator_<_Tp> begin();
    //! returns the read-only sparse matrix iterator at the matrix beginning
    SparseMatConstIterator begin() const;
    //! returns the read-only sparse matrix iterator at the matrix beginning
    template<typename _Tp> SparseMatConstIterator_<_Tp> begin() const;
    //!@}
    /*!
       return the sparse matrix iterator pointing to the element following the last sparse matrix element
    */
    //! returns the sparse matrix iterator at the matrix end
    SparseMatIterator end();
    //! returns the read-only sparse matrix iterator at the matrix end
    SparseMatConstIterator end() const;
    //! returns the typed sparse matrix iterator at the matrix end
    template<typename _Tp> SparseMatIterator_<_Tp> end();
    //! returns the typed read-only sparse matrix iterator at the matrix end
    template<typename _Tp> SparseMatConstIterator_<_Tp> end() const;

    //! returns the value stored in the sparse martix node
    template<typename _Tp> _Tp& value(Node* n);
    //! returns the value stored in the sparse martix node
    template<typename _Tp> const _Tp& value(const Node* n) const;

    ////////////// some internal-use methods ///////////////
    Node* node(size_t nidx);
    const Node* node(size_t nidx) const;

    uchar* newNode(const int* idx, size_t hashval);
    void removeNode(size_t hidx, size_t nidx, size_t previdx);
    void resizeHashTab(size_t newsize);

    int flags;
    Hdr* hdr;
};



///////////////////////////////// SparseMat_<_Tp> ////////////////////////////////////

/** @brief Template sparse n-dimensional array class derived from SparseMat

SparseMat_ is a thin wrapper on top of SparseMat created in the same way as Mat_ . It simplifies
notation of some operations:
@code
    int sz[] = {10, 20, 30};
    SparseMat_<double> M(3, sz);
    ...
    M.ref(1, 2, 3) = M(4, 5, 6) + M(7, 8, 9);
@endcode
 */
template<typename _Tp> class SparseMat_ : public SparseMat
{
public:
    typedef SparseMatIterator_<_Tp> iterator;
    typedef SparseMatConstIterator_<_Tp> const_iterator;

    //! the default constructor
    SparseMat_();
    //! the full constructor equivalent to SparseMat(dims, _sizes, DataType<_Tp>::type)
    SparseMat_(int dims, const int* _sizes);
    //! the copy constructor. If DataType<_Tp>.type != m.type(), the m elements are converted
    SparseMat_(const SparseMat& m);
    //! the copy constructor. This is O(1) operation - no data is copied
    SparseMat_(const SparseMat_& m);
    //! converts dense matrix to the sparse form
    SparseMat_(const Mat& m);
    //! converts the old-style sparse matrix to the C++ class. All the elements are copied
    //SparseMat_(const CvSparseMat* m);
    //! the assignment operator. If DataType<_Tp>.type != m.type(), the m elements are converted
    SparseMat_& operator = (const SparseMat& m);
    //! the assignment operator. This is O(1) operation - no data is copied
    SparseMat_& operator = (const SparseMat_& m);
    //! converts dense matrix to the sparse form
    SparseMat_& operator = (const Mat& m);

    //! makes full copy of the matrix. All the elements are duplicated
    CV_NODISCARD_STD SparseMat_ clone() const;
    //! equivalent to cv::SparseMat::create(dims, _sizes, DataType<_Tp>::type)
    void create(int dims, const int* _sizes);
    //! converts sparse matrix to the old-style CvSparseMat. All the elements are copied
    //operator CvSparseMat*() const;

    //! returns type of the matrix elements
    int type() const;
    //! returns depth of the matrix elements
    int depth() const;
    //! returns the number of channels in each matrix element
    int channels() const;

    //! equivalent to SparseMat::ref<_Tp>(i0, hashval)
    _Tp& ref(int i0, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(i0, i1, hashval)
    _Tp& ref(int i0, int i1, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(i0, i1, i2, hashval)
    _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(idx, hashval)
    _Tp& ref(const int* idx, size_t* hashval=0);

    //! equivalent to SparseMat::value<_Tp>(i0, hashval)
    _Tp operator()(int i0, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(i0, i1, hashval)
    _Tp operator()(int i0, int i1, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(i0, i1, i2, hashval)
    _Tp operator()(int i0, int i1, int i2, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(idx, hashval)
    _Tp operator()(const int* idx, size_t* hashval=0) const;

    //! returns sparse matrix iterator pointing to the first sparse matrix element
    SparseMatIterator_<_Tp> begin();
    //! returns read-only sparse matrix iterator pointing to the first sparse matrix element
    SparseMatConstIterator_<_Tp> begin() const;
    //! returns sparse matrix iterator pointing to the element following the last sparse matrix element
    SparseMatIterator_<_Tp> end();
    //! returns read-only sparse matrix iterator pointing to the element following the last sparse matrix element
    SparseMatConstIterator_<_Tp> end() const;
};



////////////////////////////////// MatConstIterator //////////////////////////////////

class CV_EXPORTS MatConstIterator
{
public:
    typedef uchar* value_type;
    typedef ptrdiff_t difference_type;
    typedef const uchar** pointer;
    typedef uchar* reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator(const Mat* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, const int* _idx);
    //! copy constructor
    MatConstIterator(const MatConstIterator& it);

    //! copy operator
    MatConstIterator& operator = (const MatConstIterator& it);
    //! returns the current matrix element
    const uchar* operator *() const;
    //! returns the i-th matrix element, relative to the current
    const uchar* operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator& operator --();
    //! decrements the iterator
    MatConstIterator operator --(int);
    //! increments the iterator
    MatConstIterator& operator ++();
    //! increments the iterator
    MatConstIterator operator ++(int);
    //! returns the current iterator position
    Point pos() const;
    //! returns the current iterator position
    void pos(int* _idx) const;

    ptrdiff_t lpos() const;
    void seek(ptrdiff_t ofs, bool relative = false);
    void seek(const int* _idx, bool relative = false);

    const Mat* m;
    size_t elemSize;
    const uchar* ptr;
    const uchar* sliceStart;
    const uchar* sliceEnd;
};



////////////////////////////////// MatConstIterator_ /////////////////////////////////

/** @brief Matrix read-only iterator
 */
template<typename _Tp>
class MatConstIterator_ : public MatConstIterator
{
public:
    typedef _Tp value_type;
    typedef ptrdiff_t difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatConstIterator_(const MatConstIterator_& it);

    //! copy operator
    MatConstIterator_& operator = (const MatConstIterator_& it);
    //! returns the current matrix element
    const _Tp& operator *() const;
    //! returns the i-th matrix element, relative to the current
    const _Tp& operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator_& operator --();
    //! decrements the iterator
    MatConstIterator_ operator --(int);
    //! increments the iterator
    MatConstIterator_& operator ++();
    //! increments the iterator
    MatConstIterator_ operator ++(int);
    //! returns the current iterator position
    Point pos() const;
};



//////////////////////////////////// MatIterator_ ////////////////////////////////////

/** @brief Matrix read-write iterator
*/
template<typename _Tp>
class MatIterator_ : public MatConstIterator_<_Tp>
{
public:
    typedef _Tp* pointer;
    typedef _Tp& reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! the default constructor
    MatIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatIterator_(Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatIterator_(const MatIterator_& it);
    //! copy operator
    MatIterator_& operator = (const MatIterator_<_Tp>& it );

    //! returns the current matrix element
    _Tp& operator *() const;
    //! returns the i-th matrix element, relative to the current
    _Tp& operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatIterator_& operator --();
    //! decrements the iterator
    MatIterator_ operator --(int);
    //! increments the iterator
    MatIterator_& operator ++();
    //! increments the iterator
    MatIterator_ operator ++(int);
};



/////////////////////////////// SparseMatConstIterator ///////////////////////////////

/**  @brief Read-Only Sparse Matrix Iterator.

 Here is how to use the iterator to compute the sum of floating-point sparse matrix elements:

 \code
 SparseMatConstIterator it = m.begin(), it_end = m.end();
 double s = 0;
 CV_Assert( m.type() == CV_32F );
 for( ; it != it_end; ++it )
    s += it.value<float>();
 \endcode
*/
class CV_EXPORTS SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatConstIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator(const SparseMatConstIterator& it);

    //! the assignment operator
    SparseMatConstIterator& operator = (const SparseMatConstIterator& it);

    //! template method returning the current matrix element
    template<typename _Tp> const _Tp& value() const;
    //! returns the current node of the sparse matrix. it.node->idx is the current element index
    const SparseMat::Node* node() const;

    //! moves iterator to the previous element
    SparseMatConstIterator& operator --();
    //! moves iterator to the previous element
    SparseMatConstIterator operator --(int);
    //! moves iterator to the next element
    SparseMatConstIterator& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator operator ++(int);

    //! moves iterator to the element after the last element
    void seekEnd();

    const SparseMat* m;
    size_t hashidx;
    uchar* ptr;
};



////////////////////////////////// SparseMatIterator /////////////////////////////////

/** @brief  Read-write Sparse Matrix Iterator

 The class is similar to cv::SparseMatConstIterator,
 but can be used for in-place modification of the matrix elements.
*/
class CV_EXPORTS SparseMatIterator : public SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator(SparseMat* _m);
    //! the full constructor setting the iterator to the specified sparse matrix element
    SparseMatIterator(SparseMat* _m, const int* idx);
    //! the copy constructor
    SparseMatIterator(const SparseMatIterator& it);

    //! the assignment operator
    SparseMatIterator& operator = (const SparseMatIterator& it);
    //! returns read-write reference to the current sparse matrix element
    template<typename _Tp> _Tp& value() const;
    //! returns pointer to the current sparse matrix node. it.node->idx is the index of the current element (do not modify it!)
    SparseMat::Node* node() const;

    //! moves iterator to the next element
    SparseMatIterator& operator ++();
    //! moves iterator to the next element
    SparseMatIterator operator ++(int);
};



/////////////////////////////// SparseMatConstIterator_ //////////////////////////////

/** @brief  Template Read-Only Sparse Matrix Iterator Class.

 This is the derived from SparseMatConstIterator class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class SparseMatConstIterator_ : public SparseMatConstIterator
{
public:

    typedef std::forward_iterator_tag iterator_category;

    //! the default constructor
    SparseMatConstIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator_(const SparseMat_<_Tp>* _m);
    SparseMatConstIterator_(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator_(const SparseMatConstIterator_& it);

    //! the assignment operator
    SparseMatConstIterator_& operator = (const SparseMatConstIterator_& it);
    //! the element access operator
    const _Tp& operator *() const;

    //! moves iterator to the next element
    SparseMatConstIterator_& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator_ operator ++(int);
};



///////////////////////////////// SparseMatIterator_ /////////////////////////////////

/** @brief  Template Read-Write Sparse Matrix Iterator Class.

 This is the derived from cv::SparseMatConstIterator_ class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class SparseMatIterator_ : public SparseMatConstIterator_<_Tp>
{
public:

    typedef std::forward_iterator_tag iterator_category;

    //! the default constructor
    SparseMatIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator_(SparseMat_<_Tp>* _m);
    SparseMatIterator_(SparseMat* _m);
    //! the copy constructor
    SparseMatIterator_(const SparseMatIterator_& it);

    //! the assignment operator
    SparseMatIterator_& operator = (const SparseMatIterator_& it);
    //! returns the reference to the current element
    _Tp& operator *() const;

    //! moves the iterator to the next element
    SparseMatIterator_& operator ++();
    //! moves the iterator to the next element
    SparseMatIterator_ operator ++(int);
};



/////////////////////////////////// NAryMatIterator //////////////////////////////////

/** @brief n-ary multi-dimensional array iterator.

Use the class to implement unary, binary, and, generally, n-ary element-wise operations on
multi-dimensional arrays. Some of the arguments of an n-ary function may be continuous arrays, some
may be not. It is possible to use conventional MatIterator 's for each array but incrementing all of
the iterators after each small operations may be a big overhead. In this case consider using
NAryMatIterator to iterate through several matrices simultaneously as long as they have the same
geometry (dimensionality and all the dimension sizes are the same). On each iteration `it.planes[0]`,
`it.planes[1]`,... will be the slices of the corresponding matrices.

The example below illustrates how you can compute a normalized and threshold 3D color histogram:
@code
    void computeNormalizedColorHist(const Mat& image, Mat& hist, int N, double minProb)
    {
        const int histSize[] = {N, N, N};

        // make sure that the histogram has a proper size and type
        hist.create(3, histSize, CV_32F);

        // and clear it
        hist = Scalar(0);

        // the loop below assumes that the image
        // is a 8-bit 3-channel. check it.
        CV_Assert(image.type() == CV_8UC3);
        MatConstIterator_<Vec3b> it = image.begin<Vec3b>(),
                                 it_end = image.end<Vec3b>();
        for( ; it != it_end; ++it )
        {
            const Vec3b& pix = *it;
            hist.at<float>(pix[0]*N/256, pix[1]*N/256, pix[2]*N/256) += 1.f;
        }

        minProb *= image.rows*image.cols;

        // initialize iterator (the style is different from STL).
        // after initialization the iterator will contain
        // the number of slices or planes the iterator will go through.
        // it simultaneously increments iterators for several matrices
        // supplied as a null terminated list of pointers
        const Mat* arrays[] = {&hist, 0};
        Mat planes[1];
        NAryMatIterator itNAry(arrays, planes, 1);
        double s = 0;
        // iterate through the matrix. on each iteration
        // itNAry.planes[i] (of type Mat) will be set to the current plane
        // of the i-th n-dim matrix passed to the iterator constructor.
        for(int p = 0; p < itNAry.nplanes; p++, ++itNAry)
        {
            threshold(itNAry.planes[0], itNAry.planes[0], minProb, 0, THRESH_TOZERO);
            s += sum(itNAry.planes[0])[0];
        }

        s = 1./s;
        itNAry = NAryMatIterator(arrays, planes, 1);
        for(int p = 0; p < itNAry.nplanes; p++, ++itNAry)
            itNAry.planes[0] *= s;
    }
@endcode
 */
class CV_EXPORTS NAryMatIterator
{
public:
    //! the default constructor
    NAryMatIterator();
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, uchar** ptrs, int narrays=-1);
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, Mat* planes, int narrays=-1);
    //! the separate iterator initialization method
    void init(const Mat** arrays, Mat* planes, uchar** ptrs, int narrays=-1);

    //! proceeds to the next plane of every iterated matrix
    NAryMatIterator& operator ++();
    //! proceeds to the next plane of every iterated matrix (postfix increment operator)
    NAryMatIterator operator ++(int);

    //! the iterated arrays
    const Mat** arrays;
    //! the current planes
    Mat* planes;
    //! data pointers
    uchar** ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;
protected:
    int iterdepth;
    size_t idx;
};



///////////////////////////////// Matrix Expressions /////////////////////////////////

class CV_EXPORTS MatOp
{
public:
    MatOp();
    virtual ~MatOp();

    virtual bool elementWise(const MatExpr& expr) const;
    virtual void assign(const MatExpr& expr, Mat& m, int type=-1) const = 0;
    virtual void roi(const MatExpr& expr, const Range& rowRange,
                     const Range& colRange, MatExpr& res) const;
    virtual void diag(const MatExpr& expr, int d, MatExpr& res) const;
    virtual void augAssignAdd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignSubtract(const MatExpr& expr, Mat& m) const;
    virtual void augAssignMultiply(const MatExpr& expr, Mat& m) const;
    virtual void augAssignDivide(const MatExpr& expr, Mat& m) const;
    virtual void augAssignAnd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignOr(const MatExpr& expr, Mat& m) const;
    virtual void augAssignXor(const MatExpr& expr, Mat& m) const;

    virtual void add(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void add(const MatExpr& expr1, const Scalar& s, MatExpr& res) const;

    virtual void subtract(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void subtract(const Scalar& s, const MatExpr& expr, MatExpr& res) const;

    virtual void multiply(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void multiply(const MatExpr& expr1, double s, MatExpr& res) const;

    virtual void divide(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void divide(double s, const MatExpr& expr, MatExpr& res) const;

    virtual void abs(const MatExpr& expr, MatExpr& res) const;

    virtual void transpose(const MatExpr& expr, MatExpr& res) const;
    virtual void matmul(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void invert(const MatExpr& expr, int method, MatExpr& res) const;

    virtual Size size(const MatExpr& expr) const;
    virtual int type(const MatExpr& expr) const;
};

/** @brief Matrix expression representation
@anchor MatrixExpressions
This is a list of implemented matrix operations that can be combined in arbitrary complex
expressions (here A, B stand for matrices ( cv::Mat ), s for a cv::Scalar, alpha for a
real-valued scalar ( double )):
-   Addition, subtraction, negation: `A+B`, `A-B`, `A+s`, `A-s`, `s+A`, `s-A`, `-A`
-   Scaling: `A*alpha`
-   Per-element multiplication and division: `A.mul(B)`, `A/B`, `alpha/A`
-   Matrix multiplication: `A*B`
-   Transposition: `A.t()` (means A<sup>T</sup>)
-   Matrix inversion and pseudo-inversion, solving linear systems and least-squares problems:
    `A.inv([method]) (~ A<sup>-1</sup>)`,   `A.inv([method])*B (~ X: AX=B)`
-   Comparison: `A cmpop B`, `A cmpop alpha`, `alpha cmpop A`, where *cmpop* is one of
  `>`, `>=`, `==`, `!=`, `<=`, `<`. The result of comparison is an 8-bit single channel mask whose
    elements are set to 255 (if the particular element or pair of elements satisfy the condition) or
    0.
-   Bitwise logical operations: `A logicop B`, `A logicop s`, `s logicop A`, `~A`, where *logicop* is one of
  `&`, `|`, `^`.
-   Element-wise minimum and maximum: cv::min(A, B), cv::min(A, alpha), cv::max(A, B), cv::max(A, alpha)
-   Element-wise absolute value: cv::abs(A)
-   Cross-product, dot-product: `A.cross(B)`, `A.dot(B)`
-   Any function of matrix or matrices and scalars that returns a matrix or a scalar, such as cv::norm,
    cv::mean, cv::sum, cv::countNonZero, cv::trace, cv::determinant, cv::repeat, and others.
-   Matrix initializers ( Mat::eye(), Mat::zeros(), Mat::ones() ), matrix comma-separated
    initializers, matrix constructors and operators that extract sub-matrices (see cv::Mat description).
-   Mat_<destination_type>() constructors to cast the result to the proper type.
@note Comma-separated initializers and probably some other operations may require additional
explicit Mat() or Mat_<T>() constructor calls to resolve a possible ambiguity.

Here are examples of matrix expressions:
@code
    // compute pseudo-inverse of A, equivalent to A.inv(DECOMP_SVD)
    SVD svd(A);
    Mat pinvA = svd.vt.t()*Mat::diag(1./svd.w)*svd.u.t();

    // compute the new vector of parameters in the Levenberg-Marquardt algorithm
    x -= (A.t()*A + lambda*Mat::eye(A.cols,A.cols,A.type())).inv(DECOMP_CHOLESKY)*(A.t()*err);

    // sharpen image using "unsharp mask" algorithm
    Mat blurred; double sigma = 1, threshold = 5, amount = 1;
    GaussianBlur(img, blurred, Size(), sigma, sigma);
    Mat lowContrastMask = abs(img - blurred) < threshold;
    Mat sharpened = img*(1+amount) + blurred*(-amount);
    img.copyTo(sharpened, lowContrastMask);
@endcode
*/
class CV_EXPORTS MatExpr
{
public:
    MatExpr();
    explicit MatExpr(const Mat& m);

    MatExpr(const MatOp* _op, int _flags, const Mat& _a = Mat(), const Mat& _b = Mat(),
            const Mat& _c = Mat(), double _alpha = 1, double _beta = 1, const Scalar& _s = Scalar());

    operator Mat() const;
    template<typename _Tp> operator Mat_<_Tp>() const;

    Size size() const;
    int type() const;

    MatExpr row(int y) const;
    MatExpr col(int x) const;
    MatExpr diag(int d = 0) const;
    MatExpr operator()( const Range& rowRange, const Range& colRange ) const;
    MatExpr operator()( const Rect& roi ) const;

    MatExpr t() const;
    MatExpr inv(int method = DECOMP_LU) const;
    MatExpr mul(const MatExpr& e, double scale=1) const;
    MatExpr mul(const Mat& m, double scale=1) const;

    Mat cross(const Mat& m) const;
    double dot(const Mat& m) const;

    void swap(MatExpr& b);

    const MatOp* op;
    int flags;

    Mat a, b, c;
    double alpha, beta;
    Scalar s;
};

//! @} core_basic

//! @relates cv::MatExpr
//! @{
CV_EXPORTS MatExpr operator + (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator + (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator + (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e1, const MatExpr& e2);
template<typename _Tp, int m, int n> static inline
MatExpr operator + (const Mat& a, const Matx<_Tp, m, n>& b) { return a + Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator + (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) + b; }

CV_EXPORTS MatExpr operator - (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator - (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator - (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e1, const MatExpr& e2);
template<typename _Tp, int m, int n> static inline
MatExpr operator - (const Mat& a, const Matx<_Tp, m, n>& b) { return a - Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator - (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) - b; }

CV_EXPORTS MatExpr operator - (const Mat& m);
CV_EXPORTS MatExpr operator - (const MatExpr& e);

CV_EXPORTS MatExpr operator * (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator * (const Mat& a, double s);
CV_EXPORTS MatExpr operator * (double s, const Mat& a);
CV_EXPORTS MatExpr operator * (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator * (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator * (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e1, const MatExpr& e2);
template<typename _Tp, int m, int n> static inline
MatExpr operator * (const Mat& a, const Matx<_Tp, m, n>& b) { return a * Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator * (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) * b; }

CV_EXPORTS MatExpr operator / (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator / (const Mat& a, double s);
CV_EXPORTS MatExpr operator / (double s, const Mat& a);
CV_EXPORTS MatExpr operator / (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator / (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator / (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e1, const MatExpr& e2);
template<typename _Tp, int m, int n> static inline
MatExpr operator / (const Mat& a, const Matx<_Tp, m, n>& b) { return a / Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator / (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) / b; }

CV_EXPORTS MatExpr operator < (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator < (const Mat& a, double s);
CV_EXPORTS MatExpr operator < (double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator < (const Mat& a, const Matx<_Tp, m, n>& b) { return a < Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator < (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) < b; }

CV_EXPORTS MatExpr operator <= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator <= (const Mat& a, double s);
CV_EXPORTS MatExpr operator <= (double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator <= (const Mat& a, const Matx<_Tp, m, n>& b) { return a <= Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator <= (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) <= b; }

CV_EXPORTS MatExpr operator == (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator == (const Mat& a, double s);
CV_EXPORTS MatExpr operator == (double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator == (const Mat& a, const Matx<_Tp, m, n>& b) { return a == Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator == (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) == b; }

CV_EXPORTS MatExpr operator != (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator != (const Mat& a, double s);
CV_EXPORTS MatExpr operator != (double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator != (const Mat& a, const Matx<_Tp, m, n>& b) { return a != Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator != (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) != b; }

CV_EXPORTS MatExpr operator >= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator >= (const Mat& a, double s);
CV_EXPORTS MatExpr operator >= (double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator >= (const Mat& a, const Matx<_Tp, m, n>& b) { return a >= Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator >= (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) >= b; }

CV_EXPORTS MatExpr operator > (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator > (const Mat& a, double s);
CV_EXPORTS MatExpr operator > (double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator > (const Mat& a, const Matx<_Tp, m, n>& b) { return a > Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator > (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) > b; }

CV_EXPORTS MatExpr operator & (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator & (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator & (const Scalar& s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator & (const Mat& a, const Matx<_Tp, m, n>& b) { return a & Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator & (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) & b; }

CV_EXPORTS MatExpr operator | (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator | (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator | (const Scalar& s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator | (const Mat& a, const Matx<_Tp, m, n>& b) { return a | Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator | (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) | b; }

CV_EXPORTS MatExpr operator ^ (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator ^ (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator ^ (const Scalar& s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr operator ^ (const Mat& a, const Matx<_Tp, m, n>& b) { return a ^ Mat(b); }
template<typename _Tp, int m, int n> static inline
MatExpr operator ^ (const Matx<_Tp, m, n>& a, const Mat& b) { return Mat(a) ^ b; }

CV_EXPORTS MatExpr operator ~(const Mat& m);

CV_EXPORTS MatExpr min(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr min(const Mat& a, double s);
CV_EXPORTS MatExpr min(double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr min (const Mat& a, const Matx<_Tp, m, n>& b) { return min(a, Mat(b)); }
template<typename _Tp, int m, int n> static inline
MatExpr min (const Matx<_Tp, m, n>& a, const Mat& b) { return min(Mat(a), b); }

CV_EXPORTS MatExpr max(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr max(const Mat& a, double s);
CV_EXPORTS MatExpr max(double s, const Mat& a);
template<typename _Tp, int m, int n> static inline
MatExpr max (const Mat& a, const Matx<_Tp, m, n>& b) { return max(a, Mat(b)); }
template<typename _Tp, int m, int n> static inline
MatExpr max (const Matx<_Tp, m, n>& a, const Mat& b) { return max(Mat(a), b); }

/** @brief Calculates an absolute value of each matrix element.

abs is a meta-function that is expanded to one of absdiff or convertScaleAbs forms:
- C = abs(A-B) is equivalent to `absdiff(A, B, C)`
- C = abs(A) is equivalent to `absdiff(A, Scalar::all(0), C)`
- C = `Mat_<Vec<uchar,n> >(abs(A*alpha + beta))` is equivalent to `convertScaleAbs(A, C, alpha,
beta)`

The output matrix has the same size and the same type as the input one except for the last case,
where C is depth=CV_8U .
@param m matrix.
@sa @ref MatrixExpressions, absdiff, convertScaleAbs
 */
CV_EXPORTS MatExpr abs(const Mat& m);
/** @overload
@param e matrix expression.
*/
CV_EXPORTS MatExpr abs(const MatExpr& e);
//! @} relates cv::MatExpr

} // cv

#include "opencv2/core/mat.inl.hpp"

#endif // OPENCV_CORE_MAT_HPP
