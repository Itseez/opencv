// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/mat.hpp"

namespace cv {

/*************************************************************************************************\
                                        Input/Output Array
\*************************************************************************************************/

Mat _InputArray::getMat_(int i) const
{
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == MAT )
    {
        const Mat* m = (const Mat*)obj;
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == UMAT )
    {
        const UMat* m = (const UMat*)obj;
        if( i < 0 )
            return m->getMat(accessFlags);
        return m->getMat(accessFlags).row(i);
    }

    if (k == MATX)
    {
        CV_Assert( i < 0 );
        return Mat(sz, flags, obj);
    }

    if (k == STD_VECTOR || k == STD_BOOL_VECTOR || k == STD_VECTOR_VECTOR ||
        k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT) {
      CV_Assert(ops != nullptr);
      return ops->getMat_(*this, i);
    }

    if( k == NONE )
        return Mat();

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = (const Mat*)obj;
        CV_Assert( 0 <= i && i < sz.height );

        return v[i];
    }
    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call mapHost/unmapHost methods for ogl::Buffer object");
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call download method for cuda::GpuMat object");
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );

        const cuda::HostMem* cuda_mem = (const cuda::HostMem*)obj;

        return cuda_mem->createMatHeader();
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

UMat _InputArray::getUMat(int i) const
{
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == UMAT )
    {
        const UMat* m = (const UMat*)obj;
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == STD_VECTOR_UMAT )
    {
        CV_Assert(ops != nullptr);
        return ops->getUMat(*this, i);
    }

    if( k == MAT )
    {
        const Mat* m = (const Mat*)obj;
        if( i < 0 )
            return m->getUMat(accessFlags);
        return m->row(i).getUMat(accessFlags);
    }

    return getMat(i).getUMat(accessFlags);
}

void _InputArray::getMatVector(std::vector<Mat>& mv) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        const Mat& m = *(const Mat*)obj;
        int n = (int)m.size[0];
        mv.resize(n);
        CV_Assert(m.dims >= 2);

        for( int i = 0; i < n; i++ )
            mv[i] = m.dims <= 2 ? Mat(1, m.cols, m.type(), (void*)m.ptr(i)) :
                Mat(m.dims-1, &m.size[1], m.type(), (void*)m.ptr(i), &m.step[1]);
        return;
    }

    if (k == MATX)
    {
        size_t n = sz.height, esz = CV_ELEM_SIZE(flags);
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = Mat(1, sz.width, CV_MAT_TYPE(flags), (uchar*)obj + esz*sz.width*i);
        return;
    }

    if (k == STD_VECTOR || k == STD_VECTOR_VECTOR || k == STD_VECTOR_MAT ||
        k == STD_VECTOR_UMAT) {
      CV_Assert(ops != nullptr);
      mv = ops->getMatVector(*this);
      return;
    }

    if( k == NONE )
    {
        mv.clear();
        return;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = (const Mat*)obj;
        size_t n = sz.height;
        mv.resize(n);

        for( size_t i = 0; i < n; i++ )
            mv[i] = v[i];
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _InputArray::getUMatVector(std::vector<UMat>& umv) const
{
    _InputArray::KindFlag k = kind();
    AccessFlag accessFlags = flags & ACCESS_MASK;

    if( k == NONE )
    {
        umv.clear();
        return;
    }

    if (k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT) {
      CV_Assert(ops != nullptr);
      umv = ops->getUMatVector(*this);
      return;
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* v = (const Mat*)obj;
        size_t n = sz.height;
        umv.resize(n);

        for( size_t i = 0; i < n; i++ )
            umv[i] = v[i].getUMat(accessFlags);
        return;
    }
    if( k == UMAT )
    {
        UMat& v = *(UMat*)obj;
        umv.resize(1);
        umv[0] = v;
        return;
    }
    if( k == MAT )
    {
        Mat& v = *(Mat*)obj;
        umv.resize(1);
        umv[0] = v.getUMat(accessFlags);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

cuda::GpuMat _InputArray::getGpuMat() const
{
#ifdef HAVE_CUDA
    _InputArray::KindFlag k = kind();

    if (k == CUDA_GPU_MAT)
    {
        const cuda::GpuMat* d_mat = (const cuda::GpuMat*)obj;
        return *d_mat;
    }

    if (k == CUDA_HOST_MEM)
    {
        const cuda::HostMem* cuda_mem = (const cuda::HostMem*)obj;
        return cuda_mem->createGpuMatHeader();
    }

    if (k == OPENGL_BUFFER)
    {
        CV_Error(cv::Error::StsNotImplemented, "You should explicitly call mapDevice/unmapDevice methods for ogl::Buffer object");
    }

    if (k == NONE)
        return cuda::GpuMat();

    CV_Error(cv::Error::StsNotImplemented, "getGpuMat is available only for cuda::GpuMat and cuda::HostMem");
#else
    CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
}
void _InputArray::getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const
{
#ifdef HAVE_CUDA
    _InputArray::KindFlag k = kind();
    if (k == STD_VECTOR_CUDA_GPU_MAT)
    {
        gpumv = *(std::vector<cuda::GpuMat>*)obj;
    }
#else
    CV_UNUSED(gpumv);
    CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
}
ogl::Buffer _InputArray::getOGlBuffer() const
{
    _InputArray::KindFlag k = kind();

    CV_Assert(k == OPENGL_BUFFER);

    const ogl::Buffer* gl_buf = (const ogl::Buffer*)obj;
    return *gl_buf;
}

_InputArray::KindFlag _InputArray::kind() const
{
    KindFlag k = flags & KIND_MASK;
#if CV_VERSION_MAJOR < 5
    CV_DbgAssert(k != EXPR);
    CV_DbgAssert(k != STD_ARRAY);
#endif
    return k;
}

int _InputArray::rows(int i) const
{
    return size(i).height;
}

int _InputArray::cols(int i) const
{
    return size(i).width;
}

Size _InputArray::size(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->size();
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->size();
    }

    if (k == MATX)
    {
        CV_Assert( i < 0 );
        return sz;
    }
    if (k == STD_VECTOR || k == STD_BOOL_VECTOR || k == STD_VECTOR_VECTOR ||
        k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT ||
        k == STD_VECTOR_CUDA_GPU_MAT) {
      CV_Assert(ops != nullptr);
      return ops->size(*this, i);
    }

    if( k == NONE )
        return Size();

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return sz.height==0 ? Size() : Size(sz.height, 1);
        CV_Assert( i < sz.height );

        return vv[i].size();
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        const ogl::Buffer* buf = (const ogl::Buffer*)obj;
        return buf->size();
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        const cuda::GpuMat* d_mat = (const cuda::GpuMat*)obj;
        return d_mat->size();
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );
        const cuda::HostMem* cuda_mem = (const cuda::HostMem*)obj;
        return cuda_mem->size();
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

int _InputArray::sizend(int* arrsz, int i) const
{
    int j, d = 0;
    _InputArray::KindFlag k = kind();

    const bool is_nd_vector = (k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT) && (i >= 0);
    const bool is_1d_vector = k == STD_VECTOR && i < 0;

    if( k == NONE )
        ;
    else if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat& m = *(const Mat*)obj;
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( k == UMAT )
    {
        CV_Assert( i < 0 );
        const UMat& m = *(const UMat*)obj;
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if( is_nd_vector || is_1d_vector )
    {
        CV_Assert(ops != nullptr);
        return ops->sizend(*this, arrsz, i);
    }
    else if( k == STD_ARRAY_MAT && i >= 0 )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert( i < sz.height );
        const Mat& m = vv[i];
        d = m.dims;
        if(arrsz)
            for(j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else
    {
        CV_CheckLE(dims(i), 2, "Not supported");
        Size sz2d = size(i);
        d = 2;
        if(arrsz)
        {
            arrsz[0] = sz2d.height;
            arrsz[1] = sz2d.width;
        }
    }

    return d;
}

bool _InputArray::empty(int i) const
{
    if (i >= 0) {
        switch (_InputArray::KindFlag k = kind(); k) {
        case STD_VECTOR_MAT:
        case STD_VECTOR_UMAT:
        case STD_VECTOR_VECTOR:
            CV_Assert(ops != nullptr);
            return ops->empty(*this, i);
        default:
            CV_Error(Error::StsNotImplemented, "");
        }
    }
    return empty();
}

MatShape _InputArray::shape(int i) const
{
    int sizes[CV_MAX_DIM];
    int dims = sizend(sizes, i);

    if (dims == 0 && empty(i))
        return MatShape();
    return MatShape(dims, sizes);
}

bool _InputArray::sameSize(const _InputArray& arr) const
{
    _InputArray::KindFlag k1 = kind(), k2 = arr.kind();
    Size sz1;

    if( k1 == MAT )
    {
        const Mat* m = ((const Mat*)obj);
        if( k2 == MAT )
            return m->size == ((const Mat*)arr.obj)->size;
        if( k2 == UMAT )
            return m->size == ((const UMat*)arr.obj)->size;
        if( m->dims > 2 )
            return false;
        sz1 = m->size();
    }
    else if( k1 == UMAT )
    {
        const UMat* m = ((const UMat*)obj);
        if( k2 == MAT )
            return m->size == ((const Mat*)arr.obj)->size;
        if( k2 == UMAT )
            return m->size == ((const UMat*)arr.obj)->size;
        if( m->dims > 2 )
            return false;
        sz1 = m->size();
    }
    else
        sz1 = size();
    if( arr.dims() > 2 )
        return false;
    return sz1 == arr.size();
}

int _InputArray::dims(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->dims;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->dims;
    }

    if (k == MATX)
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        CV_Assert( i < 0 );
        return 1;
    }

    if( k == NONE )
        return 0;

    if (k == STD_VECTOR_VECTOR || k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT) {
      CV_Assert(ops != nullptr);
      return ops->dims(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return 1;
        CV_Assert( i < sz.height );

        return vv[i].dims;
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    if( k == CUDA_HOST_MEM )
    {
        CV_Assert( i < 0 );
        return 2;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

size_t _InputArray::total(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->total();
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->total();
    }

    if( k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT )
    {
        CV_Assert(ops != nullptr);
        return ops->total(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( i < 0 )
            return sz.height;

        CV_Assert( i < sz.height );
        return vv[i].total();
    }

    return size(i).area();
}

int _InputArray::type(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return ((const Mat*)obj)->type();

    if( k == UMAT )
        return ((const UMat*)obj)->type();

    if( k == MATX || k == STD_VECTOR || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return CV_MAT_TYPE(flags);

    if( k == NONE )
        return -1;

    if (k == STD_VECTOR_UMAT || k == STD_VECTOR_MAT || k == STD_VECTOR_CUDA_GPU_MAT) {
      CV_Assert(ops != nullptr);
      return ops->type(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        if( sz.height == 0 )
        {
            CV_Assert((flags & FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }
        CV_Assert( i < sz.height );
        return vv[i >= 0 ? i : 0].type();
    }

    if( k == OPENGL_BUFFER )
        return ((const ogl::Buffer*)obj)->type();

    if( k == CUDA_GPU_MAT )
        return ((const cuda::GpuMat*)obj)->type();

    if( k == CUDA_HOST_MEM )
        return ((const cuda::HostMem*)obj)->type();

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

int _InputArray::depth(int i) const
{
    return CV_MAT_DEPTH(type(i));
}

int _InputArray::channels(int i) const
{
    return CV_MAT_CN(type(i));
}

bool _InputArray::empty() const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return ((const Mat*)obj)->empty();

    if( k == UMAT )
        return ((const UMat*)obj)->empty();

    if (k == MATX)
        return false;

    if (k == STD_VECTOR || k == STD_BOOL_VECTOR || k == STD_VECTOR_VECTOR || k == STD_VECTOR_MAT ||
        k == STD_VECTOR_UMAT || k == STD_VECTOR_CUDA_GPU_MAT) {
        CV_Assert(ops != nullptr);
        return ops->empty(*this);
    }

    if( k == NONE )
        return true;

    if( k == STD_ARRAY_MAT )
    {
        return sz.height == 0;
    }

    if( k == OPENGL_BUFFER )
        return ((const ogl::Buffer*)obj)->empty();

    if( k == CUDA_GPU_MAT )
        return ((const cuda::GpuMat*)obj)->empty();

    if( k == CUDA_HOST_MEM )
        return ((const cuda::HostMem*)obj)->empty();

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

bool _InputArray::isContinuous(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return i < 0 ? ((const Mat*)obj)->isContinuous() : true;

    if( k == UMAT )
        return i < 0 ? ((const UMat*)obj)->isContinuous() : true;

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return true;

    if( k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT )
    {
        CV_Assert(ops != nullptr);
        return ops->isContinuous(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert(i >= 0 && i < sz.height);
        return vv[i].isContinuous();
    }

    if( k == CUDA_GPU_MAT )
      return i < 0 ? ((const cuda::GpuMat*)obj)->isContinuous() : true;

    CV_Error(cv::Error::StsNotImplemented, "Unknown/unsupported array type");
}

bool _InputArray::isSubmatrix(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
        return i < 0 ? ((const Mat*)obj)->isSubmatrix() : false;

    if( k == UMAT )
        return i < 0 ? ((const UMat*)obj)->isSubmatrix() : false;

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return false;

    if( k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT )
    {
        CV_Assert(ops != nullptr);
        return ops->isSubmatrix(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert(i >= 0 && i < sz.height);
        return vv[i].isSubmatrix();
    }

    CV_Error(cv::Error::StsNotImplemented, "");
}

size_t _InputArray::offset(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        const Mat * const m = ((const Mat*)obj);
        return (size_t)(m->ptr() - m->datastart);
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->offset;
    }

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return 0;

    if( k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT || k == STD_VECTOR_CUDA_GPU_MAT )
    {
        CV_Assert(ops != nullptr);
        return ops->offset(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert( i >= 0 && i < sz.height );
        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        const cuda::GpuMat * const m = ((const cuda::GpuMat*)obj);
        return (size_t)(m->data - m->datastart);
    }

    CV_Error(Error::StsNotImplemented, "");
}

size_t _InputArray::step(int i) const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->step;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        return ((const UMat*)obj)->step;
    }

    if( k == MATX || k == STD_VECTOR ||
        k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR )
        return 0;

    if( k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT || k == STD_VECTOR_CUDA_GPU_MAT )
    {
        CV_Assert(ops != nullptr);
        return ops->step(*this, i);
    }

    if( k == STD_ARRAY_MAT )
    {
        const Mat* vv = (const Mat*)obj;
        CV_Assert( i >= 0 && i < sz.height );
        return vv[i].step;
    }

    if( k == CUDA_GPU_MAT )
    {
        CV_Assert( i < 0 );
        return ((const cuda::GpuMat*)obj)->step;
    }

    CV_Error(Error::StsNotImplemented, "");
}

void _InputArray::copyTo(const _OutputArray& arr) const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        arr.release();
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        Mat m = getMat();
        m.copyTo(arr);
    }
    else if( k == UMAT )
        ((UMat*)obj)->copyTo(arr);
#ifdef HAVE_CUDA
    else if (k == CUDA_GPU_MAT)
        ((cuda::GpuMat*)obj)->copyTo(arr);
#endif
    else
        CV_Error(Error::StsNotImplemented, "");
}

void _InputArray::copyTo(const _OutputArray& arr, const _InputArray & mask) const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        arr.release();
    else if( k == MAT || k == MATX || k == STD_VECTOR || k == STD_BOOL_VECTOR )
    {
        Mat m = getMat();
        m.copyTo(arr, mask);
    }
    else if( k == UMAT )
        ((UMat*)obj)->copyTo(arr, mask);
#ifdef HAVE_CUDA
    else if (k == CUDA_GPU_MAT)
        ((cuda::GpuMat*)obj)->copyTo(arr, mask);
#endif
    else
        CV_Error(Error::StsNotImplemented, "");
}

bool _OutputArray::fixedSize() const
{
    return (flags & FIXED_SIZE) == FIXED_SIZE;
}

bool _OutputArray::fixedType() const
{
    return (flags & FIXED_TYPE) == FIXED_TYPE;
}

void _OutputArray::create(Size _sz, int mtype, int i, bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    _InputArray::KindFlag k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((Mat*)obj)->size.operator()() == _sz);
        CV_Assert(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == UMAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((UMat*)obj)->size.operator()() == _sz);
        CV_Assert(!fixedType() || ((UMat*)obj)->type() == mtype);
        ((UMat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == CUDA_GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::GpuMat*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((cuda::GpuMat*)obj)->type() == mtype);
#ifdef HAVE_CUDA
        ((cuda::GpuMat*)obj)->create(_sz, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((ogl::Buffer*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((ogl::Buffer*)obj)->type() == mtype);
#ifdef HAVE_OPENGL
        ((ogl::Buffer*)obj)->create(_sz, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
#endif
    }
    if( k == CUDA_HOST_MEM && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::HostMem*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((cuda::HostMem*)obj)->type() == mtype);
#ifdef HAVE_CUDA
        ((cuda::HostMem*)obj)->create(_sz, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    int sizes[] = {_sz.height, _sz.width};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int _rows, int _cols, int mtype, int i, bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    _InputArray::KindFlag k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((Mat*)obj)->size.operator()() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_rows, _cols, mtype);
        return;
    }
    if( k == UMAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((UMat*)obj)->size.operator()() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((UMat*)obj)->type() == mtype);
        ((UMat*)obj)->create(_rows, _cols, mtype);
        return;
    }
    if( k == CUDA_GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::GpuMat*)obj)->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((cuda::GpuMat*)obj)->type() == mtype);
#ifdef HAVE_CUDA
        ((cuda::GpuMat*)obj)->create(_rows, _cols, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((ogl::Buffer*)obj)->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((ogl::Buffer*)obj)->type() == mtype);
#ifdef HAVE_OPENGL
        ((ogl::Buffer*)obj)->create(_rows, _cols, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
#endif
    }
    if( k == CUDA_HOST_MEM && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((cuda::HostMem*)obj)->size() == Size(_cols, _rows));
        CV_Assert(!fixedType() || ((cuda::HostMem*)obj)->type() == mtype);
#ifdef HAVE_CUDA
        ((cuda::HostMem*)obj)->create(_rows, _cols, mtype);
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    int sizes[] = {_rows, _cols};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int d, const int* sizes, int mtype, int i,
                          bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    int size0 = d > 0 ? sizes[0] : 1, size1 = d > 1 ? sizes[1] : 1;
    _InputArray::KindFlag k = kind();
    mtype = CV_MAT_TYPE(mtype);

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        Mat& m = *(Mat*)obj;
        CV_Assert(!(m.empty() && fixedType() && fixedSize()) && "Can't reallocate empty Mat with locked layout (probably due to misused 'const' modifier)");
        if (!m.empty() && d <= 2 && m.dims <= 2 &&
            m.type() == mtype &&
            ((m.rows == size0 && m.cols == size1) ||
            (allowTransposed && m.rows == size1 && m.cols == size0 && m.isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_DEPTH(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_CheckTypeEQ(m.type(), CV_MAT_TYPE(mtype), "Can't reallocate Mat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m.dims, d, "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m.size[j], sizes[j], "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
        }
        m.create(d, sizes, mtype);
        return;
    }

    if( k == UMAT )
    {
        CV_Assert( i < 0 );
        UMat& m = *(UMat*)obj;
        CV_Assert(!(m.empty() && fixedType() && fixedSize()) && "Can't reallocate empty UMat with locked layout (probably due to misused 'const' modifier)");
        if (!m.empty() && d <= 2 && m.dims <= 2 &&
            m.type() == mtype &&
            ((m.rows == size0 && m.cols == size1) ||
            (allowTransposed && m.rows == size1 && m.cols == size0 && m.isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_CheckTypeEQ(m.type(), CV_MAT_TYPE(mtype), "Can't reallocate UMat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m.dims, d, "Can't reallocate UMat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m.size[j], sizes[j], "Can't reallocate UMat with locked size (probably due to misused 'const' modifier)");
        }
        m.create(d, sizes, mtype);
        return;
    }

    if( k == MATX )
    {
        CV_Assert( i < 0 );
        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0) );
        CV_CheckLE(d, 2, "");
        Size requested_size(d == 2 ? sizes[1] : 1, d >= 1 ? sizes[0] : 1);
        if (sz.width == 1 || sz.height == 1)
        {
            // NB: 1D arrays assume allowTransposed=true (see #4159)
            int total_1d = std::max(sz.width, sz.height);
            CV_Check(requested_size, std::max(requested_size.width, requested_size.height) == total_1d, "");
        }
        else
        {
            if (!allowTransposed)
            {
                CV_CheckEQ(requested_size, sz, "");
            }
            else
            {
                CV_Check(requested_size,
                        (requested_size == sz || (requested_size.height == sz.width && requested_size.width == sz.height)),
                        "");
            }
        }
        return;
    }

    if( k == STD_VECTOR || k == STD_VECTOR_VECTOR || k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT )
    {
        CV_Assert(ops != nullptr);
        return ops->create(*this, d, sizes, mtype, i, allowTransposed, fixedDepthMask);
    }

    if( k == NONE )
    {
        CV_Error(cv::Error::StsNullPtr, "create() called for the missing output array" );
    }

    if( k == STD_ARRAY_MAT )
    {
        Mat* v = (Mat*)obj;

        if( i < 0 )
        {
            CV_Assert( d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
            size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0, len0 = sz.height;

            CV_Assert(len == len0);
            if( fixedType() )
            {
                int _type = CV_MAT_TYPE(flags);
                for( size_t j = len0; j < len; j++ )
                {
                    if( v[j].type() == _type )
                        continue;
                    CV_Assert( v[j].empty() );
                    v[j].flags = (v[j].flags & ~CV_MAT_TYPE_MASK) | _type;
                }
            }
            return;
        }

        CV_Assert( i < sz.height );
        Mat& m = v[i];

        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( d == 2 && m.dims == 2 && m.data &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }

        if(fixedSize())
        {
            CV_Assert(m.dims == d);
            for(int j = 0; j < d; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }

        m.create(d, sizes, mtype);
        return;
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _OutputArray::create(const MatShape& shape, int mtype, int i,
                          bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    if (shape.dims < 0) {
        release();
    } else {
        create(shape.dims, shape.p, mtype, i, allowTransposed, fixedDepthMask);
    }
}

void _OutputArray::createSameSize(const _InputArray& arr, int mtype) const
{
    int arrsz[CV_MAX_DIM], d = arr.sizend(arrsz);
    create(d, arrsz, mtype);
}

void _OutputArray::fit(int d, const int* sizes, int mtype, int i,
                       bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    int size0 = d > 0 ? sizes[0] : 1, size1 = d > 1 ? sizes[1] : 1;
    _InputArray::KindFlag k = kind();
    mtype = CV_MAT_TYPE(mtype);

    if( (k == MAT && i < 0) || (k == STD_VECTOR_MAT && i >= 0) )
    {
        Mat* m;
        if (k == MAT)
            m = (Mat*)obj;
        else {
            std::vector<Mat>& v = *(std::vector<Mat>*)obj;
            CV_Assert((size_t)i < v.size());
            m = &v[i];
        }
        CV_Assert(!(m->empty() && fixedType() && fixedSize()) && "Can't reallocate empty Mat with locked layout (probably due to misused 'const' modifier)");
        if (!m->empty() && d <= 2 && m->dims <= 2 &&
            m->type() == mtype &&
            ((m->rows == size0 && m->cols == size1) ||
             (allowTransposed && m->rows == size1 && m->cols == size0 && m->isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m->channels() && ((1 << CV_MAT_DEPTH(flags)) & fixedDepthMask) != 0 )
                mtype = m->type();
            else
                CV_CheckTypeEQ(m->type(), CV_MAT_TYPE(mtype), "Can't reallocate Mat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m->dims, d, "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m->size[j], sizes[j], "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
        }
        m->fit(d, sizes, mtype);
        return;
    }

    if( (k == UMAT && i < 0) || (k == STD_VECTOR_UMAT && i >= 0) )
    {
        UMat* m;
        if (k == UMAT)
            m = (UMat*)obj;
        else {
            std::vector<UMat>& v = *(std::vector<UMat>*)obj;
            CV_Assert((size_t)i < v.size());
            m = &v[i];
        }
        CV_Assert(!(m->empty() && fixedType() && fixedSize()) && "Can't reallocate empty Mat with locked layout (probably due to misused 'const' modifier)");
        if (!m->empty() && d <= 2 && m->dims <= 2 &&
            m->type() == mtype &&
            ((m->rows == size0 && m->cols == size1) ||
             (allowTransposed && m->rows == size1 && m->cols == size0 && m->isContinuous())))
        {
            return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m->channels() && ((1 << CV_MAT_DEPTH(flags)) & fixedDepthMask) != 0 )
                mtype = m->type();
            else
                CV_CheckTypeEQ(m->type(), CV_MAT_TYPE(mtype), "Can't reallocate Mat with locked type (probably due to misused 'const' modifier)");
        }
        if(fixedSize())
        {
            CV_CheckEQ(m->dims, d, "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
            for(int j = 0; j < d; ++j)
                CV_CheckEQ(m->size[j], sizes[j], "Can't reallocate Mat with locked size (probably due to misused 'const' modifier)");
        }
        m->fit(d, sizes, mtype);
        return;
    }

    create(d, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::release() const
{
    CV_Assert(!fixedSize());

    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        ((Mat*)obj)->release();
        return;
    }

    if( k == UMAT )
    {
        ((UMat*)obj)->release();
        return;
    }

    if( k == CUDA_GPU_MAT )
    {
#ifdef HAVE_CUDA
        ((cuda::GpuMat*)obj)->release();
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }

    if( k == CUDA_HOST_MEM )
    {
#ifdef HAVE_CUDA
        ((cuda::HostMem*)obj)->release();
        return;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }

    if( k == OPENGL_BUFFER )
    {
#ifdef HAVE_OPENGL
        ((ogl::Buffer*)obj)->release();
        return;
#else
        CV_Error(Error::StsNotImplemented, "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
#endif
    }

    if( k == NONE )
        return;

    if( k == STD_VECTOR || k == STD_VECTOR_VECTOR || k == STD_VECTOR_MAT || k == STD_VECTOR_UMAT ||
        k == STD_VECTOR_CUDA_GPU_MAT )
    {
        CV_Assert(ops != nullptr);
        return ops->release(*this);
    }

    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

void _OutputArray::clear() const
{
    _InputArray::KindFlag k = kind();

    if( k == MAT )
    {
        CV_Assert(!fixedSize());
        ((Mat*)obj)->resize(0);
        return;
    }

    release();
}

bool _OutputArray::needed() const
{
    return kind() != NONE;
}

Mat& _OutputArray::getMatRef(int i) const
{
    _InputArray::KindFlag k = kind();
    if( i < 0 )
    {
        CV_Assert( k == MAT );
        return *(Mat*)obj;
    }

    CV_Assert( k == STD_VECTOR_MAT || k == STD_ARRAY_MAT );

    if( k == STD_VECTOR_MAT )
    {
        CV_Assert(ops != nullptr);
        return ops->getMatRef(*this, i);
    }
    else
    {
        Mat* v = (Mat*)obj;
        CV_Assert( 0 <= i && i < sz.height );
        return v[i];
    }
}

UMat& _OutputArray::getUMatRef(int i) const
{
    _InputArray::KindFlag k = kind();
    if( i < 0 )
    {
        CV_Assert( k == UMAT );
        return *(UMat*)obj;
    }
    else
    {
        CV_Assert( k == STD_VECTOR_UMAT );
        CV_Assert(ops != nullptr);
        return ops->getUMatRef(*this, i);
    }
}

cuda::GpuMat& _OutputArray::getGpuMatRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert( k == CUDA_GPU_MAT );
    return *(cuda::GpuMat*)obj;
}
std::vector<cuda::GpuMat>& _OutputArray::getGpuMatVecRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert(k == STD_VECTOR_CUDA_GPU_MAT);
    return *(std::vector<cuda::GpuMat>*)obj;
}

ogl::Buffer& _OutputArray::getOGlBufferRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert( k == OPENGL_BUFFER );
    return *(ogl::Buffer*)obj;
}

cuda::HostMem& _OutputArray::getHostMemRef() const
{
    _InputArray::KindFlag k = kind();
    CV_Assert( k == CUDA_HOST_MEM );
    return *(cuda::HostMem*)obj;
}

void _OutputArray::setTo(const _InputArray& arr, const _InputArray & mask) const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        ;
    else if (k == MAT || k == MATX || k == STD_VECTOR)
    {
        Mat m = getMat();
        m.setTo(arr, mask);
    }
    else if( k == UMAT )
        ((UMat*)obj)->setTo(arr, mask);
    else if( k == CUDA_GPU_MAT )
    {
#ifdef HAVE_CUDA
        Mat value = arr.getMat();
        CV_Assert( checkScalar(value, type(), arr.kind(), _InputArray::CUDA_GPU_MAT) );
        ((cuda::GpuMat*)obj)->setTo(Scalar(Vec<double, 4>(value.ptr<double>())), mask);
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
#endif
    }
    else
        CV_Error(Error::StsNotImplemented, "");
}

void _OutputArray::setZero() const
{
    _InputArray::KindFlag k = kind();

    if( k == NONE )
        ;
    else if (k == MAT || k == MATX || k == STD_VECTOR)
    {
        Mat m = getMat();
        m.setZero();
    }
    else
    {
        setTo(Scalar::all(0), noArray());
    }
}

void _OutputArray::assign(const UMat& u) const
{
    _InputArray::KindFlag k = kind();
    if (k == UMAT)
    {
        *(UMat*)obj = u;
    }
    else if (k == MAT)
    {
        u.copyTo(*(Mat*)obj); // TODO check u.getMat()
    }
    else if (k == MATX)
    {
        u.copyTo(getMat()); // TODO check u.getMat()
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::assign(const Mat& m) const
{
    _InputArray::KindFlag k = kind();
    if (k == UMAT)
    {
        m.copyTo(*(UMat*)obj); // TODO check m.getUMat()
    }
    else if (k == MAT)
    {
        *(Mat*)obj = m;
    }
    else if (k == MATX)
    {
        m.copyTo(getMat());
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::move(UMat& u) const
{
    if (fixedSize())
    {
        // TODO Performance warning
        assign(u);
        return;
    }
    int k = kind();
    if (k == UMAT)
    {
        *(UMat*)obj = std::move(u);
    }
    else if (k == MAT)
    {
        u.copyTo(*(Mat*)obj); // TODO check u.getMat()
        u.release();
    }
    else if (k == MATX)
    {
        u.copyTo(getMat()); // TODO check u.getMat()
        u.release();
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::move(Mat& m) const
{
    if (fixedSize())
    {
        // TODO Performance warning
        assign(m);
        return;
    }
    int k = kind();
    if (k == UMAT)
    {
        m.copyTo(*(UMat*)obj); // TODO check m.getUMat()
        m.release();
    }
    else if (k == MAT)
    {
        *(Mat*)obj = std::move(m);
    }
    else if (k == MATX)
    {
        m.copyTo(getMat());
        m.release();
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::assign(const std::vector<UMat>& v) const
{
    _InputArray::KindFlag k = kind();
    if (k == STD_VECTOR_UMAT || k == STD_VECTOR_MAT)
    {
        CV_Assert(ops != nullptr);
        ops->assign(*this, v);
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


void _OutputArray::assign(const std::vector<Mat>& v) const
{
    _InputArray::KindFlag k = kind();
    if (k == STD_VECTOR_UMAT || k == STD_VECTOR_MAT)
    {
        CV_Assert(ops != nullptr);
        ops->assign(*this, v);
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "");
    }
}


static _InputOutputArray _none;
InputOutputArray noArray() { return _none; }

template<>
[[gnu::visibility("default")]] Mat _ArrayOps<std::vector<cuda::GpuMat>>::getMat_(const _InputArray& self, const int i) const
{
    std::vector<cuda::GpuMat>& v = get(self.getObj());
    CV_Assert(i < 0);
    const int type = CV_MAT_TYPE(self.getFlags());
    const int width = static_cast<int>(v.size());
    return v.empty() ? Mat() : Mat(1, &width, type, v.data());
}

[[noreturn]] static void noCudaError() {
  CV_Error(
      Error::StsNotImplemented,
      "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
}

template<>
[[gnu::visibility("default")]] Size _ArrayOps<std::vector<cuda::GpuMat>>::size(const _InputArray& self, const int i) const
{
    if constexpr (!have_cuda) {
        noCudaError();
    } else {
        const std::vector<cuda::GpuMat>& v = get(self.getObj());
        if (i < 0) {
            return v.empty() ? Size() : Size(static_cast<int>(v.size()), 1);
        }

        const auto index = static_cast<std::size_t>(i);
        CV_Assert(index < v.size());
        return v[index].size();
    }
}

template<>
[[gnu::visibility("default")]] int _ArrayOps<std::vector<cuda::GpuMat>>::sizend(const _InputArray& self, int* const arraySize, const int i) const
{
    std::vector<cuda::GpuMat>& v = get(self.getObj());
    CV_Assert(i < 0);
    Size sz2d = Size(v.size(), 1);
    if (arraySize != nullptr) {
        arraySize[0] = sz2d.width;
    }

    return 1;
}

template<>
[[gnu::visibility("default")]] int _ArrayOps<std::vector<cuda::GpuMat>>::type(const _InputArray& self, const int i) const
{
    if constexpr (!have_cuda) {
      noCudaError();
    } else {
        std::vector<cuda::GpuMat>& v = get(self.getObj());
        if (v.empty()) {
            const int flags = self.getFlags();
            CV_Assert((flags & _InputArray::FIXED_TYPE) != 0);
            return CV_MAT_TYPE(flags);
        }

        CV_Assert(i < static_cast<int>(v.size()));
        return v[i >= 0 ? i : 0].type();
    }
}

template<>
[[gnu::visibility("default")]] std::size_t _ArrayOps<std::vector<cuda::GpuMat>>::offset(const _InputArray& self, const std::size_t i) const
{
    const std::vector<cuda::GpuMat>& v = get(self.getObj());
    CV_Assert(i < v.size());
    return static_cast<std::size_t>(v[i].data - v[i].datastart);
}

template<>
[[gnu::visibility("default")]] std::size_t _ArrayOps<std::vector<cuda::GpuMat>>::step(const _InputArray& self, const std::size_t i) const
{
    const std::vector<cuda::GpuMat>& v = get(self.getObj());
    CV_Assert(i < v.size());
    return v[i].step;
}

template<>
[[gnu::visibility("default")]]
void _ArrayOps<std::vector<cuda::GpuMat>>::create(const _OutputArray& arr,
                                                  const int d,
                                                  const int* const sizes,
                                                  int mtype,
                                                  const int i,
                                                  bool /*allowTransposed*/,
                                                  const _OutputArray::DepthMask fixedDepthMask) const
{
    std::vector<cuda::GpuMat>& v = get(arr.getObj());
    const int size0 = d > 0 ? sizes[0] : 1;
    const int size1 = d > 1 ? sizes[1] : 1;
    CV_Assert(d <= 2);
    CV_Assert(size0 == 1 || size1 == 1 || size0 * size1 == 0);

    const std::size_t len = size0 * size1 > 0 ? size0 + size1 - 1 : 0;

    CV_Assert(i < 0);
    const int type0 = CV_MAT_TYPE(arr.getFlags());
    CV_Assert(mtype == type0 || (CV_MAT_CN(mtype) == CV_MAT_CN(type0) &&
                                    ((1 << type0) & fixedDepthMask) != 0));
    v.resize(len);
}

template<>
[[gnu::visibility("default")]] void _ArrayOps<std::vector<cuda::GpuMat>>::release(const _OutputArray& self) const
{
    if constexpr (!have_cuda) {
        noCudaError();
    } else {
        get(self.getObj()).clear();
    }
}

} // cv::
