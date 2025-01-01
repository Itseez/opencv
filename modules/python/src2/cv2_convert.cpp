// must be defined before importing numpy headers
// https://numpy.org/doc/1.17/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL opencv_ARRAY_API

#include "cv2_convert.hpp"
#include "cv2_numpy.hpp"
#include "cv2_util.hpp"
#include "opencv2/core/utils/logger.hpp"

PyTypeObject* pyopencv_Mat_TypePtr = nullptr;

//======================================================================================================================

using namespace cv;

template <typename T>
static std::string pycv_dumpArray(const T* arr, int n)
{
    std::ostringstream out;
    out << "[";
    for (int i = 0; i < n; ++i)
        out << " " << arr[i];
    out << " ]";
    return out.str();
}

static inline std::string getArrayTypeName(PyArrayObject* arr)
{
    PyArray_Descr* dtype = PyArray_DESCR(arr);
    PySafeObject dtype_str(PyObject_Str(reinterpret_cast<PyObject*>(dtype)));
    if (!dtype_str)
    {
        // Fallback to typenum value
        return cv::format("%d", PyArray_TYPE(arr));
    }
    std::string type_name;
    if (!getUnicodeString(dtype_str, type_name))
    {
        // Failed to get string from bytes object - clear set TypeError and
        // fallback to typenum value
        PyErr_Clear();
        return cv::format("%d", PyArray_TYPE(arr));
    }
    return type_name;
}

//======================================================================================================================

// --- Mat

// special case, when the converter needs full ArgInfo structure
template<>
bool pyopencv_to(PyObject* o, Mat& m, const ArgInfo& info)
{
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &GetNumpyAllocator();
        return true;
    }

    if( PyInt_Check(o) )
    {
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        if ( info.arithm_op_src )
        {
            // Normally cv.XXX(x) means cv.XXX( (x, 0., 0., 0.) );
            // However  cv.add(mat,x) means cv::add(mat, (x,x,x,x) ).
            v[1] = v[0];
            v[2] = v[0];
            v[3] = v[0];
        }
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyFloat_Check(o) )
    {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};

       if ( info.arithm_op_src )
        {
            // Normally cv.XXX(x) means cv.XXX( (x, 0., 0., 0.) );
            // However  cv.add(mat,x) means cv::add(mat, (x,x,x,x) ).
            v[1] = v[0];
            v[2] = v[0];
            v[3] = v[0];
        }
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyTuple_Check(o) )
    {
        // see https://github.com/opencv/opencv/issues/24057
        const int sz  = (int)PyTuple_Size((PyObject*)o);
        const int sz2 = info.arithm_op_src ? std::max(4, sz) : sz; // Scalar has 4 elements.
        m = Mat::zeros(sz2, 1, CV_64F);
        for( int i = 0; i < sz; i++ )
        {
            PyObject* oi = PyTuple_GetItem(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    if (info.outputarg && !PyArray_ISWRITEABLE(oarr))
    {
        failmsg("%s marked as output argument, but provided NumPy array "
                "marked as readonly", info.name);
        return false;
    }

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_HALF ? CV_16F :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG )
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        }
        else
        {
            const std::string dtype_name = getArrayTypeName(oarr);
            failmsg("%s data type = %s is not supported", info.name,
                    dtype_name.c_str());
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);

    CV_LOG_DEBUG(NULL, "Incoming ndarray '" << info.name << "': ndims=" << ndims << "  _sizes=" << pycv_dumpArray(_sizes, ndims) << "  _strides=" << pycv_dumpArray(_strides, ndims));

    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX && !info.nd_mat;
    if (pyopencv_Mat_TypePtr && PyObject_TypeCheck(o, pyopencv_Mat_TypePtr))
    {
        bool wrapChannels = false;
        PyObject* pyobj_wrap_channels = PyObject_GetAttrString(o, "wrap_channels");
        if (pyobj_wrap_channels)
        {
            if (!pyopencv_to_safe(pyobj_wrap_channels, wrapChannels, ArgInfo("cv.Mat.wrap_channels", 0)))
            {
                // TODO extra message
                Py_DECREF(pyobj_wrap_channels);
                return false;
            }
            Py_DECREF(pyobj_wrap_channels);
        }
        ismultichannel = wrapChannels && ndims >= 1;
    }

    for( int i = ndims-1; i >= 0 && !needcopy; i-- )
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
        if( (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _sizes[i] > 1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if (ismultichannel)
    {
        int channels = ndims >= 1 ? (int)_sizes[ndims - 1] : 1;
        if (channels > CV_CN_MAX)
        {
            failmsg("%s unable to wrap channels, too high (%d > CV_CN_MAX=%d)", info.name, (int)channels, (int)CV_CN_MAX);
            return false;
        }
        ndims--;
        type |= CV_MAKETYPE(0, channels);

        if (ndims >= 1 && _strides[ndims - 1] != (npy_intp)elemsize*_sizes[ndims])
            needcopy = true;

        elemsize = CV_ELEM_SIZE(type);
    }

    if (needcopy)
    {
        if (info.outputarg)
        {
            failmsg("Layout of the output array %s is incompatible with cv::Mat", info.name);
            return false;
        }

        if( needcast ) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    int size[CV_MAX_DIM+1] = {};
    size_t step[CV_MAX_DIM+1] = {};

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for ( int i = ndims - 1; i >= 0; --i )
    {
        size[i] = (int)_sizes[i];
        if ( size[i] > 1 )
        {
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        }
        else
        {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // see https://github.com/opencv/opencv/issues/24057
    if ( ( info.arithm_op_src ) && ( ndims == 1 ) && ( size[0] <= 4 ) )
    {
        const int sz  = size[0]; // Real Data Length(1, 2, 3 or 4)
        const int sz2 = 4;       // Scalar has 4 elements.
        m = Mat::zeros(sz2, 1, CV_64F);

        const char *base_ptr = PyArray_BYTES(oarr);
        for(int i = 0; i < sz; i++ )
        {
            PyObject* oi = PyArray_GETITEM(oarr, base_ptr + step[0] * i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("%s has some non-numerical elements", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    // handle degenerate case
    // FIXIT: Don't force 1D for Scalars
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

#if 1
    CV_LOG_DEBUG(NULL, "Construct Mat: ndims=" << ndims << " size=" << pycv_dumpArray(size, ndims) << "  step=" << pycv_dumpArray(step, ndims) << "  type=" << cv::typeToString(type));
#endif

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = GetNumpyAllocator().allocate(o, ndims, size, type, step);
    m.addref();

    if( !needcopy )
    {
        Py_INCREF(o);
    }
    m.allocator = &GetNumpyAllocator();

    // Whether oarr is an ROI
    PyObject *obase = PyArray_BASE(oarr);
    if (obase)
    {
        PyArray_Chunk chunk;
        int res = PyArray_BufferConverter(obase, &chunk);
        if (res == NPY_SUCCEED)
        {
            uchar *start = (uchar *)chunk.ptr;
            uchar *end = start + chunk.len;
            if (start < m.datastart || end > m.dataend)
            {
                m.datastart = start;
                m.datalimit = m.dataend = end;
                m.flags |= Mat::SUBMATRIX_FLAG;
            }
        }
    }

    return true;
}

template<>
PyObject* pyopencv_from(const cv::Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    cv::Mat temp, *p = (cv::Mat*)&m;
    if(!p->u || p->allocator != &GetNumpyAllocator())
    {
        temp.allocator = &GetNumpyAllocator();
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}

// --- bool

template<>
bool pyopencv_to(PyObject* obj, bool& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj) || PyArray_IsIntegerScalar(obj))
    {
        npy_bool npy_value = NPY_FALSE;
        const int ret_code = PyArray_BoolConverter(obj, &npy_value);
        if (ret_code >= 0)
        {
            value = (npy_value == NPY_TRUE);
            return true;
        }
    }
    failmsg("Argument '%s' is not convertable to bool", info.name);
    return false;
}

template<>
PyObject* pyopencv_from(const bool& value)
{
    return PyBool_FromLong(value);
}

// --- ptr

template<>
bool pyopencv_to(PyObject* obj, void*& ptr, const ArgInfo& info)
{
    CV_UNUSED(info);
    if (!obj || obj == Py_None)
        return true;

    if (!PyLong_Check(obj))
        return false;
    ptr = PyLong_AsVoidPtr(obj);
    return ptr != NULL && !PyErr_Occurred();
}

PyObject* pyopencv_from(void*& ptr)
{
    return PyLong_FromVoidPtr(ptr);
}

// -- Scalar

template<>
bool pyopencv_to(PyObject *o, Scalar& s, const ArgInfo& info)
{
    if(!o || o == Py_None)
        return true;
    if (PySequence_Check(o)) {
        if (4 < PySequence_Size(o))
        {
            failmsg("Scalar value for argument '%s' is longer than 4", info.name);
            return false;
        }
        for (Py_ssize_t i = 0; i < PySequence_Size(o); i++) {
            SafeSeqItem item_wrap(o, i);
            PyObject *item = item_wrap.item;
            if (PyFloat_Check(item) || PyInt_Check(item)) {
                s[(int)i] = PyFloat_AsDouble(item);
            } else {
                failmsg("Scalar value for argument '%s' is not numeric", info.name);
                return false;
            }
        }
    } else {
        if (PyFloat_Check(o) || PyInt_Check(o)) {
            s = PyFloat_AsDouble(o);
        } else {
            failmsg("Scalar value for argument '%s' is not numeric", info.name);
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const Scalar& src)
{
    return Py_BuildValue("(dddd)", src[0], src[1], src[2], src[3]);
}

// --- size_t

template<>
bool pyopencv_to(PyObject* obj, size_t& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be integer type, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        if (PyLong_Check(obj))
        {
#if defined(CV_PYTHON_3)
            value = PyLong_AsSize_t(obj);
#else
    #if ULONG_MAX == SIZE_MAX
            value = PyLong_AsUnsignedLong(obj);
    #else
            value = PyLong_AsUnsignedLongLong(obj);
    #endif
#endif
        }
#if !defined(CV_PYTHON_3)
        // Python 2.x has PyIntObject which is not a subtype of PyLongObject
        // Overflow check here is unnecessary because object will be converted to long on the
        // interpreter side
        else if (PyInt_Check(obj))
        {
            const long res = PyInt_AsLong(obj);
            if (res < 0) {
                failmsg("Argument '%s' can not be safely parsed to 'size_t'", info.name);
                return false;
            }
    #if ULONG_MAX == SIZE_MAX
            value = PyInt_AsUnsignedLongMask(obj);
    #else
            value = PyInt_AsUnsignedLongLongMask(obj);
    #endif
        }
#endif
        else
        {
            const bool isParsed = parseNumpyScalar<size_t>(obj, value);
            if (!isParsed) {
                failmsg("Argument '%s' can not be safely parsed to 'size_t'", info.name);
                return false;
            }
        }
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const size_t& value)
{
    return PyLong_FromSize_t(value);
}

// --- int

template<>
bool pyopencv_to(PyObject* obj, int& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be integer, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        value = PyArray_PyIntAsInt(obj);
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !CV_HAS_CONVERSION_ERROR(value);
}

template<>
PyObject* pyopencv_from(const int& value)
{
    return PyInt_FromLong(value);
}

// --- int64

template<>
bool pyopencv_to(PyObject* obj, int64& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be integer, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        value = PyLong_AsLongLong(obj);
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !CV_HAS_CONVERSION_ERROR(value);
}

template<>
PyObject* pyopencv_from(const int64& value)
{
    return PyLong_FromLongLong(value);
}


// --- uchar

template<>
bool pyopencv_to(PyObject* obj, uchar& value, const ArgInfo& info)
{
    CV_UNUSED(info);
    if(!obj || obj == Py_None)
        return true;
    int ivalue = (int)PyInt_AsLong(obj);
    value = cv::saturate_cast<uchar>(ivalue);
    return ivalue != -1 || !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const uchar& value)
{
    return PyInt_FromLong(value);
}

// --- char

template<>
bool pyopencv_to(PyObject* obj, char& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be an integer, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        value = saturate_cast<char>(PyArray_PyIntAsInt(obj));
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !CV_HAS_CONVERSION_ERROR(value);
}

// --- double

template<>
bool pyopencv_to(PyObject* obj, double& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be double, not bool", info.name);
        return false;
    }
    if (PyArray_IsPythonNumber(obj))
    {
        if (PyLong_Check(obj))
        {
            value = PyLong_AsDouble(obj);
        }
        else
        {
            value = PyFloat_AsDouble(obj);
        }
    }
    else if (PyArray_CheckScalar(obj))
    {
        const bool isParsed = parseNumpyScalar<double>(obj, value);
        if (!isParsed) {
            failmsg("Argument '%s' can not be safely parsed to 'double'", info.name);
            return false;
        }
    }
    else
    {
        failmsg("Argument '%s' can not be treated as a double", info.name);
        return false;
    }
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const double& value)
{
    return PyFloat_FromDouble(value);
}

// --- float

template<>
bool pyopencv_to(PyObject* obj, float& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be float, not bool", info.name);
        return false;
    }
    if (PyArray_IsPythonNumber(obj))
    {
        if (PyLong_Check(obj))
        {
            double res = PyLong_AsDouble(obj);
            value = static_cast<float>(res);
        }
        else
        {
            double res = PyFloat_AsDouble(obj);
            value = static_cast<float>(res);
        }
    }
    else if (PyArray_CheckScalar(obj))
    {
       const bool isParsed = parseNumpyScalar<float>(obj, value);
        if (!isParsed) {
            failmsg("Argument '%s' can not be safely parsed to 'float'", info.name);
            return false;
        }
    }
    else
    {
        failmsg("Argument '%s' can't be treated as a float", info.name);
        return false;
    }
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const float& value)
{
    return PyFloat_FromDouble(value);
}

// --- string

template<>
bool pyopencv_to(PyObject* obj, String &value, const ArgInfo& info)
{
    if(!obj || obj == Py_None)
    {
        return true;
    }
    std::string str;

#if ((PY_VERSION_HEX >= 0x03060000) && !defined(Py_LIMITED_API)) || (Py_LIMITED_API >= 0x03060000)
    if (info.pathlike)
    {
        obj = PyOS_FSPath(obj);
        if (PyErr_Occurred())
        {
            failmsg("Expected '%s' to be a str or path-like object", info.name);
            return false;
        }
    }
#endif
    if (getUnicodeString(obj, str))
    {
        value = str;
        return true;
    }
    else
    {
        // If error hasn't been already set by Python conversion functions
        if (!PyErr_Occurred())
        {
            // Direct access to underlying slots of PyObjectType is not allowed
            // when limited API is enabled
#ifdef Py_LIMITED_API
            failmsg("Can't convert object to 'str' for '%s'", info.name);
#else
            failmsg("Can't convert object of type '%s' to 'str' for '%s'",
                    obj->ob_type->tp_name, info.name);
#endif
        }
    }
    return false;
}

template<>
PyObject* pyopencv_from(const String& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}

#if CV_VERSION_MAJOR == 3
template<>
PyObject* pyopencv_from(const std::string& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}
#endif

// --- Size

template<>
bool pyopencv_to(PyObject* obj, Size& sz, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(sz.width),
                                RefWrapper<int>(sz.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Size& sz)
{
    return Py_BuildValue("(ii)", sz.width, sz.height);
}

template<>
bool pyopencv_to(PyObject* obj, Size_<float>& sz, const ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(sz.width),
                                  RefWrapper<float>(sz.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Size_<float>& sz)
{
    return Py_BuildValue("(ff)", sz.width, sz.height);
}

// --- Rect

template<>
bool pyopencv_to(PyObject* obj, Rect& r, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(r.x), RefWrapper<int>(r.y),
                                RefWrapper<int>(r.width),
                                RefWrapper<int>(r.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Rect& r)
{
    return Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, Rect2f& r, const ArgInfo& info)
{
    RefWrapper<float> values[] = {
        RefWrapper<float>(r.x), RefWrapper<float>(r.y),
        RefWrapper<float>(r.width), RefWrapper<float>(r.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Rect2f& r)
{
    return Py_BuildValue("(ffff)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, Rect2d& r, const ArgInfo& info)
{
    RefWrapper<double> values[] = {
        RefWrapper<double>(r.x), RefWrapper<double>(r.y),
        RefWrapper<double>(r.width), RefWrapper<double>(r.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Rect2d& r)
{
    return Py_BuildValue("(dddd)", r.x, r.y, r.width, r.height);
}

// --- RotatedRect

static inline bool convertToRotatedRect(PyObject* obj, RotatedRect& dst)
{
    PyObject* type = PyObject_Type(obj);
    if (getPyObjectAttr(type, "__module__") == MODULESTR &&
        getPyObjectNameAttr(type) == "RotatedRect")
    {
        struct pyopencv_RotatedRect_t
        {
            PyObject_HEAD
            cv::RotatedRect v;
        };
        dst = reinterpret_cast<pyopencv_RotatedRect_t*>(obj)->v;

        Py_DECREF(type);
        return true;
    }
    Py_DECREF(type);
    return false;
}

template<>
bool pyopencv_to(PyObject* obj, RotatedRect& dst, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    // This is a workaround for compatibility with an initialization from tuple.
    // Allows import RotatedRect as an object.
    if (convertToRotatedRect(obj, dst))
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s' as RotatedRect."
                "Input argument doesn't provide sequence protocol",
                info.name);
        return false;
    }
    const std::size_t sequenceSize = PySequence_Size(obj);
    if (sequenceSize != 3)
    {
        failmsg("Can't parse '%s' as RotatedRect. Expected sequence length 3, got %lu",
                info.name, sequenceSize);
        return false;
    }
    {
        const String centerItemName = format("'%s' center point", info.name);
        const ArgInfo centerItemInfo(centerItemName.c_str(), 0);
        SafeSeqItem centerItem(obj, 0);
        if (!pyopencv_to(centerItem.item, dst.center, centerItemInfo))
        {
            return false;
        }
    }
    {
        const String sizeItemName = format("'%s' size", info.name);
        const ArgInfo sizeItemInfo(sizeItemName.c_str(), 0);
        SafeSeqItem sizeItem(obj, 1);
        if (!pyopencv_to(sizeItem.item, dst.size, sizeItemInfo))
        {
            return false;
        }
    }
    {
        const String angleItemName = format("'%s' angle", info.name);
        const ArgInfo angleItemInfo(angleItemName.c_str(), 0);
        SafeSeqItem angleItem(obj, 2);
        if (!pyopencv_to(angleItem.item, dst.angle, angleItemInfo))
        {
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const RotatedRect& src)
{
    return Py_BuildValue("((ff)(ff)f)", src.center.x, src.center.y, src.size.width, src.size.height, src.angle);
}

// --- Range

template<>
bool pyopencv_to(PyObject* obj, Range& r, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (PyObject_Size(obj) == 0)
    {
        r = Range::all();
        return true;
    }
    RefWrapper<int> values[] = {RefWrapper<int>(r.start), RefWrapper<int>(r.end)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Range& r)
{
    return Py_BuildValue("(ii)", r.start, r.end);
}

// --- Point

template<>
bool pyopencv_to(PyObject* obj, Point& p, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(p.x), RefWrapper<int>(p.y)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point& p)
{
    return Py_BuildValue("(ii)", p.x, p.y);
}

template <>
bool pyopencv_to(PyObject* obj, Point2f& p, const ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(p.x),
                                  RefWrapper<float>(p.y)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point2f& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

template<>
bool pyopencv_to(PyObject* obj, Point2d& p, const ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(p.x),
                                   RefWrapper<double>(p.y)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point2d& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

template<>
bool pyopencv_to(PyObject* obj, Point3i& p, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(p.x),
                                RefWrapper<int>(p.y),
                                RefWrapper<int>(p.z)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point3i& p)
{
    return Py_BuildValue("(iii)", p.x, p.y, p.z);
}

template<>
bool pyopencv_to(PyObject* obj, Point3f& p, const ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(p.x),
                                  RefWrapper<float>(p.y),
                                  RefWrapper<float>(p.z)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point3f& p)
{
    return Py_BuildValue("(ddd)", p.x, p.y, p.z);
}

template<>
bool pyopencv_to(PyObject* obj, Point3d& p, const ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(p.x),
                                   RefWrapper<double>(p.y),
                                   RefWrapper<double>(p.z)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point3d& p)
{
    return Py_BuildValue("(ddd)", p.x, p.y, p.z);
}

// --- Vec

bool pyopencv_to(PyObject* obj, Vec4d& v, ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(v[0]), RefWrapper<double>(v[1]),
                                   RefWrapper<double>(v[2]), RefWrapper<double>(v[3])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec4d& v)
{
    return Py_BuildValue("(dddd)", v[0], v[1], v[2], v[3]);
}

bool pyopencv_to(PyObject* obj, Vec4f& v, ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(v[0]), RefWrapper<float>(v[1]),
                                  RefWrapper<float>(v[2]), RefWrapper<float>(v[3])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec4f& v)
{
    return Py_BuildValue("(ffff)", v[0], v[1], v[2], v[3]);
}

bool pyopencv_to(PyObject* obj, Vec4i& v, ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(v[0]), RefWrapper<int>(v[1]),
                                RefWrapper<int>(v[2]), RefWrapper<int>(v[3])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec4i& v)
{
    return Py_BuildValue("(iiii)", v[0], v[1], v[2], v[3]);
}

bool pyopencv_to(PyObject* obj, Vec3d& v, ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(v[0]),
                                   RefWrapper<double>(v[1]),
                                   RefWrapper<double>(v[2])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec3d& v)
{
    return Py_BuildValue("(ddd)", v[0], v[1], v[2]);
}

bool pyopencv_to(PyObject* obj, Vec3f& v, ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(v[0]),
                                  RefWrapper<float>(v[1]),
                                  RefWrapper<float>(v[2])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec3f& v)
{
    return Py_BuildValue("(fff)", v[0], v[1], v[2]);
}

bool pyopencv_to(PyObject* obj, Vec3i& v, ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(v[0]), RefWrapper<int>(v[1]),
                                RefWrapper<int>(v[2])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec3i& v)
{
    return Py_BuildValue("(iii)", v[0], v[1], v[2]);
}

bool pyopencv_to(PyObject* obj, Vec2d& v, ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(v[0]),
                                   RefWrapper<double>(v[1])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec2d& v)
{
    return Py_BuildValue("(dd)", v[0], v[1]);
}

bool pyopencv_to(PyObject* obj, Vec2f& v, ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(v[0]),
                                  RefWrapper<float>(v[1])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec2f& v)
{
    return Py_BuildValue("(ff)", v[0], v[1]);
}

bool pyopencv_to(PyObject* obj, Vec2i& v, ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(v[0]), RefWrapper<int>(v[1])};
    return parseSequence(obj, values, info);
}

PyObject* pyopencv_from(const Vec2i& v)
{
    return Py_BuildValue("(ii)", v[0], v[1]);
}


// --- TermCriteria

template<>
bool pyopencv_to(PyObject* obj, TermCriteria& dst, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s' as TermCriteria."
                "Input argument doesn't provide sequence protocol",
                info.name);
        return false;
    }
    const std::size_t sequenceSize = PySequence_Size(obj);
    if (sequenceSize != 3) {
        failmsg("Can't parse '%s' as TermCriteria. Expected sequence length 3, "
                "got %lu",
                info.name, sequenceSize);
        return false;
    }
    {
        const String typeItemName = format("'%s' criteria type", info.name);
        const ArgInfo typeItemInfo(typeItemName.c_str(), 0);
        SafeSeqItem typeItem(obj, 0);
        if (!pyopencv_to(typeItem.item, dst.type, typeItemInfo))
        {
            return false;
        }
    }
    {
        const String maxCountItemName = format("'%s' max count", info.name);
        const ArgInfo maxCountItemInfo(maxCountItemName.c_str(), 0);
        SafeSeqItem maxCountItem(obj, 1);
        if (!pyopencv_to(maxCountItem.item, dst.maxCount, maxCountItemInfo))
        {
            return false;
        }
    }
    {
        const String epsilonItemName = format("'%s' epsilon", info.name);
        const ArgInfo epsilonItemInfo(epsilonItemName.c_str(), 0);
        SafeSeqItem epsilonItem(obj, 2);
        if (!pyopencv_to(epsilonItem.item, dst.epsilon, epsilonItemInfo))
        {
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const TermCriteria& src)
{
    return Py_BuildValue("(iid)", src.type, src.maxCount, src.epsilon);
}

// --- Moments

template<>
PyObject* pyopencv_from(const Moments& m)
{
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
                         "m00", m.m00, "m10", m.m10, "m01", m.m01,
                         "m20", m.m20, "m11", m.m11, "m02", m.m02,
                         "m30", m.m30, "m21", m.m21, "m12", m.m12, "m03", m.m03,
                         "mu20", m.mu20, "mu11", m.mu11, "mu02", m.mu02,
                         "mu30", m.mu30, "mu21", m.mu21, "mu12", m.mu12, "mu03", m.mu03,
                         "nu20", m.nu20, "nu11", m.nu11, "nu02", m.nu02,
                         "nu30", m.nu30, "nu21", m.nu21, "nu12", m.nu12, "nu03", m.nu03);
}

// --- pair

template<>
PyObject* pyopencv_from(const std::pair<int, double>& src)
{
    return Py_BuildValue("(id)", src.first, src.second);
}
