#ifdef HAVE_OPENCV_VIDEOIO
typedef std::vector<VideoCaptureAPIs> vector_VideoCaptureAPIs;
typedef std::vector<VideoCapture> vector_VideoCapture;

template<> struct pyopencvVecConverter<cv::VideoCaptureAPIs>
{
    static bool to(PyObject* obj, std::vector<cv::VideoCaptureAPIs>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cv::VideoCaptureAPIs>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *o, std::vector<cv::VideoCaptureAPIs>& apis, const ArgInfo& info)
{
  return pyopencvVecConverter<cv::VideoCaptureAPIs>::to(o, apis, info);
}

template<> bool pyopencv_to(PyObject* obj, cv::VideoCapture& stream, const ArgInfo& info)
{
    Ptr<VideoCapture> * obj_getp = nullptr;
    if (!pyopencv_VideoCapture_getp(obj, obj_getp))
        return (failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)") != nullptr);

    stream = **obj_getp;
    return true;
}

class PythonStreamReader : public cv::IStreamReader
{
public:
    PythonStreamReader(PyObject* _obj = nullptr) : obj(_obj)
    {
        if (obj)
            Py_INCREF(obj);
    }

    ~PythonStreamReader()
    {
        if (obj)
            Py_DECREF(obj);
    }

    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        if (!obj)
            return 0;

        PyObject* ioBase = reinterpret_cast<PyObject*>(obj);

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* py_size = pyopencv_from(static_cast<int>(size));

        PyObject* res = PyObject_CallMethodObjArgs(ioBase, PyString_FromString("read"), py_size, NULL);
        bool hasPyReadError = PyErr_Occurred() != nullptr;
        char* src = PyBytes_AsString(res);
        size_t len = static_cast<size_t>(PyBytes_Size(res));
        bool hasPyBytesError = PyErr_Occurred() != nullptr;
        if (src && len <= static_cast<size_t>(size))
        {
            std::memcpy(buffer, src, len);
        }
        Py_DECREF(res);
        Py_DECREF(py_size);

        PyGILState_Release(gstate);

        if (hasPyReadError)
            CV_Error(cv::Error::StsError, "Python .read() call error");
        if (hasPyBytesError)
            CV_Error(cv::Error::StsError, "Python buffer access error");

        CV_CheckLE(len, static_cast<size_t>(size), "Stream chunk size should be less or equal than requested size");

        return len;
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        if (!obj)
            return 0;

        PyObject* ioBase = reinterpret_cast<PyObject*>(obj);

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* py_offset = pyopencv_from(static_cast<int>(offset));
        PyObject* py_whence = pyopencv_from(way);

        PyObject* res = PyObject_CallMethodObjArgs(ioBase, PyString_FromString("seek"), py_offset, py_whence, NULL);
        bool hasPySeekError = PyErr_Occurred() != nullptr;
        long long pos = PyLong_AsLongLong(res);
        bool hasPyConvertError = PyErr_Occurred() != nullptr;
        Py_DECREF(res);
        Py_DECREF(py_offset);
        Py_DECREF(py_whence);

        PyGILState_Release(gstate);

        if (hasPySeekError)
            CV_Error(cv::Error::StsError, "Python .seek() call error");
        if (hasPyConvertError)
            CV_Error(cv::Error::StsError, "Python .seek() result => long long conversion error");
        return pos;
    }

private:
    PyObject* obj;
};

template<>
bool pyopencv_to(PyObject* obj, Ptr<cv::IStreamReader>& p, const ArgInfo&)
{
    if (!obj)
        return false;

    PyObject* ioModule = PyImport_ImportModule("io");
    PyObject* type = PyObject_GetAttrString(ioModule, "BufferedIOBase");
    Py_DECREF(ioModule);
    bool isValidPyType = PyObject_IsInstance(obj, type) == 1;
    Py_DECREF(type);

    if (!isValidPyType)
    {
        PyErr_SetString(PyExc_TypeError, "Input stream should be derived from io.BufferedIOBase");
        return false;
    }

    if (!PyErr_Occurred()) {
        p = makePtr<PythonStreamReader>(obj);
        return true;
    }
    return false;
}

#endif // HAVE_OPENCV_VIDEOIO
