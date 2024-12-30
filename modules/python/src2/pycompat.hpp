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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

// Defines for Python 2/3 compatibility.
#ifndef __PYCOMPAT_HPP__
#define __PYCOMPAT_HPP__

#include <string>

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOGDI

#include <windows.h>

static unsigned int code_page = GetACP();

#endif

#if PY_MAJOR_VERSION >= 3

// Python3 treats all ints as longs, PyInt_X functions have been removed.
#define PyInt_Check PyLong_Check
#define PyInt_CheckExact PyLong_CheckExact
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
#define PyInt_FromLong PyLong_FromLong
#define PyNumber_Int PyNumber_Long

#if defined(_WIN32)

static inline PyObject* PyString_FromString(const char* u)
{
    if (code_page == 65001) return PyUnicode_FromString(u);
    return PyUnicode_DecodeMBCS(u, strlen(u), NULL);
}
static inline PyObject* PyString_FromStringAndSize(const char* u, Py_ssize_t size)
{
    if (code_page == 65001) return PyUnicode_FromStringAndSize(u, size);
    return PyUnicode_DecodeMBCS(u, size, NULL);
}
#else
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif

#endif // PY_MAJOR >=3

#ifndef PyType_CheckExact
#define PyType_CheckExact(obj) (Py_TYPE(op) == &PyType_Type)
#endif // !PyType_CheckExact

static inline bool getUnicodeString(PyObject * obj, std::string &str)
{
    bool res = false;
    if (PyUnicode_Check(obj))
    {
        PyObject* bytes;
#if defined(_WIN32)
        if (code_page == 65001) bytes = PyUnicode_AsUTF8String(obj);
        else bytes = PyUnicode_AsMBCSString(obj);
#else
        bytes = PyUnicode_AsUTF8String(obj);
#endif
        if (PyBytes_Check(bytes))
        {
            const char * raw = PyBytes_AsString(bytes);
            if (raw)
            {
                str = std::string(raw);
                res = true;
            }
        }
        Py_XDECREF(bytes);
    }
#if PY_MAJOR_VERSION < 3
    else if (PyString_Check(obj))
    {
        const char * raw = PyString_AsString(obj);
        if (raw)
        {
            str = std::string(raw);
            res = true;
        }
    }
#endif
    return res;
}

static inline
std::string getPyObjectAttr(PyObject* obj, const char* attrName)
{
    std::string obj_name;
    PyObject* cls_name_obj = PyObject_GetAttrString(obj, attrName);
    if (cls_name_obj && !getUnicodeString(cls_name_obj, obj_name)) {
        obj_name.clear();
    }
    #ifndef Py_LIMITED_API
    if (PyType_CheckExact(obj) && obj_name.empty())
    {
        obj_name = reinterpret_cast<PyTypeObject*>(obj)->tp_name;
    }
    #endif
    if (obj_name.empty()) {
        obj_name = "<UNAVAILABLE>";
    }
    return obj_name;
}

static inline
std::string getPyObjectNameAttr(PyObject* obj)
{
    return getPyObjectAttr(obj, "__name__");
}

//==================================================================================================

#define CV_PY_FN_WITH_KW_(fn, flags) (PyCFunction)(void*)(PyCFunctionWithKeywords)(fn), (flags) | METH_VARARGS | METH_KEYWORDS
#define CV_PY_FN_NOARGS_(fn, flags) (PyCFunction)(fn), (flags) | METH_NOARGS

#define CV_PY_FN_WITH_KW(fn) CV_PY_FN_WITH_KW_(fn, 0)
#define CV_PY_FN_NOARGS(fn) CV_PY_FN_NOARGS_(fn, 0)

#define CV_PY_TO_CLASS(TYPE)                                                                          \
template<>                                                                                            \
bool pyopencv_to(PyObject* dst, TYPE& src, const ArgInfo& info)                                       \
{                                                                                                     \
    if (!dst || dst == Py_None)                                                                       \
        return true;                                                                                  \
    Ptr<TYPE> ptr;                                                                                    \
                                                                                                      \
    if (!pyopencv_to(dst, ptr, info)) return false;                                                   \
    src = *ptr;                                                                                       \
    return true;                                                                                      \
}

#define CV_PY_FROM_CLASS(TYPE)                                                                        \
template<>                                                                                            \
PyObject* pyopencv_from(const TYPE& src)                                                              \
{                                                                                                     \
    Ptr<TYPE> ptr(new TYPE());                                                                        \
                                                                                                      \
    *ptr = src;                                                                                       \
    return pyopencv_from(ptr);                                                                        \
}

#define CV_PY_TO_CLASS_PTR(TYPE)                                                                      \
template<>                                                                                            \
bool pyopencv_to(PyObject* dst, TYPE*& src, const ArgInfo& info)                                      \
{                                                                                                     \
    if (!dst || dst == Py_None)                                                                       \
        return true;                                                                                  \
    Ptr<TYPE> ptr;                                                                                    \
                                                                                                      \
    if (!pyopencv_to(dst, ptr, info)) return false;                                                   \
    src = ptr;                                                                                        \
    return true;                                                                                      \
}

#define CV_PY_FROM_CLASS_PTR(TYPE)                                                                    \
static PyObject* pyopencv_from(TYPE*& src)                                                            \
{                                                                                                     \
    return pyopencv_from(Ptr<TYPE>(src));                                                             \
}

#define CV_PY_TO_ENUM(TYPE)                                                                           \
template<>                                                                                            \
bool pyopencv_to(PyObject* dst, TYPE& src, const ArgInfo& info)                                       \
{                                                                                                     \
    if (!dst || dst == Py_None)                                                                       \
        return true;                                                                                  \
    int underlying = 0;                                                  \
                                                                                                      \
    if (!pyopencv_to(dst, underlying, info)) return false;                                            \
    src = static_cast<TYPE>(underlying);                                                              \
    return true;                                                                                      \
}

#define CV_PY_FROM_ENUM(TYPE)                                                                         \
template<>                                                                                            \
PyObject* pyopencv_from(const TYPE& src)                                                              \
{                                                                                                     \
    return pyopencv_from(static_cast<int>(src));                         \
}

//==================================================================================================

#if PY_MAJOR_VERSION >= 3
#define CVPY_TYPE_HEAD PyVarObject_HEAD_INIT(&PyType_Type, 0)
#define CVPY_TYPE_INCREF(T) Py_INCREF(T)
#else
#define CVPY_TYPE_HEAD PyObject_HEAD_INIT(&PyType_Type) 0,
#define CVPY_TYPE_INCREF(T) _Py_INC_REFTOTAL _Py_REF_DEBUG_COMMA (T)->ob_refcnt++
#endif


#define CVPY_TYPE_DECLARE(EXPORT_NAME, CLASS_ID, STORAGE, SNAME, SCOPE) \
    struct pyopencv_##CLASS_ID##_t \
    { \
        PyObject_HEAD \
        STORAGE v; \
    }; \
    static PyTypeObject pyopencv_##CLASS_ID##_TypeXXX = \
    { \
        CVPY_TYPE_HEAD \
        MODULESTR SCOPE"."#EXPORT_NAME, \
        sizeof(pyopencv_##CLASS_ID##_t), \
    }; \
    static PyTypeObject * pyopencv_##CLASS_ID##_TypePtr = &pyopencv_##CLASS_ID##_TypeXXX; \
    static bool pyopencv_##CLASS_ID##_getp(PyObject * self, STORAGE * & dst) \
    { \
        if (PyObject_TypeCheck(self, pyopencv_##CLASS_ID##_TypePtr)) \
        { \
            dst = &(((pyopencv_##CLASS_ID##_t*)self)->v); \
            return true; \
        } \
        return false; \
    } \
    static PyObject * pyopencv_##CLASS_ID##_Instance(const STORAGE &r) \
    { \
        pyopencv_##CLASS_ID##_t *m = PyObject_NEW(pyopencv_##CLASS_ID##_t, pyopencv_##CLASS_ID##_TypePtr); \
        new (&(m->v)) STORAGE(r); \
        return (PyObject*)m; \
    } \
    static void pyopencv_##CLASS_ID##_dealloc(PyObject* self) \
    { \
        ((pyopencv_##CLASS_ID##_t*)self)->v.STORAGE::~SNAME(); \
        PyObject_Del(self); \
    } \
    static PyObject* pyopencv_##CLASS_ID##_repr(PyObject* self) \
    { \
        char str[1000]; \
        snprintf(str, sizeof(str), "< " MODULESTR SCOPE"."#EXPORT_NAME" %p>", self); \
        return PyString_FromString(str); \
    }


#define CVPY_TYPE_INIT_STATIC(EXPORT_NAME, CLASS_ID, ERROR_HANDLER, BASE, CONSTRUCTOR, SCOPE) \
    { \
        pyopencv_##CLASS_ID##_TypePtr->tp_base = pyopencv_##BASE##_TypePtr; \
        pyopencv_##CLASS_ID##_TypePtr->tp_dealloc = pyopencv_##CLASS_ID##_dealloc; \
        pyopencv_##CLASS_ID##_TypePtr->tp_repr = pyopencv_##CLASS_ID##_repr; \
        pyopencv_##CLASS_ID##_TypePtr->tp_getset = pyopencv_##CLASS_ID##_getseters; \
        pyopencv_##CLASS_ID##_TypePtr->tp_init = (initproc) CONSTRUCTOR; \
        pyopencv_##CLASS_ID##_TypePtr->tp_methods = pyopencv_##CLASS_ID##_methods; \
        pyopencv_##CLASS_ID##_TypePtr->tp_alloc = PyType_GenericAlloc; \
        pyopencv_##CLASS_ID##_TypePtr->tp_new = PyType_GenericNew; \
        pyopencv_##CLASS_ID##_TypePtr->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE; \
        if (PyType_Ready(pyopencv_##CLASS_ID##_TypePtr) != 0) \
        { \
            ERROR_HANDLER; \
        } \
        CVPY_TYPE_INCREF(pyopencv_##CLASS_ID##_TypePtr); \
        if (!registerNewType(m, #EXPORT_NAME, (PyObject*)pyopencv_##CLASS_ID##_TypePtr, SCOPE)) \
        { \
            printf("Failed to register a new type: " #EXPORT_NAME  ", base (" #BASE ") in " SCOPE " \n"); \
            ERROR_HANDLER; \
        } \
    }

//==================================================================================================

#define CVPY_TYPE_DECLARE_DYNAMIC(EXPORT_NAME, CLASS_ID, STORAGE, SNAME, SCOPE) \
    struct pyopencv_##CLASS_ID##_t \
    { \
        PyObject_HEAD \
        STORAGE v; \
    }; \
    static PyObject * pyopencv_##CLASS_ID##_TypePtr = 0; \
    static bool pyopencv_##CLASS_ID##_getp(PyObject * self, STORAGE * & dst) \
    { \
        if (PyObject_TypeCheck(self, (PyTypeObject*)pyopencv_##CLASS_ID##_TypePtr)) \
        { \
            dst = &(((pyopencv_##CLASS_ID##_t*)self)->v); \
            return true; \
        } \
        return false; \
    } \
    static PyObject * pyopencv_##CLASS_ID##_Instance(const STORAGE &r) \
    { \
        pyopencv_##CLASS_ID##_t *m = PyObject_New(pyopencv_##CLASS_ID##_t, (PyTypeObject*)pyopencv_##CLASS_ID##_TypePtr); \
        new (&(m->v)) STORAGE(r); \
        return (PyObject*)m; \
    } \
    static void pyopencv_##CLASS_ID##_dealloc(PyObject* self) \
    { \
        ((pyopencv_##CLASS_ID##_t*)self)->v.STORAGE::~SNAME(); \
        PyObject_Del(self); \
    } \
    static PyObject* pyopencv_##CLASS_ID##_repr(PyObject* self) \
    { \
        char str[1000]; \
        snprintf(str, sizeof(str), "< " MODULESTR SCOPE"."#EXPORT_NAME" %p>", self); \
        return PyString_FromString(str); \
    } \
    static PyType_Slot pyopencv_##CLASS_ID##_Slots[] =  \
    { \
        {Py_tp_dealloc, 0}, \
        {Py_tp_repr, 0}, \
        {Py_tp_getset, 0}, \
        {Py_tp_init, 0}, \
        {Py_tp_methods, 0}, \
        {Py_tp_alloc, 0}, \
        {Py_tp_new, 0}, \
        {0, 0} \
    }; \
    static PyType_Spec pyopencv_##CLASS_ID##_Spec = \
    { \
        MODULESTR SCOPE"."#EXPORT_NAME, \
        sizeof(pyopencv_##CLASS_ID##_t), \
        0, \
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, \
        pyopencv_##CLASS_ID##_Slots  \
    };

#define CVPY_TYPE_INIT_DYNAMIC(EXPORT_NAME, CLASS_ID, ERROR_HANDLER, BASE, CONSTRUCTOR, SCOPE) \
    { \
        pyopencv_##CLASS_ID##_Slots[0].pfunc /*tp_dealloc*/ = (void*)pyopencv_##CLASS_ID##_dealloc; \
        pyopencv_##CLASS_ID##_Slots[1].pfunc /*tp_repr*/ = (void*)pyopencv_##CLASS_ID##_repr; \
        pyopencv_##CLASS_ID##_Slots[2].pfunc /*tp_getset*/ = (void*)pyopencv_##CLASS_ID##_getseters; \
        pyopencv_##CLASS_ID##_Slots[3].pfunc /*tp_init*/ = (void*) CONSTRUCTOR; \
        pyopencv_##CLASS_ID##_Slots[4].pfunc /*tp_methods*/ = pyopencv_##CLASS_ID##_methods; \
        pyopencv_##CLASS_ID##_Slots[5].pfunc /*tp_alloc*/ = (void*)PyType_GenericAlloc; \
        pyopencv_##CLASS_ID##_Slots[6].pfunc /*tp_new*/ = (void*)PyType_GenericNew; \
        PyObject * bases = 0; \
        if (pyopencv_##BASE##_TypePtr) \
            bases = PyTuple_Pack(1, pyopencv_##BASE##_TypePtr); \
        pyopencv_##CLASS_ID##_TypePtr = PyType_FromSpecWithBases(&pyopencv_##CLASS_ID##_Spec, bases); \
        if (!pyopencv_##CLASS_ID##_TypePtr) \
        { \
            printf("Failed to create type from spec: " #CLASS_ID ", base (" #BASE ")\n"); \
            ERROR_HANDLER; \
        } \
        if (!registerNewType(m, #EXPORT_NAME, (PyObject*)pyopencv_##CLASS_ID##_TypePtr, SCOPE)) \
        { \
            printf("Failed to register a new type: " #EXPORT_NAME  ", base (" #BASE ") in " SCOPE " \n"); \
            Py_DECREF(pyopencv_##CLASS_ID##_TypePtr); \
            ERROR_HANDLER; \
        } \
        Py_DECREF(pyopencv_##CLASS_ID##_TypePtr); \
    }

// Debug module load:
//
// else \
// { \
//     printf("Init: " #NAME ", base (" #BASE ") -> %p" "\n", pyopencv_##NAME##_TypePtr); \
// } \


#endif // END HEADER GUARD
