#!/usr/bin/env python
from __future__ import print_function

import sys
import ctypes
from functools import partial
from collections import namedtuple
import sys

if sys.version_info[0] < 3:
    from collections import Sequence
else:
    from collections.abc import Sequence

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest


def is_numeric(dtype):
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def get_limits(dtype):
    if not is_numeric(dtype):
        return None, None

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    return info.min, info.max


def get_conversion_error_msg(value, expected, actual):
    return 'Conversion "{}" of type "{}" failed\nExpected: "{}" vs Actual "{}"'.format(
        value, type(value).__name__, expected, actual
    )


def get_no_exception_msg(value):
    return 'Exception is not risen for {} of type {}'.format(value, type(value).__name__)


def rpad(src, dst_size, pad_value=0):
    """Extend `src` up to `dst_size` with given value.

    Args:
        src (np.ndarray | tuple | list): 1d array like object to pad.
        dst_size (_type_): Desired `src` size after padding.
        pad_value (int, optional): Padding value. Defaults to 0.

    Returns:
        np.ndarray: 1d array with len == `dst_size`.
    """
    src = np.asarray(src)
    if len(src.shape) != 1:
        raise ValueError("Only 1d arrays are supported")

    # Considering the meaning, it is desirable to use np.pad().
    # However, the old numpy doesn't include the following fixes and cannot work as expected.
    # So an alternative fix that combines np.append() and np.fill() is used.
    # https://docs.scipy.org/doc/numpy-1.13.0/release.html#support-for-returning-arrays-of-arbitrary-dimensions-in-apply-along-axis

    return np.append(src, np.full( dst_size - len(src), pad_value, dtype=src.dtype) )

def get_ocv_arithm_op_table(apply_saturation=False):
    def saturate(func):
        def wrapped_func(x, y):
            dst_dtype = x.dtype
            if apply_saturation:
                if np.issubdtype(x.dtype, np.integer):
                    x = x.astype(np.int64)
            # Apply padding or truncation for array-like `y` inputs
            if not isinstance(y, (float, int)):
                if len(y) > x.shape[-1]:
                    y = y[:x.shape[-1]]
                else:
                    y = rpad(y, x.shape[-1], pad_value=0)

            dst = func(x, y)
            if apply_saturation:
                min_val, max_val = get_limits(dst_dtype)
                dst = np.clip(dst, min_val, max_val)
            return dst.astype(dst_dtype)
        return wrapped_func

    @saturate
    def subtract(x, y):
        return x - y

    @saturate
    def add(x, y):
        return x + y

    @saturate
    def divide(x, y):
        if not isinstance(y, (int, float)):
            dst_dtype = np.result_type(x, y)
            y = np.array(y).astype(dst_dtype)
            _, max_value = get_limits(dst_dtype)
            y[y == 0] = max_value

        # to compatible between python2 and python3, it calicurates with float.
        # python2: int / int = int
        # python3: int / int = float
        dst = 1.0 * x / y

        if np.issubdtype(x.dtype, np.integer):
            dst = np.rint(dst)
        return dst

    @saturate
    def multiply(x, y):
        return x * y

    @saturate
    def absdiff(x, y):
        res = np.abs(x - y)
        return res

    return {
        cv.subtract: subtract,
        cv.add: add,
        cv.multiply: multiply,
        cv.divide: divide,
        cv.absdiff: absdiff
    }


class Bindings(NewOpenCVTests):

    def test_inheritance(self):
        bm = cv.StereoBM_create()
        bm.getPreFilterCap()  # from StereoBM
        bm.getBlockSize()  # from SteroMatcher

        boost = cv.ml.Boost_create()
        boost.getBoostType()  # from ml::Boost
        boost.getMaxDepth()  # from ml::DTrees
        boost.isClassifier()  # from ml::StatModel

    def test_raiseGeneralException(self):
        with self.assertRaises((cv.error,),
                            msg='C++ exception is not propagated to Python in the right way') as cm:
            cv.utils.testRaiseGeneralException()
        self.assertEqual(str(cm.exception), 'exception text')

    def test_redirectError(self):
        try:
            cv.imshow("", None)  # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass

        handler_called = [False]

        def test_error_handler(status, func_name, err_msg, file_name, line):
            handler_called[0] = True

        cv.redirectError(test_error_handler)
        try:
            cv.imshow("", None)  # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            self.assertEqual(handler_called[0], True)
            pass

        cv.redirectError(None)
        try:
            cv.imshow("", None)  # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass

    def test_overload_resolution_can_choose_correct_overload(self):
        val = 123
        point = (51, 165)
        self.assertEqual(cv.utils.testOverloadResolution(val, point),
                         'overload (int={}, point=(x={}, y={}))'.format(val, *point),
                         "Can't select first overload if all arguments are provided as positional")

        self.assertEqual(cv.utils.testOverloadResolution(val, point=point),
                         'overload (int={}, point=(x={}, y={}))'.format(val, *point),
                         "Can't select first overload if one of the arguments are provided as keyword")

        self.assertEqual(cv.utils.testOverloadResolution(val),
                         'overload (int={}, point=(x=42, y=24))'.format(val),
                         "Can't select first overload if one of the arguments has default value")

        rect = (1, 5, 10, 23)
        self.assertEqual(cv.utils.testOverloadResolution(rect),
                         'overload (rect=(x={}, y={}, w={}, h={}))'.format(*rect),
                         "Can't select second overload if all arguments are provided")

    def test_overload_resolution_fails(self):
        def test_overload_resolution(msg, *args, **kwargs):
            no_exception_msg = 'Overload resolution failed without any exception for: "{}"'.format(msg)
            wrong_exception_msg = 'Overload resolution failed with wrong exception type for: "{}"'.format(msg)
            with self.assertRaises((cv.error, Exception), msg=no_exception_msg) as cm:
                res = cv.utils.testOverloadResolution(*args, **kwargs)
                self.fail("Unexpected result for {}: '{}'".format(msg, res))
            self.assertEqual(type(cm.exception), cv.error, wrong_exception_msg)

        test_overload_resolution('wrong second arg type (keyword arg)', 5, point=(1, 2, 3))
        test_overload_resolution('wrong second arg type', 5, 2)
        test_overload_resolution('wrong first arg', 3.4, (12, 21))
        test_overload_resolution('wrong first arg, no second arg', 4.5)
        test_overload_resolution('wrong args number for first overload', 3, (12, 21), 123)
        test_overload_resolution('wrong args number for second overload', (3, 12, 12, 1), (12, 21))
        # One of the common problems
        test_overload_resolution('rect with float coordinates', (4.5, 4, 2, 1))
        test_overload_resolution('rect with wrong number of coordinates', (4, 4, 1))

    def test_properties_with_reserved_keywords_names_are_transformed(self):
        obj = cv.utils.ClassWithKeywordProperties(except_arg=23)
        self.assertTrue(hasattr(obj, "lambda_"),
                        msg="Class doesn't have RW property with converted name")
        try:
            obj.lambda_ = 32
        except Exception as e:
            self.fail("Failed to set value to RW property. Error: {}".format(e))

        self.assertTrue(hasattr(obj, "except_"),
                        msg="Class doesn't have readonly property with converted name")
        self.assertEqual(obj.except_, 23,
                         msg="Can't access readonly property value")
        with self.assertRaises(AttributeError):
            obj.except_ = 32

    def test_maketype(self):
        data = {
            cv.CV_8UC3: [cv.CV_8U, 3, cv.CV_8UC],
            cv.CV_16SC1: [cv.CV_16S, 1, cv.CV_16SC],
            cv.CV_32FC4: [cv.CV_32F, 4, cv.CV_32FC],
            cv.CV_64FC2: [cv.CV_64F, 2, cv.CV_64FC],
            cv.CV_8SC4: [cv.CV_8S, 4, cv.CV_8SC],
            cv.CV_16UC2: [cv.CV_16U, 2, cv.CV_16UC],
            cv.CV_32SC1: [cv.CV_32S, 1, cv.CV_32SC],
            cv.CV_16FC3: [cv.CV_16F, 3, cv.CV_16FC],
        }
        for ref, (depth, channels, func) in data.items():
            self.assertEqual(ref, cv.CV_MAKETYPE(depth, channels))
            self.assertEqual(ref, func(channels))


class Arguments(NewOpenCVTests):

    def _try_to_convert(self, conversion, value):
        try:
            result = conversion(value).lower()
        except Exception as e:
            self.fail(
                '{} "{}" is risen for conversion {} of type {}'.format(
                    type(e).__name__, e, value, type(value).__name__
                )
            )
        else:
            return result

    def test_InputArray(self):
        res1 = cv.utils.dumpInputArray(None)
        # self.assertEqual(res1, "InputArray: noArray()")  # not supported
        self.assertEqual(res1, "InputArray: empty()=true kind=0x00010000 flags=0x01010000 total(-1)=0 dims(-1)=0 size(-1)=0x0 type(-1)=CV_8UC1")
        res2_1 = cv.utils.dumpInputArray((1, 2))
        self.assertEqual(res2_1, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=2 dims(-1)=2 size(-1)=1x2 type(-1)=CV_64FC1")
        res2_2 = cv.utils.dumpInputArray(1.5)  # Scalar(1.5, 1.5, 1.5, 1.5)
        self.assertEqual(res2_2, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=4 dims(-1)=2 size(-1)=1x4 type(-1)=CV_64FC1")
        a = np.array([[1, 2], [3, 4], [5, 6]])
        res3 = cv.utils.dumpInputArray(a)  # 32SC1
        self.assertEqual(res3, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=6 dims(-1)=2 size(-1)=2x3 type(-1)=CV_32SC1")
        a = np.array([[[1, 2], [3, 4], [5, 6]]], dtype='f')
        res4 = cv.utils.dumpInputArray(a)  # 32FC2
        self.assertEqual(res4, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=3 dims(-1)=2 size(-1)=3x1 type(-1)=CV_32FC2")
        a = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=float)
        res5 = cv.utils.dumpInputArray(a)  # 64FC2
        self.assertEqual(res5, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=3 dims(-1)=2 size(-1)=1x3 type(-1)=CV_64FC2")
        a = np.zeros((2,3,4), dtype='f')
        res6 = cv.utils.dumpInputArray(a)
        self.assertEqual(res6, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=6 dims(-1)=2 size(-1)=3x2 type(-1)=CV_32FC4")
        a = np.zeros((2,3,4,5), dtype='f')
        res7 = cv.utils.dumpInputArray(a)
        self.assertEqual(res7, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=120 dims(-1)=4 size(-1)=[2 3 4 5] type(-1)=CV_32FC1")

    def test_InputArrayOfArrays(self):
        res1 = cv.utils.dumpInputArrayOfArrays(None)
        # self.assertEqual(res1, "InputArray: noArray()")  # not supported
        self.assertEqual(res1, "InputArrayOfArrays: empty()=true kind=0x00050000 flags=0x01050000 total(-1)=0 dims(-1)=1 size(-1)=0x0")
        res2_1 = cv.utils.dumpInputArrayOfArrays((1, 2))  # { Scalar:all(1), Scalar::all(2) }
        self.assertEqual(res2_1, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_64FC1 dims(0)=2 size(0)=1x4")
        res2_2 = cv.utils.dumpInputArrayOfArrays([1.5])
        self.assertEqual(res2_2, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=1 dims(-1)=1 size(-1)=1x1 type(0)=CV_64FC1 dims(0)=2 size(0)=1x4")
        a = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        res3 = cv.utils.dumpInputArrayOfArrays([a, b])
        self.assertEqual(res3, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_32SC1 dims(0)=2 size(0)=2x3")
        c = np.array([[[1, 2], [3, 4], [5, 6]]], dtype='f')
        res4 = cv.utils.dumpInputArrayOfArrays([c, a, b])
        self.assertEqual(res4, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=3 dims(-1)=1 size(-1)=3x1 type(0)=CV_32FC2 dims(0)=2 size(0)=3x1")
        a = np.zeros((2,3,4), dtype='f')
        res5 = cv.utils.dumpInputArrayOfArrays([a, b])
        self.assertEqual(res5, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_32FC4 dims(0)=2 size(0)=3x2")
        # TODO: fix conversion error
        #a = np.zeros((2,3,4,5), dtype='f')
        #res6 = cv.utils.dumpInputArray([a, b])
        #self.assertEqual(res6, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_32FC1 dims(0)=4 size(0)=[2 3 4 5]")

    def test_unsupported_numpy_data_types_string_description(self):
        for dtype in (object, str, np.complex128):
            test_array = np.zeros((4, 4, 3), dtype=dtype)
            msg = ".*type = {} is not supported".format(test_array.dtype)
            if sys.version_info[0] < 3:
                self.assertRaisesRegexp(
                    Exception, msg, cv.utils.dumpInputArray, test_array
                )
            else:
                self.assertRaisesRegex(
                    Exception, msg, cv.utils.dumpInputArray, test_array
                )

    def test_numpy_writeable_flag_is_preserved(self):
        array = np.zeros((10, 10, 1), dtype=np.uint8)
        array.setflags(write=False)
        with self.assertRaises(Exception):
            cv.rectangle(array, (0, 0), (5, 5), (255), 2)

    def test_20968(self):
        pixel = np.uint8([[[40, 50, 200]]])
        _ = cv.cvtColor(pixel, cv.COLOR_RGB2BGR)  # should not raise exception

    def test_parse_to_bool_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpBool)
        for convertible_true in (True, 1, 64, np.int8(123), np.int16(11), np.int32(2),
                                 np.int64(1), np.bool_(12)):
            actual = try_to_convert(convertible_true)
            self.assertEqual('bool: true', actual,
                             msg=get_conversion_error_msg(convertible_true, 'bool: true', actual))

        for convertible_false in (False, 0, np.uint8(0), np.bool_(0), np.int_(0)):
            actual = try_to_convert(convertible_false)
            self.assertEqual('bool: false', actual,
                             msg=get_conversion_error_msg(convertible_false, 'bool: false', actual))

    def test_parse_to_bool_not_convertible(self):
        for not_convertible in (1.2, np.float32(2.3), 's', 'str', (1, 2), [1, 2], complex(1, 1),
                                complex(imag=2), complex(1.1), np.array([1, 0], dtype=bool)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpBool(not_convertible)

    def test_parse_to_bool_convertible_extra(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpBool)
        _, max_size_t = get_limits(ctypes.c_size_t)
        for convertible_true in (-1, max_size_t):
            actual = try_to_convert(convertible_true)
            self.assertEqual('bool: true', actual,
                             msg=get_conversion_error_msg(convertible_true, 'bool: true', actual))

    def test_parse_to_bool_not_convertible_extra(self):
        for not_convertible in (np.array([False]), np.array([True])):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpBool(not_convertible)

    def test_parse_to_int_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpInt)
        min_int, max_int = get_limits(ctypes.c_int)
        for convertible in (-10, -1, 2, int(43.2), np.uint8(15), np.int8(33), np.int16(-13),
                            np.int32(4), np.int64(345), (23), min_int, max_int, np.int_(33)):
            expected = 'int: {0:d}'.format(convertible)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_int_not_convertible(self):
        min_int, max_int = get_limits(ctypes.c_int)
        for not_convertible in (1.2, float(3), np.float32(4), np.double(45), 's', 'str',
                                np.array([1, 2]), (1,), [1, 2], min_int - 1, max_int + 1,
                                complex(1, 1), complex(imag=2), complex(1.1)):
            with self.assertRaises((TypeError, OverflowError, ValueError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpInt(not_convertible)

    def test_parse_to_int_not_convertible_extra(self):
        for not_convertible in (np.bool_(True), True, False, np.float32(2.3),
                                np.array([3, ], dtype=int), np.array([-2, ], dtype=np.int32),
                                np.array([11, ], dtype=np.uint8)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpInt(not_convertible)

    def test_parse_to_int64_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpInt64)
        min_int64, max_int64 = get_limits(ctypes.c_longlong)
        for convertible in (-10, -1, 2, int(43.2), np.uint8(15), np.int8(33), np.int16(-13),
                            np.int32(4), np.int64(345), (23), min_int64, max_int64, np.int_(33)):
            expected = 'int64: {0:d}'.format(convertible)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_int64_not_convertible(self):
        min_int64, max_int64 = get_limits(ctypes.c_longlong)
        for not_convertible in (1.2, np.float32(4), float(3), np.double(45), 's', 'str',
                                np.array([1, 2]), (1,), [1, 2], min_int64 - 1, max_int64 + 1,
                                complex(1, 1), complex(imag=2), complex(1.1), np.bool_(True),
                                True, False, np.float32(2.3), np.array([3, ], dtype=int),
                                np.array([-2, ], dtype=np.int32), np.array([11, ], dtype=np.uint8)):
            with self.assertRaises((TypeError, OverflowError, ValueError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpInt64(not_convertible)

    def test_parse_to_size_t_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpSizeT)
        _, max_uint = get_limits(ctypes.c_uint)
        for convertible in (2, max_uint, (12), np.uint8(34), np.int8(12), np.int16(23),
                            np.int32(123), np.int64(344), np.uint64(3), np.uint16(2), np.uint32(5),
                            np.uint(44)):
            expected = 'size_t: {0:d}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_size_t_not_convertible(self):
        min_long, _ = get_limits(ctypes.c_long)
        for not_convertible in (1.2, True, False, np.bool_(True), np.float32(4), float(3),
                                np.double(45), 's', 'str', np.array([1, 2]), (1,), [1, 2],
                                np.float64(6), complex(1, 1), complex(imag=2), complex(1.1),
                                -1, min_long, np.int8(-35)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpSizeT(not_convertible)

    def test_parse_to_size_t_convertible_extra(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpSizeT)
        _, max_size_t = get_limits(ctypes.c_size_t)
        for convertible in (max_size_t,):
            expected = 'size_t: {0:d}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_size_t_not_convertible_extra(self):
        for not_convertible in (np.bool_(True), True, False, np.array([123, ], dtype=np.uint8),):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpSizeT(not_convertible)

    def test_parse_to_float_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpFloat)
        min_float, max_float = get_limits(ctypes.c_float)
        for convertible in (2, -13, 1.24, np.float32(32.45), float(32), np.double(12.23),
                            np.float32(-12.3), np.float64(3.22), min_float,
                            max_float, np.inf, -np.inf, float('Inf'), -float('Inf'),
                            np.double(np.inf), np.double(-np.inf), np.double(float('Inf')),
                            np.double(-float('Inf'))):
            expected = 'Float: {0:.2f}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

        # Workaround for Windows NaN tests due to Visual C runtime
        # special floating point values (indefinite NaN)
        for nan in (float('NaN'), np.nan, np.float32(np.nan), np.double(np.nan),
                    np.double(float('NaN'))):
            actual = try_to_convert(nan)
            self.assertIn('nan', actual, msg="Can't convert nan of type {} to float. "
                          "Actual: {}".format(type(nan).__name__, actual))

        min_double, max_double = get_limits(ctypes.c_double)
        for inf in (min_float * 10, max_float * 10, min_double, max_double):
            expected = 'float: {}inf'.format('-' if inf < 0 else '')
            actual = try_to_convert(inf)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(inf, expected, actual))

    def test_parse_to_float_not_convertible(self):
        for not_convertible in ('s', 'str', (12,), [1, 2], np.array([1, 2], dtype=float),
                                np.array([1, 2], dtype=np.double), complex(1, 1), complex(imag=2),
                                complex(1.1)):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpFloat(not_convertible)

    def test_parse_to_float_not_convertible_extra(self):
        for not_convertible in (np.bool_(False), True, False, np.array([123, ], dtype=int),
                                np.array([1., ]), np.array([False]),
                                np.array([True])):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpFloat(not_convertible)

    def test_parse_to_double_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpDouble)
        min_float, max_float = get_limits(ctypes.c_float)
        min_double, max_double = get_limits(ctypes.c_double)
        for convertible in (2, -13, 1.24, np.float32(32.45), float(2), np.double(12.23),
                            np.float32(-12.3), np.float64(3.22), min_float,
                            max_float, min_double, max_double, np.inf, -np.inf, float('Inf'),
                            -float('Inf'), np.double(np.inf), np.double(-np.inf),
                            np.double(float('Inf')), np.double(-float('Inf'))):
            expected = 'Double: {0:.2f}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

        # Workaround for Windows NaN tests due to Visual C runtime
        # special floating point values (indefinite NaN)
        for nan in (float('NaN'), np.nan, np.double(np.nan),
                    np.double(float('NaN'))):
            actual = try_to_convert(nan)
            self.assertIn('nan', actual, msg="Can't convert nan of type {} to double. "
                          "Actual: {}".format(type(nan).__name__, actual))

    def test_parse_to_double_not_convertible(self):
        for not_convertible in ('s', 'str', (12,), [1, 2], np.array([1, 2], dtype=np.float32),
                                np.array([1, 2], dtype=np.double), complex(1, 1), complex(imag=2),
                                complex(1.1)):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpDouble(not_convertible)

    def test_parse_to_double_not_convertible_extra(self):
        for not_convertible in (np.bool_(False), True, False, np.array([123, ], dtype=int),
                                np.array([1., ]), np.array([False]),
                                np.array([12.4], dtype=np.double), np.array([True])):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpDouble(not_convertible)

    def test_parse_to_cstring_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpCString)
        for convertible in ('', 's', 'str', str(123), ('char'), np.str_('test2')):
            expected = 'string: ' + convertible
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_cstring_not_convertible(self):
        for not_convertible in ((12,), ('t', 'e', 's', 't'), np.array(['123', ]),
                                np.array(['t', 'e', 's', 't']), 1, -1.4, True, False, None):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpCString(not_convertible)

    def test_parse_to_string_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpString)
        for convertible in (None, '', 's', 'str', str(123), np.str_('test2')):
            expected = 'string: ' + (convertible if convertible else '')
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_string_not_convertible(self):
        for not_convertible in ((12,), ('t', 'e', 's', 't'), np.array(['123', ]),
                                np.array(['t', 'e', 's', 't']), 1, True, False):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpString(not_convertible)

    def test_parse_to_rect_convertible(self):
        Rect = namedtuple('Rect', ('x', 'y', 'w', 'h'))
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpRect)
        for convertible in ((1, 2, 4, 5), [5, 3, 10, 20], np.array([10, 20, 23, 10]),
                            Rect(10, 30, 40, 55), tuple(np.array([40, 20, 24, 20])),
                            list(np.array([20, 40, 30, 35]))):
            expected = 'rect: (x={}, y={}, w={}, h={})'.format(*convertible)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_rect_not_convertible(self):
        for not_convertible in (np.empty(shape=(4, 1)), (), [], np.array([]), (12, ),
                                [3, 4, 5, 10, 123], {1: 2, 3:4, 5:10, 6:30},
                                '1234', np.array([1, 2, 3, 4], dtype=np.float32),
                                np.array([[1, 2], [3, 4], [5, 6], [6, 8]]), (1, 2, 5, 1.5)):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpRect(not_convertible)

    def test_parse_to_rotated_rect_convertible(self):
        RotatedRect = namedtuple('RotatedRect', ('center', 'size', 'angle'))
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpRotatedRect)
        for convertible in (((2.5, 2.5), (10., 20.), 12.5), [[1.5, 10.5], (12.5, 51.5), 10],
                            RotatedRect((10, 40), np.array([10.5, 20.5]), 5),
                            np.array([[10, 6], [50, 50], 5.5], dtype=object)):
            center, size, angle = convertible
            expected = 'rotated_rect: (c_x={:.6f}, c_y={:.6f}, w={:.6f},' \
                       ' h={:.6f}, a={:.6f})'.format(center[0], center[1],
                                                     size[0], size[1], angle)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))


    def test_wrap_rotated_rect(self):
        center = (34.5, 52.)
        size = (565.0, 140.0)
        angle = -177.5
        rect1 = cv.RotatedRect(center, size, angle)
        self.assertEqual(rect1.center, center)
        self.assertEqual(rect1.size, size)
        self.assertEqual(rect1.angle, angle)

        pts = [[ 319.7845, -5.6109037],
               [ 313.6778, 134.25586],
               [-250.78448, 109.6109],
               [-244.6778, -30.25586]]
        self.assertLess(np.max(np.abs(rect1.points() - pts)), 1e-4)

        rect2 = cv.RotatedRect(pts[0], pts[1], pts[2])
        _, inter_pts = cv.rotatedRectangleIntersection(rect1, rect2)
        self.assertLess(np.max(np.abs(inter_pts.reshape(-1, 2) - pts)), 1e-4)

    def test_result_rotated_rect_boundingRect2f(self):
        center = (0, 0)
        size = (10, 10)
        angle = 0
        gold_box = (-5.0, -5.0, 10.0, 10.0)
        rect1 = cv.RotatedRect(center, size, angle)
        bbox = rect1.boundingRect2f()
        self.assertEqual(gold_box, bbox)

    def test_parse_to_rotated_rect_not_convertible(self):
        for not_convertible in ([], (), np.array([]), (123, (45, 34), 1), {1: 2, 3: 4}, 123,
                                np.array([[123, 123, 14], [1, 3], 56], dtype=object), '123'):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpRotatedRect(not_convertible)

    def test_parse_to_term_criteria_convertible(self):
        TermCriteria = namedtuple('TermCriteria', ('type', 'max_count', 'epsilon'))
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpTermCriteria)
        for convertible in ((1, 10, 1e-3), [2, 30, 1e-1], np.array([10, 20, 0.5], dtype=object),
                            TermCriteria(0, 5, 0.1)):
            expected = 'term_criteria: (type={}, max_count={}, epsilon={:.6f}'.format(*convertible)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_term_criteria_not_convertible(self):
        for not_convertible in ([], (), np.array([]), [1, 4], (10,), (1.5, 34, 0.1),
                                {1: 5, 3: 5, 10: 10}, '145'):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpTermCriteria(not_convertible)

    def test_parse_to_range_convertible_to_all(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpRange)
        for convertible in ((), [], np.array([])):
            expected = 'range: all'
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_range_convertible(self):
        Range = namedtuple('Range', ('start', 'end'))
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpRange)
        for convertible in ((10, 20), [-1, 3], np.array([10, 24]), Range(-4, 6)):
            expected = 'range: (s={}, e={})'.format(*convertible)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_range_not_convertible(self):
        for not_convertible in ((1, ), [40, ], np.array([1, 4, 6]), {'a': 1, 'b': 40},
                                (1.5, 13.5), [3, 6.7], np.array([6.3, 2.1]), '14, 4'):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpRange(not_convertible)

    def test_reserved_keywords_are_transformed(self):
        default_lambda_value = 2
        default_from_value = 3
        format_str = "arg={}, lambda={}, from={}"
        self.assertEqual(
            cv.utils.testReservedKeywordConversion(20), format_str.format(20, default_lambda_value, default_from_value)
        )
        self.assertEqual(
            cv.utils.testReservedKeywordConversion(10, lambda_=10), format_str.format(10, 10, default_from_value)
        )
        self.assertEqual(
            cv.utils.testReservedKeywordConversion(10, from_=10), format_str.format(10, default_lambda_value, 10)
        )
        self.assertEqual(
            cv.utils.testReservedKeywordConversion(20, lambda_=-4, from_=12), format_str.format(20, -4, 12)
        )

    def test_parse_vector_int_convertible(self):
        np.random.seed(123098765)
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpVectorOfInt)
        arr = np.random.randint(-20, 20, 40).astype(np.int32).reshape(10, 2, 2)
        int_min, int_max = get_limits(ctypes.c_int)
        for convertible in ((int_min, 1, 2, 3, int_max), [40, 50], tuple(),
                            np.array([int_min, -10, 24, int_max], dtype=np.int32),
                            np.array([10, 230, 12], dtype=np.uint8), arr[:, 0, 1],):
            expected = "[" + ", ".join(map(str, convertible)) + "]"
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_vector_int_not_convertible(self):
        np.random.seed(123098765)
        arr = np.random.randint(-20, 20, 40).astype(np.float32).reshape(10, 2, 2)
        int_min, int_max = get_limits(ctypes.c_int)
        test_dict = {1: 2, 3: 10, 10: 20}
        for not_convertible in ((int_min, 1, 2.5, 3, int_max), [True, 50], 'test', test_dict,
                                reversed([1, 2, 3]),
                                np.array([int_min, -10, 24, [1, 2]], dtype=object),
                                np.array([[1, 2], [3, 4]]), arr[:, 0, 1],):
            with self.assertRaises(TypeError, msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpVectorOfInt(not_convertible)

    def test_parse_vector_double_convertible(self):
        np.random.seed(1230965)
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpVectorOfDouble)
        arr = np.random.randint(-20, 20, 40).astype(np.int32).reshape(10, 2, 2)
        for convertible in ((1, 2.12, 3.5), [40, 50], tuple(),
                            np.array([-10, 24], dtype=np.int32),
                            np.array([-12.5, 1.4], dtype=np.double),
                            np.array([10, 230, 12], dtype=np.float32), arr[:, 0, 1], ):
            expected = "[" + ", ".join(map(lambda v: "{:.2f}".format(v), convertible)) + "]"
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_vector_double_not_convertible(self):
        test_dict = {1: 2, 3: 10, 10: 20}
        for not_convertible in (('t', 'e', 's', 't'), [True, 50.55], 'test', test_dict,
                                np.array([-10.1, 24.5, [1, 2]], dtype=object),
                                np.array([[1, 2], [3, 4]]),):
            with self.assertRaises(TypeError, msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpVectorOfDouble(not_convertible)

    def test_parse_vector_rect_convertible(self):
        np.random.seed(1238765)
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpVectorOfRect)
        arr_of_rect_int32 = np.random.randint(5, 20, 4 * 3).astype(np.int32).reshape(3, 4)
        arr_of_rect_cast = np.random.randint(10, 40, 4 * 5).astype(np.uint8).reshape(5, 4)
        for convertible in (((1, 2, 3, 4), (10, -20, 30, 10)), arr_of_rect_int32, arr_of_rect_cast,
                            arr_of_rect_int32.astype(np.int8), [[5, 3, 1, 4]],
                            ((np.int8(4), np.uint8(10), int(32), np.int16(55)),)):
            expected = "[" + ", ".join(map(lambda v: "[x={}, y={}, w={}, h={}]".format(*v), convertible)) + "]"
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_vector_rect_not_convertible(self):
        np.random.seed(1238765)
        arr = np.random.randint(5, 20, 4 * 3).astype(np.float32).reshape(3, 4)
        for not_convertible in (((1, 2, 3, 4), (10.5, -20, 30.1, 10)), arr,
                                [[5, 3, 1, 4], []],
                                ((float(4), np.uint8(10), int(32), np.int16(55)),)):
            with self.assertRaises(TypeError, msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpVectorOfRect(not_convertible)

    def test_vector_general_return(self):
        expected_number_of_mats = 5
        expected_shape = (10, 10, 3)
        expected_type = np.uint8
        mats = cv.utils.generateVectorOfMat(5, 10, 10, cv.CV_8UC3)
        self.assertTrue(isinstance(mats, tuple),
                        "Vector of Mats objects should be returned as tuple. Got: {}".format(type(mats)))
        self.assertEqual(len(mats), expected_number_of_mats, "Returned array has wrong length")
        for mat in mats:
            self.assertEqual(mat.shape, expected_shape, "Returned Mat has wrong shape")
            self.assertEqual(mat.dtype, expected_type, "Returned Mat has wrong elements type")
        empty_mats = cv.utils.generateVectorOfMat(0, 10, 10, cv.CV_32FC1)
        self.assertTrue(isinstance(empty_mats, tuple),
                        "Empty vector should be returned as empty tuple. Got: {}".format(type(mats)))
        self.assertEqual(len(empty_mats), 0, "Vector of size 0 should be returned as tuple of length 0")

    def test_vector_fast_return(self):
        expected_shape = (5, 4)
        rects = cv.utils.generateVectorOfRect(expected_shape[0])
        self.assertTrue(isinstance(rects, np.ndarray),
                        "Vector of rectangles should be returned as numpy array. Got: {}".format(type(rects)))
        self.assertEqual(rects.dtype, np.int32, "Vector of rectangles has wrong elements type")
        self.assertEqual(rects.shape, expected_shape, "Vector of rectangles has wrong shape")
        empty_rects = cv.utils.generateVectorOfRect(0)
        self.assertTrue(isinstance(empty_rects, tuple),
                        "Empty vector should be returned as empty tuple. Got: {}".format(type(empty_rects)))
        self.assertEqual(len(empty_rects), 0, "Vector of size 0 should be returned as tuple of length 0")

        expected_shape = (10,)
        ints = cv.utils.generateVectorOfInt(expected_shape[0])
        self.assertTrue(isinstance(ints, np.ndarray),
                        "Vector of integers should be returned as numpy array. Got: {}".format(type(ints)))
        self.assertEqual(ints.dtype, np.int32, "Vector of integers has wrong elements type")
        self.assertEqual(ints.shape, expected_shape, "Vector of integers has wrong shape.")

    def test_result_rotated_rect_issue_20930(self):
        rr = cv.utils.testRotatedRect(10, 20, 100, 200, 45)
        self.assertTrue(isinstance(rr, tuple), msg=type(rr))
        self.assertEqual(len(rr), 3)

        rrv = cv.utils.testRotatedRectVector(10, 20, 100, 200, 45)
        self.assertTrue(isinstance(rrv, tuple), msg=type(rrv))
        self.assertEqual(len(rrv), 10)

        rr = rrv[0]
        self.assertTrue(isinstance(rr, tuple), msg=type(rrv))
        self.assertEqual(len(rr), 3)

    def test_array_range(self):
        m = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9],
                      [10, 11, 12, 13, 14],
                      [15, 16, 17, 18, 19]], dtype=np.uint8)
        roi_min, roi_max, _, _ = cv.minMaxLoc(m[1:3, 1:4])
        self.assertEqual(roi_min, 6)
        self.assertEqual(roi_max, 13)

    def test_nested_function_availability(self):
        self.assertTrue(hasattr(cv.utils, "nested"),
                        msg="Module is not generated for nested namespace")
        self.assertTrue(hasattr(cv.utils.nested, "testEchoBooleanFunction"),
                        msg="Function in nested module is not available")

        if sys.version_info[0] < 3:
            # Nested submodule is managed only by the global submodules dictionary
            # and parent native module
            expected_ref_count = 2
        else:
            # Nested submodule is managed by the global submodules dictionary,
            # parent native module and Python part of the submodule
            expected_ref_count = 3

        # `getrefcount` temporary increases reference counter by 1
        actual_ref_count = sys.getrefcount(cv.utils.nested) - 1

        self.assertEqual(actual_ref_count, expected_ref_count,
                         msg="Nested submodule reference counter has wrong value\n"
                         "Expected: {}. Actual: {}".format(expected_ref_count, actual_ref_count))
        for flag in (True, False):
            self.assertEqual(flag, cv.utils.nested.testEchoBooleanFunction(flag),
                             msg="Function in nested module returns wrong result")

    def test_class_from_submodule_has_global_alias(self):
        self.assertTrue(hasattr(cv.ml, "Boost"),
                        msg="Class is not registered in the submodule")
        self.assertTrue(hasattr(cv, "ml_Boost"),
                        msg="Class from submodule doesn't have alias in the "
                        "global module")
        self.assertEqual(cv.ml.Boost, cv.ml_Boost,
                         msg="Classes from submodules and global module don't refer "
                         "to the same type")

    def test_inner_class_has_global_alias(self):
        self.assertTrue(hasattr(cv.SimpleBlobDetector, "Params"),
                        msg="Class is not registered as inner class")
        self.assertTrue(hasattr(cv, "SimpleBlobDetector_Params"),
                        msg="Inner class doesn't have alias in the global module")
        self.assertEqual(cv.SimpleBlobDetector.Params, cv.SimpleBlobDetector_Params,
                         msg="Inner class and class in global module don't refer "
                         "to the same type")

    def test_export_class_with_different_name(self):
        self.assertTrue(hasattr(cv.utils.nested, "ExportClassName"),
                        msg="Class with export alias is not registered in the submodule")
        self.assertTrue(hasattr(cv, "utils_nested_ExportClassName"),
                        msg="Class with export alias doesn't have alias in the "
                        "global module")
        self.assertEqual(cv.utils.nested.ExportClassName.originalName(), "OriginalClassName")

        instance = cv.utils.nested.ExportClassName.create()
        self.assertTrue(isinstance(instance, cv.utils.nested.ExportClassName),
                        msg="Factory function returns wrong class instance: {}".format(type(instance)))
        self.assertTrue(hasattr(cv.utils.nested, "ExportClassName_create"),
                        msg="Factory function should have alias in the same module as the class")
        # self.assertFalse(hasattr(cv.utils.nested, "OriginalClassName_create"),
        #                  msg="Factory function should not be registered with original class name, "\
        #                  "when class has different export name")

    def test_export_inner_class_of_class_exported_with_different_name(self):
        if not hasattr(cv.utils.nested, "ExportClassName"):
            raise unittest.SkipTest(
                "Outer class with export alias is not registered in the submodule")

        self.assertTrue(hasattr(cv.utils.nested.ExportClassName, "Params"),
                        msg="Inner class with export alias is not registered in "
                        "the outer class")
        self.assertTrue(hasattr(cv, "utils_nested_ExportClassName_Params"),
                        msg="Inner class with export alias is not registered in "
                        "global module")
        params = cv.utils.nested.ExportClassName.Params()
        params.int_value = 45
        params.float_value = 4.5

        instance = cv.utils.nested.ExportClassName.create(params)
        self.assertTrue(isinstance(instance, cv.utils.nested.ExportClassName),
                        msg="Factory function returns wrong class instance: {}".format(type(instance)))
        self.assertEqual(
            params.int_value, instance.getIntParam(),
            msg="Class initialized with wrong integer parameter. Expected: {}. Actual: {}".format(
                params.int_value, instance.getIntParam()
            )
        )
        self.assertEqual(
            params.float_value, instance.getFloatParam(),
            msg="Class initialized with wrong integer parameter. Expected: {}. Actual: {}".format(
                params.float_value, instance.getFloatParam()
            )
        )

    def test_named_arguments_without_parameters(self):
        src = np.ones((5, 5, 3), dtype=np.uint8)
        arguments_dump, src_copy = cv.utils.copyMatAndDumpNamedArguments(src)
        np.testing.assert_equal(src, src_copy)
        self.assertEqual(arguments_dump, 'lambda=-1, sigma=0.0')

    def test_named_arguments_without_output_argument(self):
        src = np.zeros((2, 2, 3), dtype=np.uint8)
        arguments_dump, src_copy = cv.utils.copyMatAndDumpNamedArguments(
            src, lambda_=15, sigma=3.5
        )
        np.testing.assert_equal(src, src_copy)
        self.assertEqual(arguments_dump, 'lambda=15, sigma=3.5')

    def test_named_arguments_with_output_argument(self):
        src = np.zeros((3, 3, 3), dtype=np.uint8)
        dst = np.ones_like(src)
        arguments_dump, src_copy = cv.utils.copyMatAndDumpNamedArguments(
            src, dst, lambda_=25, sigma=5.5
        )
        np.testing.assert_equal(src, src_copy)
        np.testing.assert_equal(dst, src_copy)
        self.assertEqual(arguments_dump, 'lambda=25, sigma=5.5')

    def test_arithm_op_without_saturation(self):
        np.random.seed(4231568)
        src = np.random.randint(20, 40, 8 * 4 * 3).astype(np.uint8).reshape(8, 4, 3)
        operations = get_ocv_arithm_op_table(apply_saturation=False)
        for ocv_op, numpy_op in operations.items():
            for val in (2, 4, (5, ), (6, 4), (2., 4., 1.),
                        np.uint8([1, 2, 2]), np.float64([5, 2, 6, 3]),):
                dst = ocv_op(src, val)
                expected = numpy_op(src, val)
                # Temporarily allows a difference of 1 for arm64 workaround.
                self.assertLess(np.max(np.abs(dst - expected)), 2,
                  msg="Operation '{}' is failed for {}".format(ocv_op.__name__, val ) )

    def test_arithm_op_with_saturation(self):
        np.random.seed(4231568)
        src = np.random.randint(20, 40, 4 * 8 * 4).astype(np.uint8).reshape(4, 8, 4)
        operations = get_ocv_arithm_op_table(apply_saturation=True)

        for ocv_op, numpy_op in operations.items():
            for val in (10, 4, (40, ), (15, 12), (25., 41., 15.),
                        np.uint8([1, 2, 20]), np.float64([50, 21, 64, 30]),):
                dst = ocv_op(src, val)
                expected = numpy_op(src, val)
                # Temporarily allows a difference of 1 for arm64 workaround.
                self.assertLess(np.max(np.abs(dst - expected)), 2,
                  msg="Saturated Operation '{}' is failed for {}".format(ocv_op.__name__, val ) )

class CanUsePurePythonModuleFunction(NewOpenCVTests):
    def test_can_get_ocv_version(self):
        import sys
        if sys.version_info[0] < 3:
            raise unittest.SkipTest('Python 2.x is not supported')

        self.assertEqual(cv.misc.get_ocv_version(), cv.__version__,
                         "Can't get package version using Python misc module")

    def test_native_method_can_be_patched(self):
        import sys

        if sys.version_info[0] < 3:
            raise unittest.SkipTest('Python 2.x is not supported')

        res = cv.utils.testOverwriteNativeMethod(10)
        self.assertTrue(isinstance(res, Sequence),
                        msg="Overwritten method should return sequence. "
                            "Got: {} of type {}".format(res, type(res)))
        self.assertSequenceEqual(res, (11, 10),
                                 msg="Failed to overwrite native method")
        res = cv.utils._native.testOverwriteNativeMethod(123)
        self.assertEqual(res, 123, msg="Failed to call native method implementation")

    def test_default_matx_argument(self):
        res = cv.utils.dumpVec2i()
        self.assertEqual(res, "Vec2i(42, 24)",
                         msg="Default argument is not properly handled")
        res = cv.utils.dumpVec2i((12, 21))
        self.assertEqual(res, "Vec2i(12, 21)")


class SamplesFindFile(NewOpenCVTests):

    def test_ExistedFile(self):
        res = cv.samples.findFile('HappyFish.jpg', False)
        self.assertNotEqual(res, '')

    def test_MissingFile(self):
        res = cv.samples.findFile('non_existed.file', False)
        self.assertEqual(res, '')

    def test_MissingFileException(self):
        try:
            _res = cv.samples.findFile('non_existed.file', True)
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass

class AlgorithmImplHit(NewOpenCVTests):
    def test_callable(self):
        res = cv.getDefaultAlgorithmHint()
        self.assertTrue(res is not None)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
