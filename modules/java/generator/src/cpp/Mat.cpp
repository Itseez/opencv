// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core.hpp"

#define LOG_TAG "org.opencv.core.Mat"
#include "common.h"

using namespace cv;

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
  std::string what = "unknown exception";
  jclass je = 0;

  if(e) {
    std::string exception_type = "std::exception";

    if(dynamic_cast<const cv::Exception*>(e)) {
      exception_type = "cv::Exception";
      je = env->FindClass("org/opencv/core/CvException");
    }

    what = exception_type + ": " + e->what();
  }

  if(!je) je = env->FindClass("java/lang/Exception");
  env->ThrowNew(je, what.c_str());

  LOGE("%s caught %s", method, what.c_str());
  CV_UNUSED(method);        // avoid "unused" warning
}

// jint could be int or int32_t so casting jint* to int* in general wouldn't work
static std::vector<int> convertJintArrayToVector(JNIEnv* env, jintArray in) {
    std::vector<int> out;
    int len = env->GetArrayLength(in);
    jint* inArray = env->GetIntArrayElements(in, 0);
    for ( int i = 0; i < len; i++ ) {
        out.push_back(inArray[i]);
    }
    env->ReleaseIntArrayElements(in, inArray, 0);
    return out;
}

extern "C" {


//
//   MatXXX::MatXXX()
//


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__
  (JNIEnv*, jclass);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__
  (JNIEnv*, jclass)
{
    LOGD("Mat::n_1Mat__()");
    return (jlong) new cv::Mat();
}



//
//   Mat::Mat(int rows, int cols, int type, void* data)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIILjava_nio_ByteBuffer_2
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jobject data);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIILjava_nio_ByteBuffer_2
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jobject data)
{
    static const char method_name[] = "Mat::n_1Mat__IIILjava_nio_ByteBuffer_2()";
    try {
        LOGD("%s", method_name);
        return (jlong) new Mat( rows, cols, type, (void*)env->GetDirectBufferAddress(data) );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


/*
 * Class:     org_opencv_core_Mat
 * Method:    n_Mat
 * Signature: (IIILjava/nio/ByteBuffer;J)J
 *
 * Mat::Mat(int rows, int cols, int type, void* data, size_t step)
 */
JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIILjava_nio_ByteBuffer_2J
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jobject data, jlong step);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIILjava_nio_ByteBuffer_2J
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jobject data, jlong step)
{
    static const char method_name[] = "Mat::n_1Mat__IIILjava_nio_ByteBuffer_2J()";
    try {
        LOGD("%s", method_name);
        return (jlong) new Mat(rows, cols, type, (void*)env->GetDirectBufferAddress(data), (size_t)step);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


//
//   Mat::Mat(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    static const char method_name[] = "Mat::n_1Mat__III()";
    try {
        LOGD("%s", method_name);
        return (jlong) new Mat( rows, cols, type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//   Mat::Mat(int[] sizes, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__I_3II
  (JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__I_3II
  (JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type)
{
    static const char method_name[] = "Mat::n_1Mat__I_3II()";
    try {
        LOGD("%s", method_name);
        std::vector<int> sizes = convertJintArrayToVector(env, sizesArray);
        return (jlong) new Mat( ndims, sizes.data(), type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//   Mat::Mat(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    static const char method_name[] = "Mat::n_1Mat__DDI()";
    try {
        LOGD("%s", method_name);
        Size size((int)size_width, (int)size_height);
        return (jlong) new Mat( size, type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//   Mat::Mat(int rows, int cols, int type, Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIIDDDD
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIIDDDD
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    static const char method_name[] = "Mat::n_1Mat__IIIDDDD()";
    try {
        LOGD("%s", method_name);
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        return (jlong) new Mat( rows, cols, type, s );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//   Mat::Mat(Size size, int type, Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDIDDDD
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDIDDDD
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    static const char method_name[] = "Mat::n_1Mat__DDIDDDD()";
    try {
        LOGD("%s", method_name);
        Size size((int)size_width, (int)size_height);
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        return (jlong) new Mat( size, type, s );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//   Mat::Mat(int[] sizes, int type, Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__I_3IIDDDD
  (JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__I_3IIDDDD
  (JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    static const char method_name[] = "Mat::n_1Mat__I_3IIDDDD()";
    try {
        LOGD("%s", method_name);
        std::vector<int> sizes = convertJintArrayToVector(env, sizesArray);
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        return (jlong) new Mat( ndims, sizes.data(), type, s );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//   Mat::Mat(Mat m, Range rowRange, Range colRange = Range::all())
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JIIII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JIIII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end)
{
    static const char method_name[] = "Mat::n_1Mat__JIIII()";
    try {
        LOGD("%s", method_name);
        Range rowRange(rowRange_start, rowRange_end);
        Range colRange(colRange_start, colRange_end);
        return (jlong) new Mat( (*(Mat*)m_nativeObj), rowRange, colRange );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

jint getObjectIntField(JNIEnv* env, jobject obj, const char * fieldName);

jint getObjectIntField(JNIEnv* env, jobject obj, const char * fieldName) {
    jfieldID fid; /* store the field ID */

    /* Get a reference to obj's class */
    jclass cls = env->GetObjectClass(obj);

    /* Look for the instance field s in cls */
    fid = env->GetFieldID(cls, fieldName, "I");
    if (fid == NULL)
    {
        return 0; /* failed to find the field */
    }

    /* Read the instance field s */
    return env->GetIntField(obj, fid);
}

#define RANGE_START_FIELD    "start"
#define RANGE_END_FIELD      "end"

//
//   Mat::Mat(Mat m, Range[] ranges)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__J_3Lorg_opencv_core_Range_2
  (JNIEnv* env, jclass, jlong m_nativeObj, jobjectArray rangesArray);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__J_3Lorg_opencv_core_Range_2
  (JNIEnv* env, jclass, jlong m_nativeObj, jobjectArray rangesArray)
{
    static const char method_name[] = "Mat::n_1Mat__J_3Lorg_opencv_core_Range_2()";
    try {
        LOGD("%s", method_name);
        std::vector<Range> ranges;
        int rangeCount = env->GetArrayLength(rangesArray);
        for (int i = 0; i < rangeCount; i++) {
            jobject range = env->GetObjectArrayElement(rangesArray, i);
            jint start = getObjectIntField(env, range, RANGE_START_FIELD);
            jint end = getObjectIntField(env, range, RANGE_END_FIELD);
            ranges.push_back(Range(start, end));
        }
        return (jlong) new Mat( (*(Mat*)m_nativeObj), ranges );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end);


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end)
{
    static const char method_name[] = "Mat::n_1Mat__JII()";
    try {
        LOGD("%s", method_name);
        Range rowRange(rowRange_start, rowRange_end);
        return (jlong) new Mat( (*(Mat*)m_nativeObj), rowRange );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


//
//  Mat Mat::adjustROI(int dtop, int dbottom, int dleft, int dright)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1adjustROI
  (JNIEnv* env, jclass, jlong self, jint dtop, jint dbottom, jint dleft, jint dright);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1adjustROI
  (JNIEnv* env, jclass, jlong self, jint dtop, jint dbottom, jint dleft, jint dright)
{
    static const char method_name[] = "Mat::n_1adjustROI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->adjustROI( dtop, dbottom, dleft, dright );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  void Mat::assignTo(Mat m, int type = -1)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint type)
{
    static const char method_name[] = "Mat::n_1assignTo__JJI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->assignTo( (*(Mat*)m_nativeObj), type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}


JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1assignTo__JJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->assignTo( (*(Mat*)m_nativeObj) );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  int Mat::channels()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1channels
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1channels
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1channels()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->channels(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  int Mat::checkVector(int elemChannels, int depth = -1, bool requireContinuous = true)
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JIIZ
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth, jboolean requireContinuous);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JIIZ
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth, jboolean requireContinuous)
{
    static const char method_name[] = "Mat::n_1checkVector__JIIZ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->checkVector( elemChannels, depth, requireContinuous );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JII
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JII
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth)
{
    static const char method_name[] = "Mat::n_1checkVector__JII()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->checkVector( elemChannels, depth );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JI
  (JNIEnv* env, jclass, jlong self, jint elemChannels);


JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JI
  (JNIEnv* env, jclass, jlong self, jint elemChannels)
{
    static const char method_name[] = "Mat::n_1checkVector__JI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->checkVector( elemChannels );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::clone()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1clone
  (JNIEnv* env, jclass, jlong self);


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1clone
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1clone()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->clone(  );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::col(int x)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1col
  (JNIEnv* env, jclass, jlong self, jint x);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1col
  (JNIEnv* env, jclass, jlong self, jint x)
{
    static const char method_name[] = "Mat::n_1col()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->col( x );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::colRange(int startcol, int endcol)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1colRange
  (JNIEnv* env, jclass, jlong self, jint startcol, jint endcol);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1colRange
  (JNIEnv* env, jclass, jlong self, jint startcol, jint endcol)
{
    static const char method_name[] = "Mat::n_1colRange()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->colRange( startcol, endcol );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  int Mat::dims()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1dims
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1dims
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1dims()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->dims;
    } catch(const cv::Exception& e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  int Mat::cols()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1cols
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1cols
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1cols()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->cols;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//  int Mat::size(int i)
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1size_1i__JI
  (JNIEnv* env, jclass, jlong self, jint i);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1size_1i__JI
  (JNIEnv* env, jclass, jlong self, jint i)
{
    static const char method_name[] = "Mat::n_1size_1i__JI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->size[i];
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//  void Mat::convertTo(Mat& m, int rtype, double alpha = 1, double beta = 0)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJIDD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha, jdouble beta);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJIDD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha, jdouble beta)
{
    static const char method_name[] = "Mat::n_1convertTo__JJIDD()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->convertTo( m, rtype, alpha, beta );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}


JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJID
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJID
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha)
{
    static const char method_name[] = "Mat::n_1convertTo__JJID()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->convertTo( m, rtype, alpha );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}


JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype)
{
    static const char method_name[] = "Mat::n_1convertTo__JJI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->convertTo( m, rtype );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  void Mat::copyTo(Mat& m)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1copyTo__JJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->copyTo( m );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  void Mat::copyTo(Mat& m, Mat mask)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jlong mask_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jlong mask_nativeObj)
{
    static const char method_name[] = "Mat::n_1copyTo__JJJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat& mask = *((Mat*)mask_nativeObj);
        me->copyTo( m, mask );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  void Mat::create(int rows, int cols, int type)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JIII
  (JNIEnv* env, jclass, jlong self, jint rows, jint cols, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JIII
  (JNIEnv* env, jclass, jlong self, jint rows, jint cols, jint type)
{
    static const char method_name[] = "Mat::n_1create__JIII()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->create( rows, cols, type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  void Mat::create(Size size, int type)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JDDI
  (JNIEnv* env, jclass, jlong self, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JDDI
  (JNIEnv* env, jclass, jlong self, jdouble size_width, jdouble size_height, jint type)
{
    static const char method_name[] = "Mat::n_1create__JDDI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Size size((int)size_width, (int)size_height);
        me->create( size, type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  void Mat::create(int[] sizes, int type)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JI_3II
  (JNIEnv* env, jclass, jlong self, jint ndims, jintArray sizesArray, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JI_3II
  (JNIEnv* env, jclass, jlong self, jint ndims, jintArray sizesArray, jint type)
{
    static const char method_name[] = "Mat::n_1create__JI_3II()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self;
        std::vector<int> sizes = convertJintArrayToVector(env, sizesArray);
        me->create( ndims, sizes.data(), type );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  Mat Mat::copySize(Mat m)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copySize
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copySize
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1copySize()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self;
        Mat& m = *((Mat*)m_nativeObj);
        me->copySize( m );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  Mat Mat::cross(Mat m)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1cross
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1cross
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1cross()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = me->cross( m );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  long Mat::dataAddr()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1dataAddr
  (JNIEnv*, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1dataAddr
  (JNIEnv*, jclass, jlong self)
{
    LOGD("Mat::n_1dataAddr()");
    Mat* me = (Mat*) self; //TODO: check for NULL
    return (jlong) me->data;
}



//
//  int Mat::depth()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1depth
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1depth
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1depth()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->depth(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::diag(int d = 0)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__JI
  (JNIEnv* env, jclass, jlong self, jint d);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__JI
  (JNIEnv* env, jclass, jlong self, jint d)
{
    static const char method_name[] = "Mat::n_1diag__JI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->diag( d );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}




//
// static Mat Mat::diag(Mat d)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__J
  (JNIEnv* env, jclass, jlong d_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__J
  (JNIEnv* env, jclass, jlong d_nativeObj)
{
    static const char method_name[] = "Mat::n_1diag__J()";
    try {
        LOGD("%s", method_name);
        Mat _retval_ = Mat::diag( (*(Mat*)d_nativeObj) );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  double Mat::dot(Mat m)
//

JNIEXPORT jdouble JNICALL Java_org_opencv_core_Mat_n_1dot
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jdouble JNICALL Java_org_opencv_core_Mat_n_1dot
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1dot()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        return me->dot( m );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  size_t Mat::elemSize()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1elemSize()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->elemSize(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  size_t Mat::elemSize1()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize1
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize1
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1elemSize1()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->elemSize1(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  bool Mat::empty()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1empty
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1empty
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1empty()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->empty(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::eye(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    static const char method_name[] = "Mat::n_1eye__III()";
    try {
        LOGD("%s", method_name);
        Mat _retval_ = Mat::eye( rows, cols, type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::eye(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    static const char method_name[] = "Mat::n_1eye__DDI()";
    try {
        LOGD("%s", method_name);
        Size size((int)size_width, (int)size_height);
        Mat _retval_ = Mat::eye( size, type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::inv(int method = DECOMP_LU)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__JI
  (JNIEnv* env, jclass, jlong self, jint method);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__JI
  (JNIEnv* env, jclass, jlong self, jint method)
{
    static const char method_name[] = "Mat::n_1inv__JI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->inv( method );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__J
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__J
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1inv__J()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->inv(  );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  bool Mat::isContinuous()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isContinuous
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isContinuous
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1isContinuous()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->isContinuous(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  bool Mat::isSubmatrix()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isSubmatrix
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isSubmatrix
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1isSubmatrix()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->isSubmatrix(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  void Mat::locateROI(Size wholeSize, Point ofs)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_locateROI_10
  (JNIEnv* env, jclass, jlong self, jdoubleArray wholeSize_out, jdoubleArray ofs_out);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_locateROI_10
  (JNIEnv* env, jclass, jlong self, jdoubleArray wholeSize_out, jdoubleArray ofs_out)
{
    static const char method_name[] = "core::locateROI_10()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Size wholeSize;
        Point ofs;
        me->locateROI( wholeSize, ofs );
        jdouble tmp_wholeSize[2] = {(jdouble)wholeSize.width, (jdouble)wholeSize.height}; env->SetDoubleArrayRegion(wholeSize_out, 0, 2, tmp_wholeSize);  jdouble tmp_ofs[2] = {(jdouble)ofs.x, (jdouble)ofs.y}; env->SetDoubleArrayRegion(ofs_out, 0, 2, tmp_ofs);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  Mat Mat::mul(Mat m, double scale = 1)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jdouble scale);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jdouble scale)
{
    static const char method_name[] = "Mat::n_1mul__JJD()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = me->mul( m, scale );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1mul__JJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = me->mul( m );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


//
// Mat Mat Mat::matMul(Mat m)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1matMul__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1matMul__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1matMul__JJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = (*me) * m;
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


//
// static Mat Mat::ones(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    static const char method_name[] = "Mat::n_1ones__III()";
    try {
        LOGD("%s", method_name);
        Mat _retval_ = Mat::ones( rows, cols, type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::ones(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    static const char method_name[] = "Mat::n_1ones__DDI()";
    try {
        LOGD("%s", method_name);
        Size size((int)size_width, (int)size_height);
        Mat _retval_ = Mat::ones( size, type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::ones(int[] sizes, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__I_3II
  (JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__I_3II
  (JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type)
{
    static const char method_name[] = "Mat::n_1ones__I_3II()";
    try {
        LOGD("%s", method_name);
        std::vector<int> sizes = convertJintArrayToVector(env, sizesArray);
        Mat _retval_ = Mat::ones( ndims, sizes.data(), type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  void Mat::push_back(Mat m)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1push_1back
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1push_1back
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    static const char method_name[] = "Mat::n_1push_1back()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->push_back( (*(Mat*)m_nativeObj) );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  void Mat::release()
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1release
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1release
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1release()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->release(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
}



//
//  Mat Mat::reshape(int cn, int rows = 0)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JII
  (JNIEnv* env, jclass, jlong self, jint cn, jint rows);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JII
  (JNIEnv* env, jclass, jlong self, jint cn, jint rows)
{
    static const char method_name[] = "Mat::n_1reshape__JII()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->reshape( cn, rows );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JI
  (JNIEnv* env, jclass, jlong self, jint cn);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JI
  (JNIEnv* env, jclass, jlong self, jint cn)
{
    static const char method_name[] = "Mat::n_1reshape__JI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->reshape( cn );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//  Mat Mat::reshape(int cn, int[] newshape)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape_11
  (JNIEnv* env, jclass, jlong self, jint cn, jint newndims, jintArray newshape);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape_11
  (JNIEnv* env, jclass, jlong self, jint cn, jint newndims, jintArray newshape)
{
    static const char method_name[] = "Mat::n_1reshape_11";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        std::vector<int> newsz = convertJintArrayToVector(env, newshape);
        Mat _retval_ = me->reshape( cn, newndims, newsz.data() );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//  Mat Mat::row(int y)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1row
  (JNIEnv* env, jclass, jlong self, jint y);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1row
  (JNIEnv* env, jclass, jlong self, jint y)
{
    static const char method_name[] = "Mat::n_1row()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->row( y );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::rowRange(int startrow, int endrow)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1rowRange
  (JNIEnv* env, jclass, jlong self, jint startrow, jint endrow);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1rowRange
  (JNIEnv* env, jclass, jlong self, jint startrow, jint endrow)
{
    static const char method_name[] = "Mat::n_1rowRange()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->rowRange( startrow, endrow );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  int Mat::rows()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1rows
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1rows
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1rows()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->rows;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::operator =(Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDD
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDD
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    static const char method_name[] = "Mat::n_1setTo__JDDDD()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        Mat _retval_ = me->operator =( s );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::setTo(Scalar value, Mat mask = Mat())
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDDJ
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3, jlong mask_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDDJ
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3, jlong mask_nativeObj)
{
    static const char method_name[] = "Mat::n_1setTo__JDDDDJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        Mat& mask = *((Mat*)mask_nativeObj);
        Mat _retval_ = me->setTo( s, mask );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::setTo(Mat value, Mat mask = Mat())
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj, jlong mask_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj, jlong mask_nativeObj)
{
    static const char method_name[] = "Mat::n_1setTo__JJJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& value = *((Mat*)value_nativeObj);
        Mat& mask = *((Mat*)mask_nativeObj);
        Mat _retval_ = me->setTo( value, mask );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj)
{
    static const char method_name[] = "Mat::n_1setTo__JJ()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& value = *((Mat*)value_nativeObj);
        Mat _retval_ = me->setTo( value );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Size Mat::size()
//

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_n_1size
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_n_1size
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1size()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Size _retval_ = me->size(  );
        jdoubleArray _da_retval_ = env->NewDoubleArray(2);
        jdouble _tmp_retval_[2] = {(jdouble)_retval_.width, (jdouble)_retval_.height};
        env->SetDoubleArrayRegion(_da_retval_, 0, 2, _tmp_retval_);
        return _da_retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  size_t Mat::step1(int i = 0)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__JI
  (JNIEnv* env, jclass, jlong self, jint i);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__JI
  (JNIEnv* env, jclass, jlong self, jint i)
{
    static const char method_name[] = "Mat::n_1step1__JI()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->step1( i );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__J
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__J
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1step1__J()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->step1(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//  Mat Mat::operator()(Range rowRange, Range colRange)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat_1rr
  (JNIEnv* env, jclass, jlong self, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat_1rr
  (JNIEnv* env, jclass, jlong self, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end)
{
    static const char method_name[] = "Mat::n_1submat_1rr()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Range rowRange(rowRange_start, rowRange_end);
        Range colRange(colRange_start, colRange_end);
        Mat _retval_ = me->operator()( rowRange, colRange );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

//
//  Mat Mat::operator()(Range[] ranges)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat_1ranges
(JNIEnv* env, jclass, jlong self, jobjectArray rangesArray);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat_1ranges
(JNIEnv* env, jclass, jlong self, jobjectArray rangesArray)
{
    static const char method_name[] = "Mat::n_1submat_1ranges()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self;
        std::vector<Range> ranges;
        int rangeCount = env->GetArrayLength(rangesArray);
        for (int i = 0; i < rangeCount; i++) {
            jobject range = env->GetObjectArrayElement(rangesArray, i);
            jint start = getObjectIntField(env, range, RANGE_START_FIELD);
            jint end = getObjectIntField(env, range, RANGE_END_FIELD);
            ranges.push_back(Range(start, end));
        }
        Mat _retval_ = me->operator()( ranges );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::operator()(Rect roi)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat
  (JNIEnv* env, jclass, jlong self, jint roi_x, jint roi_y, jint roi_width, jint roi_height);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat
  (JNIEnv* env, jclass, jlong self, jint roi_x, jint roi_y, jint roi_width, jint roi_height)
{
    static const char method_name[] = "Mat::n_1submat()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Rect roi(roi_x, roi_y, roi_width, roi_height);
        Mat _retval_ = me->operator()( roi );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  Mat Mat::t()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1t
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1t
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1t()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->t(  );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  size_t Mat::total()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1total
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1total
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1total()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->total(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  int Mat::type()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1type
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1type
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "Mat::n_1type()";
    try {
        LOGD("%s", method_name);
        Mat* me = (Mat*) self; //TODO: check for NULL
        return me->type(  );
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::zeros(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    static const char method_name[] = "Mat::n_1zeros__III()";
    try {
        LOGD("%s", method_name);
        Mat _retval_ = Mat::zeros( rows, cols, type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::zeros(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    static const char method_name[] = "Mat::n_1zeros__DDI()";
    try {
        LOGD("%s", method_name);
        Size size((int)size_width, (int)size_height);
        Mat _retval_ = Mat::zeros( size, type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
// static Mat Mat::zeros(int[] sizes, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__I_3II
(JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__I_3II
(JNIEnv* env, jclass, jint ndims, jintArray sizesArray, jint type)
{
    static const char method_name[] = "Mat::n_1zeros__I_3II()";
    try {
        LOGD("%s", method_name);
        std::vector<int> sizes = convertJintArrayToVector(env, sizesArray);
        Mat _retval_ = Mat::zeros( ndims, sizes.data(), type );
        return (jlong) new Mat(_retval_);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}



//
//  native support for deleting native object
//  static void Mat::n_delete( __int64 self )
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1delete
  (JNIEnv*, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1delete
  (JNIEnv*, jclass, jlong self)
{
    delete (Mat*) self;
}

} // extern "C"

namespace {
  /// map java-array-types to assigned data
  template<class T> struct JavaOpenCVTrait;

/// less typing for specialisations
#define JOCvT(t,s,c1,c2) \
  template<> struct JavaOpenCVTrait<t##Array> { \
    typedef t value_type;    /* type of array element */ \
    static const char get[]; /* name of getter */ \
    static const char put[]; /* name of putter */ \
    enum {cvtype_1 = c1, cvtype_2 = c2 }; /* allowed OpenCV-types */ \
  }; \
  const char JavaOpenCVTrait<t##Array>::get[] = "Mat::nGet" s "()"; \
  const char JavaOpenCVTrait<t##Array>::put[] = "Mat::nPut" s "()"

  JOCvT(jbyte, "B", CV_8U, CV_8S);
  JOCvT(jshort, "S", CV_16U, CV_16S);
  JOCvT(jint, "I", CV_32S, CV_32S);
  JOCvT(jfloat, "F", CV_32F, CV_32F);
  JOCvT(jdouble, "D", CV_64F, CV_64F);
#undef JOCvT
}

static size_t idx2Offset(cv::Mat* mat, std::vector<int>& indices) {
    size_t offset = indices[0];
    for (int dim=1; dim < mat->dims; dim++) {
        offset = offset*mat->size[dim] + indices[dim];
    }
    return offset;
}

static void offset2Idx(cv::Mat* mat, size_t offset, std::vector<int>& indices) {
    for (int dim=mat->dims-1; dim>=0; dim--) {
        indices[dim] = offset % mat->size[dim];
        offset = (offset - indices[dim]) / mat->size[dim];
    }
}

// returns true if final index was reached
static bool updateIdx(cv::Mat* mat, std::vector<int>& indices, size_t inc) {
    size_t currentOffset = idx2Offset(mat, indices);
    size_t newOffset = currentOffset + inc;
    bool reachedEnd = newOffset>=(size_t)mat->total();
    offset2Idx(mat, reachedEnd?0:newOffset, indices);
    return reachedEnd;
}

template<typename T> static int mat_copy_data(cv::Mat* m, std::vector<int>& idx, int count, char* buff, bool isPut) {
    if(! m) return 0;
    if(! buff) return 0;

    size_t countBytes = count * sizeof(T);
    size_t remainingBytes = (size_t)(m->total() - idx2Offset(m, idx))*m->elemSize();
    countBytes = (countBytes>remainingBytes)?remainingBytes:countBytes;
    int res = (int)countBytes;

    if( m->isContinuous() )
    {
        if (isPut) {
            memcpy(m->ptr(idx.data()), buff, countBytes);
        } else {
            memcpy(buff, m->ptr(idx.data()), countBytes);
        }
    } else {
        size_t blockSize = m->size[m->dims-1] * m->elemSize();
        size_t firstPartialBlockSize = (m->size[m->dims-1] - idx[m->dims-1]) * m->step[m->dims-1];;
        for (int dim=m->dims-2; dim>=0 && blockSize == m->step[dim]; dim--) {
            blockSize *= m->size[dim];
            firstPartialBlockSize += (m->size[dim] - (idx[dim]+1)) * m->step[dim];
        }
        size_t copyCount = (countBytes<firstPartialBlockSize)?countBytes:firstPartialBlockSize;
        uchar* data = m->ptr(idx.data());
        while(countBytes>0){
            if (isPut) {
                memcpy(data, buff, copyCount);
            } else {
                memcpy(buff, data, copyCount);
            }
            updateIdx(m, idx, copyCount / m->elemSize());
            countBytes -= copyCount;
            buff += copyCount;
            copyCount = countBytes<blockSize?countBytes:blockSize;
            data = m->ptr(idx.data());
        }
    }
    return res;
}

template<typename T> static int mat_put_idx(cv::Mat* m, std::vector<int>& idx, int count, int offset, char* buff)
{
    return mat_copy_data<T>(m, idx, count, buff + offset, true);
}

template<typename T> static int mat_put(cv::Mat* m, int row, int col, int count, int offset, char* buff)
{
    int indicesArray[] = { row, col };
    std::vector<int> indices(indicesArray, indicesArray+2);
    return mat_put_idx<T>(m, indices, count, offset, buff);
}

template<class ARRAY> static jint java_mat_put(JNIEnv* env, jlong self, jint row, jint col, jint count, jint offset, ARRAY vals)
{
    static const char *method_name = JavaOpenCVTrait<ARRAY>::put;
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_1 && me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_2) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_put<typename JavaOpenCVTrait<ARRAY>::value_type>(me, row, col, count, offset, values);
        env->ReleasePrimitiveArrayCritical(vals, values, JNI_ABORT);
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

template<class ARRAY> static jint java_mat_put_idx(JNIEnv* env, jlong self, jintArray idxArray, jint count, jint offset, ARRAY vals)
{
    static const char *method_name = JavaOpenCVTrait<ARRAY>::put;
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_1 && me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_2) return 0; // incompatible type
        std::vector<int> idx = convertJintArrayToVector(env, idxArray);
        for (int i = 0; i < me->dims ; i++ ) {
            if (me->size[i]<=idx[i]) return 0;
        }
        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_put_idx<typename JavaOpenCVTrait<ARRAY>::value_type>(me, idx, count, offset, values);
        env->ReleasePrimitiveArrayCritical(vals, values, JNI_ABORT);
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

extern "C" {

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals)
{
  return java_mat_put(env, self, row, col, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutBIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutBIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jbyteArray vals)
{
    return java_mat_put_idx(env, self, idxArray, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutBwOffset
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jint offset, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutBwOffset
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jint offset, jbyteArray vals)
{
  return java_mat_put(env, self, row, col, count, offset, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutBwIdxOffset
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jint offset, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutBwIdxOffset
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jint offset, jbyteArray vals)
{
    return java_mat_put_idx(env, self, idxArray, count, offset, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals)
{
  return java_mat_put(env, self, row, col, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutSIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jshortArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutSIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jshortArray vals)
{
    return java_mat_put_idx(env, self, idxArray, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals)
{
  return java_mat_put(env, self, row, col, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutIIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jintArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutIIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jintArray vals)
{
    return java_mat_put_idx(env, self, idxArray, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals)
{
  return java_mat_put(env, self, row, col, count, 0, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutFIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jfloatArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutFIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jfloatArray vals)
{
    return java_mat_put_idx(env, self, idxArray, count, 0, vals);
}

// unlike other nPut()-s this one (with double[]) should convert input values to correct type
#define PUT_ITEM(T, R, C) { T*dst = (T*)me->ptr(R, C); for(int ch=0; ch<me->channels() && count>0; count--,ch++,src++,dst++) *dst = cv::saturate_cast<T>(*src); }

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals)
{
    static const char* method_name = JavaOpenCVTrait<jdoubleArray>::put;
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(!me || !me->data) return 0;  // no native object behind
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        int rest = ((me->rows - row) * me->cols - col) * me->channels();
        if(count>rest) count = rest;
        int res = count;
        double* values = (double*)env->GetPrimitiveArrayCritical(vals, 0);
        double* src = values;
        int r, c;
        for(c=col; c<me->cols && count>0; c++)
        {
            switch(me->depth()) {
                case CV_8U:  PUT_ITEM(uchar,  row, c); break;
                case CV_8S:  PUT_ITEM(schar,  row, c); break;
                case CV_16U: PUT_ITEM(ushort, row, c); break;
                case CV_16S: PUT_ITEM(short,  row, c); break;
                case CV_32S: PUT_ITEM(int,    row, c); break;
                case CV_32F: PUT_ITEM(float,  row, c); break;
                case CV_64F: PUT_ITEM(double, row, c); break;
            }
        }

        for(r=row+1; r<me->rows && count>0; r++)
            for(c=0; c<me->cols && count>0; c++)
            {
                switch(me->depth()) {
                    case CV_8U:  PUT_ITEM(uchar,  r, c); break;
                    case CV_8S:  PUT_ITEM(schar,  r, c); break;
                    case CV_16U: PUT_ITEM(ushort, r, c); break;
                    case CV_16S: PUT_ITEM(short,  r, c); break;
                    case CV_32S: PUT_ITEM(int,    r, c); break;
                    case CV_32F: PUT_ITEM(float,  r, c); break;
                    case CV_64F: PUT_ITEM(double, r, c); break;
                }
            }

        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

// unlike other nPut()-s this one (with double[]) should convert input values to correct type
#define PUT_ITEM_IDX(T, I) { T*dst = (T*)me->ptr(I); for(int ch=0; ch<me->channels() && count>0; count--,ch++,src++,dst++) *dst = cv::saturate_cast<T>(*src); }

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutDIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jdoubleArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutDIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jdoubleArray vals)
{
    static const char* method_name = JavaOpenCVTrait<jdoubleArray>::put;
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(!me || !me->data) return 0;  // no native object behind
        std::vector<int> idx = convertJintArrayToVector(env, idxArray);
        for (int i=0; i<me->dims; i++) {
            if (me->size[i]<=idx[i]) return 0; // indexes out of range
        }
        int rest = me->channels();
        for (int i=0; i<me->dims; i++) {
            rest *= (me->size[i] - idx[i]);
        }
        if(count>rest) count = rest;
        int res = count;
        double* values = (double*)env->GetPrimitiveArrayCritical(vals, 0);
        double* src = values;
        bool reachedFinalIndex = false;
        for(; !reachedFinalIndex && count>0; reachedFinalIndex = updateIdx(me, idx, 1))
        {
            switch(me->depth()) {
                case CV_8U:  PUT_ITEM_IDX(uchar,  idx.data()); break;
                case CV_8S:  PUT_ITEM_IDX(schar,  idx.data()); break;
                case CV_16U: PUT_ITEM_IDX(ushort, idx.data()); break;
                case CV_16S: PUT_ITEM_IDX(short,  idx.data()); break;
                case CV_32S: PUT_ITEM_IDX(int,    idx.data()); break;
                case CV_32F: PUT_ITEM_IDX(float,  idx.data()); break;
                case CV_64F: PUT_ITEM_IDX(double, idx.data()); break;
            }
        }
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

} // extern "C"

template<typename T> static int mat_get_idx(cv::Mat* m, std::vector<int>& idx, int count, char* buff)
{
    return mat_copy_data<T>(m, idx, count, buff, false);
}

template<typename T> static int mat_get(cv::Mat* m, int row, int col, int count, char* buff)
{
    int indicesArray[] = { row, col };
    std::vector<int> indices(indicesArray, indicesArray+2);
    return mat_get_idx<T>(m, indices, count, buff);
}

template<class ARRAY> static jint java_mat_get(JNIEnv* env, jlong self, jint row, jint col, jint count, ARRAY vals) {
    static const char *method_name = JavaOpenCVTrait<ARRAY>::get;
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_1 && me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_2) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get<typename JavaOpenCVTrait<ARRAY>::value_type>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

template<class ARRAY> static jint java_mat_get_idx(JNIEnv* env, jlong self, jintArray idxArray, jint count, ARRAY vals) {
    static const char *method_name = JavaOpenCVTrait<ARRAY>::get;
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_1 && me->depth() != JavaOpenCVTrait<ARRAY>::cvtype_2) return 0; // incompatible type
        std::vector<int> idx = convertJintArrayToVector(env, idxArray);
        for (int i = 0; i < me->dims ; i++ ) {
            if (me->size[i]<=idx[i]) return 0;
        }

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get_idx<typename JavaOpenCVTrait<ARRAY>::value_type>(me, idx, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

extern "C" {

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals)
{
  return java_mat_get(env, self, row, col, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetBIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetBIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jbyteArray vals)
{
    return java_mat_get_idx(env, self, idxArray, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals)
{
  return java_mat_get(env, self, row, col, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetSIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jshortArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetSIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jshortArray vals)
{
    return java_mat_get_idx(env, self, idxArray, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals)
{
  return java_mat_get(env, self, row, col, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetIIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jintArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetIIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jintArray vals)
{
    return java_mat_get_idx(env, self, idxArray, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals)
{
  return java_mat_get(env, self, row, col, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetFIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jfloatArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetFIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jfloatArray vals)
{
    return java_mat_get_idx(env, self, idxArray, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals)
{
  return java_mat_get(env, self, row, col, count, vals);
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetDIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jdoubleArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetDIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray, jint count, jdoubleArray vals)
{
    return java_mat_get_idx(env, self, idxArray, count, vals);
}

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_nGet
    (JNIEnv* env, jclass, jlong self, jint row, jint col);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_nGet
    (JNIEnv* env, jclass, jlong self, jint row, jint col)
{
    static const char method_name[] = "Mat::nGet()";
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        jdoubleArray res = env->NewDoubleArray(me->channels());
        if(res){
            jdouble buff[CV_CN_MAX];//me->channels()
            int i;
            switch(me->depth()){
                case CV_8U:  for(i=0; i<me->channels(); i++) buff[i] = *((unsigned char*) me->ptr(row, col) + i); break;
                case CV_8S:  for(i=0; i<me->channels(); i++) buff[i] = *((signed char*)   me->ptr(row, col) + i); break;
                case CV_16U: for(i=0; i<me->channels(); i++) buff[i] = *((unsigned short*)me->ptr(row, col) + i); break;
                case CV_16S: for(i=0; i<me->channels(); i++) buff[i] = *((signed short*)  me->ptr(row, col) + i); break;
                case CV_32S: for(i=0; i<me->channels(); i++) buff[i] = *((int*)           me->ptr(row, col) + i); break;
                case CV_32F: for(i=0; i<me->channels(); i++) buff[i] = *((float*)         me->ptr(row, col) + i); break;
                case CV_64F: for(i=0; i<me->channels(); i++) buff[i] = *((double*)        me->ptr(row, col) + i); break;
            }
            env->SetDoubleArrayRegion(res, 0, me->channels(), buff);
        }
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_nGetIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_nGetIdx
    (JNIEnv* env, jclass, jlong self, jintArray idxArray)
{
    static const char method_name[] = "Mat::nGetIdx()";
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        std::vector<int> idx = convertJintArrayToVector(env, idxArray);
        for (int i=0; i<me->dims; i++) {
            if (me->size[i]<=idx[i]) return 0; // indexes out of range
        }

        jdoubleArray res = env->NewDoubleArray(me->channels());
        if(res){
            jdouble buff[CV_CN_MAX];//me->channels()
            int i;
            switch(me->depth()){
                case CV_8U:  for(i=0; i<me->channels(); i++) buff[i] = *((unsigned char*) me->ptr(idx.data()) + i); break;
                case CV_8S:  for(i=0; i<me->channels(); i++) buff[i] = *((signed char*)   me->ptr(idx.data()) + i); break;
                case CV_16U: for(i=0; i<me->channels(); i++) buff[i] = *((unsigned short*)me->ptr(idx.data()) + i); break;
                case CV_16S: for(i=0; i<me->channels(); i++) buff[i] = *((signed short*)  me->ptr(idx.data()) + i); break;
                case CV_32S: for(i=0; i<me->channels(); i++) buff[i] = *((int*)           me->ptr(idx.data()) + i); break;
                case CV_32F: for(i=0; i<me->channels(); i++) buff[i] = *((float*)         me->ptr(idx.data()) + i); break;
                case CV_64F: for(i=0; i<me->channels(); i++) buff[i] = *((double*)        me->ptr(idx.data()) + i); break;
            }
            env->SetDoubleArrayRegion(res, 0, me->channels(), buff);
        }
        return res;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

JNIEXPORT jstring JNICALL Java_org_opencv_core_Mat_nDump
  (JNIEnv *env, jclass, jlong self);

JNIEXPORT jstring JNICALL Java_org_opencv_core_Mat_nDump
  (JNIEnv *env, jclass, jlong self)
{
    static const char method_name[] = "Mat::nDump()";
    try {
        LOGD("%s", method_name);
        cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
        String s;
        Ptr<Formatted> fmtd = Formatter::get()->format(*me);
        for(const char* str = fmtd->next(); str; str = fmtd->next())
        {
            s = s + String(str);
        }
        return env->NewStringUTF(s.c_str());
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}


} // extern "C"
