// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////// tests for matrix operations and math functions ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "test_precomp.hpp"
#include <float.h>
#include <math.h>
#include "opencv2/core/softfloat.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace opencv_test { namespace {

/// !!! NOTE !!! These tests happily avoid overflow cases & out-of-range arguments
/// so that output arrays contain neigher Inf's nor Nan's.
/// Handling such cases would require special modification of check function
/// (validate_test_results) => TBD.
/// Also, need some logarithmic-scale generation of input data. Right now it is done (in some tests)
/// by generating min/max boundaries for random data in logarimithic scale, but
/// within the same test case all the input array elements are of the same order.

class Core_MathTest : public cvtest::ArrayTest
{
public:
    typedef cvtest::ArrayTest Base;
    Core_MathTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types);
    double get_success_error_level( int /*test_case_idx*/, int i, int j );
    bool test_nd;
};


Core_MathTest::Core_MathTest()
{
    optional_mask = false;

    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    test_nd = false;
}


double Core_MathTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    return test_mat[i][j].depth() == CV_32F ? FLT_EPSILON*128 : DBL_EPSILON*1024;
}


void Core_MathTest::get_test_array_types_and_sizes( int test_case_idx,
                                                     vector<vector<Size> >& sizes,
                                                     vector<vector<int> >& types)
{
    RNG& rng = cv::theRNG();
    int depth = cvtest::randInt(rng)%2 + CV_32F;
    int cn = cvtest::randInt(rng) % 4 + 1, type = CV_MAKETYPE(depth, cn);
    size_t i, j;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    for( i = 0; i < test_array.size(); i++ )
    {
        size_t count = test_array[i].size();
        for( j = 0; j < count; j++ )
            types[i][j] = type;
    }
    test_nd = cvtest::randInt(rng)%3 == 0;
}


////////// pow /////////////

class Core_PowTest : public Core_MathTest
{
public:
    typedef Core_MathTest Base;
    Core_PowTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                        vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );
    double power;
};


Core_PowTest::Core_PowTest()
{
    power = 0;
}


void Core_PowTest::get_test_array_types_and_sizes( int test_case_idx,
                                                    vector<vector<Size> >& sizes,
                                                    vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int depth = cvtest::randInt(rng) % (CV_64F+1);
    int cn = cvtest::randInt(rng) % 4 + 1;
    size_t i, j;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth += depth == CV_8S;

    if( depth < CV_32F || cvtest::randInt(rng)%8 == 0 )
        // integer power
        power = (int)(cvtest::randInt(rng)%21 - 10);
    else
    {
        i = cvtest::randInt(rng)%17;
        power = i == 16 ? 1./3 : i == 15 ? 0.5 : i == 14 ? -0.5 : cvtest::randReal(rng)*10 - 5;
    }

    for( i = 0; i < test_array.size(); i++ )
    {
        size_t count = test_array[i].size();
        int type = CV_MAKETYPE(depth, cn);
        for( j = 0; j < count; j++ )
            types[i][j] = type;
    }
    test_nd = cvtest::randInt(rng)%3 == 0;
}


double Core_PowTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = test_mat[i][j].depth();
    if( depth < CV_32F )
        return power == cvRound(power) && power >= 0 ? 0 : 1;
    else
    {
        return depth != CV_64F ? Base::get_success_error_level( test_case_idx, i, j ) : DBL_EPSILON*1024*1.11;
    }
}


void Core_PowTest::get_minmax_bounds( int /*i*/, int /*j*/, int type, Scalar& low, Scalar& high )
{
    double l, u = cvtest::randInt(cv::theRNG())%1000 + 1;
    if( power > 0 )
    {
        double mval = cvtest::getMaxVal(type);
        double u1 = pow(mval,1./power)*2;
        u = MIN(u,u1);
    }

    l = power == cvRound(power) ? -u : FLT_EPSILON;
    low = Scalar::all(l);
    high = Scalar::all(u);
}


void Core_PowTest::run_func()
{
    if(!test_nd)
    {
        if( fabs(power-1./3) <= DBL_EPSILON && test_mat[INPUT][0].depth() == CV_32F )
        {
            Mat a = test_mat[INPUT][0], b = test_mat[OUTPUT][0];

            a = a.reshape(1);
            b = b.reshape(1);
            for( int i = 0; i < a.rows; i++ )
            {
                b.at<float>(i,0) = (float)fabs(cubeRoot(a.at<float>(i,0)));
                for( int j = 1; j < a.cols; j++ )
                    b.at<float>(i,j) = (float)fabs(cv::cubeRoot(a.at<float>(i,j)));
            }
        }
        else
            cv::pow( test_mat[INPUT][0], power, test_mat[OUTPUT][0] );
    }
    else
    {
        Mat& a = test_mat[INPUT][0];
        Mat& b = test_mat[OUTPUT][0];
        if(power == 0.5)
            cv::sqrt(a, b);
        else
            cv::pow(a, power, b);
    }
}


inline static int ipow( int a, int power )
{
    int b = 1;
    while( power > 0 )
    {
        if( power&1 )
            b *= a, power--;
        else
            a *= a, power >>= 1;
    }
    return b;
}


inline static double ipow( double a, int power )
{
    double b = 1.;
    while( power > 0 )
    {
        if( power&1 )
            b *= a, power--;
        else
            a *= a, power >>= 1;
    }
    return b;
}


void Core_PowTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& a = test_mat[INPUT][0];
    Mat& b = test_mat[REF_OUTPUT][0];

    int depth = a.depth();
    int ncols = a.cols*a.channels();
    int ipower = cvRound(power), apower = abs(ipower);
    int i, j;

    for( i = 0; i < a.rows; i++ )
    {
        const uchar* a_data = a.ptr(i);
        uchar* b_data = b.ptr(i);

        switch( depth )
        {
            case CV_8U:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((uchar*)a_data)[j];
                        ((uchar*)b_data)[j] = (uchar)(val == 0 ? 255 : val == 1 ? 1 :
                                                      val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((uchar*)a_data)[j];
                        val = ipow( val, ipower );
                        ((uchar*)b_data)[j] = saturate_cast<uchar>(val);
                    }
                break;
            case CV_8S:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((schar*)a_data)[j];
                        ((schar*)b_data)[j] = (schar)(val == 0 ? 127 : val == 1 ? 1 :
                                                    val ==-1 ? 1-2*(ipower&1) :
                                                    val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((schar*)a_data)[j];
                        val = ipow( val, ipower );
                        ((schar*)b_data)[j] = saturate_cast<schar>(val);
                    }
                break;
            case CV_16U:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((ushort*)a_data)[j];
                        ((ushort*)b_data)[j] = (ushort)(val == 0 ? 65535 : val == 1 ? 1 :
                                                        val ==-1 ? 1-2*(ipower&1) :
                                                        val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((ushort*)a_data)[j];
                        val = ipow( val, ipower );
                        ((ushort*)b_data)[j] = saturate_cast<ushort>(val);
                    }
                break;
            case CV_16S:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((short*)a_data)[j];
                        ((short*)b_data)[j] = (short)(val == 0 ? 32767 : val == 1 ? 1 :
                                                      val ==-1 ? 1-2*(ipower&1) :
                                                      val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((short*)a_data)[j];
                        val = ipow( val, ipower );
                        ((short*)b_data)[j] = saturate_cast<short>(val);
                    }
                break;
            case CV_32S:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((int*)a_data)[j];
                        ((int*)b_data)[j] = val == 0 ? INT_MAX : val == 1 ? 1 :
                        val ==-1 ? 1-2*(ipower&1) :
                        val == 2 && ipower == -1 ? 1 : 0;
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((int*)a_data)[j];
                        val = ipow( val, ipower );
                        ((int*)b_data)[j] = val;
                    }
                break;
            case CV_32F:
                if( power != ipower )
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((float*)a_data)[j];
                        val = pow( fabs(val), power );
                        ((float*)b_data)[j] = (float)val;
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((float*)a_data)[j];
                        if( ipower < 0 )
                            val = 1./val;
                        val = ipow( val, apower );
                        ((float*)b_data)[j] = (float)val;
                    }
                break;
            case CV_64F:
                if( power != ipower )
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((double*)a_data)[j];
                        val = pow( fabs(val), power );
                        ((double*)b_data)[j] = (double)val;
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((double*)a_data)[j];
                        if( ipower < 0 )
                            val = 1./val;
                        val = ipow( val, apower );
                        ((double*)b_data)[j] = (double)val;
                    }
                break;
        }
    }
}

///////////////////////////////////////// matrix tests ////////////////////////////////////////////

class Core_MatrixTest : public cvtest::ArrayTest
{
public:
    typedef cvtest::ArrayTest Base;
    Core_MatrixTest( int in_count, int out_count,
                       bool allow_int, bool scalar_output, int max_cn );
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                        vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    bool allow_int;
    bool scalar_output;
    int max_cn;
};


Core_MatrixTest::Core_MatrixTest( int in_count, int out_count,
                                      bool _allow_int, bool _scalar_output, int _max_cn )
: allow_int(_allow_int), scalar_output(_scalar_output), max_cn(_max_cn)
{
    int i;
    for( i = 0; i < in_count; i++ )
        test_array[INPUT].push_back(NULL);

    for( i = 0; i < out_count; i++ )
    {
        test_array[OUTPUT].push_back(NULL);
        test_array[REF_OUTPUT].push_back(NULL);
    }

    element_wise_relative_error = false;
}


void Core_MatrixTest::get_test_array_types_and_sizes( int test_case_idx,
                                                       vector<vector<Size> >& sizes,
                                                       vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int depth = cvtest::randInt(rng) % (allow_int ? CV_64F+1 : 2);
    int cn = cvtest::randInt(rng) % max_cn + 1;
    size_t i, j;

    if( allow_int )
        depth += depth == CV_8S;
    else
        depth += CV_32F;

    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    for( i = 0; i < test_array.size(); i++ )
    {
        size_t count = test_array[i].size();
        int flag = (i == OUTPUT || i == REF_OUTPUT) && scalar_output;
        int type = !flag ? CV_MAKETYPE(depth, cn) : CV_64FC1;

        for( j = 0; j < count; j++ )
        {
            types[i][j] = type;
            if( flag )
                sizes[i][j] = Size( 4, 1 );
        }
    }
}


double Core_MatrixTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int input_depth = test_mat[INPUT][0].depth();
    double input_precision = input_depth < CV_32F ? 0 : input_depth == CV_32F ? 5e-5 : 5e-10;
    double output_precision = Base::get_success_error_level( test_case_idx, i, j );
    return MAX(input_precision, output_precision);
}


///////////////// Trace /////////////////////

class Core_TraceTest : public Core_MatrixTest
{
public:
    Core_TraceTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_TraceTest::Core_TraceTest() : Core_MatrixTest( 1, 1, true, true, 4 )
{
}


void Core_TraceTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) = cv::trace(test_mat[INPUT][0]);
}


void Core_TraceTest::prepare_to_validation( int )
{
    Mat& mat = test_mat[INPUT][0];
    int count = MIN( mat.rows, mat.cols );
    Mat diag(count, 1, mat.type(), mat.ptr(), mat.step + mat.elemSize());
    Scalar r = cvtest::mean(diag);
    r *= (double)count;

    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = r;
}


///////// dotproduct //////////

class Core_DotProductTest : public Core_MatrixTest
{
public:
    Core_DotProductTest();
protected:
    void run_func() CV_OVERRIDE;
    void prepare_to_validation( int test_case_idx ) CV_OVERRIDE;
    double get_success_error_level( int test_case_idx, int i, int j ) CV_OVERRIDE;
};


Core_DotProductTest::Core_DotProductTest() : Core_MatrixTest( 2, 1, true, true, 4 )
{
}


void Core_DotProductTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) = Scalar(test_mat[INPUT][0].dot(test_mat[INPUT][1]));
}


void Core_DotProductTest::prepare_to_validation( int )
{
    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = Scalar(cvtest::crossCorr( test_mat[INPUT][0], test_mat[INPUT][1] ));
}

double Core_DotProductTest::get_success_error_level( int test_case_idx, int i, int j )
{
#ifdef __riscv
    const int depth = test_mat[i][j].depth();
    if (depth == CV_64F)
        return 1.7e-5;
#endif
    return Core_MatrixTest::get_success_error_level( test_case_idx, i, j );
}

///////// crossproduct //////////

class Core_CrossProductTest : public Core_MatrixTest
{
public:
    Core_CrossProductTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                        vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_CrossProductTest::Core_CrossProductTest() : Core_MatrixTest( 2, 1, false, false, 1 )
{
}


void Core_CrossProductTest::get_test_array_types_and_sizes( int,
                                                             vector<vector<Size> >& sizes,
                                                             vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int depth = cvtest::randInt(rng) % 2 + CV_32F;
    int cn = cvtest::randInt(rng) & 1 ? 3 : 1, type = CV_MAKETYPE(depth, cn);
    Size sz;

    types[INPUT][0] = types[INPUT][1] = types[OUTPUT][0] = types[REF_OUTPUT][0] = type;

    if( cn == 3 )
        sz = Size(1,1);
    else if( cvtest::randInt(rng) & 1 )
        sz = Size(3,1);
    else
        sz = Size(1,3);

    sizes[INPUT][0] = sizes[INPUT][1] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sz;
}


void Core_CrossProductTest::run_func()
{
    test_mat[OUTPUT][0] = test_mat[INPUT][0].cross(test_mat[INPUT][1]);
}


void Core_CrossProductTest::prepare_to_validation( int )
{
    cv::Vec<double, 3>  a, b, c;

    test_mat[INPUT][0].reshape(1, 1).copyTo(a);
    test_mat[INPUT][1].reshape(1, 1).copyTo(b);

    c.val[2] = a.val[0]*b.val[1] - a.val[1]*b.val[0];
    c.val[1] = -a.val[0]*b.val[2] + a.val[2]*b.val[0];
    c.val[0] = a.val[1]*b.val[2] - a.val[2]*b.val[1];

    Mat &ref = test_mat[REF_OUTPUT][0];
    Mat(c).reshape(ref.channels(), ref.rows).convertTo(ref, ref.type());
}


///////////////// gemm /////////////////////

class Core_GEMMTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_GEMMTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int tabc_flag;
    double alpha, beta;
};

Core_GEMMTest::Core_GEMMTest() : Core_MatrixTest( 5, 1, false, false, 2 )
{
    test_case_count = 100;
    max_log_array_size = 10;
    tabc_flag = 0;
    alpha = beta = 0;
}


void Core_GEMMTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    Size sizeA;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizeA = sizes[INPUT][0];
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sizeA;
    sizes[INPUT][2] = sizes[INPUT][3] = Size(1,1);
    types[INPUT][2] = types[INPUT][3] &= ~CV_MAT_CN_MASK;

    tabc_flag = cvtest::randInt(rng) & 7;

    switch( tabc_flag & (cv::GEMM_1_T|cv::GEMM_2_T) )
    {
        case 0:
            sizes[INPUT][1].height = sizes[INPUT][0].width;
            sizes[OUTPUT][0].height = sizes[INPUT][0].height;
            sizes[OUTPUT][0].width = sizes[INPUT][1].width;
            break;
        case cv::GEMM_2_T:
            sizes[INPUT][1].width = sizes[INPUT][0].width;
            sizes[OUTPUT][0].height = sizes[INPUT][0].height;
            sizes[OUTPUT][0].width = sizes[INPUT][1].height;
            break;
        case cv::GEMM_1_T:
            sizes[INPUT][1].height = sizes[INPUT][0].height;
            sizes[OUTPUT][0].height = sizes[INPUT][0].width;
            sizes[OUTPUT][0].width = sizes[INPUT][1].width;
            break;
        case cv::GEMM_1_T | cv::GEMM_2_T:
            sizes[INPUT][1].width = sizes[INPUT][0].height;
            sizes[OUTPUT][0].height = sizes[INPUT][0].width;
            sizes[OUTPUT][0].width = sizes[INPUT][1].height;
            break;
    }

    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];

    if( cvtest::randInt(rng) & 1 )
        sizes[INPUT][4] = Size(0,0);
    else if( !(tabc_flag & cv::GEMM_3_T) )
        sizes[INPUT][4] = sizes[OUTPUT][0];
    else
    {
        sizes[INPUT][4].width = sizes[OUTPUT][0].height;
        sizes[INPUT][4].height = sizes[OUTPUT][0].width;
    }
}


int Core_GEMMTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        test_mat[INPUT][2](Rect(0, 0, 1, 1)).convertTo(Mat(1, 1, CV_64F, &alpha), CV_64F);
        test_mat[INPUT][3](Rect(0, 0, 1, 1)).convertTo(Mat(1, 1, CV_64F, &beta), CV_64F);
    }
    return code;
}


void Core_GEMMTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = Scalar::all(-10.);
    high = Scalar::all(10.);
}


void Core_GEMMTest::run_func()
{
    /*printf("tabc_flags=At:%d,Bt:%d,Ct:%d; A(%d x %d), B(%d x %d), C(%d x %d)\n",
           (tabc_flag & GEMM_1_T) != 0, (tabc_flag & GEMM_2_T) != 0, (tabc_flag & GEMM_3_T) != 0,
           test_mat[INPUT][0].rows, test_mat[INPUT][0].cols,
           test_mat[INPUT][1].rows, test_mat[INPUT][1].cols,
           test_mat[INPUT][4].rows, test_mat[INPUT][4].cols);*/
    cv::gemm( test_mat[INPUT][0], test_mat[INPUT][1], alpha,
              test_mat[INPUT][4], beta, test_mat[OUTPUT][0], tabc_flag );
}


void Core_GEMMTest::prepare_to_validation( int )
{
    cvtest::gemm( test_mat[INPUT][0], test_mat[INPUT][1], alpha,
             !test_mat[INPUT][4].empty() ? test_mat[INPUT][4] : Mat(),
             beta, test_mat[REF_OUTPUT][0], tabc_flag );
}


///////////////// multransposed /////////////////////

class Core_MulTransposedTest : public Core_MatrixTest
{
public:
    Core_MulTransposedTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    bool order;
};


Core_MulTransposedTest::Core_MulTransposedTest() : Core_MatrixTest( 2, 1, false, false, 1 )
{
    test_case_count = 100;
    order = false;
    test_array[TEMP].push_back(NULL);
}


void Core_MulTransposedTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int bits = cvtest::randInt(rng);
    int src_type = cvtest::randInt(rng) % 5;
    int dst_type = cvtest::randInt(rng) % 2;

    src_type = src_type == 0 ? CV_8U : src_type == 1 ? CV_16U : src_type == 2 ? CV_16S :
    src_type == 3 ? CV_32F : CV_64F;
    dst_type = CV_32F;
    dst_type = MAX( dst_type, src_type );

    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( bits & 1 )
        sizes[INPUT][1] = Size(0,0);
    else
    {
        sizes[INPUT][1] = sizes[INPUT][0];
        if( bits & 2 )
            sizes[INPUT][1].height = 1;
        if( bits & 4 )
            sizes[INPUT][1].width = 1;
    }

    sizes[TEMP][0] = sizes[INPUT][0];
    types[INPUT][0] = src_type;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][1] = types[TEMP][0] = dst_type;

    order = (bits & 8) != 0;
    sizes[OUTPUT][0].width = sizes[OUTPUT][0].height = (order == false) ?
    sizes[INPUT][0].height : sizes[INPUT][0].width;
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


void Core_MulTransposedTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cv::Scalar::all(-10.);
    high = cv::Scalar::all(10.);
}


void Core_MulTransposedTest::run_func()
{
    cv::mulTransposed( test_mat[INPUT][0], test_mat[OUTPUT][0], order, test_mat[INPUT][1].empty() ? noArray() : test_mat[INPUT][1] );
}


void Core_MulTransposedTest::prepare_to_validation( int )
{
    const Mat& src = test_mat[INPUT][0];
    Mat delta = test_mat[INPUT][1];
    Mat& temp = test_mat[TEMP][0];
    if( !delta.empty() )
    {
        if( delta.rows < src.rows || delta.cols < src.cols )
        {
            cv::repeat( delta, src.rows/delta.rows, src.cols/delta.cols, temp);
            delta = temp;
        }
        cvtest::add( src, 1, delta, -1, Scalar::all(0), temp, temp.type());
    }
    else
        src.convertTo(temp, temp.type());

    cvtest::gemm( temp, temp, 1., Mat(), 0, test_mat[REF_OUTPUT][0], order == 0 ? GEMM_2_T : GEMM_1_T );
}


///////////////// Transform /////////////////////

class Core_TransformTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_TransformTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );

    double scale;
    bool diagMtx;
};


Core_TransformTest::Core_TransformTest() : Core_MatrixTest( 3, 1, true, false, 4 )
{
    scale = 1;
    diagMtx = false;
}


void Core_TransformTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int bits = cvtest::randInt(rng);
    int depth, dst_cn, mat_cols, mattype;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    mat_cols = CV_MAT_CN(types[INPUT][0]);
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    dst_cn = cvtest::randInt(rng) % 4 + 1;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, dst_cn);

    mattype = depth < CV_32S ? CV_32F : depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;

    scale = 1./((cvtest::randInt(rng)%4)*50+1);

    if( bits & 2 )
    {
        mat_cols += (bits & 4) != 0;
    }
    diagMtx = (bits & 16) != 0;

    sizes[INPUT][1] = Size(mat_cols,dst_cn);
}


int Core_TransformTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        Mat& m = test_mat[INPUT][1];
        cvtest::add(m, scale, m, 0, Scalar::all(0), m, m.type() );
        if(diagMtx)
        {
            Mat mask = Mat::eye(m.rows, m.cols, CV_8U)*255;
            mask = ~mask;
            m.setTo(Scalar::all(0), mask);
        }
    }
    return code;
}


double Core_TransformTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 1 : depth <= CV_32S ? 9 : Base::get_success_error_level( test_case_idx, i, j );
}

void Core_TransformTest::run_func()
{
    cv::transform( test_mat[INPUT][0], test_mat[OUTPUT][0], test_mat[INPUT][1]);
}


void Core_TransformTest::prepare_to_validation( int )
{
    Mat transmat = test_mat[INPUT][1];

    cvtest::transform( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], transmat, Mat() );
}

class Core_TransformLargeTest : public Core_TransformTest
{
public:
    typedef Core_MatrixTest Base;
protected:
    void get_test_array_types_and_sizes(int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types);
};

void Core_TransformLargeTest::get_test_array_types_and_sizes(int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types)
{
    RNG& rng = cv::theRNG();
    int bits = cvtest::randInt(rng);
    int depth, dst_cn, mat_cols, mattype;
    Base::get_test_array_types_and_sizes(test_case_idx, sizes, types);
    for (unsigned int j = 0; j < sizes.size(); j++)
    {
        for (unsigned int i = 0; i < sizes[j].size(); i++)
        {
            sizes[j][i].width *= 4;
        }
    }

    mat_cols = CV_MAT_CN(types[INPUT][0]);
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    dst_cn = cvtest::randInt(rng) % 4 + 1;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, dst_cn);

    mattype = depth < CV_32S ? CV_32F : depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;

    scale = 1. / ((cvtest::randInt(rng) % 4) * 50 + 1);

    if (bits & 2)
    {
        mat_cols += (bits & 4) != 0;
    }
    diagMtx = (bits & 16) != 0;

    sizes[INPUT][1] = Size(mat_cols, dst_cn);
}



///////////////// PerspectiveTransform /////////////////////

class Core_PerspectiveTransformTest : public Core_MatrixTest
{
public:
    Core_PerspectiveTransformTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_PerspectiveTransformTest::Core_PerspectiveTransformTest() : Core_MatrixTest( 2, 1, false, false, 2 )
{
}


void Core_PerspectiveTransformTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int bits = cvtest::randInt(rng);
    int depth, cn, mattype;
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    cn = CV_MAT_CN(types[INPUT][0]) + 1;
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);

    mattype = depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;
    sizes[INPUT][1] = Size(cn + 1, cn + 1);
}


double Core_PerspectiveTransformTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_32F ? 1e-4 : depth == CV_64F ? 1e-8 :
    Core_MatrixTest::get_success_error_level(test_case_idx, i, j);
}


void Core_PerspectiveTransformTest::run_func()
{
    perspectiveTransform( test_mat[INPUT][0], test_mat[OUTPUT][0], test_mat[INPUT][1] );
}


static void cvTsPerspectiveTransform( const Mat & a, Mat & b, const Mat & transmat )
{
    int i, j, cols;
    int cn, depth, mat_depth;
    double mat[16] = {0.0};

    cn = a.channels();
    depth = a.depth();
    mat_depth = transmat.depth();
    cols = transmat.cols;

    // prepare cn x (cn + 1) transform matrix
    if( mat_depth == CV_32F )
    {
        for( i = 0; i < transmat.rows; i++ )
            for( j = 0; j < cols; j++ )
                mat[i*cols + j] = transmat.at<float>(i, j);
    }
    else
    {
        CV_Assert( mat_depth == CV_64F );
        for( i = 0; i < transmat.rows; i++ )
            for( j = 0; j < cols; j++ )
                mat[i*cols + j] = transmat.at<double>(i, j);
    }

    // transform data
    cols = a.cols * cn;
    vector<double> buf(cols);

    for( i = 0; i < a.rows; i++ )
    {
        switch( depth )
        {
            case CV_32F:
                for( j = 0; j < cols; j++ )
                    buf[j] = a.at<float>(i, j);
                break;
            case CV_64F:
                for( j = 0; j < cols; j++ )
                    buf[j] = a.at<double>(i, j);
                break;
            default:
                CV_Assert(0);
        }

        switch( cn )
        {
            case 2:
                for( j = 0; j < cols; j += 2 )
                {
                    double t0 = buf[j]*mat[0] + buf[j+1]*mat[1] + mat[2];
                    double t1 = buf[j]*mat[3] + buf[j+1]*mat[4] + mat[5];
                    double w = buf[j]*mat[6] + buf[j+1]*mat[7] + mat[8];
                    w = w ? 1./w : 0;
                    buf[j] = t0*w;
                    buf[j+1] = t1*w;
                }
                break;
            case 3:
                for( j = 0; j < cols; j += 3 )
                {
                    double t0 = buf[j]*mat[0] + buf[j+1]*mat[1] + buf[j+2]*mat[2] + mat[3];
                    double t1 = buf[j]*mat[4] + buf[j+1]*mat[5] + buf[j+2]*mat[6] + mat[7];
                    double t2 = buf[j]*mat[8] + buf[j+1]*mat[9] + buf[j+2]*mat[10] + mat[11];
                    double w = buf[j]*mat[12] + buf[j+1]*mat[13] + buf[j+2]*mat[14] + mat[15];
                    w = w ? 1./w : 0;
                    buf[j] = t0*w;
                    buf[j+1] = t1*w;
                    buf[j+2] = t2*w;
                }
                break;
            default:
                CV_Assert(0);
        }

        switch( depth )
        {
            case CV_32F:
                for( j = 0; j < cols; j++ )
                    b.at<float>(i, j) = (float)buf[j];
                break;
            case CV_64F:
                for( j = 0; j < cols; j++ )
                    b.at<double>(i, j) = buf[j];
                break;
            default:
                CV_Assert(0);
        }
    }
}


void Core_PerspectiveTransformTest::prepare_to_validation( int )
{
    cvTsPerspectiveTransform( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[INPUT][1] );
}

///////////////// Mahalanobis /////////////////////

class Core_MahalanobisTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_MahalanobisTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_MahalanobisTest::Core_MahalanobisTest() : Core_MatrixTest( 3, 1, false, true, 1 )
{
    test_case_count = 100;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_MahalanobisTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( cvtest::randInt(rng) & 1 )
        sizes[INPUT][0].width = sizes[INPUT][1].width = 1;
    else
        sizes[INPUT][0].height = sizes[INPUT][1].height = 1;

    sizes[TEMP][0] = sizes[TEMP][1] = sizes[INPUT][0];
    sizes[INPUT][2].width = sizes[INPUT][2].height = sizes[INPUT][0].width + sizes[INPUT][0].height - 1;
    sizes[TEMP][2] = sizes[INPUT][2];
    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[INPUT][0];
}

int Core_MahalanobisTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        // make sure that the inverted "covariation" matrix is symmetrix and positively defined.
        cvtest::gemm( test_mat[INPUT][2], test_mat[INPUT][2], 1., Mat(), 0., test_mat[TEMP][2], GEMM_2_T );
        cvtest::copy( test_mat[TEMP][2], test_mat[INPUT][2] );
    }

    return code;
}


void Core_MahalanobisTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) =
        cv::Mahalanobis(test_mat[INPUT][0], test_mat[INPUT][1], test_mat[INPUT][2]);
}

void Core_MahalanobisTest::prepare_to_validation( int )
{
    cvtest::add( test_mat[INPUT][0], 1., test_mat[INPUT][1], -1.,
                Scalar::all(0), test_mat[TEMP][0], test_mat[TEMP][0].type() );
    if( test_mat[INPUT][0].rows == 1 )
        cvtest::gemm( test_mat[TEMP][0], test_mat[INPUT][2], 1.,
                 Mat(), 0., test_mat[TEMP][1], 0 );
    else
        cvtest::gemm( test_mat[INPUT][2], test_mat[TEMP][0], 1.,
                 Mat(), 0., test_mat[TEMP][1], 0 );

    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = cv::Scalar(sqrt(cvtest::crossCorr(test_mat[TEMP][0], test_mat[TEMP][1])));
}


static void cvTsFloodWithZeros( Mat& mat, RNG& rng )
{
    int k, total = mat.rows*mat.cols, type = mat.type();
    int zero_total = cvtest::randInt(rng) % total;
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    for( k = 0; k < zero_total; k++ )
    {
        int i = cvtest::randInt(rng) % mat.rows;
        int j = cvtest::randInt(rng) % mat.cols;

        if( type == CV_32FC1 )
            mat.at<float>(i,j) = 0.f;
        else
            mat.at<double>(i,j) = 0.;
    }
}


///////////////// determinant /////////////////////

class Core_DetTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_DetTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_DetTest::Core_DetTest() : Core_MatrixTest( 1, 1, false, true, 1 )
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push_back(NULL);
}


void Core_DetTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    sizes[INPUT][0].width = sizes[INPUT][0].height;
    sizes[TEMP][0] = sizes[INPUT][0];
    types[TEMP][0] = CV_64FC1;
}


void Core_DetTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cv::Scalar::all(-2.);
    high = cv::Scalar::all(2.);
}


double Core_DetTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return test_mat[INPUT][0].depth() == CV_32F ? 1e-2 : 1e-5;
}


int Core_DetTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
        cvTsFloodWithZeros( test_mat[INPUT][0], cv::theRNG() );

    return code;
}


void Core_DetTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) = cv::determinant(test_mat[INPUT][0]);
}


// LU method that chooses the optimal in a column pivot element
static double cvTsLU( cv::Mat a )
{
    CV_Assert(a.type() == CV_64FC1);
    int i, j, k, N = a.rows, N1 = a.cols, Nm = MIN(N, N1), step = (int)(a.step/sizeof(double));
    double *a0 = a.ptr<double>();
    double t, det = 1.;

    for( i = 0; i < Nm; i++ )
    {
        double max_val = fabs(a0[i*step + i]);
        double *a1, *a2;
        k = i;

        for( j = i+1; j < N; j++ )
        {
            t = fabs(a0[j*step + i]);
            if( max_val < t )
            {
                max_val = t;
                k = j;
            }
        }

        if( k != i )
        {
            for( j = i; j < N1; j++ )
                CV_SWAP( a0[i*step + j], a0[k*step + j], t );
            det = -det;
        }

        if( max_val == 0 )
        {
            return 0.;
        }

        a1 = a0 + i*step;
        a2 = a1 + step;

        for( j = i+1; j < N; j++, a2 += step )
        {
            t = a2[i]/a1[i];
            for( k = i+1; k < N1; k++ )
                a2[k] -= t*a1[k];
        }

        det *= a1[i];
    }

    return det;
}


void Core_DetTest::prepare_to_validation( int )
{
    test_mat[INPUT][0].convertTo(test_mat[TEMP][0], test_mat[TEMP][0].type());
    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = cv::Scalar(cvTsLU(test_mat[TEMP][0]));
}


///////////////// invert /////////////////////

class Core_InvertTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_InvertTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int method, rank;
    double result;
};


Core_InvertTest::Core_InvertTest()
: Core_MatrixTest( 1, 1, false, false, 1 ), method(0), rank(0), result(0.)
{
    test_case_count = 100;
    max_log_array_size = 6; // errors with larger arrays (>100x100)
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_InvertTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int bits = cvtest::randInt(rng);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );

    if( (bits & 3) == 0 )
    {
        method = cv::DECOMP_SVD;
        if( bits & 4 )
        {
            sizes[INPUT][0] = Size(min_size, min_size);
            if( bits & 16 )
                method = cv::DECOMP_CHOLESKY;
        }
    }
    else
    {
        method = cv::DECOMP_LU;
        sizes[INPUT][0] = Size(min_size, min_size);
    }

    sizes[TEMP][0].width = sizes[INPUT][0].height;
    sizes[TEMP][0].height = sizes[INPUT][0].width;
    sizes[TEMP][1] = sizes[INPUT][0];
    types[TEMP][0] = types[INPUT][0];
    types[TEMP][1] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size(min_size, min_size);
}


double Core_InvertTest::get_success_error_level( int /*test_case_idx*/, int, int )
{
    return test_mat[OUTPUT][0].depth() == CV_32F ? 1e-2 : 1e-6;
}

int Core_InvertTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        cvTsFloodWithZeros( test_mat[INPUT][0], cv::theRNG() );

        if( method == cv::DECOMP_CHOLESKY )
        {
            cvtest::gemm( test_mat[INPUT][0], test_mat[INPUT][0], 1.,
                     Mat(), 0., test_mat[TEMP][0], cv::GEMM_2_T );
            cvtest::copy( test_mat[TEMP][0], test_mat[INPUT][0] );
        }
    }

    return code;
}



void Core_InvertTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cv::Scalar::all(-1.);
    high = cv::Scalar::all(1.);
}

void Core_InvertTest::run_func()
{
    result = cv::invert(test_mat[INPUT][0], test_mat[TEMP][0], method);
}


static double cvTsSVDet( const Mat & mat, double* ratio )
{
    int i, nm = MIN( mat.rows, mat.cols );
    Mat w( nm, 1, mat.type() );
    double det = 1.;

    cv::SVD::compute( mat, w );

    if( mat.type() == CV_32FC1 )
    {
        for( i = 0; i < nm; i++ )
            det *= w.at<float>(i);
        *ratio = w.at<float>(nm-1) < FLT_EPSILON ? 0 : w.at<float>(nm-1)/w.at<float>(0);
    }
    else
    {
        for( i = 0; i < nm; i++ )
            det *= w.at<double>(i);
        *ratio = w.at<double>(nm-1) < FLT_EPSILON ? 0 : w.at<double>(nm-1)/w.at<double>(0);
    }

    return det;
}

void Core_InvertTest::prepare_to_validation( int )
{
    Mat& input = test_mat[INPUT][0];
    Mat& temp0 = test_mat[TEMP][0];
    Mat& temp1 = test_mat[TEMP][1];
    Mat& dst0 = test_mat[REF_OUTPUT][0];
    Mat& dst = test_mat[OUTPUT][0];
    double ratio = 0, det = cvTsSVDet( input, &ratio );
    double threshold = (input.depth() == CV_32F ? FLT_EPSILON : DBL_EPSILON)*1000;

    cvtest::convert( input, temp1, temp1.type() );

    if( det < threshold ||
       ((method == cv::DECOMP_LU || method == cv::DECOMP_CHOLESKY) && (result == 0 || ratio < threshold)) ||
       ((method == cv::DECOMP_SVD || method == cv::DECOMP_EIG) && result < threshold) )
    {
        dst = Scalar::all(0);
        dst0 = Scalar::all(0);
        return;
    }

    if( input.rows >= input.cols )
        cvtest::gemm( temp0, input, 1., Mat(), 0., dst, 0 );
    else
        cvtest::gemm( input, temp0, 1., Mat(), 0., dst, 0 );

    cv::setIdentity( dst0, Scalar::all(1) );
}


///////////////// solve /////////////////////

class Core_SolveTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_SolveTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int method, rank;
    double result;
};


Core_SolveTest::Core_SolveTest() : Core_MatrixTest( 2, 1, false, false, 1 ), method(0), rank(0), result(0.)
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_SolveTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = cv::theRNG();
    int bits = cvtest::randInt(rng);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    cv::Size in_sz = cv::Size(sizes[INPUT][0]);
    if( in_sz.width > in_sz.height )
        in_sz = cv::Size(in_sz.height, in_sz.width);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = in_sz;
    int min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );

    if( (bits & 3) == 0 )
    {
        method = cv::DECOMP_SVD;
        if( bits & 4 )
        {
            sizes[INPUT][0] = Size(min_size, min_size);
            /*if( bits & 8 )
             method = cv::DECOMP_EIG;*/
        }
    }
    else
    {
        method = cv::DECOMP_LU;
        sizes[INPUT][0] = Size(min_size, min_size);
    }

    sizes[INPUT][1].height = sizes[INPUT][0].height;
    sizes[TEMP][0].width = sizes[INPUT][1].width;
    sizes[TEMP][0].height = sizes[INPUT][0].width;
    sizes[TEMP][1] = sizes[INPUT][0];
    types[TEMP][0] = types[INPUT][0];
    types[TEMP][1] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size(sizes[INPUT][1].width, min_size);
}


int Core_SolveTest::prepare_test_case( int test_case_idx )
{
    return Core_MatrixTest::prepare_test_case( test_case_idx );
}


void Core_SolveTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cv::Scalar::all(-1.);
    high = cv::Scalar::all(1.);
}


double Core_SolveTest::get_success_error_level( int /*test_case_idx*/, int, int )
{
    return test_mat[OUTPUT][0].depth() == CV_32F ? 5e-2 : 1e-8;
}


void Core_SolveTest::run_func()
{
    result = cv::solve(test_mat[INPUT][0], test_mat[INPUT][1], test_mat[TEMP][0], method);
}

void Core_SolveTest::prepare_to_validation( int )
{
    //int rank = test_mat[REF_OUTPUT][0].rows;
    Mat& input = test_mat[INPUT][0];
    Mat& dst = test_mat[OUTPUT][0];
    Mat& dst0 = test_mat[REF_OUTPUT][0];

    if( method == cv::DECOMP_LU )
    {
        if( result == 0 )
        {
            Mat& temp1 = test_mat[TEMP][1];
            cvtest::convert(input, temp1, temp1.type());
            dst = Scalar::all(0);
            double det = cvTsLU( temp1 );
            dst0 = Scalar::all(det != 0);
            return;
        }

        double threshold = (input.type() == CV_32F ? FLT_EPSILON : DBL_EPSILON)*1000;
        double ratio = 0, det = cvTsSVDet( input, &ratio );
        if( det < threshold || ratio < threshold )
        {
            dst = Scalar::all(0);
            dst0 = Scalar::all(0);
            return;
        }
    }

    Mat* pdst = input.rows <= input.cols ? &test_mat[OUTPUT][0] : &test_mat[INPUT][1];

    cvtest::gemm( input, test_mat[TEMP][0], 1., test_mat[INPUT][1], -1., *pdst, 0 );
    if( pdst != &dst )
        cvtest::gemm( input, *pdst, 1., Mat(), 0., dst, cv::GEMM_1_T );
    dst0 = Scalar::all(0);
}



typedef std::complex<double> complex_type;

struct pred_complex
{
    bool operator() (const complex_type& lhs, const complex_type& rhs) const
    {
        return fabs(lhs.real() - rhs.real()) > fabs(rhs.real())*FLT_EPSILON ? lhs.real() < rhs.real() : lhs.imag() < rhs.imag();
    }
};

struct pred_double
{
    bool operator() (const double& lhs, const double& rhs) const
    {
        return lhs < rhs;
    }
};

class Core_SolvePolyTest : public cvtest::BaseTest
{
public:
    Core_SolvePolyTest();
    ~Core_SolvePolyTest();
protected:
    virtual void run( int start_from );
};

Core_SolvePolyTest::Core_SolvePolyTest() {}

Core_SolvePolyTest::~Core_SolvePolyTest() {}

void Core_SolvePolyTest::run( int )
{
    RNG& rng = cv::theRNG();
    double range = 50;
    double err_eps = 1e-4;

    for (int idx = 0, max_idx = 1000, progress = 0; idx < max_idx; ++idx)
    {
        progress = update_progress(progress, idx-1, max_idx, 0);
        int n = cvtest::randInt(rng) % 13 + 1;
        std::vector<complex_type> r(n), ar(n), c(n + 1, 0);
        std::vector<double> a(n + 1), u(n * 2), ar1(n), ar2(n);

        int rr_odds = 3; // odds that we get a real root
        for (int j = 0; j < n;)
        {
            if (cvtest::randInt(rng) % rr_odds == 0 || j == n - 1)
                r[j++] = cvtest::randReal(rng) * range;
            else
            {
                r[j] = complex_type(cvtest::randReal(rng) * range,
                                    cvtest::randReal(rng) * range + 1);
                r[j + 1] = std::conj(r[j]);
                j += 2;
            }
        }

        for (int j = 0, k = 1 << n, jj, kk; j < k; ++j)
        {
            int p = 0;
            complex_type v(1);
            for (jj = 0, kk = 1; jj < n && !(j & kk); ++jj, ++p, kk <<= 1)
                ;
            for (; jj < n; ++jj, kk <<= 1)
            {
                if (j & kk)
                    v *= -r[jj];
                else
                    ++p;
            }
            c[p] += v;
        }

        bool pass = false;
        double div = 0, s = 0;
        int cubic_case = idx & 1;
        for (int maxiter = 100; !pass && maxiter < 10000; maxiter *= 2, cubic_case = (cubic_case + 1) % 2)
        {
            for (int j = 0; j < n + 1; ++j)
                a[j] = c[j].real();

            Mat amat(n + 1, 1, CV_64FC1, &a[0]), umat(n, 1, CV_64FC2, &u[0]);
            cv::solvePoly(amat, umat, maxiter);

            for (int j = 0; j < n; ++j)
                ar[j] = complex_type(u[j * 2], u[j * 2 + 1]);

            std::sort(r.begin(), r.end(), pred_complex());
            std::sort(ar.begin(), ar.end(), pred_complex());

            pass = true;
            if( n == 3 )
            {
                ar2.resize(n);
                cv::Mat _umat2(3, 1, CV_64F, &ar2[0]), umat2 = _umat2;
                cv::flip(amat, amat, 0);
                int nr2;
                if( cubic_case == 0 )
                    nr2 = cv::solveCubic(amat,umat2);
                else
                    nr2 = cv::solveCubic(cv::Mat_<float>(amat), umat2);
                cv::flip(amat, amat, 0);
                if(nr2 > 0)
                    std::sort(ar2.begin(), ar2.begin()+nr2, pred_double());
                ar2.resize(nr2);

                int nr1 = 0;
                for(int j = 0; j < n; j++)
                    if( fabs(r[j].imag()) < DBL_EPSILON )
                        ar1[nr1++] = r[j].real();

                pass = pass && nr1 == nr2;
                if( nr2 > 0 )
                {
                    div = s = 0;
                    for(int j = 0; j < nr1; j++)
                    {
                        s += fabs(ar1[j]);
                        div += fabs(ar1[j] - ar2[j]);
                    }
                    div /= s;
                    pass = pass && div < err_eps;
                }
            }

            div = s = 0;
            for (int j = 0; j < n; ++j)
            {
                s += fabs(r[j].real()) + fabs(r[j].imag());
                div += sqrt(pow(r[j].real() - ar[j].real(), 2) + pow(r[j].imag() - ar[j].imag(), 2));
            }
            div /= s;
            pass = pass && div < err_eps;
        }

        //test x^3 = 0
        cv::Mat coeffs_5623(4, 1, CV_64FC1);
        cv::Mat r_5623(3, 1, CV_64FC2);
        coeffs_5623.at<double>(0) = 1;
        coeffs_5623.at<double>(1) = 0;
        coeffs_5623.at<double>(2) = 0;
        coeffs_5623.at<double>(3) = 0;
        double prec_5623 = cv::solveCubic(coeffs_5623, r_5623);
        pass = pass && r_5623.at<double>(0) == 0 && r_5623.at<double>(1) == 0 && r_5623.at<double>(2) == 0;
        pass = pass && prec_5623 == 1;

        if (!pass)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            ts->printf( cvtest::TS::LOG, "too big diff = %g\n", div );

            for (size_t j=0;j<ar2.size();++j)
                ts->printf( cvtest::TS::LOG, "ar2[%d]=%g\n", j, ar2[j]);
            ts->printf(cvtest::TS::LOG, "\n");

            for (size_t j=0;j<r.size();++j)
                ts->printf( cvtest::TS::LOG, "r[%d]=(%g, %g)\n", j, r[j].real(), r[j].imag());
            ts->printf( cvtest::TS::LOG, "\n" );
            for (size_t j=0;j<ar.size();++j)
                ts->printf( cvtest::TS::LOG, "ar[%d]=(%g, %g)\n", j, ar[j].real(), ar[j].imag());
            break;
        }
    }
}

template<typename T>
static void checkRoot(Mat& r, T re, T im)
{
    for (int i = 0; i < r.cols*r.rows; i++)
    {
        Vec<T, 2> v = *(Vec<T, 2>*)r.ptr(i);
        if (fabs(re - v[0]) < 1e-6 && fabs(im - v[1]) < 1e-6)
        {
            v[0] = std::numeric_limits<T>::quiet_NaN();
            v[1] = std::numeric_limits<T>::quiet_NaN();
            return;
        }
    }
    GTEST_NONFATAL_FAILURE_("Can't find root") << "(" << re << ", " << im << ")";
}
TEST(Core_SolvePoly, regression_5599)
{
    // x^4 - x^2 = 0, roots: 1, -1, 0, 0
    cv::Mat coefs = (cv::Mat_<float>(1,5) << 0, 0, -1, 0, 1 );
    {
        cv::Mat r;
        double prec;
        prec = cv::solvePoly(coefs, r);
        EXPECT_LE(prec, 1e-6);
        EXPECT_EQ(4u, r.total());
        //std::cout << "Preciseness = " << prec << std::endl;
        //std::cout << "roots:\n" << r << "\n" << std::endl;
        ASSERT_EQ(CV_32FC2, r.type());
        checkRoot<float>(r, 1, 0);
        checkRoot<float>(r, -1, 0);
        checkRoot<float>(r, 0, 0);
        checkRoot<float>(r, 0, 0);
    }
    // x^2 - 2x + 1 = 0,  roots: 1, 1
    coefs = (cv::Mat_<float>(1,3) << 1, -2, 1 );
    {
        cv::Mat r;
        double prec;
        prec = cv::solvePoly(coefs, r);
        EXPECT_LE(prec, 1e-6);
        EXPECT_EQ(2u, r.total());
        //std::cout << "Preciseness = " << prec << std::endl;
        //std::cout << "roots:\n" << r << "\n" << std::endl;
        ASSERT_EQ(CV_32FC2, r.type());
        checkRoot<float>(r, 1, 0);
        checkRoot<float>(r, 1, 0);
    }
}

class Core_PhaseTest : public cvtest::BaseTest
{
    int t;
public:
    Core_PhaseTest(int t_) : t(t_) {}
    ~Core_PhaseTest() {}
protected:
    virtual void run(int)
    {
        const float maxAngleDiff = 0.5; //in degrees
        const int axisCount = 8;
        const int dim = theRNG().uniform(1,10);
        const float scale = theRNG().uniform(1.f, 100.f);
        Mat x(axisCount + 1, dim, t),
            y(axisCount + 1, dim, t);
        Mat anglesInDegrees(axisCount + 1, dim, t);

        // fill the data
        x.row(0).setTo(Scalar(0));
        y.row(0).setTo(Scalar(0));
        anglesInDegrees.row(0).setTo(Scalar(0));

        x.row(1).setTo(Scalar(scale));
        y.row(1).setTo(Scalar(0));
        anglesInDegrees.row(1).setTo(Scalar(0));

        x.row(2).setTo(Scalar(scale));
        y.row(2).setTo(Scalar(scale));
        anglesInDegrees.row(2).setTo(Scalar(45));

        x.row(3).setTo(Scalar(0));
        y.row(3).setTo(Scalar(scale));
        anglesInDegrees.row(3).setTo(Scalar(90));

        x.row(4).setTo(Scalar(-scale));
        y.row(4).setTo(Scalar(scale));
        anglesInDegrees.row(4).setTo(Scalar(135));

        x.row(5).setTo(Scalar(-scale));
        y.row(5).setTo(Scalar(0));
        anglesInDegrees.row(5).setTo(Scalar(180));

        x.row(6).setTo(Scalar(-scale));
        y.row(6).setTo(Scalar(-scale));
        anglesInDegrees.row(6).setTo(Scalar(225));

        x.row(7).setTo(Scalar(0));
        y.row(7).setTo(Scalar(-scale));
        anglesInDegrees.row(7).setTo(Scalar(270));

        x.row(8).setTo(Scalar(scale));
        y.row(8).setTo(Scalar(-scale));
        anglesInDegrees.row(8).setTo(Scalar(315));

        Mat resInRad, resInDeg;
        phase(x, y, resInRad, false);
        phase(x, y, resInDeg, true);

        CV_Assert(resInRad.size() == x.size());
        CV_Assert(resInRad.type() == x.type());

        CV_Assert(resInDeg.size() == x.size());
        CV_Assert(resInDeg.type() == x.type());

        // check the result
        int outOfRangeCount = countNonZero((resInDeg > 360) | (resInDeg < 0));
        if(outOfRangeCount > 0)
        {
            ts->printf(cvtest::TS::LOG, "There are result angles that are out of range [0, 360] (part of them is %f)\n",
                       static_cast<float>(outOfRangeCount)/resInDeg.total());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }

        Mat diff = abs(anglesInDegrees - resInDeg);
        size_t errDegCount = diff.total() - countNonZero((diff < maxAngleDiff) | ((360 - diff) < maxAngleDiff));
        if(errDegCount > 0)
        {
            ts->printf(cvtest::TS::LOG, "There are incorrect result angles (in degrees) (part of them is %f)\n",
                       static_cast<float>(errDegCount)/resInDeg.total());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }

        Mat convertedRes = resInRad * 180. / CV_PI;
        double normDiff = cvtest::norm(convertedRes - resInDeg, NORM_INF);
        if(normDiff > FLT_EPSILON * 180.)
        {
            ts->printf(cvtest::TS::LOG, "There are incorrect result angles (in radians)\n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }

        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Core_CheckRange_Empty, accuracy)
{
    cv::Mat m;
    ASSERT_TRUE( cv::checkRange(m) );
}

TEST(Core_CheckRange_INT_MAX, accuracy)
{
    cv::Mat m(3, 3, CV_32SC1, cv::Scalar(INT_MAX));
    ASSERT_FALSE( cv::checkRange(m, true, 0, 0, INT_MAX) );
    ASSERT_TRUE( cv::checkRange(m) );
}

TEST(Core_CheckRange_INT_MAX1, accuracy)
{
    cv::Mat m(3, 3, CV_32SC1, cv::Scalar(INT_MAX));
    ASSERT_TRUE( cv::checkRange(m, true, 0, 0, (float)((double)INT_MAX+1.0f)) );
    ASSERT_TRUE( cv::checkRange(m) );
}

template <typename T> class Core_CheckRange : public testing::Test {};

TYPED_TEST_CASE_P(Core_CheckRange);

TYPED_TEST_P(Core_CheckRange, Negative)
{
    double min_bound = 4.5;
    double max_bound = 16.0;

    TypeParam data[] = {5, 10, 15, 10, 10, 2, 8, 12, 14};
    cv::Mat src = cv::Mat(3,3, cv::DataDepth<TypeParam>::value, data);

    cv::Point bad_pt(0, 0);

    ASSERT_FALSE(checkRange(src, true, &bad_pt, min_bound, max_bound));
    ASSERT_EQ(bad_pt.x, 2);
    ASSERT_EQ(bad_pt.y, 1);
}

TYPED_TEST_P(Core_CheckRange, Negative3CN)
{
    double min_bound = 4.5;
    double max_bound = 16.0;

    TypeParam data[] = { 5,  6,  7,   10, 11, 12,   13, 14, 15,
                        10, 11, 12,   10, 11, 12,    2,  5,  6,
                         8,  8,  8,   12, 12, 12,   14, 14, 14};
    cv::Mat src = cv::Mat(3,3, CV_MAKETYPE(cv::DataDepth<TypeParam>::value, 3), data);

    cv::Point bad_pt(0, 0);

    ASSERT_FALSE(checkRange(src, true, &bad_pt, min_bound, max_bound));
    ASSERT_EQ(bad_pt.x, 2);
    ASSERT_EQ(bad_pt.y, 1);
}

TYPED_TEST_P(Core_CheckRange, Positive)
{
    double min_bound = -1;
    double max_bound = 16.0;

    TypeParam data[] = {5, 10, 15, 4, 10, 2, 8, 12, 14};
    cv::Mat src = cv::Mat(3,3, cv::DataDepth<TypeParam>::value, data);

    cv::Point bad_pt(0, 0);

    ASSERT_TRUE(checkRange(src, true, &bad_pt, min_bound, max_bound));
    ASSERT_EQ(bad_pt.x, 0);
    ASSERT_EQ(bad_pt.y, 0);
}

TYPED_TEST_P(Core_CheckRange, Bounds)
{
    double min_bound = 24.5;
    double max_bound = 1.0;

    TypeParam data[] = {5, 10, 15, 4, 10, 2, 8, 12, 14};
    cv::Mat src = cv::Mat(3,3, cv::DataDepth<TypeParam>::value, data);

    cv::Point bad_pt(0, 0);

    ASSERT_FALSE(checkRange(src, true, &bad_pt, min_bound, max_bound));
    ASSERT_EQ(bad_pt.x, 0);
    ASSERT_EQ(bad_pt.y, 0);
}

TYPED_TEST_P(Core_CheckRange, Zero)
{
    double min_bound = 0.0;
    double max_bound = 0.1;

    cv::Mat src1 = cv::Mat::zeros(3, 3, cv::DataDepth<TypeParam>::value);

    int sizes[] = {5, 6, 7};
    cv::Mat src2 = cv::Mat::zeros(3, sizes, cv::DataDepth<TypeParam>::value);

    ASSERT_TRUE( checkRange(src1, true, NULL, min_bound, max_bound) );
    ASSERT_TRUE( checkRange(src2, true, NULL, min_bound, max_bound) );
}

TYPED_TEST_P(Core_CheckRange, One)
{
    double min_bound = 1.0;
    double max_bound = 1.1;

    cv::Mat src1 = cv::Mat::ones(3, 3, cv::DataDepth<TypeParam>::value);

    int sizes[] = {5, 6, 7};
    cv::Mat src2 = cv::Mat::ones(3, sizes, cv::DataDepth<TypeParam>::value);

    ASSERT_TRUE( checkRange(src1, true, NULL, min_bound, max_bound) );
    ASSERT_TRUE( checkRange(src2, true, NULL, min_bound, max_bound) );
}

TEST(Core_CheckRange, NaN)
{
    float data[] = { 5,  6,  7,   10, 11, 12,   13, 14, 15,
                    10, 11, 12,   10, 11, 12,   5,  5,  std::numeric_limits<float>::quiet_NaN(),
                     8,  8,  8,   12, 12, 12,   14, 14, 14};
    cv::Mat src = cv::Mat(3,3, CV_32FC3, data);

    cv::Point bad_pt(0, 0);

    ASSERT_FALSE(checkRange(src, true, &bad_pt));
    ASSERT_EQ(bad_pt.x, 2);
    ASSERT_EQ(bad_pt.y, 1);
}

TEST(Core_CheckRange, Inf)
{
    float data[] = { 5,  6,  7,   10, 11, 12,   13, 14, 15,
                    10, 11, 12,   10, 11, 12,   5,  5,  std::numeric_limits<float>::infinity(),
                     8,  8,  8,   12, 12, 12,   14, 14, 14};
    cv::Mat src = cv::Mat(3,3, CV_32FC3, data);

    cv::Point bad_pt(0, 0);

    ASSERT_FALSE(checkRange(src, true, &bad_pt));
    ASSERT_EQ(bad_pt.x, 2);
    ASSERT_EQ(bad_pt.y, 1);
}

TEST(Core_CheckRange, Inf_Minus)
{
    float data[] = { 5,  6,  7,   10, 11, 12,   13, 14, 15,
                    10, 11, 12,   10, 11, 12,   5,  5,  -std::numeric_limits<float>::infinity(),
                     8,  8,  8,   12, 12, 12,   14, 14, 14};
    cv::Mat src = cv::Mat(3,3, CV_32FC3, data);

    cv::Point bad_pt(0, 0);

    ASSERT_FALSE(checkRange(src, true, &bad_pt));
    ASSERT_EQ(bad_pt.x, 2);
    ASSERT_EQ(bad_pt.y, 1);
}

REGISTER_TYPED_TEST_CASE_P(Core_CheckRange, Negative, Negative3CN, Positive, Bounds, Zero, One);

typedef ::testing::Types<signed char,unsigned char, signed short, unsigned short, signed int> mat_data_types;
INSTANTIATE_TYPED_TEST_CASE_P(Negative_Test, Core_CheckRange, mat_data_types);

TEST(Core_Invert, small)
{
    cv::Mat a = (cv::Mat_<float>(3,3) << 2.42104644730331, 1.81444796521479, -3.98072565304758, 0, 7.08389214348967e-3, 5.55326770986007e-3, 0,0, 7.44556154284261e-3);
    //cv::randu(a, -1, 1);

    cv::Mat b = a.t()*a;
    cv::Mat c, i = Mat_<float>::eye(3, 3);
    cv::invert(b, c, cv::DECOMP_LU); //std::cout << b*c << std::endl;
    ASSERT_LT( cvtest::norm(b*c, i, NORM_INF), 0.1 );
    cv::invert(b, c, cv::DECOMP_SVD); //std::cout << b*c << std::endl;
    ASSERT_LT( cvtest::norm(b*c, i, NORM_INF), 0.1 );
    cv::invert(b, c, cv::DECOMP_CHOLESKY); //std::cout << b*c << std::endl;
    ASSERT_LT( cvtest::norm(b*c, i, NORM_INF), 0.1 );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Core_CrossProduct, accuracy) { Core_CrossProductTest test; test.safe_run(); }
TEST(Core_Determinant, accuracy) { Core_DetTest test; test.safe_run(); }
TEST(Core_DotProduct, accuracy) { Core_DotProductTest test; test.safe_run(); }
TEST(Core_GEMM, accuracy) { Core_GEMMTest test; test.safe_run(); }
TEST(Core_Invert, accuracy) { Core_InvertTest test; test.safe_run(); }
TEST(Core_Mahalanobis, accuracy) { Core_MahalanobisTest test; test.safe_run(); }
TEST(Core_MulTransposed, accuracy) { Core_MulTransposedTest test; test.safe_run(); }
TEST(Core_Transform, accuracy) { Core_TransformTest test; test.safe_run(); }
TEST(Core_TransformLarge, accuracy) { Core_TransformLargeTest test; test.safe_run(); }
TEST(Core_PerspectiveTransform, accuracy) { Core_PerspectiveTransformTest test; test.safe_run(); }
TEST(Core_Pow, accuracy) { Core_PowTest test; test.safe_run(); }
TEST(Core_SolveLinearSystem, accuracy) { Core_SolveTest test; test.safe_run(); }
TEST(Core_Trace, accuracy) { Core_TraceTest test; test.safe_run(); }
TEST(Core_SolvePoly, accuracy) { Core_SolvePolyTest test; test.safe_run(); }
TEST(Core_Phase, accuracy32f) { Core_PhaseTest test(CV_32FC1); test.safe_run(); }
TEST(Core_Phase, accuracy64f) { Core_PhaseTest test(CV_64FC1); test.safe_run(); }

TEST(Core_SVD, flt)
{
    float a[] = {
    1.23377746e+011f, -7.05490125e+010f, -4.18380882e+010f, -11693456.f,
    -39091328.f, 77492224.f, -7.05490125e+010f, 2.36211143e+011f,
    -3.51093473e+010f, 70773408.f, -4.83386156e+005f, -129560368.f,
    -4.18380882e+010f, -3.51093473e+010f, 9.25311222e+010f, -49052424.f,
    43922752.f, 12176842.f, -11693456.f, 70773408.f, -49052424.f, 8.40836094e+004f,
    5.17475293e+003f, -1.16122949e+004f, -39091328.f, -4.83386156e+005f,
    43922752.f, 5.17475293e+003f, 5.16047969e+004f, 5.68887842e+003f, 77492224.f,
    -129560368.f, 12176842.f, -1.16122949e+004f, 5.68887842e+003f,
    1.28060578e+005f
    };

    float b[] = {
    283751232.f, 2.61604198e+009f, -745033216.f, 2.31125625e+005f,
    -4.52429188e+005f, -1.37596525e+006f
    };

    Mat A(6, 6, CV_32F, a);
    Mat B(6, 1, CV_32F, b);
    Mat X, B1;
    solve(A, B, X, DECOMP_SVD);
    B1 = A*X;
    EXPECT_LE(cvtest::norm(B1, B, NORM_L2 + NORM_RELATIVE), FLT_EPSILON*10);
}


// TODO: eigenvv, invsqrt, cbrt, fastarctan, (round, floor, ceil(?)),

enum
{
    MAT_N_DIM_C1,
    MAT_N_1_CDIM,
    MAT_1_N_CDIM,
    MAT_N_DIM_C1_NONCONT,
    MAT_N_1_CDIM_NONCONT,
    VECTOR
};

class CV_KMeansSingularTest : public cvtest::BaseTest
{
public:
    CV_KMeansSingularTest() {}
    ~CV_KMeansSingularTest() {}
protected:
    void run(int inVariant)
    {
        RNG& rng = cv::theRNG();
        int i, iter = 0, N = 0, N0 = 0, K = 0, dims = 0;
        Mat labels;

        {
            const int MAX_DIM=5;
            int MAX_POINTS = 100, maxIter = 100;
            for( iter = 0; iter < maxIter; iter++ )
            {
                ts->update_context(this, iter, true);
                dims = rng.uniform(inVariant == MAT_1_N_CDIM ? 2 : 1, MAX_DIM+1);
                N = rng.uniform(2, MAX_POINTS+1);
                N0 = rng.uniform(1, MAX(N/10, 2));
                K = rng.uniform(1, N+1);

                Mat centers;

                if (inVariant == VECTOR)
                {
                    dims = 2;

                    std::vector<cv::Point2f> data0(N0);
                    rng.fill(data0, RNG::UNIFORM, -1, 1);

                    std::vector<cv::Point2f> data(N);
                    for( i = 0; i < N; i++ )
                        data[i] = data0[rng.uniform(0, N0)];

                    kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
                           5, KMEANS_PP_CENTERS, centers);
                }
                else
                {
                    Mat data0(N0, dims, CV_32F);
                    rng.fill(data0, RNG::UNIFORM, -1, 1);

                    Mat data;

                    switch (inVariant)
                    {
                    case MAT_N_DIM_C1:
                        data.create(N, dims, CV_32F);
                        for( i = 0; i < N; i++ )
                            data0.row(rng.uniform(0, N0)).copyTo(data.row(i));
                        break;

                    case MAT_N_1_CDIM:
                        data.create(N, 1, CV_32FC(dims));
                        for( i = 0; i < N; i++ )
                            memcpy(data.ptr(i), data0.ptr(rng.uniform(0, N0)), dims * sizeof(float));
                        break;

                    case MAT_1_N_CDIM:
                        data.create(1, N, CV_32FC(dims));
                        for( i = 0; i < N; i++ )
                            memcpy(data.ptr() + i * dims * sizeof(float), data0.ptr(rng.uniform(0, N0)), dims * sizeof(float));
                        break;

                    case MAT_N_DIM_C1_NONCONT:
                        data.create(N, dims + 5, CV_32F);
                        data = data(Range(0, N), Range(0, dims));
                        for( i = 0; i < N; i++ )
                            data0.row(rng.uniform(0, N0)).copyTo(data.row(i));
                        break;

                    case MAT_N_1_CDIM_NONCONT:
                        data.create(N, 3, CV_32FC(dims));
                        data = data.colRange(0, 1);
                        for( i = 0; i < N; i++ )
                            memcpy(data.ptr(i), data0.ptr(rng.uniform(0, N0)), dims * sizeof(float));
                        break;
                    }

                    kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
                           5, KMEANS_PP_CENTERS, centers);
                }

                ASSERT_EQ(centers.rows, K);
                ASSERT_EQ(labels.rows, N);

                Mat hist(K, 1, CV_32S, Scalar(0));
                for( i = 0; i < N; i++ )
                {
                    int l = labels.at<int>(i);
                    ASSERT_GE(l, 0);
                    ASSERT_LT(l, K);
                    hist.at<int>(l)++;
                }
                for( i = 0; i < K; i++ )
                    ASSERT_GT(hist.at<int>(i), 0);
            }
        }
    }
};

TEST(Core_KMeans, singular) { CV_KMeansSingularTest test; test.safe_run(MAT_N_DIM_C1); }

CV_ENUM(KMeansInputVariant, MAT_N_DIM_C1, MAT_N_1_CDIM, MAT_1_N_CDIM, MAT_N_DIM_C1_NONCONT, MAT_N_1_CDIM_NONCONT, VECTOR)

typedef testing::TestWithParam<KMeansInputVariant> Core_KMeans_InputVariants;

TEST_P(Core_KMeans_InputVariants, singular)
{
    CV_KMeansSingularTest test;
    test.safe_run(GetParam());
}

INSTANTIATE_TEST_CASE_P(AllVariants, Core_KMeans_InputVariants, KMeansInputVariant::all());

TEST(Core_KMeans, compactness)
{
    const int N = 1024;
    const int attempts = 4;
    const TermCriteria crit = TermCriteria(TermCriteria::COUNT, 5, 0); // low number of iterations
    cvtest::TS& ts = *cvtest::TS::ptr();
    for (int K = 1; K <= N; K *= 2)
    {
        Mat data(N, 1, CV_32FC2);
        cvtest::randUni(ts.get_rng(), data, Scalar(-200, -200), Scalar(200, 200));
        Mat labels, centers;
        double compactness = kmeans(data, K, labels, crit, attempts, KMEANS_PP_CENTERS, centers);
        centers = centers.reshape(2);
        EXPECT_EQ(labels.rows, N);
        EXPECT_EQ(centers.rows, K);
        EXPECT_GE(compactness, 0.0);
        double expected = 0.0;
        for (int i = 0; i < N; ++i)
        {
            int l = labels.at<int>(i);
            Point2f d = data.at<Point2f>(i) - centers.at<Point2f>(l);
            expected += d.x * d.x + d.y * d.y;
        }
        EXPECT_NEAR(expected, compactness, expected * 1e-8);
        if (K == N)
        {
            EXPECT_DOUBLE_EQ(compactness, 0.0);
        }
    }
}

TEST(Core_KMeans, bad_input)
{
    const int N = 100;
    const int attempts = 4;
    const TermCriteria crit = TermCriteria(TermCriteria::COUNT, 5, 0); // low number of iterations
    const int K = 3;
    Mat data(N, 1, CV_32FC2);
    cv::randu(data, Scalar(-200, -200), Scalar(200, 200));
    {
        SCOPED_TRACE("Huge value");
        data.at<Vec2f>(10, 0) = Vec2f(1e20f, 0);
        Mat labels, centers;
        EXPECT_ANY_THROW(kmeans(data, K, labels, crit, attempts, KMEANS_PP_CENTERS, centers));
    }
    {
        SCOPED_TRACE("Negative value");
        data.at<Vec2f>(10, 0) = Vec2f(0, -1e20f);
        Mat labels, centers;
        EXPECT_ANY_THROW(kmeans(data, K, labels, crit, attempts, KMEANS_PP_CENTERS, centers));
    }
    {
        SCOPED_TRACE("NaN");
        data.at<Vec2f>(10, 0) = Vec2f(0, std::numeric_limits<float>::quiet_NaN());
        Mat labels, centers;
        EXPECT_ANY_THROW(kmeans(data, K, labels, crit, attempts, KMEANS_PP_CENTERS, centers));
    }
}

TEST(CovariationMatrixVectorOfMat, accuracy)
{
    unsigned int col_problem_size = 8, row_problem_size = 8, vector_size = 16;
    cv::Mat src(vector_size, col_problem_size * row_problem_size, CV_32F);
    int singleMatFlags = cv::COVAR_ROWS;

    cv::Mat gold;
    cv::Mat goldMean;
    cv::randu(src,cv::Scalar(-128), cv::Scalar(128));
    cv::calcCovarMatrix(src,gold,goldMean,singleMatFlags,CV_32F);
    std::vector<cv::Mat> srcVec;
    for(size_t i = 0; i < vector_size; i++)
    {
        srcVec.push_back(src.row(static_cast<int>(i)).reshape(0,col_problem_size));
    }

    cv::Mat actual;
    cv::Mat actualMean;
    cv::calcCovarMatrix(srcVec, actual, actualMean,singleMatFlags,CV_32F);

    cv::Mat diff;
    cv::absdiff(gold, actual, diff);
    cv::Scalar s = cv::sum(diff);
    ASSERT_EQ(s.dot(s), 0.0);

    cv::Mat meanDiff;
    cv::absdiff(goldMean, actualMean.reshape(0,1), meanDiff);
    cv::Scalar sDiff = cv::sum(meanDiff);
    ASSERT_EQ(sDiff.dot(sDiff), 0.0);
}

TEST(CovariationMatrixVectorOfMatWithMean, accuracy)
{
    unsigned int col_problem_size = 8, row_problem_size = 8, vector_size = 16;
    cv::Mat src(vector_size, col_problem_size * row_problem_size, CV_32F);
    int singleMatFlags = cv::COVAR_ROWS | cv::COVAR_USE_AVG;

    cv::Mat gold;
    cv::randu(src,cv::Scalar(-128), cv::Scalar(128));
    cv::Mat goldMean;

    cv::reduce(src, goldMean, 0, REDUCE_AVG, CV_32F);

    cv::calcCovarMatrix(src,gold,goldMean,singleMatFlags,CV_32F);

    std::vector<cv::Mat> srcVec;
    for(size_t i = 0; i < vector_size; i++)
    {
        srcVec.push_back(src.row(static_cast<int>(i)).reshape(0,col_problem_size));
    }

    cv::Mat actual;
    cv::Mat actualMean = goldMean.reshape(0, row_problem_size);
    cv::calcCovarMatrix(srcVec, actual, actualMean,singleMatFlags,CV_32F);

    cv::Mat diff;
    cv::absdiff(gold, actual, diff);
    cv::Scalar s = cv::sum(diff);
    ASSERT_EQ(s.dot(s), 0.0);

    cv::Mat meanDiff;
    cv::absdiff(goldMean, actualMean.reshape(0,1), meanDiff);
    cv::Scalar sDiff = cv::sum(meanDiff);
    ASSERT_EQ(sDiff.dot(sDiff), 0.0);
}

TEST(Core_Pow, special)
{
    for( int i = 0; i < 100; i++ )
    {
        int n = theRNG().uniform(1, 30);
        Mat mtx0(1, n, CV_8S), mtx, result;
        randu(mtx0, -5, 5);

        int type = theRNG().uniform(0, 2) ? CV_64F : CV_32F;
        double eps = type == CV_32F ? 1e-3 : 1e-10;
        mtx0.convertTo(mtx, type);
        // generate power from [-n, n] interval with 1/8 step - enough to check various cases.
        const int max_pf = 3;
        int pf = theRNG().uniform(0, max_pf*2+1);
        double power = ((1 << pf) - (1 << (max_pf*2-1)))/16.;
        int ipower = cvRound(power);
        bool is_ipower = ipower == power;
        cv::pow(mtx, power, result);
        for( int j = 0; j < n; j++ )
        {
            double val = type == CV_32F ? (double)mtx.at<float>(j) : mtx.at<double>(j);
            double r = type == CV_32F ? (double)result.at<float>(j) : result.at<double>(j);
            double r0;
            if( power == 0. )
                r0 = 1;
            else if( is_ipower )
            {
                r0 = 1;
                for( int k = 0; k < std::abs(ipower); k++ )
                    r0 *= val;
                if( ipower < 0 )
                    r0 = 1./r0;
            }
            else
                r0 = std::pow(val, power);
            if( cvIsInf(r0) )
            {
                ASSERT_TRUE(cvIsInf(r) != 0);
            }
            else if( cvIsNaN(r0) )
            {
                ASSERT_TRUE(cvIsNaN(r) != 0);
            }
            else
            {
                ASSERT_TRUE(cvIsInf(r) == 0 && cvIsNaN(r) == 0);
                ASSERT_LT(fabs(r - r0), eps);
            }
        }
    }
}

TEST(Core_Cholesky, accuracy64f)
{
    const int n = 5;
    Mat A(n, n, CV_64F), refA;
    Mat mean(1, 1, CV_64F);
    *mean.ptr<double>() = 10.0;
    Mat dev(1, 1, CV_64F);
    *dev.ptr<double>() = 10.0;
    RNG rng(10);
    rng.fill(A, RNG::NORMAL, mean, dev);
    A = A*A.t();
    A.copyTo(refA);
    Cholesky(A.ptr<double>(), A.step, n, NULL, 0, 0);

   for (int i = 0; i < A.rows; i++)
       for (int j = i + 1; j < A.cols; j++)
           A.at<double>(i, j) = 0.0;
   EXPECT_LE(cvtest::norm(refA, A*A.t(), cv::NORM_L2 | cv::NORM_RELATIVE), FLT_EPSILON);
}

TEST(Core_QR_Solver, accuracy64f)
{
    int m = 20, n = 18;
    Mat A(m, m, CV_64F);
    Mat B(m, n, CV_64F);
    Mat mean(1, 1, CV_64F);
    *mean.ptr<double>() = 10.0;
    Mat dev(1, 1, CV_64F);
    *dev.ptr<double>() = 10.0;
    RNG rng(10);
    rng.fill(A, RNG::NORMAL, mean, dev);
    rng.fill(B, RNG::NORMAL, mean, dev);
    A = A*A.t();
    Mat solutionQR;

    //solve system with square matrix
    solve(A, B, solutionQR, DECOMP_QR);
    EXPECT_LE(cvtest::norm(A*solutionQR, B, cv::NORM_L2 | cv::NORM_RELATIVE), FLT_EPSILON);

    A = Mat(m, n, CV_64F);
    B = Mat(m, n, CV_64F);
    rng.fill(A, RNG::NORMAL, mean, dev);
    rng.fill(B, RNG::NORMAL, mean, dev);

    //solve normal system
    solve(A, B, solutionQR, DECOMP_QR | DECOMP_NORMAL);
    EXPECT_LE(cvtest::norm(A.t()*(A*solutionQR), A.t()*B, cv::NORM_L2 | cv::NORM_RELATIVE), FLT_EPSILON);

    //solve overdeterminated system as a least squares problem
    Mat solutionSVD;
    solve(A, B, solutionQR, DECOMP_QR);
    solve(A, B, solutionSVD, DECOMP_SVD);
    EXPECT_LE(cvtest::norm(solutionQR, solutionSVD, cv::NORM_L2 | cv::NORM_RELATIVE), FLT_EPSILON);

    //solve system with singular matrix
    A = Mat(10, 10, CV_64F);
    B = Mat(10, 1, CV_64F);
    rng.fill(A, RNG::NORMAL, mean, dev);
    rng.fill(B, RNG::NORMAL, mean, dev);
    for (int i = 0; i < A.cols; i++)
      A.at<double>(0, i) = A.at<double>(1, i);
    ASSERT_FALSE(solve(A, B, solutionQR, DECOMP_QR));
}

TEST(Core_Solve, regression_11888)
{
    cv::Matx<float, 3, 2> A(
        2, 1,
        3, 1,
        6, 1
    );
    cv::Vec<float, 3> b(4, 5, 7);
    cv::Matx<float, 2, 1> xQR = A.solve(b, DECOMP_QR);
    cv::Matx<float, 2, 1> xSVD = A.solve(b, DECOMP_SVD);
    EXPECT_LE(cvtest::norm(xQR, xSVD, NORM_L2 | NORM_RELATIVE), 0.001);
    cv::Matx<float, 2, 3> iA = A.inv(DECOMP_SVD);
    EXPECT_LE(cvtest::norm(iA*A, Matx<float, 2, 2>::eye(), NORM_L2), 1e-3);
    EXPECT_ANY_THROW({
       /*cv::Matx<float, 2, 1> xLU =*/ A.solve(b, DECOMP_LU);
       std::cout << "FATAL ERROR" << std::endl;
    });
}

TEST(Core_Solve, Matx_2_2)
{
    cv::Matx<float, 2, 2> A(
        2, 1,
        1, 1
    );
    cv::Vec<float, 2> b(4, 5);
    cv::Matx<float, 2, 1> xLU = A.solve(b, DECOMP_LU);
    cv::Matx<float, 2, 1> xQR = A.solve(b, DECOMP_QR);
    cv::Matx<float, 2, 1> xSVD = A.solve(b, DECOMP_SVD);
    EXPECT_LE(cvtest::norm(xQR, xSVD, NORM_L2 | NORM_RELATIVE), 1e-3);
    EXPECT_LE(cvtest::norm(xQR, xLU, NORM_L2 | NORM_RELATIVE), 1e-3);
    cv::Matx<float, 2, 2> iA = A.inv(DECOMP_SVD);
    EXPECT_LE(cvtest::norm(iA*A, Matx<float, 2, 2>::eye(), NORM_L2), 1e-3);
}
TEST(Core_Solve, Matx_3_3)
{
    cv::Matx<float, 3, 3> A(
        2, 1, 0,
        0, 1, 1,
        1, 0, 1
    );
    cv::Vec<float, 3> b(4, 5, 6);
    cv::Matx<float, 3, 1> xLU = A.solve(b, DECOMP_LU);
    cv::Matx<float, 3, 1> xQR = A.solve(b, DECOMP_QR);
    cv::Matx<float, 3, 1> xSVD = A.solve(b, DECOMP_SVD);
    EXPECT_LE(cvtest::norm(xQR, xSVD, NORM_L2 | NORM_RELATIVE), 1e-3);
    EXPECT_LE(cvtest::norm(xQR, xLU, NORM_L2 | NORM_RELATIVE), 1e-3);
    cv::Matx<float, 3, 3> iA = A.inv(DECOMP_SVD);
    EXPECT_LE(cvtest::norm(iA*A, Matx<float, 3, 3>::eye(), NORM_L2), 1e-3);
}

TEST(Core_Solve, Matx_4_4)
{
    cv::Matx<float, 4, 4> A(
        2, 1, 0, 4,
        0, 1, 1, 3,
        1, 0, 1, 2,
        2, 2, 0, 1
    );
    cv::Vec<float, 4> b(4, 5, 6, 7);
    cv::Matx<float, 4, 1> xLU = A.solve(b, DECOMP_LU);
    cv::Matx<float, 4, 1> xQR = A.solve(b, DECOMP_QR);
    cv::Matx<float, 4, 1> xSVD = A.solve(b, DECOMP_SVD);
    EXPECT_LE(cvtest::norm(xQR, xSVD, NORM_L2 | NORM_RELATIVE), 1e-3);
    EXPECT_LE(cvtest::norm(xQR, xLU, NORM_L2 | NORM_RELATIVE), 1e-3);
    cv::Matx<float, 4, 4> iA = A.inv(DECOMP_SVD);
    EXPECT_LE(cvtest::norm(iA*A, Matx<float, 4, 4>::eye(), NORM_L2), 1e-3);
}

softdouble naiveExp(softdouble x)
{
    int exponent = x.getExp();
    int sign = x.getSign() ? -1 : 1;
    if(sign < 0 && exponent >= 10) return softdouble::inf();
    softdouble mantissa = x.getFrac();
    //Taylor series for mantissa
    uint64 fac[20] = {1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
                    39916800, 479001600, 6227020800, 87178291200, 1307674368000,
                    20922789888000, 355687428096000, 6402373705728000, 121645100408832000,
                    2432902008176640000};
    softdouble sum = softdouble::one();
    // 21! > (2 ** 64)
    for(int i = 20; i > 0; i--)
        sum += pow(mantissa, softdouble(i))/softdouble(fac[i-1]);
    if(exponent >= 0)
    {
        exponent = (1 << exponent);
        return pow(sum, softdouble(exponent*sign));
    }
    else
    {
        if(sign < 0) sum = softdouble::one()/sum;
        exponent = -exponent;
        for(int j = 0; j < exponent; j++)
            sum = sqrt(sum);
        return sum;
    }
}

static float makeFP32(int sign, int exponent, int significand)
{
    Cv32suf x;
    x.u = (unsigned)(((sign & 1) << 31) | ((exponent&255) << 23) | (significand & 0x7fffff));
    return x.f;
}

static float makeRandomFP32(RNG& rng, int sign, int exprange)
{
    if( sign == -1 )
        sign = rng() % 2;
    int exponent = rng() % exprange;
    int significand = rng() % (1 << 23);
    return makeFP32(sign, exponent, significand);
}

TEST(Core_SoftFloat, exp32)
{
    //special cases
    EXPECT_TRUE(exp( softfloat::nan()).isNaN());
    EXPECT_TRUE(exp( softfloat::inf()).isInf());
    EXPECT_EQ  (exp(-softfloat::inf()), softfloat::zero());

    //ln(FLT_MAX) ~ 88.722
    const softfloat ln_max(88.722f);
    vector<softfloat> inputs;
    RNG rng(0);
    inputs.push_back(softfloat::zero());
    inputs.push_back(softfloat::one());
    inputs.push_back(softfloat::min());
    for(int i = 0; i < 50000; i++)
    {
        float x = makeRandomFP32(rng, -1, 10+127 //bigger exponent will produce inf
                                 );
        if(softfloat(x) > ln_max)
            x = rng.uniform(0.0f, (float)ln_max);
        inputs.push_back(softfloat(x));
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        softfloat x(inputs[i]);
        softfloat y = exp(x);
        ASSERT_TRUE(!y.isNaN());
        ASSERT_TRUE(!y.isInf());
        ASSERT_GE(y, softfloat::zero());
        softfloat ygood = naiveExp(x);
        softfloat diff = abs(ygood - y);
        const softfloat eps = softfloat::eps();
        if(diff > eps)
        {
            ASSERT_LE(diff/max(abs(y), abs(ygood)), eps);
        }
    }
}

TEST(Core_SoftFloat, exp64)
{
    //special cases
    EXPECT_TRUE(exp( softdouble::nan()).isNaN());
    EXPECT_TRUE(exp( softdouble::inf()).isInf());
    EXPECT_EQ  (exp(-softdouble::inf()), softdouble::zero());

    //ln(DBL_MAX) ~ 709.7827
    const softdouble ln_max(709.7827);
    vector<softdouble> inputs;
    RNG rng(0);
    inputs.push_back(softdouble::zero());
    inputs.push_back(softdouble::one());
    inputs.push_back(softdouble::min());
    for(int i = 0; i < 50000; i++)
    {
        Cv64suf x;
        uint64 sign = rng() % 2;
        uint64 exponent = rng() % (10 + 1023); //bigger exponent will produce inf
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.u = (sign << 63) | (exponent << 52) | mantissa;
        if(softdouble(x.f) > ln_max)
            x.f = rng.uniform(0.0, (double)ln_max);
        inputs.push_back(softdouble(x.f));
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        softdouble x(inputs[i]);
        softdouble y = exp(x);
        ASSERT_TRUE(!y.isNaN());
        ASSERT_TRUE(!y.isInf());
        ASSERT_GE(y, softdouble::zero());
        softdouble ygood = naiveExp(x);
        softdouble diff = abs(ygood - y);
        const softdouble eps = softdouble::eps();
        if(diff > eps)
        {
            ASSERT_LE(diff/max(abs(y), abs(ygood)), softdouble(8192)*eps);
        }
    }
}

TEST(Core_SoftFloat, log32)
{
    const int nValues = 50000;
    RNG rng(0);
    //special cases
    EXPECT_TRUE(log(softfloat::nan()).isNaN());
    for(int i = 0; i < nValues; i++)
    {
        softfloat x32(makeRandomFP32(rng, 1, 255));
        ASSERT_TRUE(log(x32).isNaN());
    }
    EXPECT_TRUE(log(softfloat::zero()).isInf());

    vector<softfloat> inputs;

    inputs.push_back(softfloat::one());
    inputs.push_back(softfloat(exp(softfloat::one())));
    inputs.push_back(softfloat::min());
    inputs.push_back(softfloat::max());
    for(int i = 0; i < nValues; i++)
    {
        inputs.push_back(softfloat(makeRandomFP32(rng, 0, 255)));
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        softfloat x(inputs[i]);
        softfloat y = log(x);
        ASSERT_TRUE(!y.isNaN());
        ASSERT_TRUE(!y.isInf());
        softfloat ex = exp(y);
        softfloat diff = abs(ex - x);
        // 88 is approx estimate of max exp() argument
        ASSERT_TRUE(!ex.isInf() || (y > softfloat(88)));
        const softfloat eps2 = softfloat().setExp(-17);
        if(!ex.isInf() && diff > softfloat::eps())
        {
            ASSERT_LT(diff/max(abs(ex), x), eps2);
        }
    }
}

TEST(Core_SoftFloat, log64)
{
    const int nValues = 50000;
    RNG rng(0);
    //special cases
    EXPECT_TRUE(log(softdouble::nan()).isNaN());
    for(int i = 0; i < nValues; i++)
    {
        Cv64suf x;
        uint64 sign = 1;
        uint64 exponent = rng() % 2047;
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.u = (sign << 63) | (exponent << 52) | mantissa;
        softdouble x64(x.f);
        ASSERT_TRUE(log(x64).isNaN());
    }
    EXPECT_TRUE(log(softdouble::zero()).isInf());

    vector<softdouble> inputs;
    inputs.push_back(softdouble::one());
    inputs.push_back(exp(softdouble::one()));
    inputs.push_back(softdouble::min());
    inputs.push_back(softdouble::max());
    for(int i = 0; i < nValues; i++)
    {
        Cv64suf x;
        uint64 sign = 0;
        uint64 exponent = rng() % 2047;
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.u = (sign << 63) | (exponent << 52) | mantissa;
        inputs.push_back(softdouble(x.f));
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        softdouble x(inputs[i]);
        softdouble y = log(x);
        ASSERT_TRUE(!y.isNaN());
        ASSERT_TRUE(!y.isInf());
        softdouble ex = exp(y);
        softdouble diff = abs(ex - x);
        // 700 is approx estimate of max exp() argument
        ASSERT_TRUE(!ex.isInf() || (y > softdouble(700)));
        const softdouble eps2 = softdouble().setExp(-41);
        if(!ex.isInf() && diff > softdouble::eps())
        {
            ASSERT_LT(diff/max(abs(ex), x), eps2);
        }
    }
}

TEST(Core_SoftFloat, cbrt32)
{
    vector<softfloat> inputs;
    RNG rng(0);
    inputs.push_back(softfloat::zero());
    inputs.push_back(softfloat::one());
    inputs.push_back(softfloat::max());
    inputs.push_back(softfloat::min());
    for(int i = 0; i < 50000; i++)
    {
        inputs.push_back(softfloat(makeRandomFP32(rng, -1, 255)));
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        softfloat x(inputs[i]);
        softfloat y = cbrt(x);
        ASSERT_TRUE(!y.isNaN());
        ASSERT_TRUE(!y.isInf());
        softfloat cube = y*y*y;
        softfloat diff = abs(x - cube);
        const softfloat eps = softfloat::eps();
        if(diff > eps)
        {
            ASSERT_LT(diff/max(abs(x), abs(cube)), softfloat(4)*eps);
        }
    }
}

TEST(Core_SoftFloat, pow32)
{
    const softfloat zero = softfloat::zero(), one = softfloat::one();
    const softfloat  inf = softfloat::inf(),  nan = softfloat::nan();
    const size_t nValues = 5000;
    RNG rng(0);
    //x ** nan == nan
    for(size_t i = 0; i < nValues; i++)
    {
        Cv32suf x;
        x.u = rng();
        ASSERT_TRUE(pow(softfloat(x.f), nan).isNaN());
    }
    //x ** inf check
    for(size_t i = 0; i < nValues; i++)
    {
        Cv32suf x;
        x.u = rng();
        softfloat x32(x.f);
        softfloat ax = abs(x32);
        if(x32.isNaN())
        {
            ASSERT_TRUE(pow(x32, inf).isNaN());
        }
        if(ax > one)
        {
            ASSERT_TRUE(pow(x32,  inf).isInf());
            ASSERT_EQ  (pow(x32, -inf), zero);
        }
        if(ax < one && ax > zero)
        {
            ASSERT_TRUE(pow(x32, -inf).isInf());
            ASSERT_EQ  (pow(x32,  inf), zero);
        }
    }
    //+-1 ** inf
    EXPECT_TRUE(pow( one, inf).isNaN());
    EXPECT_TRUE(pow(-one, inf).isNaN());

    // x ** 0 == 1
    for(size_t i = 0; i < nValues; i++)
    {
        Cv32suf x;
        x.u = rng();
        ASSERT_EQ(pow(softfloat(x.f), zero), one);
    }

    // x ** 1 == x
    for(size_t i = 0; i < nValues; i++)
    {
        Cv32suf x;
        x.u = rng();
        softfloat x32(x.f);
        softfloat val = pow(x32, one);
        // don't compare val and x32 directly because x != x if x is nan
        ASSERT_EQ(val.v, x32.v);
    }

    // nan ** y == nan, if y != 0
    for(size_t i = 0; i < nValues; i++)
    {
        unsigned u = rng();
        softfloat x32 = softfloat::fromRaw(u);
        x32 = (x32 != softfloat::zero()) ? x32 : softfloat::min();
        ASSERT_TRUE(pow(nan, x32).isNaN());
    }
    // nan ** 0 == 1
    EXPECT_EQ(pow(nan, zero), one);

    // inf ** y == 0, if y < 0
    // inf ** y == inf, if y > 0
    for(size_t i = 0; i < nValues; i++)
    {
        float x = makeRandomFP32(rng, 0, 255);
        softfloat x32 = softfloat(x);
        ASSERT_TRUE(pow( inf, x32).isInf());
        ASSERT_TRUE(pow(-inf, x32).isInf());
        ASSERT_EQ(pow( inf, -x32), zero);
        ASSERT_EQ(pow(-inf, -x32), zero);
    }

    // x ** y ==   (-x) ** y, if y % 2 == 0
    // x ** y == - (-x) ** y, if y % 2 == 1
    // x ** y == nan, if x < 0 and y is not integer
    for(size_t i = 0; i < nValues; i++)
    {
        softfloat x32(makeRandomFP32(rng, 1, 255));
        softfloat y32(makeRandomFP32(rng, -1, 23+127 //bigger exponent produces integer numbers only
                                     ));
        int yi = cvRound(y32);
        if(y32 != softfloat(yi))
            ASSERT_TRUE(pow(x32, y32).isNaN());
        else if(yi % 2)
            ASSERT_EQ(pow(-x32, y32), -pow(x32, y32));
        else
            ASSERT_EQ(pow(-x32, y32),  pow(x32, y32));
    }

    // (0 ** 0) == 1
    EXPECT_EQ(pow(zero, zero), one);

    // 0 ** y == inf, if y < 0
    // 0 ** y == 0, if y > 0
    for(size_t i = 0; i < nValues; i++)
    {
        softfloat x32(makeRandomFP32(rng, 0, 255));
        ASSERT_TRUE(pow(zero, -x32).isInf());
        if(x32 != one)
        {
            ASSERT_EQ(pow(zero, x32), zero);
        }
    }
}

TEST(Core_SoftFloat, pow64)
{
    const softdouble zero = softdouble::zero(), one = softdouble::one();
    const softdouble  inf = softdouble::inf(),  nan = softdouble::nan();

    const size_t nValues = 5000;
    RNG rng(0);

    //x ** nan == nan
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        x.u = ((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng);
        ASSERT_TRUE(pow(softdouble(x.f), nan).isNaN());
    }
    //x ** inf check
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        x.u = ((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng);
        softdouble x64(x.f);
        softdouble ax = abs(x64);
        if(x64.isNaN())
        {
            ASSERT_TRUE(pow(x64, inf).isNaN());
        }
        if(ax > one)
        {
            ASSERT_TRUE(pow(x64, inf).isInf());
            ASSERT_EQ(pow(x64, -inf), zero);
        }
        if(ax < one && ax > zero)
        {
            ASSERT_TRUE(pow(x64, -inf).isInf());
            ASSERT_EQ(pow(x64, inf), zero);
        }
    }
    //+-1 ** inf
    EXPECT_TRUE(pow( one, inf).isNaN());
    EXPECT_TRUE(pow(-one, inf).isNaN());

    // x ** 0 == 1
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        x.u = ((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng);
        ASSERT_EQ(pow(softdouble(x.f), zero), one);
    }

    // x ** 1 == x
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        x.u = ((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng);
        softdouble x64(x.f);
        softdouble val = pow(x64, one);
        // don't compare val and x64 directly because x != x if x is nan
        ASSERT_EQ(val.v, x64.v);
    }

    // nan ** y == nan, if y != 0
    for(size_t i = 0; i < nValues; i++)
    {
        uint64 u = ((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng);
        softdouble x64 = softdouble::fromRaw(u);
        x64 = (x64 != softdouble::zero()) ? x64 : softdouble::min();
        ASSERT_TRUE(pow(nan, x64).isNaN());
    }
    // nan ** 0 == 1
    EXPECT_EQ(pow(nan, zero), one);

    // inf ** y == 0, if y < 0
    // inf ** y == inf, if y > 0
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        uint64 sign = 0;
        uint64 exponent = rng() % 2047;
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.u = (sign << 63) | (exponent << 52) | mantissa;
        softdouble x64(x.f);
        ASSERT_TRUE(pow( inf, x64).isInf());
        ASSERT_TRUE(pow(-inf, x64).isInf());
        ASSERT_EQ(pow( inf, -x64), zero);
        ASSERT_EQ(pow(-inf, -x64), zero);
    }

    // x ** y ==   (-x) ** y, if y % 2 == 0
    // x ** y == - (-x) ** y, if y % 2 == 1
    // x ** y == nan, if x < 0 and y is not integer
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        uint64 sign = 1;
        uint64 exponent = rng() % 2047;
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.u = (sign << 63) | (exponent << 52) | mantissa;
        softdouble x64(x.f);
        Cv64suf y;
        sign = rng() % 2;
        //bigger exponent produces integer numbers only
        //exponent = rng() % (52 + 1023);
        //bigger exponent is too big
        exponent = rng() % (23 + 1023);
        mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        y.u = (sign << 63) | (exponent << 52) | mantissa;
        softdouble y64(y.f);
        uint64 yi = cvRound(y64);
        if(y64 != softdouble(yi))
            ASSERT_TRUE(pow(x64, y64).isNaN());
        else if(yi % 2)
            ASSERT_EQ(pow(-x64, y64), -pow(x64, y64));
        else
            ASSERT_EQ(pow(-x64, y64),  pow(x64, y64));
    }

    // (0 ** 0) == 1
    EXPECT_EQ(pow(zero, zero), one);

    // 0 ** y == inf, if y < 0
    // 0 ** y == 0, if y > 0
    for(size_t i = 0; i < nValues; i++)
    {
        Cv64suf x;
        uint64 sign = 0;
        uint64 exponent = rng() % 2047;
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.u = (sign << 63) | (exponent << 52) | mantissa;
        softdouble x64(x.f);

        ASSERT_TRUE(pow(zero, -x64).isInf());
        if(x64 != one)
        {
            ASSERT_EQ(pow(zero, x64), zero);
        }
    }
}

TEST(Core_SoftFloat, sincos64)
{
    static const softdouble
            two = softdouble(2), three = softdouble(3),
            half = softdouble::one()/two,
            zero = softdouble::zero(), one = softdouble::one(),
            pi = softdouble::pi(), piby2 = pi/two, eps = softdouble::eps(),
            sin45 = sqrt(two)/two, sin60 = sqrt(three)/two;

    softdouble vstdAngles[] =
    //x, sin(x), cos(x)
    {
            zero,              zero,   one,
            pi/softdouble(6),  half, sin60,
            pi/softdouble(4), sin45, sin45,
            pi/three, sin60,  half,
    };
    vector<softdouble> stdAngles;
    stdAngles.assign(vstdAngles, vstdAngles + 3*4);

    static const softdouble stdEps = eps.setExp(-39);
    const size_t nStdValues = 5000;
    for(size_t i = 0; i < nStdValues; i++)
    {
        for(size_t k = 0; k < stdAngles.size()/3; k++)
        {
            softdouble x = stdAngles[k*3] + pi*softdouble(2*((int)i-(int)nStdValues/2));
            softdouble s = stdAngles[k*3+1];
            softdouble c = stdAngles[k*3+2];
            ASSERT_LE(abs(sin(x) - s), stdEps);
            ASSERT_LE(abs(cos(x) - c), stdEps);
            //sin(x+pi/2) = cos(x)
            ASSERT_LE(abs(sin(x + piby2) - c), stdEps);
            //sin(x+pi) = -sin(x)
            ASSERT_LE(abs(sin(x + pi) + s), stdEps);
            //cos(x+pi/2) = -sin(x)
            ASSERT_LE(abs(cos(x+piby2) + s), stdEps);
            //cos(x+pi) = -cos(x)
            ASSERT_LE(abs(cos(x+pi) + c), stdEps);
        }
    }

    // sin(x) is NaN iff x ix NaN or Inf
    EXPECT_TRUE(sin(softdouble::inf()).isNaN());
    EXPECT_TRUE(sin(softdouble::nan()).isNaN());

    vector<int> exponents;
    exponents.push_back(0);
    for(int i = 1; i < 52; i++)
    {
        exponents.push_back( i);
        exponents.push_back(-i);
    }
    exponents.push_back(256); exponents.push_back(-256);
    exponents.push_back(512); exponents.push_back(-512);
    exponents.push_back(1022); exponents.push_back(-1022);

    vector<softdouble> inputs;
    RNG rng(0);

    static const size_t nValues = 1 << 18;
    for(size_t i = 0; i < nValues; i++)
    {
        softdouble x;
        uint64 mantissa = (((long long int)((unsigned int)(rng)) << 32 ) | (unsigned int)(rng)) & ((1LL << 52) - 1);
        x.v = mantissa;
        x = x.setSign((rng() % 2) != 0);
        x = x.setExp(exponents[rng() % exponents.size()]);
        inputs.push_back(x);
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        softdouble x = inputs[i];

        int xexp = x.getExp();
        softdouble randEps = eps.setExp(std::max(xexp-52, -46));
        softdouble sx = sin(x);
        softdouble cx = cos(x);
        ASSERT_FALSE(sx.isInf()); ASSERT_FALSE(cx.isInf());
        ASSERT_FALSE(sx.isNaN()); ASSERT_FALSE(cx.isNaN());
        ASSERT_LE(abs(sx), one); ASSERT_LE(abs(cx), one);
        ASSERT_LE(abs((sx*sx + cx*cx) - one), eps);
        ASSERT_LE(abs(sin(x*two) - two*sx*cx), randEps);
        ASSERT_LE(abs(cos(x*two) - (cx*cx - sx*sx)), randEps);
        ASSERT_LE(abs(sin(-x) + sx), eps);
        ASSERT_LE(abs(cos(-x) - cx), eps);
        ASSERT_LE(abs(sin(x + piby2) - cx), randEps);
        ASSERT_LE(abs(sin(x + pi) + sx), randEps);
        ASSERT_LE(abs(cos(x+piby2) + sx), randEps);
        ASSERT_LE(abs(cos(x+pi) + cx), randEps);
    }
}

TEST(Core_SoftFloat, CvRound)
{
    struct
    {
        uint64_t inVal;
        int64_t out64;
        int32_t out32;
    } _values[] =
    {
        { 0x0123456789abcdefU,                     0,             0 }, // 3.51270056408850369812238561681E-303
        { 0x0000000000000000U,                     0,             0 }, // 0
        { 0x8000000000000000U,                     0,             0 }, // -0
        { 0x000123456789abcdU,                     0,             0 }, // 1.5822747438273385725152200433E-309
        { 0x800123456789abcdU,                     0,             0 }, // -1.5822747438273385725152200433E-309
        { 0x7ff0000000000000U,             INT64_MAX,     INT32_MAX }, // +inf
        { 0xfff0000000000000U,             INT64_MIN,     INT32_MIN }, // -inf
        { 0x7ff0000000000001U,             INT64_MAX,     INT32_MAX }, // nan(casts to maximum value)
        { 0xfff0000000000001U,             INT64_MAX,     INT32_MAX }, // nan(casts to maximum value)
        { 0x7ffa5a5a5a5a5a5aU,             INT64_MAX,     INT32_MAX }, // nan(casts to maximum value)
        { 0xfffa5a5a5a5a5a5aU,             INT64_MAX,     INT32_MAX }, // nan(casts to maximum value)
        { 0x7fe123456789abcdU,             INT64_MAX,     INT32_MAX }, // 9.627645455595956656406699747E307
        { 0xffe123456789abcdU,             INT64_MIN,     INT32_MIN }, // -9.627645455595956656406699747E307
        { 0x43ffffffffffffffU,             INT64_MAX,     INT32_MAX }, // (2^53-1)*2^12
        { 0xc3ffffffffffffffU,             INT64_MIN,     INT32_MIN }, // -(2^53-1)*2^12
        { 0x43f0000000000000U,             INT64_MAX,     INT32_MAX }, // 2^64
        { 0xc3f0000000000000U,             INT64_MIN,     INT32_MIN }, // -2^64
        { 0x43efffffffffffffU,             INT64_MAX,     INT32_MAX }, // (2^53-1)*2^11
        { 0xc3efffffffffffffU,             INT64_MIN,     INT32_MIN }, // -(2^53-1)*2^11
        { 0x43e0000000000000U,             INT64_MAX,     INT32_MAX }, // 2^63
        { 0xc3e0000000000000U, -0x7fffffffffffffff-1,     INT32_MIN }, // -2^63
        { 0x43dfffffffffffffU,    0x7ffffffffffffc00,     INT32_MAX }, // (2^53-1)*2^10
        { 0xc3dfffffffffffffU,   -0x7ffffffffffffc00,     INT32_MIN }, // -(2^53-1)*2^10
        { 0x433fffffffffffffU,      0x1fffffffffffff,     INT32_MAX }, // (2^53-1)
        { 0xc33fffffffffffffU,     -0x1fffffffffffff,     INT32_MIN }, // -(2^53-1)
        { 0x432fffffffffffffU,      0x10000000000000,     INT32_MAX }, // (2^52-1) + 0.5
        { 0xc32fffffffffffffU,     -0x10000000000000,     INT32_MIN }, // -(2^52-1) - 0.5
        { 0x431fffffffffffffU,       0x8000000000000,     INT32_MAX }, // (2^51-1) + 0.75
        { 0xc31fffffffffffffU,      -0x8000000000000,     INT32_MIN }, // -(2^51-1) - 0.75
        { 0x431ffffffffffffeU,       0x8000000000000,     INT32_MAX }, // (2^51-1) + 0.5
        { 0xc31ffffffffffffeU,      -0x8000000000000,     INT32_MIN }, // -(2^51-1) - 0.5
        { 0x431ffffffffffffdU,       0x7ffffffffffff,     INT32_MAX }, // (2^51-1) + 0.25
        { 0xc31ffffffffffffdU,      -0x7ffffffffffff,     INT32_MIN }, // -(2^51-1) - 0.25

        { 0x41f0000000000000U,           0x100000000,     INT32_MAX }, // 2^32 = 4294967296
        { 0xc1f0000000000000U,          -0x100000000,     INT32_MIN }, // -2^32 = -4294967296
        { 0x41efffffffffffffU,           0x100000000,     INT32_MAX }, // 4294967295.99999952316284179688
        { 0xc1efffffffffffffU,          -0x100000000,     INT32_MIN }, // -4294967295.99999952316284179688
        { 0x41effffffff00000U,           0x100000000,     INT32_MAX }, // (2^32-1) + 0.5 = 4294967295.5
        { 0xc1effffffff00000U,          -0x100000000,     INT32_MIN }, // -(2^32-1) - 0.5 = -4294967295.5
        { 0x41efffffffe00000U,          0xffffffffll,     INT32_MAX }, // (2^32-1)
        { 0xc1efffffffe00000U,         -0xffffffffll,     INT32_MIN }, // -(2^32-1)
        { 0x41e0000000000000U,          0x80000000ll,     INT32_MAX }, // 2^31 = 2147483648
        { 0xc1e0000000000000U,         -0x80000000ll, -0x7fffffff-1 }, // -2^31 = -2147483648
        { 0x41dfffffffffffffU,          0x80000000ll,     INT32_MAX }, // 2147483647.99999976158142089844
        { 0xc1dfffffffffffffU,         -0x80000000ll, -0x7fffffff-1 }, // -2147483647.99999976158142089844

        { 0x41dffffffff00000U,          0x80000000ll,     INT32_MAX }, // (2^31-1) + 0.75
        { 0xc1dffffffff00000U,         -0x80000000ll, -0x7fffffff-1 }, // -(2^31-1) - 0.75
        { 0x41dfffffffe00001U,          0x80000000ll,     INT32_MAX }, // (2^31-1) + 0.5 + 2^-22
        { 0xc1dfffffffe00001U,         -0x80000000ll, -0x7fffffff-1 }, // -(2^31-1) - 0.5 - 2^-22
        { 0x41dfffffffe00000U,          0x80000000ll,     INT32_MAX }, // (2^31-1) + 0.5
        { 0xc1dfffffffe00000U,         -0x80000000ll, -0x7fffffff-1 }, // -(2^31-1) - 0.5
        { 0x41dfffffffdfffffU,            0x7fffffff,    0x7fffffff }, // (2^31-1) + 0.5 - 2^-22
        { 0xc1dfffffffdfffffU,           -0x7fffffff,   -0x7fffffff }, // -(2^31-1) - 0.5 + 2^-22
        { 0x41dfffffffd00000U,            0x7fffffff,    0x7fffffff }, // (2^31-1) + 0.25
        { 0xc1dfffffffd00000U,           -0x7fffffff,   -0x7fffffff }, // -(2^31-1) - 0.25
        { 0x41dfffffffc00000U,            0x7fffffff,    0x7fffffff }, // (2^31-1)
        { 0xc1dfffffffc00000U,           -0x7fffffff,   -0x7fffffff }, // -(2^31-1)
        { 0x41d0000000000000U,            0x40000000,    0x40000000 }, // 2^30 = 2147483648
        { 0xc1d0000000000000U,           -0x40000000,   -0x40000000 }, // -2^30 = -2147483648

        { 0x4006000000000000U,                     3,             3 }, // 2.75
        { 0xc006000000000000U,                    -3,            -3 }, // -2.75
        { 0x4004000000000001U,                     3,             3 }, // 2.5 + 2^-51
        { 0xc004000000000001U,                    -3,            -3 }, // -2.5 - 2^-51
        { 0x4004000000000000U,                     2,             2 }, // 2.5
        { 0xc004000000000000U,                    -2,            -2 }, // -2.5
        { 0x4003ffffffffffffU,                     2,             2 }, // 2.5 - 2^-51
        { 0xc003ffffffffffffU,                    -2,            -2 }, // -2.5 + 2^-51
        { 0x4002000000000000U,                     2,             2 }, // 2.25
        { 0xc002000000000000U,                    -2,            -2 }, // -2.25

        { 0x3ffc000000000000U,                     2,             2 }, // 1.75
        { 0xbffc000000000000U,                    -2,            -2 }, // -1.75
        { 0x3ff8000000000001U,                     2,             2 }, // 1.5 + 2^-52
        { 0xbff8000000000001U,                    -2,            -2 }, // -1.5 - 2^-52
        { 0x3ff8000000000000U,                     2,             2 }, // 1.5
        { 0xbff8000000000000U,                    -2,            -2 }, // -1.5
        { 0x3ff7ffffffffffffU,                     1,             1 }, // 1.5 - 2^-52
        { 0xbff7ffffffffffffU,                    -1,            -1 }, // -1.5 + 2^-52
        { 0x3ff4000000000000U,                     1,             1 }, // 1.25
        { 0xbff4000000000000U,                    -1,            -1 }, // -1.25

        { 0x3fe8000000000000U,                     1,             1 }, // 0.75
        { 0xbfe8000000000000U,                    -1,            -1 }, // -0.75
        { 0x3fe0000000000001U,                     1,             1 }, // 0.5 + 2^-53
        { 0xbfe0000000000001U,                    -1,            -1 }, // -0.5 - 2^-53
        { 0x3fe0000000000000U,                     0,             0 }, // 0.5
        { 0xbfe0000000000000U,                     0,             0 }, // -0.5

        { 0x3fd8000000000000U,                     0,             0 }, // 0.375
        { 0xbfd8000000000000U,                     0,             0 }, // -0.375
        { 0x3fd0000000000000U,                     0,             0 }, // 0.25
        { 0xbfd0000000000000U,                     0,             0 }, // -0.25

        { 0x0ff123456789abcdU,                     0,             0 }, // 6.89918601543515033558134828315E-232
        { 0x8ff123456789abcdU,                     0,             0 }  // -6.89918601543515033558134828315E-232
    };
    struct testvalues
    {
        softdouble inVal;
        int64_t out64;
        int32_t out32;
    } *values = (testvalues*)_values;

    for (int i = 0, maxi = sizeof(_values) / sizeof(_values[0]); i < maxi; i++)
    {
        EXPECT_EQ(values[i].out64, cvRound64(values[i].inVal));
        EXPECT_EQ(values[i].out64, saturate_cast<int64_t>(values[i].inVal));
        EXPECT_EQ((uint64_t)(values[i].out64), saturate_cast<uint64_t>(values[i].inVal));
        EXPECT_EQ(values[i].out32, cvRound(values[i].inVal));
        EXPECT_EQ(values[i].out32, saturate_cast<int32_t>(values[i].inVal));
        EXPECT_EQ((uint32_t)(values[i].out32), saturate_cast<uint32_t>(values[i].inVal));
    }
}

template<typename T>
static void checkRounding(T in, int outCeil, int outFloor)
{
    EXPECT_EQ(outCeil,cvCeil(in));
    EXPECT_EQ(outFloor,cvFloor(in));

    /* cvRound is not expected to be IEEE compliant. The implementation
       should round to one of the above. */
    EXPECT_TRUE((cvRound(in) == outCeil) || (cvRound(in) == outFloor));
}

TEST(Core_FastMath, InlineRoundingOps)
{
    struct
    {
        double in;
        int outCeil;
        int outFloor;
    } values[] =
    {
        // Values are chosen to convert to binary float 32/64 exactly
        { 1.0, 1, 1 },
        { 1.5, 2, 1 },
        { -1.5, -1, -2}
    };

    for (int i = 0, maxi = sizeof(values) / sizeof(values[0]); i < maxi; i++)
    {
        checkRounding<double>(values[i].in, values[i].outCeil, values[i].outFloor);
        checkRounding<float>((float)values[i].in, values[i].outCeil, values[i].outFloor);
    }
}

TEST(Core_FastMath, InlineNaN)
{
    EXPECT_EQ( cvIsNaN((float) NAN), 1);
    EXPECT_EQ( cvIsNaN((float) -NAN), 1);
    EXPECT_EQ( cvIsNaN(0.0f), 0);
    EXPECT_EQ( cvIsNaN((double) NAN), 1);
    EXPECT_EQ( cvIsNaN((double) -NAN), 1);
    EXPECT_EQ( cvIsNaN(0.0), 0);

    // Regression: check the +/-Inf cases
    Cv64suf suf;
    suf.u = 0x7FF0000000000000UL;
    EXPECT_EQ( cvIsNaN(suf.f), 0);
    suf.u = 0xFFF0000000000000UL;
    EXPECT_EQ( cvIsNaN(suf.f), 0);
}

TEST(Core_FastMath, InlineIsInf)
{
    // Assume HUGE_VAL is infinity. Strictly speaking, may not always be true.
    EXPECT_EQ( cvIsInf((float) HUGE_VAL), 1);
    EXPECT_EQ( cvIsInf((float) -HUGE_VAL), 1);
    EXPECT_EQ( cvIsInf(0.0f), 0);
    EXPECT_EQ( cvIsInf((double) HUGE_VAL), 1);
    EXPECT_EQ( cvIsInf((double) -HUGE_VAL), 1);
    EXPECT_EQ( cvIsInf(0.0), 0);

    // Regression: check the cases of 0x7FF00000xxxxxxxx
    Cv64suf suf;
    suf.u = 0x7FF0000000000001UL;
    EXPECT_EQ( cvIsInf(suf.f), 0);
    suf.u = 0x7FF0000012345678UL;
    EXPECT_EQ( cvIsInf(suf.f), 0);
}

TEST(Core_BFloat, CornerCases)
{
    float data[] = {0.f, -0.f, 1.f, -1.f, expf(1.f), FLT_MAX, -FLT_MAX,
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()
    };
    size_t n0 = sizeof(data)/sizeof(data[0]);
    for (size_t i = 0; i < n0; i++) {
        float x = data[i];
        Cv32suf suf0, suf1;
        suf0.f = x;
        uint16_t x0 = (uint16_t)(suf0.u >> 16);
        bfloat x1 = bfloat(x);
        suf0.u = x0 << 16;
        suf1.f = (float)x1;
        //printf("%zu. orig = %f, restored (old) = %f, restored (new) = %f\n", i, x, suf0.f, suf1.f);
        if (suf0.u != suf1.u) {
            EXPECT_LE(fabs(suf1.f - x), fabs(suf0.f - x));
        }
    }
}

TEST(Core_BFloat, convert)
{
    size_t N = 1 << 20;
    std::vector<float> bigdata(N);
    RNG& rng = theRNG();
    int m_max = 1 << 24;
    double m_scale = 1./m_max;
    for (size_t i = 0; i < N; i++) {
        double m = (m_max + rng.uniform(0, m_max))*m_scale;
        double e = pow(2., rng.uniform(-127, 127));
        double s = rng.uniform(0, 2)*2 - 1;
        float x = (float)(s*m*e);
        bigdata[i] = x;
    }

    double sum0 = 0, sqsum0 = 0, maxerr0 = 0, maxerr1 = 0, sum1 = 0, sqsum1 = 0;
    for (size_t i = 0; i < N; i++) {
        float x = bigdata[i];
        Cv32suf suf0, suf1;
        suf0.f = suf1.f = x;
        uint16_t x0 = (uint16_t)(suf0.u >> 16);
        bfloat x1 = bfloat(suf1.f);
        suf0.u = x0 << 16;
        suf1.f = (float)x1;
        double err0 = fabs(x - suf0.f)/(fabs(x) + DBL_EPSILON);
        double err1 = fabs(x - suf1.f)/(fabs(x) + DBL_EPSILON);
        maxerr0 = std::max(maxerr0, err0);
        maxerr1 = std::max(maxerr1, err1);
        sum0 += err0;
        sqsum0 += err0*err0;
        sum1 += err1;
        sqsum1 += err1*err1;
    }
    double mean0 = sum0/N;
    double stddev0 = sqrt(std::max(sqsum0 - N*mean0*mean0, 0.)/(N-1));
    double mean1 = sum1/N;
    double stddev1 = sqrt(std::max(sqsum1 - N*mean1*mean1, 0.)/(N-1));

    //if (maxerr1 > maxerr0 || mean1 > mean0 || stddev1 > stddev0)
    {
        printf("maxerr0 = %g, mean0 = %g, stddev0 = %g\nmaxerr1 = %g, mean1 = %g, stddev1 = %g\n",
               maxerr0, mean0, stddev0, maxerr1, mean1, stddev1);
    }

    EXPECT_LE(maxerr1, maxerr0);
    EXPECT_LE(mean1, mean0);
    EXPECT_LE(stddev1, stddev0);

#if CV_SIMD || CV_SIMD_SCALABLE
    //printf("checking vector part ...\n");
    std::vector<bfloat> bfdata(N);
    int vlanes = VTraits<v_float32>::vlanes();
    for (size_t i = 0; i < N; i += vlanes)
    {
        v_float32 x = vx_load(&bigdata[i]);
        v_pack_store(&bfdata[i], x);
    }

    int vecerr = 0;
    for (size_t i = 0; i < N; i++) {
        Cv32suf suf0, suf1;
        suf0.f = (float)bfloat(bigdata[i]);
        suf1.f = (float)bfdata[i];
        vecerr += suf0.u != suf1.u;
    }
    EXPECT_EQ(0, vecerr);
#endif
}

}} // namespace
/* End of file. */
