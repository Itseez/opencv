/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

void __wrap_printf_func(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char buffer[256];
    vsnprintf (buffer, sizeof(buffer), fmt, args);
    cvtest::TS::ptr()->printf(cvtest::TS::SUMMARY, buffer);
    va_end(args);
}

#define PRINT_TO_LOG __wrap_printf_func

#define SHOW_IMAGE
#undef SHOW_IMAGE

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ImageWarpBaseTest
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_ImageWarpBaseTest :
    public cvtest::BaseTest
{
public:
    enum { cell_size = 10 };

    CV_ImageWarpBaseTest();
    virtual ~CV_ImageWarpBaseTest();

    virtual void run(int);
protected:
    virtual void generate_test_data();

    virtual void run_func() = 0;
    virtual void run_reference_func() = 0;
    virtual float get_success_error_level(int _interpolation, int _depth) const;
    virtual void validate_results() const;
    virtual void prepare_test_data_for_reference_func();

    Size randSize(RNG& rng) const;

    String interpolation_to_string(int inter_type) const;

    int interpolation;
    Mat src;
    Mat dst;
    Mat reference_dst;
};

CV_ImageWarpBaseTest::CV_ImageWarpBaseTest() :
    BaseTest(), interpolation(-1),
    src(), dst(), reference_dst()
{
    test_case_count = 40;
    ts->set_failed_test_info(cvtest::TS::OK);
}

CV_ImageWarpBaseTest::~CV_ImageWarpBaseTest()
{
}

String CV_ImageWarpBaseTest::interpolation_to_string(int inter) const
{
    bool inverse = (inter & WARP_INVERSE_MAP) != 0;
    inter &= ~WARP_INVERSE_MAP;
    String str;

    if (inter == INTER_NEAREST)
        str = "INTER_NEAREST";
    else if (inter == INTER_LINEAR)
        str = "INTER_LINEAR";
    else if (inter == INTER_LINEAR_EXACT)
        str = "INTER_LINEAR_EXACT";
    else if (inter == INTER_AREA)
        str = "INTER_AREA";
    else if (inter == INTER_CUBIC)
        str = "INTER_CUBIC";
    else if (inter == INTER_LANCZOS4)
        str = "INTER_LANCZOS4";
    else if (inter == INTER_LANCZOS4 + 1)
        str = "INTER_AREA_FAST";

    if (inverse)
        str += " | WARP_INVERSE_MAP";

    return str.empty() ? "Unsupported/Unknown interpolation type" : str;
}

Size CV_ImageWarpBaseTest::randSize(RNG& rng) const
{
    Size size;
    size.width = static_cast<int>(std::exp(rng.uniform(1.0f, 7.0f)));
    size.height = static_cast<int>(std::exp(rng.uniform(1.0f, 7.0f)));

    return size;
}

void CV_ImageWarpBaseTest::generate_test_data()
{
    RNG& rng = ts->get_rng();

    // generating the src matrix structure
    Size ssize = randSize(rng), dsize;

    int depth = rng.uniform(0, CV_64F);
    while (depth == CV_8S || depth == CV_32S)
        depth = rng.uniform(0, CV_64F);

    int cn = rng.uniform(1, 5);

    src.create(ssize, CV_MAKE_TYPE(depth, cn));

    // generating the src matrix
    int x, y;
    if (cvtest::randInt(rng) % 2)
    {
        for (y = 0; y < ssize.height; y += cell_size)
            for (x = 0; x < ssize.width; x += cell_size)
                rectangle(src, Point(x, y), Point(x + std::min<int>(cell_size, ssize.width - x), y +
                        std::min<int>(cell_size, ssize.height - y)), Scalar::all((x + y) % 2 ? 255: 0), cv::FILLED);
    }
    else
    {
        src = Scalar::all(255);
        for (y = cell_size; y < src.rows; y += cell_size)
            line(src, Point2i(0, y), Point2i(src.cols, y), Scalar::all(0), 1);
        for (x = cell_size; x < src.cols; x += cell_size)
            line(src, Point2i(x, 0), Point2i(x, src.rows), Scalar::all(0), 1);
    }

    // generating an interpolation type
    interpolation = rng.uniform(0, cv::INTER_LANCZOS4 + 1);

    // generating the dst matrix structure
    double scale_x, scale_y;
    if (interpolation == INTER_AREA)
    {
        bool area_fast = rng.uniform(0., 1.) > 0.5;
        if (area_fast)
        {
            scale_x = rng.uniform(2, 5);
            scale_y = rng.uniform(2, 5);
        }
        else
        {
            scale_x = rng.uniform(1.0, 3.0);
            scale_y = rng.uniform(1.0, 3.0);
        }
    }
    else
    {
        scale_x = rng.uniform(0.4, 4.0);
        scale_y = rng.uniform(0.4, 4.0);
    }
    CV_Assert(scale_x > 0.0f && scale_y > 0.0f);

    dsize.width = saturate_cast<int>((ssize.width + scale_x - 1) / scale_x);
    dsize.height = saturate_cast<int>((ssize.height + scale_y - 1) / scale_y);

    dst = Mat::zeros(dsize, src.type());
    reference_dst = Mat::zeros(dst.size(), CV_MAKE_TYPE(CV_32F, dst.channels()));

    scale_x = src.cols / static_cast<double>(dst.cols);
    scale_y = src.rows / static_cast<double>(dst.rows);

    if (interpolation == INTER_AREA && (scale_x < 1.0 || scale_y < 1.0))
        interpolation = INTER_LINEAR;
}

void CV_ImageWarpBaseTest::run(int)
{
    for (int i = 0; i < test_case_count; ++i)
    {
        generate_test_data();
        run_func();
        run_reference_func();
        if (ts->get_err_code() < 0)
            break;
        validate_results();
        if (ts->get_err_code() < 0)
            break;
        ts->update_context(this, i, true);
    }
    ts->set_gtest_status();
}

float CV_ImageWarpBaseTest::get_success_error_level(int _interpolation, int) const
{
    if (_interpolation == INTER_CUBIC)
        return 1.0f;
    else if (_interpolation == INTER_LANCZOS4)
        return 1.0f;
    else if (_interpolation == INTER_NEAREST)
        return 255.0f;  // FIXIT: check is not reliable for Black/White (0/255) images
    else if (_interpolation == INTER_AREA)
        return 2.0f;
    else
        return 1.0f;
}

void CV_ImageWarpBaseTest::validate_results() const
{
    Mat _dst;
    dst.convertTo(_dst, reference_dst.depth());

    Size dsize = dst.size(), ssize = src.size();
    int cn = _dst.channels();
    dsize.width *= cn;
    float t = get_success_error_level(interpolation & INTER_MAX, dst.depth());

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const float* rD = reference_dst.ptr<float>(dy);
        const float* D = _dst.ptr<float>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
            if (fabs(rD[dx] - D[dx]) > t &&
//                fabs(rD[dx] - D[dx]) < 250.0f &&
                rD[dx] <= 255.0f && D[dx] <= 255.0f && rD[dx] >= 0.0f && D[dx] >= 0.0f)
            {
                PRINT_TO_LOG("\nNorm of the difference: %lf\n", cvtest::norm(reference_dst, _dst, NORM_INF));
                PRINT_TO_LOG("Error in (dx, dy): (%d, %d)\n", dx / cn + 1, dy + 1);
                PRINT_TO_LOG("Tuple (rD, D): (%f, %f)\n", rD[dx], D[dx]);
                PRINT_TO_LOG("Dsize: (%d, %d)\n", dsize.width / cn, dsize.height);
                PRINT_TO_LOG("Ssize: (%d, %d)\n", src.cols, src.rows);

                double scale_x = static_cast<double>(ssize.width) / dsize.width;
                double scale_y = static_cast<double>(ssize.height) / dsize.height;
                bool area_fast = interpolation == INTER_AREA &&
                    fabs(scale_x - cvRound(scale_x)) < FLT_EPSILON &&
                    fabs(scale_y - cvRound(scale_y)) < FLT_EPSILON;
                if (area_fast)
                {
                    scale_y = cvRound(scale_y);
                    scale_x = cvRound(scale_x);
                }

                PRINT_TO_LOG("Interpolation: %s\n", interpolation_to_string(area_fast ? INTER_LANCZOS4 + 1 : interpolation).c_str());
                PRINT_TO_LOG("Scale (x, y): (%lf, %lf)\n", scale_x, scale_y);
                PRINT_TO_LOG("Elemsize: %d\n", src.elemSize1());
                PRINT_TO_LOG("Channels: %d\n", cn);

#ifdef SHOW_IMAGE
                const std::string w1("OpenCV impl (run func)"), w2("Reference func"), w3("Src image"), w4("Diff");
                namedWindow(w1, cv::WINDOW_KEEPRATIO);
                namedWindow(w2, cv::WINDOW_KEEPRATIO);
                namedWindow(w3, cv::WINDOW_KEEPRATIO);
                namedWindow(w4, cv::WINDOW_KEEPRATIO);

                Mat diff;
                absdiff(reference_dst, _dst, diff);

                imshow(w1, dst);
                imshow(w2, reference_dst);
                imshow(w3, src);
                imshow(w4, diff);

                waitKey();
#endif

                const int radius = 3;
                int rmin = MAX(dy - radius, 0), rmax = MIN(dy + radius, dsize.height);
                int cmin = MAX(dx / cn - radius, 0), cmax = MIN(dx / cn + radius, dsize.width);

                std::cout << "opencv result:\n" << dst(Range(rmin, rmax), Range(cmin, cmax)) << std::endl;
                std::cout << "reference result:\n" << reference_dst(Range(rmin, rmax), Range(cmin, cmax)) << std::endl;

                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
    }
}

void CV_ImageWarpBaseTest::prepare_test_data_for_reference_func()
{
    if (src.depth() != CV_32F)
    {
        Mat tmp;
        src.convertTo(tmp, CV_32F);
        src = tmp;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resize
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_Resize_Test :
    public CV_ImageWarpBaseTest
{
public:
    CV_Resize_Test();
    virtual ~CV_Resize_Test();

protected:
    virtual void generate_test_data();

    virtual void run_func();
    virtual void run_reference_func();

private:
    double scale_x;
    double scale_y;
    bool area_fast;

    void resize_generic();
    void resize_area();
    double getWeight(double a, double b, int x);

    typedef std::vector<std::pair<int, double> > dim;
    void generate_buffer(double scale, dim& _dim);
    void resize_1d(const Mat& _src, Mat& _dst, int dy, const dim& _dim);
};

CV_Resize_Test::CV_Resize_Test() :
    CV_ImageWarpBaseTest(), scale_x(),
    scale_y(), area_fast(false)
{
}

CV_Resize_Test::~CV_Resize_Test()
{
}

namespace
{
    void interpolateLinear(float x, float* coeffs)
    {
        coeffs[0] = 1.f - x;
        coeffs[1] = x;
    }

    void interpolateCubic(float x, float* coeffs)
    {
        const float A = -0.75f;

        coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
        coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
        coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
        coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
    }

    void interpolateLanczos4(float x, float* coeffs)
    {
        static const double s45 = 0.70710678118654752440084436210485;
        static const double cs[][2]=
        {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

        if( x < FLT_EPSILON )
        {
            for( int i = 0; i < 8; i++ )
                coeffs[i] = 0;
            coeffs[3] = 1;
            return;
        }

        float sum = 0;
        double y0=-(x+3)*CV_PI*0.25, s0 = sin(y0), c0=cos(y0);
        for(int i = 0; i < 8; i++ )
        {
            double y = -(x+3-i)*CV_PI*0.25;
            coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
            sum += coeffs[i];
        }

        sum = 1.f/sum;
        for(int i = 0; i < 8; i++ )
            coeffs[i] *= sum;
    }

    typedef void (*interpolate_method)(float x, float* coeffs);
    interpolate_method inter_array[] = { &interpolateLinear, &interpolateCubic, &interpolateLanczos4 };
}

void CV_Resize_Test::generate_test_data()
{
    RNG& rng = ts->get_rng();

    // generating the src matrix structure
    Size ssize = randSize(rng), dsize;

    int depth = rng.uniform(0, CV_64F);
    while (depth == CV_8S || depth == CV_32S)
        depth = rng.uniform(0, CV_64F);

    int cn = rng.uniform(1, 4);

    src.create(ssize, CV_MAKE_TYPE(depth, cn));

    // generating the src matrix
    int x, y;
    if (cvtest::randInt(rng) % 2)
    {
        for (y = 0; y < ssize.height; y += cell_size)
            for (x = 0; x < ssize.width; x += cell_size)
                rectangle(src, Point(x, y), Point(x + std::min<int>(cell_size, ssize.width - x), y +
                        std::min<int>(cell_size, ssize.height - y)), Scalar::all((x + y) % 2 ? 255: 0), cv::FILLED);
    }
    else
    {
        src = Scalar::all(255);
        for (y = cell_size; y < src.rows; y += cell_size)
            line(src, Point2i(0, y), Point2i(src.cols, y), Scalar::all(0), 1);
        for (x = cell_size; x < src.cols; x += cell_size)
            line(src, Point2i(x, 0), Point2i(x, src.rows), Scalar::all(0), 1);
    }

    // generating an interpolation type
    interpolation = rng.uniform(0, cv::INTER_MAX - 1);

    // generating the dst matrix structure
    if (interpolation == INTER_AREA)
    {
        area_fast = rng.uniform(0., 1.) > 0.5;
        if (area_fast)
        {
            scale_x = rng.uniform(2, 5);
            scale_y = rng.uniform(2, 5);
        }
        else
        {
            scale_x = rng.uniform(1.0, 3.0);
            scale_y = rng.uniform(1.0, 3.0);
        }
    }
    else
    {
        scale_x = rng.uniform(0.4, 4.0);
        scale_y = rng.uniform(0.4, 4.0);
    }
    CV_Assert(scale_x > 0.0f && scale_y > 0.0f);

    dsize.width = saturate_cast<int>((ssize.width + scale_x - 1) / scale_x);
    dsize.height = saturate_cast<int>((ssize.height + scale_y - 1) / scale_y);

    dst = Mat::zeros(dsize, src.type());
    reference_dst = Mat::zeros(dst.size(), CV_MAKE_TYPE(CV_32F, dst.channels()));

    scale_x = src.cols / static_cast<double>(dst.cols);
    scale_y = src.rows / static_cast<double>(dst.rows);

    if (interpolation == INTER_AREA && (scale_x < 1.0 || scale_y < 1.0))
        interpolation = INTER_LINEAR_EXACT;
    if (interpolation == INTER_LINEAR_EXACT && (depth == CV_32F || depth == CV_64F))
        interpolation = INTER_LINEAR;

    area_fast = interpolation == INTER_AREA &&
        fabs(scale_x - cvRound(scale_x)) < FLT_EPSILON &&
        fabs(scale_y - cvRound(scale_y)) < FLT_EPSILON;
    if (area_fast)
    {
        scale_x = cvRound(scale_x);
        scale_y = cvRound(scale_y);
    }
}

void CV_Resize_Test::run_func()
{
    cv::resize(src, dst, dst.size(), 0, 0, interpolation);
}

void CV_Resize_Test::run_reference_func()
{
    CV_ImageWarpBaseTest::prepare_test_data_for_reference_func();

    if (interpolation == INTER_AREA)
        resize_area();
    else
        resize_generic();
}

double CV_Resize_Test::getWeight(double a, double b, int x)
{
    double w = std::min(static_cast<double>(x + 1), b) - std::max(static_cast<double>(x), a);
    CV_Assert(w >= 0);
    return w;
}

void CV_Resize_Test::resize_area()
{
    Size ssize = src.size(), dsize = reference_dst.size();
    CV_Assert(!ssize.empty() && !dsize.empty());
    int cn = src.channels();

    CV_Assert(scale_x >= 1.0 && scale_y >= 1.0);

    double fsy0 = 0, fsy1 = scale_y;
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        float* yD = reference_dst.ptr<float>(dy);
        int isy0 = cvFloor(fsy0), isy1 = std::min(cvFloor(fsy1), ssize.height - 1);
        CV_Assert(isy1 <= ssize.height && isy0 < ssize.height);

        double fsx0 = 0, fsx1 = scale_x;

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            float* xyD = yD + cn * dx;
            int isx0 = cvFloor(fsx0), isx1 = std::min(ssize.width - 1, cvFloor(fsx1));

            CV_Assert(isx1 <= ssize.width);
            CV_Assert(isx0 < ssize.width);

            // for each pixel of dst
            for (int r = 0; r < cn; ++r)
            {
                xyD[r] = 0.0f;
                double area = 0.0;
                for (int sy = isy0; sy <= isy1; ++sy)
                {
                    const float* yS = src.ptr<float>(sy);
                    for (int sx = isx0; sx <= isx1; ++sx)
                    {
                        double wy = getWeight(fsy0, fsy1, sy);
                        double wx = getWeight(fsx0, fsx1, sx);
                        double w = wx * wy;
                        xyD[r] += static_cast<float>(yS[sx * cn + r] * w);
                        area += w;
                    }
                }

                CV_Assert(area != 0);
                // norming pixel
                xyD[r] = static_cast<float>(xyD[r] / area);
            }
            fsx1 = std::min((fsx0 = fsx1) + scale_x, static_cast<double>(ssize.width));
        }
        fsy1 = std::min((fsy0 = fsy1) + scale_y, static_cast<double>(ssize.height));
    }
}

// for interpolation type : INTER_LINEAR, INTER_LINEAR_EXACT, INTER_CUBIC, INTER_LANCZOS4
void CV_Resize_Test::resize_1d(const Mat& _src, Mat& _dst, int dy, const dim& _dim)
{
    Size dsize = _dst.size();
    int cn = _dst.channels();
    float* yD = _dst.ptr<float>(dy);

    if (interpolation == INTER_NEAREST)
    {
        const float* yS = _src.ptr<float>(dy);
        for (int dx = 0; dx < dsize.width; ++dx)
        {
            int isx = _dim[dx].first;
            const float* xyS = yS + isx * cn;
            float* xyD = yD + dx * cn;

            for (int r = 0; r < cn; ++r)
                xyD[r] = xyS[r];
        }
    }
    else if (interpolation == INTER_LINEAR || interpolation == INTER_LINEAR_EXACT || interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4)
    {
        interpolate_method inter_func = inter_array[interpolation - (interpolation == INTER_LANCZOS4 ? 2 : interpolation == INTER_LINEAR_EXACT ? 5 : 1)];
        size_t elemsize = _src.elemSize();

        int ofs = 0, ksize = 2;
        if (interpolation == INTER_CUBIC)
            ofs = 1, ksize = 4;
        else if (interpolation == INTER_LANCZOS4)
            ofs = 3, ksize = 8;

        Mat _extended_src_row(1, _src.cols + ksize * 2, _src.type());
        const uchar* srow = _src.ptr(dy);
        memcpy(_extended_src_row.ptr() + elemsize * ksize, srow, _src.step);
        for (int k = 0; k < ksize; ++k)
        {
            memcpy(_extended_src_row.ptr() + k * elemsize, srow, elemsize);
            memcpy(_extended_src_row.ptr() + (ksize + k) * elemsize + _src.step, srow + _src.step - elemsize, elemsize);
        }

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            int isx = _dim[dx].first;
            double fsx = _dim[dx].second;

            float *xyD = yD + dx * cn;
            const float* xyS = _extended_src_row.ptr<float>(0) + (isx + ksize - ofs) * cn;

            float w[8];
            inter_func(static_cast<float>(fsx), w);

            for (int r = 0; r < cn; ++r)
            {
                xyD[r] = 0;
                for (int k = 0; k < ksize; ++k)
                    xyD[r] += w[k] * xyS[k * cn + r];
            }
        }
    }
    else
        CV_Assert(0);
}

void CV_Resize_Test::generate_buffer(double scale, dim& _dim)
{
    size_t length = _dim.size();
    for (size_t dx = 0; dx < length; ++dx)
    {
        double fsx = scale * (dx + 0.5) - 0.5;
        int isx = cvFloor(fsx);
        _dim[dx] = std::make_pair(isx, fsx - isx);
    }
}

void CV_Resize_Test::resize_generic()
{
    Size dsize = reference_dst.size(), ssize = src.size();
    CV_Assert(!dsize.empty() && !ssize.empty());

    dim dims[] = { dim(dsize.width), dim(dsize.height) };
    if (interpolation == INTER_NEAREST)
    {
        for (int dx = 0; dx < dsize.width; ++dx)
            dims[0][dx].first = std::min(cvFloor(dx * scale_x), ssize.width - 1);
        for (int dy = 0; dy < dsize.height; ++dy)
            dims[1][dy].first = std::min(cvFloor(dy * scale_y), ssize.height - 1);
    }
    else
    {
        generate_buffer(scale_x, dims[0]);
        generate_buffer(scale_y, dims[1]);
    }

    Mat tmp(ssize.height, dsize.width, reference_dst.type());
    for (int dy = 0; dy < tmp.rows; ++dy)
        resize_1d(src, tmp, dy, dims[0]);

    cv::Mat tmp_t(tmp.cols, tmp.rows, tmp.type());
    cvtest::transpose(tmp, tmp_t);
    cv::Mat reference_dst_t(reference_dst.cols, reference_dst.rows, reference_dst.type());
    cvtest::transpose(reference_dst, reference_dst_t);

    for (int dy = 0; dy < tmp_t.rows; ++dy)
        resize_1d(tmp_t, reference_dst_t, dy, dims[1]);

    cvtest::transpose(reference_dst_t, reference_dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// remap
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_Remap_Test :
    public CV_ImageWarpBaseTest
{
public:
    CV_Remap_Test();

    virtual ~CV_Remap_Test();

private:
    typedef void (CV_Remap_Test::*remap_func)(const Mat&, Mat&);

protected:
    virtual void generate_test_data();
    virtual void prepare_test_data_for_reference_func();

    virtual void run_func();
    virtual void run_reference_func();

    template<typename T>
    void new_nearest_c1(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                       const T *bval, int borderType_x, int borderType_y);
    template<typename T>
    void new_nearest_c3(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                       const T *bval, int borderType_x, int borderType_y);
    template<typename T>
    void new_nearest_c4(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                       const T *bval, int borderType_x, int borderType_y);

    template<typename T>
    void new_linear_c1(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                       const T *bval, int borderType_x, int borderType_y);
    template<typename T>
    void new_linear_c3(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                       const T *bval, int borderType_x, int borderType_y);
    template<typename T>
    void new_linear_c4(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                       const T *bval, int borderType_x, int borderType_y);

    Mat mapx, mapy;
    int borderType;
    Scalar borderValue;

    remap_func funcs[2];

private:
    template <typename T> void new_remap(const Mat&, Mat&, int);
    void remap_nearest(const Mat&, Mat&);
    void remap_generic(const Mat&, Mat&);

    void convert_maps();
    const char* borderType_to_string() const;
    virtual void validate_results() const;
};

CV_Remap_Test::CV_Remap_Test() :
    CV_ImageWarpBaseTest(), borderType(-1)
{
    funcs[0] = &CV_Remap_Test::remap_nearest;
    funcs[1] = &CV_Remap_Test::remap_generic;
}

CV_Remap_Test::~CV_Remap_Test()
{
}

void CV_Remap_Test::generate_test_data()
{
    CV_ImageWarpBaseTest::generate_test_data();

    RNG& rng = ts->get_rng();
    borderType = rng.uniform(1, BORDER_WRAP);
    borderValue = Scalar::all(rng.uniform(0, 255));

    // generating the mapx, mapy matrices
    static const int mapx_types[] = { CV_16SC2, CV_32FC1, CV_32FC2 };
    mapx.create(dst.size(), mapx_types[rng.uniform(0, sizeof(mapx_types) / sizeof(int))]);
    mapy.release();

    const int n = std::min(std::min(src.cols, src.rows) / 10 + 1, 2);
    float _n = 0; //static_cast<float>(-n);

    switch (mapx.type())
    {
        case CV_16SC2:
        {
            MatIterator_<Vec2s> begin_x = mapx.begin<Vec2s>(), end_x = mapx.end<Vec2s>();
            for ( ; begin_x != end_x; ++begin_x)
            {
                (*begin_x)[0] = static_cast<short>(rng.uniform(static_cast<int>(_n), std::max(src.cols + n - 1, 0)));
                (*begin_x)[1] = static_cast<short>(rng.uniform(static_cast<int>(_n), std::max(src.rows + n - 1, 0)));
            }

            if (interpolation != INTER_NEAREST)
            {
                static const int mapy_types[] = { CV_16UC1, CV_16SC1 };
                mapy.create(dst.size(), mapy_types[rng.uniform(0, sizeof(mapy_types) / sizeof(int))]);

                switch (mapy.type())
                {
                    case CV_16UC1:
                    {
                        MatIterator_<ushort> begin_y = mapy.begin<ushort>(), end_y = mapy.end<ushort>();
                        for ( ; begin_y != end_y; ++begin_y)
                            *begin_y = static_cast<ushort>(rng.uniform(0, 1024));
                    }
                    break;

                    case CV_16SC1:
                    {
                        MatIterator_<short> begin_y = mapy.begin<short>(), end_y = mapy.end<short>();
                        for ( ; begin_y != end_y; ++begin_y)
                            *begin_y = static_cast<short>(rng.uniform(0, 1024));
                    }
                    break;
                }
            }
        }
        break;

        case CV_32FC1:
        {
            mapy.create(dst.size(), CV_32FC1);
            float fscols = static_cast<float>(std::max(src.cols - 1 + n, 0)),
                    fsrows = static_cast<float>(std::max(src.rows - 1 + n, 0));
            MatIterator_<float> begin_x = mapx.begin<float>(), end_x = mapx.end<float>();
            MatIterator_<float> begin_y = mapy.begin<float>();
            for ( ; begin_x != end_x; ++begin_x, ++begin_y)
            {
                *begin_x = rng.uniform(_n, fscols);
                *begin_y = rng.uniform(_n, fsrows);
            }
        }
        break;

        case CV_32FC2:
        {
            float fscols = static_cast<float>(std::max(src.cols - 1 + n, 0)),
                    fsrows = static_cast<float>(std::max(src.rows - 1 + n, 0));
            int width = mapx.cols << 1;

            for (int y = 0; y < mapx.rows; ++y)
            {
                float * ptr = mapx.ptr<float>(y);

                for (int x = 0; x < width; x += 2)
                {
                    ptr[x] = rng.uniform(_n, fscols);
                    ptr[x + 1] = rng.uniform(_n, fsrows);
                }
            }
        }
        break;

        default:
            CV_Assert(0);
        break;
    }
}

void CV_Remap_Test::run_func()
{
    remap(src, dst, mapx, mapy, interpolation, borderType, borderValue);
}

void CV_Remap_Test::convert_maps()
{
    if (mapx.type() != CV_16SC2)
        convertMaps(mapx.clone(), mapy.clone(), mapx, mapy, CV_16SC2, interpolation == INTER_NEAREST);
    else if (interpolation != INTER_NEAREST)
        if (mapy.type() != CV_16UC1)
            mapy.clone().convertTo(mapy, CV_16UC1);

    if (interpolation == INTER_NEAREST)
        mapy = Mat();
    CV_Assert(((interpolation == INTER_NEAREST && mapy.empty()) || mapy.type() == CV_16UC1 ||
               mapy.type() == CV_16SC1) && mapx.type() == CV_16SC2);
}

const char* CV_Remap_Test::borderType_to_string() const
{
    if (borderType == BORDER_CONSTANT)
        return "BORDER_CONSTANT";
    if (borderType == BORDER_REPLICATE)
        return "BORDER_REPLICATE";
    if (borderType == BORDER_REFLECT)
        return "BORDER_REFLECT";
    if (borderType == BORDER_WRAP)
        return "BORDER_WRAP";
    if (borderType == BORDER_REFLECT_101)
        return "BORDER_REFLECT_101";
    return "Unsupported/Unknown border type";
}

void CV_Remap_Test::prepare_test_data_for_reference_func()
{
    CV_ImageWarpBaseTest::prepare_test_data_for_reference_func();
    convert_maps();
}

void CV_Remap_Test::run_reference_func()
{
    if (interpolation == INTER_AREA)
        interpolation = INTER_LINEAR;

    if ((interpolation == INTER_LINEAR) && mapx.depth() == CV_32F) {
        int src_depth = src.depth(), src_channels = src.channels();
        Mat tmp = Mat::zeros(dst.size(), dst.type());
        if (src_depth == CV_8U && (src_channels == 1 || src_channels == 3 || src_channels == 4)) {
            new_remap<uint8_t>(src, tmp, interpolation);
            tmp.convertTo(reference_dst, reference_dst.depth());
            return;
        } else if (src_depth == CV_16U && (src_channels == 1 || src_channels == 3 || src_channels == 4)) {
            new_remap<uint16_t>(src, tmp, interpolation);
            tmp.convertTo(reference_dst, reference_dst.depth());
            return;
        } else if (src_depth == CV_32F && (src_channels == 1 || src_channels == 3 || src_channels == 4)) {
            new_remap<float>(src, tmp, interpolation);
            tmp.convertTo(reference_dst, reference_dst.depth());
            return;
        }
    }

    prepare_test_data_for_reference_func();

    int index = interpolation == INTER_NEAREST ? 0 : 1;
    (this->*funcs[index])(src, reference_dst);
}

#define WARP_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn, dy, dx) \
    if ((((unsigned)(ix + dx) < (unsigned)srccols) & ((unsigned)(iy + dy) < (unsigned)srcrows)) != 0) { \
        size_t ofs = dy*srcstep + dx*cn; \
        for (int ci = 0; ci < cn; ci++) { pxy[2*dy*cn+dx*cn+ci] = srcptr[ofs+ci];} \
    } else if (borderType == BORDER_CONSTANT) { \
        for (int ci = 0; ci < cn; ci++) { pxy[2*dy*cn+dx*cn+ci] = bval[ci];} \
    } else if (borderType == BORDER_TRANSPARENT) { \
        for (int ci = 0; ci < cn; ci++) { pxy[2*dy*cn+dx*cn+ci] = dstptr[x*cn+ci];} \
    } else { \
        int ix_ = borderInterpolate(ix + dx, srccols, borderType_x); \
        int iy_ = borderInterpolate(iy + dy, srcrows, borderType_y); \
        size_t glob_ofs = iy_*srcstep + ix_*cn; \
        for (int ci = 0; ci < cn; ci++) { pxy[2*dy*cn+dx*cn+ci] = srcptr_[glob_ofs+ci];} \
    }
#define WARP_NEAREST_SHUFFLE_FETCH_PIXEL_IN_RANGE(cn) \
    for (int ci = 0; ci < cn; ci++) { \
        pxy[ci] = srcptr[ci]; \
    }
#define WARP_LINEAR_SHUFFLE_FETCH_PIXEL_IN_RANGE(cn) \
    for (int ci = 0; ci < cn; ci++) { \
        pxy[ci] = srcptr[ci]; \
        pxy[ci+cn] = srcptr[ci+cn]; \
        pxy[ci+cn*2] = srcptr[srcstep+ci]; \
        pxy[ci+cn*3] = srcptr[srcstep+ci+cn]; \
    }
#define WARP_NEAREST_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn) \
    WARP_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn, 0, 0);
#define WARP_LINEAR_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn) \
    WARP_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn, 0, 0); \
    WARP_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn, 0, 1); \
    WARP_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn, 1, 0); \
    WARP_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn, 1, 1);
#define WARP_SHUFFLE(inter, cn) \
    if ((((unsigned)ix < (unsigned)(srccols-1)) & \
        ((unsigned)iy < (unsigned)(srcrows-1))) != 0) { \
        WARP_##inter##_SHUFFLE_FETCH_PIXEL_IN_RANGE(cn) \
    } else { \
        if ((borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) && \
            (((unsigned)(ix+1) >= (unsigned)(srccols+1))| \
            ((unsigned)(iy+1) >= (unsigned)(srcrows+1))) != 0) { \
            if (borderType == BORDER_CONSTANT) { \
                for (int ci = 0; ci < cn; ci++) { dstptr[x*cn+ci] = bval[ci]; } \
            } \
            return; \
        } \
        WARP_##inter##_SHUFFLE_FETCH_PIXEL_OUT_RANGE(cn) \
    }

template<typename T>
void CV_Remap_Test::new_nearest_c1(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                    const T *bval, int borderType_x, int borderType_y) {
    int ix = (int)roundf(sx), iy = (int)roundf(sy);

    T pxy[1];
    const T *srcptr = srcptr_ + srcstep*iy + ix;
    WARP_SHUFFLE(NEAREST, 1);

    dstptr[x+0] = saturate_cast<T>(pxy[0]);
}
template<typename T>
void CV_Remap_Test::new_nearest_c3(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                    const T *bval, int borderType_x, int borderType_y) {
    int ix = (int)roundf(sx), iy = (int)roundf(sy);

    T pxy[3];
    const T *srcptr = srcptr_ + srcstep*iy + ix*3;
    WARP_SHUFFLE(NEAREST, 3);

    dstptr[x*3+0] = saturate_cast<T>(pxy[0]);
    dstptr[x*3+1] = saturate_cast<T>(pxy[1]);
    dstptr[x*3+2] = saturate_cast<T>(pxy[2]);
}
template<typename T>
void CV_Remap_Test::new_nearest_c4(int x, float sx, float sy, const T *srcptr_, T *dstptr, int srccols, int srcrows, size_t srcstep,
                    const T *bval, int borderType_x, int borderType_y) {
    int ix = (int)roundf(sx), iy = (int)roundf(sy);

    T pxy[4];
    const T *srcptr = srcptr_ + srcstep*iy + ix*4;
    WARP_SHUFFLE(NEAREST, 4);

    dstptr[x*4+0] = saturate_cast<T>(pxy[0]);
    dstptr[x*4+1] = saturate_cast<T>(pxy[1]);
    dstptr[x*4+2] = saturate_cast<T>(pxy[2]);
    dstptr[x*4+3] = saturate_cast<T>(pxy[3]);
}

template<typename T>
static inline void warp_linear_calc(int cn, const T *pxy, T *dst, float sx, float sy)
{
    for (int ci = 0; ci < cn; ci++) {
        float p00 = pxy[ci];
        float p01 = pxy[ci+cn];
        float p10 = pxy[ci+cn*2];
        float p11 = pxy[ci+cn*3];
        float v0 = p00 + sx*(p01 - p00);
        float v1 = p10 + sx*(p11 - p10);
        v0 += sy*(v1 - v0);
        dst[ci] = saturate_cast<T>(v0);
    }
}
template<typename T>
void CV_Remap_Test::new_linear_c1(int x, float sx, float sy, const T *srcptr_, T *dstptr,
                                  int srccols, int srcrows, size_t srcstep,
                                  const T *bval, int borderType_x, int borderType_y)
{
    int ix = (int)floorf(sx), iy = (int)floorf(sy);
    sx -= ix; sy -= iy;

    T pxy[4];
    const T *srcptr = srcptr_ + srcstep*iy + ix;

    WARP_SHUFFLE(LINEAR, 1);

    warp_linear_calc(1, pxy, dstptr+x, sx, sy);
}
template<typename T>
void CV_Remap_Test::new_linear_c3(int x, float sx, float sy, const T *srcptr_, T *dstptr,
                                  int srccols, int srcrows, size_t srcstep,
                                  const T *bval, int borderType_x, int borderType_y)
{
    int ix = (int)floorf(sx), iy = (int)floorf(sy);
    sx -= ix; sy -= iy;

    T pxy[12];
    const T *srcptr = srcptr_ + srcstep*iy + ix*3;

    WARP_SHUFFLE(LINEAR, 3);

    warp_linear_calc(3, pxy, dstptr+x*3, sx, sy);
}
template<typename T>
void CV_Remap_Test::new_linear_c4(int x, float sx, float sy, const T *srcptr_, T *dstptr,
                                  int srccols, int srcrows, size_t srcstep,
                                  const T *bval, int borderType_x, int borderType_y)
{
    int ix = (int)floorf(sx), iy = (int)floorf(sy);
    sx -= ix; sy -= iy;

    T pxy[16];
    const T *srcptr = srcptr_ + srcstep*iy + ix*4;

    WARP_SHUFFLE(LINEAR, 4);

    warp_linear_calc(4, pxy, dstptr+x*4, sx, sy);
}

template <typename T>
void CV_Remap_Test::new_remap(const Mat &_src, Mat &_dst, int inter) {
    CV_UNUSED(inter);
    int src_channels = _src.channels();
    CV_CheckTrue(_src.channels() == 1 || _src.channels() == 3 || _src.channels() == 4, "");
    CV_CheckTrue(mapx.depth() == CV_32F, "");
    CV_CheckTrue(mapx.channels() == 1 || mapx.channels() == 2, "");

    auto *srcptr_ = _src.ptr<const T>();
    auto *dstptr_ = _dst.ptr<T>();
    size_t srcstep = _src.step/sizeof(T), dststep = _dst.step/sizeof(T);
    int srccols = _src.cols, srcrows = _src.rows;
    int dstcols = _dst.cols, dstrows = _dst.rows;

    T bval[] = {
        saturate_cast<T>(borderValue[0]),
        saturate_cast<T>(borderValue[1]),
        saturate_cast<T>(borderValue[2]),
        saturate_cast<T>(borderValue[3]),
    };

    int borderType_x = borderType != BORDER_CONSTANT &&
                       borderType != BORDER_TRANSPARENT &&
                       srccols <= 1 ? BORDER_REPLICATE : borderType;
    int borderType_y = borderType != BORDER_CONSTANT &&
                       borderType != BORDER_TRANSPARENT &&
                       srcrows <= 1 ? BORDER_REPLICATE : borderType;

    const float *mapx_data = mapx.ptr<const float>(),
                *mapy_data = mapy.ptr<const float>();
    int mapx_channels = mapx.channels();
    for (int y = 0; y < dstrows; y++) {
        T* dstptr = dstptr_ + y*dststep;
        for (int x = 0; x < dstcols; x++) {
            float sx, sy;
            size_t offset = y * dstcols + x;
            if (mapx_channels == 1) {
                sx = mapx_data[offset];
                sy = mapy_data[offset];
            } else { // mapx_channels == 2
                sx = mapx_data[2*offset];
                sy = mapx_data[2*offset+1];
            }

            if (src_channels == 3) {
                new_linear_c3(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            } else if (src_channels == 4) {
                new_linear_c4(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            } else {
                new_linear_c1(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            }
        }
    }
}

void CV_Remap_Test::remap_nearest(const Mat& _src, Mat& _dst)
{
    CV_Assert(_src.depth() == CV_32F && _dst.type() == _src.type());
    CV_Assert(mapx.type() == CV_16SC2 && mapy.empty());

    Size ssize = _src.size(), dsize = _dst.size();
    CV_Assert(!ssize.empty() && !dsize.empty());
    int cn = _src.channels();

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const short* yM = mapx.ptr<short>(dy);
        float* yD = _dst.ptr<float>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            float* xyD = yD + cn * dx;
            int sx = yM[dx * 2], sy = yM[dx * 2 + 1];

            if (sx >= 0 && sx < ssize.width && sy >= 0 && sy < ssize.height)
            {
                const float *xyS = _src.ptr<float>(sy) + sx * cn;

                for (int r = 0; r < cn; ++r)
                    xyD[r] = xyS[r];
            }
            else if (borderType != BORDER_TRANSPARENT)
            {
                if (borderType == BORDER_CONSTANT)
                    for (int r = 0; r < cn; ++r)
                        xyD[r] = saturate_cast<float>(borderValue[r]);
                else
                {
                    sx = borderInterpolate(sx, ssize.width, borderType);
                    sy = borderInterpolate(sy, ssize.height, borderType);
                    CV_Assert(sx >= 0 && sy >= 0 && sx < ssize.width && sy < ssize.height);

                    const float *xyS = _src.ptr<float>(sy) + sx * cn;

                    for (int r = 0; r < cn; ++r)
                        xyD[r] = xyS[r];
                }
            }
        }
    }
}

void CV_Remap_Test::remap_generic(const Mat& _src, Mat& _dst)
{
    CV_Assert(mapx.type() == CV_16SC2 && mapy.type() == CV_16UC1);

    int ksize = 2;
    if (interpolation == INTER_CUBIC)
        ksize = 4;
    else if (interpolation == INTER_LANCZOS4)
        ksize = 8;
    else if (interpolation != INTER_LINEAR)
        CV_Assert(0);
    int ofs = (ksize / 2) - 1;

    CV_Assert(_src.depth() == CV_32F && _dst.type() == _src.type());
    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels(), width1 = std::max(ssize.width - ksize + 1, 0),
        height1 = std::max(ssize.height - ksize + 1, 0);

    float ix[8], w[16];
    interpolate_method inter_func = inter_array[interpolation - (interpolation == INTER_LANCZOS4 ? 2 : 1)];

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const short* yMx = mapx.ptr<short>(dy);
        const ushort* yMy = mapy.ptr<ushort>(dy);

        float* yD = _dst.ptr<float>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            float* xyD = yD + dx * cn;
            float sx = yMx[dx * 2], sy = yMx[dx * 2 + 1];
            int isx = cvFloor(sx), isy = cvFloor(sy);

            inter_func((yMy[dx] & (INTER_TAB_SIZE - 1)) / static_cast<float>(INTER_TAB_SIZE), w);
            inter_func(((yMy[dx] >> INTER_BITS) & (INTER_TAB_SIZE - 1)) / static_cast<float>(INTER_TAB_SIZE), w + ksize);

            isx -= ofs;
            isy -= ofs;

            if (isx >= 0 && isx < width1 && isy >= 0 && isy < height1)
            {
                for (int r = 0; r < cn; ++r)
                {
                    for (int y = 0; y < ksize; ++y)
                    {
                        const float* xyS = _src.ptr<float>(isy + y) + isx * cn;

                        ix[y] = 0;
                        for (int i = 0; i < ksize; ++i)
                            ix[y] += w[i] * xyS[i * cn + r];
                    }
                    xyD[r] = 0;
                    for (int i = 0; i < ksize; ++i)
                        xyD[r] += w[ksize + i] * ix[i];
                }
            }
            else if (borderType != BORDER_TRANSPARENT)
            {
                int ar_x[8], ar_y[8];

                for (int k = 0; k < ksize; k++)
                {
                    ar_x[k] = borderInterpolate(isx + k, ssize.width, borderType) * cn;
                    ar_y[k] = borderInterpolate(isy + k, ssize.height, borderType);
                }

                for (int r = 0; r < cn; r++)
                {
                    xyD[r] = 0;
                    for (int i = 0; i < ksize; ++i)
                    {
                        ix[i] = 0;
                        if (ar_y[i] >= 0)
                        {
                            const float* yS = _src.ptr<float>(ar_y[i]);
                            for (int j = 0; j < ksize; ++j)
                                ix[i] += saturate_cast<float>((ar_x[j] >= 0 ? yS[ar_x[j] + r] : borderValue[r]) * w[j]);
                        }
                        else
                            for (int j = 0; j < ksize; ++j)
                                ix[i] += saturate_cast<float>(borderValue[r] * w[j]);
                    }
                    for (int i = 0; i < ksize; ++i)
                        xyD[r] += saturate_cast<float>(w[ksize + i] * ix[i]);
                }
            }
        }
    }
}

void CV_Remap_Test::validate_results() const
{
    CV_ImageWarpBaseTest::validate_results();
    if (cvtest::TS::ptr()->get_err_code() == cvtest::TS::FAIL_BAD_ACCURACY)
    {
        PRINT_TO_LOG("BorderType: %s\n", borderType_to_string());
        PRINT_TO_LOG("BorderValue: (%f, %f, %f, %f)\n",
                     borderValue[0], borderValue[1], borderValue[2], borderValue[3]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// warpAffine
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_WarpAffine_Test :
    public CV_Remap_Test
{
public:
    CV_WarpAffine_Test();

    virtual ~CV_WarpAffine_Test();

protected:
    virtual void generate_test_data();
    virtual float get_success_error_level(int _interpolation, int _depth) const;

    virtual void run_func();
    virtual void run_reference_func();

    Mat M;
private:
    void warpAffine(const Mat&, Mat&);

    template<typename T>
    void new_warpAffine(const Mat&, Mat&, const Mat&, int);
};

CV_WarpAffine_Test::CV_WarpAffine_Test() :
    CV_Remap_Test()
{
}

CV_WarpAffine_Test::~CV_WarpAffine_Test()
{
}

void CV_WarpAffine_Test::generate_test_data()
{
    CV_Remap_Test::generate_test_data();

    RNG& rng = ts->get_rng();

    // generating the M 2x3 matrix
    static const int depths[] = { CV_32FC1, CV_64FC1 };

    // generating 2d matrix
    M = getRotationMatrix2D(Point2f(src.cols / 2.f, src.rows / 2.f),
        rng.uniform(-180.f, 180.f), rng.uniform(0.4f, 2.0f));
    int depth = depths[rng.uniform(0, sizeof(depths) / sizeof(depths[0]))];
    if (M.depth() != depth)
    {
        Mat tmp;
        M.convertTo(tmp, depth);
        M = tmp;
    }

    // warp_matrix is inverse
    if (rng.uniform(0., 1.) > 0)
        interpolation |= cv::WARP_INVERSE_MAP;
}

void CV_WarpAffine_Test::run_func()
{
    cv::warpAffine(src, dst, M, dst.size(), interpolation, borderType, borderValue);
}

float CV_WarpAffine_Test::get_success_error_level(int _interpolation, int _depth) const
{
    return _depth == CV_8U ? 0.f : CV_ImageWarpBaseTest::get_success_error_level(_interpolation, _depth);
}

void CV_WarpAffine_Test::run_reference_func()
{
    Mat tmp = Mat::zeros(dst.size(), dst.type());
    warpAffine(src, tmp);
    tmp.convertTo(reference_dst, reference_dst.depth());
}

template<typename T>
void CV_WarpAffine_Test::new_warpAffine(const Mat &_src, Mat &_dst, const Mat &tM, int inter)
{
    CV_UNUSED(inter);
    int num_channels = _dst.channels();
    CV_CheckTrue(num_channels == 1 || num_channels == 3 || num_channels == 4, "");

    auto *srcptr_ = _src.ptr<const T>();
    auto *dstptr_ = _dst.ptr<T>();
    size_t srcstep = _src.step/sizeof(T), dststep = _dst.step/sizeof(T);
    int srccols = _src.cols, srcrows = _src.rows;
    int dstcols = _dst.cols, dstrows = _dst.rows;

    Mat ttM;
    tM.convertTo(ttM, CV_32F);
    auto *_M = ttM.ptr<const float>();

    T bval[] = {
        saturate_cast<T>(borderValue[0]),
        saturate_cast<T>(borderValue[1]),
        saturate_cast<T>(borderValue[2]),
        saturate_cast<T>(borderValue[3]),
    };

    int borderType_x = borderType != BORDER_CONSTANT &&
                       borderType != BORDER_TRANSPARENT &&
                       srccols <= 1 ? BORDER_REPLICATE : borderType;
    int borderType_y = borderType != BORDER_CONSTANT &&
                       borderType != BORDER_TRANSPARENT &&
                       srcrows <= 1 ? BORDER_REPLICATE : borderType;

    for (int y = 0; y < dstrows; y++) {
        T* dstptr = dstptr_ + y*dststep;
        for (int x = 0; x < dstcols; x++) {
            float sx = x*_M[0] + y*_M[1] + _M[2];
            float sy = x*_M[3] + y*_M[4] + _M[5];

            if (num_channels == 3) {
                new_linear_c3(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            } else if (num_channels == 4) {
                new_linear_c4(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            } else {
                new_linear_c1(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            }
        }
    }
}

void CV_WarpAffine_Test::warpAffine(const Mat& _src, Mat& _dst)
{
    Size dsize = _dst.size();

    CV_Assert(!_src.empty());
    CV_Assert(!dsize.empty());
    CV_Assert(_src.type() == _dst.type());

    Mat tM;
    M.convertTo(tM, CV_64F);

    int inter = interpolation & INTER_MAX;
    if (inter == INTER_AREA)
        inter = INTER_LINEAR;

    mapx.create(dsize, CV_16SC2);
    if (inter != INTER_NEAREST)
        mapy.create(dsize, CV_16SC1);
    else
        mapy = Mat();

    if (!(interpolation & cv::WARP_INVERSE_MAP))
        invertAffineTransform(tM.clone(), tM);

    if (inter == INTER_LINEAR) {
        int dst_depth = _dst.depth(), dst_channels = _dst.channels();
        if (dst_depth == CV_8U && (dst_channels == 1 || dst_channels == 3 || dst_channels == 4)) {
            return new_warpAffine<uint8_t>(_src, _dst, tM, inter);
        } else if (dst_depth == CV_16U && (dst_channels == 1 || dst_channels == 3 || dst_channels == 4)) {
            return new_warpAffine<uint16_t>(_src, _dst, tM, inter);
        } else if (dst_depth == CV_32F && (dst_channels == 1 || dst_channels == 3 || dst_channels == 4)) {
            return new_warpAffine<float>(_src, _dst, tM, inter);
        }
    }

    const int AB_BITS = MAX(10, (int)INTER_BITS);
    const int AB_SCALE = 1 << AB_BITS;
    int round_delta = (inter == INTER_NEAREST) ? AB_SCALE / 2 : (AB_SCALE / INTER_TAB_SIZE / 2);

    const softdouble* data_tM = tM.ptr<softdouble>(0);
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        short* yM = mapx.ptr<short>(dy);
        for (int dx = 0; dx < dsize.width; ++dx, yM += 2)
        {
            int v1 = saturate_cast<int>(saturate_cast<int>(data_tM[0] * dx * AB_SCALE) +
                    saturate_cast<int>((data_tM[1] * dy + data_tM[2]) * AB_SCALE) + round_delta),
                v2 = saturate_cast<int>(saturate_cast<int>(data_tM[3] * dx * AB_SCALE) +
                    saturate_cast<int>((data_tM[4] * dy + data_tM[5]) * AB_SCALE) + round_delta);
            v1 >>= AB_BITS - INTER_BITS;
            v2 >>= AB_BITS - INTER_BITS;

            yM[0] = saturate_cast<short>(v1 >> INTER_BITS);
            yM[1] = saturate_cast<short>(v2 >> INTER_BITS);

            if (inter != INTER_NEAREST)
                mapy.ptr<short>(dy)[dx] = ((v2 & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (v1 & (INTER_TAB_SIZE - 1)));
        }
    }

    CV_Assert(mapx.type() == CV_16SC2 && ((inter == INTER_NEAREST && mapy.empty()) || mapy.type() == CV_16SC1));
    cv::remap(_src, _dst, mapx, mapy, inter, borderType, borderValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// warpPerspective
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_WarpPerspective_Test :
    public CV_WarpAffine_Test
{
public:
    CV_WarpPerspective_Test();

    virtual ~CV_WarpPerspective_Test();

protected:
    virtual void generate_test_data();
    virtual float get_success_error_level(int _interpolation, int _depth) const;

    virtual void run_func();
    virtual void run_reference_func();

private:
    void warpPerspective(const Mat&, Mat&);

    template<typename T>
    void new_warpPerspective(const Mat&, Mat&, const Mat&, int);
};

CV_WarpPerspective_Test::CV_WarpPerspective_Test() :
    CV_WarpAffine_Test()
{
}

CV_WarpPerspective_Test::~CV_WarpPerspective_Test()
{
}

void CV_WarpPerspective_Test::generate_test_data()
{
    CV_Remap_Test::generate_test_data();

    // generating the M 3x3 matrix
    RNG& rng = ts->get_rng();

    float cols = static_cast<float>(src.cols), rows = static_cast<float>(src.rows);
    Point2f sp[] = { Point2f(0.0f, 0.0f), Point2f(cols, 0.0f), Point2f(0.0f, rows), Point2f(cols, rows) };
    Point2f dp[] = { Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)),
        Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)),
        Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)),
        Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)) };
    M = getPerspectiveTransform(sp, dp);

    static const int depths[] = { CV_32F, CV_64F };
    int depth = depths[rng.uniform(0, 2)];
    M.clone().convertTo(M, depth);
}

void CV_WarpPerspective_Test::run_func()
{
    cv::warpPerspective(src, dst, M, dst.size(), interpolation, borderType, borderValue);
}

float CV_WarpPerspective_Test::get_success_error_level(int _interpolation, int _depth) const
{
    return CV_ImageWarpBaseTest::get_success_error_level(_interpolation, _depth);
}

void CV_WarpPerspective_Test::run_reference_func()
{
    Mat tmp = Mat::zeros(dst.size(), dst.type());
    warpPerspective(src, tmp);
    tmp.convertTo(reference_dst, reference_dst.depth());
}

template<typename T>
void CV_WarpPerspective_Test::new_warpPerspective(const Mat &_src, Mat &_dst, const Mat &tM, int inter)
{
    CV_UNUSED(inter);
    int num_channels = _dst.channels();
    CV_CheckTrue(num_channels == 1 || num_channels == 3 || num_channels == 4, "");

    auto *srcptr_ = _src.ptr<const T>();
    auto *dstptr_ = _dst.ptr<T>();
    size_t srcstep = _src.step/sizeof(T), dststep = _dst.step/sizeof(T);
    int srccols = _src.cols, srcrows = _src.rows;
    int dstcols = _dst.cols, dstrows = _dst.rows;

    Mat tmp;
    tM.convertTo(tmp, CV_32F);
    auto *_M = tmp.ptr<const float>();

    T bval[] = {
        saturate_cast<T>(borderValue[0]),
        saturate_cast<T>(borderValue[1]),
        saturate_cast<T>(borderValue[2]),
        saturate_cast<T>(borderValue[3]),
    };

    int borderType_x = borderType != BORDER_CONSTANT &&
                       borderType != BORDER_TRANSPARENT &&
                       srccols <= 1 ? BORDER_REPLICATE : borderType;
    int borderType_y = borderType != BORDER_CONSTANT &&
                       borderType != BORDER_TRANSPARENT &&
                       srcrows <= 1 ? BORDER_REPLICATE : borderType;

    for (int y = 0; y < dstrows; y++) {
        T* dstptr = dstptr_ + y*dststep;
        for (int x = 0; x < dstcols; x++) {
            float w = x*_M[6] + y*_M[7] + _M[8];
            float sx = (x*_M[0] + y*_M[1] + _M[2]) / w;
            float sy = (x*_M[3] + y*_M[4] + _M[5]) / w;

            if (num_channels == 3) {
                new_linear_c3(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            } else if (num_channels == 4) {
                new_linear_c4(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            } else {
                new_linear_c1(x, sx, sy, srcptr_, dstptr, srccols, srcrows, srcstep, bval, borderType_x, borderType_y);
            }
        }
    }
}

void CV_WarpPerspective_Test::warpPerspective(const Mat& _src, Mat& _dst)
{
    Size ssize = _src.size(), dsize = _dst.size();

    CV_Assert(!ssize.empty());
    CV_Assert(!dsize.empty());
    CV_Assert(_src.type() == _dst.type());

    if (M.depth() != CV_64F)
    {
        Mat tmp;
        M.convertTo(tmp, CV_64F);
        M = tmp;
    }

    if (!(interpolation & cv::WARP_INVERSE_MAP))
    {
        Mat tmp;
        invert(M, tmp);
        M = tmp;
    }

    int inter = interpolation & INTER_MAX;
    if (inter == INTER_AREA)
        inter = INTER_LINEAR;

    if (inter == INTER_LINEAR) {
        int dst_depth = _dst.depth(), dst_channels = _dst.channels();
        if (dst_depth == CV_8U && (dst_channels == 1 || dst_channels == 3 || dst_channels == 4)) {
            return new_warpPerspective<uint8_t>(_src, _dst, M, inter);
        } else if (dst_depth == CV_16U && (dst_channels == 1 || dst_channels == 3 || dst_channels == 4)) {
            return new_warpPerspective<uint16_t>(_src, _dst, M, inter);
        } else if (dst_depth == CV_32F && (dst_channels == 1 || dst_channels == 3 || dst_channels == 4)) {
            return new_warpPerspective<float>(_src, _dst, M, inter);
        }
    }

    mapx.create(dsize, CV_16SC2);
    if (inter != INTER_NEAREST)
        mapy.create(dsize, CV_16SC1);
    else
        mapy = Mat();

    double* tM = M.ptr<double>(0);
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        short* yMx = mapx.ptr<short>(dy);

        for (int dx = 0; dx < dsize.width; ++dx, yMx += 2)
        {
            double den = tM[6] * dx + tM[7] * dy + tM[8];
            den = den ? 1.0 / den : 0.0;

            if (inter == INTER_NEAREST)
            {
                yMx[0] = saturate_cast<short>((tM[0] * dx + tM[1] * dy + tM[2]) * den);
                yMx[1] = saturate_cast<short>((tM[3] * dx + tM[4] * dy + tM[5]) * den);
                continue;
            }

            den *= static_cast<double>(INTER_TAB_SIZE);
            int v0 = saturate_cast<int>((tM[0] * dx + tM[1] * dy + tM[2]) * den);
            int v1 = saturate_cast<int>((tM[3] * dx + tM[4] * dy + tM[5]) * den);

            yMx[0] = saturate_cast<short>(v0 >> INTER_BITS);
            yMx[1] = saturate_cast<short>(v1 >> INTER_BITS);
            mapy.ptr<short>(dy)[dx] = saturate_cast<short>((v1 & (INTER_TAB_SIZE - 1)) *
                    INTER_TAB_SIZE + (v0 & (INTER_TAB_SIZE - 1)));
        }
    }

    CV_Assert(mapx.type() == CV_16SC2 && ((inter == INTER_NEAREST && mapy.empty()) || mapy.type() == CV_16SC1));
    cv::remap(_src, _dst, mapx, mapy, inter, borderType, borderValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Imgproc_Resize_Test, accuracy) { CV_Resize_Test test; test.safe_run(); }
TEST(Imgproc_Remap_Test, accuracy) { CV_Remap_Test test; test.safe_run(); }
TEST(Imgproc_WarpAffine_Test, accuracy) { CV_WarpAffine_Test test; test.safe_run(); }
TEST(Imgproc_WarpPerspective_Test, accuracy) { CV_WarpPerspective_Test test; test.safe_run(); }

////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef OPENCV_TEST_BIGDATA

CV_ENUM(Interpolation, INTER_NEAREST, INTER_LINEAR, INTER_LINEAR_EXACT, INTER_CUBIC, INTER_AREA)

class Imgproc_Resize :
        public ::testing::TestWithParam<Interpolation>
{
public:
    virtual void SetUp()
    {
        inter = GetParam();
    }

protected:
    int inter;
};

TEST_P(Imgproc_Resize, BigSize)
{
    cv::Mat src(46342, 46342, CV_8UC3, cv::Scalar::all(10)), dst;
    ASSERT_FALSE(src.empty());

    ASSERT_NO_THROW(cv::resize(src, dst, cv::Size(), 0.5, 0.5, inter));
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_Resize, Interpolation::all());

#endif

}} // namespace
