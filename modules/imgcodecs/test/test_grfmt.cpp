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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<string, int> File_Mode;
typedef testing::TestWithParam<File_Mode> Imgcodecs_FileMode;

TEST_P(Imgcodecs_FileMode, regression)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + get<0>(GetParam());
    const int mode = get<1>(GetParam());

    const Mat single = imread(filename, mode);
    ASSERT_FALSE(single.empty());

    vector<Mat> pages;
    ASSERT_TRUE(imreadmulti(filename, pages, mode));
    ASSERT_FALSE(pages.empty());
    const Mat page = pages[0];
    ASSERT_FALSE(page.empty());

    EXPECT_EQ(page.channels(), single.channels());
    EXPECT_EQ(page.depth(), single.depth());
    EXPECT_EQ(page.size().height, single.size().height);
    EXPECT_EQ(page.size().width, single.size().width);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), page, single);
}

const string all_images[] =
{
#if (defined(HAVE_JASPER) && defined(OPENCV_IMGCODECS_ENABLE_JASPER_TESTS)) \
    || defined(HAVE_OPENJPEG)
    "readwrite/Rome.jp2",
    "readwrite/Bretagne2.jp2",
    "readwrite/Bretagne2.jp2",
    "readwrite/Grey.jp2",
    "readwrite/Grey.jp2",
    "readwrite/balloon.j2c",
#endif

#ifdef HAVE_GDCM
    "readwrite/int16-mono1.dcm",
    "readwrite/uint8-mono2.dcm",
    "readwrite/uint16-mono2.dcm",
    "readwrite/uint8-rgb.dcm",
#endif
#if defined(HAVE_PNG) || defined(HAVE_SPNG)
    "readwrite/color_palette_alpha.png",
#endif
#ifdef HAVE_TIFF
    "readwrite/multipage.tif",
#endif
    "readwrite/ordinary.bmp",
    "readwrite/rle8.bmp",
#ifdef HAVE_JPEG
    "readwrite/test_1_c1.jpg",
#endif
#ifdef HAVE_IMGCODEC_HDR
    "readwrite/rle.hdr"
#endif
};

const int basic_modes[] =
{
    IMREAD_UNCHANGED,
    IMREAD_GRAYSCALE,
    IMREAD_COLOR,
    IMREAD_COLOR_RGB,
    IMREAD_ANYDEPTH,
    IMREAD_ANYCOLOR
};

INSTANTIATE_TEST_CASE_P(All, Imgcodecs_FileMode,
                        testing::Combine(
                            testing::ValuesIn(all_images),
                            testing::ValuesIn(basic_modes)));

// GDAL does not support "hdr", "dcm" and has problems with JPEG2000 files (jp2, j2c)
struct notForGDAL {
    bool operator()(const string &name) const {
        const string &ext = name.substr(name.size() - 3, 3);
        return ext == "hdr" || ext == "dcm" || ext == "jp2" || ext == "j2c" ||
                name.find("rle8.bmp") != std::string::npos;
    }
};

inline vector<string> gdal_images()
{
    vector<string> res;
    std::back_insert_iterator< vector<string> > it(res);
    std::remove_copy_if(all_images, all_images + sizeof(all_images)/sizeof(all_images[0]), it, notForGDAL());
    return res;
}

INSTANTIATE_TEST_CASE_P(GDAL, Imgcodecs_FileMode,
                        testing::Combine(
                            testing::ValuesIn(gdal_images()),
                            testing::Values(IMREAD_LOAD_GDAL)));

//==================================================================================================

typedef tuple<string, Size> Ext_Size;
typedef testing::TestWithParam<Ext_Size> Imgcodecs_ExtSize;

TEST_P(Imgcodecs_ExtSize, write_imageseq)
{
    const string ext = get<0>(GetParam());
    const Size size = get<1>(GetParam());
    const Point2i center = Point2i(size.width / 2, size.height / 2);
    const int radius = std::min(size.height, size.width / 4);

    for (int cn = 1; cn <= 4; cn++)
    {
        SCOPED_TRACE(format("channels %d", cn));
        std::vector<int> parameters;
        if (cn == 2)
            continue;
        if (cn == 4 && ext != ".tiff")
            continue;
        if (cn > 1 && (ext == ".pbm" || ext == ".pgm"))
            continue;
        if (cn != 3 && ext == ".ppm")
            continue;
        string filename = cv::tempfile(format("%d%s", cn, ext.c_str()).c_str());

        Mat img_gt(size, CV_MAKETYPE(CV_8U, cn), Scalar::all(0));
        circle(img_gt, center, radius, Scalar::all(255));

#if 1
        if (ext == ".pbm" || ext == ".pgm" || ext == ".ppm")
        {
            parameters.push_back(IMWRITE_PXM_BINARY);
            parameters.push_back(0);
        }
#endif
        ASSERT_TRUE(imwrite(filename, img_gt, parameters));
        Mat img = imread(filename, IMREAD_UNCHANGED);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(img.size(), img.size());
        EXPECT_EQ(img.type(), img.type());
        EXPECT_EQ(cn, img.channels());


        if (ext == ".jpg")
        {
            // JPEG format does not provide 100% accuracy
            // using fuzzy image comparison
            double n = cvtest::norm(img, img_gt, NORM_L1);
            double expected = 0.07 * img.size().area();
            EXPECT_LT(n, expected);
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(10, 0), img, img_gt);
        }
        else if (ext == ".pfm")
        {
            img_gt.convertTo(img_gt, CV_MAKETYPE(CV_32F, img.channels()));
            double n = cvtest::norm(img, img_gt, NORM_L2);
            EXPECT_LT(n, 1.);
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_gt);
        }
        else
        {
            double n = cvtest::norm(img, img_gt, NORM_L2);
            EXPECT_LT(n, 1.);
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_gt);
        }

#if 0
        imshow("loaded", img);
        waitKey(0);
#else
        EXPECT_EQ(0, remove(filename.c_str()));
#endif
    }
}

const string all_exts[] =
{
#if defined(HAVE_PNG) || defined(HAVE_SPNG)
    ".png",
#endif
#ifdef HAVE_TIFF
    ".tiff",
#endif
#ifdef HAVE_JPEG
    ".jpg",
#endif
    ".bmp",
#ifdef HAVE_IMGCODEC_PXM
    ".pam",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
#endif
#ifdef HAVE_IMGCODEC_PFM
    ".pfm",
#endif
};

vector<Size> all_sizes()
{
    vector<Size> res;
    for (int k = 1; k <= 5; ++k)
        res.push_back(Size(640 * k, 480 * k));
    return res;
}

INSTANTIATE_TEST_CASE_P(All, Imgcodecs_ExtSize,
                        testing::Combine(
                            testing::ValuesIn(all_exts),
                            testing::ValuesIn(all_sizes())));

#ifdef HAVE_IMGCODEC_PXM
typedef testing::TestWithParam<bool> Imgcodecs_pbm;
TEST_P(Imgcodecs_pbm, write_read)
{
    bool binary = GetParam();
    const String ext = "pbm";
    const string full_name = cv::tempfile(ext.c_str());

    Size size(640, 480);
    const Point2i center = Point2i(size.width / 2, size.height / 2);
    const int radius = std::min(size.height, size.width / 4);
    Mat image(size, CV_8UC1, Scalar::all(0));
    circle(image, center, radius, Scalar::all(255));

    vector<int> pbm_params;
    pbm_params.push_back(IMWRITE_PXM_BINARY);
    pbm_params.push_back(binary);

    imwrite( full_name, image, pbm_params );
    Mat loaded = imread(full_name, IMREAD_UNCHANGED);
    ASSERT_FALSE(loaded.empty());

    EXPECT_EQ(0, cvtest::norm(loaded, image, NORM_INF));

    FILE *f = fopen(full_name.c_str(), "rb");
    ASSERT_TRUE(f != NULL);
    ASSERT_EQ('P', getc(f));
    ASSERT_EQ('1' + (binary ? 3 : 0), getc(f));
    fclose(f);
    EXPECT_EQ(0, remove(full_name.c_str()));
}

INSTANTIATE_TEST_CASE_P(All, Imgcodecs_pbm, testing::Bool());
#endif


//==================================================================================================

TEST(Imgcodecs_Bmp, read_rle8)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    Mat rle = imread(root + "readwrite/rle8.bmp");
    ASSERT_FALSE(rle.empty());
    Mat ord = imread(root + "readwrite/ordinary.bmp");
    ASSERT_FALSE(ord.empty());
    EXPECT_LE(cvtest::norm(rle, ord, NORM_L2), 1.e-10);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), rle, ord);
}

TEST(Imgcodecs_Bmp, read_32bit_rgb)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_32bit_rgb.bmp";

    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC3, img.type());
}

TEST(Imgcodecs_Bmp, rgba_bit_mask)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_rgba_mask.bmp";

    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC4, img.type());

    const uchar* data = img.ptr();
    ASSERT_EQ(data[3], 255);
}

TEST(Imgcodecs_Bmp, read_32bit_xrgb)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_32bit_xrgb.bmp";

    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC4, img.type());

    const uchar* data = img.ptr();
    ASSERT_EQ(data[3], 255);
}

TEST(Imgcodecs_Bmp, rgba_scale)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_rgba_scale.bmp";

    Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC4, img.type());

    uchar* data = img.ptr();
    ASSERT_EQ(data[0], 255);
    ASSERT_EQ(data[1], 255);
    ASSERT_EQ(data[2], 255);
    ASSERT_EQ(data[3], 255);

    img = cv::imread(filenameInput, IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC3, img.type());

    img = cv::imread(filenameInput, IMREAD_COLOR_RGB);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC3, img.type());

    data = img.ptr();
    ASSERT_EQ(data[0], 255);
    ASSERT_EQ(data[1], 255);
    ASSERT_EQ(data[2], 255);

    img = cv::imread(filenameInput, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC1, img.type());

    data = img.ptr();
    ASSERT_EQ(data[0], 255);
}

#ifdef HAVE_IMGCODEC_HDR
TEST(Imgcodecs_Hdr, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "/readwrite/";
    string name_rle = folder + "rle.hdr";
    string name_no_rle = folder + "no_rle.hdr";
    Mat img_rle = imread(name_rle, -1);
    ASSERT_FALSE(img_rle.empty()) << "Could not open " << name_rle;
    Mat img_no_rle = imread(name_no_rle, -1);
    ASSERT_FALSE(img_no_rle.empty()) << "Could not open " << name_no_rle;

    EXPECT_EQ(cvtest::norm(img_rle, img_no_rle, NORM_INF), 0.0);

    string tmp_file_name = tempfile(".hdr");
    vector<int> param(2);
    param[0] = IMWRITE_HDR_COMPRESSION;
    for(int i = 0; i < 2; i++) {
        param[1] = i;
        imwrite(tmp_file_name, img_rle, param);
        Mat written_img = imread(tmp_file_name, -1);
        EXPECT_EQ(cvtest::norm(written_img, img_rle, NORM_INF), 0.0);
    }
    remove(tmp_file_name.c_str());
}

TEST(Imgcodecs_Hdr, regression_imencode)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "/readwrite/";
    string name = folder + "rle.hdr";
    Mat img_ref = imread(name, -1);
    ASSERT_FALSE(img_ref.empty()) << "Could not open " << name;

    vector<int> params(2);
    params[0] = IMWRITE_HDR_COMPRESSION;
    {
        vector<uchar> buf;
        params[1] = IMWRITE_HDR_COMPRESSION_NONE;
        imencode(".hdr", img_ref, buf, params);
        Mat img = imdecode(buf, -1);
        EXPECT_EQ(cvtest::norm(img_ref, img, NORM_INF), 0.0);
    }
    {
        vector<uchar> buf;
        params[1] = IMWRITE_HDR_COMPRESSION_RLE;
        imencode(".hdr", img_ref, buf, params);
        Mat img = imdecode(buf, -1);
        EXPECT_EQ(cvtest::norm(img_ref, img, NORM_INF), 0.0);
    }
}

#endif

#ifdef HAVE_IMGCODEC_PXM
TEST(Imgcodecs_Pam, read_write)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "readwrite/";
    string filepath = folder + "lena.pam";

    cv::Mat img = cv::imread(filepath);
    ASSERT_FALSE(img.empty());

    std::vector<int> params;
    params.push_back(IMWRITE_PAM_TUPLETYPE);
    params.push_back(IMWRITE_PAM_FORMAT_RGB);

    string writefile = cv::tempfile(".pam");
    EXPECT_NO_THROW(cv::imwrite(writefile, img, params));
    cv::Mat reread = cv::imread(writefile);

    string writefile_no_param = cv::tempfile(".pam");
    EXPECT_NO_THROW(cv::imwrite(writefile_no_param, img));
    cv::Mat reread_no_param = cv::imread(writefile_no_param);

    EXPECT_EQ(0, cvtest::norm(reread, reread_no_param, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(img, reread, NORM_INF));

    remove(writefile.c_str());
    remove(writefile_no_param.c_str());
}
#endif

#ifdef HAVE_IMGCODEC_PFM
TEST(Imgcodecs_Pfm, read_write)
{
  Mat img = imread(findDataFile("readwrite/lena.pam"));
  ASSERT_FALSE(img.empty());
  img.convertTo(img, CV_32F, 1/255.0f);

  std::vector<int> params;
  string writefile = cv::tempfile(".pfm");
  EXPECT_NO_THROW(cv::imwrite(writefile, img, params));
  cv::Mat reread = cv::imread(writefile, IMREAD_UNCHANGED);

  string writefile_no_param = cv::tempfile(".pfm");
  EXPECT_NO_THROW(cv::imwrite(writefile_no_param, img));
  cv::Mat reread_no_param = cv::imread(writefile_no_param, IMREAD_UNCHANGED);

  EXPECT_EQ(0, cvtest::norm(reread, reread_no_param, NORM_INF));
  EXPECT_EQ(0, cvtest::norm(img, reread, NORM_INF));

  EXPECT_EQ(0, remove(writefile.c_str()));
  EXPECT_EQ(0, remove(writefile_no_param.c_str()));
}
#endif

TEST(Imgcodecs, write_parameter_type)
{
    cv::Mat m(10, 10, CV_8UC1, cv::Scalar::all(0));
    cv::Mat1b m_type = cv::Mat1b::zeros(10, 10);
    string tmp_file = cv::tempfile(".bmp");
    EXPECT_NO_THROW(cv::imwrite(tmp_file, cv::Mat(m * 2))) << "* Failed with cv::Mat";
    EXPECT_NO_THROW(cv::imwrite(tmp_file, m * 2)) << "* Failed with cv::MatExpr";
    EXPECT_NO_THROW(cv::imwrite(tmp_file, m_type)) << "* Failed with cv::Mat_";
    EXPECT_NO_THROW(cv::imwrite(tmp_file, m_type * 2)) << "* Failed with cv::MatExpr(Mat_)";
    cv::Matx<uchar, 10, 10> matx;
    EXPECT_NO_THROW(cv::imwrite(tmp_file, matx)) << "* Failed with cv::Matx";
    EXPECT_EQ(0, remove(tmp_file.c_str()));
}

TEST(Imgcodecs, imdecode_user_buffer)
{
    cv::Mat encoded = cv::Mat::zeros(1, 1024, CV_8UC1);
    cv::Mat user_buffer(1, 1024, CV_8UC1);
    cv::Mat result = cv::imdecode(encoded, IMREAD_ANYCOLOR, &user_buffer);
    EXPECT_TRUE(result.empty());
    // the function does not release user-provided buffer
    EXPECT_FALSE(user_buffer.empty());

    result = cv::imdecode(encoded, IMREAD_ANYCOLOR);
    EXPECT_TRUE(result.empty());
}

}} // namespace
