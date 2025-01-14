// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

namespace opencv_test { namespace {

/* < <file_name, image_size>, <imread mode, scale> > */
typedef tuple< tuple<string, Size>, tuple<ImreadModes, int> > Imgcodecs_Resize_t;

typedef testing::TestWithParam< Imgcodecs_Resize_t > Imgcodecs_Resize;

/* resize_flag_and_dims = <imread_flag, scale>*/
const tuple <ImreadModes, int> resize_flag_and_dims[] =
{
    make_tuple(IMREAD_UNCHANGED, 1),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_2, 2),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_4, 4),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_8, 8),
    make_tuple(IMREAD_REDUCED_COLOR_2, 2),
    make_tuple(IMREAD_REDUCED_COLOR_4, 4),
    make_tuple(IMREAD_REDUCED_COLOR_8, 8)
};

const tuple<string, Size> images[] =
{
#ifdef HAVE_JPEG
    make_tuple<string, Size>("../cv/imgproc/stuff.jpg", Size(640, 480)),
#endif
#if defined(HAVE_PNG) || defined(HAVE_SPNG)
    make_tuple<string, Size>("../cv/shared/pic1.png", Size(400, 300)),
#endif
    make_tuple<string, Size>("../highgui/readwrite/ordinary.bmp", Size(480, 272)),
};

TEST_P(Imgcodecs_Resize, imread_reduce_flags)
{
    const string file_name = findDataFile(get<0>(get<0>(GetParam())));
    const Size imageSize = get<1>(get<0>(GetParam()));

    const int imread_flag = get<0>(get<1>(GetParam()));
    const int scale = get<1>(get<1>(GetParam()));

    const int cols = imageSize.width / scale;
    const int rows = imageSize.height / scale;
    {
        Mat img = imread(file_name, imread_flag);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cols, img.cols);
        EXPECT_EQ(rows, img.rows);
    }
}

//==================================================================================================

TEST_P(Imgcodecs_Resize, imdecode_reduce_flags)
{
    const string file_name = findDataFile(get<0>(get<0>(GetParam())));
    const Size imageSize = get<1>(get<0>(GetParam()));

    const int imread_flag = get<0>(get<1>(GetParam()));
    const int scale = get<1>(get<1>(GetParam()));

    const int cols = imageSize.width / scale;
    const int rows = imageSize.height / scale;

    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(file_name.c_str(), mode);
    ASSERT_TRUE(ifs.is_open());

    ifs.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    std::vector<char> content(sz);
    ifs.read((char*)content.data(), sz);
    ASSERT_FALSE(ifs.fail());

    {
        Mat img = imdecode(Mat(content), imread_flag);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cols, img.cols);
        EXPECT_EQ(rows, img.rows);
    }
}

//==================================================================================================

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Resize,
        testing::Combine(
            testing::ValuesIn(images),
            testing::ValuesIn(resize_flag_and_dims)
            )
        );

//==================================================================================================

TEST(Imgcodecs_Image, read_write_bmp)
{
    const size_t IMAGE_COUNT = 10;
    const double thresDbell = 32;

    for (size_t i = 0; i < IMAGE_COUNT; ++i)
    {
        stringstream s; s << i;
        const string digit = s.str();
        const string src_name = TS::ptr()->get_data_path() + "../python/images/QCIF_0" + digit + ".bmp";
        const string dst_name = cv::tempfile((digit + ".bmp").c_str());
        Mat image = imread(src_name);
        ASSERT_FALSE(image.empty());

        resize(image, image, Size(968, 757), 0.0, 0.0, INTER_CUBIC);
        imwrite(dst_name, image);
        Mat loaded = imread(dst_name);
        ASSERT_FALSE(loaded.empty());

        double psnr = cvtest::PSNR(loaded, image);
        EXPECT_GT(psnr, thresDbell);

        vector<uchar> from_file;

        FILE *f = fopen(dst_name.c_str(), "rb");
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        from_file.resize((size_t)len);
        fseek(f, 0, SEEK_SET);
        from_file.resize(fread(&from_file[0], 1, from_file.size(), f));
        fclose(f);

        vector<uchar> buf;
        imencode(".bmp", image, buf);
        ASSERT_EQ(buf, from_file);

        Mat buf_loaded = imdecode(Mat(buf), 1);
        ASSERT_FALSE(buf_loaded.empty());

        psnr = cvtest::PSNR(buf_loaded, image);
        EXPECT_GT(psnr, thresDbell);

        EXPECT_EQ(0, remove(dst_name.c_str()));
    }
}

//==================================================================================================

typedef string Ext;
typedef testing::TestWithParam<Ext> Imgcodecs_Image;

const string exts[] = {
#if defined(HAVE_PNG) || defined(HAVE_SPNG)
    "png",
#endif
#ifdef HAVE_TIFF
    "tiff",
#endif
#ifdef HAVE_JPEG
    "jpg",
#endif
#ifdef HAVE_JPEGXL
    "jxl",
#endif
#if (defined(HAVE_JASPER) && defined(OPENCV_IMGCODECS_ENABLE_JASPER_TESTS)) \
    || defined(HAVE_OPENJPEG)
    "jp2",
#endif
#if 0 /*defined HAVE_OPENEXR && !defined __APPLE__*/
    "exr",
#endif
    "bmp",
#ifdef HAVE_IMGCODEC_PXM
    "ppm",
#endif
#ifdef HAVE_IMGCODEC_SUNRASTER
    "ras",
#endif
};

static
void test_image_io(const Mat& image, const std::string& fname, const std::string& ext, int imreadFlag, double psnrThreshold)
{
    vector<uchar> buf;
    ASSERT_NO_THROW(imencode("." + ext, image, buf));

    ASSERT_NO_THROW(imwrite(fname, image));

    FILE *f = fopen(fname.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    cout << "File size: " << len << " bytes" << endl;
    EXPECT_GT(len, 1024) << "File is small. Test or implementation is broken";
    fseek(f, 0, SEEK_SET);
    vector<uchar> file_buf((size_t)len);
    EXPECT_EQ(len, (long)fread(&file_buf[0], 1, (size_t)len, f));
    fclose(f); f = NULL;

    EXPECT_EQ(buf, file_buf) << "imwrite() / imencode() calls must provide the same output (bit-exact)";

    Mat buf_loaded = imdecode(Mat(buf), imreadFlag);
    EXPECT_FALSE(buf_loaded.empty());

    if (imreadFlag & IMREAD_COLOR_RGB && imreadFlag != -1)
    {
        cvtColor(buf_loaded, buf_loaded, COLOR_RGB2BGR);
    }

    Mat loaded = imread(fname, imreadFlag);
    EXPECT_FALSE(loaded.empty());

    if (imreadFlag & IMREAD_COLOR_RGB && imreadFlag != -1)
    {
        cvtColor(loaded, loaded, COLOR_RGB2BGR);
    }

    EXPECT_EQ(0, cv::norm(loaded, buf_loaded, NORM_INF)) << "imread() and imdecode() calls must provide the same result (bit-exact)";

    double psnr = cvtest::PSNR(loaded, image);
    EXPECT_GT(psnr, psnrThreshold);

    // not necessary due bitexact check above
    //double buf_psnr = cvtest::PSNR(buf_loaded, image);
    //EXPECT_GT(buf_psnr, psnrThreshold);

#if 0  // debug
    if (psnr <= psnrThreshold /*|| buf_psnr <= thresDbell*/)
    {
        cout << "File: " << fname << endl;
        imshow("origin", image);
        imshow("imread", loaded);
        imshow("imdecode", buf_loaded);
        waitKey();
    }
#endif
}

TEST_P(Imgcodecs_Image, read_write_BGR)
{
    const string ext = this->GetParam();
    const string fname = cv::tempfile(ext.c_str());

    double psnrThreshold = 100;
    if (ext == "jpg")
        psnrThreshold = 32;
    if (ext == "jxl")
        psnrThreshold = 30;
#if defined(HAVE_JASPER)
    if (ext == "jp2")
        psnrThreshold = 95;
#elif defined(HAVE_OPENJPEG)
    if (ext == "jp2")
        psnrThreshold = 35;
#endif

    Mat image = generateTestImageBGR();
    EXPECT_NO_THROW(test_image_io(image, fname, ext, IMREAD_COLOR, psnrThreshold));
    EXPECT_NO_THROW(test_image_io(image, fname, ext, IMREAD_COLOR_RGB, psnrThreshold));

    EXPECT_EQ(0, remove(fname.c_str()));
}

TEST_P(Imgcodecs_Image, read_write_GRAYSCALE)
{
    const string ext = this->GetParam();

    if (false
        || ext == "ppm"  // grayscale is not implemented
        || ext == "ras"  // broken (black result)
    )
        throw SkipTestException("GRAYSCALE mode is not supported");

    const string fname = cv::tempfile(ext.c_str());

    double psnrThreshold = 100;
    if (ext == "jpg")
        psnrThreshold = 40;
    if (ext == "jxl")
        psnrThreshold = 40;
#if defined(HAVE_JASPER)
    if (ext == "jp2")
        psnrThreshold = 70;
#elif defined(HAVE_OPENJPEG)
    if (ext == "jp2")
        psnrThreshold = 35;
#endif

    Mat image = generateTestImageGrayscale();
    EXPECT_NO_THROW(test_image_io(image, fname, ext, IMREAD_GRAYSCALE, psnrThreshold));

    EXPECT_EQ(0, remove(fname.c_str()));
}

INSTANTIATE_TEST_CASE_P(imgcodecs, Imgcodecs_Image, testing::ValuesIn(exts));

TEST(Imgcodecs_Image, regression_9376)
{
    String path = findDataFile("readwrite/regression_9376.bmp");
    Mat m = imread(path);
    ASSERT_FALSE(m.empty());
    EXPECT_EQ(32, m.cols);
    EXPECT_EQ(32, m.rows);
}

TEST(Imgcodecs_Image, imread_overload)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string imgName = findDataFile("../highgui/readwrite/ordinary.bmp");

    Mat ref = imread(imgName);
    ASSERT_FALSE(ref.empty());
    {
        Mat img(ref.size(), ref.type(), Scalar::all(0)); // existing image
        void * ptr = img.data;
        imread(imgName, img);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cv::norm(ref, img, NORM_INF), 0);
        EXPECT_EQ(img.data, ptr); // no reallocation
    }
    {
        Mat img; // empty image
        imread(imgName, img);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cv::norm(ref, img, NORM_INF), 0);
    }
    {
        UMat img; // empty UMat
        imread(imgName, img);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cv::norm(ref, img, NORM_INF), 0);
    }
}

//==================================================================================================

TEST(Imgcodecs_Image, write_umat)
{
    const string src_name = TS::ptr()->get_data_path() + "../python/images/baboon.bmp";
    const string dst_name = cv::tempfile(".bmp");

    Mat image1 = imread(src_name);
    ASSERT_FALSE(image1.empty());

    UMat image1_umat = image1.getUMat(ACCESS_RW);

    imwrite(dst_name, image1_umat);

    Mat image2 = imread(dst_name);
    ASSERT_FALSE(image2.empty());

    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), image1, image2);
    EXPECT_EQ(0, remove(dst_name.c_str()));
}

#ifdef HAVE_TIFF
TEST(Imgcodecs_Image, multipage_collection_size)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";

    ImageCollection collection(filename, IMREAD_ANYCOLOR);
    EXPECT_EQ((std::size_t)6, collection.size());
}

TEST(Imgcodecs_Image, multipage_collection_read_pages_iterator)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";
    const string page_files[] = {
            root + "readwrite/multipage_p1.tif",
            root + "readwrite/multipage_p2.tif",
            root + "readwrite/multipage_p3.tif",
            root + "readwrite/multipage_p4.tif",
            root + "readwrite/multipage_p5.tif",
            root + "readwrite/multipage_p6.tif"
    };

    ImageCollection collection(filename, IMREAD_ANYCOLOR);

    auto collectionBegin = collection.begin();
    for(size_t i = 0; i < collection.size(); ++i, ++collectionBegin)
    {
        double diff = cv::norm(collectionBegin.operator*(), imread(page_files[i]), NORM_INF);
        EXPECT_EQ(0., diff);
    }
}

TEST(Imgcodecs_Image, multipage_collection_two_iterator)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";
    const string page_files[] = {
            root + "readwrite/multipage_p1.tif",
            root + "readwrite/multipage_p2.tif",
            root + "readwrite/multipage_p3.tif",
            root + "readwrite/multipage_p4.tif",
            root + "readwrite/multipage_p5.tif",
            root + "readwrite/multipage_p6.tif"
    };

    ImageCollection collection(filename, IMREAD_ANYCOLOR);
    auto firstIter = collection.begin();
    auto secondIter = collection.begin();

    // Decode all odd pages then decode even pages -> 1, 0, 3, 2 ...
    firstIter++;
    for(size_t i = 1; i < collection.size(); i += 2, ++firstIter, ++firstIter, ++secondIter, ++secondIter) {
        Mat mat = *firstIter;
        double diff = cv::norm(mat, imread(page_files[i]), NORM_INF);
        EXPECT_EQ(0., diff);
        Mat evenMat = *secondIter;
        diff = cv::norm(evenMat, imread(page_files[i-1]), NORM_INF);
        EXPECT_EQ(0., diff);
    }
}

TEST(Imgcodecs_Image, multipage_collection_operator_plusplus)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";

    // operator++ test
    ImageCollection collection(filename, IMREAD_ANYCOLOR);
    auto firstIter = collection.begin();
    auto secondIter = firstIter++;

    // firstIter points to second page, secondIter points to first page
    double diff = cv::norm(*firstIter, *secondIter, NORM_INF);
    EXPECT_NE(diff, 0.);
}

TEST(Imgcodecs_Image, multipage_collection_backward_decoding)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";
    const string page_files[] = {
            root + "readwrite/multipage_p1.tif",
            root + "readwrite/multipage_p2.tif",
            root + "readwrite/multipage_p3.tif",
            root + "readwrite/multipage_p4.tif",
            root + "readwrite/multipage_p5.tif",
            root + "readwrite/multipage_p6.tif"
    };

    ImageCollection collection(filename, IMREAD_ANYCOLOR);
    EXPECT_EQ((size_t)6, collection.size());

    // backward decoding -> 5,4,3,2,1,0
    for(int i = (int)collection.size() - 1; i >= 0; --i)
    {
        cv::Mat ithPage = imread(page_files[i]);
        EXPECT_FALSE(ithPage.empty());
        double diff = cv::norm(collection[i], ithPage, NORM_INF);
        EXPECT_EQ(diff, 0.);
    }

    for(int i = 0; i < (int)collection.size(); ++i)
    {
        collection.releaseCache(i);
    }

    double diff = cv::norm(collection[2], imread(page_files[2]), NORM_INF);
    EXPECT_EQ(diff, 0.);
}

TEST(ImgCodecs, multipage_collection_decoding_range_based_for_loop_test)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";
    const string page_files[] = {
            root + "readwrite/multipage_p1.tif",
            root + "readwrite/multipage_p2.tif",
            root + "readwrite/multipage_p3.tif",
            root + "readwrite/multipage_p4.tif",
            root + "readwrite/multipage_p5.tif",
            root + "readwrite/multipage_p6.tif"
    };

    ImageCollection collection(filename, IMREAD_ANYCOLOR);

    size_t index = 0;
    for(auto &i: collection)
    {
        cv::Mat ithPage = imread(page_files[index]);
        EXPECT_FALSE(ithPage.empty());
        double diff = cv::norm(i, ithPage, NORM_INF);
        EXPECT_EQ(0., diff);
        ++index;
    }
    EXPECT_EQ(index, collection.size());

    index = 0;
    for(auto &&i: collection)
    {
        cv::Mat ithPage = imread(page_files[index]);
        EXPECT_FALSE(ithPage.empty());
        double diff = cv::norm(i, ithPage, NORM_INF);
        EXPECT_EQ(0., diff);
        ++index;
    }
    EXPECT_EQ(index, collection.size());
}

TEST(ImgCodecs, multipage_collection_two_iterator_operatorpp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";

    ImageCollection imcol(filename, IMREAD_ANYCOLOR);

    auto it0 = imcol.begin(), it1 = it0, it2 = it0;
    vector<Mat> img(6);
    for (int i = 0; i < 6; i++) {
        img[i] = *it0;
        it0->release();
        ++it0;
    }

    for (int i = 0; i < 3; i++) {
        ++it2;
    }

    for (int i = 0; i < 3; i++) {
         auto img2 = *it2;
         auto img1 = *it1;
         ++it2;
         ++it1;
         EXPECT_TRUE(cv::norm(img2, img[i+3], NORM_INF) == 0);
         EXPECT_TRUE(cv::norm(img1, img[i], NORM_INF) == 0);
    }
}

// See https://github.com/opencv/opencv/issues/26207
TEST(Imgcodecs, imencodemulti_regression_26207)
{
    vector<Mat> imgs;
    const cv::Mat img(100, 100, CV_8UC1, cv::Scalar::all(0));
    imgs.push_back(img);
    std::vector<uchar> buf;
    bool ret = false;

    // Encode single image
    EXPECT_NO_THROW(ret = imencode(".tiff", img, buf));
    EXPECT_TRUE(ret);
    EXPECT_NO_THROW(ret = imencode(".tiff", imgs, buf));
    EXPECT_TRUE(ret);
    EXPECT_NO_THROW(ret = imencodemulti(".tiff", imgs, buf));
    EXPECT_TRUE(ret);

    // Encode multiple images
    imgs.push_back(img.clone());
    EXPECT_NO_THROW(ret = imencode(".tiff", imgs, buf));
    EXPECT_TRUE(ret);
    EXPECT_NO_THROW(ret = imencodemulti(".tiff", imgs, buf));
    EXPECT_TRUE(ret);

    // Count stored images from buffer.
    // imcount() doesn't support buffer, so encoded buffer outputs to file temporary.
    const size_t len = buf.size();
    const string filename = cv::tempfile(".tiff");
    FILE *f = fopen(filename.c_str(), "wb");
    EXPECT_NE(f, nullptr);
    EXPECT_EQ(len, fwrite(&buf[0], 1, len, f));
    fclose(f);

    EXPECT_EQ(2, (int)imcount(filename));
    EXPECT_EQ(0, remove(filename.c_str()));
}
#endif

// See https://github.com/opencv/opencv/pull/26211
// ( related with https://github.com/opencv/opencv/issues/26207 )
TEST(Imgcodecs, imencode_regression_26207_extra)
{
    // CV_32F is not supported depth for BMP Encoder.
    // Encoded buffer contains CV_8U image which is fallbacked.
    const cv::Mat src(100, 100, CV_32FC1, cv::Scalar::all(0));
    std::vector<uchar> buf;
    bool ret = false;
    EXPECT_NO_THROW(ret = imencode(".bmp", src, buf));
    EXPECT_TRUE(ret);

    cv::Mat dst;
    EXPECT_NO_THROW(dst = imdecode(buf, IMREAD_GRAYSCALE));
    EXPECT_FALSE(dst.empty());
    EXPECT_EQ(CV_8UC1, dst.type());
}
TEST(Imgcodecs, imwrite_regression_26207_extra)
{
    // CV_32F is not supported depth for BMP Encoder.
    // Encoded buffer contains CV_8U image which is fallbacked.
    const cv::Mat src(100, 100, CV_32FC1, cv::Scalar::all(0));
    const string filename = cv::tempfile(".bmp");
    bool ret = false;
    EXPECT_NO_THROW(ret = imwrite(filename, src));
    EXPECT_TRUE(ret);

    cv::Mat dst;
    EXPECT_NO_THROW(dst = imread(filename, IMREAD_GRAYSCALE));
    EXPECT_FALSE(dst.empty());
    EXPECT_EQ(CV_8UC1, dst.type());
    EXPECT_EQ(0, remove(filename.c_str()));
}

TEST(Imgcodecs_Params, imwrite_regression_22752)
{
    const Mat img(16, 16, CV_8UC3, cv::Scalar::all(0));
    vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
//  params.push_back(100)); // Forget it.
    EXPECT_ANY_THROW(cv::imwrite("test.jpg", img, params));  // parameters size or missing JPEG codec
}

TEST(Imgcodecs_Params, imencode_regression_22752)
{
    const Mat img(16, 16, CV_8UC3, cv::Scalar::all(0));
    vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
//  params.push_back(100)); // Forget it.
    vector<uchar> buf;
    EXPECT_ANY_THROW(cv::imencode("test.jpg", img, buf, params));  // parameters size or missing JPEG codec
}

}} // namespace
