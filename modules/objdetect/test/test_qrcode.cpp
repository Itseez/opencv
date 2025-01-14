// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "test_qr_utils.hpp"
#include "opencv2/imgproc.hpp"

namespace opencv_test { namespace {

std::string qrcode_images_name[] = {
  "version_1_down.jpg", "version_1_left.jpg", "version_1_right.jpg", "version_1_up.jpg", "version_1_top.jpg",
  "version_2_down.jpg", "version_2_left.jpg", "version_2_right.jpg", "version_2_up.jpg", "version_2_top.jpg",
  "version_3_down.jpg", "version_3_left.jpg", "version_3_right.jpg", "version_3_up.jpg", "version_3_top.jpg",
  "version_4_down.jpg", "version_4_left.jpg", "version_4_right.jpg", "version_4_up.jpg", "version_4_top.jpg",
  "version_5_down.jpg", "version_5_left.jpg", /*"version_5_right.jpg",*/ "version_5_up.jpg", "version_5_top.jpg",
  "russian.jpg", "kanji.jpg", "link_github_ocv.jpg", "link_ocv.jpg", "link_wiki_cv.jpg"
// version_5_right.jpg DISABLED after tile fix, PR #22025
};

// Todo: fix corner align in big QRs to enable close_5.png
std::string qrcode_images_close[] = {
  "close_1.png", "close_2.png", "close_3.png", "close_4.png"//, "close_5.png"
};
std::string qrcode_images_monitor[] = {
  "monitor_1.png", "monitor_2.png", "monitor_3.png", "monitor_4.png", "monitor_5.png"
};
std::string qrcode_images_curved[] = {
  "curved_1.jpg", "curved_2.jpg", "curved_3.jpg", /*"curved_4.jpg",*/ "curved_5.jpg", /*"curved_6.jpg",*/ "curved_7.jpg", "curved_8.jpg"
};
// curved_4.jpg, curved_6.jpg DISABLED after tile fix, PR #22025
std::string qrcode_images_multiple[] = {
  "2_qrcodes.png", "3_close_qrcodes.png", "3_qrcodes.png", "4_qrcodes.png",
  "5_qrcodes.png", "6_qrcodes.png", "7_qrcodes.png", "8_close_qrcodes.png"
};

static std::set<std::pair<std::string, std::string>> disabled_samples = {{"5_qrcodes.png", "aruco_based"}};

//#define UPDATE_QRCODE_TEST_DATA
#ifdef  UPDATE_QRCODE_TEST_DATA

TEST(Objdetect_QRCode, generate_test_data)
{
    const std::string root = "qrcode/";
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "test_images" << "[";
    size_t images_count = sizeof(qrcode_images_name) / sizeof(qrcode_images_name[0]);
    for (size_t i = 0; i < images_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_name[i];
        std::string image_path = findDataFile(root + qrcode_images_name[i]);
        std::vector<Point> corners;
        Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
        std::string decoded_info;
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        EXPECT_TRUE(detectQRCode(src, corners));
        EXPECT_TRUE(decodeQRCode(src, corners, decoded_info, straight_barcode));

        file_config << "x" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
        file_config << "]";
        file_config << "y" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
        file_config << "]";
        file_config << "info" << decoded_info;
        file_config << "}";
    }
    file_config << "]";
    file_config.release();
}

TEST(Objdetect_QRCode_Close, generate_test_data)
{
    const std::string root = "qrcode/close/";
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "close_images" << "[";
    size_t close_count = sizeof(qrcode_images_close) / sizeof(qrcode_images_close[0]);
    for (size_t i = 0; i < close_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_close[i];
        std::string image_path = findDataFile(root + qrcode_images_close[i]);
        std::vector<Point> corners;
        Mat src = imread(image_path, IMREAD_GRAYSCALE), barcode, straight_barcode;
        std::string decoded_info;
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        const double min_side = std::min(src.size().width, src.size().height);
        double coeff_expansion = 1024.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR_EXACT);
        EXPECT_TRUE(detectQRCode(barcode, corners));
        EXPECT_TRUE(decodeQRCode(barcode, corners, decoded_info, straight_barcode));

        file_config << "x" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
        file_config << "]";
        file_config << "y" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
        file_config << "]";
        file_config << "info" << decoded_info;
        file_config << "}";
    }
    file_config << "]";
    file_config.release();
}
TEST(Objdetect_QRCode_Monitor, generate_test_data)
{
    const std::string root = "qrcode/monitor/";
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "monitor_images" << "[";
    size_t monitor_count = sizeof(qrcode_images_monitor) / sizeof(qrcode_images_monitor[0]);
    for (size_t i = 0; i < monitor_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_monitor[i];
        std::string image_path = findDataFile(root + qrcode_images_monitor[i]);
        std::vector<Point> corners;
        Mat src = imread(image_path, IMREAD_GRAYSCALE), barcode, straight_barcode;
        std::string decoded_info;
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        const double min_side = std::min(src.size().width, src.size().height);
        double coeff_expansion = 1024.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR_EXACT);
        EXPECT_TRUE(detectQRCode(barcode, corners));
        EXPECT_TRUE(decodeQRCode(barcode, corners, decoded_info, straight_barcode));

        file_config << "x" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
        file_config << "]";
        file_config << "y" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
        file_config << "]";
        file_config << "info" << decoded_info;
        file_config << "}";
    }
    file_config << "]";
    file_config.release();
}
TEST(Objdetect_QRCode_Curved, generate_test_data)
{
    const std::string root = "qrcode/curved/";
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "test_images" << "[";
    size_t images_count = sizeof(qrcode_images_curved) / sizeof(qrcode_images_curved[0]);
    for (size_t i = 0; i < images_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_curved[i];
        std::string image_path = findDataFile(root + qrcode_images_curved[i]);
        std::vector<Point> corners;
        Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
        std::string decoded_info;
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        EXPECT_TRUE(detectQRCode(src, corners));
        EXPECT_TRUE(decodeCurvedQRCode(src, corners, decoded_info, straight_barcode));

        file_config << "x" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
        file_config << "]";
        file_config << "y" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
        file_config << "]";
        file_config << "info" << decoded_info;
        file_config << "}";
    }
    file_config << "]";
    file_config.release();
}
TEST(Objdetect_QRCode_Multi, generate_test_data)
{
    const std::string root = "qrcode/multiple/";
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "multiple_images" << "[:";
    size_t multiple_count = sizeof(qrcode_images_multiple) / sizeof(qrcode_images_multiple[0]);
    for (size_t i = 0; i < multiple_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_multiple[i];
        std::string image_path = findDataFile(root + qrcode_images_multiple[i]);
        Mat src = imread(image_path);

        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        std::vector<Point> corners;
        QRCodeDetector qrcode;
        EXPECT_TRUE(qrcode.detectMulti(src, corners));
        std::vector<cv::String> decoded_info;
        std::vector<Mat> straight_barcode;
        EXPECT_TRUE(qrcode.decodeMulti(src, corners, decoded_info, straight_barcode));

        file_config << "x" << "[:";
        for(size_t j = 0; j < corners.size(); j += 4)
        {
            file_config << "[:";
            for (size_t k = 0; k < 4; k++)
            {
                file_config << corners[j + k].x;
            }
            file_config << "]";
        }
        file_config << "]";
        file_config << "y" << "[:";
        for(size_t j = 0; j < corners.size(); j += 4)
        {
            file_config << "[:";
            for (size_t k = 0; k < 4; k++)
            {
                file_config << corners[j + k].y;
            }
            file_config << "]";
        }
        file_config << "]";
        file_config << "info";
        file_config << "[:";

        for(size_t j = 0; j < decoded_info.size(); j++)
        {
            file_config << decoded_info[j];
        }
        file_config << "]";
        file_config << "}";
    }

    file_config << "]";
    file_config.release();
}

#else

typedef testing::TestWithParam< std::string > Objdetect_QRCode;
TEST_P(Objdetect_QRCode, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/";
    const int pixels_error = 3;

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector<Point> corners;
    std::string decoded_info;
    QRCodeDetector qrcode;
    decoded_info = qrcode.detectAndDecode(src, corners, straight_barcode);
    ASSERT_FALSE(corners.empty());
    ASSERT_FALSE(decoded_info.empty());
    int expected_barcode_type = CV_8UC1;
    EXPECT_EQ(expected_barcode_type, straight_barcode.type());
    check_qr(root, name_current_image, "test_images", corners, {decoded_info}, pixels_error);
}

typedef testing::TestWithParam< std::string > Objdetect_QRCode_Close;
TEST_P(Objdetect_QRCode_Close, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/close/";
    const int pixels_error = 3;

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), barcode, straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    const double min_side = std::min(src.size().width, src.size().height);
    double coeff_expansion = 1024.0 / min_side;
    const int width  = cvRound(src.size().width  * coeff_expansion);
    const int height = cvRound(src.size().height  * coeff_expansion);
    Size new_size(width, height);
    resize(src, barcode, new_size, 0, 0, INTER_LINEAR_EXACT);
    std::vector<Point> corners;
    std::string decoded_info;
    QRCodeDetector qrcode;
    decoded_info = qrcode.detectAndDecode(barcode, corners, straight_barcode);
    ASSERT_FALSE(corners.empty());
    ASSERT_FALSE(decoded_info.empty());
    int expected_barcode_type = CV_8UC1;
    EXPECT_EQ(expected_barcode_type, straight_barcode.type());
    check_qr(root, name_current_image, "close_images", corners, {decoded_info}, pixels_error);
}

typedef testing::TestWithParam< std::string > Objdetect_QRCode_Monitor;
TEST_P(Objdetect_QRCode_Monitor, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/monitor/";
    const int pixels_error = 3;

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), barcode, straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    const double min_side = std::min(src.size().width, src.size().height);
    double coeff_expansion = 1024.0 / min_side;
    const int width  = cvRound(src.size().width  * coeff_expansion);
    const int height = cvRound(src.size().height  * coeff_expansion);
    Size new_size(width, height);
    resize(src, barcode, new_size, 0, 0, INTER_LINEAR_EXACT);
    std::vector<Point> corners;
    std::string decoded_info;
    QRCodeDetector qrcode;
    decoded_info = qrcode.detectAndDecode(barcode, corners, straight_barcode);
    ASSERT_FALSE(corners.empty());
    ASSERT_FALSE(decoded_info.empty());
    int expected_barcode_type = CV_8UC1;
    EXPECT_EQ(expected_barcode_type, straight_barcode.type());
    check_qr(root, name_current_image, "monitor_images", corners, {decoded_info}, pixels_error);
}

typedef testing::TestWithParam< std::string > Objdetect_QRCode_Curved;
TEST_P(Objdetect_QRCode_Curved, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/curved/";
    const int pixels_error = 3;

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector<Point> corners;
    std::string decoded_info;
    QRCodeDetector qrcode;
    decoded_info = qrcode.detectAndDecodeCurved(src, corners, straight_barcode);
    ASSERT_FALSE(corners.empty());
    ASSERT_FALSE(decoded_info.empty());
    int expected_barcode_type = CV_8UC1;
    EXPECT_EQ(expected_barcode_type, straight_barcode.type());
    check_qr(root, name_current_image, "test_images", corners, {decoded_info}, pixels_error);
}

typedef testing::TestWithParam<std::tuple<std::string, std::string>> Objdetect_QRCode_Multi;
TEST_P(Objdetect_QRCode_Multi, regression)
{
    const std::string name_current_image = get<0>(GetParam());
    const std::string root = "qrcode/multiple/";
    const std::string method = get<1>(GetParam());
    const int pixels_error = 4;

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    if (disabled_samples.find({name_current_image, method}) != disabled_samples.end())
        throw SkipTestException(name_current_image + " is disabled sample for method " + method);
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<Point> corners;
    std::vector<cv::String> decoded_info;
    std::vector<Mat> straight_barcode;
    EXPECT_TRUE(qrcode.detectAndDecodeMulti(src, decoded_info, corners, straight_barcode));
    ASSERT_FALSE(corners.empty());
    ASSERT_FALSE(decoded_info.empty());
    int expected_barcode_type = CV_8UC1;
    for(size_t i = 0; i < straight_barcode.size(); i++)
        EXPECT_EQ(expected_barcode_type, straight_barcode[i].type());
    check_qr(root, name_current_image, "multiple_images", corners, decoded_info, pixels_error, true);
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode, testing::ValuesIn(qrcode_images_name));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Close, testing::ValuesIn(qrcode_images_close));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Monitor, testing::ValuesIn(qrcode_images_monitor));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Curved, testing::ValuesIn(qrcode_images_curved));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Multi, testing::Combine(testing::ValuesIn(qrcode_images_multiple),
    testing::Values("contours_based", "aruco_based")));

TEST(Objdetect_QRCode_decodeMulti, decode_regression_16491)
{
    Mat zero_image = Mat::zeros(256, 256, CV_8UC1);
    Point corners_[] = {Point(16, 16), Point(128, 16), Point(128, 128), Point(16, 128),
                        Point(16, 16), Point(128, 16), Point(128, 128), Point(16, 128)};
    std::vector<Point> vec_corners;
    int array_size = 8;
    vec_corners.assign(corners_, corners_ + array_size);
    std::vector<cv::String> decoded_info;
    std::vector<Mat> straight_barcode;
    QRCodeDetector vec_qrcode;
    EXPECT_NO_THROW(vec_qrcode.decodeMulti(zero_image, vec_corners, decoded_info, straight_barcode));

    Mat mat_corners(2, 4, CV_32SC2, (void*)&vec_corners[0]);
    QRCodeDetector mat_qrcode;
    EXPECT_NO_THROW(mat_qrcode.decodeMulti(zero_image, mat_corners, decoded_info, straight_barcode));
}

typedef testing::TestWithParam<std::string> Objdetect_QRCode_detectMulti;
TEST_P(Objdetect_QRCode_detectMulti, detect_regression_16961)
{
    const std::string method = GetParam();
    const std::string name_current_image = "9_qrcodes.jpg";
    const std::string root = "qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<Point> corners;
    EXPECT_TRUE(qrcode.detectMulti(src, corners));
    ASSERT_FALSE(corners.empty());
    size_t expect_corners_size = 36;
    EXPECT_EQ(corners.size(), expect_corners_size);
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_detectMulti, testing::Values("contours_based", "aruco_based"));
typedef testing::TestWithParam<std::string> Objdetect_QRCode_detectAndDecodeMulti;
TEST_P(Objdetect_QRCode_detectAndDecodeMulti, check_output_parameters_type_19363)
{
    const std::string name_current_image = "9_qrcodes.jpg";
    const std::string root = "qrcode/multiple/";
    const std::string method = GetParam();

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<Point> corners;
    std::vector<cv::String> decoded_info;
#if 0  // FIXIT: OutputArray::create() type check
    std::vector<Mat2b> straight_barcode_nchannels;
    EXPECT_ANY_THROW(qrcode->detectAndDecodeMulti(src, decoded_info, corners, straight_barcode_nchannels));
#endif

    int expected_barcode_type = CV_8UC1;
    std::vector<Mat1b> straight_barcode;
    EXPECT_TRUE(qrcode.detectAndDecodeMulti(src, decoded_info, corners, straight_barcode));
    ASSERT_FALSE(corners.empty());
    for(size_t i = 0; i < straight_barcode.size(); i++)
        EXPECT_EQ(expected_barcode_type, straight_barcode[i].type());
}
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_detectAndDecodeMulti, testing::Values("contours_based", "aruco_based"));


TEST(Objdetect_QRCode_detect, detect_regression_20882)
{
    const std::string name_current_image = "qrcode_near_the_end.jpg";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    QRCodeDetector qrcode;
    std::vector<Point> corners;
    Mat straight_barcode;
    cv::String decoded_info;
    EXPECT_TRUE(qrcode.detect(src, corners));
    EXPECT_TRUE(!corners.empty());
    EXPECT_NO_THROW(qrcode.decode(src, corners, straight_barcode));
}

TEST(Objdetect_QRCode_basic, not_found_qrcode)
{
    std::vector<Point> corners;
    Mat straight_barcode;
    std::string decoded_info;
    Mat zero_image = Mat::zeros(256, 256, CV_8UC1);
    QRCodeDetector qrcode;
    EXPECT_FALSE(qrcode.detect(zero_image, corners));
    corners = std::vector<Point>(4);
    EXPECT_ANY_THROW(qrcode.decode(zero_image, corners, straight_barcode));
}

TEST(Objdetect_QRCode_detect, detect_regression_21287)
{
    const std::string name_current_image = "issue_21287.png";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    QRCodeDetector qrcode;
    std::vector<Point> corners;
    Mat straight_barcode;
    cv::String decoded_info;
    EXPECT_TRUE(qrcode.detect(src, corners));
    EXPECT_TRUE(!corners.empty());
    EXPECT_NO_THROW(qrcode.decode(src, corners, straight_barcode));
}

TEST(Objdetect_QRCode_detect_flipped, regression_23249)
{

    const std::vector<std::pair<std::string, std::string>> flipped_images =
    // image name , expected result
    {{"flipped_1.png", "The key is /qrcod_OMevpf"},
     {"flipped_2.png", "A26"}};

    const std::string root = "qrcode/flipped/";

    for(const auto &flipped_image : flipped_images){
        const std::string &image_name = flipped_image.first;

        std::string image_path = findDataFile(root + image_name);
        Mat src = imread(image_path);
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        QRCodeDetector qrcode;
        std::vector<Point> corners;
        Mat straight_barcode;
        cv::String decoded_info;
        EXPECT_TRUE(qrcode.detect(src, corners));
        EXPECT_TRUE(!corners.empty());
        std::string decoded_msg;
        const std::string &expect_msg = flipped_image.second;
        EXPECT_NO_THROW(decoded_msg = qrcode.decode(src, corners, straight_barcode));
        ASSERT_FALSE(straight_barcode.empty()) << "Can't decode qrimage.";
        EXPECT_EQ(expect_msg, decoded_msg);
    }
}

// @author Kumataro, https://github.com/Kumataro
TEST(Objdetect_QRCode_decode, decode_regression_21929)
{
    const cv::String expect_msg = "OpenCV";
    Mat qrImg;
    QRCodeEncoder::Params params;
    params.version = 8; // 49x49
    Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
    qrcode_enc->encode(expect_msg, qrImg);

    Mat src;
    cv::resize(qrImg, src, Size(200,200), 1.0, 1.0, INTER_NEAREST);

    QRCodeDetector qrcode;
    std::vector<Point> corners;
    Mat straight_barcode;

    EXPECT_TRUE(qrcode.detect(src, corners));
    EXPECT_TRUE(!corners.empty());
    cv::String decoded_msg;
    EXPECT_NO_THROW(decoded_msg = qrcode.decode(src, corners, straight_barcode));
    ASSERT_FALSE(straight_barcode.empty()) << "Can't decode qrimage.";
    EXPECT_EQ(expect_msg, decoded_msg);
}

TEST(Objdetect_QRCode_decode, decode_regression_version_25)
{
    const cv::String expect_msg = "OpenCV";
    Mat qrImg;
    QRCodeEncoder::Params params;
    params.version = 25; // 117x117
    Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
    qrcode_enc->encode(expect_msg, qrImg);

    Mat src;
    cv::resize(qrImg, src, qrImg.size()*3, 1.0, 1.0, INTER_NEAREST);

    QRCodeDetector qrcode;
    std::vector<Point> corners;
    Mat straight_barcode;

    EXPECT_TRUE(qrcode.detect(src, corners));
    EXPECT_TRUE(!corners.empty());

    cv::String decoded_msg;
    EXPECT_NO_THROW(decoded_msg = qrcode.decode(src, corners, straight_barcode));
    ASSERT_FALSE(straight_barcode.empty()) << "Can't decode qrimage.";
    EXPECT_EQ(expect_msg, decoded_msg);
}

TEST_P(Objdetect_QRCode_detectAndDecodeMulti, decode_9_qrcodes_version7)
{
    const std::string name_current_image = "9_qrcodes_version7.jpg";
    const std::string root = "qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    const std::string method = GetParam();
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<Point> corners;
    std::vector<cv::String> decoded_info;

    std::vector<Mat1b> straight_barcode;
    qrcode.detectAndDecodeMulti(src, decoded_info, corners, straight_barcode);
    EXPECT_EQ(9ull, decoded_info.size());
    const string gold_info = "I love OpenCV, QR Code version = 7, error correction = level Quartile";
    for (const auto& info : decoded_info) {
        EXPECT_EQ(info, gold_info);
    }
}

#endif // UPDATE_QRCODE_TEST_DATA

TEST(Objdetect_QRCode_detectAndDecode, utf8_output)
{
    const std::string name_current_image = "umlaut.png";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    QRCodeDetector qrcode;
    std::vector<Point> corners;
    Mat straight;
    std::string decoded_info = qrcode.detectAndDecode(src, corners, straight);
    EXPECT_FALSE(decoded_info.empty());
    EXPECT_NE(decoded_info.find("M\xc3\xbcllheimstrasse"), std::string::npos);
}

TEST_P(Objdetect_QRCode_detectAndDecodeMulti, detect_regression_24679)
{
    const std::string name_current_image = "issue_24679.png";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat img = imread(image_path);
    const std::string method = GetParam();
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<cv::String> decoded_info;
    ASSERT_TRUE(qrcode.detectAndDecodeMulti(img, decoded_info));
    EXPECT_EQ(decoded_info.size(), 4U);
}

TEST_P(Objdetect_QRCode_detectAndDecodeMulti, detect_regression_24011)
{
    const std::string name_current_image = "issue_24011.jpg";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat img = imread(image_path);
    const std::string method = GetParam();
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<cv::String> decoded_info;
    ASSERT_TRUE(qrcode.detectAndDecodeMulti(img, decoded_info));
    EXPECT_EQ(decoded_info.size(), 2U);
}

TEST(Objdetect_QRCode_detect, detect_regression_24450)
{
    const std::string name_current_image = "issue_24450.png";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat img = imread(image_path);
    GraphicalCodeDetector qrcode = QRCodeDetector();
    std::vector<Point2f> points;
    ASSERT_TRUE(qrcode.detect(img, points));
    EXPECT_EQ(points.size(), 4U);
    img.at<Vec3b>(img.rows - 1, 296) = {};
    ASSERT_TRUE(qrcode.detect(img, points));
    EXPECT_EQ(points.size(), 4U);
}

TEST(Objdetect_QRCode_detect, detect_regression_22892)
{
    const std::string name_current_image = "issue_22892.png";
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat img = imread(image_path);

    QRCodeDetector qrcode;
    std::vector<Point> corners;
    Mat straight_code;
    qrcode.detectAndDecodeCurved(img, corners, straight_code);
    EXPECT_EQ(corners.size(), 4U);
}

}} // namespace
