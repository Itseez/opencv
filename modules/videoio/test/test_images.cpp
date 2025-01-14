// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio/utils.private.hpp"

using namespace std;

namespace opencv_test { namespace {

struct ImageCollection
{
    string dirname;
    string base;
    string ext;
    size_t first_idx;
    size_t last_idx;
    size_t width;
public:
    ImageCollection(const char *dirname_template = "opencv_test_images")
        : first_idx(0), last_idx(0), width(0)
    {
        dirname = cv::tempfile(dirname_template);
        cv::utils::fs::createDirectory(dirname);
    }
    ~ImageCollection()
    {
        cleanup();
    }
    void cleanup()
    {
        cv::utils::fs::remove_all(dirname);
    }
    void generate(size_t count, size_t first = 0, size_t width_ = 4, const string & base_ = "test", const string & ext_ = "png")
    {
        base = base_;
        ext = ext_;
        first_idx = first;
        last_idx = first + count - 1;
        width = width_;
        for (size_t idx = first_idx; idx <= last_idx; ++idx)
        {
            const string filename = getFilename(idx);
            imwrite(filename, getFrame(idx));
        }
    }
    string getFilename(size_t idx = 0) const
    {
        ostringstream buf;
        buf << dirname << "/" << base << setw(width) << setfill('0') << idx << "." << ext;
        return buf.str();
    }
    string getPatternFilename() const
    {
        ostringstream buf;
        buf << dirname << "/" << base << "%0" << width << "d" << "." << ext;
        return buf.str();
    }
    string getFirstFilename() const
    {
        return getFilename(first_idx);
    }
    Mat getFirstFrame() const
    {
        return getFrame(first_idx);
    }
    size_t getCount() const
    {
        return last_idx - first_idx + 1;
    }
    string getDirname() const
    {
        return dirname;
    }
    static Mat getFrame(size_t idx)
    {
        const int sz = 100; // 100x100 or bigger
        Mat res(sz, sz, CV_8UC3, Scalar::all(0));
        circle(res, Point(idx % 100), idx % 50, Scalar::all(255), 2, LINE_8);
        return res;
    }
};

//==================================================================================================

TEST(videoio_images, basic_read)
{
    ImageCollection col;
    col.generate(20);
    VideoCapture cap(col.getFirstFilename(), CAP_IMAGES);
    ASSERT_TRUE(cap.isOpened());
    size_t idx = 0;
    while (cap.isOpened()) // TODO: isOpened is always true, even if there are no more images
    {
        Mat img;
        const bool read_res = cap.read(img);
        if (!read_res)
            break;
        EXPECT_MAT_N_DIFF(img, col.getFrame(idx), 0);
        ++idx;
    }
    EXPECT_EQ(col.getCount(), idx);
}

TEST(videoio_images, basic_write)
{
    // writer should create files: test0000.png, ... test0019.png
    ImageCollection col;
    col.generate(1);
    VideoWriter wri(col.getFirstFilename(), CAP_IMAGES, 0, 0, col.getFrame(0).size());
    ASSERT_TRUE(wri.isOpened());
    size_t idx = 0;
    while (wri.isOpened())
    {
        wri << col.getFrame(idx);
        Mat actual = imread(col.getFilename(idx));
        EXPECT_MAT_N_DIFF(col.getFrame(idx), actual, 0);
        if (++idx >= 20)
            break;
    }
    wri.release();
    ASSERT_FALSE(wri.isOpened());
}

TEST(videoio_images, bad)
{
    ImageCollection col;
    {
        ostringstream buf; buf << col.getDirname() << "/missing0000.png";
        VideoCapture cap(buf.str(), CAP_IMAGES);
        EXPECT_FALSE(cap.isOpened());
        Mat img;
        EXPECT_FALSE(cap.read(img));
    }
}

TEST(videoio_images, seek)
{
    // check files: test0005.png, ..., test0024.png
    // seek to valid and invalid frame numbers
    // position is zero-based: valid frame numbers are 0, ..., 19
    const int count = 20;
    ImageCollection col;
    col.generate(count, 5);
    VideoCapture cap(col.getFirstFilename(), CAP_IMAGES);
    ASSERT_TRUE(cap.isOpened());
    EXPECT_EQ((size_t)count, (size_t)cap.get(CAP_PROP_FRAME_COUNT));
    vector<int> positions { count / 2, 0, 1, count - 1, count, count + 100, -1, -100 };
    for (const auto &pos : positions)
    {
        Mat img;
        const bool res = cap.set(CAP_PROP_POS_FRAMES, pos);
        if (pos >= count || pos < 0) // invalid position
        {
//            EXPECT_FALSE(res); // TODO: backend clamps invalid value to valid range, actual result is 'true'
        }
        else
        {
            EXPECT_TRUE(res);
            EXPECT_GE(1., cap.get(CAP_PROP_POS_AVI_RATIO));
            EXPECT_NEAR((double)pos / (count - 1),  cap.get(CAP_PROP_POS_AVI_RATIO), 1e-2);
            EXPECT_EQ(pos, static_cast<decltype(pos)>(cap.get(CAP_PROP_POS_FRAMES)));
            EXPECT_TRUE(cap.read(img));
            EXPECT_MAT_N_DIFF(img, col.getFrame(col.first_idx + pos), 0);
        }
    }
}

TEST(videoio_images, pattern_overflow)
{
    // check files: test0.png, ..., test11.png
    ImageCollection col;
    col.generate(12, 0, 1);

    {
        VideoCapture cap(col.getFirstFilename(), CAP_IMAGES);
        ASSERT_TRUE(cap.isOpened());
        for (size_t idx = col.first_idx; idx <= col.last_idx; ++idx)
        {
            Mat img;
            EXPECT_TRUE(cap.read(img));
            EXPECT_MAT_N_DIFF(img, col.getFrame(idx), 0);
        }
    }
    {
        VideoCapture cap(col.getPatternFilename(), CAP_IMAGES);
        ASSERT_TRUE(cap.isOpened());
        for (size_t idx = col.first_idx; idx <= col.last_idx; ++idx)
        {
            Mat img;
            EXPECT_TRUE(cap.read(img));
            EXPECT_MAT_N_DIFF(img, col.getFrame(idx), 0);
        }
    }
}

TEST(videoio_images, pattern_max)
{
    // max supported number width for starting image is 9 digits
    // but following images can be read as well
    // test999999999.png ; test1000000000.png
    ImageCollection col;
    col.generate(2, 1000000000 - 1);
    {
        VideoCapture cap(col.getFirstFilename(), CAP_IMAGES);
        ASSERT_TRUE(cap.isOpened());
        Mat img;
        EXPECT_TRUE(cap.read(img));
        EXPECT_MAT_N_DIFF(img, col.getFrame(col.first_idx), 0);
        EXPECT_TRUE(cap.read(img));
        EXPECT_MAT_N_DIFF(img, col.getFrame(col.first_idx + 1), 0);
    }
    {
        VideoWriter wri(col.getFirstFilename(), CAP_IMAGES, 0, 0, col.getFirstFrame().size());
        ASSERT_TRUE(wri.isOpened());
        Mat img = col.getFrame(0);
        wri.write(img);
        wri.write(img);
        Mat actual;
        actual = imread(col.getFilename(col.first_idx));
        EXPECT_MAT_N_DIFF(actual, img, 0);
        actual = imread(col.getFilename(col.first_idx));
        EXPECT_MAT_N_DIFF(actual, img, 0);
    }
}

TEST(videoio_images, extract_pattern)
{
    unsigned offset = 0;

    // Min and max values
    EXPECT_EQ("%01d.png", cv::icvExtractPattern("0.png", &offset));
    EXPECT_EQ(0u, offset);
    EXPECT_EQ("%09d.png", cv::icvExtractPattern("999999999.png", &offset));
    EXPECT_EQ(999999999u, offset);

    // Regular usage - start, end, middle
    EXPECT_EQ("abc%04ddef.png", cv::icvExtractPattern("abc0048def.png", &offset));
    EXPECT_EQ(48u, offset);
    EXPECT_EQ("%05dabcdef.png", cv::icvExtractPattern("00049abcdef.png", &offset));
    EXPECT_EQ(49u, offset);
    EXPECT_EQ("abcdef%06d.png", cv::icvExtractPattern("abcdef000050.png", &offset));
    EXPECT_EQ(50u, offset);

    // Minus handling (should not handle)
    EXPECT_EQ("abcdef-%01d.png", cv::icvExtractPattern("abcdef-8.png", &offset));
    EXPECT_EQ(8u, offset);

    // Two numbers (should select first)
    // TODO: shouldn't it be last number?
    EXPECT_EQ("%01d-abcdef-8.png", cv::icvExtractPattern("7-abcdef-8.png", &offset));
    EXPECT_EQ(7u, offset);

    // Paths (should select filename)
    EXPECT_EQ("images005/abcdef%03d.png", cv::icvExtractPattern("images005/abcdef006.png", &offset));
    EXPECT_EQ(6u, offset);
    // TODO: fix
    // EXPECT_EQ("images03\\abcdef%02d.png", cv::icvExtractPattern("images03\\abcdef04.png", &offset));
    // EXPECT_EQ(4, offset);
    EXPECT_EQ("/home/user/test/0/3348/../../3442/./0/1/3/4/5/14304324234/%01d.png",
              cv::icvExtractPattern("/home/user/test/0/3348/../../3442/./0/1/3/4/5/14304324234/2.png", &offset));
    EXPECT_EQ(2u, offset);

    // Patterns '%0?[0-9][du]'
    EXPECT_EQ("test%d.png", cv::icvExtractPattern("test%d.png", &offset));
    EXPECT_EQ(0u, offset);
    EXPECT_EQ("test%0d.png", cv::icvExtractPattern("test%0d.png", &offset));
    EXPECT_EQ(0u, offset);
    EXPECT_EQ("test%09d.png", cv::icvExtractPattern("test%09d.png", &offset));
    EXPECT_EQ(0u, offset);
    EXPECT_EQ("test%5u.png", cv::icvExtractPattern("test%5u.png", &offset));
    EXPECT_EQ(0u, offset);

    // Invalid arguments
    EXPECT_THROW(cv::icvExtractPattern(string(), &offset), cv::Exception);
    // TODO: fix?
    // EXPECT_EQ(0u, offset);
    EXPECT_THROW(cv::icvExtractPattern("test%010d.png", &offset), cv::Exception);
    EXPECT_EQ(0u, offset);
    EXPECT_THROW(cv::icvExtractPattern("1000000000.png", &offset), cv::Exception);
    EXPECT_EQ(0u, offset);
    EXPECT_THROW(cv::icvExtractPattern("1.png", NULL), cv::Exception);
}

TEST(videoio_images, bug_26457)
{
    ImageCollection col;
    col.generate(1u);
    ASSERT_EQ(col.getCount(), 1u);

    VideoCapture cap(col.getFirstFilename(), CAP_IMAGES);
    ASSERT_TRUE(cap.isOpened());

    Mat img;
    const bool read_res = cap.read(img);
    EXPECT_TRUE(read_res);
    EXPECT_MAT_N_DIFF(img, col.getFirstFrame(), 0);
}

// TODO: should writer overwrite files?
// TODO: is clamping good for seeking?
// TODO: missing files? E.g. 3, 4, 6, 7, 8 (should it finish OR jump over OR return empty frame?)
// TODO: non-numbered files (https://github.com/opencv/opencv/pull/23815)
// TODO: when opening with pattern (e.g. test%01d.png), first frame can be only 0 (test0.png)

}} // opencv_test::<anonymous>::
