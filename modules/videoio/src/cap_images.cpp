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
// Copyright (C) 2008, Nils Hasler, all rights reserved.
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

// Author: Nils Hasler <hasler@mpi-inf.mpg.de>
//
//         Max-Planck-Institut Informatik

//
// capture video from a sequence of images
// the filename when opening can either be a printf pattern such as
// video%04d.png or the first frame of the sequence i.e. video0001.png
//

#include "precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/videoio/utils.private.hpp"

#if 0
#define CV_WARN(message)
#else
#define CV_WARN(message) CV_LOG_INFO(NULL, "CAP_IMAGES warning: %s (%s:%d)" << message)
#endif

namespace cv {

class CvCapture_Images: public IVideoCapture
{
public:
    void init()
    {
        filename_pattern.clear();
        frame.release();
        currentframe = firstframe = 0;
        length = 0;
        grabbedInOpen = false;
    }
    CvCapture_Images()
    {
        init();
    }
    CvCapture_Images(const String& _filename)
    {
        init();
        open(_filename);
    }

    virtual ~CvCapture_Images() CV_OVERRIDE
    {
        close();
    }
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int, OutputArray) CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE;
    virtual int getCaptureDomain() /*const*/ CV_OVERRIDE { return cv::CAP_IMAGES; }

    bool open(const String&);
    void close();
protected:
    std::string filename_pattern; // actually a printf-pattern
    unsigned currentframe;
    unsigned firstframe; // number of first frame
    unsigned length; // length of sequence

    Mat frame;
    bool grabbedInOpen;
};

void CvCapture_Images::close()
{
    init();
}

bool CvCapture_Images::grabFrame()
{
    if (length == 1 && currentframe >= length)
        return false;
    const cv::String filename = cv::format(filename_pattern.c_str(), (int)(firstframe + currentframe));
    CV_Assert(!filename.empty());

    if (grabbedInOpen)
    {
        grabbedInOpen = false;
        ++currentframe;

        return !frame.empty();
    }

    frame = imread(filename, IMREAD_UNCHANGED);
    if( !frame.empty() )
        currentframe++;

    return !frame.empty();
}

bool CvCapture_Images::retrieveFrame(int, OutputArray out)
{
    frame.copyTo(out);
    return grabbedInOpen ? false : !frame.empty();
}


double CvCapture_Images::getProperty(int id) const
{
    switch(id)
    {
    case cv::CAP_PROP_POS_MSEC:
        CV_WARN("collections of images don't have framerates");
        return 0;
    case cv::CAP_PROP_POS_FRAMES:
        return currentframe;
    case cv::CAP_PROP_FRAME_COUNT:
        return length;
    case cv::CAP_PROP_POS_AVI_RATIO:
        return (double)currentframe / (double)(length - 1);
    case cv::CAP_PROP_FRAME_WIDTH:
        return frame.cols;
    case cv::CAP_PROP_FRAME_HEIGHT:
        return frame.rows;
    case cv::CAP_PROP_FPS:
        CV_WARN("collections of images don't have framerates");
        return 1;
    case cv::CAP_PROP_FOURCC:
        CV_WARN("collections of images don't have 4-character codes");
        return 0;
    }
    return 0;
}

bool CvCapture_Images::setProperty(int id, double value)
{
    switch(id)
    {
    case cv::CAP_PROP_POS_MSEC:
    case cv::CAP_PROP_POS_FRAMES:
        if(value < 0) {
            CV_WARN("seeking to negative positions does not work - clamping");
            value = 0;
        }
        if(value >= length) {
            CV_WARN("seeking beyond end of sequence - clamping");
            value = length - 1;
        }
        currentframe = cvRound(value);
        if (currentframe != 0)
            grabbedInOpen = false; // grabbed frame is not valid anymore
        return true;
    case cv::CAP_PROP_POS_AVI_RATIO:
        if(value > 1) {
            CV_WARN("seeking beyond end of sequence - clamping");
            value = 1;
        } else if(value < 0) {
            CV_WARN("seeking to negative positions does not work - clamping");
            value = 0;
        }
        currentframe = cvRound((length - 1) * value);
        if (currentframe != 0)
            grabbedInOpen = false; // grabbed frame is not valid anymore
        return true;
    }
    CV_WARN("unknown/unhandled property");
    return false;
}

// static
std::string icvExtractPattern(const std::string& filename, unsigned *offset)
{
    size_t len = filename.size();
    CV_Assert(!filename.empty());
    CV_Assert(offset);

    *offset = 0;

    // check whether this is a valid image sequence filename
    std::string::size_type pos = filename.find('%');
    if (pos != std::string::npos)
    {
        pos++; CV_Assert(pos < len);
        if (filename[pos] == '0') // optional zero prefix
        {
            pos++; CV_Assert(pos < len);
        }
        if (filename[pos] >= '1' && filename[pos] <= '9') // optional numeric size (1..9) (one symbol only)
        {
            pos++; CV_Assert(pos < len);
        }
        if (filename[pos] == 'd' || filename[pos] == 'u')
        {
            pos++;
            if (pos == len)
                return filename;  // end of string '...%5d'
            CV_Assert(pos < len);
            if (filename.find('%', pos) == std::string::npos)
                return filename;  // no more patterns
            CV_Error_(Error::StsBadArg, ("CAP_IMAGES: invalid multiple patterns: %s", filename.c_str()));
        }
        CV_Error_(Error::StsBadArg, ("CAP_IMAGES: error, expected '0?[1-9][du]' pattern, got: %s", filename.c_str()));
    }
    else // no pattern filename was given - extract the pattern
    {
        pos = filename.rfind('/');
#ifdef _WIN32
        if (pos == std::string::npos)
            pos = filename.rfind('\\');
#endif
        if (pos != std::string::npos)
            pos++;
        else
            pos = 0;

        while (pos < len && !isdigit(filename[pos])) pos++;

        if (pos == len)
            return "";

        std::string::size_type pos0 = pos;

        const int64_t max_number = 1000000000;
        CV_Assert(max_number < INT_MAX); // offset is 'int'

        int number_str_size = 0;
        uint64_t number = 0;
        while (pos < len && isdigit(filename[pos]))
        {
            char ch = filename[pos];
            number = (number * 10) + (uint64_t)((int)ch - (int)'0');
            CV_Assert(number < max_number);
            number_str_size++;
            CV_Assert(number_str_size <= 64);  // don't allow huge zero prefixes
            pos++;
        }
        CV_Assert(number_str_size > 0);

        *offset = (int)number;

        std::string result;
        if (pos0 > 0)
            result += filename.substr(0, pos0);
        result += cv::format("%%0%dd", number_str_size);
        if (pos < len)
            result += filename.substr(pos);
        CV_LOG_INFO(NULL, "Pattern: " << result << " @ " << number);
        return result;
    }
}


bool CvCapture_Images::open(const std::string& _filename)
{
    unsigned offset = 0;
    close();

    CV_Assert(!_filename.empty());
    filename_pattern = icvExtractPattern(_filename, &offset);
    if (filename_pattern.empty())
    {
        filename_pattern = _filename;
        if (!utils::fs::exists(filename_pattern))
        {
            CV_LOG_INFO(NULL, "CAP_IMAGES: File does not exist: " << filename_pattern);
            close();
            return false;
        }
        if (!haveImageReader(filename_pattern))
        {
            CV_LOG_INFO(NULL, "CAP_IMAGES: File is not an image: " << filename_pattern);
            close();
            return false;
        }
        length = 1;
    }
    else
    {
        // determine the length of the sequence
        for (length = 0; ;)
        {
            cv::String filename = cv::format(filename_pattern.c_str(), (int)(offset + length));
            if (!utils::fs::exists(filename))
            {
                if (length == 0 && offset == 0) // allow starting with 0 or 1
                {
                    offset++;
                    continue;
                }
                CV_LOG_INFO(NULL, "CAP_IMAGES: File does not exist: " << filename);
                break;
            }

            if(!haveImageReader(filename))
            {
                CV_LOG_INFO(NULL, "CAP_IMAGES: File is not an image: " << filename);
                break;
            }

            length++;
        }

        if (length == 0)
        {
            close();
            return false;
        }

        firstframe = offset;
    }
    // grab frame to enable properties retrieval
    bool grabRes = CvCapture_Images::grabFrame();
    grabbedInOpen = true;
    currentframe = 0;
    return grabRes;
}

bool CvCapture_Images::isOpened() const
{
    return !filename_pattern.empty();
}

Ptr<IVideoCapture> create_Images_capture(const std::string &filename)
{
    return makePtr<CvCapture_Images>(filename);
}

//
//
// image sequence writer
//
//
class CvVideoWriter_Images CV_FINAL : public IVideoWriter
{
public:
    CvVideoWriter_Images(const std::string & _filename);
    void close();

    ~CvVideoWriter_Images() CV_OVERRIDE { close(); }
    double getProperty(int) const CV_OVERRIDE { return 0; }
    bool setProperty( int, double ) CV_OVERRIDE; // FIXIT doesn't work: IVideoWriter interface only!
    bool isOpened() const CV_OVERRIDE { return !filename_pattern.empty(); }
    void write( InputArray ) CV_OVERRIDE;
    int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_IMAGES; }
protected:
    std::string filename_pattern;
    unsigned currentframe;
    std::vector<int> params;
};

void CvVideoWriter_Images::write(InputArray image)
{
    CV_Assert(!filename_pattern.empty());
    cv::String filename = cv::format(filename_pattern.c_str(), (int)currentframe);
    CV_Assert(!filename.empty());

    std::vector<int> image_params = params;
    image_params.push_back(0); // append parameters 'stop' mark
    image_params.push_back(0);

    cv::Mat img = image.getMat();
    cv::imwrite(filename, img, image_params);
    currentframe++;
}

void CvVideoWriter_Images::close()
{
    filename_pattern.clear();
    currentframe = 0;
    params.clear();
}


CvVideoWriter_Images::CvVideoWriter_Images(const std::string & _filename)
{
    unsigned offset = 0;
    close();

    filename_pattern = icvExtractPattern(_filename, &offset);
    CV_Assert(!filename_pattern.empty());

    cv::String filename = cv::format(filename_pattern.c_str(), (int)currentframe);
    if (!cv::haveImageWriter(filename))
    {
        close();
    }

    currentframe = offset;
    params.clear();
}


bool CvVideoWriter_Images::setProperty( int id, double value )
{
    if (id >= cv::CAP_PROP_IMAGES_BASE && id < cv::CAP_PROP_IMAGES_LAST)
    {
        params.push_back( id - cv::CAP_PROP_IMAGES_BASE );
        params.push_back( static_cast<int>( value ) );
        return true;
    }
    return false; // not supported
}

Ptr<IVideoWriter> create_Images_writer(const std::string &filename, int, double, const Size &,
                                       const cv::VideoWriterParameters&)
{
    return makePtr<CvVideoWriter_Images>(filename);
}

} // cv::
