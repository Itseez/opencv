// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _GRFMT_JPEGXL_H_
#define _GRFMT_JPEGXL_H_

#ifdef HAVE_JPEGXL

#include "grfmt_base.hpp"
#include <jxl/decode_cxx.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <vector>
#include <memory>

// Jpeg XL codec

namespace cv
{

/**
* @brief JpegXL codec using libjxl library
*/

class JpegXLDecoder CV_FINAL : public BaseImageDecoder
{
public:

    JpegXLDecoder();
    virtual ~JpegXLDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();
    size_t signatureLength() const CV_OVERRIDE;
    bool checkSignature( const String& signature ) const CV_OVERRIDE;

    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:
    std::unique_ptr<FILE, int (*)(FILE*)> m_f;
    JxlDecoderPtr m_decoder;
    JxlThreadParallelRunnerPtr m_parallel_runner;
    JxlPixelFormat m_format;
    int m_convert;
    std::vector<uint8_t> m_read_buffer;
    JxlDecoderStatus m_status;

private:
    bool read(Mat* pimg);
};


class JpegXLEncoder CV_FINAL : public BaseImageEncoder
{
public:
    JpegXLEncoder();
    virtual ~JpegXLEncoder();

    bool isFormatSupported( int depth ) const CV_OVERRIDE;
    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;
};

}

#endif

#endif/*_GRFMT_JPEGXL_H_*/
