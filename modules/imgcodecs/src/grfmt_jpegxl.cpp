// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "grfmt_jpegxl.hpp"

#ifdef HAVE_JPEGXL

#include <jxl/encode_cxx.h>
#include <jxl/version.h>
#include <opencv2/core/utils/logger.hpp>

namespace cv
{

/////////////////////// JpegXLDecoder ///////////////////

JpegXLDecoder::JpegXLDecoder() : m_f(nullptr, &fclose)
{
    m_signature = "\xFF\x0A";
    m_decoder = nullptr;
    m_buf_supported = false;
    m_type = m_convert = -1;
    m_status = JXL_DEC_NEED_MORE_INPUT;
}

JpegXLDecoder::~JpegXLDecoder()
{
    close();
}

void JpegXLDecoder::close()
{
    if (m_decoder)
        m_decoder.release();
    if (m_f)
        m_f.release();
    m_read_buffer = {};
    m_width = m_height = 0;
    m_type = m_convert = -1;
    m_status = JXL_DEC_NEED_MORE_INPUT;
}

// see https://github.com/libjxl/libjxl/blob/v0.10.0/doc/format_overview.md
size_t JpegXLDecoder::signatureLength() const
{
    return 12; // For an ISOBMFF-based container
}

bool JpegXLDecoder::checkSignature( const String& signature ) const
{
    // A "naked" codestream.
    if (
        ( signature.size() >= 2 ) &&
        ( memcmp( signature.c_str(), "\xFF\x0A", 2 ) == 0 )
    )
    {
        return true;
    }

    // An ISOBMFF-based container.
    // 0x0000_000C_4A58_4C20_0D0A_870A.
    if (
        ( signature.size() >= 12 ) &&
        ( memcmp( signature.c_str(), "\x00\x00\x00\x0C\x4A\x58\x4C\x20\x0D\x0A\x87\x0A", 12 ) == 0 )
    )
    {
        return true;
    }

    return false;
}

ImageDecoder JpegXLDecoder::newDecoder() const
{
    return makePtr<JpegXLDecoder>();
}

bool JpegXLDecoder::read(Mat* pimg)
{
    // Open file
    if (!m_f) {
        m_f.reset(fopen(m_filename.c_str(), "rb"));
        if (!m_f)
            return false;
    }

    // Initialize decoder
    if (!m_decoder) {
        m_decoder = JxlDecoderMake(nullptr);
        if (!m_decoder)
            return false;
        // Subscribe to the basic info event
        JxlDecoderStatus status = JxlDecoderSubscribeEvents(m_decoder.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE);
        if (status != JXL_DEC_SUCCESS)
            return false;
    }

    // Set up parallel m_parallel_runner
    if (!m_parallel_runner) {
        m_parallel_runner = JxlThreadParallelRunnerMake(nullptr, cv::getNumThreads());
        if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(m_decoder.get(),
                                                            JxlThreadParallelRunner,
                                                            m_parallel_runner.get())) {
            return false;
        }
    }

    // Create buffer for reading
    const size_t read_buffer_size = 16384;  // 16KB chunks
    if (m_read_buffer.capacity() < read_buffer_size)
        m_read_buffer.resize(read_buffer_size);

    // Create image if needed
    if (m_type != -1 && pimg) {
        pimg->create(m_height, m_width, m_type);
        if (!pimg->isContinuous())
            return false;
        if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(m_decoder.get(),
                                                           &m_format,
                                                           pimg->ptr<uint8_t>(),
                                                           pimg->total() * pimg->elemSize())) {
            return false;
        }
    }

    // Start decoding loop
    do {
        // Check if we need more input
        if (m_status == JXL_DEC_NEED_MORE_INPUT) {
            size_t remaining = JxlDecoderReleaseInput(m_decoder.get());
            // Move any remaining bytes to the beginning
            if (remaining > 0)
                memmove(m_read_buffer.data(), m_read_buffer.data() + m_read_buffer.size() - remaining, remaining);
            // Read more data from file
            size_t bytes_read = fread(m_read_buffer.data() + remaining,
                                    1, m_read_buffer.size() - remaining, m_f.get());
            if (bytes_read == 0) {
                if (ferror(m_f.get())) {
                    CV_LOG_WARNING(NULL, "Error reading input file");
                    return false;
                }
                // If we reached EOF but decoder needs more input, file is truncated
                if (m_status == JXL_DEC_NEED_MORE_INPUT) {
                    CV_LOG_WARNING(NULL, "Truncated JXL file");
                    return false;
                }
            }

            // Set input buffer
            if (JXL_DEC_SUCCESS != JxlDecoderSetInput(m_decoder.get(),
                                                      m_read_buffer.data(),
                                                      bytes_read + remaining)) {
                return false;
            }
        }

        // Get the next decoder status
        m_status = JxlDecoderProcessInput(m_decoder.get());

        // Handle different decoder states
        switch (m_status) {
            case JXL_DEC_BASIC_INFO: {
                if (m_type != -1)
                    return false;
                JxlBasicInfo info;
                if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(m_decoder.get(), &info))
                    return false;

                // total channels (Color + Alpha)
                const uint32_t ncn = info.num_color_channels + info.num_extra_channels;

                m_width = info.xsize;
                m_height = info.ysize;
                m_format = {
                    ncn,
                    JXL_TYPE_UINT8, // (temporary)
                    JXL_LITTLE_ENDIAN, // endianness
                    0 // align stride to bytes
                };
                if (!m_use_rgb) {
                    switch (ncn) {
                    case 3:
                        m_convert = cv::COLOR_RGB2BGR;
                        break;
                    case 4:
                        m_convert = cv::COLOR_RGBA2BGRA;
                        break;
                    default:
                        m_convert = -1;
                    }
                }
                if (info.exponent_bits_per_sample > 0) {
                    m_format.data_type = JXL_TYPE_FLOAT;
                    m_type = CV_MAKETYPE( CV_32F, ncn );
                } else {
                    switch (info.bits_per_sample) {
                        case 8:
                            m_format.data_type = JXL_TYPE_UINT8;
                            m_type = CV_MAKETYPE( CV_8U, ncn );
                            break;
                        case 16:
                            m_format.data_type = JXL_TYPE_UINT16;
                            m_type = CV_MAKETYPE( CV_16U, ncn );
                            break;
                        default:
                            return false;
                    }
                }
                if (!pimg)
                    return true;
                break;
            }
            case JXL_DEC_FULL_IMAGE: {
                // Image is ready
                if (m_convert != -1)
                    cv::cvtColor(*pimg, *pimg, m_convert);
                break;
            }
            case JXL_DEC_ERROR: {
                close();
                return false;
            }
            default:
                break;
        }
    } while (m_status != JXL_DEC_SUCCESS);

    return true;
}

bool JpegXLDecoder::readHeader()
{
    close();
    return read(nullptr);
}

bool JpegXLDecoder::readData(Mat& img)
{
    if (!m_decoder || m_width == 0 || m_height == 0)
        return false;
    return read(&img);
}

/////////////////////// JpegXLEncoder ///////////////////

JpegXLEncoder::JpegXLEncoder()
{
    m_description = "JPEG XL files (*.jxl)";
    m_buf_supported = true;
}

JpegXLEncoder::~JpegXLEncoder()
{
}

ImageEncoder JpegXLEncoder::newEncoder() const
{
    return makePtr<JpegXLEncoder>();
}

bool JpegXLEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U || depth == CV_32F;
}

bool JpegXLEncoder::write(const Mat& img, const std::vector<int>& params)
{
    m_last_error.clear();

    JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
    if (!encoder)
        return false;

    JxlThreadParallelRunnerPtr runner = JxlThreadParallelRunnerMake(
        /*memory_manager=*/nullptr, cv::getNumThreads());
    if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(encoder.get(),  JxlThreadParallelRunner, runner.get()))
        return false;

    CV_CheckDepth(img.depth(),
             ( img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F ),
             "JPEG XL encoder only supports CV_8U, CV_16U, CV_32F");
    CV_CheckChannels(img.channels(),
             ( img.channels() == 1 || img.channels() == 3 || img.channels() == 4) ,
             "JPEG XL encoder only supports 1, 3, 4 channels");

    WLByteStream strm;
    if( m_buf ) {
        if( !strm.open( *m_buf ) )
            return false;
    }
    else if( !strm.open( m_filename )) {
        return false;
    }

    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = img.cols;
    info.ysize = img.rows;
    info.uses_original_profile = JXL_FALSE;

    if( img.channels() == 4 )
    {
        info.num_color_channels = 3;
        info.num_extra_channels = 1;

        info.bits_per_sample =
        info.alpha_bits      = 8 * static_cast<int>(img.elemSize1());

        info.exponent_bits_per_sample =
        info.alpha_exponent_bits      = img.depth() == CV_32F ? 8 : 0;
    }else{
        info.num_color_channels = img.channels();
        info.bits_per_sample = 8 * static_cast<int>(img.elemSize1());
        info.exponent_bits_per_sample = img.depth() == CV_32F ? 8 : 0;
    }

    if (JxlEncoderSetBasicInfo(encoder.get(), &info) != JXL_ENC_SUCCESS)
        return false;

    JxlDataType type = JXL_TYPE_UINT8;
    if (img.depth() == CV_32F)
        type = JXL_TYPE_FLOAT;
    else if (img.depth() == CV_16U)
        type = JXL_TYPE_UINT16;
    JxlPixelFormat format = {(uint32_t)img.channels(), type, JXL_NATIVE_ENDIAN, 0};
    JxlColorEncoding color_encoding = {};
    JXL_BOOL is_gray(format.num_channels < 3 ? JXL_TRUE : JXL_FALSE);
    JxlColorEncodingSetToSRGB(&color_encoding, is_gray);
    if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(encoder.get(), &color_encoding))
        return false;

    Mat image;
    switch ( img.channels() ) {
    case 3:
        cv::cvtColor(img, image, cv::COLOR_BGR2RGB);
        break;
    case 4:
        cv::cvtColor(img, image, cv::COLOR_BGRA2RGBA);
        break;
    case 1:
    default:
        if(img.isContinuous()) {
            image = img;
        } else {
            image = img.clone(); // reconstruction as continuous image.
        }
        break;
    }
    if (!image.isContinuous())
        return false;

    JxlEncoderFrameSettings* frame_settings = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
    // set frame settings from params if available
    for( size_t i = 0; i < params.size(); i += 2 )
    {
        if( params[i] == IMWRITE_JPEGXL_QUALITY )
        {
#if JPEGXL_MAJOR_VERSION > 0 || JPEGXL_MINOR_VERSION >= 10
            int quality = params[i+1];
            quality = MIN(MAX(quality, 0), 100);
            const float distance = JxlEncoderDistanceFromQuality(static_cast<float>(quality));
            JxlEncoderSetFrameDistance(frame_settings, distance);
            if (distance == 0)
                JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
#else
            CV_LOG_ONCE_WARNING(NULL, "Quality parameter is supported with libjxl v0.10.0 or later");
#endif
        }
        if( params[i] == IMWRITE_JPEGXL_DISTANCE )
        {
            int distance = params[i+1];
            distance = MIN(MAX(distance, 0), 25);
            JxlEncoderSetFrameDistance(frame_settings, distance);
            if (distance == 0)
                JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
        }
        if( params[i] == IMWRITE_JPEGXL_EFFORT )
        {
            int effort = params[i+1];
            effort = MIN(MAX(effort, 1), 10);
            JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort);
        }
        if( params[i] == IMWRITE_JPEGXL_DECODING_SPEED )
        {
            int speed = params[i+1];
            speed = MIN(MAX(speed, 0), 4);
            JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_DECODING_SPEED, speed);
        }
    }
    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(frame_settings, &format,
            static_cast<const void*>(image.ptr<uint8_t>()),
            image.total() * image.elemSize())) {
        return false;
    }
    JxlEncoderCloseInput(encoder.get());

    const size_t buffer_size = 16384;  // 16KB chunks

    std::vector<uint8_t> compressed(buffer_size);
    JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
    while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        uint8_t* next_out = compressed.data();
        size_t avail_out = buffer_size;
        process_result = JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
        if (JXL_ENC_ERROR == process_result)
            return false;
        const size_t write_size = buffer_size - avail_out;
        if ( strm.putBytes(compressed.data(), write_size) == false )
            return false;
    }
    return true;
}

}

#endif

/* End of file. */
