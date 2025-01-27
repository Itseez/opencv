// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Stefan Brüns <stefan.bruens@rwth-aachen.de>

#include "precomp.hpp"

#ifdef HAVE_OPENJPEG
#include "grfmt_jpeg2000_openjpeg.hpp"

#include "opencv2/core/utils/logger.hpp"

namespace cv {

namespace {

String colorspaceName(COLOR_SPACE colorspace)
{
    switch (colorspace)
    {
    case OPJ_CLRSPC_CMYK:
        return "CMYK";
    case OPJ_CLRSPC_SRGB:
        return "sRGB";
    case OPJ_CLRSPC_EYCC:
        return "e-YCC";
    case OPJ_CLRSPC_GRAY:
        return "grayscale";
    case OPJ_CLRSPC_SYCC:
        return "YUV";
    case OPJ_CLRSPC_UNKNOWN:
        return "unknown";
    case OPJ_CLRSPC_UNSPECIFIED:
        return "unspecified";
    default:
        CV_Error(Error::StsNotImplemented, "Invalid colorspace");
    }
}

template <class T>
struct ConstItTraits {
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;
};

template <class T>
struct NonConstItTraits {
   using value_type = T;
   using difference_type = std::ptrdiff_t;
   using pointer = T*;
   using reference = T&;
};

/**
 * Iterator over the channel in continuous chunk of the memory e.g. in the one row of a Mat
 * No bounds checks are preformed due to keeping this class as simple as possible while
 * fulfilling RandomAccessIterator naming requirements.
 *
 * @tparam Traits holds information about value type and constness of the defined types
 */
template <class Traits>
class ChannelsIterator
{
public:
    using difference_type = typename Traits::difference_type;
    using value_type = typename Traits::value_type;
    using pointer = typename Traits::pointer;
    using reference = typename Traits::reference;
    using iterator_category = std::random_access_iterator_tag;

    ChannelsIterator(pointer ptr, std::size_t channel, std::size_t channels_count)
        : ptr_ { ptr + channel }, step_ { channels_count }
    {
    }

    /* Element Access */
    reference operator*() const
    {
        return *ptr_;
    }

    pointer operator->() const
    {
        return &(operator*());
    }

    reference operator[](difference_type n) const
    {
        return *(*this + n);
    }

    /* Iterator movement */
    ChannelsIterator<Traits>& operator++()
    {
        ptr_ += step_;
        return *this;
    }

    ChannelsIterator<Traits> operator++(int)
    {
        ChannelsIterator ret(*this);
        ++(*this);
        return ret;
    }

    ChannelsIterator<Traits>& operator--()
    {
        ptr_ -= step_;
        return *this;
    }

    ChannelsIterator<Traits> operator--(int)
    {
        ChannelsIterator ret(*this);
        --(*this);
        return ret;
    }

    ChannelsIterator<Traits>& operator-=(difference_type n)
    {
        ptr_ -= n * step_;
        return *this;
    }

    ChannelsIterator<Traits>& operator+=(difference_type n)
    {
        ptr_ += n * step_;
        return *this;
    }

    ChannelsIterator<Traits> operator-(difference_type n) const
    {
        return ChannelsIterator<Traits>(*this) -= n;
    }

    ChannelsIterator<Traits> operator+(difference_type n) const
    {
        return ChannelsIterator<Traits>(*this) += n;
    }

    difference_type operator-(const ChannelsIterator<Traits>& other) const
    {
        return (ptr_ - other.ptr_) / step_;
    }

    /* Comparision */
    bool operator==(const ChannelsIterator<Traits>& other) const CV_NOEXCEPT
    {
        return ptr_ == other.ptr_;
    }

    bool operator!=(const ChannelsIterator<Traits>& other) const CV_NOEXCEPT
    {
        return !(*this == other);
    }

    bool operator<(const ChannelsIterator<Traits>& other) const CV_NOEXCEPT
    {
        return ptr_ < other.ptr_;
    }

    bool operator>(const ChannelsIterator<Traits>& other) const CV_NOEXCEPT
    {
        return other < *this;
    }

    bool operator>=(const ChannelsIterator<Traits>& other) const CV_NOEXCEPT
    {
        return !(*this < other);
    }

    bool operator<=(const ChannelsIterator<Traits>& other) const CV_NOEXCEPT
    {
        return !(other < *this);
    }

private:
    pointer ptr_{nullptr};
    std::size_t step_{1};
};

template <class Traits>
inline ChannelsIterator<Traits> operator+(typename Traits::difference_type n, const ChannelsIterator<Traits>& it)
{
    return it + n;
}

template<typename OutT, typename InT>
void copyToMatImpl(std::vector<InT*>&& in, Mat& out, uint8_t shift)
{
    using ChannelsIt = ChannelsIterator<NonConstItTraits<OutT>>;

    Size size = out.size();
    if (out.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    const bool isShiftRequired = shift != 0;

    const std::size_t channelsCount = in.size();

    if (isShiftRequired)
    {
        for (int i = 0; i < size.height; ++i)
        {
            auto rowPtr = out.ptr<OutT>(i);
            for (std::size_t c = 0; c < channelsCount; ++c)
            {
                const auto first = in[c];
                const auto last = first + size.width;
                auto dOut = ChannelsIt(rowPtr, c, channelsCount);
                std::transform(first, last, dOut, [shift](InT val) -> OutT { return static_cast<OutT>(val >> shift); });
                in[c] += size.width;
            }
        }
    }
    else
    {
        for (int i = 0; i < size.height; ++i)
        {
            auto rowPtr = out.ptr<OutT>(i);
            for (std::size_t c = 0; c < channelsCount; ++c)
            {
                const auto first = in[c];
                const auto last = first + size.width;
                auto dOut = ChannelsIt(rowPtr, c, channelsCount);
                std::transform(first, last, dOut, [](InT val) -> OutT { return static_cast<OutT>(val); });
                in[c] += size.width;
            }
        }
    }
}

template<typename InT>
void copyToMat(std::vector<const InT*>&& in, Mat& out, uint8_t shift)
{
    switch (out.depth())
    {
    case CV_8U:
        copyToMatImpl<uint8_t>(std::move(in), out, shift);
        break;
    case CV_16U:
        copyToMatImpl<uint16_t>(std::move(in), out, shift);
        break;
    default:
        CV_Error(Error::StsNotImplemented, "only depth CV_8U and CV16_U are supported");
    }
}

template<typename InT, typename OutT>
void copyFromMatImpl(const Mat& in, std::vector<OutT*>&& out)
{
    using ChannelsIt = ChannelsIterator<ConstItTraits<InT>>;

    Size size = in.size();
    if (in.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    const std::size_t outChannelsCount = out.size();

    for (int i = 0; i < size.height; ++i)
    {
        const InT* row = in.ptr<InT>(i);
        for (std::size_t c = 0; c < outChannelsCount; ++c)
        {
            auto first = ChannelsIt(row, c, outChannelsCount);
            auto last = first + size.width;
            out[c] = std::copy(first, last, out[c]);
        }
    }
}

template<typename OutT>
void copyFromMat(const Mat& in, std::vector<OutT*>&& out)
{
    switch (in.depth())
    {
    case CV_8U:
        copyFromMatImpl<uint8_t>(in, std::move(out));
        break;
    case CV_16U:
        copyFromMatImpl<uint16_t>(in, std::move(out));
        break;
    default:
        CV_Error(Error::StsNotImplemented, "only depth CV_8U and CV16_U are supported");
    }
}

void errorLogCallback(const char* msg, void* /* userData */)
{
    CV_LOG_ERROR(NULL, cv::format("OpenJPEG2000: %s", msg));
}

void warningLogCallback(const char* msg, void* /* userData */)
{
    CV_LOG_WARNING(NULL, cv::format("OpenJPEG2000: %s", msg));
}

void setupLogCallbacks(opj_codec_t* codec)
{
    if (!opj_set_error_handler(codec, errorLogCallback, nullptr))
    {
        CV_LOG_WARNING(NULL, "OpenJPEG2000: can not set error log handler");
    }
    if (!opj_set_warning_handler(codec, warningLogCallback, nullptr))
    {
        CV_LOG_WARNING(NULL, "OpenJPEG2000: can not set warning log handler");
    }
}

opj_dparameters setupDecoderParameters()
{
    opj_dparameters parameters;
    opj_set_default_decoder_parameters(&parameters);
    return parameters;
}

opj_cparameters setupEncoderParameters(const std::vector<int>& params, const int numChannels)
{
    opj_cparameters parameters;
    opj_set_default_encoder_parameters(&parameters);
    parameters.tcp_rates[0] = 4.0f;
    parameters.tcp_numlayers = 1;
    parameters.cp_disto_alloc = 1;
    parameters.tcp_mct = numChannels >= 3 ? 1 : 0;

    for (size_t i = 0; i < params.size(); i += 2)
    {
        switch (params[i])
        {
        case cv::IMWRITE_JPEG2000_COMPRESSION_X1000:
            if (params[i + 1] > 0)
            {
                parameters.tcp_rates[0] = 1000.f / std::min(params[i + 1], 1000);
            }
            break;
        default:
            CV_LOG_WARNING(NULL, "OpenJPEG2000(encoder): skip unsupported parameter: " << params[i]);
            break;
        }
    }
    return parameters;
}

bool decodeSRGBData(const opj_image_t& inImg, cv::Mat& outImg, uint8_t shift, bool use_rgb)
{
    using ImageComponents = std::vector<const OPJ_INT32*>;

    const int inChannels = inImg.numcomps;
    const int outChannels = outImg.channels();

    if (outChannels == 1)
    {
        // Assume gray (+ alpha) for 1 channels -> gray
        if (inChannels <= 2)
        {
            copyToMat(ImageComponents { inImg.comps[0].data }, outImg, shift);
        }
        // Assume RGB for >= 3 channels -> gray
        else
        {
            Mat tmp(outImg.size(), CV_MAKETYPE(outImg.depth(), 3));
            copyToMat(ImageComponents { inImg.comps[2].data, inImg.comps[1].data, inImg.comps[0].data },
                      tmp, shift);
            cvtColor(tmp, outImg, COLOR_BGR2GRAY);
        }
        return true;
    }

    if (inChannels >= 3)
    {
        int swap_rb = use_rgb ? 0 : 2;
        // Assume RGB (+ alpha) for 3 channels -> BGR
        ImageComponents incomps { inImg.comps[swap_rb].data, inImg.comps[1].data, inImg.comps[swap_rb^2].data };
        // Assume RGBA for 4 channels -> BGRA
        if (outChannels > 3)
        {
            incomps.push_back(inImg.comps[3].data);
        }
        copyToMat(std::move(incomps), outImg, shift);
        return true;
    }
    CV_LOG_ERROR(NULL,
                 cv::format("OpenJPEG2000: unsupported conversion from %d components to %d for SRGB image decoding",
                            inChannels, outChannels));
    return false;
}

bool decodeGrayscaleData(const opj_image_t& inImg, cv::Mat& outImg, uint8_t shift, bool)
{
    using ImageComponents = std::vector<const OPJ_INT32*>;

    const int inChannels = inImg.numcomps;
    const int outChannels = outImg.channels();

    if (outChannels == 1 || outChannels == 3)
    {
        copyToMat(ImageComponents(outChannels, inImg.comps[0].data), outImg, shift);
        return true;
    }
    CV_LOG_ERROR(NULL,
                 cv::format("OpenJPEG2000: unsupported conversion from %d components to %d for Grayscale image decoding",
                            inChannels, outChannels));
    return false;
}

bool decodeSYCCData(const opj_image_t& inImg, cv::Mat& outImg, uint8_t shift, bool use_rgb)
{
    using ImageComponents = std::vector<const OPJ_INT32*>;

    const int inChannels = inImg.numcomps;
    const int outChannels = outImg.channels();

    if (outChannels == 1) {
        copyToMat(ImageComponents { inImg.comps[0].data }, outImg, shift);
        return true;
    }

    if (outChannels == 3 && inChannels >= 3) {
        copyToMat(ImageComponents { inImg.comps[0].data, inImg.comps[1].data, inImg.comps[2].data },
                  outImg, shift);
        if (use_rgb)
            cvtColor(outImg, outImg, COLOR_YUV2RGB);
        else
            cvtColor(outImg, outImg, COLOR_YUV2BGR);
        return true;
    }

    CV_LOG_ERROR(NULL,
                 cv::format("OpenJPEG2000: unsupported conversion from %d components to %d for YUV image decoding",
                            inChannels, outChannels));
    return false;
}

OPJ_SIZE_T opjReadFromBuffer(void* dist, OPJ_SIZE_T count, detail::OpjMemoryBuffer* buffer)
{
    const OPJ_SIZE_T bytesToRead = std::min(buffer->availableBytes(), count);
    if (bytesToRead > 0)
    {
        memcpy(dist, buffer->pos, bytesToRead);
        buffer->pos += bytesToRead;
        return bytesToRead;
    }
    else
    {
        return static_cast<OPJ_SIZE_T>(-1);
    }
}

OPJ_SIZE_T opjSkipFromBuffer(OPJ_SIZE_T count, detail::OpjMemoryBuffer* buffer) {
    const OPJ_SIZE_T bytesToSkip = std::min(buffer->availableBytes(), count);
    if (bytesToSkip > 0)
    {
        buffer->pos += bytesToSkip;
        return bytesToSkip;
    }
    else
    {
        return static_cast<OPJ_SIZE_T>(-1);
    }
}

OPJ_BOOL opjSeekFromBuffer(OPJ_OFF_T count, detail::OpjMemoryBuffer* buffer)
{
    // Count should stay positive to prevent unsigned overflow
    CV_DbgAssert(count > 0);
    // To provide proper comparison between OPJ_OFF_T and OPJ_SIZE_T, both should be
    // casted to uint64_t (On 32-bit systems sizeof(size_t) might be 4)
    CV_DbgAssert(static_cast<uint64_t>(count) < static_cast<uint64_t>(std::numeric_limits<OPJ_SIZE_T>::max()));
    const OPJ_SIZE_T pos = std::min(buffer->length, static_cast<OPJ_SIZE_T>(count));
    buffer->pos = buffer->begin + pos;
    return OPJ_TRUE;
}

detail::StreamPtr opjCreateBufferInputStream(detail::OpjMemoryBuffer* buf)
{
    detail::StreamPtr stream{ opj_stream_default_create(/* isInput */ true) };
    if (stream)
    {
        opj_stream_set_user_data(stream.get(), static_cast<void*>(buf), nullptr);
        opj_stream_set_user_data_length(stream.get(), buf->length);

        opj_stream_set_read_function(stream.get(), (opj_stream_read_fn)(opjReadFromBuffer));
        opj_stream_set_skip_function(stream.get(), (opj_stream_skip_fn)(opjSkipFromBuffer));
        opj_stream_set_seek_function(stream.get(), (opj_stream_seek_fn)(opjSeekFromBuffer));
    }
    return stream;
}

} // namespace <anonymous>

/////////////////////// Jpeg2KOpjDecoder ///////////////////

namespace detail {

Jpeg2KOpjDecoderBase::Jpeg2KOpjDecoderBase(OPJ_CODEC_FORMAT format)
    : format_(format)
{
    m_buf_supported = true;
}


bool Jpeg2KOpjDecoderBase::readHeader()
{
    if (!m_buf.empty()) {
        opjBuf_ = detail::OpjMemoryBuffer(m_buf);
        stream_ = opjCreateBufferInputStream(&opjBuf_);
    }
    else
    {
        stream_.reset(opj_stream_create_default_file_stream(m_filename.c_str(), OPJ_STREAM_READ));
    }
    if (!stream_)
        return false;

    codec_.reset(opj_create_decompress(format_));
    if (!codec_)
        return false;

    // Callbacks are cleared, when opj_destroy_codec is called,
    // They can provide some additional information for the user, about what goes wrong
    setupLogCallbacks(codec_.get());

    opj_dparameters parameters = setupDecoderParameters();
    if (!opj_setup_decoder(codec_.get(), &parameters))
        return false;

    {
        opj_image_t* rawImage;
        if (!opj_read_header(stream_.get(), codec_.get(), &rawImage))
            return false;

        image_.reset(rawImage);
    }

    m_width = image_->x1 - image_->x0;
    m_height = image_->y1 - image_->y0;

    /* Different components may have different precision,
     * so check all.
     */
    bool hasAlpha = false;
    const int numcomps = image_->numcomps;
    CV_Check(numcomps, numcomps >= 1 && numcomps <= 4, "Unsupported number of components");
    for (int i = 0; i < numcomps; i++)
    {
        const opj_image_comp_t& comp = image_->comps[i];

        if (comp.sgnd)
        {
            CV_Error(Error::StsNotImplemented, cv::format("OpenJPEG2000: Component %d/%d is signed", i, numcomps));
        }

        if (hasAlpha && comp.alpha)
        {
            CV_Error(Error::StsNotImplemented, cv::format("OpenJPEG2000: Component %d/%d is duplicate alpha channel", i, numcomps));
        }

        hasAlpha |= comp.alpha != 0;

        if (comp.prec > 64)
        {
            CV_Error(Error::StsNotImplemented, "OpenJPEG2000: precision > 64 is not supported");
        }
        m_maxPrec = std::max(m_maxPrec, comp.prec);
    }

    if (m_maxPrec < 8) {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: Precision < 8 not supported");
    } else if (m_maxPrec == 8) {
        m_type = CV_MAKETYPE(CV_8U, numcomps);
    } else if (m_maxPrec <= 16) {
        m_type = CV_MAKETYPE(CV_16U, numcomps);
    } else if (m_maxPrec <= 23) {
        m_type = CV_MAKETYPE(CV_32F, numcomps);
    } else {
        m_type = CV_MAKETYPE(CV_64F, numcomps);
    }
    return true;
}

bool Jpeg2KOpjDecoderBase::readData( Mat& img )
{
    using DecodeFunc = bool(*)(const opj_image_t&, cv::Mat&, uint8_t shift, bool use_rgb);

    if (!opj_decode(codec_.get(), stream_.get(), image_.get()))
    {
        CV_Error(Error::StsError, "OpenJPEG2000: Decoding is failed");
    }

    if (img.channels() == 2)
    {
        CV_Error(Error::StsNotImplemented,
                 cv::format("OpenJPEG2000: Unsupported number of output channels. IN: %d OUT: 2", image_->numcomps));
    }

    DecodeFunc decode = nullptr;
    switch (image_->color_space)
    {
    case OPJ_CLRSPC_UNKNOWN:
        /* FALLTHRU */
    case OPJ_CLRSPC_UNSPECIFIED:
        CV_LOG_WARNING(NULL, "OpenJPEG2000: Image has unknown or unspecified color space, SRGB is assumed");
        /* FALLTHRU */
    case OPJ_CLRSPC_SRGB:
        decode = decodeSRGBData;
        break;
    case OPJ_CLRSPC_GRAY:
        decode = decodeGrayscaleData;
        break;
    case OPJ_CLRSPC_SYCC:
        decode = decodeSYCCData;
        break;
    default:
        CV_Error(Error::StsNotImplemented,
                 cv::format("OpenJPEG2000: Unsupported color space conversion: %s -> %s",
                            colorspaceName(image_->color_space).c_str(),
                            (img.channels() == 1) ? "gray" : "BGR"));
    }

    const int depth = img.depth();
    const OPJ_UINT32 outPrec = [depth]() {
        if (depth == CV_8U) return 8;
        if (depth == CV_16U) return 16;
        CV_Error(Error::StsNotImplemented,
                 cv::format("OpenJPEG2000: output precision > 16 not supported: target depth %d", depth));
    }();
    const uint8_t shift = outPrec > m_maxPrec ? 0 : (uint8_t)(m_maxPrec - outPrec); // prec <= 64

    const int inChannels = image_->numcomps;

    CV_Assert(inChannels > 0);
    CV_Assert(image_->comps);
    for (int c = 0; c < inChannels; c++)
    {
        const opj_image_comp_t& comp = image_->comps[c];
        CV_CheckEQ((int)comp.dx, 1, "OpenJPEG2000: tiles are not supported");
        CV_CheckEQ((int)comp.dy, 1, "OpenJPEG2000: tiles are not supported");
        CV_CheckEQ((int)comp.x0, 0, "OpenJPEG2000: tiles are not supported");
        CV_CheckEQ((int)comp.y0, 0, "OpenJPEG2000: tiles are not supported");
        CV_CheckEQ((int)comp.w, img.cols, "OpenJPEG2000: tiles are not supported");
        CV_CheckEQ((int)comp.h, img.rows, "OpenJPEG2000: tiles are not supported");
        CV_Assert(comp.data && "OpenJPEG2000: missing component data (unsupported / broken input)");
    }

    return decode(*image_, img, shift, m_use_rgb);
}

} // namespace detail

Jpeg2KJP2OpjDecoder::Jpeg2KJP2OpjDecoder()
    : Jpeg2KOpjDecoderBase(OPJ_CODEC_JP2)
{
    static const unsigned char JP2Signature[] = { 0, 0, 0, 0x0c, 'j', 'P', ' ', ' ', 13, 10, 0x87, 10 };
    m_signature = String((const char*) JP2Signature, sizeof(JP2Signature));
}

ImageDecoder Jpeg2KJP2OpjDecoder::newDecoder() const
{
    return makePtr<Jpeg2KJP2OpjDecoder>();
}

Jpeg2KJ2KOpjDecoder::Jpeg2KJ2KOpjDecoder()
    : Jpeg2KOpjDecoderBase(OPJ_CODEC_J2K)
{
    static const unsigned char J2KSignature[] = { 0xff, 0x4f, 0xff, 0x51 };
    m_signature = String((const char*) J2KSignature, sizeof(J2KSignature));
}

ImageDecoder Jpeg2KJ2KOpjDecoder::newDecoder() const
{
    return makePtr<Jpeg2KJ2KOpjDecoder>();
}

/////////////////////// Jpeg2KOpjEncoder ///////////////////

Jpeg2KOpjEncoder::Jpeg2KOpjEncoder()
{
    m_description = "JPEG-2000 files (*.jp2)";
}

ImageEncoder Jpeg2KOpjEncoder::newEncoder() const
{
    return makePtr<Jpeg2KOpjEncoder>();
}

bool Jpeg2KOpjEncoder::isFormatSupported(int depth) const
{
    return depth == CV_8U || depth == CV_16U;
}

bool Jpeg2KOpjEncoder::write(const Mat& img, const std::vector<int>& params)
{
    CV_Assert(params.size() % 2 == 0);

    const int channels = img.channels();
    CV_DbgAssert(channels > 0); // passed matrix is not empty
    if (channels > 4)
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: only BGR(a) and gray (+ alpha) images supported");
    }

    const int depth = img.depth();

    const OPJ_UINT32 outPrec = [depth]() {
        if (depth == CV_8U) return 8;
        if (depth == CV_16U) return 16;
        CV_Error(Error::StsNotImplemented,
                 cv::format("OpenJPEG2000: image precision > 16 not supported. Got: %d", depth));
    }();

    opj_cparameters parameters = setupEncoderParameters(params, img.channels());

    std::vector<opj_image_cmptparm_t> compparams(channels);
    for (int i = 0; i < channels; i++) {
        compparams[i].prec = outPrec;
        compparams[i].bpp = outPrec;
        compparams[i].sgnd = 0; // unsigned for now
        compparams[i].dx = parameters.subsampling_dx;
        compparams[i].dy = parameters.subsampling_dy;
        compparams[i].w = img.size().width;
        compparams[i].h = img.size().height;
    }


    auto colorspace = (channels > 2) ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY;
    detail::ImagePtr image(opj_image_create(channels, compparams.data(), colorspace));
    if (!image)
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: can not create image");
    }

    if (channels == 2 || channels == 4)
    {
        image->comps[channels - 1].alpha = 1;
    }
    // we want the full image
    image->x0 = 0;
    image->y0 = 0;
    image->x1 = compparams[0].dx * compparams[0].w;
    image->y1 = compparams[0].dy * compparams[0].h;

    // fill the component data arrays
    std::vector<OPJ_INT32*> outcomps(channels, nullptr);
    if (channels == 1)
    {
        outcomps.assign({ image->comps[0].data });
    }
    else if (channels == 2)
    {
        outcomps.assign({ image->comps[0].data, image->comps[1].data });
    }
    // Reversed order for BGR -> RGB conversion
    else if (channels == 3)
    {
        outcomps.assign({ image->comps[2].data, image->comps[1].data, image->comps[0].data });
    }
    else if (channels == 4)
    {
        outcomps.assign({ image->comps[2].data, image->comps[1].data, image->comps[0].data,
                          image->comps[3].data });
    }
    // outcomps holds pointers to the data, so the actual data will be modified but won't be freed
    // The container is not needed after data was copied
    copyFromMat(img, std::move(outcomps));

    detail::CodecPtr codec(opj_create_compress(OPJ_CODEC_JP2));
    if (!codec) {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: can not create compression codec");
    }

    setupLogCallbacks(codec.get());

    if (!opj_setup_encoder(codec.get(), &parameters, image.get()))
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: Can not setup encoder");
    }

    detail::StreamPtr stream(opj_stream_create_default_file_stream(m_filename.c_str(), OPJ_STREAM_WRITE));
    if (!stream)
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: Can not create stream");
    }

    if (!opj_start_compress(codec.get(), image.get(), stream.get()))
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: Can not start compression");
    }

    if (!opj_encode(codec.get(), stream.get()))
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: Encoding failed");
    }

    if (!opj_end_compress(codec.get(), stream.get()))
    {
        CV_Error(Error::StsNotImplemented, "OpenJPEG2000: Can not end compression");
    }

    return true;
}


} // namespace cv

#endif
