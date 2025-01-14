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

//
//  Loading and saving images.
//

#include "precomp.hpp"
#include "grfmts.hpp"
#include "utils.hpp"
#include "exif.hpp"
#undef min
#undef max
#include <iostream>
#include <fstream>
#include <cerrno>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/imgcodecs.hpp>



/****************************************************************************************\
*                                      Image Codecs                                      *
\****************************************************************************************/

namespace cv {

static const size_t CV_IO_MAX_IMAGE_PARAMS = cv::utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_PARAMS", 50);
static const size_t CV_IO_MAX_IMAGE_WIDTH = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_WIDTH", 1 << 20);
static const size_t CV_IO_MAX_IMAGE_HEIGHT = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_HEIGHT", 1 << 20);
static const size_t CV_IO_MAX_IMAGE_PIXELS = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_PIXELS", 1 << 30);

static Size validateInputImageSize(const Size& size)
{
    CV_Assert(size.width > 0);
    CV_Assert(static_cast<size_t>(size.width) <= CV_IO_MAX_IMAGE_WIDTH);
    CV_Assert(size.height > 0);
    CV_Assert(static_cast<size_t>(size.height) <= CV_IO_MAX_IMAGE_HEIGHT);
    uint64 pixels = (uint64)size.width * (uint64)size.height;
    CV_Assert(pixels <= CV_IO_MAX_IMAGE_PIXELS);
    return size;
}


static inline int calcType(int type, int flags)
{
    if ( (flags & (IMREAD_COLOR | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH)) == (IMREAD_COLOR | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH))
        return type;

    if( (flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED )
    {
        if( (flags & IMREAD_ANYDEPTH) == 0 )
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if( (flags & IMREAD_COLOR) != 0 || (flags & IMREAD_COLOR_RGB) != 0 ||
           ((flags & IMREAD_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1) )
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }
    return type;
}

namespace {

class ByteStreamBuffer: public std::streambuf
{
public:
    ByteStreamBuffer(char* base, size_t length)
    {
        setg(base, base, base + length);
    }

protected:
    virtual pos_type seekoff( off_type offset,
                              std::ios_base::seekdir dir,
                              std::ios_base::openmode ) CV_OVERRIDE
    {
        char* whence = eback();
        if (dir == std::ios_base::cur)
        {
            whence = gptr();
        }
        else if (dir == std::ios_base::end)
        {
            whence = egptr();
        }
        char* to = whence + offset;

        // check limits
        if (to >= eback() && to <= egptr())
        {
            setg(eback(), to, egptr());
            return gptr() - eback();
        }

        return -1;
    }
};

}

/**
 * @struct ImageCodecInitializer
 *
 * Container which stores the registered codecs to be used by OpenCV
*/
struct ImageCodecInitializer
{
    /**
     * Default Constructor for the ImageCodeInitializer
    */
    ImageCodecInitializer()
    {
        /// BMP Support
        decoders.push_back( makePtr<BmpDecoder>() );
        encoders.push_back( makePtr<BmpEncoder>() );

    #ifdef HAVE_IMGCODEC_GIF
        decoders.push_back( makePtr<GifDecoder>() );
        encoders.push_back( makePtr<GifEncoder>() );
    #endif
    #ifdef HAVE_AVIF
        decoders.push_back(makePtr<AvifDecoder>());
        encoders.push_back(makePtr<AvifEncoder>());
    #endif
    #ifdef HAVE_IMGCODEC_HDR
        decoders.push_back( makePtr<HdrDecoder>() );
        encoders.push_back( makePtr<HdrEncoder>() );
    #endif
    #ifdef HAVE_JPEG
        decoders.push_back( makePtr<JpegDecoder>() );
        encoders.push_back( makePtr<JpegEncoder>() );
    #endif
    #ifdef HAVE_WEBP
        decoders.push_back( makePtr<WebPDecoder>() );
        encoders.push_back( makePtr<WebPEncoder>() );
    #endif
    #ifdef HAVE_IMGCODEC_SUNRASTER
        decoders.push_back( makePtr<SunRasterDecoder>() );
        encoders.push_back( makePtr<SunRasterEncoder>() );
    #endif
    #ifdef HAVE_IMGCODEC_PXM
        decoders.push_back( makePtr<PxMDecoder>() );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_AUTO) );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_PBM) );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_PGM) );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_PPM) );
        decoders.push_back( makePtr<PAMDecoder>() );
        encoders.push_back( makePtr<PAMEncoder>() );
    #endif
    #ifdef HAVE_IMGCODEC_PFM
        decoders.push_back( makePtr<PFMDecoder>() );
        encoders.push_back( makePtr<PFMEncoder>() );
    #endif
    #ifdef HAVE_TIFF
        decoders.push_back( makePtr<TiffDecoder>() );
        encoders.push_back( makePtr<TiffEncoder>() );
    #endif
    #ifdef HAVE_SPNG
        decoders.push_back( makePtr<SPngDecoder>() );
        encoders.push_back( makePtr<SPngEncoder>() );
    #elif defined(HAVE_PNG)
        decoders.push_back( makePtr<PngDecoder>() );
        encoders.push_back( makePtr<PngEncoder>() );
    #endif
    #ifdef HAVE_GDCM
        decoders.push_back( makePtr<DICOMDecoder>() );
    #endif
    #ifdef HAVE_JASPER
        decoders.push_back( makePtr<Jpeg2KDecoder>() );
        encoders.push_back( makePtr<Jpeg2KEncoder>() );
    #endif
    #ifdef HAVE_JPEGXL
        decoders.push_back( makePtr<JpegXLDecoder>() );
        encoders.push_back( makePtr<JpegXLEncoder>() );
    #endif
    #ifdef HAVE_OPENJPEG
        decoders.push_back( makePtr<Jpeg2KJP2OpjDecoder>() );
        decoders.push_back( makePtr<Jpeg2KJ2KOpjDecoder>() );
        encoders.push_back( makePtr<Jpeg2KOpjEncoder>() );
    #endif
    #ifdef HAVE_OPENEXR
        decoders.push_back( makePtr<ExrDecoder>() );
        encoders.push_back( makePtr<ExrEncoder>() );
    #endif

    #ifdef HAVE_GDAL
        /// Attach the GDAL Decoder
        decoders.push_back( makePtr<GdalDecoder>() );
    #endif/*HAVE_GDAL*/
    }

    std::vector<ImageDecoder> decoders;
    std::vector<ImageEncoder> encoders;
};

static
ImageCodecInitializer& getCodecs()
{
    static ImageCodecInitializer g_codecs;
    return g_codecs;
}

/**
 * Find the decoders
 *
 * @param[in] filename File to search
 *
 * @return Image decoder to parse image file.
*/
static ImageDecoder findDecoder( const String& filename ) {

    size_t i, maxlen = 0;

    /// iterate through list of registered codecs
    ImageCodecInitializer& codecs = getCodecs();
    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        size_t len = codecs.decoders[i]->signatureLength();
        maxlen = std::max(maxlen, len);
    }

    /// Open the file
    FILE* f= fopen( filename.c_str(), "rb" );

    /// in the event of a failure, return an empty image decoder
    if( !f ) {
        CV_LOG_WARNING(NULL, "imread_('" << filename << "'): can't open/read file: check file path/integrity");
        return ImageDecoder();
    }

    // read the file signature
    String signature(maxlen, ' ');
    maxlen = fread( (void*)signature.c_str(), 1, maxlen, f );
    fclose(f);
    signature = signature.substr(0, maxlen);

    /// compare signature against all decoders
    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        if( codecs.decoders[i]->checkSignature(signature) )
            return codecs.decoders[i]->newDecoder();
    }

    /// If no decoder was found, return base type
    return ImageDecoder();
}

static ImageDecoder findDecoder( const Mat& buf )
{
    size_t i, maxlen = 0;

    if( buf.rows*buf.cols < 1 || !buf.isContinuous() )
        return ImageDecoder();

    ImageCodecInitializer& codecs = getCodecs();
    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        size_t len = codecs.decoders[i]->signatureLength();
        maxlen = std::max(maxlen, len);
    }

    String signature(maxlen, ' ');
    size_t bufSize = buf.rows*buf.cols*buf.elemSize();
    maxlen = std::min(maxlen, bufSize);
    memcpy( (void*)signature.c_str(), buf.data, maxlen );

    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        if( codecs.decoders[i]->checkSignature(signature) )
            return codecs.decoders[i]->newDecoder();
    }

    return ImageDecoder();
}

static ImageEncoder findEncoder( const String& _ext )
{
    if( _ext.size() <= 1 )
        return ImageEncoder();

    const char* ext = strrchr( _ext.c_str(), '.' );
    if( !ext )
        return ImageEncoder();
    int len = 0;
    for( ext++; len < 128 && isalnum(ext[len]); len++ )
        ;

    ImageCodecInitializer& codecs = getCodecs();
    for( size_t i = 0; i < codecs.encoders.size(); i++ )
    {
        String description = codecs.encoders[i]->getDescription();
        const char* descr = strchr( description.c_str(), '(' );

        while( descr )
        {
            descr = strchr( descr + 1, '.' );
            if( !descr )
                break;
            int j = 0;
            for( descr++; j < len && isalnum(descr[j]) ; j++ )
            {
                int c1 = tolower(ext[j]);
                int c2 = tolower(descr[j]);
                if( c1 != c2 )
                    break;
            }
            if( j == len && !isalnum(descr[j]))
                return codecs.encoders[i]->newEncoder();
            descr += j;
        }
    }

    return ImageEncoder();
}


static void ExifTransform(int orientation, OutputArray img)
{
    switch( orientation )
    {
        case    IMAGE_ORIENTATION_TL: //0th row == visual top, 0th column == visual left-hand side
            //do nothing, the image already has proper orientation
            break;
        case    IMAGE_ORIENTATION_TR: //0th row == visual top, 0th column == visual right-hand side
            flip(img, img, 1); //flip horizontally
            break;
        case    IMAGE_ORIENTATION_BR: //0th row == visual bottom, 0th column == visual right-hand side
            flip(img, img, -1);//flip both horizontally and vertically
            break;
        case    IMAGE_ORIENTATION_BL: //0th row == visual bottom, 0th column == visual left-hand side
            flip(img, img, 0); //flip vertically
            break;
        case    IMAGE_ORIENTATION_LT: //0th row == visual left-hand side, 0th column == visual top
            transpose(img, img);
            break;
        case    IMAGE_ORIENTATION_RT: //0th row == visual right-hand side, 0th column == visual top
            transpose(img, img);
            flip(img, img, 1); //flip horizontally
            break;
        case    IMAGE_ORIENTATION_RB: //0th row == visual right-hand side, 0th column == visual bottom
            transpose(img, img);
            flip(img, img, -1); //flip both horizontally and vertically
            break;
        case    IMAGE_ORIENTATION_LB: //0th row == visual left-hand side, 0th column == visual bottom
            transpose(img, img);
            flip(img, img, 0); //flip vertically
            break;
        default:
            //by default the image read has normal (JPEG_ORIENTATION_TL) orientation
            break;
    }
}

static void ApplyExifOrientation(ExifEntry_t orientationTag, OutputArray img)
{
    int orientation = IMAGE_ORIENTATION_TL;

    if (orientationTag.tag != INVALID_TAG)
    {
        orientation = orientationTag.field_u16; //orientation is unsigned short, so check field_u16
        ExifTransform(orientation, img);
    }
}

/**
 * Read an image into memory and return the information
 *
 * @param[in] filename File to load
 * @param[in] flags Flags
 * @param[in] mat Reference to C++ Mat object (If LOAD_MAT)
 *
*/
static bool
imread_( const String& filename, int flags, OutputArray mat )
{
    /// Search for the relevant decoder to handle the imagery
    ImageDecoder decoder;

#ifdef HAVE_GDAL
    if(flags != IMREAD_UNCHANGED && (flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL ){
        decoder = GdalDecoder().newDecoder();
    }else{
#endif
        decoder = findDecoder( filename );
#ifdef HAVE_GDAL
    }
#endif

    /// if no decoder was found, return nothing.
    if( !decoder ){
        return 0;
    }

    int scale_denom = 1;
    if( flags > IMREAD_LOAD_GDAL )
    {
        if( flags & IMREAD_REDUCED_GRAYSCALE_2 )
            scale_denom = 2;
        else if( flags & IMREAD_REDUCED_GRAYSCALE_4 )
            scale_denom = 4;
        else if( flags & IMREAD_REDUCED_GRAYSCALE_8 )
            scale_denom = 8;
    }

    // Try to decode image by RGB instead of BGR.
    if (flags & IMREAD_COLOR_RGB && flags != IMREAD_UNCHANGED)
    {
        decoder->setRGB(true);
    }

    /// set the scale_denom in the driver
    decoder->setScale( scale_denom );

    /// set the filename in the driver
    decoder->setSource( filename );

    try
    {
        // read the header to make sure it succeeds
        if( !decoder->readHeader() )
            return 0;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imread_('" << filename << "'): can't read header: " << e.what());
        return 0;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imread_('" << filename << "'): can't read header: unknown exception");
        return 0;
    }


    // established the required input image size
    Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

    // grab the decoded type
    const int type = calcType(decoder->type(), flags);

    if (mat.empty())
    {
        mat.create( size.height, size.width, type );
    }
    else
    {
        CV_CheckEQ(size, mat.size(), "");
        CV_CheckTypeEQ(type, mat.type(), "");
        CV_Assert(mat.isContinuous());
    }

    // read the image data
    Mat real_mat = mat.getMat();
    const void * original_ptr = real_mat.data;
    bool success = false;
    try
    {
        if (decoder->readData(real_mat))
        {
            CV_CheckTrue(original_ptr == real_mat.data, "Internal imread issue");
            success = true;
        }
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imread_('" << filename << "'): can't read data: " << e.what());
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imread_('" << filename << "'): can't read data: unknown exception");
    }
    if (!success)
    {
        mat.release();
        return false;
    }

    if( decoder->setScale( scale_denom ) > 1 ) // if decoder is JpegDecoder then decoder->setScale always returns 1
    {
        resize( mat, mat, Size( size.width / scale_denom, size.height / scale_denom ), 0, 0, INTER_LINEAR_EXACT);
    }

    /// optionally rotate the data if EXIF orientation flag says so
    if (!mat.empty() && (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED )
    {
        ApplyExifOrientation(decoder->getExifTag(ORIENTATION), mat);
    }

    return true;
}


static bool
imreadmulti_(const String& filename, int flags, std::vector<Mat>& mats, int start, int count)
{
    /// Search for the relevant decoder to handle the imagery
    ImageDecoder decoder;

    CV_CheckGE(start, 0, "Start index cannont be < 0");

#ifdef HAVE_GDAL
    if (flags != IMREAD_UNCHANGED && (flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL) {
        decoder = GdalDecoder().newDecoder();
    }
    else {
#endif
        decoder = findDecoder(filename);
#ifdef HAVE_GDAL
    }
#endif

    /// if no decoder was found, return nothing.
    if (!decoder) {
        return 0;
    }

    if (count < 0) {
        count = std::numeric_limits<int>::max();
    }

    if (flags & IMREAD_COLOR_RGB && flags != IMREAD_UNCHANGED)
        decoder->setRGB(true);

    /// set the filename in the driver
    decoder->setSource(filename);

    // read the header to make sure it succeeds
    try
    {
        // read the header to make sure it succeeds
        if (!decoder->readHeader())
            return 0;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read header: " << e.what());
        return 0;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read header: unknown exception");
        return 0;
    }

    int current = start;

    while (current > 0)
    {
        if (!decoder->nextPage())
        {
            return false;
        }
        --current;
    }

    while (current < count)
    {
        // grab the decoded type
        const int type = calcType(decoder->type(), flags);

        // established the required input image size
        Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

        // read the image data
        Mat mat(size.height, size.width, type);
        bool success = false;
        try
        {
            if (decoder->readData(mat))
                success = true;
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read data: " << e.what());
        }
        catch (...)
        {
            CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read data: unknown exception");
        }
        if (!success)
            break;

        // optionally rotate the data if EXIF' orientation flag says so
        if ((flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED)
        {
            ApplyExifOrientation(decoder->getExifTag(ORIENTATION), mat);
        }

        mats.push_back(mat);
        if (!decoder->nextPage())
        {
            break;
        }
        ++current;
    }

    return !mats.empty();
}

/**
 * Read an image
 *
 *  This function merely calls the actual implementation above and returns itself.
 *
 * @param[in] filename File to load
 * @param[in] flags Flags you wish to set.
*/
Mat imread( const String& filename, int flags )
{
    CV_TRACE_FUNCTION();

    /// create the basic container
    Mat img;

    /// load the data
    imread_( filename, flags, img );

    /// return a reference to the data
    return img;
}

void imread( const String& filename, OutputArray dst, int flags )
{
    CV_TRACE_FUNCTION();

    /// load the data
    imread_(filename, flags, dst);
}

/**
* Read a multi-page image
*
*  This function merely calls the actual implementation above and returns itself.
*
* @param[in] filename File to load
* @param[in] mats Reference to C++ vector<Mat> object to hold the images
* @param[in] flags Flags you wish to set.
*
*/
bool imreadmulti(const String& filename, std::vector<Mat>& mats, int flags)
{
    CV_TRACE_FUNCTION();

    return imreadmulti_(filename, flags, mats, 0, -1);
}


bool imreadmulti(const String& filename, std::vector<Mat>& mats, int start, int count, int flags)
{
    CV_TRACE_FUNCTION();

    return imreadmulti_(filename, flags, mats, start, count);
}

static bool
imreadanimation_(const String& filename, int flags, int start, int count, Animation& animation)
{
    bool success = false;
    if (start < 0) {
        start = 0;
    }
    if (count < 0) {
        count = INT16_MAX;
    }

    /// Search for the relevant decoder to handle the imagery
    ImageDecoder decoder;
    decoder = findDecoder(filename);

    /// if no decoder was found, return false.
    if (!decoder) {
        CV_LOG_WARNING(NULL, "Decoder for " << filename << " not found!\n");
        return false;
    }

    /// set the filename in the driver
    decoder->setSource(filename);
    // read the header to make sure it succeeds
    try
    {
        // read the header to make sure it succeeds
        if (!decoder->readHeader())
            return false;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imreadanimation_('" << filename << "'): can't read header: " << e.what());
        return false;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imreadanimation_('" << filename << "'): can't read header: unknown exception");
        return false;
    }

    int current = 0;
    int frame_count = (int)decoder->getFrameCount();
    count = count + start > frame_count ? frame_count - start : count;

    uint64 pixels = (uint64)decoder->width() * (uint64)decoder->height() * (uint64)(count + 4);
    if (pixels > CV_IO_MAX_IMAGE_PIXELS) {
        CV_LOG_WARNING(NULL, "\nyou are trying to read " << pixels <<
            " bytes that exceed CV_IO_MAX_IMAGE_PIXELS.\n");
        return false;
    }

    while (current < start + count)
    {
        // grab the decoded type
        const int type = calcType(decoder->type(), flags);

        // established the required input image size
        Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

        // read the image data
        Mat mat(size.height, size.width, type);
        success = false;
        try
        {
            if (decoder->readData(mat))
                success = true;
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "imreadanimation_('" << filename << "'): can't read data: " << e.what());
        }
        catch (...)
        {
            CV_LOG_ERROR(NULL, "imreadanimation_('" << filename << "'): can't read data: unknown exception");
        }
        if (!success)
            break;

        // optionally rotate the data if EXIF' orientation flag says so
        if ((flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED)
        {
            ApplyExifOrientation(decoder->getExifTag(ORIENTATION), mat);
        }

        if (current >= start)
        {
            int duration = decoder->animation().durations.size() > 0 ? decoder->animation().durations.back() : 1000;
            animation.durations.push_back(duration);
            animation.frames.push_back(mat);
        }

        if (!decoder->nextPage())
        {
            break;
        }
        ++current;
    }
    animation.bgcolor = decoder->animation().bgcolor;
    animation.loop_count = decoder->animation().loop_count;

    return success;
}

bool imreadanimation(const String& filename, CV_OUT Animation& animation, int start, int count)
{
    CV_TRACE_FUNCTION();

    return imreadanimation_(filename, IMREAD_UNCHANGED, start, count, animation);
}

static
size_t imcount_(const String& filename, int flags)
{
    try{
        ImageCollection collection(filename, flags);
        return collection.size();
    } catch(cv::Exception const& e) {
        // Reading header or finding decoder for the filename is failed
        CV_LOG_ERROR(NULL, "imcount_('" << filename << "'): can't read header or can't find decoder: " << e.what());
    }
    return 0;
}

size_t imcount(const String& filename, int flags)
{
    CV_TRACE_FUNCTION();

    return imcount_(filename, flags);
}


static bool imwrite_( const String& filename, const std::vector<Mat>& img_vec,
                      const std::vector<int>& params, bool flipv )
{
    bool isMultiImg = img_vec.size() > 1;
    std::vector<Mat> write_vec;

    ImageEncoder encoder = findEncoder( filename );
    if( !encoder )
        CV_Error( Error::StsError, "could not find a writer for the specified extension" );

    for (size_t page = 0; page < img_vec.size(); page++)
    {
        Mat image = img_vec[page];
        CV_Assert(!image.empty());

        CV_Assert( image.channels() == 1 || image.channels() == 3 || image.channels() == 4 );

        Mat temp;
        if( !encoder->isFormatSupported(image.depth()) )
        {
            CV_LOG_ONCE_WARNING(NULL, "Unsupported depth image for selected encoder is fallbacked to CV_8U.");
            CV_Assert( encoder->isFormatSupported(CV_8U) );
            image.convertTo( temp, CV_8U );
            image = temp;
        }

        if( flipv )
        {
            flip(image, temp, 0);
            image = temp;
        }

        write_vec.push_back(image);
    }

    encoder->setDestination( filename );

    CV_Check(params.size(), (params.size() & 1) == 0, "Encoding 'params' must be key-value pairs");
    CV_CheckLE(params.size(), (size_t)(CV_IO_MAX_IMAGE_PARAMS*2), "");
    bool code = false;
    try
    {
        if (!isMultiImg)
            code = encoder->write( write_vec[0], params );
        else
            code = encoder->writemulti( write_vec, params ); //to be implemented

        if (!code)
        {
            FILE* f = fopen( filename.c_str(), "wb" );
            if ( !f )
            {
                if (errno == EACCES)
                {
                    CV_LOG_WARNING(NULL, "imwrite_('" << filename << "'): can't open file for writing: permission denied");
                }
            }
            else
            {
                fclose(f);
                remove(filename.c_str());
            }
        }
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imwrite_('" << filename << "'): can't write data: " << e.what());
        code = false;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imwrite_('" << filename << "'): can't write data: unknown exception");
        code = false;
    }

    return code;
}

bool imwrite( const String& filename, InputArray _img,
              const std::vector<int>& params )
{
    CV_TRACE_FUNCTION();

    CV_Assert(!_img.empty());

    std::vector<Mat> img_vec;
    if (_img.isMatVector() || _img.isUMatVector())
        _img.getMatVector(img_vec);
    else
        img_vec.push_back(_img.getMat());

    CV_Assert(!img_vec.empty());
    return imwrite_(filename, img_vec, params, false);
}

static bool imwriteanimation_(const String& filename, const Animation& animation, const std::vector<int>& params)
{
    ImageEncoder encoder = findEncoder(filename);
    if (!encoder)
        CV_Error(Error::StsError, "could not find a writer for the specified extension");

    encoder->setDestination(filename);

    bool code = false;
    try
    {
        code = encoder->writeanimation(animation, params);

        if (!code)
        {
            FILE* f = fopen(filename.c_str(), "wb");
            if (!f)
            {
                if (errno == EACCES)
                {
                    CV_LOG_ERROR(NULL, "imwriteanimation_('" << filename << "'): can't open file for writing: permission denied");
                }
            }
            else
            {
                fclose(f);
                remove(filename.c_str());
            }
        }
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imwriteanimation_('" << filename << "'): can't write data: " << e.what());
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imwriteanimation_('" << filename << "'): can't write data: unknown exception");
    }

    return code;
}

bool imwriteanimation(const String& filename, const Animation& animation, const std::vector<int>& params)
{
    CV_Assert(!animation.frames.empty());
    CV_Assert(animation.frames.size() == animation.durations.size());
    return imwriteanimation_(filename, animation, params);
}

static bool
imdecode_( const Mat& buf, int flags, Mat& mat )
{
    CV_Assert(!buf.empty());
    CV_Assert(buf.isContinuous());
    CV_Assert(buf.checkVector(1, CV_8U) > 0);
    Mat buf_row = buf.reshape(1, 1);  // decoders expects single row, avoid issues with vector columns

    String filename;

    ImageDecoder decoder = findDecoder(buf_row);
    if( !decoder )
        return false;

    int scale_denom = 1;
    if( flags > IMREAD_LOAD_GDAL )
    {
        if( flags & IMREAD_REDUCED_GRAYSCALE_2 )
            scale_denom = 2;
        else if( flags & IMREAD_REDUCED_GRAYSCALE_4 )
            scale_denom = 4;
        else if( flags & IMREAD_REDUCED_GRAYSCALE_8 )
            scale_denom = 8;
    }

    // Try to decode image by RGB instead of BGR.
    if (flags & IMREAD_COLOR_RGB && flags != IMREAD_UNCHANGED)
    {
        decoder->setRGB(true);
    }

    /// set the scale_denom in the driver
    decoder->setScale( scale_denom );

    if( !decoder->setSource(buf_row) )
    {
        filename = tempfile();
        FILE* f = fopen( filename.c_str(), "wb" );
        if( !f )
            return false;
        size_t bufSize = buf_row.total()*buf.elemSize();
        if (fwrite(buf_row.ptr(), 1, bufSize, f) != bufSize)
        {
            fclose( f );
            CV_Error( Error::StsError, "failed to write image data to temporary file" );
        }
        if( fclose(f) != 0 )
        {
            CV_Error( Error::StsError, "failed to write image data to temporary file" );
        }
        decoder->setSource(filename);
    }

    bool success = false;
    try
    {
        if (decoder->readHeader())
            success = true;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imdecode_('" << filename << "'): can't read header: " << e.what());
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imdecode_('" << filename << "'): can't read header: unknown exception");
    }
    if (!success)
    {
        decoder.release();
        if (!filename.empty())
        {
            if (0 != remove(filename.c_str()))
            {
                CV_LOG_WARNING(NULL, "unable to remove temporary file:" << filename);
            }
        }
        return false;
    }

    // established the required input image size
    Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

    const int type = calcType(decoder->type(), flags);

    mat.create( size.height, size.width, type );

    success = false;
    try
    {
        if (decoder->readData(mat))
            success = true;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imdecode_('" << filename << "'): can't read data: " << e.what());
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imdecode_('" << filename << "'): can't read data: unknown exception");
    }

    if (!filename.empty())
    {
        if (0 != remove(filename.c_str()))
        {
            CV_LOG_WARNING(NULL, "unable to remove temporary file: " << filename);
        }
    }

    if (!success)
    {
        return false;
    }

    if( decoder->setScale( scale_denom ) > 1 ) // if decoder is JpegDecoder then decoder->setScale always returns 1
    {
        resize(mat, mat, Size( size.width / scale_denom, size.height / scale_denom ), 0, 0, INTER_LINEAR_EXACT);
    }

    /// optionally rotate the data if EXIF' orientation flag says so
    if (!mat.empty() && (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED)
    {
        ApplyExifOrientation(decoder->getExifTag(ORIENTATION), mat);
    }

    return true;
}


Mat imdecode( InputArray _buf, int flags )
{
    CV_TRACE_FUNCTION();

    Mat buf = _buf.getMat(), img;
    if (!imdecode_(buf, flags, img))
        img.release();

    return img;
}

Mat imdecode( InputArray _buf, int flags, Mat* dst )
{
    CV_TRACE_FUNCTION();

    Mat buf = _buf.getMat(), img;
    dst = dst ? dst : &img;
    if (imdecode_(buf, flags, *dst))
        return *dst;
    else
        return cv::Mat();
}

static bool
imdecodemulti_(const Mat& buf, int flags, std::vector<Mat>& mats, int start, int count)
{
    CV_Assert(!buf.empty());
    CV_Assert(buf.isContinuous());
    CV_Assert(buf.checkVector(1, CV_8U) > 0);
    Mat buf_row = buf.reshape(1, 1);  // decoders expects single row, avoid issues with vector columns

    String filename;

    ImageDecoder decoder = findDecoder(buf_row);
    if (!decoder)
        return false;

    // Try to decode image by RGB instead of BGR.
    if (flags & IMREAD_COLOR_RGB && flags != IMREAD_UNCHANGED)
    {
        decoder->setRGB(true);
    }

    if (count < 0) {
        count = std::numeric_limits<int>::max();
    }

    if (!decoder->setSource(buf_row))
    {
        filename = tempfile();
        FILE* f = fopen(filename.c_str(), "wb");
        if (!f)
            return false;
        size_t bufSize = buf_row.total() * buf.elemSize();
        if (fwrite(buf_row.ptr(), 1, bufSize, f) != bufSize)
        {
            fclose(f);
            CV_Error(Error::StsError, "failed to write image data to temporary file");
        }
        if (fclose(f) != 0)
        {
            CV_Error(Error::StsError, "failed to write image data to temporary file");
        }
        decoder->setSource(filename);
    }

    // read the header to make sure it succeeds
    bool success = false;
    try
    {
        // read the header to make sure it succeeds
        if (decoder->readHeader())
            success = true;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read header: " << e.what());
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read header: unknown exception");
    }

    int current = start;
    while (success && current > 0)
    {
        if (!decoder->nextPage())
        {
            success = false;
            break;
        }
        --current;
    }

    if (!success)
    {
        decoder.release();
        if (!filename.empty())
        {
            if (0 != remove(filename.c_str()))
            {
                CV_LOG_WARNING(NULL, "unable to remove temporary file: " << filename);
            }
        }
        return 0;
    }

    while (current < count)
    {
        // grab the decoded type
        const int type = calcType(decoder->type(), flags);

        // established the required input image size
        Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

        // read the image data
        Mat mat(size.height, size.width, type);
        success = false;
        try
        {
            if (decoder->readData(mat))
                success = true;
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read data: " << e.what());
        }
        catch (...)
        {
            CV_LOG_ERROR(NULL, "imreadmulti_('" << filename << "'): can't read data: unknown exception");
        }
        if (!success)
            break;

        // optionally rotate the data if EXIF' orientation flag says so
        if ((flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED)
        {
            ApplyExifOrientation(decoder->getExifTag(ORIENTATION), mat);
        }

        mats.push_back(mat);
        if (!decoder->nextPage())
        {
            break;
        }
        ++current;
    }

    if (!filename.empty())
    {
        if (0 != remove(filename.c_str()))
        {
            CV_LOG_WARNING(NULL, "unable to remove temporary file: " << filename);
        }
    }

    if (!success)
        mats.clear();
    return !mats.empty();
}

bool imdecodemulti(InputArray _buf, int flags, CV_OUT std::vector<Mat>& mats, const Range& range)
{
    CV_TRACE_FUNCTION();

    Mat buf = _buf.getMat();
    if (range == Range::all())
    {
        return imdecodemulti_(buf, flags, mats, 0, -1);
    }
    else
    {
        CV_CheckGE(range.start, 0, "Range start cannot be negative.");
        CV_CheckGT(range.size(), 0, "Range cannot be empty.");
        return imdecodemulti_(buf, flags, mats, range.start, range.size());
    }
}

bool imencode( const String& ext, InputArray _img,
               std::vector<uchar>& buf, const std::vector<int>& params )
{
    CV_TRACE_FUNCTION();

    ImageEncoder encoder = findEncoder( ext );
    if( !encoder )
        CV_Error( Error::StsError, "could not find encoder for the specified extension" );

    std::vector<Mat> img_vec;
    CV_Assert(!_img.empty());
    if (_img.isMatVector() || _img.isUMatVector())
        _img.getMatVector(img_vec);
    else
        img_vec.push_back(_img.getMat());

    CV_Assert(!img_vec.empty());
    const bool isMultiImg = img_vec.size() > 1;

    std::vector<Mat> write_vec;
    for (size_t page = 0; page < img_vec.size(); page++)
    {
        Mat image = img_vec[page];
        CV_Assert(!image.empty());

        const int channels = image.channels();
        CV_Assert( channels == 1 || channels == 3 || channels == 4 );

        Mat temp;
        if( !encoder->isFormatSupported(image.depth()) )
        {
            CV_LOG_ONCE_WARNING(NULL, "Unsupported depth image for selected encoder is fallbacked to CV_8U.");
            CV_Assert( encoder->isFormatSupported(CV_8U) );
            image.convertTo( temp, CV_8U );
            image = temp;
        }

        write_vec.push_back(image);
    }

    CV_Check(params.size(), (params.size() & 1) == 0, "Encoding 'params' must be key-value pairs");
    CV_CheckLE(params.size(), (size_t)(CV_IO_MAX_IMAGE_PARAMS*2), "");

    bool code = false;
    String filename;
    if( !encoder->setDestination(buf) )
    {
        filename = tempfile();
        code = encoder->setDestination(filename);
        CV_Assert( code );
    }

    try {
        if (!isMultiImg)
            code = encoder->write(write_vec[0], params);
        else
            code = encoder->writemulti(write_vec, params);

        encoder->throwOnEror();
        CV_Assert( code );
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "imencode(): can't encode data: " << e.what());
        code = false;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "imencode(): can't encode data: unknown exception");
        code = false;
    }

    if( !filename.empty() && code )
    {
        FILE* f = fopen( filename.c_str(), "rb" );
        CV_Assert(f != 0);
        fseek( f, 0, SEEK_END );
        long pos = ftell(f);
        buf.resize((size_t)pos);
        fseek( f, 0, SEEK_SET );
        buf.resize(fread( &buf[0], 1, buf.size(), f ));
        fclose(f);
        remove(filename.c_str());
    }
    return code;
}

bool imencodemulti( const String& ext, InputArrayOfArrays imgs,
                    std::vector<uchar>& buf, const std::vector<int>& params)
{
    return imencode(ext, imgs, buf, params);
}

bool haveImageReader( const String& filename )
{
    ImageDecoder decoder = cv::findDecoder(filename);
    return !decoder.empty();
}

bool haveImageWriter( const String& filename )
{
    cv::ImageEncoder encoder = cv::findEncoder(filename);
    return !encoder.empty();
}

class ImageCollection::Impl {
public:
    Impl() = default;
    Impl(const std::string&  filename, int flags);
    void init(String const& filename, int flags);
    size_t size() const;
    Mat& at(int index);
    Mat& operator[](int index);
    void releaseCache(int index);
    ImageCollection::iterator begin(ImageCollection* ptr);
    ImageCollection::iterator end(ImageCollection* ptr);
    Mat read();
    int width() const;
    int height() const;
    bool readHeader();
    Mat readData();
    bool advance();
    int currentIndex() const;
    void reset();

private:
    String m_filename;
    int m_flags{};
    std::size_t m_size{};
    int m_width{};
    int m_height{};
    int m_current{};
    std::vector<cv::Mat> m_pages;
    ImageDecoder m_decoder;
};

ImageCollection::Impl::Impl(std::string const& filename, int flags) {
    this->init(filename, flags);
}

void ImageCollection::Impl::init(String const& filename, int flags) {
    m_filename = filename;
    m_flags = flags;

#ifdef HAVE_GDAL
    if (m_flags != IMREAD_UNCHANGED && (m_flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL) {
        m_decoder = GdalDecoder().newDecoder();
    }
    else {
#endif
    m_decoder = findDecoder(filename);
#ifdef HAVE_GDAL
    }
#endif


    CV_Assert(m_decoder);
    m_decoder->setSource(filename);
    CV_Assert(m_decoder->readHeader());

    m_size = m_decoder->getFrameCount();
    m_pages.resize(m_size);
}

size_t ImageCollection::Impl::size() const { return m_size; }

Mat ImageCollection::Impl::read() {
    auto result = this->readHeader();
    if(!result) {
        return {};
    }
    return this->readData();
}

int ImageCollection::Impl::width() const {
    return m_width;
}

int ImageCollection::Impl::height() const {
    return m_height;
}

bool ImageCollection::Impl::readHeader() {
    bool status = m_decoder->readHeader();
    m_width = m_decoder->width();
    m_height = m_decoder->height();
    return status;
}

// readHeader must be called before calling this method
Mat ImageCollection::Impl::readData() {
    const int type = calcType(m_decoder->type(), m_flags);

    // established the required input image size
    Size size = validateInputImageSize(Size(m_width, m_height));

    Mat mat(size.height, size.width, type);
    bool success = false;
    try {
        if (m_decoder->readData(mat))
            success = true;
    }
    catch (const cv::Exception &e) {
        CV_LOG_ERROR(NULL, "ImageCollection class: can't read data: " << e.what());
    }
    catch (...) {
        CV_LOG_ERROR(NULL, "ImageCollection class:: can't read data: unknown exception");
    }
    if (!success)
        return cv::Mat();

    if ((m_flags & IMREAD_IGNORE_ORIENTATION) == 0 && m_flags != IMREAD_UNCHANGED) {
        ApplyExifOrientation(m_decoder->getExifTag(ORIENTATION), mat);
    }

    return mat;
}

bool ImageCollection::Impl::advance() {  ++m_current; return m_decoder->nextPage(); }

int ImageCollection::Impl::currentIndex() const { return m_current; }

ImageCollection::iterator ImageCollection::Impl::begin(ImageCollection* ptr) { return ImageCollection::iterator(ptr); }

ImageCollection::iterator ImageCollection::Impl::end(ImageCollection* ptr) { return ImageCollection::iterator(ptr, static_cast<int>(this->size())); }

void ImageCollection::Impl::reset() {
    m_current = 0;
#ifdef HAVE_GDAL
    if (m_flags != IMREAD_UNCHANGED && (m_flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL) {
        m_decoder = GdalDecoder().newDecoder();
    }
    else {
#endif
    m_decoder = findDecoder(m_filename);
#ifdef HAVE_GDAL
    }
#endif

    m_decoder->setSource(m_filename);
    m_decoder->readHeader();
}

Mat& ImageCollection::Impl::at(int index) {
    CV_Assert(index >= 0 && size_t(index) < m_size);
    return operator[](index);
}

Mat& ImageCollection::Impl::operator[](int index) {
    if(m_pages.at(index).empty()) {
        // We can't go backward in multi images. If the page is not in vector yet,
        // go back to first page and advance until the desired page and read it into memory
        if(m_current != index) {
            reset();
            for(int i = 0; i != index && advance(); ++i) {}
        }
        m_pages[index] = read();
    }
    return m_pages[index];
}

void ImageCollection::Impl::releaseCache(int index) {
    CV_Assert(index >= 0 && size_t(index) < m_size);
    m_pages[index].release();
}

/* ImageCollection API*/

ImageCollection::ImageCollection() : pImpl(new Impl()) {}

ImageCollection::ImageCollection(const std::string& filename, int flags) : pImpl(new Impl(filename, flags)) {}

void ImageCollection::init(const String& img, int flags) { pImpl->init(img, flags); }

size_t ImageCollection::size() const { return pImpl->size(); }

const Mat& ImageCollection::at(int index) { return pImpl->at(index); }

const Mat& ImageCollection::operator[](int index) { return pImpl->operator[](index); }

void ImageCollection::releaseCache(int index) { pImpl->releaseCache(index); }

Ptr<ImageCollection::Impl> ImageCollection::getImpl() { return pImpl; }

/* Iterator API */

ImageCollection::iterator ImageCollection::begin() { return pImpl->begin(this); }

ImageCollection::iterator ImageCollection::end() { return pImpl->end(this); }

ImageCollection::iterator::iterator(ImageCollection* col) : m_pCollection(col), m_curr(0) {}

ImageCollection::iterator::iterator(ImageCollection* col, int end) : m_pCollection(col), m_curr(end) {}

Mat& ImageCollection::iterator::operator*() {
    CV_Assert(m_pCollection);
    return m_pCollection->getImpl()->operator[](m_curr);
}

Mat* ImageCollection::iterator::operator->() {
    CV_Assert(m_pCollection);
    return &m_pCollection->getImpl()->operator[](m_curr);
}

ImageCollection::iterator& ImageCollection::iterator::operator++() {
    if(m_pCollection->pImpl->currentIndex() == m_curr) {
        m_pCollection->pImpl->advance();
    }
    m_curr++;
    return *this;
}

ImageCollection::iterator ImageCollection::iterator::operator++(int) {
    iterator tmp = *this;
    ++(*this);
    return tmp;
}

Animation::Animation(int loopCount, Scalar bgColor)
    : loop_count(loopCount), bgcolor(bgColor)
{
    if (loopCount < 0 || loopCount > 0xffff)
        this->loop_count = 0; // loop_count should be non-negative
}

}

/* End of file. */
