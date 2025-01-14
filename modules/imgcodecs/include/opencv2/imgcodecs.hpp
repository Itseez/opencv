/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#ifndef OPENCV_IMGCODECS_HPP
#define OPENCV_IMGCODECS_HPP

#include "opencv2/core.hpp"

/**
  @defgroup imgcodecs Image file reading and writing
  @{
    @defgroup imgcodecs_flags Flags used for image file reading and writing
    @defgroup imgcodecs_ios iOS glue
    @defgroup imgcodecs_macosx MacOS(OSX) glue
  @}
*/

//////////////////////////////// image codec ////////////////////////////////
namespace cv
{

//! @addtogroup imgcodecs
//! @{

//! @addtogroup imgcodecs_flags
//! @{

//! Imread flags
enum ImreadModes {
       IMREAD_UNCHANGED            = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
       IMREAD_GRAYSCALE            = 0,  //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
       IMREAD_COLOR_BGR            = 1,  //!< If set, always convert image to the 3 channel BGR color image.
       IMREAD_COLOR                = 1,  //!< Same as IMREAD_COLOR_BGR.
       IMREAD_ANYDEPTH             = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
       IMREAD_ANYCOLOR             = 4,  //!< If set, the image is read in any possible color format.
       IMREAD_LOAD_GDAL            = 8,  //!< If set, use the gdal driver for loading the image.
       IMREAD_REDUCED_GRAYSCALE_2  = 16, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
       IMREAD_REDUCED_COLOR_2      = 17, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
       IMREAD_REDUCED_GRAYSCALE_4  = 32, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
       IMREAD_REDUCED_COLOR_4      = 33, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
       IMREAD_REDUCED_GRAYSCALE_8  = 64, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
       IMREAD_REDUCED_COLOR_8      = 65, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
       IMREAD_IGNORE_ORIENTATION   = 128, //!< If set, do not rotate the image according to EXIF's orientation flag.
       IMREAD_COLOR_RGB            = 256, //!< If set, always convert image to the 3 channel RGB color image.
     };

//! Imwrite flags
enum ImwriteFlags {
       IMWRITE_JPEG_QUALITY        = 1,  //!< For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
       IMWRITE_JPEG_PROGRESSIVE    = 2,  //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_OPTIMIZE       = 3,  //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_RST_INTERVAL   = 4,  //!< JPEG restart interval, 0 - 65535, default is 0 - no restart.
       IMWRITE_JPEG_LUMA_QUALITY   = 5,  //!< Separate luma quality level, 0 - 100, default is -1 - don't use. If JPEG_LIB_VERSION < 70, Not supported.
       IMWRITE_JPEG_CHROMA_QUALITY = 6,  //!< Separate chroma quality level, 0 - 100, default is -1 - don't use. If JPEG_LIB_VERSION < 70, Not supported.
       IMWRITE_JPEG_SAMPLING_FACTOR = 7, //!< For JPEG, set sampling factor. See cv::ImwriteJPEGSamplingFactorParams.
       IMWRITE_PNG_COMPRESSION     = 16, //!< For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).
       IMWRITE_PNG_STRATEGY        = 17, //!< One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
       IMWRITE_PNG_BILEVEL         = 18, //!< Binary level PNG, 0 or 1, default is 0.
       IMWRITE_PXM_BINARY          = 32, //!< For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
       IMWRITE_EXR_TYPE            = (3 << 4) + 0 /* 48 */, //!< override EXR storage type (FLOAT (FP32) is default)
       IMWRITE_EXR_COMPRESSION     = (3 << 4) + 1 /* 49 */, //!< override EXR compression type (ZIP_COMPRESSION = 3 is default)
       IMWRITE_EXR_DWA_COMPRESSION_LEVEL = (3 << 4) + 2 /* 50 */, //!< override EXR DWA compression level (45 is default)
       IMWRITE_WEBP_QUALITY        = 64, //!< For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default (without any parameter) and for quality above 100 the lossless compression is used.
       IMWRITE_HDR_COMPRESSION     = (5 << 4) + 0 /* 80 */, //!< specify HDR compression
       IMWRITE_PAM_TUPLETYPE       = 128,//!< For PAM, sets the TUPLETYPE field to the corresponding string value that is defined for the format
       IMWRITE_TIFF_RESUNIT        = 256,//!< For TIFF, use to specify which DPI resolution unit to set; see libtiff documentation for valid values
       IMWRITE_TIFF_XDPI           = 257,//!< For TIFF, use to specify the X direction DPI
       IMWRITE_TIFF_YDPI           = 258,//!< For TIFF, use to specify the Y direction DPI
       IMWRITE_TIFF_COMPRESSION    = 259,//!< For TIFF, use to specify the image compression scheme. See cv::ImwriteTiffCompressionFlags. Note, for images whose depth is CV_32F, only libtiff's SGILOG compression scheme is used. For other supported depths, the compression scheme can be specified by this flag; LZW compression is the default.
       IMWRITE_TIFF_ROWSPERSTRIP   = 278,//!< For TIFF, use to specify the number of rows per strip.
       IMWRITE_TIFF_PREDICTOR      = 317,//!< For TIFF, use to specify predictor. See cv::ImwriteTiffPredictorFlags.
       IMWRITE_JPEG2000_COMPRESSION_X1000 = 272,//!< For JPEG2000, use to specify the target compression rate (multiplied by 1000). The value can be from 0 to 1000. Default is 1000.
       IMWRITE_AVIF_QUALITY        = 512,//!< For AVIF, it can be a quality between 0 and 100 (the higher the better). Default is 95.
       IMWRITE_AVIF_DEPTH          = 513,//!< For AVIF, it can be 8, 10 or 12. If >8, it is stored/read as CV_32F. Default is 8.
       IMWRITE_AVIF_SPEED          = 514,//!< For AVIF, it is between 0 (slowest) and (fastest). Default is 9.
       IMWRITE_JPEGXL_QUALITY      = 640,//!< For JPEG XL, it can be a quality from 0 to 100 (the higher is the better). Default value is 95. If set, distance parameter is re-calicurated from quality level automatically. This parameter request libjxl v0.10 or later.
       IMWRITE_JPEGXL_EFFORT       = 641,//!< For JPEG XL, encoder effort/speed level without affecting decoding speed; it is between 1 (fastest) and 10 (slowest). Default is 7.
       IMWRITE_JPEGXL_DISTANCE     = 642,//!< For JPEG XL, distance level for lossy compression: target max butteraugli distance, lower = higher quality, 0 = lossless; range: 0 .. 25. Default is 1.
       IMWRITE_JPEGXL_DECODING_SPEED = 643,//!< For JPEG XL, decoding speed tier for the provided options; minimum is 0 (slowest to decode, best quality/density), and maximum is 4 (fastest to decode, at the cost of some quality/density). Default is 0.
       IMWRITE_GIF_LOOP            = 1024,//!< For GIF, it can be a loop flag from 0 to 65535. Default is 0 - loop forever.
       IMWRITE_GIF_SPEED           = 1025,//!< For GIF, it is between 1 (slowest) and 100 (fastest). Default is 96.
       IMWRITE_GIF_QUALITY         = 1026, //!< For GIF, it can be a quality from 1 to 8. Default is 2. See cv::ImwriteGifCompressionFlags.
       IMWRITE_GIF_DITHER          = 1027, //!< For GIF, it can be a quality from -1(most dither) to 3(no dither). Default is 0.
       IMWRITE_GIF_TRANSPARENCY    = 1028, //!< For GIF, the alpha channel lower than this will be set to transparent. Default is 1.
       IMWRITE_GIF_COLORTABLE      = 1029  //!< For GIF, 0 means global color table is used, 1 means local color table is used. Default is 0.
};

enum ImwriteJPEGSamplingFactorParams {
       IMWRITE_JPEG_SAMPLING_FACTOR_411 = 0x411111, //!< 4x1,1x1,1x1
       IMWRITE_JPEG_SAMPLING_FACTOR_420 = 0x221111, //!< 2x2,1x1,1x1(Default)
       IMWRITE_JPEG_SAMPLING_FACTOR_422 = 0x211111, //!< 2x1,1x1,1x1
       IMWRITE_JPEG_SAMPLING_FACTOR_440 = 0x121111, //!< 1x2,1x1,1x1
       IMWRITE_JPEG_SAMPLING_FACTOR_444 = 0x111111  //!< 1x1,1x1,1x1(No subsampling)
     };

enum ImwriteTiffCompressionFlags {
        IMWRITE_TIFF_COMPRESSION_NONE = 1,            //!< dump mode
        IMWRITE_TIFF_COMPRESSION_CCITTRLE = 2,        //!< CCITT modified Huffman RLE
        IMWRITE_TIFF_COMPRESSION_CCITTFAX3 = 3,       //!< CCITT Group 3 fax encoding
        IMWRITE_TIFF_COMPRESSION_CCITT_T4 = 3,        //!< CCITT T.4 (TIFF 6 name)
        IMWRITE_TIFF_COMPRESSION_CCITTFAX4 = 4,       //!< CCITT Group 4 fax encoding
        IMWRITE_TIFF_COMPRESSION_CCITT_T6 = 4,        //!< CCITT T.6 (TIFF 6 name)
        IMWRITE_TIFF_COMPRESSION_LZW = 5,             //!< Lempel-Ziv  & Welch
        IMWRITE_TIFF_COMPRESSION_OJPEG = 6,           //!< !6.0 JPEG
        IMWRITE_TIFF_COMPRESSION_JPEG = 7,            //!< %JPEG DCT compression
        IMWRITE_TIFF_COMPRESSION_T85 = 9,             //!< !TIFF/FX T.85 JBIG compression
        IMWRITE_TIFF_COMPRESSION_T43 = 10,            //!< !TIFF/FX T.43 colour by layered JBIG compression
        IMWRITE_TIFF_COMPRESSION_NEXT = 32766,        //!< NeXT 2-bit RLE
        IMWRITE_TIFF_COMPRESSION_CCITTRLEW = 32771,   //!< #1 w/ word alignment
        IMWRITE_TIFF_COMPRESSION_PACKBITS = 32773,    //!< Macintosh RLE
        IMWRITE_TIFF_COMPRESSION_THUNDERSCAN = 32809, //!< ThunderScan RLE
        IMWRITE_TIFF_COMPRESSION_IT8CTPAD = 32895,    //!< IT8 CT w/padding
        IMWRITE_TIFF_COMPRESSION_IT8LW = 32896,       //!< IT8 Linework RLE
        IMWRITE_TIFF_COMPRESSION_IT8MP = 32897,       //!< IT8 Monochrome picture
        IMWRITE_TIFF_COMPRESSION_IT8BL = 32898,       //!< IT8 Binary line art
        IMWRITE_TIFF_COMPRESSION_PIXARFILM = 32908,   //!< Pixar companded 10bit LZW
        IMWRITE_TIFF_COMPRESSION_PIXARLOG = 32909,    //!< Pixar companded 11bit ZIP
        IMWRITE_TIFF_COMPRESSION_DEFLATE = 32946,     //!< Deflate compression, legacy tag
        IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE = 8,   //!< Deflate compression, as recognized by Adobe
        IMWRITE_TIFF_COMPRESSION_DCS = 32947,         //!< Kodak DCS encoding
        IMWRITE_TIFF_COMPRESSION_JBIG = 34661,        //!< ISO JBIG
        IMWRITE_TIFF_COMPRESSION_SGILOG = 34676,      //!< SGI Log Luminance RLE
        IMWRITE_TIFF_COMPRESSION_SGILOG24 = 34677,    //!< SGI Log 24-bit packed
        IMWRITE_TIFF_COMPRESSION_JP2000 = 34712,      //!< Leadtools JPEG2000
        IMWRITE_TIFF_COMPRESSION_LERC = 34887,        //!< ESRI Lerc codec: https://github.com/Esri/lerc
        IMWRITE_TIFF_COMPRESSION_LZMA = 34925,        //!< LZMA2
        IMWRITE_TIFF_COMPRESSION_ZSTD = 50000,        //!< ZSTD: WARNING not registered in Adobe-maintained registry
        IMWRITE_TIFF_COMPRESSION_WEBP = 50001,        //!< WEBP: WARNING not registered in Adobe-maintained registry
        IMWRITE_TIFF_COMPRESSION_JXL = 50002          //!< JPEGXL: WARNING not registered in Adobe-maintained registry
};

enum ImwriteTiffPredictorFlags {
        IMWRITE_TIFF_PREDICTOR_NONE = 1,              //!< no prediction scheme used
        IMWRITE_TIFF_PREDICTOR_HORIZONTAL = 2,        //!< horizontal differencing
        IMWRITE_TIFF_PREDICTOR_FLOATINGPOINT = 3      //!< floating point predictor

};

enum ImwriteEXRTypeFlags {
       /*IMWRITE_EXR_TYPE_UNIT = 0, //!< not supported */
       IMWRITE_EXR_TYPE_HALF   = 1, //!< store as HALF (FP16)
       IMWRITE_EXR_TYPE_FLOAT  = 2  //!< store as FP32 (default)
     };

enum ImwriteEXRCompressionFlags {
       IMWRITE_EXR_COMPRESSION_NO    = 0, //!< no compression
       IMWRITE_EXR_COMPRESSION_RLE   = 1, //!< run length encoding
       IMWRITE_EXR_COMPRESSION_ZIPS  = 2, //!< zlib compression, one scan line at a time
       IMWRITE_EXR_COMPRESSION_ZIP   = 3, //!< zlib compression, in blocks of 16 scan lines
       IMWRITE_EXR_COMPRESSION_PIZ   = 4, //!< piz-based wavelet compression
       IMWRITE_EXR_COMPRESSION_PXR24 = 5, //!< lossy 24-bit float compression
       IMWRITE_EXR_COMPRESSION_B44   = 6, //!< lossy 4-by-4 pixel block compression, fixed compression rate
       IMWRITE_EXR_COMPRESSION_B44A  = 7, //!< lossy 4-by-4 pixel block compression, flat fields are compressed more
       IMWRITE_EXR_COMPRESSION_DWAA  = 8, //!< lossy DCT based compression, in blocks of 32 scanlines. More efficient for partial buffer access. Supported since OpenEXR 2.2.0.
       IMWRITE_EXR_COMPRESSION_DWAB  = 9, //!< lossy DCT based compression, in blocks of 256 scanlines. More efficient space wise and faster to decode full frames than DWAA_COMPRESSION. Supported since OpenEXR 2.2.0.
     };

//! Imwrite PNG specific flags used to tune the compression algorithm.
/** These flags will be modify the way of PNG image compression and will be passed to the underlying zlib processing stage.

-   The effect of IMWRITE_PNG_STRATEGY_FILTERED is to force more Huffman coding and less string matching; it is somewhat intermediate between IMWRITE_PNG_STRATEGY_DEFAULT and IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY.
-   IMWRITE_PNG_STRATEGY_RLE is designed to be almost as fast as IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, but give better compression for PNG image data.
-   The strategy parameter only affects the compression ratio but not the correctness of the compressed output even if it is not set appropriately.
-   IMWRITE_PNG_STRATEGY_FIXED prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.
*/
enum ImwritePNGFlags {
       IMWRITE_PNG_STRATEGY_DEFAULT      = 0, //!< Use this value for normal data.
       IMWRITE_PNG_STRATEGY_FILTERED     = 1, //!< Use this value for data produced by a filter (or predictor).Filtered data consists mostly of small values with a somewhat random distribution. In this case, the compression algorithm is tuned to compress them better.
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2, //!< Use this value to force Huffman encoding only (no string match).
       IMWRITE_PNG_STRATEGY_RLE          = 3, //!< Use this value to limit match distances to one (run-length encoding).
       IMWRITE_PNG_STRATEGY_FIXED        = 4  //!< Using this value prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.
     };

//! Imwrite PAM specific tupletype flags used to define the 'TUPLETYPE' field of a PAM file.
enum ImwritePAMFlags {
       IMWRITE_PAM_FORMAT_NULL            = 0,
       IMWRITE_PAM_FORMAT_BLACKANDWHITE   = 1,
       IMWRITE_PAM_FORMAT_GRAYSCALE       = 2,
       IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3,
       IMWRITE_PAM_FORMAT_RGB             = 4,
       IMWRITE_PAM_FORMAT_RGB_ALPHA       = 5
     };

//! Imwrite HDR specific values for IMWRITE_HDR_COMPRESSION parameter key
enum ImwriteHDRCompressionFlags {
    IMWRITE_HDR_COMPRESSION_NONE = 0,
    IMWRITE_HDR_COMPRESSION_RLE = 1
};

//! Imwrite GIF specific values for IMWRITE_GIF_QUALITY parameter key, if larger than 3, then its related to the size of the color table.
enum ImwriteGIFCompressionFlags {
    IMWRITE_GIF_FAST_NO_DITHER       = 1,
    IMWRITE_GIF_FAST_FLOYD_DITHER    = 2,
    IMWRITE_GIF_COLORTABLE_SIZE_8    = 3,
    IMWRITE_GIF_COLORTABLE_SIZE_16   = 4,
    IMWRITE_GIF_COLORTABLE_SIZE_32   = 5,
    IMWRITE_GIF_COLORTABLE_SIZE_64   = 6,
    IMWRITE_GIF_COLORTABLE_SIZE_128  = 7,
    IMWRITE_GIF_COLORTABLE_SIZE_256  = 8
};

//! @} imgcodecs_flags

/** @brief Represents an animation with multiple frames.
The `Animation` struct is designed to store and manage data for animated sequences such as those from animated formats (e.g., GIF, AVIF, APNG, WebP).
It provides support for looping, background color settings, frame timing, and frame storage.
*/
struct CV_EXPORTS_W_SIMPLE Animation
{
    //! Number of times the animation should loop. 0 means infinite looping.
    CV_PROP_RW int loop_count;
    //! Background color of the animation in BGRA format.
    CV_PROP_RW Scalar bgcolor;
    //! Duration for each frame in milliseconds.
    CV_PROP_RW std::vector<int> durations;
    //! Vector of frames, where each Mat represents a single frame.
    CV_PROP_RW std::vector<Mat> frames;

    /** @brief Constructs an Animation object with optional loop count and background color.

    @param loopCount An integer representing the number of times the animation should loop:
    - `0` (default) indicates infinite looping, meaning the animation will replay continuously.
    - Positive values denote finite repeat counts, allowing the animation to play a limited number of times.
    - If a negative value or a value beyond the maximum of `0xffff` (65535) is provided, it is reset to `0`
    (infinite looping) to maintain valid bounds.

    @param bgColor A `Scalar` object representing the background color in BGRA format:
    - Defaults to `Scalar()`, indicating an empty color (usually transparent if supported).
    - This background color provides a solid fill behind frames that have transparency, ensuring a consistent display appearance.
    */
    Animation(int loopCount = 0, Scalar bgColor = Scalar());
};

/** @brief Loads an image from a file.

@anchor imread

The `imread` function loads an image from the specified file and returns OpenCV matrix. If the image cannot be
read (because of a missing file, improper permissions, or unsupported/invalid format), the function
returns an empty matrix.

Currently, the following file formats are supported:

-   Windows bitmaps - \*.bmp, \*.dib (always supported)
-   GIF files - \*.gif (always supported)
-   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Note* section)
-   JPEG 2000 files - \*.jp2 (see the *Note* section)
-   Portable Network Graphics - \*.png (see the *Note* section)
-   WebP - \*.webp (see the *Note* section)
-   AVIF - \*.avif (see the *Note* section)
-   Portable image format - \*.pbm, \*.pgm, \*.ppm, \*.pxm, \*.pnm (always supported)
-   PFM files - \*.pfm (see the *Note* section)
-   Sun rasters - \*.sr, \*.ras (always supported)
-   TIFF files - \*.tiff, \*.tif (see the *Note* section)
-   OpenEXR Image files - \*.exr (see the *Note* section)
-   Radiance HDR - \*.hdr, \*.pic (always supported)
-   Raster and Vector geospatial data supported by GDAL (see the *Note* section)

@note
-   The function determines the type of an image by its content, not by the file extension.
-   In the case of color images, the decoded images will have the channels stored in **B G R** order.
-   When using IMREAD_GRAYSCALE, the codec's internal grayscale conversion will be used, if available.
    Results may differ from the output of cvtColor().
-   On Microsoft Windows\* and Mac OS\*, the codecs shipped with OpenCV (libjpeg, libpng, libtiff,
    and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs, and TIFFs. On Mac OS,
    there is also an option to use native Mac OS image readers. However, beware that currently these
    native image loaders give images with different pixel values because of the color management embedded
    into Mac OS.
-   On Linux\*, BSD flavors, and other Unix-like open-source operating systems, OpenCV looks for
    codecs supplied with the OS. Ensure the relevant packages are installed (including development
    files, such as "libjpeg-dev" in Debian\* and Ubuntu\*) to get codec support, or turn
    on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.
-   If the *WITH_GDAL* flag is set to true in CMake and @ref IMREAD_LOAD_GDAL is used to load the image,
    the [GDAL](http://www.gdal.org) driver will be used to decode the image, supporting
    [Raster](http://www.gdal.org/formats_list.html) and [Vector](http://www.gdal.org/ogr_formats.html) formats.
-   If EXIF information is embedded in the image file, the EXIF orientation will be taken into account,
    and thus the image will be rotated accordingly unless the flags @ref IMREAD_IGNORE_ORIENTATION
    or @ref IMREAD_UNCHANGED are passed.
-   Use the IMREAD_UNCHANGED flag to preserve the floating-point values from PFM images.
-   By default, the number of pixels must be less than 2^30. This limit can be changed by setting
    the environment variable `OPENCV_IO_MAX_IMAGE_PIXELS`. See @ref tutorial_env_reference.

@param filename Name of the file to be loaded.
@param flags Flag that can take values of `cv::ImreadModes`.
*/
CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR_BGR );

/** @brief Loads an image from a file.

This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts and the return value.
@param filename Name of file to be loaded.
@param dst object in which the image will be loaded.
@param flags Flag that can take values of cv::ImreadModes
@note
The image passing through the img parameter can be pre-allocated. The memory is reused if the shape and the type match with the load image.
 */
CV_EXPORTS_W void imread( const String& filename, OutputArray dst, int flags = IMREAD_COLOR_BGR );

/** @brief Loads a multi-page image from a file.

The function imreadmulti loads a multi-page image from the specified file into a vector of Mat objects.
@param filename Name of file to be loaded.
@param mats A vector of Mat objects holding each page.
@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
@sa cv::imread
*/
CV_EXPORTS_W bool imreadmulti(const String& filename, CV_OUT std::vector<Mat>& mats, int flags = IMREAD_ANYCOLOR);

/** @brief Loads images of a multi-page image from a file.

The function imreadmulti loads a specified range from a multi-page image from the specified file into a vector of Mat objects.
@param filename Name of file to be loaded.
@param mats A vector of Mat objects holding each page.
@param start Start index of the image to load
@param count Count number of images to load
@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
@sa cv::imread
*/
CV_EXPORTS_W bool imreadmulti(const String& filename, CV_OUT std::vector<Mat>& mats, int start, int count, int flags = IMREAD_ANYCOLOR);

/** @example samples/cpp/tutorial_code/imgcodecs/animations.cpp
An example to show usage of cv::imreadanimation and cv::imwriteanimation functions.
Check @ref tutorial_animations "the corresponding tutorial" for more details
*/

/** @brief Loads frames from an animated image file into an Animation structure.

The function imreadanimation loads frames from an animated image file (e.g., GIF, AVIF, APNG, WEBP) into the provided Animation struct.

@param filename A string containing the path to the file.
@param animation A reference to an Animation structure where the loaded frames will be stored. It should be initialized before the function is called.
@param start The index of the first frame to load. This is optional and defaults to 0.
@param count The number of frames to load. This is optional and defaults to 32767.

@return Returns true if the file was successfully loaded and frames were extracted; returns false otherwise.
*/
CV_EXPORTS_W bool imreadanimation(const String& filename, CV_OUT Animation& animation, int start = 0, int count = INT16_MAX);

/** @brief Saves an Animation to a specified file.

The function imwriteanimation saves the provided Animation data to the specified file in an animated format.
Supported formats depend on the implementation and may include formats like GIF, AVIF, APNG, or WEBP.

@param filename The name of the file where the animation will be saved. The file extension determines the format.
@param animation A constant reference to an Animation struct containing the frames and metadata to be saved.
@param params Optional format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ...).
These parameters are used to specify additional options for the encoding process. Refer to `cv::ImwriteFlags` for details on possible parameters.

@return Returns true if the animation was successfully saved; returns false otherwise.
*/
CV_EXPORTS_W bool imwriteanimation(const String& filename, const Animation& animation, const std::vector<int>& params = std::vector<int>());

/** @brief Returns the number of images inside the given file

The function imcount returns the number of pages in a multi-page image (e.g. TIFF), the number of frames in an animation (e.g. AVIF), and 1 otherwise.
If the image cannot be decoded, 0 is returned.
@param filename Name of file to be loaded.
@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
@todo when cv::IMREAD_LOAD_GDAL flag used the return value will be 0 or 1 because OpenCV's GDAL decoder doesn't support multi-page reading yet.
*/
CV_EXPORTS_W size_t imcount(const String& filename, int flags = IMREAD_ANYCOLOR);

/** @brief Saves an image to a specified file.

The function imwrite saves the image to the specified file. The image format is chosen based on the
filename extension (see cv::imread for the list of extensions). In general, only 8-bit unsigned (CV_8U)
single-channel or 3-channel (with 'BGR' channel order) images
can be saved using this function, with these exceptions:

- With OpenEXR encoder, only 32-bit float (CV_32F) images can be saved.
  - 8-bit unsigned (CV_8U) images are not supported.
- With Radiance HDR encoder, non 64-bit float (CV_64F) images can be saved.
  - All images will be converted to 32-bit float (CV_32F).
- With JPEG 2000 encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved.
- With JPEG XL encoder, 8-bit unsigned (CV_8U), 16-bit unsigned (CV_16U) and 32-bit float(CV_32F) images can be saved.
  - JPEG XL images with an alpha channel can be saved using this function.
    To do this, create 8-bit (or 16-bit, 32-bit float) 4-channel image BGRA, where the alpha channel goes last.
    Fully transparent pixels should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535/1.0.
- With PAM encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved.
- With PNG encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved.
  - PNG images with an alpha channel can be saved using this function. To do this, create
    8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels
    should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535 (see the code sample below).
- With PGM/PPM encoder, 8-bit unsigned (CV_8U) and 16-bit unsigned (CV_16U) images can be saved.
- With TIFF encoder, 8-bit unsigned (CV_8U), 8-bit signed (CV_8S),
                     16-bit unsigned (CV_16U), 16-bit signed (CV_16S),
                     32-bit unsigned (CV_32U), 32-bit signed (CV_32S),
                     64-bit unsigned (CV_64U), 64-bit signed (CV_64S),
                     32-bit float (CV_32F) and 64-bit float (CV_64F) images can be saved.
  - Multiple images (vector of Mat) can be saved in TIFF format (see the code sample below).
  - 32-bit float 3-channel (CV_32FC3) TIFF images will be saved
    using the LogLuv high dynamic range encoding (4 bytes per pixel)

If the image format is not supported, the image will be converted to 8-bit unsigned (CV_8U) and saved that way.

If the format, depth or channel order is different, use
Mat::convertTo and cv::cvtColor to convert it before saving. Or, use the universal FileStorage I/O
functions to save the image to XML or YAML format.

The sample below shows how to create a BGRA image, how to set custom compression parameters and save it to a PNG file.
It also demonstrates how to save multiple images in a TIFF file:
@include snippets/imgcodecs_imwrite.cpp
@param filename Name of the file.
@param img (Mat or vector of Mat) Image or Images to be saved.
@param params Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see cv::ImwriteFlags
*/
CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());

//! @brief multi-image overload for bindings
CV_WRAP static inline
bool imwritemulti(const String& filename, InputArrayOfArrays img,
                  const std::vector<int>& params = std::vector<int>())
{
    return imwrite(filename, img, params);
}

/** @brief Reads an image from a buffer in memory.

The function imdecode reads an image from the specified buffer in the memory. If the buffer is too short or
contains invalid data, the function returns an empty matrix ( Mat::data==NULL ).

See cv::imread for the list of supported formats and flags description.

@note In the case of color images, the decoded images will have the channels stored in **B G R** order.
@param buf Input array or vector of bytes.
@param flags The same flags as in cv::imread, see cv::ImreadModes.
*/
CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );

/** @overload
@param buf Input array or vector of bytes.
@param flags The same flags as in cv::imread, see cv::ImreadModes.
@param dst The optional output placeholder for the decoded matrix. It can save the image
reallocations when the function is called repeatedly for images of the same size. In case of decoder
failure the function returns empty cv::Mat object, but does not release user-provided dst buffer.
*/
CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst);

/** @brief Reads a multi-page image from a buffer in memory.

The function imdecodemulti reads a multi-page image from the specified buffer in the memory. If the buffer is too short or
contains invalid data, the function returns false.

See cv::imreadmulti for the list of supported formats and flags description.

@note In the case of color images, the decoded images will have the channels stored in **B G R** order.
@param buf Input array or vector of bytes.
@param flags The same flags as in cv::imread, see cv::ImreadModes.
@param mats A vector of Mat objects holding each page, if more than one.
@param range A continuous selection of pages.
*/
CV_EXPORTS_W bool imdecodemulti(InputArray buf, int flags, CV_OUT std::vector<Mat>& mats, const cv::Range& range = Range::all());

/** @brief Encodes an image into a memory buffer.

The function imencode compresses the image and stores it in the memory buffer that is resized to fit the
result. See cv::imwrite for the list of supported formats and flags description.

@param ext File extension that defines the output format. Must include a leading period.
@param img Image to be compressed.
@param buf Output buffer resized to fit the compressed image.
@param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.
*/
CV_EXPORTS_W bool imencode( const String& ext, InputArray img,
                            CV_OUT std::vector<uchar>& buf,
                            const std::vector<int>& params = std::vector<int>());

/** @brief Encodes array of images into a memory buffer.

The function is analog to cv::imencode for in-memory multi-page image compression.
See cv::imwrite for the list of supported formats and flags description.

@param ext File extension that defines the output format. Must include a leading period.
@param imgs Vector of images to be written.
@param buf Output buffer resized to fit the compressed data.
@param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.
*/
CV_EXPORTS_W bool imencodemulti( const String& ext, InputArrayOfArrays imgs,
                                 CV_OUT std::vector<uchar>& buf,
                                 const std::vector<int>& params = std::vector<int>());

/** @brief Checks if the specified image file can be decoded by OpenCV.

The function haveImageReader checks if OpenCV is capable of reading the specified file.
This can be useful for verifying support for a given image format before attempting to load an image.

@param filename The name of the file to be checked.
@return true if an image reader for the specified file is available and the file can be opened, false otherwise.

@note The function checks the availability of image codecs that are either built into OpenCV or dynamically loaded.
It does not check for the actual existence of the file but rather the ability to read the specified file type.
If the file cannot be opened or the format is unsupported, the function will return false.

@sa cv::haveImageWriter, cv::imread, cv::imdecode
*/
CV_EXPORTS_W bool haveImageReader( const String& filename );

/** @brief Checks if the specified image file or specified file extension can be encoded by OpenCV.

The function haveImageWriter checks if OpenCV is capable of writing images with the specified file extension.
This can be useful for verifying support for a given image format before attempting to save an image.

@param filename The name of the file or the file extension (e.g., ".jpg", ".png").
It is recommended to provide the file extension rather than the full file name.
@return true if an image writer for the specified extension is available, false otherwise.

@note The function checks the availability of image codecs that are either built into OpenCV or dynamically loaded.
It does not check for the actual existence of the file but rather the ability to write files of the given type.

@sa cv::haveImageReader, cv::imwrite, cv::imencode
*/
CV_EXPORTS_W bool haveImageWriter( const String& filename );

/** @brief To read multi-page images on demand

The ImageCollection class provides iterator API to read multi-page images on demand. Create iterator
to the collection of the images and iterate over the collection. Decode the necessary page with operator*.

The performance of page decoding is O(1) if collection is increment sequentially. If the user wants to access random page,
then the time Complexity is O(n) because the collection has to be reinitialized every time in order to go to the correct page.
However, the intermediate pages are not decoded during the process, so typically it's quite fast.
This is required because multi-page codecs does not support going backwards.
After decoding the one page, it is stored inside the collection cache. Hence, trying to get Mat object from already decoded page is O(1).
If you need memory, you can use .releaseCache() method to release cached index.
The space complexity is O(n) if all pages are decoded into memory. The user is able to decode and release images on demand.
*/
class CV_EXPORTS ImageCollection {
public:
    struct CV_EXPORTS iterator {
        iterator(ImageCollection* col);
        iterator(ImageCollection* col, int end);
        Mat& operator*();
        Mat* operator->();
        iterator& operator++();
        iterator operator++(int);
        friend bool operator== (const iterator& a, const iterator& b) { return a.m_curr == b.m_curr; }
        friend bool operator!= (const iterator& a, const iterator& b) { return a.m_curr != b.m_curr; }

    private:
        ImageCollection* m_pCollection;
        int m_curr;
    };

    ImageCollection();
    ImageCollection(const String& filename, int flags);
    void init(const String& img, int flags);
    size_t size() const;
    const Mat& at(int index);
    const Mat& operator[](int index);
    void releaseCache(int index);
    iterator begin();
    iterator end();

    class Impl;
    Ptr<Impl> getImpl();
protected:
    Ptr<Impl> pImpl;
};

//! @} imgcodecs

} // cv

#endif //OPENCV_IMGCODECS_HPP
