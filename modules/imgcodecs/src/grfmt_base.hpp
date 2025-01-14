// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef _GRFMT_BASE_H_
#define _GRFMT_BASE_H_

#include "utils.hpp"
#include "bitstrm.hpp"
#include "exif.hpp"

namespace cv
{

class BaseImageDecoder;
class BaseImageEncoder;
typedef Ptr<BaseImageEncoder> ImageEncoder;
typedef Ptr<BaseImageDecoder> ImageDecoder;

/**
 * @brief Base class for image decoders.
 *
 * The BaseImageDecoder class provides an abstract interface for decoding various image formats.
 * It defines common functionality like setting the image source, reading image headers,
 * and handling EXIF metadata. Derived classes must implement methods for reading image headers
 * and image data to handle format-specific decoding logic.
 */

class BaseImageDecoder {
public:
    /**
     * @brief Constructor for BaseImageDecoder.
     * Initializes the object and sets default values for member variables.
     */
    BaseImageDecoder();

    /**
     * @brief Virtual destructor for BaseImageDecoder.
     * Ensures proper cleanup of derived classes when deleted via a pointer to BaseImageDecoder.
     */
    virtual ~BaseImageDecoder() {}

    /**
     * @brief Get the width of the image.
     * @return The width of the image (in pixels).
     */
    int width() const { return m_width; }

    /**
     * @brief Get the height of the image.
     * @return The height of the image (in pixels).
     */
    int height() const { return m_height; }

    /**
     * @brief Get the number of frames in the image or animation.
     * @return The number of frames in the image.
     */
    size_t getFrameCount() const { return m_frame_count; }

    /**
     * @brief Get the type of the image (e.g., color format, depth).
     * @return The type of the image.
     */
    virtual int type() const { return m_type; }

    /**
     * @brief Fetch a specific EXIF tag from the image's metadata.
     * @param tag The EXIF tag to retrieve.
     * @return The EXIF entry corresponding to the tag.
     */
    ExifEntry_t getExifTag(const ExifTagName tag) const;

    /**
     * @brief Set the image source from a file.
     * @param filename The name of the file to load the image from.
     * @return true if the source was successfully set, false otherwise.
     */
    virtual bool setSource(const String& filename);

    /**
     * @brief Set the image source from a memory buffer.
     * @param buf The buffer containing the image data.
     * @return true if the source was successfully set, false otherwise.
     */
    virtual bool setSource(const Mat& buf);

    /**
     * @brief Set the scale factor for the image.
     * @param scale_denom The denominator of the scale factor (image is scaled down by 1/scale_denom).
     * @return The scale factor that was set.
     */
    virtual int setScale(const int& scale_denom);

    /**
     * @brief Read the image header to extract basic properties (width, height, type).
     * This is a pure virtual function that must be implemented by derived classes.
     * @return true if the header was successfully read, false otherwise.
     */
    virtual bool readHeader() = 0;

    /**
     * @brief Read the image data into a Mat object.
     * This is a pure virtual function that must be implemented by derived classes.
     * @param img The Mat object where the image data will be stored.
     * @return true if the data was successfully read, false otherwise.
     */
    virtual bool readData(Mat& img) = 0;

    /**
     * @brief Set whether to decode the image in RGB order instead of the default BGR.
     * @param useRGB If true, the image will be decoded in RGB order.
     */
    virtual void setRGB(bool useRGB);

    /**
     * @brief Advance to the next page or frame of the image, if applicable.
     * The default implementation does nothing and returns false.
     * @return true if there is another page/frame, false otherwise.
     */
    virtual bool nextPage() { return false; }

    /**
     * @brief Get the length of the format signature used to identify the image format.
     * @return The length of the signature.
     */
    virtual size_t signatureLength() const;

    /**
     * @brief Check if the provided signature matches the expected format signature.
     * @param signature The signature to check.
     * @return true if the signature matches, false otherwise.
     */
    virtual bool checkSignature(const String& signature) const;

    const Animation& animation() const { return m_animation; };

    /**
     * @brief Create and return a new instance of the derived image decoder.
     * @return A new ImageDecoder object.
     */
    virtual ImageDecoder newDecoder() const;

protected:
    int m_width;          ///< Width of the image (set by readHeader).
    int m_height;         ///< Height of the image (set by readHeader).
    int m_type;           ///< Image type (e.g., color depth, channel order).
    int m_scale_denom;    ///< Scale factor denominator for resizing the image.
    String m_filename;    ///< Name of the file that is being decoded.
    String m_signature;   ///< Signature for identifying the image format.
    Mat m_buf;            ///< Buffer holding the image data when loaded from memory.
    bool m_buf_supported; ///< Flag indicating whether buffer-based loading is supported.
    bool m_use_rgb;       ///< Flag indicating whether to decode the image in RGB order.
    ExifReader m_exif;    ///< Object for reading EXIF metadata from the image.
    size_t m_frame_count; ///< Number of frames in the image (for animations and multi-page images).
    Animation m_animation;
};


/**
 * @brief Base class for image encoders.
 *
 * The BaseImageEncoder class provides an abstract interface for encoding images in various formats.
 * It defines common functionality like setting the destination (file or memory buffer), checking if
 * the format supports a specific image depth, and writing image data. Derived classes must implement
 * methods like writing the image data to handle format-specific encoding logic.
 */
class BaseImageEncoder {
public:
    /**
     * @brief Constructor for BaseImageEncoder.
     * Initializes the object and sets default values for member variables.
     */
    BaseImageEncoder();

    /**
     * @brief Virtual destructor for BaseImageEncoder.
     * Ensures proper cleanup of derived classes when deleted via a pointer to BaseImageEncoder.
     */
    virtual ~BaseImageEncoder() {}

    /**
     * @brief Checks if the image format supports a specific image depth.
     * @param depth The depth (bit depth) of the image.
     * @return true if the format supports the specified depth, false otherwise.
     */
    virtual bool isFormatSupported(int depth) const;

    /**
     * @brief Set the destination for encoding as a file.
     * @param filename The name of the file to which the image will be written.
     * @return true if the destination was successfully set, false otherwise.
     */
    virtual bool setDestination(const String& filename);

    /**
     * @brief Set the destination for encoding as a memory buffer.
     * @param buf A reference to the buffer where the encoded image data will be stored.
     * @return true if the destination was successfully set, false otherwise.
     */
    virtual bool setDestination(std::vector<uchar>& buf);

    /**
     * @brief Encode and write the image data.
     * This is a pure virtual function that must be implemented by derived classes.
     * @param img The Mat object containing the image data to be encoded.
     * @param params A vector of parameters controlling the encoding process (e.g., compression level).
     * @return true if the image was successfully written, false otherwise.
     */
    virtual bool write(const Mat& img, const std::vector<int>& params) = 0;

    /**
     * @brief Encode and write multiple images (e.g., for animated formats).
     * By default, this method returns false, indicating that the format does not support multi-image encoding.
     * @param img_vec A vector of Mat objects containing the images to be encoded.
     * @param params A vector of parameters controlling the encoding process.
     * @return true if multiple images were successfully written, false otherwise.
     */
    virtual bool writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params);

    virtual bool writeanimation(const Animation& animation, const std::vector<int>& params);

    /**
     * @brief Get a description of the image encoder (e.g., the format it supports).
     * @return A string describing the encoder.
     */
    virtual String getDescription() const;

    /**
     * @brief Create and return a new instance of the derived image encoder.
     * @return A new ImageEncoder object.
     */
    virtual ImageEncoder newEncoder() const;

    /**
     * @brief Throw an exception based on the last error encountered during encoding.
     * This method can be used to propagate error conditions back to the caller.
     */
    virtual void throwOnEror() const;

protected:
    String m_description;    ///< Description of the encoder (e.g., format name, capabilities).
    String m_filename;       ///< Destination file name for encoded data.
    std::vector<uchar>* m_buf; ///< Pointer to the buffer for encoded data if using memory-based destination.
    bool m_buf_supported;    ///< Flag indicating whether buffer-based encoding is supported.
    String m_last_error;     ///< Stores the last error message encountered during encoding.
};

}

#endif/*_GRFMT_BASE_H_*/
