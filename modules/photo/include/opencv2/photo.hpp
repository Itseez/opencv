/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_PHOTO_HPP
#define OPENCV_PHOTO_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/**
@defgroup photo Computational Photography

This module includes photo processing algorithms
@{
    @defgroup photo_inpaint Inpainting
    @defgroup photo_denoise Denoising
    @defgroup photo_hdr HDR imaging

    This section describes high dynamic range imaging algorithms namely tonemapping, exposure alignment,
    camera calibration with multiple exposures and exposure fusion.

    @defgroup photo_decolor Contrast Preserving Decolorization

    Useful links:

    http://www.cse.cuhk.edu.hk/leojia/projects/color2gray/index.html

    @defgroup photo_clone Seamless Cloning

    Useful links:

    https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp

    @defgroup photo_render Non-Photorealistic Rendering

    Useful links:

    http://www.inf.ufrgs.br/~eslgastal/DomainTransform

    https://www.learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/

@}
  */

namespace cv
{

//! @addtogroup photo
//! @{

//! @addtogroup photo_inpaint
//! @{
//! the inpainting algorithm
enum
{
    INPAINT_NS    = 0, //!< Use Navier-Stokes based method
    INPAINT_TELEA = 1 //!< Use the algorithm proposed by Alexandru Telea @cite Telea04
};

/** @brief Restores the selected region in an image using the region neighborhood.

@param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.
@param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that
needs to be inpainted.
@param dst Output image with the same size and type as src .
@param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered
by the algorithm.
@param flags Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA

The function reconstructs the selected image area from the pixel near the area boundary. The
function may be used to remove dust and scratches from a scanned photo, or to remove undesirable
objects from still images or video. See <http://en.wikipedia.org/wiki/Inpainting> for more details.

@note
   -   An example using the inpainting technique can be found at
        opencv_source_code/samples/cpp/inpaint.cpp
   -   (Python) An example using the inpainting technique can be found at
        opencv_source_code/samples/python/inpaint.py
 */
CV_EXPORTS_W void inpaint( InputArray src, InputArray inpaintMask,
        OutputArray dst, double inpaintRadius, int flags );

//! @} photo_inpaint

//! @addtogroup photo_denoise
//! @{

/** @brief Perform image denoising using Non-local Means Denoising algorithm
<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational
optimizations. Noise expected to be a gaussian white noise

@param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
@param dst Output image with the same size and type as src .
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength. Big h value perfectly removes noise but also
removes image details, smaller h value preserves details but also preserves some noise

This function expected to be applied to grayscale images. For colored images look at
fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
image to CIELAB colorspace and then separately denoise L and AB components with different h
parameter.
 */
CV_EXPORTS_W void fastNlMeansDenoising( InputArray src, OutputArray dst, float h = 3,
        int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Perform image denoising using Non-local Means Denoising algorithm
<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational
optimizations. Noise expected to be a gaussian white noise

@param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
2-channel, 3-channel or 4-channel image.
@param dst Output image with the same size and type as src .
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Array of parameters regulating filter strength, either one
parameter applied to all channels or one per channel in dst. Big h value
perfectly removes noise but also removes image details, smaller h
value preserves details but also preserves some noise
@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1

This function expected to be applied to grayscale images. For colored images look at
fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
image to CIELAB colorspace and then separately denoise L and AB components with different h
parameter.
 */
CV_EXPORTS_W void fastNlMeansDenoising( InputArray src, OutputArray dst,
                                        const std::vector<float>& h,
                                        int templateWindowSize = 7, int searchWindowSize = 21,
                                        int normType = NORM_L2);

/** @brief Modification of fastNlMeansDenoising function for colored images

@param src Input 8-bit 3-channel image.
@param dst Output image with the same size and type as src .
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
removes noise but also removes image details, smaller h value preserves details but also preserves
some noise
@param hColor The same as h but for color components. For most images value equals 10
will be enough to remove colored noise and do not distort colors

The function converts image to CIELAB colorspace and then separately denoise L and AB components
with given h parameters using fastNlMeansDenoising function.
 */
CV_EXPORTS_W void fastNlMeansDenoisingColored( InputArray src, OutputArray dst,
        float h = 3, float hColor = 3,
        int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
captured in small period of time. For example video. This version of the function is for grayscale
images or for manual manipulation with colorspaces. See @cite Buades2005DenoisingIS for more details
(open access [here](https://static.aminer.org/pdf/PDF/000/317/196/spatio_temporal_wiener_filtering_of_image_sequences_using_a_parametric.pdf)).

@param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or
4-channel images sequence. All images should have the same type and
size.
@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
@param temporalWindowSize Number of surrounding images to use for target image denoising. Should
be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
imgToDenoiseIndex + temporalWindowSize / 2 from srcImgs will be used to denoise
srcImgs[imgToDenoiseIndex] image.
@param dst Output image with the same size and type as srcImgs images.
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength. Bigger h value
perfectly removes noise but also removes image details, smaller h
value preserves details but also preserves some noise
 */
CV_EXPORTS_W void fastNlMeansDenoisingMulti( InputArrayOfArrays srcImgs, OutputArray dst,
        int imgToDenoiseIndex, int temporalWindowSize,
        float h = 3, int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Modification of fastNlMeansDenoising function for images sequence where consecutive images have been
captured in small period of time. For example video. This version of the function is for grayscale
images or for manual manipulation with colorspaces. See @cite Buades2005DenoisingIS for more details
(open access [here](https://static.aminer.org/pdf/PDF/000/317/196/spatio_temporal_wiener_filtering_of_image_sequences_using_a_parametric.pdf)).

@param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
2-channel, 3-channel or 4-channel images sequence. All images should
have the same type and size.
@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
@param temporalWindowSize Number of surrounding images to use for target image denoising. Should
be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
imgToDenoiseIndex + temporalWindowSize / 2 from srcImgs will be used to denoise
srcImgs[imgToDenoiseIndex] image.
@param dst Output image with the same size and type as srcImgs images.
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Array of parameters regulating filter strength, either one
parameter applied to all channels or one per channel in dst. Big h value
perfectly removes noise but also removes image details, smaller h
value preserves details but also preserves some noise
@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1
 */
CV_EXPORTS_W void fastNlMeansDenoisingMulti( InputArrayOfArrays srcImgs, OutputArray dst,
                                             int imgToDenoiseIndex, int temporalWindowSize,
                                             const std::vector<float>& h,
                                             int templateWindowSize = 7, int searchWindowSize = 21,
                                             int normType = NORM_L2);

/** @brief Modification of fastNlMeansDenoisingMulti function for colored images sequences

@param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
size.
@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
@param temporalWindowSize Number of surrounding images to use for target image denoising. Should
be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
imgToDenoiseIndex + temporalWindowSize / 2 from srcImgs will be used to denoise
srcImgs[imgToDenoiseIndex] image.
@param dst Output image with the same size and type as srcImgs images.
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
removes noise but also removes image details, smaller h value preserves details but also preserves
some noise.
@param hColor The same as h but for color components.

The function converts images to CIELAB colorspace and then separately denoise L and AB components
with given h parameters using fastNlMeansDenoisingMulti function.
 */
CV_EXPORTS_W void fastNlMeansDenoisingColoredMulti( InputArrayOfArrays srcImgs, OutputArray dst,
        int imgToDenoiseIndex, int temporalWindowSize,
        float h = 3, float hColor = 3,
        int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Primal-dual algorithm is an algorithm for solving special types of variational problems (that is,
finding a function to minimize some functional). As the image denoising, in particular, may be seen
as the variational problem, primal-dual algorithm then can be used to perform denoising and this is
exactly what is implemented.

It should be noted, that this implementation was taken from the July 2013 blog entry
@cite MA13 , which also contained (slightly more general) ready-to-use source code on Python.
Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky at the end
of July 2013 and finally it was slightly adapted by later authors.

Although the thorough discussion and justification of the algorithm involved may be found in
@cite ChambolleEtAl, it might make sense to skim over it here, following @cite MA13 . To begin
with, we consider the 1-byte gray-level images as the functions from the rectangular domain of
pixels (it may be seen as set
\f$\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}\f$ for some
\f$m,\;n\in\mathbb{N}\f$) into \f$\{0,1,\dots,255\}\f$. We shall denote the noised images as \f$f_i\f$ and with
this view, given some image \f$x\f$ of the same size, we may measure how bad it is by the formula

\f[\left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|\f]

\f$\|\|\cdot\|\|\f$ here denotes \f$L_2\f$-norm and as you see, the first addend states that we want our
image to be smooth (ideally, having zero gradient, thus being constant) and the second states that
we want our result to be close to the observations we've got. If we treat \f$x\f$ as a function, this is
exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes into play.

@param observations This array should contain one or more noised versions of the image that is to
be restored.
@param result Here the denoised image will be stored. There is no need to do pre-allocation of
storage space, as it will be automatically allocated, if necessary.
@param lambda Corresponds to \f$\lambda\f$ in the formulas above. As it is enlarged, the smooth
(blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly
speaking, as it becomes smaller, the result will be more blur but more sever outliers will be
removed.
@param niters Number of iterations that the algorithm will run. Of course, as more iterations as
better, but it is hard to quantitatively refine this statement, so just use the default and
increase it if the results are poor.
 */
CV_EXPORTS_W void denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda=1.0, int niters=30);

//! @} photo_denoise

//! @addtogroup photo_hdr
//! @{

enum { LDR_SIZE = 256 };

/** @brief Base class for tonemapping algorithms - tools that are used to map HDR image to 8-bit range.
 */
class CV_EXPORTS_W Tonemap : public Algorithm
{
public:
    /** @brief Tonemaps image

    @param src source image - CV_32FC3 Mat (float 32 bits 3 channels)
    @param dst destination image - CV_32FC3 Mat with values in [0, 1] range
     */
    CV_WRAP virtual void process(InputArray src, OutputArray dst) = 0;

    CV_WRAP virtual float getGamma() const = 0;
    CV_WRAP virtual void setGamma(float gamma) = 0;
};

/** @brief Creates simple linear mapper with gamma correction

@param gamma positive value for gamma correction. Gamma value of 1.0 implies no correction, gamma
equal to 2.2f is suitable for most displays.
Generally gamma \> 1 brightens the image and gamma \< 1 darkens it.
 */
CV_EXPORTS_W Ptr<Tonemap> createTonemap(float gamma = 1.0f);

/** @brief Adaptive logarithmic mapping is a fast global tonemapping algorithm that scales the image in
logarithmic domain.

Since it's a global operator the same function is applied to all the pixels, it is controlled by the
bias parameter.

Optional saturation enhancement is possible as described in @cite FL02 .

For more information see @cite DM03 .
 */
class CV_EXPORTS_W TonemapDrago : public Tonemap
{
public:

    CV_WRAP virtual float getSaturation() const = 0;
    CV_WRAP virtual void setSaturation(float saturation) = 0;

    CV_WRAP virtual float getBias() const = 0;
    CV_WRAP virtual void setBias(float bias) = 0;
};

/** @brief Creates TonemapDrago object

@param gamma gamma value for gamma correction. See createTonemap
@param saturation positive saturation enhancement value. 1.0 preserves saturation, values greater
than 1 increase saturation and values less than 1 decrease it.
@param bias value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give best
results, default value is 0.85.
 */
CV_EXPORTS_W Ptr<TonemapDrago> createTonemapDrago(float gamma = 1.0f, float saturation = 1.0f, float bias = 0.85f);


/** @brief This is a global tonemapping operator that models human visual system.

Mapping function is controlled by adaptation parameter, that is computed using light adaptation and
color adaptation.

For more information see @cite RD05 .
 */
class CV_EXPORTS_W TonemapReinhard : public Tonemap
{
public:
    CV_WRAP virtual float getIntensity() const = 0;
    CV_WRAP virtual void setIntensity(float intensity) = 0;

    CV_WRAP virtual float getLightAdaptation() const = 0;
    CV_WRAP virtual void setLightAdaptation(float light_adapt) = 0;

    CV_WRAP virtual float getColorAdaptation() const = 0;
    CV_WRAP virtual void setColorAdaptation(float color_adapt) = 0;
};

/** @brief Creates TonemapReinhard object

@param gamma gamma value for gamma correction. See createTonemap
@param intensity result intensity in [-8, 8] range. Greater intensity produces brighter results.
@param light_adapt light adaptation in [0, 1] range. If 1 adaptation is based only on pixel
value, if 0 it's global, otherwise it's a weighted mean of this two cases.
@param color_adapt chromatic adaptation in [0, 1] range. If 1 channels are treated independently,
if 0 adaptation level is the same for each channel.
 */
CV_EXPORTS_W Ptr<TonemapReinhard>
createTonemapReinhard(float gamma = 1.0f, float intensity = 0.0f, float light_adapt = 1.0f, float color_adapt = 0.0f);

/** @brief This algorithm transforms image to contrast using gradients on all levels of gaussian pyramid,
transforms contrast values to HVS response and scales the response. After this the image is
reconstructed from new contrast values.

For more information see @cite MM06 .
 */
class CV_EXPORTS_W TonemapMantiuk : public Tonemap
{
public:
    CV_WRAP virtual float getScale() const = 0;
    CV_WRAP virtual void setScale(float scale) = 0;

    CV_WRAP virtual float getSaturation() const = 0;
    CV_WRAP virtual void setSaturation(float saturation) = 0;
};

/** @brief Creates TonemapMantiuk object

@param gamma gamma value for gamma correction. See createTonemap
@param scale contrast scale factor. HVS response is multiplied by this parameter, thus compressing
dynamic range. Values from 0.6 to 0.9 produce best results.
@param saturation saturation enhancement value. See createTonemapDrago
 */
CV_EXPORTS_W Ptr<TonemapMantiuk>
createTonemapMantiuk(float gamma = 1.0f, float scale = 0.7f, float saturation = 1.0f);

/** @brief The base class for algorithms that align images of the same scene with different exposures
 */
class CV_EXPORTS_W AlignExposures : public Algorithm
{
public:
    /** @brief Aligns images

    @param src vector of input images
    @param dst vector of aligned images
    @param times vector of exposure time values for each image
    @param response 256x1 matrix with inverse camera response function for each pixel value, it should
    have the same number of channels as images.
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, std::vector<Mat>& dst,
                                 InputArray times, InputArray response) = 0;
};

/** @brief This algorithm converts images to median threshold bitmaps (1 for pixels brighter than median
luminance and 0 otherwise) and than aligns the resulting bitmaps using bit operations.

It is invariant to exposure, so exposure values and camera response are not necessary.

In this implementation new image regions are filled with zeros.

For more information see @cite GW03 .
 */
class CV_EXPORTS_W AlignMTB : public AlignExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, std::vector<Mat>& dst,
                                 InputArray times, InputArray response) CV_OVERRIDE = 0;

    /** @brief Short version of process, that doesn't take extra arguments.

    @param src vector of input images
    @param dst vector of aligned images
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, std::vector<Mat>& dst) = 0;

    /** @brief Calculates shift between two images, i. e. how to shift the second image to correspond it with the
    first.

    @param img0 first image
    @param img1 second image
     */
    CV_WRAP virtual Point calculateShift(InputArray img0, InputArray img1) = 0;
    /** @brief Helper function, that shift Mat filling new regions with zeros.

    @param src input image
    @param dst result image
    @param shift shift value
     */
    CV_WRAP virtual void shiftMat(InputArray src, OutputArray dst, const Point shift) = 0;
    /** @brief Computes median threshold and exclude bitmaps of given image.

    @param img input image
    @param tb median threshold bitmap
    @param eb exclude bitmap
     */
    CV_WRAP virtual void computeBitmaps(InputArray img, OutputArray tb, OutputArray eb) = 0;

    CV_WRAP virtual int getMaxBits() const = 0;
    CV_WRAP virtual void setMaxBits(int max_bits) = 0;

    CV_WRAP virtual int getExcludeRange() const = 0;
    CV_WRAP virtual void setExcludeRange(int exclude_range) = 0;

    CV_WRAP virtual bool getCut() const = 0;
    CV_WRAP virtual void setCut(bool value) = 0;
};

/** @brief Creates AlignMTB object

@param max_bits logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 are
usually good enough (31 and 63 pixels shift respectively).
@param exclude_range range for exclusion bitmap that is constructed to suppress noise around the
median value.
@param cut if true cuts images, otherwise fills the new regions with zeros.
 */
CV_EXPORTS_W Ptr<AlignMTB> createAlignMTB(int max_bits = 6, int exclude_range = 4, bool cut = true);

/** @brief The base class for camera response calibration algorithms.
 */
class CV_EXPORTS_W CalibrateCRF : public Algorithm
{
public:
    /** @brief Recovers inverse camera response.

    @param src vector of input images
    @param dst 256x1 matrix with inverse camera response function
    @param times vector of exposure time values for each image
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst, InputArray times) = 0;
};

/** @brief Inverse camera response function is extracted for each brightness value by minimizing an objective
function as linear system. Objective function is constructed using pixel values on the same position
in all images, extra term is added to make the result smoother.

For more information see @cite DM97 .
 */
class CV_EXPORTS_W CalibrateDebevec : public CalibrateCRF
{
public:
    CV_WRAP virtual float getLambda() const = 0;
    CV_WRAP virtual void setLambda(float lambda) = 0;

    CV_WRAP virtual int getSamples() const = 0;
    CV_WRAP virtual void setSamples(int samples) = 0;

    CV_WRAP virtual bool getRandom() const = 0;
    CV_WRAP virtual void setRandom(bool random) = 0;
};

/** @brief Creates CalibrateDebevec object

@param samples number of pixel locations to use
@param lambda smoothness term weight. Greater values produce smoother results, but can alter the
response.
@param random if true sample pixel locations are chosen at random, otherwise they form a
rectangular grid.
 */
CV_EXPORTS_W Ptr<CalibrateDebevec> createCalibrateDebevec(int samples = 70, float lambda = 10.0f, bool random = false);

/** @brief Inverse camera response function is extracted for each brightness value by minimizing an objective
function as linear system. This algorithm uses all image pixels.

For more information see @cite RB99 .
 */
class CV_EXPORTS_W CalibrateRobertson : public CalibrateCRF
{
public:
    CV_WRAP virtual int getMaxIter() const = 0;
    CV_WRAP virtual void setMaxIter(int max_iter) = 0;

    CV_WRAP virtual float getThreshold() const = 0;
    CV_WRAP virtual void setThreshold(float threshold) = 0;

    CV_WRAP virtual Mat getRadiance() const = 0;
};

/** @brief Creates CalibrateRobertson object

@param max_iter maximal number of Gauss-Seidel solver iterations.
@param threshold target difference between results of two successive steps of the minimization.
 */
CV_EXPORTS_W Ptr<CalibrateRobertson> createCalibrateRobertson(int max_iter = 30, float threshold = 0.01f);

/** @brief The base class algorithms that can merge exposure sequence to a single image.
 */
class CV_EXPORTS_W MergeExposures : public Algorithm
{
public:
    /** @brief Merges images.

    @param src vector of input images
    @param dst result image
    @param times vector of exposure time values for each image
    @param response 256x1 matrix with inverse camera response function for each pixel value, it should
    have the same number of channels as images.
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) = 0;
};

/** @brief The resulting HDR image is calculated as weighted average of the exposures considering exposure
values and camera response.

For more information see @cite DM97 .
 */
class CV_EXPORTS_W MergeDebevec : public MergeExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) CV_OVERRIDE = 0;
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst, InputArray times) = 0;
};

/** @brief Creates MergeDebevec object
 */
CV_EXPORTS_W Ptr<MergeDebevec> createMergeDebevec();

/** @brief Pixels are weighted using contrast, saturation and well-exposedness measures, than images are
combined using laplacian pyramids.

The resulting image weight is constructed as weighted average of contrast, saturation and
well-exposedness measures.

The resulting image doesn't require tonemapping and can be converted to 8-bit image by multiplying
by 255, but it's recommended to apply gamma correction and/or linear tonemapping.

For more information see @cite MK07 .
 */
class CV_EXPORTS_W MergeMertens : public MergeExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) CV_OVERRIDE = 0;
    /** @brief Short version of process, that doesn't take extra arguments.

    @param src vector of input images
    @param dst result image
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst) = 0;

    CV_WRAP virtual float getContrastWeight() const = 0;
    CV_WRAP virtual void setContrastWeight(float contrast_weiht) = 0;

    CV_WRAP virtual float getSaturationWeight() const = 0;
    CV_WRAP virtual void setSaturationWeight(float saturation_weight) = 0;

    CV_WRAP virtual float getExposureWeight() const = 0;
    CV_WRAP virtual void setExposureWeight(float exposure_weight) = 0;
};

/** @brief Creates MergeMertens object

@param contrast_weight contrast measure weight. See MergeMertens.
@param saturation_weight saturation measure weight
@param exposure_weight well-exposedness measure weight
 */
CV_EXPORTS_W Ptr<MergeMertens>
createMergeMertens(float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f);

/** @brief The resulting HDR image is calculated as weighted average of the exposures considering exposure
values and camera response.

For more information see @cite RB99 .
 */
class CV_EXPORTS_W MergeRobertson : public MergeExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) CV_OVERRIDE = 0;
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst, InputArray times) = 0;
};

/** @brief Creates MergeRobertson object
 */
CV_EXPORTS_W Ptr<MergeRobertson> createMergeRobertson();

//! @} photo_hdr

//! @addtogroup photo_decolor
//! @{

/** @brief Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylized
black-and-white photograph rendering, and in many single channel image processing applications
@cite CL12 .

@param src Input 8-bit 3-channel image.
@param grayscale Output 8-bit 1-channel image.
@param color_boost Output 8-bit 3-channel image.

This function is to be applied on color images.
 */
CV_EXPORTS_W void decolor( InputArray src, OutputArray grayscale, OutputArray color_boost);

//! @} photo_decolor

//! @addtogroup photo_clone
//! @{


//! Flags for the seamlessClone algorithm
enum SeamlessCloneFlags
{
    /**
    @brief Normal seamless cloning.
    This method is ideal for inserting objects with complex outlines into a new background.
    It preserves the original appearance and lighting of the inserted object, ensuring a natural blend.
     */
    NORMAL_CLONE = 1,

    /**
    @brief Mixed seamless cloning.
    This method addresses cases where simple color-based selection or alpha masking is time-consuming
    and may result in undesirable halos. By combining structure from the source and texture from the
    destination, mixed seamless cloning is highly effective, even with loosely defined selections.
     */
    MIXED_CLONE = 2,

    /**
    @brief Monochrome transfer cloning.
    This method allows users to replace specific features of an object, such as grayscale textures
    or patterns, with alternative features. It is particularly useful for artistic effects or
    targeted object modifications.
     */
    MONOCHROME_TRANSFER = 3,

    /**
    @brief Enhanced normal seamless cloning.
    Similar to `NORMAL_CLONE`, but with an advanced approach to ROI (Region of Interest) calculation.
    This mode processes a larger source region by considering the entire mask area instead of only
    the bounding rectangle of non-zero pixels.
     */
    NORMAL_CLONE_WIDE = 9,

    /**
    @brief Enhanced mixed seamless cloning.
    Similar to `MIXED_CLONE`, but with an advanced approach to ROI (Region of Interest) calculation.
    This mode processes a larger source region by considering the entire mask area instead of only
    the bounding rectangle of non-zero pixels.
     */
    MIXED_CLONE_WIDE = 10,

    /**
    @brief Enhanced monochrome transfer cloning.
    Similar to `MONOCHROME_TRANSFER`, but with an advanced approach to ROI (Region of Interest) calculation.
    This mode processes a larger source region by considering the entire mask area instead of only
    the bounding rectangle of non-zero pixels.
     */
    MONOCHROME_TRANSFER_WIDE = 11
};


/** @example samples/cpp/tutorial_code/photo/seamless_cloning/cloning_demo.cpp
An example using seamlessClone function
*/
/** @example samples/cpp/snippets/cloning_demo.cpp
An example using illuminationChange, colorChange, seamlessClone, textureFlattening functions
*/
/** @brief Performs seamless cloning to blend a region from a source image into a destination image.
This function is designed for local image editing, allowing changes restricted to a region
(manually selected as the ROI) to be applied effortlessly and seamlessly. These changes can
range from slight distortions to complete replacement by novel content @cite PM03.

@param src The source image (8-bit 3-channel), from which a region will be blended into the destination.
@param dst The destination image (8-bit 3-channel), where the src image will be blended.
@param mask A binary mask (8-bit, 1, 3, or 4-channel) specifying the region in the source image to blend.
Non-zero pixels indicate the region to be blended. If an empty Mat is provided, a mask with
all non-zero pixels is created internally.
@param p The point where the center of the src image is placed in the dst image.
@param blend The output image that stores the result of the seamless cloning. It has the same size and type as `dst`.
@param flags Flags that control the type of cloning method, can take values of `cv::SeamlessCloneFlags`.
 */
CV_EXPORTS_W void seamlessClone( InputArray src, InputArray dst, InputArray mask, Point p,
        OutputArray blend, int flags);

/** @brief Given an original color image, two differently colored versions of this image can be mixed
seamlessly.

@param src Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param dst Output image with the same size and type as src .
@param red_mul R-channel multiply factor.
@param green_mul G-channel multiply factor.
@param blue_mul B-channel multiply factor.

Multiplication factor is between .5 to 2.5.
 */
CV_EXPORTS_W void colorChange(InputArray src, InputArray mask, OutputArray dst, float red_mul = 1.0f,
        float green_mul = 1.0f, float blue_mul = 1.0f);

/** @brief Applying an appropriate non-linear transformation to the gradient field inside the selection and
then integrating back with a Poisson solver, modifies locally the apparent illumination of an image.

@param src Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param dst Output image with the same size and type as src.
@param alpha Value ranges between 0-2.
@param beta Value ranges between 0-2.

This is useful to highlight under-exposed foreground objects or to reduce specular reflections.
 */
CV_EXPORTS_W void illuminationChange(InputArray src, InputArray mask, OutputArray dst,
        float alpha = 0.2f, float beta = 0.4f);

/** @brief By retaining only the gradients at edge locations, before integrating with the Poisson solver, one
washes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge %Detector is used.

@param src Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param dst Output image with the same size and type as src.
@param low_threshold %Range from 0 to 100.
@param high_threshold Value \> 100.
@param kernel_size The size of the Sobel kernel to be used.

@note
The algorithm assumes that the color of the source image is close to that of the destination. This
assumption means that when the colors don't match, the source image color gets tinted toward the
color of the destination image.
 */
CV_EXPORTS_W void textureFlattening(InputArray src, InputArray mask, OutputArray dst,
        float low_threshold = 30, float high_threshold = 45,
        int kernel_size = 3);

//! @} photo_clone

//! @addtogroup photo_render
//! @{

//! Edge preserving filters
enum
{
    RECURS_FILTER = 1, //!< Recursive Filtering
    NORMCONV_FILTER = 2 //!< Normalized Convolution Filtering
};

/** @brief Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing
filters are used in many different applications @cite EM11 .

@param src Input 8-bit 3-channel image.
@param dst Output 8-bit 3-channel image.
@param flags Edge preserving filters: cv::RECURS_FILTER or cv::NORMCONV_FILTER
@param sigma_s %Range between 0 to 200.
@param sigma_r %Range between 0 to 1.
 */
CV_EXPORTS_W void edgePreservingFilter(InputArray src, OutputArray dst, int flags = 1,
        float sigma_s = 60, float sigma_r = 0.4f);

/** @brief This filter enhances the details of a particular image.

@param src Input 8-bit 3-channel image.
@param dst Output image with the same size and type as src.
@param sigma_s %Range between 0 to 200.
@param sigma_r %Range between 0 to 1.
 */
CV_EXPORTS_W void detailEnhance(InputArray src, OutputArray dst, float sigma_s = 10,
        float sigma_r = 0.15f);

/** @example samples/cpp/tutorial_code/photo/non_photorealistic_rendering/npr_demo.cpp
An example using non-photorealistic line drawing functions
*/
/** @brief Pencil-like non-photorealistic line drawing

@param src Input 8-bit 3-channel image.
@param dst1 Output 8-bit 1-channel image.
@param dst2 Output image with the same size and type as src.
@param sigma_s %Range between 0 to 200.
@param sigma_r %Range between 0 to 1.
@param shade_factor %Range between 0 to 0.1.
 */
CV_EXPORTS_W void pencilSketch(InputArray src, OutputArray dst1, OutputArray dst2,
        float sigma_s = 60, float sigma_r = 0.07f, float shade_factor = 0.02f);

/** @brief Stylization aims to produce digital imagery with a wide variety of effects not focused on
photorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low
contrast while preserving, or enhancing, high-contrast features.

@param src Input 8-bit 3-channel image.
@param dst Output image with the same size and type as src.
@param sigma_s %Range between 0 to 200.
@param sigma_r %Range between 0 to 1.
 */
CV_EXPORTS_W void stylization(InputArray src, OutputArray dst, float sigma_s = 60,
        float sigma_r = 0.45f);

//! @} photo_render

//! @} photo

} // cv

#endif
