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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_OBJDETECT_HPP
#define OPENCV_OBJDETECT_HPP

#include "opencv2/core.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/graphical_code_detector.hpp"

/**
@defgroup objdetect Object Detection

@{
    @defgroup objdetect_cascade_classifier Cascade Classifier for Object Detection

    The object detector described below has been initially proposed by Paul Viola @cite Viola01 and
    improved by Rainer Lienhart @cite Lienhart02 .

    First, a classifier (namely a *cascade of boosted classifiers working with haar-like features*) is
    trained with a few hundred sample views of a particular object (i.e., a face or a car), called
    positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary
    images of the same size.

    After a classifier is trained, it can be applied to a region of interest (of the same size as used
    during the training) in an input image. The classifier outputs a "1" if the region is likely to show
    the object (i.e., face/car), and "0" otherwise. To search for the object in the whole image one can
    move the search window across the image and check every location using the classifier. The
    classifier is designed so that it can be easily "resized" in order to be able to find the objects of
    interest at different sizes, which is more efficient than resizing the image itself. So, to find an
    object of an unknown size in the image the scan procedure should be done several times at different
    scales.

    The word "cascade" in the classifier name means that the resultant classifier consists of several
    simpler classifiers (*stages*) that are applied subsequently to a region of interest until at some
    stage the candidate is rejected or all the stages are passed. The word "boosted" means that the
    classifiers at every stage of the cascade are complex themselves and they are built out of basic
    classifiers using one of four different boosting techniques (weighted voting). Currently Discrete
    Adaboost, Real Adaboost, Gentle Adaboost and Logitboost are supported. The basic classifiers are
    decision-tree classifiers with at least 2 leaves. Haar-like features are the input to the basic
    classifiers, and are calculated as described below. The current algorithm uses the following
    Haar-like features:

    ![image](pics/haarfeatures.png)

    The feature used in a particular classifier is specified by its shape (1a, 2b etc.), position within
    the region of interest and the scale (this scale is not the same as the scale used at the detection
    stage, though these two scales are multiplied). For example, in the case of the third line feature
    (2c) the response is calculated as the difference between the sum of image pixels under the
    rectangle covering the whole feature (including the two white stripes and the black stripe in the
    middle) and the sum of the image pixels under the black stripe multiplied by 3 in order to
    compensate for the differences in the size of areas. The sums of pixel values over a rectangular
    regions are calculated rapidly using integral images (see below and the integral description).

    Check @ref tutorial_cascade_classifier "the corresponding tutorial" for more details.

    The following reference is for the detection part only. There is a separate application called
    opencv_traincascade that can train a cascade of boosted classifiers from a set of samples.

    @note In the new C++ interface it is also possible to use LBP (local binary pattern) features in
    addition to Haar-like features. .. [Viola01] Paul Viola and Michael J. Jones. Rapid Object Detection
    using a Boosted Cascade of Simple Features. IEEE CVPR, 2001. The paper is available online at
    <https://github.com/SvHey/thesis/blob/master/Literature/ObjectDetection/violaJones_CVPR2001.pdf>

    @defgroup objdetect_hog HOG (Histogram of Oriented Gradients) descriptor and object detector
    @defgroup objdetect_barcode Barcode detection and decoding
    @defgroup objdetect_qrcode QRCode detection and encoding
    @defgroup objdetect_dnn_face DNN-based face detection and recognition

    Check @ref tutorial_dnn_face "the corresponding tutorial" for more details.

    @defgroup objdetect_common Common functions and classes
    @defgroup objdetect_aruco ArUco markers and boards detection for robust camera pose estimation
    @{
        ArUco Marker Detection
        Square fiducial markers (also known as Augmented Reality Markers) are useful for easy,
        fast and robust camera pose estimation.

        The main functionality of ArucoDetector class is detection of markers in an image. If the markers are grouped
        as a board, then you can try to recover the missing markers with ArucoDetector::refineDetectedMarkers().
        ArUco markers can also be used for advanced chessboard corner finding. To do this, group the markers in the
        CharucoBoard and find the corners of the chessboard with the CharucoDetector::detectBoard().

        The implementation is based on the ArUco Library by R. Muñoz-Salinas and S. Garrido-Jurado @cite Aruco2014.

        Markers can also be detected based on the AprilTag 2 @cite wang2016iros fiducial detection method.

        @sa @cite Aruco2014
        This code has been originally developed by Sergio Garrido-Jurado as a project
        for Google Summer of Code 2015 (GSoC 15).
    @}

@}
 */

typedef struct CvHaarClassifierCascade CvHaarClassifierCascade;

namespace cv
{

//! @addtogroup objdetect_common
//! @{

///////////////////////////// Object Detection ////////////////////////////

/** @brief This class is used for grouping object candidates detected by Cascade Classifier, HOG etc.

instance of the class is to be passed to cv::partition
 */
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps * ((std::min)(r1.width, r2.width) + (std::min)(r1.height, r2.height)) * 0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

/** @brief Groups the object candidate rectangles.

@param rectList Input/output vector of rectangles. Output vector includes retained and grouped
rectangles. (The Python list is not modified in place.)
@param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a
group of rectangles to retain it.
@param eps Relative difference between sides of the rectangles to merge them into a group.

The function is a wrapper for the generic function partition . It clusters all the input rectangles
using the rectangle equivalence criteria that combines rectangles with similar sizes and similar
locations. The similarity is defined by eps. When eps=0 , no clustering is done at all. If
\f$\texttt{eps}\rightarrow +\inf\f$ , all the rectangles are put in one cluster. Then, the small
clusters containing less than or equal to groupThreshold rectangles are rejected. In each other
cluster, the average rectangle is computed and put into the output rectangle list.
 */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights,
                                  int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold,
                                  double eps, std::vector<int>* weights, std::vector<double>* levelWeights );
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                                  std::vector<double>& levelWeights, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                                            std::vector<double>& foundScales,
                                            double detectThreshold = 0.0, Size winDetSize = Size(64, 128));
//! @}

//! @addtogroup objdetect_cascade_classifier
//! @{

template<> struct DefaultDeleter<CvHaarClassifierCascade>{ CV_EXPORTS void operator ()(CvHaarClassifierCascade* obj) const; };

enum { CASCADE_DO_CANNY_PRUNING    = 1,
       CASCADE_SCALE_IMAGE         = 2,
       CASCADE_FIND_BIGGEST_OBJECT = 4,
       CASCADE_DO_ROUGH_SEARCH     = 8
     };

class CV_EXPORTS_W BaseCascadeClassifier : public Algorithm
{
public:
    virtual ~BaseCascadeClassifier();
    virtual bool empty() const CV_OVERRIDE = 0;
    virtual bool load( const String& filename ) = 0;
    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           CV_OUT std::vector<int>& numDetections,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                                   CV_OUT std::vector<Rect>& objects,
                                   CV_OUT std::vector<int>& rejectLevels,
                                   CV_OUT std::vector<double>& levelWeights,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   Size minSize, Size maxSize,
                                   bool outputRejectLevels ) = 0;

    virtual bool isOldFormatCascade() const = 0;
    virtual Size getOriginalWindowSize() const = 0;
    virtual int getFeatureType() const = 0;
    virtual void* getOldCascade() = 0;

    class CV_EXPORTS MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}
        virtual Mat generateMask(const Mat& src)=0;
        virtual void initializeMask(const Mat& /*src*/) { }
    };
    virtual void setMaskGenerator(const Ptr<MaskGenerator>& maskGenerator) = 0;
    virtual Ptr<MaskGenerator> getMaskGenerator() = 0;
};

/** @example samples/cpp/facedetect.cpp
This program demonstrates usage of the Cascade classifier class
\image html Cascade_Classifier_Tutorial_Result_Haar.jpg "Sample screenshot" width=321 height=254
*/
/** @brief Cascade classifier class for object detection.
 */
class CV_EXPORTS_W CascadeClassifier
{
public:
    CV_WRAP CascadeClassifier();
    /** @brief Loads a classifier from a file.

    @param filename Name of the file from which the classifier is loaded.
     */
    CV_WRAP CascadeClassifier(const String& filename);
    ~CascadeClassifier();
    /** @brief Checks whether the classifier has been loaded.
    */
    CV_WRAP bool empty() const;
    /** @brief Loads a classifier from a file.

    @param filename Name of the file from which the classifier is loaded. The file may contain an old
    HAAR classifier trained by the haartraining application or a new cascade classifier trained by the
    traincascade application.
     */
    CV_WRAP bool load( const String& filename );
    /** @brief Reads a classifier from a FileStorage node.

    @note The file may contain a new cascade classifier (trained by the traincascade application) only.
     */
    CV_WRAP bool read( const FileNode& node );

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.

    @param image Matrix of the type CV_8U containing an image where objects are detected.
    @param objects Vector of rectangles where each rectangle contains the detected object, the
    rectangles may be partially outside the original image.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param flags Parameter with the same meaning for an old cascade as in the function
    cvHaarDetectObjects. It is not used for a new cascade.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.
    @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
    */
    CV_WRAP void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size() );

    /** @overload
    @param image Matrix of the type CV_8U containing an image where objects are detected.
    @param objects Vector of rectangles where each rectangle contains the detected object, the
    rectangles may be partially outside the original image.
    @param numDetections Vector of detection numbers for the corresponding objects. An object's number
    of detections is the number of neighboring positively classified rectangles that were joined
    together to form the object.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param flags Parameter with the same meaning for an old cascade as in the function
    cvHaarDetectObjects. It is not used for a new cascade.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.
    @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
    */
    CV_WRAP_AS(detectMultiScale2) void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& numDetections,
                          double scaleFactor=1.1,
                          int minNeighbors=3, int flags=0,
                          Size minSize=Size(),
                          Size maxSize=Size() );

    /** @overload
    This function allows you to retrieve the final stage decision certainty of classification.
    For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
    For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    This value can then be used to separate strong from weaker classifications.

    A code sample on how to use it efficiently can be found below:
    @code
    Mat img;
    vector<double> weights;
    vector<int> levels;
    vector<Rect> detections;
    CascadeClassifier model("/path/to/your/model.xml");
    model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
    cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
    @endcode
    */
    CV_WRAP_AS(detectMultiScale3) void detectMultiScale( InputArray image,
                                  CV_OUT std::vector<Rect>& objects,
                                  CV_OUT std::vector<int>& rejectLevels,
                                  CV_OUT std::vector<double>& levelWeights,
                                  double scaleFactor = 1.1,
                                  int minNeighbors = 3, int flags = 0,
                                  Size minSize = Size(),
                                  Size maxSize = Size(),
                                  bool outputRejectLevels = false );

    CV_WRAP bool isOldFormatCascade() const;
    CV_WRAP Size getOriginalWindowSize() const;
    CV_WRAP int getFeatureType() const;
    void* getOldCascade();

    CV_WRAP static bool convert(const String& oldcascade, const String& newcascade);

    void setMaskGenerator(const Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator);
    Ptr<BaseCascadeClassifier::MaskGenerator> getMaskGenerator();

    Ptr<BaseCascadeClassifier> cc;
};

CV_EXPORTS Ptr<BaseCascadeClassifier::MaskGenerator> createFaceDetectionMaskGenerator();
//! @}

//! @addtogroup objdetect_hog
//! @{
//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

//! struct for detection region of interest (ROI)
struct DetectionROI
{
   //! scale(size) of the bounding box
   double scale;
   //! set of requested locations to be evaluated
   std::vector<cv::Point> locations;
   //! vector that will contain confidence values for each location
   std::vector<double> confidences;
};

/**@brief Implementation of HOG (Histogram of Oriented Gradients) descriptor and object detector.

the HOG descriptor algorithm introduced by Navneet Dalal and Bill Triggs @cite Dalal2005 .

useful links:

https://hal.inria.fr/inria-00548512/document/

https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

https://software.intel.com/en-us/ipp-dev-reference-histogram-of-oriented-gradients-hog-descriptor

http://www.learnopencv.com/histogram-of-oriented-gradients

http://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial

 */
struct CV_EXPORTS_W HOGDescriptor
{
public:
    enum HistogramNormType { L2Hys = 0 //!< Default histogramNormType
         };
    enum { DEFAULT_NLEVELS = 64 //!< Default nlevels value.
         };
    enum DescriptorStorageFormat { DESCR_FORMAT_COL_BY_COL, DESCR_FORMAT_ROW_BY_ROW };

    /**@brief Creates the HOG descriptor and detector with default parameters.

    aqual to HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9 )
    */
    CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
        cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
        histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
        free_coef(-1.f), nlevels(HOGDescriptor::DEFAULT_NLEVELS), signedGradient(false)
    {}

    /** @overload
    @param _winSize sets winSize with given value.
    @param _blockSize sets blockSize with given value.
    @param _blockStride sets blockStride with given value.
    @param _cellSize sets cellSize with given value.
    @param _nbins sets nbins with given value.
    @param _derivAperture sets derivAperture with given value.
    @param _winSigma sets winSigma with given value.
    @param _histogramNormType sets histogramNormType with given value.
    @param _L2HysThreshold sets L2HysThreshold with given value.
    @param _gammaCorrection sets gammaCorrection with given value.
    @param _nlevels sets nlevels with given value.
    @param _signedGradient sets signedGradient with given value.
    */
    CV_WRAP HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
                  Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
                  HOGDescriptor::HistogramNormType _histogramNormType=HOGDescriptor::L2Hys,
                  double _L2HysThreshold=0.2, bool _gammaCorrection=false,
                  int _nlevels=HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient=false)
    : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
    nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
    histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
    gammaCorrection(_gammaCorrection), free_coef(-1.f), nlevels(_nlevels), signedGradient(_signedGradient)
    {}

    /** @overload

    Creates the HOG descriptor and detector and loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file.
    @param filename The file name containing HOGDescriptor properties and coefficients for the linear SVM classifier.
    */
    CV_WRAP HOGDescriptor(const String& filename)
    {
        load(filename);
    }

    /** @overload
    @param d the HOGDescriptor which cloned to create a new one.
    */
    HOGDescriptor(const HOGDescriptor& d)
    {
        d.copyTo(*this);
    }

    /**@brief Default destructor.
    */
    virtual ~HOGDescriptor() {}

    /**@brief Returns the number of coefficients required for the classification.
    */
    CV_WRAP size_t getDescriptorSize() const;

    /** @brief Checks if detector size equal to descriptor size.
    */
    CV_WRAP bool checkDetectorSize() const;

    /** @brief Returns winSigma value
    */
    CV_WRAP double getWinSigma() const;

    /**@example samples/cpp/peopledetect.cpp
    */
    /**@brief Sets coefficients for the linear SVM classifier.
    @param svmdetector coefficients for the linear SVM classifier.
    */
    CV_WRAP virtual void setSVMDetector(InputArray svmdetector);

    /** @brief Reads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file node.
    @param fn File node
    */
    virtual bool read(FileNode& fn);

    /** @brief Stores HOGDescriptor parameters and coefficients for the linear SVM classifier in a file storage.
    @param fs File storage
    @param objname Object name
    */
    virtual void write(FileStorage& fs, const String& objname) const;

    /** @brief loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file
    @param filename Name of the file to read.
    @param objname The optional name of the node to read (if empty, the first top-level node will be used).
    */
    CV_WRAP virtual bool load(const String& filename, const String& objname = String());

    /** @brief saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file
    @param filename File name
    @param objname Object name
    */
    CV_WRAP virtual void save(const String& filename, const String& objname = String()) const;

    /** @brief clones the HOGDescriptor
    @param c cloned HOGDescriptor
    */
    virtual void copyTo(HOGDescriptor& c) const;

    /**@example samples/cpp/train_HOG.cpp
    */
    /** @brief Computes HOG descriptors of given image.
    @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
    @param descriptors Matrix of the type CV_32F
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param locations Vector of Point
    */
    CV_WRAP virtual void compute(InputArray img,
                         CV_OUT std::vector<float>& descriptors,
                         Size winStride = Size(), Size padding = Size(),
                         const std::vector<Point>& locations = std::vector<Point>()) const;

    /** @brief Performs object detection without a multi-scale window.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
    @param weights Vector that will contain confidence values for each detected object.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param searchLocations Vector of Point includes set of requested locations to be evaluated.
    */
    CV_WRAP virtual void detect(InputArray img, CV_OUT std::vector<Point>& foundLocations,
                        CV_OUT std::vector<double>& weights,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations = std::vector<Point>()) const;

    /** @brief Performs object detection without a multi-scale window.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param searchLocations Vector of Point includes locations to search.
    */
    virtual void detect(InputArray img, CV_OUT std::vector<Point>& foundLocations,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations=std::vector<Point>()) const;

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
    @param foundWeights Vector that will contain confidence values for each detected object.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param scale Coefficient of the detection window increase.
    @param groupThreshold Coefficient to regulate the similarity threshold. When detected, some objects can be covered
    by many rectangles. 0 means not to perform grouping.
    @param useMeanshiftGrouping indicates grouping algorithm
    */
    CV_WRAP virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
                                  Size winStride = Size(), Size padding = Size(), double scale = 1.05,
                                  double groupThreshold = 2.0, bool useMeanshiftGrouping = false) const;

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param winStride Window stride. It must be a multiple of block stride.
    @param padding Padding
    @param scale Coefficient of the detection window increase.
    @param groupThreshold Coefficient to regulate the similarity threshold. When detected, some objects can be covered
    by many rectangles. 0 means not to perform grouping.
    @param useMeanshiftGrouping indicates grouping algorithm
    */
    virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  double hitThreshold = 0, Size winStride = Size(),
                                  Size padding = Size(), double scale = 1.05,
                                  double groupThreshold = 2.0, bool useMeanshiftGrouping = false) const;

    /** @brief  Computes gradients and quantized gradient orientations.
    @param img Matrix contains the image to be computed
    @param grad Matrix of type CV_32FC2 contains computed gradients
    @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
    @param paddingTL Padding from top-left
    @param paddingBR Padding from bottom-right
    */
    CV_WRAP virtual void computeGradient(InputArray img, InputOutputArray grad, InputOutputArray angleOfs,
                                 Size paddingTL = Size(), Size paddingBR = Size()) const;

    /** @brief Returns coefficients of the classifier trained for people detection (for 64x128 windows).
    */
    CV_WRAP static std::vector<float> getDefaultPeopleDetector();

    /**@example samples/tapi/hog.cpp
    */
    /** @brief Returns coefficients of the classifier trained for people detection (for 48x96 windows).
    */
    CV_WRAP static std::vector<float> getDaimlerPeopleDetector();

    //! Detection window size. Align to block size and block stride. Default value is Size(64,128).
    CV_PROP Size winSize;

    //! Block size in pixels. Align to cell size. Default value is Size(16,16).
    CV_PROP Size blockSize;

    //! Block stride. It must be a multiple of cell size. Default value is Size(8,8).
    CV_PROP Size blockStride;

    //! Cell size. Default value is Size(8,8).
    CV_PROP Size cellSize;

    //! Number of bins used in the calculation of histogram of gradients. Default value is 9.
    CV_PROP int nbins;

    //! not documented
    CV_PROP int derivAperture;

    //! Gaussian smoothing window parameter.
    CV_PROP double winSigma;

    //! histogramNormType
    CV_PROP HOGDescriptor::HistogramNormType histogramNormType;

    //! L2-Hys normalization method shrinkage.
    CV_PROP double L2HysThreshold;

    //! Flag to specify whether the gamma correction preprocessing is required or not.
    CV_PROP bool gammaCorrection;

    //! coefficients for the linear SVM classifier.
    CV_PROP std::vector<float> svmDetector;

    //! coefficients for the linear SVM classifier used when OpenCL is enabled
    UMat oclSvmDetector;

    //! not documented
    float free_coef;

    //! Maximum number of detection window increases. Default value is 64
    CV_PROP int nlevels;

    //! Indicates signed gradient will be used or not
    CV_PROP bool signedGradient;

    /** @brief evaluate specified ROI and return confidence value for each location
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param locations Vector of Point
    @param foundLocations Vector of Point where each Point is detected object's top-left point.
    @param confidences confidences
    @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
    it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
    the free coefficient is omitted (which is allowed), you can specify it manually here
    @param winStride winStride
    @param padding padding
    */
    virtual void detectROI(InputArray img, const std::vector<cv::Point> &locations,
                                   CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
                                   double hitThreshold = 0, cv::Size winStride = Size(),
                                   cv::Size padding = Size()) const;

    /** @brief evaluate specified ROI and return confidence value for each location in multiple scales
    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
    @param locations Vector of DetectionROI
    @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified
    in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
    */
    virtual void detectMultiScaleROI(InputArray img,
                                     CV_OUT std::vector<cv::Rect>& foundLocations,
                                     std::vector<DetectionROI>& locations,
                                     double hitThreshold = 0,
                                     int groupThreshold = 0) const;

    /** @brief Groups the object candidate rectangles.
    @param rectList  Input/output vector of rectangles. Output vector includes retained and grouped rectangles. (The Python list is not modified in place.)
    @param weights Input/output vector of weights of rectangles. Output vector includes weights of retained and grouped rectangles. (The Python list is not modified in place.)
    @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
    @param eps Relative difference between sides of the rectangles to merge them into a group.
    */
    void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const;
};
//! @}

//! @addtogroup objdetect_qrcode
//! @{

class CV_EXPORTS_W QRCodeEncoder {
protected:
    QRCodeEncoder();  // use ::create()
public:
    virtual ~QRCodeEncoder();

    enum EncodeMode {
        MODE_AUTO              = -1,
        MODE_NUMERIC           = 1, // 0b0001
        MODE_ALPHANUMERIC      = 2, // 0b0010
        MODE_BYTE              = 4, // 0b0100
        MODE_ECI               = 7, // 0b0111
        MODE_KANJI             = 8, // 0b1000
        MODE_STRUCTURED_APPEND = 3  // 0b0011
    };

    enum CorrectionLevel {
        CORRECT_LEVEL_L = 0,
        CORRECT_LEVEL_M = 1,
        CORRECT_LEVEL_Q = 2,
        CORRECT_LEVEL_H = 3
    };

    enum ECIEncodings {
        ECI_UTF8 = 26
    };

    /** @brief QR code encoder parameters. */
    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();

        //! The optional version of QR code (by default - maximum possible depending on the length of the string).
        CV_PROP_RW int version;

        //! The optional level of error correction (by default - the lowest).
        CV_PROP_RW CorrectionLevel correction_level;

        //! The optional encoding mode - Numeric, Alphanumeric, Byte, Kanji, ECI or Structured Append.
        CV_PROP_RW EncodeMode mode;

        //! The optional number of QR codes to generate in Structured Append mode.
        CV_PROP_RW int structure_number;
    };

    /** @brief Constructor
    @param parameters QR code encoder parameters QRCodeEncoder::Params
    */
    static CV_WRAP
    Ptr<QRCodeEncoder> create(const QRCodeEncoder::Params& parameters = QRCodeEncoder::Params());

    /** @brief Generates QR code from input string.
     @param encoded_info Input string to encode.
     @param qrcode Generated QR code.
    */
    CV_WRAP virtual void encode(const String& encoded_info, OutputArray qrcode) = 0;

    /** @brief Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.
     @param encoded_info Input string to encode.
     @param qrcodes Vector of generated QR codes.
    */
    CV_WRAP virtual void encodeStructuredAppend(const String& encoded_info, OutputArrayOfArrays qrcodes) = 0;

};
class CV_EXPORTS_W_SIMPLE QRCodeDetector : public GraphicalCodeDetector
{
public:
    CV_WRAP QRCodeDetector();

    /** @brief sets the epsilon used during the horizontal scan of QR code stop marker detection.
     @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
    */
    CV_WRAP QRCodeDetector& setEpsX(double epsX);
    /** @brief sets the epsilon used during the vertical scan of QR code stop marker detection.
     @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
     */
    CV_WRAP QRCodeDetector& setEpsY(double epsY);

    /** @brief use markers to improve the position of the corners of the QR code
     *
     * alignmentMarkers using by default
     */
    CV_WRAP QRCodeDetector& setUseAlignmentMarkers(bool useAlignmentMarkers);

    /** @brief Decodes QR code on a curved surface in image once it's found by the detect() method.

     Returns UTF8-encoded output string or empty string if the code cannot be decoded.
     @param img grayscale or color (BGR) image containing QR code.
     @param points Quadrangle vertices found by detect() method (or some other algorithm).
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP cv::String decodeCurved(InputArray img, InputArray points, OutputArray straight_qrcode = noArray());

    /** @brief Both detects and decodes QR code on a curved surface

     @param img grayscale or color (BGR) image containing QR code.
     @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP std::string detectAndDecodeCurved(InputArray img, OutputArray points=noArray(),
                                              OutputArray straight_qrcode = noArray());
};

class CV_EXPORTS_W_SIMPLE QRCodeDetectorAruco : public GraphicalCodeDetector {
public:
    CV_WRAP QRCodeDetectorAruco();

    struct CV_EXPORTS_W_SIMPLE Params {
        CV_WRAP Params();

        /** @brief The minimum allowed pixel size of a QR module in the smallest image in the image pyramid, default 4.f */
        CV_PROP_RW float minModuleSizeInPyramid;

        /** @brief The maximum allowed relative rotation for finder patterns in the same QR code, default pi/12 */
        CV_PROP_RW float maxRotation;

        /** @brief The maximum allowed relative mismatch in module sizes for finder patterns in the same QR code, default 1.75f */
        CV_PROP_RW float maxModuleSizeMismatch;

        /** @brief The maximum allowed module relative mismatch for timing pattern module, default 2.f
         *
         * If relative mismatch of timing pattern module more this value, penalty points will be added.
         * If a lot of penalty points are added, QR code will be rejected. */
        CV_PROP_RW float maxTimingPatternMismatch;

        /** @brief The maximum allowed percentage of penalty points out of total pins in timing pattern, default 0.4f */
        CV_PROP_RW float maxPenalties;

        /** @brief The maximum allowed relative color mismatch in the timing pattern, default 0.2f*/
        CV_PROP_RW float maxColorsMismatch;

        /** @brief The algorithm find QR codes with almost minimum timing pattern score and minimum size, default 0.9f
         *
         * The QR code with the minimum "timing pattern score" and minimum "size" is selected as the best QR code.
         * If for the current QR code "timing pattern score" * scaleTimingPatternScore < "previous timing pattern score" and "size" < "previous size", then
         * current QR code set as the best QR code. */
        CV_PROP_RW float scaleTimingPatternScore;
    };

    /** @brief QR code detector constructor for Aruco-based algorithm. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP explicit QRCodeDetectorAruco(const QRCodeDetectorAruco::Params& params);

    /** @brief Detector parameters getter. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP const QRCodeDetectorAruco::Params& getDetectorParameters() const;

    /** @brief Detector parameters setter. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP QRCodeDetectorAruco& setDetectorParameters(const QRCodeDetectorAruco::Params& params);

    /** @brief Aruco detector parameters are used to search for the finder patterns. */
    CV_WRAP const aruco::DetectorParameters& getArucoParameters() const;

    /** @brief Aruco detector parameters are used to search for the finder patterns. */
    CV_WRAP void setArucoParameters(const aruco::DetectorParameters& params);
};

class CV_EXPORTS_W_SIMPLE QRCodeDetectorWeChat : public GraphicalCodeDetector
{
public:
    /** @brief Initialize the QRCodeDetectorWeChat.
     *
     * Parameters allow to load _optional_ Detection and Super Resolution DNN model for better quality.
     * @param detection_model_path model file path for the detection model
     * @param super_resolution_model_path model file path for the super resolution model
     * @param graphical_detector detector to be optimized
     * @param detector_iou_thres nms iou threshold for detection part
     * @param score_thres score threshold for detection part
     * @param reference_size the length of the image to align during pre-processing before detection
     */
    CV_WRAP QRCodeDetectorWeChat(const std::string& detection_model_path = "",
                            const std::string& super_resolution_model_path = "",
                            Ptr<GraphicalCodeDetector> graphical_detector = cv::makePtr<QRCodeDetectorAruco>(),
                            const float detector_iou_thres = 0.6,
                            const float score_thres = 0.3,
                            const int reference_size = 720);
    /** @brief Set the reference size during the pre-processing before detection.
    * @param reference_size the length of the image to align during pre-processing before detection
    */
    CV_WRAP void setDetectorReferenceSize(int reference_size);

    /** @brief Set the score threshold during the processing of detection results.
    * @param score_thres score threshold for detection part
    */
    CV_WRAP void setDetectorScoreThres(float score_thres);

    /** @brief Set the detector iou threshold during the processing of detection results.
    * @param iou_thres nms iou threshold for detection part
    */
    CV_WRAP void setDetectorIouThres(float iou_thres);
};
//! @}
}

#include "opencv2/objdetect/detection_based_tracker.hpp"
#include "opencv2/objdetect/face.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"
#include "opencv2/objdetect/barcode.hpp"

#endif
