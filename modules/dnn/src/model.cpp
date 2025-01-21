// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "math_utils.hpp"
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <iterator>

#include <opencv2/imgproc.hpp>

namespace cv {
namespace dnn {

struct Model::Impl
{
//protected:
    Net    net;

    Size   size;
    Scalar mean;
    Scalar scale = Scalar::all(1.0);
    bool   swapRB = false;
    bool   crop = false;
    Mat    blob;
    std::vector<String> outNames;

public:
    virtual ~Impl() {}
    Impl() {}
    Impl(const Impl&) = delete;
    Impl(Impl&&) = delete;

    virtual Net& getNetwork() const { return const_cast<Net&>(net); }

    virtual void setPreferableBackend(Backend backendId) { net.setPreferableBackend(backendId); }
    virtual void setPreferableTarget(Target targetId) { net.setPreferableTarget(targetId); }
    virtual void enableWinograd(bool useWinograd) { net.enableWinograd(useWinograd); }

    virtual
    void initNet(const Net& network)
    {
        CV_TRACE_FUNCTION();
        net = network;

        outNames = net.getUnconnectedOutLayersNames();
        std::vector<MatShape> inLayerShapes;
        std::vector<MatShape> outLayerShapes;
        net.getLayerShapes(MatShape(), 0, inLayerShapes, outLayerShapes);
        if (!inLayerShapes.empty() && inLayerShapes[0].size() == 4)
            size = Size(inLayerShapes[0][3], inLayerShapes[0][2]);
        else
            size = Size();
    }

    /*virtual*/
    void setInputParams(double scale_, const Size& size_, const Scalar& mean_,
                        bool swapRB_, bool crop_)
    {
        size = size_;
        mean = mean_;
        scale = Scalar::all(scale_);
        crop = crop_;
        swapRB = swapRB_;
    }
    /*virtual*/
    void setInputSize(const Size& size_)
    {
        size = size_;
    }
    /*virtual*/
    void setInputMean(const Scalar& mean_)
    {
        mean = mean_;
    }
    /*virtual*/
    void setInputScale(const Scalar& scale_)
    {
        scale = scale_;
    }
    /*virtual*/
    void setInputCrop(bool crop_)
    {
        crop = crop_;
    }
    /*virtual*/
    void setInputSwapRB(bool swapRB_)
    {
        swapRB = swapRB_;
    }
    /*virtual*/
    void setOutputNames(const std::vector<String>& outNames_)
    {
        outNames = outNames_;
    }

    /*virtual*/
    void processFrame(InputArray frame, OutputArrayOfArrays outs)
    {
        CV_TRACE_FUNCTION();
        if (size.empty())
            CV_Error(Error::StsBadSize, "Input size not specified");

        Image2BlobParams param;
        param.scalefactor = scale;
        param.size = size;
        param.mean = mean;
        param.swapRB = swapRB;
        if (crop)
        {
            param.paddingmode = DNN_PMODE_CROP_CENTER;
        }
        Mat blob = dnn::blobFromImageWithParams(frame, param); // [1, 10, 10, 4]

        net.setInput(blob);

        // Faster-RCNN or R-FCN
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
        {
            Mat imInfo(Matx13f(size.height, size.width, 1.6f));
            net.setInput(imInfo, "im_info");
        }

        net.forward(outs, outNames);
    }
};

Model::Model()
    : impl(makePtr<Impl>())
{
    // nothing
}

Model::Model(const String& model, const String& config)
    : Model()
{
    impl->initNet(readNet(model, config));
}

Model::Model(const Net& network)
    : Model()
{
    impl->initNet(network);
}

Net& Model::getNetwork_() const
{
    CV_DbgAssert(impl);
    return impl->getNetwork();
}

Model& Model::setPreferableBackend(Backend backendId)
{
    CV_DbgAssert(impl);
    impl->setPreferableBackend(backendId);
    return *this;
}

Model& Model::setPreferableTarget(Target targetId)
{
    CV_DbgAssert(impl);
    impl->setPreferableTarget(targetId);
    return *this;
}

Model& Model::enableWinograd(bool useWinograd)
{
    CV_DbgAssert(impl);
    impl->enableWinograd(useWinograd);
    return *this;
}

Model& Model::setInputSize(const Size& size)
{
    CV_DbgAssert(impl);
    impl->setInputSize(size);
    return *this;
}

Model& Model::setInputMean(const Scalar& mean)
{
    CV_DbgAssert(impl);
    impl->setInputMean(mean);
    return *this;
}

Model& Model::setInputScale(const Scalar& scale_)
{
    CV_DbgAssert(impl);

    Scalar scale = broadcastRealScalar(scale_);
    impl->setInputScale(scale);
    return *this;
}

Model& Model::setInputCrop(bool crop)
{
    CV_DbgAssert(impl);
    impl->setInputCrop(crop);
    return *this;
}

Model& Model::setInputSwapRB(bool swapRB)
{
    CV_DbgAssert(impl);
    impl->setInputSwapRB(swapRB);
    return *this;
}

Model& Model::setOutputNames(const std::vector<String>& outNames)
{
    CV_DbgAssert(impl);
    impl->setOutputNames(outNames);
    return *this;
}

void Model::setInputParams(double scale, const Size& size, const Scalar& mean,
                           bool swapRB, bool crop)
{
    CV_DbgAssert(impl);
    impl->setInputParams(scale, size, mean, swapRB, crop);
}

void Model::predict(InputArray frame, OutputArrayOfArrays outs) const
{
    CV_DbgAssert(impl);
    impl->processFrame(frame, outs);
}


class ClassificationModel_Impl : public Model::Impl
{
public:
    virtual ~ClassificationModel_Impl() {}
    ClassificationModel_Impl() : Impl() {}
    ClassificationModel_Impl(const ClassificationModel_Impl&) = delete;
    ClassificationModel_Impl(ClassificationModel_Impl&&) = delete;

    void setEnableSoftmaxPostProcessing(bool enable)
    {
        applySoftmax = enable;
    }

    bool getEnableSoftmaxPostProcessing() const
    {
        return applySoftmax;
    }

    std::pair<int, float> classify(InputArray frame)
    {
        std::vector<Mat> outs;
        processFrame(frame, outs);
        CV_Assert(outs.size() == 1);

        Mat out = outs[0].reshape(1, 1);

        if(getEnableSoftmaxPostProcessing())
        {
            softmax(out, out);
        }

        double conf;
        Point maxLoc;
        cv::minMaxLoc(out, nullptr, &conf, nullptr, &maxLoc);
        return {maxLoc.x, static_cast<float>(conf)};
    }

protected:
    void softmax(InputArray inblob, OutputArray outblob)
    {
        const Mat input = inblob.getMat();
        outblob.create(inblob.size(), inblob.type());

        Mat exp;
        const float max = *std::max_element(input.begin<float>(), input.end<float>());
        cv::exp((input - max), exp);
        outblob.getMat() = exp / cv::sum(exp)[0];
    }

protected:
    bool applySoftmax = false;
};

ClassificationModel::ClassificationModel()
    : Model()
{
    // nothing
}

ClassificationModel::ClassificationModel(const String& model, const String& config)
    : ClassificationModel(readNet(model, config))
{
    // nothing
}

ClassificationModel::ClassificationModel(const Net& network)
    : Model()
{
    impl = makePtr<ClassificationModel_Impl>();
    impl->initNet(network);
}

ClassificationModel& ClassificationModel::setEnableSoftmaxPostProcessing(bool enable)
{
    CV_Assert(impl != nullptr && impl.dynamicCast<ClassificationModel_Impl>() != nullptr);
    impl.dynamicCast<ClassificationModel_Impl>()->setEnableSoftmaxPostProcessing(enable);
    return *this;
}

bool ClassificationModel::getEnableSoftmaxPostProcessing() const
{
    CV_Assert(impl != nullptr && impl.dynamicCast<ClassificationModel_Impl>() != nullptr);
    return impl.dynamicCast<ClassificationModel_Impl>()->getEnableSoftmaxPostProcessing();
}

std::pair<int, float> ClassificationModel::classify(InputArray frame)
{
    CV_Assert(impl != nullptr && impl.dynamicCast<ClassificationModel_Impl>() != nullptr);
    return impl.dynamicCast<ClassificationModel_Impl>()->classify(frame);
}

void ClassificationModel::classify(InputArray frame, int& classId, float& conf)
{
    std::tie(classId, conf) = classify(frame);
}

KeypointsModel::KeypointsModel(const String& model, const String& config)
    : Model(model, config) {}

KeypointsModel::KeypointsModel(const Net& network) : Model(network) {}

std::vector<Point2f> KeypointsModel::estimate(InputArray frame, float thresh)
{

    int frameHeight = frame.rows();
    int frameWidth = frame.cols();
    std::vector<Mat> outs;

    impl->processFrame(frame, outs);
    CV_Assert(outs.size() == 1);
    Mat output = outs[0];

    const int nPoints = output.size[1];
    std::vector<Point2f> points;

    // If output is a map, extract the keypoints
    if (output.dims == 4)
    {
        int height = output.size[2];
        int width = output.size[3];

        // find the position of the keypoints (ignore the background)
        for (int n=0; n < nPoints - 1; n++)
        {
            // Probability map of corresponding keypoint
            Mat probMap(height, width, CV_32F, output.ptr(0, n));

            Point2f p(-1, -1);
            Point maxLoc;
            double prob;
            minMaxLoc(probMap, NULL, &prob, NULL, &maxLoc);
            if (prob > thresh)
            {
                p = maxLoc;
                p.x *= (float)frameWidth / width;
                p.y *= (float)frameHeight / height;
            }
            points.push_back(p);
        }
    }
    // Otherwise the output is a vector of keypoints and we can just return it
    else
    {
        for (int n=0; n < nPoints; n++)
        {
            Point2f p;
            p.x = *output.ptr<float>(0, n, 0);
            p.y = *output.ptr<float>(0, n, 1);
            points.push_back(p);
        }
    }
    return points;
}

SegmentationModel::SegmentationModel(const String& model, const String& config)
    : Model(model, config) {}

SegmentationModel::SegmentationModel(const Net& network) : Model(network) {}

void SegmentationModel::segment(InputArray frame, OutputArray mask)
{
    std::vector<Mat> outs;
    impl->processFrame(frame, outs);
    // default output is the first one
    if(outs.size() > 1)
        outs.resize(1);
    Mat score = outs[0];

    const int chns = score.size[1];
    const int rows = score.size[2];
    const int cols = score.size[3];

    mask.create(rows, cols, CV_8U);
    Mat classIds = mask.getMat();
    classIds.setTo(0);
    Mat maxVal(rows, cols, CV_32F, score.data);

    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = classIds.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = ch;
                }
            }
        }
    }
}

class DetectionModel_Impl : public Model::Impl
{
public:
    virtual ~DetectionModel_Impl() {}
    DetectionModel_Impl() : Impl() {}
    DetectionModel_Impl(const DetectionModel_Impl&) = delete;
    DetectionModel_Impl(DetectionModel_Impl&&) = delete;

    void disableRegionNMS(Net& net)
    {
        for (String& name : net.getUnconnectedOutLayersNames())
        {
            int layerId = net.getLayerId(name);
            Ptr<RegionLayer> layer = net.getLayer(layerId).dynamicCast<RegionLayer>();
            if (!layer.empty())
            {
                layer->nmsThreshold = 0;
            }
        }
    }

    void setNmsAcrossClasses(bool value) {
        nmsAcrossClasses = value;
    }

    bool getNmsAcrossClasses() {
        return nmsAcrossClasses;
    }

private:
    bool nmsAcrossClasses = false;
};

DetectionModel::DetectionModel(const String& model, const String& config)
    : DetectionModel(readNet(model, config))
{
    // nothing
}

DetectionModel::DetectionModel(const Net& network) : Model()
{
    impl = makePtr<DetectionModel_Impl>();
    impl->initNet(network);
    impl.dynamicCast<DetectionModel_Impl>()->disableRegionNMS(getNetwork_());  // FIXIT Move to DetectionModel::Impl::initNet()
}

DetectionModel::DetectionModel() : Model()
{
    // nothing
}

DetectionModel& DetectionModel::setNmsAcrossClasses(bool value)
{
    CV_Assert(impl != nullptr && impl.dynamicCast<DetectionModel_Impl>() != nullptr); // remove once default constructor is removed

    impl.dynamicCast<DetectionModel_Impl>()->setNmsAcrossClasses(value);
    return *this;
}

bool DetectionModel::getNmsAcrossClasses()
{
    CV_Assert(impl != nullptr && impl.dynamicCast<DetectionModel_Impl>() != nullptr); // remove once default constructor is removed

    return impl.dynamicCast<DetectionModel_Impl>()->getNmsAcrossClasses();
}

void DetectionModel::detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                            CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect>& boxes,
                            float confThreshold, float nmsThreshold)
{
    CV_Assert(impl != nullptr && impl.dynamicCast<DetectionModel_Impl>() != nullptr); // remove once default constructor is removed

    std::vector<Mat> detections;
    impl->processFrame(frame, detections);

    boxes.clear();
    confidences.clear();
    classIds.clear();

    int frameWidth  = frame.cols();
    int frameHeight = frame.rows();
    if (getNetwork_().getLayer(0)->outputNameToIndex("im_info") != -1)
    {
        frameWidth = impl->size.width;
        frameHeight = impl->size.height;
    }

    std::vector<String> layerNames = getNetwork_().getLayerNames();
    int lastLayerId = getNetwork_().getLayerId(layerNames.back());
    Ptr<Layer> lastLayer = getNetwork_().getLayer(lastLayerId);

    if (lastLayer->type == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        for (int i = 0; i < detections.size(); ++i)
        {
            float* data = (float*)detections[i].data;
            for (int j = 0; j < detections[i].total(); j += 7)
            {
                float conf = data[j + 2];
                if (conf < confThreshold)
                    continue;

                int left   = data[j + 3];
                int top    = data[j + 4];
                int right  = data[j + 5];
                int bottom = data[j + 6];
                int width  = right  - left + 1;
                int height = bottom - top + 1;

                if (width <= 2 || height <= 2)
                {
                    left   = data[j + 3] * frameWidth;
                    top    = data[j + 4] * frameHeight;
                    right  = data[j + 5] * frameWidth;
                    bottom = data[j + 6] * frameHeight;
                    width  = right  - left + 1;
                    height = bottom - top + 1;
                }

                left   = std::max(0, std::min(left, frameWidth - 1));
                top    = std::max(0, std::min(top, frameHeight - 1));
                width  = std::max(1, std::min(width, frameWidth - left));
                height = std::max(1, std::min(height, frameHeight - top));
                boxes.emplace_back(left, top, width, height);

                classIds.push_back(static_cast<int>(data[j + 1]));
                confidences.push_back(conf);
            }
        }
    }
    else if (lastLayer->type == "Region")
    {
        std::vector<int> predClassIds;
        std::vector<Rect> predBoxes;
        std::vector<float> predConfidences;
        for (int i = 0; i < detections.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)detections[i].data;
            for (int j = 0; j < detections[i].rows; ++j, data += detections[i].cols)
            {

                Mat scores = detections[i].row(j).colRange(5, detections[i].cols);
                Point classIdPoint;
                double conf;
                minMaxLoc(scores, nullptr, &conf, nullptr, &classIdPoint);

                if (static_cast<float>(conf) < confThreshold)
                    continue;

                int centerX = data[0] * frameWidth;
                int centerY = data[1] * frameHeight;
                int width   = data[2] * frameWidth;
                int height  = data[3] * frameHeight;

                int left = std::max(0, std::min(centerX - width / 2, frameWidth - 1));
                int top  = std::max(0, std::min(centerY - height / 2, frameHeight - 1));
                width    = std::max(1, std::min(width, frameWidth - left));
                height   = std::max(1, std::min(height, frameHeight - top));

                predClassIds.push_back(classIdPoint.x);
                predConfidences.push_back(static_cast<float>(conf));
                predBoxes.emplace_back(left, top, width, height);
            }
        }

        if (nmsThreshold)
        {
            if (getNmsAcrossClasses())
            {
                std::vector<int> indices;
                NMSBoxes(predBoxes, predConfidences, confThreshold, nmsThreshold, indices);
                for (int idx : indices)
                {
                    boxes.push_back(predBoxes[idx]);
                    confidences.push_back(predConfidences[idx]);
                    classIds.push_back(predClassIds[idx]);
                }
            }
            else
            {
                std::map<int, std::vector<size_t> > class2indices;
                for (size_t i = 0; i < predClassIds.size(); i++)
                {
                    if (predConfidences[i] >= confThreshold)
                    {
                        class2indices[predClassIds[i]].push_back(i);
                    }
                }
                for (const auto& it : class2indices)
                {
                    std::vector<Rect> localBoxes;
                    std::vector<float> localConfidences;
                    for (size_t idx : it.second)
                    {
                        localBoxes.push_back(predBoxes[idx]);
                        localConfidences.push_back(predConfidences[idx]);
                    }
                    std::vector<int> indices;
                    NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, indices);
                    classIds.resize(classIds.size() + indices.size(), it.first);
                    for (int idx : indices)
                    {
                        boxes.push_back(localBoxes[idx]);
                        confidences.push_back(localConfidences[idx]);
                    }
                }
            }
        }
        else
        {
            boxes       = std::move(predBoxes);
            classIds    = std::move(predClassIds);
            confidences = std::move(predConfidences);
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: \"" + lastLayer->type + "\"");
}

class YOLODetectionModel_Impl : public DetectionModel_Impl
{

    ImagePaddingMode paddingMode;
    float padValue;
    Size2f frameSize;

    public:
    YOLODetectionModel_Impl() {
        //nothing
    }

    void processFrame(InputArray frame, OutputArrayOfArrays outs)
    {
        CV_TRACE_FUNCTION();
        if (size.empty())
            CV_Error(Error::StsBadSize, "Input size not specified");

        frameSize.width = frame.cols();
        frameSize.height = frame.rows();

        Image2BlobParams param;
        param.scalefactor = scale;
        param.size = size;
        param.mean = mean;
        param.swapRB = swapRB;
        param.borderValue = padValue;
        param.paddingmode = paddingMode;

        Mat blob = dnn::blobFromImageWithParams(frame, param);
        net.setInput(blob);

        net.forward(outs, outNames);
    }

    void setPaddingValue(const float padValue_){
        padValue = padValue_;
    }

    void setPaddingMode(const ImagePaddingMode paddingmode_){
        if (paddingmode_ == ImagePaddingMode::DNN_PMODE_NULL ||
            paddingmode_ == ImagePaddingMode::DNN_PMODE_CROP_CENTER ||
            paddingmode_ == ImagePaddingMode::DNN_PMODE_LETTERBOX){
            paddingMode = paddingmode_;
            } else {
                CV_Error(Error::StsNotImplemented, "Unsupported padding mode");
            }
    }

    void setInputParams(double scale_, const Size& size_, const Scalar& mean_,
                        bool swapRB_, ImagePaddingMode paddingMode_, float padValue_)
    {
        size = size_;
        mean = mean_;
        scale = Scalar::all(scale_);
        swapRB = swapRB_;
        paddingMode = paddingMode_;
        padValue = padValue_;
    }

    void detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                            CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect>& boxes,
                            float confThreshold, float nmsThreshold){

        std::vector<Mat> detections;
        processFrame(frame, detections);

        YOLODetectionModel::postProccess(
            detections,
            boxes,
            confidences,
            classIds,
            size,
            confThreshold,
            nmsThreshold,
            getNmsAcrossClasses()
            );

            Image2BlobParams paramNet;
            paramNet.scalefactor = scale;
            paramNet.size = size;
            paramNet.mean = mean;
            paramNet.swapRB = swapRB;
            paramNet.paddingmode = paddingMode;
            paramNet.blobRectsToImageRects(boxes, boxes, frameSize);
    }
};

YOLODetectionModel::YOLODetectionModel(const String& model, const String& config)
{
    impl = makePtr<YOLODetectionModel_Impl>();
    impl->initNet(readNet(model, config));
}

YOLODetectionModel::YOLODetectionModel(const String& onnx)
{
    impl = makePtr<YOLODetectionModel_Impl>();
    impl->initNet(readNetFromONNX(onnx));
}

YOLODetectionModel::YOLODetectionModel()
{
    impl = std::static_pointer_cast<Model::Impl>(makePtr<YOLODetectionModel_Impl>());
}

void YOLODetectionModel::detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                            CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect>& boxes,
                            float confThreshold, float nmsThreshold) {
    impl.dynamicCast<YOLODetectionModel_Impl>()->detect(frame, classIds,
                            confidences, boxes, confThreshold, nmsThreshold);
}

void YOLODetectionModel::postProccess(
    std::vector<Mat>& detections,
    CV_OUT std::vector<Rect>& keep_boxes,
    CV_OUT std::vector<float>& keep_confidences,
    CV_OUT std::vector<int>& keep_classIds,
    Size inputImgSize,
    const float confThreshold,
    const float nmsThreshold,
    const bool nmsAcrossClasses
    ){

    bool yolov8 = false;
    bool darknet = false;

    if (detections[0].dims == 2){
        darknet = true;
    }

    if (!darknet && detections[0].size[1] < detections[0].size[2]) {
        yolov8 = true;  // Set the correct flag based on tensor shape
    }

    // Retrieve
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;

    if (yolov8){
        for(auto & detection : detections){
            cv::transposeND(detection, {0, 2, 1}, detection);
        }
    }

    // each row is [cx, cy, w, h, conf_obj, conf_class1, ..., conf_class80]
    for (auto preds : detections)
    {
        if (!darknet)
            preds = preds.reshape(1, preds.size[1]);

        for (int i = 0; i < preds.rows; ++i)
        {
            // filter out non objects
            float obj_conf = (!yolov8) ? preds.at<float>(i, 4) : 1.0f;
            if (obj_conf < confThreshold)
                continue;

            Mat scores = preds.row(i).colRange((!yolov8) ? 5 : 4, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);

            conf = (!yolov8) ? conf * obj_conf : conf;
            if (conf < confThreshold)
                continue;

            // get bbox coords
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];

            // [x1, y1, x2, y2]
            double x1 = cx - 0.5 * w;
            double y1 = cy - 0.5 * h;
            double x2 = cx + 0.5 * w;
            double y2 = cy + 0.5 * h;


            int width  = x2 - x1 + 1;
            int height = y2 - y1 + 1;

            if (width <= 2 || height <= 2)
            {
                x1  = x1 * inputImgSize.width;
                y1  = y1 * inputImgSize.height;
                x2  = x2 * inputImgSize.width;
                y2  = y2 * inputImgSize.height;
                width  = x2 - x1 + 1;
                height = y2 - y1 + 1;
            }

            boxes.emplace_back(Rect(x1, y1, x2, y2));
            classIds.emplace_back(maxLoc.x);
            confidences.emplace_back(conf);
        }
    }

    // NMS
    if (nmsAcrossClasses)
    {
        std::vector<int> keep_idx;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, keep_idx);
        for (auto i : keep_idx)
        {
            keep_classIds.emplace_back(classIds[i]);
            keep_confidences.emplace_back(confidences[i]);
            keep_boxes.emplace_back(boxes[i]);
        }
    }
    else
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        for (const auto& it : class2indices)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            for (size_t idx : it.second)
            {
                localBoxes.push_back(boxes[idx]);
                localConfidences.push_back(confidences[idx]);
            }
            std::vector<int> indices;
            NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, indices);

            keep_classIds.resize(keep_classIds.size() + indices.size(), it.first);
            for (int idx : indices)
            {
                keep_boxes.push_back(localBoxes[idx]);
                keep_confidences.push_back(localConfidences[idx]);
            }
        }
    }
    // convert boxes to xywh
    for(auto & box : keep_boxes){
        box.width = box.width - box.x;
        box.height = box.height - box.y;
    }
}

YOLODetectionModel& YOLODetectionModel::setPaddingMode(const ImagePaddingMode paddingMode){
    impl.dynamicCast<YOLODetectionModel_Impl>()->setPaddingMode(paddingMode);
    return *this;
}

YOLODetectionModel& YOLODetectionModel::setPaddingValue(const float PadingValue){
    impl.dynamicCast<YOLODetectionModel_Impl>()->setPaddingValue(PadingValue);
    return *this;
}

struct TextRecognitionModel_Impl : public Model::Impl
{
    std::string decodeType;
    std::vector<std::string> vocabulary;

    int beamSize = 10;
    int vocPruneSize = 0;

    TextRecognitionModel_Impl()
    {
        CV_TRACE_FUNCTION();
    }

    TextRecognitionModel_Impl(const Net& network)
    {
        CV_TRACE_FUNCTION();
        initNet(network);
    }

    inline
    void setVocabulary(const std::vector<std::string>& inputVoc)
    {
        vocabulary = inputVoc;
    }

    inline
    void setDecodeType(const std::string& type)
    {
        decodeType = type;
    }

    inline
    void setDecodeOptsCTCPrefixBeamSearch(int beam, int vocPrune)
    {
        beamSize = beam;
        vocPruneSize = vocPrune;
    }

    virtual
    std::string decode(const Mat& prediction)
    {
        CV_TRACE_FUNCTION();
        CV_Assert(!prediction.empty());
        if (decodeType.empty())
            CV_Error(Error::StsBadArg, "TextRecognitionModel: decodeType is not specified");
        if (vocabulary.empty())
            CV_Error(Error::StsBadArg, "TextRecognitionModel: vocabulary is not specified");

        std::string decodeSeq;
        if (decodeType == "CTC-greedy") {
            decodeSeq = ctcGreedyDecode(prediction);
        } else if (decodeType == "CTC-prefix-beam-search") {
            decodeSeq = ctcPrefixBeamSearchDecode(prediction);
        } else if (decodeType.length() == 0) {
            CV_Error(Error::StsBadArg, "Please set decodeType");
        } else {
            CV_Error_(Error::StsBadArg, ("Unsupported decodeType: %s", decodeType.c_str()));
        }

        return decodeSeq;
    }

    virtual
    std::string ctcGreedyDecode(const Mat& prediction)
    {
        std::string decodeSeq;
        CV_CheckEQ(prediction.dims, 3, "");
        CV_CheckType(prediction.type(), CV_32FC1, "");
        const int vocLength = (int)(vocabulary.size());
        CV_CheckLE(prediction.size[1], vocLength, "");
        bool ctcFlag = true;
        int lastLoc = 0;
        for (int i = 0; i < prediction.size[0]; i++)
        {
            const float* pred = prediction.ptr<float>(i);
            int maxLoc = 0;
            float maxScore = pred[0];
            for (int j = 1; j < vocLength + 1; j++)
            {
                float score = pred[j];
                if (maxScore < score)
                {
                    maxScore = score;
                    maxLoc = j;
                }
            }

            if (maxLoc > 0)
            {
                std::string currentChar = vocabulary.at(maxLoc - 1);
                if (maxLoc != lastLoc || ctcFlag)
                {
                    lastLoc = maxLoc;
                    decodeSeq += currentChar;
                    ctcFlag = false;
                }
            }
            else
            {
                ctcFlag = true;
            }
        }
        return decodeSeq;
    }

    struct PrefixScore
    {
        // blank ending score
        float pB;
        // none blank ending score
        float pNB;

        PrefixScore() : pB(kNegativeInfinity), pNB(kNegativeInfinity)
        {

        }
        PrefixScore(float pB, float pNB) : pB(pB), pNB(pNB)
        {

        }
    };

    struct PrefixHash
    {
        size_t operator()(const std::vector<int>& prefix) const
        {
              // BKDR hash
              unsigned int seed = 131;
              size_t hash = 0;
              for (size_t i = 0; i < prefix.size(); i++)
              {
                  hash = hash * seed + prefix[i];
              }
              return hash;
        }
    };

    static
    std::vector<std::pair<float, int>> TopK(
                      const float* predictions, int length, int k)
    {
        std::vector<std::pair<float, int>> results;
        // No prune.
        if (k <= 0)
        {
            for (int i = 0; i < length; ++i)
            {
                results.emplace_back(predictions[i], i);
            }
            return results;
        }

        for (int i = 0; i < k; ++i)
        {
            results.emplace_back(predictions[i], i);
        }
        std::make_heap(results.begin(), results.end(), std::greater<std::pair<float, int>>{});

        for (int i = k; i < length; ++i)
        {
            if (predictions[i] > results.front().first)
            {
                std::pop_heap(results.begin(), results.end(), std::greater<std::pair<float, int>>{});
                results.pop_back();
                results.emplace_back(predictions[i], i);
                std::push_heap(results.begin(), results.end(), std::greater<std::pair<float, int>>{});
            }
        }
        return results;
    }

    static inline
    bool PrefixScoreCompare(
            const std::pair<std::vector<int>, PrefixScore>& a,
            const std::pair<std::vector<int>, PrefixScore>& b)
    {
            float probA = LogAdd(a.second.pB, a.second.pNB);
            float probB = LogAdd(b.second.pB, b.second.pNB);
            return probA > probB;
    }

    virtual
    std::string ctcPrefixBeamSearchDecode(const Mat& prediction) {
          // CTC prefix beam search decode.
          // For more detail, refer to:
          // https://distill.pub/2017/ctc/#inference
          // https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0i
          using Beam = std::vector<std::pair<std::vector<int>, PrefixScore>>;
          using BeamInDict = std::unordered_map<std::vector<int>, PrefixScore, PrefixHash>;

          CV_CheckType(prediction.type(), CV_32FC1, "");
          CV_CheckEQ(prediction.dims, 3, "");
          CV_CheckEQ(prediction.size[1], 1, "");
          CV_CheckEQ(prediction.size[2], (int)vocabulary.size() + 1, "");  // Length add 1 for ctc blank

          std::string decodeSeq;
          Beam beam = {std::make_pair(std::vector<int>(), PrefixScore(0.0, kNegativeInfinity))};
          for (int i = 0; i < prediction.size[0]; i++)
          {
              // Loop over time
              BeamInDict nextBeam;
              const float* pred = prediction.ptr<float>(i);
              std::vector<std::pair<float, int>> topkPreds =
                  TopK(pred, vocabulary.size() + 1, vocPruneSize);
              for (const auto& each : topkPreds)
              {
                  // Loop over vocabulary
                  float prob = each.first;
                  int token = each.second;
                  for (const auto& it : beam)
                  {
                      const std::vector<int>& prefix = it.first;
                      const PrefixScore& prefixScore = it.second;
                      if (token == 0)  // 0 stands for ctc blank
                      {
                          PrefixScore& nextScore = nextBeam[prefix];
                          nextScore.pB = LogAdd(nextScore.pB,
                              LogAdd(prefixScore.pB + prob, prefixScore.pNB + prob));
                          continue;
                      }

                      std::vector<int> nPrefix(prefix);
                      nPrefix.push_back(token);
                      PrefixScore& nextScore = nextBeam[nPrefix];
                      if (prefix.size() > 0 && token == prefix.back())
                      {
                          nextScore.pNB = LogAdd(nextScore.pNB, prefixScore.pB + prob);
                          PrefixScore& mScore = nextBeam[prefix];
                          mScore.pNB = LogAdd(mScore.pNB, prefixScore.pNB + prob);
                      }
                      else
                      {
                          nextScore.pNB = LogAdd(nextScore.pNB,
                              LogAdd(prefixScore.pB + prob, prefixScore.pNB + prob));
                      }
                  }
              }
              // Beam prune
              Beam newBeam(nextBeam.begin(), nextBeam.end());
              int newBeamSize = std::min(static_cast<int>(newBeam.size()), beamSize);
              std::nth_element(newBeam.begin(), newBeam.begin() + newBeamSize,
                     newBeam.end(), PrefixScoreCompare);
              newBeam.resize(newBeamSize);
              std::sort(newBeam.begin(), newBeam.end(), PrefixScoreCompare);
              beam = std::move(newBeam);
          }

          CV_Assert(!beam.empty());
          for (int token : beam[0].first)
          {
              CV_Check(token, token > 0 && token <= vocabulary.size(), "");
              decodeSeq += vocabulary.at(token - 1);
          }
          return decodeSeq;
    }

    virtual
    std::string recognize(InputArray frame)
    {
        CV_TRACE_FUNCTION();
        std::vector<Mat> outs;
        processFrame(frame, outs);
        CV_CheckEQ(outs.size(), (size_t)1, "");
        return decode(outs[0]);
    }

    virtual
    void recognize(InputArray frame, InputArrayOfArrays roiRects, CV_OUT std::vector<std::string>& results)
    {
        CV_TRACE_FUNCTION();
        results.clear();
        if (roiRects.empty())
        {
            auto s = recognize(frame);
            results.push_back(s);
            return;
        }

        std::vector<Rect> rects;
        roiRects.copyTo(rects);

        // Predict for each RoI
        Mat input = frame.getMat();
        for (size_t i = 0; i < rects.size(); i++)
        {
            Rect roiRect = rects[i];
            Mat roi = input(roiRect);
            auto s = recognize(roi);
            results.push_back(s);
        }
    }

    static inline
    TextRecognitionModel_Impl& from(const std::shared_ptr<Model::Impl>& ptr)
    {
        CV_Assert(ptr);
        return *((TextRecognitionModel_Impl*)ptr.get());
    }
};

TextRecognitionModel::TextRecognitionModel()
{
    impl = std::static_pointer_cast<Model::Impl>(makePtr<TextRecognitionModel_Impl>());
}

TextRecognitionModel::TextRecognitionModel(const Net& network)
{
    impl = std::static_pointer_cast<Model::Impl>(std::make_shared<TextRecognitionModel_Impl>(network));
}

TextRecognitionModel& TextRecognitionModel::setDecodeType(const std::string& decodeType)
{
    TextRecognitionModel_Impl::from(impl).setDecodeType(decodeType);
    return *this;
}

const std::string& TextRecognitionModel::getDecodeType() const
{
    return TextRecognitionModel_Impl::from(impl).decodeType;
}

TextRecognitionModel& TextRecognitionModel::setDecodeOptsCTCPrefixBeamSearch(int beamSize, int vocPruneSize)
{
    TextRecognitionModel_Impl::from(impl).setDecodeOptsCTCPrefixBeamSearch(beamSize, vocPruneSize);
    return *this;
}

TextRecognitionModel& TextRecognitionModel::setVocabulary(const std::vector<std::string>& inputVoc)
{
    TextRecognitionModel_Impl::from(impl).setVocabulary(inputVoc);
    return *this;
}

const std::vector<std::string>& TextRecognitionModel::getVocabulary() const
{
    return TextRecognitionModel_Impl::from(impl).vocabulary;
}

std::string TextRecognitionModel::recognize(InputArray frame) const
{
    return TextRecognitionModel_Impl::from(impl).recognize(frame);
}

void TextRecognitionModel::recognize(InputArray frame, InputArrayOfArrays roiRects, CV_OUT std::vector<std::string>& results) const
{
    TextRecognitionModel_Impl::from(impl).recognize(frame, roiRects, results);
}


///////////////////////////////////////// Text Detection /////////////////////////////////////////

struct TextDetectionModel_Impl : public Model::Impl
{
    TextDetectionModel_Impl() {}

    TextDetectionModel_Impl(const Net& network)
    {
        CV_TRACE_FUNCTION();
        initNet(network);
    }

    virtual
    std::vector< std::vector<Point2f> > detect(InputArray frame, CV_OUT std::vector<float>& confidences)
    {
        CV_TRACE_FUNCTION();
        std::vector<RotatedRect> rects = detectTextRectangles(frame, confidences);
        std::vector< std::vector<Point2f> > results;
        for (const RotatedRect& rect : rects)
        {
            Point2f vertices[4] = {};
            rect.points(vertices);
            std::vector<Point2f> result = { vertices[0], vertices[1], vertices[2], vertices[3] };
            results.emplace_back(result);
        }
        return results;
    }

    virtual
    std::vector< std::vector<Point2f> > detect(InputArray frame)
    {
        CV_TRACE_FUNCTION();
        std::vector<float> confidences;
        return detect(frame, confidences);
    }

    virtual
    std::vector<RotatedRect> detectTextRectangles(InputArray frame, CV_OUT std::vector<float>& confidences)
    {
        CV_Error(Error::StsNotImplemented, "");
    }

    virtual
    std::vector<cv::RotatedRect> detectTextRectangles(InputArray frame)
    {
        CV_TRACE_FUNCTION();
        std::vector<float> confidences;
        return detectTextRectangles(frame, confidences);
    }

    static inline
    TextDetectionModel_Impl& from(const std::shared_ptr<Model::Impl>& ptr)
    {
        CV_Assert(ptr);
        return *((TextDetectionModel_Impl*)ptr.get());
    }
};


TextDetectionModel::TextDetectionModel()
    : Model()
{
    // nothing
}

static
void to32s(
        const std::vector< std::vector<Point2f> >& detections_f,
        CV_OUT std::vector< std::vector<Point> >& detections
)
{
    detections.resize(detections_f.size());
    for (size_t i = 0; i < detections_f.size(); i++)
    {
        const auto& contour_f = detections_f[i];
        std::vector<Point> contour(contour_f.size());
        for (size_t j = 0; j < contour_f.size(); j++)
        {
            contour[j].x = cvRound(contour_f[j].x);
            contour[j].y = cvRound(contour_f[j].y);
        }
        swap(detections[i], contour);
    }
}

void TextDetectionModel::detect(
        InputArray frame,
        CV_OUT std::vector< std::vector<Point> >& detections,
        CV_OUT std::vector<float>& confidences
) const
{
    std::vector< std::vector<Point2f> > detections_f = TextDetectionModel_Impl::from(impl).detect(frame, confidences);
    to32s(detections_f, detections);
    return;
}

void TextDetectionModel::detect(
        InputArray frame,
        CV_OUT std::vector< std::vector<Point> >& detections
) const
{
    std::vector< std::vector<Point2f> > detections_f = TextDetectionModel_Impl::from(impl).detect(frame);
    to32s(detections_f, detections);
    return;
}

void TextDetectionModel::detectTextRectangles(
        InputArray frame,
        CV_OUT std::vector<cv::RotatedRect>& detections,
        CV_OUT std::vector<float>& confidences
) const
{
    detections = TextDetectionModel_Impl::from(impl).detectTextRectangles(frame, confidences);
    return;
}

void TextDetectionModel::detectTextRectangles(
        InputArray frame,
        CV_OUT std::vector<cv::RotatedRect>& detections
) const
{
    detections = TextDetectionModel_Impl::from(impl).detectTextRectangles(frame);
    return;
}


struct TextDetectionModel_EAST_Impl : public TextDetectionModel_Impl
{
    float confThreshold;
    float nmsThreshold;

    TextDetectionModel_EAST_Impl()
        : confThreshold(0.5f)
        , nmsThreshold(0.0f)
    {
        CV_TRACE_FUNCTION();
    }

    TextDetectionModel_EAST_Impl(const Net& network)
        : TextDetectionModel_EAST_Impl()
    {
        CV_TRACE_FUNCTION();
        initNet(network);
    }

    void setConfidenceThreshold(float confThreshold_) { confThreshold = confThreshold_; }
    float getConfidenceThreshold() const { return confThreshold; }

    void setNMSThreshold(float nmsThreshold_) { nmsThreshold = nmsThreshold_; }
    float getNMSThreshold() const { return nmsThreshold; }

    // TODO: According to article EAST supports quadrangles output: https://arxiv.org/pdf/1704.03155.pdf
#if 0
    virtual
    std::vector< std::vector<Point2f> > detect(InputArray frame, CV_OUT std::vector<float>& confidences) CV_OVERRIDE
#endif

    virtual
    std::vector<cv::RotatedRect> detectTextRectangles(InputArray frame, CV_OUT std::vector<float>& confidences) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        std::vector<cv::RotatedRect> results;

        std::vector<Mat> outs;
        processFrame(frame, outs);
        CV_CheckEQ(outs.size(), (size_t)2, "");
        Mat geometry = outs[0];
        Mat scoreMap = outs[1];

        CV_CheckEQ(scoreMap.dims, 4, "");
        CV_CheckEQ(geometry.dims, 4, "");
        CV_CheckEQ(scoreMap.size[0], 1, "");
        CV_CheckEQ(geometry.size[0], 1, "");
        CV_CheckEQ(scoreMap.size[1], 1, "");
        CV_CheckEQ(geometry.size[1], 5, "");
        CV_CheckEQ(scoreMap.size[2], geometry.size[2], "");
        CV_CheckEQ(scoreMap.size[3], geometry.size[3], "");

        CV_CheckType(scoreMap.type(), CV_32FC1, "");
        CV_CheckType(geometry.type(), CV_32FC1, "");

        std::vector<RotatedRect> boxes;
        std::vector<float> scores;
        const int height = scoreMap.size[2];
        const int width = scoreMap.size[3];
        for (int y = 0; y < height; ++y)
        {
            const float* scoresData = scoreMap.ptr<float>(0, 0, y);
            const float* x0_data = geometry.ptr<float>(0, 0, y);
            const float* x1_data = geometry.ptr<float>(0, 1, y);
            const float* x2_data = geometry.ptr<float>(0, 2, y);
            const float* x3_data = geometry.ptr<float>(0, 3, y);
            const float* anglesData = geometry.ptr<float>(0, 4, y);
            for (int x = 0; x < width; ++x)
            {
                float score = scoresData[x];
                if (score < confThreshold)
                    continue;

                float offsetX = x * 4.0f, offsetY = y * 4.0f;
                float angle = anglesData[x];
                float cosA = std::cos(angle);
                float sinA = std::sin(angle);
                float h = x0_data[x] + x2_data[x];
                float w = x1_data[x] + x3_data[x];

                Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                               offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
                Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
                Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
                boxes.push_back(RotatedRect(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI));
                scores.push_back(score);
            }
        }

        // Apply non-maximum suppression procedure.
        std::vector<int> indices;
        NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);

        confidences.clear();
        confidences.reserve(indices.size());

        // Re-scale
        Point2f ratio((float)frame.cols() / size.width, (float)frame.rows() / size.height);
        bool isUniformRatio = std::fabs(ratio.x - ratio.y) <= 0.01f;
        for (uint i = 0; i < indices.size(); i++)
        {
            auto idx = indices[i];

            auto conf = scores[idx];
            confidences.push_back(conf);

            RotatedRect& box0 = boxes[idx];

            if (isUniformRatio)
            {
                RotatedRect box = box0;
                box.center.x *= ratio.x;
                box.center.y *= ratio.y;
                box.size.width *= ratio.x;
                box.size.height *= ratio.y;
                results.emplace_back(box);
            }
            else
            {
                Point2f vertices[4] = {};
                box0.points(vertices);
                for (int j = 0; j < 4; j++)
                {
                    vertices[j].x *= ratio.x;
                    vertices[j].y *= ratio.y;
                }
                RotatedRect box = minAreaRect(Mat(4, 1, CV_32FC2, (void*)vertices));

                // minArea() rect is not normalized, it may return rectangles rotated by +90/-90
                float angle_diff = std::fabs(box.angle - box0.angle);
                while (angle_diff >= (90 + 45))
                {
                    box.angle += (box.angle < box0.angle) ? 180 : -180;
                    angle_diff = std::fabs(box.angle - box0.angle);
                }
                if (angle_diff > 45)  // avoid ~90 degree turns
                {
                    std::swap(box.size.width, box.size.height);
                    if (box.angle < box0.angle)
                        box.angle += 90;
                    else if (box.angle > box0.angle)
                        box.angle -= 90;
                }
                // CV_DbgAssert(std::fabs(box.angle - box0.angle) <= 45);

                results.emplace_back(box);
            }
        }

        return results;
    }

    static inline
    TextDetectionModel_EAST_Impl& from(const std::shared_ptr<Model::Impl>& ptr)
    {
        CV_Assert(ptr);
        return *((TextDetectionModel_EAST_Impl*)ptr.get());
    }
};


TextDetectionModel_EAST::TextDetectionModel_EAST()
    : TextDetectionModel()
{
    impl = std::static_pointer_cast<Model::Impl>(makePtr<TextDetectionModel_EAST_Impl>());
}

TextDetectionModel_EAST::TextDetectionModel_EAST(const Net& network)
    : TextDetectionModel()
{
    impl = std::static_pointer_cast<Model::Impl>(makePtr<TextDetectionModel_EAST_Impl>(network));
}

TextDetectionModel_EAST& TextDetectionModel_EAST::setConfidenceThreshold(float confThreshold)
{
    TextDetectionModel_EAST_Impl::from(impl).setConfidenceThreshold(confThreshold);
    return *this;
}
float TextDetectionModel_EAST::getConfidenceThreshold() const
{
    return TextDetectionModel_EAST_Impl::from(impl).getConfidenceThreshold();
}

TextDetectionModel_EAST& TextDetectionModel_EAST::setNMSThreshold(float nmsThreshold)
{
    TextDetectionModel_EAST_Impl::from(impl).setNMSThreshold(nmsThreshold);
    return *this;
}
float TextDetectionModel_EAST::getNMSThreshold() const
{
    return TextDetectionModel_EAST_Impl::from(impl).getNMSThreshold();
}



struct TextDetectionModel_DB_Impl : public TextDetectionModel_Impl
{
    float binaryThreshold;
    float polygonThreshold;
    double unclipRatio;
    int maxCandidates;

    TextDetectionModel_DB_Impl()
        : binaryThreshold(0.3f)
        , polygonThreshold(0.5f)
        , unclipRatio(2.0f)
        , maxCandidates(0)
    {
        CV_TRACE_FUNCTION();
    }

    TextDetectionModel_DB_Impl(const Net& network)
        : TextDetectionModel_DB_Impl()
    {
        CV_TRACE_FUNCTION();
        initNet(network);
    }

    void setBinaryThreshold(float binaryThreshold_) { binaryThreshold = binaryThreshold_; }
    float getBinaryThreshold() const { return binaryThreshold; }

    void setPolygonThreshold(float polygonThreshold_) { polygonThreshold = polygonThreshold_; }
    float getPolygonThreshold() const { return polygonThreshold; }

    void setUnclipRatio(double unclipRatio_) { unclipRatio = unclipRatio_; }
    double getUnclipRatio() const { return unclipRatio; }

    void setMaxCandidates(int maxCandidates_) { maxCandidates = maxCandidates_; }
    int getMaxCandidates() const { return maxCandidates; }


    virtual
    std::vector<cv::RotatedRect> detectTextRectangles(InputArray frame, CV_OUT std::vector<float>& confidences) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        std::vector< std::vector<Point2f> > contours = detect(frame, confidences);
        std::vector<cv::RotatedRect> results; results.reserve(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            auto& contour = contours[i];
            RotatedRect box = minAreaRect(contour);

            // minArea() rect is not normalized, it may return rectangles with angle=-90 or height < width
            const float angle_threshold = 60;  // do not expect vertical text, TODO detection algo property
            bool swap_size = false;
            if (box.size.width < box.size.height)  // horizontal-wide text area is expected
                swap_size = true;
            else if (std::fabs(box.angle) >= angle_threshold)  // don't work with vertical rectangles
                swap_size = true;
            if (swap_size)
            {
                std::swap(box.size.width, box.size.height);
                if (box.angle < 0)
                    box.angle += 90;
                else if (box.angle > 0)
                    box.angle -= 90;
            }

            results.push_back(box);
        }
        return results;
    }

    std::vector< std::vector<Point2f> > detect(InputArray frame, CV_OUT std::vector<float>& confidences) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        std::vector< std::vector<Point2f> > results;
        confidences.clear();
        std::vector<Mat> outs;
        processFrame(frame, outs);
        CV_Assert(outs.size() == 1);
        Mat binary = outs[0];

        // Threshold
        Mat bitmap;
        threshold(binary, bitmap, binaryThreshold, 255, THRESH_BINARY);

        // Scale ratio
        float scaleHeight = (float)(frame.rows()) / (float)(binary.size[0]);
        float scaleWidth = (float)(frame.cols()) / (float)(binary.size[1]);

        // Find contours
        std::vector< std::vector<Point> > contours;
        bitmap.convertTo(bitmap, CV_8UC1);
        findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        // Candidate number limitation
        size_t numCandidate = std::min(contours.size(), (size_t)(maxCandidates > 0 ? maxCandidates : INT_MAX));

        for (size_t i = 0; i < numCandidate; i++)
        {
            std::vector<Point>& contour = contours[i];

            // Calculate text contour score
            float score = contourScore(binary, contour);
            if (score < polygonThreshold)
                continue;

            // Rescale
            std::vector<Point> contourScaled; contourScaled.reserve(contour.size());
            for (size_t j = 0; j < contour.size(); j++)
            {
                contourScaled.push_back(Point(int(contour[j].x * scaleWidth),
                                              int(contour[j].y * scaleHeight)));
            }

            // Unclip
            RotatedRect box = minAreaRect(contourScaled);
            float minLen = std::min(box.size.height/scaleWidth, box.size.width/scaleHeight);

            // Filter very small boxes
            if (minLen < 3)
                continue;

            // minArea() rect is not normalized, it may return rectangles with angle=-90 or height < width
            const float angle_threshold = 60;  // do not expect vertical text, TODO detection algo property
            bool swap_size = false;
            if (box.size.width < box.size.height)  // horizontal-wide text area is expected
                swap_size = true;
            else if (std::fabs(box.angle) >= angle_threshold)  // don't work with vertical rectangles
                swap_size = true;
            if (swap_size)
            {
                std::swap(box.size.width, box.size.height);
                if (box.angle < 0)
                    box.angle += 90;
                else if (box.angle > 0)
                    box.angle -= 90;
            }

            Point2f vertex[4];
            box.points(vertex);  // order: bl, tl, tr, br
            std::vector<Point2f> approx;
            for (int j = 0; j < 4; j++)
                approx.emplace_back(vertex[j]);
            std::vector<Point2f> polygon;
            unclip(approx, polygon, unclipRatio);
            if (polygon.empty())
                continue;
            results.push_back(polygon);
            confidences.push_back(score);
        }

        return results;
    }

    // According to https://github.com/MhLiao/DB/blob/master/structure/representers/seg_detector_representer.py (2020-10)
    static double contourScore(const Mat& binary, const std::vector<Point>& contour)
    {
        Rect rect = boundingRect(contour);
        int xmin = std::max(rect.x, 0);
        int xmax = std::min(rect.x + rect.width, binary.cols - 1);
        int ymin = std::max(rect.y, 0);
        int ymax = std::min(rect.y + rect.height, binary.rows - 1);

        Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

        Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
        std::vector<Point> roiContour;
        for (size_t i = 0; i < contour.size(); i++) {
            Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
            roiContour.push_back(pt);
        }
        std::vector<std::vector<Point>> roiContours = {roiContour};
        fillPoly(mask, roiContours, Scalar(1));
        double score = cv::mean(binROI, mask).val[0];

        return score;
    }

    // According to https://github.com/MhLiao/DB/blob/master/structure/representers/seg_detector_representer.py (2020-10)
    static void unclip(const std::vector<Point2f>& inPoly, std::vector<Point2f> &outPoly, const double unclipRatio)
    {
        double area = contourArea(inPoly);
        double length = arcLength(inPoly, true);

        if(length == 0.)
            return;

        double distance = area * unclipRatio / length;

        size_t numPoints = inPoly.size();
        std::vector<std::vector<Point2f>> newLines;
        for (size_t i = 0; i < numPoints; i++) {
            std::vector<Point2f> newLine;
            Point pt1 = inPoly[i];
            Point pt2 = inPoly[(i - 1) % numPoints];
            Point vec = pt1 - pt2;
            float unclipDis = (float)(distance / norm(vec));
            Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
            newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
            newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
            newLines.push_back(newLine);
        }

        size_t numLines = newLines.size();
        for (size_t i = 0; i < numLines; i++) {
            Point2f a = newLines[i][0];
            Point2f b = newLines[i][1];
            Point2f c = newLines[(i + 1) % numLines][0];
            Point2f d = newLines[(i + 1) % numLines][1];
            Point2f pt;
            Point2f v1 = b - a;
            Point2f v2 = d - c;
            double cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

            if( fabs(cosAngle) > 0.7 ) {
                pt.x = (b.x + c.x) * 0.5;
                pt.y = (b.y + c.y) * 0.5;
            } else {
                double denom = a.x * (double)(d.y - c.y) + b.x * (double)(c.y - d.y) +
                               d.x * (double)(b.y - a.y) + c.x * (double)(a.y - b.y);
                double num = a.x * (double)(d.y - c.y) + c.x * (double)(a.y - d.y) + d.x * (double)(c.y - a.y);
                double s = num / denom;

                pt.x = a.x + s*(b.x - a.x);
                pt.y = a.y + s*(b.y - a.y);
            }


            outPoly.push_back(pt);
        }
    }


    static inline
    TextDetectionModel_DB_Impl& from(const std::shared_ptr<Model::Impl>& ptr)
    {
        CV_Assert(ptr);
        return *((TextDetectionModel_DB_Impl*)ptr.get());
    }
};


TextDetectionModel_DB::TextDetectionModel_DB()
    : TextDetectionModel()
{
    impl = std::static_pointer_cast<Model::Impl>(makePtr<TextDetectionModel_DB_Impl>());
}

TextDetectionModel_DB::TextDetectionModel_DB(const Net& network)
    : TextDetectionModel()
{
    impl = std::static_pointer_cast<Model::Impl>(makePtr<TextDetectionModel_DB_Impl>(network));
}

TextDetectionModel_DB& TextDetectionModel_DB::setBinaryThreshold(float binaryThreshold)
{
    TextDetectionModel_DB_Impl::from(impl).setBinaryThreshold(binaryThreshold);
    return *this;
}
float TextDetectionModel_DB::getBinaryThreshold() const
{
    return TextDetectionModel_DB_Impl::from(impl).getBinaryThreshold();
}

TextDetectionModel_DB& TextDetectionModel_DB::setPolygonThreshold(float polygonThreshold)
{
    TextDetectionModel_DB_Impl::from(impl).setPolygonThreshold(polygonThreshold);
    return *this;
}
float TextDetectionModel_DB::getPolygonThreshold() const
{
    return TextDetectionModel_DB_Impl::from(impl).getPolygonThreshold();
}

TextDetectionModel_DB& TextDetectionModel_DB::setUnclipRatio(double unclipRatio)
{
    TextDetectionModel_DB_Impl::from(impl).setUnclipRatio(unclipRatio);
    return *this;
}
double TextDetectionModel_DB::getUnclipRatio() const
{
    return TextDetectionModel_DB_Impl::from(impl).getUnclipRatio();
}

TextDetectionModel_DB& TextDetectionModel_DB::setMaxCandidates(int maxCandidates)
{
    TextDetectionModel_DB_Impl::from(impl).setMaxCandidates(maxCandidates);
    return *this;
}
int TextDetectionModel_DB::getMaxCandidates() const
{
    return TextDetectionModel_DB_Impl::from(impl).getMaxCandidates();
}


}}  // namespace
