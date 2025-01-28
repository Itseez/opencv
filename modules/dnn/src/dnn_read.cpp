// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


Net readNet(const String& _model, const String& _config, const String& _framework)
{
    String framework = toLowerCase(_framework);
    String model = _model;
    String config = _config;
    const std::string modelExt = model.substr(model.rfind('.') + 1);
    const std::string configExt = config.substr(config.rfind('.') + 1);
    if (framework == "caffe" || modelExt == "caffemodel" || configExt == "caffemodel" || modelExt == "prototxt" || configExt == "prototxt")
    {
        if (modelExt == "prototxt" || configExt == "caffemodel")
            std::swap(model, config);
        return readNetFromCaffe(config, model);
    }
    if (framework == "tensorflow" || modelExt == "pb" || configExt == "pb" || modelExt == "pbtxt" || configExt == "pbtxt")
    {
        if (modelExt == "pbtxt" || configExt == "pb")
            std::swap(model, config);
        return readNetFromTensorflow(model, config);
    }
    if (framework == "tflite" || modelExt == "tflite")
    {
        return readNetFromTFLite(model);
    }
    if (framework == "torch" || modelExt == "t7" || modelExt == "net" || configExt == "t7" || configExt == "net")
    {
        return readNetFromTorch(model.empty() ? config : model);
    }
    if (framework == "darknet" || modelExt == "weights" || configExt == "weights" || modelExt == "cfg" || configExt == "cfg")
    {
        if (modelExt == "cfg" || configExt == "weights")
            std::swap(model, config);
        return readNetFromDarknet(config, model);
    }
    if (framework == "dldt" || framework == "openvino" ||
        modelExt == "bin" || configExt == "bin" ||
        modelExt == "xml" || configExt == "xml")
    {
        if (modelExt == "xml" || configExt == "bin" || modelExt == "onnx")
            std::swap(model, config);
        return readNetFromModelOptimizer(config, model);
    }
    if (framework == "tensorrt" && (modelExt == "onnx" || modelExt == "trt"))
    {
        return readNetFromTensorRT(model);
    }
    if (framework == "onnx" || modelExt == "onnx")
    {
        return readNetFromONNX(model);
    }
    CV_Error(Error::StsError, "Cannot determine an origin framework of files: " + model + (config.empty() ? "" : ", " + config));
}

Net readNet(const String& _framework, const std::vector<uchar>& bufferModel,
        const std::vector<uchar>& bufferConfig)
{
    String framework = toLowerCase(_framework);
    if (framework == "onnx")
        return readNetFromONNX(bufferModel);
    else if (framework == "caffe")
        return readNetFromCaffe(bufferConfig, bufferModel);
    else if (framework == "tensorflow")
        return readNetFromTensorflow(bufferModel, bufferConfig);
    else if (framework == "darknet")
        return readNetFromDarknet(bufferConfig, bufferModel);
    else if (framework == "torch")
        CV_Error(Error::StsNotImplemented, "Reading Torch models from buffers");
    else if (framework == "dldt" || framework == "openvino")
        return readNetFromModelOptimizer(bufferConfig, bufferModel);
    else if (framework == "tflite")
        return readNetFromTFLite(bufferModel);
    CV_Error(Error::StsError, "Cannot determine an origin framework with a name " + framework);
}

Net readNetFromModelOptimizer(const String& xml, const String& bin)
{
    return Net::readFromModelOptimizer(xml, bin);
}

Net readNetFromModelOptimizer(const std::vector<uchar>& bufferCfg, const std::vector<uchar>& bufferModel)
{
    return Net::readFromModelOptimizer(bufferCfg, bufferModel);
}

Net readNetFromModelOptimizer(
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize)
{
    return Net::readFromModelOptimizer(
            bufferModelConfigPtr, bufferModelConfigSize,
            bufferWeightsPtr, bufferWeightsSize);
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
