// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include <mutex>
#include <map>
#include <cstring> // memcpy

#include "opencv2/dnn/dnn.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "../net_impl.hpp"

#ifdef HAVE_TRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "./trt_utils.h"
#include "./trt_logger.h"
#endif

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

TrtConfig::TrtConfig(const TrtConfig& config)
{
    deviceId = config.deviceId;
    useCache = config.useCache;
    cachePath = config.cachePath;
    useFP16 = config.useFP16;
    useTimeCache = config.useTimeCache;
    inputShape = config.inputShape;
    inputName = config.inputName;
}

TrtConfig::TrtConfig() :deviceId(0), useCache(false), cachePath(""), useFP16(false), useTimeCache(false), inputName("")
{}

TrtConfig& TrtConfig::operator=(const TrtConfig& config)
{
    deviceId = config.deviceId;
    useCache = config.useCache;
    cachePath = config.cachePath;
    useFP16 = config.useFP16;
    useTimeCache = config.useTimeCache;
    inputShape = config.inputShape;
    inputName = config.inputName;

    return *this;
}

#ifdef HAVE_TRT
#define DNN_TENSORRT_UNSUPPORTED() CV_Error(Error::StsError, "DNN/TenosrRT Backend: unsupported function!")

using namespace dnn_trt;

#define OPT_MAX_WORK_SPACE_SIZE ((size_t)1 << 30)

inline int convertTrt2CVType(const ::nvinfer1::DataType type)
{
    int cvType = -1;

    if (::nvinfer1::DataType::kFLOAT == type)
    {
        cvType = CV_32F;
    }
    else if (::nvinfer1::DataType::kHALF == type)
    {
        cvType = CV_16F;
    }
    else if (::nvinfer1::DataType::kINT8 == type)
    {
        cvType = CV_8S;
    }
    else if (::nvinfer1::DataType::kINT32 == type)
    {
        cvType = CV_32S;
    }
    else if (::nvinfer1::DataType::kUINT8 == type)
    {
        cvType = CV_8U;
    }
    else
    {
        CV_Error(CV_StsError, "TensorRT: Unsupported Trt Tensor type!");
    }

    return cvType;
}

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& getTensorrtLogger(bool verbose_log = false)
{
    const auto log_level = verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING;
    static TensorrtLogger trt_logger(log_level);

    if (log_level != trt_logger.get_level())
    {
        trt_logger.set_level(verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    }

    return trt_logger;
}

class NetImplTrt CV_FINAL : public Net::Impl
{
public:
    typedef Net::Impl Base;

    explicit NetImplTrt(const Ptr<Net::Impl>& basePtr)
        : Net::Impl()
    {
        CV_LOG_INFO(NULL, "Initializing NetImplTrt");
        basePtr_ = basePtr;
        init();

        CV_LOG_INFO(NULL, "Finished initializing NetImplTrt");
    }

    ~NetImplTrt()
    {
        for (int i = 0; i < bufferListDevice.size(); i++)
        {
            cudaFree(bufferListDevice[i]);
        }

        cudaStreamDestroy(stream_);
    }

    void init()
    {
        CV_TRACE_FUNCTION();
        CV_Assert(basePtr_);
        Net::Impl& base = *basePtr_;
        CV_Assert(!base.netWasAllocated);
        CV_Assert(!base.netWasQuantized); // does not support quantized net for now
        netInputLayer = base.netInputLayer;
        blobsToKeep = base.blobsToKeep;
        layers = base.layers;
        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
        {
            LayerData& ld = it->second;
            ld.resetAllocation();
        }
        layerNameToId = base.layerNameToId;
        outputNameToId = base.outputNameToId;
        preferableBackend = DNN_BACKEND_TENSORRT;
        preferableTarget = DNN_TARGET_TENSORRT; // force using TensorRT
        hasDynamicShapes = base.hasDynamicShapes;
        CV_Assert(base.backendWrappers.empty());  //backendWrappers = base.backendWrappers;
        lastLayerId = base.lastLayerId;
        netWasAllocated = base.netWasAllocated;
        netWasQuantized = base.netWasQuantized;
        fusion = base.fusion;
    }

    bool empty() const override
    {
        return builder_ != nullptr;
    }

    void setPreferableBackend(Net& net, int backendId) override
    {
        if (backendId == preferableBackend)
            return;  // no-op
        else
            CV_Error(Error::StsError, "DNN: Can't switch backend from TensorRT to other");
        Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
        impl_ptr_ref = basePtr_;
        basePtr_->setPreferableBackend(net, backendId);
    }

    void setPreferableTarget(int targetId) override
    {
        if (targetId != preferableTarget)
        {
            CV_Error(Error::StsError, "DNN: Can't switch target from TensorRT to other");
        }
    }

    Ptr<BackendWrapper> wrap(Mat& host) override
    {
        CV_Error(Error::StsNotImplemented, "DNN: Asynchronous forward is supported for Inference Engine backend only");
        return Ptr<BackendWrapper>();
    }

    // void fuseLayers(const std::vector<LayerPin>& blobsToKeep_); // fusion is done in the CANN graph engine

    void initBackend(const std::vector<LayerPin>& blobsToKeep_) override;

    void forwardLayer(LayerData& ld) override;

    void fuseLayers(const std::vector<LayerPin>& ) override
    {
        CV_LOG_INFO(NULL, "DNN TensorRT Backend not-implementation, skip the fuseLayers function!");
    }

    void allocateLayers(const std::vector<LayerPin>& )
    {
        CV_LOG_INFO(NULL, "DNN TensorRT Backend not-implementation, skip the allocateLayers function!");
    }

    void NetImplTrt::setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean);

    Mat forward(const String& outputName) override;

    void forward(OutputArrayOfArrays outputBlobs, const String& outputName) override;

    void forward(OutputArrayOfArrays outputBlobs, const std::vector<String>& outBlobNames) override;

    void forward(std::vector<std::vector<Mat>>& outputBlobs, const std::vector<String>& outBlobNames) override;

    AsyncArray forwardAsync(const String& outputName) override
    {
        CV_Error(Error::StsNotImplemented, "DNN: Asynchronous forward is supported for Inference Engine backend only");
    }

    // internal readNet function
    void readNet(const String& model, const TrtConfig& config);
    void readNet(const char* buffer, size_t sizeBuffer, const String &ext, const TrtConfig& config);

private:
    // Allocate Host memory and binding to the engine.
    void allocMem();
    void tensors2Mats(const std::vector<int>& outputIdxs, std::vector<Mat>& outputMat);

    int getOutputIndex(const String &name);
    int getInputIndex(const String& name);

    TrtConfig configTRT;
    std::vector<int> input_idxs;
    std::vector<int> output_idxs;

    //<<<<<<<<<<<<<<<<<< Common resource  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    int inputCount = 0;
    int outputCount = 0;
    std::vector<std::string> inputNamesString; // reserve model input name.
    std::vector<std::string> outputNamesString;

    std::vector<MatShape> inputMatShape;
    std::vector<MatShape> outputMatShape;

    //<<<<<<<<<<<<<<<<<< TensorRT resource  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    std::vector<::nvinfer1::Dims> inputMatShapeTrt;                      //!< The dimensions of the input to the network.
    std::vector<::nvinfer1::Dims> outputMatShapeTrt;                      //!< The dimensions of the input to the network.

    // The following two buffer list contains both input/output. It orgnizes as input_idxs/output_idxs
    std::vector<std::pair<AutoBuffer<uchar>, size_t>> bufferListHost; // pointer and size (can be overwritten by user)
    std::vector<void *> bufferListDevice;                  // pointer to GPU memory

    Ptr<::nvinfer1::IRuntime> runtime_;
    Ptr<::nvinfer1::ICudaEngine> engine_;
    Ptr<::nvinfer1::IExecutionContext> context_;

    Ptr<::nvinfer1::IBuilder> builder_;
    Ptr<::nvinfer1::INetworkDefinition> network_;
    Ptr<::nvinfer1::IBuilderConfig> config_;
    const bool verboseLog = false;

    //<<<<<<<<<<<<<<<<<< CUDA resource  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    cudaStream_t stream_ = nullptr;
    int device_id_;
    std::string compute_capability_;

    cv::Mutex mutex;
};

void NetImplTrt::initBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_LOG_INFO(NULL, "DNN TensorRT Backend skip the initBackend function!");
}

void NetImplTrt::forwardLayer(LayerData& ld)
{
    CV_LOG_INFO(NULL, "DNN TensorRT Backend skip the forwardLayer function!");
}

int NetImplTrt::getOutputIndex(const String &name)
{
    int indexRes = -1;

    // if the name is empty, we need to check out if the mode only need 1 input,
    // if it's true, then we set this input as this input.
    if (name.empty())
    {
        CV_Assert(outputNamesString.size() == 1 && "Please set the input name, the default input name can only be used in single input model.");
        indexRes = 0;
    }

    // find input index to get shape info.
    if (indexRes == -1)
    {
        auto iter = std::find(outputNamesString.begin(), outputNamesString.end(), name);

        if (iter != outputNamesString.end())
        {
            indexRes = iter - outputNamesString.begin();
        }
    }

    return indexRes;
}

int NetImplTrt::getInputIndex(const String& name)
{
    int indexRes = -1;

    // if the name is empty, we need to check out if the mode only need 1 input,
    // if it's true, then we set this input as this input.
    if (name.empty())
    {
        CV_Assert(inputNamesString.size() == 1 && "Please set the input name, the default input name can only be used in single input model.");
        indexRes = 0;
    }

    // find input index to get shape info.
    if (indexRes == -1)
    {
        auto iter = std::find(inputNamesString.begin(), inputNamesString.end(), name);

        if (iter != inputNamesString.end())
        {
            indexRes = iter - inputNamesString.begin();
        }
    }

    return indexRes;
}

void NetImplTrt::setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean)
{
    // TODO support the scalefactor and mean
    if (scalefactor != 1.0 || mean != Scalar())
    {
        CV_Error(Error::StsUnsupportedFormat, cv::format("DNN TensorRT: Failed to setInput with scalefactor or mean, please use blobFromImage!"));
    }

    int indexRes = getInputIndex(name);
    CV_Assert(indexRes != -1 && indexRes < inputCount && "TensorRT Backend: indexRes error called in setInput()!");

    // Trt model has two type of input: shape tensor and execution tensor.
    bool isShapeTensor = engine_->isShapeInferenceIO(name.c_str());
    CV_Assert(!isShapeTensor && "DNN TensorRT: Unsupported execution tensor input right now!");
    Mat blob_ = blob.getMat();

    std::vector<int> tensorShape = inputMatShape[indexRes];
    std::vector<int> matShape = shape(blob_);
    size_t totalValueT = total(tensorShape);
    size_t totalValueM = blob_.total();

    if (totalValueT != totalValueM || tensorShape.size() != matShape.size())
    {
        if (tensorShape.size() == 4 && matShape.size() == 4)
        {
            CV_LOG_WARNING(NULL, cv::format("The given input shape is [%d x %d x %d x %d], and the expected input shape is [%d x %d x %d x %d]", matShape[0], matShape[1], matShape[2], matShape[3], tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]));
        }

        if (tensorShape.size() == 3 && matShape.size() == 3)
            CV_LOG_WARNING(NULL, cv::format("The given input shape is [%d x %d x %d], and the expected input shape is [%d x %d x %d]", matShape[0], matShape[1], matShape[2], tensorShape[0], tensorShape[1], tensorShape[2]));
        CV_Error(CV_StsError, "The input shape dose not match the expacted input shape! \n");
    }

    int bindingIdx = input_idxs[indexRes];// idx in tensorrt
    int cvType = convertTrt2CVType(engine_->getBindingDataType(bindingIdx));
    CV_Assert(cvType == blob_.depth() && "The input Mat type is not match the Trt Tensor type!");

    memcpy(bufferListHost[bindingIdx].first.data(), blob_.data, bufferListHost[bindingIdx].second);
}

Mat NetImplTrt::forward(const String& outputName)
{
    std::vector<String> outputNames = {outputName};
    if (outputName.empty())
    {
        CV_Assert(outputNamesString.size() == 1 && "forward error! Please set the correct output name at .forward(\"SET_OUTPUT_NAME_HERE\")!");
        outputNames[0] = outputNamesString[0];
    }

    std::vector<Mat> outs;
    this->forward(outs, outputNames);

    return outs[0];
}

void NetImplTrt::forward(OutputArrayOfArrays outputBlobs, const String& outputName)
{
    std::vector<String> outputNames = {outputName};

    if (outputName.empty())
    {
        CV_Assert(outputNamesString.size() == 1 && "forward error! Please set the correct output name at .forward(\"SET_OUTPUT_NAME_HERE\")!");
        outputNames[0] = outputNamesString[0];
    }

    return this->forward(outputBlobs, outputNames);
}

void NetImplTrt::forward(OutputArrayOfArrays outputBlobs,
        const std::vector<String>& outBlobNames)
{
    CV_Assert(!empty());
    CV_Assert(outputBlobs.isMatVector());
    // Output depth can be CV_32F or CV_8S
    std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();
    int outSize = outBlobNames.size();
    CV_Assert(outSize <= outputCount && "OpenCV DNN forward() error, expected value exceeds existing value.");

    std::vector<int> outputIdx(outSize, -1);

    for(int i = 0; i < outSize; i++)
    {
        int res = getOutputIndex(outBlobNames[i]);

        if (res == -1) // un-found output name
        {
            CV_Error(Error::StsBadArg, cv::String("DNN TensorRT: Can not found the expacted output name = " + outputNamesString[i] + "!"));
            return;
        }

        outputIdx[i] = res;
    }

    for (int i = 0; i < inputCount; i++)
    {
        cudaMemcpyAsync(bufferListDevice[input_idxs[i]], bufferListHost[input_idxs[i]].first.data(),
            bufferListHost[input_idxs[i]].second, cudaMemcpyHostToDevice, stream_);
    }

    context_->enqueueV3(stream_);

    for (int i = 0; i < outputCount; i++)
    {
        cudaMemcpyAsync(bufferListHost[output_idxs[i]].first.data(), bufferListDevice[output_idxs[i]],
            bufferListHost[output_idxs[i]].second, cudaMemcpyDeviceToHost, stream_);
    }

    tensors2Mats(outputIdx, outputvec);
    cudaStreamSynchronize(stream_);
}

void NetImplTrt::forward(std::vector<std::vector<Mat>>& outputBlobs,
        const std::vector<String>& outBlobNames)
{
    outputBlobs.clear();
    std::vector<Mat> outs;
    this->forward(outs, outBlobNames);
    outputBlobs.push_back(outs);
}

// alloc gpu memory by cuda.
void NetImplTrt::allocMem()
{
    int allIONb = inputCount + outputCount;

    bufferListDevice.resize(allIONb, nullptr);
    bufferListHost.resize(allIONb, {AutoBuffer<uchar>(), 0});

    for (int i = 0; i < input_idxs.size(); i++)
    {
        int idx = input_idxs[i];
        CV_Assert(idx >=0 && idx < allIONb);
        int cvType = convertTrt2CVType(engine_->getBindingDataType(idx));
        size_t dataSize = CV_ELEM_SIZE1(cvType) * total(inputMatShape[i]);
        bufferListHost[idx].first.allocate(dataSize);
        CV_Assert(bufferListHost[idx].first.data());
        bufferListHost[idx].second = dataSize;
        cudaMalloc(&bufferListDevice[idx], dataSize);
        CV_Assert(bufferListDevice[idx]);

        context_->setTensorAddress(inputNamesString[i].c_str(), bufferListDevice[idx]);
    }

    for (int i = 0; i < output_idxs.size(); i++)
    {
        int idx = output_idxs[i];
        CV_Assert(idx >=0 && idx < allIONb);
        int cvType = convertTrt2CVType(engine_->getBindingDataType(idx));
        size_t dataSize = CV_ELEM_SIZE1(cvType) * total(outputMatShape[i]);
        bufferListHost[idx].first.allocate(dataSize);
        bufferListHost[idx].second = dataSize;
        CV_Assert(bufferListHost[idx].first.data());
        cudaMalloc(&bufferListDevice[idx], dataSize);
        CV_Assert(bufferListDevice[idx]);

        context_->setTensorAddress(outputNamesString[i].c_str(), bufferListDevice[idx]);
    }
}

// convert GPU tensorrt to opencv cpu Mats.
void NetImplTrt::tensors2Mats(const std::vector<int>& outputIdxs, std::vector<Mat>& outs)
{
    if (outs.empty() || outs.size() != outputIdxs.size())
        outs.resize(outputIdxs.size());

    for (int i = 0; i < outputIdxs.size(); i++)
    {
        int idx = outputIdxs[i];
        int bindingIdx = output_idxs[idx];
        int cvType = convertTrt2CVType(engine_->getBindingDataType(bindingIdx));

        CV_Assert(cvType != -1 && "Unsupported data type");
        Mat(outputMatShape[idx], cvType, bufferListHost[bindingIdx].first.data()).copyTo(outs[idx]);
    }
}

// remove the file name and return the file prefix path.
static inline std::string removeFileName(const std::string& filePath)
{
    size_t pos = filePath.find_last_of("/\\");

    if (pos != std::string::npos)
    {
        return filePath.substr(0, pos);
    }
    else
    {
        return "";
    }
}

// return file name with file extention suffix
static inline std::string extractFileName(const std::string& filePath)
{
    size_t pos = filePath.find_last_of("/\\");

    if (pos != std::string::npos)
    {
        return filePath.substr(pos + 1);
    }
    else
    {
        return filePath;
    }
}

// remove file extention suffix
static inline std::string removeFileSuffix(const std::string& filePath)
{
    size_t pos = filePath.find_last_of(".");

    if (pos != std::string::npos)
    {
        return filePath.substr(0, pos);
    }
    else
    {
        return filePath;
    }
}

static inline MatShape convertDim2Shape(const ::nvinfer1::Dims& dim)
{
    MatShape shape(dim.nbDims, 0);
    memcpy(shape.data(), dim.d, dim.nbDims * sizeof(int));

    return shape;
}

static inline ::nvinfer1::Dims convertShape2Dim(const MatShape& shape)
{
    ::nvinfer1::Dims dim;
    dim.nbDims = shape.size();
    memcpy(dim.d, shape.data(), shape.size() * sizeof(int));

    return dim;
}


static std::vector<char> loadTimingCacheFile(const std::string inFileName)
{
    std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);

    if (!iFile)
    {
        CV_LOG_INFO(NULL, cv::String("[TensorRT EP] Could not read timing cache from: "+inFileName
                            +". A new timing cache will be generated and written."));
        return std::vector<char>();
    }

    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    std::vector<char> content(fsize);
    iFile.read(content.data(), fsize);
    iFile.close();

    return content;
}

static void saveTimingCacheFile(const std::string outFileName, const nvinfer1::IHostMemory* blob)
{
    std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);

    if (!oFile)
    {
        CV_LOG_INFO(NULL, cv::String("[TensorRT EP] Could not write timing cache to: "+outFileName));
        return;
    }

    oFile.write((char*)blob->data(), blob->size());
    oFile.close();
}

static std::string get_compute_capability(int device_id)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    return "_" + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
}

void NetImplTrt::readNet(const String& model, const TrtConfig& configTRT)
{
    device_id_ = configTRT.deviceId;
    std::string trt_model_filename = ""; // Contain the cache file name, when input is ONNX and need to save cache.
    std::string path_to_write = "";

    int GPU_count = 0;
    cudaGetDeviceCount(&GPU_count);
    CV_Assert(device_id_ >= 0 && device_id_ < GPU_count && "The device id does not exist!");

    // TODO support multi-GPU
    cudaSetDevice(device_id_); // set specific GPU device for TensorRT backend.

    this->compute_capability_ = get_compute_capability(device_id_);

    const std::string modelExt = model.substr(model.rfind('.') + 1);
    bool is_trt_model = false;

    if (configTRT.useCache)
    {
        if (!configTRT.cachePath.empty())
        {
            path_to_write = configTRT.cachePath + "/";
        }
        else
        {
            path_to_write = removeFileName(model) + "/";
        }
    }

    if (modelExt == "trt")
    {
        is_trt_model = true;
    }
    else if (modelExt == "onnx")
    {
        // The input is onnx, try to find out if there is a converted model under the cachePath.
        if (configTRT.useCache)
        {
            std::string modelFileName = removeFileSuffix(extractFileName(model));
            trt_model_filename = path_to_write + modelFileName + ".trt\0";
            std::ifstream trtFile(trt_model_filename);

            if (trtFile.is_open())  // related cache file exist, use cache file.
            {
                CV_LOG_INFO(NULL, cv::String("DNN TensorRT backend: found Trt cache model:" + trt_model_filename));
                is_trt_model = true;
            }
            else // no cache file was found,
            {
                is_trt_model = false;
            }
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to load file with extension: %s !", modelExt.c_str()));

    {
        AutoLock lock(mutex);
        runtime_ = Ptr<::nvinfer1::IRuntime>(::nvinfer1::createInferRuntime(getTensorrtLogger()));
    }

    if (!runtime_)
    {
        CV_Error(Error::StsError, "DNN TensorRT backend: Failed to create runtime!");
        return;
    }

    std::string timeCachingPath = "";
    /*** create engine from model file ***/
    if (is_trt_model)
    {
        /* Just load TensorRT model (serialized model) */
        std::ifstream engine_file(trt_model_filename, std::ios::binary | std::ios::in);
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);

        {
            AutoLock lock(mutex);
            engine_ = Ptr<::nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        }

        engine_file.close();
        if (!engine_)
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to create engine!");
            return;
        }
    }
    else // is onnx
    {
        if (configTRT.useTimeCache)
        {
            timeCachingPath = getTimingCachePath(path_to_write, this->compute_capability_);
        }

        /* Create a TensorRT model from another format */
        AutoLock lock(mutex);
        builder_ = Ptr<::nvinfer1::IBuilder>(::nvinfer1::createInferBuilder(getTensorrtLogger()));

        const auto explicitBatch = 1U << static_cast<uint32_t>(::nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network_ = Ptr<::nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
        config_ = Ptr<::nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());

        if (!configTRT.inputName.empty() && !configTRT.inputShape.empty())
        {
            ::nvinfer1::IOptimizationProfile* profile = builder_->createOptimizationProfile();
            ::nvinfer1::Dims dim = convertShape2Dim(configTRT.inputShape);
            profile->setDimensions(configTRT.inputName.c_str(), ::nvinfer1::OptProfileSelector::kMIN, dim);
            profile->setDimensions(configTRT.inputName.c_str(), ::nvinfer1::OptProfileSelector::kOPT, dim);
            profile->setDimensions(configTRT.inputName.c_str(), ::nvinfer1::OptProfileSelector::kMAX, dim);

            config_->addOptimizationProfile(profile);
        }

        auto parser = Ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, getTensorrtLogger()));

        if (!parser->parseFromFile(model.c_str(), (int)::nvinfer1::ILogger::Severity::kWARNING))
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to parse onnx file!");
            return;
        }

        config_->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);

        if (configTRT.useFP16)
        {
            config_->setFlag(::nvinfer1::BuilderFlag::kFP16);
        }

        // Trying to load time cache file
        std::unique_ptr<nvinfer1::ITimingCache> timing_cache = nullptr;
        if (configTRT.useTimeCache)
        {
            // Loading time cache file, create a fresh cache if the file doesn't exist.
            std::vector<char> loaded_timing_cache = loadTimingCacheFile(timeCachingPath);
            timing_cache.reset(config_->createTimingCache(static_cast<const void*>(loaded_timing_cache.data()), loaded_timing_cache.size()));
            if (timing_cache == nullptr)
            {
                CV_Error(Error::StsError, "TensorRT backend: Failed to create timing cache!");
                return;
            }
            config_->setTimingCache(*timing_cache, false);
        }

        auto plan = Ptr<::nvinfer1::IHostMemory>(builder_->buildSerializedNetwork(*network_, *config_));

        if (!plan)
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to serialized network!");
            return;
        }

        engine_ = Ptr<::nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

        if (!engine_)
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to create engine!");
            return;
        }

        /* save serialized model to cache folder */
        if (configTRT.useCache && !configTRT.cachePath.empty())
        {
            std::ofstream ofs(std::string(trt_model_filename), std::ios::out | std::ios::binary);
            ofs.write((char*)(plan->data()), plan->size());
            ofs.close();
        }

        // Trying to save time cache file.
        if (configTRT.useTimeCache)
        {
            auto timing_cache = config_->getTimingCache();
            std::unique_ptr<nvinfer1::IHostMemory> timingCacheHostData{timing_cache->serialize()};

            if (timingCacheHostData == nullptr)
            {
                CV_Error(Error::StsError, cv::String("TensorRT backend: could not serialize timing cache:"+trt_model_filename));
                return;
            }
            saveTimingCacheFile(timeCachingPath, timingCacheHostData.get());
        }
    }

    {
        AutoLock lock(mutex);
        context_ = Ptr<::nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    }

    if (!context_)
    {
        CV_Error(Error::StsError, "TensorRT backend: Failed to create context!");
        return;
    }

    // parsing model i/o name, dim, data type.
    int ioNb = engine_->getNbIOTensors();

    for (int i = 0; i < ioNb; i++)
    {
        bool isInput = engine_->bindingIsInput(i);
        std::string name = std::string(engine_->getBindingName(i));
        ::nvinfer1::Dims dim = engine_->getBindingDimensions(i);
        MatShape shape = convertDim2Shape(dim);

        if (isInput)
        {
            inputCount++;
            inputNamesString.push_back(name);
            inputMatShape.push_back(shape);
            input_idxs.push_back(i);
            inputMatShapeTrt.push_back(dim);
        }
        else
        {
            outputCount++;
            outputNamesString.push_back(name);
            outputMatShape.push_back(shape);
            output_idxs.push_back(i);
            outputMatShapeTrt.push_back(dim);
        }
    }

    this->allocMem();
}

void NetImplTrt::readNet(const char* buffer, size_t sizeBuffer, const String &ext, const TrtConfig& config)
{
    // TODO
}

// TensorRT read Net function.
Net readNetFromTensorRT(const String& trtFile, const TrtConfig& config)
{
    Net net;

    Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
    Ptr<NetImplTrt> impl_ptr_trt = makePtr<NetImplTrt>(impl_ptr_ref);

    impl_ptr_trt->readNet(trtFile, config);
    impl_ptr_ref = impl_ptr_trt;
    return net;
}

#else

Net readNetFromTensorRT(const String& trtFile, const TrtConfig& config)
{
    CV_Error(Error::StsError, "TenosrRT Backend: unsupport TensorRT, please recompile OpenCV with TensorRT!");
    Net net;
    return net;
}

#endif // HAVE_TRT

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
