// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_SRC_NET_IMPL_HPP__
#define __OPENCV_DNN_SRC_NET_IMPL_HPP__

#include "op_halide.hpp"
#include "op_inf_engine.hpp"
#include "ie_ngraph.hpp"
#include "op_vkcom.hpp"
#include "op_cuda.hpp"
#include "op_webnn.hpp"
#include "op_timvx.hpp"
#include "op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/layer_reg.private.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/logger.hpp>

#include "layer_internals.hpp"  // LayerPin LayerData DataLayer

#include "legacy_backend.hpp"  // wrapMat BlobManager OpenCLBackendWrapper

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::make_pair;
using std::string;

// NB: Implementation is divided between of multiple .cpp files
struct Net::Impl : public detail::NetImplBase
{
    typedef std::map<int, LayerShapes> LayersShapesMap;
    typedef std::map<int, LayerData> MapIdToLayerData;

    virtual ~Impl();
    Impl();
    Impl(const Impl&) = delete;

    // Inheritance support
    Ptr<Net::Impl> basePtr_;

    Ptr<DataLayer> netInputLayer;
    std::vector<LayerPin> blobsToKeep;
    MapIdToLayerData layers;
    std::map<String, int> layerNameToId;
    std::map<std::string, int> outputNameToId;  // use registerOutput() to populate outputs
    BlobManager blobManager;
    int preferableBackend;
    int preferableTarget;
    String halideConfigFile;
    bool hasDynamicShapes;
    // Map host data to backend specific wrapper.
    std::map<void*, Ptr<BackendWrapper>> backendWrappers;

    int lastLayerId;

    bool netWasAllocated;
    bool netWasQuantized;
    bool fusion;
    bool isAsync;  // FIXIT: drop
    bool useWinograd;
    std::vector<int64> layersTimings;


    virtual bool empty() const;
    virtual void setPreferableBackend(Net& net, int backendId);
    virtual void setPreferableTarget(int targetId);

    // FIXIT use inheritance
    virtual Ptr<BackendWrapper> wrap(Mat& host);


    virtual void clear();


    virtual void validateBackendAndTarget();

    void setUpNet(const std::vector<LayerPin>& blobsToKeep_ = std::vector<LayerPin>());


    virtual Ptr<Layer> createLayerInstance(const LayerData& ld) const
    {
        return LayerFactory::createLayerInstance(ld.type, const_cast<LayerParams&>(ld.params));
    }
    Ptr<Layer> getLayerInstance(LayerData& ld) const
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(type, "type", ld.type.c_str());

        if (ld.layerInstance)
            return ld.layerInstance;

        ld.layerInstance = createLayerInstance(ld);
        if (!ld.layerInstance && basePtr_)
        {
            ld.layerInstance = basePtr_->createLayerInstance(ld);
            CV_LOG_IF_DEBUG(NULL, ld.layerInstance, "Created layer \"" + ld.name + "\" of type \"" + ld.type + "\" from upstream layers registry");
        }
        if (!ld.layerInstance)
        {
            CV_Error(Error::StsError, "Can't create layer \"" + ld.name + "\" of type \"" + ld.type + "\"");
        }

        return ld.layerInstance;
    }

    Ptr<Layer> getLayer(int layerId) const;
    Ptr<Layer> getLayer(const LayerId& layerId) const;

    int getLayerId(const String& layerName) const;

    int getLayerId(int id) const;

    int getLayerId(DictValue& layerDesc) const;

    String getLayerName(int id) const;

    LayerData& getLayerData(int id) const;

    LayerData& getLayerData(const String& layerName) const;

    LayerData& getLayerData(const DictValue& layerDesc) const;

    static void addLayerInput(LayerData& ld, int inNum, LayerPin from);

    int resolvePinOutputName(LayerData& ld, const String& outName) const;

    LayerPin getPinByAlias(const String& layerName) const;

    std::vector<LayerPin> getLayerOutPins(const String& layerName) const;

    // FIXIT remove dtype
    int addLayer(const String& name, const String& type, const int& dtype, LayerParams& params);

    int addLayerToPrev(const String& name, const String& type, const int& dtype, LayerParams& params);


    void connect(int outLayerId, int outNum, int inLayerId, int inNum);

    int registerOutput(const std::string& outputName, int layerId, int outputPort);

    // FIXIT drop "unconnected" API
    std::vector<int> getUnconnectedOutLayers() const;
    std::vector<String> getUnconnectedOutLayersNames() /*const*/;

    void setInputsNames(const std::vector<String>& inputBlobNames);
    void setInputShape(const String& inputName, const MatShape& shape);
    virtual void setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean);
    Mat getParam(int layer, int numParam) const;
    void setParam(int layer, int numParam, const Mat& blob);
    std::vector<Ptr<Layer>> getLayerInputs(int layerId) const;
    std::vector<String> getLayerNames() const;

    // TODO drop?
    void getLayerTypes(std::vector<String>& layersTypes) const;
    int getLayersCount(const String& layerType) const;


    virtual void initBackend(const std::vector<LayerPin>& blobsToKeep_);

    void setHalideScheduler(const String& scheduler);
#ifdef HAVE_HALIDE
    void compileHalide();
    void initHalideBackend();
#endif

#ifdef HAVE_WEBNN
    void addWebnnOutputs(LayerData& ld);
    void initWebnnBackend(const std::vector<LayerPin>& blobsToKeep_);
#endif

#ifdef HAVE_VULKAN
    Ptr<vkcom::Context> context;
    void initVkComBackend();
#endif

#ifdef HAVE_TIMVX
    // Create timVxInfo for reserve tvGraphList.
    TimVXInfo timVxInfo = TimVXInfo();
    void tvUpdateConfictMap(int graphIndex, LayerData& ld, std::vector<std::vector<int> >& graphConflictMap);
    void tvConvertToOutputNode(const LayerData& ld, Ptr<TimVXBackendWrapper>& targetWrap);
    void initTimVXBackend();
#endif

#ifdef HAVE_CUDA
    struct CudaInfo_t
    {
        CudaInfo_t(cuda4dnn::csl::CSLContext ctxt, cuda4dnn::csl::Stream d2h_stream_)
            : context(std::move(ctxt))
            , d2h_stream(std::move(d2h_stream_))
        {}
        cuda4dnn::csl::CSLContext context;
        cuda4dnn::csl::Stream d2h_stream;
        cuda4dnn::csl::Workspace workspace;
    };

    std::unique_ptr<CudaInfo_t> cudaInfo;

    void initCUDABackend(const std::vector<LayerPin>& blobsToKeep_);
#endif

    void allocateLayer(int lid, const LayersShapesMap& layersShapes);

    // TODO add getter
    void enableFusion(bool fusion_);

    virtual void fuseLayers(const std::vector<LayerPin>& blobsToKeep_);
    void enableWinograd(bool useWinograd_);

    void allocateLayers(const std::vector<LayerPin>& blobsToKeep_);

    virtual void forwardLayer(LayerData& ld);

    void forwardToLayer(LayerData& ld, bool clearFlags = true);

    virtual Mat forward(const String& outputName);
    virtual AsyncArray forwardAsync(const String& outputName);
    virtual void forward(OutputArrayOfArrays outputBlobs, const String& outputName);
    virtual void forward(OutputArrayOfArrays outputBlobs,
            const std::vector<String>& outBlobNames);
    virtual void forward(std::vector<std::vector<Mat>>& outputBlobs,
            const std::vector<String>& outBlobNames);


    void getLayerShapesRecursively(int id, LayersShapesMap& inOutShapes);

    void getLayersShapes(
            const ShapesVec& netInputShapes,
            std::vector<int>& layersIds,
            std::vector<ShapesVec>& inLayersShapes,
            std::vector<ShapesVec>& outLayersShapes) /*const*/;

    void getLayersShapes(const ShapesVec& netInputShapes,
            LayersShapesMap& inOutShapes);

    void getLayerShapes(const ShapesVec& netInputShapes,
            const int layerId,
            LayerShapes& shapes);

    void updateLayersShapes();

    int64 getFLOPS(const std::vector<MatShape>& netInputShapes) /*const*/;
    int64 getFLOPS(
            const int layerId,
            const std::vector<MatShape>& netInputShapes) /*const*/;

    void getMemoryConsumption(
            const int layerId,
            const std::vector<MatShape>& netInputShapes,
            size_t& weights, size_t& blobs) /*const*/;
    void getMemoryConsumption(
            const std::vector<MatShape>& netInputShapes,
            size_t& weights, size_t& blobs) /*const*/;
    void getMemoryConsumption(
            const std::vector<MatShape>& netInputShapes,
            std::vector<int>& layerIds, std::vector<size_t>& weights,
            std::vector<size_t>& blobs) /*const*/;
    int64 getPerfProfile(std::vector<double>& timings) const;

    // TODO drop
    LayerPin getLatestLayerPin(const std::vector<LayerPin>& pins) const;

    Mat getBlob(const LayerPin& pin) const;

    Mat getBlob(String outputName) const;

    virtual AsyncArray getBlobAsync(const LayerPin& pin);

    AsyncArray getBlobAsync(String outputName);

    string dump(bool forceAllocation = false) const;
    string dumpToPbtxt(bool forceAllocation = false) const;

    void dumpNetworkToFile() const;

    // FIXIT drop from inference API
    Net quantize(Net& net, InputArrayOfArrays calibData, int inputsDtype, int outputsDtype, bool perChannel) /*const*/;
    void getInputDetails(std::vector<float>& scales, std::vector<int>& zeropoints) /*const*/;
    void getOutputDetails(std::vector<float>& scales, std::vector<int>& zeropoints) /*const*/;

};  // Net::Impl


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // __OPENCV_DNN_SRC_NET_IMPL_HPP__
