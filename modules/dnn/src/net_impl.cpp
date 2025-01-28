// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


static int g_networkId = 0;


detail::NetImplBase::NetImplBase()
    : networkId(CV_XADD(&g_networkId, 1))
    , networkDumpCounter(0)
    , dumpLevel(getParam_DNN_NETWORK_DUMP())
{
    // nothing
}


std::string detail::NetImplBase::getDumpFileNameBase() const
{
    std::string dumpFileNameBase = cv::format("ocv_dnn_net_%05d_%02d", networkId, networkDumpCounter++);
    return dumpFileNameBase;
}


Net::Impl::~Impl()
{
#ifdef HAVE_VULKAN
    if (context)
        context->reset();
#endif
}


Net::Impl::Impl()
{
    // allocate fake net input layer
    netInputLayer = Ptr<DataLayer>(new DataLayer());
    LayerData& inpl = layers.insert(make_pair(0, LayerData())).first->second;
    inpl.id = 0;
    netInputLayer->name = inpl.name = "_input";
    inpl.type = "__NetInputLayer__";
    inpl.layerInstance = netInputLayer;
    layerNameToId.insert(std::make_pair(inpl.name, inpl.id));

    lastLayerId = 0;
    netWasAllocated = false;
    netWasQuantized = false;
    fusion = true;
    isAsync = false;
    preferableBackend = (Backend)getParam_DNN_BACKEND_DEFAULT();
    preferableTarget = DNN_TARGET_CPU;
    hasDynamicShapes = false;
    useWinograd = true;
}


bool Net::Impl::empty() const
{
    return layers.size() <= 1;  // first layer is default Data layer
}


void Net::Impl::clear()
{
    CV_TRACE_FUNCTION();

    MapIdToLayerData::iterator it;
    for (it = layers.begin(); it != layers.end(); it++)
    {
        if (it->second.id != 0)
        {
            it->second.inputBlobs.clear();
            it->second.outputBlobs.clear();
            it->second.internals.clear();
        }
        it->second.skip = false;
        // it->second.consumers.clear();
        Ptr<Layer> currLayer = it->second.layerInstance;

        if (currLayer.empty())
            continue;

        currLayer->unsetAttached();
    }
    netWasAllocated = false;
    layersTimings.clear();
}


void Net::Impl::validateBackendAndTarget()
{
    CV_TRACE_FUNCTION();

    CV_Assert(preferableBackend != DNN_BACKEND_OPENCV ||
              preferableTarget == DNN_TARGET_CPU ||
              preferableTarget == DNN_TARGET_CPU_FP16 ||
              preferableTarget == DNN_TARGET_OPENCL ||
              preferableTarget == DNN_TARGET_OPENCL_FP16);
    CV_Assert(preferableBackend != DNN_BACKEND_HALIDE ||
              preferableTarget == DNN_TARGET_CPU ||
              preferableTarget == DNN_TARGET_OPENCL);
#ifdef HAVE_WEBNN
    if (preferableBackend == DNN_BACKEND_WEBNN)
    {
        CV_Assert(preferableTarget == DNN_TARGET_CPU ||
                  preferableTarget == DNN_TARGET_OPENCL);
    }
#endif
    CV_Assert(preferableBackend != DNN_BACKEND_VKCOM ||
              preferableTarget == DNN_TARGET_VULKAN);
    CV_Assert(preferableBackend != DNN_BACKEND_CUDA ||
              IS_DNN_CUDA_TARGET(preferableTarget));
    CV_Assert(preferableBackend != DNN_BACKEND_TIMVX ||
              preferableTarget == DNN_TARGET_NPU);

    CV_Assert(preferableBackend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && "Inheritance internal error");
}

void Net::Impl::setUpNet(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();

    if (dumpLevel && networkDumpCounter == 0)
    {
        dumpNetworkToFile();
    }

    validateBackendAndTarget();

    if (!netWasAllocated || this->blobsToKeep != blobsToKeep_)
    {
        if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
#ifndef HAVE_OPENCL
        {
            CV_LOG_WARNING(NULL, "DNN: OpenCL target is not available in this OpenCV build, switching to CPU.");
            preferableTarget = DNN_TARGET_CPU;
        }
#else
        {
            if (!getParam_DNN_OPENCL_ALLOW_ALL_DEVICES())
            {
                // Current implementation is only valid for GPU (#11494)
                if (ocl::Device::getDefault().type() != ocl::Device::TYPE_GPU)
                {
                    CV_LOG_WARNING(NULL, "DNN: OpenCL target is not supported with current OpenCL device (tested with GPUs only), switching to CPU.");
                    preferableTarget = DNN_TARGET_CPU;
                }
                else if (preferableTarget == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
                {
                    CV_LOG_WARNING(NULL,
                            "DNN: OpenCL target with fp16 precision is not supported "
                            "with current OpenCL device (tested with Intel GPUs only), "
                            "switching to OpenCL with fp32 precision.");
                    preferableTarget = DNN_TARGET_OPENCL;
                }
            }
        }
#endif
        if (preferableBackend == DNN_BACKEND_VKCOM && !haveVulkan())
        {
            preferableBackend = DNN_BACKEND_OPENCV;
            preferableTarget = DNN_TARGET_CPU;
        }

        if (preferableBackend == DNN_BACKEND_CUDA && !haveCUDA())
        {
#ifdef HAVE_CUDA
            CV_LOG_WARNING(NULL, "unable to use CUDA backend; switching to CPU");
#else
            CV_LOG_WARNING(NULL, "DNN module was not built with CUDA backend; switching to CPU");
#endif
            preferableBackend = DNN_BACKEND_OPENCV;
            preferableTarget = DNN_TARGET_CPU;
        }

        if (preferableBackend == DNN_BACKEND_TIMVX && !haveTimVX())
        {
            preferableBackend = DNN_BACKEND_OPENCV;
            preferableTarget = DNN_TARGET_CPU;
        }

        clear();

        if (hasDynamicShapes)
        {
            updateLayersShapes();
        }

        this->blobsToKeep = blobsToKeep_;

        allocateLayers(blobsToKeep_);

        MapIdToLayerData::iterator it = layers.find(0);
        CV_Assert(it != layers.end());
        it->second.skip = netInputLayer->skip;

        initBackend(blobsToKeep_);

        if (!netWasAllocated)
        {
#ifdef HAVE_HALIDE
            if (preferableBackend == DNN_BACKEND_HALIDE)
                compileHalide();
#else
            CV_Assert(preferableBackend != DNN_BACKEND_HALIDE);
#endif
        }

        netWasAllocated = true;

        if (dumpLevel)
        {
            dumpNetworkToFile();
        }
    }
}


Ptr<Layer> Net::Impl::getLayer(int layerId) const
{
    LayerData& ld = getLayerData(layerId);
    return getLayerInstance(ld);
}


Ptr<Layer> Net::Impl::getLayer(const LayerId& layerId) const
{
    LayerData& ld = getLayerData(layerId);
    return getLayerInstance(ld);
}


int Net::Impl::getLayerId(const String& layerName) const
{
    std::map<String, int>::const_iterator it = layerNameToId.find(layerName);
    return (it != layerNameToId.end()) ? it->second : -1;
}


int Net::Impl::getLayerId(int id) const
{
    MapIdToLayerData::const_iterator it = layers.find(id);
    return (it != layers.end()) ? id : -1;
}


int Net::Impl::getLayerId(DictValue& layerDesc) const
{
    if (layerDesc.isInt())
        return getLayerId(layerDesc.get<int>());
    else if (layerDesc.isString())
        return getLayerId(layerDesc.get<String>());

    CV_Assert(layerDesc.isInt() || layerDesc.isString());
    return -1;
}


String Net::Impl::getLayerName(int id) const
{
    MapIdToLayerData::const_iterator it = layers.find(id);
    return (it != layers.end()) ? it->second.name : "(unknown layer)";
}


LayerData& Net::Impl::getLayerData(int id) const
{
    MapIdToLayerData::const_iterator it = layers.find(id);

    if (it == layers.end())
        CV_Error(Error::StsObjectNotFound, format("Layer with requested id=%d not found", id));

    return const_cast<LayerData&>(it->second);
}


LayerData& Net::Impl::getLayerData(const String& layerName) const
{
    int id = getLayerId(layerName);

    if (id < 0)
        CV_Error(Error::StsError, "Requested layer \"" + layerName + "\" not found");

    return getLayerData(id);
}


LayerData& Net::Impl::getLayerData(const DictValue& layerDesc) const
{
    CV_Assert(layerDesc.isInt() || layerDesc.isString());
    if (layerDesc.isInt())
        return getLayerData(layerDesc.get<int>());
    else /*if (layerDesc.isString())*/
        return getLayerData(layerDesc.get<String>());
}


/*static*/
void Net::Impl::addLayerInput(LayerData& ld, int inNum, LayerPin from)
{
    if ((int)ld.inputBlobsId.size() <= inNum)
    {
        ld.inputBlobsId.resize(inNum + 1);
    }
    else
    {
        LayerPin storedFrom = ld.inputBlobsId[inNum];
        if (storedFrom.valid() && !storedFrom.equal(from))
            CV_Error(Error::StsError, format("Input #%d of layer \"%s\" already was connected",
                                             inNum, ld.name.c_str()));
    }

    ld.inputBlobsId[inNum] = from;
}


int Net::Impl::resolvePinOutputName(LayerData& ld, const String& outName) const
{
    if (outName.empty())
        return 0;
    return getLayerInstance(ld)->outputNameToIndex(outName);
}


LayerPin Net::Impl::getPinByAlias(const String& layerName) const
{
    LayerPin pin;
    pin.lid = (layerName.empty()) ? 0 : getLayerId(layerName);

    if (pin.lid >= 0)
        pin.oid = resolvePinOutputName(getLayerData(pin.lid), layerName);

    return pin;
}


std::vector<LayerPin> Net::Impl::getLayerOutPins(const String& layerName) const
{
    int lid = (layerName.empty()) ? 0 : getLayerId(layerName);

    MapIdToLayerData::const_iterator it = layers.find(lid);
    if (it == layers.end())
        CV_Error_(Error::StsOutOfRange, ("Layer #%d is not valid", lid));
    const size_t nOutputs = it->second.outputBlobs.size();

    std::vector<LayerPin> pins;
    for (int i = 0; i < nOutputs; i++)
    {
        pins.push_back(LayerPin(lid, i));
    }

    return pins;
}


// FIXIT remove dtype
int Net::Impl::addLayer(const String& name, const String& type, const int& dtype, LayerParams& params)
{
    int id = getLayerId(name);
    if (id >= 0)
    {
        if (!DNN_DIAGNOSTICS_RUN || type != "NotImplemented")
        {
            CV_Error(Error::StsBadArg, "Layer \"" + name + "\" already into net");
            return -1;
        }
        else
        {
            LayerData& ld = layers.find(id)->second;
            ld.type = type;
            ld.params = params;
            return -1;
        }
    }

    id = ++lastLayerId;
    layerNameToId.insert(std::make_pair(name, id));
    layers.insert(std::make_pair(id, LayerData(id, name, type, dtype, params)));
    if (params.get<bool>("has_dynamic_shapes", false))
        hasDynamicShapes = true;

    if (dtype == CV_8S)
        netWasQuantized = true;

    return id;
}


int Net::Impl::addLayerToPrev(const String& name, const String& type, const int& dtype, LayerParams& params)
{
    int prvLid = lastLayerId;
    int newLid = addLayer(name, type, dtype, params);
    connect(prvLid, 0, newLid, 0);
    return newLid;
}


void Net::Impl::connect(int outLayerId, int outNum, int inLayerId, int inNum)
{
    CV_Assert(outLayerId < inLayerId);
    LayerData& ldOut = getLayerData(outLayerId);
    LayerData& ldInp = getLayerData(inLayerId);

    addLayerInput(ldInp, inNum, LayerPin(outLayerId, outNum));
    ldOut.requiredOutputs.insert(outNum);
    ldOut.consumers.push_back(LayerPin(inLayerId, outNum));

    CV_LOG_VERBOSE(NULL, 0, "DNN: connect(" << outLayerId << ":" << outNum << " ==> " << inLayerId << ":" << inNum << ")");
}


int Net::Impl::registerOutput(const std::string& outputName, int layerId, int outputPort)
{
    int checkLayerId = getLayerId(outputName);
    if (checkLayerId >= 0)
    {
        if (checkLayerId == layerId)
        {
            if (outputPort == 0)
            {
                // layer name correlates with its output name
                CV_LOG_DEBUG(NULL, "DNN: register output='" << outputName << "': reuse layer with the same name and id=" << layerId << " to be linked");
                outputNameToId.insert(std::make_pair(outputName, layerId));
                return checkLayerId;
            }
        }
        CV_Error_(Error::StsBadArg, ("Layer with name='%s' already exists id=%d (to be linked with %d:%d)", outputName.c_str(), checkLayerId, layerId, outputPort));
    }
#if 0  // TODO
    if (outputPort == 0)
        // make alias only, need to adopt getUnconnectedOutLayers() call
#endif
    LayerParams outputLayerParams;
    outputLayerParams.name = outputName;
    outputLayerParams.type = "Identity";
    int dtype = CV_32F;  // FIXIT remove
    int outputLayerId = addLayer(outputLayerParams.name, outputLayerParams.type, dtype, outputLayerParams);
    connect(layerId, outputPort, outputLayerId, 0);
    CV_LOG_DEBUG(NULL, "DNN: register output='" << outputName << "' id=" << outputLayerId << " defined as " << layerId << ":" << outputPort);
    outputNameToId.insert(std::make_pair(outputName, outputLayerId));
    return outputLayerId;
}


void Net::Impl::allocateLayer(int lid, const LayersShapesMap& layersShapes)
{
    CV_TRACE_FUNCTION();

    LayerData& ld = layers[lid];

    // already allocated
    if (ld.flag)
        return;

    size_t ninputs = ld.inputBlobsId.size();
#if 0
    printf("layer %s:", ld.name.c_str());
    for (size_t i = 0; i < ninputs; i++)
    {
        int inp_lid = ld.inputBlobsId[i].lid;
        LayerData &inp_ld = layers[inp_lid];
        int inp_outputs = (int)inp_ld.outputBlobs.size();
        std::cout << " " << inp_ld.name << "(" << inp_outputs;

        for( int j = 0; j < inp_outputs; j++ )
        {
            std::cout << (j == 0 ? ": " : ", ") << inp_ld.outputBlobs[j].size;
        }
        std::cout << ")";
    }
    printf("\n");
#endif

    // determine parent layers
    for (size_t i = 0; i < ninputs; i++)
        ld.inputLayersId.insert(ld.inputBlobsId[i].lid);

    // allocate parents
    for (std::set<int>::const_iterator i = ld.inputLayersId.begin(); i != ld.inputLayersId.end(); i++)
        allocateLayer(*i, layersShapes);

    // bind inputs
    if (ld.id == 0)  // DataLayer
    {
        ninputs = netInputLayer->inputsData.size();
        ld.inputBlobsWrappers.resize(ninputs);
        for (size_t i = 0; i < ninputs; i++)
            ld.inputBlobsWrappers[i] = wrap(netInputLayer->inputsData[i]);
    }
    else
    {
        ld.inputBlobs.resize(ninputs);
        ld.inputBlobsWrappers.resize(ninputs);
        for (size_t i = 0; i < ninputs; i++)
        {
            LayerPin from = ld.inputBlobsId[i];
            CV_Assert(from.valid());
            CV_DbgAssert(layers.count(from.lid) && (int)layers[from.lid].outputBlobs.size() > from.oid);
            ld.inputBlobs[i] = &layers[from.lid].outputBlobs[from.oid];
            ld.inputBlobsWrappers[i] = layers[from.lid].outputBlobsWrappers[from.oid];
        }
    }

    LayersShapesMap::const_iterator layerShapesIt = layersShapes.find(lid);

    CV_Assert(layerShapesIt != layersShapes.end());

    if (preferableBackend == DNN_BACKEND_OPENCV && preferableTarget == DNN_TARGET_OPENCL_FP16 && ld.dtype == CV_32F)
        ld.dtype = CV_16F;

    std::vector<LayerPin> pinsForInternalBlobs;
    blobManager.allocateBlobsForLayer(ld, layerShapesIt->second, pinsForInternalBlobs);
    ld.outputBlobsWrappers.resize(ld.outputBlobs.size());
    for (int i = 0; i < ld.outputBlobs.size(); ++i)
        ld.outputBlobsWrappers[i] = wrap(ld.outputBlobs[i]);

    /* CUDA & CANN backend has its own system for internal blobs; we don't need these */
    ld.internalBlobsWrappers.resize((preferableBackend == DNN_BACKEND_CUDA || preferableBackend == DNN_BACKEND_TIMVX || preferableBackend == DNN_BACKEND_CANN) ? 0 : ld.internals.size());
    for (int i = 0; i < ld.internalBlobsWrappers.size(); ++i)
        ld.internalBlobsWrappers[i] = wrap(ld.internals[i]);

    Ptr<Layer> layerPtr = getLayerInstance(ld);
    {
        std::vector<Mat> inps(ld.inputBlobs.size());
        for (int i = 0; i < ld.inputBlobs.size(); ++i)
        {
            inps[i] = *ld.inputBlobs[i];
        }
        layerPtr->finalize(inps, ld.outputBlobs);
        layerPtr->preferableTarget = preferableTarget;
#if 0
        std::cout << "\toutputs:";
        size_t noutputs = ld.outputBlobs.size();
        for (size_t j = 0; j < noutputs; j++)
        {
            std::cout << (j == 0 ? " " : ", ") << ld.outputBlobs[j].size;
        }
        std::cout << "\n";
#endif
    }

    // After allocation of layer, we decrease counters to it's input blobs.
    blobManager.releaseReferences(ld.inputBlobsId);
    blobManager.releaseReferences(pinsForInternalBlobs);

    ld.flag = 1;
}


void Net::Impl::allocateLayers(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();

    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
        it->second.flag = 0;

    CV_Assert(!layers[0].outputBlobs.empty());
    ShapesVec inputShapes;
    for (int i = 0; i < layers[0].outputBlobs.size(); i++)
    {
        Mat& inp = layers[0].outputBlobs[i];
        CV_Assert(inp.total());
        if (preferableBackend == DNN_BACKEND_OPENCV &&
            preferableTarget == DNN_TARGET_OPENCL_FP16 &&
            layers[0].dtype == CV_32F)
        {
            layers[0].outputBlobs[i].create(inp.dims, inp.size, CV_16F);
        }
        inputShapes.push_back(shape(inp));
    }
    LayersShapesMap layersShapes;
    getLayersShapes(inputShapes, layersShapes);

    blobManager.reset();
    backendWrappers.clear();

    for (auto& layer : layers)
    {
        auto& ld = layer.second;
        ld.inputBlobsWrappers.clear();
        ld.outputBlobsWrappers.clear();
        ld.internalBlobsWrappers.clear();
    }

    // Fake references to input blobs.
    for (int i = 0; i < layers[0].outputBlobs.size(); ++i)
        blobManager.addReference(LayerPin(0, i));
    for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); ++it)
    {
        const LayerData& ld = it->second;
        blobManager.addReferences(ld.inputBlobsId);
    }

    for (int i = 0; i < blobsToKeep_.size(); i++)
    {
        blobManager.addReference(blobsToKeep_[i]);
    }

    for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); it++)
    {
        int lid = it->first;
        allocateLayer(lid, layersShapes);
    }

    layersTimings.resize(lastLayerId + 1, 0);
    fuseLayers(blobsToKeep_);
}


void Net::Impl::forwardLayer(LayerData& ld)
{
    CV_TRACE_FUNCTION();

    Ptr<Layer> layer = ld.layerInstance;

    if (!ld.skip)
    {
        TickMeter tm;
        tm.start();

#ifndef HAVE_VULKAN
        std::map<int, Ptr<BackendNode>>::const_iterator it = ld.backendNodes.find(preferableBackend);
#else
        std::map<int, Ptr<BackendNode>>::iterator it = ld.backendNodes.find(preferableBackend);
#endif
        if (preferableBackend == DNN_BACKEND_OPENCV || it == ld.backendNodes.end() || it->second.empty())
        {
            if (isAsync)
                CV_Error(Error::StsNotImplemented, "Default implementation fallbacks in asynchronous mode");

            if (!layer->supportBackend(DNN_BACKEND_OPENCV))
                CV_Error(Error::StsNotImplemented, format("Layer \"%s\" of type \"%s\" unsupported on OpenCV backend",
                                                   ld.name.c_str(), ld.type.c_str()));

#ifdef HAVE_OPENCL
            if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
            {
                std::vector<UMat> umat_inputBlobs = OpenCLBackendWrapper::getUMatVector(ld.inputBlobsWrappers);
                std::vector<UMat> umat_outputBlobs = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
                std::vector<UMat> umat_internalBlobs = OpenCLBackendWrapper::getUMatVector(ld.internalBlobsWrappers);
                layer->forward(umat_inputBlobs,
                               umat_outputBlobs,
                               umat_internalBlobs);
                if (getParam_DNN_CHECK_NAN_INF())
                {
                    bool fail = false;
                    for (size_t i = 0; i < umat_outputBlobs.size(); ++i)
                    {
                        UMat& u = umat_outputBlobs[i];
                        Mat m;
                        if (u.depth() == CV_16F)  // FP16
                            u.convertTo(m, CV_32F);
                        else
                            m = u.getMat(ACCESS_READ);
                        if (!checkRange(m))
                        {
                            CV_LOG_WARNING(NULL, "NaN detected in layer output: id=" << ld.id << " name=" << layer->name
                                           << " output id=" << i << " output shape=" << shape(m));
                            fail = true;
                        }
                        else if (!checkRange(m, true, NULL, -1e6, 1e6))
                        {
                            CV_LOG_WARNING(NULL, "Inf detected in layer output: id=" << ld.id << " name=" << layer->name
                                           << " output id=" << i << " output shape=" << shape(m));
                            fail = true;
                        }
                    }
                    if (fail)
                    {
                        for (size_t i = 0; i < umat_inputBlobs.size(); ++i)
                        {
                            UMat& u = umat_inputBlobs[i];
                            Mat m;
                            if (u.depth() == CV_16F)  // FP16
                                u.convertTo(m, CV_32F);
                            else
                                m = u.getMat(ACCESS_READ);
                            std::cout << "INPUT " << i << " " << cv::typeToString(u.type()) << " " << shape(m) << std::endl;
                            if (getParam_DNN_CHECK_NAN_INF_DUMP()) std::cout << m.reshape(1, 1) << std::endl;
                        }
                        for (size_t i = 0; i < umat_outputBlobs.size(); ++i)
                        {
                            UMat& u = umat_outputBlobs[i];
                            Mat m;
                            if (u.depth() == CV_16F)  // FP16
                                u.convertTo(m, CV_32F);
                            else
                                m = u.getMat(ACCESS_READ);
                            std::cout << "OUTPUT " << i << " " << cv::typeToString(u.type()) << " " << shape(m) << std::endl;
                            if (getParam_DNN_CHECK_NAN_INF_DUMP()) std::cout << m.reshape(1, 1) << std::endl;
                        }
                        for (size_t i = 0; i < umat_internalBlobs.size(); ++i)
                        {
                            UMat& u = umat_internalBlobs[i];
                            Mat m;
                            if (u.depth() == CV_16F)  // FP16
                                u.convertTo(m, CV_32F);
                            else
                                m = u.getMat(ACCESS_READ);
                            std::cout << "INTERNAL " << i << " " << shape(m) << std::endl;
                            if (getParam_DNN_CHECK_NAN_INF_DUMP()) std::cout << cv::typeToString(u.type()) << " " << m.reshape(1, 1) << std::endl;
                        }
                        if (getParam_DNN_CHECK_NAN_INF_RAISE_ERROR())
                            CV_Assert(!fail);
                    }
                }
                OpenCLBackendWrapper::update(ld.outputBlobsWrappers, umat_outputBlobs);
            }
            else
#endif
            {
                for (int i = 0, n = ld.inputBlobsWrappers.size(); i < n; ++i)
                {
                    if (!ld.inputBlobsWrappers[i].empty())
                        ld.inputBlobsWrappers[i]->copyToHost();
                }

                std::vector<Mat> inps(ld.inputBlobs.size());
                for (int i = 0; i < ld.inputBlobs.size(); ++i)
                {
                    inps[i] = *ld.inputBlobs[i];
                }
                layer->forward(inps, ld.outputBlobs, ld.internals);

                if (getParam_DNN_CHECK_NAN_INF())
                {
                    bool fail = false;
                    for (size_t i = 0; i < ld.outputBlobs.size(); ++i)
                    {
                        const Mat& m = ld.outputBlobs[i];
                        if (!checkRange(m))
                        {
                            CV_LOG_WARNING(NULL, "NaN detected in layer output: "
                                << cv::format("id=%d name=%s output id=%zu output shape=", ld.id, layer->name.c_str(), i) << shape(m));
                            fail = true;
                        }
                        else if (!checkRange(m, true, NULL, -1e6, 1e6))
                        {
                            CV_LOG_WARNING(NULL, "Inf detected in layer output: "
                                << cv::format("id=%d name=%s output id=%zu output shape=", ld.id, layer->name.c_str(), i) << shape(m));
                            fail = true;
                        }
                    }
                    if (fail)
                    {
                        for (size_t i = 0; i < ld.inputBlobs.size(); ++i)
                        {
                            const Mat* pM = ld.inputBlobs[i];
                            if (!pM)
                            {
                                std::cout << "INPUT " << i << " is NULL" << std::endl;
                                continue;
                            }
                            const Mat& m = *pM;
                            std::cout << "INPUT " << i << " " << cv::typeToString(m.type()) << " " << shape(m) << std::endl;
                            if (getParam_DNN_CHECK_NAN_INF_DUMP()) std::cout << m.reshape(1, 1) << std::endl;
                        }
                        for (size_t i = 0; i < ld.outputBlobs.size(); ++i)
                        {
                            const Mat& m = ld.outputBlobs[i];
                            std::cout << "OUTPUT " << i << " " << cv::typeToString(m.type()) << " " << shape(m) << std::endl;
                            if (getParam_DNN_CHECK_NAN_INF_DUMP()) std::cout << m.reshape(1, 1) << std::endl;
                        }
                        for (size_t i = 0; i < ld.internals.size(); ++i)
                        {
                            const Mat& m = ld.internals[i];
                            std::cout << "INTERNAL " << i << " " << cv::typeToString(m.type()) << " " << shape(m) << std::endl;
                            if (getParam_DNN_CHECK_NAN_INF_DUMP()) std::cout << m.reshape(1, 1) << std::endl;
                        }
                        if (getParam_DNN_CHECK_NAN_INF_RAISE_ERROR())
                            CV_Assert(!fail);
                    }
                }

                for (int i = 0, n = ld.outputBlobsWrappers.size(); i < n; ++i)
                {
                    if (!ld.outputBlobsWrappers[i].empty())
                        ld.outputBlobsWrappers[i]->setHostDirty();
                }
            }
        }
        else
        {
            Ptr<BackendNode> node = it->second;
            CV_Assert(!node.empty());
            if (preferableBackend == DNN_BACKEND_CUDA)
            {
                CV_Assert(haveCUDA());

#ifdef HAVE_CUDA
                Ptr<CUDABackendNode> cudaNode = node.dynamicCast<CUDABackendNode>();
                CV_Assert(!cudaNode.empty());

                cudaNode->forward(ld.inputBlobsWrappers, ld.outputBlobsWrappers, cudaInfo->workspace);

                for (auto id : ld.cudaD2HBackgroundTransfers)
                {
                    auto wrapper = ld.outputBlobsWrappers[id].dynamicCast<CUDABackendWrapper>();
                    wrapper->copyToHostInBackground();
                }
#endif
            }
            else if (preferableBackend == DNN_BACKEND_HALIDE)
            {
                forwardHalide(ld.outputBlobsWrappers, node);
            }
            else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            {
                CV_Assert(preferableBackend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && "Inheritance internal error");
            }
            else if (preferableBackend == DNN_BACKEND_WEBNN)
            {
                forwardWebnn(ld.outputBlobsWrappers, node, isAsync);
            }
            else if (preferableBackend == DNN_BACKEND_TIMVX)
            {
                forwardTimVX(ld.outputBlobsWrappers, node);
            }
#ifdef HAVE_VULKAN
            else if (preferableBackend == DNN_BACKEND_VKCOM)
            {
                try
                {
                    forwardVkCom(ld.outputBlobsWrappers, node);
                }
                catch (const cv::Exception& e)
                {
                    CV_LOG_ERROR(NULL, "forwardVkCom failed, fallback to CPU implementation. " << e.what());
                    it->second = Ptr<BackendNode>();
                    forwardLayer(ld);
                }
            }
#endif
            else
            {
                CV_Error(Error::StsNotImplemented, cv::format("Unknown backend identifier: %d", preferableBackend));
            }
        }

        tm.stop();
        int64 t = tm.getTimeTicks();
        layersTimings[ld.id] = (t > 0) ? t : t + 1;  // zero for skipped layers only
    }
    else
    {
        layersTimings[ld.id] = 0;
    }

    ld.flag = 1;
}


void Net::Impl::forwardToLayer(LayerData& ld, bool clearFlags)
{
    CV_TRACE_FUNCTION();

    if (clearFlags)
    {
        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
            it->second.flag = 0;
    }

    // already was forwarded
    if (ld.flag)
        return;

    // forward parents
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end() && (it->second.id < ld.id); ++it)
    {
        LayerData& ld = it->second;
        if (ld.flag)
            continue;
        forwardLayer(ld);
    }

    // forward itself
    forwardLayer(ld);

#ifdef HAVE_CUDA
    if (preferableBackend == DNN_BACKEND_CUDA)
        cudaInfo->context.stream.synchronize();
#endif
}


Mat Net::Impl::forward(const String& outputName)
{
    CV_Assert(!empty());
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    String layerName = outputName;

    if (layerName.empty())
    {
        std::vector<String> layerNames = getLayerNames();
        CV_Assert(!layerNames.empty());
        layerName = layerNames.back();
    }

    std::vector<LayerPin> pins(1, getPinByAlias(layerName));
    setUpNet(pins);
    forwardToLayer(getLayerData(layerName));

    return getBlob(layerName);
}


AsyncArray Net::Impl::forwardAsync(const String& outputName)
{
    CV_Assert(!empty());
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    String layerName = outputName;

    if (layerName.empty())
    {
        std::vector<String> layerNames = getLayerNames();
        CV_Assert(!layerNames.empty());
        layerName = layerNames.back();
    }

    std::vector<LayerPin> pins(1, getPinByAlias(layerName));
    setUpNet(pins);

    if (preferableBackend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        CV_Error(Error::StsNotImplemented, "DNN: Asynchronous forward is supported for Inference Engine backend only");

    isAsync = true;
    forwardToLayer(getLayerData(layerName));
    isAsync = false;

    return getBlobAsync(layerName);
}


void Net::Impl::forward(OutputArrayOfArrays outputBlobs, const String& outputName)
{
    CV_Assert(!empty());
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    String layerName = outputName;

    if (layerName.empty())
    {
        std::vector<String> layerNames = getLayerNames();
        CV_Assert(!layerNames.empty());
        layerName = layerNames.back();
    }

    std::vector<LayerPin> pins(1, getPinByAlias(layerName));
    setUpNet(pins);
    forwardToLayer(getLayerData(layerName));

    LayerPin pin = getPinByAlias(layerName);
    LayerData& ld = layers[pin.lid];

    if (outputBlobs.isUMat())
    {
        getBlob(layerName).copyTo(outputBlobs);
    }
    else if (outputBlobs.isMat())
    {
        outputBlobs.assign(getBlob(layerName));
    }
    else if (outputBlobs.isMatVector())
    {
        // The DNN_TARGET_CPU and DNN_TARGET_CPU_FP16 both use the CPU memory, do not need the copyToHost.
        if (preferableTarget != DNN_TARGET_CPU && preferableTarget != DNN_TARGET_CPU_FP16)
        {
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                CV_Assert(!ld.outputBlobsWrappers[i].empty());
                ld.outputBlobsWrappers[i]->copyToHost();
            }
        }
        if (ld.outputBlobs[0].depth() == CV_16F)
        {
            std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();
            outputvec.resize(ld.outputBlobs.size());
            for (int i = 0; i < outputvec.size(); i++)
                ld.outputBlobs[i].convertTo(outputvec[i], CV_32F);
        }
        else
        {
            // Output depth can be CV_32F or CV_8S
            std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();
            outputvec = ld.outputBlobs;
        }
    }
    else if (outputBlobs.isUMatVector())
    {
        std::vector<UMat>& outputvec = *(std::vector<UMat>*)outputBlobs.getObj();

#ifdef HAVE_OPENCL
        if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
        {
            if (preferableTarget == DNN_TARGET_OPENCL)
                outputvec = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
            else if (preferableTarget == DNN_TARGET_OPENCL_FP16)
            {
                std::vector<UMat> out_vec = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
                outputvec.resize(out_vec.size());
                for (int i = 0; i < out_vec.size(); i++)
                    out_vec[i].convertTo(outputvec[i], CV_32F);
            }
        }
        else
#endif
        {
            outputvec.resize(ld.outputBlobs.size());
            for (int i = 0; i < outputvec.size(); ++i)
                ld.outputBlobs[i].copyTo(outputvec[i]);
        }
    }
}


void Net::Impl::forward(OutputArrayOfArrays outputBlobs,
        const std::vector<String>& outBlobNames)
{
    CV_Assert(!empty());
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    std::vector<LayerPin> pins;
    for (int i = 0; i < outBlobNames.size(); i++)
    {
        pins.push_back(getPinByAlias(outBlobNames[i]));
    }

    setUpNet(pins);

    LayerPin out = getLatestLayerPin(pins);

    forwardToLayer(getLayerData(out.lid));

    std::vector<Mat> matvec;
    for (int i = 0; i < pins.size(); i++)
    {
        matvec.push_back(getBlob(pins[i]));
    }

    outputBlobs.create((int)matvec.size(), 1, CV_32F/*FIXIT*/, -1);  // allocate vector
    outputBlobs.assign(matvec);
}


void Net::Impl::forward(std::vector<std::vector<Mat>>& outputBlobs,
        const std::vector<String>& outBlobNames)
{
    CV_Assert(!empty());
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    std::vector<LayerPin> pins;
    for (int i = 0; i < outBlobNames.size(); i++)
    {
        pins.push_back(getPinByAlias(outBlobNames[i]));
    }

    setUpNet(pins);

    LayerPin out = getLatestLayerPin(pins);

    forwardToLayer(getLayerData(out.lid));

    outputBlobs.resize(outBlobNames.size());
    for (int i = 0; i < outBlobNames.size(); i++)
    {
        std::vector<LayerPin> lp = getLayerOutPins(outBlobNames[i]);
        outputBlobs[i].resize(lp.size());
        for (int j = 0; j < lp.size(); j++)
        {
            outputBlobs[i][j] = getBlob(lp[j]);
        }
    }
}


void Net::Impl::getLayerShapesRecursively(int id, LayersShapesMap& inOutShapes)
{
    CV_CheckGE(id, 0, "");
    CV_CheckLT(id, (int)layers.size(), "");
    LayerData& layerData = layers[id];
    std::vector<LayerPin>& inputLayerIds = layerData.inputBlobsId;
    LayerShapes& layerShapes = inOutShapes[id];

    if (id == 0 && layerShapes.in[0].empty())
    {
        if (!layerData.outputBlobs.empty())
        {
            ShapesVec shapes;
            for (int i = 0; i < layerData.outputBlobs.size(); i++)
            {
                Mat& inp = layerData.outputBlobs[i];
                CV_Assert(!inp.empty());
                shapes.push_back(shape(inp));
            }
            layerShapes.in = shapes;
        }
        else
        {
            const std::vector<MatShape>& inputShapes = netInputLayer->shapes;
            bool none = true;
            for (size_t i = 0; i < inputShapes.size(); i++)
            {
                if (!inputShapes[i].empty())
                {
                    none = false;
                    break;
                }
            }
            if (none)
            {
                layerShapes.out.clear();
                return;
            }
            else
            {
                layerShapes.in = inputShapes;
            }
        }
    }

    if (layerShapes.in.empty())
    {
        for (int i = 0; i < inputLayerIds.size(); i++)
        {
            int layerId = inputLayerIds[i].lid;
            LayersShapesMap::const_iterator it = inOutShapes.find(layerId);
            if (it == inOutShapes.end() || it->second.out.empty())
            {
                getLayerShapesRecursively(layerId, inOutShapes);
                it = inOutShapes.find(layerId);
                CV_Assert(it != inOutShapes.end());
            }
            const int out_port = inputLayerIds[i].oid;
            CV_CheckLT(out_port, (int)it->second.out.size(), "");
            const MatShape& shape = it->second.out[out_port];
            layerShapes.in.push_back(shape);
        }
    }
    const ShapesVec& is = layerShapes.in;
    ShapesVec& os = layerShapes.out;
    ShapesVec& ints = layerShapes.internal;
    int requiredOutputs = layerData.requiredOutputs.size();
    const Ptr<Layer>& l = getLayerInstance(layerData);
    CV_Assert(l);
    bool layerSupportInPlace = false;
    try
    {
        layerSupportInPlace = l->getMemoryShapes(is, requiredOutputs, os, ints);
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "OPENCV/DNN: [" << l->type << "]:(" << l->name << "): getMemoryShapes() throws exception." <<
                " inputs=" << is.size() <<
                " outputs=" << os.size() << "/" << requiredOutputs <<
                " blobs=" << l->blobs.size());
        for (size_t i = 0; i < is.size(); ++i)
        {
            CV_LOG_ERROR(NULL, "    input[" << i << "] = " << toString(is[i]));
        }
        for (size_t i = 0; i < os.size(); ++i)
        {
            CV_LOG_ERROR(NULL, "    output[" << i << "] = " << toString(os[i]));
        }
        for (size_t i = 0; i < l->blobs.size(); ++i)
        {
            CV_LOG_ERROR(NULL, "    blobs[" << i << "] = " << typeToString(l->blobs[i].type()) << " " << toString(shape(l->blobs[i])));
        }
        CV_LOG_ERROR(NULL, "Exception message: " << e.what());
        throw;
    }
    layerShapes.supportInPlace = layerSupportInPlace;

    try
    {
        for (int i = 0; i < ints.size(); i++)
            CV_CheckGT(total(ints[i]), 0, "");

        for (int i = 0; i < os.size(); i++)
            CV_CheckGT(total(os[i]), 0, "");
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "OPENCV/DNN: [" << l->type << "]:(" << l->name << "): getMemoryShapes() post validation failed." <<
                " inputs=" << is.size() <<
                " outputs=" << os.size() << "/" << requiredOutputs <<
                " blobs=" << l->blobs.size() <<
                " inplace=" << layerSupportInPlace);
        for (size_t i = 0; i < is.size(); ++i)
        {
            CV_LOG_ERROR(NULL, "    input[" << i << "] = " << toString(is[i]));
        }
        for (size_t i = 0; i < os.size(); ++i)
        {
            CV_LOG_ERROR(NULL, "    output[" << i << "] = " << toString(os[i]));
        }
        for (size_t i = 0; i < l->blobs.size(); ++i)
        {
            CV_LOG_ERROR(NULL, "    blobs[" << i << "] = " << typeToString(l->blobs[i].type()) << " " << toString(shape(l->blobs[i])));
        }
        CV_LOG_ERROR(NULL, "Exception message: " << e.what());
        throw;
    }
}

void Net::Impl::getLayersShapes(
        const ShapesVec& netInputShapes,
        std::vector<int>& layersIds,
        std::vector<ShapesVec>& inLayersShapes,
        std::vector<ShapesVec>& outLayersShapes) /*const*/
{
    layersIds.clear();
    inLayersShapes.clear();
    outLayersShapes.clear();

    Impl::LayersShapesMap inOutShapes;
    getLayersShapes(netInputShapes, inOutShapes);

    for (Impl::LayersShapesMap::const_iterator it = inOutShapes.begin();
            it != inOutShapes.end(); it++)
    {
        layersIds.push_back(it->first);
        inLayersShapes.push_back(it->second.in);
        outLayersShapes.push_back(it->second.out);
    }
}


void Net::Impl::getLayersShapes(const ShapesVec& netInputShapes,
        LayersShapesMap& inOutShapes)
{
    inOutShapes.clear();

    inOutShapes[0].in = netInputShapes;  // insert shape for first input layer
    for (MapIdToLayerData::const_iterator it = layers.begin();
            it != layers.end(); it++)
    {
        getLayerShapesRecursively(it->first, inOutShapes);
    }
}

void Net::Impl::getLayerShapes(const ShapesVec& netInputShapes,
        const int layerId,
        LayerShapes& shapes)
{
    LayersShapesMap inOutShapes;
    inOutShapes[0].in = netInputShapes;  // insert shape for first input layer
    getLayerShapesRecursively(layerId, inOutShapes);
    shapes = inOutShapes[layerId];
}

void Net::Impl::updateLayersShapes()
{
    CV_LOG_DEBUG(NULL, "updateLayersShapes() with layers.size=" << layers.size());
    CV_Assert(netInputLayer);
    DataLayer& inputLayer = *netInputLayer;
    LayerData& inputLayerData = layers[0];
    CV_Assert(inputLayerData.layerInstance.get() == &inputLayer);
    CV_Assert(!inputLayerData.outputBlobs.empty());
    ShapesVec inputShapes;
    for (int i = 0; i < inputLayerData.outputBlobs.size(); i++)
    {
        Mat& inp = inputLayerData.outputBlobs[i];
        CV_Assert(!inp.empty());
        if (preferableBackend == DNN_BACKEND_OPENCV &&  // FIXIT: wrong place for output allocation
            preferableTarget == DNN_TARGET_OPENCL_FP16 &&
            inputLayerData.dtype == CV_32F)
        {
            inp.create(inp.dims, inp.size, CV_16F);
        }
        inputShapes.push_back(shape(inp));
    }
    CV_LOG_DEBUG(NULL, toString(inputShapes, "Network input shapes"));
    LayersShapesMap layersShapes;
    layersShapes[0].in = inputShapes;
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
    {
        int layerId = it->first;
        LayerData& layerData = it->second;
        const std::vector<LayerPin>& inputLayerIds = layerData.inputBlobsId;
        LayerShapes& layerShapes = layersShapes[layerId];
        CV_LOG_DEBUG(NULL, "layer " << layerId << ": [" << layerData.type << "]:(" << layerData.name << ") with inputs.size=" << inputLayerIds.size());
        if (layerShapes.in.empty())
        {
            for (int i = 0; i < inputLayerIds.size(); i++)
            {
                const LayerPin& inputPin = inputLayerIds[i];
                int inputLayerId = inputPin.lid;
                CV_LOG_DEBUG(NULL, "    input[" << i << "] " << inputLayerId << ":" << inputPin.oid << " as [" << layers[inputLayerId].type << "]:(" << layers[inputLayerId].name << ")");
                LayersShapesMap::const_iterator inputIt = layersShapes.find(inputLayerId);
                if (inputIt == layersShapes.end() || inputIt->second.out.empty())
                {
                    getLayerShapesRecursively(inputLayerId, layersShapes);
                }
                const MatShape& shape = layersShapes[inputLayerId].out[inputPin.oid];
                layerShapes.in.push_back(shape);
            }
            getLayerInstance(layerData)->updateMemoryShapes(layerShapes.in);
        }
        CV_LOG_DEBUG(NULL, "Layer " << layerId << ": " << toString(layerShapes.in, "input shapes"));
        CV_LOG_IF_DEBUG(NULL, !layerShapes.out.empty(), "Layer " << layerId << ": " << toString(layerShapes.out, "output shapes"));
        CV_LOG_IF_DEBUG(NULL, !layerShapes.internal.empty(), "Layer " << layerId << ": " << toString(layerShapes.internal, "internal shapes"));
    }
    CV_LOG_DEBUG(NULL, "updateLayersShapes() - DONE");
}


LayerPin Net::Impl::getLatestLayerPin(const std::vector<LayerPin>& pins) const
{
    return *std::max_element(pins.begin(), pins.end());
}

Mat Net::Impl::getBlob(const LayerPin& pin) const
{
    CV_TRACE_FUNCTION();

    if (!pin.valid())
        CV_Error(Error::StsObjectNotFound, "Requested blob not found");

    MapIdToLayerData::const_iterator it = layers.find(pin.lid);
    if (it == layers.end())
        CV_Error_(Error::StsOutOfRange, ("Layer #%d is not valid (output #%d requested)", pin.lid, pin.oid));

    const LayerData& ld = it->second;
    if ((size_t)pin.oid >= ld.outputBlobs.size())
    {
        CV_Error(Error::StsOutOfRange, format("Layer \"%s\" produce only %zu outputs, "
                                              "the #%d was requested",
                                               ld.name.c_str(), ld.outputBlobs.size(), pin.oid));
    }
    if (preferableTarget != DNN_TARGET_CPU && preferableTarget != DNN_TARGET_CPU_FP16)
    {
        CV_Assert(!ld.outputBlobsWrappers.empty() && !ld.outputBlobsWrappers[pin.oid].empty());
        // Transfer data to CPU if it's require.
        ld.outputBlobsWrappers[pin.oid]->copyToHost();
    }

    if (ld.outputBlobs[pin.oid].depth() == CV_16F)
    {
        Mat output_blob;
        ld.outputBlobs[pin.oid].convertTo(output_blob, CV_32F);
        return output_blob;
    }
    else
        return ld.outputBlobs[pin.oid];
}

Mat Net::Impl::getBlob(String outputName) const
{
    return getBlob(getPinByAlias(outputName));
}


AsyncArray Net::Impl::getBlobAsync(const LayerPin& pin)
{
    CV_TRACE_FUNCTION();
    CV_Error(Error::StsNotImplemented, "DNN: OpenVINO/nGraph backend is required");
}


AsyncArray Net::Impl::getBlobAsync(String outputName)
{
    return getBlobAsync(getPinByAlias(outputName));
}


void Net::Impl::setInputsNames(const std::vector<String>& inputBlobNames)
{
    CV_Assert(netInputLayer);
    netInputLayer->setNames(inputBlobNames);
}


void Net::Impl::setInputShape(const String& inputName, const MatShape& shape)
{
    CV_Assert(netInputLayer);
    netInputLayer->setInputShape(inputName, shape);
}


void Net::Impl::setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean)
{
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    LayerPin pin;
    pin.lid = 0;
    pin.oid = resolvePinOutputName(getLayerData(pin.lid), name);

    if (!pin.valid())
        CV_Error(Error::StsObjectNotFound, "Requested blob \"" + name + "\" not found");

    Mat blob_ = blob.getMat();  // can't use InputArray directly due MatExpr stuff
    MatShape blobShape = shape(blob_);

#if 0  // TODO: DNNTestNetwork.MobileNet_SSD_Caffe_Different_Width_Height/0
    if (pin.lid == 0)
    {
        CV_Assert(!netInputLayer.empty());
        const DataLayer& netInputLayer = *(this->netInputLayer);
        if (!netInputLayer.shapes.empty())
        {
            CV_CheckLT(pin.oid, (int)netInputLayer.shapes.size(), "");
            const MatShape& inputShapeLimitation = netInputLayer.shapes[pin.oid];
            if (!inputShapeLimitation.empty())
            {
                CV_CheckEQ(inputShapeLimitation.size(), blobShape.size(), "");
                const size_t dims = inputShapeLimitation.size();
                for (size_t dim = 0; dim < dims; dim++)
                {
                    if (dims >= 3 && dim == 0 && inputShapeLimitation[0] == 1)
                        continue;  // don't limit batch
                    CV_CheckEQ(inputShapeLimitation[dim], blobShape[dim], "");
                }
            }
        }
    }
#endif

    LayerData& ld = layers[pin.lid];
    const int numInputs = std::max(pin.oid + 1, (int)ld.requiredOutputs.size());
    ld.outputBlobs.resize(numInputs);
    ld.outputBlobsWrappers.resize(numInputs);
    netInputLayer->inputsData.resize(numInputs);
    netInputLayer->scaleFactors.resize(numInputs);
    netInputLayer->means.resize(numInputs);

    MatShape prevShape = shape(netInputLayer->inputsData[pin.oid]);
    bool oldShape = prevShape == blobShape;

    blob_.copyTo(netInputLayer->inputsData[pin.oid]);
    if (!oldShape)
        ld.outputBlobs[pin.oid] = netInputLayer->inputsData[pin.oid];

    if (!ld.outputBlobsWrappers[pin.oid].empty())
    {
        ld.outputBlobsWrappers[pin.oid]->setHostDirty();
    }
    netInputLayer->scaleFactors[pin.oid] = scalefactor;
    netInputLayer->means[pin.oid] = mean;
    netWasAllocated = netWasAllocated && oldShape;
}


Mat Net::Impl::getParam(int layer, int numParam) const
{
    LayerData& ld = getLayerData(layer);
    std::vector<Mat>& layerBlobs = getLayerInstance(ld)->blobs;
    CV_Assert(numParam < (int)layerBlobs.size());
    return layerBlobs[numParam];
}

void Net::Impl::setParam(int layer, int numParam, const Mat& blob)
{
    LayerData& ld = getLayerData(layer);

    // FIXIT we should not modify "execution" instance
    std::vector<Mat>& layerBlobs = getLayerInstance(ld)->blobs;
    CV_Assert(numParam < (int)layerBlobs.size());
    // we don't make strong checks, use this function carefully
    layerBlobs[numParam] = blob;
}


static
string dumpLayerParameterSize(const string& name, const LayerParams& lp)
{
    std::ostringstream out(name, std::ios::ate);
    DictValue param = lp.get(name);
    switch (param.size())
    {
    case 1: out << " : "; break;
    case 2: out << " (HxW): "; break;
    case 3: out << " (DxHxW): "; break;
    default:
        CV_LOG_INFO(NULL, format("DNN/dumpLayerParameterSize(): Unsupported '%s' size = %d", name.c_str(), param.size()));
        out << ": ";
    }
    for (size_t i = 0; i < param.size(); i++)
    {
        if (i > 0)
            out << " x ";
        out << param.get<int>(i);
    }
    return out.str();
}

string Net::Impl::dump(bool forceAllocation) const
{
    bool hasInput = !netInputLayer->inputsData.empty();
    if (forceAllocation)
    {
        if (!netWasAllocated)
            const_cast<Net::Impl*>(this)->setUpNet();
    }

    std::ostringstream out;
    const std::map<int, LayerData>& map = layers;

    Backend prefBackend = (Backend)preferableBackend;
    std::vector<std::vector<int>> skippedLayers;
    std::vector<int> skipId;
    std::vector<int> allLayers(map.size(), -1);
    int idPrev = -1;
    Ptr<BackendNode> prevNode;
    for (std::map<int, LayerData>::const_reverse_iterator rit = map.rbegin(); rit != map.rend(); ++rit)
    {
        std::map<int, Ptr<BackendNode>>::const_iterator itBackend = rit->second.backendNodes.find(prefBackend);
        if (prefBackend == DNN_BACKEND_OPENCV || itBackend == rit->second.backendNodes.end() || itBackend->second.empty())
        {
            if (rit->second.skip)
                skipId.push_back(rit->first);
            else if (!skipId.empty())
            {
                if (prefBackend == DNN_BACKEND_OPENCV || prevNode.empty())
                    skipId.push_back(rit->first);
                else if (idPrev != -1)
                    skipId.push_back(idPrev);

                std::sort(skipId.begin(), skipId.end());
                for (int i = 0; i < skipId.size(); i++)
                {
                    allLayers[skipId[i]] = skippedLayers.size();
                }
                skippedLayers.push_back(skipId);
                skipId.clear();
            }
        }
        else
        {
            if (itBackend->second == prevNode)
            {
                if (idPrev != -1)
                    skipId.push_back(idPrev);
            }
            else if (!skipId.empty())
            {
                if (idPrev != -1)
                    skipId.push_back(idPrev);
                std::sort(skipId.begin(), skipId.end());
                for (int i = 0; i < skipId.size(); i++)
                {
                    allLayers[skipId[i]] = skippedLayers.size();
                }
                skippedLayers.push_back(skipId);
                skipId.clear();
            }
            idPrev = rit->first;
            prevNode = itBackend->second;
        }
    }
    std::vector<string> colors = { "#ffffb3", "#fccde5", "#8dd3c7", "#bebada", "#80b1d3", "#fdb462", "#ff4848", "#b35151", "#b266ff", "#b266ff", "#3cb371", "#ffcab3", "#aaa000"};
    string backend;
    switch (prefBackend)
    {
    case DNN_BACKEND_DEFAULT: backend = "DEFAULT/"; break;
    case DNN_BACKEND_HALIDE: backend = "HALIDE/"; break;
    case DNN_BACKEND_INFERENCE_ENGINE:  // fallthru
    case DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019:  // fallthru
    case DNN_BACKEND_INFERENCE_ENGINE_NGRAPH: backend = "OpenVINO/"; break;
    case DNN_BACKEND_OPENCV: backend = "OCV/"; break;
    case DNN_BACKEND_VKCOM: backend = "VULKAN/"; break;
    case DNN_BACKEND_CUDA: backend = "CUDA/"; break;
    case DNN_BACKEND_WEBNN: backend = "WEBNN/"; break;
    case DNN_BACKEND_TIMVX: backend = "TIMVX/"; break;
    case DNN_BACKEND_CANN: backend = "CANN/"; break;
    case DNN_BACKEND_TENSORRT: backend = "TENSORRT/"; break;
        // don't use default:
    }
    out << "digraph G {\n";
    // Add nodes
    for (std::map<int, LayerData>::const_iterator it = map.begin(); it != map.end(); ++it)
    {
        const LayerData& ld = it->second;
        string name = ld.params.name;
        std::vector<int> clusterIds(1, it->first);
        if (allLayers[it->first] == -1 && !name.empty())
        {
            out << "\t\"" << name << "\" [label=\"";
        }
        else if (name.empty() || it->first != skippedLayers[allLayers[it->first]][0])
        {
            continue;
        }
        else  // first node in cluster : it->first == skippedLayers[allLayers[it->first]][0]
        {
            int cluster = allLayers[it->first];
            out << "\t\""
                << "cluster_" << cluster << "\" [label=\"{";
            clusterIds = skippedLayers[allLayers[it->first]];  // vertices in current cluster
        }
        for (int i = 0; i < clusterIds.size(); i++)
        {
            CV_DbgAssert(map.find(clusterIds[i]) != map.end());
            const LayerParams& lp = map.find(clusterIds[i])->second.params;
            if (!lp.name.empty())
            {
                if (i > 0)
                {
                    out << " | ";
                }
                out << lp.name << "\\n"
                    << lp.type << "\\n";  // align center
                if (lp.has("kernel_size"))
                {
                    string kernel = dumpLayerParameterSize("kernel_size", lp);
                    out << kernel;
                    out << "\\l";  // align left
                }
                else if (lp.has("kernel_h") && lp.has("kernel_w"))
                {
                    DictValue h = lp.get("kernel_h");
                    DictValue w = lp.get("kernel_w");
                    out << "kernel (HxW): " << h << " x " << w;
                    out << "\\l";  // align left
                }
                if (lp.has("stride"))
                {
                    string stride = dumpLayerParameterSize("stride", lp);
                    out << stride;
                    out << "\\l";  // align left
                }
                else if (lp.has("stride_h") && lp.has("stride_w"))
                {
                    DictValue h = lp.get("stride_h");
                    DictValue w = lp.get("stride_w");
                    out << "stride (HxW): " << h << " x " << w;
                    out << "\\l";  // align left
                }
                if (lp.has("dilation"))
                {
                    string dilation = dumpLayerParameterSize("dilation", lp);
                    out << dilation;
                    out << "\\l";  // align left
                }
                else if (lp.has("dilation_h") && lp.has("dilation_w"))
                {
                    DictValue h = lp.get("dilation_h");
                    DictValue w = lp.get("dilation_w");
                    out << "dilation (HxW): " << h << " x " << w;
                    out << "\\l";  // align left
                }
                if (lp.has("pad"))
                {
                    DictValue pad = lp.get("pad");
                    out << "pad ";
                    switch (pad.size())
                    {
                    case 1: out << ": " << pad; break;
                    case 2:
                        out << "(HxW): (" << pad.get<int>(0) << " x " << pad.get<int>(1) << ")";
                        break;
                    case 4:
                        out << "(HxW): (" << pad.get<int>(0) << ", " << pad.get<int>(2)
                            << ") x (" << pad.get<int>(1) << ", " << pad.get<int>(3) << ")";
                        break;
                    case 6:
                        out << "(DxHxW): (" << pad.get<int>(0) << ", " << pad.get<int>(3)
                            << ") x (" << pad.get<int>(1) << ", " << pad.get<int>(4)
                            << ") x (" << pad.get<int>(2) << ", " << pad.get<int>(5) << ")";
                        break;
                    default: CV_Error(Error::StsNotImplemented, format("Unsupported pad size = %d", pad.size()));
                    }
                    out << "\\l";  // align left
                }
                else if (lp.has("pad_l") && lp.has("pad_t") && lp.has("pad_r") && lp.has("pad_b"))
                {
                    DictValue l = lp.get("pad_l");
                    DictValue t = lp.get("pad_t");
                    DictValue r = lp.get("pad_r");
                    DictValue b = lp.get("pad_b");
                    out << "pad (HxW): (" << t << ", " << b << ") x (" << l << ", " << r << ")";
                    out << "\\l";  // align left
                }
                else if (lp.has("pooled_w") || lp.has("pooled_h"))
                {
                    DictValue h = lp.get("pooled_h");
                    DictValue w = lp.get("pooled_w");
                    out << "pad pooled (HxW): " << h << " x " << w;
                    out << "\\l";  // align left
                }
                if (lp.has("pool"))
                {
                    out << "pool: " << lp.get("pool");
                    out << "\\l";  // align left
                }
                if (lp.has("global_pooling"))
                {
                    out << "global_pooling: " << lp.get("global_pooling");
                    out << "\\l";  // align left
                }
                if (lp.has("group"))
                {
                    out << "group: " << lp.get("group");
                    out << "\\l";  // align left
                }
            }
        }
        if (!ld.outputBlobs.empty())
        {
            out << "output: " << ld.outputBlobs[0].size;
            out << "\\l";  // align left
        }

        Ptr<BackendNode> layerBackend;
        std::map<int, Ptr<BackendNode>>::const_iterator ibn = ld.backendNodes.find(prefBackend);
        if (ibn != ld.backendNodes.end())
            layerBackend = ibn->second;
        out << (!layerBackend.empty() ? backend : "OCV/");
        int colorId = 0;
        const Target target = ld.layerInstance.empty()
                ? DNN_TARGET_CPU
                : (Target)(ld.layerInstance->preferableTarget);  // TODO fix preferableTarget type
        switch (target)
        {
        case DNN_TARGET_CPU:
            out << "CPU";
            colorId = layerBackend.empty() ? 0 : 5;
            break;
        case DNN_TARGET_OPENCL:
            out << "OCL";
            colorId = 1;
            break;
        case DNN_TARGET_OPENCL_FP16:
            out << "OCL_FP16";
            colorId = 2;
            break;
        case DNN_TARGET_MYRIAD:
            out << "MYRIAD";
            colorId = 3;
            break;
        case DNN_TARGET_HDDL:
            out << "HDDL";
            colorId = 8;
            break;
        case DNN_TARGET_VULKAN:
            out << "VULKAN";
            colorId = 7;
            break;
        case DNN_TARGET_FPGA:
            out << "FPGA";
            colorId = 4;
            break;
        case DNN_TARGET_CUDA:
            out << "CUDA";
            colorId = 5;
            break;
        case DNN_TARGET_CUDA_FP16:
            out << "CUDA_FP16";
            colorId = 6;
            break;
        case DNN_TARGET_NPU:
            out << "NPU";
            colorId = 9;
            break;
        case DNN_TARGET_CPU_FP16:
            out << "CPU_FP16";
            colorId = 10;
            break;
        case DNN_TARGET_TENSORRT:
            out << "TENSORRT";
            colorId = 11;
            break;
            // don't use default:
        }
        CV_Assert(colorId < colors.size());
        out << "\\n";  // align center
        out << ((clusterIds.size() == 1) ? "\" " : " }\" ");
        out << "fillcolor=\"" << colors[colorId] << "\" ";
        out << "style=filled ";
        out << "shape=" << ((clusterIds.size() == 1) ? "box" : "record") << "]\n";
    }
    out << '\n';
    // Add edges
    int inputsSize = hasInput ? netInputLayer->outNames.size() : 0;
    for (std::map<int, LayerData>::const_iterator it = map.begin(); it != map.end(); ++it)
    {
        const LayerData& ld = it->second;
        if (allLayers[it->first] == -1)  // node
        {
            for (int i = 0; i < ld.consumers.size(); i++)
            {
                int outId = ld.consumers[i].lid;
                if (it == map.begin() && inputsSize > 1)
                    out << "\t\"" << ld.name << "_" << i << "\""
                        << " -> ";
                else
                    out << "\t\"" << ld.name << "\""
                        << " -> ";
                if (allLayers[outId] == -1)  // node
                {
                    CV_DbgAssert(map.find(outId) != map.end());
                    out << "\"" << map.find(outId)->second.name << "\"\n";
                }
                else  // cluster
                {
                    out << "\""
                        << "cluster_" << allLayers[outId] << "\"\n";
                }
            }
        }
        else if (it->first == skippedLayers[allLayers[it->first]].back())  // edges from last layer in cluster
        {
            for (int i = 0; i < ld.consumers.size(); i++)
            {
                int outId = ld.consumers[i].lid;
                if (allLayers[outId] == -1)  // node
                {
                    CV_DbgAssert(map.find(outId) != map.end());
                    out << "\t\""
                        << "cluster_" << allLayers[it->first] << "\""
                        << " -> ";
                    out << "\"" << map.find(outId)->second.name << "\"\n";
                }
                else if (allLayers[outId] != allLayers[it->first])
                {  // another cluster
                    out << "\t\""
                        << "cluster_" << allLayers[it->first] << "\""
                        << " -> ";
                    out << "\""
                        << "cluster_" << allLayers[outId] << "\"\n";
                }
            }
        }
    }
    out << "}\n";
    return out.str();
}

static void dumpTensorToString(std::ostringstream &out, const Mat &m, const int num_indent_spaces = 4) {
    string indent_spaces(num_indent_spaces, ' ');

    int type = 1;
    /* Check TensorProto::DataType from https://github.com/onnx/onnx/blob/main/onnx/onnx.proto */
    switch (m.type()) {
        case CV_32F: break;
        case CV_8U:  type = 2; break;
        case CV_8S:  type = 3; break;
        case CV_16U: type = 4; break;
        case CV_16S: type = 5; break;
        case CV_32S: type = 6; break;
#if CV_VERSION_MAJOR > 4
        case CV_64S: type = 7; break;
        // STRING: 8
        case CV_BOOL: type = 9; break;
#endif
        case CV_16F: type = 10; break;
        case CV_64F: type = 11; break;
#if CV_VERSION_MAJOR > 4
        case CV_32U: type = 12; break;
        case CV_64U: type = 13; break;
        // COMPLEX64: 14
        // COMPLEX128: 15
        case CV_16BF: type = 16; break;
#endif
        default: CV_Error(Error::StsUnsupportedFormat, "Type of mat is not supported");
    }
    const auto &mshape = shape(m);

    out << indent_spaces << "type {\n"
        << indent_spaces << "  tensor_type {\n"
        << indent_spaces << "    elem_type: " << type << "\n";
    out << indent_spaces << "    shape {\n";
    for (size_t i = 0; i < mshape.size(); i++) {
        out << indent_spaces << format("      dim { dim_value: %d }\n", mshape[i]);
    }
    out << indent_spaces << "    }\n" // shape{}
        << indent_spaces << "  }\n" // tensor_type{}
        << indent_spaces << "}\n"; // type{}
}


static void dumpParamToString(std::ostringstream &out, const std::string &key, const DictValue &value, const int num_indent_spaces = 2) {
    std::string indent_spaces(num_indent_spaces, ' ');

    out << indent_spaces << "attribute {\n"
        << indent_spaces << format("  name: \"%s\"\n", key.c_str());
    if (value.size() == 1) {
        if (value.isString()) {
            out << indent_spaces << format("  type: STRING\n")
                << indent_spaces << format("  s: \"%s\"\n", value.getStringValue(0).c_str());
        } else if (value.isInt()) {
            out << indent_spaces << format("  type: INT\n")
                << indent_spaces << format("  i: %d\n", value.getIntValue(0));
        } else if (value.isReal()) {
            out << indent_spaces << format("  type: FLOAT\n")
                << indent_spaces << format("  f: %f\n", value.getRealValue(0));
        } else {
            out << indent_spaces << format("  type: UNKNOWN-SCALAR\n");
        }
    } else {
        if (value.isString()) {
            out << indent_spaces << format("  type: STRINGS\n");
        } else if (value.isInt()) {
            out << indent_spaces << format("  type: INTS\n");
        } else if (value.isReal()) {
            out << indent_spaces << format("  type: FLOATS\n");
        } else {
            out << indent_spaces << format("  type: UNKNOWN-ARRAY\n");
        }
        for (int i = 0; i < value.size(); i++) {
            if (value.isString()) {
                out << indent_spaces << format("  strings: \"%s\"\n", value.getStringValue(i).c_str());
            } else if (value.isInt()) {
                out << indent_spaces << format("  ints: %d\n", value.getIntValue(i));
            } else if (value.isReal()) {
                out << indent_spaces << format("  floats: %f\n", value.getRealValue());
            }
        }
    }
    out << indent_spaces << "}\n"; // attribute{}
}

static void dumpLayerToString(std::ostringstream &out,
                              const std::vector<std::string> &inputs,
                              const std::vector<std::string> &outputs,
                              const std::string &name,
                              const std::string &op_type,
                              const LayerParams &params,
                              const std::string &backend_name,
                              const std::string &target_name,
                              const int num_indent_spaces = 2) {
    std::string indent_spaces(num_indent_spaces, ' ');

    for (size_t i = 0; i < inputs.size(); i++) {
        out << indent_spaces << format("input: \"%s\"\n", inputs[i].c_str());
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        out << indent_spaces << format("output: \"%s\"\n", outputs[i].c_str());
    }
    if (!name.empty()) {
        out << indent_spaces << format("name: \"%s\"\n", name.c_str());
    }
    if (!op_type.empty()) {
        out << indent_spaces << format("op_type: \"%s\"\n", op_type.c_str());
    }
    if (!params.name.empty()) {
        for (auto param_iter = params.begin(); param_iter != params.end(); param_iter++) {
            auto key = param_iter->first;
            auto value = param_iter->second;
            dumpParamToString(out, key, value, num_indent_spaces);
        }
    }
    if (!backend_name.empty()) {
        DictValue dvb(backend_name);
        dumpParamToString(out, "Backend", dvb, num_indent_spaces);
    }
    if (!target_name.empty()) {
        DictValue dvt(target_name);
        dumpParamToString(out, "Target", dvt, num_indent_spaces);
    }
}

string Net::Impl::dumpToPbtxt(bool forceAllocation) const {
    if (forceAllocation && !netWasAllocated) {
        const_cast<Net::Impl*>(this)->setUpNet();
    }

    std::ostringstream out;
    const std::map<int, LayerData> &map = layers;
    std::map<String, Mat*> value_info;

    Backend prefBackend = (Backend)preferableBackend;
    Target prefTarget = (Target)preferableTarget;

    auto GetBackendName = [] (int backendId) {
        std::string backend = "Unknown";
        switch (backendId) {
            case DNN_BACKEND_DEFAULT:   backend = "DEFAULT"; break;
            #if CV_VERSION_MAJOR <= 4
            case DNN_BACKEND_HALIDE:    backend = "HALIDE"; break;
            #endif
            case DNN_BACKEND_INFERENCE_ENGINE:  // fallthru
            case DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019:  // fallthru
            case DNN_BACKEND_INFERENCE_ENGINE_NGRAPH: backend = "OpenVINO"; break;
            case DNN_BACKEND_OPENCV:    backend = "OCV"; break;
            case DNN_BACKEND_VKCOM:     backend = "VULKAN"; break;
            case DNN_BACKEND_CUDA:      backend = "CUDA"; break;
            case DNN_BACKEND_WEBNN:     backend = "WEBNN"; break;
            case DNN_BACKEND_TIMVX:     backend = "TIMVX"; break;
            case DNN_BACKEND_CANN:      backend = "CANN"; break;
            case DNN_BACKEND_TENSORRT:      backend = "TENSORRT"; break;
        }
        return backend;
    };
    auto GetTargetName = [] (int targetId) {
        std::string target = "Unknown";
        switch (targetId) {
            case DNN_TARGET_CPU:         target = "CPU"; break;
            case DNN_TARGET_OPENCL:      target = "OCL"; break;
            case DNN_TARGET_OPENCL_FP16: target = "OCL_FP16"; break;
            case DNN_TARGET_MYRIAD:      target = "MYRIAD"; break;
            case DNN_TARGET_VULKAN:      target = "VULKAN"; break;
            case DNN_TARGET_FPGA:        target = "FPGA"; break;
            case DNN_TARGET_CUDA:        target = "CUDA"; break;
            case DNN_TARGET_CUDA_FP16:   target = "CUDA_FP16"; break;
            case DNN_TARGET_HDDL:        target = "HDDL"; break;
            case DNN_TARGET_NPU:         target = "NPU"; break;
            case DNN_TARGET_CPU_FP16:    target = "CPU_FP16"; break;
        }
        return target;
    };

    const int num_indent_spaces = 2;
    std::string indent_spaces(num_indent_spaces, ' ');
    out << "producer_name: \"opencv dnn\"\n"
        << "producer_version: \"" << getVersionString() << "\"\n"
        << "graph {\n";
    // Add nodes, inputs and outputs
    for (std::map<int, LayerData>::const_iterator iter = map.begin(); iter != map.end(); iter++) {
        auto &ld = iter->second;
        if (ld.id == 0) {
            for (int i = 0; i < ld.outputBlobs.size(); i++) {
                const auto &name = netInputLayer->outNames.empty() ? cv::format("%s_%d", ld.name.c_str(), i) : netInputLayer->outNames[i];
                out << indent_spaces << "input {\n"
                    << indent_spaces << format("  name: \"%s\"\n", name.c_str());
                // Add shape
                if (!ld.outputBlobs.empty()) {
                    dumpTensorToString(out, ld.outputBlobs[i], num_indent_spaces + 2);
                }
                out << indent_spaces << "}\n"; // input{}
            }
        } else if (ld.consumers.size() == 0) {
            out << indent_spaces << "output {\n"
                << indent_spaces << format("  name: \"%s\"\n", ld.name.c_str());
            // Add shape
            if (!ld.outputBlobs.empty()) {
                dumpTensorToString(out, ld.outputBlobs.front(), num_indent_spaces + 2);
            }
            out << indent_spaces << "}\n"; // output{}
        } else {
            out << indent_spaces << "node {\n";
            const auto &name = ld.name;
            const auto &op_type = "cv::dnn::" + ld.type;
            std::vector<std::string> inputs, outputs;
            // Collect names of inputs
            for (size_t i = 0; i < ld.inputBlobsId.size(); i++) {
                int lid = ld.inputBlobsId[i].lid;
                int oid = ld.inputBlobsId[i].oid;
                std::string name;
                if (lid == 0) {
                    name = netInputLayer->outNames.empty() ? cv::format("%s_%d", ld.name.c_str(), oid) : netInputLayer->outNames[oid];
                } else {
                    name = format("%s_output%d", map.find(lid)->second.name.c_str(), oid);
                    if (!ld.inputBlobs.empty()) {
                        value_info.insert({name, ld.inputBlobs[i]});
                    }
                }
                inputs.push_back(name);
            }
            // Collect names of outputs
            for (size_t i = 0; i < ld.consumers.size(); i++) {
                int lid = ld.consumers[i].lid;
                const auto &layer_output_layer = map.find(lid)->second;
                std::string name;
                if (layer_output_layer.consumers.size() == 0) {
                    name = layer_output_layer.name;
                } else {
                    name = format("%s_output%zu", ld.name.c_str(), i);
                }
                outputs.push_back(name);
            }
            const auto &params = ld.params;
            // Collect backend and target
            const Backend backend = ld.backendNodes.find(prefBackend) == ld.backendNodes.end() ? DNN_BACKEND_OPENCV : prefBackend;
            const std::string backend_name = GetBackendName(backend);
            const Target target = ld.layerInstance.empty() ? DNN_TARGET_CPU : (Target)(ld.layerInstance->preferableTarget);
            const std::string target_name = GetTargetName(target);
            dumpLayerToString(out, inputs, outputs, name, op_type, params, backend_name, target_name, num_indent_spaces + 2);
            out << indent_spaces << "}\n"; // node{}
        }
    }
    // Add value_info
    for (std::map<String, Mat*>::const_iterator iter = value_info.begin(); iter != value_info.end(); iter++) {
        out << indent_spaces << "value_info {\n"
            << indent_spaces << format("  name: \"%s\"\n", iter->first.c_str());
        dumpTensorToString(out, *(iter->second), num_indent_spaces + 2);
        out << indent_spaces << "}\n"; // value_info{}
    }
    out << "}\n"; // graph{}

    // Add preferable backend and target as metadata
    out << "metadata_props {\n";
    out << indent_spaces << format("  key: \"%s\"", "Preferable Backend")
        << indent_spaces << format("  value: \"%s\"", GetBackendName(prefBackend).c_str());
    out << "}\n"; // metadata_props{}
    out << "metadata_props {\n";
    out << indent_spaces << format("  key: \"%s\"", "Preferable Target")
        << indent_spaces << format("  value: \"%s\"", GetTargetName(prefTarget).c_str());
    out << "}\n"; // metadata_props{}

    return out.str();
}

void Net::Impl::dumpNetworkToFile() const
{
#ifndef OPENCV_DNN_DISABLE_NETWORK_AUTO_DUMP
    string dumpFileNameBase = getDumpFileNameBase();
    string dumpFileName = dumpFileNameBase + ".pbtxt";
    try
    {
        string dumpStr = dumpToPbtxt();
        std::ofstream out(dumpFileName.c_str(), std::ios::out | std::ios::binary);
        out << dumpStr;
    }
    catch (const std::exception& e)
    {
        std::ofstream out((dumpFileName + ".error").c_str(), std::ios::out);
        out << "Exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::ofstream out((dumpFileName + ".error").c_str(), std::ios::out);
        out << "Can't dump: unknown exception" << std::endl;
    }
#endif
}


std::vector<Ptr<Layer>> Net::Impl::getLayerInputs(int layerId) const
{
    LayerData& ld = getLayerData(layerId);

    std::vector<Ptr<Layer>> inputLayers;
    inputLayers.reserve(ld.inputBlobsId.size());
    for (int i = 0; i < ld.inputBlobsId.size(); ++i)
    {
        inputLayers.push_back(getLayer(ld.inputBlobsId[i].lid));
    }
    return inputLayers;
}

std::vector<String> Net::Impl::getLayerNames() const
{
    std::vector<String> res;
    res.reserve(layers.size());

    Impl::MapIdToLayerData::const_iterator it;
    for (it = layers.begin(); it != layers.end(); it++)
    {
        if (it->second.id)  // skip Data layer
            res.push_back(it->second.name);
    }

    return res;
}


// FIXIT drop "unconnected" API
std::vector<int> Net::Impl::getUnconnectedOutLayers() const
{
    std::vector<int> layersIds;

    // registerOutput() flow
    if (!outputNameToId.empty())
    {
        for (std::map<std::string, int>::const_iterator it = outputNameToId.begin(); it != outputNameToId.end(); ++it)
        {
            layersIds.push_back(it->second);
        }
        return layersIds;
    }

    Impl::MapIdToLayerData::const_iterator it;
    for (it = layers.begin(); it != layers.end(); it++)
    {
        int lid = it->first;
        const LayerData& ld = it->second;

        if (ld.requiredOutputs.size() == 0)
            layersIds.push_back(lid);
    }

    return layersIds;
}


// FIXIT drop "unconnected" API
std::vector<String> Net::Impl::getUnconnectedOutLayersNames() /*const*/
{
    std::vector<int> ids = getUnconnectedOutLayers();
    const size_t n = ids.size();
    std::vector<String> names(n);
    for (size_t i = 0; i < n; ++i)
    {
        names[i] = layers[ids[i]].name;
    }
    return names;
}


int64 Net::Impl::getFLOPS(const std::vector<MatShape>& netInputShapes) /*const*/
{
    int64 flops = 0;
    std::vector<int> ids;
    std::vector<std::vector<MatShape>> inShapes, outShapes;
    getLayersShapes(netInputShapes, ids, inShapes, outShapes);
    CV_Assert(inShapes.size() == outShapes.size());
    CV_Assert(inShapes.size() == ids.size());

    for (int i = 0; i < ids.size(); i++)
    {
        flops += getLayerInstance(layers[ids[i]])->getFLOPS(inShapes[i], outShapes[i]);
    }

    return flops;
}


int64 Net::Impl::getFLOPS(
        const int layerId,
        const std::vector<MatShape>& netInputShapes) /*const*/
{
    Impl::MapIdToLayerData::const_iterator layer = layers.find(layerId);
    CV_Assert(layer != layers.end());

    LayerShapes shapes;
    getLayerShapes(netInputShapes, layerId, shapes);

    return getLayerInstance(const_cast<LayerData&>(layer->second))->getFLOPS(shapes.in, shapes.out);
}


void Net::Impl::getMemoryConsumption(
        const int layerId,
        const std::vector<MatShape>& netInputShapes,
        size_t& weights, size_t& blobs) /*const*/
{
    Impl::MapIdToLayerData::const_iterator layer = layers.find(layerId);
    CV_Assert(layer != layers.end());

    weights = blobs = 0;

    for (int i = 0; i < layer->second.params.blobs.size(); i++)
    {
        const Mat& weightsBlob = layer->second.params.blobs[i];
        weights += weightsBlob.total() * weightsBlob.elemSize();
    }

    LayerShapes shapes;
    getLayerShapes(netInputShapes, layerId, shapes);
    const ShapesVec& outLayerShapes = shapes.out;

    // FIXIT netWasQuantized check is not enough - per layer check should be done
    size_t elemSize = netWasQuantized ? sizeof(char) : sizeof(float);
    for (int i = 0; i < outLayerShapes.size(); i++)
    {
        blobs += total(outLayerShapes[i]) * elemSize;
    }
}


void Net::Impl::getMemoryConsumption(
        const std::vector<MatShape>& netInputShapes,
        size_t& weights, size_t& blobs) /*const*/
{
    std::vector<int> layerIds;
    std::vector<size_t> w, b;
    getMemoryConsumption(netInputShapes, layerIds, w, b);

    weights = blobs = 0;
    for (int i = 0; i < layerIds.size(); i++)
    {
        weights += w[i];
        blobs += b[i];
    }
}


int64 Net::Impl::getPerfProfile(std::vector<double>& timings) const
{
    timings = std::vector<double>(layersTimings.begin() + 1, layersTimings.end());
    int64 total = (int64)std::accumulate(timings.begin(), timings.end(), 0.0);
    return total;
}

void Net::Impl::getMemoryConsumption(
        const std::vector<MatShape>& netInputShapes,
        std::vector<int>& layerIds, std::vector<size_t>& weights,
        std::vector<size_t>& blobs) /*const*/
{
    layerIds.clear();
    weights.clear();
    blobs.clear();

    std::vector<std::vector<MatShape>> inLayerShapes, outLayerShapes;

    getLayersShapes(netInputShapes, layerIds, inLayerShapes, outLayerShapes);
    // FIXIT netWasQuantized check is not enough - per layer check should be done
    size_t elemSize = netWasQuantized ? sizeof(char) : sizeof(float);
    for (int i = 0; i < layerIds.size(); i++)
    {
        int w = 0, b = 0;
        Impl::MapIdToLayerData::const_iterator layer = layers.find(layerIds[i]);
        CV_Assert(layer != layers.end());

        for (int j = 0; j < layer->second.params.blobs.size(); j++)
        {
            const Mat& weightsBlob = layer->second.params.blobs[j];
            w += weightsBlob.total() * weightsBlob.elemSize();
        }

        for (int j = 0; j < outLayerShapes[i].size(); j++)
        {
            b += total(outLayerShapes[i][j]) * elemSize;
        }

        weights.push_back(w);
        blobs.push_back(b);
    }
}

void Net::Impl::enableWinograd(bool useWinograd_)
{
    if (useWinograd != useWinograd_)
    {
        useWinograd = useWinograd_;

        for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData &ld = layers[lid];
            Ptr<Layer>& currLayer = ld.layerInstance;

            if (ld.type == "Convolution")
            {
                ld.params.set("use_winograd", useWinograd_);
                Ptr<ConvolutionLayer> convLayer = ld.layerInstance.dynamicCast<ConvolutionLayer>();
                if (!convLayer.empty())
                    convLayer->useWinograd = useWinograd_;
            }

            if (ld.type == "ConvolutionInt8")
            {
                Ptr<ConvolutionLayerInt8> convLayer = currLayer.dynamicCast<ConvolutionLayerInt8>();
                ld.params.set("use_winograd", useWinograd_);
                if (!convLayer.empty())
                    convLayer->useWinograd = useWinograd_;
            }
        }
    }
}


// TODO drop?
void Net::Impl::getLayerTypes(std::vector<String>& layersTypes) const
{
    layersTypes.clear();

    std::map<String, int> layers_type_map;
    for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); it++)
    {
        if (layers_type_map.find(it->second.type) == layers_type_map.end())
            layers_type_map[it->second.type] = 0;
        layers_type_map[it->second.type]++;
    }

    for (std::map<String, int>::const_iterator it = layers_type_map.begin(); it != layers_type_map.end(); it++)
    {
        layersTypes.push_back(it->first);
    }
}


// TODO drop?
int Net::Impl::getLayersCount(const String& layerType) const
{
    int count = 0;
    for (Impl::MapIdToLayerData::const_iterator it = layers.begin();
            it != layers.end(); it++)
    {
        if (it->second.type == layerType)
            count++;
    }
    return count;
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
