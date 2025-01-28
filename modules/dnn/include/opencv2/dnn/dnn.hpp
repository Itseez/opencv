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

#ifndef OPENCV_DNN_DNN_HPP
#define OPENCV_DNN_DNN_HPP

#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/core/async.hpp"

#include "../dnn/version.hpp"

#include <opencv2/dnn/dict.hpp>

namespace cv {
namespace dnn {

namespace accessor {
class DnnNetAccessor;  // forward declaration
}

CV__DNN_INLINE_NS_BEGIN
//! @addtogroup dnn
//! @{

    typedef std::vector<int> MatShape;

    /**
     * @brief Enum of computation backends supported by layers.
     * @see Net::setPreferableBackend
     */
    enum Backend
    {
        //! DNN_BACKEND_DEFAULT equals to OPENCV_DNN_BACKEND_DEFAULT, which can be defined using CMake or a configuration parameter
        DNN_BACKEND_DEFAULT = 0,
        DNN_BACKEND_HALIDE,
        DNN_BACKEND_INFERENCE_ENGINE,            //!< Intel OpenVINO computational backend
                                                 //!< @note Tutorial how to build OpenCV with OpenVINO: @ref tutorial_dnn_openvino
        DNN_BACKEND_OPENCV,
        DNN_BACKEND_VKCOM,
        DNN_BACKEND_CUDA,
        DNN_BACKEND_WEBNN,
        DNN_BACKEND_TIMVX,
        DNN_BACKEND_CANN,
        DNN_BACKEND_TENSORRT,
#if defined(__OPENCV_BUILD) || defined(BUILD_PLUGIN)
#if !defined(OPENCV_BINDING_PARSER)
        DNN_BACKEND_INFERENCE_ENGINE_NGRAPH = 1000000,     // internal - use DNN_BACKEND_INFERENCE_ENGINE + setInferenceEngineBackendType()
        DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019,      // internal - use DNN_BACKEND_INFERENCE_ENGINE + setInferenceEngineBackendType()
#endif
#endif
    };

    /**
     * @brief Enum of target devices for computations.
     * @see Net::setPreferableTarget
     */
    enum Target
    {
        DNN_TARGET_CPU = 0,
        DNN_TARGET_OPENCL,
        DNN_TARGET_OPENCL_FP16,
        DNN_TARGET_MYRIAD,
        DNN_TARGET_VULKAN,
        DNN_TARGET_FPGA,  //!< FPGA device with CPU fallbacks using Inference Engine's Heterogeneous plugin.
        DNN_TARGET_CUDA,
        DNN_TARGET_CUDA_FP16,
        DNN_TARGET_HDDL,
        DNN_TARGET_NPU,
        DNN_TARGET_CPU_FP16, // Only the ARM platform is supported. Low precision computing, accelerate model inference.
        DNN_TARGET_TENSORRT,
    };

    /**
     * @brief Enum of data layout for model inference.
     * @see Image2BlobParams
     */
    enum DataLayout
    {
        DNN_LAYOUT_UNKNOWN = 0,
        DNN_LAYOUT_ND = 1,        //!< OpenCV data layout for 2D data.
        DNN_LAYOUT_NCHW = 2,      //!< OpenCV data layout for 4D data.
        DNN_LAYOUT_NCDHW = 3,      //!< OpenCV data layout for 5D data.
        DNN_LAYOUT_NHWC = 4,      //!< Tensorflow-like data layout for 4D data.
        DNN_LAYOUT_NDHWC = 5,      //!< Tensorflow-like data layout for 5D data.
        DNN_LAYOUT_PLANAR = 6,     //!< Tensorflow-like data layout, it should only be used at tf or tflite model parsing.
    };

    CV_EXPORTS std::vector< std::pair<Backend, Target> > getAvailableBackends();
    CV_EXPORTS_W std::vector<Target> getAvailableTargets(dnn::Backend be);

    /**
     * @brief Enables detailed logging of the DNN model loading with CV DNN API.
     * @param[in] isDiagnosticsMode Indicates whether diagnostic mode should be set.
     *
     * Diagnostic mode provides detailed logging of the model loading stage to explore
     * potential problems (ex.: not implemented layer type).
     *
     * @note In diagnostic mode series of assertions will be skipped, it can lead to the
     * expected application crash.
     */
    CV_EXPORTS void enableModelDiagnostics(bool isDiagnosticsMode);

    /** @brief This class provides all data needed to initialize layer.
     *
     * It includes dictionary with scalar params (which can be read by using Dict interface),
     * blob params #blobs and optional meta information: #name and #type of layer instance.
    */
    class CV_EXPORTS LayerParams : public Dict
    {
    public:
        //TODO: Add ability to name blob params
        std::vector<Mat> blobs; //!< List of learned parameters stored as blobs.

        String name; //!< Name of the layer instance (optional, can be used internal purposes).
        String type; //!< Type name which was used for creating layer by layer factory (optional).
    };

   /**
    * @brief Derivatives of this class encapsulates functions of certain backends.
    */
    class BackendNode
    {
    public:
        explicit BackendNode(int backendId);

        virtual ~BackendNode(); //!< Virtual destructor to make polymorphism.

        int backendId; //!< Backend identifier.
    };

    /**
     * @brief Derivatives of this class wraps cv::Mat for different backends and targets.
     */
    class BackendWrapper
    {
    public:
        BackendWrapper(int backendId, int targetId);

        /**
         * @brief Wrap cv::Mat for specific backend and target.
         * @param[in] targetId Target identifier.
         * @param[in] m cv::Mat for wrapping.
         *
         * Make CPU->GPU data transfer if it's require for the target.
         */
        BackendWrapper(int targetId, const cv::Mat& m);

        /**
         * @brief Make wrapper for reused cv::Mat.
         * @param[in] base Wrapper of cv::Mat that will be reused.
         * @param[in] shape Specific shape.
         *
         * Initialize wrapper from another one. It'll wrap the same host CPU
         * memory and mustn't allocate memory on device(i.e. GPU). It might
         * has different shape. Use in case of CPU memory reusing for reuse
         * associated memory on device too.
         */
        BackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape);

        virtual ~BackendWrapper(); //!< Virtual destructor to make polymorphism.

        /**
         * @brief Transfer data to CPU host memory.
         */
        virtual void copyToHost() = 0;

        /**
         * @brief Indicate that an actual data is on CPU.
         */
        virtual void setHostDirty() = 0;

        int backendId;  //!< Backend identifier.
        int targetId;   //!< Target identifier.
    };

    class CV_EXPORTS ActivationLayer;

    /** @brief This interface class allows to build new Layers - are building blocks of networks.
     *
     * Each class, derived from Layer, must implement forward() method to compute outputs.
     * Also before using the new layer into networks you must register your layer by using one of @ref dnnLayerFactory "LayerFactory" macros.
     */
    class CV_EXPORTS_W Layer : public Algorithm
    {
    public:

        //! List of learned parameters must be stored here to allow read them by using Net::getParam().
        CV_PROP_RW std::vector<Mat> blobs;

        /** @brief Computes and sets internal parameters according to inputs, outputs and blobs.
         *  @deprecated Use Layer::finalize(InputArrayOfArrays, OutputArrayOfArrays) instead
         *  @param[in]  input  vector of already allocated input blobs
         *  @param[out] output vector of already allocated output blobs
         *
         * This method is called after network has allocated all memory for input and output blobs
         * and before inferencing.
         */
        CV_DEPRECATED_EXTERNAL
        virtual void finalize(const std::vector<Mat*> &input, std::vector<Mat> &output);

        /** @brief Computes and sets internal parameters according to inputs, outputs and blobs.
         *  @param[in]  inputs  vector of already allocated input blobs
         *  @param[out] outputs vector of already allocated output blobs
         *
         * This method is called after network has allocated all memory for input and output blobs
         * and before inferencing.
         */
        CV_WRAP virtual void finalize(InputArrayOfArrays inputs, OutputArrayOfArrays outputs);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @deprecated Use Layer::forward(InputArrayOfArrays, OutputArrayOfArrays, OutputArrayOfArrays) instead
         *  @param[in]  input  the input blobs.
         *  @param[out] output allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        CV_DEPRECATED_EXTERNAL
        virtual void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @param[in]  inputs  the input blobs.
         *  @param[out] outputs allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals);

        /** @brief Tries to quantize the given layer and compute the quantization parameters required for fixed point implementation.
         *  @param[in] scales input and output scales.
         *  @param[in] zeropoints input and output zeropoints.
         *  @param[out] params Quantized parameters required for fixed point implementation of that layer.
         *  @returns True if layer can be quantized.
         */
        virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                                 const std::vector<std::vector<int> > &zeropoints, LayerParams& params);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @param[in]  inputs  the input blobs.
         *  @param[out] outputs allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        void forward_fallback(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals);

        /** @brief
         * @overload
         * @deprecated Use Layer::finalize(InputArrayOfArrays, OutputArrayOfArrays) instead
         */
        CV_DEPRECATED_EXTERNAL
        void finalize(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs);

        /** @brief
         * @overload
         * @deprecated Use Layer::finalize(InputArrayOfArrays, OutputArrayOfArrays) instead
         */
        CV_DEPRECATED std::vector<Mat> finalize(const std::vector<Mat> &inputs);

        /** @brief Allocates layer and computes output.
         *  @deprecated This method will be removed in the future release.
         */
        CV_DEPRECATED CV_WRAP void run(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs,
                                       CV_IN_OUT std::vector<Mat> &internals);

        /** @brief Returns index of input blob into the input array.
         *  @param inputName label of input blob
         *
         * Each layer input and output can be labeled to easily identify them using "%<layer_name%>[.output_name]" notation.
         * This method maps label of input blob to its index into input vector.
         */
        virtual int inputNameToIndex(String inputName);  // FIXIT const
        /** @brief Returns index of output blob in output array.
         *  @see inputNameToIndex()
         */
        CV_WRAP virtual int outputNameToIndex(const String& outputName);  // FIXIT const

        /**
         * @brief Ask layer if it support specific backend for doing computations.
         * @param[in] backendId computation backend identifier.
         * @see Backend
         */
        virtual bool supportBackend(int backendId);  // FIXIT const

        /**
         * @brief Returns Halide backend node.
         * @param[in] inputs Input Halide buffers.
         * @see BackendNode, BackendWrapper
         *
         * Input buffers should be exactly the same that will be used in forward invocations.
         * Despite we can use Halide::ImageParam based on input shape only,
         * it helps prevent some memory management issues (if something wrong,
         * Halide tests will be failed).
         */
        virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs);

        virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs, const std::vector<Ptr<BackendNode> >& nodes);

        virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs, std::vector<Ptr<BackendWrapper> > &outputs);

        virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> > &inputs, const std::vector<Ptr<BackendNode> >& nodes);

        /**
         * @brief Returns a CUDA backend node
         *
         * @param   context  void pointer to CSLContext object
         * @param   inputs   layer inputs
         * @param   outputs  layer outputs
         */
        virtual Ptr<BackendNode> initCUDA(
            void *context,
            const std::vector<Ptr<BackendWrapper>>& inputs,
            const std::vector<Ptr<BackendWrapper>>& outputs
        );

        /**
         * @brief Returns a TimVX backend node
         *
         * @param   timVxInfo  void pointer to CSLContext object
         * @param   inputsWrapper   layer inputs
         * @param   outputsWrapper  layer outputs
         * @param   isLast if the node is the last one of the TimVX Graph.
         */
        virtual Ptr<BackendNode> initTimVX(void* timVxInfo,
                                           const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                           const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                           bool isLast);

        /**
         * @brief Returns a CANN backend node
         *
         * @param   inputs   input tensors of CANN operator
         * @param   outputs  output tensors of CANN operator
         * @param   nodes           nodes of input tensors
         */
        virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                          const std::vector<Ptr<BackendWrapper> > &outputs,
                                          const std::vector<Ptr<BackendNode> >& nodes);

       /**
        * @brief Automatic Halide scheduling based on layer hyper-parameters.
        * @param[in] node Backend node with Halide functions.
        * @param[in] inputs Blobs that will be used in forward invocations.
        * @param[in] outputs Blobs that will be used in forward invocations.
        * @param[in] targetId Target identifier
        * @see BackendNode, Target
        *
        * Layer don't use own Halide::Func members because we can have applied
        * layers fusing. In this way the fused function should be scheduled.
        */
        virtual void applyHalideScheduler(Ptr<BackendNode>& node,
                                          const std::vector<Mat*> &inputs,
                                          const std::vector<Mat> &outputs,
                                          int targetId) const;

        /**
         * @brief Implement layers fusing.
         * @param[in] node Backend node of bottom layer.
         * @see BackendNode
         *
         * Actual for graph-based backends. If layer attached successfully,
         * returns non-empty cv::Ptr to node of the same backend.
         * Fuse only over the last function.
         */
        virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node);

        /**
         * @brief Tries to attach to the layer the subsequent activation layer, i.e. do the layer fusion in a partial case.
         * @param[in] layer The subsequent activation layer.
         *
         * Returns true if the activation layer has been attached successfully.
         */
        virtual bool setActivation(const Ptr<ActivationLayer>& layer);

        /**
         * @brief Try to fuse current layer with a next one
         * @param[in] top Next layer to be fused.
         * @returns True if fusion was performed.
         */
        virtual bool tryFuse(Ptr<Layer>& top);

        /**
         * @brief Returns parameters of layers with channel-wise multiplication and addition.
         * @param[out] scale Channel-wise multipliers. Total number of values should
         *                   be equal to number of channels.
         * @param[out] shift Channel-wise offsets. Total number of values should
         *                   be equal to number of channels.
         *
         * Some layers can fuse their transformations with further layers.
         * In example, convolution + batch normalization. This way base layer
         * use weights from layer after it. Fused layer is skipped.
         * By default, @p scale and @p shift are empty that means layer has no
         * element-wise multiplications or additions.
         */
        virtual void getScaleShift(Mat& scale, Mat& shift) const;

        /**
         * @brief Returns scale and zeropoint of layers
         * @param[out] scale Output scale
         * @param[out] zeropoint Output zeropoint
         *
         * By default, @p scale is 1 and @p zeropoint is 0.
         */
        virtual void getScaleZeropoint(float& scale, int& zeropoint) const;


        /**
         * @brief "Detaches" all the layers, attached to particular layer.
         */
        virtual void unsetAttached();

        virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                     const int requiredOutputs,
                                     std::vector<MatShape> &outputs,
                                     std::vector<MatShape> &internals) const;

        virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                               const std::vector<MatShape> &outputs) const {CV_UNUSED(inputs); CV_UNUSED(outputs); return 0;}

        virtual bool updateMemoryShapes(const std::vector<MatShape> &inputs);

        CV_PROP String name; //!< Name of the layer instance, can be used for logging or other internal purposes.
        CV_PROP String type; //!< Type name which was used for creating layer by layer factory.
        CV_PROP int preferableTarget; //!< prefer target for layer forwarding

        Layer();
        explicit Layer(const LayerParams &params);      //!< Initializes only #name, #type and #blobs fields.
        void setParamsFrom(const LayerParams &params);  //!< Initializes only #name, #type and #blobs fields.
        virtual ~Layer();
    };

    /** @brief This class allows to create and manipulate comprehensive artificial neural networks.
     *
     * Neural network is presented as directed acyclic graph (DAG), where vertices are Layer instances,
     * and edges specify relationships between layers inputs and outputs.
     *
     * Each network layer has unique integer id and unique string name inside its network.
     * LayerId can store either layer name or layer id.
     *
     * This class supports reference counting of its instances, i. e. copies point to the same instance.
     */
    class CV_EXPORTS_W_SIMPLE Net
    {
    public:

        CV_WRAP Net();  //!< Default constructor.
        CV_WRAP ~Net(); //!< Destructor frees the net only if there aren't references to the net anymore.

        /** @brief Create a network from Intel's Model Optimizer intermediate representation (IR).
         *  @param[in] xml XML configuration file with network's topology.
         *  @param[in] bin Binary file with trained weights.
         *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
         *  backend.
         */
        CV_WRAP static Net readFromModelOptimizer(CV_WRAP_FILE_PATH const String& xml, CV_WRAP_FILE_PATH const String& bin);

        /** @brief Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).
         *  @param[in] bufferModelConfig buffer with model's configuration.
         *  @param[in] bufferWeights buffer with model's trained weights.
         *  @returns Net object.
         */
        CV_WRAP static
        Net readFromModelOptimizer(const std::vector<uchar>& bufferModelConfig, const std::vector<uchar>& bufferWeights);

        /** @brief Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).
         *  @param[in] bufferModelConfigPtr buffer pointer of model's configuration.
         *  @param[in] bufferModelConfigSize buffer size of model's configuration.
         *  @param[in] bufferWeightsPtr buffer pointer of model's trained weights.
         *  @param[in] bufferWeightsSize buffer size of model's trained weights.
         *  @returns Net object.
         */
        static
        Net readFromModelOptimizer(const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
                                            const uchar* bufferWeightsPtr, size_t bufferWeightsSize);

        /** Returns true if there are no layers in the network. */
        CV_WRAP bool empty() const;

        /** @brief Dump net to String
         *  @returns String with structure, hyperparameters, backend, target and fusion
         *  Call method after setInput(). To see correct backend, target and fusion run after forward().
         */
        CV_WRAP String dump();
        /** @brief Dump net structure, hyperparameters, backend, target and fusion to dot file
         *  @param path   path to output file with .dot extension
         *  @see dump()
         */
        CV_WRAP void dumpToFile(CV_WRAP_FILE_PATH const String& path);
        /** @brief Dump net structure, hyperparameters, backend, target and fusion to pbtxt file
         *  @param path   path to output file with .pbtxt extension
         *
         *  Use Netron (https://netron.app) to open the target file to visualize the model.
         *  Call method after setInput(). To see correct backend, target and fusion run after forward().
        */
        CV_WRAP void dumpToPbtxt(CV_WRAP_FILE_PATH const String& path);

        /** @brief Adds new layer to the net.
         *  @param name   unique name of the adding layer.
         *  @param type   typename of the adding layer (type must be registered in LayerRegister).
         *  @param dtype  datatype of output blobs.
         *  @param params parameters which will be used to initialize the creating layer.
         *  @returns unique identifier of created layer, or -1 if a failure will happen.
         */
        CV_WRAP int addLayer(const String &name, const String &type, const int &dtype, LayerParams &params);

        /** @overload Datatype of output blobs set to default CV_32F */
        int addLayer(const String &name, const String &type, LayerParams &params);

        /** @brief Adds new layer and connects its first input to the first output of previously added layer.
         *  @see addLayer()
         */
        CV_WRAP int addLayerToPrev(const String &name, const String &type, const int &dtype, LayerParams &params);

        /** @overload */
        int addLayerToPrev(const String &name, const String &type, LayerParams &params);

        /** @brief Converts string name of the layer to the integer identifier.
         *  @returns id of the layer, or -1 if the layer wasn't found.
         */
        CV_WRAP int getLayerId(const String &layer) const;

        CV_WRAP std::vector<String> getLayerNames() const;

        /** @brief Container for strings and integers.
         *
         * @deprecated Use getLayerId() with int result.
         */
        typedef DictValue LayerId;

        /** @brief Returns pointer to layer with specified id or name which the network use. */
        CV_WRAP Ptr<Layer> getLayer(int layerId) const;
        /** @overload
         *  @deprecated Use int getLayerId(const String &layer)
         */
        CV_WRAP inline Ptr<Layer> getLayer(const String& layerName) const { return getLayer(getLayerId(layerName)); }
        /** @overload
         *  @deprecated to be removed
         */
        CV_WRAP Ptr<Layer> getLayer(const LayerId& layerId) const;

        /** @brief Returns pointers to input layers of specific layer. */
        std::vector<Ptr<Layer> > getLayerInputs(int layerId) const; // FIXIT: CV_WRAP

        /** @brief Connects output of the first layer to input of the second layer.
         *  @param outPin descriptor of the first layer output.
         *  @param inpPin descriptor of the second layer input.
         *
         * Descriptors have the following template <DFN>&lt;layer_name&gt;[.input_number]</DFN>:
         * - the first part of the template <DFN>layer_name</DFN> is string name of the added layer.
         *   If this part is empty then the network input pseudo layer will be used;
         * - the second optional part of the template <DFN>input_number</DFN>
         *   is either number of the layer input, either label one.
         *   If this part is omitted then the first layer input will be used.
         *
         *  @see setNetInputs(), Layer::inputNameToIndex(), Layer::outputNameToIndex()
         */
        CV_WRAP void connect(String outPin, String inpPin);

        /** @brief Connects #@p outNum output of the first layer to #@p inNum input of the second layer.
         *  @param outLayerId identifier of the first layer
         *  @param outNum number of the first layer output
         *  @param inpLayerId identifier of the second layer
         *  @param inpNum number of the second layer input
         */
        void connect(int outLayerId, int outNum, int inpLayerId, int inpNum);

        /** @brief Registers network output with name
         *
         *  Function may create additional 'Identity' layer.
         *
         *  @param outputName identifier of the output
         *  @param layerId identifier of the second layer
         *  @param outputPort number of the second layer input
         *
         *  @returns index of bound layer (the same as layerId or newly created)
         */
        int registerOutput(const std::string& outputName, int layerId, int outputPort);

        /** @brief Sets outputs names of the network input pseudo layer.
         *
         * Each net always has special own the network input pseudo layer with id=0.
         * This layer stores the user blobs only and don't make any computations.
         * In fact, this layer provides the only way to pass user data into the network.
         * As any other layer, this layer can label its outputs and this function provides an easy way to do this.
         */
        CV_WRAP void setInputsNames(const std::vector<String> &inputBlobNames);

        /** @brief Specify shape of network input.
         */
        CV_WRAP void setInputShape(const String &inputName, const MatShape& shape);

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputName name for layer which output is needed to get
         *  @return blob for first output of specified layer.
         *  @details By default runs forward pass for the whole network.
         */
        CV_WRAP Mat forward(const String& outputName = String());

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputName name for layer which output is needed to get
         *  @details By default runs forward pass for the whole network.
         *
         *  This is an asynchronous version of forward(const String&).
         *  dnn::DNN_BACKEND_INFERENCE_ENGINE backend is required.
         */
        CV_WRAP AsyncArray forwardAsync(const String& outputName = String());

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputBlobs contains all output blobs for specified layer.
         *  @param outputName name for layer which output is needed to get
         *  @details If @p outputName is empty, runs forward pass for the whole network.
         */
        CV_WRAP void forward(CV_ND OutputArrayOfArrays outputBlobs, const String& outputName = String());

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains blobs for first outputs of specified layers.
         *  @param outBlobNames names for layers which outputs are needed to get
         */
        CV_WRAP void forward(CV_ND OutputArrayOfArrays outputBlobs,
                             const std::vector<String>& outBlobNames);

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains all output blobs for each layer specified in @p outBlobNames.
         *  @param outBlobNames names for layers which outputs are needed to get
         */
        CV_WRAP_AS(forwardAndRetrieve) void forward(CV_OUT std::vector<std::vector<Mat> >& outputBlobs,
                                                    const std::vector<String>& outBlobNames);

        /** @brief Returns a quantized Net from a floating-point Net.
         *  @param calibData Calibration data to compute the quantization parameters.
         *  @param inputsDtype Datatype of quantized net's inputs. Can be CV_32F or CV_8S.
         *  @param outputsDtype Datatype of quantized net's outputs. Can be CV_32F or CV_8S.
         *  @param perChannel Quantization granularity of quantized Net. The default is true, that means quantize model
         *  in per-channel way (channel-wise). Set it false to quantize model in per-tensor way (or tensor-wise).
         */
        CV_WRAP Net quantize(InputArrayOfArrays calibData, int inputsDtype, int outputsDtype, bool perChannel=true);

        /** @brief Returns input scale and zeropoint for a quantized Net.
         *  @param scales output parameter for returning input scales.
         *  @param zeropoints output parameter for returning input zeropoints.
         */
        CV_WRAP void getInputDetails(CV_OUT std::vector<float>& scales, CV_OUT std::vector<int>& zeropoints) const;

        /** @brief Returns output scale and zeropoint for a quantized Net.
         *  @param scales output parameter for returning output scales.
         *  @param zeropoints output parameter for returning output zeropoints.
         */
        CV_WRAP void getOutputDetails(CV_OUT std::vector<float>& scales, CV_OUT std::vector<int>& zeropoints) const;

        /**
         * @brief Compile Halide layers.
         * @param[in] scheduler Path to YAML file with scheduling directives.
         * @see setPreferableBackend
         *
         * Schedule layers that support Halide backend. Then compile them for
         * specific target. For layers that not represented in scheduling file
         * or if no manual scheduling used at all, automatic scheduling will be applied.
         */
        CV_WRAP void setHalideScheduler(const String& scheduler);

        /**
         * @brief Ask network to use specific computation backend where it supported.
         * @param[in] backendId backend identifier.
         * @see Backend
         */
        CV_WRAP void setPreferableBackend(int backendId);

        /**
         * @brief Ask network to make computations on specific target device.
         * @param[in] targetId target identifier.
         * @see Target
         *
         * List of supported combinations backend / target:
         * |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |  DNN_BACKEND_CUDA |
         * |------------------------|--------------------|------------------------------|--------------------|-------------------|
         * | DNN_TARGET_CPU         |                  + |                            + |                  + |                   |
         * | DNN_TARGET_OPENCL      |                  + |                            + |                  + |                   |
         * | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |                   |
         * | DNN_TARGET_MYRIAD      |                    |                            + |                    |                   |
         * | DNN_TARGET_FPGA        |                    |                            + |                    |                   |
         * | DNN_TARGET_CUDA        |                    |                              |                    |                 + |
         * | DNN_TARGET_CUDA_FP16   |                    |                              |                    |                 + |
         * | DNN_TARGET_HDDL        |                    |                            + |                    |                   |
         */
        CV_WRAP void setPreferableTarget(int targetId);

        /** @brief Sets the new input value for the network
         *  @param blob        A new blob. Should have CV_32F or CV_8U depth.
         *  @param name        A name of input layer.
         *  @param scalefactor An optional normalization scale.
         *  @param mean        An optional mean subtraction values.
         *  @see connect(String, String) to know format of the descriptor.
         *
         *  If scale or mean values are specified, a final input blob is computed
         *  as:
         * \f[input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\f]
         */
        CV_WRAP void setInput(CV_ND InputArray blob, const String& name = "",
                              double scalefactor = 1.0, const Scalar& mean = Scalar());

        /** @brief Sets the new value for the learned param of the layer.
         *  @param layer name or id of the layer.
         *  @param numParam index of the layer parameter in the Layer::blobs array.
         *  @param blob the new value.
         *  @see Layer::blobs
         *  @note If shape of the new blob differs from the previous shape,
         *  then the following forward pass may fail.
        */
        CV_WRAP void setParam(int layer, int numParam, CV_ND const Mat &blob);
        CV_WRAP inline void setParam(const String& layerName, int numParam, CV_ND const Mat &blob) { return setParam(getLayerId(layerName), numParam, blob); }

        /** @brief Returns parameter blob of the layer.
         *  @param layer name or id of the layer.
         *  @param numParam index of the layer parameter in the Layer::blobs array.
         *  @see Layer::blobs
         */
        CV_WRAP Mat getParam(int layer, int numParam = 0) const;
        CV_WRAP inline Mat getParam(const String& layerName, int numParam = 0) const { return getParam(getLayerId(layerName), numParam); }

        /** @brief Returns indexes of layers with unconnected outputs.
         *
         * FIXIT: Rework API to registerOutput() approach, deprecate this call
         */
        CV_WRAP std::vector<int> getUnconnectedOutLayers() const;

        /** @brief Returns names of layers with unconnected outputs.
         *
         * FIXIT: Rework API to registerOutput() approach, deprecate this call
         */
        CV_WRAP std::vector<String> getUnconnectedOutLayersNames() const;

        /** @brief Returns input and output shapes for all layers in loaded model;
         *  preliminary inferencing isn't necessary.
         *  @param netInputShapes shapes for all input blobs in net input layer.
         *  @param layersIds output parameter for layer IDs.
         *  @param inLayersShapes output parameter for input layers shapes;
         * order is the same as in layersIds
         *  @param outLayersShapes output parameter for output layers shapes;
         * order is the same as in layersIds
         */
        CV_WRAP void getLayersShapes(const std::vector<MatShape>& netInputShapes,
                                     CV_OUT std::vector<int>& layersIds,
                                     CV_OUT std::vector<std::vector<MatShape> >& inLayersShapes,
                                     CV_OUT std::vector<std::vector<MatShape> >& outLayersShapes) const;

        /** @overload */
        CV_WRAP void getLayersShapes(const MatShape& netInputShape,
                                     CV_OUT std::vector<int>& layersIds,
                                     CV_OUT std::vector<std::vector<MatShape> >& inLayersShapes,
                                     CV_OUT std::vector<std::vector<MatShape> >& outLayersShapes) const;

        /** @brief Returns input and output shapes for layer with specified
         * id in loaded model; preliminary inferencing isn't necessary.
         *  @param netInputShape shape input blob in net input layer.
         *  @param layerId id for layer.
         *  @param inLayerShapes output parameter for input layers shapes;
         * order is the same as in layersIds
         *  @param outLayerShapes output parameter for output layers shapes;
         * order is the same as in layersIds
         */
        void getLayerShapes(const MatShape& netInputShape,
                                    const int layerId,
                                    CV_OUT std::vector<MatShape>& inLayerShapes,
                                    CV_OUT std::vector<MatShape>& outLayerShapes) const; // FIXIT: CV_WRAP

        /** @overload */
        void getLayerShapes(const std::vector<MatShape>& netInputShapes,
                                    const int layerId,
                                    CV_OUT std::vector<MatShape>& inLayerShapes,
                                    CV_OUT std::vector<MatShape>& outLayerShapes) const; // FIXIT: CV_WRAP

        /** @brief Computes FLOP for whole loaded model with specified input shapes.
         * @param netInputShapes vector of shapes for all net inputs.
         * @returns computed FLOP.
         */
        CV_WRAP int64 getFLOPS(const std::vector<MatShape>& netInputShapes) const;
        /** @overload */
        CV_WRAP int64 getFLOPS(const MatShape& netInputShape) const;
        /** @overload */
        CV_WRAP int64 getFLOPS(const int layerId,
                               const std::vector<MatShape>& netInputShapes) const;
        /** @overload */
        CV_WRAP int64 getFLOPS(const int layerId,
                               const MatShape& netInputShape) const;

        /** @brief Returns list of types for layer used in model.
         * @param layersTypes output parameter for returning types.
         */
        CV_WRAP void getLayerTypes(CV_OUT std::vector<String>& layersTypes) const;

        /** @brief Returns count of layers of specified type.
         * @param layerType type.
         * @returns count of layers
         */
        CV_WRAP int getLayersCount(const String& layerType) const;

        /** @brief Computes bytes number which are required to store
         * all weights and intermediate blobs for model.
         * @param netInputShapes vector of shapes for all net inputs.
         * @param weights output parameter to store resulting bytes for weights.
         * @param blobs output parameter to store resulting bytes for intermediate blobs.
         */
        void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const; // FIXIT: CV_WRAP
        /** @overload */
        CV_WRAP void getMemoryConsumption(const MatShape& netInputShape,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const;
        /** @overload */
        CV_WRAP void getMemoryConsumption(const int layerId,
                                          const std::vector<MatShape>& netInputShapes,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const;
        /** @overload */
        CV_WRAP void getMemoryConsumption(const int layerId,
                                          const MatShape& netInputShape,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const;

        /** @brief Computes bytes number which are required to store
         * all weights and intermediate blobs for each layer.
         * @param netInputShapes vector of shapes for all net inputs.
         * @param layerIds output vector to save layer IDs.
         * @param weights output parameter to store resulting bytes for weights.
         * @param blobs output parameter to store resulting bytes for intermediate blobs.
         */
        void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                          CV_OUT std::vector<int>& layerIds,
                                          CV_OUT std::vector<size_t>& weights,
                                          CV_OUT std::vector<size_t>& blobs) const; // FIXIT: CV_WRAP
        /** @overload */
        void getMemoryConsumption(const MatShape& netInputShape,
                                          CV_OUT std::vector<int>& layerIds,
                                          CV_OUT std::vector<size_t>& weights,
                                          CV_OUT std::vector<size_t>& blobs) const; // FIXIT: CV_WRAP

        /** @brief Enables or disables layer fusion in the network.
         * @param fusion true to enable the fusion, false to disable. The fusion is enabled by default.
         */
        CV_WRAP void enableFusion(bool fusion);

        /** @brief Enables or disables the Winograd compute branch. The Winograd compute branch can speed up
         * 3x3 Convolution at a small loss of accuracy.
        * @param useWinograd true to enable the Winograd compute branch. The default is true.
        */
        CV_WRAP void enableWinograd(bool useWinograd);

        /** @brief Returns overall time for inference and timings (in ticks) for layers.
         *
         * Indexes in returned vector correspond to layers ids. Some layers can be fused with others,
         * in this case zero ticks count will be return for that skipped layers. Supported by DNN_BACKEND_OPENCV on DNN_TARGET_CPU only.
         *
         * @param[out] timings vector for tick timings for all layers.
         * @return overall ticks for model inference.
         */
        CV_WRAP int64 getPerfProfile(CV_OUT std::vector<double>& timings);


        struct Impl;
        inline Impl* getImpl() const { return impl.get(); }
        inline Impl& getImplRef() const { CV_DbgAssert(impl); return *impl.get(); }
        friend class accessor::DnnNetAccessor;
    protected:
        Ptr<Impl> impl;
    };

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
    *  @param cfgFile      path to the .cfg file with text description of the network architecture.
    *  @param darknetModel path to the .weights file with learned network.
    *  @returns Network object that ready to do forward, throw an exception in failure cases.
    */
    CV_EXPORTS_W Net readNetFromDarknet(CV_WRAP_FILE_PATH const String &cfgFile, CV_WRAP_FILE_PATH const String &darknetModel = String());

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
     *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
     *  @param bufferModel A buffer contains a content of .weights file with learned network.
     *  @returns Net object.
     */
    CV_EXPORTS_W Net readNetFromDarknet(const std::vector<uchar>& bufferCfg,
                                        const std::vector<uchar>& bufferModel = std::vector<uchar>());

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
     *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
     *  @param lenCfg      Number of bytes to read from bufferCfg
     *  @param bufferModel A buffer contains a content of .weights file with learned network.
     *  @param lenModel    Number of bytes to read from bufferModel
     *  @returns Net object.
     */
    CV_EXPORTS Net readNetFromDarknet(const char *bufferCfg, size_t lenCfg,
                                      const char *bufferModel = NULL, size_t lenModel = 0);

    /** @brief Reads a network model stored in <a href="http://caffe.berkeleyvision.org">Caffe</a> framework's format.
      * @param prototxt   path to the .prototxt file with text description of the network architecture.
      * @param caffeModel path to the .caffemodel file with learned network.
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromCaffe(CV_WRAP_FILE_PATH const String &prototxt, CV_WRAP_FILE_PATH const String &caffeModel = String());

    /** @brief Reads a network model stored in Caffe model in memory.
      * @param bufferProto buffer containing the content of the .prototxt file
      * @param bufferModel buffer containing the content of the .caffemodel file
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromCaffe(const std::vector<uchar>& bufferProto,
                                      const std::vector<uchar>& bufferModel = std::vector<uchar>());

    /** @brief Reads a network model stored in Caffe model in memory.
      * @details This is an overloaded member function, provided for convenience.
      * It differs from the above function only in what argument(s) it accepts.
      * @param bufferProto buffer containing the content of the .prototxt file
      * @param lenProto length of bufferProto
      * @param bufferModel buffer containing the content of the .caffemodel file
      * @param lenModel length of bufferModel
      * @returns Net object.
      */
    CV_EXPORTS Net readNetFromCaffe(const char *bufferProto, size_t lenProto,
                                    const char *bufferModel = NULL, size_t lenModel = 0);

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
      * @param model  path to the .pb file with binary protobuf description of the network architecture
      * @param config path to the .pbtxt file that contains text graph definition in protobuf format.
      *               Resulting Net object is built by text graph using weights from a binary one that
      *               let us make it more flexible.
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromTensorflow(CV_WRAP_FILE_PATH const String &model, CV_WRAP_FILE_PATH const String &config = String());

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
      * @param bufferModel buffer containing the content of the pb file
      * @param bufferConfig buffer containing the content of the pbtxt file
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromTensorflow(const std::vector<uchar>& bufferModel,
                                           const std::vector<uchar>& bufferConfig = std::vector<uchar>());

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
      * @details This is an overloaded member function, provided for convenience.
      * It differs from the above function only in what argument(s) it accepts.
      * @param bufferModel buffer containing the content of the pb file
      * @param lenModel length of bufferModel
      * @param bufferConfig buffer containing the content of the pbtxt file
      * @param lenConfig length of bufferConfig
      */
    CV_EXPORTS Net readNetFromTensorflow(const char *bufferModel, size_t lenModel,
                                         const char *bufferConfig = NULL, size_t lenConfig = 0);

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/lite">TFLite</a> framework's format.
      * @param model  path to the .tflite file with binary flatbuffers description of the network architecture
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromTFLite(CV_WRAP_FILE_PATH const String &model);

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/lite">TFLite</a> framework's format.
      * @param bufferModel buffer containing the content of the tflite file
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromTFLite(const std::vector<uchar>& bufferModel);

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/lite">TFLite</a> framework's format.
      * @details This is an overloaded member function, provided for convenience.
      * It differs from the above function only in what argument(s) it accepts.
      * @param bufferModel buffer containing the content of the tflite file
      * @param lenModel length of bufferModel
      */
    CV_EXPORTS Net readNetFromTFLite(const char *bufferModel, size_t lenModel);

    /**
     *  @brief Reads a network model stored in <a href="http://torch.ch">Torch7</a> framework's format.
     *  @param model    path to the file, dumped from Torch by using torch.save() function.
     *  @param isBinary specifies whether the network was serialized in ascii mode or binary.
     *  @param evaluate specifies testing phase of network. If true, it's similar to evaluate() method in Torch.
     *  @returns Net object.
     *
     *  @note Ascii mode of Torch serializer is more preferable, because binary mode extensively use `long` type of C language,
     *  which has various bit-length on different systems.
     *
     * The loading file must contain serialized <a href="https://github.com/torch/nn/blob/master/doc/module.md">nn.Module</a> object
     * with importing network. Try to eliminate a custom objects from serialazing data to avoid importing errors.
     *
     * List of supported layers (i.e. object instances derived from Torch nn.Module class):
     * - nn.Sequential
     * - nn.Parallel
     * - nn.Concat
     * - nn.Linear
     * - nn.SpatialConvolution
     * - nn.SpatialMaxPooling, nn.SpatialAveragePooling
     * - nn.ReLU, nn.TanH, nn.Sigmoid
     * - nn.Reshape
     * - nn.SoftMax, nn.LogSoftMax
     *
     * Also some equivalents of these classes from cunn, cudnn, and fbcunn may be successfully imported.
     */
     CV_EXPORTS_W Net readNetFromTorch(CV_WRAP_FILE_PATH const String &model, bool isBinary = true, bool evaluate = true);

     /**
      * @brief Read deep learning network represented in one of the supported formats.
      * @param[in] model Binary file contains trained weights. The following file
      *                  extensions are expected for models from different frameworks:
      *                  * `*.caffemodel` (Caffe, http://caffe.berkeleyvision.org/)
      *                  * `*.pb` (TensorFlow, https://www.tensorflow.org/)
      *                  * `*.t7` | `*.net` (Torch, http://torch.ch/)
      *                  * `*.weights` (Darknet, https://pjreddie.com/darknet/)
      *                  * `*.bin` | `*.onnx` (OpenVINO, https://software.intel.com/openvino-toolkit)
      *                  * `*.onnx` (ONNX, https://onnx.ai/)
      * @param[in] config Text file contains network configuration. It could be a
      *                   file with the following extensions:
      *                  * `*.prototxt` (Caffe, http://caffe.berkeleyvision.org/)
      *                  * `*.pbtxt` (TensorFlow, https://www.tensorflow.org/)
      *                  * `*.cfg` (Darknet, https://pjreddie.com/darknet/)
      *                  * `*.xml` (OpenVINO, https://software.intel.com/openvino-toolkit)
      * @param[in] framework Explicit framework name tag to determine a format.
      * @returns Net object.
      *
      * This function automatically detects an origin framework of trained model
      * and calls an appropriate function such @ref readNetFromCaffe, @ref readNetFromTensorflow,
      * @ref readNetFromTorch or @ref readNetFromDarknet. An order of @p model and @p config
      * arguments does not matter.
      */
     CV_EXPORTS_W Net readNet(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config = "", const String& framework = "");

     /**
      * @brief Read deep learning network represented in one of the supported formats.
      * @details This is an overloaded member function, provided for convenience.
      *          It differs from the above function only in what argument(s) it accepts.
      * @param[in] framework    Name of origin framework.
      * @param[in] bufferModel  A buffer with a content of binary file with weights
      * @param[in] bufferConfig A buffer with a content of text file contains network configuration.
      * @returns Net object.
      */
     CV_EXPORTS_W Net readNet(const String& framework, const std::vector<uchar>& bufferModel,
                              const std::vector<uchar>& bufferConfig = std::vector<uchar>());

    /** @brief Loads blob which was serialized as torch.Tensor object of Torch7 framework.
     *  @warning This function has the same limitations as readNetFromTorch().
     */
    CV_EXPORTS_W Mat readTorchBlob(const String &filename, bool isBinary = true);

    /** @brief Load a network from Intel's Model Optimizer intermediate representation.
     *  @param[in] xml XML configuration file with network's topology.
     *  @param[in] bin Binary file with trained weights.
     *  @returns Net object.
     *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
     *  backend.
     */
    CV_EXPORTS_W
    Net readNetFromModelOptimizer(CV_WRAP_FILE_PATH const String &xml, CV_WRAP_FILE_PATH const String &bin = "");

    /** @brief Load a network from Intel's Model Optimizer intermediate representation.
     *  @param[in] bufferModelConfig Buffer contains XML configuration with network's topology.
     *  @param[in] bufferWeights Buffer contains binary data with trained weights.
     *  @returns Net object.
     *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
     *  backend.
     */
    CV_EXPORTS_W
    Net readNetFromModelOptimizer(const std::vector<uchar>& bufferModelConfig, const std::vector<uchar>& bufferWeights);

    /** @brief Load a network from Intel's Model Optimizer intermediate representation.
     *  @param[in] bufferModelConfigPtr Pointer to buffer which contains XML configuration with network's topology.
     *  @param[in] bufferModelConfigSize Binary size of XML configuration data.
     *  @param[in] bufferWeightsPtr Pointer to buffer which contains binary data with trained weights.
     *  @param[in] bufferWeightsSize Binary size of trained weights data.
     *  @returns Net object.
     *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
     *  backend.
     */
    CV_EXPORTS
    Net readNetFromModelOptimizer(const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
                                           const uchar* bufferWeightsPtr, size_t bufferWeightsSize);

    /** @brief Reads a network model <a href="https://onnx.ai/">ONNX</a>.
     *  @param onnxFile path to the .onnx file with text description of the network architecture.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     */
    CV_EXPORTS_W Net readNetFromONNX(CV_WRAP_FILE_PATH const String &onnxFile);

    /** @brief Reads a network model from <a href="https://onnx.ai/">ONNX</a>
     *         in-memory buffer.
     *  @param buffer memory address of the first byte of the buffer.
     *  @param sizeBuffer size of the buffer.
     *  @returns Network object that ready to do forward, throw an exception
     *        in failure cases.
     */
    CV_EXPORTS Net readNetFromONNX(const char* buffer, size_t sizeBuffer);

    /** @brief Reads a network model from <a href="https://onnx.ai/">ONNX</a>
     *         in-memory buffer.
     *  @param buffer in-memory buffer that stores the ONNX model bytes.
     *  @returns Network object that ready to do forward, throw an exception
     *        in failure cases.
     */
    CV_EXPORTS_W Net readNetFromONNX(const std::vector<uchar>& buffer);

     /** @brief This struct contains all the configuration of TensorRT backend.
      */
    struct CV_EXPORTS_W_SIMPLE TrtConfig
    {
        CV_PROP_RW int deviceId;      //!< Running TensorRT with specific GPU
        CV_PROP_RW bool useCache;     //!< It will speed up the ONNX model initialization. Write back the converted trt file to where the input model is located.
        CV_PROP_RW String cachePath;   //!< Setting the path to write back the converted .trt file to specific path instead of the input file location.
        CV_PROP_RW bool useFP16;      //!< Inference model with FP16, need hardware supports FP16, it may affect the accuracy.
        CV_PROP_RW bool useTimeCache; //!< To speed up the ONNX -> trt converting processing, it will write the time cache file to cachePath.
        CV_PROP_RW String inputName;            //!< Setting the input name and shape to convert dynamic shape ONNX to trt. Current only support single input.
        CV_PROP_RW std::vector<int> inputShape;//!< Setting the specific input shape.

        TrtConfig();

        // create Trt config by default config.
        TrtConfig(const TrtConfig& config);

        TrtConfig& operator=(const TrtConfig& config);
    };

    /** @brief Reads a trt model directly or <a href="https://onnx.ai/">ONNX</a>.
     *  @param trtFile path to the original .onnx file or converted .trt file.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     *  @param config model runing config
     *
     *  @note
     *  When the input is ONNX, it will take few mins to convert the ONNX model to trt first than to read it.
     */
    CV_EXPORTS_W Net readNetFromTensorRT(CV_WRAP_FILE_PATH const String &trtFile, const TrtConfig& config = TrtConfig());

    /** @brief Creates blob from .pb file.
     *  @param path to the .pb file with input tensor.
     *  @returns Mat.
     */
    CV_EXPORTS_W Mat readTensorFromONNX(CV_WRAP_FILE_PATH const String& path);

    /** @brief Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,
     *  subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels.
     *  @param image input image (with 1-, 3- or 4-channels).
     *  @param scalefactor multiplier for @p images values.
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @param crop flag which indicates whether image will be cropped after resize or not
     *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
     *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
     *  @returns 4-dimensional Mat with NCHW dimensions order.
     *
     * @note
     * The order and usage of `scalefactor` and `mean` are (input - mean) * scalefactor.
     */
    CV_EXPORTS_W Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                   int ddepth=CV_32F);

    /** @brief Creates 4-dimensional blob from image.
     *  @details This is an overloaded member function, provided for convenience.
     *           It differs from the above function only in what argument(s) it accepts.
     */
    CV_EXPORTS void blobFromImage(InputArray image, OutputArray blob, double scalefactor=1.0,
                                  const Size& size = Size(), const Scalar& mean = Scalar(),
                                  bool swapRB=false, bool crop=false, int ddepth=CV_32F);


    /** @brief Creates 4-dimensional blob from series of images. Optionally resizes and
     *  crops @p images from center, subtract @p mean values, scales values by @p scalefactor,
     *  swap Blue and Red channels.
     *  @param images input images (all with 1-, 3- or 4-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p images values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @param crop flag which indicates whether image will be cropped after resize or not
     *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
     *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
     *  @returns 4-dimensional Mat with NCHW dimensions order.
     *
     * @note
     * The order and usage of `scalefactor` and `mean` are (input - mean) * scalefactor.
     */
    CV_EXPORTS_W Mat blobFromImages(InputArrayOfArrays images, double scalefactor=1.0,
                                    Size size = Size(), const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                    int ddepth=CV_32F);

    /** @brief Creates 4-dimensional blob from series of images.
     *  @details This is an overloaded member function, provided for convenience.
     *           It differs from the above function only in what argument(s) it accepts.
     */
    CV_EXPORTS void blobFromImages(InputArrayOfArrays images, OutputArray blob,
                                   double scalefactor=1.0, Size size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                   int ddepth=CV_32F);

    /**
     * @brief Enum of image processing mode.
     * To facilitate the specialization pre-processing requirements of the dnn model.
     * For example, the `letter box` often used in the Yolo series of models.
     * @see Image2BlobParams
     */
    enum ImagePaddingMode
    {
        DNN_PMODE_NULL = 0,        // !< Default. Resize to required input size without extra processing.
        DNN_PMODE_CROP_CENTER = 1, // !< Image will be cropped after resize.
        DNN_PMODE_LETTERBOX = 2,   // !< Resize image to the desired size while preserving the aspect ratio of original image.
    };

    /** @brief Processing params of image to blob.
     *
     * It includes all possible image processing operations and corresponding parameters.
     *
     * @see blobFromImageWithParams
     *
     * @note
     * The order and usage of `scalefactor` and `mean` are (input - mean) * scalefactor.
     * The order and usage of `scalefactor`, `size`, `mean`, `swapRB`, and `ddepth` are consistent
     * with the function of @ref blobFromImage.
    */
    struct CV_EXPORTS_W_SIMPLE Image2BlobParams
    {
        CV_WRAP Image2BlobParams();
        CV_WRAP Image2BlobParams(const Scalar& scalefactor, const Size& size = Size(), const Scalar& mean = Scalar(),
                            bool swapRB = false, int ddepth = CV_32F, DataLayout datalayout = DNN_LAYOUT_NCHW,
                            ImagePaddingMode mode = DNN_PMODE_NULL, Scalar borderValue = 0.0);

        CV_PROP_RW Scalar scalefactor; //!< scalefactor multiplier for input image values.
        CV_PROP_RW Size size;    //!< Spatial size for output image.
        CV_PROP_RW Scalar mean;  //!< Scalar with mean values which are subtracted from channels.
        CV_PROP_RW bool swapRB;  //!< Flag which indicates that swap first and last channels
        CV_PROP_RW int ddepth;   //!< Depth of output blob. Choose CV_32F or CV_8U.
        CV_PROP_RW DataLayout datalayout; //!< Order of output dimensions. Choose DNN_LAYOUT_NCHW or DNN_LAYOUT_NHWC.
        CV_PROP_RW ImagePaddingMode paddingmode;   //!< Image padding mode. @see ImagePaddingMode.
        CV_PROP_RW Scalar borderValue;   //!< Value used in padding mode for padding.

        /** @brief Get rectangle coordinates in original image system from rectangle in blob coordinates.
         *  @param rBlob rect in blob coordinates.
         *  @param size original input image size.
         *  @returns rectangle in original image coordinates.
         */
        CV_WRAP Rect blobRectToImageRect(const Rect &rBlob, const Size &size);

        /** @brief Get rectangle coordinates in original image system from rectangle in blob coordinates.
         *  @param rBlob rect in blob coordinates.
         *  @param rImg result rect in image coordinates.
         *  @param size original input image size.
         */
        CV_WRAP void blobRectsToImageRects(const std::vector<Rect> &rBlob, CV_OUT std::vector<Rect>& rImg, const Size& size);
    };

    /** @brief Creates 4-dimensional blob from image with given params.
     *
     *  @details This function is an extension of @ref blobFromImage to meet more image preprocess needs.
     *  Given input image and preprocessing parameters, and function outputs the blob.
     *
     *  @param image input image (all with 1-, 3- or 4-channels).
     *  @param param struct of Image2BlobParams, contains all parameters needed by processing of image to blob.
     *  @return 4-dimensional Mat.
     */
    CV_EXPORTS_W Mat blobFromImageWithParams(InputArray image, const Image2BlobParams& param = Image2BlobParams());

    /** @overload */
    CV_EXPORTS_W void blobFromImageWithParams(InputArray image, OutputArray blob, const Image2BlobParams& param = Image2BlobParams());

    /** @brief Creates 4-dimensional blob from series of images with given params.
     *
     *  @details This function is an extension of @ref blobFromImages to meet more image preprocess needs.
     *  Given input image and preprocessing parameters, and function outputs the blob.
     *
     *  @param images input image (all with 1-, 3- or 4-channels).
     *  @param param struct of Image2BlobParams, contains all parameters needed by processing of image to blob.
     *  @returns 4-dimensional Mat.
     */
    CV_EXPORTS_W Mat blobFromImagesWithParams(InputArrayOfArrays images, const Image2BlobParams& param = Image2BlobParams());

    /** @overload */
    CV_EXPORTS_W void blobFromImagesWithParams(InputArrayOfArrays images, OutputArray blob, const Image2BlobParams& param = Image2BlobParams());

    /** @brief Parse a 4D blob and output the images it contains as 2D arrays through a simpler data structure
     *  (std::vector<cv::Mat>).
     *  @param[in] blob_ 4 dimensional array (images, channels, height, width) in floating point precision (CV_32F) from
     *  which you would like to extract the images.
     *  @param[out] images_ array of 2D Mat containing the images extracted from the blob in floating point precision
     *  (CV_32F). They are non normalized neither mean added. The number of returned images equals the first dimension
     *  of the blob (batch size). Every image has a number of channels equals to the second dimension of the blob (depth).
     */
    CV_EXPORTS_W void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_);

    /** @brief Convert all weights of Caffe network to half precision floating point.
     * @param src Path to origin model from Caffe framework contains single
     *            precision floating point weights (usually has `.caffemodel` extension).
     * @param dst Path to destination model with updated weights.
     * @param layersTypes Set of layers types which parameters will be converted.
     *                    By default, converts only Convolutional and Fully-Connected layers'
     *                    weights.
     *
     * @note Shrinked model has no origin float32 weights so it can't be used
     *       in origin Caffe framework anymore. However the structure of data
     *       is taken from NVidia's Caffe fork: https://github.com/NVIDIA/caffe.
     *       So the resulting model may be used there.
     */
    CV_EXPORTS_W void shrinkCaffeModel(CV_WRAP_FILE_PATH const String& src, CV_WRAP_FILE_PATH const String& dst,
                                       const std::vector<String>& layersTypes = std::vector<String>());

    /** @brief Create a text representation for a binary network stored in protocol buffer format.
     *  @param[in] model  A path to binary network.
     *  @param[in] output A path to output text file to be created.
     *
     *  @note To reduce output file size, trained weights are not included.
     */
    CV_EXPORTS_W void writeTextGraph(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& output);

    /** @brief Performs non maximum suppression given boxes and corresponding scores.

     * @param bboxes a set of bounding boxes to apply NMS.
     * @param scores a set of corresponding confidences.
     * @param score_threshold a threshold used to filter boxes by score.
     * @param nms_threshold a threshold used in non maximum suppression.
     * @param indices the kept indices of bboxes after NMS.
     * @param eta a coefficient in adaptive threshold formula: \f$nms\_threshold_{i+1}=eta\cdot nms\_threshold_i\f$.
     * @param top_k if `>0`, keep at most @p top_k picked indices.
     */
    CV_EXPORTS void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                               const float score_threshold, const float nms_threshold,
                               CV_OUT std::vector<int>& indices,
                               const float eta = 1.f, const int top_k = 0);

    CV_EXPORTS_W void NMSBoxes(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores,
                               const float score_threshold, const float nms_threshold,
                               CV_OUT std::vector<int>& indices,
                               const float eta = 1.f, const int top_k = 0);

    CV_EXPORTS_AS(NMSBoxesRotated) void NMSBoxes(const std::vector<RotatedRect>& bboxes, const std::vector<float>& scores,
                             const float score_threshold, const float nms_threshold,
                             CV_OUT std::vector<int>& indices,
                             const float eta = 1.f, const int top_k = 0);

    /** @brief Performs batched non maximum suppression on given boxes and corresponding scores across different classes.

     * @param bboxes a set of bounding boxes to apply NMS.
     * @param scores a set of corresponding confidences.
     * @param class_ids a set of corresponding class ids. Ids are integer and usually start from 0.
     * @param score_threshold a threshold used to filter boxes by score.
     * @param nms_threshold a threshold used in non maximum suppression.
     * @param indices the kept indices of bboxes after NMS.
     * @param eta a coefficient in adaptive threshold formula: \f$nms\_threshold_{i+1}=eta\cdot nms\_threshold_i\f$.
     * @param top_k if `>0`, keep at most @p top_k picked indices.
     */
    CV_EXPORTS void NMSBoxesBatched(const std::vector<Rect>& bboxes, const std::vector<float>& scores, const std::vector<int>& class_ids,
                                    const float score_threshold, const float nms_threshold,
                                    CV_OUT std::vector<int>& indices,
                                    const float eta = 1.f, const int top_k = 0);

    CV_EXPORTS_W void NMSBoxesBatched(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores, const std::vector<int>& class_ids,
                                      const float score_threshold, const float nms_threshold,
                                      CV_OUT std::vector<int>& indices,
                                      const float eta = 1.f, const int top_k = 0);

    /**
     * @brief Enum of Soft NMS methods.
     * @see softNMSBoxes
     */
    enum class SoftNMSMethod
    {
        SOFTNMS_LINEAR = 1,
        SOFTNMS_GAUSSIAN = 2
    };

    /** @brief Performs soft non maximum suppression given boxes and corresponding scores.
     * Reference: https://arxiv.org/abs/1704.04503
     * @param bboxes a set of bounding boxes to apply Soft NMS.
     * @param scores a set of corresponding confidences.
     * @param updated_scores a set of corresponding updated confidences.
     * @param score_threshold a threshold used to filter boxes by score.
     * @param nms_threshold a threshold used in non maximum suppression.
     * @param indices the kept indices of bboxes after NMS.
     * @param top_k keep at most @p top_k picked indices.
     * @param sigma parameter of Gaussian weighting.
     * @param method Gaussian or linear.
     * @see SoftNMSMethod
     */
    CV_EXPORTS_W void softNMSBoxes(const std::vector<Rect>& bboxes,
                                   const std::vector<float>& scores,
                                   CV_OUT std::vector<float>& updated_scores,
                                   const float score_threshold,
                                   const float nms_threshold,
                                   CV_OUT std::vector<int>& indices,
                                   size_t top_k = 0,
                                   const float sigma = 0.5,
                                   SoftNMSMethod method = SoftNMSMethod::SOFTNMS_GAUSSIAN);


     /** @brief This class is presented high-level API for neural networks.
      *
      * Model allows to set params for preprocessing input image.
      * Model creates net from file with trained weights and config,
      * sets preprocessing input and runs forward pass.
      */
     class CV_EXPORTS_W_SIMPLE Model
     {
     public:
         CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
         Model();

         Model(const Model&) = default;
         Model(Model&&) = default;
         Model& operator=(const Model&) = default;
         Model& operator=(Model&&) = default;

         /**
          * @brief Create model from deep learning network represented in one of the supported formats.
          * An order of @p model and @p config arguments does not matter.
          * @param[in] model Binary file contains trained weights.
          * @param[in] config Text file contains network configuration.
          */
         CV_WRAP Model(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config = "");

         /**
          * @brief Create model from deep learning network.
          * @param[in] network Net object.
          */
         CV_WRAP Model(const Net& network);

         /** @brief Set input size for frame.
          *  @param[in] size New input size.
          *  @note If shape of the new blob less than 0, then frame size not change.
         */
         CV_WRAP Model& setInputSize(const Size& size);

         /** @overload
         *  @param[in] width New input width.
         *  @param[in] height New input height.
         */
         CV_WRAP inline
         Model& setInputSize(int width, int height) { return setInputSize(Size(width, height)); }

         /** @brief Set mean value for frame.
          *  @param[in] mean Scalar with mean values which are subtracted from channels.
         */
         CV_WRAP Model& setInputMean(const Scalar& mean);

         /** @brief Set scalefactor value for frame.
          *  @param[in] scale Multiplier for frame values.
         */
         CV_WRAP Model& setInputScale(const Scalar& scale);

         /** @brief Set flag crop for frame.
          *  @param[in] crop Flag which indicates whether image will be cropped after resize or not.
         */
         CV_WRAP Model& setInputCrop(bool crop);

         /** @brief Set flag swapRB for frame.
          *  @param[in] swapRB Flag which indicates that swap first and last channels.
         */
         CV_WRAP Model& setInputSwapRB(bool swapRB);

         /** @brief Set output names for frame.
          *  @param[in] outNames Names for output layers.
         */
         CV_WRAP Model& setOutputNames(const std::vector<String>& outNames);

         /** @brief Set preprocessing parameters for frame.
         *  @param[in] size New input size.
         *  @param[in] mean Scalar with mean values which are subtracted from channels.
         *  @param[in] scale Multiplier for frame values.
         *  @param[in] swapRB Flag which indicates that swap first and last channels.
         *  @param[in] crop Flag which indicates whether image will be cropped after resize or not.
         *  blob(n, c, y, x) = scale * resize( frame(y, x, c) ) - mean(c) )
         */
         CV_WRAP void setInputParams(double scale = 1.0, const Size& size = Size(),
                                     const Scalar& mean = Scalar(), bool swapRB = false, bool crop = false);

         /** @brief Given the @p input frame, create input blob, run net and return the output @p blobs.
          *  @param[in]  frame  The input image.
          *  @param[out] outs Allocated output blobs, which will store results of the computation.
          */
         CV_WRAP void predict(InputArray frame, OutputArrayOfArrays outs) const;


         // ============================== Net proxy methods ==============================
         // Never expose methods with network implementation details, like:
         // - addLayer, addLayerToPrev, connect, setInputsNames, setInputShape, setParam, getParam
         // - getLayer*, getUnconnectedOutLayers, getUnconnectedOutLayersNames, getLayersShapes
         // - forward* methods, setInput

         /// @sa Net::setPreferableBackend
         CV_WRAP Model& setPreferableBackend(dnn::Backend backendId);
         /// @sa Net::setPreferableTarget
         CV_WRAP Model& setPreferableTarget(dnn::Target targetId);

         /// @sa Net::enableWinograd
         CV_WRAP Model& enableWinograd(bool useWinograd);

         CV_DEPRECATED_EXTERNAL
         operator Net&() const { return getNetwork_(); }

     //protected: - internal/tests usage only
         Net& getNetwork_() const;
         inline Net& getNetwork_() { return const_cast<const Model*>(this)->getNetwork_(); }

         struct Impl;
         inline Impl* getImpl() const { return impl.get(); }
         inline Impl& getImplRef() const { CV_DbgAssert(impl); return *impl.get(); }
     protected:
         Ptr<Impl> impl;
     };

     /** @brief This class represents high-level API for classification models.
      *
      * ClassificationModel allows to set params for preprocessing input image.
      * ClassificationModel creates net from file with trained weights and config,
      * sets preprocessing input, runs forward pass and return top-1 prediction.
      */
     class CV_EXPORTS_W_SIMPLE ClassificationModel : public Model
     {
     public:
         CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
         ClassificationModel();

         /**
          * @brief Create classification model from network represented in one of the supported formats.
          * An order of @p model and @p config arguments does not matter.
          * @param[in] model Binary file contains trained weights.
          * @param[in] config Text file contains network configuration.
          */
          CV_WRAP ClassificationModel(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config = "");

         /**
          * @brief Create model from deep learning network.
          * @param[in] network Net object.
          */
         CV_WRAP ClassificationModel(const Net& network);

         /**
          * @brief Set enable/disable softmax post processing option.
          *
          * If this option is true, softmax is applied after forward inference within the classify() function
          * to convert the confidences range to [0.0-1.0].
          * This function allows you to toggle this behavior.
          * Please turn true when not contain softmax layer in model.
          * @param[in] enable Set enable softmax post processing within the classify() function.
          */
         CV_WRAP ClassificationModel& setEnableSoftmaxPostProcessing(bool enable);

         /**
          * @brief Get enable/disable softmax post processing option.
          *
          * This option defaults to false, softmax post processing is not applied within the classify() function.
          */
         CV_WRAP bool getEnableSoftmaxPostProcessing() const;

         /** @brief Given the @p input frame, create input blob, run net and return top-1 prediction.
          *  @param[in]  frame  The input image.
          */
         std::pair<int, float> classify(InputArray frame);

         /** @overload */
         CV_WRAP void classify(InputArray frame, CV_OUT int& classId, CV_OUT float& conf);
     };

     /** @brief This class represents high-level API for keypoints models
      *
      * KeypointsModel allows to set params for preprocessing input image.
      * KeypointsModel creates net from file with trained weights and config,
      * sets preprocessing input, runs forward pass and returns the x and y coordinates of each detected keypoint
      */
     class CV_EXPORTS_W_SIMPLE KeypointsModel: public Model
     {
     public:
         /**
          * @brief Create keypoints model from network represented in one of the supported formats.
          * An order of @p model and @p config arguments does not matter.
          * @param[in] model Binary file contains trained weights.
          * @param[in] config Text file contains network configuration.
          */
          CV_WRAP KeypointsModel(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config = "");

         /**
          * @brief Create model from deep learning network.
          * @param[in] network Net object.
          */
         CV_WRAP KeypointsModel(const Net& network);

         /** @brief Given the @p input frame, create input blob, run net
          *  @param[in]  frame  The input image.
          *  @param thresh minimum confidence threshold to select a keypoint
          *  @returns a vector holding the x and y coordinates of each detected keypoint
          *
          */
         CV_WRAP std::vector<Point2f> estimate(InputArray frame, float thresh=0.5);
     };

     /** @brief This class represents high-level API for segmentation  models
      *
      * SegmentationModel allows to set params for preprocessing input image.
      * SegmentationModel creates net from file with trained weights and config,
      * sets preprocessing input, runs forward pass and returns the class prediction for each pixel.
      */
     class CV_EXPORTS_W_SIMPLE SegmentationModel: public Model
     {
     public:
         /**
          * @brief Create segmentation model from network represented in one of the supported formats.
          * An order of @p model and @p config arguments does not matter.
          * @param[in] model Binary file contains trained weights.
          * @param[in] config Text file contains network configuration.
          */
          CV_WRAP SegmentationModel(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config = "");

         /**
          * @brief Create model from deep learning network.
          * @param[in] network Net object.
          */
         CV_WRAP SegmentationModel(const Net& network);

         /** @brief Given the @p input frame, create input blob, run net
          *  @param[in]  frame  The input image.
          *  @param[out] mask Allocated class prediction for each pixel
          */
         CV_WRAP void segment(InputArray frame, OutputArray mask);
     };

     /** @brief This class represents high-level API for object detection networks.
      *
      * DetectionModel allows to set params for preprocessing input image.
      * DetectionModel creates net from file with trained weights and config,
      * sets preprocessing input, runs forward pass and return result detections.
      * For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.
      */
     class CV_EXPORTS_W_SIMPLE DetectionModel : public Model
     {
     public:
         /**
          * @brief Create detection model from network represented in one of the supported formats.
          * An order of @p model and @p config arguments does not matter.
          * @param[in] model Binary file contains trained weights.
          * @param[in] config Text file contains network configuration.
          */
         CV_WRAP DetectionModel(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config = "");

         /**
          * @brief Create model from deep learning network.
          * @param[in] network Net object.
          */
         CV_WRAP DetectionModel(const Net& network);

         CV_DEPRECATED_EXTERNAL  // avoid using in C++ code (need to fix bindings first)
         DetectionModel();

         /**
          * @brief nmsAcrossClasses defaults to false,
          * such that when non max suppression is used during the detect() function, it will do so per-class.
          * This function allows you to toggle this behaviour.
          * @param[in] value The new value for nmsAcrossClasses
          */
         CV_WRAP DetectionModel& setNmsAcrossClasses(bool value);

         /**
          * @brief Getter for nmsAcrossClasses. This variable defaults to false,
          * such that when non max suppression is used during the detect() function, it will do so only per-class
          */
         CV_WRAP bool getNmsAcrossClasses();

         /** @brief Given the @p input frame, create input blob, run net and return result detections.
          *  @param[in]  frame  The input image.
          *  @param[out] classIds Class indexes in result detection.
          *  @param[out] confidences A set of corresponding confidences.
          *  @param[out] boxes A set of bounding boxes.
          *  @param[in] confThreshold A threshold used to filter boxes by confidences.
          *  @param[in] nmsThreshold A threshold used in non maximum suppression.
          */
         CV_WRAP void detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                             CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect>& boxes,
                             float confThreshold = 0.5f, float nmsThreshold = 0.0f);
     };


/** @brief This class represents high-level API for text recognition networks.
 *
 * TextRecognitionModel allows to set params for preprocessing input image.
 * TextRecognitionModel creates net from file with trained weights and config,
 * sets preprocessing input, runs forward pass and return recognition result.
 * For TextRecognitionModel, CRNN-CTC is supported.
 */
class CV_EXPORTS_W_SIMPLE TextRecognitionModel : public Model
{
public:
    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    TextRecognitionModel();

    /**
     * @brief Create Text Recognition model from deep learning network
     * Call setDecodeType() and setVocabulary() after constructor to initialize the decoding method
     * @param[in] network Net object
     */
    CV_WRAP TextRecognitionModel(const Net& network);

    /**
     * @brief Create text recognition model from network represented in one of the supported formats
     * Call setDecodeType() and setVocabulary() after constructor to initialize the decoding method
     * @param[in] model Binary file contains trained weights
     * @param[in] config Text file contains network configuration
     */
    CV_WRAP inline
    TextRecognitionModel(CV_WRAP_FILE_PATH const std::string& model, CV_WRAP_FILE_PATH const std::string& config = "")
        : TextRecognitionModel(readNet(model, config)) { /* nothing */ }

    /**
     * @brief Set the decoding method of translating the network output into string
     * @param[in] decodeType The decoding method of translating the network output into string, currently supported type:
     *    - `"CTC-greedy"` greedy decoding for the output of CTC-based methods
     *    - `"CTC-prefix-beam-search"` Prefix beam search decoding for the output of CTC-based methods
     */
    CV_WRAP
    TextRecognitionModel& setDecodeType(const std::string& decodeType);

    /**
     * @brief Get the decoding method
     * @return the decoding method
     */
    CV_WRAP
    const std::string& getDecodeType() const;

    /**
     * @brief Set the decoding method options for `"CTC-prefix-beam-search"` decode usage
     * @param[in] beamSize Beam size for search
     * @param[in] vocPruneSize Parameter to optimize big vocabulary search,
     * only take top @p vocPruneSize tokens in each search step, @p vocPruneSize <= 0 stands for disable this prune.
     */
    CV_WRAP
    TextRecognitionModel& setDecodeOptsCTCPrefixBeamSearch(int beamSize, int vocPruneSize = 0);

    /**
     * @brief Set the vocabulary for recognition.
     * @param[in] vocabulary the associated vocabulary of the network.
     */
    CV_WRAP
    TextRecognitionModel& setVocabulary(const std::vector<std::string>& vocabulary);

    /**
     * @brief Get the vocabulary for recognition.
     * @return vocabulary the associated vocabulary
     */
    CV_WRAP
    const std::vector<std::string>& getVocabulary() const;

    /**
     * @brief Given the @p input frame, create input blob, run net and return recognition result
     * @param[in] frame The input image
     * @return The text recognition result
     */
    CV_WRAP
    std::string recognize(InputArray frame) const;

    /**
     * @brief Given the @p input frame, create input blob, run net and return recognition result
     * @param[in] frame The input image
     * @param[in] roiRects List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
     * @param[out] results A set of text recognition results.
     */
    CV_WRAP
    void recognize(InputArray frame, InputArrayOfArrays roiRects, CV_OUT std::vector<std::string>& results) const;
};


/** @brief Base class for text detection networks
 */
class CV_EXPORTS_W_SIMPLE TextDetectionModel : public Model
{
protected:
    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    TextDetectionModel();

public:

    /** @brief Performs detection
     *
     * Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections.
     *
     * Each result is quadrangle's 4 points in this order:
     * - bottom-left
     * - top-left
     * - top-right
     * - bottom-right
     *
     * Use cv::getPerspectiveTransform function to retrieve image region without perspective transformations.
     *
     * @note If DL model doesn't support that kind of output then result may be derived from detectTextRectangles() output.
     *
     * @param[in] frame The input image
     * @param[out] detections array with detections' quadrangles (4 points per result)
     * @param[out] confidences array with detection confidences
     */
    CV_WRAP
    void detect(
            InputArray frame,
            CV_OUT std::vector< std::vector<Point> >& detections,
            CV_OUT std::vector<float>& confidences
    ) const;

    /** @overload */
    CV_WRAP
    void detect(
            InputArray frame,
            CV_OUT std::vector< std::vector<Point> >& detections
    ) const;

    /** @brief Performs detection
     *
     * Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections.
     *
     * Each result is rotated rectangle.
     *
     * @note Result may be inaccurate in case of strong perspective transformations.
     *
     * @param[in] frame the input image
     * @param[out] detections array with detections' RotationRect results
     * @param[out] confidences array with detection confidences
     */
    CV_WRAP
    void detectTextRectangles(
            InputArray frame,
            CV_OUT std::vector<cv::RotatedRect>& detections,
            CV_OUT std::vector<float>& confidences
    ) const;

    /** @overload */
    CV_WRAP
    void detectTextRectangles(
            InputArray frame,
            CV_OUT std::vector<cv::RotatedRect>& detections
    ) const;
};

/** @brief This class represents high-level API for text detection DL networks compatible with EAST model.
 *
 * Configurable parameters:
 * - (float) confThreshold - used to filter boxes by confidences, default: 0.5f
 * - (float) nmsThreshold - used in non maximum suppression, default: 0.0f
 */
class CV_EXPORTS_W_SIMPLE TextDetectionModel_EAST : public TextDetectionModel
{
public:
    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    TextDetectionModel_EAST();

    /**
     * @brief Create text detection algorithm from deep learning network
     * @param[in] network Net object
     */
    CV_WRAP TextDetectionModel_EAST(const Net& network);

    /**
     * @brief Create text detection model from network represented in one of the supported formats.
     * An order of @p model and @p config arguments does not matter.
     * @param[in] model Binary file contains trained weights.
     * @param[in] config Text file contains network configuration.
     */
    CV_WRAP inline
    TextDetectionModel_EAST(CV_WRAP_FILE_PATH const std::string& model, CV_WRAP_FILE_PATH const std::string& config = "")
        : TextDetectionModel_EAST(readNet(model, config)) { /* nothing */ }

    /**
     * @brief Set the detection confidence threshold
     * @param[in] confThreshold A threshold used to filter boxes by confidences
     */
    CV_WRAP
    TextDetectionModel_EAST& setConfidenceThreshold(float confThreshold);

    /**
     * @brief Get the detection confidence threshold
     */
    CV_WRAP
    float getConfidenceThreshold() const;

    /**
     * @brief Set the detection NMS filter threshold
     * @param[in] nmsThreshold A threshold used in non maximum suppression
     */
    CV_WRAP
    TextDetectionModel_EAST& setNMSThreshold(float nmsThreshold);

    /**
     * @brief Get the detection confidence threshold
     */
    CV_WRAP
    float getNMSThreshold() const;
};

/** @brief This class represents high-level API for text detection DL networks compatible with DB model.
 *
 * Related publications: @cite liao2020real
 * Paper: https://arxiv.org/abs/1911.08947
 * For more information about the hyper-parameters setting, please refer to https://github.com/MhLiao/DB
 *
 * Configurable parameters:
 * - (float) binaryThreshold - The threshold of the binary map. It is usually set to 0.3.
 * - (float) polygonThreshold - The threshold of text polygons. It is usually set to 0.5, 0.6, and 0.7. Default is 0.5f
 * - (double) unclipRatio - The unclip ratio of the detected text region, which determines the output size. It is usually set to 2.0.
 * - (int) maxCandidates - The max number of the output results.
 */
class CV_EXPORTS_W_SIMPLE TextDetectionModel_DB : public TextDetectionModel
{
public:
    CV_DEPRECATED_EXTERNAL  // avoid using in C++ code, will be moved to "protected" (need to fix bindings first)
    TextDetectionModel_DB();

    /**
     * @brief Create text detection algorithm from deep learning network.
     * @param[in] network Net object.
     */
    CV_WRAP TextDetectionModel_DB(const Net& network);

    /**
     * @brief Create text detection model from network represented in one of the supported formats.
     * An order of @p model and @p config arguments does not matter.
     * @param[in] model Binary file contains trained weights.
     * @param[in] config Text file contains network configuration.
     */
    CV_WRAP inline
    TextDetectionModel_DB(CV_WRAP_FILE_PATH const std::string& model, CV_WRAP_FILE_PATH const std::string& config = "")
        : TextDetectionModel_DB(readNet(model, config)) { /* nothing */ }

    CV_WRAP TextDetectionModel_DB& setBinaryThreshold(float binaryThreshold);
    CV_WRAP float getBinaryThreshold() const;

    CV_WRAP TextDetectionModel_DB& setPolygonThreshold(float polygonThreshold);
    CV_WRAP float getPolygonThreshold() const;

    CV_WRAP TextDetectionModel_DB& setUnclipRatio(double unclipRatio);
    CV_WRAP double getUnclipRatio() const;

    CV_WRAP TextDetectionModel_DB& setMaxCandidates(int maxCandidates);
    CV_WRAP int getMaxCandidates() const;
};

//! @}
CV__DNN_INLINE_NS_END
}
}

#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/dnn.inl.hpp>

/// @deprecated Include this header directly from application. Automatic inclusion will be removed
#include <opencv2/dnn/utils/inference_engine.hpp>

#endif  /* OPENCV_DNN_DNN_HPP */
