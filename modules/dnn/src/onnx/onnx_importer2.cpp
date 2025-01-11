// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/layer_reg.private.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <opencv2/core/utils/configuration.private.hpp>

#ifdef HAVE_PROTOBUF

#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <limits>
#include <set>
#include <string>

#if defined _MSC_VER && _MSC_VER < 1910/*MSVS 2017*/
#pragma warning(push)
#pragma warning(disable: 4503)  // decorated name length exceeded, name was truncated
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

#include "onnx_graph_simplifier.hpp"
#endif
namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

extern bool DNN_DIAGNOSTICS_RUN;

#ifdef HAVE_PROTOBUF

template <typename T>
static T getScalarFromMat(Mat m)
{
    CV_Assert(m.total() == 1);
    return m.at<T>(0);
}

static int dataType2cv(opencv_onnx::TensorProto_DataType dt)
{
    return
        dt == opencv_onnx::TensorProto_DataType_UINT8 ? CV_8U :
        dt == opencv_onnx::TensorProto_DataType_INT8 ? CV_8S :
        dt == opencv_onnx::TensorProto_DataType_UINT16 ? CV_16U :
        dt == opencv_onnx::TensorProto_DataType_INT16 ? CV_16S :
        dt == opencv_onnx::TensorProto_DataType_UINT32 ? CV_32U :
        dt == opencv_onnx::TensorProto_DataType_INT32 ? CV_32S :
        dt == opencv_onnx::TensorProto_DataType_UINT64 ? CV_64U :
        dt == opencv_onnx::TensorProto_DataType_INT64 ? CV_64S :
        dt == opencv_onnx::TensorProto_DataType_FLOAT ? CV_32F :
        dt == opencv_onnx::TensorProto_DataType_DOUBLE ? CV_64F :
        dt == opencv_onnx::TensorProto_DataType_FLOAT16 ? CV_16F :
        dt == opencv_onnx::TensorProto_DataType_COMPLEX64 ? CV_32FC2 :
        dt == opencv_onnx::TensorProto_DataType_COMPLEX128 ? CV_64FC2 :
        dt == opencv_onnx::TensorProto_DataType_BOOL ? CV_Bool : -1;
}

static std::string dataType2str(opencv_onnx::TensorProto_DataType dt)
{
    const char* str =
    dt == opencv_onnx::TensorProto_DataType_UNDEFINED ? "UNDEFINED" :
    dt == opencv_onnx::TensorProto_DataType_STRING ? "STRING" :
    dt == opencv_onnx::TensorProto_DataType_UINT8 ? "UINT8" :
    dt == opencv_onnx::TensorProto_DataType_INT8 ? "INT8" :
    dt == opencv_onnx::TensorProto_DataType_UINT16 ? "UINT16" :
    dt == opencv_onnx::TensorProto_DataType_INT16 ? "INT16" :
    dt == opencv_onnx::TensorProto_DataType_UINT32 ? "UINT32" :
    dt == opencv_onnx::TensorProto_DataType_INT32 ? "INT32" :
    dt == opencv_onnx::TensorProto_DataType_UINT64 ? "UINT64" :
    dt == opencv_onnx::TensorProto_DataType_INT64 ? "INT64" :
    dt == opencv_onnx::TensorProto_DataType_FLOAT ? "FLOAT" :
    dt == opencv_onnx::TensorProto_DataType_FLOAT16 ? "FLOAT16" :
    dt == opencv_onnx::TensorProto_DataType_BOOL ? "BOOL" :
    dt == opencv_onnx::TensorProto_DataType_COMPLEX64 ? "COMPLEX64" :
    dt == opencv_onnx::TensorProto_DataType_COMPLEX128 ? "COMPLEX128" : nullptr;
    if (!str)
        return format("<unknown_type #%d>", (int)dt);
    return std::string(str);
}

static Mat getMatFromTensor2(const opencv_onnx::TensorProto& tensor_proto)
{
    Mat m = getMatFromTensor(tensor_proto, false);
    m.dims = (int)tensor_proto.dims_size();
    return m;
}

class ONNXImporter2
{
public:
    ONNXImporter2();

    Net parseFile(const char *onnxFile);
    Net parseBuffer(const void* buffer, size_t sizeBuffer);

protected:
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;
    opencv_onnx::ModelProto model_proto;

    Net parseModel();
    Ptr<Graph> parseGraph(opencv_onnx::GraphProto* graph_proto, bool mainGraph);
    void parseNode(const opencv_onnx::NodeProto& node_proto);
    bool parseValueInfo(const opencv_onnx::ValueInfoProto& valueInfoProto, ArgData& data);
    Mat parseTensor(const opencv_onnx::TensorProto& tensorProto);
    void rememberMissingOp(const std::string& opname);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);

    void addLayer(LayerParams& layerParams,
                  const opencv_onnx::NodeProto& node_proto,
                  int max_inputs = std::numeric_limits<int>::max());
    void setParamsDtype(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    void raiseError() {
        have_errors = true;
    }

    Net net;
    Net::Impl* netimpl;
    std::string onnxFilename;
    Ptr<Graph> curr_graph;
    opencv_onnx::GraphProto* curr_graph_proto;
    std::vector<Ptr<Layer> > curr_prog;
    std::vector<Arg> node_inputs, node_outputs;

    std::string framework_name;
    std::set<std::string> missing_ops;

    // Used when Onnx does not contain node names.
    // In this case each node is assigned a name 'onnx_node!<current global_node_idx value>'
    int global_node_idx;
    bool have_errors;

    typedef void (ONNXImporter2::*ONNXImporterNodeParser)(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    typedef std::map<std::string, ONNXImporterNodeParser> DispatchMap;
    typedef std::map<std::string, DispatchMap> DomainDispatchMap;

    DomainDispatchMap domain_dispatch_map;
    std::string getLayerTypeDomain(const opencv_onnx::NodeProto& node_proto);
    const DispatchMap& getDispatchMap(const opencv_onnx::NodeProto& node_proto);
    void buildDispatchMap_ONNX_AI(int opset_version);
    void buildDispatchMap_COM_MICROSOFT(int opset_version);

    // Domain: 'ai.onnx' (default)
    // URL: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    void parseAbs                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseArgMinMax            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseAveragePool          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseBatchNormalization   (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseCast                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseClip                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConcat               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConstant             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConstantOfShape      (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConv                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConvTranspose        (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseCumSum               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseDepthSpaceOps        (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseDetectionOutput      (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseEinsum               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseElementWise          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseElu                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseExpand               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseFlatten              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGather               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGatherElements       (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGemm                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGlobalPool           (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGRU                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseImageScaler          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseInstanceNormalization(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLayerNorm            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLeakyRelu            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLRN                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLSTM                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseMatMul               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseMaxPool              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseMaxUnpool            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseNeg                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parsePad                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parsePRelu                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseRange                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseReduce               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseRelu                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseResize               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseReshape              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseScatter              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseShape                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSimpleLayers         (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSlice                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSoftMax              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSplit                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSqueeze              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTanh                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTile                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTranspose            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseUnsqueeze            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseUpsample             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTopK2                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    // Domain: com.microsoft
    // URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
    void parseAttention            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseDequantizeLinear     (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQuantizeLinear       (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseCustomLayer          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQAvgPool             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQConcat              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQConv                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQEltwise             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQGemm                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQLeakyRelu           (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQMatMul              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQSigmoid             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    //void parseQSoftmax             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    int onnx_opset;  // OperatorSetIdProto for 'onnx' domain
    std::map<std::string, int> onnx_opset_map;  // map from OperatorSetIdProto
    void parseOperatorSet();

    const std::string str_domain_ai_onnx = "ai.onnx";

    bool useLegacyNames;
    bool getParamUseLegacyNames()
    {
        //bool param = utils::getConfigurationParameterBool("OPENCV_DNN_ONNX_USE_LEGACY_NAMES", false);
        //return param;
        return true;
    }
    std::string extractNodeName(const opencv_onnx::NodeProto& node_proto);
};

ONNXImporter2::ONNXImporter2() :
    onnx_opset(0),
    useLegacyNames(getParamUseLegacyNames())
{
    netimpl = net.getImpl();
}

Net ONNXImporter2::parseFile(const char *onnxFilename_)
{
    CV_Assert(onnxFilename_);
    onnxFilename = onnxFilename_;
    CV_LOG_DEBUG(NULL, "DNN/ONNX: processing ONNX model from file: " << onnxFilename);

    std::fstream input(onnxFilename, std::ios::in | std::ios::binary);
    if (!input)
    {
        CV_Error(Error::StsBadArg, format("Can't read ONNX file: %s", onnxFilename_));
    }

    if (!model_proto.ParseFromIstream(&input))
    {
        CV_Error(Error::StsUnsupportedFormat, format("Failed to parse ONNX model: %s", onnxFilename_));
    }

    return parseModel();
}

Net ONNXImporter2::parseBuffer(const void* buffer, size_t sizeBuffer)
{
    onnxFilename = std::string();
    CV_LOG_DEBUG(NULL, "DNN/ONNX: processing in-memory ONNX model (" << sizeBuffer << " bytes)");

    struct _Buf: public std::streambuf
    {
        _Buf(const void* buffer, size_t sizeBuffer)
        {
            char* p = (char*)buffer;
            setg(p, p, p + sizeBuffer);
        }
    };

    _Buf buf(buffer, sizeBuffer);
    std::istream input(&buf);

    if (!model_proto.ParseFromIstream(&input))
        CV_Error(Error::StsUnsupportedFormat, "Failed to parse onnx model from in-memory byte array.");

    return parseModel();
}


inline void replaceLayerParam(LayerParams& layerParams, const String& oldKey, const String& newKey)
{
    if (layerParams.has(oldKey)) {
        layerParams.set(newKey, layerParams.get(oldKey));
        layerParams.erase(oldKey);
    }
}

/*static void runLayer(LayerParams& params, const std::vector<Mat>& inputs,
              std::vector<Mat>& outputs)
{
    Ptr<Layer> layer = LayerFactory::createLayerInstance(params.type, params);
    CV_Assert((bool)layer);

    std::vector<MatShape> inpShapes(inputs.size());
    std::vector<MatType> inpTypes(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inpShapes[i] = shape(inputs[i]);
        inpTypes[i] = inputs[i].type();
    }

    std::vector<MatShape> outShapes, internalShapes;
    std::vector<MatType> outTypes, internalTypes;
    layer->getMemoryShapes(inpShapes, 0, outShapes, internalShapes);
    layer->getTypes(inpTypes, outShapes.size(), internalShapes.size(), outTypes, internalTypes);

    std::vector<Mat> internals(internalShapes.size());
    outputs.resize(outShapes.size());
    for (size_t i = 0; i < outShapes.size(); ++i)
        outputs[i].create(outShapes[i], outTypes[i]);
    for (size_t i = 0; i < internalShapes.size(); ++i)
        internals[i].create(internalShapes[i], internalTypes[i]);

    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);
}*/

/*std::map<std::string, Mat> ONNXImporter2::getGraphTensors(
                                        const opencv_onnx::GraphProto& graph_proto)
{
    std::map<std::string, Mat> layers_weights;

    for (int i = 0; i < graph_proto.initializer_size(); i++)
    {
        const opencv_onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
        dumpTensorProto(i, tensor_proto, "initializer");
        Mat mat = getMatFromTensor2(tensor_proto);
        releaseONNXTensor(const_cast<opencv_onnx::TensorProto&>(tensor_proto));  // drop already loaded data

        if (DNN_DIAGNOSTICS_RUN && mat.empty())
            continue;

        layers_weights.insert(std::make_pair(tensor_proto.name(), mat));
        constBlobsExtraInfo.insert(std::make_pair(tensor_proto.name(), TensorInfo(tensor_proto.dims_size())));
    }
    return layers_weights;
}*/

static DictValue parse(const ::google::protobuf::RepeatedField< ::google::protobuf::int64>& src) {
    std::vector<int32_t> dst(src.size());
    convertInt64ToInt32(src, dst, src.size());
    return DictValue::arrayInt(&dst[0], src.size());
}

static DictValue parseStr(const ::google::protobuf::RepeatedPtrField< ::std::string>& src) {
    return DictValue::arrayString(src.begin(), static_cast<int>(src.size()));
}

LayerParams ONNXImporter2::getLayerParams(const opencv_onnx::NodeProto& node_proto)
{
    LayerParams lp;
    for(int i = 0; i < node_proto.attribute_size(); i++)
    {
        opencv_onnx::AttributeProto attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();

        try
        {
            if(attribute_name == "kernel_shape")
            {
                CV_Assert(attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("kernel_size", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "strides")
            {
                CV_Assert(attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("stride", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "pads")
            {
                if (node_proto.op_type() == "Pad")
                {
                    // Padding layer.
                    // Paddings are in order begin0, begin1, .. beginN, end0, end1, ..., endN.
                    // We need to shuffle it to begin0, end0, begin1, end1, ...
                    CV_Assert(attribute_proto.ints_size() % 2 == 0);
                    const int dims = attribute_proto.ints_size() / 2;
                    std::vector<int32_t> paddings;
                    paddings.reserve(attribute_proto.ints_size());
                    for (int i = 0; i < dims; ++i)
                    {
                        paddings.push_back(attribute_proto.ints(i));
                        paddings.push_back(attribute_proto.ints(dims + i));
                    }
                    lp.set("paddings", DictValue::arrayInt(&paddings[0], paddings.size()));
                }
                else
                {
                    // Convolution or pooling.
                    CV_Assert(attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 4 || attribute_proto.ints_size() == 6);
                    lp.set("pad", parse(attribute_proto.ints()));
                }
            }
            else if(attribute_name == "auto_pad")
            {
                if (attribute_proto.s() == "SAME_UPPER" || attribute_proto.s() == "SAME_LOWER") {
                    lp.set("pad_mode",  "SAME");
                }
                else if (attribute_proto.s() == "VALID") {
                    lp.set("pad_mode", "VALID");
                }
            }
            else if(attribute_name == "dilations")
            {
                CV_Assert(attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("dilation", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "activations" && node_proto.op_type() == "LSTM")
            {
                lp.set(attribute_name, parseStr(attribute_proto.strings()));
            }
            else if (attribute_proto.has_i())
            {
                ::google::protobuf::int64 src = attribute_proto.i();
                if (src < std::numeric_limits<int32_t>::min() || src > std::numeric_limits<int32_t>::max())
                    CV_Error(Error::StsOutOfRange, "Input is out of OpenCV 32S range");
                else
                    lp.set(attribute_name, saturate_cast<int32_t>(src));
            }
            else if (attribute_proto.has_f())
            {
                lp.set(attribute_name, attribute_proto.f());
            }
            else if (attribute_proto.has_s())
            {
                lp.set(attribute_name, attribute_proto.s());
            }
            else if (attribute_proto.floats_size() > 0)
            {
                lp.set(attribute_name, DictValue::arrayReal(
                    attribute_proto.floats().data(), attribute_proto.floats_size()));
            }
            else if (attribute_proto.ints_size() > 0)
            {
                lp.set(attribute_name, parse(attribute_proto.ints()));
            }
            else if (attribute_proto.has_t())
            {
                opencv_onnx::TensorProto tensor = attribute_proto.t();
                Mat blob = getMatFromTensor2(tensor);
                lp.blobs.push_back(blob);
                lp.set("original_dims_of_mat", tensor.dims_size());
            }
            else if (attribute_proto.has_g())
            {
                CV_Error(Error::StsNotImplemented, format("DNN/ONNX/Attribute[%s]: 'Graph' is not supported", attribute_name.c_str()));
            }
            else if (attribute_proto.graphs_size() > 0)
            {
                CV_Error(Error::StsNotImplemented,
                        format("DNN/ONNX/Attribute[%s]: 'Graphs' (%d) in attributes is not supported",
                                attribute_name.c_str(), attribute_proto.graphs_size())
                );
            }
            else if (attribute_proto.strings_size() > 0)
            {
                std::string msg = format("DNN/ONNX/Attribute[%s]: 'Strings' (%d) are not supported",
                        attribute_name.c_str(), attribute_proto.strings_size());
                CV_LOG_ERROR(NULL, msg);
                for (int i = 0; i < attribute_proto.strings_size(); i++)
                {
                    CV_LOG_ERROR(NULL, "    Attribute[" << attribute_name << "].string(" << i << ") = '" << attribute_proto.strings(i) << "'");
                }
                CV_Error(Error::StsNotImplemented, msg);
            }
            else if (attribute_proto.tensors_size() > 0)
            {
                CV_Error(Error::StsNotImplemented,
                        format("DNN/ONNX/Attribute[%s]: 'Tensors' (%d) in attributes are not supported",
                                attribute_name.c_str(), attribute_proto.tensors_size())
                );
            }
            else
            {
                CV_Error(Error::StsNotImplemented, format("DNN/ONNX/Attribute[%s]: unsupported attribute format", attribute_name.c_str()));
            }
        }
        catch (const cv::Exception& e)
        {
            CV_UNUSED(e);
            if (DNN_DIAGNOSTICS_RUN)
            {
                CV_LOG_ERROR(NULL, "DNN/ONNX: Potential problem with processing attributes for node " << node_proto.name() << " Attribute " << attribute_name.c_str()
                );
                continue;
            }
            throw;
        }
    }
    return lp;
}

void ONNXImporter2::parseOperatorSet()
{
    int ir_version = model_proto.has_ir_version() ? static_cast<int>(model_proto.ir_version()) : -1;
    if (ir_version < 3)
        return;

    int opset_size = model_proto.opset_import_size();
    if (opset_size <= 0)
    {
        CV_LOG_INFO(NULL, "DNN/ONNX: missing opset information")
        return;
    }

    for (int i = 0; i < opset_size; ++i)
    {
        const ::opencv_onnx::OperatorSetIdProto& opset_entry = model_proto.opset_import(i);
        const std::string& domain = opset_entry.has_domain() ? opset_entry.domain() : std::string();
        int version = opset_entry.has_version() ? opset_entry.version() : -1;
        if (domain.empty() || domain == str_domain_ai_onnx)
        {
            // ONNX opset covered by specification: https://github.com/onnx/onnx/blob/master/docs/Operators.md
            onnx_opset = std::max(onnx_opset, version);
            onnx_opset_map[str_domain_ai_onnx] = onnx_opset;
        }
        else
        {
            CV_LOG_DEBUG(NULL, "DNN/ONNX: using non-standard ONNX opset[" << i << "]: domain='" << domain << "' version=" << version);
            onnx_opset_map[domain] = onnx_opset;
        }
    }

    CV_LOG_INFO(NULL, "DNN/ONNX: ONNX opset version = " << onnx_opset);

    buildDispatchMap_ONNX_AI(onnx_opset);
    for (const auto& pair : onnx_opset_map)
    {
        if (pair.first == str_domain_ai_onnx)
        {
            continue;  // done above
        }
        else if (pair.first == "com.microsoft")
        {
            buildDispatchMap_COM_MICROSOFT(pair.second);
        }
        else
        {
            CV_LOG_INFO(NULL, "DNN/ONNX: unknown domain='" << pair.first << "' version=" << pair.second << ". No dispatch map, you may need to register 'custom' layers.");
        }
    }
}

/*static bool ifInt8Output(const String& layerType)
{
    // Contains all node types whose output should be int8 when it get int8 input.
    // ai.onnx opset 15
    static std::vector<String> input8output8List = {
            "QuantizeLinear",
            "QLinearAdd",
            "QLinearMul",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearLeakyRelu",
            "QLinearSigmoid",
            "QLinearConcat",
            "QGemm",
            "QLinearSoftmax",
            "QLinearConv",
            "QLinearMatMul",
            "MaxPool",
            "ReduceMax",
            "ReduceMin",
            "Split",
            "Clip",
            "Abs",
            "Transpose",
            "Squeeze",
            "Flatten",
            "Unsqueeze",
            "Expand",
            "Reshape",
            "Pad",
            "Gather",
            "Concat",
            "Resize",
            "SpaceToDepth",
            "DepthToSpace",
            "Pow",
            "Add",
            "Sub",
            "Mul",
            "Div"
    };
    auto layerIt = std::find(input8output8List.begin(), input8output8List.end(), layerType);
    return layerIt != input8output8List.end();
}*/

Net ONNXImporter2::parseModel()
{
    global_node_idx = 0;
    have_errors = false;
    CV_Assert(model_proto.has_graph());
    opencv_onnx::GraphProto* graph_proto = model_proto.mutable_graph();

    std::string framework_version;
    if (model_proto.has_producer_name())
        framework_name = model_proto.producer_name();
    if (model_proto.has_producer_version())
        framework_version = model_proto.producer_version();

    CV_LOG_INFO(NULL, "DNN/ONNX: loading ONNX"
            << (model_proto.has_ir_version() ? format(" v%d", (int)model_proto.ir_version()) : cv::String())
            << " model produced by '" << framework_name << "'"
            << (framework_version.empty() ? cv::String() : format(":%s", framework_version.c_str()))
            << ". Number of nodes = " << graph_proto->node_size()
            << ", initializers = " << graph_proto->initializer_size()
            << ", inputs = " << graph_proto->input_size()
            << ", outputs = " << graph_proto->output_size()
            );

    parseOperatorSet();
    Ptr<Graph> mainGraph = parseGraph(graph_proto, true);
    netimpl->mainGraph = mainGraph;
    netimpl->modelFormat = DNN_MODEL_ONNX;
    netimpl->originalLayout = DATA_LAYOUT_NCHW;
    netimpl->onnx_opset = onnx_opset;

    if (have_errors) {
        std::stringstream sstrm;
        sstrm << "DNN/ONNX: the model ";
        if (!onnxFilename.empty())
            sstrm << "'"  << onnxFilename << "' ";
        sstrm << "cannot be loaded with the new parser. Trying the older parser. ";
        if (!missing_ops.empty()) {
            sstrm << " Unsupported operations:\n";
            auto it = missing_ops.begin();
            size_t i, nmissing = missing_ops.size();
            for (i = 0; i < nmissing; i++, ++it) {
                sstrm << "\t" << *it << (i+1 < nmissing ? ",\n" : "\n");
            }
        }
        CV_LOG_WARNING(NULL, sstrm.str());
        return Net();
    }
    netimpl->prepareForInference();
    // ************ uncomment for debugging **********
    //net.dumpToStream(std::cout);
    return net;
}

bool ONNXImporter2::parseValueInfo(const opencv_onnx::ValueInfoProto& valueInfoProto, ArgData& data)
{
    CV_Assert(valueInfoProto.has_name());
    CV_Assert(valueInfoProto.has_type());
    const opencv_onnx::TypeProto& typeProto = valueInfoProto.type();
    CV_Assert(typeProto.has_tensor_type());
    const opencv_onnx::TypeProto::Tensor& tensor = typeProto.tensor_type();
    CV_Assert(tensor.has_shape());
    const opencv_onnx::TensorShapeProto& tensorShape = tensor.shape();
    auto elem_type = tensor.elem_type();

    data.type = dataType2cv(elem_type);
    if (data.type < 0) {
        CV_Error(Error::StsNotImplemented, format("unsupported datatype '%s'", dataType2str(elem_type).c_str()));
    }

    int dim_size = tensorShape.dim_size();
    CV_CheckGE(dim_size, 0, "");
    MatShape shape(dim_size);
    for (int j = 0; j < dim_size; ++j)
    {
        const opencv_onnx::TensorShapeProto_Dimension& dimension = tensorShape.dim(j);
        int64_t val_j;
        if (dimension.has_dim_value()) {
            val_j = dimension.dim_value();
        } else if (dimension.has_dim_param()) {
            const std::string& param_j = dimension.dim_param();
            val_j = net.findDim(param_j, true);
        } else {
            raiseError();
            return false;
        }
        //CV_Assert(0 <= val_j && val_j <= INT_MAX);
        shape[j] = (int)val_j;
    }
    data.shape = shape;
    return true;
}

Mat ONNXImporter2::parseTensor(const opencv_onnx::TensorProto& tensor_proto)
{
    return getMatFromTensor2(tensor_proto);
}

Ptr<Graph> ONNXImporter2::parseGraph(opencv_onnx::GraphProto* graph_proto, bool mainGraph_)
{
    CV_LOG_DEBUG(NULL, "DNN/ONNX: parsing graph '" << graph_proto->name() << "' of " << graph_proto->node_size() << " nodes");
    simplifySubgraphs(*graph_proto);
    int n_nodes = graph_proto->node_size();
    CV_LOG_DEBUG(NULL, "DNN/ONNX: simplified the graph to " << n_nodes << " nodes");

    opencv_onnx::GraphProto* saved_graph_proto = curr_graph_proto;
    Ptr<Graph> saved_graph = curr_graph;
    std::vector<Ptr<Layer> > saved_prog;

    curr_graph_proto = graph_proto;
    std::vector<Arg> inputs, outputs;

    // parse constant tensors
    int n_consts = graph_proto->initializer_size();
    for (int i = 0; i < n_consts; i++) {
        //const opencv_onnx::
        const opencv_onnx::TensorProto& const_i = graph_proto->initializer(i);
        Mat t = parseTensor(const_i);
        netimpl->newConstArg(const_i.name(), t);
    }

    // parse graph inputs
    int n_inputs = graph_proto->input_size();
    for (int i = 0; i < n_inputs; i++) {
        const opencv_onnx::ValueInfoProto& input_i = graph_proto->input(i);
        if (net.haveArg(input_i.name()))
            continue;
        Arg arg = netimpl->newArg(input_i.name(), mainGraph_ ? DNN_ARG_INPUT : DNN_ARG_TEMP);
        if (!parseValueInfo(input_i, netimpl->args.at(arg.idx))) {
            raiseError();
            return Ptr<Graph>();
        }
        inputs.push_back(arg);
    }

    // parse graph outputs
    int n_outputs = graph_proto->output_size();
    for (int i = 0; i < n_outputs; i++) {
        const opencv_onnx::ValueInfoProto& output_i = graph_proto->output(i);
        Arg arg = netimpl->newArg(output_i.name(), mainGraph_ ? DNN_ARG_OUTPUT : DNN_ARG_TEMP);
        if (!parseValueInfo(output_i, netimpl->args.at(arg.idx))) {
           raiseError();
           return Ptr<Graph>();
        }
        outputs.push_back(arg);
    }

    curr_graph = netimpl->newGraph(graph_proto->name(), inputs, mainGraph_);
    curr_graph->setOutputs(outputs);

    std::swap(saved_prog, curr_prog);

    std::vector<Ptr<Layer> > prog;
    for (int i = 0; i < n_nodes && !have_errors; i++) {
        parseNode(graph_proto->node(i));
    }

    curr_graph->setProg(curr_prog);
    curr_prog = saved_prog;

    Ptr<Graph> just_constructed = curr_graph;
    curr_graph_proto = saved_graph_proto;
    curr_graph = saved_graph;

    return just_constructed;
}

std::string ONNXImporter2::getLayerTypeDomain(const opencv_onnx::NodeProto& node_proto)
{
    if (!node_proto.has_domain())
        return str_domain_ai_onnx;
    const std::string& domain = node_proto.domain();
    if (domain.empty())
        return str_domain_ai_onnx;
    return domain;
}

const ONNXImporter2::DispatchMap& ONNXImporter2::getDispatchMap(const opencv_onnx::NodeProto& node_proto)
{
    static DispatchMap empty_map;
    const std::string& layer_type_domain = getLayerTypeDomain(node_proto);
    auto it = domain_dispatch_map.find(layer_type_domain);
    if (it == domain_dispatch_map.end())
    {
        return empty_map;
    }

    return it->second;
}

std::string ONNXImporter2::extractNodeName(const opencv_onnx::NodeProto& node_proto)
{
    // We need to rework DNN outputs API, this is a workaround for #21698
    if (node_proto.has_name() && !node_proto.name().empty())
    {
        if (useLegacyNames)
            return node_proto.name();
        return format("onnx_node!%s", node_proto.name().c_str());
    }
    return format("onnx_node!%d", global_node_idx++);
}

void ONNXImporter2::rememberMissingOp(const std::string& opname)
{
    missing_ops.insert(opname);
    raiseError();
}

void ONNXImporter2::parseNode(const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.output_size() >= 1);
    std::string node_name = extractNodeName(node_proto);
    const std::string& layer_type = node_proto.op_type();
    std::string layer_type_domain = getLayerTypeDomain(node_proto);
    const auto& dispatch = getDispatchMap(node_proto);

    /*CV_LOG_INFO(NULL, "DNN/ONNX: processing node '" << node_name << "' ("
                << layer_type << ") with " << node_proto.input_size() << " inputs and "
                << node_proto.output_size() << " outputs from domain '"
                << layer_type_domain << "'");*/

    if (dispatch.empty())
    {
        CV_LOG_ERROR(NULL, "DNN/ONNX: missing dispatch map for domain='" << layer_type_domain << "'");
        rememberMissingOp(layer_type);
        return;
    }

    node_inputs.clear();
    node_outputs.clear();

    int n_inputs = node_proto.input_size();
    for (int i = 0; i < n_inputs; i++) {
        const std::string& arg_name = node_proto.input(i);
        if (!net.haveArg(arg_name)) {
            CV_LOG_ERROR(NULL, "DNN/ONNX: unknown input '" << arg_name << "' of node '" << node_name << "'");
            raiseError();
        }
        Arg arg = net.getArg(arg_name);
        /*ArgData adata = net.argData(arg);
        printf("%s (%s), arg '%s'/'%s': adata.kind = %s, type=%s\n", node_name.c_str(), layer_type.c_str(),
               arg_name.c_str(), adata.name.c_str(),
               argKindToString(adata.kind).c_str(), typeToString(adata.type).c_str());*/
        node_inputs.push_back(arg);
    }

    int n_outputs = node_proto.output_size();
    for (int i = 0; i < n_outputs; i++) {
        const std::string& arg_name = node_proto.output(i);
        Arg arg = net.getArg(arg_name);
        node_outputs.push_back(arg);
    }

    LayerParams layerParams;
    try
    {
        layerParams = getLayerParams(node_proto);

        layerParams.name = node_name;
        layerParams.type = layer_type;

        DispatchMap::const_iterator iter = dispatch.find(layer_type);
        if (iter != dispatch.end())
        {
            if (!have_errors)
                CALL_MEMBER_FN(*this, iter->second)(layerParams, node_proto);
        } else if (!have_errors) {
            //try customly parsing the layer without explicit dispatch map
            parseCustomLayer(layerParams, node_proto);
        } else {
            rememberMissingOp(layer_type);
        }
    }
    catch (const cv::Exception& e)
    {
        raiseError();
        CV_LOG_INFO(NULL, "DNN/ONNX: error '" << e.what() << "' occurred when processing node '" << node_name
                    << "' (" << layer_type << ") with "
                    << node_proto.input_size() << " inputs and "
                    << node_proto.output_size() << " outputs from domain '"
                    << layer_type_domain << "'");
        for (int i = 0; i < n_inputs; i++)
        {
            CV_LOG_INFO(NULL, "    Input[" << i << "] = '" << node_proto.input(i) << "'");
        }
        for (int i = 0; i < n_outputs; i++)
        {
            CV_LOG_INFO(NULL, "    Output[" << i << "] = '" << node_proto.output(i) << "'");
        }
    }
}

void ONNXImporter2::addLayer(LayerParams& layerParams,
                             const opencv_onnx::NodeProto& node_proto,
                             int max_inputs)
{
    Ptr<Layer> layer = LayerFactory::createLayerInstance(layerParams.type, layerParams);
    if (!layer) {
        rememberMissingOp(layerParams.type);
        return;
    }
    size_t actual_inputs = std::min((size_t)max_inputs, node_inputs.size());
    layer->inputs = node_inputs;
    layer->inputs.resize(actual_inputs);
    layer->outputs = node_outputs;
    layer->netimpl = netimpl;
    CV_Assert(netimpl->dump_indent == 3);
    curr_prog.push_back(layer);
}

void ONNXImporter2::parseNeg(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Power";
    layerParams.set("scale", -1);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseCustomLayer(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    parseSimpleLayers(layerParams, node_proto);
}

void ONNXImporter2::parseArgMinMax(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const std::string& layer_type = node_proto.op_type();
    layerParams.type = "Arg";
    layerParams.set("op", layer_type == "ArgMax" ? "max" : "min");
    addLayer(layerParams, node_proto);
}

static void setCeilMode(LayerParams& layerParams)
{
    // auto_pad attribute is deprecated and uses ceil
    if (layerParams.has("pad_mode"))
    {
        layerParams.set("ceil_mode", true);
    }
    else if (!layerParams.has("ceil_mode"))
    {
        layerParams.set("ceil_mode", false);
    }
}

void ONNXImporter2::parseMaxUnpool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "MaxUnpool";

    DictValue kernel_shape = layerParams.get("kernel_size");
    CV_Assert(kernel_shape.size() == 2);
    layerParams.set("pool_k_w", kernel_shape.get<int>(0));
    layerParams.set("pool_k_h", kernel_shape.get<int>(1));

    int pool_pad_w = 0, pool_pad_h = 0;
    if (layerParams.has("pad"))
    {
        DictValue pads = layerParams.get("pad");
        CV_CheckEQ(pads.size(), 2, "");
        pool_pad_w = pads.get<int>(0);
        pool_pad_h = pads.get<int>(1);
    }
    layerParams.set("pool_pad_w", pool_pad_w);
    layerParams.set("pool_pad_h", pool_pad_h);


    int pool_stride_w = 1, pool_stride_h = 1;
    if (layerParams.has("stride"))
    {
        DictValue strides = layerParams.get("stride");
        CV_CheckEQ(strides.size(), 2, "");
        pool_stride_w = strides.get<int>(0);
        pool_stride_h = strides.get<int>(1);
    }
    layerParams.set("pool_stride_w", pool_stride_w);
    layerParams.set("pool_stride_h", pool_stride_h);

    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseMaxPool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type = (depth == CV_8S) ? "PoolingInt8" : "Pooling";
    layerParams.set("pool", "MAX");
    setCeilMode(layerParams);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseAveragePool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Pooling";
    layerParams.set("pool", "AVE");
    setCeilMode(layerParams);
    layerParams.set("ave_pool_padded_area", framework_name == "pytorch");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseGlobalPool(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    const std::string& layer_type = node_proto.op_type();
    const std::string output_name = node_proto.output(0);

    CV_Assert(node_proto.input_size() == 1);
    layerParams.type = "Pooling";
    String pool;
    if (layer_type == "GlobalMaxPool")
        pool = "MAX";
    else if (layer_type == "GlobalAveragePool")
        pool = "AVE";
    else
        CV_Error(Error::StsNotImplemented, "Unsupported Pooling type of " + layer_type + " operation.");

    CV_Assert(!layerParams.has("axes"));
    layerParams.set("global_pooling", true);
    layerParams.set("pool", pool);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseReduce(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Reduce";
    const auto& op_type = node_proto.op_type();
    String reduce_type;
    if (op_type == "ReduceMax")
        reduce_type = "MAX";
    else if (op_type == "ReduceMean")
        reduce_type = "MEAN";
    else if (op_type == "ReduceMin")
        reduce_type = "MIN";
    else if (op_type == "ReduceProd")
        reduce_type = "PROD";
    else if (op_type == "ReduceSum")
        reduce_type = "SUM";
    else if (op_type == "ReduceL1")
        reduce_type = "L1";
    else if (op_type == "ReduceL2")
        reduce_type = "L2";
    else if (op_type == "ReduceLogSum")
        reduce_type = "LOG_SUM";
    else if (op_type == "ReduceLogSumExp")
        reduce_type = "LOG_SUM_EXP";
    else if (op_type == "ReduceSumSquare")
        reduce_type = "SUM_SQUARE";
    else
        CV_Error(Error::StsNotImplemented, "DNN/ONNX: " + op_type + " is not supported.");
    layerParams.set("reduce", reduce_type);

    int num_inputs = node_proto.input_size();
    CV_Check(num_inputs, num_inputs >= 1 && num_inputs <= 2, "DNN/ONNX: Reduce layers should have at least one input and at most two inputs");

    bool const_axis_input = false;
    if (num_inputs >= 2) {
        CV_CheckTrue(net.isConstArg(node_inputs[1]), "Reduce layer doesn't support non contant axes");
        const_axis_input = true;
    }

    // "axes" is turned to one of the inputs since opset 18,
    // except for ReduceSum, which has "axes" input since opset 13.
    if (const_axis_input) {
        Mat mat_axes = net.argTensor(node_inputs[1]);
        int num_axes = (int)mat_axes.total();
        std::vector<int> axes(num_axes);
        for (int i = 0; i < num_axes; ++i)
            axes[i] = mat_axes.at<int64_t>(i);
        layerParams.set("axes", DictValue::arrayInt(&axes[0], num_axes));
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseSlice(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 1 || (3 <= ninputs && ninputs <= 5));
    layerParams.type = "Slice2";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseSplit(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckGE(node_proto.input_size(), 1, "");
    CV_CheckLE(node_proto.input_size(), 2, "");
    layerParams.type = "Split2";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseConstant(LayerParams& layerParams, const opencv_onnx::NodeProto&)
{
    CV_Assert(node_inputs.empty());
    CV_Assert(node_outputs.size() == 1);
    CV_Assert(layerParams.blobs.size() == 1);
    Mat m = layerParams.blobs[0];
    Arg out = node_outputs[0];
    ArgData& data = netimpl->args.at(out.idx);
    data.kind = DNN_ARG_CONST;
    data.type = m.type();
    data.shape = m.shape();
    netimpl->__tensors__.at(out.idx) = m;
}

// BUG: https://github.com/opencv/opencv/issues/26308
void ONNXImporter2::parseLSTM(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto lstm_proto = node_proto_;
    layerParams.type = "LSTM2";

    layerParams.set("is_onnx", true);
    layerParams.set("reverse", layerParams.get<String>("direction", "") == "reverse");
    layerParams.set("bidirectional", layerParams.get<String>("direction", "") == "bidirectional");



    bool need_yc = lstm_proto.output_size() > 2 && !lstm_proto.output(2).empty();
    bool need_yh = lstm_proto.output_size() > 1 && !lstm_proto.output(1).empty();
    bool need_y  = lstm_proto.output_size() > 0 && !lstm_proto.output(0).empty();

    const std::string y_name = need_y ? lstm_proto.output(0) : "";
    const std::string yh_name = need_yh ? lstm_proto.output(1) : "";
    const std::string yc_name = need_yc ? lstm_proto.output(2) : "";

    layerParams.set("produce_cell_output", need_yc);
    layerParams.set("produce_output_yh", need_yh);


    if (lstm_proto.input_size() == 8)
        layerParams.set("use_peephole", true);


    addLayer(layerParams, lstm_proto);
}

 // BUG: https://github.com/opencv/opencv/issues/26309
void ONNXImporter2::parseGRU(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    rememberMissingOp(node_proto_.op_type());
}

void ONNXImporter2::parseImageScaler(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const float scale = layerParams.has("scale") ? layerParams.get<float>("scale") : 1.0f;
    layerParams.erase("scale");

    if (layerParams.has("bias"))
    {
        layerParams.type = "Scale";
        layerParams.blobs.push_back(
                Mat(Size(1,  layerParams.get("bias").size()), CV_32FC1, scale));

        layerParams.set("bias_term", true);
        Mat bias(1, layerParams.get("bias").size(), CV_32FC1);
        for (int j = 0; j < bias.total(); j++) {
            bias.at<float>(0, j) = layerParams.get("bias").getRealValue(j);
        }
        layerParams.blobs.push_back(bias);
        layerParams.erase("bias");
    }
    else {
        layerParams.set("scale", scale);
        layerParams.type = "Power";
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseClip(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ReLU6";
    float min_value = -FLT_MAX, max_value = FLT_MAX;
    int input_size = node_proto.input_size();
    CV_Check(input_size, 1 <= input_size && input_size <= 3, "");

    if (input_size >= 2 && !node_proto.input(1).empty())
    {
        Mat m;
        CV_Assert(net.isConstArg(node_inputs[1]));
        net.argTensor(node_inputs[1]).convertTo(m, CV_32F);
        CV_Assert(m.total() == 1);
        min_value = m.at<float>(0);
    }

    if (input_size == 3 && !node_proto.input(2).empty())
    {
        Mat m;
        CV_Assert(net.isConstArg(node_inputs[2]));
        net.argTensor(node_inputs[2]).convertTo(m, CV_32F);
        CV_Assert(m.total() == 1);
        max_value = m.at<float>(0);
    }

    layerParams.set("min_value", layerParams.get<float>("min", min_value));
    layerParams.set("max_value", layerParams.get<float>("max", max_value));
    addLayer(layerParams, node_proto, 1);
}

void ONNXImporter2::parseLeakyRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ReLU";
    layerParams.set("negative_slope", layerParams.get<float>("alpha", 0.01));
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ReLU";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseElu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ELU";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseTanh(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "TanH";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseAbs(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "AbsVal";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parsePRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "PReLU";
    CV_Assert(node_inputs.size() == 2);
    CV_Assert(net.isConstArg(node_inputs[1]));
    layerParams.blobs.push_back(net.argTensor(node_inputs[1]));
    addLayer(layerParams, node_proto, 1);
}

void ONNXImporter2::parseLRN(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    replaceLayerParam(layerParams, "size", "local_size");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseInstanceNormalization(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto) {
    int num_inputs = node_proto.input_size();
    CV_CheckEQ(num_inputs, 3, "DNN/ONNXImporter2 - InstanceNorm: three inputs are required");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseBatchNormalization(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    if (node_proto.input_size() != 5)
        CV_Error(Error::StsNotImplemented,
                 "Expected input, scale, bias, mean and var");

    layerParams.type = "BatchNorm";
    replaceLayerParam(layerParams, "epsilon", "eps");
    replaceLayerParam(layerParams, "spatial", "use_global_stats");

    CV_Assert(net.isConstArg(node_inputs[3]));
    CV_Assert(net.isConstArg(node_inputs[4]));

    Mat meanData = net.argTensor(node_inputs[3]);
    Mat stdData =  net.argTensor(node_inputs[4]);

    layerParams.blobs.push_back(meanData);
    layerParams.blobs.push_back(stdData);

    if (!node_proto.input(1).empty()) {
        layerParams.set("has_weight", true);
        CV_Assert(net.isConstArg(node_inputs[1]));
        layerParams.blobs.push_back(net.argTensor(node_inputs[1]));  // weightData
    } else {
        layerParams.set("has_weight", false);
    }

    if (!node_proto.input(2).empty()) {
        layerParams.set("has_bias", true);
        CV_Assert(net.isConstArg(node_inputs[1]));
        layerParams.blobs.push_back(net.argTensor(node_inputs[2]));  // biasData
    } else {
        layerParams.set("has_bias", false);
    }
    addLayer(layerParams, node_proto, 1);
}

void ONNXImporter2::parseGemm(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Gemm";
    int n_inputs = node_proto.input_size();
    CV_Assert(2 <= n_inputs && n_inputs <= 3);

    if (net.isConstArg(node_inputs[1]) && (n_inputs == 2 || net.isConstArg(node_inputs[2]))) {
        Mat B = net.argTensor(node_inputs[1]);
        layerParams.blobs.push_back(B);
        if (n_inputs > 2) {
            Mat bias = net.argTensor(node_inputs[2]);
            layerParams.blobs.push_back(bias);
        }
        n_inputs = 1;
    }

    addLayer(layerParams, node_proto, n_inputs);
}

void ONNXImporter2::parseMatMul(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto) {
    int n_inputs = node_proto.input_size();
    CV_Assert(2 <= n_inputs && n_inputs <= 3);

    if (net.isConstArg(node_inputs[1]) && (n_inputs == 2 || net.isConstArg(node_inputs[2]))) {
        Mat B = net.argTensor(node_inputs[1]);
        layerParams.blobs.push_back(B);
        if (n_inputs > 2) {
            Mat bias = net.argTensor(node_inputs[2]);
            layerParams.blobs.push_back(bias);
        }
        n_inputs = 1;
    }
    addLayer(layerParams, node_proto, n_inputs);
}

void ONNXImporter2::parseConv(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int n_inputs = node_proto.input_size();
    CV_Assert(2 <= n_inputs && n_inputs <= 3);
    layerParams.type = "Convolution";

    if (net.isConstArg(node_inputs[1]) && (n_inputs == 2 || net.isConstArg(node_inputs[2]))) {
        Mat weights = net.argTensor(node_inputs[1]);
        layerParams.blobs.push_back(weights);
        if (n_inputs > 2) {
            Mat bias = net.argTensor(node_inputs[2]);
            layerParams.blobs.push_back(bias);
        }
        n_inputs = 1;
    }
    addLayer(layerParams, node_proto, n_inputs);
}

void ONNXImporter2::parseConvTranspose(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int n_inputs = node_proto.input_size();
    CV_Assert(2 <= n_inputs && n_inputs <= 3);
    layerParams.type = "Deconvolution";

    layerParams.set("bias_term", node_proto.input_size() == 3);

    if (net.isConstArg(node_inputs[1]) && (n_inputs == 2 || net.isConstArg(node_inputs[2]))) {
        Mat weights = net.argTensor(node_inputs[1]);
        layerParams.blobs.push_back(weights);
        if (n_inputs > 2) {
            Mat bias = net.argTensor(node_inputs[2]);
            layerParams.blobs.push_back(bias);
        }
        n_inputs = 1;
    }

    if (!layerParams.has("kernel_size"))
        CV_Error(Error::StsNotImplemented,
                 "Required attribute 'kernel_size' is not present.");

    if (layerParams.has("output_shape"))
    {
        const DictValue& outShape = layerParams.get("output_shape");
        DictValue strides = layerParams.get("stride");
        DictValue kernel = layerParams.get("kernel_size");

        String padMode;
        std::vector<int> adjust_pads;
        if (layerParams.has("pad_mode"))
        {
            padMode = toUpperCase(layerParams.get<String>("pad_mode"));
            if (padMode != "SAME" && padMode != "VALID")
                CV_Error(Error::StsError, "Unsupported padding mode " + padMode);

            for (int i = 0; i < strides.size(); i++)
            {
                int sz = outShape.get<int>(2 + i);
                int stride = strides.get<int>(i);
                adjust_pads.push_back(padMode == "SAME"? (sz - 1) % stride :
                                                         (sz - kernel.get<int>(i)) % stride);
            }
            layerParams.set("adj", DictValue::arrayInt(&adjust_pads[0], (int)adjust_pads.size()));
        }
    }
    else if (layerParams.has("output_padding"))
    {
        replaceLayerParam(layerParams, "output_padding", "adj");
    }
    addLayer(layerParams, node_proto, n_inputs);
}

void ONNXImporter2::parseTranspose(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 1);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseSqueeze(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Squeeze";
    CV_Assert(node_proto.input_size() <= 2);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseFlatten(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 1);
    layerParams.set("onnx", true);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseUnsqueeze(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert((node_proto.input_size() == 1 && layerParams.has("axes")) ||
              node_proto.input_size() == 2);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseExpand(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Expand2";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseReshape(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    bool have_shape_attr = layerParams.has("shape");
    CV_Assert((node_proto.input_size() == 2 && !have_shape_attr) ||
              (node_proto.input_size() == 1 && have_shape_attr));
    layerParams.type = "Reshape2";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parsePad(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Pad2";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseShape(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 1);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseCast(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    opencv_onnx::TensorProto_DataType onnx_type = (opencv_onnx::TensorProto_DataType)layerParams.get<int>("to");
    int type = dataType2cv(onnx_type);

    layerParams.type = "Cast";
    layerParams.set("outputType", type);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseConstantOfShape(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ConstantOfShape";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseGather(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Gather2";
    CV_CheckEQ(node_proto.input_size(), 2, "");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseGatherElements(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 2, "GatherElements: two inputs are required");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseConcat(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.output_size(), 1, "");
    layerParams.type = "Concat2";
    addLayer(layerParams, node_proto);
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
void ONNXImporter2::parseResize(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    layerParams.type = "Resize";
    String interp_mode = layerParams.get<String>("coordinate_transformation_mode", "half_pixel");

    CV_Assert(interp_mode != "tf_crop_and_resize");
    bool halfPixel = interp_mode == "tf_half_pixel_for_nn" || interp_mode == "half_pixel" || interp_mode == "pytorch_half_pixel";

    layerParams.set("align_corners", interp_mode == "align_corners");
    layerParams.set("half_pixel_centers", halfPixel);
    if (layerParams.get<String>("mode") == "linear")
    {
        layerParams.set("mode", halfPixel ? "opencv_linear" : "bilinear");
    }

    if (layerParams.get<String>("mode") == "linear" && framework_name == "pytorch")
        layerParams.set("mode", "opencv_linear");

    // opset-10: input = [X, scales]
    // opset-11: input = [X, roi, scales] or [x, roi, scales, sizes]
    // opset-13: may have empty input, [X, "", "", sizes] or [x, "", scales]
    int scalesInputId = ninputs == 2 ? 1 : 2;
    Arg scalesArg = node_inputs[scalesInputId];
    Mat scales;
    if(scalesArg.idx > 0 && netimpl->isConstArg(scalesArg))
        scales = netimpl->argTensor(scalesArg);

    if (ninputs >= 3 && interp_mode == "tf_crop_and_resize") {
        int roiInputId = 1;
        Arg roiArg = node_inputs[roiInputId];
        if (!netimpl->isConstArg(roiArg)) {
            CV_Error(Error::StsNotImplemented, "ONNX/Resize: only empty ROI is supported");
        }
        Mat roi = netimpl->argTensor(roiArg);
        if (!roi.empty()) {
            CV_Error(Error::StsNotImplemented, "ONNX/Resize: only empty ROI is supported");
        }
    }

    if (scales.total() == 4)
    {
        CV_CheckEQ(scales.total(), (size_t)4, "HCHW layout is expected");
        CV_CheckEQ(scales.type(), CV_32F, "Scales should have 32F type");
        layerParams.set("zoom_factor_y", scales.at<float>(2));
        layerParams.set("zoom_factor_x", scales.at<float>(3));
        ninputs = 1;
    }
    else if (ninputs >= 4)  // opset-11 [x, roi, scales, sizes] or opset-13: input = [X, "", "", sizes]
    {
        Arg sizesArg = node_inputs[3];
        if (netimpl->isConstArg(sizesArg))
        {
            Mat shapes_ = netimpl->argTensor(sizesArg), shapes;
            CV_CheckEQ(shapes_.total(), (size_t)4, "HCHW layout is expected");
            shapes_.convertTo(shapes, CV_32S);
            layerParams.set("width", shapes.at<int>(3));
            layerParams.set("height", shapes.at<int>(2));
            ninputs = 1;
        }
    }
    replaceLayerParam(layerParams, "mode", "interpolation");
    addLayer(layerParams, node_proto, ninputs);
}

void ONNXImporter2::parseUpsample(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int n_inputs = node_proto.input_size();
    //fused from Resize Subgraph
    if (layerParams.has("coordinate_transformation_mode"))
    {
        String interp_mode = layerParams.get<String>("coordinate_transformation_mode");
        CV_Assert(interp_mode != "tf_crop_and_resize");

        bool halfPixel = interp_mode == "tf_half_pixel_for_nn" || interp_mode == "half_pixel" || interp_mode == "pytorch_half_pixel";

        layerParams.set("align_corners", interp_mode == "align_corners");
        layerParams.set("half_pixel_centers", halfPixel);
        if (layerParams.get<String>("mode") == "linear")
        {
            layerParams.set("mode", halfPixel ? "opencv_linear" : "bilinear");
        }
    }
    if (layerParams.get<String>("mode") == "linear" && framework_name == "pytorch")
        layerParams.set("mode", "opencv_linear");

    layerParams.type = "Resize";
    if (layerParams.has("scales"))
    {
        // Pytorch layer
        DictValue scales = layerParams.get("scales");
        CV_Assert(scales.size() == 4);
        layerParams.set("zoom_factor_y", scales.getIntValue(2));
        layerParams.set("zoom_factor_x", scales.getIntValue(3));
    }
    else if (layerParams.has("height_scale") && layerParams.has("width_scale"))
    {
        // Caffe2 layer
        replaceLayerParam(layerParams, "height_scale", "zoom_factor_y");
        replaceLayerParam(layerParams, "width_scale", "zoom_factor_x");
    }
    else
    {
        CV_Assert(n_inputs >= 2);
        // scales as input
        if (net.isConstArg(node_inputs[1])) {
            Mat scales;
            net.argTensor(node_inputs[1]).convertTo(scales, CV_32F);
            CV_Assert(scales.total() == 4);
            layerParams.set("zoom_factor_y", scales.at<float>(2));
            layerParams.set("zoom_factor_x", scales.at<float>(3));
            n_inputs = 1;
        }
    }
    replaceLayerParam(layerParams, "mode", "interpolation");
    addLayer(layerParams, node_proto, n_inputs);
}

void ONNXImporter2::parseSoftMax(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const std::string& layer_type = node_proto.op_type();
    int axis;
    if (onnx_opset != 0 && onnx_opset <= 11) {
        axis = layerParams.get<int>("axis", 1);
    } else {
        axis = layerParams.get<int>("axis", -1);
    }
    layerParams.set<int>("axis", axis);
    layerParams.type = "Softmax";
    layerParams.set("log_softmax", layer_type == "LogSoftmax");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseDetectionOutput(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 3, "");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseCumSum(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 2);
    layerParams.type = "CumSum";
    if (net.isConstArg(node_inputs[1]))
    {
        Mat axisTensor;
        net.argTensor(node_inputs[1]).convertTo(axisTensor, CV_32S);
        CV_Assert(axisTensor.total() == 1);
        int axis = axisTensor.at<int>(0);
        layerParams.set("axis", axis);
        ninputs = 1;
    }
    addLayer(layerParams, node_proto, ninputs);
}

// "Equal" "Greater" "Less" "Pow" "Add" "Sub" "Mul" "Div" "Sum" "Min" "Max" "GreaterOrEqual" "LessOrEqual" "And" "Or" "Xor"
void ONNXImporter2::parseElementWise(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    String op_type = toLowerCase(node_proto.op_type());

    layerParams.type = "NaryEltwise";
    layerParams.set("operation", toLowerCase(node_proto.op_type()));
    if (node_proto.op_type() == "Mod") {
        if (layerParams.get<int>("fmod", 0)) {
            layerParams.set("operation", "fmod");
        };
    }
    // add element-wise layer
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseDepthSpaceOps(LayerParams &layerParams, const opencv_onnx::NodeProto& node_proto) {
    CV_CheckTrue(layerParams.has("blocksize"), "blocksize is required but not found");
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseTopK2(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto) {
    layerParams.type = "TopK2";
    addLayer(layerParams, node_proto);
}

// Currently we only support range with all constant inputs
void ONNXImporter2::parseRange(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 3); // 0 - start, 1 - limit, 2 - delta
    layerParams.type = "Range";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseScatter(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 3, "Scatter: three inputs are required.");
    layerParams.type = "Scatter";
    if (node_proto.op_type() == "ScatterND")
        layerParams.type = "ScatterND";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseTile(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    // for Tile>1, only the case of 'repeats' being constant is supported.
    // 'repeats' is treated as a parameter instead of an input to determine shape in pre-run.
    layerParams.type = "Tile2";
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseLayerNorm(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int n_inputs = node_proto.input_size();
    CV_Assert(2 <= n_inputs && n_inputs <= 3);
    if (net.isConstArg(node_inputs[1]) && (n_inputs == 2 || net.isConstArg(node_inputs[2]))) {
        Mat scale = net.argTensor(node_inputs[1]);
        layerParams.blobs.push_back(scale);
        if (n_inputs > 2) {
            Mat bias = net.argTensor(node_inputs[2]);
            layerParams.blobs.push_back(bias);
        }
        n_inputs = 1;
    }
    addLayer(layerParams, node_proto, n_inputs);
}

void ONNXImporter2::parseSimpleLayers(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseEinsum(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    // Check if of equation is valid
    std::string equation = layerParams.get<std::string>("equation");
    CV_CheckFalse(equation.empty(), "Equation is empty");

    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseDequantizeLinear(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQuantizeLinear(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int dt = layerParams.get<int>("output_dtype", -1);
    if (dt >= 0)
    {
        layerParams.set<int>("output_dtype", dataType2cv((opencv_onnx::TensorProto_DataType)dt));
    }
    addLayer(layerParams, node_proto);
}

// BUG: https://github.com/opencv/opencv/issues/26310
/*void ONNXImporter2::parseQConv(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 8 || ninputs == 9);

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int inp_zp = (int)getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    if (layerParams.has("pad"))
    {
        bool asymmetricPadding = false;
        DictValue pads = layerParams.get("pad");
        const int dims = pads.size() / 2;

        for (int i = 0; i < dims; ++i)
        {
            if (pads.get<int>(i) != pads.get<int>(i + dims))
            {
                asymmetricPadding = true;
                break;
            }
        }
        if (asymmetricPadding && pads.size() == 4)
        {
            layerParams.erase("pad");
            std::vector<int> paddings(4, 0);
            for (int i = 0; i < dims; ++i)
            {
                paddings.push_back(pads.get<int>(i));
                paddings.push_back(pads.get<int>(dims + i));
            }
            LayerParams padLp;
            padLp.name = layerParams.name + "/pad";
            padLp.type = "PaddingInt8";
            padLp.set("paddings", DictValue::arrayInt(&paddings[0], paddings.size()));
            padLp.set("depth", CV_8S);
            padLp.set<double>("value", (double)inp_zp);

            opencv_onnx::NodeProto proto;
            proto.add_input(node_proto.input(0));
            proto.add_output(padLp.name);

            addLayer(padLp, proto);
            node_proto.set_input(0, padLp.name);
        }
    }

    Mat weights = getBlob(node_proto, 3);
    int outCn = weights.size[0];
    Mat w_scale = getBlob(node_proto, 4);
    CV_Assert(w_scale.total() == 1 || w_scale.total() == outCn);
    bool per_channel = w_scale.total() == outCn;
    Mat wt_sc = (w_scale.total() == outCn) ? w_scale : Mat(1, outCn, CV_32F, Scalar(w_scale.at<float>(0)));

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 6));
    int8_t out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 7));

    Mat bias = (ninputs == 9) ? getBlob(node_proto, 8) : Mat::zeros(1, outCn, CV_32S);

    Mat weights_2d = weights.reshape(1, outCn);
    Mat biasFused(1, outCn, CV_32S);
    Mat outputMultiplier(1, outCn, CV_32F);
    for (int i = 0; i < outCn; i++)
    {
        biasFused.at<int>(i) = bias.at<int>(i) - inp_zp*(cv::sum(weights_2d.row(i))[0]);
        outputMultiplier.at<float>(i) = (inp_sc * wt_sc.at<float>(i)) / out_sc;
    }

    layerParams.type = "ConvolutionInt8";
    layerParams.set("num_output", outCn);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("input_scale",inp_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("per_channel", per_channel);
    layerParams.blobs.push_back(weights);
    layerParams.blobs.push_back(biasFused);
    layerParams.blobs.push_back(outputMultiplier);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQMatMul(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 8);

    if (constBlobs.find(node_proto.input(3)) == constBlobs.end())
        CV_Error(Error::StsNotImplemented, "Variable weights is not supported");

    int firstInpDims = outShapes[node_proto.input(0)].size();

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    Mat weights = getBlob(node_proto, 3).t();
    int outCn = weights.size[0];
    int secondInpDims = weights.dims;

    Mat w_scale = getBlob(node_proto, 4);
    CV_Assert(w_scale.total() == 1 || w_scale.total() == outCn);
    bool per_channel = w_scale.total() == outCn ? true : false;
    Mat wt_sc = (w_scale.total() == outCn) ? w_scale : Mat(1, outCn, CV_32F, Scalar(w_scale.at<float>(0)));

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 6));
    int8_t out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 7));

    Mat bias(1, outCn, CV_32S);
    Mat outputMultiplier(1, outCn, CV_32F);
    for (int i = 0; i < outCn; i++)
    {
        bias.at<int>(i) = -inp_zp*(cv::sum(weights.row(i))[0]);
        outputMultiplier.at<float>(i) = (inp_sc * wt_sc.at<float>(i)) / out_sc;
    }

    layerParams.type = "InnerProductInt8";
    layerParams.set("num_output", outCn);
    layerParams.set("axis", firstInpDims - secondInpDims + 1);
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("per_channel", per_channel);

    layerParams.blobs.push_back(weights);
    layerParams.blobs.push_back(bias);
    layerParams.blobs.push_back(outputMultiplier);
    addLayer(layerParams, node_proto);
}

// A * B + C = Y, we require that the dimension of A is [m, k], and the dimension of B is [n, k].
// And the dim of output Y is [m, n]
void ONNXImporter2::parseQGemm(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 8 || ninputs == 9);

    layerParams.type = "InnerProductInt8";

    if (constBlobs.find(node_proto.input(3)) == constBlobs.end())
        CV_Error(Error::StsNotImplemented, "Variable weights is not supported");

    Mat weights = getBlob(node_proto, 3);

    if (!layerParams.get<int>("transB", 0))
    {
        transpose(weights, weights);
    }

    CV_Assert(layerParams.get<float>("alpha", 1) == 1.0f);
    CV_Assert(layerParams.get<int>("transA", 0) == 0);

    int firstInpDims = outShapes[node_proto.input(0)].size();

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    int outCn = weights.size[0];
    int secondInpDims = weights.dims;

    Mat w_scale = getBlob(node_proto, 4);
    CV_Assert(w_scale.total() == 1 || w_scale.total() == outCn);
    bool per_channel = w_scale.total() == outCn;
    Mat wt_sc = (w_scale.total() == outCn) ? w_scale : Mat(1, outCn, CV_32F, Scalar(w_scale.at<float>(0)));

    Mat w_zp = getBlob(node_proto, 5);
    int8_t* ptrZp = w_zp.ptr<int8_t>(0);

    for (int i = 0; i < w_zp.total(); i++)
    {
        if (ptrZp[i] != (int8_t)0)
            CV_Error(Error::StsUnsupportedFormat, "The zero-point non-zero case of W is not supported!");
    }

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 7));
    int8_t out_zp = ninputs == 9 ? getScalarFromMat<int8_t>(getBlob(node_proto, 8)) : 0;

    Mat bias;
    if (constBlobs.find(node_proto.input(6)) != constBlobs.end())
        bias = getBlob(node_proto, 6);
    if (bias.empty())
        bias = Mat::zeros(1, outCn, CV_32S);

    Mat biasFused(1, outCn, CV_32S);
    Mat outputMultiplier(1, outCn, CV_32F);
    for (int i = 0; i < outCn; i++)
    {
        biasFused.at<int>(i) = bias.at<int>(i) - inp_zp*(cv::sum(weights.row(i))[0]);
        outputMultiplier.at<float>(i) = (inp_sc * wt_sc.at<float>(i)) / out_sc;
    }

    layerParams.type = "InnerProductInt8";
    layerParams.set("num_output", outCn);
    layerParams.set("axis", firstInpDims - secondInpDims + 1);
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("per_channel", per_channel);

    layerParams.blobs.push_back(weights);
    layerParams.blobs.push_back(biasFused);
    layerParams.blobs.push_back(outputMultiplier);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQEltwise(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    CV_Assert(node_proto.input_size() == 7 || node_proto.input_size() == 8);
    std::string op = (node_proto.op_type() == "QLinearAdd") ? "sum" : "prod";
    int constId = -1;
    for (int i = 0; i < 4; i += 3)
    {
        if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            constId = i;
    }

    float inp_0_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_0_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    float inp_1_sc = getScalarFromMat<float>(getBlob(node_proto, 4));
    int8_t inp_1_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 5));

    // Set 2nd input as the const input
    if (constId == 0)
    {
        cv::swap(inp_0_sc, inp_1_sc);
        cv::swap(inp_0_zp, inp_1_zp);
    }

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 6));

    int8_t out_zp = 0;
    if (node_proto.input_size() == 8)
        out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 7));

    std::vector<float> inp_scales = {inp_0_sc, inp_1_sc};
    std::vector<int8_t> inp_zps = {inp_0_zp, inp_1_zp};

    std::vector<float> coeffs;
    float offset;
    if (op == "sum")
    {
        coeffs = {inp_scales[0]/out_sc, inp_scales[1]/out_sc};
        offset = out_zp - coeffs[0]*inp_zps[0] - coeffs[1]*inp_zps[1];
    }
    else
    {
        coeffs = {inp_scales[0]/out_sc, inp_scales[1]};
        offset = out_zp;
    }

    if (constId != -1)
    {
        Mat blob = getBlob(node_proto, constId);
        if (blob.total() == 1)
        {
            float val = inp_scales[1] * (blob.at<int8_t>(0) - inp_zps[1]);
            float scale = inp_scales[0] / out_sc;
            if (op == "prod")
                scale *= val;

            float shift = out_zp - scale*inp_zps[0];
            if (op == "sum")
                shift += (val/out_sc);

            LayerParams rescaleParams;
            rescaleParams.name = layerParams.name;
            rescaleParams.type = "Requantize";
            rescaleParams.set("depth", CV_8S);
            rescaleParams.set("scale", scale);
            rescaleParams.set("shift", shift);
            rescaleParams.set("isEltwise", true);
            addLayer(rescaleParams, node_proto);
            return;
        }
        else
        {
            MatShape inpShape = outShapes[node_proto.input(3 - constId)];
            if (blob.dims == 2)
                blob = blob.t();

            if (shape(blob) == inpShape)
            {
                LayerParams constParams;
                constParams.name = layerParams.name + "/const";
                constParams.type = "ConstInt8";
                constParams.set("depth", CV_8S);
                constParams.set("scales", inp_1_sc);
                constParams.set("zeropoints", inp_1_zp);
                constParams.blobs.push_back(blob);

                int id = net.addLayer(constParams.name, constParams.type, CV_8S, constParams);
                layer_id.insert(std::make_pair(constParams.name, LayerInfo(id, 0, CV_8S)));
                outShapes[constParams.name] = shape(blob);
                node_proto.set_input(constId, constParams.name);

                layerParams.type = "EltwiseInt8";
                layerParams.set("operation", op);
                layerParams.set("coeff", DictValue::arrayReal(coeffs.data(), coeffs.size()));
                layerParams.set("offset", offset);
            }
            else
            {
                layerParams.type = "ScaleInt8";
                layerParams.set("bias_term", op == "sum");
                int axis = 1;
                for (int i = 0; i < graph_proto->initializer_size(); i++)
                {
                    opencv_onnx::TensorProto tensor_proto = graph_proto->initializer(i);
                    if (tensor_proto.name() == node_proto.input(constId))
                    {
                        axis = inpShape.size() - tensor_proto.dims_size();
                        break;
                    }
                }
                layerParams.set("axis", axis);
                blob = blob.reshape(1, 1);
                Mat blob_dequantized;
                blob.convertTo(blob_dequantized, CV_32F, inp_scales[1], -(inp_scales[1] * inp_zps[1]));
                layerParams.blobs.push_back(blob_dequantized);
            }
        }
    }
    else if (outShapes[node_proto.input(0)] == outShapes[node_proto.input(3)])
    {
        layerParams.type = "EltwiseInt8";
        layerParams.set("operation", op);
        layerParams.set("coeff", DictValue::arrayReal(coeffs.data(), coeffs.size()));
        layerParams.set("offset", offset);
    }
    else
    {
        layerParams.type = "ScaleInt8";
        layerParams.set("bias_term", op == "sum");
    }

    layerParams.set("input_scales", DictValue::arrayReal(inp_scales.data(), inp_scales.size()));
    layerParams.set("input_zeropoints", DictValue::arrayInt(inp_zps.data(), inp_zps.size()));
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);

    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQLeakyRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 4 || node_proto.input_size() == 5);

    float slope = layerParams.get<float>("alpha");
    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t out_zp = node_proto.input_size() == 4 ? 0 : getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    Mat lookUpTable(1, 256, CV_8S);
    int8_t* table = lookUpTable.ptr<int8_t>();
    for (int i = -128; i < 128; i++)
    {
        float x = inp_sc*(i - inp_zp);
        float y = x >= 0.f ? x : slope*x;
        int quantized = out_zp + cvRound(y/out_sc);
        table[i+128] = saturate_cast<int8_t>(quantized);
    }

    layerParams.type = "ReLUInt8";
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("slope", slope);
    layerParams.blobs.push_back(lookUpTable);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQSigmoid(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 4 || node_proto.input_size() == 5);

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t out_zp = node_proto.input_size() == 4 ? 0 : getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    Mat lookUpTable(1, 256, CV_8S);
    int8_t* table = lookUpTable.ptr<int8_t>();
    for (int i = -128; i < 128; i++)
    {
        float x = inp_sc*(i - inp_zp);
        float y = 1.f/(1.f + std::exp(-x));
        int quantized = out_zp + cvRound(y/out_sc);
        table[i+128] = saturate_cast<int8_t>(quantized);
    }

    layerParams.type = "SigmoidInt8";
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.blobs.push_back(lookUpTable);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQAvgPool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 4 || node_proto.input_size() == 5);

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t out_zp = node_proto.input_size() == 4 ? 0 : getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    layerParams.type = "PoolingInt8";
    layerParams.set("pool", "ave");
    layerParams.set("global_pooling", node_proto.op_type() == "QLinearGlobalAveragePool");
    layerParams.set("multiplier", inp_sc/out_sc);
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQConcat(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    layerParams.type = "ConcatInt8";
    int num_inputs = node_proto.input_size();

    float out_scale = getScalarFromMat<float>(getBlob(node_proto, 0));
    int8_t out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 1));

    for (int i = 2; i < num_inputs; i += 3)
    {
        float inp_scale = getScalarFromMat<float>(getBlob(node_proto, i + 1));
        int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, i + 2));

        if (inp_scale != out_scale || inp_zp != out_zp)
        {
            float scale = inp_scale/out_scale;
            float shift = out_zp - scale*inp_zp;

            if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            {
                Mat blob = getBlob(node_proto, i);
                Mat blob_rescaled;
                blob.convertTo(blob_rescaled, CV_8S, scale, shift);
                constBlobs[node_proto.input(i)] = blob_rescaled;
            }
            else
            {
                LayerParams rescaleParams;
                rescaleParams.name = node_proto.input(i) + "/rescale";
                rescaleParams.type = "Requantize";
                rescaleParams.set("depth", CV_8S);
                rescaleParams.set("scale", scale);
                rescaleParams.set("shift", shift);
                rescaleParams.set("isEltwise", false);

                opencv_onnx::NodeProto proto;
                proto.add_input(node_proto.input(i));
                proto.add_output(rescaleParams.name);
                addLayer(rescaleParams, proto);
                node_proto.set_input(i, rescaleParams.name);
            }
        }
    }

    bool hasVariableInps = false;
    for (int i = 2; i < num_inputs; i += 3)
    {
        if (layer_id.find(node_proto.input(i)) != layer_id.end())
        {
            hasVariableInps = true;
            break;
        }
    }

    if (!hasVariableInps)
    {
        std::vector<Mat> inputs, concatenated;
        MatShape inputShape;
        for (size_t i = 2; i < num_inputs; i += 3)
        {
            Mat blob = getBlob(node_proto, i);
            if (blob.size.dims() > inputShape.size())
            {
                inputShape = shape(blob);
            }
            inputs.push_back(blob);
        }

        int axis = layerParams.get<int>("axis", 1);
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            MatShape targetShape = inputShape;
            targetShape[axis] = shape(inputs[i])[axis];
            CV_CheckEQ(total(targetShape), total(shape(inputs[i])), "");
            inputs[i] = inputs[i].reshape(0, targetShape);
        }
        runLayer(layerParams, inputs, concatenated);
        CV_Assert(concatenated.size() == 1);
        addConstant(layerParams.name, concatenated[0]);
        return;
    }
    else
    {
        for (int i = 2; i < num_inputs; i += 3)
        {
            if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            {
                LayerParams constParams;
                constParams.name = node_proto.input(i);
                constParams.type = "ConstInt8";
                constParams.blobs.push_back(getBlob(node_proto, i));
                constParams.set("depth", CV_8S);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }
        }
    }
    layerParams.set("scales", out_scale);
    layerParams.set("zeropoints", out_zp);
    addLayer(layerParams, node_proto);
}

void ONNXImporter2::parseQSoftmax(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 5, "DNN/ONNX: QLinearSoftmax requires 5 inputs, X, X_scale, X_zero_point, Y_scale, Y_zero_point");

    int opset = layerParams.get<int>("opset");
    if (opset < 13) {
        layerParams.set("coerced_2d", true);
    }

    float x_scale = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t x_zero_point = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float y_scale = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t y_zero_point = getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    layerParams.type = "SoftmaxInt8";
    // layerParams also has "axis" and "opset" attrs
    layerParams.set("input_scale", x_scale);
    layerParams.set("input_zeropoint", x_zero_point);
    layerParams.set("scales", y_scale);
    layerParams.set("zeropoints", y_zero_point);
    addLayer(layerParams, node_proto);
}*/

void ONNXImporter2::parseAttention(LayerParams& params, const opencv_onnx::NodeProto& node_proto) {
    int i, n_inputs = node_proto.input_size();
    CV_CheckTrue(params.has("num_heads"), "ONNXImporter2/parseAttention: num_heads is required but missing");
    CV_CheckTrue(params.has("qkv_hidden_sizes"), "ONNXImporter2/parseAttention: qkv_hidden_sizes is required but missing");

    auto param_qkv_hidden_sizes = params.get("qkv_hidden_sizes");
    CV_CheckEQ(param_qkv_hidden_sizes.size(), 3, "ONNXImporter2/parseAttention: qkv_hidden_sizes is must and only have three elements");

    for (i = 1; i < n_inputs; i++) {
        if (!net.isConstArg(node_inputs[i]))
            break;
    }

    if (i == n_inputs) {
        for (i = 1; i < n_inputs; i++) {
            Mat blob = net.argTensor(node_inputs[i]);
            params.blobs.push_back(blob);
        }
        n_inputs = 1;
    }

    addLayer(params, node_proto, n_inputs);
}

// Domain: ai.onnx (default)
// URL: https://github.com/onnx/onnx/blob/master/docs/Operators.md
void ONNXImporter2::buildDispatchMap_ONNX_AI(int opset_version)
{
    CV_UNUSED(opset_version);
    DispatchMap dispatch;

    dispatch["ArgMax"] = dispatch["ArgMin"] = &ONNXImporter2::parseArgMinMax;
    dispatch["MaxUnpool"] = &ONNXImporter2::parseMaxUnpool;
    dispatch["MaxPool"] = &ONNXImporter2::parseMaxPool;
    dispatch["AveragePool"] = &ONNXImporter2::parseAveragePool;
    dispatch["GlobalAveragePool"] = dispatch["GlobalMaxPool"] = &ONNXImporter2::parseGlobalPool;
    dispatch["ReduceMax"] = dispatch["ReduceMin"] = dispatch["ReduceMean"] = dispatch["ReduceSum"] =
            dispatch["ReduceSumSquare"] = dispatch["ReduceProd"] = dispatch["ReduceL1"] =
            dispatch["ReduceL2"] = dispatch["ReduceLogSum"] = dispatch["ReduceLogSumExp"] = &ONNXImporter2::parseReduce;
    dispatch["Slice"] = &ONNXImporter2::parseSlice;
    dispatch["Split"] = &ONNXImporter2::parseSplit;
    dispatch["Neg"] = &ONNXImporter2::parseNeg;
    dispatch["Constant"] = &ONNXImporter2::parseConstant;
    dispatch["LSTM"] = &ONNXImporter2::parseLSTM;
    dispatch["GRU"] = &ONNXImporter2::parseGRU;
    dispatch["ImageScaler"] = &ONNXImporter2::parseImageScaler;
    dispatch["Clip"] = &ONNXImporter2::parseClip;
    dispatch["LeakyRelu"] = &ONNXImporter2::parseLeakyRelu;
    dispatch["Relu"] = &ONNXImporter2::parseRelu;
    dispatch["Elu"] = &ONNXImporter2::parseElu;
    dispatch["Tanh"] = &ONNXImporter2::parseTanh;
    dispatch["Abs"] = &ONNXImporter2::parseAbs;
    dispatch["PRelu"] = &ONNXImporter2::parsePRelu;
    dispatch["LRN"] = &ONNXImporter2::parseLRN;
    dispatch["InstanceNormalization"] = &ONNXImporter2::parseInstanceNormalization;
    dispatch["BatchNormalization"] = &ONNXImporter2::parseBatchNormalization;
    dispatch["Gemm"] = &ONNXImporter2::parseGemm;
    dispatch["MatMul"] = &ONNXImporter2::parseMatMul;
    dispatch["Conv"] = &ONNXImporter2::parseConv;
    dispatch["ConvTranspose"] = &ONNXImporter2::parseConvTranspose;
    dispatch["Transpose"] = &ONNXImporter2::parseTranspose;
    dispatch["Squeeze"] = &ONNXImporter2::parseSqueeze;
    dispatch["Flatten"] = &ONNXImporter2::parseFlatten;
    dispatch["Unsqueeze"] = &ONNXImporter2::parseUnsqueeze;
    dispatch["Expand"] = &ONNXImporter2::parseExpand;
    dispatch["Reshape"] = &ONNXImporter2::parseReshape;
    dispatch["Pad"] = &ONNXImporter2::parsePad;
    dispatch["Shape"] = &ONNXImporter2::parseShape;
    dispatch["Cast"] = &ONNXImporter2::parseCast;
    dispatch["ConstantFill"] = dispatch["ConstantOfShape"] = &ONNXImporter2::parseConstantOfShape;
    dispatch["Gather"] = &ONNXImporter2::parseGather;
    dispatch["GatherElements"] = &ONNXImporter2::parseGatherElements;
    dispatch["Concat"] = &ONNXImporter2::parseConcat;
    dispatch["Resize"] = &ONNXImporter2::parseResize;
    dispatch["Upsample"] = &ONNXImporter2::parseUpsample;
    dispatch["SoftMax"] = dispatch["Softmax"] = dispatch["LogSoftmax"] = &ONNXImporter2::parseSoftMax;
    dispatch["DetectionOutput"] = &ONNXImporter2::parseDetectionOutput;
    dispatch["CumSum"] = &ONNXImporter2::parseCumSum;
    dispatch["SpaceToDepth"] = dispatch["DepthToSpace"] = &ONNXImporter2::parseDepthSpaceOps;
    dispatch["ScatterElements"] = dispatch["Scatter"] = dispatch["ScatterND"] = &ONNXImporter2::parseScatter;
    dispatch["Tile"] = &ONNXImporter2::parseTile;
    dispatch["LayerNormalization"] = &ONNXImporter2::parseLayerNorm;
    dispatch["GroupNormalization"] = &ONNXImporter2::parseInstanceNormalization;

    dispatch["Equal"] = dispatch["Greater"] = dispatch["Less"] = dispatch["Pow"] = dispatch["Add"] =
            dispatch["Sub"] = dispatch["Mul"] = dispatch["Div"] = dispatch["GreaterOrEqual"] =
            dispatch["LessOrEqual"] = dispatch["Mod"] = dispatch["And"] = dispatch["Or"] = dispatch["Xor"] = &ONNXImporter2::parseElementWise;

    dispatch["Sum"] = dispatch["Min"] = dispatch["Max"] = dispatch["Mean"] = &ONNXImporter2::parseElementWise;
    dispatch["Where"] = &ONNXImporter2::parseElementWise;
    dispatch["Range"] = &ONNXImporter2::parseRange;
    dispatch["Einsum"] = &ONNXImporter2::parseEinsum;
    dispatch["TopK"] = &ONNXImporter2::parseTopK2;  // Add this line

    std::vector<std::string> simpleLayers {
        "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos",
        "Cosh", "Dropout", "Erf", "Exp", "Floor", "HardSigmoid", "HardSwish",
        "Identity", "Log", "Not", "Round", "Reciprocal", "Selu", "Sign", "Sigmoid", "Sin", "Sinh",
        "Softplus", "Softsign", "Shrink", "Sqrt", "Tan", "ThresholdedRelu", "Gelu",
        "GeluApproximation"
    };
    for (const auto& name : simpleLayers)
    {
        dispatch[name] = &ONNXImporter2::parseSimpleLayers;
    }

    // BUG: https://github.com/opencv/opencv/issues/26310
    // ai.onnx: opset 10+
    dispatch["DequantizeLinear"] = &ONNXImporter2::parseDequantizeLinear;
    dispatch["QuantizeLinear"] = &ONNXImporter2::parseQuantizeLinear;
    //dispatch["QLinearConv"] = &ONNXImporter2::parseQConv;
    //dispatch["QLinearMatMul"] = &ONNXImporter2::parseQMatMul;

    // com.microsft: This operator is added for compatibility via onnx graph simplifier.
    //               Opset domain cannot be modified from onnx_graph_simplifier.cpp so this
    //               operator cannot be parsed if only added in buildDispatchMap_COM_MICROSOFT
    dispatch["Attention"] = &ONNXImporter2::parseAttention;

    domain_dispatch_map[str_domain_ai_onnx] = dispatch;
}

// Domain: com.microsoft
// URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
void ONNXImporter2::buildDispatchMap_COM_MICROSOFT(int opset_version)
{
    CV_UNUSED(opset_version);
    DispatchMap dispatch;

    // BUG: https://github.com/opencv/opencv/issues/26310
    //dispatch["QLinearAdd"] = dispatch["QLinearMul"] = &ONNXImporter2::parseQEltwise;
    //dispatch["QLinearAveragePool"] = dispatch["QLinearGlobalAveragePool"] = &ONNXImporter2::parseQAvgPool;
    //dispatch["QLinearLeakyRelu"] = &ONNXImporter2::parseQLeakyRelu;
    //dispatch["QLinearSigmoid"] = &ONNXImporter2::parseQSigmoid;
    //dispatch["QLinearConcat"] = &ONNXImporter2::parseQConcat;
    //dispatch["QGemm"] = &ONNXImporter2::parseQGemm;
    //dispatch["QLinearSoftmax"] = &ONNXImporter2::parseQSoftmax;
    dispatch["Attention"] = &ONNXImporter2::parseAttention;

    domain_dispatch_map["com.microsoft"] = dispatch;
}


Net readNetFromONNX2(const String& onnxFile)
{
    ONNXImporter2 importer;
    Net net = importer.parseFile(onnxFile.c_str());
    if (net.getMainGraph()) {
        net.getImpl()->modelFileName = onnxFile;
    }
    return net;
}

Net readNetFromONNX2(const char* buffer, size_t size)
{
    ONNXImporter2 importer;
    return importer.parseBuffer(buffer, size);
}

Net readNetFromONNX2(const std::vector<uchar>& buffer)
{
    ONNXImporter2 importer;
    return importer.parseBuffer(buffer.data(), buffer.size());
}

#else  // HAVE_PROTOBUF

#define DNN_PROTOBUF_UNSUPPORTED() CV_Error(Error::StsError, "DNN/ONNX: Build OpenCV with Protobuf to import ONNX models")

Net readNetFromONNX2(const String&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromONNX2(const char*, size_t) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromONNX2(const std::vector<uchar>&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

#endif  // HAVE_PROTOBUF

CV__DNN_INLINE_NS_END
}} // namespace
