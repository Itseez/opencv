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

#include "../precomp.hpp"
#include "../net_impl.hpp"

#ifdef HAVE_PROTOBUF
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/reflection.h>
#include "caffe_io.hpp"
#endif

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/fp_control_utils.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_PROTOBUF
using ::google::protobuf::RepeatedFieldRef;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;

namespace
{

template<typename T>
static cv::String toString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

static inline
MatShape parseBlobShape(const caffe::BlobShape& _input_shape)
{
    MatShape shape;
    for (int i = 0; i < _input_shape.dim_size(); i++)
    {
        shape.push_back((int)_input_shape.dim(i));
    }
    return shape;
}

class CaffeImporter
{
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    caffe::NetParameter net;
    caffe::NetParameter netBinary;

public:

    CaffeImporter(const char *prototxt, const char *caffeModel)
    {
        CV_TRACE_FUNCTION();

        ReadNetParamsFromTextFileOrDie(prototxt, &net);

        if (caffeModel && caffeModel[0])
            ReadNetParamsFromBinaryFileOrDie(caffeModel, &netBinary);
    }

    CaffeImporter(const char *dataProto, size_t lenProto,
                  const char *dataModel, size_t lenModel)
    {
        CV_TRACE_FUNCTION();

        ReadNetParamsFromTextBufferOrDie(dataProto, lenProto, &net);

        if (dataModel != NULL && lenModel > 0)
            ReadNetParamsFromBinaryBufferOrDie(dataModel, lenModel, &netBinary);
    }

    void extractCustomParams(const google::protobuf::UnknownFieldSet& unknownFields, cv::dnn::LayerParams &params)
    {
        const int numFields = unknownFields.field_count();
        for (int i = 0; i < numFields; ++i)
        {
            const google::protobuf::UnknownField& field = unknownFields.field(i);
            CV_Assert(field.type() == google::protobuf::UnknownField::TYPE_GROUP);
            CV_CheckGE(field.group().field_count(), 2, "UnknownField should have at least 2 items: name and value");
            std::string fieldName(field.group().field(0).length_delimited());
            std::string fieldValue(field.group().field(1).length_delimited());
            params.set(fieldName, fieldValue);
        }
    }

    void addParam(const Message &msg, const FieldDescriptor *field, cv::dnn::LayerParams &params)
    {
        const Reflection *refl = msg.GetReflection();
        int type = field->cpp_type();
        bool isRepeated = field->is_repeated();
        const std::string name(field->name());

        #define SET_UP_FILED(getter, arrayConstr, gtype)                                    \
            if (isRepeated) {                                                               \
                const RepeatedFieldRef<gtype> v = refl->GetRepeatedFieldRef<gtype>(msg, field);  \
                params.set(name, DictValue::arrayConstr(v.begin(), (int)v.size()));                  \
            }                                                                               \
            else {                                                                          \
                params.set(name, refl->getter(msg, field));                               \
            }

        switch (type)
        {
        case FieldDescriptor::CPPTYPE_INT32:
            SET_UP_FILED(GetInt32, arrayInt, ::google::protobuf::int32);
            break;
        case FieldDescriptor::CPPTYPE_UINT32:
            SET_UP_FILED(GetUInt32, arrayInt, ::google::protobuf::uint32);
            break;
        case FieldDescriptor::CPPTYPE_INT64:
            SET_UP_FILED(GetInt32, arrayInt, ::google::protobuf::int64);
            break;
        case FieldDescriptor::CPPTYPE_UINT64:
            SET_UP_FILED(GetUInt32, arrayInt, ::google::protobuf::uint64);
            break;
        case FieldDescriptor::CPPTYPE_BOOL:
            SET_UP_FILED(GetBool, arrayInt, bool);
            break;
        case FieldDescriptor::CPPTYPE_DOUBLE:
            SET_UP_FILED(GetDouble, arrayReal, double);
            break;
        case FieldDescriptor::CPPTYPE_FLOAT:
            SET_UP_FILED(GetFloat, arrayReal, float);
            break;
        case FieldDescriptor::CPPTYPE_STRING:
            if (isRepeated) {
                const RepeatedFieldRef<std::string> v = refl->GetRepeatedFieldRef<std::string>(msg, field);
                params.set(name, DictValue::arrayString(v.begin(), (int)v.size()));
            }
            else {
                params.set(name, refl->GetString(msg, field));
            }
            break;
        case FieldDescriptor::CPPTYPE_ENUM:
            if (isRepeated) {
                int size = refl->FieldSize(msg, field);
                std::vector<cv::String> buf(size);
                for (int i = 0; i < size; i++)
                    buf[i] = refl->GetRepeatedEnum(msg, field, i)->name();
                params.set(name, DictValue::arrayString(buf.begin(), size));
            }
            else {
                params.set(name, std::string(refl->GetEnum(msg, field)->name()));
            }
            break;
        default:
            CV_Error(Error::StsError, "Unknown type \"" + String(field->type_name()) + "\" in prototxt");
        }
    }

    inline static bool ends_with_param(const std::string &str)
    {
        static const std::string _param("_param");
        return (str.size() >= _param.size()) && str.compare(str.size() - _param.size(), _param.size(), _param) == 0;
    }

    void extractLayerParams(const Message &msg, cv::dnn::LayerParams &params, bool isInternal = false)
    {
        const Descriptor *msgDesc = msg.GetDescriptor();
        const Reflection *msgRefl = msg.GetReflection();

        for (int fieldId = 0; fieldId < msgDesc->field_count(); fieldId++)
        {
            const FieldDescriptor *fd = msgDesc->field(fieldId);

            if (!isInternal && !ends_with_param(std::string(fd->name())))
                continue;

            const google::protobuf::UnknownFieldSet& unknownFields = msgRefl->GetUnknownFields(msg);
            bool hasData =  fd->is_required() ||
                            (fd->is_optional() && msgRefl->HasField(msg, fd)) ||
                            (fd->is_repeated() && msgRefl->FieldSize(msg, fd) > 0) ||
                            !unknownFields.empty();
            if (!hasData)
                continue;

            extractCustomParams(unknownFields, params);
            if (fd->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE)
            {
                if (fd->is_repeated()) //Extract only first item!
                    extractLayerParams(msgRefl->GetRepeatedMessage(msg, fd, 0), params, true);
                else
                    extractLayerParams(msgRefl->GetMessage(msg, fd), params, true);
            }
            else
            {
                addParam(msg, fd, params);
            }
        }
    }

    void blobShapeFromProto(const caffe::BlobProto &pbBlob, MatShape& shape)
    {
        shape.clear();
        if (pbBlob.has_num() || pbBlob.has_channels() || pbBlob.has_height() || pbBlob.has_width())
        {
            shape.push_back(pbBlob.num());
            shape.push_back(pbBlob.channels());
            shape.push_back(pbBlob.height());
            shape.push_back(pbBlob.width());
        }
        else if (pbBlob.has_shape())
        {
            shape = parseBlobShape(pbBlob.shape());
        }
        else
            shape.resize(1, 1);  // Is a scalar.
    }

    void blobFromProto(const caffe::BlobProto &pbBlob, cv::Mat &dstBlob)
    {
        MatShape shape;
        blobShapeFromProto(pbBlob, shape);

        dstBlob.create((int)shape.size(), &shape[0], CV_32F);
        if (pbBlob.data_size())
        {
            // Single precision floats.
            CV_Assert(pbBlob.data_size() == (int)dstBlob.total());

            CV_DbgAssert(pbBlob.GetDescriptor()->FindFieldByLowercaseName("data")->cpp_type() == FieldDescriptor::CPPTYPE_FLOAT);
            Mat(dstBlob.dims, &dstBlob.size[0], CV_32F, (void*)pbBlob.data().data()).copyTo(dstBlob);
        }
        else
        {
            CV_Assert(pbBlob.has_raw_data());
            const std::string& raw_data = pbBlob.raw_data();
            if (pbBlob.raw_data_type() == caffe::FLOAT16)
            {
                // Half precision floats.
                CV_Assert(raw_data.size() / 2 == (int)dstBlob.total());

                Mat halfs((int)shape.size(), &shape[0], CV_16FC1, (void*)raw_data.c_str());
                halfs.convertTo(dstBlob, CV_32F);
            }
            else if (pbBlob.raw_data_type() == caffe::FLOAT)
            {
                CV_Assert(raw_data.size() / 4 == (int)dstBlob.total());
                Mat((int)shape.size(), &shape[0], CV_32FC1, (void*)raw_data.c_str()).copyTo(dstBlob);
            }
            else
                CV_Error(Error::StsNotImplemented, "Unexpected blob data type");
        }
    }

    void extractBinaryLayerParams(const caffe::LayerParameter& layer, LayerParams& layerParams)
    {
        const std::string &name = layer.name();

        int li;
        for (li = 0; li != netBinary.layer_size(); li++)
        {
            const caffe::LayerParameter& binLayer = netBinary.layer(li);
            // Break if the layer name is the same and the blobs are not cleared
            if (binLayer.name() == name && binLayer.blobs_size() != 0)
                break;
        }

        if (li == netBinary.layer_size())
            return;

        caffe::LayerParameter* binLayer = netBinary.mutable_layer(li);
        const int numBlobs = binLayer->blobs_size();
        std::vector<caffe::BlobProto*> blobs(numBlobs);
        binLayer->mutable_blobs()->ExtractSubrange(0, numBlobs, blobs.data());
        layerParams.blobs.resize(numBlobs);
        for (int bi = 0; bi < numBlobs; bi++)
        {
            blobFromProto(*blobs[bi], layerParams.blobs[bi]);
            delete blobs[bi];
        }
    }

    Ptr<Layer> addLayer(Net& dstNet,
                        const String& type,
                        const String& name,
                        LayerParams& layerParams,
                        const std::vector<String>& inputs,
                        const std::vector<String>& outputs)
    {
        layerParams.type = type;
        layerParams.name = name;
        Ptr<Layer> layer = LayerFactory::createLayerInstance(type, layerParams);
        if (!layer) {
            CV_Error(Error::StsError, "Can't create layer " + name + " with type " + type);
            return nullptr;
        }

        for (const String& inputName : inputs)
            layer->inputs.push_back(dstNet.getArg(inputName));
        for (const String& outputName : outputs)
            layer->outputs.push_back(dstNet.getArg(outputName));
        layer->netimpl = dstNet.getImpl();
        CV_Assert(dstNet.getImpl()->dump_indent == 3);
        return layer;
    }

    struct BlobNote
    {
        BlobNote(const std::string &_name, int _layerId, int _outNum) :
            name(_name), layerId(_layerId), outNum(_outNum) {}

        std::string name;
        int layerId, outNum;
    };

    std::vector<BlobNote> addedBlobs;
    std::map<String, int> layerCounter;

    void populateNet(Net dstNet, bool newEngine)
    {
        CV_TRACE_FUNCTION();

        int layersSize = net.layer_size();
        layerCounter.clear();

        // OLD ENGINE
        if(!newEngine)
        {
            addedBlobs.clear();
            addedBlobs.reserve(layersSize + 1);
        }
        std::vector<String> netInputs(net.input_size());
        std::vector<MatShape> inp_shapes;

        // NEW ENGINE
        Net::Impl* netImpl = dstNet.getImpl();
        std::vector<Ptr<Layer>> curr_prog;
        std::vector<Arg> modelInputs, modelOutputs;

        {
            int net_input_size = net.input_size();
            for (int inNum = 0; inNum < net_input_size; inNum++)
            {
                if (newEngine)
                {
                    modelInputs.push_back(netImpl->newArg(net.input(inNum), DNN_ARG_INPUT));
                    netImpl->args.at(modelInputs.back().idx).type = CV_32F;
                }
                else
                {
                    addedBlobs.push_back(BlobNote(net.input(inNum), 0, inNum));
                    netInputs[inNum] = net.input(inNum);
                }
            }

            if (net.input_dim_size() > 0)  // deprecated in Caffe proto
            {
                int net_input_dim_size = net.input_dim_size();
                CV_Check(net_input_dim_size, net_input_dim_size % 4 == 0, "");
                CV_CheckEQ(net_input_dim_size, net_input_size * 4, "");
                for (int inp_id = 0; inp_id < net_input_size; inp_id++)
                {
                    int dim = inp_id * 4;
                    MatShape shape(4);
                    shape[0] = net.input_dim(dim);
                    shape[1] = net.input_dim(dim+1);
                    shape[2] = net.input_dim(dim+2);
                    shape[3] = net.input_dim(dim+3);
                    if (newEngine)
                        netImpl->args.at(modelInputs[inp_id].idx).shape = shape;
                    else
                        inp_shapes.push_back(shape);
                }
            }
            else if (net.input_shape_size() > 0)  // deprecated in Caffe proto
            {
                int net_input_shape_size = net.input_shape_size();
                CV_CheckEQ(net_input_shape_size, net_input_size, "");
                for (int inp_id = 0; inp_id < net_input_shape_size; inp_id++)
                {
                    MatShape shape = parseBlobShape(net.input_shape(inp_id));
                    if (newEngine)
                        netImpl->args.at(modelInputs[inp_id].idx).shape = shape;
                    else
                        inp_shapes.push_back(shape);
                }
            }
            else
            {
                for (int inp_id = 0; inp_id < net_input_size; inp_id++)
                {
                    MatShape shape; // empty
                    if (newEngine)
                        netImpl->args.at(modelInputs[inp_id].idx).shape = shape;
                    else
                        inp_shapes.push_back(shape);
                }
            }
        }

        if (newEngine && net.layer(layersSize - 1).type() == "Silence")
        {
            CV_LOG_WARNING(NULL, "Caffe parser: Silence layer was ignored");
            layersSize--;
        }

        for (int li = 0; li < layersSize; li++)
        {
            const caffe::LayerParameter &layer = net.layer(li);
            String name = layer.name();
            String type = layer.type();
            LayerParams layerParams;

            extractLayerParams(layer, layerParams);
            extractBinaryLayerParams(layer, layerParams);

            if (newEngine && li == layersSize - 1)
            {
                for (int outNum = 0; outNum < layer.top_size(); outNum++)
                    modelOutputs.push_back(netImpl->newArg(layer.top(outNum), DNN_ARG_OUTPUT));
            }

            int repetitions = layerCounter[name]++;
            if (repetitions)
                name += String("_") + toString(repetitions);

            if (type == "Input")
            {
                for (int outNum = 0; outNum < layer.top_size(); outNum++)
                {
                    if (newEngine)
                    {
                        modelInputs.push_back(netImpl->newArg(layer.top(outNum), DNN_ARG_INPUT));
                        netImpl->args.at(modelInputs.back().idx).type = CV_32F;
                    }
                    else
                    {
                        addOutput(layer, 0, outNum);
                        addedBlobs.back().outNum = netInputs.size();
                        netInputs.push_back(addedBlobs.back().name);
                    }
                }
                if (layer.has_input_param())
                {
                    const caffe::InputParameter &inputParameter = layer.input_param();
                    int input_shape_size = inputParameter.shape_size();
                    CV_CheckEQ(input_shape_size, layer.top_size(), "");
                    for (int inp_id = 0; inp_id < input_shape_size; inp_id++)
                    {
                        MatShape shape = parseBlobShape(inputParameter.shape(inp_id));
                        if (newEngine)
                        {
                            int inputIdx = modelInputs.size() - input_shape_size + inp_id;
                            netImpl->args.at(modelInputs[inputIdx].idx).shape = shape;
                        }
                        else
                        {
                            inp_shapes.push_back(shape);
                        }
                    }
                }
                continue;
            }
            else if (type == "BatchNorm")
            {
                if (!layerParams.get<bool>("use_global_stats", true))
                {
                    CV_Assert_N(layer.bottom_size() == 1, layer.top_size() == 1);

                    LayerParams mvnParams;
                    mvnParams.set("eps", layerParams.get<float>("eps", 1e-5));
                    std::string mvnName = name + "/mvn";

                    int repetitions = layerCounter[mvnName]++;
                    if (repetitions)
                        mvnName += String("_") + toString(repetitions);

                    if (newEngine)
                    {
                        Ptr<Layer> netLayer = addLayer(
                            dstNet, "MVN", mvnName, mvnParams,
                            {layer.bottom(0)},
                            {layer.top(0)});
                        curr_prog.push_back(netLayer);
                        continue;
                    }
                    else
                    {
                        int mvnId = dstNet.addLayer(mvnName, "MVN", mvnParams);
                        addInput(layer.bottom(0), mvnId, 0, dstNet);
                        addOutput(layer, mvnId, 0);
                        net.mutable_layer(li)->set_bottom(0, layer.top(0));
                        layerParams.blobs[0].setTo(0);  // mean
                        layerParams.blobs[1].setTo(1);  // std
                    }
                }
            }
            else if (type == "Axpy")
            {
                CV_Assert_N(layer.bottom_size() == 3, layer.top_size() == 1);

                std::string scaleName = name + "/scale";
                int repetitions = layerCounter[scaleName]++;
                if (repetitions) {
                    scaleName += String("_") + toString(repetitions);
                }

                LayerParams scaleParams;
                scaleParams.set("axis", 1);
                scaleParams.set("has_bias", false);

                if (newEngine)
                {
                    std::string intermediateTensor = scaleName + "_intermediate_output";
                    Ptr<Layer> netLayerScale= addLayer(
                        dstNet, "Scale", scaleName, scaleParams,
                        {layer.bottom(2), layer.bottom(0)},
                        {intermediateTensor});
                    curr_prog.push_back(netLayerScale);

                    LayerParams eltwiseParams;
                    Ptr<Layer> netLayerEltwise = addLayer(
                        dstNet, "Eltwise", name, eltwiseParams,
                        {intermediateTensor, layer.bottom(1)},
                        {layer.top(0)});
                    curr_prog.push_back(netLayerEltwise);
                    continue;
                }
                else
                {
                    int scaleId = dstNet.addLayer(scaleName, "Scale", scaleParams);
                    addInput(layer.bottom(2), scaleId, 0, dstNet);
                    addInput(layer.bottom(0), scaleId, 1, dstNet);
                    addOutput(layer, scaleId, 0);
                    net.mutable_layer(li)->set_bottom(0, layer.top(0));
                    net.mutable_layer(li)->mutable_bottom()->RemoveLast();
                    type = "Eltwise";
                }
            }
            else if (type == "Resample")
            {
                CV_Assert(layer.bottom_size() == 1 || layer.bottom_size() == 2);
                type = "Resize";
                String interp = toLowerCase(layerParams.get<String>("type"));
                layerParams.set("interpolation", interp == "linear" ? "bilinear" : interp);

                if (layerParams.has("factor"))
                {
                    float factor = layerParams.get<float>("factor");
                    CV_Assert(layer.bottom_size() != 2 || factor == 1.0);
                    layerParams.set("zoom_factor", factor);

                    if ((interp == "linear" && factor != 1.0) ||
                        (interp == "nearest" && factor < 1.0))
                        CV_Error(Error::StsNotImplemented, "Unsupported Resample mode");
                }
            }
            else if ("Convolution" == type)
            {
                CV_Assert(layer.bottom_size() == layer.top_size());
                for (int i = 0; i < layer.bottom_size(); i++)
                {
                    if (newEngine)
                    {
                        Ptr<Layer> netLayer = addLayer(
                            dstNet, type, layer.top(i), layerParams,
                            {layer.bottom(i)}, {layer.top(i)});
                        curr_prog.push_back(netLayer);
                    }
                    else
                    {
                        int conv_id = dstNet.addLayer(layer.top(i), type, layerParams);
                        addInput(layer.bottom(i), conv_id, 0, dstNet);
                        addedBlobs.push_back(BlobNote(layer.top(i), conv_id, 0));
                    }
                }
                continue;
            }
            else if ("ConvolutionDepthwise" == type)
            {
                type = "Convolution";
            }
            else if (type == "Softmax"){
                // set default axis to 1
                if(!layerParams.has("axis"))
                    layerParams.set("axis", 1);
            }
            else if ("Proposal" == type)
            {
                if (newEngine && layer.top_size() == 1)
                {
                    // Add unused optional second output and create the Proposal layer
                    std::vector<string> layerInputs;
                    for (int inNum = 0; inNum < layer.bottom_size(); inNum++)
                        layerInputs.push_back(layer.bottom(inNum));
                    Ptr<Layer> netLayer = addLayer(
                        dstNet, type, name, layerParams,
                        layerInputs, {layer.top(0), name + "___output_scores"});
                    curr_prog.push_back(netLayer);
                    continue;
                }
            }
            else if ("Silence" == type)
            {
                if (newEngine)
                {
                    CV_LOG_WARNING(NULL, "Caffe parser: Silence layer was ignored");
                    continue;
                }
            }

            if (newEngine)
            {
                std::vector<string> layerInputs, layerOutputs;
                for (int inNum = 0; inNum < layer.bottom_size(); inNum++)
                    layerInputs.push_back(layer.bottom(inNum));
                for (int outNum = 0; outNum < layer.top_size(); outNum++)
                    layerOutputs.push_back(layer.top(outNum));

                Ptr<Layer> netLayer = addLayer(
                    dstNet, type, name, layerParams,
                    layerInputs, layerOutputs);
                curr_prog.push_back(netLayer);
            }
            else
            {
                int id = dstNet.addLayer(name, type, layerParams);
                for (int inNum = 0; inNum < layer.bottom_size(); inNum++)
                    addInput(layer.bottom(inNum), id, inNum, dstNet);
                for (int outNum = 0; outNum < layer.top_size(); outNum++)
                    addOutput(layer, id, outNum);
            }
        }

        if (newEngine)
        {
            Ptr<Graph> curr_graph = netImpl->newGraph(net.name(), modelInputs, true);
            curr_graph->setOutputs(modelOutputs);
            curr_graph->setProg(curr_prog);

            netImpl->mainGraph = curr_graph;
            netImpl->modelFormat = DNN_MODEL_CAFFE;
            netImpl->originalLayout = DATA_LAYOUT_NCHW;
            netImpl->prepareForInference();
        }
        else
        {
            dstNet.setInputsNames(netInputs);

            if (inp_shapes.size() > 0)
            {
                CV_CheckEQ(inp_shapes.size(), netInputs.size(), "");
                for (int inp_id = 0; inp_id < inp_shapes.size(); inp_id++)
                    dstNet.setInputShape(netInputs[inp_id], inp_shapes[inp_id]);
            }

            addedBlobs.clear();
        }
    }

    void addOutput(const caffe::LayerParameter &layer, int layerId, int outNum)
    {
        const std::string &name = layer.top(outNum);

        bool haveDups = false;
        for (int idx = (int)addedBlobs.size() - 1; idx >= 0; idx--)
        {
            if (addedBlobs[idx].name == name)
            {
                haveDups = true;
                break;
            }
        }

        if (haveDups)
        {
            bool isInplace = layer.bottom_size() > outNum && layer.bottom(outNum) == name;
            if (!isInplace)
                CV_Error(Error::StsBadArg, "Duplicate blobs produced by multiple sources");
        }

        addedBlobs.push_back(BlobNote(name, layerId, outNum));
    }

    void addInput(const std::string &name, int layerId, int inNum, Net &dstNet)
    {
        int idx;
        for (idx = (int)addedBlobs.size() - 1; idx >= 0; idx--)
        {
            if (addedBlobs[idx].name == name)
                break;
        }

        if (idx < 0)
        {
            CV_Error(Error::StsObjectNotFound, "Can't find output blob \"" + name + "\"");
        }

        dstNet.connect(addedBlobs[idx].layerId, addedBlobs[idx].outNum, layerId, inNum);
    }
};

}

Net readNetFromCaffe(const String &prototxt,
                     const String &caffeModel, /*= String()*/
                     int engine)
{
    static const int engine_forced = (int)utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", ENGINE_AUTO);
    if(engine_forced != ENGINE_AUTO)
        engine = engine_forced;

    CaffeImporter caffeImporter(prototxt.c_str(), caffeModel.c_str());
    Net net;
    caffeImporter.populateNet(net, engine == ENGINE_NEW || engine == ENGINE_AUTO);
    return net;
}

Net readNetFromCaffe(const char *bufferProto, size_t lenProto,
                     const char *bufferModel, size_t lenModel,
                     int engine)
{
    static const int engine_forced = (int)utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", ENGINE_AUTO);
    if(engine_forced != ENGINE_AUTO)
        engine = engine_forced;

    CaffeImporter caffeImporter(bufferProto, lenProto, bufferModel, lenModel);
    Net net;
    caffeImporter.populateNet(net, engine == ENGINE_NEW || engine == ENGINE_AUTO);
    return net;
}

Net readNetFromCaffe(const std::vector<uchar>& bufferProto,
                     const std::vector<uchar>& bufferModel,
                     int engine)
{
    const char* bufferProtoPtr = reinterpret_cast<const char*>(&bufferProto[0]);
    const char* bufferModelPtr = bufferModel.empty() ? NULL :
                                 reinterpret_cast<const char*>(&bufferModel[0]);
    return readNetFromCaffe(bufferProtoPtr, bufferProto.size(),
                            bufferModelPtr, bufferModel.size(),
                            engine);
}

#else  // HAVE_PROTOBUF

#define DNN_PROTOBUF_UNSUPPORTED() CV_Error(Error::StsError, "DNN/Caffe: Build OpenCV with Protobuf to import Caffe models")

Net readNetFromCaffe(const String &, const String &, int) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromCaffe(const char *, size_t, const char *, size_t, int) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromCaffe(const std::vector<uchar>&, const std::vector<uchar>&, int) {
    DNN_PROTOBUF_UNSUPPORTED();
}

#endif  // HAVE_PROTOBUF

CV__DNN_INLINE_NS_END
}} // namespace
