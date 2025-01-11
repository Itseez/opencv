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

#include "precomp.hpp"
#include <opencv2/dnn/layer.details.hpp>

#if defined(HAVE_PROTOBUF) && !defined(BUILD_PLUGIN)
#include <google/protobuf/stubs/common.h>
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

static Mutex* __initialization_mutex = NULL;
Mutex& getInitializationMutex()
{
    if (__initialization_mutex == NULL)
        __initialization_mutex = new Mutex();
    return *__initialization_mutex;
}
// force initialization (single-threaded environment)
Mutex* __initialization_mutex_initializer = &getInitializationMutex();

#if defined(HAVE_PROTOBUF) && !defined(BUILD_PLUGIN)
namespace {
using namespace google::protobuf;
class ProtobufShutdown {
public:
    bool initialized;
    ProtobufShutdown() : initialized(true) {}
    ~ProtobufShutdown()
    {
        initialized = false;
        google::protobuf::ShutdownProtobufLibrary();
    }
};
} // namespace
#endif

void initializeLayerFactory()
{
    CV_TRACE_FUNCTION();

#if defined(HAVE_PROTOBUF) && !defined(BUILD_PLUGIN)
    static ProtobufShutdown protobufShutdown; CV_UNUSED(protobufShutdown);
#endif

    CV_DNN_REGISTER_LAYER_CLASS(Concat,         ConcatLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Concat2,        Concat2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(ConstantOfShape, ConstantOfShapeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(CropAndResize,  CropAndResizeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(DequantizeLinear, DequantizeLinearLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Expand2,        Expand2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(Flatten,        FlattenLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Interp,         InterpLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Pad2,           Pad2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(QuantizeLinear, QuantizeLinearLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Range,          RangeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reshape,        ReshapeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reshape2,       Reshape2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(Resize,         ResizeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Shape,          ShapeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Slice,          SliceLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Slice2,         Slice2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(Split,          SplitLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Split2,         Split2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(Squeeze,        SqueezeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Tile2,          Tile2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(Transpose,      TransposeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Unsqueeze,      UnsqueezeLayer);

    CV_DNN_REGISTER_LAYER_CLASS(Convolution,    ConvolutionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Deconvolution,  DeconvolutionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Pooling,        PoolingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ROIPooling,     PoolingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PSROIPooling,   PoolingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reduce,         ReduceLayer);
    CV_DNN_REGISTER_LAYER_CLASS(LRN,            LRNLayer);
    CV_DNN_REGISTER_LAYER_CLASS(InnerProduct,   InnerProductLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Gemm,           GemmLayer);
    CV_DNN_REGISTER_LAYER_CLASS(MatMul,         MatMulLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Softmax,        SoftmaxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(SoftMax,        SoftmaxLayer);  // For compatibility. See https://github.com/opencv/opencv/issues/16877
    CV_DNN_REGISTER_LAYER_CLASS(MVN,            MVNLayer);

    CV_DNN_REGISTER_LAYER_CLASS(ReLU,           ReLULayer);
    CV_DNN_REGISTER_LAYER_CLASS(ReLU6,          ReLU6Layer);
    CV_DNN_REGISTER_LAYER_CLASS(ChannelsPReLU,  ChannelsPReLULayer);
    CV_DNN_REGISTER_LAYER_CLASS(PReLU,          ChannelsPReLULayer);
    CV_DNN_REGISTER_LAYER_CLASS(Sigmoid,        SigmoidLayer);
    CV_DNN_REGISTER_LAYER_CLASS(TanH,           TanHLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Swish,          SwishLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Mish,           MishLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ELU,            ELULayer);
    CV_DNN_REGISTER_LAYER_CLASS(BNLL,           BNLLLayer);
    CV_DNN_REGISTER_LAYER_CLASS(AbsVal,         AbsLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Power,          PowerLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Exp,            ExpLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Ceil,           CeilLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Floor,          FloorLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Log,            LogLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Round,          RoundLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Sqrt,           SqrtLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Not,            NotLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Acos,           AcosLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Acosh,          AcoshLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Asin,           AsinLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Asinh,          AsinhLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Atan,           AtanLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Atanh,          AtanhLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Cos,            CosLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Cosh,           CoshLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Erf,            ErfLayer);
    CV_DNN_REGISTER_LAYER_CLASS(HardSwish,      HardSwishLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Sin,            SinLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Sinh,           SinhLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Sign,           SignLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Shrink,         ShrinkLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Softplus,       SoftplusLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Softsign,       SoftsignLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Tan,            TanLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Celu,           CeluLayer);
    CV_DNN_REGISTER_LAYER_CLASS(HardSigmoid,    HardSigmoidLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Selu,           SeluLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ThresholdedRelu,ThresholdedReluLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Gelu,           GeluLayer);
    CV_DNN_REGISTER_LAYER_CLASS(GeluApproximation, GeluApproximationLayer);
    CV_DNN_REGISTER_LAYER_CLASS(BatchNorm,      BatchNormLayer);
    CV_DNN_REGISTER_LAYER_CLASS(MaxUnpool,      MaxUnpoolLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Dropout,        BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Identity,       BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Silence,        BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Const,          ConstLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Arg,            ArgLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reciprocal,     ReciprocalLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Gather,         GatherLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Gather2,        Gather2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(GatherElements, GatherElementsLayer);
    CV_DNN_REGISTER_LAYER_CLASS(LayerNormalization, LayerNormLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Expand,         ExpandLayer);
    CV_DNN_REGISTER_LAYER_CLASS(InstanceNormalization, InstanceNormLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Attention,      AttentionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(GroupNormalization, GroupNormLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Cast,           CastLayer);
    CV_DNN_REGISTER_LAYER_CLASS(DepthToSpace,   DepthToSpaceLayer)
    CV_DNN_REGISTER_LAYER_CLASS(SpaceToDepth,   SpaceToDepthLayer)
    CV_DNN_REGISTER_LAYER_CLASS(DepthToSpaceInt8, DepthToSpaceLayer)
    CV_DNN_REGISTER_LAYER_CLASS(SpaceToDepthInt8, SpaceToDepthLayer)

    CV_DNN_REGISTER_LAYER_CLASS(Crop,           CropLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Eltwise,        EltwiseLayer);
    CV_DNN_REGISTER_LAYER_CLASS(NaryEltwise,    NaryEltwiseLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Permute,        PermuteLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ShuffleChannel, ShuffleChannelLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PriorBox,       PriorBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PriorBoxClustered, PriorBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reorg,          ReorgLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Region,         RegionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(DetectionOutput, DetectionOutputLayer);
    CV_DNN_REGISTER_LAYER_CLASS(NormalizeBBox,  NormalizeBBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Normalize,      NormalizeBBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Shift,          ShiftLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Padding,        PaddingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Proposal,       ProposalLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Scale,          ScaleLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Compare,        CompareLayer);
    CV_DNN_REGISTER_LAYER_CLASS(DataAugmentation, DataAugmentationLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Correlation,    CorrelationLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Accum,          AccumLayer);
    CV_DNN_REGISTER_LAYER_CLASS(FlowWarp,       FlowWarpLayer);

    CV_DNN_REGISTER_LAYER_CLASS(LSTM,           LSTMLayer);
    CV_DNN_REGISTER_LAYER_CLASS(LSTM2,          LSTM2Layer);
    CV_DNN_REGISTER_LAYER_CLASS(GRU,            GRULayer);
    CV_DNN_REGISTER_LAYER_CLASS(CumSum,         CumSumLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Einsum,         EinsumLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Hardmax,        HardmaxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(GatherND,       GatherNDLayer);

    CV_DNN_REGISTER_LAYER_CLASS(Scatter,        ScatterLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ScatterND,      ScatterNDLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Tile,           TileLayer);
    CV_DNN_REGISTER_LAYER_CLASS(TopK,           TopKLayer);
    CV_DNN_REGISTER_LAYER_CLASS(TopK2,          TopK2Layer);

    CV_DNN_REGISTER_LAYER_CLASS(Quantize,         QuantizeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Dequantize,       DequantizeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Requantize,       RequantizeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ConvolutionInt8,  ConvolutionLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(InnerProductInt8, InnerProductLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(PoolingInt8,      PoolingLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(EltwiseInt8,      EltwiseLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(BatchNormInt8,    BatchNormLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(ScaleInt8,        ScaleLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(ShiftInt8,        ShiftLayerInt8);

    CV_DNN_REGISTER_LAYER_CLASS(ReLUInt8,         ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(ReLU6Int8,        ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(SigmoidInt8,      ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(TanHInt8,         ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(SwishInt8,        ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(HardSwishInt8,    ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(MishInt8,         ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(ELUInt8,          ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(BNLLInt8,         ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(AbsValInt8,       ActivationLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(SoftmaxInt8,      SoftmaxLayerInt8);
    CV_DNN_REGISTER_LAYER_CLASS(SoftMaxInt8,      SoftmaxLayerInt8);

    CV_DNN_REGISTER_LAYER_CLASS(ConcatInt8,       ConcatLayer);
    CV_DNN_REGISTER_LAYER_CLASS(FlattenInt8,      FlattenLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PaddingInt8,      PaddingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(BlankInt8,        BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(DropoutInt8,      BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(IdentityInt8,     BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(SilenceInt8,      BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ConstInt8,        ConstLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ReshapeInt8,      ReshapeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ResizeInt8,       ResizeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(SplitInt8,        SplitLayer);
    CV_DNN_REGISTER_LAYER_CLASS(SliceInt8,        SliceLayer);
    CV_DNN_REGISTER_LAYER_CLASS(CropInt8,         CropLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PermuteInt8,      PermuteLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ReorgInt8,        ReorgLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ShuffleChannelInt8, ShuffleChannelLayer);
}

CV__DNN_INLINE_NS_END
}} // namespace
