// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "../op_timvx.hpp"
#include "../ie_ngraph.hpp"
#include <iostream>
#include <numeric>

#include "cpu_kernels/convolution.hpp"

namespace cv
{
namespace dnn
{

#if CV_SIMD128
static inline void v_expand_mul_add(const v_int8x16& a, const v_int8x16& b,
                                    v_int32x4& out0, v_int32x4& out1, v_int32x4& out2, v_int32x4& out3)
{
    v_int16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);

    v_int32x4 t0, t1;
    v_mul_expand(a0, b0, t0, t1);
    out0 = v_add(out0, t0); out1 = v_add(out1, t1);

    v_mul_expand(a1, b1, t0, t1);
    out2 = v_add(out2, t0); out3 = v_add(out3, t1);
}
#endif

class BaseConvolutionLayerInt8Impl : public ConvolutionLayerInt8
{
public:
    BaseConvolutionLayerInt8Impl(const LayerParams &params)
    {
        setParamsFrom(params);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations, padMode, adjust_pads, useWinograd);

        numOutput = params.get<int>("num_output");
        int ngroups = params.get<int>("group", 1);
        CV_Assert(numOutput % ngroups == 0);

        input_sc = params.get<float>("input_scale");
        input_zp = params.get<int>("input_zeropoint");
        output_zp = params.get<int>("zeropoints");
        output_sc = params.get<float>("scales");
        per_channel = params.get<bool>("per_channel", true);

        if (kernel_size.size() == 2) {
            kernel = Size(kernel_size[1], kernel_size[0]);
            stride = Size(strides[1], strides[0]);
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in convolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
            dilation = Size(dilations[1], dilations[0]);

            adjustPad.height = adjust_pads[0];
            adjustPad.width = adjust_pads[1];
        }

        for (int i = 0; i < adjust_pads.size(); i++) {
            CV_Assert(adjust_pads[i] < strides[i]);
        }
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // blobs[0] - Weights (INT8)
        // blobs[1] - Biases (INT32)
        // blobs[2] - Multipliers for convolution output stage (FP32)
        CV_Assert(!inputs.empty() && blobs.size() == 3);
        MatSize weightShape = blobs[0].size;

        CV_Assert(inputs[0].dims == outputs[0].dims);
        if (weightShape.dims() == 3)
        {
            kernel_size.resize(1, kernel_size[0]);
            strides.resize(1, strides[0]);
            dilations.resize(1, dilations[0]);
            pads_begin.resize(1, pads_begin[0]);
            pads_end.resize(1, pads_end[0]);
        }
        CV_Assert(weightShape.dims() == kernel_size.size() + 2);
        for (int i = 0; i < kernel_size.size(); i++) {
            CV_Assert(weightShape[i + 2] == kernel_size[i]);
        }

        const Mat &input = inputs[0];
        CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || input.dims == 4 || input.dims == 5) && input.type() == CV_8S);
        for (size_t i = 0; i < outputs.size(); i++)
        {
            CV_Assert(inputs[i].type() == input.type());
            CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || inputs[i].dims == 4 || inputs[i].dims == 5) && inputs[i].size[1] == input.size[1]);
            for (int j = 0; j < inputs[i].dims; j++) {
                CV_Assert(inputs[i].size[j] == input.size[j]);
            }
        }

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(inpShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in convolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
        }
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
               (stride.height == 1 && stride.width == 1) &&
               (dilation.height == 1 && dilation.width == 1);
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Mat w, b;
        top->getScaleShift(w, b);
        if (w.empty() && b.empty())
            return false;

        CV_Assert((w.empty() || w.type() == CV_32F) &&
                  (b.empty() || b.type() == CV_32F));

        float new_sc;
        int new_zp;
        top->getScaleZeropoint(new_sc, new_zp);
        fuseWeights(w, b, new_sc);
        output_sc = new_sc;
        output_zp = new_zp;
        return true;
    }

    virtual void fuseWeights(const Mat& w_, const Mat& b_, const float& new_sc) = 0;
};

class ConvolutionLayerInt8Impl CV_FINAL : public BaseConvolutionLayerInt8Impl
{
public:
    enum { VEC_ALIGN = 32, DFT_TYPE = CV_8S };
    Mat weightsMat;
    std::vector<int> biasvec;
    std::vector<float> outputMultiplier;
    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;
    Ptr<FastQConv> fastQConvImpl;

    ConvolutionLayerInt8Impl(const LayerParams &params) : BaseConvolutionLayerInt8Impl(params){}

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        int dims = inpShape.size();
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int inpGroupCn = blobs[0].size[1];
        int ksize = inpGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(inpD * inpH * inpW, ksize);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        size_t ksize = kernel_size.size();

#ifdef HAVE_TIMVX
        if (backendId == DNN_BACKEND_TIMVX)
        {
            /* only Conv1d and Conv2d supported. */
            if (ksize == 2 || ksize == 1)
                return true;
            return false;
        }
#endif
        // Only default backend and Conv1D/Conv2D/Conv3D are supported
        return (backendId == DNN_BACKEND_OPENCV && ksize >= 1 && ksize <= 3) ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        const int* weightShape = blobs[0].size.p;
        CV_Assert(blobs[1].total() == (size_t)weightShape[0]);

        internals.clear();

        CV_Assert(inputs.size() != 0);
        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());

        int outCn = weightShape[0];
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);
        outShape.push_back(outCn);

        int inpCn = inputs[0][1];
        if (padMode.empty())
        {
            for (int i = 0; i < inpShape.size(); i++)
                outShape.push_back((inpShape[i] + pads_begin[i] + pads_end[i] - dilations[i] * (kernel_size[i] - 1) - 1) / strides[i] + 1);
        }
        else
        {
            getConvPoolOutParams(inpShape, kernel_size, strides, padMode, dilations, outShape);
        }

        int ngroups = inpCn / weightShape[1];
        if (ngroups == 0 || ngroups * weightShape[1] != inpCn)
            CV_Error(Error::StsError, format("Number of input channels should "
                     "be multiple of %d but got %d", weightShape[1], inpCn));
        CV_Assert(ngroups > 0 && inpCn % ngroups == 0 && outCn % ngroups == 0);

        outputs.resize(1, outShape);

        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerInt8Impl::finalize(inputs_arr, outputs_arr);

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        // prepare weightsMat where each row is aligned and has enough zero padding on the right to
        // use vectorized (i.e. with intrinsics) loops without tail processing
        Mat wm = blobs[0].reshape(1, numOutput);
        if( wm.step1() % VEC_ALIGN != 0 )
        {
            int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
            Mat wm_buffer = Mat(numOutput, newcols, wm.type());
            Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
            wm_padding.setTo(Scalar::all(0));
            Mat wm_aligned = wm_buffer.colRange(0, wm.cols);
            wm.copyTo(wm_aligned);
            wm = wm_aligned;
        }
        weightsMat = wm;

        Mat biasMat = blobs[1];
        biasvec.resize(numOutput+2);

        Mat outMult = blobs[2];
        outputMultiplier.resize(numOutput+2);
        for(int i = 0; i < numOutput; i++ )
        {
            biasvec[i] = biasMat.at<int>(i);
            outputMultiplier[i] = outMult.at<float>(i);
        }
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        // TODO! add activation in convolution.
#ifdef HAVE_TIMVX
        if (preferableTarget == DNN_TARGET_NPU)
            return false;
#endif
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty())
        {
            activ = activ_int8;
            if (!activ_int8->blobs.empty())
                activ_int8->blobs[0].convertTo(activationLUT, CV_32S);
            return true;
        }
        return false;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        return BaseConvolutionLayerInt8Impl::tryFuse(top);
    }

    void fuseWeights(const Mat& w_, const Mat& b_, const float& new_sc) CV_OVERRIDE
    {
        const int outCn = weightsMat.size[0];
        Mat w = w_.total() == 1 ? Mat(1, outCn, CV_32F, Scalar(w_.at<float>(0))) : w_;
        Mat b = b_.total() == 1 ? Mat(1, outCn, CV_32F, Scalar(b_.at<float>(0))) : b_;
        CV_Assert_N(!weightsMat.empty(), biasvec.size() == outCn + 2,
                    w.empty() || outCn == w.total(), b.empty() || outCn == b.total());

        for (int i = 0; i < outCn; ++i)
        {
            float off = outputMultiplier[i] * output_sc;
            if (!w.empty())
                off *= w.at<float>(i);

            if (!b.empty())
                biasvec[i] += (int)std::round(b.at<float>(i)/off);

            outputMultiplier[i] = off/new_sc;
        }
        biasvec[outCn] = biasvec[outCn+1] = biasvec[outCn-1];
        outputMultiplier[outCn] = outputMultiplier[outCn+1] = outputMultiplier[outCn-1];
    }

    virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        /* TODO :support GroupConv;
        Ref:
        https://github.com/VeriSilicon/TIM-VX/blob/main/docs/Operators.md#conv2d
        Link Reference: https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/ops/conv1d_test.cc
        */

        // tvGraph Initialization.
        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        Mat tvWeightMat = blobs[0];

        std::vector<int> tvBiasVec;
        tvBiasVec.assign(biasvec.begin(), biasvec.end() - 2);
        Mat tvBiasMat(tvBiasVec);

        for (int i = 0; i < numOutput; i++)
        {
            tvBiasVec[i] += input_zp * (cv::sum(blobs[0].row(i))[0]);
        }

        // Padding Type
        tim::vx::PadType tvPadType;

        if (padMode.empty())
        {
            tvPadType = tim::vx::PadType::AUTO; // TODO! check the padding type.
        }
        else if(padMode == "VALID")
        {
            tvPadType = tim::vx::PadType::VALID;
        }
        else if (padMode == "SAME")
        {
            tvPadType = tim::vx::PadType::SAME;
        }
        else
        {
            CV_Error(Error::StsError, "Unsupported padding mode in TimVXBackend!");
        }

        size_t ksize = kernel_size.size();

        std::vector<int> inputsIndex;
        std::vector<int> outputsIndex;

        CV_Assert(inputsWrapper.size() == 1);
        CV_Assert(ksize == 2 || ksize == 1);

        std::vector<float> weight_scs, bias_scs;
        std::vector<int32_t> weight_zps, bias_zps;

        weight_scs.resize(numOutput);
        bias_scs.resize(numOutput);

        for (int i = 0; i < numOutput; i++)
        {
            bias_scs[i] = outputMultiplier[i] * output_sc;
            weight_scs[i] = bias_scs[i] / input_sc;
        }

        weight_zps.assign(numOutput, 0);
        bias_zps.assign(numOutput, 0);

        bool tvSymmetric;
        tvSymmetric = getQuantType(weight_scs, numOutput);

        // input Tensor
        auto inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        int input_index = -1, weight_index = -1, bias_index = -1, output_index = -1;

        if (inputWrapper->isTensor())
        {
            input_index = tvGraph->getTensorIndex(inputWrapper->getTensor());
            if (input_index == -1)
            {
                // Copy To New inputWrapper
                Mat tmp = inputWrapper->getMat();
                inputWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tmp));
            }
        }

        if (!inputWrapper->isTensor())
        {
            Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, input_sc, input_zp));
            inputWrapper->createTensor(graph, tim::vx::TensorAttribute::INPUT, tvInputQuant);
            input_index = tvGraph->addWrapper(inputWrapper);
        }
        inputsIndex.push_back(input_index);

        // weight Tensor
        auto tvConvWeightShape = shape(tvWeightMat);
        Mat tvInputMat = inputWrapper->getMat();
        // calculate group value.
        int group = tvInputMat.size[1] / tvWeightMat.size[1];

        // TODO! It will be supported in future.
        if (tvSymmetric && tvWeightMat.total() == tvConvWeightShape[0])
            return Ptr<TimVXBackendNode>();
        // Reverse weight shape From OpenCV NCHW to TimVX WHCN.
        std::reverse(tvConvWeightShape.begin(), tvConvWeightShape.end());

        Ptr<TimVXBackendWrapper> weightWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tvWeightMat));
        Ptr<tim::vx::Quantization> weightQuant;

        if (tvSymmetric)
        {
            int wtChanneldim = tvWeightMat.dims - 1;
            weightQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, wtChanneldim,
                                              weight_scs, weight_zps));
        }
        else
        {
            weightQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, weight_scs[0], 0));
        }
        weightWrapper->createTensor(graph,tim::vx::TensorAttribute::CONSTANT, weightQuant);

        weight_index = tvGraph->addWrapper(weightWrapper);
        inputsIndex.push_back(weight_index);

        // Bias Tensor
        Ptr<TimVXBackendWrapper> biasWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tvBiasMat));
        Ptr<tim::vx::Quantization> biasQuant;

        if (tvSymmetric)
        {
            biasQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0,
                                              bias_scs, bias_zps));
        }
        else
        {
            biasQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, weight_scs[0] * input_sc, 0));
        }

        biasWrapper->createTensor(graph, tim::vx::TensorAttribute::CONSTANT, biasQuant);
        bias_index = tvGraph->addWrapper(biasWrapper);
        inputsIndex.push_back(bias_index);
        // Output tensor
        CV_Assert(outputsWrapper.size() == 1);
        auto outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, output_sc, output_zp));

        if (isLast)
        {
            // From OpenCV NCHW, to TimVX WHCN
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }

        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvConv;

        if (ksize == 2)  // for conv2d
        {
            int multiplier = 0;
            if(group == tvConvWeightShape[3] && group != 1)
                multiplier = 1;
            if (group == 1 || (group == tvConvWeightShape[3] && group != 1)) // Conv2D || DeConv2D
            {
                if (tvPadType == tim::vx::PadType::AUTO) {
                    tvConv = graph->CreateOperation<tim::vx::ops::Conv2d>(
                            tvConvWeightShape[3], tvPadType,
                            std::array<uint32_t, 2>({(uint32_t) kernel_size[1], (uint32_t) kernel_size[0]}),
                            std::array<uint32_t, 2>({(uint32_t) strides[1], (uint32_t) strides[0]}),
                            std::array<uint32_t, 2>({(uint32_t) dilations[1], (uint32_t) dilations[0]}),
                            std::array<uint32_t, 4>({(uint32_t) pads_begin[1], (uint32_t) pads_end[1],
                                                     (uint32_t) pads_begin[0], (uint32_t) pads_end[0]}),
                            multiplier);
                }
                else
                {
                    tvConv = graph->CreateOperation<tim::vx::ops::Conv2d>(
                            tvPadType,
                            std::array<uint32_t, 2>({(uint32_t) strides[1], (uint32_t) strides[0]}),
                            std::array<uint32_t, 2>({(uint32_t) dilations[1], (uint32_t) dilations[0]}),
                            multiplier);
                }
            }
            else
            {
                // GroupedConv2d
                if (tvPadType == tim::vx::PadType::AUTO)
                {
                    tvConv = graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
                            std::array<uint32_t, 4>({(uint32_t) pads_begin[1], (uint32_t) pads_end[1],
                                                     (uint32_t) pads_begin[0], (uint32_t) pads_end[0]}),
                            std::array<uint32_t, 2>({(uint32_t)strides[1], (uint32_t)strides[0]}),
                            std::array<uint32_t, 2>({(uint32_t)dilations[1], (uint32_t)dilations[0]}),
                            group);
                }
                else
                {
                    tvConv = graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
                            tvPadType,
                            std::array<uint32_t, 2>({(uint32_t)strides[1], (uint32_t)strides[0]}),
                            std::array<uint32_t, 2>({(uint32_t)dilations[1], (uint32_t)dilations[0]}),
                            group);
                }
            }
        }
        else
        {
            // for Conv1d
            if (group != 1)
                CV_Error( cv::Error::StsNotImplemented, " Grouped Conv1d or Depth-Wise Conv1d are not supported by "
                                                "TimVX Backend. Please try OpenCV Backend.");
            tvConv = graph->CreateOperation<tim::vx::ops::Conv1d>(
                    tvConvWeightShape[2], tvPadType, (uint32_t)kernel_size[0],
                    (uint32_t)strides[0],(uint32_t)dilations[0],
                    std::array<uint32_t, 2>({(uint32_t)pads_begin[0], (uint32_t)pads_end[0]}));
        }
        // Create TimVXBackendNode
        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvConv, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        CV_Assert_N(inputs.size() >= 1, nodes.size() >= 1);
        CV_CheckTypeEQ(weightsMat.type(), CV_8S, "");
        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<size_t> dims = ieInpNode.get_shape();
        CV_Check(dims.size(), dims.size() >= 3 && dims.size() <= 5, "");
        CV_Assert(ieInpNode.get_element_type() == ov::element::f32);
        ov::Output<ov::Node> ieWeights;
        if (nodes.size() > 1)
            ieWeights = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        const int inpCn = dims[1];
        const int inpGroupCn = nodes.size() > 1 ? ieWeights.get_shape()[1] : blobs[0].size[1];
        const int group = inpCn / inpGroupCn;

        std::vector<size_t> kernel_shape;
        if (group != 1)
        {
            kernel_shape.push_back(group);
        }
        kernel_shape.push_back(numOutput / group);
        kernel_shape.push_back(inpCn / group);
        std::copy(kernel_size.begin(), kernel_size.end(), back_inserter(kernel_shape));

        if (nodes.size() == 1)
        {
            ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::i8, kernel_shape, blobs[0].data);
        }
        else
        {
            auto shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                             ov::Shape{kernel_shape.size()}, std::vector<int64_t>(kernel_shape.begin(), kernel_shape.end()));
            ieWeights  = std::make_shared<ov::op::v1::Reshape>(ieWeights, shape, true);
        }

        ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
        if (!padMode.empty())
            pad_type = padMode == "VALID" ? ov::op::PadType::VALID : ov::op::PadType::SAME_UPPER;

        ieInpNode = ngraphDequantize(ieInpNode, input_sc, input_zp);

        const float low = -128, high = 127;
        std::vector<float> inpLows(numOutput, low);
        std::vector<float> inpHighs(numOutput, high);
        std::vector<float> outLows(numOutput);
        std::vector<float> outHighs(numOutput);
        std::vector<size_t> quantShape(kernel_shape.size(), 1);
        if (group != 1)
        {
            quantShape[0] = group;
            quantShape[1] = numOutput / group;
        }
        else
        {
            quantShape[0] = numOutput;
        }

        for (int i = 0; i < numOutput; ++i) {
            outLows[i] = low * outputMultiplier[i] * output_sc / input_sc;
            outHighs[i] = high * outputMultiplier[i] * output_sc / input_sc;
        }
        ieWeights = std::make_shared<ov::op::v0::Convert>(ieWeights, ov::element::f32);
        ieWeights = std::make_shared<ov::op::v0::FakeQuantize>(ieWeights,
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, quantShape, inpLows.data()),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, quantShape, inpHighs.data()),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, quantShape, outLows.data()),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, quantShape, outHighs.data()),
            256 // levels
        );

        ov::Output<ov::Node> conv_node;
        if (group != 1) {
            conv_node = std::make_shared<ov::op::v1::GroupConvolution>(
                                ieInpNode, ieWeights,
                                ov::Strides(strides),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_end.begin(),   pads_end.end())),
                                ov::Strides(dilations),
                                pad_type);
        } else {
            conv_node = std::make_shared<ov::op::v1::Convolution>(
                                ieInpNode, ieWeights,
                                ov::Strides(strides),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                                ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_end.begin(), pads_end.end())),
                                ov::Strides(dilations),
                                pad_type);
        }

        std::vector<size_t> shape(conv_node.get_shape().size(), 1);
        shape[1] = conv_node.get_shape()[1];
        if (biasvec.size() || nodes.size() == 3)
        {
            std::shared_ptr<ov::Node> bias;
            if (nodes.size() == 3)
            {
                auto bias_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                    ov::Shape{shape.size()}, std::vector<int64_t>(shape.begin(), shape.end()));
                bias = std::make_shared<ov::op::v1::Reshape>(nodes[2].dynamicCast<InfEngineNgraphNode>()->node, bias_shape, true);
            }
            else
            {
                std::vector<float> ovBias(numOutput);
                for (int i = 0; i < numOutput; ++i) {
                    ovBias[i] = (biasvec[i] + input_zp * cv::sum(blobs[0].row(i))[0]) * outputMultiplier[i] * output_sc;
                }
                bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape(shape), ovBias.data());
            }
            conv_node = std::make_shared<ov::op::v1::Add>(conv_node, bias, ov::op::AutoBroadcastType::NUMPY);
        }

        conv_node = ngraphQuantize(conv_node, output_sc, output_zp);

        return new InfEngineNgraphNode(conv_node);
    }
#endif  // HAVE_DNN_NGRAPH

    class ParallelConv : public cv::ParallelLoopBody
    {
    public:
        enum { BLK_SIZE = 32, BLK_SIZE_CN = 64 };

        const Mat* input_;
        const Mat* weights_;
        Mat* output_;
        int outShape[4]; // used only for conv2d
        std::vector<size_t> kernel_size, pads_begin, pads_end, strides, dilations;
        int ngroups_, nstripes_;
        std::vector<int> ofstab_;
        const std::vector<int>* biasvec_;
        const Mat* activLUT_;
        const ActivationLayerInt8* activ_;
        bool is1x1_;
        bool useAVX2;
        bool useAVX512;
        bool useLASX;
        bool useRVV;
        int blk_size_cn;
        int inpZp, outZp;
        const std::vector<float>* multiplier;

        ParallelConv()
            : input_(0), weights_(0), output_(0), ngroups_(0), nstripes_(0),
              biasvec_(0), activLUT_(0), activ_(0), is1x1_(false), useAVX2(false), useAVX512(false), useLASX(false), useRVV(false)
            , blk_size_cn(0), inpZp(0), outZp(0), multiplier(0)
        {}

        static void run( const Mat& input, Mat& output, const Mat& weights, const std::vector<float>& multipliers,
                         const std::vector<int>& biasvec, const Mat& activLUT,
                         const std::vector<size_t>& kernel_size, const std::vector<size_t>& strides,
                         const std::vector<size_t>& pads_begin, const std::vector<size_t>& pads_end,
                         const std::vector<size_t>& dilations,
                         const ActivationLayerInt8* activ, int ngroups, int nstripes, int inp_Zp, int out_Zp)
        {
            size_t karea = std::accumulate(kernel_size.begin(), kernel_size.end(),
                                           1, std::multiplies<size_t>());
            bool isConv1D = input.dims == 3;
            bool isConv2D = input.dims == 4;
            bool isConv3D = input.dims == 5;
            CV_CheckEQ(static_cast<int>(kernel_size.size()), input.dims - 2, "");
            CV_Assert_N(input.dims == output.dims,
                       input.size[0] == output.size[0],
                       weights.rows == output.size[1],
                       weights.cols == (input.size[1]/ngroups)*karea,
                       input.type() == CV_8SC1,
                       output.type() == CV_32SC1,
                       input.type() == weights.type(),
                       input.isContinuous(),
                       output.isContinuous(),
                       biasvec.size() == (size_t)output.size[1]+2);
            CV_Check(weights.step1(), weights.step1() % VEC_ALIGN == 0, "");
            ParallelConv p;

            p.input_ = &input;
            p.weights_ = &weights;
            p.output_ = &output;
            int max_ind = isConv1D? 3: 4;
            for( int i = 0; i < max_ind; i++ ) p.outShape[i] = output.size[i];
            p.outShape[1] /= ngroups;

            p.kernel_size = kernel_size; p.strides = strides; p.dilations = dilations;
            p.pads_begin = pads_begin; p.pads_end = pads_end;

            p.ngroups_ = ngroups;
            p.nstripes_ = nstripes;

            int inpCnAll = input.size[1];
            int depth = (input.dims == 5) ? input.size[2] : 1;
            int width = input.size[input.dims - 1];
            int height = isConv1D? 1 : input.size[input.dims - 2];
            int inpCn = inpCnAll / ngroups;

            p.is1x1_ = (isConv2D && kernel_size[0] == 1 && kernel_size[1] == 1 &&
                       pads_begin[0] == 0  && pads_begin[1] == 0) ||
                       (isConv1D && pads_begin[0] == 0 && kernel_size[0] == 1);

            p.useAVX2   = checkHardwareSupport(CPU_AVX2) && isConv2D;
            p.useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX  && isConv2D;

            p.useLASX   = checkHardwareSupport(CPU_LASX) && isConv2D;
            p.useRVV   = checkHardwareSupport(CPU_RVV) && isConv2D;

            int kernel_d = isConv3D? kernel_size[0] : 1;
            int kernel_h = isConv1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();

            int blk_size_cn0 = cvCeil(1600./(kernel_w*kernel_h));
            int ncn = 32;
            while (ncn*2 < blk_size_cn0 && ncn < inpCn)
                ncn *= 2;
            ncn = std::min(ncn, inpCn);
            p.blk_size_cn = ncn;

            int dil_d = isConv3D? dilations[0] : 1;
            int dil_h = isConv1D? 1 : dilations[dilations.size() - 2];
            int dil_w = dilations.back();

            p.inpZp = inp_Zp;
            p.outZp = out_Zp;
            p.multiplier = &multipliers;

            p.ofstab_.resize(karea * ncn);
            int* ofstab = &p.ofstab_[0];

            if (isConv1D)
            {
                for( int k = 0; k < ncn; k++ )
                    for( int k_c = 0; k_c < kernel_w; k_c++ )
                        ofstab[k*kernel_w + k_c] = k*width + k_c*dil_w;
            }
            else if (isConv2D)
            {
                for( int k = 0; k < ncn; k++ )
                    for( int k_r = 0; k_r < kernel_h; k_r++ )
                        for( int k_c = 0; k_c < kernel_w; k_c++ )
                            ofstab[(k*kernel_h + k_r)*kernel_w + k_c] =
                                   (k*height + k_r*dil_h)*width + k_c*dil_w;
            }
            else
            {
                for( int k = 0; k < ncn; k++ )
                    for (int k_d = 0; k_d < kernel_d; k_d++)
                        for( int k_r = 0; k_r < kernel_h; k_r++ )
                            for( int k_c = 0; k_c < kernel_w; k_c++ )
                                ofstab[(k*kernel_d*kernel_h + k_d*kernel_h + k_r)*kernel_w + k_c] =
                                       (k*depth*height + k_d*dil_d*height + k_r*dil_h)*width + k_c*dil_w;
            }

            p.biasvec_ = &biasvec;
            p.activLUT_ = &activLUT;
            p.activ_ = !activLUT.empty() ? activ : 0;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        virtual void operator ()(const Range &r0) const CV_OVERRIDE
        {
            const int valign = ConvolutionLayerInt8Impl::VEC_ALIGN;
            int ngroups = ngroups_, batchSize = input_->size[0]*ngroups;
            bool isConv1D = input_->dims == 3;
            bool isConv2D = input_->dims == 4;
            bool isConv3D = input_->dims == 5;

            int outW = output_->size[output_->dims - 1];
            int outH = isConv1D? 1 : output_->size[output_->dims - 2];
            int outCn = output_->size[1]/ngroups;

            int depth = isConv3D? input_->size[2] : 1;
            int height = isConv1D? 1 : input_->size[input_->dims - 2];
            int width = input_->size[input_->dims - 1];
            int inpCn = input_->size[1]/ngroups;

            const int nstripes = nstripes_;

            int kernel_d = isConv3D? kernel_size[0] : 1;
            int kernel_h = isConv1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();
            int karea = kernel_w*kernel_h*kernel_d;

            int pad_d = isConv3D? pads_begin[0] : 0;
            int pad_t = isConv1D? 0 : pads_begin[pads_begin.size() - 2];
            int pad_l = pads_begin.back();

            int stride_d = isConv3D? strides[0] : 0;
            int stride_h = isConv1D? 0 : strides[strides.size() - 2];
            int stride_w = strides.back();

            int dilation_d = isConv3D? dilations[0] : 1;
            int dilation_h = isConv1D? 1 : dilations[dilations.size() - 2];
            int dilation_w = dilations.back();

            int i, j, k, d;
            int inpPlaneSize = (int)input_->total(2);
            int outPlaneSize = (int)output_->total(2);
            bool is1x1 = is1x1_;

            int stripesPerSample;
            int stripeSize;
            Range r = r0;
            bool depthWiseConvolution = !is1x1 && isConv2D && ngroups > 1 && inpCn == 1 &&
                outCn == 1 && kernel_d == 1 && dilation_d == 1 && stride_d == 0 && pad_d == 0 &&
                width >= 16 + dilation_w*(kernel_w - 1);
            // for now only 3x3 depth-wise convolutions are supported
            depthWiseConvolution = depthWiseConvolution && kernel_w == 3 && kernel_h == 3 &&
                // computing at most 1 pixel from each side can involve padding
                max(stride_w, dilation_w) >= pad_l && max(stride_h, dilation_h) >= pad_t &&
                pad_l <= 1 && pad_t <= 1;

            if( !depthWiseConvolution && nstripes >= batchSize*2 )
            {
                stripesPerSample = nstripes/batchSize;
                stripeSize = (int)alignSize((outPlaneSize + stripesPerSample - 1)/stripesPerSample, 8);
                stripeSize = std::min(stripeSize, outPlaneSize);
            }
            else
            {
                stripesPerSample = 1;
                int samplesPerStripe = std::max((batchSize + nstripes - 1)/nstripes, 1);
                r.start *= samplesPerStripe;
                r.end *= samplesPerStripe;
                stripeSize = outPlaneSize;
            }

            const int8_t* data_inp0_ = input_->ptr<int8_t>();
            const int* ofstab = &ofstab_[0];
            const int8_t* wptr_orig_ = weights_->ptr<int8_t>();
            size_t wstep = weights_->step1();
            const int* biasptr_ = &biasvec_->at(0);
            const float* multptr_ = &multiplier->at(0);
            const int* lutptr_ = !activLUT_->empty() ? activLUT_->ptr<int>() : 0;
            int* data_out0_ = output_->ptr<int>();
            AutoBuffer<int8_t> rowbuf0_;
            int8_t* rowbuf0 = 0;
            bool use_rowbuf = !depthWiseConvolution;
            int blk_size = depthWiseConvolution ? outPlaneSize : min((int)BLK_SIZE, stripeSize);

            // im2row buffer is not used for depth-wise convolution
            if(use_rowbuf)
            {
                size_t rowbufsz = alignSize(karea*blk_size_cn, valign)*min((int)BLK_SIZE, blk_size);
                //printf("karea=%d, blk_size_cn=%d, rowbufsz=%d, stripeSize=%d\n", karea, blk_size_cn, (int)rowbufsz, stripeSize);
                rowbuf0_.allocate(rowbufsz + valign);
                rowbuf0 = alignPtr(rowbuf0_.data(), (int)(valign*sizeof(int8_t)));
                // we clear the buffer once; ultimately, it lets us to avoid
                // tail processing after running the unrolled/vectorized loop.
                // the main idea is to make sure that the tail (a.k.a. padding) of each row
                // (i.e. the elements with indices between vsz=karea*ncn and vsz_a)
                // does not contain NaNs or Infs. Because the padding in the weights
                // matrix is explicitly initialized with 0's, we handle all other
                // cases nicely, i.e. we can skip expliciting re-initialization
                // of the padding - we just retain elements from the previous iteration
                // of the loop over channels (cn0).
                memset(rowbuf0, (int8_t)inpZp, rowbufsz*sizeof(rowbuf0[0]) );
            }

            for( int stripe = r.start; stripe < r.end; stripe++ )
            {
                int subsampleIdx = stripe/stripesPerSample;
                if( subsampleIdx >= batchSize )
                    break;
                int stripeStart = (int)((stripe - subsampleIdx*stripesPerSample)*stripeSize);
                int stripeEnd = (int)std::min(stripeStart + stripeSize, outPlaneSize);
                const int8_t* data_inp0 = data_inp0_ + subsampleIdx*inpPlaneSize*inpCn;
                int* data_out0 = data_out0_ + subsampleIdx*outPlaneSize*outCn;
                int startOutCn = (subsampleIdx % ngroups)*outCn;
                const int8_t* wptr_orig = wptr_orig_ + wstep*startOutCn;
                const int* biasptr = biasptr_ + startOutCn;
                const float* multptr = multptr_ + startOutCn;

                for( int cn0 = 0; cn0 < inpCn; cn0 += blk_size_cn )
                {
                    int cn1 = std::min(cn0 + blk_size_cn, inpCn);
                    int ncn = cn1 - cn0, vsz = karea*ncn;
                    int vsz_a = (int)alignSize(vsz, valign);
                    const int8_t* wptr = wptr_orig + cn0*karea;

                    for( int ofs0 = stripeStart; ofs0 < stripeEnd; ofs0 += blk_size )
                    {
                        int ofs, ofs1 = std::min(ofs0 + blk_size, stripeEnd);
                        int bsz = ofs1 - ofs0;

                        int out_d = ofs0 / (outH * outW);
                        int out_i = (ofs0 - out_d * outH * outW) / outW;
                        int out_j = ofs0 % outW;

                        if (depthWiseConvolution)
                        {
                            CV_Assert(out_i == 0 && out_j == 0);
                            int in_d = out_d * stride_d - pad_d;
                            const int8_t* inptr_ = data_inp0 + (cn0*depth*height + in_d*height)*width;
                            int* outptr_ = data_out0 + ofs0;

                        #if CV_TRY_AVX2
                            if(useAVX2)
                                opt_AVX2::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, multptr, inptr_, height, width, outptr_, out_d, outH, outW, inpZp, outZp);
                            else
                        #endif
                        #if CV_TRY_LASX
                            if(useLASX)
                                opt_LASX::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, multptr, inptr_, height, width, outptr_, out_d, outH, outW, inpZp, outZp);
                            else
                        #endif
                        #if CV_TRY_RVV && defined(__riscv_v_intrinsic) && __riscv_v_intrinsic>=11000
                            if(useRVV)
                                opt_RVV::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, multptr, inptr_, height, width, outptr_, out_d, outH, outW, inpZp, outZp);
                            else
                        #endif
                        #if CV_RVP052
                            if(isConv2D)
                                opt_RVP052::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, multptr, inptr_, height, width, outptr_, out_d, outH, outW, inpZp, outZp);
                            else
                        #endif
                            {
                                const int8_t w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                                             w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                                             w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
                                int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
                                int bias = biasptr[out_d], biasCopy;
                                float mult = multptr[out_d];

                                for (int out_i = 0; out_i < outH; out_i++)
                                {
                                    int in_i = out_i * stride_h - pad_t, out_j = 0;
                                    const int8_t* imgptr0 = inptr_ + in_i*width;
                                    const int8_t* imgptr1 = imgptr0 + dilation_h*width;
                                    const int8_t* imgptr2 = imgptr0 + (dilation_h*2)*width;
                                    int8_t w00 = w00_, w01 = w01_, w02 = w02_;
                                    int8_t w20 = w20_, w21 = w21_, w22 = w22_;
                                    int out, out1;
                                    // Bias has a fused offset component. bias = bias_quantized - input_zeropoint*sum_of_weights.
                                    // In some cases below, certain weights are not used for convolution or set to zero.
                                    // So we create a copy of bias at the start and remove the weight's components as necessary.
                                    biasCopy = bias;

                                    if (in_i < 0)
                                    {
                                        biasCopy += inpZp * (w00 + w01 + w02);
                                        w00 = w01 = w02 = 0;
                                        imgptr0 = imgptr1;
                                    }
                                    else if (in_i + dilation_h*(kernel_h-1) >= height)
                                    {
                                        biasCopy += inpZp * (w20 + w21 + w22);
                                        w20 = w21 = w22 = 0;
                                        imgptr2 = imgptr1;
                                    }
                                    int* outptr = outptr_ + out_i*outW;
                                    if (pad_l > 0)
                                    {
                                        out = (int)imgptr0[0]*w01 + (int)imgptr0[dilation_w]*w02 +
                                              (int)imgptr1[0]*w11 + (int)imgptr1[dilation_w]*w12 +
                                              (int)imgptr2[0]*w21 + (int)imgptr2[dilation_w]*w22 +
                                              biasCopy + inpZp*(w00 + w10 + w20);
                                        out1 = outZp + (int)std::round(out*mult);
                                        outptr[0] = std::min(std::max(out1, -128), 127);
                                        out_j = 1;
                                    }
                                #if CV_SIMD128
                                    if( stride_w == 1 )
                                    {
                                        const int out_delta = 16;
                                        v_int8x16 vw00 = v_setall_s8(w00), vw01 = v_setall_s8(w01), vw02 = v_setall_s8(w02),
                                                  vw10 = v_setall_s8(w10), vw11 = v_setall_s8(w11), vw12 = v_setall_s8(w12),
                                                  vw20 = v_setall_s8(w20), vw21 = v_setall_s8(w21), vw22 = v_setall_s8(w22);
                                        v_int32x4 vout0, vout1, vout2, vout3, vbias = v_setall_s32(biasCopy), voutzp = v_setall_s32(outZp),
                                                  outmin = v_setall_s32(-128), outmax = v_setall_s32(127);
                                        v_float32x4 vmult = v_setall_f32(mult);
                                        for( ; out_j < outW1; out_j += out_delta )
                                        {
                                            if (out_j + out_delta > outW1)
                                            {
                                                if (out_j <= pad_l)
                                                    break;
                                                out_j = outW1 - out_delta;
                                            }
                                            int in_j = out_j * stride_w - pad_l;
                                            v_int8x16 v00 = v_load(imgptr0 + in_j),
                                                      v01 = v_load(imgptr0 + in_j + dilation_w),
                                                      v02 = v_load(imgptr0 + in_j + dilation_w*2),
                                                      v10 = v_load(imgptr1 + in_j),
                                                      v11 = v_load(imgptr1 + in_j + dilation_w),
                                                      v12 = v_load(imgptr1 + in_j + dilation_w*2),
                                                      v20 = v_load(imgptr2 + in_j),
                                                      v21 = v_load(imgptr2 + in_j + dilation_w),
                                                      v22 = v_load(imgptr2 + in_j + dilation_w*2);

                                            vout0 = vout1 = vout2 = vout3 = vbias;
                                            v_expand_mul_add(v00, vw00, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v01, vw01, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v02, vw02, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v10, vw10, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v11, vw11, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v12, vw12, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v20, vw20, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v21, vw21, vout0, vout1, vout2, vout3);
                                            v_expand_mul_add(v22, vw22, vout0, vout1, vout2, vout3);

                                            vout0 = v_add(voutzp, v_round(v_mul(v_cvt_f32(vout0), vmult)));
                                            vout1 = v_add(voutzp, v_round(v_mul(v_cvt_f32(vout1), vmult)));
                                            vout2 = v_add(voutzp, v_round(v_mul(v_cvt_f32(vout2), vmult)));
                                            vout3 = v_add(voutzp, v_round(v_mul(v_cvt_f32(vout3), vmult)));

                                            vout0 = v_min(v_max(vout0, outmin), outmax);
                                            vout1 = v_min(v_max(vout1, outmin), outmax);
                                            vout2 = v_min(v_max(vout2, outmin), outmax);
                                            vout3 = v_min(v_max(vout3, outmin), outmax);

                                            v_store(outptr + out_j, vout0);
                                            v_store(outptr + out_j + 4, vout1);
                                            v_store(outptr + out_j + 8, vout2);
                                            v_store(outptr + out_j + 12, vout3);
                                        }
                                    }
                                #endif
                                    for (; out_j < outW1; out_j++)
                                    {
                                        int in_j = out_j * stride_w - pad_l;
                                        out = (int)imgptr0[in_j]*w00 + (int)imgptr0[in_j + dilation_w]*w01 + (int)imgptr0[in_j + dilation_w*2]*w02 +
                                              (int)imgptr1[in_j]*w10 + (int)imgptr1[in_j + dilation_w]*w11 + (int)imgptr1[in_j + dilation_w*2]*w12 +
                                              (int)imgptr2[in_j]*w20 + (int)imgptr2[in_j + dilation_w]*w21 + (int)imgptr2[in_j + dilation_w*2]*w22 + biasCopy;
                                        out1 = outZp + (int)std::round(out*mult);
                                        outptr[out_j] = std::min(std::max(out1, -128), 127);
                                    }

                                    for (; out_j < outW; out_j++ )
                                    {
                                        int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
                                        int s0 = 1, s1 = 1, s2 = 1;
                                        if (in_j0 >= width)
                                        {
                                            in_j0 = 0;
                                            s0 = 0;
                                            biasCopy += inpZp*(w00 + w10 + w20);
                                        }
                                        if (in_j1 >= width)
                                        {
                                            in_j1 = 0;
                                            s1 = 0;
                                            biasCopy += inpZp*(w01 + w11 + w21);
                                        }
                                        if (in_j2 >= width)
                                        {
                                            in_j2 = 0;
                                            s2 = 0;
                                            biasCopy += inpZp*(w02 + w12 + w22);
                                        }
                                        out = (int)imgptr0[in_j0]*w00*s0 + (int)imgptr0[in_j1]*w01*s1 + (int)imgptr0[in_j2]*w02*s2 +
                                              (int)imgptr1[in_j0]*w10*s0 + (int)imgptr1[in_j1]*w11*s1 + (int)imgptr1[in_j2]*w12*s2 +
                                              (int)imgptr2[in_j0]*w20*s0 + (int)imgptr2[in_j1]*w21*s1 + (int)imgptr2[in_j2]*w22*s2 + biasCopy;
                                        out1 = outZp + (int)std::round(out*mult);
                                        outptr[out_j] = std::min(std::max(out1, -128), 127);
                                    }
                                }
                            }
                            continue;
                        }
                        // do im2row for a part of input tensor
                        int8_t* rowbuf = rowbuf0;

                        if (isConv1D)
                        {
                            for( ofs = ofs0; ofs < ofs1; out_j = 0, ++out_i )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_j = out_j * stride_w - pad_l;
                                const int8_t* imgptr = data_inp0 + cn0*width + in_j;
                                ofs += delta;

                                // do im2row for a part of input tensor
                                if( is1x1 )
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w )
                                    {
                                        for( k = 0; k < vsz; k++ )
                                            rowbuf[k] = imgptr[k*inpPlaneSize];
                                    }
                                }
                                else
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                    {
                                        // this condition should be true for most of the tensor elements, i.e.
                                        // most of the time the kernel aperture is inside the tensor X-Y plane.
                                        if( out_j + 2 <= out_j1 && 0 <= in_j && in_j + stride_w*2 <= width - (kernel_w-1)*dilation_w )
                                        {
                                            for( k = 0; k < vsz; k++ )
                                            {
                                                int k1 = ofstab[k];
                                                int8_t v0 = imgptr[k1];
                                                int8_t v1 = imgptr[k1 + stride_w];
                                                rowbuf[k] = v0;
                                                rowbuf[k+vsz_a] = v1;
                                            }
                                            out_j++;
                                            rowbuf += vsz_a;
                                            imgptr += stride_w;
                                            in_j += stride_w;
                                        }
                                        else
                                        {
                                            int i0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                            int i1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                            // here some non-continuous sub-row of the row will not be
                                            // filled from the tensor; we need to make sure that the uncovered
                                            // elements are explicitly set to 0's. the easiest way is to
                                            // set all the elements to 0's before the loop.
                                            memset(rowbuf, (int8_t)inpZp, vsz*sizeof(rowbuf[0]));
                                            for( k = 0; k < ncn; k++ )
                                            {
                                                for( i = i0; i < i1; i++ )
                                                {
                                                    int imgofs = k*width + i*dilation_w;
                                                    rowbuf[k*kernel_w + i] = imgptr[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else if (isConv2D)
                        {
                            if( is1x1 && stride_w == 1 && stride_h == 1 )
                            {
                                const int8_t* imgptr = data_inp0 + (cn0*height + out_i)*width + out_j;
                                for( int j = 0; j < bsz; j++, rowbuf += vsz_a )
                                {
                                    if( j + 4 <= bsz )
                                    {
                                        k = 0;
                                        for( ; k < vsz; k++ )
                                        {
                                            const int8_t* inp = imgptr + j + k*inpPlaneSize;
                                            int8_t v0 = inp[0], v1 = inp[1], v2 = inp[2], v3 = inp[3];
                                            rowbuf[k] = v0;
                                            rowbuf[k + vsz_a] = v1;
                                            rowbuf[k + vsz_a*2] = v2;
                                            rowbuf[k + vsz_a*3] = v3;
                                        }
                                        j += 3;
                                        rowbuf += vsz_a*3;
                                    }
                                    else
                                    {
                                        for( k = 0; k < vsz; k++ )
                                        {
                                            rowbuf[k] = imgptr[j + k*inpPlaneSize];
                                        }
                                    }
                                }
                            }
                            else
                            for( ofs = ofs0; ofs < ofs1; out_j = 0, ++out_i )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_i = out_i * stride_h - pad_t;
                                int in_j = out_j * stride_w - pad_l;
                                const int8_t* imgptr = data_inp0 + (cn0*height + in_i)*width + in_j;
                                ofs += delta;

                                // do im2row for a part of input tensor
                                if( is1x1 )
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w )
                                    {
                                        for( k = 0; k < vsz; k++ )
                                            rowbuf[k] = imgptr[k*inpPlaneSize];
                                    }
                                }
                                else
                                {
                                    bool ok_i = 0 <= in_i && in_i < height - (kernel_h-1)*dilation_h;
                                    int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                    int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);

                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                    {
                                        // this condition should be true for most of the tensor elements, i.e.
                                        // most of the time the kernel aperture is inside the tensor X-Y plane.
                                        if( ok_i && out_j + 2 <= out_j1 && 0 <= in_j && in_j + stride_w*2 <= width - (kernel_w-1)*dilation_w )
                                        {
                                            for( k = 0; k < vsz; k++ )
                                            {
                                                int k1 = ofstab[k];
                                                int8_t v0 = imgptr[k1];
                                                int8_t v1 = imgptr[k1 + stride_w];
                                                rowbuf[k] = v0;
                                                rowbuf[k+vsz_a] = v1;
                                            }
                                            out_j++;
                                            rowbuf += vsz_a;
                                            imgptr += stride_w;
                                            in_j += stride_w;
                                        }
                                        else
                                        {
                                            int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                            int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                            // here some non-continuous sub-row of the row will not be
                                            // filled from the tensor; we need to make sure that the uncovered
                                            // elements are explicitly set to 0's. the easiest way is to
                                            // set all the elements to 0's before the loop.
                                            memset(rowbuf, (int8_t)inpZp, vsz*sizeof(rowbuf[0]));
                                            for( k = 0; k < ncn; k++ )
                                            {
                                                for( i = i0; i < i1; i++ )
                                                {
                                                    for( j = j0; j < j1; j++ )
                                                    {
                                                        int imgofs = k*(width*height) + i*(dilation_h*width) + j*dilation_w;
                                                        rowbuf[(k*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            for( ofs = ofs0; ofs < ofs1; out_d += (out_i + 1) / outH, out_i = (out_i + 1) % outH, out_j = 0 )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_d = out_d * stride_d - pad_d;
                                int in_i = out_i * stride_h - pad_t;
                                int in_j = out_j * stride_w - pad_l;
                                const int8_t* imgptr = data_inp0 + (cn0*depth*height + in_d*height + in_i)*width + in_j;
                                ofs += delta;

                                int d0 = std::max(0, (-in_d + dilation_d - 1) / dilation_d);
                                int d1 = std::min(kernel_d, (depth - in_d + dilation_d - 1) / dilation_d);

                                int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);

                                for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                {
                                    int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                    int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                    // here some non-continuous sub-row of the row will not be
                                    // filled from the tensor; we need to make sure that the uncovered
                                    // elements are explicitly set to 0's. the easiest way is to
                                    // set all the elements to 0's before the loop.
                                    memset(rowbuf, (int8_t)inpZp, vsz*sizeof(rowbuf[0]));
                                    for( k = 0; k < ncn; k++ )
                                    {
                                        for ( d = d0; d < d1; d++)
                                        {
                                            for( i = i0; i < i1; i++ )
                                            {
                                                for( j = j0; j < j1; j++ )
                                                {
                                                    int imgofs = k*(depth*width*height) + d*dilation_d*width*height + i*(dilation_h*width) + j*dilation_w;
                                                    rowbuf[(k*kernel_d*kernel_h + d*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // now compute dot product of the weights
                        // and im2row-transformed part of the tensor
                    #if CV_TRY_AVX512_SKX
                        if(useAVX512)
                            opt_AVX2::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multptr, cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                    #if CV_TRY_AVX2
                        if(useAVX2)
                            opt_AVX2::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multptr, cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                    #if CV_TRY_LASX
                        if(useLASX)
                            opt_LASX::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multptr, cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                    #if CV_TRY_RVV && defined(__riscv_v_intrinsic) && __riscv_v_intrinsic>=11000
                        if(useRVV)
                            opt_RVV::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multptr, cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                    #if CV_RVP052
                        if(isConv2D)
                            opt_RVP052::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multptr, cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                        for( int i = 0; i < outCn; i += 2 )
                        {
                            const int8_t* wptr0 = wptr + i*wstep;
                            const int8_t* wptr1 = wptr0 + wstep;
                            int* outptr0 = data_out0 + ofs0 + i*outPlaneSize;
                            int* outptr1 = outptr0 + outPlaneSize;
                            int bias0 = biasptr[i], bias1 = biasptr[i+1];
                            float mult0 = multptr[i], mult1 = multptr[i+1];

                            if( i+1 >= outCn )
                            {
                                wptr1 = wptr0;
                                outptr1 = outptr0;
                                bias1 = bias0;
                                mult1 = mult0;
                            }
                            int j = 0;
                        #if CV_SIMD128
                            v_int32x4 voutzp = v_setall_s32(outZp), outmin = v_setall_s32(-128), outmax = v_setall_s32(127);
                            v_float32x4 vmult0 = v_setall_f32(mult0), vmult1 = v_setall_f32(mult1);
                            for( ; j <= bsz - 4; j += 4 )
                            {
                                const int8_t* rptr = rowbuf0 + j*vsz_a;
                                v_int32x4 s0, s1;

                                if( cn0 == 0 )
                                {
                                    s0 = v_setall_s32(bias0);
                                    s1 = v_setall_s32(bias1);
                                }
                                else
                                {
                                    s0 = v_load(outptr0 + j);
                                    s1 = v_load(outptr1 + j);
                                }

                                v_int32x4 vs00 = v_setzero_s32(), vs01 = v_setzero_s32(),
                                          vs02 = v_setzero_s32(), vs03 = v_setzero_s32(),
                                          vs10 = v_setzero_s32(), vs11 = v_setzero_s32(),
                                          vs12 = v_setzero_s32(), vs13 = v_setzero_s32();
                                for( k = 0; k < vsz; k += 16, rptr += 16 )
                                {
                                    v_int8x16 w0 = v_load_aligned(wptr0 + k);
                                    v_int8x16 w1 = v_load_aligned(wptr1 + k);
                                    v_int8x16 r0 = v_load_aligned(rptr);
                                    v_int8x16 r1 = v_load_aligned(rptr + vsz_a);
                                    v_int8x16 r2 = v_load_aligned(rptr + vsz_a*2);
                                    v_int8x16 r3 = v_load_aligned(rptr + vsz_a*3);

                                    vs00 = v_dotprod_expand_fast(w0, r0, vs00);
                                    vs01 = v_dotprod_expand_fast(w0, r1, vs01);
                                    vs02 = v_dotprod_expand_fast(w0, r2, vs02);
                                    vs03 = v_dotprod_expand_fast(w0, r3, vs03);

                                    vs10 = v_dotprod_expand_fast(w1, r0, vs10);
                                    vs11 = v_dotprod_expand_fast(w1, r1, vs11);
                                    vs12 = v_dotprod_expand_fast(w1, r2, vs12);
                                    vs13 = v_dotprod_expand_fast(w1, r3, vs13);
                                }
                                s0 = v_add(s0, v_int32x4(v_reduce_sum(vs00), v_reduce_sum(vs01), v_reduce_sum(vs02), v_reduce_sum(vs03)));
                                s1 = v_add(s1, v_int32x4(v_reduce_sum(vs10), v_reduce_sum(vs11), v_reduce_sum(vs12), v_reduce_sum(vs13)));
                                if( cn1 == inpCn )
                                {
                                    s0 = v_add(voutzp, v_round(v_mul(v_cvt_f32(s0), vmult0)));
                                    s1 = v_add(voutzp, v_round(v_mul(v_cvt_f32(s1), vmult1)));

                                    s0 = v_min(v_max(s0, outmin), outmax);
                                    s1 = v_min(v_max(s1, outmin), outmax);
                                }
                                v_store(outptr0 + j, s0);
                                v_store(outptr1 + j, s1);
                            }
                        #endif
                            for( ; j < bsz; j++ )
                            {
                                const int8_t* rptr = rowbuf0 + j*vsz_a;
                                int s00, s10;

                                if( cn0 == 0 )
                                {
                                    s00 = bias0;
                                    s10 = bias1;
                                }
                                else
                                {
                                    s00 = outptr0[j];
                                    s10 = outptr1[j];
                                }

                                for( k = 0; k < vsz; k++ )
                                {
                                    int8_t r0 = rptr[k];
                                    s00 += (int)wptr0[k] * r0;
                                    s10 += (int)wptr1[k] * r0;
                                }
                                if( cn1 == inpCn )
                                {
                                    int out0 = outZp + (int)std::round(s00*mult0);
                                    int out1 = outZp + (int)std::round(s10*mult1);

                                    s00 = std::min(std::max(out0, -128), 127);
                                    s10 = std::min(std::max(out1, -128), 127);
                                }

                                outptr0[j] = s00;
                                outptr1[j] = s10;
                            }
                        }
                    }
                }
                if( activ_ )
                    activ_->forwardSlice(data_out0 + stripeStart, lutptr_,
                                         data_out0 + stripeStart, (int)(stripeEnd - stripeStart),
                                         outPlaneSize, startOutCn, startOutCn + outCn);
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

#if CV_SSE3
        uint32_t ftzMode = _MM_GET_FLUSH_ZERO_MODE();
        uint32_t dazMode = _MM_GET_DENORMALS_ZERO_MODE();
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        /*if (inputs[0].dims > 3) {
            printf("conv %s: input (%d x %d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2], inputs[0].size[3],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }
        else {
            printf("conv %s: input (%d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }*/

        int inpGroupCn = blobs[0].size[1];
        CV_Assert_N(inputs.size() == (size_t)1, inputs[0].size[1] % inpGroupCn == 0,
                    outputs.size() == 1, inputs[0].data != outputs[0].data);

        int ngroups = inputs[0].size[1] / inpGroupCn;
        CV_Assert(outputs[0].size[1] % ngroups == 0);

        int nstripes = std::max(getNumThreads(), 1);
        Mat outputInt32 = Mat(shape(outputs[0]), CV_32S);

        // TODO! optimize the fastQConv by AVX/AVX2, need to rethink the Pack_layout.
        // TODO: optimize Conv1D and Conv3D. Currently, the fastQConv branch is slower in DepthWise Conv3D and Conv1D cases.
        // Currently, on x86 and Loongson platform, the fastQConv branch is slower than ParallelConv branch.
        // So we keep it for now.
#if CV_NEON && CV_NEON_AARCH64
        int conv_dim = CONV_2D;
        if (inputs[0].dims == 3)
            conv_dim = CONV_1D;
        if (inputs[0].dims == 5)
            conv_dim = CONV_3D;

        if (conv_dim == CONV_2D)
        {
            if (!fastQConvImpl)
            {
                int K = outputs[0].size[1];
                int C = inputs[0].size[1];

                CV_Assert(outputs[0].size[1] % ngroups == 0);
                fastQConvImpl = initFastQConv(weightsMat, &biasvec[0], ngroups, K, C, kernel_size, strides, dilations,
                                              pads_begin, pads_end, conv_dim, outputMultiplier, input_sc, input_zp,
                                              output_sc, output_zp, per_channel);
            }

            runFastQConv(inputs[0], outputs[0], fastQConvImpl, nstripes, activ);
        }
        else
#endif
        {
            ParallelConv::run(inputs[0], outputInt32, weightsMat, outputMultiplier, biasvec, activationLUT, kernel_size, strides,
                              pads_begin, pads_end, dilations, activ.get(), ngroups, nstripes, input_zp, output_zp);

            outputInt32.convertTo(outputs[0], CV_8S);
        }

#if CV_SSE3
        _MM_SET_FLUSH_ZERO_MODE(ftzMode);
        _MM_SET_DENORMALS_ZERO_MODE(dazMode);
#endif
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size());

        int64 flops = 0;
        int karea = std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<size_t>());
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i])*(CV_BIG_INT(2)*karea*inputs[i][1] + 1);
        }
        return flops;
    }
};

Ptr<BaseConvolutionLayer> ConvolutionLayerInt8::create(const LayerParams &params)
{
    return Ptr<BaseConvolutionLayer>(new ConvolutionLayerInt8Impl(params));
}

}
}
