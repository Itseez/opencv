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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>
#include <limits>
#include <cfenv>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/activation.hpp"
using namespace cv::dnn::cuda4dnn;
#endif
#include <opencv2/core/utils/logger.hpp>

namespace cv
{
namespace dnn
{

using std::abs;
using std::exp;
using std::expm1;
using std::tanh;
using std::pow;
using std::ceil;
using std::floor;
using std::log;
using std::log1p;
using std::sqrt;
using std::round;
using std::acos;
using std::acosh;
using std::asin;
using std::asinh;
using std::atan;
using std::atanh;
using std::cos;
using std::cosh;
using std::erf;
using std::sin;
using std::sinh;
using std::tan;

template<typename Func>
class ElementWiseLayer : public Func::Layer
{
public:
    class PBody : public cv::ParallelLoopBody
    {
    public:
        const Func* func_;
        const Mat* src_;
        Mat* dst_;
        int nstripes_;

        PBody(const Func &func, const Mat &src, Mat& dst, int nstripes)
        {
            func_ = &func;
            src_ = &src;
            dst_ = &dst;
            nstripes_ = nstripes;
        }

        void operator()(const Range &r) const CV_OVERRIDE
        {
            int nstripes = nstripes_, nsamples = 1, outCn = 1;
            size_t planeSize = 1;

            if (src_->dims > 1)
            {
                nsamples = src_->size[0];
                outCn = src_->size[1];
            }
            else
                outCn = src_->size[0];

            for (int i = 2; i < src_->dims; ++i)
                planeSize *= src_->size[i];

            size_t stripeSize = (planeSize + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, planeSize);

            for( int i = 0; i < nsamples; i++ )
            {
                const float* srcptr = src_->ptr<float>(i) + stripeStart;
                float* dstptr = dst_->ptr<float>(i) + stripeStart;
                func_->apply(srcptr, dstptr, stripeStart, (int)(stripeEnd - stripeStart), planeSize, 0, outCn);
            }
        }
    };

    ElementWiseLayer(const Func &f=Func()) { func = f; }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return func.supportBackend(backendId, this->preferableTarget);
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
        func.finalize();
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        return func.initCannOp(Layer::name, inputs, nodes);
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto node = func.initNgraphAPI(ieInpNode);
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        auto operand = func.initWebnnAPI(webnnGraphBuilder, webnnInpOperand);
        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }
#endif


    virtual bool tryFuse(Ptr<dnn::Layer>& top) CV_OVERRIDE
    {
        return func.tryFuse(top);
    }

    void getScaleShift(Mat& scale_, Mat& shift_) const CV_OVERRIDE
    {
        func.getScaleShift(scale_, shift_);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(this->preferableTarget),
                   func.applyOCL(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F)
        {
            Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat &src = inputs[i];
            Mat &dst = outputs[i];
            CV_Assert(src.size == dst.size && src.type() == dst.type() &&
                      src.isContinuous() && dst.isContinuous() && src.type() == CV_32F);

            const int nstripes = getNumThreads();
            PBody body(func, src, dst, nstripes);
            parallel_for_(Range(0, nstripes), body, nstripes);
        }
    }

    void forwardSlice(const float* src, float* dst, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        func.apply(src, dst, -1, len, planeSize, cn0, cn1);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        return func.initCUDA(Layer::preferableTarget, context->stream);
    }
#endif

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        long flops = 0;
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i]) * func.getFLOPSPerElement();
        }
        return flops;
    }

    Func func;
};

#ifdef HAVE_OPENCL
static String oclGetTMacro(const UMat &m)
{
    String str_name = ocl::typeToStr(m.type());

    if (str_name == "short")
        str_name = "half";

    return format("-DT=%s -Dconvert_T=convert_%s ", str_name.c_str(), str_name.c_str());
}
#endif

struct BaseFunctor
{
    void finalize() {}

    bool tryFuse(Ptr<dnn::Layer>&) { return false; }

    void getScaleShift(Mat&, Mat&) const {}
};

struct ReLUFunctor : public BaseFunctor
{
    typedef ReLULayer Layer;
    float slope;

    explicit ReLUFunctor(float slope_=1.f) : slope(slope_) {}

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_DNN_NGRAPH
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
#ifdef HAVE_WEBNN
        if (backendId == DNN_BACKEND_WEBNN) {
            // TODO: support PRELU
            if (slope != 0)
            {
                CV_LOG_WARNING(NULL, "PRELU is not supported now.");
            }
            return slope == 0;
        }
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const
    {
        CV_UNUSED(stripeStart);
        float s = slope;
        for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
        {
            int i = 0;
#if CV_SIMD128
            v_float32x4 s4 = v_setall_f32(s), z = v_setzero_f32();
            for( ; i <= len - 16; i += 16 )
            {
                v_float32x4 x0 = v_load(srcptr + i);
                v_float32x4 x1 = v_load(srcptr + i + 4);
                v_float32x4 x2 = v_load(srcptr + i + 8);
                v_float32x4 x3 = v_load(srcptr + i + 12);
                x0 = v_select(v_ge(x0, z), x0, v_mul(x0, s4));
                x1 = v_select(v_ge(x1, z), x1, v_mul(x1, s4));
                x2 = v_select(v_ge(x2, z), x2, v_mul(x2, s4));
                x3 = v_select(v_ge(x3, z), x3, v_mul(x3, s4));
                v_store(dstptr + i, x0);
                v_store(dstptr + i + 4, x1);
                v_store(dstptr + i + 8, x2);
                v_store(dstptr + i + 12, x3);
            }
#endif
            for( ; i < len; i++ )
            {
                float x = srcptr[i];
                dstptr[i] = x >= 0.f ? x : s*x;
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ReLUOp>(target, stream, slope);
    }
#endif

#ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        const char *buildoptSlope = (slope == 0) ? "-DRELU_NO_SLOPE" : "";
        String buildopt = oclGetTMacro(src) + buildoptSlope;

        if (!ker.create("ReLUForward", ocl::dnn::activations_oclsrc, buildopt))
            return false;

        if (slope != 0)
            ker.set(3, (float)slope);

        return true;
    }

    bool applyOCL(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat& src = inputs[i];
            UMat& dst = outputs[i];
            CV_Assert(src.isContinuous() && dst.isContinuous() && !src.offset && !dst.offset);

            ocl::Kernel kernel;
            CV_Assert(initKernel(kernel, src));
            kernel.set(0, (int)src.total());
            kernel.set(1, ocl::KernelArg::PtrReadOnly(src));
            kernel.set(2, ocl::KernelArg::PtrWriteOnly(dst));

            size_t gSize = src.total();
            CV_Assert(kernel.run(1, &gSize, NULL, false));
        }

        return true;
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x_desc = x->getTensorDesc();

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        if (slope)
        {
            auto op = std::make_shared<ge::op::LeakyRelu>(name);

            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);

            op->set_attr_negative_slope(slope);

            op->update_output_desc_y(*output_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }

        auto op = std::make_shared<ge::op::Relu>(name);

        op->set_input_x_by_name(*op_x, x->name.c_str());
        op->update_input_desc_x(*x_desc);

        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        if (slope) {
            auto param = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &slope);
            return std::make_shared<ov::op::v0::PRelu>(node, param);
        }
        return std::make_shared<ov::op::v0::Relu>(node);
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    ml::Operand initWebnnAPI(const ml::GraphBuilder& builder, const ml::Operand& input)
    {
        return builder.Relu(input);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

struct ReLU6Functor : public BaseFunctor
{
    typedef ReLU6Layer Layer;
    float minValue, maxValue;

    ReLU6Functor(float minValue_ = 0.0f, float maxValue_ = 6.0f)
        : minValue(minValue_), maxValue(maxValue_)
    {
        CV_Assert(minValue <= maxValue);
    }

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_WEBNN ||
               backendId == DNN_BACKEND_CANN;
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const
    {
        CV_UNUSED(stripeStart);
        for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
        {
            int i = 0;
#if CV_SIMD128
            v_float32x4 minV = v_setall_f32(minValue), maxV = v_setall_f32(maxValue);
            for( ; i <= len - 16; i += 16 )
            {
                v_float32x4 x0 = v_load(srcptr + i);
                v_float32x4 x1 = v_load(srcptr + i + 4);
                v_float32x4 x2 = v_load(srcptr + i + 8);
                v_float32x4 x3 = v_load(srcptr + i + 12);
                x0 = v_min(v_max(minV, x0), maxV);
                x1 = v_min(v_max(minV, x1), maxV);
                x2 = v_min(v_max(minV, x2), maxV);
                x3 = v_min(v_max(minV, x3), maxV);
                v_store(dstptr + i, x0);
                v_store(dstptr + i + 4, x1);
                v_store(dstptr + i + 8, x2);
                v_store(dstptr + i + 12, x3);
            }
#endif
            for( ; i < len; i++ )
            {
                float x = srcptr[i];
                if (x >= minValue)
                    dstptr[i] = x <= maxValue ? x : maxValue;
                else
                    dstptr[i] = minValue;
            }
        }
    }

#ifdef HAVE_OPENCL
    bool applyOCL(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);
        String buildopt = oclGetTMacro(inputs[0]);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat& src = inputs[i];
            UMat& dst = outputs[i];

            ocl::Kernel kernel("ReLU6Forward", ocl::dnn::activations_oclsrc, buildopt);
            kernel.set(0, (int)src.total());
            kernel.set(1, ocl::KernelArg::PtrReadOnly(src));
            kernel.set(2, ocl::KernelArg::PtrWriteOnly(dst));
            kernel.set(3, (float)minValue);
            kernel.set(4, (float)maxValue);

            size_t gSize = src.total();
            CV_Assert(kernel.run(1, &gSize, NULL, false));
        }

        return true;
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ClippedReLUOp>(target, stream, minValue, maxValue);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::ClipByValue>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        Mat min_value_mat(1, 1, CV_32F, Scalar(minValue));
        std::vector<int> shape_{1};
        auto op_const_minv = std::make_shared<CannConstOp>(min_value_mat.data, min_value_mat.type(), shape_, cv::format("%s_min_value", name.c_str()));
        op->set_input_clip_value_min(*(op_const_minv->getOp()));
        op->update_input_desc_clip_value_min(*(op_const_minv->getTensorDesc()));

        Mat max_value_mat(1, 1, CV_32F, Scalar(maxValue));
        auto op_const_maxv = std::make_shared<CannConstOp>(max_value_mat.data, max_value_mat.type(), shape_, cv::format("%s_max_value", name.c_str()));
        op->set_input_clip_value_max(*(op_const_maxv->getOp()));
        op->update_input_desc_clip_value_max(*(op_const_maxv->getTensorDesc()));

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif


#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Clamp>(node, minValue, maxValue);
    }
#endif  // HAVE_DNN_NGRAPH



#ifdef HAVE_WEBNN
    ml::Operand initWebnnAPI(const ml::GraphBuilder& builder, const ml::Operand& input)
    {
        ml::ClampOptions clampOptions;
        clampOptions.minValue = minValue;
        clampOptions.maxValue = maxValue;
        return builder.Clamp(input, &clampOptions);
    }
#endif

    int64 getFLOPSPerElement() const { return 2; }
};

template <class T>
struct BaseDefaultFunctor : public BaseFunctor
{
    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const
    {
        CV_UNUSED(stripeStart);
        for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
        {
            for( int i = 0; i < len; i++ )
            {
                float x = srcptr[i];
                dstptr[i] = static_cast<const T*>(this)->calculate(x);
            }
        }
    }

#ifdef HAVE_OPENCL
    bool applyOCL(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);
        String buildopt = oclGetTMacro(inputs[0]);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat& src = inputs[i];
            UMat& dst = outputs[i];

            ocl::Kernel kernel(ocl_kernel_name, ocl::dnn::activations_oclsrc, buildopt);
            kernel.set(0, static_cast<int>(src.total()));
            kernel.set(1, ocl::KernelArg::PtrReadOnly(src));
            kernel.set(2, ocl::KernelArg::PtrWriteOnly(dst));
            static_cast<const T*>(this)->setKernelParams(kernel);

            size_t gSize = src.total();
            CV_Assert(kernel.run(1, &gSize, nullptr, false));
        }

        return true;
    }
#endif

    inline void setKernelParams(ocl::Kernel& kernel) const {}

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        CV_Error(Error::StsNotImplemented, "");
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        CV_Error(Error::StsNotImplemented, "");
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        CV_Error(Error::StsNotImplemented, "");
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    ml::Operand initWebnnAPI(const ml::GraphBuilder& builder, const ml::Operand& input)
    {
        CV_Error(Error::StsNotImplemented, "");
    }
#endif

private:
    static const char* const ocl_kernel_name;
};

namespace {
    // Refer to v_erf in modules/core/include/opencv2/core/hal/intrin_math.hpp
    constexpr float c_erf_coef0 = 0.3275911f;
    constexpr float c_erf_coef1 = 1.061405429f;
    constexpr float c_erf_coef2 = -1.453152027f;
    constexpr float c_erf_coef3 = 1.421413741f;
    constexpr float c_erf_coef4 = -0.284496736f;
    constexpr float c_erf_coef5 = 0.254829592f;

    inline float erf_approx(float v) {
        float t = 1.f / fmaf(fabsf(v), c_erf_coef0, 1.f);
        float r = fmaf(c_erf_coef1, t, c_erf_coef2);
        r = fmaf(r, t, c_erf_coef3);
        r = fmaf(r, t, c_erf_coef4);
        r = fmaf(r, t, c_erf_coef5);
        r = 1.f - r * t * expf(-v * v);
        return std::copysignf(r, v);
    }
}

struct GeluFunctor : public BaseFunctor {
    using Layer = GeluLayer;
    int vlanes;

    explicit GeluFunctor() {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int) {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            // 0.5f * x * (1.0f + erf(x * M_SQRT1_2));
            v_float32 half = vx_setall_f32(0.5f),
                      one = vx_setall_f32(1.0f),
                      reciprocal_sqrt2 = vx_setall_f32(M_SQRT1_2);
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x0 = vx_load(srcptr + i);

                // t = x * M_SQRT1_2
                v_float32 t0 = v_mul(reciprocal_sqrt2, x0);

                // t = 1.0f + t
                t0 = v_add(one, v_erf(t0));

                // x = 0.5 * x
                x0 = v_mul(half, x0);

                // x = x * t
                x0 = v_mul(x0, t0);

                vx_store(dstptr + i, x0);
            }
#endif
            // 0.5f * x * (1.0f + erf(x * M_SQRT1_2));
            for( ; i < len; i++ )
            {
                float x = srcptr[i];
                dstptr[i] = 0.5f * x * (1.0f + erf_approx(x * M_SQRT1_2));
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::GeluOp>(target, stream);
    }
#endif

#ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        String buildopt = oclGetTMacro(src);

        if (!ker.create("GeluForward", ocl::dnn::activations_oclsrc, buildopt))
            return false;

        return true;
    }

    bool applyOCL(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat& src = inputs[i];
            UMat& dst = outputs[i];
            CV_Assert(src.isContinuous() && dst.isContinuous() && !src.offset && !dst.offset);

            ocl::Kernel kernel;
            CV_Assert(initKernel(kernel, src));
            kernel.set(0, (int)src.total());
            kernel.set(1, ocl::KernelArg::PtrReadOnly(src));
            kernel.set(2, ocl::KernelArg::PtrWriteOnly(dst));

            size_t gSize = src.total();
            CV_Assert(kernel.run(1, &gSize, NULL, false));
        }

        return true;
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Gelu>(node);
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        CV_Error(Error::StsNotImplemented, "");
    }
#endif // HAVE_CANN

    int64 getFLOPSPerElement() const { return 100; }
};

namespace GeluApproximationConstants
{
    static constexpr float sqrt_2_pi = 0.7978845834732056f;
    static constexpr float coef_sqrt_2_pi = 0.044714998453855515f * sqrt_2_pi;
}

struct GeluApproximationFunctor : public BaseDefaultFunctor<GeluApproximationFunctor>
{
    typedef GeluApproximationLayer Layer;

    explicit GeluApproximationFunctor() {}

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    inline float calculate(float x) const
    {
        return 0.5f * x * (1.f + tanh(x * (GeluApproximationConstants::sqrt_2_pi +
                                           GeluApproximationConstants::coef_sqrt_2_pi * x * x)));
    }

    int64 getFLOPSPerElement() const { return 100; }
};

template<>
const char* const BaseDefaultFunctor<GeluApproximationFunctor>::ocl_kernel_name = "GeluApproximationForward";

struct TanHFunctor : public BaseDefaultFunctor<TanHFunctor>
{
    typedef TanHLayer Layer;

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        return tanh(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::TanHOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Tanh>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Tanh>(node);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const TanHFunctor::BaseDefaultFunctor<TanHFunctor>::ocl_kernel_name = "TanHForward";

struct SwishFunctor : public BaseDefaultFunctor<SwishFunctor>
{
    using Layer = SwishLayer;

    int vlanes;

    explicit SwishFunctor() {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        return x / (1.f + exp(-x));
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            // x / (1.f + exp(-x));
            v_float32 one = vx_setall_f32(1.0f),
                      zero = vx_setzero_f32();
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x = vx_load(srcptr + i);

                v_float32 t = v_sub(zero, x);
                t = v_exp(t);
                t = v_add(one, t);
                t = v_div(x, t);

                vx_store(dstptr + i, t);
            }
#endif
            // In case SIMD is not available or len < vlanes
            for (; i < len; i++) {
                dstptr[i] = calculate(srcptr[i]);
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SwishOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Swish>(name);

        op->set_attr_scale(1.0f);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(node);
        return std::make_shared<ov::op::v1::Multiply>(node, sigmoid);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 3; }
};

template<>
const char* const SwishFunctor::BaseDefaultFunctor<SwishFunctor>::ocl_kernel_name = "SwishForward";

namespace {
    constexpr float MISH_THRESHOLD = -36.73f;
}

/*
    This implementation is derived from
    https://github.com/vpisarev/ficus/blob/3c9a8b78f49e17489c5e1fd6dd5dd487348c99c2/lib/NN/OpElemwise.fx#L110
*/
struct MishFunctor : public BaseDefaultFunctor<MishFunctor>
{
    using Layer = MishLayer;

    int vlanes;

    explicit MishFunctor() {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        float y = x > MISH_THRESHOLD ? std::exp(-x) : 1.f;
        x *= x > MISH_THRESHOLD ? 1.f : 0.f;
        return x * (1 + 2 * y) / (1 + 2 * y + 2 * y * y);
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 v_threshold = vx_setall_f32(MISH_THRESHOLD), one = vx_setall_f32(1.f), z = vx_setzero_f32();
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x = vx_load(srcptr + i);

                x = v_select(v_le(x, v_threshold), z, x);
                v_float32 y = v_exp(v_sub(z, x));
                v_float32 _2y = v_add(y, y),
                          _2ya1 = v_add(_2y, one);
                x = v_div(v_mul(x, _2ya1), v_add(_2ya1, v_mul(_2y, y)));

                vx_store(dstptr + i, x);
            }
#endif
            // In case SIMD is not available or len < vlanes
            for (; i < len; i++) {
                dstptr[i] = calculate(srcptr[i]);
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::MishOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Mish>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v4::Mish>(node);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 3; }
};

template<>
const char* const MishFunctor::BaseDefaultFunctor<MishFunctor>::ocl_kernel_name = "MishForward";

struct SigmoidFunctor : public BaseDefaultFunctor<SigmoidFunctor>
{
    typedef SigmoidLayer Layer;

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        float y;
        if (x >= 0)
            y = 1.f / (1.f + exp(-x));
        else {
            y = exp(x);
            y = y / (1 + y);
        }
        return y;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SigmoidOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Sigmoid>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Sigmoid>(node);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 3; }
};

template<>
const char* const SigmoidFunctor::BaseDefaultFunctor<SigmoidFunctor>::ocl_kernel_name = "SigmoidForward";

struct ELUFunctor : public BaseDefaultFunctor<ELUFunctor>
{
    using Layer = ELULayer;

    float alpha;
    int vlanes;

    explicit ELUFunctor(float alpha_ = 1.f) : alpha(alpha_) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        return x >= 0.f ? x : alpha * (exp(x) - 1.f);
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 z = vx_setzero_f32(), v_alpha = vx_setall_f32(alpha), one = vx_setall_f32(1.0f);
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x = vx_load(srcptr + i);

                v_float32 t = v_mul(v_alpha, v_sub(v_exp(x), one));
                x = v_select(v_ge(x, z), x, t);

                vx_store(dstptr + i, x);
            }
#endif
            // In case SIMD is not available or len < vlanes
            for (; i < len; i++) {
                dstptr[i] = calculate(srcptr[i]);
            }
        }
    }

    inline void setKernelParams(ocl::Kernel& kernel) const
    {
        kernel.set(3, alpha);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ELUOp>(target, stream, alpha);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Elu>(name);

        op->set_attr_alpha(alpha);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Elu>(node, alpha);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 2; }
};

template<>
const char* const ELUFunctor::BaseDefaultFunctor<ELUFunctor>::ocl_kernel_name = "ELUForward";

struct AbsValFunctor : public BaseDefaultFunctor<AbsValFunctor>
{
    typedef AbsLayer Layer;

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        return abs(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AbsValOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Abs>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Abs>(node);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const AbsValFunctor::BaseDefaultFunctor<AbsValFunctor>::ocl_kernel_name = "AbsValForward";

struct BNLLFunctor : public BaseDefaultFunctor<BNLLFunctor>
{
    typedef BNLLLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        // https://github.com/BVLC/caffe/blame/1.0/src/caffe/layers/bnll_layer.cpp#L17
        return x > 0 ? x + log(1.f + exp(-x)) : log(1.f + exp(x));
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::BNLLOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::BNLL>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

    int64 getFLOPSPerElement() const { return 5; }
};

template<>
const char* const BNLLFunctor::BaseDefaultFunctor<BNLLFunctor>::ocl_kernel_name = "BNLLForward";

struct CeilFunctor : public BaseDefaultFunctor<CeilFunctor>
{
    typedef CeilLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return ceil(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::CeilOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::BNLL>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<CeilFunctor>::ocl_kernel_name = "CeilForward";

struct FloorFunctor : public BaseDefaultFunctor<FloorFunctor>
{
    typedef FloorLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA   ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        return floor(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::FloorOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::Floor>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<FloorFunctor>::ocl_kernel_name = "FloorForward";

struct LogFunctor : public BaseDefaultFunctor<LogFunctor>
{
    typedef LogLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return log(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::LogOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<LogFunctor>::ocl_kernel_name = "LogForward";

struct RoundFunctor : public BaseDefaultFunctor<RoundFunctor>
{
    typedef RoundLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        // Rounds to even numbers in halfway cases, so 2.5 -> 2, -2.5 -> -2
        int old_rounding_direction = fegetround();
        fesetround(FE_TONEAREST);
        float y = std::nearbyint(x);
        fesetround(old_rounding_direction);
        return y;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::RoundOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 2; }
};

template<>
const char* const BaseDefaultFunctor<RoundFunctor>::ocl_kernel_name = "RoundForward";

struct SqrtFunctor : public BaseDefaultFunctor<SqrtFunctor>
{
    typedef SqrtLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return sqrt(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SqrtOp>(target, stream);
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        return std::make_shared<ov::op::v0::Sqrt>(node);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<SqrtFunctor>::ocl_kernel_name = "SqrtForward";

struct AcosFunctor : public BaseDefaultFunctor<AcosFunctor>
{
    typedef AcosLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return acos(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AcosOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<AcosFunctor>::ocl_kernel_name = "AcosForward";

struct AcoshFunctor : public BaseDefaultFunctor<AcoshFunctor>
{
    typedef AcoshLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return acosh(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AcoshOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<AcoshFunctor>::ocl_kernel_name = "AcoshForward";

struct AsinFunctor : public BaseDefaultFunctor<AsinFunctor>
{
    typedef AsinLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return asin(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AsinOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<AsinFunctor>::ocl_kernel_name = "AsinForward";

struct AsinhFunctor : public BaseDefaultFunctor<AsinhFunctor>
{
    typedef AsinhLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return asinh(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AsinhOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<AsinhFunctor>::ocl_kernel_name = "AsinhForward";

struct AtanFunctor : public BaseDefaultFunctor<AtanFunctor>
{
    typedef AtanLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return atan(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AtanOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<AtanFunctor>::ocl_kernel_name = "AtanForward";

struct AtanhFunctor : public BaseDefaultFunctor<AtanhFunctor>
{
    typedef AtanhLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return atanh(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::AtanhOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<AtanhFunctor>::ocl_kernel_name = "AtanhForward";

struct CosFunctor : public BaseDefaultFunctor<CosFunctor>
{
    typedef CosLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return cos(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::CosOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<CosFunctor>::ocl_kernel_name = "CosForward";

struct CoshFunctor : public BaseDefaultFunctor<CoshFunctor>
{
    typedef CoshLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return cosh(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::CoshOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<CoshFunctor>::ocl_kernel_name = "CoshForward";

struct ErfFunctor : public BaseDefaultFunctor<ErfFunctor>
{
    typedef ErfLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return erf(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ErfOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<ErfFunctor>::ocl_kernel_name = "ErfForward";

struct HardSwishFunctor : public BaseDefaultFunctor<HardSwishFunctor>
{
    using Layer = HardSwishLayer;
    int vlanes;

    explicit HardSwishFunctor() {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA   ||
               backendId == DNN_BACKEND_CANN;
    }

    inline float calculate(float x) const
    {
        return x * std::max(0.f, std::min(1.f, x / 6.f + 0.5f));
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 zero = vx_setzero_f32(), one = vx_setall_f32(1.0f),
                      half = vx_setall_f32(0.5f), sixth = vx_setall_f32(1 / 6.0f);
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x = vx_load(srcptr + i);

                v_float32 t = v_add(v_mul(x, sixth), half);
                t = v_min(one, t);
                t = v_max(zero, t);
                t = v_mul(x, t);

                vx_store(dstptr + i, t);
            }
#endif
            // In case SIMD is not available or len > vlanes
            for (; i < len; i++) {
                dstptr[i] = calculate(srcptr[i]);
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::HardSwishOp>(target, stream);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();

        auto op = std::make_shared<ge::op::HardSwish>(name);

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto x_desc = x->getTensorDesc();
        op->update_input_desc_x(*x_desc);

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<HardSwishFunctor>::ocl_kernel_name = "HardSwishForward";

struct SinFunctor : public BaseDefaultFunctor<SinFunctor>
{
    typedef SinLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return sin(x);
    }

#ifdef HAVE_CUDA
        Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
        {
            return make_cuda_node<cuda4dnn::SinOp>(target, stream);
        }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<SinFunctor>::ocl_kernel_name = "SinForward";

struct SinhFunctor : public BaseDefaultFunctor<SinhFunctor>
{
    typedef SinhLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return sinh(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SinhOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<SinhFunctor>::ocl_kernel_name = "SinhForward";

struct SoftplusFunctor : public BaseDefaultFunctor<SoftplusFunctor>
{
    typedef SoftplusLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return log1p(exp(x));
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SoftplusOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<SoftplusFunctor>::ocl_kernel_name = "SoftplusForward";

struct SoftsignFunctor : public BaseDefaultFunctor<SoftsignFunctor>
{
    typedef SoftsignLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return x / (1.f + abs(x));
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SoftsignOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<SoftsignFunctor>::ocl_kernel_name = "SoftsignForward";

struct TanFunctor : public BaseDefaultFunctor<TanFunctor>
{
    typedef TanLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return tan(x);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::TanOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<TanFunctor>::ocl_kernel_name = "TanForward";

struct CeluFunctor : public BaseDefaultFunctor<CeluFunctor>
{
    using Layer = CeluLayer;

    float alpha;
    int vlanes;

    explicit CeluFunctor(float alpha_ = 1.f) : alpha(alpha_) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return std::max(0.f, x) + std::min(0.f, alpha * expm1(x / alpha));
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 zero = vx_setzero_f32(), v_alpha = vx_setall_f32(alpha),
                      one = vx_setall_f32(1.0f), v_ralpha = vx_setall_f32(1.0f / alpha);
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x = vx_load(srcptr + i);

                v_float32 t = v_min(zero, v_mul(v_alpha, v_sub(v_exp(v_mul(x, v_ralpha)), one)));
                t = v_add(v_max(zero, x), t);

                vx_store(dstptr + i, t);
            }
#endif
            // In case SIMD is not available or len < vlanes
            for (; i < len; i++) {
                dstptr[i] = calculate(srcptr[i]);
            }
        }
    }

    inline void setKernelParams(ocl::Kernel& kernel) const
    {
        kernel.set(3, alpha);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::CeluOp>(target, stream, alpha);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<CeluFunctor>::ocl_kernel_name = "CeluForward";

struct HardSigmoidFunctor : public BaseDefaultFunctor<HardSigmoidFunctor>
{
    typedef HardSigmoidLayer Layer;

    float alpha;
    float beta;

    explicit HardSigmoidFunctor(float alpha_ = 0.2f, float beta_ = 0.5f) : alpha(alpha_), beta(beta_) {}

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return max(0.f, min(1.f, alpha * x + beta));
    }

    inline void setKernelParams(ocl::Kernel& kernel) const
    {
        kernel.set(3, alpha);
        kernel.set(4, beta);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::HardSigmoidOp>(target, stream, alpha, beta);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<HardSigmoidFunctor>::ocl_kernel_name = "HardSigmoidForward";

struct SeluFunctor : public BaseDefaultFunctor<SeluFunctor>
{
    using Layer = SeluLayer;

    float alpha;
    float gamma;
    int vlanes;

    explicit SeluFunctor(float alpha_ = 1.67326319217681884765625f,
                         float gamma_ = 1.05070102214813232421875f)
        : alpha(alpha_), gamma(gamma_) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vlanes = VTraits<v_float32>::vlanes();
#else
        vlanes = 1;
#endif
    }

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return gamma * (x > 0.f ? x : alpha * expm1(x));
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const {
        CV_UNUSED(stripeStart);
        for (int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize) {
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            v_float32 z = vx_setzero_f32(), one = vx_setall_f32(1.0f),
                      v_alpha = vx_setall_f32(alpha), v_gamma = vx_setall_f32(gamma);
            for (; i <= len - vlanes; i += vlanes) {
                v_float32 x = vx_load(srcptr + i);

                v_float32 t = v_mul(v_alpha, v_sub(v_exp(x), one));
                x = v_select(v_le(x, z), t, x);
                x = v_mul(v_gamma, x);

                vx_store(dstptr + i, x);
            }
#endif
            // In case SIMD is not available or len > vlanes
            for (; i < len; i++) {
                dstptr[i] = calculate(srcptr[i]);
            }
        }
    }

    inline void setKernelParams(ocl::Kernel& kernel) const
    {
        kernel.set(3, alpha);
        kernel.set(4, gamma);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SeluOp>(target, stream, alpha, gamma);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<SeluFunctor>::ocl_kernel_name = "SeluForward";

struct ThresholdedReluFunctor : public BaseDefaultFunctor<ThresholdedReluFunctor>
{
    typedef ThresholdedReluLayer Layer;

    float alpha;

    explicit ThresholdedReluFunctor(float alpha_ = 1.f) : alpha(alpha_) {}


    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return x > alpha ? x : 0.f;
    }

    inline void setKernelParams(ocl::Kernel& kernel) const
    {
        kernel.set(3, alpha);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ThresholdedReluOp>(target, stream, alpha);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const BaseDefaultFunctor<ThresholdedReluFunctor>::ocl_kernel_name = "ThresholdedReluForward";

struct PowerFunctor : public BaseFunctor
{
    typedef PowerLayer Layer;

    float power, scale, shift;
    float originPower, originScale, originShift;

    explicit PowerFunctor(float power_ = 1.f, float scale_ = 1.f, float shift_ = 0.f)
        : power(power_), scale(scale_), shift(shift_),
          originPower(power_), originScale(scale_), originShift(shift_) {}

    bool supportBackend(int backendId, int targetId)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        {
            return backendId == DNN_BACKEND_OPENCV ||
                   backendId == DNN_BACKEND_CUDA;
        }
    }

    void finalize()
    {
        power = originPower;
        scale = originScale;
        shift = originShift;
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const
    {
        CV_UNUSED(stripeStart);
        float a = scale, b = shift, p = power;
        if( p == 1.f )
        {
            for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
            {
                for( int i = 0; i < len; i++ )
                {
                    float x = srcptr[i];
                    dstptr[i] = a*x + b;
                }
            }
        }
        else
        {
            for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
            {
                for( int i = 0; i < len; i++ )
                {
                    float x = srcptr[i];
                    dstptr[i] = pow(a*x + b, p);
                }
            }
        }
    }

#ifdef HAVE_OPENCL
    bool applyOCL(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);
        String buildopt = oclGetTMacro(inputs[0]);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat& src = inputs[i];
            UMat& dst = outputs[i];

            ocl::Kernel kernel("PowForward", ocl::dnn::activations_oclsrc, buildopt);
            kernel.set(0, (int)src.total());
            kernel.set(1, ocl::KernelArg::PtrReadOnly(src));
            kernel.set(2, ocl::KernelArg::PtrWriteOnly(dst));
            kernel.set(3, (float)power);
            kernel.set(4, (float)scale);
            kernel.set(5, (float)shift);

            size_t gSize = src.total();
            CV_Assert(kernel.run(1, &gSize, NULL, false));
        }

        return true;
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::PowerOp>(target, stream, power, scale, shift);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        CV_Error(Error::StsNotImplemented, "");
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                 ov::Shape{1}, &scale);
        auto shift_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                 ov::Shape{1}, &shift);

        auto mul = std::make_shared<ov::op::v1::Multiply>(scale_node, node, ov::op::AutoBroadcastType::NUMPY);
        auto scale_shift = std::make_shared<ov::op::v1::Add>(mul, shift_node, ov::op::AutoBroadcastType::NUMPY);

        if (power == 1)
            return scale_shift;

        auto power_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                 ov::Shape{1}, &power);
        return std::make_shared<ov::op::v1::Power>(scale_shift, power_node, ov::op::AutoBroadcastType::NUMPY);
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    ml::Operand initWebnnAPI(const ml::GraphBuilder& builder, const ml::Operand& input)
    {
        CV_Error(Error::StsNotImplemented, "");
        ml::Operand operand;
        return operand;
    }
#endif

    bool tryFuse(Ptr<dnn::Layer>& top)
    {
        if (power != 1.0f && shift != 0.0f)
            return false;

        Mat w, b;
        top->getScaleShift(w, b);
        if ((w.empty() && b.empty()) || w.total() > 1 || b.total() > 1)
            return false;

        float nextScale = w.empty() ? 1.0f : w.at<float>(0);
        float nextShift = b.empty() ? 0.0f : b.at<float>(0);
        scale = std::pow(scale, power) * nextScale;
        shift = nextScale * shift + nextShift;
        return true;
    }

    void getScaleShift(Mat& _scale, Mat& _shift) const
    {
        if (power == 1.0f)
        {
            _scale = Mat(1, 1, CV_32F, Scalar(scale));
            _shift = Mat(1, 1, CV_32F, Scalar(shift));
        }
    }

    int64 getFLOPSPerElement() const { return power == 1 ? 2 : 10; }
};

struct ExpFunctor : public BaseDefaultFunctor<ExpFunctor>
{
    typedef ExpLayer Layer;
    float base, scale, shift;
    float normScale, normShift;

    ExpFunctor(float base_ = -1.f, float scale_ = 1.f, float shift_ = 0.f)
        : base(base_), scale(scale_), shift(shift_)
    {
        // For base > 0 :
        // y     = base^(scale * input + shift)
        // ln(y) = ln(base)*(scale * input + shift)
        // y     = exp((ln(base)*scale) * input + (ln(base)*shift))
        // y     = exp(normalized_scale * input + normalized_shift)
        CV_Check(base, base == -1.f || base > 0.f, "Unsupported 'base' value");
        const float ln_base = (base == -1.f) ? 1.f : log(base);
        normScale = scale * ln_base;
        normShift = shift * ln_base;
    }

    bool supportBackend(int backendId, int targetId)
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    inline float calculate(float x) const
    {
        return exp(normScale * x + normShift);
    }

    inline void setKernelParams(ocl::Kernel& kernel) const
    {
        kernel.set(3, normScale);
        kernel.set(4, normShift);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ExpOp>(target, stream, normScale, normShift);
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                 ov::Shape{1}, &normScale);
        auto shift_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                 ov::Shape{1}, &normShift);
        auto mul = std::make_shared<ov::op::v1::Multiply>(scale_node, node, ov::op::AutoBroadcastType::NUMPY);
        auto scale_shift = std::make_shared<ov::op::v1::Add>(mul, shift_node, ov::op::AutoBroadcastType::NUMPY);
        return std::make_shared<ov::op::v0::Exp>(scale_shift);
    }
#endif  // HAVE_DNN_NGRAPH

    int64 getFLOPSPerElement() const { return 3; }
};

template<>
const char* const ExpFunctor::BaseDefaultFunctor<ExpFunctor>::ocl_kernel_name = "ExpForward";

struct ChannelsPReLUFunctor : public BaseFunctor
{
    typedef ChannelsPReLULayer Layer;
    Mat scale;
#ifdef HAVE_OPENCL
    UMat scale_umat;
    std::string oclKernelName = "ChannelsPReLUForward";
#endif

    explicit ChannelsPReLUFunctor(const Mat& scale_=Mat()) : scale(scale_)
    {
    }

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const
    {
        CV_UNUSED(stripeStart);
        CV_Assert(scale.isContinuous() && scale.type() == CV_32F);

        const float* scaleptr = scale.ptr<float>();
        CV_Assert( 0 <= cn0 && cn0 < cn1 && cn1 <= (int)scale.total() );

        for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize )
        {
            float s = scaleptr[cn];
            int i = 0;
        #if CV_SIMD128
            v_float32x4 s4 = v_setall_f32(s), z = v_setzero_f32();
            for( ; i <= len - 16; i += 16 )
            {
                v_float32x4 x0 = v_load(srcptr + i);
                v_float32x4 x1 = v_load(srcptr + i + 4);
                v_float32x4 x2 = v_load(srcptr + i + 8);
                v_float32x4 x3 = v_load(srcptr + i + 12);
                x0 = v_select(v_ge(x0, z), x0, v_mul(x0, s4));
                x1 = v_select(v_ge(x1, z), x1, v_mul(x1, s4));
                x2 = v_select(v_ge(x2, z), x2, v_mul(x2, s4));
                x3 = v_select(v_ge(x3, z), x3, v_mul(x3, s4));
                v_store(dstptr + i, x0);
                v_store(dstptr + i + 4, x1);
                v_store(dstptr + i + 8, x2);
                v_store(dstptr + i + 12, x3);
            }
        #endif
            for( ; i < len; i++ )
            {
                float x = srcptr[i];
                dstptr[i] = x >= 0.f ? x : s*x;
            }
        }
    }

#ifdef HAVE_OPENCL
    bool applyOCL(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        if (scale_umat.empty())
            scale.copyTo(scale_umat);

        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);
        String buildopt = oclGetTMacro(inputs[0]);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat& src = inputs[i];
            UMat& dst = outputs[i];

            ocl::Kernel kernel(oclKernelName.c_str(), ocl::dnn::activations_oclsrc, buildopt);
            kernel.set(0, (int)src.total());
            kernel.set(1, (int)src.size[1]);
            kernel.set(2, (int)total(shape(src), 2));
            kernel.set(3, ocl::KernelArg::PtrReadOnly(src));
            kernel.set(4, ocl::KernelArg::PtrWriteOnly(dst));
            kernel.set(5, ocl::KernelArg::PtrReadOnly(scale_umat));

            size_t gSize = src.total();
            CV_Assert(kernel.run(1, &gSize, NULL, false));
        }

        return true;
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ChannelwiseReLUOp>(target, stream, scale);
    }
#endif

#ifdef HAVE_CANN
    Ptr<BackendNode> initCannOp(const std::string& name,
                                const std::vector<Ptr<BackendWrapper> > &inputs,
                                const std::vector<Ptr<BackendNode> >& nodes)
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x_desc = x->getTensorDesc();

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        auto op = std::make_shared<ge::op::PRelu>(name);

        op->set_input_x_by_name(*op_x, x->name.c_str());
        op->update_input_desc_x(*x_desc);

        std::vector<int> shape_{scale.size[0]}; // scale should be a 1d of shape [n] tensor, and it is a 2d mat of shape [n, 1] in opencv
        auto op_const_slope = std::make_shared<CannConstOp>(scale.data, scale.type(), shape_, cv::format("%s_weight", name.c_str()));
        op->set_input_weight(*(op_const_slope->getOp()));
        op->update_input_desc_weight(*(op_const_slope->getTensorDesc()));

        op->update_output_desc_y(*output_desc);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        const size_t numChannels = scale.total();
        auto slope = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{numChannels}, scale.data);
        return std::make_shared<ov::op::v0::PRelu>(node, slope);
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    ml::Operand initWebnnAPI(const ml::GraphBuilder& builder, const ml::Operand& input)
    {
        CV_Error(Error::StsNotImplemented, "");
        ml::Operand operand;
        return operand;
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

struct PReLUFunctor : public ChannelsPReLUFunctor
{
    explicit PReLUFunctor(const Mat& scale_=Mat()) : ChannelsPReLUFunctor(scale_)
    {
#ifdef HAVE_OPENCL
        oclKernelName = "PReLUForward";
#endif
    }

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CANN ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    void apply(const float* srcptr, float* dstptr, int stripeStart, int len, size_t planeSize, int cn0, int cn1) const
    {
        CV_UNUSED(stripeStart);
        CV_Assert(scale.isContinuous() && scale.type() == CV_32F);

        if (stripeStart < 0)
            CV_Error(Error::StsNotImplemented, "PReLUFunctor requires stripe offset parameter");

        const float* scaleptr = scale.ptr<float>() + cn0 * planeSize + stripeStart;
        for( int cn = cn0; cn < cn1; cn++, srcptr += planeSize, dstptr += planeSize, scaleptr += planeSize )
        {
            int i = 0;
        #if CV_SIMD128
            v_float32x4 z = v_setzero_f32();
            for( ; i <= len - 16; i += 16 )
            {
                v_float32x4 x0 = v_load(srcptr + i);
                v_float32x4 x1 = v_load(srcptr + i + 4);
                v_float32x4 x2 = v_load(srcptr + i + 8);
                v_float32x4 x3 = v_load(srcptr + i + 12);
                v_float32x4 s0 = v_load(scaleptr + i);
                v_float32x4 s1 = v_load(scaleptr + i + 4);
                v_float32x4 s2 = v_load(scaleptr + i + 8);
                v_float32x4 s3 = v_load(scaleptr + i + 12);
                x0 = v_select(v_ge(x0, z), x0, v_mul(x0, s0));
                x1 = v_select(v_ge(x1, z), x1, v_mul(x1, s1));
                x2 = v_select(v_ge(x2, z), x2, v_mul(x2, s2));
                x3 = v_select(v_ge(x3, z), x3, v_mul(x3, s3));
                v_store(dstptr + i, x0);
                v_store(dstptr + i + 4, x1);
                v_store(dstptr + i + 8, x2);
                v_store(dstptr + i + 12, x3);
            }
        #endif
            for( ; i < len; i++ )
            {
                float x = srcptr[i];
                float s = scaleptr[i];
                dstptr[i] = x >= 0.f ? x : s*x;
            }
        }
    }

#ifdef HAVE_DNN_NGRAPH
    std::shared_ptr<ov::Node> initNgraphAPI(const ov::Output<ov::Node>& node)
    {
        auto shape = getShape<size_t>(scale);
        auto slope = std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, scale.ptr<float>());
        return std::make_shared<ov::op::v0::PRelu>(node, slope);
    }
#endif  // HAVE_DNN_NGRAPH
};

struct SignFunctor : public BaseDefaultFunctor<SignFunctor>
{
    typedef SignLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::SignOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const SignFunctor::BaseDefaultFunctor<SignFunctor>::ocl_kernel_name = "SignForward";


struct ShrinkFunctor : public BaseDefaultFunctor<ShrinkFunctor>
{
    typedef ShrinkLayer Layer;
    float bias;
    float lambd;

    explicit ShrinkFunctor(float bias_ = 0.0f, float lambd_ = 0.5f) : bias(bias_), lambd(lambd_) {}

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return x > lambd ? x - bias : (x < -lambd ? x + bias : 0.f);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ShrinkOp>(target, stream, bias, lambd);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const ShrinkFunctor::BaseDefaultFunctor<ShrinkFunctor>::ocl_kernel_name = "ShrinkForward";

struct ReciprocalFunctor : public BaseDefaultFunctor<ReciprocalFunctor>
{
    typedef ReciprocalLayer Layer;

    bool supportBackend(int backendId, int)
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    inline float calculate(float x) const
    {
        return 1.f/x;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(int target, csl::Stream stream)
    {
        return make_cuda_node<cuda4dnn::ReciprocalOp>(target, stream);
    }
#endif

    int64 getFLOPSPerElement() const { return 1; }
};

template<>
const char* const ReciprocalFunctor::BaseDefaultFunctor<ReciprocalFunctor>::ocl_kernel_name = "ReciprocalForward";


Ptr<ReLULayer> ReLULayer::create(const LayerParams& params)
{
    float negativeSlope = params.get<float>("negative_slope", 0.f);
    Ptr<ReLULayer> l(new ElementWiseLayer<ReLUFunctor>(ReLUFunctor(negativeSlope)));
    l->setParamsFrom(params);
    l->negativeSlope = negativeSlope;

    return l;
}

Ptr<ReLU6Layer> ReLU6Layer::create(const LayerParams& params)
{
    float minValue = params.get<float>("min_value", 0.0f);
    float maxValue = params.get<float>("max_value", 6.0f);
    Ptr<ReLU6Layer> l(new ElementWiseLayer<ReLU6Functor>(ReLU6Functor(minValue, maxValue)));
    l->setParamsFrom(params);
    l->minValue = minValue;
    l->maxValue = maxValue;

    return l;
}

Ptr<GeluLayer> GeluLayer::create(const LayerParams& params)
{
    Ptr<GeluLayer> l(new ElementWiseLayer<GeluFunctor>(GeluFunctor()));
    l->setParamsFrom(params);

    return l;
}

Ptr<GeluApproximationLayer> GeluApproximationLayer::create(const LayerParams& params)
{
    Ptr<GeluApproximationLayer> l(new ElementWiseLayer<GeluApproximationFunctor>(GeluApproximationFunctor()));
    l->setParamsFrom(params);

    return l;
}

Ptr<TanHLayer> TanHLayer::create(const LayerParams& params)
{
    Ptr<TanHLayer> l(new ElementWiseLayer<TanHFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SwishLayer> SwishLayer::create(const LayerParams& params)
{
    Ptr<SwishLayer> l(new ElementWiseLayer<SwishFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<MishLayer> MishLayer::create(const LayerParams& params)
{
    Ptr<MishLayer> l(new ElementWiseLayer<MishFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SigmoidLayer> SigmoidLayer::create(const LayerParams& params)
{
    Ptr<SigmoidLayer> l(new ElementWiseLayer<SigmoidFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<ELULayer> ELULayer::create(const LayerParams& params)
{
    float alpha = params.get<float>("alpha", 1.0f);
    Ptr<ELULayer> l(new ElementWiseLayer<ELUFunctor>(ELUFunctor(alpha)));
    l->setParamsFrom(params);
    l->alpha = alpha;

    return l;
}

Ptr<AbsLayer> AbsLayer::create(const LayerParams& params)
{
    Ptr<AbsLayer> l(new ElementWiseLayer<AbsValFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<BNLLLayer> BNLLLayer::create(const LayerParams& params)
{
    Ptr<BNLLLayer> l(new ElementWiseLayer<BNLLFunctor>());
    l->setParamsFrom(params);

    return l;
}


Ptr<CeilLayer> CeilLayer::create(const LayerParams& params)
{
    Ptr<CeilLayer> l(new ElementWiseLayer<CeilFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<FloorLayer> FloorLayer::create(const LayerParams& params)
{
    Ptr<FloorLayer> l(new ElementWiseLayer<FloorFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<LogLayer> LogLayer::create(const LayerParams& params)
{
    Ptr<LogLayer> l(new ElementWiseLayer<LogFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<RoundLayer> RoundLayer::create(const LayerParams& params)
{
    Ptr<RoundLayer> l(new ElementWiseLayer<RoundFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SqrtLayer> SqrtLayer::create(const LayerParams& params)
{
    Ptr<SqrtLayer> l(new ElementWiseLayer<SqrtFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<AcosLayer> AcosLayer::create(const LayerParams& params)
{
    Ptr<AcosLayer> l(new ElementWiseLayer<AcosFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<AcoshLayer> AcoshLayer::create(const LayerParams& params)
{
    Ptr<AcoshLayer> l(new ElementWiseLayer<AcoshFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<AsinLayer> AsinLayer::create(const LayerParams& params)
{
    Ptr<AsinLayer> l(new ElementWiseLayer<AsinFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<AsinhLayer> AsinhLayer::create(const LayerParams& params)
{
    Ptr<AsinhLayer> l(new ElementWiseLayer<AsinhFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<AtanLayer> AtanLayer::create(const LayerParams& params)
{
    Ptr<AtanLayer> l(new ElementWiseLayer<AtanFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<AtanhLayer> AtanhLayer::create(const LayerParams& params)
{
    Ptr<AtanhLayer> l(new ElementWiseLayer<AtanhFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<CosLayer> CosLayer::create(const LayerParams& params)
{
    Ptr<CosLayer> l(new ElementWiseLayer<CosFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<CoshLayer> CoshLayer::create(const LayerParams& params)
{
    Ptr<CoshLayer> l(new ElementWiseLayer<CoshFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<ErfLayer> ErfLayer::create(const LayerParams& params)
{
    Ptr<ErfLayer> l(new ElementWiseLayer<ErfFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<HardSwishLayer> HardSwishLayer::create(const LayerParams& params)
{
    Ptr<HardSwishLayer> l(new ElementWiseLayer<HardSwishFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SinLayer> SinLayer::create(const LayerParams& params)
{
    Ptr<SinLayer> l(new ElementWiseLayer<SinFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SinhLayer> SinhLayer::create(const LayerParams& params)
{
    Ptr<SinhLayer> l(new ElementWiseLayer<SinhFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SoftplusLayer> SoftplusLayer::create(const LayerParams& params)
{
    Ptr<SoftplusLayer> l(new ElementWiseLayer<SoftplusFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<SoftsignLayer> SoftsignLayer::create(const LayerParams& params)
{
    Ptr<SoftsignLayer> l(new ElementWiseLayer<SoftsignFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<TanLayer> TanLayer::create(const LayerParams& params)
{
    Ptr<TanLayer> l(new ElementWiseLayer<TanFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<CeluLayer> CeluLayer::create(const LayerParams& params)
{
    float alpha = params.get<float>("alpha", 1.f);
    Ptr<CeluLayer> l(new ElementWiseLayer<CeluFunctor>(CeluFunctor(alpha)));
    l->setParamsFrom(params);
    l->alpha = alpha;

    return l;
}

Ptr<HardSigmoidLayer> HardSigmoidLayer::create(const LayerParams& params)
{
    float alpha = params.get<float>("alpha", 0.2f);
    float beta = params.get<float>("beta", 0.5f);
    Ptr<HardSigmoidLayer> l(new ElementWiseLayer<HardSigmoidFunctor>(HardSigmoidFunctor(alpha, beta)));
    l->setParamsFrom(params);
    l->alpha = alpha;
    l->beta = beta;

    return l;
}

Ptr<SeluLayer> SeluLayer::create(const LayerParams& params)
{
    float alpha = params.get<float>("alpha", 1.67326319217681884765625f);
    float gamma = params.get<float>("gamma", 1.05070102214813232421875f);
    Ptr<SeluLayer> l(new ElementWiseLayer<SeluFunctor>(SeluFunctor(alpha, gamma)));
    l->setParamsFrom(params);
    l->alpha = alpha;
    l->gamma = gamma;

    return l;
}

Ptr<ThresholdedReluLayer> ThresholdedReluLayer::create(const LayerParams& params)
{
    float alpha = params.get<float>("alpha", 1.f);
    Ptr<ThresholdedReluLayer> l(new ElementWiseLayer<ThresholdedReluFunctor>(ThresholdedReluFunctor(alpha)));
    l->setParamsFrom(params);
    l->alpha = alpha;

    return l;
}

Ptr<PowerLayer> PowerLayer::create(const LayerParams& params)
{
    float power = params.get<float>("power", 1.0f);
    float scale = params.get<float>("scale", 1.0f);
    float shift = params.get<float>("shift", 0.0f);
    Ptr<PowerLayer> l(new ElementWiseLayer<PowerFunctor>(PowerFunctor(power, scale, shift)));
    l->setParamsFrom(params);
    l->power = power;
    l->scale = scale;
    l->shift = shift;

    return l;
}

Ptr<ExpLayer> ExpLayer::create(const LayerParams& params)
{
    float base = params.get<float>("base", -1.0f);
    float scale = params.get<float>("scale", 1.0f);
    float shift = params.get<float>("shift", 0.0f);
    Ptr<ExpLayer> l(new ElementWiseLayer<ExpFunctor>(ExpFunctor(base, scale, shift)));
    l->setParamsFrom(params);
    l->base = base;
    l->scale = scale;
    l->shift = shift;

    return l;
}

Ptr<Layer> ChannelsPReLULayer::create(const LayerParams& params)
{
    CV_Assert(params.blobs.size() == 1);
    Mat scale = params.blobs[0];
    float slope = *scale.ptr<float>();
    if (scale.total() == 1 || countNonZero(scale != slope) == 0)
    {
        LayerParams reluParams = params;
        reluParams.set("negative_slope", slope);
        return ReLULayer::create(reluParams);
    }

    Ptr<Layer> l;
    // Check first two dimensions of scale (batch, channels)
    MatShape scaleShape = shape(scale);
    if (std::count_if(scaleShape.begin(), scaleShape.end(), [](int d){ return d != 1;}) > 1)
    {
        l = new ElementWiseLayer<PReLUFunctor>(PReLUFunctor(scale));
    }
    else
    {
        l = new ElementWiseLayer<ChannelsPReLUFunctor>(ChannelsPReLUFunctor(scale));
    }
    l->setParamsFrom(params);

    return l;
}

Ptr<SignLayer> SignLayer::create(const LayerParams& params)
{
    Ptr<SignLayer> l(new ElementWiseLayer<SignFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<ReciprocalLayer> ReciprocalLayer::create(const LayerParams& params)
{
    Ptr<ReciprocalLayer> l(new ElementWiseLayer<ReciprocalFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<ShrinkLayer> ShrinkLayer::create(const LayerParams& params)
{
    float bias = params.get<float>("bias", 0.f);
    float lambd = params.get<float>("lambd", 0.5f);
    Ptr<ShrinkLayer> l(new ElementWiseLayer<ShrinkFunctor>(ShrinkFunctor(bias, lambd)));
    l->setParamsFrom(params);
    l->bias = bias;
    l->lambd = lambd;

    return l;
}
}
}
