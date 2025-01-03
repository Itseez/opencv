// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../op_cann.hpp"
#include <opencv2/imgproc.hpp>

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#include <openvino/op/interpolate.hpp>
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/resize.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {
namespace detail {
void getVector(InputArrayOfArrays images_, std::vector<Mat> &images) {
    images_.getMatVector(images);
}

void getVector(InputArrayOfArrays images_, std::vector<UMat> &images) {
    images_.getUMatVector(images);
}

bool isSame(const cv::UMat &a, const cv::UMat &b) {
    return a.u == b.u;
}

bool isSame(const cv::Mat &a, const cv::Mat &b) {
    return a.data == b.data;
}

cv::Mat getPlanes(cv::Mat &inp, int numPlanes, int height) {
    return inp.reshape(1, numPlanes * height);
}

cv::Mat getPlanes(cv::UMat &inp, int numPlanes, int height) {
    Mat m = inp.getMat(cv::ACCESS_RW);
    return getPlanes(m, numPlanes, height);
}
}
class ResizeLayerImpl : public ResizeLayer
{
public:
    ResizeLayerImpl(const LayerParams& params) : zoomFactorWidth(params.get<float>("zoom_factor_x", params.get<float>("zoom_factor", 0))),
                                                 zoomFactorHeight(params.get<float>("zoom_factor_y", params.get<float>("zoom_factor", 0))),
                                                 scaleWidth(0), scaleHeight(0)
    {
        setParamsFrom(params);
        outWidth = params.get<float>("width", 0);
        outHeight = params.get<float>("height", 0);
        if (params.has("zoom_factor"))
        {
            CV_Assert(!params.has("zoom_factor_x") && !params.has("zoom_factor_y"));
        }
        else if (params.has("zoom_factor_x") || params.has("zoom_factor_y"))
        {
            CV_Assert(params.has("zoom_factor_x") && params.has("zoom_factor_y"));
        }
        interpolation = params.get<String>("interpolation");
        CV_Check(interpolation, interpolation == "nearest" || interpolation == "opencv_linear" || interpolation == "bilinear", "");

        alignCorners = params.get<bool>("align_corners", false);
        halfPixelCenters = params.get<bool>("half_pixel_centers", false);
        if (interpolation == "opencv_linear")
            halfPixelCenters = true;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() == 1 || inputs.size() == 2, inputs[0].size() == 4);
        outputs.resize(1, inputs[0]);
        if (inputs.size() == 1) {
            outputs[0][2] = zoomFactorHeight > 0 ? (outputs[0][2] * zoomFactorHeight) : outHeight;
            outputs[0][3] = zoomFactorWidth > 0 ? (outputs[0][3] * zoomFactorWidth) : outWidth;
        } else {
            CV_CheckGE(inputs[1].size(), (size_t)4, "");
            outputs[0][2] = inputs[1][2];
            outputs[0][3] = inputs[1][3];
        }
        // We can work in-place (do nothing) if input shape == output shape.
        return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
            return interpolation == "nearest" || interpolation == "bilinear" || interpolation == "opencv_linear";

        if (backendId == DNN_BACKEND_CANN)
            return interpolation == "nearest" || interpolation == "bilinear" || interpolation == "opencv_linear";

#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            return (interpolation == "nearest" && scaleWidth == scaleHeight) ||
                   (interpolation == "bilinear");
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        outHeight = outputs[0].size[2];
        outWidth = outputs[0].size[3];
        if (alignCorners && outHeight > 1)
            scaleHeight = static_cast<float>(inputs[0].size[2] - 1) / (outHeight - 1);
        else
            scaleHeight = static_cast<float>(inputs[0].size[2]) / outHeight;

        if (alignCorners && outWidth > 1)
            scaleWidth = static_cast<float>(inputs[0].size[3] - 1) / (outWidth - 1);
        else
            scaleWidth = static_cast<float>(inputs[0].size[3]) / outWidth;
    }


    template <typename Tmat>
    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Tmat> inputs, outputs, internals;
        dnn::detail::getVector(inputs_arr, inputs);
        dnn::detail::getVector(outputs_arr, outputs);
        dnn::detail::getVector(internals, internals);

        if (outHeight == inputs[0].size[2] && outWidth == inputs[0].size[3])
        {
            // outputs[0] = inputs[0] doesn't work due to BlobManager optimizations
            if (!dnn::detail::isSame(inputs[0], outputs[0]))
            {
                inputs[0].copyTo(outputs[0]);
            }
            return;
        }

        Tmat& inp = inputs[0];
        Tmat& out = outputs[0];
        int depth = inp.depth();
        if ((interpolation == "nearest" && !alignCorners && !halfPixelCenters) || (interpolation == "opencv_linear" && depth != CV_8S) ||
            (interpolation == "bilinear" && halfPixelCenters && depth != CV_8S))
        {
            // INTER_LINEAR Resize mode does not support INT8 inputs
            InterpolationFlags mode = interpolation == "nearest" ? INTER_NEAREST : INTER_LINEAR;
            for (size_t n = 0; n < inputs[0].size[0]; ++n)
            {
                for (size_t ch = 0; ch < inputs[0].size[1]; ++ch)
                {
                    Tmat src = getPlane(inp, n, ch);
                    Tmat dst = getPlane(out, n, ch);
                    resize(src, dst, Size(outWidth, outHeight), 0, 0, mode);
                }
            }
        }
        else if (interpolation == "nearest")
        {
            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int inpSpatialSize = inpHeight * inpWidth;
            const int outSpatialSize = outHeight * outWidth;
            const int numPlanes = inp.size[0] * inp.size[1];
            CV_Assert_N(inp.isContinuous(), out.isContinuous());

            Mat inpPlanes = dnn::detail::getPlanes(inp, numPlanes, inpHeight);
            Mat outPlanes = dnn::detail::getPlanes(out, numPlanes, outHeight);

            float heightOffset = 0.0f;
            float widthOffset = 0.0f;

            if (halfPixelCenters)
            {
                heightOffset = 0.5f * scaleHeight;
                widthOffset = 0.5f * scaleWidth;
            }

            if (depth == CV_8S)
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    float input_y = y * scaleHeight + heightOffset;
                    int y0 = halfPixelCenters ? std::floor(input_y) : lroundf(input_y);
                    y0 = std::min(y0, inpHeight - 1);

                    const int8_t* inpData_row = inpPlanes.ptr<int8_t>(y0);

                    for (int x = 0; x < outWidth; ++x)
                    {
                        float input_x = x * scaleWidth + widthOffset;
                        int x0 = halfPixelCenters ? std::floor(input_x) : lroundf(input_x);
                        x0 = std::min(x0, inpWidth - 1);

                        int8_t* outData = outPlanes.ptr<int8_t>(y, x);
                        const int8_t* inpData_row_c = inpData_row;

                        for (int c = 0; c < numPlanes; ++c)
                        {
                            *outData = inpData_row_c[x0];

                            inpData_row_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    float input_y = y * scaleHeight + heightOffset;
                    int y0 = halfPixelCenters ? std::floor(input_y) : lroundf(input_y);
                    y0 = std::min(y0, inpHeight - 1);

                    const float* inpData_row = inpPlanes.ptr<float>(y0);

                    for (int x = 0; x < outWidth; ++x)
                    {
                        float input_x = x * scaleWidth + widthOffset;
                        int x0 = halfPixelCenters ? std::floor(input_x) : lroundf(input_x);
                        x0 = std::min(x0, inpWidth - 1);

                        float* outData = outPlanes.ptr<float>(y, x);
                        const float* inpData_row_c = inpData_row;

                        for (int c = 0; c < numPlanes; ++c)
                        {
                            *outData = inpData_row_c[x0];

                            inpData_row_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
        }
        else if (interpolation == "bilinear" || interpolation == "opencv_linear")
        {
            const int inpHeight = inp.size[2];
            const int inpWidth = inp.size[3];
            const int inpSpatialSize = inpHeight * inpWidth;
            const int outSpatialSize = outHeight * outWidth;
            const int numPlanes = inp.size[0] * inp.size[1];
            CV_Assert_N(inp.isContinuous(), out.isContinuous());

            Mat inpPlanes = dnn::detail::getPlanes(inp, numPlanes, inpHeight);
            Mat outPlanes = dnn::detail::getPlanes(out, numPlanes, outHeight);
            if (depth == CV_8S)
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    float input_y = halfPixelCenters ? std::max((y + 0.5f) * scaleHeight - 0.5f, 0.0f) : y * scaleHeight;
                    int y0 = static_cast<int>(input_y);
                    const int8_t* inpData_row0 = inpPlanes.ptr<int8_t>(y0);
                    const int8_t* inpData_row1 = inpPlanes.ptr<int8_t>(std::min(y0 + 1, inpHeight - 1));
                    for (int x = 0; x < outWidth; ++x)
                    {
                        float input_x = halfPixelCenters ? std::max((x + 0.5f) * scaleWidth - 0.5f, 0.0f) : x * scaleWidth;
                        int x0 = static_cast<int>(input_x);
                        int x1 = std::min(x0 + 1, inpWidth - 1);

                        int8_t* outData = outPlanes.ptr<int8_t>(y, x);
                        const int8_t* inpData_row0_c = inpData_row0;
                        const int8_t* inpData_row1_c = inpData_row1;
                        for (int c = 0; c < numPlanes; ++c)
                        {
                            *outData = static_cast<int8_t>(inpData_row0_c[x0] +
                                (input_y - y0) * (inpData_row1_c[x0] - inpData_row0_c[x0]) +
                                (input_x - x0) * (inpData_row0_c[x1] - inpData_row0_c[x0] +
                                (input_y - y0) * (inpData_row1_c[x1] - inpData_row0_c[x1] - inpData_row1_c[x0] + inpData_row0_c[x0])));

                            inpData_row0_c += inpSpatialSize;
                            inpData_row1_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < outHeight; ++y)
                {
                    float input_y = y * scaleHeight;
                    int y0 = static_cast<int>(input_y);
                    const float* inpData_row0 = inpPlanes.ptr<float>(y0);
                    const float* inpData_row1 = inpPlanes.ptr<float>(std::min(y0 + 1, inpHeight - 1));
                    for (int x = 0; x < outWidth; ++x)
                    {
                        float input_x = x * scaleWidth;
                        int x0 = static_cast<int>(input_x);
                        int x1 = std::min(x0 + 1, inpWidth - 1);

                        float* outData = outPlanes.ptr<float>(y, x);
                        const float* inpData_row0_c = inpData_row0;
                        const float* inpData_row1_c = inpData_row1;
                        for (int c = 0; c < numPlanes; ++c)
                        {
                            *outData = inpData_row0_c[x0] +
                                (input_y - y0) * (inpData_row1_c[x0] - inpData_row0_c[x0]) +
                                (input_x - x0) * (inpData_row0_c[x1] - inpData_row0_c[x0] +
                                (input_y - y0) * (inpData_row1_c[x1] - inpData_row0_c[x1] - inpData_row1_c[x0] + inpData_row0_c[x0]));

                            inpData_row0_c += inpSpatialSize;
                            inpData_row1_c += inpSpatialSize;
                            outData += outSpatialSize;
                        }
                    }
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown interpolation: " + interpolation);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.kind() == _InputArray::STD_VECTOR_UMAT) {
            forward<UMat>(inputs_arr, outputs_arr, internals_arr);
        } else if (inputs_arr.kind() == _InputArray::STD_VECTOR_MAT) {
            forward<Mat>(inputs_arr, outputs_arr, internals_arr);
        }
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto x_desc = x->getTensorDesc();
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto output_y_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        // create operator
        if (interpolation == "nearest")
        {
            auto op = std::make_shared<ge::op::ResizeNearestNeighborV2>(name);

            // set attributes
            op->set_attr_align_corners(alignCorners);
            op->set_attr_half_pixel_centers(halfPixelCenters);

            // set inputs : x
            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);
            // set inputs : size
            std::vector<int> shape_of_size_mat{2};
            std::vector<int> size_vec{outHeight, outWidth};
            Mat size_mat(shape_of_size_mat, CV_32S, size_vec.data());
            auto op_const_size = std::make_shared<CannConstOp>(size_mat.data, size_mat.type(), shape_of_size_mat, cv::format("%s_size", name.c_str()));
            op->set_input_size(*(op_const_size->getOp()));
            op->update_input_desc_size(*(op_const_size->getTensorDesc()));

            // set outputs
            op->update_output_desc_y(*output_y_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }
        else if (interpolation == "opencv_linear" || interpolation == "bilinear")
        {
            auto op = std::make_shared<ge::op::ResizeBilinearV2D>(name);

            // set attributes
            op->set_attr_align_corners(alignCorners);
            op->set_attr_half_pixel_centers(halfPixelCenters);
            std::vector<int64_t> taget_size{(int64_t)outHeight, (int64_t)outWidth};
            op->set_attr_size(taget_size);

            // set inputs : x
            op->set_input_x_by_name(*op_x, x->name.c_str());
            op->update_input_desc_x(*x_desc);

            // set outputs
            op->update_output_desc_y(*output_y_desc);

            return Ptr<BackendNode>(new CannBackendNode(op));
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported interpolation by CANN backend: " + interpolation);
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        ov::op::v4::Interpolate::InterpolateAttrs attrs;

        if (interpolation == "nearest") {
            attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        } else if (interpolation == "bilinear") {
            attrs.mode = ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX;
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        } else {
            CV_Error(Error::StsNotImplemented, format("Unsupported interpolation: %s", interpolation.c_str()));
        }
        attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SIZES;

        CV_Assert(!halfPixelCenters || !alignCorners);
        if (halfPixelCenters) {
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        } else if (alignCorners) {
            attrs.coordinate_transformation_mode = ov::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        }

        attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;


        std::vector<int64_t> shape = {outHeight, outWidth};
        auto out_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, shape.data());

        auto& input_shape = ieInpNode.get_shape();
        CV_Assert_N(input_shape[2] != 0, input_shape[3] != 0);
        std::vector<float> scales = {static_cast<float>(outHeight) / input_shape[2], static_cast<float>(outWidth) / input_shape[3]};
        auto scales_shape = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, scales.data());

        auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
        auto interp = std::make_shared<ov::op::v4::Interpolate>(ieInpNode, out_shape, scales_shape, axes, attrs);
        return Ptr<BackendNode>(new InfEngineNgraphNode(interp));
    }
#endif  // HAVE_DNN_NGRAPH


#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        cuda4dnn::ResizeConfiguration config;
        if (interpolation == "nearest")
        {
            config.type = InterpolationType::NEAREST_NEIGHBOUR;
            config.align_corners = alignCorners;
            config.half_pixel_centers = halfPixelCenters;
        }
        else if (interpolation == "bilinear")
        {
            config.type = InterpolationType::BILINEAR;
            config.align_corners = alignCorners;
            config.half_pixel_centers = halfPixelCenters;
        }
        else if (interpolation == "opencv_linear")
        {
            config.type = InterpolationType::BILINEAR;
            config.align_corners = false;
            config.half_pixel_centers = true;
        }
        else
            CV_Error(Error::StsNotImplemented, "Requested interpolation mode is not available in resize layer.");
        return make_cuda_node<cuda4dnn::ResizeOp>(preferableTarget, std::move(context->stream), config);
    }
#endif

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        return true;
    }

protected:
    int outWidth, outHeight;
    const float zoomFactorWidth, zoomFactorHeight;
    String interpolation;
    float scaleWidth, scaleHeight;
    bool alignCorners;
    bool halfPixelCenters;
};


Ptr<ResizeLayer> ResizeLayer::create(const LayerParams& params)
{
    return Ptr<ResizeLayer>(new ResizeLayerImpl(params));
}

class InterpLayerImpl CV_FINAL : public ResizeLayerImpl
{
public:
    InterpLayerImpl(const LayerParams& params) : ResizeLayerImpl(params) {}

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() == 1, inputs[0].size() == 4);
        outputs.resize(1, inputs[0]);
        outputs[0][2] = zoomFactorHeight > 0 ? (1 + zoomFactorHeight * (outputs[0][2] - 1)) : outHeight;
        outputs[0][3] = zoomFactorWidth > 0 ? (1 + zoomFactorWidth * (outputs[0][3] - 1)) : outWidth;
        // We can work in-place (do nothing) if input shape == output shape.
        return (outputs[0][2] == inputs[0][2]) && (outputs[0][3] == inputs[0][3]);
    }
};

Ptr<Layer> InterpLayer::create(const LayerParams& params)
{
    LayerParams lp(params);
    lp.set("interpolation", "bilinear");
    lp.set("align_corners", true);
    return Ptr<Layer>(new InterpLayerImpl(lp));
}

}  // namespace dnn
}  // namespace cv
