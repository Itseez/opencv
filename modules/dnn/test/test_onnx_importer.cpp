// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <numeric>
namespace opencv_test { namespace {

void yoloPostProcessing(
    std::vector<Mat>& outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold,
    const std::string& model_name,
    const int nc=80);

template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    return findDataFile(std::string("dnn/onnx/") + filename, required);
}

class Test_ONNX_layers : public DNNTestLayer
{
public:
    bool required;

    Test_ONNX_layers() : required(true) { }

    enum Extension
    {
        npy,
        pb
    };

    void testInputShapes(const Net& net, const std::vector<Mat>& inps)
    {
        std::vector<MatShape> inLayerShapes;
        std::vector<MatShape> outLayerShapes;
        std::vector<MatShape> suggestedShapes;
        std::vector<int> suggestedTypes;
        for (const Mat& inp: inps) {
            suggestedShapes.push_back(inp.shape());
            suggestedTypes.push_back(inp.type());
        }
        net.getLayerShapes(suggestedShapes, suggestedTypes, 0, inLayerShapes, outLayerShapes);
        ASSERT_EQ(inLayerShapes.size(), inps.size());

        for (int i = 0; i < inps.size(); ++i) {
            bool hasDynamicShapes = inLayerShapes[i].empty();
            MatShape inpshape_i = inps[i].shape();
            if (hasDynamicShapes)
                continue;
            if (inLayerShapes[i].size() == 0 && inpshape_i.dims == 1) {
                // [TODO] sometimes sample .onnx models from ONNX conformance suit
                // specify scalars as inputs, but we test them using 1D input.
                // the tests need to be adjusted
                continue;
            }
            if (inLayerShapes[i].size() == 1) {  // 1D input
                ASSERT_EQ(shape(inLayerShapes[i][0]), inpshape_i);
            } else {
                // Compare all axes except batch dimension which is variable.
                inLayerShapes[i][0] = inpshape_i[0];
                if (inLayerShapes[i] != inpshape_i) {
                    ASSERT_EQ(inLayerShapes[i], shape(inps[i]));
                }
            }
        }
    }

    void testONNXModels(const String& basename, const Extension ext = npy,
                        double l1 = 0, double lInf = 0, const bool useSoftmax = false,
                        bool checkNoFallbacks = true, int numInps = 1,
                        bool testShapes = true, bool useWinograd = true)
    {
        String onnxmodel = _tf("models/" + basename + ".onnx", required);
        std::vector<Mat> inps(numInps);
        Mat ref;
        if (ext == npy) {
            for (int i = 0; i < numInps; ++i)
                inps[i] = blobFromNPY(_tf("data/input_" + basename + (numInps > 1 ? format("_%d", i) : "") + ".npy"));
            ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));
        }
        else if (ext == pb) {
            for (int i = 0; i < numInps; ++i)
                inps[i] = readTensorFromONNX(_tf("data/input_" + basename + (numInps > 1 ? format("_%d", i) : "") + ".pb"));
            ref = readTensorFromONNX(_tf("data/output_" + basename + ".pb"));
        }
        else
            CV_Error(Error::StsUnsupportedFormat, "Unsupported extension");

        checkBackend(&inps[0], &ref);
        Net net = readNetFromONNX(onnxmodel);
        ASSERT_FALSE(net.empty());

        if (testShapes)
            testInputShapes(net, inps);

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.enableWinograd(useWinograd);

        std::vector<String> inputNames;
        for (int i = 0; i < numInps; ++i)
            inputNames.push_back(format("%d", i));
        net.setInputsNames(inputNames);

        for (int i = 0; i < numInps; ++i)
            net.setInput(inps[i], inputNames[i]);
        Mat out = net.forward("");

        if (useSoftmax)
        {
            LayerParams lp;
            Net netSoftmax;
            netSoftmax.addLayerToPrev("softmaxLayer", "Softmax", lp);
            netSoftmax.setPreferableBackend(DNN_BACKEND_OPENCV);

            netSoftmax.setInput(out);
            out = netSoftmax.forward();

            netSoftmax.setInput(ref);
            ref = netSoftmax.forward();
        }
        if (ref.dims != out.dims) {
            if (ref.dims <= 1)
                ref = ref.reshape(1, out.rows);
            if (out.dims <= 1)
                out = out.reshape(1, ref.rows);
        }
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        {
            l1 = std::max(l1, 1.4e-3);
            lInf = std::max(lInf, 8e-3);
        }

        EXPECT_EQ(ref.shape(), out.shape());
        normAssert(ref, out, basename.c_str(), l1 ? l1 : default_l1, lInf ? lInf : default_lInf);
        if (checkNoFallbacks)
            expectNoFallbacksFromIE(net);
    }
};

TEST_P(Test_ONNX_layers, InstanceNorm)
{
    if (target == DNN_TARGET_MYRIAD)
        testONNXModels("instancenorm", npy, 0, 0, false, false);
    else
        testONNXModels("instancenorm", npy);
}

TEST_P(Test_ONNX_layers, MaxPooling)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("maxpooling", npy, 0, 0, false, false);
}
TEST_P(Test_ONNX_layers, MaxPooling_2)
{
    testONNXModels("two_maxpooling", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testONNXModels("convolution");
    testONNXModels("conv_asymmetric_pads");
}

TEST_P(Test_ONNX_layers, Convolution_variable_weight)
{
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // not supported
    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN); // not supported
    String basename = "conv_variable_w";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    for (int i = 0; i < 2; i++)
    {
        Mat input = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_0.npy"));
        Mat weights = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_1.npy"));
        Mat ref = blobFromNPY(_tf("data/output_" + basename + format("_%d", i) + ".npy"));

        net.setInput(input, "0");
        net.setInput(weights, "1");

        Mat out = net.forward();
        normAssert(ref, out, "", default_l1, default_lInf);
    }
}

TEST_P(Test_ONNX_layers, Convolution_variable_weight_bias)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // openvino/src/plugins/intel_myriad/common/src/ngraph/transformations/extract_dynamic_batch/slice_convolution.cpp:14 Expecting operation v1::GroupConvolution GroupConvolution_6904725 (Reshape_17[0]:f32{1,4,5,5}, Reshape_6904719[0]:f32{4,1,1,2,2}) -> (f32{1,4,4,4}) to have constant kernel, got Reshape_6904719[0]:f32{4,1,1,2,2}
    // openvino\src\plugins\intel_myriad\common\src\ngraph\transformations\extract_dynamic_batch\slice_convolution.cpp:15 Expecting operation v1::GroupConvolution GroupConvolution_6904692 (Reshape_17[0]:f32{1,4,5,5}, Reshape_6904686[0]:f32{4,1,1,2,2}) -> (f32{1,4,4,4}) to have constant kernel, got Reshape_6904686[0]:f32{4,1,1,2,2}
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // accuracy (depends on OpenCL version / HW)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU &&
        getInferenceEngineCPUType() == CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_ARM_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // supports only <= 2 inputs

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN); // not supported

    String basename = "conv_variable_wb";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    for (int i = 0; i < 2; i++)
    {
        Mat input = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_0.npy"));
        Mat weights = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_1.npy"));
        Mat bias = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_2.npy"));
        Mat ref = blobFromNPY(_tf("data/output_" + basename + format("_%d", i) + ".npy"));

        net.setInput(input, "0");
        net.setInput(weights, "1");
        net.setInput(bias, "bias");

        Mat out = net.forward();
        normAssert(ref, out, "", default_l1, default_lInf);
    }
}

TEST_P(Test_ONNX_layers, Gather)
{
    testONNXModels("gather", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Gather_Scalar)
{
    testONNXModels("gather_scalar", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, GatherMulti)
{
    // GPU plugin unsupported slice for constant
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("gather_multi", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Gather_shared_indices) {
    testONNXModels("gather_shared_indices", npy, 0, 0, false, false, 1);
}

TEST_P(Test_ONNX_layers, Two_resizes_with_shared_subgraphs) {
    testONNXModels("two_resizes_with_shared_subgraphs", npy, 0, 0, false, false, 3, /*testShapes*/ false);
}

TEST_P(Test_ONNX_layers, Convolution3D)
{
    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
    {
        // CUDA_FP16: cuDNN did not return a suitable algorithm for convolution.
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    }
    testONNXModels("conv3d");
}

TEST_P(Test_ONNX_layers, Convolution3D_bias)
{
    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
    {
        // CUDA_FP16: cuDNN did not return a suitable algorithm for convolution.
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    }
    testONNXModels("conv3d_bias");
    testONNXModels("conv3d_depthwise_bias"); // kernel 1x1
}

TEST_P(Test_ONNX_layers, Two_convolution)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    // Reference output values are in range [-0.855, 0.611]
    testONNXModels("two_convolution");
}

TEST_P(Test_ONNX_layers, Deconvolution)
{
    testONNXModels("deconvolution", npy, 0, 0, false, false);
    testONNXModels("two_deconvolution", npy, 0, 0, false, false);
    testONNXModels("deconvolution_group", npy, 0, 0, false, false);
    testONNXModels("deconvolution_output_shape", npy, 0, 0, false, false);
    if (target != DNN_TARGET_CUDA_FP16) // bug
        testONNXModels("deconv_adjpad_2d", npy, 0, 0, false, false);
}

// BUG: https://github.com/opencv/opencv/issues/26307
TEST_P(Test_ONNX_layers, DISABLED_Deconvolution3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/frontend/frontend.cpp:592 Failed to compile layer "2":
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 2@weights Const data got different desc and content byte sizes (162 and 486 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/frontend/frontend.cpp:439 Failed to compile layer "2":
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 2@weights Const data got different desc and content byte sizes (162 and 486 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif

    if (backend == DNN_BACKEND_OPENCV)
        throw SkipTestException("OpenCV backend is not supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    testONNXModels("deconv3d");
}

// BUG: https://github.com/opencv/opencv/issues/26307
TEST_P(Test_ONNX_layers, DISABLED_Deconvolution3D_bias)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/frontend/frontend.cpp:592 Failed to compile layer "3":
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 3@weights Const data got different desc and content byte sizes (270 and 810 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/frontend/frontend.cpp:439 Failed to compile layer "2":
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 2@weights Const data got different desc and content byte sizes (162 and 486 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif

    if (backend == DNN_BACKEND_OPENCV)
        throw SkipTestException("OpenCV backend is not supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    testONNXModels("deconv3d_bias");
}

// BUG: https://github.com/opencv/opencv/issues/26307
TEST_P(Test_ONNX_layers, DISABLED_Deconvolution3D_pad)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/frontend/frontend.cpp:592 Failed to compile layer "3":
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 3@weights Const data got different desc and content byte sizes (108 and 432 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/frontend/frontend.cpp:439 Failed to compile layer "2":
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 2@weights Const data got different desc and content byte sizes (162 and 486 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif

    //if (backend == DNN_BACKEND_OPENCV)
        throw SkipTestException("OpenCV backend is not supported");  // FIXIT use tags

    //if (backend == DNN_BACKEND_VKCOM)
    //    applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    //testONNXModels("deconv3d_pad");
}

// BUG: https://github.com/opencv/opencv/issues/26307
TEST_P(Test_ONNX_layers, DISABLED_Deconvolution3D_adjpad)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/frontend/frontend.cpp:592 Failed to compile layer "3":
        // [ GENERAL_ERROR ] openvino/src/plugins/intel_myriad/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 3@weights Const data got different desc and content byte sizes (90 and 180 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/frontend/frontend.cpp:439 Failed to compile layer "2":
        // [ GENERAL_ERROR ] vpu/graph_transformer/src/model/model.cpp:198 duplicateData error: while duplicating 2@weights Const data got different desc and content byte sizes (162 and 486 respectively)
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif

    if (backend == DNN_BACKEND_OPENCV)
        throw SkipTestException("OpenCV backend is not supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    testONNXModels("deconv3d_adjpad");
}

TEST_P(Test_ONNX_layers, Dropout)
{
    testONNXModels("dropout");
}

TEST_P(Test_ONNX_layers, Linear)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    testONNXModels("linear");
}

TEST_P(Test_ONNX_layers, ReLU)
{
    testONNXModels("ReLU");
}

TEST_P(Test_ONNX_layers, PReLU)
{
    testONNXModels("PReLU_slope");
}

TEST_P(Test_ONNX_layers, Clip)
{
    testONNXModels("clip", npy);
}

TEST_P(Test_ONNX_layers, Clip_init)
{
    testONNXModels("clip_init_min_max");
    testONNXModels("clip_init_min");
    testONNXModels("clip_init_max");
}

TEST_P(Test_ONNX_layers, Shape)
{
    testONNXModels("shape_of_constant");
}

TEST_P(Test_ONNX_layers, ReduceMean)
{
    testONNXModels("reduce_mean");
    testONNXModels("reduce_mean_axis1");
    testONNXModels("reduce_mean_axis2");
}

TEST_P(Test_ONNX_layers, ReduceSum)
{
    testONNXModels("reduce_sum");
    testONNXModels("reduce_sum_axis_dynamic_batch");
}

TEST_P(Test_ONNX_layers, ReduceMax)
{
    testONNXModels("reduce_max");
}
TEST_P(Test_ONNX_layers, ReduceMax_axis_0)
{
    testONNXModels("reduce_max_axis_0");
}
TEST_P(Test_ONNX_layers, ReduceMax_axis_1)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // [ GENERAL_ERROR ]  AssertionFailed: !out.networkInputs.empty()
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("reduce_max_axis_1");
}

TEST_P(Test_ONNX_layers, Min)
{
    testONNXModels("min", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, ArgLayer)
{
    if (backend != DNN_BACKEND_OPENCV || target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");  // FIXIT use tags

    testONNXModels("argmax");
    testONNXModels("argmin");
}

TEST_P(Test_ONNX_layers, Scale)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy (inf/nan)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // IE exception: mkldnn_node.cpp:238 Ngraph operation Reshape with name ReduceMean_0 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Ngraph operation Reshape with name ReduceMean_0 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("scale");
}

TEST_P(Test_ONNX_layers, Scale_broadcast)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // doesn't support broadcasting
    testONNXModels("scale_broadcast", npy, 0, 0, false, true, 3);
}

TEST_P(Test_ONNX_layers, Scale_broadcast_mid)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // doesn't support broadcasting
    testONNXModels("scale_broadcast_mid", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, ReduceMean3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
#endif
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    testONNXModels("reduce_mean3d");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testONNXModels("maxpooling_sigmoid");
}

TEST_P(Test_ONNX_layers, Cast)
{
    testONNXModels("cast");
}

TEST_P(Test_ONNX_layers, Power)
{
    testONNXModels("pow2", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Exp)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("exp");
}

TEST_P(Test_ONNX_layers, Elementwise_Ceil)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("ceil");
}

TEST_P(Test_ONNX_layers, Elementwise_Floor)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("floor");
}

TEST_P(Test_ONNX_layers, Elementwise_Log)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("log");
}

TEST_P(Test_ONNX_layers, Elementwise_Round)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("round");
}

TEST_P(Test_ONNX_layers, Elementwise_Sqrt)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("sqrt");
#endif
}

TEST_P(Test_ONNX_layers, Elementwise_not)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("not");
}

TEST_P(Test_ONNX_layers, Compare_EQ)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    testONNXModels("equal");
}

TEST_P(Test_ONNX_layers, Compare_GT)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    testONNXModels("greater");
}

TEST_P(Test_ONNX_layers, Compare_LT)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    testONNXModels("less");
}

TEST_P(Test_ONNX_layers, Compare_GTorEQ)
{
    testONNXModels("greater_or_equal");
}

TEST_P(Test_ONNX_layers, Compare_LEorEQ)
{
    testONNXModels("less_or_equal");
}

TEST_P(Test_ONNX_layers, CompareSameDims_EQ)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    testONNXModels("equal_same_dims", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, CompareSameDims_GT)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    testONNXModels("greater_same_dims", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, CompareSameDims_LT)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    // IE exception: Function contains several inputs and outputs with one friendly name!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    testONNXModels("less_same_dims", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, Concatenation)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    testONNXModels("concatenation");
    testONNXModels("concat_const_blobs");
}

TEST_P(Test_ONNX_layers, CumSumExclusiveInplace)
{
    testONNXModels("cumsum_exclusive_inplace");
}

TEST_P(Test_ONNX_layers, RangeFloat)
{
    testONNXModels("range_float");
    testONNXModels("range_float_negative");
}

TEST_P(Test_ONNX_layers, RangeInt32)
{
    testONNXModels("range_int32");
    testONNXModels("range_int32_negative");
}

TEST_P(Test_ONNX_layers, RangeInt64)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // OpenVINO uses int32 precision for int64 operations
    testONNXModels("range_int64");
    testONNXModels("range_int64_negative");
}

TEST_P(Test_ONNX_layers, Eltwise3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
#endif
    testONNXModels("eltwise3d");
}

TEST_P(Test_ONNX_layers, AveragePooling)
{
    testONNXModels("average_pooling");
}

TEST_P(Test_ONNX_layers, MaxPooling3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // accuracy
        if (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16)
            applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
            );
        // IE exception: [ GENERAL_ERROR ]  AssertionFailed: !expired()
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // accuracy
        if (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16)
            applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
            );
        // IE exception: [ GENERAL_ERROR ]  AssertionFailed: !expired()
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
#endif
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    testONNXModels("max_pool3d", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, AvePooling3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
#endif
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    testONNXModels("ave_pool3d");
}

TEST_P(Test_ONNX_layers, PoolConv3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
#endif
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
    {
        // CUDA_FP16: cuDNN did not return a suitable algorithm for convolution.
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    }

    testONNXModels("pool_conv_3d");
}

TEST_P(Test_ONNX_layers, BatchNormalization)
{
    testONNXModels("batch_norm");
}

TEST_P(Test_ONNX_layers, BatchNormalization3D)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    testONNXModels("batch_norm_3d");
}

TEST_P(Test_ONNX_layers, BatchNormalizationUnfused)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
#endif
    testONNXModels("frozenBatchNorm2d");
}

TEST_P(Test_ONNX_layers, BatchNormalizationSubgraph)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
#endif
    testONNXModels("batch_norm_subgraph");
}

TEST_P(Test_ONNX_layers, NormalizeFusionSubgraph)
{
    testONNXModels("normalize_fusion");
}

TEST_P(Test_ONNX_layers, Transpose)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    testONNXModels("transpose");
}

TEST_P(Test_ONNX_layers, Multiplication)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("mul");
}

TEST_P(Test_ONNX_layers, MatMul_2d)
{
    testONNXModels("matmul_2d");
}
TEST_P(Test_ONNX_layers, MatMul_3d)
{
    testONNXModels("matmul_3d");
}
TEST_P(Test_ONNX_layers, MatMul_4d)
{
    testONNXModels("matmul_4d");
}

TEST_P(Test_ONNX_layers, MatMul_2d_init)
{
    testONNXModels("matmul_2d_init");
}
TEST_P(Test_ONNX_layers, MatMul_3d_init)
{
    testONNXModels("matmul_3d_init");
}
TEST_P(Test_ONNX_layers, MatMul_4d_init)
{
    testONNXModels("matmul_4d_init");
}
TEST_P(Test_ONNX_layers, MatMul_init_2)
{
    testONNXModels("matmul_init_2");
}
TEST_P(Test_ONNX_layers, MatMul_init_bcast)
{
    testONNXModels("matmul_init_bcast");
}

TEST_P(Test_ONNX_layers, MatMul_bcast_3dx2d) {
    testONNXModels("matmul_bcast");
}

TEST_P(Test_ONNX_layers, MatMulAdd)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    testONNXModels("matmul_add");
}

TEST_P(Test_ONNX_layers, Expand)
{
    testONNXModels("expand");
}

TEST_P(Test_ONNX_layers, ExpandIdentity) {
    testONNXModels("expand_identity");
}

TEST_P(Test_ONNX_layers, ExpandBatch) {
    testONNXModels("expand_batch");
}

TEST_P(Test_ONNX_layers, ExpandChannels) {
    testONNXModels("expand_channels");
}

TEST_P(Test_ONNX_layers, ExpandNegBatch) {
    testONNXModels("expand_neg_batch");
}

TEST_P(Test_ONNX_layers, ExpandHW)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("expand_hw");
}

TEST_P(Test_ONNX_layers, Constant)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
       applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("constant");
}

TEST_P(Test_ONNX_layers, Padding)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2019010000)
    testONNXModels("padding", npy, 0, 0, false, false);
#else
    testONNXModels("padding");
#endif
}

TEST_P(Test_ONNX_layers, Resize)
{
    testONNXModels("resize_nearest");
    testONNXModels("tf_half_pixel_for_nn");
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("resize_bilinear");
}

TEST_P(Test_ONNX_layers, ResizeUnfused)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("upsample_unfused_torch1.2");
    testONNXModels("upsample_unfused_opset9_torch1.4");
    testONNXModels("resize_nearest_unfused_opset11_torch1.4");
    testONNXModels("resize_nearest_unfused_opset11_torch1.3");
    testONNXModels("resize_bilinear_unfused_opset11_torch1.4");
}

TEST_P(Test_ONNX_layers, ResizeUnfusedTwoInputs)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2023000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("upsample_unfused_two_inputs_opset9_torch1.4", npy, 0, 0, false, true, 2);
    // BUG: https://github.com/opencv/opencv/issues/26291
    // testONNXModels("upsample_unfused_two_inputs_opset11_torch1.4", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, MultyInputs)
{
    testONNXModels("multy_inputs", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, Broadcast)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("channel_broadcast", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, DynamicResize)
{
    testONNXModels("dynamic_resize_9", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_10", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_11", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_13", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_scale_9", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_scale_10", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_scale_11", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_scale_13", npy, 0, 0, false, true, 2);

    testONNXModels("resize_size_opset11");
    testONNXModels("resize_size_opset13");
}

TEST_P(Test_ONNX_layers, Resize_HumanSeg)
{
    testONNXModels("resize_humanseg");
}

TEST_P(Test_ONNX_layers, Div)
{
    const String model =  _tf("models/div.onnx");
    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    // Reference output values range is -68.80928, 2.991873. So to avoid computational
    // difference for FP16 we'll perform reversed division (just swap inputs).
    Mat inp1 = blobFromNPY(_tf("data/input_div_1.npy"));
    Mat inp2 = blobFromNPY(_tf("data/input_div_0.npy"));
    Mat ref  = blobFromNPY(_tf("data/output_div.npy"));
    cv::divide(1.0, ref, ref);
    checkBackend(&inp1, &ref);

    net.setInput(inp1, "0");
    net.setInput(inp2, "1");
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);

    // NaryEltwise layer suuports only CPU for now
    testONNXModels("div_test_1x1", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, DynamicReshape)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    testONNXModels("dynamic_reshape");
    testONNXModels("dynamic_reshape_opset_11");
    testONNXModels("flatten_by_prod");
    testONNXModels("flatten_const");
}

TEST_P(Test_ONNX_layers, Reshape)
{
    testONNXModels("unsqueeze");
    testONNXModels("unsqueeze_opset_13");
}

TEST_P(Test_ONNX_layers, Unsqueeze_Neg_Axes)
{
    testONNXModels("unsqueeze_neg_axes");
}

TEST_P(Test_ONNX_layers, Squeeze)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    testONNXModels("squeeze");
    testONNXModels("squeeze_axes_op13");
}

TEST_P(Test_ONNX_layers, ReduceL2)
{
    testONNXModels("reduceL2");
    testONNXModels("reduceL2_subgraph");
    testONNXModels("reduceL2_subgraph_2");
    testONNXModels("reduceL2_subgraph2_2");
}

TEST_P(Test_ONNX_layers, Split)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2023000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("split_0");
    testONNXModels("split_1");
    testONNXModels("split_2");
    testONNXModels("split_3");
    testONNXModels("split_4");
    testONNXModels("split_5");
    testONNXModels("split_6");
    testONNXModels("split_neg_axis");
}

// Mul inside with 0-d tensor, output should be A x 1, but is 1 x A. PR #22652
TEST_P(Test_ONNX_layers, DISABLED_Split_sizes_0d)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("split_sizes");
}

TEST_P(Test_ONNX_layers, Slice)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2019010000)
    testONNXModels("slice", npy, 0, 0, false, false);
#else
    testONNXModels("slice");
    testONNXModels("slice_neg_starts");
    testONNXModels("slice_opset_11");
    testONNXModels("slice_neg_steps", pb);
#endif
}

TEST_P(Test_ONNX_layers, Slice_Steps_2DInput)
{
    testONNXModels("slice_opset_11_steps_2d");
}

TEST_P(Test_ONNX_layers, Slice_Steps_3DInput)
{
    testONNXModels("slice_opset_11_steps_3d");
}

TEST_P(Test_ONNX_layers, Slice_Steps_4DInput)
{
    testONNXModels("slice_opset_11_steps_4d");
}

TEST_P(Test_ONNX_layers, Slice_Steps_5DInput)
{
    testONNXModels("slice_opset_11_steps_5d");
}

TEST_P(Test_ONNX_layers, Slice_Nonseq_Axes)
{
    testONNXModels("slice_nonseq_axes");
    testONNXModels("slice_nonseq_axes_steps");
    testONNXModels("slice_nonseq_miss_axes_steps");
}

TEST_P(Test_ONNX_layers, Slice_Neg_Axes)
{
    testONNXModels("slice_neg_axes");
    testONNXModels("slice_neg_axes_steps");
    testONNXModels("slice_neg_miss_axes_steps");
}

TEST_P(Test_ONNX_layers, Softmax)
{
    testONNXModels("softmax");
    testONNXModels("log_softmax", npy, 0, 0, false, false);
    testONNXModels("softmax_unfused");
}

TEST_P(Test_ONNX_layers, Split_EltwiseMax)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2023000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    testONNXModels("split_max");
}
// Fails with the new engine. Output shape [?, N, OutputSize] not supported by new graph engine
TEST_P(Test_ONNX_layers, LSTM_Activations)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // TODO: fix this test for OpenVINO

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Node Block1326/lstm/reshape_0/permute was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name Block1237_Output_0_before_reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif

    testONNXModels("lstm_cntk_tanh", pb, 0, 0, false, false);
}

// disabled due to poor handling of 1-d mats
TEST_P(Test_ONNX_layers, DISABLED_LSTM)
{
    testONNXModels("lstm", npy, 0, 0, false, false);
}

// disabled due to poor handling of 1-d mats
TEST_P(Test_ONNX_layers, DISABLED_LSTM_bidirectional)
{
    testONNXModels("lstm_bidirectional", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_hidden)
{
    testONNXModels("hidden_lstm", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_hidden_bidirectional)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Node Transpose_45 was not assigned on any pointed device.
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif

    testONNXModels("hidden_lstm_bi", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, GRU)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Node GRU_22 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif
    testONNXModels("gru", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, gru_cell_batchsize_50_seqlen_1)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Node GRU_22 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("gru_cell_batchsize_50_seqlen_1", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, gru_cell_batchsize_5_seqlen_5)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Node GRU_22 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("gru_cell_batchsize_5_seqlen_5", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, gru_cell_batchsize_1_seqlen_50)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Node GRU_22 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("gru_cell_batchsize_1_seqlen_50", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, GRU_bidirectional)
{
    testONNXModels("gru_bi", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_cell_forward)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Ngraph operation Reshape with name LSTM_16/lstm_y/reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("lstm_cell_forward", npy, 0, 0, false, false);
}
TEST_P(Test_ONNX_layers, LSTM_cell_bidirectional)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Ngraph operation Reshape with name LSTM_16/lstm_y/reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("lstm_cell_bidirectional", npy, 0, 0, false, false);
}
TEST_P(Test_ONNX_layers, LSTM_cell_with_peepholes)
{
    testONNXModels("lstm_cell_with_peepholes", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_cell_batchsize_50_seqlen_1)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lstm_cell_batchsize_50_seqlen_1", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_cell_batchsize_1_seqlen_50)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lstm_cell_batchsize_1_seqlen_50", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_cell_batchsize_5_seqlen_5)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lstm_cell_batchsize_5_seqlen_5", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_init_h0_c0)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lstm_init_h0_c0", npy, 0, 0, false, false, 3);
}

// epsilon is larger because onnx does not match with torch/opencv exactly
// Test uses incorrect ONNX and test data with 3 dims instead of 4.
// ONNNRuntime does not support layout=1 attiribute inference. See a detailed issue #26456
TEST_P(Test_ONNX_layers, DISABLED_LSTM_layout_seq)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lstm_layout_0", npy, 0.005, 0.005, false, false, 3);
}

// epsilon is larger because onnx does not match with torch/opencv exactly
// Test uses incorrect ONNX and test data with 3 dims instead of 4.
// ONNNRuntime does not support layout=1 attiribute inference. See a detailed issue #26456
TEST_P(Test_ONNX_layers, DISABLED_LSTM_layout_batch)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lstm_layout_1", npy, 0.005, 0.005, false, false, 3);
}

TEST_P(Test_ONNX_layers, Einsum_1D)
{
    testONNXModels("einsum_1d", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_2D)
{
    testONNXModels("einsum_2d", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_2D_Ellipses)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("einsum_2d_ellipses", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_3D)
{
    testONNXModels("einsum_3d", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_4D)
{
    testONNXModels("einsum_4d", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_5D)
{
    testONNXModels("einsum_5d", npy, 0, 0, false, false, 2);
}

// https://github.com/opencv/opencv/issues/24883
TEST_P(Test_ONNX_layers, Einsum_InnerProduct)
{
    testONNXModels("einsum_inner", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_HadamardProduct)
{
    testONNXModels("einsum_hadamard", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, Einsum_Batch_Diagonal)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("einsum_batch_diagonal", npy, 0, 0, false, false, 1);
}

TEST_P(Test_ONNX_layers, Einsum_Sum)
{
    testONNXModels("einsum_sum", npy, 0, 0, false, false, 1);
}

TEST_P(Test_ONNX_layers, Einsum_transpose)
{
    testONNXModels("einsum_transpose", npy, 0, 0, false, false, 1);
}

TEST_P(Test_ONNX_layers, Einsum_const_inputs) {
    testONNXModels("einsum_const_inputs", npy, 0, 0, false, false, 1);
}

TEST_P(Test_ONNX_layers, ReduceSum_Consts){
    testONNXModels("reducesum_consts");
}

TEST_P(Test_ONNX_layers, Pad2d_Unfused)
{
    testONNXModels("ReflectionPad2d");
    testONNXModels("ZeroPad2d");
}

TEST_P(Test_ONNX_layers, LinearWithConstant)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2020040000)
    applyTestTag(CV_TEST_TAG_DNN_SKIP_IE);
#endif
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    testONNXModels("lin_with_constant");
}

TEST_P(Test_ONNX_layers, MatmulWithTwoInputs)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2020040000)
    applyTestTag(CV_TEST_TAG_DNN_SKIP_IE);
#endif
    testONNXModels("matmul_with_two_inputs");
}

TEST_P(Test_ONNX_layers, ResizeOpset11_Torch1_6)
{
    testONNXModels("resize_opset11_torch1.6");
}

TEST_P(Test_ONNX_layers, Mish)
{
    testONNXModels("mish");
    testONNXModels("mish_no_softplus");
}

TEST_P(Test_ONNX_layers, CalculatePads)
{
    testONNXModels("calc_pads");
}

TEST_P(Test_ONNX_layers, Conv1d)
{
    testONNXModels("conv1d");
}

TEST_P(Test_ONNX_layers, Conv1d_bias)
{
    testONNXModels("conv1d_bias");
}

TEST_P(Test_ONNX_layers, Conv1d_variable_weight)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // not supported
    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN); // not supported
    String basename = "conv1d_variable_w";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    Mat weights = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    net.setInput(input, "0");
    net.setInput(weights, "1");

    Mat out = net.forward();
    normAssert(ref, out, "", default_l1, default_lInf);
}

TEST_P(Test_ONNX_layers, Conv1d_variable_weight_bias)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // not supported
    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN); // not supported
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_CPU && getInferenceEngineCPUType() == CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_ARM_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
    String basename = "conv1d_variable_wb";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    Mat weights = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    Mat bias = blobFromNPY(_tf("data/input_" + basename + "_2.npy"));
    Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    net.setInput(input, "0");
    net.setInput(weights, "1");
    net.setInput(bias, "bias");

    Mat out = net.forward();
    normAssert(ref, out, "", default_l1, default_lInf);
}

TEST_P(Test_ONNX_layers, GatherMultiOutput)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name 6 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
#endif

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LE(2021030000)
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE);
#endif

    testONNXModels("gather_multi_output", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, DynamicAxes_squeeze_and_conv)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("squeeze_and_conv_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_unsqueeze_and_conv)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("unsqueeze_and_conv_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_gather)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("gather_dynamic_axes", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, DynamicAxes_gather_scalar)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("gather_scalar_dynamic_axes", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, DynamicAxes_slice)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("slice_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_slice_opset_11)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("slice_opset_11_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_resize_opset11_torch16)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("resize_opset11_torch1.6_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_average_pooling)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("average_pooling_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_maxpooling_sigmoid)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("maxpooling_sigmoid_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_dynamic_batch)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#if INF_ENGINE_VER_MAJOR_LT(2021000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#endif
    testONNXModels("dynamic_batch");
}


TEST_P(Test_ONNX_layers, MaxPool1d)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
    {
        // 2021.4: [ GENERAL_ERROR ]  AssertionFailed: !expired()
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
    testONNXModels("maxpooling_1d");
}

TEST_P(Test_ONNX_layers, MaxPoolSigmoid1d)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
    testONNXModels("maxpooling_sigmoid_1d");
}

TEST_P(Test_ONNX_layers, MaxPool1d_Twise)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
    testONNXModels("two_maxpooling_1d");
}

TEST_P(Test_ONNX_layers, AvePool1d)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
    testONNXModels("average_pooling_1d");
}

TEST_P(Test_ONNX_layers, PoolConv1d)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }
#endif
    testONNXModels("pool_conv_1d");
}

TEST_P(Test_ONNX_layers, ConvResizePool1d)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name 15 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_MYRIAD) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#if INF_ENGINE_VER_MAJOR_EQ(2021030000)
        if (target == DNN_TARGET_OPENCL) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
#endif
    }
#endif

    const double lInf = (target == DNN_TARGET_CPU_FP16) ? 0.024 : default_lInf;
    testONNXModels("conv_resize_pool_1d", npy, default_l1, lInf);
}

TEST_P(Test_ONNX_layers, DepthWiseAdd)
{
    testONNXModels("depthwiseconv_add");
}

TEST_P(Test_ONNX_layers, DepthStride2)
{
    testONNXModels("depthwise_stride2");
}

TEST_P(Test_ONNX_layers, SubFromConst)
{
    testONNXModels("sub_from_const1");
    testONNXModels("sub_from_const_eltwise");
    testONNXModels("sub_from_const_broadcast");
}

TEST_P(Test_ONNX_layers, DivConst)
{
    testONNXModels("div_const");
}

TEST_P(Test_ONNX_layers, Gemm)
{
    testONNXModels("gemm_no_transB");
    testONNXModels("gemm_transB_0");
    testONNXModels("gemm_first_const");
}

TEST_P(Test_ONNX_layers, Gemm_bias)
{
    testONNXModels("gemm_vector_bias");
}

TEST_P(Test_ONNX_layers, Quantized_Convolution)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // TODO: fix this test for OpenVINO

    // The difference of QOperator and QDQ format:
    // https://onnxruntime.ai/docs/performance/quantization.html#onnx-quantization-representation-format.
    {
        SCOPED_TRACE("QOperator quantized model.");
        testONNXModels("quantized_conv_uint8_weights", npy, 0.004, 0.02);
        testONNXModels("quantized_conv_int8_weights", npy, 0.03, 0.5);
        testONNXModels("quantized_conv_per_channel_weights", npy, 0.06, 0.4);
        testONNXModels("quantized_conv_asymmetric_pads_int8_weights");
    }

    {
        SCOPED_TRACE("QDQ quantized model.");
        testONNXModels("quantized_conv_uint8_weights_qdq", npy, 0.004, 0.02);
        testONNXModels("quantized_conv_int8_weights_qdq", npy, 0.03, 0.5);
        testONNXModels("quantized_conv_per_channel_weights_qdq", npy, 0.06, 0.4);
    }
}

TEST_P(Test_ONNX_layers, Quantized_MatMul)
{
    testONNXModels("quantized_matmul_uint8_weights", npy, 0.008, 0.015);
    testONNXModels("quantized_matmul_int8_weights", npy, 0.06, 0.2);
    testONNXModels("quantized_matmul_per_channel_weights", npy, 0.06, 0.22);
}

TEST_P(Test_ONNX_layers, Quantized_Gemm)
{
    testONNXModels("quantized_gemm", npy);
}

TEST_P(Test_ONNX_layers, Quantized_MatMul_Variable_Weights)
{
    // Unsupported
    EXPECT_THROW(
    {
        testONNXModels("quantized_matmul_variable_inputs");
    }, cv::Exception);
}

TEST_P(Test_ONNX_layers, Quantized_Eltwise)
{
    testONNXModels("quantized_eltwise");
}

TEST_P(Test_ONNX_layers, Quantized_Eltwise_Scalar)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // TODO: fix this test for OpenVINO
    testONNXModels("quantized_eltwise_scalar");
}

TEST_P(Test_ONNX_layers, Quantized_Eltwise_Broadcast)
{
    testONNXModels("quantized_eltwise_broadcast");
}

TEST_P(Test_ONNX_layers, Quantized_LeakyReLU)
{
    testONNXModels("quantized_leaky_relu");
}

TEST_P(Test_ONNX_layers, Quantized_Sigmoid)
{
    testONNXModels("quantized_sigmoid");
}

TEST_P(Test_ONNX_layers, Quantized_MaxPool)
{
    testONNXModels("quantized_maxpool");
}

TEST_P(Test_ONNX_layers, Quantized_AvgPool)
{
    testONNXModels("quantized_avgpool");
}

TEST_P(Test_ONNX_layers, Quantized_Split)
{
    testONNXModels("quantized_split");
}

TEST_P(Test_ONNX_layers, Quantized_Pad)
{
    testONNXModels("quantized_padding");
}

TEST_P(Test_ONNX_layers, Quantized_Reshape)
{
    testONNXModels("quantized_reshape");
}

TEST_P(Test_ONNX_layers, Quantized_Transpose)
{
    testONNXModels("quantized_transpose");
}

TEST_P(Test_ONNX_layers, Quantized_Squeeze)
{
    testONNXModels("quantized_squeeze");
}

TEST_P(Test_ONNX_layers, Quantized_Unsqueeze)
{
    testONNXModels("quantized_unsqueeze");
}

TEST_P(Test_ONNX_layers, Quantized_Resize)
{
    testONNXModels("quantized_resize_nearest");
    double l1 = backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ? 0.0013 : 2e-4;
    testONNXModels("quantized_resize_bilinear", npy, l1, 0.003);
    l1 = backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ? 0.0013 : 3e-4;
    testONNXModels("quantized_resize_bilinear_align", npy, l1, 0.003);
}

TEST_P(Test_ONNX_layers, Quantized_Concat)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("quantized_concat");
    testONNXModels("quantized_concat_const_blob");
}

TEST_P(Test_ONNX_layers, Quantized_Constant)
{
    testONNXModels("quantized_constant", npy, 0.008, 0.02);
}

TEST_P(Test_ONNX_layers, OutputRegistration)
{
    testONNXModels("output_registration", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, QLinearSoftmax)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    testONNXModels("qlinearsoftmax_v11", npy, 0.002, 0.002); // 2D coerced
    testONNXModels("qlinearsoftmax_v13", npy, 0.002, 0.002);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_ONNX_layers, dnnBackendsAndTargets());

class Test_ONNX_nets : public Test_ONNX_layers
{
public:
    Test_ONNX_nets() { required = false; }
};

TEST_P(Test_ONNX_nets, Alexnet)
{
#if defined(OPENCV_32BIT_CONFIGURATION) && (defined(HAVE_OPENCL) || defined(_WIN32))
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif

    const String model =  _tf("models/alexnet.onnx", false);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.enableWinograd(false);

    Mat inp = imread(_tf("../grace_hopper_227.png"));
    Mat ref = blobFromNPY(_tf("../caffe_alexnet_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(blobFromImage(inp, 1.0f, Size(227, 227), Scalar(), false));
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
    expectNoFallbacksFromIE(net);
}

TEST_P(Test_ONNX_nets, RAFT)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG, CV_TEST_TAG_MEMORY_2GB);

    std::string weight_path = _tf("models/optical_flow_estimation_raft_2023aug.onnx", false);
    std::string img0_path = findDataFile(std::string("gpu/opticalflow/frame0.png"));
    std::string img1_path = findDataFile(std::string("gpu/opticalflow/frame1.png"));

    Size target_size{480, 360};
    auto img0 = imread(img0_path);
    auto img1 = imread(img1_path);
    auto blob0 = blobFromImage(img0, 1.0, target_size, 0, true);
    auto blob1 = blobFromImage(img1, 1.0, target_size, 0, true);

    auto net = readNet(weight_path);
    net.setInput(blob0, "0");
    net.setInput(blob1, "1");
    std::vector<std::string> outnames{"12007", "12006"};
    std::vector<Mat> outs;
    net.forward(outs, outnames);

    // output 12006 is not checked to save space in opencv_extra since its ref is > 1MB,
    // and output 12006 is calculated from 12007 so checking 12007 is sufficient.
    std::string ref_12700_path = _tf("data/output_optical_flow_estimation_raft_2023aug.npy");
    auto ref0 = blobFromNPY(ref_12700_path);
    normAssert(ref0, outs[0], "", 1e-5, 1.8e-4);
}

TEST_P(Test_ONNX_nets, Squeezenet)
{
    testONNXModels("squeezenet", pb);
}

TEST_P(Test_ONNX_nets, Googlenet)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    const String model = _tf("models/googlenet.onnx", false);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    if (target == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    std::vector<Mat> images, results;
    images.push_back( imread(_tf("../googlenet_0.png")) );
    images.push_back( imread(_tf("../googlenet_1.png")) );
    Mat ref = blobFromNPY(_tf("../googlenet_prob.npy"));
    for (int i = 0; i < 2; i++) {
        Mat inp_i = blobFromImage(images[i], 1.0f, Size(), Scalar(), false);
        net.setInput(inp_i);
        ASSERT_FALSE(net.empty());
        Mat out_i = net.forward();
        results.push_back(out_i.clone());
    }
    Mat out;
    vconcat(results, out);

    normAssert(ref, out, "", default_l1,  default_lInf);
    expectNoFallbacksFromIE(net);
}

TEST_P(Test_ONNX_nets, CaffeNet)
{
#if defined(OPENCV_32BIT_CONFIGURATION) && (defined(HAVE_OPENCL) || defined(_WIN32))
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2019030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    testONNXModels("caffenet", pb);
}

TEST_P(Test_ONNX_nets, RCNN_ILSVRC13)
{
#if defined(OPENCV_32BIT_CONFIGURATION) && (defined(HAVE_OPENCL) || defined(_WIN32))
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2019030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    // Reference output values are in range [-4.992, -1.161]
    testONNXModels("rcnn_ilsvrc13", pb, 0.0046);
}

TEST_P(Test_ONNX_nets, VGG16_bn)
{
    applyTestTag(CV_TEST_TAG_MEMORY_6GB);  // > 2.3Gb

    // output range: [-16; 27], after Softmax [0; 0.67]
    const double lInf = (target == DNN_TARGET_MYRIAD) ? 0.038 : default_lInf;
    testONNXModels("vgg16-bn", pb, default_l1, lInf, true);
}

TEST_P(Test_ONNX_nets, ZFNet)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
    testONNXModels("zfnet512", pb);
}

TEST_P(Test_ONNX_nets, ResNet18v1)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // output range: [-16; 22], after Softmax [0, 0.51]
    testONNXModels("resnet18v1", pb, default_l1, default_lInf, true, target != DNN_TARGET_MYRIAD);
}

TEST_P(Test_ONNX_nets, ResNet50v1)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // output range: [-67; 75], after Softmax [0, 0.98]
    size_t hwm0 = getTopMemoryUsageMB();
    testONNXModels("resnet50v1", pb, default_l1, default_lInf, true, target != DNN_TARGET_MYRIAD);
    size_t hwm1 = getTopMemoryUsageMB();
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_CPU)
    {
        EXPECT_LE(hwm1 - hwm0, 350) << "Top allocated memory";
    }
}

TEST_P(Test_ONNX_nets, ResNet50_Int8)
{
    testONNXModels("resnet50_int8", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, ResNet101_DUC_HDC)
{
    applyTestTag(CV_TEST_TAG_VERYLONG);

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL)
    {
        if (backend == DNN_BACKEND_OPENCV)
            applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_OPENCL : CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
        throw SkipTestException("Test is disabled for OpenCL targets");
    }
    testONNXModels("resnet101_duc_hdc", pb);
}

TEST_P(Test_ONNX_nets, TinyYolov2)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    if (cvtest::skipUnstableTests)
        throw SkipTestException("Skip unstable test");
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019
            && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16)
    )
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    if (target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X,
                     backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ?
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER :
                     CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    // output range: [-11; 8]
    double l1 =  default_l1, lInf = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 0.02;
        lInf = 0.2;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.018;
        lInf = 0.16;
    }
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.018f; lInf = 0.16f;
    }
#endif

    testONNXModels("tiny_yolo2", pb, l1, lInf, false, true, 1, true, false);
}

TEST_P(Test_ONNX_nets, CNN_MNIST)
{
    // output range: [-1952; 6574], after Softmax [0; 1]
    testONNXModels("cnn_mnist", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, MobileNet_v2)
{
    // output range: [-166; 317], after Softmax [0; 1]
    testONNXModels("mobilenetv2", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, MobileNet_v2_FP16)
{
    testONNXModels("mobilenetv2_fp16", npy, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, LResNet100E_IR)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
#endif
        CV_TEST_TAG_DEBUG_VERYLONG
    );
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }

    double l1 = default_l1, lInf = default_lInf;
    // output range: [-3; 3]
    bool useWinograd = true;
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.009;
        lInf = 0.035;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_CPU)
    {
        l1 = 4.6e-5;
        lInf = 1.9e-4;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.009;
        lInf = 0.04;
    }
    else if (target == DNN_TARGET_CPU_FP16)
    {
        useWinograd = false;
        l1 = 0.009;
        lInf = 0.035;
    }

    testONNXModels("LResNet100E_IR", pb, l1, lInf, false, true, 1, true, useWinograd);
}

TEST_P(Test_ONNX_nets, Emotion_ferplus)
{
#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X,
                     backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ?
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER :
                     CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    double l1 = default_l1;
    double lInf = default_lInf;
    bool useWinograd = true;
    // Output values are in range [-2.011, 2.111]
    if ((backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) || (target == DNN_TARGET_CUDA_FP16))
        l1 = 0.007;
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.021;
        lInf = 0.034;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_CPU || target == DNN_TARGET_OPENCL)) {
        l1 = 2.4e-4;
        lInf = 6e-4;
    }
    else if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_CPU_FP16)
    {
        useWinograd = false;
        l1 = 0.007;
    }
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.013f; lInf = 0.035f;
    }
#endif

    testONNXModels("emotion_ferplus", pb, l1, lInf, false, true, 1, true, useWinograd);
}

TEST_P(Test_ONNX_nets, Inception_v2)
{
    testONNXModels("inception_v2", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, DenseNet121)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // output range: [-87; 138], after Softmax [0; 1]
    testONNXModels("densenet121", pb, default_l1, default_lInf, true, target != DNN_TARGET_MYRIAD);
}

TEST_P(Test_ONNX_nets, Inception_v1)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif
    testONNXModels("inception_v1", pb);
}

TEST_P(Test_ONNX_nets, Shufflenet)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    }
#endif
    testONNXModels("shufflenet", pb);
}

TEST_P(Test_ONNX_nets, Resnet34_kinetics)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // IE exception: Failed to allocate graph: MYRIAD device is not opened
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        // IE exception: Function contains several inputs and outputs with one friendly name!
        if (target == DNN_TARGET_MYRIAD)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
#endif
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");  // FIXIT use tags

    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);

    String onnxmodel = findDataFile("dnn/resnet-34_kinetics.onnx", false);
    Mat image0 = imread(findDataFile("dnn/dog416.png"));
    Mat image1 = imread(findDataFile("dnn/street.png"));

    Mat ref0 = blobFromNPY(_tf("data/output_kinetics0.npy"));
    Mat ref1 = blobFromNPY(_tf("data/output_kinetics1.npy"));

    std::vector<Mat> images_0(16, image0);
    std::vector<Mat> images_1(16, image1);
    Mat blob0 = blobFromImages(images_0, 1.0, Size(112, 112), Scalar(114.7748, 107.7354, 99.4750), true, true);
    Mat blob1 = blobFromImages(images_1, 1.0, Size(112, 112), Scalar(114.7748, 107.7354, 99.4750), true, true);

    Net permute;
    LayerParams lp;
    int order[] = {1, 0, 2, 3};
    lp.set("order", DictValue::arrayInt<int*>(&order[0], 4));
    permute.addLayerToPrev("perm", "Permute", lp);

    permute.setPreferableBackend(backend);
    permute.setPreferableTarget(target);

    permute.setInput(blob0);
    Mat input0 = permute.forward().clone();

    permute.setInput(blob1);
    Mat input1 = permute.forward().clone();

    int dims[] = {1, 3, 16, 112, 112};
    input0 = input0.reshape(0, 5, &dims[0]);
    input1 = input1.reshape(0, 5, &dims[0]);

    Net net = readNetFromONNX(onnxmodel);
    ASSERT_FALSE(net.empty());
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    // output range [-5, 11]
    float l1 = 0.0013;
    float lInf = 0.009;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.02;
        lInf = 0.07;
    }
    if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.01;
        lInf = 0.06;
    }

    testInputShapes(net, {input0});

    checkBackend(&input0, &ref0);
    net.setInput(input0);
    Mat out = net.forward().clone();
    normAssert(ref0, out, "", l1, lInf);

    checkBackend(&input1, &ref1);
    net.setInput(input1);
    out = net.forward().clone();
    normAssert(ref1, out, "", l1, lInf);

    expectNoFallbacksFromIE(net);
}

TEST_P(Test_ONNX_layers, CumSum)
{
    testONNXModels("cumsum_1d_exclusive_1");
    testONNXModels("cumsum_1d_reverse");
    testONNXModels("cumsum_1d_exclusive_1_reverse");
    testONNXModels("cumsum_2d_dim_1");
    testONNXModels("cumsum_3d_dim_2");
    testONNXModels("cumsum_3d_dim_2_int32");
}

TEST_P(Test_ONNX_layers, CumSum_int64)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // OpenVINO uses int32 precision for int64 operations
    testONNXModels("cumsum_3d_dim_2_int64");
}

TEST_P(Test_ONNX_layers, ReduceSumInt64)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // OpenVINO uses int32 precision for int64 operations
    testONNXModels("reduce_sum_int64");
}

TEST_P(Test_ONNX_layers, ScatterInt32)
{
    testONNXModels("scatter_int32", npy, 0, 0, false, true, 3);
}

TEST_P(Test_ONNX_layers, ScatterInt64)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // OpenVINO uses int32 precision for int64 operations
    testONNXModels("scatter_int64", npy, 0, 0, false, true, 3);
}

TEST_P(Test_ONNX_layers, TileInt32)
{
    testONNXModels("tile_int32");
}

TEST_P(Test_ONNX_layers, TileInt64)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // OpenVINO uses int32 precision for int64 operations
    testONNXModels("tile_int64");
}

static void testYOLO(const std::string& weightPath, const std::vector<int>& refClassIds,
                     const std::vector<float>& refScores, const std::vector<Rect2d>& refBoxes,
                     Image2BlobParams imgParams, float conf_threshold = 0.3, float iou_threshold = 0.5,
                     double scores_diff = 1e-5, double boxes_iou_diff = 1e-4, const std::string test_name = "")
{
    std::string imgPath = _tf("../dog_orig_size.png");

    Mat img = imread(imgPath);

    Mat inp = blobFromImageWithParams(img, imgParams);

    Net net = readNet(weightPath);

    net.setInput(inp);
    std::vector<Mat> outs;
    std::vector<std::string> out_names = net.getUnconnectedOutLayersNames();
    net.forward(outs, out_names);
    EXPECT_EQ(outs.size(), out_names.size());
    if(outs.size() == 1)
    {
        // do nothing
    }
    else if (outs.size() == 2)
    {
        // sort outs by name. New and old DNN engines return otuput in different order!
        if(out_names[0] > out_names[1])
        {
            std::swap(out_names[0], out_names[1]);
            std::swap(outs[0], outs[1]);
        }
    }
    else if (outs.size() > 2)
    {
        CV_Error(Error::StsUnsupportedFormat, "Too many Yolo network outputs!");
    }

    // Retrieve
    std::vector<int> keep_classIds;
    std::vector<float> keep_confidences;
    std::vector<Rect2d> keep_boxes;
    yoloPostProcessing(outs, keep_classIds, keep_confidences, keep_boxes, conf_threshold, iou_threshold, test_name);

    normAssertDetections(
        refClassIds, refScores, refBoxes,
        keep_classIds, keep_confidences, keep_boxes,
        "", 0.0, scores_diff, boxes_iou_diff);
}

void yoloPostProcessing(
    std::vector<Mat>& outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold,
    const std::string& model_name,
    const int nc
){

    // Retrieve
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;

    if (model_name == "yolov8" || model_name == "yolov10" ||
        model_name == "yolov9")
    {
        cv::transposeND(outs[0], {0, 2, 1}, outs[0]);
    }

    if (model_name == "yolonas"){
        EXPECT_EQ(cv::MatShape({1, 8400, 80}), outs[0].shape());
        EXPECT_EQ(cv::MatShape({1, 8400, 4}), outs[1].shape());
        // outs contains 2 elemets of shape [1, 8400, nc] and [1, 8400, 4]. Concat them to get [1, 8400, 84]
        Mat concat_out;
        // squeeze the first dimension
        outs[0] = outs[0].reshape(1, outs[0].size[1]);
        outs[1] = outs[1].reshape(1, outs[1].size[1]);
        cv::hconcat(outs[1], outs[0], concat_out);
        outs[0] = concat_out;
        // remove the second element
        outs.pop_back();
        // unsqueeze the first dimension
        outs[0] = outs[0].reshape(0, std::vector<int>{1, outs[0].size[0], outs[0].size[1]});
    }

    // assert if last dim is nc+5 or nc+4
    CV_CheckEQ(outs[0].dims, 3, "Invalid output shape. The shape should be [1, #anchors, nc+5 or nc+4]");
    CV_CheckEQ((outs[0].size[2] == nc + 5 || outs[0].size[2] == nc + 4), true, "Invalid output shape: ");

    for (auto preds : outs){

        preds = preds.reshape(1, preds.size[1]); // [1, 8400, 85] -> [8400, 85]
        for (int i = 0; i < preds.rows; ++i)
        {
            // filter out non object
            float obj_conf = (model_name == "yolov8" || model_name == "yolonas" ||
                              model_name == "yolov9" || model_name == "yolov10") ? 1.0f : preds.at<float>(i, 4) ;
            if (obj_conf < conf_threshold)
                continue;

            Mat scores = preds.row(i).colRange((model_name == "yolov8" || model_name == "yolonas" || model_name == "yolov9" || model_name == "yolov10") ? 4 : 5, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);

            conf = (model_name == "yolov8" || model_name == "yolonas" || model_name == "yolov9" || model_name == "yolov10") ? conf : conf * obj_conf;
            if (conf < conf_threshold)
                continue;

            // get bbox coords
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];

            // [x1, y1, x2, y2]
            if (model_name == "yolonas" || model_name == "yolov10"){
                boxes.push_back(Rect2d(cx, cy, w, h));
            } else {
                boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                                        cx + 0.5 * w, cy + 0.5 * h));
            }
            classIds.push_back(maxLoc.x);
            confidences.push_back(conf);
        }
    }

    // NMS
    std::vector<int> keep_idx;
    NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);

    for (auto i : keep_idx)
    {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
}

TEST_P(Test_ONNX_nets, YOLOv10)
{

    std::string weightPath = _tf("models/yolov10s.onnx", false);

    Size targetSize{640, 480};
    float conf_threshold = 0.50;
    float iou_threshold = 0.50;

    std::vector<int> refClassIds{1, 16, 7};
    std::vector<float> refScores{0.9510f, 0.9454f, 0.8404f};

    std::vector<Rect2d> refBoxes{
        Rect2d(105.5014, 112.8838, 472.9274, 350.0603),
        Rect2d(109.8231, 185.7994, 258.5916, 452.9302),
        Rect2d(388.5018,  62.1034, 576.6399, 143.3986)
        };

    Image2BlobParams imgParams(
        Scalar::all(1 / 255.0),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX,
        Scalar::all(114)
        );

    testYOLO(
        weightPath, refClassIds, refScores, refBoxes,
        imgParams, conf_threshold, iou_threshold,
        1.0e-4, 1.0e-4, "yolov10");
}

TEST_P(Test_ONNX_nets, YOLOv9)
{

    std::string weightPath = _tf("models/yolov9t.onnx", false);

    Size targetSize{640, 480};
    float conf_threshold = 0.50;
    float iou_threshold = 0.50;

    std::vector<int> refClassIds{1, 16, 2}; // wrong class mapping for yolov9
    std::vector<float> refScores{0.959274f, 0.901125f, 0.559396f};

    std::vector<Rect2d> refBoxes{
        Rect2d(106.255, 107.927, 472.497, 350.309),
        Rect2d(108.633, 185.256, 259.287, 450.672),
        Rect2d(390.701, 62.1454, 576.928, 141.795)
        };

    Image2BlobParams imgParams(
        Scalar::all(1 / 255.0),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX,
        Scalar::all(114)
        );

    testYOLO(
        weightPath, refClassIds, refScores, refBoxes,
        imgParams, conf_threshold, iou_threshold,
        1.0e-4, 1.0e-4, "yolov9");
}
TEST_P(Test_ONNX_nets, YOLOX)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    std::string weightPath = _tf("models/yolox_s_inf_decoder.onnx", false);

    Size targetSize{640, 640};
    float conf_threshold = 0.50;
    float iou_threshold = 0.50;

    std::vector<int> refClassIds{1, 16, 7};
    std::vector<float> refScores{0.9649f, 0.9163f, 0.6879f};

    std::vector<Rect2d> refBoxes{
        Rect2d(105.5384, 179.4100, 470.6339, 428.5553),
        Rect2d(111.4482, 263.4098, 258.7438, 526.1140),
        Rect2d(389.1421, 143.9286, 577.9495, 222.0294)
        };

    Image2BlobParams imgParams(
        Scalar::all(1),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX,
        Scalar::all(114)
        );

    testYOLO(
        weightPath, refClassIds, refScores, refBoxes,
        imgParams, conf_threshold, iou_threshold,
        1.0e-4, 1.0e-4);
}

TEST_P(Test_ONNX_nets, YOLONas)
{
    // model information: https://dl.opencv.org/models/yolo-nas/Readme.md
    std::string weightPath = _tf("models/yolo_nas_s.onnx", false);

    Size targetSize{640, 640};
    float conf_threshold = 0.50;
    float iou_threshold = 0.50;

    std::vector<int> refClassIds{1, 16, 7};
    std::vector<float> refScores{0.9720f, 0.9283f, 0.8990f};
    // [x1, y1, x2, y2]
    std::vector<Rect2d> refBoxes{
        Rect2d(105.516, 173.696, 471.323, 430.433),
        Rect2d(109.241, 263.406, 259.872, 531.858),
        Rect2d(390.153, 142.492, 574.932, 222.709)
        };

    Image2BlobParams imgParams(
        Scalar::all(1/255.0),
        targetSize,
        Scalar::all(0),
        false,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX,
        Scalar::all(114)
        );

    testYOLO(
        weightPath, refClassIds, refScores, refBoxes,
        imgParams, conf_threshold, iou_threshold,
        1.0e-4, 1.0e-4, "yolonas");
}

TEST_P(Test_ONNX_nets, YOLOv8)
{
    std::string weightPath = _tf("models/yolov8n.onnx", false);

    Size targetSize{640, 640};
    float conf_threshold = 0.25;
    float iou_threshold = 0.50;

    std::vector<int> refClassIds{16, 1, 2};
    std::vector<float> refScores{0.9332f, 0.8959f, 0.6157f};
    // [x1, y1, x2, y2]
    std::vector<Rect2d> refBoxes{
        Rect2d(108.8965, 261.9094, 257.1633, 530.3049),
        Rect2d(110.4020, 192.9843, 473.4418, 429.5965),
        Rect2d(389.1603, 143.2506, 577.3542, 223.0615),
        };

    Image2BlobParams imgParams(
        Scalar::all(1/255.0),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX,
        Scalar::all(114)
        );

    testYOLO(
        weightPath, refClassIds, refScores, refBoxes,
        imgParams, conf_threshold, iou_threshold,
        1.0e-4, 1.0e-4, "yolov8");
}

// This test is mainly to test:
//  1. identity node with constant input
//  2. limited support to range operator (all inputs are constant)
//  3. parseExpand with multiple broadcast axes
//  4. 1D mat dimension issue with the output of range operator
TEST_P(Test_ONNX_nets, YOLOv7)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    std::string weightPath = _tf("models/yolov7.onnx", false);
    // Reference, which is collected with input size of 640x640
    std::vector<int> refClassIds{1, 16, 7};
    std::vector<float> refScores{0.9614331f, 0.9589417f, 0.8679074f};
    // [x1, y1, x2, y2] x 3
    std::vector<Rect2d> refBoxes{Rect2d(105.973236f, 150.16716f,  472.59012f, 466.48834f),
                                 Rect2d(109.97953f,  246.17862f, 259.83676f, 600.76624f),
                                 Rect2d(385.96185f, 83.02809f,  576.07355f,  189.82793f)};

    Size targetSize{640, 640};

    Image2BlobParams imgParams(
        Scalar::all(1/255.0),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_NULL,
        Scalar::all(0)
        );

    testYOLO(weightPath, refClassIds, refScores, refBoxes, imgParams);
}

TEST_P(Test_ONNX_nets, YOLOv6)
{
    std::string weightPath = _tf("models/yolov6n.onnx", false);

    Size targetSize{640, 640};
    float conf_threshold = 0.30;
    float iou_threshold = 0.50;

    std::vector<int> refClassIds{1, 16, 7, 1};
    std::vector<float> refScores{0.95031f, 0.87123f,  0.65453f, 0.34142f};
    // [x1, y1, x2, y2] x 3
    std::vector<Rect2d> refBoxes{Rect2d(98.84, 177.91, 473.29, 431.19),
                                 Rect2d(109.80, 265.50, 258.86, 531.97),
                                 Rect2d(387.79, 141.61, 576.98, 223.52),
                                 Rect2d(105.62, 199.24, 218.37, 389.84),
                                 };

    Image2BlobParams imgParams(
        Scalar::all(1/255.0),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX,
        Scalar::all(114)
        );

    testYOLO(
        weightPath, refClassIds, refScores, refBoxes,
        imgParams, conf_threshold, iou_threshold,
        1.0e-4, 1.0e-3);
}

TEST_P(Test_ONNX_nets, YOLOv5n)
{
    std::string weightPath = findDataFile("dnn/yolov5n.onnx", false);
    // Reference, which is collected with input size of 640x640
    std::vector<int> refClassIds{16, 2, 1};
    std::vector<float> refScores{0.749053f, 0.616853f, 0.32506f};
    // [x1, y1, x2, y2] x 4

    std::vector<Rect2d> refBoxes{Rect2d(108.088f, 239.293f, 266.196f, 607.658f),
                                 Rect2d(392.028f, 89.9233f, 579.152f, 190.447f),
                                 Rect2d(120.278f, 159.76, 214.481f, 241.473f)};

    Size targetSize{640, 640};

    Image2BlobParams imgParams(
        Scalar::all(1/255.0),
        targetSize,
        Scalar::all(0),
        true,
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_NULL,
        Scalar::all(0)
        );

    testYOLO(weightPath, refClassIds, refScores, refBoxes, imgParams);
}

TEST_P(Test_ONNX_layers, Tile)
{
    testONNXModels("tile", pb);
}

TEST_P(Test_ONNX_layers, Gelu)
{
    testONNXModels("gelu");
    testONNXModels("gelu_approximation");
}

TEST_P(Test_ONNX_layers, OpenAI_CLIP_head)
{
    testONNXModels("clip-vit-base-head");
}

TEST_P(Test_ONNX_layers, where_node)
{
    testONNXModels("where_layer");
}

TEST_P(Test_ONNX_layers, Gemm_all_attributes) {
    testONNXModels("test_gemm_all_attributes", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_alpha) {
    testONNXModels("test_gemm_alpha", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_beta) {
    testONNXModels("test_gemm_beta", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_default_matrix_bias) {
    testONNXModels("test_gemm_default_matrix_bias", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_default_no_bias) {
    testONNXModels("test_gemm_default_no_bias", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_default_scalar_bias) {
    testONNXModels("test_gemm_default_scalar_bias", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_default_single_elem_vector_bias) {
    testONNXModels("test_gemm_default_single_elem_vector_bias", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_default_vector_bias) {
    testONNXModels("test_gemm_default_vector_bias", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_default_zero_bias) {
    testONNXModels("test_gemm_default_zero_bias", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_transposeA) {
    testONNXModels("test_gemm_transposeA", pb, 0, 0, false, true, 2);
}
TEST_P(Test_ONNX_layers, Gemm_transposeB) {
    testONNXModels("test_gemm_transposeB", pb, 0, 0, false, true, 2);
}

// Note: These tests are converted from onnx/onnx so that they have constant shape as input.
// TODO: They can be moved into conformance tests once dynamic input is properly supported.
TEST_P(Test_ONNX_layers, Expand_dim_changed) {
    testONNXModels("test_expand_dim_changed", pb, 0, 0, false, true, 1);
}
TEST_P(Test_ONNX_layers, Expand_dim_unchanged) {
    testONNXModels("test_expand_dim_unchanged", pb, 0, 0, false, true, 1);
}
TEST_P(Test_ONNX_layers, Expand_shape_model1) {
    testONNXModels("test_expand_shape_model1", pb, 0, 0, false, true, 1);
}
TEST_P(Test_ONNX_layers, Expand_shape_model2) {
    testONNXModels("test_expand_shape_model2", pb, 0, 0, false, true, 1);
}
TEST_P(Test_ONNX_layers, Expand_shape_model3) {
    testONNXModels("test_expand_shape_model3", pb, 0, 0, false, true, 1);
}
TEST_P(Test_ONNX_layers, Expand_shape_model4) {
    testONNXModels("test_expand_shape_model4", pb, 0, 0, false, true, 1);
}

TEST_P(Test_ONNX_layers, Attention) {
    testONNXModels("attention");
}
TEST_P(Test_ONNX_layers, AttentionSingleHead) {
    testONNXModels("attention_single_head");
}
TEST_P(Test_ONNX_layers, PyTorchAttentionSingleHead) {
    // 5.x specific bug: https://github.com/opencv/opencv/issues/25921
    if (target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    if (target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    testONNXModels("pytorch_attention_single_head");
}

TEST_P(Test_ONNX_layers, PyTorchUnflatten){
    testONNXModels("unflatten");
}

TEST_P(Test_ONNX_nets, ViT_B_32) {
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_LONG);

    const std::string model_path = _tf("models/vit_b_32.onnx", false);

    auto net = readNet(model_path);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    auto image = imread(_tf("../googlenet_0.png"));
    auto blob = blobFromImage(image, 1.f, Size(224, 224));
    auto ref = blobFromNPY(_tf("data/output_vit_b_32.npy"));
    checkBackend(&blob, &ref);

    net.setInput(blob);
    auto out = net.forward();

    double l1 = default_l1;
    double lInf = default_lInf;
    if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.01;
        lInf = 0.06;
    }
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 0.008;
        lInf = 0.04;
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
        if (target == DNN_TARGET_CPU) {
            l1 = 4.4e-5; // Expected: (normL1) <= (l1), actual: 4.31208e-05 vs 1e-05
            lInf = 0.0002; // Expected: (normInf) <= (lInf), actual: 0.000194907 vs 0.0001
        } else if (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16) {
            l1 = 0.0092; // Expected: (normL1) <= (l1), actual: 0.00918349 vs 4.4e-05
            lInf = 0.056; // Expected: (normInf) <= (lInf), actual: 0.0556431 vs 0.0002
        }
    }

    normAssert(ref, out, "ViTB_32", l1, lInf);
}

TEST_P(Test_ONNX_nets, VitTrack) {
    auto image = imread(_tf("../dog_orig_size.png"));
    auto input0 = blobFromImage(image, 1.f, Size(128, 128));
    auto input1 = blobFromImage(image, 1.f, Size(256, 256));

    auto net = readNet(_tf("models/object_tracking_vittrack_2023sep.onnx", false));
    net.setInput(input0, "template");
    net.setInput(input1, "search");

    std::vector<std::string> output_names{"output1", "output2", "output3"};
    std::vector<Mat> outputs;
    net.forward(outputs, output_names);

    auto ref_output1 = blobFromNPY(_tf("data/output_object_tracking_vittrack_2023sep_0.npy"));
    auto ref_output2 = blobFromNPY(_tf("data/output_object_tracking_vittrack_2023sep_1.npy"));
    auto ref_output3 = blobFromNPY(_tf("data/output_object_tracking_vittrack_2023sep_2.npy"));

    normAssert(ref_output1, outputs[0], "VitTrack output1");
    normAssert(ref_output2, outputs[1], "VitTrack output2");
    normAssert(ref_output3, outputs[2], "VitTrack output3");
}

TEST_P(Test_ONNX_layers, LayerNormNoFusion) {
    testONNXModels("layer_norm_no_fusion");
}

TEST_P(Test_ONNX_layers, MatMulAddFusion) {
    double l1 = (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL) ? 0.0018 : default_l1;
    double lInf = (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL) ? 0.011 : default_lInf;
    testONNXModels("biased_matmul", npy, l1, lInf);
}

TEST_P(Test_ONNX_layers, ClipDivSharedConstant) {
    testONNXModels("clip_div_shared_constant");
}

TEST_P(Test_ONNX_layers, TopK) {
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
        backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
        backend == DNN_BACKEND_INFERENCE_ENGINE) {
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE); // OpenVINO does not support int64
    }
    auto test = [&](const std::string &basename, double l1 = 0, double lInf = 0) {
        std::string onnxmodel = _tf("models/" + basename + ".onnx", true);
        Mat input = readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
        Mat output_ref_val = readTensorFromONNX(_tf("data/output_" + basename + "_0.pb")),
            output_ref_ind = readTensorFromONNX(_tf("data/output_" + basename + "_1.pb"));

        checkBackend(&input, &output_ref_val);
        checkBackend(&input, &output_ref_ind);
        Net net = readNetFromONNX(onnxmodel);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        net.setInput(input);
        std::vector<Mat> outputs;
        net.forward(outputs, std::vector<std::string>{"values", "indices"});

        Mat output_res_val = outputs.front(),
            output_res_ind = outputs.back();

        normAssert(output_ref_val, output_res_val, (basename + " values").c_str(), l1 ? l1 : default_l1, lInf ? lInf : default_lInf);
        normAssert(output_ref_ind, output_res_ind, (basename + " indices").c_str(), l1 ? l1 : default_l1, lInf ? lInf : default_lInf);

        expectNoFallbacksFromIE(net);
    };

    test("top_k");
    test("top_k_negative_axis");
    test("top_k_smallest");
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
