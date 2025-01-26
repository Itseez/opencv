// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
namespace opencv_test { namespace {

testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargetsInt8()
{
    std::vector< tuple<Backend, Target> > targets;
    targets.push_back(make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU));
#ifdef HAVE_TIMVX
    targets.push_back(make_tuple(DNN_BACKEND_TIMVX, DNN_TARGET_NPU));
#endif
#ifdef HAVE_INF_ENGINE
    targets.push_back(make_tuple(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, DNN_TARGET_CPU));
#endif
    return testing::ValuesIn(targets);
}

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "dnn/") + filename;
}

class Test_Int8_layers : public DNNTestLayer
{
public:
    void testLayer(const String& basename, const String& importer, double l1, double lInf,
                   int numInps = 1, int numOuts = 1, bool useCaffeModel = false,
                   bool useCommonInputBlob = true, bool hasText = false, bool perChannel = true)
    {
        CV_Assert_N(numInps >= 1, numInps <= 10, numOuts >= 1, numOuts <= 10);
        std::vector<Mat> inps(numInps), inps_int8(numInps);
        std::vector<Mat> refs(numOuts), outs_int8(numOuts), outs_dequantized(numOuts);
        std::vector<float> inputScale, outputScale;
        std::vector<int> inputZp, outputZp;
        String inpPath, outPath;
        Net net, qnet;

        if (importer == "Caffe")
        {
            String prototxt = _tf("layers/" + basename + ".prototxt");
            String caffemodel = _tf("layers/" + basename + ".caffemodel");
            net = readNetFromCaffe(prototxt, useCaffeModel ? caffemodel : String());

            inpPath = _tf("layers/" + (useCommonInputBlob ? "blob" : basename + ".input"));
            outPath =  _tf("layers/" + basename);
        }
        else if (importer == "TensorFlow")
        {
            String netPath = _tf("tensorflow/" + basename + "_net.pb");
            String netConfig = hasText ? _tf("tensorflow/" + basename + "_net.pbtxt") : "";
            net = readNetFromTensorflow(netPath, netConfig);

            inpPath = _tf("tensorflow/" + basename + "_in");
            outPath = _tf("tensorflow/" + basename + "_out");
        }
        else if (importer == "ONNX")
        {
            String onnxmodel = _tf("onnx/models/" + basename + ".onnx");
            net = readNetFromONNX(onnxmodel);

            inpPath = _tf("onnx/data/input_" + basename);
            outPath = _tf("onnx/data/output_" + basename);
        }
        ASSERT_FALSE(net.empty());

        for (int i = 0; i < numInps; i++)
            inps[i] = blobFromNPY(inpPath + ((numInps > 1) ? cv::format("_%d.npy", i) : ".npy"));

        for (int i = 0; i < numOuts; i++)
            refs[i] = blobFromNPY(outPath + ((numOuts > 1) ? cv::format("_%d.npy", i) : ".npy"));

        qnet = net.quantize(inps, CV_8S, CV_8S, perChannel);
        qnet.getInputDetails(inputScale, inputZp);
        qnet.getOutputDetails(outputScale, outputZp);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);

        // Quantize inputs to int8
        // int8_value = float_value/scale + zero-point
        for (int i = 0; i < numInps; i++)
        {
            inps[i].convertTo(inps_int8[i], CV_8S, 1.f/inputScale[i], inputZp[i]);
            String inp_name = numInps > 1 ? (importer == "Caffe" ? cv::format("input_%d", i) : cv::format("%d", i)) : "";
            qnet.setInput(inps_int8[i], inp_name);
        }
        qnet.forward(outs_int8);

        // Dequantize outputs and compare with reference outputs
        // float_value = scale*(int8_value - zero-point)
        for (int i = 0; i < numOuts; i++)
        {
            outs_int8[i].convertTo(outs_dequantized[i], CV_32F, outputScale[i], -(outputScale[i] * outputZp[i]));
            normAssert(refs[i], outs_dequantized[i], basename.c_str(), l1, lInf);
        }
    }
};

TEST_P(Test_Int8_layers, Convolution1D)
{
    testLayer("conv1d", "ONNX", 0.00302, 0.00909);
    testLayer("conv1d_bias", "ONNX", 0.00306, 0.00948);

    {
        SCOPED_TRACE("Per-tensor quantize");
        testLayer("conv1d", "ONNX", 0.00302, 0.00909, 1, 1, false, true, false, false);
        testLayer("conv1d_bias", "ONNX", 0.00319, 0.00948, 1, 1, false, true, false, false);
    }
}

TEST_P(Test_Int8_layers, Convolution2D)
{
    if(backend == DNN_BACKEND_TIMVX)
        testLayer("single_conv", "TensorFlow", 0.00424, 0.02201);
    else
        testLayer("single_conv", "TensorFlow", 0.00413, 0.02201);

    testLayer("atrous_conv2d_valid", "TensorFlow", 0.0193, 0.0633);
    testLayer("atrous_conv2d_same", "TensorFlow", 0.0185, 0.1322);
    testLayer("keras_atrous_conv2d_same", "TensorFlow", 0.0056, 0.0244);

    if(backend == DNN_BACKEND_TIMVX)
        testLayer("convolution", "ONNX", 0.00534, 0.01516);
    else
        testLayer("convolution", "ONNX", 0.0052, 0.01516);

    if(backend == DNN_BACKEND_TIMVX)
        testLayer("two_convolution", "ONNX", 0.0033, 0.01);
    else
        testLayer("two_convolution", "ONNX", 0.00295, 0.00840);

    if(backend == DNN_BACKEND_TIMVX)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_TIMVX);
    testLayer("layer_convolution", "Caffe", 0.0174, 0.0758, 1, 1, true);
    testLayer("depthwise_conv2d", "TensorFlow", 0.0388, 0.169);

    {
        SCOPED_TRACE("Per-tensor quantize");
        testLayer("single_conv", "TensorFlow", 0.00413, 0.02301, 1, 1, false, true, false, false);
        testLayer("atrous_conv2d_valid", "TensorFlow", 0.027967, 0.07808, 1, 1, false, true, false, false);
        testLayer("atrous_conv2d_same", "TensorFlow", 0.01945, 0.1322, 1, 1, false, true, false, false);
        testLayer("keras_atrous_conv2d_same", "TensorFlow", 0.005677, 0.03327, 1, 1, false, true, false, false);
        testLayer("convolution", "ONNX", 0.00538, 0.01517, 1, 1, false, true, false, false);
        testLayer("two_convolution", "ONNX", 0.00295, 0.00926, 1, 1, false, true, false, false);
        testLayer("layer_convolution", "Caffe", 0.0175, 0.0759, 1, 1, true, true, false, false);
        testLayer("depthwise_conv2d", "TensorFlow", 0.041847, 0.18744, 1, 1, false, true, false, false);
    }
}

TEST_P(Test_Int8_layers, Convolution3D)
{
    testLayer("conv3d", "TensorFlow", 0.00734, 0.02434);
    testLayer("conv3d", "ONNX", 0.00353, 0.00941);
    testLayer("conv3d_bias", "ONNX", 0.00129, 0.00249);
}

TEST_P(Test_Int8_layers, Flatten)
{
    testLayer("flatten", "TensorFlow", 0.0036, 0.0069, 1, 1, false, true, true);
    testLayer("unfused_flatten", "TensorFlow", 0.0014, 0.0028);
    testLayer("unfused_flatten_unknown_batch", "TensorFlow", 0.0043, 0.0051);

    {
        SCOPED_TRACE("Per-tensor quantize");
        testLayer("conv3d", "TensorFlow", 0.00734, 0.02434, 1, 1, false, true, false, false);
        testLayer("conv3d", "ONNX", 0.00377, 0.01362, 1, 1, false, true, false, false);
        testLayer("conv3d_bias", "ONNX", 0.00201, 0.0039, 1, 1, false, true, false, false);
    }
}

TEST_P(Test_Int8_layers, Padding)
{
    if (backend == DNN_BACKEND_TIMVX)
        testLayer("padding_valid", "TensorFlow", 0.0292, 0.0105);
    else
        testLayer("padding_valid", "TensorFlow", 0.0026, 0.0064);

    if (backend == DNN_BACKEND_TIMVX)
        testLayer("padding_same", "TensorFlow", 0.0085, 0.032);
    else
        testLayer("padding_same", "TensorFlow", 0.0081, 0.032);

    if (backend == DNN_BACKEND_TIMVX)
        testLayer("spatial_padding", "TensorFlow", 0.0079, 0.028);
    else
        testLayer("spatial_padding", "TensorFlow", 0.0078, 0.028);

    testLayer("mirror_pad", "TensorFlow", 0.0064, 0.013);
    testLayer("pad_and_concat", "TensorFlow", 0.0021, 0.0098);
    testLayer("padding", "ONNX", 0.0005, 0.0069);
    testLayer("ReflectionPad2d", "ONNX", 0.00062, 0.0018);
    testLayer("ZeroPad2d", "ONNX", 0.00037, 0.0018);
}

TEST_P(Test_Int8_layers, AvePooling)
{
    // Some tests failed with OpenVINO due to wrong padded area calculation
    if (backend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        testLayer("layer_pooling_ave", "Caffe", 0.0021, 0.0075);
    testLayer("ave_pool_same", "TensorFlow", 0.00153, 0.0041);
    testLayer("average_pooling_1d", "ONNX", 0.002, 0.0048);
    if (backend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        testLayer("average_pooling", "ONNX", 0.0014, 0.0032);
    testLayer("average_pooling_dynamic_axes", "ONNX", 0.0014, 0.006);

    if (target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");
    testLayer("ave_pool3d", "TensorFlow", 0.00175, 0.0047);
    testLayer("ave_pool3d", "ONNX", 0.00063, 0.0016);
}

TEST_P(Test_Int8_layers, MaxPooling)
{
    testLayer("pool_conv_1d", "ONNX", 0.0006, 0.0015);
    if (target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");
    testLayer("pool_conv_3d", "ONNX", 0.0033, 0.0124);

    testLayer("layer_pooling_max", "Caffe", 0.0021, 0.004);
    testLayer("max_pool_even", "TensorFlow", 0.0048, 0.0139);
    testLayer("max_pool_odd_valid", "TensorFlow", 0.0043, 0.012);
    testLayer("conv_pool_nchw", "TensorFlow", 0.007, 0.025);
    testLayer("max_pool3d", "TensorFlow", 0.0025, 0.0058);
    testLayer("maxpooling_1d", "ONNX", 0.0018, 0.0037);
    testLayer("two_maxpooling_1d", "ONNX", 0.0037, 0.0052);
    testLayer("maxpooling", "ONNX", 0.0034, 0.0065);
    testLayer("two_maxpooling", "ONNX", 0.0025, 0.0052);
    testLayer("max_pool3d", "ONNX", 0.0028, 0.0069);
}

TEST_P(Test_Int8_layers, Reduce)
{
    testLayer("reduce_mean", "TensorFlow", 0.0005, 0.0014);
    testLayer("reduce_mean", "ONNX", 0.00062, 0.0014);
    testLayer("reduce_mean_axis1", "ONNX", 0.00032, 0.0007);
    testLayer("reduce_mean_axis2", "ONNX", 0.00033, 0.001);

    testLayer("reduce_sum", "TensorFlow", 0.015, 0.031);
    testLayer("reduce_sum_channel", "TensorFlow", 0.008, 0.019);
    testLayer("sum_pool_by_axis", "TensorFlow", 0.012, 0.032);
    testLayer("reduce_sum", "ONNX", 0.0025, 0.0048);

    testLayer("reduce_max", "ONNX", 0, 0);
    testLayer("reduce_max_axis_0", "ONNX", 0.0042, 0.007);
    testLayer("reduce_max_axis_1", "ONNX", 0.0018, 0.0036);

    if (target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");
    testLayer("reduce_mean3d", "ONNX", 0.00048, 0.0016);
}

TEST_P(Test_Int8_layers, ReLU)
{
    testLayer("layer_relu", "Caffe", 0.0005, 0.002);
    testLayer("ReLU", "ONNX", 0.0012, 0.0047);
}

TEST_P(Test_Int8_layers, LeakyReLU)
{
    testLayer("leaky_relu", "TensorFlow", 0.0002, 0.0004);
}

TEST_P(Test_Int8_layers, ReLU6)
{
    testLayer("keras_relu6", "TensorFlow", 0.0018, 0.0062);
    testLayer("keras_relu6", "TensorFlow", 0.0018, 0.0062, 1, 1, false, true, true);
    testLayer("clip_by_value", "TensorFlow", 0.0009, 0.002);
    testLayer("clip", "ONNX", 0.00006, 0.00037);
}

TEST_P(Test_Int8_layers, Sigmoid)
{
    testLayer("maxpooling_sigmoid", "ONNX", 0.0011, 0.0032);
}

TEST_P(Test_Int8_layers, Sigmoid_dynamic_axes)
{
    testLayer("maxpooling_sigmoid_dynamic_axes", "ONNX", 0.002, 0.0032);
}

TEST_P(Test_Int8_layers, Sigmoid_1d)
{
    testLayer("maxpooling_sigmoid_1d", "ONNX", 0.002, 0.0037);
}

TEST_P(Test_Int8_layers, Mish)
{
    testLayer("mish", "ONNX", 0.0015, 0.0025);
}

TEST_P(Test_Int8_layers, Softmax_Caffe)
{
    testLayer("layer_softmax", "Caffe", 0.0011, 0.0036);
}
TEST_P(Test_Int8_layers, Softmax_keras_TF)
{
    testLayer("keras_softmax", "TensorFlow", 0.00093, 0.0027);
}
TEST_P(Test_Int8_layers, Softmax_slim_TF)
{
    testLayer("slim_softmax", "TensorFlow", 0.0016, 0.0034);
}
TEST_P(Test_Int8_layers, Softmax_slim_v2_TF)
{
    testLayer("slim_softmax_v2", "TensorFlow", 0.0029, 0.017);
}
TEST_P(Test_Int8_layers, Softmax_ONNX)
{
    testLayer("softmax", "ONNX", 0.0016, 0.0028);
}
TEST_P(Test_Int8_layers, Softmax_log_ONNX)
{
    testLayer("log_softmax", "ONNX", 0.014, 0.025);
}
TEST_P(Test_Int8_layers, DISABLED_Softmax_unfused_ONNX)  // FIXIT Support 'Identity' layer for outputs (#22022)
{
    testLayer("softmax_unfused", "ONNX", 0.0009, 0.0021);
}

TEST_P(Test_Int8_layers, Concat)
{
    testLayer("layer_concat_shared_input", "Caffe", 0.0076, 0.029, 1, 1, true, false);
    if (backend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
        // Crashes with segfault
        testLayer("concat_axis_1", "TensorFlow", 0.0056, 0.017);
    }
    testLayer("keras_pad_concat", "TensorFlow", 0.0032, 0.0089);
    testLayer("concat_3d", "TensorFlow", 0.005, 0.014);
    testLayer("concatenation", "ONNX", 0.0032, 0.009);
}

TEST_P(Test_Int8_layers, BatchNorm)
{
    testLayer("layer_batch_norm", "Caffe", 0.0061, 0.019, 1, 1, true);
    testLayer("fused_batch_norm", "TensorFlow", 0.0063, 0.02);
    testLayer("batch_norm_text", "TensorFlow", 0.0048, 0.013, 1, 1, false, true, true);
    testLayer("unfused_batch_norm", "TensorFlow", 0.0076, 0.019);
    testLayer("fused_batch_norm_no_gamma", "TensorFlow", 0.0067, 0.015);
    testLayer("unfused_batch_norm_no_gamma", "TensorFlow", 0.0123, 0.044);
    testLayer("switch_identity", "TensorFlow", 0.0035, 0.011);
    testLayer("batch_norm3d", "TensorFlow", 0.0077, 0.02);
    testLayer("batch_norm", "ONNX", 0.0012, 0.0049);
    testLayer("batch_norm_3d", "ONNX", 0.0039, 0.012);
    testLayer("frozenBatchNorm2d", "ONNX", 0.001, 0.0018);
    testLayer("batch_norm_subgraph", "ONNX", 0.0049, 0.0098);
}

TEST_P(Test_Int8_layers, Scale)
{
    testLayer("batch_norm", "TensorFlow", 0.0028, 0.0098);
    testLayer("scale", "ONNX", 0.0025, 0.0071);
    testLayer("expand_hw", "ONNX", 0.0012, 0.0012);
    testLayer("flatten_const", "ONNX", 0.0024, 0.0048);
}

TEST_P(Test_Int8_layers, InnerProduct)
{
    testLayer("layer_inner_product", "Caffe", 0.005, 0.02, 1, 1, true);
    testLayer("matmul", "TensorFlow", 0.0061, 0.019);

    if (backend == DNN_BACKEND_TIMVX)
        testLayer("nhwc_transpose_reshape_matmul", "TensorFlow", 0.0018, 0.0175);
    else
        testLayer("nhwc_transpose_reshape_matmul", "TensorFlow", 0.0009, 0.0091);

    testLayer("nhwc_reshape_matmul", "TensorFlow", 0.03, 0.071);
    testLayer("matmul_layout", "TensorFlow", 0.035, 0.06);
    testLayer("tf2_dense", "TensorFlow", 0, 0);
    testLayer("matmul_add", "ONNX", 0.041, 0.082);
    testLayer("linear", "ONNX", 0.0027, 0.0046);

    if (backend == DNN_BACKEND_TIMVX)
        testLayer("constant", "ONNX", 0.00048, 0.0013);
    else
        testLayer("constant", "ONNX", 0.00021, 0.0006);

    testLayer("lin_with_constant", "ONNX", 0.0011, 0.0016);

    {
        SCOPED_TRACE("Per-tensor quantize");
        testLayer("layer_inner_product", "Caffe", 0.0055, 0.02, 1, 1, true, true, false, false);
        testLayer("matmul", "TensorFlow", 0.0075, 0.019, 1, 1, false, true, false, false);
        testLayer("nhwc_transpose_reshape_matmul", "TensorFlow", 0.0009, 0.0091, 1, 1, false, true, false, false);
        testLayer("nhwc_reshape_matmul", "TensorFlow", 0.037, 0.071, 1, 1, false, true, false, false);
        testLayer("matmul_layout", "TensorFlow", 0.035, 0.095, 1, 1, false, true, false, false);
        testLayer("tf2_dense", "TensorFlow", 0, 0, 1, 1, false, true, false, false);
        testLayer("matmul_add", "ONNX", 0.041, 0.082, 1, 1, false, true, false, false);
        testLayer("linear", "ONNX", 0.0027, 0.005, 1, 1, false, true, false, false);
        testLayer("constant", "ONNX", 0.00038, 0.0012, 1, 1, false, true, false, false);
        testLayer("lin_with_constant", "ONNX", 0.0011, 0.0016, 1, 1, false, true, false, false);
    }
}

TEST_P(Test_Int8_layers, Reshape)
{
    testLayer("reshape_layer", "TensorFlow", 0.0032, 0.0082);

    if (backend == DNN_BACKEND_TIMVX)
        testLayer("reshape_nchw", "TensorFlow", 0.0092, 0.0495);
    else
        testLayer("reshape_nchw", "TensorFlow", 0.0089, 0.029);

    testLayer("reshape_conv", "TensorFlow", 0.035, 0.054);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        testLayer("reshape_reduce", "TensorFlow", 0.0053, 0.011);
    else
        testLayer("reshape_reduce", "TensorFlow", 0.0042, 0.0078);
    testLayer("reshape_as_shape", "TensorFlow", 0.0014, 0.0028);
    testLayer("reshape_no_reorder", "TensorFlow", 0.0014, 0.0028);
    testLayer("shift_reshape_no_reorder", "TensorFlow", 0.0063, backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ? 0.016 : 0.014);
    testLayer("dynamic_reshape", "ONNX", 0.0047, 0.0079);
    testLayer("dynamic_reshape_opset_11", "ONNX", 0.0048, 0.0081);
    testLayer("flatten_by_prod", "ONNX", 0.0048, 0.0081);
    testLayer("squeeze", "ONNX", 0.0048, 0.0081);
    testLayer("unsqueeze", "ONNX", 0.0033, 0.0053);

    if (backend == DNN_BACKEND_TIMVX)
        testLayer("squeeze_and_conv_dynamic_axes", "ONNX", 0.006, 0.0212);
    else
        testLayer("squeeze_and_conv_dynamic_axes", "ONNX", 0.0054, 0.0154);

    testLayer("unsqueeze_and_conv_dynamic_axes", "ONNX", 0.0037, 0.0151);
}

TEST_P(Test_Int8_layers, Permute)
{
    testLayer("tf2_permute_nhwc_ncwh", "TensorFlow", 0.0028, 0.006);
    testLayer("transpose", "ONNX", 0.0015, 0.0046);
}

TEST_P(Test_Int8_layers, Identity)
{
    testLayer("expand_batch", "ONNX", 0.0027, 0.0036);
    testLayer("expand_channels", "ONNX", 0.0013, 0.0019);
    testLayer("expand_neg_batch", "ONNX", 0.00071, 0.0019);
}

TEST_P(Test_Int8_layers, Slice_split_tf)
{
    testLayer("split", "TensorFlow", 0.0033, 0.0056);
}

TEST_P(Test_Int8_layers, Slice_4d_tf)
{
    testLayer("slice_4d", "TensorFlow", 0.003, 0.0073);
}

TEST_P(Test_Int8_layers, Slice_strided_tf)
{
    testLayer("strided_slice", "TensorFlow", 0.008, 0.0142);
}

TEST_P(Test_Int8_layers, DISABLED_Slice_onnx)  // FIXIT Support 'Identity' layer for outputs (#22022)
{
    testLayer("slice", "ONNX", 0.0046, 0.0077);
}

TEST_P(Test_Int8_layers, Slice_dynamic_axes_onnx)
{
    testLayer("slice_dynamic_axes", "ONNX", 0.0039, 0.02);
}

TEST_P(Test_Int8_layers, Slice_steps_2d_onnx11)
{
    testLayer("slice_opset_11_steps_2d", "ONNX", 0.01, 0.0124);
}

TEST_P(Test_Int8_layers, Slice_steps_3d_onnx11)
{
    testLayer("slice_opset_11_steps_3d", "ONNX", 0.0068, 0.014);
}

TEST_P(Test_Int8_layers, Slice_steps_4d_onnx11)
{
    testLayer("slice_opset_11_steps_4d", "ONNX", 0.0041, 0.008);
}

TEST_P(Test_Int8_layers, Slice_steps_5d_onnx11)
{
    testLayer("slice_opset_11_steps_5d", "ONNX", 0.0085, 0.021);
}

TEST_P(Test_Int8_layers, Dropout)
{
    testLayer("layer_dropout", "Caffe", 0.0021, 0.004);
    testLayer("dropout", "ONNX", 0.0029, 0.004);
}

TEST_P(Test_Int8_layers, Eltwise)
{
    testLayer("layer_eltwise", "Caffe", 0.062, 0.15);

    if (backend == DNN_BACKEND_TIMVX)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_TIMVX);

    testLayer("conv_2_inps", "Caffe", 0.0086, 0.0232, 2, 1, true, false);
    testLayer("eltwise_sub", "TensorFlow", 0.015, 0.047);
    testLayer("eltwise_add_vec", "TensorFlow", 0.037, backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ? 0.24 : 0.21); // tflite 0.0095, 0.0365
    testLayer("eltwise_mul_vec", "TensorFlow", 0.173, 1.14); // tflite 0.0028, 0.017
    testLayer("channel_broadcast", "TensorFlow", 0.0025, 0.0063);
    testLayer("split_equals", "TensorFlow", backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ? 0.021 : 0.02, 0.065);
    testLayer("mul", "ONNX", 0.0039, 0.014);
    testLayer("split_max", "ONNX", 0.004, 0.012);
}

TEST_P(Test_Int8_layers, DepthSpaceOps) {
    auto test_layer_with_onnx_conformance_models = [&](const std::string &model_name, double l1, double lInf) {
        std::string model_path = _tf("onnx/conformance/node/test_" + model_name + "/model.onnx");
        auto net = readNet(model_path);

        // load reference inputs and outputs
        std::string data_base_path = _tf("onnx/conformance/node/test_" + model_name + "/test_data_set_0");
        Mat input = readTensorFromONNX(data_base_path + "/input_0.pb");
        Mat ref_output = readTensorFromONNX(data_base_path + "/output_0.pb");

        std::vector<float> input_scales, output_scales;
        std::vector<int> input_zeropoints, output_zeropoints;
        auto qnet = net.quantize(std::vector<Mat>{input}, CV_8S, CV_8S, false);
        qnet.getInputDetails(input_scales, input_zeropoints);
        qnet.getOutputDetails(output_scales, output_zeropoints);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);

        Mat quantized_input, quantized_output;
        input.convertTo(quantized_input, CV_8S, 1.f / input_scales.front(), input_zeropoints.front());
        qnet.setInput(quantized_input);
        quantized_output = qnet.forward();

        Mat output;
        quantized_output.convertTo(output, CV_32F, output_scales.front(), -(output_scales.front() * output_zeropoints.front()));
        normAssert(ref_output, output, model_name.c_str(), l1, lInf);
    };

    double l1 = default_l1, lInf = default_lInf;
    {
        l1 = 0.001; lInf = 0.002;
        if (backend == DNN_BACKEND_TIMVX) { l1 = 0.001; lInf = 0.002; }
        test_layer_with_onnx_conformance_models("spacetodepth", l1, lInf);
    }
    {
        l1 = 0.022; lInf = 0.044;
        if (backend == DNN_BACKEND_TIMVX) { l1 = 0.022; lInf = 0.044; }
        test_layer_with_onnx_conformance_models("spacetodepth_example", l1, lInf);
    }
    {
        l1 = 0.001; lInf = 0.002;
        if (backend == DNN_BACKEND_TIMVX) { l1 = 0.24; lInf = 0.99; }
        test_layer_with_onnx_conformance_models("depthtospace_crd_mode", l1, lInf);
    }
    test_layer_with_onnx_conformance_models("depthtospace_dcr_mode", 0.001, 0.002);
    test_layer_with_onnx_conformance_models("depthtospace_example", 0.07, 0.14);

    {
        l1 = 0.07; lInf = 0.14;
        if (backend == DNN_BACKEND_TIMVX) // diff too huge, l1 = 13.6; lInf = 27.2
            applyTestTag(CV_TEST_TAG_DNN_SKIP_TIMVX);
        test_layer_with_onnx_conformance_models("depthtospace_crd_mode_example", l1, lInf);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Int8_layers, dnnBackendsAndTargetsInt8());

class Test_Int8_nets : public DNNTestLayer
{
public:
    void testClassificationNet(Net baseNet, const Mat& blob, const Mat& ref, double l1, double lInf, bool perChannel = true)
    {
        Net qnet = baseNet.quantize(blob, CV_32F, CV_32F, perChannel);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);

        qnet.setInput(blob);
        Mat out = qnet.forward();
        normAssert(ref, out, "", l1, lInf);
    }

    void testDetectionNet(Net baseNet, const Mat& blob, const Mat& ref,
                          double confThreshold, double scoreDiff, double iouDiff, bool perChannel = true)
    {
        Net qnet = baseNet.quantize(blob, CV_32F, CV_32F, perChannel);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);

        qnet.setInput(blob);
        Mat out = qnet.forward();
        normAssertDetections(ref, out, "", confThreshold, scoreDiff, iouDiff);
    }

    void testFaster(Net baseNet, const Mat& ref, double confThreshold, double scoreDiff, double iouDiff, bool perChannel = true)
    {
        Mat inp = imread(_tf("dog416.png"));
        resize(inp, inp, Size(800, 600));
        Mat blob = blobFromImage(inp, 1.0, Size(), Scalar(102.9801, 115.9465, 122.7717), false, false);
        Mat imInfo = (Mat_<float>(1, 3) << inp.rows, inp.cols, 1.6f);

        Net qnet = baseNet.quantize(std::vector<Mat>{blob, imInfo}, CV_32F, CV_32F, perChannel);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);

        qnet.setInput(blob, "data");
        qnet.setInput(imInfo, "im_info");
        Mat out = qnet.forward();
        normAssertDetections(ref, out, "", confThreshold, scoreDiff, iouDiff);
    }

    void testONNXNet(const String& basename, double l1, double lInf, bool useSoftmax = false, bool perChannel = true)
    {
        String onnxmodel = findDataFile("dnn/onnx/models/" + basename + ".onnx", false);

        Mat blob = readTensorFromONNX(findDataFile("dnn/onnx/data/input_" + basename + ".pb"));
        Mat ref = readTensorFromONNX(findDataFile("dnn/onnx/data/output_" + basename + ".pb"));
        Net baseNet = readNetFromONNX(onnxmodel);

        Net qnet = baseNet.quantize(blob, CV_32F, CV_32F, perChannel);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);
        qnet.setInput(blob);
        Mat out = qnet.forward();

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

        normAssert(ref, out, "", l1, lInf);
    }

    void testDarknetModel(const std::string& cfg, const std::string& weights,
                          const cv::Mat& ref, double scoreDiff, double iouDiff,
                          float confThreshold = 0.24, float nmsThreshold = 0.4, bool perChannel = true)
    {
        CV_Assert(ref.cols == 7);
        std::vector<std::vector<int> > refClassIds;
        std::vector<std::vector<float> > refScores;
        std::vector<std::vector<Rect2d> > refBoxes;
        for (int i = 0; i < ref.rows; ++i)
        {
            int batchId = static_cast<int>(ref.at<float>(i, 0));
            int classId = static_cast<int>(ref.at<float>(i, 1));
            float score = ref.at<float>(i, 2);
            float left  = ref.at<float>(i, 3);
            float top   = ref.at<float>(i, 4);
            float right  = ref.at<float>(i, 5);
            float bottom = ref.at<float>(i, 6);
            Rect2d box(left, top, right - left, bottom - top);
            if (batchId >= refClassIds.size())
            {
                refClassIds.resize(batchId + 1);
                refScores.resize(batchId + 1);
                refBoxes.resize(batchId + 1);
            }
            refClassIds[batchId].push_back(classId);
            refScores[batchId].push_back(score);
            refBoxes[batchId].push_back(box);
        }

        Mat img1 = imread(_tf("dog416.png"));
        Mat img2 = imread(_tf("street.png"));
        std::vector<Mat> samples(2);
        samples[0] = img1; samples[1] = img2;

        // determine test type, whether batch or single img
        int batch_size = refClassIds.size();
        CV_Assert(batch_size == 1 || batch_size == 2);
        samples.resize(batch_size);

        Mat inp = blobFromImages(samples, 1.0/255, Size(416, 416), Scalar(), true, false);

        Net baseNet = readNetFromDarknet(findDataFile("dnn/" + cfg), findDataFile("dnn/" + weights, false));
        Net qnet = baseNet.quantize(inp, CV_32F, CV_32F, perChannel);
        qnet.setPreferableBackend(backend);
        qnet.setPreferableTarget(target);
        qnet.setInput(inp);
        std::vector<Mat> outs;
        qnet.forward(outs, qnet.getUnconnectedOutLayersNames());

        for (int b = 0; b < batch_size; ++b)
        {
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect2d> boxes;
            for (int i = 0; i < outs.size(); ++i)
            {
                Mat out;
                if (batch_size > 1){
                    // get the sample slice from 3D matrix (batch, box, classes+5)
                    Range ranges[3] = {Range(b, b+1), Range::all(), Range::all()};
                    out = outs[i](ranges).reshape(1, outs[i].size[1]);
                }else{
                    out = outs[i];
                }
                for (int j = 0; j < out.rows; ++j)
                {
                    Mat scores = out.row(j).colRange(5, out.cols);
                    double confidence;
                    Point maxLoc;
                    minMaxLoc(scores, 0, &confidence, 0, &maxLoc);

                    if (confidence > confThreshold) {
                        float* detection = out.ptr<float>(j);
                        double centerX = detection[0];
                        double centerY = detection[1];
                        double width = detection[2];
                        double height = detection[3];
                        boxes.push_back(Rect2d(centerX - 0.5 * width, centerY - 0.5 * height,
                                            width, height));
                        confidences.push_back(confidence);
                        classIds.push_back(maxLoc.x);
                    }
                }
            }

            // here we need NMS of boxes
            std::vector<int> indices;
            NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

            std::vector<int> nms_classIds;
            std::vector<float> nms_confidences;
            std::vector<Rect2d> nms_boxes;

            for (size_t i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];
                Rect2d box = boxes[idx];
                float conf = confidences[idx];
                int class_id = classIds[idx];
                nms_boxes.push_back(box);
                nms_confidences.push_back(conf);
                nms_classIds.push_back(class_id);
            }

            if (cvIsNaN(iouDiff))
            {
                if (b == 0)
                    std::cout << "Skip accuracy checks" << std::endl;
                continue;
            }

            normAssertDetections(refClassIds[b], refScores[b], refBoxes[b], nms_classIds, nms_confidences, nms_boxes,
                                 format("batch size %d, sample %d\n", batch_size, b).c_str(), confThreshold, scoreDiff, iouDiff);
        }
    }
};

TEST_P(Test_Int8_nets, AlexNet)
{
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);


    Net net = readNetFromCaffe(findDataFile("dnn/bvlc_alexnet.prototxt"),
                               findDataFile("dnn/bvlc_alexnet.caffemodel", false));

    Mat inp = imread(_tf("grace_hopper_227.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(227, 227), Scalar(), false);
    Mat ref = blobFromNPY(_tf("caffe_alexnet_prob.npy"));

    float l1 = 1e-4, lInf = 0.003;
    testClassificationNet(net, blob, ref, l1, lInf);
}

TEST_P(Test_Int8_nets, GoogLeNet)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromCaffe(findDataFile("dnn/bvlc_googlenet.prototxt"),
                               findDataFile("dnn/bvlc_googlenet.caffemodel", false));

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(_tf("googlenet_0.png")) );
    inpMats.push_back( imread(_tf("googlenet_1.png")) );
    Mat blob = blobFromImages(inpMats, 1.0, Size(224, 224), Scalar(), false);
    Mat ref = blobFromNPY(_tf("googlenet_prob.npy"));

    float l1 = 2e-4, lInf = 0.07;
    testClassificationNet(net, blob, ref, l1, lInf);
}

TEST_P(Test_Int8_nets, ResNet50)
{
    applyTestTag(
        target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromCaffe(findDataFile("dnn/ResNet-50-deploy.prototxt"),
                               findDataFile("dnn/ResNet-50-model.caffemodel", false));

    Mat inp = imread(_tf("googlenet_0.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(224, 224), Scalar(), false);
    Mat ref = blobFromNPY(_tf("resnet50_prob.npy"));

    float l1 = 3e-4, lInf = 0.05;
    testClassificationNet(net, blob, ref, l1, lInf);

    {
        SCOPED_TRACE("Per-tensor quantize");
        testClassificationNet(net, blob, ref, l1, lInf, false);
    }
}

TEST_P(Test_Int8_nets, DenseNet121)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    Net net = readNetFromCaffe(findDataFile("dnn/DenseNet_121.prototxt", false),
                               findDataFile("dnn/DenseNet_121.caffemodel", false));

    Mat inp = imread(_tf("dog416.png"));
    Mat blob = blobFromImage(inp, 1.0 / 255.0, Size(224, 224), Scalar(), true, true);
    Mat ref = blobFromNPY(_tf("densenet_121_output.npy"));

    float l1 = 0.76, lInf = 3.31; // seems wrong
    testClassificationNet(net, blob, ref, l1, lInf);
}

TEST_P(Test_Int8_nets, SqueezeNet_v1_1)
{
    if(target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromCaffe(findDataFile("dnn/squeezenet_v1.1.prototxt"),
                               findDataFile("dnn/squeezenet_v1.1.caffemodel", false));

    Mat inp = imread(_tf("googlenet_0.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(227, 227), Scalar(), false, true);
    Mat ref = blobFromNPY(_tf("squeezenet_v1.1_prob.npy"));

    float l1 = 3e-4, lInf = 0.056;
    testClassificationNet(net, blob, ref, l1, lInf);
}

TEST_P(Test_Int8_nets, CaffeNet)
{
#if defined(OPENCV_32BIT_CONFIGURATION) && (defined(HAVE_OPENCL) || defined(_WIN32))
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    float l1 = 4e-5, lInf = 0.0025;
    testONNXNet("caffenet", l1, lInf);
}

TEST_P(Test_Int8_nets, RCNN_ILSVRC13)
{
#if defined(OPENCV_32BIT_CONFIGURATION) && (defined(HAVE_OPENCL) || defined(_WIN32))
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#else
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#endif

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    float l1 = 0.02, lInf = 0.047;
    testONNXNet("rcnn_ilsvrc13", l1, lInf);
}

TEST_P(Test_Int8_nets, Inception_v2)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    testONNXNet("inception_v2",  default_l1,  default_lInf, true);
}

TEST_P(Test_Int8_nets, MobileNet_v2)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    testONNXNet("mobilenetv2", default_l1, default_lInf, true);
}

TEST_P(Test_Int8_nets, Shufflenet)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    testONNXNet("shufflenet", default_l1, default_lInf);
}

TEST_P(Test_Int8_nets, MobileNet_SSD)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromCaffe(findDataFile("dnn/MobileNetSSD_deploy_19e3ec3.prototxt", false),
                               findDataFile("dnn/MobileNetSSD_deploy_19e3ec3.caffemodel", false));

    Mat inp = imread(_tf("street.png"));
    Mat blob = blobFromImage(inp, 1.0 / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    Mat ref = blobFromNPY(_tf("mobilenet_ssd_caffe_out.npy"));

    float confThreshold = FLT_MIN, scoreDiff = 0.084, iouDiff = 0.43;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, MobileNet_v1_SSD)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromTensorflow(findDataFile("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", false),
                                    findDataFile("dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt"));

    Mat inp = imread(_tf("dog416.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(300, 300), Scalar(), true, false);
    Mat ref = blobFromNPY(_tf("tensorflow/ssd_mobilenet_v1_coco_2017_11_17.detection_out.npy"));

    float confThreshold = 0.5, scoreDiff = 0.034, iouDiff = 0.13;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, MobileNet_v1_SSD_PPN)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromTensorflow(findDataFile("dnn/ssd_mobilenet_v1_ppn_coco.pb", false),
                                    findDataFile("dnn/ssd_mobilenet_v1_ppn_coco.pbtxt"));

    Mat inp = imread(_tf("dog416.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(300, 300), Scalar(), true, false);
    Mat ref = blobFromNPY(_tf("tensorflow/ssd_mobilenet_v1_ppn_coco.detection_out.npy"));

    float confThreshold = 0.51, scoreDiff = 0.05, iouDiff = 0.06;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, Inception_v2_SSD)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);

    Net net = readNetFromTensorflow(findDataFile("dnn/ssd_inception_v2_coco_2017_11_17.pb", false),
                                    findDataFile("dnn/ssd_inception_v2_coco_2017_11_17.pbtxt"));

    Mat inp = imread(_tf("street.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(300, 300), Scalar(), true, false);
    Mat ref = (Mat_<float>(5, 7) << 0, 1, 0.90176028, 0.19872092, 0.36311883, 0.26461923, 0.63498729,
                                    0, 3, 0.93569964, 0.64865261, 0.45906419, 0.80675775, 0.65708131,
                                    0, 3, 0.75838411, 0.44668293, 0.45907149, 0.49459291, 0.52197015,
                                    0, 10, 0.95932811, 0.38349164, 0.32528657, 0.40387636, 0.39165527,
                                    0, 10, 0.93973452, 0.66561931, 0.37841269, 0.68074018, 0.42907384);

    float confThreshold = 0.5, scoreDiff = 0.0114, iouDiff = 0.22;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, opencv_face_detector)
{
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    Net net = readNetFromCaffe(findDataFile("dnn/opencv_face_detector.prototxt"),
                               findDataFile("dnn/opencv_face_detector.caffemodel", false));

    Mat inp = imread(findDataFile("gpu/lbpcascade/er.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);
    Mat ref = (Mat_<float>(6, 7) << 0, 1, 0.99520785, 0.80997437, 0.16379407, 0.87996572, 0.26685631,
                                    0, 1, 0.9934696, 0.2831718, 0.50738752, 0.345781, 0.5985168,
                                    0, 1, 0.99096733, 0.13629119, 0.24892329, 0.19756334, 0.3310290,
                                    0, 1, 0.98977017, 0.23901358, 0.09084064, 0.29902688, 0.1769477,
                                    0, 1, 0.97203469, 0.67965847, 0.06876482, 0.73999709, 0.1513494,
                                    0, 1, 0.95097077, 0.51901293, 0.45863652, 0.5777427, 0.5347801);

    float confThreshold = 0.5, scoreDiff = 0.002, iouDiff = 0.4;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, EfficientDet)
{
    if (cvtest::skipUnstableTests)
        throw SkipTestException("Skip unstable test");  // detail: https://github.com/opencv/opencv/pull/23167

    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);
    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_TIMVX)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_TIMVX);

    if (target != DNN_TARGET_CPU)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
    }
    Net net = readNetFromTensorflow(findDataFile("dnn/efficientdet-d0.pb", false),
                                    findDataFile("dnn/efficientdet-d0.pbtxt"));

    Mat inp = imread(_tf("dog416.png"));
    Mat blob = blobFromImage(inp, 1.0/255, Size(512, 512), Scalar(123.675, 116.28, 103.53));
    Mat ref = (Mat_<float>(3, 7) << 0, 1, 0.8437444, 0.153996080160141, 0.20534580945968628, 0.7463544607162476, 0.7414066195487976,
                                    0, 17, 0.8245924, 0.16657517850399017, 0.3996818959712982, 0.4111558794975281, 0.9306337833404541,
                                    0, 7, 0.8039304, 0.6118435263633728, 0.13175517320632935, 0.9065558314323425, 0.2943994700908661);

    float confThreshold = 0.65, scoreDiff = 0.3, iouDiff = 0.18;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);

    {
        SCOPED_TRACE("Per-tensor quantize");
        testDetectionNet(net, blob, ref, 0.85, scoreDiff, iouDiff, false);
    }
}

TEST_P(Test_Int8_nets, FasterRCNN_resnet50)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    Net net = readNetFromTensorflow(findDataFile("dnn/faster_rcnn_resnet50_coco_2018_01_28.pb", false),
                                    findDataFile("dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt"));

    Mat inp = imread(_tf("dog416.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(800, 600), Scalar(), true, false);
    Mat ref = blobFromNPY(_tf("tensorflow/faster_rcnn_resnet50_coco_2018_01_28.detection_out.npy"));

    float confThreshold = 0.8, scoreDiff = 0.05, iouDiff = 0.15;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, FasterRCNN_inceptionv2)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    Net net = readNetFromTensorflow(findDataFile("dnn/faster_rcnn_inception_v2_coco_2018_01_28.pb", false),
                                    findDataFile("dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"));

    Mat inp = imread(_tf("dog416.png"));
    Mat blob = blobFromImage(inp, 1.0, Size(800, 600), Scalar(), true, false);
    Mat ref = blobFromNPY(_tf("tensorflow/faster_rcnn_inception_v2_coco_2018_01_28.detection_out.npy"));

    float confThreshold = 0.5, scoreDiff = 0.21, iouDiff = 0.1;
    testDetectionNet(net, blob, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, FasterRCNN_vgg16)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        CV_TEST_TAG_MEMORY_2GB,
#endif
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    Net net = readNetFromCaffe(findDataFile("dnn/faster_rcnn_vgg16.prototxt"),
                               findDataFile("dnn/VGG16_faster_rcnn_final.caffemodel", false));

    Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.949398, 99.2454, 210.141, 601.205, 462.849,
                                    0, 7, 0.997022, 481.841, 92.3218, 722.685, 175.953,
                                    0, 12, 0.993028, 133.221, 189.377, 350.994, 563.166);

    float confThreshold = 0.8, scoreDiff = 0.048, iouDiff = 0.35;
    testFaster(net, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, FasterRCNN_zf)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
#endif
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    Net net = readNetFromCaffe(findDataFile("dnn/faster_rcnn_zf.prototxt"),
                               findDataFile("dnn/ZF_faster_rcnn_final.caffemodel", false));

    Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.90121, 120.407, 115.83, 570.586, 528.395,
                                    0, 7, 0.988779, 469.849, 75.1756, 718.64, 186.762,
                                    0, 12, 0.967198, 138.588, 206.843, 329.766, 553.176);

    float confThreshold = 0.8, scoreDiff = 0.021, iouDiff = 0.1;
    testFaster(net, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, RFCN)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_2GB),
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Net net = readNetFromCaffe(findDataFile("dnn/rfcn_pascal_voc_resnet50.prototxt"),
                               findDataFile("dnn/resnet50_rfcn_final.caffemodel", false));

    Mat ref = (Mat_<float>(2, 7) << 0, 7, 0.991359, 491.822, 81.1668, 702.573, 178.234,
                                    0, 12, 0.94786, 132.093, 223.903, 338.077, 566.16);

    float confThreshold = 0.8, scoreDiff = 0.15, iouDiff = 0.11;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
        iouDiff = 0.12;
    }
    testFaster(net, ref, confThreshold, scoreDiff, iouDiff);
}

TEST_P(Test_Int8_nets, YoloVoc)
{
    applyTestTag(
#if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
        CV_TEST_TAG_MEMORY_2GB,
#else
        CV_TEST_TAG_MEMORY_1GB,
#endif
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Mat ref = (Mat_<float>(6, 7) << 0, 6,  0.750469f, 0.577374f, 0.127391f, 0.902949f, 0.300809f,
                                    0, 1,  0.780879f, 0.270762f, 0.264102f, 0.732475f, 0.745412f,
                                    0, 11, 0.901615f, 0.1386f,   0.338509f, 0.421337f, 0.938789f,
                                    1, 14, 0.623813f, 0.183179f, 0.381921f, 0.247726f, 0.625847f,
                                    1, 6,  0.667770f, 0.446555f, 0.453578f, 0.499986f, 0.519167f,
                                    1, 6,  0.844947f, 0.637058f, 0.460398f, 0.828508f, 0.66427f);

    std::string config_file = "yolo-voc.cfg";
    std::string weights_file = "yolo-voc.weights";

    double scoreDiff = 0.12, iouDiff = 0.3;
    {
    SCOPED_TRACE("batch size 1");
    testDarknetModel(config_file, weights_file, ref.rowRange(0, 3), scoreDiff, iouDiff);
    }

    {
    SCOPED_TRACE("batch size 2");
    testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff);
    }
}

TEST_P(Test_Int8_nets, TinyYoloVoc)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_512MB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    Mat ref = (Mat_<float>(4, 7) << 0, 6,  0.761967f, 0.579042f, 0.159161f, 0.894482f, 0.31994f,
                                    0, 11, 0.780595f, 0.129696f, 0.386467f, 0.445275f, 0.920994f,
                                    1, 6,  0.651450f, 0.460526f, 0.458019f, 0.522527f, 0.5341f,
                                    1, 6,  0.928758f, 0.651024f, 0.463539f, 0.823784f, 0.654998f);

    std::string config_file = "tiny-yolo-voc.cfg";
    std::string weights_file = "tiny-yolo-voc.weights";

    double scoreDiff = 0.043, iouDiff = 0.12;
    {
    SCOPED_TRACE("batch size 1");
    testDarknetModel(config_file, weights_file, ref.rowRange(0, 2), scoreDiff, iouDiff);
        {
            SCOPED_TRACE("Per-tensor quantize");
            testDarknetModel(config_file, weights_file, ref.rowRange(0, 2), 0.1, 0.2, 0.24, 0.6, false);
        }
    }

    {
    SCOPED_TRACE("batch size 2");
    testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff);

        {
            SCOPED_TRACE("Per-tensor quantize");
            testDarknetModel(config_file, weights_file, ref, 0.1, 0.2, 0.24, 0.6, false);
        }
    }
}

TEST_P(Test_Int8_nets, YOLOv3)
{
    applyTestTag(
            CV_TEST_TAG_LONG,
            (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
            CV_TEST_TAG_DEBUG_VERYLONG
            );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    const int N0 = 3;
    const int N1 = 6;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.998836f, 0.160024f, 0.389964f, 0.417885f, 0.943716f,
0, 1, 0.987908f, 0.150913f, 0.221933f, 0.742255f, 0.746261f,
0, 7, 0.952983f, 0.614621f, 0.150257f, 0.901368f, 0.289251f,

1, 2, 0.997412f, 0.647584f, 0.459939f, 0.821037f, 0.663947f,
1, 2, 0.989633f, 0.450719f, 0.463353f, 0.496306f, 0.522258f,
1, 0, 0.980053f, 0.195856f, 0.378454f, 0.258626f, 0.629257f,
1, 9, 0.785341f, 0.665503f, 0.373543f, 0.688893f, 0.439244f,
1, 9, 0.733275f, 0.376029f, 0.315694f, 0.401776f, 0.395165f,
1, 9, 0.384815f, 0.659824f, 0.372389f, 0.673927f, 0.429412f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    std::string config_file = "yolov3.cfg";
    std::string weights_file = "yolov3.weights";

    double scoreDiff = 0.08, iouDiff = 0.21, confThreshold = 0.4;
    {
        SCOPED_TRACE("batch size 1");
        testDarknetModel(config_file, weights_file, ref.rowRange(0, N0), scoreDiff, iouDiff, confThreshold);
    }

    {
        SCOPED_TRACE("batch size 2");
        testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff, confThreshold);
    }
}

TEST_P(Test_Int8_nets, YOLOv4)
{
    applyTestTag(
            CV_TEST_TAG_LONG,
            (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
            CV_TEST_TAG_DEBUG_VERYLONG
            );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);

    const int N0 = 3;
    const int N1 = 7;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.992194f, 0.172375f, 0.402458f, 0.403918f, 0.932801f,
0, 1, 0.988326f, 0.166708f, 0.228236f, 0.737208f, 0.735803f,
0, 7, 0.94639f, 0.602523f, 0.130399f, 0.901623f, 0.298452f,

1, 2, 0.99761f, 0.646556f, 0.45985f, 0.816041f, 0.659067f,
1, 0, 0.988913f, 0.201726f, 0.360282f, 0.266181f, 0.631728f,
1, 2, 0.98233f, 0.452007f, 0.462217f, 0.495612f, 0.521687f,
1, 9, 0.919195f, 0.374642f, 0.316524f, 0.398126f, 0.393714f,
1, 9, 0.856303f, 0.666842f, 0.372215f, 0.685539f, 0.44141f,
1, 9, 0.313516f, 0.656791f, 0.374734f, 0.671959f, 0.438371f,
1, 9, 0.256625f, 0.940232f, 0.326931f, 0.967586f, 0.374002f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    std::string config_file = "yolov4.cfg";
    std::string weights_file = "yolov4.weights";
    double scoreDiff = 0.15, iouDiff = 0.2;
    float confThreshold = 0.26;
    {
        SCOPED_TRACE("batch size 1");
        testDarknetModel(config_file, weights_file, ref.rowRange(0, N0), scoreDiff, iouDiff, confThreshold);
    }

    {
        SCOPED_TRACE("batch size 2");

        testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff, confThreshold);
    }
}

TEST_P(Test_Int8_nets, YOLOv4_tiny)
{
    applyTestTag(
        target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB
    );

    if (target == DNN_TARGET_OPENCL_FP16 && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_OPENCL && !ocl::Device::getDefault().isIntel())
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    const float confThreshold = 0.6;

    const int N0 = 2;
    const int N1 = 3;
    static const float ref_[/* (N0 + N1) * 7 */] = {
0, 16, 0.912199f, 0.169926f, 0.350896f, 0.422704f, 0.941837f,
0, 7, 0.845388f, 0.617568f, 0.13961f, 0.9008f, 0.29315f,

1, 2, 0.997789f, 0.657455f, 0.459714f, 0.809122f, 0.656829f,
1, 2, 0.924423f, 0.442872f, 0.470127f, 0.49816f, 0.516516f,
1, 0, 0.728307f, 0.202607f, 0.369828f, 0.259445f, 0.613846f,
    };
    Mat ref(N0 + N1, 7, CV_32FC1, (void*)ref_);

    std::string config_file = "yolov4-tiny-2020-12.cfg";
    std::string weights_file = "yolov4-tiny-2020-12.weights";
    double scoreDiff = 0.12;
    double iouDiff = target == DNN_TARGET_OPENCL_FP16 ? 0.2 : 0.118;

    {
        SCOPED_TRACE("batch size 1");
        testDarknetModel(config_file, weights_file, ref.rowRange(0, N0), scoreDiff, iouDiff, confThreshold);

        {
            SCOPED_TRACE("Per-tensor quantize");
            testDarknetModel(config_file, weights_file, ref.rowRange(0, N0), scoreDiff, 0.224, 0.7, 0.4, false);
        }
    }

    throw SkipTestException("batch2: bad accuracy on second image");
    /* bad accuracy on second image
    {
        SCOPED_TRACE("batch size 2");
        testDarknetModel(config_file, weights_file, ref, scoreDiff, iouDiff, confThreshold);
    }
    */
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Int8_nets, dnnBackendsAndTargetsInt8());

}} // namespace
