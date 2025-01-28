// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Used in accuracy and perf tests as a content of .cpp file
// Note: don't use "precomp.hpp" here
#include "opencv2/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/dnn.hpp"
#include "test_common.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif  // _WIN32

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void PrintTo(const cv::dnn::Backend& v, std::ostream* os)
{
    switch (v) {
    case DNN_BACKEND_DEFAULT: *os << "DEFAULT"; return;
    case DNN_BACKEND_HALIDE: *os << "HALIDE"; return;
    case DNN_BACKEND_INFERENCE_ENGINE: *os << "DLIE*"; return;
    case DNN_BACKEND_VKCOM: *os << "VKCOM"; return;
    case DNN_BACKEND_OPENCV: *os << "OCV"; return;
    case DNN_BACKEND_CUDA: *os << "CUDA"; return;
    case DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019: *os << "DLIE"; return;
    case DNN_BACKEND_INFERENCE_ENGINE_NGRAPH: *os << "NGRAPH"; return;
    case DNN_BACKEND_WEBNN: *os << "WEBNN"; return;
    case DNN_BACKEND_TIMVX: *os << "TIMVX"; return;
    case DNN_BACKEND_CANN: *os << "CANN"; return;
    case DNN_BACKEND_TENSORRT: *os << "TENSORRT"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_BACKEND_UNKNOWN(" << (int)v << ")";
}

void PrintTo(const cv::dnn::Target& v, std::ostream* os)
{
    switch (v) {
    case DNN_TARGET_CPU: *os << "CPU"; return;
    case DNN_TARGET_OPENCL: *os << "OCL"; return;
    case DNN_TARGET_OPENCL_FP16: *os << "OCL_FP16"; return;
    case DNN_TARGET_MYRIAD: *os << "MYRIAD"; return;
    case DNN_TARGET_HDDL: *os << "HDDL"; return;
    case DNN_TARGET_VULKAN: *os << "VULKAN"; return;
    case DNN_TARGET_FPGA: *os << "FPGA"; return;
    case DNN_TARGET_CUDA: *os << "CUDA"; return;
    case DNN_TARGET_CUDA_FP16: *os << "CUDA_FP16"; return;
    case DNN_TARGET_NPU: *os << "NPU"; return;
    case DNN_TARGET_CPU_FP16: *os << "CPU_FP16"; return;
    case DNN_TARGET_TENSORRT: *os << "TENSORRT"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_TARGET_UNKNOWN(" << (int)v << ")";
}

void PrintTo(const tuple<cv::dnn::Backend, cv::dnn::Target> v, std::ostream* os)
{
    PrintTo(get<0>(v), os);
    *os << "/";
    PrintTo(get<1>(v), os);
}

CV__DNN_INLINE_NS_END
}} // namespace



namespace opencv_test {

void normAssert(
        cv::InputArray ref, cv::InputArray test, const char *comment /*= ""*/,
        double l1 /*= 0.00001*/, double lInf /*= 0.0001*/)
{
    double normL1 = cvtest::norm(ref, test, cv::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment << "  |ref| = " << cvtest::norm(ref, cv::NORM_INF);

    double normInf = cvtest::norm(ref, test, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment << "  |ref| = " << cvtest::norm(ref, cv::NORM_INF);
}

std::vector<cv::Rect2d> matToBoxes(const cv::Mat& m)
{
    EXPECT_EQ(m.type(), CV_32FC1);
    EXPECT_EQ(m.dims, 2);
    EXPECT_EQ(m.cols, 4);

    std::vector<cv::Rect2d> boxes(m.rows);
    for (int i = 0; i < m.rows; ++i)
    {
        CV_Assert(m.row(i).isContinuous());
        const float* data = m.ptr<float>(i);
        double l = data[0], t = data[1], r = data[2], b = data[3];
        boxes[i] = cv::Rect2d(l, t, r - l, b - t);
    }
    return boxes;
}

void normAssertDetections(
        const std::vector<int>& refClassIds,
        const std::vector<float>& refScores,
        const std::vector<cv::Rect2d>& refBoxes,
        const std::vector<int>& testClassIds,
        const std::vector<float>& testScores,
        const std::vector<cv::Rect2d>& testBoxes,
        const char *comment /*= ""*/, double confThreshold /*= 0.0*/,
        double scores_diff /*= 1e-5*/, double boxes_iou_diff /*= 1e-4*/)
{
    ASSERT_FALSE(testClassIds.empty()) << "No detections";
    std::vector<bool> matchedRefBoxes(refBoxes.size(), false);
    std::vector<double> refBoxesIoUDiff(refBoxes.size(), 1.0);
    for (int i = 0; i < testBoxes.size(); ++i)
    {
        //cout << "Test[i=" << i << "]: score=" << testScores[i] << " id=" << testClassIds[i] << " box " << testBoxes[i] << endl;
        double testScore = testScores[i];
        if (testScore < confThreshold)
            continue;

        int testClassId = testClassIds[i];
        const cv::Rect2d& testBox = testBoxes[i];
        bool matched = false;
        double topIoU = 0;
        for (int j = 0; j < refBoxes.size() && !matched; ++j)
        {
            if (!matchedRefBoxes[j] && testClassId == refClassIds[j] &&
                std::abs(testScore - refScores[j]) < scores_diff)
            {
                double interArea = (testBox & refBoxes[j]).area();
                double iou = interArea / (testBox.area() + refBoxes[j].area() - interArea);
                topIoU = std::max(topIoU, iou);
                refBoxesIoUDiff[j] = std::min(refBoxesIoUDiff[j], 1.0f - iou);
                if (1.0 - iou < boxes_iou_diff)
                {
                    matched = true;
                    matchedRefBoxes[j] = true;
                }
            }
        }
        if (!matched)
        {
            std::cout << cv::format("Unmatched prediction: class %d score %f box ",
                                    testClassId, testScore) << testBox << std::endl;
            std::cout << "Highest IoU: " << topIoU << std::endl;
        }
        EXPECT_TRUE(matched) << comment;
    }

    // Check unmatched reference detections.
    for (int i = 0; i < refBoxes.size(); ++i)
    {
        if (!matchedRefBoxes[i] && refScores[i] > confThreshold)
        {
            std::cout << cv::format("Unmatched reference: class %d score %f box ",
                                    refClassIds[i], refScores[i]) << refBoxes[i]
                << " IoU diff: " << refBoxesIoUDiff[i]
                << std::endl;
            EXPECT_LE(refScores[i], confThreshold) << comment;
        }
    }
}

// For SSD-based object detection networks which produce output of shape 1x1xNx7
// where N is a number of detections and an every detection is represented by
// a vector [batchId, classId, confidence, left, top, right, bottom].
void normAssertDetections(
        cv::Mat ref, cv::Mat out, const char *comment /*= ""*/,
        double confThreshold /*= 0.0*/, double scores_diff /*= 1e-5*/,
        double boxes_iou_diff /*= 1e-4*/)
{
    CV_Assert(ref.total() % 7 == 0);
    CV_Assert(out.total() % 7 == 0);
    ref = ref.reshape(1, ref.total() / 7);
    out = out.reshape(1, out.total() / 7);

    cv::Mat refClassIds, testClassIds;
    ref.col(1).convertTo(refClassIds, CV_32SC1);
    out.col(1).convertTo(testClassIds, CV_32SC1);
    std::vector<float> refScores(ref.col(2)), testScores(out.col(2));
    std::vector<cv::Rect2d> refBoxes = matToBoxes(ref.colRange(3, 7));
    std::vector<cv::Rect2d> testBoxes = matToBoxes(out.colRange(3, 7));
    normAssertDetections(refClassIds, refScores, refBoxes, testClassIds, testScores,
                         testBoxes, comment, confThreshold, scores_diff, boxes_iou_diff);
}

// For text detection networks
// Curved text polygon is not supported in the current version.
// (concave polygon is invalid input to intersectConvexConvex)
void normAssertTextDetections(
        const std::vector<std::vector<Point>>& gtPolys,
        const std::vector<std::vector<Point>>& testPolys,
        const char *comment /*= ""*/, double boxes_iou_diff /*= 1e-4*/)
{
    std::vector<bool> matchedRefBoxes(gtPolys.size(), false);
    for (uint i = 0; i < testPolys.size(); ++i)
    {
        const std::vector<Point>& testPoly = testPolys[i];
        bool matched = false;
        double topIoU = 0;
        for (uint j = 0; j < gtPolys.size() && !matched; ++j)
        {
            if (!matchedRefBoxes[j])
            {
                std::vector<Point> intersectionPolygon;
                float intersectArea = intersectConvexConvex(testPoly, gtPolys[j], intersectionPolygon, true);
                double iou = intersectArea / (contourArea(testPoly) + contourArea(gtPolys[j]) - intersectArea);
                topIoU = std::max(topIoU, iou);
                if (1.0 - iou < boxes_iou_diff)
                {
                    matched = true;
                    matchedRefBoxes[j] = true;
                }
            }
        }
        if (!matched) {
            std::cout << cv::format("Unmatched-det:") << testPoly << std::endl;
            std::cout << "Highest IoU: " << topIoU << std::endl;
        }
        EXPECT_TRUE(matched) << comment;
    }

    // Check unmatched groundtruth.
    for (uint i = 0; i < gtPolys.size(); ++i)
    {
        if (!matchedRefBoxes[i]) {
            std::cout << cv::format("Unmatched-gt:") << gtPolys[i] << std::endl;
        }
        EXPECT_TRUE(matchedRefBoxes[i]);
    }
}

void readFileContent(const std::string& filename, CV_OUT std::vector<char>& content)
{
    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(filename.c_str(), mode);
    ASSERT_TRUE(ifs.is_open());

    content.clear();

    ifs.seekg(0, std::ios::end);
    const size_t sz = ifs.tellg();
    content.resize(sz);
    ifs.seekg(0, std::ios::beg);

    ifs.read((char*)content.data(), sz);
    ASSERT_FALSE(ifs.fail());
}


testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargets(
        bool withInferenceEngine /*= true*/,
        bool withHalide /*= false*/,
        bool withCpuOCV /*= true*/,
        bool withVkCom /*= true*/,
        bool withCUDA /*= true*/,
        bool withNgraph /*= true*/,
        bool withWebnn /*= false*/,
        bool withCann /*= true*/,
        bool withTensorRT /*= true*/
)
{
    bool withVPU = validateVPUType();

    std::vector< tuple<Backend, Target> > targets;
    std::vector< Target > available;
    if (withHalide)
    {
        available = getAvailableTargets(DNN_BACKEND_HALIDE);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
            targets.push_back(make_tuple(DNN_BACKEND_HALIDE, *i));
    }
    if (withInferenceEngine)
    {
        available = getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
        {
            if ((*i == DNN_TARGET_MYRIAD || *i == DNN_TARGET_HDDL) && !withVPU)
                continue;
            targets.push_back(make_tuple(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019, *i));
        }
    }
    if (withNgraph)
    {
        available = getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
        {
            if ((*i == DNN_TARGET_MYRIAD || *i == DNN_TARGET_HDDL) && !withVPU)
                continue;
            targets.push_back(make_tuple(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, *i));
        }

    }
    if (withVkCom)
    {
        available = getAvailableTargets(DNN_BACKEND_VKCOM);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
            targets.push_back(make_tuple(DNN_BACKEND_VKCOM, *i));
    }

#ifdef HAVE_CUDA
    if(withCUDA)
    {
        for (auto target : getAvailableTargets(DNN_BACKEND_CUDA))
            targets.push_back(make_tuple(DNN_BACKEND_CUDA, target));
    }
#endif

#ifdef HAVE_WEBNN
    if (withWebnn)
    {
        for (auto target : getAvailableTargets(DNN_BACKEND_WEBNN)) {
            targets.push_back(make_tuple(DNN_BACKEND_WEBNN, target));
        }
    }
#else
    CV_UNUSED(withWebnn);
#endif

#ifdef HAVE_CANN
    if (withCann)
    {
        for (auto target : getAvailableTargets(DNN_BACKEND_CANN))
            targets.push_back(make_tuple(DNN_BACKEND_CANN, target));
    }
#else
    CV_UNUSED(withCann);
#endif // HAVE_CANN

#ifdef HAVE_TRT
    if (withTensorRT)
    {
        available = getAvailableTargets(DNN_BACKEND_TENSORRT);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
            targets.push_back(make_tuple(DNN_BACKEND_TENSORRT, *i));
    }
#else
    CV_UNUSED(withTensorRT);
#endif // HAVE_TRT

    {
        available = getAvailableTargets(DNN_BACKEND_OPENCV);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
        {
            if (!withCpuOCV && *i == DNN_TARGET_CPU)
                continue;
            targets.push_back(make_tuple(DNN_BACKEND_OPENCV, *i));
        }
    }
    if (targets.empty())  // validate at least CPU mode
        targets.push_back(make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU));
    return testing::ValuesIn(targets);
}

testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargetsIE()
{
#ifdef HAVE_INF_ENGINE
    bool withVPU = validateVPUType();

    std::vector< tuple<Backend, Target> > targets;
    std::vector< Target > available;

    {
        available = getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
        {
            if ((*i == DNN_TARGET_MYRIAD || *i == DNN_TARGET_HDDL) && !withVPU)
                continue;
            targets.push_back(make_tuple(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, *i));
        }

    }

    return testing::ValuesIn(targets);
#else
    return testing::ValuesIn(std::vector< tuple<Backend, Target> >());
#endif
}

static std::string getTestInferenceEngineVPUType()
{
    static std::string param_vpu_type = utils::getConfigurationParameterString("OPENCV_TEST_DNN_IE_VPU_TYPE", "");
    return param_vpu_type;
}

static bool validateVPUType_()
{
    std::string test_vpu_type = getTestInferenceEngineVPUType();
    if (test_vpu_type == "DISABLED" || test_vpu_type == "disabled")
    {
        return false;
    }

    std::vector<Target> available = getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE);
    bool have_vpu_target = false;
    for (std::vector<Target>::const_iterator i = available.begin(); i != available.end(); ++i)
    {
        if (*i == DNN_TARGET_MYRIAD || *i == DNN_TARGET_HDDL)
        {
            have_vpu_target = true;
            break;
        }
    }

    if (test_vpu_type.empty())
    {
        if (have_vpu_target)
        {
            CV_LOG_INFO(NULL, "OpenCV-DNN-Test: VPU type for testing is not specified via 'OPENCV_TEST_DNN_IE_VPU_TYPE' parameter.")
        }
    }
    else
    {
        if (!have_vpu_target)
        {
            CV_LOG_FATAL(NULL, "OpenCV-DNN-Test: 'OPENCV_TEST_DNN_IE_VPU_TYPE' parameter requires VPU of type = '" << test_vpu_type << "', but VPU is not detected. STOP.");
            exit(1);
        }
        std::string dnn_vpu_type = getInferenceEngineVPUType();
        if (dnn_vpu_type != test_vpu_type)
        {
            CV_LOG_FATAL(NULL, "OpenCV-DNN-Test: 'testing' and 'detected' VPU types mismatch: '" << test_vpu_type << "' vs '" << dnn_vpu_type << "'. STOP.");
            exit(1);
        }
    }
    if (have_vpu_target)
    {
        std::string dnn_vpu_type = getInferenceEngineVPUType();
        if (dnn_vpu_type == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2)
            registerGlobalSkipTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_2);
        if (dnn_vpu_type == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
            registerGlobalSkipTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
    }
    return true;
}

bool validateVPUType()
{
    static bool result = validateVPUType_();
    return result;
}


void initDNNTests()
{
    cvtest::addDataSearchEnv("OPENCV_DNN_TEST_DATA_PATH");

    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND,
        CV_TEST_TAG_DNN_SKIP_CPU, CV_TEST_TAG_DNN_SKIP_CPU_FP16,
        CV_TEST_TAG_DNN_SKIP_OPENCL, CV_TEST_TAG_DNN_SKIP_OPENCL_FP16
    );
#if defined(HAVE_HALIDE)
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_HALIDE
    );
#endif
#if defined(INF_ENGINE_RELEASE)
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_IE,
#if INF_ENGINE_VER_MAJOR_EQ(2018050000)
        CV_TEST_TAG_DNN_SKIP_IE_2018R5,
#elif INF_ENGINE_VER_MAJOR_EQ(2019010000)
        CV_TEST_TAG_DNN_SKIP_IE_2019R1,
# if INF_ENGINE_RELEASE == 2019010100
        CV_TEST_TAG_DNN_SKIP_IE_2019R1_1,
# endif
#elif INF_ENGINE_VER_MAJOR_EQ(2019020000)
        CV_TEST_TAG_DNN_SKIP_IE_2019R2,
#elif INF_ENGINE_VER_MAJOR_EQ(2019030000)
        CV_TEST_TAG_DNN_SKIP_IE_2019R3,
#endif
#ifdef HAVE_DNN_NGRAPH
        CV_TEST_TAG_DNN_SKIP_IE_NGRAPH,
#endif
#ifdef HAVE_DNN_IE_NN_BUILDER_2019
        CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER,
#endif
        CV_TEST_TAG_DNN_SKIP_IE_CPU
    );
    registerGlobalSkipTag(
        // see validateVPUType(): CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_2, CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X
        CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16
    );
#endif
#ifdef HAVE_VULKAN
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_VULKAN
    );
#endif
#ifdef HAVE_CUDA
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_CUDA, CV_TEST_TAG_DNN_SKIP_CUDA_FP32, CV_TEST_TAG_DNN_SKIP_CUDA_FP16
    );
#endif
#ifdef HAVE_TIMVX
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_TIMVX
    );
#endif
#ifdef HAVE_CANN
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_CANN
    );
#endif
    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE,
        CV_TEST_TAG_DNN_SKIP_PARSER
    );
}

size_t DNNTestLayer::getTopMemoryUsageMB()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS proc;
    GetProcessMemoryInfo(GetCurrentProcess(), &proc, sizeof(proc));
    return proc.PeakWorkingSetSize / pow(1024, 2);  // bytes to megabytes
#else
    std::ifstream status("/proc/self/status");
    std::string line, title;
    while (std::getline(status, line))
    {
        std::istringstream iss(line);
        iss >> title;
        if (title == "VmHWM:")
        {
            size_t mem;
            iss >> mem;
            return mem / 1024;
        }
    }
    return 0l;
#endif
}

} // namespace
