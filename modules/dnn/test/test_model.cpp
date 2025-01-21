// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "npy_blob.hpp"
namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    String rootFolder = "dnn/";
    return findDataFile(rootFolder + filename, required);
}


class Test_Model : public DNNTestLayer
{
public:
    void testDetectModel(const std::string& weights, const std::string& cfg,
                         const std::string& imgPath, const std::vector<int>& refClassIds,
                         const std::vector<float>& refConfidences,
                         const std::vector<Rect2d>& refBoxes,
                         double scoreDiff, double iouDiff,
                         double confThreshold = 0.24, double nmsThreshold = 0.0,
                         const Size& size = {-1, -1}, Scalar mean = Scalar(),
                         double scale = 1.0, bool swapRB = false, bool crop = false,
                         bool nmsAcrossClasses = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        DetectionModel model(weights, cfg);

        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        model.setNmsAcrossClasses(nmsAcrossClasses);
        if (target == DNN_TARGET_CPU_FP16)
            model.enableWinograd(false);

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect> boxes;

        model.detect(frame, classIds, confidences, boxes, confThreshold, nmsThreshold);

        std::vector<Rect2d> boxesDouble(boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            boxesDouble[i] = boxes[i];
        }
        normAssertDetections(refClassIds, refConfidences, refBoxes, classIds,
                             confidences, boxesDouble, "",
                             confThreshold, scoreDiff, iouDiff);
    }

    void testClassifyModel(const std::string& weights, const std::string& cfg,
                           const std::string& imgPath, std::pair<int, float> ref, float norm,
                           const Size& size = {-1, -1}, Scalar mean = Scalar(),
                           double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        ClassificationModel model(weights, cfg);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        std::pair<int, float> prediction = model.classify(frame);
        EXPECT_EQ(prediction.first, ref.first);
        ASSERT_NEAR(prediction.second, ref.second, norm);
    }

    void testKeypointsModel(const std::string& weights, const std::string& cfg,
                            const Mat& frame, const Mat& exp, float norm,
                            const Size& size = {-1, -1}, Scalar mean = Scalar(),
                            double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        std::vector<Point2f> points;

        KeypointsModel model(weights, cfg);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        points = model.estimate(frame, 0.5);

        Mat out = Mat(points).reshape(1);
        normAssert(exp, out, "", norm, norm);
    }

    void testSegmentationModel(const std::string& weights_file, const std::string& config_file,
                               const std::string& inImgPath, const std::string& outImgPath,
                               float norm, const Size& size = {-1, -1}, Scalar mean = Scalar(),
                               double scale = 1.0, bool swapRB = false, bool crop = false, const std::string outname = "")
    {
        checkBackend();

        Mat frame = imread(inImgPath);
        Mat mask;
        Mat exp = imread(outImgPath, 0);

        SegmentationModel model(weights_file, config_file);
        model.setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        if(!outname.empty())
            model.setOutputNames({outname});

        model.segment(frame, mask);
        normAssert(mask, exp, "", norm, norm);
    }

    void testTextRecognitionModel(const std::string& weights, const std::string& cfg,
                                  const std::string& imgPath, const std::string& seq,
                                  const std::string& decodeType, const std::vector<std::string>& vocabulary,
                                  const Size& size = {-1, -1}, Scalar mean = Scalar(),
                                  double scale = 1.0, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath, IMREAD_GRAYSCALE);

        TextRecognitionModel model(weights, cfg);
        model.setDecodeType(decodeType)
             .setVocabulary(vocabulary)
             .setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        std::string result = model.recognize(frame);
        EXPECT_EQ(result, seq) << "Full frame: " << imgPath;

        std::vector<Rect> rois;
        rois.push_back(Rect(0, 0, frame.cols, frame.rows));
        rois.push_back(Rect(0, 0, frame.cols, frame.rows));  // twice
        std::vector<std::string> results;
        model.recognize(frame, rois, results);
        EXPECT_EQ((size_t)2u, results.size()) << "ROI: " << imgPath;
        EXPECT_EQ(results[0], seq) << "ROI[0]: " << imgPath;
        EXPECT_EQ(results[1], seq) << "ROI[1]: " << imgPath;
    }

    void testTextDetectionModelByDB(const std::string& weights, const std::string& cfg,
                                    const std::string& imgPath, const std::vector<std::vector<Point>>& gt,
                                    float binThresh, float polyThresh,
                                    uint maxCandidates, double unclipRatio,
                                    const Size& size = {-1, -1}, Scalar mean = Scalar(), Scalar scale = Scalar::all(1.0),
                                    double boxes_iou_diff = 0.05, bool swapRB = false, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);

        TextDetectionModel_DB model(weights, cfg);
        model.setBinaryThreshold(binThresh)
             .setPolygonThreshold(polyThresh)
             .setUnclipRatio(unclipRatio)
             .setMaxCandidates(maxCandidates)
             .setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        // 1. Check common TextDetectionModel API through RotatedRect
        std::vector<cv::RotatedRect> results;
        model.detectTextRectangles(frame, results);

        EXPECT_GT(results.size(), (size_t)0);

        std::vector< std::vector<Point> > contours;
        for (size_t i = 0; i < results.size(); i++)
        {
            const RotatedRect& box = results[i];
            Mat contour;
            boxPoints(box, contour);
            std::vector<Point> contour2i(4);
            for (int i = 0; i < 4; i++)
            {
                contour2i[i].x = cvRound(contour.at<float>(i, 0));
                contour2i[i].y = cvRound(contour.at<float>(i, 1));
            }
            contours.push_back(contour2i);
        }
#if 0 // test debug
        Mat result = frame.clone();
        drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
        imshow("result", result); // imwrite("result.png", result);
        waitKey(0);
#endif
        normAssertTextDetections(gt, contours, "", boxes_iou_diff);

        // 2. Check quadrangle-based API
        // std::vector< std::vector<Point> > contours;
        model.detect(frame, contours);

#if 0 // test debug
        Mat result = frame.clone();
        drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
        imshow("result_contours", result); // imwrite("result_contours.png", result);
        waitKey(0);
#endif
        normAssertTextDetections(gt, contours, "", boxes_iou_diff);
    }

    void testTextDetectionModelByEAST(
            const std::string& weights, const std::string& cfg,
            const std::string& imgPath, const std::vector<RotatedRect>& gt,
            float confThresh, float nmsThresh,
            const Size& size = {-1, -1}, Scalar mean = Scalar(),
            double scale = 1.0, bool swapRB = false, bool crop = false,
            double eps_center = 5/*pixels*/, double eps_size = 5/*pixels*/, double eps_angle = 1
    )
    {
        checkBackend();

        Mat frame = imread(imgPath);

        TextDetectionModel_EAST model(weights, cfg);
        model.setConfidenceThreshold(confThresh)
             .setNMSThreshold(nmsThresh)
             .setInputSize(size).setInputMean(mean).setInputScale(scale)
             .setInputSwapRB(swapRB).setInputCrop(crop);

        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        std::vector<cv::RotatedRect> results;
        model.detectTextRectangles(frame, results);

        EXPECT_EQ(results.size(), (size_t)1);
        for (size_t i = 0; i < results.size(); i++)
        {
            const RotatedRect& box = results[i];
#if 0 // test debug
            Mat contour;
            boxPoints(box, contour);
            std::vector<Point> contour2i(4);
            for (int i = 0; i < 4; i++)
            {
                contour2i[i].x = cvRound(contour.at<float>(i, 0));
                contour2i[i].y = cvRound(contour.at<float>(i, 1));
            }
            std::vector< std::vector<Point> > contours;
            contours.push_back(contour2i);

            Mat result = frame.clone();
            drawContours(result, contours, -1, Scalar(0, 0, 255), 1);
            imshow("result", result); //imwrite("result.png", result);
            waitKey(0);
#endif
            const RotatedRect& gtBox = gt[i];
            EXPECT_NEAR(box.center.x, gtBox.center.x, eps_center);
            EXPECT_NEAR(box.center.y, gtBox.center.y, eps_center);
            EXPECT_NEAR(box.size.width, gtBox.size.width, eps_size);
            EXPECT_NEAR(box.size.height, gtBox.size.height, eps_size);
            EXPECT_NEAR(box.angle, gtBox.angle, eps_angle);
        }
    }

    void testYOLODetectionModel(const std::string& weights, const std::string& cfg,
                                const std::string& imgPath, const std::vector<int>& refClassIds,
                                const std::vector<float>& refConfidences,
                                const std::vector<Rect2d>& refBoxes,
                                double scoreDiff, double iouDiff,
                                double confThreshold = 0.24, double nmsThreshold = 0.0,
                                const Size& size = {-1, -1}, Scalar mean = Scalar(),
                                double scale = 1.0, bool swapRB = false, bool crop = false,
                                bool nmsAcrossClasses = false, float paddingValue = 0.0,
                                ImagePaddingMode paddingMode = DNN_PMODE_NULL){
        checkBackend();

        Mat frame = imread(imgPath);

        YOLODetectionModel model;
        // Initialize model based on the condition
        if (cfg == "") {
            // ONNX test
            model = YOLODetectionModel(weights);
        } else {
            // Darknet test
            model = YOLODetectionModel(weights, cfg);
        }

        model.setPaddingMode(paddingMode).setPaddingValue(paddingValue)
             .setInputSize(size).setInputMean(mean)
             .setInputScale(scale).setInputSwapRB(swapRB)
             .setInputCrop(crop);


        model.setPreferableBackend(backend);
        model.setPreferableTarget(target);

        model.setNmsAcrossClasses(nmsAcrossClasses);
        if (target == DNN_TARGET_CPU_FP16)
            model.enableWinograd(false);

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect> boxes;

        model.detect(frame, classIds, confidences, boxes, confThreshold, nmsThreshold);

        // for (int i=0; i < boxes.size(); i++){
        //     std::cout << boxes[i].x << " " << boxes[i].y << " " << boxes[i].width << " " << boxes[i].height << " " << confidences[i] << " " << classIds[i] << std::endl;
        // }


        std::vector<Rect2d> boxesDouble(boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            boxesDouble[i] = boxes[i];
        }
        normAssertDetections(refClassIds, refConfidences, refBoxes, classIds,
                             confidences, boxesDouble, "",
                             confThreshold, scoreDiff, iouDiff);
    }
};

TEST_P(Test_Model, Classify)
{
    std::pair<int, float> ref(652, 0.641789);

    std::string img_path = _tf("grace_hopper_227.png");
    std::string config_file = _tf("bvlc_alexnet.prototxt");
    std::string weights_file = _tf("bvlc_alexnet.caffemodel", false);

    Size size{227, 227};
    float norm = 1e-4;

    testClassifyModel(weights_file, config_file, img_path, ref, norm, size);
}


TEST_P(Test_Model, DetectRegion)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    // FIXIT DNN_BACKEND_INFERENCE_ENGINE is misused
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
#endif

#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
#endif

    std::vector<int> refClassIds = {6, 1, 11};
    std::vector<float> refConfidences = {0.750469f, 0.780879f, 0.901615f};
    std::vector<Rect2d> refBoxes = {Rect2d(240, 53, 135, 72),
                                    Rect2d(112, 109, 192, 200),
                                    Rect2d(58, 141, 117, 249)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("yolo-voc.weights", false);
    std::string config_file = _tf("yolo-voc.cfg");

    double scale = 1.0 / 255.0;
    Size size{416, 416};
    bool swapRB = true;

    double confThreshold = 0.24;
    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.397 : 0.4;
    double scoreDiff = 8e-5, iouDiff = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1e-2;
        iouDiff = 1.6e-2;
    }

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences,
                    refBoxes, scoreDiff, iouDiff, confThreshold, nmsThreshold, size,
                    Scalar(), scale, swapRB);
}

TEST_P(Test_Model, DetectRegionWithNmsAcrossClasses)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
#endif

#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
#endif

    std::vector<int> refClassIds = { 6, 11 };
    std::vector<float> refConfidences = { 0.750469f, 0.901615f };
    std::vector<Rect2d> refBoxes = { Rect2d(240, 53, 135, 72),
                                    Rect2d(58, 141, 117, 249) };

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("yolo-voc.weights", false);
    std::string config_file = _tf("yolo-voc.cfg");

    double scale = 1.0 / 255.0;
    Size size{ 416, 416 };
    bool swapRB = true;
    bool crop = false;
    bool nmsAcrossClasses = true;

    double confThreshold = 0.24;
    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.15: 0.15;
    double scoreDiff = 8e-5, iouDiff = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1e-2;
        iouDiff = 1.6e-2;
    }

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences,
        refBoxes, scoreDiff, iouDiff, confThreshold, nmsThreshold, size,
        Scalar(), scale, swapRB, crop,
        nmsAcrossClasses);
}

TEST_P(Test_Model, DetectionOutput)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:427:
    // While validating node 'v1::Reshape bbox_pred_reshape (ave_bbox_pred_rois[0]:f32{1,8,1,1}, Constant_388[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
    // Requested output shape {1,300,8,1} is incompatible with input shape {1, 8, 1, 1}
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Exception: Function contains several inputs and outputs with one friendly name! (HETERO bug?)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    // FIXIT DNN_BACKEND_INFERENCE_ENGINE is misused
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

    std::vector<int> refClassIds = {7, 12};
    std::vector<float> refConfidences = {0.991359f, 0.94786f};
    std::vector<Rect2d> refBoxes = {Rect2d(491, 81, 212, 98),
                                    Rect2d(132, 223, 207, 344)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("resnet50_rfcn_final.caffemodel", false);
    std::string config_file = _tf("rfcn_pascal_voc_resnet50.prototxt");

    Scalar mean = Scalar(102.9801, 115.9465, 122.7717);
    Size size{800, 600};

    double scoreDiff = default_l1, iouDiff = 1e-5;
    float confThreshold = 0.8;
    double nmsThreshold = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        if (backend == DNN_BACKEND_OPENCV)
            scoreDiff = 4e-3;
        else
            scoreDiff = 2e-2;
        iouDiff = 1.8e-1;
    }
#if defined(INF_ENGINE_RELEASE)
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            scoreDiff = 0.05;
            iouDiff = 0.08;
        }
#endif

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean);
}


TEST_P(Test_Model, DetectionMobilenetSSD)
{
    Mat ref = blobFromNPY(_tf("mobilenet_ssd_caffe_out.npy"));
    ref = ref.reshape(1, ref.size[2]);

    std::string img_path = _tf("street.png");
    Mat frame = imread(img_path);
    int frameWidth  = frame.cols;
    int frameHeight = frame.rows;

    std::vector<int> refClassIds;
    std::vector<float> refConfidences;
    std::vector<Rect2d> refBoxes;
    for (int i = 0; i < ref.rows; i++)
    {
        refClassIds.emplace_back(ref.at<float>(i, 1));
        refConfidences.emplace_back(ref.at<float>(i, 2));
        int left   = ref.at<float>(i, 3) * frameWidth;
        int top    = ref.at<float>(i, 4) * frameHeight;
        int right  = ref.at<float>(i, 5) * frameWidth;
        int bottom = ref.at<float>(i, 6) * frameHeight;
        int width  = right  - left + 1;
        int height = bottom - top + 1;
        refBoxes.emplace_back(left, top, width, height);
    }

    std::string weights_file = _tf("MobileNetSSD_deploy_19e3ec3.caffemodel", false);
    std::string config_file = _tf("MobileNetSSD_deploy_19e3ec3.prototxt");

    Scalar mean = Scalar(127.5, 127.5, 127.5);
    double scale = 1.0 / 127.5;
    Size size{300, 300};

    double scoreDiff = 1e-5, iouDiff = 1e-5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 1.7e-2;
        iouDiff = 6.91e-2;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.017;
        if (getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
            iouDiff = 0.1;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.0028;
        iouDiff = 1e-2;
    }
    float confThreshold = FLT_MIN;
    double nmsThreshold = 0.0;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale);
}

TEST_P(Test_Model, Keypoints_pose)
{
    if (target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
#ifdef HAVE_INF_ENGINE
    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Mat inp = imread(_tf("pose.png"));
    std::string weights = _tf("onnx/models/lightweight_pose_estimation_201912.onnx", false);
    float kpdata[] = {
        237.65625f, 78.25f, 237.65625f, 136.9375f,
        190.125f, 136.9375f, 142.59375f, 195.625f, 79.21875f, 176.0625f, 285.1875f, 117.375f,
        348.5625f, 195.625f, 396.09375f, 176.0625f, 205.96875f, 313.0f, 205.96875f, 430.375f,
        205.96875f, 528.1875f, 269.34375f, 293.4375f, 253.5f, 430.375f, 237.65625f, 528.1875f,
        221.8125f, 58.6875f, 253.5f, 58.6875f, 205.96875f, 78.25f, 253.5f, 58.6875f
    };
    Mat exp(18, 2, CV_32FC1, kpdata);

    Size size{256, 256};
    float norm = 1e-4;
    double scale = 1.0/255;
    Scalar mean = Scalar(128, 128, 128);
    bool swapRB = false;

    // Ref. Range: [58.6875, 508.625]
    if (target == DNN_TARGET_CUDA_FP16)
        norm = 20; // l1 = 1.5, lInf = 20

    testKeypointsModel(weights, "", inp, exp, norm, size, mean, scale, swapRB);
}

TEST_P(Test_Model, Keypoints_face)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Mat inp = imread(_tf("gray_face.png"), 0);
    std::string weights = _tf("onnx/models/facial_keypoints.onnx", false);
    Mat exp = blobFromNPY(_tf("facial_keypoints_exp.npy"));

    Size size{224, 224};
    double scale = 1.0/255;
    Scalar mean = Scalar();
    bool swapRB = false;

    // Ref. Range: [-1.1784188, 1.7758257]
    float norm = 1e-4;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
        norm = 5e-3;
    if (target == DNN_TARGET_MYRIAD)
    {
        // Myriad2: l1 = 0.0004, lInf = 0.002
        // MyriadX: l1 = 0.003, lInf = 0.009
        norm = 0.009;
    }
    if (target == DNN_TARGET_CUDA_FP16)
        norm = 0.004; // l1 = 0.0006, lInf = 0.004

    testKeypointsModel(weights, "", inp, exp, norm, size, mean, scale, swapRB);
}

TEST_P(Test_Model, Detection_normalized)
{
    std::string img_path = _tf("grace_hopper_227.png");
    std::vector<int> refClassIds = {15};
    std::vector<float> refConfidences = {0.999222f};
    std::vector<Rect2d> refBoxes = {Rect2d(0, 4, 227, 222)};

    std::string weights_file = _tf("MobileNetSSD_deploy_19e3ec3.caffemodel", false);
    std::string config_file = _tf("MobileNetSSD_deploy_19e3ec3.prototxt");

    Scalar mean = Scalar(127.5, 127.5, 127.5);
    double scale = 1.0 / 127.5;
    Size size{300, 300};

    double scoreDiff = 1e-5, iouDiff = 1e-5;
    float confThreshold = FLT_MIN;
    double nmsThreshold = 0.0;
    if (target == DNN_TARGET_CUDA)
    {
        scoreDiff = 3e-4;
        iouDiff = 0.018;
    }
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 5e-3;
        iouDiff = 0.09;
    }
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.02;
        iouDiff = 0.1f;
    }
#endif
    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale);
}

TEST_P(Test_Model, Segmentation)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );

    float norm = 0;

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Failed to allocate graph: NC_ERROR
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // accuracy
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
    {
        norm = 25.0f;  // depends on OS/OpenCL version
    }
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Failed to allocate graph: NC_ERROR
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // cnn_network_ngraph_impl.cpp:104 Function contains several inputs and outputs with one friendly name: 'upscore2'!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // cnn_network_ngraph_impl.cpp:104 Function contains several inputs and outputs with one friendly name: 'upscore2'!
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    // Failed to allocate graph: NC_ERROR
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    if ((backend == DNN_BACKEND_OPENCV && (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16))
        || (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16))
    {
        norm = 7.0f;  // l1 = 0.01 lInf = 7
    }

    std::string inp = _tf("dog416.png");
    std::string weights_file = _tf("onnx/models/fcn-resnet50-12.onnx", false);
    std::string exp = _tf("segmentation_exp.png");

    Size size{128, 128};
    double scale = 0.019;
    Scalar mean = Scalar(0.485*255, 0.456*255, 0.406*255);
    bool swapRB = true;

    testSegmentationModel(weights_file, "", inp, exp, norm, size, mean, scale, swapRB, false, "out");
}

TEST_P(Test_Model, TextRecognition)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // FIXIT: dnn/src/ie_ngraph.cpp:494: error: (-215:Assertion failed) !inps.empty() in function 'createNet'
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_CPU, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    // Node Transpose_79 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name 71 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif

    std::string imgPath = _tf("text_rec_test.png");
    std::string weightPath = _tf("onnx/models/crnn.onnx", false);
    std::string seq = "welcome";

    Size size{100, 32};
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5);
    std::string decodeType = "CTC-greedy";
    std::vector<std::string> vocabulary = {"0","1","2","3","4","5","6","7","8","9",
                                           "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};

    testTextRecognitionModel(weightPath, "", imgPath, seq, decodeType, vocabulary, size, mean, scale);
}

TEST_P(Test_Model, TextRecognitionWithCTCPrefixBeamSearch)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Node Transpose_79 was not assigned on any pointed device
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE Exception: Ngraph operation Reshape with name 71 has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#endif


    std::string imgPath = _tf("text_rec_test.png");
    std::string weightPath = _tf("onnx/models/crnn.onnx", false);
    std::string seq = "welcome";

    Size size{100, 32};
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5);
    std::string decodeType = "CTC-prefix-beam-search";
    std::vector<std::string> vocabulary = {"0","1","2","3","4","5","6","7","8","9",
                                           "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};

    testTextRecognitionModel(weightPath, "", imgPath, seq, decodeType, vocabulary, size, mean, scale);
}

TEST_P(Test_Model, TextDetectionByDB)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    if (target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

    std::string imgPath = _tf("text_det_test1.png");
    std::string weightPathDB = _tf("onnx/models/DB_TD500_resnet50.onnx", false);
    std::string weightPathPPDB = _tf("onnx/models/PP_OCRv3_DB_text_det.onnx", false);

    // GroundTruth
    std::vector<std::vector<Point>> gt = {
        { Point(142, 193), Point(136, 164), Point(213, 150), Point(219, 178) },
        { Point(136, 165), Point(122, 114), Point(319, 71), Point(330, 122) }
    };

    Size size{736, 736};
    Scalar scaleDB = Scalar::all(1.0 / 255.0);
    Scalar meanDB = Scalar(122.67891434, 116.66876762, 104.00698793);

    // new mean and stddev
    Scalar meanPPDB = Scalar(123.675, 116.28, 103.53);
    Scalar stddevPPDB = Scalar(0.229, 0.224, 0.225);
    Scalar scalePPDB = scaleDB / stddevPPDB;

    float binThresh = 0.3;
    float polyThresh = 0.5;
    uint maxCandidates = 200;
    double unclipRatio = 2.0;

    {
    SCOPED_TRACE("Original DB");
    testTextDetectionModelByDB(weightPathDB, "", imgPath, gt, binThresh, polyThresh, maxCandidates, unclipRatio, size, meanDB, scaleDB, 0.05f);
    }

    {
    SCOPED_TRACE("PP-OCRDBv3");
    testTextDetectionModelByDB(weightPathPPDB, "", imgPath, gt, binThresh, polyThresh, maxCandidates, unclipRatio, size, meanPPDB, scalePPDB, 0.21f);
    }
}

TEST_P(Test_Model, TextDetectionByEAST)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    std::string imgPath = _tf("text_det_test2.jpg");
    std::string weightPath = _tf("frozen_east_text_detection.pb", false);

    // GroundTruth
    std::vector<RotatedRect> gt = {
        RotatedRect(Point2f(657.55f, 409.5f), Size2f(316.84f, 62.45f), -4.79)
    };

    // Model parameters
    Size size{320, 320};
    double scale = 1.0;
    Scalar mean = Scalar(123.68, 116.78, 103.94);
    bool swapRB = true;

    // Detection algorithm parameters
    float confThresh = 0.5;
    float nmsThresh = 0.4;

    double eps_center = 5/*pixels*/;
    double eps_size = 5/*pixels*/;
    double eps_angle = 1;

    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        eps_center = 10;
        eps_size = 25;
        eps_angle = 3;
    }

    testTextDetectionModelByEAST(weightPath, "", imgPath, gt, confThresh, nmsThresh, size, mean, scale, swapRB, false/*crop*/,
        eps_center, eps_size, eps_angle
    );
}

TEST_P(Test_Model, YoloDetectionDarknet)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:427:
    // While validating node 'v1::Reshape bbox_pred_reshape (ave_bbox_pred_rois[0]:f32{1,8,1,1}, Constant_388[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
    // Requested output shape {1,300,8,1} is incompatible with input shape {1, 8, 1, 1}
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Exception: Function contains several inputs and outputs with one friendly name! (HETERO bug?)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    // FIXIT DNN_BACKEND_INFERENCE_ENGINE is misused
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

    std::vector<int> refClassIds = {16, 1, 7};
    std::vector<float> refConfidences = {0.985363f, 0.976825f, 0.937539f};
    std::vector<Rect2d> refBoxes = {Rect2d(71, 167, 97, 221),
                                    Rect2d(69, 94, 237, 212),
                                    Rect2d(250, 54, 125, 70)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("yolov4.weights", false);
    std::string config_file = _tf("yolov4.cfg");

    Size size{416, 416};
    Scalar mean = Scalar(0);
    double scale = 1/255.0;
    bool swapRB = true;
    bool crop = false;
    bool nmsAcrossClasses = true;
    float paddingValue = 0.0;
    ImagePaddingMode paddingMode = DNN_PMODE_NULL;


    double scoreDiff = default_l1, iouDiff = 1e-5;
    float confThreshold = 0.5;
    double nmsThreshold = 0.5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        if (backend == DNN_BACKEND_OPENCV)
            scoreDiff = 4e-3;
        else
            scoreDiff = 2e-2;
        iouDiff = 1.8e-1;
    }
#if defined(INF_ENGINE_RELEASE)
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            scoreDiff = 0.05;
            iouDiff = 0.08;
        }
#endif

    testYOLODetectionModel(weights_file, config_file, img_path,
                           refClassIds, refConfidences, refBoxes,
                           scoreDiff, iouDiff, confThreshold,
                           nmsThreshold, size, mean, scale,
                           swapRB, crop, nmsAcrossClasses, paddingValue,
                           paddingMode);
}

TEST_P(Test_Model, YoloDetectionONNX)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:427:
    // While validating node 'v1::Reshape bbox_pred_reshape (ave_bbox_pred_rois[0]:f32{1,8,1,1}, Constant_388[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
    // Requested output shape {1,300,8,1} is incompatible with input shape {1, 8, 1, 1}
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // Exception: Function contains several inputs and outputs with one friendly name! (HETERO bug?)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    // FIXIT DNN_BACKEND_INFERENCE_ENGINE is misused
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

    if (target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
#endif

    std::vector<int> refClassIds = {1, 16, 7};
    std::vector<float> refConfidences = {0.96067f, 0.900769f, 0.854416f};
    std::vector<Rect2d> refBoxes = {Rect2d(66, 94, 241, 208),
                                    Rect2d(70, 161, 96, 228),
                                    Rect2d(252, 53, 123, 70),
                                    };

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("/onnx/models/yolox_s_inf_decoder.onnx", false);
    std::string config_file = "";

    Size size{640, 640};
    Scalar mean = Scalar(0);
    double scale = 1.0;
    bool swapRB = true;
    bool crop = false;
    bool nmsAcrossClasses = false;
    float paddingValue = 114.0;
    ImagePaddingMode paddingMode = DNN_PMODE_LETTERBOX;


    double scoreDiff = default_l1, iouDiff = 1e-5;
    float confThreshold = 0.5;
    double nmsThreshold = 0.5;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        if (backend == DNN_BACKEND_OPENCV)
            scoreDiff = 4e-3;
        else
            scoreDiff = 2e-2;
        iouDiff = 1.8e-1;
    }
#if defined(INF_ENGINE_RELEASE)
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            scoreDiff = 0.05;
            iouDiff = 0.08;
        }
#endif

    testYOLODetectionModel(weights_file, config_file, img_path,
                           refClassIds, refConfidences, refBoxes,
                           scoreDiff, iouDiff, confThreshold,
                           nmsThreshold, size, mean, scale,
                           swapRB, crop, nmsAcrossClasses, paddingValue,
                           paddingMode);
}
INSTANTIATE_TEST_CASE_P(/**/, Test_Model, dnnBackendsAndTargets());

}} // namespace
