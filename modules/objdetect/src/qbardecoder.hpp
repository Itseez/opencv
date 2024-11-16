#ifndef QBARDECODER_H
#define QBARDECODER_H

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <memory>
#include <unordered_set>

// zxing
#include "zxing/zxing.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/binarizer.hpp"
#include "zxing/multi_format_reader.hpp"
#include "zxing/result.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/common/hybrid_binarizer.hpp"
#include "zxing/common/fast_window_binarizer.hpp"
#include "zxing/common/simple_adaptive_binarizer.hpp"
#include "zxing/common/image_cut.hpp"
#include "zxing/common/unicom_block.hpp"
#include "zxing/qrcode/qrcode_reader.hpp"
#include "zxing/binary_bitmap.hpp"
#include "zxing/decode_hints.hpp"
#include "zxing/common/string_utils.hpp"
#include "zxing/common/util/inireader.hpp"

// qbar
#include "qbarsource.hpp"
#include "binarzermgr.hpp"
#include "qbarstruct.hpp"

// ai
#include "detector/qbar_detector.hpp"
#include "detector/align.hpp"
#include "sr_scale/super_scale.hpp"

namespace cv {
class QBarDecoder
{
friend class ParallelDecode;

public:
    QBarDecoder() = default;
    ~QBarDecoder() = default;

    void SetReaders(const std::unordered_set<QBAR_READER> &readers) { readers_ = readers; }
    void setDetectorReferenceSize(int reference_size) {
        if (_init_detector_model_) {
            detector_->setReferenceSize(reference_size);
        }
    }
    void setDetectorScoreThres(float score_thres) {
        if (_init_detector_model_) {
            detector_->setScoreThres(score_thres);
        }
    }
    void setDetectorIouThres(float iou_thres) {
        if (_init_detector_model_) {
            detector_->setIouThres(iou_thres);
        }
    }
    void setDecoderIouThres(float iou_thres) {
        this->iou_thres = iou_thres;
    }   

    void Detect(Mat srcImage, std::vector<DetectInfo> &bboxes);
    QBAR_RESULT Decode(Mat& srcImage);
    std::vector<QBAR_RESULT> Decode(Mat srcImage, std::vector<DetectInfo> &_detect_results_);

    int InitAIModel(QBAR_ML_MODE &ml_mode);

private:
    int Decode(zxing::Ref<zxing::LuminanceSource> source, zxing::Ref<zxing::Result> &result, zxing::DecodeHints& decodeHints);
    QBAR_RESULT ProcessResult(zxing::Result *zx_result);
    void AddFormatsToDecodeHints(zxing::DecodeHints &hints);
    void nms(std::vector<QBAR_RESULT>& results, float NMS_THRESH);

    Mat cropObj(const Mat& img, const DetectInfo& bbox, Align& aligner);
    std::vector<float> getScaleList(const int width, const int height);

    zxing::MultiFormatReader reader_;
    BinarizerMgr binarizer_mgr_;
    std::unordered_set<QBAR_READER> readers_;
    std::string output_charset_ = "UTF-8";

    float iou_thres;

    // AI Model
    bool _init_detector_model_ = false;
    bool _init_sr_model_ = false;
    std::shared_ptr<QBarDetector> detector_;
    std::shared_ptr<SuperScale> sr_;

    std::mutex sr_mutex;
};
}  // namespace cv
#endif
