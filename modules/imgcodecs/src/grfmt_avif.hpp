// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef _GRFMT_AVIF_H_
#define _GRFMT_AVIF_H_

#include "grfmt_base.hpp"

#ifdef HAVE_AVIF

struct avifDecoder;
struct avifEncoder;
struct avifRWData;

namespace cv {

class AvifDecoder CV_FINAL : public BaseImageDecoder {
 public:
  AvifDecoder();
  ~AvifDecoder();

  bool readHeader() CV_OVERRIDE;
  bool readData(Mat& img) CV_OVERRIDE;
  bool nextPage() CV_OVERRIDE;

  size_t signatureLength() const CV_OVERRIDE;
  bool checkSignature(const String& signature) const CV_OVERRIDE;
  ImageDecoder newDecoder() const CV_OVERRIDE;

 protected:
  int channels_;
  int bit_depth_;
  avifDecoder* decoder_;
  bool is_first_image_;
};

class AvifEncoder CV_FINAL : public BaseImageEncoder {
 public:
  AvifEncoder();
  ~AvifEncoder() CV_OVERRIDE;

  bool isFormatSupported(int depth) const CV_OVERRIDE;
  bool write(const Mat& img, const std::vector<int>& params) CV_OVERRIDE;
  bool writeanimation(const Animation& animation, const std::vector<int>& params) CV_OVERRIDE;

  ImageEncoder newEncoder() const CV_OVERRIDE;

 private:
  avifEncoder* encoder_;
};

}  // namespace cv

#endif

#endif /*_GRFMT_AVIF_H_*/
