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
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "hdr_common.hpp"

namespace cv
{

class MergeDebevecImpl CV_FINAL : public MergeDebevec
{
public:
    MergeDebevecImpl() :
        name("MergeDebevec"),
        weights(triangleWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times, InputArray input_response) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);

        int channels = images[0].channels();
        Size size = images[0].size();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        dst.create(images[0].size(), CV_32FCC);
        Mat result = dst.getMat();

        Mat response = input_response.getMat();

        if(response.empty()) {
            response = linearResponse(channels);
            response.at<Vec3f>(0) = response.at<Vec3f>(1);
        }

        Mat log_response;
        log(response, log_response);
        CV_Assert(log_response.rows == LDR_SIZE && log_response.cols == 1 &&
                  log_response.channels() == channels);

        Mat exp_values(times.clone());
        log(exp_values, exp_values);

        result = Mat::zeros(size, CV_32FCC);
        std::vector<Mat> result_split;
        split(result, result_split);
        Mat weight_sum = Mat::zeros(size, CV_32F);

        for(size_t i = 0; i < images.size(); i++) {
            std::vector<Mat> splitted;
            split(images[i], splitted);

            Mat w = Mat::zeros(size, CV_32F);
            for(int c = 0; c < channels; c++) {
                LUT(splitted[c], weights, splitted[c]);
                w += splitted[c];
            }
            w /= channels;

            Mat response_img;
            LUT(images[i], log_response, response_img);
            split(response_img, splitted);
            for(int c = 0; c < channels; c++) {
                result_split[c] += w.mul(splitted[c] - exp_values.at<float>((int)i));
            }
            weight_sum += w;
        }
        weight_sum = 1.0f / weight_sum;
        for(int c = 0; c < channels; c++) {
            result_split[c] = result_split[c].mul(weight_sum);
        }
        merge(result_split, result);
        exp(result, result);
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray times) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        process(src, dst, times, Mat());
    }

protected:
    String name;
    Mat weights;
};

Ptr<MergeDebevec> createMergeDebevec()
{
    return makePtr<MergeDebevecImpl>();
}

class MergeMertensImpl CV_FINAL : public MergeMertens
{
public:
    MergeMertensImpl(float _wcon, float _wsat, float _wexp) :
        name("MergeMertens"),
        wcon(_wcon),
        wsat(_wsat),
        wexp(_wexp)
    {
    }

    void process(InputArrayOfArrays src, OutputArrayOfArrays dst, InputArray, InputArray) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        process(src, dst);
    }

    void process(InputArrayOfArrays src, OutputArray dst) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        checkImageDimensions(images);

        int channels = images[0].channels();
        CV_Assert(channels == 1 || channels == 3);
        Size size = images[0].size();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        std::vector<Mat> weights(images.size());
        Mat weight_sum = Mat::zeros(size, CV_32F);
        Mutex weight_sum_mutex;

        parallel_for_(Range(0, static_cast<int>(images.size())), [&](const Range& range) {
            for(int i = range.start; i < range.end; i++) {
                Mat img, gray, contrast, saturation, wellexp;
                std::vector<Mat> splitted(channels);

                images[i].convertTo(img, CV_32F, 1.0f/255.0f);
                if(channels == 3) {
                    cvtColor(img, gray, COLOR_RGB2GRAY);
                } else {
                    img.copyTo(gray);
                }
                images[i] = img;
                split(img, splitted);

                Laplacian(gray, contrast, CV_32F);
                contrast = abs(contrast);

                Mat mean = Mat::zeros(size, CV_32F);
                for(int c = 0; c < channels; c++) {
                    mean += splitted[c];
                }
                mean /= channels;

                saturation = Mat::zeros(size, CV_32F);
                for(int c = 0; c < channels;  c++) {
                    Mat deviation = splitted[c] - mean;
                    pow(deviation, 2.0f, deviation);
                    saturation += deviation;
                }
                sqrt(saturation, saturation);

                wellexp = Mat::ones(size, CV_32F);
                for(int c = 0; c < channels; c++) {
                    Mat expo = splitted[c] - 0.5f;
                    pow(expo, 2.0f, expo);
                    expo = -expo / 0.08f;
                    exp(expo, expo);
                    wellexp = wellexp.mul(expo);
                }

                pow(contrast, wcon, contrast);
                pow(saturation, wsat, saturation);
                pow(wellexp, wexp, wellexp);

                weights[i] = contrast;
                if(channels == 3) {
                    weights[i] = weights[i].mul(saturation);
                }
                weights[i] = weights[i].mul(wellexp) + 1e-12f;

                AutoLock lock(weight_sum_mutex);
                weight_sum += weights[i];
            }
        });

        int maxlevel = static_cast<int>(logf(static_cast<float>(min(size.width, size.height))) / logf(2.0f));
        std::vector<Mat> res_pyr(maxlevel + 1);
        std::vector<Mutex> res_pyr_mutexes(maxlevel + 1);

        parallel_for_(Range(0, static_cast<int>(images.size())), [&](const Range& range) {
            for(int i = range.start; i < range.end; i++) {
                weights[i] /= weight_sum;

                std::vector<Mat> img_pyr, weight_pyr;
                buildPyramid(images[i], img_pyr, maxlevel);
                buildPyramid(weights[i], weight_pyr, maxlevel);

                for(int lvl = 0; lvl < maxlevel; lvl++) {
                    Mat up;
                    pyrUp(img_pyr[lvl + 1], up, img_pyr[lvl].size());
                    img_pyr[lvl] -= up;
                }
                for(int lvl = 0; lvl <= maxlevel; lvl++) {
                    std::vector<Mat> splitted(channels);
                    split(img_pyr[lvl], splitted);
                    for(int c = 0; c < channels; c++) {
                        splitted[c] = splitted[c].mul(weight_pyr[lvl]);
                    }
                    merge(splitted, img_pyr[lvl]);

                    AutoLock lock(res_pyr_mutexes[lvl]);
                    if(res_pyr[lvl].empty()) {
                        res_pyr[lvl] = img_pyr[lvl];
                    } else {
                        res_pyr[lvl] += img_pyr[lvl];
                    }
                }
            }
        });
        for(int lvl = maxlevel; lvl > 0; lvl--) {
            Mat up;
            pyrUp(res_pyr[lvl], up, res_pyr[lvl - 1].size());
            res_pyr[lvl - 1] += up;
        }
        dst.create(size, CV_32FCC);
        res_pyr[0].copyTo(dst);
    }

    float getContrastWeight() const CV_OVERRIDE { return wcon; }
    void setContrastWeight(float val) CV_OVERRIDE { wcon = val; }

    float getSaturationWeight() const CV_OVERRIDE { return wsat; }
    void setSaturationWeight(float val) CV_OVERRIDE { wsat = val; }

    float getExposureWeight() const CV_OVERRIDE { return wexp; }
    void setExposureWeight(float val) CV_OVERRIDE { wexp = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "contrast_weight" << wcon
           << "saturation_weight" << wsat
           << "exposure_weight" << wexp;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        wcon = fn["contrast_weight"];
        wsat = fn["saturation_weight"];
        wexp = fn["exposure_weight"];
    }

protected:
    String name;
    float wcon, wsat, wexp;
};

Ptr<MergeMertens> createMergeMertens(float wcon, float wsat, float wexp)
{
    return makePtr<MergeMertensImpl>(wcon, wsat, wexp);
}

class MergeRobertsonImpl CV_FINAL : public MergeRobertson
{
public:
    MergeRobertsonImpl() :
        name("MergeRobertson"),
        weight(RobertsonWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times, InputArray input_response) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);

        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        dst.create(images[0].size(), CV_32FCC);
        Mat result = dst.getMat();

        Mat response = input_response.getMat();
        if(response.empty()) {
            float middle = static_cast<float>(LDR_SIZE) / 2.0f;
            response = linearResponse(channels) / middle;
        }
        CV_Assert(response.rows == LDR_SIZE && response.cols == 1 &&
                  response.channels() == channels);

        result = Mat::zeros(images[0].size(), CV_32FCC);
        Mat wsum = Mat::zeros(images[0].size(), CV_32FCC);
        for(size_t i = 0; i < images.size(); i++) {
            Mat im, w;
            LUT(images[i], weight, w);
            LUT(images[i], response, im);

            result += times.at<float>((int)i) * w.mul(im);
            wsum += times.at<float>((int)i) * times.at<float>((int)i) * w;
        }
        result = result.mul(1 / (wsum + Scalar::all(DBL_EPSILON)));
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray times) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        process(src, dst, times, Mat());
    }

protected:
    String name;
    Mat weight;
};

Ptr<MergeRobertson> createMergeRobertson()
{
    return makePtr<MergeRobertsonImpl>();
}

}
