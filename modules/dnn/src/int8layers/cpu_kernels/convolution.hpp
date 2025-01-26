// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FAST_CONVOLUTION_HPP
#define OPENCV_FAST_CONVOLUTION_HPP

#include "opencv2/core/hal/intrin.hpp"

// NOTE: the difference of CONV_NR_PACK and CONV_NR is that the former is used for converting to blockdata layout,
// while the latter is used to meet the reqirements of register number.

#ifndef CONV_PRAM
#define CONV_PRAM

#if CV_NEON && CV_NEON_AARCH64  // 32 registers.
#define CONV_MR 4
#define CONV_NR 24
#elif CV_NEON              // 16 registers.
#define CONV_MR 4
#define CONV_NR 12
#else // SIMD 128, AVX or AVX2
#define CONV_MR 4
#define CONV_NR 24
#endif

#if CV_TRY_AVX || CV_TRY_AVX2
#define CONV_PACKN 4 // Should change this pack to 1 or 8, TODO! rethink the strategy of AVX/AVX2 platform.
#else
#define CONV_PACKN 4
#endif

// NOTE that: CONV_TYPE_DEPTHWISE is for 3x3 depthwise conv, and others depthwise will be set as CONV_TYPE_DEPTHWISE_REMAIN.
enum { QCONV_TYPE_GENERIC=0, QCONV_TYPE_DEPTHWISE=1};
enum { CONV_1D = 0, CONV_2D = 1, CONV_3D = 2 };
#endif

namespace cv {
namespace dnn {

struct FastQConv
{
    int ngroups;
    int K, C, Hk, Wk, Dk;
    int stride_h, stride_w, stride_d;
    int dilation_h, dilation_w, dilation_d;
    int pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_behind;

    std::vector<char> weightsBuf;     // For generic Conv 2D
    char* weightsBufPtr;
    std::vector<int> biasBuf;
    int conv_type;
    int conv_dim;  // Flag for conv1d, conv2d, or conv3d.

    std::vector<float> outputMultiplier;
    float input_sc;
    int input_zp;
    float output_sc;
    int output_zp;
    bool per_channel;

#if CV_SIMD128
    bool useSIMD128 = true;
#else
    bool useSIMD128 = false;
#endif

#if CV_NEON && CV_NEON_AARCH64 && defined(__ARM_FEATURE_DOTPROD)
    bool useNEON = checkHardwareSupport(CPU_NEON);
#else
    bool useNEON = false;
#endif

    bool useAVX   = checkHardwareSupport(CPU_AVX);
    bool useAVX2  = checkHardwareSupport(CPU_AVX2);
    bool useRVV   = checkHardwareSupport(CPU_RVV);
};

// return a FastConv instance.
Ptr<FastQConv> initFastQConv(
        InputArray weightsMat,
        int* srcBias,
        int ngroups,
        int K, int C,
        const std::vector<size_t>& kernel_size,
        const std::vector<size_t>& strides,
        const std::vector<size_t>& dilations,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end,
        int conv_dim,
        const std::vector<float>& outputMultiplier,
        float input_sc,
        int input_zp,
        float output_sc,
        int output_zp,
        bool per_channel);

// It contains different computing branches, like winograd, 1x1 conv.
// The type of activationLUT is int32.
void runFastQConv(InputArray _input, OutputArray _output, const Ptr<FastQConv>& conv, int ntasks,
                   const Ptr<ActivationLayerInt8>& actLayer);

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastQConv>& conv, ActivationLayerInt8* activ_INT8,
                  const  Mat& activationLUT);

namespace opt_NEON
{
#if CV_NEON
void convBlock_INT8(int np, const char* _a, const char* _b, int* c, int ldc, bool init_c, const int width, const int convMR, const int convNR);
#endif
}

} // namespace dnn
} // namespace cv

#endif //OPENCV_FAST_CONVOLUTION_HPP
