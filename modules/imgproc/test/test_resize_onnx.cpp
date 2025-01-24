// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

struct ResizeOnnx
{
    int interpolate;
    Size szsrc, szref, szdst;
    Point2d scale;
    float cubic;
    /* make sure insrc is:
     *   (1) integer
     *   (2) range [-127, 127]
     *   (3) all non-positive or non-negative */
    vector<double> insrc, inref;

    void rand_roi(RNG& rng, Mat& src, Size size, int type)
    {
        int const border = 16;
        int t = rng.next() % border;
        int b = rng.next() % border;
        int l = rng.next() % border;
        int r = rng.next() % border;
        if (rng.next() & 1)
        {
            src.create(size.height + t + b, size.width + l + r, type);
            src.setTo(127);
            src = src(Rect(l, t, size.width, size.height));
        }
        else
            src.create(size, type);
    }

    void run()
    {
        CV_CheckGE(static_cast<int>(insrc.size()), szsrc.area(), "unexpected src size");
        CV_CheckEQ(static_cast<int>(inref.size()), szref.area(), "unexpected ref size");
        Mat iS(szsrc, CV_64F, insrc.data());
        Mat iR(szref, CV_64F, inref.data());
        Mat S = iS, R = iR, nS, nR;
        // values after convertTo need to be all positive or all negative
        double alpha[8] = {1, -1, 5, 5, 0, -3, -2, 4};
        double  beta[8] = {0, -0, 0, 2, 7, -1, -6, 9};
        RNG& rng = TS::ptr()->get_rng();
        for (int cn = 1; cn <= 8; ++cn)
        {
            if (cn > 1)
            {
                iS.convertTo(nS, -1, alpha[cn - 1], beta[cn - 1]);
                iR.convertTo(nR, -1, alpha[cn - 1], beta[cn - 1]);
                merge(vector<Mat>{S, nS}, S);
                merge(vector<Mat>{R, nR}, R);
            }
            for (int depth = CV_8U; depth <= CV_64F; ++depth)
            {
                double eps = (depth <= CV_32S) ? 1.0 : 1e-3;
                int type = CV_MAKETYPE(depth, cn);
                Mat src, ref, dst;
                rand_roi(rng, src, szsrc, type);
                if (szdst.area())
                    rand_roi(rng, dst, szdst, type);
                S.convertTo(src, type);
                R.convertTo(ref, type);
                resizeOnnx(src, dst, szdst, scale, interpolate, cubic);
                // nearest must give bit-same result
                if ((interpolate & INTER_SAMPLER_MASK) == INTER_NEAREST)
                    EXPECT_MAT_NEAR(ref, dst, 0.0);
                // cvRound(4.5) = 4, but when doing resize with int, we may get 5
                else
                    EXPECT_MAT_NEAR(ref, dst, eps);
            }
        }
    }
};

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-128

TEST(ResizeOnnx, downsample_scales_cubic)
{
    ResizeOnnx{
        INTER_CUBIC,
        Size(4, 4), Size(3, 3), Size(), Point2d(0.8, 0.8), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.47119141, 2.78125   ,  4.08251953,
             6.71142578, 8.02148438,  9.32275391,
            11.91650391, 13.2265625, 14.52783203,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_scales_cubic_A_n0p5_exclude_outside)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_EXCLUDE_OUTSIDE,
        Size(4, 4), Size(3, 3), Size(), Point2d(0.8, 0.8), -0.5f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.36812675,  2.6695014 ,  4.0133367 ,
             6.57362535,  7.875     ,  9.2188353 ,
            11.94896657, 13.25034122, 14.59417652,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_scales_cubic_align_corners)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_ALIGN_CORNERS,
        Size(4, 4), Size(3, 3), Size(), Point2d(0.8, 0.8), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.0       ,  2.39519159,  3.79038317,
             6.58076634,  7.97595793,  9.37114951,
            12.16153268, 13.55672427, 14.95191585,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_scales_cubic_antialias)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_ANTIALIAS,
        Size(4, 4), Size(2, 2), Size(), Point2d(0.6, 0.6), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            2.5180721,  4.2858863,
            9.589329 , 11.357142 ,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_scales_linear)
{
    ResizeOnnx{
        INTER_LINEAR,
        Size(4, 2), Size(2, 1), Size(), Point2d(0.6, 0.6), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8},
        {2.6666665, 4.3333331}
    }.run();
}

TEST(ResizeOnnx, downsample_scales_linear_align_corners)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_ALIGN_CORNERS,
        Size(4, 2), Size(2, 1), Size(), Point2d(0.6, 0.6), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8},
        {1.0, 3.142857}
    }.run();
}

TEST(ResizeOnnx, downsample_scales_linear_antialias)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_ANTIALIAS,
        Size(4, 4), Size(2, 2), Size(), Point2d(0.6, 0.6), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            2.875,  4.5,
            9.375, 11.0,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_scales_linear_half_pixel_symmetric)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_HALF_PIXEL_SYMMETRIC,
        Size(4, 1), Size(2, 1), Size(), Point2d(0.6, 1.0), -0.75f,
        {1, 2, 3, 4},
        {1.6666667, 3.3333333}
    }.run();
}

TEST(ResizeOnnx, downsample_scales_nearest)
{
    ResizeOnnx{
        INTER_NEAREST,
        Size(4, 2), Size(2, 1), Size(), Point2d(0.6, 0.6), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8},
        {1, 3}
    }.run();
}

TEST(ResizeOnnx, downsample_sizes_cubic)
{
    ResizeOnnx{
        INTER_CUBIC,
        Size(4, 4), Size(3, 3), Size(3, 3), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.63078704,  3.00462963,  4.37847222,
             7.12615741,  8.5       ,  9.87384259,
            12.62152778, 13.99537037, 15.36921296,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_sizes_cubic_antialias)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_ANTIALIAS,
        Size(4, 4), Size(3, 3), Size(3, 3), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.7750092,   3.1200073,  4.4650054,
             7.1550016,   8.5      ,  9.844998 ,
            12.534994 ,  13.8799925, 15.224991 ,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_sizes_linear_antialias)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_ANTIALIAS,
        Size(4, 4), Size(3, 3), Size(3, 3), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             2.3636363,  3.590909,   4.818182,
             7.2727275,  8.5      ,  9.727273,
            12.181818 , 13.409091,  14.636364,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_sizes_linear_pytorch_half_pixel)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_HALF_PIXEL_PYTORCH,
        Size(4, 4), Size(1, 3), Size(1, 3), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.6666666,
             7.0      ,
            12.333333 ,
        }
    }.run();
}

TEST(ResizeOnnx, downsample_sizes_nearest)
{
    ResizeOnnx{
        INTER_NEAREST,
        Size(4, 2), Size(3, 1), Size(3, 1), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8},
        {1, 2, 4}
    }.run();
}

TEST(ResizeOnnx, upsample_scales_cubic)
{
    ResizeOnnx{
        INTER_CUBIC,
        Size(4, 4), Size(8, 8), Size(), Point2d(2.0, 2.0), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            0.47265625, 0.76953125, 1.24609375, 1.875, 2.28125, 2.91015625, 3.38671875, 3.68359375,
            1.66015625, 1.95703125, 2.43359375, 3.0625, 3.46875, 4.09765625, 4.57421875, 4.87109375,
            3.56640625, 3.86328125, 4.33984375, 4.96875, 5.375, 6.00390625, 6.48046875, 6.77734375,
            6.08203125, 6.37890625, 6.85546875, 7.484375, 7.890625, 8.51953125, 8.99609375, 9.29296875,
            7.70703125, 8.00390625, 8.48046875, 9.109375, 9.515625, 10.14453125, 10.62109375, 10.91796875,
            10.22265625, 10.51953125, 10.99609375, 11.625, 12.03125, 12.66015625, 13.13671875, 13.43359375,
            12.12890625, 12.42578125, 12.90234375, 13.53125, 13.9375, 14.56640625, 15.04296875, 15.33984375,
            13.31640625, 13.61328125, 14.08984375, 14.71875, 15.125, 15.75390625, 16.23046875, 16.52734375,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_cubic_A_n0p5_exclude_outside)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_EXCLUDE_OUTSIDE,
        Size(4, 4), Size(8, 8), Size(), Point2d(2.0, 2.0), -0.5f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            0.55882353, 0.81494204, 1.35698249, 1.89705882, 2.39705882, 2.93713516, 3.47917561, 3.73529412,
            1.58329755, 1.83941606, 2.38145651, 2.92153285, 3.42153285, 3.96160918, 4.50364964, 4.75976814,
            3.75145936, 4.00757787, 4.54961832, 5.08969466, 5.58969466, 6.12977099, 6.67181144, 6.92792995,
            5.91176471, 6.16788321, 6.70992366, 7.25, 7.75, 8.29007634, 8.83211679, 9.08823529,
            7.91176471, 8.16788321, 8.70992366, 9.25, 9.75, 10.29007634, 10.83211679, 11.08823529,
            10.07207005, 10.32818856, 10.87022901, 11.41030534, 11.91030534, 12.45038168, 12.99242213, 13.24854064,
            12.24023186, 12.49635036, 13.03839082, 13.57846715, 14.07846715, 14.61854349, 15.16058394, 15.41670245,
            13.26470588, 13.52082439, 14.06286484, 14.60294118, 15.10294118, 15.64301751, 16.18505796, 16.44117647,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_cubic_align_corners)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_ALIGN_CORNERS,
        Size(4, 4), Size(8, 8), Size(), Point2d(2.0, 2.0), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            1.0, 1.34110787, 1.80029155, 2.32944606, 2.67055394, 3.19970845, 3.65889213, 4.0,
            2.36443149, 2.70553936, 3.16472303, 3.69387755, 4.03498542, 4.56413994, 5.02332362, 5.36443149,
            4.20116618, 4.54227405, 5.00145773, 5.53061224, 5.87172012, 6.40087464, 6.86005831, 7.20116618,
            6.31778426, 6.65889213, 7.1180758, 7.64723032, 7.98833819, 8.51749271, 8.97667638, 9.31778426,
            7.68221574, 8.02332362, 8.48250729, 9.01166181, 9.35276968, 9.8819242, 10.34110787, 10.68221574,
            9.79883382, 10.13994169, 10.59912536, 11.12827988, 11.46938776, 11.99854227, 12.45772595, 12.79883382,
            11.63556851, 11.97667638, 12.43586006, 12.96501458, 13.30612245, 13.83527697, 14.29446064, 14.63556851,
            13.0, 13.34110787, 13.80029155, 14.32944606, 14.67055394, 15.19970845, 15.65889213, 16.0,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_cubic_asymmetric)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_ASYMMETRIC,
        Size(4, 4), Size(8, 8), Size(), Point2d(2.0, 2.0), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            1.0, 1.40625, 2.0, 2.5, 3.0, 3.59375, 4.0, 4.09375,
            2.625, 3.03125, 3.625, 4.125, 4.625, 5.21875, 5.625, 5.71875,
            5.0, 5.40625, 6.0, 6.5, 7.0, 7.59375, 8.0, 8.09375,
            7.0, 7.40625, 8.0, 8.5, 9.0, 9.59375, 10.0, 10.09375,
            9.0, 9.40625, 10.0, 10.5, 11.0, 11.59375, 12.0, 12.09375,
            11.375, 11.78125, 12.375, 12.875, 13.375, 13.96875, 14.375, 14.46875,
            13.0, 13.40625, 14.0, 14.5, 15.0, 15.59375, 16.0, 16.09375,
            13.375, 13.78125, 14.375, 14.875, 15.375, 15.96875, 16.375, 16.46875,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_linear)
{
    ResizeOnnx{
        INTER_LINEAR,
        Size(2, 2), Size(4, 4), Size(), Point2d(2.0, 2.0), -0.75f,
        {1, 2, 3, 4},
        {
            1.0, 1.25, 1.75, 2.0,
            1.5, 1.75, 2.25, 2.5,
            2.5, 2.75, 3.25, 3.5,
            3.0, 3.25, 3.75, 4.0,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_linear_align_corners)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_ALIGN_CORNERS,
        Size(2, 2), Size(4, 4), Size(), Point2d(2.0, 2.0), -0.75f,
        {1, 2, 3, 4},
        {
            1.0, 1.33333333, 1.66666667, 2.0,
            1.66666667, 2.0, 2.33333333, 2.66666667,
            2.33333333, 2.66666667, 3.0, 3.33333333,
            3.0, 3.33333333, 3.66666667, 4.0,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_linear_half_pixel_symmetric)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_HALF_PIXEL_SYMMETRIC,
        Size(2, 2), Size(5, 4), Size(), Point2d(2.94, 2.3), -0.75f,
        {1, 2, 3, 4},
        {
            1.0       , 1.15986395, 1.5       , 1.84013605, 2.0       ,
            1.56521738, 1.72508133, 2.06521738, 2.40535343, 2.56521738,
            2.43478262, 2.59464657, 2.93478262, 3.27491867, 3.43478262,
            3.0       , 3.15986395, 3.5       , 3.84013605, 4.0       ,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_scales_nearest)
{
    ResizeOnnx{
        INTER_NEAREST,
        Size(2, 2), Size(6, 4), Size(), Point2d(3.0, 2.0), -0.75f,
        {1, 2, 3, 4},
        {
            1, 1, 1, 2, 2, 2,
            1, 1, 1, 2, 2, 2,
            3, 3, 3, 4, 4, 4,
            3, 3, 3, 4, 4, 4,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_sizes_cubic)
{
    ResizeOnnx{
        INTER_CUBIC,
        Size(4, 4), Size(10, 9), Size(10, 9), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            0.45507922, 0.64057922, 0.97157922, 1.42257922, 1.90732922, 2.22332922, 2.70807922, 3.15907922, 3.49007922, 3.67557922,
            1.39437963, 1.57987963, 1.91087963, 2.36187963, 2.84662963, 3.16262963, 3.64737963, 4.09837963, 4.42937963, 4.61487963,
            2.95130693, 3.13680693, 3.46780693, 3.91880693, 4.40355693, 4.71955693, 5.20430693, 5.65530693, 5.98630693, 6.17180693,
            5.20525069, 5.39075069, 5.72175069, 6.17275069, 6.65750069, 6.97350069, 7.45825069, 7.90925069, 8.24025069, 8.42575069,
            6.88975, 7.07525, 7.40625, 7.85725, 8.342, 8.658, 9.14275, 9.59375, 9.92475, 10.11025,
            8.57424931, 8.75974931, 9.09074931, 9.54174931, 10.02649931, 10.34249931, 10.82724931, 11.27824931, 11.60924931, 11.79474931,
            10.82819307, 11.01369307, 11.34469307, 11.79569307, 12.28044307, 12.59644307, 13.08119307, 13.53219307, 13.86319307, 14.04869307,
            12.38512037, 12.57062037, 12.90162037, 13.35262037, 13.83737037, 14.15337037, 14.63812037, 15.08912037, 15.42012037, 15.60562037,
            13.32442078, 13.50992078, 13.84092078, 14.29192078, 14.77667078, 15.09267078, 15.57742078, 16.02842078, 16.35942078, 16.54492078,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_sizes_nearest)
{
    ResizeOnnx{
        INTER_NEAREST,
        Size(2, 2), Size(8, 7), Size(8, 7), Point2d(), -0.75f,
        {1, 2, 3, 4},
        {
            1, 1, 1, 1, 2, 2, 2, 2,
            1, 1, 1, 1, 2, 2, 2, 2,
            1, 1, 1, 1, 2, 2, 2, 2,
            1, 1, 1, 1, 2, 2, 2, 2,
            3, 3, 3, 3, 4, 4, 4, 4,
            3, 3, 3, 3, 4, 4, 4, 4,
            3, 3, 3, 3, 4, 4, 4, 4,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_sizes_nearest_ceil_half_pixel)
{
    ResizeOnnx{
        INTER_NEAREST | INTER_NEAREST_CEIL,
        Size(4, 4), Size(8, 8), Size(8, 8), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            1, 2, 2, 3, 3, 4, 4, 4,
            5, 6, 6, 7, 7, 8, 8, 8,
            5, 6, 6, 7, 7, 8, 8, 8,
            9, 10, 10, 11, 11, 12, 12, 12,
            9, 10, 10, 11, 11, 12, 12, 12,
            13, 14, 14, 15, 15, 16, 16, 16,
            13, 14, 14, 15, 15, 16, 16, 16,
            13, 14, 14, 15, 15, 16, 16, 16,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_sizes_nearest_floor_align_corners)
{
    ResizeOnnx{
        INTER_NEAREST | INTER_NEAREST_FLOOR | INTER_ALIGN_CORNERS,
        Size(4, 4), Size(8, 8), Size(8, 8), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            1, 1, 1, 2, 2, 3, 3, 4,
            1, 1, 1, 2, 2, 3, 3, 4,
            1, 1, 1, 2, 2, 3, 3, 4,
            5, 5, 5, 6, 6, 7, 7, 8,
            5, 5, 5, 6, 6, 7, 7, 8,
            9, 9, 9, 10, 10, 11, 11, 12,
            9, 9, 9, 10, 10, 11, 11, 12,
            13, 13, 13, 14, 14, 15, 15, 16,
        }
    }.run();
}

TEST(ResizeOnnx, upsample_sizes_nearest_round_prefer_ceil_asymmetric)
{
    ResizeOnnx{
        INTER_NEAREST | INTER_NEAREST_PREFER_CEIL | INTER_ASYMMETRIC,
        Size(4, 4), Size(8, 8), Size(8, 8), Point2d(), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
            1, 2, 2, 3, 3, 4, 4, 4,
            5, 6, 6, 7, 7, 8, 8, 8,
            5, 6, 6, 7, 7, 8, 8, 8,
            9, 10, 10, 11, 11, 12, 12, 12,
            9, 10, 10, 11, 11, 12, 12, 12,
            13, 14, 14, 15, 15, 16, 16, 16,
            13, 14, 14, 15, 15, 16, 16, 16,
            13, 14, 14, 15, 15, 16, 16, 16,
        }
    }.run();
}

/*
import numpy as np
import onnx
from onnx.reference.ops.op_resize import (
    _interpolate_nd,
    _cubic_coeffs, _cubic_coeffs_antialias,
    _linear_coeffs, _linear_coeffs_antialias
)
data = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
scales = np.array([0.8, 0.8], dtype=np.float64)
*/

/*
output = _interpolate_nd(
    data,
    lambda x, s: _cubic_coeffs_antialias(x, s, A=-0.5),
    scale_factors=scales,
    exclude_outside=True,
)
*/
TEST(ResizeOnnx, downsample_scales_cubic_antialias_A_n0p5_exclude_outside)
{
    ResizeOnnx{
        INTER_CUBIC | INTER_ANTIALIAS | INTER_EXCLUDE_OUTSIDE,
        Size(4, 4), Size(3, 3), Size(), Point2d(0.8, 0.8), -0.5f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             1.68342335,  2.90749817,  4.22822584,
             6.57972264,  7.80379747,  9.12452513,
            11.86263331, 13.08670813, 14.4074358 ,
        }
    }.run();
}

/*
output = _interpolate_nd(
    data,
    _linear_coeffs_antialias,
    scale_factors=scales,
    exclude_outside=True,
)
*/
TEST(ResizeOnnx, downsample_scales_linear_antialias_exclude_outside)
{
    ResizeOnnx{
        INTER_LINEAR | INTER_ANTIALIAS | INTER_EXCLUDE_OUTSIDE,
        Size(4, 4), Size(3, 3), Size(), Point2d(0.8, 0.8), -0.75f,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {
             2.25      ,  3.41666667,  4.58333333,
             6.91666667,  8.08333333,  9.25      ,
            11.58333333, 12.75      , 13.91666667,
        }
    }.run();
}

}}
