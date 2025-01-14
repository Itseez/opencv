#ifndef CV_IMGPROC_HAL_HPP
#define CV_IMGPROC_HAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/hal/interface.h"

namespace cv { namespace hal {

//! @addtogroup imgproc_hal_functions
//! @{

//---------------------------
//! @cond IGNORED

struct CV_EXPORTS Filter2D
{
    CV_DEPRECATED static Ptr<hal::Filter2D> create(uchar * , size_t , int ,
                                     int , int ,
                                     int , int ,
                                     int , int ,
                                     int , double ,
                                     int , int ,
                                     bool , bool );
    virtual void apply(uchar * , size_t ,
                       uchar * , size_t ,
                       int , int ,
                       int , int ,
                       int , int ) = 0;
    virtual ~Filter2D() {}
};

struct CV_EXPORTS SepFilter2D
{
    CV_DEPRECATED static Ptr<hal::SepFilter2D> create(int , int , int ,
                                        uchar * , int ,
                                        uchar * , int ,
                                        int , int ,
                                        double , int );
    virtual void apply(uchar * , size_t ,
                       uchar * , size_t ,
                       int , int ,
                       int , int ,
                       int , int ) = 0;
    virtual ~SepFilter2D() {}
};


struct CV_EXPORTS Morph
{
    CV_DEPRECATED static Ptr<hal::Morph> create(int , int , int , int , int ,
                                    int , uchar * , size_t ,
                                    int , int ,
                                    int , int ,
                                    int , const double *,
                                    int , bool , bool );
    virtual void apply(uchar * , size_t , uchar * , size_t , int , int ,
                       int , int , int , int ,
                       int , int , int , int ) = 0;
    virtual ~Morph() {}
};

//! @endcond
//---------------------------

CV_EXPORTS void filter2D(int stype, int dtype, int kernel_type,
                         uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int full_width, int full_height,
                         int offset_x, int offset_y,
                         uchar * kernel_data, size_t kernel_step,
                         int kernel_width, int kernel_height,
                         int anchor_x, int anchor_y,
                         double delta, int borderType,
                         bool isSubmatrix);

CV_EXPORTS void sepFilter2D(int stype, int dtype, int ktype,
                            uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int full_width, int full_height,
                            int offset_x, int offset_y,
                            uchar * kernelx_data, int kernelx_len,
                            uchar * kernely_data, int kernely_len,
                            int anchor_x, int anchor_y,
                            double delta, int borderType);

CV_EXPORTS void morph(int op, int src_type, int dst_type,
                      uchar * src_data, size_t src_step,
                      uchar * dst_data, size_t dst_step,
                      int width, int height,
                      int roi_width, int roi_height, int roi_x, int roi_y,
                      int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                      int kernel_type, uchar * kernel_data, size_t kernel_step,
                      int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                      int borderType, const double borderValue[4],
                      int iterations, bool isSubmatrix);


CV_EXPORTS void resize(int src_type,
                       const uchar * src_data, size_t src_step, int src_width, int src_height,
                       uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation);

CV_EXPORTS void warpAffine(int src_type,
                           const uchar * src_data, size_t src_step, int src_width, int src_height,
                           uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                           const double M[6], int interpolation, int borderType, const double borderValue[4]);

CV_EXPORTS void warpAffineBlocklineNN(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw);

CV_EXPORTS void warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw);

CV_EXPORTS void warpPerspective(int src_type,
                               const uchar * src_data, size_t src_step, int src_width, int src_height,
                               uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                               const double M[9], int interpolation, int borderType, const double borderValue[4]);

CV_EXPORTS void warpPerspectiveBlocklineNN(const double *M, short* xy, double X0, double Y0, double W0, int bw);

CV_EXPORTS void warpPerspectiveBlockline(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw);

CV_EXPORTS void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, int dcn, bool swapBlue);

CV_EXPORTS void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                               uchar * dst_data, size_t dst_step,
                               int width, int height,
                               int scn, bool swapBlue, int greenBits);

CV_EXPORTS void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                               uchar * dst_data, size_t dst_step,
                               int width, int height,
                               int dcn, bool swapBlue, int greenBits);

CV_EXPORTS void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             int depth, int scn, bool swapBlue);

CV_EXPORTS void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             int depth, int dcn);

CV_EXPORTS void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int width, int height,
                                int greenBits);

CV_EXPORTS void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int width, int height,
                                int greenBits);
CV_EXPORTS void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue, bool isCbCr,
                            AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue, bool isCbCr,
                            AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue);

CV_EXPORTS void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue);

CV_EXPORTS void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV);

CV_EXPORTS void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV);

CV_EXPORTS void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue, bool isLab, bool srgb);

CV_EXPORTS void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue, bool isLab, bool srgb);

CV_EXPORTS void cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    int dcn, bool swapBlue, int uIdx,
                                    AlgorithmHint hint = ALGO_HINT_DEFAULT);

//! Separate Y and UV planes
CV_EXPORTS void cvtTwoPlaneYUVtoBGR(const uchar * y_data, const uchar * uv_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    int dcn, bool swapBlue, int uIdx,
                                    AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtTwoPlaneYUVtoBGR(const uchar * y_data, size_t y_step, const uchar * uv_data, size_t uv_step,
                                    uchar * dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    int dcn, bool swapBlue, int uIdx,
                                    AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                      uchar * dst_data, size_t dst_step,
                                      int dst_width, int dst_height,
                                      int dcn, bool swapBlue, int uIdx,
                                      AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                                      uchar * dst_data, size_t dst_step,
                                      int width, int height,
                                      int scn, bool swapBlue, int uIdx,
                                      AlgorithmHint hint = ALGO_HINT_DEFAULT);

//! Separate Y and UV planes
CV_EXPORTS void cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                                    uchar * y_data, uchar * uv_data, size_t dst_step,
                                    int width, int height,
                                    int scn, bool swapBlue, int uIdx);

CV_EXPORTS void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int width, int height,
                                    int dcn, bool swapBlue, int uIdx, int ycn,
                                    AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtOnePlaneBGRtoYUV(const uchar * src_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int width, int height,
                                    int scn, bool swapBlue, int uIdx, int ycn,
                                    AlgorithmHint hint = ALGO_HINT_DEFAULT);

CV_EXPORTS void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                                        uchar * dst_data, size_t dst_step,
                                        int width, int height);

CV_EXPORTS void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                                        uchar * dst_data, size_t dst_step,
                                        int width, int height);

CV_EXPORTS void integral(int depth, int sdepth, int sqdepth,
                         const uchar* src, size_t srcstep,
                         uchar* sum, size_t sumstep,
                         uchar* sqsum, size_t sqsumstep,
                         uchar* tilted, size_t tstep,
                         int width, int height, int cn);

//! @}

}}

#endif // CV_IMGPROC_HAL_HPP
