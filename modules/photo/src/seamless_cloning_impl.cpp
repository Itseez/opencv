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

#include "seamless_cloning.hpp"

using namespace cv;
using namespace std;


void Cloning::computeGradientX( const Mat &img, Mat &gx)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0,2) = 1;
    kernel.at<char>(0,1) = -1;

    if(img.channels() == 3)
    {
        filter2D(img, gx, CV_32F, kernel);
    }
    else if (img.channels() == 1)
    {
        filter2D(img, gx, CV_32F, kernel);
        cvtColor(gx, gx, COLOR_GRAY2BGR);
    }
}

void Cloning::computeGradientY( const Mat &img, Mat &gy)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(2,0) = 1;
    kernel.at<char>(1,0) = -1;

    if(img.channels() == 3)
    {
        filter2D(img, gy, CV_32F, kernel);
    }
    else if (img.channels() == 1)
    {
        filter2D(img, gy, CV_32F, kernel);
        cvtColor(gy, gy, COLOR_GRAY2BGR);
    }
}

void Cloning::computeLaplacianX( const Mat &img, Mat &laplacianX)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0,0) = -1;
    kernel.at<char>(0,1) = 1;
    filter2D(img, laplacianX, CV_32F, kernel);
}

void Cloning::computeLaplacianY( const Mat &img, Mat &laplacianY)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(0,0) = -1;
    kernel.at<char>(1,0) = 1;
    filter2D(img, laplacianY, CV_32F, kernel);
}

void Cloning::dst(const Mat& src, Mat& dest, bool invert)
{
    Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

    int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE: DFT_ROWS;

    src.copyTo(temp(Rect(1,0, src.cols, src.rows)));

    for(int j = 0 ; j < src.rows ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        const float * srcLinePtr = src.ptr<float>(j);
        for(int i = 0 ; i < src.cols ; ++i)
        {
            tempLinePtr[src.cols + 2 + i] = - srcLinePtr[src.cols - 1 - i];
        }
    }

    Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};
    Mat complex;

    merge(planes, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

    for(int j = 0 ; j < src.cols ; ++j)
    {
        float * tempLinePtr = temp.ptr<float>(j);
        for(int i = 0 ; i < src.rows ; ++i)
        {
            float val = planes[1].ptr<float>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = - val;
        }
    }

    Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};

    merge(planes2, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes2);

    temp = planes2[1].t();
    temp(Rect( 0, 1, src.cols, src.rows)).copyTo(dest);
}

void Cloning::solve(const Mat &img, Mat& mod_diff, Mat &result)
{
    const int w = img.cols;
    const int h = img.rows;

    Mat res;
    dst(mod_diff, res);

    for(int j = 0 ; j < h-2; j++)
    {
        float * resLinePtr = res.ptr<float>(j);
        for(int i = 0 ; i < w-2; i++)
        {
            resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }

    dst(res, mod_diff, true);

    unsigned char *  resLinePtr = result.ptr<unsigned char>(0);
    const unsigned char * imgLinePtr = img.ptr<unsigned char>(0);
    const float * interpLinePtr = NULL;

     //first col
    for(int i = 0 ; i < w ; ++i)
        result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

    for(int j = 1 ; j < h-1 ; ++j)
    {
        resLinePtr = result.ptr<unsigned char>(j);
        imgLinePtr  = img.ptr<unsigned char>(j);
        interpLinePtr = mod_diff.ptr<float>(j-1);

        //first row
        resLinePtr[0] = imgLinePtr[0];

        for(int i = 1 ; i < w-1 ; ++i)
        {
            //saturate cast is not used here, because it behaves differently from the previous implementation
            //most notable, saturate_cast rounds before truncating, here it's the opposite.
            float value = interpLinePtr[i-1];
            if(value < 0.)
                resLinePtr[i] = 0;
            else if (value > 255.0)
                resLinePtr[i] = 255;
            else
                resLinePtr[i] = static_cast<unsigned char>(value);
        }

        //last row
        resLinePtr[w-1] = imgLinePtr[w-1];
    }

    //last col
    resLinePtr = result.ptr<unsigned char>(h-1);
    imgLinePtr = img.ptr<unsigned char>(h-1);
    for(int i = 0 ; i < w ; ++i)
        resLinePtr[i] = imgLinePtr[i];
}

void Cloning::poissonSolver(const Mat &img, Mat &laplacianX , Mat &laplacianY, Mat &result)
{
    const int w = img.cols;
    const int h = img.rows;

    Mat lap = laplacianX + laplacianY;

    Mat bound = img.clone();

    rectangle(bound, Point(1, 1), Point(img.cols-2, img.rows-2), Scalar::all(0), -1);
    Mat boundary_points;
    Laplacian(bound, boundary_points, CV_32F);

    boundary_points = lap - boundary_points;

    Mat mod_diff = boundary_points(Rect(1, 1, w-2, h-2));

    solve(img,mod_diff,result);
}

void Cloning::initVariables(const Mat &destination, const Mat &binaryMask)
{
    destinationGradientX = Mat(destination.size(),CV_32FC3);
    destinationGradientY = Mat(destination.size(),CV_32FC3);
    patchGradientX = Mat(destination.size(),CV_32FC3);
    patchGradientY = Mat(destination.size(),CV_32FC3);

    binaryMaskFloat = Mat(binaryMask.size(),CV_32FC1);
    binaryMaskFloatInverted = Mat(binaryMask.size(),CV_32FC1);

    //init of the filters used in the dst
    const int w = destination.cols;
    filter_X.resize(w - 2);
    double scale = CV_PI / (w - 1);
    for(int i = 0 ; i < w-2 ; ++i)
        filter_X[i] = 2.0f * (float)std::cos(scale * (i + 1));

    const int h  = destination.rows;
    filter_Y.resize(h - 2);
    scale = CV_PI / (h - 1);
    for(int j = 0 ; j < h - 2 ; ++j)
        filter_Y[j] = 2.0f * (float)std::cos(scale * (j + 1));
}

void Cloning::computeDerivatives(const Mat& destination, const Mat &patch, Mat &binaryMask)
{
    initVariables(destination, binaryMask);

    computeGradientX(destination, destinationGradientX);
    computeGradientY(destination, destinationGradientY);

    computeGradientX(patch, patchGradientX);
    computeGradientY(patch, patchGradientY);

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));
    erode(binaryMask, binaryMask, Kernel, Point(-1,-1), 3);

    binaryMask.convertTo(binaryMaskFloat, CV_32FC1, 1.0/255.0);
}

void Cloning::scalarProduct(Mat mat, float r, float g, float b)
{
    vector <Mat> channels;
    split(mat,channels);
    multiply(channels[2],r,channels[2]);
    multiply(channels[1],g,channels[1]);
    multiply(channels[0],b,channels[0]);
    merge(channels,mat);
}

void Cloning::arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const
{
    vector <Mat> lhs_channels;
    vector <Mat> result_channels;

    split(lhs,lhs_channels);
    split(result,result_channels);

    for(int chan = 0 ; chan < 3 ; ++chan)
        multiply(lhs_channels[chan],rhs,result_channels[chan]);

    merge(result_channels,result);
}

void Cloning::poisson(const Mat &destination)
{
    Mat laplacianX = destinationGradientX + patchGradientX;
    Mat laplacianY = destinationGradientY + patchGradientY;

    computeLaplacianX(laplacianX,laplacianX);
    computeLaplacianY(laplacianY,laplacianY);

    split(laplacianX,rgbx_channel);
    split(laplacianY,rgby_channel);

    split(destination,output);

    for(int chan = 0 ; chan < 3 ; ++chan)
    {
        poissonSolver(output[chan], rgbx_channel[chan], rgby_channel[chan], output[chan]);
    }
}

void Cloning::evaluate(const Mat &I, Mat &wmask, const Mat &cloned)
{
    bitwise_not(wmask,wmask);

    wmask.convertTo(binaryMaskFloatInverted,CV_32FC1,1.0/255.0);

    arrayProduct(destinationGradientX, binaryMaskFloatInverted, destinationGradientX);
    arrayProduct(destinationGradientY, binaryMaskFloatInverted, destinationGradientY);

    poisson(I);

    merge(output,cloned);
}

void Cloning::normalClone(const Mat &destination, const Mat &patch, Mat &binaryMask, Mat &cloned, int flag)
{
    const int w = destination.cols;
    const int h = destination.rows;
    const int channel = destination.channels();
    const int n_elem_in_line = w * channel;

    computeDerivatives(destination,patch,binaryMask);

    switch(flag)
    {
        case NORMAL_CLONE:
        case NORMAL_CLONE_WIDE:
            arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
            arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
            break;

        case MIXED_CLONE:
        case MIXED_CLONE_WIDE:
        {
            AutoBuffer<int> maskIndices(n_elem_in_line);
            for (int i = 0; i < n_elem_in_line; ++i)
                maskIndices[i] = i / channel;

            for(int i=0;i < h; i++)
            {
                float * patchXLinePtr = patchGradientX.ptr<float>(i);
                float * patchYLinePtr = patchGradientY.ptr<float>(i);
                const float * destinationXLinePtr = destinationGradientX.ptr<float>(i);
                const float * destinationYLinePtr = destinationGradientY.ptr<float>(i);
                const float * binaryMaskLinePtr = binaryMaskFloat.ptr<float>(i);

                for(int j=0; j < n_elem_in_line; j++)
                {
                    int maskIndex = maskIndices[j];

                    if(abs(patchXLinePtr[j] - patchYLinePtr[j]) >
                       abs(destinationXLinePtr[j] - destinationYLinePtr[j]))
                    {
                        patchXLinePtr[j] *= binaryMaskLinePtr[maskIndex];
                        patchYLinePtr[j] *= binaryMaskLinePtr[maskIndex];
                    }
                    else
                    {
                        patchXLinePtr[j] = destinationXLinePtr[j]
                            * binaryMaskLinePtr[maskIndex];
                        patchYLinePtr[j] = destinationYLinePtr[j]
                            * binaryMaskLinePtr[maskIndex];
                    }
                }
            }
        }
        break;

        case MONOCHROME_TRANSFER:
        case MONOCHROME_TRANSFER_WIDE:
            Mat gray;
            cvtColor(patch, gray, COLOR_BGR2GRAY );

            computeGradientX(gray,patchGradientX);
            computeGradientY(gray,patchGradientY);

            arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
            arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
        break;

    }

    evaluate(destination,binaryMask,cloned);
}

void Cloning::localColorChange(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul=1.0,
                                 float green_mul=1.0, float blue_mul=1.0)
{
    computeDerivatives(I,mask,wmask);

    arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);
    scalarProduct(patchGradientX,red_mul,green_mul,blue_mul);
    scalarProduct(patchGradientY,red_mul,green_mul,blue_mul);

    evaluate(I,wmask,cloned);
}

void Cloning::illuminationChange(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta)
{
    CV_INSTRUMENT_REGION();

    computeDerivatives(I,mask,wmask);

    arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);

    Mat mag;
    magnitude(patchGradientX,patchGradientY,mag);

    Mat multX, multY, multx_temp, multy_temp;

    multiply(patchGradientX,pow(alpha,beta),multX);
    pow(mag,-1*beta, multx_temp);
    multiply(multX,multx_temp, patchGradientX);
    patchNaNs(patchGradientX);

    multiply(patchGradientY,pow(alpha,beta),multY);
    pow(mag,-1*beta, multy_temp);
    multiply(multY,multy_temp,patchGradientY);
    patchNaNs(patchGradientY);

    Mat zeroMask = (patchGradientX != 0);

    patchGradientX.copyTo(patchGradientX, zeroMask);
    patchGradientY.copyTo(patchGradientY, zeroMask);

    evaluate(I,wmask,cloned);
}

void Cloning::textureFlatten(Mat &I, Mat &mask, Mat &wmask, float low_threshold,
        float high_threshold, int kernel_size, Mat &cloned)
{
    computeDerivatives(I,mask,wmask);

    Mat out;
    Canny(mask,out,low_threshold,high_threshold,kernel_size);

    Mat zeros = Mat::zeros(patchGradientX.size(), CV_32FC3);
    Mat zerosMask = (out != 255);
    zeros.copyTo(patchGradientX, zerosMask);
    zeros.copyTo(patchGradientY, zerosMask);

    arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);

    evaluate(I,wmask,cloned);
}
