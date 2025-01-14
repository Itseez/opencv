/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_HAL_UTILS_HPP_INCLUDED
#define OPENCV_FASTCV_HAL_UTILS_HPP_INCLUDED

#include "fastcv.h"
#include <opencv2/core/utils/logger.hpp>

#define INITIALIZATION_CHECK                                        \
{                                                                   \
    if (!FastCvContext::getContext().isInitialized)                 \
    {                                                               \
        return CV_HAL_ERROR_UNKNOWN;                                \
    }                                                               \
}

#define CV_HAL_RETURN(status, func)                                         \
{                                                                           \
    if( status == FASTCV_SUCCESS )                                          \
    {                                                                       \
        CV_LOG_DEBUG(NULL, "FastCV HAL for "<<#func<<" run successfully!"); \
        return CV_HAL_ERROR_OK;                                             \
    }                                                                       \
    else if(status == FASTCV_EBADPARAM || status == FASTCV_EUNALIGNPARAM || \
            status == FASTCV_EUNSUPPORTED || status == FASTCV_EHWQDSP ||    \
            status == FASTCV_EHWGPU)                                        \
    {                                                                       \
        CV_LOG_DEBUG(NULL, "FastCV status:"<<getFastCVErrorString(status)   \
            <<", Switching to default OpenCV solution!");                   \
        return CV_HAL_ERROR_NOT_IMPLEMENTED;                                \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        CV_LOG_ERROR(NULL,"FastCV error:"<<getFastCVErrorString(status));   \
        return CV_HAL_ERROR_UNKNOWN;                                        \
    }                                                                       \
}

#define CV_HAL_RETURN_NOT_IMPLEMENTED(reason)                           \
{                                                                       \
    CV_LOG_DEBUG(NULL,"Switching to default OpenCV\nInfo: "<<reason);   \
    return CV_HAL_ERROR_NOT_IMPLEMENTED;                                \
}

#define FCV_KernelSize_SHIFT 3
#define FCV_MAKETYPE(ksize,depth) ((ksize<<FCV_KernelSize_SHIFT) + depth)
#define FCV_CMP_EQ(val1,val2) (fabs(val1 - val2) < FLT_EPSILON)

const char* getFastCVErrorString(int status);
const char* borderToString(int border);
const char* interpolationToString(int interpolation);

struct FastCvContext
{
public:
    // initialize at first call
    // Defines a static local variable context. Variable is created only once.
    static FastCvContext& getContext()
    {
        static FastCvContext context;
        return context;
    }

    FastCvContext()
    {
        if (fcvSetOperationMode(FASTCV_OP_CPU_PERFORMANCE) != 0)
        {
            CV_LOG_WARNING(NULL, "Failed to switch FastCV operation mode");
            isInitialized = false;
        }
        else
        {
            CV_LOG_INFO(NULL, "FastCV Operation Mode Switched");
            isInitialized = true;
        }
    }

    bool isInitialized;
};

#endif