// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "op_halide.hpp"
#include "op_inf_engine.hpp"
#include "ie_ngraph.hpp"
#include "op_vkcom.hpp"
#include "op_cuda.hpp"
#include "op_webnn.hpp"
#include "op_timvx.hpp"
#include "op_cann.hpp"

#include "halide_scheduler.hpp"

#include "backend.hpp"
#include "factory.hpp"

#ifdef HAVE_CUDA
#include "cuda4dnn/init.hpp"
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


class BackendRegistry
{
public:
    typedef std::vector< std::pair<Backend, Target> > BackendsList;
    const BackendsList & getBackends() const { return backends; }
    static BackendRegistry & getRegistry()
    {
        static BackendRegistry impl;
        return impl;
    }


private:
    BackendRegistry()
    {
#ifdef HAVE_HALIDE
        backends.push_back(std::make_pair(DNN_BACKEND_HALIDE, DNN_TARGET_CPU));
#ifdef HAVE_OPENCL
        if (cv::ocl::useOpenCL())
            backends.push_back(std::make_pair(DNN_BACKEND_HALIDE, DNN_TARGET_OPENCL));
#endif
#endif  // HAVE_HALIDE

        bool haveBackendOpenVINO = false;
#ifdef HAVE_INF_ENGINE
        haveBackendOpenVINO = true;
#elif defined(ENABLE_PLUGINS)
        {
            auto factory = dnn_backend::createPluginDNNBackendFactory("openvino");
            if (factory)
            {
                auto backend = factory->createNetworkBackend();
                if (backend)
                    haveBackendOpenVINO = true;
            }
        }
#endif

        bool haveBackendCPU_FP16 = false;
#if defined(__arm64__) && __arm64__
        haveBackendCPU_FP16 = true;
#endif

        if (haveBackendOpenVINO && openvino::checkTarget(DNN_TARGET_CPU))
        {
            backends.push_back(std::make_pair(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, DNN_TARGET_CPU));
        }
        if (haveBackendOpenVINO && openvino::checkTarget(DNN_TARGET_MYRIAD))
        {
            backends.push_back(std::make_pair(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, DNN_TARGET_MYRIAD));
        }
        if (haveBackendOpenVINO && openvino::checkTarget(DNN_TARGET_HDDL))
        {
            backends.push_back(std::make_pair(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, DNN_TARGET_HDDL));
        }
#ifdef HAVE_OPENCL
        if (cv::ocl::useOpenCL() && ocl::Device::getDefault().isIntel())
        {
            if (haveBackendOpenVINO && openvino::checkTarget(DNN_TARGET_OPENCL))
            {
                backends.push_back(std::make_pair(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, DNN_TARGET_OPENCL));
            }
            if (haveBackendOpenVINO && openvino::checkTarget(DNN_TARGET_OPENCL_FP16))
            {
                backends.push_back(std::make_pair(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, DNN_TARGET_OPENCL_FP16));
            }
        }
#endif  // HAVE_OPENCL

#ifdef HAVE_WEBNN
        if (haveWebnn())
        {
            backends.push_back(std::make_pair(DNN_BACKEND_WEBNN, DNN_TARGET_CPU));
        }
#endif  // HAVE_WEBNN

#ifdef HAVE_OPENCL
        if (cv::ocl::useOpenCL())
        {
            backends.push_back(std::make_pair(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL));
            backends.push_back(std::make_pair(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL_FP16));
        }
#endif

        backends.push_back(std::make_pair(DNN_BACKEND_OPENCV, DNN_TARGET_CPU));

        if (haveBackendCPU_FP16)
            backends.push_back(std::make_pair(DNN_BACKEND_OPENCV, DNN_TARGET_CPU_FP16));

#ifdef HAVE_VULKAN
        if (haveVulkan())
            backends.push_back(std::make_pair(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN));
#endif

#ifdef HAVE_CUDA
        cuda4dnn::checkVersions();

        bool hasCudaCompatible = false;
        bool hasCudaFP16 = false;
        for (int i = 0; i < cuda4dnn::getDeviceCount(); i++)
        {
            if (cuda4dnn::isDeviceCompatible(i))
            {
                hasCudaCompatible = true;
                if (cuda4dnn::doesDeviceSupportFP16(i))
                {
                    hasCudaFP16 = true;
                    break; // we already have all we need here
                }
            }
        }

        if (hasCudaCompatible)
        {
            backends.push_back(std::make_pair(DNN_BACKEND_CUDA, DNN_TARGET_CUDA));
            if (hasCudaFP16)
                backends.push_back(std::make_pair(DNN_BACKEND_CUDA, DNN_TARGET_CUDA_FP16));
        }
#endif

#ifdef HAVE_TIMVX
        if (haveTimVX())
        {
            backends.push_back(std::make_pair(DNN_BACKEND_TIMVX, DNN_TARGET_NPU));
        }
#endif

#ifdef HAVE_CANN
        backends.push_back(std::make_pair(DNN_BACKEND_CANN, DNN_TARGET_NPU));
#endif

#ifdef HAVE_TRT
        backends.push_back(std::make_pair(DNN_BACKEND_TENSORRT, DNN_TARGET_TENSORRT));
#endif
    }

    BackendsList backends;
};


std::vector<std::pair<Backend, Target>> getAvailableBackends()
{
    return BackendRegistry::getRegistry().getBackends();
}

std::vector<Target> getAvailableTargets(Backend be)
{
    if (be == DNN_BACKEND_DEFAULT)
        be = (Backend)getParam_DNN_BACKEND_DEFAULT();

    if (be == DNN_BACKEND_INFERENCE_ENGINE)
        be = DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;

    std::vector<Target> result;
    const BackendRegistry::BackendsList all_backends = getAvailableBackends();
    for (BackendRegistry::BackendsList::const_iterator i = all_backends.begin(); i != all_backends.end(); ++i)
    {
        if (i->first == be)
            result.push_back(i->second);
    }
    return result;
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
