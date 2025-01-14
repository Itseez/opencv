// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_CUDA
#include "op_cuda.hpp"
#include "cuda4dnn/init.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN


void Net::Impl::initCUDABackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_Assert(preferableBackend == DNN_BACKEND_CUDA);

    if (!cudaInfo) /* we need to check only once */
        cuda4dnn::checkVersions();

    if (cuda4dnn::getDeviceCount() <= 0)
        CV_Error(Error::StsError, "No CUDA capable device found.");

    if (cuda4dnn::getDevice() < 0)
        CV_Error(Error::StsError, "No CUDA capable device selected.");

    if (!cuda4dnn::isDeviceCompatible())
        CV_Error(Error::GpuNotSupported, "OpenCV was not built to work with the selected device. Please check CUDA_ARCH_PTX or CUDA_ARCH_BIN in your build configuration.");

    if (preferableTarget == DNN_TARGET_CUDA_FP16 && !cuda4dnn::doesDeviceSupportFP16())
    {
        CV_LOG_WARNING(NULL, "The selected CUDA device does not support FP16 target; switching to FP32 target.");
        preferableTarget = DNN_TARGET_CUDA;
    }

    if (!cudaInfo)
    {
        cuda4dnn::csl::CSLContext context;
        context.stream = cuda4dnn::csl::Stream(true);
        context.cublas_handle = cuda4dnn::csl::cublas::Handle(context.stream);
        context.cudnn_handle = cuda4dnn::csl::cudnn::Handle(context.stream);

        auto d2h_stream = cuda4dnn::csl::Stream(true);  // stream for background D2H data transfers
        cudaInfo = std::unique_ptr<CudaInfo_t>(new CudaInfo_t(std::move(context), std::move(d2h_stream)));
    }

    cudaInfo->workspace = cuda4dnn::csl::Workspace();  // release workspace memory if any

    for (auto& layer : layers)
    {
        auto& ld = layer.second;

        if (ld.id == 0 && netInputLayer->supportBackend(preferableBackend))
        {
            for (auto& wrapper : ld.inputBlobsWrappers)
            {
                auto cudaWrapper = wrapper.dynamicCast<CUDABackendWrapper>();
                cudaWrapper->setStream(cudaInfo->context.stream, cudaInfo->d2h_stream);
            }
        }

        for (auto& wrapper : ld.outputBlobsWrappers)
        {
            auto cudaWrapper = wrapper.dynamicCast<CUDABackendWrapper>();
            cudaWrapper->setStream(cudaInfo->context.stream, cudaInfo->d2h_stream);
        }
    }

    for (auto& layer : layers)
    {
        auto& ld = layer.second;
        auto& layerInstance = ld.layerInstance;

        if (!layerInstance->supportBackend(DNN_BACKEND_CUDA))
        {
            std::ostringstream os;
            os << "CUDA backend will fallback to the CPU implementation for the layer \"" << ld.name
               << "\" of type " << ld.type << '\n';
            CV_LOG_INFO(NULL, os.str().c_str());
            continue;
        }

        /* we make a copy so that `initCUDA` doesn't modify `cudaInfo->context` */
        auto context = cudaInfo->context;
        auto node = layerInstance->initCUDA(&context, ld.inputBlobsWrappers, ld.outputBlobsWrappers);
        ld.backendNodes[DNN_BACKEND_CUDA] = node;

        if(!node.empty())
        {
            auto cudaNode = node.dynamicCast<CUDABackendNode>();
            cudaInfo->workspace.require(cudaNode->get_workspace_memory_in_bytes());
        }
    }

    if (blobsToKeep_.size() > 1)
    {
        for (const auto& pin : blobsToKeep_)
        {
            LayerData& ld = layers[pin.lid];
            ld.cudaD2HBackgroundTransfers.push_back(pin.oid);
        }
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // HAVE_CUDA

namespace cv { namespace dnn {

bool haveCUDA()
{
#ifdef HAVE_CUDA
    int dev = 0;
    static bool ret = (cudaGetDevice(&dev) == cudaSuccess);
    return ret;
#else
    return false;
#endif
}

}}  // namespace cv::dnn
