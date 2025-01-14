// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_OP_CUDA_HPP
#define OPENCV_DNN_SRC_OP_CUDA_HPP

#ifdef HAVE_CUDA
#include "cuda4dnn/csl/stream.hpp"
#include "cuda4dnn/csl/event.hpp"
#include "cuda4dnn/csl/cublas.hpp"
#include "cuda4dnn/csl/cudnn.hpp"
#include "cuda4dnn/csl/tensor.hpp"
#include "cuda4dnn/csl/memory.hpp"
#include "cuda4dnn/csl/workspace.hpp"
#include "cuda4dnn/kernels/fp_conversion.hpp"
#endif

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <iterator>

namespace cv { namespace dnn {

    constexpr bool IS_DNN_CUDA_TARGET(int id) {
        return id == DNN_TARGET_CUDA_FP16 || id == DNN_TARGET_CUDA;
    }

    bool haveCUDA();

#ifdef HAVE_CUDA
    namespace cuda4dnn { namespace csl {
        struct CSLContext {
            Stream stream;
            cublas::Handle cublas_handle;
            cudnn::Handle cudnn_handle;
        };

        /** @brief creates Tensor object from cv::Mat (only the header is created, i.e. no data is copied)
         *
         * \tparam      T   element type for the tensor
         * \param[in]   mat cv::Mat from which the shape must be inferred
         *
         * \return a Tensor object with the shape of \p mat
         */
        template <class T>
        Tensor<T> makeTensorHeader(const Mat& mat) {
            auto sizes = shape(mat);
            return Tensor<T>(std::begin(sizes), std::end(sizes));
        }

        template <class T> inline
        void copyMatToTensorImpl(const Mat& srcMat, const TensorSpan<T> destTensor, const Stream& stream) {
            CV_Assert(srcMat.total() >= destTensor.size());

            Mat temp = srcMat.isContinuous() ? srcMat : srcMat.clone();
            CV_Assert(temp.isContinuous());

            memcpy<T>(destTensor.get(), reinterpret_cast<T*>(temp.data), destTensor.size(), stream);
        }

        /** @brief copies data from a cv::Mat to TensorType
         *
         * \tparam  T   the type of the elements contained in TensorType object
         *
         * \param[in]   srcMat      source matrix
         * \param[out]  destTensor  destination tensor
         * \param       stream      CUDA stream to use for the memory transfer
         *
         * The memory copy starts from beginning \p srcMat. The number of elements copied is
         * equal to the number of elements in \p destTensor.
         *
         * Pre-conditions:
         * - \p srcMat must contain elements of type CV_32F
         * - the size of \p srcMat must be larger than or equal to the size of \p destTensor
         *
         * @note best performance when \p srcMat is continuous and page-locked
         * @note blocks calling thread if \p srcMat is not page-locked
         */
        template <class T>
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<T> destTensor, const Stream& stream);

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<half> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_32F, "");
            CV_Assert(srcMat.total() >= destTensor.size());

            Mat temp;
            srcMat.convertTo(temp, CV_16F);
            CV_Assert(temp.isContinuous());

            memcpy<half>(destTensor.get(), reinterpret_cast<half*>(temp.data), destTensor.size(), stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<float> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_32F, "");
            copyMatToTensorImpl(srcMat, destTensor, stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<int8_t> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_8S, "");
            copyMatToTensorImpl(srcMat, destTensor, stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<uint8_t> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_8U, "");
            copyMatToTensorImpl(srcMat, destTensor, stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<int32_t> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_32S, "");
            copyMatToTensorImpl(srcMat, destTensor, stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<int64_t> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_64S, "");
            copyMatToTensorImpl(srcMat, destTensor, stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<bool> destTensor, const Stream& stream) {
            CV_CheckTypeEQ(srcMat.type(), CV_Bool, "");
            copyMatToTensorImpl(srcMat, destTensor, stream);
        }

        /** @brief copies data from a TensorType to a cv::Mat
         *
         * \tparam  T   the type of the elements contained in TensorType object
         *
         * \param[in]   srcTensor   source tensor
         * \param[out]  destMat     destination matrix
         * \param       stream      CUDA stream to use for the memory transfer
         *
         * The entire memory block held by the \p srcTensor is copied to \p destMat.
         *
         * Pre-conditions:
         * - \p destMat must contain elements of type CV_32F
         * - the size of \p destMat must be larger than or equal to the size of \p srcTensor
         *
         * @note best performance when \p destMat is continuous and page-locked
         * @note blocks calling thread if \p destMat is not page-locked
         */
        template <class T>
        void copyTensorToMat(TensorView<T> srcTensor, Mat& destMat, const Stream& stream);

        template <> inline
        void copyTensorToMat(TensorView<half> srcTensor, Mat& destMat, const Stream& stream) {
            CV_CheckTypeEQ(destMat.type(), CV_32F, "Unsupported type");
            CV_Assert(destMat.total() >= srcTensor.size());

            Mat temp(shape(destMat), CV_16F);
            CV_Assert(temp.isContinuous());

            memcpy<half>(reinterpret_cast<half*>(temp.data), srcTensor.get(), srcTensor.size(), stream);

            temp.convertTo(destMat, CV_32F);
        }

        template <> inline
        void copyTensorToMat(TensorView<float> srcTensor, Mat& destMat, const Stream& stream) {
            CV_CheckTypeEQ(destMat.type(), CV_32F, "Unsupported type");
            CV_Assert(destMat.total() >= srcTensor.size());

            Mat temp = destMat.isContinuous() ? destMat : destMat.clone();
            CV_Assert(temp.isContinuous());

            memcpy<float>(reinterpret_cast<float*>(temp.data), srcTensor.get(), srcTensor.size(), stream);

            if (temp.data != destMat.data)
                temp.copyTo(destMat);
        }
    }} /* namespace cuda4dnn::csl */

    /** base class for CUDA operation nodes (for all supported targets) */
    class CUDABackendNode : public BackendNode {
    public:
        CUDABackendNode() : BackendNode(DNN_BACKEND_CUDA) { }
        virtual ~CUDABackendNode() { }

        virtual void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            cuda4dnn::csl::Workspace& workspace) = 0;

        virtual std::size_t get_workspace_memory_in_bytes() const noexcept { return 0; }
    };

    /** @brief utility function which creates CUDA node of correct type from `targetId`
     *
     * CUDA operation nodes take the type of data they operate on as a template parameter.
     * For example, ConcatOp<float> is an operation node which concats tensors of `float` type
     * into a tensor of `float` type.
     *
     * This utility function aids the creation of nodes of different types and eliminates the
     * need for CUDA target constants (`DNN_TARGET_XXX`) to appear in the operation code which
     * reduces coupling between modules.
     *
     * Example:
     * template <class T>
     * class ConcatOp : public CUDABackendNode;
     *
     * // returns a cv::Ptr to a ConcatOp<half> object
     * auto node = make_cuda_node<ConcatOp>(DNN_TARGET_CUDA_FP16, axis);
     *
     * // returns a cv::Ptr to a ConcatOp<float> object
     * auto node = make_cuda_node<ConcatOp>(DNN_TARGET_CUDA, axis);
     */
    template <template <class> class NodeType, class ...Args>
    cv::Ptr<BackendNode> make_cuda_node(int targetId, Args&& ...args) {
        switch (targetId)
        {
        case DNN_TARGET_CUDA_FP16:
            return Ptr<BackendNode>(new NodeType<half>(std::forward<Args>(args)...));
        case DNN_TARGET_CUDA:
            return Ptr<BackendNode>(new NodeType<float>(std::forward<Args>(args)...));
        default:
            CV_Assert(IS_DNN_CUDA_TARGET(targetId));
        }
        return Ptr<BackendNode>();
    }

    template <template <class> class NodeType, class ...Args>
    cv::Ptr<BackendNode> make_cuda_node_with_type(int targetId, int hostMatType, Args&& ...args) {
        CV_CheckType(hostMatType, hostMatType == CV_32F || hostMatType == CV_8S || hostMatType == CV_8U || hostMatType == CV_32S || hostMatType == CV_64S, "");

        if (hostMatType == CV_8S)
            return Ptr<BackendNode>(new NodeType<int8_t>(std::forward<Args>(args)...));
        else if (hostMatType == CV_8U)
            return Ptr<BackendNode>(new NodeType<uint8_t>(std::forward<Args>(args)...));
        else if (hostMatType == CV_32S)
            return Ptr<BackendNode>(new NodeType<int32_t>(std::forward<Args>(args)...));
        else if (hostMatType == CV_64S)
            return Ptr<BackendNode>(new NodeType<int64_t>(std::forward<Args>(args)...));
        else if (hostMatType == CV_32F)
        {
            if (targetId == DNN_TARGET_CUDA_FP16)
                return Ptr<BackendNode>(new NodeType<half>(std::forward<Args>(args)...));
            else if (targetId == DNN_TARGET_CUDA)
                return Ptr<BackendNode>(new NodeType<float>(std::forward<Args>(args)...));
        }
        CV_Error(Error::BadDepth, "Unsupported mat type");
        return Ptr<BackendNode>();
    }

    template <template <class, class> class NodeType, class T_INDEX, class ...Args>
    cv::Ptr<BackendNode> make_cuda_node_with_indices(int targetId, int hostMatType, Args&& ...args) {
        CV_CheckType(hostMatType, hostMatType == CV_32F || hostMatType == CV_8S || hostMatType == CV_8U || hostMatType == CV_32S || hostMatType == CV_64S, "");

        if (hostMatType == CV_8S)
            return Ptr<BackendNode>(new NodeType<int8_t, T_INDEX>(std::forward<Args>(args)...));
        else if (hostMatType == CV_8U)
            return Ptr<BackendNode>(new NodeType<uint8_t, T_INDEX>(std::forward<Args>(args)...));
        else if (hostMatType == CV_32S)
            return Ptr<BackendNode>(new NodeType<int32_t, T_INDEX>(std::forward<Args>(args)...));
        else if (hostMatType == CV_64S)
            return Ptr<BackendNode>(new NodeType<int64_t, T_INDEX>(std::forward<Args>(args)...));
        else if (hostMatType == CV_32F)
        {
            if (targetId == DNN_TARGET_CUDA_FP16)
                return Ptr<BackendNode>(new NodeType<half, T_INDEX>(std::forward<Args>(args)...));
            else if (targetId == DNN_TARGET_CUDA)
                return Ptr<BackendNode>(new NodeType<float, T_INDEX>(std::forward<Args>(args)...));
        }
        CV_Error(Error::BadDepth, "Unsupported mat type");
        return Ptr<BackendNode>();
    }

    template <template <class> class NodeType, class ...Args>
    cv::Ptr<BackendNode> make_cuda_node_bool(Args&& ...args) {
        return Ptr<BackendNode>(new NodeType<bool>(std::forward<Args>(args)...));
    }

    /* base class for all CUDA backend/target wrappers */
    class CUDABackendWrapper : public BackendWrapper {
    public:
        CUDABackendWrapper(int targetId) : BackendWrapper(DNN_BACKEND_CUDA, targetId) { }
        virtual ~CUDABackendWrapper() { }

        void copyToHost() override = 0;
        virtual void copyToHostInBackground() = 0;
        void setHostDirty() override = 0;

        virtual void copyToDevice() = 0;
        virtual void setDeviceDirty() = 0;

        virtual MatShape getShape() const noexcept = 0;
        virtual std::size_t getRank() const noexcept = 0;

        /** @note setting the stream updates the stream for all wrappers which use the same tensor */
        virtual void setStream(cuda4dnn::csl::Stream stream, cuda4dnn::csl::Stream h2d_stream) noexcept = 0;

        virtual void update(const MatShape& shape, std::size_t offset) = 0;
    };

    namespace cuda4dnn { namespace detail {

        template <class DEVICE_T, class HOST_T>
        void convert_D2H(const cv::Mat& mat, cuda4dnn::csl::View<DEVICE_T> view, cuda4dnn::csl::ManagedPtr<HOST_T>& device_temp, const cuda4dnn::csl::Stream& stream);

        template <> inline
        void convert_D2H<half, float>(const cv::Mat& mat, cuda4dnn::csl::View<half> view, cuda4dnn::csl::ManagedPtr<float>& device_temp, const cuda4dnn::csl::Stream& stream) {
            if (device_temp.size() < view.size())
                device_temp.reset(view.size());
            auto temp_span = cuda4dnn::csl::Span<float>(device_temp.get(), view.size());

            cuda4dnn::kernels::fp16_to_fp32(stream, temp_span, view);
            cuda4dnn::csl::memcpy<float>(reinterpret_cast<float*>(mat.data), temp_span.data(), view.size(), stream);
        }

        template <> inline
        void convert_D2H<float, float>(const cv::Mat& mat, cuda4dnn::csl::View<float> view, cuda4dnn::csl::ManagedPtr<float>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<float>(reinterpret_cast<float*>(mat.data), view.data(), view.size(), stream);
        }

        template <> inline
        void convert_D2H<int8_t, int8_t>(const cv::Mat& mat, cuda4dnn::csl::View<int8_t> view, cuda4dnn::csl::ManagedPtr<int8_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<int8_t>(reinterpret_cast<int8_t*>(mat.data), view.data(), view.size(), stream);
        }

        template <> inline
        void convert_D2H<uint8_t, uint8_t>(const cv::Mat& mat, cuda4dnn::csl::View<uint8_t> view, cuda4dnn::csl::ManagedPtr<uint8_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<uint8_t>(reinterpret_cast<uint8_t*>(mat.data), view.data(), view.size(), stream);
        }

        template <> inline
        void convert_D2H<int32_t, int32_t>(const cv::Mat& mat, cuda4dnn::csl::View<int32_t> view, cuda4dnn::csl::ManagedPtr<int32_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<int32_t>(reinterpret_cast<int32_t*>(mat.data), view.data(), view.size(), stream);
        }

        template <> inline
        void convert_D2H<int64_t, int64_t>(const cv::Mat& mat, cuda4dnn::csl::View<int64_t> view, cuda4dnn::csl::ManagedPtr<int64_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<int64_t>(reinterpret_cast<int64_t*>(mat.data), view.data(), view.size(), stream);
        }

        template <> inline
        void convert_D2H<bool, bool>(const cv::Mat& mat, cuda4dnn::csl::View<bool> view, cuda4dnn::csl::ManagedPtr<bool>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<bool>(reinterpret_cast<bool*>(mat.data), view.data(), view.size(), stream);
        }

        template <class DEVICE_T, class HOST_T>
        void convert_D2H_background(const cv::Mat& mat, cuda4dnn::csl::View<DEVICE_T> view, cuda4dnn::csl::ManagedPtr<HOST_T>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event);

        template <> inline
        void convert_D2H_background<half, float>(const cv::Mat& mat, cuda4dnn::csl::View<half> view, cuda4dnn::csl::ManagedPtr<float>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            if (device_temp.size() < view.size())
                device_temp.reset(view.size());
            auto temp_span = cuda4dnn::csl::Span<float>(device_temp.get(), view.size());

            /* The conversion kernel should can be executed in the background stream for better
             * performance. We do it in the inference stream to prevent an unexplained performance
             * regression on RTX 2080 Ti. Executing conversion kernel in the background stream causes
             * everything to slow down (even operations that appear before the background transfer).
             *
             * TODO: identify the cause and move conversion kernel to the background stream
             */
            cuda4dnn::kernels::fp16_to_fp32(stream, temp_span, view);

            d2h_event.record(stream); // mark position in inference stream
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event); // don't start transfer until data is available
            cuda4dnn::csl::memcpy<float>(reinterpret_cast<float*>(mat.data), temp_span.data(), view.size(), d2h_stream);
        }

        template <> inline
        void convert_D2H_background<float, float>(const cv::Mat& mat, cuda4dnn::csl::View<float> view, cuda4dnn::csl::ManagedPtr<float>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            d2h_event.record(stream);
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event);
            cuda4dnn::csl::memcpy<float>(reinterpret_cast<float*>(mat.data), view.data(), view.size(), d2h_stream);
        }

        template <> inline
        void convert_D2H_background<int8_t, int8_t>(const cv::Mat& mat, cuda4dnn::csl::View<int8_t> view, cuda4dnn::csl::ManagedPtr<int8_t>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            d2h_event.record(stream);
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event);
            cuda4dnn::csl::memcpy<int8_t>(reinterpret_cast<int8_t*>(mat.data), view.data(), view.size(), d2h_stream);
        }

        template <> inline
        void convert_D2H_background<uint8_t, uint8_t>(const cv::Mat& mat, cuda4dnn::csl::View<uint8_t> view, cuda4dnn::csl::ManagedPtr<uint8_t>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            d2h_event.record(stream);
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event);
            cuda4dnn::csl::memcpy<uint8_t>(reinterpret_cast<uint8_t*>(mat.data), view.data(), view.size(), d2h_stream);
        }

        template <> inline
        void convert_D2H_background<int32_t, int32_t>(const cv::Mat& mat, cuda4dnn::csl::View<int32_t> view, cuda4dnn::csl::ManagedPtr<int32_t>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            d2h_event.record(stream);
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event);
            cuda4dnn::csl::memcpy<int32_t>(reinterpret_cast<int32_t*>(mat.data), view.data(), view.size(), d2h_stream);
        }

        template <> inline
        void convert_D2H_background<int64_t, int64_t>(const cv::Mat& mat, cuda4dnn::csl::View<int64_t> view, cuda4dnn::csl::ManagedPtr<int64_t>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            d2h_event.record(stream);
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event);
            cuda4dnn::csl::memcpy<int64_t>(reinterpret_cast<int64_t*>(mat.data), view.data(), view.size(), d2h_stream);
        }

        template <> inline
        void convert_D2H_background<bool, bool>(const cv::Mat& mat, cuda4dnn::csl::View<bool> view, cuda4dnn::csl::ManagedPtr<bool>& device_temp, const cuda4dnn::csl::Stream& stream, const cuda4dnn::csl::Stream& d2h_stream, cuda4dnn::csl::Event& d2h_event) {
            d2h_event.record(stream);
            cuda4dnn::csl::StreamWaitOnEvent(d2h_stream, d2h_event);
            cuda4dnn::csl::memcpy<bool>(reinterpret_cast<bool*>(mat.data), view.data(), view.size(), d2h_stream);
        }

        template <class DEVICE_T, class HOST_T>
        void convert_H2D(cuda4dnn::csl::Span<DEVICE_T> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<HOST_T>& device_temp, const cuda4dnn::csl::Stream& stream);

        template <> inline
        void convert_H2D<half, float>(cuda4dnn::csl::Span<half> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<float>& device_temp, const cuda4dnn::csl::Stream& stream) {
            if (device_temp.size() < span.size())
                device_temp.reset(span.size());
            auto temp_span = cuda4dnn::csl::Span<float>(device_temp.get(), span.size());

            cuda4dnn::csl::memcpy<float>(temp_span.data(), reinterpret_cast<float*>(mat.data), span.size(), stream);
            cuda4dnn::kernels::fp32_to_fp16(stream, span, temp_span);
        }

        template <> inline
        void convert_H2D<float, float>(cuda4dnn::csl::Span<float> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<float>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<float>(span.data(), reinterpret_cast<float*>(mat.data), span.size(), stream);
        }

        template <> inline
        void convert_H2D<int8_t, int8_t>(cuda4dnn::csl::Span<int8_t> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<int8_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<int8_t>(span.data(), reinterpret_cast<int8_t*>(mat.data), span.size(), stream);
        }

        template <> inline
        void convert_H2D<uint8_t, uint8_t>(cuda4dnn::csl::Span<uint8_t> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<uint8_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<uint8_t>(span.data(), reinterpret_cast<uint8_t*>(mat.data), span.size(), stream);
        }

        template <> inline
        void convert_H2D<int32_t, int32_t>(cuda4dnn::csl::Span<int32_t> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<int32_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<int32_t>(span.data(), reinterpret_cast<int32_t*>(mat.data), span.size(), stream);
        }

        template <> inline
        void convert_H2D<int64_t, int64_t>(cuda4dnn::csl::Span<int64_t> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<int64_t>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<int64_t>(span.data(), reinterpret_cast<int64_t*>(mat.data), span.size(), stream);
        }

        template <> inline
        void convert_H2D<bool, bool>(cuda4dnn::csl::Span<bool> span, const cv::Mat& mat, cuda4dnn::csl::ManagedPtr<bool>& device_temp, const cuda4dnn::csl::Stream& stream) {
            cuda4dnn::csl::memcpy<bool>(span.data(), reinterpret_cast<bool*>(mat.data), span.size(), stream);
        }
    }} /* namespace cuda4dnn::detail */

    template <class DEVICE_T, class HOST_T, int TargetID>
    class GenericCUDABackendWrapper final : public CUDABackendWrapper {
    public:
        using value_type = DEVICE_T;
        using tensor_span_type = cuda4dnn::csl::TensorSpan<value_type>;
        using tensor_view_type = cuda4dnn::csl::TensorView<value_type>;

        /* Pre-conditions:
         * - there must be no other instance of `GenericCUDABackendWrapper` which wraps the host memory used by `m`
         * - the host memory must remain allocated throughout the lifetime of this object
         *
         * Post-conditions:
         * - the host memory used by \p m "may" be page-locked
         */
        GenericCUDABackendWrapper(Mat& m)
            : CUDABackendWrapper(TargetID)
        {
            shape = cv::dnn::shape(m);
            hostMatDepth = m.depth();
            offset = 0;

            shared_block = std::make_shared<shared_block_type>();
            shared_block->host_dirty = true;
            shared_block->device_dirty = false;

            shared_block->host = m;

            try {
                shared_block->memGuard = cuda4dnn::csl::MemoryLockGuard(m.data, m.total() * m.elemSize());
            } catch (...) {
                /* a common reason for failure is that the host system (for example, a Jetson device) does not support it */
                /* we ignore the failure as this is just an optimization and not a requirement */
            }

            shared_block->device = cuda4dnn::csl::ManagedPtr<DEVICE_T>(m.total());
        }

        GenericCUDABackendWrapper(const Ptr<BackendWrapper>& base_, const MatShape& shape_)
            : CUDABackendWrapper(TargetID)
        {
            const Ptr<GenericCUDABackendWrapper> base = base_.dynamicCast<GenericCUDABackendWrapper>();
            CV_Assert(base);

            shape = shape_;
            hostMatDepth = base_->getHostMatDepth();
            offset = 0;
            shared_block = base->shared_block;

            auto numel = total(shape_);
            if (numel > shared_block->device.size())
            {
                /* if the host memory was already page-locked, release it and register again with the new size */
                shared_block->memGuard = cuda4dnn::csl::MemoryLockGuard();
                try {
                    CV_Assert(shared_block->host.type() == CV_32F);
                    shared_block->memGuard = cuda4dnn::csl::MemoryLockGuard(shared_block->host.data, numel * sizeof(float));
                } catch (...) {
                    /* a common reason for failure is that the host system (for example, a Jetson device) does not support it */
                    /* we ignore the failure as this is just an optimization and not a requirement */
                }
                shared_block->device.reset(numel);
            }
        }

        static Ptr<BackendWrapper> create(Mat& m) {
            return Ptr<BackendWrapper>(new GenericCUDABackendWrapper(m));
        }

        static Ptr<BackendWrapper> create(const Ptr<BackendWrapper>& base, const MatShape& shape) {
            return Ptr<BackendWrapper>(new GenericCUDABackendWrapper(base, shape));
        }

        void copyToHost() override {
            if (shared_block->device_dirty) {
                CV_Assert(offset == 0); /* we cannot track each piece of the memory separately */

                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                /* If the wrapper is being reused, the device tensor might be larger in size than the wrapper.
                 * Using the device tensor does not give incorrect code but leads to unused region of memory being copied.
                 *
                 * We use a view to ensure that only the required region of memory is copied.
                 */
                auto view = tensor_view_type(shared_block->device.get(), std::begin(shape), std::end(shape));

                auto& mat = shared_block->host;
                CV_Assert(mat.isContinuous());

                cuda4dnn::detail::convert_D2H<DEVICE_T, HOST_T>(mat, view, shared_block->device_temp, shared_block->stream);
                shared_block->stream.synchronize();
            } else if(shared_block->d2h_event && shared_block->d2h_event.busy()) {
                /* wait for the background copy to finish */
                shared_block->d2h_event.synchronize();
            }
        }

        void copyToHostInBackground() override {
            CV_Assert(shared_block->d2h_stream);
            if (shared_block->device_dirty) {
                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                auto view = tensor_view_type(shared_block->device.get(), std::begin(shape), std::end(shape));

                auto& mat = shared_block->host;
                CV_Assert(mat.isContinuous());

                if (!shared_block->d2h_event)
                    shared_block->d2h_event = cuda4dnn::csl::Event(true);
                cuda4dnn::detail::convert_D2H_background<DEVICE_T, HOST_T>(mat, view, shared_block->device_temp, shared_block->stream, shared_block->d2h_stream, shared_block->d2h_event);
                shared_block->d2h_event.record(shared_block->d2h_stream); // record position so that we can check status later
            }
        }

        void setHostDirty() override {
            shared_block->device_dirty = false;
            shared_block->host_dirty = true;
        }

        void copyToDevice() override {
            if (shared_block->host_dirty) {
                CV_Assert(offset == 0); /* we cannot track each piece of the memory separately */

                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                auto span = tensor_span_type(shared_block->device.get(), std::begin(shape), std::end(shape));

                auto& mat = shared_block->host;
                CV_Assert(mat.isContinuous());

                cuda4dnn::detail::convert_H2D<DEVICE_T, HOST_T>(span, mat, shared_block->device_temp, shared_block->stream);
            }
        }

        void setDeviceDirty() override {
            shared_block->device_dirty = true;
            shared_block->host_dirty = false;
        }

        MatShape getShape() const noexcept override { return shape; }

        std::size_t getRank() const noexcept override { return shape.size(); }

        void setStream(cuda4dnn::csl::Stream stream, cuda4dnn::csl::Stream d2h_stream) noexcept override {
            shared_block->stream = std::move(stream);
            shared_block->d2h_stream = std::move(d2h_stream);
        }

        void update(const MatShape& shape_, std::size_t offset_) override {
            std::size_t total = shape_.total();
            if (offset_ + total > shared_block->device.size()) {
                CV_Error(Error::BadOffset, "shape and offset provided can potentially leads to OOB access");
            }
            shape = shape_;
            offset = offset_;
        }

        cv::Mat getMutableHostMat() noexcept {
            CV_Assert(offset == 0); /* we cannot track each piece of the memory separately */
            copyToHost();
            setHostDirty();
            return shared_block->host;
        }

        const cv::Mat getImmutableHostMat() const noexcept {
            CV_Assert(offset == 0); /* we cannot track each piece of the memory separately */
            copyToHost();
            return shared_block->host;
        }

        /* Optimization Note: use getSpan() and getView() judiciously
         *
         * getSpan() is meant to be used when the memory is going to be modified
         * getView() is meant to be used when the memory is only going to be read
         *
         * getSpan() marks the device memory as dirty but getView() does not
         *
         * getView() implicitly performs host to device memory transfer if required
         * getSpan() does not perform any synchronization (use copyToDevice if sync. is required)
         */
        tensor_span_type getSpan() noexcept {
            setDeviceDirty();
            return tensor_span_type(shared_block->device.get() + offset, std::begin(shape), std::end(shape));
        }

        tensor_view_type getView() noexcept {
            copyToDevice();
            return tensor_view_type(shared_block->device.get() + offset, std::begin(shape), std::end(shape));
        }

    private:
        /* The same tensor memory can be reused by different layers whenever possible.
         * Hence, it is possible for different backend wrappers to point to the same memory.
         * However, it may use only a part of that memory and have a different shape.
         *
         * We store the common information such as device tensor and its corresponding host memory in
         * a shared block. The shared block is shared by all backend wrappers which use the same memory.
         * The shape, which can be different for different wrappers, is stored as a member object.
         */

        MatShape shape;
        std::size_t offset;

        struct shared_block_type {
            bool host_dirty;
            bool device_dirty;

            cv::Mat host;
            cuda4dnn::csl::MemoryLockGuard memGuard; /* keeps host memory page-locked if possible */

            cuda4dnn::csl::ManagedPtr<DEVICE_T> device;
            cuda4dnn::csl::ManagedPtr<HOST_T> device_temp; /* use for conversions */
            cuda4dnn::csl::Stream stream;

            cuda4dnn::csl::Event d2h_event;
            cuda4dnn::csl::Stream d2h_stream;
        };

        std::shared_ptr<shared_block_type> shared_block;
    };

    using CUDABackendWrapperFP16 = GenericCUDABackendWrapper<half, float, DNN_TARGET_CUDA_FP16>;
    using CUDABackendWrapperFP32 = GenericCUDABackendWrapper<float, float, DNN_TARGET_CUDA>;
    using CUDABackendWrapperINT8 = GenericCUDABackendWrapper<int8_t, int8_t, DNN_TARGET_CUDA>;
    using CUDABackendWrapperUINT8 = GenericCUDABackendWrapper<uint8_t, uint8_t, DNN_TARGET_CUDA>;
    using CUDABackendWrapperINT32 = GenericCUDABackendWrapper<int32_t, int32_t, DNN_TARGET_CUDA>;
    using CUDABackendWrapperINT64 = GenericCUDABackendWrapper<int64_t, int64_t, DNN_TARGET_CUDA>;
    using CUDABackendWrapperBOOL = GenericCUDABackendWrapper<bool, bool, DNN_TARGET_CUDA>;

    template <class T> struct GetCUDABackendWrapperType_ { };
    template <> struct GetCUDABackendWrapperType_<half> { typedef CUDABackendWrapperFP16 type; };
    template <> struct GetCUDABackendWrapperType_<float> { typedef CUDABackendWrapperFP32 type; };
    template <> struct GetCUDABackendWrapperType_<int8_t> { typedef CUDABackendWrapperINT8 type; };
    template <> struct GetCUDABackendWrapperType_<uint8_t> { typedef CUDABackendWrapperUINT8 type; };
    template <> struct GetCUDABackendWrapperType_<int32_t> { typedef CUDABackendWrapperINT32 type; };
    template <> struct GetCUDABackendWrapperType_<int64_t> { typedef CUDABackendWrapperINT64 type; };
    template <> struct GetCUDABackendWrapperType_<bool> { typedef CUDABackendWrapperBOOL type; };

    template <class T>
    using GetCUDABackendWrapperType = typename GetCUDABackendWrapperType_<T>::type;

#endif
}} /* namespace cv::dnn */

#endif /* OPENCV_DNN_SRC_OP_CUDA_HPP */
