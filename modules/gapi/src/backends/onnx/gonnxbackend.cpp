// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"
#include "backends/onnx/gonnxbackend.hpp"

#ifdef HAVE_ONNX

#include "backends/onnx/dml_ep.hpp"
#include "backends/onnx/coreml_ep.hpp"

#include <ade/util/algorithm.hpp> // any_of
#include <ade/util/zip_range.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/own/convert.hpp>
#include <opencv2/gapi/gframe.hpp>
#include <codecvt> // wstring_convert

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!
#include "logger.hpp"

namespace {
struct ONNXCallContext;
}

namespace cv {
namespace gimpl {
namespace onnx {

enum TensorPosition : int {
    INPUT,
    OUTPUT
};

static std::string pdims(const std::vector<int64_t> &dims) {
    std::stringstream ss;
    auto it = dims.begin();
    ss << *it++;
    for (; it != dims.end(); ++it) {
        ss << '/' << *it;
    }
    return ss.str();
}

struct TensorInfo {
    TensorInfo() = default;

    explicit TensorInfo(const Ort::ConstTensorTypeAndShapeInfo &info)
        : dims(info.GetShape())
        , type(info.GetElementType())
        , is_dynamic(ade::util::find(dims, -1) != dims.end()) {

        // Double-check if the tensor is really dynamic
        // Allow N to be -1
        if (is_dynamic
            && dims[0] == -1
            && dims.size() > 1
            && std::find(dims.begin() + 1, dims.end(), -1) == dims.end()) {

            GAPI_LOG_WARNING(NULL, "Promoting N=-1 to N=1 for tensor " << pdims(dims));
            dims[0] = 1;
            is_dynamic = false;
        }

        if (!is_dynamic) {
            size = std::accumulate(dims.begin(),
                                   dims.end(),
                                   static_cast<int64_t>(1),
                                   std::multiplies<int64_t>());
        }
        // Heuristic: check if the tensor is grayscale input
        if (dims.size() == 4u
            && dims[0]  == 1
            && dims[1]  == 1
            && dims[2]   > 1
            && dims[3]   > 1) {
            is_grayscale = true;
        }
    }

    std::string name;
    std::vector<int64_t> dims;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    int64_t size = -1;

    bool normalize = true;

    bool is_dynamic = false;
    bool is_grayscale = false;

    struct MeanStdev {
        cv::Scalar mean;
        cv::Scalar stdev;
    };
    cv::util::optional<MeanStdev> mstd;
};

using Views = std::vector<std::unique_ptr<cv::MediaFrame::View>>;

class ONNXCompiled {
    // ONNX Resources
    // NOTE: Env must live with the session, otherwise segfaults.
    Ort::Env this_env{nullptr};
    Ort::Session this_session{nullptr};
    Ort::MemoryInfo this_memory_info{nullptr};

    std::vector<TensorInfo> in_tensor_info;
    std::vector<TensorInfo> out_tensor_info;
    bool is_dynamic = false;
    bool is_postproc = false;

    // G-API <Net> description
    gapi::onnx::detail::ParamDesc params;

    // Input/output tensor information
    std::vector<TensorInfo> getTensorInfo(TensorPosition pos);

    // Run-time data structures
    std::vector<cv::Mat> in_data;
    std::vector<cv::Mat> out_data;

    void Run(const std::vector<cv::Mat>& ins,
             std::vector<cv::Mat>& outs);

    std::vector<std::string> in_names_without_const;
public:
    explicit ONNXCompiled(const gapi::onnx::detail::ParamDesc &pp);

    // Extract the information about output layer #i
    cv::GMatDesc outMeta(int i) const;

    // Assign input/output info
    std::size_t numInputs() const { return params.num_in; }
    std::size_t numOutputs() const { return params.num_out; }
    void setInput(int i, const cv::Mat &m);
    void setOutput(int idx, cv::Mat &m);
    cv::Mat allocOutput(int i) const;
    // Gets exMat from input
    void extractMat(ONNXCallContext &ctx, const size_t in_idx, Views &views);
    // Extracted cv::Mat from input cv::Mat/cv::MediaFrame
    cv::Mat exMat;
    // Run with the assigned inputs/outputs
    void run();
};

static void addCUDAExecutionProvider(Ort::SessionOptions *session_options,
                                     const cv::gapi::onnx::ep::CUDA &cuda_ep) {
     OrtCUDAProviderOptions options{};
     options.device_id = cuda_ep.device_id;

     try {
        session_options->AppendExecutionProvider_CUDA(options);
     } catch (const std::exception &e) {
         std::stringstream ss;
         ss << "ONNX Backend: Failed to enable CUDA"
            << " Execution Provider: " << e.what();
         cv::util::throw_error(std::runtime_error(ss.str()));
     }
}

static void addTensorRTExecutionProvider(Ort::SessionOptions *session_options,
                                         const cv::gapi::onnx::ep::TensorRT &trt_ep) {
     OrtTensorRTProviderOptions options{};
     options.device_id = trt_ep.device_id;

     try {
        session_options->AppendExecutionProvider_TensorRT(options);
     } catch (const std::exception &e) {
         std::stringstream ss;
         ss << "ONNX Backend: Failed to enable TensorRT"
            << " Execution Provider: " << e.what();
         cv::util::throw_error(std::runtime_error(ss.str()));
     }
}

static void addOpenVINOExecutionProvider(Ort::SessionOptions *session_options,
                                         const cv::gapi::onnx::ep::OpenVINO &ov_ep) {
     std::unordered_map<std::string, std::string> options;

     try {
        // If the OpenVINO Execution Provider object was initialized with a parameters map,
        // those parameters are used directly.
        // Otherwise, the function constructs the options map from the individual member
        // variables of the OpenVINO object.
        if (ov_ep.params_map.empty()) {
            options = {
                {"device_type", ov_ep.device_type},
                {"cache_dir", ov_ep.cache_dir},
                {"num_of_threads", ov_ep.num_of_threads > 0 ? std::to_string(ov_ep.num_of_threads) : ""},
                {"enable_opencl_throttling", ov_ep.enable_opencl_throttling ? "True" : "False"},
                {"enable_dynamic_shapes", ov_ep.enable_dynamic_shapes ? "True" : "False"},
            };
        } else {
            options.insert(ov_ep.params_map.begin(), ov_ep.params_map.end());
        }
        //  AppendExecutionProvider function expects a const std::unordered_map as its second argument
        session_options->AppendExecutionProvider("OpenVINO", options);
     } catch (const std::exception &e) {
         std::stringstream ss;
         ss << "ONNX Backend: Failed to enable OpenVINO"
            << " Execution Provider: " << e.what();
         cv::util::throw_error(std::runtime_error(ss.str()));
     }
}

static void addExecutionProvider(Ort::SessionOptions          *session_options,
                                 const cv::gapi::onnx::ep::EP &execution_provider) {
    namespace ep = cv::gapi::onnx::ep;
    switch (execution_provider.index()) {
        case ep::EP::index_of<ep::OpenVINO>(): {
             GAPI_LOG_INFO(NULL, "OpenVINO Execution Provider is added.");
             const auto &ov_ep = cv::util::get<ep::OpenVINO>(execution_provider);
             addOpenVINOExecutionProvider(session_options, ov_ep);
             break;
        }
        case ep::EP::index_of<ep::DirectML>(): {
            GAPI_LOG_INFO(NULL, "DirectML Execution Provider is added.");
            const auto &dml_ep = cv::util::get<ep::DirectML>(execution_provider);
            addDMLExecutionProvider(session_options, dml_ep);
            break;
        }
        case ep::EP::index_of<ep::CoreML>(): {
            GAPI_LOG_INFO(NULL, "CoreML Execution Provider is added.");
            const auto &coreml_ep = cv::util::get<ep::CoreML>(execution_provider);
            addCoreMLExecutionProvider(session_options, coreml_ep);
            break;
        }
        case ep::EP::index_of<ep::CUDA>(): {
            GAPI_LOG_INFO(NULL, "CUDA Execution Provider is added.");
            const auto &cuda_ep = cv::util::get<ep::CUDA>(execution_provider);
            addCUDAExecutionProvider(session_options, cuda_ep);
            break;
        }
        case ep::EP::index_of<ep::TensorRT>(): {
            GAPI_LOG_INFO(NULL, "TensorRT Execution Provider is added.");
            const auto &trt_ep = cv::util::get<ep::TensorRT>(execution_provider);
            addTensorRTExecutionProvider(session_options, trt_ep);
            break;
        }
        default:
            GAPI_LOG_INFO(NULL, "CPU Execution Provider is added.");
            break;
    }
}

} // namespace onnx
} // namespace gimpl
} // namespace cv

namespace {

inline std::vector<const char*> getCharNames(const std::vector<std::string>& names) {
    std::vector<const char*> out_vec;
    for (const auto& el : names) {
        out_vec.push_back(el.data());
    }
    return out_vec;
}

inline int getIdxByName(const std::vector<cv::gimpl::onnx::TensorInfo>& info, const std::string& name) {
    // FIXME: Cache the ordering
    const auto it = ade::util::find_if(info, [&](const cv::gimpl::onnx::TensorInfo &i) {
            return i.name == name;
        });
    GAPI_Assert(it != info.end());
    return std::distance(info.begin(), it);
}

inline int toCV(ONNXTensorElementDataType prec) {
    switch (prec) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return CV_8U;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return CV_32F;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return CV_32S;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return CV_32S;
    default: GAPI_Error("ONNX. Unsupported data type");
    }
    return -1;
}

inline std::vector<int> toCV(const std::vector<int64_t> &vsz) {
    std::vector<int> result;
    result.reserve(vsz.size());
    for (auto sz : vsz) {
        result.push_back(ade::util::checked_cast<int>(sz));
    }
    return result;
}

inline void copyFromONNX(Ort::Value &v, cv::Mat& mat) {
    const auto info = v.GetTensorTypeAndShapeInfo();
    const auto prec = info.GetElementType();
    const auto shape = toCV(info.GetShape());
    mat.create(shape, toCV(prec));
    switch (prec) {
#define HANDLE(E,T)                                          \
        case E: std::copy_n(v.GetTensorMutableData<T>(),     \
                            mat.total(),                     \
                            reinterpret_cast<T*>(mat.data)); \
            break;
        HANDLE(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t);
        HANDLE(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float);
        HANDLE(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int);
#undef HANDLE
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            GAPI_LOG_WARNING(NULL, "INT64 isn't supported for cv::Mat. Conversion to INT32 is used.");
            cv::gimpl::convertInt64ToInt32(v.GetTensorMutableData<int64_t>(),
                                           reinterpret_cast<int*>(mat.data),
                                           mat.total());
            break;
        }
    default: GAPI_Error("ONNX. Unsupported data type");
    }
}

inline std::vector<int64_t> toORT(const cv::MatSize &sz) {
    return cv::to_own<int64_t>(sz);
}

inline void preprocess(const cv::Mat& src,
                       const cv::gimpl::onnx::TensorInfo& ti,
                             cv::Mat& dst) {
    // CNN input type
    const auto type = toCV(ti.type);
    if (src.depth() != CV_8U) {
        // Just pass the tensor as-is.
        // No layout or dimension transformations done here!
        // TODO: This needs to be aligned across all NN backends.
        const auto tensor_dims = toORT(src.size);
        if (tensor_dims.size() == ti.dims.size()) {
            for (size_t i = 0; i < ti.dims.size(); ++i) {
                GAPI_Assert((ti.dims[i] == -1 || ti.dims[i] == tensor_dims[i]) &&
                            "Non-U8 tensor dimensions should match with all non-dynamic NN input dimensions");
            }
        } else {
            GAPI_Error("Non-U8 tensor size should match with NN input");
        }

        dst = src;
    } else {
        // 8U input: full preprocessing path
        GAPI_Assert(src.depth() == CV_8U && "Only 8U data type is supported for preproc");
        GAPI_Assert((ti.dims.size() == 4u || ti.dims.size() == 3u)
                    && "Only NCHW/NHWC/CHW/HWC layouts are supported for preproc");

        const bool with_batch = ti.dims.size() == 4u ? true : false;
        const int shift = with_batch ? 0 : 1;

        GAPI_Assert((type == CV_8U || type == CV_32F)
                    && "Only 8U and 32F model input is supported for 8U input data");

        // Assess the expected input layout
        const bool is_hwc = [&](int ch) {
            if (ti.is_grayscale)               return false; // 1,1,h,w
            else if (ti.dims[3 - shift] == ch) return true;  // ?,_,_,c
            else if (ti.dims[1 - shift] == ch) return false; // ?,c,_,_
            else cv::util::throw_error(std::logic_error("Couldn't identify input tensor layout"));
        } (src.channels());

        int new_c = src.channels();
        cv::Mat csc;
        if (ti.is_grayscale && new_c == 3) {
            cv::cvtColor(src, csc, cv::COLOR_BGR2GRAY);
            new_c = 1;
        } else {
            csc = src;
        }

        // NHWC vs NCHW
        int new_h = -1, new_w = -1;
        if (ti.is_dynamic) {
            // reuse h & w from the input image
            new_h = src.rows;
            new_w = src.cols;
        } else {
            // take h & w from the ONNX tensor info
            new_h = ti.dims[(is_hwc ? 1 : 2) - shift];
            new_w = ti.dims[(is_hwc ? 2 : 3) - shift];
        }
        GAPI_Assert(new_h != -1 && new_w != -1);

        cv::Mat rsz, pp;
        cv::resize(csc, rsz, cv::Size(new_w, new_h));
        if (src.depth() == CV_8U && type == CV_32F) {
            rsz.convertTo(pp, type, ti.normalize ? 1.f / 255 : 1.f);

            if (ti.mstd.has_value()) {
                pp -= ti.mstd->mean;
                pp /= ti.mstd->stdev;
            }
        } else {
            pp = rsz;
        }

        if (!is_hwc && new_c > 1) {
            // Convert to CHW
            dst.create(cv::Size(new_w, new_h * new_c), type);
            std::vector<cv::Mat> planes(new_c);
            for (int ch = 0; ch < new_c; ++ch) {
                planes[ch] = dst.rowRange(ch * new_h, (ch + 1) * new_h);
            }
            cv::split(pp, planes);
        } else {
            // Keep HWC
            dst = pp;
        }

        // Ensure dst is a tensor shape (not a 2D image)
        if (ti.is_dynamic) {
            // Reshape to input dimensions
            const std::vector<int> out_dims = is_hwc
                ? with_batch
                    ? std::vector<int>{1, new_h, new_w, new_c}
                    : std::vector<int>{new_h, new_w, new_c}
                : with_batch
                    ? std::vector<int>{1, new_c, new_h, new_w}
                    : std::vector<int>{new_c, new_h, new_w};
            dst = dst.reshape(1, out_dims);
        } else {
            // Reshape to ONNX dimensions (no -1s there!)
            dst = dst.reshape(1, toCV(ti.dims));
        }
    }
}

void preprocess(const cv::MediaFrame::View& view,
                const cv::GFrameDesc& desc,
                      cv::Mat& dst) {
    // This overload constructs cv::Mat from cv::MediaFrame
    switch (desc.fmt) {
        case cv::MediaFormat::BGR: {
            dst = cv::Mat(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            break;
        }
        case cv::MediaFormat::NV12: {
            const auto y_plane  = cv::Mat(desc.size, CV_8UC1, view.ptr[0], view.stride[0]);
            const auto uv_plane = cv::Mat(desc.size / 2, CV_8UC2, view.ptr[1], view.stride[1]);
            cvtColorTwoPlane(y_plane, uv_plane, dst, cv::COLOR_YUV2BGR_NV12);
            break;
        }
        default:
            GAPI_Error("Unsupported media format for ONNX backend");
    }
}

template <typename T>
inline Ort::Value createTensor(const cv::Mat& data) {
    auto ort_dims = toORT(data.size);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value tensor = Ort::Value::CreateTensor<T>(allocator, ort_dims.data(), ort_dims.size());
    T* tensor_data = tensor.GetTensorMutableData<T>();
    std::copy_n(data.ptr<T>(), data.total(), tensor_data);
    return tensor;
}

inline Ort::Value createTensor(const cv::gimpl::onnx::TensorInfo& tensor_params,
                               const cv::Mat& data) {
    GAPI_Assert(data.isContinuous ());
    switch (tensor_params.type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return createTensor<uint8_t>(data);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return createTensor<float>(data);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return createTensor<int32_t>(data);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:{
        // cv::Mat does not support int64 data directly.
        // Following steps are applied to create an ONNX tensor from cv::Mat data:
        // - First create a new ONNX tensor 'i64_tensor' with data type int64_t using the default allocator
        // - Next retrieve a pointer to the mutable data buffer of 'i64_tensor'
        // - Convert the data from int32 (see toCV function) to int64 and deep copy it into 'i64_tensor'
        auto ort_dims = toORT(data.size);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::Value i64_tensor = Ort::Value::CreateTensor<int64_t>(allocator,
                                                                  ort_dims.data(),
                                                                  ort_dims.size());
        int64_t* tensor_data = i64_tensor.GetTensorMutableData<int64_t>();

        cv::gimpl::convertInt32ToInt64(data.ptr<int>(),
                                       tensor_data,
                                       data.total());
        return i64_tensor;
    }
    default:
        GAPI_Error("ONNX. Unsupported data type");
    }
    return Ort::Value{nullptr};
}

struct ONNXUnit {
    static const char *name() { return "ONNXModelConfig"; }

    std::shared_ptr<cv::gimpl::onnx::ONNXCompiled> oc;

    explicit ONNXUnit(const cv::gapi::onnx::detail::ParamDesc &pp)
        : oc(new cv::gimpl::onnx::ONNXCompiled(pp)) {
    }
};

struct ONNXCallContext {
    // Input parameters passed to an inference operation.
    std::vector<cv::GArg> args;
    cv::GShapes in_shapes;
    //FIXME: avoid conversion of arguments from internal representation to OpenCV one on each call
    //to OCV kernel. (This can be achieved by a two single time conversions in GCPUExecutable::run,
    //once on enter for input and output arguments, and once before return for output arguments only
    //FIXME: check if the above applies to this backend (taken from CPU)
    std::unordered_map<std::size_t, cv::GRunArgP> results;

    // Generic accessor API
    template<typename T>
    const T& inArg(std::size_t input) { return args.at(input).get<T>(); }

    // Syntax sugar
    const cv::Mat&   inMat(std::size_t input) {
        return inArg<cv::Mat>(input);
    }

    const cv::MediaFrame& inFrame(std::size_t input) {
        return inArg<cv::MediaFrame>(input);
    }

    cv::Mat&         outMatR(std::size_t output) {
        return *cv::util::get<cv::Mat*>(results.at(output));
    }

    template<typename T> std::vector<T>& outVecR(std::size_t output) { // FIXME: the same issue
        return outVecRef(output).wref<T>();
    }
    cv::detail::VectorRef& outVecRef(std::size_t output) {
        return cv::util::get<cv::detail::VectorRef>(results.at(output));
    }
};

struct ONNXCallable {
    static const char *name() { return "ONNXRequestCallable"; }
    using Run = std::function<void(const ONNXUnit &, ONNXCallContext &)>;
    Run run;
};

struct KImpl {
    cv::gimpl::CustomMetaFunction::CM customMetaFunc;
    ONNXCallable::Run run;
};

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GONNXModel = ade::TypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    , ONNXUnit
    , ONNXCallable
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGONNXModel = ade::ConstTypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    , ONNXUnit
    , ONNXCallable
    >;
} // anonymous namespace

// GCPUExcecutable implementation //////////////////////////////////////////////
cv::gimpl::onnx::GONNXExecutable::GONNXExecutable(const ade::Graph &g,
                                                  const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g) {
    // FIXME: Currently this backend is capable to run a single inference node only.
    // Need to extend our island fusion with merge/not-to-merge decision making parametrization
    GConstGONNXModel iem(g);

    for (auto &nh : nodes) {
        switch (m_gm.metadata(nh).get<NodeType>().t) {
        case NodeType::OP:
            if (this_nh == nullptr) {
                this_nh = nh;
            }
            else {
                util::throw_error(std::logic_error("Multi-node inference is not supported!"));
            }
            break;

        case NodeType::DATA: {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST_VAL) {
                util::throw_error(std::logic_error("No const data supported in backend!"));
            }
            if (desc.storage == Data::Storage::INTERNAL) {
                util::throw_error(std::logic_error("No internal data supported in backend!"));
            }
            break;
        }
        default: util::throw_error(std::logic_error("Unsupported NodeType"));
        }
    }
}

// FIXME: Document what it does
cv::GArg cv::gimpl::onnx::GONNXExecutable::packArg(const cv::GArg &arg) {
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY
                && arg.kind != cv::detail::ArgKind::GOPAQUE
                && arg.kind != cv::detail::ArgKind::GFRAME);

    if (arg.kind != cv::detail::ArgKind::GOBJREF) {
        util::throw_error(std::logic_error("Inference supports G-types ONLY!"));
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:    return GArg(m_res.slot<cv::Mat>()[ref.id]);

    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));

    // Note: .at() is intentional for GOpaque as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GOPAQUE:  return GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));

    case GShape::GFRAME:   return GArg(m_res.slot<cv::MediaFrame>().at(ref.id));

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::onnx::GONNXExecutable::run(std::vector<InObj>  &&input_objs,
                                           std::vector<OutObj> &&output_objs) {
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second);

    // FIXME: Running just a single node now.
    // Not sure if need to support many of them, though
    // FIXME: Make this island-unmergeable?
    const auto &op = m_gm.metadata(this_nh).get<Op>();

    // Initialize kernel's execution context:
    // - Input parameters
    ONNXCallContext context;
    context.args.reserve(op.args.size());
    using namespace std::placeholders;
    ade::util::transform(op.args,
                         std::back_inserter(context.args),
                         std::bind(&GONNXExecutable::packArg, this, _1));

    // NB: Need to store inputs shape to recognize GFrame/GMat
    context.in_shapes.reserve(op.args.size());
    ade::util::transform(op.args,
                         std::back_inserter(context.in_shapes),
                         [](const cv::GArg& arg) {
                             return arg.get<cv::gimpl::RcDesc>().shape;
                         });

    // - Output parameters.
    for (const auto &out_it : ade::util::indexed(op.outs)) {
        // FIXME: Can the same GArg type resolution mechanism be reused here?
        const auto out_port  = ade::util::index(out_it);
        const auto out_desc  = ade::util::value(out_it);
        context.results[out_port] = magazine::getObjPtr(m_res, out_desc);
    }

    // And now trigger the execution
    GConstGONNXModel giem(m_g);
    const auto &uu = giem.metadata(this_nh).get<ONNXUnit>();
    const auto &kk = giem.metadata(this_nh).get<ONNXCallable>();
    kk.run(uu, context);

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second);
}

namespace cv {
namespace gimpl {
namespace onnx {

static GraphOptimizationLevel convertToGraphOptimizationLevel(const int opt_level) {
    switch (opt_level) {
    case ORT_DISABLE_ALL:
        return ORT_DISABLE_ALL;
    case ORT_ENABLE_BASIC:
        return ORT_ENABLE_BASIC;
    case ORT_ENABLE_EXTENDED:
        return ORT_ENABLE_EXTENDED;
    case ORT_ENABLE_ALL:
        return ORT_ENABLE_ALL;
    default:
        if (opt_level > ORT_ENABLE_ALL) {  // relax constraint
            return ORT_ENABLE_ALL;
        }
        else {
            cv::util::throw_error(std::invalid_argument("Invalid argument opt_level = " + std::to_string(opt_level)));
        }
    }
}

ONNXCompiled::ONNXCompiled(const gapi::onnx::detail::ParamDesc &pp)
    : params(pp) {
    // Validate input parameters before allocating any resources
    if (params.num_in > 1u && params.num_in != params.input_names.size()) {
        cv::util::throw_error(std::logic_error("Please specify input layer names for "
                                               + params.model_path));
    }
    if (params.num_out > 1u && params.num_out != params.output_names.size()) {
        cv::util::throw_error(std::logic_error("Please specify output layer names for "
                                               + params.model_path));
    }
    // Create and initialize the ONNX session
    Ort::SessionOptions session_options;
    GAPI_LOG_INFO(NULL, "Adding Execution Providers for \"" << pp.model_path << "\"");
    for (const auto &ep : pp.execution_providers) {
        cv::gimpl::onnx::addExecutionProvider(&session_options, ep);
    }

    for (const auto &option : pp.session_options) {
        session_options.AddConfigEntry(option.first.c_str(), option.second.c_str());
    }

    if (pp.disable_mem_pattern) {
        session_options.DisableMemPattern();
    }

    if (pp.opt_level.has_value()) {
        session_options.SetGraphOptimizationLevel(convertToGraphOptimizationLevel(pp.opt_level.value()));
    }
    this_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
#ifndef _WIN32
    this_session = Ort::Session(this_env, params.model_path.data(), session_options);
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::wstring w_model_path = converter.from_bytes(params.model_path.data());
    this_session = Ort::Session(this_env, w_model_path.data(), session_options);
#endif
    this_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    in_tensor_info = getTensorInfo(INPUT);
    out_tensor_info = getTensorInfo(OUTPUT);

    const auto is_dyn = [](const TensorInfo &ti) {
        return ti.is_dynamic;
    };
    is_dynamic = ade::util::any_of(in_tensor_info, is_dyn)
              || ade::util::any_of(out_tensor_info, is_dyn);
    if (is_dynamic && !params.custom_post_proc) {
        util::throw_error(std::logic_error("This network has dynamic shapes. "
                                           "Please provide a custom post-processing function "
                                           "(.cfgPostProc) in network parameters"));
    }
    is_postproc = (params.custom_post_proc != nullptr);

    // Update parameters based on session information
    if (params.num_in == 1u && params.input_names.empty()) {
        params.input_names = { in_tensor_info.front().name };
    }
    if (params.num_out == 1u && params.output_names.empty()) {
        params.output_names = { out_tensor_info.front().name };
    }

    // Validate what is supported currently
    GAPI_Assert(std::all_of(in_tensor_info.begin(),
                            in_tensor_info.end(),
                            [](const cv::gimpl::onnx::TensorInfo &p) {
                                return p.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                    || p.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                                    || p.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                                    || p.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
                            })
                && "Only FP32, INT32, INT64 and U8 inputs for NN are supported");

    // Put mean and std in appropriate tensor params
    if (!params.mean.empty() || !params.stdev.empty()) {
        GAPI_Assert(params.mean.size() == params.stdev.size() &&
                    params.mean.size() == params.input_names.size());
        for (auto idx : ade::util::iota(params.num_in)) {
            const auto ort_idx = getIdxByName(in_tensor_info, params.input_names[idx]);
            using M = TensorInfo::MeanStdev;
            in_tensor_info[ort_idx].mstd = util::make_optional(M{ params.mean[idx]
                                                                , params.stdev[idx] });
        }
    }

    // Update normalize flags for input tensors
    if (!params.normalize.empty()) {
        for (auto idx : ade::util::iota(params.num_in)) {
            const auto ort_idx = getIdxByName(in_tensor_info, params.input_names[idx]);
            in_tensor_info[ort_idx].normalize = params.normalize[idx];
        }
    }

    if (!params.const_inputs.empty()) {
        // Form input names order without const input names
        in_names_without_const.clear();
        std::copy_if(params.input_names.begin(), params.input_names.end(),
                     std::back_inserter(in_names_without_const),
                     [&](const std::string& name) {
                        const auto it = params.const_inputs.find(name);
                        return it == params.const_inputs.end();
                     });
    }

    // Pre-allocate vectors (not buffers) for runtime info
    in_data.resize(params.num_in);
    out_data.resize(params.num_out);
}

std::vector<TensorInfo> ONNXCompiled::getTensorInfo(TensorPosition pos) {
    GAPI_Assert(pos == INPUT || pos == OUTPUT);

    const auto num_nodes = pos == INPUT
        ? this_session.GetInputCount()
        : this_session.GetOutputCount();

    std::vector<TensorInfo> tensor_info;
    tensor_info.reserve(num_nodes);

    Ort::AllocatorWithDefaultOptions allocator;
    for (auto i : ade::util::iota(num_nodes)) {
        const auto info = pos == INPUT
            ? this_session.GetInputTypeInfo(i)
            : this_session.GetOutputTypeInfo(i);
        tensor_info.emplace_back(info.GetTensorTypeAndShapeInfo());

        Ort::AllocatedStringPtr name_p = pos == INPUT
            ? this_session.GetInputNameAllocated(i, allocator)
            : this_session.GetOutputNameAllocated(i, allocator);
        tensor_info.back().name = std::string(name_p.get());
    }

    return tensor_info;
}

cv::GMatDesc ONNXCompiled::outMeta(int idx) const {
    if (is_dynamic || is_postproc) {
        GAPI_Assert(!params.out_metas.empty()
                    && "Metadata must be specified if NN has dynamic inputs or post-processing function is used!");
        return params.out_metas.at(idx);
    }
    const auto ort_idx = getIdxByName(out_tensor_info, params.output_names[idx]);
    return cv::GMatDesc(toCV(out_tensor_info[ort_idx].type),
                        toCV(out_tensor_info[ort_idx].dims));
}

void ONNXCompiled::setInput(int in_idx, const cv::Mat &m) {
    GAPI_Assert(!m.empty() && "Input data can't be empty!");
    const auto in_name = params.input_names[in_idx];
    const auto ort_idx = getIdxByName(in_tensor_info, in_name);
    preprocess(m, in_tensor_info[ort_idx], in_data[in_idx]);
}

void ONNXCompiled::extractMat(ONNXCallContext &ctx, const size_t in_idx, Views& views) {
    switch (ctx.in_shapes[in_idx]) {
        case cv::GShape::GFRAME: {
            const cv::MediaFrame& frame = ctx.inFrame(in_idx);
            views.emplace_back(new cv::MediaFrame::View(frame.access(cv::MediaFrame::Access::R)));
            GAPI_Assert(views.size() <= numInputs());
            preprocess(*views.back(), frame.desc(), exMat);
            break;
        }
        case cv::GShape::GMAT: {
            exMat = ctx.inMat(in_idx);
            break;
        }
        default: {
            GAPI_Assert("Unsupported input shape for ONNX backend");
        }
    }
}

void ONNXCompiled::setOutput(int i, cv::Mat &m)
{
    // FIXME: No need in double-indexing?
    out_data[i] = m;
}

cv::Mat ONNXCompiled::allocOutput(int i) const {
    cv::Mat m;
    m.create(toCV(out_tensor_info[i].dims),
             toCV(out_tensor_info[i].type));
    return m;
}

void ONNXCompiled::Run(const std::vector<cv::Mat>& ins,
                       std::vector<cv::Mat>& outs) {
    std::vector<Ort::Value> in_tensors, out_tensors;

    // Layer names order for run
    auto input_names = (in_names_without_const.empty() && params.const_inputs.empty())
                       ? params.input_names
                       : in_names_without_const;
    // Creates tensors for unique names that don't contain constant input
    for (const auto it : ade::util::indexed(input_names)) {
        auto i         = ade::util::index(it);
        auto in_name   = ade::util::value(it);
        const auto idx = getIdxByName(in_tensor_info, in_name);
        in_tensors.emplace_back(createTensor(in_tensor_info[idx],
                                             ins[i]));
    }

    for (auto &&c_in_pair : params.const_inputs) {
        const auto idx = getIdxByName(in_tensor_info, c_in_pair.first);
        in_tensors.emplace_back(createTensor(in_tensor_info[idx],
                                             c_in_pair.second.first));
        // Puts const input names in sequence for Run
        // ONNXRuntime can match input tensors to CNN inputs by names
        input_names.emplace_back(c_in_pair.first);
    }
    GAPI_Assert(input_names.size() == this_session.GetInputCount());

    auto in_run_names  = getCharNames(input_names);
    if (!is_dynamic && !is_postproc) {
        // Easy path - just run the session which is bound to G-API's
        // internal data
        for (auto i : ade::util::iota(params.output_names.size())) {
        out_tensors.emplace_back(createTensor(out_tensor_info[i],
                                              outs[i]));
        }
        auto out_run_names = getCharNames(params.output_names);
        this_session.Run(Ort::RunOptions{nullptr},
                         in_run_names.data(),
                         &in_tensors.front(),
                         input_names.size(),
                         out_run_names.data(),
                         &out_tensors.front(),
                         params.output_names.size());
        if (out_tensor_info[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            // cv::Mat does not support int64 output data.
            // Conversion from int 64 to int 32 is carried in the copyFromONNX function
            // The output is written to out_mat
            for (auto &&iter : ade::util::zip(ade::util::toRange(out_tensors),
                                              ade::util::toRange(outs))) {
                auto &out_tensor = std::get<0>(iter);
                auto &out_mat = std::get<1>(iter);
                copyFromONNX(out_tensor, out_mat);
            }
        }
    } else {
        // Hard path - run session & user-defined post-processing
        // NOTE: use another list of output names here
        std::vector<const char*> out_names;
        out_names.reserve(outs.size());
        params.names_to_remap.empty()
            ? ade::util::transform(out_tensor_info, std::back_inserter(out_names),
                                   [] (const TensorInfo& ti) { return ti.name.c_str(); })
            : ade::util::transform(params.names_to_remap, std::back_inserter(out_names),
                                   [] (const std::string& ntr) { return ntr.c_str(); });

        auto outputs = this_session.Run(Ort::RunOptions{nullptr},
                                        in_run_names.data(),
                                        &in_tensors.front(),
                                        input_names.size(),
                                        out_names.data(),
                                        out_names.size());
        std::unordered_map<std::string, cv::Mat> onnx_outputs;
        std::unordered_map<std::string, cv::Mat> gapi_outputs;

        GAPI_Assert(outputs.size() == out_names.size());
        // Fill in ONNX tensors
        for (auto &&iter : ade::util::zip(ade::util::toRange(out_names),
                                          ade::util::toRange(outputs))) {
            const auto &out_name   = std::get<0>(iter);
                  auto &out_tensor = std::get<1>(iter);
            copyFromONNX(out_tensor, onnx_outputs[out_name]);
        }
        std::vector<uint8_t *> tracked_mat_ptrs;
        // Fill in G-API outputs
        for (auto &&it: ade::util::indexed(params.output_names)) {
            gapi_outputs[ade::util::value(it)] = outs[ade::util::index(it)];
            tracked_mat_ptrs.push_back(outs[ade::util::index(it)].data);
        }
        params.custom_post_proc(onnx_outputs, gapi_outputs);
        // Checking for possible data reallocation after remapping
        GAPI_Assert(tracked_mat_ptrs.size() == params.output_names.size());
        for (auto &&iter : ade::util::zip(ade::util::toRange(tracked_mat_ptrs),
                                          ade::util::toRange(params.output_names))) {
            const auto &original_data = std::get<0>(iter);
            const auto &received_data = gapi_outputs.at(std::get<1>(iter)).data;
            if (original_data != received_data) {
                cv::util::throw_error
                    (std::logic_error
                     ("OpenCV kernel output parameter was reallocated after remapping of ONNX output. \n"
                      "Incorrect logic in remapping function?"));
            }
        }
    }
}

void ONNXCompiled::run() {
    Run(in_data, out_data);
}

static void checkInputMeta(const cv::GMetaArg mm) {
    switch (mm.index()) {
        case cv::GMetaArg::index_of<cv::GMatDesc>(): break;
        case cv::GMetaArg::index_of<cv::GFrameDesc>(): {
            const auto &meta = util::get<cv::GFrameDesc>(mm);
            switch (meta.fmt) {
                case cv::MediaFormat::NV12: break;
                case cv::MediaFormat::BGR:  break;
                default:
                    GAPI_Error("Unsupported media format for ONNX backend");
            } break;
        } break;
        default:
            util::throw_error(std::runtime_error("Unsupported input meta for ONNX backend"));
    }
}

struct Infer: public cv::detail::KernelTag {
    using API = cv::GInferBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::onnx::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        cv::GMetaArgs result;

        GConstGONNXModel gm(gr);
        const auto &uu = gm.metadata(nh).get<ONNXUnit>();

        GAPI_Assert(uu.oc->numInputs() == in_metas.size()
                    && "Known input layers count doesn't match input meta count");
        for (auto &&mm : in_metas) {
            checkInputMeta(mm);
        }
        for (auto &&idx : ade::util::iota(uu.oc->numOutputs())) {
            result.emplace_back(uu.oc->outMeta(idx));
        }
        return result;
    }

    static void run(const ONNXUnit &uu, ONNXCallContext &ctx) {
        Views views;
        for (auto &&idx : ade::util::iota(uu.oc->numInputs())) {
            uu.oc->extractMat(ctx, idx, views);
            uu.oc->setInput(idx, uu.oc->exMat);
        }
        for (auto &&idx : ade::util::iota(uu.oc->numOutputs())) {
            uu.oc->setOutput(idx, ctx.outMatR(idx));
        }
        uu.oc->run();
    }
};

struct InferROI: public cv::detail::KernelTag {
    using API = cv::GInferROIBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::onnx::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        cv::GMetaArgs result;

        GConstGONNXModel gm(gr);
        const auto &uu = gm.metadata(nh).get<ONNXUnit>();
        GAPI_Assert(1u == uu.oc->numInputs());
        GAPI_Assert(2u == in_metas.size());
        checkInputMeta(in_metas.at(1));
        for (auto &&idx : ade::util::iota(uu.oc->numOutputs())) {
            result.emplace_back(uu.oc->outMeta(idx));
        }
        return result;
    }

    static void run(const ONNXUnit &uu, ONNXCallContext &ctx) {
        Views views;
        // non-generic version for now, per the InferROI's definition
        GAPI_Assert(uu.oc->numInputs() == 1u);
        const auto& this_roi = ctx.inArg<cv::detail::OpaqueRef>(0).rref<cv::Rect>();
        uu.oc->extractMat(ctx, 1, views);
        uu.oc->setInput(0, uu.oc->exMat(this_roi));
        for (auto &&idx : ade::util::iota(uu.oc->numOutputs())) {
            uu.oc->setOutput(idx, ctx.outMatR(idx));
        }
        uu.oc->run();
    }
};

struct InferList: public cv::detail::KernelTag {
    using API = cv::GInferListBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::onnx::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        GConstGONNXModel gm(gr);
        const auto &uu = gm.metadata(nh).get<ONNXUnit>();

        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.oc->numInputs() == (in_metas.size() - 1u)
                    && "Known input layers count doesn't match input meta count");

        for (auto i : ade::util::iota(uu.oc->numInputs())) {
            const auto &mm = in_metas[i + 1];
            checkInputMeta(mm);
        }

        // roi-list version is much easier at the moment.
        // All our outputs are vectors which don't have
        // metadata at the moment - so just create a vector of
        // "empty" array metadatas of the required size.
        return cv::GMetaArgs(uu.oc->numOutputs(),
                             cv::GMetaArg{cv::empty_array_desc()});
    }

    static void run(const ONNXUnit &uu, ONNXCallContext &ctx) {
        Views views;
        // non-generic version for now:
        // - assumes input 0 is always ROI list
        // - assumes all inputs/outputs are always Mats
        GAPI_Assert(uu.oc->numInputs() == 1); // roi list is not counted in net's inputs

        const auto& in_roi_vec = ctx.inArg<cv::detail::VectorRef>(0u).rref<cv::Rect>();

        for (auto i : ade::util::iota(uu.oc->numOutputs())) {
            ctx.outVecR<cv::Mat>(i).clear();
        }
        uu.oc->extractMat(ctx, 1, views);
        for (const auto &rc : in_roi_vec) {
            uu.oc->setInput(0, uu.oc->exMat(rc));
            std::vector<cv::Mat> out_mats(uu.oc->numOutputs());
            for (auto i : ade::util::iota(uu.oc->numOutputs())) {
                out_mats[i] = uu.oc->allocOutput(i);
                uu.oc->setOutput(i, out_mats[i]);
            }
            uu.oc->run();
            for (auto i : ade::util::iota(uu.oc->numOutputs())) {
                std::vector<cv::Mat> &out_vec = ctx.outVecR<cv::Mat>(i);
                out_vec.push_back(std::move(out_mats[i]));
            }
        }
    }
};

struct InferList2: public cv::detail::KernelTag {
    using API = cv::GInferList2Base;
    static cv::gapi::GBackend backend()  { return cv::gapi::onnx::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {

        GConstGONNXModel gm(gr);
        const auto &uu = gm.metadata(nh).get<ONNXUnit>();

        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.oc->numInputs() == (in_metas.size() - 1u)
                    && "Known input layers count doesn't match input meta count");

        // In contrast to InferList, the InferList2 has only one
        // "full-frame" image argument, and all the rest are arrays of
        // ether ROI or blobs. So here we set the 0th arg image format
        // to all inputs which are ROI-based (skipping the
        // "blob"-based ones)
        // FIXME: this is filtering not done, actually! GArrayDesc has
        // no hint for type!
        const auto &mm_0   = in_metas[0u];
        switch (in_metas[0u].index()) {
            case cv::GMetaArg::index_of<cv::GMatDesc>(): {
                const auto &meta_0 = util::get<cv::GMatDesc>(mm_0);
                GAPI_Assert(   !meta_0.isND()
                            && !meta_0.planar
                            && "Only images are supported as the 0th argument");
                break;
            }
            case cv::GMetaArg::index_of<cv::GFrameDesc>(): {
                const auto &meta_0 = util::get<cv::GFrameDesc>(mm_0);
                GAPI_Assert(   (meta_0.fmt == cv::MediaFormat::BGR)
                            || (meta_0.fmt == cv::MediaFormat::NV12));
                GAPI_Assert((meta_0.size.height !=0) && (meta_0.size.width !=0));
                break;
            }
            default:
                util::throw_error(std::runtime_error("Unsupported input meta for ONNX backend"));
        }
        if (util::holds_alternative<cv::GMatDesc>(mm_0)) {
            const auto &meta_0 = util::get<cv::GMatDesc>(mm_0);
            GAPI_Assert(   !meta_0.isND()
                        && !meta_0.planar
                        && "Only images are supported as the 0th argument");
        }
        for (auto i : ade::util::iota(uu.oc->numInputs())) {
            const auto &mm = in_metas[i + 1];
            GAPI_Assert(util::holds_alternative<cv::GArrayDesc>(mm)
                        && "Non-array inputs are not supported");
        }

        // roi-list version is much easier at the moment.
        // All our outputs are vectors which don't have
        // metadata at the moment - so just create a vector of
        // "empty" array metadatas of the required size.
        return cv::GMetaArgs(uu.oc->numOutputs(),
                             cv::GMetaArg{cv::empty_array_desc()});
    }

    static void run(const ONNXUnit &uu, ONNXCallContext &ctx) {
        Views views;
        GAPI_Assert(ctx.args.size() > 1u
                    && "This operation must have at least two arguments");
        uu.oc->extractMat(ctx, 0, views);
        // Since we do a ROI list inference, always assume our input buffer is image
        // Take the next argument, which must be vector (of any kind).
        // Use this only to obtain the ROI list size (sizes of all
        // other vectors must be equal to this one)
        const auto list_size = ctx.inArg<cv::detail::VectorRef>(1u).size();

        for (auto i : ade::util::iota(uu.oc->numOutputs())) {
            ctx.outVecR<cv::Mat>(i).clear();
        }
        // For every ROI in the list {{{
        for (const auto &list_idx : ade::util::iota(list_size)) {
            std::vector<Ort::Value> in_tensors, out_tensors;
            std::vector<cv::Mat> in_mats(uu.oc->numInputs());
            // For every input of the net {{{
            for (auto in_idx : ade::util::iota(uu.oc->numInputs())) {
                const auto &this_vec = ctx.inArg<cv::detail::VectorRef>(in_idx+1u);
                GAPI_Assert(this_vec.size() == list_size);
                // Prepare input {{{
                //   FIXME: Terrible run-time logic based on RTTI!
                //   FIXME: Will never work on non-RTTI systems!
                //   FIXME: Need to replace with a static type tags
                //   (like with serialization) instead!
                if (this_vec.holds<cv::Rect>()) {
                    // ROI case - create an ROI blob
                    const auto &vec = this_vec.rref<cv::Rect>();
                    uu.oc->setInput(in_idx, uu.oc->exMat(vec[list_idx]));
                } else if (this_vec.holds<cv::Mat>()) {
                    // Mat case - create a regular blob
                    // FIXME: NOW Assume Mats are always BLOBS (not
                    // images)
                    const auto &vec = this_vec.rref<cv::Mat>();
                    uu.oc->setInput(in_idx, vec[list_idx]);
                } else {
                    GAPI_Error("Only Rect and Mat types are supported for infer list 2!");
                }
                // }}} (Prepare input)
            } // }}} (For every input of the net)

            std::vector<cv::Mat> out_mats(uu.oc->numOutputs());
            for (auto i : ade::util::iota(uu.oc->numOutputs())) {
                out_mats[i] = uu.oc->allocOutput(i);
                uu.oc->setOutput(i, out_mats[i]);
            }
            uu.oc->run();

            for (auto i : ade::util::iota(uu.oc->numOutputs())) {
                std::vector<cv::Mat> &out_vec = ctx.outVecR<cv::Mat>(i);
                out_vec.push_back(std::move(out_mats[i]));
            }
        } // }}} (For every ROI in the list)
    }
};

} // namespace onnx
} // namespace gapi
} // namespace cv

namespace {
    class GONNXBackendImpl final: public cv::gapi::GBackend::Priv {
        virtual void unpackKernel(ade::Graph            &gr,
                                  const ade::NodeHandle &nh,
                                  const cv::GKernelImpl &ii) override {
            using namespace cv::gimpl;
            // FIXME: Introduce a DNNBackend interface which'd specify
            // the framework for this???
            GONNXModel gm(gr);
            auto &np = gm.metadata(nh).get<NetworkParams>();
            auto &pp = cv::util::any_cast<cv::gapi::onnx::detail::ParamDesc>(np.opaque);
            const auto &ki = cv::util::any_cast<KImpl>(ii.opaque);

            GModel::Graph model(gr);
            auto& op = model.metadata(nh).get<Op>();
            if (pp.is_generic) {
                auto& info = cv::util::any_cast<cv::detail::InOutInfo>(op.params);

                for (const auto& layer_name : info.in_names)
                {
                    pp.input_names.push_back(layer_name);
                    if (!pp.generic_mstd.empty()) {
                        const auto &ms = pp.generic_mstd.at(layer_name);
                        pp.mean.push_back(ms.first);
                        pp.stdev.push_back(ms.second);
                    }
                    if (!pp.generic_norm.empty()) {
                        pp.normalize.push_back(pp.generic_norm.at(layer_name));
                    }
                }
                pp.num_in = info.in_names.size();

                // Incorporate extra parameters associated with input layer names
                // FIXME(DM): The current form assumes ALL input layers require
                // this information, this is obviously not correct

                for (const auto& a : info.out_names)
                {
                    pp.output_names.push_back(a);
                }
                pp.num_out = info.out_names.size();
            } // if(is_generic) -- note, the structure is already filled at the user
              // end when a non-generic Params are used

            gm.metadata(nh).set(ONNXUnit{pp});
            gm.metadata(nh).set(ONNXCallable{ki.run});
            gm.metadata(nh).set(CustomMetaFunction{ki.customMetaFunc});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override {
            return EPtr{new cv::gimpl::onnx::GONNXExecutable(graph, nodes)};
        }

        virtual cv::GKernelPackage auxiliaryKernels() const override {
            return cv::gapi::kernels< cv::gimpl::onnx::Infer
                                    , cv::gimpl::onnx::InferROI
                                    , cv::gimpl::onnx::InferList
                                    , cv::gimpl::onnx::InferList2
                                    >();
        }
    };
}

cv::gapi::GBackend cv::gapi::onnx::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GONNXBackendImpl>());
    return this_backend;
}
#else // HAVE_ONNX

cv::gapi::GBackend cv::gapi::onnx::backend() {
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("G-API has been compiled without ONNX support"));
}
#endif // HAVE_ONNX
