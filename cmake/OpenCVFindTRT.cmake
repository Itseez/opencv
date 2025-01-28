set(TRT_SDK "" CACHE PATH "TensorRT SDK directory.")

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
  message ( STATUS  "TensorRT backend: CUDAToolkit_FOUND=${CUDAToolkit_FOUND}!")
  message ( STATUS  "TensorRT backend: CUDAToolkit_VERSION=${CUDAToolkit_VERSION}" )
else()
  message(STATUS "TensorRT backend: CAN NOT found CUDAToolkit! Please check if enviroment has the CUDA_PATH!")
endif()

set(TRT_SDK_LIB_PATH "${TRT_SDK}/lib")
set(TRT_SDK_INC "${TRT_SDK}/include")

find_library(TRT_SDK_LIB "nvinfer" PATHS "${TRT_SDK_LIB_PATH}" NO_DEFAULT_PATH)
find_library(TRT_SDK_LIB_ONNX "nvonnxparser" PATHS "${TRT_SDK_LIB_PATH}" NO_DEFAULT_PATH)

set(TRT_FOUND 0)
if(TRT_SDK_LIB AND TRT_SDK_LIB_ONNX AND CUDAToolkit_FOUND)
  set(TRT_FOUND 1)
else()
  message(STATUS "TensorRT backend: Failed to find nvinfer in ${TRT_SDK_LIB_PATH}. Please note that ${TRT_SDK} must contain the lib and include folder! Turning off TensorRT backend!")
  set(TRT_FOUND 0)
endif()

if(TRT_FOUND)
  set(HAVE_TRT 1)
endif()

MARK_AS_ADVANCED(
  TRT_SDK_INC
  TRT_SDK_LIB
  TRT_SDK_LIB_ONNX
)
