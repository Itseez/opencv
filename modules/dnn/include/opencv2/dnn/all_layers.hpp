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

#ifndef OPENCV_DNN_DNN_ALL_LAYERS_HPP
#define OPENCV_DNN_DNN_ALL_LAYERS_HPP
#include <opencv2/dnn.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
//! @addtogroup dnn
//! @{

/** @defgroup dnnLayerList Partial List of Implemented Layers
  @{
  This subsection of dnn module contains information about built-in layers and their descriptions.

  Classes listed here, in fact, provides C++ API for creating instances of built-in layers.
  In addition to this way of layers instantiation, there is a more common factory API (see @ref dnnLayerFactory), it allows to create layers dynamically (by name) and register new ones.
  You can use both API, but factory API is less convenient for native C++ programming and basically designed for use inside importers (see @ref readNetFromCaffe(), @ref readNetFromTensorflow()).

  Built-in layers partially reproduce functionality of corresponding ONNX, TensorFlow and Caffe layers.
  In particular, the following layers and Caffe importer were tested to reproduce <a href="http://caffe.berkeleyvision.org/tutorial/layers.html">Caffe</a> functionality:
  - Convolution
  - Deconvolution
  - Pooling
  - InnerProduct
  - TanH, ReLU, Sigmoid, BNLL, Power, AbsVal
  - Softmax
  - Reshape, Flatten, Slice, Split
  - LRN
  - MVN
  - Dropout (since it does nothing on forward pass -))
*/

    class CV_EXPORTS BlankLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams &params);
    };

    /**
     * Constant layer produces the same data blob at an every forward pass.
     */
    class CV_EXPORTS ConstLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams &params);
    };

    /**
     * Constant layer produces the same data blob at an every forward pass.
     */
    class CV_EXPORTS ConstantOfShapeLayer : public Layer
    {
    public:
        static Ptr<ConstantOfShapeLayer> create(const LayerParams &params);
    };

    //! LSTM recurrent layer
    class CV_EXPORTS LSTMLayer : public Layer
    {
    public:
        /** Creates instance of LSTM layer */
        static Ptr<LSTMLayer> create(const LayerParams& params);

        /** @deprecated Use LayerParams::blobs instead.
        @brief Set trained weights for LSTM layer.

        LSTM behavior on each step is defined by current input, previous output, previous cell state and learned weights.

        Let @f$x_t@f$ be current input, @f$h_t@f$ be current output, @f$c_t@f$ be current state.
        Than current output and current cell state is computed as follows:
        @f{eqnarray*}{
        h_t &= o_t \odot tanh(c_t),               \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \\
        @f}
        where @f$\odot@f$ is per-element multiply operation and @f$i_t, f_t, o_t, g_t@f$ is internal gates that are computed using learned weights.

        Gates are computed as follows:
        @f{eqnarray*}{
        i_t &= sigmoid&(W_{xi} x_t + W_{hi} h_{t-1} + b_i), \\
        f_t &= sigmoid&(W_{xf} x_t + W_{hf} h_{t-1} + b_f), \\
        o_t &= sigmoid&(W_{xo} x_t + W_{ho} h_{t-1} + b_o), \\
        g_t &= tanh   &(W_{xg} x_t + W_{hg} h_{t-1} + b_g), \\
        @f}
        where @f$W_{x?}@f$, @f$W_{h?}@f$ and @f$b_{?}@f$ are learned weights represented as matrices:
        @f$W_{x?} \in R^{N_h \times N_x}@f$, @f$W_{h?} \in R^{N_h \times N_h}@f$, @f$b_? \in R^{N_h}@f$.

        For simplicity and performance purposes we use @f$ W_x = [W_{xi}; W_{xf}; W_{xo}, W_{xg}] @f$
        (i.e. @f$W_x@f$ is vertical concatenation of @f$ W_{x?} @f$), @f$ W_x \in R^{4N_h \times N_x} @f$.
        The same for @f$ W_h = [W_{hi}; W_{hf}; W_{ho}, W_{hg}], W_h \in R^{4N_h \times N_h} @f$
        and for @f$ b = [b_i; b_f, b_o, b_g]@f$, @f$b \in R^{4N_h} @f$.

        @param Wh is matrix defining how previous output is transformed to internal gates (i.e. according to above mentioned notation is @f$ W_h @f$)
        @param Wx is matrix defining how current input is transformed to internal gates (i.e. according to above mentioned notation is @f$ W_x @f$)
        @param b  is bias vector (i.e. according to above mentioned notation is @f$ b @f$)
        */
        CV_DEPRECATED virtual void setWeights(const Mat &Wh, const Mat &Wx, const Mat &b) = 0;

        /** @brief Specifies shape of output blob which will be [[`T`], `N`] + @p outTailShape.
          * @details If this parameter is empty or unset then @p outTailShape = [`Wh`.size(0)] will be used,
          * where `Wh` is parameter from setWeights().
          */
        virtual void setOutShape(const MatShape &outTailShape = MatShape()) = 0;

        /** @deprecated Use flag `produce_cell_output` in LayerParams.
          * @brief Specifies either interpret first dimension of input blob as timestamp dimension either as sample.
          *
          * If flag is set to true then shape of input blob will be interpreted as [`T`, `N`, `[data dims]`] where `T` specifies number of timestamps, `N` is number of independent streams.
          * In this case each forward() call will iterate through `T` timestamps and update layer's state `T` times.
          *
          * If flag is set to false then shape of input blob will be interpreted as [`N`, `[data dims]`].
          * In this case each forward() call will make one iteration and produce one timestamp with shape [`N`, `[out dims]`].
          */
        CV_DEPRECATED virtual void setUseTimstampsDim(bool use = true) = 0;

        /** @deprecated Use flag `use_timestamp_dim` in LayerParams.
         * @brief If this flag is set to true then layer will produce @f$ c_t @f$ as second output.
         * @details Shape of the second output is the same as first output.
         */
        CV_DEPRECATED virtual void setProduceCellOutput(bool produce = false) = 0;

        /* In common case it use single input with @f$x_t@f$ values to compute output(s) @f$h_t@f$ (and @f$c_t@f$).
         * @param input should contain packed values @f$x_t@f$
         * @param output contains computed outputs: @f$h_t@f$ (and @f$c_t@f$ if setProduceCellOutput() flag was set to true).
         *
         * If setUseTimstampsDim() is set to true then @p input[0] should has at least two dimensions with the following shape: [`T`, `N`, `[data dims]`],
         * where `T` specifies number of timestamps, `N` is number of independent streams (i.e. @f$ x_{t_0 + t}^{stream} @f$ is stored inside @p input[0][t, stream, ...]).
         *
         * If setUseTimstampsDim() is set to false then @p input[0] should contain single timestamp, its shape should has form [`N`, `[data dims]`] with at least one dimension.
         * (i.e. @f$ x_{t}^{stream} @f$ is stored inside @p input[0][stream, ...]).
        */

        int inputNameToIndex(String inputName) CV_OVERRIDE;
        int outputNameToIndex(const String& outputName) CV_OVERRIDE;
    };

    class CV_EXPORTS LSTM2Layer : public Layer
    {
    public:
        /** Creates instance of LSTM layer */
        static Ptr<LSTM2Layer> create(const LayerParams& params);
    };

    /** @brief GRU recurrent one-layer
     *
     * Accepts input sequence and computes the final hidden state for each element in the batch.
     *
     * - input[0] containing the features of the input sequence.
     * input[0] should have shape [`T`, `N`, `data_dims`] where `T` is sequence length, `N` is batch size, `data_dims` is input size
     * - output would have shape [`T`, `N`, `D` * `hidden_size`] where `D = 2` if layer is bidirectional otherwise `D = 1`
     *
     * Depends on the following attributes:
     * - hidden_size - Number of neurons in the hidden layer
     * - direction - RNN could be bidirectional or forward
     *
     * The final hidden state @f$ h_t @f$ computes by the following formulas:
     *
     @f{eqnarray*}{
     r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
     z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
     n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
     h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)} \\
     @f}
     * Where @f$x_t@f$ is current input, @f$h_{(t-1)}@f$ is previous or initial hidden state.
     *
     * @f$W_{x?}@f$, @f$W_{h?}@f$ and @f$b_{?}@f$ are learned weights represented as matrices:
     * @f$W_{x?} \in R^{N_h \times N_x}@f$, @f$W_{h?} \in R^{N_h \times N_h}@f$, @f$b_? \in R^{N_h}@f$.
     *
     * @f$\odot@f$ is per-element multiply operation.
    */
    class CV_EXPORTS GRULayer : public Layer
    {
    public:
        /** Creates instance of GRU layer */
        static Ptr<GRULayer> create(const LayerParams& params);
    };

    /** @brief Classical recurrent layer

    Accepts two inputs @f$x_t@f$ and @f$h_{t-1}@f$ and compute two outputs @f$o_t@f$ and @f$h_t@f$.

    - input: should contain packed input @f$x_t@f$.
    - output: should contain output @f$o_t@f$ (and @f$h_t@f$ if setProduceHiddenOutput() is set to true).

    input[0] should have shape [`T`, `N`, `data_dims`] where `T` and `N` is number of timestamps and number of independent samples of @f$x_t@f$ respectively.

    output[0] will have shape [`T`, `N`, @f$N_o@f$], where @f$N_o@f$ is number of rows in @f$ W_{xo} @f$ matrix.

    If setProduceHiddenOutput() is set to true then @p output[1] will contain a Mat with shape [`T`, `N`, @f$N_h@f$], where @f$N_h@f$ is number of rows in @f$ W_{hh} @f$ matrix.
    */
    class CV_EXPORTS RNNLayer : public Layer
    {
    public:
        /** Creates instance of RNNLayer */
        static Ptr<RNNLayer> create(const LayerParams& params);

        /** Setups learned weights.

        Recurrent-layer behavior on each step is defined by current input @f$ x_t @f$, previous state @f$ h_t @f$ and learned weights as follows:
        @f{eqnarray*}{
        h_t &= tanh&(W_{hh} h_{t-1} + W_{xh} x_t + b_h),  \\
        o_t &= tanh&(W_{ho} h_t + b_o),
        @f}

        @param Wxh is @f$ W_{xh} @f$ matrix
        @param bh  is @f$ b_{h}  @f$ vector
        @param Whh is @f$ W_{hh} @f$ matrix
        @param Who is @f$ W_{xo} @f$ matrix
        @param bo  is @f$ b_{o}  @f$ vector
        */
        virtual void setWeights(const Mat &Wxh, const Mat &bh, const Mat &Whh, const Mat &Who, const Mat &bo) = 0;

        /** @brief If this flag is set to true then layer will produce @f$ h_t @f$ as second output.
         * @details Shape of the second output is the same as first output.
         */
        virtual void setProduceHiddenOutput(bool produce = false) = 0;

    };

    /** @brief This function performs array summation based
    * on the Einstein summation convention. The function
    * allows for concise expressions of various mathematical
    * operations using subscripts.
    *
    * By default, the labels are placed in alphabetical
    * order at the end of the output.
    * For example:
    * if `c = einsum("i,j", a, b)`, then `c[i,j] == a[i]*b[j]`.
    * However, if `c = einsum("j,i", a, b)`, then `c[i,j] = a[j]*b[i]`.
    * Alternatively, you can control the output order or prevent
    * an axis from being summed/force an axis to be summed
    * by providing indices for the output.
    * For example:
    * `diag(a)`         -> `einsum("ii->i", a)`
    * `sum(a, axis=0)`  -> `einsum("i...->", a)`
    * Subscripts at the beginning and end may be specified
    * by putting an ellipsis "..." in the middle.
    * For instance, the function `einsum("i...i", a)` takes
    * the diagonal of the first and last dimensions of
    * the operand, and `einsum("ij...,jk...->ik...")` performs
    * the matrix product using the first two indices
    * of each operand instead of the last two.
    * When there is only one operand, no axes being summed,
    *  and no output parameter, this function returns
    * a view into the operand instead of creating a copy.
     */
    class CV_EXPORTS EinsumLayer : public Layer
    {
    public:
        static Ptr<EinsumLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS HardmaxLayer : public Layer
    {
    public:
        static Ptr<HardmaxLayer> create(const LayerParams& params);
    };

    /** @brief GatherND layer
     *
     * GatherND takes two inputs data and indices of rank r >= 1 and q >= 1 respectively,
     * and an optional attribute batch_dims. It gathers slices from data into an output tensor.
     */
    class CV_EXPORTS GatherNDLayer : public Layer
    {
    public:
        static Ptr<GatherNDLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS BaseConvolutionLayer : public Layer
    {
    public:
        CV_DEPRECATED_EXTERNAL Size kernel, stride, pad, dilation, adjustPad;
        std::vector<size_t> adjust_pads;
        std::vector<size_t> kernel_size, strides, dilations;
        std::vector<size_t> pads_begin, pads_end;
        String padMode;
        int numOutput;
    };

    class CV_EXPORTS ConvolutionLayer : public BaseConvolutionLayer
    {
    public:
        static Ptr<BaseConvolutionLayer> create(const LayerParams& params);
        bool fusedActivation = false;
        bool fusedAdd = false;
        bool useWinograd = true; // Flag whether to use Winograd to speed up 3x3 convolution.
    };

    class CV_EXPORTS ConvolutionLayerInt8 : public BaseConvolutionLayer
    {
    public:
        int input_zp, output_zp;
        float input_sc, output_sc;

        // quantization type flag. The perChannel default is true, that means it contains the parameters
        // of per-Channel quantization. Otherwise, that means this layer contains per-Tensor quantized parameters.
        bool per_channel;
        bool useWinograd = false; // Flag whether to use Winograd to speed up 3x3 convolution.
        static Ptr<BaseConvolutionLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS DeconvolutionLayer : public BaseConvolutionLayer
    {
    public:
        static Ptr<BaseConvolutionLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS LRNLayer : public Layer
    {
    public:
        int type;

        int size;
        float alpha, beta, bias;
        bool normBySize;

        static Ptr<LRNLayer> create(const LayerParams& params);
    };


    /** @brief ArgMax/ArgMin layer
     * @note returns indices as floats, which means the supported range is [-2^24; 2^24]
     */
    class CV_EXPORTS ArgLayer : public Layer
    {
    public:
        static Ptr<ArgLayer> create(const LayerParams& params);
    };

    /** @brief Gather layer
     */
    class CV_EXPORTS GatherLayer : public Layer
    {
    public:
        static Ptr<GatherLayer> create(const LayerParams& params);
    };

    // ONNX-compliant implementation of Gather
    class CV_EXPORTS Gather2Layer : public Layer
    {
    public:
        int axis;

        static Ptr<Gather2Layer> create(const LayerParams& params);
    };

    /** @brief GatherElements layer
    * GatherElements takes two inputs data and indices of the same rank r >= 1 and an optional attribute axis and works such that:
    *   output[i][j][k] = data[index[i][j][k]][j][k] if axis = 0 and r = 3
    *   output[i][j][k] = data[i][index[i][j][k]][k] if axis = 1 and r = 3
    *   output[i][j][k] = data[i][j][index[i][j][k]] if axis = 2 and r = 3
    *
    * Gather, on the other hand, takes a data tensor of rank r >= 1, and indices tensor of rank q, and works such that:
    *   it gathers the enteries along axis dimension of the input data indexed by indices and concatenates them in an output tensor of rank q + (r - 1)
    *   e.g. If axis = 0, let k = indices[i_{0}, ..., i_{q-1}] then output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]:
     **/
    class CV_EXPORTS GatherElementsLayer : public Layer
    {
    public:
        static Ptr<GatherElementsLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS PoolingLayer : public Layer
    {
    public:
        int type;
        std::vector<size_t> kernel_size, strides;
        std::vector<size_t> pads_begin, pads_end;
        bool globalPooling; //!< Flag is true if at least one of the axes is global pooled.
        std::vector<bool> isGlobalPooling;
        bool computeMaxIdx;
        String padMode;
        bool ceilMode;
        // If true for average pooling with padding, divide an every output region
        // by a whole kernel area. Otherwise exclude zero padded values and divide
        // by number of real values.
        bool avePoolPaddedArea;
        // ROIPooling parameters.
        Size pooledSize;
        float spatialScale;
        // PSROIPooling parameters.
        int psRoiOutChannels;

        static Ptr<PoolingLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS PoolingLayerInt8 : public PoolingLayer
    {
    public:
        int input_zp, output_zp;
        float input_sc, output_sc;
        static Ptr<PoolingLayerInt8> create(const LayerParams& params);
    };

    class CV_EXPORTS ReduceLayer : public Layer
    {
    public:
        static Ptr<ReduceLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS SoftmaxLayer : public Layer
    {
    public:
        bool logSoftMax;

        static Ptr<SoftmaxLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS SoftmaxLayerInt8 : public SoftmaxLayer
    {
    public:
        float output_sc;
        int output_zp;
        static Ptr<SoftmaxLayerInt8> create(const LayerParams& params);
    };

    /**
     * `InnerProduct`, `MatMul` and `Gemm` operations are all implemented by Fully Connected Layer.
     * Parameter `is_matmul` is used to distinguish `MatMul` and `Gemm` from `InnerProduct`.
     */
    class CV_EXPORTS InnerProductLayer : public Layer
    {
    public:
        int axis;
        static Ptr<InnerProductLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS InnerProductLayerInt8 : public InnerProductLayer
    {
    public:
        int input_zp, output_zp;
        float input_sc, output_sc;

        // quantization type flag. The perChannel default is true, that means it contains the parameters
        // of per-Channel quantization. Otherwise, that means this layer contains per-Tensor quantized parameters.
        bool per_channel;
        static Ptr<InnerProductLayerInt8> create(const LayerParams& params);
    };

    class CV_EXPORTS MVNLayer : public Layer
    {
    public:
        float eps;
        bool normVariance, acrossChannels;

        static Ptr<MVNLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS ShapeLayer : public Layer
    {
    public:
        int start, end;

        static Ptr<ShapeLayer> create(const LayerParams& params);
    };

    /* Reshaping */

    class CV_EXPORTS ReshapeLayer : public Layer
    {
    public:
        MatShape newShapeDesc;
        Range newShapeRange;

        static Ptr<ReshapeLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS Reshape2Layer : public Layer
    {
    public:
        MatShape newShapeDesc;

        static Ptr<Reshape2Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS FlattenLayer : public Layer
    {
    public:
        static Ptr<FlattenLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SqueezeLayer : public Layer
    {
    public:
        std::vector<int> axes;

        static Ptr<SqueezeLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS UnsqueezeLayer : public Layer
    {
    public:
        std::vector<int> axes;

        static Ptr<UnsqueezeLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS RangeLayer : public Layer
    {
    public:
        static Ptr<RangeLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS QuantizeLayer : public Layer
    {
    public:
        std::vector<float> scales;
        std::vector<int> zeropoints;
        static Ptr<QuantizeLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS QuantizeLinearLayer : public Layer
    {
    public:
        int axis;
        int block_size;
        int output_dtype;
        bool saturate;

        static Ptr<QuantizeLinearLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS DequantizeLayer : public Layer
    {
    public:
        std::vector<float> scales;
        std::vector<int> zeropoints;
        static Ptr<DequantizeLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS DequantizeLinearLayer : public Layer
    {
    public:
        int axis;
        int block_size;

        static Ptr<DequantizeLinearLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS RequantizeLayer : public Layer
    {
    public:
        float scale, shift;
        static Ptr<RequantizeLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ConcatLayer : public Layer
    {
    public:
        int axis;
        /**
         * @brief Add zero padding in case of concatenation of blobs with different
         * spatial sizes.
         *
         * Details: https://github.com/torch/nn/blob/master/doc/containers.md#depthconcat
         */
        bool padding;
        int paddingValue;

        static Ptr<ConcatLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS Concat2Layer : public Layer
    {
    public:
        int axis;

        static Ptr<Concat2Layer> create(const LayerParams &params);
    };

    class CV_EXPORTS SplitLayer : public Layer
    {
    public:
        int outputsCount; //!< Number of copies that will be produced (is ignored when negative).

        static Ptr<SplitLayer> create(const LayerParams &params);
    };

    // ONNX-compliant version of Split
    class CV_EXPORTS Split2Layer : public Layer
    {
    public:
        int axis;
        std::vector<int> split;

        static Ptr<Split2Layer> create(const LayerParams& params);
    };

    /**
     * Slice layer has several modes:
     * 1. Caffe mode
     * @param[in] axis Axis of split operation
     * @param[in] slice_point Array of split points
     *
     * Number of output blobs equals to number of split points plus one. The
     * first blob is a slice on input from 0 to @p slice_point[0] - 1 by @p axis,
     * the second output blob is a slice of input from @p slice_point[0] to
     * @p slice_point[1] - 1 by @p axis and the last output blob is a slice of
     * input from @p slice_point[-1] up to the end of @p axis size.
     *
     * 2. TensorFlow mode
     * @param begin Vector of start indices
     * @param size Vector of sizes
     *
     * More convenient numpy-like slice. One and only output blob
     * is a slice `input[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1], ...]`
     *
     * 3. Torch mode
     * @param axis Axis of split operation
     *
     * Split input blob on the equal parts by @p axis.
     */
    class CV_EXPORTS SliceLayer : public Layer
    {
    public:
        /**
         * @brief Vector of slice ranges.
         *
         * The first dimension equals number of output blobs.
         * Inner vector has slice ranges for the first number of input dimensions.
         */
        std::vector<std::vector<Range> > sliceRanges;
        std::vector<std::vector<int> > sliceSteps;
        int axis;
        int num_split;

        static Ptr<SliceLayer> create(const LayerParams &params);
    };

    // ONNX-compliant version of Slice
    class CV_EXPORTS Slice2Layer : public Layer
    {
    public:
        std::vector<int> starts, ends, axes;

        static Ptr<Slice2Layer> create(const LayerParams &params);
    };

    class CV_EXPORTS PermuteLayer : public Layer
    {
    public:
        static Ptr<PermuteLayer> create(const LayerParams& params);
    };

    // ONNX-compliant version of Transpose
    // (previously implemented in PermuteLayer)
    class CV_EXPORTS TransposeLayer : public Layer
    {
    public:
        std::vector<int> perm;

        static Ptr<TransposeLayer> create(const LayerParams& params);
    };

    /**
     * Permute channels of 4-dimensional input blob.
     * @param group Number of groups to split input channels and pick in turns
     *              into output blob.
     *
     * \f[ groupSize = \frac{number\ of\ channels}{group} \f]
     * \f[ output(n, c, h, w) = input(n, groupSize \times (c \% group) + \lfloor \frac{c}{group} \rfloor, h, w) \f]
     * Read more at https://arxiv.org/pdf/1707.01083.pdf
     */
    class CV_EXPORTS ShuffleChannelLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);

        int group;
    };

    /**
     * @brief Adds extra values for specific axes.
     * @param paddings Vector of paddings in format
     *                 @code
     *                 [ pad_before, pad_after,  // [0]th dimension
     *                   pad_before, pad_after,  // [1]st dimension
     *                   ...
     *                   pad_before, pad_after ] // [n]th dimension
     *                 @endcode
     *                 that represents number of padded values at every dimension
     *                 starting from the first one. The rest of dimensions won't
     *                 be padded.
     * @param value Value to be padded. Defaults to zero.
     * @param type Padding type: 'constant', 'reflect'
     * @param input_dims Torch's parameter. If @p input_dims is not equal to the
     *                   actual input dimensionality then the `[0]th` dimension
     *                   is considered as a batch dimension and @p paddings are shifted
     *                   to a one dimension. Defaults to `-1` that means padding
     *                   corresponding to @p paddings.
     */
    class CV_EXPORTS PaddingLayer : public Layer
    {
    public:
        static Ptr<PaddingLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS Pad2Layer : public Layer
    {
    public:
        static Ptr<Pad2Layer> create(const LayerParams& params);
    };

    /* Activations */
    class CV_EXPORTS ActivationLayer : public Layer
    {
    public:
        virtual void forwardSlice(const float* src, float* dst, int len,
                                  size_t outPlaneSize, int cn0, int cn1) const {}
        virtual void forwardSlice(const int* src, const int* lut, int* dst, int len,
                                  size_t outPlaneSize, int cn0, int cn1) const {}
        virtual void forwardSlice(const int8_t* src, const int8_t* lut, int8_t* dst, int len,
                                  size_t outPlaneSize, int cn0, int cn1) const {}
    };

    class CV_EXPORTS ReLULayer : public ActivationLayer
    {
    public:
        float negativeSlope;

        static Ptr<ReLULayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ReLU6Layer : public ActivationLayer
    {
    public:
        float minValue, maxValue;

        static Ptr<ReLU6Layer> create(const LayerParams &params);
    };

    class CV_EXPORTS ChannelsPReLULayer : public ActivationLayer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS ELULayer : public ActivationLayer
    {
    public:
        float alpha;

        static Ptr<ELULayer> create(const LayerParams &params);
    };

    class CV_EXPORTS TanHLayer : public ActivationLayer
    {
    public:
        static Ptr<TanHLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SwishLayer : public ActivationLayer
    {
    public:
        static Ptr<SwishLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS MishLayer : public ActivationLayer
    {
    public:
        static Ptr<MishLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SigmoidLayer : public ActivationLayer
    {
    public:
        static Ptr<SigmoidLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS BNLLLayer : public ActivationLayer
    {
    public:
        static Ptr<BNLLLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AbsLayer : public ActivationLayer
    {
    public:
        static Ptr<AbsLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS PowerLayer : public ActivationLayer
    {
    public:
        float power, scale, shift;

        static Ptr<PowerLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ExpLayer : public ActivationLayer
    {
    public:
        float base, scale, shift;

        static Ptr<ExpLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS CeilLayer : public ActivationLayer
    {
    public:
        static Ptr<CeilLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS FloorLayer : public ActivationLayer
    {
    public:
        static Ptr<FloorLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS LogLayer : public ActivationLayer
    {
    public:
        static Ptr<LogLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS RoundLayer : public ActivationLayer
    {
    public:
        static Ptr<RoundLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SqrtLayer : public ActivationLayer
    {
    public:
        static Ptr<SqrtLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS NotLayer : public Layer
    {
    public:
        static Ptr<NotLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AcosLayer : public ActivationLayer
    {
    public:
        static Ptr<AcosLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AcoshLayer : public ActivationLayer
    {
    public:
        static Ptr<AcoshLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AsinLayer : public ActivationLayer
    {
    public:
        static Ptr<AsinLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AsinhLayer : public ActivationLayer
    {
    public:
        static Ptr<AsinhLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AtanLayer : public ActivationLayer
    {
    public:
        static Ptr<AtanLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AtanhLayer : public ActivationLayer
    {
    public:
        static Ptr<AtanhLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS CosLayer : public ActivationLayer
    {
    public:
        static Ptr<CosLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS CoshLayer : public ActivationLayer
    {
    public:
        static Ptr<CoshLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ErfLayer : public ActivationLayer
    {
    public:
        static Ptr<ErfLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS HardSwishLayer : public ActivationLayer
    {
    public:
        static Ptr<HardSwishLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SinLayer : public ActivationLayer
    {
    public:
        static Ptr<SinLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SinhLayer : public ActivationLayer
    {
    public:
        static Ptr<SinhLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SoftplusLayer : public ActivationLayer
    {
    public:
        static Ptr<SoftplusLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SoftsignLayer : public ActivationLayer
    {
    public:
        static Ptr<SoftsignLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS TanLayer : public ActivationLayer
    {
    public:
        static Ptr<TanLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS CeluLayer : public ActivationLayer
    {
    public:
        float alpha;

        static Ptr<CeluLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS HardSigmoidLayer : public ActivationLayer
    {
    public:
        float alpha;
        float beta;

        static Ptr<HardSigmoidLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SeluLayer : public ActivationLayer
    {
    public:
        float alpha;
        float gamma;

        static Ptr<SeluLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS GeluLayer : public ActivationLayer
    {
    public:
        static Ptr<GeluLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS GeluApproximationLayer : public ActivationLayer
    {
    public:
        static Ptr<GeluApproximationLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ThresholdedReluLayer : public ActivationLayer
    {
    public:
        float alpha;

        static Ptr<ThresholdedReluLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ActivationLayerInt8 : public ActivationLayer
    {
    public:
        static Ptr<ActivationLayerInt8> create(const LayerParams &params);
    };

    class CV_EXPORTS SignLayer : public ActivationLayer
    {
    public:
        static Ptr<SignLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ShrinkLayer : public ActivationLayer
    {
    public:
        float bias;
        float lambd;
        static Ptr<ShrinkLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ReciprocalLayer : public ActivationLayer
    {
    public:
        static Ptr<ReciprocalLayer> create(const LayerParams &params);
    };

    /* Layers used in semantic segmentation */

    class CV_EXPORTS CropLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams &params);
    };

    /** @brief Element wise operation on inputs

    Extra optional parameters:
    - "operation" as string. Values are "sum" (default), "prod", "max", "div", "min"
    - "coeff" as float array. Specify weights of inputs for SUM operation
    - "output_channels_mode" as string. Values are "same" (default, all input must have the same layout), "input_0", "input_0_truncate", "max_input_channels"
    */
    class CV_EXPORTS EltwiseLayer : public Layer
    {
    public:
        static Ptr<EltwiseLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS EltwiseLayerInt8 : public Layer
    {
    public:
        static Ptr<EltwiseLayerInt8> create(const LayerParams &params);
    };

    class CV_EXPORTS NaryEltwiseLayer : public Layer
    {
    public:
        static Ptr<NaryEltwiseLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS BatchNormLayer : public ActivationLayer
    {
    public:
        bool hasWeights, hasBias;
        float epsilon;

        static Ptr<BatchNormLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS BatchNormLayerInt8 : public BatchNormLayer
    {
    public:
        float input_sc, output_sc;
        int input_zp, output_zp;
        static Ptr<BatchNormLayerInt8> create(const LayerParams &params);
    };

    class CV_EXPORTS MaxUnpoolLayer : public Layer
    {
    public:
        Size poolKernel;
        Size poolPad;
        Size poolStride;

        static Ptr<MaxUnpoolLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ScaleLayer : public Layer
    {
    public:
        bool hasBias;
        int axis;
        String mode;

        static Ptr<ScaleLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS ScaleLayerInt8 : public ScaleLayer
    {
    public:
        float output_sc;
        int output_zp;
        static Ptr<ScaleLayerInt8> create(const LayerParams &params);
    };

    class CV_EXPORTS ShiftLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS ShiftLayerInt8 : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS CompareLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS DataAugmentationLayer : public Layer
    {
    public:
        static Ptr<DataAugmentationLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS CorrelationLayer : public Layer
    {
    public:
        static Ptr<CorrelationLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS AccumLayer : public Layer
    {
    public:
        static Ptr<AccumLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS FlowWarpLayer : public Layer
    {
    public:
        static Ptr<FlowWarpLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS PriorBoxLayer : public Layer
    {
    public:
        static Ptr<PriorBoxLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS ReorgLayer : public Layer
    {
    public:
        static Ptr<ReorgLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS RegionLayer : public Layer
    {
    public:
        float nmsThreshold;

        static Ptr<RegionLayer> create(const LayerParams& params);
    };

    /**
     * @brief Detection output layer.
     *
     * The layer size is: @f$ (1 \times 1 \times N \times 7) @f$
     *    where N is [keep_top_k] parameter multiplied by batch size. Each row is:
     *    [image_id, label, confidence, xmin, ymin, xmax, ymax]
     *    where image_id is the index of image input in the batch.
     */
    class CV_EXPORTS DetectionOutputLayer : public Layer
    {
    public:
        static Ptr<DetectionOutputLayer> create(const LayerParams& params);
    };

    /**
     * @brief \f$ L_p \f$ - normalization layer.
     * @param p Normalization factor. The most common `p = 1` for \f$ L_1 \f$ -
     *          normalization or `p = 2` for \f$ L_2 \f$ - normalization or a custom one.
     * @param eps Parameter \f$ \epsilon \f$ to prevent a division by zero.
     * @param across_spatial If true, normalize an input across all non-batch dimensions.
     *                       Otherwise normalize an every channel separately.
     *
     * Across spatial:
     * @f[
     * norm = \sqrt[p]{\epsilon + \sum_{x, y, c} |src(x, y, c)|^p } \\
     * dst(x, y, c) = \frac{ src(x, y, c) }{norm}
     * @f]
     *
     * Channel wise normalization:
     * @f[
     * norm(c) = \sqrt[p]{\epsilon + \sum_{x, y} |src(x, y, c)|^p } \\
     * dst(x, y, c) = \frac{ src(x, y, c) }{norm(c)}
     * @f]
     *
     * Where `x, y` - spatial coordinates, `c` - channel.
     *
     * An every sample in the batch is normalized separately. Optionally,
     * output is scaled by the trained parameters.
     */
    class CV_EXPORTS NormalizeBBoxLayer : public Layer
    {
    public:
        float pnorm, epsilon;
        CV_DEPRECATED_EXTERNAL bool acrossSpatial;

        static Ptr<NormalizeBBoxLayer> create(const LayerParams& params);
    };

    /**
     * @brief Resize input 4-dimensional blob by nearest neighbor or bilinear strategy.
     *
     * Layer is used to support TensorFlow's resize_nearest_neighbor and resize_bilinear ops.
     */
    class CV_EXPORTS ResizeLayer : public Layer
    {
    public:
        static Ptr<ResizeLayer> create(const LayerParams& params);
    };

    /**
     * @brief Bilinear resize layer from https://github.com/cdmh/deeplab-public-ver2
     *
     * It differs from @ref ResizeLayer in output shape and resize scales computations.
     */
    class CV_EXPORTS InterpLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS ProposalLayer : public Layer
    {
    public:
        static Ptr<ProposalLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS CropAndResizeLayer : public Layer
    {
    public:
        static Ptr<Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS CumSumLayer : public Layer
    {
    public:
        int exclusive;
        int reverse;

        static Ptr<CumSumLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS ScatterLayer : public Layer
    {
    public:
        static Ptr<ScatterLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS ScatterNDLayer : public Layer
    {
    public:
        static Ptr<ScatterNDLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS TileLayer : public Layer
    {
    public:
        static Ptr<TileLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS Tile2Layer : public Layer
    {
    public:
        static Ptr<Tile2Layer> create(const LayerParams& params);
    };

    class CV_EXPORTS LayerNormLayer : public Layer
    {
    public:
        CV_DEPRECATED_EXTERNAL bool hasBias; // Deprecated, preserve for compatibility
        int axis;
        float epsilon;

        static Ptr<LayerNormLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS GemmLayer : public Layer {
    public:
        bool trans_a;
        bool trans_b;
        float alpha;
        float beta;

        static Ptr<GemmLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS MatMulLayer : public Layer {
     public:
        static Ptr<MatMulLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS ExpandLayer : public Layer
    {
    public:
        static Ptr<ExpandLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS Expand2Layer : public Layer
    {
    public:
        static Ptr<Expand2Layer> create(const LayerParams &params);
    };

    class CV_EXPORTS InstanceNormLayer : public Layer {
    public:
        float epsilon;

        static Ptr<InstanceNormLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS AttentionLayer : public Layer {
     public:
        static Ptr<AttentionLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS GroupNormLayer : public Layer {
    public:
        static Ptr<GroupNormLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS CastLayer : public Layer {
    public:
        static Ptr<CastLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS DepthToSpaceLayer : public Layer {
    public:
        static Ptr<DepthToSpaceLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS SpaceToDepthLayer : public Layer {
    public:
        static Ptr<SpaceToDepthLayer> create(const LayerParams &params);
    };

    class CV_EXPORTS TopKLayer : public Layer
    {
    public:
        static Ptr<TopKLayer> create(const LayerParams& params);
    };

    class CV_EXPORTS TopK2Layer : public Layer
    {
    public:
        static Ptr<TopK2Layer> create(const LayerParams& params);
    };

//! @}
//! @}
CV__DNN_INLINE_NS_END
}
}
#endif
