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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

using namespace cv::cuda;

typedef unsigned char uchar;
typedef unsigned short ushort;

//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        __device__ __forceinline__ float norm2(const float& v) { return v*v; }
        __device__ __forceinline__ float norm2(const float2& v) { return v.x*v.x + v.y*v.y; }
        __device__ __forceinline__ float norm2(const float3& v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
        __device__ __forceinline__ float norm2(const float4& v) { return v.x*v.x + v.y*v.y + v.z*v.z  + v.w*v.w; }

        template<typename T, typename B>
        __global__ void nlm_kernel(const PtrStep<T> src, PtrStepSz<T> dst, const B b, int search_radius, int block_radius, float noise_mult)
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type value_type;

            const int i = blockDim.y * blockIdx.y + threadIdx.y;
            const int j = blockDim.x * blockIdx.x + threadIdx.x;

            if (j >= dst.cols || i >= dst.rows)
                return;

            int bsize = search_radius + block_radius;
            int search_window = 2 * search_radius + 1;
            float minus_search_window2_inv = -1.f/(search_window * search_window);

            value_type sum1 = VecTraits<value_type>::all(0);
            float sum2 = 0.f;

            if (j - bsize >= 0 && j + bsize < dst.cols && i - bsize >= 0 && i + bsize < dst.rows)
            {
                for(float y = -search_radius; y <= search_radius; ++y)
                    for(float x = -search_radius; x <= search_radius; ++x)
                    {
                        float dist2 = 0;
                        for(float ty = -block_radius; ty <= block_radius; ++ty)
                            for(float tx = -block_radius; tx <= block_radius; ++tx)
                            {
                                value_type bv = saturate_cast<value_type>(src(i + y + ty, j + x + tx));
                                value_type av = saturate_cast<value_type>(src(i +     ty, j +     tx));

                                dist2 += norm2(av - bv);
                            }

                        float w = __expf(dist2 * noise_mult + (x * x + y * y) * minus_search_window2_inv);

                        /*if (i == 255 && j == 255)
                            printf("%f %f\n", w, dist2 * minus_h2_inv + (x * x + y * y) * minus_search_window2_inv);*/

                        sum1 = sum1 + w * saturate_cast<value_type>(src(i + y, j + x));
                        sum2 += w;
                    }
            }
            else
            {
                for(float y = -search_radius; y <= search_radius; ++y)
                    for(float x = -search_radius; x <= search_radius; ++x)
                    {
                        float dist2 = 0;
                        for(float ty = -block_radius; ty <= block_radius; ++ty)
                            for(float tx = -block_radius; tx <= block_radius; ++tx)
                            {
                                value_type bv = saturate_cast<value_type>(b.at(i + y + ty, j + x + tx, src));
                                value_type av = saturate_cast<value_type>(b.at(i +     ty, j +     tx, src));
                                dist2 += norm2(av - bv);
                            }

                        float w = __expf(dist2 * noise_mult + (x * x + y * y) * minus_search_window2_inv);

                        sum1 = sum1 + w * saturate_cast<value_type>(b.at(i + y, j + x, src));
                        sum2 += w;
                    }

            }

            dst(i, j) = saturate_cast<T>(sum1 / sum2);

        }

        template<typename T, template <typename> class B>
        void nlm_caller(const PtrStepSzb src, PtrStepSzb dst, int search_radius, int block_radius, float h, cudaStream_t stream)
        {
            dim3 block (32, 8);
            dim3 grid (divUp (src.cols, block.x), divUp (src.rows, block.y));

            B<T> b(src.rows, src.cols);

            int block_window = 2 * block_radius + 1;
            float minus_h2_inv = -1.f/(h * h * VecTraits<T>::cn);
            float noise_mult = minus_h2_inv/(block_window * block_window);

            cudaSafeCall( cudaFuncSetCacheConfig (nlm_kernel<T, B<T> >, cudaFuncCachePreferL1) );
            nlm_kernel<<<grid, block>>>((PtrStepSz<T>)src, (PtrStepSz<T>)dst, b, search_radius, block_radius, noise_mult);
            cudaSafeCall ( cudaGetLastError () );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template<typename T>
        void nlm_bruteforce_gpu(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream)
        {
            typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb dst, int search_radius, int block_radius, float h, cudaStream_t stream);

            static func_t funcs[] =
            {
                nlm_caller<T, BrdConstant>,
                nlm_caller<T, BrdReplicate>,
                nlm_caller<T, BrdReflect>,
                nlm_caller<T, BrdWrap>,
                nlm_caller<T, BrdReflect101>
            };
            funcs[borderMode](src, dst, search_radius, block_radius, h, stream);
        }

        template void nlm_bruteforce_gpu<uchar>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
        template void nlm_bruteforce_gpu<uchar2>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
        template void nlm_bruteforce_gpu<uchar3>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);

        template void nlm_bruteforce_gpu<ushort>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
    }
}}}

//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing (fast approximate version)

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {

        template <int cn> struct Unroll;
        template <> struct Unroll<1>
        {
            template <int BLOCK_SIZE>
            static __device__ __forceinline__ thrust::tuple<volatile float*, volatile float*> smem_tuple(float* smem)
            {
                return cv::cuda::device::smem_tuple(smem, smem + BLOCK_SIZE);
            }

            static __device__ __forceinline__ thrust::tuple<float&, float&> tie(float& val1, float& val2)
            {
                return thrust::tie(val1, val2);
            }

            static __device__ __forceinline__ const thrust::tuple<plus<float>, plus<float> > op()
            {
                plus<float> op;
                return thrust::make_tuple(op, op);
            }
        };
        template <> struct Unroll<2>
        {
            template <int BLOCK_SIZE>
            static __device__ __forceinline__ thrust::tuple<volatile float*, volatile float*, volatile float*> smem_tuple(float* smem)
            {
                return cv::cuda::device::smem_tuple(smem, smem + BLOCK_SIZE, smem + 2 * BLOCK_SIZE);
            }

            static __device__ __forceinline__ thrust::tuple<float&, float&, float&> tie(float& val1, float2& val2)
            {
                return thrust::tie(val1, val2.x, val2.y);
            }

            static __device__ __forceinline__ const thrust::tuple<plus<float>, plus<float>, plus<float> > op()
            {
                plus<float> op;
                return thrust::make_tuple(op, op, op);
            }
        };
        template <> struct Unroll<3>
        {
            template <int BLOCK_SIZE>
            static __device__ __forceinline__ thrust::tuple<volatile float*, volatile float*, volatile float*, volatile float*> smem_tuple(float* smem)
            {
                return cv::cuda::device::smem_tuple(smem, smem + BLOCK_SIZE, smem + 2 * BLOCK_SIZE, smem + 3 * BLOCK_SIZE);
            }

            static __device__ __forceinline__ thrust::tuple<float&, float&, float&, float&> tie(float& val1, float3& val2)
            {
                return thrust::tie(val1, val2.x, val2.y, val2.z);
            }

            static __device__ __forceinline__ const thrust::tuple<plus<float>, plus<float>, plus<float>, plus<float> > op()
            {
                plus<float> op;
                return thrust::make_tuple(op, op, op, op);
            }
        };
        template <> struct Unroll<4>
        {
            template <int BLOCK_SIZE>
            static __device__ __forceinline__ thrust::tuple<volatile float*, volatile float*, volatile float*, volatile float*, volatile float*> smem_tuple(float* smem)
            {
                return cv::cuda::device::smem_tuple(smem, smem + BLOCK_SIZE, smem + 2 * BLOCK_SIZE, smem + 3 * BLOCK_SIZE, smem + 4 * BLOCK_SIZE);
            }

            static __device__ __forceinline__ thrust::tuple<float&, float&, float&, float&, float&> tie(float& val1, float4& val2)
            {
                return thrust::tie(val1, val2.x, val2.y, val2.z, val2.w);
            }

            static __device__ __forceinline__ const thrust::tuple<plus<float>, plus<float>, plus<float>, plus<float>, plus<float> > op()
            {
                plus<float> op;
                return thrust::make_tuple(op, op, op, op, op);
            }
        };

        __device__ __forceinline__ int calcDist(const uchar&  a, const uchar&  b) { return (a-b)*(a-b); }
        __device__ __forceinline__ int calcDist(const ushort& a, const ushort& b) { return ::abs(a - b);}
        __device__ __forceinline__ int calcDist(const uchar2& a, const uchar2& b) { return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y); }
        __device__ __forceinline__ int calcDist(const uchar3& a, const uchar3& b) { return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z); }

        template <class T> struct FastNonLocalMeans
        {
            enum
            {
                CTA_SIZE = 128,

                TILE_COLS = 128,
                TILE_ROWS = 32,

                STRIDE = CTA_SIZE
            };

            struct plus
            {
                __device__ __forceinline__ float operator()(float v1, float v2) const { return v1 + v2; }
            };

            int search_radius;
            int block_radius;

            int search_window;
            int block_window;
            float minus_h2_inv;

            FastNonLocalMeans(int search_window_, int block_window_, float h) : search_radius(search_window_/2), block_radius(block_window_/2),
                search_window(search_window_), block_window(block_window_), minus_h2_inv(-1.f/(h * h * VecTraits<T>::cn)) {}

            PtrStep<T> src;
            mutable PtrStepi buffer;

            __device__ __forceinline__ void initSums_BruteForce(int i, int j, int* dist_sums, PtrStepi& col_sums, PtrStepi& up_col_sums) const
            {
                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    dist_sums[index] = 0;

                    for(int tx = 0; tx < block_window; ++tx)
                        col_sums(tx, index) = 0;

                    int y = index / search_window;
                    int x = index - y * search_window;

                    int ay = i;
                    int ax = j;

                    int by = i + y - search_radius;
                    int bx = j + x - search_radius;

#if 1
                    for (int tx = -block_radius; tx <= block_radius; ++tx)
                    {
                        int col_sum = 0;
                        for (int ty = -block_radius; ty <= block_radius; ++ty)
                        {
                            int dist = calcDist(src(ay + ty, ax + tx), src(by + ty, bx + tx));

                            dist_sums[index] += dist;
                            col_sum += dist;
                        }
                        col_sums(tx + block_radius, index) = col_sum;
                    }
#else
                    for (int ty = -block_radius; ty <= block_radius; ++ty)
                        for (int tx = -block_radius; tx <= block_radius; ++tx)
                        {
                            int dist = calcDist(src(ay + ty, ax + tx), src(by + ty, bx + tx));

                            dist_sums[index] += dist;
                            col_sums(tx + block_radius, index) += dist;
                        }
#endif

                    up_col_sums(j, index) = col_sums(block_window - 1, index);
                }
            }

            __device__ __forceinline__ void shiftRight_FirstRow(int i, int j, int first, int* dist_sums, PtrStepi& col_sums, PtrStepi& up_col_sums) const
            {
                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    int y = index / search_window;
                    int x = index - y * search_window;

                    int ay = i;
                    int ax = j + block_radius;

                    int by = i + y - search_radius;
                    int bx = j + x - search_radius + block_radius;

                    int col_sum = 0;

                    for (int ty = -block_radius; ty <= block_radius; ++ty)
                        col_sum += calcDist(src(ay + ty, ax), src(by + ty, bx));

                    dist_sums[index] += col_sum - col_sums(first, index);

                    col_sums(first, index) = col_sum;
                    up_col_sums(j, index) = col_sum;
                }
            }

            __device__ __forceinline__ void shiftRight_UpSums(int i, int j, int first, int* dist_sums, PtrStepi& col_sums, PtrStepi& up_col_sums) const
            {
                int ay = i;
                int ax = j + block_radius;

                T a_up   = src(ay - block_radius - 1, ax);
                T a_down = src(ay + block_radius, ax);

                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    int y = index / search_window;
                    int x = index - y * search_window;

                    int by = i + y - search_radius;
                    int bx = j + x - search_radius + block_radius;

                    T b_up   = src(by - block_radius - 1, bx);
                    T b_down = src(by + block_radius, bx);

                    int col_sum = up_col_sums(j, index) + calcDist(a_down, b_down) - calcDist(a_up, b_up);

                    dist_sums[index] += col_sum  - col_sums(first, index);
                    col_sums(first, index) = col_sum;
                    up_col_sums(j, index) = col_sum;
                }
            }

            __device__ __forceinline__ void convolve_window(int i, int j, const int* dist_sums, T& dst) const
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_type;

                float weights_sum = 0;
                sum_type sum = VecTraits<sum_type>::all(0);

                float bw2_inv = 1.f/(block_window * block_window);

                int sx = j - search_radius;
                int sy = i - search_radius;

                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    int y = index / search_window;
                    int x = index - y * search_window;

                    float avg_dist = dist_sums[index] * bw2_inv;
                    float weight = __expf(avg_dist * minus_h2_inv);
                    weights_sum += weight;

                    sum = sum + weight * saturate_cast<sum_type>(src(sy + y, sx + x));
                }

                __shared__ float cta_buffer[CTA_SIZE * (VecTraits<T>::cn + 1)];

                reduce<CTA_SIZE>(Unroll<VecTraits<T>::cn>::template smem_tuple<CTA_SIZE>(cta_buffer),
                                 Unroll<VecTraits<T>::cn>::tie(weights_sum, sum),
                                 threadIdx.x,
                                 Unroll<VecTraits<T>::cn>::op());

                if (threadIdx.x == 0)
                    dst = saturate_cast<T>(sum / weights_sum);
            }

            __device__ __forceinline__ void operator()(PtrStepSz<T>& dst) const
            {
                int tbx = blockIdx.x * TILE_COLS;
                int tby = blockIdx.y * TILE_ROWS;

                int tex = ::min(tbx + TILE_COLS, dst.cols);
                int tey = ::min(tby + TILE_ROWS, dst.rows);

                PtrStepi col_sums;
                col_sums.data = buffer.ptr(dst.cols + blockIdx.x * block_window) + blockIdx.y * search_window * search_window;
                col_sums.step = buffer.step;

                PtrStepi up_col_sums;
                up_col_sums.data = buffer.data + blockIdx.y * search_window * search_window;
                up_col_sums.step = buffer.step;

                extern __shared__ int dist_sums[]; //search_window * search_window

                int first = 0;

                for (int i = tby; i < tey; ++i)
                    for (int j = tbx; j < tex; ++j)
                    {
                        __syncthreads();

                        if (j == tbx)
                        {
                            initSums_BruteForce(i, j, dist_sums, col_sums, up_col_sums);
                            first = 0;
                        }
                        else
                        {
                            if (i == tby)
                              shiftRight_FirstRow(i, j, first, dist_sums, col_sums, up_col_sums);
                            else
                              shiftRight_UpSums(i, j, first, dist_sums, col_sums, up_col_sums);

                            first = (first + 1) % block_window;
                        }

                        __syncthreads();

                        convolve_window(i, j, dist_sums, dst(i, j));
                    }
            }

        };

        template<typename T>
        __global__ void fast_nlm_kernel(const FastNonLocalMeans<T> fnlm, PtrStepSz<T> dst) { fnlm(dst); }

        void nln_fast_get_buffer_size(const PtrStepSzb& src, int search_window, int block_window, int& buffer_cols, int& buffer_rows)
        {
            typedef FastNonLocalMeans<uchar> FNLM;
            dim3 grid(divUp(src.cols, FNLM::TILE_COLS), divUp(src.rows, FNLM::TILE_ROWS));

            buffer_cols = search_window * search_window * grid.y;
            buffer_rows = src.cols + block_window * grid.x;
        }

        template<typename T>
        void nlm_fast_gpu(const PtrStepSzb& src, PtrStepSzb dst, PtrStepi buffer,
                          int search_window, int block_window, float h, cudaStream_t stream)
        {
            typedef FastNonLocalMeans<T> FNLM;
            FNLM fnlm(search_window, block_window, h);

            fnlm.src = (PtrStepSz<T>)src;
            fnlm.buffer = buffer;

            dim3 block(FNLM::CTA_SIZE, 1);
            dim3 grid(divUp(src.cols, FNLM::TILE_COLS), divUp(src.rows, FNLM::TILE_ROWS));
            int smem = search_window * search_window * sizeof(int);


            fast_nlm_kernel<<<grid, block, smem>>>(fnlm, (PtrStepSz<T>)dst);
            cudaSafeCall ( cudaGetLastError () );
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void nlm_fast_gpu<uchar>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float,  cudaStream_t);
        template void nlm_fast_gpu<uchar2>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);
        template void nlm_fast_gpu<uchar3>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);
        template void nlm_fast_gpu<ushort>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);

        __global__ void fnlm_split_kernel(const PtrStepSz<uchar3> lab, PtrStepb l, PtrStep<uchar2> ab)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < lab.cols && y < lab.rows)
            {
                uchar3 p = lab(y, x);
                ab(y,x) = make_uchar2(p.y, p.z);
                l(y,x) = p.x;
            }
        }

        void fnlm_split_channels(const PtrStepSz<uchar3>& lab, PtrStepb l, PtrStep<uchar2> ab, cudaStream_t stream)
        {
            dim3 b(32, 8);
            dim3 g(divUp(lab.cols, b.x), divUp(lab.rows, b.y));

            fnlm_split_kernel<<<g, b>>>(lab, l, ab);
            cudaSafeCall ( cudaGetLastError () );
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void fnlm_merge_kernel(const PtrStepb l, const PtrStep<uchar2> ab, PtrStepSz<uchar3> lab)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < lab.cols && y < lab.rows)
            {
                uchar2 p = ab(y, x);
                lab(y, x) = make_uchar3(l(y, x), p.x, p.y);
            }
        }

        void fnlm_merge_channels(const PtrStepb& l, const PtrStep<uchar2>& ab, PtrStepSz<uchar3> lab, cudaStream_t stream)
        {
            dim3 b(32, 8);
            dim3 g(divUp(lab.cols, b.x), divUp(lab.rows, b.y));

            fnlm_merge_kernel<<<g, b>>>(l, ab, lab);
            cudaSafeCall ( cudaGetLastError () );
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}
