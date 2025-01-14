/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_CORE_ARRAY_HELPERS_HPP
#define OPENCV_CORE_ARRAY_HELPERS_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

namespace cv {
struct _ArrayOpsBase {
  virtual Mat getMat_(const _InputArray& self, int i = -1) const = 0;
  virtual UMat getUMat(const _InputArray& self, int i) const = 0;
  virtual std::vector<Mat> getMatVector(const _InputArray& self) const = 0;
  virtual std::vector<UMat> getUMatVector(const _InputArray& self) const = 0;
  virtual int dims(const _InputArray& self, int i) const = 0;
  virtual Size size(const _InputArray& self, int i) const = 0;
  virtual int sizend(const _InputArray& self, int* arraySize, int i) const = 0;
  virtual int type(const _InputArray& self, int i) const = 0;
  virtual std::size_t total(const _InputArray& self, int i) const = 0;
  virtual int isContinuous(const _InputArray& self, int i) const = 0;
  virtual int isSubmatrix(const _InputArray& self, int i) const = 0;
  virtual int empty(const _InputArray& self) const = 0;
  virtual int empty(const _InputArray& self, int i) const = 0;
  virtual std::size_t offset(const _InputArray& self, std::size_t i) const = 0;
  virtual std::size_t step(const _InputArray& self, std::size_t i) const = 0;

  virtual void create(const _OutputArray& arr,
                      int d,
                      const int* sizes,
                      int mtype,
                      int i,
                      bool allowTransposed,
                      _OutputArray::DepthMask fixedDepthMask) const = 0;
  virtual void release(const _OutputArray& self) const = 0;
  virtual Mat& getMatRef(const _OutputArray& self, int i) const = 0;
  virtual UMat& getUMatRef(const _OutputArray& self, int i) const = 0;
  virtual void assign(const _OutputArray& self, const std::vector<Mat>& other) const = 0;
  virtual void assign(const _OutputArray& self, const std::vector<UMat>& other) const = 0;
protected:
  ~_ArrayOpsBase() = default;
};

template<class T>
inline constexpr bool is_Mat = std::is_same_v<T, Mat>;

template<class T>
inline constexpr bool is_Mat<Mat_<T>> = true;

template<class T>
inline constexpr bool is_UMat = std::is_same_v<T, UMat>;

template<class T>
inline constexpr bool is_cuda_GpuMat = std::is_same_v<T, cuda::GpuMat>;

template<class T>
inline constexpr bool is_vector = false;

template<class T, class Alloc>
inline constexpr bool is_vector<std::vector<T, Alloc>> = true;

#ifdef HAVE_CUDA
inline constexpr bool have_cuda = true;
#else
inline constexpr bool have_cuda = false;
#endif // HAVE_CUDA

template<class T>
struct _ArrayOps final : _ArrayOpsBase {
  Mat getMat_(const _InputArray& self, const int i) const final
  {
    T& v = get(self.getObj());
    using value_type = typename T::value_type;
    [[maybe_unused]] const auto index = static_cast<std::size_t>(i);

    if constexpr (std::is_same_v<value_type, bool>) {
      CV_Assert(i < 0);
      constexpr int type = CV_8U;
      if (v.empty()) {
        return Mat();
      }

      const int n = static_cast<int>(v.size());
      Mat m(1, &n, type);
      std::copy(v.begin(), v.end(), m.data);
      return m;
    }
    else if constexpr (is_Mat<value_type>) {
      CV_Assert(0 <= i);
      CV_Assert(index < v.size());
      return v[index];
    }
    else if constexpr (is_UMat<value_type>) {
      CV_Assert(0 <= i);
      CV_Assert(index < v.size());
      return v[index].getMat(self.getFlags() & ACCESS_MASK);
    }
    else if constexpr (is_vector<value_type>) {
      const int type = self.type(i);
      CV_Assert(0 <= i);
      CV_Assert(index < v.size());
      auto& sub_v = v[i];
      const int v_size = static_cast<int>(sub_v.size());
      return sub_v.empty() ? Mat() : Mat(1, &v_size, type, sub_v.data());
    }
    else {
      CV_Assert(i < 0);
      const int type = CV_MAT_TYPE(self.getFlags());
      const int width = static_cast<int>(v.size());
      return v.empty() ? Mat() : Mat(1, &width, type, v.data());
    }
  }

  UMat getUMat(const _InputArray& self, const int i) const final {
    using value_type = typename T::value_type;

    if constexpr (is_UMat<value_type>) {
      T& v = get(self.getObj());
      CV_Assert(0 <= i);
      CV_Assert(static_cast<std::size_t>(i) < v.size());
      return v[i];
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  std::vector<Mat> getMatVector(const _InputArray& self) const final
  {
    using value_type = typename T::value_type;
    auto& v = get(self.getObj());
    if constexpr (is_Mat<value_type>) {
      return {v.begin(), v.end()};
    }
    else if constexpr (!is_cuda_GpuMat<value_type>) {
      const int flags = self.getFlags();
      [[maybe_unused]] const int type = CV_MAT_TYPE(flags);
      [[maybe_unused]] const int column_number = CV_MAT_CN(flags);

      std::vector<Mat> result;
      result.reserve(v.size());

      for (std::size_t i = 0; i < v.size(); ++i) {
        if constexpr (is_UMat<value_type>) {
          result.emplace_back(v[i].getMat(flags & ACCESS_MASK));
        }
        else if constexpr (is_vector<value_type>) {
          result.emplace_back(size(self, i), type, static_cast<void*>(v[i].data()));
        }
        else if constexpr (!std::is_same_v<value_type, bool>) {
          result.emplace_back(1, column_number, type, static_cast<void*>(&v[i]));
        }
      }

      return result;
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  std::vector<UMat> getUMatVector(const _InputArray& self) const final
  {
    using value_type = typename T::value_type;
    auto& v = get(self.getObj());
    [[maybe_unused]] const int flags = self.getFlags();

    if constexpr (is_UMat<value_type>) {
      return v;
    }
    else if constexpr (is_Mat<value_type>) {
      std::vector<UMat> result;
      result.reserve(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        result.emplace_back(v[i].getUMat(flags & ACCESS_MASK));
      }

      return result;
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  int dims(const _InputArray& self, const int i) const final
  {
    using value_type = typename T::value_type;
    T& v = get(self.getObj());
    if constexpr (is_vector<value_type> || is_Mat<value_type> || is_UMat<value_type>) {
      if (i < 0) {
        return 1;
      }

      if constexpr (is_vector<value_type>) {
        CV_Assert(static_cast<std::size_t>(i) < v.size());
        return 2;
      }
      else {
        return v[i].dims;
      }
    }
    else {
      CV_Assert(i < 0);
      return 1;
    }
  }

  Size size(const _InputArray& self, const int i) const final {
    using value_type = typename T::value_type;
    auto& v = get(self.getObj());

    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      if (i < 0) {
        return v.empty() ? Size() : Size(static_cast<int>(v.size()), 1);
      }

      const auto index = static_cast<std::size_t>(i);
      CV_Assert(index < v.size());
      return v[index].size();
    }
    else if constexpr (is_vector<value_type>) {
      if (i < 0) {
        return v.empty() ? Size() : Size(static_cast<int>(v.size()), 1);
      }

      const auto index = static_cast<std::size_t>(i);
      CV_Assert(index < v.size());
      return Size(static_cast<int>(v[index].size()), 1);
    }
    else {
      CV_Assert(i < 0);
      return Size(static_cast<int>(v.size()), 1);
    }
  }

  int sizend(const _InputArray& self, int* const arraySize, const int i) const final
  {
    using value_type = typename T::value_type;
    T& v = get(self.getObj());
    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      CV_Assert(i >= 0);
      CV_Assert(static_cast<std::size_t>(i) < v.size());

      const auto& m = v[i];
      const int result = m.dims;
      if (arraySize != nullptr) {
        for (int j = 0; j < result; ++j) {
          arraySize[j] = m.size.p[j];
        }
      }

      return result;
    }
    else {
      CV_Assert(i < 0);
      Size sz2d = Size(v.size(), 1);
      if (arraySize != nullptr) {
        arraySize[0] = sz2d.width;
      }

      return 1;
    }
  }

  int type(const _InputArray& self, const int i) const final
  {
    using value_type = typename T::value_type;
    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      T& v = get(self.getObj());
      if (v.empty()) {
        const int flags = self.getFlags();
        CV_Assert((flags & _InputArray::FIXED_TYPE) != 0);
        return CV_MAT_TYPE(flags);
      }

      CV_Assert(i < static_cast<int>(v.size()));
      return v[i >= 0 ? i : 0].type();
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  std::size_t total(const _InputArray& self, const int i) const final {
    using value_type = typename T::value_type;
    T& v = get(self.getObj());
    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      if (i < 0) {
        return v.size();
      }

      CV_Assert(i < static_cast<int>(v.size()));
      return v[i].total();
    } else {
      CV_Assert(false && "unreachable");
    }
  }

  int isContinuous(const _InputArray& self, const int i) const final
  {
    auto& v = get(self.getObj());
    using value_type = typename T::value_type;
    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      CV_Assert(i >= 0);
      CV_Assert(i < static_cast<int>(v.size()));
      return v[i].isContinuous();
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  int isSubmatrix(const _InputArray& self, const int i) const final
  {
    using value_type = typename T::value_type;
    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      T& v = get(self.getObj());
      CV_Assert(i >= 0);
      CV_Assert(i < static_cast<int>(v.size()));
      return v[i].isSubmatrix();
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  int empty(const _InputArray& self) const final
  {
    return get(self.getObj()).empty();
  }

  int empty(const _InputArray& self, const int i) const final
  {
    using value_type = typename T::value_type;
    const T& v = get(self.getObj());
    if constexpr (is_Mat<value_type> || is_UMat<value_type> || is_vector<value_type>) {
      CV_Assert(i >= 0);
      CV_Assert(i < static_cast<int>(v.size()));
      return v[i].empty();
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  std::size_t offset(const _InputArray& self, const std::size_t i) const final
  {
    using value_type = typename T::value_type;
    const T& v = get(self.getObj());
    CV_Assert(i < v.size());

    if constexpr (is_Mat<value_type>) {
      return static_cast<std::size_t>(v[i].ptr() - v[i].datastart);
    }
    else if constexpr (is_UMat<value_type>) {
      return v[i].offset;
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  std::size_t step(const _InputArray& self, const std::size_t i) const final
  {
    using value_type = typename T::value_type;
    const auto& v = get(self.getObj());
    CV_Assert(i < v.size());

    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      return v[i].step;
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  void create(const _OutputArray& arr,
              const int d,
              const int* const sizes,
              int mtype,
              const int i,
              const bool allowTransposed,
              const _OutputArray::DepthMask fixedDepthMask) const final
  {
    using value_type = typename T::value_type;
    auto& v = get(arr.getObj());

    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      if (i < 0) {
        CV_Assert(d == 2);
        CV_Assert(sizes[0] == 1 || sizes[1] == 1 || sizes[0] * sizes[1] == 0);
        const std::size_t len = sizes[0] * sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0;
        const std::size_t len0 = v.size();

        CV_Assert(!arr.fixedSize() || len == len0);
        v.resize(len);
        if (arr.fixedType()) {
          const int _type = CV_MAT_TYPE(arr.getFlags());
          for (std::size_t j = len0; j < len; ++j) {
            if (v[j].type() == _type) {
              continue;
            }

            CV_Assert(v[j].empty());
            v[j].flags = (v[j].flags & ~CV_MAT_TYPE_MASK) | _type;
          }
        }

        return;
      }

      CV_Assert(i < static_cast<int>(v.size()));
      auto& m = mattify(v[i]);

      if (allowTransposed) {
        if (!m.isContinuous()) {
          CV_Assert(!arr.fixedType() && !arr.fixedSize());
          m.release();
        }

        const bool same_type = m.type() == mtype;
        const bool same_dimensions = d == 2 && m.dims == 2 && m.rows == sizes[1] && m.cols == sizes[0];
        if (get_data(m) != nullptr && same_type && same_dimensions) {
          return;
        }
      }

      if (arr.fixedType()) {
        const bool same_channels = CV_MAT_CN(mtype) == m.channels();
        const bool has_depth = ((1 << CV_MAT_TYPE(arr.getFlags())) & fixedDepthMask) != 0;
        if (same_channels && has_depth) {
          mtype = m.type();
        }
        else {
          CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }
      }

      if (arr.fixedSize()) {
        CV_Assert(m.dims == d);
        for (int j = 0; j < d; ++j) {
          CV_Assert(m.size[j] == sizes[j]);
        }
      }

      m.create(d, sizes, mtype);
      return;
    }
    else if constexpr (!std::is_same_v<bool, value_type>) {
      const int size0 = d > 0 ? sizes[0] : 1;
      const int size1 = d > 1 ? sizes[1] : 1;
      CV_Assert(d <= 2);
      CV_Assert(size0 == 1 || size1 == 1 || size0 * size1 == 0);

      const std::size_t len = size0 * size1 > 0 ? size0 + size1 - 1 : 0;

      if constexpr (is_vector<value_type>) {
        if (i < 0) {
          CV_Assert(!arr.fixedSize() || len == v.size());
          v.resize(len);
          return;
        }

        CV_Assert(i < static_cast<int>(v.size()));
        v[i].resize(len);
      }
      else {
        CV_Assert(i < 0);
        const int type0 = CV_MAT_TYPE(arr.getFlags());
        CV_Assert(mtype == type0 || (CV_MAT_CN(mtype) == CV_MAT_CN(type0) && ((1 << type0) & fixedDepthMask) != 0));
        v.resize(len);
      }
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  void release(const _OutputArray& self) const override
  {
    get(self.getObj()).clear();
  }

  Mat& getMatRef(const _OutputArray& self, const int i) const final
  {
    if constexpr (is_Mat<typename T::value_type>) {
      CV_Assert(i >= 0);
      T& v = get(self.getObj());
      CV_Assert(i < static_cast<int>(v.size()));
      return v[i];
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  UMat& getUMatRef(const _OutputArray& self, const int i) const final
  {
    if constexpr (is_UMat<typename T::value_type>) {
      CV_Assert(i >= 0);
      T& v = get(self.getObj());
      CV_Assert(i < static_cast<int>(v.size()));
      return v[i];
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  void assign(const _OutputArray& self, const std::vector<Mat>& other) const final
  {
    return assign_impl(self, other);
  }

  void assign(const _OutputArray& self, const std::vector<UMat>& other) const final
  {
    return assign_impl(self, other);
  }

  static const T& get(const void* const data)
  {
    return *static_cast<const T*>(data);
  }

  static T& get(void* const data)
  {
    return *static_cast<T*>(data);
  }

  template<class U>
  static auto get_data(U& m)
  {
    if constexpr (is_Mat<U>) {
      return m.data;
    }
    else if constexpr (is_UMat<U>) {
      return m.u;
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  template<class U>
  static auto& mattify(U& m)
  {
    if constexpr (is_Mat<U>) {
      return static_cast<Mat&>(m);
    }
    else if constexpr (is_UMat<U>) {
      return m;
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }

  template<class U>
  static void assign_impl(const _OutputArray& self, const std::vector<U>& other)
  {
    using value_type = typename T::value_type;
    if constexpr (is_Mat<value_type> || is_UMat<value_type>) {
      auto& this_v = get(self.getObj());
      CV_Assert(this_v.size() == other.size());

      for (std::size_t i = 0; i < other.size(); ++i) {
        const auto& m = other[i];
        auto& this_m = this_v[i];
        if (this_m.u != nullptr && this_m.u == m.u) {
          continue;
        }

        m.copyTo(this_m);
      }
    }
    else {
      CV_Assert(false && "unreachable");
    }
  }
};

template<>
Mat _ArrayOps<std::vector<cuda::GpuMat>>::getMat_(const _InputArray& self, const int i) const;

template<>
Size _ArrayOps<std::vector<cuda::GpuMat>>::size(const _InputArray& self, const int i) const;

template<>
int _ArrayOps<std::vector<cuda::GpuMat>>::sizend(const _InputArray& self, int* const arraySize, const int i) const;

template<>
std::size_t _ArrayOps<std::vector<cuda::GpuMat>>::offset(const _InputArray& self, const std::size_t i) const;

template<>
std::size_t _ArrayOps<std::vector<cuda::GpuMat>>::step(const _InputArray& self, const std::size_t i) const;

template<>
void _ArrayOps<std::vector<cuda::GpuMat>>::create(const _OutputArray& arr,
                                                  int d,
                                                  const int* sizes,
                                                  int mtype,
                                                  int i,
                                                  bool allowTransposed,
                                                  _OutputArray::DepthMask fixedDepthMask) const;

template<>
void _ArrayOps<std::vector<cuda::GpuMat>>::release(const _OutputArray& self) const;

template<class T>
inline constexpr _ArrayOps<T> array_ops;

template<class T>
_InputArray::_InputArray(const int _flags, T* const _obj)
: _InputArray(_flags, _obj, {})
{}

template<class T>
_InputArray::_InputArray(const int _flags, T* const _obj, const Size _sz)
: flags(_flags)
, obj(_obj)
, sz(_sz)
{
  if constexpr (is_vector<T>) {
    ops = &array_ops<T>;
  }
}

template<class T>
_InputArray::_InputArray(const int _flags, const T* const _obj, const Size _sz)
: flags(_flags)
, obj(const_cast<T*>(_obj))
, sz(_sz)
{
  if constexpr (is_vector<T>) {
    ops = &array_ops<T>;
  }
}
} // namespace cv

#pragma GCC diagnostic pop

#endif // OPENCV_CORE_ARRAY_HELPERS_HPP
