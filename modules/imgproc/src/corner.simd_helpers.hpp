/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of OpenCV project.
// See opencv/LICENSE for the actual licensing terms.
// Contributed by Giles Payne
//
//M*/

#ifndef OPENCV_CORNER_SIMD_HELPER_HPP
#define OPENCV_CORNER_SIMD_HELPER_HPP

namespace cv {

template<class T> T muladd(const T& t1, const T& t2, const T& t3) { return (t1 * t2) + t3; }

template<class T, T(*load)(const float*), void(*store)(float*, const T&), T(*setall)(float), T(*sqrt)(const T&), T(*muladd)(const T&, const T&, const T&) = muladd> void calcMinEigenValLine(int& j, int width, int lanes, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst) {
    T half = setall(0.5f);
    for (; j <= width - lanes; j += lanes) {
        T a = load(cov_x2 + j) * half;
        T b = load(cov_xy + j);
        T c = load(cov_y2 + j) * half;
        T t = a - c;
        t = muladd(b, b , (t * t));
        store(dst + j, (a + c) - sqrt(t));
    }
}

template<class T, T(*load)(const float*), void(*store)(float*, const T&), T(*setall)(float)> void calcHarrisLine(int& j, int width, int lanes, const float* cov_x2, const float* cov_xy, const float* cov_y2, float* dst, double k) {
    T tk = setall((float)k);
    for (; j <= width - lanes; j += lanes) {
        T a = load(cov_x2 + j);
        T b = load(cov_xy + j);
        T c = load(cov_y2 + j);
        store(dst + j, a*c - b*b - tk*(a + c)*(a + c));
    }
}

template<class T, T(*load)(const float*), void(*store)(float*, const T&)> void cornerEigenValsVecsLine(int& j, int width, int lanes, const float* dxdata, const float* dydata, float* cov_x2, float* cov_xy, float* cov_y2) {
    for (; j <= width - lanes; j += lanes) {
        T dx = load(dxdata + j);
        T dy = load(dydata + j);

        store(cov_x2 + j, dx * dx);
        store(cov_xy + j, dx * dy);
        store(cov_y2 + j, dy * dy);
    }
}

} // namespace

#endif
