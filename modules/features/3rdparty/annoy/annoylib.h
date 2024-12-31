// Copyright (c) 2013 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.


#ifndef ANNOY_ANNOYLIB_H
#define ANNOY_ANNOYLIB_H

#include <stdio.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stddef.h>

#if defined(_MSC_VER) && _MSC_VER == 1500
typedef unsigned char     uint8_t;
typedef signed __int32    int32_t;
typedef unsigned __int64  uint64_t;
typedef signed __int64    int64_t;
#else
#include <stdint.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
 // a bit hacky, but override some definitions to support 64 bit
 #define off_t int64_t
 #define lseek_getsize(fd) _lseeki64(fd, 0, SEEK_END)
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include "mman.h"
 #include <windows.h>
#else
 #include <sys/mman.h>
 #define lseek_getsize(fd) lseek(fd, 0, SEEK_END)
#endif

#include <cerrno>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
#include <map>

#if __cplusplus >= 201103L
#include <type_traits>
#endif

#ifdef ANNOYLIB_MULTITHREADED_BUILD
#include <thread>
#include <mutex>
#include <shared_mutex>
#endif

#ifdef _MSC_VER
// Needed for Visual Studio to disable runtime checks for mempcy
#pragma runtime_checks("s", off)
#endif

// This allows others to supply their own logger / error printer without
// requiring Annoy to import their headers. See RcppAnnoy for a use case.
#ifndef __ERROR_PRINTER_OVERRIDE__
  #define annoylib_showUpdate(...) { fprintf(stderr, __VA_ARGS__ ); }
#else
  #define annoylib_showUpdate(...) { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); }
#endif

// Portable alloc definition, cf Writing R Extensions, Section 1.6.4
#ifdef __GNUC__
  // Includes GCC, clang and Intel compilers
  # undef alloca
  # define alloca(x) __builtin_alloca((x))
#elif defined(__sun) || defined(_AIX)
  // this is necessary (and sufficient) for Solaris 10 and AIX 6:
  # include <alloca.h>
#endif

// We let the v array in the Node struct take whatever space is needed, so this is a mostly insignificant number.
// Compilers need *some* size defined for the v array, and some memory checking tools will flag for buffer overruns if this is set too low.
#define ANNOYLIB_V_ARRAY_SIZE 65536

#ifndef _MSC_VER
#define annoylib_popcount __builtin_popcountll
#else // See #293, #358
#define annoylib_popcount cole_popcount
#endif

#if !defined(NO_MANUAL_VECTORIZATION) && defined(__GNUC__) && (__GNUC__ >6) && defined(__AVX512F__)  // See #402
#define ANNOYLIB_USE_AVX512
#elif !defined(NO_MANUAL_VECTORIZATION) && defined(__AVX__) && defined (__SSE__) && defined(__SSE2__) && defined(__SSE3__)
#define ANNOYLIB_USE_AVX
#else
#endif

#if defined(ANNOYLIB_USE_AVX) || defined(ANNOYLIB_USE_AVX512)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <x86intrin.h>
#endif
#endif

#if !defined(__MINGW32__)
#define ANNOYLIB_FTRUNCATE_SIZE(x) static_cast<int64_t>(x)
#else
#define ANNOYLIB_FTRUNCATE_SIZE(x) (x)
#endif

namespace cvannoy {

inline void set_error_from_errno(char **error, const char* msg) {
  annoylib_showUpdate("%s: %s (%d)\n", msg, strerror(errno), errno);
  if (error) {
    *error = (char *)malloc(256);  // TODO: win doesn't support snprintf
    snprintf(*error, 255, "%s: %s (%d)", msg, strerror(errno), errno);
  }
}

inline void set_error_from_string(char **error, const char* msg) {
  annoylib_showUpdate("%s\n", msg);
  if (error) {
    *error = (char *)malloc(strlen(msg) + 1);
    strcpy(*error, msg);
  }
}


using std::vector;
using std::pair;
using std::numeric_limits;
using std::make_pair;
using std::map;

inline bool remap_memory_and_truncate(void** _ptr, int _fd, size_t old_size, size_t new_size) {
#ifdef __linux__
    *_ptr = mremap(*_ptr, old_size, new_size, MREMAP_MAYMOVE);
    bool ok = ftruncate(_fd, new_size) != -1;
#else
    munmap(*_ptr, old_size);
    bool ok = ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(new_size)) != -1;
#ifdef MAP_POPULATE
    *_ptr = mmap(*_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    *_ptr = mmap(*_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
#endif
    return ok;
}

namespace {

template<typename S, typename Node>
inline Node* get_node_ptr(const void* _nodes, const size_t _s, const S i) {
  return (Node*)((uint8_t *)_nodes + (_s * i));
}

template<typename T>
inline T dot(const T* x, const T* y, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    s += (*x) * (*y);
    x++;
    y++;
  }
  return s;
}

template<typename T>
inline T manhattan_distance(const T* x, const T* y, int f) {
  T d = 0.0;
  for (int i = 0; i < f; i++)
    d += fabs(x[i] - y[i]);
  return d;
}

template<typename T>
inline T euclidean_distance(const T* x, const T* y, int f) {
  // Don't use dot-product: avoid catastrophic cancellation in #314.
  T d = 0.0;
  for (int i = 0; i < f; ++i) {
    const T tmp=*x - *y;
    d += tmp * tmp;
    ++x;
    ++y;
  }
  return d;
}

#ifdef ANNOYLIB_USE_AVX
// Horizontal single sum of 256bit vector.
inline float hsum256_ps_avx(__m256 v) {
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  return _mm_cvtss_f32(x32);
}

template<>
inline float dot<float>(const float* x, const float *y, int f) {
  float result = 0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

template<>
inline float manhattan_distance<float>(const float* x, const float* y, int f) {
  float result = 0;
  int i = f;
  if (f > 7) {
    __m256 manhattan = _mm256_setzero_ps();
    __m256 minus_zero = _mm256_set1_ps(-0.0f);
    for (; i > 7; i -= 8) {
      const __m256 x_minus_y = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
      const __m256 distance = _mm256_andnot_ps(minus_zero, x_minus_y); // Absolute value of x_minus_y (forces sign bit to zero)
      manhattan = _mm256_add_ps(manhattan, distance);
      x += 8;
      y += 8;
    }
    // Sum all floats in manhattan register.
    result = hsum256_ps_avx(manhattan);
  }
  // Don't forget the remaining values.
  for (; i > 0; i--) {
    result += fabsf(*x - *y);
    x++;
    y++;
  }
  return result;
}

template<>
inline float euclidean_distance<float>(const float* x, const float* y, int f) {
  float result=0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
      d = _mm256_add_ps(d, _mm256_mul_ps(diff, diff)); // no support for fmadd in AVX...
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result = hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x - *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return result;
}

#endif

#ifdef ANNOYLIB_USE_AVX512
template<>
inline float dot<float>(const float* x, const float *y, int f) {
  float result = 0;
  if (f > 15) {
    __m512 d = _mm512_setzero_ps();
    for (; f > 15; f -= 16) {
      //AVX512F includes FMA
      d = _mm512_fmadd_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y), d);
      x += 16;
      y += 16;
    }
    // Sum all floats in dot register.
    result += _mm512_reduce_add_ps(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

template<>
inline float manhattan_distance<float>(const float* x, const float* y, int f) {
  float result = 0;
  int i = f;
  if (f > 15) {
    __m512 manhattan = _mm512_setzero_ps();
    for (; i > 15; i -= 16) {
      const __m512 x_minus_y = _mm512_sub_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y));
      manhattan = _mm512_add_ps(manhattan, _mm512_abs_ps(x_minus_y));
      x += 16;
      y += 16;
    }
    // Sum all floats in manhattan register.
    result = _mm512_reduce_add_ps(manhattan);
  }
  // Don't forget the remaining values.
  for (; i > 0; i--) {
    result += fabsf(*x - *y);
    x++;
    y++;
  }
  return result;
}

template<>
inline float euclidean_distance<float>(const float* x, const float* y, int f) {
  float result=0;
  if (f > 15) {
    __m512 d = _mm512_setzero_ps();
    for (; f > 15; f -= 16) {
      const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y));
      d = _mm512_fmadd_ps(diff, diff, d);
      x += 16;
      y += 16;
    }
    // Sum all floats in dot register.
    result = _mm512_reduce_add_ps(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x - *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return result;
}

#endif


template<typename T, typename Random, typename Distance, typename Node>
inline void two_means(const vector<Node*>& nodes, int f, Random& random, bool cosine, Node* p, Node* q) {
  /*
    This algorithm is a huge heuristic. Empirically it works really well, but I
    can't motivate it well. The basic idea is to keep two centroids and assign
    points to either one of them. We weight each centroid by the number of points
    assigned to it, so to balance it.
  */
  static int iteration_steps = 200;
  size_t count = nodes.size();

  size_t i = random.index(count);
  size_t j = random.index(count-1);
  j += (j >= i); // ensure that i != j

  Distance::template copy_node<T, Node>(p, nodes[i], f);
  Distance::template copy_node<T, Node>(q, nodes[j], f);

  if (cosine) { Distance::template normalize<T, Node>(p, f); Distance::template normalize<T, Node>(q, f); }
  Distance::init_node(p, f);
  Distance::init_node(q, f);

  int ic = 1, jc = 1;
  for (int l = 0; l < iteration_steps; l++) {
    size_t k = random.index(count);
    T di = ic * Distance::distance(p, nodes[k], f),
      dj = jc * Distance::distance(q, nodes[k], f);
    T norm = cosine ? Distance::template get_norm<T, Node>(nodes[k], f) : 1;
    if (!(norm > T(0))) {
      continue;
    }
    if (di < dj) {
      Distance::update_mean(p, nodes[k], norm, ic, f);
      Distance::init_node(p, f);
      ic++;
    } else if (dj < di) {
      Distance::update_mean(q, nodes[k], norm, jc, f);
      Distance::init_node(q, f);
      jc++;
    }
  }
}
} // namespace

struct Base {
  template<typename T, typename S, typename Node>
  static inline void preprocess(void* /*nodes*/, size_t /*_s*/, const S /*node_count*/, const int /*f*/) {
    // Override this in specific metric structs below if you need to do any pre-processing
    // on the entire set of nodes passed into this index.
  }

  template<typename T, typename S, typename Node>
  static inline void postprocess(void* /*nodes*/, size_t /*_s*/, const S /*node_count*/, const int /*f*/) {
    // Override this in specific metric structs below if you need to do any post-processing
    // on the entire set of nodes passed into this index.
  }

  template<typename Node>
  static inline void zero_value(Node* /*dest*/) {
    // Initialize any fields that require sane defaults within this node.
  }

  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
  }

  template<typename T, typename Node>
  static inline T get_norm(Node* node, int f) {
      return sqrt(dot(node->v, node->v, f));
  }

  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = Base::get_norm<T, Node>(node, f);
    if (norm > 0) {
      for (int z = 0; z < f; z++)
        node->v[z] /= norm;
    }
  }

  template<typename T, typename Node>
  static inline void update_mean(Node* mean, Node* new_node, T norm, int c, int f) {
      for (int z = 0; z < f; z++)
        mean->v[z] = (mean->v[z] * c + new_node->v[z] / norm) / (c + 1);
  }
};

struct Angular : Base {
  template<typename S, typename T>
  struct Node {
    /*
     * We store a binary tree where each node has two things
     * - A vector associated with it
     * - Two children
     * All nodes occupy the same amount of memory
     * All nodes with n_descendants == 1 are leaf nodes.
     * A memory optimization is that for nodes with 2 <= n_descendants <= K,
     * we skip the vector. Instead we store a list of all descendants. K is
     * determined by the number of items that fits in the space of the vector.
     * For nodes with n_descendants == 1 the vector is a data point.
     * For nodes with n_descendants > K the vector is the normal of the split plane.
     * Note that we can't really do sizeof(node<T>) because we cheat and allocate
     * more memory to be able to fit the vector outside
     */
    S n_descendants;
    union {
      S children[2]; // Will possibly store more than 2
      T norm;
    };
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    T pp = x->norm ? x->norm : dot(x->v, x->v, f); // For backwards compatibility reasons, we need to fall back and compute the norm here
    T qq = y->norm ? y->norm : dot(y->v, y->v, f);
    T pq = dot(x->v, y->v, f);
    T ppqq = pp * qq;
    if (ppqq > 0) return T(2.0 - 2.0 * pq / sqrt(ppqq));
    else return T(2.0); // cos is 0
  }
  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random.flip() != 0? true : false;
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    two_means<T, Random, Angular, Node<S, T> >(nodes, f, random, true, p, q);
    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    // Used when requesting distances from Python layer
    // Turns out sometimes the squared distance is -0.0
    // so we have to make sure it's a positive number.
    return sqrt(std::max(distance, T(0)));
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
    n->norm = dot(n->v, n->v, f);
  }
  static const char* name() {
    return "angular";
  }
};


struct DotProduct : Angular {
  template<typename S, typename T>
  struct Node {
    /*
     * This is an extension of the Angular node with extra attributes for the DotProduct metric.
     * It has dot_factor which is needed to reduce the task to Angular distance metric (see the preprocess method)
     * and also a built flag that helps to compute exact dot products when an index is already built.
     */
    S n_descendants;
    S children[2]; // Will possibly store more than 2
    T dot_factor;
    T norm;
    bool built;
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };

  static const char* name() {
    return "dot";
  }

  template<typename T, typename Node>
  static inline T get_norm(Node* node, int f) {
      return sqrt(dot(node->v, node->v, f) + node->dot_factor * node->dot_factor);
  }

  template<typename T, typename Node>
  static inline void update_mean(Node* mean, Node* new_node, T norm, int c, int f) {
      for (int z = 0; z < f; z++)
        mean->v[z] = (mean->v[z] * c + new_node->v[z] / norm) / (c + 1);
      mean->dot_factor = (mean->dot_factor * c + new_node->dot_factor / norm) / (c + 1);
  }

  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    if (x->built || y->built) {
      // When index is already built, we don't need angular distances to retrieve NNs
      // Thus, we can return dot product scores itself
      return -dot(x->v, y->v, f);
    }

    // Calculated by analogy with the angular case
    T pp = x->norm ? x->norm : dot(x->v, x->v, f) + x->dot_factor * x->dot_factor;
    T qq = y->norm ? y->norm : dot(y->v, y->v, f) + y->dot_factor * y->dot_factor;
    T pq = dot(x->v, y->v, f) + x->dot_factor * y->dot_factor;
    T ppqq = pp * qq;

    if (ppqq > 0) return T(2.0 - 2.0 * pq / sqrt(ppqq));
    else return T(2.0);
  }

  template<typename Node>
  static inline void zero_value(Node* dest) {
    dest->dot_factor = 0;
  }

  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
    n->built = false;
    n->norm = dot(n->v, n->v, f) + n->dot_factor * n->dot_factor;
  }

  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
    dest->dot_factor = source->dot_factor;
  }

  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    DotProduct::zero_value(p);
    DotProduct::zero_value(q);
    two_means<T, Random, DotProduct, Node<S, T> >(nodes, f, random, true, p, q);
    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    n->dot_factor = p->dot_factor - q->dot_factor;
    DotProduct::normalize<T, Node<S, T> >(n, f);
  }

  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = T(sqrt(dot(node->v, node->v, f) + pow(node->dot_factor, 2)));
    if (norm > 0) {
      for (int z = 0; z < f; z++)
        node->v[z] /= norm;
      node->dot_factor /= norm;
    }
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f);
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const Node<S, T>* y, int f) {
    return dot(n->v, y->v, f) + n->dot_factor * y->dot_factor;
  }

  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random.flip() != 0? true : false;
  }

  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return (bool)random.flip();
  }

  template<typename T>
  static inline T normalized_distance(T distance) {
    return -distance;
  }

  template<typename T, typename S, typename Node>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // This uses a method from Microsoft Research for transforming inner product spaces to cosine/angular-compatible spaces.
    // (Bachrach et al., 2014, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf)

    // Step one: compute the norm of each vector and store that in its extra dimension (f-1)
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      T d = dot(node->v, node->v, f);
      T norm = d < 0 ? 0 : sqrt(d);
      node->dot_factor = norm;
      node->built = false;
    }

    // Step two: find the maximum norm
    T max_norm = 0;
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      if (node->dot_factor > max_norm) {
        max_norm = node->dot_factor;
      }
    }

    // Step three: set each vector's extra dimension to sqrt(max_norm^2 - norm^2)
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      T node_norm = node->dot_factor;
      T squared_norm_diff = pow(max_norm, static_cast<T>(2.0)) - pow(node_norm, static_cast<T>(2.0));
      T dot_factor = squared_norm_diff < 0 ? 0 : sqrt(squared_norm_diff);

      node->norm = pow(max_norm, static_cast<T>(2.0));
      node->dot_factor = dot_factor;
    }
  }

  template<typename T, typename S, typename Node>
  static inline void postprocess(void* nodes, size_t _s, const S node_count, const int /*f*/) {
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      // When an index is built, we will remember it in index item nodes to compute distances differently
      node->built = true;
    }
  }
};

struct Hamming : Base {
  template<typename S, typename T>
  struct Node {
    S n_descendants;
    S children[2];
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };

  static const size_t max_iterations = 20;

  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    return distance - (margin != (unsigned int) child_nr);
  }

  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::max();
  }
  template<typename T>
  static inline int cole_popcount(T v) {
    // Note: Only used with MSVC 9, which lacks intrinsics and fails to
    // calculate std::bitset::count for v > 32bit. Uses the generalized
    // approach by Eric Cole.
    // See https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSet64
    v = v - ((v >> 1) & (T)~(T)0/3);
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);
    v = (v + (v >> 4)) & (T)~(T)0/255*15;
    return (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * 8;
  }
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    size_t dist = 0;
    for (int i = 0; i < f; i++) {
      dist += annoylib_popcount(x->v[i] ^ y->v[i]);
    }
    return T(dist);
  }
  template<typename S, typename T>
  static inline bool margin(const Node<S, T>* n, const T* y, int /*f*/) {
    static const size_t n_bits = sizeof(T) * 8;
    T chunk = n->v[0] / n_bits;
    return (y[chunk] & (static_cast<T>(1) << (n_bits - 1 - (n->v[0] % n_bits)))) != 0;
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& /*random*/) {
    return margin(n, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t /*s*/, Random& random, Node<S, T>* n) {
    size_t cur_size = 0;
    size_t i = 0;
    int dim = f * 8 * sizeof(T);
    for (; i < max_iterations; i++) {
      // choose random position to split at
      n->v[0] = T(random.index(dim));
      cur_size = 0;
      for (typename vector<Node<S, T>*>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
        if (margin(n, (*it)->v, f)) {
          cur_size++;
        }
      }
      if (cur_size > 0 && cur_size < nodes.size()) {
        break;
      }
    }
    // brute-force search for splitting coordinate
    if (i == max_iterations) {
      int j = 0;
      for (; j < dim; j++) {
        n->v[0] = T(j);
        cur_size = 0;
        for (typename vector<Node<S, T>*>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
          if (margin(n, (*it)->v, f)) {
            cur_size++;
          }
        }
        if (cur_size > 0 && cur_size < nodes.size()) {
          break;
        }
      }
    }
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return distance;
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* /*n*/, int /*f*/) {
  }
  static const char* name() {
    return "hamming";
  }
};


struct Minkowski : Base {
  template<typename S, typename T>
  struct Node {
    S n_descendants;
    T a; // need an extra constant term to determine the offset of the plane
    S children[2];
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };
  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return n->a + dot(n->v, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random.flip() != 0? true : false;
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
};


struct Euclidean : Minkowski {
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return euclidean_distance(x->v, y->v, f);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    two_means<T, Random, Euclidean, Node<S, T> >(nodes, f, random, false, p, q);

    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
    n->a = 0.0;
    for (int z = 0; z < f; z++)
      n->a += -n->v[z] * (p->v[z] + q->v[z]) / 2;
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return sqrt(std::max(distance, T(0)));
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* /*n*/, int /*f*/) {
  }
  static const char* name() {
    return "euclidean";
  }

};

struct Manhattan : Minkowski {
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return manhattan_distance(x->v, y->v, f);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    two_means<T, Random, Manhattan, Node<S, T> >(nodes, f, random, false, p, q);

    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
    n->a = 0.0;
    for (int z = 0; z < f; z++)
      n->a += -n->v[z] * (p->v[z] + q->v[z]) / 2;
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return std::max(distance, T(0));
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* /*n*/, int /*f*/) {
  }
  static const char* name() {
    return "manhattan";
  }
};

template<typename S, typename T, typename R = uint64_t>
class AnnoyIndexInterface {
 public:
  // Note that the methods with an **error argument will allocate memory and write the pointer to that string if error is non-NULL
  virtual ~AnnoyIndexInterface() {};
  virtual bool add_item(S item, const T* w, char** error=NULL) = 0;
  virtual bool build(int q, int n_threads=-1, char** error=NULL) = 0;
  virtual bool unbuild(char** error=NULL) = 0;
  virtual bool save(const char* filename, bool prefault=false, char** error=NULL) = 0;
  virtual void unload() = 0;
  virtual bool load(const char* filename, bool prefault=false, char** error=NULL) = 0;
  virtual T get_distance(S i, S j) const = 0;
  virtual void get_nns_by_item(S item, size_t n, int search_k, vector<S>* result, vector<T>* distances) const = 0;
  virtual void get_nns_by_vector(const T* w, size_t n, int search_k, vector<S>* result, vector<T>* distances) const = 0;
  virtual S get_n_items() const = 0;
  virtual S get_n_trees() const = 0;
  virtual S get_n_nodes(S root) const = 0;
  virtual void verbose(bool v) = 0;
  virtual void get_item(S item, T* v) const = 0;
  virtual void set_seed(R q) = 0;
  virtual bool on_disk_build(const char* filename, char** error=NULL) = 0;
};

template<typename S, typename T, typename Distance, typename Random, class ThreadedBuildPolicy>
  class AnnoyIndex : public AnnoyIndexInterface<S, T,
#if __cplusplus >= 201103L
    typename std::remove_const<decltype(Random::default_seed)>::type
#else
    typename Random::seed_type
#endif
    > {
  /*
   * We use random projection to build a forest of binary trees of all items.
   * Basically just split the hyperspace into two sides by a hyperplane,
   * then recursively split each of those subtrees etc.
   * We create a tree like this q times. The default q is determined automatically
   * in such a way that we at most use 2x as much memory as the vectors take.
   */
public:
  typedef Distance D;
  typedef typename D::template Node<S, T> Node;
#if __cplusplus >= 201103L
  typedef typename std::remove_const<decltype(Random::default_seed)>::type R;
#else
  typedef typename Random::seed_type R;
#endif

protected:
  const int _f;
  size_t _s;
  S _n_items;
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  S _n_nodes;
  S _nodes_size;
  vector<S> _roots;
  int _tree_idx;
  map<int, S> _rt_order;
  S _K;
  R _seed;
  bool _loaded;
  bool _verbose;
  int _fd;
  bool _on_disk;
  bool _built;
public:

   AnnoyIndex(int f) : _f(f), _seed(Random::default_seed) {
    _s = offsetof(Node, v) + _f * sizeof(T); // Size of each node
    _verbose = false;
    _built = false;
    _K = (S) (((size_t) (_s - offsetof(Node, children))) / sizeof(S)); // Max number of descendants to fit into node
    reinitialize(); // Reset everything
  }
  ~AnnoyIndex() {
    unload();
  }

  int get_f() const {
    return _f;
  }

  bool add_item(S item, const T* w, char** error=NULL) override {
    return add_item_impl(item, w, error);
  }

  template<typename W>
  bool add_item_impl(S item, const W& w, char** error=NULL) {
    if (_loaded) {
      set_error_from_string(error, "You can't add an item to a loaded index");
      return false;
    }
    _allocate_size(item + 1);
    Node* n = _get(item);

    D::zero_value(n);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    for (int z = 0; z < _f; z++)
      n->v[z] = w[z];

    D::init_node(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;

    return true;
  }

  bool on_disk_build(const char* file, char** error=NULL) override {
    _on_disk = true;
#ifndef _MSC_VER
    _fd = open(file, O_RDWR | O_CREAT | O_TRUNC, (int) 0600);
#else
    _fd = _open(file, _O_RDWR | _O_CREAT | _O_TRUNC, (int) 0600);
#endif
    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open");
      _fd = 0;
      return false;
    }
    _nodes_size = 1;
    if (ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(_s) * ANNOYLIB_FTRUNCATE_SIZE(_nodes_size)) == -1) {
      set_error_from_errno(error, "Unable to truncate");
      return false;
    }
#ifdef MAP_POPULATE
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
    return true;
  }

  bool build(int q, int n_threads=-1, char** error=NULL) override {
    if (_loaded) {
      set_error_from_string(error, "You can't build a loaded index");
      return false;
    }

    if (_built) {
      set_error_from_string(error, "You can't build a built index");
      return false;
    }

    D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _n_nodes = _n_items;

    ThreadedBuildPolicy::template build<S, T>(this, q, n_threads);

    if (q == -1) {
      // Remove extra trees if any
      int num_nodes = _n_nodes;
      typename map<int, S>::reverse_iterator rit;
      for (rit = _rt_order.rbegin(); rit != _rt_order.rend(); ++rit) {
        S rt_idx = rit->second;
        S num_tree_nodes = get_n_nodes(rt_idx);
        num_nodes -= num_tree_nodes;

        // the criteria of stopping building trees when tree number is not
        // specified: _n_nodes >= 2 * _n_items
        if (num_nodes > 2 * _n_items) {
          auto it = std::remove(_roots.begin(), _roots.end(), rt_idx);
          _roots.erase(it, _roots.end());
          // _n_nodes -= num_tree_nodes;
        }
        else
          break;
      }
    }

    // Also, copy the roots into the last segment of the array
    // This way we can load them faster without reading the whole file
    _allocate_size(_n_nodes + (S)_roots.size());
    for (size_t i = 0; i < _roots.size(); i++)
      memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);
    _n_nodes += S(_roots.size());

    if (_verbose) annoylib_showUpdate("has %d nodes\n", _n_nodes);

    if (_on_disk) {
      if (!remap_memory_and_truncate(&_nodes, _fd,
          static_cast<size_t>(_s) * static_cast<size_t>(_nodes_size),
          static_cast<size_t>(_s) * static_cast<size_t>(_n_nodes))) {
        // TODO: this probably creates an index in a corrupt state... not sure what to do
        set_error_from_errno(error, "Unable to truncate");
        return false;
      }
      _nodes_size = _n_nodes;
    }

    D::template postprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _built = true;
    return true;
  }

  bool unbuild(char** error=NULL) override {
    if (_loaded) {
      set_error_from_string(error, "You can't unbuild a loaded index");
      return false;
    }

    _roots.clear();
    _tree_idx = 0;
    _rt_order.clear();
    _n_nodes = _n_items;
    _built = false;

    return true;
  }

  bool save(const char* filename, bool prefault=false, char** error=NULL) override {
    if (!_built) {
      set_error_from_string(error, "You can't save an index that hasn't been built");
      return false;
    }
    if (_on_disk) {
      return true;
    } else {
      // Delete file if it already exists (See issue #335)
#ifndef _MSC_VER
      unlink(filename);
#else
      _unlink(filename);
#endif

      FILE *f = fopen(filename, "wb");
      if (f == NULL) {
        set_error_from_errno(error, "Unable to open");
        return false;
      }

      if (fwrite(_nodes, _s, _n_nodes, f) != (size_t) _n_nodes) {
        set_error_from_errno(error, "Unable to write");
        return false;
      }

      if (fclose(f) == EOF) {
        set_error_from_errno(error, "Unable to close");
        return false;
      }

      unload();
      return load(filename, prefault, error);
    }
  }

  void reinitialize() {
    _fd = 0;
    _nodes = NULL;
    _loaded = false;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _on_disk = false;
    _seed = Random::default_seed;
    _roots.clear();
    _tree_idx = 0;
    _rt_order.clear();
  }

  void unload() override {
    if (_on_disk && _fd) {
#ifndef _MSC_VER
      close(_fd);
#else
      _close(_fd);
#endif
      munmap(_nodes, _s * _nodes_size);
    } else {
      if (_fd) {
        // we have mmapped data
#ifndef _MSC_VER
        close(_fd);
#else
        _close(_fd);
#endif
        munmap(_nodes, _n_nodes * _s);
      } else if (_nodes) {
        // We have heap allocated data
        free(_nodes);
      }
    }
    reinitialize();
    if (_verbose) annoylib_showUpdate("unloaded\n");
  }

  bool load(const char* filename, bool prefault=false, char** error=NULL) override {
#ifndef _MSC_VER
    _fd = open(filename, O_RDONLY);
#else
    _fd = _open(filename, _O_RDONLY);
#endif
    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open");
      _fd = 0;
      return false;
    }
    off_t size = lseek_getsize(_fd);
    if (size == -1) {
      set_error_from_errno(error, "Unable to get size");
      return false;
    } else if (size == 0) {
      set_error_from_errno(error, "Size of file is zero");
      return false;
    } else if (size % _s) {
      // Something is fishy with this index!
      set_error_from_errno(error, "Index size is not a multiple of vector size. Ensure you are opening using the same metric you used to create the index.");
      return false;
    }

    int flags = MAP_SHARED;
    if (prefault) {
#ifdef MAP_POPULATE
      flags |= MAP_POPULATE;
#else
      annoylib_showUpdate("prefault is set to true, but MAP_POPULATE is not defined on this platform");
#endif
    }
    _nodes = (Node*)mmap(0, size, PROT_READ, flags, _fd, 0);
    _n_nodes = (S)(size / _s);

    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    _roots.clear();
    S m = -1;
    for (S i = _n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == -1 || k == m) {
        _roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    // hacky fix: since the last root precedes the copy of all roots, delete it
    if (_roots.size() > 1 && _get(_roots.front())->children[0] == _get(_roots.back())->children[0])
      _roots.pop_back();
    _loaded = true;
    _built = true;
    _n_items = m;
    if (_verbose) annoylib_showUpdate("found %zu roots with degree %d\n", _roots.size(), m);
    return true;
  }

  T get_distance(S i, S j) const override {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, int search_k, vector<S>* result, vector<T>* distances) const override {
    // TODO: handle OOB
    const Node* m = _get(item);
    _get_all_nns(m->v, n, search_k, result, distances);
  }

  void get_nns_by_vector(const T* w, size_t n, int search_k, vector<S>* result, vector<T>* distances) const override {
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const override {
    return _n_items;
  }

  S get_n_trees() const override {
    return (S)_roots.size();
  }

  S get_n_nodes(S root) const override {
    if (root == 0 || root < _n_items)
      return 0;

    const Node* m = _get(root);
    return 1 + get_n_nodes(m->children[0]) + get_n_nodes(m->children[1]);
  }

  void verbose(bool v) override {
    _verbose = v;
  }

  void get_item(S item, T* v) const override {
    // TODO: handle OOB
    Node* m = _get(item);
    memcpy(v, m->v, (_f) * sizeof(T));
  }

  void set_seed(R seed) override {
    _seed = seed;
  }

  void thread_build(int q, ThreadedBuildPolicy& threaded_build_policy) {
    Random _random(_seed);

    vector<S> thread_roots;
    while (1) {
      if (q == -1) {
        threaded_build_policy.lock_n_nodes();
        if (_n_nodes >= 2 * _n_items) {
          threaded_build_policy.unlock_n_nodes();
          break;
        }
        threaded_build_policy.unlock_n_nodes();
      } else {
        if (thread_roots.size() >= (size_t)q) {
          break;
        }
      }

      if (_verbose) annoylib_showUpdate("pass %zd...\n", thread_roots.size());

      vector<S> indices;
      threaded_build_policy.lock_shared_nodes();
      for (S i = 0; i < _n_items; i++) {
        if (_get(i)->n_descendants >= 1) { // Issue #223
          indices.push_back(i);
        }
      }
      threaded_build_policy.unlock_shared_nodes();

      int tree_idx;
      S root = _make_tree(indices, true, _random, tree_idx, threaded_build_policy);
      thread_roots.push_back(root);

      threaded_build_policy.lock_root_order();
      _rt_order.insert(make_pair(tree_idx, root));
      threaded_build_policy.unlock_root_order();
    }

    threaded_build_policy.lock_roots();
    _roots.insert(_roots.end(), thread_roots.begin(), thread_roots.end());
    threaded_build_policy.unlock_roots();
  }

protected:
  void _reallocate_nodes(S n) {
    const double reallocation_factor = 1.3;
    S new_nodes_size = std::max(n, (S) ((_nodes_size + 1) * reallocation_factor));
    void *old = _nodes;

    if (_on_disk) {
      if (!remap_memory_and_truncate(&_nodes, _fd,
          static_cast<size_t>(_s) * static_cast<size_t>(_nodes_size),
          static_cast<size_t>(_s) * static_cast<size_t>(new_nodes_size)) &&
          _verbose)
          annoylib_showUpdate("File truncation error\n");
    } else {
      _nodes = realloc(_nodes, _s * new_nodes_size);
      memset((char *) _nodes + (_nodes_size * _s) / sizeof(char), 0, (new_nodes_size - _nodes_size) * _s);
    }

    _nodes_size = new_nodes_size;
    if (_verbose) annoylib_showUpdate("Reallocating to %d nodes: old_address=%p, new_address=%p\n", new_nodes_size, old, _nodes);
  }

  void _allocate_size(S n, ThreadedBuildPolicy& threaded_build_policy) {
    if (n > _nodes_size) {
      threaded_build_policy.lock_nodes();
      _reallocate_nodes(n);
      threaded_build_policy.unlock_nodes();
    }
  }

  void _allocate_size(S n) {
    if (n > _nodes_size) {
      _reallocate_nodes(n);
    }
  }

  Node* _get(const S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }

  double _split_imbalance(const vector<S>& left_indices, const vector<S>& right_indices) {
    double ls = (float)left_indices.size();
    double rs = (float)right_indices.size();
    double f = ls / (ls + rs + 1e-9);  // Avoid 0/0
    return std::max(f, 1-f);
  }

  S _make_tree(const vector<S>& indices, bool is_root, Random& _random, int& tree_idx, ThreadedBuildPolicy& threaded_build_policy) {
    // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
    // There's some regrettable complications caused by the problem that root nodes have to be "special":
    // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
    // 2. Root nodes with only 1 child need to be a "dummy" parent
    // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K
    if (indices.size() == 1 && !is_root)
      return indices[0];

    if (indices.size() <= (size_t)_K && (!is_root || (size_t)_n_items <= (size_t)_K || indices.size() == 1)) {
      threaded_build_policy.lock_n_nodes();
      _allocate_size(_n_nodes + 1, threaded_build_policy);
      S item = _n_nodes++;
      threaded_build_policy.unlock_n_nodes();

      threaded_build_policy.lock_shared_nodes();
      Node* m = _get(item);
      m->n_descendants = is_root ? _n_items : (S)indices.size();

      // Using std::copy instead of a loop seems to resolve issues #3 and #13,
      // probably because gcc 4.8 goes overboard with optimizations.
      // Using memcpy instead of std::copy for MSVC compatibility. #235
      // Only copy when necessary to avoid crash in MSVC 9. #293
      if (!indices.empty())
        memcpy(m->children, &indices[0], indices.size() * sizeof(S));

      threaded_build_policy.unlock_shared_nodes();
      return item;
    }

    // each tree gets its own seed
    if(is_root)
    {
      threaded_build_policy.lock_tree_idx();
      tree_idx = _tree_idx;
      _random.set_seed(_seed + 3 * (_tree_idx++));
      threaded_build_policy.unlock_tree_idx();
    }

    threaded_build_policy.lock_shared_nodes();
    vector<Node*> children;
    for (size_t i = 0; i < indices.size(); i++) {
      S j = indices[i];
      Node* n = _get(j);
      if (n)
        children.push_back(n);
    }

    vector<S> children_indices[2];
    Node* m = (Node*)alloca(_s);

    for (int attempt = 0; attempt < 3; attempt++) {
      children_indices[0].clear();
      children_indices[1].clear();
      D::create_split(children, _f, _s, _random, m);

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        Node* n = _get(j);
        if (n) {
          bool side = D::side(m, n, _f, _random);
          children_indices[side].push_back(j);
        } else {
          annoylib_showUpdate("No node for index %d?\n", j);
        }
      }

      if (_split_imbalance(children_indices[0], children_indices[1]) < 0.95)
        break;
    }
    threaded_build_policy.unlock_shared_nodes();

    // If we didn't find a hyperplane, just randomize sides as a last option
    while (_split_imbalance(children_indices[0], children_indices[1]) > 0.99) {
      if (_verbose)
        annoylib_showUpdate("\tNo hyperplane found (left has %zu children, right has %zu children)\n",
          children_indices[0].size(), children_indices[1].size());

      children_indices[0].clear();
      children_indices[1].clear();

      // Set the vector to 0.0
      for (int z = 0; z < _f; z++)
        m->v[z] = 0;

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        // Just randomize...
        children_indices[_random.flip()].push_back(j);
      }
    }

    int flip = (children_indices[0].size() > children_indices[1].size());

    m->n_descendants = is_root ? _n_items : (S)indices.size();
    for (int side = 0; side < 2; side++) {
      // run _make_tree for the smallest child first (for cache locality)
      m->children[side^flip] = _make_tree(children_indices[side^flip], false, _random, tree_idx, threaded_build_policy);
    }

    threaded_build_policy.lock_n_nodes();
    _allocate_size(_n_nodes + 1, threaded_build_policy);
    S item = _n_nodes++;
    threaded_build_policy.unlock_n_nodes();

    threaded_build_policy.lock_shared_nodes();
    memcpy(_get(item), m, _s);
    threaded_build_policy.unlock_shared_nodes();

    return item;
  }

  void _get_all_nns(const T* v, size_t n, int search_k, vector<S>* result, vector<T>* distances) const {
    Node* v_node = (Node *)alloca(_s);
    D::template zero_value<Node>(v_node);
    memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);

    std::priority_queue<pair<T, S> > q;

    if (search_k == -1) {
      search_k = int(n * _roots.size());
    }

    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(make_pair(Distance::template pq_initial_value<T>(), _roots[i]));
    }

    std::vector<S> nns;
    while (nns.size() < (size_t)search_k && !q.empty()) {
      const pair<T, S>& top = q.top();
      T d = top.first;
      S i = top.second;
      Node* nd = _get(i);
      q.pop();
      if (nd->n_descendants == 1 && i < _n_items) {
        nns.push_back(i);
      } else if (nd->n_descendants <= _K) {
        const S* dst = nd->children;
        nns.insert(nns.end(), dst, &dst[nd->n_descendants]);
      } else {
        T margin = D::margin(nd, v, _f);
        q.push(make_pair(D::pq_distance(d, margin, 1), static_cast<S>(nd->children[1])));
        q.push(make_pair(D::pq_distance(d, margin, 0), static_cast<S>(nd->children[0])));
      }
    }

    // Get distances for all items
    // To avoid calculating distance multiple times for any items, sort by id
    std::sort(nns.begin(), nns.end());
    vector<pair<T, S> > nns_dist;
    S last = -1;
    for (size_t i = 0; i < nns.size(); i++) {
      S j = nns[i];
      if (j == last)
        continue;
      last = j;
      if (_get(j)->n_descendants == 1)  // This is only to guard a really obscure case, #284
        nns_dist.push_back(make_pair(D::distance(v_node, _get(j), _f), j));
    }

    size_t m = nns_dist.size();
    size_t p = n < m ? n : m; // Return this many items
    std::partial_sort(nns_dist.begin(), nns_dist.begin() + p, nns_dist.end());
    for (size_t i = 0; i < p; i++) {
      if (distances)
        distances->push_back(D::normalized_distance(nns_dist[i].first));
      result->push_back(nns_dist[i].second);
    }
  }
};

class AnnoyIndexSingleThreadedBuildPolicy {
public:
  template<typename S, typename T, typename D, typename Random>
  static void build(AnnoyIndex<S, T, D, Random, AnnoyIndexSingleThreadedBuildPolicy>* annoy, int q, int /*n_threads*/) {
    AnnoyIndexSingleThreadedBuildPolicy threaded_build_policy;
    annoy->thread_build(q, threaded_build_policy);
  }

  void lock_n_nodes() {}
  void unlock_n_nodes() {}

  void lock_nodes() {}
  void unlock_nodes() {}

  void lock_shared_nodes() {}
  void unlock_shared_nodes() {}

  void lock_roots() {}
  void unlock_roots() {}

  void lock_tree_idx() {}
  void unlock_tree_idx() {}

  void lock_root_order() {}
  void unlock_root_order() {}
};

#ifdef ANNOYLIB_MULTITHREADED_BUILD
class AnnoyIndexMultiThreadedBuildPolicy {
private:
  std::shared_timed_mutex nodes_mutex;
  std::mutex n_nodes_mutex;
  std::mutex roots_mutex;
  std::mutex tree_idx_mutex;
  std::mutex root_order_mutex;

public:
  template<typename S, typename T, typename D, typename Random>
  static void build(AnnoyIndex<S, T, D, Random, AnnoyIndexMultiThreadedBuildPolicy>* annoy, int q, int n_threads) {
    AnnoyIndexMultiThreadedBuildPolicy threaded_build_policy;
    if (n_threads == -1) {
      // If the hardware_concurrency() value is not well defined or not computable, it returns 0.
      // We guard against this by using at least 1 thread.
      n_threads = std::max(1, (int)std::thread::hardware_concurrency());
      // n_threads = 1;
    }

    vector<std::thread> threads(n_threads);

    for (int thread_idx = 0; thread_idx < n_threads; thread_idx++) {
      int trees_per_thread = q == -1 ? -1 : (int)floor((q + thread_idx) / n_threads);

      threads[thread_idx] = std::thread(
        &AnnoyIndex<S, T, D, Random, AnnoyIndexMultiThreadedBuildPolicy>::thread_build,
        annoy,
        trees_per_thread,
        std::ref(threaded_build_policy)
      );
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  void lock_n_nodes() {
    n_nodes_mutex.lock();
  }
  void unlock_n_nodes() {
    n_nodes_mutex.unlock();
  }

  void lock_nodes() {
    nodes_mutex.lock();
  }
  void unlock_nodes() {
    nodes_mutex.unlock();
  }

  void lock_shared_nodes() {
    nodes_mutex.lock_shared();
  }
  void unlock_shared_nodes() {
    nodes_mutex.unlock_shared();
  }

  void lock_roots() {
    roots_mutex.lock();
  }
  void unlock_roots() {
    roots_mutex.unlock();
  }

  void lock_tree_idx() {
    tree_idx_mutex.lock();
  }
  void unlock_tree_idx() {
    tree_idx_mutex.unlock();
  }

  void lock_root_order() {
    root_order_mutex.lock();
  }
  void unlock_root_order() {
    root_order_mutex.unlock();
  }
};
#endif

}

#endif
// vim: tabstop=2 shiftwidth=2
