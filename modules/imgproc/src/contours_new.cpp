// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/imgproc.hpp"
#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/check.hpp"
#include "opencv2/core/utils/logger.hpp"
#include <iostream>
#include <array>
#include <limits>
#include <map>

#include "contours_common.hpp"

using namespace std;
using namespace cv;

//==============================================================================

namespace {

template <typename T>
struct Trait
{
};

static const schar MASK8_RIGHT = '\x80';  // 1000 0000
static const schar MASK8_NEW = '\x02';  // 0000 0010 (+2)
static const schar MASK8_FLAGS = '\xFE';  // 1111 1110 (-2)
static const schar MASK8_BLACK = '\x01';  // 0000 0001 - black pixel

static const schar MASK8_LVAL = '\x7F';  // 0111 1111 (for table)

template <>
struct Trait<schar>
{
    static inline bool checkValue(const schar* elem, const schar*)
    {
        return *elem != 0;
    }
    static inline bool isVal(const schar* elem, const schar*)
    {
        return *elem == MASK8_BLACK;
    }
    static inline bool isRight(const schar* elem, const schar*)
    {
        return (*elem & MASK8_RIGHT) != 0;
    }
    static inline void setRightFlag(schar* elem, const schar*, schar nbd)
    {
        *elem = nbd | MASK8_RIGHT;
    }
    static inline void setNewFlag(schar* elem, const schar*, schar nbd)
    {
        *elem = nbd;
    }
};

static const int MASK_RIGHT = 0x80000000;  // 100..000
static const int MASK_NEW = 0x40000000;  // 010..000
static const int MASK_FLAGS = 0xC0000000;  // right + new
static const int MASK_VAL = 0x3FFFFFFF;  // ~flags - pixel label

template <>
struct Trait<int>
{
    static inline bool checkValue(const int* elem, const int* elem0)
    {
        return (*elem & MASK_VAL) == (*elem0 & MASK_VAL);
    }
    static inline bool isVal(const int* elem, const int* elem0)
    {
        return *elem == (*elem0 & MASK_VAL);
    }
    static inline bool isRight(const int* elem, const int* elem0)
    {
        return (*elem & MASK_RIGHT) == (*elem0 & MASK8_RIGHT);
    }
    static inline void setRightFlag(int* elem, const int* elem0, int)
    {
        *elem = (*elem0 & MASK_VAL) | MASK_NEW | MASK_RIGHT;
    }
    static inline void setNewFlag(int* elem, const int* elem0, int)
    {
        *elem = (*elem0 & MASK_VAL) | MASK_NEW;
    }
};

}  // namespace


//==============================================================================


namespace {

template <typename T>
static bool icvTraceContour(Mat& image, const Point& start, const Point& end, bool isHole)
{
    const T* stop_ptr = image.ptr<T>(end.y, end.x);
    const size_t step = image.step1();
    const T *i0 = image.ptr<T>(start.y, start.x), *i1, *i3, *i4 = NULL;
    const schar s_end = isHole ? 0 : 4;

    schar s = s_end;
    do
    {
        s = (s - 1) & 7;
        i1 = i0 + getDelta(s, step);
    }
    while (!Trait<T>::checkValue(i1, i0) && s != s_end);

    i3 = i0;

    // check single pixel domain
    if (s != s_end)
    {
        // follow border
        for (;;)
        {
            CV_Assert(i3 != NULL);
            s = clamp_direction(s);
            while (s < MAX_SIZE - 1)
            {
                ++s;
                i4 = i3 + getDelta(s, step);
                CV_Assert(i4 != NULL);
                if (Trait<T>::checkValue(i4, i0))
                    break;
            }

            if (i3 == stop_ptr)
            {
                if (!Trait<T>::isRight(i3, i0))
                {
                    // it's the only contour
                    return true;
                }

                // check if this is the last contour
                // encountered during a raster scan
                const T* i5;
                schar t = s;
                while (true)
                {
                    t = (t - 1) & 7;
                    i5 = i3 + getDelta(t, step);
                    if (*i5 != 0)
                        break;
                    if (t == 0)
                        return true;
                }
            }

            if ((i4 == i0 && i3 == i1))
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }  // end of border following loop
    }
    else
    {
        return i3 == stop_ptr;
    }

    return false;
}

template <typename T>
static void icvFetchContourEx(Mat& image,
                              const Point& start,
                              T nbd,
                              Contour& res_contour,
                              const bool isDirect)
{
    const size_t step = image.step1();
    T *i0 = image.ptr<T>(start.y, start.x), *i1, *i3, *i4 = NULL;

    Point pt = res_contour.origin;

    cv::Rect rect(pt.x, pt.y, pt.x, pt.y);

    schar s_end = res_contour.isHole ? 0 : 4;
    schar s = s_end;
    do
    {
        s = (s - 1) & 7;
        i1 = i0 + getDelta(s, step);
    }
    while (!Trait<T>::checkValue(i1, i0) && s != s_end);

    if (s == s_end)
    {
        Trait<T>::setRightFlag(i0, i0, nbd);
        if (!res_contour.isChain)
        {
            res_contour.pts.push_back(pt);
        }
    }
    else
    {
        i3 = i0;
        schar prev_s = s ^ 4;

        // follow border
        for (;;)
        {
            s_end = s;
            s = clamp_direction(s);
            while (s < MAX_SIZE - 1)
            {
                ++s;
                i4 = i3 + getDelta(s, step);
                CV_Assert(i4 != NULL);
                if (Trait<T>::checkValue(i4, i0))
                    break;
            }
            s &= 7;

            // check "right" bound
            if ((unsigned)(s - 1) < (unsigned)s_end)
            {
                Trait<T>::setRightFlag(i3, i0, nbd);
            }
            else if (Trait<T>::isVal(i3, i0))
            {
                Trait<T>::setNewFlag(i3, i0, nbd);
            }

            if (res_contour.isChain)
            {
                res_contour.codes.push_back(s);
            }
            else if (s != prev_s || isDirect)
            {
                res_contour.pts.push_back(pt);
            }

            if (s != prev_s)
            {
                // update bounds
                if (pt.x < rect.x)
                    rect.x = pt.x;
                else if (pt.x > rect.width)
                    rect.width = pt.x;

                if (pt.y < rect.y)
                    rect.y = pt.y;
                else if (pt.y > rect.height)
                    rect.height = pt.y;
            }

            prev_s = s;
            pt += chainCodeDeltas[s];

            if (i4 == i0 && i3 == i1)
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }
    }
    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;
    res_contour.brect = rect;
}

}  // namespace


//==============================================================================

//
// Raster->Chain Tree (Suzuki algorithms)
//

// Structure that is used for sequential retrieving contours from the image.
// It supports both hierarchical and plane variants of Suzuki algorithm.
struct ContourScanner_
{
    Mat image;
    Point offset;  // ROI offset: coordinates, added to each contour point
    Point pt;  // current scanner position
    Point lnbd;  // position of the last met contour
    schar nbd;  // current mark val
    int approx_method1;  // approx method when tracing
    int approx_method2;  // final approx method
    int mode;
    CTree tree;
    array<int, 128> ctable;

public:
    ContourScanner_() {}
    ~ContourScanner_() {}
    inline bool isInt() const
    {
        return (this->mode == RETR_FLOODFILL);
    }
    inline bool isSimple() const
    {
        return (this->mode == RETR_EXTERNAL || this->mode == RETR_LIST);
    }

    CNode& makeContour(schar& nbd_, const bool is_hole, const int x, const int y);
    bool contourScan(const int prev, int& p, Point& last_pos, const int x, const int y);
    int findFirstBoundingContour(const Point& last_pos, const int y, const int lval, int par);
    int findNextX(int x, int y, int& prev, int& p);
    bool findNext();

    static shared_ptr<ContourScanner_> create(Mat img, int mode, int method, Point offset);
};  // class ContourScanner_

typedef shared_ptr<ContourScanner_> ContourScanner;


shared_ptr<ContourScanner_> ContourScanner_::create(Mat img, int mode, int method, Point offset)
{
    if (mode == RETR_CCOMP && img.type() == CV_32SC1)
        mode = RETR_FLOODFILL;

    if (mode == RETR_FLOODFILL)
        CV_CheckTypeEQ(img.type(), CV_32SC1, "RETR_FLOODFILL mode supports only CV_32SC1 images");
    else
        CV_CheckTypeEQ(img.type(),
                       CV_8UC1,
                       "Modes other than RETR_FLOODFILL and RETR_CCOMP support only CV_8UC1 "
                       "images");

    CV_Check(mode,
             mode == RETR_EXTERNAL || mode == RETR_LIST || mode == RETR_CCOMP ||
                 mode == RETR_TREE || mode == RETR_FLOODFILL,
             "Wrong extraction mode");

    CV_Check(method,
             method == 0 || method == CHAIN_APPROX_NONE || method == CHAIN_APPROX_SIMPLE ||
                 method == CHAIN_APPROX_TC89_L1 || method == CHAIN_APPROX_TC89_KCOS,
             "Wrong approximation method");

    Size size = img.size();
    CV_Assert(size.height >= 1);

    shared_ptr<ContourScanner_> scanner = make_shared<ContourScanner_>();
    scanner->image = img;
    scanner->mode = mode;
    scanner->offset = offset;
    scanner->pt = Point(1, 1);
    scanner->lnbd = Point(0, 1);
    scanner->nbd = 2;
    CNode& root = scanner->tree.newElem();
    CV_Assert(root.self() == 0);
    root.body.isHole = true;
    root.body.brect = Rect(Point(0, 0), size);
    scanner->ctable.fill(-1);
    scanner->approx_method2 = scanner->approx_method1 = method;
    if (method == CHAIN_APPROX_TC89_L1 || method == CHAIN_APPROX_TC89_KCOS)
        scanner->approx_method1 = CHAIN_CODE;
    return scanner;
}

CNode& ContourScanner_::makeContour(schar& nbd_, const bool is_hole, const int x, const int y)
{
    const bool isChain = (this->approx_method1 == CHAIN_CODE);  // TODO: get rid of old constant
    const bool isDirect = (this->approx_method1 == CHAIN_APPROX_NONE);

    const Point start_pt(x - (is_hole ? 1 : 0), y);

    CNode& res = tree.newElem();
    res.body.isHole = is_hole;
    res.body.isChain = isChain;
    res.body.origin = start_pt + offset;
    if (isSimple())
    {
        icvFetchContourEx<schar>(this->image, start_pt, MASK8_NEW, res.body, isDirect);
    }
    else
    {
        schar lval;
        if (isInt())
        {
            const int start_val = this->image.at<int>(start_pt);
            lval = start_val & MASK8_LVAL;
            icvFetchContourEx<int>(this->image, start_pt, 0, res.body, isDirect);
        }
        else
        {
            lval = nbd_;
            // change nbd
            nbd_ = (nbd_ + 1) & MASK8_LVAL;
            if (nbd_ == 0)
                nbd_ = MASK8_BLACK | MASK8_NEW;
            icvFetchContourEx<schar>(this->image, start_pt, lval, res.body, isDirect);
        }
        res.body.brect.x -= this->offset.x;
        res.body.brect.y -= this->offset.y;
        res.ctable_next = this->ctable[lval];
        this->ctable[lval] = res.self();
    }
    const Point prev_origin = res.body.origin;
    res.body.origin = start_pt;
    if (this->approx_method1 != this->approx_method2)
    {
        CV_Assert(res.body.isChain);
        res.body.pts = approximateChainTC89(res.body.codes, prev_origin, this->approx_method2);
        res.body.isChain = false;
    }
    return res;
}

bool ContourScanner_::contourScan(const int prev, int& p, Point& last_pos, const int x, const int y)
{
    bool is_hole = false;

    /* if not external contour */
    if (isInt())
    {
        if (!(((prev & MASK_FLAGS) != 0 || prev == 0) && (p & MASK_FLAGS) == 0))
        {
            if ((prev & MASK_FLAGS) != 0 || ((p & MASK_FLAGS) != 0))
                return false;

            if (prev & MASK_FLAGS)
            {
                last_pos.x = x - 1;
            }
            is_hole = true;
        }
    }
    else
    {
        if (!(prev == 0 && p == 1))
        {
            if (p != 0 || prev < 1)
                return false;

            if (prev & MASK8_FLAGS)
            {
                last_pos.x = x - 1;
            }
            is_hole = true;
        }
    }

    if (mode == RETR_EXTERNAL && (is_hole || this->image.at<schar>(last_pos) > 0))
    {
        return false;
    }

    /* find contour parent */
    int main_parent = -1;
    if (isSimple() || (!is_hole && (mode == RETR_CCOMP || mode == RETR_FLOODFILL)) ||
        last_pos.x <= 0)
    {
        main_parent = 0;
    }
    else
    {
        int lval;
        if (isInt())
            lval = this->image.at<int>(last_pos.y, last_pos.x) & MASK8_LVAL;
        else
            lval = this->image.at<schar>(last_pos.y, last_pos.x) & MASK8_LVAL;

        main_parent = findFirstBoundingContour(last_pos, y, lval, main_parent);

        // if current contour is a hole and previous contour is a hole or
        // current contour is external and previous contour is external then
        // the parent of the contour is the parent of the previous contour else
        // the parent is the previous contour itself.
        {
            CNode& main_parent_elem = tree.elem(main_parent);
            if (main_parent_elem.body.isHole == is_hole)
            {
                if (main_parent_elem.parent != -1)
                {
                    main_parent = main_parent_elem.parent;
                }
                else
                {
                    main_parent = 0;
                }
            }
        }

        // hole flag of the parent must differ from the flag of the contour
        {
            CNode& main_parent_elem = tree.elem(main_parent);
            CV_Assert(main_parent_elem.body.isHole != is_hole);
        }
    }

    last_pos.x = x - (is_hole ? 1 : 0);

    schar nbd_ = this->nbd;
    CNode& new_contour = makeContour(nbd_, is_hole, x, y);
    if (new_contour.parent == -1)
    {
        tree.addChild(main_parent, new_contour.self());
    }
    this->pt.x = !isInt() ? (x + 1) : (x + 1 - (is_hole ? 1 : 0));
    this->pt.y = y;
    this->nbd = nbd_;
    return true;
}

int ContourScanner_::findFirstBoundingContour(const Point& last_pos,
                                              const int y,
                                              const int lval,
                                              int par)
{
    const Point end_point(last_pos.x, y);
    int res = par;
    int cur = ctable[lval];
    while (cur != -1)
    {
        CNode& cur_elem = tree.elem(cur);
        if (((last_pos.x - cur_elem.body.brect.x) < cur_elem.body.brect.width) &&
            ((last_pos.y - cur_elem.body.brect.y) < cur_elem.body.brect.height))
        {
            if (res != -1)
            {
                CNode& res_elem = tree.elem(res);
                const Point origin = res_elem.body.origin;
                const bool isHole = res_elem.body.isHole;
                if (isInt())
                {
                    if (icvTraceContour<int>(this->image, origin, end_point, isHole))
                        break;
                }
                else
                {
                    if (icvTraceContour<schar>(this->image, origin, end_point, isHole))
                        break;
                }
            }
            res = cur;
        }
        cur = cur_elem.ctable_next;
    }
    return res;
}

int ContourScanner_::findNextX(int x, int y, int& prev, int& p)
{
    const int width = this->image.size().width - 1;
    if (isInt())
    {
        for (; x < width &&
               ((p = this->image.at<int>(y, x)) == prev || (p & MASK_VAL) == (prev & MASK_VAL));
             x++)
            prev = p;
    }
    else
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if ((p = this->image.at<schar>(y, x)) != prev)
        {
            return x;
        }
        else
        {
            v_uint8 v_prev = vx_setall_u8((uchar)prev);
            for (; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
            {
                v_uint8 vmask = (v_ne(vx_load(this->image.ptr<uchar>(y, x)), v_prev));
                if (v_check_any(vmask))
                {
                    x += v_scan_forward(vmask);
                    p = this->image.at<schar>(y, x);
                    return x;
                }
            }
        }
#endif
        for (; x < width && (p = this->image.at<schar>(y, x)) == prev; x++)
            ;
    }
    return x;
}

bool ContourScanner_::findNext()
{
    int x = this->pt.x;
    int y = this->pt.y;
    int width = this->image.size().width - 1;
    int height = this->image.size().height - 1;
    Point last_pos = this->lnbd;
    int prev = isInt() ? this->image.at<int>(y, x - 1) : this->image.at<schar>(y, x - 1);

    for (; y < height; y++)
    {
        int p = 0;
        for (; x < width; x++)
        {
            x = findNextX(x, y, prev, p);
            if (x >= width)
                break;
            if (contourScan(prev, p, last_pos, x, y))
            {
                this->lnbd = last_pos;
                return true;
            }
            else
            {
                prev = p;
                if ((isInt() && (prev & MASK_FLAGS)) || (!isInt() && (prev & MASK8_FLAGS)))
                {
                    last_pos.x = x;
                }
            }
        }
        last_pos = Point(0, y + 1);
        x = 1;
        prev = 0;
    }

    return false;
}

//==============================================================================

void cv::findContours(InputArray _image,
                      OutputArrayOfArrays _contours,
                      OutputArray _hierarchy,
                      int mode,
                      int method,
                      Point offset)
{
    CV_INSTRUMENT_REGION();

    // TODO: remove this block in future
    if (method == 5 /*CV_LINK_RUNS*/)
    {
        CV_LOG_ONCE_WARNING(NULL,
                            "LINK_RUNS mode has been extracted to separate function: "
                            "cv::findContoursLinkRuns. "
                            "Calling through cv::findContours will be removed in future.");
        CV_CheckTrue(!_hierarchy.needed() || mode == RETR_CCOMP,
                     "LINK_RUNS mode supports only simplified hierarchy output (mode=RETR_CCOMP)");
        findContoursLinkRuns(_image, _contours, _hierarchy);
        return;
    }

    // TODO: need enum value, need way to return contour starting points with chain codes
    if (method == 0 /*CV_CHAIN_CODE*/)
    {
        CV_LOG_ONCE_WARNING(NULL,
                            "Chain code output is an experimental feature and might change in "
                            "future!");
    }

    // Sanity check: output must be of type vector<vector<Point>>
    CV_Assert((_contours.kind() == _InputArray::STD_VECTOR_VECTOR) ||
              (_contours.kind() == _InputArray::STD_VECTOR_MAT) ||
              (_contours.kind() == _InputArray::STD_VECTOR_UMAT));

    const int res_type = (method == 0 /*CV_CHAIN_CODE*/) ? CV_8SC1 : CV_32SC2;
    if (!_contours.empty())
    {
        CV_CheckTypeEQ(_contours.type(),
                       res_type,
                       "Contours must have type CV_8SC1 (chain code) or CV_32SC2 (other methods)");
    }

    if (_hierarchy.needed())
        _hierarchy.clear();

    // preprocess
    Mat image;
    copyMakeBorder(_image, image, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED, Scalar(0));
    if (image.type() != CV_32SC1)
        threshold(image, image, 0, 1, THRESH_BINARY);

    // find contours
    ContourScanner scanner = ContourScanner_::create(image, mode, method, offset + Point(-1, -1));
    while (scanner->findNext())
    {
    }

    contourTreeToResults(scanner->tree, res_type, _contours, _hierarchy);
}

void cv::findContours(InputArray _image,
                      OutputArrayOfArrays _contours,
                      int mode,
                      int method,
                      Point offset)
{
    CV_INSTRUMENT_REGION();
    findContours(_image, _contours, noArray(), mode, method, offset);
}
