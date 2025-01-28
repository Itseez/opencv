// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "contours_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

using namespace cv;
using namespace std;

//==============================================================================

namespace {

inline static int findStartContourPoint(uchar* src_data, Size img_size, int j)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_uint8 v_zero = vx_setzero_u8();
    for (; j <= img_size.width - VTraits<v_uint8>::vlanes(); j += VTraits<v_uint8>::vlanes())
    {
        v_uint8 vmask = (v_ne(vx_load((uchar*)(src_data + j)), v_zero));
        if (v_check_any(vmask))
        {
            j += v_scan_forward(vmask);
            return j;
        }
    }
#endif
    for (; j < img_size.width && !src_data[j]; ++j)
        ;
    return j;
}

inline static int findEndContourPoint(uchar* src_data, Size img_size, int j)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if (j < img_size.width && !src_data[j])
    {
        return j;
    }
    else
    {
        v_uint8 v_zero = vx_setzero_u8();
        for (; j <= img_size.width - VTraits<v_uint8>::vlanes(); j += VTraits<v_uint8>::vlanes())
        {
            v_uint8 vmask = (v_eq(vx_load((uchar*)(src_data + j)), v_zero));
            if (v_check_any(vmask))
            {
                j += v_scan_forward(vmask);
                return j;
            }
        }
    }
#endif
    for (; j < img_size.width && src_data[j]; ++j)
        ;

    return j;
}

//==============================================================================

struct LinkRunPoint
{
    int link;
    int next;
    Point pt;
    LinkRunPoint() : link(-1), next(-1) {}
    LinkRunPoint(const Point& pt_) : link(-1), next(-1), pt(pt_) {}
};

typedef LinkRunPoint LRP;

//==============================================================================

class LinkRunner
{
public:
    enum LinkConnectionDirection
    {
        ICV_SINGLE = 0,
        ICV_CONNECTING_ABOVE = 1,
        ICV_CONNECTING_BELOW = -1,
    };

    CTree tree;

    vector<LRP> rns;
    vector<int> ext_rns;
    vector<int> int_rns;

    ContourPointsStorage::storage_t pointsStorage;
    ContourCodesStorage::storage_t codesStorage;

public:
    LinkRunner(void)
    {
        tree.newElem(Contour(&pointsStorage, &codesStorage));
        rns.reserve(100);
    }
    void process(Mat& image);
    void convertLinks(int& first, int& prev, bool isHole);
    void establishLinks(int& prev_point,
                        int upper_run,
                        int lower_run,
                        const int upper_total,
                        const int lower_total);
};

void LinkRunner::convertLinks(int& first, int& prev, bool isHole)
{
    const vector<int>& contours = isHole ? int_rns : ext_rns;
    int count = 0;
    for (int j = 0; j < (int)contours.size(); j++, count++)
    {
        int start = contours[j];
        int cur = start;

        if (rns[cur].link == -1)
            continue;

        CNode& node = tree.newElem(Contour(&pointsStorage, &codesStorage));
        node.body.isHole = isHole;

        do
        {
            node.body.addPoint(rns[cur].pt);
            int p_temp = cur;
            cur = rns[cur].link;
            rns[p_temp].link = -1;
        }
        while (cur != start);

        if (first == 0)
        {
            tree.addChild(0, node.self());
            prev = first = node.self();
        }
        else
        {
            tree.addSiblingAfter(prev, node.self());
            prev = node.self();
        }
    }
}
void LinkRunner::establishLinks(int& prev_point,
                                int upper_run,
                                int lower_run,
                                const int upper_total,
                                const int lower_total)
{
    int k, n;
    int connect_flag = ICV_SINGLE;
    for (k = 0, n = 0; k < upper_total / 2 && n < lower_total / 2;)
    {
        switch (connect_flag)
        {
            case ICV_SINGLE:
                if (rns[rns[upper_run].next].pt.x < rns[rns[lower_run].next].pt.x)
                {
                    if (rns[rns[upper_run].next].pt.x >= rns[lower_run].pt.x - 1)
                    {
                        rns[lower_run].link = upper_run;
                        connect_flag = ICV_CONNECTING_ABOVE;
                        prev_point = rns[upper_run].next;
                    }
                    else
                        rns[rns[upper_run].next].link = upper_run;
                    k++;
                    upper_run = rns[rns[upper_run].next].next;
                }
                else
                {
                    if (rns[upper_run].pt.x <= rns[rns[lower_run].next].pt.x + 1)
                    {
                        rns[lower_run].link = upper_run;
                        connect_flag = ICV_CONNECTING_BELOW;
                        prev_point = rns[lower_run].next;
                    }
                    else
                    {
                        rns[lower_run].link = rns[lower_run].next;
                        // First point of contour
                        ext_rns.push_back(lower_run);
                    }
                    n++;
                    lower_run = rns[rns[lower_run].next].next;
                }
                break;
            case ICV_CONNECTING_ABOVE:
                if (rns[upper_run].pt.x > rns[rns[lower_run].next].pt.x + 1)
                {
                    rns[prev_point].link = rns[lower_run].next;
                    connect_flag = ICV_SINGLE;
                    n++;
                    lower_run = rns[rns[lower_run].next].next;
                }
                else
                {
                    rns[prev_point].link = upper_run;
                    if (rns[rns[upper_run].next].pt.x < rns[rns[lower_run].next].pt.x)
                    {
                        k++;
                        prev_point = rns[upper_run].next;
                        upper_run = rns[rns[upper_run].next].next;
                    }
                    else
                    {
                        connect_flag = ICV_CONNECTING_BELOW;
                        prev_point = rns[lower_run].next;
                        n++;
                        lower_run = rns[rns[lower_run].next].next;
                    }
                }
                break;
            case ICV_CONNECTING_BELOW:
                if (rns[lower_run].pt.x > rns[rns[upper_run].next].pt.x + 1)
                {
                    rns[rns[upper_run].next].link = prev_point;
                    connect_flag = ICV_SINGLE;
                    k++;
                    upper_run = rns[rns[upper_run].next].next;
                }
                else
                {
                    // First point of contour
                    int_rns.push_back(lower_run);

                    rns[lower_run].link = prev_point;
                    if (rns[rns[lower_run].next].pt.x < rns[rns[upper_run].next].pt.x)
                    {
                        n++;
                        prev_point = rns[lower_run].next;
                        lower_run = rns[rns[lower_run].next].next;
                    }
                    else
                    {
                        connect_flag = ICV_CONNECTING_ABOVE;
                        k++;
                        prev_point = rns[upper_run].next;
                        upper_run = rns[rns[upper_run].next].next;
                    }
                }
                break;
        }
    }  // k, n

    for (; n < lower_total / 2; n++)
    {
        if (connect_flag != ICV_SINGLE)
        {
            rns[prev_point].link = rns[lower_run].next;
            connect_flag = ICV_SINGLE;
            lower_run = rns[rns[lower_run].next].next;
            continue;
        }
        rns[lower_run].link = rns[lower_run].next;

        // First point of contour
        ext_rns.push_back(lower_run);
        lower_run = rns[rns[lower_run].next].next;
    }

    for (; k < upper_total / 2; k++)
    {
        if (connect_flag != ICV_SINGLE)
        {
            rns[rns[upper_run].next].link = prev_point;
            connect_flag = ICV_SINGLE;
            upper_run = rns[rns[upper_run].next].next;
            continue;
        }
        rns[rns[upper_run].next].link = upper_run;
        upper_run = rns[rns[upper_run].next].next;
    }
}


void LinkRunner::process(Mat& image)
{
    const Size sz = image.size();
    int j;
    int lower_total;
    int upper_total;
    int all_total;

    Point cur_point;

    rns.reserve(sz.height);  // optimization, assuming some contours exist

    // First line. None of runs is binded
    rns.push_back(LRP());
    int upper_line = (int)rns.size() - 1;
    int cur = upper_line;
    for (j = 0; j < sz.width;)
    {
        j = findStartContourPoint(image.ptr<uchar>(), sz, j);

        if (j == sz.width)
            break;

        cur_point.x = j;

        rns.push_back(LRP(cur_point));
        rns[cur].next = (int)rns.size() - 1;
        cur = rns[cur].next;

        j = findEndContourPoint(image.ptr<uchar>(), sz, j + 1);

        cur_point.x = j - 1;

        rns.push_back(LRP(cur_point));
        rns[cur].next = (int)rns.size() - 1;
        rns[cur].link = rns[cur].next;

        // First point of contour
        ext_rns.push_back(cur);
        cur = rns[cur].next;
    }
    upper_line = rns[upper_line].next;
    upper_total = (int)rns.size() - 1;  // runs->total - 1;

    int last_elem = cur;
    rns[cur].next = -1;
    int prev_point = -1;
    int lower_line = -1;
    for (int i = 1; i < sz.height; i++)
    {
        // Find runs in next line
        cur_point.y = i;
        all_total = (int)rns.size();  // runs->total;
        for (j = 0; j < sz.width;)
        {
            j = findStartContourPoint(image.ptr<uchar>(i), sz, j);

            if (j == sz.width)
                break;

            cur_point.x = j;

            rns.push_back(LRP(cur_point));
            rns[cur].next = (int)rns.size() - 1;
            cur = rns[cur].next;

            j = findEndContourPoint(image.ptr<uchar>(i), sz, j + 1);

            cur_point.x = j - 1;
            rns.push_back(LRP(cur_point));
            cur = rns[cur].next = (int)rns.size() - 1;
        }  // j
        lower_line = rns[last_elem].next;
        lower_total = (int)rns.size() - all_total;  // runs->total - all_total;
        last_elem = cur;
        rns[cur].next = -1;

        CV_DbgAssert(rns.size() < (size_t)numeric_limits<int>::max());

        // Find links between runs of lower_line and upper_line
        establishLinks(prev_point, upper_line, lower_line, upper_total, lower_total);

        upper_line = lower_line;
        upper_total = lower_total;
    }  // i

    // the last line of image
    int upper_run = upper_line;
    for (int k = 0; k < upper_total / 2; k++)
    {
        rns[rns[upper_run].next].link = upper_run;
        upper_run = rns[rns[upper_run].next].next;
    }

    int first = 0;
    int prev = 0;
    convertLinks(first, prev, false);
    convertLinks(first, prev, true);
}

}  // namespace

//==============================================================================

void cv::findContoursLinkRuns(InputArray _image,
                              OutputArrayOfArrays _contours,
                              OutputArray _hierarchy)
{
    CV_INSTRUMENT_REGION();

    CV_CheckType(_image.type(),
                 _image.type() == CV_8UC1 || _image.type() == CV_8SC1,
                 "Bad input image type, must be CV_8UC1 or CV_8SC1");

    // Sanity check: output must be of type vector<vector<Point>>
    CV_Assert(_contours.kind() == _InputArray::STD_VECTOR_VECTOR ||
              _contours.kind() == _InputArray::STD_VECTOR_MAT ||
              _contours.kind() == _InputArray::STD_VECTOR_UMAT);

    if (!_contours.empty())
        CV_CheckTypeEQ(_contours.type(), CV_32SC2, "Contours must have type CV_32SC2");

    if (_hierarchy.needed())
        _hierarchy.clear();

    Mat image = _image.getMat();

    LinkRunner runner;
    runner.process(image);

    contourTreeToResults(runner.tree, CV_32SC2, _contours, _hierarchy);
}


void cv::findContoursLinkRuns(InputArray _image, OutputArrayOfArrays _contours)
{
    CV_INSTRUMENT_REGION();
    findContoursLinkRuns(_image, _contours, noArray());
}
