// //////////////////////////////////////////////////////////////////////////////////////
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
//

// //////////////////////////////////////////////////////////////////////////////////////
// Author: Sajjad Taheri, University of California, Irvine. sajjadt[at]uci[dot]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015 The Regents of the University of California (Regents)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the University nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

QUnit.module('Image Processing', {});

QUnit.test('applyColorMap', function(assert) {
    {
        let src = cv.matFromArray(2, 1, cv.CV_8U, [50,100]);
        cv.applyColorMap(src, src, cv.COLORMAP_BONE);

        // Verify result.
        let expected = new Uint8Array([60,44,44,119,89,87]);

        assert.deepEqual(src.data, expected);
        src.delete();
    }
});

QUnit.test('blendLinear', function(assert) {
    {
        let src1 = cv.matFromArray(2, 1, cv.CV_8U, [50,100]);
        let src2 = cv.matFromArray(2, 1, cv.CV_8U, [200,20]);
        let weights1 = cv.matFromArray(2, 1, cv.CV_32F, [0.4,0.5]);
        let weights2 = cv.matFromArray(2, 1, cv.CV_32F, [0.6,0.5]);
        let dst = new cv.Mat();
        cv.blendLinear(src1, src2, weights1, weights2, dst);

        // Verify result.
        let expected = new Uint8Array([140,60]);

        assert.deepEqual(dst.data, expected);
        src1.delete();
        src2.delete();
        weights1.delete();
        weights2.delete();
        dst.delete();
    }
});

QUnit.test('createHanningWindow', function(assert) {
    {
        let dst = new cv.Mat();
        cv.createHanningWindow(dst, new cv.Size(5, 3), cv.CV_32F);

        // Verify result.
        let expected = cv.matFromArray(3, 5, cv.CV_32F, [0.,0.,0.,0.,0.,0.,0.70710677,1.,0.70710677,0.,0.,0.,0.,0.,0.]);

        assert.deepEqual(dst.data, expected.data);
        dst.delete();
        expected.delete();
    }
});

QUnit.test('test_imgProc', function(assert) {
    // calcHist
    {
        let vec1 = new cv.Mat.ones(new cv.Size(20, 20), cv.CV_8UC1); // eslint-disable-line new-cap
        let source = new cv.MatVector();
        source.push_back(vec1);
        let channels = [0];
        let histSize = [256];
        let ranges =[0, 256];

        let hist = new cv.Mat();
        let mask = new cv.Mat();
        let binSize = cv._malloc(4);
        let binView = new Int32Array(cv.HEAP8.buffer, binSize);
        binView[0] = 10;
        cv.calcHist(source, channels, mask, hist, histSize, ranges, false);

        // hist should contains a N X 1 array.
        let size = hist.size();
        assert.equal(size.height, 1);
        assert.equal(size.width, 256);

        // default parameters
        cv.calcHist(source, channels, mask, hist, histSize, ranges);
        size = hist.size();
        assert.equal(size.height, 1);
        assert.equal(size.width, 256);

        // Do we need to verify data in histogram?
        // let dataView = hist.data;

        // Free resource
        cv._free(binSize);
        mask.delete();
        hist.delete();
    }

    // cvtColor
    {
        let source = new cv.Mat(10, 10, cv.CV_8UC3);
        let dest = new cv.Mat();

        cv.cvtColor(source, dest, cv.COLOR_BGR2GRAY, 0);
        assert.equal(dest.channels(), 1);

        cv.cvtColor(source, dest, cv.COLOR_BGR2GRAY);
        assert.equal(dest.channels(), 1);

        cv.cvtColor(source, dest, cv.COLOR_BGR2BGRA, 0);
        assert.equal(dest.channels(), 4);

        cv.cvtColor(source, dest, cv.COLOR_BGR2BGRA);
        assert.equal(dest.channels(), 4);

        dest.delete();
        source.delete();
    }

    // equalizeHist
    {
        let source = new cv.Mat(10, 10, cv.CV_8UC1);
        let dest = new cv.Mat();

        cv.equalizeHist(source, dest);

        // eualizeHist changes the content of a image, but does not alter meta data
        // of it.
        assert.equal(source.channels(), dest.channels());
        assert.equal(source.type(), dest.type());

        dest.delete();
        source.delete();
    }

    // floodFill
    {
        let center = new cv.Point(5, 5);
        let rect = new cv.Rect(0, 0, 0, 0);
        let img = new cv.Mat.zeros(10, 10, cv.CV_8UC1);
        let color = new cv.Scalar (255);
        cv.circle(img, center, 3, color, 1);

        let edge = new cv.Mat();
        cv.Canny(img, edge, 100, 255);
        cv.copyMakeBorder(edge, edge, 1, 1, 1, 1, cv.BORDER_REPLICATE);

        let expected_img_data = new Uint8Array([
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
            0,   0,   0, 255, 255, 255, 255, 255,   0,   0,
            0,   0,   0, 255,   0, 255,   0, 255,   0,   0,
            0,   0, 255, 255, 255, 255,   0,   0, 255,   0,
            0,   0,   0, 255,   0,   0,   0, 255,   0,   0,
            0,   0,   0, 255, 255,   0, 255, 255,   0,   0,
            0,   0,   0,   0,   0, 255,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0]);

        let img_elem = 10*10*1;
        let expected_img_data_ptr = cv._malloc(img_elem);
        let expected_img_data_heap = new Uint8Array(cv.HEAPU8.buffer,
                                                    expected_img_data_ptr,
                                                    img_elem);
        expected_img_data_heap.set(new Uint8Array(expected_img_data.buffer));

        let expected_img = new cv.Mat(  10, 10, cv.CV_8UC1, expected_img_data_ptr, 0);

        let expected_rect = new cv.Rect(3,3,3,3);

        let compare_result = new cv.Mat(10, 10, cv.CV_8UC1);

        cv.floodFill(img, edge, center, color, rect);

        cv.compare (img, expected_img, compare_result, cv.CMP_EQ);

        // expect every pixels are the same.
        assert.equal (cv.countNonZero(compare_result), img.total());
        assert.equal (rect.x, expected_rect.x);
        assert.equal (rect.y, expected_rect.y);
        assert.equal (rect.width, expected_rect.width);
        assert.equal (rect.height, expected_rect.height);

        img.delete();
        edge.delete();
        expected_img.delete();
        compare_result.delete();
    }
});

QUnit.test('Drawing Functions', function(assert) {
    // fillPoly
    {
        let img_width = 6;
        let img_height = 6;

        let img = new cv.Mat.zeros(img_height, img_width, cv.CV_8UC1);

        let npts = 4;
        let square_point_data = new Uint8Array([
            1, 1,
            4, 1,
            4, 4,
            1, 4]);
        let square_points = cv.matFromArray(npts, 1, cv.CV_32SC2, square_point_data);
        let pts = new cv.MatVector();
        pts.push_back (square_points);
        let color = new cv.Scalar (255);

        let expected_img_data = new Uint8Array([
            0,   0,   0,   0,   0,   0,
            0, 255, 255, 255, 255,   0,
            0, 255, 255, 255, 255,   0,
            0, 255, 255, 255, 255,   0,
            0, 255, 255, 255, 255,   0,
            0,   0,   0,   0,   0,   0]);
        let expected_img = cv.matFromArray(img_height, img_width, cv.CV_8UC1, expected_img_data);

        cv.fillPoly(img, pts, color);

        let compare_result = new cv.Mat(img_height, img_width, cv.CV_8UC1);

        cv.compare (img, expected_img, compare_result, cv.CMP_EQ);

        // expect every pixels are the same.
        assert.equal (cv.countNonZero(compare_result), img.total());

        img.delete();
        square_points.delete();
        pts.delete();
        expected_img.delete();
        compare_result.delete();
    }

    // fillConvexPoly
    {
        let img_width = 6;
        let img_height = 6;

        let img = new cv.Mat.zeros(img_height, img_width, cv.CV_8UC1);

        let npts = 4;
        let square_point_data = new Uint8Array([
            1, 1,
            4, 1,
            4, 4,
            1, 4]);
        let square_points = cv.matFromArray(npts, 1, cv.CV_32SC2, square_point_data);
        let color = new cv.Scalar (255);

        let expected_img_data = new Uint8Array([
            0,   0,   0,   0,   0,   0,
            0, 255, 255, 255, 255,   0,
            0, 255, 255, 255, 255,   0,
            0, 255, 255, 255, 255,   0,
            0, 255, 255, 255, 255,   0,
            0,   0,   0,   0,   0,   0]);
        let expected_img = cv.matFromArray(img_height, img_width, cv.CV_8UC1, expected_img_data);

        cv.fillConvexPoly(img, square_points, color);

        let compare_result = new cv.Mat(img_height, img_width, cv.CV_8UC1);

        cv.compare (img, expected_img, compare_result, cv.CMP_EQ);

        // expect every pixels are the same.
        assert.equal (cv.countNonZero(compare_result), img.total());

        img.delete();
        square_points.delete();
        expected_img.delete();
        compare_result.delete();
    }
});

QUnit.test('test_segmentation', function(assert) {
    const THRESHOLD = 127.0;
    const THRESHOLD_MAX = 210.0;

    // threshold
    {
        let source = new cv.Mat(1, 5, cv.CV_8UC1);
        let sourceView = source.data;
        sourceView[0] = 0; // < threshold
        sourceView[1] = 100; // < threshold
        sourceView[2] = 200; // > threshold

        let dest = new cv.Mat();

        cv.threshold(source, dest, THRESHOLD, THRESHOLD_MAX, cv.THRESH_BINARY);

        let destView = dest.data;
        assert.equal(destView[0], 0);
        assert.equal(destView[1], 0);
        assert.equal(destView[2], THRESHOLD_MAX);
    }

    // adaptiveThreshold
    {
        let source = cv.Mat.zeros(1, 5, cv.CV_8UC1);
        let sourceView = source.data;
        sourceView[0] = 50;
        sourceView[1] = 150;
        sourceView[2] = 200;

        let dest = new cv.Mat();
        const C = 0;
        const blockSize = 3;
        cv.adaptiveThreshold(source, dest, THRESHOLD_MAX,
                             cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, C);

        let destView = dest.data;
        assert.equal(destView[0], 0);
        assert.equal(destView[1], THRESHOLD_MAX);
        assert.equal(destView[2], THRESHOLD_MAX);
    }
});

QUnit.test('test_shape', function(assert) {
    // moments
    {
        let points = new cv.Mat(1, 4, cv.CV_32SC2);
        let data32S = points.data32S;
        data32S[0]=50;
        data32S[1]=56;
        data32S[2]=53;
        data32S[3]=53;
        data32S[4]=46;
        data32S[5]=54;
        data32S[6]=49;
        data32S[7]=51;

        let m = cv.moments(points, false);
        let area = cv.contourArea(points, false);

        assert.equal(m.m00, 0);
        assert.equal(m.m01, 0);
        assert.equal(m.m10, 0);
        assert.equal(area, 0);

        // default parameters
        m = cv.moments(points);
        area = cv.contourArea(points);
        assert.equal(m.m00, 0);
        assert.equal(m.m01, 0);
        assert.equal(m.m10, 0);
        assert.equal(area, 0);

        points.delete();
    }
});

QUnit.test('test_min_enclosing', function(assert) {
    // minEnclosingCircle
    {
        let points = new cv.Mat(4, 1, cv.CV_32FC2);

        points.data32F[0] = 0;
        points.data32F[1] = 0;
        points.data32F[2] = 1;
        points.data32F[3] = 0;
        points.data32F[4] = 1;
        points.data32F[5] = 1;
        points.data32F[6] = 0;
        points.data32F[7] = 1;

        let circle = cv.minEnclosingCircle(points);

        assert.deepEqual(circle.center, {x: 0.5, y: 0.5});
        assert.ok(Math.abs(circle.radius - Math.sqrt(2) / 2) < 0.001);

        points.delete();
    }

    // minEnclosingTriangle
    {
        let dst = cv.Mat.zeros(80, 80, cv.CV_8U);
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        let triangle = new cv.Mat();

        cv.drawMarker(dst, new cv.Point(40, 40), new cv.Scalar(255));
        cv.findContoursLinkRuns(dst,contours,hierarchy);
        cv.minEnclosingTriangle(contours.get(0),triangle);

        // Verify result.
        const triangleData = triangle.data32F;
        assert.deepEqual(triangleData[0], triangleData[4]);
        assert.deepEqual(triangleData[1], 20);
        assert.deepEqual(triangleData[2], 30);
        assert.deepEqual(triangleData[3], 40);
        assert.deepEqual(triangleData[5], 60);

        dst.delete();
        contours.delete();
        hierarchy.delete();
        triangle.delete();
    }
});

QUnit.test('test_filter', function(assert) {
    // blur
    {
        let mat1 = cv.Mat.ones(5, 5, cv.CV_8UC3);
        let mat2 = new cv.Mat();

        cv.blur(mat1, mat2, {height: 3, width: 3}, {x: -1, y: -1}, cv.BORDER_DEFAULT);

        // Verify result.
        let size = mat2.size();
        assert.equal(mat2.channels(), 3);
        assert.equal(size.height, 5);
        assert.equal(size.width, 5);

        cv.blur(mat1, mat2, {height: 3, width: 3}, {x: -1, y: -1});

        // Verify result.
        size = mat2.size();
        assert.equal(mat2.channels(), 3);
        assert.equal(size.height, 5);
        assert.equal(size.width, 5);

        cv.blur(mat1, mat2, {height: 3, width: 3});

        // Verify result.
        size = mat2.size();
        assert.equal(mat2.channels(), 3);
        assert.equal(size.height, 5);
        assert.equal(size.width, 5);

        mat1.delete();
        mat2.delete();
    }

    // GaussianBlur
    {
        let mat1 = cv.Mat.ones(7, 7, cv.CV_8UC1);
        let mat2 = new cv.Mat();

        cv.GaussianBlur(mat1, mat2, new cv.Size(3, 3), 0, 0, // eslint-disable-line new-cap
                        cv.BORDER_DEFAULT);

        // Verify result.
        let size = mat2.size();
        assert.equal(mat2.channels(), 1);
        assert.equal(size.height, 7);
        assert.equal(size.width, 7);
        mat1.delete();
        mat2.delete();
    }

    // spatialGradient
    {
        let src = cv.matFromArray(4, 4, cv.CV_8U, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
        let dx = new cv.Mat();
        let dy = new cv.Mat();
        cv.spatialGradient(src, dx, dy);

        // Verify result.
        let expected_dx = new cv.Mat();
        let expected_dy = new cv.Mat();
        cv.Sobel(src, expected_dx, cv.CV_16SC1, 1, 0, 3);
        cv.Sobel(src, expected_dy, cv.CV_16SC1, 0, 1, 3);

        assert.deepEqual(dx.data, expected_dx.data);
        assert.deepEqual(dy.data, expected_dy.data);

        src.delete();
        dx.delete();
        dy.delete();
        expected_dx.delete();
        expected_dy.delete();
    }

    // sqrBoxFilter
    {
        let src = cv.matFromArray(2, 3, cv.CV_8U, [1,2,1,1,2,1]);
        let dst = new cv.Mat();
        cv.sqrBoxFilter(src, dst, cv.CV_32F, new cv.Size(3, 3));

        // Verify result.
        let expected = cv.matFromArray(2, 3, cv.CV_32F,[3.0,2.0,3.0,3.0,2.0,3.0]);

        assert.deepEqual(dst.data, expected.data);
        src.delete();
        dst.delete();
        expected.delete();
    }

    // stackBlur
    {
        let src = cv.matFromArray(2, 3, cv.CV_8U, [10,25,30,45,50,60]);
        cv.stackBlur(src, src, new cv.Size(3, 3));

        // Verify result.
        let expected = new Uint8Array([22,29,36,38,43,50]);

        assert.deepEqual(src.data, expected);
        src.delete();
    }

    // medianBlur
    {
        let mat1 = cv.Mat.ones(9, 9, cv.CV_8UC3);
        let mat2 = new cv.Mat();

        cv.medianBlur(mat1, mat2, 3);

        // Verify result.
        let size = mat2.size();

        assert.equal(mat2.channels(), 3);
        assert.equal(size.height, 9);
        assert.equal(size.width, 9);
        mat1.delete();
        mat2.delete();
    }

    // bilateralFilter
    {
        let mat1 = cv.Mat.ones(11, 11, cv.CV_8UC3);
        let mat2 = new cv.Mat();

        cv.bilateralFilter(mat1, mat2, 3, 6, 1.5, cv.BORDER_DEFAULT);

        // Verify result.
        let size = mat2.size();
        assert.equal(mat2.channels(), 3);
        assert.equal(size.height, 11);
        assert.equal(size.width, 11);

        // default parameters
        cv.bilateralFilter(mat1, mat2, 3, 6, 1.5);
        // Verify result.
        size = mat2.size();
        assert.equal(mat2.channels(), 3);
        assert.equal(size.height, 11);
        assert.equal(size.width, 11);

        mat1.delete();
        mat2.delete();
    }
});

QUnit.test('test_watershed', function(assert) {
    {
        let mat = cv.Mat.ones(11, 11, cv.CV_8UC3);
        let out = new cv.Mat(11, 11, cv.CV_32SC1);

        cv.watershed(mat, out);

        // Verify result.
        let size = out.size();
        assert.equal(out.channels(), 1);
        assert.equal(size.height, 11);
        assert.equal(size.width, 11);
        assert.equal(out.elemSize1(), 4);

        mat.delete();
        out.delete();
    }
});

QUnit.test('test_distanceTransform', function(assert) {
    {
        let mat = cv.Mat.ones(11, 11, cv.CV_8UC1);
        let out = new cv.Mat(11, 11, cv.CV_32FC1);
        let labels = new cv.Mat(11, 11, cv.CV_32FC1);
        const maskSize = 3;
        cv.distanceTransform(mat, out, cv.DIST_L2, maskSize, cv.CV_32F);

        // Verify result.
        let size = out.size();
        assert.equal(out.channels(), 1);
        assert.equal(size.height, 11);
        assert.equal(size.width, 11);
        assert.equal(out.elemSize1(), 4);

        cv.distanceTransformWithLabels(mat, out, labels, cv.DIST_L2, maskSize,
                                       cv.DIST_LABEL_CCOMP);

        // Verify result.
        size = out.size();
        assert.equal(out.channels(), 1);
        assert.equal(size.height, 11);
        assert.equal(size.width, 11);
        assert.equal(out.elemSize1(), 4);

        size = labels.size();
        assert.equal(labels.channels(), 1);
        assert.equal(size.height, 11);
        assert.equal(size.width, 11);
        assert.equal(labels.elemSize1(), 4);

        mat.delete();
        out.delete();
        labels.delete();
    }
});

QUnit.test('test_integral', function(assert) {
    {
        let mat = cv.Mat.eye({height: 100, width: 100}, cv.CV_8UC3);
        let sum = new cv.Mat();
        let sqSum = new cv.Mat();
        let title = new cv.Mat();

        cv.integral(mat, sum, -1);

        // Verify result.
        let size = sum.size();
        assert.equal(sum.channels(), 3);
        assert.equal(size.height, 100+1);
        assert.equal(size.width, 100+1);

        cv.integral2(mat, sum, sqSum, -1, -1);
        // Verify result.
        size = sum.size();
        assert.equal(sum.channels(), 3);
        assert.equal(size.height, 100+1);
        assert.equal(size.width, 100+1);

        size = sqSum.size();
        assert.equal(sqSum.channels(), 3);
        assert.equal(size.height, 100+1);
        assert.equal(size.width, 100+1);

        mat.delete();
        sum.delete();
        sqSum.delete();
        title.delete();
    }
});

QUnit.test('test_rotatedRectangleIntersection', function(assert) {
    {
        let dst = cv.Mat.zeros(80, 80, cv.CV_8U);
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        let intersectionPoints = new cv.Mat();

        cv.drawMarker(dst, new cv.Point(40, 40), new cv.Scalar(255));
        cv.findContoursLinkRuns(dst,contours,hierarchy);
        let rr1 = cv.minAreaRect(contours.get(0));
        let rr2 = cv.minAreaRect(contours.get(0));
        let rr3 = new cv.RotatedRect({x: 40, y: 40}, {height: 10, width: 20}, 45);

        let intersectionType = cv.rotatedRectangleIntersection(rr1, rr2, intersectionPoints);

        // Verify result.
        assert.deepEqual(intersectionType, cv.INTERSECT_FULL);
        intersectionPoints.convertTo(intersectionPoints, cv.CV_32S);
        let intersectionPointsData = intersectionPoints.data32S;
        assert.deepEqual(intersectionPointsData[0], 30);
        assert.deepEqual(intersectionPointsData[1], 40);
        assert.deepEqual(intersectionPointsData[2], 40);
        assert.deepEqual(intersectionPointsData[3], 30);
        assert.deepEqual(intersectionPointsData[4], 50);
        assert.deepEqual(intersectionPointsData[5], 40);
        assert.deepEqual(intersectionPointsData[6], 40);
        assert.deepEqual(intersectionPointsData[7], 50);

        intersectionType = cv.rotatedRectangleIntersection(rr1, rr3, intersectionPoints);

        // Verify result.
        assert.deepEqual(intersectionType, cv.INTERSECT_PARTIAL);
        intersectionPoints.convertTo(intersectionPoints, cv.CV_32S);
        intersectionPointsData = intersectionPoints.data32S;
        assert.deepEqual(intersectionPointsData[0], 39);
        assert.deepEqual(intersectionPointsData[1], 31);
        assert.deepEqual(intersectionPointsData[2], 49);
        assert.deepEqual(intersectionPointsData[3], 41);
        assert.deepEqual(intersectionPointsData[4], 41);
        assert.deepEqual(intersectionPointsData[5], 49);
        assert.deepEqual(intersectionPointsData[6], 31);
        assert.deepEqual(intersectionPointsData[7], 39);

        dst.delete();
        contours.delete();
        hierarchy.delete();
        intersectionPoints.delete();
    }
});

QUnit.test('warpPolar', function(assert) {
  const lines = new cv.Mat(255, 255, cv.CV_8U, new cv.Scalar(0));
  for (let r = 0; r < lines.rows; r++) {
    lines.row(r).setTo(new cv.Scalar(r));
  }
  cv.warpPolar(lines, lines, { width: 5, height: 5 }, new cv.Point(2, 2), 3,
    cv.INTER_CUBIC | cv.WARP_FILL_OUTLIERS | cv.WARP_INVERSE_MAP);
  assert.ok(lines instanceof cv.Mat);
  assert.deepEqual(Array.from(lines.data), [
    159, 172, 191, 210, 223,
    146, 159, 191, 223, 236,
    128, 128,   0,   0,   0,
    109,  96,  64,  32,  19,
     96,  83,  64,  45,  32
  ]);
});

QUnit.test('IntelligentScissorsMB', function(assert) {
  const lines = new cv.Mat(50, 100, cv.CV_8U, new cv.Scalar(0));
  lines.row(10).setTo(new cv.Scalar(255));
  assert.ok(lines instanceof cv.Mat);

  let tool = new cv.segmentation_IntelligentScissorsMB();
  tool.applyImage(lines);
  assert.ok(lines instanceof cv.Mat);
  lines.delete();

  tool.buildMap(new cv.Point(10, 10));

  let contour = new cv.Mat();
  tool.getContour(new cv.Point(50, 10), contour);
  assert.equal(contour.type(), cv.CV_32SC2);
  assert.ok(contour.total() == 41, contour.total());

  tool.getContour(new cv.Point(80, 10), contour);
  assert.equal(contour.type(), cv.CV_32SC2);
  assert.ok(contour.total() == 71, contour.total());
});
