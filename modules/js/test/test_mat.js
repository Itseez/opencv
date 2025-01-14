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

QUnit.module('CoreMat', {});

QUnit.test('test_mat_creation', function(assert) {
    // Mat constructors.
    // Mat::Mat(int rows, int cols, int type)
    {
        let mat = new cv.Mat(10, 20, cv.CV_8UC3);

        assert.equal(mat.type(), cv.CV_8UC3);
        assert.equal(mat.depth(), cv.CV_8U);
        assert.equal(mat.channels(), 3);
        assert.ok(mat.empty() === false);

        let size = mat.size();
        assert.equal(size.height, 10);
        assert.equal(size.width, 20);

        mat.delete();
    }

    // Mat::Mat(const Mat &)
    {
        // Copy from another Mat
        let mat1 = new cv.Mat(10, 20, cv.CV_8UC3);
        let mat2 = new cv.Mat(mat1);

        assert.equal(mat2.type(), mat1.type());
        assert.equal(mat2.depth(), mat1.depth());
        assert.equal(mat2.channels(), mat1.channels());
        assert.equal(mat2.empty(), mat1.empty());

        let size1 = mat1.size;
        let size2 = mat2.size();
        assert.ok(size1[0] === size2[0]);
        assert.ok(size1[1] === size2[1]);

        mat1.delete();
        mat2.delete();
    }

    // Mat::Mat(int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
    {
        // 10 * 10 and one channel
        let data = cv._malloc(10 * 10 * 1);
        let mat = new cv.Mat(10, 10, cv.CV_8UC1, data, 0);

        assert.equal(mat.type(), cv.CV_8UC1);
        assert.equal(mat.depth(), cv.CV_8U);
        assert.equal(mat.channels(), 1);
        assert.ok(mat.empty() === false);

        let size = mat.size();
        assert.ok(size.height === 10);
        assert.ok(size.width === 10);

        mat.delete();
    }

    // Mat::Mat(int rows, int cols, int type, const Scalar& scalar)
    {
        // 2 * 2 8UC4 mat
        let mat = new cv.Mat(2, 2, cv.CV_8UC4, [0, 1, 2, 3]);

        for (let r = 0; r < mat.rows; r++) {
            for (let c = 0; c < mat.cols; c++) {
                let element = mat.ptr(r, c);
                assert.equal(element[0], 0);
                assert.equal(element[1], 1);
                assert.equal(element[2], 2);
                assert.equal(element[3], 3);
            }
        }

        mat.delete();
    }

    //  Mat::create(int, int, int)
    {
        let mat = new cv.Mat();
        mat.create(10, 5, cv.CV_8UC3);
        let size = mat.size();

        assert.ok(mat.type() === cv.CV_8UC3);
        assert.ok(size.height === 10);
        assert.ok(size.width === 5);
        assert.ok(mat.channels() === 3);

        mat.delete();
    }
    //  Mat::create(Size, int)
    {
        let mat = new cv.Mat();
        mat.create({height: 10, width: 5}, cv.CV_8UC4);
        let size = mat.size();

        assert.ok(mat.type() === cv.CV_8UC4);
        assert.ok(size.height === 10);
        assert.ok(size.width === 5);
        assert.ok(mat.channels() === 4);

        mat.delete();
    }
    //   clone
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
        let mat2 = mat.mat_clone();

        assert.equal(mat.channels, mat2.channels);
        assert.equal(mat.size().height, mat2.size().height);
        assert.equal(mat.size().width, mat2.size().width);

        assert.deepEqual(mat.data, mat2.data);


        mat.delete();
        mat2.delete();
    }
    // copyTo
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
        let mat2 = new cv.Mat();
        mat.copyTo(mat2);

        assert.equal(mat.channels, mat2.channels);
        assert.equal(mat.size().height, mat2.size().height);
        assert.equal(mat.size().width, mat2.size().width);

        assert.deepEqual(mat.data, mat2.data);


        mat.delete();
        mat2.delete();
    }
    // copyTo1
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
        let mat2 = new cv.Mat();
        let mask = new cv.Mat(5, 5, cv.CV_8UC1, new cv.Scalar(1));
        mat.copyTo(mat2, mask);

        assert.equal(mat.channels, mat2.channels);
        assert.equal(mat.size().height, mat2.size().height);
        assert.equal(mat.size().width, mat2.size().width);

        assert.deepEqual(mat.data, mat2.data);


        mat.delete();
        mat2.delete();
        mask.delete();
    }

    // matFromArray
    {
        let arrayC1 = [0, -1, 2, -3];
        let arrayC2 = [0, -1, 2, -3, 4, -5, 6, -7];
        let arrayC3 = [0, -1, 2, -3, 4, -5, 6, -7, 9, -9, 10, -11];
        let arrayC4 = [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, 13, 14, 15];

        let mat8UC1 = cv.matFromArray(2, 2, cv.CV_8UC1, arrayC1);
        let mat8UC2 = cv.matFromArray(2, 2, cv.CV_8UC2, arrayC2);
        let mat8UC3 = cv.matFromArray(2, 2, cv.CV_8UC3, arrayC3);
        let mat8UC4 = cv.matFromArray(2, 2, cv.CV_8UC4, arrayC4);

        let mat8SC1 = cv.matFromArray(2, 2, cv.CV_8SC1, arrayC1);
        let mat8SC2 = cv.matFromArray(2, 2, cv.CV_8SC2, arrayC2);
        let mat8SC3 = cv.matFromArray(2, 2, cv.CV_8SC3, arrayC3);
        let mat8SC4 = cv.matFromArray(2, 2, cv.CV_8SC4, arrayC4);

        let mat16UC1 = cv.matFromArray(2, 2, cv.CV_16UC1, arrayC1);
        let mat16UC2 = cv.matFromArray(2, 2, cv.CV_16UC2, arrayC2);
        let mat16UC3 = cv.matFromArray(2, 2, cv.CV_16UC3, arrayC3);
        let mat16UC4 = cv.matFromArray(2, 2, cv.CV_16UC4, arrayC4);

        let mat16SC1 = cv.matFromArray(2, 2, cv.CV_16SC1, arrayC1);
        let mat16SC2 = cv.matFromArray(2, 2, cv.CV_16SC2, arrayC2);
        let mat16SC3 = cv.matFromArray(2, 2, cv.CV_16SC3, arrayC3);
        let mat16SC4 = cv.matFromArray(2, 2, cv.CV_16SC4, arrayC4);

        let mat32SC1 = cv.matFromArray(2, 2, cv.CV_32SC1, arrayC1);
        let mat32SC2 = cv.matFromArray(2, 2, cv.CV_32SC2, arrayC2);
        let mat32SC3 = cv.matFromArray(2, 2, cv.CV_32SC3, arrayC3);
        let mat32SC4 = cv.matFromArray(2, 2, cv.CV_32SC4, arrayC4);

        let mat32FC1 = cv.matFromArray(2, 2, cv.CV_32FC1, arrayC1);
        let mat32FC2 = cv.matFromArray(2, 2, cv.CV_32FC2, arrayC2);
        let mat32FC3 = cv.matFromArray(2, 2, cv.CV_32FC3, arrayC3);
        let mat32FC4 = cv.matFromArray(2, 2, cv.CV_32FC4, arrayC4);

        let mat64FC1 = cv.matFromArray(2, 2, cv.CV_64FC1, arrayC1);
        let mat64FC2 = cv.matFromArray(2, 2, cv.CV_64FC2, arrayC2);
        let mat64FC3 = cv.matFromArray(2, 2, cv.CV_64FC3, arrayC3);
        let mat64FC4 = cv.matFromArray(2, 2, cv.CV_64FC4, arrayC4);

        assert.deepEqual(mat8UC1.data, new Uint8Array(arrayC1));
        assert.deepEqual(mat8UC2.data, new Uint8Array(arrayC2));
        assert.deepEqual(mat8UC3.data, new Uint8Array(arrayC3));
        assert.deepEqual(mat8UC4.data, new Uint8Array(arrayC4));

        assert.deepEqual(mat8SC1.data8S, new Int8Array(arrayC1));
        assert.deepEqual(mat8SC2.data8S, new Int8Array(arrayC2));
        assert.deepEqual(mat8SC3.data8S, new Int8Array(arrayC3));
        assert.deepEqual(mat8SC4.data8S, new Int8Array(arrayC4));

        assert.deepEqual(mat16UC1.data16U, new Uint16Array(arrayC1));
        assert.deepEqual(mat16UC2.data16U, new Uint16Array(arrayC2));
        assert.deepEqual(mat16UC3.data16U, new Uint16Array(arrayC3));
        assert.deepEqual(mat16UC4.data16U, new Uint16Array(arrayC4));

        assert.deepEqual(mat16SC1.data16S, new Int16Array(arrayC1));
        assert.deepEqual(mat16SC2.data16S, new Int16Array(arrayC2));
        assert.deepEqual(mat16SC3.data16S, new Int16Array(arrayC3));
        assert.deepEqual(mat16SC4.data16S, new Int16Array(arrayC4));

        assert.deepEqual(mat32SC1.data32S, new Int32Array(arrayC1));
        assert.deepEqual(mat32SC2.data32S, new Int32Array(arrayC2));
        assert.deepEqual(mat32SC3.data32S, new Int32Array(arrayC3));
        assert.deepEqual(mat32SC4.data32S, new Int32Array(arrayC4));

        assert.deepEqual(mat32FC1.data32F, new Float32Array(arrayC1));
        assert.deepEqual(mat32FC2.data32F, new Float32Array(arrayC2));
        assert.deepEqual(mat32FC3.data32F, new Float32Array(arrayC3));
        assert.deepEqual(mat32FC4.data32F, new Float32Array(arrayC4));

        assert.deepEqual(mat64FC1.data64F, new Float64Array(arrayC1));
        assert.deepEqual(mat64FC2.data64F, new Float64Array(arrayC2));
        assert.deepEqual(mat64FC3.data64F, new Float64Array(arrayC3));
        assert.deepEqual(mat64FC4.data64F, new Float64Array(arrayC4));

        mat8UC1.delete();
        mat8UC2.delete();
        mat8UC3.delete();
        mat8UC4.delete();
        mat8SC1.delete();
        mat8SC2.delete();
        mat8SC3.delete();
        mat8SC4.delete();
        mat16UC1.delete();
        mat16UC2.delete();
        mat16UC3.delete();
        mat16UC4.delete();
        mat16SC1.delete();
        mat16SC2.delete();
        mat16SC3.delete();
        mat16SC4.delete();
        mat32SC1.delete();
        mat32SC2.delete();
        mat32SC3.delete();
        mat32SC4.delete();
        mat32FC1.delete();
        mat32FC2.delete();
        mat32FC3.delete();
        mat32FC4.delete();
        mat64FC1.delete();
        mat64FC2.delete();
        mat64FC3.delete();
        mat64FC4.delete();
    }

    // matFromImageData
    {
        // Only test in browser
        if (typeof window === 'undefined') {
            return;
        }
        let canvas = window.document.createElement('canvas');
        canvas.width = 2;
        canvas.height = 2;
        let ctx = canvas.getContext('2d');
        ctx.fillStyle='#FF0000';
        ctx.fillRect(0, 0, 1, 1);
        ctx.fillRect(1, 1, 1, 1);

        let imageData = ctx.getImageData(0, 0, 2, 2);
        let mat = cv.matFromImageData(imageData);

        assert.deepEqual(mat.data, new Uint8Array(imageData.data));

        mat.delete();
    }

    // Mat(mat)
    {
        let mat = new cv.Mat(2, 2, cv.CV_8UC4, new cv.Scalar(1, 0, 1, 0));
        let mat1 = new cv.Mat(mat);
        let mat2 = mat;

        assert.equal(mat.rows, mat1.rows);
        assert.equal(mat.cols, mat1.cols);
        assert.equal(mat.type(), mat1.type());
        assert.deepEqual(mat.data, mat1.data);

        mat.delete();

        assert.equal(mat1.isDeleted(), false);
        assert.equal(mat2.isDeleted(), true);

        mat1.delete();
    }

    // mat.setTo
    {
        let mat = new cv.Mat(2, 2, cv.CV_8UC4);
        let s = [0, 1, 2, 3];

        mat.setTo(s);

        assert.deepEqual(mat.ptr(0, 0), new Uint8Array(s));
        assert.deepEqual(mat.ptr(0, 1), new Uint8Array(s));
        assert.deepEqual(mat.ptr(1, 0), new Uint8Array(s));
        assert.deepEqual(mat.ptr(1, 1), new Uint8Array(s));

        let s1 = [0, 0, 0, 0];
        mat.setTo(s1);
        let mask = cv.matFromArray(2, 2, cv.CV_8UC1, [0, 1, 0, 1]);
        mat.setTo(s, mask);

        assert.deepEqual(mat.ptr(0, 0), new Uint8Array(s1));
        assert.deepEqual(mat.ptr(0, 1), new Uint8Array(s));
        assert.deepEqual(mat.ptr(1, 0), new Uint8Array(s1));
        assert.deepEqual(mat.ptr(1, 1), new Uint8Array(s));

        mat.delete();
        mask.delete();
    }
});

QUnit.test('test_mat_ptr', function(assert) {
    const RValue = 3;
    const GValue = 7;
    const BValue = 197;

    // cv.CV_8UC1 + Mat::ptr(int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_8UC1);
        let view = mat.data;

        // Alter matrix[2, 1].
        let step = 10;
        view[2 * step + 1] = RValue;

        // Access matrix[2, 1].
        view = mat.ptr(2);

        assert.equal(view[1], RValue);

        mat.delete();
    }

    // cv.CV_8UC3 + Mat::ptr(int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_8UC3);
        let view = mat.data;

        // Alter matrix[2, 1].
        let step = 3 * 10;
        view[2 * step + 3] = RValue;
        view[2 * step + 3 + 1] = GValue;
        view[2 * step + 3 + 2] = BValue;

        // Access matrix[2, 1].
        view = mat.ptr(2);

        assert.equal(view[3], RValue);
        assert.equal(view[3 + 1], GValue);
        assert.equal(view[3 + 2], BValue);

        mat.delete();
    }

    // cv.CV_8UC3 + Mat::ptr(int, int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_8UC3);
        let view = mat.data;

        // Alter matrix[2, 1].
        let step = 3 * 10;
        view[2 * step + 3] = RValue;
        view[2 * step + 3 + 1] = GValue;
        view[2 * step + 3 + 2] = BValue;

        // Access matrix[2, 1].
        view = mat.ptr(2, 1);

        assert.equal(view[0], RValue);
        assert.equal(view[1], GValue);
        assert.equal(view[2], BValue);

        mat.delete();
    }

    const RValueF32 = 3.3;
    const GValueF32 = 7.3;
    const BValueF32 = 197.3;
    const EPSILON = 0.001;

    // cv.CV_32FC1 + Mat::ptr(int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_32FC1);
        let view = mat.data32F;

        // Alter matrix[2, 1].
        let step = 10;
        view[2 * step + 1] = RValueF32;

        // Access matrix[2, 1].
        view = mat.floatPtr(2);

        assert.ok(Math.abs(view[1] - RValueF32) < EPSILON);

        mat.delete();
    }

    // cv.CV_32FC3 + Mat::ptr(int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_32FC3);
        let view = mat.data32F;

        // Alter matrix[2, 1].
        let step = mat.step1(0);
        view[2 * step + 3] = RValueF32;
        view[2 * step + 3 + 1] = GValueF32;
        view[2 * step + 3 + 2] = BValueF32;

        // Access matrix[2, 1].
        view = mat.floatPtr(2);

        assert.ok(Math.abs(view[3] - RValueF32) < EPSILON);
        assert.ok(Math.abs(view[3 + 1] - GValueF32) < EPSILON);
        assert.ok(Math.abs(view[3 + 2] - BValueF32) < EPSILON);

        mat.delete();
    }

    // cv.CV_32FC3 + Mat::ptr(int, int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_32FC3);
        let view = mat.data32F;

        // Alter matrix[2, 1].
        let step = mat.step1(0);
        view[2 * step + 3] = RValueF32;
        view[2 * step + 3 + 1] = GValueF32;
        view[2 * step + 3 + 2] = BValueF32;

        // Access matrix[2, 1].
        view = mat.floatPtr(2, 1);

        assert.ok(Math.abs(view[0] - RValueF32) < EPSILON);
        assert.ok(Math.abs(view[1] - GValueF32) < EPSILON);
        assert.ok(Math.abs(view[2] - BValueF32) < EPSILON);

        mat.delete();
    }
});

QUnit.test('test_mat_zeros', function(assert) {
    let zeros = new Uint8Array(10*10).fill(0);
    // Mat::zeros(int, int, int)
    {
        let mat = cv.Mat.zeros(10, 10, cv.CV_8UC1);
        let view = mat.data;

        assert.deepEqual(view, zeros);

        mat.delete();
    }

    // Mat::zeros(Size, int)
    {
        let mat = cv.Mat.zeros({height: 10, width: 10}, cv.CV_8UC1);
        let view = mat.data;

        assert.deepEqual(view, zeros);

        mat.delete();
    }
});

QUnit.test('test_mat_ones', function(assert) {
    let ones = new Uint8Array(10*10).fill(1);
    // Mat::ones(int, int, int)
    {
        let mat = cv.Mat.ones(10, 10, cv.CV_8UC1);
        let view = mat.data;

        assert.deepEqual(view, ones);
    }
    // Mat::ones(Size, int)
    {
        let mat = cv.Mat.ones({height: 10, width: 10}, cv.CV_8UC1);
        let view = mat.data;

        assert.deepEqual(view, ones);
    }
});

QUnit.test('test_mat_eye', function(assert) {
    let eye4by4 = new Uint8Array([1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1]);
    // Mat::eye(int, int, int)
    {
        let mat = cv.Mat.eye(4, 4, cv.CV_8UC1);
        let view = mat.data;

        assert.deepEqual(view, eye4by4);
    }

    // Mat::eye(Size, int)
    {
        let mat = cv.Mat.eye({height: 4, width: 4}, cv.CV_8UC1);
        let view = mat.data;

        assert.deepEqual(view, eye4by4);
    }
});

QUnit.test('test_mat_miscs', function(assert) {
    // Mat::col(int)
    {
        let mat = cv.matFromArray(2, 2, cv.CV_8UC2, [1, 2, 3, 4, 5, 6, 7, 8]);
        let col = mat.col(1);

        assert.equal(col.isContinuous(), false);
        assert.equal(col.ptr(0, 0)[0], 3);
        assert.equal(col.ptr(0, 0)[1], 4);
        assert.equal(col.ptr(1, 0)[0], 7);
        assert.equal(col.ptr(1, 0)[1], 8);

        col.delete();
        mat.delete();
    }

    // Mat::row(int)
    {
        let mat = cv.Mat.zeros(5, 5, cv.CV_8UC2);
        let row = mat.row(1);
        let view = row.data;
        assert.equal(view[0], 0);
        assert.equal(view[4], 0);

        row.delete();
        mat.delete();
    }

    // Mat::convertTo(Mat, int, double, double)
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        let grayMat = cv.Mat.zeros(5, 5, cv.CV_8UC1);

        mat.convertTo(grayMat, cv.CV_8U, 2, 1);
        // dest = 2 * source(x, y) + 1.
        let view = grayMat.data;
        assert.equal(view[0], (1 * 2) + 1);

        mat.convertTo(grayMat, cv.CV_8U);
        // dest = 1 * source(x, y) + 0.
        assert.equal(view[0], 1);

        mat.convertTo(grayMat, cv.CV_8U, 2);
        // dest = 2 * source(x, y) + 0.
        assert.equal(view[0], 2);

        grayMat.delete();
        mat.delete();
    }

    // split
    {
        const R =7;
        const G =13;
        const B =29;

        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        let view = mat.data;
        view[0] = R;
        view[1] = G;
        view[2] = B;

        let bgrPlanes = new cv.MatVector();
        cv.split(mat, bgrPlanes);
        assert.equal(bgrPlanes.size(), 3);

        let rMat = bgrPlanes.get(0);
        view = rMat.data;
        assert.equal(view[0], R);

        let gMat = bgrPlanes.get(1);
        view = gMat.data;
        assert.equal(view[0], G);

        let bMat = bgrPlanes.get(2);
        view = bMat.data;
        assert.equal(view[0], B);

        mat.delete();
        rMat.delete();
        gMat.delete();
        bgrPlanes.delete();
        bMat.delete();
    }

    // elemSize
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        assert.equal(mat.elemSize(), 3);
        assert.equal(mat.elemSize1(), 1);

        let mat2 = cv.Mat.zeros(5, 5, cv.CV_8UC1);
        assert.equal(mat2.elemSize(), 1);
        assert.equal(mat2.elemSize1(), 1);

        let mat3 = cv.Mat.eye(5, 5, cv.CV_16UC3);
        assert.equal(mat3.elemSize(), 2 * 3);
        assert.equal(mat3.elemSize1(), 2);

        mat.delete();
        mat2.delete();
        mat3.delete();
    }

    // step
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        assert.equal(mat.step[0], 15);
        assert.equal(mat.step[1], 3);

        let mat2 = cv.Mat.zeros(5, 5, cv.CV_8UC1);
        assert.equal(mat2.step[0], 5);
        assert.equal(mat2.step[1], 1);

        let mat3 = cv.Mat.eye(5, 5, cv.CV_16UC3);
        assert.equal(mat3.step[0], 30);
        assert.equal(mat3.step[1], 6);

        mat.delete();
        mat2.delete();
        mat3.delete();
    }

    // dot
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
        let mat2 = cv.Mat.eye(5, 5, cv.CV_8UC1);

        assert.equal(mat.dot(mat), 25);
        assert.equal(mat.dot(mat2), 5);
        assert.equal(mat2.dot(mat2), 5);

        mat.delete();
        mat2.delete();
    }

    // mul
    {
        const FACTOR = 5;
        let mat = cv.Mat.ones(4, 4, cv.CV_8UC1);
        let mat2 = cv.Mat.eye(4, 4, cv.CV_8UC1);

        let expected = new Uint8Array([FACTOR, 0, 0, 0,
                                       0, FACTOR, 0, 0,
                                       0, 0, FACTOR, 0,
                                       0, 0, 0, FACTOR]);
        let mat3 = mat.mul(mat2, FACTOR);

        assert.deepEqual(mat3.data, expected);

        mat.delete();
        mat2.delete();
        mat3.delete();
    }
});


QUnit.test('test mat access', function(assert) {
    // test memory view
    {
        let data = new Uint8Array([0, 0, 0, 255, 0, 1, 2, 3]);
        let dataPtr = cv._malloc(8);

        let dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 8);
        dataHeap.set(new Uint8Array(data.buffer));

        let mat = new cv.Mat(8, 1, cv.CV_8UC1, dataPtr, 0);


        let unsignedCharView = new Uint8Array(data.buffer);
        let charView = new Int8Array(data.buffer);
        let shortView = new Int16Array(data.buffer);
        let unsignedShortView = new Uint16Array(data.buffer);
        let intView = new Int32Array(data.buffer);
        let float32View = new Float32Array(data.buffer);
        let float64View = new Float64Array(data.buffer);


        assert.deepEqual(unsignedCharView, mat.data);
        assert.deepEqual(charView, mat.data8S);
        assert.deepEqual(shortView, mat.data16S);
        assert.deepEqual(unsignedShortView, mat.data16U);
        assert.deepEqual(intView, mat.data32S);
        assert.deepEqual(float32View, mat.data32F);
        assert.deepEqual(float64View, mat.data64F);
    }

    // test ucharAt(i)
    {
        let data = new Uint8Array([0, 0, 0, 255, 0, 1, 2, 3]);
        let dataPtr = cv._malloc(8);

        let dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 8);
        dataHeap.set(new Uint8Array(data.buffer));

        let mat = new cv.Mat(8, 1, cv.CV_8UC1, dataPtr, 0);

        assert.equal(mat.ucharAt(0), 0);
        assert.equal(mat.ucharAt(1), 0);
        assert.equal(mat.ucharAt(2), 0);
        assert.equal(mat.ucharAt(3), 255);
        assert.equal(mat.ucharAt(4), 0);
        assert.equal(mat.ucharAt(5), 1);
        assert.equal(mat.ucharAt(6), 2);
        assert.equal(mat.ucharAt(7), 3);
    }

    // test ushortAt(i)
    {
        let data = new Uint16Array([0, 1000, 65000, 255, 0, 1, 2, 3]);
        let dataPtr = cv._malloc(16);

        let dataHeap = new Uint16Array(cv.HEAPU8.buffer, dataPtr, 8);
        dataHeap.set(new Uint16Array(data.buffer));

        let mat = new cv.Mat(8, 1, cv.CV_16SC1, dataPtr, 0);

        assert.equal(mat.ushortAt(0), 0);
        assert.equal(mat.ushortAt(1), 1000);
        assert.equal(mat.ushortAt(2), 65000);
        assert.equal(mat.ushortAt(3), 255);
        assert.equal(mat.ushortAt(4), 0);
        assert.equal(mat.ushortAt(5), 1);
        assert.equal(mat.ushortAt(6), 2);
        assert.equal(mat.ushortAt(7), 3);
    }

    // test intAt(i)
    {
        let data = new Int32Array([0, -1000, 65000, 255, -2000000, -1, 2, 3]);
        let dataPtr = cv._malloc(32);

        let dataHeap = new Int32Array(cv.HEAPU32.buffer, dataPtr, 8);
        dataHeap.set(new Int32Array(data.buffer));

        let mat = new cv.Mat(8, 1, cv.CV_32SC1, dataPtr, 0);

        assert.equal(mat.intAt(0), 0);
        assert.equal(mat.intAt(1), -1000);
        assert.equal(mat.intAt(2), 65000);
        assert.equal(mat.intAt(3), 255);
        assert.equal(mat.intAt(4), -2000000);
        assert.equal(mat.intAt(5), -1);
        assert.equal(mat.intAt(6), 2);
        assert.equal(mat.intAt(7), 3);
    }

    // test floatAt(i)
    {
        const EPSILON = 0.001;
        let data = new Float32Array([0, -10.5, 650.001, 255, -20.1, -1.2, 2, 3.5]);
        let dataPtr = cv._malloc(32);

        let dataHeap = new Float32Array(cv.HEAPU32.buffer, dataPtr, 8);
        dataHeap.set(new Float32Array(data.buffer));

        let mat = new cv.Mat(8, 1, cv.CV_32FC1, dataPtr, 0);

        assert.equal(Math.abs(mat.floatAt(0)-0) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(1)+10.5) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(2)-650.001) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(3)-255) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(4)+20.1) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(5)+1.2) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(6)-2) < EPSILON, true);
        assert.equal(Math.abs(mat.floatAt(7)-3.5) < EPSILON, true);
    }

    // test intAt(i,j)
    {
        let mat = cv.Mat.eye({height: 3, width: 3}, cv.CV_32SC1);

        assert.equal(mat.intAt(0, 0), 1);
        assert.equal(mat.intAt(0, 1), 0);
        assert.equal(mat.intAt(0, 2), 0);
        assert.equal(mat.intAt(1, 0), 0);
        assert.equal(mat.intAt(1, 1), 1);
        assert.equal(mat.intAt(1, 2), 0);
        assert.equal(mat.intAt(2, 0), 0);
        assert.equal(mat.intAt(2, 1), 0);
        assert.equal(mat.intAt(2, 2), 1);

        mat.delete();
    }
});

QUnit.test('test_mat_operations', function(assert) {
    // test minMaxLoc
    {
        let src = cv.Mat.ones(4, 4, cv.CV_8UC1);

        src.data[2] = 0;
        src.data[5] = 2;

        let result = cv.minMaxLoc(src);

        assert.equal(result.minVal, 0);
        assert.equal(result.maxVal, 2);
        assert.deepEqual(result.minLoc, {x: 2, y: 0});
        assert.deepEqual(result.maxLoc, {x: 1, y: 1});

        src.delete();
    }
});

QUnit.test('test_mat_roi', function(assert) {
    // test minMaxLoc
    {
        let mat = cv.matFromArray(2, 2, cv.CV_8UC1, [0, 1, 2, 3]);
        let roi = mat.roi(new cv.Rect(1, 1, 1, 1));

        assert.equal(roi.rows, 1);
        assert.equal(roi.cols, 1);
        assert.deepEqual(roi.data, new Uint8Array([mat.ucharAt(1, 1)]));

        mat.delete();
        roi.delete();
    }
});


QUnit.test('test_mat_range', function(assert) {
    {
        let src = cv.matFromArray(2, 2, cv.CV_8UC1, [0, 1, 2, 3]);
        let mat = src.colRange(0, 1);

        assert.equal(mat.isContinuous(), false);
        assert.equal(mat.rows, 2);
        assert.equal(mat.cols, 1);
        assert.equal(mat.ucharAt(0), 0);
        assert.equal(mat.ucharAt(1), 2);

        mat.delete();

        mat = src.colRange({start: 0, end: 1});

        assert.equal(mat.isContinuous(), false);
        assert.equal(mat.rows, 2);
        assert.equal(mat.cols, 1);
        assert.equal(mat.ucharAt(0), 0);
        assert.equal(mat.ucharAt(1), 2);

        mat.delete();

        mat = src.rowRange(1, 2);

        assert.equal(mat.rows, 1);
        assert.equal(mat.cols, 2);
        assert.deepEqual(mat.data, new Uint8Array([2, 3]));

        mat.delete();

        mat = src.rowRange({start: 1, end: 2});

        assert.equal(mat.rows, 1);
        assert.equal(mat.cols, 2);
        assert.deepEqual(mat.data, new Uint8Array([2, 3]));

        mat.delete();

        src.delete();
    }
});

QUnit.test('test_mat_diag', function(assert) {
    // test diag
    {
        let mat = cv.matFromArray(3, 3, cv.CV_8UC1, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let d = mat.diag();
        let d1 = mat.diag(1);
        let d2 = mat.diag(-1);

        assert.equal(mat.isContinuous(), true);
        assert.equal(d.isContinuous(), false);
        assert.equal(d1.isContinuous(), false);
        assert.equal(d2.isContinuous(), false);

        assert.equal(d.ucharAt(0), 0);
        assert.equal(d.ucharAt(1), 4);
        assert.equal(d.ucharAt(2), 8);

        assert.equal(d1.ucharAt(0), 1);
        assert.equal(d1.ucharAt(1), 5);

        assert.equal(d2.ucharAt(0), 3);
        assert.equal(d2.ucharAt(1), 7);

        mat.delete();
        d.delete();
        d1.delete();
        d2.delete();
    }
});
