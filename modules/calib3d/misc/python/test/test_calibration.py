#!/usr/bin/env python


'''
Camera calibration script for distorted images using chessboard samples.
This script reads distorted images, calculates the camera calibration parameters, and writes undistorted images.
'''


# Python 2/3 compatibility
from __future__ import print_function


import numpy as np
import cv2 as cv


# Import a custom test class for OpenCV (assumes the environment is set up for testing OpenCV modules)
from tests_common import NewOpenCVTests


class calibration_test(NewOpenCVTests):
    '''
    A test class for performing camera calibration and validating its accuracy.
    Inherits from NewOpenCVTests for OpenCV-specific testing utilities.
    '''


    def test_calibration(self):
        '''
        Test the camera calibration process with a set of chessboard images.
        Steps:
        1. Load sample images.
        2. Detect chessboard corners in each image.
        3. Perform camera calibration using the detected points.
        4. Validate the calibration results against predefined values.
        '''


        # List to hold chessboard image file names
        img_names = []
        for i in range(1, 15):
            if i < 10:
                # For single-digit indices, add a leading zero to the file name
                img_names.append('samples/data/left0{}.jpg'.format(str(i)))
            elif i != 10:
                # For double-digit indices, use the standard format
                img_names.append('samples/data/left{}.jpg'.format(str(i)))


        # Define chessboard parameters
        square_size = 1.0  # Size of a single square on the chessboard (arbitrary units)
        pattern_size = (9, 6)  # Number of inner corners per chessboard row and column


        # Generate 3D points for the chessboard corners (in the chessboard's coordinate space)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size


        # Arrays to store object points and image points for all images
        obj_points = []  # 3D points in the real world (chessboard corners)
        img_points = []  # 2D points in the image plane


        h, w = 0, 0  # Variables to store image height and width


        # Iterate through each image file
        for fn in img_names:
            # Load the sample image
            img = self.get_sample(fn, 0)
            if img is None:
                # Skip if the image cannot be loaded
                continue


            h, w = img.shape[:2]  # Get image dimensions


            # Detect chessboard corners in the image
            found, corners = cv.findChessboardCorners(img, pattern_size)
            if found:
                # Refine corner detection for better accuracy
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)


            if not found:
                # Skip the image if corners are not found
                continue


            # Append detected points to the lists
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)


        # Perform camera calibration
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(
            obj_points, img_points, (w, h), None, None, flags=0
        )


        # Define tolerances for validation
        eps = 0.01  # Tolerance for RMS error
        normCamEps = 10.0  # Tolerance for camera matrix
        normDistEps = 0.05  # Tolerance for distortion coefficients


        # Predefined calibration results for validation
        cameraMatrixTest = [[532.80992189, 0., 342.4952186],
                            [0., 532.93346422, 233.8879292],
                            [0., 0., 1.]]


        distCoeffsTest = [-0.281325576, 0.0291130406, 0.0012123433, -0.000140825372, 0.154865844]


        # Validate calibration results
        self.assertLess(abs(rms - 0.196334638034), eps)
        self.assertLess(cv.norm(camera_matrix - cameraMatrixTest, cv.NORM_L1), normCamEps)
        self.assertLess(cv.norm(dist_coefs - distCoeffsTest, cv.NORM_L1), normDistEps)


    def test_projectPoints(self):
        '''
        Test the projection of 3D points onto the 2D image plane using the camera matrix and distortion coefficients.
        '''
        # Define 3D object points in the world coordinate system
        objectPoints = np.array([[181.24588, 87.80361, 11.421074],
                                  [87.17948, 184.75563, 37.223446],
                                  [22.558456, 45.495266, 246.05797]], dtype=np.float32)


        # Define rotation and translation vectors (camera extrinsics)
        rvec = np.array([[0.9357548, -0.28316498, 0.21019171],
                         [0.30293274, 0.9505806, -0.06803132],
                         [-0.18054008, 0.12733458, 0.9752903]], dtype=np.float32)
        tvec = np.array([69.32692, 17.602057, 135.77672], dtype=np.float32)


        # Define camera matrix and zero distortion coefficients (for simplicity)
        cameraMatrix = np.array([[214.0047, 26.98735, 253.37799],
                                  [189.8172, 10.038101, 18.862494],
                                  [114.07123, 200.87277, 194.56332]], dtype=np.float32)
        distCoeffs = np.zeros((4, 1), dtype=np.float32)


        # Project the 3D points onto the 2D image plane
        imagePoints, jacobian = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)


        # Verify that the projection results are not None
        self.assertTrue(imagePoints is not None)
        self.assertTrue(jacobian is not None)


if __name__ == '__main__':
    # Run the tests using the OpenCV test framework
    NewOpenCVTests.bootstrap()


