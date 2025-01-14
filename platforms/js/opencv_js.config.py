# Classes and methods whitelist

core = {
    '': [
        'absdiff', 'add', 'addWeighted', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'cartToPolar',
        'compare', 'convertScaleAbs', 'copyMakeBorder', 'countNonZero', 'determinant', 'dft', 'divide', 'eigen',
        'exp', 'flip', 'getOptimalDFTSize','gemm', 'hconcat', 'inRange', 'invert', 'kmeans', 'log', 'magnitude',
        'max', 'mean', 'meanStdDev', 'merge', 'min', 'minMaxLoc', 'mixChannels', 'multiply', 'norm', 'normalize',
        'perspectiveTransform', 'polarToCart', 'pow', 'randn', 'randu', 'reduce', 'repeat', 'rotate', 'setIdentity', 'setRNGSeed',
        'solve', 'solvePoly', 'split', 'sqrt', 'subtract', 'trace', 'transform', 'transpose', 'vconcat',
        'setLogLevel', 'getLogLevel',
        'LUT',
    ],
    'Algorithm': [],
}

imgproc = {
    '': [
        'adaptiveThreshold',
        'applyColorMap',
        'approxPolyDP',
        'approxPolyN',
        'arcLength',
        'arrowedLine',
        'bilateralFilter',
        'blendLinear',
        'blur',
        'boundingRect',
        'boxFilter',
        'calcBackProject',
        'calcHist',
        'Canny',
        'circle',
        'clipLine',
        'compareHist',
        'connectedComponents',
        'connectedComponentsWithStats',
        'contourArea',
        'convertMaps',
        'convexHull',
        'convexityDefects',
        'cornerHarris',
        'cornerMinEigenVal',
        'createCLAHE',
        'createHanningWindow',
        'createLineSegmentDetector',
        'cvtColor',
        'demosaicing',
        'dilate',
        'distanceTransform',
        'distanceTransformWithLabels',
        'divSpectrums',
        'drawContours',
        'drawMarker',
        'ellipse',
        'ellipse2Poly',
        'equalizeHist',
        'erode',
        'fillConvexPoly',
        'fillPoly',
        'filter2D',
        'findContours',
        'findContoursLinkRuns',
        'fitEllipse',
        'fitEllipseAMS',
        'fitEllipseDirect',
        'fitLine',
        'floodFill',
        'GaussianBlur',
        'getAffineTransform',
        'getFontScaleFromHeight',
        'getPerspectiveTransform',
        'getRectSubPix',
        'getRotationMatrix2D',
        'getStructuringElement',
        'goodFeaturesToTrack',
        'grabCut',
        'HoughLines',
        'HoughLinesP',
        'HoughCircles',
        'HuMoments',
        'integral',
        'integral2',
        'intersectConvexConvex',
        'invertAffineTransform',
        'isContourConvex',
        'Laplacian',
        'line',
        'matchShapes',
        'matchTemplate',
        'medianBlur',
        'minAreaRect',
        'minEnclosingCircle',
        'minEnclosingTriangle',
        'moments',
        'morphologyEx',
        'pointPolygonTest',
        'polylines',
        'preCornerDetect',
        'putText',
        'pyrDown',
        'pyrUp',
        'rectangle',
        'remap',
        'resize',
        'rotatedRectangleIntersection',
        'Scharr',
        'sepFilter2D',
        'Sobel',
        'spatialGradient',
        'sqrBoxFilter',
        'stackBlur',
        'threshold',
        'warpAffine',
        'warpPerspective',
        'warpPolar',
        'watershed',
    ],
    'CLAHE': ['apply', 'collectGarbage', 'getClipLimit', 'getTilesGridSize', 'setClipLimit', 'setTilesGridSize'],
    'segmentation_IntelligentScissorsMB': [
        'IntelligentScissorsMB',
        'setWeights',
        'setGradientMagnitudeMaxLimit',
        'setEdgeFeatureZeroCrossingParameters',
        'setEdgeFeatureCannyParameters',
        'applyImage',
        'applyImageFeatures',
        'buildMap',
        'getContour'
    ],
}

objdetect = {'': ['getPredefinedDictionary', 'extendDictionary',
                  'drawDetectedMarkers', 'generateImageMarker', 'drawDetectedCornersCharuco',
                  'drawDetectedDiamonds'],
             'GraphicalCodeDetector': ['decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti'],
             'QRCodeDetector': ['QRCodeDetector', 'decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti', 'decodeCurved', 'detectAndDecodeCurved', 'setEpsX', 'setEpsY'],
             'aruco_PredefinedDictionaryType': [],
             'aruco_Dictionary': ['Dictionary', 'getDistanceToId', 'generateImageMarker', 'getByteListFromBits', 'getBitsFromByteList'],
             'aruco_Board': ['Board', 'matchImagePoints', 'generateImage'],
             'aruco_GridBoard': ['GridBoard', 'generateImage', 'getGridSize', 'getMarkerLength', 'getMarkerSeparation', 'matchImagePoints'],
             'aruco_CharucoParameters': ['CharucoParameters'],
             'aruco_CharucoBoard': ['CharucoBoard', 'generateImage', 'getChessboardCorners', 'getNearestMarkerCorners', 'checkCharucoCornersCollinear', 'matchImagePoints', 'getLegacyPattern', 'setLegacyPattern'],
             'aruco_DetectorParameters': ['DetectorParameters'],
             'aruco_RefineParameters': ['RefineParameters'],
             'aruco_ArucoDetector': ['ArucoDetector', 'detectMarkers', 'refineDetectedMarkers', 'setDictionary', 'setDetectorParameters', 'setRefineParameters'],
             'aruco_CharucoDetector': ['CharucoDetector', 'setBoard', 'setCharucoParameters', 'setDetectorParameters', 'setRefineParameters', 'detectBoard', 'detectDiamonds'],
             'QRCodeDetectorAruco_Params': ['Params'],
             'QRCodeDetectorAruco': ['QRCodeDetectorAruco', 'decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti', 'setDetectorParameters', 'setArucoParameters'],
             'barcode_BarcodeDetector': ['BarcodeDetector', 'decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti', 'decodeWithType', 'detectAndDecodeWithType'],
             'FaceDetectorYN': ['setInputSize', 'getInputSize', 'setScoreThreshold', 'getScoreThreshold', 'setNMSThreshold', 'getNMSThreshold',
                                'setTopK', 'getTopK', 'detect', 'create'],
}

video = {
    '': [
        'CamShift',
        'calcOpticalFlowFarneback',
        'calcOpticalFlowPyrLK',
        'createBackgroundSubtractorMOG2',
        'findTransformECC',
        'meanShift',
    ],
    'BackgroundSubtractorMOG2': ['BackgroundSubtractorMOG2', 'apply'],
    'BackgroundSubtractor': ['apply', 'getBackgroundImage'],
    # issue #21070: 'Tracker': ['init', 'update'],
    'TrackerMIL': ['create'],
    'TrackerMIL_Params': [],
}

dnn = {'dnn_Net': ['setInput', 'forward', 'setPreferableBackend','getUnconnectedOutLayersNames'],
       '': ['readNetFromCaffe', 'readNetFromTensorflow', 'readNetFromDarknet',
            'readNetFromONNX', 'readNetFromTFLite', 'readNet', 'blobFromImage']}

features = {'Feature2D': ['detect', 'compute', 'detectAndCompute', 'descriptorSize', 'descriptorType', 'defaultNorm', 'empty', 'getDefaultName'],
              'ORB': ['create', 'setMaxFeatures', 'setScaleFactor', 'setNLevels', 'setEdgeThreshold', 'setFastThreshold', 'setFirstLevel', 'setWTA_K', 'setScoreType', 'setPatchSize', 'getFastThreshold', 'getDefaultName'],
              'MSER': ['create', 'detectRegions', 'setDelta', 'getDelta', 'setMinArea', 'getMinArea', 'setMaxArea', 'getMaxArea', 'setPass2Only', 'getPass2Only', 'getDefaultName'],
              'FastFeatureDetector': ['create', 'setThreshold', 'getThreshold', 'setNonmaxSuppression', 'getNonmaxSuppression', 'setType', 'getType', 'getDefaultName'],
              'GFTTDetector': ['create', 'setMaxFeatures', 'getMaxFeatures', 'setQualityLevel', 'getQualityLevel', 'setMinDistance', 'getMinDistance', 'setBlockSize', 'getBlockSize', 'setHarrisDetector', 'getHarrisDetector', 'setK', 'getK', 'getDefaultName'],
              'SimpleBlobDetector': ['create', 'setParams', 'getParams', 'getDefaultName'],
              'SimpleBlobDetector_Params': [],
              'DescriptorMatcher': ['add', 'clear', 'empty', 'isMaskSupported', 'train', 'match', 'knnMatch', 'radiusMatch', 'clone', 'create'],
              'BFMatcher': ['isMaskSupported', 'create'],
              '': ['drawKeypoints', 'drawMatches', 'drawMatchesKnn']}

photo = {'': ['createAlignMTB', 'createCalibrateDebevec', 'createCalibrateRobertson', \
              'createMergeDebevec', 'createMergeMertens', 'createMergeRobertson', \
              'createTonemapDrago', 'createTonemapMantiuk', 'createTonemapReinhard', 'inpaint'],
        'CalibrateCRF': ['process'],
        'AlignExposures': ['process'],
        'AlignMTB' : ['calculateShift', 'shiftMat', 'computeBitmaps', 'getMaxBits', 'setMaxBits', \
                      'getExcludeRange', 'setExcludeRange', 'getCut', 'setCut'],
        'CalibrateDebevec' : ['getLambda', 'setLambda', 'getSamples', 'setSamples', 'getRandom', 'setRandom'],
        'CalibrateRobertson' : ['getMaxIter', 'setMaxIter', 'getThreshold', 'setThreshold', 'getRadiance'],
        'MergeExposures' : ['process'],
        'MergeDebevec' : ['process'],
        'MergeMertens' : ['process', 'getContrastWeight', 'setContrastWeight', 'getSaturationWeight', \
                          'setSaturationWeight', 'getExposureWeight', 'setExposureWeight'],
        'MergeRobertson' : ['process'],
        'Tonemap' : ['process' , 'getGamma', 'setGamma'],
        'TonemapDrago' : ['getSaturation', 'setSaturation', 'getBias', 'setBias', \
                          'getSigmaColor', 'setSigmaColor', 'getSigmaSpace','setSigmaSpace'],
        'TonemapMantiuk' : ['getScale', 'setScale', 'getSaturation', 'setSaturation'],
        'TonemapReinhard' : ['getIntensity', 'setIntensity', 'getLightAdaptation', 'setLightAdaptation', \
                             'getColorAdaptation', 'setColorAdaptation']
        }

_3d = {
    '': [
        'findHomography',
        'calibrateCameraExtended',
        'drawFrameAxes',
        'estimateAffine2D',
        'getDefaultNewCameraMatrix',
        'initUndistortRectifyMap',
        'Rodrigues',
        'solvePnP',
        'solvePnPRansac',
        'solvePnPRefineLM',
        'projectPoints',
        'undistort',
    ],
}

calib = {
    '': [

        # cv::fisheye namespace
        'fisheye_initUndistortRectifyMap',
        'fisheye_projectPoints',
    ],
    'UsacParams': ['UsacParams']
}


white_list = makeWhiteList([core, imgproc, objdetect, video, dnn, features, photo, _3d, calib])

# namespace_prefix_override['dnn'] = ''  # compatibility stuff (enabled by default)
# namespace_prefix_override['aruco'] = ''  # compatibility stuff (enabled by default)
