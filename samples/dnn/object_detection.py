import cv2 as cv
import argparse
import sys
import numpy as np

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL)

parser = argparse.ArgumentParser(description='Use this script to run object detection deep learning networks using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', required=True,
                    help='Path to a binary file of model contains trained weights. '
                         'It could be a file with extensions .caffemodel (Caffe), '
                         '.pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet)')
parser.add_argument('--config',
                    help='Path to a text file of model contains network configuration. '
                         'It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet)')
parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet'],
                    help='Optional name of an origin framework of the model. '
                         'Detect it automatically if it does not set.')
parser.add_argument('--classes', help='Optional path to a text file with names of classes to label detected objects.')
parser.add_argument('--mean', nargs='+', type=float, default=[0, 0, 0],
                    help='Preprocess input image by subtracting mean values. '
                         'Mean values should be in BGR order.')
parser.add_argument('--scale', type=float, default=1.0,
                    help='Preprocess input image by multiplying on a scale factor.')
parser.add_argument('--width', type=int,
                    help='Preprocess input image by resizing to a specific width.')
parser.add_argument('--height', type=int,
                    help='Preprocess input image by resizing to a specific height.')
parser.add_argument('--rgb', action='store_true',
                    help='Indicate that model works with RGB input images instead BGR ones.')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                    help="Choose one of computation backends: "
                         "%d: default C++ backend, "
                         "%d: Halide language (http://halide-lang.org/), "
                         "%d: Intel's Deep Learning Inference Engine (https://software.seek.intel.com/deep-learning-deployment)" % backends)
parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                    help='Choose one of target computation devices: '
                         '%d: CPU target (by default), '
                         '%d: OpenCL' % targets)
args = parser.parse_args()

# Load names of classes
classes = None
if args.classes:
    with open(args.classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

# Load a network
modelExt = args.model[args.model.rfind('.'):]
if args.framework == 'caffe' or modelExt == '.caffemodel':
    net = cv.dnn.readNetFromCaffe(args.config, args.model)
elif args.framework == 'tensorflow' or modelExt == '.pb':
    net = cv.dnn.readNetFromTensorflow(args.model, args.config)
elif args.framework == 'torch' or modelExt in ['.t7', '.net']:
    net = cv.dnn.readNetFromTorch(args.model)
elif args.framework == 'darknet' or modelExt == '.weights':
    net = cv.dnn.readNetFromDarknet(args.config, args.model)
else:
    print('Cannot determine an origin framework of model from file %s' % args.model)
    sys.exit(0)

net.setPreferableBackend(args.backend)
net.setPreferableTarget(args.target)

confThreshold = args.thr

def postprocess(frame, out):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % confidence

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > confThreshold:
                left = int(detection[3])
                top = int(detection[4])
                right = int(detection[5])
                bottom = int(detection[6])
                classId = int(detection[1]) - 1  # Skip background label
                drawPred(classId, confidence, left, top, right, bottom)
    elif lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > confThreshold:
                left = int(detection[3] * frameWidth)
                top = int(detection[4] * frameHeight)
                right = int(detection[5] * frameWidth)
                bottom = int(detection[6] * frameHeight)
                classId = int(detection[1]) - 1  # Skip background label
                drawPred(classId, confidence, left, top, right, bottom)
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        for detection in out:
            confidences = detection[5:]
            classId = np.argmax(confidences)
            confidence = confidences[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = center_x - width / 2
                top = center_y - height / 2
                drawPred(classId, confidence, left, top, left + width, top + height)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

def callback(pos):
    global confThreshold
    confThreshold = pos / 100.0

cv.createTrackbar('Confidence threshold, %', winName, int(confThreshold * 100), 99, callback)

cap = cv.VideoCapture(args.input if args.input else 0)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Create a 4D blob from a frame.
    inpWidth = args.width if args.width else frameWidth
    inpHeight = args.height if args.height else frameHeight
    blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=False)

    # Run a model
    net.setInput(blob)
    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        frame = cv.resize(frame, (inpWidth, inpHeight))
        net.setInput(np.array([inpHeight, inpWidth, 1.6], dtype=np.float32), 'im_info');
    out = net.forward()

    postprocess(frame, out)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv.imshow(winName, frame)
