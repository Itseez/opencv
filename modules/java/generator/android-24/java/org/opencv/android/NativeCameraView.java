package org.opencv.android;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.MatOfInt;

import org.opencv.imgproc.Imgproc;

import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.Log;
import android.view.ViewGroup.LayoutParams;

/**
 * This class is an implementation of a bridge between SurfaceView and OpenCV VideoCapture.
 * The class  is experimental implementation and not recoomended for production usage.
 */
public class NativeCameraView extends CameraBridgeViewBase {

    public static final String TAG = "NativeCameraView";
    private boolean mStopThread;
    private Thread mThread;

    protected VideoCapture mCamera;
    protected RotatedCameraFrame mFrame;

    public NativeCameraView(Context context, int cameraId) {
        super(context, cameraId);
    }

    public NativeCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    @Override
    protected boolean connectCamera(int width, int height) {

        /* 1. We need to instantiate camera
         * 2. We need to start thread which will be getting frames
         */
        /* First step - initialize camera connection */
        if (!initializeCamera(width, height))
            return false;

        /* now we can start update thread */
        mThread = new Thread(new CameraWorker());
        mThread.start();

        return true;
    }

    @Override
    protected void disconnectCamera() {
        /* 1. We need to stop thread which updating the frames
         * 2. Stop camera and release it
         */
        if (mThread != null) {
            try {
                mStopThread = true;
                mThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                mThread =  null;
                mStopThread = false;
            }
        }

        /* Now release camera */
        releaseCamera();
    }

    public static class OpenCvSizeAccessor implements ListItemAccessor {

        public int getWidth(Object obj) {
            Size size  = (Size)obj;
            return (int)size.width;
        }

        public int getHeight(Object obj) {
            Size size  = (Size)obj;
            return (int)size.height;
        }

    }

    private boolean initializeCamera(int width, int height) {
        synchronized (this) {
            Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
            int localCameraIndex = mCameraIndex;
            if (mCameraIndex == CAMERA_ID_ANY) {
                Log.d(TAG, "Try to open default camera");
                localCameraIndex = 0;
            } else if (mCameraIndex == CAMERA_ID_BACK) {
                Log.i(TAG, "Trying to open back camera");
                for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                    Camera.getCameraInfo( camIdx, cameraInfo );
                    if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                        localCameraIndex = camIdx;
                        break;
                    }
                }
            } else if (mCameraIndex == CAMERA_ID_FRONT) {
                Log.i(TAG, "Trying to open front camera");
                for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                    Camera.getCameraInfo( camIdx, cameraInfo );
                    if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                        localCameraIndex = camIdx;
                        break;
                    }
                }
            }

            if (localCameraIndex == CAMERA_ID_BACK) {
                Log.e(TAG, "Back camera not found!");
                return false;
            } else if (localCameraIndex == CAMERA_ID_FRONT) {
                Log.e(TAG, "Front camera not found!");
                return false;
            }

            MatOfInt params = new MatOfInt(Videoio.CAP_PROP_FRAME_WIDTH, width,
                                           Videoio.CAP_PROP_FRAME_HEIGHT, height);

            Log.d(TAG, "Try to open camera with index " + localCameraIndex);
            mCamera = new VideoCapture(localCameraIndex, Videoio.CAP_ANDROID, params);

            if (mCamera == null)
                return false;
            if (mCamera.isOpened() == false)
                return false;

            if (mCameraIndex != CAMERA_ID_BACK && mCameraIndex != CAMERA_ID_FRONT)
                Camera.getCameraInfo(localCameraIndex, cameraInfo);
            int frameRotation = getFrameRotation(
                    cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT,
                    cameraInfo.orientation);

            mFrame = new RotatedCameraFrame(new NativeCameraFrame(mCamera), frameRotation);

            if (frameRotation % 180 == 0) {
                mFrameWidth = (int) mCamera.get(Videoio.CAP_PROP_FRAME_WIDTH);
                mFrameHeight = (int) mCamera.get(Videoio.CAP_PROP_FRAME_HEIGHT);
            } else {
                mFrameWidth = (int) mCamera.get(Videoio.CAP_PROP_FRAME_HEIGHT);
                mFrameHeight = (int) mCamera.get(Videoio.CAP_PROP_FRAME_WIDTH);
            }

            if ((getLayoutParams().width == LayoutParams.MATCH_PARENT) && (getLayoutParams().height == LayoutParams.MATCH_PARENT))
                mScale = Math.min(((float)height)/mFrameHeight, ((float)width)/mFrameWidth);
            else
                mScale = 0;

            if (mFpsMeter != null) {
                mFpsMeter.setResolution(mFrameWidth, mFrameHeight);
            }

            AllocateCache();
        }

        Log.i(TAG, "Selected camera frame size = (" + mFrameWidth + ", " + mFrameHeight + ")");

        return true;
    }

    private void releaseCamera() {
        synchronized (this) {
            if (mFrame != null) {
                mFrame.mFrame.release();
                mFrame.release();
            }
            if (mCamera != null) mCamera.release();
        }
    }

    private static class NativeCameraFrame implements CvCameraViewFrame {

        @Override
        public Mat rgba() {
            mCapture.set(Videoio.CAP_PROP_FOURCC, VideoWriter.fourcc('R','G','B','4'));
            mCapture.retrieve(mRgba);
            Log.d(TAG, "Retrieved frame with size " + mRgba.cols() + "x" + mRgba.rows() + " and channels: " + mRgba.channels());
            return mRgba;
        }

        @Override
        public Mat gray() {
            mCapture.set(Videoio.CAP_PROP_FOURCC, VideoWriter.fourcc('G','R','E','Y'));
            mCapture.retrieve(mGray);
            Log.d(TAG, "Retrieved frame with size " + mGray.cols() + "x" + mGray.rows() + " and channels: " + mGray.channels());
            return mGray;
        }

        public NativeCameraFrame(VideoCapture capture) {
            mCapture = capture;
            mGray = new Mat();
            mRgba = new Mat();
        }

        @Override
        public void release() {
            if (mGray != null) mGray.release();
            if (mRgba != null) mRgba.release();
        }

        private VideoCapture mCapture;
        private Mat mRgba;
        private Mat mGray;
    };

    private class CameraWorker implements Runnable {

        public void run() {
            do {
                if (!mCamera.grab()) {
                    Log.e(TAG, "Camera frame grab failed");
                    break;
                }

                deliverAndDrawFrame(mFrame);
            } while (!mStopThread);
        }
    }

}
