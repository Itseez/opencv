package org.opencv.samples.recorder;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.videoio.Videoio;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.Collections;
import java.util.List;

public class RecorderActivity extends CameraActivity implements CvCameraViewListener2, View.OnClickListener {
    private static final String TAG = "OCVSample::Activity";
    private static final String FILENAME_MP4 = "sample_video1.mp4";
    private static final String FILENAME_AVI = "sample_video1.avi";

    private static final int STATUS_FINISHED_PLAYBACK = 0;
    private static final int STATUS_PREVIEW = 1;
    private static final int STATUS_RECORDING = 2;
    private static final int STATUS_PLAYING = 3;
    private static final int STATUS_ERROR = 4;

    private String mVideoFilename;
    private boolean mUseBuiltInMJPG = false;

    private int mStatus = STATUS_FINISHED_PLAYBACK;
    private int mFPS = 30;
    private int mWidth = 0, mHeight = 0;

    private CameraBridgeViewBase mOpenCvCameraView;
    private ImageView mImageView;
    private Button mTriggerButton;
    private TextView mStatusTextView;
    Runnable mPlayerThread;

    private VideoWriter mVideoWriter = null;
    private VideoCapture mVideoCapture = null;
    private Mat mVideoFrame;

    public RecorderActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.recorder_surface_view);

        mStatusTextView = (TextView) findViewById(R.id.textview1);
        mStatusTextView.bringToFront();

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            mStatus = STATUS_ERROR;
            mStatusTextView.setText("Error: Can't initialize OpenCV");
            return;
        }

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.recorder_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.GONE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.disableView();

        mImageView = (ImageView) findViewById(R.id.image_view);

        mTriggerButton = (Button) findViewById(R.id.btn1);
        mTriggerButton.setOnClickListener(this);
        mTriggerButton.bringToFront();

        if (mUseBuiltInMJPG)
            mVideoFilename = getFilesDir() + "/" + FILENAME_AVI;
        else
            mVideoFilename = getFilesDir() + "/" + FILENAME_MP4;
    }

    @Override
    public void onPause()
    {
        Log.d(TAG, "Pause");
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        mImageView.setVisibility(SurfaceView.GONE);
        if (mVideoWriter != null) {
            mVideoWriter.release();
            mVideoWriter = null;
        }
        if (mVideoCapture != null) {
            mVideoCapture.release();
            mVideoCapture = null;
        }
        mStatus = STATUS_FINISHED_PLAYBACK;
        mStatusTextView.setText("Status: Finished playback");
        mTriggerButton.setText("Start Camera");

        mVideoFrame.release();
    }

    @Override
    public void onResume()
    {
        Log.d(TAG, "onResume");
        super.onResume();

        mVideoFrame = new Mat();

        changeStatus();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        Log.d(TAG, "called onDestroy");
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        if (mVideoWriter != null)
            mVideoWriter.release();
        if (mVideoCapture != null)
            mVideoCapture.release();
    }

    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "Camera view started " + String.valueOf(width) + "x" + String.valueOf(height));
        mWidth = width;
        mHeight = height;
    }

    public void onCameraViewStopped() {
        Log.d(TAG, "Camera view stopped");
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        Log.d(TAG, "Camera frame arrived");

        mVideoFrame = inputFrame.rgba();

        Log.d(TAG, "Size: " + mVideoFrame.width() + "x" + mVideoFrame.height());

        if (mVideoWriter != null && mVideoWriter.isOpened()) {
            mVideoWriter.write(mVideoFrame);
        }

        return mVideoFrame;
    }

    @Override
    public void onClick(View view) {
        Log.i(TAG,"onClick event");
        changeStatus();
    }

    public void changeStatus() {
        switch(mStatus) {
            case STATUS_ERROR:
                Toast.makeText(this, "Error", Toast.LENGTH_LONG).show();
                break;
            case STATUS_FINISHED_PLAYBACK:
                if (!startPreview()) {
                    setErrorStatus();
                    break;
                }
                mStatus = STATUS_PREVIEW;
                mStatusTextView.setText("Status: Camera preview");
                mTriggerButton.setText("Start recording");
                break;
            case STATUS_PREVIEW:
                if (!startRecording()) {
                    setErrorStatus();
                    break;
                }
                mStatus = STATUS_RECORDING;
                mStatusTextView.setText("Status: recording video");
                mTriggerButton.setText(" Stop and play video");
                break;
            case STATUS_RECORDING:
                if (!stopRecording()) {
                    setErrorStatus();
                    break;
                }
                if (!startPlayback()) {
                    setErrorStatus();
                    break;
                }
                mStatus = STATUS_PLAYING;
                mStatusTextView.setText("Status: Playing video");
                mTriggerButton.setText("Stop playback");
                break;
            case STATUS_PLAYING:
                if (!stopPlayback()) {
                    setErrorStatus();
                    break;
                }
                mStatus = STATUS_FINISHED_PLAYBACK;
                mStatusTextView.setText("Status: Finished playback");
                mTriggerButton.setText("Start Camera");
                break;
        }
    }

    public void setErrorStatus() {
        mStatus = STATUS_ERROR;
        mStatusTextView.setText("Status: Error");
    }

    public boolean startPreview() {
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setVisibility(View.VISIBLE);
        return true;
    }

    public boolean startRecording() {
        Log.i(TAG,"Starting recording");

        File file = new File(mVideoFilename);
        file.delete();

        mVideoWriter = new VideoWriter();
        if (!mUseBuiltInMJPG) {
            mVideoWriter.open(mVideoFilename, Videoio.CAP_ANDROID, VideoWriter.fourcc('H', '2', '6', '4'), mFPS, new Size(mWidth, mHeight));
            if (!mVideoWriter.isOpened()) {
                Log.i(TAG,"Can't record H264. Switching to MJPG");
                mUseBuiltInMJPG = true;
                mVideoFilename = getFilesDir() + "/" + FILENAME_AVI;
            }
        }

        if (mUseBuiltInMJPG) {
            mVideoWriter.open(mVideoFilename, VideoWriter.fourcc('M', 'J', 'P', 'G'), mFPS, new Size(mWidth, mHeight));
        }

        Log.d(TAG, "Size: " + String.valueOf(mWidth) + "x" + String.valueOf(mHeight));
        Log.d(TAG, "File: " + mVideoFilename);

        if (mVideoWriter.isOpened()) {
            Toast.makeText(this, "Record started to file " + mVideoFilename, Toast.LENGTH_LONG).show();
            return true;
        } else {
            Toast.makeText(this, "Failed to start a record", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    public boolean stopRecording() {
        Log.i(TAG, "Finishing recording");
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setVisibility(SurfaceView.GONE);
        mVideoWriter.release();
        mVideoWriter = null;
        return true;
    }

    public boolean startPlayback() {
        mImageView.setVisibility(SurfaceView.VISIBLE);

        if (!mUseBuiltInMJPG){
            mVideoCapture = new VideoCapture(mVideoFilename, Videoio.CAP_ANDROID);
        } else {
            mVideoCapture = new VideoCapture(mVideoFilename, Videoio.CAP_OPENCV_MJPEG);
        }

        if (mVideoCapture == null || !mVideoCapture.isOpened()) {
            Log.e(TAG, "Can't open video");
            Toast.makeText(this, "Can't open file " + mVideoFilename, Toast.LENGTH_SHORT).show();
            return false;
        }

        if (!mUseBuiltInMJPG){
            mVideoCapture.set(Videoio.CAP_PROP_FOURCC, VideoWriter.fourcc('R','G','B','4'));
        }

        Toast.makeText(this, "Starting playback from file " + mVideoFilename, Toast.LENGTH_SHORT).show();

        mPlayerThread = new Runnable() {
            @Override
            public void run() {
                if (mVideoCapture == null || !mVideoCapture.isOpened()) {
                    return;
                }
                mVideoCapture.read(mVideoFrame);
                if (mVideoFrame.empty()) {
                    if (mStatus == STATUS_PLAYING) {
                        changeStatus();
                    }
                    return;
                }

                // MJPEG codec will output BGR only. So we need to convert to RGBA.
                if (mUseBuiltInMJPG) {
                    Imgproc.cvtColor(mVideoFrame, mVideoFrame, Imgproc.COLOR_BGR2RGBA);
                }

                Bitmap bmp = Bitmap.createBitmap(mVideoFrame.cols(), mVideoFrame.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mVideoFrame, bmp);
                mImageView.setImageBitmap(bmp);
                Handler h = new Handler();
                h.postDelayed(this, 33);
            }
        };

        mPlayerThread.run();
        return true;
    }

    public boolean stopPlayback() {
        mVideoCapture.release();
        mVideoCapture = null;
        mImageView.setVisibility(SurfaceView.GONE);
        return true;
    }

}
