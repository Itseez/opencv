/**
 * @file HoughLines_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables

/** General variables */
Mat src, canny_edge, sobel_edge;
Mat src_gray;
Mat standard_hough, probabilistic_hough, weighted_hough;
int min_threshold = 50;
int max_trackbar = 150;
int weightedhough_max_trackbar = 100000;

const char* standard_name = "Standard Hough Lines Demo";
const char* probabilistic_name = "Probabilistic Hough Lines Demo";
const char* weighted_name = "Weighted Hough Lines Demo";

int s_trackbar = max_trackbar;
int p_trackbar = max_trackbar;
int e_trackbar = 60;
int w_trackbar = 60000;

/// Function Headers
void help();
void Standard_Hough( int, void* );
void Probabilistic_Hough( int, void* );
void Weighted_Hough( int, void* );

/**
 * @function main
 */
int main( int argc, char** argv )
{
   // Read the image
    String imageName("building.jpg"); // by default
    if (argc > 1)
    {
        imageName = argv[1];
    }
    src = imread( samples::findFile( imageName ), IMREAD_COLOR );

   if( src.empty() )
     { help();
       return -1;
     }

   /// Pass the image to gray
   cvtColor( src, src_gray, COLOR_RGB2GRAY );

   /// Apply Canny/Sobel edge detector
   Canny( src_gray, canny_edge, 50, 200, 3 );
   Sobel( src_gray, sobel_edge, CV_16S, 1, 0 ); // dx(order of the derivative x)=1,dy=0

   /// Create Trackbars for Thresholds
   char thresh_label[50];
   snprintf( thresh_label, sizeof(thresh_label), "Thres: %d + input", min_threshold );
   namedWindow( standard_name, WINDOW_AUTOSIZE );
   createTrackbar( thresh_label, standard_name, &s_trackbar, max_trackbar, Standard_Hough );

   namedWindow( probabilistic_name, WINDOW_AUTOSIZE );
   createTrackbar( thresh_label, probabilistic_name, &p_trackbar, max_trackbar, Probabilistic_Hough );

   char edge_thresh_label[50];
   sprintf( edge_thresh_label, "Edge Thres: input" );
   namedWindow( weighted_name, WINDOW_AUTOSIZE);
   createTrackbar( edge_thresh_label, weighted_name, &e_trackbar, max_trackbar, Weighted_Hough);
   createTrackbar( thresh_label, weighted_name, &w_trackbar, weightedhough_max_trackbar, Weighted_Hough);

   /// Initialize
   Standard_Hough(0, 0);
   Probabilistic_Hough(0, 0);
   Weighted_Hough(0, 0);
   waitKey(0);
   return 0;
}

/**
 * @function help
 * @brief Indications of how to run this program and why is it for
 */
void help()
{
  printf("\t Hough Transform to detect lines \n ");
  printf("\t---------------------------------\n ");
  printf(" Usage: ./HoughLines_Demo <image_name> \n");
}

/**
 * @function Standard_Hough
 */
void Standard_Hough( int, void* )
{
  vector<Vec2f> s_lines;
  cvtColor( canny_edge, standard_hough, COLOR_GRAY2BGR );

  /// 1. Use Standard Hough Transform
  HoughLines( canny_edge, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );

  /// Show the result
  for( size_t i = 0; i < s_lines.size(); i++ )
     {
      float r = s_lines[i][0], t = s_lines[i][1];
      double cos_t = cos(t), sin_t = sin(t);
      double x0 = r*cos_t, y0 = r*sin_t;
      double alpha = 1000;

       Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
       Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
       line( standard_hough, pt1, pt2, Scalar(255,0,0), 3, LINE_AA);
     }

   imshow( standard_name, standard_hough );
}

/**
 * @function Probabilistic_Hough
 */
void Probabilistic_Hough( int, void* )
{
  vector<Vec4i> p_lines;
  cvtColor( canny_edge, probabilistic_hough, COLOR_GRAY2BGR );

  /// 2. Use Probabilistic Hough Transform
  HoughLinesP( canny_edge, p_lines, 1, CV_PI/180, min_threshold + p_trackbar, 30, 10 );

  /// Show the result
  for( size_t i = 0; i < p_lines.size(); i++ )
     {
       Vec4i l = p_lines[i];
       line( probabilistic_hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, LINE_AA);
     }

   imshow( probabilistic_name, probabilistic_hough );
}

/**
 * @function Weighted_Hough
 * This can detect lines based on the edge intensities.
 */
void Weighted_Hough( int, void* )
{
  vector<Vec2f> s_lines;

  /// prepare
  Mat edge_img;
  convertScaleAbs(sobel_edge, edge_img );
  // use same threshold for edge with Hough.
  threshold( edge_img, edge_img, e_trackbar, 255, cv::THRESH_TOZERO);
  cvtColor( edge_img, weighted_hough, COLOR_GRAY2BGR );

  /// 3. Use Weighted Hough Transform
  const bool use_edgeval{true};
  HoughLines( edge_img, s_lines, 1, CV_PI/180, min_threshold + w_trackbar, 0, 0, 0, CV_PI, use_edgeval);

  /// Show the result
  for( size_t i = 0; i < s_lines.size(); i++ )
     {
      float r = s_lines[i][0], t = s_lines[i][1];
      double cos_t = cos(t), sin_t = sin(t);
      double x0 = r*cos_t, y0 = r*sin_t;
      double alpha = 1000;

       Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
       Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
       line( weighted_hough, pt1, pt2, Scalar(255,0,0), 3, LINE_AA );
     }

   imshow( weighted_name, weighted_hough );
}
