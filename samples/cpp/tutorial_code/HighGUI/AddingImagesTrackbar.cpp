/**
 * @file AddingImagesTrackbar.cpp
 * @brief Simple linear blender ( dst = alpha*src1 + beta*src2 )
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using std::cout;

/** Global Variables */
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

/** Matrices to store images */
Mat src1;
Mat src2;
Mat dst;

//![on_trackbar]
/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
static void on_trackbar(int pos, void* userdata) {
    // Print trackbar position
    std::cout << "Trackbar position: " << pos << std::endl;

    // If userdata is provided, use it
    if (userdata) {
        int* user_data = static_cast<int*>(userdata);
        std::cout << "User data: " << *user_data << std::endl;
    }

    // Calculate alpha and beta values
    alpha = (double)pos / alpha_slider_max;
    beta = (1.0 - alpha);

    // Perform linear blending
    addWeighted(src1, alpha, src2, beta, 0.0, dst);

    // Display the result
    imshow("Linear Blend", dst);
}
//![on_trackbar]

/**
 * @function main
 * @brief Main function
 */
int main(void) {
      // Read images (both must be of the same size and type)
    src1 = imread(samples::findFile("LinuxLogo.jpg"));
    src2 = imread(samples::findFile("WindowsLogo.jpg"));

    if (src1.empty()) {
        cout << "Error loading src1 \n";
        return -1;
    }
    if (src2.empty()) {
        cout << "Error loading src2 \n";
        return -1;
    }

    // Initialize trackbar value
    alpha_slider = 0;

    // Create a window
    namedWindow("Linear Blend", WINDOW_AUTOSIZE);

    // Create a trackbar
    char TrackbarName[50];
    snprintf(TrackbarName, sizeof(TrackbarName), "Alpha x %d", alpha_slider_max);

    // Example userdata: Pass a pointer to an integer as userdata
    int user_value = 42; // Arbitrary value for demonstration

    createTrackbar(TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar, &user_value);

    // Show initial result
    on_trackbar(alpha_slider, nullptr);

    // Wait for user input
    waitKey(0);
    return 0;
}
