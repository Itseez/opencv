#include <opencv2/opencv.hpp>
#include "tracklet.hpp" // Include the Tracklet header file

int main() {
    // Create a sample RGB feature (a 100x100 black image)
    cv::Mat rgb_feature = cv::Mat::zeros(100, 100, CV_8UC3);

    // Create a Tracklet object
    vas::ot::Tracklet tracklet;

    // Add the RGB feature to the Tracklet
    tracklet.AddRgbFeature(rgb_feature);

    // Retrieve the RGB features from the Tracklet
    std::deque<cv::Mat> *features = tracklet.GetRgbFeatures();

    // Check if features were retrieved successfully
    if (features && !features->empty()) {
        std::cout << "RGB features found!" << std::endl;

        // Display the first RGB feature (for verification)
        cv::imshow("RGB Feature", features->front());
        cv::waitKey(0); // Wait for a key press to close the window
    } else {
        std::cout << "No RGB features found." << std::endl;
    }

    return 0;
}
