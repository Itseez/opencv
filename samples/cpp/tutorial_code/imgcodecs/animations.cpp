#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main( int argc, const char** argv )
{
    std::string filename = argc > 1 ? argv[1] : "animated_image.webp";

    //! [write_animation]
    if (argc == 1)
    {
        Animation animation_to_save;
        Mat image(128, 256, CV_8UC4, Scalar(150, 150, 150, 255));
        int duration = 200;

        for (int i = 0; i < 10; ++i) {
            animation_to_save.frames.push_back(image.clone());
            putText(animation_to_save.frames[i], format("Frame %d", i), Point(30, 80), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 100, 0, 255), 2);
            animation_to_save.durations.push_back(duration);
        }
        imwriteanimation("animated_image.webp", animation_to_save, { IMWRITE_WEBP_QUALITY, 100 });
    }
    //! [write_animation]

    //! [init_animation]
    Animation animation;
    //! [init_animation]

    //! [read_animation]
    bool success = imreadanimation(filename, animation);
    if (!success) {
        std::cerr << "Failed to load animation frames\n";
        return -1;
    }
    //! [read_animation]

    //! [show_animation]
    while (true)
    for (size_t i = 0; i < animation.frames.size(); ++i) {
        imshow("Animation", animation.frames[i]);
        int key_code = waitKey(animation.durations[i]); // Delay between frames
        if (key_code == 27)
            exit(0);
    }
    //! [show_animation]

    return 0;
}
