import cv2 as cv
import numpy as np

def main(filename):
    ## [write_animation]
    if filename == "animated_image.webp":
        # Create an Animation instance to save
        animation_to_save = cv.Animation()

        # Generate a base image with a specific color
        image = np.full((128, 256, 4), (150, 150, 150, 255), dtype=np.uint8)
        duration = 200
        frames = []
        durations = []

        # Populate frames and durations in the Animation object
        for i in range(10):
            frame = image.copy()
            cv.putText(frame, f"Frame {i}", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0, 255), 2)
            frames.append(frame)
            durations.append(duration)

        animation_to_save.frames = frames
        animation_to_save.durations = durations

        # Write the animation to file
        cv.imwriteanimation(filename, animation_to_save, [cv.IMWRITE_WEBP_QUALITY, 100])
        ## [write_animation]

    ## [init_animation]
    animation = cv.Animation()
    ## [init_animation]

    ## [read_animation]
    success, animation = cv.imreadanimation(filename)
    if not success:
        print("Failed to load animation frames")
        return
    ## [read_animation]

    ## [show_animation]
    while True:
        for i, frame in enumerate(animation.frames):
            cv.imshow("Animation", frame)
            key_code = cv.waitKey(animation.durations[i])
            if key_code == 27:  # Exit if 'Esc' key is pressed
                return
    ## [show_animation]

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "animated_image.webp")
