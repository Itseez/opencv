@cond CUDA_MODULES
GPU-Accelerated Computer Vision (cuda module) {#tutorial_table_of_content_gpu}
=============================================

@prev_tutorial{tutorial_table_of_content_ios}

Squeeze out every little computation power from your system by using the power of your video card to
run the OpenCV algorithms.

-   @subpage tutorial_gpu_basics_similarity

    *Languages:* C++

    *Compatibility:* \> OpenCV 2.0

    *Author:* Bernát Gábor

    This will give a good grasp on how to approach coding on the GPU module, once you already know
    how to handle the other modules. As a test case it will port the similarity methods from the
    tutorial @ref tutorial_video_input_psnr_ssim to the GPU.

-   @subpage tutorial_gpu_thrust_interop

    *Languages:* C++

    *Compatibility:* \>= OpenCV 3.0

    This tutorial will show you how to wrap a GpuMat into a thrust iterator in order to be able to
    use the functions in the thrust library.
@endcond
