set(the_description "Object Detection")
ocv_define_module(objdetect
    opencv_core
    opencv_imgproc
    opencv_calib3d
    OPTIONAL
        opencv_dnn
    WRAP
        python
        java
        objc
        js
)

if(HAVE_QUIRC)
    get_property(QUIRC_INCLUDE GLOBAL PROPERTY QUIRC_INCLUDE_DIR)
    ocv_include_directories(${QUIRC_INCLUDE})
    ocv_target_link_libraries(${the_module} quirc)
endif()