
project(${the_module}_android)

if(ANDROID_EXECUTABLE)
  set(OPENCV_JAVA_DIR "${OpenCV_BINARY_DIR}/android_sdk" CACHE INTERNAL "")
else()  # gradle
  set(OPENCV_JAVA_DIR "${ANDROID_BUILD_BASE_DIR}/opencv" CACHE INTERNAL "")
endif()
set(OPENCV_ANDROID_LIB_DIR "${OPENCV_JAVA_DIR}" CACHE INTERNAL "")  # for OpenCV samples

file(REMOVE_RECURSE "${OPENCV_JAVA_DIR}")
file(MAKE_DIRECTORY "${OPENCV_JAVA_DIR}/bin")
set(java_src_dir "${OPENCV_JAVA_DIR}/src")
file(MAKE_DIRECTORY "${java_src_dir}")

ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/java" "${java_src_dir}")

set(SOURSE_SETS_JNI_LIBS_SRC_DIRS "'native/libs'")
set(SOURSE_SETS_JAVA_SRC_DIRS "'java/src'")
set(SOURSE_SETS_RES_SRC_DIRS "'java/res'")
set(SOURSE_SETS_MANIFEST_SRC_FILE "'java/AndroidManifest.xml'")
set(BUILD_GRADLE_COMPILE_OPTIONS "
    android {
        buildFeatures {
            buildConfig true
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_${ANDROID_GRADLE_JAVA_VERSION_INIT}
        targetCompatibility JavaVersion.VERSION_${ANDROID_GRADLE_JAVA_VERSION_INIT}
    }
")
set(MAVEN_PUBLISH_PLUGIN_DECLARATION "apply plugin: 'maven-publish'")
set(BUILD_GRADLE_ANDROID_PUBLISHING_CONFIG "
    buildFeatures {
        prefabPublishing true
        buildConfig true
    }

    prefab {
        opencv_jni_shared {
            headers 'native/jni/include'
        }
    }

    publishing {
        singleVariant('release') {
            withSourcesJar()
            withJavadocJar()
        }
    }
")

set(BUILD_GRADLE_PUBLISHING_CONFIG "
publishing {
    publications {
        release(MavenPublication) {
            groupId = 'org.opencv'
            artifactId = 'opencv'
            version = '${OPENCV_VERSION_PLAIN}'

            afterEvaluate {
               from components.release
           }
        }
    }
    repositories {
        maven {
            name = 'myrepo'
            url = \"\${project.buildDir}/repo\"
        }
    }
}
")

if(ANDROID_EXECUTABLE)

ocv_assert(ANDROID_TOOLS_Pkg_Revision GREATER 13)

ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android/java" "${java_src_dir}")

# calc default SDK Target
android_get_compatible_target(ANDROID_SDK_COMPATIBLE_TARGET ${ANDROID_NATIVE_API_LEVEL} ${ANDROID_SDK_TARGET} 14)
if(ANDROID_SDK_COMPATIBLE_TARGET)
  set(ANDROID_SDK_COMPATIBLE_TARGET "${ANDROID_SDK_COMPATIBLE_TARGET}" CACHE INTERNAL "")
endif()
string(REGEX REPLACE "android-" "" android_sdk_target_num ${ANDROID_SDK_COMPATIBLE_TARGET})

if( (ANDROID_SDK_TARGET AND ANDROID_SDK_TARGET LESS 21) OR (android_sdk_target_num LESS 21) )
  message(STATUS "[OpenCV for Android SDK]: A new OpenGL Camera Bridge (CameraGLSurfaceView, CameraGLRendererBase, CameraRenderer, Camera2Renderer) is disabled, because ANDROID_SDK_TARGET (${android_sdk_target_num}) < 21")
else()
  ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android-21/java" "${java_src_dir}")
endif()

if( (ANDROID_SDK_TARGET AND ANDROID_SDK_TARGET LESS 24) OR (android_sdk_target_num LESS 24) )
  message(STATUS "[OpenCV for Android SDK]: An experiemntal Native Camera is disabled, because ANDROID_SDK_TARGET (${android_sdk_target_num}) < 24")
else()
  ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android-24/java" "${java_src_dir}")
endif()

# copy boilerplate
file(GLOB_RECURSE seed_project_files_rel RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/" "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/*")
list(REMOVE_ITEM seed_project_files_rel "${ANDROID_MANIFEST_FILE}")
foreach(file ${seed_project_files_rel})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${file}" "${OPENCV_JAVA_DIR}/${file}" @ONLY)
  list(APPEND depends "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${file}")
  get_filename_component(install_subdir "${file}" PATH)
  install(FILES "${OPENCV_JAVA_DIR}/${file}" DESTINATION "${JAVA_INSTALL_ROOT}/${install_subdir}" COMPONENT java)
endforeach()

list(APPEND depends gen_opencv_java_source "${OPENCV_DEPHELPER}/gen_opencv_java_source")
ocv_copyfiles_add_target(${the_module}_android_source_copy JAVA_SRC_COPY "Copy Java(Android SDK) source files" ${depends})
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_android_source_copy")  # force rebuild after CMake run

set(depends ${the_module}_android_source_copy "${OPENCV_DEPHELPER}/${the_module}_android_source_copy")

# generate Android library project
set(android_sdk_project_files ${ANDROID_LIB_PROJECT_FILES})  # build.xml;local.properties;proguard-project.txt;project.properties
ocv_list_add_prefix(android_sdk_project_files "${OPENCV_JAVA_DIR}/")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${ANDROID_MANIFEST_FILE}" "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" @ONLY)

add_custom_command(
    OUTPUT ${android_sdk_project_files} "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E remove ${android_sdk_project_files}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}"
    COMMAND ${ANDROID_EXECUTABLE} --silent create lib-project --path \"${OPENCV_JAVA_DIR}\" --target \"${ANDROID_SDK_COMPATIBLE_TARGET}\" --name OpenCV --package org.opencv 2>\"${CMAKE_CURRENT_BINARY_DIR}/create_lib_project.log\"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}" "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}"
    WORKING_DIRECTORY "${OPENCV_JAVA_DIR}"
    MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/${ANDROID_MANIFEST_FILE}"
    DEPENDS ${depends}
    COMMENT "Generating OpenCV Android library project. SDK target: ${lib_target_sdk_target}"
)
list(APPEND depends ${android_sdk_project_files} "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}")

install(DIRECTORY "${OPENCV_JAVA_DIR}/src" DESTINATION "${JAVA_INSTALL_ROOT}" COMPONENT java)
install(FILES "${OPENCV_JAVA_DIR}/${ANDROID_PROJECT_PROPERTIES_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT java)
install(FILES "${OPENCV_JAVA_DIR}/${ANDROID_MANIFEST_FILE}" DESTINATION ${JAVA_INSTALL_ROOT} COMPONENT java)

# build jar
set(JAR_FILE "${OpenCV_BINARY_DIR}/bin/classes.jar")
# build the library project
# normally we should do this after a native part, but for a library project we can build the java part first
add_custom_command(
    OUTPUT "${JAR_FILE}" "${OPENCV_DEPHELPER}/${the_module}_android"
    COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug -Djava.target=1.6 -Djava.source=1.6
    COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/${the_module}_android"
    WORKING_DIRECTORY "${OPENCV_JAVA_DIR}"
    DEPENDS ${depends}
    COMMENT "Building OpenCV Android library project"
)

add_custom_target(${the_module}_android DEPENDS "${OPENCV_DEPHELPER}/${the_module}_android" SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/android_lib/${ANDROID_MANIFEST_FILE}")
add_dependencies(${the_module} ${the_module}_android)

# empty 'gen' and 'res' folders
install(CODE "
FILE(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${JAVA_INSTALL_ROOT}/gen\")
FILE(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${JAVA_INSTALL_ROOT}/res\")
" COMPONENT java)

ocv_update(ANDROID_COMPILE_SDK_VERSION "27")
ocv_update(ANDROID_MIN_SDK_VERSION "14")
if(ANDROID_NATIVE_API_LEVEL GREATER 21)
  ocv_update(ANDROID_TARGET_SDK_VERSION "${ANDROID_NATIVE_API_LEVEL}")
else()
  ocv_update(ANDROID_TARGET_SDK_VERSION "21")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.gradle.in" "${CMAKE_CURRENT_BINARY_DIR}/build.gradle" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build.gradle" DESTINATION ${JAVA_INSTALL_ROOT}/.. COMPONENT java)


else()  # gradle build
#
# Android Gradle-based project
#

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.gradle.in" "${ANDROID_TMP_INSTALL_BASE_DIR}/opencv/build.gradle" @ONLY)

#TODO: INSTALL ONLY
ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android/java" "${java_src_dir}")
ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android-21/java" "${java_src_dir}")
ocv_copyfiles_append_dir(JAVA_SRC_COPY "${OPENCV_JAVA_BINDINGS_DIR}/gen/android-24/java" "${java_src_dir}")

# copy boilerplate
set(SOURSE_SETS_JNI_LIBS_SRC_DIRS "'../../jni'")
set(SOURSE_SETS_JAVA_SRC_DIRS "'src'")
set(SOURSE_SETS_RES_SRC_DIRS "'${OpenCV_SOURCE_DIR}/modules/java/android_sdk/android_gradle_lib/res'")
set(SOURSE_SETS_MANIFEST_SRC_FILE "'AndroidManifest.xml'")
set(MAVEN_PUBLISH_PLUGIN_DECLARATION "")
set(BUILD_GRADLE_ANDROID_PUBLISHING_CONFIG "")
set(BUILD_GRADLE_PUBLISHING_CONFIG "")

set(__base_dir "${CMAKE_CURRENT_SOURCE_DIR}/android_gradle_lib/")
file(GLOB_RECURSE seed_project_files_rel RELATIVE "${__base_dir}/" "${__base_dir}/*")
list(REMOVE_ITEM seed_project_files_rel "${ANDROID_MANIFEST_FILE}")
foreach(file ${seed_project_files_rel})
  configure_file("${__base_dir}/${file}" "${OPENCV_JAVA_DIR}/${file}" @ONLY)
  list(APPEND depends "${__base_dir}/${file}")
  get_filename_component(install_subdir "${file}" PATH)
  if(NOT file STREQUAL "build.gradle")
    install(FILES "${OPENCV_JAVA_DIR}/${file}" DESTINATION "${JAVA_INSTALL_ROOT}/${install_subdir}" COMPONENT java)
  endif()
endforeach()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/build.gradle.in" "${OPENCV_JAVA_DIR}/build.gradle" @ONLY)

# copy libcxx_helper
set(__base_dir "${CMAKE_CURRENT_SOURCE_DIR}/")
file(GLOB_RECURSE __files_rel RELATIVE "${__base_dir}/" "${__base_dir}/libcxx_helper/*")
foreach(file ${__files_rel})
  configure_file("${__base_dir}/${file}" "${OPENCV_JAVA_DIR}/${file}" @ONLY)
  list(APPEND depends "${__base_dir}/${file}")
  get_filename_component(install_subdir "${file}" PATH)
  install(FILES "${OPENCV_JAVA_DIR}/${file}" DESTINATION "${JAVA_INSTALL_ROOT}/../${install_subdir}" COMPONENT java)
endforeach()

list(APPEND depends gen_opencv_java_source "${OPENCV_DEPHELPER}/gen_opencv_java_source")
ocv_copyfiles_add_target(${the_module}_android_source_copy JAVA_SRC_COPY "Copy Java(Android SDK) source files" ${depends})
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_android_source_copy")  # force rebuild after CMake run

set(depends ${the_module}_android_source_copy "${OPENCV_DEPHELPER}/${the_module}_android_source_copy")

# build jar
set(AAR_FILE "${OPENCV_JAVA_DIR}/build/outputs/aar/opencv-release.aar")
add_custom_command(
    OUTPUT "${AAR_FILE}" "${OPENCV_DEPHELPER}/${the_module}_android"
    COMMAND ./gradlew ${OPENCV_GRADLE_VERBOSE_OPTIONS} "opencv:assemble"
    COMMAND ${CMAKE_COMMAND} -E touch "${OPENCV_DEPHELPER}/${the_module}_android"
    WORKING_DIRECTORY "${ANDROID_BUILD_BASE_DIR}"
    DEPENDS ${depends} ${the_module}
    COMMENT "Building OpenCV Android library project"
)
file(REMOVE "${OPENCV_DEPHELPER}/${the_module}_android")  # force rebuild after CMake run

add_custom_target(${the_module}_android ALL DEPENDS "${OPENCV_DEPHELPER}/${the_module}_android" SOURCES "${__base_dir}/${ANDROID_MANIFEST_FILE}")

install(FILES "${ANDROID_TMP_INSTALL_BASE_DIR}/opencv/build.gradle" DESTINATION ${JAVA_INSTALL_ROOT}/.. COMPONENT java)

install(DIRECTORY "${java_src_dir}" DESTINATION "${JAVA_INSTALL_ROOT}" COMPONENT java)

endif()  # gradle build
