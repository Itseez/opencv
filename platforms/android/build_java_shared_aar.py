#!/usr/bin/env python

import argparse
from os import path
import os
import re
import shutil
import string
import subprocess


COPY_FROM_SDK_TO_ANDROID_PROJECT = [
    ["sdk/native/jni/include", "OpenCV/src/main/cpp/include"],
    ["sdk/java/src/org", "OpenCV/src/main/java/org"],
    ["sdk/java/res", "OpenCV/src/main/res"]
]

COPY_FROM_SDK_TO_APK = [
    ["sdk/native/libs/<ABI>/lib<LIB_NAME>.so", "jni/<ABI>/lib<LIB_NAME>.so"],
    ["sdk/native/libs/<ABI>/lib<LIB_NAME>.so", "prefab/modules/<LIB_NAME>/libs/android.<ABI>/lib<LIB_NAME>.so"],
]

ANDROID_PROJECT_TEMPLATE_DIR = path.join(path.dirname(__file__), "aar-template")
TEMP_DIR = "build_java_shared"
ANDROID_PROJECT_DIR = path.join(TEMP_DIR, "AndroidProject")
COMPILED_AAR_PATH_1 = path.join(ANDROID_PROJECT_DIR, "OpenCV/build/outputs/aar/OpenCV-release.aar") # original package name
COMPILED_AAR_PATH_2 = path.join(ANDROID_PROJECT_DIR, "OpenCV/build/outputs/aar/opencv-release.aar") # lower case package name
AAR_UNZIPPED_DIR = path.join(TEMP_DIR, "aar_unzipped")
FINAL_AAR_PATH_TEMPLATE = "outputs/opencv_java_shared_<OPENCV_VERSION>.aar"
FINAL_REPO_PATH = "outputs/maven_repo"
MAVEN_PACKAGE_NAME = "opencv"

def fill_template(src_path, dst_path, args_dict):
    with open(src_path, "r") as f:
        template_text = f.read()
    template = string.Template(template_text)
    text = template.safe_substitute(args_dict)
    with open(dst_path, "w") as f:
        f.write(text)

def get_opencv_version(opencv_sdk_path):
    version_hpp_path = path.join(opencv_sdk_path, "sdk/native/jni/include/opencv2/core/version.hpp")
    with open(version_hpp_path, "rt") as f:
        data = f.read()
        major = re.search(r'^#define\W+CV_VERSION_MAJOR\W+(\d+)$', data, re.MULTILINE).group(1)
        minor = re.search(r'^#define\W+CV_VERSION_MINOR\W+(\d+)$', data, re.MULTILINE).group(1)
        revision = re.search(r'^#define\W+CV_VERSION_REVISION\W+(\d+)$', data, re.MULTILINE).group(1)
        return "%(major)s.%(minor)s.%(revision)s" % locals()

def get_ndk_version(ndk_path):
    props_path = path.join(ndk_path, "source.properties")
    with open(props_path, "rt") as f:
        data = f.read()
        version = re.search(r'Pkg\.Revision\W+=\W+(\d+\.\d+\.\d+)', data).group(1)
        return version.strip()


def get_compiled_aar_path(path1, path2):
    if path.exists(path1):
        return path1
    elif path.exists(path2):
        return path2
    else:
        raise Exception("Can't find compiled AAR path in [" + path1 + ", " + path2 + "]")

def cleanup(paths_to_remove):
    exists = False
    for p in paths_to_remove:
        if path.exists(p):
            exists = True
            if path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            print("Removed", p)
    if not exists:
        print("Nothing to remove")

def main(args):
    opencv_version = get_opencv_version(args.opencv_sdk_path)
    ndk_version = get_ndk_version(args.ndk_location)
    print("Detected ndk_version:", ndk_version)
    abis = os.listdir(path.join(args.opencv_sdk_path, "sdk/native/libs"))
    lib_name = "opencv_java" + opencv_version.split(".")[0]
    final_aar_path = FINAL_AAR_PATH_TEMPLATE.replace("<OPENCV_VERSION>", opencv_version)

    print("Removing data from previous runs...")
    cleanup([TEMP_DIR, final_aar_path, path.join(FINAL_REPO_PATH, "org/opencv", MAVEN_PACKAGE_NAME)])

    print("Preparing Android project...")
    # ANDROID_PROJECT_TEMPLATE_DIR contains an Android project template that creates AAR
    shutil.copytree(ANDROID_PROJECT_TEMPLATE_DIR, ANDROID_PROJECT_DIR)

    # Configuring the Android project to Java + shared C++ lib version
    shutil.rmtree(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/include"))

    fill_template(path.join(ANDROID_PROJECT_DIR, "OpenCV/build.gradle.template"),
                  path.join(ANDROID_PROJECT_DIR, "OpenCV/build.gradle"),
                  {"LIB_NAME": lib_name,
                   "LIB_TYPE": "c++_shared",
                   "PACKAGE_NAME": MAVEN_PACKAGE_NAME,
                   "OPENCV_VERSION": opencv_version,
                   "NDK_VERSION": ndk_version,
                   "COMPILE_SDK": args.android_compile_sdk,
                   "MIN_SDK": args.android_min_sdk,
                   "TARGET_SDK": args.android_target_sdk,
                   "ABI_FILTERS": ", ".join(['"' + x + '"' for x in abis]),
                   "JAVA_VERSION": args.java_version,
                   })
    fill_template(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt.template"),
                  path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt"),
                  {"LIB_NAME": lib_name, "LIB_TYPE": "SHARED"})

    local_props = ""
    if args.ndk_location:
        local_props += "ndk.dir=" + args.ndk_location + "\n"
    if args.cmake_location:
        local_props += "cmake.dir=" + args.cmake_location + "\n"

    if local_props:
        with open(path.join(ANDROID_PROJECT_DIR, "local.properties"), "wt") as f:
            f.write(local_props)

    # Copying Java code and C++ public headers from SDK to the Android project
    for src, dst in COPY_FROM_SDK_TO_ANDROID_PROJECT:
        shutil.copytree(path.join(args.opencv_sdk_path, src),
                        path.join(ANDROID_PROJECT_DIR, dst))

    print("Running gradle assembleRelease...")
    # Running gradle to build the Android project
    cmd = ["./gradlew", "assembleRelease"]
    if args.offline:
        cmd = cmd + ["--offline"]
    subprocess.run(cmd, shell=False, cwd=ANDROID_PROJECT_DIR, check=True)

    print("Adding libs to AAR...")
    # The created AAR package doesn't contain C++ shared libs.
    # We need to add them manually.
    # AAR package is just a zip archive.
    complied_aar_path = get_compiled_aar_path(COMPILED_AAR_PATH_1, COMPILED_AAR_PATH_2) # two possible paths
    shutil.unpack_archive(complied_aar_path, AAR_UNZIPPED_DIR, "zip")

    for abi in abis:
        for src, dst in COPY_FROM_SDK_TO_APK:
            src = src.replace("<ABI>", abi).replace("<LIB_NAME>", lib_name)
            dst = dst.replace("<ABI>", abi).replace("<LIB_NAME>", lib_name)
            shutil.copy(path.join(args.opencv_sdk_path, src),
                path.join(AAR_UNZIPPED_DIR, dst))

    # Creating final AAR zip archive
    os.makedirs("outputs", exist_ok=True)
    shutil.make_archive(final_aar_path, "zip", AAR_UNZIPPED_DIR, ".")
    os.rename(final_aar_path + ".zip", final_aar_path)

    print("Creating local maven repo...")

    shutil.copy(final_aar_path, path.join(ANDROID_PROJECT_DIR, "OpenCV/opencv-release.aar"))

    print("Creating a maven repo from project sources (with sources jar and javadoc jar)...")
    cmd = ["./gradlew", "publishReleasePublicationToMyrepoRepository"]
    if args.offline:
        cmd = cmd + ["--offline"]
    subprocess.run(cmd, shell=False, cwd=ANDROID_PROJECT_DIR, check=True)

    os.makedirs(path.join(FINAL_REPO_PATH, "org/opencv"), exist_ok=True)
    shutil.move(path.join(ANDROID_PROJECT_DIR, "OpenCV/build/repo/org/opencv", MAVEN_PACKAGE_NAME),
                path.join(FINAL_REPO_PATH, "org/opencv", MAVEN_PACKAGE_NAME))

    print("Creating a maven repo from modified AAR (with cpp libraries)...")
    cmd = ["./gradlew", "publishModifiedPublicationToMyrepoRepository"]
    if args.offline:
        cmd = cmd + ["--offline"]
    subprocess.run(cmd, shell=False, cwd=ANDROID_PROJECT_DIR, check=True)

    # Replacing AAR from the first maven repo with modified AAR from the second maven repo
    shutil.copytree(path.join(ANDROID_PROJECT_DIR, "OpenCV/build/repo/org/opencv", MAVEN_PACKAGE_NAME),
                    path.join(FINAL_REPO_PATH, "org/opencv", MAVEN_PACKAGE_NAME),
                    dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds AAR with Java and shared C++ libs from OpenCV SDK")
    parser.add_argument('opencv_sdk_path')
    parser.add_argument('--android_compile_sdk', default="34")
    parser.add_argument('--android_min_sdk', default="21")
    parser.add_argument('--android_target_sdk', default="34")
    parser.add_argument('--java_version', default="17")
    parser.add_argument('--ndk_location', default="")
    parser.add_argument('--cmake_location', default="")
    parser.add_argument('--offline', action="store_true", help="Force Gradle use offline mode")
    args = parser.parse_args()

    main(args)
