#!/usr/bin/env python
"""
The script builds OpenCV.framework for iOS.
The built framework is universal, it can be used to build app and run it on either iOS simulator or real device.

Usage:
    ./build_framework.py <outputdir>

By cmake conventions (and especially if you work with OpenCV repository),
the output dir should not be a subdirectory of OpenCV source tree.

Script will create <outputdir>, if it's missing, and a few its subdirectories:

    <outputdir>
        build/
            iPhoneOS-*/
               [cmake-generated build tree for an iOS device target]
            iPhoneSimulator-*/
               [cmake-generated build tree for iOS simulator]
        {framework_name}.framework/
            [the framework content]
        samples/
            [sample projects]
        docs/
            [documentation]

The script should handle minor OpenCV updates efficiently
- it does not recompile the library from scratch each time.
However, {framework_name}.framework directory is erased and recreated on each run.

Adding --dynamic parameter will build {framework_name}.framework as App Store dynamic framework. Only iOS 8+ versions are supported.
"""

from __future__ import print_function, unicode_literals
import glob, os, os.path, shutil, string, sys, argparse, traceback, multiprocessing, codecs, io
from subprocess import check_call, check_output, CalledProcessError

if sys.version_info >= (3, 8): # Python 3.8+
    def copy_tree(src, dst):
        shutil.copytree(src, dst, dirs_exist_ok=True)
else:
    from distutils.dir_util import copy_tree

sys.path.insert(0, os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../apple'))
from cv_build_utils import execute, print_error, get_xcode_major, get_xcode_setting, get_xcode_version, get_cmake_version

IPHONEOS_DEPLOYMENT_TARGET='9.0'  # default, can be changed via command line options or environment variable

CURRENT_FILE_DIR = os.path.dirname(__file__)


class Builder:
    def __init__(self, opencv, contrib, dynamic, bitcodedisabled, exclude, disable, enablenonfree, targets, debug, debug_info, framework_name, run_tests, build_docs, swiftdisabled):
        self.opencv = os.path.abspath(opencv)
        self.contrib = None
        if contrib:
            modpath = os.path.join(contrib, "modules")
            if os.path.isdir(modpath):
                self.contrib = os.path.abspath(modpath)
            else:
                print("Note: contrib repository is bad - modules subfolder not found", file=sys.stderr)
        self.dynamic = dynamic
        self.bitcodedisabled = bitcodedisabled
        self.exclude = exclude
        self.build_objc_wrapper = not "objc" in self.exclude
        self.disable = disable
        self.enablenonfree = enablenonfree
        self.targets = targets
        self.debug = debug
        self.debug_info = debug_info
        self.framework_name = framework_name
        self.run_tests = run_tests
        self.build_docs = build_docs
        self.swiftdisabled = swiftdisabled

    def checkCMakeVersion(self):
        if get_xcode_version() >= (12, 2):
            assert get_cmake_version() >= (3, 19), "CMake 3.19 or later is required when building with Xcode 12.2 or greater. Current version is {}".format(get_cmake_version())
        else:
            assert get_cmake_version() >= (3, 17), "CMake 3.17 or later is required. Current version is {}".format(get_cmake_version())

    def getBuildDir(self, parent, target):

        res = os.path.join(parent, 'build-%s-%s' % (target[0].lower(), target[1].lower()))

        if not os.path.isdir(res):
            os.makedirs(res)
        return os.path.abspath(res)

    def _build(self, outdir):
        self.checkCMakeVersion()
        outdir = os.path.abspath(outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        main_working_dir = os.path.join(outdir, "build")
        dirs = []

        xcode_ver = get_xcode_major()

        # build each architecture separately
        alltargets = []

        for target_group in self.targets:
            for arch in target_group[0]:
                current = ( arch, target_group[1] )
                alltargets.append(current)

        for target in alltargets:
            main_build_dir = self.getBuildDir(main_working_dir, target)
            dirs.append(main_build_dir)

            cmake_flags = []
            if self.contrib:
                cmake_flags.append("-DOPENCV_EXTRA_MODULES_PATH=%s" % self.contrib)
            if xcode_ver >= 7 and target[1] == 'iPhoneOS' and self.bitcodedisabled == False:
                cmake_flags.append("-DCMAKE_C_FLAGS=-fembed-bitcode")
                cmake_flags.append("-DCMAKE_CXX_FLAGS=-fembed-bitcode")
            if xcode_ver >= 7 and target[1] == 'Catalyst':
                sdk_path = check_output(["xcodebuild", "-version", "-sdk", "macosx", "Path"]).decode('utf-8').rstrip()
                c_flags = [
                    "-target %s-apple-ios14.0-macabi" % target[0],  # e.g. x86_64-apple-ios13.2-macabi # -mmacosx-version-min=10.15
                    "-isysroot %s" % sdk_path,
                    "-iframework %s/System/iOSSupport/System/Library/Frameworks" % sdk_path,
                    "-isystem %s/System/iOSSupport/usr/include" % sdk_path,
                ]
                if self.bitcodedisabled == False:
                    c_flags.append("-fembed-bitcode")
                cmake_flags.append("-DCMAKE_C_FLAGS=" + " ".join(c_flags))
                cmake_flags.append("-DCMAKE_CXX_FLAGS=" + " ".join(c_flags))
                cmake_flags.append("-DCMAKE_EXE_LINKER_FLAGS=" + " ".join(c_flags))

                # CMake cannot compile Swift for Catalyst https://gitlab.kitware.com/cmake/cmake/-/issues/21436
                # cmake_flags.append("-DCMAKE_Swift_FLAGS=" + " " + target_flag)
                cmake_flags.append("-DSWIFT_DISABLED=1")

                cmake_flags.append("-DIOS=1")  # Build the iOS codebase
                cmake_flags.append("-DMAC_CATALYST=1")  # Set a flag for Mac Catalyst, just in case we need it
                cmake_flags.append("-DWITH_OPENCL=OFF")  # Disable OpenCL; it isn't compatible with iOS
                cmake_flags.append("-DCMAKE_OSX_SYSROOT=%s" % sdk_path)
                cmake_flags.append("-DCMAKE_CXX_COMPILER_WORKS=TRUE")
                cmake_flags.append("-DCMAKE_C_COMPILER_WORKS=TRUE")
            self.buildOne(target[0], target[1], main_build_dir, cmake_flags)

            if not self.dynamic:
                self.mergeLibs(main_build_dir)
            else:
                self.makeDynamicLib(main_build_dir)
        self.makeFramework(outdir, dirs)
        if self.build_objc_wrapper:
            if self.run_tests:
                check_call([sys.argv[0].replace("build_framework", "run_tests"), "--framework_dir=" + outdir, "--framework_name=" + self.framework_name, dirs[0] +  "/modules/objc_bindings_generator/{}/test".format(self.getObjcTarget(target[1]))])
            else:
                print("To run tests call:")
                print(sys.argv[0].replace("build_framework", "run_tests") + " --framework_dir=" + outdir + " --framework_name=" + self.framework_name + " " + dirs[0] +  "/modules/objc_bindings_generator/{}/test".format(self.getObjcTarget(target[1])))
            if self.build_docs:
                check_call([sys.argv[0].replace("build_framework", "build_docs"), dirs[0] + "/modules/objc/framework_build"])
                doc_path = os.path.join(dirs[0], "modules", "objc", "doc_build", "docs")
                if os.path.exists(doc_path):
                    shutil.copytree(doc_path, os.path.join(outdir, "docs"))
                    shutil.copyfile(os.path.join(self.opencv, "doc", "opencv.ico"), os.path.join(outdir, "docs", "favicon.ico"))
            else:
                print("To build docs call:")
                print(sys.argv[0].replace("build_framework", "build_docs") + " " + dirs[0] + "/modules/objc/framework_build")
            self.copy_samples(outdir)
            if self.swiftdisabled:
                swift_sources_dir = os.path.join(outdir, "SwiftSources")
                if not os.path.exists(swift_sources_dir):
                    os.makedirs(swift_sources_dir)
                for root, dirs, files in os.walk(dirs[0]):
                    for file in files:
                        if file.endswith(".swift") and file.find("Test") == -1:
                            with io.open(os.path.join(root, file), encoding="utf-8", errors="ignore") as file_in:
                                body = file_in.read()
                            if body.find("import Foundation") != -1:
                                insert_pos = body.find("import Foundation") + len("import Foundation") + 1
                                body = body[:insert_pos] + "import " + self.framework_name + "\n" + body[insert_pos:]
                            else:
                                body = "import " + self.framework_name + "\n\n" + body
                            with codecs.open(os.path.join(swift_sources_dir, file), "w", "utf-8") as file_out:
                                file_out.write(body)

    def build(self, outdir):
        try:
            self._build(outdir)
        except Exception as e:
            print_error(e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    def getToolchain(self, arch, target):
        return None

    def getConfiguration(self):
        return "Debug" if self.debug else "Release"

    def getCMakeArgs(self, arch, target):

        args = [
            "cmake",
            "-GXcode",
            "-DAPPLE_FRAMEWORK=ON",
            "-DCMAKE_INSTALL_PREFIX=install",
            "-DCMAKE_BUILD_TYPE=%s" % self.getConfiguration(),
            "-DOPENCV_INCLUDE_INSTALL_PATH=include",
            "-DOPENCV_3P_LIB_INSTALL_PATH=lib/3rdparty",
            "-DFRAMEWORK_NAME=%s" % self.framework_name,
        ]
        if self.dynamic:
            args += [
                "-DDYNAMIC_PLIST=ON"
            ]
        if self.enablenonfree:
            args += [
                "-DOPENCV_ENABLE_NONFREE=ON"
            ]
        if self.debug_info:
            args += [
                "-DBUILD_WITH_DEBUG_INFO=ON"
            ]

        if len(self.exclude) > 0:
            args += ["-DBUILD_opencv_%s=OFF" % m for m in self.exclude]

        if len(self.disable) > 0:
            args += ["-DWITH_%s=OFF" % f for f in self.disable]

        return args

    def getBuildCommand(self, arch, target):

        buildcmd = [
            "xcodebuild",
        ]

        if (self.dynamic or self.build_objc_wrapper) and not self.bitcodedisabled and target == "iPhoneOS":
            buildcmd.append("BITCODE_GENERATION_MODE=bitcode")

        buildcmd += [
            "IPHONEOS_DEPLOYMENT_TARGET=" + os.environ['IPHONEOS_DEPLOYMENT_TARGET'],
            "ARCHS=%s" % arch,
            "-sdk", target.lower(),
            "-configuration", self.getConfiguration(),
            "-parallelizeTargets",
            "-jobs", str(multiprocessing.cpu_count()),
        ]

        return buildcmd

    def getInfoPlist(self, builddirs):
        return os.path.join(builddirs[0], "ios", "Info.plist")

    def getObjcTarget(self, target):
        # Obj-C generation target
        return 'ios'

    def makeCMakeCmd(self, arch, target, dir, cmakeargs = []):
        toolchain = self.getToolchain(arch, target)
        cmakecmd = self.getCMakeArgs(arch, target) + \
            (["-DCMAKE_TOOLCHAIN_FILE=%s" % toolchain] if toolchain is not None else [])
        if target.lower().startswith("iphoneos") or target.lower().startswith("xros"):
            cmakecmd.append("-DCPU_BASELINE=DETECT")
        if target.lower().startswith("iphonesimulator") or target.lower().startswith("xrsimulator"):
            build_arch = check_output(["uname", "-m"]).decode('utf-8').rstrip()
            if build_arch != arch:
                print("build_arch (%s) != arch (%s)" % (build_arch, arch))
                cmakecmd.append("-DCMAKE_SYSTEM_PROCESSOR=" + arch)
                cmakecmd.append("-DCMAKE_OSX_ARCHITECTURES=" + arch)
                cmakecmd.append("-DCPU_BASELINE=DETECT")
                cmakecmd.append("-DCMAKE_CROSSCOMPILING=ON")
                cmakecmd.append("-DOPENCV_WORKAROUND_CMAKE_20989=ON")
        if target.lower() == "catalyst":
            build_arch = check_output(["uname", "-m"]).decode('utf-8').rstrip()
            if build_arch != arch:
                print("build_arch (%s) != arch (%s)" % (build_arch, arch))
                cmakecmd.append("-DCMAKE_SYSTEM_PROCESSOR=" + arch)
                cmakecmd.append("-DCMAKE_OSX_ARCHITECTURES=" + arch)
                cmakecmd.append("-DCPU_BASELINE=DETECT")
                cmakecmd.append("-DCMAKE_CROSSCOMPILING=ON")
                cmakecmd.append("-DOPENCV_WORKAROUND_CMAKE_20989=ON")
        if target.lower() == "macosx":
            build_arch = check_output(["uname", "-m"]).decode('utf-8').rstrip()
            if build_arch != arch:
                print("build_arch (%s) != arch (%s)" % (build_arch, arch))
                cmakecmd.append("-DCMAKE_SYSTEM_PROCESSOR=" + arch)
                cmakecmd.append("-DCMAKE_OSX_ARCHITECTURES=" + arch)
                cmakecmd.append("-DCPU_BASELINE=DETECT")
                cmakecmd.append("-DCMAKE_CROSSCOMPILING=ON")
                cmakecmd.append("-DOPENCV_WORKAROUND_CMAKE_20989=ON")

        cmakecmd.append(dir)
        cmakecmd.extend(cmakeargs)
        return cmakecmd

    def buildOne(self, arch, target, builddir, cmakeargs = []):
        # Run cmake
        #toolchain = self.getToolchain(arch, target)
        #cmakecmd = self.getCMakeArgs(arch, target) + \
        #    (["-DCMAKE_TOOLCHAIN_FILE=%s" % toolchain] if toolchain is not None else [])
        #if target.lower().startswith("iphoneos"):
        #    cmakecmd.append("-DCPU_BASELINE=DETECT")
        #cmakecmd.append(self.opencv)
        #cmakecmd.extend(cmakeargs)
        cmakecmd = self.makeCMakeCmd(arch, target, self.opencv, cmakeargs)
        print("")
        print("=================================")
        print("CMake")
        print("=================================")
        print("")
        execute(cmakecmd, cwd = builddir)
        print("")
        print("=================================")
        print("Xcodebuild")
        print("=================================")
        print("")

        # Clean and build
        clean_dir = os.path.join(builddir, "install")
        if os.path.isdir(clean_dir):
            shutil.rmtree(clean_dir)
        buildcmd = self.getBuildCommand(arch, target)
        execute(buildcmd + ["-target", "ALL_BUILD", "build"], cwd = builddir)
        execute(["cmake", "-DBUILD_TYPE=%s" % self.getConfiguration(), "-P", "cmake_install.cmake"], cwd = builddir)
        if self.build_objc_wrapper:
            cmakecmd = self.makeCMakeCmd(arch, target, builddir + "/modules/objc_bindings_generator/{}/gen".format(self.getObjcTarget(target)), cmakeargs)
            if self.swiftdisabled:
                cmakecmd.append("-DSWIFT_DISABLED=1")
            cmakecmd.append("-DBUILD_ROOT=%s" % builddir)
            cmakecmd.append("-DCMAKE_INSTALL_NAME_TOOL=install_name_tool")
            cmakecmd.append("--no-warn-unused-cli")
            execute(cmakecmd, cwd = builddir + "/modules/objc/framework_build")

            execute(buildcmd + ["-target", "ALL_BUILD", "build"], cwd = builddir + "/modules/objc/framework_build")
            execute(["cmake", "-DBUILD_TYPE=%s" % self.getConfiguration(), "-DCMAKE_INSTALL_PREFIX=%s" % (builddir + "/install"), "-P", "cmake_install.cmake"], cwd = builddir + "/modules/objc/framework_build")

    def mergeLibs(self, builddir):
        res = os.path.join(builddir, "lib", self.getConfiguration(), "libopencv_merged.a")
        libs = glob.glob(os.path.join(builddir, "install", "lib", "*.a"))
        module = [os.path.join(builddir, "install", "lib", self.framework_name + ".framework", self.framework_name)] if self.build_objc_wrapper else []

        libs3 = glob.glob(os.path.join(builddir, "install", "lib", "3rdparty", "*.a"))
        print("Merging libraries:\n\t%s" % "\n\t".join(libs + libs3 + module), file=sys.stderr)
        execute(["libtool", "-static", "-o", res] + libs + libs3 + module)

    def makeDynamicLib(self, builddir):
        target = builddir[(builddir.rfind("build-") + 6):]
        target_platform = target[(target.rfind("-") + 1):]
        is_device = target_platform == "iphoneos" or target_platform == "visionos" or target_platform == "catalyst"
        framework_dir = os.path.join(builddir, "install", "lib", self.framework_name + ".framework")
        if not os.path.exists(framework_dir):
            os.makedirs(framework_dir)
        res = os.path.join(framework_dir, self.framework_name)
        libs = glob.glob(os.path.join(builddir, "install", "lib", "*.a"))
        if self.build_objc_wrapper:
            module = [os.path.join(builddir, "lib", self.getConfiguration(), self.framework_name + ".framework", self.framework_name)]
        else:
            module = []

        libs3 = glob.glob(os.path.join(builddir, "install", "lib", "3rdparty", "*.a"))

        if os.environ.get('IPHONEOS_DEPLOYMENT_TARGET'):
            link_target = target[:target.find("-")] + "-apple-ios" + os.environ['IPHONEOS_DEPLOYMENT_TARGET'] + ("-simulator" if target.endswith("simulator") else "")
        else:
            if target_platform == "catalyst":
                link_target = "%s-apple-ios14.0-macabi" % target[:target.find("-")]
            else:
                link_target = "%s-apple-darwin" % target[:target.find("-")]
        bitcode_flags = ["-fembed-bitcode", "-Xlinker", "-bitcode_verify"] if is_device and not self.bitcodedisabled else []
        toolchain_dir = get_xcode_setting("TOOLCHAIN_DIR", builddir)
        sdk_dir = get_xcode_setting("SDK_DIR", builddir)
        framework_options = []
        swift_link_dirs = ["-L" + toolchain_dir + "/usr/lib/swift/" + target_platform, "-L/usr/lib/swift"]
        if target_platform == "catalyst":
            swift_link_dirs = ["-L" + toolchain_dir + "/usr/lib/swift/" + "maccatalyst", "-L/usr/lib/swift"]
            framework_options = [
                "-iframework", "%s/System/iOSSupport/System/Library/Frameworks" % sdk_dir,
                "-framework", "AVFoundation", "-framework", "UIKit", "-framework", "CoreGraphics",
                "-framework", "CoreImage", "-framework", "CoreMedia", "-framework", "QuartzCore",
            ]
        elif target_platform == "macosx":
            framework_options = [
                "-framework", "AVFoundation", "-framework", "AppKit", "-framework", "CoreGraphics",
                "-framework", "CoreImage", "-framework", "CoreMedia", "-framework", "QuartzCore",
                "-framework", "Accelerate", "-framework", "OpenCL",
            ]
        elif target_platform == "iphoneos" or target_platform == "iphonesimulator" or  target_platform == "xros" or target_platform == "xrsimulator":
            framework_options = [
                "-iframework", "%s/System/iOSSupport/System/Library/Frameworks" % sdk_dir,
                "-framework", "AVFoundation", "-framework", "CoreGraphics",
                "-framework", "CoreImage", "-framework", "CoreMedia", "-framework", "QuartzCore",
                "-framework", "Accelerate", "-framework", "UIKit", "-framework", "CoreVideo",
            ]
        execute([
            "clang++",
            "-Xlinker", "-rpath",
            "-Xlinker", "/usr/lib/swift",
            "-target", link_target,
            "-isysroot", sdk_dir,] +
            framework_options + [
            "-install_name", "@rpath/" + self.framework_name + ".framework/" + self.framework_name,
            "-dynamiclib", "-dead_strip", "-fobjc-link-runtime", "-all_load",
            "-o", res
        ] + swift_link_dirs + bitcode_flags + module + libs + libs3)

    def makeFramework(self, outdir, builddirs):
        name = self.framework_name

        # set the current dir to the dst root
        framework_dir = os.path.join(outdir, "%s.framework" % name)
        if os.path.isdir(framework_dir):
            shutil.rmtree(framework_dir)
        os.makedirs(framework_dir)

        dstdir = os.path.join(framework_dir, "Versions", "A")

        # copy headers from one of build folders
        shutil.copytree(os.path.join(builddirs[0], "install", "include", "opencv2"), os.path.join(dstdir, "Headers"))
        if name != "opencv2":
            for dirname, dirs, files in os.walk(os.path.join(dstdir, "Headers")):
                for filename in files:
                    filepath = os.path.join(dirname, filename)
                    with codecs.open(filepath, "r", "utf-8") as file:
                        body = file.read()
                    body = body.replace("include \"opencv2/", "include \"" + name + "/")
                    body = body.replace("include <opencv2/", "include <" + name + "/")
                    with codecs.open(filepath, "w", "utf-8") as file:
                        file.write(body)
        if self.build_objc_wrapper:
            copy_tree(os.path.join(builddirs[0], "install", "lib", name + ".framework", "Headers"), os.path.join(dstdir, "Headers"))
            platform_name_map = {
                    "arm": "armv7-apple-ios",
                    "arm64": "arm64-apple-ios",
                    "i386": "i386-apple-ios-simulator",
                    "x86_64": "x86_64-apple-ios-simulator",
                } if builddirs[0].find("iphone") != -1 else {
                    "x86_64": "x86_64-apple-macos",
                    "arm64": "arm64-apple-macos",
                }
            for d in builddirs:
                copy_tree(os.path.join(d, "install", "lib", name + ".framework", "Modules"), os.path.join(dstdir, "Modules"))
            for dirname, dirs, files in os.walk(os.path.join(dstdir, "Modules")):
                for filename in files:
                    filestem = os.path.splitext(filename)[0]
                    fileext = os.path.splitext(filename)[1]
                    if filestem in platform_name_map:
                        os.rename(os.path.join(dirname, filename), os.path.join(dirname, platform_name_map[filestem] + fileext))

        # make universal static lib
        if self.dynamic:
            libs = [os.path.join(d, "install", "lib", name + ".framework", name) for d in builddirs]
        else:
            libs = [os.path.join(d, "lib", self.getConfiguration(), "libopencv_merged.a") for d in builddirs]
        lipocmd = ["lipo", "-create"]
        lipocmd.extend(libs)
        lipocmd.extend(["-o", os.path.join(dstdir, name)])
        print("Creating universal library from:\n\t%s" % "\n\t".join(libs), file=sys.stderr)
        execute(lipocmd)

        # copy Info.plist
        resdir = os.path.join(dstdir, "Resources")
        os.makedirs(resdir)
        shutil.copyfile(self.getInfoPlist(builddirs), os.path.join(resdir, "Info.plist"))

        # make symbolic links
        links = [
            (["A"], ["Versions", "Current"]),
            (["Versions", "Current", "Headers"], ["Headers"]),
            (["Versions", "Current", "Resources"], ["Resources"]),
            (["Versions", "Current", "Modules"], ["Modules"]),
            (["Versions", "Current", name], [name])
        ]
        for l in links:
            s = os.path.join(*l[0])
            d = os.path.join(framework_dir, *l[1])
            os.symlink(s, d)

        # Copy Apple privacy manifest
        shutil.copyfile(os.path.join(CURRENT_FILE_DIR, "PrivacyInfo.xcprivacy"),
                        os.path.join(resdir, "PrivacyInfo.xcprivacy"))

    def copy_samples(self, outdir):
        return

class iOSBuilder(Builder):

    def getToolchain(self, arch, target):
        toolchain = os.path.join(self.opencv, "platforms", "ios", "cmake", "Toolchains", "Toolchain-%s_Xcode.cmake" % target)
        return toolchain

    def getCMakeArgs(self, arch, target):
        args = Builder.getCMakeArgs(self, arch, target)
        args = args + [
            '-DIOS_ARCH=%s' % arch
        ]
        return args

    def copy_samples(self, outdir):
        print('Copying samples to: ' + outdir)
        samples_dir = os.path.join(outdir, "samples")
        if os.path.exists(samples_dir):
            shutil.rmtree(samples_dir)
        shutil.copytree(os.path.join(self.opencv, "samples", "swift", "ios"), samples_dir)
        if self.framework_name != "OpenCV":
            for dirname, dirs, files in os.walk(samples_dir):
                for filename in files:
                    if not filename.endswith((".h", ".swift", ".pbxproj")):
                        continue
                    filepath = os.path.join(dirname, filename)
                    with open(filepath) as file:
                        body = file.read()
                    body = body.replace("import OpenCV", "import " + self.framework_name)
                    body = body.replace("#import <OpenCV/OpenCV.h>", "#import <" + self.framework_name + "/" + self.framework_name + ".h>")
                    body = body.replace("OpenCV.framework", self.framework_name + ".framework")
                    body = body.replace("../../OpenCV/**", "../../" + self.framework_name + "/**")
                    with open(filepath, "w") as file:
                        file.write(body)


if __name__ == "__main__":
    folder = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "../.."))
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for iOS.')
    # TODO: When we can make breaking changes, we should make the out argument explicit and required like in build_xcframework.py.
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built framework')
    parser.add_argument('--opencv', metavar='DIR', default=folder, help='folder with opencv repository (default is "../.." relative to script location)')
    parser.add_argument('--contrib', metavar='DIR', default=None, help='folder with opencv_contrib repository (default is "None" - build only main framework)')
    parser.add_argument('--without', metavar='MODULE', default=[], action='append', help='OpenCV modules to exclude from the framework. To exclude multiple, specify this flag again, e.g. "--without video --without objc"')
    parser.add_argument('--disable', metavar='FEATURE', default=[], action='append', help='OpenCV features to disable (add WITH_*=OFF). To disable multiple, specify this flag again, e.g. "--disable tbb --disable openmp"')
    parser.add_argument('--dynamic', default=False, action='store_true', help='build dynamic framework (default is "False" - builds static framework)')
    parser.add_argument('--disable-bitcode', default=False, dest='bitcodedisabled', action='store_true', help='disable bitcode (enabled by default)')
    parser.add_argument('--iphoneos_deployment_target', default=os.environ.get('IPHONEOS_DEPLOYMENT_TARGET', IPHONEOS_DEPLOYMENT_TARGET), help='specify IPHONEOS_DEPLOYMENT_TARGET')
    parser.add_argument('--build_only_specified_archs', default=False, action='store_true', help='if enabled, only directly specified archs are built and defaults are ignored')
    parser.add_argument('--iphoneos_archs', default=None, help='select iPhoneOS target ARCHS. Default is "armv7,armv7s,arm64"')
    parser.add_argument('--iphonesimulator_archs', default=None, help='select iPhoneSimulator target ARCHS. Default is "i386,x86_64"')
    parser.add_argument('--enable_nonfree', default=False, dest='enablenonfree', action='store_true', help='enable non-free modules (disabled by default)')
    parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='Build "Debug" binaries (disabled by default)')
    parser.add_argument('--debug_info', default=False, dest='debug_info', action='store_true', help='Build with debug information (useful for Release mode: BUILD_WITH_DEBUG_INFO=ON)')
    parser.add_argument('--framework_name', default='opencv2', dest='framework_name', help='Name of OpenCV framework (default: opencv2, will change to OpenCV in future version)')
    parser.add_argument('--legacy_build', default=False, dest='legacy_build', action='store_true', help='Build legacy opencv2 framework (default: False, equivalent to "--framework_name=opencv2 --without=objc")')
    parser.add_argument('--run_tests', default=False, dest='run_tests', action='store_true', help='Run tests')
    parser.add_argument('--build_docs', default=False, dest='build_docs', action='store_true', help='Build docs')
    parser.add_argument('--disable-swift', default=False, dest='swiftdisabled', action='store_true', help='Disable building of Swift extensions')

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("The following args are not recognized and will not be used: %s" % unknown_args)

    os.environ['IPHONEOS_DEPLOYMENT_TARGET'] = args.iphoneos_deployment_target
    print('Using IPHONEOS_DEPLOYMENT_TARGET=' + os.environ['IPHONEOS_DEPLOYMENT_TARGET'])

    iphoneos_archs = None
    if args.iphoneos_archs:
        iphoneos_archs = args.iphoneos_archs.split(',')
    elif not args.build_only_specified_archs:
        # Supply defaults
        iphoneos_archs = ["armv7", "armv7s", "arm64"]
    print('Using iPhoneOS ARCHS=' + str(iphoneos_archs))

    iphonesimulator_archs = None
    if args.iphonesimulator_archs:
        iphonesimulator_archs = args.iphonesimulator_archs.split(',')
    elif not args.build_only_specified_archs:
        # Supply defaults
        iphonesimulator_archs = ["i386", "x86_64"]
    print('Using iPhoneSimulator ARCHS=' + str(iphonesimulator_archs))

    # Prevent the build from happening if the same architecture is specified for multiple platforms.
    # When `lipo` is run to stitch the frameworks together into a fat framework, it'll fail, so it's
    # better to stop here while we're ahead.
    if iphoneos_archs and iphonesimulator_archs:
        duplicate_archs = set(iphoneos_archs).intersection(iphonesimulator_archs)
        if duplicate_archs:
            print_error("Cannot have the same architecture for multiple platforms in a fat framework! Consider using build_xcframework.py in the apple platform folder instead. Duplicate archs are %s" % duplicate_archs)
            exit(1)

    if args.legacy_build:
        args.framework_name = "opencv2"
        if not "objc" in args.without:
            args.without.append("objc")

    targets = []
    if os.environ.get('BUILD_PRECOMMIT', None):
        if not iphoneos_archs:
            print_error("--iphoneos_archs must have at least one value")
            sys.exit(1)
        targets.append((iphoneos_archs, "iPhoneOS"))
    else:
        if not iphoneos_archs and not iphonesimulator_archs:
            print_error("--iphoneos_archs and --iphonesimulator_archs are undefined; nothing will be built.")
            sys.exit(1)
        if iphoneos_archs:
            targets.append((iphoneos_archs, "iPhoneOS"))
        if iphonesimulator_archs:
            targets.append((iphonesimulator_archs, "iPhoneSimulator"))

    b = iOSBuilder(args.opencv, args.contrib, args.dynamic, args.bitcodedisabled, args.without, args.disable, args.enablenonfree, targets, args.debug, args.debug_info, args.framework_name, args.run_tests, args.build_docs, args.swiftdisabled)

    b.build(args.out)
