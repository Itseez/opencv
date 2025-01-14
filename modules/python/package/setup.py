import os
import setuptools


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_module_typing_stub_files(root_module_path):
    stub_files = []
    for module_path, _, files in os.walk(root_module_path):
        stub_files.extend(
            map(lambda p: os.path.join(module_path, p),
                filter(lambda f: f.endswith(".pyi"), files))
        )
    return stub_files


def main():
    os.chdir(SCRIPT_DIR)

    package_name = 'opencv'
    package_version = os.environ.get('OPENCV_VERSION', '5.0.0-alpha')  # TODO

    long_description = 'Open Source Computer Vision Library Python bindings'  # TODO

    root_module_path = os.path.join(SCRIPT_DIR, "cv2")
    py_typed_path = os.path.join(root_module_path, "py.typed")
    typing_stub_files = []
    if os.path.isfile(py_typed_path):
        typing_stub_files = collect_module_typing_stub_files(root_module_path)
        if len(typing_stub_files) > 0:
            typing_stub_files.append(py_typed_path)

    setuptools.setup(
        name=package_name,
        version=package_version,
        url='https://github.com/opencv/opencv',
        license='Apache 2.0',
        description='OpenCV python bindings',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        package_data={
            "cv2": typing_stub_files
        },
        maintainer="OpenCV Team",
        install_requires="numpy",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: C++",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Software Development",
        ],
    )


if __name__ == '__main__':
    main()
