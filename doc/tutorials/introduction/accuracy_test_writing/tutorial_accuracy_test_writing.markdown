# Writing Accuracy Tests {#tutorial_accuracy_test_writing}
====================================================

This testing framework is based on Google Tests

There are two major types of C++ tests: *accuracy/regression* tests and *performance* tests. Each module can have two test binaries: `opencv_test_<MODULE_NAME>` and `opencv_perf_<MODULE_NAME>` (or `opencv_test_<MODULE_NAME>d` and `opencv_perf_<MODULE_NAME>d` in Windows debug binaries), and two tests folders: `<opencv(_contrib)/modules/<MODULE_NAME>/test>` and `<opencv(_contrib)/modules/<MODULE_NAME>/perf>`. These applications can be built for every supported platform (Windows/Linux/MacOS/Android).

## ACCURACY TESTS

### Work directory

Most of modules have their own directory for accuracy tests: `opencv(_contrib)/modules/(moduleName)/test/...`

### Dir Structure

- `test_precomp.hpp` - file for includes
- `test_main.cpp` - main file of test
- `test_smth1.cpp` - files with the test's bodies
- `test_smth2.cpp`
- `...`

### Test structure

```c++
// name of this case is "name1.name2"
TEST(name1, name2)
{
    ASSERT_....;
}
```
More information abut the code for the tests can be found in the GoogleTest Documentation: https://github.com/google/googletest/blob/master/docs/primer.md

### Run

Before the first run of new tests, you need to reconfigure the project with CMake.

If you want to run **ALL TESTS**, you need to run the test sample:

- (*Windows*) run the executable file `<BUILD_DIR>/bin/Debug(Release)/opencv_test_<MODULE_NAME>.exe`
- (*Linux*) run in terminal `./<BUILD_DIR>/bin/opencv_test_<MODULE_NAME>`.

If you want to run **CERTAIN TESTS**:.

- If you work in IDE you need to find the project preference named `Command Arguments` or similar and write there the key like `--gtest_filter=<KEY>`.  [gtest_filter documentation](https://github.com/google/googletest/blob/273f8cb059a4e7b089731036392422b5ef489791/docs/advanced.md#running-a-subset-of-the-tests)
- If you want to use your system:
  - (*Windows*) run the executable file `<BUILD_DIR>/bin/Debug(Release)/opencv_test_<MODULE_NAME>.exe --gtest_filter=<KEY>`.
  - (*Linux*) run in terminal `./<BUILD_DIR>/bin/opencv_test_<MODULE_NAME> --gtest_filter=<KEY>`.

# TEST DATA

Repository with the test data (opencv_extra): https://github.com/opencv/opencv_extra

Before working with data you need to set environment variable `OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata`

# ASSERTION

**Assertion documentation**:

- Link: https://github.com/google/googletest/blob/master/googletest/docs/primer.md

# BuildBot

OpenCV's continuous integration system is available here:

- [http://pullrequest.opencv.org](http://pullrequest.opencv.org/)
- [http://pullrequest.opencv.org/contrib](https://pullrequest.opencv.org/#/summary/contrib)

This system check code style, build all OpenCV and run tests. You can read more about this bot [here](https://pullrequest.opencv.org/buildbot/).

The BuildBot dashboard looks like a grid where rows are the `pull requests` and columns are `id, author, Title & description, Docs, Builds `.

There are some conditions of samples:

- green (success)

  It means that your code was successful build, all tests passed.

- orange (warnings)

  Your code has warnings. You need to click on the number of build and find the orange line in this page. There are two types of buildbot logs: `stdio` and `other like warnings, tests and ets`. `stdio` type show you all build logs. `other` show you only useful logs.

  For example: `warning` logs show only warning logs.

- red (failure)

  - Docs

    Code style check has failed. Click on the number of build and click on `stdio` . There are the error code style logs.

  - Builds

    Building has failed. Click on the number of build and click on `stdio` . There are the all building logs. Errors are marked the red color.

- yellow (building)

  Your project is building now.

- blue (scheduled)

  Building your project is scheduled.