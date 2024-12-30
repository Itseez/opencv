# -*- coding:utf-8 -*-
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import tempfile
import sys
import shutil

from tests_common import NewOpenCVTests, unittest

str_unicode = u"测试tést"


@unittest.skipIf(os.name != 'nt' and sys.getfilesystemencoding() != "utf-8", "environment variable `LANG` must be `utf8` in linux")
class UnicodeSupportTest(NewOpenCVTests):
    tmp_path = None

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp(str_unicode)

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    @unittest.skipIf(os.name != 'nt' or cv.getVersionMajor() <= 3 or sys.version_info.major == 2, "only windows and opencv4+ and python3")
    def test_fs_get_cache_dir(self):
        tmp_bak = os.environ["TMP"]
        os.environ["TMP"] = self.tmp_path

        path = cv.utils.fs.getCacheDirectoryForDownloads()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.isdir(path))
        self.assertIn(self.tmp_path, path)

        os.environ["TMP"] = tmp_bak

    def test_im_rw(self):
        data_write = np.random.randint(0, 255, (4, 4, 3))
        file_path = os.path.join(self.tmp_path, str_unicode + ".png")
        cv.imwrite(file_path, data_write)
        self.assertTrue(os.path.exists(file_path))
        data_read = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        self.assertTrue(np.array_equal(data_write, data_read))

    @unittest.skipIf(os.name != 'nt', "known issues that exist on non-Windows systems")
    def test_video_write(self):
        file_path = os.path.join(self.tmp_path, str_unicode + ".mp4")
        cv.VideoWriter(file_path, cv.VideoWriter_fourcc('P', 'I', 'M', '1'), 1, (1, 1))
        self.assertTrue(os.path.exists(file_path))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
