# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import stat
import shutil
import unittest
from unittest import mock

from msserviceprofiler.ms_service_profiler_ext.compare_tools.collector import FileCollector
from msserviceprofiler.ms_service_profiler_ext.common.csv_fields import BatchCSVFields
from msserviceprofiler.msguard.security import open_s


class TestFileCollector(unittest.TestCase):
    """ FileCollector Interface
    1. Constructor: __init__(pattern, max_iter)
    2. Method: collect_pairs(dir_path_a, dir_path_b)

    Abnormal Cases:
    1. Test Constructor
        1.1 pattern not regex
        1.2 max_iter not integer
        1.3 max_iter negative
        1.4 both not valid
    2. Test Method
        2.1 dir_path_a not dir
        2.2 dir_path_b not dir
        2.3 both not valid
        2.4 dir_path_a contains no files
        2.5 dir_path_a contains too many files
        2.6 dir_path_a contains fewer files
        2.7 dir_path_a contains a wanted filename but undesired filetype
        2.8 dir_path_a contains a wanted filename but a large file
        2.9 dir_path_a contains a wanted filename but a soft link
        2.10 dir_path_a contains a wanted filename but risk
    Normal Case:
    1. pattern regex, max_iter positive int, two dirs and all files
    """
    @classmethod
    def setUpClass(cls):
        cls.abnormal_patterns = [1, 1., "1"]
        cls.abnormal_max_iters = [1., "1"]

        cls.normal_pattern = re.compile(r'(batch|service|request)_summary\.csv|profiler\.db')
        cls.normal_max_iter = 100

        cls.collector = FileCollector(cls.normal_pattern, cls.normal_max_iter)

    def setUp(self):
        self.dir_path_a = "dir_path_a"
        if not os.path.isdir(self.dir_path_a):
            os.mkdir("dir_path_a", 0o750)

        self.dir_path_b = "dir_path_b"
        if not os.path.isdir(self.dir_path_b):
            os.mkdir("dir_path_b", 0o750)

    def test_constructor_but_not_regex_should_raise_value_error(self):
        for pattern in self.abnormal_patterns:
            with self.subTest(pattern=pattern):
                self.assertRaises(TypeError, FileCollector, pattern, self.normal_max_iter)

    def test_constructor_but_max_iter_not_valid_should_raise_value_error(self):
        for max_iter in self.abnormal_max_iters:
            with self.subTest(max_iter=max_iter):
                self.assertRaises(TypeError, FileCollector, self.normal_pattern, max_iter)
        self.assertRaises(ValueError, FileCollector, self.normal_pattern, -1)

    def test_collect_pairs_but_not_dir(self):
        self.assertRaises(OSError, self.collector.collect_pairs, __file__, __file__)

    def test_collect_pairs_but_contains_no_files(self):
        self.collector.collect_pairs(self.dir_path_a, self.dir_path_b)

    def test_collect_pairs_but_contains_too_many_files(self):
        with mock.patch('os.listdir', return_value=['a'] * 200):
            self.assertRaises(RuntimeError, self.collector.collect_pairs, self.dir_path_a, self.dir_path_b)

    def test_collect_pairs_but_contains_inconsistent_files(self):
        with open_s(os.path.join(self.dir_path_a, BatchCSVFields.PATH_NAME), 'w'):
            self.collector.collect_pairs(self.dir_path_a, self.dir_path_b)

    def test_collect_pairs_but_contains_undesired_file_types(self):
        os.mkdir(os.path.join(self.dir_path_a, BatchCSVFields.PATH_NAME))
        self.collector.collect_pairs(self.dir_path_a, self.dir_path_b)

    def test_collect_pairs_but_contains_soft_links(self):
        os.symlink(self.dir_path_a, os.path.join(self.dir_path_a, BatchCSVFields.PATH_NAME))
        self.collector.collect_pairs(self.dir_path_a, self.dir_path_b)

    def test_collect_pairs_but_contains_large_files(self):
        with open_s(os.path.join(self.dir_path_a, BatchCSVFields.PATH_NAME), 'w'):
            def large_file_side_effect(path: str):
                if path == os.path.join(self.dir_path_a, BatchCSVFields.PATH_NAME):
                    return os.stat_result([stat.S_IFREG | 0o640, 0, 0, 0, 0, 0, 30 * 1024 * 1024 * 1024, 0, 0, 0])
                return os.stat(path)

            with mock.patch('os.stat', side_effect=large_file_side_effect):
                self.collector.collect_pairs(self.dir_path_a, self.dir_path_b)

    def tearDown(self):
        if os.path.isdir(self.dir_path_a):
            shutil.rmtree(self.dir_path_a)

        if os.path.isdir(self.dir_path_b):
            shutil.rmtree(self.dir_path_b)
