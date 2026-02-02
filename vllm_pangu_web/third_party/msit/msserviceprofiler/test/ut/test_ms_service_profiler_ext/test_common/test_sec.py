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

import os
import stat
import shutil
import unittest
from unittest import mock

from msserviceprofiler.ms_service_profiler_ext.common.sec import (
    list_dir_common_check, traverse_dir_common_check,
    read_file_common_check, execute_file_common_check
)


class TestSec(unittest.TestCase):
    def setUp(self):
        self.stat_dict = dict(
            st_mode=0, st_ino=0, st_dev=0, st_nlink=0,
            st_uid=0, st_gid=0, 
            st_size=0, 
            st_atime=0, st_mtime=0, st_ctime=0
        )
        
        self.mock_os_access = mock.patch('os.access', return_value=True)
        
    def test_list_dir_common_check_not_exists_should_raise(self):
        self.assertRaises(OSError, list_dir_common_check, "a", raise_argparse=False)

    def test_list_dir_common_check_file_name_too_long_should_raise(self):
        self.assertRaises(OSError, list_dir_common_check, "a" * 4096, raise_argparse=False)
    
    def test_list_dir_common_check_other_file_type_should_raise(self):
        other_file_type = [stat.S_IFLNK, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK, stat.S_IFREG]
        for file_type in other_file_type:
            with self.subTest(file_type=file_type):
                self.stat_dict['st_mode'] = file_type | 0o777
                with mock.patch('os.stat', return_value=os.stat_result(self.stat_dict.values())):
                    self.assertRaises(OSError, list_dir_common_check, "a", raise_argparse=False)

    def test_read_file_common_check_not_file_should_raise(self):
        other_file_type = [stat.S_IFLNK, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK, stat.S_IFDIR]
        for file_type in other_file_type:
            with self.subTest(file_type=file_type):
                self.stat_dict['st_mode'] = file_type | 0o777
                with mock.patch('os.stat', return_value=os.stat_result(self.stat_dict.values())):
                    self.assertRaises(OSError, read_file_common_check, "a", raise_argparse=False)
                    
    def test_read_file_common_check_file_oversize_should_raise(self):
        self.mock_os_access.start()
        self.stat_dict['st_mode'] = stat.S_IFREG | 0o750
        self.stat_dict['st_size'] = 3 * 1024 * 1024 * 1024
        with mock.patch('os.stat', return_value=os.stat_result(self.stat_dict.values())):
            self.assertRaises(OSError, list_dir_common_check, "a.csv", raise_argparse=False)

    def tearDown(self):
        self.mock_os_access.stop()
