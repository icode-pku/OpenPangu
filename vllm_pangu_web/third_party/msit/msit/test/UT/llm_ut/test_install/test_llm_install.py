# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import unittest
from unittest import mock

from components.llm.msit_llm.__install__ import LlmInstall
from components.utils.file_utils import FileCheckException


class TestLlmInstall(unittest.TestCase):
    @mock.patch('components.llm.msit_llm.__install__.subprocess.run')
    def test_build_extra_with_find_links(self, mock_run):
        find_links_path = os.path.dirname(os.path.realpath(__file__))
        LlmInstall.build_extra(find_links=find_links_path)

        self.assertEqual(os.environ.get('AIT_INSTALL_FIND_LINKS'), os.path.realpath(find_links_path))
        mock_run.assert_called_once()
        del os.environ['AIT_INSTALL_FIND_LINKS']

        find_links_path = os.path.join(find_links_path, 'non-existent_directory')
        with self.assertRaises(FileCheckException) as context:
            LlmInstall.build_extra(find_links=find_links_path)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    @mock.patch('components.llm.msit_llm.__install__.subprocess.run')
    def test_download_extra(self, mock_run):
        dest_path = os.path.dirname(os.path.realpath(__file__))
        LlmInstall.download_extra(dest_path)

        self.assertEqual(os.environ.get('AIT_DOWNLOAD_PATH'), os.path.realpath(dest_path))
        mock_run.assert_called_once()
        del os.environ['AIT_DOWNLOAD_PATH']

        dest_path = os.path.join(dest_path, 'non-existent_directory')
        with self.assertRaises(FileCheckException) as context:
            LlmInstall.download_extra(dest_path)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    @mock.patch('components.llm.msit_llm.__install__.subprocess.run')
    def test_build_extra_on_windows(self, mock_run):
        with mock.patch('components.llm.msit_llm.__install__.sys.platform', 'win32'):
            LlmInstall.build_extra()
            mock_run.assert_not_called()

    @mock.patch('components.llm.msit_llm.__install__.subprocess.run')
    def test_download_extra_on_windows(self, mock_run):
        with mock.patch('components.llm.msit_llm.__install__.sys.platform', 'win32'):
            LlmInstall.download_extra('/dummy/path')
            mock_run.assert_not_called()

    def test_check_with_missing_libopchecker(self):
        check_res = LlmInstall.check()
        expected_msg = "[warnning] build libopchecker.so failed. will make the opchecker feature unusable. " \
                       "use `msit build-extra llm` to try again"
        self.assertIn(expected_msg, check_res)