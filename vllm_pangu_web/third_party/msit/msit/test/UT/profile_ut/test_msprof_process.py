# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import unittest
from unittest.mock import patch, MagicMock
from components.profile.msit_prof.msprof.msprof_process import (
    remove_invalid_chars,
    msprof_run_profiling,
    args_rules, msprof_process
)

class TestMsprofUtils(unittest.TestCase):
    def test_remove_invalid_chars(self):
        # Test normal case
        self.assertEqual(remove_invalid_chars("test`$|;&><"), "test")
        # Test with no invalid chars
        self.assertEqual(remove_invalid_chars("test"), "test")

    @patch('os.system')
    def test_msprof_run_profiling_success(self, mock_os_system):
        # Mock os.system to return 0
        mock_os_system.return_value = 0
        args = MagicMock()
        args.output = '/path/to/output'
        args.application = '/path/to/app'
        # Other args should be set here
        msprof_bin = '/path/to/msprof'
        msprof_run_profiling(args, msprof_bin)
        # Assert that os.system was called with the correct command
        mock_os_system.assert_called_once()

    @patch('os.system')
    def test_msprof_run_profiling_failure(self, mock_os_system):
        # Mock os.system to return non-zero
        mock_os_system.return_value = 1
        args = MagicMock()
        args.output = '/path/to/output'
        args.application = '/path/to/app'
        # Other args should be set here
        msprof_bin = '/path/to/msprof'
        with self.assertRaises(RuntimeError):
            msprof_run_profiling(args, msprof_bin)

    def test_args_rules_output_too_long(self):
        args = MagicMock()
        args.output = 'a' * 256  # PATH_MAX_LENGTH + 1
        with self.assertRaises(RuntimeError):
            args_rules(args)

    def test_args_rules_application_not_set(self):
        args = MagicMock()
        args.application = None
        with self.assertRaises(RuntimeError):
            args_rules(args)

    def test_args_rules_application_too_long(self):
        args = MagicMock()
        args.application = 'a' * 256  # PATH_MAX_LENGTH + 1
        with self.assertRaises(RuntimeError):
            args_rules(args)

    def test_args_rules_invalid_flag(self):
        args = MagicMock()
        args.model_execution = 'invalid'
        with self.assertRaises(RuntimeError):
            args_rules(args)

    @patch('components.profile.msit_prof.msprof.msprof_process.args_rules')
    @patch('shutil.which')
    @patch('os.getenv')
    @patch('components.profile.msit_prof.msprof.msprof_process.msprof_run_profiling')
    def test_msprof_process_success(self, mock_msprof_run_profiling, mock_os_getenv, mock_shutil_which, mock_args_rules):
        mock_args = MagicMock()
        mock_args_rules.return_value = mock_args
        mock_msprof_bin = '/path/to/msprof'
        mock_shutil_which.return_value = mock_msprof_bin
        mock_os_getenv.return_value = None
        mock_msprof_run_profiling.return_value = None
        args = MagicMock()

        self.assertEqual(msprof_process(args), 0)
        mock_args_rules.assert_called_once_with(args)
        mock_msprof_run_profiling.assert_called_once_with(mock_args, mock_msprof_bin)

    @patch('shutil.which')
    def test_msprof_process_runtime_error(self, mock_shutil_which):
        # Mock which to return msprof binary path
        mock_shutil_which.return_value = '/path/to/msprof'
        args = MagicMock()
        args.output = '/path/to/output'
        args.application = '/path/to/app'
        # Set args to trigger a RuntimeError in args_rules
        args.application = None
        self.assertEqual(msprof_process(args), 1)

if __name__ == '__main__':
    unittest.main()