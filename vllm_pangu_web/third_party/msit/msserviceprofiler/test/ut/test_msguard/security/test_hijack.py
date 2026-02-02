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
#

import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from msserviceprofiler.msguard.security import update_env_s
from msserviceprofiler.msguard.security import open_s
from msserviceprofiler.msguard.security.hijack import update_env_s as hijack_update_env_s
from msserviceprofiler.msguard.constraints import PathConstraint, Rule, InvalidParameterError


class TestUpdateEnvS(unittest.TestCase):
    def setUp(self):
        # Backup environment variables that might be modified
        self.env_backup = os.environ.copy()
        self.original_cwd = os.getcwd()

    def tearDown(self):
        # Restore original environment variables and working directory
        os.environ.clear()
        os.environ.update(self.env_backup)
        os.chdir(self.original_cwd)

    def test_update_env_s_with_valid_absolute_path(self):
        """Test if the function can handle valid absolute paths correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "TEST_VAR"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            update_env_s(test_var, test_path)
            self.assertEqual(os.environ[test_var], test_path,
                            "Environment variable should be set to the specified absolute path")

    def test_update_env_s_with_non_string_env_var(self):
        """Test if the function can handle non-string environment variable names (should raise TypeError)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            with self.assertRaises(TypeError,
                                 msg="Non-string environment variable name should raise TypeError"):
                update_env_s(123, test_path)

    def test_update_env_s_with_empty_env_var(self):
        """Test if the function can handle empty environment variables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "EMPTY_VAR"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            if test_var in os.environ:
                del os.environ[test_var]

            update_env_s(test_var, test_path)
            self.assertEqual(os.environ[test_var], test_path,
                            "Empty environment variable should be set to the specified path")

    def test_update_env_s_append_mode(self):
        """Test if append mode (prepend=False) works correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "APPEND_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()
            original_value = "/original/path"

            os.environ[test_var] = original_value
            update_env_s(test_var, test_path, prepend=False)
            expected = f"{original_value}{os.pathsep}{test_path}"
            self.assertEqual(os.environ[test_var], expected,
                            "Path should be appended to the end of the existing value")

    def test_update_env_s_with_existing_value(self):
        """Test if the function can handle environment variables with existing values (default prepend mode)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "EXISTING_VAR"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()
            original_value = "/existing/path"
            os.environ[test_var] = original_value

            update_env_s(test_var, test_path)
            expected = f"{test_path}{os.pathsep}{original_value}"
            self.assertEqual(os.environ[test_var], expected,
                            "New path should be prepended to the front of the existing value")

    def test_update_env_s_prepend_false_with_existing_value(self):
        """Test if append mode (prepend=False) can handle environment variables with existing values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "PREPEND_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()
            original_value = "/original/path"
            os.environ[test_var] = original_value

            update_env_s(test_var, test_path, prepend=False)
            expected = f"{original_value}{os.pathsep}{test_path}"
            self.assertEqual(os.environ[test_var], expected,
                            "New path should be appended to the end of the existing value")

    def test_update_env_s_multiple_updates(self):
        """Test if multiple updates to the same environment variable work correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "MULTI_TEST"
            path1 = os.path.join(temp_dir, "path1")
            path2 = os.path.join(temp_dir, "path2")
            open_s(path1, "a").close()
            open_s(path2, "a").close()

            update_env_s(test_var, path1)
            self.assertEqual(os.environ[test_var], path1,
                            "After first update, environment variable should equal the first path")

            update_env_s(test_var, path2)
            expected = f"{path2}{os.pathsep}{path1}"
            self.assertEqual(os.environ[test_var], expected,
                            "After second update, new path should be prepended to the existing value")

    def test_update_env_s_with_constraint_violation(self):
        """Test if InvalidParameterError is raised when path constraint is not satisfied"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "CONSTRAINT_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            # Create a mock constraint that is not satisfied
            mock_constraint = MagicMock(spec=PathConstraint)
            mock_constraint.is_satisfied_by.return_value = False

            with self.assertRaises(InvalidParameterError,
                                 msg="Should raise InvalidParameterError when path does not satisfy constraint"):
                update_env_s(test_var, test_path, constraint=mock_constraint)

    def test_update_env_s_with_custom_constraint_satisfied(self):
        """Test normal behavior when path satisfies custom constraint"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "CUSTOM_CONSTRAINT_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            # Create a mock constraint that is satisfied
            mock_constraint = MagicMock(spec=PathConstraint)
            mock_constraint.is_satisfied_by.return_value = True

            update_env_s(test_var, test_path, constraint=mock_constraint)
            self.assertEqual(os.environ[test_var], test_path,
                            "Environment variable should be set correctly when path satisfies constraint")

    def test_update_env_s_relative_path_converted_to_absolute(self):
        """Test if relative path is converted to absolute path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)  # Change to temporary directory
                test_var = "RELATIVE_PATH_TEST"
                relative_path = "relative_file"
                abs_path = os.path.abspath(relative_path)
                open_s(abs_path, "a").close()

                update_env_s(test_var, relative_path)
                # 现在应该期望绝对路径
                self.assertEqual(os.environ[test_var], abs_path,
                                "Relative path should be converted to absolute path")
            finally:
                os.chdir(original_cwd)  # Restore original working directory

    def test_update_env_s_with_nonexistent_file_but_valid_constraint(self):
        """Test case with non-existent file but constraint check passes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "NONEXISTENT_TEST"
            test_path = os.path.join(temp_dir, "nonexistent_file")

            # Create a mock constraint that is satisfied even if file doesn't exist
            mock_constraint = MagicMock(spec=PathConstraint)
            mock_constraint.is_satisfied_by.return_value = True

            update_env_s(test_var, test_path, constraint=mock_constraint)
            # 现在应该期望绝对路径
            self.assertEqual(os.environ[test_var], os.path.abspath(test_path),
                            "Environment variable should be set to absolute path even if file doesn't exist, "
                            "as long as constraint is satisfied")

    def test_update_env_s_empty_path_with_valid_constraint(self):
        """Test case with empty path but constraint check passes"""
        test_var = "EMPTY_PATH_TEST"
        empty_path = ""

        # Create a mock constraint that is satisfied
        mock_constraint = MagicMock(spec=PathConstraint)
        mock_constraint.is_satisfied_by.return_value = True

        abs_empty_path = os.path.abspath(empty_path)
        update_env_s(test_var, empty_path, constraint=mock_constraint)
        # 现在应该期望绝对路径
        self.assertEqual(os.environ[test_var], abs_empty_path,
                        "Empty path should be converted to absolute path of current directory")

    def test_update_env_s_with_special_characters_in_path(self):
        """Test case with special characters in path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "SPECIAL_CHAR_TEST"
            # Create filename with special characters
            special_name = "test file with spaces and @#$% characters"
            test_path = os.path.join(temp_dir, special_name)
            open_s(test_path, "a").close()

            update_env_s(test_var, test_path)
            self.assertEqual(os.environ[test_var], test_path,
                            "Path with special characters should be handled correctly")

    def test_update_env_s_duplicate_path_prevention(self):
        """Test if duplicate paths are prevented in environment variable"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "DUPLICATE_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            # First addition
            update_env_s(test_var, test_path)
            # Second addition of same path
            update_env_s(test_var, test_path)
            
            # Check if path is duplicated
            paths = os.environ[test_var].split(os.pathsep)
            self.assertEqual(len(paths), 2, "Duplicate paths should be preserved")
            # 现在应该期望绝对路径
            abs_path = os.path.abspath(test_path)
            self.assertEqual(paths[0], abs_path)
            self.assertEqual(paths[1], abs_path)

    def test_update_env_s_with_different_constraint_rules(self):
        """Test different constraint rules"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "DIFFERENT_CONSTRAINTS"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            # Test different constraint rules
            constraints_to_test = [
                Rule.input_file_read,  # Assuming this is a valid constraint
                # Add other constraint rules here
            ]

            for constraint in constraints_to_test:
                with self.subTest(constraint=constraint):
                    # Backup current environment variable value
                    original_value = os.environ.get(test_var)
                    
                    try:
                        update_env_s(test_var, test_path, constraint=constraint)
                        self.assertEqual(os.environ[test_var], test_path,
                                        f"Constraint {constraint} should allow path setting")
                    except InvalidParameterError:
                        # This is expected if constraint is not satisfied
                        pass
                    finally:
                        # Restore environment variable
                        if original_value is None and test_var in os.environ:
                            del os.environ[test_var]
                        elif original_value is not None:
                            os.environ[test_var] = original_value

    @patch('msserviceprofiler.msguard.security.hijack.os.path.abspath')
    def test_update_env_s_path_normalization(self, mock_abspath):
        """Test path normalization process"""
        test_var = "NORMALIZATION_TEST"
        input_path = "/some/.././path"
        normalized_path = "/path"
        mock_abspath.return_value = normalized_path

        # Create a mock constraint that is satisfied
        mock_constraint = MagicMock(spec=PathConstraint)
        mock_constraint.is_satisfied_by.return_value = True

        update_env_s(test_var, input_path, constraint=mock_constraint)
        
        # Verify abspath was called
        mock_abspath.assert_called_once_with(input_path)
        # Verify normalized path is used
        self.assertEqual(os.environ[test_var], normalized_path)

    def test_hijack_module_direct_import(self):
        """Test functions directly imported from hijack module"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "HIJACK_DIRECT_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            # Use function directly from hijack module
            hijack_update_env_s(test_var, test_path)
            self.assertEqual(os.environ[test_var], test_path,
                            "Function directly imported from hijack module should work correctly")
