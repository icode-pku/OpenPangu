# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch
from itertools import product

from components.utils.util import (confirmation_interaction, 
                                   check_file_ext, safe_int,
                                   check_file_size_based_on_ext, filter_cmd, safe_get)


class TestUtil(unittest.TestCase):
    
    def test_confirmation_interaction_yes(self):
        yes_input = ['y', 'Y', 'yes', 'YES', 'Yes', 'yES']
        
        for i in yes_input:
            with self.subTest(i):
                with patch('builtins.input', return_value=i):
                    self.assertTrue(confirmation_interaction(""))
    
    def test_confirmation_interaction_no(self):
        no_input = ['n', 'no', 'abc', 'EOF']
        
        for i in no_input:
            with self.subTest(i):
                with patch('builtins.input', return_value=i):
                    self.assertFalse(confirmation_interaction(""))
                    
    def test_check_file_ext_type_error(self):
        paths = [1, 2.5, -3+1j]
        exts = [range(10), set(), dict()]
        
        for path, ext in product(paths, exts):
            with self.subTest(path=path, ext=ext):
                self.assertRaises(TypeError, check_file_ext, path, ext)
    
    def test_check_file_ext_not_match(self):
        path = 'model.onnx'
        exts = ['onnx', 'py', '.py', '.cpp']
        
        for ext in exts:
            with self.subTest(path=path, ext=ext):
                self.assertFalse(check_file_ext(path, ext))
    
    def test_check_file_ext_not_match(self):
        paths = ['model.onnx', 'test.py', 'main.cpp']
        exts = ['.onnx', '.py', '.cpp']
        
        for path, ext in zip(paths, exts):
            with self.subTest(path=path, ext=ext):
                self.assertTrue(check_file_ext(path, ext))
    
    def test_check_file_size_based_on_ext_type_error(self):
        paths = [1, 2.5, -3+1j]
        
        for path in paths:
            with self.subTest(path=path):
                self.assertRaises(TypeError, check_file_size_based_on_ext, path)
                
    def test_check_file_size_based_on_ext_large_size(self):
        exts = ['.csv', '.json', '.txt', '.onnx', '.ini', '.py', '.pth', '.bin']
        
        with patch('os.path.getsize', return_value=500 * 1024 * 1024 * 1024):
            for ext in exts:
                with self.subTest(ext=ext):
                    self.assertFalse(check_file_size_based_on_ext('random_file', ext))
                    self.assertFalse(check_file_size_based_on_ext('random_file' + ext))
        
            with patch('builtins.input', return_value='n'):
                self.assertFalse(check_file_size_based_on_ext('random_file'))
        
    
    def test_check_file_size_based_on_ext_normal_size(self):
        config_file_size = 8 * 1024
        text_file_size = 8 * 1024 * 1024
        onnx_model_size = 1 * 1024 * 1024 * 1024
        model_weigtht_size = 8 * 1024 * 1024 * 1024
        
        exts = ['.ini', '.csv', '.json', '.txt', '.py', '.onnx', '.pth', '.bin']
        sizes = [config_file_size] + [text_file_size] * 4 + [onnx_model_size] + [model_weigtht_size] * 2
        
        for ext, size in zip(exts, sizes): 
            with patch('os.path.getsize', return_value=size):
                with self.subTest(ext=ext, size=size):
                    self.assertTrue(check_file_size_based_on_ext('random_file', ext))
                    self.assertTrue(check_file_size_based_on_ext('random_file' + ext))
                    self.assertTrue(check_file_size_based_on_ext('random_file'))

    def test_safe_int_when_value_is_valid(self):
        self.assertEqual(safe_int("2147483647"), 2147483647)  # Max 32-bit signed int
        self.assertEqual(safe_int("-2147483648"), -2147483648)  # Min 32-bit signed int
        self.assertEqual(safe_int("999999999999999999"), 999999999999999999) # big int

    def test_safe_int_given_invalid_value_then_raise_value_error(self):
        with self.assertRaises(ValueError) as context:
            safe_int("aasdf", "ENV")
        assert "The value of the variable ENV is not valid, what we need is a value that can be convert to int." == str(context.exception)

class TestFilterCmd(unittest.TestCase):
    def test_valid_characters(self):
        input_args = ["hello", "world123", "file_name.txt", "path/to/file", "a-b-c", "A_B_C", "1 2 3", "var=value"]
        expected = input_args.copy()
        self.assertEqual(filter_cmd(input_args), expected)

    def test_invalid_characters_raises_error(self):
        input_args = ["hello!", "world@123", "file$name", "path/to|file", "a{b}c", "A#B#C"]
        for arg in input_args:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for input: {arg}"):
                filter_cmd([arg])

    def test_mixed_valid_invalid_raises_error(self):
        input_args = ["valid", "inval!d", "good123", "bad@arg", "ok"]
        with self.assertRaises(ValueError):
            filter_cmd(input_args)

    def test_empty_input(self):
        self.assertEqual(filter_cmd([]), [])

    def test_non_string_input(self):
        input_args = [123, 45.67, True, None]
        expected = ["123", "45.67", "True", "None"]
        self.assertEqual(filter_cmd(input_args), expected)

    def test_whitespace_only(self):
        self.assertEqual(filter_cmd([" ", "   "]), [" ", "   "])

    def test_edge_cases(self):
        input_args = ["", "-._ /=", "a"*1000]
        with self.assertRaises(ValueError):
            filter_cmd(input_args)

    def test_non_ascii_raises_error(self):
        input_args = ["héllo", "世界", "café"]
        for arg in input_args:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for input: {arg}"):
                filter_cmd([arg])

class TestSafeGet(unittest.TestCase):
    def test_safe_get_list_valid_index(self):
        data = [10, 20, 30]
        self.assertEqual(safe_get(data, 0), 10)
        self.assertEqual(safe_get(data, 2), 30)

    def test_safe_get_list_invalid_index(self):
        data = [1, 2, 3]
        with self.assertRaises(IndexError):
            safe_get(data, 3)
        with self.assertRaises(IndexError):
            safe_get(data, -1)
        with self.assertRaises(IndexError):
            safe_get(data, "0")

    def test_safe_get_dict_valid_key(self):
        d = {"a": 1, "b": 2}
        self.assertEqual(safe_get(d, "a"), 1)
        self.assertEqual(safe_get(d, "b"), 2)

    def test_safe_get_dict_invalid_key(self):
        d = {"x": 100}
        with self.assertRaises(KeyError):
            safe_get(d, "y")
        with self.assertRaises(KeyError):
            safe_get(d, 1)

    def test_safe_get_invalid_container(self):
        with self.assertRaises(TypeError):
            safe_get("notalistordict", 0)
        with self.assertRaises(TypeError):
            safe_get(123, 0)
        with self.assertRaises(TypeError):
            safe_get(None, 0)
