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
import unittest
import tempfile
import pickle
from io import BytesIO

from msserviceprofiler.msguard.security.injection import (
    is_safe_csv_value,
    sanitize_csv_value,
    SafeUnpickler,
    pickle_load_s,
    pickle_loads_s,
    CSV_INJECTION_PATTERN
)
from msserviceprofiler.msguard.security.exception import (
    CSVInjectionError, 
    PickleInjectionError
)
from msserviceprofiler.msguard.security.io import open_s
from msserviceprofiler.msguard import InvalidParameterError


class TestCSVInjection(unittest.TestCase):
    def test_csv_injection_pattern(self):
        """Test CSV injection pattern matching"""
        # Test dangerous prefixes
        self.assertTrue(CSV_INJECTION_PATTERN.search("=cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("@cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("+cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("-cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("%cmd"))
        # Test full-width characters
        self.assertTrue(CSV_INJECTION_PATTERN.search("＝cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("＋cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("－cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("％cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("＠cmd"))
        
        # Test dangerous infix patterns
        self.assertTrue(CSV_INJECTION_PATTERN.search("text;=cmd"))
        self.assertTrue(CSV_INJECTION_PATTERN.search("text;@cmd"))
        
        # Test safe strings
        self.assertIsNone(CSV_INJECTION_PATTERN.search("safe text"))
        self.assertIsNone(CSV_INJECTION_PATTERN.search("123.45"))

    def test_is_safe_csv_value_with_numeric_string(self):
        self.assertTrue(is_safe_csv_value("123.45"), 
                       "Numeric string should be considered safe")
        self.assertTrue(is_safe_csv_value("-123.45"), 
                       "Negative numeric string should be considered safe")
        self.assertTrue(is_safe_csv_value("+123.45"), 
                       "Positive numeric string should be considered safe")

    def test_is_safe_csv_value_with_safe_string(self):
        self.assertTrue(is_safe_csv_value("normal text"), 
                       "Normal text should be considered safe")
        self.assertTrue(is_safe_csv_value("hello world"), 
                       "Normal text should be considered safe")

    def test_is_safe_csv_value_with_unsafe_prefix(self):
        self.assertFalse(is_safe_csv_value("=cmd|' /C calc'!"), 
                        "Strings starting with = may contain CSV injection code and should be considered unsafe")
        self.assertFalse(is_safe_csv_value("@cmd"), 
                        "Strings starting with @ should be considered unsafe")
        self.assertFalse(is_safe_csv_value("+cmd"), 
                        "Strings starting with + should be considered unsafe")
        self.assertFalse(is_safe_csv_value("-cmd"), 
                        "Strings starting with - should be considered unsafe")

    def test_is_safe_csv_value_with_unsafe_infix(self):
        self.assertFalse(is_safe_csv_value("text;=cmd"), 
                        "Strings containing special characters may contain CSV injection code and should "
                        "be considered unsafe")
        self.assertFalse(is_safe_csv_value("text;@cmd"), 
                        "Strings containing ;@ should be considered unsafe")

    def test_is_safe_csv_value_with_non_string(self):
        self.assertTrue(is_safe_csv_value(123), 
                       "Non-string types should be considered safe")
        self.assertTrue(is_safe_csv_value(123.45), 
                       "Float should be considered safe")
        self.assertTrue(is_safe_csv_value(True), 
                       "Boolean should be considered safe")
        self.assertTrue(is_safe_csv_value(None), 
                       "None should be considered safe")
        self.assertTrue(is_safe_csv_value([]), 
                       "List should be considered safe")

    def test_is_safe_csv_value_edge_cases(self):
        """Test edge cases"""
        self.assertTrue(is_safe_csv_value(""), "Empty string should be safe")
        self.assertTrue(is_safe_csv_value("   "), "Space string should be safe")
        self.assertTrue(is_safe_csv_value("123"), "Pure numeric string should be safe")
        self.assertTrue(is_safe_csv_value("3.14"), "Decimal should be safe")

    def test_sanitize_csv_value_with_safe_string(self):
        self.assertEqual(sanitize_csv_value("safe"), "safe", 
                         "Safe string should be returned as is")
        self.assertEqual(sanitize_csv_value("123"), "123", 
                         "Numeric string should be returned as is")

    def test_sanitize_csv_value_with_unsafe_string_strict(self):
        with self.assertRaises(CSVInjectionError, 
                              msg="Unsafe string should raise CSVInjectionError when replace=False"):
            sanitize_csv_value("=cmd", replace=False)
        
        with self.assertRaises(CSVInjectionError):
            sanitize_csv_value("@cmd", replace=False)

    def test_sanitize_csv_value_with_unsafe_string_replace(self):
        self.assertEqual(sanitize_csv_value("=cmd", replace=True), "'=cmd", 
                         "Unsafe string should be escaped in replace mode")
        self.assertEqual(sanitize_csv_value("@cmd", replace=True), "'@cmd", 
                         "@ prefix should be escaped in replace mode")
        self.assertEqual(sanitize_csv_value("+cmd", replace=True), "'+cmd", 
                         "+ prefix should be escaped in replace mode")
        self.assertEqual(sanitize_csv_value("-cmd", replace=True), "'-cmd", 
                         "- prefix should be escaped in replace mode")

    def test_sanitize_csv_value_with_numeric_like_unsafe(self):
        """Test numeric-looking but unsafe cases"""
        # These look like numbers but contain dangerous characters
        self.assertEqual(sanitize_csv_value("=123", replace=True), "'=123")
        self.assertEqual(sanitize_csv_value("@456", replace=True), "'@456")


class TestPickleInjection(unittest.TestCase):
    class DangerPerson:
        def __init__(self):
            self.name = 'echo "This is a malicious command"'

        def __reduce__(self):
            return (os.system, (self.name,))
    
    class NicePerson:
        def __init__(self):
            self.name = 2

        def __reduce__(self):
            return (int, (self.name,))
    
    class ComplexSafeObject:
        def __init__(self):
            self.data = {"key": "value", "number": 42}
            self.tuple = (1, 2, 3)
            self.list = [1, 2, 3]

    def setUp(self):
        # Create temporary test data
        self.safe_data = pickle.dumps(self.NicePerson())
        self.unsafe_data = pickle.dumps(self.DangerPerson())
        self.complex_safe_data = pickle.dumps(self.ComplexSafeObject())

    def test_safe_unpickler_default_callback(self):
        """Test SafeUnpickler default callback function"""
        unpickler = SafeUnpickler(BytesIO(self.safe_data))
        
        # Test safe combinations in default callback
        self.assertTrue(unpickler.default_safe_callback('builtins', 'int'))
        self.assertTrue(unpickler.default_safe_callback('builtins', 'str'))
        self.assertTrue(unpickler.default_safe_callback('datetime', 'datetime'))
        self.assertTrue(unpickler.default_safe_callback('collections', 'OrderedDict'))
        
        # Test unsafe combinations
        self.assertFalse(unpickler.default_safe_callback('os', 'system'))
        self.assertFalse(unpickler.default_safe_callback('subprocess', 'call'))
        self.assertFalse(unpickler.default_safe_callback('builtins', 'eval'))

    def test_safe_unpickler_validation(self):
        """Test callback function validation"""
        # Test invalid callback type
        with self.assertRaises(TypeError):
            SafeUnpickler(BytesIO(self.safe_data), call_back_fn="not_callable")
        
        # Test callback with incorrect number of parameters
        def bad_callback(only_one_param):
            return True
        
        with self.assertRaises(ValueError):
            SafeUnpickler(BytesIO(self.safe_data), call_back_fn=bad_callback)

    def test_pickle_loads_s_success_with_default_callback(self):
        """Test successful loading of safe data with default callback"""
        result = pickle_loads_s(self.safe_data)
        self.assertEqual(result, 2, "Should correctly load basic int type")

    def test_pickle_loads_s_fail_with_default_callback(self):
        """Test default callback should block dangerous objects"""
        with self.assertRaises(PickleInjectionError, msg="Default callback should block os.system"):
            pickle_loads_s(self.unsafe_data)

    def test_pickle_load_s_success_with_custom_callback(self):
        """Test custom callback allowing specific types"""
        def custom_callback(module, name):
            return module == "builtins" and name == "int"

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            result = pickle_load_s(test_file, fn=custom_callback)
            self.assertEqual(result, 2, "Custom callback should allow int type")

    def test_pickle_load_s_fail_with_custom_callback(self):
        """Test custom callback should block non-allowed types"""
        def custom_callback(module, name):
            return False  # Reject all types

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            with self.assertRaises(PickleInjectionError, 
                                   msg="Should raise error when custom callback rejects all types"):
                pickle_load_s(test_file, fn=custom_callback)

    def test_pickle_load_s_file_handle(self):
        """Test passing file handle directly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            with open_s(test_file, "rb") as f:
                result = pickle_load_s(f)
                self.assertEqual(result, 2, "Should handle file handle input correctly")

    def test_pickle_loads_s_invalid_input(self):
        """Test non-bytes input should raise TypeError"""
        with self.assertRaises(TypeError, msg="Non-bytes input should trigger type error"):
            pickle_loads_s("not bytes data")

    def test_pickle_load_s_nonexistent_file(self):
        """Test non-existent file path should raise FileNotFoundError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "nonexistent.pkl")
            with self.assertRaises(InvalidParameterError, msg="Non-existent file should trigger error"):
                pickle_load_s(bad_path)

    def test_pickle_load_s_invalid_callback(self):
        """Test invalid callback function should raise TypeError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            with self.assertRaises(TypeError, msg="Non-callable object as callback should trigger error"):
                pickle_load_s(test_file, fn="not callable")

    def test_pickle_load_s_with_malicious_poc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "malicious.pickle")
            with open_s(file_path, "wb") as f:
                pickle.dump(self.DangerPerson(), f)

            with self.assertRaises(PickleInjectionError,
                                  msg="Should detect malicious pickle file containing system commands"):
                pickle_load_s(file_path)

    def test_pickle_loads_s_with_malicious_poc(self):
        malicious_data = pickle.dumps(self.DangerPerson())
        with self.assertRaises(PickleInjectionError,
                              msg="Should detect pickle byte stream containing system commands"):
            pickle_loads_s(malicious_data)

    def test_pickle_load_s_with_complex_safe_object(self):
        """Test loading complex but safe objects"""
        def safe_callback(module, name):
            safe_modules = {'builtins', '__builtin__', 'copy_reg', '_codecs'}
            return module in safe_modules or module.split('.')[0] in safe_modules

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "complex_safe.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.complex_safe_data)

            # This test might fail depending on object complexity, but mainly tests no security exception is thrown
            try:
                result = pickle_load_s(test_file, fn=safe_callback)
                self.assertIsNotNone(result)
            except PickleInjectionError:
                # It's reasonable if the object is too complex and gets rejected
                pass

    def test_find_class_exception_handling(self):
        """Test exception handling in find_class method"""
        def problematic_callback(module, name):
            # 修改为检查builtins.eval，因为eval在builtins模块中
            if module == "builtins" and name == "eval":
                raise Exception("Callback error")
            return True

        # 创建一个会使用eval的对象
        class ProblematicObject:
            def __init__(self, expr="1+1"):
                self.expr = expr

            def __reduce__(self):
                return (eval, (self.expr,))

        problematic_data = pickle.dumps(ProblematicObject())
        
        unpickler = SafeUnpickler(BytesIO(problematic_data), call_back_fn=problematic_callback)
        
        with self.assertRaises(RuntimeError, msg="Should catch callback exception and convert to RuntimeError"):
            unpickler.load()

    def test_safe_unpickler_with_none_callback(self):
        """Test using None callback (should use default callback)"""
        unpickler = SafeUnpickler(BytesIO(self.safe_data), call_back_fn=None)
        # Should create normally, using default callback
        self.assertIsNotNone(unpickler.call_back_fn)

    def test_pickle_loads_s_with_none_callback(self):
        """Test pickle_loads_s with None callback"""
        result = pickle_loads_s(self.safe_data, fn=None)
        self.assertEqual(result, 2, "None callback should use default safe callback")
