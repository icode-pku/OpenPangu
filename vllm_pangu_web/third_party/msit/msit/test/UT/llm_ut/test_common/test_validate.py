# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
import functools
from unittest.mock import patch
from msit_llm.common.validate import validate_parameters_by_func, validate_parameters_by_type


class TestValidateParametersByType(unittest.TestCase):

    def test_valid_parameter_types(self):
        @validate_parameters_by_type({'arg1': [int, float], 'arg2': [str]})
        def sample_function(arg1, arg2):
            return f"{arg1} - {arg2}"

        # 测试有效输入
        self.assertEqual(sample_function(10, "test"), "10 - test")
        self.assertEqual(sample_function(10.5, "test"), "10.5 - test")

    def test_invalid_parameter_type(self):
        @validate_parameters_by_type({'arg1': [int], 'arg2': [str]})
        def sample_function(arg1, arg2):
            return f"{arg1} - {arg2}"

        # 测试无效输入
        with self.assertRaises(TypeError) as context:
            sample_function("not an int", "test")
        self.assertIn("expected", str(context.exception))

    def test_none_type_allowed(self):
        @validate_parameters_by_type({'arg1': [int, None], 'arg2': [str]})
        def sample_function(arg1, arg2):
            return f"{arg1} - {arg2}"

        # 测试 None 值
        self.assertEqual(sample_function(None, "test"), "None - test")

    def test_empty_parameter_constraints(self):
        with self.assertRaises(ValueError) as context:
            @validate_parameters_by_type({})
            def sample_function(arg):
                return arg
        self.assertEqual(str(context.exception), "Parameter constraints should not be empty.")

    def test_invalid_parameter_constraints_type(self):
        with self.assertRaises(TypeError) as context:
            @validate_parameters_by_type("invalid_type")
            def sample_function(arg):
                return arg
        self.assertIn("expects dict", str(context.exception))

    def test_invalid_key_in_constraints(self):
        with self.assertRaises(ValueError) as context:
            @validate_parameters_by_type({123: [int]})
            def sample_function(arg):
                return arg
        self.assertIn("only supports string", str(context.exception))

    def test_invalid_value_in_constraints(self):
        with self.assertRaises(ValueError) as context:
            @validate_parameters_by_type({'arg1': 123})
            def sample_function(arg):
                return arg
        self.assertIn("only supports tuple or list", str(context.exception))

    def test_function_with_kwargs(self):
        @validate_parameters_by_type({'arg1': [int], 'arg2': [str]})
        def sample_function(arg1, arg2):
            return f"{arg1} - {arg2}"

        # 测试有效的 kwargs
        self.assertEqual(sample_function(arg1=5, arg2="test"), "5 - test")

    def test_function_with_args(self):
        @validate_parameters_by_type({'arg1': [int], 'arg2': [str]})
        def sample_function(arg1, arg2):
            return f"{arg1} - {arg2}"

        # 测试有效的 args
        self.assertEqual(sample_function(3, "hello"), "3 - hello")


class TestValidateParametersByFunc(unittest.TestCase):
    def test_validate_parameters_by_func_type_error_when_not_dict(self):
        """
        测试当传入的parameter_constraints不是字典类型时，是否会抛出TypeError异常。
        """
        with self.assertRaises(TypeError):
            validate_parameters_by_func("not_a_dict")
 
    def test_validate_parameters_by_func_value_error_when_empty(self):
        """
        测试当传入的parameter_constraints为空字典时，是否会抛出ValueError异常。
        """
        with self.assertRaises(ValueError):
            validate_parameters_by_func({})
 
    def test_validate_parameters_by_func_value_error_when_key_not_str(self):
        """
        测试当parameter_constraints的键不是字符串类型时，是否会抛出ValueError异常。
        """
        constraints = {1: ["constraint"]}
        with self.assertRaises(ValueError):
            validate_parameters_by_func(constraints)
 
    def test_validate_parameters_by_func_value_error_when_value_not_tuple_or_list(self):
        """
        测试当parameter_constraints的值不是元组或列表类型时，是否会抛出ValueError异常。
        """
        constraints = {"key": "not_tuple_or_list"}
        with self.assertRaises(ValueError):
            validate_parameters_by_func(constraints)
 
    def test_decorator_with_in_class_false_valid_parameter(self):
        """
        测试当in_class为False时，装饰器对函数参数的约束验证功能是否正常。
        这里模拟传入合法参数，函数应该正常执行并返回预期结果。
        """
        def valid_check(arg):
            return isinstance(arg, int)
 
        parameter_constraints = {
            "arg1": [valid_check]
        }
 
        @validate_parameters_by_func(parameter_constraints, in_class=False)
        def test_valid_function(arg1):
            return arg1 + 1
 
        result = test_valid_function(5)
        self.assertEqual(result, 6)
 
    def test_decorator_with_in_class_false_invalid_parameter(self):
        """
        测试当in_class为False时，装饰器对函数参数的约束验证功能在传入无效参数时是否能正确抛出异常。
        """
        def valid_check(arg):
            return isinstance(arg, int)
 
        parameter_constraints = {
            "arg1": [valid_check]
        }
 
        @validate_parameters_by_func(parameter_constraints, in_class=False)
        def test_invalid_function(arg1):
            return arg1 + 1
 
        with self.assertRaises(RuntimeError):
            test_invalid_function("invalid_arg")
 
    def test_decorator_with_callable_check_raising_exception(self):
        """
        测试当约束检查函数（callable check_item）本身抛出异常时，装饰器是否能正确捕获并重新抛出RuntimeError异常。
        """
        def raising_check(arg):
            raise ValueError("Some error in check function")
 
        parameter_constraints = {
            "arg1": [raising_check]
        }
 
        @validate_parameters_by_func(parameter_constraints, in_class=False)
        def test_raising_function(arg1):
            return arg1 + 1
 
        with self.assertRaises(RuntimeError) as cm:
            test_raising_function(5)
 
        self.assertTrue(str(cm.exception).startswith("In the running function `test_raising_function`,"))
 
    def test_decorator_with_invalid_check_item_not_callable(self):
        """
        测试当约束中的检查项不是可调用对象时，装饰器是否能正确抛出TypeError异常。
        """
        parameter_constraints = {
            "arg1": ["not_callable_item"]
        }
 
        @validate_parameters_by_func(parameter_constraints, in_class=False)
        def test_non_callable_function(arg1):
            return arg1 + 1
 
        with self.assertRaises(TypeError) as cm:
            test_non_callable_function(5)
 
        self.assertIsInstance(cm.exception, TypeError)
