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

from unittest.mock import Mock, patch, call
from unittest.mock import ANY

import pytest

from msserviceprofiler.vllm_profiler.dynamic_hook import (
    FuncCallContext, 
    DynamicHooker, 
    register_dynamic_hook, 
    make_default_time_hook, 
    HandlerResolver
)


@pytest.fixture
def sample_func_call_context():
    """提供示例函数调用上下文的 fixture"""
    mock_func = Mock()
    mock_this = Mock()
    mock_args = (1, 2, 3)
    mock_kwargs = {'key': 'value'}
    mock_ret_val = "result"
    
    return FuncCallContext(
        func_obj=mock_func,
        this_obj=mock_this,
        args=mock_args,
        kwargs=mock_kwargs,
        ret_val=mock_ret_val
    )


@pytest.fixture
def sample_hook_list():
    """提供示例 hook 列表的 fixture"""
    return [
        ('module.path', 'ClassName.method_name'),
        ('another.module', 'function_name')
    ]


@pytest.fixture
def mock_hook_func():
    """提供模拟 hook 函数的 fixture"""
    mock = Mock()
    mock.__name__ = "mock_hook_func"
    return mock


@pytest.fixture
def sample_attributes():
    """提供示例属性的 fixture"""
    return [
        {"name": "input_length", "expr": "len(kwargs['input_ids'])"},
        {"name": "output_length", "expr": "len(return)"},
        {"name": "model_name", "expr": "this.model_name"}
    ]


class TestFuncCallContext:
    """测试 FuncCallContext 数据类"""
    
    @staticmethod
    def test_func_call_context_initialization(sample_func_call_context):
        """测试 FuncCallContext 初始化"""
        ctx = sample_func_call_context
        
        assert ctx.func_obj is not None
        assert ctx.this_obj is not None
        assert ctx.args == (1, 2, 3)
        assert ctx.kwargs == {'key': 'value'}
        assert ctx.ret_val == "result"


class TestDynamicHooker:
    """测试 DynamicHooker 类"""

    @staticmethod
    def test_dynamic_hooker_initialization(sample_hook_list, mock_hook_func):
        """测试 DynamicHooker 初始化"""
        hooker = DynamicHooker(
            hook_list=sample_hook_list,
            hook_func=mock_hook_func,
            min_version="1.0",
            max_version="2.0",
            caller_filter="test_filter"
        )
        
        assert hooker.vllm_version == ("1.0", "2.0")
        assert hooker.applied_hook_func_name == mock_hook_func.__name__
        assert hooker.hook_list == sample_hook_list
        assert hooker.caller_filter == "test_filter"
        assert hooker.hook_func == mock_hook_func

    @staticmethod
    def test_dynamic_hooker_initialization_minimal(sample_hook_list, mock_hook_func):
        """测试 DynamicHooker 初始化（最小参数）"""
        hooker = DynamicHooker(
            hook_list=sample_hook_list,
            hook_func=mock_hook_func,
            min_version=None,
            max_version=None,
            caller_filter=None
        )
        
        assert hooker.vllm_version == (None, None)
        assert hooker.caller_filter is None

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.import_object_from_string')
    def test_dynamic_hooker_init(mock_import_object, sample_hook_list, mock_hook_func):
        """测试 DynamicHooker init 方法"""
        # 模拟导入的对象
        mock_point1 = Mock()
        mock_point2 = Mock()
        mock_import_object.side_effect = [mock_point1, mock_point2]
        
        hooker = DynamicHooker(
            hook_list=sample_hook_list,
            hook_func=mock_hook_func,
            min_version=None,
            max_version=None,
            caller_filter=None
        )
        
        # 模拟父类的 do_hook 方法
        with patch.object(hooker, 'do_hook') as mock_do_hook:
            hooker.init()
            
            # 验证导入调用
            assert mock_import_object.call_count == 2
            mock_import_object.assert_has_calls([
                call('module.path', 'ClassName.method_name'),
                call('another.module', 'function_name')
            ])
            
            # 验证 do_hook 调用
            mock_do_hook.assert_called_once()
            call_args = mock_do_hook.call_args
            assert call_args[1]['hook_points'] == [mock_point1, mock_point2]
            assert call_args[1]['pname'] is None


class TestRegisterDynamicHook:
    """测试 register_dynamic_hook 函数"""

    @staticmethod
    def test_register_dynamic_hook(sample_hook_list, mock_hook_func):
        """测试 register_dynamic_hook 函数"""
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook.DynamicHooker') as mock_dynamic_hooker:
            mock_hooker_instance = Mock()
            mock_dynamic_hooker.return_value = mock_hooker_instance
            
            result = register_dynamic_hook(
                hook_list=sample_hook_list,
                hook_func=mock_hook_func,
                min_version="1.0",
                max_version="2.0",
                caller_filter="test_filter"
            )
            
            # 验证 DynamicHooker 初始化
            mock_dynamic_hooker.assert_called_once_with(
                hook_list=sample_hook_list,
                hook_func=mock_hook_func,
                min_version="1.0",
                max_version="2.0",
                caller_filter="test_filter"
            )
            
            # 验证注册调用
            mock_hooker_instance.register.assert_called_once()
            assert result == mock_hooker_instance

    @staticmethod
    def test_register_dynamic_hook_default_args(sample_hook_list, mock_hook_func):
        """测试 register_dynamic_hook 函数（默认参数）"""
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook.DynamicHooker') as mock_dynamic_hooker:
            mock_hooker_instance = Mock()
            mock_dynamic_hooker.return_value = mock_hooker_instance
            
            result = register_dynamic_hook(
                hook_list=sample_hook_list,
                hook_func=mock_hook_func
            )
            
            mock_dynamic_hooker.assert_called_once_with(
                hook_list=sample_hook_list,
                hook_func=mock_hook_func,
                min_version=None,
                max_version=None,
                caller_filter=None
            )


class TestMakeDefaultTimeHook:
    """测试 make_default_time_hook 函数"""

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.Profiler')
    def test_make_default_time_hook_with_profiler_no_attributes(mock_profiler):
        """测试有 profiler 但无属性的情况"""
        mock_profiler_instance = Mock()
        mock_profiler.return_value.domain.return_value.span_start.return_value = mock_profiler_instance
        
        result_func = make_default_time_hook("test_domain", "test_name")
        
        mock_original = Mock(return_value="result")
        mock_args = (1, 2, 3)
        mock_kwargs = {'key': 'value'}
        
        result = result_func(mock_original, *mock_args, **mock_kwargs)
        
        # 验证 Profiler 调用
        mock_profiler.assert_called_once()
        mock_profiler_instance.span_end.assert_called_once()
        mock_original.assert_called_once_with(*mock_args, **mock_kwargs)
        assert result == "result"

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.Profiler')
    def test_make_default_time_hook_with_attributes(mock_profiler, sample_attributes):
        """测试有属性和 profiler 的情况"""
        mock_profiler_instance = Mock()
        mock_profiler.return_value.domain.return_value.span_start.return_value = mock_profiler_instance
        
        result_func = make_default_time_hook("test_domain", "test_name", sample_attributes)
        
        mock_original = Mock(return_value="result")
        mock_args = (Mock(),)  # 第一个参数作为 self
        mock_kwargs = {'input_ids': [1, 2, 3]}
        
        # 模拟内部函数的行为
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook._safe_eval_expr') as mock_safe_eval:
            mock_safe_eval.side_effect = [6, 3, "test_model"]  # 模拟三个属性的返回值
            
            result = result_func(mock_original, *mock_args, **mock_kwargs)
            
            # 验证属性设置
            assert mock_profiler_instance.attr.call_count == 3
            mock_profiler_instance.attr.assert_has_calls([
                call("input_length", 6),
                call("output_length", 3),
                call("model_name", "test_model")
            ])

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.Profiler')
    def test_make_default_time_hook_attribute_eval_failure(mock_profiler, sample_attributes):
        """测试属性表达式执行失败的情况"""
        mock_profiler_instance = Mock()
        mock_profiler.return_value.domain.return_value.span_start.return_value = mock_profiler_instance
        
        result_func = make_default_time_hook("test_domain", "test_name", sample_attributes)
        
        mock_original = Mock(return_value="result")
        mock_args = (Mock(),)
        mock_kwargs = {'input_ids': [1, 2, 3]}
        
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook._safe_eval_expr') as mock_safe_eval:
            mock_safe_eval.return_value = None  # 所有表达式执行失败
            
            result = result_func(mock_original, *mock_args, **mock_kwargs)
            
            # 验证没有属性被设置
            mock_profiler_instance.attr.assert_not_called()

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.Profiler')
    def test_make_default_time_hook_invalid_attributes(mock_profiler):
        """测试无效属性配置的情况"""
        mock_profiler_instance = Mock()
        mock_profiler.return_value.domain.return_value.span_start.return_value = mock_profiler_instance
        
        invalid_attributes = [
            {"name": "valid", "expr": "len(args)"},  # 有效
            {"name": "", "expr": "len(kwargs)"},     # 无效：空名称
            {"name": "no_expr"},                     # 无效：缺少表达式
            {"expr": "len(return)"},                 # 无效：缺少名称
        ]
        
        result_func = make_default_time_hook("test_domain", "test_name", invalid_attributes)
        
        mock_original = Mock(return_value="result")
        
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook._safe_eval_expr') as mock_safe_eval:
            mock_safe_eval.return_value = 5
            
            result = result_func(mock_original, 1, 2, 3)
            
            # 只有第一个有效属性被处理
            mock_safe_eval.assert_called_once_with("len(args)", ANY)
            mock_profiler_instance.attr.assert_called_once_with("valid", 5)


class TestHandlerResolver:
    """测试 HandlerResolver 类"""
    
    @staticmethod
    def test_handler_resolver_initialization():
        """测试 HandlerResolver 初始化"""
        resolver = HandlerResolver(prefer_builtin=True)
        assert resolver.prefer_builtin is True
        
        resolver2 = HandlerResolver(prefer_builtin=False)
        assert resolver2.prefer_builtin is False

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.importlib.import_module')
    def test_try_import_success(mock_import_module):
        """测试成功导入 handler"""
        mock_module = Mock()
        mock_handler = Mock()
        mock_import_module.return_value = mock_module
        mock_module.test_handler = mock_handler
        
        result = HandlerResolver._try_import("some.module:test_handler")
        
        mock_import_module.assert_called_once_with("some.module")
        assert result == mock_handler

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.importlib.import_module')
    def test_try_import_module_not_found(mock_import_module):
        """测试导入模块失败"""
        mock_import_module.side_effect = ImportError("Module not found")
        
        result = HandlerResolver._try_import("nonexistent.module:handler")
        
        assert result is None

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.importlib.import_module')
    def test_try_import_function_not_found(mock_import_module):
        """测试导入函数不存在"""
        mock_module = Mock()
        mock_module.test_handler = None  # 函数不存在
        mock_import_module.return_value = mock_module
        
        result = HandlerResolver._try_import("some.module:nonexistent_handler")
        
        assert result is None

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_explicit_timer(mock_make_default):
        """测试解析显式 timer handler"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain",
            "name": "TestName",
            "handler": "timer"
        }
        points = [('module', 'function')]
        
        result = resolver.resolve(item, points)
        
        mock_make_default.assert_called_once_with("TestDomain", "TestName", None)
        assert result == mock_timer

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_none_handler(mock_make_default):
        """测试解析 None handler（隐式 timer）"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain",
            "name": "TestName"
            # 没有 handler
        }
        points = [('module', 'function')]
        
        result = resolver.resolve(item, points)
        
        mock_make_default.assert_called_once_with("TestDomain", "TestName", None)
        assert result == mock_timer

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.HandlerResolver._try_import')
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_custom_handler_success(mock_make_default, mock_try_import):
        """测试成功解析自定义 handler"""
        mock_custom_handler = Mock()
        mock_try_import.return_value = mock_custom_handler
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain",
            "name": "TestName",
            "handler": "custom.module:handler_func"
        }
        points = [('module', 'function')]
        
        result = resolver.resolve(item, points)
        
        mock_try_import.assert_called_once_with("custom.module:handler_func")
        assert result == mock_custom_handler
        mock_make_default.assert_not_called()

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.HandlerResolver._try_import')
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_custom_handler_fallback(mock_make_default, mock_try_import):
        """测试自定义 handler 导入失败回退到 timer"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        mock_try_import.return_value = None  # 导入失败
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain",
            "name": "TestName",
            "handler": "custom.module:handler_func"
        }
        points = [('module', 'function')]
        
        result = resolver.resolve(item, points)
        
        mock_try_import.assert_called_once_with("custom.module:handler_func")
        mock_make_default.assert_called_once_with("TestDomain", "TestName", None)
        assert result == mock_timer

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_other_handler_value(mock_make_default):
        """测试解析其他 handler 值（回退到 timer）"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain",
            "name": "TestName",
            "handler": "builtin"  # 不是 "timer" 或自定义导入格式
        }
        points = [('module', 'function')]
        
        result = resolver.resolve(item, points)
        
        mock_make_default.assert_called_once_with("TestDomain", "TestName", None)
        assert result == mock_timer

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_with_attributes(mock_make_default):
        """测试解析带有属性的 handler"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        
        resolver = HandlerResolver()
        attributes = [{"name": "test_attr", "expr": "len(args)"}]
        item = {
            "domain": "TestDomain",
            "name": "TestName",
            "handler": "timer",
            "attributes": attributes
        }
        points = [('module', 'function')]
        
        result = resolver.resolve(item, points)
        
        mock_make_default.assert_called_once_with("TestDomain", "TestName", attributes)
        assert result == mock_timer

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_name_from_points(mock_make_default):
        """测试从 points 中提取名称"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain"
            # 没有 name，应该从 points 中提取
        }
        points = [('module', 'function_name')]
        
        result = resolver.resolve(item, points)
        
        mock_make_default.assert_called_once_with("TestDomain", "function_name", None)
        assert result == mock_timer

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.dynamic_hook.make_default_time_hook')
    def test_resolve_default_name(mock_make_default):
        """测试使用默认名称"""
        mock_timer = Mock()
        mock_make_default.return_value = mock_timer
        
        resolver = HandlerResolver()
        item = {
            "domain": "TestDomain"
            # 没有 name，也没有 points
        }
        points = []  # 空列表
        
        result = resolver.resolve(item, points)
        
        mock_make_default.assert_called_once_with("TestDomain", "custom", None)
        assert result == mock_timer


class TestInternalFunctions:
    """测试内部辅助函数"""
    
    @staticmethod
    def test_get_object_attribute(sample_func_call_context):
        """测试 _get_object_attribute 函数"""
        
        # 创建 hook 函数以访问内部函数
        hook_func = make_default_time_hook("test", "test")
        
        # 测试获取存在的属性
        mock_obj = Mock()
        mock_obj.test_attr = "test_value"
        result = hook_func.__globals__['_get_object_attribute'](mock_obj, "test_attr")
        assert result == "test_value"
        
        # 测试获取不存在的属性
        result = hook_func.__globals__['_get_object_attribute'](mock_obj, "nonexistent_attr")
        assert result is None

    @staticmethod
    def test_extract_named_parameters(sample_func_call_context):
        """测试 _extract_named_parameters 函数"""
        
        hook_func = make_default_time_hook("test", "test")
        
        # 创建有签名的函数
        def test_func(a, b, c=3, d=4):
            pass
        
        args = (1, 2)
        kwargs = {'d': 5}
        
        result = hook_func.__globals__['_extract_named_parameters'](test_func, args, kwargs)
        
        assert 'a' in result
        assert 'b' in result
        assert 'c' in result  # 使用默认值
        assert 'd' in result
        assert result['a'] == 1
        assert result['b'] == 2
        assert result['c'] == 3
        assert result['d'] == 5

    @staticmethod
    def test_build_safe_locals(sample_func_call_context):
        """测试 _build_safe_locals 函数"""

        hook_func = make_default_time_hook("test", "test")

        # 模拟有 self 参数的情况
        mock_self = Mock()
        mock_self.model_name = "test_model"

        def test_func(self, arg1, arg2):
            pass

        ctx = FuncCallContext(
            func_obj=test_func,
            this_obj=mock_self,
            args=(mock_self, "arg1_value", "arg2_value"),
            kwargs={},
            ret_val="result_value"
        )

        safe_locals = hook_func.__globals__['_build_safe_locals'](ctx)

        # 验证基本变量
        assert safe_locals['this'] == mock_self
        assert safe_locals['args'] == (mock_self, "arg1_value", "arg2_value")
        assert safe_locals['kwargs'] == {}
        assert safe_locals['return'] == "result_value"

        # 验证具名参数
        assert 'self' in safe_locals
        assert 'arg1' in safe_locals
        assert 'arg2' in safe_locals

    @staticmethod
    @pytest.mark.parametrize("expr,expected", [
        ("len(args)", True),  # 安全表达式
        ("import os", False),  # 危险关键字
        ("__import__('os')", False),  # 危险函数
        ("eval('1+1')", False),  # 危险函数
        ("args[0] + args[1]", False),  # 危险操作符
        ("len(kwargs.get('key', []))", True),  # 安全函数调用
        ("unknown_func()", False),  # 未知函数
        ("(1 + 2) * 3", False),  # 算术运算
        ("args[0] | len", True),  # 管道操作（在后续验证）
    ])
    def test_validate_expression_safety(expr, expected):
        """测试 _validate_expression_safety 函数"""
        
        hook_func = make_default_time_hook("test", "test")
        
        result = hook_func.__globals__['_validate_expression_safety'](expr)
        assert result == expected

    @staticmethod
    def test_execute_direct_expression(sample_func_call_context):
        """测试 _execute_direct_expression 函数"""
        
        hook_func = make_default_time_hook("test", "test")
        
        safe_locals = {
            'args': (1, 2, 3),
            'kwargs': {'key': 'value'},
            'return': "result",
            'len': len,
            'str': str
        }
        
        # 测试安全表达式
        result = hook_func.__globals__['_execute_direct_expression']("len(args)", safe_locals)
        assert result == 3
        
        # 测试危险表达式（应该返回 None）
        result = hook_func.__globals__['_execute_direct_expression']("import os", safe_locals)
        assert result is None
        
        # 测试无效表达式
        result = hook_func.__globals__['_execute_direct_expression']("invalid_syntax", safe_locals)
        assert result is None

    @staticmethod
    @pytest.mark.parametrize("input_val,operation,expected", [
        ([1, 2, 3], 'len', 3),  # len 操作
        ("hello", 'str', "hello"),  # str 操作
        (Mock(test_attr="value"), 'attr test_attr', "value"),  # attr 操作
        ([1, 2, 3], 'unknown', None),  # 未知操作
        (None, 'len', None),  # None 输入
    ])
    def test_apply_pipe_operation(input_val, operation, expected):
        """测试 _apply_pipe_operation 函数"""
        
        hook_func = make_default_time_hook("test", "test")
        
        result = hook_func.__globals__['_apply_pipe_operation'](input_val, operation)
        
        if expected is None:
            assert result is None
        else:
            assert result == expected

    @staticmethod
    def test_execute_pipe_expression(sample_func_call_context):
        """测试 _execute_pipe_expression 函数"""
        
        hook_func = make_default_time_hook("test", "test")
        
        safe_locals = {
            'args': ([1, 2, 3],),
            'kwargs': {'key': 'value'},
            'return': "hello world",
            'len': len,
            'str': str
        }
        
        # 测试简单表达式
        result = hook_func.__globals__['_execute_pipe_expression']("len(args[0])", safe_locals)
        assert result == 3
        
        # 测试管道表达式
        result = hook_func.__globals__['_execute_pipe_expression']("args[0] | len", safe_locals)
        assert result == 3
        
        # 测试多步管道
        result = hook_func.__globals__['_execute_pipe_expression']("return | len | str", safe_locals)
        assert result == "11"  # len("hello world") = 11, then str(11) = "11"

    @staticmethod
    def test_safe_eval_expr(sample_func_call_context):
        """测试 _safe_eval_expr 函数"""
        
        hook_func = make_default_time_hook("test", "test")
        
        # 模拟成功的表达式执行
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook._build_safe_locals') as mock_build_locals, \
             patch('msserviceprofiler.vllm_profiler.dynamic_hook._execute_pipe_expression') as mock_execute:
            mock_build_locals.return_value = {'args': (1, 2, 3), 'len': len}
            mock_execute.return_value = 3
            
            result = hook_func.__globals__['_safe_eval_expr']("len(args)", sample_func_call_context)
            
            mock_build_locals.assert_called_once_with(sample_func_call_context)
            mock_execute.assert_called_once_with("len(args)", {'args': (1, 2, 3), 'len': len})
            assert result == 3
        
        # 模拟表达式执行失败
        with patch('msserviceprofiler.vllm_profiler.dynamic_hook._build_safe_locals') as mock_build_locals, \
             patch('msserviceprofiler.vllm_profiler.dynamic_hook._execute_pipe_expression') as mock_execute:
            mock_build_locals.return_value = {'args': (1, 2, 3), 'len': len}
            mock_execute.side_effect = Exception("Test error")
            
            result = hook_func.__globals__['_safe_eval_expr']("len(args)", sample_func_call_context)
            
            assert result is None
