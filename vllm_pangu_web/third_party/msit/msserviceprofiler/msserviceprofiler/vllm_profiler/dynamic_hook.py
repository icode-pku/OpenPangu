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

import importlib
import inspect
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Dict, Any

from ms_service_profiler import Profiler, Level

from .logger import logger
from .module_hook import VLLMHookerBase, import_object_from_string


@dataclass
class FuncCallContext:
    """函数调用上下文，封装表达式执行所需的参数。
    
    Attributes:
        func_obj: 被调用的函数对象
        this_obj: this 对象（通常是 self）
        args: 位置参数元组
        kwargs: 关键字参数字典
        ret_val: 函数返回值
    """
    func_obj: Any
    this_obj: Any
    args: Tuple
    kwargs: Dict
    ret_val: Any


class DynamicHooker(VLLMHookerBase):
    """用于在运行时基于配置注册的 Hooker。
    
    该类继承自 VLLMHookerBase，提供基于配置文件的动态 hook 功能。
    支持版本范围限制和调用者过滤。
    
    Attributes:
        vllm_version (Tuple[Optional[str], Optional[str]]): 支持的 vLLM 版本范围
        applied_hook_func_name (str): 应用的 hook 函数名称
        hook_list (List[Tuple[str, str]]): hook 点列表
        caller_filter (Optional[str]): 调用者过滤条件
        hook_func (Callable): hook 处理函数
    """
    
    def __init__(self, hook_list: List[Tuple[str, str]], hook_func: Callable,
                 min_version: Optional[str], max_version: Optional[str], caller_filter: Optional[str]):
        """初始化 DynamicHooker。
        
        Args:
            hook_list: hook 点列表，格式为 [(import_path, func_path), ...]
            hook_func: hook 处理函数
            min_version: 支持的最小版本
            max_version: 支持的最大版本
            caller_filter: 调用者过滤条件
        """
        super().__init__()
        self.vllm_version = (min_version, max_version)
        self.applied_hook_func_name = getattr(hook_func, "__name__", str(hook_func))
        self.hook_list = list(hook_list)
        self.caller_filter = caller_filter
        self.hook_func = hook_func

    def init(self):
        """初始化 hook 点并应用 hooks。
        
        从 hook_list 中导入目标对象，然后应用 hook 处理函数。
        """
        points = [import_object_from_string(import_path, func_path) for import_path, func_path in self.hook_list]
        self.do_hook(
            hook_points=points,
            profiler_func_maker=lambda ori_func: lambda *args, **kwargs: self.hook_func(ori_func, *args, **kwargs),
            pname=self.caller_filter,
        )


def register_dynamic_hook(hook_list: List[Tuple[str, str]], hook_func: Callable,
                          min_version: Optional[str] = None, max_version: Optional[str] = None,
                          caller_filter: Optional[str] = None):
    """注册一个基于配置文件的动态 Hooker。
    
    Args:
        hook_list: hook 点列表，格式为 [(import_path, func_path), ...]
        hook_func: hook 处理函数
        min_version: 支持的最小版本，默认为 None
        max_version: 支持的最大版本，默认为 None
        caller_filter: 调用者过滤条件，默认为 None
    
    Returns:
        DynamicHooker: 注册的 hooker 实例
    """
    hooker = DynamicHooker(
        hook_list=hook_list,
        hook_func=hook_func,
        min_version=min_version,
        max_version=max_version,
        caller_filter=caller_filter,
    )
    hooker.register()
    return hooker


def _get_object_attribute(obj, attr_name):
    """获取对象属性。对于不存在的属性返回 None"""
    try:
        return object.__getattribute__(obj, attr_name)
    except Exception:
        return None


def _extract_named_parameters(func_obj, args_tuple, kwargs_dict):
    """提取函数的具名参数。"""
    try:
        sig = inspect.signature(func_obj)
        bound = sig.bind_partial(*args_tuple, **kwargs_dict)
        try:
            bound.apply_defaults()
        except Exception as e:
            logger.debug("apply_defaults failed: %s", e)
        return dict(bound.arguments)
    except Exception:
        return {}


def _build_safe_locals(ctx: FuncCallContext):
    """构建安全的本地变量环境。"""
    named_params = _extract_named_parameters(ctx.func_obj, ctx.args, ctx.kwargs)
    self_obj = named_params.get("self", ctx.this_obj)
    safe_locals = {
        "this": self_obj,
        "args": ctx.args,
        "kwargs": ctx.kwargs,
        "return": ctx.ret_val,
        "self": self_obj,
        "len": len,
        "str": str,
        "attr": _get_object_attribute,
    }
    try:
        safe_locals.update({k: v for k, v in named_params.items() if k != "self"})
    except Exception as e:
        logger.warning(f"Failed to update safe locals: {e}")
    return safe_locals


def _validate_expression_safety(expr_str):
    """验证表达式安全性，只允许预定义的安全操作。"""
    dangerous_chars = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'raw_input']
    # 允许管道符 '|'
    dangerous_ops = ['+', '-', '*', '/', '%', '**', '//', '&', '^', '~', '<<', '>>']
    expr_lower = expr_str.lower()
    for dangerous in dangerous_chars:
        if dangerous in expr_lower:
            logger.warning(f"Expression contains dangerous keyword: {dangerous}")
            return False
    for op in dangerous_ops:
        if op in expr_str:
            logger.warning(f"Expression contains dangerous operator: {op}")
            return False
    if expr_str.count('(') != expr_str.count(')'):
        logger.warning(f"Unmatched parentheses in expression: {expr_str}")
        return False
    if '(' in expr_str and ')' in expr_str:
        func_name = expr_str.split('(')[0].strip()
        allowed_functions = ['len', 'str', 'int', 'float', 'bool', 'attr']
        if func_name not in allowed_functions:
            logger.warning(f"Function call not allowed: {func_name}")
            return False
    return True


def _execute_direct_expression(expr_str, safe_locals):
    """安全执行表达式，严格控制输入参数。"""
    try:
        if not _validate_expression_safety(expr_str):
            return None
        # 特殊处理关键名 'return'（不是合法标识符）
        trimmed = expr_str.strip()
        if trimmed == 'return':
            return safe_locals.get('return')
        safe_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            }
        }
        return eval(expr_str, safe_globals, safe_locals)
    except Exception as e:
        logger.warning(f"Safe eval failed: {expr_str}, err={e}")
        return None


def _apply_pipe_operation(result, operation):
    """应用单个管道操作。"""
    if operation == 'str':
        return str(result)
    elif operation == 'len':
        return len(result) if result is not None else None
    elif operation.startswith('attr '):
        attr_name = operation[5:].strip()
        return _get_object_attribute(result, attr_name)
    else:
        logger.warning(f"Unknown pipe operation: {operation}")
        return None


def _execute_pipe_expression(expr_str, safe_locals):
    """执行管道表达式。"""
    if '|' not in expr_str:
        return _execute_direct_expression(expr_str, safe_locals)
    parts = [part.strip() for part in expr_str.split('|')]
    if len(parts) < 2:
        return _execute_direct_expression(expr_str, safe_locals)
    result = _execute_direct_expression(parts[0], safe_locals)
    for operation in parts[1:]:
        result = _apply_pipe_operation(result, operation)
        if result is None:
            break
    return result


def _safe_eval_expr(expr: str, ctx: FuncCallContext):
    """安全执行表达式，支持管道操作和 attr 操作。"""
    try:
        safe_locals = _build_safe_locals(ctx)
        return _execute_pipe_expression(expr, safe_locals)
    except Exception as e:
        logger.warning(f"Pipe eval failed: {expr}, err={e}")
        return None


@dataclass
class AttrContext:
    """属性采集上下文。"""
    prof: Any
    attributes: Optional[List[Dict[str, Any]]]
    real_func: Callable
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    ret_val: Any


def _collect_attributes(ctx: AttrContext):
    """采集自定义属性并设置到 Profiler。
    
    Args:
        ctx: 属性采集上下文（包含 profiler、配置、原函数、入参与返回值）
    """
    if not isinstance(ctx.attributes, list):
        return
    
    for item in ctx.attributes:
        attr_name = item.get("name")
        expr = item.get("expr")
        if not attr_name or not expr:
            continue
        
        call_ctx = FuncCallContext(
            func_obj=ctx.real_func,
            this_obj=ctx.args[0] if ctx.args else None,
            args=ctx.args,
            kwargs=ctx.kwargs,
            ret_val=ctx.ret_val,
        )
        val = _safe_eval_expr(expr, call_ctx)
        if val is not None:
            ctx.prof.attr(attr_name, val)


def make_default_time_hook(domain: str, name: str, attributes: Optional[List[Dict[str, Any]]] = None) -> Callable:
    """生成一个默认的耗时统计 hook 处理函数，并支持自定义属性采集。
    
    该函数创建一个用于性能分析的 hook 处理函数，支持：
    - 基本的耗时统计
    - 自定义属性采集
    - 安全的表达式执行
    - 自动识别同步/异步函数
    
    Args:
        domain: 性能分析域名称
        name: hook 名称
        attributes: 自定义属性采集配置列表，可选。每个属性包含：
            - name (str): 采集项名称
            - expr (str): 表达式，如 "len(kwargs['input_ids'])"、"str(args[0])"、"kwargs['any'].attr"
    
    Returns:
        Callable: hook 处理函数
        
    Note:
        attributes 结构示例：
        [
            {"name": "input_length", "expr": "len(kwargs['input_ids'])"},
            {"name": "output_length", "expr": "len(return)"},
            {"name": "model_name", "expr": "this.model_name"}
        ]
    """
    if Profiler is None:
        def _noop(original_func, *args, **kwargs):
            return original_func(*args, **kwargs)
        return _noop

    def _default(original_func, *args, **kwargs):
        """默认的 hook 处理函数，自动处理同步/异步函数。
        
        Args:
            original_func: 原始函数或 TrackableOriginalFunc
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            原始函数的返回值（同步）或协程对象（异步）
        """
        # 获取真正的函数对象用于类型判断
        real_func = getattr(original_func, "original_func", original_func)
        
        # 异步函数：返回协程，在协程内部计时
        if inspect.iscoroutinefunction(real_func):
            async def _async_wrapper():
                prof = Profiler(Level.INFO).domain(domain).span_start(name)
                try:
                    ret = await original_func(*args, **kwargs)
                    _collect_attributes(
                        AttrContext(
                            prof=prof,
                            attributes=attributes,
                            real_func=real_func,
                            args=args,
                            kwargs=kwargs,
                            ret_val=ret,
                        )
                    )
                    return ret
                finally:
                    prof.span_end()
            
            return _async_wrapper()
        # 同步函数：直接在当前调用栈内计时
        else:
            prof = Profiler(Level.INFO).domain(domain).span_start(name)
            try:
                ret = original_func(*args, **kwargs)
                _collect_attributes(
                    AttrContext(
                        prof=prof,
                        attributes=attributes,
                        real_func=real_func,
                        args=args,
                        kwargs=kwargs,
                        ret_val=ret,
                    )
                )
                return ret
            finally:
                prof.span_end()

    return _default


class HandlerResolver:
    """Handler解析器：有可导入的自定义 handler 则用之，否则使用 timer。
    
    该类负责根据配置解析 handler 函数，支持：
    - 内置 timer handler
    - 自定义导入的 handler
    - 自动回退机制
    
    Attributes:
        prefer_builtin (bool): 是否优先使用内置 handler
    """
    
    def __init__(self, prefer_builtin: bool = True):
        """初始化 HandlerResolver。
        
        Args:
            prefer_builtin: 是否优先使用内置 handler，默认为 True
        """
        self.prefer_builtin = prefer_builtin

    @staticmethod
    def resolve(item: Dict[str, Any], points: List[Tuple[str, str]]) -> Callable:
        """根据配置解析handler函数。
        
        Args:
            item: 配置项字典
            points: hook 点列表
            
        Returns:
            Callable: 解析出的 handler 函数
        """
        domain = item.get("domain") or "Custom"
        name = item.get("name") or (points[0][1] if points else "custom")
        attributes = item.get("attributes")  # 新增：自定义属性采集配置
        handler_val = item.get("handler")
        handler_lower = handler_val.lower() if isinstance(handler_val, str) else None

        # 显式 timer
        if handler_lower == "timer" or handler_val is None:
            return make_default_time_hook(domain, name, attributes)

        # 自定义 import 形式
        if isinstance(handler_val, str) and ":" in handler_val:
            func = HandlerResolver._try_import(handler_val)
            if func is not None:
                return func
            logger.warning(f"Failed to import handler '{handler_val}', fallback to timer")
            return make_default_time_hook(domain, name, attributes)

        # 其他值（含 builtin）一律按 timer 处理
        return make_default_time_hook(domain, name, attributes)

    @staticmethod
    def _try_import(handler_val: str) -> Optional[Callable]:
        """尝试按 'pkg.mod:func' 导入自定义 handler，失败返回 None。
        
        Args:
            handler_val: handler 路径字符串，格式为 'pkg.mod:func'
            
        Returns:
            Optional[Callable]: 导入的 handler 函数，失败时返回 None
        """
        try:
            mod, func_name = handler_val.split(":", 1)
            mod_obj = importlib.import_module(mod)
            # Avoid Mock auto-creation: inspect module dict directly
            value = getattr(mod_obj, "__dict__", {}).get(func_name, None)
            return value if callable(value) else None
        except Exception:
            return None
