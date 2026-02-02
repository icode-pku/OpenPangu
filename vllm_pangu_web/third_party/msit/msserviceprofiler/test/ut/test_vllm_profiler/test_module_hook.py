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
import sys
import importlib
import inspect
from unittest.mock import patch, MagicMock
from packaging.version import Version
import pytest

from msserviceprofiler.vllm_profiler.module_hook import (
    import_object_from_string,
    HookHelper,
    VLLMHookerBase,
    vllm_hook
)
from msserviceprofiler.vllm_profiler.registry import get_hook_registry, clear_hook_registry


# Test module setup
@pytest.fixture
def cleanup_hook_registry():
    """Clear hook registry before each test"""
    clear_hook_registry()


# Test cases for import_object_from_string
def test_import_object_from_string_given_valid_path_when_importing_module_then_returns_object():
    """Test importing a valid module-level function"""
    result = import_object_from_string("os", "path")
    assert result == importlib.import_module("os").path


def test_import_object_from_string_given_nested_attribute_when_importing_then_returns_object():
    """Test importing nested attributes"""
    result = import_object_from_string("collections", "defaultdict.__class__")
    from collections import defaultdict

    assert result == defaultdict.__class__


def test_import_object_from_string_given_invalid_module_when_importing_then_returns_none():
    """Test handling of non-existent module"""
    result = import_object_from_string("nonexistent_module", "anything")
    assert result is None


def test_import_object_from_string_given_invalid_attribute_when_importing_then_returns_none():
    """Test handling of non-existent attribute"""
    result = import_object_from_string("os", "nonexistent_attr")
    assert result is None


def test_import_object_from_string_given_empty_path_when_importing_then_returns_none():
    """Test handling of empty path"""
    result = import_object_from_string("", "")
    assert result is None


# Test cases for HookHelper
class SampleClass:
    @staticmethod
    def static_method():
        return "original static"

    @classmethod
    def class_method(cls):
        return "original class"

    def instance_method(self):
        return "original instance"


def sample_function():
    return "original function"


def test_hookhelper_get_location_given_function_when_getting_location_then_returns_correct_info():
    """Test getting location info for regular function"""
    location, attr_name = HookHelper.get_location(sample_function)
    assert attr_name == "sample_function"
    assert location.__name__.split(".")[-1] == "test_module_hook"


def test_hookhelper_get_location_given_class_method_when_getting_location_then_returns_correct_info():
    """Test getting location info for class method"""
    location, attr_name = HookHelper.get_location(SampleClass.instance_method)
    assert attr_name == "instance_method"
    assert location.__name__ == "SampleClass"


def test_hookhelper_replace_given_function_when_replacing_then_successful():
    """Test function replacement"""
    original = sample_function

    def new_func():
        return "new function"

    helper = HookHelper(original, new_func)
    helper.replace()
    assert sample_function() == "new function"
    helper.recover()


def test_hookhelper_replace_given_static_method_when_replacing_then_successful():
    """Test static method replacement"""
    original = SampleClass.static_method

    def new_func():
        return "new static"

    helper = HookHelper(original, new_func)
    helper.replace()
    assert SampleClass.static_method() == "new static"
    helper.recover()


def test_hookhelper_replace_given_non_callable_when_initializing_then_raises_error():
    """Test handling of non-callable replacement"""
    with pytest.raises(ValueError):
        HookHelper("not a function", lambda x: x)


# Test cases for VLLMHookerBase
class FakeHooker(VLLMHookerBase):
    def init(self):
        def profiler_maker(ori_func):
            return lambda *args, **kwargs: "profiled:" + ori_func(*args, **kwargs)

        self.do_hook([sample_function], profiler_maker)


def test_vllmhookerbase_support_version_given_version_in_range_when_checking_then_returns_true():
    """Test version support within range"""
    hooker = FakeHooker()
    hooker.vllm_version = ("1.0.0", "2.0.0")
    assert hooker.support_version("1.5.0")


def test_vllmhookerbase_support_version_given_version_out_of_range_when_checking_then_returns_false():
    """Test version support outside range"""
    hooker = FakeHooker()
    hooker.vllm_version = ("1.0.0", "2.0.0")
    assert not hooker.support_version("3.0.0")


def test_vllmhookerbase_do_hook_given_hook_points_when_applying_then_functions_replaced(cleanup_hook_registry):
    """Test hook application"""
    hooker = FakeHooker()
    hooker.init()
    assert sample_function().startswith("profiled:")
    hooker.hooks[0].recover()


# Test cases for vllm_hook decorator
@vllm_hook(
    hook_points=[("vllm_profiler.module_hook", "sample_function")],
    min_version="1.0.0",
    max_version="2.0.0",
)
def sample_profiler(ori_func, *args, **kwargs):
    return "decorator:" + ori_func(*args, **kwargs)


# Additional edge case tests
def test_hookhelper_replace_given_missing_parent_class_when_replacing_then_raises_error():
    """Test handling of missing parent class during replacement"""

    class FakeFunc:
        __module__ = "builtins"
        __qualname__ = "NonExistentClass.method"

    with pytest.raises(ValueError):
        HookHelper(FakeFunc(), lambda x: x)


def test_import_object_from_string_given_malformed_path_when_importing_then_returns_none():
    """Test handling of malformed import path"""
    result = import_object_from_string("os.path", "join..split")
    assert result is None


def test_vllm_hook_given_empty_hook_points_when_registering_then_no_error(cleanup_hook_registry):
    """Test handling of empty hook points"""

    @vllm_hook(hook_points=[])
    def empty_profiler(ori_func, *args, **kwargs):
        pass

    assert len(get_hook_registry()) == 1
    # Shouldn't raise when init is called
    get_hook_registry()[0].init()


def test_vllmhookerbase_do_hook_given_caller_filter_when_calling_then_filters_correctly():
    """Test caller filter functionality"""

    class FilterHooker(VLLMHookerBase):
        def init(self):
            def profiler_maker(ori_func):
                return lambda *args, **kwargs: "filtered"

            self.do_hook([sample_function], profiler_maker, pname="test_caller")

    def test_caller():
        return sample_function()

    hooker = FilterHooker()
    hooker.init()

    flag = {"match": False}

    def fake_get_parents_name(*args, **kwargs):
        return "test_caller" if flag["match"] else "non_match"

    with patch(
        "msserviceprofiler.vllm_profiler.module_hook.get_parents_name",
        side_effect=fake_get_parents_name,
    ):
        # When caller name does not match, the hook should be bypassed.
        assert sample_function() == "original function"

        # Flip the flag so that the next invocation reports the expected caller.
        flag["match"] = True
        assert test_caller() == "filtered"

    hooker.hooks[0].recover()
