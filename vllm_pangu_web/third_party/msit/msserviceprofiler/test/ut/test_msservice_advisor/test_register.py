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
from unittest.mock import patch, MagicMock

import pytest

from msserviceprofiler.msservice_advisor.profiling_analyze.register import (
    register_analyze,
    cached,
    answer,
    REGISTRY,
    ANSWERS,
    SUGGESTION_TYPES,
)


# Fixture to reset the registry and answers between tests
@pytest.fixture(autouse=True)
def reset_state():
    """Reset the REGISTRY and ANSWERS before each test"""
    REGISTRY.clear()
    for key in ANSWERS:
        ANSWERS[key].clear()
    yield


# Test register_analyze decorator
def test_register_analyze_given_function_when_decorated_then_added_to_registry():
    @register_analyze()
    def test_func():
        return "test"

    assert "test_func" in REGISTRY
    assert REGISTRY["test_func"]() == "test"


def test_register_analyze_given_custom_name_when_decorated_then_added_with_custom_name():
    @register_analyze("custom_name")
    def test_func():
        return "test"

    assert "custom_name" in REGISTRY
    assert REGISTRY["custom_name"]() == "test"


def test_register_analyze_given_multiple_functions_when_decorated_then_all_added():
    @register_analyze("func1")
    def test_func1():
        return 1

    @register_analyze()
    def test_func2():
        return 2

    assert len(REGISTRY) == 2
    assert REGISTRY["func1"]() == 1
    assert REGISTRY["test_func2"]() == 2


# Test cached decorator
def test_cached_given_function_when_called_twice_then_caches_result():
    mock_func = MagicMock(return_value="result")

    @cached()
    def test_func():
        return mock_func()

    assert test_func() == "result"
    assert test_func() == "result"
    assert mock_func.call_count == 1


def test_cached_given_different_functions_when_called_then_caches_separately():
    @cached()
    def func1():
        return 1

    @cached()
    def func2():
        return 2

    assert func1() == 1
    assert func2() == 2
    assert func1() == 1  # Still returns cached 1


def test_cached_given_function_with_args_when_called_then_ignores_args_in_cache():
    mock_func = MagicMock(return_value="result")

    @cached()
    def test_func(arg1, arg2=None):
        return mock_func()

    assert test_func(1) == "result"
    assert test_func(2, arg2="test") == "result"  # Returns cached result
    assert mock_func.call_count == 1


# Test answer decorator
def test_answer_given_valid_inputs_when_called_then_updates_answers():
    answer("env", "memory", "increase", "needs more memory")

    assert "memory" in ANSWERS["env"]
    assert ("increase", "needs more memory") in ANSWERS["env"]["memory"]


def test_answer_given_multiple_calls_when_called_then_appends_to_list():
    answer("config", "timeout", "set", "to 30s")
    answer("config", "timeout", "verify", "value is correct")

    assert len(ANSWERS["config"]["timeout"]) == 2
    assert ("set", "to 30s") in ANSWERS["config"]["timeout"]
    assert ("verify", "value is correct") in ANSWERS["config"]["timeout"]


def test_answer_given_all_suggestion_types_when_called_then_updates_correct_type():
    for stype in SUGGESTION_TYPES:
        answer(stype, "test_item", "action", "reason")
        assert "test_item" in ANSWERS[stype]


def test_answer_given_unknown_suggestion_type_when_called_then_raises_keyerror():
    with pytest.raises(KeyError):
        answer("invalid_type", "item", "action", "reason")


# Test integration between components
def test_register_and_answer_integration():
    @register_analyze("memory_check")
    def check_memory():
        answer("env", "memory", "increase", "system needs more RAM")
        return True

    assert check_memory()
    assert "memory_check" in REGISTRY
    assert "memory" in ANSWERS["env"]
    assert ("increase", "system needs more RAM") in ANSWERS["env"]["memory"]


def test_cached_and_register_integration():
    mock_func = MagicMock(return_value="result")

    @register_analyze("cached_test")
    @cached()
    def test_func():
        return mock_func()

    # First call should execute
    assert test_func() == "result"
    # Second call should use cache
    assert test_func() == "result"
    assert mock_func.call_count == 1
    # Should also be in registry
    assert "cached_test" in REGISTRY
    assert REGISTRY["cached_test"]() == "result"
