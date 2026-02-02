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
# limitations under the License.zn
from collections import namedtuple
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from msserviceprofiler.msservice_advisor.profiling_analyze import base_analyze
from msserviceprofiler.msservice_advisor.profiling_analyze.register import REGISTRY, ANSWERS
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import TARGETS, SUGGESTION_TYPES


# Test fixtures
@pytest.fixture(autouse=True)
def reset_state():
    """Reset the REGISTRY and ANSWERS before each test"""
    REGISTRY.clear()
    for key in ANSWERS:
        ANSWERS[key].clear()
    yield


# Test get_dict_value_by_pos
def test_get_dict_value_by_pos_given_valid_path_when_called_then_returns_value():
    test_dict = {"BackendConfig": {"ModelDeployConfig": {"ModelConfig": [{"npuMemSize": 1024}]}}}
    result = base_analyze.get_dict_value_by_pos(test_dict, "BackendConfig:ModelDeployConfig:ModelConfig:0:npuMemSize")
    assert result == 1024


def test_get_dict_value_by_pos_given_invalid_path_when_called_then_returns_none():
    test_dict = {"key1": {"key2": "value"}}
    assert base_analyze.get_dict_value_by_pos(test_dict, "key1:invalid") is None
    assert base_analyze.get_dict_value_by_pos(test_dict, "invalid:path") is None


def test_get_dict_value_by_pos_given_empty_dict_when_called_then_returns_none():
    assert base_analyze.get_dict_value_by_pos({}, "any:path") is None


def test_get_dict_value_by_pos_given_list_index_when_called_then_returns_value():
    test_data = {"list": [{"item": "value"}]}
    assert base_analyze.get_dict_value_by_pos(test_data, "list:0:item") == "value"


# Test npu_mem_size_checker
def test_npu_mem_size_checker_given_valid_config_when_npu_mem_size_not_minus1_then_adds_answer():
    test_config = {"BackendConfig": {"ModelDeployConfig": {"ModelConfig": [{"npuMemSize": 1024}]}}}

    with patch.object(base_analyze.logger, "info") as mock_log:
        base_analyze.npu_mem_size_checker(
            test_config, {}, {}  # benchmark_instance  # mindie_server_log_path  # profiling_params
        )

        mock_log.assert_called_with("获取目前 numMemSize 的值为 1024, 并不是 -1")
        assert (
            "set to -1",
            "设置为-1，将由服务化自动根据剩余的显存数量，配置block数量，会尽量用满显存空间",
        ) in ANSWERS[SUGGESTION_TYPES.config]["npuMemSize"]


def test_npu_mem_size_checker_given_npu_mem_size_minus1_when_called_then_no_action():
    test_config = {"BackendConfig": {"ModelDeployConfig": {"ModelConfig": [{"npuMemSize": -1}]}}}

    base_analyze.npu_mem_size_checker(test_config, {}, {})

    assert "npuMemSize" not in ANSWERS[SUGGESTION_TYPES.config]


# Test check_prefill_latency
def test_check_prefill_latency_given_first_token_target_and_support_select_batch_true_then_recommends_disable():
    benchmark_data = {"results_per_request": {"1": {"latency": [10.5]}, "2": {"latency": [12.3]}}}
    test_config = {"BackendConfig": {"ScheduleConfig": {"supportSelectBatch": True}}}

    with patch.object(base_analyze.logger, "debug") as mock_debug:
        base_analyze.check_prefill_latency(
            test_config, benchmark_data, namedtuple("test", ["target"])(TARGETS.FirstTokenTime)
        )

        # Verify logging was called
        assert mock_debug.call_count >= 3

        # Verify answer was added
        assert ("set to False", "关闭 supportSelectBatch 可降低首 token 时延") in ANSWERS[SUGGESTION_TYPES.config][
            "support_select_batch"
        ]


def test_check_prefill_latency_given_throughput_target_and_support_select_batch_false_then_recommends_enable():
    benchmark_data = {"results_per_request": {"1": {"latency": [8.2]}, "2": {"latency": [9.1]}}}
    test_config = {"BackendConfig": {"ScheduleConfig": {"supportSelectBatch": False}}}

    base_analyze.check_prefill_latency(
        test_config, benchmark_data, namedtuple("test", ["target"])(TARGETS.Throughput)
    )

    assert ("set to True", "开启 supportSelectBatch 可降低首 Throughput 时延") in ANSWERS[SUGGESTION_TYPES.config][
        "support_select_batch"
    ]


def test_check_prefill_latency_given_empty_results_when_called_then_no_crash():
    benchmark_data = {"results_per_request": {}}
    test_config = {"BackendConfig": {"ScheduleConfig": {"supportSelectBatch": True}}}

    # Should not raise any exceptions
    base_analyze.check_prefill_latency(
        test_config, benchmark_data, namedtuple("test", ["target"])(TARGETS.FirstTokenTime)
    )


# Test histogram logging edge cases
def test_check_prefill_latency_with_single_latency_value():
    benchmark_data = {"results_per_request": {"1": {"latency": [5.5]}}}
    test_config = {"BackendConfig": {"ScheduleConfig": {"supportSelectBatch": True}}}

    with patch.object(base_analyze.logger, "debug") as mock_debug:
        base_analyze.check_prefill_latency(
            test_config, benchmark_data, namedtuple("test", ["target"])(TARGETS.FirstTokenTime)
        )

        # Should still log histogram info
        assert mock_debug.call_count >= 3


def test_check_prefill_latency_with_missing_latency_values():
    benchmark_data = {"results_per_request": {"1": {"other_metric": 1.0}, "2": {"latency": []}}}
    test_config = {"BackendConfig": {"ScheduleConfig": {"supportSelectBatch": True}}}

    # Should not crash
    base_analyze.check_prefill_latency(
        test_config, benchmark_data, namedtuple("test", ["target"])(TARGETS.FirstTokenTime)
    )
