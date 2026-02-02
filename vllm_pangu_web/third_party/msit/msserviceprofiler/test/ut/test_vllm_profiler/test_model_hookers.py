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

import sys
import os
import threading
import contextlib
from collections import namedtuple
from unittest.mock import patch, MagicMock, call
import pytest

from msserviceprofiler.vllm_profiler.vllm_v1 import model_hookers
from msserviceprofiler.vllm_profiler.vllm_v1.utils import create_state_getter

from .fake_ms_service_profiler import Profiler, Level


# Reset profiler and state before each test
@pytest.fixture(autouse=True)
def reset_state():
    # 重置状态获取器，以清空内部的线程本地状态
    model_hookers._get_state = create_state_getter(model_hookers.HookState)
    Profiler.reset()
    yield


# Test helpers
SchedulerOutput = namedtuple(
    "SchedulerOutput",
    [
        "scheduled_new_reqs",
        "scheduled_cached_reqs",
        "num_scheduled_tokens",
        "finished_req_ids",
        "total_num_scheduled_tokens",
    ],
)
Request = namedtuple("Request", ["req_id", "prompt_token_ids", "num_computed_tokens"])


def create_request(request_id, token_count=10, computed_tokens=0):
    return Request(req_id=request_id, prompt_token_ids=[0] * token_count, num_computed_tokens=computed_tokens)


def test_get_state_given_first_call_when_no_existing_state_then_create_new_state():
    # 重新绑定获取器，确保是“第一次”获取
    model_hookers._get_state = create_state_getter(model_hookers.HookState)
    state = model_hookers._get_state()
    assert isinstance(state, model_hookers.HookState)
    # 再次获取应返回同一实例
    assert model_hookers._get_state() is state


def test_get_state_given_existing_state_when_called_then_return_same_instance():
    # 首次获取并保存
    state1 = model_hookers._get_state()
    # 再次获取应返回相同实例
    assert model_hookers._get_state() is state1


def test_compute_logits_given_valid_input_when_called_then_profile_span():
    mock_original = MagicMock(return_value="logits")
    mock_this = MagicMock()

    result = model_hookers.compute_logits(mock_original, mock_this, "input_ids", "scores")

    mock_original.assert_called_with(mock_this, "input_ids", "scores")
    assert result == "logits"
    assert len(Profiler.instance_calls) == 1
    calls = Profiler.instance_calls[0]
    assert ("span_start", "computing_logits") in calls
    assert "span_end" in calls


def test_sampler_forward_given_valid_input_when_called_then_profile_span():
    mock_original = MagicMock(return_value="samples")
    mock_this = MagicMock()

    result = model_hookers.sampler_forward(mock_original, mock_this, "input_ids")

    mock_original.assert_called_with(mock_this, "input_ids")
    assert result == "samples"
    assert len(Profiler.instance_calls) == 1
    calls = Profiler.instance_calls[0]
    assert ("span_start", "sample") in calls
    assert "span_end" in calls


def test_execute_model_given_new_requests_when_processing_then_update_state_and_profile():
    state = model_hookers.HookState()
    req1 = create_request("req1", token_count=5)
    req2 = create_request("req2", token_count=3)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[req1, req2],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req1": 5, "req2": 3},
        finished_req_ids=[],
        total_num_scheduled_tokens=8,
    )

    mock_original = MagicMock(return_value="output")

    with patch.object(model_hookers, "_get_state", return_value=state):
        result = model_hookers.execute_model(mock_original, MagicMock(), scheduler_output)

    assert result == "output"
    assert state.request_id_to_prompt_token_len == {"req1": 5, "req2": 3}

    # Verify profiling calls
    assert len(Profiler.instance_calls) == 2  # One for batch, one for forward

    # Check batch profiling
    batch_calls = Profiler.instance_calls[0]
    # 允许实现附带额外字段（如 type），仅校验 rid 与 iter
    res_entry = next(x for x in batch_calls if isinstance(x, tuple) and x[0] == "res")
    res_payload = res_entry[1]
    assert [{"rid": d["rid"], "iter": d["iter"]} for d in res_payload] == [
        {"rid": "req1", "iter": 0},
        {"rid": "req2", "iter": 0},
    ]
    assert ("attr", "batch_type", "Prefill") in batch_calls
    assert ("attr", "batch_size", 8) in batch_calls
    assert ("span_start", "modelExec") in batch_calls

    # Check forward profiler setup
    assert state.forward_profiler is not None


def test_execute_model_given_prefill_batch_when_processing_then_set_prefill_flag():
    state = model_hookers.HookState()
    state.request_id_to_prompt_token_len = {"req1": 5, "req2": 10}
    req1 = create_request("req1", token_count=10, computed_tokens=5)  # Partial processing
    req2 = create_request("req2", token_count=5, computed_tokens=5)  # Completed

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[req1, req2],
        num_scheduled_tokens={"req1": 5, "req2": 3},
        finished_req_ids=[],
        total_num_scheduled_tokens=8,
    )

    mock_original = MagicMock()

    with patch.object(model_hookers, "_get_state", return_value=state):
        model_hookers.execute_model(mock_original, MagicMock(), scheduler_output)

    # 包含一个 Prefill 与一个 Decode，当前实现返回 "Prefill,Decode"
    batch_calls = Profiler.instance_calls[0]
    assert ("attr", "batch_type", "Prefill,Decode") in batch_calls


def test_execute_model_given_decode_batch_when_processing_then_set_decode_flag():
    state = model_hookers.HookState()
    req1 = create_request("req1", token_count=10, computed_tokens=10)  # Completed
    req2 = create_request("req2", token_count=5, computed_tokens=5)  # Completed

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[req1, req2],
        num_scheduled_tokens={"req1": 5, "req2": 3},
        finished_req_ids=[],
        total_num_scheduled_tokens=8,
    )

    mock_original = MagicMock()

    with patch.object(model_hookers, "_get_state", return_value=state):
        model_hookers.execute_model(mock_original, MagicMock(), scheduler_output)

    # Should detect decode because all requests have computed tokens >= prompt length
    batch_calls = Profiler.instance_calls[0]
    assert ("attr", "batch_type", "Decode") in batch_calls


def test_execute_model_given_no_requests_when_processing_then_no_profiling():
    state = model_hookers.HookState()
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        finished_req_ids=[],
        total_num_scheduled_tokens=0,
    )

    mock_original = MagicMock(return_value="output")

    with patch.object(model_hookers, "_get_state", return_value=state):
        result = model_hookers.execute_model(mock_original, MagicMock(), scheduler_output)

    assert result == "output"
    assert len(Profiler.instance_calls) == 0
    assert state.forward_profiler is None


def test_set_forward_context_given_no_forward_profiler_when_used_then_create_new_profiler():
    state = model_hookers.HookState()
    mock_original = MagicMock()

    with patch.object(model_hookers, "_get_state", return_value=state):
        with model_hookers.set_forward_context(mock_original):
            pass

    assert len(Profiler.instance_calls) == 1
    calls = Profiler.instance_calls[0]
    assert ("span_start", "set_forward_context") in calls
    assert "span_end" in calls
    assert state.forward_profiler is None


def test_set_forward_context_given_existing_forward_profiler_when_used_then_reuse_profiler():
    state = model_hookers.HookState()
    state.forward_profiler = Profiler(Level.INFO)
    mock_original = MagicMock()

    with patch.object(model_hookers, "_get_state", return_value=state):
        with model_hookers.set_forward_context(mock_original):
            pass

    # Should use existing profiler instead of creating new one
    assert len(Profiler.instance_calls) > 0
    assert state.forward_profiler is None  # Should be cleared after use


def test_set_forward_context_given_context_manager_when_used_then_call_original():
    mock_original = MagicMock()
    mock_context = MagicMock()
    mock_original.return_value = mock_context

    with patch.object(model_hookers, "_get_state", return_value=model_hookers.HookState()):
        with model_hookers.set_forward_context(mock_original):
            pass

    mock_original.assert_called_once()
    mock_context.__enter__.assert_called_once()
    mock_context.__exit__.assert_called_once()


def test_hook_state_initialization():
    state = model_hookers.HookState()
    assert state.forward_profiler is None
    assert state.execute_model_first_run is True
    assert state.begin_forward_first_run is True
    assert not state.request_id_to_prompt_token_len
    assert not state.request_id_to_iter
