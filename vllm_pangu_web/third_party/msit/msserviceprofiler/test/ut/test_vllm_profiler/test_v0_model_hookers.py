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
import threading
from collections import namedtuple
from unittest.mock import MagicMock
import pytest

from msserviceprofiler.vllm_profiler.vllm_v0 import model_hookers

from .fake_ms_service_profiler import Profiler, Level


class FakeSeqData:
    def __init__(self, length, prompt_len):
        self._len = length
        self.prompt_token_ids = [0] * prompt_len

    def get_len(self):
        return self._len


class FakeSeqMetadata:
    def __init__(self, rid, is_prompt, seq_data):
        self.request_id = rid
        self.is_prompt = is_prompt
        self.seq_data = seq_data


class FakeExecuteModelReq:
    def __init__(self, seq_group_metadata_list):
        self.seq_group_metadata_list = seq_group_metadata_list


class FakeAttnMeta:
    def __init__(self, prefill_metadata):
        self.prefill_metadata = prefill_metadata


class FakeModelInput:
    def __init__(self, prefill_metadata, req_map, shape0):
        self.attn_metadata = FakeAttnMeta(prefill_metadata)
        self.request_ids_to_seq_ids = req_map
        self.input_tokens = MagicMock()
        self.input_tokens.shape = (shape0, 10)


# Reset thread local state between tests
@pytest.fixture(autouse=True)
def reset_state():
    if hasattr(model_hookers._thread_local, "hook_state"):
        delattr(model_hookers._thread_local, "hook_state")
    Profiler.reset()
    yield
    if hasattr(model_hookers._thread_local, "hook_state"):
        delattr(model_hookers._thread_local, "hook_state")
    Profiler.reset()



@pytest.mark.parametrize("seq_data_empty", [False, True])
@pytest.mark.parametrize("is_prompt_values", [[True, False], [False, False]])
def test_handle_execute_model_given_metadata_when_various_prompts_then_correct_batch_type(
    seq_data_empty, is_prompt_values
):
    # Arrange
    seq_list = []
    for idx, is_prompt in enumerate(is_prompt_values):
        if seq_data_empty:
            seq_data = {}
        else:
            seq_data = {0: FakeSeqData(length=5, prompt_len=3)}
        seq_list.append(FakeSeqMetadata(rid=f"rid{idx}", is_prompt=is_prompt, seq_data=seq_data))
    req = FakeExecuteModelReq(seq_list)
    original_func = MagicMock(return_value="ok")

    # Act
    result = model_hookers.handle_execute_model(original_func, "this", req)

    # Assert
    assert result == "ok"
    # Validate profiler calls
    calls_flat = sum(Profiler.instance_calls, [])
    # Ensure we got both rid entries
    assert any(c[0] == "res" for c in calls_flat)
    # Batch_type matches expectation
    expected_batch_type = "Prefill" if any(is_prompt_values) else "Decode"
    assert any(c == ("attr", "batch_type", expected_batch_type) for c in calls_flat)


def test_execute_model_given_first_run_when_called_then_skip_profiling():
    original_func = MagicMock(return_value="first_ok")
    model_input = FakeModelInput(True, {"id1": 1}, 2)

    # First call should skip profiling
    result = model_hookers.execute_model(original_func, "this", model_input, "kv")
    assert result == "first_ok"
    # No profiler recorded
    assert not Profiler.instance_calls


@pytest.mark.parametrize("prefill", [True, False])
def test_execute_model_given_second_run_when_prefill_varies_then_batch_type_correct(prefill):
    # Prepare state to skip first run branch
    state = model_hookers._get_state()
    state.execute_model_first_run = False
    original_func = MagicMock(return_value="ok")
    model_input = FakeModelInput(prefill, {"id1": 1, "id2": 2}, 3)

    result = model_hookers.execute_model(original_func, "this", model_input, "kv")
    assert result == "ok"
    calls_flat = sum(Profiler.instance_calls, [])
    expected_type = "Prefill" if prefill else "Decode"
    assert any(c == ("attr", "batch_type", expected_type) for c in calls_flat)
    assert any(c == ("attr", "batch_size", 3) for c in calls_flat)


def test_begin_forward_given_first_run_when_called_then_skip_append():
    original_func = MagicMock(return_value="res")
    model_input = FakeModelInput(True, {"id1": 1}, 1)

    result = model_hookers.begin_forward(original_func, "this", model_input)
    assert result == "res"
    state = model_hookers._get_state()
    assert state.forward_profiler == []


def test_begin_forward_given_second_run_when_called_then_append_profiler():
    state = model_hookers._get_state()
    state.begin_forward_first_run = False
    original_func = MagicMock(return_value="res")
    model_input = FakeModelInput(True, {"id1": 1, "id2": 2}, 1)

    result = model_hookers.begin_forward(original_func, "this", model_input)
    assert result == "res"
    state = model_hookers._get_state()
    assert len(state.forward_profiler) == 1
    calls_flat = sum(Profiler.instance_calls, [])
    assert any(c[0] == "res" for c in calls_flat)


def test_set_forward_context_given_profiler_exists_when_enter_then_start_and_end_called():
    state = model_hookers._get_state()
    dummy_prof = Profiler(Level.INFO)
    state.forward_profiler.append(dummy_prof)

    def orig_ctx():
        @contextlib.contextmanager
        def cm():
            yield

        return cm()

    import contextlib

    with model_hookers.set_forward_context(orig_ctx):
        pass
    calls_flat = sum(Profiler.instance_calls, [])
    assert ("span_start", "forward") in calls_flat
    assert "span_end" in calls_flat


def test_set_forward_context_given_no_profiler_when_enter_then_no_start_or_end():
    def orig_ctx():
        @contextlib.contextmanager
        def cm():
            yield

        return cm()

    import contextlib

    with model_hookers.set_forward_context(orig_ctx):
        pass
    calls_flat = sum(Profiler.instance_calls, [])
    assert ("span_start", "forward") not in calls_flat
    assert "span_end" not in calls_flat
