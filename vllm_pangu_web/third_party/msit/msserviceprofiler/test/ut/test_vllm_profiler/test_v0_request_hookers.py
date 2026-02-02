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
import types
from unittest.mock import MagicMock
import pytest

from msserviceprofiler.vllm_profiler.vllm_v0 import request_hookers

from .fake_ms_service_profiler import Profiler, Level


def test_prof_add_request_given_valid_input_when_called_then_logs_events():
    request_hookers.prof_add_request("req123", "prompt text")
    assert len(Profiler.instance_calls) == 2
    # First call chain
    assert ("domain", "Request") in Profiler.instance_calls[0]
    assert ("res", "req123") in Profiler.instance_calls[0]
    assert ("event", "httpReq") in Profiler.instance_calls[0]
    # Second call chain
    assert ("event", "tokenize") in Profiler.instance_calls[1]


@pytest.mark.parametrize("func_name", ["add_request_063", "add_request_084"])
def test_add_request_sync_given_valid_input_when_called_then_profiler_and_original_called(func_name):
    called = {}

    def original_func(this, request_id, prompt, *a, **kw):
        called["called"] = (this, request_id, prompt, a, kw)
        return "ok"

    this = object()
    func = getattr(request_hookers, func_name)
    result = func(original_func, this, "req1", "promptX", 1, key="v")
    assert result == "ok"
    # Profiler should be called first
    assert any(("event", "httpReq") in chain for chain in Profiler.instance_calls)
    # Original function called with same args
    assert called["called"][1] == "req1"
    assert called["called"][2] == "promptX"


def make_seq_group(finished=True, prompt_len=3, output_len=4):
    sg = MagicMock()
    sg.is_finished.return_value = finished
    seq = MagicMock()
    seq.get_prompt_len.return_value = prompt_len
    seq.get_output_len.return_value = output_len
    sg.seqs = [seq]
    return sg


def make_ctx(output_queue_data):
    ctx = types.SimpleNamespace()
    ctx.output_queue = output_queue_data
    return ctx


def test_process_model_outputs_given_empty_queue_when_called_then_returns_original_result():
    def original_func(this, ctx, request_id=None, *a, **kw):
        return "empty-ok"

    ctx = make_ctx([])
    result = request_hookers.process_model_outputs(original_func, object(), ctx)
    assert result == "empty-ok"
    assert not Profiler.instance_calls


def test_process_model_outputs_given_non_empty_queue_and_finished_seq_when_called_then_logs_and_returns():
    def original_func(this, ctx, request_id=None, *a, **kw):
        return "res-ok"

    seq_group = make_seq_group(finished=True, prompt_len=10, output_len=5)
    meta = types.SimpleNamespace(request_id="reqX")
    scheduler_outputs = types.SimpleNamespace(scheduled_seq_groups=[types.SimpleNamespace(seq_group=seq_group)])
    skip = set()
    ctx = make_ctx([(None, [meta], scheduler_outputs, None, None, None, skip)])
    result = request_hookers.process_model_outputs(original_func, object(), ctx)
    assert result == "res-ok"
    flat_calls = [
        item for chain in Profiler.instance_calls for item in chain
    ]
    assert ("metric", "recvTokenSize", 10) in flat_calls
    assert ("metric", "replyTokenSize", 5) in flat_calls
    assert ("event", "detokenize") in flat_calls


def test_process_model_outputs_given_skip_index_when_called_then_skips_token_metrics():
    def original_func(this, ctx, request_id=None, *a, **kw):
        return "res-skip"

    seq_group = make_seq_group(finished=True)
    meta = types.SimpleNamespace(request_id="reqY")
    scheduler_outputs = types.SimpleNamespace(scheduled_seq_groups=[types.SimpleNamespace(seq_group=seq_group)])
    skip = {0}
    ctx = make_ctx([(None, [meta], scheduler_outputs, None, None, None, skip)])
    result = request_hookers.process_model_outputs(original_func, object(), ctx)
    assert result == "res-skip"
    # Should have no metric logs but still detokenize
    flat_calls = [
        item for chain in Profiler.instance_calls for item in chain
    ]
    assert ("metric", "recvTokenSize", 3) not in flat_calls
    assert ("event", "detokenize") in flat_calls


def test_validate_output_given_finished_true_when_called_then_logs_and_returns():
    def original_func(this, output, output_type):
        return "val-ok"

    output = types.SimpleNamespace(
        finished=True,
        request_id="reqVal",
        prompt_token_ids=[1, 2, 3],
        outputs=[types.SimpleNamespace(token_ids=[4, 5])],
    )
    result = request_hookers.validate_output(original_func, object(), output, "ot")
    assert result == "val-ok"
    flat_calls = [
        item for chain in Profiler.instance_calls for item in chain
    ]
    assert ("metric", "recvTokenSize", 3) in flat_calls
    assert ("metric", "replyTokenSize", 2) in flat_calls


def test_validate_output_given_finished_false_when_called_then_no_metrics_logged():
    def original_func(this, output, output_type):
        return "no-metrics"

    output = types.SimpleNamespace(
        finished=False, request_id="reqVal2", prompt_token_ids=[1], outputs=[types.SimpleNamespace(token_ids=[2])]
    )
    result = request_hookers.validate_output(original_func, object(), output, "ot")
    assert result == "no-metrics"
    flat_calls = [
        item for chain in Profiler.instance_calls for item in chain
    ]
    assert ("metric", "recvTokenSize", 1) not in flat_calls
    assert ("metric", "replyTokenSize", 1) not in flat_calls
