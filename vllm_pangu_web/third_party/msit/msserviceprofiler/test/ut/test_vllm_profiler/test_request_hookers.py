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
from collections import namedtuple
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from msserviceprofiler.vllm_profiler.vllm_v1 import request_hookers

from .fake_ms_service_profiler import Profiler, Level


@pytest.mark.asyncio
async def test_add_request_async_given_request_id_and_prompt_when_called_then_events_logged_and_original_called():
    # Setup
    original_func = AsyncMock(return_value="original_result")
    mock_this = MagicMock()
    request_id = "test_req_123"
    prompt = "Hello world"

    # Execute
    result = await request_hookers.add_request_async(original_func, mock_this, request_id, prompt)

    # Verify
    assert result == "original_result"
    original_func.assert_awaited_once_with(mock_this, request_id, prompt)

    # Check profiler calls
    assert len(Profiler.instance_calls) == 2

    http_req_calls = Profiler.instance_calls[0]
    assert http_req_calls == [("domain", "Engine"), ("res", request_id), ("event", "httpReq")]

    encode_calls = Profiler.instance_calls[1]
    assert encode_calls == [("domain", "Engine"), ("res", request_id), ("event", "tokenize")]


def create_mock_output(request_id, finished=True, has_stats=True):
    mock_output = MagicMock()
    mock_output.request_id = request_id
    mock_output.finish_reason = "stop" if finished else None

    mock_state = MagicMock()
    mock_state.prompt_token_ids = [1, 2, 3]  # recv_token_size=3

    if has_stats:
        mock_stats = MagicMock()
        mock_stats.num_generation_tokens = 5  # reply_token_size=5
        mock_state.stats = mock_stats
    else:
        mock_state.stats = None

    return mock_output, mock_state


def test_process_outputs_given_empty_outputs_when_called_then_no_events_and_original_called():
    # Setup
    original_func = MagicMock(return_value="empty_result")
    mock_this = MagicMock()

    # Execute
    result = request_hookers.process_outputs(original_func, mock_this, [])

    # Verify
    assert result == "empty_result"
    original_func.assert_called_once_with(mock_this, [])
    assert len(Profiler.instance_calls) == 0


def test_process_outputs_given_finished_request_when_called_then_httpres_logged_with_metrics():
    # Setup
    original_func = MagicMock(return_value="result")
    mock_output, mock_state = create_mock_output("req_finished")
    mock_this = MagicMock(request_states={"req_finished": mock_state})

    # Execute
    result = request_hookers.process_outputs(original_func, mock_this, [mock_output])

    # Verify
    assert result == "result"
    assert len(Profiler.instance_calls) == 2

    # httpRes calls
    http_res_calls = Profiler.instance_calls[0]
    assert http_res_calls == [
        ("domain", "Engine"),
        ("res", "req_finished"),
        ("metric", "recvTokenSize", 3),
        ("metric", "replyTokenSize", 5),
        ("event", "httpRes"),
    ]

    # detokenize calls
    decode_calls = Profiler.instance_calls[1]
    assert decode_calls == [("domain", "Engine"), ("res", ["req_finished"]), ("event", "detokenize")]


def test_process_outputs_given_unfinished_request_when_httpres_not_logged():
    # Setup
    mock_output, _ = create_mock_output("req_unfinished", finished=False)
    mock_this = MagicMock(request_states={})
    original_func = MagicMock()

    # Execute
    request_hookers.process_outputs(original_func, mock_this, [mock_output])

    # Verify - Only detokenize should be logged
    assert len(Profiler.instance_calls) == 1
    decode_calls = Profiler.instance_calls[0]
    assert decode_calls[-1] == ("event", "detokenize")


def test_process_outputs_given_missing_request_state_when_httpres_not_logged():
    # Setup
    mock_output, _ = create_mock_output("req_missing_state")
    mock_this = MagicMock(request_states={})  # No state for request
    original_func = MagicMock()

    # Execute
    request_hookers.process_outputs(original_func, mock_this, [mock_output])

    # Verify
    assert len(Profiler.instance_calls) == 1
    assert Profiler.instance_calls[0][-1] == ("event", "detokenize")


def test_process_outputs_given_missing_stats_when_none_reply_tokens():
    # Setup
    mock_output, mock_state = create_mock_output("req_no_stats", has_stats=False)
    mock_this = MagicMock(request_states={"req_no_stats": mock_state})
    original_func = MagicMock()

    # Execute
    request_hookers.process_outputs(original_func, mock_this, [mock_output])

    # Verify
    http_res_calls = Profiler.instance_calls[0]
    metrics = [call[1:] for call in http_res_calls if call[0] == "metric"]
    assert metrics == [("recvTokenSize", 3), ("replyTokenSize", None)]  # Missing stats


def test_process_outputs_given_multiple_requests_then_events_logged():
    # Setup
    finished1, state1 = create_mock_output("req_finished1")
    finished2, state2 = create_mock_output("req_finished2")
    unfinished, _ = create_mock_output("req_unfinished", finished=False)

    mock_this = MagicMock(request_states={"req_finished1": state1, "req_finished2": state2})
    outputs = [finished1, unfinished, finished2]
    original_func = MagicMock()

    # Execute
    request_hookers.process_outputs(original_func, mock_this, outputs)

    # Verify
    assert len(Profiler.instance_calls) == 3  # 2 httpRes + 1 detokenize

    # First httpRes
    assert Profiler.instance_calls[0][1][1] == "req_finished1"  # res ID
    assert Profiler.instance_calls[0][-1] == ("event", "httpRes")

    # Second httpRes
    assert Profiler.instance_calls[1][1][1] == "req_finished2"
    assert Profiler.instance_calls[1][-1] == ("event", "httpRes")

    # detokenize with all request IDs
    decode_end_res = Profiler.instance_calls[2][1][1]
    assert set(decode_end_res) == {"req_finished1", "req_unfinished", "req_finished2"}


def test_process_outputs_given_arguments_then_correctly():
    # Setup
    mock_output, mock_state = create_mock_output("req_args_test")
    mock_this = MagicMock(request_states={"req_args_test": mock_state})
    original_func = MagicMock(return_value="test_result")
    extra_args = ("arg1", "arg2")
    extra_kwargs = {"key": "value"}

    # Execute
    result = request_hookers.process_outputs(original_func, mock_this, [mock_output], *extra_args, **extra_kwargs)

    # Verify
    assert result == "test_result"
    original_func.assert_called_once_with(mock_this, [mock_output], *extra_args, **extra_kwargs)
