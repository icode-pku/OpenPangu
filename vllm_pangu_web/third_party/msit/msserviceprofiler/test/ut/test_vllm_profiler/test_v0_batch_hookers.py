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
from collections import deque, Counter
from unittest.mock import MagicMock
import pytest

from msserviceprofiler.vllm_profiler.vllm_v0 import batch_hookers

from .fake_ms_service_profiler import Profiler, Level


def test_compare_deques_given_nonempty_when_diff_then_correct_counter():
    q1 = deque([1, 2, 2, 3])
    q2 = deque([2, 3])
    result = batch_hookers.compare_deques(q1, q2)
    assert isinstance(result, Counter)
    assert result == Counter({1: 1, 2: 1})


def test_compare_deques_given_equal_when_no_diff_then_empty_counter():
    q1 = deque([1, 2])
    q2 = deque([1, 2])
    result = batch_hookers.compare_deques(q1, q2)
    assert result == Counter()


class DummySeqGroup:
    def __init__(self, request_id):
        self.request_id = request_id


def test_queue_profiler_given_changes_when_elements_removed_and_added_then_logs_both():
    before = [DummySeqGroup("A"), DummySeqGroup("B")]
    after = [DummySeqGroup("B"), DummySeqGroup("C")]
    batch_hookers.queue_profiler(before, after, "test_queue")
    # Should log Dequeue for A and Enqueue for C
    all_calls = sum(Profiler.instance_calls, [])
    assert any([c[0] == "event" and c[1] == "Dequeue" for c in all_calls])
    assert any([c[0] == "event" and c[1] == "Enqueue" for c in all_calls])


class DummySeq:
    def __init__(self, token_ids):
        self._ids = token_ids

    def get_token_ids(self):
        return self._ids


class DummyRunning:
    def __init__(self, rid, prompt_len, gen_len):
        self.request_id = rid
        self.prompt_token_ids = [0] * prompt_len
        self.seqs = [DummySeq([0] * prompt_len + [1] * gen_len)] if gen_len >= 0 else []


class DummyMeta:
    def __init__(self, rid, chunk_size):
        self.request_id = rid
        self.token_chunk_size = chunk_size


def make_this_running(running):
    return MagicMock(running=running)


@pytest.mark.parametrize(
    "func_name,args",
    [
        ("abort_seq_group", ("req1",)),
        ("allocate_and_set_running", (MagicMock(request_id="r1"),)),
        ("swap_in", (MagicMock(request_id="r1"),)),
        ("add_seq_group_to_running", (MagicMock(request_id="r2"),)),
        ("add_seq_group", (MagicMock(request_id="r3"),)),
        ("add_seq_group_to_swapped", (MagicMock(request_id="r4"),)),
        ("add_processed_request", ("req5",)),
    ],
)
def test_simple_hooks_log_and_call_original(func_name, args):
    called = {}

    def orig_func(*a, **k):
        called["yes"] = True

    this = MagicMock(running=[MagicMock(request_id="rx")], waiting=[1], swapped=[1])
    func = getattr(batch_hookers, func_name)
    func(orig_func, this, *args)
    assert called.get("yes", False) is True
    assert Profiler.instance_calls  # some logging occurred


def test_swap_out_given_can_swap_out_true_then_logs():
    seq_group = MagicMock(request_id="s1")
    bm = MagicMock(can_swap_out=lambda sg: True)
    this = MagicMock(block_manager=bm)
    called = {}

    def orig_func(*a, **k):
        called["yes"] = True

    batch_hookers.swap_out(orig_func, this, seq_group)
    assert called["yes"] is True
    assert any([c[0] == "metric_inc" for c in sum(Profiler.instance_calls, [])])


def test_swap_out_given_can_swap_out_false_then_no_logs():
    seq_group = MagicMock(request_id="s2")
    bm = MagicMock(can_swap_out=lambda sg: False)
    this = MagicMock(block_manager=bm)
    batch_hookers.swap_out(lambda *a, **k: None, this, seq_group)
    assert not any([c[0] == "metric_inc" for c in sum(Profiler.instance_calls, [])])
