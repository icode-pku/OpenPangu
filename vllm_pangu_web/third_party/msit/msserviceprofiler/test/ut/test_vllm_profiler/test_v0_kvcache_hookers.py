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
from unittest.mock import MagicMock
import pytest

from msserviceprofiler.vllm_profiler.vllm_v0 import kvcache_hookers

from .fake_ms_service_profiler import Profiler, Level


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset GLOBAL_REQUEST_DICT and Profiler calls before each test."""
    kvcache_hookers.GLOBAL_REQUEST_DICT.clear()
    yield
    kvcache_hookers.GLOBAL_REQUEST_DICT.clear()


class DummyBlockTable:
    def __init__(self, blocks=None):
        self._blocks = blocks or []


class DummyThis:
    def __init__(self, block_tables):
        self.block_tables = block_tables


class DummySeq:
    def __init__(self, seq_id):
        self.seq_id = seq_id


class DummySeqGroup:
    def __init__(self, request_id, seqs):
        self.request_id = request_id
        self.seqs = seqs


class DummyStats:
    def __init__(self, cpu_cache_usage_sys, gpu_cache_usage_sys):
        self.cpu_cache_usage_sys = cpu_cache_usage_sys
        self.gpu_cache_usage_sys = gpu_cache_usage_sys


class DummyScheduler:
    def __init__(self, free_blocks):
        self.block_manager = MagicMock()
        self.block_manager.get_num_free_gpu_blocks.return_value = free_blocks


def test_allocate_given_seqgroup_when_called_then_updates_global_and_profiles():
    seq_group = DummySeqGroup("req1", [DummySeq(1)])
    this = DummyThis(block_tables={})
    orig_func = MagicMock()
    kvcache_hookers.allocate(orig_func, this, seq_group)
    assert "req1" in kvcache_hookers.GLOBAL_REQUEST_DICT
    assert kvcache_hookers.GLOBAL_REQUEST_DICT["req1"] == seq_group.seqs
    assert any("Allocate" in call for call in sum(Profiler.instance_calls, []))
    orig_func.assert_called_once()


def test_allocate_given_empty_seqs_when_called_then_still_records():
    seq_group = DummySeqGroup("req2", [])
    this = DummyThis(block_tables={"a": 1})
    orig_func = MagicMock()
    kvcache_hookers.allocate(orig_func, this, seq_group)
    assert kvcache_hookers.GLOBAL_REQUEST_DICT["req2"] == []
    assert any("Allocate" in call for call in sum(Profiler.instance_calls, []))


@pytest.mark.parametrize("seq_in_dict", [True, False])
def test_append_slots_given_seq_presence_variants_then_correct_request_id(seq_in_dict):
    seq = DummySeq(seq_id=123)
    request_id = "reqX"
    if seq_in_dict:
        kvcache_hookers.GLOBAL_REQUEST_DICT[request_id] = [seq]
    this = DummyThis(block_tables={123: DummyBlockTable(blocks=[1, 2])})
    orig_func = MagicMock(return_value="new_cows")
    result = kvcache_hookers.append_slots(orig_func, this, seq, 5)
    assert result == "new_cows"
    calls_flat = sum(Profiler.instance_calls, [])
    assert any("AppendSlot" in call for call in calls_flat)
    assert any("blocks" in call for call in calls_flat)
    orig_func.assert_called_once()


def test_append_slots_given_missing_blocks_attr_then_defaults_to_empty():
    seq = DummySeq(seq_id=999)
    kvcache_hookers.GLOBAL_REQUEST_DICT["reqZ"] = [seq]

    class NoBlocks:
        pass

    this = DummyThis(block_tables={999: NoBlocks()})
    orig_func = MagicMock(return_value="nc")
    res = kvcache_hookers.append_slots(orig_func, this, seq, 1)
    assert res == "nc"


@pytest.mark.parametrize(
    "func,expected_attr", [(kvcache_hookers.swap_in, "swap_in"), (kvcache_hookers.swap_out, "swap_out")]
)
def test_swap_in_out_given_seqgroup_then_profiles(func, expected_attr):
    seq_group = DummySeqGroup("reqY", [DummySeq(1)])
    this = DummyThis(block_tables={"a": 1})
    orig_func = MagicMock(return_value="res")
    result = func(orig_func, this, seq_group)
    assert result == "res"
    calls_flat = sum(Profiler.instance_calls, [])
    assert any(expected_attr in call for call in calls_flat)
    orig_func.assert_called_once()


@pytest.mark.parametrize("seq_in_dict", [True, False])
def test_free_given_seq_presence_variants_then_correct_request_id(seq_in_dict):
    seq = DummySeq(seq_id=77)
    if seq_in_dict:
        kvcache_hookers.GLOBAL_REQUEST_DICT["req77"] = [seq]
    this = DummyThis(block_tables={})
    orig_func = MagicMock()
    kvcache_hookers.free(orig_func, this, seq)
    calls_flat = sum(Profiler.instance_calls, [])
    assert any("Free" in call for call in calls_flat)
    orig_func.assert_called_once()


def test_get_stats_given_schedulers_then_profiles_and_returns_stats():
    this = MagicMock()
    this.scheduler = [DummyScheduler(3), DummyScheduler(4)]
    stats = DummyStats(cpu_cache_usage_sys=0.5, gpu_cache_usage_sys=0.8)
    orig_func = MagicMock(return_value=stats)
    result = kvcache_hookers.get_stats(orig_func, this)
    assert result is stats
    calls_flat = sum(Profiler.instance_calls, [])
    assert any("GetCacheHitRate" in call for call in calls_flat)
    assert any(("attr", "cpuHitCache", 0.5) in calls_flat for _ in [0])
    orig_func.assert_called_once()
