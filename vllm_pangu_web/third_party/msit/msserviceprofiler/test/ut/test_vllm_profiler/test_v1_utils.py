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


import threading
import time
from typing import List

import pytest

from msserviceprofiler.vllm_profiler.vllm_v1.utils import (
    _iter_cached_req_id_and_num_comp,
    _iter_new_req_id_and_num_comp,
    classify_requests,
    SharedHookState,
    create_state_getter,
)


class _DummyNewReq:
    def __init__(self, req_id: str, prompt_token_ids: List[int], num_computed_tokens: int):
        self.req_id = req_id
        self.prompt_token_ids = prompt_token_ids
        self.num_computed_tokens = num_computed_tokens


class _DummyCachedItem:
    def __init__(self, req_id: str, num_computed_tokens: int):
        self.req_id = req_id
        self.num_computed_tokens = num_computed_tokens


class _DummySchedOutput:
    def __init__(
        self,
        scheduled_new_reqs=None,
        scheduled_cached_reqs=None,
        num_scheduled_tokens=None,
        finished_req_ids=None,
    ):
        self.scheduled_new_reqs = scheduled_new_reqs or []
        self.scheduled_cached_reqs = scheduled_cached_reqs
        self.num_scheduled_tokens = num_scheduled_tokens or {}
        self.finished_req_ids = finished_req_ids or set()


def test_iter_cached_req_id_and_num_comp_various_inputs():
    # None -> yields nothing
    assert not list(_iter_cached_req_id_and_num_comp(None))

    # Object with req_ids and num_computed_tokens
    class _DummyCachedObj:
        def __init__(self):
            self.req_ids = ["r1", "r2"]
            self.num_computed_tokens = [3, 5]

    out = list(_iter_cached_req_id_and_num_comp(_DummyCachedObj()))
    assert out == [("r1", 3), ("r2", 5)]

    # Iterable of items with attributes
    cached_items = [_DummyCachedItem("r3", 7), _DummyCachedItem("r4", 9)]
    out = list(_iter_cached_req_id_and_num_comp(cached_items))
    assert out == [("r3", 7), ("r4", 9)]

    # Non-iterable without req_ids/num_computed_tokens -> yields nothing
    class _NotIterable:
        pass

    assert not list(_iter_cached_req_id_and_num_comp(_NotIterable()))


def test_iter_new_req_id_and_num_comp():
    assert not list(_iter_new_req_id_and_num_comp(None))
    new_reqs = [_DummyNewReq("a", [1, 2], 1), _DummyNewReq("b", [3], 0)]
    out = list(_iter_new_req_id_and_num_comp(new_reqs))
    assert out == [("a", 1), ("b", 0)]


def test_classify_requests_prefill_only_and_cleanup():
    state = SharedHookState()

    # r1: prompt 5, num_comp 3 -> prefill
    new_reqs = [_DummyNewReq("r1", [1, 2, 3, 4, 5], 3)]
    num_scheduled_tokens = {"r1": 5}
    sched = _DummySchedOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=None,
        num_scheduled_tokens=num_scheduled_tokens,
        finished_req_ids={"r1"},
    )

    request_id_list, request_id_with_iter_list, batch_type = classify_requests(state, sched)

    assert request_id_list == [{"rid": "r1"}]
    assert request_id_with_iter_list == [{"rid": "r1", "iter": 0, "type": 0,
            "num_scheduled_tokens": 5, "num_prompt_tokens": 5, "num_computed_tokens": 3}]
    assert batch_type == "Prefill"

    # r1 finished -> state cleanup
    assert "r1" not in state.request_id_to_prompt_token_len
    assert "r1" not in state.request_id_to_iter


def test_classify_requests_decode_only_and_iter_increment():
    state = SharedHookState()

    # r2: prompt 3, num_comp 3 -> decode (not prefill)
    new_reqs = [_DummyNewReq("r2", [1, 2, 3], 3)]
    num_scheduled_tokens = {"r2": 3}
    sched = _DummySchedOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=None,
        num_scheduled_tokens=num_scheduled_tokens,
        finished_req_ids=set(),
    )

    _, req_with_iter, batch_type = classify_requests(state, sched)
    assert req_with_iter == [{"rid": "r2", "iter": 0, "type": 1,
            "num_scheduled_tokens": 3, "num_prompt_tokens": 3, "num_computed_tokens": 3}]
    assert batch_type == "Decode"

    # Second call with same r2 should increment iter to 1
    sched2 = _DummySchedOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[_DummyCachedItem("r2", 4)],
        num_scheduled_tokens={"r2": 1},
        finished_req_ids=set(),
    )
    _, req_with_iter2, _ = classify_requests(state, sched2)
    assert req_with_iter2 == [{"rid": "r2", "iter": 1, "type": 1,
            "num_scheduled_tokens": 1, "num_prompt_tokens": 3, "num_computed_tokens": 4}]


def test_classify_requests_mixed_prefill_and_decode():
    state = SharedHookState()

    # rA prefill: prompt 4, num_comp 2
    # rB decode: prompt 2, num_comp 2
    new_reqs = [
        _DummyNewReq("rA", [1, 2, 3, 4], 2),
        _DummyNewReq("rB", [7, 8], 2),
    ]
    num_scheduled_tokens = {"rA": 3, "rB": 1}
    sched = _DummySchedOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=None,
        num_scheduled_tokens=num_scheduled_tokens,
        finished_req_ids=set(),
    )

    request_id_list, req_with_iter, batch_type = classify_requests(state, sched)

    assert sorted(request_id_list, key=lambda x: x["rid"]) == [{"rid": "rA"}, {"rid": "rB"}]
    # order aligns with dict iteration; avoid strict order by sorting for assertion
    sorted_req_with_iter = sorted(req_with_iter, key=lambda x: x["rid"])
    assert sorted_req_with_iter == [
        {"rid": "rA", "iter": 0, "type": 0,
            "num_scheduled_tokens": 3, "num_prompt_tokens": 4, "num_computed_tokens": 2},
        {"rid": "rB", "iter": 0, "type": 1,
            "num_scheduled_tokens": 1, "num_prompt_tokens": 2, "num_computed_tokens": 2},
    ]
    assert batch_type == "Prefill,Decode"


def test_create_state_getter_thread_locality():
    class _MyState(SharedHookState):
        def __init__(self):
            super().__init__()
            self.marker = object()

    get_state = create_state_getter(_MyState)

    s_main_1 = get_state()
    s_main_2 = get_state()
    assert s_main_1 is s_main_2  # same thread -> same instance

    results = []

    def _worker():
        s = get_state()
        results.append(s)

    th = threading.Thread(target=_worker)
    th.start()
    th.join(timeout=2)

    assert len(results) == 1
    assert results[0] is not s_main_1  # different thread -> different instance
