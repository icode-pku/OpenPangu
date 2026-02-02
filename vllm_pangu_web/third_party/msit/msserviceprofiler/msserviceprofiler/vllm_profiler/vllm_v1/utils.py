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

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Generator
import threading


def _iter_cached_req_id_and_num_comp(cached) -> Generator[Tuple[str, int], None, None]:
    """迭代器，用于迭代 CachedRequestData 中的请求 ID 和计算的 token 数量。
    
    Args:
        cached: CachedRequestData 对象
        
    Yields:
        Tuple[str, int]: (请求ID, 计算的token数量)
    """
    if cached is None:
        return
    req_ids = getattr(cached, "req_ids", None)
    if req_ids is not None:
        for rid, num_comp in zip(cached.req_ids, cached.num_computed_tokens):
            yield rid, num_comp
        return
    try:
        for item in cached:
            rid = getattr(item, "req_id", None)
            num_comp = getattr(item, "num_computed_tokens", None)
            if rid is not None and num_comp is not None:
                yield rid, num_comp
    except TypeError:
        return


def _iter_new_req_id_and_num_comp(new_reqs) -> Generator[Tuple[str, int], None, None]:
    """迭代器，用于迭代 list[NewRequestData] 中的请求 ID 和计算的 token 数量。
    
    Args:
        new_reqs: NewRequestData 列表
        
    Yields:
        Tuple[str, int]: (请求ID, 计算的token数量)
    """
    for item in new_reqs or []:
        yield item.req_id, item.num_computed_tokens


def classify_requests(state: Any, scheduler_output: Any) -> Tuple[List[Dict], List[Dict], str]:
    """统一的分类与构造逻辑。
    
    Args:
        state: 状态对象
        scheduler_output: 调度器输出
        
    Returns:
        Tuple[List[Dict], List[Dict], str]: 
            - request_id_list: [{"rid": str}, ...]
            - request_id_with_iter_list: [{"rid": str, "iter": int, "type": 0|1}, ...]
            - batch_type: "Prefill" | "Decode" | "Prefill,Decode"
    """

    # 记录新请求的 prompt 长度
    for new_req in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        state.request_id_to_prompt_token_len[new_req.req_id] = len(new_req.prompt_token_ids)

    # 构建 num_computed_tokens 映射
    num_comp_by_rid: Dict[str, int] = {}
    for rid, num_comp in (
        list(_iter_cached_req_id_and_num_comp(getattr(scheduler_output, "scheduled_cached_reqs", None)))
        + list(_iter_new_req_id_and_num_comp(getattr(scheduler_output, "scheduled_new_reqs", None)))
    ):
        num_comp_by_rid[rid] = num_comp

    request_id_list: List[Dict] = []
    request_id_with_iter_list: List[Dict] = []
    has_prefill, has_decode = False, False

    for request_id, num in scheduler_output.num_scheduled_tokens.items():
        request_id_list.append({"rid": request_id})
        iter_ = state.request_id_to_iter.get(request_id, -1) + 1
        state.request_id_to_iter[request_id] = iter_

        prompt_len = state.request_id_to_prompt_token_len.get(request_id)
        num_comp = num_comp_by_rid.get(request_id)
        is_prefill_request = False
        if prompt_len is not None and num_comp is not None:
            is_prefill_request = (num_comp < prompt_len)
        if is_prefill_request:
            has_prefill = True
        else:
            has_decode = True

        request_id_with_iter_list.append({
            "rid": request_id,
            "iter": iter_,
            "type": 0 if is_prefill_request else 1,
            "num_scheduled_tokens": num,
            "num_prompt_tokens": prompt_len,
            "num_computed_tokens": num_comp
        })

        if request_id in scheduler_output.finished_req_ids:
            state.request_id_to_prompt_token_len.pop(request_id, None)
            state.request_id_to_iter.pop(request_id, None)

    if has_prefill and has_decode:
        batch_type = "Prefill,Decode"
    elif has_prefill:
        batch_type = "Prefill"
    else:
        batch_type = "Decode"

    return request_id_list, request_id_with_iter_list, batch_type


class SharedHookState:
    """共享的 hook 状态类。
    
    该类用于在多个 hook 之间共享状态信息，包括：
    - 请求ID到prompt token长度的映射
    - 请求ID到迭代次数的映射
    
    Attributes:
        request_id_to_prompt_token_len (Dict[str, int]): 请求ID到prompt token长度的映射
        request_id_to_iter (Dict[str, int]): 请求ID到迭代次数的映射
    """
    
    def __init__(self):
        """初始化 SharedHookState。"""
        self.request_id_to_prompt_token_len = {}
        self.request_id_to_iter = {}


def create_state_getter(state_class):
    """为每个调用方创建独立的线程本地 state getter。
    
    该函数创建一个线程本地的状态获取器，确保每个线程都有独立的状态实例。
    
    Args:
        state_class: 状态类
        
    Returns:
        Callable: 状态获取函数
        
    Example:
        class MyState(SharedHookState):
            ...
        _get_state = create_state_getter(MyState)
    """
    _thread_local = threading.local()

    def _get_state():
        if not hasattr(_thread_local, "hook_state"):
            _thread_local.hook_state = state_class()
        return _thread_local.hook_state

    return _get_state
