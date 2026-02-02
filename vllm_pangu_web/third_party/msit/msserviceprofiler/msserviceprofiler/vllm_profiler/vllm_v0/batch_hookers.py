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
from collections import Counter
from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook


def compare_deques(queue1, queue2):
    counter1 = Counter(queue1)
    counter2 = Counter(queue2)
    diff = counter1 - counter2
    return diff


def queue_profiler(before_queue, after_queue, queue_name):
    # 队列元素减少
    less_queue = compare_deques(before_queue, after_queue)
    rid_list = []
    for seq_group in less_queue: # V1 note: SequenceGroup == Request
        rid_list.append(seq_group.request_id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list[:])
        prof.metric("QueueSize", len(after_queue)).metric_scope("QueueName", queue_name).event("Dequeue")

    # 队列元素增加
    add_queue = compare_deques(after_queue, before_queue)
    rid_list.clear()
    for seq_group in add_queue: # V1 note: SequenceGroup == Request
        rid_list.append(seq_group.request_id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list)
        prof.metric("QueueSize", len(after_queue)).metric_scope("QueueName", queue_name).event("Enqueue")


@vllm_hook(("vllm.core.scheduler", "Scheduler.schedule"), min_version="0.6.3")
def schedule(original_func, this, *args, **kwargs):
    from vllm.sequence import SequenceGroupMetadata

    prof = Profiler(Level.INFO).domain("BatchSchedule").span_start("batchFrameworkProcessing")
    seq_group_metadata_list, scheduler_outputs, allow_async_output_proc = original_func(this, *args, **kwargs)

    iter_map, is_prefill = {}, False
    for seq_group in this.running:
        if len(seq_group.seqs) == 0:
            continue
        prompt_len = len(seq_group.prompt_token_ids)
        generated_len = len(seq_group.seqs[0].get_token_ids()) - prompt_len
        iter_map[seq_group.request_id] = generated_len
        is_prefill = is_prefill or (generated_len == 0)
    prof.attr("batch_type", "Prefill" if is_prefill else "Decode")

    rid_list = []
    for metadata in seq_group_metadata_list:
        if isinstance(metadata, SequenceGroupMetadata):
            iter_ = iter_map.get(metadata.request_id, metadata.token_chunk_size)
            data = {"rid": metadata.request_id, "iter": iter_}
            rid_list.append(data)
    prof.res(rid_list)
    prof.span_end()

    return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc


@vllm_hook(("vllm.core.scheduler", "Scheduler.abort_seq_group"), min_version="0.6.3")
def abort_seq_group(original_func, this, request_id, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("BatchSchedule").res(request_id)
    prof.metric_inc("FINISHED_ABORTED", 1).event("ReqState")
    original_func(this, request_id, *args, **kwargs)


@vllm_hook(("vllm.core.scheduler", "Scheduler._allocate_and_set_running"), min_version="0.6.3")
def allocate_and_set_running(original_func, this, seq_group, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
    prof.metric_inc("RUNNING", 1).metric_inc("WAITING", -1).event("ReqState")
    original_func(this, seq_group, *args, **kwargs)


@vllm_hook(("vllm.core.scheduler", "Scheduler._preempt_by_recompute"), min_version="0.6.3")
def preempt_by_recompute(original_func, this, seq_group, *args, **kwargs):
    from vllm.sequence import SequenceStatus

    seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    if len(seqs) != 1:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
        prof.metric_inc("RUNNING", -1).metric_inc("WAITING", 1).event("ReqState")
    original_func(this, seq_group, *args, **kwargs)


@vllm_hook(("vllm.core.scheduler", "Scheduler._swap_in"), min_version="0.6.3")
def swap_in(original_func, this, seq_group, *args, **kwargs):
    prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
    prof.metric_inc("RUNNING", 1).metric_inc("SWAPPED", -1).event("ReqState")
    original_func(this, seq_group, *args, **kwargs)


@vllm_hook(("vllm.core.scheduler", "Scheduler._swap_out"), min_version="0.6.3")
def swap_out(original_func, this, seq_group, *args, **kwargs):
    if this.block_manager.can_swap_out(seq_group):
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
        prof.metric_inc("RUNNING", -1).metric_inc("SWAPPED", 1).event("ReqState")
    original_func(this, seq_group, *args, **kwargs)


@vllm_hook(("vllm.core.scheduler", "Scheduler.free_finished_seq_groups"), min_version="0.6.3")
def free_finished_seq_groups(original_func, this, *args, **kwargs):
    before_running_queue = this.running.copy()
    for seq_group in this.running:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
        prof.metric_inc("RUNNING", -1).metric_inc("FINISHED", 1).event("ReqState")
    original_func(this, *args, **kwargs)
    queue_profiler(before_running_queue, this.running, "running")


@vllm_hook(("vllm.core.scheduler", "Scheduler._add_seq_group_to_running"), min_version="0.6.3")
def add_seq_group_to_running(original_func, this, seq_group, *args, **kwargs):
    original_func(this, seq_group, *args, **kwargs)
    prof = Profiler(Level.INFO).domain("BatchSchedule").res([seq_group.request_id])
    prof.metric("QueueSize", len(this.running)).metric_scope("QueueName", "running").event("Enqueue")


@vllm_hook(("vllm.core.scheduler", "Scheduler._schedule_priority_preemption"), min_version="0.6.3")
def schedule_priority_preemption(original_func, this, budget, *args, **kwargs):
    before_waiting_queue = this.waiting.copy()
    before_running_queue = this.running.copy()
    force_preemption_count = original_func(this, budget, *args, **kwargs)
    queue_profiler(before_waiting_queue, this.waiting, "waiting")
    queue_profiler(before_running_queue, this.running, "running")
    return force_preemption_count


@vllm_hook(("vllm.core.scheduler", "Scheduler._schedule_default"), min_version="0.6.3")
def schedule_default(original_func, this, *args, **kwargs):
    before_swapped_queue = this.swapped.copy()
    before_running_queue = this.running.copy()
    before_waiting_queue = this.waiting.copy()
    scheduler_outputs = original_func(this, *args, **kwargs)
    queue_profiler(before_swapped_queue, this.swapped, "swapped")
    queue_profiler(before_running_queue, this.running, "running")
    queue_profiler(before_waiting_queue, this.waiting, "waiting")
    return scheduler_outputs


@vllm_hook(("vllm.core.scheduler", "Scheduler._schedule_chunked_prefill"), min_version="0.6.3")
def schedule_chunked_prefill(original_func, this, *args, **kwargs):
    before_running_queue = this.running.copy()
    before_waiting_queue = this.waiting.copy()
    before_swapped_queue = this.swapped.copy()
    scheduler_outputs = original_func(this, *args, **kwargs)
    queue_profiler(before_running_queue, this.running, "running")
    queue_profiler(before_waiting_queue, this.waiting, "waiting")
    queue_profiler(before_swapped_queue, this.swapped, "swapped")
    return scheduler_outputs


@vllm_hook(("vllm.core.scheduler", "Scheduler.add_seq_group"), min_version="0.6.3")
def add_seq_group(original_func, this, seq_group, *args, **kwargs):
    original_func(this, seq_group, *args, **kwargs)
    prof = Profiler(Level.INFO).domain("BatchSchedule").res([seq_group.request_id])
    prof.metric("QueueSize", len(this.waiting)).metric_scope("QueueName", "waiting").event("Enqueue")


@vllm_hook(("vllm.core.scheduler", "Scheduler._add_seq_group_to_swapped"), min_version="0.6.3")
def add_seq_group_to_swapped(original_func, this, seq_group, *args, **kwargs):
    original_func(this, seq_group, *args, **kwargs)
    prof = Profiler(Level.INFO).domain("BatchSchedule").res([seq_group.request_id])
    prof.metric("QueueSize", len(this.swapped)).metric_scope("QueueName", "swapped").event("Enqueue")


@vllm_hook(("vllm.engine.llm_engine", "LLMEngine._add_processed_request"), min_version="0.6.3")
def add_processed_request(original_func, this, request_id, *args, **kwargs):
    original_func(this, request_id, *args, **kwargs)
    Profiler(Level.INFO).domain("Request").res(request_id).metric_inc("WAITING", 1).event("ReqState")
