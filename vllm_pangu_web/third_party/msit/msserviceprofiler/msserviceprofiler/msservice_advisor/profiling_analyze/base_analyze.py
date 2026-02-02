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

import numpy as np
from msserviceprofiler.msservice_advisor.profiling_analyze.register import register_analyze, cached, answer
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import TARGETS, SUGGESTION_TYPES, logger
from msserviceprofiler.msservice_advisor.profiling_analyze.utils import get_dict_value_by_pos


@register_analyze()
def npu_mem_size_checker(mindie_service_config, benchmark_instance, profiling_params):
    npu_mem_size_pos = "BackendConfig:ModelDeployConfig:ModelConfig:0:npuMemSize"
    npu_mem_size = get_dict_value_by_pos(mindie_service_config, npu_mem_size_pos)
    if npu_mem_size is not None and npu_mem_size != -1:
        logger.info(f"获取目前 numMemSize 的值为 {npu_mem_size}, 并不是 -1")
        answer(
            suggesion_type=SUGGESTION_TYPES.config,
            suggesion_item="npuMemSize",
            action="set to -1",
            reason="设置为-1，将由服务化自动根据剩余的显存数量，配置block数量，会尽量用满显存空间",
        )


@register_analyze()
def check_prefill_latency(mindie_service_config, benchmark_instance, profiling_params):
    target = profiling_params.target
    if benchmark_instance:
        results_per_request = benchmark_instance.get("results_per_request", {}).values()
        prefill_latencies = np.array([ii["latency"][0] for ii in results_per_request if len(ii.get("latency", [])) > 0])

        if len(prefill_latencies) > 0:
            counts, buckets = np.histogram(prefill_latencies)
            bucket_keys = ["{:.2f}-{:.2f}".format(ii, jj) for ii, jj in zip(buckets[:-1], buckets[1:])]
            bucket_keys_max_len = len(max(bucket_keys, key=len))
            logger.debug("First token latency:")
            logger.debug(" " * (4 + bucket_keys_max_len - 14) + "Bucket [0, max]: Count")
            logger.debug(" " * 4 + "-" * bucket_keys_max_len + ": ------")
            for bucket_key, count in zip(bucket_keys, counts):
                logger.debug(" " * (4 + bucket_keys_max_len - len(bucket_key)) + "{}: {}".format(bucket_key, count))

    if mindie_service_config:
        support_select_batch = get_dict_value_by_pos(
            mindie_service_config, "BackendConfig:ScheduleConfig:supportSelectBatch"
        )
        logger.info(f"Got support_select_batch: {support_select_batch}")
        if target == TARGETS.FirstTokenTime and support_select_batch:
            answer(
                suggesion_type=SUGGESTION_TYPES.config,
                suggesion_item="support_select_batch",
                action="set to False",
                reason="关闭 supportSelectBatch 可降低首 token 时延",
            )
        elif target == TARGETS.Throughput and not support_select_batch:
            answer(
                suggesion_type=SUGGESTION_TYPES.config,
                suggesion_item="support_select_batch",
                action="set to True",
                reason="开启 supportSelectBatch 可降低首 Throughput 时延",
            )