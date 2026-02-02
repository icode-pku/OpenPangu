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
from ..logger import logger


@vllm_hook(("vllm.v1.core.kv_cache_manager", "KVCacheManager.free"), min_version="0.9.1")
def free(original_func, this, request, *args, **kwargs):
    ret = original_func(this, request, *args, **kwargs)
    num_blocks = this.block_pool.get_num_free_blocks()
    usage_percent = this.usage
    prof = Profiler(Level.INFO).domain("Schedule.KVCache").res(request.request_id)
    prof.metric("deviceBlock", num_blocks) \
        .metric("UsagePercent", usage_percent) \
        .event("Free")
    return ret


@vllm_hook(("vllm.v1.core.kv_cache_manager", "KVCacheManager.get_computed_blocks"), min_version="0.9.1")
def get_computed_blocks(original_func, this, request, *args, **kwargs):
    ret = original_func(this, request, *args, **kwargs)
    if len(ret) > 1 and request.num_tokens > 0:
        num_new_computed_tokens = ret[1]
        cur_hit_rate = num_new_computed_tokens / request.num_tokens
        prof = Profiler(Level.INFO).domain("Schedule.KVCache").res(request.request_id)
        prof.attr("hitCache", cur_hit_rate).event("CacheHitRate")
    return ret
