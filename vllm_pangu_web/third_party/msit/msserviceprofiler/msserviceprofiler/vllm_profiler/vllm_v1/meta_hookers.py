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

from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook
from .utils import SharedHookState, create_state_getter


class MetaCollectionState(SharedHookState):
    def __init__(self):
        super().__init__()
        self.has_collected = False  # 添加状态标记属性


# 线程本地存储获取器（每文件独立线程状态）
_get_state = create_state_getter(MetaCollectionState)


@vllm_hook(("vllm.v1.engine.core", "DPEngineCoreProc.add_request"), min_version="0.9.1")
def init_data_parallel(original_func, this, vllm_config, *args, **kwargs):
    ret = original_func(this, vllm_config, *args, **kwargs)

    state = _get_state()

    # 若此进程还没采集过dpRankId,则添加meta数据
    if not state.has_collected:
        Profiler(Level.INFO).add_meta_info("dpRankId", this.dp_rank)
        state.has_collected = True  # 更新状态类中的属性

    return ret
