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


@vllm_hook(
    hook_points=[
        ("vllm.engine.async_llm_engine", "AsyncLLMEngine.add_request"),
        ("vllm.v1.engine.async_llm", "AsyncLLM.add_request")
    ],
    min_version="0.9.1"
)    
async def add_request_async(original_func, this, request_id, prompt, *args, **kwargs):
    Profiler(Level.INFO).domain("Engine").res(request_id).event("httpReq")
    Profiler(Level.INFO).domain("Engine").res(request_id).event("tokenize")
    return await original_func(this, request_id, prompt, *args, **kwargs)


@vllm_hook(("vllm.v1.engine.output_processor", "OutputProcessor.process_outputs"), min_version="0.9.1")
def process_outputs(original_func, this, engine_core_outputs, *args, **kwargs):
    if len(engine_core_outputs) == 0:
        return original_func(this, engine_core_outputs, *args, **kwargs)

    request_id_list = []
    for engine_core_output in engine_core_outputs:
        request_id = engine_core_output.request_id
        request_state = this.request_states.get(request_id)
        request_id_list.append(request_id)
        if request_state and engine_core_output.finish_reason is not None:
            recv_token_size = len(request_state.prompt_token_ids)
            reply_token_size = (request_state.stats.num_generation_tokens if request_state.stats else None)
            
            profiler = Profiler(Level.INFO).domain("Engine").res(request_id)
            profiler = profiler.\
                metric("recvTokenSize", recv_token_size).\
                metric("replyTokenSize", reply_token_size)
            profiler.event("httpRes")

    ret = original_func(this, engine_core_outputs, *args, **kwargs)
    prof = Profiler(Level.INFO).domain("Engine").res(request_id_list)
    prof.event("detokenize")
    return ret
