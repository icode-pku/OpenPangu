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
import types
import importlib
from unittest.mock import patch

import pytest


# 会话级 fixture，只对当前目录的测试应用 mock
@pytest.fixture(autouse=True, scope="session")
def mock_profiler_module():
    # 保存原始模块状态
    original_module = sys.modules.get("ms_service_profiler")
    
    # 创建 mock 模块
    from .fake_ms_service_profiler import Profiler, Level
    mock_module = types.ModuleType("ms_service_profiler")
    mock_module.Profiler = Profiler
    mock_module.Level = Level
    
    # 应用 mock
    sys.modules["ms_service_profiler"] = mock_module
    
    # 重新加载 batch_hookers 模块确保使用 mock
    modules_to_reload = [
        "msserviceprofiler.vllm_profiler.vllm_v0.batch_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v1.batch_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v0.model_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v1.model_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v0.kvcache_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v1.kvcache_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v0.request_hookers",
        "msserviceprofiler.vllm_profiler.vllm_v1.request_hookers",
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    
    yield
    
    # 恢复原始模块
    if original_module:
        sys.modules["ms_service_profiler"] = original_module
    else:
        del sys.modules["ms_service_profiler"]
    
    # 重新加载模块以恢复原始状态
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])


# 在每个测试函数前重置 Profiler
@pytest.fixture(autouse=True)
def reset_profiler():
    from .fake_ms_service_profiler import Profiler
    Profiler.reset()
    yield
    Profiler.reset()


@pytest.fixture(autouse=True)
def patch_model_hookers_synchronize():
    with patch("msserviceprofiler.vllm_profiler.vllm_v1.model_hookers.synchronize"):
        yield
