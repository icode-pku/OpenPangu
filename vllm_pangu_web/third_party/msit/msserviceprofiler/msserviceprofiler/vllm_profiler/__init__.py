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

from .logger import set_log_level
from .service_profiler import ServiceProfiler

set_log_level("info")  # Default is info, put here for user changes


_service_profiler = ServiceProfiler()


def register_service_profiler():
    """初始化服务分析器（向后兼容接口）"""
    _service_profiler.initialize()
