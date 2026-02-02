# -*- coding: utf-8 -*-
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

from .base import BaseCollector


class EnvCollector(BaseCollector):
    ENV_FILTERS = [
        "ASCEND", "MINDIE", "ATB_", "HCCL_", "MIES", 
        "RANKTABLE", "GE_", "TORCH", "ACL_", "NPU_",
        "LCCL_", "LCAL_", "OPS", "INF_"
    ]

    def __init__(self, error_handler=None, *, filter_env: bool = False):
        super().__init__(error_handler)
        self.error_handler.type = "env"
        self.filter_env = filter_env

    def _collect_data(self):
        try:
            env_items = os.environ.items()
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function="_collect_data",
                lineno=35,
                what="使用 'os.environ' 获取环境信息失败",
                reason=str(e)
            )
            env_items = {}

        if self.filter_env:
            return {
                k: v for k, v in env_items
                if any(item in k for item in self.ENV_FILTERS)
            }
        return dict(env_items)
