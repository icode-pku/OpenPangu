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
from .base import NodeChecker
from ..utils import get_model_type
from ..utils import Traverser


class EnvChecker(NodeChecker):
    def __init__(self, *, error_handler=None, rule_manager=None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "env"

    def _get_rules(self):
        if self.rule_manager.framework == "vllm":
            return self.rule_manager.get_rules()["env"]
        
        model_type = get_model_type()
        if not model_type or "deepseek" not in model_type:
            self.rule_manager.scene = "pd_mix"
        else:
            self.rule_manager.scene = "pd_mix_dsr1"
        return self.rule_manager.get_rules()['env']

    def _check(self, results) -> None:
        super()._check(results)
        
        # 在vllm框架下额外检查LD_PRELOAD环境变量
        if self.rule_manager.framework == "vllm":
            self._check_ld_preload_for_jemalloc()

    def _check_ld_preload_for_jemalloc(self):
        """检查LD_PRELOAD环境变量是否包含jemalloc库"""
        ld_preload = os.environ.get('LD_PRELOAD', '')
        
        if not ld_preload:
            self.error_handler.add_error(
                path="LD_PRELOAD",
                actual="<missing>",
                expected="",
                reason="LD_PRELOAD环境变量未设置，建议设置jemalloc库以提高内存分配性能",
                severity="low"
            )
            return
        
        # 分割LD_PRELOAD中的库路径
        libraries = [lib.strip() for lib in ld_preload.split(':') if lib.strip()]
        
        # 检查是否包含jemalloc相关的库
        jemalloc_found = False
        for lib in libraries:
            if 'jemalloc' in lib or 'libjemalloc' in lib:
                jemalloc_found = True
                break
        
        if not jemalloc_found:
            self.error_handler.add_error(
                path="LD_PRELOAD",
                actual=ld_preload,
                expected=ld_preload,
                reason="LD_PRELOAD环境变量未包含jemalloc库，建议添加jemalloc库以提高内存分配性能",
                severity="low"
            )
