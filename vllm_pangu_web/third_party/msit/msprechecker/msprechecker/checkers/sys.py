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

import subprocess
from .base import NodeChecker
from ..utils import Traverser


class SysChecker(NodeChecker):
    def __init__(self, *, error_handler=None, rule_manager=None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "system"

    def _get_rules(self):
        if self.rule_manager.framework == "vllm" or self.rule_manager.scene == "vllm":
            self.rule_manager.scene == "default"
            return self.rule_manager.get_rules().get("sys")
        self.rule_manager.scene = "default"
        return self.rule_manager.get_rules().get("sys")

    def _check(self, results) -> None:
        # 先执行原有的节点检查
        visited_nodes = Traverser.traverse(results)
        rules = self._get_rules()
        self._validate_nodes(rules, visited_nodes)
        
        # 在vllm框架下额外检查jemalloc安装
        if self.rule_manager.framework == "vllm":
            self._check_jemalloc_installation()

    def _check_jemalloc_installation(self):
        """检查jemalloc是否通过包管理器安装"""
        jemalloc_installed = self._is_jemalloc_installed()
        
        if not jemalloc_installed:
            self.error_handler.add_error(
                path="jemalloc",
                actual="not installed",
                expected="installed",
                reason="jemalloc未通过apt/yum安装，建议安装以提高性能: "
                       "Ubuntu/Debian: sudo apt-get install libjemalloc-dev, "
                       "CentOS/RHEL: sudo yum install jemalloc",
                severity="low"
            )

    def _is_jemalloc_installed(self) -> bool:
        """检查jemalloc是否通过包管理器安装"""
        # 先尝试Ubuntu/Debian的apt
        try:
            result_apt = subprocess.run(
                ['/usr/bin/apt', 'list', '--installed', 'libjemalloc*'],
                capture_output=True,
                text=True,
                check=False
            )
            if result_apt.returncode == 0 and 'libjemalloc' in result_apt.stdout:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # 再尝试CentOS/RHEL的yum
        try:
            result_yum = subprocess.run(
                ['/usr/bin/yum', 'list', 'installed', 'jemalloc*'],
                capture_output=True,
                text=True,
                check=False
            )
            if result_yum.returncode == 0 and 'jemalloc' in result_yum.stdout:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return False