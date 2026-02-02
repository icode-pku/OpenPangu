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
import platform

import yaml
from msguard.security import open_s

from ..utils import (
    NpuType, ErrorSeverity,
    get_npu_count, get_npu_type,
    global_logger
)


class RuleManager:
    """规则管理器，负责加载内置规则和用户自定义规则并进行合并"""
    
    SCENE_MAPPING = {
        "user_config": "config_check_dsr1_pd.yaml",
        "mindie_env": "env_check_dsr1_pd.yaml",
        "pd_disaggregation": "config_check_pd.yaml",
        "pd_disaggregation_single_container": "config_check_pd_single_container.yaml",
        "pd_mix": "pd_mix_check.yaml",
        "pd_mix_dsr1": "pd_mix_check_dsr1.yaml",
        "ep": "ep_default.yaml",
        "default": "default.yaml",
    }
    
    ARCH_MAPPING = {
        "x86_64": "x86",
        "aarch64": "arm"
    }


    FRAMEWORK_MAPPING = {
        "vllm": "vllm",
    }
    
    def __init__(self, *, scene=None, framework=None, custom_rule_path=None):
        self.scene = scene
        self.framework = framework
        self.custom_rule_path = custom_rule_path

        self._npu_type, _ = get_npu_type()
        if not self._npu_type:
            self._npu_type = NpuType.TP_A2

        self._npu_count = get_npu_count()
        self._arch = platform.machine().lower()

    @staticmethod
    def create_rule(type_, value, reason="", severity=ErrorSeverity.ERR_HIGH):
        return {
            "expected":
                {
                    "type": type_,
                    "value": value
                },
            "reason": reason,
            "severity": severity
        }

    def get_rules(self):
        # 1. 获取内置规则
        rules = self._get_builtin_rules()
        
        # 2. 获取用户自定义规则（如果有）
        custom_rules = self._get_custom_rules()
        
        # 3. 更新 env 规则
        if 'env' in custom_rules and 'env' in rules:
            custom_env_rules = custom_rules['env']
            rules['env'].update(custom_env_rules)

        return rules

    def _get_builtin_rules(self):
        """获取指定场景的内置规则"""
        if not self.scene:
            return {}

        rule_file = self.SCENE_MAPPING[self.scene]
        cur_dir = os.path.dirname(__file__)

        # 处理framework的情况
        if self.framework == "vllm":
            if self.framework not in self.FRAMEWORK_MAPPING:
                raise ValueError(
                f"Expected 'scene' to be {', '.join(self.FRAMEWORK_MAPPING)}. Got {self.framework} instead."
            )
            # 获取框架目录
            framework_dir = self.FRAMEWORK_MAPPING.get(self.framework)

            # 构建新路径 <preset><framework><scene>
            rule_path = os.path.join(cur_dir, framework_dir, rule_file)
            with open_s(rule_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        if self.scene not in self.SCENE_MAPPING:
            raise ValueError(
                f"Expected 'scene' to be {', '.join(self.SCENE_MAPPING)}. Got {self.scene} instead."
            )

        # 特殊处理default场景
        if self.scene == "default":
            rule_path = os.path.join(cur_dir, rule_file)
        else:
            # 根据NPU类型和架构获取正确的规则路径
            if self._npu_type == NpuType.TP_A2:
                arch_dir = self.ARCH_MAPPING.get(self._arch)
                if not arch_dir:
                    global_logger.warning(
                        "Unsupported architecture: %s. Using '%s' as a fall back",
                        self._arch,
                        NpuType.TP_A2.display
                    )
                    arch_dir = "arm"   
                rule_path = os.path.join(cur_dir, "A2", arch_dir, rule_file)

                if self._arch == "x86_64" and self._npu_count != 16:
                    global_logger.warning(
                        "Unsupported type: %s x86_64 but %s chips (expected 16 chips). Use '%s' as a fall back",
                        NpuType.TP_A2.display,
                        self._npu_count,
                        NpuType.TP_A3.display
                    )
                    rule_path = os.path.join(cur_dir, "A3", rule_file)
            else:  # A3 or default to A3
                rule_path = os.path.join(cur_dir, "A3", rule_file)
        
        with open_s(rule_path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f)
            except Exception as e:
                raise ValueError("Invalid rule yaml file: %r" % rule_path) from e
    
    def _get_custom_rules(self):
        """获取用户自定义规则，如果没有则返回空字典"""
        if not self.custom_rule_path:
            return {}

        try:
            with open_s(self.custom_rule_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            global_logger.warning("'--custom-config-path' passed an insecure path, skipped\n%s", e)
            return {}
