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

import re
import json

from msguard.security import open_s

from .base import BaseCollector


class ConfigCollector(BaseCollector):
    def __init__(self, error_handler=None, *, config_path: str = None):
        super().__init__(error_handler)
        self.config_path = config_path

    @staticmethod
    def _build_context_hierarchy(lines, depth_changes):
        """构建每行的上下文层级关系"""
        line_count = len(lines)
        context_hierarchy = [[] for _ in range(line_count)]

        active_blocks = []
        for line_num in range(line_count):
            if line_num in depth_changes:
                current_depth = depth_changes[line_num]
                
                if not active_blocks or current_depth > depth_changes[active_blocks[-1]]:
                    active_blocks.append(line_num)
                elif current_depth < depth_changes[active_blocks[-1]]:
                    active_blocks.pop()
                
                context_hierarchy[line_num] = active_blocks.copy()

        # 填充没有直接上下文的行
        previous_context = []
        for i, context in enumerate(context_hierarchy):
            if not context:
                context_hierarchy[i] = previous_context
            else:
                previous_context = context

        return context_hierarchy

    def _collect_data(self):
        lines = []
        depth_changes = {}
        key_location_map = {}

        current_depth = 0
        parent_key = ''
        last_key = ''

        key_pattern = re.compile('\s*"([^"]+)"\s*:\s*')
        with open_s(self.config_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                line = line.rstrip('\n')
                lines.append(line)

                if not line.strip():
                    continue

                mo = key_pattern.search(line)
                if mo:
                    key = mo.group(1)
                    key = f'{parent_key}.{key}' if parent_key else key
                    col_start = line.find('"')
                    key_location_map[key] = (line_no, col_start)
                    last_key = key
                
                previous_depth = current_depth
                for char in line:
                    if char in '{[':
                        current_depth += 1
                    elif char in '}]':
                        current_depth -= 1
                
                if current_depth > previous_depth:
                    parent_key = last_key
                    depth_changes[line_no] = current_depth
                elif current_depth < previous_depth:
                    dot_num = parent_key.count('.')
                    parent_key = '' if dot_num < 2 else parent_key.rsplit('.', 2)[0]
                    depth_changes[line_no] = current_depth
            
            return (
                json.loads('\n'.join(lines)), lines,
                key_location_map, self._build_context_hierarchy(lines, depth_changes)
            )


class UserConfigCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "user config"


class MindIEEnvCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "mindie env"


class ModelConfigCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "model config"


class MIESConfigCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "mies config"
