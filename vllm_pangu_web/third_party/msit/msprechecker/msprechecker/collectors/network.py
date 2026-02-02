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

import shlex
import shutil
import subprocess

from msguard import Rule

from .base import BaseCollector


class PingCollector(BaseCollector):
    def __init__(self, error_handler=None, *, rank_table=None):
        super().__init__(error_handler)
        self.rank_table = rank_table

        ping_cmd = "/usr/bin/ping"
        if not Rule.input_file_exec.is_satisfied_by(ping_cmd):
            self._ping_cmd = None
        else:
            self._ping_cmd = ping_cmd + " -c 3 -q -W 2 {}"

    def _collect_data(self):
        result = {}

        if not self._ping_cmd:
            self.error_handler.add_error(
                filename=__file__, function='_collect_data',
                lineno=36, what="当前环境没有 'ping' 命令或者权限不符合要求",
                reason="需要使用 'ping' 命令来进行多机的连通性检测"
            )
            return result

        host_to_devices = self.rank_table.host_to_devices
        if not host_to_devices:
            self.error_handler.add_error(
                filename=__file__, function='_collect_data',
                lineno=47, what="传入的 'rank table' 没有解析出任何信息",
                reason="请检查 'rank table' 是否符合格式规范"
            )
            return result

        for host in host_to_devices:
            try:
                output = subprocess.check_output(
                    shlex.split(self._ping_cmd.format(host)),
                    stderr=subprocess.STDOUT, text=True, timeout=5
                )
            except Exception:
                output = "ping failed"
            result[host] = output

        return result
