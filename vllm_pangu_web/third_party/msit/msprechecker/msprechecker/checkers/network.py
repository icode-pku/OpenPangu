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

from .base import BaseChecker


class PingChecker(BaseChecker):
    def _check(self, results):
        success_pattern = "3 received, 0% packet loss"

        for host, ping_result in results.items():
            if success_pattern not in ping_result:
                self.error_handler.add_error(
                    path=f'当前机器 -x-> {host}', expected=success_pattern, actual=ping_result,
                    reason=f'当前机器 ping 主机 {host} 失败，请检查 rank table',
                    severity="high"
                )
