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
import time

import psutil

from .base import BaseStressCollector
from ...utils import SimpleProgressBar


class CPUStressCollector(BaseStressCollector):
    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.error_handler.type = "cpu stress"

    def _collect_data(self):
        if not self.torch:
            return []

        cpu_ids = os.cpu_count()
        output = {device_id: 0 for device_id in range(cpu_ids)}

        self.torch.set_num_threads(cpu_ids)

        for cpu_id in SimpleProgressBar(range(cpu_ids)):
            start_time = time.time()
            self._matmul_stress_test('cpu', cpu_id)
            end_time = time.time()
            cpu_time = (end_time - start_time) * 1000

            output[cpu_id] = cpu_time
        return output
    
    def _get_free_memory(self, device):
        return psutil.virtual_memory().available
