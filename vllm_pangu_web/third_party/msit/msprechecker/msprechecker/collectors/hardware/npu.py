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

from .base import BaseStressCollector
from ...utils import SimpleProgressBar


class NPUStressCollector(BaseStressCollector):
    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.error_handler.type = "npu stress"

        try:
            import torch_npu
        except ImportError as e:
            self.error_handler.add_error(
                filename=__file__,
                function='__init__',
                lineno=25,
                what="当前环境没有安装 torch_npu",
                reason=str(e)
            )
            self.torch_npu = None
        else:
            self.torch_npu = torch_npu

    def _collect_data(self):
        if not self.torch_npu:
            return []

        device_ids = self.torch_npu.npu.device_count()
        output = {device_id: 0 for device_id in range(device_ids)}

        for device_id in SimpleProgressBar(range(device_ids)):
            # 创建事件对象，用于记录运算的开始和结束时间
            start_event = self.torch_npu.npu.Event(enable_timing=True)
            end_event = self.torch_npu.npu.Event(enable_timing=True)

            start_event.record()
            self._matmul_stress_test('npu', device_id)
            end_event.record()

            # 同步当前流，确保全部运算均已完成
            self.torch_npu.npu.current_stream().synchronize()
            npu_time = start_event.elapsed_time(end_event)
            output[device_id] = npu_time

        return output

    def _get_free_memory(self, device):
        if self.torch.npu.is_available():
            total_memory = self.torch.npu.get_device_properties(device).total_memory
            used_memory = self.torch.npu.memory_allocated(device)
            return total_memory - used_memory
        else:
            self.error_handler.add_error(
                filename=__file__,
                function='_get_free_memory',
                lineno=62,
                what=f"尝试获取 device '{device}' 上剩余的内存失败",
                reason="'self.torch.npu.is_available()' 返回了 False"
            )
            return 0
