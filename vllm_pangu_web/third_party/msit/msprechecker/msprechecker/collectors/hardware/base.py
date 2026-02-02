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
import math
from abc import abstractmethod

import yaml
from msguard.security import open_s

from ..base import BaseCollector


class BaseStressCollector(BaseCollector):
    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        try:
            import torch
        except ImportError as e:
            self.error_handler.add_error(
                reason=str(e),
                filename=__file__,
                function='__init__',
                lineno=29,
                what="当前环境没有安装 torch",
            )
            self.torch = None
        else:
            self.torch = torch
    
    @staticmethod
    def _calculate_tensor_memory(shape):
        return math.prod(shape) * 4 # default to float32
    
    @staticmethod
    def _get_shape_by_type(check_type):
        env_check_dir = os.path.dirname(__file__)
        yaml_file = os.path.join(env_check_dir, "matmul_shape.yaml")
        with open_s(yaml_file, "r") as f:
            shape_dict = yaml.safe_load(f)

        batch_size = shape_dict[check_type]["batch_size"]
        seq_len = shape_dict[check_type]["seq_len"]
        hidden_size = shape_dict[check_type]["hidden_size"]
        intermediate_size = shape_dict[check_type]["intermediate_size"]

        return batch_size, seq_len, hidden_size, intermediate_size

    @abstractmethod
    def _get_free_memory(self, device):
        pass

    def _matmul_stress_test(self, device_type, device_id):
        check_type = f"{device_type}_check"
        device_pos = f"{device_type}:{device_id}"

        batch_size, seq_len, hidden_size, intermediate_size = self._get_shape_by_type(check_type)

        if not self._check_memory_for_matmul(
            batch_size, seq_len, hidden_size, intermediate_size, device_pos
        ):
            return

        # 执行多次矩阵运算：mat_c + mat_a × mat_b
        for _ in range(10):
            mat_a = self.torch.randn(batch_size, seq_len, hidden_size).to(device_pos)
            mat_b = self.torch.randn(batch_size, hidden_size, intermediate_size).to(device_pos)
            mat_c = self.torch.randn(seq_len, intermediate_size).to(device_pos)
            self.torch.addbmm(mat_c, mat_a, mat_b)

    def _check_memory_for_matmul(self, batch_size, seq_len, hidden_size, intermediate_size, device_pos):
        mat_a_mem = self._calculate_tensor_memory((batch_size, seq_len, hidden_size))
        mat_b_mem = self._calculate_tensor_memory((batch_size, hidden_size, intermediate_size))
        mat_c_mem = self._calculate_tensor_memory((seq_len, intermediate_size))
        total_required = mat_a_mem + mat_b_mem + mat_c_mem

        free_memory = self._get_free_memory(device_pos)
        safety_margin = 0.2

        available_with_margin = free_memory * (1 - safety_margin)
        has_enough_mem = total_required <= available_with_margin

        if not has_enough_mem:
            self.error_handler.add_error(
                reason=f"需要 {total_required / 1024 ** 2:.2f}MB, 可用 {free_memory / 1024 ** 2:.2f}MB (已预留 20%)",
                severity="high",
                filename=__file__,
                function="_check_memory_for_matmul",
                lineno=94,
                what="内存不足，无法进行压测"
            )
            return False
        
        return True
