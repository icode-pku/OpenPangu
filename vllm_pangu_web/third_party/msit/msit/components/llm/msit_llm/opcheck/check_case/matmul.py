# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import torch
import torch_npu
import numpy as np

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckMatmulOperation(operation_test.OperationTest):
    def golden_flp(self, transpose_a: bool, transpose_b: bool, in_tensor_0, in_tensor_1):
        if transpose_a:
            if len(in_tensor_0.shape) == 2:
                in_tensor_0 = torch.permute(in_tensor_0, (1, 0))
            if len(in_tensor_0.shape) == 3:
                in_tensor_0 = torch.permute(in_tensor_0, (0, 2, 1))
        if len(in_tensor_1.shape) == 4:
            in_tensor_1 = torch.permute(in_tensor_1, (0, 2, 1, 3))
            if in_tensor_1.shape[0] == 1:
                in_tensor_1 = in_tensor_1.reshape(in_tensor_1.shape[1], in_tensor_1.shape[2] * in_tensor_1.shape[3])
            else:
                in_tensor_1 = in_tensor_1.reshape(in_tensor_1.shape[0], in_tensor_1.shape[1],
                                                  in_tensor_1.shape[2] * in_tensor_1.shape[3])
        if transpose_b:
            if len(in_tensor_1.shape) == 2:
                in_tensor_1 = torch.permute(in_tensor_1, (1, 0))
            if len(in_tensor_1.shape) == 3:
                in_tensor_1 = torch.permute(in_tensor_1, (0, 2, 1))
        golden_result = torch.matmul(in_tensor_0, in_tensor_1)
        return golden_result

    def golden_calc(self, in_tensors):
        transpose_a = self.op_param.get("transposeA", None)
        transpose_b = self.op_param.get("transposeB", None)
        golden_result = self.golden_flp(transpose_a, transpose_b, in_tensors[0], in_tensors[1])
        golden_result = torch.tensor(golden_result).half()
        return [golden_result]

    def test(self):
        ret = self.validate_param("transposeA", "transposeB")
        if not ret:
            return
        self.execute()