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

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckElewiseSubOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        out = []
        seq_len = self.op_param.get("seqLen", None)
        head_num = self.op_param.get("headNum", None)
        for i, s in enumerate(seq_len):
            for _ in range(head_num):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    def test_2d_half(self):
        ret = self.validate_param("seqLen", "headNum")
        if not ret:
            return
        self.execute()