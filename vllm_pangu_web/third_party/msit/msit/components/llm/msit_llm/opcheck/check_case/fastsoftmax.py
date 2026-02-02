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
import torch.nn as nn

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckFastSoftMaxOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        data_input = in_tensors[0]
        seq_len_list = self.op_param.get('qSeqLen', None)
        head_num_imm = self.op_param.get('headNum', None)
        golden = torch.empty_like(data_input)

        start = 0
        for seq_len in seq_len_list:
            end = start + head_num_imm * seq_len * seq_len
            cur_data_input = data_input[start:end].reshape(-1, seq_len)
            cur_golden = torch.softmax(cur_data_input.to(torch.float32), dim=-1).to(torch.float16)
            golden[start:end] = cur_golden.reshape(-1)
            start = end
        return [golden]

    def test_fastsoftmax(self):
        ret = self.validate_param("qSeqLen", "headNum")
        if not ret:
            return
        self.execute()