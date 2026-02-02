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

from msit_llm.opcheck import operation_test


class OpcheckPadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        tmp_out = in_tensors[0]
        padding_offset = in_tensors[1]
        seq_len = in_tensors[2]
        input_ids = in_tensors[3]
        batch = input_ids.shape[0]
        hidden_dim = tmp_out.shape[1]
        max_seq_len = input_ids.shape[1]

        golden_result = torch.zeros((batch, hidden_dim))
        temp_val = 0
        for i in range(batch):
            temp_val = temp_val + seq_len[i][0]
            golden_result[i] = tmp_out[temp_val - 1]
        return [golden_result]

    def test(self):
        self.execute()