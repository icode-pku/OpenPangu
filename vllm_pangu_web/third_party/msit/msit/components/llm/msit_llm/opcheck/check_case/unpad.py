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


class OpcheckUnpadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        input_ids = in_tensors[0]
        cum_offsets_now = in_tensors[1].reshape(-1)
        token_num = in_tensors[2]
        seq_len = in_tensors[3]
        batch = in_tensors[0].shape[0]
        total_length_imm = in_tensors[0].shape[1]

        x_remove_padding = input_ids[0][0:seq_len[0]]
        for i in range(1, batch):
            x_remove_padding = torch.concat((x_remove_padding, input_ids[i][0:seq_len[i]]))
        x_remove_padding = torch.pad(x_remove_padding, (0, (batch * total_length_imm - token_num[0][0])))
        cum_offsets_out = torch.zeros(batch)
        for i in range(1, batch):
            cum_offsets_out[i] = cum_offsets_now[i - 1]
        padding_offset = seq_len[0] * [0]
        for i in range(1, batch):
            temp_pad_out = seq_len[i] * [cum_offsets_now[i - 1]]
            padding_offset = torch.concat((padding_offset, temp_pad_out))
        zero_offset = torch.zeros((1, batch * total_length_imm - token_num[0][0]))
        padding_offset = torch.append(padding_offset, zero_offset)
        x_remove_padding = x_remove_padding.reshape(1, batch * total_length_imm).long()
        cum_offsets_out = cum_offsets_out.reshape(batch, 1).int()
        padding_offset = padding_offset.reshape(1, batch * total_length_imm).int()
        return [x_remove_padding, cum_offsets_out, padding_offset] # 输出 int64, int32, int32

    def test(self): 
        self.execute()