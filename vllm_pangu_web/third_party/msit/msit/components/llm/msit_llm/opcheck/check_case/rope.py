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


class OpcheckUnpadRopeOperation(operation_test.OperationTest):
    def rotate_half(self, x):
        x0, x1 = x.chunk(2, -1)
        return torch.concat((-x1, x0), dim=x0.ndim - 1)

    def golden_func1(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = in_tensors[4].tolist()
        batch = in_tensors[4].shape[0]
        hidden_size = in_tensors[0].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        q_list = []
        k_list = []
        offset = 0
        for i, _ in enumerate(range(batch)):
            cur_seqlen = seqlen[i]
            next_offset = offset + cur_seqlen
            qlayer = in_tensors[0][offset:next_offset].view(cur_seqlen, head_num, head_size)
            q0, q1 = qlayer.chunk(2, -1)
            klayer = in_tensors[1][offset:next_offset].view(cur_seqlen, head_num, head_size)
            k0, k1 = klayer.chunk(2, -1)
            cos0, cos1 = in_tensors[2][offset:next_offset].unsqueeze(1).chunk(2, -1)
            sin0, sin1 = in_tensors[3][offset:next_offset].unsqueeze(1).chunk(2, -1)
            q0 = (q0 * cos0) + (self.rotate_half(q0) * sin0)
            k0 = (k0 * cos0) + (self.rotate_half(k0) * sin0)
            q1 = (q1 * cos1) + (self.rotate_half(q1) * sin1)
            k1 = (k1 * cos1) + (self.rotate_half(k1) * sin1)
            q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(cur_seqlen, hidden_size)
            q_list.append(q)
            k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(cur_seqlen, hidden_size)
            k_list.append(k)
            offset = next_offset
        q_sum = torch.concat(tuple(q_list), dim=0)
        k_sum = torch.concat(tuple(k_list), dim=0)
        return [q_sum, k_sum]

    def golden_func2(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        hidden_size = in_tensors[0].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        qlayer = in_tensors[0].view(seqlen, batch, head_num, head_size)
        q0, q1 = qlayer.chunk(2, -1)
        klayer = in_tensors[1].view(seqlen, batch, head_num, head_size)
        k0, k1 = klayer.chunk(2, -1)
        cos0, cos1 = in_tensors[2].view(seqlen, batch, 1, head_size).chunk(2, -1)
        sin0, sin1 = in_tensors[3].view(seqlen, batch, 1, head_size).chunk(2, -1)
        q0 = (q0 * cos0) + (self.rotate_half(q0) * sin0)
        k0 = (k0 * cos0) + (self.rotate_half(k0) * sin0)
        q1 = (q1 * cos1) + (self.rotate_half(q1) * sin1)
        k1 = (k1 * cos1) + (self.rotate_half(k1) * sin1)
        q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(ntoken, hidden_size)
        k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(ntoken, hidden_size)
        return [q, k]

    def golden_func3(self, in_tensors):
        if len(in_tensors[0].size()) == 4:
            seqlen = in_tensors[0].size()[1]
            batch = in_tensors[0].size()[0]
            q_head_num = in_tensors[0].size()[2]
            k_head_num = in_tensors[1].size()[2]
        else:
            ntoken = in_tensors[0].size()[0]
            seqlen = int(in_tensors[4][0])
            batch = ntoken // seqlen
            hidden_sizeq = in_tensors[0].size()[1]
            head_size = in_tensors[2].size()[1]
            q_head_num = hidden_sizeq // head_size
            hidden_sizek = in_tensors[1].size()[1]
            k_head_num = hidden_sizek // head_size
        rot_dim = in_tensors[2].size()[1]

        if batch == 0:
            batch = 1
            seqlen = ntoken
        q = in_tensors[0]
        k = in_tensors[1]
        qshaped = q.reshape(batch, -1, q_head_num, rot_dim // 2, 2)
        kshaped = k.reshape(batch, -1, k_head_num, rot_dim // 2, 2)
        cos = in_tensors[2].view(-1, 2)[:, 0].view(batch, -1, 1, qshaped.size(3))
        sin = in_tensors[3].view(-1, 2)[:, 0].view(batch, -1, 1, qshaped.size(3))

        q_out2 = torch.stack(
            [
                qshaped[..., 0] * cos - qshaped[..., 1] * sin,
                qshaped[..., 1] * cos + qshaped[..., 0] * sin,
            ],
            -1,
        )

        q_out2 = q_out2.flatten(3)
        k_out2 = torch.stack(
            [
                kshaped[..., 0] * cos - kshaped[..., 1] * sin,
                kshaped[..., 1] * cos + kshaped[..., 0] * sin,
            ],
            -1,
        )
        k_out2 = k_out2.flatten(3)

        if len(in_tensors[0].size()) == 4:
            return [q_out2, k_out2]
        else:
            return [q_out2.view(ntoken, hidden_sizeq), k_out2.view(ntoken, hidden_sizek)]

    def golden_func4(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        if batch == 0:
            batch = 1
            seqlen = ntoken
        hidden_size = in_tensors[0].size()[1]
        hidden_size1 = in_tensors[1].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        head_num1 = hidden_size1 // head_size
        q = in_tensors[0].view(batch, seqlen, head_num, head_size)
        k = in_tensors[1].view(batch, seqlen, head_num1, head_size)
        cos = in_tensors[2].view(batch, seqlen, head_size).unsqueeze(2)
        sin = in_tensors[3].view(batch, seqlen, head_size).unsqueeze(2)
        q_embed = ((q * cos) + (self.rotate_half(q) * sin)).view(ntoken, hidden_size)
        k_embed = ((k * cos) + (self.rotate_half(k) * sin)).view(ntoken, hidden_size1)
        return [q_embed, k_embed]

    def golden_calc(self, in_tensors):
        rotary_coeff = self.op_param.get('rotaryCoeff', None)
        if rotary_coeff == 4:
            if in_tensors[4].size()[0] == 3:
                return self.golden_func1(in_tensors)
            else:
                return self.golden_func2(in_tensors)
        elif rotary_coeff == 64:
            return self.golden_func3(in_tensors)
        else:
            return self.golden_func4(in_tensors)

    def test(self):
        ret = self.validate_param("rotaryCoeff")
        if not ret:
            return
        self.execute()