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


def gen_softmax_grad(head_num, seq_len):
    x = torch.randn([head_num * seq_len, seq_len]).to(torch.float32)
    x.requires_grad = True
    y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float32)
    y.retain_grad()
    w = torch.randn_like(x).to(torch.float32)
    loss = (w * y).sum()
    loss.backward()
    return (y.detach().to(torch.float16), y.grad.detach().to(torch.float16), x.grad.detach().to(torch.float16))


class OpcheckFastSoftMaxGradOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        batch_size_imm = 4
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm, ]).to(torch.int64)
        seq_len = torch.randint(100, 300, [batch_size_imm, ]).to(torch.int32)
        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.concat(y_input_list)
        y_grad = torch.concat(y_grad_list)
        golden = torch.concat(golden_list)

        return [golden]

    def test_fastsoftmaxgrad(self):
        self.execute()