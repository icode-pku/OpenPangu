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


class OpcheckGatingOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        topk = in_tensors[0]
        topk_expert_num = self.op_param.get("topkExpertNum", 1) # 每个token选择多少个专家
        cum_sum_num = self.op_param.get("cumSumNum", 1) # 一共有多少个专家

        expert_bin_count = torch.zeros(cum_sum_num, dtype=torch.int32)
        for i in range(topk.shape[0]):
            expert_index = topk[i].item()
            expert_bin_count[expert_index] += 1

        cum_sum_golden = torch.zeros(cum_sum_num)
        used_expert_index = torch.zeros(cum_sum_num)

        cum_sum_value = 0
        for i in range(cum_sum_num):
            cum_sum_value += expert_bin_count[i]
            cum_sum_golden[i] = cum_sum_value

        original_index_arr = torch.zeros(topk.shape[0], dtype=torch.int32)
        token_index_arr = torch.zeros(topk.shape[0], dtype=torch.int32)
        for i in range(topk.shape[0]):
            expert_index = topk[i].item()
            token_index = int(i / topk_expert_num)
            original_index = 1
            tmp_sorted_token_index = used_expert_index[expert_index] if expert_index == 0 \
                else cum_sum_golden[expert_index - 1] + used_expert_index[expert_index]
            used_expert_index[expert_index] += 1
            original_index_arr[int(tmp_sorted_token_index)] = original_index
            token_index_arr[int(tmp_sorted_token_index)] = token_index

        return [token_index_arr, cum_sum_num, original_index_arr]

    def test(self):
        ret = self.validate_param("topkExpertNum", "cumSumNum")
        if not ret:
            return
        self.execute()