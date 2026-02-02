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


class OpcheckNonzeroOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        num_non_negative = torch.count_nonzero(in_tensors[0])
        padding_num = in_tensors[0].numel() - num_non_negative
        padding = torch.zeros(len(in_tensors[0].shape), padding_num)
        result = torch.stack(list(torch.nonzero(in_tensors[0], as_tuple=True)))
        result = torch.concat((result, padding), dim=1).long()

        return [result, torch.tensor(num_non_negative).long()]

    def test(self):
        self.execute()