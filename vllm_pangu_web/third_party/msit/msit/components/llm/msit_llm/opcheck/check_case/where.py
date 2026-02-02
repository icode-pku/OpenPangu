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


class OpcheckWhereOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.where(in_tensors[0].bool(), in_tensors[1], in_tensors[2])
        return [golden_result]

    def test(self):
        self.execute()