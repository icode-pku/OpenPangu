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


class OpcheckSliceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        offset_list = self.op_param.get('offsets', None)
        size_list = self.op_param.get('size', None)
        for index, offset in enumerate(offset_list):
            offset_list[index] = offset if offset >= 0 else offset + in_tensors[0].shape[index]
        for index, size in enumerate(size_list):
            size_list[index] = size if size != -1 else in_tensors[0].shape[index] - offset_list[index]
        self.validate_int_range(len(offset_list), [2, 3, 4], "len(offsets)")
        if len(offset_list) == 2:
            return [in_tensors[0][offset_list[0]: offset_list[0] + size_list[0], 
                    offset_list[1]: offset_list[1] + size_list[1]]]
        elif len(offset_list) == 3:
            return [in_tensors[0][offset_list[0]: offset_list[0] + size_list[0], 
                    offset_list[1]: offset_list[1] + size_list[1], offset_list[2]: offset_list[2] + size_list[2]]]
        else:
            return [in_tensors[0][offset_list[0]: offset_list[0] + size_list[0], 
                    offset_list[1]: offset_list[1] + size_list[1], offset_list[2]: offset_list[2] + size_list[2], 
                    offset_list[3]: offset_list[3] + size_list[3]]]

    def test(self):
        ret = self.validate_param("offsets", "size")
        if not ret:
            return
        self.execute()