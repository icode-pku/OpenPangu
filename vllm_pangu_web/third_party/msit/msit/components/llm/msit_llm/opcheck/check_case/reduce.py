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

from enum import Enum
import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class ReduceType(Enum):
    REDUCE_UNDEFINED = 0
    REDUCE_MAX = 1
    REDUCE_MIN = 2
    REDUCE_SUM = 3


class OpcheckReduceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        op_type = self.op_param.get('reduceType', ReduceType.REDUCE_UNDEFINED.value)
        reduce_type_support_list = [
            ReduceType.REDUCE_MAX.value,
            ReduceType.REDUCE_MIN.value,
            ReduceType.REDUCE_SUM.value
        ]
        self.validate_int_range(op_type, reduce_type_support_list, "reduceType")
        axis = self.op_param.get('axis', None)
        if op_type == ReduceType.REDUCE_MAX.value:
            return [in_tensors[0].amax(axis)[0]]
        elif op_type == ReduceType.REDUCE_MIN.value:
            return [in_tensors[0].amin(axis)[0]]
        else:
            return [torch.sum(in_tensors[0], axis)]

    def test(self):
        ret = self.validate_param("reduceType", "axis")
        if not ret:
            return
        self.execute()