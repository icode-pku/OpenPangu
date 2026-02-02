# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

from msit_opcheck.conversion.dtype_convert import bfloat16_conversion_v2, DATA_TYPE_MAP
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.golden_funcs.mat_mul import matmul


class BatchMatMulOperation(OperationTest):
    def golden_calc(self, in_tensors):
        # input & params
        # 校验输入张量列表的长度，至少需要2个输入
        if len(in_tensors) < 2:
            raise ValueError("Insufficient input tensors, at least 2 input tensors are required")
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        for attr in self.op_param['attr']:
            if attr['key'] == 'adj_x1':
                trans_a = attr['value']['b']
            if attr['key'] == 'adj_x2':
                trans_b = attr['value']['b']
        # bias
        bias = None
        if len(in_tensors) > 2:
            bias = in_tensors[2]

        inputs = [x1, x2, trans_a, trans_b, out_dtype, bias]
        res = matmul(inputs)
        return [res]

    def test_batch_matmul(self):
        self.execute()
