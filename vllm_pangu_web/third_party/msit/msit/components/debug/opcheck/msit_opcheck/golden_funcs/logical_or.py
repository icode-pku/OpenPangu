# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

import numpy as np

from msit_opcheck.utils import broadcast_to_maxshape
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.constants import FLOAT16


class LogicalOrOperation(OperationTest):
    def golden_calc(self, in_tensors):
        # 校验输入张量列表长度
        if len(in_tensors) < 2:
            raise ValueError("Insufficient input tensors, at least 2 input tensors are required")
            
        x1, x2 = in_tensors[:2]
        
        # 校验op_param字典结构
        if not isinstance(self.op_param, dict) or 'output_desc' not in self.op_param:
            raise ValueError("Invalid op_param format")
        if not isinstance(self.op_param['output_desc'], list) or len(self.op_param['output_desc']) == 0:
            raise ValueError("Invalid output_desc format")
        if not isinstance(self.op_param['output_desc'][0], dict) or 'dtype' not in self.op_param['output_desc'][0]:
            raise ValueError("Invalid output_desc[0] format")
            
        # 校验dtype是否在数据类型映射表中
        dtype = self.op_param['output_desc'][0]['dtype']
        if dtype not in DATA_TYPE_MAP:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        out_type = DATA_TYPE_MAP[dtype]
        shape_list = broadcast_to_maxshape([x1.shape, x2.shape])
        x1 = x1.astype(FLOAT16)
        x2 = x2.astype(FLOAT16)
        x1 = np.broadcast_to(x1, shape_list[-1])
        x2 = np.broadcast_to(x2, shape_list[-1])
        return [np.maximum(x1, x2).astype(out_type, copy=False)]

    def test_logical_or(self):
        self.execute()
