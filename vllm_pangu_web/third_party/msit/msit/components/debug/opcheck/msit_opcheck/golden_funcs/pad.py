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

from collections import deque
import numpy as np
import torch

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.constants import BFLOAT16


class PadOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input_data, paddings = in_tensors
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                input_format = attr['value']['s']
        bf16_mark = False
        if True:
            pad_shape = deque()
            for i in range(len(paddings)):
                pad_shape.append(paddings[len(paddings) - 1 - i][0])
                pad_shape.append(paddings[len(paddings) - 1 - i][1])

            if (input_format == "NC1HWC0"):
                pad_shape.appendleft(0)
                pad_shape.appendleft(0)
            if BFLOAT16 in str(input_data.dtype):
                bf16_mark = True
                input_data = input_data.astype(np.float32)
            input_data_tensor = torch.from_numpy(input_data)
            golden = torch.constant_pad_nd(input_data_tensor, tuple(pad_shape), 0)
            if bf16_mark:
                golden.to(torch.bfloat16)
            res = golden.numpy()
        return [res]

    def test_pad(self):
        self.execute()


