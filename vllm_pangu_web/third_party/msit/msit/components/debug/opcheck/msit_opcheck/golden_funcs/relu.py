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

import numpy as np

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.constants import FLOAT32, BFLOAT16


class ReluOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0]
        if BFLOAT16 in str(input0.dtype):
            x = input0.astype(FLOAT32)
            res = np.maximum(x, 0)
            res = res.astype(input0.dtype, copy=False)
        else:
            res = np.maximum(input0, 0)
        return [res]

    def test_relu(self):
        self.execute()