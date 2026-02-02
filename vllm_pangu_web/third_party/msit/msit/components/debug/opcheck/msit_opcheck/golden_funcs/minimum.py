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

import tensorflow as tf
import numpy as np

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP, numpy_to_torch_tensor, bfloat16_conversion


class MinimumOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]

        res = np.minimum(x1, x2)
        out_dtype = bfloat16_conversion(out_dtype)
        res = res.astype(out_dtype[0])
        return [res]

    def test_minimum(self):
        self.execute()