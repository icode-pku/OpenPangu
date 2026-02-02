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
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP, numpy_to_torch_tensor
from msit_opcheck.constants import FLOAT32, BFLOAT16


class SigmoidOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0]
        if input0.dtype == tf.bfloat16.as_numpy_dtype:
            input0 = input0.astype(FLOAT32)
        tensor_neg = input0 * (-1)
        tensor_exp = np.exp(tensor_neg)
        tensor_add = tensor_exp + 1
        res = 1 / tensor_add
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        if out_dtype == BFLOAT16:
            res = res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
        else:
            res = res.astype(out_dtype, copy=False)

        return [res]

    def test_sigmoid(self):
        self.execute()
        
