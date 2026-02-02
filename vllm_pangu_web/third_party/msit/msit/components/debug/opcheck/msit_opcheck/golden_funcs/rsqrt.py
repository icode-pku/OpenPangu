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

import torch
import numpy as np
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.constants import BFLOAT16


class RsqrtOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0]
        input0 = torch.tensor(input0, dtype=torch.float32)
        output_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        res = torch.rsqrt(input0).numpy()
        if output_dtype == BFLOAT16:
            res = res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
        else:
            res = res.astype(output_dtype, copy=False)
        return [res]
    
    def test_rsqrt(self):
        self.execute()