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
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.utils import broadcast_to_maxshape
from msit_opcheck.constants import BFLOAT16, FLOAT16


class SelectOperation(OperationTest):
    def golden_calc(self, in_tensors):
        condition, x1, x2 = in_tensors[:3]
        output_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        shape_x1 = x1.shape
        shape_x2 = x2.shape
        shape_condition = condition.shape
        shape_list = broadcast_to_maxshape([shape_x1, shape_x2, shape_condition])
        x1 = np.broadcast_to(x1, shape_list[-1])
        x2 = np.broadcast_to(x2, shape_list[-1])
        condition = np.broadcast_to(condition, shape_list[-1])
        ones = np.ones(shape_list[-1], dtype=FLOAT16)
        equal_var = np.equal(condition, ones)
        if output_dtype == BFLOAT16:
            return [np.where(equal_var, x1, x2).astype(tf.bfloat16.as_numpy_dtype)]
        else:	
            return [np.where(equal_var, x1, x2).astype(output_dtype)]
    
    def test_select(self):
        self.execute()