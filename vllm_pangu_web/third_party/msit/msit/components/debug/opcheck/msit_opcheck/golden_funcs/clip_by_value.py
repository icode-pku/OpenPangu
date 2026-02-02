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

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.constants import FLOAT32, BFLOAT16


class ClipByValueOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input_t = in_tensors[0]
        clip_value_min = in_tensors[1]
        clip_value_max = in_tensors[2]
        if BFLOAT16 in str(input_t.dtype):
            input_t = input_t.astype(FLOAT32)
            clip_value_min = clip_value_min.astype(FLOAT32)
            clip_value_max = clip_value_max.astype(FLOAT32)
        min_ = np.minimum(input_t, clip_value_max)
        res = np.maximum(min_, clip_value_min)
        if BFLOAT16 in str(input_t.dtype):
            return [res.astype(input_t.dtype)]
        return [res]
    
    def test_clip_by_value(self):
        self.execute()