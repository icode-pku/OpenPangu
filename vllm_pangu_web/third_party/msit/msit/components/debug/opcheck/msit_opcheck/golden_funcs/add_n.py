# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP


class AddOperation(OperationTest):
    def golden_calc(self, in_tensors):
        res = in_tensors[0]
        for i in range(1, len(in_tensors)):
            res = np.add(res, in_tensors[i])
        return [res.astype(DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']])]
    
    def test_add(self):
        self.execute()