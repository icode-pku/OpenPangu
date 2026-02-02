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


class BiasAddOperation(OperationTest):
    def golden_calc(self, in_tensors):
        value, bias = in_tensors
        bias_shape = [1] * (value.ndim - 1) + [bias.shape[0]]
        bias_reshape = bias.reshape(bias_shape)
        res = np.add(value, bias_reshape)
        return [res]
    
    def test_bias_add(self):
        self.execute()

