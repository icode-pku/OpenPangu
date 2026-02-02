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
from msit_opcheck.conversion.shape_convert import update_axis_for_npu_inner_format, is_transformable


class ConcatOperation(OperationTest):
    def golden_calc(self, in_tensors):
        ori_shape, ori_format = None, None
        input_format = self.op_param['input_desc'][0]['layout']
        for attr in self.op_param['attr']:
            if attr['key'] == 'concat_dim':
                axis = attr['value']['i']
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                ori_format = attr['value']['s']
            if attr['key'] == 'origin_shape':
                ori_shape = attr['value']['list']['i']
        if 'axis' not in locals():
            axis = in_tensors[-1].item()
            in_tensors = in_tensors[:-1]
        if ori_shape is not None and not is_transformable(input_format, ori_format):
            axis = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        return [np.concatenate(in_tensors, axis=axis)]

    def test_concat(self):
        self.execute()