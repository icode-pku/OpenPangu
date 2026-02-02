# -*- coding: utf-8 -*-
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
import tensorflow.compat.v1 as tf
import torch

from msit_opcheck.conversion.dtype_convert import bfloat16_conversion_v2, DATA_TYPE_MAP
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.shape_convert import format_transformation_map

tf.disable_v2_behavior()


def hf_32_input_gerenate(input_fp32):   
    input_hf32 = input_fp32.view(np.int32)
    input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
    input_hf32 = np.left_shift(input_hf32, 12)
    input_hf32 = input_hf32.view(np.float32)
    return input_hf32


def matmul(inputs):
    x1, x2, trans_a, trans_b, out_dtype, bias = inputs
    tf.compat.v1.disable_eager_execution()

    if x1.dtype == 'float32':
        a_data = x1.astype('float64')
        b_data = x2.astype('float64')
    else:
        a_data = x1.astype('float32')
        b_data = x2.astype('float32')

    a = torch.from_numpy(a_data)
    b = torch.from_numpy(b_data)
    if trans_a:
        if len(a.shape) == 2:
            a = a.t()
        else:
            a = a.transpose(-1, -2)
    if trans_b:
        if len(b.shape) == 2:
            b = b.t()
        else:
            b = b.transpose(-1, -2)
    res_pt = torch.matmul(a, b).numpy() # (1, 1, 16, 16)
    if bias is not None:
        if x1.dtype == 'float32':
            bias = bias.astype('float64')
        else:
            bias = bias.astype('float32')
        res_pt = torch.from_numpy(res_pt)
        bias = torch.from_numpy(bias)
        res_pt = torch.add(res_pt, bias).numpy()
    output_dtype = bfloat16_conversion_v2([out_dtype])
    return res_pt.astype(output_dtype[0], copy=False)


class MatmulOperation(OperationTest):
    def golden_calc(self, in_tensors):
        # input & params
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        for attr in self.op_param['attr']:
            if attr['key'] == 'transpose_x1':
                trans_a = attr['value']['b']
            if attr['key'] == 'transpose_x2':
                trans_b = attr['value']['b']

        # bias
        bias = None
        if len(in_tensors) > 2:
            bias = in_tensors[2]
            for attr in self.op_param['input_desc'][2]['attr']:
                if attr['key'] == 'origin_format':
                    bias_ori_format = attr['value']['s']
                if attr['key'] == 'origin_shape':
                    bias_ori_shape = attr['value']['list']['i']
            bias_new_format = self.op_param['input_desc'][2]['layout']
            bias = format_transformation_map[bias_new_format][bias_ori_format](bias, bias_new_format, bias_ori_shape)

        inputs = [x1, x2, trans_a, trans_b, out_dtype, bias]
        res = matmul(inputs)
        return [res]

    def test_matmul(self):
        self.execute()
