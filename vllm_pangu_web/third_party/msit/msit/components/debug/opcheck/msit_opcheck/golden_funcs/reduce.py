# -*- coding: utf-8 -*-
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

import functools

import numpy
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import bfloat16_conversion, DATA_TYPE_MAP
from msit_opcheck.constants import FLOAT32, FLOAT16, BFLOAT16

D_BOUNDS = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
    "int32": (-2147483648, 2147483647),
    "uint32": (0, 4294967295),
    "int64": (-9223372036854775808, 9223372036854775807),
    "uint64": (0, 18446744073709551615),
    "float16": (-65504.0, 65504.0),
    "float32": (-3.4028234663852886e+38, 3.4028234663852886e+38)
}


def _reduce_x_get_axis(op_info):
    axis = []
    for attr in op_info['attr']:
        if attr['key'] == 'axes':
            axis = attr['value']['list']['i']
    return axis


def _eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


class ReduceSumOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        axis = _reduce_x_get_axis(self.op_param)

        x_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][0]['dtype']]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]

        axis = _eliminate_duplicate_axes(axis, x)

        if not axis:
            return [x]

        if x_dtype in (FLOAT16, BFLOAT16):
            x = x.astype(numpy.float32)
            res = numpy.sum(x, axis=axis)
        else:
            res = numpy.sum(x, axis=axis)
        if out_dtype == BFLOAT16:
            return [res.astype(tf.bfloat16.as_numpy_dtype, copy=False)]

        return [res.astype(out_dtype, copy=False)]

    def test_reduce_sum(self):
        self.execute()


class ReduceMeanOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        axis = _reduce_x_get_axis(self.op_param)
        x_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][0]['dtype']]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]

        axis = _eliminate_duplicate_axes(axis, x)
        if not axis:
            return [x]

        reduce_shape = [x.shape[idx] for idx, _ in enumerate(x.shape) if idx in axis]
        cof_value = functools.reduce(lambda x, y: x * y, reduce_shape)
        if x_dtype in (FLOAT16, BFLOAT16):
            x = x.astype(numpy.float32) / cof_value
            y = numpy.sum(x, axis=axis)
            if out_dtype[0] == FLOAT16:
                y = y.astype(numpy.float16)
            elif out_dtype[0] == BFLOAT16:
                output_dtype = bfloat16_conversion(out_dtype)
                y = y.astype(output_dtype[0])
            return [y]
        else:
            x = x / cof_value
            return [numpy.sum(x, axis=axis, keepdims=True)]


    def test_reduce_mean(self):
        self.execute()