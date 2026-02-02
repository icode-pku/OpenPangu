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

import numpy
import torch
import tensorflow.compat.v1 as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import bfloat16_conversion, DATA_TYPE_MAP
from msit_opcheck.constants import FLOAT32, BFLOAT16, COMPLEX32, BOOL


class MulOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        output_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        x1_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][0]['dtype']]
        x2_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][1]['dtype']]
        input_dtype = [x1_dtype, x2_dtype]
        if COMPLEX32 in input_dtype:
            x_real, x_imag = numpy.split(x1, 2, axis=-1)
            y_real, y_imag = numpy.split(x2, 2, axis=-1)
            z_real = x_real * y_real - x_imag * y_imag
            z_imag = x_real * y_imag + x_imag * y_real
            res = numpy.concatenate((z_real, z_imag), axis=-1)
            return [res]
        elif BOOL in input_dtype:
            x1 = torch.tensor(x1)
            x2 = torch.tensor(x2)
            res = torch.mul(x1, x2).detach().numpy()
            return [res]
        else:
            if x1_dtype == BFLOAT16 or x2_dtype == BFLOAT16:
                x1 = x1.astype(FLOAT32)
                x2 = x2.astype(FLOAT32)
            tf.disable_v2_behavior()
            tensor_x1 = tf.compat.v1.placeholder(x1.dtype, shape=x1.shape)
            tensor_x2 = tf.compat.v1.placeholder(x2.dtype, shape=x2.shape)
            feed_dict = {tensor_x1: x1, tensor_x2: x2}
            out = tf.multiply(tensor_x1, tensor_x2)
            with tf.compat.v1.Session() as sess:
                res = sess.run(out, feed_dict=feed_dict)
            output_dtype = bfloat16_conversion(output_dtype)
            res = res.astype(output_dtype[0])
            return [res]

    def test_mul(self):
        self.execute()
