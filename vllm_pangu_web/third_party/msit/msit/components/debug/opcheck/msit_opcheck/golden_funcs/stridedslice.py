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

import tensorflow.compat.v1 as tf

from msit_opcheck.operation_test import OperationTest


class StridedSliceOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        if len(in_tensors) > 1:
            begin = in_tensors[1]
            end = in_tensors[2]
            strides = in_tensors[3]
        else:
            for attr in self.op_param['attr']:
                if attr['key'] == 'begin':
                    begin = attr['value']['list']['i']
                if attr['key'] == 'end':
                    end = attr['value']['list']['i']
                if attr['key'] == 'strides':
                    strides = attr['value']['list']['i']
        for attr in self.op_param['attr']:
            if attr['key'] == 'begin_mask':
                begin_mask = attr['value']['i']
            if attr['key'] == 'end_mask':
                end_mask = attr['value']['i']
            if attr['key'] == 'ellipsis_mask':
                ellipsis_mask = attr['value']['i']
            if attr['key'] == 'new_axis_mask':
                new_axis_mask = attr['value']['i']
            if attr['key'] == 'shrink_axis_mask':
                shrink_axis_mask = attr['value']['i']
        tf.disable_v2_behavior()
        x_holder = tf.placeholder(x.dtype, shape=x.shape)
        res = tf.strided_slice(x_holder, begin, end, strides, begin_mask, end_mask,
                            ellipsis_mask, new_axis_mask, shrink_axis_mask)
        with tf.Session() as sess:
            result = sess.run(res, feed_dict={x_holder: x})
        return [result]

    def test_stride_slice(self):
        self.execute()
