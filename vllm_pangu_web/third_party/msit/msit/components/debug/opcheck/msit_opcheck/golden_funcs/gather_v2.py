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

import tensorflow as tf

from msit_opcheck.operation_test import OperationTest


class GatherOperation(OperationTest):
    def golden_calc(self, in_tensors):
        params_data = in_tensors[0]
        indices_data = in_tensors[1]
        axis = 0
        batch_dims = 0
        if len(in_tensors) > 2:
            tmp_axis = in_tensors[2]
            if tmp_axis.ndim == 0 or (tmp_axis.ndim == 1 and tmp_axis.size == 1):
                axis = tmp_axis[0]
        else:
            for attr in self.op_param['attr']:
                if attr['key'] == 'axis':
                    axis = attr['value']['i']
                if attr['key'] == 'batch_dims':
                    batch_dims = attr['value']['i']

        params_shape_len = len(params_data.shape)
        indices_shape_len = len(indices_data.shape)
        batch_dims = batch_dims if batch_dims >= 0 else batch_dims + indices_shape_len
        axis = axis if axis >= 0 else axis + params_shape_len

        tf.compat.v1.disable_eager_execution()
        params_shape = params_data.shape
        indices_shape = indices_data.shape

        params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
        indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)
        with tf.compat.v1.Session() as sess:
            gather_res = tf.compat.v1.gather(params, indices, axis=axis, batch_dims=batch_dims, name=None)
            res = sess.run(gather_res, feed_dict={params: params_data, indices: indices_data})
        return [res]

    def test_gather(self):
        self.execute()