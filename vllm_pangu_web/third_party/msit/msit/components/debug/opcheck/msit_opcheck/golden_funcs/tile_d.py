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

import tensorflow as tf

from msit_opcheck.operation_test import OperationTest


class TileDOperation(OperationTest):
    def golden_calc(self, in_tensors):
        tf.compat.v1.disable_eager_execution()
        x = in_tensors[0]
        for attr in self.op_param['attr']:
            if attr['key'] == 'multiples':
                multiples = attr['value']['list']['i']
        if 'multiples' not in locals():
            multiples = in_tensors[1]

        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        out = tf.tile(tensor_x, multiples)
        with tf.compat.v1.Session() as sess:
            res = sess.run(out, feed_dict={tensor_x: x})
        return [res]

    def test_tiled(self):
        self.execute()