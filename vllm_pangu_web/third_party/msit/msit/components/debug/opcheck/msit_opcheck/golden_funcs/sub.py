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

import numpy as np
import torch
import tensorflow.compat.v1 as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.utils import broadcast_to_maxshape


class SubOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        type_ = x1.dtype
        if type_ == "bool":
            x1, x2 = torch.tensor(x1), torch.tensor(x2)
            res = torch.logical_xor(x1, x2).numpy()
        else:
            shape_list = broadcast_to_maxshape([x1.shape, x2.shape])
            x1 = np.broadcast_to(x1, shape_list[-1])
            x2 = np.broadcast_to(x2, shape_list[-1])
            tf.disable_v2_behavior()
            tensor_x1 = tf.placeholder(x1.dtype, shape=x1.shape)
            tensor_x2 = tf.placeholder(x2.dtype, shape=x2.shape)
            out = tf.subtract(tensor_x1, tensor_x2)
            feed_dict = {tensor_x1: x1, tensor_x2: x2}
            with tf.Session() as sess:
                res = sess.run(out, feed_dict=feed_dict)
        return [res]

    def test_sub(self):
        self.execute()