# Copyright (c) 2023 Huawei Technologies Co., Ltd.
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

import unittest

from model_evaluation.common.enum import Framework
from model_evaluation.data import OpMap


class TestOpMap(unittest.TestCase):

    def setUp(self) -> None:
        self.onnx_op_map = OpMap.load_op_map(Framework.ONNX)
        self.tf_op_map = OpMap.load_op_map(Framework.TF)
        self.caffe_op_map = OpMap.load_op_map(Framework.CAFFE)

    def test_load_op_map_fail_case(self):
        op_map = None
        try:
            op_map = OpMap.load_op_map(Framework.UNKNOWN)
        except RuntimeError:
            pass
        self.assertIsNone(op_map)

    def test_load_op_map_success_case(self):
        self.assertIsNotNone(self.onnx_op_map)
        self.assertIsNotNone(self.tf_op_map)
        self.assertIsNotNone(self.caffe_op_map)

    def test_map_op_success_case(self):
        inner_op = self.caffe_op_map.map_op('Convolution')
        self.assertEqual(inner_op, 'Conv2D')

        inner_op = self.tf_op_map.map_op('Conv2D')
        self.assertEqual(inner_op, 'Conv2D')

    def test_map_onnx_op_success_case(self):
        inner_op = self.onnx_op_map.map_onnx_op('Add', 11)
        self.assertEqual(inner_op, 'Add')

        inner_op = self.onnx_op_map.map_onnx_op('Add', 0)
        self.assertIsNone(inner_op)


if __name__ == "__main__":
    unittest.main()
