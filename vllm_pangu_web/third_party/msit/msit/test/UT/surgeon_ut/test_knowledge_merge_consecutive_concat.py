# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import unittest

import numpy as np
import onnx
from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_concat import KnowledgeMergeConsecutiveConcat
from helper import KnowledgeTestHelper, OptimizationConfig


def make_c2_concat_model(onnx_name, x, y, z, diff_axis=False):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, z.shape)

    input_o = helper.make_tensor_value_info("O", TensorProto.FLOAT, None)

    axis = 1 if diff_axis else 0
    node_concat0 = helper.make_node("Concat", ["X", "Y"], ["X_S"], "Concat0", axis=0)
    node_concat1 = helper.make_node("Concat", ["X_S", "Z"], ["O"], "Concat1", axis=axis)

    graph = helper.make_graph(
        [node_concat0, node_concat1], "continue_concat_test",
        [input_x, input_y, input_z], [input_o]
    )
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


class TestKnowledgeMergeConsecutiveConcat(unittest.TestCase, KnowledgeTestHelper):

    def test_merge_c2_concat(self):
        x = np.random.randn(2, 1, 2).astype(np.float32)
        y = np.random.randn(1, 1, 2).astype(np.float32)
        z = np.random.randn(3, 1, 2).astype(np.float32)

        onnx_name = "c2_concat"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c2_concat_model(onnx_ori, x, y, z, False)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveConcat(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [
            {
                'X': np.random.randn(*x.shape).astype(x.dtype),
                'Y': np.random.randn(*y.shape).astype(y.dtype),
                'Z': np.random.randn(*z.shape).astype(z.dtype),
            }
            for _ in range(10)
        ]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c2_diff_axis_concat(self):
        x = np.random.randn(2, 1, 2).astype(np.float32)
        y = np.random.randn(1, 1, 2).astype(np.float32)
        z = np.random.randn(3, 1, 2).astype(np.float32)

        onnx_name = "c2_concat_diff_axis"
        onnx_ori = f"./{onnx_name}.onnx"

        make_c2_concat_model(onnx_ori, x, y, z, True)
        graph = OnnxGraph.parse(onnx_ori)

        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveConcat(),
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=False))

    def tearDown(self):
        super().tearDown()
        for filename in os.listdir('.'):
            if filename.startswith('c2_concat'):
                os.remove(filename)


if __name__ == "__main__":
    unittest.main()
