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


import unittest
import numpy as np

from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, OnnxNode
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from test_node_common import is_ph_equal, is_ini_equal, is_node_equal
from test_graph_basic import is_graph_equal


def create_graph(name: str = 'test_graph'):
    input_ = OnnxPlaceHolder('input', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('output', np.dtype('float32'), [1, 3, 224, 224])
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['sqrt0_output'], attrs={})
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    node_4 = OnnxNode('flatten4', 'Flatten', inputs=['relu3_output'], outputs=['flatten4_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_0, node_1, node_2, node_3, node_4],
        inputs=[input_],
        outputs=[output],
    )


def create_graph2(name: str = 'test_graph2'):
    input_ = OnnxPlaceHolder('input', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('output', np.dtype('float32'), [1, 3, 224, 224])
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['sqrt0_output'], attrs={})
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_0, node_1, node_2],
        inputs=[input_],
        outputs=[output],
    )


def create_subgraph(name: str = "test_subgraph"):
    input_ = OnnxPlaceHolder('sqrt0_output', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('relu3_output', np.dtype('float32'), [1, 3, 224, 224])
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_1, node_2, node_3],
        inputs=[input_],
        outputs=[output],
        value_infos=[
            OnnxPlaceHolder('relu1_output', np.dtype('float32'), [1, 3, 224, 224]),
            OnnxPlaceHolder('sqrt2_output', np.dtype('float32'), [1, 3, 224, 224]),
        ]
    )


def create_subgraph_with_specific_input_dtype(name: str = "test_subgraph"):
    input_ = OnnxPlaceHolder('sqrt0_output', np.dtype('int8'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('relu3_output', np.dtype('float32'), [1, 3, 224, 224])
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_1, node_2, node_3],
        inputs=[input_],
        outputs=[output],
        value_infos=[
            OnnxPlaceHolder('relu1_output', np.dtype('float32'), [1, 3, 224, 224]),
            OnnxPlaceHolder('sqrt2_output', np.dtype('float32'), [1, 3, 224, 224]),
        ]
    )


def create_subgraph_with_specific_input_shape(name: str = "test_subgraph"):
    input_ = OnnxPlaceHolder('sqrt0_output', np.dtype('float32'), [4, 3, 640, 640])
    output = OnnxPlaceHolder('relu3_output', np.dtype('float32'), [1, 3, 224, 224])
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_1, node_2, node_3],
        inputs=[input_],
        outputs=[output],
        value_infos=[
            OnnxPlaceHolder('relu1_output', np.dtype('float32'), [1, 3, 224, 224]),
            OnnxPlaceHolder('sqrt2_output', np.dtype('float32'), [1, 3, 224, 224]),
        ]
    )


def create_subgraph_with_specific_input_shape_and_dtype(name: str = "test_subgraph"):
    input_ = OnnxPlaceHolder('sqrt0_output', np.dtype('int8'), [4, 3, 640, 640])
    output = OnnxPlaceHolder('relu3_output', np.dtype('float32'), [1, 3, 224, 224])
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_1, node_2, node_3],
        inputs=[input_],
        outputs=[output],
        value_infos=[
            OnnxPlaceHolder('relu1_output', np.dtype('float32'), [1, 3, 224, 224]),
            OnnxPlaceHolder('sqrt2_output', np.dtype('float32'), [1, 3, 224, 224]),
        ]
    )


class TestGraphExtract(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(OnnxNode, is_node_equal)
        self.addTypeEqualityFunc(OnnxPlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(OnnxInitializer, is_ini_equal)
        self.addTypeEqualityFunc(OnnxGraph, is_graph_equal)
        self.graph = create_graph()
        self.graph2 = create_graph2()
        self.subgraph = create_subgraph()
        self.subgraph_with_input_shape = create_subgraph_with_specific_input_shape()
        self.subgraph_with_input_dtype = create_subgraph_with_specific_input_dtype()
        self.subgraph_with_input_shape_and_dtype = create_subgraph_with_specific_input_shape_and_dtype()

    def test_extract_subgraph_when_basic_params_then_pass(self):
        sub_graph = self.graph.extract_subgraph(start_node_names=["relu1"],
                                                end_node_names=["relu3"])
        self.assertEqual(sub_graph, self.subgraph)

    def test_extract_subgraph_when_add_input_shape_then_pass(self):
        sub_graph = self.graph.extract_subgraph(start_node_names=["relu1"],
                                                end_node_names=["relu3"],
                                                input_shape="sqrt0_output:4,3,640,640")
        self.assertEqual(sub_graph, self.subgraph_with_input_shape)

    def test_extract_subgraph_when_add_input_dytpe_then_pass(self):
        sub_graph = self.graph.extract_subgraph(start_node_names=["relu1"],
                                                end_node_names=["relu3"],
                                                input_dtype="sqrt0_output:int8")
        self.assertEqual(sub_graph, self.subgraph_with_input_dtype)

    def test_extract_subgraph_when_add_input_shape_and_dytpe_then_pass(self):
        sub_graph = self.graph.extract_subgraph(start_node_names=["relu1"],
                                                end_node_names=["relu3"],
                                                input_shape="sqrt0_output:4,3,640,640",
                                                input_dtype="sqrt0_output:int8")
        self.assertEqual(sub_graph, self.subgraph_with_input_shape_and_dtype)


if __name__ == '__main__':
    unittest.main()
