# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
from unittest.mock import MagicMock, patch

from msit_graph.inspect.scan import DynamicShape, execute


class TestDynamicShape(unittest.TestCase):
    def test_is_dynamic_shape(self):
        node = MagicMock()
        attr1 = MagicMock()
        attr1.name = "_is_unknown_shape"

        attr1.i = 1
        attr2 = MagicMock()
        attr2.name = "_force_unknown_shape"
        attr2.i = 1

        attr3 = MagicMock()
        attr3.name = "other_attr"
        attr3.i = 0

        node.attribute = [attr1, attr2, attr3]
        dynamic_shape = DynamicShape(None)
        self.assertTrue(dynamic_shape.is_dynamic_shape(node))

        attr1.i = 0
        attr2.i = 0
        self.assertFalse(dynamic_shape.is_dynamic_shape(node))

    def test_add_dynamic_op(self):
        dynamic_shape = DynamicShape(None)
        parent_name = "ParentNode"
        sub_node_name = "SubNode"
        inputs = "Input1"
        outputs = "Output1"

        dynamic_shape.add_dynamic_op(parent_name, sub_node_name, inputs, outputs)

        self.assertIn((parent_name, sub_node_name, inputs, outputs), dynamic_shape.dynamic_to_static_edges)

    def test_process_node(self):
        sub_node = MagicMock()
        sub_node.name = "SubNode"
        sub_node.input = ["Input2"]
        sub_node.output = ["Output2"]
        sub_node.attribute = []

        attr_with_graph = MagicMock()
        attr_with_graph.g = MagicMock(node=[sub_node])

        node = MagicMock()
        node.name = "ParentNode"
        node.input = ["Input1"]
        node.output = ["Output1"]
        node.attribute = [attr_with_graph]

        dynamic_shape = DynamicShape(None)
        dynamic_shape.is_dynamic_shape = MagicMock(return_value=True)
        dynamic_shape.dynamic_to_static_edges = []
        dynamic_shape.process_node(node)

        expected = ("ParentNode", "SubNode", ["Input2"], ["Output2"])
        self.assertIn(expected, dynamic_shape.dynamic_to_static_edges)

    def test_find_dynamic_shape_op(self):
        node1 = MagicMock(name="Node1", attribute=[], input="Input1", output="Output1")
        node2 = MagicMock(name="Node2", attribute=[], input="Input2", output="Output2")
        
        node1.name = "Node1"
        node1.input = "Input1"
        node1.output = "Output1"
        
        node2.name = "Node2"
        node2.input = "Input2"
        node2.output = "Output2"
        
        graph = MagicMock(node=[node1, node2])
        dynamic_shape = DynamicShape(graph)
        dynamic_shape.is_dynamic_shape = MagicMock(return_value=True)
        
        def append_dynamic_to_static_edges(node):
            dynamic_shape.dynamic_to_static_edges.append(
                (node.name, node.name, node.input, node.output)
            )
        dynamic_shape.process_node = MagicMock(side_effect=append_dynamic_to_static_edges)

        result = dynamic_shape.find_dynamic_shape_op()
        expected = [
            ('Node1', 'Node1', 'Input1', 'Output1'),
            ('Node2', 'Node2', 'Input2', 'Output2'),
        ]
        self.assertEqual(result, expected)

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("msit_graph.inspect.scan.save_dym_op")
    @patch("msit_graph.graph_extract.graph_extract.GraphAnalyze.load_graph_def_from_pbtxt")
    def test_execute(self, mock_load_graph, mock_save_dym_op, mock_isdir, mock_exists):
        args = MagicMock(type="dshape", input="input.pbtxt", output="/fake/output")
        mock_graph = MagicMock()
        mock_load_graph.return_value = mock_graph
        mock_exists.return_value = True
        mock_isdir.return_value = True

        execute(args)
        mock_load_graph.assert_called_once_with("input.pbtxt")
        mock_save_dym_op.assert_called_once()
