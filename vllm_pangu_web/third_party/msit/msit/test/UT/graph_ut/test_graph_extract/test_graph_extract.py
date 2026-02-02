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
import sys
import unittest
from argparse import Namespace
from unittest.mock import MagicMock

from msit_graph.graph_extract.graph_extract import GraphAnalyze, GraphSummary


class TestGraphAnalyze(unittest.TestCase):

    def setUp(self):
        # Mocking the onnx and text_format modules
        self.onnx_mock = sys.modules['onnx']
        self.text_format_mock = sys.modules['google.protobuf.text_format']

        # Mocking the ms_open function
        self.ms_open_mock = MagicMock()
        sys.modules['components.utils.file_open_check'].ms_open = self.ms_open_mock

        # Setting up constants
        GraphAnalyze.LOOKUP_FORWARD = 1
        GraphAnalyze.LOOKUP_BACKWARD = 2
        GraphAnalyze.LOOKUP_ALL = 3

    def test__append_file_name_suffix_given_path_without_extension_when_valid_then_correct_suffix_added(self):
        result = GraphAnalyze._append_file_name_suffix("path/to/file", "suffix")
        self.assertEqual(result, "path/to/file_suffix")

    def test__append_file_name_suffix_given_path_with_extension_when_valid_then_correct_suffix_added(self):
        result = GraphAnalyze._append_file_name_suffix("path/to/file.txt", "suffix")
        self.assertEqual(result, "path/to/file_suffix.txt")

    def test__get_node_name_given_tensor_name_with_carrot_when_valid_then_correct_name_returned(self):
        result = GraphAnalyze._get_node_name("^node_name")
        self.assertEqual(result, "node_name")

    def test__get_node_name_given_tensor_name_with_colon_when_valid_then_correct_name_returned(self):
        result = GraphAnalyze._get_node_name("node_name:1")
        self.assertEqual(result, "node_name")

    def test__get_node_name_given_tensor_name_without_special_chars_when_valid_then_correct_name_returned(self):
        result = GraphAnalyze._get_node_name("node_name")
        self.assertEqual(result, "node_name")

    def test__get_node_index_and_name_given_tensor_name_with_carrot_when_valid_then_correct_tuple_returned(self):
        result = GraphAnalyze._get_node_index_and_name("^node_name")
        self.assertEqual(result, (0, "node_name"))

    def test__get_node_index_and_name_given_tensor_name_with_colon_when_valid_then_correct_tuple_returned(self):
        result = GraphAnalyze._get_node_index_and_name("node_name:1")
        self.assertEqual(result, (1, "node_name"))

    def test__get_node_index_and_name_given_tensor_name_without_colon_when_invalid_then_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            GraphAnalyze._get_node_index_and_name("node_name")
        self.assertIn("Invalid tensor name format: node_name", str(context.exception))

    def test__get_node_index_and_name_given_tensor_name_with_non_int_index_when_invalid_then_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            GraphAnalyze._get_node_index_and_name("node_name:abc")
        self.assertIn("Index part of tensor name node_name:abc is not an integer", str(context.exception))

    def test__lookup_dump_nodes_given_forward_lookup_when_valid_then_correct_nodes_returned(self):
        args = Namespace(stop_name=None, layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = "node1"
        lookup_directions = GraphAnalyze.LOOKUP_FORWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1", "node2"})

    def test__lookup_dump_nodes_given_backward_lookup_when_valid_then_correct_nodes_returned(self):
        args = Namespace(stop_name=None, layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = "node2"
        lookup_directions = GraphAnalyze.LOOKUP_BACKWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1", "node2"})

    def test__lookup_dump_nodes_given_stop_name_when_valid_then_stops_at_correct_node(self):
        args = Namespace(stop_name="node2", layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = "node1"
        lookup_directions = GraphAnalyze.LOOKUP_FORWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1"})
