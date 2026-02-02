# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import argparse

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

import onnx
from google.protobuf import text_format

from components.utils.log import logger
from components.utils.file_open_check import (
    ms_open, 
    OpenException, 
    MAX_SIZE_LIMITE_CONFIG_FILE, 
    MAX_SIZE_LIMITE_NORMAL_FILE,
)
from components.utils.constants import MAX_DEPTH_LIMIT

ITERATIONS = 100


class GraphSummary:
    def __init__(self):
        # Maps node names to their input names
        self.names_to_input_names = {}
        # Maps node names to their data input names
        self.names_to_data_input_names = {}
        # Maps node names to their control input names
        self.names_to_ctrl_input_names = {}
        # Maps node names to their output names (list of outputs)
        self.names_to_output_names = defaultdict(list)
        # Maps node names to their corresponding ONNX Node objects
        self.names_to_node = {}
        # Keeps track of node sequences for maintaining original order
        self.names_to_seq_num = {}


class GraphAnalyze:

    LOOKUP_FORWARD = 1
    LOOKUP_BACKWARD = 2
    LOOKUP_ALL = 3

    STRIP_CONST = 1
    STRIP_ATTR_WITHOUT_SHAPE = 2
    STRIP_ATTR = 3

    def __init__(self):
        pass

    @staticmethod
    def validate_file_path(file_path, expected_extension):
        if not os.path.isfile(file_path):
            logger.error("File not found: %r" % file_path)
            raise FileNotFoundError("File not found: %r" % file_path)
        if not file_path.endswith(expected_extension):
            logger.error("Incorrect file extension for %r. Expected %s" % (file_path, expected_extension))
            raise ValueError("Incorrect file extension for %r. Expected %s" % (file_path, expected_extension))

    @staticmethod
    def load_graph_def_from_pbtxt(path):
        """Loads an ONNX graph definition from a binary protocol buffer file."""
        logger.info("Loading %r, the graph maybe huge, please wait a minute..." % path)
        data = None
        try:
            GraphAnalyze.validate_file_path(path, ".pbtxt")
            with ms_open(path, "rb", MAX_SIZE_LIMITE_NORMAL_FILE) as f:
                data = f.read()
        except OpenException as oe:
            logger.error(f"OpenException occurred: {oe}")
        except FileNotFoundError as fnf:
            logger.error(f"File not found: {fnf}")
        except ValueError as ve:
            logger.error(f"Incorrect file extension: {ve} ")
        except PermissionError as pe:
            logger.error(f"Permission error: {pe}")
        except text_format.ParseError as pe:
            logger.error(f"Parse error: {pe}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        if not data:
            return None
        model = onnx.ModelProto()
        text_format.Parse(data, model)
        logger.info("Load %r success!" % path)
        return model.graph

    @staticmethod
    def extract_sub_graph(args):
        """Extracts a sub-graph from the input graph based on specified criteria."""
        input_path = args.input
        output_path = args.output
        if not output_path:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = GraphAnalyze._append_file_name_suffix(input_path, timestamp)

        logger.info("Begin to read in graph from file %r" % input_path)
        graph_def = GraphAnalyze.load_graph_def_from_pbtxt(input_path)

        gs = GraphAnalyze._build_graph_summary(graph_def)

        center_node, start_node, end_node = args.center_node, args.start_node, args.end_node
        dump_node_names = set()

        if center_node:
            """Either center diffusion mode or start-end mode"""
            dump_node_names.update(GraphAnalyze._find_nodes_by_center_name(args, gs, center_node))
        elif start_node and end_node:
            dump_node_names.update(GraphAnalyze._find_nodes_between_start_and_end(gs, start_node, end_node))

        if not dump_node_names:
            logger.error("The input center_node or (start_node, end_node) is invalid! Please check!")
            return -1
        GraphAnalyze._generate_graph(gs, dump_node_names, output_path, args.without_leaves)
        return 0

    @staticmethod
    def strip(input_path, level=3, output_path=None):
        """Strips Cons/ Data Node and Attribute from an ONNX model file."""
        if level < GraphAnalyze.STRIP_CONST or level > GraphAnalyze.STRIP_ATTR:
            logger.warning(f"Wrong level value = {level}, strip the graph with default value(3).")
            level = GraphAnalyze.STRIP_ATTR
        if not output_path:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = GraphAnalyze._append_file_name_suffix(input_path, timestamp)
        logger.info("Begin to read from %r and write to %r" % (input_path, output_path))

        out = GraphAnalyze._process_file(input_path, level)

        logger.info("save to %r, total node = %d" % (output_path, len(out.node)))

        # Save graph
        model_def = onnx.helper.make_model(out, producer_name='onnx-subgraph')
        with ms_open(output_path, 'w') as f:
            f.write(text_format.MessageToString(model_def))

    @staticmethod
    def print_graph_stat(input_path):
        """Prints statistics about the operations in an ONNX graph."""
        graph_def = GraphAnalyze.load_graph_def_from_pbtxt(input_path)

        op_stat = Counter()
        
        def process_node(node, depth=0):
            if depth > MAX_DEPTH_LIMIT:
                raise RecursionError(
                    f"Exceeded maximum recursion depth {MAX_DEPTH_LIMIT} when process node"
                )
            op_stat[node.op_type] += 1
            for attr in node.attribute:
                if attr.HasField('g'):  # Check if the attribute has a subgraph
                    for subnode in attr.g.node:
                        process_node(subnode, depth=depth + 1)
        
        for node in graph_def.node:
            process_node(node)

        logger.info("Graph stats:")
        for op, count in sorted(op_stat.items(), key=lambda x: x[0]):
            logger.info(f"\t{op} = {count}")

    @staticmethod
    def _append_file_name_suffix(path, suffix):
        """Appends a suffix to the base name of the file in the given path."""
        tokens = path.rsplit('.', 1)  # Split the path by the last occurrence of '.'
        if len(tokens) == 1:
            # If there is no '.', treat the whole path as the base name
            return tokens[0] + '_' + suffix
        else:
            # If there is an extension, split into base name and extension
            base_name, extension = tokens
            return f"{base_name}_{suffix}.{extension}"

    @staticmethod
    def _save_graph_def(graph_def, path, as_text=False):
        """Saves an ONNX graph definition to a file, optionally in text format."""
        model_def = onnx.helper.make_model(graph_def, producer_name='onnx-subgraph')
        if as_text:
            try:
                with ms_open(path, "w") as f:
                    f.write(text_format.MessageToString(model_def))
            except OpenException as oe:
                logger.error(f"OpenException occurred: {oe}")
            except FileNotFoundError as fnf:
                logger.error(f"File not found: {fnf}")
            except PermissionError as pe:
                logger.error(f"Permission error: {pe}")
            except text_format.ParseError as pe:
                logger.error(f"Parse error: {pe}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
        else:
            onnx.save(model_def, path)

    @staticmethod
    def _get_node_name(tensor_name):
        """Extracts the node name from a tensor name."""
        if tensor_name.startswith("^"):
            return tensor_name[1:]
        return tensor_name.split(":")[0]

    @staticmethod
    def _get_node_index_and_name(tensor_name):
        """Extracts the index and name from a tensor name."""
        if tensor_name.startswith("^"):
            return 0, tensor_name[1:]
        if not tensor_name:
            return 0, ""
        name_and_index = tensor_name.rsplit(":", 1)
        if len(name_and_index) != 2:
            logger.error(f"Invalid node input {tensor_name}")
            raise ValueError(f"Invalid tensor name format: {tensor_name}")
        
        name, index = name_and_index
        try:
            return int(index), name
        except ValueError as e:
            logger.error(f"Index part of tensor name {tensor_name} is not an integer")
            raise ValueError(f"Index part of tensor name {tensor_name} is not an integer") from e

    @staticmethod
    def _nodes_to_gs(nodes, gs, start_seq, depth=0):
        """
        Populates a GraphSummary object with information about nodes.
        
        Args:
            nodes (list): List of nodes to process.
            gs (GraphSummary): The GraphSummary object to populate.
            start_seq (int): Starting sequence number for node indexing.
            
        Returns:
            int: The next sequence number after processing all nodes.
        """
        seq = start_seq
        if depth > ITERATIONS:
            raise RuntimeError("The number of cycles has exceeded 100 and the program is terminated.")
        for node in nodes:
            node_name = GraphAnalyze._get_node_name(node.name)
            gs.names_to_node[node_name] = node
            input_indexes_and_names = [GraphAnalyze._get_node_index_and_name(input_name) for input_name in node.input]
            # Separate data inputs and control inputs based on index
            gs.names_to_data_input_names[node_name] = [
                index_and_name[1] 
                for index_and_name in input_indexes_and_names 
                if len(index_and_name) >= 2 and index_and_name[0] >= 0
            ]
            gs.names_to_ctrl_input_names[node_name] = [
                index_and_name[1] 
                for index_and_name in input_indexes_and_names 
                if len(index_and_name) >= 2 and index_and_name[0] < 0
            ]
            gs.names_to_ctrl_input_names[node_name].sort()
            gs.names_to_input_names[node_name] = (
                gs.names_to_data_input_names[node_name] + gs.names_to_ctrl_input_names[node_name]
            )

            if not node.output and gs.names_to_output_names.get(node_name):
                logger.warning(f"The node {node_name} does not have output record, but found input record from others")

            for input_name in node.input:
                gs.names_to_output_names[GraphAnalyze._get_node_name(input_name)].append(node_name)
            gs.names_to_seq_num[node_name] = seq
            seq += 1
            for attr in node.attribute:
                if attr.name == "graph":
                    seq = GraphAnalyze._nodes_to_gs(attr.g.node, gs, seq, depth + 1)

        return seq

    @staticmethod
    def _build_graph_summary(graph_def):
        """Builds a GraphSummary object from a graph definition."""
        gs = GraphSummary()
        seq = 0
        GraphAnalyze._nodes_to_gs(graph_def.node, gs, 0)
        return gs

    @staticmethod
    def _lookup_dump_nodes(args, gs, center_node, lookup_directions):
        """Finds nodes to dump based on specified criteria."""

        dump_tasks = deque()
        stop_name = args.stop_name

        # Add initial tasks based on lookup directions
        if lookup_directions & GraphAnalyze.LOOKUP_FORWARD:
            dump_tasks.append((center_node, 'forward', 0))
        if lookup_directions & GraphAnalyze.LOOKUP_BACKWARD:
            dump_tasks.append((center_node, 'backward', 0))

        dump_node_names = set()

        # Process tasks until none remain
        while dump_tasks:
            node_name, direction, depth = dump_tasks.popleft()
            
            if not node_name or node_name == stop_name:
                continue
            
            node = gs.names_to_node.get(node_name)
            if not node:
                continue

            if depth >= args.layer_number:
                continue

            dump_node_names.add(node_name)

            # Extend tasks based on the direction of lookup
            if direction == 'forward':
                output_node_names = gs.names_to_output_names.get(node_name, [])
                dump_tasks.extend(
                    (output_node_name, direction, depth + 1) for output_node_name in output_node_names
                )
            else:
                input_node_names = gs.names_to_input_names.get(node_name, [])
                dump_tasks.extend(
                    (input_node_name, direction, depth + 1) for input_node_name in input_node_names
                )

        return dump_node_names

    @staticmethod
    def _generate_graph(gs, backbone_names, output_path, without_leaves):
        """
        Generates a sub-graph based on specified backbone nodes.
        
        Args:
            gs (GraphSummary): The GraphSummary object containing graph information.
            backbone_names (list): List of backbone node names to include in the sub-graph.
            output_path (str): Path where the generated sub-graph will be saved.
            without_leaves (bool): If True, do not include leaf nodes in the sub-graph.
            
        Returns:
            None
        """
        dump_names = set()
        for backbone in backbone_names:
            if not backbone:
                continue
            dump_names.add(backbone)
            if not without_leaves:
                # Add input and output nodes of the backbone nodes if including leaves
                for input_name in gs.names_to_input_names.get(backbone, []):
                    dump_names.add(input_name)
                for output_name in gs.names_to_output_names.get(backbone, []):
                    dump_names.add(output_name)

        dump_nodes = []
        for dump_name in dump_names:
            dump_node = gs.names_to_node.get(dump_name, None)
            if dump_node:
                dump_nodes.append(dump_node)
        dump_nodes.sort(key=lambda node: gs.names_to_seq_num[node.name])

        out = onnx.GraphProto()
        for dump_node in dump_nodes:
            out.node.extend([copy.deepcopy(dump_node)])

        logger.info("Save to %r" % output_path)
        logger.info(f"Total nodes = {len(out.node)}")
        GraphAnalyze._save_graph_def(out, output_path, as_text=True)

    @staticmethod
    def _find_nodes_by_center_name(args, gs, center_node):
        """Finds nodes starting from specified names."""
        lookup_directions = GraphAnalyze.LOOKUP_ALL
        if args.only_forward:
            lookup_directions ^= GraphAnalyze.LOOKUP_BACKWARD
        if args.only_backward:
            lookup_directions ^= GraphAnalyze.LOOKUP_FORWARD
        if not lookup_directions:
            logger.error("The --only_forward and --only_backward cannot exist at the same time")
            return set()
        if center_node not in gs.names_to_node:
            logger.error("The node %r can not be found in graph file, please check.", center_node)
            return set()

        logger.info("Begin to find dump nodes, center node %r.", center_node)
        return GraphAnalyze._lookup_dump_nodes(
            args, gs, center_node, lookup_directions
        )

    @staticmethod
    def _find_nodes_between_start_and_end(gs, start_name=None, end_name=None):
        """Finds nodes between specified start and end names."""

        if start_name not in gs.names_to_output_names:
            logger.error("Can not find the node %r's output node." % start_name)
            return set()
        if end_name not in gs.names_to_input_names:
            logger.error("Can not find the node %r's input node." % end_name)
            return set()

        logger.info("Begin to lookup nodes from %r to %r...", start_name, end_name)
        dump_nodes = set()

        nodes_to_end = {start_name}
        que = deque([end_name])
        
        while que:
            node_name = que.popleft()
            if not node_name or node_name in nodes_to_end or node_name == start_name:
                continue
            nodes_to_end.add(node_name)
            que.extend(gs.names_to_input_names.get(node_name, []))

        dump_nodes_this_pattern = set()
        que = deque([start_name])
        
        while que:
            node_name = que.popleft()
            if node_name in dump_nodes_this_pattern or node_name == end_name or node_name not in nodes_to_end:
                continue
            dump_nodes_this_pattern.add(node_name)
            que.extend(gs.names_to_output_names.get(node_name, []))

        dump_nodes_this_pattern.add(end_name)
        dump_nodes.update(dump_nodes_this_pattern)

        return dump_nodes

    @staticmethod
    def _process_file(input_path, level):
        """
        Processes a subgraph by iterating through its nodes, copying nodes that are not of type
        "ge:Const" or "ge:Data", and removes all attributes from the copied nodes.

        Args:
            subgraph (onnx.GraphProto): The subgraph to process.
        """
        graph = GraphAnalyze.load_graph_def_from_pbtxt(input_path)
        logger.info(f"\t total node = {len(graph.node)}")

        out = onnx.GraphProto()
        nodes_copy = []

        def strip_node(node_copy):
            if level == GraphAnalyze.STRIP_ATTR_WITHOUT_SHAPE:
                # Retain only attributes with 'shape' in their name
                shape_attributes = [attr for attr in node_copy.attribute if 'shape' in attr.name]
                node_copy.ClearField('attribute')
                node_copy.attribute.extend(shape_attributes)                      
            elif level == GraphAnalyze.STRIP_ATTR:
                node_copy.ClearField('attribute')  # Remove all attributes
            return node_copy

        def process_node(node):
            if node.op_type not in ("ge:Const", "ge:Data"):
                node_copy = copy.deepcopy(node)
                nodes_copy.append(strip_node(node_copy))

        def process_subgraph(subgraph):
            subgraph_nodes = []
            for subnode in subgraph.node:
                if subnode.op_type not in ("ge:Const", "ge:Data"):
                    subnode_copy = copy.deepcopy(subnode)
                    subgraph_nodes.append(strip_node(subnode_copy))
            subgraph.ClearField('node')
            subgraph.node.extend(subgraph_nodes)

        for node in graph.node:
            if node.op_type == "subgraph":
                node_copy = copy.deepcopy(node)
                for attr in node_copy.attribute:
                    if attr.HasField('g'):  # Check if the attribute has a subgraph
                        process_subgraph(attr.g)
                nodes_copy.append(node_copy)
            else:
                process_node(node)

        nodes_copy.sort(key=lambda node: node.name)
        out.node.extend(nodes_copy)
        return out

