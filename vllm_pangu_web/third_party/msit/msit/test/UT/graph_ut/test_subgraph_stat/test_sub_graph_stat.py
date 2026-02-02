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

from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from msit_graph.subgraph_stat.subgraph_stat import (
    SimpleNode,
    parse_pbtxt,
    bfs_subgraph,
    get_subtree,
    generate_subgraphs,
    find_duplicate_subgraphs,
    has_subgraph,
    calculate_average_durations,
    preprocess_subgraph,
)


# Fixture to mock load_graph_def_from_pbtxt
@pytest.fixture
def mock_load_graph_def():
    with patch('msit_graph.graph_extract.graph_extract.GraphAnalyze.load_graph_def_from_pbtxt') as mock_load:
        yield mock_load


# Test SimpleNode class
def test_simple_node_init_given_name_and_type_when_valid_then_attributes_set():
    node = SimpleNode("Node1", "OpType1")
    assert node.name == "Node1"
    assert node.typename == "OpType1"
    assert node.inputs == []
    assert node.outputs == []


def test_parse_pbtxt_given_empty_graph_def_when_empty_then_empty_nodes():
    graph_def = MagicMock()
    graph_def.node = []
    nodes = parse_pbtxt(graph_def)
    assert nodes == {}


# Test bfs_subgraph function
def test_bfs_subgraph_given_root_name_and_nodes_when_valid_then_return_subgraph_nodes():
    nodes = {
        "Node1": SimpleNode("Node1", "OpType1"),
        "Node2": SimpleNode("Node2", "OpType2"),
        "Node3": SimpleNode("Node3", "OpType3"),
        "Node4": SimpleNode("Node4", "OpType4")
    }
    nodes["Node1"].outputs = ["Node2", "Node3"]
    nodes["Node2"].outputs = ["Node4"]
    subgraph = bfs_subgraph("Node1", nodes, max_nodes=3)
    assert len(subgraph) == 3
    assert "Node1" in subgraph
    assert "Node2" in subgraph
    assert "Node3" in subgraph


def test_bfs_subgraph_given_root_name_and_nodes_when_max_nodes_zero_then_empty_list():
    nodes = {
        "Node1": SimpleNode("Node1", "OpType1")
    }
    subgraph = bfs_subgraph("Node1", nodes, max_nodes=0)
    assert subgraph == []


# Test get_subtree function
def test_get_subtree_given_node_name_and_nodes_when_valid_then_return_subtree_nodes():
    nodes = {
        "Node1": SimpleNode("Node1", "OpType1"),
        "Node2": SimpleNode("Node2", "OpType2"),
        "Node3": SimpleNode("Node3", "OpType3")
    }
    nodes["Node1"].outputs = ["Node2"]
    nodes["Node2"].outputs = ["Node3"]
    subtree = get_subtree(nodes, "Node1")
    assert "Node1" in subtree
    assert "Node2" in subtree
    assert "Node3" in subtree


def test_get_subtree_given_node_name_and_nodes_when_no_outputs_then_single_node():
    nodes = {
        "Node1": SimpleNode("Node1", "OpType1")
    }
    subtree = get_subtree(nodes, "Node1")
    assert "Node1" in subtree
    assert len(subtree) == 1


# Test generate_subgraphs function
def test_generate_subgraphs_given_root_name_nodes_and_bfs_nodes_when_valid_then_unique_subgraphs():
    nodes = {
        "Node1": SimpleNode("Node1", "OpType1"),
        "Node2": SimpleNode("Node2", "OpType2"),
        "Node3": SimpleNode("Node3", "OpType3")
    }
    nodes["Node1"].outputs = ["Node2", "Node3"]
    bfs_nodes = ["Node1", "Node2", "Node3"]
    subgraphs = generate_subgraphs("Node1", nodes, bfs_nodes, 1)
    assert len(subgraphs) >= 1
    assert ("OpType1", "OpType2", "OpType3") in subgraphs


# Test find_duplicate_subgraphs function
def test_find_duplicate_subgraphs_given_graphs_when_duplicates_then_count_duplicates():
    graphs = [
        {
            "Node1": SimpleNode("Node1", "OpType1"),
            "Node2": SimpleNode("Node2", "OpType2"),
            "Node3": SimpleNode("Node3", "OpType3")
        },
        {
            "Node1": SimpleNode("Node1", "OpType1"),
            "Node2": SimpleNode("Node2", "OpType2"),
            "Node3": SimpleNode("Node3", "OpType3")
        }
    ]
    subgraph_count, subgraph_root = find_duplicate_subgraphs(graphs, max_nodes=3)
    assert subgraph_count[("OpType1", "OpType2", "OpType3")] == 0


# Test has_subgraph function
def test_has_subgraph_given_graph_def_when_has_subgraph_then_return_true():
    graph_def = MagicMock()
    graph_def.node = [
        MagicMock(op_type="subgraph"),
        MagicMock(op_type="OpType1")
    ]
    assert has_subgraph(graph_def) is True


def test_has_subgraph_given_graph_def_when_no_subgraph_then_return_false():
    graph_def = MagicMock()
    graph_def.node = [
        MagicMock(op_type="OpType1"),
        MagicMock(op_type="OpType2")
    ]
    assert has_subgraph(graph_def) is False


# Test calculate_average_durations function
def test_calculate_average_durations_given_input2_df_when_valid_then_calculate_mean():
    data = {
        'OP Type': ['Op1', 'Op1', 'Op2'],
        'Task Duration(us)': [100, 200, 150]
    }
    input2_df = pd.DataFrame(data)
    average_durations = calculate_average_durations(input2_df)
    assert average_durations.loc[average_durations['OP Type'] == 'Op1', 'Average Task Duration(us)'].values[0] == 150.0
    assert average_durations.loc[average_durations['OP Type'] == 'Op2', 'Average Task Duration(us)'].values[0] == 150.0


# Test preprocess_subgraph function
def test_preprocess_subgraph_given_subgraph_str_when_has_ge_prefix_then_remove_prefix():
    subgraph_tuple = ('ge:OpType1', 'ge:OpType2')
    cleaned_subgraph = preprocess_subgraph(subgraph_tuple)
    assert cleaned_subgraph == ('OpType1', 'OpType2')


def test_preprocess_subgraph_given_subgraph_str_when_no_ge_prefix_then_return_original():
    subgraph_tuple = ('OpType1', 'OpType2')
    cleaned_subgraph = preprocess_subgraph(subgraph_tuple)
    assert cleaned_subgraph == ('OpType1', 'OpType2')