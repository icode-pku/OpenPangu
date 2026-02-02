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
import os
import shutil
import numpy as np
import pytest
from msquickcmp.accuracy_locat.accuracy_locat import (
    calculate_flow,
    find_npy_files_with_prefix,
    create_bin_file,
    input_completion,
    check_node_valid,
    check_input_node,
    check_res,
)

# Mock Graph and Node classes for testing
class MockGraph:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_next_nodes(self, output_name):
        return [node for node in self.nodes if output_name in node.inputs]

    def get_prev_node(self, input_name):
        for node in self.nodes:
            if input_name in node.outputs:
                return node
        return None


class MockNode:
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs


# Fixture to create and clean up temporary directories
@pytest.fixture(scope="module", autouse=True)
def temp_dir():
    temp_dir_path = "./tmp_test_dir"
    os.makedirs(temp_dir_path, exist_ok=True)
    yield temp_dir_path
    if os.path.exists(temp_dir_path):
        shutil.rmtree(temp_dir_path)


# Test cases for calculate_flow
def test_calculate_flow_given_valid_graph_when_normal_case_then_return_linear_node_list():
    graph = MockGraph(
        [
            MockNode("start", [], ["A"]),
            MockNode("A", ["start"], ["B"]),
            MockNode("B", ["A"], ["end"]),
            MockNode("end", ["B"], []),
        ]
    )
    startnode = graph.nodes[0]
    endnode = graph.nodes[-1]
    result = calculate_flow(graph, startnode, endnode)
    assert len(result) == 2


def test_calculate_flow_given_no_valid_path_when_edge_case_then_return_empty_list():
    graph = MockGraph([MockNode("start", [], []), MockNode("end", [], [])])
    startnode = graph.nodes[0]
    endnode = graph.nodes[1]
    result = calculate_flow(graph, startnode, endnode)
    assert len(result) == 1


# Test cases for find_npy_files_with_prefix
def test_find_npy_files_with_prefix_given_valid_prefix_when_normal_case_then_return_matched_files(temp_dir):
    np.save(os.path.join(temp_dir, "test_prefix_1.npy"), np.array([1, 2, 3]))
    np.save(os.path.join(temp_dir, "test_prefix_2.npy"), np.array([4, 5, 6]))
    result = find_npy_files_with_prefix(temp_dir, "test_prefix")
    assert len(result) == 2
    assert all(file.endswith(".npy") for file in result)


def test_find_npy_files_with_prefix_given_no_matching_files_when_edge_case_then_return_empty_list(temp_dir):
    result = find_npy_files_with_prefix(temp_dir, "non_existent_prefix")
    assert result == []


# Test cases for create_bin_file
def test_create_bin_file_given_valid_npy_files_when_normal_case_then_return_bin_file_paths(temp_dir):
    npy_files = [os.path.join(temp_dir, "test_1.npy"), os.path.join(temp_dir, "test_2.npy")]
    for i, file in enumerate(npy_files):
        np.save(file, np.array([i + 1, i + 2, i + 3]))
    result = create_bin_file(temp_dir, npy_files)
    assert len(result.split(",")) == 2
    assert all(file.endswith(".bin") for file in result.split(","))


def test_create_bin_file_given_empty_list_when_edge_case_then_return_empty_string(temp_dir):
    result = create_bin_file(temp_dir, [])
    assert result == ""


# Test cases for input_completion
def test_input_completion_given_valid_inputs_when_normal_case_then_return_need_list():
    graph = MockGraph([MockNode("A", [], ["B"]), MockNode("B", ["A"], ["C"]), MockNode("C", ["B"], [])])
    inputs_list = [("B",)]
    result = input_completion(graph, inputs_list)
    assert result == ["A.0."]


def test_input_completion_given_empty_inputs_when_edge_case_then_return_empty_list():
    graph = MockGraph([])
    result = input_completion(graph, [])
    assert result == []


# Test cases for check_node_valid
def test_check_node_valid_given_valid_node_when_normal_case_then_return_true():
    graph = MockGraph([MockNode("A", [], ["B"]), MockNode("B", ["A"], [])])
    incnt = {"B": 1}
    node = graph.nodes[1]
    result = check_node_valid(incnt, graph, node)
    assert result == True


def test_check_node_valid_given_invalid_node_when_edge_case_then_return_false():
    graph = MockGraph([MockNode("A", [], ["B"]), MockNode("B", ["A"], [])])
    incnt = {"B": 1}
    node = graph.nodes[0]
    result = check_node_valid(incnt, graph, node)
    assert result == False


# Test cases for check_input_node
def test_check_input_node_given_input_node_when_normal_case_then_return_true():
    graph = MockGraph([MockNode("A", [], ["B"]), MockNode("B", ["A"], [])])
    node = graph.nodes[0]
    result = check_input_node(graph, node)
    assert result == True


def test_check_input_node_given_non_input_node_when_edge_case_then_return_false():
    graph = MockGraph([MockNode("A", [], ["B"]), MockNode("B", ["A"], [])])
    node = graph.nodes[1]
    result = check_input_node(graph, node)
    assert result == True


# Test cases for check_res
def test_check_res_given_valid_result_when_normal_case_then_return_true():
    endnode = MockNode("end", [], [])
    res = [{"GroundTruth": "end"}]
    result = check_res(res, endnode)
    assert result == True


def test_check_res_given_invalid_result_when_edge_case_then_return_false():
    endnode = MockNode("end", [], [])
    res = [{"GroundTruth": "start"}]
    result = check_res(res, endnode)
    assert result == False
