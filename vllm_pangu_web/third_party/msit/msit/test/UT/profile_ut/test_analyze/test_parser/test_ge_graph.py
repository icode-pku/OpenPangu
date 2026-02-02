import json
import os
import pytest
from msit_prof.analyze.parser.ge_graph import get_all_subgraph


@pytest.fixture
def normal_graph_file():
    """创建包含测试数据的临时JSON文件"""
    graph_data = {
        "graph": [
            {"id": 1, "name": "graph1"},
            {"id": 2, "name": "graph2"}
        ]
    }
    file_path = "./test_graph.json"
    with open(file_path, 'w') as f:
        json.dump(graph_data, f)
    yield file_path
    os.remove(file_path)


@pytest.fixture
def bad_graph_file():
    """创建包含测试数据的临时JSON文件"""
    graph_data = {
        "nodes": [],
        "edges": [],
    }
    file_path = "./test_graph.json"
    with open(file_path, 'w') as f:
        json.dump(graph_data, f)
    yield file_path
    os.remove(file_path)


def test_get_all_subgraph_given_valid_path_when_opened_then_yield_graphs(normal_graph_file):
    """测试正常文件打开且JSON结构正确时的场景"""
    result = list(get_all_subgraph(normal_graph_file))
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["name"] == "graph2"


def test_get_all_subgraph_given_missing_graph_key_when_loaded_then_empty(bad_graph_file):
    """测试JSON缺少graph键时的场景"""
    result = list(get_all_subgraph(bad_graph_file))
    assert len(result) == 0